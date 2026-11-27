"""NPU Engram offload through MemFabric SHARED DRAM pools.

By default each node holds a complete replica, row-sharded over its schedulers.
A3 pools can also span P/D instances: configured owners contribute DRAM, while
other ranks only map the pool. Each owner packs all same-width table chunks
into one slot. Registration uses a single uniform grid, without table handles.

A5 stages encoded rows through entry_gather and retains scales in HBM. A3 packs
each row's codes and E8M0 scales together in DRAM, then reads and dequantizes
INT8/FP8 into BF16 in one device kernel.
Both paths retain the pool and device metadata throughout graph replay.
Cross-instance loading is synchronized separately from the P/D process groups
before either side can capture or execute a lookup.
"""

from __future__ import annotations

import atexit
import ctypes
import datetime
import logging
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

import torch
from torch import nn

from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import is_ascend_910c

# The dequant helper is shared with the device-resident path on purpose: the
# offloaded lookup must stay bit-identical to it (and to the GPU gather kernel).
from sglang.srt.layers.engram import EngramLayout, _e8m0_to_float
from sglang.srt.mem_cache.pool_host.npu_host_memory import (
    memfabric_host_memory_enabled,
)
from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule
from sglang.srt.utils.device import is_device_stream_capturing

logger = logging.getLogger(__name__)

_GIB = 1 << 30
# One row must fit one 120KB UB ping-pong slot inside the entry_gather AIV
# kernel (OFFLOAD_ENTRY_GATHER_MAX_ENTRY_BYTES in acc_offload.h).
_MAX_ENTRY_BYTES = 120 * 1024
# The share pool holds at most 16 node-local ranks.
_MAX_LOCAL_WORLD = 16
# Heartbeat for the per-forward lookup logs; the first call is always logged.
_OFFLOAD_LOG_EVERY = 1000


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _host_tensor(addr: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A torch view over pool memory at `addr` (same trick as offload.empty)."""
    numel = 1
    for size in shape:
        numel *= size
    itemsize = torch.tensor([], dtype=dtype).element_size()
    buf = (ctypes.c_int8 * (numel * itemsize)).from_address(addr)
    return torch.frombuffer(buf, dtype=dtype).reshape(shape)


def _current_device_id() -> int:
    """The NPU device the pool binds to (a module function so CPU-only tests
    can stub it without touching torch_npu)."""
    return torch.npu.current_device()


@dataclass(frozen=True)
class EngramOffloadTable:
    """One layer's weight table chunk inside this rank's pool slot.

    Row `i` of the layer table lives at
    `pool_base + block_offset + (i // rows) * slot + (i % rows) * entry_bytes`
    where i // rows selects owner_ranks[i // rows]. The slot stride skips
    each owner's unused tail; block_offset selects the layer within a slot.
    """

    layer_hash_index: int
    num_embeddings: int
    head_dim: int
    # host view of this rank's chunk, [rows, entry_bytes] uint8
    pool_weight: torch.Tensor
    # chunk rows per slot (ceil sharding; the last chunk's tail is unused)
    rows: int
    # chunk start inside every slot
    block_offset: int
    # row pitch in bytes (head_dim for arch35 fp8)
    entry_bytes: int


class EngramOffloadManager:
    """Owns the process-wide acc_offload SHARED entity and the engram layout."""

    def __init__(
        self,
        entity,
        device_id: int,
        local_world: int,
        rank: int,
        slot: int,
        tables: list[EngramOffloadTable],
        owner_ranks: list[int],
        load_store=None,
    ):
        self._entity = entity
        self._device_id = device_id
        self.local_world = local_world
        self.rank = rank
        self.slot = slot
        self._tables = tables
        self.owner_ranks = owner_ranks
        self.owner_index = owner_ranks.index(rank) if rank in owner_ranks else None
        self.rows_per_slot = sum(table.rows for table in tables)
        self._owners: Optional[torch.Tensor] = None
        self._load_store = load_store

    def prepare_owners(self, device: torch.device) -> None:
        if self._owners is None and self.owner_ranks != list(
            range(len(self.owner_ranks))
        ):
            self._owners = torch.tensor(
                self.owner_ranks, dtype=torch.int64, device=device
            )

    def encode_ids(self, ids: torch.Tensor, table: EngramOffloadTable) -> torch.Tensor:
        self.prepare_owners(ids.device)
        owner = ids // table.rows
        if self._owners is not None:
            owner = self._owners[owner]
        return (
            owner * self.rows_per_slot
            + table.block_offset // table.entry_bytes
            + ids % table.rows
        )

    def finish_load(self, layer_index: int) -> None:
        if self._load_store is not None:
            self._load_store.set(f"loaded/{layer_index}/{self.rank}", b"1")
            self._load_store.wait(
                [f"loaded/{layer_index}/{rank}" for rank in self.owner_ranks]
            )

    def table(self, layer_hash_index: int) -> EngramOffloadTable:
        return self._tables[layer_hash_index]

    def entry_gather(self, dst, ids: torch.Tensor, count, device: torch.device) -> int:
        """Async gather on the current NPU stream; ordering with the following
        dequant ops comes from the stream itself."""
        return self._entity.entry_gather(dst, ids, count, device)

    def uninitialize(self) -> None:
        logger.info(
            "[engram_offload] rank %d: pool uninitialize (collective teardown)",
            self.rank,
        )
        self._entity.uninitialize()


def _node_local_pool_coords(tp_size: int, tp_rank: int) -> tuple[int, int]:
    """(world, rank) of the scheduler set that shares this node's pool.

    1. SGLANG_NODE_LOCAL_WORLD / SGLANG_NODE_LOCAL_RANK — a launcher-provided
       dense node-local scheduler index, if one exists.
    2. LOCAL_WORLD_SIZE / LOCAL_RANK — torchrun-style launches carry the same
       fact.
    3. Without pipeline parallelism, TP ranks are contiguous per node, so
       derive the local scheduler index from the distributed TP layout.
       Other multi-node layouts require an explicit local index.
    """
    for world_var, rank_var in (
        ("SGLANG_NODE_LOCAL_WORLD", "SGLANG_NODE_LOCAL_RANK"),
        ("LOCAL_WORLD_SIZE", "LOCAL_RANK"),
    ):
        world, rank = os.environ.get(world_var), os.environ.get(rank_var)
        if world is not None and rank is not None:
            return int(world), int(rank)
    parallel = get_parallel()
    if parallel.nnodes == 1:
        return tp_size, tp_rank
    if parallel.pp_size == 1 and tp_size % parallel.nnodes == 0:
        local_world = tp_size // parallel.nnodes
        return local_world, tp_rank % local_world
    raise RuntimeError(
        "engram offload on a multi-node run needs a node-local scheduler "
        "index: set SGLANG_NODE_LOCAL_WORLD / SGLANG_NODE_LOCAL_RANK, or "
        "launch via torchrun so LOCAL_WORLD_SIZE / LOCAL_RANK are present"
    )


def _make_offload_entity(offload_module):
    """The acc_offload entity this process's engram pool lives in.

    Wheels with the multi-instance API (offload.Offload) get a dedicated
    entity, isolated from the HiCache L2 host pool, which initializes the
    module-level default. Old wheels only have the default: fall back to it,
    but refuse when the HiCache memfabric switch is on — one entity's
    scene/reserve/flags cannot serve both consumers.
    """
    if hasattr(offload_module, "Offload"):
        return offload_module.Offload()
    if memfabric_host_memory_enabled():
        raise RuntimeError(
            "engram host offload cannot share the memfabric acc_offload "
            "entity with SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric on this "
            "memfabric_hybrid wheel (no offload.Offload multi-instance API); "
            "upgrade the wheel, or unset one of the two switches"
        )
    logger.warning(
        "[engram_offload] this memfabric_hybrid wheel has no offload.Offload "
        "multi-instance API; using the module-level default entity (upgrade "
        "the wheel to coexist with the HiCache memfabric host pool)"
    )
    return offload_module


def _open_engram_offload(layout: EngramLayout) -> EngramOffloadManager:
    try:
        from memfabric_hybrid import offload
    except ImportError as e:
        raise RuntimeError(
            "the NPU engram host offload needs the memfabric_hybrid "
            "acc_offload module: install the memfabric-hybrid wheel and point "
            f"MEMFABRIC_HYBRID_EXTEND_LIB_PATH at the built kernel library "
            f"(import failed: {e}); drop --engram-config placement 'cpu' (or "
            "unset SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE) to keep "
            "device-resident engram tables"
        ) from e

    parallel = get_parallel()
    # Same-node pipeline stages hold different layers' tables, so they cannot
    # share one slot space (a stage would flush its chunk over another
    # stage's); fail fast while the corruption would otherwise be silent.
    assert max(parallel.pp_size // parallel.nnodes, 1) == 1, (
        "engram offload needs every node-local scheduler in one TP group; "
        f"same-node pipeline parallelism is unsupported (pp_size="
        f"{parallel.pp_size}, nnodes={parallel.nnodes})"
    )
    # One pool per node, shared by every scheduler process on it: dp replicas
    # hold identical weights, so sharding the chunks across all of them keeps
    # a single replica per node. Members are numbered by the node-local index
    # above — tp_rank alone would collide across same-node dp groups in the
    # acc_offload store.
    local_world, rank = _node_local_pool_coords(parallel.tp_size, parallel.tp_rank)
    assert 1 <= local_world <= _MAX_LOCAL_WORLD, (
        f"engram offload pool needs 1..{_MAX_LOCAL_WORLD} node-local ranks, "
        f"got {local_world}"
    )

    world = envs.SGLANG_ENGRAM_OFFLOAD_WORLD_SIZE.get() or local_world
    rank_offset = envs.SGLANG_ENGRAM_OFFLOAD_RANK_OFFSET.get()
    if not 0 <= rank < local_world:
        raise ValueError("Engram node-local rank must be within local_world_size")
    rank += rank_offset
    if (
        world < local_world
        or world % local_world
        or rank_offset % local_world
        or not 0 <= rank < world
    ):
        raise ValueError(
            "Engram pool ranks must be dense, machine-contiguous and within world_size"
        )
    owners = envs.SGLANG_ENGRAM_OFFLOAD_OWNER_RANKS.get()
    owner_ranks = (
        [int(value) for value in owners.split(",")] if owners else list(range(world))
    )
    if (
        not owner_ranks
        or len(set(owner_ranks)) != len(owner_ranks)
        or any(r < 0 or r >= world for r in owner_ranks)
    ):
        raise ValueError(
            "Engram owner ranks must be unique pool ranks within world_size"
        )
    owner_index = owner_ranks.index(rank) if rank in owner_ranks else None
    store_url = envs.SGLANG_ENGRAM_OFFLOAD_STORE_URL.get()
    address = urlsplit(store_url)
    if world != local_world and (
        address.scheme != "tcp"
        or not address.hostname
        or not address.port
        or address.port >= 65535
    ):
        raise ValueError(
            "A cross-instance Engram pool requires an explicit tcp://host:port store URL"
        )

    entity = _make_offload_entity(offload)
    # INT8/FP8 codes use one byte per element; the pool and entry_gather treat
    # entry_bytes as the row pitch and never interpret the row width.
    entry_bytes = layout.head_dim
    if is_ascend_910c():
        entry_bytes += layout.head_dim // FP8_BLOCK_SIZE
    assert 1 <= entry_bytes <= _MAX_ENTRY_BYTES, (
        f"engram row {entry_bytes}B exceeds the {_MAX_ENTRY_BYTES}B entry_gather limit"
    )
    # Per-layer chunk layout in layer_hash_index order, packed back to back
    # from the slot start (row pitch = entry_bytes); every rank runs
    # the same arithmetic, so block offsets inside each slot are identical.
    # Ceil: the last chunk's tail rows stay unused (hash ids stay below
    # num_embeddings, so the slack is never addressed).
    chunks = []
    offset = 0
    for num_embeddings in layout.num_embeddings:
        rows = -(-num_embeddings // len(owner_ranks))
        chunks.append((offset, rows, num_embeddings))
        offset += rows * entry_bytes
    used = offset
    slot = _align_up(used, _GIB)
    logger.info(
        "[engram_offload] pool init: slot=%.2f GiB, tables=%.2f MiB (%.1f%% "
        "used), row=%dB, world=%d rank=%d",
        slot / _GIB,
        used / (1 << 20),
        100.0 * used / slot,
        entry_bytes,
        world,
        rank,
    )

    device_id = _current_device_id()
    config = entity.OffloadConfig()
    config.device_id = device_id
    config.reserve_size = slot
    # Consumers reserve the same virtual stride without allocating host DRAM.
    config.alloc_size = slot if owner_index is not None else 0
    config.world_size = world
    if world != local_world:
        config.local_world_size = local_world
    config.rank_id = rank
    config.scene = entity.Scene.SHARED
    config.flags = 0  # vmm mode: dva == hva == gva, peer slots readable
    if store_url:
        try:
            config.store_url = store_url
        except AttributeError as e:
            raise RuntimeError(
                "SGLANG_ENGRAM_OFFLOAD_STORE_URL needs a memfabric_hybrid "
                "wheel whose OffloadConfig carries store_url"
            ) from e
    ret = entity.initialize(config)
    if ret != 0:
        detail = ""
        err_fn = getattr(entity, "get_last_err_msg", None)
        if err_fn is not None:
            detail = f": {err_fn()}"
        raise RuntimeError(
            f"[engram_offload] offload.initialize failed ({ret}){detail}; on "
            "the SHARED scene every rank of the node must pass the same "
            "config (check SGLANG_ENGRAM_OFFLOAD_STORE_URL)"
        )

    # One whole-slot malloc; the chunks are carved at the fixed offsets above
    # so a peer chunk is one registered slot stride away.
    slot_base = entity.malloc(slot, 0) if owner_index is not None else 0
    if owner_index is not None and slot_base == 0:
        raise RuntimeError(
            f"[engram_offload] offload.malloc({slot}) failed: host DRAM "
            "exhausted for the engram pool"
        )

    ret = entity.register_entry_table(entry_bytes, used // entry_bytes)
    if ret != 0:
        raise RuntimeError(f"[engram_offload] register_entry_table failed ({ret})")
    tables = []
    for layer_hash_index, (block_offset, rows, num_embeddings) in enumerate(chunks):
        logger.info(
            "[engram_offload] rank %d: registered table %d (E=%d "
            "rows_per_owner=%d block_off=%d entry=%dB)",
            rank,
            layer_hash_index,
            num_embeddings,
            rows,
            block_offset,
            entry_bytes,
        )
        tables.append(
            EngramOffloadTable(
                layer_hash_index=layer_hash_index,
                num_embeddings=num_embeddings,
                head_dim=layout.head_dim,
                pool_weight=(
                    _host_tensor(
                        slot_base + block_offset, (rows, entry_bytes), torch.uint8
                    )
                    if owner_index is not None
                    else torch.empty((0, entry_bytes), dtype=torch.uint8, device="cpu")
                ),
                rows=rows,
                block_offset=block_offset,
                entry_bytes=entry_bytes,
            )
        )
    load_store = None
    if world != local_world:
        # P and D have independent process groups. Publish completed CPU writes
        # through a separate store before either side captures a remote read.
        load_store = torch.distributed.TCPStore(
            address.hostname,
            address.port + 1,
            world,
            rank == 0,
            timeout=datetime.timedelta(minutes=30),
            wait_for_workers=False,
        )
        fingerprint = repr((layout, tuple(owner_ranks))).encode()
        load_store.set(f"layout/{rank}", fingerprint)
        keys = [f"layout/{peer}" for peer in range(world)]
        load_store.wait(keys)
        if any(load_store.get(key) != fingerprint for key in keys):
            raise ValueError(
                "Engram pool members disagree on table layout or owner ranks"
            )
    manager = EngramOffloadManager(
        entity, device_id, local_world, rank, slot, tables, owner_ranks, load_store
    )
    # Collective: every rank of the node must leave together.
    atexit.register(manager.uninitialize)
    return manager


_MANAGER: Optional[EngramOffloadManager] = None


def get_engram_offload_manager(layout: EngramLayout) -> EngramOffloadManager:
    """Process-wide singleton; `layout` only matters for the first call, which
    happens in the first NpuEngramOffloadBackend constructor (model build order
    is rank-uniform, so the collective initialize is safe)."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = _open_engram_offload(layout)
    return _MANAGER


class _OffloadForwardBuffers:
    """Fixed-address device buffers for the offloaded lookup path: the NPU
    graph capture only rewrites their contents, never their addresses. The
    staging slice is carved so its data_ptr is entry-aligned, as entry_gather
    requires."""

    def __init__(self, capacity: int, entry_bytes: int, device: torch.device):
        self.capacity = capacity
        self.ids = torch.empty(capacity, dtype=torch.int64, device=device)
        self.count = torch.empty(1, dtype=torch.int32, device=device)
        # entry_gather packs the gathered rows at dst + i * entry_bytes.
        self._backing = torch.empty(
            capacity * entry_bytes + entry_bytes - 1, dtype=torch.uint8, device=device
        )
        self.staging = self._backing[-self._backing.data_ptr() % entry_bytes :]


class NpuEngramOffloadBackend:
    """EngramTableBackend for A3/A5 shared DRAM. Each instance can read the
    complete logical table. One instance per EngramEmbedding; all instances
    share the process-wide pool via get_engram_offload_manager."""

    full_replica = True

    def __init__(
        self, layout: EngramLayout, dim: int, layer_id: int, storage: dict | None = None
    ):
        layer_hash_index = layout.layer_ids.index(layer_id)
        self._layer_hash_index = layer_hash_index
        self._dim = dim
        self._n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self._code_dtype = torch.int8 if storage is not None else torch.float8_e4m3fn
        self._packed = is_ascend_910c()
        self._manager = get_engram_offload_manager(layout)
        self._table = self._manager.table(layer_hash_index)
        # Chunk position follows owner_ranks, which may span P/D instances;
        # consumers keep zero rows and only map the owners' table chunks.
        self.rows = self._table.rows if self._manager.owner_index is not None else 0
        self.row_start = (self._manager.owner_index or 0) * self._table.rows
        self._finalized = False
        self._buffers: Optional[_OffloadForwardBuffers] = None
        self._calls = 0
        self._embed: Optional[nn.Module] = None
        self._scale: Optional[torch.Tensor] = None
        self._mapped_bases: Optional[torch.Tensor] = None
        self._staging_weight: Optional[torch.Tensor] = None
        self._staging_scale: Optional[torch.Tensor] = None

    def install_params(self, embed: nn.Module) -> None:
        """Create the parameter surface on the framework module: the weight is
        meta (load_weights still routes through it; the checkpoint slice lands
        in pinned staging that finish_load flushes into the GVA pool). A3 packs
        the corresponding scale chunks beside the codes. A5 retains the full
        E8M0 scale table in HBM because every rank gathers arbitrary rows."""
        self._embed = embed
        self._device = torch.empty(0).device
        embed.rows = self.rows
        embed.row_start = self.row_start
        embed.weight = nn.Parameter(
            torch.empty(self.rows, self._dim, dtype=self._code_dtype, device="meta"),
            requires_grad=False,
        )
        embed.weight.weight_loader = self._load_weight_rows
        self._staging_weight = torch.empty(
            self.rows,
            self._dim,
            dtype=self._code_dtype,
            device="cpu",
            pin_memory=True,
        )
        embed.scale = nn.Parameter(
            torch.empty(
                self.rows if self._packed else self._table.num_embeddings,
                self._dim // FP8_BLOCK_SIZE,
                dtype=torch.float8_e8m0fnu,
                device="meta" if self._packed else self._device,
            ),
            requires_grad=False,
        )
        embed.scale.weight_loader = self._load_scale_rows
        self._scale = embed.scale
        if self._packed:
            self._staging_scale = torch.empty(
                (self.rows, self._dim // FP8_BLOCK_SIZE),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            query = getattr(self._manager._entity, "get_mapped_pool", None)
            base, stride = query() if query is not None else (0, 0)
            if not base or stride != self._manager.slot:
                raise RuntimeError(
                    "A3 Engram offload requires MemFabric mapped-pool support"
                )
            self._mapped_bases = torch.tensor(
                [
                    base + rank * stride + self._table.block_offset
                    for rank in self._manager.owner_ranks
                ],
                dtype=torch.int64,
                device=self._device,
            )
        logger.info(
            "[engram_offload] rank %d: layer %d OFFLOAD mode engaged "
            "(E=%d chunk_rows=%d entry=%dB block_off=%d; "
            "weight meta + pinned staging, mapped=%s code_dtype=%s)",
            self._manager.rank,
            self._layer_hash_index,
            self._table.num_embeddings,
            self.rows,
            self._table.entry_bytes,
            self._table.block_offset,
            self._packed,
            self._code_dtype,
        )

    def _load_weight_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if loaded_weight.dtype != self._code_dtype:
            raise ValueError(
                f"Engram checkpoint dtype {loaded_weight.dtype} differs from configured {self._code_dtype}"
            )
        rows = loaded_weight[self.row_start : self.row_start + self.rows]
        # The last chunk can be shorter than self.rows (ceil sharding); the
        # unbacked tail is never addressed, but zero it so the pool chunk
        # holds no uninitialized memory. The bytes view keeps this working
        # for the fp8 dtype (in-place zero has no fp8 support).
        self._staging_weight[: rows.shape[0]].copy_(rows)
        if rows.shape[0] < self.rows:
            self._staging_weight[rows.shape[0] :].view(torch.uint8).zero_()

    def _load_scale_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if self._packed:
            rows = loaded_weight[self.row_start : self.row_start + self.rows]
            self._staging_scale[: rows.shape[0]].copy_(rows.view(torch.uint8))
            if rows.shape[0] < self.rows:
                self._staging_scale[rows.shape[0] :].zero_()
        else:
            param.data.copy_(loaded_weight)

    def finish_load(self, label: str) -> None:
        """Flush the pinned staging chunk into the GVA pool and pre-size the
        fixed forward buffers; called once after all checkpoint weights are
        loaded (EngramEmbedding.finish_load -> here)."""
        if self._finalized:
            return
        assert self._embed is not None and self._scale is not None, (
            "install_params must run before finish_load"
        )
        if self._staging_weight is not None:
            self._table.pool_weight[:, : self._dim].copy_(
                self._staging_weight.view(torch.uint8)
            )
            self._staging_weight = None
        if self._staging_scale is not None:
            self._table.pool_weight[:, self._dim :].copy_(self._staging_scale)
            self._staging_scale = None
        self._manager.finish_load(self._layer_hash_index)
        self._finalized = True
        logger.info(
            "[engram_offload] rank %d: %s weight chunk flushed to GVA pool "
            "(%d rows x %dB @ slot off %d)",
            self._manager.rank,
            label,
            self.rows,
            self._table.entry_bytes,
            self._table.block_offset,
        )
        if self._mapped_bases is not None:
            return
        self._manager.prepare_owners(self._scale.device)
        # Size the fixed buffers ONCE for the largest batch any phase can
        # produce: captured graphs bake these addresses into their launches,
        # so a later realloc would leave smaller graphs with dangling pointers.
        try:
            chunked_prefill = get_schedule().chunked_prefill_size or 0
            decode_bs = get_exec().graph.cuda_graph_config.decode.max_bs or 0
        except Exception:
            # Standalone tests without the runtime context: fall back to
            # sizing on first use (still frozen afterwards).
            chunked_prefill, decode_bs = 0, 0
        # Either source may be -1/None (= disabled); never size on it.
        max_tokens = max(chunked_prefill, decode_bs, 0)
        if max_tokens:
            self._buffers = _OffloadForwardBuffers(
                max_tokens * self._n_hash_cols,
                self._table.entry_bytes,
                self._scale.device,
            )
            logger.info(
                "[engram_offload] rank %d: %s forward buffers pre-sized and "
                "frozen (capacity=%d ids for max %d tokens x %d cols)",
                self._manager.rank,
                label,
                self._buffers.capacity,
                max_tokens,
                self._n_hash_cols,
            )

    def _ensure_buffers(
        self, count: int, device: torch.device
    ) -> _OffloadForwardBuffers:
        buffers = self._buffers
        if buffers is None:
            # Preallocation was skipped (no runtime context): size on first
            # use, then freeze.
            self._buffers = _OffloadForwardBuffers(
                count, self._table.entry_bytes, device
            )
            return self._buffers
        if buffers.capacity >= count:
            return buffers
        if is_device_stream_capturing(device):
            raise RuntimeError(
                f"[engram_offload] frozen buffers (capacity {buffers.capacity}) "
                f"cannot serve a capture of {count} ids; preallocation "
                "undersized the graph batch (check chunked_prefill_size and "
                "the decode max batch size)"
            )
        # Oversized eager call (outside capture): a throwaway buffer keeps the
        # frozen one intact — address stability is only required for replays.
        logger.warning(
            "[engram_offload] rank %d: layer %d eager lookup of %d ids "
            "exceeds frozen capacity %d; using a throwaway buffer",
            self._manager.rank,
            self._layer_hash_index,
            count,
            buffers.capacity,
        )
        return _OffloadForwardBuffers(count, self._table.entry_bytes, device)

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        """indices [T, n_hash_cols] -> bf16 [T, n_hash_cols, head_dim]."""
        assert self._finalized, "engram offload rows are not flushed yet"
        if self._mapped_bases is not None:
            from sglang.kernels.ops.embeddings.engram_mapped_gather import (
                mapped_engram_gather,
            )

            return mapped_engram_gather(
                self._mapped_bases,
                indices,
                self._table.rows,
                self._table.num_embeddings,
                self._dim,
                FP8_BLOCK_SIZE,
                self._code_dtype == torch.float8_e4m3fn,
            )
        num_tokens, num_cols = indices.shape
        count = indices.numel()
        if count == 0:
            # Empty batch (e.g. a DP-attention idle dummy forward): return
            # without touching the buffers — sizing on first use at zero would
            # freeze this layer onto a zero-capacity buffer forever.
            return torch.empty(
                num_tokens,
                num_cols,
                self._table.head_dim,
                dtype=torch.bfloat16,
                device=indices.device,
            )
        buffers = self._ensure_buffers(count, indices.device)
        # hash_ids' per-layer slice is strided ([T, L, H] cut on dim 1); copy_
        # into the flat ids buffer needs matching shapes, hence the [T, H]
        # view — the strided source still copies in one d2d pass.
        encoded = self._manager.encode_ids(indices, self._table)
        buffers.ids[:count].view(indices.shape).copy_(encoded)
        buffers.count.fill_(count)
        self._calls += 1
        if self._calls == 1 or self._calls % _OFFLOAD_LOG_EVERY == 0:
            logger.info(
                "[engram_offload] rank %d: layer %d entry_gather call #%d "
                "(%d ids, capacity=%d)",
                self._manager.rank,
                self._layer_hash_index,
                self._calls,
                count,
                buffers.capacity,
            )
        ret = self._manager.entry_gather(
            buffers.staging,
            buffers.ids,
            buffers.count,
            indices.device,
        )
        if ret != 0:
            raise RuntimeError(f"[engram_offload] entry_gather failed with code {ret}")
        staged = buffers.staging[: count * self._table.entry_bytes]
        # Dequantize with the device path's math — the shared _e8m0_to_float
        # keeps the offload lookup bit-identical (arch35 has no fp8 index or
        # e8m0 cast op). Each staged entry is exactly one unpadded fp8 row;
        # the scale table is device-resident in full.
        rows = (
            staged.view(self._code_dtype)
            .reshape(num_tokens, num_cols, self._table.head_dim)
            .float()
            .unflatten(-1, (-1, FP8_BLOCK_SIZE))
        )
        scales = _e8m0_to_float(self._scale.view(torch.uint8)[indices]).unsqueeze(-1)
        return (rows * scales).flatten(-2).to(torch.bfloat16)
