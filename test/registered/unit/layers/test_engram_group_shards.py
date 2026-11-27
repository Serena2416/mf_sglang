"""Engram group_shards config checks and group-store slot claiming. CPU only."""

import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from sglang.srt.arg_groups.engram_config import EngramConfig
from sglang.srt.layers.engram import _check_group_layout, _claim_group_slot
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GROUP = {"replication": "group_shards", "group_size": 4, "group_store": "/s"}


class TestEngramGroupShardsConfig(CustomTestCase):
    def test_group_shards_reads_peer_shards(self):
        for lookup in ("auto", "peer_read"):
            config = EngramConfig.from_dict({**GROUP, "lookup": lookup})
            self.assertEqual(config.lookup, "peer_read")
            self.assertEqual((config.group_size, config.group_store), (4, "/s"))

    def test_rejects_bad_group_shards_configs(self):
        bad = [
            {"replication": "group_shards", "group_store": "/s"},
            {"replication": "group_shards", "group_size": 4},
            {**GROUP, "group_size": "4"},
            {**GROUP, "lookup": "all_reduce"},
            {"group_size": 4},
            {"replication": "per_node", "placement": "cpu", "group_store": "/s"},
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                EngramConfig.from_dict(raw)


class TestEngramPlacementNPU(CustomTestCase):
    """'npu' is the NPU spelling of the 'cpu' host offload; _resolve_engram_config
    turns it into CPU on an NPU server, so only the parsing rules live here."""

    def test_npu_placement_parses_and_tracks_cpu_rules(self):
        config = EngramConfig.from_dict({"placement": "npu"})
        self.assertEqual(config.placement, "npu")
        # The NPU acc_offload pool is always node-shared, so per_node rides along.
        self.assertEqual(
            EngramConfig.from_dict(
                {"placement": "npu", "replication": "per_node"}
            ).replication,
            "per_node",
        )

    def test_npu_placement_rejects_gpu_shard_readers(self):
        # peer_read reads the shards in place and needs placement 'gpu';
        # group_shards resolves to peer_read, so npu rejects both.
        for raw, pattern in (
            ({"placement": "npu", "lookup": "peer_read"}, "peer_read"),
            (
                {
                    "placement": "npu",
                    "replication": "group_shards",
                    "group_size": 4,
                    "group_store": "/s",
                },
                "group_shards",
            ),
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, pattern):
                EngramConfig.from_dict(raw)


class TestEngramGroupStore(CustomTestCase):
    def test_concurrent_claims_get_unique_slots(self):
        world = 4
        with tempfile.TemporaryDirectory() as store:
            with ThreadPoolExecutor(world) as pool:
                slots = list(
                    pool.map(_claim_group_slot, [store] * world, [world] * world)
                )
            self.assertEqual(sorted(slots), list(range(world)))
            with self.assertRaisesRegex(RuntimeError, "slots are taken"):
                _claim_group_slot(store, world)

    def test_rejects_a_store_of_another_deployment(self):
        with tempfile.TemporaryDirectory() as store:
            self.assertEqual(_claim_group_slot(store, 2), 0)
            with self.assertRaisesRegex(RuntimeError, "is not for"):
                _claim_group_slot(store, 4)

    def test_rejects_a_non_empty_store(self):
        with tempfile.TemporaryDirectory() as store:
            open(os.path.join(store, "stale"), "w").close()
            with self.assertRaisesRegex(RuntimeError, "is not empty"):
                _claim_group_slot(store, 2)
            self.assertFalse(os.path.exists(os.path.join(store, "deployment")))

    def test_rejects_a_peer_with_another_table_layout(self):
        mine = dict(num_embeddings=1024, dim=512, mxfp4=False)
        _check_group_layout(mine, dict(mine), 1)
        for peer in ({**mine, "mxfp4": True}, {**mine, "dim": 256}, None):
            with (
                self.subTest(peer=peer),
                self.assertRaisesRegex(RuntimeError, "same Engram table"),
            ):
                _check_group_layout(mine, peer, 1)


if __name__ == "__main__":
    unittest.main()
