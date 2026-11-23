# Copyright (C) 2026 Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Focused filesystem-ingress tests for the MACE LongMemEval-V2 plugin.

Copy the plugin module into the official benchmark's memory_modules package
before running this file from the benchmark repository.
"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from memory_modules.mace_lmev2_memory import (
    MAX_SCREENSHOT_BYTES,
    MaceLongMemEvalV2Memory,
)


class MaceLongMemEvalV2IngressTests(unittest.TestCase):
    def make_memory(self, root: Path) -> MaceLongMemEvalV2Memory:
        memory = object.__new__(MaceLongMemEvalV2Memory)
        memory.source_screenshot_root = root / "source"
        memory.asset_root = root / "workspace" / "assets"
        memory.asset_root.mkdir(parents=True)
        memory.domain = "web"
        return memory

    def trajectory(self, screenshot: str) -> dict[str, object]:
        return {
            "id": "trajectory-1",
            "domain": "web",
            "goal": "Create an order",
            "start_url": "https://example.invalid/",
            "outcome": "success",
            "metadata": {"must_preserve": True},
            "states": [
                {
                    "url": "https://example.invalid/orders",
                    "action": None,
                    "thought": "Open order page",
                    "accessibility_tree": "Orders. New order button.",
                    "screenshot": screenshot,
                    "state_index": 7,
                    "step": 0,
                }
            ],
        }

    def test_ingress_copies_image_binds_digest_and_preserves_canonical_original(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_file = source_root / "screenshots" / "state.png"
            source_file.parent.mkdir(parents=True)
            source_bytes = b"fixture screenshot bytes"
            source_file.write_bytes(source_bytes)
            memory = self.make_memory(root)
            original = self.trajectory("screenshots/state.png")

            trajectory_id, trajectory_key, serialized = memory._prepare_trajectory(original)
            normalized = json.loads(serialized)
            state = normalized["states"][0]
            copied = memory.asset_root / state["screenshot"]

            self.assertEqual(trajectory_id, "trajectory-1")
            self.assertTrue(copied.is_file())
            self.assertEqual(copied.read_bytes(), source_bytes)
            self.assertEqual(
                state["mace_screenshot_sha256"],
                hashlib.sha256(source_bytes).hexdigest(),
            )
            self.assertEqual(state["screenshot"], f"{trajectory_key}/state_000000.png")
            self.assertEqual(
                json.loads(normalized["mace_original_trajectory_json"]),
                original,
            )
            self.assertEqual(
                normalized["mace_original_trajectory_sha256"],
                hashlib.sha256(
                    json.dumps(
                        original,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            )

    def test_absolute_screenshot_outside_source_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            outside = root / "outside.png"
            outside.write_bytes(b"outside")
            memory = self.make_memory(root)

            with self.assertRaises(RuntimeError):
                memory._resolve_source_screenshot(str(outside))

    def test_parent_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            (root / "outside.png").write_bytes(b"outside")
            memory = self.make_memory(root)

            with self.assertRaises(RuntimeError):
                memory._resolve_source_screenshot("../outside.png")

    def test_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            outside = root / "outside.png"
            outside.write_bytes(b"outside")
            link = source_root / "escape.png"
            link.symlink_to(outside)
            memory = self.make_memory(root)

            with self.assertRaises(RuntimeError):
                memory._resolve_source_screenshot("escape.png")

    def test_oversized_screenshot_is_rejected_before_reading_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_file = root / "source" / "large.png"
            source_file.parent.mkdir(parents=True)
            with source_file.open("wb") as handle:
                handle.truncate(MAX_SCREENSHOT_BYTES + 1)
            memory = self.make_memory(root)

            with self.assertRaises(RuntimeError):
                memory._prepare_trajectory(self.trajectory("large.png"))

    def test_domain_mismatch_is_rejected_before_asset_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memory = self.make_memory(root)
            wrong_domain = self.trajectory("unused.png")
            wrong_domain["domain"] = "enterprise"

            with self.assertRaises(RuntimeError):
                memory._prepare_trajectory(wrong_domain)
            self.assertEqual(list(memory.asset_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
