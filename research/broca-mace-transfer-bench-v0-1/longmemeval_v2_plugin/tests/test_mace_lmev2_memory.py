# Copyright (C) 2026 Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Focused filesystem-ingress tests for the MACE LongMemEval-V2 plugin.

Copy the plugin module into the official benchmark's memory_modules package
before running this file from the benchmark repository.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from memory_modules.mace_lmev2_memory import (
    MAX_SCREENSHOT_BYTES,
    MaceLongMemEvalV2Memory,
)


class MaceLongMemEvalV2IngressTests(unittest.TestCase):
    def make_memory(self, root: Path) -> MaceLongMemEvalV2Memory:
        memory = object.__new__(MaceLongMemEvalV2Memory)
        memory.source_screenshot_root = root / "source"
        memory.source_screenshot_root.mkdir(parents=True, exist_ok=True)
        root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        memory._source_root_fd = os.open(memory.source_screenshot_root, root_flags)
        self.addCleanup(os.close, memory._source_root_fd)
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

    def test_symlink_swap_after_path_resolution_is_rejected_by_openat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            source_file = source_root / "safe.png"
            source_file.write_bytes(b"original bytes")
            outside = root / "outside.png"
            outside.write_bytes(b"outside bytes")
            memory = self.make_memory(root)

            resolved = memory._resolve_source_screenshot("safe.png")
            source_file.unlink()
            source_file.symlink_to(outside)

            with self.assertRaisesRegex(RuntimeError, "without following links"):
                memory._read_source_screenshot(resolved)

    def test_intermediate_directory_symlink_swap_is_rejected_by_openat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_dir = source_root / "screenshots"
            source_dir.mkdir(parents=True)
            source_file = source_dir / "safe.png"
            source_file.write_bytes(b"original bytes")
            outside_dir = root / "outside"
            outside_dir.mkdir()
            (outside_dir / "safe.png").write_bytes(b"outside bytes")
            memory = self.make_memory(root)

            resolved = memory._resolve_source_screenshot("screenshots/safe.png")
            source_dir.rename(source_root / "screenshots-original")
            source_dir.symlink_to(outside_dir, target_is_directory=True)

            with self.assertRaisesRegex(RuntimeError, "without following links"):
                memory._read_source_screenshot(resolved)

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

    def test_unknown_sidecar_outcome_poisons_instance_and_blocks_reuse(self) -> None:
        class FakePipe:
            def __init__(self, response: str = "") -> None:
                self.response = response
                self.written = ""
                self.closed = False

            def write(self, value: str) -> None:
                self.written += value

            def flush(self) -> None:
                return None

            def readline(self) -> str:
                return self.response

        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = FakePipe()
                self.stdout = FakePipe("")
                self.terminated = False
                self.killed = False

            def poll(self) -> int | None:
                return 0 if self.terminated or self.killed else None

            def terminate(self) -> None:
                self.terminated = True

            def kill(self) -> None:
                self.killed = True

            def wait(self, timeout: float | None = None) -> int:
                _ = timeout
                return 0

        memory = object.__new__(MaceLongMemEvalV2Memory)
        memory._lock = threading.RLock()
        memory._process = FakeProcess()
        memory._poisoned = False
        memory._closed = False

        with self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
            memory._rpc({"op": "insert", "trajectory": {"id": "ambiguous"}})
        self.assertTrue(memory._poisoned)
        self.assertTrue(memory._process.terminated)
        with self.assertRaisesRegex(RuntimeError, "until reload/reconciliation"):
            memory._rpc({"op": "query", "query": "must not execute"})

    def test_candidate_evidence_receipt_is_bound_to_run_invocation_query_and_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memory = self.make_memory(root)
            memory.run_id = "run-2026-11-23"
            memory.sidecar_binary_sha256 = "a" * 64
            memory.candidate_evidence_dir = root / "workspace" / "candidate_evidence"
            query = "query"
            query_digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
            candidate = {
                "schema_version": "mace_candidate_evidence_v0_1",
                "record_type": "candidate_evidence",
                "run_id": memory.run_id,
                "candidate_id": "opaque-invocation-1",
                "backend_id": "mace_lmev2_bm25_text",
                "backend_revision_sha256": memory.sidecar_binary_sha256,
                "query_scope": "longmemeval-v2:domain:web",
                "query_purpose": "benchmark:memory-context",
                "query_sha256": query_digest,
                "evidence_id": "lmev2:text:t1:chunk1",
                "source_id": "t1",
                "source_sha256": "b" * 64,
                "payload_sha256": "c" * 64,
                "payload_ref": "lmev2://web/trajectories/t1/chunks/chunk1",
                "modality": "text",
                "retrieval_channel": "trajectory_bm25_text",
                "retrieval_rank": 1,
                "retrieval_score": 0.5,
                "parent_evidence_ids": [],
                "derivation_receipt_sha256": None,
            }
            candidate_jsonl = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
            response = {
                "candidate_evidence_jsonl": candidate_jsonl,
                "candidate_record_count": 1,
                "candidate_set_sha256": "d" * 64,
            }

            memory._persist_candidate_envelope("opaque-invocation-1", query, response)
            files = list(memory.candidate_evidence_dir.iterdir())
            self.assertEqual(len(files), 2)
            receipt_file = next(path for path in files if path.name.endswith(".receipt.json"))
            receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
            self.assertEqual(receipt["run_id"], memory.run_id)
            self.assertEqual(receipt["candidate_id"], "opaque-invocation-1")
            self.assertEqual(receipt["candidate_set_sha256"], "d" * 64)
            self.assertEqual(
                receipt["candidate_jsonl_sha256"],
                hashlib.sha256(candidate_jsonl.encode("utf-8")).hexdigest(),
            )
            memory._verify_persisted_candidate_evidence()

            candidate_path = next(path for path in files if path.suffix == ".jsonl")
            candidate_path.write_text(candidate_jsonl + "\\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "JSONL digest mismatch"):
                memory._verify_persisted_candidate_evidence()

            changed = dict(candidate)
            changed["payload_sha256"] = "e" * 64
            changed_jsonl = json.dumps(changed, sort_keys=True, separators=(",", ":"))
            changed_response = dict(response, candidate_evidence_jsonl=changed_jsonl)
            with self.assertRaisesRegex(RuntimeError, "different evidence"):
                memory._persist_candidate_envelope(
                    "opaque-invocation-1", query, changed_response
                )

    def test_restore_verifier_rejects_orphan_candidate_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memory = self.make_memory(root)
            memory.run_id = "run-orphan"
            memory.sidecar_binary_sha256 = "a" * 64
            memory.candidate_evidence_dir = root / "workspace" / "candidate_evidence"
            memory._persist_candidate_envelope(
                "opaque-invocation",
                "query",
                {
                    "candidate_evidence_jsonl": "",
                    "candidate_record_count": 0,
                    "candidate_set_sha256": None,
                },
            )
            (memory.candidate_evidence_dir / "orphan.jsonl").write_text(
                "orphan", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "orphan or unreferenced"):
                memory._verify_persisted_candidate_evidence()

    def test_zero_candidate_query_still_persists_an_empty_result_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memory = self.make_memory(root)
            memory.run_id = "run-empty"
            memory.sidecar_binary_sha256 = "a" * 64
            memory.candidate_evidence_dir = root / "workspace" / "candidate_evidence"
            memory._persist_candidate_envelope(
                "opaque-empty-invocation",
                "no match",
                {
                    "candidate_evidence_jsonl": "",
                    "candidate_record_count": 0,
                    "candidate_set_sha256": None,
                },
            )
            files = list(memory.candidate_evidence_dir.iterdir())
            self.assertEqual(len(files), 2)
            self.assertEqual(
                len([path for path in files if path.suffix == ".jsonl"]),
                1,
            )
            receipt = json.loads(
                next(path for path in files if path.name.endswith(".receipt.json"))
                .read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["candidate_record_count"], 0)
            self.assertIsNone(receipt["candidate_set_sha256"])

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
