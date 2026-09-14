#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import rsk_lockfile_repair_lineage as lineage
from test_rsk_lockfile_delta import BASE, FEATURE_AFTER, FEATURE_BASE, RSK, metadata


def q(
    base: str,
    head: str,
    post: str,
    cargo_metadata: bytes | None = None,
) -> dict[str, object]:
    return lineage.qualify_repair_lineage(
        base.encode(), head.encode(), post.encode(), cargo_metadata
    )


class LockfileRepairLineageTests(unittest.TestCase):
    def test_generation_mode_qualifies_cargo_candidate(self) -> None:
        report = q(BASE, BASE, BASE + RSK)
        self.assertEqual(report["mode"], "generated-candidate")
        self.assertEqual(report["status"], "accepted")
        self.assertTrue(report["head_equals_base"])
        self.assertFalse(report["post_cargo_equals_head"])
        self.assertEqual(
            report["candidate_delta"]["status"], "rsk-path-records-added"
        )

    def test_generation_mode_accepts_proved_feature_unification_edge(self) -> None:
        report = q(FEATURE_BASE, FEATURE_BASE, FEATURE_AFTER, metadata())
        self.assertEqual(report["mode"], "generated-candidate")
        self.assertEqual(
            report["candidate_delta"]["status"],
            "rsk-path-records-and-proved-feature-edges-added",
        )

    def test_already_repaired_generation_head_is_idempotent(self) -> None:
        repaired = BASE + RSK
        report = q(repaired, repaired, repaired)
        self.assertEqual(report["mode"], "already-qualified-idempotent")
        self.assertTrue(report["post_cargo_equals_head"])

    def test_committed_candidate_requires_exact_post_cargo_bytes(self) -> None:
        repaired = BASE + RSK
        report = q(BASE, repaired, repaired)
        self.assertEqual(report["mode"], "committed-candidate-verified")
        self.assertTrue(report["post_cargo_equals_head"])
        self.assertEqual(
            report["committed_delta"]["status"], "rsk-path-records-added"
        )

    def test_committed_feature_candidate_requires_exact_post_cargo_bytes(self) -> None:
        report = q(FEATURE_BASE, FEATURE_AFTER, FEATURE_AFTER, metadata())
        self.assertEqual(report["mode"], "committed-candidate-verified")
        self.assertEqual(
            report["committed_delta"]["status"],
            "rsk-path-records-and-proved-feature-edges-added",
        )

    def test_committed_candidate_rejects_post_cargo_reordering(self) -> None:
        repaired = BASE + RSK
        reordered = RSK + BASE
        with self.assertRaisesRegex(
            lineage.LockfileLineageError, "not byte-idempotent"
        ):
            q(BASE, repaired, reordered)

    def test_committed_candidate_rejects_unrelated_base_to_head_drift(self) -> None:
        bad_head = (BASE + RSK).replace('version = "1.2.3"', 'version = "1.2.4"')
        with self.assertRaisesRegex(Exception, "non-RSK package graph drift"):
            q(BASE, bad_head, bad_head)

    def test_generation_rejects_unrelated_cargo_drift(self) -> None:
        bad_post = (BASE + RSK).replace('checksum = "d"', 'checksum = "e"')
        with self.assertRaisesRegex(Exception, "non-RSK package graph drift"):
            q(BASE, BASE, bad_post)

    def test_main_discovers_sibling_cargo_metadata_for_feature_edge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "Cargo.lock.base"
            head = root / "Cargo.lock.head"
            post = root / "Cargo.lock.post-cargo"
            metadata_path = root / "cargo-metadata.json"
            report_path = root / "lineage-report.json"
            base.write_text(FEATURE_BASE)
            head.write_text(FEATURE_BASE)
            post.write_text(FEATURE_AFTER)
            metadata_path.write_bytes(metadata())

            rc = lineage.main(
                [
                    str(base),
                    str(head),
                    str(post),
                    "--json-out",
                    str(report_path),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "accepted")
            self.assertEqual(report["mode"], "generated-candidate")
            self.assertEqual(
                report["candidate_delta"]["status"],
                "rsk-path-records-and-proved-feature-edges-added",
            )

    def test_main_writes_structured_rejection_report(self) -> None:
        bad_post = (BASE + RSK).replace('checksum = "d"', 'checksum = "e"')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.lock"
            head = root / "head.lock"
            post = root / "post.lock"
            report_path = root / "lineage-report.json"
            base.write_text(BASE)
            head.write_text(BASE)
            post.write_text(bad_post)

            rc = lineage.main(
                [
                    str(base),
                    str(head),
                    str(post),
                    "--json-out",
                    str(report_path),
                ]
            )
            self.assertEqual(rc, 1)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "rejected")
            self.assertEqual(
                report["reason_code"], "non_rsk_package_record_mutation"
            )
            self.assertIn("base_sha256", report)
            self.assertIn("head_sha256", report)
            self.assertIn("post_cargo_sha256", report)


if __name__ == "__main__":
    unittest.main()
