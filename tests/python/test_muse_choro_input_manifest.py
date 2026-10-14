"""Regression tests for the fail-closed Choro approved-input manifest."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTAKE = ROOT / "crates/domains/symthaea-muse/CHORO_SOURCE_INTAKE_V1.json"
MANIFEST_TEMPLATE = ROOT / "crates/domains/symthaea-muse/CHORO_APPROVED_INPUT_MANIFEST_V1.json"
APPROVAL_TEMPLATE = ROOT / "crates/domains/symthaea-muse/CHORO_APPROVED_INPUT_APPROVAL_V1.json"
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from muse_choro_input_manifest import (  # noqa: E402
    ManifestError,
    evaluate_manifest,
    review_scope_sha256,
)

ARTIFACT_BYTES = b"synthetic fixture bytes only; not score content"


def approved_fixture() -> tuple[dict, dict]:
    digest = hashlib.sha256(ARTIFACT_BYTES).hexdigest()
    manifest = {
        "schema": "muse-choro-approved-input-manifest-v1",
        "manifest_id": "TEST-CHORO-MANIFEST",
        "scope": {
            "purpose": "synthetic validator fixture only",
            "requested_uses": ["empirical_analysis"],
            "commercial_use": True,
            "jurisdictions": ["US", "ZA"],
            "planned_outputs": ["aggregate_metrics"],
            "retention_policy": "delete synthetic fixture after test",
        },
        "required_rights_dimensions": [
            "underlying_composition",
            "score_edition_or_arrangement",
            "scan_or_digital_file",
            "third_party_materials",
        ],
        "sources": [{
            "intake_source_id": "imslp-choro-mineiro-rocha-2022",
            "source_record_url": "https://imslp.org/wiki/Choro_mineiro_%28Rocha%2C_Davi%29",
            "source_file_url": "https://example.org/synthetic-score-artifact.pdf",
            "artifact_path": "synthetic-score-artifact.bin",
            "artifact_sha256": digest,
            "artifact_bytes": len(ARTIFACT_BYTES),
            "retrieved_on": "2026-10-14",
            "selected_work_ids": ["synthetic fixture work only"],
            "selection_basis": "fabricated validator fixture; not a real source selection",
            "rights_review": {
                "decision": "approved",
                "reviewer_id": "independent-rights-reviewer",
                "reviewed_on": "2026-10-14",
                "review_evidence_url": "https://example.org/synthetic-review",
                "license_or_basis_url": "https://creativecommons.org/licenses/by/4.0/",
                "attribution_text": "Synthetic fixture attribution; no source content.",
                "reviewed_artifact_sha256": digest,
                "reviewed_scope_sha256": None,
                "permitted_uses": ["empirical_analysis"],
                "jurisdictions": ["US", "ZA"],
                "commercial_use_allowed": True,
                "rights_dimensions": {
                    name: {
                        "decision": "approved",
                        "rationale": "Synthetic complete test fixture, not a legal finding.",
                        "evidence_url": "https://example.org/synthetic-review/" + name,
                    }
                    for name in (
                        "underlying_composition",
                        "score_edition_or_arrangement",
                        "scan_or_digital_file",
                        "third_party_materials",
                    )
                },
            },
        }],
    }
    manifest["sources"][0]["rights_review"]["reviewed_scope_sha256"] = review_scope_sha256(
        manifest, manifest["sources"][0]
    )
    return manifest, {
        "schema": "muse-choro-input-manifest-approval-v1",
        "status": "approved",
        "manifest_sha256": None,
        "approver_id": "separate-approver",
        "approved_on": "2026-10-14",
        "scope_statement": "Synthetic fixture only; not a real rights approval.",
    }


class ChoroInputManifestTests(unittest.TestCase):
    def test_cli_requires_authorization_for_source_bound_analysis(self) -> None:
        script = ROOT / "scripts/muse_choro_input_manifest.py"
        checked = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        report = json.loads(checked.stdout)
        self.assertFalse(report["authorized"])
        self.assertEqual(report["authorization_status"], "blocked")

        required = subprocess.run(
            [sys.executable, str(script), "--require-authorized"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(required.returncode, 1)
        self.assertIn("BLOCKED: no source-bound analysis authorization", required.stderr)

    def test_repository_template_is_valid_but_blocked(self) -> None:
        report = evaluate_manifest(MANIFEST_TEMPLATE, APPROVAL_TEMPLATE, INTAKE)
        self.assertFalse(report["authorized"])
        self.assertEqual(report["authorization_status"], "blocked")
        self.assertEqual(report["source_count"], 0)
        self.assertIn(
            "No exact source artifacts are in the approved-input manifest.",
            report["blockers"],
        )
        self.assertTrue(any("Commercial/non-commercial purpose is unresolved" in b
                            for b in report["blockers"]))
        self.assertTrue(any("No intended jurisdiction" in b for b in report["blockers"]))
        contract = json.loads(MANIFEST_TEMPLATE.read_text(encoding="utf-8"))["source_entry_contract"]
        self.assertIn("selected_work_ids", contract["required_fields"])
        self.assertIn("reviewed_scope_sha256", contract["rights_review_required_fields"])
        self.assertIn("review_scope_digest", contract)

    def test_explicit_approval_binds_exact_manifest_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "synthetic-score-artifact.bin").write_bytes(ARTIFACT_BYTES)
            manifest, approval = approved_fixture()
            manifest_path = root / "manifest.json"
            approval_path = root / "approval.json"
            intake_path = INTAKE
            manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            manifest_path.write_bytes(manifest_bytes)
            approval["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            approval_path.write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")
            report = evaluate_manifest(manifest_path, approval_path, intake_path, root)
            self.assertTrue(report["authorized"], report["blockers"])
            self.assertEqual(report["source_ids"], ["imslp-choro-mineiro-rocha-2022"])
            self.assertEqual(
                report["artifacts"][0]["artifact_sha256"],
                manifest["sources"][0]["artifact_sha256"],
            )
            self.assertEqual(report["artifacts"][0]["artifact_bytes"], len(ARTIFACT_BYTES))
            self.assertEqual(
                report["artifacts"][0]["artifact_path"],
                "synthetic-score-artifact.bin",
            )
            self.assertEqual(
                report["artifacts"][0]["selected_work_ids"],
                ["synthetic fixture work only"],
            )
            self.assertEqual(
                report["artifacts"][0]["review_scope_sha256"],
                manifest["sources"][0]["rights_review"]["reviewed_scope_sha256"],
            )
            self.assertEqual(
                report["approval_receipt_sha256"],
                hashlib.sha256(approval_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                report["intake_sha256"],
                hashlib.sha256(intake_path.read_bytes()).hexdigest(),
            )

            # Any change to manifest bytes invalidates the detached approval receipt.
            manifest_path.write_bytes(manifest_bytes + b" ")
            report = evaluate_manifest(manifest_path, approval_path, intake_path, root)
            self.assertFalse(report["authorized"])
            self.assertIn(
                "Approval receipt SHA-256 does not match the exact manifest bytes.",
                report["blockers"],
            )

    def test_on_disk_artifact_mismatch_blocks_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "synthetic-score-artifact.bin"
            artifact.write_bytes(ARTIFACT_BYTES)
            manifest, approval = approved_fixture()
            mpath = root / "manifest.json"
            apath = root / "approval.json"
            raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\\n").encode()
            mpath.write_bytes(raw)
            approval["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
            apath.write_text(json.dumps(approval, indent=2) + "\\n", encoding="utf-8")
            artifact.write_bytes(b"mutated synthetic fixture")
            report = evaluate_manifest(mpath, apath, INTAKE, root)
            self.assertFalse(report["authorized"])
            self.assertTrue(any("on-disk artifact SHA-256 does not match manifest" in b
                                for b in report["blockers"]))
            self.assertTrue(any("byte count does not match manifest" in b
                                for b in report["blockers"]))

    def test_unreviewed_artifact_hash_cannot_authorize(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "synthetic-score-artifact.bin").write_bytes(ARTIFACT_BYTES)
            manifest, approval = approved_fixture()
            manifest["sources"][0]["rights_review"]["reviewed_artifact_sha256"] = "b" * 64
            mpath = root / "manifest.json"
            apath = root / "approval.json"
            raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            mpath.write_bytes(raw)
            approval["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
            apath.write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")
            report = evaluate_manifest(mpath, apath, INTAKE, root)
            self.assertFalse(report["authorized"])
            self.assertTrue(any("does not bind the exact artifact SHA-256" in b
                                for b in report["blockers"]))

    def test_unclear_rights_layer_blocks_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "synthetic-score-artifact.bin").write_bytes(ARTIFACT_BYTES)
            manifest, approval = approved_fixture()
            manifest["sources"][0]["rights_review"]["rights_dimensions"]["scan_or_digital_file"]["decision"] = "unclear"
            mpath = root / "manifest.json"
            apath = root / "approval.json"
            raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            mpath.write_bytes(raw)
            approval["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
            apath.write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")
            report = evaluate_manifest(mpath, apath, INTAKE, root)
            self.assertFalse(report["authorized"])
            self.assertTrue(any("scan_or_digital_file is 'unclear'" in b
                                for b in report["blockers"]))

    def test_scope_mismatch_blocks_use(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "synthetic-score-artifact.bin").write_bytes(ARTIFACT_BYTES)
            manifest, approval = approved_fixture()
            manifest["scope"]["requested_uses"] = ["model_training"]
            manifest["scope"]["commercial_use"] = False
            mpath = root / "manifest.json"
            apath = root / "approval.json"
            raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            mpath.write_bytes(raw)
            approval["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
            apath.write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")
            report = evaluate_manifest(mpath, apath, INTAKE, root)
            self.assertFalse(report["authorized"])
            self.assertTrue(any("does not permit requested uses" in b for b in report["blockers"]))

    def test_approver_must_be_independent_of_rights_reviewers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "synthetic-score-artifact.bin").write_bytes(ARTIFACT_BYTES)
            manifest, approval = approved_fixture()
            approval["approver_id"] = "independent-rights-reviewer"
            mpath = root / "manifest.json"
            apath = root / "approval.json"
            raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            mpath.write_bytes(raw)
            approval["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
            apath.write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")
            report = evaluate_manifest(mpath, apath, INTAKE, root)
            self.assertFalse(report["authorized"])
            self.assertIn(
                "The manifest approver must be independent of every source rights reviewer.",
                report["blockers"],
            )

    def test_selected_work_change_invalidates_rights_review_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "synthetic-score-artifact.bin").write_bytes(ARTIFACT_BYTES)
            manifest, approval = approved_fixture()
            manifest["sources"][0]["selected_work_ids"] = ["another synthetic work"]
            mpath = root / "manifest.json"
            apath = root / "approval.json"
            raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            mpath.write_bytes(raw)
            approval["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
            apath.write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")
            report = evaluate_manifest(mpath, apath, INTAKE, root)
            self.assertFalse(report["authorized"])
            self.assertTrue(any(
                "does not bind selected works and the exact requested scope" in blocker
                for blocker in report["blockers"]
            ))

    def test_source_record_url_must_match_source_intake_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "synthetic-score-artifact.bin").write_bytes(ARTIFACT_BYTES)
            manifest, approval = approved_fixture()
            manifest["sources"][0]["source_record_url"] = "https://example.org/unrelated-record"
            mpath = root / "manifest.json"
            apath = root / "approval.json"
            raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\\n").encode()
            mpath.write_bytes(raw)
            approval["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
            apath.write_text(json.dumps(approval, indent=2) + "\\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "does not match the exact intake record"):
                evaluate_manifest(mpath, apath, INTAKE, root)

    def test_review_cannot_predate_artifact_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "synthetic-score-artifact.bin").write_bytes(ARTIFACT_BYTES)
            manifest, approval = approved_fixture()
            manifest["sources"][0]["retrieved_on"] = "2026-10-15"
            manifest["sources"][0]["rights_review"]["reviewed_on"] = "2026-10-14"
            mpath = root / "manifest.json"
            apath = root / "approval.json"
            raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\\n").encode()
            mpath.write_bytes(raw)
            approval["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
            apath.write_text(json.dumps(approval, indent=2) + "\\n", encoding="utf-8")
            report = evaluate_manifest(mpath, apath, INTAKE, root)
            self.assertFalse(report["authorized"])
            self.assertTrue(any("predates retrieval" in blocker for blocker in report["blockers"]))

    def test_unknown_intake_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "synthetic-score-artifact.bin").write_bytes(ARTIFACT_BYTES)
            manifest, approval = approved_fixture()
            manifest["sources"][0]["intake_source_id"] = "not-in-source-intake"
            mpath = root / "manifest.json"
            apath = root / "approval.json"
            raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            mpath.write_bytes(raw)
            approval["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
            apath.write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "absent from CHORO_SOURCE_INTAKE_V1"):
                evaluate_manifest(mpath, apath, INTAKE, root)


if __name__ == "__main__":
    unittest.main()
