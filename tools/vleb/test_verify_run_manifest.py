#!/usr/bin/env python3
"""Adversarial tests for VLEB manifest integrity and provenance enforcement."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from test_run_manifest import valid_manifest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path(__file__).with_name("verify_run_manifest.py")
SPEC = importlib.util.spec_from_file_location("vleb_verify_run_manifest", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load verify_run_manifest.py")
verifier = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


class ManifestBundle:
    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="vleb-run-manifest-test-")
        self.root = Path(self.tmp.name)
        self.schema = ROOT / "tools/vleb/run-manifest.schema.json"
        self.paths = {
            "task": "task.json",
            "protocol": "protocol.md",
            "patch": "candidate.patch",
            "auditor": "audit_receipt.py",
            "task_schema": "task-contract.schema.json",
            "receipt_schema": "evidence-receipt.schema.json",
            "catalog": "mutations.json",
            "results": "mutation-results.json",
            "bundle": "evidence-bundle.zip",
            "receipt": "receipt.json",
        }
        files = {
            "task": jbytes({"task_id": "vleb-private-task-0001"}),
            "protocol": b"VLEB frozen protocol\n",
            "patch": b"synthetic candidate patch\n",
            "auditor": b"synthetic auditor source\n",
            "task_schema": jbytes({"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}),
            "receipt_schema": jbytes({"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}),
            "catalog": jbytes({"mutations": ["M1", "M2", "M3", "M4"]}),
            "results": jbytes({"killed": 4, "survived": 0, "unresolved": 0}),
            "bundle": b"synthetic evidence bundle bytes\n",
            "receipt": jbytes({"outcome": "pass"}),
        }
        for key, rel in self.paths.items():
            (self.root / rel).write_bytes(files[key])
        h = {key: sha(files[key]) for key in files}

        self.manifest = valid_manifest()
        self.manifest["run_kind"] = "development_smoke"
        self.manifest["task"].update({
            "exposure": "public_development",
            "independent_reviewer_count": 0,
            "baseline_empty_patch_failed": False,
            "reference_solution_passed": False,
            "contract_path": self.paths["task"],
            "contract_sha256": h["task"],
        })
        self.manifest["protocol"] = {"path": self.paths["protocol"], "sha256": h["protocol"]}
        self.manifest["candidate"]["patch_path"] = self.paths["patch"]
        self.manifest["candidate"]["patch_sha256"] = h["patch"]
        self.manifest["auditor"].update({
            "source_path": self.paths["auditor"],
            "source_sha256": h["auditor"],
            "task_schema_path": self.paths["task_schema"],
            "task_schema_sha256": h["task_schema"],
            "receipt_schema_path": self.paths["receipt_schema"],
            "receipt_schema_sha256": h["receipt_schema"],
        })
        self.manifest["mutation_campaign"].update({
            "catalog_path": self.paths["catalog"],
            "catalog_sha256": h["catalog"],
            "results_path": self.paths["results"],
            "results_sha256": h["results"],
        })
        self.manifest["evidence_bundle"].update({
            "path": self.paths["bundle"],
            "sha256": h["bundle"],
            "receipt_path": self.paths["receipt"],
            "receipt_sha256": h["receipt"],
        })
        self.manifest["provenance"].update({
            "attestation_required": False,
            "provider": "none",
            "subject_path": self.paths["bundle"],
        })
        self.manifest_path = self.root / "run-manifest.json"
        self.write_manifest()
        self.trusted_sha = sha(self.manifest_path.read_bytes())

    def write_manifest(self) -> None:
        self.manifest_path.write_bytes(jbytes(self.manifest))

    def audit(self, **kwargs: Any) -> dict[str, Any]:
        self.write_manifest()
        repin_for_test = kwargs.pop("repin_for_test", False)
        expected = kwargs.pop(
            "expected_manifest_sha256",
            sha(self.manifest_path.read_bytes()) if repin_for_test else self.trusted_sha,
        )
        return verifier.validate_manifest(
            self.manifest_path,
            self.schema,
            self.root,
            expected,
            verify_attestation=kwargs.pop("verify_attestation", False),
            **kwargs,
        )

    def close(self) -> None:
        self.tmp.cleanup()


class RunManifestVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = ManifestBundle()

    def tearDown(self) -> None:
        self.bundle.close()

    def assert_issue(self, result: dict[str, Any], code: str) -> None:
        self.assertFalse(result["eligible"], result)
        self.assertIn(code, {row["code"] for row in result["issues"]}, result)

    def test_development_manifest_hashes_and_schema_pass(self) -> None:
        result = self.bundle.audit()
        self.assertTrue(result["eligible"], result["issues"])
        self.assertEqual(result["verdict"], "pass")

    def test_out_of_band_manifest_pin_is_required(self) -> None:
        result = self.bundle.audit(expected_manifest_sha256="0" * 64)
        self.assert_issue(result, "manifest_pin_mismatch")

    def test_referenced_file_digest_is_recomputed(self) -> None:
        (self.bundle.root / self.bundle.paths["patch"]).write_bytes(b"changed after manifest lock")
        self.assert_issue(self.bundle.audit(), "artifact_digest_mismatch")

    def test_path_traversal_is_rejected_by_schema(self) -> None:
        self.bundle.manifest["protocol"]["path"] = "../protocol.md"
        self.assert_issue(self.bundle.audit(repin_for_test=True), "schema_invalid")

    def test_mutation_accounting_inconsistency_is_rejected(self) -> None:
        self.bundle.manifest["mutation_campaign"]["killed"] = 3
        self.assert_issue(self.bundle.audit(repin_for_test=True), "mutation_accounting_inconsistent")

    def test_confirmatory_run_cannot_skip_attestation(self) -> None:
        self.bundle.manifest["run_kind"] = "confirmatory"
        self.bundle.manifest["task"]["exposure"] = "private_confirmatory"
        self.bundle.manifest["task"]["independent_reviewer_count"] = 1
        self.bundle.manifest["task"]["baseline_empty_patch_failed"] = True
        self.bundle.manifest["task"]["reference_solution_passed"] = True
        self.bundle.manifest["provenance"]["attestation_required"] = True
        self.bundle.manifest["provenance"]["provider"] = "github_actions_sigstore"
        result = self.bundle.audit(verify_attestation=False, repin_for_test=True)
        self.assert_issue(result, "attestation_verification_disabled")

    def test_confirmatory_signer_must_match_workflow(self) -> None:
        self.bundle.manifest["run_kind"] = "confirmatory"
        self.bundle.manifest["task"]["exposure"] = "private_confirmatory"
        self.bundle.manifest["task"]["independent_reviewer_count"] = 1
        self.bundle.manifest["task"]["baseline_empty_patch_failed"] = True
        self.bundle.manifest["task"]["reference_solution_passed"] = True
        self.bundle.manifest["provenance"]["attestation_required"] = True
        self.bundle.manifest["provenance"]["provider"] = "github_actions_sigstore"
        self.bundle.manifest["provenance"]["signer_workflow"] = ".github/workflows/untrusted.yml"
        self.assert_issue(self.bundle.audit(repin_for_test=True), "confirmatory_signer_mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
