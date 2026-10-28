#!/usr/bin/env python3
"""Adversarial tests for VLEB manifest integrity and provenance enforcement."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
import zipfile
from unittest.mock import Mock, patch
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
            "claim_auditor": "audit_legacy_claim.py",
            "claim_schema": "legacy-claim-evidence.schema.json",
            "witness_receipt_schema": "witness-execution-receipt.schema.json",
            "witness_receipt_producer": "record_witness_execution.py",
            "claim_source": "legacy.cbl",
            "claim_verifier": "vleb-cobol-index",
            "claim_witness": "index.json",
            "claim_verdict": "verdict.json",
            "claim_stdout": "index.stdout",
            "claim_stderr": "index.stderr",
            "claim_execution_receipt": "claims/COPY.BANK.REC.receipt.json",
            "claim": "claims/COPY.BANK.REC.json",
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
            "claim_auditor": b"copied from repository below",
            "claim_schema": b"copied from repository below",
            "witness_receipt_schema": b"copied from repository below",
            "witness_receipt_producer": b"copied from repository below",
            "claim_source": b"       COPY BANK-REC.\n",
            "claim_verifier": b"synthetic fixed-format source index verifier\n",
            "claim_witness": b'{"copy_references":["BANK-REC"]}\n',
            "claim_verdict": b"written from fixture below",
            "claim_stdout": b'indexed COPY BANK-REC\n',
            "claim_stderr": b"",
            "claim_execution_receipt": b"written from fixture below",
            "claim": b"written from fixture below",
        }
        for key, rel in self.paths.items():
            if key != "bundle":
                target = self.root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(files[key])
        shutil.copyfile(ROOT / "tools/vleb/audit_legacy_claim.py", self.root / self.paths["claim_auditor"])
        shutil.copyfile(ROOT / "tools/vleb/legacy-claim-evidence.schema.json", self.root / self.paths["claim_schema"])
        shutil.copyfile(ROOT / "tools/vleb/witness-execution-receipt.schema.json", self.root / self.paths["witness_receipt_schema"])
        shutil.copyfile(ROOT / "tools/vleb/record_witness_execution.py", self.root / self.paths["witness_receipt_producer"])
        source_raw = (self.root / self.paths["claim_source"]).read_bytes()
        witness_raw = (self.root / self.paths["claim_witness"]).read_bytes()
        stdout_raw = (self.root / self.paths["claim_stdout"]).read_bytes()
        stderr_raw = (self.root / self.paths["claim_stderr"]).read_bytes()
        witness_producer_sha = sha((self.root / self.paths["witness_receipt_producer"]).read_bytes())
        witness_receipt_schema_sha = sha((self.root / self.paths["witness_receipt_schema"]).read_bytes())
        verifier_sha = sha((self.root / self.paths["claim_verifier"]).read_bytes())
        verdict = {
            "schema_version": "vleb-witness-verdict-v1",
            "task_id": "vleb-private-task-0001",
            "claim_id": "COPY.BANK.REC",
            "source_sha256": sha(source_raw),
            "verifier_sha256": verifier_sha,
            "result": "supports",
        }
        verdict_raw = jbytes(verdict)
        (self.root / self.paths["claim_verdict"]).write_bytes(verdict_raw)
        self.verifier_sha = verifier_sha
        invocation_argv = ["vleb-cobol-index", "--format", "fixed", "legacy.cbl"]
        invocation_sha = sha(json.dumps({
            "argv": invocation_argv,
            "working_directory": ".",
            "timeout_ms": 120000,
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        witness_receipt = {
            "schema_version": "vleb-witness-execution-receipt-v1",
            "task_id": "vleb-private-task-0001",
            "claim_id": "COPY.BANK.REC",
            "source_sha256": sha(source_raw),
            "run_binding": {
                "repository_full_name": "Luminous-Dynamics/symthaea",
                "repository_id": 123456789,
                "run_id": "90001",
                "run_attempt": 1,
                "source_repository_digest": "4" * 40,
                "source_repository_ref": "refs/heads/main",
                "candidate_checkout_revision": "2" * 40,
                "workflow_sha": "1" * 40,
                "workflow_ref": "refs/heads/main",
            },
            "runtime": {
                "immutable_image_id": "sha256:" + "d" * 64,
                "network_mode": "none",
                "network_mode_observed": True,
            },
            "producer": {
                "id": "vleb-record-witness-execution",
                "source_sha256": witness_producer_sha,
            },
            "verifier": {
                "kind": "source_index",
                "id": "test-source-indexer",
                "sha256": verifier_sha,
                "path": self.paths["claim_verifier"],
                "argv_index": 0,
            },
            "invocation": {
                "argv": invocation_argv,
                "argv_sha256": invocation_sha,
                "working_directory": ".",
                "timeout_ms": 120000,
                "environment_sha256": "e" * 64,
            },
            "inputs": [
                {"path": self.paths["claim_source"], "sha256": sha(source_raw)},
                {"path": self.paths["claim_verifier"], "sha256": verifier_sha},
            ],
            "outputs": [
                {"path": self.paths["claim_stdout"], "sha256": sha(stdout_raw), "role": "stdout"},
                {"path": self.paths["claim_stderr"], "sha256": sha(stderr_raw), "role": "stderr"},
                {"path": self.paths["claim_witness"], "sha256": sha(witness_raw), "role": "artifact"},
                {"path": self.paths["claim_verdict"], "sha256": sha(verdict_raw), "role": "verdict"},
            ],
            "input_drift_paths": [],
            "process": {"exit_code": 0, "signal": None, "timed_out": False},
            "observed_result": "supports",
        }
        execution_receipt_raw = jbytes(witness_receipt)
        (self.root / self.paths["claim_execution_receipt"]).write_bytes(execution_receipt_raw)
        claim = {
            "schema_version": "vleb-legacy-claim-v1",
            "task_id": "vleb-private-task-0001",
            "claim_id": "COPY.BANK.REC",
            "claim_type": "copybook_reference",
            "claim_text": "The source references copybook BANK-REC.",
            "claim_scope": "textual_anchor",
            "source": {
                "path": self.paths["claim_source"],
                "sha256": sha(source_raw),
                "spans": [{
                    "line_start": 1,
                    "line_end": 1,
                    "column_start": 8,
                    "column_end": 21,
                    "excerpt_sha256": sha(source_raw.splitlines()[0][7:21]),
                }],
            },
            "verdict": "supported",
            "witnesses": [{
                "witness_id": "index-output-1",
                "kind": "source_index",
                "artifact_path": self.paths["claim_witness"],
                "artifact_sha256": sha(witness_raw),
                "execution_receipt_path": self.paths["claim_execution_receipt"],
                "execution_receipt_sha256": sha(execution_receipt_raw),
                "invocation_sha256": invocation_sha,
                "verifier_id": "test-source-indexer",
                "verifier_sha256": verifier_sha,
                "verifier_path": self.paths["claim_verifier"],
                "verifier_argv_index": 0,
                "verdict_path": self.paths["claim_verdict"],
                "observed_result": "supports",
            }],
            "unresolved_reason": None,
        }
        (self.root / self.paths["claim"]).write_bytes(jbytes(claim))
        self.create_bundle()
        h = {
            key: sha((self.root / rel).read_bytes())
            for key, rel in self.paths.items()
            if key != "bundle"
        }
        h["bundle"] = sha((self.root / self.paths["bundle"]).read_bytes())

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
            "artifact_count": len([key for key in self.paths if key != "bundle"]),
        })
        self.manifest["claim_audit"].update({
            "auditor_path": self.paths["claim_auditor"],
            "auditor_sha256": h["claim_auditor"],
            "schema_path": self.paths["claim_schema"],
            "schema_sha256": h["claim_schema"],
            "witness_receipt_schema_path": self.paths["witness_receipt_schema"],
            "witness_receipt_schema_sha256": h["witness_receipt_schema"],
            "witness_receipt_producer_path": self.paths["witness_receipt_producer"],
            "witness_receipt_producer_sha256": h["witness_receipt_producer"],
            "claims": [{
                "claim_id": "COPY.BANK.REC",
                "claim_path": self.paths["claim"],
                "claim_sha256": h["claim"],
                "source_sha256": h["claim_source"],
            }],
        })
        self.manifest["provenance"].update({
            "attestation_required": False,
            "provider": "none",
            "subject_path": self.paths["bundle"],
        })
        self.manifest_path = self.root / "run-manifest.json"
        self.trusted_auditor_sha256 = self.manifest["auditor"]["source_sha256"]
        self.trusted_verifier_binary_sha256 = self.manifest["auditor"]["verifier_binary_sha256"]
        self.write_manifest()
        self.trusted_sha = sha(self.manifest_path.read_bytes())

    def create_bundle(self) -> None:
        bundle_path = self.root / self.paths["bundle"]
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for key, rel in self.paths.items():
                if key == "bundle":
                    continue
                archive.write(self.root / rel, arcname=rel)

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
            expected_verifier_sha256=kwargs.pop(
                "expected_verifier_sha256", sha(MODULE_PATH.read_bytes())
            ),
            expected_schema_sha256=kwargs.pop(
                "expected_schema_sha256", sha(self.schema.read_bytes())
            ),
            expected_auditor_sha256=kwargs.pop(
                "expected_auditor_sha256", self.trusted_auditor_sha256
            ),
            expected_verifier_binary_sha256=kwargs.pop(
                "expected_verifier_binary_sha256", self.trusted_verifier_binary_sha256
            ),
            verify_attestation=kwargs.pop("verify_attestation", False),
            expected_claim_auditor_sha256=kwargs.pop(
                "expected_claim_auditor_sha256",
                sha((ROOT / "tools/vleb/audit_legacy_claim.py").read_bytes()),
            ),
            expected_claim_schema_sha256=kwargs.pop(
                "expected_claim_schema_sha256",
                sha((ROOT / "tools/vleb/legacy-claim-evidence.schema.json").read_bytes()),
            ),
            expected_witness_receipt_schema_sha256=kwargs.pop(
                "expected_witness_receipt_schema_sha256",
                sha((ROOT / "tools/vleb/witness-execution-receipt.schema.json").read_bytes()),
            ),
            expected_witness_receipt_producer_sha256=kwargs.pop(
                "expected_witness_receipt_producer_sha256",
                sha((ROOT / "tools/vleb/record_witness_execution.py").read_bytes()),
            ),
            trusted_claim_verifier_pins=kwargs.pop(
                "trusted_claim_verifier_pins", {"source_index": self.verifier_sha}
            ),
            **kwargs,
        )

    def close(self) -> None:
        self.tmp.cleanup()


def synthetic_attestation(manifest: dict[str, Any]) -> bytes:
    repo = manifest["repository"]
    candidate = manifest["candidate"]
    bundle = manifest["evidence_bundle"]
    provenance = manifest["provenance"]
    signer_uri = f"https://github.com/{repo['full_name']}/{repo['workflow_path']}@{repo['workflow_ref']}"
    run_uri = (
        f"https://github.com/{repo['full_name']}/actions/runs/"
        f"{repo['run_id']}/attempts/{repo['run_attempt']}"
    )
    return jbytes([{
        "verificationResult": {
            "signature": {
                "certificate": {
                    "subjectAlternativeName": {"type": "URI", "value": signer_uri},
                    "issuer": "https://token.actions.githubusercontent.com",
                    "sourceRepositoryIdentifier": str(repo["repository_id"]),
                    "sourceRepositoryDigest": repo["source_repository_digest"],
                    "sourceRepositoryRef": repo["source_repository_ref"],
                    "githubWorkflowRepository": repo["full_name"],
                    "githubWorkflowRef": repo["workflow_ref"],
                    "githubWorkflowSHA": repo["workflow_sha"],
                    "buildSignerURI": signer_uri,
                    "runInvocationURI": run_uri,
                }
            },
            "statement": {
                "predicateType": provenance["predicate_type"],
                "subject": [{
                    "name": bundle["path"],
                    "digest": {"sha256": bundle["sha256"]},
                }],
            },
        }
    }])


class RunManifestVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = ManifestBundle()

    def tearDown(self) -> None:
        self.bundle.close()

    def assert_issue(self, result: dict[str, Any], code: str) -> None:
        self.assertFalse(result["eligible"], result)
        self.assertIn(code, {row["code"] for row in result["issues"]}, result)

    def audit_one_execution_receipt(self) -> list[dict[str, str]]:
        claim = json.loads((self.bundle.root / self.bundle.paths["claim"]).read_text(encoding="utf-8"))
        witness = claim["witnesses"][0]
        receipt_schema = json.loads(
            (self.bundle.root / self.bundle.paths["witness_receipt_schema"]).read_text(encoding="utf-8")
        )
        return verifier.audit_witness_execution_receipt(
            witness["execution_receipt_path"],
            witness["execution_receipt_sha256"],
            self.bundle.root,
            manifest=self.bundle.manifest,
            claim=claim,
            witness=witness,
            receipt_schema=receipt_schema,
            expected_producer_sha256=sha(
                (ROOT / "tools/vleb/record_witness_execution.py").read_bytes()
            ),
            trusted_claim_verifier_pins={"source_index": self.bundle.verifier_sha},
        )

    def test_execution_receipt_matches_exact_task_run_source_and_runtime(self) -> None:
        self.assertEqual(self.audit_one_execution_receipt(), [])

    def test_execution_receipt_from_another_run_attempt_is_rejected_even_when_rehashed(self) -> None:
        claim_path = self.bundle.root / self.bundle.paths["claim"]
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        witness = claim["witnesses"][0]
        receipt_path = self.bundle.root / witness["execution_receipt_path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["run_binding"]["run_attempt"] = 9
        raw = jbytes(receipt)
        receipt_path.write_bytes(raw)
        witness["execution_receipt_sha256"] = sha(raw)
        claim_path.write_bytes(jbytes(claim))
        issues = self.audit_one_execution_receipt()
        self.assertIn("witness_receipt_run_binding_mismatch", {row["code"] for row in issues})

    def test_execution_receipt_rejects_duplicate_json_keys_even_after_rehash(self) -> None:
        claim_path = self.bundle.root / self.bundle.paths["claim"]
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        witness = claim["witnesses"][0]
        receipt_path = self.bundle.root / witness["execution_receipt_path"]
        raw = receipt_path.read_bytes().rstrip()
        self.assertTrue(raw.endswith(b"}"))
        duplicate_raw = (
            raw[:-1]
            + b', "observed_result":"contradicts"}\\n'
        )
        receipt_path.write_bytes(duplicate_raw)
        witness["execution_receipt_sha256"] = sha(duplicate_raw)
        claim_path.write_bytes(jbytes(claim))

        issues = self.audit_one_execution_receipt()
        self.assertIn(
            "witness_receipt_json_invalid",
            {row["code"] for row in issues},
        )

    def test_execution_receipt_rejects_declared_input_drift(self) -> None:
        claim_path = self.bundle.root / self.bundle.paths["claim"]
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        witness = claim["witnesses"][0]
        receipt_path = self.bundle.root / witness["execution_receipt_path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["input_drift_paths"] = [self.bundle.paths["claim_source"]]
        raw = jbytes(receipt)
        receipt_path.write_bytes(raw)
        witness["execution_receipt_sha256"] = sha(raw)
        claim_path.write_bytes(jbytes(claim))

        issues = self.audit_one_execution_receipt()
        self.assertIn("witness_receipt_input_drift_declared", {row["code"] for row in issues})

    def test_execution_receipt_rejects_mutated_output_bytes(self) -> None:
        (self.bundle.root / self.bundle.paths["claim_witness"]).write_bytes(b'{"copy_references":[]}\\n')
        issues = self.audit_one_execution_receipt()
        codes = {row["code"] for row in issues}
        self.assertIn("witness_receipt_output_digest_mismatch", codes)

    def test_execution_receipt_rejects_verifier_file_tampering(self) -> None:
        verifier_path = self.bundle.root / self.bundle.paths["claim_verifier"]
        verifier_path.write_bytes(b"tampered verifier bytes\\n")
        issues = self.audit_one_execution_receipt()
        codes = {row["code"] for row in issues}
        self.assertIn("witness_verifier_file_digest_mismatch", codes)
        self.assertIn("witness_receipt_input_digest_mismatch", codes)

    def test_execution_receipt_rejects_verdict_tampering(self) -> None:
        verdict_path = self.bundle.root / self.bundle.paths["claim_verdict"]
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        verdict["result"] = "contradicts"
        verdict_path.write_bytes(jbytes(verdict))
        issues = self.audit_one_execution_receipt()
        codes = {row["code"] for row in issues}
        self.assertIn("witness_receipt_output_digest_mismatch", codes)
        self.assertIn("witness_claim_result_mismatch", codes)

    def test_rehashed_receipt_cannot_substitute_an_unapproved_command(self) -> None:
        claim_path = self.bundle.root / self.bundle.paths["claim"]
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        witness = claim["witnesses"][0]
        receipt_path = self.bundle.root / witness["execution_receipt_path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["invocation"]["argv"][0] = "/tmp/unapproved-verifier"
        receipt["invocation"]["argv_sha256"] = verifier.canonical_invocation_digest(
            receipt["invocation"]["argv"],
            receipt["invocation"]["working_directory"],
            receipt["invocation"]["timeout_ms"],
        )
        raw = jbytes(receipt)
        receipt_path.write_bytes(raw)
        witness["execution_receipt_sha256"] = sha(raw)
        claim_path.write_bytes(jbytes(claim))

        issues = self.audit_one_execution_receipt()
        self.assertIn("witness_receipt_invocation_mismatch", {row["code"] for row in issues})

    def test_execution_receipt_cannot_alias_witness_artifact_to_stdout(self) -> None:
        claim_path = self.bundle.root / self.bundle.paths["claim"]
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        witness = claim["witnesses"][0]
        receipt_path = self.bundle.root / witness["execution_receipt_path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for output in receipt["outputs"]:
            if output["path"] == witness["artifact_path"]:
                output["role"] = "stdout"
        raw = jbytes(receipt)
        receipt_path.write_bytes(raw)
        witness["execution_receipt_sha256"] = sha(raw)
        claim_path.write_bytes(jbytes(claim))
        issues = self.audit_one_execution_receipt()
        self.assertIn("witness_receipt_artifact_binding_mismatch", {row["code"] for row in issues})

    def test_execution_receipt_rejects_wrong_external_verifier_pin(self) -> None:
        claim = json.loads((self.bundle.root / self.bundle.paths["claim"]).read_text(encoding="utf-8"))
        witness = claim["witnesses"][0]
        receipt_schema = json.loads(
            (self.bundle.root / self.bundle.paths["witness_receipt_schema"]).read_text(encoding="utf-8")
        )
        issues = verifier.audit_witness_execution_receipt(
            witness["execution_receipt_path"],
            witness["execution_receipt_sha256"],
            self.bundle.root,
            manifest=self.bundle.manifest,
            claim=claim,
            witness=witness,
            receipt_schema=receipt_schema,
            expected_producer_sha256=sha(
                (ROOT / "tools/vleb/record_witness_execution.py").read_bytes()
            ),
            trusted_claim_verifier_pins={"source_index": "e" * 64},
        )
        self.assertIn("witness_receipt_verifier_mismatch", {row["code"] for row in issues})

    def test_claim_audit_passes_only_with_pinned_source_and_witness(self) -> None:
        result = verifier.audit_claim_evidence_bundle(
            self.bundle.manifest,
            self.bundle.root,
            expected_claim_auditor_sha256=sha((ROOT / "tools/vleb/audit_legacy_claim.py").read_bytes()),
            expected_claim_schema_sha256=sha((ROOT / "tools/vleb/legacy-claim-evidence.schema.json").read_bytes()),
            expected_witness_receipt_schema_sha256=sha((ROOT / "tools/vleb/witness-execution-receipt.schema.json").read_bytes()),
            expected_witness_receipt_producer_sha256=sha((ROOT / "tools/vleb/record_witness_execution.py").read_bytes()),
            trusted_claim_verifier_pins={"source_index": self.bundle.verifier_sha},
        )
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["claim_count"], 1)
        self.assertEqual(result["supported"], 1)
        self.assertEqual(result["contradicted"], 0)
        self.assertEqual(result["unresolved"], 0)

    def test_claim_audit_missing_out_of_band_verifier_pin_fails_closed(self) -> None:
        result = verifier.audit_claim_evidence_bundle(
            self.bundle.manifest,
            self.bundle.root,
            expected_claim_auditor_sha256=sha((ROOT / "tools/vleb/audit_legacy_claim.py").read_bytes()),
            expected_claim_schema_sha256=sha((ROOT / "tools/vleb/legacy-claim-evidence.schema.json").read_bytes()),
            expected_witness_receipt_schema_sha256=sha((ROOT / "tools/vleb/witness-execution-receipt.schema.json").read_bytes()),
            expected_witness_receipt_producer_sha256=sha((ROOT / "tools/vleb/record_witness_execution.py").read_bytes()),
            trusted_claim_verifier_pins={},
        )
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["claim_count"], 1)
        self.assertEqual(result["unresolved"], 1)
        self.assertEqual(result["claims"][0]["status"], "unresolved")
        self.assertIn("claim_verifier_pins_missing", {row["code"] for row in result["issues"]})

    def test_claim_audit_rejects_witness_artifact_tampering(self) -> None:
        witness = self.bundle.root / self.bundle.paths["claim_witness"]
        witness.write_bytes(b'{"copy_references":[]}\\n')
        result = verifier.audit_claim_evidence_bundle(
            self.bundle.manifest,
            self.bundle.root,
            expected_claim_auditor_sha256=sha((ROOT / "tools/vleb/audit_legacy_claim.py").read_bytes()),
            expected_claim_schema_sha256=sha((ROOT / "tools/vleb/legacy-claim-evidence.schema.json").read_bytes()),
            expected_witness_receipt_schema_sha256=sha((ROOT / "tools/vleb/witness-execution-receipt.schema.json").read_bytes()),
            expected_witness_receipt_producer_sha256=sha((ROOT / "tools/vleb/record_witness_execution.py").read_bytes()),
            trusted_claim_verifier_pins={"source_index": self.bundle.verifier_sha},
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("claim_evidence_audit_failed", {row["code"] for row in result["issues"]})

    def test_claim_audit_rejects_source_revision_mismatch(self) -> None:
        source = self.bundle.root / self.bundle.paths["claim_source"]
        source.write_bytes(b"       COPY OTHER-REC.\\n")
        result = verifier.audit_claim_evidence_bundle(
            self.bundle.manifest,
            self.bundle.root,
            expected_claim_auditor_sha256=sha((ROOT / "tools/vleb/audit_legacy_claim.py").read_bytes()),
            expected_claim_schema_sha256=sha((ROOT / "tools/vleb/legacy-claim-evidence.schema.json").read_bytes()),
            expected_witness_receipt_schema_sha256=sha((ROOT / "tools/vleb/witness-execution-receipt.schema.json").read_bytes()),
            expected_witness_receipt_producer_sha256=sha((ROOT / "tools/vleb/record_witness_execution.py").read_bytes()),
            trusted_claim_verifier_pins={"source_index": self.bundle.verifier_sha},
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("claim_evidence_audit_failed", {row["code"] for row in result["issues"]})

    def test_claim_audit_malformed_witness_kind_fails_closed_without_crashing(self) -> None:
        claim_path = self.bundle.root / self.bundle.paths["claim"]
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        claim["witnesses"][0]["kind"] = ["source_index"]
        raw = jbytes(claim)
        claim_path.write_bytes(raw)
        self.bundle.manifest["claim_audit"]["claims"][0]["claim_sha256"] = sha(raw)

        result = verifier.audit_claim_evidence_bundle(
            self.bundle.manifest,
            self.bundle.root,
            expected_claim_auditor_sha256=sha((ROOT / "tools/vleb/audit_legacy_claim.py").read_bytes()),
            expected_claim_schema_sha256=sha((ROOT / "tools/vleb/legacy-claim-evidence.schema.json").read_bytes()),
            expected_witness_receipt_schema_sha256=sha((ROOT / "tools/vleb/witness-execution-receipt.schema.json").read_bytes()),
            expected_witness_receipt_producer_sha256=sha((ROOT / "tools/vleb/record_witness_execution.py").read_bytes()),
            trusted_claim_verifier_pins={"source_index": self.bundle.verifier_sha},
        )
        self.assertEqual(result["status"], "failed", result)
        self.assertIn("claim_verifier_pin_missing", {row["code"] for row in result["issues"]})

    def test_claim_audit_rejects_auditor_or_schema_pin_mismatch(self) -> None:
        result = verifier.audit_claim_evidence_bundle(
            self.bundle.manifest,
            self.bundle.root,
            expected_claim_auditor_sha256="0" * 64,
            expected_claim_schema_sha256="0" * 64,
            expected_witness_receipt_schema_sha256=sha((ROOT / "tools/vleb/witness-execution-receipt.schema.json").read_bytes()),
            expected_witness_receipt_producer_sha256=sha((ROOT / "tools/vleb/record_witness_execution.py").read_bytes()),
            trusted_claim_verifier_pins={"source_index": self.bundle.verifier_sha},
        )
        self.assertEqual(result["status"], "failed")
        codes = {row["code"] for row in result["issues"]}
        self.assertIn("claim_auditor_pin_mismatch", codes)
        self.assertIn("claim_schema_pin_mismatch", codes)

    def test_development_manifest_hashes_and_schema_pass(self) -> None:
        result = self.bundle.audit()
        self.assertTrue(result["eligible"], result["issues"])
        self.assertEqual(result["verdict"], "pass")

    def test_out_of_band_manifest_pin_is_required(self) -> None:
        result = self.bundle.audit(expected_manifest_sha256="0" * 64)
        self.assert_issue(result, "manifest_pin_mismatch")

    def test_verifier_implementation_pin_is_required(self) -> None:
        result = self.bundle.audit(expected_verifier_sha256="0" * 64)
        self.assert_issue(result, "verifier_pin_mismatch")

    def test_manifest_schema_pin_is_required(self) -> None:
        result = self.bundle.audit(expected_schema_sha256="0" * 64)
        self.assert_issue(result, "schema_pin_mismatch")

    def test_auditor_source_pin_is_required(self) -> None:
        result = self.bundle.audit(expected_auditor_sha256="0" * 64)
        self.assert_issue(result, "auditor_pin_mismatch")

    def test_receipt_verifier_binary_pin_is_required(self) -> None:
        result = self.bundle.audit(expected_verifier_binary_sha256="0" * 64)
        self.assert_issue(result, "receipt_verifier_binary_pin_mismatch")

    def test_unbundled_loose_file_cannot_override_attested_bytes(self) -> None:
        (self.bundle.root / self.bundle.paths["patch"]).write_bytes(b"changed only outside the sealed ZIP")
        result = self.bundle.audit()
        self.assertTrue(result["eligible"], result["issues"])

    def test_referenced_file_digest_is_recomputed_from_attested_bundle(self) -> None:
        (self.bundle.root / self.bundle.paths["patch"]).write_bytes(b"changed inside rebuilt ZIP")
        self.bundle.create_bundle()
        bundle_path = self.bundle.root / self.bundle.paths["bundle"]
        self.bundle.manifest["evidence_bundle"]["sha256"] = sha(bundle_path.read_bytes())
        self.assert_issue(self.bundle.audit(repin_for_test=True), "artifact_digest_mismatch")

    def test_bundle_path_traversal_is_rejected(self) -> None:
        bundle_path = self.bundle.root / self.bundle.paths["bundle"]
        with zipfile.ZipFile(bundle_path, "w") as archive:
            archive.writestr("../outside.txt", b"must not escape")
        self.bundle.manifest["evidence_bundle"]["sha256"] = sha(bundle_path.read_bytes())
        self.assert_issue(self.bundle.audit(repin_for_test=True), "evidence_bundle_invalid")

    def test_bundle_artifact_count_must_match_zip_inventory(self) -> None:
        self.bundle.manifest["evidence_bundle"]["artifact_count"] = 99
        self.assert_issue(self.bundle.audit(repin_for_test=True), "evidence_bundle_inventory_mismatch")

    def test_path_traversal_is_rejected_by_schema(self) -> None:
        self.bundle.manifest["protocol"]["path"] = "../protocol.md"
        self.assert_issue(self.bundle.audit(repin_for_test=True), "schema_invalid")

    def test_mutation_accounting_inconsistency_is_rejected(self) -> None:
        self.bundle.manifest["mutation_campaign"]["killed"] = 3
        self.assert_issue(self.bundle.audit(repin_for_test=True), "mutation_accounting_inconsistent")

    def test_confirmatory_bundle_is_not_extracted_before_attestation_verifies(self) -> None:
        self.bundle.manifest["run_kind"] = "confirmatory"
        self.bundle.manifest["task"]["exposure"] = "private_confirmatory"
        self.bundle.manifest["task"]["independent_reviewer_count"] = 1
        self.bundle.manifest["task"]["baseline_empty_patch_failed"] = True
        self.bundle.manifest["task"]["reference_solution_passed"] = True
        self.bundle.manifest["provenance"]["attestation_required"] = True
        self.bundle.manifest["provenance"]["provider"] = "github_actions_sigstore"
        bundle_path = self.bundle.root / self.bundle.paths["bundle"]
        bundle_path.write_bytes(b"untrusted non-ZIP bytes")
        self.bundle.manifest["evidence_bundle"]["sha256"] = sha(bundle_path.read_bytes())

        with patch.object(
            verifier.subprocess,
            "run",
            return_value=Mock(returncode=1, stdout=b"signature verification failed"),
        ), patch.object(
            verifier,
            "extract_evidence_bundle",
            side_effect=AssertionError("must not extract before attestation verification"),
        ) as extract:
            result = self.bundle.audit(verify_attestation=True, repin_for_test=True)

        self.assertFalse(result["eligible"], result)
        self.assertIn("attestation_verification_failed", {row["code"] for row in result["issues"]})
        self.assertNotIn("evidence_bundle_invalid", {row["code"] for row in result["issues"]})
        extract.assert_not_called()

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

    def test_confirmatory_self_hosted_runner_is_rejected_by_schema(self) -> None:
        self.bundle.manifest["run_kind"] = "confirmatory"
        self.bundle.manifest["task"]["exposure"] = "private_confirmatory"
        self.bundle.manifest["task"]["independent_reviewer_count"] = 1
        self.bundle.manifest["task"]["baseline_empty_patch_failed"] = True
        self.bundle.manifest["task"]["reference_solution_passed"] = True
        self.bundle.manifest["provenance"]["attestation_required"] = True
        self.bundle.manifest["provenance"]["provider"] = "github_actions_sigstore"
        self.bundle.manifest["runtime"]["runner_environment"] = "self_hosted"
        self.assert_issue(self.bundle.audit(repin_for_test=True), "schema_invalid")

    def test_confirmatory_cli_pins_certificate_and_disallows_self_hosted_runners(self) -> None:
        self.bundle.manifest["run_kind"] = "confirmatory"
        self.bundle.manifest["task"]["exposure"] = "private_confirmatory"
        self.bundle.manifest["task"]["independent_reviewer_count"] = 1
        self.bundle.manifest["task"]["baseline_empty_patch_failed"] = True
        self.bundle.manifest["task"]["reference_solution_passed"] = True
        self.bundle.manifest["provenance"]["attestation_required"] = True
        self.bundle.manifest["provenance"]["provider"] = "github_actions_sigstore"
        attestation = synthetic_attestation(self.bundle.manifest)
        audit_result = jbytes({"eligible": True, "verdict": "pass", "issues": []})

        real_subprocess_run = verifier.subprocess.run

        def fake_run(argv: list[str], **kwargs: Any) -> Mock:
            if argv[0] == "gh":
                return Mock(returncode=0, stdout=attestation)
            if any(str(arg).endswith("audit_receipt.py") for arg in argv):
                return Mock(returncode=0, stdout=audit_result)
            return real_subprocess_run(argv, **kwargs)

        with patch.object(verifier.subprocess, "run", side_effect=fake_run) as mocked_run:
            result = self.bundle.audit(verify_attestation=True, repin_for_test=True)

        self.assertTrue(result["eligible"], result["issues"])
        self.assertTrue(result["score_eligible"], result)
        self.assertEqual(result["independent_claim_audit"]["status"], "passed")
        self.assertEqual(result["independent_claim_audit"]["supported"], 1)
        attestation_argv = mocked_run.call_args_list[0].args[0]
        for flag in (
            "--cert-identity",
            "--cert-oidc-issuer",
            "--deny-self-hosted-runners",
            "--source-digest",
            "--source-ref",
            "--signer-workflow",
            "--predicate-type",
        ):
            self.assertIn(flag, attestation_argv)
        self.assertIn(
            "https://github.com/Luminous-Dynamics/symthaea/.github/workflows/vleb-confirmatory.yml@refs/heads/main",
            attestation_argv,
        )


    def test_confirmatory_score_is_ineligible_without_claim_verifier_pin(self) -> None:
        self.bundle.manifest["run_kind"] = "confirmatory"
        self.bundle.manifest["task"]["exposure"] = "private_confirmatory"
        self.bundle.manifest["task"]["independent_reviewer_count"] = 1
        self.bundle.manifest["task"]["baseline_empty_patch_failed"] = True
        self.bundle.manifest["task"]["reference_solution_passed"] = True
        self.bundle.manifest["provenance"]["attestation_required"] = True
        self.bundle.manifest["provenance"]["provider"] = "github_actions_sigstore"
        attestation = synthetic_attestation(self.bundle.manifest)
        audit_result = jbytes({"eligible": True, "verdict": "pass", "issues": []})

        def fake_run(argv: list[str], **kwargs: Any) -> Mock:
            if argv[0] == "gh":
                return Mock(returncode=0, stdout=attestation)
            if any(str(arg).endswith("audit_receipt.py") for arg in argv):
                return Mock(returncode=0, stdout=audit_result)
            raise AssertionError(f"claim auditor should not run without trusted pins: {argv!r}")

        with patch.object(verifier.subprocess, "run", side_effect=fake_run):
            result = self.bundle.audit(
                verify_attestation=True,
                repin_for_test=True,
                trusted_claim_verifier_pins={},
            )

        self.assertFalse(result["score_eligible"], result)
        self.assertEqual(result["independent_claim_audit"]["status"], "unresolved")
        self.assertIn("claim_verifier_pins_missing", {row["code"] for row in result["issues"]})

    def test_attestation_certificate_and_subject_positive_control(self) -> None:
        raw = synthetic_attestation(self.bundle.manifest)
        self.assertEqual(verifier.attestation_output_issues(raw, self.bundle.manifest), [])

    def test_workflow_source_and_candidate_revision_are_distinct_bindings(self) -> None:
        manifest = self.bundle.manifest
        self.assertNotEqual(manifest["repository"]["workflow_sha"], manifest["repository"]["source_repository_digest"])
        self.assertNotEqual(manifest["repository"]["source_repository_digest"], manifest["candidate"]["checkout_revision"])
        self.assertEqual(verifier.attestation_output_issues(synthetic_attestation(manifest), manifest), [])

    def test_attestation_repository_id_mismatch_is_rejected(self) -> None:
        output = json.loads(synthetic_attestation(self.bundle.manifest))
        output[0]["verificationResult"]["signature"]["certificate"]["sourceRepositoryIdentifier"] = "999999"
        self.assertIn("attestation_repository_id_mismatch", {
            row["code"] for row in verifier.attestation_output_issues(jbytes(output), self.bundle.manifest)
        })

    def test_attestation_source_revision_mismatch_is_rejected(self) -> None:
        output = json.loads(synthetic_attestation(self.bundle.manifest))
        output[0]["verificationResult"]["signature"]["certificate"]["sourceRepositoryDigest"] = "9" * 40
        self.assertIn("attestation_source_digest_mismatch", {
            row["code"] for row in verifier.attestation_output_issues(jbytes(output), self.bundle.manifest)
        })

    def test_attestation_ref_and_workflow_identity_mismatch_are_rejected(self) -> None:
        output = json.loads(synthetic_attestation(self.bundle.manifest))
        cert = output[0]["verificationResult"]["signature"]["certificate"]
        cert["sourceRepositoryRef"] = "refs/heads/untrusted"
        cert["buildSignerURI"] = "https://github.com/untrusted/repo/.github/workflows/other.yml@refs/heads/main"
        cert["subjectAlternativeName"]["value"] = "https://github.com/untrusted/repo/.github/workflows/other.yml@refs/heads/main"
        codes = {row["code"] for row in verifier.attestation_output_issues(jbytes(output), self.bundle.manifest)}
        self.assertIn("attestation_source_ref_mismatch", codes)
        self.assertIn("attestation_signer_uri_mismatch", codes)
        self.assertIn("attestation_certificate_identity_mismatch", codes)

    def test_attestation_run_attempt_mismatch_is_rejected(self) -> None:
        output = json.loads(synthetic_attestation(self.bundle.manifest))
        output[0]["verificationResult"]["signature"]["certificate"]["runInvocationURI"] += "/wrong"
        self.assertIn("attestation_run_id_mismatch", {
            row["code"] for row in verifier.attestation_output_issues(jbytes(output), self.bundle.manifest)
        })

    def test_attestation_subject_digest_mismatch_is_rejected(self) -> None:
        output = json.loads(synthetic_attestation(self.bundle.manifest))
        output[0]["verificationResult"]["statement"]["subject"][0]["digest"]["sha256"] = "0" * 64
        self.assertIn("attestation_subject_digest_mismatch", {
            row["code"] for row in verifier.attestation_output_issues(jbytes(output), self.bundle.manifest)
        })

    def test_attestation_predicate_type_mismatch_is_rejected(self) -> None:
        output = json.loads(synthetic_attestation(self.bundle.manifest))
        output[0]["verificationResult"]["statement"]["predicateType"] = "https://example.invalid/other"
        self.assertIn("attestation_predicate_type_mismatch", {
            row["code"] for row in verifier.attestation_output_issues(jbytes(output), self.bundle.manifest)
        })

    def test_malformed_attestation_output_is_unresolved_not_passed(self) -> None:
        self.assertIn("attestation_json_invalid", {
            row["code"] for row in verifier.attestation_output_issues(b"not-json", self.bundle.manifest)
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
