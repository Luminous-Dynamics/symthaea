#!/usr/bin/env python3
"""Integration tests for runtime claim scoring through evaluator execution receipts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


manifest_verifier = load_module("vleb_runtime_claim_manifest_verifier", ROOT / "tools/vleb/verify_run_manifest.py")
execution_runner = load_module("vleb_runtime_claim_execution_runner", ROOT / "tools/vleb/run_execution_witness.py")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


class RuntimeClaimFixture:
    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="vleb-runtime-claim-gate-")
        self.root = Path(self.tmp.name)
        self.source_path = self.root / "legacy.cbl"
        self.source_raw = b"       COPY BANK-REC.\n"
        self.source_path.write_bytes(self.source_raw)
        self.source_sha = sha(self.source_raw)

        self.paths = {
            "claim_auditor": "tools/vleb/audit_legacy_claim.py",
            "claim_schema": "tools/vleb/legacy-claim-evidence.schema.json",
            "witness_runner": "tools/vleb/run_execution_witness.py",
            "witness_auditor": "tools/vleb/audit_execution_witness.py",
            "witness_receipt_schema": "tools/vleb/execution-witness-receipt.schema.json",
            "witness_plan_schema": "tools/vleb/execution-witness-plan.schema.json",
            "plan": "plans/transaction-check.json",
            "receipt": "receipts/transaction-check.json",
            "claim": "claims/TX.ACCOUNT.BALANCE.json",
        }
        for key in ("claim_auditor", "claim_schema", "witness_runner", "witness_auditor",
                    "witness_receipt_schema", "witness_plan_schema", "plan", "receipt", "claim"):
            (self.root / self.paths[key]).parent.mkdir(parents=True, exist_ok=True)
        for key, source in (
            ("claim_auditor", ROOT / "tools/vleb/audit_legacy_claim.py"),
            ("claim_schema", ROOT / "tools/vleb/legacy-claim-evidence.schema.json"),
            ("witness_runner", ROOT / "tools/vleb/run_execution_witness.py"),
            ("witness_auditor", ROOT / "tools/vleb/audit_execution_witness.py"),
            ("witness_receipt_schema", ROOT / "tools/vleb/execution-witness-receipt.schema.json"),
            ("witness_plan_schema", ROOT / "tools/vleb/execution-witness-plan.schema.json"),
        ):
            shutil.copyfile(source, self.root / self.paths[key])

        executable = str(Path(sys.executable).resolve(strict=True))
        executable_sha = sha(Path(executable).read_bytes())
        self.identity = {
            "task_id": "vleb-runtime-task-0001",
            "claim_id": "TX.ACCOUNT.BALANCE",
            "repository": {"full_name": "Luminous-Dynamics/symthaea", "id": 123456789},
            "run": {"run_id": "90001001", "attempt": 3},
            "candidate_revision": "c" * 40,
            "source_sha256": self.source_sha,
            "runtime": {
                "platform": sys.platform,
                "immutable_image_id": "sha256:" + "d" * 64,
                "network_mode": "none",
            },
        }
        plan = {
            "schema_version": "vleb-execution-witness-plan-v1",
            **self.identity,
            "verifier": {
                "id": "synthetic-runtime-check",
                "executable_path": executable,
                "executable_sha256": executable_sha,
            },
            "invocation": {
                "argv": [
                    executable,
                    "-c",
                    "import pathlib; assert 'COPY BANK-REC.' in pathlib.Path('legacy.cbl').read_text(); print('fixture invariant observed')",
                ],
                "working_directory": ".",
                "environment": {"PYTHONIOENCODING": "utf-8"},
                "timeout_seconds": 10,
                "expected_exit_code": 0,
            },
            "source_path": "legacy.cbl",
            "inputs": [{"path": "legacy.cbl", "sha256": self.source_sha}],
            "outputs": {"stdout_path": "outputs/transaction.stdout", "stderr_path": "outputs/transaction.stderr"},
        }
        self.plan_raw = jbytes(plan)
        (self.root / self.paths["plan"]).write_bytes(self.plan_raw)
        self.plan_sha = sha(self.plan_raw)
        self.runner_sha = sha(Path(execution_runner.__file__).read_bytes())
        self.receipt = execution_runner.record_execution(
            plan_path=Path(self.paths["plan"]),
            evidence_root=self.root,
            receipt_path=self.paths["receipt"],
            expected_plan_sha256=self.plan_sha,
            expected_runner_sha256=self.runner_sha,
            expected_verifier_sha256=executable_sha,
            trusted_identity=self.identity,
        )
        receipt_sha = self.receipt["receipt_sha256"]
        claim = {
            "schema_version": "vleb-legacy-claim-v1",
            "task_id": self.identity["task_id"],
            "claim_id": self.identity["claim_id"],
            "claim_type": "transaction_effect",
            "claim_text": "The evaluator's independent execution witness completed the pinned transaction-invariant check.",
            "claim_scope": "runtime_behavior",
            "source": {
                "path": "legacy.cbl",
                "sha256": self.source_sha,
                "spans": [{
                    "line_start": 1,
                    "line_end": 1,
                    "column_start": 8,
                    "column_end": 21,
                    "excerpt_sha256": sha(self.source_raw.splitlines()[0][7:21]),
                }],
            },
            "verdict": "supported",
            "witnesses": [{
                "witness_id": "runtime-check-1",
                "kind": "independent_execution",
                "artifact_path": self.paths["receipt"],
                "artifact_sha256": receipt_sha,
                "verifier_id": "synthetic-runtime-check",
                "verifier_sha256": executable_sha,
                "observed_result": "supports",
            }],
            "unresolved_reason": None,
        }
        self.claim_raw = jbytes(claim)
        (self.root / self.paths["claim"]).write_bytes(self.claim_raw)
        self.claim_sha = sha(self.claim_raw)

        self.claim_auditor_sha = sha((self.root / self.paths["claim_auditor"]).read_bytes())
        self.claim_schema_sha = sha((self.root / self.paths["claim_schema"]).read_bytes())
        self.witness_auditor_sha = sha((self.root / self.paths["witness_auditor"]).read_bytes())
        self.witness_receipt_schema_sha = sha((self.root / self.paths["witness_receipt_schema"]).read_bytes())
        self.witness_plan_schema_sha = sha((self.root / self.paths["witness_plan_schema"]).read_bytes())

        self.manifest = {
            "task": {"task_id": self.identity["task_id"]},
            "repository": {
                "full_name": self.identity["repository"]["full_name"],
                "repository_id": self.identity["repository"]["id"],
                "run_id": self.identity["run"]["run_id"],
                "run_attempt": self.identity["run"]["attempt"],
            },
            "candidate": {"checkout_revision": self.identity["candidate_revision"]},
            "runtime": self.identity["runtime"],
            "claim_audit": {
                "auditor_path": self.paths["claim_auditor"],
                "auditor_sha256": self.claim_auditor_sha,
                "schema_path": self.paths["claim_schema"],
                "schema_sha256": self.claim_schema_sha,
                "execution_witness_toolchain": {
                    "runner_path": self.paths["witness_runner"],
                    "runner_sha256": self.runner_sha,
                    "auditor_path": self.paths["witness_auditor"],
                    "auditor_sha256": self.witness_auditor_sha,
                    "receipt_schema_path": self.paths["witness_receipt_schema"],
                    "receipt_schema_sha256": self.witness_receipt_schema_sha,
                    "plan_schema_path": self.paths["witness_plan_schema"],
                    "plan_schema_sha256": self.witness_plan_schema_sha,
                },
                "claims": [{
                    "claim_id": self.identity["claim_id"],
                    "claim_path": self.paths["claim"],
                    "claim_sha256": self.claim_sha,
                    "source_sha256": self.source_sha,
                    "execution_witness": {
                        "receipt_path": self.paths["receipt"],
                        "receipt_sha256": receipt_sha,
                        "plan_path": self.paths["plan"],
                        "plan_sha256": self.plan_sha,
                    },
                }],
            },
        }

    def audit(self, **overrides: Any) -> dict[str, Any]:
        args = {
            "expected_claim_auditor_sha256": self.claim_auditor_sha,
            "expected_claim_schema_sha256": self.claim_schema_sha,
            "trusted_claim_verifier_pins": {
                "independent_execution": self.identity and self.receipt["receipt"]["verifier"]["executable_sha256"]
            },
            "expected_witness_auditor_sha256": self.witness_auditor_sha,
            "expected_witness_receipt_schema_sha256": self.witness_receipt_schema_sha,
            "expected_witness_plan_schema_sha256": self.witness_plan_schema_sha,
            "expected_witness_runner_sha256": self.runner_sha,
        }
        args.update(overrides)
        return manifest_verifier.audit_claim_evidence_bundle(self.manifest, self.root, **args)

    def close(self) -> None:
        self.tmp.cleanup()


class RuntimeClaimGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = RuntimeClaimFixture()

    def tearDown(self) -> None:
        self.fx.close()

    def test_runtime_claim_requires_audited_execution_receipt(self) -> None:
        result = self.fx.audit()
        self.assertEqual(result["status"], "passed", result["issues"])
        self.assertEqual(result["runtime_claims"], 1)
        self.assertEqual(result["execution_witnesses_audited"], 1)
        self.assertEqual(result["supported"], 1)
        self.assertEqual(result["contradicted"], 0)
        self.assertEqual(result["unresolved"], 0)

    def test_runtime_receipt_cannot_be_replayed_into_another_attempt(self) -> None:
        self.fx.manifest["repository"]["run_attempt"] += 1
        result = self.fx.audit()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["execution_witnesses_audited"], 0)
        self.assertEqual(result["unresolved"], 1)

    def test_runtime_claim_without_receipt_is_ineligible(self) -> None:
        del self.fx.manifest["claim_audit"]["claims"][0]["execution_witness"]
        result = self.fx.audit()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["runtime_claims"], 1)
        self.assertEqual(result["execution_witnesses_audited"], 0)
        self.assertEqual(result["unresolved"], 1)

    def test_contradicted_claim_cannot_leave_gate_passed(self) -> None:
        claim_path = self.fx.root / self.fx.paths["claim"]
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        claim["verdict"] = "contradicted"
        claim["witnesses"][0]["observed_result"] = "contradicts"
        raw = jbytes(claim)
        claim_path.write_bytes(raw)
        self.fx.manifest["claim_audit"]["claims"][0]["claim_sha256"] = sha(raw)
        self.fx.manifest["claim_audit"]["claims"][0]["execution_witness"]["receipt_sha256"] = self.fx.receipt["receipt_sha256"]
        receipt = json.loads((self.fx.root / self.fx.paths["receipt"]).read_text(encoding="utf-8"))
        receipt["producer_sha256"] = self.fx.runner_sha
        (self.fx.root / self.fx.paths["receipt"]).write_bytes(jbytes(receipt))
        self.fx.manifest["claim_audit"]["claims"][0]["execution_witness"]["receipt_sha256"] = sha((self.fx.root / self.fx.paths["receipt"]).read_bytes())
        claim["witnesses"][0]["artifact_sha256"] = self.fx.manifest["claim_audit"]["claims"][0]["execution_witness"]["receipt_sha256"]
        raw = jbytes(claim)
        claim_path.write_bytes(raw)
        self.fx.manifest["claim_audit"]["claims"][0]["claim_sha256"] = sha(raw)
        result = self.fx.audit()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["contradicted"], 1)
        self.assertIn("claim_contradicted", {row["code"] for row in result["issues"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
