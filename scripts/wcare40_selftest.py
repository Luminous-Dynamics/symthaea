#!/usr/bin/env python3
"""Dependency-free synthetic campaign for WCARE-40 replication semantics."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERIFIER = HERE / "wcare40_verify_replication.py"
PROTOCOL = "wcare40-execution-replication-v1"
W39_PROTOCOL = "wcare39-execution-capsule-v1"
DOMAIN_FIELDS = {
    "BuilderIdentity": "builder_identity_commitment_sha256",
    "Organization": "organization_commitment_sha256",
    "Infrastructure": "infrastructure_commitment_sha256",
    "ToolchainLineage": "toolchain_lineage_commitment_sha256",
    "OperatorProcess": "operator_process_commitment_sha256",
    "EvidenceSource": "evidence_source_commitment_sha256",
}


def h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return sha(path)


def base_prepared(env_label: str = "env-a") -> dict:
    return {
        "protocol_version": W39_PROTOCOL,
        "authority": "MeasurementOnly",
        "capsule_phase": "PREPARED",
        "classification": "CAPSULE_PREPARED",
        "environment_integrity": "QUALIFIED",
        "subject_outcome": "NOT_RUN",
        "subject_git_head": "a" * 40,
        "worktree_clean": True,
        "repository_root_commitment_sha256": h("repo"),
        "subject_digests": {"subject.py": h("subject")},
        "command_plan_sha256": h("command-plan"),
        "platform": {
            "os": "SyntheticOS", "kernel_release": "1", "architecture": "x86_64",
            "python_implementation": "CPython", "python_version": "3.13.0",
            "locale": "C.UTF-8", "timezone": "UTC",
        },
        "tools": [{
            "role": "python-runtime", "executable_path": "basename:python3;path_sha256:" + h("python-path"),
            "executable_sha256": h("python-binary"), "version_output_sha256": h("Python 3.13.0"),
            "version_output": "Python 3.13.0",
        }],
        "materials": [
            {"path": "Cargo.lock", "required": True, "present": True, "sha256": h("cargo-lock")},
            {"path": "flake.lock", "required": True, "present": True, "sha256": h("flake-lock")},
            {"path": "rust-toolchain.toml", "required": True, "present": True, "sha256": h("toolchain")},
        ],
        "safe_environment": [],
        "ambient_environment_sha256": h(env_label),
        "network_policy_declared": "Unspecified",
        "sandbox_policy_declared": "Unspecified",
        "deterministic_seed_commitments": {"seed": h("seed")},
        "commands": [{
            "stage_id": "qualify", "argv": ["python3", "subject.py"], "cwd": "", "timeout_seconds": 30,
            "started_utc": None, "finished_utc": None, "exit_code": None, "termination_class": "NotRun",
            "stdout_sha256": None, "stderr_sha256": None, "output_receipt_sha256": None,
            "subject_outcome": "NOT_RUN",
        }],
        "prepared_capsule_sha256": None,
        "drift_fields": [],
        "created_utc": "2026-09-13T18:00:00Z",
        "network_isolation_established": False,
        "sandbox_enforcement_established": False,
        "independent_builder_established": False,
        "phenomenal_experience_established": False,
        "suffering_established": False,
        "moral_patienthood_established": False,
        "binding_consent_established": False,
        "veto_authority_granted": False,
        "self_preservation_authority_granted": False,
        "runtime_authority_granted": False,
    }


def write_lineage(root: Path, replica_id: str, outcome: str, receipt_digest: str, env_label: str) -> tuple[Path, Path, str, str]:
    prepared = base_prepared(env_label)
    prepared_path = root / f"{replica_id}-prepared.json"
    prepared_sha = write_json(prepared_path, prepared)
    final = deepcopy(prepared)
    final["capsule_phase"] = "FINAL"
    final["classification"] = "QUALIFIED_EXECUTION"
    final["environment_integrity"] = "QUALIFIED"
    final["subject_outcome"] = outcome
    final["prepared_capsule_sha256"] = prepared_sha
    final["created_utc"] = "2026-09-13T18:01:00Z"
    command = final["commands"][0]
    command.update({
        "started_utc": "2026-09-13T18:00:10Z",
        "finished_utc": "2026-09-13T18:00:11Z",
        "exit_code": 0 if outcome == "PASS" else 7,
        "termination_class": "Exited",
        "stdout_sha256": h(f"stdout-{replica_id}"),
        "stderr_sha256": h(f"stderr-{replica_id}"),
        "output_receipt_sha256": receipt_digest,
        "subject_outcome": outcome,
    })
    final_path = root / f"{replica_id}-final.json"
    final_sha = write_json(final_path, final)
    return prepared_path, final_path, prepared_sha, final_sha


def builder_domains(label: str) -> dict[str, str]:
    return {
        "builder_identity_commitment_sha256": h(f"builder-{label}"),
        "organization_commitment_sha256": h(f"org-{label}"),
        "infrastructure_commitment_sha256": h(f"infra-{label}"),
        "toolchain_lineage_commitment_sha256": h(f"toolchain-{label}"),
        "operator_process_commitment_sha256": h(f"operator-{label}"),
        "evidence_source_commitment_sha256": h(f"evidence-{label}"),
    }


def shared_domains(left: dict[str, str], right: dict[str, str]) -> list[str]:
    return sorted(label for label, field in DOMAIN_FIELDS.items() if left[field] == right[field])


def execute_case(
    root: Path,
    name: str,
    slot_domains: dict[str, dict[str, str]],
    outcomes: dict[str, str | None],
    receipt_digests: dict[str, str],
    relation_overrides: dict[tuple[str, str], str] | None = None,
    forged_shared_domains: dict[tuple[str, str], list[str]] | None = None,
    min_qualified: int = 2,
    min_components: int = 2,
) -> tuple[int, dict]:
    case = root / name
    case.mkdir()
    replica_ids = sorted(slot_domains)
    plan = {
        "protocol_version": PROTOCOL,
        "wcare39_protocol_sha256": sha(ROOT / "docs/release/evidence/WCARE39_EXECUTION_CAPSULE_PROTOCOL_V1.md"),
        "wcare39_runner_sha256": sha(ROOT / "scripts/wcare39_execution_capsule.py"),
        "wcare39_integrity_sha256": sha(ROOT / "scripts/wcare39-integrity.sh"),
        "plan_created_utc": "2026-09-13T17:00:00Z",
        "minimum_qualified_replicas": min_qualified,
        "minimum_effective_independent_components": min_components,
        "accepted_builder_provenance_strengths": ["ExternalVerified", "InstitutionalAttestation"],
        "accepted_independent_relation_strengths": ["ExternalVerified", "InstitutionalAttestation"],
        "required_receipt_stage_ids": ["qualify"],
        "require_subject_outcome_agreement": True,
        "require_required_receipt_agreement": True,
        "require_no_conflict_of_interest": True,
        "replica_slots": [
            {"replica_id": replica_id, "builder_identity_commitment_sha256": slot_domains[replica_id]["builder_identity_commitment_sha256"]}
            for replica_id in replica_ids
        ],
    }
    plan_path = case / "plan.json"
    plan_sha = write_json(plan_path, plan)

    capsules: list[Path] = []
    provenance_paths: list[Path] = []
    capsule_hashes: dict[str, tuple[str | None, str | None]] = {}
    for index, replica_id in enumerate(replica_ids):
        outcome = outcomes[replica_id]
        if outcome is None:
            prepared_sha = final_sha = None
        else:
            prepared_path, final_path, prepared_sha, final_sha = write_lineage(
                case, replica_id, outcome, receipt_digests[replica_id], f"env-{index}"
            )
            capsules.extend((prepared_path, final_path))
        capsule_hashes[replica_id] = (prepared_sha, final_sha)
        prov = {
            "protocol_version": PROTOCOL,
            "plan_sha256": plan_sha,
            "replica_id": replica_id,
            "execution_observed": outcome is not None,
            "wcare39_prepared_capsule_sha256": prepared_sha,
            "wcare39_final_capsule_sha256": final_sha,
            **slot_domains[replica_id],
            "provenance_strength": "ExternalVerified",
            "conflict_of_interest": False,
            "collected_utc": "2026-09-13T17:30:00Z",
        }
        path = case / f"{replica_id}-provenance.json"
        write_json(path, prov)
        provenance_paths.append(path)

    relation_paths: list[Path] = []
    overrides = relation_overrides or {}
    forged = forged_shared_domains or {}
    for i, left in enumerate(replica_ids):
        for right in replica_ids[i + 1:]:
            pair = (left, right)
            actual_shared = shared_domains(slot_domains[left], slot_domains[right])
            relation = {
                "protocol_version": PROTOCOL,
                "plan_sha256": plan_sha,
                "left_replica_id": left,
                "right_replica_id": right,
                "left_builder_identity_commitment_sha256": slot_domains[left]["builder_identity_commitment_sha256"],
                "right_builder_identity_commitment_sha256": slot_domains[right]["builder_identity_commitment_sha256"],
                "relation": overrides.get(pair, "Independent"),
                "declared_shared_fault_domains": forged.get(pair, actual_shared),
                "relation_evidence_strength": "ExternalVerified",
                "assessed_utc": "2026-09-13T17:40:00Z",
            }
            path = case / f"{left}-{right}-relation.json"
            write_json(path, relation)
            relation_paths.append(path)

    argv = [sys.executable, str(VERIFIER), str(plan_path), "--capsules", *map(str, capsules), "--provenance", *map(str, provenance_paths), "--relations", *map(str, relation_paths)]
    proc = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=900)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{name}: non-json output: {proc.stdout!r} {proc.stderr!r}") from exc
    return proc.returncode, payload


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="wcare40-") as tmp:
        root = Path(tmp)
        same_receipt = h("qualified-evidence")

        domains = {"a": builder_domains("a"), "b": builder_domains("b"), "c": builder_domains("c")}
        outcomes = {"a": "PASS", "b": "PASS", "c": "PASS"}
        receipts = {key: same_receipt for key in domains}
        code, supported = execute_case(root, "supported", domains, outcomes, receipts, min_qualified=2, min_components=2)
        assert code == 0 and supported["disposition"] == "REPLICATION_SUPPORTED", supported
        assert supported["effective_independent_components"] == 3, supported
        assert supported["accepted_independent_pair_count"] == 3, supported
        assert supported["subject_outcome_agreement_met"] is True and supported["agreed_subject_outcome"] == "PASS", supported
        assert supported["builder_authentication_established"] is False, supported
        assert supported["preregistration_temporal_precedence_established"] is False, supported
        assert len(supported["wcare39_compare_receipt_sha256_by_replica"]) == 3, supported
        assert is_hex64(supported["wcare39_integrity_receipt_sha256"]), supported

        retry_domains = {"a": builder_domains("same"), "b": builder_domains("same"), "c": builder_domains("c")}
        code, retry = execute_case(root, "same-builder", retry_domains, outcomes, receipts, min_qualified=3, min_components=3)
        assert code == 2 and retry["disposition"] == "REPLICATION_LIMITED", retry
        assert retry["raw_distinct_builder_identity_count"] == 2, retry
        assert retry["effective_independent_components"] == 2, retry
        assert retry["downgraded_independent_pair_count"] >= 1, retry

        contradicted_outcomes = {"a": "PASS", "b": "FAIL"}
        two_domains = {"a": builder_domains("a"), "b": builder_domains("b")}
        two_receipts = {"a": same_receipt, "b": same_receipt}
        code, contradicted = execute_case(root, "outcome-contradiction", two_domains, contradicted_outcomes, two_receipts)
        assert code == 1 and contradicted["disposition"] == "REPLICATION_CONTRADICTED", contradicted
        assert contradicted["pass_replica_ids"] == ["a"] and contradicted["fail_replica_ids"] == ["b"], contradicted

        mismatch_receipts = {"a": h("receipt-a"), "b": h("receipt-b")}
        code, receipt_conflict = execute_case(root, "receipt-contradiction", two_domains, {"a": "PASS", "b": "PASS"}, mismatch_receipts)
        assert code == 1 and receipt_conflict["disposition"] == "REPLICATION_CONTRADICTED", receipt_conflict
        assert receipt_conflict["required_receipt_agreement_met"] is False, receipt_conflict

        code, missing = execute_case(root, "missing-slot", two_domains, {"a": "PASS", "b": None}, two_receipts)
        assert code == 3 and missing["disposition"] == "INFRASTRUCTURE_INDETERMINATE", missing
        assert missing["missing_replica_ids"] == ["b"], missing

        shared_org = {"a": builder_domains("a"), "b": builder_domains("b")}
        shared_org["b"]["organization_commitment_sha256"] = shared_org["a"]["organization_commitment_sha256"]
        code, forged = execute_case(
            root, "forged-shared-domain", shared_org, {"a": "PASS", "b": "PASS"}, two_receipts,
            forged_shared_domains={("a", "b"): []},
        )
        assert code == 4 and forged["disposition"] == "REPLICATION_INVALID", forged
        assert "relation_shared_fault_domain_mismatch" in forged["detail"], forged

        one_domain = {"a": builder_domains("a")}
        code, one = execute_case(root, "one-run", one_domain, {"a": "PASS"}, {"a": same_receipt}, min_qualified=2, min_components=2)
        assert code == 4 and one["disposition"] == "REPLICATION_INVALID", one

    print(json.dumps({
        "authority": "MeasurementOnly",
        "classification": "PASS_WCARE40_SELFTEST",
        "independent_replication_supported": True,
        "same_builder_retry_does_not_multiply_independence": True,
        "pass_fail_contradiction_preserved": True,
        "required_receipt_contradiction_preserved": True,
        "missing_preregistered_slot_is_indeterminate": True,
        "forged_shared_fault_domain_rejected": True,
        "single_execution_cannot_be_replication": True,
        "runtime_authority_granted": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


def is_hex64(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


if __name__ == "__main__":
    raise SystemExit(main())
