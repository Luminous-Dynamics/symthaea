#!/usr/bin/env python3
"""Official WCARE-40 front door.

Validates internal WCARE-39 PREPARED/FINAL outcome semantics before delegating
replication/independence recomputation to the frozen WCARE-40 core verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROTOCOL = "wcare40-execution-replication-v1"
W39_PROTOCOL = "wcare39-execution-capsule-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return raw, value


def false_claims() -> dict[str, bool]:
    return {
        "preregistration_temporal_precedence_established": False,
        "builder_authentication_established": False,
        "independent_builder_identity_established_beyond_commitments": False,
        "reviewer_independence_established": False,
        "subject_correctness_established": False,
        "network_isolation_established": False,
        "sandbox_enforcement_established": False,
        "phenomenal_experience_established": False,
        "suffering_established": False,
        "moral_patienthood_established": False,
        "binding_consent_established": False,
        "veto_authority_granted": False,
        "self_preservation_authority_granted": False,
        "runtime_authority_granted": False,
    }


def invalid(detail: str, **extra: Any) -> int:
    payload: dict[str, Any] = {
        "authority": "MeasurementOnly",
        "protocol_version": PROTOCOL,
        "disposition": "REPLICATION_INVALID",
        "detail": detail,
        **false_claims(),
    }
    payload.update(extra)
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return 4


def aggregate_outcome(commands: object) -> str:
    if not isinstance(commands, list) or not commands:
        raise ValueError("wcare39_commands_missing_or_empty")
    outcomes: list[str] = []
    for entry in commands:
        if not isinstance(entry, dict):
            raise ValueError("wcare39_command_not_object")
        outcome = entry.get("subject_outcome")
        if outcome not in {"PASS", "FAIL", "INVALID", "INDETERMINATE", "NOT_RUN"}:
            raise ValueError("wcare39_command_subject_outcome_invalid")
        outcomes.append(outcome)
    for value in ("INVALID", "INDETERMINATE", "FAIL"):
        if value in outcomes:
            return value
    return "PASS" if all(value == "PASS" for value in outcomes) else "NOT_RUN"


def validate_prepared(prepared: dict[str, Any], replica_id: str) -> None:
    if prepared.get("protocol_version") != W39_PROTOCOL:
        raise ValueError(f"prepared_protocol_mismatch:{replica_id}")
    if prepared.get("authority") != "MeasurementOnly":
        raise ValueError(f"prepared_authority_mismatch:{replica_id}")
    if prepared.get("capsule_phase") != "PREPARED":
        raise ValueError(f"prepared_phase_mismatch:{replica_id}")
    if prepared.get("classification") != "CAPSULE_PREPARED":
        raise ValueError(f"prepared_classification_mismatch:{replica_id}")
    if prepared.get("environment_integrity") != "QUALIFIED":
        raise ValueError(f"prepared_environment_integrity_mismatch:{replica_id}")
    if prepared.get("subject_outcome") != "NOT_RUN":
        raise ValueError(f"prepared_subject_outcome_not_not_run:{replica_id}")
    if prepared.get("prepared_capsule_sha256") is not None:
        raise ValueError(f"prepared_self_binding_must_be_null:{replica_id}")
    if prepared.get("drift_fields") != []:
        raise ValueError(f"prepared_drift_fields_not_empty:{replica_id}")
    if prepared.get("worktree_clean") is not True:
        raise ValueError(f"prepared_worktree_not_clean:{replica_id}")
    commands = prepared.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError(f"prepared_commands_missing:{replica_id}")
    for entry in commands:
        if not isinstance(entry, dict) or entry.get("subject_outcome") != "NOT_RUN" or entry.get("termination_class") != "NotRun":
            raise ValueError(f"prepared_command_not_pristine:{replica_id}")


def validate_final(final: dict[str, Any], replica_id: str) -> None:
    if final.get("protocol_version") != W39_PROTOCOL or final.get("authority") != "MeasurementOnly":
        raise ValueError(f"final_protocol_or_authority_mismatch:{replica_id}")
    if final.get("capsule_phase") != "FINAL":
        raise ValueError(f"final_phase_mismatch:{replica_id}")
    derived = aggregate_outcome(final.get("commands"))
    if final.get("subject_outcome") != derived:
        raise ValueError(f"final_subject_outcome_recomputation_mismatch:{replica_id}:{derived}")
    classification = final.get("classification")
    integrity = final.get("environment_integrity")
    drift = final.get("drift_fields")
    if classification == "QUALIFIED_EXECUTION":
        if integrity != "QUALIFIED" or drift != []:
            raise ValueError(f"qualified_final_integrity_mismatch:{replica_id}")
    elif classification == "ENVIRONMENT_DRIFT":
        if integrity != "DRIFTED" or not isinstance(drift, list) or not drift:
            raise ValueError(f"drift_final_integrity_mismatch:{replica_id}")
    elif classification == "INFRASTRUCTURE_INDETERMINATE":
        if integrity != "INDETERMINATE":
            raise ValueError(f"indeterminate_final_integrity_mismatch:{replica_id}")
    else:
        raise ValueError(f"final_classification_not_composable:{replica_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--capsules", nargs="*", default=[], type=Path)
    parser.add_argument("--provenance", nargs="*", default=[], type=Path)
    parser.add_argument("--relations", nargs="*", default=[], type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    core = root / "scripts/wcare40_verify_replication.py"
    frontdoor = Path(__file__).resolve()
    try:
        core_sha = sha256_bytes(core.read_bytes())
        frontdoor_sha = sha256_bytes(frontdoor.read_bytes())
        capsule_by_sha: dict[str, dict[str, Any]] = {}
        for path in args.capsules:
            raw, capsule = read_object(path)
            digest = sha256_bytes(raw)
            if digest in capsule_by_sha:
                return invalid("duplicate_capsule_bytes_at_frontdoor", capsule_sha256=digest)
            capsule_by_sha[digest] = capsule
        provenance: list[dict[str, Any]] = []
        for path in args.provenance:
            _raw, receipt = read_object(path)
            provenance.append(receipt)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return invalid(f"frontdoor_artifact_read_failed:{type(exc).__name__}:{exc}")

    seen_replicas: set[str] = set()
    try:
        for receipt in provenance:
            replica_id = receipt.get("replica_id")
            if not isinstance(replica_id, str) or not replica_id or replica_id in seen_replicas:
                raise ValueError("frontdoor_provenance_replica_invalid_or_duplicate")
            seen_replicas.add(replica_id)
            observed = receipt.get("execution_observed")
            if observed is True:
                prepared_sha = receipt.get("wcare39_prepared_capsule_sha256")
                final_sha = receipt.get("wcare39_final_capsule_sha256")
                if prepared_sha not in capsule_by_sha or final_sha not in capsule_by_sha:
                    raise ValueError(f"frontdoor_missing_bound_capsule:{replica_id}")
                validate_prepared(capsule_by_sha[prepared_sha], replica_id)
                validate_final(capsule_by_sha[final_sha], replica_id)
            elif observed is not False:
                raise ValueError(f"frontdoor_execution_observed_not_boolean:{replica_id}")
    except ValueError as exc:
        return invalid(str(exc), wcare40_core_verifier_sha256=core_sha, wcare40_frontdoor_sha256=frontdoor_sha)

    command = [
        sys.executable,
        str(core),
        str(args.plan),
        "--capsules",
        *[str(path) for path in args.capsules],
        "--provenance",
        *[str(path) for path in args.provenance],
        "--relations",
        *[str(path) for path in args.relations],
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=3600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        payload = {
            "authority": "MeasurementOnly",
            "protocol_version": PROTOCOL,
            "disposition": "INFRASTRUCTURE_INDETERMINATE",
            "detail": f"wcare40_core_execution_failed:{type(exc).__name__}",
            "wcare40_core_verifier_sha256": core_sha,
            "wcare40_frontdoor_sha256": frontdoor_sha,
            **false_claims(),
        }
        sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        return 3

    try:
        result = json.loads(completed.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return invalid(
            "wcare40_core_nonjson_output",
            wcare40_core_verifier_sha256=core_sha,
            wcare40_frontdoor_sha256=frontdoor_sha,
        )
    if not isinstance(result, dict):
        return invalid(
            "wcare40_core_output_not_object",
            wcare40_core_verifier_sha256=core_sha,
            wcare40_frontdoor_sha256=frontdoor_sha,
        )

    result["wcare40_core_verifier_sha256"] = core_sha
    result["wcare40_frontdoor_sha256"] = frontdoor_sha
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    disposition = result.get("disposition")
    expected = {
        "REPLICATION_SUPPORTED": 0,
        "REPLICATION_CONTRADICTED": 1,
        "REPLICATION_LIMITED": 2,
        "INFRASTRUCTURE_INDETERMINATE": 3,
        "REPLICATION_INVALID": 4,
    }.get(disposition, 4)
    if completed.returncode != expected:
        return 4
    return expected


if __name__ == "__main__":
    raise SystemExit(main())
