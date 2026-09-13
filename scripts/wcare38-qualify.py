#!/usr/bin/env python3
"""Official WCARE-38 qualifier.

Proof chain:
  exact WCARE-36 verifier -> validated WCARE-36 upper bound -> WCARE-38
  authentication overlay.

The overlay is never allowed to strengthen the validated WCARE-36 result.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROTOCOL = "wcare38-authenticated-panel-v1"
QUALIFYING_DISPOSITIONS = {
    "AUTHENTICATED_PANEL_SUPPORTED",
    "AUTHENTICATED_PANEL_LIMITED",
    "INFRASTRUCTURE_INDETERMINATE",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_object(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be object")
    return raw, value


def parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or len(value) != 20 or not value.endswith("Z"):
        raise ValueError("timestamp must be canonical YYYY-MM-DDTHH:MM:SSZ")
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("timestamp is not canonical")
    return parsed


def emit_invalid(detail: str, **extra: object) -> int:
    payload = {
        "authority": "MeasurementOnly",
        "protocol_version": PROTOCOL,
        "disposition": "AUTHENTICATION_INVALID",
        "detail": detail,
        "wcare36_baseline_integrity_verified": False,
        "reviewer_correctness_established": False,
        "objective_moral_truth_established": False,
        "universal_cultural_validity_established": False,
        "phenomenal_experience_established": False,
        "suffering_established": False,
        "moral_patienthood_established": False,
        "binding_consent_established": False,
        "veto_authority_granted": False,
        "self_preservation_authority_granted": False,
        "runtime_authority_granted": False,
    }
    payload.update(extra)
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return 4


def sha_set(result: dict, field: str) -> set[str]:
    value = result.get(field)
    if not isinstance(value, list) or len(value) != len(set(value)):
        raise ValueError(f"{field}_not_unique_array")
    output: set[str] = set()
    for item in value:
        if (
            not isinstance(item, str)
            or len(item) != 64
            or any(ch not in "0123456789abcdef" for ch in item)
        ):
            raise ValueError(f"{field}_contains_invalid_sha256")
        output.add(item)
    return output


def verify_partition(
    result: dict,
    required_field: str,
    authenticated_field: str,
    unauthenticated_field: str,
    indeterminate_field: str,
) -> set[str]:
    required = sha_set(result, required_field)
    authenticated = sha_set(result, authenticated_field)
    unauthenticated = sha_set(result, unauthenticated_field)
    indeterminate = sha_set(result, indeterminate_field)
    if authenticated & unauthenticated or authenticated & indeterminate or unauthenticated & indeterminate:
        raise ValueError(f"{required_field}_authentication_states_overlap")
    if authenticated | unauthenticated | indeterminate != required:
        raise ValueError(f"{required_field}_authentication_partition_incomplete")
    return required


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wcare38_plan", type=Path)
    parser.add_argument("wcare35_result", type=Path)
    parser.add_argument("wcare36_plan", type=Path)
    parser.add_argument("wcare36_result", type=Path)
    parser.add_argument("attestation_manifest", type=Path)
    parser.add_argument("--provenance", nargs="+", required=True, type=Path)
    parser.add_argument("--relations", nargs="+", required=True, type=Path)
    args = parser.parse_args()

    try:
        _plan_raw, plan = load_object(args.wcare38_plan)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return emit_invalid(f"plan_read_or_parse_failed:{type(exc).__name__}")

    if plan.get("protocol_version") != PROTOCOL:
        return emit_invalid("protocol_version_mismatch")

    try:
        plan_created = parse_utc(plan.get("plan_created_utc"))
        evaluation_time = parse_utc(plan.get("evaluation_utc"))
    except ValueError:
        return emit_invalid("plan_timestamp_invalid")
    if plan_created > evaluation_time:
        return emit_invalid("plan_created_after_evaluation_time")

    root = Path(__file__).resolve().parent.parent
    w36_verifier = root / "scripts" / "wcare36_verify_independence.py"
    overlay_engine = root / "scripts" / "wcare38_qualify_authenticated_panel.py"
    try:
        w36_verifier_sha = sha256_bytes(w36_verifier.read_bytes())
    except OSError:
        return emit_invalid("wcare36_verifier_unavailable")
    if plan.get("wcare36_verifier_sha256") != w36_verifier_sha:
        return emit_invalid(
            "wcare36_verifier_digest_mismatch",
            observed_wcare36_verifier_sha256=w36_verifier_sha,
        )

    baseline_command = [
        sys.executable,
        str(w36_verifier),
        str(args.wcare36_plan),
        str(args.wcare35_result),
        str(args.wcare36_result),
        "--provenance",
        *[str(path) for path in args.provenance],
        "--relations",
        *[str(path) for path in args.relations],
    ]
    try:
        baseline = subprocess.run(
            baseline_command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        payload = {
            "authority": "MeasurementOnly",
            "protocol_version": PROTOCOL,
            "disposition": "INFRASTRUCTURE_INDETERMINATE",
            "detail": f"wcare36_verifier_execution_failed:{type(exc).__name__}",
            "wcare36_verifier_sha256": w36_verifier_sha,
            "wcare36_baseline_integrity_verified": False,
            "runtime_authority_granted": False,
        }
        sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        return 3

    try:
        baseline_result = json.loads(baseline.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return emit_invalid(
            "wcare36_verifier_nonjson_output",
            wcare36_verifier_sha256=w36_verifier_sha,
        )
    if (
        baseline.returncode != 0
        or not isinstance(baseline_result, dict)
        or baseline_result.get("independence_integrity_verified") is not True
    ):
        return emit_invalid(
            "wcare36_baseline_not_integrity_verified",
            wcare36_verifier_sha256=w36_verifier_sha,
            wcare36_verifier_exit=baseline.returncode,
            wcare36_verifier_disposition=(
                baseline_result.get("disposition")
                if isinstance(baseline_result, dict)
                else None
            ),
        )

    overlay_command = [
        sys.executable,
        str(overlay_engine),
        str(args.wcare38_plan),
        str(args.wcare35_result),
        str(args.wcare36_plan),
        str(args.wcare36_result),
        str(args.attestation_manifest),
        "--provenance",
        *[str(path) for path in args.provenance],
        "--relations",
        *[str(path) for path in args.relations],
    ]
    try:
        overlay = subprocess.run(
            overlay_command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=1800,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        payload = {
            "authority": "MeasurementOnly",
            "protocol_version": PROTOCOL,
            "disposition": "INFRASTRUCTURE_INDETERMINATE",
            "detail": f"wcare38_overlay_execution_failed:{type(exc).__name__}",
            "wcare36_verifier_sha256": w36_verifier_sha,
            "wcare36_baseline_integrity_verified": True,
            "runtime_authority_granted": False,
        }
        sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        return 3

    try:
        result = json.loads(overlay.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return emit_invalid(
            "wcare38_overlay_nonjson_output",
            wcare36_verifier_sha256=w36_verifier_sha,
            wcare36_baseline_integrity_verified=True,
        )
    if not isinstance(result, dict):
        return emit_invalid(
            "wcare38_overlay_output_not_object",
            wcare36_verifier_sha256=w36_verifier_sha,
            wcare36_baseline_integrity_verified=True,
        )

    if result.get("disposition") in QUALIFYING_DISPOSITIONS:
        try:
            required_provenance = verify_partition(
                result,
                "required_provenance_receipt_sha256s",
                "authenticated_provenance_receipt_sha256s",
                "unauthenticated_provenance_receipt_sha256s",
                "indeterminate_provenance_receipt_sha256s",
            )
            required_relations = verify_partition(
                result,
                "required_relation_receipt_sha256s",
                "authenticated_relation_receipt_sha256s",
                "unauthenticated_relation_receipt_sha256s",
                "indeterminate_relation_receipt_sha256s",
            )
            _manifest_raw, manifest = load_object(args.attestation_manifest)
            entries = manifest.get("entries")
            if not isinstance(entries, list):
                raise ValueError("manifest_entries_not_array")
            manifest_subjects: list[str] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("manifest_entry_not_object")
                subject = entry.get("subject_receipt_sha256")
                if (
                    not isinstance(subject, str)
                    or len(subject) != 64
                    or any(ch not in "0123456789abcdef" for ch in subject)
                ):
                    raise ValueError("manifest_subject_invalid_sha256")
                manifest_subjects.append(subject)
            if len(manifest_subjects) != len(set(manifest_subjects)):
                raise ValueError("manifest_subject_duplicate")
            supplemental = sorted(set(manifest_subjects) - required_provenance - required_relations)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            return emit_invalid(
                f"authentication_partition_or_manifest_invalid:{exc}",
                wcare36_verifier_sha256=w36_verifier_sha,
                wcare36_baseline_integrity_verified=True,
            )
        result["provenance_authentication_partition_complete"] = True
        result["relation_authentication_partition_complete"] = True
        result["supplemental_attestation_subject_sha256s"] = supplemental
        result["supplemental_packages_contribute_weight"] = False

    # The front door appends the exact baseline verifier evidence. The overlay
    # remains responsible for the monotonic graph/authentication derivation.
    result["wcare36_verifier_sha256"] = w36_verifier_sha
    result["wcare36_baseline_integrity_verified"] = True
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")

    if result.get("disposition") == "AUTHENTICATED_PANEL_SUPPORTED":
        return 0
    if result.get("disposition") == "AUTHENTICATED_PANEL_LIMITED":
        return 2
    if result.get("disposition") == "INFRASTRUCTURE_INDETERMINATE":
        return 3
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
