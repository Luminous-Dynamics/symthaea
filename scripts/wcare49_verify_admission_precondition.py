#!/usr/bin/env python3
"""WCARE-49 exact WCARE-47Q hosted admission precondition verifier.

MeasurementOnly. This verifier was preregistered while the exact WCARE-47Q run
was still queued. It does not perform lock admission; it verifies that the
already-frozen WCARE-47Q event reached the exact required result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

AUTHORITY = "MeasurementOnly"
RUN_ID = 34871019291
RUN_NUMBER = 4
RUN_ATTEMPT = 1
RUN_NAME = "WCARE-47Q Exact FINAL Lock Admission"
RUN_PATH = ".github/workflows/wcare47q-final-lock-admission.yml"
RUN_BRANCH = "wcare-47q-final-lock-admission"
RUN_HEAD = "8eb8af15af464ae6c49d20de225aa501ec4bedaf"
FINAL = "5bc23735f545b1b82044820f0e02ece59be04b4a"
LOCK_SHA256 = "7288b7dd64b533a4e08ae9a66ff120fb69d5885effd80b0e3b7f51455028d976"
LOCK_BLOB = "1a9b126e3d5062bd3be5290bfa5930d189760ceb"
PROTOCOL = "wcare47-standalone-lock-admission-v1"

REQUIRED_STEPS = {
    "Checkout exact qualification harness",
    "Verify one-commit sibling qualification lineage",
    "Checkout exact WCARE-48 FINAL candidate",
    "Bind frozen verifier and exact candidate",
    "Install exact Rust toolchain",
    "Execute frozen WCARE-47 admission theorem",
}

EXPECTED_RESULT = {
    "authority": AUTHORITY,
    "classification": "LOCK_ADMITTED",
    "detail": "exact_lock_admitted_for_dependency_resolution_only",
    "head": FINAL,
    "wcare46_head": "8a3cacb449b923ceee32b6b08e2c811ce532c676",
    "exact_source_subject_bound": True,
    "rust_toolchain_subject_bound": True,
    "candidate_lock_present": True,
    "lock_sha256": LOCK_SHA256,
    "lock_git_blob": LOCK_BLOB,
    "lock_format": 4,
    "package_count": 45,
    "registry_checksum_policy_satisfied": True,
    "rustc_1_96_0_verified": True,
    "metadata_locked_passed": True,
    "tests_locked_passed": True,
    "source_postflight_unchanged": True,
    "lock_generation_provenance_established": False,
    "lock_admitted": True,
    "wcare42_executable_qualification_established": False,
    "builder_authentication_established": False,
    "preregistration_temporal_precedence_established": False,
    "runtime_authority_granted": False,
}

HEX64 = re.compile(r"^[0-9a-f]{64}$")
RESULT_HASH_MARKER = re.compile(r"WCARE47Q_RESULT_SHA256=([0-9a-f]{64})")


class EvidenceFailure(Exception):
    pass


class ProtocolFailure(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ProtocolFailure(f"json_invalid:{path.name}:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ProtocolFailure(f"json_not_object:{path.name}")
    return value, raw


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise EvidenceFailure(detail)


def base(classification: str, detail: str) -> dict:
    return {
        "authority": AUTHORITY,
        "classification": classification,
        "detail": detail,
        "wcare47q_run_id": str(RUN_ID),
        "wcare47q_run_number": str(RUN_NUMBER),
        "wcare47q_run_attempt": str(RUN_ATTEMPT),
        "wcare47q_head": RUN_HEAD,
        "final_subject": FINAL,
        "lock_sha256": LOCK_SHA256,
        "lock_git_blob": LOCK_BLOB,
        "run_json_sha256": None,
        "jobs_json_sha256": None,
        "job_log_sha256": None,
        "wcare47_result_sha256": None,
        "lock_admitted": False,
        "execution_inputs_closed": False,
        "wcare42_executable_qualification_established": False,
        "runtime_authority_granted": False,
    }


def validate_run_identity(run: dict) -> None:
    expected = {
        "id": RUN_ID,
        "name": RUN_NAME,
        "path": RUN_PATH,
        "head_branch": RUN_BRANCH,
        "head_sha": RUN_HEAD,
        "run_number": RUN_NUMBER,
        "run_attempt": RUN_ATTEMPT,
        "event": "push",
    }
    for key, expected_value in expected.items():
        require(run.get(key) == expected_value, f"run_identity_mismatch:{key}:{run.get(key)!r}")


def extract_payload_lines(log: str) -> list[str]:
    payloads: list[str] = []
    for raw_line in log.splitlines():
        brace = raw_line.find("{")
        if brace >= 0:
            payloads.append(raw_line[brace:].strip())
    return payloads


def extract_result(log: str) -> tuple[dict, str]:
    marker_matches = RESULT_HASH_MARKER.findall(log)
    require(len(marker_matches) == 1, f"result_hash_marker_count:{len(marker_matches)}")
    marker_hash = marker_matches[0]

    unique: dict[str, dict] = {}
    for payload in extract_payload_lines(log):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or value.get("protocol_version") != PROTOCOL:
            continue
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if sha256_bytes((canonical + "\n").encode("utf-8")) == marker_hash:
            unique.setdefault(canonical, value)

    require(len(unique) == 1, f"unique_result_matching_hash_count:{len(unique)}")
    canonical, result = next(iter(unique.items()))
    for key, expected in EXPECTED_RESULT.items():
        require(result.get(key) == expected, f"admission_result_mismatch:{key}:{result.get(key)!r}")
    cargo = result.get("cargo_identity")
    require(isinstance(cargo, str) and cargo.startswith("cargo 1.96.0 "), "admission_cargo_identity_mismatch")
    require("PASS_WCARE47Q_EXACT_FINAL_LOCK_ADMISSION" in log, "admission_pass_marker_missing")
    require("PASS_PROTOCOL_INTEGRITY" in log, "wcare47_protocol_integrity_marker_missing")
    require("PASS_WCARE47_CURRENT_STATE:LOCK_ADMITTED" in log, "wcare47_lock_admitted_marker_missing")
    return result, marker_hash


def verify(run_path: Path, jobs_path: Path | None, log_path: Path | None) -> tuple[dict, int]:
    run, run_raw = load_json(run_path)
    validate_run_identity(run)

    status = run.get("status")
    if status != "completed":
        require(status in {"queued", "in_progress", "pending", "requested", "waiting"}, f"unexpected_run_status:{status}")
        out = base("ADMISSION_PRECONDITION_PENDING", "exact_wcare47q_run_not_completed")
        out["run_json_sha256"] = sha256_bytes(run_raw)
        return out, 3

    out = base("ADMISSION_PRECONDITION_FAILED", "uninitialized")
    out["run_json_sha256"] = sha256_bytes(run_raw)
    require(run.get("conclusion") == "success", f"wcare47q_run_conclusion:{run.get('conclusion')}")
    if jobs_path is None or log_path is None:
        raise ProtocolFailure("completed_run_requires_jobs_and_log")

    jobs, jobs_raw = load_json(jobs_path)
    out["jobs_json_sha256"] = sha256_bytes(jobs_raw)
    job_list = jobs.get("jobs")
    require(isinstance(job_list, list), "jobs_list_missing")
    matching = [job for job in job_list if isinstance(job, dict) and job.get("name") == "qualify"]
    require(len(matching) == 1, f"qualify_job_count:{len(matching)}")
    job = matching[0]
    require(job.get("status") == "completed", f"qualify_job_status:{job.get('status')}")
    require(job.get("conclusion") == "success", f"qualify_job_conclusion:{job.get('conclusion')}")

    steps = job.get("steps")
    require(isinstance(steps, list), "qualify_steps_missing")
    by_name = {step.get("name"): step for step in steps if isinstance(step, dict) and isinstance(step.get("name"), str)}
    for name in REQUIRED_STEPS:
        require(name in by_name, f"required_step_missing:{name}")
        require(by_name[name].get("status") == "completed", f"required_step_status:{name}:{by_name[name].get('status')}")
        require(by_name[name].get("conclusion") == "success", f"required_step_conclusion:{name}:{by_name[name].get('conclusion')}")

    log_raw = log_path.read_bytes()
    out["job_log_sha256"] = sha256_bytes(log_raw)
    try:
        log = log_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolFailure("job_log_not_utf8") from exc

    _, result_hash = extract_result(log)
    require(HEX64.fullmatch(result_hash) is not None, "invalid_result_hash")

    out["classification"] = "ADMISSION_PRECONDITION_SATISFIED"
    out["detail"] = "exact_hosted_wcare47q_lock_admission_verified"
    out["wcare47_result_sha256"] = result_hash
    out["lock_admitted"] = True
    return out, 0


def emit(value: dict, code: int) -> int:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-json", required=True, type=Path)
    parser.add_argument("--jobs-json", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()

    try:
        result, code = verify(args.run_json, args.jobs_json, args.log)
        return emit(result, code)
    except EvidenceFailure as exc:
        return emit(base("ADMISSION_PRECONDITION_FAILED", str(exc)), 1)
    except (ProtocolFailure, OSError) as exc:
        return emit(base("INVALID_ADMISSION_PRECONDITION_PROTOCOL", str(exc)), 4)


if __name__ == "__main__":
    raise SystemExit(main())
