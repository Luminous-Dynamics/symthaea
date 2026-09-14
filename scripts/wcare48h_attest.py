#!/usr/bin/env python3
"""WCARE-48H verifier for exact GitHub-hosted WCARE-48 generation evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

AUTHORITY = "MeasurementOnly"
RUN_ID = 34836223949
RUN_NUMBER = 1
RUN_ATTEMPT = 1
WORKFLOW_NAME = "WCARE-48 Observed Lock Generation"
WORKFLOW_PATH = ".github/workflows/wcare48-lock-generation.yml"
BRANCH = "wcare-48-observed-lock-generation"
PREPARED = "52d1d9fb741250ab8bcab205113689a8cc9431bb"
WCARE48V = "890ab746618f0a57853db1e38cedb7c1b500a89d"
LOCK = "tools/wcare42_builder_attestation_verifier/Cargo.lock"
RECEIPT = "docs/release/evidence/WCARE48_LOCK_GENERATION_RECEIPT_V1.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
COMMIT_LINE = re.compile(
    r"^[^\n]*\[[^\]]*\s([0-9a-f]{7,40})\]\s"
    r"evidence\(wcare48\): add observed standalone lock candidate\s*$",
    re.MULTILINE,
)
REQUIRED_STEPS = {
    "Checkout exact PREPARED subject",
    "Verify exact PREPARED checkout",
    "Install exact Rust toolchain",
    "Generate and preflight standalone lock",
    "Commit exact generated evidence",
}


class HostFailure(Exception):
    pass


class ProtocolFailure(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ProtocolFailure(f"json_invalid:{path.name}:{type(exc).__name__}") from exc
    if not isinstance(obj, dict):
        raise ProtocolFailure(f"json_not_object:{path.name}")
    return obj, raw


def run_git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise ProtocolFailure(f"git_failed:{' '.join(args)}:{proc.stderr.strip()}")
    return proc.stdout.strip()


def base(classification: str, detail: str) -> dict:
    return {
        "authority": AUTHORITY,
        "classification": classification,
        "detail": detail,
        "run_id": str(RUN_ID),
        "prepared_head": PREPARED,
        "wcare48v_subject": WCARE48V,
        "candidate": None,
        "run_json_sha256": None,
        "jobs_json_sha256": None,
        "job_log_sha256": None,
        "wcare48v_result_sha256": None,
        "receipt_run_identity_bound": False,
        "github_host_run_attested": False,
        "lock_admitted": False,
        "wcare42_executable_qualification_established": False,
        "runtime_authority_granted": False,
    }


def emit(result: dict, code: int) -> int:
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return code


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise HostFailure(detail)


def validate_run_identity(run: dict) -> None:
    expected = {
        "id": RUN_ID,
        "name": WORKFLOW_NAME,
        "path": WORKFLOW_PATH,
        "head_branch": BRANCH,
        "head_sha": PREPARED,
        "event": "push",
        "run_number": RUN_NUMBER,
        "run_attempt": RUN_ATTEMPT,
    }
    for key, value in expected.items():
        require(run.get(key) == value, f"run_identity_mismatch:{key}:{run.get(key)!r}")


def indeterminate(run_raw: bytes) -> dict:
    out = base("HOST_RUN_INDETERMINATE", "exact_run_not_completed")
    out["run_json_sha256"] = sha256_bytes(run_raw)
    return out


def get_candidate_receipt(root: Path, candidate: str) -> dict:
    raw = subprocess.run(
        ["git", "show", f"{candidate}:{RECEIPT}"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if raw.returncode != 0:
        raise HostFailure("candidate_receipt_missing")
    try:
        receipt = json.loads(raw.stdout.decode("utf-8"))
    except Exception as exc:
        raise HostFailure(f"candidate_receipt_invalid:{type(exc).__name__}") from exc
    if not isinstance(receipt, dict):
        raise HostFailure("candidate_receipt_not_object")
    return receipt


def parse_single_log_value(log: str, label: str, pattern: str) -> str:
    matches = re.findall(
        rf"^[^\n]*{re.escape(label)}=({pattern})\s*$",
        log,
        flags=re.MULTILINE,
    )
    require(len(matches) == 1, f"host_log_{label.lower()}_count:{len(matches)}")
    return matches[0]


def attest(
    root: Path,
    run_path: Path,
    jobs_path: Path | None,
    log_path: Path | None,
    candidate: str | None,
    branch_head: str | None,
    wcare48v_path: Path | None,
) -> tuple[dict, int]:
    run, run_raw = load_json(run_path)
    validate_run_identity(run)

    status = run.get("status")
    if status != "completed":
        require(
            status in {"queued", "in_progress", "pending", "requested", "waiting"},
            f"unexpected_run_status:{status}",
        )
        return indeterminate(run_raw), 3

    out = base("HOST_RUN_FAILED", "uninitialized")
    out["run_json_sha256"] = sha256_bytes(run_raw)
    require(run.get("conclusion") == "success", f"host_run_conclusion:{run.get('conclusion')}")

    if jobs_path is None or log_path is None or candidate is None or branch_head is None or wcare48v_path is None:
        raise ProtocolFailure("completed_run_requires_jobs_log_candidate_and_wcare48v")
    require(HEX40.fullmatch(candidate) is not None, "candidate_not_full_sha")
    require(branch_head == candidate, "branch_head_candidate_mismatch")

    jobs, jobs_raw = load_json(jobs_path)
    out["jobs_json_sha256"] = sha256_bytes(jobs_raw)
    job_list = jobs.get("jobs")
    require(isinstance(job_list, list), "jobs_list_missing")
    generate = [job for job in job_list if isinstance(job, dict) and job.get("name") == "generate"]
    require(len(generate) == 1, f"generate_job_count:{len(generate)}")
    job = generate[0]
    require(job.get("status") == "completed", f"generate_job_status:{job.get('status')}")
    require(job.get("conclusion") == "success", f"generate_job_conclusion:{job.get('conclusion')}")

    steps = job.get("steps")
    require(isinstance(steps, list), "generate_job_steps_missing")
    observed = {}
    for step in steps:
        if isinstance(step, dict) and isinstance(step.get("name"), str):
            observed[step["name"]] = step
    for name in REQUIRED_STEPS:
        require(name in observed, f"required_step_missing:{name}")
        require(observed[name].get("status") == "completed", f"required_step_status:{name}:{observed[name].get('status')}")
        require(observed[name].get("conclusion") == "success", f"required_step_conclusion:{name}:{observed[name].get('conclusion')}")

    log_raw = log_path.read_bytes()
    out["job_log_sha256"] = sha256_bytes(log_raw)
    try:
        log = log_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolFailure("job_log_not_utf8") from exc
    require(log.count("PASS_WCARE48_GENERATION_PRECOMMIT") == 1, "host_log_precommit_marker_count")
    lock_sha = parse_single_log_value(log, "LOCK_SHA256", r"[0-9a-f]{64}")
    lock_blob = parse_single_log_value(log, "LOCK_GIT_BLOB_CANDIDATE", r"[0-9a-f]{40}")
    package_count_text = parse_single_log_value(log, "PACKAGE_COUNT", r"[1-9][0-9]*")
    package_count = int(package_count_text)

    commit_matches = COMMIT_LINE.findall(log)
    require(len(commit_matches) == 1, f"host_log_commit_line_count:{len(commit_matches)}")
    commit_prefix = commit_matches[0]
    require(candidate.startswith(commit_prefix), "host_log_commit_prefix_mismatch")
    require("-> wcare-48-observed-lock-generation" in log, "host_log_push_target_missing")

    resolved = run_git(root, "rev-parse", "--verify", f"{candidate}^{{commit}}")
    require(resolved == candidate, "candidate_commit_missing")
    require(run_git(root, "rev-parse", f"{candidate}^") == PREPARED, "candidate_parent_mismatch")

    receipt = get_candidate_receipt(root, candidate)
    require(receipt.get("github_run_id") == str(RUN_ID), "receipt_run_id_mismatch")
    require(receipt.get("github_run_number") == str(RUN_NUMBER), "receipt_run_number_mismatch")
    require(receipt.get("github_run_attempt") == str(RUN_ATTEMPT), "receipt_run_attempt_mismatch")
    require(receipt.get("lock_sha256") == lock_sha, "host_receipt_lock_sha_mismatch")
    require(receipt.get("repeat_lock_sha256") == lock_sha, "host_receipt_repeat_sha_mismatch")
    require(receipt.get("lock_git_blob_candidate") == lock_blob, "host_receipt_lock_blob_mismatch")
    require(receipt.get("package_count") == package_count, "host_receipt_package_count_mismatch")

    committed_blob = run_git(root, "rev-parse", f"{candidate}:{LOCK}")
    require(committed_blob == lock_blob, "host_committed_lock_blob_mismatch")
    lock_proc = subprocess.run(
        ["git", "show", f"{candidate}:{LOCK}"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(lock_proc.returncode == 0, "host_committed_lock_missing")
    require(sha256_bytes(lock_proc.stdout) == lock_sha, "host_committed_lock_sha_mismatch")

    wcare48v, wcare48v_raw = load_json(wcare48v_path)
    out["wcare48v_result_sha256"] = sha256_bytes(wcare48v_raw)
    require(wcare48v.get("classification") == "FINAL_CHILD_VALID", "wcare48v_not_final_valid")
    require(wcare48v.get("target") == candidate, "wcare48v_target_mismatch")
    require(wcare48v.get("final_child_structurally_valid") is True, "wcare48v_structural_false")
    require(wcare48v.get("receipt_run_identity_bound") is True, "wcare48v_receipt_run_unbound")
    require(wcare48v.get("github_host_run_attested") is False, "wcare48v_host_overclaim")
    require(wcare48v.get("lock_admitted") is False, "wcare48v_lock_admission_overclaim")
    require(wcare48v.get("runtime_authority_granted") is False, "wcare48v_runtime_overclaim")

    out["classification"] = "HOST_RUN_ATTESTED"
    out["detail"] = "exact_github_host_run_and_final_child_consistent"
    out["candidate"] = candidate
    out["receipt_run_identity_bound"] = True
    out["github_host_run_attested"] = True
    out["lock_sha256"] = lock_sha
    out["lock_git_blob"] = lock_blob
    out["package_count"] = package_count
    return out, 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--jobs-json", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--candidate")
    parser.add_argument("--branch-head")
    parser.add_argument("--wcare48v-json", type=Path)
    args = parser.parse_args()

    root_proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if root_proc.returncode != 0:
        return emit(base("INVALID_HOST_ATTESTATION_PROTOCOL", "not_in_git_worktree"), 4)
    root = Path(root_proc.stdout.strip())

    try:
        result, code = attest(
            root,
            args.run_json,
            args.jobs_json,
            args.log,
            args.candidate,
            args.branch_head,
            args.wcare48v_json,
        )
        return emit(result, code)
    except HostFailure as exc:
        out = base("HOST_RUN_FAILED", str(exc))
        try:
            out["run_json_sha256"] = sha256_bytes(args.run_json.read_bytes())
        except OSError:
            pass
        return emit(out, 1)
    except (ProtocolFailure, OSError) as exc:
        return emit(base("INVALID_HOST_ATTESTATION_PROTOCOL", str(exc)), 4)


if __name__ == "__main__":
    raise SystemExit(main())
