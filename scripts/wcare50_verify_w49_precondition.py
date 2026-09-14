#!/usr/bin/env python3
"""Verify the exact preregistered WCARE-49 hosted closure event for WCARE-50."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

AUTHORITY = "MeasurementOnly"
RUN_ID = 34877774593
RUN_NUMBER = 7
RUN_ATTEMPT = 1
RUN_NAME = "WCARE-49 Preregistered Execution Input Closure Gate"
RUN_PATH = ".github/workflows/wcare49-execution-input-closure.yml"
RUN_BRANCH = "wcare-49-execution-input-closure"
RUN_HEAD = "720c7ebf6c03b5698c48a4d2315ea9c4d25107e7"
FINAL = "5bc23735f545b1b82044820f0e02ece59be04b4a"
PREPARED = "52d1d9fb741250ab8bcab205113689a8cc9431bb"
LOCK_SHA256 = "7288b7dd64b533a4e08ae9a66ff120fb69d5885effd80b0e3b7f51455028d976"
LOCK_BLOB = "1a9b126e3d5062bd3be5290bfa5930d189760ceb"
W48V_HEAD = "890ab746618f0a57853db1e38cedb7c1b500a89d"
W48V_RESULT_SHA256 = "19c4eb912fee92e67a8e0071da44582e21aa99f9a0695441538dbe5bf4f78b1c"
W48H_HEAD = "2a3f2ca7f91911351b817d96565753f0dee06f54"
W48H_RUN = "34856005699"
W47Q_HEAD = "8eb8af15af464ae6c49d20de225aa501ec4bedaf"
W47Q_RUN = "34871019291"
PROTOCOL = "wcare49-execution-input-closure-v1"
RESULT_HASH_MARKER = re.compile(r"WCARE49_RESULT_SHA256=([0-9a-f]{64})")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_STEPS = {
    "Checkout exact WCARE-49 harness",
    "Verify one-commit sibling harness and self-tests",
    "Retrieve and verify exact WCARE-47Q admission event",
    "Checkout exact admitted WCARE-48 FINAL",
    "Install exact Rust toolchain for closure execution",
    "Execute bounded WCARE-49 input closure",
}


class EvidenceFailure(Exception):
    pass


class ProtocolFailure(Exception):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise EvidenceFailure(detail)


def load_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ProtocolFailure(f"json_invalid:{path.name}:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ProtocolFailure(f"json_not_object:{path.name}")
    return value, raw


def base(classification: str, detail: str) -> dict:
    return {
        "authority": AUTHORITY,
        "classification": classification,
        "detail": detail,
        "wcare49_run_id": str(RUN_ID),
        "wcare49_run_number": str(RUN_NUMBER),
        "wcare49_run_attempt": str(RUN_ATTEMPT),
        "wcare49_head": RUN_HEAD,
        "final_subject": FINAL,
        "lock_sha256": LOCK_SHA256,
        "lock_git_blob": LOCK_BLOB,
        "run_json_sha256": None,
        "jobs_json_sha256": None,
        "job_log_sha256": None,
        "wcare49_result_sha256": None,
        "wcare47_result_sha256": None,
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


def extract_json_payloads(log: str) -> list[dict]:
    rows: list[dict] = []
    for line in log.splitlines():
        brace = line.find("{")
        if brace < 0:
            continue
        try:
            value = json.loads(line[brace:].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def validate_result(result: dict) -> None:
    expected = {
        "authority": AUTHORITY,
        "protocol_version": PROTOCOL,
        "classification": "EXECUTION_INPUTS_CLOSED",
        "detail": "exact_admitted_lock_executed_under_bounded_repository_toolchain_input_envelope",
        "final_subject": FINAL,
        "prepared_parent": PREPARED,
        "wcare48v_head": W48V_HEAD,
        "wcare48v_result_sha256": W48V_RESULT_SHA256,
        "wcare48h_head": W48H_HEAD,
        "wcare48h_run_id": W48H_RUN,
        "wcare47q_head": W47Q_HEAD,
        "wcare47q_run_id": W47Q_RUN,
        "lock_sha256_pre": LOCK_SHA256,
        "lock_sha256_post": LOCK_SHA256,
        "lock_git_blob_pre": LOCK_BLOB,
        "lock_git_blob_post": LOCK_BLOB,
        "lock_format": 4,
        "package_count": 45,
        "cargo_config_search_reached_filesystem_root": True,
        "cargo_config_census": [],
        "cargo_home_config_names_checked": ["config.toml", "config"],
        "dynamic_environment_override_keys": [],
        "cargo_home_outside_repository": True,
        "cargo_home_initially_empty": True,
        "target_dir_outside_repository": True,
        "target_dir_initially_empty": True,
        "metadata_locked_passed": True,
        "tests_locked_passed": True,
        "golden_test_passed": True,
        "source_postflight_unchanged": True,
        "checkout_postflight_clean_including_ignored_untracked": True,
        "lock_admitted": True,
        "execution_inputs_closed": True,
        "full_machine_hermeticity_established": False,
        "trusted_hardware_established": False,
        "builder_authentication_established": False,
        "independent_host_reproducibility_established": False,
        "wcare42_executable_qualification_established": False,
        "runtime_authority_granted": False,
    }
    for key, expected_value in expected.items():
        require(result.get(key) == expected_value, f"closure_result_mismatch:{key}:{result.get(key)!r}")

    w47_hash = result.get("wcare47_result_sha256")
    require(isinstance(w47_hash, str) and HEX64.fullmatch(w47_hash) is not None, "wcare47_result_hash_invalid")
    require(isinstance(result.get("admission_precondition_file_sha256"), str) and HEX64.fullmatch(result["admission_precondition_file_sha256"]) is not None, "admission_precondition_hash_invalid")

    rustc = result.get("rustc_version")
    cargo = result.get("cargo_version")
    require(isinstance(rustc, str) and rustc.startswith("rustc 1.96.0 "), "rustc_identity_mismatch")
    require(isinstance(cargo, str) and cargo.startswith("cargo 1.96.0 "), "cargo_identity_mismatch")
    require(isinstance(result.get("rustc_verbose"), str) and "release: 1.96.0" in result["rustc_verbose"], "rustc_verbose_mismatch")
    require(isinstance(result.get("cargo_verbose"), str) and result["cargo_verbose"].splitlines()[0].startswith("cargo 1.96.0 "), "cargo_verbose_mismatch")

    env = result.get("environment_census")
    require(isinstance(env, dict), "environment_census_missing")
    require(all(value in (None, "") for value in env.values()), "environment_census_not_empty")

    paths = result.get("cargo_config_paths_checked")
    require(isinstance(paths, list) and len(paths) >= 6, "cargo_config_path_census_too_small")
    repo_paths = [item.get("path") for item in paths if isinstance(item, dict) and item.get("scope") == "repository"]
    require(repo_paths == [
        "tools/wcare42_builder_attestation_verifier/.cargo/config.toml",
        "tools/wcare42_builder_attestation_verifier/.cargo/config",
        "tools/.cargo/config.toml",
        "tools/.cargo/config",
        ".cargo/config.toml",
        ".cargo/config",
    ], f"repository_cargo_config_census_mismatch:{repo_paths!r}")
    for item in paths:
        require(isinstance(item, dict), "cargo_config_descriptor_not_object")
        if item.get("scope") == "ambient_parent":
            require(isinstance(item.get("ancestor_depth"), int) and item["ancestor_depth"] >= 1, "ambient_depth_invalid")
            require(item.get("name") in {".cargo/config.toml", ".cargo/config"}, "ambient_name_invalid")
            locator = item.get("path_locator_sha256")
            require(isinstance(locator, str) and HEX64.fullmatch(locator) is not None, "ambient_locator_invalid")
        else:
            require(item.get("scope") == "repository", "cargo_config_scope_invalid")

    commands = result.get("commands")
    require(isinstance(commands, list) and len(commands) == 3, "command_census_mismatch")
    expected_argv = [
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        ["cargo", "test", "--locked"],
        ["cargo", "test", "--locked", "--test", "golden"],
    ]
    for command, argv in zip(commands, expected_argv):
        require(isinstance(command, dict) and command.get("argv") == argv, f"command_argv_mismatch:{command!r}")
        require(command.get("returncode") == 0, "command_nonzero")
        require(isinstance(command.get("stdout_sha256"), str) and HEX64.fullmatch(command["stdout_sha256"]) is not None, "command_stdout_hash_invalid")
        require(isinstance(command.get("stderr_sha256"), str) and HEX64.fullmatch(command["stderr_sha256"]) is not None, "command_stderr_hash_invalid")


def extract_result(log: str) -> tuple[dict, str]:
    markers = RESULT_HASH_MARKER.findall(log)
    require(len(markers) == 1, f"wcare49_result_hash_marker_count:{len(markers)}")
    marker = markers[0]
    unique: dict[str, dict] = {}
    for value in extract_json_payloads(log):
        if value.get("protocol_version") != PROTOCOL:
            continue
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if sha256((canonical + "\n").encode("utf-8")) == marker:
            unique.setdefault(canonical, value)
    require(len(unique) == 1, f"wcare49_unique_result_matching_hash_count:{len(unique)}")
    result = next(iter(unique.values()))
    validate_result(result)
    require("PASS_WCARE49_PREREGISTERED_HARNESS" in log, "wcare49_harness_pass_marker_missing")
    require("PASS_WCARE49_EXACT_ADMISSION_PRECONDITION" in log, "wcare49_admission_pass_marker_missing")
    require("PASS_WCARE49_EXECUTION_INPUTS_CLOSED" in log, "wcare49_closure_pass_marker_missing")
    return result, marker


def verify(run_path: Path, jobs_path: Path | None, log_path: Path | None) -> tuple[dict, int]:
    run, run_raw = load_json(run_path)
    validate_run_identity(run)
    status = run.get("status")
    if status != "completed":
        require(status in {"queued", "in_progress", "pending", "requested", "waiting"}, f"unexpected_run_status:{status}")
        out = base("WCARE49_PRECONDITION_PENDING", "exact_wcare49_run_not_completed")
        out["run_json_sha256"] = sha256(run_raw)
        return out, 3

    require(run.get("conclusion") == "success", f"wcare49_run_conclusion:{run.get('conclusion')}")
    if jobs_path is None or log_path is None:
        raise ProtocolFailure("completed_run_requires_jobs_and_log")

    jobs, jobs_raw = load_json(jobs_path)
    job_list = jobs.get("jobs")
    require(isinstance(job_list, list), "jobs_list_missing")
    matching = [job for job in job_list if isinstance(job, dict) and job.get("name") == "gate"]
    require(len(matching) == 1, f"gate_job_count:{len(matching)}")
    job = matching[0]
    require(job.get("status") == "completed" and job.get("conclusion") == "success", "gate_job_not_successful")
    steps = job.get("steps")
    require(isinstance(steps, list), "gate_steps_missing")
    by_name = {step.get("name"): step for step in steps if isinstance(step, dict)}
    for name in REQUIRED_STEPS:
        require(name in by_name, f"required_step_missing:{name}")
        require(by_name[name].get("status") == "completed" and by_name[name].get("conclusion") == "success", f"required_step_not_successful:{name}")

    log_raw = log_path.read_bytes()
    try:
        log = log_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolFailure("wcare49_log_not_utf8") from exc
    result, marker = extract_result(log)

    out = base("WCARE49_PRECONDITION_SATISFIED", "exact_hosted_wcare49_execution_input_closure_verified")
    out["run_json_sha256"] = sha256(run_raw)
    out["jobs_json_sha256"] = sha256(jobs_raw)
    out["job_log_sha256"] = sha256(log_raw)
    out["wcare49_result_sha256"] = marker
    out["wcare47_result_sha256"] = result["wcare47_result_sha256"]
    out["execution_inputs_closed"] = True
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
        value, code = verify(args.run_json, args.jobs_json, args.log)
        return emit(value, code)
    except EvidenceFailure as exc:
        return emit(base("WCARE49_PRECONDITION_FAILED", str(exc)), 1)
    except (ProtocolFailure, OSError) as exc:
        return emit(base("INVALID_WCARE49_PRECONDITION_PROTOCOL", str(exc)), 4)


if __name__ == "__main__":
    raise SystemExit(main())
