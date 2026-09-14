#!/usr/bin/env python3
"""Execute the frozen WCARE-42 qualifier over the frozen WCARE-50 fixture matrix.

MeasurementOnly. Synthetic fixture acceptance qualifies the verifier execution
path; it does not authenticate any real builder or establish subject correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

AUTHORITY = "MeasurementOnly"
FINAL = "5bc23735f545b1b82044820f0e02ece59be04b4a"
PREPARED = "52d1d9fb741250ab8bcab205113689a8cc9431bb"
W49_HEAD = "720c7ebf6c03b5698c48a4d2315ea9c4d25107e7"
W49_GUARD_BLOB = "8170c65551eba6fa8d1356229a78667e6c634f4a"
W42_PROTOCOL = "wcare42-builder-attestation-verifier-v1"
PACKAGE_SHA256 = "c6b66a25b5a04649306d81a4275019885edddd84d90bd4510a522441c3a8a9d9"
LOCK_SHA256 = "7288b7dd64b533a4e08ae9a66ff120fb69d5885effd80b0e3b7f51455028d976"
LOCK_BLOB = "1a9b126e3d5062bd3be5290bfa5930d189760ceb"
QUALIFIER = "scripts/wcare42-qualify.sh"
MANIFEST = "tools/wcare42_builder_attestation_verifier/Cargo.toml"
VERIFIER_DIR = "tools/wcare42_builder_attestation_verifier"

EXTRA_FROZEN_BLOBS = {
    "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_VERIFIER_PROTOCOL_V1.md": "1cd6e3f5f2f4edcf4fccf45b35c4c68bdb9c12a5",
    "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_RESULT_SCHEMA_V1.json": "b2d6b4b61af46c8924cdb95b2f958df3d3d7ab96",
    "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_GOLDEN_VECTOR_V1.json": "dc42f404537795f37a9bf178d3f30fd52c74a355",
    QUALIFIER: "8a80d1a74e409503a72b4607425cc61d8072ca2e",
}
EXPECTED_RESULT_KEYS = {
    "authority",
    "verifier_protocol_version",
    "wcare41_protocol_version",
    "disposition",
    "detail",
    "envelope_sha256",
    "issuer_trust_policy_sha256",
    "wcare40_plan_sha256",
    "wcare40_result_sha256",
    "subject_receipt_sha256",
    "canonical_message_sha256",
    "issuer_key_id",
    "subject_kind",
    "result_bound_by_signature",
    "checks",
    "attestation_accepted",
    "builder_authentication_established",
    "preregistration_temporal_precedence_established",
    "subject_correctness_established",
    "runtime_authority_granted",
}
EXPECTED_CHECK_KEYS = {
    "signature_valid",
    "subject_binding_valid",
    "canonicalization_valid",
    "issuer_key_present",
    "key_valid_at_issue_time",
    "revocation_policy_satisfied",
    "attestation_current_at_evaluation",
    "scope_authorized",
    "claimed_strength_authorized",
    "issuer_trusted_for_claim",
}


class QualificationFailure(Exception):
    pass


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise QualificationFailure(detail)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_guard(path: Path):
    spec = importlib.util.spec_from_file_location("wcare49_exact_input_guard", path)
    require(spec is not None and spec.loader is not None, "wcare49_guard_import_spec_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_text(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(proc.returncode == 0, f"git_failed:{' '.join(args)}:{proc.stderr.strip()}")
    return proc.stdout.strip()


def verify_extra_blobs(root: Path, phase: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for path, expected in EXTRA_FROZEN_BLOBS.items():
        actual = git_text(root, "rev-parse", f"HEAD:{path}")
        require(actual == expected, f"{phase}_extra_blob_mismatch:{path}:{actual}")
        values[path] = actual
    return values


def load_w49_precondition(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise QualificationFailure(f"wcare49_precondition_invalid_json:{type(exc).__name__}") from exc
    expected = {
        "authority": AUTHORITY,
        "classification": "WCARE49_PRECONDITION_SATISFIED",
        "detail": "exact_hosted_wcare49_execution_input_closure_verified",
        "wcare49_run_id": "34877774593",
        "wcare49_run_number": "7",
        "wcare49_run_attempt": "1",
        "wcare49_head": W49_HEAD,
        "final_subject": FINAL,
        "lock_sha256": LOCK_SHA256,
        "lock_git_blob": LOCK_BLOB,
        "execution_inputs_closed": True,
        "wcare42_executable_qualification_established": False,
        "runtime_authority_granted": False,
    }
    require(isinstance(value, dict), "wcare49_precondition_not_object")
    for key, expected_value in expected.items():
        require(value.get(key) == expected_value, f"wcare49_precondition_mismatch:{key}:{value.get(key)!r}")
    for key in ("run_json_sha256", "jobs_json_sha256", "job_log_sha256", "wcare49_result_sha256", "wcare47_result_sha256"):
        item = value.get(key)
        require(isinstance(item, str) and len(item) == 64 and all(ch in "0123456789abcdef" for ch in item), f"wcare49_precondition_hash_invalid:{key}")
    return value, sha256(raw)


def load_package(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    require(sha256(raw) == PACKAGE_SHA256, "fixture_package_sha256_mismatch")
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise QualificationFailure(f"fixture_package_invalid_json:{type(exc).__name__}") from exc
    require(isinstance(value, dict), "fixture_package_not_object")
    require(value.get("package_version") == "wcare50-wcare42-executable-qualification-fixtures-v1", "fixture_package_version_mismatch")
    require(value.get("synthetic_only") is True, "fixture_package_synthetic_boundary_missing")
    require(value.get("authority") == AUTHORITY, "fixture_package_authority_mismatch")
    cases = value.get("cases")
    require(isinstance(cases, list) and len(cases) == 11, "fixture_case_count_mismatch")
    return value, sha256(raw)


def verify_materialized_case(case: dict, case_dir: Path) -> None:
    files = case.get("files")
    digests = case.get("file_sha256")
    require(isinstance(files, dict) and isinstance(digests, dict), f"fixture_files_missing:{case.get('name')}")
    for name, expected in digests.items():
        path = case_dir / name
        require(path.is_file(), f"fixture_materialized_file_missing:{case['name']}:{name}")
        raw = path.read_bytes()
        require(sha256(raw) == expected, f"fixture_materialized_digest_mismatch:{case['name']}:{name}")
        require(raw.decode("utf-8") == files[name], f"fixture_materialized_bytes_mismatch:{case['name']}:{name}")


def extract_verifier_result(stdout: bytes) -> dict:
    rows: list[dict] = []
    for line in stdout.decode("utf-8", errors="strict").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("verifier_protocol_version") == W42_PROTOCOL:
            rows.append(value)
    require(len(rows) == 1, f"wcare42_result_row_count:{len(rows)}")
    return rows[0]


def validate_case_result(case: dict, case_dir: Path, result: dict, returncode: int) -> dict:
    name = case["name"]
    require(returncode == case["expected_exit_code"], f"case_exit_code_mismatch:{name}:{returncode}")
    require(set(result) == EXPECTED_RESULT_KEYS, f"case_result_fields_mismatch:{name}")
    require(result.get("authority") == AUTHORITY, f"case_authority_mismatch:{name}")
    require(result.get("disposition") == case["expected_disposition"], f"case_disposition_mismatch:{name}:{result.get('disposition')}")
    require(result.get("result_bound_by_signature") == case["expected_result_bound_by_signature"], f"case_result_binding_mismatch:{name}")
    require(result.get("attestation_accepted") is (case["expected_disposition"] == "ATTESTATION_ACCEPTED"), f"case_accepted_flag_mismatch:{name}")
    require(result.get("builder_authentication_established") is False, f"case_builder_auth_promotion:{name}")
    require(result.get("preregistration_temporal_precedence_established") is False, f"case_temporal_promotion:{name}")
    require(result.get("subject_correctness_established") is False, f"case_subject_correctness_promotion:{name}")
    require(result.get("runtime_authority_granted") is False, f"case_runtime_authority_promotion:{name}")
    checks = result.get("checks")
    require(isinstance(checks, dict) and set(checks) == EXPECTED_CHECK_KEYS, f"case_check_fields_mismatch:{name}")
    require(checks == case["expected_checks"], f"case_checks_mismatch:{name}:{checks!r}")

    digest_map = {
        "envelope_sha256": "envelope.json",
        "issuer_trust_policy_sha256": "policy.json",
        "wcare40_plan_sha256": "plan.json",
        "wcare40_result_sha256": "result.json",
        "subject_receipt_sha256": "subject.json",
    }
    for result_key, file_name in digest_map.items():
        expected = case["file_sha256"][file_name]
        require(result.get(result_key) == expected, f"case_input_digest_mismatch:{name}:{result_key}")
    require(isinstance(result.get("canonical_message_sha256"), str) or result.get("canonical_message_sha256") is None, f"case_canonical_hash_type:{name}")

    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    return {
        "name": name,
        "expected_disposition": case["expected_disposition"],
        "exit_code": returncode,
        "result_sha256": sha256(canonical),
        "canonical_message_sha256": result.get("canonical_message_sha256"),
        "result_bound_by_signature": result.get("result_bound_by_signature"),
        "attestation_accepted": result.get("attestation_accepted"),
    }


def run_case(root: Path, case: dict, case_dir: Path) -> dict:
    argv = [
        "bash",
        QUALIFIER,
        str((case_dir / "envelope.json").resolve()),
        str((case_dir / "policy.json").resolve()),
        str((case_dir / "plan.json").resolve()),
        str((case_dir / "result.json").resolve()),
        str((case_dir / "subject.json").resolve()),
        case["evaluation_utc"],
    ]
    proc = subprocess.run(argv, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    result = extract_verifier_result(proc.stdout)
    summary = validate_case_result(case, case_dir, result, proc.returncode)
    summary["stdout_sha256"] = sha256(proc.stdout)
    summary["stderr_sha256"] = sha256(proc.stderr)
    return summary


def execute(candidate: Path, guard_path: Path, precondition_path: Path, package_path: Path, fixture_root: Path) -> dict:
    candidate = candidate.resolve()
    guard_path = guard_path.resolve()
    fixture_root = fixture_root.resolve()
    require(git_text(candidate, "rev-parse", "HEAD") == FINAL, "candidate_head_mismatch")
    require(git_text(candidate, "rev-parse", "HEAD^") == PREPARED, "candidate_parent_mismatch")

    guard_root = guard_path.parents[1]
    require(git_text(guard_root, "rev-parse", "HEAD") == W49_HEAD, "wcare49_guard_checkout_head_mismatch")
    require(git_text(guard_root, "rev-parse", f"HEAD:{guard_path.relative_to(guard_root).as_posix()}") == W49_GUARD_BLOB, "wcare49_guard_blob_mismatch")
    guard = load_guard(guard_path)

    w49, w49_file_sha = load_w49_precondition(precondition_path)
    package, package_file_sha = load_package(package_path)

    guard.assert_checkout_clean(candidate, "wcare50_preflight")
    pre_blobs = guard.verify_frozen_blobs(candidate, "wcare50_preflight")
    extra_pre = verify_extra_blobs(candidate, "wcare50_preflight")
    package_count = guard.validate_lock(candidate)
    config_paths, config_found = guard.assert_empty_config_census(candidate)
    require(config_found == [], "wcare50_preflight_cargo_config_not_empty")
    env_fixed, env_dynamic = guard.environment_census()
    cargo_home, cargo_home_empty = guard.assert_external_empty_dir(candidate, "CARGO_HOME")
    target_dir, target_empty = guard.assert_external_empty_dir(candidate, "CARGO_TARGET_DIR")
    cargo_home_names = guard.assert_cargo_home_configs_absent(cargo_home, "wcare50_preflight")

    verifier_dir = candidate / VERIFIER_DIR
    rustc = guard.tool_output(verifier_dir, ["rustc", "--version"])
    cargo = guard.tool_output(verifier_dir, ["cargo", "--version"])
    require(rustc.startswith("rustc 1.96.0 "), "wcare50_rustc_version_mismatch")
    require(cargo.startswith("cargo 1.96.0 "), "wcare50_cargo_version_mismatch")

    baseline = [
        guard.run_command(verifier_dir, ["cargo", "test", "--locked"]),
        guard.run_command(verifier_dir, ["cargo", "test", "--locked", "--test", "golden"]),
    ]

    case_results: list[dict] = []
    for case in package["cases"]:
        case_dir = fixture_root / case["name"]
        verify_materialized_case(case, case_dir)
        case_results.append(run_case(candidate, case, case_dir))

    guard.assert_checkout_clean(candidate, "wcare50_postflight")
    post_blobs = guard.verify_frozen_blobs(candidate, "wcare50_postflight")
    extra_post = verify_extra_blobs(candidate, "wcare50_postflight")
    require(post_blobs == pre_blobs and extra_post == extra_pre, "wcare50_source_blob_map_changed")
    config_paths_post, config_found_post = guard.assert_empty_config_census(candidate)
    require(config_paths_post == config_paths and config_found_post == config_found, "wcare50_cargo_config_census_changed")
    require(guard.assert_cargo_home_configs_absent(cargo_home, "wcare50_postflight") == cargo_home_names, "wcare50_cargo_home_config_changed")
    env_fixed_post, env_dynamic_post = guard.environment_census()
    require(env_fixed_post == env_fixed and env_dynamic_post == env_dynamic, "wcare50_environment_census_changed")
    require(sha256((candidate / "tools/wcare42_builder_attestation_verifier/Cargo.lock").read_bytes()) == LOCK_SHA256, "wcare50_lock_sha_changed")
    require(git_text(candidate, "rev-parse", "HEAD:tools/wcare42_builder_attestation_verifier/Cargo.lock") == LOCK_BLOB, "wcare50_lock_blob_changed")

    accepted = [row["name"] for row in case_results if row["attestation_accepted"]]
    require(accepted == ["accepted_provenance", "accepted_relation", "accepted_pre_result"], f"unexpected_fixture_acceptance_set:{accepted!r}")

    return {
        "authority": AUTHORITY,
        "protocol_version": "wcare50-wcare42-executable-qualification-v1",
        "classification": "WCARE42_EXECUTABLE_QUALIFIED",
        "detail": "frozen_wcare42_qualifier_passed_preregistered_synthetic_disposition_matrix",
        "final_subject": FINAL,
        "prepared_parent": PREPARED,
        "wcare49_head": W49_HEAD,
        "wcare49_result_sha256": w49["wcare49_result_sha256"],
        "wcare47_result_sha256": w49["wcare47_result_sha256"],
        "wcare49_precondition_file_sha256": w49_file_sha,
        "wcare49_guard_blob": W49_GUARD_BLOB,
        "fixture_package_sha256": package_file_sha,
        "fixture_case_count": len(case_results),
        "fixture_accepted_case_count": len(accepted),
        "fixture_accepted_cases": accepted,
        "package_count": package_count,
        "cargo_config_search_reached_filesystem_root": True,
        "cargo_config_census": config_found,
        "cargo_home_config_names_checked": cargo_home_names,
        "environment_census": env_fixed,
        "dynamic_environment_override_keys": env_dynamic,
        "cargo_home_outside_repository": True,
        "cargo_home_initially_empty": cargo_home_empty,
        "target_dir_outside_repository": True,
        "target_dir_initially_empty": target_empty,
        "rustc_version": rustc,
        "cargo_version": cargo,
        "baseline_locked_tests": baseline,
        "case_results": case_results,
        "source_postflight_unchanged": True,
        "checkout_postflight_clean_including_ignored_untracked": True,
        "lock_admitted": True,
        "execution_inputs_closed": True,
        "cryptographic_execution_qualified": True,
        "wcare42_executable_qualification_established": True,
        "builder_authentication_established": False,
        "preregistration_temporal_precedence_established": False,
        "subject_correctness_established": False,
        "full_machine_hermeticity_established": False,
        "trusted_hardware_established": False,
        "runtime_authority_granted": False,
    }


def emit(value: dict, code: int) -> int:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--w49-guard", required=True, type=Path)
    parser.add_argument("--w49-precondition-json", required=True, type=Path)
    parser.add_argument("--fixture-package-json", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = execute(
            args.candidate,
            args.w49_guard,
            args.w49_precondition_json,
            args.fixture_package_json,
            args.fixture_root,
        )
        return emit(result, 0)
    except (QualificationFailure, OSError, UnicodeError) as exc:
        return emit({
            "authority": AUTHORITY,
            "protocol_version": "wcare50-wcare42-executable-qualification-v1",
            "classification": "WCARE42_EXECUTABLE_QUALIFICATION_FAILED",
            "detail": str(exc),
            "final_subject": FINAL,
            "cryptographic_execution_qualified": False,
            "wcare42_executable_qualification_established": False,
            "builder_authentication_established": False,
            "preregistration_temporal_precedence_established": False,
            "subject_correctness_established": False,
            "runtime_authority_granted": False,
        }, 1)


if __name__ == "__main__":
    raise SystemExit(main())
