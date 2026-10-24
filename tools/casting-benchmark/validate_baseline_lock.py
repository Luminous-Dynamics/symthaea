#!/usr/bin/env python3
"""Fail-closed structural and execution-readiness validator for casting baseline locks.

This standard-library tool validates recorded evidence. It does not fetch source,
verify external artifact downloads, install environments, or run algorithm smoke tests.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("docs/research/CASTING_EXTERNAL_BASELINE_LOCK_V1.json")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

TOP_KEYS = {
    "schema_version", "protocol_id", "lock_id", "status", "execution_eligible",
    "inspected_on", "candidate_release", "required_method_ids", "method_pins",
    "unresolved_baselines", "environment_lock", "execution_blockers",
    "execution_rules", "references",
}
RELEASE_KEYS = {
    "distribution", "version", "source_repository", "source_tag", "source_commit",
    "release_date", "pypi_project_url", "release_artifacts", "pyproject",
    "python_requires", "declared_runtime_requirement_floors", "requirement_warning",
}
ARTIFACT_KEYS = {"sdist", "wheel"}
ARTIFACT_ENTRY_KEYS = {"filename", "sha256"}
PYPROJECT_KEYS = {"path", "git_blob_sha1"}
METHOD_KEYS = {
    "baseline_id", "track", "package", "package_version", "source_tag",
    "source_commit", "source_path", "source_blob_sha1", "symbol", "pin_status",
    "execution_eligible",
}
UNRESOLVED_KEYS = {"baseline_id", "status", "blocker"}
ENVIRONMENT_KEYS = {
    "status", "lockfile_path", "lockfile_sha256", "python_version",
    "framework_versions", "platform", "device", "dependency_artifact_hashes",
    "execution_smoke_test", "note",
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def loads_no_duplicate_keys(raw: str) -> Any:
    """Parse JSON while refusing ambiguous duplicate keys at every object depth."""
    return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _exact_keys(value: Any, expected: set[str], label: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{label} must be a JSON object")
        return False
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        errors.append(f"{label} missing required keys: {', '.join(missing)}")
    if unknown:
        errors.append(f"{label} has unknown keys: {', '.join(unknown)}")
    return not missing and not unknown


def _check_string(value: Any, label: str, errors: list[str]) -> None:
    if not _is_nonempty_string(value):
        errors.append(f"{label} must be a non-empty string")


def _check_hash(value: Any, pattern: re.Pattern[str], label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        expected_length = 40 if pattern.pattern == SHA1_RE.pattern else 64
        errors.append(f"{label} must be a lowercase hexadecimal digest of length {expected_length}")


def validate_lock(document: Any, for_execution: bool = False) -> list[str]:
    """Return all structural errors; execution mode additionally requires full immutable locks."""
    errors: list[str] = []
    if not _exact_keys(document, TOP_KEYS, "manifest", errors):
        return errors

    if document["schema_version"] != "1.0.0":
        errors.append("schema_version must be 1.0.0")
    if document["protocol_id"] != "CASTING-MO-BO-001":
        errors.append("protocol_id must be CASTING-MO-BO-001")
    _check_string(document["lock_id"], "lock_id", errors)
    if not isinstance(document["inspected_on"], str) or DATE_RE.fullmatch(document["inspected_on"]) is None:
        errors.append("inspected_on must use YYYY-MM-DD")
    if not isinstance(document["execution_eligible"], bool):
        errors.append("execution_eligible must be a boolean")
    if not isinstance(document["status"], str) or document["status"] not in {
        "source_pins_recorded_environment_unlocked", "locked_for_execution"
    }:
        errors.append("status must be source_pins_recorded_environment_unlocked or locked_for_execution")

    release = document["candidate_release"]
    if _exact_keys(release, RELEASE_KEYS, "candidate_release", errors):
        for key in ("distribution", "version", "source_repository", "source_tag",
                    "pypi_project_url", "python_requires", "requirement_warning"):
            _check_string(release[key], f"candidate_release.{key}", errors)
        _check_hash(release["source_commit"], SHA1_RE, "candidate_release.source_commit", errors)
        if not isinstance(release["release_date"], str) or DATE_RE.fullmatch(release["release_date"]) is None:
            errors.append("candidate_release.release_date must use YYYY-MM-DD")
        artifacts = release["release_artifacts"]
        if _exact_keys(artifacts, ARTIFACT_KEYS, "candidate_release.release_artifacts", errors):
            for artifact_name, entry in artifacts.items():
                if _exact_keys(entry, ARTIFACT_ENTRY_KEYS, f"release_artifacts.{artifact_name}", errors):
                    _check_string(entry["filename"], f"release_artifacts.{artifact_name}.filename", errors)
                    _check_hash(entry["sha256"], SHA256_RE, f"release_artifacts.{artifact_name}.sha256", errors)
        pyproject = release["pyproject"]
        if _exact_keys(pyproject, PYPROJECT_KEYS, "candidate_release.pyproject", errors):
            _check_string(pyproject["path"], "candidate_release.pyproject.path", errors)
            _check_hash(pyproject["git_blob_sha1"], SHA1_RE, "candidate_release.pyproject.git_blob_sha1", errors)
        floors = release["declared_runtime_requirement_floors"]
        if not isinstance(floors, dict) or not floors:
            errors.append("candidate_release.declared_runtime_requirement_floors must be a non-empty object")
        elif any(not _is_nonempty_string(k) or not _is_nonempty_string(v) for k, v in floors.items()):
            errors.append("runtime requirement floors must map non-empty names to non-empty descriptions")

    required = document["required_method_ids"]
    if not isinstance(required, list) or not required or any(not _is_nonempty_string(v) for v in required):
        errors.append("required_method_ids must be a non-empty list of non-empty strings")
        required_ids: set[str] = set()
    else:
        required_ids = set(required)
        if len(required_ids) != len(required):
            errors.append("required_method_ids must be unique")

    pins = document["method_pins"]
    pin_ids: list[str] = []
    if not isinstance(pins, list):
        errors.append("method_pins must be a list")
        pins = []
    for index, pin in enumerate(pins):
        label = f"method_pins[{index}]"
        if not _exact_keys(pin, METHOD_KEYS, label, errors):
            continue
        for key in ("baseline_id", "track", "package", "package_version", "source_tag",
                    "source_path", "symbol", "pin_status"):
            _check_string(pin[key], f"{label}.{key}", errors)
        _check_hash(pin["source_commit"], SHA1_RE, f"{label}.source_commit", errors)
        _check_hash(pin["source_blob_sha1"], SHA1_RE, f"{label}.source_blob_sha1", errors)
        if not isinstance(pin["pin_status"], str) or pin["pin_status"] not in {
            "source_pinned_environment_unlocked", "locked_for_execution"
        }:
            errors.append(f"{label}.pin_status is not a supported status")
        if pin["track"] not in {"single_fidelity", "multi_fidelity", "non_bayesian", "repository_baseline", "test_fixture"}:
            errors.append(f"{label}.track is not a supported track")
        if not isinstance(pin["execution_eligible"], bool):
            errors.append(f"{label}.execution_eligible must be a boolean")
        if pin["pin_status"] == "source_pinned_environment_unlocked" and pin["execution_eligible"] is not False:
            errors.append(f"{label} cannot be execution-eligible while its environment is unlocked")
        if pin["pin_status"] == "locked_for_execution" and pin["execution_eligible"] is not True:
            errors.append(f"{label} marked locked_for_execution must be execution-eligible")
        release = document["candidate_release"]
        if isinstance(release, dict) and pin["package"] == release.get("distribution"):
            for pin_key, release_key in (
                ("package_version", "version"),
                ("source_tag", "source_tag"),
                ("source_commit", "source_commit"),
            ):
                if pin[pin_key] != release.get(release_key):
                    errors.append(f"{label}.{pin_key} must match candidate_release.{release_key}")
        pin_ids.append(pin["baseline_id"])

    if len(set(pin_ids)) != len(pin_ids):
        errors.append("method_pins baseline_id values must be unique")

    unresolved = document["unresolved_baselines"]
    unresolved_ids: list[str] = []
    if not isinstance(unresolved, list):
        errors.append("unresolved_baselines must be a list")
        unresolved = []
    for index, item in enumerate(unresolved):
        label = f"unresolved_baselines[{index}]"
        if not _exact_keys(item, UNRESOLVED_KEYS, label, errors):
            continue
        _check_string(item["baseline_id"], f"{label}.baseline_id", errors)
        _check_string(item["blocker"], f"{label}.blocker", errors)
        if item["status"] != "not_pinned":
            errors.append(f"{label}.status must be not_pinned")
        unresolved_ids.append(item["baseline_id"])

    if len(set(unresolved_ids)) != len(unresolved_ids):
        errors.append("unresolved_baselines baseline_id values must be unique")
    if set(pin_ids) & set(unresolved_ids):
        errors.append("a baseline cannot be both pinned and unresolved")
    if set(pin_ids) | set(unresolved_ids) != required_ids:
        missing = sorted(required_ids - set(pin_ids) - set(unresolved_ids))
        extra = sorted((set(pin_ids) | set(unresolved_ids)) - required_ids)
        errors.append(f"baseline coverage mismatch; missing={missing}, extra={extra}")

    env = document["environment_lock"]
    if _exact_keys(env, ENVIRONMENT_KEYS, "environment_lock", errors):
        if not isinstance(env["status"], str) or env["status"] not in {"not_locked", "locked"}:
            errors.append("environment_lock.status must be not_locked or locked")
        if env["status"] == "not_locked":
            if env["lockfile_path"] is not None or env["lockfile_sha256"] is not None:
                errors.append("unlocked environment must not claim lockfile_path or lockfile_sha256")
        else:
            _check_string(env["lockfile_path"], "environment_lock.lockfile_path", errors)
            _check_hash(env["lockfile_sha256"], SHA256_RE, "environment_lock.lockfile_sha256", errors)
            _check_string(env["python_version"], "environment_lock.python_version", errors)
            _check_string(env["platform"], "environment_lock.platform", errors)
            _check_string(env["device"], "environment_lock.device", errors)
            if not isinstance(env["framework_versions"], dict) or not env["framework_versions"]:
                errors.append("locked environment requires non-empty framework_versions")
            elif any(not _is_nonempty_string(k) or not _is_nonempty_string(v) for k, v in env["framework_versions"].items()):
                errors.append("framework_versions must contain exact non-empty versions")
            if not isinstance(env["dependency_artifact_hashes"], dict) or not env["dependency_artifact_hashes"]:
                errors.append("locked environment requires dependency_artifact_hashes")
            else:
                for package_name, digest in env["dependency_artifact_hashes"].items():
                    if not _is_nonempty_string(package_name) or not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                        errors.append("dependency_artifact_hashes must map package names to SHA-256 digests")
                        break
            if env["execution_smoke_test"] != "passed":
                errors.append("locked environment requires execution_smoke_test=passed")
        _check_string(env["note"], "environment_lock.note", errors)

    blockers = document["execution_blockers"]
    if not isinstance(blockers, list) or any(not _is_nonempty_string(v) for v in blockers):
        errors.append("execution_blockers must be a list of non-empty strings")

    for key in ("execution_rules", "references"):
        if not isinstance(document[key], list) or any(not _is_nonempty_string(v) for v in document[key]):
            errors.append(f"{key} must be a list of non-empty strings")

    logically_ready = (
        not unresolved
        and isinstance(env, dict)
        and env.get("status") == "locked"
        and env.get("execution_smoke_test") == "passed"
        and len(pin_ids) == len(required_ids)
        and set(pin_ids) == required_ids
        and all(isinstance(pin, dict) and pin.get("pin_status") == "locked_for_execution"
                and pin.get("execution_eligible") is True for pin in pins)
        and blockers == []
    )
    if document["execution_eligible"] is True:
        if not logically_ready:
            errors.append("execution_eligible=true contradicts unresolved pins, environment, smoke-test or blocker state")
        if document["status"] != "locked_for_execution":
            errors.append("execution_eligible=true requires status=locked_for_execution")
    elif document["status"] == "locked_for_execution":
        errors.append("status=locked_for_execution requires execution_eligible=true")

    if for_execution and not logically_ready:
        errors.append("execution gate blocked: all methods, environment digest and smoke tests must be locked")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--for-execution", action="store_true",
        help="fail unless all required methods and runtime environment are immutably locked and smoke-tested",
    )
    args = parser.parse_args(argv)
    try:
        document = loads_no_duplicate_keys(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"BASELINE_LOCK_INVALID: {exc}", file=sys.stderr)
        return 2

    errors = validate_lock(document, for_execution=args.for_execution)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 3 if args.for_execution else 2

    if document["execution_eligible"]:
        print("BASELINE_LOCK_READY_FOR_EXECUTION")
    else:
        print("BASELINE_LOCK_STRUCTURE_VALID_BUT_EXECUTION_BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
