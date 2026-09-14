#!/usr/bin/env python3
"""WCARE-48V independent FINAL-child structure verifier.

MeasurementOnly. This verifier lives on a sibling lineage so its contract can
predate the generated WCARE-48 FINAL child.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

AUTHORITY = "MeasurementOnly"
PROTOCOL = "wcare48-observed-lock-generation-v1"
PREPARED = "52d1d9fb741250ab8bcab205113689a8cc9431bb"
WCARE47 = "0321a22986e5c6bf1a6b099d30dc75ccc33d9deb"
REPOSITORY = "Luminous-Dynamics/symthaea"
BRANCH = "wcare-48-observed-lock-generation"
WORKFLOW_REF = (
    "Luminous-Dynamics/symthaea/.github/workflows/"
    "wcare48-lock-generation.yml@refs/heads/wcare-48-observed-lock-generation"
)

LOCK = "tools/wcare42_builder_attestation_verifier/Cargo.lock"
RECEIPT = "docs/release/evidence/WCARE48_LOCK_GENERATION_RECEIPT_V1.json"
GEN_SCRIPT = "scripts/wcare48-generate-lock.sh"
GEN_WORKFLOW = ".github/workflows/wcare48-lock-generation.yml"
GEN_PROTOCOL = "docs/release/evidence/WCARE48_LOCK_GENERATION_PROTOCOL_V1.md"

FROZEN_BLOBS = {
    "tools/wcare42_builder_attestation_verifier/Cargo.toml":
        "5410040e5616241dd4ba581af8f297675d083830",
    "tools/wcare42_builder_attestation_verifier/src/main.rs":
        "1c300a455f054d118e55556aac81b623824629bc",
    "tools/wcare42_builder_attestation_verifier/tests/golden.rs":
        "666242f74fb302f9be2b62fa4b3050f3ee9ffefd",
    "rust-toolchain.toml":
        "4f0430eac96d545bcfaa0df23ce475faf4aee96a",
}

EXPECTED_RECEIPT_KEYS = {
    "authority", "protocol_version", "repository", "branch", "prepared_head",
    "prepared_parent", "prepared_tree", "github_run_id", "github_run_number",
    "github_run_attempt", "github_workflow_ref", "runner_os", "runner_arch",
    "runner_name", "image_os", "image_version", "rustc_version",
    "cargo_version", "rustc_verbose", "cargo_verbose",
    "generation_script_blob", "workflow_blob", "protocol_blob",
    "manifest_blob", "verifier_source_blob", "golden_test_blob",
    "rust_toolchain_blob", "lock_sha256", "lock_git_blob_candidate",
    "lock_format", "package_count", "registry_checksum_policy_satisfied",
    "metadata_locked_passed", "tests_locked_passed",
    "same_runner_repeat_generation_match", "repeat_lock_sha256",
    "source_postflight_unchanged", "github_host_observed_generation_lineage",
    "external_generation_provenance_established",
    "builder_authentication_established", "external_preregistration_established",
    "independent_host_reproducibility_established",
    "wcare42_executable_qualification_established",
    "runtime_authority_granted", "recorded_at_utc",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DIGITS = re.compile(r"^[0-9]+$")


class InvalidFinal(Exception):
    pass


class InvalidVerifier(Exception):
    pass


def run(root: Path, *argv: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def decode(data: bytes) -> str:
    return data.decode("utf-8", errors="strict").strip()


def git_text(root: Path, *argv: str) -> str:
    proc = run(root, "git", *argv)
    if proc.returncode != 0:
        raise InvalidVerifier(
            f"git_failed:{' '.join(argv)}:"
            f"{proc.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return decode(proc.stdout)


def resolve_commit(root: Path, revision: str) -> str | None:
    proc = run(root, "git", "rev-parse", "--verify", f"{revision}^{{commit}}")
    if proc.returncode != 0:
        return None
    value = decode(proc.stdout)
    return value if HEX40.fullmatch(value) else None


def blob_at(root: Path, revision: str, path: str) -> str | None:
    proc = run(root, "git", "rev-parse", f"{revision}:{path}")
    if proc.returncode != 0:
        return None
    value = decode(proc.stdout)
    return value if HEX40.fullmatch(value) else None


def bytes_at(root: Path, revision: str, path: str) -> bytes:
    proc = run(root, "git", "show", f"{revision}:{path}")
    if proc.returncode != 0:
        raise InvalidFinal(f"missing_path:{path}")
    return proc.stdout


def strict_json(data: bytes) -> dict:
    def no_dupes(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise InvalidFinal(f"duplicate_receipt_key:{key}")
            out[key] = value
        return out

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=no_dupes)
    except InvalidFinal:
        raise
    except Exception as exc:
        raise InvalidFinal(f"receipt_json_invalid:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise InvalidFinal("receipt_not_object")
    if set(value) != EXPECTED_RECEIPT_KEYS:
        missing = sorted(EXPECTED_RECEIPT_KEYS - set(value))
        extra = sorted(set(value) - EXPECTED_RECEIPT_KEYS)
        raise InvalidFinal(f"receipt_keyset_mismatch:missing={missing}:extra={extra}")
    return value


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise InvalidFinal(detail)


def validate_lock(lock_bytes: bytes, receipt: dict) -> None:
    try:
        data = tomllib.loads(lock_bytes.decode("utf-8"))
    except Exception as exc:
        raise InvalidFinal(f"lock_parse_failed:{type(exc).__name__}") from exc

    require(data.get("version") == 4, "lock_format_not_v4")
    require(receipt["lock_format"] == 4, "receipt_lock_format_not_v4")
    packages = data.get("package")
    require(isinstance(packages, list) and bool(packages), "lock_package_census_missing")
    require(receipt["package_count"] == len(packages), "package_count_mismatch")

    roots = 0
    for package in packages:
        require(isinstance(package, dict), "non_object_lock_package")
        name = package.get("name")
        source = package.get("source")
        checksum = package.get("checksum")
        if name == "wcare42-builder-attestation-verifier" and source is None:
            roots += 1
            continue
        require(
            isinstance(source, str) and source.startswith("registry+"),
            f"non_registry_dependency:{name}",
        )
        require(
            isinstance(checksum, str) and HEX64.fullmatch(checksum) is not None,
            f"invalid_registry_checksum:{name}",
        )
    require(roots == 1, f"standalone_root_count:{roots}")


def target_missing() -> dict:
    return {
        "authority": AUTHORITY,
        "classification": "TARGET_MISSING",
        "detail": "target_commit_missing",
        "prepared_head": PREPARED,
        "target": None,
        "final_child_structurally_valid": False,
        "lock_admitted": False,
        "wcare42_executable_qualification_established": False,
        "runtime_authority_granted": False,
    }


def verify(root: Path, target_revision: str) -> dict:
    prepared = resolve_commit(root, PREPARED)
    if prepared != PREPARED:
        raise InvalidVerifier("prepared_commit_missing_or_drifted")
    if resolve_commit(root, f"{PREPARED}^") != WCARE47:
        raise InvalidVerifier("prepared_parent_drift")

    for path, expected in FROZEN_BLOBS.items():
        actual = blob_at(root, PREPARED, path)
        if actual != expected:
            raise InvalidVerifier(f"prepared_blob_drift:{path}:{actual}")

    target = resolve_commit(root, target_revision)
    if target is None or target == PREPARED:
        return target_missing()

    parent = resolve_commit(root, f"{target}^")
    require(parent == PREPARED, f"final_parent_mismatch:{parent}")

    diff = git_text(
        root, "diff-tree", "--no-commit-id", "--name-status", "-r", PREPARED, target
    )
    rows = [line.split("\t") for line in diff.splitlines() if line]
    require(len(rows) == 2, f"final_change_count:{len(rows)}")
    require(all(len(row) == 2 for row in rows), "unexpected_rename_or_copy_status")
    require(all(row[0] == "A" for row in rows), f"final_paths_not_additions:{rows}")
    require(
        {row[1] for row in rows} == {LOCK, RECEIPT},
        f"final_path_set_mismatch:{rows}",
    )

    receipt = strict_json(bytes_at(root, target, RECEIPT))
    lock_bytes = bytes_at(root, target, LOCK)

    fixed = {
        "authority": AUTHORITY,
        "protocol_version": PROTOCOL,
        "repository": REPOSITORY,
        "branch": BRANCH,
        "prepared_head": PREPARED,
        "prepared_parent": WCARE47,
        "prepared_tree": git_text(root, "rev-parse", f"{PREPARED}^{{tree}}"),
        "github_workflow_ref": WORKFLOW_REF,
        "generation_script_blob": blob_at(root, PREPARED, GEN_SCRIPT),
        "workflow_blob": blob_at(root, PREPARED, GEN_WORKFLOW),
        "protocol_blob": blob_at(root, PREPARED, GEN_PROTOCOL),
        "manifest_blob": FROZEN_BLOBS[
            "tools/wcare42_builder_attestation_verifier/Cargo.toml"
        ],
        "verifier_source_blob": FROZEN_BLOBS[
            "tools/wcare42_builder_attestation_verifier/src/main.rs"
        ],
        "golden_test_blob": FROZEN_BLOBS[
            "tools/wcare42_builder_attestation_verifier/tests/golden.rs"
        ],
        "rust_toolchain_blob": FROZEN_BLOBS["rust-toolchain.toml"],
    }
    for key, expected in fixed.items():
        require(receipt[key] == expected, f"receipt_binding_mismatch:{key}")

    for key in ("github_run_id", "github_run_number", "github_run_attempt"):
        require(
            isinstance(receipt[key], str) and DIGITS.fullmatch(receipt[key]) is not None,
            f"invalid_numeric_receipt_field:{key}",
        )
    require(
        isinstance(receipt["rustc_version"], str)
        and receipt["rustc_version"].startswith("rustc 1.96.0 "),
        "unexpected_receipt_rustc_version",
    )
    require(
        isinstance(receipt["cargo_version"], str)
        and receipt["cargo_version"].startswith("cargo 1.96.0 "),
        "unexpected_receipt_cargo_version",
    )
    require(
        isinstance(receipt["rustc_verbose"], str)
        and "release: 1.96.0" in receipt["rustc_verbose"],
        "unexpected_receipt_rustc_verbose",
    )
    require(
        isinstance(receipt["cargo_verbose"], str)
        and receipt["cargo_verbose"].splitlines()[0].startswith("cargo 1.96.0 "),
        "unexpected_receipt_cargo_verbose",
    )
    for key in ("runner_os", "runner_arch", "runner_name"):
        require(
            isinstance(receipt[key], str) and bool(receipt[key]),
            f"missing_runner_identity:{key}",
        )
    for key in ("image_os", "image_version"):
        require(
            receipt[key] is None or isinstance(receipt[key], str),
            f"invalid_image_field:{key}",
        )

    lock_blob = blob_at(root, target, LOCK)
    require(lock_blob is not None, "committed_lock_blob_missing")
    require(receipt["lock_git_blob_candidate"] == lock_blob, "lock_git_blob_mismatch")
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    require(receipt["lock_sha256"] == lock_sha, "lock_sha256_mismatch")
    require(receipt["repeat_lock_sha256"] == lock_sha, "repeat_lock_sha256_mismatch")
    validate_lock(lock_bytes, receipt)

    for key in (
        "registry_checksum_policy_satisfied",
        "metadata_locked_passed",
        "tests_locked_passed",
        "same_runner_repeat_generation_match",
        "source_postflight_unchanged",
        "github_host_observed_generation_lineage",
    ):
        require(receipt[key] is True, f"required_true_field_false:{key}")
    for key in (
        "external_generation_provenance_established",
        "builder_authentication_established",
        "external_preregistration_established",
        "independent_host_reproducibility_established",
        "wcare42_executable_qualification_established",
        "runtime_authority_granted",
    ):
        require(receipt[key] is False, f"forbidden_promotion:{key}")

    try:
        timestamp = datetime.fromisoformat(receipt["recorded_at_utc"])
    except Exception as exc:
        raise InvalidFinal("invalid_recorded_at_utc") from exc
    require(timestamp.tzinfo is not None, "recorded_at_utc_not_timezone_aware")

    for path, expected in FROZEN_BLOBS.items():
        require(blob_at(root, target, path) == expected, f"final_frozen_blob_drift:{path}")
    for path in (GEN_SCRIPT, GEN_WORKFLOW, GEN_PROTOCOL):
        require(
            blob_at(root, target, path) == blob_at(root, PREPARED, path),
            f"final_generation_contract_drift:{path}",
        )

    message = git_text(root, "show", "-s", "--format=%B", target)
    prepared_lines = [
        line for line in message.splitlines() if line.startswith("Prepared-Head: ")
    ]
    run_lines = [
        line for line in message.splitlines() if line.startswith("GitHub-Run-Id: ")
    ]
    require(
        prepared_lines == [f"Prepared-Head: {PREPARED}"],
        "prepared_trailer_mismatch",
    )
    require(
        run_lines == [f"GitHub-Run-Id: {receipt['github_run_id']}"],
        "github_run_trailer_mismatch",
    )

    identity = git_text(root, "show", "-s", "--format=%an <%ae>|%cn <%ce>", target)
    expected_identity = (
        "github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>|"
        "github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>"
    )
    require(identity == expected_identity, f"commit_identity_mismatch:{identity}")

    return {
        "authority": AUTHORITY,
        "classification": "FINAL_CHILD_VALID",
        "detail": "exact_final_child_and_receipt_consistent",
        "prepared_head": PREPARED,
        "target": target,
        "github_run_id": receipt["github_run_id"],
        "lock_git_blob": lock_blob,
        "lock_sha256": lock_sha,
        "package_count": receipt["package_count"],
        "final_child_structurally_valid": True,
        "lock_admitted": False,
        "wcare42_executable_qualification_established": False,
        "runtime_authority_granted": False,
    }


def emit(result: dict, code: int) -> int:
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return code


def main() -> int:
    root_proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if root_proc.returncode != 0:
        return emit({
            "authority": AUTHORITY,
            "classification": "INVALID_VERIFIER",
            "detail": "not_in_git_worktree",
            "prepared_head": PREPARED,
            "target": None,
            "final_child_structurally_valid": False,
            "lock_admitted": False,
            "wcare42_executable_qualification_established": False,
            "runtime_authority_granted": False,
        }, 4)

    root = Path(root_proc.stdout.decode("utf-8").strip())
    target_revision = (
        sys.argv[1] if len(sys.argv) == 2 else "wcare-48-observed-lock-generation"
    )

    try:
        result = verify(root, target_revision)
        if result["classification"] == "TARGET_MISSING":
            return emit(result, 3)
        return emit(result, 0)
    except InvalidFinal as exc:
        return emit({
            "authority": AUTHORITY,
            "classification": "FINAL_CHILD_INVALID",
            "detail": str(exc),
            "prepared_head": PREPARED,
            "target": resolve_commit(root, target_revision),
            "final_child_structurally_valid": False,
            "lock_admitted": False,
            "wcare42_executable_qualification_established": False,
            "runtime_authority_granted": False,
        }, 1)
    except (InvalidVerifier, UnicodeError) as exc:
        return emit({
            "authority": AUTHORITY,
            "classification": "INVALID_VERIFIER",
            "detail": str(exc),
            "prepared_head": PREPARED,
            "target": None,
            "final_child_structurally_valid": False,
            "lock_admitted": False,
            "wcare42_executable_qualification_established": False,
            "runtime_authority_granted": False,
        }, 4)


if __name__ == "__main__":
    raise SystemExit(main())
