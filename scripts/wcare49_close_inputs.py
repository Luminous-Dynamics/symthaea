#!/usr/bin/env python3
"""WCARE-49 bounded repository/toolchain execution-input closure.

MeasurementOnly. This does not establish full machine hermeticity or WCARE-42
executable qualification. It executes only after exact WCARE-47Q admission has
been independently verified by wcare49_verify_admission_precondition.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib

AUTHORITY = "MeasurementOnly"
FINAL = "5bc23735f545b1b82044820f0e02ece59be04b4a"
PREPARED = "52d1d9fb741250ab8bcab205113689a8cc9431bb"
LOCK = "tools/wcare42_builder_attestation_verifier/Cargo.lock"
MANIFEST = "tools/wcare42_builder_attestation_verifier/Cargo.toml"
VERIFIER_DIR = "tools/wcare42_builder_attestation_verifier"
LOCK_SHA256 = "7288b7dd64b533a4e08ae9a66ff120fb69d5885effd80b0e3b7f51455028d976"
LOCK_BLOB = "1a9b126e3d5062bd3be5290bfa5930d189760ceb"
WCARE48V_HEAD = "890ab746618f0a57853db1e38cedb7c1b500a89d"
WCARE48V_RESULT_SHA256 = "19c4eb912fee92e67a8e0071da44582e21aa99f9a0695441538dbe5bf4f78b1c"
WCARE48H_HEAD = "2a3f2ca7f91911351b817d96565753f0dee06f54"
WCARE48H_RUN_ID = "34856005699"
WCARE47Q_HEAD = "8eb8af15af464ae6c49d20de225aa501ec4bedaf"
WCARE47Q_RUN_ID = "34871019291"

FROZEN_BLOBS = {
    MANIFEST: "5410040e5616241dd4ba581af8f297675d083830",
    "tools/wcare42_builder_attestation_verifier/src/main.rs": "1c300a455f054d118e55556aac81b623824629bc",
    "tools/wcare42_builder_attestation_verifier/tests/golden.rs": "666242f74fb302f9be2b62fa4b3050f3ee9ffefd",
    "rust-toolchain.toml": "4f0430eac96d545bcfaa0df23ce475faf4aee96a",
    LOCK: LOCK_BLOB,
}

FIXED_OVERRIDE_KEYS = (
    "RUSTC",
    "RUSTDOC",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "RUSTC_BOOTSTRAP",
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "RUSTDOCFLAGS",
    "CARGO_BUILD_RUSTC",
    "CARGO_BUILD_RUSTC_WRAPPER",
    "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
    "CARGO_BUILD_RUSTFLAGS",
    "CARGO_BUILD_TARGET",
)
DYNAMIC_OVERRIDE_PATTERNS = (
    re.compile(r"^CARGO_TARGET_.*_(?:RUSTFLAGS|LINKER|RUNNER)$"),
    re.compile(r"^CARGO_PROFILE_"),
    re.compile(r"^CARGO_SOURCE_"),
    re.compile(r"^CARGO_REGISTRIES_"),
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CARGO_CONFIG_NAMES = ("config.toml", "config")


class ClosureFailure(Exception):
    pass


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ClosureFailure(detail)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def locator_sha256(path: Path) -> str:
    return sha256_bytes(os.fsencode(str(path)))


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    proc = subprocess.run(
        ["git", *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if check and proc.returncode != 0:
        raise ClosureFailure(
            f"git_failed:{' '.join(args)}:{proc.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return proc


def git_text(root: Path, *args: str) -> str:
    return git(root, *args).stdout.decode("utf-8", errors="strict").strip()


def blob(root: Path, path: str) -> str:
    return git_text(root, "rev-parse", f"HEAD:{path}")


def assert_checkout_clean(root: Path, phase: str) -> None:
    require(not git_text(root, "status", "--porcelain=v1", "--untracked-files=all"), f"{phase}_status_not_clean")
    require(git(root, "diff", "--quiet").returncode == 0, f"{phase}_tracked_diff")
    require(git(root, "diff", "--cached", "--quiet").returncode == 0, f"{phase}_cached_diff")
    ordinary = git_text(root, "ls-files", "--others", "--exclude-standard")
    ignored = git_text(root, "ls-files", "--others", "--ignored", "--exclude-standard")
    require(not ordinary, f"{phase}_untracked_files:{ordinary}")
    require(not ignored, f"{phase}_ignored_untracked_files:{ignored}")


def cargo_config_candidates(root: Path, *, test_ceiling: Path | None = None) -> list[tuple[Path, int]]:
    """Enumerate Cargo hierarchical config candidates from verifier cwd upward.

    Production callers omit test_ceiling, which means walking all the way to the
    filesystem root. A ceiling exists only so hermetic self-tests do not inspect
    the machine running the test.
    """
    root = root.resolve()
    current = (root / VERIFIER_DIR).resolve()
    ceiling = test_ceiling.resolve() if test_ceiling is not None else None
    if ceiling is not None:
        require(current == ceiling or ceiling in current.parents, "test_ceiling_not_ancestor")

    candidates: list[tuple[Path, int]] = []
    ambient_depth = 0
    while True:
        if current == root:
            depth = 0
        elif root in current.parents:
            depth = 0
        else:
            ambient_depth += 1
            depth = ambient_depth
        for name in CARGO_CONFIG_NAMES:
            candidates.append((current / ".cargo" / name, depth))

        if ceiling is not None and current == ceiling:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return candidates


def config_descriptor(root: Path, path: Path, ambient_depth: int) -> dict:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return {
            "scope": "ambient_parent",
            "ancestor_depth": ambient_depth,
            "name": f".cargo/{path.name}",
            "path_locator_sha256": locator_sha256(path),
        }
    return {"scope": "repository", "path": rel.as_posix()}


def assert_empty_config_census(
    root: Path, *, test_ceiling: Path | None = None
) -> tuple[list[dict], list[dict]]:
    root = root.resolve()
    checked: list[dict] = []
    found: list[dict] = []
    for path, ambient_depth in cargo_config_candidates(root, test_ceiling=test_ceiling):
        descriptor = config_descriptor(root, path, ambient_depth)
        checked.append(descriptor)
        if path.exists() or path.is_symlink():
            item = dict(descriptor)
            item["symlink"] = path.is_symlink()
            if item["scope"] == "repository":
                rel = item["path"]
                item["tracked"] = (
                    git(root, "ls-files", "--error-unmatch", "--", rel, check=False).returncode == 0
                )
            found.append(item)
    require(not found, f"cargo_config_census_not_empty:{json.dumps(found, sort_keys=True, separators=(',', ':'))}")
    return checked, found


def environment_census() -> tuple[dict[str, str | None], list[str]]:
    fixed = {key: os.environ.get(key) for key in FIXED_OVERRIDE_KEYS}
    nonempty_fixed = sorted(key for key, value in fixed.items() if value not in (None, ""))
    dynamic = sorted(
        key
        for key, value in os.environ.items()
        if value and any(pattern.match(key) for pattern in DYNAMIC_OVERRIDE_PATTERNS)
    )
    require(not nonempty_fixed, f"semantic_environment_override:{nonempty_fixed}")
    require(not dynamic, f"dynamic_cargo_environment_override:{dynamic}")
    return fixed, dynamic


def assert_external_empty_dir(root: Path, env_key: str) -> tuple[Path, bool]:
    raw = os.environ.get(env_key)
    require(bool(raw), f"{env_key.lower()}_missing")
    path = Path(raw).resolve()
    require(path.is_absolute(), f"{env_key.lower()}_not_absolute")
    require(path != root and root not in path.parents, f"{env_key.lower()}_inside_repository")
    if path.exists():
        require(path.is_dir(), f"{env_key.lower()}_not_directory")
        initially_empty = not any(path.iterdir())
    else:
        initially_empty = True
    require(initially_empty, f"{env_key.lower()}_not_empty")
    path.mkdir(parents=True, exist_ok=True)
    return path, initially_empty


def assert_cargo_home_configs_absent(cargo_home: Path, phase: str) -> list[str]:
    checked: list[str] = []
    found: list[str] = []
    for name in CARGO_CONFIG_NAMES:
        checked.append(name)
        path = cargo_home / name
        if path.exists() or path.is_symlink():
            found.append(name)
    require(not found, f"{phase}_cargo_home_config_present:{found}")
    return checked


def validate_admission(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ClosureFailure(f"admission_json_invalid:{type(exc).__name__}") from exc
    require(isinstance(value, dict), "admission_json_not_object")
    expected = {
        "authority": AUTHORITY,
        "classification": "ADMISSION_PRECONDITION_SATISFIED",
        "detail": "exact_hosted_wcare47q_lock_admission_verified",
        "wcare47q_run_id": WCARE47Q_RUN_ID,
        "wcare47q_run_number": "4",
        "wcare47q_run_attempt": "1",
        "wcare47q_head": WCARE47Q_HEAD,
        "final_subject": FINAL,
        "lock_sha256": LOCK_SHA256,
        "lock_git_blob": LOCK_BLOB,
        "lock_admitted": True,
        "execution_inputs_closed": False,
        "wcare42_executable_qualification_established": False,
        "runtime_authority_granted": False,
    }
    for key, expected_value in expected.items():
        require(value.get(key) == expected_value, f"admission_precondition_mismatch:{key}:{value.get(key)!r}")
    result_hash = value.get("wcare47_result_sha256")
    require(isinstance(result_hash, str) and HEX64.fullmatch(result_hash) is not None, "admission_result_hash_invalid")
    return value, sha256_bytes(raw)


def validate_lock(root: Path) -> int:
    lock_path = root / LOCK
    raw = lock_path.read_bytes()
    require(sha256_bytes(raw) == LOCK_SHA256, "working_lock_sha256_mismatch")
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ClosureFailure(f"lock_parse_failed:{type(exc).__name__}") from exc
    require(data.get("version") == 4, "lock_format_not_v4")
    packages = data.get("package")
    require(isinstance(packages, list) and len(packages) == 45, "lock_package_count_mismatch")
    roots = 0
    for package in packages:
        require(isinstance(package, dict), "lock_package_not_object")
        name = package.get("name")
        source = package.get("source")
        checksum = package.get("checksum")
        if name == "wcare42-builder-attestation-verifier" and source is None:
            roots += 1
            continue
        require(isinstance(source, str) and source.startswith("registry+"), f"non_registry_dependency:{name}")
        require(isinstance(checksum, str) and HEX64.fullmatch(checksum) is not None, f"invalid_dependency_checksum:{name}")
    require(roots == 1, f"standalone_root_count:{roots}")
    return len(packages)


def run_command(cwd: Path, argv: list[str]) -> dict:
    proc = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")[-4000:]
        raise ClosureFailure(f"command_failed:{' '.join(argv)}:{proc.returncode}:{stderr}")
    return {
        "argv": argv,
        "stdout_sha256": sha256_bytes(proc.stdout),
        "stderr_sha256": sha256_bytes(proc.stderr),
        "returncode": proc.returncode,
    }


def tool_output(cwd: Path, argv: list[str]) -> str:
    proc = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(proc.returncode == 0, f"tool_identity_failed:{' '.join(argv)}")
    return proc.stdout.strip()


def verify_frozen_blobs(root: Path, phase: str) -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in FROZEN_BLOBS.items():
        actual = blob(root, path)
        require(actual == expected, f"{phase}_blob_mismatch:{path}:{actual}")
        observed[path] = actual
    return observed


def close(root: Path, admission_path: Path) -> dict:
    root = root.resolve()
    require(git_text(root, "rev-parse", "HEAD") == FINAL, "final_head_mismatch")
    require(git_text(root, "rev-parse", "HEAD^") == PREPARED, "final_parent_mismatch")
    assert_checkout_clean(root, "preflight")
    pre_blobs = verify_frozen_blobs(root, "preflight")
    package_count = validate_lock(root)
    checked_configs, found_configs = assert_empty_config_census(root)
    env_fixed, env_dynamic = environment_census()
    cargo_home, cargo_home_empty = assert_external_empty_dir(root, "CARGO_HOME")
    target_dir, target_empty = assert_external_empty_dir(root, "CARGO_TARGET_DIR")
    cargo_home_configs = assert_cargo_home_configs_absent(cargo_home, "preflight")
    admission, admission_file_sha = validate_admission(admission_path)

    verifier_dir = root / VERIFIER_DIR
    rustc_version = tool_output(verifier_dir, ["rustc", "--version"])
    cargo_version = tool_output(verifier_dir, ["cargo", "--version"])
    rustc_verbose = tool_output(verifier_dir, ["rustc", "-Vv"])
    cargo_verbose = tool_output(verifier_dir, ["cargo", "-Vv"])
    require(rustc_version.startswith("rustc 1.96.0 "), "unexpected_rustc_version")
    require(cargo_version.startswith("cargo 1.96.0 "), "unexpected_cargo_version")
    require("release: 1.96.0" in rustc_verbose, "unexpected_rustc_verbose")
    require(cargo_verbose.splitlines()[0].startswith("cargo 1.96.0 "), "unexpected_cargo_verbose")

    lock_pre_sha = sha256_bytes((root / LOCK).read_bytes())
    lock_pre_blob = blob(root, LOCK)
    commands = [
        run_command(verifier_dir, ["cargo", "metadata", "--locked", "--format-version", "1"]),
        run_command(verifier_dir, ["cargo", "test", "--locked"]),
        run_command(verifier_dir, ["cargo", "test", "--locked", "--test", "golden"]),
    ]

    assert_checkout_clean(root, "postflight")
    post_blobs = verify_frozen_blobs(root, "postflight")
    require(post_blobs == pre_blobs, "source_blob_map_changed")
    checked_post, found_post = assert_empty_config_census(root)
    require(checked_post == checked_configs and found_post == found_configs, "cargo_config_census_changed")
    require(assert_cargo_home_configs_absent(cargo_home, "postflight") == cargo_home_configs, "cargo_home_config_census_changed")
    env_fixed_post, env_dynamic_post = environment_census()
    require(env_fixed_post == env_fixed and env_dynamic_post == env_dynamic, "environment_census_changed")
    lock_post_sha = sha256_bytes((root / LOCK).read_bytes())
    lock_post_blob = blob(root, LOCK)
    require(lock_post_sha == lock_pre_sha == LOCK_SHA256, "lock_sha_postflight_mismatch")
    require(lock_post_blob == lock_pre_blob == LOCK_BLOB, "lock_blob_postflight_mismatch")

    return {
        "authority": AUTHORITY,
        "protocol_version": "wcare49-execution-input-closure-v1",
        "classification": "EXECUTION_INPUTS_CLOSED",
        "detail": "exact_admitted_lock_executed_under_bounded_repository_toolchain_input_envelope",
        "final_subject": FINAL,
        "prepared_parent": PREPARED,
        "wcare48v_head": WCARE48V_HEAD,
        "wcare48v_result_sha256": WCARE48V_RESULT_SHA256,
        "wcare48h_head": WCARE48H_HEAD,
        "wcare48h_run_id": WCARE48H_RUN_ID,
        "wcare47q_head": WCARE47Q_HEAD,
        "wcare47q_run_id": WCARE47Q_RUN_ID,
        "wcare47_result_sha256": admission["wcare47_result_sha256"],
        "admission_precondition_file_sha256": admission_file_sha,
        "lock_sha256_pre": lock_pre_sha,
        "lock_sha256_post": lock_post_sha,
        "lock_git_blob_pre": lock_pre_blob,
        "lock_git_blob_post": lock_post_blob,
        "lock_format": 4,
        "package_count": package_count,
        "cargo_config_search_reached_filesystem_root": True,
        "cargo_config_paths_checked": checked_configs,
        "cargo_config_census": found_configs,
        "cargo_home_config_names_checked": cargo_home_configs,
        "environment_census": env_fixed,
        "dynamic_environment_override_keys": env_dynamic,
        "cargo_home_outside_repository": True,
        "cargo_home_initially_empty": cargo_home_empty,
        "cargo_home_path_sha256": locator_sha256(cargo_home),
        "target_dir_outside_repository": True,
        "target_dir_initially_empty": target_empty,
        "target_dir_path_sha256": locator_sha256(target_dir),
        "rustc_version": rustc_version,
        "cargo_version": cargo_version,
        "rustc_verbose": rustc_verbose,
        "cargo_verbose": cargo_verbose,
        "commands": commands,
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


def emit(value: dict, code: int) -> int:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission-json", required=True, type=Path)
    args = parser.parse_args()

    root_proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if root_proc.returncode != 0:
        return emit(
            {
                "authority": AUTHORITY,
                "classification": "INVALID_EXECUTION_INPUT_PROTOCOL",
                "detail": "not_in_git_worktree",
                "execution_inputs_closed": False,
                "runtime_authority_granted": False,
            },
            4,
        )
    root = Path(root_proc.stdout.strip()).resolve()

    try:
        return emit(close(root, args.admission_json.resolve()), 0)
    except (ClosureFailure, OSError, UnicodeError) as exc:
        return emit(
            {
                "authority": AUTHORITY,
                "protocol_version": "wcare49-execution-input-closure-v1",
                "classification": "EXECUTION_INPUTS_NOT_CLOSED",
                "detail": str(exc),
                "final_subject": FINAL,
                "lock_admitted": True,
                "execution_inputs_closed": False,
                "full_machine_hermeticity_established": False,
                "trusted_hardware_established": False,
                "builder_authentication_established": False,
                "independent_host_reproducibility_established": False,
                "wcare42_executable_qualification_established": False,
                "runtime_authority_granted": False,
            },
            1,
        )


if __name__ == "__main__":
    raise SystemExit(main())
