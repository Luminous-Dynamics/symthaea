#!/usr/bin/env python3
"""Fresh no-Nix verifier for an isolated Workbench -version observation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import verify_workbench_invocation_isolation_profile as isolation_verifier
import verify_workbench_root_nar_capture as nar_verifier

SCHEMA = "symthaea-workbench-isolated-version-invocation-receipt-v1"
VERIFICATION_SCHEMA = "symthaea-workbench-isolated-version-invocation-verification-v1"
STATUS_SUCCESS = "observed-isolated-version-invocation-unqualified"
VERSION_LINE_RE = re.compile(r"(?m)^Version:[ \t]+([0-9]+\.[0-9]+\.[0-9]+)[ \t]*\r?$")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}\Z")
ROOT_ROLES = {
    "cwd", "home", "xdg_config_home", "xdg_cache_home", "xdg_data_home",
    "xdg_state_home", "xdg_runtime_dir", "xdg_config_dirs", "xdg_data_dirs", "tmpdir",
}
TOP_KEYS = {
    "schema", "status", "closure_capture_digest", "root", "executable", "implementations",
    "profiles", "command", "environment", "isolation", "diagnostics", "stdout", "stderr",
    "version_output", "authority", "capture_digest",
}
EXECUTABLE_KEYS = {"path", "pre_sha256", "post_sha256"}
IMPLEMENTATION_KEYS = {"producer_sha256", "isolation_verifier_sha256"}
PROFILE_KEYS = {"execution_capsule_sha256", "invocation_isolation_sha256"}
COMMAND_KEYS = {"argv", "exit_code", "cwd", "stdin", "close_fds", "pass_fds", "umask_octal"}
ENV_KEYS = {"stage", "inherit_host_environment", "entries", "entry_environment_sha256"}
ISOLATION_KEYS = {"roots", "cleanup_confirmed"}
LIFECYCLE_KEYS = {"path", "directory_mode_octal", "empty_before", "reuse_allowed", "cleanup_confirmed"}
STREAM_KEYS = {"path", "byte_length", "sha256"}
VERSION_KEYS = {"aggregation", "byte_length", "sha256", "parsed_version"}
DIAGNOSTIC_KEYS = {
    "cpu_vendor", "cpu_family", "cpu_model", "cpu_stepping", "cpu_flags_digest", "kernel_release"
}
RECEIPT_AUTHORITY_KEYS = {
    "program_membership_verified", "invocation_profile_verified", "invocation_executed",
    "version_output_bound", "same_host_repeatability_established", "path_equivalence_established",
    "cross_cpu_equivalence_established", "workbench_execution_qualified",
    "scientific_execution_qualified", "transform_executed", "atlas_correctness_established",
    "fmq010_established", "neural_alignment_established", "consciousness_evidence",
}


class VerificationError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise VerificationError(f"{label}: closed-world schema mismatch")
    return value


def strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise VerificationError(f"{label}: boolean required")
    return value


def strict_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise VerificationError(f"{label}: integer required")
    return value


def sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise VerificationError(f"{label}: canonical sha256:<64 lowercase hex> required")
    return value


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise VerificationError(f"JSON object: duplicate key: {key}")
        out[key] = value
    return out


def strict_json_bytes(data: bytes, label: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError(f"{label}: UTF-8 required") from exc
    try:
        return json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"{label}: invalid JSON") from exc


def safe_file(root_dir: Path, rel: str) -> Path:
    pure = PurePosixPath(rel)
    if pure.is_absolute() or not pure.parts or any(part in {".", ".."} for part in pure.parts):
        raise VerificationError("sidecar path: canonical relative path required")
    root = root_dir.resolve(strict=True)
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise VerificationError(f"sidecar missing: {rel}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise VerificationError(f"sidecar escapes capture root: {rel}") from exc
    if candidate.is_symlink() or not resolved.is_file() or not stat.S_ISREG(resolved.stat().st_mode):
        raise VerificationError(f"sidecar must be regular non-symlink file: {rel}")
    return resolved


def verify_inventory(root_dir: Path) -> None:
    root = root_dir.resolve(strict=True)
    expected = {"receipt.json", "raw/wb-version.stdout", "raw/wb-version.stderr"}
    actual: set[str] = set()
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise VerificationError(f"capture inventory: symlink forbidden: {rel}")
        if path.is_file():
            actual.add(rel)
    if actual != expected:
        raise VerificationError(
            f"capture inventory mismatch; missing={sorted(expected-actual)} extra={sorted(actual-expected)}"
        )


def verify_stream(root: Path, value: Any, rel: str, label: str) -> bytes:
    stream = exact(value, STREAM_KEYS, label)
    if stream["path"] != rel:
        raise VerificationError(f"{label}: fixed sidecar path mismatch")
    length = strict_int(stream["byte_length"], f"{label} byte_length")
    if length < 0:
        raise VerificationError(f"{label}: non-negative byte length required")
    expected_digest = sha256(stream["sha256"], f"{label} sha256")
    path = safe_file(root, rel)
    data = path.read_bytes()
    if len(data) != length or digest_bytes(data) != expected_digest:
        raise VerificationError(f"{label}: retained byte mismatch")
    return data


def independently_parse_version(stdout: bytes, stderr: bytes, expected: str) -> str:
    try:
        text = (stdout + stderr).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError("Workbench -version output: UTF-8 required") from exc
    matches = VERSION_LINE_RE.findall(text)
    if matches != [expected]:
        raise VerificationError(f"Workbench -version output: exact Version line for {expected} required")
    return matches[0]


def require_parent_program(parent: Any, parent_verifier_path: Path) -> dict[str, Any]:
    keys = {
        "schema", "status", "root", "root_nar_sha256", "closure_digest", "closure_capture_digest",
        "root_nar_capture_digest", "implementations", "target", "authority",
    }
    parent = exact(parent, keys, "parent root-NAR verification")
    if parent["schema"] != "symthaea-workbench-root-nar-capture-verification-v1":
        raise VerificationError("parent root-NAR verification: schema mismatch")
    if parent["status"] != "verified-root-nar-regular-executable-membership":
        raise VerificationError("parent root-NAR verification: regular executable membership required")
    implementations = exact(
        parent["implementations"],
        {"verifier_sha256", "closure_verifier_sha256", "membership_verifier_sha256", "nar_capture_producer_sha256"},
        "parent root-NAR implementations",
    )
    if implementations["verifier_sha256"] != digest_file(parent_verifier_path):
        raise VerificationError("parent root-NAR verification: verifier source digest mismatch")
    required_authority = {
        "closure_receipt_verified": True,
        "root_nar_capture_verified": True,
        "target_membership_verified": True,
        "root_main_program_regular_executable_verified": True,
        "symlink_resolution_verified": False,
        "workbench_execution_qualified": False,
        "transform_executed": False,
        "fmq010_established": False,
        "neural_alignment_established": False,
        "consciousness_evidence": False,
    }
    if parent["authority"] != required_authority:
        raise VerificationError("parent root-NAR verification: authority mismatch")
    target = exact(
        parent["target"],
        {"node_type", "executable", "content_length", "content_sha256", "symlink_target_hex"},
        "parent target",
    )
    if target["node_type"] != "regular" or target["executable"] is not True:
        raise VerificationError("parent target: executable regular file required")
    if strict_int(target["content_length"], "parent target content_length") < 1:
        raise VerificationError("parent target: positive content length required")
    sha256(target["content_sha256"], "parent target content sha256")
    if target["symlink_target_hex"] is not None:
        raise VerificationError("parent target: regular file cannot carry symlink target")
    return parent


def verify_receipt(
    capture_dir: Path,
    *,
    expected_parent: dict[str, Any],
    execution_profile_path: Path,
    isolation_profile_path: Path,
    isolation_verifier_path: Path,
    producer_path: Path,
) -> dict[str, Any]:
    root_dir = capture_dir.resolve(strict=True)
    receipt_path = safe_file(root_dir, "receipt.json")
    receipt_bytes = receipt_path.read_bytes()
    receipt = exact(strict_json_bytes(receipt_bytes, "version invocation receipt"), TOP_KEYS, "version invocation receipt")
    if receipt_bytes != canonical_json_bytes(receipt) + b"\n":
        raise VerificationError("version invocation receipt: exact canonical JSON bytes required")
    if receipt["schema"] != SCHEMA or receipt["status"] != STATUS_SUCCESS:
        raise VerificationError("version invocation receipt: successful observation required")

    parent_root = expected_parent["root"]
    parent_program_sha = expected_parent["target"]["content_sha256"]
    expected_executable = parent_root + "/bin/wb_command"
    if receipt["root"] != parent_root:
        raise VerificationError("version invocation receipt: root mismatch")
    if receipt["closure_capture_digest"] != expected_parent["closure_capture_digest"]:
        raise VerificationError("version invocation receipt: closure capture digest mismatch")

    executable = exact(receipt["executable"], EXECUTABLE_KEYS, "executable")
    if executable["path"] != expected_executable:
        raise VerificationError("executable: exact root-main-program path required")
    if executable["pre_sha256"] != parent_program_sha or executable["post_sha256"] != parent_program_sha:
        raise VerificationError("executable: before/after bytes differ from independently verified program")

    implementations = exact(receipt["implementations"], IMPLEMENTATION_KEYS, "implementations")
    if implementations["producer_sha256"] != digest_file(producer_path):
        raise VerificationError("receipt: producer source digest mismatch")
    if implementations["isolation_verifier_sha256"] != digest_file(isolation_verifier_path):
        raise VerificationError("receipt: isolation verifier source digest mismatch")

    profiles = exact(receipt["profiles"], PROFILE_KEYS, "profiles")
    if profiles["execution_capsule_sha256"] != digest_file(execution_profile_path):
        raise VerificationError("receipt: execution capsule profile digest mismatch")
    if profiles["invocation_isolation_sha256"] != digest_file(isolation_profile_path):
        raise VerificationError("receipt: invocation isolation profile digest mismatch")

    execution_profile = isolation_verifier.load(execution_profile_path)
    isolation_profile = isolation_verifier.verify_profile(
        isolation_verifier.load(isolation_profile_path),
        execution_profile,
    )
    expected_version = execution_profile["nixpkgs_package"]["version"]

    command = exact(receipt["command"], COMMAND_KEYS, "command")
    if command["argv"] != [expected_executable, "-version"]:
        raise VerificationError("command: exact absolute wb_command -version argv required")
    if strict_int(command["exit_code"], "command exit_code") != 0:
        raise VerificationError("command: zero exit required")
    if command["stdin"] != "devnull":
        raise VerificationError("command: stdin must be /dev/null")
    if strict_bool(command["close_fds"], "command close_fds") is not True or command["pass_fds"] != []:
        raise VerificationError("command: unrelated inherited file descriptors forbidden")
    if command["umask_octal"] != "0077":
        raise VerificationError("command: exact umask required")

    isolation = exact(receipt["isolation"], ISOLATION_KEYS, "isolation")
    if strict_bool(isolation["cleanup_confirmed"], "isolation cleanup_confirmed") is not True:
        raise VerificationError("isolation: cleanup confirmation required")
    roots = exact(isolation["roots"], ROOT_ROLES, "isolation roots")
    root_paths: dict[str, str] = {}
    for role, raw in roots.items():
        lifecycle = exact(raw, LIFECYCLE_KEYS, f"isolation root {role}")
        path = lifecycle["path"]
        if not isinstance(path, str) or not path.startswith("/"):
            raise VerificationError(f"isolation root {role}: absolute path required")
        if lifecycle["directory_mode_octal"] != "0700":
            raise VerificationError(f"isolation root {role}: mode 0700 required")
        if strict_bool(lifecycle["empty_before"], f"isolation root {role} empty_before") is not True:
            raise VerificationError(f"isolation root {role}: empty-before required")
        if strict_bool(lifecycle["reuse_allowed"], f"isolation root {role} reuse_allowed") is not False:
            raise VerificationError(f"isolation root {role}: reuse forbidden")
        if strict_bool(lifecycle["cleanup_confirmed"], f"isolation root {role} cleanup_confirmed") is not True:
            raise VerificationError(f"isolation root {role}: cleanup required")
        root_paths[role] = path
    if len(set(root_paths.values())) != len(root_paths):
        raise VerificationError("isolation roots: roles must have distinct directories")
    if command["cwd"] != root_paths["cwd"]:
        raise VerificationError("command: cwd differs from isolated cwd")

    environment = exact(receipt["environment"], ENV_KEYS, "environment")
    if environment["stage"] != "root-main-program-execve":
        raise VerificationError("environment: exact root-main-program execve stage required")
    if strict_bool(environment["inherit_host_environment"], "environment inherit_host_environment") is not False:
        raise VerificationError("environment: host inheritance forbidden")
    entries = environment["entries"]
    if not isinstance(entries, dict):
        raise VerificationError("environment entries: object required")
    expected_entries = dict(isolation_profile["process_environment"]["fixed"])
    binding_to_role = {
        "invocation.cwd": "cwd",
        "invocation.home": "home",
        "invocation.xdg_config_home": "xdg_config_home",
        "invocation.xdg_cache_home": "xdg_cache_home",
        "invocation.xdg_data_home": "xdg_data_home",
        "invocation.xdg_state_home": "xdg_state_home",
        "invocation.xdg_runtime_dir": "xdg_runtime_dir",
        "invocation.xdg_config_dirs": "xdg_config_dirs",
        "invocation.xdg_data_dirs": "xdg_data_dirs",
        "invocation.tmpdir": "tmpdir",
    }
    for env_name, binding in isolation_profile["process_environment"]["dynamic_bindings"].items():
        expected_entries[env_name] = root_paths[binding_to_role[binding]]
    if entries != expected_entries:
        raise VerificationError("environment: exact fixed/dynamic entry environment mismatch")
    if entries["PWD"] != command["cwd"]:
        raise VerificationError("environment: PWD must equal actual invocation cwd")
    if not (entries["TMPDIR"] == entries["TMP"] == entries["TEMP"] == root_paths["tmpdir"]):
        raise VerificationError("environment: temporary aliases must share isolated temp root")
    if environment["entry_environment_sha256"] != digest_bytes(canonical_json_bytes(entries)):
        raise VerificationError("environment: entry digest mismatch")

    stdout = verify_stream(root_dir, receipt["stdout"], "raw/wb-version.stdout", "stdout")
    stderr = verify_stream(root_dir, receipt["stderr"], "raw/wb-version.stderr", "stderr")
    combined = stdout + stderr
    version_output = exact(receipt["version_output"], VERSION_KEYS, "version output")
    if version_output["aggregation"] != "stdout-then-stderr-v1":
        raise VerificationError("version output: exact historical stdout+stderr aggregation required")
    if strict_int(version_output["byte_length"], "version output byte_length") != len(combined):
        raise VerificationError("version output: byte length mismatch")
    if version_output["sha256"] != digest_bytes(combined):
        raise VerificationError("version output: digest mismatch")
    parsed = independently_parse_version(stdout, stderr, expected_version)
    if version_output["parsed_version"] != parsed:
        raise VerificationError("version output: producer parsed-version mismatch")

    diagnostics = exact(receipt["diagnostics"], DIAGNOSTIC_KEYS, "diagnostics")
    for key in ("cpu_vendor", "cpu_family", "cpu_model", "cpu_stepping", "kernel_release"):
        if not isinstance(diagnostics[key], str) or not diagnostics[key]:
            raise VerificationError(f"diagnostics: non-empty {key} required")
    sha256(diagnostics["cpu_flags_digest"], "diagnostics cpu_flags_digest")

    authority = exact(receipt["authority"], RECEIPT_AUTHORITY_KEYS, "receipt authority")
    for key, value in authority.items():
        if strict_bool(value, f"receipt authority {key}") is not False:
            raise VerificationError(f"receipt authority: producer escalation forbidden: {key}")

    capture_digest = sha256(receipt["capture_digest"], "capture digest")
    expected_capture_digest = digest_bytes(
        canonical_json_bytes({k: v for k, v in receipt.items() if k != "capture_digest"})
    )
    if capture_digest != expected_capture_digest:
        raise VerificationError("receipt: capture digest mismatch")
    verify_inventory(root_dir)
    return receipt


def verify_pipeline(
    closure_receipt_dir: Path,
    nar_capture_dir: Path,
    version_capture_dir: Path,
    execution_profile_path: Path,
    isolation_profile_path: Path,
    lock_path: Path,
    closure_producer_path: Path,
    normalizer_path: Path,
    closure_verifier_path: Path,
    nar_capture_producer_path: Path,
    membership_path: Path,
    parent_nar_verifier_path: Path,
    isolation_verifier_path: Path,
    version_producer_path: Path,
) -> dict[str, Any]:
    if Path(nar_verifier.__file__).resolve() != parent_nar_verifier_path.resolve():
        raise VerificationError("parent root-NAR verifier: imported source path mismatch")
    if Path(isolation_verifier.__file__).resolve() != isolation_verifier_path.resolve():
        raise VerificationError("isolation verifier: imported source path mismatch")

    try:
        parent = nar_verifier.verify_pipeline(
            closure_receipt_dir,
            nar_capture_dir,
            execution_profile_path,
            lock_path,
            closure_producer_path,
            normalizer_path,
            closure_verifier_path,
            nar_capture_producer_path,
            membership_path,
        )
    except Exception as exc:
        raise VerificationError(f"parent root-NAR verification failed: {exc}") from exc
    parent = require_parent_program(parent, parent_nar_verifier_path)

    receipt = verify_receipt(
        version_capture_dir,
        expected_parent=parent,
        execution_profile_path=execution_profile_path,
        isolation_profile_path=isolation_profile_path,
        isolation_verifier_path=isolation_verifier_path,
        producer_path=version_producer_path,
    )
    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "verified-isolated-workbench-version-invocation",
        "root": parent["root"],
        "qualification_platform": "x86_64-linux",
        "closure_digest": parent["closure_digest"],
        "program_content_sha256": parent["target"]["content_sha256"],
        "version": receipt["version_output"]["parsed_version"],
        "version_output_sha256": receipt["version_output"]["sha256"],
        "entry_environment_sha256": receipt["environment"]["entry_environment_sha256"],
        "diagnostics": receipt["diagnostics"],
        "implementations": {
            "verifier_sha256": digest_file(Path(__file__)),
            "producer_sha256": digest_file(version_producer_path),
            "parent_nar_verifier_sha256": digest_file(parent_nar_verifier_path),
            "isolation_verifier_sha256": digest_file(isolation_verifier_path),
        },
        "authority": {
            "program_membership_verified": True,
            "invocation_profile_verified": True,
            "invocation_executed": True,
            "version_output_bound": True,
            "same_host_repeatability_established": False,
            "path_equivalence_established": False,
            "cross_cpu_equivalence_established": False,
            "workbench_execution_qualified": False,
            "scientific_execution_qualified": False,
            "transform_executed": False,
            "atlas_correctness_established": False,
            "fmq010_established": False,
            "neural_alignment_established": False,
            "consciousness_evidence": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-receipt-dir", required=True, type=Path)
    parser.add_argument("--nar-capture-dir", required=True, type=Path)
    parser.add_argument("--version-capture-dir", required=True, type=Path)
    parser.add_argument("--execution-profile", required=True, type=Path)
    parser.add_argument("--isolation-profile", required=True, type=Path)
    parser.add_argument("--flake-lock", required=True, type=Path)
    parser.add_argument("--closure-producer", required=True, type=Path)
    parser.add_argument("--normalizer", required=True, type=Path)
    parser.add_argument("--closure-verifier", required=True, type=Path)
    parser.add_argument("--nar-capture-producer", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--parent-nar-verifier", required=True, type=Path)
    parser.add_argument("--isolation-verifier", required=True, type=Path)
    parser.add_argument("--version-producer", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_pipeline(
            args.closure_receipt_dir,
            args.nar_capture_dir,
            args.version_capture_dir,
            args.execution_profile,
            args.isolation_profile,
            args.flake_lock,
            args.closure_producer,
            args.normalizer,
            args.closure_verifier,
            args.nar_capture_producer,
            args.membership,
            args.parent_nar_verifier,
            args.isolation_verifier,
            args.version_producer,
        )
    except (VerificationError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
