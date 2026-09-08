#!/usr/bin/env python3
"""Capture a Lineage-B candidate run without minting Workbench identity locally."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

from hcpmmp_neuromaps_common import (
    ContractError,
    REQUIRED_INPUT_ROLES,
    RUN_SCHEMA,
    canonical_json_bytes,
    digest_bytes,
    digest_file,
    exact,
    load_method,
    load_run,
    nonempty,
    sha256,
)

CAPTURE_PROFILE = "symthaea-hcpmmp1-lineage-b-verified-run-capture-v2"
PROFILE_SCHEMA = "symthaea-hcpmmp1-lineage-b-verified-run-capture-profile-v2"
VERIFICATION_SCHEMA = "symthaea-workbench-isolated-version-invocation-verification-v1"
VERIFICATION_STATUS = "verified-isolated-workbench-version-invocation"
STORE_PATH_RE = re.compile(r"^/nix/store/[0123456789abcdfghijklmnpqrsvwxyz]{32}-[^/\r\n]+\Z")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}\Z")

PROFILE_KEYS = {
    "schema", "status", "qualification_source", "qualified_workbench",
    "required_verification_authority", "authority",
}
QUALIFICATION_SOURCE_KEYS = {
    "pr", "head_sha", "workflow_run_id", "verification_file_path",
    "verification_file_sha256", "independent_verification_archive_sha256",
    "raw_evidence_archive_sha256",
}
QUALIFIED_WORKBENCH_KEYS = {
    "root", "relative_main_program", "program_content_sha256", "version",
    "version_output_sha256", "closure_digest", "qualification_platform",
}
PROFILE_AUTHORITY_KEYS = {
    "verified_workbench_observation_bound", "run_manifest_captured",
    "local_closure_reverified", "path_equivalence_established",
    "cross_cpu_equivalence_established", "scientific_execution_qualified",
    "transform_executed", "atlas_correctness_established", "fmq010_established",
    "neural_alignment_established", "consciousness_evidence",
}
VERIFICATION_KEYS = {
    "schema", "status", "root", "qualification_platform", "closure_digest",
    "program_content_sha256", "version", "version_output_sha256",
    "entry_environment_sha256", "diagnostics", "implementations", "authority",
}
DIAGNOSTIC_KEYS = {
    "cpu_vendor", "cpu_family", "cpu_model", "cpu_stepping",
    "cpu_flags_digest", "kernel_release",
}
IMPLEMENTATION_KEYS = {
    "verifier_sha256", "producer_sha256", "parent_nar_verifier_sha256",
    "isolation_verifier_sha256",
}
VERIFICATION_AUTHORITY_KEYS = {
    "program_membership_verified", "invocation_profile_verified", "invocation_executed",
    "version_output_bound", "same_host_repeatability_established",
    "path_equivalence_established", "cross_cpu_equivalence_established",
    "workbench_execution_qualified", "scientific_execution_qualified",
    "transform_executed", "atlas_correctness_established", "fmq010_established",
    "neural_alignment_established", "consciousness_evidence",
}
CAPTURE_AUTHORITY = {
    "qualified_workbench_verification_bound": True,
    "local_program_bytes_match": True,
    "operator_input_bytes_captured": True,
    "local_closure_reverified": False,
    "workbench_execution_qualified": False,
    "scientific_execution_qualified": False,
    "transform_executed": False,
    "atlas_correctness_established": False,
    "fmq010_established": False,
    "neural_alignment_established": False,
    "consciousness_evidence": False,
}


class CaptureError(ContractError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise CaptureError(f"JSON object: duplicate key: {key}")
        out[key] = value
    return out


def load_strict_bytes(data: bytes, label: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CaptureError(f"{label}: UTF-8 required") from exc
    try:
        return json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise CaptureError(f"{label}: invalid JSON") from exc


def strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise CaptureError(f"{label}: boolean required")
    return value


def strict_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise CaptureError(f"{label}: integer required")
    return value


def canonical_sha256(value: Any, label: str) -> str:
    try:
        return sha256(value, label)
    except ContractError as exc:
        raise CaptureError(str(exc)) from exc


def parse_inputs(items: Iterable[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise CaptureError("input: expected ROLE=PATH")
        role, raw = item.split("=", 1)
        if role not in REQUIRED_INPUT_ROLES:
            raise CaptureError(f"input: unknown role {role!r}")
        if role in parsed:
            raise CaptureError(f"input: duplicate role {role!r}")
        if not raw:
            raise CaptureError(f"input {role}: empty path")
        parsed[role] = Path(raw)
    missing = sorted(REQUIRED_INPUT_ROLES - set(parsed))
    if missing:
        raise CaptureError(f"input: missing roles: {', '.join(missing)}")
    return parsed


def resolve_regular(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise CaptureError(f"{label}: direct symlink forbidden")
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as exc:
        raise CaptureError(f"{label}: file not found") from exc
    if not resolved.is_file():
        raise CaptureError(f"{label}: regular file required")
    return resolved


def digest_regular_file(path: Path, label: str) -> str:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CaptureError(f"{label}: unable to open regular file") from exc
    h = hashlib.sha256()
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise CaptureError(f"{label}: opened object is not a regular file")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    finally:
        os.close(fd)
    return "sha256:" + h.hexdigest()


def verify_profile_document(value: Any) -> dict[str, Any]:
    profile = exact(value, PROFILE_KEYS, "verified-run profile")
    if profile["schema"] != PROFILE_SCHEMA or profile["status"] != "qualified-workbench-observation-bound":
        raise CaptureError("verified-run profile: schema/status mismatch")

    source = exact(profile["qualification_source"], QUALIFICATION_SOURCE_KEYS, "qualification source")
    if strict_int(source["pr"], "qualification source pr") < 1:
        raise CaptureError("qualification source: positive PR number required")
    if strict_int(source["workflow_run_id"], "qualification source workflow_run_id") < 1:
        raise CaptureError("qualification source: positive workflow run id required")
    if not isinstance(source["head_sha"], str) or not GIT_SHA_RE.fullmatch(source["head_sha"]):
        raise CaptureError("qualification source: exact 40-hex Git head required")
    if not isinstance(source["verification_file_path"], str) or not source["verification_file_path"]:
        raise CaptureError("qualification source: verification file path required")
    for key in (
        "verification_file_sha256", "independent_verification_archive_sha256",
        "raw_evidence_archive_sha256",
    ):
        canonical_sha256(source[key], f"qualification source {key}")

    qualified = exact(profile["qualified_workbench"], QUALIFIED_WORKBENCH_KEYS, "qualified workbench")
    if not isinstance(qualified["root"], str) or not STORE_PATH_RE.fullmatch(qualified["root"]):
        raise CaptureError("qualified workbench: canonical Nix store root required")
    if qualified["relative_main_program"] != "bin/wb_command":
        raise CaptureError("qualified workbench: exact relative main program required")
    for key in ("program_content_sha256", "version_output_sha256", "closure_digest"):
        canonical_sha256(qualified[key], f"qualified workbench {key}")
    if not isinstance(qualified["version"], str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", qualified["version"]):
        raise CaptureError("qualified workbench: semantic version required")
    if qualified["qualification_platform"] != "x86_64-linux":
        raise CaptureError("qualified workbench: v2 requires x86_64-linux")

    required_authority = exact(
        profile["required_verification_authority"],
        VERIFICATION_AUTHORITY_KEYS,
        "required verification authority",
    )
    for key, authority_value in required_authority.items():
        strict_bool(authority_value, f"required verification authority {key}")
    true_keys = {
        "program_membership_verified", "invocation_profile_verified",
        "invocation_executed", "version_output_bound",
    }
    for key in VERIFICATION_AUTHORITY_KEYS:
        if required_authority[key] is not (key in true_keys):
            raise CaptureError(f"required verification authority: unexpected value for {key}")

    authority = exact(profile["authority"], PROFILE_AUTHORITY_KEYS, "profile authority")
    for key, authority_value in authority.items():
        strict_bool(authority_value, f"profile authority {key}")
    if authority["verified_workbench_observation_bound"] is not True:
        raise CaptureError("profile authority: qualified observation binding required")
    for key in PROFILE_AUTHORITY_KEYS - {"verified_workbench_observation_bound"}:
        if authority[key] is not False:
            raise CaptureError(f"profile authority escalation forbidden: {key}")
    return profile


def verify_profile_bytes(raw: bytes) -> dict[str, Any]:
    profile = verify_profile_document(load_strict_bytes(raw, "verified-run profile"))
    if raw != canonical_json_bytes(profile) + b"\n":
        raise CaptureError("verified-run profile: exact canonical JSON bytes required")
    return profile


def verify_retained_workbench_bytes(
    profile: dict[str, Any],
    raw: bytes,
) -> dict[str, Any]:
    expected_file_digest = profile["qualification_source"]["verification_file_sha256"]
    if digest_bytes(raw) != expected_file_digest:
        raise CaptureError("workbench verification: external retained file root mismatch")

    verification = exact(
        load_strict_bytes(raw, "workbench verification"),
        VERIFICATION_KEYS,
        "workbench verification",
    )
    if raw != canonical_json_bytes(verification) + b"\n":
        raise CaptureError("workbench verification: exact canonical retained bytes required")
    if verification["schema"] != VERIFICATION_SCHEMA or verification["status"] != VERIFICATION_STATUS:
        raise CaptureError("workbench verification: schema/status mismatch")

    qualified = profile["qualified_workbench"]
    expected = {
        "root": qualified["root"],
        "qualification_platform": qualified["qualification_platform"],
        "closure_digest": qualified["closure_digest"],
        "program_content_sha256": qualified["program_content_sha256"],
        "version": qualified["version"],
        "version_output_sha256": qualified["version_output_sha256"],
    }
    for key, expected_value in expected.items():
        if verification[key] != expected_value:
            raise CaptureError(f"workbench verification: profile mismatch: {key}")

    canonical_sha256(verification["entry_environment_sha256"], "workbench verification entry environment")
    diagnostics = exact(verification["diagnostics"], DIAGNOSTIC_KEYS, "workbench diagnostics")
    for key in ("cpu_vendor", "cpu_family", "cpu_model", "cpu_stepping", "kernel_release"):
        if not isinstance(diagnostics[key], str) or not diagnostics[key]:
            raise CaptureError(f"workbench diagnostics: non-empty {key} required")
    canonical_sha256(diagnostics["cpu_flags_digest"], "workbench diagnostics CPU flags")
    implementations = exact(verification["implementations"], IMPLEMENTATION_KEYS, "workbench implementations")
    for key, implementation_digest in implementations.items():
        canonical_sha256(implementation_digest, f"workbench implementation {key}")

    authority = exact(verification["authority"], VERIFICATION_AUTHORITY_KEYS, "workbench verification authority")
    if authority != profile["required_verification_authority"]:
        raise CaptureError("workbench verification: authority mismatch")
    return verification


def qualified_program_path(profile: dict[str, Any], verification: dict[str, Any]) -> Path:
    expected = Path(verification["root"]) / profile["qualified_workbench"]["relative_main_program"]
    if expected.is_symlink():
        raise CaptureError("qualified workbench: main program symlink forbidden")
    try:
        resolved = expected.resolve(strict=True)
    except OSError as exc:
        raise CaptureError("qualified workbench: local program not found") from exc
    if resolved != expected or not resolved.is_file():
        raise CaptureError("qualified workbench: exact regular local program required")
    return resolved


def capture_manifest(
    method_manifest: Path,
    profile_path: Path,
    verification_path: Path,
    input_items: Iterable[str],
    execution_id: str,
    authorization_reference: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    method_manifest = resolve_regular(method_manifest, "method manifest")
    method = load_method(method_manifest)
    profile_path = resolve_regular(profile_path, "verified-run profile")
    verification_path = resolve_regular(verification_path, "workbench verification")

    profile_raw = profile_path.read_bytes()
    verification_raw = verification_path.read_bytes()
    profile = verify_profile_bytes(profile_raw)
    verification = verify_retained_workbench_bytes(profile, verification_raw)
    profile_file_sha256 = digest_bytes(profile_raw)
    verification_file_sha256 = digest_bytes(verification_raw)

    nonempty(execution_id, "execution_id")
    nonempty(authorization_reference, "authorization_reference")
    supplied = parse_inputs(input_items)

    resolved_inputs: dict[str, Path] = {}
    first_input_digests: dict[str, str] = {}
    for role in sorted(REQUIRED_INPUT_ROLES):
        input_path = resolve_regular(supplied[role], f"input {role}")
        resolved_inputs[role] = input_path
        first_input_digests[role] = digest_regular_file(input_path, f"input {role}")

    program = qualified_program_path(profile, verification)
    program_before = digest_regular_file(program, "qualified workbench program")
    if program_before != verification["program_content_sha256"]:
        raise CaptureError("qualified workbench: local program bytes differ from retained verification")

    for role in sorted(REQUIRED_INPUT_ROLES):
        if digest_regular_file(resolved_inputs[role], f"input {role}") != first_input_digests[role]:
            raise CaptureError(f"input {role}: bytes changed during capture")
    program_after = digest_regular_file(program, "qualified workbench program")
    if program_after != program_before:
        raise CaptureError("qualified workbench: local program bytes changed during capture")

    inputs = {
        role: {"path": str(resolved_inputs[role]), "sha256": first_input_digests[role]}
        for role in sorted(REQUIRED_INPUT_ROLES)
    }
    doc = {
        "schema": RUN_SCHEMA,
        "method_manifest_digest": digest_file(method_manifest),
        "execution_id": execution_id,
        "authorization_reference": authorization_reference,
        "workbench": {
            "path": str(program),
            "sha256": verification["program_content_sha256"],
            "version_output_sha256": verification["version_output_sha256"],
        },
        "inputs": inputs,
    }
    with tempfile.TemporaryDirectory(prefix="symthaea-hcpmmp-verified-run-capture-") as td:
        candidate = Path(td) / "run.json"
        candidate.write_bytes(canonical_json_bytes(doc) + b"\n")
        load_run(candidate, method, method_manifest)

    capture_metadata = {
        "verified_run_profile_sha256": profile_file_sha256,
        "workbench_verification_file_sha256": verification_file_sha256,
        "qualified_workbench_head": profile["qualification_source"]["head_sha"],
        "closure_digest": verification["closure_digest"],
        "authority": dict(CAPTURE_AUTHORITY),
    }
    return doc, capture_metadata


def write_new(path: Path, doc: dict[str, Any]) -> Path:
    requested = path.expanduser()
    if not requested.name or requested.name in {".", ".."}:
        raise CaptureError("output: canonical file name required")
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    payload = canonical_json_bytes(doc) + b"\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=parent)
    tmp = Path(tmp_name)
    try:
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, target)
        except FileExistsError as exc:
            raise CaptureError("output: refusing to overwrite existing manifest") from exc
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture a Lineage-B run manifest from a retained qualified Workbench verification; never execute Workbench."
    )
    parser.add_argument("--method-manifest", required=True, type=Path)
    parser.add_argument("--verified-run-profile", required=True, type=Path)
    parser.add_argument("--workbench-verification", required=True, type=Path)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--authorization-reference", required=True)
    parser.add_argument("--input", action="append", required=True, metavar="ROLE=PATH")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        doc, metadata = capture_manifest(
            args.method_manifest,
            args.verified_run_profile,
            args.workbench_verification,
            args.input,
            args.execution_id,
            args.authorization_reference,
        )
        target = write_new(args.output, doc)
        receipt = {
            "profile": CAPTURE_PROFILE,
            "run_manifest_file_sha256": digest_file(target),
            **metadata,
        }
    except (CaptureError, ContractError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
