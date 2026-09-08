#!/usr/bin/env python3
"""Fresh-runner verifier for a captured Workbench root NAR and wb_command membership."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "symthaea-workbench-root-nar-capture-receipt-v1"
VERIFICATION_SCHEMA = "symthaea-workbench-root-nar-capture-verification-v1"
STATUS_SUCCESS = "observed-root-nar-unqualified"
STATUS_FAILURE = "root-nar-observation-incomplete-unqualified"
STORE_PATH_RE = re.compile(r"^/nix/store/[0123456789abcdfghijklmnpqrsvwxyz]{32}-[^/\r\n]+\Z")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}\Z")
TOP_KEYS = {
    "schema", "status", "closure_capture_digest", "root", "implementation_sha256",
    "command", "nar", "stderr", "authority", "capture_digest",
}
COMMAND_KEYS = {"argv", "exit_code"}
STREAM_KEYS = {"path", "byte_length", "sha256"}
AUTHORITY_KEYS = {
    "root_nar_capture_verified", "target_membership_verified",
    "root_main_program_regular_executable_verified", "workbench_execution_qualified",
    "transform_executed", "fmq010_established", "neural_alignment_established",
    "consciousness_evidence",
}


class VerificationError(ValueError):
    pass


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


def canonical_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise VerificationError(f"{label}: canonical sha256:<64 lowercase hex> required")
    return value


def canonical_store_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not STORE_PATH_RE.fullmatch(value):
        raise VerificationError(f"{label}: canonical /nix/store path required")
    return value


def strict_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise VerificationError(f"{label}: integer required")
    return value


def exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise VerificationError(f"{label}: closed-world schema mismatch")
    return value


def safe_file(root_dir: Path, rel: str) -> Path:
    if not isinstance(rel, str):
        raise VerificationError("sidecar path: string required")
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
        raise VerificationError(f"sidecar must be a regular non-symlink file: {rel}")
    return resolved


def verify_stream(root: Path, value: Any, expected_rel: str, label: str) -> Path:
    stream = exact(value, STREAM_KEYS, label)
    if stream["path"] != expected_rel:
        raise VerificationError(f"{label}: fixed sidecar path mismatch")
    length = strict_int(stream["byte_length"], f"{label} byte_length")
    if length < 0:
        raise VerificationError(f"{label}: non-negative byte length required")
    expected_digest = canonical_sha256(stream["sha256"], f"{label} sha256")
    path = safe_file(root, expected_rel)
    if path.stat().st_size != length or digest_file(path) != expected_digest:
        raise VerificationError(f"{label}: retained bytes mismatch")
    return path


def verify_inventory(root_dir: Path) -> None:
    root = root_dir.resolve(strict=True)
    expected = {"receipt.json", "raw/root.nar", "raw/nix-store-dump.stderr"}
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


def verify_dump_receipt(
    capture_dir: Path,
    *,
    expected_root: str,
    expected_closure_capture_digest: str,
    producer_path: Path,
    require_success: bool = True,
) -> tuple[dict[str, Any], Path]:
    root = capture_dir.resolve(strict=True)
    if not root.is_dir():
        raise VerificationError("root-NAR capture: directory required")
    receipt_path = safe_file(root, "receipt.json")
    receipt_bytes = receipt_path.read_bytes()
    receipt = exact(strict_json_bytes(receipt_bytes, "root-NAR receipt"), TOP_KEYS, "root-NAR receipt")
    if receipt_bytes != canonical_json_bytes(receipt) + b"\n":
        raise VerificationError("root-NAR receipt: exact canonical JSON bytes required")
    if receipt["schema"] != SCHEMA:
        raise VerificationError("root-NAR receipt: schema mismatch")
    if receipt["status"] not in {STATUS_SUCCESS, STATUS_FAILURE}:
        raise VerificationError("root-NAR receipt: status mismatch")

    expected_root = canonical_store_path(expected_root, "verified closure root")
    if receipt["root"] != expected_root:
        raise VerificationError("root-NAR receipt: root differs from independently verified closure root")
    expected_closure_capture_digest = canonical_sha256(
        expected_closure_capture_digest, "verified closure capture digest"
    )
    if receipt["closure_capture_digest"] != expected_closure_capture_digest:
        raise VerificationError("root-NAR receipt: closure capture digest mismatch")
    if receipt["implementation_sha256"] != digest_file(producer_path):
        raise VerificationError("root-NAR receipt: producer source digest mismatch")

    command = exact(receipt["command"], COMMAND_KEYS, "root-NAR dump command")
    expected_argv = ["nix-store", "--dump", expected_root]
    if command["argv"] != expected_argv:
        raise VerificationError("root-NAR dump command: exact argv mismatch")
    rc = strict_int(command["exit_code"], "root-NAR dump command exit_code")
    nar_path = verify_stream(root, receipt["nar"], "raw/root.nar", "root NAR")
    verify_stream(root, receipt["stderr"], "raw/nix-store-dump.stderr", "root NAR stderr")

    authority = exact(receipt["authority"], AUTHORITY_KEYS, "root-NAR receipt authority")
    for key, value in authority.items():
        if type(value) is not bool or value is not False:
            raise VerificationError(f"root-NAR receipt authority: escalation forbidden: {key}")

    capture_digest = canonical_sha256(receipt["capture_digest"], "root-NAR capture digest")
    expected_capture_digest = digest_bytes(
        canonical_json_bytes({key: value for key, value in receipt.items() if key != "capture_digest"})
    )
    if capture_digest != expected_capture_digest:
        raise VerificationError("root-NAR receipt: capture digest mismatch")
    verify_inventory(root)

    expected_status = STATUS_SUCCESS if rc == 0 else STATUS_FAILURE
    if receipt["status"] != expected_status:
        raise VerificationError("root-NAR receipt: status/exit-code mismatch")
    if require_success and rc != 0:
        raise VerificationError("root-NAR observation: successful dump required")
    return receipt, nar_path


def project_verified_root(identity: Any, expected_closure_digest: str) -> tuple[str, str]:
    if not isinstance(identity, dict):
        raise VerificationError("verified closure identity: object required")
    if identity.get("closure_digest") != expected_closure_digest:
        raise VerificationError("verified closure identity: closure digest mismatch")
    root = canonical_store_path(identity.get("root"), "verified closure identity root")
    entries = identity.get("entries")
    if not isinstance(entries, list):
        raise VerificationError("verified closure identity: entries list required")
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("path") == root]
    if len(matches) != 1:
        raise VerificationError("verified closure identity: exact root entry required")
    expected_nar = canonical_sha256(matches[0].get("nar_sha256"), "verified root NAR SHA-256")
    return root, expected_nar


def verify_pipeline(
    closure_receipt_dir: Path,
    capture_dir: Path,
    profile_path: Path,
    lock_path: Path,
    closure_producer_path: Path,
    normalizer_path: Path,
    closure_verifier_path: Path,
    nar_capture_producer_path: Path,
    membership_path: Path,
) -> dict[str, Any]:
    import verify_workbench_nix_closure_capture as closure_verifier
    import workbench_root_nar_membership as membership

    if Path(closure_verifier.__file__).resolve() != closure_verifier_path.resolve():
        raise VerificationError("closure verifier: imported implementation differs from supplied source path")
    if Path(membership.__file__).resolve() != membership_path.resolve():
        raise VerificationError("membership verifier: imported implementation differs from supplied source path")

    try:
        closure_verification = closure_verifier.verify_receipt(
            closure_receipt_dir,
            profile_path,
            lock_path,
            closure_producer_path,
            normalizer_path,
        )
    except Exception as exc:
        raise VerificationError(f"closure receipt verification failed: {exc}") from exc
    closure_verification_keys = {
        "schema", "status", "verifier_sha256", "capture_digest", "closure_digest", "authority"
    }
    if not isinstance(closure_verification, dict) or set(closure_verification) != closure_verification_keys:
        raise VerificationError("closure verifier: exact result schema required")
    if closure_verification["schema"] != "symthaea-workbench-nix-closure-capture-verification-v1":
        raise VerificationError("closure verifier: schema mismatch")
    if closure_verification["status"] != "verified-complete-observation":
        raise VerificationError("closure receipt: verified complete observation required")
    required_closure_authority = {
        "capture_receipt_verified": True,
        "workbench_execution_qualified": False,
        "transform_executed": False,
        "fmq010_established": False,
        "neural_alignment_established": False,
        "consciousness_evidence": False,
    }
    if closure_verification["authority"] != required_closure_authority:
        raise VerificationError("closure verifier: authority boundary mismatch")
    closure_digest = canonical_sha256(closure_verification["closure_digest"], "verified closure digest")
    closure_capture_digest = canonical_sha256(
        closure_verification["capture_digest"], "verified closure capture digest"
    )
    if closure_verification["verifier_sha256"] != digest_file(closure_verifier_path):
        raise VerificationError("closure verifier: self-reported source digest mismatch")

    identity_path = closure_verifier.safe_file(
        closure_receipt_dir.resolve(strict=True), "normalized/closure_identity.json"
    )
    identity_bytes = identity_path.read_bytes()
    identity = strict_json_bytes(identity_bytes, "verified closure identity")
    try:
        validated_identity = closure_verifier.closure.validate_identity(identity)
    except Exception as exc:
        raise VerificationError(f"verified closure identity invalid: {exc}") from exc
    if identity_bytes != canonical_json_bytes(validated_identity) + b"\n":
        raise VerificationError("verified closure identity: exact canonical bytes required")
    root, expected_root_nar = project_verified_root(validated_identity, closure_digest)

    dump_receipt, nar_path = verify_dump_receipt(
        capture_dir,
        expected_root=root,
        expected_closure_capture_digest=closure_capture_digest,
        producer_path=nar_capture_producer_path,
        require_success=True,
    )
    try:
        membership_result = membership.verify_membership(nar_path, expected_root_nar, "bin/wb_command")
    except Exception as exc:
        raise VerificationError(f"root-NAR membership verification failed: {exc}") from exc
    membership_keys = {"schema", "status", "nar_sha256", "target_path", "target", "authority"}
    if not isinstance(membership_result, dict) or set(membership_result) != membership_keys:
        raise VerificationError("root-NAR membership: exact result schema required")
    if membership_result["schema"] != "symthaea-workbench-root-nar-membership-v1":
        raise VerificationError("root-NAR membership: schema mismatch")
    if membership_result["status"] != "verified-nar-membership-only":
        raise VerificationError("root-NAR membership: qualified verifier status required")
    if membership_result["nar_sha256"] != expected_root_nar:
        raise VerificationError("root-NAR membership: verified NAR hash mismatch")
    if membership_result["target_path"] != "bin/wb_command":
        raise VerificationError("root-NAR membership: exact target path required")
    membership_authority = membership_result["authority"]
    required_membership_authority = {
        "nar_bytes_match_verified_root": True,
        "target_membership_verified": True,
        "workbench_execution_qualified": False,
        "transform_executed": False,
        "fmq010_established": False,
        "neural_alignment_established": False,
        "consciousness_evidence": False,
    }
    if membership_authority != required_membership_authority:
        raise VerificationError("root-NAR membership: authority boundary mismatch")

    target = membership_result["target"]
    target_keys = {"node_type", "executable", "content_length", "content_sha256", "symlink_target_hex"}
    if not isinstance(target, dict) or set(target) != target_keys:
        raise VerificationError("root-NAR membership: exact target observation required")
    regular_executable = (
        target.get("node_type") == "regular"
        and target.get("executable") is True
        and isinstance(target.get("content_sha256"), str)
    )
    if target.get("node_type") == "symlink":
        status = "verified-root-nar-symlink-membership"
    elif regular_executable:
        status = "verified-root-nar-regular-executable-membership"
    else:
        status = "verified-root-nar-membership"

    return {
        "schema": VERIFICATION_SCHEMA,
        "status": status,
        "root": root,
        "root_nar_sha256": expected_root_nar,
        "closure_digest": closure_digest,
        "closure_capture_digest": closure_capture_digest,
        "root_nar_capture_digest": dump_receipt["capture_digest"],
        "implementations": {
            "verifier_sha256": digest_file(Path(__file__)),
            "closure_verifier_sha256": digest_file(closure_verifier_path),
            "membership_verifier_sha256": digest_file(membership_path),
            "nar_capture_producer_sha256": digest_file(nar_capture_producer_path),
        },
        "target": target,
        "authority": {
            "closure_receipt_verified": True,
            "root_nar_capture_verified": True,
            "target_membership_verified": True,
            "root_main_program_regular_executable_verified": regular_executable,
            "symlink_resolution_verified": False,
            "workbench_execution_qualified": False,
            "transform_executed": False,
            "fmq010_established": False,
            "neural_alignment_established": False,
            "consciousness_evidence": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-receipt-dir", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--flake-lock", required=True, type=Path)
    parser.add_argument("--closure-producer", required=True, type=Path)
    parser.add_argument("--normalizer", required=True, type=Path)
    parser.add_argument("--closure-verifier", required=True, type=Path)
    parser.add_argument("--nar-capture-producer", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_pipeline(
            args.closure_receipt_dir,
            args.capture_dir,
            args.profile,
            args.flake_lock,
            args.closure_producer,
            args.normalizer,
            args.closure_verifier,
            args.nar_capture_producer,
            args.membership,
        )
    except (VerificationError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
