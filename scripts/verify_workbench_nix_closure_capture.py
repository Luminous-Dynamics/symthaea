#!/usr/bin/env python3
"""Independent hostile-input verifier for Workbench Nix closure capture receipts."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import workbench_nix_closure_identity as closure

SCHEMA = "symthaea-workbench-nix-closure-capture-receipt-v1"
PROFILE_SCHEMA = "symthaea-workbench-execution-capsule-profile-v1"
STATUS_SUCCESS = "observed-normalized-unqualified"
STATUS_FAILURE = "observation-incomplete-unqualified"
STORE_DIR = "/nix/store"
EXPECTED_NIX_VERSION = "2.33.6"
NIX_VERSION_ARGV = ["nix", "--version"]
PLATFORM_ARGV = ["nix", "eval", "--raw", "--impure", "--expr", "builtins.currentSystem"]
REALIZE_PREFIX = ["nix", "build", "--no-link", "--print-out-paths", "--no-write-lock-file"]
PATH_INFO_PREFIX = ["nix", "path-info", "--json", "--json-format", "2", "--recursive"]
VERSION_RE = re.compile(r"^nix \(Nix\) ([0-9]+\.[0-9]+\.[0-9]+)\r?\n?$", re.ASCII)
TOP_KEYS = {
    "schema", "status", "selection", "root", "implementations", "commands", "normalized",
    "facts", "authority", "raw_observation_digest", "capture_digest",
}
SELECTION_KEYS = {
    "profile_sha256", "flake_lock_sha256", "root_input", "locked_node", "owner", "repo", "rev",
    "nar_hash", "attribute", "qualification_platform", "installable",
}
IMPLEMENTATION_KEYS = {"capture_sha256", "normalizer_sha256"}
COMMANDS_KEYS = {"nix_version", "platform", "realization", "path_info"}
COMMAND_KEYS = {"argv", "exit_code", "stdout", "stderr"}
STREAM_KEYS = {"path", "byte_length", "sha256"}
NORMALIZED_KEYS = {"nix_version", "platform", "closure_identity_path", "closure_identity_sha256", "closure_digest"}
FACT_KEYS = {
    "selection_revalidated", "nix_version_observed", "platform_observed", "realization_command_observed",
    "realized_root_observed", "path_info_observed", "canonical_closure_identity_compiled",
}
AUTHORITY_KEYS = {
    "closure_capture_qualified", "workbench_execution_qualified", "transform_executed",
    "atlas_correctness_established", "fmq010_established", "neural_alignment_established",
    "consciousness_evidence",
}
FIXED_STREAM_PATHS = {
    "nix_version": ("raw/nix-version.stdout", "raw/nix-version.stderr"),
    "platform": ("raw/platform.stdout", "raw/platform.stderr"),
    "realization": ("raw/realization.stdout", "raw/realization.stderr"),
    "path_info": ("raw/path-info.stdout", "raw/path-info.stderr"),
}


class VerificationError(ValueError):
    pass


def exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise VerificationError(f"{label}: closed-world schema mismatch")
    return value


def strict_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        requirement = "integer required" if minimum is None else f"integer >= {minimum} required"
        raise VerificationError(f"{label}: {requirement}")
    return value


def strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise VerificationError(f"{label}: boolean required")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def canonical_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise VerificationError(f"{label}: canonical sha256 hex required")
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


def strict_json_file(path: Path, label: str) -> Any:
    return strict_json_bytes(path.read_bytes(), label)


def derive_expected_selection(profile_path: Path, lock_path: Path) -> dict[str, Any]:
    profile = strict_json_file(profile_path, "capsule profile")
    if not isinstance(profile, dict) or profile.get("schema") != PROFILE_SCHEMA:
        raise VerificationError("capsule profile: schema mismatch")
    required = {"qualification_platform", "flake_selection", "nixpkgs_package"}
    if not required.issubset(profile):
        raise VerificationError("capsule profile: required selection fields missing")
    fs = profile["flake_selection"]
    pkg = profile["nixpkgs_package"]
    if not isinstance(fs, dict) or set(fs) != {"root_input", "locked_node", "rev", "nar_hash"}:
        raise VerificationError("capsule profile: flake selection schema mismatch")
    if not isinstance(pkg, dict) or not isinstance(pkg.get("attribute"), str) or not pkg["attribute"]:
        raise VerificationError("capsule profile: package attribute required")
    if profile["qualification_platform"] != "x86_64-linux":
        raise VerificationError("capsule profile: v1 requires x86_64-linux")

    lock = strict_json_file(lock_path, "flake.lock")
    if not isinstance(lock, dict) or set(lock) != {"nodes", "root", "version"}:
        raise VerificationError("flake.lock: top-level schema mismatch")
    nodes = lock["nodes"]
    root_name = lock["root"]
    if not isinstance(nodes, dict) or not isinstance(root_name, str) or root_name not in nodes:
        raise VerificationError("flake.lock: root node missing")
    root = nodes[root_name]
    if not isinstance(root, dict) or not isinstance(root.get("inputs"), dict):
        raise VerificationError("flake.lock: root inputs missing")
    root_input = fs["root_input"]
    locked_node = fs["locked_node"]
    if root["inputs"].get(root_input) != locked_node:
        raise VerificationError("flake.lock: root input selection mismatch")
    node = nodes.get(locked_node)
    if not isinstance(node, dict) or not isinstance(node.get("locked"), dict):
        raise VerificationError("flake.lock: locked node missing")
    locked = node["locked"]
    for field in ("owner", "repo", "rev", "narHash", "type"):
        if field not in locked:
            raise VerificationError(f"flake.lock: selected node missing {field}")
    if locked["type"] != "github" or locked["rev"] != fs["rev"] or locked["narHash"] != fs["nar_hash"]:
        raise VerificationError("flake.lock: selected node identity mismatch")
    owner, repo = locked["owner"], locked["repo"]
    if not isinstance(owner, str) or not isinstance(repo, str) or not owner or not repo:
        raise VerificationError("flake.lock: selected owner/repo required")
    attr = pkg["attribute"]
    return {
        "profile_sha256": file_digest(profile_path),
        "flake_lock_sha256": file_digest(lock_path),
        "root_input": root_input,
        "locked_node": locked_node,
        "owner": owner,
        "repo": repo,
        "rev": fs["rev"],
        "nar_hash": fs["nar_hash"],
        "attribute": attr,
        "qualification_platform": profile["qualification_platform"],
        "installable": f"github:{owner}/{repo}/{fs['rev']}#{attr}",
    }


def sri_sha256_to_hex(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("sha256-"):
        raise VerificationError("NAR hash: sha256 SRI required")
    payload = value[7:]
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise VerificationError("NAR hash: canonical base64 required") from exc
    if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != payload:
        raise VerificationError("NAR hash: canonical SHA-256 SRI required")
    return "sha256:" + raw.hex()


def parse_path_info_v2(data: bytes) -> list[dict[str, Any]]:
    value = strict_json_bytes(data, "path-info stdout")
    if not isinstance(value, dict) or set(value) != {"version", "storeDir", "info"}:
        raise VerificationError("path-info: exact v2 top-level schema required")
    if type(value["version"]) is not int or value["version"] != 2:
        raise VerificationError("path-info: JSON format version 2 required")
    if value["storeDir"] != STORE_DIR:
        raise VerificationError("path-info: /nix/store required")
    info = value["info"]
    if not isinstance(info, dict) or not info:
        raise VerificationError("path-info: non-empty info map required")
    entries: list[dict[str, Any]] = []
    for basename, record in info.items():
        if not isinstance(basename, str) or not basename or "/" in basename or "\n" in basename or "\r" in basename:
            raise VerificationError("path-info: canonical store basename required")
        if not isinstance(record, dict) or "narHash" not in record or "references" not in record:
            raise VerificationError(f"path-info {basename}: narHash and references required")
        refs = record["references"]
        if not isinstance(refs, list) or any(
            not isinstance(ref, str) or not ref or "/" in ref or "\n" in ref or "\r" in ref for ref in refs
        ):
            raise VerificationError(f"path-info {basename}: store-basename references required")
        entries.append({
            "path": f"{STORE_DIR}/{basename}",
            "nar_sha256": sri_sha256_to_hex(record["narHash"]),
            "references": [f"{STORE_DIR}/{ref}" for ref in refs],
        })
    return entries


def parse_nix_version(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError("nix --version: UTF-8 required") from exc
    match = VERSION_RE.fullmatch(text)
    if not match or match.group(1) != EXPECTED_NIX_VERSION:
        raise VerificationError(f"nix --version: exact Nix {EXPECTED_NIX_VERSION} required")
    return match.group(1)


def parse_platform(data: bytes) -> str:
    try:
        value = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError("platform: UTF-8 required") from exc
    if not value or any(ch.isspace() for ch in value):
        raise VerificationError("platform: one raw Nix system token required")
    return value


def parse_realized_root(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError("realization stdout: UTF-8 required") from exc
    lines = text.splitlines()
    if len(lines) != 1 or text not in {lines[0], lines[0] + "\n"}:
        raise VerificationError("realization stdout: exactly one store path required")
    try:
        return closure.store_path(lines[0], "realized Workbench root")
    except closure.ContractError as exc:
        raise VerificationError(str(exc)) from exc


def safe_file(receipt_root: Path, rel: str) -> Path:
    if not isinstance(rel, str):
        raise VerificationError("receipt sidecar path: string required")
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or "." in p.parts or not p.parts:
        raise VerificationError("receipt sidecar path: canonical relative path required")
    root = receipt_root.resolve(strict=True)
    candidate = root.joinpath(*p.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise VerificationError(f"receipt sidecar missing: {rel}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise VerificationError(f"receipt sidecar escapes receipt root: {rel}") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise VerificationError(f"receipt sidecar must be a regular non-symlink file: {rel}")
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise VerificationError(f"receipt sidecar must be regular: {rel}")
    return resolved


def verify_stream(receipt_root: Path, value: Any, expected_rel: str, label: str) -> bytes:
    stream = exact(value, STREAM_KEYS, label)
    if stream["path"] != expected_rel:
        raise VerificationError(f"{label}: fixed sidecar path mismatch")
    length = strict_int(stream["byte_length"], f"{label} byte_length", minimum=0)
    digest = canonical_digest(stream["sha256"], f"{label} sha256")
    data = safe_file(receipt_root, expected_rel).read_bytes()
    if len(data) != length or digest_bytes(data) != digest:
        raise VerificationError(f"{label}: retained bytes mismatch")
    return data


def verify_command(receipt_root: Path, name: str, value: Any, expected_argv: list[str]) -> tuple[int, bytes, bytes]:
    command = exact(value, COMMAND_KEYS, f"command {name}")
    argv = command["argv"]
    if not isinstance(argv, list) or any(not isinstance(x, str) for x in argv) or argv != expected_argv:
        raise VerificationError(f"command {name}: exact argv mismatch")
    rc = strict_int(command["exit_code"], f"command {name} exit_code")
    out_rel, err_rel = FIXED_STREAM_PATHS[name]
    stdout = verify_stream(receipt_root, command["stdout"], out_rel, f"command {name} stdout")
    stderr = verify_stream(receipt_root, command["stderr"], err_rel, f"command {name} stderr")
    return rc, stdout, stderr


def verify_file_inventory(receipt_root: Path, expected_files: set[str]) -> None:
    root = receipt_root.resolve(strict=True)
    actual: set[str] = set()
    for item in root.rglob("*"):
        rel = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise VerificationError(f"receipt inventory: symlink forbidden: {rel}")
        if item.is_file():
            actual.add(rel)
    if actual != expected_files:
        missing = sorted(expected_files - actual)
        extra = sorted(actual - expected_files)
        raise VerificationError(f"receipt inventory mismatch; missing={missing} extra={extra}")


def verify_receipt(
    receipt_dir: Path,
    profile_path: Path,
    lock_path: Path,
    producer_path: Path,
    normalizer_path: Path,
) -> dict[str, Any]:
    receipt_root = receipt_dir.resolve(strict=True)
    if not receipt_root.is_dir():
        raise VerificationError("receipt: directory required")
    receipt_file = safe_file(receipt_root, "receipt.json")
    receipt_bytes = receipt_file.read_bytes()
    receipt = exact(strict_json_bytes(receipt_bytes, "receipt"), TOP_KEYS, "receipt")
    if receipt_bytes != canonical_json_bytes(receipt) + b"\n":
        raise VerificationError("receipt: exact canonical JSON bytes required")
    if receipt["schema"] != SCHEMA:
        raise VerificationError("receipt: schema mismatch")
    if receipt["status"] not in {STATUS_SUCCESS, STATUS_FAILURE}:
        raise VerificationError("receipt: status mismatch")

    expected_selection = derive_expected_selection(profile_path, lock_path)
    selection = exact(receipt["selection"], SELECTION_KEYS, "receipt selection")
    if selection != expected_selection:
        raise VerificationError("receipt selection: exact profile/lock selection mismatch")

    impl = exact(receipt["implementations"], IMPLEMENTATION_KEYS, "receipt implementations")
    if Path(closure.__file__).resolve() != normalizer_path.resolve():
        raise VerificationError("normalizer: imported implementation differs from supplied source path")
    if impl != {"capture_sha256": file_digest(producer_path), "normalizer_sha256": file_digest(normalizer_path)}:
        raise VerificationError("receipt implementations: source digest mismatch")

    commands = exact(receipt["commands"], COMMANDS_KEYS, "receipt commands")
    version_rc, version_out, _ = verify_command(receipt_root, "nix_version", commands["nix_version"], NIX_VERSION_ARGV)
    platform_rc, platform_out, _ = verify_command(receipt_root, "platform", commands["platform"], PLATFORM_ARGV)
    realization_argv = REALIZE_PREFIX + [expected_selection["installable"]]
    realize_rc, realize_out, _ = verify_command(receipt_root, "realization", commands["realization"], realization_argv)

    root: str | None = None
    if realize_rc == 0:
        try:
            root = parse_realized_root(realize_out)
        except VerificationError:
            root = None
    if receipt["root"] != root:
        raise VerificationError("receipt root: must be exactly derived from realization stdout")

    path_rc: int | None = None
    path_out: bytes | None = None
    if root is None:
        if commands["path_info"] is not None:
            raise VerificationError("path-info: must be absent when no realized root exists")
    else:
        if commands["path_info"] is None:
            raise VerificationError("path-info: command record required for observed root")
        path_rc, path_out, _ = verify_command(receipt_root, "path_info", commands["path_info"], PATH_INFO_PREFIX + [root])

    nix_version: str | None = None
    platform: str | None = None
    identity: dict[str, Any] | None = None
    normalization_ok = False
    if version_rc == 0 and platform_rc == 0 and root is not None and path_rc == 0 and path_out is not None:
        try:
            nix_version = parse_nix_version(version_out)
            platform = parse_platform(platform_out)
            if platform != expected_selection["qualification_platform"]:
                raise VerificationError("platform: observed platform differs from profile")
            entries = parse_path_info_v2(path_out)
            identity = closure.compile_identity(root, entries)
            normalization_ok = True
        except (VerificationError, closure.ContractError):
            normalization_ok = False
            nix_version = None
            platform = None
            identity = None

    facts = exact(receipt["facts"], FACT_KEYS, "receipt facts")
    for key in FACT_KEYS:
        strict_bool(facts[key], f"receipt fact {key}")
    expected_facts = {
        "selection_revalidated": True,
        "nix_version_observed": version_rc == 0,
        "platform_observed": platform_rc == 0,
        "realization_command_observed": True,
        "realized_root_observed": root is not None,
        "path_info_observed": commands["path_info"] is not None,
        "canonical_closure_identity_compiled": normalization_ok,
    }
    if facts != expected_facts:
        raise VerificationError("receipt facts: derived fact mismatch")

    authority = exact(receipt["authority"], AUTHORITY_KEYS, "receipt authority")
    for key in AUTHORITY_KEYS:
        if strict_bool(authority[key], f"receipt authority {key}") is not False:
            raise VerificationError(f"receipt authority: escalation forbidden: {key}")

    normalized = exact(receipt["normalized"], NORMALIZED_KEYS, "receipt normalized")
    expected_files = {
        "receipt.json",
        "raw/nix-version.stdout", "raw/nix-version.stderr",
        "raw/platform.stdout", "raw/platform.stderr",
        "raw/realization.stdout", "raw/realization.stderr",
    }
    if commands["path_info"] is not None:
        expected_files |= {"raw/path-info.stdout", "raw/path-info.stderr"}

    if normalization_ok:
        assert identity is not None and nix_version is not None and platform is not None
        if receipt["status"] != STATUS_SUCCESS:
            raise VerificationError("receipt status: complete normalized observation must be success status")
        if normalized["nix_version"] != nix_version or normalized["platform"] != platform:
            raise VerificationError("receipt normalized: version/platform mismatch")
        if normalized["closure_identity_path"] != "normalized/closure_identity.json":
            raise VerificationError("receipt normalized: fixed closure identity path required")
        identity_file = safe_file(receipt_root, "normalized/closure_identity.json")
        identity_bytes = identity_file.read_bytes()
        expected_identity_bytes = canonical_json_bytes(identity) + b"\n"
        if identity_bytes != expected_identity_bytes:
            raise VerificationError("normalized closure identity: exact canonical bytes mismatch")
        loaded_identity = strict_json_bytes(identity_bytes, "normalized closure identity")
        try:
            closure.validate_identity(loaded_identity)
        except closure.ContractError as exc:
            raise VerificationError(str(exc)) from exc
        if loaded_identity != identity:
            raise VerificationError("normalized closure identity: raw reconstruction mismatch")
        if normalized["closure_identity_sha256"] != digest_bytes(identity_bytes):
            raise VerificationError("receipt normalized: closure identity file digest mismatch")
        if normalized["closure_digest"] != identity["closure_digest"]:
            raise VerificationError("receipt normalized: closure digest mismatch")
        expected_files.add("normalized/closure_identity.json")
    else:
        if receipt["status"] != STATUS_FAILURE:
            raise VerificationError("receipt status: incomplete observation must be failure status")
        if normalized != {
            "nix_version": None,
            "platform": None,
            "closure_identity_path": None,
            "closure_identity_sha256": None,
            "closure_digest": None,
        }:
            raise VerificationError("receipt normalized: incomplete observation must not retain normalized claims")

    raw_digest = canonical_digest(receipt["raw_observation_digest"], "receipt raw_observation_digest")
    if raw_digest != digest_bytes(canonical_json_bytes(commands)):
        raise VerificationError("receipt raw observation digest mismatch")
    capture_digest = canonical_digest(receipt["capture_digest"], "receipt capture_digest")
    expected_capture_digest = digest_bytes(canonical_json_bytes({k: v for k, v in receipt.items() if k != "capture_digest"}))
    if capture_digest != expected_capture_digest:
        raise VerificationError("receipt capture digest mismatch")

    verify_file_inventory(receipt_root, expected_files)
    return {
        "schema": "symthaea-workbench-nix-closure-capture-verification-v1",
        "status": "verified-complete-observation" if normalization_ok else "verified-incomplete-observation",
        "verifier_sha256": file_digest(Path(__file__)),
        "capture_digest": capture_digest,
        "closure_digest": identity["closure_digest"] if identity is not None else None,
        "authority": {
            "capture_receipt_verified": True,
            "workbench_execution_qualified": False,
            "transform_executed": False,
            "fmq010_established": False,
            "neural_alignment_established": False,
            "consciousness_evidence": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dir", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--flake-lock", required=True, type=Path)
    parser.add_argument("--producer", required=True, type=Path)
    parser.add_argument("--normalizer", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_receipt(args.receipt_dir, args.profile, args.flake_lock, args.producer, args.normalizer)
    except (VerificationError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
