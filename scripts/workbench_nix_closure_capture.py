#!/usr/bin/env python3
"""Raw, non-authorizing Nix realization/closure capture for Workbench."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import workbench_nix_closure_identity as closure

SCHEMA = "symthaea-workbench-nix-closure-capture-receipt-v1"
PROFILE_SCHEMA = "symthaea-workbench-execution-capsule-profile-v1"
STATUS_SUCCESS = "observed-normalized-unqualified"
STATUS_FAILURE = "observation-incomplete-unqualified"
STORE_DIR = "/nix/store"
NIX_VERSION_ARGV = ["nix", "--version"]
PLATFORM_ARGV = ["nix", "eval", "--raw", "--impure", "--expr", "builtins.currentSystem"]
REALIZE_PREFIX = ["nix", "build", "--no-link", "--print-out-paths", "--no-write-lock-file"]
PATH_INFO_PREFIX = ["nix", "path-info", "--json", "--json-format", "2", "--recursive"]
VERSION_RE = re.compile(r"^nix \(Nix\) ([0-9]+\.[0-9]+\.[0-9]+)\r?\n?$", re.ASCII)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise closure.ContractError(f"JSON object: duplicate key: {key}")
        out[key] = value
    return out


def strict_json_bytes(data: bytes, label: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise closure.ContractError(f"{label}: UTF-8 required") from exc
    return json.loads(text, object_pairs_hook=reject_duplicate_keys)


def strict_json_file(path: Path, label: str) -> Any:
    return strict_json_bytes(path.read_bytes(), label)


def derive_selection(profile_path: Path, lock_path: Path) -> dict[str, Any]:
    profile = strict_json_file(profile_path, "capsule profile")
    if not isinstance(profile, dict) or profile.get("schema") != PROFILE_SCHEMA:
        raise closure.ContractError("capsule profile: schema mismatch")
    required_profile = {"qualification_platform", "flake_selection", "nixpkgs_package"}
    if not required_profile.issubset(profile):
        raise closure.ContractError("capsule profile: required selection fields missing")
    fs = profile["flake_selection"]
    pkg = profile["nixpkgs_package"]
    if not isinstance(fs, dict) or set(fs) != {"root_input", "locked_node", "rev", "nar_hash"}:
        raise closure.ContractError("capsule profile: flake selection schema mismatch")
    if not isinstance(pkg, dict) or not isinstance(pkg.get("attribute"), str) or not pkg["attribute"]:
        raise closure.ContractError("capsule profile: package attribute required")
    platform = profile["qualification_platform"]
    if platform != "x86_64-linux":
        raise closure.ContractError("capsule profile: v1 requires x86_64-linux")

    lock = strict_json_file(lock_path, "flake.lock")
    if not isinstance(lock, dict) or set(lock) != {"nodes", "root", "version"}:
        raise closure.ContractError("flake.lock: top-level schema mismatch")
    nodes = lock["nodes"]
    root_name = lock["root"]
    if not isinstance(nodes, dict) or not isinstance(root_name, str) or root_name not in nodes:
        raise closure.ContractError("flake.lock: root node missing")
    root = nodes[root_name]
    if not isinstance(root, dict) or not isinstance(root.get("inputs"), dict):
        raise closure.ContractError("flake.lock: root inputs missing")
    root_input = fs["root_input"]
    locked_node = fs["locked_node"]
    if root["inputs"].get(root_input) != locked_node:
        raise closure.ContractError("flake.lock: root input does not select profiled locked node")
    node = nodes.get(locked_node)
    if not isinstance(node, dict) or not isinstance(node.get("locked"), dict):
        raise closure.ContractError("flake.lock: selected locked node missing")
    locked = node["locked"]
    for field in ("owner", "repo", "rev", "narHash", "type"):
        if field not in locked:
            raise closure.ContractError(f"flake.lock: selected node missing {field}")
    if locked["type"] != "github" or locked["rev"] != fs["rev"] or locked["narHash"] != fs["nar_hash"]:
        raise closure.ContractError("flake.lock: selected node identity mismatch")
    owner, repo = locked["owner"], locked["repo"]
    if not isinstance(owner, str) or not isinstance(repo, str) or not owner or not repo:
        raise closure.ContractError("flake.lock: selected GitHub owner/repo required")
    attr = pkg["attribute"]
    installable = f"github:{owner}/{repo}/{fs['rev']}#{attr}"
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
        "qualification_platform": platform,
        "installable": installable,
    }


def sri_sha256_to_hex(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("sha256-"):
        raise closure.ContractError("NAR hash: sha256 SRI required")
    payload = value[7:]
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise closure.ContractError("NAR hash: canonical base64 required") from exc
    if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != payload:
        raise closure.ContractError("NAR hash: canonical SHA-256 SRI required")
    return "sha256:" + raw.hex()


def parse_path_info_v2(data: bytes) -> list[dict[str, Any]]:
    value = strict_json_bytes(data, "path-info stdout")
    if not isinstance(value, dict) or set(value) != {"version", "storeDir", "info"}:
        raise closure.ContractError("path-info: exact v2 top-level schema required")
    if type(value["version"]) is not int or value["version"] != 2:
        raise closure.ContractError("path-info: JSON format version 2 required")
    if value["storeDir"] != STORE_DIR:
        raise closure.ContractError("path-info: /nix/store required")
    info = value["info"]
    if not isinstance(info, dict) or not info:
        raise closure.ContractError("path-info: non-empty info map required")
    entries: list[dict[str, Any]] = []
    for basename, record in info.items():
        if not isinstance(basename, str) or not basename or "/" in basename or "\n" in basename or "\r" in basename:
            raise closure.ContractError("path-info: canonical store basename required")
        if not isinstance(record, dict) or "narHash" not in record or "references" not in record:
            raise closure.ContractError(f"path-info {basename}: narHash and references required")
        refs = record["references"]
        if not isinstance(refs, list) or any(
            not isinstance(ref, str) or not ref or "/" in ref or "\n" in ref or "\r" in ref for ref in refs
        ):
            raise closure.ContractError(f"path-info {basename}: store-basename references required")
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
        raise closure.ContractError("nix --version: UTF-8 required") from exc
    match = VERSION_RE.fullmatch(text)
    if not match:
        raise closure.ContractError("nix --version: exact semantic-version output required")
    return match.group(1)


def parse_platform(data: bytes) -> str:
    try:
        value = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise closure.ContractError("platform observation: UTF-8 required") from exc
    if not value or any(ch.isspace() for ch in value):
        raise closure.ContractError("platform observation: one raw Nix system token required")
    return value


def parse_realized_root(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise closure.ContractError("nix build stdout: UTF-8 required") from exc
    lines = text.splitlines()
    if len(lines) != 1 or text not in {lines[0], lines[0] + "\n"}:
        raise closure.ContractError("nix build stdout: exactly one store path required")
    return closure.store_path(lines[0], "realized Workbench root")


def run_command(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def command_record(name: str, argv: list[str], proc: subprocess.CompletedProcess[bytes], raw_dir: Path) -> dict[str, Any]:
    out_rel = f"raw/{name}.stdout"
    err_rel = f"raw/{name}.stderr"
    (raw_dir / f"{name}.stdout").write_bytes(proc.stdout)
    (raw_dir / f"{name}.stderr").write_bytes(proc.stderr)
    return {
        "argv": argv,
        "exit_code": int(proc.returncode),
        "stdout": {"path": out_rel, "byte_length": len(proc.stdout), "sha256": digest_bytes(proc.stdout)},
        "stderr": {"path": err_rel, "byte_length": len(proc.stderr), "sha256": digest_bytes(proc.stderr)},
    }


def capture(profile_path: Path, lock_path: Path, out_dir: Path) -> int:
    selection = derive_selection(profile_path, lock_path)
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise closure.ContractError("capture output: no-overwrite destination required")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp.", dir=out_dir.parent))
    try:
        assert temp is not None
        raw_dir = temp / "raw"
        norm_dir = temp / "normalized"
        raw_dir.mkdir()
        norm_dir.mkdir()

        commands: dict[str, Any] = {}
        version_proc = run_command(NIX_VERSION_ARGV)
        commands["nix_version"] = command_record("nix-version", NIX_VERSION_ARGV, version_proc, raw_dir)

        platform_proc = run_command(PLATFORM_ARGV)
        commands["platform"] = command_record("platform", PLATFORM_ARGV, platform_proc, raw_dir)

        realize_argv = REALIZE_PREFIX + [selection["installable"]]
        realize_proc = run_command(realize_argv)
        commands["realization"] = command_record("realization", realize_argv, realize_proc, raw_dir)

        root: str | None = None
        path_proc: subprocess.CompletedProcess[bytes] | None = None
        if realize_proc.returncode == 0:
            try:
                root = parse_realized_root(realize_proc.stdout)
            except closure.ContractError:
                root = None
        if root is not None:
            path_argv = PATH_INFO_PREFIX + [root]
            path_proc = run_command(path_argv)
            commands["path_info"] = command_record("path-info", path_argv, path_proc, raw_dir)
        else:
            commands["path_info"] = None

        normalized = {
            "nix_version": None,
            "platform": None,
            "closure_identity_path": None,
            "closure_identity_sha256": None,
            "closure_digest": None,
        }
        facts = {
            "selection_revalidated": True,
            "nix_version_observed": version_proc.returncode == 0,
            "platform_observed": platform_proc.returncode == 0,
            "realization_command_observed": True,
            "realized_root_observed": root is not None,
            "path_info_observed": path_proc is not None,
            "canonical_closure_identity_compiled": False,
        }
        success = False
        if (
            version_proc.returncode == 0
            and platform_proc.returncode == 0
            and root is not None
            and path_proc is not None
            and path_proc.returncode == 0
        ):
            try:
                nix_version = parse_nix_version(version_proc.stdout)
                platform = parse_platform(platform_proc.stdout)
                if platform != selection["qualification_platform"]:
                    raise closure.ContractError("observed platform does not match profile qualification platform")
                entries = parse_path_info_v2(path_proc.stdout)
                identity = closure.compile_identity(root, entries)
                identity_bytes = canonical_json_bytes(identity) + b"\n"
                identity_rel = "normalized/closure_identity.json"
                (temp / identity_rel).write_bytes(identity_bytes)
                normalized = {
                    "nix_version": nix_version,
                    "platform": platform,
                    "closure_identity_path": identity_rel,
                    "closure_identity_sha256": digest_bytes(identity_bytes),
                    "closure_digest": identity["closure_digest"],
                }
                facts["canonical_closure_identity_compiled"] = True
                success = True
            except (closure.ContractError, json.JSONDecodeError):
                success = False

        authority = {
            "closure_capture_qualified": False,
            "workbench_execution_qualified": False,
            "transform_executed": False,
            "atlas_correctness_established": False,
            "fmq010_established": False,
            "neural_alignment_established": False,
            "consciousness_evidence": False,
        }
        receipt: dict[str, Any] = {
            "schema": SCHEMA,
            "status": STATUS_SUCCESS if success else STATUS_FAILURE,
            "selection": selection,
            "root": root,
            "implementations": {
                "capture_sha256": file_digest(Path(__file__)),
                "normalizer_sha256": file_digest(Path(closure.__file__)),
            },
            "commands": commands,
            "normalized": normalized,
            "facts": facts,
            "authority": authority,
            "raw_observation_digest": digest_bytes(canonical_json_bytes(commands)),
            "capture_digest": "",
        }
        receipt["capture_digest"] = digest_bytes(
            canonical_json_bytes({k: v for k, v in receipt.items() if k != "capture_digest"})
        )
        (temp / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
        os.replace(temp, out_dir)
        temp = None
        return 0 if success else 2
    finally:
        if temp is not None and temp.exists():
            shutil.rmtree(temp)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--flake-lock", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        return capture(args.profile, args.flake_lock, args.out)
    except (closure.ContractError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
