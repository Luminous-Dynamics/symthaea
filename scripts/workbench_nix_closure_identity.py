#!/usr/bin/env python3
"""Pure, non-authorizing Nix runtime-closure identity compiler for Workbench."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "symthaea-workbench-nix-closure-identity-v1"
ENTRY_KEYS = {"path", "nar_sha256", "references"}
IDENTITY_KEYS = {"schema", "root", "entry_count", "entries", "closure_digest"}
STORE_PATH_RE = re.compile(r"^/nix/store/[0123456789abcdfghijklmnpqrsvwxyz]{32}-[^/\r\n]+\Z")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}\Z")


class ContractError(ValueError):
    pass


def exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContractError(f"{label}: closed-world schema mismatch")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def store_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not STORE_PATH_RE.fullmatch(value):
        raise ContractError(f"{label}: canonical /nix/store path required")
    return value


def sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{label}: canonical sha256:<64 lowercase hex> required")
    return value


def positive_count(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ContractError(f"{label}: positive integer required")
    return value


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"JSON object: duplicate key: {key}")
        value[key] = item
    return value


def _canonicalize_entries(entries: Any) -> list[dict[str, Any]]:
    if not isinstance(entries, list) or not entries:
        raise ContractError("closure entries: non-empty list required")
    canonical: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(entries):
        entry = exact(raw, ENTRY_KEYS, f"closure entry {index}")
        path = store_path(entry["path"], f"closure entry {index} path")
        if path in seen_paths:
            raise ContractError(f"closure entry {index}: duplicate store path")
        seen_paths.add(path)
        nar = sha256(entry["nar_sha256"], f"closure entry {index} NAR SHA-256")
        refs_raw = entry["references"]
        if not isinstance(refs_raw, list):
            raise ContractError(f"closure entry {index}: references list required")
        refs = [store_path(ref, f"closure entry {index} reference") for ref in refs_raw]
        if len(refs) != len(set(refs)):
            raise ContractError(f"closure entry {index}: duplicate reference")
        canonical.append({"path": path, "nar_sha256": nar, "references": sorted(refs)})
    canonical.sort(key=lambda item: item["path"])
    path_set = {item["path"] for item in canonical}
    for entry in canonical:
        missing = [ref for ref in entry["references"] if ref not in path_set]
        if missing:
            raise ContractError(f"closure graph: reference outside closure from {entry['path']}: {missing[0]}")
    return canonical


def _require_exact_root_reachability(root: str, entries: list[dict[str, Any]]) -> None:
    by_path = {entry["path"]: entry for entry in entries}
    if root not in by_path:
        raise ContractError("closure root: root path not present in closure")

    reachable: set[str] = set()
    pending = [root]
    while pending:
        path = pending.pop()
        if path in reachable:
            continue
        reachable.add(path)
        pending.extend(by_path[path]["references"])

    supplied = set(by_path)
    if reachable != supplied:
        orphan = min(supplied - reachable)
        raise ContractError(f"closure graph: entry not reachable from root: {orphan}")


def _digest_payload(root: str, entries: list[dict[str, Any]]) -> str:
    return digest_bytes(canonical_json_bytes({"schema": SCHEMA, "root": root, "entries": entries}))


def compile_identity(root: Any, entries: Any) -> dict[str, Any]:
    root_path = store_path(root, "closure root")
    canonical = _canonicalize_entries(entries)
    _require_exact_root_reachability(root_path, canonical)
    return {
        "schema": SCHEMA,
        "root": root_path,
        "entry_count": len(canonical),
        "entries": canonical,
        "closure_digest": _digest_payload(root_path, canonical),
    }


def validate_identity(value: Any) -> dict[str, Any]:
    identity = exact(value, IDENTITY_KEYS, "closure identity")
    if identity["schema"] != SCHEMA:
        raise ContractError("closure identity: schema mismatch")
    root = store_path(identity["root"], "closure identity root")
    canonical = _canonicalize_entries(identity["entries"])
    if identity["entries"] != canonical:
        raise ContractError("closure identity: entries are not canonically ordered/normalized")
    _require_exact_root_reachability(root, canonical)
    count = positive_count(identity["entry_count"], "closure identity entry count")
    if count != len(canonical):
        raise ContractError("closure identity: entry count mismatch")
    stored = sha256(identity["closure_digest"], "closure identity digest")
    if stored != _digest_payload(root, canonical):
        raise ContractError("closure identity: digest mismatch")
    return identity


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_object_keys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    compile_parser = sub.add_parser("compile")
    compile_parser.add_argument("--root", required=True)
    compile_parser.add_argument("--entries", required=True, type=Path)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--identity", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "compile":
            result = compile_identity(args.root, load(args.entries))
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        else:
            result = validate_identity(load(args.identity))
            print(json.dumps({
                "schema": SCHEMA,
                "status": "validated-identity-only",
                "closure_digest": result["closure_digest"],
            }, sort_keys=True, separators=(",", ":")))
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
