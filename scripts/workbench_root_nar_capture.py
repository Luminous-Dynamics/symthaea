#!/usr/bin/env python3
"""Unqualified raw Workbench root-NAR observation producer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

SCHEMA = "symthaea-workbench-root-nar-capture-receipt-v1"
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


class CaptureError(ValueError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise CaptureError(f"JSON object: duplicate key: {key}")
        out[key] = value
    return out


def strict_json_file(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise CaptureError(f"{label}: UTF-8 required") from exc
    except json.JSONDecodeError as exc:
        raise CaptureError(f"{label}: invalid JSON") from exc


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


def stream_record(path: Path, rel: str) -> dict[str, Any]:
    return {"path": rel, "byte_length": path.stat().st_size, "sha256": digest_file(path)}


def canonical_store_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not STORE_PATH_RE.fullmatch(value):
        raise CaptureError(f"{label}: canonical /nix/store path required")
    return value


def canonical_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise CaptureError(f"{label}: canonical sha256:<64 lowercase hex> required")
    return value


def read_closure_subject(receipt_dir: Path) -> tuple[str, str]:
    receipt = strict_json_file(receipt_dir / "receipt.json", "closure receipt")
    if not isinstance(receipt, dict):
        raise CaptureError("closure receipt: object required")
    if receipt.get("status") != "observed-normalized-unqualified":
        raise CaptureError("closure receipt: complete normalized observation required")
    root = canonical_store_path(receipt.get("root"), "closure receipt root")
    capture_digest = canonical_sha256(receipt.get("capture_digest"), "closure receipt capture digest")
    return root, capture_digest


def default_dump_runner(argv: list[str], stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=False,
            close_fds=True,
        )
    return proc.returncode


def capture(
    closure_receipt_dir: Path,
    out_dir: Path,
    *,
    runner: Callable[[list[str], Path, Path], int] = default_dump_runner,
) -> int:
    if out_dir.exists():
        raise CaptureError("capture destination already exists; evidence is create-only")
    root, closure_capture_digest = read_closure_subject(closure_receipt_dir)
    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=parent))
    try:
        raw = temp / "raw"
        raw.mkdir(mode=0o700)
        nar_path = raw / "root.nar"
        stderr_path = raw / "nix-store-dump.stderr"
        argv = ["nix-store", "--dump", root]
        rc = runner(argv, nar_path, stderr_path)
        if type(rc) is not int:
            raise CaptureError("dump runner: integer exit code required")

        nar_record = stream_record(nar_path, "raw/root.nar")
        stderr_record = stream_record(stderr_path, "raw/nix-store-dump.stderr")
        success = rc == 0
        authority = {key: False for key in AUTHORITY_KEYS}
        receipt: dict[str, Any] = {
            "schema": SCHEMA,
            "status": STATUS_SUCCESS if success else STATUS_FAILURE,
            "closure_capture_digest": closure_capture_digest,
            "root": root,
            "implementation_sha256": digest_file(Path(__file__)),
            "command": {"argv": argv, "exit_code": rc},
            "nar": nar_record,
            "stderr": stderr_record,
            "authority": authority,
            "capture_digest": "",
        }
        receipt["capture_digest"] = digest_bytes(
            canonical_json_bytes({key: value for key, value in receipt.items() if key != "capture_digest"})
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
    parser.add_argument("--closure-receipt-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        return capture(args.closure_receipt_dir, args.out)
    except (CaptureError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
