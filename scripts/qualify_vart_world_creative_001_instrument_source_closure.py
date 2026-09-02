#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class InstrumentClosureError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InstrumentClosureError(f"missing qualification receipt: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InstrumentClosureError(f"invalid qualification receipt: {exc}") from exc


def run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check and proc.returncode != 0:
        raise InstrumentClosureError(
            f"command failed ({proc.returncode}): {argv!r}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def git(repo: Path, *args: str) -> str:
    return run(["git", "-C", str(repo), *args]).stdout.strip()


def require_hex(value: Any, length: int, label: str) -> str:
    pattern = HEX40 if length == 40 else HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise InstrumentClosureError(f"{label} must be {length}-hex")
    return value


def meaningful_status(repo: Path) -> list[str]:
    lines = git(repo, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    out: list[str] = []
    for line in lines:
        path = line[3:] if len(line) >= 4 else line
        if "/__pycache__/" in f"/{path}" or path.endswith(".pyc"):
            continue
        out.append(line)
    return out


def verify_qualification(receipt_path: Path, head: str, tree: str) -> tuple[str, str, str]:
    obj = read_json(receipt_path)
    if not isinstance(obj, dict):
        raise InstrumentClosureError("instrument qualification root must be object")
    if (
        obj.get("schema") != "symthaea.vart-world-creative-001.instrument-qualification.v1"
        or obj.get("experiment_id") != EXPERIMENT_ID
        or obj.get("status") != "qualified"
        or obj.get("all_suites_pass") is not True
    ):
        raise InstrumentClosureError("instrument qualification receipt is not qualified")
    if obj.get("confirmatory_execution_authorized") is not False or obj.get("claim_authorized") is not False:
        raise InstrumentClosureError("instrument qualification authority violation")
    source = obj.get("instrument_source")
    if not isinstance(source, dict) or source.get("head") != head or source.get("tree") != tree:
        raise InstrumentClosureError("instrument qualification source identity mismatch")
    manifest = require_hex(obj.get("instrument_manifest_sha256"), 64, "instrument_manifest_sha256")
    environment = require_hex(obj.get("instrument_environment_digest"), 64, "instrument_environment_digest")
    return sha256_file(receipt_path), manifest, environment


def qualify(
    *,
    repo: Path,
    remote_name: str,
    repository_full_name: str,
    ref: str,
    qualification_receipt: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    dirty = meaningful_status(repo)
    if dirty:
        raise InstrumentClosureError(f"instrument checkout is dirty: {dirty}")
    head = require_hex(git(repo, "rev-parse", "HEAD"), 40, "instrument HEAD")
    tree = require_hex(git(repo, "rev-parse", "HEAD^{tree}"), 40, "instrument TREE")
    qualification_sha, manifest_sha, environment_digest = verify_qualification(
        qualification_receipt, head, tree
    )

    remote_url = git(repo, "remote", "get-url", remote_name)
    if not remote_url:
        raise InstrumentClosureError(f"remote has no URL: {remote_name}")
    lines = run(["git", "ls-remote", remote_url, ref]).stdout.splitlines()
    exact = [line.split() for line in lines if len(line.split()) == 2 and line.split()[1] == ref]
    if len(exact) != 1:
        raise InstrumentClosureError(f"remote ref did not resolve uniquely: {ref}: {lines!r}")
    fetched_head = require_hex(exact[0][0], 40, "remote fetched HEAD")
    if fetched_head != head:
        raise InstrumentClosureError(f"remote ref HEAD mismatch: {fetched_head} != {head}")

    with tempfile.TemporaryDirectory(prefix="vart-instrument-closure-") as td:
        fresh = Path(td) / "checkout"
        run(["git", "init", "--quiet", str(fresh)])
        run(["git", "-C", str(fresh), "remote", "add", "origin", remote_url])
        run(["git", "-C", str(fresh), "fetch", "--quiet", "--no-tags", "origin", ref])
        fresh_head = require_hex(git(fresh, "rev-parse", "FETCH_HEAD"), 40, "fresh HEAD")
        fresh_tree = require_hex(git(fresh, "rev-parse", "FETCH_HEAD^{tree}"), 40, "fresh TREE")
        if fresh_head != head or fresh_tree != tree:
            raise InstrumentClosureError(
                f"fresh checkout identity mismatch: {fresh_head}/{fresh_tree} != {head}/{tree}"
            )
        run(["git", "-C", str(fresh), "checkout", "--quiet", "--detach", fresh_head])
        if git(fresh, "status", "--porcelain=v1", "--untracked-files=all"):
            raise InstrumentClosureError("fresh instrument checkout is dirty")

    return {
        "schema": "symthaea.vart-world-creative-001.instrument-source-closure.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "qualified",
        "instrument_source": {"head": head, "tree": tree},
        "remote": {
            "repository_full_name": repository_full_name,
            "remote_name": remote_name,
            "remote_url": remote_url,
            "ref": ref,
            "fetch_verified": True,
            "fetched_head": fetched_head,
            "fetched_tree": tree,
            "fresh_checkout_verified": True,
            "fresh_checkout_head": head,
            "fresh_checkout_tree": tree,
        },
        "qualification": {
            "instrument_qualification_receipt_sha256": qualification_sha,
            "instrument_manifest_sha256": manifest_sha,
            "instrument_environment_digest": environment_digest,
            "all_suites_pass": True,
        },
        "qualified_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualify VART instrument source closure")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--repository-full-name", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--qualification-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        raise InstrumentClosureError(f"refusing to overwrite instrument source closure: {out}")
    result = qualify(
        repo=args.repo,
        remote_name=args.remote,
        repository_full_name=args.repository_full_name,
        ref=args.ref,
        qualification_receipt=args.qualification_receipt.resolve(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(canonical_bytes(result) + b"\n")
    print(json.dumps({
        "verdict": "VART_INSTRUMENT_SOURCE_CLOSURE_QUALIFIED",
        "instrument_source_head": result["instrument_source"]["head"],
        "instrument_source_tree": result["instrument_source"]["tree"],
        "instrument_source_closure_sha256": sha256_file(out),
        "instrument_manifest_sha256": result["qualification"]["instrument_manifest_sha256"],
        "instrument_environment_digest": result["qualification"]["instrument_environment_digest"],
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InstrumentClosureError, OSError, ValueError) as exc:
        print(json.dumps({
            "verdict": "VART_INSTRUMENT_SOURCE_CLOSURE_REJECT",
            "error": str(exc),
            "confirmatory_execution_authorized": False,
            "claim_authorized": False,
        }, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
