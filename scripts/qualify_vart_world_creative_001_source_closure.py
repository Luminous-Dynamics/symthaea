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
V05_HEAD = "33820b3d9e904280e6264719fe7717cb2e5dd5bb"
V05_TREE = "e93c6dbfa05b602100ff924efaa5d95f92ef5a65"
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class SourceClosureError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(argv: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check and proc.returncode != 0:
        raise SourceClosureError(
            f"command failed ({proc.returncode}): {argv!r}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def git(repo: Path, *args: str) -> str:
    return run(["git", "-C", str(repo), *args]).stdout.strip()


def require_hex40(value: str, label: str) -> str:
    if HEX40.fullmatch(value) is None:
        raise SourceClosureError(f"{label} is not a 40-hex git identity: {value!r}")
    return value


def require_clean(repo: Path) -> None:
    dirty = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise SourceClosureError(f"subject source checkout is dirty: {dirty}")


def tree_of(repo: Path, commit: str) -> str:
    return git(repo, "rev-parse", f"{commit}^{{tree}}")


def require_ancestor(repo: Path, ancestor: str, descendant: str, label: str) -> None:
    proc = run(["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant], check=False)
    if proc.returncode != 0:
        raise SourceClosureError(f"{label} is not an ancestor of confirmatory source")


def lock_manifest(repo: Path, relpaths: list[str]) -> tuple[list[dict[str, str]], str]:
    if not relpaths:
        raise SourceClosureError("at least one --lock-file is required")
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for rel in relpaths:
        path = (repo / rel).resolve()
        try:
            normalized = path.relative_to(repo).as_posix()
        except ValueError as exc:
            raise SourceClosureError(f"lock file must be inside subject checkout: {rel}") from exc
        if normalized in seen:
            raise SourceClosureError(f"duplicate lock file: {normalized}")
        seen.add(normalized)
        if not path.is_file():
            raise SourceClosureError(f"missing lock/toolchain file: {normalized}")
        entries.append({"path": normalized, "sha256": sha256_file(path)})
    entries.sort(key=lambda item: item["path"])
    return entries, sha256_bytes(canonical_bytes(entries))


def qualify(
    *,
    repo: Path,
    remote_name: str,
    repository_full_name: str,
    ref: str,
    pilot_head: str,
    pilot_tree: str,
    parent_v05a_head: str,
    parent_v05a_tree: str,
    lock_files: list[str],
    environment_receipt: Path,
    qualification_receipt: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    require_clean(repo)
    head = require_hex40(git(repo, "rev-parse", "HEAD"), "subject HEAD")
    tree = require_hex40(git(repo, "rev-parse", "HEAD^{tree}"), "subject TREE")
    pilot_head = require_hex40(pilot_head, "pilot HEAD")
    pilot_tree = require_hex40(pilot_tree, "pilot TREE")
    parent_v05a_head = require_hex40(parent_v05a_head, "v0.5-A HEAD")
    parent_v05a_tree = require_hex40(parent_v05a_tree, "v0.5-A TREE")

    if parent_v05a_head != V05_HEAD or parent_v05a_tree != V05_TREE:
        raise SourceClosureError("qualified v0.5-A parent identity mismatch")
    if tree_of(repo, pilot_head) != pilot_tree:
        raise SourceClosureError("pilot predecessor TREE mismatch")
    if tree_of(repo, parent_v05a_head) != parent_v05a_tree:
        raise SourceClosureError("qualified v0.5-A TREE mismatch in subject repository")
    require_ancestor(repo, parent_v05a_head, head, "qualified v0.5-A parent")
    require_ancestor(repo, pilot_head, head, "pilot predecessor")

    remote_url = git(repo, "remote", "get-url", remote_name)
    if not remote_url:
        raise SourceClosureError(f"remote has no URL: {remote_name}")
    remote_lines = run(["git", "ls-remote", remote_url, ref]).stdout.splitlines()
    exact = [line.split() for line in remote_lines if len(line.split()) == 2 and line.split()[1] == ref]
    if len(exact) != 1:
        raise SourceClosureError(f"remote ref did not resolve uniquely: {ref}: {remote_lines!r}")
    fetched_head = require_hex40(exact[0][0], "remote fetched HEAD")
    if fetched_head != head:
        raise SourceClosureError(f"remote ref HEAD mismatch: {fetched_head} != {head}")

    with tempfile.TemporaryDirectory(prefix="vart-source-closure-") as td:
        fresh = Path(td) / "checkout"
        run(["git", "init", "--quiet", str(fresh)])
        run(["git", "-C", str(fresh), "remote", "add", "origin", remote_url])
        run(["git", "-C", str(fresh), "fetch", "--quiet", "--no-tags", "origin", ref])
        fresh_head = require_hex40(git(fresh, "rev-parse", "FETCH_HEAD"), "fresh checkout HEAD")
        fresh_tree = require_hex40(git(fresh, "rev-parse", "FETCH_HEAD^{tree}"), "fresh checkout TREE")
        if fresh_head != head or fresh_tree != tree:
            raise SourceClosureError(
                f"fresh checkout identity mismatch: {fresh_head}/{fresh_tree} != {head}/{tree}"
            )
        run(["git", "-C", str(fresh), "checkout", "--quiet", "--detach", fresh_head])
        if git(fresh, "status", "--porcelain=v1", "--untracked-files=all"):
            raise SourceClosureError("fresh checkout is unexpectedly dirty")

    if not environment_receipt.is_file():
        raise SourceClosureError(f"missing environment receipt: {environment_receipt}")
    if not qualification_receipt.is_file():
        raise SourceClosureError(f"missing qualification receipt: {qualification_receipt}")
    locks, locks_sha = lock_manifest(repo, lock_files)

    return {
        "schema": "symthaea.vart-world-creative-001.source-closure.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "qualified",
        "confirmatory_source": {
            "head": head,
            "tree": tree,
            "parent_v05a_head": parent_v05a_head,
            "parent_v05a_tree": parent_v05a_tree,
        },
        "pilot_predecessor": {
            "head": pilot_head,
            "tree": pilot_tree,
            "is_ancestor_of_confirmatory_source": True,
        },
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
        "reproduction": {
            "environment_digest": sha256_file(environment_receipt),
            "lock_manifest_sha256": locks_sha,
            "lock_manifest": locks,
            "qualification_receipt_sha256": sha256_file(qualification_receipt),
            "independent_checkout_gate": True,
        },
        "qualified_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualify VART confirmatory subject source closure")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--repository-full-name", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--pilot-head", required=True)
    parser.add_argument("--pilot-tree", required=True)
    parser.add_argument("--parent-v05a-head", default=V05_HEAD)
    parser.add_argument("--parent-v05a-tree", default=V05_TREE)
    parser.add_argument("--lock-file", action="append", dest="lock_files", required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--qualification-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        raise SourceClosureError(f"refusing to overwrite source closure receipt: {out}")
    result = qualify(
        repo=args.repo,
        remote_name=args.remote,
        repository_full_name=args.repository_full_name,
        ref=args.ref,
        pilot_head=args.pilot_head,
        pilot_tree=args.pilot_tree,
        parent_v05a_head=args.parent_v05a_head,
        parent_v05a_tree=args.parent_v05a_tree,
        lock_files=args.lock_files,
        environment_receipt=args.environment_receipt.resolve(),
        qualification_receipt=args.qualification_receipt.resolve(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(canonical_bytes(result) + b"\n")
    print(json.dumps({
        "verdict": "VART_SOURCE_CLOSURE_QUALIFIED",
        "confirmatory_source_head": result["confirmatory_source"]["head"],
        "confirmatory_source_tree": result["confirmatory_source"]["tree"],
        "source_closure_sha256": sha256_file(out),
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SourceClosureError, OSError, ValueError) as exc:
        print(json.dumps({
            "verdict": "VART_SOURCE_CLOSURE_REJECT",
            "error": str(exc),
            "confirmatory_execution_authorized": False,
            "claim_authorized": False,
        }, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
