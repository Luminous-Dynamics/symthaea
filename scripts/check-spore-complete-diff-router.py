#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fail-open complete-diff classifier for future Spore CI fanout routing.

This script does NOT suppress CI. It proves the classification primitive that a
future always-triggered router may use after validation ownership is established.

Security properties:
- changed paths come from the local Git object graph, never GitHub path filters;
- the diff is merge-base -> exact HEAD and uses --no-renames, so cross-boundary
  moves expose both the deleted old path and added new path;
- authorization must come from exact-head routing/ownership receipts;
- empty, oversized, malformed, ambiguous, mixed, or unauthorised diffs require
  full CI;
- only a non-empty complete diff wholly contained in authorised exact roots can
  become focused-only eligible.

The default focused-only size ceiling is intentionally conservative. It is a
routing safety policy, not a Git/GitHub limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCHEMA = "spore-complete-diff-routing-decision-v1"
AUTH_SCHEMA = "spore-ci-routing-contract-v3"
OWNERSHIP_SCHEMA = "spore-ci-path-package-ownership-v1"

DECISION_FOCUSED = "FOCUSED_ONLY_ELIGIBLE"
DECISION_FULL = "FULL_CI_REQUIRED"

MAX_FOCUSED_ONLY_FILES = 512
_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


@dataclass(frozen=True)
class Authorization:
    roots: tuple[str, ...]
    authorization_sha256: str
    ownership_sha256: str


class ClassificationError(ValueError):
    """Internal validation failure that must fail open to full CI."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_set_sha256(values: Iterable[str]) -> str:
    encoded = b"".join(value.encode("utf-8") + b"\0" for value in sorted(set(values)))
    return sha256_bytes(encoded)


def read_json_bytes(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ClassificationError(f"{path}: receipt must be a JSON object")
    return value, raw


def validate_object_id(value: str, label: str) -> None:
    if not _OBJECT_ID.fullmatch(value):
        raise ClassificationError(f"{label} is not a canonical Git object id")


def validate_repo_path(path: str) -> None:
    if not path:
        raise ClassificationError("empty changed path")
    if path.startswith("/"):
        raise ClassificationError(f"absolute changed path: {path!r}")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ClassificationError(f"non-canonical changed path: {path!r}")


def authorized_roots(patterns: Iterable[str]) -> tuple[str, ...]:
    roots: list[str] = []
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern.endswith("/**"):
            raise ClassificationError(f"unsupported authorization pattern: {pattern!r}")
        root = pattern[:-3]
        if not root or root.endswith("/"):
            raise ClassificationError(f"malformed authorization root: {pattern!r}")
        validate_repo_path(root)
        roots.append(root)
    if len(set(roots)) != len(roots):
        raise ClassificationError("duplicate authorization roots")
    return tuple(sorted(roots))


def classify_paths(
    paths: Iterable[str],
    authorization_patterns: Iterable[str],
    *,
    max_files: int = MAX_FOCUSED_ONLY_FILES,
) -> dict[str, object]:
    if max_files < 1:
        raise ClassificationError("max_files must be positive")

    unique = tuple(sorted(set(paths)))
    result: dict[str, object] = {
        "decision": DECISION_FULL,
        "changed_file_count": len(unique),
        "changed_paths_sha256": canonical_set_sha256(unique),
        "max_focused_only_files": max_files,
        "authorized_roots": [],
        "authorized_roots_sha256": canonical_set_sha256(()),
        "roots_used": [],
    }

    if not unique:
        result["reason"] = "empty-diff"
        return result
    if len(unique) > max_files:
        result["reason"] = "focused-only-size-limit-exceeded"
        return result

    try:
        for path in unique:
            validate_repo_path(path)
        roots = authorized_roots(authorization_patterns)
    except ClassificationError as error:
        result["reason"] = "malformed-routing-input"
        result["detail"] = str(error)
        return result

    result["authorized_roots"] = list(roots)
    result["authorized_roots_sha256"] = canonical_set_sha256(roots)
    if not roots:
        result["reason"] = "no-live-routing-authorization"
        return result

    used: set[str] = set()
    for path in unique:
        matches = [root for root in roots if path.startswith(root + "/")]
        if not matches:
            result["reason"] = "cross-cutting-or-unknown-path"
            result["offending_path"] = path
            return result
        if len(matches) != 1:
            result["reason"] = "ambiguous-authorized-root"
            result["offending_path"] = path
            result["matching_roots"] = matches
            return result
        used.add(matches[0])

    result["decision"] = DECISION_FOCUSED
    result["reason"] = "complete-diff-contained-in-authorized-roots"
    result["roots_used"] = sorted(used)
    return result


def run_git(repo: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise ClassificationError(f"git {' '.join(args)} failed: {stderr}")
    return proc.stdout


def exact_head_tree(repo: Path, head: str) -> str:
    actual = run_git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    if actual != head:
        raise ClassificationError(f"checked-out HEAD {actual} != requested head {head}")
    tree = run_git(repo, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    return tree


def complete_changed_paths(repo: Path, base: str, head: str) -> tuple[str, tuple[str, ...]]:
    validate_object_id(base, "base")
    validate_object_id(head, "head")

    # Verify both commit objects exist locally. A future workflow should use a
    # full-history checkout/fetch before calling this classifier.
    run_git(repo, "cat-file", "-e", f"{base}^{{commit}}")
    run_git(repo, "cat-file", "-e", f"{head}^{{commit}}")

    merge_bases = [
        line
        for line in run_git(repo, "merge-base", "--all", base, head)
        .decode("ascii")
        .splitlines()
        if line
    ]
    if len(merge_bases) != 1:
        raise ClassificationError(
            f"expected exactly one merge base, got {len(merge_bases)}"
        )
    merge_base = merge_bases[0]

    # --no-renames is a safety property: a move from cross-cutting -> authorized
    # is represented as deletion of the old path plus addition of the new path,
    # so the cross-cutting origin cannot disappear behind rename detection.
    raw = run_git(
        repo,
        "diff",
        "--name-only",
        "-z",
        "--no-renames",
        merge_base,
        head,
        "--",
    )
    items = raw.split(b"\0")
    if items and items[-1] == b"":
        items.pop()

    paths: list[str] = []
    for item in items:
        try:
            path = item.decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            raise ClassificationError(
                "non-UTF-8 changed path requires full CI"
            ) from error
        paths.append(path)
    return merge_base, tuple(paths)


def load_authorization(
    authorization_path: Path,
    ownership_path: Path,
    *,
    expected_head: str,
    expected_tree: str,
) -> Authorization:
    auth, auth_raw = read_json_bytes(authorization_path)
    ownership, ownership_raw = read_json_bytes(ownership_path)

    if auth.get("schema") != AUTH_SCHEMA or auth.get("status") != "PASS":
        raise ClassificationError("routing authorization receipt is not PASS v3")
    if auth.get("mode") != "bootstrap":
        raise ClassificationError("routing authorization must come from unfiltered bootstrap mode")
    if auth.get("native_pr_path_filter_policy") != "PROHIBITED":
        raise ClassificationError("native PR path filtering is not prohibited")
    if auth.get("future_fanout_router") != "COMPLETE_GIT_DIFF_REQUIRED":
        raise ClassificationError("authorization does not require complete-diff routing")
    if auth.get("source_commit") != expected_head:
        raise ClassificationError("authorization receipt is for a different source head")
    if auth.get("evaluated_tree") != expected_tree:
        raise ClassificationError("authorization receipt is for a different source tree")
    if auth.get("qualification_script_present") is not True:
        raise ClassificationError("live qualification authority is absent")

    audited_blobs = auth.get("audited_qualification_script_blobs")
    actual_blob = auth.get("qualification_script_blob_sha1")
    if (
        not isinstance(audited_blobs, list)
        or not all(isinstance(item, str) for item in audited_blobs)
        or not isinstance(actual_blob, str)
        or actual_blob not in audited_blobs
    ):
        raise ClassificationError("qualification Git blob is not in the audited authority set")

    audited_sha256 = auth.get("audited_qualification_script_sha256")
    actual_sha256 = auth.get("qualification_script_sha256")
    if not isinstance(audited_sha256, str) or not audited_sha256:
        raise ClassificationError("audited qualification SHA-256 is absent")
    if actual_sha256 != audited_sha256:
        raise ClassificationError("qualification SHA-256 is not the audited authority")

    patterns = auth.get("authorized_boot_only")
    if not isinstance(patterns, list) or not patterns or not all(
        isinstance(item, str) for item in patterns
    ):
        raise ClassificationError("authorization receipt has no usable authorized roots")
    candidates = auth.get("candidate_boot_only")
    if not isinstance(candidates, list) or not all(isinstance(item, str) for item in candidates):
        raise ClassificationError("authorization receipt lacks candidate root set")
    if not set(patterns).issubset(set(candidates)):
        raise ClassificationError("authorized roots are not a subset of candidate roots")

    if auth.get("path_package_ownership_status") != "PASS":
        raise ClassificationError("authorization receipt does not bind PASS ownership")

    expected_ownership_hash = auth.get("path_package_ownership_sha256")
    actual_ownership_hash = sha256_bytes(ownership_raw)
    if expected_ownership_hash != actual_ownership_hash:
        raise ClassificationError("ownership receipt hash does not match authorization")

    if ownership.get("schema") != OWNERSHIP_SCHEMA or ownership.get("status") != "PASS":
        raise ClassificationError("path/package ownership receipt is not PASS v1")
    if ownership.get("qualification_script_present") is not True:
        raise ClassificationError("ownership receipt was not produced with live qualification authority")
    owners = ownership.get("owners")
    if not isinstance(owners, dict):
        raise ClassificationError("ownership receipt lacks owners")
    if set(owners) != set(patterns):
        raise ClassificationError("ownership roots do not equal authorized roots")

    # Validate pattern shape now, before classification.
    authorized_roots(patterns)

    return Authorization(
        roots=tuple(patterns),
        authorization_sha256=sha256_bytes(auth_raw),
        ownership_sha256=actual_ownership_hash,
    )


def decision_from_git(
    repo: Path,
    base: str,
    head: str,
    authorization_path: Path,
    ownership_path: Path,
    *,
    max_files: int = MAX_FOCUSED_ONLY_FILES,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "status": "PASS",
        "decision": DECISION_FULL,
        "base_commit": base,
        "head_commit": head,
        "max_focused_only_files": max_files,
    }
    try:
        tree = exact_head_tree(repo, head)
        receipt["head_tree"] = tree
        merge_base, paths = complete_changed_paths(repo, base, head)
        receipt["merge_base"] = merge_base

        authorization = load_authorization(
            authorization_path,
            ownership_path,
            expected_head=head,
            expected_tree=tree,
        )
        receipt["authorization_receipt_sha256"] = authorization.authorization_sha256
        receipt["ownership_receipt_sha256"] = authorization.ownership_sha256

        classification = classify_paths(
            paths,
            authorization.roots,
            max_files=max_files,
        )
        receipt.update(classification)
    except (OSError, UnicodeError, json.JSONDecodeError, ClassificationError) as error:
        receipt["decision"] = DECISION_FULL
        receipt["reason"] = "router-validation-failed"
        receipt["detail"] = str(error)

    return receipt


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def self_test() -> None:
    allowed = (
        "crates/domains/symthaea-boot-protocol/**",
        "crates/domains/symthaea-quicken-fb/**",
    )

    assert classify_paths([], allowed)["decision"] == DECISION_FULL
    assert classify_paths(
        ["crates/domains/symthaea-boot-protocol/src/lib.rs"], ()
    )["reason"] == "no-live-routing-authorization"

    one = classify_paths(
        ["crates/domains/symthaea-boot-protocol/src/lib.rs"], allowed
    )
    assert one["decision"] == DECISION_FOCUSED

    multi = classify_paths(
        [
            "crates/domains/symthaea-boot-protocol/src/lib.rs",
            "crates/domains/symthaea-quicken-fb/src/main.rs",
        ],
        allowed,
    )
    assert multi["decision"] == DECISION_FOCUSED

    mixed = classify_paths(
        [
            "crates/domains/symthaea-boot-protocol/src/lib.rs",
            "Cargo.toml",
        ],
        allowed,
    )
    assert mixed["decision"] == DECISION_FULL
    assert mixed["offending_path"] == "Cargo.toml"

    traversal = classify_paths(
        ["crates/domains/symthaea-boot-protocol/../Cargo.toml"], allowed
    )
    assert traversal["decision"] == DECISION_FULL

    overlapping = classify_paths(
        ["crates/domains/symthaea-boot-protocol/src/lib.rs"],
        (
            "crates/domains/**",
            "crates/domains/symthaea-boot-protocol/**",
        ),
    )
    assert overlapping["reason"] == "ambiguous-authorized-root"

    too_many = classify_paths(
        [f"crates/domains/symthaea-boot-protocol/src/v{i}.rs" for i in range(3)],
        allowed,
        max_files=2,
    )
    assert too_many["reason"] == "focused-only-size-limit-exceeded"

    # Receipt-binding proof. A valid exact-head authorization can classify a
    # complete boot-only diff, while stale head/tree or tampered ownership bytes
    # fail open to full CI.
    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory)
        run_git(repo, "init", "-q")
        run_git(repo, "config", "user.email", "ci@example.invalid")
        run_git(repo, "config", "user.name", "CI")
        target = repo / "crates/domains/symthaea-boot-protocol/src"
        target.mkdir(parents=True)
        (target / "lib.rs").write_text("pub fn v() {}\n", encoding="utf-8")
        run_git(repo, "add", ".")
        run_git(repo, "commit", "-q", "-m", "base")
        base = run_git(repo, "rev-parse", "HEAD").decode("ascii").strip()
        (target / "lib.rs").write_text("pub fn v() { let _ = 1; }\n", encoding="utf-8")
        run_git(repo, "add", ".")
        run_git(repo, "commit", "-q", "-m", "head")
        head = run_git(repo, "rev-parse", "HEAD").decode("ascii").strip()
        tree = run_git(repo, "rev-parse", "HEAD^{tree}").decode("ascii").strip()

        ownership = {
            "schema": OWNERSHIP_SCHEMA,
            "status": "PASS",
            "qualification_script_present": True,
            "owners": {
                allowed[0]: {
                    "manifest": "crates/domains/symthaea-boot-protocol/Cargo.toml",
                    "manifest_sha256": "0" * 64,
                    "package": "symthaea-boot-protocol",
                }
            },
        }
        ownership_path = repo / "ownership.json"
        write_json(ownership_path, ownership)
        ownership_hash = sha256_bytes(ownership_path.read_bytes())

        auth = {
            "schema": AUTH_SCHEMA,
            "status": "PASS",
            "mode": "bootstrap",
            "native_pr_path_filter_policy": "PROHIBITED",
            "future_fanout_router": "COMPLETE_GIT_DIFF_REQUIRED",
            "source_commit": head,
            "evaluated_tree": tree,
            "qualification_script_present": True,
            "qualification_script_blob_sha1": "1" * 40,
            "audited_qualification_script_blobs": ["1" * 40],
            "qualification_script_sha256": "2" * 64,
            "audited_qualification_script_sha256": "2" * 64,
            "candidate_boot_only": [allowed[0]],
            "authorized_boot_only": [allowed[0]],
            "path_package_ownership_status": "PASS",
            "path_package_ownership_sha256": ownership_hash,
        }
        auth_path = repo / "auth.json"
        write_json(auth_path, auth)

        decision = decision_from_git(repo, base, head, auth_path, ownership_path)
        assert decision["decision"] == DECISION_FOCUSED

        stale = dict(auth)
        stale["source_commit"] = "0" * 40
        write_json(auth_path, stale)
        decision = decision_from_git(repo, base, head, auth_path, ownership_path)
        assert decision["decision"] == DECISION_FULL
        assert decision["reason"] == "router-validation-failed"

        write_json(auth_path, auth)
        ownership["owners"][allowed[0]]["package"] = "tampered"
        write_json(ownership_path, ownership)
        decision = decision_from_git(repo, base, head, auth_path, ownership_path)
        assert decision["decision"] == DECISION_FULL
        assert decision["reason"] == "router-validation-failed"

    # Integration proof for rename safety. With --no-renames a move from a
    # cross-cutting location into an authorized root must expose BOTH paths.
    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory)
        run_git(repo, "init", "-q")
        run_git(repo, "config", "user.email", "ci@example.invalid")
        run_git(repo, "config", "user.name", "CI")
        (repo / "shared.txt").write_text("x\n", encoding="utf-8")
        run_git(repo, "add", "shared.txt")
        run_git(repo, "commit", "-q", "-m", "base")
        base = run_git(repo, "rev-parse", "HEAD").decode("ascii").strip()

        destination = repo / "crates/domains/symthaea-boot-protocol/src"
        destination.mkdir(parents=True)
        run_git(
            repo,
            "mv",
            "shared.txt",
            "crates/domains/symthaea-boot-protocol/src/moved.txt",
        )
        run_git(repo, "commit", "-q", "-m", "move")
        head = run_git(repo, "rev-parse", "HEAD").decode("ascii").strip()

        merge_base, paths = complete_changed_paths(repo, base, head)
        assert merge_base == base
        assert set(paths) == {
            "shared.txt",
            "crates/domains/symthaea-boot-protocol/src/moved.txt",
        }
        result = classify_paths(paths, allowed)
        assert result["decision"] == DECISION_FULL
        assert result["offending_path"] == "shared.txt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--authorization-receipt", type=Path)
    parser.add_argument("--ownership-receipt", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--max-focused-only-files",
        type=int,
        default=MAX_FOCUSED_ONLY_FILES,
    )
    args = parser.parse_args()

    try:
        if args.self_test:
            self_test()
            print("spore-complete-diff-router: self-test PASS")
            return 0

        required = {
            "--base": args.base,
            "--head": args.head,
            "--authorization-receipt": args.authorization_receipt,
            "--ownership-receipt": args.ownership_receipt,
            "--receipt": args.receipt,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ClassificationError(
                "missing required arguments: " + ", ".join(missing)
            )

        decision = decision_from_git(
            args.repo,
            args.base,
            args.head,
            args.authorization_receipt,
            args.ownership_receipt,
            max_files=args.max_focused_only_files,
        )
        write_json(args.receipt, decision)
        sys.stdout.write(json.dumps(decision, sort_keys=True, indent=2) + "\n")
        return 0
    except (OSError, ClassificationError, AssertionError) as error:
        print(f"spore-complete-diff-router: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
