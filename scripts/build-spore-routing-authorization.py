#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build exact-head Spore routing authorization using trusted-base policy code.

This module is intended to be EXECUTED FROM THE TRUSTED BASE checkout while the
pull-request head is supplied only as a Git worktree/data source. It never imports
or executes Python/shell/workflow code from the untrusted source root.

Security properties:
- trusted validation policy comes from the sibling base-branch checker module;
- head commit/tree identity is explicit and exact;
- every head-controlled policy input is required to be a regular Git blob, never
  a symlink/submodule/tree;
- the focused/general workflow and qualification script are parsed as text/data;
- qualification authority is dual-bound by audited Git blob SHA-1 and SHA-256;
- candidate Cargo manifests are regular Git blobs whose [package].name must match
  the trusted validation-owner map;
- receipts are exact-head/tree-bound and suitable as input to the separate
  complete-diff classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

AUTH_SCHEMA = "spore-ci-routing-contract-v3"
OWNERSHIP_SCHEMA = "spore-ci-path-package-ownership-v1"
AUDITED_QUALIFICATION_SCRIPT_SHA256 = (
    "b19c1d6816bacd51d6948171be749b810b6e94871481b2c51e49a1bc9a132efc"
)
_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
REGULAR_BLOB_MODES = {"100644", "100755"}


class AuthorizationError(ValueError):
    """A routing trust precondition failed."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_git(repo: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise AuthorizationError(
            f"git {' '.join(args)} failed: "
            + proc.stderr.decode("utf-8", "replace").strip()
        )
    return proc.stdout


def validate_object_id(value: str, label: str) -> None:
    if not _OBJECT_ID.fullmatch(value):
        raise AuthorizationError(f"{label} is not a canonical Git object id")


def parse_ls_tree(raw: bytes, expected_path: str) -> tuple[str, str, str]:
    """Return (mode, type, object_id) for exactly one Git path."""
    records = raw.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    if len(records) != 1:
        raise AuthorizationError(
            f"expected exactly one tree entry for {expected_path!r}, got {len(records)}"
        )
    try:
        meta, path_raw = records[0].split(b"\t", 1)
        mode_raw, type_raw, object_raw = meta.split(b" ", 2)
        path = path_raw.decode("utf-8", "strict")
        mode = mode_raw.decode("ascii")
        kind = type_raw.decode("ascii")
        object_id = object_raw.decode("ascii")
    except (ValueError, UnicodeError) as error:
        raise AuthorizationError(f"malformed git ls-tree entry for {expected_path!r}") from error
    if path != expected_path:
        raise AuthorizationError(
            f"tree entry path mismatch: expected {expected_path!r}, got {path!r}"
        )
    validate_object_id(object_id, f"object id for {expected_path}")
    return mode, kind, object_id


def tree_entry(repo: Path, head: str, path: str) -> tuple[str, str, str] | None:
    raw = run_git(repo, "ls-tree", "-z", head, "--", path)
    if not raw:
        return None
    return parse_ls_tree(raw, path)


def regular_blob_bytes(
    repo: Path,
    head: str,
    path: str,
    *,
    required: bool = True,
) -> tuple[bytes, str, str] | None:
    entry = tree_entry(repo, head, path)
    if entry is None:
        if required:
            raise AuthorizationError(f"missing required Git path: {path}")
        return None
    mode, kind, object_id = entry
    if kind != "blob" or mode not in REGULAR_BLOB_MODES:
        raise AuthorizationError(
            f"{path}: expected regular blob mode, got mode={mode!r} type={kind!r}"
        )
    raw = run_git(repo, "cat-file", "blob", object_id)
    return raw, mode, object_id


def load_trusted_checker():
    checker = Path(__file__).resolve().with_name("check-spore-ci-routing-contract.py")
    spec = importlib.util.spec_from_file_location("trusted_spore_ci_routing", checker)
    if spec is None or spec.loader is None:
        raise AuthorizationError(f"cannot load trusted checker: {checker}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_authorization(
    source_root: Path,
    head: str,
) -> tuple[dict[str, object], dict[str, object]]:
    validate_object_id(head, "head")
    actual_head = run_git(source_root, "rev-parse", "HEAD").decode("ascii").strip()
    if actual_head != head:
        raise AuthorizationError(f"source HEAD {actual_head} != requested head {head}")
    tree = run_git(source_root, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    validate_object_id(tree, "head tree")

    trusted = load_trusted_checker()

    focused_path = ".github/workflows/spore-boot-stack.yml"
    general_path = ".github/workflows/ci.yml"
    qualification_path = "scripts/check-spore-boot-stack.sh"

    focused_entry = regular_blob_bytes(source_root, head, focused_path)
    general_entry = regular_blob_bytes(source_root, head, general_path)
    qualification_entry = regular_blob_bytes(
        source_root, head, qualification_path, required=False
    )
    assert focused_entry is not None and general_entry is not None

    focused_raw, focused_mode, focused_object = focused_entry
    general_raw, general_mode, general_object = general_entry
    try:
        focused_text = focused_raw.decode("utf-8", "strict")
        general_text = general_raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise AuthorizationError("routing workflow input is not UTF-8") from error

    qualification_text: str | None = None
    qualification_blob: str | None = None
    qualification_sha256: str | None = None
    qualification_mode: str | None = None
    if qualification_entry is not None:
        qualification_raw, qualification_mode, qualification_blob = qualification_entry
        try:
            qualification_text = qualification_raw.decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            raise AuthorizationError("qualification script is not UTF-8") from error
        qualification_sha256 = sha256_bytes(qualification_raw)
        if qualification_sha256 != AUDITED_QUALIFICATION_SCRIPT_SHA256:
            raise AuthorizationError(
                "qualification script SHA-256 is not the audited routing authority"
            )
        recomputed_blob = trusted.git_blob_sha1(qualification_raw)
        if recomputed_blob != qualification_blob:
            raise AuthorizationError(
                "qualification Git object identity does not match its exact bytes"
            )

    # Trusted BASE policy code evaluates untrusted HEAD text as inert data.
    auth = trusted.validate_texts(
        focused_text,
        general_text,
        qualification_text,
        qualification_blob,
        "bootstrap",
    )
    if auth.get("schema") != AUTH_SCHEMA or auth.get("status") != "PASS":
        raise AuthorizationError("trusted routing policy did not produce PASS v3")

    auth["source_commit"] = head
    auth["evaluated_tree"] = tree
    auth["native_pr_path_filter_policy"] = "PROHIBITED"
    auth["future_fanout_router"] = "COMPLETE_GIT_DIFF_REQUIRED"
    auth["audited_qualification_script_sha256"] = AUDITED_QUALIFICATION_SCRIPT_SHA256
    auth["focused_workflow_object"] = focused_object
    auth["focused_workflow_mode"] = focused_mode
    auth["focused_workflow_sha256"] = sha256_bytes(focused_raw)
    auth["general_workflow_object"] = general_object
    auth["general_workflow_mode"] = general_mode
    auth["general_workflow_sha256"] = sha256_bytes(general_raw)
    auth["qualification_script_mode"] = qualification_mode
    auth["qualification_script_sha256"] = qualification_sha256

    owners: dict[str, object] = {}
    if qualification_entry is None:
        ownership_status = "NOT_APPLICABLE_PREBOOT"
        if auth.get("authorized_boot_only") != []:
            raise AuthorizationError("authorization exists without live qualification script")
    else:
        ownership_status = "PASS"
        authorized = auth.get("authorized_boot_only")
        if not isinstance(authorized, list) or not authorized:
            raise AuthorizationError("live qualification authority authorized no roots")
        for pattern in authorized:
            if not isinstance(pattern, str) or not pattern.endswith("/**"):
                raise AuthorizationError(f"unsupported authorization pattern: {pattern!r}")
            owner = trusted.VALIDATION_OWNERS.get(pattern)
            if owner is None:
                raise AuthorizationError(f"no trusted owner mapping for {pattern}")
            _trigger, expected_package = owner
            manifest_path = pattern[:-3] + "/Cargo.toml"
            manifest_entry = regular_blob_bytes(source_root, head, manifest_path)
            assert manifest_entry is not None
            manifest_raw, manifest_mode, manifest_object = manifest_entry
            try:
                parsed = tomllib.loads(manifest_raw.decode("utf-8", "strict"))
            except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
                raise AuthorizationError(f"invalid Cargo manifest: {manifest_path}") from error
            actual_package = parsed.get("package", {}).get("name")
            if actual_package != expected_package:
                raise AuthorizationError(
                    f"{manifest_path}: expected package {expected_package!r}, "
                    f"got {actual_package!r}"
                )
            owners[pattern] = {
                "manifest": manifest_path,
                "manifest_mode": manifest_mode,
                "manifest_object": manifest_object,
                "manifest_sha256": sha256_bytes(manifest_raw),
                "package": actual_package,
            }

    ownership: dict[str, object] = {
        "schema": OWNERSHIP_SCHEMA,
        "status": ownership_status,
        "qualification_script_present": qualification_entry is not None,
        "source_commit": head,
        "evaluated_tree": tree,
        "owners": owners,
    }
    ownership_raw = (json.dumps(ownership, sort_keys=True, indent=2) + "\n").encode("utf-8")
    auth["path_package_ownership_status"] = ownership_status
    auth["path_package_ownership_sha256"] = sha256_bytes(ownership_raw)
    return auth, ownership


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def self_test() -> None:
    # Git-tree mode theorem: regular blobs are accepted, symlinks are rejected.
    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory)
        run_git(repo, "init", "-q")
        run_git(repo, "config", "user.email", "ci@example.invalid")
        run_git(repo, "config", "user.name", "CI")
        (repo / "regular.txt").write_text("ok\n", encoding="utf-8")
        (repo / "target.txt").write_text("target\n", encoding="utf-8")
        (repo / "link.txt").symlink_to("target.txt")
        run_git(repo, "add", ".")
        run_git(repo, "commit", "-q", "-m", "fixture")
        head = run_git(repo, "rev-parse", "HEAD").decode("ascii").strip()

        regular = regular_blob_bytes(repo, head, "regular.txt")
        assert regular is not None and regular[0] == b"ok\n"
        try:
            regular_blob_bytes(repo, head, "link.txt")
        except AuthorizationError as error:
            assert "regular blob mode" in str(error)
        else:
            raise AssertionError("symlink was accepted as routing authority input")

        missing = regular_blob_bytes(repo, head, "missing.txt", required=False)
        assert missing is None

    print("spore-routing-authorization-builder: self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--head")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--ownership-receipt", type=Path)
    args = parser.parse_args()

    try:
        if args.self_test:
            self_test()
            return 0
        required = {
            "--source-root": args.source_root,
            "--head": args.head,
            "--receipt": args.receipt,
            "--ownership-receipt": args.ownership_receipt,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise AuthorizationError("missing required arguments: " + ", ".join(missing))

        auth, ownership = build_authorization(args.source_root, args.head)
        write_json(args.ownership_receipt, ownership)
        # Recompute the ownership hash from the exact serialized bytes emitted.
        auth["path_package_ownership_sha256"] = sha256_bytes(
            args.ownership_receipt.read_bytes()
        )
        write_json(args.receipt, auth)
        print(json.dumps(auth, sort_keys=True, indent=2))
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError, AuthorizationError, AssertionError) as error:
        print(f"spore-routing-authorization-builder: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
