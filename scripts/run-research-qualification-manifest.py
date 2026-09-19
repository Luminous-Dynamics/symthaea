#!/usr/bin/env python3
"""Trusted dual-checkout adapter for research bootstrap qualification.

The adapter itself must execute from the trusted default-branch checkout. It
validates a data-only manifest in a separate candidate checkout and invokes the
trusted portable harness against that candidate working tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SCHEMA = "symthaea.research-qualifier-manifest.v1"
BINDING_SCHEMA = "symthaea.research-qualifier-binding.v2"
HARNESS_RECEIPT_SCHEMA = "symthaea.research-bootstrap-receipt.v2"
PROGRAM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
PATH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUST_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MANIFEST_ROOT = PurePosixPath(".github/research-qualifiers")
WORKFLOW_PATH = ".github/workflows/research-bootstrap-qualification.yml"
ADAPTER_PATH = "scripts/run-research-qualification-manifest.py"
HARNESS_PATH = "scripts/qualify-research-crate.sh"

LIST_RECEIPT_KEYS = frozenset({"package", "source_path", "source_object", "qualifier_path"})
SINGLETON_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "program",
        "authority",
        "scientific_claim",
        "source_subject_sha",
        "qualifier_sha",
        "qualifier_parent_count",
        "harness_sha256",
        "rustc_version",
        "cargo_version",
        "rustfmt_version",
        "clippy_version",
        "rust_host_triple",
        "generated_lock_sha256",
        "lock_patch_sha256",
        "scope_gate",
        "source_immutable_gate",
        "lock_additive_gate",
        "cargo_check_gate",
        "rustfmt_gate",
        "cargo_test_gate",
        "clippy_gate",
        "postflight_worktree_immutable_gate",
    }
)
ALLOWED_RECEIPT_KEYS = SINGLETON_RECEIPT_KEYS | LIST_RECEIPT_KEYS
GATE_KEYS = frozenset(
    {
        "scope_gate",
        "source_immutable_gate",
        "lock_additive_gate",
        "cargo_check_gate",
        "rustfmt_gate",
        "cargo_test_gate",
        "clippy_gate",
        "postflight_worktree_immutable_gate",
    }
)


class QualificationError(RuntimeError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def parse_json(raw: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise QualificationError(f"invalid JSON: {error}") from error


def canonical_repo_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise QualificationError(f"{label} must be a non-empty bounded string")
    if not PATH_RE.fullmatch(value) or ":" in value:
        raise QualificationError(f"{label} contains unsupported path characters: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith("./")
        or value.endswith("/")
        or "//" in value
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise QualificationError(f"{label} must be canonical repository-relative path: {value!r}")
    return value


def string_list(value: Any, label: str, pattern: re.Pattern[str] | None = None) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 32:
        raise QualificationError(f"{label} must contain 1..32 entries")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise QualificationError(f"every {label} entry must be a non-empty string")
        if pattern is not None and not pattern.fullmatch(item):
            raise QualificationError(f"invalid {label} entry: {item!r}")
        result.append(item)
    if len(set(result)) != len(result):
        raise QualificationError(f"{label} must not contain duplicates")
    return result


def validate_manifest(payload: Any, manifest_path: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise QualificationError("manifest root must be an object")
    allowed = {"schema", "program", "source_parent", "expected_rust", "packages", "source_paths"}
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown:
        raise QualificationError(f"unknown manifest fields: {', '.join(unknown)}")
    if missing:
        raise QualificationError(f"missing manifest fields: {', '.join(missing)}")
    if payload["schema"] != SCHEMA:
        raise QualificationError(f"schema must equal {SCHEMA!r}")

    program = payload["program"]
    source_parent = payload["source_parent"]
    expected_rust = payload["expected_rust"]
    if not isinstance(program, str) or not PROGRAM_RE.fullmatch(program):
        raise QualificationError("invalid program identity")
    if not isinstance(source_parent, str) or not SHA40_RE.fullmatch(source_parent):
        raise QualificationError("source_parent must be exactly 40 lowercase hex characters")
    if not isinstance(expected_rust, str) or not RUST_RE.fullmatch(expected_rust):
        raise QualificationError("expected_rust must have x.y.z form")

    packages = string_list(payload["packages"], "packages", PACKAGE_RE)
    source_paths = [
        canonical_repo_path(item, "source_paths")
        for item in string_list(payload["source_paths"], "source_paths")
    ]

    manifest_path = canonical_repo_path(manifest_path, "manifest path")
    path = PurePosixPath(manifest_path)
    if path.parent != MANIFEST_ROOT:
        raise QualificationError(f"manifest must live directly under {MANIFEST_ROOT}/")
    if path.name != f"{program}.json":
        raise QualificationError(f"manifest filename must equal {program}.json")

    return {
        "schema": SCHEMA,
        "program": program,
        "source_parent": source_parent,
        "expected_rust": expected_rust,
        "packages": packages,
        "source_paths": source_paths,
    }


def load_manifest(candidate_root: Path, manifest_path: str) -> dict[str, Any]:
    relative = canonical_repo_path(manifest_path, "manifest path")
    path = candidate_root / relative
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise QualificationError(f"manifest not found: {relative}") from error
    return validate_manifest(parse_json(raw), relative)


def canonical_manifest_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise QualificationError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def resolve_root(value: str, label: str) -> Path:
    path = Path(value).resolve(strict=True)
    if not path.is_dir():
        raise QualificationError(f"{label} is not a directory: {path}")
    return path


def assert_separate_roots(trusted_root: Path, candidate_root: Path) -> None:
    if trusted_root == candidate_root:
        raise QualificationError("trusted and candidate roots must be distinct")
    if trusted_root in candidate_root.parents or candidate_root in trusted_root.parents:
        raise QualificationError("trusted and candidate roots must not contain one another")


def verify_trusted_checkout(trusted_root: Path, expected_policy_sha: str) -> dict[str, str]:
    if not SHA40_RE.fullmatch(expected_policy_sha):
        raise QualificationError("trusted policy SHA must be 40 lowercase hex characters")
    actual = git(trusted_root, "rev-parse", "HEAD")
    if actual != expected_policy_sha:
        raise QualificationError(f"trusted checkout mismatch: expected {expected_policy_sha}, observed {actual}")

    expected_adapter = (trusted_root / ADAPTER_PATH).resolve()
    if Path(__file__).resolve() != expected_adapter:
        raise QualificationError("adapter is not executing from the trusted checkout")

    identities: dict[str, str] = {"trusted_policy_sha": actual}
    for key, relative in (
        ("trusted_workflow_sha256", WORKFLOW_PATH),
        ("trusted_adapter_sha256", ADAPTER_PATH),
        ("trusted_harness_sha256", HARNESS_PATH),
    ):
        path = trusted_root / relative
        if not path.is_file():
            raise QualificationError(f"trusted policy file missing: {relative}")
        identities[key] = sha256_file(path)
    return identities


def verify_candidate_subject(
    candidate_root: Path,
    manifest: dict[str, Any],
    manifest_path: str,
    expected_head: str,
    expected_base: str,
) -> None:
    if not SHA40_RE.fullmatch(expected_head) or not SHA40_RE.fullmatch(expected_base):
        raise QualificationError("candidate head/base must be 40 lowercase hex characters")
    actual_head = git(candidate_root, "rev-parse", "HEAD")
    if actual_head != expected_head:
        raise QualificationError(f"candidate HEAD mismatch: expected {expected_head}, observed {actual_head}")

    parents = git(candidate_root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if parents != [expected_head, expected_base]:
        raise QualificationError(f"candidate must have exactly sole parent {expected_base}; observed {parents[1:]}")
    if manifest["source_parent"] != expected_base:
        raise QualificationError("manifest source_parent must equal the candidate sole parent")

    changed = sorted(
        item
        for item in git(candidate_root, "diff", "--name-only", expected_base, expected_head).splitlines()
        if item
    )
    if changed != [manifest_path]:
        raise QualificationError(f"candidate diff must contain exactly {manifest_path!r}; observed {changed!r}")

    for source_path in manifest["source_paths"]:
        base_obj = git(candidate_root, "rev-parse", f"{expected_base}:{source_path}")
        head_obj = git(candidate_root, "rev-parse", f"{expected_head}:{source_path}")
        if base_obj != head_obj:
            raise QualificationError(f"source path changed in qualifier commit: {source_path}")


def fetch_live_pr(repo: str, pr_number: int, token: str) -> dict[str, Any]:
    if not REPO_RE.fullmatch(repo):
        raise QualificationError("repository must have owner/name form")
    if pr_number <= 0:
        raise QualificationError("PR number must be positive")
    if not token:
        raise QualificationError("GITHUB_TOKEN is required for live PR verification")
    request = Request(
        f"https://api.github.com/repos/{repo}/pulls/{pr_number}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "symthaea-research-qualification-adapter-v2",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        raise QualificationError(f"live PR verification failed: {error}") from error
    if not isinstance(payload, dict):
        raise QualificationError("live PR response was not an object")
    return payload


def verify_live_pr(
    repo: str,
    pr_number: int,
    token: str,
    expected_head: str,
    expected_base: str,
) -> None:
    pr = fetch_live_pr(repo, pr_number, token)
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
    base_repo = base.get("repo") if isinstance(base.get("repo"), dict) else {}
    checks = {
        "state=open": pr.get("state") == "open",
        "draft=false": pr.get("draft") is False,
        "head.sha": head.get("sha") == expected_head,
        "base.sha": base.get("sha") == expected_base,
        "same head repo": head_repo.get("full_name") == repo,
        "same base repo": base_repo.get("full_name") == repo,
    }
    failed = [label for label, ok in checks.items() if not ok]
    if failed:
        raise QualificationError(f"live PR state no longer matches frozen subject: {', '.join(failed)}")


def parse_receipt(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    if not path.is_file():
        raise QualificationError("harness returned success without receipt.txt")
    scalars: dict[str, str] = {}
    lists: dict[str, list[str]] = {key: [] for key in LIST_RECEIPT_KEYS}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line or "=" not in raw_line:
            raise QualificationError(f"malformed receipt line {line_no}")
        key, value = raw_line.split("=", 1)
        if not KEY_RE.fullmatch(key) or key not in ALLOWED_RECEIPT_KEYS:
            raise QualificationError(f"unknown receipt key on line {line_no}: {key!r}")
        if not value or "\r" in value or "\n" in value:
            raise QualificationError(f"invalid receipt value for {key!r}")
        if key in LIST_RECEIPT_KEYS:
            if value in lists[key]:
                raise QualificationError(f"duplicate receipt list value for {key!r}: {value!r}")
            lists[key].append(value)
        else:
            if key in scalars:
                raise QualificationError(f"duplicate singleton receipt key: {key!r}")
            scalars[key] = value
    missing = sorted(SINGLETON_RECEIPT_KEYS - set(scalars))
    if missing:
        raise QualificationError(f"receipt missing singleton keys: {', '.join(missing)}")
    return scalars, lists


def validate_receipt(
    receipt_path: Path,
    manifest: dict[str, Any],
    manifest_path: str,
    expected_head: str,
    expected_base: str,
    expected_harness_sha256: str,
) -> None:
    scalars, lists = parse_receipt(receipt_path)
    exact = {
        "schema": HARNESS_RECEIPT_SCHEMA,
        "program": manifest["program"],
        "authority": "bootstrap-source-qualification-only",
        "scientific_claim": "NONE",
        "source_subject_sha": expected_base,
        "qualifier_sha": expected_head,
        "qualifier_parent_count": "1",
        "harness_sha256": expected_harness_sha256,
    }
    for key, expected in exact.items():
        if scalars[key] != expected:
            raise QualificationError(f"receipt {key} mismatch: expected {expected!r}, observed {scalars[key]!r}")
    for key in GATE_KEYS:
        if scalars[key] != "PASS":
            raise QualificationError(f"receipt gate {key} did not PASS")
    for key in ("generated_lock_sha256", "lock_patch_sha256"):
        if not SHA256_RE.fullmatch(scalars[key]):
            raise QualificationError(f"receipt {key} is not a SHA-256 digest")
    for key in ("rustc_version", "cargo_version", "rustfmt_version", "clippy_version", "rust_host_triple"):
        if not scalars[key].strip():
            raise QualificationError(f"receipt {key} is empty")

    if lists["package"] != manifest["packages"]:
        raise QualificationError("receipt package list does not equal manifest package list")
    if lists["source_path"] != manifest["source_paths"]:
        raise QualificationError("receipt source_path list does not equal manifest source_paths")
    if lists["qualifier_path"] != [manifest_path]:
        raise QualificationError("receipt qualifier_path does not equal manifest path")
    if len(lists["source_object"]) != len(manifest["source_paths"]):
        raise QualificationError("receipt source_object count does not equal source path count")
    for expected_path, encoded in zip(manifest["source_paths"], lists["source_object"], strict=True):
        prefix = f"{expected_path}:"
        if not encoded.startswith(prefix) or not SHA40_RE.fullmatch(encoded[len(prefix):]):
            raise QualificationError(f"invalid source_object binding for {expected_path}")


def github_output(path: str, values: dict[str, str]) -> None:
    destination = Path(path)
    with destination.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise QualificationError(f"GitHub output {key} contains a newline")
            handle.write(f"{key}={value}\n")


def resolve_command(args: argparse.Namespace) -> int:
    candidate_root = resolve_root(args.candidate_root, "candidate root")
    manifest_path = canonical_repo_path(args.manifest, "manifest path")
    manifest = load_manifest(candidate_root, manifest_path)
    verify_candidate_subject(candidate_root, manifest, manifest_path, args.expected_head, args.expected_base)
    outputs = {
        "program": manifest["program"],
        "expected_rust": manifest["expected_rust"],
        "source_parent": manifest["source_parent"],
        "manifest_sha256": canonical_manifest_digest(manifest),
        "manifest_path": manifest_path,
    }
    if args.github_output:
        github_output(args.github_output, outputs)
    print(json.dumps(outputs, sort_keys=True))
    return 0


def run_command(args: argparse.Namespace) -> int:
    trusted_root = resolve_root(args.trusted_root, "trusted root")
    candidate_root = resolve_root(args.candidate_root, "candidate root")
    assert_separate_roots(trusted_root, candidate_root)
    trusted_before = verify_trusted_checkout(trusted_root, args.trusted_policy_sha)

    manifest_path = canonical_repo_path(args.manifest, "manifest path")
    manifest = load_manifest(candidate_root, manifest_path)
    verify_candidate_subject(candidate_root, manifest, manifest_path, args.expected_head, args.expected_base)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir == trusted_root or trusted_root in output_dir.parents:
        raise QualificationError("evidence output must not be inside trusted checkout")
    if output_dir == candidate_root or candidate_root in output_dir.parents:
        raise QualificationError("evidence output must not be inside candidate checkout")

    harness = trusted_root / HARNESS_PATH
    command = [
        "bash",
        str(harness),
        "--program",
        manifest["program"],
        "--source-parent",
        manifest["source_parent"],
        "--expected-head",
        args.expected_head,
        "--expected-rust",
        manifest["expected_rust"],
    ]
    for package in manifest["packages"]:
        command.extend(["--package", package])
    for source_path in manifest["source_paths"]:
        command.extend(["--source-path", source_path])
    command.extend(["--qualifier-path", manifest_path, "--output-dir", str(output_dir)])

    completed = subprocess.run(command, cwd=candidate_root, check=False)
    if completed.returncode != 0:
        return completed.returncode

    trusted_after = verify_trusted_checkout(trusted_root, args.trusted_policy_sha)
    if trusted_after != trusted_before:
        raise QualificationError("trusted workflow/adapter/harness bytes changed during candidate execution")

    receipt = output_dir / "receipt.txt"
    validate_receipt(
        receipt,
        manifest,
        manifest_path,
        args.expected_head,
        args.expected_base,
        trusted_before["trusted_harness_sha256"],
    )

    binding = {
        "schema": BINDING_SCHEMA,
        "authority": "adapter-binding-only",
        "scientific_claim": "NONE",
        "program": manifest["program"],
        "manifest_path": manifest_path,
        "manifest_sha256": canonical_manifest_digest(manifest),
        "source_parent": args.expected_base,
        "qualifier_head": args.expected_head,
        **trusted_before,
        "harness_receipt_sha256": sha256_file(receipt),
    }
    (output_dir / "manifest-binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(binding, sort_keys=True))
    return 0


def expect_failure(fn: Any, message: str) -> None:
    try:
        fn()
    except QualificationError:
        return
    raise AssertionError(message)


def self_test() -> None:
    manifest_payload = {
        "schema": SCHEMA,
        "program": "SCI-TEST-AQ",
        "source_parent": "1" * 40,
        "expected_rust": "1.96.0",
        "packages": ["symthaea-science-research"],
        "source_paths": ["crates/core/symthaea-science-research"],
    }
    manifest_path = ".github/research-qualifiers/SCI-TEST-AQ.json"
    manifest = validate_manifest(manifest_payload, manifest_path)
    digest = canonical_manifest_digest(manifest)
    assert SHA256_RE.fullmatch(digest)

    unknown = dict(manifest_payload)
    unknown["authority"] = "qualified"
    expect_failure(lambda: validate_manifest(unknown, manifest_path), "unknown manifest field accepted")
    expect_failure(
        lambda: parse_json('{"schema":"first","schema":"second"}'),
        "duplicate JSON key accepted",
    )
    bad_path = dict(manifest_payload)
    bad_path["source_paths"] = ["crates//core"]
    expect_failure(lambda: validate_manifest(bad_path, manifest_path), "non-canonical path accepted")

    harness_sha = "2" * 64
    base_receipt = [
        f"schema={HARNESS_RECEIPT_SCHEMA}",
        "program=SCI-TEST-AQ",
        "authority=bootstrap-source-qualification-only",
        "scientific_claim=NONE",
        f"source_subject_sha={'1' * 40}",
        f"qualifier_sha={'3' * 40}",
        "qualifier_parent_count=1",
        f"harness_sha256={harness_sha}",
        "rustc_version=rustc 1.96.0 (test)",
        "cargo_version=cargo 1.96.0 (test)",
        "rustfmt_version=rustfmt 1.8.0-stable (test)",
        "clippy_version=clippy 0.1.96 (test)",
        "rust_host_triple=x86_64-unknown-linux-gnu",
        f"generated_lock_sha256={'4' * 64}",
        f"lock_patch_sha256={'5' * 64}",
        "package=symthaea-science-research",
        "source_path=crates/core/symthaea-science-research",
        f"source_object=crates/core/symthaea-science-research:{'6' * 40}",
        f"qualifier_path={manifest_path}",
        "scope_gate=PASS",
        "source_immutable_gate=PASS",
        "lock_additive_gate=PASS",
        "cargo_check_gate=PASS",
        "rustfmt_gate=PASS",
        "cargo_test_gate=PASS",
        "clippy_gate=PASS",
        "postflight_worktree_immutable_gate=PASS",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        receipt_path = Path(tmp) / "receipt.txt"
        receipt_path.write_text("\n".join(base_receipt) + "\n", encoding="utf-8")
        validate_receipt(receipt_path, manifest, manifest_path, "3" * 40, "1" * 40, harness_sha)

        duplicate = base_receipt + ["clippy_version=second"]
        receipt_path.write_text("\n".join(duplicate) + "\n", encoding="utf-8")
        expect_failure(
            lambda: validate_receipt(receipt_path, manifest, manifest_path, "3" * 40, "1" * 40, harness_sha),
            "duplicate singleton receipt key accepted",
        )

        bad_gate = ["clippy_gate=FAIL" if line == "clippy_gate=PASS" else line for line in base_receipt]
        receipt_path.write_text("\n".join(bad_gate) + "\n", encoding="utf-8")
        expect_failure(
            lambda: validate_receipt(receipt_path, manifest, manifest_path, "3" * 40, "1" * 40, harness_sha),
            "failed gate accepted",
        )

        unknown_receipt = base_receipt + ["authority_upgrade=qualified"]
        receipt_path.write_text("\n".join(unknown_receipt) + "\n", encoding="utf-8")
        expect_failure(
            lambda: validate_receipt(receipt_path, manifest, manifest_path, "3" * 40, "1" * 40, harness_sha),
            "unknown receipt field accepted",
        )

        receipt_path.write_text("\n".join(base_receipt) + "\n", encoding="utf-8")
        expect_failure(
            lambda: validate_receipt(receipt_path, manifest, manifest_path, "3" * 40, "1" * 40, "7" * 64),
            "wrong trusted harness hash accepted",
        )

    print("research_qualification_manifest_adapter_self_test=PASS")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--self-test", action="store_true")
    sub = root.add_subparsers(dest="command")

    resolve = sub.add_parser("resolve")
    resolve.add_argument("--candidate-root", required=True)
    resolve.add_argument("--manifest", required=True)
    resolve.add_argument("--expected-head", required=True)
    resolve.add_argument("--expected-base", required=True)
    resolve.add_argument("--github-output")

    run = sub.add_parser("run")
    run.add_argument("--trusted-root", required=True)
    run.add_argument("--trusted-policy-sha", required=True)
    run.add_argument("--candidate-root", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--expected-head", required=True)
    run.add_argument("--expected-base", required=True)
    run.add_argument("--output-dir", required=True)
    verify = sub.add_parser("verify-pr")
    verify.add_argument("--repo", required=True)
    verify.add_argument("--pr-number", type=int, required=True)
    verify.add_argument("--expected-head", required=True)
    verify.add_argument("--expected-base", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    try:
        if args.command == "resolve":
            return resolve_command(args)
        if args.command == "run":
            forbidden = sorted(
                key for key in os.environ
                if key in {"GITHUB_TOKEN", "GH_TOKEN"} or key.startswith("ACTIONS_ID_TOKEN_")
            )
            if forbidden:
                raise QualificationError(
                    "credential-bearing environment is forbidden during candidate execution: "
                    + ", ".join(forbidden)
                )
            return run_command(args)
        if args.command == "verify-pr":
            verify_live_pr(
                args.repo,
                args.pr_number,
                os.environ.get("GITHUB_TOKEN", ""),
                args.expected_head,
                args.expected_base,
            )
            print("live_pr_subject=PASS")
            return 0
        raise QualificationError("a command is required: resolve, verify-pr, or run")
    except QualificationError as error:
        print(f"research qualification adapter error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
