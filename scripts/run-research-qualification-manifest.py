#!/usr/bin/env python3
"""Validate a research qualifier manifest and invoke the portable harness.

The GitHub workflow performs a minimal diff-scope preflight before this helper
is executed. This helper then validates exact manifest semantics and either
emits scheduler metadata (`resolve`) or executes `qualify-research-crate.sh`
(`run`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

SCHEMA = "symthaea.research-qualifier-manifest.v1"
BINDING_SCHEMA = "symthaea.research-qualifier-manifest-binding.v1"
PROGRAM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
PATH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RUST_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
MANIFEST_ROOT = pathlib.PurePosixPath(".github/research-qualifiers")
HARNESS = pathlib.Path(__file__).with_name("qualify-research-crate.sh")


class ManifestError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_manifest_json(raw: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ManifestError(f"invalid manifest JSON: {error}") from error


def _canonical_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or not PATH_RE.fullmatch(value):
        raise ManifestError(f"{label} contains unsupported path characters: {value!r}")
    path = pathlib.PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or value.startswith("./")
        or path.as_posix() != value
    ):
        raise ManifestError(f"{label} must be canonical repository-relative path: {value!r}")
    return value


def _string_list(value: Any, label: str, pattern: re.Pattern[str] | None = None) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{label} must be a non-empty array")
    if len(value) > 32:
        raise ManifestError(f"{label} exceeds the v1 maximum of 32 entries")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ManifestError(f"every {label} entry must be a non-empty string")
        if pattern is not None and not pattern.fullmatch(item):
            raise ManifestError(f"invalid {label} entry: {item!r}")
        result.append(item)
    if len(set(result)) != len(result):
        raise ManifestError(f"{label} must not contain duplicates")
    return result


def validate_manifest(payload: Any, manifest_path: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ManifestError("manifest root must be an object")
    allowed = {
        "schema",
        "program",
        "source_parent",
        "expected_rust",
        "packages",
        "source_paths",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ManifestError(f"unknown manifest fields: {', '.join(unknown)}")
    missing = sorted(allowed - set(payload))
    if missing:
        raise ManifestError(f"missing manifest fields: {', '.join(missing)}")
    if payload.get("schema") != SCHEMA:
        raise ManifestError(f"schema must be {SCHEMA!r}")

    program = payload.get("program")
    if not isinstance(program, str) or not PROGRAM_RE.fullmatch(program):
        raise ManifestError("program contains unsupported characters or is too long")
    source_parent = payload.get("source_parent")
    if not isinstance(source_parent, str) or not SHA_RE.fullmatch(source_parent):
        raise ManifestError("source_parent must be exactly 40 lowercase hex characters")
    expected_rust = payload.get("expected_rust")
    if not isinstance(expected_rust, str) or not RUST_RE.fullmatch(expected_rust):
        raise ManifestError("expected_rust must have x.y.z form")

    packages = _string_list(payload.get("packages"), "packages", PACKAGE_RE)
    source_paths = [
        _canonical_path(item, "source_paths")
        for item in _string_list(payload.get("source_paths"), "source_paths")
    ]

    manifest_path = _canonical_path(manifest_path, "manifest path")
    path = pathlib.PurePosixPath(manifest_path)
    if path.parent != MANIFEST_ROOT:
        raise ManifestError(
            f"manifest must live directly under {MANIFEST_ROOT.as_posix()}/"
        )
    if path.name != f"{program}.json":
        raise ManifestError(
            f"manifest filename must equal <program>.json; expected {program}.json"
        )

    return {
        "schema": SCHEMA,
        "program": program,
        "source_parent": source_parent,
        "expected_rust": expected_rust,
        "packages": packages,
        "source_paths": source_paths,
    }


def load_manifest(path: pathlib.Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ManifestError(f"manifest not found: {path}") from error
    return validate_manifest(_parse_manifest_json(raw), path.as_posix())


def manifest_digest(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise ManifestError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def verify_subject(
    manifest: dict[str, Any], manifest_path: str, expected_head: str, expected_base: str
) -> None:
    if not SHA_RE.fullmatch(expected_head):
        raise ManifestError("expected head must be exactly 40 lowercase hex characters")
    if not SHA_RE.fullmatch(expected_base):
        raise ManifestError("expected base must be exactly 40 lowercase hex characters")
    actual_head = _git("rev-parse", "HEAD")
    actual_parent = _git("rev-parse", "HEAD^")
    if actual_head != expected_head:
        raise ManifestError(
            f"HEAD mismatch: expected {expected_head}, observed {actual_head}"
        )
    if actual_parent != expected_base:
        raise ManifestError(
            f"immediate parent mismatch: expected base {expected_base}, observed {actual_parent}"
        )
    if manifest["source_parent"] != expected_base:
        raise ManifestError(
            "manifest source_parent must equal the exact PR base / immediate parent"
        )
    changed = sorted(
        item
        for item in _git("diff", "--name-only", expected_base, expected_head).splitlines()
        if item
    )
    if changed != [manifest_path]:
        raise ManifestError(
            f"qualifier diff must contain exactly {manifest_path!r}; observed {changed!r}"
        )


def _write_github_output(path: str, values: dict[str, str]) -> None:
    destination = pathlib.Path(path)
    with destination.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ManifestError(f"output {key} contains a newline")
            handle.write(f"{key}={value}\n")


def resolve(args: argparse.Namespace) -> int:
    manifest = load_manifest(pathlib.Path(args.manifest))
    verify_subject(manifest, args.manifest, args.expected_head, args.expected_base)
    digest = manifest_digest(manifest)
    outputs = {
        "program": manifest["program"],
        "expected_rust": manifest["expected_rust"],
        "source_parent": manifest["source_parent"],
        "manifest_path": args.manifest,
        "manifest_sha256": digest,
    }
    if args.github_output:
        _write_github_output(args.github_output, outputs)
    print(json.dumps(outputs, sort_keys=True))
    return 0


def run_qualification(args: argparse.Namespace) -> int:
    manifest = load_manifest(pathlib.Path(args.manifest))
    verify_subject(manifest, args.manifest, args.expected_head, args.expected_base)
    digest = manifest_digest(manifest)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "bash",
        str(HARNESS),
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
    command.extend(["--qualifier-path", args.manifest])
    command.extend(["--output-dir", str(output_dir)])

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode

    harness_receipt = output_dir / "receipt.txt"
    if not harness_receipt.is_file():
        raise ManifestError("qualification harness returned success without receipt.txt")
    receipt_sha = hashlib.sha256(harness_receipt.read_bytes()).hexdigest()
    binding = {
        "schema": BINDING_SCHEMA,
        "authority": "adapter-binding-only",
        "scientific_claim": "NONE",
        "program": manifest["program"],
        "manifest_path": args.manifest,
        "manifest_sha256": digest,
        "source_parent": manifest["source_parent"],
        "qualifier_head": args.expected_head,
        "harness_receipt_sha256": receipt_sha,
    }
    (output_dir / "manifest-binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(binding, sort_keys=True))
    return 0


def self_test() -> None:
    payload = {
        "schema": SCHEMA,
        "program": "SCI-TEST-AQ",
        "source_parent": "1" * 40,
        "expected_rust": "1.96.0",
        "packages": ["symthaea-science-research"],
        "source_paths": ["crates/core/symthaea-science-research"],
    }
    manifest = validate_manifest(payload, ".github/research-qualifiers/SCI-TEST-AQ.json")
    digest = manifest_digest(manifest)
    assert len(digest) == 64
    assert digest == manifest_digest(dict(reversed(list(manifest.items()))))

    changed = dict(payload)
    changed["expected_rust"] = "1.97.0"
    assert manifest_digest(
        validate_manifest(changed, ".github/research-qualifiers/SCI-TEST-AQ.json")
    ) != digest

    duplicate = dict(payload)
    duplicate["packages"] = ["x", "x"]
    try:
        validate_manifest(duplicate, ".github/research-qualifiers/SCI-TEST-AQ.json")
    except ManifestError:
        pass
    else:
        raise AssertionError("duplicate packages were accepted")

    unknown = dict(payload)
    unknown["authority"] = "qualified"
    try:
        validate_manifest(unknown, ".github/research-qualifiers/SCI-TEST-AQ.json")
    except ManifestError:
        pass
    else:
        raise AssertionError("unknown authority field was accepted")

    try:
        validate_manifest(payload, ".github/research-qualifiers/other.json")
    except ManifestError:
        pass
    else:
        raise AssertionError("manifest filename/program mismatch was accepted")

    try:
        _parse_manifest_json('{"schema":"first","schema":"second"}')
    except ManifestError:
        pass
    else:
        raise AssertionError("duplicate JSON keys were accepted")

    ambiguous = dict(payload)
    ambiguous["source_paths"] = ["crates//core/symthaea-science-research"]
    try:
        validate_manifest(ambiguous, ".github/research-qualifiers/SCI-TEST-AQ.json")
    except ManifestError:
        pass
    else:
        raise AssertionError("non-canonical source path was accepted")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--self-test", action="store_true")
    subparsers = root.add_subparsers(dest="command")

    for name in ("resolve", "run"):
        child = subparsers.add_parser(name)
        child.add_argument("--manifest", required=True)
        child.add_argument("--expected-head", required=True)
        child.add_argument("--expected-base", required=True)
        if name == "resolve":
            child.add_argument("--github-output")
        else:
            child.add_argument("--output-dir", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        print("research qualification manifest self-test: PASS")
        return 0
    try:
        if args.command == "resolve":
            return resolve(args)
        if args.command == "run":
            return run_qualification(args)
        raise ManifestError("a command is required: resolve or run")
    except ManifestError as error:
        print(f"research qualification manifest error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
