#!/usr/bin/env python3
"""Bind declared research packages to exact frozen source roots without executing Cargo.

This theorem is intentionally narrower than build closure. It proves that every
package named by a trusted research program profile has exactly one explicitly
declared source root whose frozen Cargo.toml declares that exact package name.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tomllib
from typing import Any

CLOSURE_SCHEMA = "symthaea.research-package-root-closure.v1"
AUTHORITY = "package-root-binding-only"
SCIENTIFIC_CLAIM = "NONE"
PROFILE_VALIDATOR_PATH = "scripts/validate-research-program-profile.py"
VALIDATOR_PATH = "scripts/validate-research-package-root-closure.py"
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")


class ClosureError(RuntimeError):
    pass


def git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ClosureError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def git_bytes(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ClosureError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return sha256_bytes(raw)


def verify_trusted_file(root: Path, relative: str, executing_file: str | None = None) -> tuple[str, str]:
    root = root.resolve(strict=True)
    unresolved = root / relative
    if unresolved.is_symlink():
        raise ClosureError(f"trusted policy path must not be a symlink: {relative}")
    resolved = unresolved.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ClosureError(f"trusted policy path is invalid: {relative}")
    if executing_file is not None and Path(executing_file).resolve() != resolved:
        raise ClosureError(f"{relative} is not executing from the trusted checkout")
    line = git(root, "ls-tree", "HEAD", "--", relative)
    if "\t" not in line:
        raise ClosureError(f"trusted policy path is not tracked: {relative}")
    metadata, observed = line.split("\t", 1)
    fields = metadata.split()
    if len(fields) != 3 or observed != relative or fields[0] != "100644" or fields[1] != "blob":
        raise ClosureError(f"trusted policy path must be exact 100644/blob: {relative}")
    blob = fields[2]
    if git(root, "hash-object", "--", relative) != blob:
        raise ClosureError(f"trusted policy worktree bytes differ from HEAD: {relative}")
    return blob, sha256_bytes(resolved.read_bytes())


def load_verified_module(root: Path, relative: str, module_name: str) -> Any:
    verify_trusted_file(root, relative)
    path = root / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ClosureError(f"could not construct import for {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_entry(root: Path, ref: str, path: str) -> tuple[str, str, str] | None:
    line = git(root, "ls-tree", ref, "--", path)
    if not line:
        return None
    lines = line.splitlines()
    if len(lines) != 1 or "\t" not in lines[0]:
        raise ClosureError(f"ambiguous Git entry for {ref}:{path}")
    metadata, observed = lines[0].split("\t", 1)
    fields = metadata.split()
    if len(fields) != 3 or observed != path:
        raise ClosureError(f"malformed Git entry for {ref}:{path}")
    return fields[0], fields[1], fields[2]


def require_same_object(root: Path, base: str, head: str, path: str) -> tuple[str, str, str]:
    base_entry = git_entry(root, base, path)
    head_entry = git_entry(root, head, path)
    if base_entry is None:
        raise ClosureError(f"profile source path does not exist in source parent: {path}")
    if head_entry != base_entry:
        raise ClosureError(f"profile source path changed across qualifier commit: {path}")
    return base_entry


def package_name_from_cargo_toml(raw: bytes, path: str) -> str | None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ClosureError(f"Cargo.toml is not UTF-8 at {path}") from error
    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ClosureError(f"invalid Cargo.toml at {path}: {error}") from error
    package = payload.get("package")
    if package is None:
        return None
    if not isinstance(package, dict):
        raise ClosureError(f"[package] must be a TOML table at {path}")
    name = package.get("name")
    if name is None:
        return None
    if not isinstance(name, str) or not name:
        raise ClosureError(f"[package].name must be a non-empty string at {path}")
    return name


def derive_closure(
    trusted_root: Path,
    trusted_policy_sha: str,
    candidate_root: Path,
    manifest_path: str,
    expected_head: str,
    expected_base: str,
) -> tuple[dict[str, Any], str]:
    trusted_root = trusted_root.resolve(strict=True)
    candidate_root = candidate_root.resolve(strict=True)
    if not SHA40_RE.fullmatch(trusted_policy_sha):
        raise ClosureError("trusted policy SHA must be 40 lowercase hex characters")
    if not SHA40_RE.fullmatch(expected_head) or not SHA40_RE.fullmatch(expected_base):
        raise ClosureError("candidate head/base must be 40 lowercase hex characters")
    if git(trusted_root, "rev-parse", "HEAD") != trusted_policy_sha:
        raise ClosureError("trusted checkout does not equal expected policy SHA")
    verify_trusted_file(trusted_root, VALIDATOR_PATH, __file__)

    profile_validator = load_verified_module(
        trusted_root,
        PROFILE_VALIDATOR_PATH,
        "symthaea_research_program_profile_validator_for_package_roots",
    )
    manifest, profile, profile_identity = profile_validator.resolve_contract(
        trusted_root,
        trusted_policy_sha,
        candidate_root,
        manifest_path,
        expected_head,
        expected_base,
    )

    if git(candidate_root, "rev-parse", "HEAD") != expected_head:
        raise ClosureError("candidate checkout HEAD changed after profile resolution")

    roots_by_package: dict[str, dict[str, str]] = {}
    for source_path in profile["source_paths"]:
        mode, object_type, source_object = require_same_object(
            candidate_root, expected_base, expected_head, source_path
        )
        if object_type != "tree":
            continue
        if mode != "040000":
            raise ClosureError(f"directory source path has unexpected Git mode: {source_path}")

        cargo_path = (PurePosixPath(source_path) / "Cargo.toml").as_posix()
        cargo_entry = git_entry(candidate_root, expected_base, cargo_path)
        if cargo_entry is None:
            continue
        if cargo_entry[0:2] != ("100644", "blob"):
            raise ClosureError(f"package manifest must be exact 100644/blob: {cargo_path}")
        if git_entry(candidate_root, expected_head, cargo_path) != cargo_entry:
            raise ClosureError(f"package manifest changed across qualifier commit: {cargo_path}")

        raw = git_bytes(candidate_root, "show", f"{expected_base}:{cargo_path}")
        package_name = package_name_from_cargo_toml(raw, cargo_path)
        if package_name is None:
            continue
        if package_name not in profile["packages"]:
            raise ClosureError(
                f"declared source root contains Cargo package absent from profile packages: "
                f"{source_path} -> {package_name}"
            )
        if package_name in roots_by_package:
            raise ClosureError(f"declared package has multiple source roots: {package_name}")

        roots_by_package[package_name] = {
            "package": package_name,
            "source_root": source_path,
            "source_tree_git_object": source_object,
            "cargo_manifest_path": cargo_path,
            "cargo_manifest_git_blob": cargo_entry[2],
            "cargo_manifest_sha256": sha256_bytes(raw),
        }

    missing = [package for package in profile["packages"] if package not in roots_by_package]
    if missing:
        raise ClosureError(
            "declared packages lack an exact declared Cargo source root: " + ", ".join(missing)
        )

    ordered_roots = [roots_by_package[package] for package in profile["packages"]]
    manifest_semantic = {
        "schema": manifest["schema"],
        "program": manifest["program"],
        "source_parent": manifest["source_parent"],
        "expected_rust": manifest["expected_rust"],
        "packages": manifest["packages"],
        "source_paths": manifest["source_paths"],
    }
    closure = {
        "schema": CLOSURE_SCHEMA,
        "authority": AUTHORITY,
        "scientific_claim": SCIENTIFIC_CLAIM,
        "program": profile["program"],
        "manifest_path": manifest_path,
        "manifest_sha256": canonical_digest(manifest_semantic),
        "source_parent": expected_base,
        "qualifier_head": expected_head,
        "program_profile_path": profile_identity["program_profile_path"],
        "program_profile_git_blob": profile_identity["program_profile_git_blob"],
        "program_profile_file_sha256": profile_identity["program_profile_file_sha256"],
        "program_profile_canonical_sha256": profile_identity["program_profile_canonical_sha256"],
        "package_roots": ordered_roots,
    }
    return closure, canonical_digest(closure)


def github_output(path: str, values: dict[str, str]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ClosureError(f"GitHub output {key} contains newline")
            handle.write(f"{key}={value}\n")


def self_test() -> None:
    assert package_name_from_cargo_toml(b'[package]\nname = "alpha"\n', "Cargo.toml") == "alpha"
    assert package_name_from_cargo_toml(b'[workspace]\nmembers = []\n', "Cargo.toml") is None
    try:
        package_name_from_cargo_toml(b'[package]\nname = ["bad"]\n', "Cargo.toml")
    except ClosureError:
        pass
    else:
        raise AssertionError("non-string package name was accepted")

    exact = b'[package]\nname = "alpha"\n'
    assert sha256_bytes(exact) != sha256_bytes(exact.rstrip(b"\n"))
    sample = {
        "schema": CLOSURE_SCHEMA,
        "authority": AUTHORITY,
        "scientific_claim": SCIENTIFIC_CLAIM,
        "program": "SCI-TEST-A",
        "package_roots": [
            {
                "package": "alpha",
                "source_root": "crates/alpha",
                "source_tree_git_object": "1" * 40,
                "cargo_manifest_path": "crates/alpha/Cargo.toml",
                "cargo_manifest_git_blob": "2" * 40,
                "cargo_manifest_sha256": "3" * 64,
            }
        ],
    }
    assert len(canonical_digest(sample)) == 64
    print("research_package_root_closure_self_test=PASS")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--trusted-root")
    p.add_argument("--trusted-policy-sha")
    p.add_argument("--candidate-root")
    p.add_argument("--manifest")
    p.add_argument("--expected-head")
    p.add_argument("--expected-base")
    p.add_argument("--write-closure")
    p.add_argument("--github-output")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (
        "trusted_root",
        "trusted_policy_sha",
        "candidate_root",
        "manifest",
        "expected_head",
        "expected_base",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        print(f"package-root closure error: missing required arguments: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        closure, digest = derive_closure(
            Path(args.trusted_root),
            args.trusted_policy_sha,
            Path(args.candidate_root),
            args.manifest,
            args.expected_head,
            args.expected_base,
        )
        if args.write_closure:
            destination = Path(args.write_closure)
            if destination.exists() or destination.is_symlink():
                raise ClosureError("closure destination must not already exist")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(closure, indent=2, sort_keys=True) + "\n")
        outputs = {
            "program": closure["program"],
            "package_root_count": str(len(closure["package_roots"])),
            "package_root_closure_sha256": digest,
            "program_profile_canonical_sha256": closure["program_profile_canonical_sha256"],
        }
        if args.github_output:
            github_output(args.github_output, outputs)
        print(json.dumps({**outputs, "closure": closure}, sort_keys=True))
        return 0
    except Exception as error:
        print(f"research package-root closure error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
