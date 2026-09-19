#!/usr/bin/env python3
"""Trusted canonical program-profile validator for research qualification.

This layer does not execute candidate code. It binds the candidate's data-only
qualifier manifest to a canonical program scope stored in the exact trusted
policy checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any

PROFILE_SCHEMA = "symthaea.research-program-profile.v1"
PROFILE_ROOT = PurePosixPath(".github/research-program-profiles")
ADAPTER_PATH = "scripts/run-research-qualification-manifest.py"
VALIDATOR_PATH = "scripts/validate-research-program-profile.py"


class ProfileError(RuntimeError):
    pass


def bootstrap_git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ProfileError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def verify_bootstrap_trusted_file(
    root: Path, relative: str, executing_file: str | None = None
) -> tuple[str, str]:
    root_resolved = root.resolve(strict=True)
    unresolved = root_resolved / relative
    if unresolved.is_symlink():
        raise ProfileError(f"trusted policy path must not be a symlink: {relative}")
    resolved = unresolved.resolve(strict=True)
    if root_resolved not in resolved.parents or not resolved.is_file():
        raise ProfileError(f"trusted policy path is invalid: {relative}")
    if executing_file is not None and Path(executing_file).resolve() != resolved:
        raise ProfileError(f"{relative} is not executing from the trusted checkout")

    line = bootstrap_git(root_resolved, "ls-tree", "HEAD", "--", relative)
    if "\t" not in line:
        raise ProfileError(f"trusted policy path is not tracked: {relative}")
    metadata, observed_path = line.split("\t", 1)
    fields = metadata.split()
    if len(fields) != 3 or observed_path != relative:
        raise ProfileError(f"ambiguous trusted Git object for {relative}")
    mode, object_type, object_sha = fields
    if (mode, object_type) != ("100644", "blob"):
        raise ProfileError(f"trusted policy path must be 100644/blob: {relative}")
    worktree_blob = bootstrap_git(root_resolved, "hash-object", "--", relative)
    if worktree_blob != object_sha:
        raise ProfileError(f"trusted policy worktree bytes differ from HEAD: {relative}")
    return object_sha, sha256_bytes(resolved.read_bytes())


def load_adapter(trusted_root: Path) -> Any:
    path = trusted_root / ADAPTER_PATH
    verify_bootstrap_trusted_file(trusted_root, ADAPTER_PATH)
    spec = importlib.util.spec_from_file_location("symthaea_research_qualification_adapter", path)
    if spec is None or spec.loader is None:
        raise ProfileError("could not construct trusted adapter import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(adapter: Any, root: Path, *args: str) -> str:
    try:
        return adapter.git(root, *args)
    except Exception as error:
        raise ProfileError(str(error)) from error


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_profile_digest(profile: dict[str, Any]) -> str:
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return sha256_bytes(payload)


def profile_path_for(program: str) -> str:
    return (PROFILE_ROOT / f"{program}.json").as_posix()


def validate_profile(payload: Any, expected_program: str, adapter: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProfileError("program profile root must be an object")
    allowed = {"schema", "program", "expected_rust", "packages", "source_paths"}
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown:
        raise ProfileError(f"unknown program profile fields: {', '.join(unknown)}")
    if missing:
        raise ProfileError(f"missing program profile fields: {', '.join(missing)}")
    if payload["schema"] != PROFILE_SCHEMA:
        raise ProfileError(f"profile schema must equal {PROFILE_SCHEMA!r}")
    program = payload["program"]
    if not isinstance(program, str) or not adapter.PROGRAM_RE.fullmatch(program):
        raise ProfileError("invalid program profile identity")
    if program != expected_program:
        raise ProfileError("program profile identity does not match qualifier manifest program")
    expected_rust = payload["expected_rust"]
    if not isinstance(expected_rust, str) or not adapter.RUST_RE.fullmatch(expected_rust):
        raise ProfileError("program profile expected_rust must have x.y.z form")
    packages = adapter.string_list(payload["packages"], "program profile packages", adapter.PACKAGE_RE)
    source_paths = [
        adapter.canonical_repo_path(item, "program profile source_paths")
        for item in adapter.string_list(payload["source_paths"], "program profile source_paths")
    ]
    return {
        "schema": PROFILE_SCHEMA,
        "program": program,
        "expected_rust": expected_rust,
        "packages": packages,
        "source_paths": source_paths,
    }


def require_scope_match(profile: dict[str, Any], manifest: dict[str, Any]) -> None:
    for key in ("program", "expected_rust", "packages", "source_paths"):
        if profile[key] != manifest[key]:
            raise ProfileError(f"manifest {key} does not equal trusted program profile")


def verify_trusted_file(root: Path, relative: str, adapter: Any, executing_file: str | None = None) -> tuple[str, str]:
    # Bootstrap verification is intentionally independent of the imported B2 adapter.
    # The adapter itself is one of the trusted files whose bytes must be proven first.
    del adapter
    return verify_bootstrap_trusted_file(root, relative, executing_file)


def load_profile(trusted_root: Path, program: str, adapter: Any) -> tuple[dict[str, Any], dict[str, str]]:
    relative = profile_path_for(program)
    git_blob, file_sha256 = verify_trusted_file(trusted_root, relative, adapter)
    path = trusted_root / relative
    try:
        payload = adapter.parse_json(path.read_text(encoding="utf-8"))
    except Exception as error:
        if isinstance(error, ProfileError):
            raise
        raise ProfileError(f"invalid trusted program profile: {error}") from error
    profile = validate_profile(payload, program, adapter)
    return profile, {
        "program_profile_path": relative,
        "program_profile_git_blob": git_blob,
        "program_profile_file_sha256": file_sha256,
        "program_profile_canonical_sha256": canonical_profile_digest(profile),
    }


def resolve_contract(
    trusted_root: Path,
    trusted_policy_sha: str,
    candidate_root: Path,
    manifest_path: str,
    expected_head: str,
    expected_base: str,
    *,
    adapter: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    trusted_root = trusted_root.resolve(strict=True)
    candidate_root = candidate_root.resolve(strict=True)
    if bootstrap_git(trusted_root, "rev-parse", "HEAD") != trusted_policy_sha:
        raise ProfileError("trusted checkout does not equal expected policy SHA")
    if adapter is None:
        adapter = load_adapter(trusted_root)
    adapter_file = getattr(adapter, "__file__", None)
    if not isinstance(adapter_file, str):
        raise ProfileError("trusted adapter module has no concrete file identity")
    verify_bootstrap_trusted_file(trusted_root, ADAPTER_PATH, adapter_file)
    verify_bootstrap_trusted_file(trusted_root, VALIDATOR_PATH, __file__)
    adapter.assert_separate_roots(trusted_root, candidate_root)

    manifest_path = adapter.canonical_repo_path(manifest_path, "manifest path")
    manifest = adapter.load_manifest(candidate_root, manifest_path)
    adapter.verify_candidate_subject(candidate_root, manifest, manifest_path, expected_head, expected_base)
    profile, profile_identity = load_profile(trusted_root, manifest["program"], adapter)
    require_scope_match(profile, manifest)
    return manifest, profile, profile_identity


def github_output(path: str, values: dict[str, str]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ProfileError(f"GitHub output {key} contains newline")
            handle.write(f"{key}={value}\n")


def run(args: argparse.Namespace) -> int:
    manifest, profile, identity = resolve_contract(
        Path(args.trusted_root),
        args.trusted_policy_sha,
        Path(args.candidate_root),
        args.manifest,
        args.expected_head,
        args.expected_base,
    )
    outputs = {
        "program": manifest["program"],
        "expected_rust": profile["expected_rust"],
        **identity,
    }
    if args.github_output:
        github_output(args.github_output, outputs)
    print(json.dumps(outputs, sort_keys=True))
    return 0


def self_test() -> None:
    import re

    class Adapter:
        PROGRAM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
        PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
        RUST_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")

        @staticmethod
        def string_list(value: Any, label: str, pattern: Any = None) -> list[str]:
            if not isinstance(value, list) or not value or len(value) > 32:
                raise ProfileError(f"{label} must contain 1..32 entries")
            result: list[str] = []
            for item in value:
                if not isinstance(item, str) or not item or (pattern and not pattern.fullmatch(item)):
                    raise ProfileError(f"invalid {label} entry")
                result.append(item)
            if len(set(result)) != len(result):
                raise ProfileError(f"duplicate {label} entry")
            return result

        @staticmethod
        def canonical_repo_path(value: str, label: str) -> str:
            path = PurePosixPath(value)
            if path.is_absolute() or value.startswith("./") or value.endswith("/") or "//" in value or ".." in path.parts or "." in path.parts or path.as_posix() != value:
                raise ProfileError(f"invalid {label}")
            return value

    adapter = Adapter()
    profile_payload = {
        "schema": PROFILE_SCHEMA,
        "program": "SCI-TEST-AQ",
        "expected_rust": "1.96.0",
        "packages": ["symthaea-science-research"],
        "source_paths": ["crates/core/symthaea-science-research"],
    }
    profile = validate_profile(profile_payload, "SCI-TEST-AQ", adapter)
    manifest = {**profile, "schema": "symthaea.research-qualifier-manifest.v1", "source_parent": "1" * 40}
    require_scope_match(profile, manifest)
    assert len(canonical_profile_digest(profile)) == 64
    assert profile_path_for("SCI-TEST-AQ") == ".github/research-program-profiles/SCI-TEST-AQ.json"

    bad = dict(profile_payload)
    bad["packages"] = ["other"]
    try:
        require_scope_match(validate_profile(bad, "SCI-TEST-AQ", adapter), manifest)
    except ProfileError:
        pass
    else:
        raise AssertionError("scope mismatch was accepted")

    unknown = dict(profile_payload)
    unknown["authority"] = "qualified"
    try:
        validate_profile(unknown, "SCI-TEST-AQ", adapter)
    except ProfileError:
        pass
    else:
        raise AssertionError("unknown profile field was accepted")

    print("research_program_profile_validator_self_test=PASS")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--trusted-root")
    p.add_argument("--trusted-policy-sha")
    p.add_argument("--candidate-root")
    p.add_argument("--manifest")
    p.add_argument("--expected-head")
    p.add_argument("--expected-base")
    p.add_argument("--github-output")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    required = ("trusted_root", "trusted_policy_sha", "candidate_root", "manifest", "expected_head", "expected_base")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        print(f"program profile error: missing required arguments: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        return run(args)
    except Exception as error:
        print(f"research program profile error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
