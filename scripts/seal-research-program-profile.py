#!/usr/bin/env python3
"""Fresh-runner final seal for canonical research program-scope authority.

This layer is deliberately separate from the B2 adapter binding. B2 proves the
exact manifest-declared scope that executed. C2 proves that this executed scope
is exactly the canonical scope for the named program in trusted default-branch
policy. Neither layer grants scientific authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

BINDING_SCHEMA = "symthaea.research-program-profile-binding.v1"
AUTHORITY = "program-scope-binding-only"
SCOPE_AUTHORITY = "trusted-program-profile-v1"
SCIENTIFIC_CLAIM = "NONE"
ADAPTER_PATH = "scripts/run-research-qualification-manifest.py"
B2_WORKFLOW_PATH = ".github/workflows/research-bootstrap-qualification.yml"
B2_HARNESS_PATH = "scripts/qualify-research-crate.sh"
B2_SUBJECT_GUARD_PATH = "scripts/validate-research-qualification-subject.py"
B2_SEALER_PATH = "scripts/seal-research-qualification-binding.py"
PROFILE_VALIDATOR_PATH = "scripts/validate-research-program-profile.py"
PROFILE_SEALER_PATH = "scripts/seal-research-program-profile.py"
BASE_BINDING_NAME = "manifest-binding.json"
PROFILE_BINDING_NAME = "program-profile-binding.json"
B2_BINDING_FIELDS = frozenset(
    {
        "schema",
        "seal_profile",
        "authority",
        "scope_authority",
        "scientific_claim",
        "program",
        "manifest_path",
        "manifest_sha256",
        "source_parent",
        "qualifier_head",
        "trusted_policy_sha",
        "trusted_workflow_sha256",
        "trusted_adapter_sha256",
        "trusted_harness_sha256",
        "trusted_subject_guard_sha256",
        "trusted_sealer_sha256",
        "harness_receipt_sha256",
        "generated_lock_sha256",
        "lock_patch_sha256",
        "admission_mode",
        "live_pr_postflight",
        "pull_request_number",
        "github_run_id",
        "github_run_attempt",
    }
)


class ProfileSealError(RuntimeError):
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
        raise ProfileSealError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path, module_name: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ProfileSealError(f"trusted module missing or unsafe: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ProfileSealError(f"could not construct import for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_trusted_policy_file(
    root: Path, relative: str, adapter: Any | None = None, executing_file: str | None = None
) -> tuple[str, str]:
    # This verification deliberately does not trust the imported B2 adapter.
    del adapter
    root_resolved = root.resolve(strict=True)
    path = root_resolved / relative
    if path.is_symlink():
        raise ProfileSealError(f"trusted policy path must not be a symlink: {relative}")
    resolved = path.resolve(strict=True)
    if root_resolved not in resolved.parents or not resolved.is_file():
        raise ProfileSealError(f"trusted policy path is invalid: {relative}")
    if executing_file is not None and Path(executing_file).resolve() != resolved:
        raise ProfileSealError(f"{relative} is not executing from the trusted checkout")
    line = bootstrap_git(root_resolved, "ls-tree", "HEAD", "--", relative)
    if "\t" not in line:
        raise ProfileSealError(f"trusted policy path is not tracked: {relative}")
    metadata, observed = line.split("\t", 1)
    fields = metadata.split()
    if len(fields) != 3 or observed != relative or fields[0] != "100644" or fields[1] != "blob":
        raise ProfileSealError(f"trusted policy path must be exact 100644/blob: {relative}")
    blob = fields[2]
    if bootstrap_git(root_resolved, "hash-object", "--", relative) != blob:
        raise ProfileSealError(f"trusted policy worktree bytes differ from HEAD: {relative}")
    return blob, sha256(resolved)


def read_json_object(path: Path, adapter: Any) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProfileSealError(f"required binding is not a regular non-symlink file: {path.name}")
    try:
        payload = adapter.parse_json(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ProfileSealError(f"invalid JSON binding {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ProfileSealError(f"binding root must be an object: {path.name}")
    return payload


def require_base_binding(
    binding: dict[str, Any],
    *,
    adapter: Any,
    manifest: dict[str, Any],
    manifest_path: str,
    expected_head: str,
    expected_base: str,
    trusted_policy_sha: str,
    trusted_policy_sha256: dict[str, str],
    github_run_id: int,
    github_run_attempt: int,
    pr_number: int | None,
) -> None:
    if set(binding) != B2_BINDING_FIELDS:
        unknown = sorted(set(binding) - B2_BINDING_FIELDS)
        missing = sorted(B2_BINDING_FIELDS - set(binding))
        raise ProfileSealError(
            f"base adapter binding field set mismatch; unknown={unknown}, missing={missing}"
        )
    required = {
        "schema": "symthaea.research-qualifier-binding.v2",
        "seal_profile": "authenticated-postflight-v1",
        "authority": "adapter-binding-only",
        "scope_authority": "manifest-declared-only",
        "scientific_claim": "NONE",
        "program": manifest["program"],
        "manifest_path": manifest_path,
        "manifest_sha256": adapter.canonical_manifest_digest(manifest),
        "source_parent": expected_base,
        "qualifier_head": expected_head,
        "trusted_policy_sha": trusted_policy_sha,
        "trusted_workflow_sha256": trusted_policy_sha256["trusted_workflow_sha256"],
        "trusted_adapter_sha256": trusted_policy_sha256["trusted_adapter_sha256"],
        "trusted_harness_sha256": trusted_policy_sha256["trusted_harness_sha256"],
        "trusted_subject_guard_sha256": trusted_policy_sha256["trusted_subject_guard_sha256"],
        "trusted_sealer_sha256": trusted_policy_sha256["trusted_sealer_sha256"],
        "admission_mode": "pull_request_target" if pr_number is not None else "workflow_dispatch",
        "live_pr_postflight": "PASS" if pr_number is not None else "NOT_APPLICABLE_MANUAL_DISPATCH",
        "pull_request_number": pr_number,
        "github_run_id": github_run_id,
        "github_run_attempt": github_run_attempt,
    }
    for key, expected in required.items():
        if binding.get(key) != expected:
            raise ProfileSealError(f"base adapter binding {key} mismatch")
    for key in (
        "trusted_workflow_sha256",
        "trusted_adapter_sha256",
        "trusted_harness_sha256",
        "trusted_subject_guard_sha256",
        "trusted_sealer_sha256",
        "harness_receipt_sha256",
        "generated_lock_sha256",
        "lock_patch_sha256",
    ):
        value = binding.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ProfileSealError(f"base adapter binding {key} is not a SHA-256 digest")


def run(args: argparse.Namespace) -> int:
    trusted_root = Path(args.trusted_root).resolve(strict=True)
    candidate_root = Path(args.candidate_root).resolve(strict=True)
    final_dir = Path(args.final_dir).resolve(strict=True)

    if bootstrap_git(trusted_root, "rev-parse", "HEAD") != args.trusted_policy_sha:
        raise ProfileSealError("trusted checkout does not equal expected policy SHA")

    b2_objects: dict[str, tuple[str, str]] = {}
    for key, relative in (
        ("trusted_workflow", B2_WORKFLOW_PATH),
        ("trusted_adapter", ADAPTER_PATH),
        ("trusted_harness", B2_HARNESS_PATH),
        ("trusted_subject_guard", B2_SUBJECT_GUARD_PATH),
        ("trusted_sealer", B2_SEALER_PATH),
    ):
        b2_objects[key] = verify_trusted_policy_file(trusted_root, relative)
    adapter_blob, adapter_sha256 = b2_objects["trusted_adapter"]
    trusted_policy_sha256 = {f"{key}_sha256": value[1] for key, value in b2_objects.items()}

    validator_blob, validator_sha256 = verify_trusted_policy_file(
        trusted_root, PROFILE_VALIDATOR_PATH
    )
    sealer_blob, sealer_sha256 = verify_trusted_policy_file(
        trusted_root, PROFILE_SEALER_PATH, executing_file=__file__
    )

    adapter = load_module(trusted_root / ADAPTER_PATH, "symthaea_research_qualification_adapter")
    profile_validator = load_module(
        trusted_root / PROFILE_VALIDATOR_PATH,
        "symthaea_research_program_profile_validator",
    )
    adapter.assert_separate_roots(trusted_root, candidate_root)

    manifest_path = adapter.canonical_repo_path(args.manifest, "manifest path")
    manifest, profile, profile_identity = profile_validator.resolve_contract(
        trusted_root,
        args.trusted_policy_sha,
        candidate_root,
        manifest_path,
        args.expected_head,
        args.expected_base,
        adapter=adapter,
    )

    base_binding_path = final_dir / BASE_BINDING_NAME
    profile_binding_path = final_dir / PROFILE_BINDING_NAME
    if profile_binding_path.exists() or profile_binding_path.is_symlink():
        raise ProfileSealError("program-profile-binding.json must not already exist")
    base_binding = read_json_object(base_binding_path, adapter)
    require_base_binding(
        base_binding,
        adapter=adapter,
        manifest=manifest,
        manifest_path=manifest_path,
        expected_head=args.expected_head,
        expected_base=args.expected_base,
        trusted_policy_sha=args.trusted_policy_sha,
        trusted_policy_sha256=trusted_policy_sha256,
        github_run_id=args.github_run_id,
        github_run_attempt=args.github_run_attempt,
        pr_number=args.pr_number,
    )

    admission_mode = "workflow_dispatch"
    live_pr_postflight = "NOT_APPLICABLE_MANUAL_DISPATCH"
    pr_number: int | None = None
    if args.pr_number is not None:
        if not args.repo:
            raise ProfileSealError("--repo is required with --pr-number")
        adapter.verify_live_pr(
            args.repo,
            args.pr_number,
            os.environ.get("GITHUB_TOKEN", ""),
            args.expected_head,
            args.expected_base,
        )
        admission_mode = "pull_request_target"
        live_pr_postflight = "PASS"
        pr_number = args.pr_number
    elif args.repo:
        raise ProfileSealError("--repo and --pr-number must be supplied together")

    if args.github_run_id <= 0 or args.github_run_attempt <= 0:
        raise ProfileSealError("GitHub run id and attempt must be positive integers")

    # Re-read all trusted identities immediately before the final write.
    if bootstrap_git(trusted_root, "rev-parse", "HEAD") != args.trusted_policy_sha:
        raise ProfileSealError("trusted policy HEAD changed before profile sealing")
    for key, relative in (
        ("trusted_workflow", B2_WORKFLOW_PATH),
        ("trusted_adapter", ADAPTER_PATH),
        ("trusted_harness", B2_HARNESS_PATH),
        ("trusted_subject_guard", B2_SUBJECT_GUARD_PATH),
        ("trusted_sealer", B2_SEALER_PATH),
    ):
        if verify_trusted_policy_file(trusted_root, relative) != b2_objects[key]:
            raise ProfileSealError(f"{key} changed before profile sealing")
    profile_after, identity_after = profile_validator.load_profile(
        trusted_root, manifest["program"], adapter
    )
    profile_validator.require_scope_match(profile_after, manifest)
    if profile_after != profile or identity_after != profile_identity:
        raise ProfileSealError("trusted program profile changed before final sealing")
    if verify_trusted_policy_file(trusted_root, PROFILE_VALIDATOR_PATH, adapter) != (validator_blob, validator_sha256):
        raise ProfileSealError("trusted profile validator changed before final sealing")
    if verify_trusted_policy_file(trusted_root, PROFILE_SEALER_PATH, adapter, __file__) != (sealer_blob, sealer_sha256):
        raise ProfileSealError("trusted profile sealer changed before final sealing")

    binding = {
        "schema": BINDING_SCHEMA,
        "authority": AUTHORITY,
        "scope_authority": SCOPE_AUTHORITY,
        "scientific_claim": SCIENTIFIC_CLAIM,
        "program": manifest["program"],
        "manifest_path": manifest_path,
        "manifest_sha256": adapter.canonical_manifest_digest(manifest),
        "manifest_binding_sha256": sha256(base_binding_path),
        "source_parent": args.expected_base,
        "qualifier_head": args.expected_head,
        "trusted_policy_sha": args.trusted_policy_sha,
        "trusted_b2_workflow_git_blob": b2_objects["trusted_workflow"][0],
        "trusted_b2_workflow_sha256": b2_objects["trusted_workflow"][1],
        "trusted_adapter_git_blob": adapter_blob,
        "trusted_adapter_sha256": adapter_sha256,
        "trusted_b2_harness_git_blob": b2_objects["trusted_harness"][0],
        "trusted_b2_harness_sha256": b2_objects["trusted_harness"][1],
        "trusted_b2_subject_guard_git_blob": b2_objects["trusted_subject_guard"][0],
        "trusted_b2_subject_guard_sha256": b2_objects["trusted_subject_guard"][1],
        "trusted_b2_sealer_git_blob": b2_objects["trusted_sealer"][0],
        "trusted_b2_sealer_sha256": b2_objects["trusted_sealer"][1],
        **identity_after,
        "trusted_profile_validator_git_blob": validator_blob,
        "trusted_profile_validator_sha256": validator_sha256,
        "trusted_profile_sealer_git_blob": sealer_blob,
        "trusted_profile_sealer_sha256": sealer_sha256,
        "admission_mode": admission_mode,
        "live_pr_postflight": live_pr_postflight,
        "pull_request_number": pr_number,
        "github_run_id": args.github_run_id,
        "github_run_attempt": args.github_run_attempt,
    }
    with profile_binding_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    print(json.dumps(binding, sort_keys=True))
    return 0


def self_test() -> None:
    assert BINDING_SCHEMA == "symthaea.research-program-profile-binding.v1"
    assert AUTHORITY == "program-scope-binding-only"
    assert SCOPE_AUTHORITY == "trusted-program-profile-v1"
    assert BASE_BINDING_NAME != PROFILE_BINDING_NAME
    assert "trusted_adapter_sha256" in B2_BINDING_FIELDS
    assert "github_run_attempt" in B2_BINDING_FIELDS
    assert B2_WORKFLOW_PATH.endswith("research-bootstrap-qualification.yml")
    assert B2_SEALER_PATH.endswith("seal-research-qualification-binding.py")
    print("research_program_profile_seal_self_test=PASS")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--trusted-root")
    p.add_argument("--trusted-policy-sha")
    p.add_argument("--candidate-root")
    p.add_argument("--manifest")
    p.add_argument("--expected-head")
    p.add_argument("--expected-base")
    p.add_argument("--final-dir")
    p.add_argument("--repo")
    p.add_argument("--pr-number", type=int)
    p.add_argument("--github-run-id", type=int)
    p.add_argument("--github-run-attempt", type=int)
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
        "final_dir",
        "github_run_id",
        "github_run_attempt",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        print(f"program profile seal error: missing required arguments: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        return run(args)
    except Exception as error:
        print(f"research program profile seal error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
