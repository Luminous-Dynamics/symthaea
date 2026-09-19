#!/usr/bin/env python3
"""Fresh-runner authenticated seal for research package-to-source-root closure."""

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

BINDING_SCHEMA = "symthaea.research-package-root-binding.v1"
SEAL_PROFILE = "authenticated-postflight-v1"
AUTHORITY = "package-root-binding-only"
SCIENTIFIC_CLAIM = "NONE"

ADAPTER_PATH = "scripts/run-research-qualification-manifest.py"
B2_WORKFLOW_PATH = ".github/workflows/research-bootstrap-qualification.yml"
B2_HARNESS_PATH = "scripts/qualify-research-crate.sh"
B2_SUBJECT_GUARD_PATH = "scripts/validate-research-qualification-subject.py"
B2_SEALER_PATH = "scripts/seal-research-qualification-binding.py"
C2_VALIDATOR_PATH = "scripts/validate-research-program-profile.py"
C2_SEALER_PATH = "scripts/seal-research-program-profile.py"
E2_VALIDATOR_PATH = "scripts/validate-research-package-root-closure.py"
E2_SEALER_PATH = "scripts/seal-research-package-root-closure.py"

B2_BINDING_NAME = "manifest-binding.json"
C2_BINDING_NAME = "program-profile-binding.json"
E2_CLOSURE_NAME = "package-root-closure.json"
E2_BINDING_NAME = "package-root-binding.json"
PRE_F2_FILES = frozenset(
    {
        "receipt.txt",
        "Cargo.lock.generated",
        "Cargo.lock.patch",
        B2_BINDING_NAME,
        C2_BINDING_NAME,
    }
)

C2_BINDING_FIELDS = frozenset(
    {
        "schema",
        "authority",
        "scope_authority",
        "scientific_claim",
        "program",
        "manifest_path",
        "manifest_sha256",
        "manifest_binding_sha256",
        "source_parent",
        "qualifier_head",
        "trusted_policy_sha",
        "trusted_b2_workflow_git_blob",
        "trusted_b2_workflow_sha256",
        "trusted_adapter_git_blob",
        "trusted_adapter_sha256",
        "trusted_b2_harness_git_blob",
        "trusted_b2_harness_sha256",
        "trusted_b2_subject_guard_git_blob",
        "trusted_b2_subject_guard_sha256",
        "trusted_b2_sealer_git_blob",
        "trusted_b2_sealer_sha256",
        "program_profile_path",
        "program_profile_git_blob",
        "program_profile_file_sha256",
        "program_profile_canonical_sha256",
        "trusted_profile_validator_git_blob",
        "trusted_profile_validator_sha256",
        "trusted_profile_sealer_git_blob",
        "trusted_profile_sealer_sha256",
        "admission_mode",
        "live_pr_postflight",
        "pull_request_number",
        "github_run_id",
        "github_run_attempt",
    }
)


class RootSealError(RuntimeError):
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
        raise RootSealError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_trusted_file(
    root: Path, relative: str, executing_file: str | None = None
) -> tuple[str, str]:
    root = root.resolve(strict=True)
    unresolved = root / relative
    if unresolved.is_symlink():
        raise RootSealError(f"trusted policy path must not be a symlink: {relative}")
    resolved = unresolved.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise RootSealError(f"trusted policy path is invalid: {relative}")
    if executing_file is not None and Path(executing_file).resolve() != resolved:
        raise RootSealError(f"{relative} is not executing from the trusted checkout")
    line = bootstrap_git(root, "ls-tree", "HEAD", "--", relative)
    if "\t" not in line:
        raise RootSealError(f"trusted policy path is not tracked: {relative}")
    metadata, observed = line.split("\t", 1)
    fields = metadata.split()
    if len(fields) != 3 or observed != relative or fields[0] != "100644" or fields[1] != "blob":
        raise RootSealError(f"trusted policy path must be exact 100644/blob: {relative}")
    blob = fields[2]
    if bootstrap_git(root, "hash-object", "--", relative) != blob:
        raise RootSealError(f"trusted policy worktree bytes differ from HEAD: {relative}")
    return blob, sha256(resolved)


def load_verified_module(root: Path, relative: str, module_name: str) -> Any:
    verify_trusted_file(root, relative)
    path = root / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RootSealError(f"could not construct import for {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json_object(path: Path, adapter: Any) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RootSealError(f"required evidence is not a regular non-symlink file: {path.name}")
    try:
        payload = adapter.parse_json(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RootSealError(f"invalid JSON evidence {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RootSealError(f"evidence root must be an object: {path.name}")
    return payload


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise RootSealError(f"{label} is not a lowercase SHA-256 digest")
    return value


def require_c2_binding(
    binding: dict[str, Any],
    *,
    closure: dict[str, Any],
    trusted_policy_sha: str,
    policy_objects: dict[str, tuple[str, str]],
    b2_binding_sha256: str,
    github_run_id: int,
    github_run_attempt: int,
    pr_number: int | None,
) -> None:
    if set(binding) != C2_BINDING_FIELDS:
        unknown = sorted(set(binding) - C2_BINDING_FIELDS)
        missing = sorted(C2_BINDING_FIELDS - set(binding))
        raise RootSealError(
            f"C2 binding field set mismatch; unknown={unknown}, missing={missing}"
        )

    admission_mode = "pull_request_target" if pr_number is not None else "workflow_dispatch"
    live_postflight = "PASS" if pr_number is not None else "NOT_APPLICABLE_MANUAL_DISPATCH"
    expected = {
        "schema": "symthaea.research-program-profile-binding.v1",
        "authority": "program-scope-binding-only",
        "scope_authority": "trusted-program-profile-v1",
        "scientific_claim": "NONE",
        "program": closure["program"],
        "manifest_path": closure["manifest_path"],
        "manifest_sha256": closure["manifest_sha256"],
        "manifest_binding_sha256": b2_binding_sha256,
        "source_parent": closure["source_parent"],
        "qualifier_head": closure["qualifier_head"],
        "trusted_policy_sha": trusted_policy_sha,
        "trusted_b2_workflow_git_blob": policy_objects["b2_workflow"][0],
        "trusted_b2_workflow_sha256": policy_objects["b2_workflow"][1],
        "trusted_adapter_git_blob": policy_objects["adapter"][0],
        "trusted_adapter_sha256": policy_objects["adapter"][1],
        "trusted_b2_harness_git_blob": policy_objects["b2_harness"][0],
        "trusted_b2_harness_sha256": policy_objects["b2_harness"][1],
        "trusted_b2_subject_guard_git_blob": policy_objects["b2_subject_guard"][0],
        "trusted_b2_subject_guard_sha256": policy_objects["b2_subject_guard"][1],
        "trusted_b2_sealer_git_blob": policy_objects["b2_sealer"][0],
        "trusted_b2_sealer_sha256": policy_objects["b2_sealer"][1],
        "program_profile_path": closure["program_profile_path"],
        "program_profile_git_blob": closure["program_profile_git_blob"],
        "program_profile_file_sha256": closure["program_profile_file_sha256"],
        "program_profile_canonical_sha256": closure["program_profile_canonical_sha256"],
        "trusted_profile_validator_git_blob": policy_objects["c2_validator"][0],
        "trusted_profile_validator_sha256": policy_objects["c2_validator"][1],
        "trusted_profile_sealer_git_blob": policy_objects["c2_sealer"][0],
        "trusted_profile_sealer_sha256": policy_objects["c2_sealer"][1],
        "admission_mode": admission_mode,
        "live_pr_postflight": live_postflight,
        "pull_request_number": pr_number,
        "github_run_id": github_run_id,
        "github_run_attempt": github_run_attempt,
    }
    for key, expected_value in expected.items():
        if binding.get(key) != expected_value:
            raise RootSealError(f"C2 binding {key} mismatch")

    for key in C2_BINDING_FIELDS:
        if key.endswith("_sha256"):
            require_sha256(binding[key], f"C2 binding {key}")


def run(args: argparse.Namespace) -> int:
    trusted_root = Path(args.trusted_root).resolve(strict=True)
    candidate_root = Path(args.candidate_root).resolve(strict=True)
    final_dir = Path(args.final_dir).resolve(strict=True)

    if trusted_root == candidate_root or trusted_root in candidate_root.parents or candidate_root in trusted_root.parents:
        raise RootSealError("trusted and candidate roots must be disjoint")
    for root, label in ((trusted_root, "trusted"), (candidate_root, "candidate")):
        if final_dir == root or root in final_dir.parents:
            raise RootSealError(f"final evidence directory must not be inside {label} checkout")

    if bootstrap_git(trusted_root, "rev-parse", "HEAD") != args.trusted_policy_sha:
        raise RootSealError("trusted checkout does not equal expected policy SHA")
    if args.github_run_id <= 0 or args.github_run_attempt <= 0:
        raise RootSealError("GitHub run id and attempt must be positive integers")

    observed_files = frozenset(path.name for path in final_dir.iterdir())
    if observed_files != PRE_F2_FILES:
        raise RootSealError(
            f"pre-F2 evidence file set mismatch: expected {sorted(PRE_F2_FILES)}, "
            f"observed {sorted(observed_files)}"
        )
    for name in PRE_F2_FILES:
        path = final_dir / name
        if path.is_symlink() or not path.is_file():
            raise RootSealError(f"pre-F2 evidence must be regular non-symlink file: {name}")

    policy_paths = {
        "b2_workflow": B2_WORKFLOW_PATH,
        "adapter": ADAPTER_PATH,
        "b2_harness": B2_HARNESS_PATH,
        "b2_subject_guard": B2_SUBJECT_GUARD_PATH,
        "b2_sealer": B2_SEALER_PATH,
        "c2_validator": C2_VALIDATOR_PATH,
        "c2_sealer": C2_SEALER_PATH,
        "e2_validator": E2_VALIDATOR_PATH,
        "e2_sealer": E2_SEALER_PATH,
    }
    policy_objects: dict[str, tuple[str, str]] = {}
    for key, relative in policy_paths.items():
        executing = __file__ if key == "e2_sealer" else None
        policy_objects[key] = verify_trusted_file(trusted_root, relative, executing)

    adapter = load_verified_module(
        trusted_root, ADAPTER_PATH, "symthaea_research_qualification_adapter_for_root_seal"
    )
    e2 = load_verified_module(
        trusted_root, E2_VALIDATOR_PATH, "symthaea_research_package_root_validator_for_seal"
    )
    adapter.assert_separate_roots(trusted_root, candidate_root)

    closure, closure_digest = e2.derive_closure(
        trusted_root,
        args.trusted_policy_sha,
        candidate_root,
        args.manifest,
        args.expected_head,
        args.expected_base,
    )

    b2_path = final_dir / B2_BINDING_NAME
    c2_path = final_dir / C2_BINDING_NAME
    closure_path = final_dir / E2_CLOSURE_NAME
    binding_path = final_dir / E2_BINDING_NAME
    b2 = read_json_object(b2_path, adapter)
    c2 = read_json_object(c2_path, adapter)
    b2_sha256 = sha256(b2_path)
    c2_sha256 = sha256(c2_path)
    require_sha256(b2_sha256, "B2 binding SHA-256")
    require_sha256(c2_sha256, "C2 binding SHA-256")

    if b2.get("schema") != "symthaea.research-qualifier-binding.v2":
        raise RootSealError("unexpected B2 binding schema")
    if b2.get("authority") != "adapter-binding-only":
        raise RootSealError("unexpected B2 authority")
    if b2.get("scope_authority") != "manifest-declared-only":
        raise RootSealError("unexpected B2 scope authority")
    if b2.get("scientific_claim") != "NONE":
        raise RootSealError("B2 scientific claim must remain NONE")

    require_c2_binding(
        c2,
        closure=closure,
        trusted_policy_sha=args.trusted_policy_sha,
        policy_objects=policy_objects,
        b2_binding_sha256=b2_sha256,
        github_run_id=args.github_run_id,
        github_run_attempt=args.github_run_attempt,
        pr_number=args.pr_number,
    )

    admission_mode = "workflow_dispatch"
    live_pr_postflight = "NOT_APPLICABLE_MANUAL_DISPATCH"
    pr_number: int | None = None
    if args.pr_number is not None:
        if not args.repo:
            raise RootSealError("--repo is required with --pr-number")
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
        raise RootSealError("--repo and --pr-number must be supplied together")

    # Re-read all trusted objects and independently re-derive closure immediately
    # before final publication.
    if bootstrap_git(trusted_root, "rev-parse", "HEAD") != args.trusted_policy_sha:
        raise RootSealError("trusted policy HEAD changed before E2 sealing")
    for key, relative in policy_paths.items():
        executing = __file__ if key == "e2_sealer" else None
        if verify_trusted_file(trusted_root, relative, executing) != policy_objects[key]:
            raise RootSealError(f"trusted policy object changed before E2 sealing: {key}")
    closure_after, digest_after = e2.derive_closure(
        trusted_root,
        args.trusted_policy_sha,
        candidate_root,
        args.manifest,
        args.expected_head,
        args.expected_base,
    )
    if closure_after != closure or digest_after != closure_digest:
        raise RootSealError("package-root closure changed before final publication")
    if sha256(b2_path) != b2_sha256 or sha256(c2_path) != c2_sha256:
        raise RootSealError("upstream binding bytes changed before E2 publication")

    with closure_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(closure_after, indent=2, sort_keys=True) + "\n")
    closure_file_sha256 = sha256(closure_path)

    binding = {
        "schema": BINDING_SCHEMA,
        "seal_profile": SEAL_PROFILE,
        "authority": AUTHORITY,
        "scientific_claim": SCIENTIFIC_CLAIM,
        "program": closure_after["program"],
        "manifest_path": closure_after["manifest_path"],
        "manifest_sha256": closure_after["manifest_sha256"],
        "source_parent": closure_after["source_parent"],
        "qualifier_head": closure_after["qualifier_head"],
        "trusted_policy_sha": args.trusted_policy_sha,
        "manifest_binding_sha256": b2_sha256,
        "program_profile_binding_sha256": c2_sha256,
        "program_profile_canonical_sha256": closure_after["program_profile_canonical_sha256"],
        "package_root_closure_sha256": closure_digest,
        "package_root_closure_file_sha256": closure_file_sha256,
        "package_root_count": len(closure_after["package_roots"]),
        "trusted_e2_validator_git_blob": policy_objects["e2_validator"][0],
        "trusted_e2_validator_sha256": policy_objects["e2_validator"][1],
        "trusted_e2_sealer_git_blob": policy_objects["e2_sealer"][0],
        "trusted_e2_sealer_sha256": policy_objects["e2_sealer"][1],
        "admission_mode": admission_mode,
        "live_pr_postflight": live_pr_postflight,
        "pull_request_number": pr_number,
        "github_run_id": args.github_run_id,
        "github_run_attempt": args.github_run_attempt,
    }
    with binding_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    print(json.dumps(binding, sort_keys=True))
    return 0


def self_test() -> None:
    assert BINDING_SCHEMA == "symthaea.research-package-root-binding.v1"
    assert SEAL_PROFILE == "authenticated-postflight-v1"
    assert AUTHORITY == "package-root-binding-only"
    assert SCIENTIFIC_CLAIM == "NONE"
    assert PRE_F2_FILES == frozenset(
        {"receipt.txt", "Cargo.lock.generated", "Cargo.lock.patch", B2_BINDING_NAME, C2_BINDING_NAME}
    )
    assert "trusted_profile_sealer_sha256" in C2_BINDING_FIELDS
    assert "github_run_attempt" in C2_BINDING_FIELDS
    print("research_package_root_seal_self_test=PASS")


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
        print(f"package-root seal error: missing required arguments: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        return run(args)
    except Exception as error:
        print(f"research package-root seal error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
