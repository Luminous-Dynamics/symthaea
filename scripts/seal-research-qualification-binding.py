#!/usr/bin/env python3
"""Authenticated final seal for trusted research bootstrap qualification.

The candidate execution adapter writes to an explicitly unsealed staging set.
This module runs on a fresh runner, revalidates trusted policy, candidate
identity, staged evidence, and live PR currentness, then mints the final adapter
binding last.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

SEAL_SCHEMA = "symthaea.research-qualifier-binding.v2"
SEAL_PROFILE = "authenticated-postflight-v1"
SCOPE_AUTHORITY = "manifest-declared-only"
ADAPTER_PATH = "scripts/run-research-qualification-manifest.py"
SUBJECT_GUARD_PATH = "scripts/validate-research-qualification-subject.py"
SEALER_PATH = "scripts/seal-research-qualification-binding.py"
STAGED_FILES = frozenset(
    {
        "receipt.txt",
        "Cargo.lock.generated",
        "Cargo.lock.patch",
        "manifest-binding.json",
    }
)
COPY_FILES = ("receipt.txt", "Cargo.lock.generated", "Cargo.lock.patch")
MAX_STAGED_BYTES = {
    "receipt.txt": 256 * 1024,
    "Cargo.lock.generated": 16 * 1024 * 1024,
    "Cargo.lock.patch": 16 * 1024 * 1024,
    "manifest-binding.json": 256 * 1024,
}


class SealError(RuntimeError):
    pass


def load_adapter(trusted_root: Path) -> Any:
    path = trusted_root / ADAPTER_PATH
    if not path.is_file():
        raise SealError(f"trusted adapter missing: {ADAPTER_PATH}")
    spec = importlib.util.spec_from_file_location(
        "symthaea_research_qualification_adapter", path
    )
    if spec is None or spec.loader is None:
        raise SealError("could not construct trusted adapter import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_trusted_file_identity(
    trusted_root: Path, relative: str, executing_file: str | None = None
) -> str:
    unresolved = trusted_root / relative
    if unresolved.is_symlink():
        raise SealError(f"trusted policy path must not be a symlink: {relative}")
    expected = unresolved.resolve(strict=True)
    if not expected.is_file():
        raise SealError(f"trusted policy file missing: {relative}")
    if executing_file is not None and Path(executing_file).resolve() != expected:
        raise SealError(f"{relative} is not executing from the trusted checkout")
    return sha256(expected)


def run_subject_guard(
    trusted_root: Path,
    candidate_root: Path,
    manifest_path: str,
    expected_head: str,
    expected_base: str,
) -> None:
    guard = trusted_root / SUBJECT_GUARD_PATH
    completed = subprocess.run(
        [
            sys.executable,
            str(guard),
            "--candidate-root",
            str(candidate_root),
            "--manifest",
            manifest_path,
            "--expected-head",
            expected_head,
            "--expected-base",
            expected_base,
        ],
        cwd=trusted_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SealError(f"trusted subject guard failed: {detail}")


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError(f"invalid staged binding {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SealError("staged binding must be a JSON object")
    return payload


def verify_staged_evidence(
    staging: Path,
    adapter: Any,
    manifest: dict[str, Any],
    manifest_path: str,
    expected_head: str,
    expected_base: str,
    expected_harness_sha256: str,
    candidate_root: Path,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    observed = frozenset(path.name for path in staging.iterdir())
    if observed != STAGED_FILES:
        raise SealError(
            "unsealed staging file set mismatch: "
            f"expected {sorted(STAGED_FILES)}, observed {sorted(observed)}"
        )
    for name in STAGED_FILES:
        path = staging / name
        if path.is_symlink() or not path.is_file():
            raise SealError(f"staged evidence must be a regular non-symlink file: {name}")
        size = path.stat().st_size
        maximum = MAX_STAGED_BYTES[name]
        if size > maximum:
            raise SealError(
                f"staged evidence exceeds bounded size for {name}: {size} > {maximum}"
            )

    receipt = staging / "receipt.txt"
    adapter.validate_receipt(
        receipt,
        manifest,
        manifest_path,
        expected_head,
        expected_base,
        expected_harness_sha256,
    )
    scalars, lists = adapter.parse_receipt(receipt)

    generated_lock = staging / "Cargo.lock.generated"
    lock_patch = staging / "Cargo.lock.patch"
    if sha256(generated_lock) != scalars["generated_lock_sha256"]:
        raise SealError("Cargo.lock.generated bytes do not match receipt digest")
    if sha256(lock_patch) != scalars["lock_patch_sha256"]:
        raise SealError("Cargo.lock.patch bytes do not match receipt digest")

    expected_source_objects = [
        f"{path}:{adapter.git(candidate_root, 'rev-parse', f'{expected_base}:{path}')}"
        for path in manifest["source_paths"]
    ]
    if lists["source_object"] != expected_source_objects:
        raise SealError(
            "receipt source_object list does not equal exact source-parent Git objects"
        )
    return scalars, lists


def require_exact_staged_binding(
    provisional: dict[str, Any],
    adapter: Any,
    trusted: dict[str, str],
    manifest: dict[str, Any],
    manifest_path: str,
    expected_head: str,
    expected_base: str,
    receipt: Path,
) -> None:
    expected = {
        "schema": adapter.BINDING_SCHEMA,
        "authority": "adapter-binding-only",
        "scientific_claim": "NONE",
        "program": manifest["program"],
        "manifest_path": manifest_path,
        "manifest_sha256": adapter.canonical_manifest_digest(manifest),
        "source_parent": expected_base,
        "qualifier_head": expected_head,
        **trusted,
        "harness_receipt_sha256": sha256(receipt),
    }
    if provisional != expected:
        unknown = sorted(set(provisional) - set(expected))
        missing = sorted(set(expected) - set(provisional))
        mismatched = sorted(
            key
            for key in set(expected) & set(provisional)
            if expected[key] != provisional[key]
        )
        raise SealError(
            "staged binding does not equal the trusted adapter contract; "
            f"unknown={unknown}, missing={missing}, mismatched={mismatched}"
        )


def run(args: argparse.Namespace) -> int:
    trusted_root = Path(args.trusted_root).resolve(strict=True)
    candidate_root = Path(args.candidate_root).resolve(strict=True)
    staging = Path(args.staging_dir).resolve(strict=True)
    final = Path(args.final_dir).resolve(strict=False)

    adapter = load_adapter(trusted_root)
    adapter.assert_separate_roots(trusted_root, candidate_root)
    trusted = adapter.verify_trusted_checkout(trusted_root, args.trusted_policy_sha)
    trusted_sealer_sha256 = verify_trusted_file_identity(
        trusted_root, SEALER_PATH, __file__
    )
    trusted_subject_guard_sha256 = verify_trusted_file_identity(
        trusted_root, SUBJECT_GUARD_PATH
    )

    manifest_path = adapter.canonical_repo_path(args.manifest, "manifest path")
    run_subject_guard(
        trusted_root,
        candidate_root,
        manifest_path,
        args.expected_head,
        args.expected_base,
    )
    manifest = adapter.load_manifest(candidate_root, manifest_path)
    adapter.verify_candidate_subject(
        candidate_root,
        manifest,
        manifest_path,
        args.expected_head,
        args.expected_base,
    )

    for root, label in ((trusted_root, "trusted"), (candidate_root, "candidate")):
        if staging == root or root in staging.parents:
            raise SealError(f"staging directory must not be inside {label} checkout")
        if final == root or root in final.parents:
            raise SealError(
                f"final evidence directory must not be inside {label} checkout"
            )
    if final == staging or staging in final.parents or final in staging.parents:
        raise SealError("staging and final evidence directories must be disjoint")
    if final.exists():
        raise SealError("final evidence directory must not already exist")

    receipt = staging / "receipt.txt"
    receipt_scalars, _receipt_lists = verify_staged_evidence(
        staging,
        adapter,
        manifest,
        manifest_path,
        args.expected_head,
        args.expected_base,
        trusted["trusted_harness_sha256"],
        candidate_root,
    )
    provisional = read_json_object(staging / "manifest-binding.json")
    require_exact_staged_binding(
        provisional,
        adapter,
        trusted,
        manifest,
        manifest_path,
        args.expected_head,
        args.expected_base,
        receipt,
    )

    admission_mode = "workflow_dispatch"
    live_postflight = "NOT_APPLICABLE_MANUAL_DISPATCH"
    pr_number: int | None = None
    if args.pr_number is not None:
        if not args.repo:
            raise SealError("--repo is required when --pr-number is supplied")
        token = os.environ.get("GITHUB_TOKEN", "")
        adapter.verify_live_pr(
            args.repo,
            args.pr_number,
            token,
            args.expected_head,
            args.expected_base,
        )
        admission_mode = "pull_request_target"
        live_postflight = "PASS"
        pr_number = args.pr_number
    elif args.repo:
        raise SealError("--repo and --pr-number must be supplied together")

    if args.github_run_id <= 0 or args.github_run_attempt <= 0:
        raise SealError("GitHub run id and attempt must be positive integers")

    trusted_after = adapter.verify_trusted_checkout(
        trusted_root, args.trusted_policy_sha
    )
    if trusted_after != trusted:
        raise SealError("trusted policy bytes changed before final sealing")
    if (
        verify_trusted_file_identity(trusted_root, SEALER_PATH, __file__)
        != trusted_sealer_sha256
    ):
        raise SealError("trusted sealer bytes changed before final sealing")
    if (
        verify_trusted_file_identity(trusted_root, SUBJECT_GUARD_PATH)
        != trusted_subject_guard_sha256
    ):
        raise SealError("trusted subject guard bytes changed before final sealing")

    final.mkdir(parents=True, exist_ok=False)
    for name in COPY_FILES:
        shutil.copy2(staging / name, final / name)

    binding = {
        "schema": SEAL_SCHEMA,
        "seal_profile": SEAL_PROFILE,
        "authority": "adapter-binding-only",
        "scope_authority": SCOPE_AUTHORITY,
        "scientific_claim": "NONE",
        "program": manifest["program"],
        "manifest_path": manifest_path,
        "manifest_sha256": adapter.canonical_manifest_digest(manifest),
        "source_parent": args.expected_base,
        "qualifier_head": args.expected_head,
        **trusted_after,
        "trusted_subject_guard_sha256": trusted_subject_guard_sha256,
        "trusted_sealer_sha256": trusted_sealer_sha256,
        "harness_receipt_sha256": sha256(final / "receipt.txt"),
        "generated_lock_sha256": receipt_scalars["generated_lock_sha256"],
        "lock_patch_sha256": receipt_scalars["lock_patch_sha256"],
        "admission_mode": admission_mode,
        "live_pr_postflight": live_postflight,
        "pull_request_number": pr_number,
        "github_run_id": args.github_run_id,
        "github_run_attempt": args.github_run_attempt,
    }

    destination = final / "manifest-binding.json"
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    print(json.dumps(binding, sort_keys=True))
    return 0


def self_test() -> None:
    assert SEAL_SCHEMA == "symthaea.research-qualifier-binding.v2"
    assert SCOPE_AUTHORITY == "manifest-declared-only"
    assert "manifest-binding.json" in STAGED_FILES
    assert "manifest-binding.json" not in COPY_FILES
    assert len(STAGED_FILES) == 4
    assert len(COPY_FILES) == 3
    assert set(MAX_STAGED_BYTES) == STAGED_FILES
    assert SUBJECT_GUARD_PATH.endswith("validate-research-qualification-subject.py")
    assert SEALER_PATH.endswith("seal-research-qualification-binding.py")
    print("research_qualification_final_seal_self_test=PASS")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--trusted-root")
    p.add_argument("--trusted-policy-sha")
    p.add_argument("--candidate-root")
    p.add_argument("--manifest")
    p.add_argument("--expected-head")
    p.add_argument("--expected-base")
    p.add_argument("--staging-dir")
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
        "staging_dir",
        "final_dir",
        "github_run_id",
        "github_run_attempt",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        print(
            f"seal error: missing required arguments: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2
    try:
        return run(args)
    except Exception as error:
        print(f"research qualification seal error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
