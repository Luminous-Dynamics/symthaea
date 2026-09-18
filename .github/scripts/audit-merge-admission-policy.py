#!/usr/bin/env python3
"""Run merge-admission attestation against a versioned policy manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any

POLICY_SCHEMA = "symthaea.merge-admission-policy.v1"
DEFAULT_POLICY = ".github/governance/merge-admission-policy-v1.json"
ATTESTOR = pathlib.Path(__file__).with_name("audit-merge-admission.py")
API_VERSION = "2026-03-10"
ATTESTOR_TIMEOUT_SECS = 120


class PolicyError(ValueError):
    pass


def _require_bool(policy: dict[str, Any], key: str) -> bool:
    value = policy.get(key)
    if not isinstance(value, bool):
        raise PolicyError(f"{key} must be boolean")
    return value


def validate_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise PolicyError("policy root must be an object")
    if policy.get("schema") != POLICY_SCHEMA:
        raise PolicyError(f"schema must be {POLICY_SCHEMA!r}")

    policy_id = policy.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id or any(ch.isspace() for ch in policy_id):
        raise PolicyError("policy_id must be a non-empty whitespace-free string")

    target = policy.get("target")
    if not isinstance(target, str) or not target:
        raise PolicyError("target must be a non-empty string")

    checks = policy.get("required_checks")
    if not isinstance(checks, list) or not checks:
        raise PolicyError("required_checks must be a non-empty array")
    if any(not isinstance(item, str) or not item for item in checks):
        raise PolicyError("every required check must be a non-empty string")
    if len(set(checks)) != len(checks):
        raise PolicyError("required_checks must not contain duplicates")

    # v1 deliberately supports only the strong policy already implemented by
    # audit-merge-admission.py. A future relaxed/expanded policy requires a new
    # schema or attestor semantics; unsupported fields must not be ignored.
    required_true = (
        "require_pull_request",
        "block_force_push",
        "block_deletion",
        "require_no_bypass",
    )
    for key in required_true:
        if not _require_bool(policy, key):
            raise PolicyError(f"{key}=false is not supported by policy v1")

    allowed = {
        "schema",
        "policy_id",
        "target",
        "required_checks",
        *required_true,
    }
    unknown = sorted(set(policy) - allowed)
    if unknown:
        raise PolicyError(f"unknown policy fields: {', '.join(unknown)}")

    return policy


def load_policy(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PolicyError(f"policy file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise PolicyError(f"invalid policy JSON: {error}") from error
    return validate_policy(payload)


def policy_sha256(policy: dict[str, Any]) -> str:
    """Digest the validated policy's canonical JSON identity."""
    canonical = json.dumps(
        policy,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise PolicyError(f"unable to hash evaluator file {path}: {error}") from error


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": "symthaea-merge-admission-policy/1",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def resolve_target(repository: str, target: str) -> str:
    if target != "~DEFAULT_BRANCH":
        if target.startswith("refs/heads/"):
            return target.removeprefix("refs/heads/")
        if target.startswith("~"):
            raise PolicyError(f"unsupported symbolic target: {target}")
        return target

    base = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    request = urllib.request.Request(
        f"{base}/repos/{repository}", headers=_github_headers(), method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise PolicyError(f"default-branch lookup failed with HTTP {error.code}") from error
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        raise PolicyError(f"default-branch lookup failed: {error}") from error

    branch = payload.get("default_branch") if isinstance(payload, dict) else None
    if not isinstance(branch, str) or not branch:
        raise PolicyError("repository default_branch was not available")
    return branch


def run_attestor(
    repository: str,
    branch: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ATTESTOR),
        "--repository",
        repository,
        "--branch",
        branch,
    ]
    for check in policy["required_checks"]:
        command.extend(["--require-check", check])

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=ATTESTOR_TIMEOUT_SECS,
        )
    except subprocess.TimeoutExpired as error:
        raise PolicyError(
            f"underlying attestor exceeded {ATTESTOR_TIMEOUT_SECS}s timeout"
        ) from error
    except OSError as error:
        raise PolicyError(f"unable to execute underlying attestor: {error}") from error

    if completed.returncode != 0:
        raise PolicyError(
            f"underlying attestor failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PolicyError("underlying attestor did not emit valid JSON") from error
    if not isinstance(receipt, dict):
        raise PolicyError("underlying attestor receipt must be an object")
    return receipt


def bind_policy(
    receipt: dict[str, Any],
    policy: dict[str, Any],
    branch: str,
    *,
    wrapper_path: pathlib.Path | None = None,
    attestor_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    required = sorted(policy["required_checks"])
    observed = receipt.get("required_checks")
    if observed != required:
        raise PolicyError(
            f"attestor required-check echo mismatch: expected {required!r}, got {observed!r}"
        )
    if receipt.get("branch") != branch:
        raise PolicyError("attestor branch echo did not match the resolved policy target")

    wrapper_path = wrapper_path or pathlib.Path(__file__)
    attestor_path = attestor_path or ATTESTOR
    bound = dict(receipt)
    bound["schema"] = "symthaea.merge-admission-policy-receipt.v1"
    bound["policy"] = {
        "schema": policy["schema"],
        "policy_id": policy["policy_id"],
        "policy_sha256": policy_sha256(policy),
        "target": policy["target"],
        "resolved_branch": branch,
        "required_checks": required,
        "require_pull_request": policy["require_pull_request"],
        "block_force_push": policy["block_force_push"],
        "block_deletion": policy["block_deletion"],
        "require_no_bypass": policy["require_no_bypass"],
    }
    bound["evaluator"] = {
        "attestor_sha256": file_sha256(attestor_path),
        "policy_wrapper_sha256": file_sha256(wrapper_path),
    }
    bound["identity_note"] = (
        "SHA-256 values identify exact policy/evaluator bytes; they do not grant "
        "qualification, trust, or authority."
    )
    return bound


def self_test() -> None:
    valid = validate_policy(
        {
            "schema": POLICY_SCHEMA,
            "policy_id": "symthaea-main@v1",
            "target": "~DEFAULT_BRANCH",
            "required_checks": ["Governance Check"],
            "require_pull_request": True,
            "block_force_push": True,
            "block_deletion": True,
            "require_no_bypass": True,
        }
    )
    assert valid["policy_id"] == "symthaea-main@v1"
    digest = policy_sha256(valid)
    assert len(digest) == 64
    changed = dict(valid)
    changed["required_checks"] = ["Different Check"]
    assert policy_sha256(changed) != digest

    duplicate = dict(valid)
    duplicate["required_checks"] = ["x", "x"]
    try:
        validate_policy(duplicate)
    except PolicyError:
        pass
    else:
        raise AssertionError("duplicate checks were accepted")

    relaxed = dict(valid)
    relaxed["require_no_bypass"] = False
    try:
        validate_policy(relaxed)
    except PolicyError:
        pass
    else:
        raise AssertionError("unsupported relaxed policy was accepted")

    unknown = dict(valid)
    unknown["future_field"] = True
    try:
        validate_policy(unknown)
    except PolicyError:
        pass
    else:
        raise AssertionError("unknown policy field was ignored")

    # Use deterministic temporary evaluator bytes so the receipt identity test
    # does not depend on whichever source revision happens to invoke self_test.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        wrapper = root / "wrapper.py"
        attestor = root / "attestor.py"
        wrapper.write_bytes(b"wrapper-v1\n")
        attestor.write_bytes(b"attestor-v1\n")
        bound = bind_policy(
            {
                "schema": "symthaea.merge-admission-receipt.v1",
                "branch": "main",
                "required_checks": ["Governance Check"],
                "verdict": "unenforced",
            },
            valid,
            "main",
            wrapper_path=wrapper,
            attestor_path=attestor,
        )
    assert bound["schema"] == "symthaea.merge-admission-policy-receipt.v1"
    assert bound["policy"]["policy_id"] == "symthaea-main@v1"
    assert bound["policy"]["policy_sha256"] == digest
    assert len(bound["evaluator"]["attestor_sha256"]) == 64
    assert len(bound["evaluator"]["policy_wrapper_sha256"]) == 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--receipt")
    parser.add_argument("--require-verified", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("merge-admission policy wrapper self-test: PASS")
        return 0
    if not args.repository or "/" not in args.repository:
        print("--repository OWNER/REPO (or GITHUB_REPOSITORY) is required", file=sys.stderr)
        return 2

    try:
        policy = load_policy(pathlib.Path(args.policy))
        branch = resolve_target(args.repository, policy["target"])
        receipt = run_attestor(args.repository, branch, policy)
        bound = bind_policy(receipt, policy, branch)
    except PolicyError as error:
        print(f"merge-admission policy error: {error}", file=sys.stderr)
        return 2

    rendered = json.dumps(bound, indent=2, sort_keys=True)
    print(rendered)
    if args.receipt:
        pathlib.Path(args.receipt).write_text(rendered + "\n", encoding="utf-8")
    if args.require_verified and bound.get("verdict") != "verified_required_checks":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
