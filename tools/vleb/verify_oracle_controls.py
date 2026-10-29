#!/usr/bin/env python3
"""Replay VLEB's empty-patch and reference-patch oracle controls from one clean revision.

This tool verifies the *oracle controls*, not candidate performance. It requires
external digests for every trusted input and always emits analysis_only output.
It does not itself provide network isolation, VM/container isolation, or signed
runner attestation; invoke it inside the separately hardened evaluator sandbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

HEX = re.compile(r"^[a-f0-9]{64}$")
REVISION = re.compile(r"^[a-f0-9]{40}$")
COMMAND_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TOP_KEYS = {"schema_version", "commands"}
COMMAND_KEYS = {
    "command_id", "argv", "working_directory", "timeout_seconds",
    "test_start_pattern", "pass_pattern", "fail_pattern",
}


class ControlError(ValueError):
    """Trusted control inputs or replay evidence failed validation."""


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ControlError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ControlError(f"non-finite JSON constant {value!r} rejected")


def strict_json_loads(raw: bytes) -> Any:
    return json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )


def require_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ControlError(f"{label} must be a lowercase SHA-256 digest")


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing, extra = sorted(expected - set(value)), sorted(set(value) - expected)
    if missing or extra:
        raise ControlError(f"{label} keys invalid: missing={missing}, unexpected={extra}")


def validate_manifest(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        raise ControlError("oracle control manifest root must be an object")
    exact_keys(value, TOP_KEYS, "oracle control manifest")
    if value["schema_version"] != "vleb-oracle-control-v1":
        raise ControlError("unsupported oracle control manifest schema_version")
    commands = value["commands"]
    if not isinstance(commands, list) or not commands:
        raise ControlError("commands must be a non-empty list")
    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    for index, row in enumerate(commands):
        label = f"commands[{index}]"
        if not isinstance(row, dict):
            raise ControlError(f"{label} must be an object")
        exact_keys(row, COMMAND_KEYS, label)
        command_id = row["command_id"]
        if not isinstance(command_id, str) or not COMMAND_ID.fullmatch(command_id):
            raise ControlError(f"{label}.command_id is invalid")
        if command_id in seen:
            raise ControlError(f"duplicate command_id {command_id!r}")
        seen.add(command_id)

        argv = row["argv"]
        if (not isinstance(argv, list) or not argv
                or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in argv)):
            raise ControlError(f"{label}.argv must be a non-empty list of non-empty strings")
        cwd = row["working_directory"]
        if not isinstance(cwd, str) or "\\" in cwd or "\x00" in cwd:
            raise ControlError(f"{label}.working_directory must be a canonical POSIX relative path")
        if cwd != ".":
            relative = PurePosixPath(cwd)
            if relative.is_absolute() or ".." in relative.parts or "." in relative.parts or str(relative) != cwd:
                raise ControlError(f"{label}.working_directory is unsafe or non-canonical")
        timeout = row["timeout_seconds"]
        if type(timeout) is not int or not 1 <= timeout <= 3600:
            raise ControlError(f"{label}.timeout_seconds must be an integer in [1, 3600]")
        for key in ("test_start_pattern", "pass_pattern", "fail_pattern"):
            pattern = row[key]
            if not isinstance(pattern, str) or not pattern:
                raise ControlError(f"{label}.{key} must be a non-empty regex")
            try:
                re.compile(pattern, re.MULTILINE)
            except re.error as exc:
                raise ControlError(f"{label}.{key} is invalid regex: {exc}") from exc
        if row["pass_pattern"] == row["fail_pattern"]:
            raise ControlError(f"{label} pass/fail patterns must differ")
        checked.append(dict(row))
    return checked


def _git(source: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(source), "-c", "core.hooksPath=/dev/null", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if result.returncode != 0:
        raise ControlError(f"git {' '.join(args)} failed ({result.returncode}): {result.stdout[-2000:]}")
    return result


def verify_source_repository(source_repo: Path, expected_revision: str) -> Path:
    if not REVISION.fullmatch(expected_revision):
        raise ControlError("expected source revision must be a lowercase 40-character commit SHA")
    root = source_repo.resolve(strict=True)
    if not root.is_dir():
        raise ControlError("source repository must be a directory")
    observed = _git(root, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
    if observed != expected_revision:
        raise ControlError(f"source HEAD {observed} differs from externally expected revision {expected_revision}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout
    if status:
        raise ControlError("source worktree must be clean before creating oracle-control worktrees")
    return root


def _path_inside(root: Path, relative: str, *, allow_dot: bool = False) -> Path:
    if allow_dot and relative == ".":
        return root
    rel = PurePosixPath(relative)
    if (not relative or "\\" in relative or "\x00" in relative or rel.is_absolute()
            or ".." in rel.parts or "." in rel.parts or str(rel) != relative):
        raise ControlError(f"unsafe relative path {relative!r}")
    cursor = root
    for component in rel.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ControlError(f"symlink path component rejected: {relative!r}")
    resolved = cursor.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ControlError(f"path escapes workspace: {relative!r}")
    return resolved


def extract_test_bundle(bundle_raw: bytes, workspace: Path) -> list[dict[str, Any]]:
    try:
        archive = zipfile.ZipFile(__import__("io").BytesIO(bundle_raw), "r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ControlError(f"test bundle is not a valid ZIP archive: {exc}") from exc

    files: list[dict[str, Any]] = []
    names: set[str] = set()
    try:
        with archive:
            for info in archive.infolist():
                name = info.filename
                if name.endswith("/"):
                    relative_name = name[:-1]
                    if not relative_name:
                        continue
                else:
                    relative_name = name
                rel = PurePosixPath(relative_name)
                if (not relative_name or "\\" in relative_name or "\x00" in relative_name
                        or rel.is_absolute() or ".." in rel.parts or "." in rel.parts
                        or str(rel) != relative_name or ".git" in rel.parts):
                    raise ControlError(f"unsafe test bundle path {name!r}")
                if relative_name in names:
                    raise ControlError(f"duplicate test bundle path {relative_name!r}")
                names.add(relative_name)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ControlError(f"symlink entry in test bundle rejected: {name!r}")
                destination = workspace.joinpath(*rel.parts)
                if info.is_dir():
                    if destination.is_symlink():
                        raise ControlError(f"test bundle directory conflicts with symlink: {name!r}")
                    if destination.exists() and not destination.is_dir():
                        raise ControlError(f"test bundle directory conflicts with existing file: {name!r}")
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if destination.exists() or destination.is_symlink():
                    raise ControlError(f"test bundle must not overwrite source files: {name!r}")
                parent = destination.parent
                cursor = workspace
                for component in rel.parts[:-1]:
                    cursor = cursor / component
                    if cursor.is_symlink():
                        raise ControlError(f"symlink directory in test bundle path rejected: {name!r}")
                    if cursor.exists() and not cursor.is_dir():
                        raise ControlError(f"test bundle parent is not a directory: {name!r}")
                    cursor.mkdir(exist_ok=True)
                resolved_parent = parent.resolve(strict=True)
                if not resolved_parent.is_relative_to(workspace.resolve(strict=True)):
                    raise ControlError(f"test bundle path escapes workspace: {name!r}")
                raw = archive.read(info)
                destination.write_bytes(raw)
                # Extracted tests are never permitted to become executable entrypoints.
                destination.chmod(0o644)
                files.append({"path": relative_name, "sha256": digest(raw), "size_bytes": len(raw)})
    except (RuntimeError, zipfile.BadZipFile) as exc:
        raise ControlError(f"test bundle extraction failed: {exc}") from exc
    if not files:
        raise ControlError("test bundle must contain at least one regular file")
    return files


def _worktree_add(source_repo: Path, revision: str, destination: Path) -> None:
    subprocess.run(
        ["git", "-C", str(source_repo), "-c", "core.hooksPath=/dev/null",
         "worktree", "add", "--detach", "--no-checkout", str(destination), revision],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    _git(destination, "checkout", "--detach", revision)
    _git(destination, "config", "core.hooksPath", "/dev/null")


def _run_arm(
    workspace: Path,
    commands: list[dict[str, Any]],
    *,
    arm_name: str,
    evidence_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    issues: list[str] = []
    for command in commands:
        command_id = command["command_id"]
        try:
            cwd = _path_inside(workspace, command["working_directory"], allow_dot=True)
        except (OSError, ControlError) as exc:
            issues.append(f"{arm_name}/{command_id}: unsafe or missing working directory: {exc}")
            continue
        combined_timeout = False
        try:
            completed = subprocess.run(
                command["argv"],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=command["timeout_seconds"],
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
            output = completed.stdout
            returncode: int | None = completed.returncode
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            returncode = None
            combined_timeout = True

        start_seen = re.search(command["test_start_pattern"], output, re.MULTILINE) is not None
        pass_seen = re.search(command["pass_pattern"], output, re.MULTILINE) is not None
        fail_seen = re.search(command["fail_pattern"], output, re.MULTILINE) is not None
        if combined_timeout:
            observed = "unresolved"
        elif not start_seen or pass_seen == fail_seen:
            observed = "unresolved"
        elif pass_seen and returncode == 0:
            observed = "passed"
        elif fail_seen and returncode != 0:
            observed = "failed"
        else:
            observed = "unresolved"

        command_dir = evidence_root / arm_name
        command_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = command_dir / f"{command_id}.stdout"
        stdout_path.write_text(output, encoding="utf-8")
        record = {
            "command_id": command_id,
            "argv": command["argv"],
            "working_directory": command["working_directory"],
            "timeout_seconds": command["timeout_seconds"],
            "return_code": returncode,
            "timed_out": combined_timeout,
            "test_start_seen": start_seen,
            "pass_marker_seen": pass_seen,
            "fail_marker_seen": fail_seen,
            "observed_test_outcome": observed,
            "combined_output_artifact": str(stdout_path.relative_to(evidence_root)),
            "combined_output_sha256": digest(output.encode("utf-8")),
        }
        results.append(record)
        if observed == "unresolved":
            issues.append(f"{arm_name}/{command_id}: test execution/summary outcome unresolved")
    return results, issues


def run_two_sided_controls(
    *,
    source_repo: Path,
    expected_source_revision: str,
    manifest_raw: bytes,
    expected_manifest_sha256: str,
    reference_patch_raw: bytes,
    expected_reference_patch_sha256: str,
    test_bundle_raw: bytes,
    expected_test_bundle_sha256: str,
    expected_verifier_sha256: str,
    evidence_root: Path,
) -> dict[str, Any]:
    for value, label in (
        (expected_manifest_sha256, "expected manifest SHA-256"),
        (expected_reference_patch_sha256, "expected reference patch SHA-256"),
        (expected_test_bundle_sha256, "expected test bundle SHA-256"),
        (expected_verifier_sha256, "expected verifier SHA-256"),
    ):
        require_digest(value, label)
    if digest(manifest_raw) != expected_manifest_sha256:
        raise ControlError("oracle control manifest differs from evaluator-pinned SHA-256")
    if digest(reference_patch_raw) != expected_reference_patch_sha256:
        raise ControlError("reference patch differs from evaluator-pinned SHA-256")
    if digest(test_bundle_raw) != expected_test_bundle_sha256:
        raise ControlError("test bundle differs from evaluator-pinned SHA-256")
    verifier_sha = digest(Path(__file__).read_bytes())
    if verifier_sha != expected_verifier_sha256:
        raise ControlError("oracle-control verifier source differs from evaluator-pinned SHA-256")
    manifest = strict_json_loads(manifest_raw)
    commands = validate_manifest(manifest)
    source = verify_source_repository(source_repo, expected_source_revision)

    evidence_path = evidence_root.resolve(strict=False)
    if evidence_path.is_relative_to(source):
        raise ControlError("evidence output directory must be outside the source repository")
    if evidence_root.exists() and (not evidence_root.is_dir() or any(evidence_root.iterdir())):
        raise ControlError("evidence output directory must be new or empty")
    evidence_root.mkdir(parents=True, exist_ok=True)
    baseline_results: list[dict[str, Any]] = []
    reference_results: list[dict[str, Any]] = []
    bundle_manifest: list[dict[str, Any]] = []
    git_worktrees: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="vleb-oracle-controls-") as temporary:
        temp_root = Path(temporary)
        baseline = temp_root / "baseline"
        reference = temp_root / "reference"
        try:
            _worktree_add(source, expected_source_revision, baseline)
            git_worktrees.append(baseline)
            _worktree_add(source, expected_source_revision, reference)
            git_worktrees.append(reference)

            bundle_manifest.append({"arm": "baseline", "files": extract_test_bundle(test_bundle_raw, baseline)})
            # Extracting the same trusted tests independently ensures both arms use identical test bytes.
            # A separate digest per extracted file is emitted for later audit.
            reference_bundle = extract_test_bundle(test_bundle_raw, reference)
            bundle_manifest.append({"arm": "reference", "files": reference_bundle})
            if reference_bundle != bundle_manifest[0]["files"]:
                raise ControlError("test bundle extraction differed between baseline and reference worktrees")

            patch_file = temp_root / "reference.patch"
            patch_file.write_bytes(reference_patch_raw)
            check = subprocess.run(
                ["git", "-C", str(reference), "-c", "core.hooksPath=/dev/null",
                 "apply", "--check", "--whitespace=error", str(patch_file)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                text=True, encoding="utf-8", errors="replace", timeout=60,
            )
            if check.returncode != 0:
                raise ControlError(f"reference patch does not apply cleanly to locked revision: {check.stdout[-2000:]}")
            apply = subprocess.run(
                ["git", "-C", str(reference), "-c", "core.hooksPath=/dev/null",
                 "apply", "--whitespace=error", str(patch_file)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                text=True, encoding="utf-8", errors="replace", timeout=60,
            )
            if apply.returncode != 0:
                raise ControlError(f"reference patch application failed: {apply.stdout[-2000:]}")

            baseline_results, baseline_issues = _run_arm(
                baseline, commands, arm_name="baseline", evidence_root=evidence_root,
            )
            reference_results, reference_issues = _run_arm(
                reference, commands, arm_name="reference", evidence_root=evidence_root,
            )

            baseline_failed = any(row["observed_test_outcome"] == "failed" for row in baseline_results)
            reference_all_passed = (
                len(reference_results) == len(commands)
                and all(row["observed_test_outcome"] == "passed" for row in reference_results)
            )
            baseline_complete = (
                len(baseline_results) == len(commands)
                and all(row["observed_test_outcome"] in ("passed", "failed") for row in baseline_results)
            )
            controls_passed = baseline_failed and baseline_complete and reference_all_passed
            issues = baseline_issues + reference_issues
            if not baseline_failed:
                issues.append("empty-patch baseline did not demonstrate a test-level failure")
            if not baseline_complete:
                issues.append("one or more empty-patch test commands were unresolved")
            if not reference_all_passed:
                issues.append("reference patch did not pass every preregistered test command")
            return {
                "schema_version": "vleb-oracle-control-result-v1",
                "valid": controls_passed and not issues,
                "two_sided_gate_passed": controls_passed and not issues,
                "baseline_empty_patch_must_fail": True,
                "reference_patch_must_pass": True,
                "source_revision": expected_source_revision,
                "source_repository_head_checked": True,
                "manifest_sha256": digest(manifest_raw),
                "reference_patch_sha256": digest(reference_patch_raw),
                "test_bundle_sha256": digest(test_bundle_raw),
                "verifier_sha256": verifier_sha,
                "test_bundle_files_by_arm": bundle_manifest,
                "baseline_results": baseline_results,
                "reference_results": reference_results,
                "baseline_test_failure_count": sum(row["observed_test_outcome"] == "failed" for row in baseline_results),
                "reference_passed_command_count": sum(row["observed_test_outcome"] == "passed" for row in reference_results),
                "command_count": len(commands),
                "issues": issues,
                "analysis_only": True,
                "score_eligible": False,
                "runner_attestation_required": True,
                "network_isolation_independently_verified": False,
                "limitations": [
                    "The tool proves the pinned empty-patch/reference-patch test controls ran on isolated Git worktrees of one revision; it does not run or score a candidate submission.",
                    "A test bundle must contain independent test files and cannot overwrite any file already present in the source revision.",
                    "The command manifest must be authored and digest-pinned by a trusted evaluator; its regex markers must identify genuine test-start and test-summary lines for the chosen test framework.",
                    "The tool does not enforce network isolation, VM/container isolation, resource limits, or signed runner provenance. Protected execution infrastructure must attest these separately.",
                    "The output is analysis-only and cannot establish benchmark-score eligibility.",
                ],
            }
        finally:
            for worktree in reversed(git_worktrees):
                subprocess.run(
                    ["git", "-C", str(source), "-c", "core.hooksPath=/dev/null",
                     "worktree", "remove", "--force", str(worktree)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                    text=True, encoding="utf-8", errors="replace", timeout=60,
                )
            subprocess.run(
                ["git", "-C", str(source), "-c", "core.hooksPath=/dev/null", "worktree", "prune"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                timeout=60,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--reference-patch", type=Path, required=True)
    parser.add_argument("--expected-reference-patch-sha256", required=True)
    parser.add_argument("--test-bundle", type=Path, required=True)
    parser.add_argument("--expected-test-bundle-sha256", required=True)
    parser.add_argument("--expected-verifier-sha256", required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        result = run_two_sided_controls(
            source_repo=args.source_repo,
            expected_source_revision=args.expected_source_revision,
            manifest_raw=args.manifest.read_bytes(),
            expected_manifest_sha256=args.expected_manifest_sha256,
            reference_patch_raw=args.reference_patch.read_bytes(),
            expected_reference_patch_sha256=args.expected_reference_patch_sha256,
            test_bundle_raw=args.test_bundle.read_bytes(),
            expected_test_bundle_sha256=args.expected_test_bundle_sha256,
            expected_verifier_sha256=args.expected_verifier_sha256,
            evidence_root=args.evidence_dir,
        )
        exit_code = 0 if result["two_sided_gate_passed"] else 1
    except (OSError, UnicodeError, ControlError, zipfile.BadZipFile, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        result = {
            "schema_version": "vleb-oracle-control-result-v1",
            "valid": False,
            "two_sided_gate_passed": False,
            "analysis_only": True,
            "score_eligible": False,
            "runner_attestation_required": True,
            "network_isolation_independently_verified": False,
            "issues": [{"code": "oracle_control_replay_rejected", "message": str(exc)}],
        }
        exit_code = 1

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
