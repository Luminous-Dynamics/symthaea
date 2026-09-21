#!/usr/bin/env python3
"""QUAL-PHI-SEM-001A: exact execution qualifier for the Phi authority-flow inventory.

This script lives on a never-merge qualifier commit that must be a direct child
of the frozen product subject. It creates a detached worktree at the exact
product commit, executes the product audit twice, verifies byte determinism and
claim-limited semantic sanity, and emits a machine-readable qualification
receipt plus stdout/stderr evidence for every command gate.

PASS_INVENTORY means only that the inventory mechanism executed successfully
against the exact frozen subject under this qualifier profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QUALIFIER_SCRIPT = "scripts/qualify_phi_sem_001a.py"
QUALIFIER_PROFILE_ID = "qual-phi-sem-001a-v1"

PRODUCT_COMMIT = "96befed9c734ace707c07fd2542d47177270bd6f"
PRODUCT_TREE = "91486e516cd2cb95f05b03a35037eb82677c8c37"
PRODUCT_AUDIT_PATH = "scripts/audit_phi_authority_flow.py"
PRODUCT_AUDIT_BLOB = "5fbbd183e3cc69ba2080633a6b613b95772739e9"
PRODUCT_PROFILE_ID = "phi-sem-001a-lexical-v1"


class QualificationFailure(RuntimeError):
    """A demonstrated failure of the exact qualification contract."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_text(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def tracked_blob(cwd: Path, path: str) -> str:
    output = git_text(cwd, "ls-files", "-s", "--", path)
    if not output:
        raise QualificationFailure(f"tracked path missing: {path}")
    lines = output.splitlines()
    if len(lines) != 1:
        raise QualificationFailure(f"expected one index entry for {path}, got {len(lines)}")
    meta, listed_path = lines[0].split("\t", 1)
    if listed_path != path:
        raise QualificationFailure(
            f"index path mismatch for {path}: returned {listed_path}"
        )
    _mode, blob, stage = meta.split()
    if stage != "0":
        raise QualificationFailure(f"unmerged index entry for {path}")
    return blob


def check(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationFailure(message)


class EvidenceRecorder:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.gates: list[dict[str, Any]] = []

    def record_command(
        self,
        name: str,
        command: list[str],
        cwd: Path,
        *,
        env: dict[str, str] | None = None,
        require_success: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        result = subprocess.run(
            command,
            cwd=cwd,
            env=merged_env,
            capture_output=True,
        )

        stdout_path = self.directory / f"{name}.stdout"
        stderr_path = self.directory / f"{name}.stderr"
        stdout_path.write_bytes(result.stdout)
        stderr_path.write_bytes(result.stderr)

        self.gates.append(
            {
                "name": name,
                "kind": "command",
                "command": command,
                "cwd_role": "product" if cwd != ROOT else "qualifier",
                "returncode": result.returncode,
                "stdout_file": stdout_path.name,
                "stdout_sha256": sha256_bytes(result.stdout),
                "stdout_bytes": len(result.stdout),
                "stderr_file": stderr_path.name,
                "stderr_sha256": sha256_bytes(result.stderr),
                "stderr_bytes": len(result.stderr),
            }
        )

        if require_success and result.returncode != 0:
            raise QualificationFailure(
                f"gate {name} failed with exit status {result.returncode}"
            )
        return result

    def record_check(self, name: str, passed: bool, details: dict[str, Any]) -> None:
        self.gates.append(
            {
                "name": name,
                "kind": "check",
                "passed": passed,
                "details": details,
            }
        )
        if not passed:
            raise QualificationFailure(f"check failed: {name}")


def qualifier_identity() -> dict[str, Any]:
    head = git_text(ROOT, "rev-parse", "HEAD")
    tree = git_text(ROOT, "rev-parse", "HEAD^{tree}")
    parent_line = git_text(ROOT, "rev-list", "--parents", "-n", "1", "HEAD")
    parent_tokens = parent_line.split()
    check(len(parent_tokens) == 2, "qualifier commit must have exactly one parent")
    parent = parent_tokens[1]
    check(
        parent == PRODUCT_COMMIT,
        f"qualifier parent drift: expected {PRODUCT_COMMIT}, got {parent}",
    )

    blob = tracked_blob(ROOT, QUALIFIER_SCRIPT)
    script_bytes = (ROOT / QUALIFIER_SCRIPT).read_bytes()
    return {
        "commit": head,
        "tree": tree,
        "parent": parent,
        "script_path": QUALIFIER_SCRIPT,
        "script_git_blob": blob,
        "script_sha256": sha256_bytes(script_bytes),
        "script_bytes": len(script_bytes),
    }


def parse_report(report_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationFailure(f"invalid inventory JSON: {exc}") from exc
    check(isinstance(value, dict), "inventory JSON root must be an object")
    return value


def validate_report(
    report: dict[str, Any], expected_audit_sha256: str, expected_audit_bytes: int
) -> dict[str, Any]:
    source = report.get("source")
    check(isinstance(source, dict), "report.source must be an object")
    check(source.get("commit") == PRODUCT_COMMIT, "report source commit mismatch")
    check(source.get("tree") == PRODUCT_TREE, "report source tree mismatch")

    check(report.get("report_version") == 1, "unexpected report_version")
    check(report.get("profile_id") == PRODUCT_PROFILE_ID, "product profile mismatch")

    artifact = report.get("audit_artifact")
    check(isinstance(artifact, dict), "report.audit_artifact must be an object")
    check(artifact.get("path") == PRODUCT_AUDIT_PATH, "audit artifact path mismatch")
    check(artifact.get("git_blob") == PRODUCT_AUDIT_BLOB, "audit artifact blob mismatch")
    check(
        artifact.get("sha256") == expected_audit_sha256,
        "audit artifact SHA-256 mismatch",
    )
    check(
        artifact.get("bytes") == expected_audit_bytes,
        "audit artifact byte length mismatch",
    )

    missing = report.get("missing_mandatory_witness_selectors")
    check(missing == [], f"mandatory witness selectors missing: {missing!r}")

    summary = report.get("summary")
    check(isinstance(summary, dict), "report.summary must be an object")
    matched_paths = summary.get("matched_paths")
    matched_lines = summary.get("matched_lines")
    check(isinstance(matched_paths, int) and matched_paths > 0, "matched_paths must be > 0")
    check(isinstance(matched_lines, int) and matched_lines > 0, "matched_lines must be > 0")

    oracle = report.get("phi_oracle_duplicate_check")
    check(isinstance(oracle, dict), "phi_oracle_duplicate_check must be an object")
    for required in (
        "workspace_active_candidate",
        "root_duplicate_candidate",
        "differences",
        "cargo_dependency_edges",
    ):
        check(required in oracle, f"phi_oracle_duplicate_check missing {required}")

    differences = oracle.get("differences")
    edges = oracle.get("cargo_dependency_edges")
    check(isinstance(differences, list), "oracle differences must be a list")
    check(isinstance(edges, list), "oracle dependency edges must be a list")

    return {
        "matched_paths": matched_paths,
        "matched_lines": matched_lines,
        "categories": summary.get("categories"),
        "profile_sha256": report.get("profile_sha256"),
        "oracle_byte_identity_equal": oracle.get("byte_identity_equal"),
        "oracle_difference_count": len(differences),
        "oracle_dependency_edge_count": len(edges),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=ROOT / "target" / "qualification" / "phi-sem-001a",
        help="directory for report copies, logs, and qualification receipt",
    )
    args = parser.parse_args()
    evidence_dir = args.evidence_dir.resolve()

    receipt: dict[str, Any] = {
        "qualifier_profile_id": QUALIFIER_PROFILE_ID,
        "status": "FAIL",
        "claim_ceiling": {
            "establishes_on_pass": [
                "exact PHI-SEM-001A product subject executed",
                "mandatory lexical witness selectors present",
                "two repeated inventory reports were byte-identical",
            ],
            "does_not_establish": [
                "runtime reachability completeness",
                "estimator correctness",
                "IIT validity",
                "consciousness",
                "epistemic confidence",
                "execution or governance authority",
            ],
        },
        "product_subject": {
            "commit": PRODUCT_COMMIT,
            "tree": PRODUCT_TREE,
            "audit_path": PRODUCT_AUDIT_PATH,
            "audit_git_blob": PRODUCT_AUDIT_BLOB,
            "profile_id": PRODUCT_PROFILE_ID,
        },
    }

    recorder = EvidenceRecorder(evidence_dir)
    temp_root: Path | None = None
    product_worktree: Path | None = None

    try:
        receipt["qualifier_subject"] = qualifier_identity()

        recorder.record_command(
            "qualifier_git_status",
            ["git", "status", "--porcelain", "--untracked-files=no"],
            ROOT,
        )
        qualifier_status = (evidence_dir / "qualifier_git_status.stdout").read_bytes()
        recorder.record_check(
            "qualifier_tracked_worktree_clean",
            qualifier_status == b"",
            {"stdout_bytes": len(qualifier_status)},
        )

        toolchain: dict[str, str] = {
            "python_runtime": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "platform": platform.platform(),
        }
        git_version = recorder.record_command(
            "git_version", ["git", "--version"], ROOT
        ).stdout.decode("utf-8", errors="replace").strip()
        uname = recorder.record_command(
            "uname", ["uname", "-a"], ROOT
        ).stdout.decode("utf-8", errors="replace").strip()
        python_version = recorder.record_command(
            "python_version", [sys.executable, "--version"], ROOT
        )
        toolchain["git"] = git_version
        toolchain["uname"] = uname
        toolchain["python_command"] = (
            python_version.stdout + python_version.stderr
        ).decode("utf-8", errors="replace").strip()
        receipt["toolchain"] = toolchain

        temp_root = Path(tempfile.mkdtemp(prefix="qual-phi-sem-001a-"))
        product_worktree = temp_root / "product"
        recorder.record_command(
            "worktree_add",
            ["git", "worktree", "add", "--detach", str(product_worktree), PRODUCT_COMMIT],
            ROOT,
        )

        product_head = git_text(product_worktree, "rev-parse", "HEAD")
        product_tree = git_text(product_worktree, "rev-parse", "HEAD^{tree}")
        recorder.record_check(
            "exact_product_subject",
            product_head == PRODUCT_COMMIT and product_tree == PRODUCT_TREE,
            {
                "observed_commit": product_head,
                "observed_tree": product_tree,
                "expected_commit": PRODUCT_COMMIT,
                "expected_tree": PRODUCT_TREE,
            },
        )

        audit_blob = tracked_blob(product_worktree, PRODUCT_AUDIT_PATH)
        audit_bytes = (product_worktree / PRODUCT_AUDIT_PATH).read_bytes()
        audit_sha256 = sha256_bytes(audit_bytes)
        recorder.record_check(
            "exact_audit_artifact",
            audit_blob == PRODUCT_AUDIT_BLOB,
            {
                "path": PRODUCT_AUDIT_PATH,
                "observed_git_blob": audit_blob,
                "expected_git_blob": PRODUCT_AUDIT_BLOB,
                "sha256": audit_sha256,
                "bytes": len(audit_bytes),
            },
        )

        product_status = recorder.record_command(
            "product_git_status_pre",
            ["git", "status", "--porcelain", "--untracked-files=no"],
            product_worktree,
        ).stdout
        recorder.record_check(
            "product_tracked_worktree_clean_pre",
            product_status == b"",
            {"stdout_bytes": len(product_status)},
        )

        pycache = temp_root / "pycache"
        compile_env = {"PYTHONPYCACHEPREFIX": str(pycache)}
        recorder.record_command(
            "py_compile",
            [sys.executable, "-m", "py_compile", PRODUCT_AUDIT_PATH],
            product_worktree,
            env=compile_env,
        )
        recorder.record_command(
            "audit_help",
            [sys.executable, PRODUCT_AUDIT_PATH, "--help"],
            product_worktree,
            env=compile_env,
        )

        run1 = evidence_dir / "phi_sem_001a_inventory_v1.run1.json"
        run2 = evidence_dir / "phi_sem_001a_inventory_v1.run2.json"

        recorder.record_command(
            "inventory_run1",
            [
                sys.executable,
                PRODUCT_AUDIT_PATH,
                "--pretty",
                "--output",
                str(run1),
            ],
            product_worktree,
            env=compile_env,
        )
        recorder.record_command(
            "inventory_run2",
            [
                sys.executable,
                PRODUCT_AUDIT_PATH,
                "--pretty",
                "--output",
                str(run2),
            ],
            product_worktree,
            env=compile_env,
        )

        run1_bytes = run1.read_bytes()
        run2_bytes = run2.read_bytes()
        recorder.record_check(
            "inventory_byte_determinism",
            run1_bytes == run2_bytes,
            {
                "run1_sha256": sha256_bytes(run1_bytes),
                "run2_sha256": sha256_bytes(run2_bytes),
                "run1_bytes": len(run1_bytes),
                "run2_bytes": len(run2_bytes),
            },
        )

        report = parse_report(run1)
        receipt["inventory_summary"] = validate_report(
            report, audit_sha256, len(audit_bytes)
        )
        receipt["inventory_report"] = {
            "retained_file": run1.name,
            "sha256": sha256_bytes(run1_bytes),
            "bytes": len(run1_bytes),
        }

        product_status_post = recorder.record_command(
            "product_git_status_post",
            ["git", "status", "--porcelain", "--untracked-files=no"],
            product_worktree,
        ).stdout
        recorder.record_check(
            "product_tracked_worktree_clean_post",
            product_status_post == b"",
            {"stdout_bytes": len(product_status_post)},
        )

        receipt["status"] = "PASS_INVENTORY"
    except (QualificationFailure, subprocess.CalledProcessError, OSError) as exc:
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        receipt["gates"] = recorder.gates
        if product_worktree is not None and product_worktree.exists():
            cleanup = subprocess.run(
                ["git", "worktree", "remove", "--force", str(product_worktree)],
                cwd=ROOT,
                capture_output=True,
            )
            receipt["worktree_cleanup"] = {
                "returncode": cleanup.returncode,
                "stdout_sha256": sha256_bytes(cleanup.stdout),
                "stderr_sha256": sha256_bytes(cleanup.stderr),
            }
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)

        receipt_path = evidence_dir / "qualification_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(
        {
            "status": receipt["status"],
            "receipt": str(evidence_dir / "qualification_receipt.json"),
        },
        sort_keys=True,
    ))
    return 0 if receipt["status"] == "PASS_INVENTORY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
