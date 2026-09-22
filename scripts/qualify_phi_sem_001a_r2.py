#!/usr/bin/env python3
"""Never-merge exact qualifier for PHI-SEM-001A-R2.

PASS_INVENTORY means only that the deterministic lexical inventory executed
against the exact frozen product and satisfied this qualifier's source-identity,
Git-object, symlink-regression, witness, and repeatability gates.
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
QUALIFIER_PATH = "scripts/qualify_phi_sem_001a_r2.py"
QUALIFIER_PROFILE = "qual-phi-sem-001a-r2-v1"

PRODUCT_COMMIT = "8347d02deb21648c89d0ee585ece7192f59e93f1"
PRODUCT_TREE = "29bea0c1de52e369ea6d457a9bf41ca36f0af963"
ENGINE_PATH = "scripts/audit_phi_authority_flow.py"
ENGINE_BLOB = "5fbbd183e3cc69ba2080633a6b613b95772739e9"
WRAPPER_PATH = "scripts/audit_phi_authority_flow_r2.py"
WRAPPER_BLOB = "ffd3c835d9dd1efe1ca74bd0ff9a979a61b88640"
PRODUCT_PROFILE = "phi-sem-001a-lexical-v2"
IO_SEMANTICS = "git-index-object-bytes-v2"

REGRESSION_SYMLINK = "papers/evaluation/psych-bench/ablation_domains.csv"
REGRESSION_MODE = "120000"
REGRESSION_BLOB = "a5164501d87561eb53e6064da94ddbe262a056af"
REGRESSION_POINTER = b"../data/psych_bench/ablation_domains.csv"


class QualificationFailure(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(cwd: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(args, cwd=cwd, env=merged, capture_output=True)


def git_bytes(cwd: Path, *args: str) -> bytes:
    result = run(cwd, "git", *args)
    if result.returncode != 0:
        raise QualificationFailure(
            f"git {' '.join(args)} failed: {result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def git_text(cwd: Path, *args: str) -> str:
    return git_bytes(cwd, *args).decode("utf-8").strip()


def tracked_entry(cwd: Path, path: str) -> tuple[str, str]:
    out = git_text(cwd, "ls-files", "-s", "--", path)
    if not out:
        raise QualificationFailure(f"tracked path missing: {path}")
    lines = out.splitlines()
    if len(lines) != 1:
        raise QualificationFailure(f"expected one index entry for {path}, got {len(lines)}")
    meta, listed = lines[0].split("\t", 1)
    mode, blob, stage = meta.split()
    if listed != path or stage != "0":
        raise QualificationFailure(f"unexpected index entry for {path}: {out}")
    return mode, blob


def blob_bytes(cwd: Path, blob: str) -> bytes:
    return git_bytes(cwd, "cat-file", "blob", blob)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationFailure(message)


class Recorder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.gates: list[dict[str, Any]] = []

    def command(
        self,
        name: str,
        cwd: Path,
        *args: str,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        result = run(cwd, *args, env=env)
        stdout_name = f"{name}.stdout"
        stderr_name = f"{name}.stderr"
        (self.root / stdout_name).write_bytes(result.stdout)
        (self.root / stderr_name).write_bytes(result.stderr)
        gate = {
            "name": name,
            "kind": "command",
            "args": list(args),
            "returncode": result.returncode,
            "stdout_file": stdout_name,
            "stdout_sha256": sha256(result.stdout),
            "stdout_bytes": len(result.stdout),
            "stderr_file": stderr_name,
            "stderr_sha256": sha256(result.stderr),
            "stderr_bytes": len(result.stderr),
        }
        self.gates.append(gate)
        if result.returncode != 0:
            raise QualificationFailure(f"gate {name} exited {result.returncode}")
        return result

    def check(self, name: str, condition: bool, **details: Any) -> None:
        self.gates.append({"name": name, "kind": "check", "passed": condition, "details": details})
        require(condition, f"check failed: {name}")


def parse_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationFailure(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), "report root must be object")
    return value


def validate_report(report: dict[str, Any], product: Path, recorder: Recorder) -> dict[str, Any]:
    source = report.get("source")
    require(isinstance(source, dict), "source must be object")
    require(source.get("commit") == PRODUCT_COMMIT, "report source commit mismatch")
    require(source.get("tree") == PRODUCT_TREE, "report source tree mismatch")
    require(report.get("report_version") == 2, "report_version must be 2")
    require(report.get("profile_id") == PRODUCT_PROFILE, "profile id mismatch")

    artifact = report.get("audit_artifact")
    require(isinstance(artifact, dict), "audit_artifact must be object")
    require(artifact.get("path") == ENGINE_PATH, "engine artifact path mismatch")
    require(artifact.get("git_blob") == ENGINE_BLOB, "engine artifact blob mismatch")
    require(artifact.get("byte_source") == IO_SEMANTICS, "engine byte-source semantics mismatch")
    engine_data = blob_bytes(product, ENGINE_BLOB)
    require(artifact.get("sha256") == sha256(engine_data), "engine SHA-256 mismatch")
    require(artifact.get("bytes") == len(engine_data), "engine byte length mismatch")

    require(report.get("missing_mandatory_witness_selectors") == [], "mandatory witness selector missing")

    inventory = report.get("inventory")
    require(isinstance(inventory, list) and inventory, "inventory must be non-empty list")
    checked = 0
    for item in inventory:
        require(isinstance(item, dict), "inventory item must be object")
        path = item.get("path")
        require(isinstance(path, str), "inventory path must be string")
        mode, blob = tracked_entry(product, path)
        require(item.get("git_mode") == mode, f"mode mismatch for {path}")
        require(item.get("git_blob") == blob, f"blob mismatch for {path}")
        data = blob_bytes(product, blob)
        require(item.get("sha256") == sha256(data), f"SHA mismatch for {path}")
        require(item.get("bytes") == len(data), f"byte length mismatch for {path}")
        checked += 1

    oracle = report.get("phi_oracle_duplicate_check")
    require(isinstance(oracle, dict), "phi_oracle_duplicate_check must be object")
    for key in ("workspace_active_candidate", "root_duplicate_candidate"):
        subtree = oracle.get(key)
        require(isinstance(subtree, dict), f"{key} must be object")
        require(
            subtree.get("fingerprint_semantics") == "relative-path+git-mode+blob+sha256-v2",
            f"{key} fingerprint semantics mismatch",
        )
        members = subtree.get("members")
        require(isinstance(members, list), f"{key}.members must be list")
        for member in members:
            require(isinstance(member, dict), f"{key} member must be object")
            require(member.get("git_mode") in {"100644", "100755", "120000"}, f"bad mode in {key}")

    summary = report.get("summary")
    require(isinstance(summary, dict), "summary must be object")
    require(summary.get("matched_paths") == checked, "summary matched_paths mismatch")
    require(isinstance(report.get("profile_sha256"), str) and len(report["profile_sha256"]) == 64, "profile_sha256 invalid")

    recorder.check("matched_items_rebound_to_git_objects", checked > 0, matched_items=checked)
    return {
        "matched_paths": checked,
        "matched_lines": summary.get("matched_lines"),
        "categories": summary.get("categories"),
        "profile_sha256": report.get("profile_sha256"),
        "oracle_byte_identity_equal": oracle.get("byte_identity_equal"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "target" / "qualification" / "phi-sem-001a-r2")
    args = parser.parse_args()
    evidence = args.evidence_dir.resolve()
    recorder = Recorder(evidence)
    receipt: dict[str, Any] = {
        "qualifier_profile": QUALIFIER_PROFILE,
        "status": "FAIL",
        "product": {
            "commit": PRODUCT_COMMIT,
            "tree": PRODUCT_TREE,
            "profile": PRODUCT_PROFILE,
            "engine_path": ENGINE_PATH,
            "engine_blob": ENGINE_BLOB,
            "wrapper_path": WRAPPER_PATH,
            "wrapper_blob": WRAPPER_BLOB,
        },
        "claim_ceiling": {
            "establishes_on_pass": [
                "exact R2 product executed twice",
                "inventory reports were byte-identical",
                "matched source identities rebound to exact Git objects",
                "known R1 symlink regression is handled as a Git symlink object",
                "mandatory lexical witness selectors are present",
            ],
            "does_not_establish": [
                "runtime reachability completeness",
                "estimator correctness",
                "IIT validity",
                "consciousness",
                "calibrated confidence",
                "expression, execution, or governance authority",
            ],
        },
    }

    temp: Path | None = None
    product: Path | None = None
    try:
        head = git_text(ROOT, "rev-parse", "HEAD")
        tree = git_text(ROOT, "rev-parse", "HEAD^{tree}")
        parents = git_text(ROOT, "rev-list", "--parents", "-n", "1", "HEAD").split()
        require(len(parents) == 2 and parents[1] == PRODUCT_COMMIT, "qualifier must be direct child of exact R2 product")
        qmode, qblob = tracked_entry(ROOT, QUALIFIER_PATH)
        qdata = blob_bytes(ROOT, qblob)
        receipt["qualifier"] = {
            "commit": head,
            "tree": tree,
            "parent": parents[1],
            "path": QUALIFIER_PATH,
            "mode": qmode,
            "blob": qblob,
            "sha256": sha256(qdata),
            "bytes": len(qdata),
        }

        status = recorder.command("qualifier_git_status", ROOT, "git", "status", "--porcelain", "--untracked-files=no").stdout
        recorder.check("qualifier_clean", status == b"", stdout_bytes=len(status))

        receipt["toolchain"] = {
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "platform": platform.platform(),
        }
        recorder.command("git_version", ROOT, "git", "--version")
        recorder.command("python_version", ROOT, sys.executable, "--version")
        recorder.command("uname", ROOT, "uname", "-a")

        temp = Path(tempfile.mkdtemp(prefix="qual-phi-sem-001a-r2-"))
        product = temp / "product"
        recorder.command("worktree_add", ROOT, "git", "worktree", "add", "--detach", str(product), PRODUCT_COMMIT)
        recorder.check(
            "exact_product_subject",
            git_text(product, "rev-parse", "HEAD") == PRODUCT_COMMIT and git_text(product, "rev-parse", "HEAD^{tree}") == PRODUCT_TREE,
            observed_commit=git_text(product, "rev-parse", "HEAD"),
            observed_tree=git_text(product, "rev-parse", "HEAD^{tree}"),
        )

        for name, path, expected_blob in (
            ("engine", ENGINE_PATH, ENGINE_BLOB),
            ("wrapper", WRAPPER_PATH, WRAPPER_BLOB),
        ):
            mode, blob = tracked_entry(product, path)
            recorder.check(f"exact_{name}_artifact", blob == expected_blob, path=path, mode=mode, observed_blob=blob, expected_blob=expected_blob)

        mode, blob = tracked_entry(product, REGRESSION_SYMLINK)
        pointer = blob_bytes(product, blob)
        recorder.check(
            "r1_symlink_regression_subject",
            mode == REGRESSION_MODE and blob == REGRESSION_BLOB and pointer == REGRESSION_POINTER,
            path=REGRESSION_SYMLINK,
            observed_mode=mode,
            observed_blob=blob,
            pointer_sha256=sha256(pointer),
            pointer_bytes=len(pointer),
        )

        pre = recorder.command("product_git_status_pre", product, "git", "status", "--porcelain", "--untracked-files=no").stdout
        recorder.check("product_clean_pre", pre == b"", stdout_bytes=len(pre))

        pycache = temp / "pycache"
        env = {"PYTHONPYCACHEPREFIX": str(pycache)}
        recorder.command("py_compile_engine", product, sys.executable, "-m", "py_compile", ENGINE_PATH, env=env)
        recorder.command("py_compile_wrapper", product, sys.executable, "-m", "py_compile", WRAPPER_PATH, env=env)
        recorder.command("wrapper_help", product, sys.executable, WRAPPER_PATH, "--help", env=env)

        run1 = evidence / "phi_sem_001a_inventory_v2.run1.json"
        run2 = evidence / "phi_sem_001a_inventory_v2.run2.json"
        recorder.command("inventory_run1", product, sys.executable, WRAPPER_PATH, "--pretty", "--output", str(run1), env=env)
        recorder.command("inventory_run2", product, sys.executable, WRAPPER_PATH, "--pretty", "--output", str(run2), env=env)

        b1 = run1.read_bytes()
        b2 = run2.read_bytes()
        recorder.check(
            "inventory_byte_determinism",
            b1 == b2,
            run1_sha256=sha256(b1),
            run2_sha256=sha256(b2),
            run1_bytes=len(b1),
            run2_bytes=len(b2),
        )
        report = parse_json(run1)
        receipt["inventory_summary"] = validate_report(report, product, recorder)
        receipt["inventory_report"] = {"file": run1.name, "sha256": sha256(b1), "bytes": len(b1)}

        post = recorder.command("product_git_status_post", product, "git", "status", "--porcelain", "--untracked-files=no").stdout
        recorder.check("product_clean_post", post == b"", stdout_bytes=len(post))
        receipt["status"] = "PASS_INVENTORY"
    except (QualificationFailure, OSError, subprocess.SubprocessError) as exc:
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        receipt["gates"] = recorder.gates
        if product is not None and product.exists():
            cleanup = run(ROOT, "git", "worktree", "remove", "--force", str(product))
            receipt["worktree_cleanup"] = {
                "returncode": cleanup.returncode,
                "stdout_sha256": sha256(cleanup.stdout),
                "stderr_sha256": sha256(cleanup.stderr),
            }
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "qualification_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({"status": receipt["status"], "receipt": str(evidence / "qualification_receipt.json")}, sort_keys=True))
    return 0 if receipt["status"] == "PASS_INVENTORY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
