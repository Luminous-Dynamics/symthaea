#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"

INSTRUMENT_FILES = [
    ".github/workflows/vart-world-creative-context.yml",
    ".github/workflows/vart-world-creative-post-pilot.yml",
    ".github/workflows/vart-world-creative-verifier.yml",
    "docs/research/VART_WORLD_CREATIVE_001_ANALYSIS_CONTRACT.template.json",
    "docs/research/VART_WORLD_CREATIVE_001_CALIBRATION_RECONSTRUCTION_V1.md",
    "docs/research/VART_WORLD_CREATIVE_001_CONFIRMATORY_FREEZE.template.json",
    "docs/research/VART_WORLD_CREATIVE_001_DUAL_SOURCE_IDENTITY_V1.md",
    "docs/research/VART_WORLD_CREATIVE_001_EVIDENCE_PACKAGE_V1.md",
    "docs/research/VART_WORLD_CREATIVE_001_EXECUTION_CONTEXT_V1.md",
    "docs/research/VART_WORLD_CREATIVE_001_EXECUTION_SEQUENCE.md",
    "docs/research/VART_WORLD_CREATIVE_001_INSTRUMENT_QUALIFICATION_V1.md",
    "docs/research/VART_WORLD_CREATIVE_001_PILOT_ANALYSIS.json",
    "docs/research/VART_WORLD_CREATIVE_001_PILOT_MATRIX.md",
    "docs/research/VART_WORLD_CREATIVE_001_PILOT_METRICS.json",
    "docs/research/VART_WORLD_CREATIVE_001_PILOT_RUN.template.json",
    "docs/research/VART_WORLD_CREATIVE_001_PLAN.template.json",
    "docs/research/VART_WORLD_CREATIVE_001_POST_PILOT_DISPOSITION.template.json",
    "docs/research/VART_WORLD_CREATIVE_001_POST_PILOT_DISPOSITION_V1.md",
    "docs/research/VART_WORLD_CREATIVE_001_RANDOM_VALID_TEST_VECTORS.json",
    "docs/research/VART_WORLD_CREATIVE_001_RANDOM_VALID_V1.md",
    "docs/research/VART_WORLD_CREATIVE_001_RUNTIME_ADAPTER_V1.md",
    "docs/research/VART_WORLD_CREATIVE_001_SOURCE_CLOSURE.template.json",
    "docs/research/VART_WORLD_CREATIVE_001_SOURCE_CLOSURE_V1.md",
    "docs/research/VART_WORLD_CREATIVE_001_STATE_EQUIVALENCE_V1.md",
    "docs/research/VART_WORLD_CREATIVE_001_STOP_GO.md",
    "docs/research/VART_WORLD_CREATIVE_001_TRIAL_MANIFEST.schema.json",
    "docs/research/VART_WORLD_CREATIVE_001_VERIFIER_NEGATIVE_SUITE.md",
    "docs/research/VART_WORLD_CREATIVE_001_WORLD_IDENTITY_V1.md",
    "scripts/audit_vart_world_creative_001_pilot.py",
    "scripts/qualify_vart_world_creative_001_instrument.py",
    "scripts/run_vart_world_creative_001_pilot.py",
    "scripts/run_vart_world_creative_001_pilot_anchored.py",
    "scripts/test_audit_vart_world_creative_001_pilot.py",
    "scripts/test_run_vart_world_creative_001_pilot_anchored.py",
    "scripts/test_verify_vart_world_creative_001.py",
    "scripts/test_verify_vart_world_creative_001_calibration.py",
    "scripts/test_verify_vart_world_creative_001_context.py",
    "scripts/test_verify_vart_world_creative_001_freeze_eligibility.py",
    "scripts/test_verify_vart_world_creative_001_identity.py",
    "scripts/test_verify_vart_world_creative_001_n1_n20.py",
    "scripts/test_verify_vart_world_creative_001_post_pilot.py",
    "scripts/test_verify_vart_world_creative_001_state.py",
    "scripts/verify_vart_world_creative_001.py",
    "scripts/verify_vart_world_creative_001_calibration.py",
    "scripts/verify_vart_world_creative_001_context.py",
    "scripts/verify_vart_world_creative_001_freeze_eligibility.py",
    "scripts/verify_vart_world_creative_001_identity.py",
    "scripts/verify_vart_world_creative_001_pilot.py",
    "scripts/verify_vart_world_creative_001_post_pilot.py",
    "scripts/verify_vart_world_creative_001_qualified.py",
    "scripts/verify_vart_world_creative_001_state.py",
]

SUITES = [
    ("core", "scripts/test_verify_vart_world_creative_001.py"),
    ("n1_n20", "scripts/test_verify_vart_world_creative_001_n1_n20.py"),
    ("execution_context", "scripts/test_verify_vart_world_creative_001_context.py"),
    ("state_equivalence", "scripts/test_verify_vart_world_creative_001_state.py"),
    ("world_identity", "scripts/test_verify_vart_world_creative_001_identity.py"),
    ("calibration", "scripts/test_verify_vart_world_creative_001_calibration.py"),
    ("pilot_anchor", "scripts/test_run_vart_world_creative_001_pilot_anchored.py"),
    ("pilot_audit", "scripts/test_audit_vart_world_creative_001_pilot.py"),
    ("post_pilot", "scripts/test_verify_vart_world_creative_001_post_pilot.py"),
    ("freeze_eligibility", "scripts/test_verify_vart_world_creative_001_freeze_eligibility.py"),
]


class QualificationError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise QualificationError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def require_clean(repo: Path, stage: str) -> None:
    dirty = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise QualificationError(f"instrument checkout dirty at {stage}: {dirty}")


def require_external(path: Path, repo: Path) -> None:
    try:
        path.relative_to(repo)
    except ValueError:
        return
    raise QualificationError(f"qualification receipt must be outside instrument checkout: {path}")


def build_manifest(repo: Path) -> tuple[list[dict[str, str]], str]:
    entries: list[dict[str, str]] = []
    for rel in INSTRUMENT_FILES:
        path = repo / rel
        if not path.is_file():
            raise QualificationError(f"missing instrument file: {rel}")
        entries.append({"path": rel, "sha256": sha256_file(path)})
    return entries, sha256_bytes(canonical_bytes(entries))


def run_suite(repo: Path, name: str, rel: str) -> dict[str, Any]:
    proc = subprocess.run([sys.executable, rel], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "name": name,
        "argv": [sys.executable, rel],
        "returncode": proc.returncode,
        "stdout_sha256": sha256_bytes(proc.stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(proc.stderr.encode("utf-8")),
        "stdout_tail": proc.stdout.strip().splitlines()[-1:] or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualify the VART-WORLD-CREATIVE-001 measurement instrument")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    out = args.out.resolve()
    require_external(out, repo)
    require_clean(repo, "qualification start")
    head = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    manifest, manifest_sha = build_manifest(repo)

    if args.manifest_only:
        print(json.dumps({
            "verdict": "INSTRUMENT_MANIFEST_VALID",
            "instrument_head": head,
            "instrument_tree": tree,
            "instrument_manifest_sha256": manifest_sha,
            "instrument_file_count": len(manifest),
            "confirmatory_execution_authorized": False,
            "claim_authorized": False,
        }, sort_keys=True))
        return 0

    suites = [run_suite(repo, name, rel) for name, rel in SUITES]
    failed = [suite["name"] for suite in suites if suite["returncode"] != 0]
    if failed:
        raise QualificationError(f"instrument suites failed: {failed}")
    require_clean(repo, "qualification end")

    receipt = {
        "schema": "symthaea.vart-world-creative-001.instrument-qualification.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "qualified",
        "instrument_source": {
            "head": head,
            "tree": tree,
            "dirty": False,
        },
        "instrument_manifest_sha256": manifest_sha,
        "instrument_file_count": len(manifest),
        "instrument_files": manifest,
        "suites": suites,
        "suite_count": len(suites),
        "all_suites_pass": True,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "qualified_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }
    if out.exists():
        raise QualificationError(f"refusing to overwrite qualification receipt: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(canonical_bytes(receipt) + b"\n")
    print(json.dumps({
        "verdict": "VART_INSTRUMENT_QUALIFIED",
        "instrument_head": head,
        "instrument_tree": tree,
        "instrument_manifest_sha256": manifest_sha,
        "qualification_receipt_sha256": sha256_file(out),
        "suite_count": len(suites),
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (QualificationError, OSError, ValueError) as exc:
        print(json.dumps({
            "verdict": "VART_INSTRUMENT_QUALIFICATION_REJECT",
            "error": str(exc),
            "confirmatory_execution_authorized": False,
            "claim_authorized": False,
        }, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
