#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
INSTRUMENT_PATTERNS = (
    "scripts/*vart_world_creative_001*.py",
    "docs/research/VART_WORLD_CREATIVE_001_*",
    ".github/workflows/vart-world-creative-*.yml",
)
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
    ("confirmatory_launch", "scripts/test_verify_vart_world_creative_001_confirmatory_launch.py"),
    ("subject_source_closure", "scripts/test_qualify_vart_world_creative_001_source_closure.py"),
    ("instrument_source_closure", "scripts/test_qualify_vart_world_creative_001_instrument_source_closure.py"),
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
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise QualificationError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def meaningful_status_lines(repo: Path) -> list[str]:
    lines = git(repo, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    meaningful: list[str] = []
    for line in lines:
        path = line[3:] if len(line) >= 4 else line
        if "/__pycache__/" in f"/{path}" or path.endswith(".pyc"):
            continue
        meaningful.append(line)
    return meaningful


def require_clean(repo: Path, stage: str) -> None:
    dirty = meaningful_status_lines(repo)
    if dirty:
        raise QualificationError(f"instrument checkout dirty at {stage}: {dirty}")


def require_external(path: Path, repo: Path) -> None:
    try:
        path.relative_to(repo)
    except ValueError:
        return
    raise QualificationError(f"qualification receipt must be outside instrument checkout: {path}")


def discover_instrument_files(repo: Path) -> list[str]:
    found: set[str] = set()
    for pattern in INSTRUMENT_PATTERNS:
        for path in repo.glob(pattern):
            if path.is_file():
                found.add(path.relative_to(repo).as_posix())
    files = sorted(found)
    if not files:
        raise QualificationError("instrument manifest discovery returned no files")
    for _, suite in SUITES:
        if suite not in found:
            raise QualificationError(f"registered suite is outside instrument manifest: {suite}")
    return files


def build_manifest(repo: Path) -> tuple[list[dict[str, str]], str]:
    entries = [
        {"path": rel, "sha256": sha256_file(repo / rel)}
        for rel in discover_instrument_files(repo)
    ]
    return entries, sha256_bytes(canonical_bytes(entries))


def environment_identity() -> dict[str, str]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }


def run_suite(repo: Path, name: str, rel: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, rel],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "name": name,
        "argv": [sys.executable, rel],
        "returncode": proc.returncode,
        "stdout_sha256": sha256_bytes(proc.stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(proc.stderr.encode("utf-8")),
        "stdout_tail": proc.stdout.strip().splitlines()[-1:] or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qualify the VART-WORLD-CREATIVE-001 measurement instrument"
    )
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
    env_identity = environment_identity()
    env_digest = sha256_bytes(canonical_bytes(env_identity))

    if args.manifest_only:
        print(json.dumps({
            "verdict": "INSTRUMENT_MANIFEST_VALID",
            "instrument_head": head,
            "instrument_tree": tree,
            "instrument_manifest_sha256": manifest_sha,
            "instrument_file_count": len(manifest),
            "instrument_environment_digest": env_digest,
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
        "instrument_source": {"head": head, "tree": tree, "dirty": False},
        "instrument_manifest_patterns": list(INSTRUMENT_PATTERNS),
        "instrument_manifest_sha256": manifest_sha,
        "instrument_file_count": len(manifest),
        "instrument_files": manifest,
        "instrument_environment": env_identity,
        "instrument_environment_digest": env_digest,
        "suites": suites,
        "suite_count": len(suites),
        "all_suites_pass": True,
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
        "instrument_environment_digest": env_digest,
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
