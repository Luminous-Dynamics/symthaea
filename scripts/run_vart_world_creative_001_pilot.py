#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
EXPECTED_V05A_HEAD = "33820b3d9e904280e6264719fe7717cb2e5dd5bb"
EXPECTED_V05A_TREE = "e93c6dbfa05b602100ff924efaa5d95f92ef5a65"
EXPECTED_VART_HEAD = "844d10609a9f03e26a06f22778db4b8cdfb6a3ef"
EXPECTED_VART_TREE = "38e5506c8f7f88d58e1ff03a77585091d9263a98"
FORBIDDEN_AGGREGATES = [
    "world_quality",
    "creative_score",
    "beauty_score",
    "cinematic_quality",
    "intelligence_score",
]
EXPECTED_CELL_IDS = [f"P{i}" for i in range(1, 9)]


class PilotError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PilotError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PilotError(f"invalid JSON at {path}: {exc}") from exc


def run_checked(argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise PilotError(
            f"command failed ({proc.returncode}): {argv!r}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def git_text(runtime_root: Path, *args: str) -> str:
    return run_checked(["git", "-C", str(runtime_root), *args], cwd=runtime_root).stdout.strip()


def require_clean_exact_source(runtime_root: Path, expected_head: str, expected_tree: str) -> tuple[str, str]:
    if not runtime_root.is_dir():
        raise PilotError(f"runtime_root is not a directory: {runtime_root}")
    actual_head = git_text(runtime_root, "rev-parse", "HEAD")
    actual_tree = git_text(runtime_root, "rev-parse", "HEAD^{tree}")
    dirty = git_text(runtime_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise PilotError("runtime source is dirty; pilot execution is fail-closed")
    if actual_head != expected_head:
        raise PilotError(f"runtime HEAD mismatch: {actual_head} != {expected_head}")
    if actual_tree != expected_tree:
        raise PilotError(f"runtime TREE mismatch: {actual_tree} != {expected_tree}")
    return actual_head, actual_tree


def resolve_path(value: str, *, base: Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def expand_argv(argv: list[str], values: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for raw in argv:
        if not isinstance(raw, str):
            raise PilotError("runtime_argv entries must all be strings")
        if raw.startswith("__REPLACE_"):
            raise PilotError("runtime_argv is unresolved; bind it to the local VART trial entrypoint")
        try:
            out.append(raw.format_map(values))
        except KeyError as exc:
            raise PilotError(f"unknown runtime_argv placeholder: {exc}") from exc
    if not out:
        raise PilotError("runtime_argv must not be empty")
    return out


def validate_contract_input(path: Path, label: str) -> str:
    if not path.is_file():
        raise PilotError(f"missing {label}: {path}")
    value = read_json(path)
    if not isinstance(value, dict) or value.get("experiment_id") != EXPERIMENT_ID:
        raise PilotError(f"{label} is not bound to {EXPERIMENT_ID}: {path}")
    if value.get("campaign") != "pilot" or value.get("noncanonical") is not True:
        raise PilotError(f"{label} must be pilot-only and noncanonical: {path}")
    return sha256_file(path)


def copy_contract(src: Path, dst: Path) -> str:
    shutil.copyfile(src, dst)
    return sha256_file(dst)


def validate_cells(cells: Any) -> list[dict[str, Any]]:
    if not isinstance(cells, list) or len(cells) != 8:
        raise PilotError("pilot must contain exactly eight cells")
    ids = [c.get("cell_id") if isinstance(c, dict) else None for c in cells]
    if ids != EXPECTED_CELL_IDS:
        raise PilotError(f"pilot cell order must be {EXPECTED_CELL_IDS}")
    seen_trials: set[str] = set()
    for cell in cells:
        for field in ["cell_id", "trial_id", "policy", "fixture", "seed", "revision_index", "paired_block_id"]:
            if field not in cell:
                raise PilotError(f"{cell.get('cell_id', '?')}: missing {field}")
        if cell["trial_id"] in seen_trials:
            raise PilotError(f"duplicate trial_id: {cell['trial_id']}")
        seen_trials.add(cell["trial_id"])
        if not isinstance(cell["seed"], int) or not (0 <= cell["seed"] <= (1 << 64) - 1):
            raise PilotError(f"{cell['cell_id']}: seed must be unsigned 64-bit")
        if not isinstance(cell["revision_index"], int) or cell["revision_index"] < 0:
            raise PilotError(f"{cell['cell_id']}: revision_index must be nonnegative integer")
    return cells


def expected_inventory(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "symthaea.vart-world-creative-001.trial-inventory.v1",
        "experiment_id": EXPERIMENT_ID,
        "campaign": "pilot",
        "noncanonical": True,
        "trial_ids": [c["trial_id"] for c in cells],
        "trial_count": len(cells),
        "confirmatory_eligible": False,
    }


def validate_emitted_manifest(pilot_root: Path, cell: dict[str, Any]) -> dict[str, Any]:
    manifest_path = pilot_root / "trials" / cell["trial_id"] / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise PilotError(f"{cell['cell_id']}: manifest must be an object")
    checks = {
        "experiment_id": EXPERIMENT_ID,
        "campaign": "pilot",
        "trial_id": cell["trial_id"],
        "policy": cell["policy"],
        "seed": cell["seed"],
        "revision_index": cell["revision_index"],
        "paired_block_id": cell["paired_block_id"],
        "included_in_confirmatory_analysis": False,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise PilotError(
                f"{cell['cell_id']}: manifest {key} mismatch: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    return manifest


def tree_closure(root: Path, *, excluded_relpaths: set[str]) -> tuple[str, list[dict[str, str]]]:
    entries: list[dict[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in excluded_relpaths:
            continue
        entries.append({"path": rel, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_json_bytes(entries)), entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed noncanonical VART-WORLD-CREATIVE-001 pilot orchestrator"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate source/contracts/verifier/commands without creating the pilot root",
    )
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg = read_json(cfg_path)
    if not isinstance(cfg, dict):
        raise PilotError("config must be a JSON object")
    if cfg.get("schema") != "symthaea.vart-world-creative-001.pilot-run.v1":
        raise PilotError("unexpected pilot config schema")
    if cfg.get("experiment_id") != EXPERIMENT_ID:
        raise PilotError("unexpected experiment_id")
    if cfg.get("campaign") != "pilot" or cfg.get("noncanonical") is not True:
        raise PilotError("orchestrator only permits the noncanonical pilot")
    if cfg.get("confirmatory_execution_authorized") is not False:
        raise PilotError("confirmatory execution must remain false")
    if cfg.get("claim_authorized") is not False:
        raise PilotError("claim authorization must remain false")

    base = cfg_path.parent
    runtime_root = resolve_path(cfg["runtime_root"], base=base)
    pilot_root = resolve_path(cfg["pilot_root"], base=base)
    cells = validate_cells(cfg.get("cells"))

    source = cfg.get("expected_source", {})
    expected_head = source.get("head")
    expected_tree = source.get("tree")
    if expected_head != EXPECTED_VART_HEAD or expected_tree != EXPECTED_VART_TREE:
        raise PilotError(
            "pilot config must bind the qualified VART runtime "
            f"{EXPECTED_VART_HEAD}/{EXPECTED_VART_TREE}"
        )
    actual_head, actual_tree = require_clean_exact_source(runtime_root, expected_head, expected_tree)

    contract_inputs = cfg.get("contract_inputs", {})
    analysis_src = resolve_path(contract_inputs["analysis_contract"], base=base)
    metrics_src = resolve_path(contract_inputs["metric_definitions"], base=base)
    analysis_source_sha = validate_contract_input(analysis_src, "pilot analysis contract")
    metric_source_sha = validate_contract_input(metrics_src, "pilot metric definitions")

    verifier_path = resolve_path(
        cfg.get("verifier_path", "../../scripts/verify_vart_world_creative_001.py"),
        base=base,
    )
    if not verifier_path.is_file():
        raise PilotError(f"independent verifier not found: {verifier_path}")

    runtime_argv_template = cfg.get("runtime_argv")
    if not isinstance(runtime_argv_template, list):
        raise PilotError("runtime_argv must be a JSON array")

    resolved_plan: list[dict[str, Any]] = []
    for cell in cells:
        values = {
            **cell,
            "runtime_root": str(runtime_root),
            "pilot_root": str(pilot_root),
            "experiment_id": EXPERIMENT_ID,
            "campaign": "pilot",
        }
        resolved_plan.append({"cell": cell, "argv": expand_argv(runtime_argv_template, values)})

    if pilot_root.exists() and any(pilot_root.iterdir()):
        raise PilotError(f"pilot_root must be fresh and empty: {pilot_root}")

    if args.dry_run:
        print(json.dumps({
            "verdict": "DRY_RUN_READY",
            "side_effect_free": True,
            "pilot_root": str(pilot_root),
            "source_head": actual_head,
            "source_tree": actual_tree,
            "analysis_contract_sha256": analysis_source_sha,
            "metric_definition_set_sha256": metric_source_sha,
            "verifier_source_sha256": sha256_file(verifier_path),
            "cell_count": len(cells),
            "resolved_commands": [item["argv"] for item in resolved_plan],
            "confirmatory_execution_authorized": False,
            "claim_authorized": False,
        }, sort_keys=True))
        return 0

    pilot_root.mkdir(parents=True, exist_ok=True)
    orch = pilot_root / "_orchestrator"
    orch.mkdir(exist_ok=False)

    analysis_sha = copy_contract(analysis_src, pilot_root / "analysis_contract.json")
    metric_sha = copy_contract(metrics_src, pilot_root / "metric_definitions.json")
    if analysis_sha != analysis_source_sha or metric_sha != metric_source_sha:
        raise PilotError("contract copy digest changed unexpectedly")

    inventory = expected_inventory(cells)
    write_json(pilot_root / "trial_inventory.json", inventory)
    inventory_sha = sha256_file(pilot_root / "trial_inventory.json")

    write_json(
        pilot_root / "primary_results.json",
        {
            "schema": "symthaea.vart-world-creative-001.pilot-primary-results.v1",
            "experiment_id": EXPERIMENT_ID,
            "campaign": "pilot",
            "noncanonical": True,
            "scientific_claims_authorized": False,
            "results": {},
        },
    )

    write_json(
        pilot_root / "confirmatory_freeze.json",
        {
            "schema": "symthaea.vart-world-creative-001.pilot-freeze.v1",
            "experiment_id": EXPERIMENT_ID,
            "campaign": "pilot",
            "noncanonical": True,
            "confirmatory_execution_authorized": False,
            "claim_authorized": False,
            "source": {
                "head": actual_head,
                "tree": actual_tree,
                "parent_v05a_head": EXPECTED_V05A_HEAD,
                "parent_v05a_tree": EXPECTED_V05A_TREE,
                "dirty": False,
            },
            "analysis_contract_sha256": analysis_sha,
            "metric_definition_set_sha256": metric_sha,
            "trial_inventory_sha256": inventory_sha,
            "forbidden_primary_aggregates": FORBIDDEN_AGGREGATES,
            "pilot_outcomes_may_set_confirmatory_thresholds": False,
            "pilot_trials_may_enter_confirmatory_analysis": False,
        },
    )

    write_json(
        orch / "resolved_plan.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "campaign": "pilot",
            "noncanonical": True,
            "source_head": actual_head,
            "source_tree": actual_tree,
            "cells": resolved_plan,
        },
    )

    cell_receipts: list[dict[str, Any]] = []
    for item in resolved_plan:
        cell = item["cell"]
        argv = item["argv"]
        env = os.environ.copy()
        env.update({
            "VART_EXPERIMENT_ID": EXPERIMENT_ID,
            "VART_CAMPAIGN": "pilot",
            "VART_NONCANONICAL": "1",
            "VART_CONFIRMATORY_EXECUTION_AUTHORIZED": "0",
            "VART_CLAIM_AUTHORIZED": "0",
            "VART_CELL_ID": str(cell["cell_id"]),
            "VART_TRIAL_ID": str(cell["trial_id"]),
            "VART_POLICY": str(cell["policy"]),
            "VART_FIXTURE": str(cell["fixture"]),
            "VART_SEED": str(cell["seed"]),
            "VART_REVISION_INDEX": str(cell["revision_index"]),
            "VART_PAIRED_BLOCK_ID": str(cell["paired_block_id"]),
            "VART_OUTPUT_ROOT": str(pilot_root),
            "VART_ANALYSIS_CONTRACT_SHA256": analysis_sha,
            "VART_METRIC_DEFINITION_SET_SHA256": metric_sha,
        })
        started = datetime.now(timezone.utc).isoformat()
        proc = subprocess.run(
            argv,
            cwd=runtime_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        log_dir = orch / "logs"
        log_dir.mkdir(exist_ok=True)
        stdout_path = log_dir / f"{cell['cell_id']}.stdout.txt"
        stderr_path = log_dir / f"{cell['cell_id']}.stderr.txt"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise PilotError(
                f"{cell['cell_id']}: runtime failed with {proc.returncode}; "
                f"see {stdout_path} and {stderr_path}"
            )
        manifest = validate_emitted_manifest(pilot_root, cell)
        if manifest.get("analysis_contract_sha256") != analysis_sha:
            raise PilotError(f"{cell['cell_id']}: analysis contract digest mismatch")
        if manifest.get("metric_definition_set_sha256") != metric_sha:
            raise PilotError(f"{cell['cell_id']}: metric definition digest mismatch")
        cell_receipts.append({
            "cell_id": cell["cell_id"],
            "trial_id": cell["trial_id"],
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "argv": argv,
            "returncode": proc.returncode,
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
            "manifest_sha256": sha256_file(
                pilot_root / "trials" / cell["trial_id"] / "manifest.json"
            ),
            "trial_state": manifest.get("trial_state"),
        })

    verifier_proc = subprocess.run(
        [sys.executable, str(verifier_path), str(pilot_root), "--json"],
        cwd=runtime_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (orch / "verifier.stdout.txt").write_text(verifier_proc.stdout, encoding="utf-8")
    (orch / "verifier.stderr.txt").write_text(verifier_proc.stderr, encoding="utf-8")
    if verifier_proc.returncode != 0:
        raise PilotError(
            "independent verifier rejected pilot evidence: "
            f"{verifier_proc.stdout.strip()} {verifier_proc.stderr.strip()}"
        )
    try:
        verifier_result = json.loads(verifier_proc.stdout)
    except json.JSONDecodeError as exc:
        raise PilotError("verifier did not emit valid --json output") from exc
    if verifier_result.get("verdict") != "ACCEPT":
        raise PilotError(f"unexpected verifier verdict: {verifier_result!r}")
    if verifier_result.get("trial_count") != len(cells):
        raise PilotError("verifier trial_count does not equal frozen pilot inventory")

    receipt_rel = "_orchestrator/PILOT_RECEIPT.json"
    closure_sha, closure_entries = tree_closure(pilot_root, excluded_relpaths={receipt_rel})
    receipt = {
        "schema": "symthaea.vart-world-creative-001.pilot-receipt.v1",
        "experiment_id": EXPERIMENT_ID,
        "campaign": "pilot",
        "noncanonical": True,
        "scientific_efficacy_claims_authorized": False,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
        "source": {
            "head": actual_head,
            "tree": actual_tree,
            "parent_v05a_head": EXPECTED_V05A_HEAD,
            "parent_v05a_tree": EXPECTED_V05A_TREE,
        },
        "cell_count": len(cells),
        "cells": cell_receipts,
        "verifier_result": verifier_result,
        "verifier_source_sha256": sha256_file(verifier_path),
        "pilot_evidence_closure_sha256": closure_sha,
        "closure_entry_count": len(closure_entries),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "bounded_statement": (
            "The noncanonical pilot may establish only instrumentation, serialization, "
            "pairing, evidence-closure, and verifier plumbing."
        ),
    }
    write_json(pilot_root / receipt_rel, receipt)
    print(json.dumps({"verdict": "PILOT_PLUMBING_PASS", **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PilotError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({
            "verdict": "PILOT_PLUMBING_REJECT",
            "error": str(exc),
            "confirmatory_execution_authorized": False,
            "claim_authorized": False,
        }, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
