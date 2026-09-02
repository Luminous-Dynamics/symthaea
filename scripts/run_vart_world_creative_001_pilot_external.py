#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_vart_world_creative_001_pilot_anchored as anchorlib

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
V05_HEAD = "33820b3d9e904280e6264719fe7717cb2e5dd5bb"
V05_TREE = "e93c6dbfa05b602100ff924efaa5d95f92ef5a65"
EXPECTED_CELL_IDS = [f"P{i}" for i in range(1, 9)]
FORBIDDEN_AGGREGATES = ["world_quality", "creative_score", "beauty_score", "cinematic_quality", "intelligence_score"]
HEX40 = set("0123456789abcdef")
HEX64 = set("0123456789abcdef")


class PilotError(RuntimeError):
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PilotError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PilotError(f"invalid JSON at {path}: {exc}") from exc


def run(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(argv, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check and proc.returncode != 0:
        raise PilotError(f"command failed ({proc.returncode}): {argv!r}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc


def git(repo: Path, *args: str) -> str:
    return run(["git", "-C", str(repo), *args], cwd=repo).stdout.strip()


def is_hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(ch in HEX40 for ch in value)


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def require_subject_source(runtime_root: Path, source: Any) -> tuple[str, str]:
    if not isinstance(source, dict):
        raise PilotError("expected_source must be an object")
    head = source.get("head")
    tree = source.get("tree")
    if not is_hex(head, 40) or not is_hex(tree, 40):
        raise PilotError("expected subject HEAD/TREE must be 40-hex")
    if source.get("parent_v05a_head") != V05_HEAD or source.get("parent_v05a_tree") != V05_TREE:
        raise PilotError("qualified v0.5-A parent identity mismatch")
    if not runtime_root.is_dir():
        raise PilotError(f"runtime_root is not a directory: {runtime_root}")
    dirty = git(runtime_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise PilotError("subject checkout is dirty; pilot execution is fail-closed")
    actual_head = git(runtime_root, "rev-parse", "HEAD")
    actual_tree = git(runtime_root, "rev-parse", "HEAD^{tree}")
    if actual_head != head or actual_tree != tree:
        raise PilotError(f"subject identity mismatch: {actual_head}/{actual_tree} != {head}/{tree}")
    parent_tree = git(runtime_root, "rev-parse", f"{V05_HEAD}^{{tree}}")
    if parent_tree != V05_TREE:
        raise PilotError("qualified v0.5-A TREE mismatch in subject checkout")
    ancestry = run(["git", "-C", str(runtime_root), "merge-base", "--is-ancestor", V05_HEAD, actual_head], cwd=runtime_root, check=False)
    if ancestry.returncode != 0:
        raise PilotError("qualified v0.5-A parent is not an ancestor of subject HEAD")
    return actual_head, actual_tree


def validate_contract(path: Path, label: str) -> str:
    value = read_json(path)
    if not isinstance(value, dict) or value.get("experiment_id") != EXPERIMENT_ID:
        raise PilotError(f"{label} is not bound to {EXPERIMENT_ID}")
    if value.get("campaign") != "pilot" or value.get("noncanonical") is not True:
        raise PilotError(f"{label} must be pilot-only and noncanonical")
    return sha256_file(path)


def expand_argv(template: Any, values: dict[str, Any]) -> list[str]:
    if not isinstance(template, list) or not template:
        raise PilotError("runtime_argv must be a non-empty array")
    out: list[str] = []
    for raw in template:
        if not isinstance(raw, str):
            raise PilotError("runtime_argv entries must be strings")
        if raw.startswith("__REPLACE_"):
            raise PilotError("runtime_argv is unresolved")
        try:
            out.append(raw.format_map(values))
        except KeyError as exc:
            raise PilotError(f"unknown runtime_argv placeholder: {exc}") from exc
    return out


def command_values(cell: dict[str, Any], runtime_root: Path, output_root: str) -> dict[str, Any]:
    return {**cell, "runtime_root": str(runtime_root), "output_root": output_root, "experiment_id": EXPERIMENT_ID, "campaign": "pilot"}


def validate_manifest(stage_root: Path, cell: dict[str, Any], analysis_sha: str, metrics_sha: str) -> dict[str, Any]:
    path = stage_root / "trials" / cell["trial_id"] / "manifest.json"
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise PilotError(f"{cell['cell_id']}: manifest must be object")
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "campaign": "pilot",
        "trial_id": cell["trial_id"],
        "policy": cell["policy"],
        "world_fixture_sha256": manifest.get("world_fixture_sha256"),
        "seed": cell["seed"],
        "revision_index": cell["revision_index"],
        "paired_block_id": cell["paired_block_id"],
        "included_in_confirmatory_analysis": False,
        "analysis_contract_sha256": analysis_sha,
        "metric_definition_set_sha256": metrics_sha,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise PilotError(f"{cell['cell_id']}: manifest {key} mismatch: {manifest.get(key)!r} != {value!r}")
    return manifest


def require_stage_layout(stage_root: Path, trial_id: str) -> Path:
    children = sorted(p.name for p in stage_root.iterdir())
    if children != ["trials"]:
        raise PilotError(f"isolated runtime wrote outside trials/: {children}")
    trials = stage_root / "trials"
    trial_children = sorted(p.name for p in trials.iterdir())
    if trial_children != [trial_id]:
        raise PilotError(f"isolated runtime wrote unexpected trial directories: {trial_children}")
    return trials / trial_id


def tree_closure(root: Path, excluded: set[str]) -> tuple[str, list[dict[str, str]]]:
    entries: list[dict[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel not in excluded:
            entries.append({"path": rel, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_bytes(entries)), entries


def require_anchor_env(config_sha: str, design_sha: str) -> str:
    anchor_sha = os.environ.get("VART_PILOT_PREEXECUTION_ANCHOR_SHA256")
    observed_design = os.environ.get("VART_PILOT_DESIGN_SHA256")
    observed_config = os.environ.get("VART_PILOT_CONFIG_SHA256")
    if not is_hex(anchor_sha, 64):
        raise PilotError("anchored execution requires VART_PILOT_PREEXECUTION_ANCHOR_SHA256")
    if observed_design != design_sha:
        raise PilotError("preexecution pilot design digest mismatch")
    if observed_config != config_sha:
        raise PilotError("preexecution pilot config digest mismatch")
    return anchor_sha


def main() -> int:
    parser = argparse.ArgumentParser(description="Dual-source external VART noncanonical pilot runner")
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg = read_json(cfg_path)
    if not isinstance(cfg, dict) or cfg.get("schema") != "symthaea.vart-world-creative-001.pilot-run.v1":
        raise PilotError("unexpected pilot config schema")
    if cfg.get("experiment_id") != EXPERIMENT_ID or cfg.get("campaign") != "pilot" or cfg.get("noncanonical") is not True:
        raise PilotError("runner accepts only the noncanonical VART pilot")
    if cfg.get("confirmatory_execution_authorized") is not False or cfg.get("claim_authorized") is not False:
        raise PilotError("confirmatory execution and claims must remain unauthorized")

    cells = cfg.get("cells")
    projection = anchorlib.paired_design_projection(cells)
    design_sha = sha256_bytes(canonical_bytes(projection))
    config_sha = sha256_file(cfg_path)
    base = cfg_path.parent
    runtime_root = resolve_path(cfg["runtime_root"], base)
    pilot_root = resolve_path(cfg["pilot_root"], base)
    subject_head, subject_tree = require_subject_source(runtime_root, cfg.get("expected_source"))

    contracts = cfg.get("contract_inputs")
    if not isinstance(contracts, dict):
        raise PilotError("contract_inputs must be object")
    analysis_src = resolve_path(contracts["analysis_contract"], base)
    metrics_src = resolve_path(contracts["metric_definitions"], base)
    analysis_sha = validate_contract(analysis_src, "pilot analysis contract")
    metrics_sha = validate_contract(metrics_src, "pilot metric definitions")
    verifier = resolve_path(cfg.get("verifier_path", "../../scripts/verify_vart_world_creative_001_pilot.py"), base)
    if not verifier.is_file():
        raise PilotError(f"independent verifier not found: {verifier}")

    argv_template = cfg.get("runtime_argv")
    preview = [{"cell": cell, "argv": expand_argv(argv_template, command_values(cell, runtime_root, "<isolated-per-cell-staging-root>"))} for cell in cells]
    if pilot_root.exists() and any(pilot_root.iterdir()):
        raise PilotError(f"pilot_root must be fresh and empty: {pilot_root}")

    if args.dry_run:
        print(json.dumps({
            "verdict": "DRY_RUN_READY",
            "side_effect_free": True,
            "dual_source": True,
            "subject_source_head": subject_head,
            "subject_source_tree": subject_tree,
            "pilot_config_sha256": config_sha,
            "pilot_design_sha256": design_sha,
            "analysis_contract_sha256": analysis_sha,
            "metric_definition_set_sha256": metrics_sha,
            "verifier_source_sha256": sha256_file(verifier),
            "cell_count": len(cells),
            "resolved_commands": [item["argv"] for item in preview],
            "confirmatory_execution_authorized": False,
            "claim_authorized": False,
        }, sort_keys=True))
        return 0

    anchor_sha = require_anchor_env(config_sha, design_sha)
    pilot_root.mkdir(parents=True, exist_ok=True)
    orch = pilot_root / "_orchestrator"
    orch.mkdir(exist_ok=False)
    trials_dest = pilot_root / "trials"
    trials_dest.mkdir(exist_ok=False)
    shutil.copyfile(analysis_src, pilot_root / "analysis_contract.json")
    shutil.copyfile(metrics_src, pilot_root / "metric_definitions.json")
    if sha256_file(pilot_root / "analysis_contract.json") != analysis_sha or sha256_file(pilot_root / "metric_definitions.json") != metrics_sha:
        raise PilotError("contract copy digest changed")

    inventory = {
        "schema": "symthaea.vart-world-creative-001.trial-inventory.v1",
        "experiment_id": EXPERIMENT_ID,
        "campaign": "pilot",
        "noncanonical": True,
        "trial_ids": [cell["trial_id"] for cell in cells],
        "trial_count": len(cells),
        "expected_trial_count": len(cells),
        "confirmatory_eligible": False,
    }
    write_json(pilot_root / "trial_inventory.json", inventory)
    inventory_sha = sha256_file(pilot_root / "trial_inventory.json")
    write_json(pilot_root / "primary_results.json", {
        "schema": "symthaea.vart-world-creative-001.pilot-primary-results.v1",
        "experiment_id": EXPERIMENT_ID, "campaign": "pilot", "noncanonical": True,
        "scientific_claims_authorized": False, "results": {},
    })
    write_json(pilot_root / "confirmatory_freeze.json", {
        "schema": "symthaea.vart-world-creative-001.pilot-freeze.v1",
        "experiment_id": EXPERIMENT_ID, "campaign": "pilot", "noncanonical": True,
        "confirmatory_execution_authorized": False, "claim_authorized": False,
        "source": {"head": subject_head, "tree": subject_tree, "parent_v05a_head": V05_HEAD, "parent_v05a_tree": V05_TREE, "dirty": False},
        "analysis_contract_sha256": analysis_sha,
        "metric_definition_set_sha256": metrics_sha,
        "trial_inventory_sha256": inventory_sha,
        "pilot_config_sha256": config_sha,
        "pilot_design_sha256": design_sha,
        "preexecution_anchor_sha256": anchor_sha,
        "forbidden_primary_aggregates": FORBIDDEN_AGGREGATES,
        "pilot_outcomes_may_set_confirmatory_thresholds": False,
        "pilot_trials_may_enter_confirmatory_analysis": False,
        "policy_output_isolation": "per-cell-private-staging-root",
    })
    write_json(orch / "resolved_plan.json", {
        "experiment_id": EXPERIMENT_ID, "campaign": "pilot", "noncanonical": True,
        "subject_source_head": subject_head, "subject_source_tree": subject_tree,
        "pilot_config_sha256": config_sha, "pilot_design_sha256": design_sha,
        "preexecution_anchor_sha256": anchor_sha,
        "policy_output_isolation": "per-cell-private-staging-root", "cells": preview,
    })

    cell_receipts: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        with tempfile.TemporaryDirectory(prefix=f"vart-{cell['cell_id'].lower()}-") as td:
            stage = Path(td)
            stage.chmod(0o700)
            argv = expand_argv(argv_template, command_values(cell, runtime_root, str(stage)))
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
                "VART_OUTPUT_ROOT": str(stage),
                "VART_ANALYSIS_CONTRACT_SHA256": analysis_sha,
                "VART_METRIC_DEFINITION_SET_SHA256": metrics_sha,
                "VART_PILOT_PREEXECUTION_ANCHOR_SHA256": anchor_sha,
                "VART_PILOT_CONFIG_SHA256": config_sha,
                "VART_PILOT_DESIGN_SHA256": design_sha,
            })
            started = datetime.now(timezone.utc).isoformat()
            proc = run(argv, cwd=runtime_root, env=env, check=False)
            logs = orch / "logs"
            logs.mkdir(exist_ok=True)
            stdout_path = logs / f"{cell['cell_id']}.stdout.txt"
            stderr_path = logs / f"{cell['cell_id']}.stderr.txt"
            stdout_path.write_text(proc.stdout, encoding="utf-8")
            stderr_path.write_text(proc.stderr, encoding="utf-8")
            if proc.returncode != 0:
                raise PilotError(f"{cell['cell_id']}: runtime failed with {proc.returncode}")
            manifest = validate_manifest(stage, cell, analysis_sha, metrics_sha)
            staged = require_stage_layout(stage, cell["trial_id"])
            dest = trials_dest / cell["trial_id"]
            if dest.exists():
                raise PilotError(f"duplicate destination trial directory: {dest}")
            shutil.move(str(staged), str(dest))
            cell_receipts.append({
                "cell_id": cell["cell_id"], "trial_id": cell["trial_id"],
                "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "argv": preview[index]["argv"], "returncode": proc.returncode,
                "stdout_sha256": sha256_file(stdout_path), "stderr_sha256": sha256_file(stderr_path),
                "manifest_sha256": sha256_file(dest / "manifest.json"),
                "trial_state": manifest.get("trial_state"), "output_isolated_from_other_policy_trials": True,
            })

    verifier_proc = run([sys.executable, str(verifier), str(pilot_root), "--json"], cwd=runtime_root, check=False)
    (orch / "verifier.stdout.txt").write_text(verifier_proc.stdout, encoding="utf-8")
    (orch / "verifier.stderr.txt").write_text(verifier_proc.stderr, encoding="utf-8")
    if verifier_proc.returncode != 0:
        raise PilotError(f"independent verifier rejected pilot evidence: {verifier_proc.stdout} {verifier_proc.stderr}")
    try:
        verifier_result = json.loads(verifier_proc.stdout)
    except json.JSONDecodeError as exc:
        raise PilotError("verifier did not emit JSON") from exc
    if verifier_result.get("verdict") != "ACCEPT" or verifier_result.get("trial_count") != len(cells):
        raise PilotError(f"unexpected verifier result: {verifier_result!r}")

    receipt_rel = "_orchestrator/PILOT_RECEIPT.json"
    closure_sha, entries = tree_closure(pilot_root, {receipt_rel})
    receipt = {
        "schema": "symthaea.vart-world-creative-001.pilot-receipt.v1",
        "experiment_id": EXPERIMENT_ID, "campaign": "pilot", "noncanonical": True,
        "scientific_efficacy_claims_authorized": False,
        "confirmatory_execution_authorized": False, "claim_authorized": False,
        "source": {"head": subject_head, "tree": subject_tree, "parent_v05a_head": V05_HEAD, "parent_v05a_tree": V05_TREE},
        "dual_source_instrument_external": True,
        "preexecution_anchor_sha256": anchor_sha,
        "pilot_config_sha256": config_sha,
        "pilot_design_sha256": design_sha,
        "policy_output_isolation": "per-cell-private-staging-root",
        "cell_count": len(cells), "cells": cell_receipts,
        "verifier_result": verifier_result, "verifier_source_sha256": sha256_file(verifier),
        "pilot_evidence_closure_sha256": closure_sha, "closure_entry_count": len(entries),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "bounded_statement": "The noncanonical pilot establishes instrumentation/evidence plumbing only.",
    }
    write_json(pilot_root / receipt_rel, receipt)
    print(json.dumps({"verdict": "PILOT_PLUMBING_PASS", **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PilotError, anchorlib.AnchorError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"verdict": "PILOT_PLUMBING_REJECT", "error": str(exc), "confirmatory_execution_authorized": False, "claim_authorized": False}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
