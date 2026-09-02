#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
EXPECTED_CELL_IDS = [f"P{i}" for i in range(1, 9)]


class AnchorError(RuntimeError):
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


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AnchorError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AnchorError(f"invalid JSON at {path}: {exc}") from exc


def write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise AnchorError(f"refusing to overwrite existing anchor/receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def canonical_design(cells: Any) -> dict[str, Any]:
    if not isinstance(cells, list) or len(cells) != 8:
        raise AnchorError("pilot must contain exactly eight cells")
    ids = [c.get("cell_id") if isinstance(c, dict) else None for c in cells]
    if ids != EXPECTED_CELL_IDS:
        raise AnchorError(f"pilot cell order must be {EXPECTED_CELL_IDS}")

    seen_trials: set[str] = set()
    blocks: dict[str, list[dict[str, Any]]] = {}
    normalized: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise AnchorError("each cell must be an object")
        required = ["cell_id", "trial_id", "policy", "fixture", "seed", "revision_index", "paired_block_id"]
        missing = [name for name in required if name not in cell]
        if missing:
            raise AnchorError(f"{cell.get('cell_id', '?')}: missing {missing}")
        if cell["trial_id"] in seen_trials:
            raise AnchorError(f"duplicate trial_id: {cell['trial_id']}")
        seen_trials.add(cell["trial_id"])
        if not isinstance(cell["seed"], int) or isinstance(cell["seed"], bool) or not (0 <= cell["seed"] <= (1 << 64) - 1):
            raise AnchorError(f"{cell['cell_id']}: seed must be unsigned 64-bit")
        if not isinstance(cell["revision_index"], int) or isinstance(cell["revision_index"], bool) or cell["revision_index"] < 0:
            raise AnchorError(f"{cell['cell_id']}: revision_index must be nonnegative integer")
        item = {name: cell[name] for name in required}
        normalized.append(item)
        blocks.setdefault(str(cell["paired_block_id"]), []).append(item)

    block_summaries: list[dict[str, Any]] = []
    for block_id in sorted(blocks):
        members = blocks[block_id]
        world_inputs = {(m["fixture"], m["seed"], m["revision_index"]) for m in members}
        if len(world_inputs) != 1:
            raise AnchorError(
                f"PAIRED_BLOCK_WORLD_INPUT_MISMATCH {block_id}: "
                f"{sorted(world_inputs, key=repr)!r}"
            )
        policies = [str(m["policy"]) for m in members]
        if len(policies) != len(set(policies)):
            raise AnchorError(f"PAIRED_BLOCK_DUPLICATE_POLICY {block_id}: {policies!r}")
        fixture, seed, revision_index = next(iter(world_inputs))
        block_summaries.append({
            "paired_block_id": block_id,
            "fixture": fixture,
            "seed": seed,
            "revision_index": revision_index,
            "policies": sorted(policies),
            "trial_ids": sorted(str(m["trial_id"]) for m in members),
        })

    return {
        "schema": "symthaea.vart-world-creative-001.pilot-design.v1",
        "experiment_id": EXPERIMENT_ID,
        "campaign": "pilot",
        "noncanonical": True,
        "cells": normalized,
        "paired_blocks": block_summaries,
    }


def design_sha256(cells: Any) -> str:
    return sha256_bytes(canonical_bytes(canonical_design(cells)))


def run_json(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    proc = subprocess.run(argv, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise AnchorError(
            f"command failed ({proc.returncode}): {argv!r}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    try:
        value = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AnchorError(f"command did not emit JSON: {argv!r}: {proc.stdout!r}") from exc
    if not isinstance(value, dict):
        raise AnchorError(f"command JSON must be an object: {argv!r}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Externally anchored noncanonical VART pilot launcher")
    parser.add_argument("config", type=Path)
    parser.add_argument("--anchor-out", type=Path, required=False)
    parser.add_argument("--attestation-out", type=Path, required=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-design-only", action="store_true")
    parser.add_argument("--runner", type=Path, default=Path(__file__).with_name("run_vart_world_creative_001_pilot.py"))
    parser.add_argument("--auditor", type=Path, default=Path(__file__).with_name("audit_vart_world_creative_001_pilot.py"))
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_json(config_path)
    if not isinstance(cfg, dict):
        raise AnchorError("config must be an object")
    if cfg.get("experiment_id") != EXPERIMENT_ID or cfg.get("campaign") != "pilot" or cfg.get("noncanonical") is not True:
        raise AnchorError("launcher accepts only the noncanonical VART pilot")
    if cfg.get("confirmatory_execution_authorized") is not False or cfg.get("claim_authorized") is not False:
        raise AnchorError("confirmatory execution and claims must remain unauthorized")

    design = canonical_design(cfg.get("cells"))
    design_sha = sha256_bytes(canonical_bytes(design))
    config_sha = sha256_file(config_path)
    source = cfg.get("expected_source")
    if not isinstance(source, dict):
        raise AnchorError("expected_source must be an object")

    preview = {
        "verdict": "PILOT_DESIGN_VALID",
        "experiment_id": EXPERIMENT_ID,
        "campaign": "pilot",
        "noncanonical": True,
        "pilot_config_sha256": config_sha,
        "pilot_design_sha256": design_sha,
        "paired_block_count": len(design["paired_blocks"]),
        "source_head": source.get("head"),
        "source_tree": source.get("tree"),
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }
    if args.validate_design_only:
        print(json.dumps(preview, sort_keys=True))
        return 0

    runner = args.runner.resolve()
    auditor = args.auditor.resolve()
    if not runner.is_file():
        raise AnchorError(f"runner not found: {runner}")
    if not auditor.is_file():
        raise AnchorError(f"auditor not found: {auditor}")

    if args.dry_run:
        result = run_json([sys.executable, str(runner), str(config_path), "--dry-run"], cwd=runner.parent)
        if result.get("verdict") != "DRY_RUN_READY":
            raise AnchorError(f"underlying runner did not return DRY_RUN_READY: {result!r}")
        print(json.dumps({**preview, "verdict": "DRY_RUN_READY", "runner": result}, sort_keys=True))
        return 0

    if args.anchor_out is None or args.attestation_out is None:
        raise AnchorError("execution requires both --anchor-out and --attestation-out outside the pilot evidence root")
    anchor_path = args.anchor_out.resolve()
    attestation_path = args.attestation_out.resolve()
    pilot_root = Path(str(cfg.get("pilot_root", ""))).expanduser()
    if not pilot_root.is_absolute():
        pilot_root = (config_path.parent / pilot_root).resolve()
    else:
        pilot_root = pilot_root.resolve()
    for external in (anchor_path, attestation_path):
        try:
            external.relative_to(pilot_root)
        except ValueError:
            pass
        else:
            raise AnchorError(f"anchor/attestation must be outside pilot evidence root: {external}")

    anchor = {
        "schema": "symthaea.vart-world-creative-001.pilot-preexecution-anchor.v1",
        "experiment_id": EXPERIMENT_ID,
        "campaign": "pilot",
        "noncanonical": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_config_sha256": config_sha,
        "pilot_design_sha256": design_sha,
        "source_head": source.get("head"),
        "source_tree": source.get("tree"),
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }
    write_new_json(anchor_path, anchor)
    anchor_sha = sha256_file(anchor_path)

    env = os.environ.copy()
    env.update({
        "VART_PILOT_PREEXECUTION_ANCHOR_SHA256": anchor_sha,
        "VART_PILOT_DESIGN_SHA256": design_sha,
        "VART_PILOT_CONFIG_SHA256": config_sha,
    })
    run_result = run_json([sys.executable, str(runner), str(config_path)], cwd=runner.parent, env=env)
    if run_result.get("verdict") != "PILOT_PLUMBING_PASS":
        raise AnchorError(f"underlying pilot did not pass plumbing: {run_result!r}")

    audit_result = run_json([sys.executable, str(auditor), str(pilot_root), "--json"], cwd=auditor.parent)
    if audit_result.get("verdict") != "PILOT_AUDIT_PASS":
        raise AnchorError(f"sealed pilot audit did not pass: {audit_result!r}")
    if audit_result.get("pilot_design_sha256") != design_sha:
        raise AnchorError(
            "PILOT_DESIGN_ANCHOR_MISMATCH: "
            f"sealed={audit_result.get('pilot_design_sha256')} anchored={design_sha}"
        )

    attestation = {
        "schema": "symthaea.vart-world-creative-001.pilot-anchor-attestation.v1",
        "experiment_id": EXPERIMENT_ID,
        "campaign": "pilot",
        "noncanonical": True,
        "preexecution_anchor_sha256": anchor_sha,
        "pilot_config_sha256": config_sha,
        "pilot_design_sha256": design_sha,
        "pilot_receipt_sha256": audit_result.get("pilot_receipt_sha256"),
        "pilot_evidence_closure_sha256": audit_result.get("pilot_evidence_closure_sha256"),
        "audit_verdict": "PILOT_AUDIT_PASS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }
    write_new_json(attestation_path, attestation)
    print(json.dumps({"verdict": "PILOT_ANCHORED_AUDIT_PASS", **attestation}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AnchorError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({
            "verdict": "PILOT_ANCHORED_REJECT",
            "error": str(exc),
            "confirmatory_execution_authorized": False,
            "claim_authorized": False,
        }, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
