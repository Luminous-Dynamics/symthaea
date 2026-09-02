#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import audit_vart_world_creative_001_pilot as audit

SUBJECT_HEAD = "a" * 40
SUBJECT_TREE = "b" * 40
INSTRUMENT_HEAD = "e" * 40
INSTRUMENT_TREE = "f" * 40
ANCHOR_SHA = "1" * 64
CONFIG_SHA = "2" * 64


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closure(root: Path) -> str:
    entries: list[dict[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == audit.RECEIPT_REL:
            continue
        entries.append({"path": rel, "sha256": sha256_file(path)})
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def base_cells() -> list[dict[str, object]]:
    return [
        {"cell_id": "P1", "trial_id": "p1", "policy": "full_symthaea", "fixture": "ordinary", "seed": 910001, "revision_index": 0, "paired_block_id": "ordinary-r0"},
        {"cell_id": "P2", "trial_id": "p2", "policy": "random_valid", "fixture": "ordinary", "seed": 910001, "revision_index": 0, "paired_block_id": "ordinary-r0"},
        {"cell_id": "P3", "trial_id": "p3", "policy": "heuristic", "fixture": "ordinary", "seed": 910001, "revision_index": 0, "paired_block_id": "ordinary-r0"},
        {"cell_id": "P4", "trial_id": "p4", "policy": "full_symthaea", "fixture": "pretty_trap", "seed": 910002, "revision_index": 0, "paired_block_id": "pretty-r0"},
        {"cell_id": "P5", "trial_id": "p5", "policy": "random_valid", "fixture": "pretty_trap", "seed": 910002, "revision_index": 0, "paired_block_id": "pretty-r0"},
        {"cell_id": "P6", "trial_id": "p6", "policy": "full_symthaea", "fixture": "memory_trap", "seed": 910003, "revision_index": 0, "paired_block_id": "memory-r0"},
    ]


def design_sha(cells: list[dict[str, object]]) -> str:
    projection = audit.verify_paired_design(cells)
    return hashlib.sha256(audit.canonical_json_bytes(projection)).hexdigest()


def build_root(
    root: Path,
    cells: list[dict[str, object]],
    *,
    dual: bool = False,
    plan_subject_head: str = SUBJECT_HEAD,
    plan_instrument_head: str = INSTRUMENT_HEAD,
    receipt_instrument_head: str = INSTRUMENT_HEAD,
    plan_anchor_sha: str = ANCHOR_SHA,
    receipt_design_sha: str | None = None,
) -> None:
    plan: dict[str, object] = {
        "experiment_id": audit.EXPERIMENT_ID,
        "campaign": "pilot",
        "noncanonical": True,
        "cells": [{"cell": cell, "argv": ["runtime"]} for cell in cells],
    }
    if dual:
        plan.update({
            "subject_source_head": plan_subject_head,
            "subject_source_tree": SUBJECT_TREE,
            "instrument_source_head": plan_instrument_head,
            "instrument_source_tree": INSTRUMENT_TREE,
            "pilot_config_sha256": CONFIG_SHA,
            "pilot_design_sha256": design_sha(cells),
            "preexecution_anchor_sha256": plan_anchor_sha,
        })
    write_json(root / audit.PLAN_REL, plan)

    receipt: dict[str, object] = {
        "schema": "symthaea.vart-world-creative-001.pilot-receipt.v1",
        "experiment_id": audit.EXPERIMENT_ID,
        "campaign": "pilot",
        "noncanonical": True,
        "scientific_efficacy_claims_authorized": False,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
        "source": {
            "head": SUBJECT_HEAD,
            "tree": SUBJECT_TREE,
            "parent_v05a_head": "c" * 40,
            "parent_v05a_tree": "d" * 40,
        },
        "cells": [{"cell_id": cell["cell_id"]} for cell in cells],
        "verifier_result": {"verdict": "ACCEPT"},
    }
    if dual:
        receipt.update({
            "dual_source_instrument_external": True,
            "instrument_source": {
                "head": receipt_instrument_head,
                "tree": INSTRUMENT_TREE,
            },
            "preexecution_anchor_sha256": ANCHOR_SHA,
            "pilot_config_sha256": CONFIG_SHA,
            "pilot_design_sha256": receipt_design_sha or design_sha(cells),
        })
    receipt["pilot_evidence_closure_sha256"] = closure(root)
    write_json(root / audit.RECEIPT_REL, receipt)


def expect_reject(root: Path, code: str) -> None:
    try:
        audit.verify_pilot(root)
    except audit.AuditError as exc:
        assert exc.code == code, f"expected {code}, got {exc.code}: {exc.detail}"
        return
    raise AssertionError(f"expected rejection {code}")


# Legacy/non-dual pilot remains auditable for diagnosing the already-run pilot.
with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-pass-") as td:
    root = Path(td)
    build_root(root, base_cells())
    result = audit.verify_pilot(root)
    assert result["verdict"] == "PILOT_AUDIT_PASS"
    assert result["paired_block_semantics"] == "PASS"
    assert result["dual_source_bound"] is False

# A1 — paired seed drift.
with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-seed-") as td:
    root = Path(td)
    cells = base_cells()
    cells[1]["seed"] = 910099
    build_root(root, cells)
    expect_reject(root, "PAIRED_BLOCK_WORLD_INPUT_MISMATCH")

# A2 — paired fixture drift.
with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-fixture-") as td:
    root = Path(td)
    cells = base_cells()
    cells[4]["fixture"] = "ordinary"
    build_root(root, cells)
    expect_reject(root, "PAIRED_BLOCK_WORLD_INPUT_MISMATCH")

# A3 — paired revision drift.
with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-revision-") as td:
    root = Path(td)
    cells = base_cells()
    cells[2]["revision_index"] = 1
    build_root(root, cells)
    expect_reject(root, "PAIRED_BLOCK_WORLD_INPUT_MISMATCH")

# A4 — duplicate policy in one paired block.
with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-duplicate-policy-") as td:
    root = Path(td)
    cells = base_cells()
    cells[1]["policy"] = "full_symthaea"
    build_root(root, cells)
    expect_reject(root, "PAIRED_BLOCK_DUPLICATE_POLICY")

# A5 — post-seal plan tamper.
with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-closure-") as td:
    root = Path(td)
    build_root(root, base_cells())
    plan = json.loads((root / audit.PLAN_REL).read_text(encoding="utf-8"))
    plan["tampered_after_seal"] = True
    write_json(root / audit.PLAN_REL, plan)
    expect_reject(root, "PILOT_AUDIT_CLOSURE_MISMATCH")

# Canonical dual-source pilot passes and exposes both immutable identities.
with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-dual-pass-") as td:
    root = Path(td)
    build_root(root, base_cells(), dual=True)
    result = audit.verify_pilot(root)
    assert result["dual_source_bound"] is True
    assert result["source_head"] == SUBJECT_HEAD
    assert result["instrument_source_head"] == INSTRUMENT_HEAD
    assert result["preexecution_anchor_sha256"] == ANCHOR_SHA

# A6 — instrument source splice between plan and receipt.
with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-instrument-splice-") as td:
    root = Path(td)
    build_root(root, base_cells(), dual=True, receipt_instrument_head="9" * 40)
    expect_reject(root, "PILOT_AUDIT_SOURCE_PAIR_MISMATCH")

# A7 — subject source splice between plan and receipt.
with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-subject-splice-") as td:
    root = Path(td)
    build_root(root, base_cells(), dual=True, plan_subject_head="8" * 40)
    expect_reject(root, "PILOT_AUDIT_SOURCE_PAIR_MISMATCH")

# A8 — receipt's claimed design digest disagrees with independent reconstruction.
with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-design-splice-") as td:
    root = Path(td)
    build_root(root, base_cells(), dual=True, receipt_design_sha="7" * 64)
    expect_reject(root, "PILOT_AUDIT_ANCHOR_MISMATCH")

# A9 — pre-execution anchor identity differs between plan and receipt.
with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-anchor-splice-") as td:
    root = Path(td)
    build_root(root, base_cells(), dual=True, plan_anchor_sha="6" * 64)
    expect_reject(root, "PILOT_AUDIT_ANCHOR_MISMATCH")

print("PASS: post-pilot legacy+dual audit acceptance + A1-A9 deterministic rejection")
