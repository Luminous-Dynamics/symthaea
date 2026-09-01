#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import audit_vart_world_creative_001_pilot as audit


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


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


def build_root(root: Path, cells: list[dict[str, object]]) -> None:
    write_json(
        root / audit.PLAN_REL,
        {
            "experiment_id": audit.EXPERIMENT_ID,
            "campaign": "pilot",
            "noncanonical": True,
            "cells": [{"cell": cell, "argv": ["runtime"]} for cell in cells],
        },
    )
    receipt = {
        "schema": "symthaea.vart-world-creative-001.pilot-receipt.v1",
        "experiment_id": audit.EXPERIMENT_ID,
        "campaign": "pilot",
        "noncanonical": True,
        "scientific_efficacy_claims_authorized": False,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
        "source": {
            "head": "a" * 40,
            "tree": "b" * 40,
            "parent_v05a_head": "c" * 40,
            "parent_v05a_tree": "d" * 40,
        },
        "cells": [{"cell_id": cell["cell_id"]} for cell in cells],
        "verifier_result": {"verdict": "ACCEPT"},
    }
    receipt["pilot_evidence_closure_sha256"] = closure(root)
    write_json(root / audit.RECEIPT_REL, receipt)


def expect_reject(root: Path, code: str) -> None:
    try:
        audit.verify_pilot(root)
    except audit.AuditError as exc:
        assert exc.code == code, f"expected {code}, got {exc.code}: {exc.detail}"
        return
    raise AssertionError(f"expected rejection {code}")


with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-pass-") as td:
    root = Path(td)
    build_root(root, base_cells())
    result = audit.verify_pilot(root)
    assert result["verdict"] == "PILOT_AUDIT_PASS"
    assert result["paired_block_semantics"] == "PASS"

with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-seed-") as td:
    root = Path(td)
    cells = base_cells()
    cells[1]["seed"] = 910099
    build_root(root, cells)
    expect_reject(root, "PAIRED_BLOCK_WORLD_INPUT_MISMATCH")

with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-fixture-") as td:
    root = Path(td)
    cells = base_cells()
    cells[4]["fixture"] = "ordinary"
    build_root(root, cells)
    expect_reject(root, "PAIRED_BLOCK_WORLD_INPUT_MISMATCH")

with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-revision-") as td:
    root = Path(td)
    cells = base_cells()
    cells[2]["revision_index"] = 1
    build_root(root, cells)
    expect_reject(root, "PAIRED_BLOCK_WORLD_INPUT_MISMATCH")

with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-duplicate-policy-") as td:
    root = Path(td)
    cells = base_cells()
    cells[1]["policy"] = "full_symthaea"
    build_root(root, cells)
    expect_reject(root, "PAIRED_BLOCK_DUPLICATE_POLICY")

with tempfile.TemporaryDirectory(prefix="vart-pilot-audit-closure-") as td:
    root = Path(td)
    build_root(root, base_cells())
    plan = json.loads((root / audit.PLAN_REL).read_text(encoding="utf-8"))
    plan["tampered_after_seal"] = True
    write_json(root / audit.PLAN_REL, plan)
    expect_reject(root, "PILOT_AUDIT_CLOSURE_MISMATCH")

print("PASS: post-pilot audit acceptance + A1-A5 deterministic rejection")
