#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path

import audit_vart_world_creative_001_pilot as auditor
import run_vart_world_creative_001_pilot_anchored as anchored


def cells() -> list[dict[str, object]]:
    return [
        {"cell_id": "P1", "trial_id": "p1", "policy": "full_symthaea", "fixture": "ordinary", "seed": 910001, "revision_index": 0, "paired_block_id": "ordinary-r0"},
        {"cell_id": "P2", "trial_id": "p2", "policy": "random_valid", "fixture": "ordinary", "seed": 910001, "revision_index": 0, "paired_block_id": "ordinary-r0"},
        {"cell_id": "P3", "trial_id": "p3", "policy": "heuristic", "fixture": "ordinary", "seed": 910001, "revision_index": 0, "paired_block_id": "ordinary-r0"},
        {"cell_id": "P4", "trial_id": "p4", "policy": "full_symthaea", "fixture": "pretty_trap", "seed": 910002, "revision_index": 0, "paired_block_id": "pretty-r0"},
        {"cell_id": "P5", "trial_id": "p5", "policy": "random_valid", "fixture": "pretty_trap", "seed": 910002, "revision_index": 0, "paired_block_id": "pretty-r0"},
        {"cell_id": "P6", "trial_id": "p6", "policy": "full_symthaea", "fixture": "memory_trap", "seed": 910003, "revision_index": 0, "paired_block_id": "memory-r0"},
        {"cell_id": "P7", "trial_id": "p7", "policy": "no_embodied_experience", "fixture": "ordinary", "seed": 910001, "revision_index": 0, "paired_block_id": "ordinary-r0"},
        {"cell_id": "P8", "trial_id": "p8", "policy": "no_counterfactual_evaluation", "fixture": "ordinary", "seed": 910001, "revision_index": 0, "paired_block_id": "ordinary-r0"},
    ]


def expect_reject(value: list[dict[str, object]], fragment: str) -> None:
    try:
        anchored.paired_design_projection(value)
    except anchored.AnchorError as exc:
        assert fragment in str(exc), (fragment, exc)
        return
    raise AssertionError(f"expected rejection containing {fragment}")


base = cells()
projection = anchored.paired_design_projection(base)
launch_digest = anchored.design_sha256(base)
audit_projection = auditor.verify_paired_design(base)
audit_digest = hashlib.sha256(auditor.canonical_json_bytes(audit_projection)).hexdigest()
assert projection == audit_projection
assert launch_digest == audit_digest

# A pre-execution seed mismatch must never reach DRY_RUN_READY.
bad = copy.deepcopy(base)
bad[1]["seed"] = 910099
expect_reject(bad, "PAIRED_BLOCK_WORLD_INPUT_MISMATCH")

# Fixture and revision drift are the same paired-world-integrity class.
bad = copy.deepcopy(base)
bad[4]["fixture"] = "ordinary"
expect_reject(bad, "PAIRED_BLOCK_WORLD_INPUT_MISMATCH")
bad = copy.deepcopy(base)
bad[2]["revision_index"] = 1
expect_reject(bad, "PAIRED_BLOCK_WORLD_INPUT_MISMATCH")

# Duplicate policy labels inside one paired block are invalid.
bad = copy.deepcopy(base)
bad[1]["policy"] = "full_symthaea"
expect_reject(bad, "PAIRED_BLOCK_DUPLICATE_POLICY")

# Python bools are ints; explicitly reject them as seeds/revision indices.
bad = copy.deepcopy(base)
bad[0]["seed"] = True
expect_reject(bad, "seed must be unsigned 64-bit")
bad = copy.deepcopy(base)
bad[0]["revision_index"] = False
expect_reject(bad, "revision_index must be nonnegative integer")

# Config identity and semantic design identity serve different purposes.
with tempfile.TemporaryDirectory(prefix="vart-anchor-test-") as td:
    root = Path(td)
    a = root / "a.json"
    b = root / "b.json"
    cfg_a = {"cells": base, "note": "first"}
    cfg_b = {"cells": base, "note": "second"}
    a.write_text(json.dumps(cfg_a, sort_keys=True), encoding="utf-8")
    b.write_text(json.dumps(cfg_b, sort_keys=True), encoding="utf-8")
    assert anchored.sha256_file(a) != anchored.sha256_file(b)
    assert anchored.design_sha256(cfg_a["cells"]) == anchored.design_sha256(cfg_b["cells"])

    evidence_root = root / "pilot"
    evidence_root.mkdir()
    try:
        anchored.require_external_path(evidence_root / "anchor.json", evidence_root)
    except anchored.AnchorError:
        pass
    else:
        raise AssertionError("anchor inside evidence root must reject")
    anchored.require_external_path(root / "external-anchor.json", evidence_root)

print("PASS: anchored pilot design digest parity + pre-execution drift rejection")
