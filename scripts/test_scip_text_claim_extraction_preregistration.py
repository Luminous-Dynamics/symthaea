#!/usr/bin/env python3
"""External hostile-input tests for the V18 extraction preregistration validator."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "scip_text_claim_extraction_preregistration.py"
PREREG = ROOT / "scripts" / "qualification" / "scip_text_claim_extraction_preregistration_v1.json"
EXPECTED = "96e2ec5e1fad2e4a955f15f1b0ec58d0ab748dc582cb5ed172f6b55ee42bacb3"


def execute(document: dict) -> tuple[int, str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prereg.json"
        path.write_text(json.dumps(document, ensure_ascii=False), "utf-8")
        proc = subprocess.run([sys.executable, str(VALIDATOR), str(path)], capture_output=True, text=True, check=False)
        return proc.returncode, proc.stdout, proc.stderr


def rejected(document: dict, name: str) -> None:
    code, out, err = execute(document)
    if code == 0:
        raise AssertionError(f"{name}: hostile mutation accepted: {out}")
    if "ERROR:" not in err:
        raise AssertionError(f"{name}: rejection lacked explicit error: {err}")


def raw_rejected(raw: str, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prereg.json"
        path.write_text(raw, "utf-8")
        proc = subprocess.run([sys.executable, str(VALIDATOR), str(path)], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            raise AssertionError(f"{name}: hostile raw input accepted")


def main() -> int:
    document = json.loads(PREREG.read_text("utf-8"))
    code, out, err = execute(document)
    assert code == 0, err
    result = json.loads(out)
    assert result["preregistration_sha256"] == EXPECTED
    assert result["authority"] == "preregistration-only"
    assert result["confirmatory_execution_authorized"] is False
    assert result["surface_fidelity_established"] is False
    assert result["runtime_capability_authorized"] is False

    mutations: list[tuple[str, Callable[[dict], None]]] = [
        ("authority-promotion", lambda d: d.__setitem__("authority", "qualified")),
        ("dimension-code", lambda d: d["dimensions"][3].__setitem__("code", 99)),
        ("dimension-order", lambda d: d["dimensions"].reverse()),
        ("confirmatory-label-leak", lambda d: d["corpus"].__setitem__("confirmatory_labels_sealed_before_candidate_evaluation", False)),
        ("source-graph-leak", lambda d: d["candidate_protocol"].__setitem__("source_graph_hidden_during_extraction", False)),
        ("source-id-leak", lambda d: d["candidate_protocol"].__setitem__("candidate_must_not_emit_source_claim_ids", False)),
        ("ground-truth-leak", lambda d: d["candidate_protocol"].__setitem__("ground_truth_inventory_hidden_during_extraction", False)),
        ("post-freeze-aligner", lambda d: d["evaluation_binding"].__setitem__("no_model_or_human_semantic_aligner_in_confirmatory_evaluation", False)),
        ("threshold-relaxation", lambda d: d["confirmatory_gates"].__setitem__("minimum_negative_sensitivity_each_dimension", "0.90")),
        ("multiple-confirmatory-attempts", lambda d: d["candidate_protocol"].__setitem__("one_confirmatory_attempt_per_frozen_candidate", False)),
        ("case-reuse", lambda d: d["corpus"].__setitem__("no_case_reuse_across_splits", False)),
        ("multi-factor-negative", lambda d: d["corpus"].__setitem__("negative_cases_change_exactly_one_semantic_factor", False)),
        ("wrong-total", lambda d: d["corpus"].__setitem__("minimum_total_confirmatory_cases", 831)),
        ("promotion-leak", lambda d: d["promotion_boundary"].__setitem__("benchmark_pass_does_not_establish_surface_fidelity", False)),
        ("unknown-top", lambda d: d.__setitem__("surface_fidelity_established", True)),
        ("unknown-nested", lambda d: d["candidate_protocol"].__setitem__("secret_ground_truth_hint", True)),
        ("unknown-gate", lambda d: d["confirmatory_gates"].__setitem__("overall_score", "0.99")),
    ]
    for name, mutate in mutations:
        changed = copy.deepcopy(document)
        mutate(changed)
        rejected(changed, name)

    raw = PREREG.read_text("utf-8")
    raw_rejected(raw.replace('"schema"', '"schema": "shadow", "schema"', 1), "duplicate-key")
    print(f"PASS_PREREGISTRATION_ADVERSARIAL preregistration_sha256={EXPECTED}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
