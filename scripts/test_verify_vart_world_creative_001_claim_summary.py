#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import verify_vart_world_creative_001_claim_summary as audit


def dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def inventory() -> dict:
    rows: list[dict] = []
    fixtures = ["ordinary", "PrettyTrap", "LocalOptimum", "HiddenDependency", "DelayedConsequence", "CounterfactualDecoy", "Path", "Plaza"]
    run_order = 0
    for i, fixture in enumerate(fixtures):
        cluster = f"{i+1:064x}"[-64:]
        for pidx, policy in enumerate(("full_symthaea", "random_valid", "heuristic")):
            lineage = f"{1000+i*10+pidx:064x}"[-64:]
            rows.append({"trial_id": f"A-{i}-{policy}-r0", "subcampaign": "001A", "policy": policy, "fixture": fixture,
                         "world_cluster_sha256": cluster, "world_lineage_sha256": lineage, "revision_index": 0, "run_order": run_order})
            run_order += 1
            if policy == "full_symthaea":
                for revision in (1, 2, 3):
                    rows.append({"trial_id": f"A-{i}-{policy}-r{revision}", "subcampaign": "001A", "policy": policy, "fixture": fixture,
                                 "world_cluster_sha256": cluster, "world_lineage_sha256": lineage, "revision_index": revision, "run_order": run_order})
                    run_order += 1
    for i in range(8):
        cluster = f"{9000+i:064x}"[-64:]
        for pidx, policy in enumerate(("full_symthaea", "no_reality_ledger_context")):
            rows.append({"trial_id": f"B-{i}-{policy}-r0", "subcampaign": "001B", "policy": policy, "fixture": "MemoryTrap",
                         "world_cluster_sha256": cluster, "world_lineage_sha256": f"{9100+i*10+pidx:064x}"[-64:], "revision_index": 0, "run_order": run_order})
            run_order += 1
    assert len(rows) == 64
    return {"schema": "symthaea.vart-world-creative-001.confirmatory-inventory.v3", "experiment_id": audit.EXPERIMENT_ID, "trials": rows}


def contrast(p: float = 1/256) -> dict:
    return {"effect": 1.0, "test": "exact_paired_randomization", "nonzero_independent_units": 8, "p_value": p,
            "confidence_interval": [0.1, 1.9], "criterion_pass": True}


def packet() -> dict:
    return {
        "schema": audit.SCHEMA,
        "experiment_id": audit.EXPERIMENT_ID,
        "reported_sample_counts": {
            "campaign_full_total": 40,
            "h1": {"full_symthaea": 8, "random_valid": 8, "heuristic": 8},
            "h2_full_lineages": 8,
            "h3": {"full_symthaea": 8, "no_reality_ledger_context": 8},
        },
        "h1": {
            "status": "supported",
            "co_primary_channels": {
                "prediction_error": {"full_vs_random_valid": contrast(), "full_vs_heuristic": contrast()},
                "declared_goal_consequence": {"full_vs_random_valid": contrast(), "full_vs_heuristic": contrast()},
                "physical_validity": {"full_vs_random_valid": contrast(), "full_vs_heuristic": contrast()},
                "protected_side_effects": {"full_vs_random_valid": contrast(), "full_vs_heuristic": contrast()},
            },
            "multiplicity_verdict": "PASS",
        },
        "h2": {
            "status": "supported",
            "lineage_slopes": [-0.082, -0.091, -0.074, -0.088, -0.095, -0.069, -0.081, -0.086],
            "test": "exact_sign_flip",
            "nonzero_independent_units": 8,
            "p_value": 1/256,
            "strict_monotonic_sequence_claimed": False,
        },
        "h3": {
            "status": "supported",
            "endpoints": {
                "counterfactual_history_confusion": {"effect": 0.875, "test": "exact_sign_test", "nonzero_independent_units": 7, "p_value": 1/128, "criterion_pass": True},
                "task_performance": {"effect": 1.0, "test": "exact_paired_randomization", "nonzero_independent_units": 8, "p_value": 1/256, "criterion_pass": True},
            },
            "multiplicity_verdict": "PASS",
        },
        "claim_labels": ["memorytrap_provenance_effect_supported"],
        "claim_authorized": False,
    }


def reject(inv: Path, pkt: Path, obj: dict, expected: str) -> None:
    dump(pkt, obj)
    try:
        audit.verify(inv, pkt)
    except audit.Reject as exc:
        assert exc.code == expected, (exc.code, exc.detail)
    else:
        raise AssertionError(f"expected {expected}")


with tempfile.TemporaryDirectory(prefix="vart-claim-audit-") as td:
    root = Path(td)
    inv = root / "inventory.json"
    pkt = root / "claim.json"
    dump(inv, inventory())

    good = packet()
    dump(pkt, good)
    result = audit.verify(inv, pkt)
    assert result["verdict"] == "CLAIM_SUMMARY_AUDIT_PASS"

    bad = copy.deepcopy(good)
    bad["reported_sample_counts"]["h1"]["full_symthaea"] = 24
    reject(inv, pkt, bad, "CLAIM_AUDIT_H1_SAMPLE_MISMATCH")

    bad = copy.deepcopy(good)
    bad["h1"]["co_primary_channels"]["prediction_error"]["full_vs_random_valid"]["p_value"] = 0.0009
    reject(inv, pkt, bad, "CLAIM_AUDIT_IMPOSSIBLE_P_VALUE")

    bad = copy.deepcopy(good)
    bad["h3"]["endpoints"]["counterfactual_history_confusion"]["p_value"] = 1/256
    reject(inv, pkt, bad, "CLAIM_AUDIT_IMPOSSIBLE_P_VALUE")

    bad = copy.deepcopy(good)
    del bad["h3"]["endpoints"]["task_performance"]
    reject(inv, pkt, bad, "CLAIM_AUDIT_H3_INCOMPLETE")

    bad = copy.deepcopy(good)
    bad["claim_labels"] = ["monotonic_causal_learning"]
    reject(inv, pkt, bad, "CLAIM_AUDIT_OVERCLAIM")

print("PASS: claim summary acceptance + H1 count/p-floor + H3 tie/endpoint + overclaim attacks")
