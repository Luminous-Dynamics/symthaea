#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
SCHEMA = "symthaea.vart-world-creative-001.claim-summary.v1"
FORBIDDEN_CLAIM_LABELS = {
    "monotonic_causal_learning",
    "general_causal_world_model_confirmed",
    "reality_ledger_universally_necessary",
    "agi_confirmed",
    "general_creativity_confirmed",
}
ALLOWED_HYPOTHESIS_STATUS = {"supported", "not_supported", "inconclusive"}


class Reject(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise Reject(code, detail)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Reject("CLAIM_AUDIT_INPUT_MISSING", str(path)) from exc
    except json.JSONDecodeError as exc:
        raise Reject("CLAIM_AUDIT_INVALID_JSON", f"{path}: {exc}") from exc


def rows_from_inventory(obj: Any) -> list[dict[str, Any]]:
    require(isinstance(obj, dict), "CLAIM_AUDIT_INVENTORY_INVALID", "inventory object")
    rows = obj.get("trials")
    require(isinstance(rows, list) and rows, "CLAIM_AUDIT_INVENTORY_INVALID", "trials")
    require(all(isinstance(r, dict) for r in rows), "CLAIM_AUDIT_INVENTORY_INVALID", "trial rows")
    return rows


def exact_one_sided_floor(nonzero_units: int) -> float:
    require(isinstance(nonzero_units, int) and nonzero_units > 0,
            "CLAIM_AUDIT_EXACT_TEST_INVALID", "nonzero_units")
    return 1.0 / float(1 << nonzero_units)


def require_exact_p(block: dict[str, Any], label: str) -> None:
    test = block.get("test")
    if test not in {"exact_sign_flip", "exact_sign_test", "exact_paired_randomization"}:
        return
    n = block.get("nonzero_independent_units")
    p = block.get("p_value")
    require(isinstance(n, int) and n > 0, "CLAIM_AUDIT_EXACT_TEST_INVALID", f"{label}.nonzero_independent_units")
    require(isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(float(p)) and 0.0 <= float(p) <= 1.0,
            "CLAIM_AUDIT_EXACT_TEST_INVALID", f"{label}.p_value")
    floor = exact_one_sided_floor(n)
    require(float(p) + 1e-15 >= floor,
            "CLAIM_AUDIT_IMPOSSIBLE_P_VALUE",
            f"{label}: p={p} < exact one-sided floor {floor} for {n} informative units")


def derived_design(rows: list[dict[str, Any]]) -> dict[str, Any]:
    h1 = [r for r in rows if r.get("subcampaign") == "001A" and r.get("revision_index") == 0 and r.get("policy") in {"full_symthaea", "random_valid", "heuristic"}]
    h1_counts = {p: sum(1 for r in h1 if r.get("policy") == p) for p in ("full_symthaea", "random_valid", "heuristic")}
    h1_clusters = {r.get("world_cluster_sha256") for r in h1}

    h2 = [r for r in rows if r.get("subcampaign") == "001A" and r.get("policy") == "full_symthaea"]
    h2_lineages: dict[str, set[int]] = {}
    for r in h2:
        lineage = r.get("world_lineage_sha256")
        revision = r.get("revision_index")
        if isinstance(lineage, str) and isinstance(revision, int):
            h2_lineages.setdefault(lineage, set()).add(revision)

    h3 = [r for r in rows if r.get("subcampaign") == "001B" and r.get("revision_index") == 0 and r.get("policy") in {"full_symthaea", "no_reality_ledger_context"}]
    h3_counts = {p: sum(1 for r in h3 if r.get("policy") == p) for p in ("full_symthaea", "no_reality_ledger_context")}
    h3_clusters = {r.get("world_cluster_sha256") for r in h3}

    total_full = sum(1 for r in rows if r.get("policy") == "full_symthaea")
    return {
        "total_trial_count": len(rows),
        "total_full_count": total_full,
        "h1_counts": h1_counts,
        "h1_cluster_count": len(h1_clusters),
        "h2_lineage_revisions": h2_lineages,
        "h3_counts": h3_counts,
        "h3_cluster_count": len(h3_clusters),
    }


def verify(inventory_path: Path, packet_path: Path) -> dict[str, Any]:
    rows = rows_from_inventory(read_json(inventory_path))
    packet = read_json(packet_path)
    require(isinstance(packet, dict) and packet.get("schema") == SCHEMA and packet.get("experiment_id") == EXPERIMENT_ID,
            "CLAIM_AUDIT_PACKET_INVALID", "identity")
    design = derived_design(rows)

    require(design["total_trial_count"] == 64, "CLAIM_AUDIT_DESIGN_MISMATCH", "expected 64 frozen revision-trials")
    require(design["total_full_count"] == 40, "CLAIM_AUDIT_DESIGN_MISMATCH", f"FULL campaign count {design['total_full_count']} != 40")
    require(design["h1_counts"] == {"full_symthaea": 8, "random_valid": 8, "heuristic": 8},
            "CLAIM_AUDIT_H1_SAMPLE_MISMATCH", str(design["h1_counts"]))
    require(design["h1_cluster_count"] == 8, "CLAIM_AUDIT_H1_SAMPLE_MISMATCH", "H1 paired cluster count")
    require(design["h3_counts"] == {"full_symthaea": 8, "no_reality_ledger_context": 8},
            "CLAIM_AUDIT_H3_SAMPLE_MISMATCH", str(design["h3_counts"]))
    require(design["h3_cluster_count"] == 8, "CLAIM_AUDIT_H3_SAMPLE_MISMATCH", "H3 paired cluster count")
    require(len(design["h2_lineage_revisions"]) == 8 and all(v == {0, 1, 2, 3} for v in design["h2_lineage_revisions"].values()),
            "CLAIM_AUDIT_H2_SAMPLE_MISMATCH", "H2 requires 8 FULL lineages with r0..r3")

    reported = packet.get("reported_sample_counts")
    require(isinstance(reported, dict), "CLAIM_AUDIT_PACKET_INVALID", "reported_sample_counts")
    require(reported.get("h1") == {"full_symthaea": 8, "random_valid": 8, "heuristic": 8},
            "CLAIM_AUDIT_H1_SAMPLE_MISMATCH", f"reported H1 counts {reported.get('h1')}")
    require(reported.get("h2_full_lineages") == 8, "CLAIM_AUDIT_H2_SAMPLE_MISMATCH", "reported H2 lineages")
    require(reported.get("h3") == {"full_symthaea": 8, "no_reality_ledger_context": 8},
            "CLAIM_AUDIT_H3_SAMPLE_MISMATCH", f"reported H3 counts {reported.get('h3')}")
    require(reported.get("campaign_full_total") == 40, "CLAIM_AUDIT_DESIGN_MISMATCH", "reported campaign FULL total")

    h1 = packet.get("h1")
    h2 = packet.get("h2")
    h3 = packet.get("h3")
    require(isinstance(h1, dict) and isinstance(h2, dict) and isinstance(h3, dict), "CLAIM_AUDIT_PACKET_INVALID", "hypothesis blocks")

    require(h1.get("status") in ALLOWED_HYPOTHESIS_STATUS, "CLAIM_AUDIT_PACKET_INVALID", "h1.status")
    require(h2.get("status") in ALLOWED_HYPOTHESIS_STATUS, "CLAIM_AUDIT_PACKET_INVALID", "h2.status")
    require(h3.get("status") in ALLOWED_HYPOTHESIS_STATUS, "CLAIM_AUDIT_PACKET_INVALID", "h3.status")

    h1_channels = h1.get("co_primary_channels")
    require(isinstance(h1_channels, dict), "CLAIM_AUDIT_H1_INCOMPLETE", "co_primary_channels")
    for required_channel in ("prediction_error", "declared_goal_consequence", "physical_validity", "protected_side_effects"):
        require(required_channel in h1_channels, "CLAIM_AUDIT_H1_INCOMPLETE", required_channel)
    for channel, block in h1_channels.items():
        if isinstance(block, dict):
            for contrast in ("full_vs_random_valid", "full_vs_heuristic"):
                if isinstance(block.get(contrast), dict):
                    require_exact_p(block[contrast], f"h1.{channel}.{contrast}")
    require(isinstance(h1.get("multiplicity_verdict"), str) and h1.get("multiplicity_verdict"),
            "CLAIM_AUDIT_H1_INCOMPLETE", "multiplicity_verdict")

    slopes = h2.get("lineage_slopes")
    require(isinstance(slopes, list) and len(slopes) == 8 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) for x in slopes),
            "CLAIM_AUDIT_H2_INCOMPLETE", "eight finite lineage slopes")
    require_exact_p(h2, "h2")
    require(h2.get("strict_monotonic_sequence_claimed") is False,
            "CLAIM_AUDIT_H2_OVERCLAIM", "negative slopes do not establish strict revision-by-revision monotonicity")

    endpoints = h3.get("endpoints")
    require(isinstance(endpoints, dict), "CLAIM_AUDIT_H3_INCOMPLETE", "endpoints")
    for endpoint in ("counterfactual_history_confusion", "task_performance"):
        require(isinstance(endpoints.get(endpoint), dict), "CLAIM_AUDIT_H3_INCOMPLETE", endpoint)
        require_exact_p(endpoints[endpoint], f"h3.{endpoint}")
    require(isinstance(h3.get("multiplicity_verdict"), str) and h3.get("multiplicity_verdict"),
            "CLAIM_AUDIT_H3_INCOMPLETE", "multiplicity_verdict")

    labels = packet.get("claim_labels", [])
    require(isinstance(labels, list) and all(isinstance(x, str) for x in labels), "CLAIM_AUDIT_PACKET_INVALID", "claim_labels")
    bad = sorted(FORBIDDEN_CLAIM_LABELS.intersection(labels))
    require(not bad, "CLAIM_AUDIT_OVERCLAIM", f"forbidden labels {bad}")

    return {
        "verdict": "CLAIM_SUMMARY_AUDIT_PASS",
        "experiment_id": EXPERIMENT_ID,
        "campaign_trial_count": design["total_trial_count"],
        "campaign_full_total": design["total_full_count"],
        "h1_paired_cluster_count": design["h1_cluster_count"],
        "h2_lineage_count": len(design["h2_lineage_revisions"]),
        "h3_paired_cluster_count": design["h3_cluster_count"],
        "claim_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit post-unblinding VART claim summary against frozen v3 design")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--claim-packet", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.inventory, args.claim_packet)
    except Reject as exc:
        payload = {"verdict": "CLAIM_SUMMARY_AUDIT_REJECT", "reason_class": exc.code, "detail": exc.detail, "claim_authorized": False}
        if args.json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(f"REJECT {exc.code}: {exc.detail}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print("CLAIM_SUMMARY_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
