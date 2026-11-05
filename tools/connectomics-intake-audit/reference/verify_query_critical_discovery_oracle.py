#!/usr/bin/env python3
"""Independent set/dictionary oracle for the Rust query-critical discovery audit."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


CRITICAL = {
    "QUERY_DECISION_FLIP",
    "ADAPTIVE_POLICY_INVALIDATED",
    "ADAPTIVE_COST_INCREASED",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def valid_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def verify(raw: bytes, data: dict, actual: dict) -> None:
    require(data["schema_version"] == 1, "unsupported schema version")
    require(valid_hex(data["dataset_revision_commit_sha"], 40), "dataset digest malformed")
    require(valid_hex(data["query"]["query_spec_sha256"], 64), "query digest malformed")

    channel_rows = sorted(data["channels"], key=lambda item: item["channel_id"])
    psu_rows = sorted(data["psus"], key=lambda item: item["psu_id"])
    proposal_rows = sorted(data["proposals"], key=lambda item: item["proposal_id"])
    alternative_rows = sorted(data["alternatives"], key=lambda item: item["alternative_id"])

    channel_ids = [item["channel_id"] for item in channel_rows]
    psu_ids = [item["psu_id"] for item in psu_rows]
    proposal_ids = [item["proposal_id"] for item in proposal_rows]
    alternative_ids = [item["alternative_id"] for item in alternative_rows]
    require(len(channel_ids) == len(set(channel_ids)), "duplicate channel IDs")
    require(len(psu_ids) == len(set(psu_ids)), "duplicate PSU IDs")
    require(len(proposal_ids) == len(set(proposal_ids)), "duplicate proposal IDs")
    require(len(alternative_ids) == len(set(alternative_ids)), "duplicate alternative IDs")

    channels = {item["channel_id"]: item for item in channel_rows}
    psus = {item["psu_id"]: item for item in psu_rows}
    alternatives = {item["alternative_id"]: item for item in alternative_rows}
    proposals_for_psu: dict[str, set[str]] = {psu_id: set() for psu_id in psus}
    channel_proposals: dict[str, list[dict]] = {channel_id: [] for channel_id in channels}
    captures_by_channel: dict[str, set[str]] = {channel_id: set() for channel_id in channels}
    capture_union: set[str] = set()

    for proposal in proposal_rows:
        channel_id = proposal["channel_id"]
        psu_id = proposal["psu_id"]
        require(channel_id in channels, "proposal has dangling channel")
        require(psu_id in psus, "proposal has dangling PSU")
        require(valid_hex(proposal["proposal_adjudication_sha256"], 64), "proposal adjudication receipt malformed")
        require(valid_hex(proposal["candidate_world_sha256"], 64), "candidate world digest malformed")
        require(valid_hex(proposal["source_evidence_sha256"], 64), "proposal source evidence digest malformed")
        proposals_for_psu[psu_id].add(proposal["proposal_id"])
        channel_proposals[channel_id].append(proposal)

        disposition = proposal["adjudication_disposition"]
        if disposition == "CONFIRMED_MATCH":
            alternative_id = proposal["matched_alternative_id"]
            require(alternative_id in alternatives, "confirmed proposal has dangling alternative")
            require(valid_hex(proposal["match_adjudication_sha256"], 64), "blinded match receipt malformed")
            require(alternatives[alternative_id]["psu_id"] == psu_id, "proposal/alternative PSU mismatch")
            captures_by_channel[channel_id].add(alternative_id)
            capture_union.add(alternative_id)
        else:
            require(disposition in {"FALSE_POSITIVE", "AMBIGUOUS", "UNADJUDICATED"}, "unknown proposal disposition")
            require("matched_alternative_id" not in proposal, "non-confirmed proposal carries an alternative")
            require("match_adjudication_sha256" not in proposal, "non-confirmed proposal carries a match receipt")

    complete = incomplete = inaccessible = zero_proposal = 0
    for psu_id, psu in psus.items():
        declared = psu["proposal_ids"]
        require(len(declared) == len(set(declared)), f"duplicate PSU proposal reference {psu_id}")
        require(set(declared) == proposals_for_psu[psu_id], f"PSU proposal ledger mismatch {psu_id}")
        require(valid_hex(psu["independent_search_receipt_sha256"], 64), "independent search receipt malformed")
        if not declared:
            zero_proposal += 1
        if psu["search_disposition"] == "COMPLETE_SEARCH":
            complete += 1
        elif psu["search_disposition"] == "INCOMPLETE_SEARCH":
            incomplete += 1
        elif psu["search_disposition"] == "SOURCE_INACCESSIBLE":
            inaccessible += 1
        else:
            raise AssertionError("unknown PSU search disposition")

    require(data["sampling_binding"]["sampled_psu_count"] == len(psus), "sampled PSU denominator mismatch")
    require(data["sampling_binding"]["psu_population_count"] >= len(psus), "sample exceeds population")

    admissible = []
    for alternative in alternative_rows:
        require(alternative["psu_id"] in psus, "alternative has dangling PSU")
        require(valid_hex(alternative["graph_sha256"], 64), "alternative graph digest malformed")
        require(valid_hex(alternative["source_evidence_sha256"], 64), "alternative source digest malformed")
        require(valid_hex(alternative["challenge_report_sha256"], 64), "challenge report digest malformed")
        require(valid_hex(alternative["alternative_adjudication_sha256"], 64), "alternative adjudication digest malformed")
        require(alternative["decision_before_met"] == data["query"]["baseline_decision_met"], "before decision differs from frozen query")
        consequence = alternative["query_consequence"]
        before_cost = alternative["before_worst_case_effort_units"]
        after_cost = alternative["after_worst_case_effort_units"]
        if consequence == "QUERY_DECISION_FLIP":
            require(alternative["decision_after_met"] != alternative["decision_before_met"], "query-flip decisions equal")
        elif consequence == "ADAPTIVE_POLICY_INVALIDATED":
            require(data["baseline"]["adaptive_policy_status"] == "POLICY_EXISTS", "invalidated a nonexistent baseline policy")
            require(alternative["challenge_report_status"] == "NO_POLICY" and after_cost is None, "invalidated policy has inconsistent after state")
        elif consequence == "ADAPTIVE_COST_INCREASED":
            require(before_cost is not None and after_cost is not None and after_cost > before_cost, "policy cost did not strictly increase")
            require(alternative["challenge_report_status"] == "POLICY_EXISTS", "increased cost but policy does not exist")
        elif consequence == "NO_QUERY_EFFECT":
            require(alternative["decision_after_met"] == alternative["decision_before_met"], "no-query-effect decision changed")
            require(before_cost == after_cost, "no-query-effect policy cost changed")
        else:
            require(consequence == "NOT_ESTIMABLE", "unknown query consequence")
        if alternative["challenge_report_status"] == "POLICY_EXISTS":
            require(after_cost is not None, "policy exists but after cost is null")
        if alternative["challenge_report_status"] == "NO_POLICY":
            require(after_cost is None, "no policy but after cost is non-null")
        if alternative["biological_disposition"] == "ADMISSIBLE":
            admissible.append(alternative)

    critical_rows = [
        item for item in admissible if item["query_consequence"] in CRITICAL
    ]
    critical_ids = sorted(item["alternative_id"] for item in critical_rows)
    captured_critical = sorted(set(critical_ids) & capture_union)
    missed_critical = sorted(set(critical_ids) - set(captured_critical))
    noncritical_admissible = len(admissible) - len(critical_rows)

    channel_summaries = []
    for channel_id in sorted(channels):
        found = sorted(captures_by_channel[channel_id])
        channel_summaries.append({
            "channel_id": channel_id,
            "proposal_count": len(channel_proposals[channel_id]),
            "confirmed_match_count": sum(
                1 for row in channel_proposals[channel_id]
                if row["adjudication_disposition"] == "CONFIRMED_MATCH"
            ),
            "unique_alternative_ids": found,
            "query_critical_alternative_ids": sorted(set(found) & set(critical_ids)),
        })

    overlaps = []
    for index, left_id in enumerate(sorted(channels)):
        left_groups = set(channels[left_id]["ancestry_group_ids"])
        for right_id in sorted(channels)[index + 1:]:
            common = sorted(left_groups & set(channels[right_id]["ancestry_group_ids"]))
            if common:
                overlaps.append({
                    "channel_a": left_id,
                    "channel_b": right_id,
                    "shared_group_ids": common,
                })

    if incomplete or inaccessible:
        status = "NOT_PROMOTABLE_INCOMPLETE_OR_INACCESSIBLE_SEARCH"
    elif data["claim_scope"] == "synthetic_known_truth":
        status = "SYNTHETIC_OBSERVED_SAMPLE_VALIDATED"
    else:
        status = "SAMPLED_DECLARED_POPULATION_STRUCTURE_VALIDATED"

    sensitivity = data.get("all_list_missed_sensitivity")
    if sensitivity is None:
        sensitivity_status = "NOT_SUPPLIED"
    else:
        require(sensitivity["owner"] == "9248", "MSE owner mismatch")
        require(valid_hex(sensitivity["report_sha256"], 64), "MSE report digest malformed")
        sensitivity_status = sensitivity["status"]

    expected = {
        "schema_version": 1,
        "tool": "connectome-query-critical-discovery-audit",
        "analysis_id": data["analysis_id"],
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_revision_commit_sha": data["dataset_revision_commit_sha"],
        "query_id": data["query"]["query_id"],
        "query_spec_sha256": data["query"]["query_spec_sha256"],
        "baseline_scenario_set_sha256": data["baseline"]["scenario_set_sha256"],
        "baseline_joint_support_sha256": data["baseline"]["joint_support_sha256"],
        "baseline_action_set_sha256": data["baseline"]["action_set_sha256"],
        "baseline_adaptive_policy_report_sha256": data["baseline"]["adaptive_policy_report_sha256"],
        "sampling_frame_sha256": data["sampling_binding"]["sampling_frame_sha256"],
        "sampling_protocol_sha256": data["sampling_binding"]["sampling_protocol_sha256"],
        "sample_audit_report_sha256": data["sampling_binding"]["sample_audit_report_sha256"],
        "psu_population_count": data["sampling_binding"]["psu_population_count"],
        "sampled_psu_count": data["sampling_binding"]["sampled_psu_count"],
        "complete_search_psu_count": complete,
        "incomplete_search_psu_count": incomplete,
        "source_inaccessible_psu_count": inaccessible,
        "zero_proposal_psu_count": zero_proposal,
        "channel_count": len(channels),
        "proposal_count": len(proposal_rows),
        "note_count": len(data.get("notes", [])),
        "confirmed_match_count": sum(
            1 for row in proposal_rows if row["adjudication_disposition"] == "CONFIRMED_MATCH"
        ),
        "alternative_count": len(alternative_rows),
        "admissible_alternative_count": len(admissible),
        "query_critical_alternative_count": len(critical_ids),
        "captured_query_critical_alternative_count": len(captured_critical),
        "missed_query_critical_alternative_ids": missed_critical,
        "query_critical_alternative_ids": critical_ids,
        "non_query_critical_admissible_alternative_count": noncritical_admissible,
        "channel_summaries": channel_summaries,
        "shared_ancestry_overlaps": overlaps,
        "all_list_missed_sensitivity_owner": "9248",
        "all_list_missed_sensitivity_status": sensitivity_status,
        "population_estimate_status": "NOT_COMPUTED_BY_THIS_TOOL_OWNER_9204",
        "independence_disposition": "NOT_ESTABLISHED_BY_ANCESTRY_LABELS",
        "biological_truth_disposition": "NOT_ESTABLISHED_BY_THIS_TOOL",
        "status": status,
        "scope_note": "This report validates cross-record structure and observed-sample counts only. It does not compute design-weighted population recall, confidence intervals, all-list-missed estimates, biological truth, source authenticity, channel independence, or global hypothesis-space completeness.",
    }
    require(actual == expected, "Rust report differs from independent Python set/dictionary oracle")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_query_critical_discovery_oracle.py INPUT REPORT", file=sys.stderr)
        return 2
    try:
        raw = Path(sys.argv[1]).read_bytes()
        data = json.loads(raw)
        report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        verify(raw, data, report)
        print(
            "verified query-critical discovery report: "
            f"sampled_psus={report['sampled_psu_count']}; "
            f"critical={report['query_critical_alternative_count']}; "
            f"captured={report['captured_query_critical_alternative_count']}; "
            f"missed={len(report['missed_query_critical_alternative_ids'])}"
        )
        return 0
    except Exception as exc:
        print(f"query-critical discovery oracle failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
