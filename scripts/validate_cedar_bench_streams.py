#!/usr/bin/env python3
"""Check generated CEDAR candidate/auditor stream integrity and separation."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(candidate_path: Path, audit_path: Path) -> None:
    candidates = read_jsonl(candidate_path)
    audits = read_jsonl(audit_path)
    expected_split = sys.argv[3] if len(sys.argv) > 3 else "all"
    require(expected_split in {"train", "development", "sealed_test", "all"}, "invalid expected split")
    expected_count = {"train": 40, "development": 20, "sealed_test": 20, "all": 80}[expected_split]
    require(len(candidates) == len(audits) == expected_count,
            f"expected {expected_count} paired trials for {expected_split}")
    candidate_ids = [row["task_id"] for row in candidates]
    audit_ids = [row["task_id"] for row in audits]
    require(len(set(candidate_ids)) == expected_count, "candidate task IDs must be unique")
    require(len(set(audit_ids)) == expected_count, "auditor task IDs must be unique")
    require(set(candidate_ids) == set(audit_ids), "candidate and auditor task IDs do not align")

    candidate_by_id = {row["task_id"]: row for row in candidates}
    audit_by_id = {row["task_id"]: row for row in audits}
    forbidden_candidate_keys = {
        "condition", "expected_behavior", "expanded_values",
        "baseline_values", "counterfactual_values", "seed_commitment",
        "selected_intervention_fact", "split", "expected_answer", "citation_contract",
    }
    for task_id, candidate in candidate_by_id.items():
        audit = audit_by_id[task_id]
        require(not (forbidden_candidate_keys & set(candidate)),
                f"{task_id}: candidate stream leaked auditor fields")
        require(audit["split"] in {"train", "development", "sealed_test"},
                f"{task_id}: invalid audit split")
        require(expected_split == "all" or audit["split"] == expected_split,
                f"{task_id}: audit split mismatch")
        require(audit["condition"] in {"baseline", "counterfactual", "placebo", "source_removed"},
                f"{task_id}: invalid auditor condition")
        source_digests = {source["source_id"]: canonical_digest(source) for source in candidate["sources"]}
        require(source_digests == audit["source_digests"],
                f"{task_id}: source digest mismatch between streams")
        require(canonical_digest(candidate["sources"]) == audit["source_bundle_sha256"],
                f"{task_id}: source-bundle digest mismatch")
        require(audit["seed_commitment"] and len(audit["seed_commitment"]) == 64,
                f"{task_id}: invalid seed commitment")
        require(isinstance(audit["expected_answer"], dict) and "answerable" in audit["expected_answer"],
                f"{task_id}: missing independent answer oracle")
        require(isinstance(audit["citation_contract"], dict)
                and "allowed_citation_source_ids" in audit["citation_contract"]
                and "forbidden_decoy_source_ids" in audit["citation_contract"]
                and "origin_groups" in audit["citation_contract"]
                and "independent_origin_count" in audit["citation_contract"]
                and "minimum_independent_origins" in audit["citation_contract"]
                and "corroboration_available" in audit["citation_contract"],
                f"{task_id}: missing citation audit contract")
        require(set(audit["citation_contract"]["allowed_citation_source_ids"]) <= set(source_digests),
                f"{task_id}: allowed citation source absent from candidate evidence")
        require(set(audit["citation_contract"]["forbidden_decoy_source_ids"]) <= set(source_digests),
                f"{task_id}: decoy citation source absent from candidate evidence")
        origin_groups: dict[str, list[str]] = {}
        for source in candidate["sources"]:
            origin = source["origin_id"]
            origin_groups.setdefault(origin, []).append(source["source_id"])
        origin_groups = {key: sorted(value) for key, value in sorted(origin_groups.items())}
        require(origin_groups == audit["citation_contract"]["origin_groups"],
                f"{task_id}: source-origin grouping differs between candidate and auditor streams")
        independent_count = len(origin_groups)
        require(independent_count == audit["citation_contract"]["independent_origin_count"],
                f"{task_id}: independent-origin count mismatch")
        require(audit["citation_contract"]["corroboration_available"]
                == (independent_count >= audit["citation_contract"]["minimum_independent_origins"]),
                f"{task_id}: corroboration-availability flag is incorrect")
        if audit["condition"] == "counterfactual":
            fact_id = audit["selected_intervention_fact"]
            require(audit["baseline_values"][fact_id] != audit["counterfactual_values"][fact_id],
                    f"{task_id}: counterfactual failed to change selected fact")
            require(audit["expanded_values"] == audit["counterfactual_values"],
                    f"{task_id}: counterfactual values were not used")
            require(audit["expected_answer"].get("answerable") is True,
                    f"{task_id}: counterfactual answer unexpectedly became unanswerable")
        elif audit["condition"] == "placebo":
            require(audit["placebo_baseline"] != audit["placebo_variant"],
                    f"{task_id}: placebo intervention did not change")
            require(audit["expanded_values"] == audit["baseline_values"],
                    f"{task_id}: placebo intervention changed target facts")
        else:
            require(audit["expanded_values"] == audit["baseline_values"],
                    f"{task_id}: non-counterfactual condition changed target facts")
        if audit["condition"] == "source_removed" and audit["removed_source_ids"]:
            require(not (set(audit["removed_source_ids"]) & set(source_digests)),
                    f"{task_id}: a removed source remains in candidate stream")

    templates = {audit["template_id"] for audit in audits}
    expected_templates = {"train": 10, "development": 5, "sealed_test": 5, "all": 20}[expected_split]
    require(len(templates) == expected_templates, f"expected {expected_templates} templates in split")
    conditions_per_template: dict[str, set[str]] = {}
    audit_groups: dict[str, dict[str, dict[str, Any]]] = {}
    for audit in audits:
        conditions_per_template.setdefault(audit["template_id"], set()).add(audit["condition"])
        audit_groups.setdefault(audit["template_id"], {})[audit["condition"]] = audit

    expected = {"baseline", "counterfactual", "placebo", "source_removed"}
    require(all(value == expected for value in conditions_per_template.values()),
            "each template must have all four paired conditions")
    for template_id, group in audit_groups.items():
        baseline = group["baseline"]
        counterfactual = group["counterfactual"]
        placebo = group["placebo"]
        require(baseline["expected_answer"] != counterfactual["expected_answer"],
                f"{template_id}: counterfactual did not change the answer oracle")
        require(baseline["expected_answer"] == placebo["expected_answer"],
                f"{template_id}: placebo intervention changed the answer oracle")
        require(baseline["citation_contract"] == counterfactual["citation_contract"]
                == placebo["citation_contract"],
                f"{template_id}: non-removal intervention changed citation eligibility")
        if template_id == "CEDAR-08":
            require(baseline["citation_contract"]["independent_origin_count"] == 2,
                    "CEDAR-08 must have two independent origins despite three URLs")
            require(len(baseline["citation_contract"]["origin_groups"]["origin-a"]) == 2,
                    "CEDAR-08 mirror URLs must share one origin group")
            require(baseline["expected_behavior"] == "TWO_INDEPENDENT_ORIGINS_CORROBORATE_VALUE_NOT_THREE_URLS",
                    "CEDAR-08 baseline must count origin families, not URLs")
            require(counterfactual["expected_behavior"] == "TRACKS_COUNTERFACTUAL_AND_PRESERVES_TWO_INDEPENDENT_ORIGINS",
                    "CEDAR-08 counterfactual must track value and retain two origins")
            require(placebo["expected_behavior"] == "PLACEBO_STABLE_AND_PRESERVES_TWO_INDEPENDENT_ORIGINS",
                    "CEDAR-08 placebo must preserve answer and origin coverage")
            require(removed["citation_contract"]["independent_origin_count"] == 1,
                    "CEDAR-08 source removal must leave one origin, not two mirrored URLs")
            require(removed["citation_contract"]["corroboration_available"] is False,
                    "CEDAR-08 must downgrade corroboration after one origin is removed")
            require(removed["expected_behavior"] == "REMAINS_SUPPORTED_BY_ONE_INDEPENDENT_ORIGIN_NOT_CROSS_CORROBORATED",
                    "CEDAR-08 must caveat single-origin support after removal")
        if template_id == "CEDAR-15":
            require(baseline["expected_behavior"] == "SOURCE_CONFLICT_REPRESENTED",
                    "CEDAR-15 baseline must explicitly represent source conflict")
            require(counterfactual["expected_behavior"] == "SOURCE_CONFLICT_REPRESENTED_AND_TRACKS_CHANGED_SOURCE",
                    "CEDAR-15 counterfactual must represent conflict and track the altered source")
            require(placebo["expected_behavior"] == "SOURCE_CONFLICT_REPRESENTED_AND_PLACEBO_STABLE",
                    "CEDAR-15 placebo must represent conflict and preserve its answer")
        removed = group["source_removed"]
        require(set(removed["citation_contract"]["allowed_citation_source_ids"])
                <= set(removed["source_digests"]),
                f"{template_id}: source-removal citation contract names a missing source")
        require(removed["expected_answer"].get("answerable") is False
                or template_id in {"CEDAR-08", "CEDAR-15"},
                f"{template_id}: source removal unexpectedly preserves an answer")
        if template_id == "CEDAR-15":
            require(removed["expected_behavior"] == "PARTIAL_CONFLICT_REQUIRES_ONE_SOURCE_CAVEAT",
                    "CEDAR-15 source removal must explicitly caveat the remaining single-source evidence")
            require(removed["expected_answer"].get("complete") is False,
                    "CEDAR-15 source removal must mark the conflict record incomplete")

    print(f"CEDAR-Bench {expected_split} streams: {expected_count}/{expected_count} candidate and auditor trials align; no condition/gold leakage detected")


def main() -> int:
    try:
        require(len(sys.argv) in {3, 4}, "usage: validate_cedar_bench_streams.py candidate.jsonl auditor.jsonl [split]")
        validate(Path(sys.argv[1]), Path(sys.argv[2]))
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"CEDAR-Bench stream validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
