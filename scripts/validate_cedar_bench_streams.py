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
        "selected_intervention_fact",
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
        if audit["condition"] == "counterfactual":
            fact_id = audit["selected_intervention_fact"]
            require(audit["baseline_values"][fact_id] != audit["counterfactual_values"][fact_id],
                    f"{task_id}: counterfactual failed to change selected fact")
            require(audit["expanded_values"] == audit["counterfactual_values"],
                    f"{task_id}: counterfactual values were not used")
        elif audit["condition"] == "placebo":
            require(audit["placebo_baseline"] != audit["placebo_variant"],
                    f"{task_id}: placebo intervention did not change")
            require(audit["expanded_values"] == audit["baseline_values"],
                    f"{task_id}: placebo intervention changed target facts")
        else:
            require(audit["expanded_values"] == audit["baseline_values"],
                    f"{task_id}: non-counterfactual condition changed target facts")
        if audit["condition"] == "source_removed" and audit["removed_source_id"]:
            require(audit["removed_source_id"] not in source_digests,
                    f"{task_id}: removed source remains in candidate stream")

    templates = {audit["template_id"] for audit in audits}
    expected_templates = {"train": 10, "development": 5, "sealed_test": 5, "all": 20}[expected_split]
    require(len(templates) == expected_templates, f"expected {expected_templates} templates in split")
    conditions_per_template: dict[str, set[str]] = {}
    for audit in audits:
        conditions_per_template.setdefault(audit["template_id"], set()).add(audit["condition"])
    expected = {"baseline", "counterfactual", "placebo", "source_removed"}
    require(all(value == expected for value in conditions_per_template.values()),
            "each template must have all four paired conditions")
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
