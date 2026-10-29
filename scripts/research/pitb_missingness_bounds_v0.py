#!/usr/bin/env python3
"""PITB paired-outcome worst-case bounds and missingness tipping grid.

Research-only diagnostic. Unknown outcomes remain in the frozen denominator;
this tool never authorizes confirmatory analysis. Known timeouts, tool failures,
not-attempted runs and invalid proofs must be recorded as non_solve (0), not
silently reclassified as unknown.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import unittest

SCHEMA_VERSION = "pitb.paired-outcome-ledger.v0"
VALID_KINDS = {"synthetic_fixture", "diagnostic_only"}
VALID_STATUS = {"success", "non_solve", "unknown"}
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ROW_KEYS = {"target_id", "component_id", "control_status", "treatment_status"}
TOP_KEYS = {"schema_version", "dataset_kind", "expected_target_ids", "rows"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def as_outcome(status: str) -> int | None:
    if status == "success":
        return 1
    if status == "non_solve":
        return 0
    if status == "unknown":
        return None
    raise ValueError(f"unregistered outcome status: {status}")


def validate_ledger(doc: dict) -> list[dict]:
    if not isinstance(doc, dict) or set(doc) != TOP_KEYS:
        raise ValueError(f"ledger keys must be exactly {sorted(TOP_KEYS)}")
    if doc["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if not isinstance(doc["dataset_kind"], str) or doc["dataset_kind"] not in VALID_KINDS:
        raise ValueError("dataset_kind must explicitly identify a synthetic or diagnostic ledger")
    ids = doc["expected_target_ids"]
    rows = doc["rows"]
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x.strip() for x in ids):
        raise ValueError("expected_target_ids must be a nonempty list of nonblank strings")
    if len(set(ids)) != len(ids):
        raise ValueError("expected_target_ids contains duplicates")
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ValueError(f"each row must contain exactly {sorted(ROW_KEYS)}")
        tid = row["target_id"]
        cid = row["component_id"]
        if not isinstance(tid, str) or not tid.strip() or tid in by_id:
            raise ValueError("target_id must be nonblank and unique")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError("component_id must be nonblank")
        for arm in ("control_status", "treatment_status"):
            if not isinstance(row[arm], str) or row[arm] not in VALID_STATUS:
                raise ValueError(f"{arm} uses an unregistered status")
        by_id[tid] = row
    expected, actual = set(ids), set(by_id)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"ledger must include exactly one row per frozen target; missing={missing}, extra={extra}")
    # Preserve the frozen target order, not whatever order the input rows use.
    return [by_id[tid] for tid in ids]


def analyze(doc: dict, rows: list[dict], input_sha256: str, source_sha256: str) -> dict:
    n = len(rows)
    lower_total = upper_total = 0.0
    complete_deltas = []
    control_observed_sum = treatment_observed_sum = 0
    missing_control = missing_treatment = 0
    pattern_counts = {"both_observed": 0, "control_only": 0, "treatment_only": 0, "both_unknown": 0}
    component_target_counts = {}

    for row in rows:
        c = as_outcome(row["control_status"])
        t = as_outcome(row["treatment_status"])
        component_target_counts[row["component_id"]] = component_target_counts.get(row["component_id"], 0) + 1
        if c is not None:
            control_observed_sum += c
        else:
            missing_control += 1
        if t is not None:
            treatment_observed_sum += t
        else:
            missing_treatment += 1

        if c is not None and t is not None:
            delta = t - c
            lower_total += delta
            upper_total += delta
            complete_deltas.append(delta)
            pattern_counts["both_observed"] += 1
        elif c is not None and t is None:
            # Treatment outcome in [0,1], control outcome fixed at c.
            lower_total += -c
            upper_total += 1 - c
            pattern_counts["control_only"] += 1
        elif c is None and t is not None:
            # Treatment outcome fixed at t, control outcome in [0,1].
            lower_total += t - 1
            upper_total += t
            pattern_counts["treatment_only"] += 1
        else:
            lower_total -= 1
            upper_total += 1
            pattern_counts["both_unknown"] += 1

    bound_lower = lower_total / n
    bound_upper = upper_total / n
    complete_case_effect = sum(complete_deltas) / len(complete_deltas) if complete_deltas else None

    # Assumption grid: q_t and q_c denote success probabilities among unknown
    # outcomes only. They are sensitivity assumptions, not estimates.
    sensitivity = []
    for q_t in GRID:
        for q_c in GRID:
            delta = (treatment_observed_sum + missing_treatment*q_t - control_observed_sum - missing_control*q_c) / n
            sensitivity.append({
                "assumed_treatment_success_probability_among_unknown": q_t,
                "assumed_control_success_probability_among_unknown": q_c,
                "implied_target_weighted_effect": delta,
                "implied_effect_percentage_points": 100*delta,
                "effect_above_zero_under_assumption": delta > 0,
                "effect_at_least_plus_10pp_under_assumption": delta >= 0.10,
            })

    unknown_targets = sum(1 for row in rows if "unknown" in (row["control_status"], row["treatment_status"]))
    return {
        "schema_version": "pitb.paired-outcome-missingness-bounds-report.v0",
        "status": "diagnostic_only_not_qualified",
        "dataset_kind": doc["dataset_kind"],
        "source_input_sha256": input_sha256,
        "analyzer_source_sha256": source_sha256,
        "target_count": n,
        "component_count": len(component_target_counts),
        "component_target_counts": dict(sorted(component_target_counts.items())),
        "unknown_control_outcomes": missing_control,
        "unknown_treatment_outcomes": missing_treatment,
        "targets_with_at_least_one_unknown_outcome": unknown_targets,
        "unknown_target_fraction": unknown_targets / n,
        "observed_pair_pattern_counts": pattern_counts,
        "complete_pair_count": len(complete_deltas),
        "complete_pair_fraction": len(complete_deltas) / n,
        "complete_case_paired_effect": complete_case_effect,
        "complete_case_paired_effect_percentage_points": None if complete_case_effect is None else 100*complete_case_effect,
        "sharp_finite_cohort_identification_bounds": {
            "lower": bound_lower,
            "upper": bound_upper,
            "lower_percentage_points": 100*bound_lower,
            "upper_percentage_points": 100*bound_upper,
            "width_percentage_points": 100*(bound_upper-bound_lower),
            "includes_zero": bound_lower <= 0 <= bound_upper,
            "lower_exceeds_zero": bound_lower > 0,
            "lower_at_least_plus_10pp": bound_lower >= 0.10,
        },
        "sensitivity_grid": sensitivity,
        "confirmatory_eligible": False,
        "confirmatory_status": "blocked_unknown_outcomes" if unknown_targets else "not_evaluated_by_diagnostic",
        "interpretation_limits": [
            "Identification bounds are exact finite-cohort worst-case bounds for binary paired outcomes; they are not confidence intervals.",
            "Unknown outcomes may not be deleted. The presence of any unknown blocks confirmatory analysis under PITB's protocol.",
            "Known timeouts, tool-unavailable attempts, not-attempted assignments and invalid proof outcomes must be encoded as non_solve, not unknown, for the operational endpoint.",
            "The q-grid is an assumption-based sensitivity analysis, not an estimate of missing outcomes or a replacement for complete receipts.",
            "The diagnostic cannot qualify proof correctness, source eligibility, dependency independence, or the statistical inference method."
        ]
    }


def self_test():
    class BoundsTests(unittest.TestCase):
        def ledger(self, rows, ids=None):
            ids = ids or [r["target_id"] for r in rows]
            return {"schema_version": SCHEMA_VERSION, "dataset_kind": "synthetic_fixture", "expected_target_ids": ids, "rows": rows}
        def row(self, tid, c, t, comp="C1"):
            return {"target_id":tid,"component_id":comp,"control_status":c,"treatment_status":t}
        def test_complete_data_bounds_collapse_to_paired_effect(self):
            rows=[self.row("a","success","non_solve"),self.row("b","non_solve","success")]
            clean=validate_ledger(self.ledger(rows))
            rep=analyze(self.ledger(rows),clean,"x","y")
            self.assertAlmostEqual(rep["sharp_finite_cohort_identification_bounds"]["lower"],0.0)
            self.assertAlmostEqual(rep["sharp_finite_cohort_identification_bounds"]["upper"],0.0)
        def test_unknown_outcome_bounds_cover_every_completion(self):
            rows=[self.row("a","success","unknown"),self.row("b","unknown","success"),self.row("c","unknown","unknown")]
            doc=self.ledger(rows); clean=validate_ledger(doc); rep=analyze(doc,clean,"x","y")
            b=rep["sharp_finite_cohort_identification_bounds"]
            for ya in (0,1):
                for yb in (0,1):
                    for zc in (0,1):
                        for zt in (0,1):
                            effect=((ya-1)+(1-yb)+(zt-zc))/3
                            self.assertLessEqual(b["lower"],effect+1e-12)
                            self.assertGreaterEqual(b["upper"],effect-1e-12)
            self.assertFalse(rep["confirmatory_eligible"])
        def test_non_solve_is_observed_zero_not_missing(self):
            rows=[self.row("a","success","non_solve"),self.row("b","non_solve","success")]
            doc=self.ledger(rows); rep=analyze(doc,validate_ledger(doc),"x","y")
            self.assertEqual(rep["unknown_control_outcomes"],0)
            self.assertEqual(rep["unknown_treatment_outcomes"],0)
            self.assertEqual(rep["sharp_finite_cohort_identification_bounds"]["lower"],0.0)
            self.assertEqual(rep["sharp_finite_cohort_identification_bounds"]["upper"],0.0)
        def test_missing_frozen_row_is_rejected(self):
            rows=[self.row("a","success","success")]
            with self.assertRaisesRegex(ValueError,"exactly one row"):
                validate_ledger(self.ledger(rows,ids=["a","b"]))
        def test_duplicate_rows_are_rejected(self):
            row=self.row("a","success","success")
            with self.assertRaisesRegex(ValueError,"unique"):
                validate_ledger(self.ledger([row,row],ids=["a"]))
        def test_unregistered_outcome_status_is_rejected(self):
            with self.assertRaisesRegex(ValueError,"unregistered"):
                validate_ledger(self.ledger([self.row("a","unknownish","success")]))
        def test_non_string_status_is_rejected_cleanly(self):
            with self.assertRaisesRegex(ValueError,"unregistered"):
                validate_ledger(self.ledger([self.row("a",["unknown"],"success")]))
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(BoundsTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger",nargs="?",type=Path)
    parser.add_argument("report",nargs="?",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        raise SystemExit(self_test())
    if args.ledger is None or args.report is None:
        parser.error("provide <ledger.json> <report.json> or --self-test")
    input_bytes=args.ledger.read_bytes()
    doc=json.loads(input_bytes)
    rows=validate_ledger(doc)
    source_path=Path(__file__).resolve()
    source_sha=sha256(source_path.read_bytes())
    report=analyze(doc,rows,sha256(input_bytes),source_sha)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"report":str(args.report),"target_count":report["target_count"],"unknown_target_fraction":report["unknown_target_fraction"],"identification_bounds_pp":[report["sharp_finite_cohort_identification_bounds"]["lower_percentage_points"],report["sharp_finite_cohort_identification_bounds"]["upper_percentage_points"]],"confirmatory_status":report["confirmatory_status"],"confirmatory_eligible":False},indent=2))

if __name__ == "__main__":
    main()
