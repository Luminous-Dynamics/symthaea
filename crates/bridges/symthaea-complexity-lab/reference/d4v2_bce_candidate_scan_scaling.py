#!/usr/bin/env python3
"""Adversarial fan-in scan scaling for the source-derived D4v2 BCE model."""
from __future__ import annotations
import json
import sys
from fractions import Fraction
from pathlib import Path

from verify_d4v2_dynamic_bce_trace import D4BCEStateModel, evaluate


class InstrumentedD4BCEStateModel(D4BCEStateModel):
    def __init__(self, clauses, nvars, selected):
        self.candidate_entries_inspected = 0
        super().__init__(clauses, nvars, selected)

    def _inprocessing(self):
        # Same transition order as the source-derived reference model; count each
        # candidate entry inspected by D4's index-1 forward scan.
        while self.pending:
            inactive_clause = self.pending.pop()
            retained = []
            for info_id in self.watchlists[inactive_clause]:
                info = self.protected[info_id]
                target = info["target"]
                blocker_var = abs(info["blocker"])
                if not self.active[target] or blocker_var in self.assignment:
                    retained.append(info_id)
                    continue
                replacement = None
                for index in range(1, len(info["candidate_clauses"])):
                    self.candidate_entries_inspected += 1
                    candidate = info["candidate_clauses"][index]
                    if self.active[candidate]:
                        replacement = index
                        break
                if replacement is not None:
                    candidates = info["candidate_clauses"]
                    candidates[0], candidates[replacement] = candidates[replacement], candidates[0]
                    new_sentinel = candidates[0]
                    self.watchlists[new_sentinel].append(info_id)
                    self.events.append(["watch_relocated", info_id, inactive_clause, new_sentinel])
                else:
                    self._save_clause(target)
                    self.active[target] = False
                    self.pending.append(target)
                    retained.append(info_id)
                    self.events.append(["blocked_by_exhausted_opposing_set", target, info_id, inactive_clause])
            self.watchlists[inactive_clause] = retained


def exact_weights(selected, salt):
    weights = {}
    for v in selected:
        false = Fraction(v + 1 + salt % 5, 2 * v + 5 + salt % 7)
        true = Fraction(v + 3 + salt % 3, 3 * v + 7 + salt % 11)
        weights[str(v)] = {
            "false": f"{false.numerator}/{false.denominator}",
            "true": f"{true.numerator}/{true.denominator}",
        }
    return weights


def run_case(k):
    nvars = k + 2
    selected = list(range(2, nvars + 1))
    clauses = [{"id": "T", "lits": [1, 2]}]
    clauses.extend(
        {"id": f"O{i}", "lits": [-1, v]}
        for i, v in enumerate(range(3, nvars + 1), 1)
    )
    ids = [c["id"] for c in clauses]
    model = InstrumentedD4BCEStateModel(clauses, nvars, selected)
    target = next(
        item for item in model.protected
        if item["target"] == "T" and item["blocker"] == 1
    )
    expected_candidates = [f"O{i}" for i in range(1, k + 1)]
    if target["candidate_clauses"] != expected_candidates:
        raise AssertionError("protected candidate list does not match fan-in fixture")

    weights = exact_weights(selected, k)
    # a=false keeps T active; b_i=true deactivates each opposing clause in order.
    model.pre_update([-2], "a=false")
    relocations = deletions = 0
    for i, variable in enumerate(range(3, nvars + 1), 1):
        after = model.pre_update([variable], f"b{i}=true")
        relocations += sum(event[0] == "watch_relocated" for event in after["events"])
        deletions += sum(event[0] == "blocked_by_exhausted_opposing_set" for event in after["events"])

    assignment = {2: False}
    assignment.update({v: True for v in range(3, nvars + 1)})
    source = evaluate(clauses, nvars, selected, weights, assignment=assignment, active_ids=ids)
    active_ids = [cid for cid in ids if model.active[cid]]
    processed = evaluate(clauses, nvars, selected, weights, assignment=assignment, active_ids=active_ids)
    if source["projected_support"] != processed["projected_support"] or source["pwmc"] != processed["pwmc"]:
        raise AssertionError("fan-in deletion changed projected support/PWMC")
    scans = model.candidate_entries_inspected
    expected_scans = (k - 1) * (k + 2) // 2
    if scans != expected_scans or relocations != k - 1 or deletions != 1 or model.active["T"]:
        raise AssertionError(
            f"fan-in transition mismatch k={k}: scans={scans}, relocations={relocations}, deletions={deletions}"
        )

    final_sentinel = target["candidate_clauses"][0]
    # Undo the whole path. Clause and assignment state return; watch placement does not.
    for _ in range(k + 1):
        model.post_update()
        model._assert_watch_invariant()
    if any(not model.active[cid] for cid in ids) or model.assignment:
        raise AssertionError("postUpdate did not restore root clause/assignment state")
    if target["candidate_clauses"][0] != final_sentinel:
        raise AssertionError("backtrack incorrectly reverted persistent sentinel")
    if final_sentinel != f"O{k}" or info_not_watched(model, target["id"]):
        raise AssertionError("persistent sentinel is not parked on final candidate")

    # Sibling: make all but the final opposing clause satisfied. The persistent
    # sentinel O_k remains active, so T must not be dynamically deleted.
    model.pre_update([-2], "sibling:a=false")
    for i, variable in enumerate(range(3, nvars), 1):
        model.pre_update([variable], f"sibling:b{i}=true")
    if not model.active["T"] or not model.active[final_sentinel]:
        raise AssertionError("sibling incorrectly inherited deletion or lost active sentinel")
    sibling_assignment = {2: False}
    sibling_assignment.update({v: True for v in range(3, nvars)})
    sibling_source = evaluate(clauses, nvars, selected, weights, assignment=sibling_assignment, active_ids=ids)
    sibling_processed = evaluate(
        clauses, nvars, selected, weights, assignment=sibling_assignment,
        active_ids=[cid for cid in ids if model.active[cid]]
    )
    if (sibling_source["projected_support"] != sibling_processed["projected_support"]
            or sibling_source["pwmc"] != sibling_processed["pwmc"]):
        raise AssertionError("sibling context changed projected support/PWMC")
    if not sibling_source["projected_support"]:
        raise AssertionError("sibling control should have nonempty projected support")
    return {
        "candidate_count": k,
        "candidate_entries_inspected": scans,
        "closed_form": expected_scans,
        "watch_relocations": relocations,
        "dynamic_deletions": deletions,
        "final_sentinel_after_backtrack": final_sentinel,
        "source_full_models_deleted_branch": source["full_models"],
        "processed_full_models_deleted_branch": processed["full_models"],
        "deleted_branch_projected_support_size": source["projected_support_size"],
        "deleted_branch_pwmc": source["pwmc"],
        "sibling_target_remains_active": model.active["T"],
        "sibling_projected_support_size": sibling_source["projected_support_size"],
        "sibling_pwmc": sibling_source["pwmc"],
    }


def info_not_watched(model, info_id):
    return not any(info_id in ids for ids in model.watchlists.values())


def main():
    rows = [run_case(k) for k in (2, 4, 8, 16, 32, 64)]
    result = {
        "format": "symthaea.d4v2.dynamic-bce-candidate-scan-scaling",
        "version": 1,
        "claim_ceiling": "instrumented_independent_source_derived_reference_model_not_D4_binary_benchmark_or_formal_proof",
        "family": {
            "target_clause": "(h OR a)",
            "opposing_clauses": "(NOT h OR b_i)",
            "blocker": "h (hidden)",
            "selected_variables": "a and all b_i",
            "branch": "a=false, then b_1=true through b_k=true",
            "expected_scans": "(k-1)(k+2)/2",
        },
        "results": rows,
        "runtime": {"python": sys.version.split()[0], "platform": sys.platform},
    }
    output = Path(__file__).with_name("d4v2_bce_candidate_scan_scaling.v1.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS fan-in scan counts match closed form for k=2,4,8,16,32,64")
    print("PASS exact projected support/PWMC and persistent-sentinel sibling controls")


if __name__ == "__main__":
    main()
