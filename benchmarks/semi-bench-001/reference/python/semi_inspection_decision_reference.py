#!/usr/bin/env python3
"""Independent finite-candidate inspection-policy oracle for SEMI-BENCH-001.

Test-only reference code. It evaluates an explicit, synthetic expected-net-value
model and computes the exact subset optimum for at most 18 candidates. It does
not estimate risk from data, control inspections, or claim physical savings.
"""
from __future__ import annotations

import math
from typing import Any


class InspectionDecisionError(ValueError):
    """An inspection-decision input violates the versioned contract."""


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise InspectionDecisionError(f"{field}: expected finite number")
    return float(value)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 12)


def _choose_risk_only(
    rows: list[dict[str, Any]],
    max_minutes: float,
    max_cost: float,
    max_per_group: int,
) -> tuple[list[str], float, float, float]:
    ordered = sorted(rows, key=lambda row: (-row["risk_probability"], row["candidate_id"]))
    selected: list[str] = []
    group_counts: dict[str, int] = {}
    total_cost = total_minutes = total_value = 0.0
    for row in ordered:
        if row["expected_net_value"] <= 0.0:
            continue
        group = row["group_id"]
        if group_counts.get(group, 0) >= max_per_group:
            continue
        next_cost = total_cost + row["inspection_cost"]
        next_minutes = total_minutes + row["inspection_minutes"]
        if next_cost <= max_cost and next_minutes <= max_minutes:
            selected.append(row["candidate_id"])
            group_counts[group] = group_counts.get(group, 0) + 1
            total_cost, total_minutes = next_cost, next_minutes
            total_value += row["expected_net_value"]
    return sorted(selected), total_value, total_cost, total_minutes


def evaluate_inspection_decision(subject: dict[str, Any]) -> dict[str, Any]:
    """Score candidates and select the exact best subset under hard constraints.

    Expected net value for candidate i:

      p_i * sensitivity_i * actionability_i * exposed_i * avoidable_loss_i
      - (1-p_i) * (1-specificity_i) * exposed_i * false_hold_cost_i
      - inspection_cost_i
      - delay_hours_i * delay_cost_per_hour_i

    The exact subset oracle maximizes expected net value under cash, inspection
    time and per-group capacity constraints. The risk-only baseline uses the
    same constraints and candidate utilities but ranks by risk alone.
    """
    if not isinstance(subject, dict) or subject.get("schema") != "luminous.semi-bench.inspection-decision.v1":
        raise InspectionDecisionError("schema: unsupported or missing")
    case_id = subject.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise InspectionDecisionError("case_id: expected non-empty string")

    budget = subject.get("budget")
    if not isinstance(budget, dict):
        raise InspectionDecisionError("budget: expected object")
    max_minutes = _finite(budget.get("max_minutes"), "budget.max_minutes")
    max_cost = _finite(budget.get("max_cash_cost"), "budget.max_cash_cost")
    if max_minutes < 0.0 or max_cost < 0.0:
        raise InspectionDecisionError("budget: values must be non-negative")
    max_per_group = budget.get("max_per_group")
    if type(max_per_group) is not int or max_per_group < 1:
        raise InspectionDecisionError("budget.max_per_group: expected positive integer")

    candidates = subject.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise InspectionDecisionError("candidates: expected non-empty array")
    if len(candidates) > 18:
        raise InspectionDecisionError("candidates: maximum 18 for exhaustive reference oracle")

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise InspectionDecisionError(f"candidates[{index}]: expected object")
        candidate_id = candidate.get("candidate_id")
        group_id = candidate.get("group_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise InspectionDecisionError(f"candidates[{index}].candidate_id: expected non-empty string")
        if candidate_id in seen:
            raise InspectionDecisionError(f"candidates[{index}].candidate_id: duplicate")
        seen.add(candidate_id)
        if not isinstance(group_id, str) or not group_id.strip():
            raise InspectionDecisionError(f"candidates[{index}].group_id: expected non-empty string")

        probability = _finite(candidate.get("risk_probability"), f"{candidate_id}.risk_probability")
        sensitivity = _finite(candidate.get("test_sensitivity"), f"{candidate_id}.test_sensitivity")
        specificity = _finite(candidate.get("test_specificity"), f"{candidate_id}.test_specificity")
        actionability = _finite(candidate.get("actionability_probability"), f"{candidate_id}.actionability_probability")
        exposed_units = _finite(candidate.get("exposed_units"), f"{candidate_id}.exposed_units")
        avoidable_loss = _finite(candidate.get("avoidable_loss_per_defect"), f"{candidate_id}.avoidable_loss_per_defect")
        false_hold_cost = _finite(candidate.get("false_hold_cost_per_unit"), f"{candidate_id}.false_hold_cost_per_unit")
        inspection_cost = _finite(candidate.get("inspection_cost"), f"{candidate_id}.inspection_cost")
        inspection_minutes = _finite(candidate.get("inspection_minutes"), f"{candidate_id}.inspection_minutes")
        delay_hours = _finite(candidate.get("delay_hours"), f"{candidate_id}.delay_hours")
        delay_cost_per_hour = _finite(candidate.get("delay_cost_per_hour"), f"{candidate_id}.delay_cost_per_hour")

        for name, value in (
            ("risk_probability", probability),
            ("test_sensitivity", sensitivity),
            ("test_specificity", specificity),
            ("actionability_probability", actionability),
        ):
            if not 0.0 <= value <= 1.0:
                raise InspectionDecisionError(f"{candidate_id}.{name}: must be in [0, 1]")
        for name, value in (
            ("exposed_units", exposed_units),
            ("avoidable_loss_per_defect", avoidable_loss),
            ("false_hold_cost_per_unit", false_hold_cost),
            ("inspection_cost", inspection_cost),
            ("inspection_minutes", inspection_minutes),
            ("delay_hours", delay_hours),
            ("delay_cost_per_hour", delay_cost_per_hour),
        ):
            if value < 0.0:
                raise InspectionDecisionError(f"{candidate_id}.{name}: must be non-negative")

        expected_prevented_loss = probability * sensitivity * actionability * exposed_units * avoidable_loss
        expected_false_hold_cost = (1.0 - probability) * (1.0 - specificity) * exposed_units * false_hold_cost
        delay_penalty = delay_hours * delay_cost_per_hour
        expected_net_value = expected_prevented_loss - expected_false_hold_cost - inspection_cost - delay_penalty
        residual_missed_defect_loss = probability * (1.0 - sensitivity * actionability) * exposed_units * avoidable_loss
        rows.append({
            "candidate_id": candidate_id,
            "group_id": group_id,
            "risk_probability": probability,
            "expected_prevented_loss": expected_prevented_loss,
            "expected_false_hold_cost": expected_false_hold_cost,
            "inspection_cost": inspection_cost,
            "delay_penalty": delay_penalty,
            "residual_missed_defect_loss": residual_missed_defect_loss,
            "expected_net_value": expected_net_value,
            "net_value_per_minute": expected_net_value / inspection_minutes if inspection_minutes > 0.0 else None,
            "inspection_minutes": inspection_minutes,
        })

    # Exact finite-candidate oracle: enumerate all subsets under hard constraints.
    n = len(rows)
    best_value = best_cost = best_minutes = 0.0
    best_ids: tuple[str, ...] = ()
    for mask in range(1, 1 << n):
        group_counts: dict[str, int] = {}
        total_value = total_cost = total_minutes = 0.0
        selected_ids: list[str] = []
        feasible = True
        for i, row in enumerate(rows):
            if mask & (1 << i):
                group = row["group_id"]
                group_counts[group] = group_counts.get(group, 0) + 1
                if group_counts[group] > max_per_group:
                    feasible = False
                    break
                total_value += row["expected_net_value"]
                total_cost += row["inspection_cost"]
                total_minutes += row["inspection_minutes"]
                selected_ids.append(row["candidate_id"])
        if not feasible or total_value <= 0.0 or total_cost > max_cost or total_minutes > max_minutes:
            continue
        candidate_ids = tuple(sorted(selected_ids))
        better = (
            total_value > best_value + 1e-12
            or (
                abs(total_value - best_value) <= 1e-12
                and (
                    total_cost < best_cost - 1e-12
                    or (
                        abs(total_cost - best_cost) <= 1e-12
                        and (
                            total_minutes < best_minutes - 1e-12
                            or (
                                abs(total_minutes - best_minutes) <= 1e-12
                                and (not best_ids or candidate_ids < best_ids)
                            )
                        )
                    )
                )
            )
        )
        if better:
            best_value, best_cost, best_minutes, best_ids = total_value, total_cost, total_minutes, candidate_ids

    risk_ids, risk_value, risk_cost, risk_minutes = _choose_risk_only(
        rows, max_minutes, max_cost, max_per_group
    )
    selected = set(best_ids)
    ledger = []
    for row in sorted(rows, key=lambda item: item["candidate_id"]):
        ledger.append({
            key: (_round(value) if isinstance(value, float) else value)
            for key, value in {**row, "selected": row["candidate_id"] in selected}.items()
        })

    return {
        "schema": "luminous.semi-bench.inspection-decision-report.v1",
        "case_id": case_id,
        "selection_policy": "exact-exhaustive-net-value-maximization",
        "budget": {
            "max_minutes": _round(max_minutes),
            "max_cash_cost": _round(max_cost),
            "max_per_group": max_per_group,
        },
        "selected_candidate_ids": list(best_ids),
        "selected_expected_net_value": _round(best_value),
        "selected_inspection_cost": _round(best_cost),
        "selected_inspection_minutes": _round(best_minutes),
        "risk_only_comparator": {
            "selected_candidate_ids": risk_ids,
            "expected_net_value": _round(risk_value),
            "inspection_cost": _round(risk_cost),
            "inspection_minutes": _round(risk_minutes),
        },
        "candidate_ledger": ledger,
        "risk_only_ranking": [
            row["candidate_id"] for row in sorted(rows, key=lambda row: (-row["risk_probability"], row["candidate_id"]))
        ],
        "expected_value_ranking": [
            row["candidate_id"] for row in sorted(rows, key=lambda row: (-row["expected_net_value"], row["candidate_id"]))
        ],
        "claim_limits": {
            "expected_values_are_model_based_not_observed": True,
            "uninspected_units_remain_unknown": True,
            "policy_optimum_is_exact_only_for_this_finite_candidate_set": True,
        },
    }


def main() -> int:
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.input or Path(__file__).resolve().parents[1] / "fixtures" / "inspection-decision-known-answer-v1.json"
    try:
        subject = json.loads(source.read_text(encoding="utf-8"))
        result = evaluate_inspection_decision(subject)
    except (OSError, json.JSONDecodeError, InspectionDecisionError) as exc:
        print(f"SEMI-BENCH inspection decision REJECT: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
