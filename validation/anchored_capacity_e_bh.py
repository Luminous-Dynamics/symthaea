#!/usr/bin/env python3
"""Capacity-aware Anchored e-BH using the exact stopped-closure Pareto DP.

For a fixed family of m valid e-values, the anchored local e-collection is
  E_S = 1 - |S|/m + (1/m) * sum_{i in S} e_i
      = 1 + (1/m) * sum_{i in S}(e_i - 1).
The ordinary e-BH rejection set B is certified by this collection. We then
select a maximum-utility certified superset of B under operational capacity.

This is a fixed-family FDR construction under arbitrary dependence, assuming
each input is a valid e-value. It is not a sequential/SupFDR method.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from stopped_eclosure_capacity_pareto import (
    as_fraction,
    fraction_pair,
    select_capacity_aware_pareto,
    verify_capacity_aware_pareto,
    _verify_separator,
)


def e_bh_baseline(
    evalues: Sequence[int | float | str | Fraction],
    alpha: int | float | str | Fraction,
) -> dict:
    """Compute canonical e-BH at level alpha, including all threshold ties."""
    e = tuple(as_fraction(x) for x in evalues)
    a = as_fraction(alpha)
    if not e:
        raise ValueError("at least one e-value is required")
    if any(x < 0 for x in e):
        raise ValueError("e-values must be nonnegative")
    if not 0 < a < 1:
        raise ValueError("alpha must lie in (0,1)")
    m = len(e)
    order = sorted(range(m), key=lambda i: (-e[i], i))
    k_star = 0
    for k in range(1, m + 1):
        if e[order[k - 1]] >= Fraction(m, 1) / (a * k):
            k_star = k
    if k_star == 0:
        threshold = None
        mask = 0
    else:
        threshold = Fraction(m, 1) / (a * k_star)
        mask = sum(1 << i for i, value in enumerate(e) if value >= threshold)
    return {
        "rejection_mask": mask,
        "rejections": [i + 1 for i in range(m) if (mask >> i) & 1],
        "k_star": k_star,
        "threshold": fraction_pair(threshold) if threshold is not None else None,
        "rejection_count": mask.bit_count(),
        "ordering": order,
    }


def anchored_e_bh_max_cardinality(
    evalues: Sequence[int | float | str | Fraction],
    alpha: int | float | str | Fraction,
) -> dict:
    """Return the largest-cardinality certified superset of ordinary e-BH.

    For uniform anchored weights 1/m, let
      h_i = alpha/m * (e_i - 1),
      baseline = sum_i [-h_i]_+.
    At fixed candidate size r and top-r ordered e-values, the closure load is
      baseline + sum_{i<=r} [1/r - [h_i]_+]_+.
    The positive terms are a suffix of the descending e-value order, so one
    binary search plus a prefix sum evaluates each r in O(log m). Sorting
    dominates at O(m log m), matching the paper's step-up complexity.
    """
    from bisect import bisect_right

    e = tuple(as_fraction(x) for x in evalues)
    a = as_fraction(alpha)
    if not e:
        raise ValueError("at least one e-value is required")
    if any(x < 0 for x in e):
        raise ValueError("e-values must be nonnegative")
    if not 0 < a < 1:
        raise ValueError("alpha must lie in (0,1)")
    m = len(e)
    ordinary = e_bh_baseline(e, a)
    order = ordinary["ordering"]
    h = tuple(a * (e[i] - 1) / m for i in order)
    baseline_load = sum(
        (max(-value, Fraction(0)) for value in h),
        Fraction(0),
    )
    positive_h = tuple(max(value, Fraction(0)) for value in h)
    negative_positive_h = tuple(-value for value in positive_h)
    prefix = [Fraction(0)]
    for value in positive_h:
        prefix.append(prefix[-1] + value)

    selected_k = 0
    diagnostics = []
    minimum_load = None
    if ordinary["rejection_count"] == 0:
        diagnostics.append({
            "cardinality": 0,
            "certified": True,
            "closure_load": [0, 1],
        })

    for r in range(max(1, ordinary["rejection_count"]), m + 1):
        threshold = Fraction(1, r)
        # positive_h is nonincreasing; terms [threshold - positive_h_i]_+
        # are positive exactly on the suffix where positive_h_i < threshold.
        first_positive = bisect_right(negative_positive_h[:r], -threshold)
        count = r - first_positive
        selected_charge = count * threshold - (prefix[r] - prefix[first_positive])
        load = baseline_load + selected_charge
        certified = load <= a
        diagnostics.append({
            "cardinality": r,
            "certified": certified,
            "closure_load": fraction_pair(load),
        })
        if certified:
            selected_k = r

    mask = sum(1 << i for i in order[:selected_k])
    if mask & ordinary["rejection_mask"] != ordinary["rejection_mask"]:
        raise ArithmeticError("largest-cardinality candidate failed to retain e-BH baseline")
    separator = _verify_separator(
        e, tuple(Fraction(1, m) for _ in range(m)), (1 << m) - 1, a, mask
    )
    if not separator["certified"]:
        raise ArithmeticError("step-up anchored baseline failed exact O(m) closure verification")
    return {
        "method": "anchored_e_bh_maximum_cardinality_step_up",
        "baseline": ordinary,
        "rejection_mask": mask,
        "rejections": [i + 1 for i in range(m) if (mask >> i) & 1],
        "rejection_count": mask.bit_count(),
        "closure_slack": (
            fraction_pair(separator["minimum_slack"])
            if separator["minimum_slack"] is not None else None
        ),
        "worst_intersection_mask": separator["worst_intersection_mask"],
        "complexity": "O(m log m) time, O(m) memory",
        "cardinality_diagnostics": diagnostics,
    }


def anchored_local_evalue(
    evalues: Sequence[int | float | str | Fraction],
    subset_mask: int,
) -> Fraction:
    """Anchored e-BH local e-value E_S = 1-|S|/m + m^{-1}sum_{i in S}e_i."""
    e = tuple(as_fraction(x) for x in evalues)
    m = len(e)
    if isinstance(subset_mask, bool) or not isinstance(subset_mask, int):
        raise ValueError("subset_mask must be an integer bit mask")
    if subset_mask <= 0 or subset_mask >= (1 << m):
        raise ValueError("subset_mask must identify a nonempty subset")
    size = subset_mask.bit_count()
    total = sum(
        (e[i] for i in range(m) if (subset_mask >> i) & 1),
        Fraction(0),
    )
    return Fraction(1) - Fraction(size, m) + total / m


def verify_anchored_baseline(
    evalues: Sequence[int | float | str | Fraction],
    alpha: int | float | str | Fraction,
    baseline_mask: int | None = None,
) -> dict:
    """Check every intersection for the anchored collection and its e-BH anchor."""
    e = tuple(as_fraction(x) for x in evalues)
    a = as_fraction(alpha)
    m = len(e)
    baseline = e_bh_baseline(e, a)
    mask = baseline["rejection_mask"] if baseline_mask is None else baseline_mask
    if isinstance(mask, bool) or not isinstance(mask, int) or mask < 0 or mask >= (1 << m):
        raise ValueError("invalid baseline rejection mask")

    # The local collection is algebraically the raw projective merger with
    # uniform global weights 1/m. Use its O(m) exact all-subset separator.
    separator = _verify_separator(
        e, tuple(Fraction(1, m) for _ in range(m)), (1 << m) - 1, a, mask
    )
    failures = []
    if mask != baseline["rejection_mask"]:
        failures.append("baseline_mask_is_not_canonical_e_bh")
    if not separator["certified"]:
        failures.append("e_bh_baseline_not_in_anchored_closure")

    # Small-family exhaustive replay makes the local-E_S identity explicit.
    direct_min_slack = None
    worst_subset = 0
    for subset in range(1, 1 << m):
        r_count = (subset & mask).bit_count()
        fdp = Fraction(r_count, mask.bit_count()) if mask else Fraction(0)
        e_s = anchored_local_evalue(e, subset)
        slack = a * e_s - fdp
        if direct_min_slack is None or slack < direct_min_slack:
            direct_min_slack, worst_subset = slack, subset
    if direct_min_slack is None:
        raise ArithmeticError("nonempty intersection universe was not enumerated")
    if mask:
        if direct_min_slack != separator["minimum_slack"]:
            failures.append("anchored_formula_disagrees_with_projective_separator")
    elif direct_min_slack < 0:
        failures.append("empty_e_bh_anchor_not_certified")

    return {
        "valid": not failures,
        "failures": failures,
        "baseline": baseline,
        "minimum_closure_slack": direct_min_slack,
        "worst_intersection_mask": worst_subset,
        "intersections_checked": (1 << m) - 1,
        "local_evalue_identity": "1 - |S|/m + sum(e_i for i in S)/m",
    }


def select_anchored_capacity_e_bh(
    evalues: Sequence[int | float | str | Fraction],
    alpha: int | float | str | Fraction,
    values: Sequence[int | float | str | Fraction],
    costs: Sequence[int | float | str | Fraction],
    capacity: int | float | str | Fraction,
    *,
    frontier_limit: int = 100_000,
    independent_frontier_limit: int = 100_000,
) -> dict:
    """Select a utility-maximizing certified superset of the e-BH baseline.

    Each e-value is assumed valid under its null, with arbitrary dependence
    allowed. Costs/values are predeclared or otherwise external to unknown
    truth labels. Capacity must be sufficient to retain the entire e-BH anchor.
    """
    e = tuple(as_fraction(x) for x in evalues)
    a = as_fraction(alpha)
    v = tuple(as_fraction(x) for x in values)
    c = tuple(as_fraction(x) for x in costs)
    cap = as_fraction(capacity)
    m = len(e)
    if len(v) != m or len(c) != m:
        raise ValueError("values and costs must match e-value count")
    if cap < 0:
        raise ValueError("capacity must be nonnegative")

    anchor_check = verify_anchored_baseline(e, a)
    if not anchor_check["valid"]:
        raise ArithmeticError(f"anchored e-BH baseline verification failed: {anchor_check}")
    baseline = anchor_check["baseline"]
    baseline_mask = baseline["rejection_mask"]
    baseline_cost = sum(
        (c[i] for i in range(m) if (baseline_mask >> i) & 1),
        Fraction(0),
    )
    if baseline_cost > cap:
        raise ValueError(
            "capacity is below the cost of the e-BH anchor; no anchored extension "
            "can retain all baseline discoveries"
        )

    maximum_cardinality = anchored_e_bh_max_cardinality(e, a)
    selected = select_capacity_aware_pareto(
        [[x] for x in e],
        [Fraction(1, m) for _ in range(m)],
        [(1 << m) - 1],
        a,
        v,
        c,
        [cap],
        initial_mandatory_mask=baseline_mask,
        frontier_limit=frontier_limit,
        verify_independent=True,
        independent_frontier_limit=independent_frontier_limit,
    )
    result = {
        "schema": "symthaea-anchored-capacity-ehb-v1",
        "guarantee_scope": (
            "fixed-family FDR <= alpha under arbitrary dependence, conditional "
            "on each input e_i being a valid marginal e-value"
        ),
        "baseline_method": "e-BH",
        "baseline": baseline,
        "anchored_max_cardinality": maximum_cardinality,
        "baseline_cost": fraction_pair(baseline_cost),
        "capacity": fraction_pair(cap),
        "local_evalue_collection": (
            "E_S = 1 - |S|/m + sum_{i in S} e_i/m"
        ),
        "selected": selected,
        "baseline_retained": (
            selected["decisions"][0]["rejection_mask"] & baseline_mask
        ) == baseline_mask,
        "limitations": [
            "fixed finite family; no online or SupFDR claim",
            "input e-value validity is assumed, not proven",
            "exact Pareto frontier may be exponential and is capped",
            "capacity must cover the baseline e-BH discoveries",
        ],
    }
    if not result["baseline_retained"]:
        raise ArithmeticError("capacity extension failed to retain the e-BH baseline")
    return result


def verify_anchored_capacity_e_bh(
    result: dict,
    evalues: Sequence[int | float | str | Fraction],
    alpha: int | float | str | Fraction,
    values: Sequence[int | float | str | Fraction],
    costs: Sequence[int | float | str | Fraction],
) -> dict:
    """Independently replay anchor identity, receipts and Pareto optimality."""
    failures = []
    e = tuple(as_fraction(x) for x in evalues)
    a = as_fraction(alpha)
    v = tuple(as_fraction(x) for x in values)
    c = tuple(as_fraction(x) for x in costs)
    m = len(e)
    if m < 1 or len(v) != m or len(c) != m:
        return {"valid": False, "failures": ["input_dimension_mismatch"]}
    if not isinstance(result, dict):
        return {"valid": False, "failures": ["invalid_result_receipt"]}
    if result.get("schema") != "symthaea-anchored-capacity-ehb-v1":
        failures.append("unsupported_result_schema")

    anchor = verify_anchored_baseline(e, a)
    if not anchor["valid"]:
        return {"valid": False, "failures": anchor["failures"]}
    baseline_mask = anchor["baseline"]["rejection_mask"]

    maximum_cardinality = anchored_e_bh_max_cardinality(e, a)
    if result.get("anchored_max_cardinality") != maximum_cardinality:
        failures.append("anchored_max_cardinality_baseline_mismatch")

    if result.get("baseline") != anchor["baseline"]:
        failures.append("baseline_receipt_mismatch")
    if result.get("baseline_retained") is not True:
        failures.append("baseline_retention_flag_mismatch")
    if result.get("local_evalue_collection") != (
        "E_S = 1 - |S|/m + sum_{i in S} e_i/m"
    ):
        failures.append("local_evalue_formula_receipt_mismatch")

    cap_pair = result.get("capacity")
    if (
        not isinstance(cap_pair, list)
        or len(cap_pair) != 2
        or any(isinstance(x, bool) or not isinstance(x, int) for x in cap_pair)
        or cap_pair[1] <= 0
    ):
        return {
            "valid": False,
            "failures": failures + ["invalid_capacity_receipt"],
        }
    capacity = Fraction(cap_pair[0], cap_pair[1])
    if capacity < 0:
        failures.append("negative_capacity_receipt")

    baseline_cost = sum(
        (c[i] for i in range(m) if (baseline_mask >> i) & 1),
        Fraction(0),
    )
    if result.get("baseline_cost") != fraction_pair(baseline_cost):
        failures.append("baseline_cost_receipt_mismatch")
    if baseline_cost > capacity:
        failures.append("capacity_below_mandatory_baseline_cost")

    selected = result.get("selected")
    if not isinstance(selected, dict):
        return {
            "valid": False,
            "failures": failures + ["missing_optimizer_receipt"],
        }
    if selected.get("schema") != "symthaea-stopped-fdr-capacity-pareto-v1":
        failures.append("optimizer_schema_mismatch")
    if selected.get("initial_mandatory_mask") != baseline_mask:
        failures.append("optimizer_mandatory_mask_mismatch")
    expected_weights = [fraction_pair(Fraction(1, m)) for _ in range(m)]
    if selected.get("weights") != expected_weights:
        failures.append("optimizer_weight_receipt_mismatch")
    if selected.get("alpha") != fraction_pair(a):
        failures.append("optimizer_alpha_receipt_mismatch")

    decisions = selected.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 1:
        return {
            "valid": False,
            "failures": failures + ["optimizer_decision_count_mismatch"],
        }
    first = decisions[0]
    if not isinstance(first, dict):
        return {
            "valid": False,
            "failures": failures + ["invalid_optimizer_decision"],
        }
    mask = first.get("rejection_mask")
    if isinstance(mask, bool) or not isinstance(mask, int) or mask < 0 or mask >= (1 << m):
        return {
            "valid": False,
            "failures": failures + ["invalid_selected_rejection_mask"],
        }

    replay = verify_capacity_aware_pareto(
        decisions,
        tuple((x,) for x in e),
        tuple(Fraction(1, m) for _ in range(m)),
        ((1 << m) - 1,),
        a,
        v,
        c,
        (capacity,),
        initial_mandatory_mask=baseline_mask,
        independent=True,
    )
    failures.extend(replay["failures"])
    if mask & baseline_mask != baseline_mask:
        failures.append("baseline_not_retained_by_selected_mask")
    if selected.get("verification", {}).get("valid") is not True:
        failures.append("optimizer_self_verification_flag_mismatch")
    if selected.get("verification", {}).get("independent_optimality_verified") is not True:
        failures.append("optimizer_self_optimality_flag_mismatch")

    return {
        "valid": not failures,
        "failures": failures,
        "baseline_intersections_checked": anchor["intersections_checked"],
        "anchored_max_cardinality": maximum_cardinality["rejection_count"],
        "pareto_optimality_verified": replay["independent_optimality_verified"],
        "selected_rejections": [i + 1 for i in range(m) if (mask >> i) & 1],
    }


if __name__ == "__main__":
    raise SystemExit("Run test_anchored_capacity_e_bh.py for exact fixtures.")
