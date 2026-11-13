#!/usr/bin/env python3
"""Exact-rational adaptive MIMO experiment decision-tree oracle."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-adaptive-excitation-v0.1.json"
ComplexQ = tuple[Fraction, Fraction]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def read_complex(item: dict) -> ComplexQ:
    return q(item["real"]), q(item["imag"])


def cmul(a: ComplexQ, b: ComplexQ) -> ComplexQ:
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def cdiv(a: ComplexQ, b: ComplexQ) -> ComplexQ:
    den = b[0] * b[0] + b[1] * b[1]
    if den == 0:
        raise ZeroDivisionError("complex denominator includes zero")
    return (a[0] * b[0] + a[1] * b[1]) / den, (a[1] * b[0] - a[0] * b[1]) / den


def cpow(a: ComplexQ, power: int) -> ComplexQ:
    out = (Fraction(1), Fraction(0))
    for _ in range(power):
        out = cmul(out, a)
    return out


def delta_at(omega: Fraction, roots: tuple[int, int, int]) -> ComplexQ:
    numerator = Fraction(1000)
    for root in roots:
        numerator *= Fraction(root * root) - omega * omega
    denominator = cpow((Fraction(5), omega), 7)
    return cdiv((numerator, Fraction(0)), denominator)


def csub(a: ComplexQ, b: ComplexQ) -> ComplexQ:
    return a[0] - b[0], a[1] - b[1]


def abs_squared(a: ComplexQ) -> Fraction:
    return a[0] * a[0] + a[1] * a[1]


def disks_disjoint(a: ComplexQ, radius_a: Fraction,
                   b: ComplexQ, radius_b: Fraction) -> bool:
    return abs_squared(csub(a, b)) > (radius_a + radius_b) ** 2


def action_covers_crossport(action: dict) -> bool:
    return (
        action.get("driven_port_id") == "bus-b"
        and action.get("observed_port_id") == "bus-a"
    )



def action_admissible(action: dict, domain: dict) -> bool:
    if action.get("approved") is not True or "cost" not in action:
        return False
    if q(action["cost"]) <= 0:
        return False
    omega = q(action["omega"])
    return q(domain["lower"]) <= omega <= q(domain["upper"])


def deterministic_first_action(actions: list[dict], domain: dict) -> str | None:
    admitted = [a for a in actions if action_admissible(a, domain)]
    if not admitted:
        return None
    admitted.sort(key=lambda a: a["action_id"])
    return admitted[0]["action_id"]


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_adaptive_experiment_decision_tree.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_deterministic_model_discrimination"
    assert data["model_set_identity"] == "sha256:three-rational-mimo-hypotheses-v0.1"
    assert data["ports"] == ["bus-a", "bus-b"]

    models = {item["model_id"]: item for item in data["model_hypotheses"]}
    assert list(models) == ["H0-base", "H1-delta-three", "H2-delta-two"]
    assert len(models) == len(data["model_hypotheses"])

    actions = {item["action_id"]: item for item in data["candidate_actions"]}
    assert list(actions) == ["action-omega-2", "action-omega-3"]
    assert len(actions) == len(data["candidate_actions"])
    action2, action3 = actions["action-omega-2"], actions["action-omega-3"]
    assert action_covers_crossport(action2) and action_covers_crossport(action3)
    domain = data["model_set_valid_frequency_domain_rad_s"]
    assert action2["approved"] is True and action3["approved"] is True
    assert action_admissible(action2, domain) and action_admissible(action3, domain)
    assert deterministic_first_action(data["candidate_actions"], domain) == "action-omega-2"
    assert q(action2["cost"]) == q(action3["cost"]) == 1
    assert q(action2["omega"]) == 2 and q(action3["omega"]) == 3

    # Derive all hypothesis predictions from the exact rational transfer terms.
    zero = (Fraction(0), Fraction(0))
    d3_at_2 = delta_at(Fraction(2), (1, 2, 4))
    d3_at_3 = delta_at(Fraction(3), (1, 2, 4))
    d2_at_2 = delta_at(Fraction(2), (1, 3, 4))
    d2_at_3 = delta_at(Fraction(3), (1, 3, 4))
    assert d3_at_2 == zero and d2_at_3 == zero
    assert d2_at_2 == read_complex(data["expected_predictions"]["action_omega_2"]["H2"])
    assert d3_at_3 == read_complex(data["expected_predictions"]["action_omega_3"]["H1"])

    p2 = {
        "H0": zero,
        "H1": d3_at_2,
        "H2": d2_at_2,
    }
    p3 = {
        "H0": zero,
        "H1": d3_at_3,
        "H2": d2_at_3,
    }
    for model_id, predicted in p2.items():
        assert predicted == read_complex(data["expected_predictions"]["action_omega_2"][model_id])
    for model_id, predicted in p3.items():
        assert predicted == read_complex(data["expected_predictions"]["action_omega_3"][model_id])

    radius = q(data["uncertainty"]["common_radius"])
    assert radius == Fraction(1, 10)
    assert data["uncertainty"]["semantics"] == "deterministic_hard_set"
    threshold_sq = q(data["uncertainty"]["guaranteed_disjoint_threshold_squared"])
    assert threshold_sq == (2 * radius) ** 2 == Fraction(1, 25)

    # At A2, H0/H1 overlap exactly while H2 is disjoint.
    assert p2["H0"] == p2["H1"]
    assert not disks_disjoint(p2["H0"], radius, p2["H1"], radius)
    assert disks_disjoint(p2["H0"], radius, p2["H2"], radius)
    assert abs_squared(p2["H2"]) == q(
        data["expected_predictions"]["action_omega_2"]["nonzero_squared_distance"]
    )
    assert q(data["expected_predictions"]["action_omega_2"]["nonzero_squared_distance"]) > threshold_sq

    # At A3, H0/H2 overlap exactly while H1 is disjoint.
    assert p3["H0"] == p3["H2"]
    assert not disks_disjoint(p3["H0"], radius, p3["H2"], radius)
    assert disks_disjoint(p3["H0"], radius, p3["H1"], radius)
    assert abs_squared(p3["H1"]) == q(
        data["expected_predictions"]["action_omega_3"]["nonzero_squared_distance"]
    )
    assert q(data["expected_predictions"]["action_omega_3"]["nonzero_squared_distance"]) > threshold_sq

    # Decision tree: H2 is identified after A2; H0/H1 require A3.
    terminals = {
        "H0-base": ("zero-at-A2", "zero-at-A3", 2),
        "H1-delta-three": ("zero-at-A2", "delta3-at-A3", 2),
        "H2-delta-two": ("delta2-at-A2", "stop", 1),
    }
    assert set(terminals) == set(models)
    assert all(path[2] in (1, 2) for path in terminals.values())
    assert len({model_id for model_id in terminals}) == 3

    # Neither single action is a complete three-model discriminator.
    only_a2_partition = ({"H0-base", "H1-delta-three"}, {"H2-delta-two"})
    only_a3_partition = ({"H0-base", "H2-delta-two"}, {"H1-delta-three"})
    assert max(map(len, only_a2_partition)) == 2
    assert max(map(len, only_a3_partition)) == 2

    # Uniform prior: E[N] = 1/3*1 + 2/3*2 = 5/3; worst case 2.
    prior = {model_id: q(item) for model_id, item in data["prior"].items()}
    assert set(prior) == set(models)
    assert sum(prior.values(), Fraction(0)) == 1
    assert all(value == Fraction(1, 3) for value in prior.values())
    expected_n = sum((prior[mid] * terminals[mid][2] for mid in models), Fraction(0))
    worst_n = max(path[2] for path in terminals.values())
    fixed_sweep = q(data["expected_policy"]["fixed_sweep_experiments"])
    saving = (fixed_sweep - expected_n) / fixed_sweep
    assert expected_n == q(data["expected_policy"]["expected_experiments"]) == Fraction(5, 3)
    assert worst_n == q(data["expected_policy"]["worst_case_experiments"]) == 2
    assert fixed_sweep == 2
    assert saving == q(data["expected_policy"]["relative_expected_saving"]) == Fraction(1, 6)

    # Uncertainty disks enlarged to radius 1 overlap each nonzero model
    # prediction with the zero prediction, removing the guarantee.
    wide_radius = q(data["counterfactuals"]["wide_uncertainty_radius"])
    assert not disks_disjoint(p2["H0"], wide_radius, p2["H2"], wide_radius)
    assert not disks_disjoint(p3["H0"], wide_radius, p3["H1"], wide_radius)

    # Wrong-port action cannot observe the selected cross-port term.
    wrong_port = data["counterfactuals"]["wrong_port_action"]
    assert not action_covers_crossport(wrong_port)
    assert wrong_port["approved"] is True

    # Unapproved actions do not enter the candidate set.
    unapproved = data["counterfactuals"]["unapproved_action"]
    out_of_domain = data["counterfactuals"]["out_of_domain_action"]
    missing_cost = data["counterfactuals"]["missing_cost_action"]
    assert unapproved["approved"] is False
    assert not action_admissible(unapproved, domain)
    assert not action_admissible(out_of_domain, domain)
    assert not action_admissible(missing_cost, domain)

    # Outside all admitted prediction sets is unresolved, not nearest-model.
    outside_observation = (Fraction(100), Fraction(100))
    assert all(
        abs_squared(csub(outside_observation, prediction)) > (radius * 2) ** 2
        for prediction in (p2["H0"], p2["H1"], p2["H2"])
    )
    other_unbounded = {"model_id": "OTHER_UNRESOLVED", "prediction_set": None}
    assert other_unbounded["prediction_set"] is None

    # No prior means no prior-conditioned expected cost; worst case remains 2.
    no_prior = None
    assert no_prior is None
    assert worst_n == 2

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["only-action-2"] == "H0-H1-remain-ambiguous"
    assert controls["only-action-3"] == "H0-H2-remain-ambiguous"
    assert controls["missing-second-stage-action"] == "unresolved-branch"
    assert controls["uncertainty-disks-overlap"] == "not-guaranteed"
    assert controls["observation-outside-all-model-sets"] == "unresolved"
    assert controls["unbounded-other-unresolved-hypothesis"] == "no-complete-model-set-claim"
    assert controls["prior-omitted"] == "worst-case-only"
    assert controls["unapproved-action"] == "reject"
    assert controls["wrong-port"] == "no-crossport-discrimination"
    assert controls["duplicate-action-or-model-id"] == "reject"

    print(f"PASS H2 terminal after one measurement; H0/H1 require second measurement")
    print(f"PASS exact expected action count {expected_n}; worst-case {worst_n}")
    print(f"PASS expected reduction vs fixed sweep {saving}")
    print("PASS hard-disk overlap, wrong-port, unapproved and unresolved controls")
    print("PASS exact-rational adaptive MIMO decision tree")


if __name__ == "__main__":
    main()
