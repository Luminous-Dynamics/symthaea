#!/usr/bin/env python3
"""Exact-rational, provenance-bound MIMO model holdout falsification."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-holdout-falsification-v0.1.json"

ComplexQ = tuple[Fraction, Fraction]
Interval = tuple[Fraction, Fraction]
IDENTITY_KEYS = (
    "campaign_id",
    "acquisition_id",
    "topology_identity",
    "sample_clock_identity",
    "port_order",
    "calibration_nuisance_id",
)


def q(value: dict[str, str]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def read_complex(value: dict[str, Any]) -> ComplexQ:
    return q(value["re"]), q(value["im"])


def cadd(left: ComplexQ, right: ComplexQ) -> ComplexQ:
    return left[0] + right[0], left[1] + right[1]


def csub(left: ComplexQ, right: ComplexQ) -> ComplexQ:
    return left[0] - right[0], left[1] - right[1]


def cmul(left: ComplexQ, right: ComplexQ) -> ComplexQ:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def cdiv(left: ComplexQ, right: ComplexQ) -> ComplexQ:
    denominator = right[0] * right[0] + right[1] * right[1]
    if denominator == 0:
        raise ZeroDivisionError("complex rational division by zero")
    return (
        (left[0] * right[0] + left[1] * right[1]) / denominator,
        (left[1] * right[0] - left[0] * right[1]) / denominator,
    )


def cpow(value: ComplexQ, exponent: int) -> ComplexQ:
    if exponent < 0:
        raise ValueError("negative powers are not used by this oracle")
    result = (Fraction(1), Fraction(0))
    for _ in range(exponent):
        result = cmul(result, value)
    return result


def delta_at_jw(omega: Fraction) -> ComplexQ:
    """Evaluate 1000*(s²+1)*(s²+4)*(s²+16)/(s+5)^7 at s=j*omega."""
    if omega < 0:
        raise ValueError("V0.1 expects nonnegative angular frequency")
    s = (Fraction(0), omega)
    s_squared = cmul(s, s)
    numerator = (Fraction(1), Fraction(0))
    for offset in (1, 4, 16):
        numerator = cmul(
            numerator,
            (s_squared[0] + offset, s_squared[1]),
        )
    denominator = cpow((Fraction(5), omega), 7)
    value = cdiv(numerator, denominator)
    return value[0] * 1000, value[1] * 1000


def interval(lower: dict[str, str], upper: dict[str, str]) -> Interval:
    lo, hi = q(lower), q(upper)
    if lo > hi:
        raise ValueError("interval lower endpoint exceeds upper endpoint")
    return lo, hi


def intersect(left: Interval, right: Interval) -> Interval | None:
    lo = max(left[0], right[0])
    hi = min(left[1], right[1])
    return (lo, hi) if lo <= hi else None


def contains(bounds: Interval, value: Fraction) -> bool:
    return bounds[0] <= value <= bounds[1]


def distance_to_interval(value: Fraction, bounds: Interval) -> Fraction:
    if value < bounds[0]:
        return bounds[0] - value
    if value > bounds[1]:
        return value - bounds[1]
    return Fraction(0)


def candidate_prediction(candidate_name: str, omega: Fraction) -> ComplexQ:
    if candidate_name == "ZA":
        return Fraction(0), Fraction(0)
    if candidate_name == "ZB":
        return delta_at_jw(omega)
    raise ValueError(f"unknown candidate: {candidate_name}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_contract(data: dict[str, Any]) -> None:
    require(
        data["schema"] == "luminous.airroad.mimo_holdout_falsification.v0_1",
        "unsupported falsification schema",
    )
    require(data["schema_version"] == "0.1.0", "unsupported schema version")
    require(
        data["evidence_class"]
        == "synthetic_exact_rational_hard_set_holdout_falsification",
        "unexpected evidence class",
    )
    require(
        data["claim_class"] == "deterministic_hard_set_model_invalidation",
        "unexpected claim class",
    )

    passport = data["acquisition_passport"]
    for key in IDENTITY_KEYS:
        require(passport.get(key) == data.get(key), f"stale acquisition passport: {key}")

    training = data["training"]
    holdout = data["holdout"]
    require(training["used_for_model_selection"] is True, "training use is not declared")
    require(training["holdout_used_for_fit"] is False, "holdout leaked into model fitting")
    require(holdout["used_for_fit"] is False, "holdout was used for fitting")
    require(
        holdout["selection_status"] == "preregistered_before_observation",
        "holdout selection was not preregistered",
    )
    require(holdout["selected_before_observation"] is True, "holdout was selected post hoc")
    require(
        q(holdout["preregistered_frequency"])
        == q(passport["holdout_frequency_pre_registered"])
        == Fraction(holdout["frequency"]),
        "holdout frequency does not match its preregistration",
    )

    training_ids = training["observation_ids"]
    training_error_ids = [item["measurement_error_id"]
                          for item in training["crossport_observations"]]
    require(len(training_ids) == len(set(training_ids)), "duplicate training observation IDs")
    require(
        len(training_error_ids) == len(set(training_error_ids)),
        "training measurement-error IDs are not independent",
    )
    require(
        len(training_ids) == len(training_error_ids) == len(training["frequencies"]),
        "training frequencies, observations and IDs do not align",
    )
    for index, observation in enumerate(training["crossport_observations"]):
        require(
            observation["observation_id"] == training_ids[index],
            "training observation identity mismatch",
        )
        require(
            Fraction(observation["frequency"]) == Fraction(training["frequencies"][index]),
            "training observation frequency mismatch",
        )
        require(
            observation["measurement_error_id"] == training_error_ids[index],
            "training measurement-error identity mismatch",
        )
    expected_training_matrix = [
        [(Fraction(2), Fraction(0)), (Fraction(0), Fraction(0))],
        [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0))],
    ]
    require(
        len(training["nominal_matrices"]) == len(training["frequencies"]),
        "training matrix/frequency count mismatch",
    )
    for encoded_matrix in training["nominal_matrices"]:
        decoded = [
            [(q(cell["numerator"]), q(cell["denominator"])) for cell in row]
            for row in encoded_matrix
        ]
        require(decoded == expected_training_matrix, "training matrix does not match frozen observations")
    require(
        holdout["observation_id"] not in set(training_ids),
        "holdout reuses a training observation ID",
    )
    require(
        holdout["holdout_sensor_error_id"] not in set(training_error_ids),
        "holdout sensor-error ID is reused from training",
    )
    require(
        holdout["independent_observation_id"] is True,
        "holdout independence is not established",
    )

    for section_name, binding in (
        ("training", training["identity_binding"]),
        ("holdout", holdout["identity_binding"]),
    ):
        for key in IDENTITY_KEYS:
            require(
                binding.get(key) == passport.get(key),
                f"{section_name} identity mismatch: {key}",
            )
    require(
        holdout["observed_port_order"] == passport["port_order"],
        "holdout port order is missing or mismatched",
    )
    require(
        holdout["port_element"] == [1, 2],
        "holdout does not measure the preregistered cross-port element",
    )
    require(
        data["calibration"]["scope"]
        == "shared_across_all_training_and_holdout_frequencies",
        "calibration nuisance is not campaign-shared",
    )
    require(
        data["calibration"]["evidence_class"] == "deterministic_hard_interval",
        "calibration is not declared as a deterministic hard interval",
    )

    for name, candidate in data["candidates"].items():
        require(
            candidate.get("model_pair_identity") == data["model_identity"],
            f"candidate {name} is bound to a stale model-pair identity",
        )

    training_error = training["measurement_error"]
    holdout_error = data["measurement_error"]
    require(
        training_error["semantics"] == "deterministic_hard_componentwise_bound",
        "training measurement error is not a deterministic hard set",
    )
    require(
        holdout_error["semantics"] == "deterministic_hard_componentwise_bound",
        "statistical covariance cannot be used as a deterministic hard error set",
    )

    # These frozen bounds must be unchanged from their preregistered values.
    for current, frozen in (
        ("real_lower", "preregistered_real_lower"),
        ("real_upper", "preregistered_real_upper"),
        ("imaginary_lower", "preregistered_imaginary_lower"),
        ("imaginary_upper", "preregistered_imaginary_upper"),
    ):
        require(
            q(holdout_error[current]) == q(holdout_error[frozen]),
            f"retrospective measurement-error inflation detected: {current}",
        )
    require(
        holdout_error["pre_registered_before_holdout"] is True,
        "measurement-error bounds were not frozen before holdout",
    )

    interval(data["calibration"]["lower"], data["calibration"]["upper"])
    interval(training_error["real_lower"], training_error["real_upper"])
    interval(training_error["imaginary_lower"], training_error["imaginary_upper"])
    interval(holdout_error["real_lower"], holdout_error["real_upper"])
    interval(holdout_error["imaginary_lower"], holdout_error["imaginary_upper"])


def training_calibration_interval(
    data: dict[str, Any],
    candidate_name: str,
) -> Interval | None:
    calibration_prior = interval(
        data["calibration"]["lower"],
        data["calibration"]["upper"],
    )
    training_error = data["training"]["measurement_error"]
    training_real_error = interval(
        training_error["real_lower"],
        training_error["real_upper"],
    )
    training_imag_error = interval(
        training_error["imaginary_lower"],
        training_error["imaginary_upper"],
    )
    feasible = calibration_prior

    for observation in data["training"]["crossport_observations"]:
        omega = Fraction(observation["frequency"])
        observed = read_complex(observation["value"])
        predicted = candidate_prediction(candidate_name, omega)

        # y = model + shared calibration + bounded sensor error.
        # Therefore xi lies in [y-model-eta_hi, y-model-eta_lo].
        xi_from_observation = (
            observed[0] - predicted[0] - training_real_error[1],
            observed[0] - predicted[0] - training_real_error[0],
        )
        next_feasible = intersect(feasible, xi_from_observation)
        if next_feasible is None:
            return None
        feasible = next_feasible

        imaginary_residual = observed[1] - predicted[1]
        if not contains(training_imag_error, imaginary_residual):
            return None
    return feasible


def evaluate_candidate(
    data: dict[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    holdout = data["holdout"]
    measurement_error = data["measurement_error"]
    observed = read_complex(holdout["observed_value"])
    omega = Fraction(holdout["frequency"])
    predicted = candidate_prediction(candidate_name, omega)

    training_xi = training_calibration_interval(data, candidate_name)
    if training_xi is None:
        return {"status": "FALSIFIED_BY_TRAINING", "reason": "training set is inconsistent"}

    real_error = interval(
        measurement_error["real_lower"],
        measurement_error["real_upper"],
    )
    imaginary_error = interval(
        measurement_error["imaginary_lower"],
        measurement_error["imaginary_upper"],
    )
    real_delta = observed[0] - predicted[0]
    imaginary_residual = observed[1] - predicted[1]

    # The one calibration coordinate must satisfy training and holdout together.
    xi_holdout = (
        real_delta - real_error[1],
        real_delta - real_error[0],
    )
    xi_joint = intersect(training_xi, xi_holdout)
    imaginary_pass = contains(imaginary_error, imaginary_residual)
    falsified = xi_joint is None or not imaginary_pass

    result: dict[str, Any] = {
        "status": "FALSIFIED" if falsified else "NOT_FALSIFIED_BY_THIS_HOLDOUT",
        "prediction": predicted,
        "observation": observed,
        "training_shared_calibration_interval": training_xi,
        "holdout_required_calibration_interval": xi_holdout,
        "joint_shared_calibration_interval": xi_joint,
        "imaginary_residual": imaginary_residual,
        "imaginary_measurement_error_pass": imaginary_pass,
    }
    if falsified:
        real_required = distance_to_interval(real_delta, training_xi)
        imaginary_required = distance_to_interval(
            imaginary_residual,
            (Fraction(0), Fraction(0)),
        )
        result["minimum_real_error_halfwidth_to_rescue"] = real_required
        result["minimum_imaginary_error_halfwidth_to_rescue"] = imaginary_required
        result["minimum_uniform_componentwise_error_halfwidth_to_rescue"] = max(
            real_required,
            imaginary_required,
        )
    return result


def validate_terminal_claim(status: str, requested_claim: str) -> None:
    if status == "NOT_FALSIFIED_BY_THIS_HOLDOUT" and requested_claim in {
        "VERIFIED_TRUE",
        "CONTINUOUS_BAND_ADEQUATE",
        "ROBUSTLY_STABLE",
    }:
        raise ValueError("a surviving holdout cannot prove truth or continuous adequacy")
    if status == "FALSIFIED" and requested_claim in {
        "CANDIDATE_B_VERIFIED_TRUE",
        "ALTERNATIVE_PROVEN_TRUE",
    }:
        raise ValueError("falsifying one candidate does not prove another candidate")


def run_negative_controls(data: dict[str, Any]) -> None:
    # A holdout placed on a training zero cannot distinguish the model pair.
    on_training_knot = json.loads(json.dumps(data))
    on_training_knot["holdout"]["frequency"] = 1
    on_training_knot["holdout"]["preregistered_frequency"] = {
        "numerator": "1", "denominator": "1"
    }
    on_training_knot["acquisition_passport"]["holdout_frequency_pre_registered"] = {
        "numerator": "1", "denominator": "1"
    }
    zero = delta_at_jw(Fraction(1))
    on_training_knot["holdout"]["observed_value"] = {
        "re": {"numerator": str(zero[0].numerator), "denominator": str(zero[0].denominator)},
        "im": {"numerator": str(zero[1].numerator), "denominator": str(zero[1].denominator)},
    }
    candidate_a_on_knot = evaluate_candidate(on_training_knot, "ZA")
    assert candidate_a_on_knot["status"] == "NOT_FALSIFIED_BY_THIS_HOLDOUT"

    controls: list[tuple[str, dict[str, Any]]] = []

    stale_model = json.loads(json.dumps(data))
    stale_model["candidates"]["ZB"]["model_pair_identity"] = "sha256:stale-model-pair"
    controls.append(("stale model-pair identity", stale_model))

    split_calibration = json.loads(json.dumps(data))
    split_calibration["holdout"]["identity_binding"]["calibration_nuisance_id"] = (
        "different-holdout-calibration"
    )
    controls.append(("split calibration nuisance", split_calibration))

    leaked_id = json.loads(json.dumps(data))
    leaked_id["holdout"]["observation_id"] = leaked_id["training"]["observation_ids"][0]
    controls.append(("training/holdout observation leakage", leaked_id))

    post_hoc = json.loads(json.dumps(data))
    post_hoc["holdout"]["selected_before_observation"] = False
    controls.append(("post-hoc holdout selection", post_hoc))

    retrospective = json.loads(json.dumps(data))
    retrospective["measurement_error"]["real_upper"] = {
        "numerator": "1", "denominator": "1"
    }
    controls.append(("retrospective uncertainty inflation", retrospective))

    statistical_as_hard = json.loads(json.dumps(data))
    statistical_as_hard["measurement_error"]["semantics"] = "statistical_covariance"
    controls.append(("statistical covariance promoted to hard set", statistical_as_hard))

    missing_port_order = json.loads(json.dumps(data))
    missing_port_order["holdout"]["observed_port_order"] = ["input-b", "input-a"]
    controls.append(("port-order mismatch", missing_port_order))

    for label, mutated in controls:
        try:
            validate_contract(mutated)
        except (KeyError, TypeError, ValueError):
            continue
        raise AssertionError(f"negative control accepted: {label}")

    for requested in (
        "VERIFIED_TRUE",
        "CONTINUOUS_BAND_ADEQUATE",
        "ROBUSTLY_STABLE",
    ):
        try:
            validate_terminal_claim("NOT_FALSIFIED_BY_THIS_HOLDOUT", requested)
        except ValueError:
            pass
        else:
            raise AssertionError(f"claim promotion accepted: {requested}")

    try:
        validate_terminal_claim("FALSIFIED", "ALTERNATIVE_PROVEN_TRUE")
    except ValueError:
        pass
    else:
        raise AssertionError("surviving alternative was promoted to proven true")


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    validate_contract(data)

    # Training frequencies are exact zeros of the hidden cross-port term.
    for frequency in data["training"]["frequencies"]:
        value = delta_at_jw(Fraction(frequency))
        assert value == (Fraction(0), Fraction(0))

    expected_holdout = read_complex(data["expected"]["delta_at_holdout"])
    actual_holdout = delta_at_jw(Fraction(data["holdout"]["frequency"]))
    assert actual_holdout == expected_holdout
    assert actual_holdout == read_complex(data["holdout"]["observed_value"])
    assert actual_holdout[0] == Fraction(-401712500, 410338673)
    assert actual_holdout[1] == Fraction(299932500, 410338673)

    candidate_a = evaluate_candidate(data, "ZA")
    candidate_b = evaluate_candidate(data, "ZB")
    assert candidate_a["status"] == data["expected"]["candidate_A_status"]
    assert candidate_b["status"] == data["expected"]["candidate_B_status"]
    assert candidate_a["status"] == "FALSIFIED"
    assert candidate_b["status"] == "NOT_FALSIFIED_BY_THIS_HOLDOUT"

    calibration_prior = interval(
        data["calibration"]["lower"],
        data["calibration"]["upper"],
    )
    error = data["measurement_error"]
    real_sensor = interval(error["real_lower"], error["real_upper"])
    combined_real = (
        calibration_prior[0] + real_sensor[0],
        calibration_prior[1] + real_sensor[1],
    )
    assert combined_real == (
        q(data["expected"]["candidate_A_real_admissible_lower"]),
        q(data["expected"]["candidate_A_real_admissible_upper"]),
    )
    assert not contains(combined_real, actual_holdout[0])

    required_real = candidate_a["minimum_real_error_halfwidth_to_rescue"]
    assert required_real == q(data["expected"]["min_measurement_error_halfwidth_to_rescue"])
    assert required_real > Fraction(92, 100)
    assert candidate_b["joint_shared_calibration_interval"] is not None
    assert candidate_b["imaginary_measurement_error_pass"] is True

    run_negative_controls(data)

    print("AIRROAD exact MIMO holdout falsification reference: PASS")
    print(f"Delta(3j): {actual_holdout}")
    print(
        "candidate A: "
        f"{candidate_a['status']}; shared calibration intersection="
        f"{candidate_a['joint_shared_calibration_interval']}"
    )
    print(
        "candidate B: "
        f"{candidate_b['status']}; shared calibration intersection="
        f"{candidate_b['joint_shared_calibration_interval']}"
    )
    print(f"candidate A minimum real error halfwidth to rescue: {required_real}")
    print("training/holdout provenance and shared calibration controls: PASS")
    print("post-hoc selection, error inflation, port and claim-promotion controls: PASS")
    print("surviving candidate truth and continuous-band adequacy: NOT ESTABLISHED")


if __name__ == "__main__":
    main()
