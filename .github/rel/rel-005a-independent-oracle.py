#!/usr/bin/env python3
"""REL-005A independent scalar oracle.

Fault-domain boundary:
- Python stdlib only; intended to run with ``python3 -I``.
- No Symthaea imports or production HDC helpers.
- The frozen fixture is duplicated as scalar constants.
- f64 predictions were preregistered on #3090 before the Rust theorem ran.

Authority: PredictionOnly. A matching production artifact supports qualification;
it does not turn this oracle into a production implementation or a physics claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

EPSILON = 1.0e-7
ZERO_ORACLE_TOL = 1.0e-12
PRODUCTION_ZERO_TOL = 1.0e-5
PRODUCTION_NONZERO_TOL = 5.0e-5

Vec = list[float]
Matrix = list[Vec]


def bind(a: Sequence[float], b: Sequence[float]) -> Vec:
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    return [x * y for x, y in zip(a, b)]


def inverse(a: Sequence[float]) -> Vec:
    return [0.0 if abs(x) < EPSILON else 1.0 / x for x in a]


def bundle(rows: Iterable[Sequence[float]]) -> Vec:
    data = [list(row) for row in rows]
    if not data:
        raise ValueError("empty bundle")
    dim = len(data[0])
    if any(len(row) != dim for row in data):
        raise ValueError("dimension mismatch")
    n = float(len(data))
    return [sum(row[i] for row in data) / n for i in range(dim)]


def fit(native: Sequence[Sequence[float]], foreign: Sequence[Sequence[float]]) -> Vec:
    if not native or len(native) != len(foreign):
        raise ValueError("invalid anchor cardinality")
    return bundle(bind(n, inverse(f)) for n, f in zip(native, foreign))


def max_abs(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    return max(abs(x - y) for x, y in zip(a, b))


def rms(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    aa = sum(x * x for x in a)
    bb = sum(y * y for y in b)
    return dot / math.sqrt(aa * bb)


def reverse(a: Sequence[float]) -> Vec:
    return list(reversed(a))


def construction_error(source: Matrix, target: Matrix, transform: Vec) -> tuple[float, float]:
    residuals: Vec = []
    maximum = 0.0
    for src, expected in zip(source, target):
        actual = bind(src, transform)
        maximum = max(maximum, max_abs(actual, expected))
        residuals.extend(x - y for x, y in zip(actual, expected))
    return maximum, math.sqrt(sum(x * x for x in residuals) / len(residuals))


def dispersion(target: Matrix, source: Matrix, center: Vec) -> tuple[float, float]:
    masks = [bind(t, inverse(s)) for t, s in zip(target, source)]
    residuals: Vec = []
    maximum = 0.0
    for mask in masks:
        maximum = max(maximum, max_abs(mask, center))
        residuals.extend(x - y for x, y in zip(mask, center))
    return maximum, math.sqrt(sum(x * x for x in residuals) / len(residuals))


def build_prediction() -> dict[str, object]:
    mask_ab = [0.5, -0.75, 0.9, -0.85, 0.8, -0.7, 0.95, -0.6]
    mask_bc = [-0.8, 0.7, -0.9, 0.65, 0.6, 0.75, -0.85, 0.9]
    anchors_a: Matrix = [
        [0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -0.95],
        [-0.9, 0.8, -0.7, 0.6, -0.5, 0.4, -0.85, 0.75],
        [0.95, -1.0, 0.9, -0.8, 0.7, -0.6, 0.5, -0.4],
        [-0.35, 0.45, -0.55, 0.65, -0.75, 0.85, -0.95, 1.0],
    ]
    anchors_b = [bind(anchor, mask_ab) for anchor in anchors_a]
    anchors_c = [bind(anchor, mask_bc) for anchor in anchors_b]

    estimated_ab = fit(anchors_b, anchors_a)
    estimated_bc = fit(anchors_c, anchors_b)
    estimated_ac = fit(anchors_c, anchors_a)
    estimated_ca = fit(anchors_a, anchors_c)

    shared_max, shared_rms = dispersion(anchors_b, anchors_a, estimated_ab)
    construction_max, construction_rms = construction_error(anchors_a, anchors_b, estimated_ab)

    held = [0.55, -0.65, 0.75, -0.85, 0.95, -1.0, 0.9, -0.8]
    expected_held_b = bind(held, mask_ab)
    actual_held_b = bind(held, estimated_ab)

    direct = bind(held, estimated_ac)
    sequential = bind(bind(held, estimated_ab), estimated_bc)
    loop = bind(bind(bind(held, estimated_ab), estimated_bc), estimated_ca)

    x = [1.0, -0.7, -0.7, 0.8, -0.5, 0.4, 0.9, -0.7]
    y = [-0.6, -0.8, -0.4, 1.0, 0.5, -0.7, 0.4, 1.0]
    fx = bind(x, estimated_ab)
    fy = bind(y, estimated_ab)
    source_bundle = bundle([x, y])
    target_bundle = bundle([fx, fy])
    transported_source_bind = bind(bind(x, y), estimated_ab)
    fixed_target_bind = bind(fx, fy)
    dressed_target_bind = bind(fixed_target_bind, inverse(estimated_ab))
    transported_unit = bind([1.0] * 8, estimated_ab)
    dressed_unit = bind(bind(transported_unit, fx), inverse(estimated_ab))
    source_cosine = cosine(x, y)
    raw_target_cosine = cosine(fx, fy)
    pulled_x = bind(fx, inverse(estimated_ab))
    pulled_y = bind(fy, inverse(estimated_ab))

    shuffled_a = [anchors_a[i] for i in (1, 2, 3, 0)]
    shuffled_transform = fit(anchors_b, shuffled_a)
    shuffled_dispersion, _ = dispersion(anchors_b, shuffled_a, shuffled_transform)

    mixed_targets = [
        bind(anchor, mask_ab if index < 2 else mask_bc)
        for index, anchor in enumerate(anchors_a)
    ]
    mixed_transform = fit(mixed_targets, anchors_a)
    mixed_dispersion, _ = dispersion(mixed_targets, anchors_a, mixed_transform)

    corrupted_targets = [row[:] for row in anchors_b]
    corrupted_targets[0][0] += 1.5
    corrupted_transform = fit(corrupted_targets, anchors_a)
    corrupted_dispersion, _ = dispersion(corrupted_targets, anchors_a, corrupted_transform)

    near_floor_source = [5.0e-8, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.0]
    near_floor_target = bind(near_floor_source, mask_ab)
    near_floor_transform = fit([near_floor_target], [near_floor_source])

    permuted_targets = [reverse(anchor) for anchor in anchors_a]
    permutation_transform = fit(permuted_targets, anchors_a)

    dimension_mismatch_rejected = False
    try:
        bind([1.0], [1.0, 2.0])
    except ValueError:
        dimension_mismatch_rejected = True
    nan_transform = fit([[math.nan, 0.5]], [[math.nan, 0.5]])

    return {
        "schema": "symthaea.rel.graft-frame-independent-oracle.v1",
        "authority": "PredictionOnly",
        "positive_fixture_within_nominal_range": True,
        "positive_inverse_floor_affected_count": 0,
        "mask_recovery_max": max_abs(estimated_ab, mask_ab),
        "mask_recovery_rms": rms(estimated_ab, mask_ab),
        "shared_mask_dispersion_max": shared_max,
        "shared_mask_dispersion_rms": shared_rms,
        "construction_anchor_max": construction_max,
        "construction_anchor_rms": construction_rms,
        "held_out_max": max_abs(actual_held_b, expected_held_b),
        "held_out_rms": rms(actual_held_b, expected_held_b),
        "direct_vs_composed_mask_max": max_abs(estimated_ac, bind(estimated_ab, estimated_bc)),
        "direct_vs_sequential_transport_max": max_abs(direct, sequential),
        "loop_closure_max": max_abs(loop, held),
        "loop_closure_rms": rms(loop, held),
        "bundle_covariance_defect": max_abs(bind(source_bundle, estimated_ab), target_bundle),
        "transported_unit_defect": max_abs(transported_unit, estimated_ab),
        "dressed_unit_action_defect": max_abs(dressed_unit, fx),
        "fixed_hadamard_covariance_defect": max_abs(transported_source_bind, fixed_target_bind),
        "dressed_hadamard_covariance_defect": max_abs(transported_source_bind, dressed_target_bind),
        "raw_cosine_drift": abs(source_cosine - raw_target_cosine),
        "pulled_back_cosine_drift": abs(source_cosine - cosine(pulled_x, pulled_y)),
        "shuffled_anchor_held_out_max": max_abs(bind(held, shuffled_transform), expected_held_b),
        "shuffled_mask_dispersion_max": shuffled_dispersion,
        "mixed_mask_held_out_max": max_abs(bind(held, mixed_transform), expected_held_b),
        "mixed_mask_dispersion_max": mixed_dispersion,
        "corrupted_anchor_held_out_max": max_abs(bind(held, corrupted_transform), expected_held_b),
        "corrupted_mask_dispersion_max": corrupted_dispersion,
        "near_floor_held_out_max": max_abs(bind(held, near_floor_transform), expected_held_b),
        "inverse_floor_affected_count": 1,
        "inverse_floor_affected_rate": 0.125,
        "permutation_held_out_max": max_abs(bind(held, permutation_transform), reverse(held)),
        "dimension_mismatch_panics": dimension_mismatch_rejected,
        "nonfinite_input_returns_nonfinite_transform": any(not math.isfinite(v) for v in nan_transform),
        "claims": {
            "fixed_hadamard_invariance_established": False,
            "fixed_cosine_invariance_established": False,
            "general_coordinate_transform_established": False,
            "physical_gauge_symmetry_established": False,
            "consciousness_claim_established": False,
        },
    }


ZERO_FIELDS = {
    "mask_recovery_max",
    "mask_recovery_rms",
    "shared_mask_dispersion_max",
    "shared_mask_dispersion_rms",
    "construction_anchor_max",
    "construction_anchor_rms",
    "held_out_max",
    "held_out_rms",
    "direct_vs_composed_mask_max",
    "direct_vs_sequential_transport_max",
    "loop_closure_max",
    "loop_closure_rms",
    "bundle_covariance_defect",
    "transported_unit_defect",
    "dressed_unit_action_defect",
    "dressed_hadamard_covariance_defect",
    "pulled_back_cosine_drift",
}

NONZERO_EXPECTED = {
    "fixed_hadamard_covariance_defect": 1.258,
    "raw_cosine_drift": 0.2749268555700186,
    "shuffled_anchor_held_out_max": 1.785408088235294,
    "shuffled_mask_dispersion_max": 0.8140625,
    "mixed_mask_held_out_max": 0.81,
    "mixed_mask_dispersion_max": 0.9,
    "corrupted_anchor_held_out_max": 0.515625,
    "corrupted_mask_dispersion_max": 2.8125,
    "near_floor_held_out_max": 0.275,
    "inverse_floor_affected_rate": 0.125,
    "permutation_held_out_max": 0.39093137254901955,
}

EXACT_EXPECTED = {
    "positive_fixture_within_nominal_range": True,
    "positive_inverse_floor_affected_count": 0,
    "inverse_floor_affected_count": 1,
    "dimension_mismatch_panics": True,
    "nonfinite_input_returns_nonfinite_transform": True,
}


def self_check(prediction: dict[str, object]) -> None:
    for field in ZERO_FIELDS:
        value = float(prediction[field])
        if abs(value) > ZERO_ORACLE_TOL:
            raise AssertionError(f"{field}: expected algebraic zero, got {value}")
    for field, expected in NONZERO_EXPECTED.items():
        value = float(prediction[field])
        if abs(value - expected) > 1.0e-12:
            raise AssertionError(f"{field}: expected {expected}, got {value}")
    for field, expected in EXACT_EXPECTED.items():
        if prediction[field] != expected:
            raise AssertionError(f"{field}: expected {expected!r}, got {prediction[field]!r}")


def compare_production(prediction: dict[str, object], production: dict[str, object]) -> None:
    errors: list[str] = []
    for field in ZERO_FIELDS:
        value = float(production[field])
        if abs(value) > PRODUCTION_ZERO_TOL:
            errors.append(f"{field}: production zero-class defect {value} > {PRODUCTION_ZERO_TOL}")
    for field in NONZERO_EXPECTED:
        expected = float(prediction[field])
        actual = float(production[field])
        if abs(actual - expected) > PRODUCTION_NONZERO_TOL:
            errors.append(
                f"{field}: production={actual} oracle={expected} "
                f"delta={abs(actual-expected)} > {PRODUCTION_NONZERO_TOL}"
            )
    for field, expected in EXACT_EXPECTED.items():
        if production[field] != expected:
            errors.append(f"{field}: production={production[field]!r} expected={expected!r}")
    expected_claims = prediction["claims"]
    production_claims = production.get("claims")
    if production_claims != expected_claims:
        errors.append(f"claims mismatch: production={production_claims!r} oracle={expected_claims!r}")
    if errors:
        raise AssertionError("production/oracle disagreement:\n- " + "\n- ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--production", type=Path)
    args = parser.parse_args()

    prediction = build_prediction()
    self_check(prediction)
    encoded = json.dumps(prediction, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")

    if args.production:
        production = json.loads(args.production.read_text(encoding="utf-8"))
        compare_production(prediction, production)
        print("REL005A_INDEPENDENT_ORACLE_COMPARISON=PASS")


if __name__ == "__main__":
    main()
