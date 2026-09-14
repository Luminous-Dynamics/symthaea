#!/usr/bin/env python3
"""Independent target-isolated synthetic analysis oracle for LQCD-021F."""

import hashlib
import json
import math

ORACLE_ID = "lqcd_beta6_synthetic_analysis_oracle_v1"
CONFIGS = 64
BLOCK_SIZE = 8
BLOCKS = CONFIGS // BLOCK_SIZE
T_VALUES = list(range(1, 9))
VECTORS = [
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (2, 0, 0),
    (2, 1, 0),
    (2, 2, 0),
    (3, 0, 0),
    (3, 1, 0),
]
RS = [math.sqrt(sum(x * x for x in v)) for v in VECTORS]
COULOMB_DELTA = [0.055, -0.020, 0.018, 0.012, -0.010, 0.006, 0.004, -0.003]
COULOMB = [1.0 / r + d for r, d in zip(RS, COULOMB_DELTA)]
FIT_INDICES = list(range(6))

TRUTH = {
    "V0": 0.61,
    "sigma": 0.052,
    "e": 0.2617993877991494,
    "l": 0.035,
}

def centered_scaled(values):
    mean = sum(values) / len(values)
    centered = [v - mean for v in values]
    rms = math.sqrt(sum(v * v for v in centered) / len(centered))
    return [v / rms for v in centered]

def deterministic_modes():
    drive = [(((73 * (i + 1) + 19) % 1009) / 1009.0) - 0.5 for i in range(CONFIGS)]
    z = []
    state = 0.0
    for value in drive:
        state = 0.78 * state + value
        z.append(state)
    z = centered_scaled(z)

    q = centered_scaled(
        [(((211 * (i + 1) + 37) % 1013) / 1013.0) - 0.5 for i in range(CONFIGS)]
    )

    r_modes = []
    for j in range(len(RS)):
        modulus = 1019 + 2 * j
        multiplier = 31 + 17 * j
        offset = 7 + 13 * j
        r_modes.append(
            centered_scaled(
                [
                    (((multiplier * (i + 1) + offset) % modulus) / modulus) - 0.5
                    for i in range(CONFIGS)
                ]
            )
        )
    return z, q, r_modes

def potential_value(r, coulomb, params):
    return (
        params["V0"]
        + params["sigma"] * r
        - params["e"] * coulomb
        + params["l"] * (coulomb - 1.0 / r)
    )

def synthetic_data():
    z, q, r_modes = deterministic_modes()
    truth_v = [potential_value(r, c, TRUTH) for r, c in zip(RS, COULOMB)]
    data = []
    for i in range(CONFIGS):
        config = []
        for j, (r, v_ground) in enumerate(zip(RS, truth_v)):
            by_t = []
            excited_amp = 0.18 + 0.015 * r
            excited_gap = 0.85
            for t in T_VALUES:
                mean_w = math.exp(-v_ground * t) * (
                    1.0 + excited_amp * math.exp(-excited_gap * t)
                )
                correlated_noise = (
                    0.014 * (0.72 * z[i] + 0.20 * q[i] * (j + 1) / len(RS))
                    + 0.006 * r_modes[j][i]
                ) * (1.0 + 0.03 * t)
                by_t.append(
                    [
                        mean_w * (1.0 + correlated_noise + 0.0025),
                        mean_w * (1.0 + correlated_noise - 0.0025),
                    ]
                )
            config.append(by_t)
        data.append(config)
    lag1 = sum(z[i] * z[i + 1] for i in range(CONFIGS - 1)) / sum(
        z[i] * z[i] for i in range(CONFIGS - 1)
    )
    return data, truth_v, lag1

def orientation_mean(orientations):
    if len(orientations) != 2:
        raise ValueError("exactly two synthetic orientations required")
    return sum(orientations) / 2.0

def mean_w(data, config_indices, r_index, t_index):
    per_config = []
    for i in config_indices:
        per_config.append(orientation_mean(data[i][r_index][t_index]))
    return sum(per_config) / len(per_config)

def extract_plateau(data, config_indices, start_t, end_t):
    values = []
    for r_index in range(len(RS)):
        ws = [mean_w(data, config_indices, r_index, t) for t in range(len(T_VALUES))]
        if any(w <= 0.0 or not math.isfinite(w) for w in ws):
            raise ValueError("Wilson-loop mean must be positive finite")
        effective = [math.log(ws[t] / ws[t + 1]) for t in range(len(ws) - 1)]
        selected = [effective[t - 1] for t in range(start_t, end_t + 1)]
        values.append(sum(selected) / len(selected))
    return values

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def matmul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]

def matvec(a, v):
    return [sum(x * y for x, y in zip(row, v)) for row in a]

def invert(matrix):
    n = len(matrix)
    work = [
        matrix[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1.0e-20:
            raise ValueError("singular matrix")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [x / scale for x in work[column]]
        for row in range(n):
            if row == column:
                continue
            factor = work[row][column]
            work[row] = [
                work[row][k] - factor * work[column][k]
                for k in range(2 * n)
            ]
    return [row[n:] for row in work]

def jackknife_covariance(replicates):
    count = len(replicates)
    width = len(replicates[0])
    means = [sum(rep[j] for rep in replicates) / count for j in range(width)]
    factor = (count - 1) / count
    covariance = []
    for a in range(width):
        row = []
        for b in range(width):
            row.append(
                factor
                * sum(
                    (rep[a] - means[a]) * (rep[b] - means[b])
                    for rep in replicates
                )
            )
        covariance.append(row)
    return means, covariance

def design_row(index, family):
    r = RS[index]
    coulomb = COULOMB[index]
    if family == "free_v0_sigma_e_l_v1":
        return [1.0, r, -coulomb, coulomb - 1.0 / r], 0.0
    if family == "fixed_e_pi_over_12_free_l_v1":
        return [1.0, r, coulomb - 1.0 / r], -TRUTH["e"] * coulomb
    if family == "fixed_e_pi_over_12_l0_v1":
        return [1.0, r], -TRUTH["e"] * coulomb
    raise ValueError("unknown fit family")

def correlated_fit(values, covariance_inverse, family):
    rows = []
    offsets = []
    y = []
    for index in FIT_INDICES:
        row, offset = design_row(index, family)
        rows.append(row)
        offsets.append(offset)
        y.append(values[index] - offset)

    xt = transpose(rows)
    xt_cinv = matmul(xt, covariance_inverse)
    normal = matmul(xt_cinv, rows)
    normal_inverse = invert(normal)
    beta = matvec(normal_inverse, matvec(xt_cinv, y))

    predicted = []
    for row, offset in zip(rows, offsets):
        predicted.append(sum(a * b for a, b in zip(row, beta)) + offset)
    residual = [values[i] - p for i, p in zip(FIT_INDICES, predicted)]
    chi2 = sum(
        residual[i] * covariance_inverse[i][j] * residual[j]
        for i in range(len(residual))
        for j in range(len(residual))
    )

    if family == "free_v0_sigma_e_l_v1":
        params = {"V0": beta[0], "sigma": beta[1], "e": beta[2], "l": beta[3]}
    elif family == "fixed_e_pi_over_12_free_l_v1":
        params = {"V0": beta[0], "sigma": beta[1], "e": TRUTH["e"], "l": beta[2]}
    else:
        params = {"V0": beta[0], "sigma": beta[1], "e": TRUTH["e"], "l": 0.0}
    return params, chi2

def sommer(params, c):
    argument = (c - params["e"]) / params["sigma"]
    if argument <= 0.0:
        raise ValueError("invalid Sommer-scale argument")
    return math.sqrt(argument)

def jackknife_error(values):
    mean = sum(values) / len(values)
    return math.sqrt(
        (len(values) - 1)
        / len(values)
        * sum((value - mean) ** 2 for value in values)
    )

def q(value):
    return round(value, 12)

def sealed_digest(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b"symthaea.lqcd.synthetic-sealed-analysis.v1\x00" + encoded).hexdigest()

def compare(sealed, benchmark):
    return {
        "sealed_digest": sealed["sealed_digest"],
        "benchmark_id": benchmark["id"],
        "delta_a_sqrt_sigma": q(
            sealed["estimate"]["a_sqrt_sigma"] - benchmark["a_sqrt_sigma"]
        ),
        "delta_r0": q(sealed["estimate"]["r0"] - benchmark["r0"]),
    }

def main():
    data, truth_v, lag1 = synthetic_data()
    full = extract_plateau(data, list(range(CONFIGS)), 5, 7)

    replicates = []
    for block in range(BLOCKS):
        included = [
            i
            for i in range(CONFIGS)
            if not (block * BLOCK_SIZE <= i < (block + 1) * BLOCK_SIZE)
        ]
        replicates.append(extract_plateau(data, included, 5, 7))

    _, covariance = jackknife_covariance(replicates)
    fit_covariance = [
        [covariance[i][j] for j in FIT_INDICES]
        for i in FIT_INDICES
    ]
    covariance_inverse = invert(fit_covariance)

    families = {}
    for family in (
        "free_v0_sigma_e_l_v1",
        "fixed_e_pi_over_12_free_l_v1",
        "fixed_e_pi_over_12_l0_v1",
    ):
        params, chi2 = correlated_fit(full, covariance_inverse, family)
        families[family] = {
            "params": {key: q(value) for key, value in params.items()},
            "chi2": q(chi2),
        }

    primary_family = "fixed_e_pi_over_12_free_l_v1"
    primary, primary_chi2 = correlated_fit(full, covariance_inverse, primary_family)
    replicate_params = [
        correlated_fit(rep, covariance_inverse, primary_family)[0]
        for rep in replicates
    ]

    cs = [1.65, 4.0, 6.0]
    rc = [sommer(primary, c) for c in cs]
    rc_replicates = [[sommer(params, c) for c in cs] for params in replicate_params]
    rc_errors = [
        jackknife_error([rep[index] for rep in rc_replicates])
        for index in range(len(cs))
    ]

    early = extract_plateau(data, list(range(CONFIGS)), 1, 3)
    early_params, _ = correlated_fit(early, covariance_inverse, primary_family)

    assert abs(primary["sigma"] - TRUTH["sigma"]) < 1.0e-4
    assert abs(primary["l"] - TRUTH["l"]) < 1.0e-5
    assert abs(families["free_v0_sigma_e_l_v1"]["params"]["e"] - TRUTH["e"]) < 1.0e-6
    assert abs(early_params["sigma"] - TRUTH["sigma"]) > 1.0e-3
    assert families["fixed_e_pi_over_12_l0_v1"]["chi2"] > 100.0
    assert lag1 > 0.5

    truth_rc = [math.sqrt((c - TRUTH["e"]) / TRUTH["sigma"]) for c in cs]
    for estimate_value, expected in zip(rc, truth_rc):
        assert abs(estimate_value - expected) < 0.01

    estimate = {
        "sigma": q(primary["sigma"]),
        "a_sqrt_sigma": q(math.sqrt(primary["sigma"])),
        "e": q(primary["e"]),
        "l": q(primary["l"]),
        "r0": q(rc[0]),
        "r4": q(rc[1]),
        "r6": q(rc[2]),
        "r0_jackknife_se": q(rc_errors[0]),
        "r4_jackknife_se": q(rc_errors[1]),
        "r6_jackknife_se": q(rc_errors[2]),
    }

    sealed = {
        "oracle_id": ORACLE_ID,
        "configuration_count": CONFIGS,
        "orientation_count_per_configuration": 2,
        "block_size": BLOCK_SIZE,
        "block_count": BLOCKS,
        "independent_resampling_unit": "configuration_block",
        "plateau_effective_t": [5, 6, 7],
        "fit_indices": FIT_INDICES,
        "fit_vectors": [list(VECTORS[i]) for i in FIT_INDICES],
        "primary_family": primary_family,
        "estimate": estimate,
        "families": families,
    }
    sealed["sealed_digest"] = sealed_digest(sealed)

    benchmark_a = {"id": "synthetic-comparator-a", "a_sqrt_sigma": 0.24, "r0": 5.0}
    benchmark_b = {"id": "synthetic-comparator-b", "a_sqrt_sigma": 0.19, "r0": 5.8}
    comparison_a = compare(sealed, benchmark_a)
    comparison_b = compare(sealed, benchmark_b)
    assert comparison_a["sealed_digest"] == comparison_b["sealed_digest"] == sealed["sealed_digest"]

    try:
        orientation_mean([1.0, 1.0, 1.0])
    except ValueError:
        duplicate_orientation_rejected = True
    else:
        raise AssertionError("duplicate orientations were accepted")

    result = {
        "oracle_id": ORACLE_ID,
        "truth": {
            "sigma": TRUTH["sigma"],
            "e": TRUTH["e"],
            "l": TRUTH["l"],
            "r0": q(truth_rc[0]),
            "r4": q(truth_rc[1]),
            "r6": q(truth_rc[2]),
        },
        "configuration_count": CONFIGS,
        "block_count": BLOCKS,
        "lag1_mode_correlation": q(lag1),
        "late_plateau_sigma": q(primary["sigma"]),
        "early_plateau_sigma": q(early_params["sigma"]),
        "excited_state_rejection_detected": True,
        "primary_family": primary_family,
        "family_results": families,
        "estimate": estimate,
        "sealed_digest": sealed["sealed_digest"],
        "benchmark_change_preserves_sealed_digest": True,
        "comparison_a": comparison_a,
        "comparison_b": comparison_b,
        "orientation_duplicates_rejected": duplicate_orientation_rejected,
        "scientific_boundary": (
            "synthetic_analysis_semantics_only_no_real_beta6_ensemble_no_ehk_agreement"
        ),
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    print("ok")
    print("result_sha256=" + hashlib.sha256(canonical.encode()).hexdigest())
    print(canonical)

if __name__ == "__main__":
    main()
