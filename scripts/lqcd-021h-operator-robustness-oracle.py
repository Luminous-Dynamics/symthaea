#!/usr/bin/env python3
"""Independent paired operator-robustness oracle for LQCD-021H."""
import copy
import hashlib
import json
import math

ORACLE_ID = "lqcd_beta6_operator_robustness_oracle_v1"
CONFIGS = 64
BLOCK_SIZE = 8
BLOCKS = CONFIGS // BLOCK_SIZE
T_VALUES = list(range(1, 8))
LATE_T = (4, 6)
ABS_TOL = 0.00125
SIGMA_MULTIPLIER = 2.0

VECTORS = [
    {"id": "axis_100", "r": 1.0, "axis": True},
    {"id": "off_110", "r": math.sqrt(2.0), "axis": False},
    {"id": "off_111", "r": math.sqrt(3.0), "axis": False},
    {"id": "off_210", "r": math.sqrt(5.0), "axis": False},
]

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

def digest(tag, value):
    return hashlib.sha256(tag + canonical(value)).hexdigest()

def centered_scaled(values):
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    rms = math.sqrt(sum(value * value for value in centered) / len(centered))
    return [value / rms for value in centered]

def modes():
    z = centered_scaled([
        (((83 * (i + 1) + 17) % 1019) / 1019.0) - 0.5
        for i in range(CONFIGS)
    ])
    q = centered_scaled([
        (((197 * (i + 1) + 41) % 1021) / 1021.0) - 0.5
        for i in range(CONFIGS)
    ])
    return z, q

def ground_potential(r):
    return 0.58 + 0.055 * r - 0.26 / r

def synthetic_data(sensitivity=False):
    z, q = modes()
    rows = []
    for i in range(CONFIGS):
        vectors = {}
        for j, vector in enumerate(VECTORS):
            operators = {}
            for operator in ("exhaustive", "bresenham"):
                values = []
                shift = 0.0
                if (
                    sensitivity
                    and not vector["axis"]
                    and operator == "bresenham"
                    and vector["id"] == "off_111"
                ):
                    shift = 0.012
                for t in T_VALUES:
                    ground = (
                        ground_potential(vector["r"])
                        + 0.0012 * z[i]
                        + 0.0004 * (j + 1) * q[i]
                    )
                    amplitude = 0.16 + 0.015 * j
                    if not vector["axis"]:
                        if operator == "bresenham":
                            amplitude *= 1.35
                        else:
                            amplitude *= 0.95

                    real = math.exp(-(ground + shift) * t) * (
                        1.0 + amplitude * math.exp(-0.8 * t)
                    )

                    if not vector["axis"]:
                        sign = 1.0 if operator == "bresenham" else -1.0
                        real *= 1.0 + sign * 0.006 * q[i] * math.exp(-0.5 * t)

                    if vector["axis"]:
                        imag = 0.0
                    else:
                        imag_scale = 0.0015 if operator == "bresenham" else -0.0010
                        imag = imag_scale * z[i] * math.exp(-0.7 * t) * real
                    values.append([real, imag])
                operators[operator] = values
            vectors[vector["id"]] = operators
        rows.append({"config_id": f"cfg-{i:04d}", "vectors": vectors})
    return rows

def validate_pairing(rows):
    ids = [row["config_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ConfigId")
    for row in rows:
        for vector in VECTORS:
            operators = row["vectors"].get(vector["id"])
            if operators is None:
                raise ValueError("missing vector")
            if set(operators) != {"exhaustive", "bresenham"}:
                raise ValueError("both operators required on every ConfigId")
            if len(operators["exhaustive"]) != len(T_VALUES):
                raise ValueError("incomplete exhaustive T coverage")
            if len(operators["bresenham"]) != len(T_VALUES):
                raise ValueError("incomplete bresenham T coverage")

def mean_real(rows, indices, vector_id, operator, t_index):
    return sum(
        rows[i]["vectors"][vector_id][operator][t_index][0] for i in indices
    ) / len(indices)

def plateau(rows, indices, vector_id, operator):
    means = [
        mean_real(rows, indices, vector_id, operator, t_index)
        for t_index in range(len(T_VALUES))
    ]
    if any(value <= 0.0 for value in means):
        raise ValueError("non-positive Wilson mean")
    effective = [
        math.log(means[t] / means[t + 1])
        for t in range(len(means) - 1)
    ]
    start, end = LATE_T
    selected = [effective[t - 1] for t in range(start, end + 1)]
    return sum(selected) / len(selected)

def paired_plateau_difference(rows, vector_id):
    validate_pairing(rows)
    full = list(range(len(rows)))
    exhaustive = plateau(rows, full, vector_id, "exhaustive")
    bresenham = plateau(rows, full, vector_id, "bresenham")
    replicates = []
    for block in range(BLOCKS):
        lo = block * BLOCK_SIZE
        hi = lo + BLOCK_SIZE
        kept = [i for i in full if not (lo <= i < hi)]
        replicates.append(
            plateau(rows, kept, vector_id, "bresenham")
            - plateau(rows, kept, vector_id, "exhaustive")
        )
    replicate_mean = sum(replicates) / len(replicates)
    se = math.sqrt(
        (len(replicates) - 1)
        / len(replicates)
        * sum((value - replicate_mean) ** 2 for value in replicates)
    )
    return {
        "exhaustive": exhaustive,
        "bresenham": bresenham,
        "difference": bresenham - exhaustive,
        "paired_jackknife_se": se,
    }

def max_raw_complex_difference(rows, vector_id):
    largest = 0.0
    for row in rows:
        a = row["vectors"][vector_id]["exhaustive"]
        b = row["vectors"][vector_id]["bresenham"]
        for av, bv in zip(a, b):
            diff = math.hypot(bv[0] - av[0], bv[1] - av[1])
            largest = max(largest, diff)
    return largest

def max_imaginary_magnitude(rows, vector_id):
    largest = 0.0
    for row in rows:
        for operator in ("exhaustive", "bresenham"):
            for value in row["vectors"][vector_id][operator]:
                largest = max(largest, abs(value[1]))
    return largest

def classify(rows):
    validate_pairing(rows)
    if len(rows) != CONFIGS or len(rows) // BLOCK_SIZE < 4:
        return {"disposition": "Inconclusive", "reason": "InsufficientPairedBlocks"}

    details = {}
    sensitivity = False
    for vector in VECTORS:
        stats = paired_plateau_difference(rows, vector["id"])
        stats["max_raw_complex_difference"] = max_raw_complex_difference(rows, vector["id"])
        stats["max_imaginary_magnitude"] = max_imaginary_magnitude(rows, vector["id"])
        stats["declared_limit"] = ABS_TOL + SIGMA_MULTIPLIER * stats["paired_jackknife_se"]

        if vector["axis"]:
            if stats["max_raw_complex_difference"] != 0.0:
                raise ValueError("axis-collapse raw equality failed")
            if stats["difference"] != 0.0 or stats["paired_jackknife_se"] != 0.0:
                raise ValueError("axis-collapse plateau equality failed")
        else:
            if stats["max_raw_complex_difference"] <= 0.0:
                raise ValueError("off-axis operators unexpectedly raw-identical")
            if stats["max_imaginary_magnitude"] <= 0.0:
                raise ValueError("imaginary diagnostic channel was lost")
            if abs(stats["difference"]) > stats["declared_limit"]:
                sensitivity = True
        details[vector["id"]] = stats

    return {
        "disposition": (
            "OperatorSensitivityDetected"
            if sensitivity
            else "OperatorRobustWithinDeclaredDomain"
        ),
        "details": details,
        "pairing": "same_ConfigId_same_T_same_vector",
        "late_time_policy": {
            "T_window": list(LATE_T),
            "absolute_tolerance": ABS_TOL,
            "paired_sigma_multiplier": SIGMA_MULTIPLIER,
        },
    }

def main():
    robust_rows = synthetic_data(sensitivity=False)
    robust = classify(robust_rows)

    sensitive_rows = synthetic_data(sensitivity=True)
    sensitive = classify(sensitive_rows)

    incomplete_rows = copy.deepcopy(robust_rows[:-1])
    inconclusive = classify(incomplete_rows)

    broken_pairing = copy.deepcopy(robust_rows)
    del broken_pairing[3]["vectors"]["off_110"]["bresenham"]
    pairing_rejected = False
    try:
        classify(broken_pairing)
    except ValueError:
        pairing_rejected = True

    result = {
        "oracle_id": ORACLE_ID,
        "robust_disposition": robust["disposition"],
        "sensitivity_disposition": sensitive["disposition"],
        "inconclusive_disposition": inconclusive["disposition"],
        "pairing_break_rejected": pairing_rejected,
        "axis_raw_exact_equality": (
            robust["details"]["axis_100"]["max_raw_complex_difference"] == 0.0
        ),
        "axis_plateau_exact_equality": (
            robust["details"]["axis_100"]["difference"] == 0.0
        ),
        "off_axis_raw_difference_present": all(
            robust["details"][vector["id"]]["max_raw_complex_difference"] > 0.0
            for vector in VECTORS if not vector["axis"]
        ),
        "imaginary_channel_retained": all(
            robust["details"][vector["id"]]["max_imaginary_magnitude"] > 0.0
            for vector in VECTORS if not vector["axis"]
        ),
        "robust_late_differences": {
            vector["id"]: robust["details"][vector["id"]]["difference"]
            for vector in VECTORS
        },
        "sensitivity_late_differences": {
            vector["id"]: sensitive["details"][vector["id"]]["difference"]
            for vector in VECTORS
        },
        "scientific_boundary": (
            "synthetic_paired_operator_robustness_semantics_only_"
            "no_real_beta6_operator_equivalence_claim"
        ),
    }

    if result["robust_disposition"] != "OperatorRobustWithinDeclaredDomain":
        raise AssertionError("robust fixture did not qualify")
    if result["sensitivity_disposition"] != "OperatorSensitivityDetected":
        raise AssertionError("sensitivity fixture did not detect late-time shift")
    if result["inconclusive_disposition"] != "Inconclusive":
        raise AssertionError("incomplete paired sample did not become Inconclusive")
    if not result["pairing_break_rejected"]:
        raise AssertionError("broken ConfigId pairing was accepted")
    if not result["axis_raw_exact_equality"] or not result["axis_plateau_exact_equality"]:
        raise AssertionError("axis exact null control failed")
    if not result["off_axis_raw_difference_present"]:
        raise AssertionError("off-axis raw difference control inactive")
    if not result["imaginary_channel_retained"]:
        raise AssertionError("imaginary diagnostic channel missing")

    result_sha = digest(b"symthaea.lqcd.021h.result.v1\0", result)
    print("ok")
    print("result_sha256=" + result_sha)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))

if __name__ == "__main__":
    main()
