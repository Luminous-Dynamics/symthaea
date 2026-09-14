#!/usr/bin/env python3
"""Independent covariance-rank admission oracle for LQCD-021F."""
import hashlib
import json

ORACLE_ID = "lqcd_covariance_rank_gate_oracle_v1"
BLOCKS = 8
MAX_EMPIRICAL_RANK = BLOCKS - 1

HADAMARD8 = [
    [ 1, 1, 1, 1, 1, 1, 1, 1],
    [ 1,-1, 1,-1, 1,-1, 1,-1],
    [ 1, 1,-1,-1, 1, 1,-1,-1],
    [ 1,-1,-1, 1, 1,-1,-1, 1],
    [ 1, 1, 1, 1,-1,-1,-1,-1],
    [ 1,-1, 1,-1,-1, 1,-1, 1],
    [ 1, 1,-1,-1,-1,-1, 1, 1],
    [ 1,-1,-1, 1,-1, 1, 1,-1],
]

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

def digest(tag, value):
    return hashlib.sha256(tag + canonical(value)).hexdigest()

def matrix_rank(matrix, tolerance=1.0e-12):
    work = [row[:] for row in matrix]
    rows = len(work)
    cols = len(work[0]) if rows else 0
    rank = 0
    col = 0
    while rank < rows and col < cols:
        pivot = max(range(rank, rows), key=lambda r: abs(work[r][col]))
        if abs(work[pivot][col]) <= tolerance:
            col += 1
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = work[rank][col]
        work[rank] = [value / scale for value in work[rank]]
        for r in range(rows):
            if r == rank:
                continue
            factor = work[r][col]
            if abs(factor) <= tolerance:
                continue
            work[r] = [
                work[r][c] - factor * work[rank][c]
                for c in range(cols)
            ]
        rank += 1
        col += 1
    return rank

def covariance(replicates):
    count = len(replicates)
    width = len(replicates[0])
    means = [
        sum(rep[j] for rep in replicates) / count
        for j in range(width)
    ]
    factor = (count - 1) / count
    return [
        [
            factor * sum(
                (rep[a] - means[a]) * (rep[b] - means[b])
                for rep in replicates
            )
            for b in range(width)
        ]
        for a in range(width)
    ]

def replicates(width):
    if width > 7:
        return [row[1:] + [0.0] * (width - 7) for row in HADAMARD8]
    return [row[1:1 + width] for row in HADAMARD8]

def admit_primary(replicate_values, requested_points):
    if len(replicate_values) != BLOCKS:
        return {"disposition": "InsufficientEvidence", "reason": "WrongReplicateCount"}
    if requested_points > MAX_EMPIRICAL_RANK:
        return {
            "disposition": "NotAdmissible",
            "reason": "RankBudgetExceeded",
            "requested_points": requested_points,
            "max_empirical_rank": MAX_EMPIRICAL_RANK,
        }
    if any(len(row) != requested_points for row in replicate_values):
        return {"disposition": "NotAdmissible", "reason": "WidthMismatch"}

    cov = covariance(replicate_values)
    rank = matrix_rank(cov)
    if rank < requested_points:
        return {
            "disposition": "NotAdmissible",
            "reason": "RankDeficient",
            "observed_rank": rank,
            "requested_points": requested_points,
        }
    return {
        "disposition": "Admissible",
        "observed_rank": rank,
        "requested_points": requested_points,
    }

def diagnostic_ridge(replicate_values, requested_points, ridge):
    primary = admit_primary(replicate_values, requested_points)
    return {
        "authority": "DiagnosticOnly",
        "ridge": ridge,
        "primary_disposition_unchanged": primary["disposition"],
        "may_authorize_primary": False,
    }

def main():
    six = replicates(6)
    seven = replicates(7)
    eight = replicates(8)

    admitted_six = admit_primary(six, 6)
    admitted_seven = admit_primary(seven, 7)
    rejected_eight = admit_primary(eight, 8)

    singular = [row[:] for row in six]
    for row in singular:
        row[5] = row[4]
    rejected_singular = admit_primary(singular, 6)

    diagnostic = diagnostic_ridge(singular, 6, 1.0e-6)

    posthoc_mutations = [
        "drop_point",
        "switch_to_diagonal_covariance",
        "change_block_size",
        "truncate_singular_mode",
        "add_primary_ridge",
        "switch_fit_family",
    ]

    result = {
        "oracle_id": ORACLE_ID,
        "block_replicates": BLOCKS,
        "maximum_empirical_covariance_rank": MAX_EMPIRICAL_RANK,
        "six_point_disposition": admitted_six["disposition"],
        "six_point_rank": admitted_six["observed_rank"],
        "seven_point_edge_disposition": admitted_seven["disposition"],
        "seven_point_rank": admitted_seven["observed_rank"],
        "eight_point_disposition": rejected_eight["disposition"],
        "eight_point_reason": rejected_eight["reason"],
        "singular_six_point_disposition": rejected_singular["disposition"],
        "singular_six_point_reason": rejected_singular["reason"],
        "singular_six_point_rank": rejected_singular["observed_rank"],
        "diagnostic_regularization_authority": diagnostic["authority"],
        "diagnostic_regularization_cannot_rescue_primary": not diagnostic["may_authorize_primary"],
        "forbidden_posthoc_primary_rescues": posthoc_mutations,
        "scientific_boundary": (
            "covariance_rank_and_primary_admission_semantics_only_"
            "no_real_beta6_fit_qualification"
        ),
    }

    if result["six_point_disposition"] != "Admissible" or result["six_point_rank"] != 6:
        raise AssertionError("six-point full-rank fixture failed")
    if result["seven_point_edge_disposition"] != "Admissible" or result["seven_point_rank"] != 7:
        raise AssertionError("rank-seven edge fixture failed")
    if result["eight_point_reason"] != "RankBudgetExceeded":
        raise AssertionError("p > B-1 was not rejected before inversion")
    if result["singular_six_point_reason"] != "RankDeficient":
        raise AssertionError("singular covariance was not rejected")
    if not result["diagnostic_regularization_cannot_rescue_primary"]:
        raise AssertionError("diagnostic ridge gained primary authority")

    result_sha = digest(b"symthaea.lqcd.021f.covrank.result.v1\0", result)
    print("ok")
    print("result_sha256=" + result_sha)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))

if __name__ == "__main__":
    main()
