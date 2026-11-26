#!/usr/bin/env python3
"""Independent standard-library geometry oracle for layout-search v0.1.

This checks deterministic geometry only. It does not model physical probabilities.
"""
from __future__ import annotations
import json

X, Y, Z = 4, 4, 3
RESERVED = {0, 15, 16, 31, 32, 47}
STRIPES, SHARDS, BUDGET = 7, 6, 2
ACTIVE = [tile for tile in range(X * Y * Z) if tile not in RESERVED]


def cuboids(dx, dy, dz):
    masks = []
    for z in range(Z - dz + 1):
        for y in range(Y - dy + 1):
            for x in range(X - dx + 1):
                masks.append({
                    zz * X * Y + yy * X + xx
                    for zz in range(z, z + dz)
                    for yy in range(y, y + dy)
                    for xx in range(x, x + dx)
                })
    return masks


def contiguous(order):
    return [order[stripe * SHARDS + shard]
            for stripe in range(STRIPES) for shard in range(SHARDS)]


def interleaved(order):
    return [order[shard * STRIPES + stripe]
            for stripe in range(STRIPES) for shard in range(SHARDS)]


def score(mapping, masks):
    failures = 0
    worst = 0
    excess = 0
    for mask in masks:
        per_stripe = [
            sum(mapping[stripe * SHARDS + shard] in mask for shard in range(SHARDS))
            for stripe in range(STRIPES)
        ]
        maximum = max(per_stripe)
        worst = max(worst, maximum)
        failures += maximum > BUDGET
        excess += sum(max(0, value - BUDGET) for value in per_stripe)
    return failures, worst, excess


def main():
    shapes = [(2, 2, 1), (3, 3, 1), (2, 2, 2), (3, 3, 2), (4, 4, 1)]
    masks = [mask for shape in shapes for mask in cuboids(*shape)]
    denominators = [len(cuboids(*shape)) for shape in shapes]
    assert denominators == [27, 12, 18, 8, 3]

    row_score = score(contiguous(ACTIVE), masks)
    spatial_score = score(interleaved(ACTIVE), masks)
    assert row_score == (50, 6, 166)
    assert spatial_score == (8, 4, 32)

    print(json.dumps({
        "evidence_class": "independent_deterministic_geometry_oracle",
        "fault_set_count": len(masks),
        "cuboid_denominators": denominators,
        "row_major": {
            "unrecoverable_fault_sets": row_score[0],
            "worst_erased_shards_in_one_stripe": row_score[1],
            "total_excess_erasures": row_score[2]
        },
        "spatial_interleaved": {
            "unrecoverable_fault_sets": spatial_score[0],
            "worst_erased_shards_in_one_stripe": spatial_score[1],
            "total_excess_erasures": spatial_score[2]
        },
        "physical_error_probabilities_estimated": False,
        "rust_tests_executed": False
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
