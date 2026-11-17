#!/usr/bin/env python3
"""Deterministically generate exact point-set orbits for D4 subgroups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "v0.1" / "cases.json"
MASK = 0xFFFFFFFF

FAMILIES = [
    ("D4", ["e", "r90", "r180", "r270", "fx", "fy", "fdiag", "fanti"], 0x5EED0000),
    ("C4", ["e", "r90", "r180", "r270"], 0x5EED1000),
    ("C2_R180", ["e", "r180"], 0x5EED2000),
    ("C2_FX", ["e", "fx"], 0x5EED3000),
    ("C1", ["e"], 0x5EED4000),
]


def xorshift32(value: int) -> int:
    value &= MASK
    value ^= (value << 13) & MASK
    value &= MASK
    value ^= value >> 17
    value &= MASK
    value ^= (value << 5) & MASK
    return value & MASK


def matrix_apply(matrix, point):
    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1],
        matrix[1][0] * point[0] + matrix[1][1] * point[1],
    )


def point_set_key(points):
    return tuple(sorted((int(point[0]), int(point[1])) for point in points))


def exact_stabilizer(points, elements):
    point_set = set(points)
    return sorted(
        element["id"]
        for element in elements
        if {
            matrix_apply(element["matrix"], point)
            for point in points
        } == point_set
    )


def generate_cases(base_fixture):
    elements = base_fixture["ambient_group"]["elements"]
    by_id = {element["id"]: element for element in elements}
    hand_authored = [
        case
        for case in base_fixture["cases"]
        if not case["case_id"].startswith("GENERATED-")
    ]
    seen = {
        point_set_key(case["geometry"]["points"])
        for case in hand_authored
        if "geometry" in case
    }
    generated = []

    for subgroup_label, subgroup_ids, base_seed in FAMILIES:
        matrices = [by_id[element_id]["matrix"] for element_id in subgroup_ids]
        target = set(subgroup_ids)
        for index in range(30):
            accepted = None
            for attempt in range(1000):
                seed = (base_seed + index * 7919 + attempt * 104729) & MASK
                state = seed or 1
                state = xorshift32(state)
                x = (state % 101) - 50
                state = xorshift32(state)
                y = (state % 101) - 50
                if x == 0 or y == 0 or x == y or x == -y:
                    continue

                points = sorted(
                    {matrix_apply(matrix, (x, y)) for matrix in matrices}
                )
                if len(points) != len(matrices):
                    continue
                key = point_set_key(points)
                if key in seen:
                    continue
                actual = set(exact_stabilizer(points, elements))
                if actual != target:
                    continue
                accepted = (seed, points, key)
                break

            if accepted is None:
                raise RuntimeError(
                    f"could not generate {subgroup_label} case {index + 1}"
                )
            seed, points, key = accepted
            seen.add(key)
            generated.append(
                {
                    "case_id": (
                        f"GENERATED-{subgroup_label}-{index + 1:02d}"
                    ),
                    "track": "exact_finite_structural_stabilizer",
                    "geometry": {
                        "dimension": 2,
                        "points": [list(point) for point in points],
                        "point_order_semantics": "unordered set",
                    },
                    "target_subgroup": subgroup_ids,
                    "expected_subgroup_label": subgroup_label,
                    "controls": {
                        "is_exact": True,
                        "generated": True,
                        "generator_version": "xorshift32-orbit-v1",
                        "seed": seed,
                        "construction": (
                            "single generic rational vector orbit under "
                            "the target D4 subgroup"
                        ),
                    },
                }
            )
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the generated cases in the frozen manifest",
    )
    args = parser.parse_args()

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    generated = generate_cases(fixture)
    authored = [
        case
        for case in fixture["cases"]
        if not case["case_id"].startswith("GENERATED-")
    ]
    replacement = authored + generated

    if args.write:
        fixture["cases"] = replacement
        fixture["benchmark_status"]["fixture_validation"] = (
            "hand_authored_and_generated_exact_fixtures_authored_not_yet_checked_by_GitHub_Actions"
        )
        fixture["benchmark_status"]["results"] = (
            "no external baseline results; 150 generated cases are exact-oracle inputs only"
        )
        FIXTURE.write_text(
            json.dumps(fixture, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"WROTE {len(replacement)} total cases ({len(generated)} generated)")
        return

    current = [
        case
        for case in fixture["cases"]
        if case["case_id"].startswith("GENERATED-")
    ]
    if current != generated:
        raise SystemExit("D4_GENERATOR_MISMATCH: regenerate with --write")
    print("D4_GENERATOR_REPLAY_PASS: 150 cases, 30 per subgroup family")


if __name__ == "__main__":
    main()
