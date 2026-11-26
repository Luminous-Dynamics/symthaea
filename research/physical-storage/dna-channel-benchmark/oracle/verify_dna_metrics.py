#!/usr/bin/env python3
"""Independent, standard-library reference for DNA channel metric fixtures."""
from __future__ import annotations
import json


def levenshtein(a: str, b: str) -> tuple[int, int, int, int]:
    rows, cols = len(a) + 1, len(b) + 1
    dp = [[(0, "S") for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = (i, "D")
    for j in range(1, cols):
        dp[0][j] = (j, "I")
    priority = {"M": 0, "S": 1, "D": 2, "I": 3}
    for i in range(1, rows):
        for j in range(1, cols):
            same = a[i-1] == b[j-1]
            options = [
                (dp[i-1][j-1][0] + (0 if same else 1), "M" if same else "S"),
                (dp[i-1][j][0] + 1, "D"),
                (dp[i][j-1][0] + 1, "I"),
            ]
            dp[i][j] = min(options, key=lambda item: (item[0], priority[item[1]]))
    i, j = len(a), len(b)
    counts = {"M": 0, "S": 0, "D": 0, "I": 0}
    while i or j:
        op = dp[i][j][1]
        counts[op] += 1
        if op in ("M", "S"):
            i -= 1
            j -= 1
        elif op == "D":
            i -= 1
        else:
            j -= 1
    return counts["S"] + counts["D"] + counts["I"], counts["S"], counts["I"], counts["D"]


def main():
    reference = "ACGTACGT"
    cases = [
        ("exact", reference, "ACGTACGT", (0, 0, 0, 0)),
        ("substitution", reference, "ACGTTCGT", (1, 1, 0, 0)),
        ("insertion", reference, "ACGTTACGT", (1, 0, 1, 0)),
        ("deletion", reference, "ACGACGT", (1, 0, 0, 1)),
        ("mixed", "ACGTACGT", "ATGTCGT", (2, 1, 0, 1)),
    ]
    measured = []
    for name, source, read, expected in cases:
        actual = levenshtein(source, read)
        assert actual == expected, (name, actual, expected)
        measured.append({"case": name, "distance": actual[0],
                         "substitutions": actual[1],
                         "insertions": actual[2], "deletions": actual[3]})

    probabilities = [0.01, 0.12, 0.83, 0.41, 0.03]
    ranked = sorted(
        [index for index, value in enumerate(probabilities) if value >= 0.10],
        key=lambda index: (-probabilities[index], index),
    )
    assert ranked == [2, 3, 1]
    trials = [ranked[:i] for i in range(1, min(4, len(ranked)) + 1)]
    assert trials == [[2], [2, 3], [2, 3, 1]]
    report = {
        "evidence_class": "independent_synthetic_algorithm_oracle",
        "edit_cases": measured,
        "soft_erasure_ranking": ranked,
        "progressive_trials": trials,
        "read_observation_multiplicity": {
            "read_records": 4, "molecule_observations": 7, "duplicates": 3
        },
        "physical_error_probabilities_estimated": False,
        "biochemical_experiment_performed": False,
        "rust_tests_executed": False
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
