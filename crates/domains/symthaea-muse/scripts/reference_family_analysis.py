#!/usr/bin/env python3
"""Independent standard-library cross-check for V8.2 family statistics.

The script intentionally reimplements the language-neutral SplitMix64 bootstrap
and the exact/sign-randomization test without importing project code.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MASK = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed & MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
        return (value ^ (value >> 31)) & MASK

    def index(self, length: int) -> int:
        return self.next_u64() % length


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def percentile_index(length: int, percentile: float) -> int:
    # Rust f64::round rounds half away from zero. Inputs here are non-negative.
    value = (length - 1) * percentile
    return min(int(math.floor(value + 0.5)), length - 1)


def bootstrap_interval(
    values: list[float], replicates: int, alpha: float, seed: int
) -> list[float]:
    rng = SplitMix64(seed)
    samples = []
    for _ in range(replicates):
        samples.append(mean([values[rng.index(len(values))] for _ in values]))
    samples.sort()
    tail = alpha / 2.0
    return [
        samples[percentile_index(len(samples), tail)],
        samples[percentile_index(len(samples), 1.0 - tail)],
    ]


def sign_randomization_p_value(
    values: list[float], replicates: int, seed: int
) -> float:
    observed = mean(values)
    if observed <= 0.0:
        return 1.0
    if len(values) <= 20:
        permutations = 1 << len(values)
        extreme = 0
        for mask in range(permutations):
            permuted = mean(
                [value if mask & (1 << index) == 0 else -value for index, value in enumerate(values)]
            )
            if permuted >= observed - float.fromhex("0x1.0000000000000p-52"):
                extreme += 1
        return extreme / permutations
    rng = SplitMix64(seed)
    extreme = 0
    for _ in range(replicates):
        permuted = mean(
            [value if rng.next_u64() & 1 == 0 else -value for value in values]
        )
        if permuted >= observed - float.fromhex("0x1.0000000000000p-52"):
            extreme += 1
    return (extreme + 1) / (replicates + 1)


def summarize(payload: dict) -> dict:
    values = [float(value) for value in payload["family_values"]]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("family_values must be finite and non-empty")
    margin = float(payload["required_margin"])
    alpha = float(payload["alpha"])
    bootstrap_replicates = int(payload["bootstrap_replicates"])
    randomization_replicates = int(payload["randomization_replicates"])
    seed = int(payload["seed"])
    centered = [value - margin for value in values]
    return {
        "mean_effect": mean(values),
        "confidence_interval": bootstrap_interval(
            values, bootstrap_replicates, alpha, seed
        ),
        "raw_one_sided_p": sign_randomization_p_value(
            centered, randomization_replicates, seed ^ 0xA11CE5E5
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    args.output.write_text(json.dumps(summarize(payload), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
