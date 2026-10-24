#!/usr/bin/env python3
"""Deterministic adversarial scenario family for multi-fidelity casting research.

This module diagnoses how different cheap-fidelity failure mechanisms distort
objective rankings and feasibility labels. It is a known-truth synthetic stress
suite, not an optimizer benchmark, casting simulator, or hostile-code sandbox.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Iterable

FIDELITY_COSTS = {"low": 1, "medium": 5, "high": 25}
FIDELITIES = ("low", "medium", "high")
SUPPORTED_DIMENSIONS = (2, 5, 10)
SUPPORTED_SCENARIOS = (
    "smooth_stationary",
    "local_bias_island",
    "warped_coordinates",
    "constraint_boundary",
    "discontinuous_step",
    "heteroscedastic_noise",
)
DEFAULT_SEEDS = (17, 23, 29)
DEFAULT_SEVERITIES = (0.5, 1.0, 2.0)
DEFAULT_DIMENSIONS = (2, 5, 10)
DEFAULT_DESIGN_COUNT = 128


SCENARIO_CONTRACTS = {
    "smooth_stationary": {
        "failure_mechanism": "Smooth additive bias; inexpensive fidelity remains correlated but systematically offset.",
        "expected_diagnostic": "Non-zero objective-rank inversions with relatively smooth discrepancy.",
    },
    "local_bias_island": {
        "failure_mechanism": "Spatially localized objective and feasibility bias concentrated in one region.",
        "expected_diagnostic": "Rank and feasibility errors cluster near the bias island; global average error may understate local risk.",
    },
    "warped_coordinates": {
        "failure_mechanism": "Low and medium fidelities evaluate a displaced input, creating nonlinear input-dependent discrepancy.",
        "expected_diagnostic": "Errors persist despite smooth high-fidelity truth and can change across dimensions.",
    },
    "constraint_boundary": {
        "failure_mechanism": "Constraint-margin bias amplified near the true feasibility boundary.",
        "expected_diagnostic": "False-feasible promotions concentrate near the decision boundary.",
    },
    "discontinuous_step": {
        "failure_mechanism": "Low-fidelity objective and constraint biases jump at a threshold that differs by fidelity.",
        "expected_diagnostic": "Smooth global correction can fail across the discontinuity.",
    },
    "heteroscedastic_noise": {
        "failure_mechanism": "Deterministic seed-indexed perturbation whose magnitude increases near boundaries and domain edges.",
        "expected_diagnostic": "Calibration and ranking degrade in difficult regions; seed pairing stays reproducible.",
    },
}


@dataclass(frozen=True)
class Design:
    candidate_id: int
    x: tuple[float, ...]


@dataclass(frozen=True)
class Observation:
    candidate_id: int
    fidelity: str
    objectives: tuple[float, float]
    constraint_margin: float
    cost_units: int

    @property
    def predicted_feasible(self) -> bool:
        return self.constraint_margin >= 0.0


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def halton_designs(count: int, dimension: int, seed: int) -> tuple[Design, ...]:
    """Generate a deterministic paired candidate pool from a seed-offset Halton sequence."""
    if count < 8:
        raise ValueError("design_count must be at least 8")
    if dimension < 2 or dimension > len(_PRIMES):
        raise ValueError(f"dimension must be between 2 and {len(_PRIMES)}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    offset = 1 + (seed % 100_003) * 13
    return tuple(
        Design(
            candidate_id=i,
            x=tuple(_radical_inverse(offset + i, _PRIMES[j]) for j in range(dimension)),
        )
        for i in range(count)
    )


_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


class MultiFidelityScenario:
    """Known-truth synthetic scenario with configurable fidelity mismatch."""

    def __init__(
        self,
        scenario_id: str,
        dimension: int = 2,
        severity: float = 1.0,
        seed: int = 17,
        fidelity_costs: dict[str, int] | None = None,
    ) -> None:
        if scenario_id not in SUPPORTED_SCENARIOS:
            raise ValueError(f"unknown scenario_id: {scenario_id}")
        if dimension not in SUPPORTED_DIMENSIONS:
            raise ValueError(f"dimension must be one of {SUPPORTED_DIMENSIONS}")
        if not math.isfinite(severity) or severity < 0.0:
            raise ValueError("severity must be finite and non-negative")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        costs = dict(FIDELITY_COSTS) if fidelity_costs is None else dict(fidelity_costs)
        if set(costs) != set(FIDELITIES):
            raise ValueError("fidelity costs must define exactly low, medium, and high")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in costs.values()):
            raise ValueError("fidelity costs must be positive integers")
        self.scenario_id = scenario_id
        self.dimension = dimension
        self.severity = severity
        self.seed = seed
        self._costs = costs
        self._designs = halton_designs(DEFAULT_DESIGN_COUNT, dimension, seed)
        self._by_id = {design.candidate_id: design for design in self._designs}

    @property
    def fidelity_costs(self) -> dict[str, int]:
        return dict(self._costs)

    @property
    def designs(self) -> tuple[Design, ...]:
        return self._designs

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        return tuple(self._by_id)

    def design(self, candidate_id: int) -> Design:
        try:
            return self._by_id[candidate_id]
        except KeyError as exc:
            raise ValueError(f"unknown candidate_id: {candidate_id}") from exc

    def _truth_at(self, x: tuple[float, ...]) -> tuple[tuple[float, float], float]:
        d = len(x)
        target0 = tuple(0.22 + 0.035 * math.sin(i + 1.0) for i in range(d))
        target1 = tuple(0.78 - 0.035 * math.cos(i + 1.0) for i in range(d))
        f0 = 0.08 + sum((x[i] - target0[i]) ** 2 for i in range(d)) / d
        f1 = 0.08 + sum((x[i] - target1[i]) ** 2 for i in range(d)) / d
        if d > 1:
            f0 += 0.012 * math.sin(4.0 * math.pi * x[0]) * math.cos(2.0 * math.pi * x[-1])
            f1 += 0.012 * math.cos(3.0 * math.pi * x[0]) * math.sin(2.0 * math.pi * x[-1])
        center = tuple(0.48 + 0.035 * math.sin(0.7 * (i + 1)) for i in range(d))
        radius = sum((x[i] - center[i]) ** 2 for i in range(d)) / d
        margin = 0.075 - radius
        return (f0, f1), margin

    def truth_for_audit_only(self, candidate_id: int) -> Observation:
        """Separate audit-only truth query; not used by a policy during the diagnostic run."""
        design = self.design(candidate_id)
        objectives, margin = self._truth_at(design.x)
        return Observation(candidate_id, "high", objectives, margin, self._costs["high"])

    def evaluate(self, candidate_id: int, fidelity: str) -> Observation:
        if fidelity not in self._costs:
            raise ValueError(f"unsupported fidelity: {fidelity}")
        design = self.design(candidate_id)
        x = design.x
        severity = self.severity
        truth, true_margin = self._truth_at(x)
        if fidelity == "high":
            objectives, margin = truth, true_margin
        else:
            scale = 0.38 if fidelity == "medium" else 1.0
            s = severity * scale
            x0 = x[0]
            xlast = x[-1]
            f0, f1 = truth
            margin = true_margin

            if self.scenario_id == "smooth_stationary":
                objectives = (
                    f0 + s * (0.12 * x0 + 0.045 * math.sin(2.0 * math.pi * (x0 + xlast))),
                    f1 + s * (-0.09 * xlast + 0.035 * math.cos(2.0 * math.pi * x0)),
                )
                margin += s * (0.11 * math.sin(3.0 * math.pi * x0 + 2.0 * math.pi * xlast) - 0.018)

            elif self.scenario_id == "local_bias_island":
                center = tuple(0.72 if i % 2 == 0 else 0.28 for i in range(self.dimension))
                distance = sum((x[i] - center[i]) ** 2 for i in range(self.dimension)) / self.dimension
                bump = math.exp(-distance / (2.0 * 0.16 ** 2))
                objectives = (f0 + s * 0.32 * bump, f1 - s * 0.24 * bump)
                margin += s * (-0.34 * bump + 0.015 * math.sin(5.0 * math.pi * x0))

            elif self.scenario_id == "warped_coordinates":
                warp = (0.045 if fidelity == "medium" else 0.15) * severity
                warped = tuple(
                    min(1.0, max(0.0, value + warp * math.sin(2.0 * math.pi * (x[(i + 1) % self.dimension] + i / 7.0))))
                    for i, value in enumerate(x)
                )
                objectives, margin = self._truth_at(warped)
                objectives = (
                    objectives[0] + s * 0.025 * math.sin(2.0 * math.pi * x0),
                    objectives[1] - s * 0.02 * math.cos(2.0 * math.pi * xlast),
                )

            elif self.scenario_id == "constraint_boundary":
                boundary_weight = math.exp(-abs(true_margin) / 0.055)
                objectives = (
                    f0 + s * 0.035 * math.sin(5.0 * math.pi * x0),
                    f1 + s * 0.035 * math.cos(4.0 * math.pi * xlast),
                )
                margin += s * 0.28 * math.sin(5.0 * math.pi * x0 + 3.0 * math.pi * xlast) * boundary_weight

            elif self.scenario_id == "discontinuous_step":
                threshold = 0.61 if fidelity == "medium" else 0.48
                step = 1.0 if x0 >= threshold else -0.25
                objective_step = 0.16 if fidelity == "medium" else 0.30
                objectives = (f0 + s * objective_step * step, f1 - s * objective_step * 0.7 * step)
                margin += s * (0.20 if x0 >= threshold else -0.11)

            else:  # heteroscedastic_noise; deterministic within a seed/design/fidelity tuple
                phase = (self.seed % 997) * 0.013
                boundary_weight = 1.0 + 2.0 * math.exp(-abs(true_margin) / 0.05)
                edge_weight = 1.0 + 0.8 * (abs(x0 - 0.5) + abs(xlast - 0.5))
                amp = s * 0.035 * boundary_weight * edge_weight
                noise0 = amp * math.sin(13.0 * x0 + 17.0 * xlast + phase)
                noise1 = amp * math.cos(19.0 * x0 - 11.0 * xlast + phase)
                objectives = (f0 + noise0, f1 + noise1)
                margin += s * 0.18 * math.sin(11.0 * x0 + 7.0 * xlast + phase) / boundary_weight

        if not all(math.isfinite(value) for value in (*objectives, margin)):
            raise ArithmeticError("scenario emitted a non-finite value")
        return Observation(candidate_id, fidelity, objectives, margin, self._costs[fidelity])


def _percentile(values: list[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise ValueError("percentile requires non-empty values and q in [0,1]")
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def audit_scenario(
    scenario_id: str,
    dimension: int,
    severity: float,
    seed: int,
    design_count: int = DEFAULT_DESIGN_COUNT,
    fidelity_costs: dict[str, int] | None = None,
) -> dict:
    """Measure fidelity mismatch against known truth; no optimizer is run by this function."""
    world = MultiFidelityScenario(scenario_id, dimension, severity, seed, fidelity_costs)
    designs = halton_designs(design_count, dimension, seed)
    high = [world.truth_for_audit_only(design.candidate_id) for design in designs]
    reports: dict[str, dict] = {}

    for fidelity in ("low", "medium"):
        observations = [world.evaluate(design.candidate_id, fidelity) for design in designs]
        rank_total = 0
        rank_inversions = 0
        pair_count = design_count * (design_count - 1) // 2
        for i, j in itertools.combinations(range(design_count), 2):
            for objective_index in range(2):
                lhs = observations[i].objectives[objective_index] - observations[j].objectives[objective_index]
                rhs = high[i].objectives[objective_index] - high[j].objectives[objective_index]
                rank_total += 1
                if lhs * rhs < -1e-15:
                    rank_inversions += 1

        feasible_flips = sum(
            obs.predicted_feasible != truth.predicted_feasible
            for obs, truth in zip(observations, high)
        )
        false_feasible = sum(
            obs.predicted_feasible and not truth.predicted_feasible
            for obs, truth in zip(observations, high)
        )
        predicted_feasible = sum(obs.predicted_feasible for obs in observations)
        reports[fidelity] = {
            "objective_rank_inversion_fraction": rank_inversions / rank_total if rank_total else 0.0,
            "objective_rank_inversion_count": rank_inversions,
            "objective_pair_comparisons": rank_total,
            "feasibility_flip_count": feasible_flips,
            "feasibility_flip_fraction": feasible_flips / design_count,
            "false_feasible_count": false_feasible,
            "false_feasible_fraction_of_predictions": (
                false_feasible / predicted_feasible if predicted_feasible else None
            ),
            "mean_absolute_objective_error": statistics.fmean(
                abs(obs.objectives[index] - truth.objectives[index])
                for obs, truth in zip(observations, high)
                for index in range(2)
            ),
            "mean_absolute_constraint_margin_error": statistics.fmean(
                abs(obs.constraint_margin - truth.constraint_margin)
                for obs, truth in zip(observations, high)
            ),
            "cost_units_per_call": world.fidelity_costs[fidelity],
        }

    return {
        "benchmark_id": "CASTING-MF-SCENARIOS-001",
        "synthetic_fixture_only": True,
        "scenario_id": scenario_id,
        "scenario_family": SCENARIO_CONTRACTS[scenario_id],
        "dimension": dimension,
        "severity": severity,
        "seed": seed,
        "design_count": design_count,
        "fidelity_costs": world.fidelity_costs,
        "high_truth_feasible_count": sum(obs.predicted_feasible for obs in high),
        "diagnostics_by_fidelity": reports,
        "scope_note": "Fidelity-diagnostic results only. No optimizer policy was run and no casting physics was validated.",
    }


def run_scenario_sweep(
    scenario_ids: Iterable[str] = SUPPORTED_SCENARIOS,
    dimensions: Iterable[int] = DEFAULT_DIMENSIONS,
    severities: Iterable[float] = DEFAULT_SEVERITIES,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    design_count: int = DEFAULT_DESIGN_COUNT,
) -> dict:
    scenarios = tuple(scenario_ids)
    dims = tuple(dimensions)
    severity_values = tuple(float(v) for v in severities)
    seed_values = tuple(seeds)
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenario_ids must be non-empty and unique")
    unknown = sorted(set(scenarios) - set(SUPPORTED_SCENARIOS))
    if unknown:
        raise ValueError(f"unsupported scenario IDs: {unknown}")
    if not dims or any(d not in SUPPORTED_DIMENSIONS for d in dims) or len(set(dims)) != len(dims):
        raise ValueError(f"dimensions must be unique values from {SUPPORTED_DIMENSIONS}")
    if not severity_values or any(not math.isfinite(v) or v < 0.0 for v in severity_values):
        raise ValueError("severities must be non-empty, finite and non-negative")
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be non-empty and unique")
    if design_count < 8:
        raise ValueError("design_count must be at least 8")

    runs = [
        audit_scenario(scenario, dimension, severity, seed, design_count)
        for scenario in scenarios
        for dimension in dims
        for severity in severity_values
        for seed in seed_values
    ]
    summaries: list[dict] = []
    for scenario in scenarios:
        for dimension in dims:
            for severity in severity_values:
                for fidelity in ("low", "medium"):
                    group = [
                        run for run in runs
                        if run["scenario_id"] == scenario and run["dimension"] == dimension
                        and run["severity"] == severity
                    ]
                    inversion = [
                        row["diagnostics_by_fidelity"][fidelity]["objective_rank_inversion_fraction"]
                        for row in group
                    ]
                    flips = [
                        row["diagnostics_by_fidelity"][fidelity]["feasibility_flip_fraction"]
                        for row in group
                    ]
                    mae = [
                        row["diagnostics_by_fidelity"][fidelity]["mean_absolute_objective_error"]
                        for row in group
                    ]
                    summaries.append({
                        "scenario_id": scenario,
                        "dimension": dimension,
                        "severity": severity,
                        "fidelity": fidelity,
                        "paired_seed_count": len(group),
                        "median_objective_rank_inversion_fraction": statistics.median(inversion),
                        "q25_objective_rank_inversion_fraction": _percentile(inversion, 0.25),
                        "q75_objective_rank_inversion_fraction": _percentile(inversion, 0.75),
                        "mean_feasibility_flip_fraction": statistics.fmean(flips),
                        "median_mean_absolute_objective_error": statistics.median(mae),
                    })

    return {
        "benchmark_id": "CASTING-MF-SCENARIOS-001",
        "synthetic_fixture_only": True,
        "scenario_ids": list(scenarios),
        "dimensions": list(dims),
        "severities": list(severity_values),
        "seeds": list(seed_values),
        "design_count": design_count,
        "run_count": len(runs),
        "summary_count": len(summaries),
        "scenario_contracts": {key: SCENARIO_CONTRACTS[key] for key in scenarios},
        "summary_by_scenario_dimension_severity_fidelity": summaries,
        "paired_runs": runs,
        "warning": (
            "This is a fidelity-mismatch diagnostic suite, not an end-to-end optimizer bake-off. "
            "All outputs are synthetic known-truth diagnostics; do not claim SOTA or foundry validation."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default=",".join(SUPPORTED_SCENARIOS),
                        help="comma-separated IDs from the scenario registry")
    parser.add_argument("--dimensions", default=",".join(str(v) for v in DEFAULT_DIMENSIONS))
    parser.add_argument("--severities", default=",".join(str(v) for v in DEFAULT_SEVERITIES))
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--design-count", type=int, default=DEFAULT_DESIGN_COUNT)
    args = parser.parse_args(argv)
    result = run_scenario_sweep(
        scenario_ids=[v.strip() for v in args.scenarios.split(",") if v.strip()],
        dimensions=[int(v.strip()) for v in args.dimensions.split(",") if v.strip()],
        severities=[float(v.strip()) for v in args.severities.split(",") if v.strip()],
        seeds=[int(v.strip()) for v in args.seeds.split(",") if v.strip()],
        design_count=args.design_count,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
