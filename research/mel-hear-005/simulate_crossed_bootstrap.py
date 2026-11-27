#!/usr/bin/env python3
"""Research-only oracle for MEL-HEAR-005 crossed-cluster inference.

Balanced complete work x performer panel; one performer per cell. This is not
production inference code. Multi-performer trials and missing/unbalanced cells
are intentionally outside v0's claim scope.
"""
from __future__ import annotations

import json
import math
import platform
import numpy as np


def simulate_work_disjoint(
    works=30, within_work_repeats=4, work_sd=0.30, observation_sd=0.25,
    simulations=1_000, bootstrap_replicates=399, seed=77,
):
    """Independent performer/work clusters; one-way work bootstrap is valid."""
    rng = np.random.default_rng(seed)
    covered, widths = 0, []
    for simulation in range(simulations):
        cluster_effect = rng.normal(0.0, work_sd, size=works)
        repeated_noise = rng.normal(
            0.0, observation_sd, size=(works, within_work_repeats)
        ).mean(axis=1)
        cluster_means = cluster_effect + repeated_noise
        boot_rng = np.random.default_rng(seed + simulation + 10_000)
        indices = boot_rng.integers(0, works, size=(bootstrap_replicates, works))
        bootstrap_means = cluster_means[indices].mean(axis=1)
        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        covered += int(lower <= 0.0 <= upper)
        widths.append(float(upper - lower))
    coverage = covered / simulations
    return {
        "scenario": "work_disjoint_performer_clusters",
        "works": works,
        "within_work_repeats": within_work_repeats,
        "simulations": simulations,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "work_bootstrap_coverage": coverage,
        "work_bootstrap_monte_carlo_se": math.sqrt(
            coverage * (1.0 - coverage) / simulations
        ),
        "work_bootstrap_mean_interval_width": float(np.mean(widths)),
    }


def simulate_crossed(
    works, performers, work_sd, performer_sd, cell_sd=0.30,
    simulations=1_000, bootstrap_replicates=499, seed=991,
):
    """Compare one-way work bootstrap with two-way pigeonhole resampling."""
    rng = np.random.default_rng(seed)
    work_only_covered = 0
    pigeonhole_covered = 0
    work_only_widths, pigeonhole_widths, estimates = [], [], []

    for simulation in range(simulations):
        paired_effect = (
            rng.normal(0.0, work_sd, size=(works, 1))
            + rng.normal(0.0, performer_sd, size=(1, performers))
            + rng.normal(0.0, cell_sd, size=(works, performers))
        )
        estimates.append(float(paired_effect.mean()))
        boot_rng = np.random.default_rng(seed + 10_000 + simulation)

        # A work-only resample omits the shared performer-axis contribution.
        work_means = paired_effect.mean(axis=1)
        work_draws = boot_rng.integers(
            0, works, size=(bootstrap_replicates, works)
        )
        work_bootstrap = work_means[work_draws].mean(axis=1)
        work_lower, work_upper = np.quantile(
            work_bootstrap, [0.025, 0.975]
        )

        # Pigeonhole bootstrap: independently draw both cluster axes.
        # Multiplicity products weight each observed work-performer cell.
        work_counts = boot_rng.multinomial(
            works, np.full(works, 1.0 / works), size=bootstrap_replicates
        )
        performer_counts = boot_rng.multinomial(
            performers,
            np.full(performers, 1.0 / performers),
            size=bootstrap_replicates,
        )
        crossed_bootstrap = np.einsum(
            "bi,ij,bj->b", work_counts, paired_effect, performer_counts
        ) / (works * performers)
        crossed_lower, crossed_upper = np.quantile(
            crossed_bootstrap, [0.025, 0.975]
        )

        work_only_covered += int(work_lower <= 0.0 <= work_upper)
        pigeonhole_covered += int(crossed_lower <= 0.0 <= crossed_upper)
        work_only_widths.append(float(work_upper - work_lower))
        pigeonhole_widths.append(float(crossed_upper - crossed_lower))

    work_coverage = work_only_covered / simulations
    crossed_coverage = pigeonhole_covered / simulations
    theoretical_sd = math.sqrt(
        work_sd**2 / works
        + performer_sd**2 / performers
        + cell_sd**2 / (works * performers)
    )
    return {
        "scenario": f"balanced_crossed_{works}_works_x_{performers}_performers",
        "works": works,
        "performers": performers,
        "simulations": simulations,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "work_only_coverage": work_coverage,
        "work_only_monte_carlo_se": math.sqrt(
            work_coverage * (1.0 - work_coverage) / simulations
        ),
        "two_way_pigeonhole_coverage": crossed_coverage,
        "two_way_monte_carlo_se": math.sqrt(
            crossed_coverage * (1.0 - crossed_coverage) / simulations
        ),
        "mean_interval_width_work_only": float(np.mean(work_only_widths)),
        "mean_interval_width_two_way": float(np.mean(pigeonhole_widths)),
        "empirical_sd_of_estimator": float(np.std(estimates, ddof=1)),
        "theoretical_sd_of_estimator": theoretical_sd,
    }


def main():
    report = {
        "schema": "mel-hear-005-crossed-bootstrap-simulation-v0",
        "research_only": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "nominal_confidence_level": 0.95,
        "scenarios": [
            simulate_work_disjoint(),
            simulate_crossed(
                works=30, performers=20, work_sd=0.18,
                performer_sd=0.22, seed=991,
            ),
            simulate_crossed(
                works=30, performers=8, work_sd=0.18,
                performer_sd=0.30, seed=1234,
            ),
        ],
        "limitations": [
            "balanced complete work-by-performer cells only",
            "one performer identity per cell; no ensemble membership sets",
            "paired effect is continuous Gaussian random-effects oracle, not a model score",
            "percentile bootstrap only; not a qualification of CR2 or wild cluster bootstrap",
            "coverage simulation does not prove real cohort exchangeability or authentication",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
