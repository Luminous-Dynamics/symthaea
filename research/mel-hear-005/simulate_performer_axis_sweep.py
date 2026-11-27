#!/usr/bin/env python3
"""Research-only performer-axis coverage sweep for MEL-HEAR-005.

Balanced complete W x P crossed Gaussian random-effects panel. This is a
simulation oracle, not production inference or a claim about real cohorts.
"""
from __future__ import annotations
import json, math, platform
import numpy as np

def simulate_axis(performer_counts=(5, 8, 10, 12, 15, 20, 30, 50),
                  simulations=1_000, bootstrap_replicates=399,
                  works=30, seed=20_261_127):
    master = np.random.default_rng(seed)
    results = []
    work_sd, performer_sd, cell_sd = 0.18, 0.30, 0.30
    batch_size = 50
    for performers in performer_counts:
        work_only_covered = two_way_covered = 0
        work_only_widths, two_way_widths = [], []
        completed = 0
        while completed < simulations:
            batch = min(batch_size, simulations - completed)
            rng = np.random.default_rng(int(master.integers(0, 2**32 - 1)))
            work_effect = rng.normal(0.0, work_sd, size=(batch, works, 1))
            performer_effect = rng.normal(0.0, performer_sd, size=(batch, 1, performers))
            cell_effect = rng.normal(0.0, cell_sd, size=(batch, works, performers))
            paired_effect = work_effect + performer_effect + cell_effect
            work_means = paired_effect.mean(axis=2)
            work_counts = rng.multinomial(works, np.full(works, 1.0 / works),
                                          size=(batch, bootstrap_replicates))
            work_bootstrap = np.einsum("sbw,sw->sb", work_counts, work_means,
                                       optimize=True) / works
            work_lower, work_upper = np.quantile(work_bootstrap, [0.025, 0.975], axis=1)
            work_counts_2 = rng.multinomial(works, np.full(works, 1.0 / works),
                                            size=(batch, bootstrap_replicates))
            performer_counts_2 = rng.multinomial(
                performers, np.full(performers, 1.0 / performers),
                size=(batch, bootstrap_replicates))
            two_way_bootstrap = np.einsum(
                "sbw,swp,sbp->sb", work_counts_2, paired_effect,
                performer_counts_2, optimize=True) / (works * performers)
            two_way_lower, two_way_upper = np.quantile(
                two_way_bootstrap, [0.025, 0.975], axis=1)
            work_only_covered += int(np.sum((work_lower <= 0.0) & (work_upper >= 0.0)))
            two_way_covered += int(np.sum((two_way_lower <= 0.0) & (two_way_upper >= 0.0)))
            work_only_widths.extend((work_upper - work_lower).tolist())
            two_way_widths.extend((two_way_upper - two_way_lower).tolist())
            completed += batch
        work_coverage, two_way_coverage = work_only_covered / simulations, two_way_covered / simulations
        results.append({
            "performers": performers,
            "work_only_coverage": work_coverage,
            "work_only_monte_carlo_se": math.sqrt(work_coverage * (1.0 - work_coverage) / simulations),
            "two_way_pigeonhole_coverage": two_way_coverage,
            "two_way_monte_carlo_se": math.sqrt(two_way_coverage * (1.0 - two_way_coverage) / simulations),
            "work_only_mean_interval_width": float(np.mean(work_only_widths)),
            "two_way_mean_interval_width": float(np.mean(two_way_widths)),
        })
    return {
        "schema": "mel-hear-005-performer-axis-sweep-v0", "research_only": True,
        "python": platform.python_version(), "numpy": np.__version__,
        "nominal_confidence_level": 0.95, "works": works,
        "simulations_per_axis": simulations, "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "random_effect_standard_deviations": {
            "source_work": work_sd, "performer": performer_sd, "work_performer_cell": cell_sd},
        "scenarios": results,
        "limitations": [
            "balanced complete work-by-performer panel only",
            "one performer identity per cell",
            "Gaussian additive random-effects data-generating process",
            "nominal percentile intervals; no small-sample correction",
            "simulation does not establish real-world sampling exchangeability"],
    }

if __name__ == "__main__":
    print(json.dumps(simulate_axis(), indent=2, sort_keys=True))
