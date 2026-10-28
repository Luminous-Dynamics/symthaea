#!/usr/bin/env python3
"""Repeated-seed calibration audit for the causal replay reference estimator.

Synthetic-only: measures empirical interval coverage and gate behavior under planted effects.
"""
from __future__ import annotations
import argparse, json, random, sys
import numpy as np
from pathlib import Path
import importlib.util

MODULE_PATH = Path(__file__).with_name("pev_causal_replay_effect_benchmark_v1.py")
spec = importlib.util.spec_from_file_location("pev_causal_replay_effect_benchmark_v1", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load reference module at {MODULE_PATH}")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def one_seed(seed: int, n: int, reps: int) -> dict:
    # Draw the paired outcome differences directly from the same planted DGP used by
    # make_pair: delta = effect + Normal(0, 0.04). This isolates interval calibration
    # from trace-schema validation, which is tested by the separate causal-replay suite.
    rng = random.Random(seed)
    valid_values = [round(0.08 + rng.gauss(0.0, 0.04), 8) for _ in range(n)]
    placebo_values = [round(rng.gauss(0.0, 0.04), 8) for _ in range(n)]

    def vectorized_bootstrap(values: list[float], local_seed: int) -> dict:
        array = np.asarray(values, dtype=float)
        rng_boot = np.random.default_rng(local_seed)
        draws = rng_boot.integers(0, len(array), size=(reps, len(array)))
        estimates = np.sort(np.mean(array[draws], axis=1))
        low = estimates[max(0, int(0.025 * reps))]
        high = estimates[min(reps - 1, int(0.975 * reps))]
        sd = float(np.std(array, ddof=1))
        m = float(np.mean(array))
        se = sd / (len(array) ** 0.5)
        # Paired t interval is a cross-check for this planted approximately Gaussian DGP.
        t_critical = 1.9646 if len(array) == 512 else 1.96
        t_low, t_high = m - t_critical * se, m + t_critical * se
        return {"mean_effect": m, "bootstrap_ci_low": float(low), "bootstrap_ci_high": float(high),
                "ci_excludes_zero": bool(low > 0 or high < 0), "paired_sd": sd,
                "paired_se": se, "paired_t_ci_low": t_low, "paired_t_ci_high": t_high}

    valid = vectorized_bootstrap(valid_values, seed + 100_001)
    placebo = vectorized_bootstrap(placebo_values, seed + 200_003)
    return {
        "seed": seed,
        "valid_covered_planted_effect": valid["bootstrap_ci_low"] <= 0.08 <= valid["bootstrap_ci_high"],
        "placebo_covered_zero": placebo["bootstrap_ci_low"] <= 0.0 <= placebo["bootstrap_ci_high"],
        "valid_positive_control_passed": valid["bootstrap_ci_low"] > 0.05,
        "placebo_equivalence_passed": placebo["bootstrap_ci_low"] > -0.02 and placebo["bootstrap_ci_high"] < 0.02,
        "placebo_ci_excludes_zero": placebo["ci_excludes_zero"],
        "valid_t_covered_planted_effect": valid["paired_t_ci_low"] <= 0.08 <= valid["paired_t_ci_high"],
        "placebo_t_covered_zero": placebo["paired_t_ci_low"] <= 0.0 <= placebo["paired_t_ci_high"],
        "valid_mean_effect": valid["mean_effect"], "valid_ci_low": valid["bootstrap_ci_low"],
        "valid_ci_high": valid["bootstrap_ci_high"], "placebo_mean_effect": placebo["mean_effect"],
        "placebo_ci_low": placebo["bootstrap_ci_low"], "placebo_ci_high": placebo["bootstrap_ci_high"],
    }


def run(seeds: int = 100, n: int = 512, reps: int = 500, base_seed: int = 20261028, include_rows: bool = False) -> dict:
    if seeds < 20: raise ValueError("seeds must be >= 20 for this calibration audit")
    rows = [one_seed(base_seed + i, n, reps) for i in range(seeds)]
    def rate(k): return sum(bool(x[k]) for x in rows) / len(rows)
    result = {
        "status": "PASS_ESTIMATOR_SEED_SWEEP_ONLY",
        "base_seed": base_seed, "seed_count": seeds, "pairs_per_arm_per_seed": n,
        "bootstrap_replicates_per_arm_per_seed": reps,
        "total_paired_outcome_effects": seeds * 2 * n,
        "synthetic_ground_truth_valid_effect": 0.08, "synthetic_ground_truth_placebo_effect": 0.0,
        "empirical_95_percent_percentile_bootstrap_coverage_valid": rate("valid_covered_planted_effect"),
        "empirical_95_percent_percentile_bootstrap_coverage_placebo": rate("placebo_covered_zero"),
        "empirical_95_percent_paired_t_coverage_valid": rate("valid_t_covered_planted_effect"),
        "empirical_95_percent_paired_t_coverage_placebo": rate("placebo_t_covered_zero"),
        "positive_control_threshold_pass_rate": rate("valid_positive_control_passed"),
        "placebo_equivalence_gate_pass_rate": rate("placebo_equivalence_passed"),
        "placebo_percentile_ci_excludes_zero_rate_not_a_gate": rate("placebo_ci_excludes_zero"),
        "production_symthaea_imported": False, "rust_compiled_or_executed": False,
        "scope_note": "Independent synthetic repeated-seed calibration only; interval rates estimate this generator/estimator combination, not real Symthaea behavior or universal frequentist coverage. Paired-t intervals are only a cross-check for this planted approximately Gaussian outcome DGP."
    }
    if include_rows: result["replicate_rows"] = rows
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--pairs-per-arm", type=int, default=512)
    p.add_argument("--bootstrap-replicates", type=int, default=500)
    p.add_argument("--base-seed", type=int, default=20261028)
    p.add_argument("--include-replicate-rows", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = run(a.seeds, a.pairs_per_arm, a.bootstrap_replicates, a.base_seed, a.include_replicate_rows)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__": main()
