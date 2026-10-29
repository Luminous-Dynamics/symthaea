#!/usr/bin/env python3
"""Exploratory Monte Carlo for the Rust anytime-valid Bernoulli change-point mixture.

This is research benchmarking code, not production monitoring. Its false-alarm guarantee comes
from the e-process construction and Rust tests, not from a finite Monte Carlo estimate. The seed,
scenario order, NumPy version and trial count are included in the CSV for reproducibility.
"""
import argparse
import csv
import math
import sys

import numpy as np

NULL_LOWER = 0.48
NULL_UPPER = 0.52
UPWARD_ALTERNATIVES = (0.55, 0.60, 0.70, 0.80, 0.90)
DOWNWARD_ALTERNATIVES = (0.45, 0.40, 0.30, 0.20, 0.10)
HAZARDS = (0.04, 0.01, 0.0025)
ALPHA = 0.05 / 12.0
COMPONENTS = tuple(
    (hazard, alternative, null_edge)
    for hazard in HAZARDS
    for alternatives, null_edge in (
        (UPWARD_ALTERNATIVES, NULL_UPPER),
        (DOWNWARD_ALTERNATIVES, NULL_LOWER),
    )
    for alternative in alternatives
)


def wilson_interval(successes, trials, z=1.959963984540054):
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def scenario_specs(seed):
    return (
        ("stationary_p050", "stationary_null", 0.50, None, seed + 0),
        ("stable_edge_p048", "stationary_null", 0.48, None, seed + 1),
        ("stable_edge_p052", "stationary_null", 0.52, None, seed + 2),
        ("selection_bias_p074", "selection_bias_control", 0.74, None, seed + 3),
        ("step_up_055", "target_shift", 0.50, 0.55, seed + 4),
        ("step_up_060", "target_shift", 0.50, 0.60, seed + 5),
        ("step_up_070", "target_shift", 0.50, 0.70, seed + 6),
        ("step_up_080", "target_shift", 0.50, 0.80, seed + 7),
        ("step_down_045", "target_shift", 0.50, 0.45, seed + 8),
        ("step_down_040", "target_shift", 0.50, 0.40, seed + 9),
        ("step_down_030", "target_shift", 0.50, 0.30, seed + 10),
        ("step_down_020", "target_shift", 0.50, 0.20, seed + 11),
    )


def run_scenario(name, scenario_class, pre_p, post_p, trials, horizon, change_after, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    count = len(COMPONENTS)
    log_survival = np.asarray([math.log1p(-h) for h, _, _ in COMPONENTS])
    log_hazard = np.asarray([math.log(h) for h, _, _ in COMPONENTS])
    log_lr_success = np.asarray(
        [math.log(q / null) for _, q, null in COMPONENTS]
    )
    log_lr_failure = np.asarray(
        [math.log((1.0 - q) / (1.0 - null)) for _, q, null in COMPONENTS]
    )

    log_active = np.full((trials, count), -np.inf, dtype=np.float64)
    first_alarm = np.zeros(trials, dtype=np.int32)
    log_threshold = -math.log(ALPHA)

    for step in range(1, horizon + 1):
        probability = post_p if post_p is not None and step > change_after else pre_p
        event = rng.random(trials) < probability
        log_lr = np.where(event[:, None], log_lr_success[None, :], log_lr_failure[None, :])
        log_start_weight = log_hazard + (step - 1) * log_survival
        log_active = log_lr + np.logaddexp(log_active, log_start_weight[None, :])
        log_tail = step * log_survival
        log_component_e = np.logaddexp(log_active, log_tail[None, :])
        log_e = np.logaddexp.reduce(log_component_e, axis=1) - math.log(count)
        first_alarm[(first_alarm == 0) & (log_e >= log_threshold)] = step

    alarm = first_alarm > 0
    no_alarm = first_alarm == 0
    any_alarm_ci = wilson_interval(int(alarm.sum()), trials)
    row = {
        "scenario": name,
        "scenario_class": scenario_class,
        "trials": trials,
        "horizon": horizon,
        "change_after": change_after if post_p is not None else "NA",
        "pre_rate": f"{pre_p:.6f}",
        "post_rate": f"{post_p:.6f}" if post_p is not None else "NA",
        "any_alarm_count": int(alarm.sum()),
        "any_alarm_estimate": float(alarm.mean()),
        "any_alarm_ci_low": any_alarm_ci[0],
        "any_alarm_ci_high": any_alarm_ci[1],
        "pre_change_alarm_count": int(((first_alarm > 0) & (first_alarm <= change_after)).sum()) if post_p is not None else "NA",
        "pre_change_alarm_estimate": float(((first_alarm > 0) & (first_alarm <= change_after)).mean()) if post_p is not None else "NA",
        "post_change_detection_count": int((first_alarm > change_after).sum()) if post_p is not None else "NA",
        "post_change_detection_estimate": float((first_alarm > change_after).mean()) if post_p is not None else "NA",
        "post_change_detection_ci_low": "NA",
        "post_change_detection_ci_high": "NA",
        "undetected_count": int(no_alarm.sum()),
        "mean_post_change_delay": "NA",
        "median_post_change_delay": "NA",
        "p90_post_change_delay": "NA",
        "seed": seed,
        "prng": "numpy.PCG64",
        "numpy_version": np.__version__,
    }
    if post_p is not None:
        detected = first_alarm > change_after
        count_detected = int(detected.sum())
        detection_ci = wilson_interval(count_detected, trials)
        row["post_change_detection_ci_low"] = detection_ci[0]
        row["post_change_detection_ci_high"] = detection_ci[1]
        if count_detected:
            delays = first_alarm[detected] - change_after
            row["mean_post_change_delay"] = float(delays.mean())
            row["median_post_change_delay"] = float(np.quantile(delays, 0.50, method="higher"))
            row["p90_post_change_delay"] = float(np.quantile(delays, 0.90, method="higher"))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--change-after", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", default="-", help="CSV destination or '-' for stdout")
    args = parser.parse_args()
    if args.trials <= 0 or args.horizon <= 0 or not 0 <= args.change_after < args.horizon:
        parser.error("require trials>0, horizon>0, and 0 <= change-after < horizon")

    rows = [
        run_scenario(
            name, scenario_class, pre_p, post_p, args.trials, args.horizon,
            args.change_after, scenario_seed,
        )
        for name, scenario_class, pre_p, post_p, scenario_seed in scenario_specs(args.seed)
    ]
    fields = tuple(rows[0].keys())
    stream = sys.stdout if args.output == "-" else open(args.output, "w", newline="", encoding="utf-8")
    try:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if stream is not sys.stdout:
            stream.close()


if __name__ == "__main__":
    main()
