#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Run the 30% BFT harness across multiple attack types and report metrics.

Usage:
    nix develop -c python 0TML/scripts/run_bft_matrix.py

Running via the Nix shell ensures numpy/pytorch can locate dynamic libraries
without relying on the Poetry virtualenv.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '0TML'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import numpy as np

os.environ.setdefault("RUN_30_BFT", "1")

from tests.test_30_bft_validation import run_30_bft_test, create_byzantine_gradient

ATTACK_TYPES = [
    "noise",
    "sign_flip",
    "zero",
    "random",
    "backdoor",
    "adaptive",
    "scaled_sign_flip",
    "stealth_backdoor",
    "temporal_drift",
    "entropy_smoothing",
]

DATASET_DISTRIBUTIONS = {
    "cifar10": ["iid", "label_skew"],
    "emnist_balanced": ["iid", "label_skew"],
    "breast_cancer": ["iid"],
}

BFT_RATIOS = [0.3, 0.4]

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results" / "bft-matrix"


def main():
    results = []
    os.environ["RUN_30_BFT"] = "1"
    for ratio in BFT_RATIOS:
        os.environ["BFT_RATIO"] = str(ratio)
        for dataset, distributions in DATASET_DISTRIBUTIONS.items():
            for attack in ATTACK_TYPES:
                for distribution in distributions:
                    print(
                        f"\n=== Running BFT harness (dataset={dataset}, distribution={distribution}, attack={attack}, ratio={ratio:.2f}) ==="
                    )
                    original = create_byzantine_gradient

                    def patched(honest_gradient, *args, **kwargs):
                        kwargs["attack_type"] = attack
                        return original(honest_gradient, *args, **kwargs)

                    try:
                        import tests.test_30_bft_validation as harness
                        harness.create_byzantine_gradient = patched
                        result = run_30_bft_test(
                            dataset_name=dataset,
                            distribution=distribution,
                            attack_suite=[attack],
                        )
                    finally:
                        harness.create_byzantine_gradient = original

                    results.append({
                        "dataset": dataset,
                        "distribution": distribution,
                        "attack_type": attack,
                        "bft_ratio": ratio,
                        **result,
                    })

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    serializable_runs = [_to_serializable(r) for r in results]
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "runs": serializable_runs,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outfile = RESULTS_DIR / f"matrix_{timestamp}.json"
    outfile.write_text(json.dumps(output, indent=2))
    print("\nSummary:")
    _write_latest_summary(serializable_runs)
    print(f"\nDetailed JSON written to {outfile}")
    os.environ.pop("BFT_RATIO", None)
    os.environ.pop("RUN_30_BFT", None)


def _write_latest_summary(results):
    from collections import defaultdict

    summary = defaultdict(list)
    for row in results:
        key = (row["dataset"], row["distribution"], row.get("bft_ratio", 0.3))
        summary[key].append(row)

    lines = [f"# BFT Matrix Summary ({datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')})", ""]
    for (dataset, distribution, ratio), runs in sorted(summary.items()):
        total = len(runs)
        successes = sum(1 for r in runs if r.get("success"))
        lines.append(f"## {dataset} — {distribution} — ratio={ratio:.2f}\n")
        avg_det = np.mean([r.get("final_detection_rate", 0.0) for r in runs]) if runs else 0.0
        avg_fp = np.mean([r.get("final_false_positive_rate", 0.0) for r in runs]) if runs else 0.0
        lines.append(f"Overall: **{successes} / {total} pass** (avg detection {avg_det:.1f}%, avg false positives {avg_fp:.1f}%)\n")
        lines.append("| Attack | Pass | Fail | Detection % | FP % |")
        lines.append("| --- | --- | --- | --- | --- |")
        for attack in sorted({r["attack_type"] for r in runs}):
            attack_runs = [r for r in runs if r["attack_type"] == attack]
            pass_count = sum(1 for r in attack_runs if r.get("success"))
            fail_count = len(attack_runs) - pass_count
            det = np.mean([r.get("final_detection_rate", 0.0) for r in attack_runs])
            fp = np.mean([r.get("final_false_positive_rate", 0.0) for r in attack_runs])
            lines.append(f"| {attack} | {pass_count} | {fail_count} | {det:.1f}% | {fp:.1f}% |")
        lines.append("")

    summary_path = RESULTS_DIR / "latest_summary.md"
    summary_path.write_text("\n".join(lines))
    print(f"Summary written to {summary_path}")


def _to_serializable(value):
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value


if __name__ == "__main__":
    main()
