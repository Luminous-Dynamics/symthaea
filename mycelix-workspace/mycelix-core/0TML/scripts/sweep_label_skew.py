#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Quick sweep for label-skew tuning parameters."""

import json
import os
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_30_bft_validation import run_30_bft_test


DATASET = "cifar10"
DISTRIBUTION = "label_skew"

POGQ_VALUES = [0.30, 0.32, 0.35, 0.38]
REP_THRESHOLDS = [0.02, 0.05, 0.08]
AGGREGATORS = ["coordinate_median", "trimmed_mean"]

RESULTS_DIR = Path("results") / "bft-matrix"


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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for aggregator in AGGREGATORS:
        os.environ["ROBUST_AGGREGATOR"] = aggregator

        for pogq in POGQ_VALUES:
            for rep in REP_THRESHOLDS:
                os.environ["BFT_POGQ_OVERRIDE"] = str(pogq)
                os.environ["BFT_REPUTATION_OVERRIDE"] = str(rep)

                print(
                    f"\n==> dataset={DATASET}, distribution={DISTRIBUTION}, "
                    f"aggregator={aggregator}, pogq={pogq:.2f}, rep={rep:.2f}"
                )
                result = run_30_bft_test(
                    dataset_name=DATASET,
                    distribution=DISTRIBUTION,
                    attack_suite=["noise", "sign_flip", "zero", "random", "backdoor", "adaptive"],
                )

                summary.append({
                    "dataset": DATASET,
                    "distribution": DISTRIBUTION,
                    "aggregator": aggregator,
                    "pogq_threshold": pogq,
                    "reputation_threshold": rep,
                    **result,
                })

    for key in ["ROBUST_AGGREGATOR", "BFT_POGQ_OVERRIDE", "BFT_REPUTATION_OVERRIDE"]:
        os.environ.pop(key, None)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outfile = RESULTS_DIR / f"label_skew_sweep_{timestamp}.json"
    outfile.write_text(json.dumps({"runs": [_to_serializable(entry) for entry in summary]}, indent=2))
    print(f"\nSweep written to {outfile}")


if __name__ == "__main__":
    os.environ["RUN_30_BFT"] = "1"
    main()
