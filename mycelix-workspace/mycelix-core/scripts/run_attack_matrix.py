#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Run the 30% BFT harness across individual attack types and ratios.

Usage:
    nix develop --command poetry run python scripts/run_attack_matrix.py

Outputs:
    - Per-run JSON reports in 0TML/tests/results/bft_attack_<attack>_<ratio>.json
    - Aggregate summary 0TML/tests/results/bft_attack_matrix.json
"""

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUN_30_BFT", "1")
os.environ.setdefault("USE_ML_DETECTOR", "1")
sys.path.insert(0, str(ROOT / "0TML"))
sys.path.insert(0, str(ROOT))

from tests.test_30_bft_validation import run_30_bft_test, create_byzantine_gradient  # type: ignore


def _parse_list(value: str | None, default):
    if not value:
        return default
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items if items else default


def _parse_float_list(value: str | None, default):
    if not value:
        return default
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(float(item))
        except ValueError:
            continue
    return result if result else default


ATTACK_TYPES = _parse_list(
    os.environ.get("ATTACK_TYPES"),
    [
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
    ],
)
BFT_RATIOS = _parse_float_list(os.environ.get("BFT_RATIOS"), [0.33, 0.40, 0.50])
DATASET = os.environ.get("ATTACK_DATASET", "cifar10")
DISTRIBUTIONS = _parse_list(
    os.environ.get("ATTACK_DISTRIBUTIONS"),
    [os.environ.get("ATTACK_DISTRIBUTION", "iid")],
)
LABEL_SKEW_ALPHAS = _parse_float_list(os.environ.get("LABEL_SKEW_ALPHAS"), [])
if not LABEL_SKEW_ALPHAS:
    alpha_single = os.environ.get("LABEL_SKEW_ALPHA")
    if alpha_single:
        try:
            LABEL_SKEW_ALPHAS = [float(alpha_single)]
        except ValueError:
            LABEL_SKEW_ALPHAS = []

RESULTS_DIR = Path("0TML/tests/results")


@contextmanager
def patch_attack(attack_type: str):
    """Temporarily override create_byzantine_gradient to force an attack."""
    original = create_byzantine_gradient

    def patched(honest_gradient, *args, **kwargs):
        kwargs["attack_type"] = attack_type
        return original(honest_gradient, *args, **kwargs)

    import tests.test_30_bft_validation as harness

    harness.create_byzantine_gradient = patched
    try:
        yield
    finally:
        harness.create_byzantine_gradient = original


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    aggregate = []

    for ratio in BFT_RATIOS:
        os.environ["RUN_30_BFT"] = "1"
        os.environ["BFT_RATIO"] = str(ratio)

        for distribution in DISTRIBUTIONS:
            os.environ["BFT_DISTRIBUTION"] = distribution
            alpha_values = LABEL_SKEW_ALPHAS if distribution == "label_skew" and LABEL_SKEW_ALPHAS else [None]

            for alpha in alpha_values:
                if alpha is not None:
                    os.environ["BFT_LABEL_SKEW_ALPHA"] = str(alpha)
                else:
                    os.environ.pop("BFT_LABEL_SKEW_ALPHA", None)

                for attack in ATTACK_TYPES:
                    label = f"attack={attack}, ratio={ratio:.2f}, distribution={distribution}"
                    if alpha is not None:
                        label += f", alpha={alpha}"
                    print(f"\n=== Running {label} ===")
                    with patch_attack(attack):
                        result = run_30_bft_test(
                            dataset_name=DATASET,
                            distribution=distribution,
                            attack_suite=[attack],
                        )

                    suffix = f"{distribution}_{attack}_{int(ratio*100)}"
                    if alpha is not None:
                        suffix += f"_alpha{int(alpha*1000)}"
                    outfile = RESULTS_DIR / f"bft_attack_{suffix}.json"
                    serializable = _to_serializable(result)
                    if isinstance(serializable, dict):
                        serializable.setdefault("dataset", DATASET)
                        serializable.setdefault("distribution", distribution)
                        if alpha is not None:
                            serializable["label_skew_alpha"] = alpha
                        serializable.setdefault("attack_type", attack)
                        serializable.setdefault("bft_ratio", ratio)
                    with outfile.open("w", encoding="utf-8") as handle:
                        json.dump(serializable, handle, indent=2)
                    print(f"  → saved {outfile}")

                    aggregate_entry = {
                        "attack_type": attack,
                        "bft_ratio": ratio,
                        "dataset": DATASET,
                        "distribution": distribution,
                        **serializable,
                    }
                    if alpha is not None:
                        aggregate_entry["label_skew_alpha"] = alpha
                    aggregate.append(aggregate_entry)

    os.environ.pop("RUN_30_BFT", None)
    os.environ.pop("BFT_RATIO", None)
    os.environ.pop("BFT_DISTRIBUTION", None)
    os.environ.pop("BFT_LABEL_SKEW_ALPHA", None)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": aggregate,
    }
    matrix_path = RESULTS_DIR / "bft_attack_matrix.json"
    matrix_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n✅ Wrote aggregate matrix to {matrix_path}")


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
