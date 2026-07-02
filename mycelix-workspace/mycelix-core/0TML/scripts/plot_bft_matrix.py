#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Plot detection and false-positive trends from the aggregated BFT matrix.

Usage:
    nix develop --command poetry run python 0TML/scripts/plot_bft_matrix.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    matrix_path = Path("0TML/tests/results/bft_matrix.json")
    if not matrix_path.exists():
        raise SystemExit(
            f"{matrix_path} not found. Generate it first with "
            "`python scripts/generate_bft_matrix.py`."
        )

    payload = json.loads(matrix_path.read_text())
    rows = sorted(payload["matrix"], key=lambda r: r["byzantine_percent"])

    percents = [row["byzantine_percent"] for row in rows]
    detection = [row["detection_rate"] * 100 for row in rows]
    false_pos = [row["false_positive_rate"] * 100 for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(percents, detection, marker="o", label="Detection Rate")
    plt.plot(percents, false_pos, marker="o", label="False Positive Rate")
    plt.axvline(33, color="gray", linestyle="--", linewidth=1, label="Classical BFT Limit (33%)")
    plt.title("BFT Matrix Trend (PoGQ=0.35)")
    plt.xlabel("% Byzantine Nodes")
    plt.ylabel("Rate (%)")
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    output_dir = Path("0TML/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bft_detection_trend.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"✅ Saved trend plot to {output_path}")


if __name__ == "__main__":
    main()
