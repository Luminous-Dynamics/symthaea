#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Export BFT matrix results as Prometheus-compatible metrics."""

import argparse
import json
from pathlib import Path


def format_labels(labels):
    return ",".join(f'{key}="{value}"' for key, value in labels.items())


def main():
    parser = argparse.ArgumentParser(description="Export MATL BFT metrics for Prometheus scraping.")
    parser.add_argument("--matrix", default="0TML/tests/results/bft_attack_matrix.json", help="Path to bft_attack_matrix.json")
    parser.add_argument("--output", default="artifacts/matl_metrics.prom", help="Output file for Prometheus metrics")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        raise SystemExit(f"Matrix file not found: {matrix_path}")

    data = json.loads(matrix_path.read_text())
    runs = data.get("runs", [])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# HELP matl_detection_rate_percent Final detection rate per attack scenario.",
        "# TYPE matl_detection_rate_percent gauge",
    ]

    for run in runs:
        labels = {
            "attack": run.get("attack_type", "unknown"),
            "distribution": run.get("distribution", "unknown"),
            "ratio": f"{int(round(run.get('bft_ratio', 0.0) * 100))}",
        }
        alpha = run.get("label_skew_alpha")
        if alpha is not None:
            labels["alpha"] = f"{alpha}"
        label_str = format_labels(labels)
        detection = run.get("final_detection_rate", 0.0)
        false_positive = run.get("final_false_positive_rate", 0.0)
        success = 1 if run.get("success") else 0

        lines.append(f"matl_detection_rate_percent{{{label_str}}} {detection}")
        lines.append(f"matl_false_positive_rate_percent{{{label_str}}} {false_positive}")
        lines.append(f"matl_regression_success{{{label_str}}} {success}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ Wrote Prometheus metrics to {output_path}")


if __name__ == "__main__":
    main()
