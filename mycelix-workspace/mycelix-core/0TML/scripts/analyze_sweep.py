#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Summarise label-skew sweep results and highlight best candidates."""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

DEFAULT_TARGET_DETECTION = 90.0
DEFAULT_TARGET_FP = 5.0


def load_runs(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    return data.get("runs", data)


def summarise(runs: List[Dict], target_detection: float, target_fp: float) -> str:
    lines: List[str] = []
    lines.append("aggregator,pogq_threshold,reputation_threshold,bft_ratio,mean_detection,mean_fp,passes")
    key_map: Dict[str, List[Dict]] = {}
    for run in runs:
        key = (
            run.get("aggregator", "coordinate_median"),
            float(run.get("pogq_threshold", 0.0)),
            float(run.get("reputation_threshold", 0.0)),
            float(run.get("bft_ratio", 0.0)),
        )
        key_map.setdefault(key, []).append(run)

    best: List[tuple] = []
    for key, rows in sorted(key_map.items()):
        det = mean(float(r.get("final_detection_rate", 0.0)) for r in rows)
        fp = mean(float(r.get("final_false_positive_rate", 0.0)) for r in rows)
        passes = det >= target_detection and fp <= target_fp
        key_str = ",".join([str(val) for val in key])
        lines.append(f"{key_str},{det:.2f},{fp:.2f},{passes}")
        best.append((passes, fp, -det, key))

    best.sort()
    best_candidates = [entry for entry in best if entry[0]]

    output = ["\nSummary (target detection >= %.2f%%, FP <= %.2f%%)" % (target_detection, target_fp)]
    output.extend(lines)
    if best_candidates:
        output.append("\nBest passing configuration: %s" % (best_candidates[0][3],))
    else:
        output.append("\nNo configuration met the target thresholds")
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSON sweep file (label_skew_sweep_*.json)")
    parser.add_argument("--target-detection", type=float, default=DEFAULT_TARGET_DETECTION,
                        help="Minimum detection rate required (%%)")
    parser.add_argument("--target-fp", type=float, default=DEFAULT_TARGET_FP,
                        help="Maximum false positive rate allowed (%%)")
    args = parser.parse_args()

    runs = load_runs(args.path)
    print(summarise(runs, args.target_detection, args.target_fp))


if __name__ == "__main__":
    main()
