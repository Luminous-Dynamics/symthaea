#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Train a ByzantineDetector from logged MATL features."""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "0TML/src"))
sys.path.insert(0, str(ROOT / "0TML"))

from zerotrustml.ml import ByzantineDetector, IsolationAnomalyDetector

def load_dataset(log_path: Path, include_committee: bool) -> tuple[np.ndarray, np.ndarray]:
    feats: List[List[float]] = []
    labels: List[int] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            feat = record.get("features")
            label = record.get("true_label")
            committee_score = record.get("committee_score", 0.0)
            if not isinstance(feat, dict) or not isinstance(label, int):
                continue
            vector = [
                float(feat.get("pogq", 0.0)),
                float(feat.get("tcdm", 0.0)),
                float(feat.get("zscore", 0.0)),
                float(feat.get("entropy", 0.0)),
                float(feat.get("gradient_norm", 0.0)),
            ]
            if include_committee:
                vector.append(float(committee_score))
            vector.append(float(feat.get("prev_alignment", 0.0)))
            feats.append(vector)
            labels.append(label)
    if not feats:
        raise ValueError("No samples loaded from log")
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Train MATL detector from log data")
    parser.add_argument("--log", default="0TML/tests/results/ml_training_data.jsonl", help="Path to feature log")
    parser.add_argument("--output", default="models/byzantine_detector_logged", help="Directory to store trained detector")
    parser.add_argument("--no-committee", dest="include_committee", action="store_false", help="Exclude committee score from features")
    parser.add_argument("--anomaly-contamination", type=float, default=0.08, help="Contamination rate for IsolationForest anomaly detector")
    parser.set_defaults(include_committee=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    X, y = load_dataset(log_path, args.include_committee)
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    positives = int(y.sum())
    print(f"  Positives: {positives}  Negatives: {len(y) - positives}")

    detector = ByzantineDetector(model="ensemble", pogq_low_threshold=0.3, pogq_high_threshold=0.7)
    detector.train(
        X,
        y,
        validate=True,
        validation_split=0.2,
        random_state=42,
        min_detection_rate=0.85,
        max_false_positive_rate=0.2,
    )

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    detector.save(str(output_path))
    print(f"✅ Saved detector to {output_path}")

    honest_mask = y == 0
    if honest_mask.sum() >= 50:
        anomaly = IsolationAnomalyDetector(contamination=args.anomaly_contamination)
        anomaly.fit(X[honest_mask])
        anomaly_path = output_path / "anomaly_iforest.joblib"
        anomaly.save(anomaly_path)
        print(f"✅ Saved IsolationForest anomaly detector to {anomaly_path}")
    else:
        print("⚠️  Not enough honest samples to train anomaly detector; skipping IsolationForest export.")


if __name__ == "__main__":
    main()
