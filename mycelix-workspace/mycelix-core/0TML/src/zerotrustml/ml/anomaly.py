# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
IsolationForest-based anomaly detector used alongside the MATL classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyModelBundle:
    model: IsolationForest
    threshold: float


class IsolationAnomalyDetector:
    """Simple wrapper around IsolationForest with persistence helpers."""

    def __init__(self, contamination: float = 0.08, random_state: int = 42):
        self.model = IsolationForest(
            contamination=min(max(contamination, 0.001), 0.5),
            random_state=random_state,
            n_estimators=200,
            max_samples="auto",
        )
        self.threshold: float = 0.0
        self.is_trained: bool = False

    def fit(self, samples: np.ndarray) -> None:
        if samples.ndim != 2:
            raise ValueError("samples must be a 2D array.")
        self.model.fit(samples)
        scores = self.model.decision_function(samples)
        # Use the 5th percentile as anomaly cut-off by default.
        self.threshold = float(np.percentile(scores, 5))
        self.is_trained = True

    def score(self, sample: Iterable[float]) -> float:
        if not self.is_trained:
            raise RuntimeError("IsolationAnomalyDetector must be trained before scoring.")
        arr = np.asarray(list(sample), dtype=np.float32).reshape(1, -1)
        decision = self.model.decision_function(arr)[0]
        return float(decision)

    def is_outlier(self, sample: Iterable[float], threshold: Optional[float] = None) -> bool:
        cutoff = self.threshold if threshold is None else threshold
        return self.score(sample) < cutoff

    def save(self, path: Path | str) -> None:
        bundle = AnomalyModelBundle(model=self.model, threshold=self.threshold)
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, path: Path | str) -> "IsolationAnomalyDetector":
        bundle: AnomalyModelBundle = joblib.load(path)
        instance = cls()
        instance.model = bundle.model
        instance.threshold = bundle.threshold
        instance.is_trained = True
        return instance


__all__ = ["IsolationAnomalyDetector"]
