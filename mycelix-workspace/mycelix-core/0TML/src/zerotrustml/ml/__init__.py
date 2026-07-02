# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ML-Enhanced Byzantine Detection Module
=======================================

This module implements machine learning classifiers for enhanced Sybil and Byzantine
node detection, achieving 95-98% detection rates with ≤3% false positives.

Components:
-----------
- feature_extractor.py: Composite feature vector generation (PoGQ, TCDM, z-score, entropy)
- classifiers.py: SVM and Random Forest models with ensemble logic
- evaluator.py: Comprehensive metrics, ROC/PR curves, confusion matrices
- detector.py: High-level detection API with PoGQ-guided fallback logic

Target Performance:
------------------
- Detection Rate: 95-98%
- False Positive Rate: ≤3%
- Inference Time: <10ms per node
- Training Data: 1000+ labeled gradient samples

Integration:
-----------
Pluggable into existing RB-BFT + PoGQ pipeline:
- Layer 1: PoGQ (fast, high-precision baseline)
- Layer 2: ML Classifier (catches sophisticated attacks)
- Layer 3: RB-BFT Reputation (temporal decay)

Usage:
------
```python
from zerotrustml.ml import ByzantineDetector

detector = ByzantineDetector(model='ensemble')  # or 'svm', 'rf'
result = detector.classify_node(gradient, node_id)
# Returns: 'HONEST' or 'BYZANTINE'
```
"""

from .feature_extractor import FeatureExtractor, CompositeFeatures, extract_features_batch
from .classifiers import SVMClassifier, RandomForestClassifier, EnsembleClassifier
from .evaluator import DetectionEvaluator
from .detector import ByzantineDetector
from .anomaly import IsolationAnomalyDetector

__all__ = [
    'FeatureExtractor',
    'CompositeFeatures',
    'extract_features_batch',
    'SVMClassifier',
    'RandomForestClassifier',
    'EnsembleClassifier',
    'DetectionEvaluator',
    'ByzantineDetector',
    'IsolationAnomalyDetector',
]

__version__ = '0.1.0'
