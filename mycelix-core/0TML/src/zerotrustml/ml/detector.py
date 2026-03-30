# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine Detector - High-Level API
===================================

Unified interface for Byzantine node detection combining:
- Fast PoGQ-based decisions for clear cases
- ML classifier for sophisticated attacks
- Simple API suitable for Rust FFI port

Usage:
------
```python
from zerotrustml.ml import ByzantineDetector

# Initialize detector
detector = ByzantineDetector(model='ensemble')

# Train on labeled data
detector.train(X_train, y_train)

# Detect Byzantine nodes
result = detector.classify_node(gradient, node_id)
# Returns: 'HONEST' or 'BYZANTINE'

# With probability
result, confidence = detector.classify_node_with_confidence(gradient, node_id)
# Returns: ('BYZANTINE', 0.95) or ('HONEST', 0.82)
```

Decision Logic:
--------------
1. Extract composite features (PoGQ, TCDM, Z-score, Entropy)
2. If PoGQ < 0.3 → Fast reject (BYZANTINE)
3. If PoGQ > 0.7 → Fast accept (HONEST)
4. Else (0.3 ≤ PoGQ ≤ 0.7) → ML classifier decision
"""

from typing import Literal, Optional, Tuple, List
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from .feature_extractor import FeatureExtractor, CompositeFeatures, extract_features_batch
from .classifiers import create_classifier, SVMClassifier, RandomForestClassifier, EnsembleClassifier
from .evaluator import DetectionEvaluator, DetectionMetrics


class ByzantineDetector:
    """
    High-level Byzantine detection system with PoGQ-guided ML inference.

    Combines fast PoGQ heuristics with ML classifier for maximum accuracy
    and efficiency.
    """

    def __init__(
        self,
        model: Literal['svm', 'rf', 'ensemble'] = 'ensemble',
        pogq_low_threshold: float = 0.3,
        pogq_high_threshold: float = 0.7,
        ml_threshold: float = 0.5,
        history_window: int = 5,
        **model_kwargs,
    ):
        """
        Args:
            model: Type of ML classifier ('svm', 'rf', or 'ensemble')
            pogq_low_threshold: PoGQ below this → instant Byzantine classification
            pogq_high_threshold: PoGQ above this → instant Honest classification
            ml_threshold: ML probability threshold for Byzantine classification
            history_window: Number of rounds to retain for TCDM
            **model_kwargs: Additional parameters for the classifier
        """
        self.model_type = model
        self.pogq_low = pogq_low_threshold
        self.pogq_high = pogq_high_threshold
        self.ml_threshold = ml_threshold

        # Initialize components
        self.feature_extractor = FeatureExtractor(history_window=history_window)
        self.classifier = create_classifier(model, **model_kwargs)

        # Training state
        self.is_trained = False

        # Statistics tracking
        self.stats = {
            'total_classifications': 0,
            'pogq_fast_reject': 0,     # PoGQ < 0.3
            'pogq_fast_accept': 0,     # PoGQ > 0.7
            'ml_classifications': 0,   # 0.3 ≤ PoGQ ≤ 0.7
        }

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validate: bool = True,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        *,
        validation_split: float = 0.2,
        random_state: Optional[int] = 42,
        min_detection_rate: float = 0.95,
        max_false_positive_rate: float = 0.03,
    ) -> Optional[DetectionMetrics]:
        """
        Train the ML classifier on labeled data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=honest, 1=Byzantine)
            validate: Whether to compute validation metrics
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            validation_split: Fraction of training samples held out for validation
                when explicit validation data is not provided.
            random_state: Seed used for deterministic splits (None → random).
            min_detection_rate: Minimum acceptable detection rate; raise ValueError
                when validation underperforms this bound.
            max_false_positive_rate: Maximum acceptable false-positive rate; raise
                ValueError when validation exceeds this bound.

        Returns:
            DetectionMetrics if validate=True, else None
        """
        X_train = X
        y_train = y
        X_eval = X_val
        y_eval = y_val

        # Auto-generate validation split when requested
        if validate:
            if X_eval is None or y_eval is None:
                if validation_split <= 0 or validation_split >= 1:
                    raise ValueError(
                        "validation_split must be within (0, 1) when validation "
                        "data is not provided."
                    )
                X_train, X_eval, y_train, y_eval = train_test_split(
                    X,
                    y,
                    test_size=validation_split,
                    stratify=y,
                    random_state=random_state,
                )
            elif len(X_eval) == 0:
                raise ValueError("Validation data provided but empty.")

        # Train classifier
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        # Optional validation
        if validate:
            evaluator = DetectionEvaluator()

            if X_eval is None or y_eval is None:
                raise ValueError(
                    "Validation requested but no validation data is available."
                )

            y_pred = self.classifier.predict(X_eval)
            y_proba = self.classifier.predict_proba(X_eval)

            metrics = evaluator.evaluate(
                y_eval,
                y_pred,
                y_proba,
                prefix=f'{self.model_type}_val_',
            )

            print(f"\n{self.model_type.upper()} Training Complete!")
            print(metrics)

            # Enforce published targets so regressions surface immediately.
            if min_detection_rate is not None and metrics.detection_rate < min_detection_rate:
                raise ValueError(
                    f"{self.model_type} detection rate "
                    f"{metrics.detection_rate:.3f} fell below target "
                    f"{min_detection_rate:.3f}"
                )
            if max_false_positive_rate is not None and metrics.false_positive_rate > max_false_positive_rate:
                raise ValueError(
                    f"{self.model_type} false positive rate "
                    f"{metrics.false_positive_rate:.3f} exceeded target "
                    f"{max_false_positive_rate:.3f}"
                )

            return metrics

        return None

    def classify_node(
        self,
        gradient: np.ndarray,
        node_id: int,
        round_num: int = 0,
        all_gradients: Optional[List[np.ndarray]] = None,
    ) -> str:
        """
        Classify a single node's gradient as HONEST or BYZANTINE.

        Decision logic:
        1. Extract features (PoGQ, TCDM, Z-score, Entropy)
        2. Fast path: PoGQ < 0.3 → BYZANTINE
        3. Fast path: PoGQ > 0.7 → HONEST
        4. Slow path: Use ML classifier for borderline cases

        Args:
            gradient: Gradient vector submitted by node
            node_id: ID of the node
            round_num: Current training round
            all_gradients: All gradients in round (for global stats)

        Returns:
            'HONEST' or 'BYZANTINE'
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            gradient=gradient,
            node_id=node_id,
            round_num=round_num,
            all_gradients=all_gradients,
        )

        self.stats['total_classifications'] += 1

        # Fast path 1: PoGQ too low → Byzantine
        if features.pogq_score < self.pogq_low:
            self.stats['pogq_fast_reject'] += 1
            return 'BYZANTINE'

        # Fast path 2: PoGQ high enough → Honest
        if features.pogq_score > self.pogq_high:
            self.stats['pogq_fast_accept'] += 1
            return 'HONEST'

        # Slow path: ML classifier for borderline cases
        if not self.is_trained:
            # Fallback if classifier not trained: use PoGQ threshold at 0.5
            return 'BYZANTINE' if features.pogq_score < 0.5 else 'HONEST'

        self.stats['ml_classifications'] += 1

        # Run ML classifier
        feature_vector = features.to_array().reshape(1, -1)
        prediction = self.classifier.predict(feature_vector)[0]

        return 'BYZANTINE' if prediction == 1 else 'HONEST'

    def classify_node_with_confidence(
        self,
        gradient: np.ndarray,
        node_id: int,
        round_num: int = 0,
        all_gradients: Optional[List[np.ndarray]] = None,
    ) -> Tuple[str, float]:
        """
        Classify node and return confidence score.

        Args:
            gradient: Gradient vector
            node_id: Node ID
            round_num: Current round
            all_gradients: All gradients in round

        Returns:
            Tuple of (classification, confidence)
            Example: ('BYZANTINE', 0.95) or ('HONEST', 0.82)
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            gradient=gradient,
            node_id=node_id,
            round_num=round_num,
            all_gradients=all_gradients,
        )

        self.stats['total_classifications'] += 1

        # Fast path 1: PoGQ too low → Byzantine (high confidence)
        if features.pogq_score < self.pogq_low:
            self.stats['pogq_fast_reject'] += 1
            confidence = 1.0 - features.pogq_score / self.pogq_low
            return ('BYZANTINE', float(confidence))

        # Fast path 2: PoGQ high enough → Honest (high confidence)
        if features.pogq_score > self.pogq_high:
            self.stats['pogq_fast_accept'] += 1
            confidence = (features.pogq_score - self.pogq_high) / (1.0 - self.pogq_high)
            return ('HONEST', float(confidence))

        # Slow path: ML classifier
        if not self.is_trained:
            # Fallback without confidence
            result = 'BYZANTINE' if features.pogq_score < 0.5 else 'HONEST'
            return (result, 0.5)

        self.stats['ml_classifications'] += 1

        # Get probability from classifier
        feature_vector = features.to_array().reshape(1, -1)
        proba = self.classifier.predict_proba(feature_vector)[0]

        # proba[1] = P(Byzantine), proba[0] = P(Honest)
        if proba[1] >= self.ml_threshold:
            return ('BYZANTINE', float(proba[1]))
        else:
            return ('HONEST', float(proba[0]))

    def classify_batch(
        self,
        gradients: List[np.ndarray],
        node_ids: List[int],
        round_num: int = 0,
    ) -> List[str]:
        """
        Classify multiple nodes efficiently.

        Args:
            gradients: List of gradient vectors
            node_ids: Corresponding node IDs
            round_num: Current round

        Returns:
            List of classifications ('HONEST' or 'BYZANTINE')
        """
        # Extract features for all gradients
        features_list = extract_features_batch(
            gradients, node_ids, round_num, self.feature_extractor
        )

        classifications = []
        for features in features_list:
            # Fast paths
            if features.pogq_score < self.pogq_low:
                self.stats['pogq_fast_reject'] += 1
                classifications.append('BYZANTINE')
            elif features.pogq_score > self.pogq_high:
                self.stats['pogq_fast_accept'] += 1
                classifications.append('HONEST')
            else:
                # ML classifier
                if not self.is_trained:
                    classifications.append(
                        'BYZANTINE' if features.pogq_score < 0.5 else 'HONEST'
                    )
                else:
                    self.stats['ml_classifications'] += 1
                    feature_vector = features.to_array().reshape(1, -1)
                    pred = self.classifier.predict(feature_vector)[0]
                    classifications.append('BYZANTINE' if pred == 1 else 'HONEST')

            self.stats['total_classifications'] += 1

        return classifications

    def get_statistics(self) -> dict:
        """
        Get detector usage statistics.

        Returns:
            Dict with classification counts and percentages
        """
        total = self.stats['total_classifications']
        if total == 0:
            return self.stats.copy()

        return {
            **self.stats,
            'pogq_fast_reject_pct': self.stats['pogq_fast_reject'] / total,
            'pogq_fast_accept_pct': self.stats['pogq_fast_accept'] / total,
            'ml_classifications_pct': self.stats['ml_classifications'] / total,
        }

    def reset_statistics(self):
        """Reset all statistics counters"""
        self.stats = {
            'total_classifications': 0,
            'pogq_fast_reject': 0,
            'pogq_fast_accept': 0,
            'ml_classifications': 0,
        }

    def save(self, path: str):
        """Save trained detector to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save classifier
        self.classifier.save(str(path / 'classifier'))

        # Save detector config
        import pickle
        with open(path / 'config.pkl', 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'pogq_low': self.pogq_low,
                'pogq_high': self.pogq_high,
                'ml_threshold': self.ml_threshold,
                'is_trained': self.is_trained,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'ByzantineDetector':
        """Load trained detector from disk"""
        path = Path(path)

        # Load config
        import pickle
        with open(path / 'config.pkl', 'rb') as f:
            config = pickle.load(f)

        # Create detector
        detector = cls(
            model=config['model_type'],
            pogq_low_threshold=config['pogq_low'],
            pogq_high_threshold=config['pogq_high'],
            ml_threshold=config['ml_threshold'],
        )

        # Load classifier
        if config['model_type'] == 'svm':
            detector.classifier = SVMClassifier.load(str(path / 'classifier'))
        elif config['model_type'] == 'rf':
            detector.classifier = RandomForestClassifier.load(str(path / 'classifier'))
        elif config['model_type'] == 'ensemble':
            from .classifiers import EnsembleClassifier
            detector.classifier = EnsembleClassifier.load(str(path / 'classifier'))

        detector.is_trained = config['is_trained']

        return detector


# Convenience function for quick testing
def quick_detect(gradient: np.ndarray, model_path: Optional[str] = None) -> str:
    """
    Quick one-off detection without managing detector state.

    Args:
        gradient: Gradient vector to classify
        model_path: Path to saved detector (optional)

    Returns:
        'HONEST' or 'BYZANTINE'
    """
    if model_path:
        detector = ByzantineDetector.load(model_path)
    else:
        # Use untrained detector (PoGQ only)
        detector = ByzantineDetector()

    return detector.classify_node(gradient, node_id=0, round_num=0)
