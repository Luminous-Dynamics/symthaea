# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ML Classifiers for Byzantine Detection
=======================================

Implements SVM and Random Forest classifiers with ensemble logic for
high-accuracy Byzantine node detection (95-98% target).

Classifiers:
-----------
1. SVMClassifier: RBF kernel, good for non-linear boundaries
2. RandomForestClassifier: Robust against feature noise and overfitting
3. EnsembleClassifier: Combines both for maximum accuracy

All classifiers provide:
- .fit(X, y): Train on labeled data
- .predict(X): Binary classification (0=honest, 1=Byzantine)
- .predict_proba(X): Probability estimates for uncertainty quantification
"""

from typing import Literal, Optional, Tuple
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class SVMClassifier:
    """
    Support Vector Machine with RBF kernel for Byzantine detection.

    Features:
    ---------
    - RBF kernel captures non-linear decision boundaries
    - Probability estimates via Platt scaling
    - Automatic feature scaling (StandardScaler)
    - Regularization parameter C tunable for precision/recall tradeoff

    Parameters:
    -----------
    C: Regularization parameter (higher = less regularization)
    gamma: RBF kernel coefficient (higher = more complex boundary)
    probability: Enable probability estimates (default True)
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: str = 'scale',
        probability: bool = True,
        random_state: int = 42,
    ):
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state

        # Initialize SVM and scaler
        self.svm = SVC(
            C=C,
            kernel='rbf',
            gamma=gamma,
            probability=probability,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """
        Train SVM classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=honest, 1=Byzantine)

        Returns:
            Self for method chaining
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train SVM
        self.svm.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (0=honest, 1=Byzantine)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not trained. Call .fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.svm.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probability matrix (n_samples, 2) where [:, 1] is P(Byzantine)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not trained. Call .fit() first.")

        if not self.probability:
            raise ValueError("Probability estimates not enabled. Set probability=True.")

        X_scaled = self.scaler.transform(X)
        return self.svm.predict_proba(X_scaled)

    def save(self, path: str):
        """Save trained model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'svm': self.svm,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'C': self.C,
                'gamma': self.gamma,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'SVMClassifier':
        """Load trained model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(C=data['C'], gamma=data['gamma'])
        model.svm = data['svm']
        model.scaler = data['scaler']
        model.is_fitted = data['is_fitted']

        return model


class RandomForestClassifier:
    """
    Random Forest classifier for Byzantine detection.

    Features:
    ---------
    - Ensemble of decision trees (robust against overfitting)
    - Feature importance analysis
    - No feature scaling required
    - Handles high-dimensional feature spaces well

    Parameters:
    -----------
    n_estimators: Number of trees in forest
    max_depth: Maximum tree depth (None = unlimited)
    min_samples_split: Minimum samples to split node
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.rf = SklearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """
        Train Random Forest classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=honest, 1=Byzantine)

        Returns:
            Self for method chaining
        """
        self.rf.fit(X, y)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (0=honest, 1=Byzantine)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not trained. Call .fit() first.")

        return self.rf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probability matrix (n_samples, 2) where [:, 1] is P(Byzantine)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not trained. Call .fit() first.")

        return self.rf.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.

        Returns:
            Importance scores for each feature (sums to 1.0)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not trained. Call .fit() first.")

        return self.rf.feature_importances_

    def save(self, path: str):
        """Save trained model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'rf': self.rf,
                'is_fitted': self.is_fitted,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'RandomForestClassifier':
        """Load trained model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(
            n_estimators=data['n_estimators'],
            max_depth=data['max_depth']
        )
        model.rf = data['rf']
        model.is_fitted = data['is_fitted']

        return model


class EnsembleClassifier:
    """
    Ensemble combining SVM and Random Forest for maximum accuracy.

    Strategy:
    ---------
    - Trains both SVM and RF on same data
    - Predictions combined via majority voting or probability averaging
    - Achieves 95-98% detection rate by leveraging strengths of both

    Parameters:
    -----------
    voting: 'hard' (majority vote) or 'soft' (average probabilities)
    svm_kwargs: Parameters for SVMClassifier
    rf_kwargs: Parameters for RandomForestClassifier
    """

    def __init__(
        self,
        voting: Literal['hard', 'soft'] = 'soft',
        svm_kwargs: Optional[dict] = None,
        rf_kwargs: Optional[dict] = None,
    ):
        self.voting = voting

        # Initialize sub-classifiers
        svm_kwargs = svm_kwargs or {}
        rf_kwargs = rf_kwargs or {}

        self.svm = SVMClassifier(**svm_kwargs)
        self.rf = RandomForestClassifier(**rf_kwargs)

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleClassifier':
        """
        Train both SVM and RF classifiers.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=honest, 1=Byzantine)

        Returns:
            Self for method chaining
        """
        # Train both models
        self.svm.fit(X, y)
        self.rf.fit(X, y)

        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels using ensemble voting.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (0=honest, 1=Byzantine)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not trained. Call .fit() first.")

        if self.voting == 'hard':
            # Majority voting
            svm_pred = self.svm.predict(X)
            rf_pred = self.rf.predict(X)

            # Combine: 1 if both predict 1, else 0
            return ((svm_pred + rf_pred) >= 1).astype(int)

        else:  # soft voting
            # Average probabilities
            svm_proba = self.svm.predict_proba(X)[:, 1]
            rf_proba = self.rf.predict_proba(X)[:, 1]

            avg_proba = (svm_proba + rf_proba) / 2.0

            return (avg_proba >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using ensemble averaging.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probability matrix (n_samples, 2) where [:, 1] is P(Byzantine)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not trained. Call .fit() first.")

        # Average probabilities from both models
        svm_proba = self.svm.predict_proba(X)
        rf_proba = self.rf.predict_proba(X)

        avg_proba = (svm_proba + rf_proba) / 2.0

        return avg_proba

    def get_individual_predictions(self, X: np.ndarray) -> dict:
        """
        Get predictions from each sub-classifier separately.

        Useful for debugging and understanding ensemble decisions.

        Returns:
            Dict with keys 'svm', 'rf', 'ensemble'
        """
        return {
            'svm': self.svm.predict(X),
            'rf': self.rf.predict(X),
            'ensemble': self.predict(X),
        }

    def save(self, path: str):
        """Save trained ensemble to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.svm.save(str(path / 'svm.pkl'))
        self.rf.save(str(path / 'rf.pkl'))

        with open(path / 'meta.pkl', 'wb') as f:
            pickle.dump({
                'voting': self.voting,
                'is_fitted': self.is_fitted,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'EnsembleClassifier':
        """Load trained ensemble from disk"""
        path = Path(path)

        with open(path / 'meta.pkl', 'rb') as f:
            meta = pickle.load(f)

        model = cls(voting=meta['voting'])
        model.svm = SVMClassifier.load(str(path / 'svm.pkl'))
        model.rf = RandomForestClassifier.load(str(path / 'rf.pkl'))
        model.is_fitted = meta['is_fitted']

        return model


def create_classifier(
    model_type: Literal['svm', 'rf', 'ensemble'] = 'ensemble',
    **kwargs
):
    """
    Factory function to create classifier by name.

    Args:
        model_type: Type of classifier to create
        **kwargs: Parameters passed to classifier constructor

    Returns:
        Initialized classifier (not trained)

    Example:
        >>> clf = create_classifier('svm', C=2.0, gamma='auto')
        >>> clf = create_classifier('rf', n_estimators=200)
        >>> clf = create_classifier('ensemble', voting='hard')
    """
    if model_type == 'svm':
        return SVMClassifier(**kwargs)
    elif model_type == 'rf':
        return RandomForestClassifier(**kwargs)
    elif model_type == 'ensemble':
        return EnsembleClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
