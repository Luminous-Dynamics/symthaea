# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Detection Evaluator for ML Byzantine Detection
==============================================

Comprehensive evaluation metrics, visualizations, and reporting for
Byzantine detection classifiers.

Metrics:
--------
- Detection Rate (Recall): TP / (TP + FN)
- False Positive Rate: FP / (FP + TN)
- Accuracy: (TP + TN) / Total
- Precision: TP / (TP + FP)
- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
- AUC-ROC: Area under ROC curve
- AUC-PR: Area under Precision-Recall curve

Visualizations:
--------------
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance (for RF)
- Classification Reports

All results saved to /results/ml/ directory.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
)
from pathlib import Path
import json

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None


@dataclass
class DetectionMetrics:
    """Container for all detection performance metrics"""

    # Core metrics
    detection_rate: float      # Recall (TP / (TP + FN))
    false_positive_rate: float # FP / (FP + TN)
    accuracy: float            # (TP + TN) / Total
    precision: float           # TP / (TP + FP)
    f1_score: float            # Harmonic mean of precision and recall

    # Advanced metrics
    auc_roc: float             # Area under ROC curve
    auc_pr: float              # Area under PR curve

    # Confusion matrix elements
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    # Metadata
    num_samples: int
    num_byzantine: int
    num_honest: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'detection_rate': float(self.detection_rate),
            'false_positive_rate': float(self.false_positive_rate),
            'accuracy': float(self.accuracy),
            'precision': float(self.precision),
            'f1_score': float(self.f1_score),
            'auc_roc': float(self.auc_roc),
            'auc_pr': float(self.auc_pr),
            'true_positives': int(self.true_positives),
            'false_positives': int(self.false_positives),
            'true_negatives': int(self.true_negatives),
            'false_negatives': int(self.false_negatives),
            'num_samples': int(self.num_samples),
            'num_byzantine': int(self.num_byzantine),
            'num_honest': int(self.num_honest),
        }

    def meets_targets(
        self,
        min_detection_rate: float = 0.95,
        max_false_positive_rate: float = 0.03,
    ) -> bool:
        """
        Check if metrics meet target performance.

        Args:
            min_detection_rate: Minimum acceptable detection rate (default 95%)
            max_false_positive_rate: Maximum acceptable FP rate (default 3%)

        Returns:
            True if both criteria met
        """
        return (
            self.detection_rate >= min_detection_rate and
            self.false_positive_rate <= max_false_positive_rate
        )

    def __str__(self) -> str:
        """Pretty-print metrics"""
        return f"""
Detection Metrics:
==================
Detection Rate:        {self.detection_rate:.1%}
False Positive Rate:   {self.false_positive_rate:.1%}
Accuracy:             {self.accuracy:.1%}
Precision:            {self.precision:.1%}
F1 Score:             {self.f1_score:.3f}
AUC-ROC:              {self.auc_roc:.3f}
AUC-PR:               {self.auc_pr:.3f}

Confusion Matrix:
  TP: {self.true_positives:4d}  FP: {self.false_positives:4d}
  FN: {self.false_negatives:4d}  TN: {self.true_negatives:4d}

Samples: {self.num_samples} ({self.num_byzantine} Byzantine, {self.num_honest} Honest)

Target: {'✅ PASS' if self.meets_targets() else '❌ FAIL'}
"""


class DetectionEvaluator:
    """
    Comprehensive evaluation for Byzantine detection classifiers.

    Computes all metrics, generates visualizations, and saves results.
    """

    def __init__(self, output_dir: str = 'results/ml'):
        """
        Args:
            output_dir: Directory to save results (plots, JSON, reports)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        prefix: str = '',
    ) -> DetectionMetrics:
        """
        Compute all evaluation metrics.

        Args:
            y_true: Ground truth labels (0=honest, 1=Byzantine)
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for ROC/PR curves)
            prefix: Prefix for saved files (e.g., 'svm_', 'ensemble_')

        Returns:
            DetectionMetrics object with all computed metrics
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Core metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0.0)
        rec = recall_score(y_true, y_pred, zero_division=0.0)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)

        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # ROC and PR curves (requires probabilities)
        if y_proba is not None:
            # ROC curve
            fpr_curve, tpr_curve, _ = roc_curve(y_true, y_proba[:, 1])
            auc_roc = auc(fpr_curve, tpr_curve)

            # Precision-Recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_true, y_proba[:, 1]
            )
            auc_pr = auc(recall_curve, precision_curve)
        else:
            auc_roc = 0.0
            auc_pr = 0.0

        metrics = DetectionMetrics(
            detection_rate=rec,
            false_positive_rate=fpr,
            accuracy=acc,
            precision=prec,
            f1_score=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            num_samples=len(y_true),
            num_byzantine=int(np.sum(y_true)),
            num_honest=int(len(y_true) - np.sum(y_true)),
        )

        # Save metrics to JSON
        with open(self.output_dir / f'{prefix}metrics.json', 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        # Generate visualizations (if available)
        if y_proba is not None and VISUALIZATION_AVAILABLE:
            self._plot_confusion_matrix(y_true, y_pred, prefix)
            self._plot_roc_curve(y_true, y_proba[:, 1], auc_roc, prefix)
            self._plot_pr_curve(y_true, y_proba[:, 1], auc_pr, prefix)
        elif y_proba is not None:
            print("⚠️  Skipping visualizations (matplotlib/seaborn not available)")

        # Save classification report
        report = classification_report(y_true, y_pred, target_names=['Honest', 'Byzantine'])
        with open(self.output_dir / f'{prefix}classification_report.txt', 'w') as f:
            f.write(report)

        return metrics

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str,
    ):
        """Generate and save confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Honest', 'Byzantine'],
            yticklabels=['Honest', 'Byzantine'],
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{prefix}confusion_matrix.png', dpi=150)
        plt.close()

    def _plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        auc_score: float,
        prefix: str,
    ):
        """Generate and save ROC curve plot"""
        fpr, tpr, _ = roc_curve(y_true, y_score)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Detection Rate)')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{prefix}roc_curve.png', dpi=150)
        plt.close()

    def _plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        auc_score: float,
        prefix: str,
    ):
        """Generate and save Precision-Recall curve plot"""
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'AUC = {auc_score:.3f}')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Detection Rate)')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{prefix}pr_curve.png', dpi=150)
        plt.close()

    def compare_models(
        self,
        results: Dict[str, DetectionMetrics],
        save_path: Optional[str] = None,
    ):
        """
        Compare multiple models side-by-side.

        Args:
            results: Dict mapping model names to their DetectionMetrics
            save_path: Optional path to save comparison plot

        Example:
            >>> results = {
            ...     'SVM': svm_metrics,
            ...     'RF': rf_metrics,
            ...     'Ensemble': ensemble_metrics,
            ... }
            >>> evaluator.compare_models(results)
        """
        # Extract metrics for comparison
        models = list(results.keys())
        detection_rates = [results[m].detection_rate for m in models]
        fp_rates = [results[m].false_positive_rate for m in models]
        f1_scores = [results[m].f1_score for m in models]

        # Create comparison plot (if visualization available)
        if VISUALIZATION_AVAILABLE:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Detection Rate
            axes[0].bar(models, detection_rates, color='green', alpha=0.7)
            axes[0].axhline(y=0.95, color='r', linestyle='--', label='Target (95%)')
            axes[0].set_ylabel('Detection Rate')
            axes[0].set_ylim([0, 1.0])
            axes[0].set_title('Detection Rate Comparison')
            axes[0].legend()

            # False Positive Rate
            axes[1].bar(models, fp_rates, color='red', alpha=0.7)
            axes[1].axhline(y=0.03, color='g', linestyle='--', label='Target (3%)')
            axes[1].set_ylabel('False Positive Rate')
            axes[1].set_ylim([0, 0.10])
            axes[1].set_title('False Positive Rate Comparison')
            axes[1].legend()

            # F1 Score
            axes[2].bar(models, f1_scores, color='blue', alpha=0.7)
            axes[2].set_ylabel('F1 Score')
            axes[2].set_ylim([0, 1.0])
            axes[2].set_title('F1 Score Comparison')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150)
            else:
                plt.savefig(self.output_dir / 'model_comparison.png', dpi=150)

            plt.close()
        else:
            print("⚠️  Skipping model comparison plot (matplotlib/seaborn not available)")

        # Print text comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Detection Rate: {metrics.detection_rate:.1%}")
            print(f"  FP Rate:        {metrics.false_positive_rate:.1%}")
            print(f"  F1 Score:       {metrics.f1_score:.3f}")
            print(f"  Target Met:     {'✅ YES' if metrics.meets_targets() else '❌ NO'}")
        print("="*60 + "\n")


def quick_evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> DetectionMetrics:
    """
    Quick evaluation without saving results (useful for debugging).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        DetectionMetrics object
    """
    evaluator = DetectionEvaluator(output_dir='/tmp/ml_eval')
    return evaluator.evaluate(y_true, y_pred, y_proba, prefix='quick_')
