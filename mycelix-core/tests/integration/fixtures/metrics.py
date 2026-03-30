# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Metrics collection and test reporting for FL integration tests.

Provides comprehensive metrics tracking including:
- Aggregation quality metrics
- Byzantine detection accuracy
- Latency measurements
- Test report generation
"""

import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Metric Data Classes
# ============================================================================

@dataclass
class AggregationMetrics:
    """Metrics for aggregation quality."""
    round_id: int
    n_participants: int
    n_byzantine: int
    n_detected: int

    # Quality metrics
    cosine_similarity: float = 0.0  # To ground truth
    l2_distance: float = 0.0
    relative_error: float = 0.0

    # Aggregation info
    strategy: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "n_participants": self.n_participants,
            "n_byzantine": self.n_byzantine,
            "n_detected": self.n_detected,
            "cosine_similarity": float(self.cosine_similarity),
            "l2_distance": float(self.l2_distance),
            "relative_error": float(self.relative_error),
            "strategy": self.strategy,
            "duration_ms": float(self.duration_ms),
        }


@dataclass
class DetectionMetrics:
    """Metrics for Byzantine detection accuracy."""
    round_id: int
    true_byzantines: Set[str]
    detected_byzantines: Set[str]

    # Derived metrics
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0

    def compute(self, all_node_ids: Set[str]):
        """Compute derived metrics."""
        self.true_positives = len(self.detected_byzantines & self.true_byzantines)
        self.false_positives = len(self.detected_byzantines - self.true_byzantines)
        self.false_negatives = len(self.true_byzantines - self.detected_byzantines)

        honest_nodes = all_node_ids - self.true_byzantines
        correctly_not_detected = honest_nodes - self.detected_byzantines
        self.true_negatives = len(correctly_not_detected)

        # Precision: TP / (TP + FP)
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)

        # Recall: TP / (TP + FN)
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)

        # F1 Score
        if self.precision + self.recall > 0:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)

        # Accuracy: (TP + TN) / Total
        total = len(all_node_ids)
        if total > 0:
            self.accuracy = (self.true_positives + self.true_negatives) / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "true_byzantines": list(self.true_byzantines),
            "detected_byzantines": list(self.detected_byzantines),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1_score": float(self.f1_score),
            "accuracy": float(self.accuracy),
        }


@dataclass
class LatencyMetrics:
    """Latency measurements for various operations."""
    operation: str
    duration_ms: float
    round_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "duration_ms": float(self.duration_ms),
            "round_id": self.round_id,
            "metadata": self.metadata,
        }


@dataclass
class RoundMetrics:
    """Complete metrics for a single FL round."""
    round_id: int
    start_time: float
    end_time: float
    success: bool
    error: Optional[str] = None

    aggregation: Optional[AggregationMetrics] = None
    detection: Optional[DetectionMetrics] = None
    latencies: List[LatencyMetrics] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "aggregation": self.aggregation.to_dict() if self.aggregation else None,
            "detection": self.detection.to_dict() if self.detection else None,
            "latencies": [l.to_dict() for l in self.latencies],
        }


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """
    Collects and aggregates metrics during FL tests.

    Tracks:
    - Per-round metrics
    - Aggregation quality over time
    - Detection accuracy
    - Latency distributions
    """

    def __init__(self):
        self._round_metrics: List[RoundMetrics] = []
        self._aggregation_metrics: List[AggregationMetrics] = []
        self._detection_metrics: List[DetectionMetrics] = []
        self._latency_metrics: List[LatencyMetrics] = []
        self._custom_metrics: Dict[str, List[Any]] = {}
        self._start_time: float = time.time()
        self._finalized: bool = False

    def start_round(self, round_id: int) -> RoundMetrics:
        """Start tracking a new round."""
        metrics = RoundMetrics(
            round_id=round_id,
            start_time=time.time(),
            end_time=0,
            success=False,
        )
        return metrics

    def end_round(self, metrics: RoundMetrics, success: bool, error: Optional[str] = None):
        """End tracking for a round."""
        metrics.end_time = time.time()
        metrics.success = success
        metrics.error = error
        self._round_metrics.append(metrics)

    def record_aggregation(
        self,
        round_id: int,
        aggregated: np.ndarray,
        ground_truth: np.ndarray,
        n_participants: int,
        n_byzantine: int,
        n_detected: int,
        strategy: str,
        duration_ms: float,
    ) -> AggregationMetrics:
        """Record aggregation quality metrics."""
        # Compute quality metrics
        agg_norm = np.linalg.norm(aggregated)
        gt_norm = np.linalg.norm(ground_truth)

        if agg_norm > 0 and gt_norm > 0:
            cosine_similarity = np.dot(aggregated, ground_truth) / (agg_norm * gt_norm)
        else:
            cosine_similarity = 0.0

        l2_distance = np.linalg.norm(aggregated - ground_truth)
        relative_error = l2_distance / gt_norm if gt_norm > 0 else l2_distance

        metrics = AggregationMetrics(
            round_id=round_id,
            n_participants=n_participants,
            n_byzantine=n_byzantine,
            n_detected=n_detected,
            cosine_similarity=float(cosine_similarity),
            l2_distance=float(l2_distance),
            relative_error=float(relative_error),
            strategy=strategy,
            duration_ms=duration_ms,
        )

        self._aggregation_metrics.append(metrics)
        return metrics

    def record_detection(
        self,
        round_id: int,
        true_byzantines: Set[str],
        detected_byzantines: Set[str],
        all_node_ids: Set[str],
    ) -> DetectionMetrics:
        """Record Byzantine detection metrics."""
        metrics = DetectionMetrics(
            round_id=round_id,
            true_byzantines=true_byzantines.copy(),
            detected_byzantines=detected_byzantines.copy(),
        )
        metrics.compute(all_node_ids)
        self._detection_metrics.append(metrics)
        return metrics

    def record_latency(
        self,
        operation: str,
        duration_ms: float,
        round_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LatencyMetrics:
        """Record a latency measurement."""
        metrics = LatencyMetrics(
            operation=operation,
            duration_ms=duration_ms,
            round_id=round_id,
            metadata=metadata or {},
        )
        self._latency_metrics.append(metrics)
        return metrics

    def record_custom(self, name: str, value: Any):
        """Record a custom metric."""
        if name not in self._custom_metrics:
            self._custom_metrics[name] = []
        self._custom_metrics[name].append(value)

    def get_round_metrics(self) -> List[RoundMetrics]:
        """Get all round metrics."""
        return self._round_metrics.copy()

    def get_aggregation_summary(self) -> Dict[str, Any]:
        """Get summary of aggregation metrics."""
        if not self._aggregation_metrics:
            return {}

        cosine_sims = [m.cosine_similarity for m in self._aggregation_metrics]
        relative_errors = [m.relative_error for m in self._aggregation_metrics]
        durations = [m.duration_ms for m in self._aggregation_metrics]

        return {
            "n_rounds": len(self._aggregation_metrics),
            "cosine_similarity": {
                "mean": float(np.mean(cosine_sims)),
                "std": float(np.std(cosine_sims)),
                "min": float(np.min(cosine_sims)),
                "max": float(np.max(cosine_sims)),
            },
            "relative_error": {
                "mean": float(np.mean(relative_errors)),
                "std": float(np.std(relative_errors)),
                "min": float(np.min(relative_errors)),
                "max": float(np.max(relative_errors)),
            },
            "duration_ms": {
                "mean": float(np.mean(durations)),
                "std": float(np.std(durations)),
                "p50": float(np.percentile(durations, 50)),
                "p95": float(np.percentile(durations, 95)),
                "p99": float(np.percentile(durations, 99)),
            },
        }

    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of detection metrics."""
        if not self._detection_metrics:
            return {}

        precisions = [m.precision for m in self._detection_metrics]
        recalls = [m.recall for m in self._detection_metrics]
        f1_scores = [m.f1_score for m in self._detection_metrics]
        accuracies = [m.accuracy for m in self._detection_metrics]

        return {
            "n_rounds": len(self._detection_metrics),
            "precision": {
                "mean": float(np.mean(precisions)),
                "std": float(np.std(precisions)),
            },
            "recall": {
                "mean": float(np.mean(recalls)),
                "std": float(np.std(recalls)),
            },
            "f1_score": {
                "mean": float(np.mean(f1_scores)),
                "std": float(np.std(f1_scores)),
            },
            "accuracy": {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies)),
            },
            "total_true_positives": sum(m.true_positives for m in self._detection_metrics),
            "total_false_positives": sum(m.false_positives for m in self._detection_metrics),
            "total_false_negatives": sum(m.false_negatives for m in self._detection_metrics),
        }

    def get_latency_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of latency metrics by operation."""
        by_operation: Dict[str, List[float]] = {}

        for m in self._latency_metrics:
            if m.operation not in by_operation:
                by_operation[m.operation] = []
            by_operation[m.operation].append(m.duration_ms)

        summary = {}
        for op, durations in by_operation.items():
            summary[op] = {
                "count": len(durations),
                "mean": float(np.mean(durations)),
                "std": float(np.std(durations)),
                "min": float(np.min(durations)),
                "max": float(np.max(durations)),
                "p50": float(np.percentile(durations, 50)),
                "p95": float(np.percentile(durations, 95)),
                "p99": float(np.percentile(durations, 99)),
            }

        return summary

    def finalize(self):
        """Finalize metrics collection."""
        self._finalized = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary."""
        return {
            "collection_duration_s": time.time() - self._start_time,
            "rounds": [m.to_dict() for m in self._round_metrics],
            "aggregation_metrics": [m.to_dict() for m in self._aggregation_metrics],
            "detection_metrics": [m.to_dict() for m in self._detection_metrics],
            "latency_metrics": [m.to_dict() for m in self._latency_metrics],
            "custom_metrics": self._custom_metrics,
            "aggregation_summary": self.get_aggregation_summary(),
            "detection_summary": self.get_detection_summary(),
            "latency_summary": self.get_latency_summary(),
        }


# ============================================================================
# Test Report
# ============================================================================

@dataclass
class TestReport:
    """
    Comprehensive test report.

    Aggregates all metrics and provides analysis.
    """
    test_name: str
    start_time: datetime
    metrics_collector: MetricsCollector
    end_time: Optional[datetime] = None
    passed: bool = True
    failure_reason: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    def add_tag(self, tag: str):
        """Add a tag to the report."""
        if tag not in self.tags:
            self.tags.append(tag)

    def set_parameter(self, key: str, value: Any):
        """Set a test parameter."""
        self.parameters[key] = value

    def fail(self, reason: str):
        """Mark test as failed."""
        self.passed = False
        self.failure_reason = reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "test_name": self.test_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_s": self.duration_s,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "tags": self.tags,
            "parameters": self.parameters,
            "metrics": self.metrics_collector.to_dict(),
        }

    def save(self, path: Path):
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Test report saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "TestReport":
        """Load report from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Create a minimal report from saved data
        report = cls(
            test_name=data["test_name"],
            start_time=datetime.fromisoformat(data["start_time"]),
            metrics_collector=MetricsCollector(),  # Empty collector
            end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
            passed=data["passed"],
            failure_reason=data.get("failure_reason"),
            tags=data.get("tags", []),
            parameters=data.get("parameters", {}),
        )
        return report

    def generate_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Test Report: {self.test_name}",
            "=" * 60,
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Duration: {self.duration_s:.2f}s",
            "",
        ]

        if self.failure_reason:
            lines.append(f"Failure Reason: {self.failure_reason}")
            lines.append("")

        if self.parameters:
            lines.append("Parameters:")
            for k, v in self.parameters.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        agg_summary = self.metrics_collector.get_aggregation_summary()
        if agg_summary:
            lines.append("Aggregation Quality:")
            lines.append(f"  Rounds: {agg_summary['n_rounds']}")
            lines.append(f"  Cosine Similarity: {agg_summary['cosine_similarity']['mean']:.4f} +/- {agg_summary['cosine_similarity']['std']:.4f}")
            lines.append(f"  Relative Error: {agg_summary['relative_error']['mean']:.4f} +/- {agg_summary['relative_error']['std']:.4f}")
            lines.append("")

        det_summary = self.metrics_collector.get_detection_summary()
        if det_summary:
            lines.append("Byzantine Detection:")
            lines.append(f"  Precision: {det_summary['precision']['mean']:.4f}")
            lines.append(f"  Recall: {det_summary['recall']['mean']:.4f}")
            lines.append(f"  F1 Score: {det_summary['f1_score']['mean']:.4f}")
            lines.append("")

        lat_summary = self.metrics_collector.get_latency_summary()
        if lat_summary:
            lines.append("Latency Summary:")
            for op, stats in lat_summary.items():
                lines.append(f"  {op}: {stats['mean']:.2f}ms (p95: {stats['p95']:.2f}ms)")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# Quality Assertions
# ============================================================================

class QualityAssertions:
    """Helper class for quality-based assertions."""

    @staticmethod
    def assert_aggregation_quality(
        aggregated: np.ndarray,
        expected: np.ndarray,
        min_cosine_similarity: float = 0.9,
        max_relative_error: float = 0.1,
    ):
        """Assert aggregation quality meets thresholds."""
        agg_norm = np.linalg.norm(aggregated)
        exp_norm = np.linalg.norm(expected)

        if agg_norm > 0 and exp_norm > 0:
            cosine_sim = np.dot(aggregated, expected) / (agg_norm * exp_norm)
        else:
            cosine_sim = 0.0

        relative_error = np.linalg.norm(aggregated - expected) / exp_norm if exp_norm > 0 else np.inf

        assert cosine_sim >= min_cosine_similarity, (
            f"Cosine similarity {cosine_sim:.4f} below threshold {min_cosine_similarity}"
        )
        assert relative_error <= max_relative_error, (
            f"Relative error {relative_error:.4f} exceeds threshold {max_relative_error}"
        )

    @staticmethod
    def assert_detection_accuracy(
        detected: Set[str],
        true_byzantines: Set[str],
        all_nodes: Set[str],
        min_precision: float = 0.8,
        min_recall: float = 0.8,
    ):
        """Assert detection accuracy meets thresholds."""
        tp = len(detected & true_byzantines)
        fp = len(detected - true_byzantines)
        fn = len(true_byzantines - detected)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        assert precision >= min_precision, (
            f"Detection precision {precision:.4f} below threshold {min_precision}"
        )
        assert recall >= min_recall, (
            f"Detection recall {recall:.4f} below threshold {min_recall}"
        )

    @staticmethod
    def assert_latency(
        latencies_ms: List[float],
        max_mean_ms: float = 1000.0,
        max_p95_ms: float = 2000.0,
    ):
        """Assert latencies meet thresholds."""
        mean = np.mean(latencies_ms)
        p95 = np.percentile(latencies_ms, 95)

        assert mean <= max_mean_ms, (
            f"Mean latency {mean:.2f}ms exceeds threshold {max_mean_ms}ms"
        )
        assert p95 <= max_p95_ms, (
            f"P95 latency {p95:.2f}ms exceeds threshold {max_p95_ms}ms"
        )
