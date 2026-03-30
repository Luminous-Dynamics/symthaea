# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
MycelixFL Observability Module

Provides metrics, structured logging, and performance tracking for
production deployments. Supports Prometheus-compatible metrics export.

Author: Luminous Dynamics
Date: December 31, 2025
"""

import time
import logging
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from collections import defaultdict
import threading
import json
from contextlib import contextmanager

from .config import (
    DETECTION_LATENCY_TARGET_MS,
    DETECTION_LATENCY_WARN_MS,
    runtime_config,
)

logger = logging.getLogger(__name__)

# Type variable for generic decorators
F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# METRICS COLLECTION
# =============================================================================

@dataclass
class MetricValue:
    """Single metric value with timestamp."""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """
    Prometheus-style counter metric.

    Counters only go up (or reset to zero).

    Example:
        >>> counter = Counter("requests_total", "Total requests processed")
        >>> counter.inc()
        >>> counter.inc(5, labels={"status": "success"})
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._values[label_key] += value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current value."""
        label_key = self._labels_to_key(labels)
        return self._values.get(label_key, 0.0)

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> str:
        """Convert labels dict to hashable key."""
        if not labels:
            return ""
        return json.dumps(labels, sort_keys=True)

    def collect(self) -> List[MetricValue]:
        """Collect all values for export."""
        result = []
        for label_key, value in self._values.items():
            labels = json.loads(label_key) if label_key else {}
            result.append(MetricValue(value=value, labels=labels))
        return result


class Gauge:
    """
    Prometheus-style gauge metric.

    Gauges can go up and down.

    Example:
        >>> gauge = Gauge("active_nodes", "Currently active nodes")
        >>> gauge.set(10)
        >>> gauge.inc()
        >>> gauge.dec()
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._values[label_key] = value

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment gauge."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._values[label_key] += value

    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement gauge."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._values[label_key] -= value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current value."""
        label_key = self._labels_to_key(labels)
        return self._values.get(label_key, 0.0)

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return ""
        return json.dumps(labels, sort_keys=True)

    def collect(self) -> List[MetricValue]:
        """Collect all values for export."""
        result = []
        for label_key, value in self._values.items():
            labels = json.loads(label_key) if label_key else {}
            result.append(MetricValue(value=value, labels=labels))
        return result


class Histogram:
    """
    Prometheus-style histogram metric.

    Tracks distribution of values.

    Example:
        >>> hist = Histogram("latency_ms", "Operation latency")
        >>> hist.observe(15.5)
        >>> print(hist.percentile(0.95))
    """

    DEFAULT_BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: tuple = DEFAULT_BUCKETS,
    ):
        self.name = name
        self.description = description
        self.buckets = sorted(buckets)
        self._values: List[float] = []
        self._bucket_counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self._bucket_counts[float('inf')] = 0
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """Record an observation."""
        with self._lock:
            self._values.append(value)
            self._sum += value
            self._count += 1

            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
            self._bucket_counts[float('inf')] += 1

    def percentile(self, p: float) -> float:
        """Calculate percentile (0-1)."""
        if not self._values:
            return 0.0
        sorted_vals = sorted(self._values)
        idx = int(len(sorted_vals) * p)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    @property
    def mean(self) -> float:
        """Calculate mean."""
        return self._sum / self._count if self._count > 0 else 0.0

    @property
    def count(self) -> int:
        """Get observation count."""
        return self._count

    def collect(self) -> Dict[str, Any]:
        """Collect histogram data for export."""
        return {
            "count": self._count,
            "sum": self._sum,
            "mean": self.mean,
            "p50": self.percentile(0.50),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
            "buckets": dict(self._bucket_counts),
        }


# =============================================================================
# METRICS REGISTRY
# =============================================================================

class MetricsRegistry:
    """
    Central registry for all MycelixFL metrics.

    Example:
        >>> registry = MetricsRegistry()
        >>> registry.rounds_total.inc()
        >>> print(registry.export_prometheus())
    """

    def __init__(self, prefix: str = "mycelix_fl"):
        self.prefix = prefix

        # Counters
        self.rounds_total = Counter(
            f"{prefix}_rounds_total",
            "Total FL rounds executed",
        )
        self.gradients_received = Counter(
            f"{prefix}_gradients_received_total",
            "Total gradients received",
        )
        self.byzantine_detected = Counter(
            f"{prefix}_byzantine_detected_total",
            "Total Byzantine nodes detected",
        )
        self.gradients_healed = Counter(
            f"{prefix}_gradients_healed_total",
            "Total gradients healed",
        )
        self.errors_total = Counter(
            f"{prefix}_errors_total",
            "Total errors encountered",
        )

        # Gauges
        self.active_nodes = Gauge(
            f"{prefix}_active_nodes",
            "Currently active nodes",
        )
        self.byzantine_ratio = Gauge(
            f"{prefix}_byzantine_ratio",
            "Current Byzantine ratio",
        )
        self.system_phi = Gauge(
            f"{prefix}_system_phi",
            "System integrated information (Φ)",
        )
        self.compression_ratio = Gauge(
            f"{prefix}_compression_ratio",
            "Current compression ratio",
        )

        # Histograms
        self.round_latency_ms = Histogram(
            f"{prefix}_round_latency_ms",
            "FL round latency in milliseconds",
        )
        self.detection_latency_ms = Histogram(
            f"{prefix}_detection_latency_ms",
            "Byzantine detection latency in milliseconds",
        )
        self.encoding_latency_ms = Histogram(
            f"{prefix}_encoding_latency_ms",
            "Gradient encoding latency in milliseconds",
        )
        self.shapley_computation_ms = Histogram(
            f"{prefix}_shapley_computation_ms",
            "Shapley value computation time in milliseconds",
        )

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = []

        # Export counters
        for counter in [
            self.rounds_total,
            self.gradients_received,
            self.byzantine_detected,
            self.gradients_healed,
            self.errors_total,
        ]:
            lines.append(f"# HELP {counter.name} {counter.description}")
            lines.append(f"# TYPE {counter.name} counter")
            for mv in counter.collect():
                labels_str = self._format_labels(mv.labels)
                lines.append(f"{counter.name}{labels_str} {mv.value}")

        # Export gauges
        for gauge in [
            self.active_nodes,
            self.byzantine_ratio,
            self.system_phi,
            self.compression_ratio,
        ]:
            lines.append(f"# HELP {gauge.name} {gauge.description}")
            lines.append(f"# TYPE {gauge.name} gauge")
            for mv in gauge.collect():
                labels_str = self._format_labels(mv.labels)
                lines.append(f"{gauge.name}{labels_str} {mv.value}")

        # Export histograms
        for hist in [
            self.round_latency_ms,
            self.detection_latency_ms,
            self.encoding_latency_ms,
            self.shapley_computation_ms,
        ]:
            data = hist.collect()
            lines.append(f"# HELP {hist.name} {hist.description}")
            lines.append(f"# TYPE {hist.name} histogram")
            lines.append(f"{hist.name}_count {data['count']}")
            lines.append(f"{hist.name}_sum {data['sum']}")
            for bucket, count in data['buckets'].items():
                if bucket == float('inf'):
                    lines.append(f'{hist.name}_bucket{{le="+Inf"}} {count}')
                else:
                    lines.append(f'{hist.name}_bucket{{le="{bucket}"}} {count}')

        return "\n".join(lines)

    def export_json(self) -> Dict[str, Any]:
        """Export all metrics as JSON."""
        return {
            "counters": {
                "rounds_total": self.rounds_total.get(),
                "gradients_received": self.gradients_received.get(),
                "byzantine_detected": self.byzantine_detected.get(),
                "gradients_healed": self.gradients_healed.get(),
                "errors_total": self.errors_total.get(),
            },
            "gauges": {
                "active_nodes": self.active_nodes.get(),
                "byzantine_ratio": self.byzantine_ratio.get(),
                "system_phi": self.system_phi.get(),
                "compression_ratio": self.compression_ratio.get(),
            },
            "histograms": {
                "round_latency_ms": self.round_latency_ms.collect(),
                "detection_latency_ms": self.detection_latency_ms.collect(),
                "encoding_latency_ms": self.encoding_latency_ms.collect(),
                "shapley_computation_ms": self.shapley_computation_ms.collect(),
            },
        }

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(parts) + "}"


# Global metrics registry
metrics = MetricsRegistry(prefix=runtime_config.metrics_prefix)


# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

class StructuredLogger:
    """
    Structured logger with context propagation.

    Outputs JSON-formatted logs for easy parsing.

    Example:
        >>> log = StructuredLogger("mycelix_fl.detection")
        >>> with log.context(round_num=5, node_count=10):
        ...     log.info("Detection complete", byzantine_count=2)
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}
        self._context_stack: List[Dict[str, Any]] = []

    @contextmanager
    def context(self, **kwargs):
        """Add context for all logs within scope."""
        self._context_stack.append(self._context.copy())
        self._context.update(kwargs)
        try:
            yield
        finally:
            self._context = self._context_stack.pop()

    def _format(self, message: str, **kwargs) -> str:
        """Format as structured JSON."""
        data = {
            "message": message,
            "timestamp": time.time(),
            **self._context,
            **kwargs,
        }
        return json.dumps(data)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(self._format(message, level="DEBUG", **kwargs))

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._logger.info(self._format(message, level="INFO", **kwargs))

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(self._format(message, level="WARNING", **kwargs))

    def error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log error message."""
        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)
        self._logger.error(self._format(message, level="ERROR", **kwargs))

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._logger.critical(self._format(message, level="CRITICAL", **kwargs))


# =============================================================================
# TIMING DECORATORS
# =============================================================================

def timed(
    metric: Optional[Histogram] = None,
    warn_threshold_ms: float = DETECTION_LATENCY_WARN_MS,
) -> Callable[[F], F]:
    """
    Decorator to time function execution.

    Args:
        metric: Optional histogram to record timing
        warn_threshold_ms: Threshold for warning log

    Example:
        >>> @timed(metrics.detection_latency_ms)
        ... def detect_byzantine(gradients):
        ...     # detection logic
        ...     pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000

                if metric is not None:
                    metric.observe(elapsed_ms)

                if elapsed_ms > warn_threshold_ms:
                    logger.warning(
                        f"{func.__name__} took {elapsed_ms:.1f}ms "
                        f"(threshold: {warn_threshold_ms:.1f}ms)"
                    )
                elif runtime_config.debug_mode:
                    logger.debug(f"{func.__name__} completed in {elapsed_ms:.1f}ms")

        return wrapper  # type: ignore
    return decorator


@contextmanager
def timed_block(name: str, metric: Optional[Histogram] = None):
    """
    Context manager for timing code blocks.

    Example:
        >>> with timed_block("shapley_computation", metrics.shapley_computation_ms):
        ...     compute_shapley_values(gradients)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        if metric is not None:
            metric.observe(elapsed_ms)
        if runtime_config.debug_mode:
            logger.debug(f"{name} completed in {elapsed_ms:.1f}ms")


# =============================================================================
# PERFORMANCE TRACKER
# =============================================================================

@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics."""
    timestamp: float
    round_num: int
    total_nodes: int
    byzantine_nodes: int
    healed_nodes: int
    detection_latency_ms: float
    round_latency_ms: float
    system_phi: float
    compression_ratio: float


class PerformanceTracker:
    """
    Track performance over multiple FL rounds.

    Example:
        >>> tracker = PerformanceTracker()
        >>> for round_num in range(100):
        ...     result = fl.execute_round(gradients)
        ...     tracker.record(round_num, result)
        >>> print(tracker.summary())
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.history: List[PerformanceSnapshot] = []
        self._lock = threading.Lock()

    def record(
        self,
        round_num: int,
        total_nodes: int,
        byzantine_nodes: int,
        healed_nodes: int,
        detection_latency_ms: float,
        round_latency_ms: float,
        system_phi: float = 0.0,
        compression_ratio: float = 1.0,
    ) -> None:
        """Record performance for a round."""
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            round_num=round_num,
            total_nodes=total_nodes,
            byzantine_nodes=byzantine_nodes,
            healed_nodes=healed_nodes,
            detection_latency_ms=detection_latency_ms,
            round_latency_ms=round_latency_ms,
            system_phi=system_phi,
            compression_ratio=compression_ratio,
        )

        with self._lock:
            self.history.append(snapshot)
            # Trim to max history
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

    def summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.history:
            return {"rounds": 0}

        detection_latencies = [s.detection_latency_ms for s in self.history]
        round_latencies = [s.round_latency_ms for s in self.history]
        byzantine_ratios = [
            s.byzantine_nodes / s.total_nodes if s.total_nodes > 0 else 0
            for s in self.history
        ]

        import numpy as np

        return {
            "rounds": len(self.history),
            "first_round": self.history[0].round_num,
            "last_round": self.history[-1].round_num,
            "detection_latency_ms": {
                "mean": float(np.mean(detection_latencies)),
                "p50": float(np.percentile(detection_latencies, 50)),
                "p95": float(np.percentile(detection_latencies, 95)),
                "p99": float(np.percentile(detection_latencies, 99)),
                "max": float(np.max(detection_latencies)),
            },
            "round_latency_ms": {
                "mean": float(np.mean(round_latencies)),
                "p50": float(np.percentile(round_latencies, 50)),
                "p95": float(np.percentile(round_latencies, 95)),
                "p99": float(np.percentile(round_latencies, 99)),
                "max": float(np.max(round_latencies)),
            },
            "byzantine_ratio": {
                "mean": float(np.mean(byzantine_ratios)),
                "max": float(np.max(byzantine_ratios)),
            },
            "sla_compliance": {
                "detection_under_100ms": sum(
                    1 for d in detection_latencies if d < DETECTION_LATENCY_TARGET_MS
                ) / len(detection_latencies),
                "detection_under_500ms": sum(
                    1 for d in detection_latencies if d < DETECTION_LATENCY_WARN_MS
                ) / len(detection_latencies),
            },
        }

    def check_health(self) -> Dict[str, Any]:
        """Check system health based on recent performance."""
        if len(self.history) < 10:
            return {"status": "insufficient_data", "rounds": len(self.history)}

        recent = self.history[-10:]

        # Check latency SLA
        avg_latency = sum(s.detection_latency_ms for s in recent) / len(recent)
        latency_ok = avg_latency < DETECTION_LATENCY_TARGET_MS

        # Check Byzantine ratio
        avg_byzantine = sum(
            s.byzantine_nodes / s.total_nodes if s.total_nodes > 0 else 0
            for s in recent
        ) / len(recent)
        byzantine_ok = avg_byzantine < 0.45

        # Overall health
        status = "healthy" if latency_ok and byzantine_ok else "degraded"

        return {
            "status": status,
            "latency_ok": latency_ok,
            "avg_latency_ms": avg_latency,
            "byzantine_ok": byzantine_ok,
            "avg_byzantine_ratio": avg_byzantine,
            "rounds_analyzed": len(recent),
        }
