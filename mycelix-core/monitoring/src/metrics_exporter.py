# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix FL Metrics Exporter

A comprehensive Prometheus metrics exporter for Mycelix FL services.
Provides HTTP endpoints for scraping metrics from FL coordinator,
Byzantine detection, and storage backends.

Usage:
    from metrics_exporter import MetricsExporter, metrics

    # Start the metrics server
    exporter = MetricsExporter(port=9100)
    await exporter.start()

    # Record metrics from your code
    metrics.gradients_received.inc()
    metrics.round_latency.observe(150.5)
    metrics.byzantine_ratio.set(0.15)

Author: Luminous Dynamics
Date: January 2026
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from aiohttp import web
import threading
import json

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus-Compatible Metric Types
# =============================================================================

class Counter:
    """Prometheus counter metric - only increases."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment counter by value."""
        if value < 0:
            raise ValueError("Counter can only be incremented")
        label_key = self._labels_to_tuple(labels)
        with self._lock:
            self._values[label_key] += value

    def labels(self, **kwargs) -> 'LabeledCounter':
        """Return a counter with specific labels."""
        return LabeledCounter(self, kwargs)

    def _labels_to_tuple(self, labels: Optional[Dict[str, str]]) -> tuple:
        if not labels:
            return ()
        return tuple(labels.get(l, "") for l in self.label_names)

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all metric values for export."""
        result = []
        for label_values, value in self._values.items():
            labels = dict(zip(self.label_names, label_values)) if label_values else {}
            result.append({"labels": labels, "value": value})
        return result


class LabeledCounter:
    """Counter with pre-set labels."""

    def __init__(self, counter: Counter, labels: Dict[str, str]):
        self._counter = counter
        self._labels = labels

    def inc(self, value: float = 1.0):
        self._counter.inc(value, self._labels)


class Gauge:
    """Prometheus gauge metric - can increase or decrease."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge to value."""
        label_key = self._labels_to_tuple(labels)
        with self._lock:
            self._values[label_key] = value

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment gauge by value."""
        label_key = self._labels_to_tuple(labels)
        with self._lock:
            self._values[label_key] += value

    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Decrement gauge by value."""
        label_key = self._labels_to_tuple(labels)
        with self._lock:
            self._values[label_key] -= value

    def labels(self, **kwargs) -> 'LabeledGauge':
        """Return a gauge with specific labels."""
        return LabeledGauge(self, kwargs)

    def _labels_to_tuple(self, labels: Optional[Dict[str, str]]) -> tuple:
        if not labels:
            return ()
        return tuple(labels.get(l, "") for l in self.label_names)

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all metric values for export."""
        result = []
        for label_values, value in self._values.items():
            labels = dict(zip(self.label_names, label_values)) if label_values else {}
            result.append({"labels": labels, "value": value})
        return result


class LabeledGauge:
    """Gauge with pre-set labels."""

    def __init__(self, gauge: Gauge, labels: Dict[str, str]):
        self._gauge = gauge
        self._labels = labels

    def set(self, value: float):
        self._gauge.set(value, self._labels)

    def inc(self, value: float = 1.0):
        self._gauge.inc(value, self._labels)

    def dec(self, value: float = 1.0):
        self._gauge.dec(value, self._labels)


class Histogram:
    """Prometheus histogram metric - tracks distribution of values."""

    DEFAULT_BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, float('inf'))

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: tuple = DEFAULT_BUCKETS
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = sorted([b for b in buckets if b != float('inf')]) + [float('inf')]
        self._bucket_counts: Dict[tuple, Dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self.buckets}
        )
        self._sum: Dict[tuple, float] = defaultdict(float)
        self._count: Dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Record an observation."""
        label_key = self._labels_to_tuple(labels)
        with self._lock:
            self._sum[label_key] += value
            self._count[label_key] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[label_key][bucket] += 1

    def labels(self, **kwargs) -> 'LabeledHistogram':
        """Return a histogram with specific labels."""
        return LabeledHistogram(self, kwargs)

    def time(self) -> 'HistogramTimer':
        """Return a timer context manager."""
        return HistogramTimer(self, None)

    def _labels_to_tuple(self, labels: Optional[Dict[str, str]]) -> tuple:
        if not labels:
            return ()
        return tuple(labels.get(l, "") for l in self.label_names)

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all metric values for export."""
        result = []
        for label_values in set(self._sum.keys()) | set(self._count.keys()):
            labels = dict(zip(self.label_names, label_values)) if label_values else {}
            result.append({
                "labels": labels,
                "buckets": dict(self._bucket_counts[label_values]),
                "sum": self._sum[label_values],
                "count": self._count[label_values]
            })
        return result


class LabeledHistogram:
    """Histogram with pre-set labels."""

    def __init__(self, histogram: Histogram, labels: Dict[str, str]):
        self._histogram = histogram
        self._labels = labels

    def observe(self, value: float):
        self._histogram.observe(value, self._labels)

    def time(self) -> 'HistogramTimer':
        return HistogramTimer(self._histogram, self._labels)


class HistogramTimer:
    """Context manager for timing operations."""

    def __init__(self, histogram: Histogram, labels: Optional[Dict[str, str]]):
        self._histogram = histogram
        self._labels = labels
        self._start: Optional[float] = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is not None:
            elapsed_ms = (time.perf_counter() - self._start) * 1000
            self._histogram.observe(elapsed_ms, self._labels)


class Summary:
    """Prometheus summary metric - calculates quantiles."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._max_samples = 1000

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Record an observation."""
        label_key = self._labels_to_tuple(labels)
        with self._lock:
            self._values[label_key].append(value)
            if len(self._values[label_key]) > self._max_samples:
                self._values[label_key] = self._values[label_key][-self._max_samples:]

    def _labels_to_tuple(self, labels: Optional[Dict[str, str]]) -> tuple:
        if not labels:
            return ()
        return tuple(labels.get(l, "") for l in self.label_names)

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all metric values for export."""
        result = []
        for label_values, values in self._values.items():
            if not values:
                continue
            labels = dict(zip(self.label_names, label_values)) if label_values else {}
            sorted_vals = sorted(values)
            count = len(sorted_vals)
            result.append({
                "labels": labels,
                "count": count,
                "sum": sum(sorted_vals),
                "quantiles": {
                    0.5: sorted_vals[int(count * 0.5)] if count else 0,
                    0.9: sorted_vals[int(count * 0.9)] if count else 0,
                    0.99: sorted_vals[int(count * 0.99)] if count else 0,
                }
            })
        return result


# =============================================================================
# Metrics Registry
# =============================================================================

class MycelixMetrics:
    """Central registry for all Mycelix FL metrics."""

    def __init__(self, prefix: str = "mycelix_fl"):
        self.prefix = prefix

        # ==================== FL Training Metrics ====================
        self.rounds_total = Counter(
            f"{prefix}_rounds_total",
            "Total FL training rounds completed"
        )
        self.gradients_received = Counter(
            f"{prefix}_gradients_received_total",
            "Total gradients received from nodes"
        )
        self.gradients_healed = Counter(
            f"{prefix}_gradients_healed_total",
            "Total gradients that required healing"
        )
        self.errors_total = Counter(
            f"{prefix}_errors_total",
            "Total errors by type",
            labels=["type"]
        )

        self.active_nodes = Gauge(
            f"{prefix}_active_nodes",
            "Currently active FL nodes"
        )
        self.byzantine_ratio = Gauge(
            f"{prefix}_byzantine_ratio",
            "Current Byzantine node ratio"
        )
        self.system_phi = Gauge(
            f"{prefix}_system_phi",
            "System integrated information (Phi)"
        )
        self.compression_ratio = Gauge(
            f"{prefix}_compression_ratio",
            "Current gradient compression ratio"
        )

        self.round_latency_ms = Histogram(
            f"{prefix}_round_latency_ms",
            "FL round latency in milliseconds",
            buckets=(100, 500, 1000, 2500, 5000, 10000, 30000, 60000)
        )
        self.encoding_latency_ms = Histogram(
            f"{prefix}_encoding_latency_ms",
            "Gradient encoding latency in milliseconds"
        )
        self.shapley_computation_ms = Histogram(
            f"{prefix}_shapley_computation_ms",
            "Shapley value computation time in milliseconds"
        )

        # ==================== Byzantine Detection Metrics ====================
        self.byzantine_detected = Counter(
            f"{prefix}_byzantine_detected_total",
            "Total Byzantine nodes detected",
            labels=["attack_type"]
        )
        self.validation_result = Counter(
            f"{prefix}_validation_result_total",
            "Gradient validation results",
            labels=["result"]
        )

        self.detector_score = Gauge(
            f"{prefix}_detector_score",
            "Byzantine detector confidence score",
            labels=["detector"]
        )
        self.node_reputation = Gauge(
            f"{prefix}_node_reputation",
            "Node reputation score",
            labels=["node_id"]
        )

        self.detection_latency_ms = Histogram(
            f"{prefix}_detection_latency_ms",
            "Byzantine detection latency in milliseconds",
            buckets=(10, 25, 50, 100, 250, 500, 1000)
        )

        # ==================== Node Connectivity Metrics ====================
        self.node_connections = Counter(
            f"{prefix}_node_connections_total",
            "Total node connection events"
        )
        self.node_disconnections = Counter(
            f"{prefix}_node_disconnections_total",
            "Total node disconnection events"
        )

        # ==================== Storage Metrics ====================
        self.storage_operations = Counter(
            f"{prefix}_storage_operations_total",
            "Storage operations by backend and type",
            labels=["backend", "operation"]
        )
        self.storage_errors = Counter(
            f"{prefix}_storage_errors_total",
            "Storage errors by backend",
            labels=["backend"]
        )

        self.storage_latency = Histogram(
            f"{prefix}_storage_operation_duration_seconds",
            "Storage operation duration in seconds",
            labels=["backend", "operation"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
        )

        # ==================== Cache Metrics ====================
        self.cache_hits = Counter(
            f"{prefix}_cache_hits_total",
            "Cache hit count"
        )
        self.cache_misses = Counter(
            f"{prefix}_cache_misses_total",
            "Cache miss count"
        )

        # ==================== Network Metrics ====================
        self.messages_sent = Counter(
            f"{prefix}_messages_sent_total",
            "Messages sent by type",
            labels=["message_type"]
        )
        self.messages_received = Counter(
            f"{prefix}_messages_received_total",
            "Messages received by type",
            labels=["message_type"]
        )

        self.message_queue_size = Gauge(
            f"{prefix}_message_queue_size",
            "Current message queue depth"
        )

        self.network_latency = Histogram(
            f"{prefix}_network_latency_seconds",
            "Network latency between services",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
        )
        self.message_queue_wait_ms = Histogram(
            f"{prefix}_message_queue_wait_ms",
            "Time spent waiting in message queue"
        )

        # ==================== Compliance Metrics ====================
        self.privacy_violations = Counter(
            f"{prefix}_privacy_violations_total",
            "Privacy violation count"
        )
        self.unencrypted_transmissions = Counter(
            f"{prefix}_unencrypted_transmissions_total",
            "Unencrypted data transmission count"
        )
        self.last_audit_entry = Gauge(
            f"{prefix}_last_audit_entry_timestamp",
            "Timestamp of last audit entry"
        )
        self.zkproof_verifications = Counter(
            f"{prefix}_zkproof_verifications_total",
            "ZK proof verifications"
        )
        self.zkproof_failures = Counter(
            f"{prefix}_zkproof_failures_total",
            "ZK proof verification failures"
        )

        # All metrics for iteration
        self._all_metrics: List = [
            self.rounds_total, self.gradients_received, self.gradients_healed,
            self.errors_total, self.active_nodes, self.byzantine_ratio,
            self.system_phi, self.compression_ratio, self.round_latency_ms,
            self.encoding_latency_ms, self.shapley_computation_ms,
            self.byzantine_detected, self.validation_result, self.detector_score,
            self.node_reputation, self.detection_latency_ms, self.node_connections,
            self.node_disconnections, self.storage_operations, self.storage_errors,
            self.storage_latency, self.cache_hits, self.cache_misses,
            self.messages_sent, self.messages_received, self.message_queue_size,
            self.network_latency, self.message_queue_wait_ms, self.privacy_violations,
            self.unencrypted_transmissions, self.last_audit_entry,
            self.zkproof_verifications, self.zkproof_failures
        ]

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = []

        for metric in self._all_metrics:
            metric_name = metric.name
            metric_type = type(metric).__name__.lower()

            lines.append(f"# HELP {metric_name} {metric.description}")
            lines.append(f"# TYPE {metric_name} {metric_type}")

            for item in metric.collect():
                labels = item.get("labels", {})
                label_str = ""
                if labels:
                    label_parts = [f'{k}="{v}"' for k, v in labels.items()]
                    label_str = "{" + ",".join(label_parts) + "}"

                if isinstance(metric, Histogram):
                    # Export histogram buckets
                    for bucket, count in item.get("buckets", {}).items():
                        bucket_label = f'le="{bucket}"' if bucket != float('inf') else 'le="+Inf"'
                        if labels:
                            bucket_label = label_str[:-1] + "," + bucket_label + "}"
                        else:
                            bucket_label = "{" + bucket_label + "}"
                        lines.append(f"{metric_name}_bucket{bucket_label} {count}")
                    lines.append(f"{metric_name}_sum{label_str} {item.get('sum', 0)}")
                    lines.append(f"{metric_name}_count{label_str} {item.get('count', 0)}")
                elif isinstance(metric, Summary):
                    for quantile, value in item.get("quantiles", {}).items():
                        q_label = f'quantile="{quantile}"'
                        if labels:
                            q_label = label_str[:-1] + "," + q_label + "}"
                        else:
                            q_label = "{" + q_label + "}"
                        lines.append(f"{metric_name}{q_label} {value}")
                    lines.append(f"{metric_name}_sum{label_str} {item.get('sum', 0)}")
                    lines.append(f"{metric_name}_count{label_str} {item.get('count', 0)}")
                else:
                    lines.append(f"{metric_name}{label_str} {item.get('value', 0)}")

        return "\n".join(lines)

    def export_json(self) -> Dict[str, Any]:
        """Export all metrics as JSON."""
        result = {}
        for metric in self._all_metrics:
            result[metric.name] = {
                "type": type(metric).__name__.lower(),
                "description": metric.description,
                "values": metric.collect()
            }
        return result


# Global metrics instance
metrics = MycelixMetrics()


# =============================================================================
# HTTP Metrics Server
# =============================================================================

class MetricsExporter:
    """HTTP server for Prometheus metrics scraping."""

    def __init__(self, port: int = 9100, host: str = None):
        import os
        if host is None:
            host = os.getenv("METRICS_HOST", "127.0.0.1")
        self.port = port
        self.host = host
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self._auth_token: Optional[str] = os.environ.get("METRICS_AUTH_TOKEN")
        # Paths that never require authentication (load balancer probes)
        self._unauthenticated_paths = {"/health", "/ready"}
        self._setup_routes()

    def _setup_routes(self):
        """Set up HTTP routes."""
        self.app.router.add_get("/metrics", self._handle_metrics)
        self.app.router.add_get("/health", self._handle_health)
        self.app.router.add_get("/ready", self._handle_ready)
        self.app.router.add_get("/json", self._handle_json)

    def _check_auth(self, request: web.Request) -> Optional[web.Response]:
        """Check Bearer token if METRICS_AUTH_TOKEN is set.

        Returns None if auth passes, or a 401 Response if it fails.
        Skips auth for health/ready probes.
        """
        if self._auth_token is None:
            return None
        if request.path in self._unauthenticated_paths:
            return None
        auth_header = request.headers.get("Authorization", "")
        if auth_header == f"Bearer {self._auth_token}":
            return None
        return web.Response(
            text=json.dumps({"error": "Unauthorized", "detail": "Missing or invalid Bearer token"}),
            status=401,
            content_type="application/json",
        )

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """Handle Prometheus metrics endpoint."""
        auth_error = self._check_auth(request)
        if auth_error is not None:
            return auth_error
        try:
            output = metrics.export_prometheus()
            return web.Response(
                text=output,
                content_type="text/plain; version=0.0.4; charset=utf-8"
            )
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return web.Response(text=f"Error: {e}", status=500)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle health check endpoint."""
        return web.Response(
            text=json.dumps({"status": "healthy", "timestamp": time.time()}),
            content_type="application/json"
        )

    async def _handle_ready(self, request: web.Request) -> web.Response:
        """Handle readiness check endpoint."""
        return web.Response(
            text=json.dumps({"status": "ready", "timestamp": time.time()}),
            content_type="application/json"
        )

    async def _handle_json(self, request: web.Request) -> web.Response:
        """Handle JSON metrics endpoint."""
        auth_error = self._check_auth(request)
        if auth_error is not None:
            return auth_error
        try:
            output = metrics.export_json()
            return web.Response(
                text=json.dumps(output, indent=2),
                content_type="application/json"
            )
        except Exception as e:
            logger.error(f"Error exporting metrics as JSON: {e}")
            return web.Response(text=f"Error: {e}", status=500)

    async def start(self):
        """Start the metrics server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        if self._auth_token is None:
            logger.warning(
                "METRICS_AUTH_TOKEN is not set — metrics authentication is DISABLED. "
                "Set the env var to require Bearer token auth on /metrics and /json endpoints."
            )
        else:
            logger.info("Bearer token authentication enabled for metrics endpoints.")
        logger.info(f"Metrics server started on http://{self.host}:{self.port}")

    async def stop(self):
        """Stop the metrics server."""
        if self.runner:
            await self.runner.cleanup()
            logger.info("Metrics server stopped")


# =============================================================================
# Utility Functions
# =============================================================================

def timed(histogram: Histogram):
    """Decorator to time function execution and record to histogram."""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                histogram.observe(elapsed_ms)

        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                histogram.observe(elapsed_ms)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def counted(counter: Counter, labels: Optional[Dict[str, str]] = None):
    """Decorator to count function invocations."""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            counter.inc(1, labels)
            return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            counter.inc(1, labels)
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# =============================================================================
# Example Usage
# =============================================================================

async def example_usage():
    """Example of using the metrics exporter."""
    # Start metrics server
    exporter = MetricsExporter(port=9100)
    await exporter.start()

    # Simulate some metrics
    for i in range(10):
        metrics.rounds_total.inc()
        metrics.gradients_received.inc(5)
        metrics.active_nodes.set(10 + i)
        metrics.byzantine_ratio.set(0.1 + i * 0.01)
        metrics.round_latency_ms.observe(100 + i * 10)
        metrics.detection_latency_ms.observe(50 + i * 5)
        metrics.detector_score.set(0.9, {"detector": "multi_layer"})
        metrics.node_reputation.set(0.8, {"node_id": f"node_{i}"})
        await asyncio.sleep(0.1)

    print(f"Metrics available at http://localhost:9100/metrics")
    print(f"JSON format at http://localhost:9100/json")

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await exporter.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
