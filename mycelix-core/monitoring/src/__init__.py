"""
Mycelix FL Monitoring Module

Provides metrics collection, export, and integration utilities
for Mycelix Federated Learning systems.

Usage:
    from monitoring.src import metrics, MetricsExporter

    # Record metrics
    metrics.rounds_total.inc()
    metrics.byzantine_ratio.set(0.15)

    # Start metrics server
    exporter = MetricsExporter(port=9100)
    await exporter.start()
"""

from .metrics_exporter import (
    MetricsExporter,
    MycelixMetrics,
    metrics,
    Counter,
    Gauge,
    Histogram,
    Summary,
    timed,
    counted,
)

__all__ = [
    'MetricsExporter',
    'MycelixMetrics',
    'metrics',
    'Counter',
    'Gauge',
    'Histogram',
    'Summary',
    'timed',
    'counted',
]

__version__ = '1.0.0'
