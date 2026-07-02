# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Production Monitoring for ZeroTrustML Credits System

Tracks key performance indicators and system health metrics for production deployment.
Provides alerting, logging, and metrics collection for operations teams.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Current system performance metrics"""

    timestamp: datetime

    # Throughput metrics
    credits_issued_last_minute: int = 0
    credits_issued_last_hour: int = 0
    events_processed_last_minute: int = 0
    events_processed_last_hour: int = 0

    # Performance metrics
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0

    # Detection metrics
    byzantine_detections_last_hour: int = 0
    detection_rate: float = 0.0

    # System health
    active_nodes: int = 0
    healthy_nodes: int = 0
    error_rate: float = 0.0
    rate_limit_violations: int = 0

    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    storage_usage_mb: float = 0.0


@dataclass
class Alert:
    """System alert for operational issues"""

    severity: str  # "info", "warning", "critical"
    category: str  # "performance", "security", "availability", "capacity"
    message: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold: float
    resolution: Optional[str] = None


class MetricsCollector:
    """Collects and aggregates metrics from ZeroTrustML Credits system"""

    def __init__(self, window_size_minutes: int = 60):
        self.window_size = window_size_minutes

        # Time-series data (sliding windows)
        self.credit_events = deque(maxlen=window_size_minutes * 60)  # 1 per second
        self.response_times = deque(maxlen=1000)  # Last 1000 requests
        self.byzantine_detections = deque(maxlen=window_size_minutes * 60)
        self.errors = deque(maxlen=window_size_minutes * 60)
        self.rate_limit_hits = deque(maxlen=window_size_minutes * 60)

        # Current state
        self.active_nodes = set()
        self.healthy_nodes = set()

        logger.info(f"MetricsCollector initialized with {window_size_minutes}min window")

    def record_credit_issued(
        self,
        node_id: str,
        credits: float,
        event_type: str,
        response_time_ms: float
    ) -> None:
        """Record a credit issuance event"""
        event = {
            'timestamp': datetime.now(),
            'node_id': node_id,
            'credits': credits,
            'event_type': event_type,
            'response_time_ms': response_time_ms
        }
        self.credit_events.append(event)
        self.response_times.append(response_time_ms)
        self.active_nodes.add(node_id)

    def record_byzantine_detection(
        self,
        detector_id: str,
        detected_id: str,
        evidence: Dict[str, Any]
    ) -> None:
        """Record a Byzantine node detection"""
        detection = {
            'timestamp': datetime.now(),
            'detector_id': detector_id,
            'detected_id': detected_id,
            'evidence': evidence
        }
        self.byzantine_detections.append(detection)

        # Remove detected node from healthy set
        if detected_id in self.healthy_nodes:
            self.healthy_nodes.remove(detected_id)

    def record_error(
        self,
        node_id: str,
        error_type: str,
        message: str
    ) -> None:
        """Record a system error"""
        error = {
            'timestamp': datetime.now(),
            'node_id': node_id,
            'error_type': error_type,
            'message': message
        }
        self.errors.append(error)

    def record_rate_limit_hit(self, node_id: str, event_type: str) -> None:
        """Record a rate limit violation"""
        hit = {
            'timestamp': datetime.now(),
            'node_id': node_id,
            'event_type': event_type
        }
        self.rate_limit_hits.append(hit)

    def record_node_health(self, node_id: str, is_healthy: bool) -> None:
        """Update node health status"""
        if is_healthy:
            self.healthy_nodes.add(node_id)
        else:
            self.healthy_nodes.discard(node_id)

    def get_current_metrics(self) -> SystemMetrics:
        """Calculate current system metrics"""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        one_hour_ago = now - timedelta(hours=1)

        # Filter events by time window
        recent_credits = [e for e in self.credit_events if e['timestamp'] > one_minute_ago]
        hourly_credits = [e for e in self.credit_events if e['timestamp'] > one_hour_ago]
        recent_detections = [d for d in self.byzantine_detections if d['timestamp'] > one_hour_ago]
        recent_errors = [e for e in self.errors if e['timestamp'] > one_minute_ago]
        recent_rate_hits = [h for h in self.rate_limit_hits if h['timestamp'] > one_hour_ago]

        # Calculate throughput
        credits_last_minute = sum(e['credits'] for e in recent_credits)
        credits_last_hour = sum(e['credits'] for e in hourly_credits)
        events_last_minute = len(recent_credits)
        events_last_hour = len(hourly_credits)

        # Calculate response times
        if self.response_times:
            sorted_times = sorted(self.response_times)
            avg_response = sum(sorted_times) / len(sorted_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            p95_response = sorted_times[p95_idx] if p95_idx < len(sorted_times) else 0
            p99_response = sorted_times[p99_idx] if p99_idx < len(sorted_times) else 0
        else:
            avg_response = p95_response = p99_response = 0.0

        # Calculate detection rate
        total_events = len(hourly_credits)
        detections = len(recent_detections)
        detection_rate = (detections / total_events) if total_events > 0 else 0.0

        # Calculate error rate
        total_recent = len(recent_credits)
        error_rate = (len(recent_errors) / total_recent) if total_recent > 0 else 0.0

        return SystemMetrics(
            timestamp=now,
            credits_issued_last_minute=int(credits_last_minute),
            credits_issued_last_hour=int(credits_last_hour),
            events_processed_last_minute=events_last_minute,
            events_processed_last_hour=events_last_hour,
            avg_response_time_ms=avg_response,
            p95_response_time_ms=p95_response,
            p99_response_time_ms=p99_response,
            byzantine_detections_last_hour=detections,
            detection_rate=detection_rate,
            active_nodes=len(self.active_nodes),
            healthy_nodes=len(self.healthy_nodes),
            error_rate=error_rate,
            rate_limit_violations=len(recent_rate_hits)
        )


class AlertManager:
    """Manages system alerts and notifications"""

    def __init__(self):
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []

        # Thresholds (configurable)
        self.thresholds = {
            'max_response_time_ms': 5000,  # 5 seconds
            'max_p99_response_time_ms': 10000,  # 10 seconds
            'max_error_rate': 0.05,  # 5%
            'min_detection_rate': 0.85,  # 85%
            'max_rate_limit_violations_per_hour': 100,
            'min_healthy_node_percentage': 0.90  # 90%
        }

        logger.info("AlertManager initialized with default thresholds")

    def check_metrics(self, metrics: SystemMetrics) -> List[Alert]:
        """Check metrics against thresholds and generate alerts"""
        new_alerts = []

        # Performance alerts
        if metrics.avg_response_time_ms > self.thresholds['max_response_time_ms']:
            alert = Alert(
                severity="warning",
                category="performance",
                message=f"Average response time ({metrics.avg_response_time_ms:.0f}ms) exceeds threshold",
                timestamp=metrics.timestamp,
                metric_name="avg_response_time_ms",
                current_value=metrics.avg_response_time_ms,
                threshold=self.thresholds['max_response_time_ms'],
                resolution="Check system load, optimize queries, scale horizontally"
            )
            new_alerts.append(alert)

        if metrics.p99_response_time_ms > self.thresholds['max_p99_response_time_ms']:
            alert = Alert(
                severity="critical",
                category="performance",
                message=f"P99 response time ({metrics.p99_response_time_ms:.0f}ms) critically high",
                timestamp=metrics.timestamp,
                metric_name="p99_response_time_ms",
                current_value=metrics.p99_response_time_ms,
                threshold=self.thresholds['max_p99_response_time_ms'],
                resolution="Immediate investigation required - possible system bottleneck"
            )
            new_alerts.append(alert)

        # Error rate alerts
        if metrics.error_rate > self.thresholds['max_error_rate']:
            alert = Alert(
                severity="critical",
                category="availability",
                message=f"Error rate ({metrics.error_rate*100:.1f}%) exceeds threshold",
                timestamp=metrics.timestamp,
                metric_name="error_rate",
                current_value=metrics.error_rate,
                threshold=self.thresholds['max_error_rate'],
                resolution="Check logs for error patterns, verify Holochain connectivity"
            )
            new_alerts.append(alert)

        # Detection rate alerts
        if metrics.detection_rate < self.thresholds['min_detection_rate'] and metrics.byzantine_detections_last_hour > 0:
            alert = Alert(
                severity="warning",
                category="security",
                message=f"Byzantine detection rate ({metrics.detection_rate*100:.1f}%) below threshold",
                timestamp=metrics.timestamp,
                metric_name="detection_rate",
                current_value=metrics.detection_rate,
                threshold=self.thresholds['min_detection_rate'],
                resolution="Verify PoGQ validation system, check validator configuration"
            )
            new_alerts.append(alert)

        # Capacity alerts
        if metrics.rate_limit_violations > self.thresholds['max_rate_limit_violations_per_hour']:
            alert = Alert(
                severity="warning",
                category="capacity",
                message=f"Rate limit violations ({metrics.rate_limit_violations}) exceed threshold",
                timestamp=metrics.timestamp,
                metric_name="rate_limit_violations",
                current_value=float(metrics.rate_limit_violations),
                threshold=float(self.thresholds['max_rate_limit_violations_per_hour']),
                resolution="Review rate limits, possible spam attack or legitimate growth"
            )
            new_alerts.append(alert)

        # Node health alerts
        if metrics.active_nodes > 0:
            healthy_percentage = metrics.healthy_nodes / metrics.active_nodes
            if healthy_percentage < self.thresholds['min_healthy_node_percentage']:
                alert = Alert(
                    severity="critical",
                    category="availability",
                    message=f"Healthy node percentage ({healthy_percentage*100:.1f}%) below threshold",
                    timestamp=metrics.timestamp,
                    metric_name="healthy_node_percentage",
                    current_value=healthy_percentage,
                    threshold=self.thresholds['min_healthy_node_percentage'],
                    resolution="Investigate node failures, check network connectivity"
                )
                new_alerts.append(alert)

        # Store alerts
        self.active_alerts = new_alerts
        self.alert_history.extend(new_alerts)

        # Log alerts
        for alert in new_alerts:
            if alert.severity == "critical":
                logger.critical(f"[ALERT] {alert.message}")
            elif alert.severity == "warning":
                logger.warning(f"[ALERT] {alert.message}")
            else:
                logger.info(f"[ALERT] {alert.message}")

        return new_alerts

    def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        if severity:
            return [a for a in self.active_alerts if a.severity == severity]
        return self.active_alerts

    def clear_alert(self, alert: Alert) -> None:
        """Mark an alert as resolved"""
        if alert in self.active_alerts:
            self.active_alerts.remove(alert)
            logger.info(f"Alert resolved: {alert.message}")


class ProductionMonitor:
    """
    Production monitoring system for ZeroTrustML Credits

    Provides:
    - Real-time metrics collection
    - Alerting on anomalies
    - Health checks
    - Performance tracking
    """

    def __init__(self, check_interval_seconds: int = 60):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.check_interval = check_interval_seconds
        self.is_running = False

        logger.info(f"ProductionMonitor initialized (check interval: {check_interval_seconds}s)")

    async def start(self) -> None:
        """Start continuous monitoring"""
        self.is_running = True
        logger.info("Production monitoring started")

        while self.is_running:
            try:
                # Collect current metrics
                metrics = self.metrics_collector.get_current_metrics()

                # Check for alerts
                alerts = self.alert_manager.check_metrics(metrics)

                # Log summary
                logger.info(
                    f"[Monitor] "
                    f"Events/min: {metrics.events_processed_last_minute}, "
                    f"Avg response: {metrics.avg_response_time_ms:.0f}ms, "
                    f"Active nodes: {metrics.active_nodes}, "
                    f"Alerts: {len(alerts)}"
                )

                # Wait for next check
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    def stop(self) -> None:
        """Stop monitoring"""
        self.is_running = False
        logger.info("Production monitoring stopped")

    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        metrics = self.metrics_collector.get_current_metrics()
        alerts = self.alert_manager.get_active_alerts()

        return {
            'timestamp': metrics.timestamp.isoformat(),
            'system_status': 'healthy' if len(alerts) == 0 else 'degraded' if not any(a.severity == 'critical' for a in alerts) else 'critical',
            'metrics': {
                'throughput': {
                    'events_per_minute': metrics.events_processed_last_minute,
                    'events_per_hour': metrics.events_processed_last_hour,
                    'credits_per_minute': metrics.credits_issued_last_minute,
                    'credits_per_hour': metrics.credits_issued_last_hour
                },
                'performance': {
                    'avg_response_ms': round(metrics.avg_response_time_ms, 2),
                    'p95_response_ms': round(metrics.p95_response_time_ms, 2),
                    'p99_response_ms': round(metrics.p99_response_time_ms, 2)
                },
                'security': {
                    'byzantine_detections_last_hour': metrics.byzantine_detections_last_hour,
                    'detection_rate': round(metrics.detection_rate, 4)
                },
                'health': {
                    'active_nodes': metrics.active_nodes,
                    'healthy_nodes': metrics.healthy_nodes,
                    'error_rate': round(metrics.error_rate, 4),
                    'rate_limit_violations': metrics.rate_limit_violations
                }
            },
            'alerts': [
                {
                    'severity': a.severity,
                    'category': a.category,
                    'message': a.message,
                    'resolution': a.resolution
                }
                for a in alerts
            ]
        }


# Example usage
async def example_monitoring():
    """Example of production monitoring setup"""
    monitor = ProductionMonitor(check_interval_seconds=60)

    # Start monitoring in background
    monitor_task = asyncio.create_task(monitor.start())

    # Simulate some events
    for i in range(10):
        monitor.metrics_collector.record_credit_issued(
            node_id=f"node_{i % 3}",
            credits=100.0,
            event_type="quality_gradient",
            response_time_ms=500 + i * 50
        )
        await asyncio.sleep(0.1)

    # Wait a bit
    await asyncio.sleep(2)

    # Get status report
    report = monitor.get_status_report()
    print(json.dumps(report, indent=2))

    # Stop monitoring
    monitor.stop()
    await monitor_task


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(example_monitoring())
