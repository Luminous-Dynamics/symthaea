# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Monitoring Layer for Hybrid ZeroTrustML

Implements:
- Prometheus metrics collection
- Real-time Byzantine detection visualization
- Network topology tracking
- Performance dashboards
"""

import time
import asyncio
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
import numpy as np

# Prometheus imports
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        start_http_server, generate_latest, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not available. Install with: pip install prometheus-client")


@dataclass
class NodeMetrics:
    """Metrics for a single node"""
    node_id: int
    reputation_score: float = 0.0
    gradients_validated: int = 0
    gradients_rejected: int = 0
    byzantine_detected: int = 0
    is_blacklisted: bool = False
    last_seen: float = field(default_factory=time.time)
    network_latency_ms: float = 0.0
    validation_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            'node_id': self.node_id,
            'reputation_score': self.reputation_score,
            'gradients_validated': self.gradients_validated,
            'gradients_rejected': self.gradients_rejected,
            'byzantine_detected': self.byzantine_detected,
            'is_blacklisted': self.is_blacklisted,
            'last_seen': self.last_seen,
            'network_latency_ms': self.network_latency_ms,
            'validation_time_ms': self.validation_time_ms
        }


class PrometheusMetrics:
    """Prometheus metrics exporter"""

    def __init__(self, node_id: int, port: int = 9090):
        """
        Initialize Prometheus metrics

        Args:
            node_id: This node's ID
            port: Port for Prometheus HTTP server
        """
        if not PROMETHEUS_AVAILABLE:
            raise RuntimeError("prometheus_client required for monitoring")

        self.node_id = node_id
        self.port = port

        # Byzantine Detection Metrics
        self.byzantine_detected = Counter(
            'zerotrustml_byzantine_detected_total',
            'Total Byzantine nodes detected',
            ['detector_node_id']
        )

        self.reputation_score = Gauge(
            'zerotrustml_reputation_score',
            'Current reputation score for nodes',
            ['node_id']
        )

        self.gradients_validated = Counter(
            'zerotrustml_gradients_validated_total',
            'Total gradients validated',
            ['node_id', 'result']
        )

        self.blacklisted_nodes = Gauge(
            'zerotrustml_blacklisted_nodes',
            'Number of blacklisted nodes'
        )

        # Network Metrics
        self.network_latency = Histogram(
            'zerotrustml_network_latency_seconds',
            'Network latency for peer communication',
            ['peer_id'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )

        self.active_connections = Gauge(
            'zerotrustml_active_connections',
            'Number of active peer connections'
        )

        self.messages_sent = Counter(
            'zerotrustml_messages_sent_total',
            'Total messages sent',
            ['message_type']
        )

        self.messages_received = Counter(
            'zerotrustml_messages_received_total',
            'Total messages received',
            ['message_type']
        )

        # Performance Metrics
        self.validation_time = Histogram(
            'zerotrustml_validation_time_seconds',
            'Time to validate gradients',
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        )

        self.gradient_size = Histogram(
            'zerotrustml_gradient_size_bytes',
            'Size of gradients',
            buckets=[1024, 10240, 102400, 1024000, 10240000]
        )

        self.compression_ratio = Histogram(
            'zerotrustml_compression_ratio',
            'Gradient compression ratio',
            buckets=[1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        )

        self.cache_operations = Counter(
            'zerotrustml_cache_operations_total',
            'Cache operations',
            ['operation', 'result']
        )

        # Storage Metrics
        self.storage_operations = Counter(
            'zerotrustml_storage_operations_total',
            'Storage operations',
            ['backend', 'operation', 'result']
        )

        self.storage_latency = Histogram(
            'zerotrustml_storage_latency_seconds',
            'Storage operation latency',
            ['backend', 'operation'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )

        # Training Metrics
        self.training_rounds = Counter(
            'zerotrustml_training_rounds_total',
            'Total training rounds completed'
        )

        self.model_accuracy = Gauge(
            'zerotrustml_model_accuracy',
            'Current model accuracy'
        )

        # Start HTTP server
        try:
            start_http_server(port)
            print(f"✓ Prometheus metrics available at http://localhost:{port}")
        except Exception as e:
            print(f"Warning: Could not start Prometheus HTTP server: {e}")

    def record_byzantine_detection(self, detected_node_id: int):
        """Record Byzantine node detection"""
        self.byzantine_detected.labels(detector_node_id=self.node_id).inc()

    def update_reputation(self, node_id: int, score: float):
        """Update reputation score"""
        self.reputation_score.labels(node_id=node_id).set(score)

    def record_validation(self, node_id: int, is_valid: bool, validation_time: float):
        """Record gradient validation"""
        result = 'valid' if is_valid else 'invalid'
        self.gradients_validated.labels(node_id=node_id, result=result).inc()
        self.validation_time.observe(validation_time)

    def record_network_latency(self, peer_id: int, latency: float):
        """Record network latency"""
        self.network_latency.labels(peer_id=peer_id).observe(latency)

    def record_message(self, msg_type: str, is_sent: bool = True):
        """Record message sent/received"""
        if is_sent:
            self.messages_sent.labels(message_type=msg_type).inc()
        else:
            self.messages_received.labels(message_type=msg_type).inc()

    def record_compression(self, original_size: int, compressed_size: int):
        """Record compression metrics"""
        self.gradient_size.observe(original_size)
        ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        self.compression_ratio.observe(ratio)

    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation"""
        result = 'hit' if hit else 'miss'
        self.cache_operations.labels(operation=operation, result=result).inc()

    def record_storage_operation(
        self,
        backend: str,
        operation: str,
        latency: float,
        success: bool
    ):
        """Record storage operation"""
        result = 'success' if success else 'error'
        self.storage_operations.labels(
            backend=backend,
            operation=operation,
            result=result
        ).inc()
        self.storage_latency.labels(
            backend=backend,
            operation=operation
        ).observe(latency)


class NetworkTopologyMonitor:
    """Monitors and visualizes network topology"""

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.peers: Dict[int, NodeMetrics] = {}
        self.connections: Set[Tuple[int, int]] = set()

    def add_peer(self, peer_id: int, metrics: Optional[NodeMetrics] = None):
        """Add or update peer"""
        if metrics is None:
            metrics = NodeMetrics(node_id=peer_id)

        self.peers[peer_id] = metrics
        self.connections.add((min(self.node_id, peer_id), max(self.node_id, peer_id)))

    def update_peer_metrics(self, peer_id: int, **kwargs):
        """Update peer metrics"""
        if peer_id in self.peers:
            for key, value in kwargs.items():
                if hasattr(self.peers[peer_id], key):
                    setattr(self.peers[peer_id], key, value)

    def remove_peer(self, peer_id: int):
        """Remove disconnected peer"""
        if peer_id in self.peers:
            del self.peers[peer_id]

        # Remove connections
        self.connections = {
            (n1, n2) for n1, n2 in self.connections
            if n1 != peer_id and n2 != peer_id
        }

    def get_topology_json(self) -> str:
        """Get topology as JSON for visualization"""
        topology = {
            'nodes': [
                {'id': self.node_id, 'is_self': True, **NodeMetrics(self.node_id).to_dict()}
            ] + [
                {'id': peer_id, 'is_self': False, **metrics.to_dict()}
                for peer_id, metrics in self.peers.items()
            ],
            'links': [
                {'source': n1, 'target': n2}
                for n1, n2 in self.connections
            ]
        }

        return json.dumps(topology, indent=2)

    def get_statistics(self) -> Dict:
        """Get network statistics"""
        total_peers = len(self.peers)
        blacklisted = sum(1 for m in self.peers.values() if m.is_blacklisted)
        avg_reputation = np.mean([m.reputation_score for m in self.peers.values()]) if self.peers else 0.0
        avg_latency = np.mean([m.network_latency_ms for m in self.peers.values()]) if self.peers else 0.0

        return {
            'total_peers': total_peers,
            'blacklisted_peers': blacklisted,
            'active_connections': len(self.connections),
            'average_reputation': avg_reputation,
            'average_latency_ms': avg_latency
        }


class ByzantineDetectionVisualizer:
    """Visualizes Byzantine detection events in real-time"""

    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self.events: List[Dict] = []

    def record_detection(
        self,
        detector_id: int,
        byzantine_id: int,
        reason: str,
        confidence: float
    ):
        """Record Byzantine detection event"""
        event = {
            'timestamp': time.time(),
            'detector_id': detector_id,
            'byzantine_id': byzantine_id,
            'reason': reason,
            'confidence': confidence
        }

        self.events.append(event)

        # Keep only recent events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def get_recent_events(self, limit: int = 100) -> List[Dict]:
        """Get recent detection events"""
        return self.events[-limit:]

    def get_detection_timeline(self) -> Dict:
        """Get detection events grouped by time"""
        timeline = defaultdict(list)

        for event in self.events:
            # Group by minute
            minute = int(event['timestamp'] / 60) * 60
            timeline[minute].append(event)

        return dict(timeline)

    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        if not self.events:
            return {
                'total_detections': 0,
                'unique_byzantine_nodes': 0,
                'detections_per_minute': 0.0
            }

        unique_byzantine = len(set(e['byzantine_id'] for e in self.events))

        # Calculate detections per minute
        if len(self.events) > 1:
            time_span = self.events[-1]['timestamp'] - self.events[0]['timestamp']
            detections_per_minute = len(self.events) / (time_span / 60) if time_span > 0 else 0.0
        else:
            detections_per_minute = 0.0

        return {
            'total_detections': len(self.events),
            'unique_byzantine_nodes': unique_byzantine,
            'detections_per_minute': detections_per_minute
        }


class UptimeMonitor:
    """
    Monitors node uptime and issues network contribution credits

    Features:
    - Tracks node activity via heartbeats
    - Calculates uptime percentage over time windows
    - Issues hourly credits to nodes with ≥95% uptime
    """

    def __init__(self, credits_integration=None, min_uptime_threshold: float = 0.95):
        """
        Initialize uptime monitor

        Args:
            credits_integration: Optional ZeroTrustMLCreditsIntegration instance
            min_uptime_threshold: Minimum uptime percentage for credits (default: 0.95)
        """
        self.credits_integration = credits_integration
        self.min_uptime_threshold = min_uptime_threshold

        # Node activity tracking
        self.node_heartbeats: Dict[int, List[float]] = defaultdict(list)
        self.node_first_seen: Dict[int, float] = {}

        # Hourly credit issuance tracking
        self.last_credit_issuance: Dict[int, float] = {}

        # Background task
        self.monitor_task: Optional[asyncio.Task] = None

    def record_heartbeat(self, node_id: int):
        """Record node activity heartbeat"""
        current_time = time.time()

        if node_id not in self.node_first_seen:
            self.node_first_seen[node_id] = current_time

        # Keep last 24 hours of heartbeats (at 1 heartbeat/min = 1440 points)
        self.node_heartbeats[node_id].append(current_time)
        if len(self.node_heartbeats[node_id]) > 1440:
            self.node_heartbeats[node_id].pop(0)

    def calculate_uptime(self, node_id: int, window_hours: float = 1.0) -> float:
        """
        Calculate uptime percentage over time window

        Args:
            node_id: Node to calculate uptime for
            window_hours: Time window in hours (default: 1 hour)

        Returns:
            Uptime percentage (0.0-1.0)
        """
        if node_id not in self.node_heartbeats:
            return 0.0

        current_time = time.time()
        window_seconds = window_hours * 3600
        window_start = current_time - window_seconds

        # Get heartbeats in window
        recent_heartbeats = [
            hb for hb in self.node_heartbeats[node_id]
            if hb >= window_start
        ]

        if not recent_heartbeats:
            return 0.0

        # Node first seen within window? Adjust window
        if self.node_first_seen[node_id] > window_start:
            window_start = self.node_first_seen[node_id]
            window_seconds = current_time - window_start

        if window_seconds <= 0:
            return 1.0  # Just joined

        # Expected heartbeats (1 per minute)
        expected_heartbeats = window_seconds / 60
        actual_heartbeats = len(recent_heartbeats)

        # Uptime = actual / expected
        uptime = min(1.0, actual_heartbeats / expected_heartbeats)
        return uptime

    async def start_monitoring(self):
        """Start background uptime monitoring and credit issuance"""
        if self.monitor_task is not None:
            return  # Already running

        self.monitor_task = asyncio.create_task(self._hourly_credit_task())

    async def stop_monitoring(self):
        """Stop background monitoring"""
        if self.monitor_task is not None:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None

    async def _hourly_credit_task(self):
        """Background task that issues credits every hour"""
        while True:
            try:
                await asyncio.sleep(3600)  # 1 hour
                await self._issue_uptime_credits()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # TODO: Replace print with proper logging - see CLEANUP_LOG.md
                print(f"Error in uptime credit task: {e}")

    async def _issue_uptime_credits(self):
        """Issue credits to nodes with sufficient uptime"""
        if not self.credits_integration:
            return

        current_time = time.time()

        for node_id in self.node_heartbeats.keys():
            # Calculate uptime over last hour
            uptime = self.calculate_uptime(node_id, window_hours=1.0)

            # Check if eligible for credits
            if uptime >= self.min_uptime_threshold:
                # Check if we've issued credits recently (prevent double-issuance)
                last_issuance = self.last_credit_issuance.get(node_id, 0)
                time_since_last = current_time - last_issuance

                if time_since_last >= 3500:  # At least 58 minutes since last issuance
                    # Issue network contribution credits
                    try:
                        await self.credits_integration.on_network_contribution(
                            node_id=f"node_{node_id}",
                            uptime_percentage=uptime,
                            reputation_level="NORMAL",  # Could be dynamic based on reputation
                            hours_online=1.0
                        )
                        self.last_credit_issuance[node_id] = current_time
                    except Exception as e:
                        # TODO: Replace print with proper logging - see CLEANUP_LOG.md
                        print(f"Error issuing uptime credits for node {node_id}: {e}")

    def get_statistics(self) -> Dict:
        """Get uptime statistics"""
        current_time = time.time()

        stats = {
            'total_nodes_tracked': len(self.node_heartbeats),
            'nodes_by_uptime': {
                'excellent (≥95%)': 0,
                'good (80-95%)': 0,
                'poor (<80%)': 0
            }
        }

        for node_id in self.node_heartbeats.keys():
            uptime = self.calculate_uptime(node_id, window_hours=1.0)

            if uptime >= 0.95:
                stats['nodes_by_uptime']['excellent (≥95%)'] += 1
            elif uptime >= 0.80:
                stats['nodes_by_uptime']['good (80-95%)'] += 1
            else:
                stats['nodes_by_uptime']['poor (<80%)'] += 1

        return stats


# Example usage
if __name__ == "__main__":
    import numpy as np

    print("Monitoring Layer - Testing Components\n")

    # Test Prometheus metrics
    if PROMETHEUS_AVAILABLE:
        print("1. Testing Prometheus Metrics")
        print("-" * 40)

        metrics = PrometheusMetrics(node_id=1, port=9091)

        # Simulate some activity
        for i in range(10):
            metrics.record_validation(node_id=2, is_valid=True, validation_time=0.001)
            metrics.update_reputation(node_id=2, score=0.9)
            metrics.record_network_latency(peer_id=2, latency=0.05)
            metrics.record_message('gradient', is_sent=True)

        print("✓ Metrics recorded")
        print(f"  Access at: http://localhost:9091")

    # Test topology monitor
    print("\n2. Testing Network Topology Monitor")
    print("-" * 40)

    topology = NetworkTopologyMonitor(node_id=1)

    # Add some peers
    topology.add_peer(2, NodeMetrics(node_id=2, reputation_score=0.9))
    topology.add_peer(3, NodeMetrics(node_id=3, reputation_score=0.7))
    topology.add_peer(4, NodeMetrics(node_id=4, reputation_score=0.3, is_blacklisted=True))

    print(f"Topology JSON:\n{topology.get_topology_json()[:200]}...")

    stats = topology.get_statistics()
    print(f"\nNetwork Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test Byzantine detection visualizer
    print("\n3. Testing Byzantine Detection Visualizer")
    print("-" * 40)

    visualizer = ByzantineDetectionVisualizer()

    # Simulate detections
    visualizer.record_detection(
        detector_id=1,
        byzantine_id=4,
        reason="Gradient validation failed",
        confidence=0.95
    )

    visualizer.record_detection(
        detector_id=2,
        byzantine_id=4,
        reason="Anomaly detected",
        confidence=0.87
    )

    stats = visualizer.get_statistics()
    print(f"Detection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✓ Monitoring Layer initialized successfully")