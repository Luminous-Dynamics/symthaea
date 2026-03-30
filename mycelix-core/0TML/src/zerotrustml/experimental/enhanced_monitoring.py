# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Enhanced Monitoring (Phase 5 Enhancement 5)

Advanced monitoring with:
- Isolation Forest anomaly detection
- Predictive failure analysis
- Automated remediation
- Real-time alerting
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from collections import deque


class AnomalyType(Enum):
    """Types of anomalies"""
    GRADIENT_ANOMALY = "gradient_anomaly"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    NETWORK_ANOMALY = "network_anomaly"
    BYZANTINE_BEHAVIOR = "byzantine_behavior"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AlertSeverity
    score: float  # Anomaly score (0-1)
    node_id: Optional[int] = None
    details: Dict = field(default_factory=dict)
    recommended_action: Optional[str] = None


@dataclass
class PredictiveAlert:
    """Predictive failure alert"""
    timestamp: datetime
    predicted_failure_time: datetime
    failure_type: str
    confidence: float
    affected_nodes: List[int]
    preventive_actions: List[str]


class IsolationForestDetector:
    """
    Isolation Forest for anomaly detection

    Uses isolation trees to detect anomalies in:
    - Gradient magnitudes
    - Validation times
    - Network latencies
    - Resource usage

    Anomalies are data points that are easily isolated (few splits needed)
    """

    def __init__(
        self,
        n_trees: int = 100,
        sample_size: int = 256,
        contamination: float = 0.1
    ):
        """
        Args:
            n_trees: Number of isolation trees
            sample_size: Sample size for each tree
            contamination: Expected proportion of anomalies
        """
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.contamination = contamination

        # Isolation trees (simplified implementation)
        self.trees: List[Dict] = []
        self.threshold: Optional[float] = None

        # Training data for online learning
        self.training_window = deque(maxlen=1000)

    def fit(self, X: np.ndarray):
        """
        Train isolation forest on normal data

        Args:
            X: Training data (shape: [n_samples, n_features])
        """
        n_samples = X.shape[0]

        # Build isolation trees
        self.trees = []
        for _ in range(self.n_trees):
            # Sample data
            if n_samples > self.sample_size:
                indices = np.random.choice(n_samples, self.sample_size, replace=False)
                sample = X[indices]
            else:
                sample = X

            # Build tree (simplified - just store split info)
            tree = self._build_tree(sample, max_depth=int(np.log2(self.sample_size)))
            self.trees.append(tree)

        # Calculate anomaly threshold
        scores = self.score_samples(X)
        self.threshold = np.percentile(scores, (1 - self.contamination) * 100)

    def _build_tree(self, X: np.ndarray, max_depth: int, depth: int = 0) -> Dict:
        """Build a single isolation tree (recursive)"""
        if depth >= max_depth or len(X) <= 1:
            return {"type": "leaf", "size": len(X)}

        # Random split
        n_features = X.shape[1]
        split_feature = np.random.randint(0, n_features)
        split_value = np.random.uniform(
            X[:, split_feature].min(),
            X[:, split_feature].max()
        )

        # Split data
        left_mask = X[:, split_feature] < split_value
        left_data = X[left_mask]
        right_data = X[~left_mask]

        if len(left_data) == 0 or len(right_data) == 0:
            return {"type": "leaf", "size": len(X)}

        # Recursive build
        return {
            "type": "node",
            "feature": split_feature,
            "value": split_value,
            "left": self._build_tree(left_data, max_depth, depth + 1),
            "right": self._build_tree(right_data, max_depth, depth + 1)
        }

    def _path_length(self, x: np.ndarray, tree: Dict, depth: int = 0) -> float:
        """Calculate path length for a sample in a tree"""
        if tree["type"] == "leaf":
            # Average path length of unsuccessful search in BST
            size = tree["size"]
            if size <= 1:
                return depth
            return depth + self._c(size)

        # Traverse tree
        if x[tree["feature"]] < tree["value"]:
            return self._path_length(x, tree["left"], depth + 1)
        else:
            return self._path_length(x, tree["right"], depth + 1)

    def _c(self, n: int) -> float:
        """Average path length of unsuccessful search in BST"""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores for samples

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.trees:
            return np.zeros(len(X))

        # Average path length across all trees
        avg_path_lengths = np.array([
            np.mean([self._path_length(x, tree) for tree in self.trees])
            for x in X
        ])

        # Anomaly score (normalized)
        c_n = self._c(self.sample_size)
        anomaly_scores = 2 ** (-avg_path_lengths / c_n)

        return anomaly_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies

        Returns:
            Binary predictions (1 = anomaly, 0 = normal)
        """
        scores = self.score_samples(X)
        if self.threshold is None:
            # Use median as threshold if not fitted
            self.threshold = np.median(scores)

        return (scores > self.threshold).astype(int)

    def update_online(self, x: np.ndarray):
        """Update detector with new sample (online learning)"""
        self.training_window.append(x)

        # Retrain periodically
        if len(self.training_window) >= 100:
            X_train = np.array(list(self.training_window))
            self.fit(X_train)


class PredictiveFailureAnalyzer:
    """
    Predicts failures before they occur using time series analysis

    Uses sliding window statistics and trend detection
    (Simplified - in production would use LSTM or Prophet)
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Historical metrics
        self.metric_history: Dict[str, deque] = {
            "validation_times": deque(maxlen=window_size),
            "error_rates": deque(maxlen=window_size),
            "memory_usage": deque(maxlen=window_size),
            "cpu_usage": deque(maxlen=window_size)
        }

    def update_metrics(
        self,
        validation_time: float,
        error_rate: float,
        memory_usage: float,
        cpu_usage: float
    ):
        """Update historical metrics"""
        self.metric_history["validation_times"].append(validation_time)
        self.metric_history["error_rates"].append(error_rate)
        self.metric_history["memory_usage"].append(memory_usage)
        self.metric_history["cpu_usage"].append(cpu_usage)

    def predict_failures(self) -> List[PredictiveAlert]:
        """
        Predict potential failures

        Returns:
            List of predicted failures with recommended actions
        """
        alerts = []

        # Check validation time trend
        val_times = list(self.metric_history["validation_times"])
        if len(val_times) >= 20:
            recent_avg = np.mean(val_times[-10:])
            baseline_avg = np.mean(val_times[:10])

            if recent_avg > baseline_avg * 2:  # 2x slowdown
                # Predict failure in next 10 minutes
                predicted_time = datetime.now() + timedelta(minutes=10)
                alerts.append(PredictiveAlert(
                    timestamp=datetime.now(),
                    predicted_failure_time=predicted_time,
                    failure_type="validation_slowdown",
                    confidence=0.8,
                    affected_nodes=[],
                    preventive_actions=[
                        "Scale up worker nodes",
                        "Enable GPU acceleration",
                        "Clear cache",
                        "Check network latency"
                    ]
                ))

        # Check error rate trend
        error_rates = list(self.metric_history["error_rates"])
        if len(error_rates) >= 20:
            recent_errors = np.mean(error_rates[-10:])
            if recent_errors > 0.1:  # >10% error rate
                predicted_time = datetime.now() + timedelta(minutes=5)
                alerts.append(PredictiveAlert(
                    timestamp=datetime.now(),
                    predicted_failure_time=predicted_time,
                    failure_type="high_error_rate",
                    confidence=0.9,
                    affected_nodes=[],
                    preventive_actions=[
                        "Investigate Byzantine nodes",
                        "Check network connectivity",
                        "Review recent configuration changes"
                    ]
                ))

        # Check memory usage trend
        memory = list(self.metric_history["memory_usage"])
        if len(memory) >= 20:
            # Linear extrapolation
            trend = np.polyfit(range(len(memory)), memory, 1)[0]
            current = memory[-1]

            if trend > 0 and current + trend * 20 > 0.95:  # Will hit 95% in 20 steps
                predicted_time = datetime.now() + timedelta(minutes=20)
                alerts.append(PredictiveAlert(
                    timestamp=datetime.now(),
                    predicted_failure_time=predicted_time,
                    failure_type="memory_exhaustion",
                    confidence=0.85,
                    affected_nodes=[],
                    preventive_actions=[
                        "Clear gradient cache",
                        "Reduce batch size",
                        "Restart with more memory",
                        "Enable gradient compression"
                    ]
                ))

        return alerts


class AutomatedRemediator:
    """
    Automatically remediates detected issues

    Actions:
    - Restart degraded nodes
    - Scale resources
    - Clear caches
    - Adjust configuration
    """

    def __init__(self):
        self.remediation_history: List[Dict] = []
        self.cooldown_periods: Dict[str, datetime] = {}
        self.cooldown_duration = timedelta(minutes=5)

    async def remediate(
        self,
        anomaly: AnomalyDetection,
        action_callback: Optional[Callable] = None
    ) -> bool:
        """
        Execute remediation action

        Args:
            anomaly: Detected anomaly
            action_callback: Callback to execute action

        Returns:
            True if remediation successful
        """
        action = anomaly.recommended_action
        if not action:
            return False

        # Check cooldown
        if action in self.cooldown_periods:
            if datetime.now() < self.cooldown_periods[action]:
                print(f"⏳ Action '{action}' in cooldown period")
                return False

        print(f"🔧 Executing remediation: {action}")

        # Execute action
        success = False
        if action_callback:
            try:
                result = action_callback(action, anomaly)
                if asyncio.iscoroutine(result):
                    success = await result
                else:
                    success = result
            except Exception as e:
                print(f"❌ Remediation failed: {e}")
                success = False
        else:
            # Simulate action
            await asyncio.sleep(0.1)
            success = True

        # Record action
        self.remediation_history.append({
            "timestamp": datetime.now(),
            "action": action,
            "anomaly_type": anomaly.anomaly_type.value,
            "success": success
        })

        # Set cooldown
        self.cooldown_periods[action] = datetime.now() + self.cooldown_duration

        return success


class EnhancedMonitoringSystem:
    """
    Complete enhanced monitoring system

    Combines:
    - Isolation Forest anomaly detection
    - Predictive failure analysis
    - Automated remediation
    - Real-time alerting
    """

    def __init__(self):
        self.anomaly_detector = IsolationForestDetector(
            n_trees=100,
            contamination=0.1
        )
        self.failure_predictor = PredictiveFailureAnalyzer(window_size=100)
        self.remediator = AutomatedRemediator()

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

        # Statistics
        self.anomalies_detected = 0
        self.failures_predicted = 0
        self.remediations_executed = 0

    def register_alert_callback(self, callback: Callable):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)

    async def process_gradient_metrics(
        self,
        node_id: int,
        gradient_norm: float,
        validation_time: float,
        pogq_score: float
    ):
        """Process gradient validation metrics"""
        # Feature vector for anomaly detection
        features = np.array([gradient_norm, validation_time, pogq_score])

        # Detect anomalies
        is_anomaly = self.anomaly_detector.predict(features.reshape(1, -1))[0]

        if is_anomaly:
            anomaly_score = self.anomaly_detector.score_samples(features.reshape(1, -1))[0]

            # Determine anomaly type
            if pogq_score < 0.3:
                anomaly_type = AnomalyType.BYZANTINE_BEHAVIOR
                severity = AlertSeverity.CRITICAL
                action = "blacklist_node"
            elif validation_time > 1000:  # >1 second
                anomaly_type = AnomalyType.PERFORMANCE_ANOMALY
                severity = AlertSeverity.WARNING
                action = "restart_node"
            else:
                anomaly_type = AnomalyType.GRADIENT_ANOMALY
                severity = AlertSeverity.INFO
                action = None

            detection = AnomalyDetection(
                timestamp=datetime.now(),
                anomaly_type=anomaly_type,
                severity=severity,
                score=float(anomaly_score),
                node_id=node_id,
                details={
                    "gradient_norm": gradient_norm,
                    "validation_time": validation_time,
                    "pogq_score": pogq_score
                },
                recommended_action=action
            )

            self.anomalies_detected += 1

            # Alert
            await self._trigger_alerts(detection)

            # Auto-remediate if critical
            if severity == AlertSeverity.CRITICAL:
                await self.remediator.remediate(detection)
                self.remediations_executed += 1

        # Update online learning
        self.anomaly_detector.update_online(features)

    async def update_system_metrics(
        self,
        validation_time: float,
        error_rate: float,
        memory_usage: float,
        cpu_usage: float
    ):
        """Update system-level metrics and predict failures"""
        self.failure_predictor.update_metrics(
            validation_time,
            error_rate,
            memory_usage,
            cpu_usage
        )

        # Predict failures
        predicted_failures = self.failure_predictor.predict_failures()

        for prediction in predicted_failures:
            self.failures_predicted += 1

            # Alert
            await self._trigger_prediction_alerts(prediction)

    async def _trigger_alerts(self, detection: AnomalyDetection):
        """Trigger alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                result = callback(detection)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"Alert callback error: {e}")

    async def _trigger_prediction_alerts(self, prediction: PredictiveAlert):
        """Trigger prediction alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                result = callback(prediction)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"Prediction callback error: {e}")

    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        return {
            "anomalies_detected": self.anomalies_detected,
            "failures_predicted": self.failures_predicted,
            "remediations_executed": self.remediations_executed,
            "remediation_history": self.remediator.remediation_history[-10:],
            "detection_rate": (
                self.anomalies_detected / 1000 if self.anomalies_detected > 0 else 0
            )
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED MONITORING - DEMONSTRATION")
    print("=" * 60)

    async def demo():
        # Initialize monitoring
        monitoring = EnhancedMonitoringSystem()

        # Register alert callback
        def alert_handler(alert):
            if isinstance(alert, AnomalyDetection):
                print(f"\n🚨 ANOMALY DETECTED:")
                print(f"   Type: {alert.anomaly_type.value}")
                print(f"   Severity: {alert.severity.value}")
                print(f"   Node: {alert.node_id}")
                print(f"   Score: {alert.score:.3f}")
            elif isinstance(alert, PredictiveAlert):
                print(f"\n⚠️  FAILURE PREDICTED:")
                print(f"   Type: {alert.failure_type}")
                print(f"   Time: {alert.predicted_failure_time}")
                print(f"   Confidence: {alert.confidence:.1%}")

        monitoring.register_alert_callback(alert_handler)

        # Simulate normal operations
        print("\n📊 Simulating normal operations...")
        for i in range(50):
            await monitoring.process_gradient_metrics(
                node_id=i % 5,
                gradient_norm=np.random.randn() * 10 + 100,
                validation_time=np.random.randn() * 50 + 200,
                pogq_score=np.random.random() * 0.3 + 0.6
            )

        # Simulate anomalies
        print("\n⚡ Simulating anomalies...")
        await monitoring.process_gradient_metrics(
            node_id=6,
            gradient_norm=1000,  # Abnormally large
            validation_time=2000,  # Very slow
            pogq_score=0.1  # Poor quality
        )

        # Update system metrics
        print("\n📈 Updating system metrics...")
        for i in range(30):
            await monitoring.update_system_metrics(
                validation_time=200 + i * 10,  # Increasing trend
                error_rate=0.01 + i * 0.005,   # Increasing errors
                memory_usage=0.5 + i * 0.01,   # Increasing memory
                cpu_usage=0.6
            )

        # Print statistics
        stats = monitoring.get_statistics()
        print(f"\n📊 Monitoring Statistics:")
        print(f"   Anomalies detected: {stats['anomalies_detected']}")
        print(f"   Failures predicted: {stats['failures_predicted']}")
        print(f"   Remediations executed: {stats['remediations_executed']}")

    asyncio.run(demo())

    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 60)