# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Tests for v1.2.0 new modules: config, exceptions, validation, observability."""
import pytest
import numpy as np


class TestConfig:
    """Test configuration module."""
    
    def test_config_imports(self):
        """Test all config exports are available."""
        from mycelix_fl.config import (
            HV_DIMENSION_DEFAULT,
            HV_DIMENSION_HIGH,
            BYZANTINE_THRESHOLD_MAX,
            BYZANTINE_THRESHOLD_DEFAULT,
            DETECTION_LATENCY_TARGET_MS,
            EPSILON,
            RuntimeConfig,
            runtime_config,
            validate_dimension,
            get_nearest_valid_dimension,
        )
        assert HV_DIMENSION_DEFAULT == 2048
        assert BYZANTINE_THRESHOLD_MAX == 0.45
        assert EPSILON == 1e-8
    
    def test_validate_dimension(self):
        """Test dimension validation."""
        from mycelix_fl.config import validate_dimension
        assert validate_dimension(2048) is True
        assert validate_dimension(1024) is True
        assert validate_dimension(1000) is False
        assert validate_dimension(0) is False
    
    def test_get_nearest_valid_dimension(self):
        """Test nearest dimension helper."""
        from mycelix_fl.config import get_nearest_valid_dimension
        assert get_nearest_valid_dimension(1000) == 1024
        assert get_nearest_valid_dimension(2000) == 2048
        assert get_nearest_valid_dimension(3000) == 2048  # Closer to 2048 than 4096
    
    def test_runtime_config(self):
        """Test runtime config is a singleton."""
        from mycelix_fl.config import runtime_config, RuntimeConfig
        assert isinstance(runtime_config, RuntimeConfig)
        assert runtime_config.hv_dimension == 2048
    
    def test_runtime_config_methods(self):
        """Test runtime config methods."""
        from mycelix_fl.config import RuntimeConfig
        config = RuntimeConfig()
        
        config.set_high_security()
        assert config.byzantine_threshold == 0.33
        
        config.set_high_performance()
        assert config.hv_dimension == 1024
        
        config.set_byzantine_threshold(0.40)
        assert config.byzantine_threshold == 0.40
        
        with pytest.raises(ValueError):
            config.set_byzantine_threshold(0.6)  # > 0.5


class TestExceptions:
    """Test exception module."""
    
    def test_base_exception(self):
        """Test base exception formatting."""
        from mycelix_fl.exceptions import MycelixFLError
        
        err = MycelixFLError(
            "Test error",
            context={"key": "value"},
            suggestion="Fix this",
        )
        assert "Test error" in str(err)
        assert "key=value" in str(err)
        assert "Fix this" in str(err)
    
    def test_configuration_error(self):
        """Test configuration error."""
        from mycelix_fl.exceptions import ConfigurationError
        
        err = ConfigurationError(
            "Bad config",
            param_name="dimension",
            param_value=1000,
            valid_range="powers of 2",
        )
        assert "Bad config" in str(err)
        assert "dimension" in str(err)
    
    def test_invalid_gradient_error(self):
        """Test gradient error."""
        from mycelix_fl.exceptions import InvalidGradientError
        
        err = InvalidGradientError(
            "Contains NaN",
            node_id="node_1",
            gradient_shape=(10000,),
            issue="contains_nan",
        )
        assert "NaN" in str(err)
        assert "node_1" in str(err)
    
    def test_insufficient_gradients_error(self):
        """Test insufficient gradients."""
        from mycelix_fl.exceptions import InsufficientGradientsError
        
        err = InsufficientGradientsError(received=2, minimum=3)
        assert "Insufficient" in str(err)
        assert "2" in str(err)
        assert "3" in str(err)
    
    def test_too_many_byzantine_error(self):
        """Test too many Byzantine error."""
        from mycelix_fl.exceptions import TooManyByzantineError
        
        err = TooManyByzantineError(
            detected_ratio=0.5,
            threshold=0.45,
            byzantine_nodes={"node_1", "node_2"},
        )
        assert "50.0%" in str(err)
        assert "45.0%" in str(err)
        assert err.byzantine_nodes == {"node_1", "node_2"}
    
    def test_exception_hierarchy(self):
        """Test exception inheritance."""
        from mycelix_fl.exceptions import (
            MycelixFLError,
            ConfigurationError,
            GradientError,
            InvalidGradientError,
            ByzantineDetectionError,
            CompressionError,
        )
        
        # All inherit from base
        assert issubclass(ConfigurationError, MycelixFLError)
        assert issubclass(GradientError, MycelixFLError)
        assert issubclass(InvalidGradientError, GradientError)
        assert issubclass(ByzantineDetectionError, MycelixFLError)
        assert issubclass(CompressionError, MycelixFLError)


class TestValidation:
    """Test validation module."""
    
    def test_validate_gradient_valid(self):
        """Test valid gradient passes."""
        from mycelix_fl.validation import validate_gradient
        
        grad = np.random.randn(10000).astype(np.float32)
        result = validate_gradient(grad)
        assert result is grad
    
    def test_validate_gradient_nan(self):
        """Test NaN detection."""
        from mycelix_fl.validation import validate_gradient
        from mycelix_fl.exceptions import InvalidGradientError
        
        grad = np.array([1.0, np.nan, 2.0])
        with pytest.raises(InvalidGradientError) as exc_info:
            validate_gradient(grad)
        assert "NaN" in str(exc_info.value)
    
    def test_validate_gradient_inf(self):
        """Test Inf detection."""
        from mycelix_fl.validation import validate_gradient
        from mycelix_fl.exceptions import InvalidGradientError
        
        grad = np.array([1.0, np.inf, 2.0])
        with pytest.raises(InvalidGradientError) as exc_info:
            validate_gradient(grad)
        assert "Inf" in str(exc_info.value)
    
    def test_validate_gradients_dict(self):
        """Test validating multiple gradients."""
        from mycelix_fl.validation import validate_gradients
        
        grads = {
            "node_1": np.random.randn(1000).astype(np.float32),
            "node_2": np.random.randn(1000).astype(np.float32),
            "node_3": np.random.randn(1000).astype(np.float32),
        }
        result = validate_gradients(grads, min_nodes=3)
        assert result == grads
    
    def test_validate_gradients_insufficient(self):
        """Test minimum node check."""
        from mycelix_fl.validation import validate_gradients
        from mycelix_fl.exceptions import InsufficientGradientsError
        
        grads = {
            "node_1": np.random.randn(1000).astype(np.float32),
        }
        with pytest.raises(InsufficientGradientsError):
            validate_gradients(grads, min_nodes=3)
    
    def test_sanitize_gradient(self):
        """Test gradient sanitization."""
        from mycelix_fl.validation import sanitize_gradient
        
        grad = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
        result = sanitize_gradient(grad, replace_nan=0.0, replace_inf=100.0)
        
        assert result[0] == 1.0
        assert result[1] == 0.0  # NaN replaced
        assert result[2] == 100.0  # +Inf replaced
        assert result[3] == -100.0  # -Inf replaced
        assert result[4] == 2.0
    
    def test_compute_gradient_stats(self):
        """Test gradient statistics."""
        from mycelix_fl.validation import compute_gradient_stats
        
        grad = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_gradient_stats(grad)
        
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["nan_ratio"] == 0.0
        assert stats["inf_ratio"] == 0.0
    
    def test_validate_byzantine_threshold(self):
        """Test Byzantine threshold validation."""
        from mycelix_fl.validation import validate_byzantine_threshold
        from mycelix_fl.exceptions import ConfigurationError
        
        assert validate_byzantine_threshold(0.45) == 0.45
        assert validate_byzantine_threshold(0.33) == 0.33
        
        with pytest.raises(ConfigurationError):
            validate_byzantine_threshold(0.6)  # > 0.45
        
        with pytest.raises(ConfigurationError):
            validate_byzantine_threshold(-0.1)  # < 0
    
    def test_gradient_validator_class(self):
        """Test stateful gradient validator."""
        from mycelix_fl.validation import GradientValidator
        
        validator = GradientValidator(min_nodes=2)
        
        batch1 = {
            "node_1": np.random.randn(100).astype(np.float32),
            "node_2": np.random.randn(100).astype(np.float32),
        }
        result = validator.validate_batch(batch1)
        
        assert len(result) == 2
        assert validator.batches_validated == 1
        assert validator.gradients_validated == 2
        assert validator.gradients_rejected == 0


class TestObservability:
    """Test observability module."""
    
    def test_counter(self):
        """Test Counter metric."""
        from mycelix_fl.observability import Counter
        
        counter = Counter("test_counter", "Test counter")
        assert counter.get() == 0.0
        
        counter.inc()
        assert counter.get() == 1.0
        
        counter.inc(5)
        assert counter.get() == 6.0
        
        # Can't decrement
        with pytest.raises(ValueError):
            counter.inc(-1)
    
    def test_counter_with_labels(self):
        """Test Counter with labels."""
        from mycelix_fl.observability import Counter
        
        counter = Counter("requests", "Total requests")
        counter.inc(1, labels={"status": "success"})
        counter.inc(2, labels={"status": "error"})
        counter.inc(1, labels={"status": "success"})
        
        assert counter.get(labels={"status": "success"}) == 2.0
        assert counter.get(labels={"status": "error"}) == 2.0
    
    def test_gauge(self):
        """Test Gauge metric."""
        from mycelix_fl.observability import Gauge
        
        gauge = Gauge("active_nodes", "Active nodes")
        gauge.set(10)
        assert gauge.get() == 10.0
        
        gauge.inc()
        assert gauge.get() == 11.0
        
        gauge.dec(5)
        assert gauge.get() == 6.0
    
    def test_histogram(self):
        """Test Histogram metric."""
        from mycelix_fl.observability import Histogram
        
        hist = Histogram("latency_ms", "Latency")
        
        for val in [10, 20, 30, 40, 50, 100, 200]:
            hist.observe(val)
        
        assert hist.count == 7
        assert hist.mean == pytest.approx(64.28, rel=0.1)
        assert hist.percentile(0.5) == 40  # Median
    
    def test_metrics_registry(self):
        """Test MetricsRegistry."""
        from mycelix_fl.observability import MetricsRegistry
        
        registry = MetricsRegistry(prefix="test")
        
        registry.rounds_total.inc()
        registry.byzantine_detected.inc(3)
        registry.active_nodes.set(10)
        registry.round_latency_ms.observe(50.0)
        
        # Export JSON
        data = registry.export_json()
        assert data["counters"]["rounds_total"] == 1.0
        assert data["counters"]["byzantine_detected"] == 3.0
        assert data["gauges"]["active_nodes"] == 10.0
        
        # Export Prometheus
        prometheus = registry.export_prometheus()
        assert "test_rounds_total" in prometheus
        assert "test_byzantine_detected_total" in prometheus
    
    def test_timed_decorator(self):
        """Test timing decorator."""
        import time
        from mycelix_fl.observability import timed, Histogram
        
        hist = Histogram("test_latency", "Test")
        
        @timed(hist)
        def slow_function():
            time.sleep(0.01)
            return "done"
        
        result = slow_function()
        assert result == "done"
        assert hist.count == 1
        assert hist.mean >= 10  # At least 10ms
    
    def test_timed_block(self):
        """Test timing context manager."""
        import time
        from mycelix_fl.observability import timed_block, Histogram
        
        hist = Histogram("block_latency", "Block")
        
        with timed_block("test_block", hist):
            time.sleep(0.01)
        
        assert hist.count == 1
        assert hist.mean >= 10  # At least 10ms
    
    def test_structured_logger(self):
        """Test structured logger."""
        import json
        from mycelix_fl.observability import StructuredLogger
        
        log = StructuredLogger("test")
        
        # Test context manager
        with log.context(round_num=5, node_count=10):
            # Verify context is set
            formatted = log._format("test message", extra="value")
            data = json.loads(formatted)
            
            assert data["message"] == "test message"
            assert data["round_num"] == 5
            assert data["node_count"] == 10
            assert data["extra"] == "value"
    
    def test_performance_tracker(self):
        """Test performance tracker."""
        from mycelix_fl.observability import PerformanceTracker
        
        tracker = PerformanceTracker()
        
        for i in range(20):
            tracker.record(
                round_num=i,
                total_nodes=10,
                byzantine_nodes=2,
                healed_nodes=1,
                detection_latency_ms=50.0 + i,
                round_latency_ms=100.0 + i * 2,
                system_phi=0.5,
            )
        
        summary = tracker.summary()
        assert summary["rounds"] == 20
        assert summary["first_round"] == 0
        assert summary["last_round"] == 19
        assert "detection_latency_ms" in summary
        assert "sla_compliance" in summary
        
        health = tracker.check_health()
        assert health["status"] in ["healthy", "degraded"]
        assert health["rounds_analyzed"] == 10  # Uses last 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
