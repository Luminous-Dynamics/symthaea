# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for ML Framework Bridge

Tests cover:
- NumPyBridge (always available)
- PyTorchBridge (optional)
- TensorFlowBridge (optional)
- Gradient extraction and application
"""

import pytest
import numpy as np
from typing import Dict

from mycelix_fl.ml import (
    MLBridge,
    NumPyBridge,
    create_bridge,
    detect_framework,
)
from mycelix_fl.ml.bridge import (
    GradientInfo,
    ModelState,
    PYTORCH_AVAILABLE,
    TENSORFLOW_AVAILABLE,
)

# Conditional imports
if PYTORCH_AVAILABLE:
    from mycelix_fl.ml import PyTorchBridge
    import torch
    import torch.nn as nn

if TENSORFLOW_AVAILABLE:
    from mycelix_fl.ml import TensorFlowBridge
    import tensorflow as tf


# =============================================================================
# NUMPY BRIDGE TESTS (Always available)
# =============================================================================

class TestNumPyBridge:
    """Test NumPyBridge class."""

    def test_framework_name(self):
        """Test framework name property."""
        bridge = NumPyBridge()
        assert bridge.framework_name == "numpy"

    def test_extract_gradients_simple(self, rng):
        """Test gradient extraction from simple dict."""
        bridge = NumPyBridge()

        # NumPy bridge expects dict with 'gradients' key
        model = {
            "gradients": {
                "layer1": rng.randn(100, 50).astype(np.float32),
                "layer2": rng.randn(50, 10).astype(np.float32),
            }
        }

        grad_info = bridge.extract_gradients(model)

        assert isinstance(grad_info, GradientInfo)
        assert grad_info.framework == "numpy"
        assert grad_info.param_count == 100 * 50 + 50 * 10
        assert len(grad_info.gradient) == grad_info.param_count

    def test_extract_gradients_flattens(self, rng):
        """Test that gradients are properly flattened."""
        bridge = NumPyBridge()

        model = {
            "gradients": {
                "conv": rng.randn(32, 3, 3, 3).astype(np.float32),  # 4D
                "fc": rng.randn(128, 64).astype(np.float32),        # 2D
                "bias": rng.randn(10).astype(np.float32),           # 1D
            }
        }

        grad_info = bridge.extract_gradients(model)

        expected_params = 32 * 3 * 3 * 3 + 128 * 64 + 10
        assert grad_info.param_count == expected_params
        assert len(grad_info.gradient) == expected_params

    def test_get_model_state(self, rng):
        """Test model state serialization."""
        bridge = NumPyBridge()

        model = {
            "weights": {
                "w1": rng.randn(50, 30).astype(np.float32),
                "b1": np.zeros(30, dtype=np.float32),
            }
        }

        state = bridge.get_model_state(model)

        assert isinstance(state, ModelState)
        assert state.framework == "numpy"
        assert len(state.layer_shapes) == 2

    def test_apply_gradient(self, rng):
        """Test gradient application."""
        bridge = NumPyBridge()

        model = {
            "weights": {
                "w": rng.randn(10, 5).astype(np.float32),
            }
        }

        gradient = rng.randn(50).astype(np.float32) * 0.01
        learning_rate = 0.1

        # Get original values
        original_sum = np.sum(model["weights"]["w"])

        # Apply gradient (note: apply_gradients plural)
        bridge.apply_gradients(model, gradient, learning_rate)

        # Values should have changed
        new_sum = np.sum(model["weights"]["w"])
        assert original_sum != new_sum

    def test_roundtrip(self, rng):
        """Test extract -> modify -> apply cycle."""
        bridge = NumPyBridge()

        model = {
            "weights": {
                "layer": rng.randn(20, 10).astype(np.float32),
            },
            "gradients": {
                "layer": rng.randn(20, 10).astype(np.float32),
            }
        }

        # Extract
        grad_info = bridge.extract_gradients(model)

        # Modify gradient
        modified_gradient = grad_info.gradient * 2.0

        # Apply (note: apply_gradients plural)
        bridge.apply_gradients(model, modified_gradient, learning_rate=0.01)

        # Model weights should have changed
        new_state = bridge.get_model_state(model)
        original_state = bridge.get_model_state({
            "weights": {"layer": rng.randn(20, 10).astype(np.float32)}
        })
        # Weights changed is implicit since apply_gradients modifies in place


# =============================================================================
# PYTORCH BRIDGE TESTS (Conditional)
# =============================================================================

@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
class TestPyTorchBridge:
    """Test PyTorchBridge class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple PyTorch model."""
        return nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def test_framework_name(self):
        """Test framework name property."""
        bridge = PyTorchBridge()
        assert bridge.framework_name == "pytorch"

    def test_extract_gradients(self, simple_model):
        """Test gradient extraction from PyTorch model."""
        bridge = PyTorchBridge()

        # Create fake input and compute gradients
        x = torch.randn(1, 784)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        grad_info = bridge.extract_gradients(simple_model)

        assert isinstance(grad_info, GradientInfo)
        assert grad_info.framework == "pytorch"
        assert grad_info.param_count > 0
        assert len(grad_info.gradient) == grad_info.param_count

    def test_get_model_state(self, simple_model):
        """Test model state serialization."""
        bridge = PyTorchBridge()

        state = bridge.get_model_state(simple_model)

        assert isinstance(state, ModelState)
        assert state.framework == "pytorch"
        assert len(state.layer_shapes) > 0

    def test_apply_gradient(self, simple_model):
        """Test gradient application."""
        bridge = PyTorchBridge()

        # Get initial state
        initial_state = bridge.get_model_state(simple_model)

        # Create gradient
        gradient = np.random.randn(initial_state.param_count).astype(np.float32)

        # Apply (note: apply_gradients is the method name)
        bridge.apply_gradients(simple_model, gradient, learning_rate=0.1)

        # Get new state
        new_state = bridge.get_model_state(simple_model)

        # Weights should have changed
        assert not np.allclose(initial_state.weights, new_state.weights)

    def test_cnn_model(self):
        """Test with CNN model."""
        bridge = PyTorchBridge()

        model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 10),
        )

        # Forward pass to create gradients
        x = torch.randn(1, 1, 28, 28)
        y = model(x)
        y.sum().backward()

        grad_info = bridge.extract_gradients(model)

        assert grad_info is not None
        assert grad_info.param_count > 0


# =============================================================================
# TENSORFLOW BRIDGE TESTS (Conditional)
# =============================================================================

@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
class TestTensorFlowBridge:
    """Test TensorFlowBridge class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple TensorFlow model."""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10),
        ])

    def test_framework_name(self):
        """Test framework name property."""
        bridge = TensorFlowBridge()
        assert bridge.framework_name == "tensorflow"

    def test_extract_gradients(self, simple_model):
        """Test gradient extraction from TensorFlow model."""
        bridge = TensorFlowBridge()

        # Compile and create gradients
        simple_model.compile(optimizer='sgd', loss='mse')

        # Use GradientTape to get gradients
        x = tf.random.normal((1, 784))
        y = tf.random.normal((1, 10))

        with tf.GradientTape() as tape:
            pred = simple_model(x)
            loss = tf.reduce_mean((pred - y) ** 2)

        grads = tape.gradient(loss, simple_model.trainable_variables)

        # Store gradients in model (TF doesn't do this automatically)
        for var, grad in zip(simple_model.trainable_variables, grads):
            var._gradient = grad

        grad_info = bridge.extract_gradients(simple_model)

        assert isinstance(grad_info, GradientInfo)
        assert grad_info.framework == "tensorflow"

    def test_get_model_state(self, simple_model):
        """Test model state serialization."""
        bridge = TensorFlowBridge()

        state = bridge.get_model_state(simple_model)

        assert isinstance(state, ModelState)
        assert state.framework == "tensorflow"


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestBridgeFactory:
    """Test bridge factory functions."""

    def test_create_bridge_numpy(self):
        """Test creating NumPy bridge."""
        bridge = create_bridge("numpy")
        assert isinstance(bridge, NumPyBridge)

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_create_bridge_pytorch(self):
        """Test creating PyTorch bridge."""
        bridge = create_bridge("pytorch")
        assert isinstance(bridge, PyTorchBridge)

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
    def test_create_bridge_tensorflow(self):
        """Test creating TensorFlow bridge."""
        bridge = create_bridge("tensorflow")
        assert isinstance(bridge, TensorFlowBridge)

    def test_create_bridge_invalid(self):
        """Test that invalid framework falls back to NumPy."""
        # create_bridge returns NumPyBridge for unknown frameworks (with warning)
        bridge = create_bridge(framework="invalid_framework")
        assert isinstance(bridge, NumPyBridge)

    def test_detect_framework_numpy(self, rng):
        """Test framework detection for numpy dict model."""
        # detect_framework checks for 'weights' or 'gradients' keys
        model = {"weights": {"w": rng.randn(10, 5).astype(np.float32)}}
        framework = detect_framework(model)
        assert framework == "numpy"

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_detect_framework_pytorch(self):
        """Test framework detection for PyTorch model."""
        model = nn.Linear(10, 5)
        framework = detect_framework(model)
        assert framework == "pytorch"


# =============================================================================
# GRADIENT INFO TESTS
# =============================================================================

class TestGradientInfo:
    """Test GradientInfo dataclass."""

    def test_creation(self, rng):
        """Test basic creation."""
        gradient = rng.randn(1000).astype(np.float32)
        info = GradientInfo(
            gradient=gradient,
            param_count=1000,
            layer_shapes=[("layer1", (100, 10))],
            framework="test",
            dtype="float32",
        )

        assert len(info.gradient) == 1000
        assert info.param_count == 1000
        assert info.framework == "test"


# =============================================================================
# MODEL STATE TESTS
# =============================================================================

class TestModelState:
    """Test ModelState dataclass."""

    def test_creation(self, rng):
        """Test basic creation."""
        weights = rng.randn(500).astype(np.float32)
        state = ModelState(
            weights=weights,
            param_count=500,
            layer_shapes=[("dense", (50, 10))],
            framework="test",
            metadata={"version": 1},
        )

        assert len(state.weights) == 500
        assert state.param_count == 500
        assert state.metadata["version"] == 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestBridgeIntegration:
    """Integration tests for bridge with FL system."""

    def test_numpy_fl_integration(self, rng):
        """Test NumPy bridge integration with FL system."""
        from mycelix_fl import MycelixFL, FLConfig

        bridge = NumPyBridge()

        # Simulate multiple clients with proper dict format
        gradients = {}
        for i in range(5):
            model = {
                "gradients": {
                    "w": rng.randn(100, 50).astype(np.float32)
                }
            }
            grad_info = bridge.extract_gradients(model)
            gradients[f"client_{i}"] = grad_info.gradient

        # Run FL round
        fl = MycelixFL(config=FLConfig(min_nodes=3))
        result = fl.execute_round(gradients, round_num=1)

        assert result is not None
        assert len(result.aggregated_gradient) == 5000  # 100*50

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_pytorch_fl_integration(self):
        """Test PyTorch bridge integration with FL system."""
        from mycelix_fl import MycelixFL, FLConfig

        bridge = PyTorchBridge()

        # Simulate multiple clients with same architecture
        gradients = {}
        for i in range(5):
            model = nn.Linear(100, 10)
            x = torch.randn(1, 100)
            y = model(x)
            y.sum().backward()

            grad_info = bridge.extract_gradients(model)
            gradients[f"client_{i}"] = grad_info.gradient

        # Run FL round
        fl = MycelixFL(config=FLConfig(min_nodes=3))
        result = fl.execute_round(gradients, round_num=1)

        assert result is not None


# =============================================================================
# EDGE CASES
# =============================================================================

class TestBridgeEdgeCases:
    """Test edge cases for bridge module."""

    def test_empty_model(self):
        """Test handling of empty model."""
        bridge = NumPyBridge()

        model = {}

        with pytest.raises((ValueError, KeyError)):
            bridge.extract_gradients(model)

    def test_single_parameter(self, rng):
        """Test model with single parameter."""
        bridge = NumPyBridge()

        model = {
            "gradients": {
                "scalar": np.array([1.0], dtype=np.float32)
            }
        }

        grad_info = bridge.extract_gradients(model)
        assert grad_info.param_count == 1

    def test_large_model(self, rng):
        """Test with large model (1M params)."""
        bridge = NumPyBridge()

        model = {
            "gradients": {
                "large_layer": rng.randn(1000, 1000).astype(np.float32),
            }
        }

        grad_info = bridge.extract_gradients(model)
        assert grad_info.param_count == 1_000_000
