# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ML Framework Bridge - Unified Interface for Federated Learning

Provides consistent API across ML frameworks:
- Gradient extraction (after backward pass)
- Model state serialization
- Aggregated gradient application
- Training callbacks for FL integration

The bridge pattern allows the FL system to work with any framework
without framework-specific code in the core logic.

Author: Luminous Dynamics
Date: December 30, 2025
"""

from __future__ import annotations  # Enable forward references

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

# TYPE_CHECKING block for type hints only
if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import tensorflow as tf

# Optional framework imports
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

logger = logging.getLogger(__name__)


@dataclass
class GradientInfo:
    """Information about extracted gradient."""
    gradient: np.ndarray  # Flattened gradient
    param_count: int      # Number of parameters
    layer_shapes: List[Tuple[str, Tuple[int, ...]]]  # Layer name -> shape
    framework: str        # Source framework
    dtype: str           # Original dtype


@dataclass
class ModelState:
    """Serialized model state."""
    weights: np.ndarray   # Flattened weights
    param_count: int
    layer_shapes: List[Tuple[str, Tuple[int, ...]]]
    framework: str
    metadata: Dict[str, Any]


class MLBridge(ABC):
    """
    Abstract base class for ML framework bridges.

    Provides unified interface for:
    - Gradient extraction
    - Model state management
    - Aggregation application
    """

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Return framework name."""
        pass

    @abstractmethod
    def extract_gradients(self, model: Any) -> GradientInfo:
        """
        Extract gradients from model after backward pass.

        Args:
            model: Framework-specific model

        Returns:
            GradientInfo with flattened gradient
        """
        pass

    @abstractmethod
    def apply_gradients(
        self,
        model: Any,
        gradient: np.ndarray,
        learning_rate: float = 1.0,
    ) -> None:
        """
        Apply aggregated gradient to model.

        Args:
            model: Framework-specific model
            gradient: Flattened gradient array
            learning_rate: Scaling factor for update
        """
        pass

    @abstractmethod
    def get_model_state(self, model: Any) -> ModelState:
        """
        Get current model weights.

        Args:
            model: Framework-specific model

        Returns:
            ModelState with flattened weights
        """
        pass

    @abstractmethod
    def set_model_state(self, model: Any, state: ModelState) -> None:
        """
        Set model weights from state.

        Args:
            model: Framework-specific model
            state: ModelState to apply
        """
        pass

    def compute_gradient_norm(self, gradient: np.ndarray) -> float:
        """Compute L2 norm of gradient."""
        return float(np.linalg.norm(gradient))

    def clip_gradients(
        self,
        gradient: np.ndarray,
        max_norm: float,
    ) -> np.ndarray:
        """
        Clip gradient to max norm.

        Args:
            gradient: Gradient array
            max_norm: Maximum L2 norm

        Returns:
            Clipped gradient
        """
        norm = np.linalg.norm(gradient)
        if norm > max_norm:
            gradient = gradient * (max_norm / norm)
        return gradient


class PyTorchBridge(MLBridge):
    """
    PyTorch framework bridge.

    Handles:
    - torch.nn.Module gradient extraction
    - State dict management
    - Gradient application
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize PyTorch bridge.

        Args:
            device: Device for operations (cpu, cuda, etc.)
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"✅ PyTorchBridge initialized (device={self.device})")

    @property
    def framework_name(self) -> str:
        return "pytorch"

    def extract_gradients(self, model: "nn.Module") -> GradientInfo:
        """
        Extract gradients from PyTorch model.

        Args:
            model: PyTorch nn.Module with gradients

        Returns:
            GradientInfo with flattened gradient
        """
        gradients = []
        layer_shapes = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu().numpy()
                gradients.append(grad.flatten())
                layer_shapes.append((name, param.shape))
            else:
                # Include zero gradient for params without grad
                gradients.append(np.zeros(param.numel(), dtype=np.float32))
                layer_shapes.append((name, param.shape))

        if not gradients:
            raise ValueError("Model has no gradients. Run backward() first.")

        gradient = np.concatenate(gradients).astype(np.float32)

        return GradientInfo(
            gradient=gradient,
            param_count=len(gradient),
            layer_shapes=layer_shapes,
            framework="pytorch",
            dtype="float32",
        )

    def apply_gradients(
        self,
        model: "nn.Module",
        gradient: np.ndarray,
        learning_rate: float = 1.0,
    ) -> None:
        """
        Apply aggregated gradient to PyTorch model.

        Args:
            model: PyTorch model
            gradient: Flattened gradient
            learning_rate: Update scaling factor
        """
        offset = 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                numel = param.numel()
                grad_chunk = gradient[offset:offset + numel]

                # Reshape and apply
                grad_tensor = torch.from_numpy(
                    grad_chunk.reshape(param.shape)
                ).to(param.device)

                param.data -= learning_rate * grad_tensor
                offset += numel

    def get_model_state(self, model: "nn.Module") -> ModelState:
        """
        Get PyTorch model state.

        Args:
            model: PyTorch model

        Returns:
            ModelState with weights
        """
        weights = []
        layer_shapes = []

        for name, param in model.named_parameters():
            w = param.detach().cpu().numpy()
            weights.append(w.flatten())
            layer_shapes.append((name, param.shape))

        weights = np.concatenate(weights).astype(np.float32)

        return ModelState(
            weights=weights,
            param_count=len(weights),
            layer_shapes=layer_shapes,
            framework="pytorch",
            metadata={"device": str(next(model.parameters()).device)},
        )

    def set_model_state(self, model: "nn.Module", state: ModelState) -> None:
        """
        Set PyTorch model state.

        Args:
            model: PyTorch model
            state: ModelState to apply
        """
        offset = 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                numel = param.numel()
                weights_chunk = state.weights[offset:offset + numel]

                param.data.copy_(
                    torch.from_numpy(
                        weights_chunk.reshape(param.shape)
                    ).to(param.device)
                )
                offset += numel

    def create_fl_callback(
        self,
        fl_round_callback: Callable[[GradientInfo, int], np.ndarray],
    ) -> Callable:
        """
        Create training callback for FL integration.

        Args:
            fl_round_callback: Function(gradient_info, round) -> aggregated_gradient

        Returns:
            Callback function to use in training loop
        """
        def callback(model: "nn.Module", epoch: int, **kwargs):
            gradient_info = self.extract_gradients(model)
            aggregated = fl_round_callback(gradient_info, epoch)
            self.apply_gradients(model, aggregated)
            return aggregated

        return callback


class TensorFlowBridge(MLBridge):
    """
    TensorFlow/Keras framework bridge.

    Handles:
    - tf.keras.Model gradient extraction
    - Weight management
    - Gradient tape integration
    """

    def __init__(self):
        """Initialize TensorFlow bridge."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")

        logger.info("✅ TensorFlowBridge initialized")

    @property
    def framework_name(self) -> str:
        return "tensorflow"

    def extract_gradients(self, model: Any) -> GradientInfo:
        """
        Extract gradients from TensorFlow model.

        Note: Requires gradients to be computed via GradientTape
        and stored in model._stored_gradients.

        Args:
            model: TensorFlow model with stored gradients

        Returns:
            GradientInfo
        """
        # Check for stored gradients (set by compute_gradients)
        if not hasattr(model, '_stored_gradients') or model._stored_gradients is None:
            raise ValueError(
                "Model has no gradients. Use compute_gradients() first."
            )

        gradients = model._stored_gradients
        layer_shapes = []
        flat_grads = []

        for var, grad in zip(model.trainable_variables, gradients):
            if grad is not None:
                flat_grads.append(grad.numpy().flatten())
                layer_shapes.append((var.name, tuple(var.shape.as_list())))
            else:
                flat_grads.append(np.zeros(np.prod(var.shape), dtype=np.float32))
                layer_shapes.append((var.name, tuple(var.shape.as_list())))

        gradient = np.concatenate(flat_grads).astype(np.float32)

        return GradientInfo(
            gradient=gradient,
            param_count=len(gradient),
            layer_shapes=layer_shapes,
            framework="tensorflow",
            dtype="float32",
        )

    def compute_gradients(
        self,
        model: Any,
        x: Any,
        y: Any,
        loss_fn: Optional[Callable] = None,
    ) -> GradientInfo:
        """
        Compute gradients using GradientTape.

        Args:
            model: TensorFlow model
            x: Input data
            y: Target data
            loss_fn: Loss function (uses MSE if None)

        Returns:
            GradientInfo
        """
        if loss_fn is None:
            loss_fn = tf.keras.losses.MeanSquaredError()

        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)

        # Store for extract_gradients
        model._stored_gradients = gradients

        return self.extract_gradients(model)

    def apply_gradients(
        self,
        model: Any,
        gradient: np.ndarray,
        learning_rate: float = 1.0,
    ) -> None:
        """
        Apply aggregated gradient to TensorFlow model.

        Args:
            model: TensorFlow model
            gradient: Flattened gradient
            learning_rate: Update scaling factor
        """
        offset = 0

        for var in model.trainable_variables:
            shape = var.shape.as_list()
            numel = int(np.prod(shape))
            grad_chunk = gradient[offset:offset + numel]

            var.assign_sub(
                learning_rate * grad_chunk.reshape(shape)
            )
            offset += numel

    def get_model_state(self, model: Any) -> ModelState:
        """Get TensorFlow model state."""
        weights = []
        layer_shapes = []

        for var in model.trainable_variables:
            w = var.numpy()
            weights.append(w.flatten())
            layer_shapes.append((var.name, tuple(var.shape.as_list())))

        weights = np.concatenate(weights).astype(np.float32)

        return ModelState(
            weights=weights,
            param_count=len(weights),
            layer_shapes=layer_shapes,
            framework="tensorflow",
            metadata={},
        )

    def set_model_state(self, model: Any, state: ModelState) -> None:
        """Set TensorFlow model state."""
        offset = 0

        for var in model.trainable_variables:
            shape = var.shape.as_list()
            numel = int(np.prod(shape))
            weights_chunk = state.weights[offset:offset + numel]

            var.assign(weights_chunk.reshape(shape))
            offset += numel


class NumPyBridge(MLBridge):
    """
    NumPy bridge for framework-agnostic operations.

    Useful for:
    - Testing and simulation
    - Simple models
    - Interoperability
    """

    def __init__(self):
        """Initialize NumPy bridge."""
        logger.info("✅ NumPyBridge initialized")

    @property
    def framework_name(self) -> str:
        return "numpy"

    def extract_gradients(self, model: Dict[str, np.ndarray]) -> GradientInfo:
        """
        Extract gradients from dict-based model.

        Model should have 'gradients' key with layer gradients.

        Args:
            model: Dict with 'gradients' key

        Returns:
            GradientInfo
        """
        if 'gradients' not in model:
            raise ValueError("Model dict must have 'gradients' key")

        gradients = model['gradients']
        flat_grads = []
        layer_shapes = []

        for name, grad in gradients.items():
            flat_grads.append(grad.flatten())
            layer_shapes.append((name, grad.shape))

        gradient = np.concatenate(flat_grads).astype(np.float32)

        return GradientInfo(
            gradient=gradient,
            param_count=len(gradient),
            layer_shapes=layer_shapes,
            framework="numpy",
            dtype="float32",
        )

    def apply_gradients(
        self,
        model: Dict[str, np.ndarray],
        gradient: np.ndarray,
        learning_rate: float = 1.0,
    ) -> None:
        """
        Apply gradient to dict-based model.

        Args:
            model: Dict with 'weights' key
            gradient: Flattened gradient
            learning_rate: Update scaling
        """
        if 'weights' not in model:
            raise ValueError("Model dict must have 'weights' key")

        offset = 0

        for name, weight in model['weights'].items():
            numel = weight.size
            grad_chunk = gradient[offset:offset + numel]

            model['weights'][name] = weight - learning_rate * grad_chunk.reshape(weight.shape)
            offset += numel

    def get_model_state(self, model: Dict[str, np.ndarray]) -> ModelState:
        """Get NumPy model state."""
        if 'weights' not in model:
            raise ValueError("Model dict must have 'weights' key")

        weights = []
        layer_shapes = []

        for name, w in model['weights'].items():
            weights.append(w.flatten())
            layer_shapes.append((name, w.shape))

        weights = np.concatenate(weights).astype(np.float32)

        return ModelState(
            weights=weights,
            param_count=len(weights),
            layer_shapes=layer_shapes,
            framework="numpy",
            metadata={},
        )

    def set_model_state(self, model: Dict[str, np.ndarray], state: ModelState) -> None:
        """Set NumPy model state."""
        if 'weights' not in model:
            model['weights'] = {}

        offset = 0

        for name, shape in state.layer_shapes:
            numel = int(np.prod(shape))
            weights_chunk = state.weights[offset:offset + numel]

            model['weights'][name] = weights_chunk.reshape(shape)
            offset += numel


def detect_framework(model: Any) -> str:
    """
    Detect ML framework from model type.

    Args:
        model: Model object

    Returns:
        Framework name: 'pytorch', 'tensorflow', 'numpy', or 'unknown'
    """
    # PyTorch
    if PYTORCH_AVAILABLE and isinstance(model, nn.Module):
        return "pytorch"

    # TensorFlow
    if TENSORFLOW_AVAILABLE:
        if isinstance(model, tf.keras.Model):
            return "tensorflow"
        if hasattr(model, 'trainable_variables'):
            return "tensorflow"

    # NumPy dict
    if isinstance(model, dict):
        if 'weights' in model or 'gradients' in model:
            return "numpy"

    return "unknown"


def create_bridge(
    model: Optional[Any] = None,
    framework: Optional[str] = None,
    **kwargs,
) -> MLBridge:
    """
    Create appropriate ML bridge for model or framework.

    Args:
        model: Optional model to detect framework
        framework: Explicit framework name
        **kwargs: Additional arguments for bridge

    Returns:
        MLBridge instance

    Example:
        >>> bridge = create_bridge(model=my_pytorch_model)
        >>> gradient_info = bridge.extract_gradients(model)
    """
    if framework is None and model is not None:
        framework = detect_framework(model)

    if framework == "pytorch":
        return PyTorchBridge(**kwargs)
    elif framework == "tensorflow":
        return TensorFlowBridge(**kwargs)
    elif framework == "numpy":
        return NumPyBridge(**kwargs)
    else:
        # Default to NumPy for maximum compatibility
        logger.warning(f"Unknown framework '{framework}', defaulting to NumPy")
        return NumPyBridge(**kwargs)


# Utility functions
def aggregate_gradients(
    gradients: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Simple weighted average of gradients.

    Args:
        gradients: List of gradient arrays
        weights: Optional weights (uniform if None)

    Returns:
        Aggregated gradient
    """
    if not gradients:
        raise ValueError("No gradients to aggregate")

    if weights is None:
        weights = [1.0 / len(gradients)] * len(gradients)

    # Ensure weights sum to 1
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Weighted sum
    aggregated = np.zeros_like(gradients[0])
    for grad, weight in zip(gradients, weights):
        aggregated += weight * grad

    return aggregated


if __name__ == "__main__":
    print("🧪 Testing ML Bridge...")

    # Test NumPy bridge
    print("\n--- NumPy Bridge ---")
    np_bridge = NumPyBridge()

    np_model = {
        'weights': {
            'layer1': np.random.randn(10, 5).astype(np.float32),
            'layer2': np.random.randn(5, 2).astype(np.float32),
        },
        'gradients': {
            'layer1': np.random.randn(10, 5).astype(np.float32),
            'layer2': np.random.randn(5, 2).astype(np.float32),
        },
    }

    grad_info = np_bridge.extract_gradients(np_model)
    print(f"   Extracted gradient: {grad_info.param_count} params")
    print(f"   Layers: {[s[0] for s in grad_info.layer_shapes]}")

    state = np_bridge.get_model_state(np_model)
    print(f"   Model state: {state.param_count} params")

    # Test PyTorch bridge if available
    if PYTORCH_AVAILABLE:
        print("\n--- PyTorch Bridge ---")
        pt_bridge = PyTorchBridge()

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 2)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = SimpleModel()
        x = torch.randn(4, 10)
        y = torch.randn(4, 2)

        # Compute gradients
        output = model(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()

        grad_info = pt_bridge.extract_gradients(model)
        print(f"   Extracted gradient: {grad_info.param_count} params")
        print(f"   Gradient norm: {np.linalg.norm(grad_info.gradient):.4f}")

        state = pt_bridge.get_model_state(model)
        print(f"   Model state: {state.param_count} params")

    print("\n✅ ML Bridge tests passed!")
