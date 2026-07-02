# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Pytest Configuration and Shared Fixtures (0TML Tests)

Provides reusable test fixtures for:
- Gradient generation (honest and Byzantine)
- Model state simulation
- FL configuration presets
- Performance timing utilities

NOTE: This conftest.py inherits from the root conftest.py which provides:
- Service availability detection (ServiceAvailability class)
- Automatic test skipping based on service availability
- Skip helper decorators (skip_if_no_pytorch, etc.)

For integration tests, use markers like:
    @pytest.mark.integration
    @pytest.mark.holochain
    def test_something():
        ...
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple
import time
from contextlib import contextmanager

# Import service availability from root conftest (available via pytest's conftest chain)
# These are re-exported for convenience
try:
    from conftest import (
        ServiceAvailability,
        skip_if_no_pytorch,
        skip_if_no_tensorflow,
        skip_if_no_holochain,
        skip_if_no_rust,
        skip_if_no_redis,
        skip_if_no_ethereum,
    )
except ImportError:
    # Fallback if running tests from this directory directly
    pass


# =============================================================================
# GRADIENT FIXTURES
# =============================================================================

@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.RandomState(42)


@pytest.fixture
def small_gradient(rng) -> np.ndarray:
    """Small gradient for quick tests (1K params)."""
    return rng.randn(1000).astype(np.float32)


@pytest.fixture
def medium_gradient(rng) -> np.ndarray:
    """Medium gradient for realistic tests (100K params)."""
    return rng.randn(100_000).astype(np.float32)


@pytest.fixture
def large_gradient(rng) -> np.ndarray:
    """Large gradient for stress tests (1M params)."""
    return rng.randn(1_000_000).astype(np.float32)


@pytest.fixture
def honest_gradients(rng) -> Dict[str, np.ndarray]:
    """
    Generate 10 honest gradients with natural variation.

    Honest gradients cluster around a true gradient direction
    with small Gaussian perturbations.
    """
    true_direction = rng.randn(10_000).astype(np.float32)
    true_direction /= np.linalg.norm(true_direction)

    gradients = {}
    for i in range(10):
        noise = rng.randn(10_000).astype(np.float32) * 0.1
        grad = true_direction + noise
        gradients[f"honest_{i}"] = grad

    return gradients


@pytest.fixture
def byzantine_gradients(rng) -> Dict[str, np.ndarray]:
    """
    Generate gradients with 30% Byzantine attackers.

    Attack types included:
    - Sign flip (negates gradient)
    - Random noise (completely random)
    - Scaling attack (extreme magnitude)
    """
    true_direction = rng.randn(10_000).astype(np.float32)
    true_direction /= np.linalg.norm(true_direction)

    gradients = {}

    # 7 honest nodes
    for i in range(7):
        noise = rng.randn(10_000).astype(np.float32) * 0.1
        gradients[f"honest_{i}"] = true_direction + noise

    # 3 Byzantine nodes (different attack types)
    # Sign flip attack
    gradients["byzantine_flip"] = -true_direction * 1.5

    # Random noise attack
    gradients["byzantine_random"] = rng.randn(10_000).astype(np.float32) * 5.0

    # Scaling attack (extreme values)
    gradients["byzantine_scale"] = true_direction * 100.0

    return gradients


@pytest.fixture
def mixed_gradients_45_percent(rng) -> Dict[str, np.ndarray]:
    """
    Generate gradients with 45% Byzantine nodes.

    This tests the system at its theoretical limit.
    """
    true_direction = rng.randn(10_000).astype(np.float32)
    true_direction /= np.linalg.norm(true_direction)

    gradients = {}

    # 11 honest nodes (55%)
    for i in range(11):
        noise = rng.randn(10_000).astype(np.float32) * 0.1
        gradients[f"honest_{i}"] = true_direction + noise

    # 9 Byzantine nodes (45%) - mixed attacks
    for i in range(3):
        gradients[f"byz_flip_{i}"] = -true_direction * (1.0 + rng.rand())
    for i in range(3):
        gradients[f"byz_random_{i}"] = rng.randn(10_000).astype(np.float32) * 3.0
    for i in range(3):
        gradients[f"byz_scale_{i}"] = true_direction * (50 + rng.rand() * 50)

    return gradients


# =============================================================================
# FL CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Default FL configuration for testing."""
    from mycelix_fl import FLConfig
    return FLConfig(
        num_rounds=10,
        min_nodes=3,
        byzantine_threshold=0.45,
        learning_rate=0.01,
    )


@pytest.fixture
def strict_config():
    """Strict configuration for high-security scenarios."""
    from mycelix_fl import FLConfig
    return FLConfig(
        num_rounds=100,
        min_nodes=10,
        byzantine_threshold=0.33,
        learning_rate=0.001,
        use_detection=True,
        use_healing=True,
    )


@pytest.fixture
def fast_config():
    """Fast configuration for quick iteration."""
    from mycelix_fl import FLConfig
    return FLConfig(
        num_rounds=5,
        min_nodes=3,
        byzantine_threshold=0.45,
        learning_rate=0.1,
        use_detection=False,
        use_healing=False,
    )


# =============================================================================
# TIMING UTILITIES
# =============================================================================

@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations."""
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  {name}: {elapsed:.2f}ms")


class TimingStats:
    """Collect timing statistics across multiple runs."""

    def __init__(self):
        self.times: List[float] = []

    def record(self, elapsed_ms: float):
        self.times.append(elapsed_ms)

    @property
    def mean(self) -> float:
        return np.mean(self.times) if self.times else 0.0

    @property
    def std(self) -> float:
        return np.std(self.times) if len(self.times) > 1 else 0.0

    @property
    def min(self) -> float:
        return min(self.times) if self.times else 0.0

    @property
    def max(self) -> float:
        return max(self.times) if self.times else 0.0

    def summary(self) -> str:
        return f"mean={self.mean:.2f}ms, std={self.std:.2f}ms, min={self.min:.2f}ms, max={self.max:.2f}ms"


@pytest.fixture
def timing_stats():
    """Fixture for collecting timing statistics."""
    return TimingStats()


# =============================================================================
# MODEL SIMULATION FIXTURES
# =============================================================================

@pytest.fixture
def simple_model_weights(rng) -> Dict[str, np.ndarray]:
    """Simulate a simple neural network's weights."""
    return {
        "layer1.weight": rng.randn(128, 784).astype(np.float32) * 0.01,
        "layer1.bias": np.zeros(128, dtype=np.float32),
        "layer2.weight": rng.randn(10, 128).astype(np.float32) * 0.01,
        "layer2.bias": np.zeros(10, dtype=np.float32),
    }


@pytest.fixture
def cnn_model_weights(rng) -> Dict[str, np.ndarray]:
    """Simulate a CNN's weights (larger model)."""
    return {
        "conv1.weight": rng.randn(32, 1, 3, 3).astype(np.float32) * 0.01,
        "conv1.bias": np.zeros(32, dtype=np.float32),
        "conv2.weight": rng.randn(64, 32, 3, 3).astype(np.float32) * 0.01,
        "conv2.bias": np.zeros(64, dtype=np.float32),
        "fc1.weight": rng.randn(128, 64 * 5 * 5).astype(np.float32) * 0.01,
        "fc1.bias": np.zeros(128, dtype=np.float32),
        "fc2.weight": rng.randn(10, 128).astype(np.float32) * 0.01,
        "fc2.bias": np.zeros(10, dtype=np.float32),
    }


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "rust: marks tests that require Rust backend"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks benchmark tests"
    )
    config.addinivalue_line(
        "markers", "attack: marks Byzantine attack simulation tests"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip slow tests unless explicitly requested."""
    if config.getoption("-m"):
        # User specified markers, don't modify
        return

    skip_slow = pytest.mark.skip(reason="slow test - run with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
