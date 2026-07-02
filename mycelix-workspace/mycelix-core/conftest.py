# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Test Configuration - Root conftest.py
=============================================

This file provides:
1. Service availability detection (Holochain, Redis, PyTorch, etc.)
2. Automatic test skipping based on service availability
3. CI-friendly skip reasons that are trackable
4. Fixtures for common test needs

IMPORTANT: Tests should NOT use pytest.skip() for missing services.
Instead, use the appropriate markers:
    @pytest.mark.holochain - requires Holochain conductor
    @pytest.mark.rust      - requires Rust backend
    @pytest.mark.pytorch   - requires PyTorch
    @pytest.mark.redis     - requires Redis
    @pytest.mark.ethereum  - requires Anvil/Ganache
"""

import os
import socket
import subprocess
import sys
from typing import Dict, Optional
from pathlib import Path

import pytest

# numpy is optional for the conftest - tests that need it will import it
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# =============================================================================
# SERVICE AVAILABILITY DETECTION
# =============================================================================

class ServiceAvailability:
    """Detect which external services are available for testing."""

    _cache: Dict[str, bool] = {}

    @classmethod
    def check_port(cls, host: str, port: int, timeout: float = 1.0) -> bool:
        """Check if a TCP port is open."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    @classmethod
    def check_command(cls, cmd: str) -> bool:
        """Check if a command is available on PATH."""
        try:
            result = subprocess.run(
                ["which", cmd],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    @classmethod
    def holochain_available(cls) -> bool:
        """Check if Holochain conductor is running."""
        if "holochain" not in cls._cache:
            # Check environment variable first
            if os.environ.get("RUN_HOLOCHAIN_TESTS"):
                cls._cache["holochain"] = True
            else:
                # Check if conductor is running on default port
                cls._cache["holochain"] = cls.check_port("127.0.0.1", 8888)
        return cls._cache["holochain"]

    @classmethod
    def rust_bridge_available(cls) -> bool:
        """Check if Rust bridge is available."""
        if "rust" not in cls._cache:
            try:
                from zerotrustml.holochain.bridges.holochain_credits_bridge_rust import (
                    HolochainCreditsBridge as RustBridge,
                )
                cls._cache["rust"] = RustBridge is not None and hasattr(RustBridge, "is_connected")
            except ImportError:
                cls._cache["rust"] = False
        return cls._cache["rust"]

    @classmethod
    def pytorch_available(cls) -> bool:
        """Check if PyTorch is available."""
        if "pytorch" not in cls._cache:
            try:
                import torch
                cls._cache["pytorch"] = True
            except ImportError:
                cls._cache["pytorch"] = False
        return cls._cache["pytorch"]

    @classmethod
    def tensorflow_available(cls) -> bool:
        """Check if TensorFlow is available."""
        if "tensorflow" not in cls._cache:
            try:
                import tensorflow
                cls._cache["tensorflow"] = True
            except ImportError:
                cls._cache["tensorflow"] = False
        return cls._cache["tensorflow"]

    @classmethod
    def redis_available(cls) -> bool:
        """Check if Redis is running."""
        if "redis" not in cls._cache:
            cls._cache["redis"] = cls.check_port("127.0.0.1", 6379)
        return cls._cache["redis"]

    @classmethod
    def postgres_available(cls) -> bool:
        """Check if PostgreSQL is running."""
        if "postgres" not in cls._cache:
            cls._cache["postgres"] = cls.check_port("127.0.0.1", 5432)
        return cls._cache["postgres"]

    @classmethod
    def anvil_available(cls) -> bool:
        """Check if Anvil (Foundry) is available."""
        if "anvil" not in cls._cache:
            cls._cache["anvil"] = cls.check_command("anvil")
        return cls._cache["anvil"]

    @classmethod
    def ganache_available(cls) -> bool:
        """Check if Ganache is available."""
        if "ganache" not in cls._cache:
            cls._cache["ganache"] = cls.check_command("ganache")
        return cls._cache["ganache"]

    @classmethod
    def solcx_available(cls) -> bool:
        """Check if py-solc-x is available."""
        if "solcx" not in cls._cache:
            try:
                from solcx import compile_source
                cls._cache["solcx"] = True
            except ImportError:
                cls._cache["solcx"] = False
        return cls._cache["solcx"]

    @classmethod
    def scipy_available(cls) -> bool:
        """Check if scipy is available."""
        if "scipy" not in cls._cache:
            try:
                import scipy
                cls._cache["scipy"] = True
            except ImportError:
                cls._cache["scipy"] = False
        return cls._cache["scipy"]

    @classmethod
    def identity_modules_available(cls) -> bool:
        """Check if identity modules are available."""
        if "identity" not in cls._cache:
            try:
                from zerotrustml.identity import DIDManager
                cls._cache["identity"] = True
            except ImportError:
                cls._cache["identity"] = False
        return cls._cache["identity"]

    @classmethod
    def prometheus_available(cls) -> bool:
        """Check if prometheus-client is available."""
        if "prometheus" not in cls._cache:
            try:
                from prometheus_client import REGISTRY
                cls._cache["prometheus"] = True
            except ImportError:
                cls._cache["prometheus"] = False
        return cls._cache["prometheus"]

    @classmethod
    def get_status_report(cls) -> Dict[str, bool]:
        """Get a full status report of all services."""
        return {
            "holochain": cls.holochain_available(),
            "rust_bridge": cls.rust_bridge_available(),
            "pytorch": cls.pytorch_available(),
            "tensorflow": cls.tensorflow_available(),
            "redis": cls.redis_available(),
            "postgres": cls.postgres_available(),
            "anvil": cls.anvil_available(),
            "ganache": cls.ganache_available(),
            "solcx": cls.solcx_available(),
            "scipy": cls.scipy_available(),
            "identity": cls.identity_modules_available(),
            "prometheus": cls.prometheus_available(),
        }


# =============================================================================
# PYTEST HOOKS
# =============================================================================

def pytest_configure(config):
    """Configure pytest with service availability info."""
    # Print service status at start of test run
    if config.getoption("verbose", 0) > 0:
        print("\n" + "=" * 70)
        print("SERVICE AVAILABILITY CHECK")
        print("=" * 70)
        status = ServiceAvailability.get_status_report()
        for service, available in status.items():
            symbol = "[OK]" if available else "[--]"
            print(f"  {symbol} {service}")
        print("=" * 70 + "\n")


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests based on service availability.

    This replaces scattered pytest.skip() calls with centralized,
    trackable skip logic.
    """

    # Build skip markers for unavailable services
    skip_reasons = {}

    if not ServiceAvailability.holochain_available():
        skip_reasons["holochain"] = pytest.mark.skip(
            reason="MISSING_SERVICE: Holochain conductor not running (set RUN_HOLOCHAIN_TESTS=1 or start conductor on port 8888)"
        )

    if not ServiceAvailability.rust_bridge_available():
        skip_reasons["rust"] = pytest.mark.skip(
            reason="MISSING_SERVICE: Rust bridge not available (build with maturin develop)"
        )

    if not ServiceAvailability.pytorch_available():
        skip_reasons["pytorch"] = pytest.mark.skip(
            reason="MISSING_DEPENDENCY: PyTorch not installed (pip install torch)"
        )

    if not ServiceAvailability.tensorflow_available():
        skip_reasons["tensorflow"] = pytest.mark.skip(
            reason="MISSING_DEPENDENCY: TensorFlow not installed (pip install tensorflow)"
        )

    if not ServiceAvailability.redis_available():
        skip_reasons["redis"] = pytest.mark.skip(
            reason="MISSING_SERVICE: Redis not running on port 6379"
        )

    if not ServiceAvailability.postgres_available():
        skip_reasons["postgres"] = pytest.mark.skip(
            reason="MISSING_SERVICE: PostgreSQL not running on port 5432"
        )

    ethereum_available = ServiceAvailability.anvil_available() or ServiceAvailability.ganache_available()
    if not ethereum_available:
        skip_reasons["ethereum"] = pytest.mark.skip(
            reason="MISSING_SERVICE: No Ethereum test network (install foundry or ganache)"
        )

    # Apply skip markers to tests
    for item in items:
        for marker_name, skip_marker in skip_reasons.items():
            if marker_name in item.keywords:
                item.add_marker(skip_marker)


def pytest_runtest_makereport(item, call):
    """
    Track skip reasons for CI reporting.

    Skips due to MISSING_SERVICE or MISSING_DEPENDENCY are tracked
    separately from other skips for CI dashboards.
    """
    if call.excinfo is not None and call.excinfo.typename == "Skipped":
        reason = str(call.excinfo.value)
        if "MISSING_SERVICE:" in reason or "MISSING_DEPENDENCY:" in reason:
            # These are expected skips when services aren't available
            # CI can track these separately
            pass


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def service_status():
    """Get service availability status."""
    return ServiceAvailability.get_status_report()


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    if not NUMPY_AVAILABLE:
        pytest.skip("MISSING_DEPENDENCY: numpy not available")
    return np.random.RandomState(42)


@pytest.fixture
def small_gradient(rng):
    """Small gradient for quick tests (1K params)."""
    if not NUMPY_AVAILABLE:
        pytest.skip("MISSING_DEPENDENCY: numpy not available")
    return rng.randn(1000).astype(np.float32)


@pytest.fixture
def medium_gradient(rng):
    """Medium gradient for realistic tests (100K params)."""
    if not NUMPY_AVAILABLE:
        pytest.skip("MISSING_DEPENDENCY: numpy not available")
    return rng.randn(100_000).astype(np.float32)


@pytest.fixture
def large_gradient(rng):
    """Large gradient for stress tests (1M params)."""
    if not NUMPY_AVAILABLE:
        pytest.skip("MISSING_DEPENDENCY: numpy not available")
    return rng.randn(1_000_000).astype(np.float32)


@pytest.fixture
def honest_gradients(rng):
    """Generate 10 honest gradients with natural variation."""
    if not NUMPY_AVAILABLE:
        pytest.skip("MISSING_DEPENDENCY: numpy not available")
    true_direction = rng.randn(10_000).astype(np.float32)
    true_direction /= np.linalg.norm(true_direction)

    gradients = {}
    for i in range(10):
        noise = rng.randn(10_000).astype(np.float32) * 0.1
        grad = true_direction + noise
        gradients[f"honest_{i}"] = grad

    return gradients


@pytest.fixture
def byzantine_gradients(rng):
    """Generate gradients with 30% Byzantine attackers."""
    if not NUMPY_AVAILABLE:
        pytest.skip("MISSING_DEPENDENCY: numpy not available")
    true_direction = rng.randn(10_000).astype(np.float32)
    true_direction /= np.linalg.norm(true_direction)

    gradients = {}

    # 7 honest nodes
    for i in range(7):
        noise = rng.randn(10_000).astype(np.float32) * 0.1
        gradients[f"honest_{i}"] = true_direction + noise

    # 3 Byzantine nodes
    gradients["byzantine_flip"] = -true_direction * 1.5
    gradients["byzantine_random"] = rng.randn(10_000).astype(np.float32) * 5.0
    gradients["byzantine_scale"] = true_direction * 100.0

    return gradients


# =============================================================================
# SKIP HELPERS
# =============================================================================

def skip_if_no_holochain():
    """Decorator to skip test if Holochain not available."""
    return pytest.mark.skipif(
        not ServiceAvailability.holochain_available(),
        reason="MISSING_SERVICE: Holochain conductor not running"
    )


def skip_if_no_rust():
    """Decorator to skip test if Rust bridge not available."""
    return pytest.mark.skipif(
        not ServiceAvailability.rust_bridge_available(),
        reason="MISSING_SERVICE: Rust bridge not available"
    )


def skip_if_no_pytorch():
    """Decorator to skip test if PyTorch not available."""
    return pytest.mark.skipif(
        not ServiceAvailability.pytorch_available(),
        reason="MISSING_DEPENDENCY: PyTorch not installed"
    )


def skip_if_no_tensorflow():
    """Decorator to skip test if TensorFlow not available."""
    return pytest.mark.skipif(
        not ServiceAvailability.tensorflow_available(),
        reason="MISSING_DEPENDENCY: TensorFlow not installed"
    )


def skip_if_no_redis():
    """Decorator to skip test if Redis not available."""
    return pytest.mark.skipif(
        not ServiceAvailability.redis_available(),
        reason="MISSING_SERVICE: Redis not running"
    )


def skip_if_no_ethereum():
    """Decorator to skip test if no Ethereum test network available."""
    return pytest.mark.skipif(
        not (ServiceAvailability.anvil_available() or ServiceAvailability.ganache_available()),
        reason="MISSING_SERVICE: No Ethereum test network"
    )


# Make helpers available for import
__all__ = [
    "ServiceAvailability",
    "skip_if_no_holochain",
    "skip_if_no_rust",
    "skip_if_no_pytorch",
    "skip_if_no_tensorflow",
    "skip_if_no_redis",
    "skip_if_no_ethereum",
]
