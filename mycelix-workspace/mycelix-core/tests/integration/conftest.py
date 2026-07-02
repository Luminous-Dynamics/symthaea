# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Pytest fixtures for Mycelix FL network integration testing.

Provides comprehensive fixtures for setting up FL networks with various
configurations including honest nodes, Byzantine nodes, and network topologies.
"""

import pytest
import asyncio
import numpy as np
import time
import json
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
from contextlib import asynccontextmanager

# Import fixtures and scenarios
from .fixtures.network import FLNetwork, NetworkConfig
from .fixtures.nodes import HonestNode, ByzantineNode, NodeBehavior
from .fixtures.metrics import MetricsCollector, TestReport


# ============================================================================
# Session-scoped fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory(prefix="mycelix_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def report_dir(temp_dir):
    """Create a directory for test reports."""
    report_path = temp_dir / "reports"
    report_path.mkdir(exist_ok=True)
    return report_path


# ============================================================================
# Network configuration fixtures
# ============================================================================

@pytest.fixture
def default_network_config():
    """Default network configuration for tests."""
    return NetworkConfig(
        num_honest_nodes=10,
        num_byzantine_nodes=0,
        gradient_dimension=1000,
        aggregation_strategy="trimmed_mean",
        byzantine_threshold=0.33,
        detection_enabled=True,
        reputation_enabled=True,
        holochain_mock=True,
        consensus_rounds=3,
        timeout_ms=5000,
    )


@pytest.fixture
def small_network_config():
    """Small network for quick tests."""
    return NetworkConfig(
        num_honest_nodes=5,
        num_byzantine_nodes=0,
        gradient_dimension=100,
        aggregation_strategy="fedavg",
        byzantine_threshold=0.33,
        detection_enabled=True,
        reputation_enabled=True,
        holochain_mock=True,
        consensus_rounds=1,
        timeout_ms=2000,
    )


@pytest.fixture
def large_network_config():
    """Large network for stress tests."""
    return NetworkConfig(
        num_honest_nodes=20,
        num_byzantine_nodes=0,
        gradient_dimension=10000,
        aggregation_strategy="krum",
        byzantine_threshold=0.33,
        detection_enabled=True,
        reputation_enabled=True,
        holochain_mock=True,
        consensus_rounds=5,
        timeout_ms=30000,
    )


@pytest.fixture
def byzantine_network_config():
    """Network with Byzantine nodes for resilience tests."""
    return NetworkConfig(
        num_honest_nodes=14,
        num_byzantine_nodes=6,  # 30% Byzantine
        gradient_dimension=1000,
        aggregation_strategy="multi_krum",
        byzantine_threshold=0.33,
        detection_enabled=True,
        reputation_enabled=True,
        holochain_mock=True,
        consensus_rounds=3,
        timeout_ms=10000,
    )


@pytest.fixture
def max_byzantine_config():
    """Network at maximum Byzantine tolerance (45%)."""
    return NetworkConfig(
        num_honest_nodes=11,
        num_byzantine_nodes=9,  # 45% Byzantine
        gradient_dimension=1000,
        aggregation_strategy="bulyan",
        byzantine_threshold=0.45,
        detection_enabled=True,
        reputation_enabled=True,
        holochain_mock=True,
        consensus_rounds=5,
        timeout_ms=15000,
    )


# ============================================================================
# Network fixtures
# ============================================================================

@pytest.fixture
async def fl_network(default_network_config):
    """Create and manage an FL network."""
    network = FLNetwork(default_network_config)
    await network.start()
    yield network
    await network.shutdown()


@pytest.fixture
async def small_fl_network(small_network_config):
    """Create a small FL network for quick tests."""
    network = FLNetwork(small_network_config)
    await network.start()
    yield network
    await network.shutdown()


@pytest.fixture
async def large_fl_network(large_network_config):
    """Create a large FL network for stress tests."""
    network = FLNetwork(large_network_config)
    await network.start()
    yield network
    await network.shutdown()


@pytest.fixture
async def byzantine_fl_network(byzantine_network_config):
    """Create a network with Byzantine nodes."""
    network = FLNetwork(byzantine_network_config)
    await network.start()
    yield network
    await network.shutdown()


@pytest.fixture
async def max_byzantine_network(max_byzantine_config):
    """Create a network at maximum Byzantine tolerance."""
    network = FLNetwork(max_byzantine_config)
    await network.start()
    yield network
    await network.shutdown()


# ============================================================================
# Node fixtures
# ============================================================================

@pytest.fixture
def honest_node_factory():
    """Factory for creating honest nodes."""
    def create_node(node_id: str, gradient_dim: int = 1000) -> HonestNode:
        return HonestNode(
            node_id=node_id,
            gradient_dimension=gradient_dim,
            behavior=NodeBehavior.HONEST,
            noise_scale=0.01,
        )
    return create_node


@pytest.fixture
def byzantine_node_factory():
    """Factory for creating Byzantine nodes with various attack types."""
    def create_node(
        node_id: str,
        attack_type: str = "random",
        gradient_dim: int = 1000
    ) -> ByzantineNode:
        return ByzantineNode(
            node_id=node_id,
            gradient_dimension=gradient_dim,
            attack_type=attack_type,
        )
    return create_node


@pytest.fixture
def gradient_factory():
    """Factory for creating test gradients."""
    def create_gradient(
        dimension: int = 1000,
        mean: float = 0.0,
        std: float = 1.0,
        seed: Optional[int] = None
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(mean, std, dimension).astype(np.float32)
    return create_gradient


# ============================================================================
# Metrics fixtures
# ============================================================================

@pytest.fixture
def metrics_collector():
    """Create a metrics collector for the test."""
    collector = MetricsCollector()
    yield collector
    collector.finalize()


@pytest.fixture
def test_report(request, report_dir, metrics_collector):
    """Create a test report that is saved after the test."""
    report = TestReport(
        test_name=request.node.name,
        start_time=datetime.now(),
        metrics_collector=metrics_collector,
    )
    yield report
    report.end_time = datetime.now()
    report_file = report_dir / f"{request.node.name}_{report.start_time.strftime('%Y%m%d_%H%M%S')}.json"
    report.save(report_file)


# ============================================================================
# Utility fixtures
# ============================================================================

@pytest.fixture
def assert_gradient_quality():
    """Fixture providing gradient quality assertion helpers."""
    def _assert_quality(
        aggregated: np.ndarray,
        expected: np.ndarray,
        tolerance: float = 0.1,
        metric: str = "cosine"
    ):
        if metric == "cosine":
            similarity = np.dot(aggregated, expected) / (
                np.linalg.norm(aggregated) * np.linalg.norm(expected)
            )
            assert similarity >= (1 - tolerance), (
                f"Cosine similarity {similarity:.4f} below threshold {1 - tolerance:.4f}"
            )
        elif metric == "l2":
            distance = np.linalg.norm(aggregated - expected)
            expected_norm = np.linalg.norm(expected)
            relative_error = distance / expected_norm if expected_norm > 0 else distance
            assert relative_error <= tolerance, (
                f"Relative L2 error {relative_error:.4f} exceeds tolerance {tolerance:.4f}"
            )
        elif metric == "linf":
            max_diff = np.max(np.abs(aggregated - expected))
            assert max_diff <= tolerance, (
                f"Max absolute difference {max_diff:.4f} exceeds tolerance {tolerance:.4f}"
            )
    return _assert_quality


@pytest.fixture
def timing_context():
    """Context manager for timing operations."""
    @asynccontextmanager
    async def _timing(operation_name: str):
        start = time.perf_counter()
        result = {"operation": operation_name, "start": start}
        try:
            yield result
        finally:
            result["end"] = time.perf_counter()
            result["duration_ms"] = (result["end"] - result["start"]) * 1000
    return _timing


@pytest.fixture
def wait_for_condition():
    """Fixture for waiting on async conditions with timeout."""
    async def _wait(
        condition: Callable[[], bool],
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.1,
        description: str = "condition"
    ):
        start = time.time()
        while time.time() - start < timeout_s:
            if condition():
                return True
            await asyncio.sleep(poll_interval_s)
        raise TimeoutError(f"Timed out waiting for {description} after {timeout_s}s")
    return _wait


# ============================================================================
# Pytest hooks and markers
# ============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "byzantine: marks tests involving Byzantine fault tolerance"
    )
    config.addinivalue_line(
        "markers", "stress: marks stress/load tests"
    )
    config.addinivalue_line(
        "markers", "network: marks tests requiring network simulation"
    )
    config.addinivalue_line(
        "markers", "healthcare: marks healthcare-specific scenario tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    if config.getoption("-m"):
        return  # User specified markers, don't modify

    # Add skip markers for slow tests unless explicitly requested
    skip_slow = pytest.mark.skip(reason="Slow test - use -m slow to run")
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow", default=False):
            pass  # Optionally skip slow tests


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--byzantine-ratio",
        type=float,
        default=0.3,
        help="Ratio of Byzantine nodes in tests"
    )
    parser.addoption(
        "--fl-rounds",
        type=int,
        default=10,
        help="Number of FL rounds for long-running tests"
    )


@pytest.fixture
def cli_options(request):
    """Access CLI options in tests."""
    return {
        "run_slow": request.config.getoption("--run-slow"),
        "byzantine_ratio": request.config.getoption("--byzantine-ratio"),
        "fl_rounds": request.config.getoption("--fl-rounds"),
    }


# ============================================================================
# FL Aggregator mock/integration
# ============================================================================

@pytest.fixture
def aggregator():
    """Create an FL aggregator instance."""
    from .fixtures.network import FLAggregator
    return FLAggregator()


@pytest.fixture
def zome_validator():
    """Create a zome-level validator for gradient validation."""
    from .fixtures.network import ZomeValidator
    return ZomeValidator()


@pytest.fixture
def reputation_bridge():
    """Create a reputation bridge for score management."""
    from .fixtures.network import ReputationBridge
    return ReputationBridge()
