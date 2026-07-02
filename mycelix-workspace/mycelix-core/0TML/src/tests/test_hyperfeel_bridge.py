"""
Tests for the Rust-backed HyperFeel bridge.

These tests are intentionally light and are skipped automatically
when the `hyperfeel_cli` binary is not available on PATH.
"""

import shutil

import numpy as np
import pytest

from zerotrustml.hyperfeel_bridge import (
    encode_gradient,
    aggregate_hypergradients,
    HyperFeelUnavailable,
)


hyperfeel_cli_available = shutil.which("hyperfeel_cli") is not None


@pytest.mark.skipif(
    not hyperfeel_cli_available,
    reason="hyperfeel_cli binary not available on PATH",
)
def test_encode_gradient_basic():
    """Encode a small gradient and check basic structure."""
    grad = np.array([0.1, -0.3, 0.7], dtype=np.float32)
    hg = encode_gradient(grad)

    assert isinstance(hg, dict)
    assert "vector" in hg
    assert "data" in hg["vector"]
    assert len(hg["vector"]["data"]) == 16_384


@pytest.mark.skipif(
    not hyperfeel_cli_available,
    reason="hyperfeel_cli binary not available on PATH",
)
def test_aggregate_hypergradients_roundtrip():
    """Aggregate two encoded hypergradients and verify the result."""
    grad1 = np.random.randn(100).astype(np.float32)
    grad2 = np.random.randn(100).astype(np.float32)

    hg1 = encode_gradient(grad1)
    hg2 = encode_gradient(grad2)

    aggregated = aggregate_hypergradients([hg1, hg2])

    assert isinstance(aggregated, dict)
    assert "vector" in aggregated
    assert "data" in aggregated["vector"]
    assert len(aggregated["vector"]["data"]) == 16_384


def test_unavailable_binary_raises():
    """If the CLI is not present, the helper should raise HyperFeelUnavailable."""
    if hyperfeel_cli_available:
        pytest.skip("hyperfeel_cli is available; this test only checks the failure path")

    grad = np.array([0.1, 0.2], dtype=np.float32)
    with pytest.raises(HyperFeelUnavailable):
        encode_gradient(grad)

