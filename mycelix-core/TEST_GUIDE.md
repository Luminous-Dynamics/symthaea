# Mycelix Testing Guide

This document describes the testing infrastructure, conventions, and best practices for the Mycelix project.

## Table of Contents

1. [Overview](#overview)
2. [Test Categories](#test-categories)
3. [Running Tests](#running-tests)
4. [Writing Tests](#writing-tests)
5. [Service Dependencies](#service-dependencies)
6. [CI Integration](#ci-integration)
7. [Troubleshooting](#troubleshooting)

---

## Overview

Mycelix uses a layered testing approach that separates tests by their external dependencies:

- **Unit Tests**: No external services required, run fast
- **Integration Tests**: Require external services (Holochain, Redis, etc.)
- **Smoke Tests**: Quick sanity checks for CI pipelines

The test infrastructure automatically detects available services and skips tests appropriately, while providing clear feedback about what's missing.

### Key Files

| File | Purpose |
|------|---------|
| `pytest.ini` | Pytest configuration with markers |
| `conftest.py` | Service detection and fixtures |
| `scripts/run_tests.py` | CI-friendly test runner |
| `0TML/src/tests/conftest.py` | Additional fixtures for ML tests |

---

## Test Categories

### Unit Tests (`@pytest.mark.unit`)

Unit tests have **no external dependencies** and should run on any machine with Python installed.

```python
import pytest

@pytest.mark.unit
def test_gradient_normalization():
    """Test gradient normalization logic."""
    gradient = np.array([3.0, 4.0])
    normalized = normalize_gradient(gradient)
    assert np.linalg.norm(normalized) == pytest.approx(1.0)
```

### Integration Tests (`@pytest.mark.integration`)

Integration tests require external services. Use specific markers to indicate which service:

```python
import pytest

@pytest.mark.integration
@pytest.mark.holochain
async def test_gradient_storage_dht():
    """Test storing gradients in Holochain DHT."""
    # Requires running Holochain conductor
    ...

@pytest.mark.integration
@pytest.mark.pytorch
def test_real_model_training():
    """Test with real PyTorch model."""
    # Requires PyTorch installed
    ...
```

### Available Markers

| Marker | Description | Service Required |
|--------|-------------|-----------------|
| `@pytest.mark.unit` | Pure unit tests | None |
| `@pytest.mark.integration` | Integration tests | Various |
| `@pytest.mark.holochain` | Holochain tests | Conductor on port 8888 |
| `@pytest.mark.rust` | Rust bridge tests | Built Rust extension |
| `@pytest.mark.pytorch` | PyTorch tests | `pip install torch` |
| `@pytest.mark.tensorflow` | TensorFlow tests | `pip install tensorflow` |
| `@pytest.mark.redis` | Redis tests | Redis on port 6379 |
| `@pytest.mark.postgres` | PostgreSQL tests | PostgreSQL on port 5432 |
| `@pytest.mark.ethereum` | Ethereum tests | Anvil or Ganache |
| `@pytest.mark.slow` | Long-running tests | None |
| `@pytest.mark.benchmark` | Performance tests | None |
| `@pytest.mark.smoke` | Quick CI smoke tests | None |

---

## Running Tests

### Using the Test Runner Script

The recommended way to run tests:

```bash
# Run unit tests only (fast, no deps)
python scripts/run_tests.py --unit

# Run integration tests
python scripts/run_tests.py --integration

# Run all tests
python scripts/run_tests.py --all

# Run smoke tests (quick CI check)
python scripts/run_tests.py --smoke

# Check service status
python scripts/run_tests.py --status
```

### Using pytest Directly

```bash
# Run unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Run specific service tests
pytest -m holochain
pytest -m pytorch

# Run multiple categories
pytest -m "unit or smoke"

# Exclude slow tests
pytest -m "not slow"

# Run with verbose output
pytest -v -m unit

# Run with coverage
pytest --cov=0TML/src --cov-report=html -m unit
```

### CI Mode

In CI mode, the test runner will **fail** if required services are unavailable:

```bash
# Fail if Holochain not available
python scripts/run_tests.py --ci --require-holochain --integration

# Fail if PyTorch not available
python scripts/run_tests.py --ci --require-pytorch --unit

# Multiple requirements
python scripts/run_tests.py --ci --require-holochain --require-redis --integration
```

Exit codes:
- `0`: All tests passed
- `1`: Some tests failed
- `2`: Missing required services (CI mode only)
- `3`: Configuration error

---

## Writing Tests

### DO: Use Markers Instead of pytest.skip()

**Bad (old pattern):**
```python
def test_holochain_integration():
    try:
        from holochain_bridge import HolochainBridge
    except ImportError:
        pytest.skip("Holochain bridge not available")  # DON'T DO THIS
    ...
```

**Good (new pattern):**
```python
@pytest.mark.integration
@pytest.mark.holochain
def test_holochain_integration():
    from holochain_bridge import HolochainBridge
    ...
```

### DO: Use conftest.py Decorators

The root `conftest.py` provides skip decorators:

```python
from conftest import skip_if_no_pytorch, skip_if_no_holochain

@skip_if_no_pytorch()
def test_pytorch_model():
    import torch
    ...

@skip_if_no_holochain()
async def test_conductor_connection():
    ...
```

### DO: Use Fixtures for Common Setup

```python
def test_byzantine_detection(byzantine_gradients, rng):
    """Test detection using fixtures from conftest.py."""
    detector = ByzantineDetector()
    results = detector.detect(byzantine_gradients)
    assert results["detection_rate"] > 0.9
```

### DON'T: Silent Skips

**Bad:**
```python
def test_feature():
    if not some_condition:
        return  # Silent skip - test appears to pass!
    ...
```

**Bad:**
```python
def test_feature():
    if not some_condition:
        pytest.skip()  # Generic skip - hard to track
    ...
```

### DO: Document Skip Reasons

If you must use pytest.skip(), use the standard prefixes:

```python
# For missing services
pytest.skip("MISSING_SERVICE: Holochain conductor not running")

# For missing dependencies
pytest.skip("MISSING_DEPENDENCY: scipy not installed")

# For legitimate platform-specific skips
pytest.skip("PLATFORM: Test only valid on Linux")

# For incomplete/WIP tests
pytest.skip("WIP: Feature not yet implemented")
```

---

## Service Dependencies

### Starting Services for Integration Tests

#### Holochain

```bash
# Option 1: Use holonix
nix develop
holochain -c conductor-config.yaml

# Option 2: Set environment variable
export RUN_HOLOCHAIN_TESTS=1
```

#### Redis

```bash
# With docker
docker run -d -p 6379:6379 redis:alpine

# With nix
nix-shell -p redis --run "redis-server"

# Native
redis-server
```

#### PostgreSQL

```bash
# With docker
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:alpine

# With nix
nix-shell -p postgresql --run "pg_ctl -D /tmp/pgdata init && pg_ctl -D /tmp/pgdata start"
```

#### Ethereum (Anvil)

```bash
# Install foundry first
curl -L https://foundry.paradigm.xyz | bash
foundryup

# Run anvil
anvil
```

### Service Status Check

```bash
python scripts/run_tests.py --status
```

Output:
```
======================================================================
SERVICE AVAILABILITY STATUS
======================================================================

External Services:
  [OK] Holochain Conductor (port 8888)
  [--] Redis (port 6379)
  [--] PostgreSQL (port 5432)
  [OK] Anvil (Foundry) (anvil binary)
  [--] Ganache (ganache binary)

Python Dependencies:
  [OK] PyTorch
  [--] TensorFlow
  [OK] SciPy
  [OK] Prometheus Client
  [--] py-solc-x
```

---

## CI Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: python scripts/run_tests.py --unit --junit-xml=test-results.xml
      - uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: test-results.xml

  integration-tests:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:alpine
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: python scripts/run_tests.py --ci --require-redis --integration --with-redis

  pytorch-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]" torch torchvision
      - run: python scripts/run_tests.py --ci --require-pytorch --with-pytorch
```

### Tracking Skipped Tests

The test infrastructure prefixes skip reasons to make them trackable:

- `MISSING_SERVICE:` - External service not running
- `MISSING_DEPENDENCY:` - Python package not installed
- `PLATFORM:` - Platform-specific limitation
- `WIP:` - Work in progress

CI dashboards can grep for these to track skip trends.

---

## Troubleshooting

### "Test was skipped but I expected it to run"

1. Check service status: `python scripts/run_tests.py --status`
2. Verify the service is actually running
3. Check if the test has the correct markers

### "Tests pass locally but fail in CI"

1. Check CI has required services
2. Use `--ci --require-X` to fail fast on missing services
3. Review skip reasons in CI logs

### "Can't find tests"

1. Check `pytest.ini` testpaths configuration
2. Ensure test files are named `test_*.py`
3. Ensure test functions are named `test_*`

### "Marker not recognized"

1. Run `pytest --markers` to see registered markers
2. Check `pytest.ini` for marker definitions
3. Ensure `--strict-markers` is set (catches typos)

### "Import errors in tests"

1. Check `pytest.ini` pythonpath configuration
2. Ensure dependencies are installed
3. Use appropriate markers for optional dependencies

---

## Migration from Old Skip Patterns

If you're updating existing tests, here's how to migrate:

### Before (scattered pytest.skip)

```python
def test_holochain_feature():
    if not os.environ.get("RUN_HOLOCHAIN_TESTS"):
        pytest.skip("Holochain not available")
    ...
```

### After (markers)

```python
@pytest.mark.integration
@pytest.mark.holochain
def test_holochain_feature():
    ...
```

### Files to Update

The following files still contain `pytest.skip()` calls that should be migrated:

1. `0TML/test_real_ml.py` - PyTorch dependency
2. `0TML/test_reconnection.py` - Rust bridge dependency
3. `0TML/tests/test_phase4_integration.py` - Multiple dependencies
4. `0TML/tests/test_polygon_attestation.py` - Ethereum dependency
5. `0TML/tests/test_holochain_integration.py` - Holochain dependency
6. ... (see grep results for full list)

---

## End-to-End (E2E) FL Pipeline Tests

The E2E tests validate the complete Federated Learning pipeline:
1. Proof generation (PoGQ)
2. DHT storage (Holochain)
3. Verification
4. FL aggregation

### E2E Test Modes

| Marker | Description | Requirements |
|--------|-------------|--------------|
| `@pytest.mark.e2e` | Full integration tests | Holochain conductor running |
| `@pytest.mark.e2e_stub` | Tests with mocked Holochain | None (CI-friendly) |

### Running E2E Tests

```bash
# Run E2E tests with mocked Holochain (CI-friendly)
pytest -m e2e_stub 0TML/tests/test_e2e_fl_pipeline.py

# Run full E2E tests with real Holochain (requires conductor)
pytest -m e2e 0TML/tests/test_e2e_fl_pipeline.py

# Run all E2E tests
pytest 0TML/tests/test_e2e_fl_pipeline.py

# Run with verbose output
pytest -v -m e2e_stub 0TML/tests/test_e2e_fl_pipeline.py

# Run performance/benchmark tests
pytest -m "e2e_stub and benchmark" 0TML/tests/test_e2e_fl_pipeline.py
```

### E2E Test Structure

The E2E test suite includes:

1. **Basic FL Round Tests** (`TestE2EFLPipelineStub`)
   - `test_e2e_fl_round_basic`: Honest nodes only
   - `test_e2e_fl_round_with_byzantine`: 3 honest + 1 Byzantine
   - `test_e2e_multiple_byzantine_attacks`: Multiple attack types
   - `test_e2e_proof_generation_and_verification`: PoGQ proof flow
   - `test_e2e_dht_storage_and_retrieval`: DHT operations
   - `test_e2e_reputation_updates`: Reputation system

2. **Edge Case Tests** (`TestEdgeCases`)
   - `test_network_partition_during_aggregation`: Network failures
   - `test_late_arriving_proofs`: Timeout handling
   - `test_invalid_proof_rejection`: Proof validation
   - `test_dht_disconnection_handling`: Connection failures
   - `test_high_latency_network`: High latency scenarios

3. **Performance Tests** (`TestPerformanceAssertions`)
   - `test_byzantine_detection_latency`: < 1ms per detection
   - `test_round_completion_time`: < 60s per round
   - `test_memory_usage_bounds`: < 500MB increase
   - `test_scalability_nodes`: Node count scaling

4. **Multi-Round Tests** (`TestMultiRoundFL`)
   - `test_multi_round_convergence`: Model convergence
   - `test_adaptive_attacker_detection`: Strategy-changing attackers

5. **Real Holochain Tests** (`TestE2EFLPipelineReal`)
   - `test_real_holochain_gradient_storage`: Real DHT storage
   - `test_real_holochain_full_round`: Full real integration

### Performance Thresholds

The E2E tests verify these performance claims:

| Metric | Target | Test |
|--------|--------|------|
| Byzantine detection latency | < 1ms per node | `test_byzantine_detection_latency` |
| Round completion time | < 60 seconds | `test_round_completion_time` |
| Memory usage increase | < 500MB | `test_memory_usage_bounds` |

### Example: Adding a New E2E Test

```python
@pytest.mark.e2e_stub
class TestMyNewFeature:
    """Tests for new FL feature."""

    @pytest.mark.asyncio
    async def test_new_feature(self, coordinator, mixed_nodes):
        """Test the new feature."""
        result = await coordinator.execute_round(mixed_nodes)

        assert result.success
        assert len(result.byzantine_detected) >= 1
        # Add your assertions
```

### CI/CD Integration

For CI pipelines, use the stub tests which don't require external services:

```yaml
# GitHub Actions example
e2e-tests:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    - run: pip install -e ".[dev]"
    - run: pytest -m e2e_stub --junit-xml=e2e-results.xml
```

For full integration testing with Holochain, see the `e2e` marker tests.

---

## Summary

1. **Use markers** instead of `pytest.skip()` for service dependencies
2. **Run unit tests** for fast feedback during development
3. **Run integration tests** when services are available
4. **Run E2E stub tests** in CI/CD pipelines
5. **Run full E2E tests** when Holochain conductor is available
6. **Use CI mode** to fail loudly on missing services
7. **Track skip reasons** with standard prefixes
8. **Check status** before debugging test failures
