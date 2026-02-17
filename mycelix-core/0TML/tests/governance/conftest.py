"""
Pytest Configuration for Governance Tests
Week 7-8 Phase 5: Testing & Validation

Shared fixtures and configuration for all governance tests
"""

import pytest
import asyncio
from typing import Dict, Any


# Configure asyncio for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as requiring asyncio"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "security: mark test as security test"
    )
    config.addinivalue_line(
        "markers",
        "performance: mark test as performance benchmark"
    )
