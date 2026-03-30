# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix FL Integration Test Suite

Comprehensive end-to-end integration tests for the Mycelix Federated Learning system.

Test Categories:
- test_gradient_roundtrip: Basic gradient lifecycle tests
- test_byzantine_rejection: Malicious gradient rejection tests
- test_reputation_flow: Reputation system integration tests
- test_full_fl_round: Complete FL round with 20 nodes
- test_100_rounds: Long-running stability tests
- test_45_percent_byzantine: Maximum Byzantine tolerance tests
- test_recovery: Network partition and recovery tests

Scenarios:
- scenarios/healthcare: HIPAA-compliant 5-hospital FL scenario
- scenarios/adversarial: Coordinated attack scenarios

Usage:
    # Run all tests
    pytest tests/integration/ -v --html=report.html

    # Run specific test categories
    pytest tests/integration/ -v -m byzantine
    pytest tests/integration/ -v -m network
    pytest tests/integration/ -v -m healthcare

    # Run slow tests (100 rounds)
    pytest tests/integration/ -v -m slow --run-slow

    # Run with custom Byzantine ratio
    pytest tests/integration/ -v --byzantine-ratio=0.4

    # Run with custom FL rounds
    pytest tests/integration/ -v --fl-rounds=50
"""

__version__ = "1.0.0"
__author__ = "Mycelix Team"
