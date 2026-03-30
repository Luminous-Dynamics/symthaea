# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Holochain Credits Bridge Integration

Tests the integration between ZeroTrustML reputation system and Holochain Credits DNA.
"""

import pytest
import pytest_asyncio
import asyncio
from zerotrustml.holochain.bridges.holochain_credits_bridge import (
    HolochainCreditsBridge,
    integrate_credits_with_reputation,
    CreditStats,
)


@pytest_asyncio.fixture
async def bridge():
    """Create Holochain Credits Bridge (mock mode)"""
    bridge = HolochainCreditsBridge(
        conductor_url="ws://localhost:8888",
        enabled=False  # Use mock mode for testing
    )
    await bridge.connect()
    return bridge


@pytest.mark.asyncio
async def test_issue_credits_quality_gradient(bridge):
    """Test issuing credits for quality gradient"""
    # Issue credits
    credits = await bridge.issue_credits(
        node_id=1,
        event_type="quality_gradient",
        pogq_score=0.9,
        gradient_hash="abc123",
        verifiers=[2, 3, 4]
    )

    # Should issue 90 credits (0.9 * 100)
    assert credits == 90

    # Check balance
    balance = await bridge.get_balance(node_id=1)
    assert balance == 90


@pytest.mark.asyncio
async def test_issue_credits_byzantine_detection(bridge):
    """Test issuing credits for Byzantine detection"""
    credits = await bridge.issue_credits(
        node_id=1,
        event_type="byzantine_detection",
        caught_node_id=5,
        verifiers=[2, 3]
    )

    # Should issue 50 credits (fixed reward)
    assert credits == 50

    balance = await bridge.get_balance(node_id=1)
    assert balance == 50


@pytest.mark.asyncio
async def test_issue_credits_peer_validation(bridge):
    """Test issuing credits for peer validation"""
    credits = await bridge.issue_credits(
        node_id=1,
        event_type="peer_validation",
        validated_node_id=3,
        gradient_hash="def456"
    )

    # Should issue 10 credits (fixed reward)
    assert credits == 10

    balance = await bridge.get_balance(node_id=1)
    assert balance == 10


@pytest.mark.asyncio
async def test_issue_credits_network_contribution(bridge):
    """Test issuing credits for network contribution"""
    credits = await bridge.issue_credits(
        node_id=1,
        event_type="network_contribution",
        uptime_hours=24
    )

    # Should issue 24 credits (1 per hour)
    assert credits == 24

    balance = await bridge.get_balance(node_id=1)
    assert balance == 24


@pytest.mark.asyncio
async def test_multiple_credits_accumulate(bridge):
    """Test that multiple credit issuances accumulate"""
    # Issue multiple credits
    await bridge.issue_credits(
        node_id=1,
        event_type="quality_gradient",
        pogq_score=0.8,
        verifiers=[2, 3, 4]
    )  # 80 credits

    await bridge.issue_credits(
        node_id=1,
        event_type="byzantine_detection",
        caught_node_id=5,
        verifiers=[2, 3]
    )  # 50 credits

    await bridge.issue_credits(
        node_id=1,
        event_type="peer_validation",
        validated_node_id=3
    )  # 10 credits

    # Total should be 80 + 50 + 10 = 140
    balance = await bridge.get_balance(node_id=1)
    assert balance == 140


@pytest.mark.asyncio
async def test_transfer_credits(bridge):
    """Test transferring credits between nodes"""
    # Give node 1 some credits
    await bridge.issue_credits(
        node_id=1,
        event_type="quality_gradient",
        pogq_score=1.0,
        verifiers=[2, 3, 4]
    )  # 100 credits

    # Transfer 50 credits to node 2
    success = await bridge.transfer(
        from_node_id=1,
        to_node_id=2,
        amount=50
    )

    assert success is True

    # Check balances
    balance_1 = await bridge.get_balance(node_id=1)
    balance_2 = await bridge.get_balance(node_id=2)

    assert balance_1 == 50
    assert balance_2 == 50


@pytest.mark.asyncio
async def test_transfer_insufficient_balance(bridge):
    """Test that transfer fails with insufficient balance"""
    # Give node 1 only 30 credits
    await bridge.issue_credits(
        node_id=1,
        event_type="peer_validation",
        validated_node_id=3
    )  # 10 credits

    await bridge.issue_credits(
        node_id=1,
        event_type="peer_validation",
        validated_node_id=4
    )  # 10 credits

    await bridge.issue_credits(
        node_id=1,
        event_type="peer_validation",
        validated_node_id=5
    )  # 10 credits

    # Try to transfer 50 credits (more than balance)
    success = await bridge.transfer(
        from_node_id=1,
        to_node_id=2,
        amount=50
    )

    assert success is False

    # Balance should be unchanged
    balance = await bridge.get_balance(node_id=1)
    assert balance == 30


@pytest.mark.asyncio
async def test_pogq_score_scaling(bridge):
    """Test that PoGQ scores correctly scale to credits"""
    test_cases = [
        (0.5, 50),   # Minimum valid score
        (0.75, 75),
        (0.9, 90),
        (1.0, 100),  # Maximum score
        (0.0, 0),    # Zero score
    ]

    for pogq_score, expected_credits in test_cases:
        credits = await bridge.issue_credits(
            node_id=1,
            event_type="quality_gradient",
            pogq_score=pogq_score,
            verifiers=[2, 3, 4]
        )

        assert credits == expected_credits, \
            f"PoGQ score {pogq_score} should give {expected_credits} credits, got {credits}"


@pytest.mark.asyncio
async def test_integrate_with_reputation_event(bridge):
    """Test integration with reputation events"""
    # Simulate reputation event from adaptive_byzantine_resistance
    reputation_event = {
        'type': 'quality_gradient',
        'pogq_score': 0.85,
        'gradient_hash': 'abc123',
        'verifiers': [2, 3, 4],
    }

    credits = await integrate_credits_with_reputation(
        bridge=bridge,
        node_id=1,
        reputation_event=reputation_event
    )

    assert credits == 85  # 0.85 * 100
    assert await bridge.get_balance(node_id=1) == 85


@pytest.mark.asyncio
async def test_multiple_nodes_independent_balances(bridge):
    """Test that multiple nodes have independent balances"""
    # Issue credits to different nodes
    await bridge.issue_credits(node_id=1, event_type="peer_validation", validated_node_id=3)  # 10
    await bridge.issue_credits(node_id=2, event_type="peer_validation", validated_node_id=4)  # 10
    await bridge.issue_credits(node_id=3, event_type="peer_validation", validated_node_id=5)  # 10

    await bridge.issue_credits(
        node_id=1,
        event_type="quality_gradient",
        pogq_score=0.5,
        verifiers=[2, 3, 4]
    )  # 50

    # Check independent balances
    assert await bridge.get_balance(node_id=1) == 60  # 10 + 50
    assert await bridge.get_balance(node_id=2) == 10
    assert await bridge.get_balance(node_id=3) == 10


@pytest.mark.asyncio
async def test_export_audit_trail_json(bridge):
    """Test exporting audit trail in JSON format"""
    # Issue some credits
    await bridge.issue_credits(
        node_id=1,
        event_type="quality_gradient",
        pogq_score=0.9,
        verifiers=[2, 3, 4]
    )

    await bridge.issue_credits(
        node_id=1,
        event_type="byzantine_detection",
        caught_node_id=5,
        verifiers=[2, 3]
    )

    # Export audit trail
    trail = await bridge.export_audit_trail(node_id=1, format="json")

    assert 'node_id' in trail
    assert trail['node_id'] == 1
    assert 'exported_at' in trail


@pytest.mark.asyncio
async def test_issuance_history_tracking(bridge):
    """Test that issuance history is tracked"""
    # Issue credits
    await bridge.issue_credits(
        node_id=1,
        event_type="quality_gradient",
        pogq_score=0.9,
        verifiers=[2, 3, 4]
    )

    await bridge.issue_credits(
        node_id=1,
        event_type="byzantine_detection",
        caught_node_id=5,
        verifiers=[2, 3]
    )

    # Check history
    assert len(bridge.issuance_history) == 2

    first = bridge.issuance_history[0]
    assert first.node_id == 1
    assert first.amount == 90
    assert first.reason == "quality_gradient"

    second = bridge.issuance_history[1]
    assert second.node_id == 1
    assert second.amount == 50
    assert second.reason == "byzantine_detection"


# Integration test with reputation system (requires adaptive_byzantine_resistance)
@pytest.mark.skipif(
    True,  # Skip by default (requires full system)
    reason="Requires adaptive_byzantine_resistance integration"
)
@pytest.mark.asyncio
async def test_full_integration_with_abr():
    """Full integration test with Byzantine resistance system"""
    from zerotrustml.experimental.adaptive_byzantine_resistance import (
        AdaptiveByzantineResistance,
    )
    import numpy as np

    # Initialize systems
    abr = AdaptiveByzantineResistance()
    bridge = HolochainCreditsBridge(enabled=False)
    await bridge.connect()

    # Simulate gradient validation
    node_id = 1
    gradient = np.random.randn(1000, 100).astype(np.float32)

    # Update reputation in ABR
    abr.update_reputation(
        node_id=node_id,
        gradient=gradient,
        validation_passed=True,
        pogq_score=0.9,
        anomaly_score=0.1
    )

    # Issue credits based on reputation
    reputation_event = {
        'type': 'quality_gradient',
        'pogq_score': 0.9,
        'gradient_hash': 'test_hash',
        'verifiers': [2, 3, 4],
    }

    credits = await integrate_credits_with_reputation(
        bridge=bridge,
        node_id=node_id,
        reputation_event=reputation_event
    )

    assert credits == 90

    # Check reputation level
    level = abr.get_reputation_level(node_id)
    print(f"Reputation level: {level}")
    print(f"Credits earned: {credits}")
    print(f"Total balance: {await bridge.get_balance(node_id)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
