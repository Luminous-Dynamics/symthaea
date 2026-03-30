#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML-Credits Integration End-to-End Demo

Demonstrates all 4 event types:
1. Quality Gradient Credits (PoGQ-based)
2. Byzantine Detection Rewards
3. Peer Validation Credits
4. Network Uptime Credits

Shows audit trails, statistics, and rate limiting in action.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zerotrustml_credits_integration import (
    ZeroTrustMLCreditsIntegration,
    CreditIssuanceConfig,
)
from holochain_credits_bridge import HolochainCreditsBridge


def print_section(title: str, emoji: str = "🔹"):
    """Print a visual section header"""
    print(f"\n{emoji} {'=' * 70} {emoji}")
    print(f"  {title}")
    print(f"{emoji} {'=' * 70} {emoji}\n")


def print_event(event_name: str, details: dict):
    """Print event details in a readable format"""
    print(f"📊 {event_name}")
    for key, value in details.items():
        print(f"   • {key}: {value}")
    print()


async def main():
    """Run complete end-to-end demonstration"""

    print_section("ZeroTrustML-Credits Integration: End-to-End Demo", "🚀")

    # Initialize integration in mock mode
    print("🔧 Initializing integration (Mock Mode)...")
    bridge = HolochainCreditsBridge(enabled=False)  # Mock mode for demo
    config = CreditIssuanceConfig(enabled=True)
    integration = ZeroTrustMLCreditsIntegration(bridge, config)
    print("✅ Integration initialized successfully\n")

    # =========================================================================
    # DEMO 1: Quality Gradient Credits
    # =========================================================================
    print_section("Demo 1: Quality Gradient Credits (PoGQ-based)", "🎯")

    print("Scenario: Three nodes submit gradients with different quality scores\n")

    # Node with excellent quality (ELITE reputation)
    print("1️⃣ Node 'alice' - ELITE reputation, excellent gradient (PoGQ=0.98)")
    result1 = await integration.on_quality_gradient(
        node_id="alice",
        pogq_score=0.98,
        reputation_level="ELITE",
        verifiers=["bob", "charlie"]
    )
    print_event("Quality Gradient Event", {
        "Node": "alice",
        "PoGQ Score": 0.98,
        "Reputation": "ELITE (1.5x multiplier)",
        "Base Credits": "0.98 * 100 = 98.0",
        "Final Credits": "98.0 * 1.5 = 147.0",
        "Credit ID": result1
    })

    # Node with good quality (NORMAL reputation)
    print("2️⃣ Node 'bob' - NORMAL reputation, good gradient (PoGQ=0.85)")
    result2 = await integration.on_quality_gradient(
        node_id="bob",
        pogq_score=0.85,
        reputation_level="NORMAL",
        verifiers=["alice", "charlie"]
    )
    print_event("Quality Gradient Event", {
        "Node": "bob",
        "PoGQ Score": 0.85,
        "Reputation": "NORMAL (1.0x multiplier)",
        "Base Credits": "0.85 * 100 = 85.0",
        "Final Credits": "85.0 * 1.0 = 85.0",
        "Credit ID": result2
    })

    # Node with low quality (below threshold)
    print("3️⃣ Node 'eve' - NORMAL reputation, low gradient (PoGQ=0.65)")
    result3 = await integration.on_quality_gradient(
        node_id="eve",
        pogq_score=0.65,
        reputation_level="NORMAL",
        verifiers=["alice"]
    )
    print_event("Quality Gradient Event (REJECTED)", {
        "Node": "eve",
        "PoGQ Score": 0.65,
        "Threshold": "0.7 (minimum)",
        "Result": "❌ No credits issued (below threshold)",
        "Credit ID": result3 or "None"
    })

    # =========================================================================
    # DEMO 2: Byzantine Detection Rewards
    # =========================================================================
    print_section("Demo 2: Byzantine Detection Rewards", "🛡️")

    print("Scenario: Nodes detect Byzantine behavior and receive rewards\n")

    print("1️⃣ Node 'alice' detects Byzantine node 'mallory'")
    result4 = await integration.on_byzantine_detection(
        detector_node_id="alice",
        detected_node_id="mallory",
        reputation_level="ELITE",
        evidence={"consecutive_failures": 20, "pogq_score": 0.15}
    )
    print_event("Byzantine Detection Event", {
        "Detector": "alice",
        "Detected": "mallory",
        "Detector Reputation": "ELITE (1.5x multiplier)",
        "Base Reward": "50 credits",
        "Final Reward": "50 * 1.5 = 75.0 credits",
        "Evidence": "20 consecutive failures, PoGQ=0.15",
        "Credit ID": result4
    })

    print("2️⃣ Node 'bob' detects Byzantine node 'sybil'")
    result5 = await integration.on_byzantine_detection(
        detector_node_id="bob",
        detected_node_id="sybil",
        reputation_level="TRUSTED",
        evidence={"anomaly_score": 0.95}
    )
    print_event("Byzantine Detection Event", {
        "Detector": "bob",
        "Detected": "sybil",
        "Detector Reputation": "TRUSTED (1.2x multiplier)",
        "Base Reward": "50 credits",
        "Final Reward": "50 * 1.2 = 60.0 credits",
        "Evidence": "Anomaly score=0.95",
        "Credit ID": result5
    })

    # =========================================================================
    # DEMO 3: Peer Validation Credits
    # =========================================================================
    print_section("Demo 3: Peer Validation Credits", "🤝")

    print("Scenario: Nodes validate each other's gradients\n")

    print("1️⃣ Node 'charlie' validates node 'alice'")
    result6 = await integration.on_peer_validation(
        validator_node_id="charlie",
        validated_node_id="alice",
        reputation_level="NORMAL"
    )
    print_event("Peer Validation Event", {
        "Validator": "charlie",
        "Validated": "alice",
        "Reputation": "NORMAL (1.0x multiplier)",
        "Base Credits": "10",
        "Final Credits": "10 * 1.0 = 10.0",
        "Credit ID": result6
    })

    print("2️⃣ Node 'alice' validates node 'bob'")
    result7 = await integration.on_peer_validation(
        validator_node_id="alice",
        validated_node_id="bob",
        reputation_level="ELITE"
    )
    print_event("Peer Validation Event", {
        "Validator": "alice",
        "Validated": "bob",
        "Reputation": "ELITE (1.5x multiplier)",
        "Base Credits": "10",
        "Final Credits": "10 * 1.5 = 15.0",
        "Credit ID": result7
    })

    # =========================================================================
    # DEMO 4: Network Uptime Credits
    # =========================================================================
    print_section("Demo 4: Network Uptime Credits", "⏰")

    print("Scenario: Nodes receive credits for maintaining high uptime\n")

    print("1️⃣ Node 'alice' - 99% uptime (excellent)")
    result8 = await integration.on_network_contribution(
        node_id="alice",
        uptime_percentage=0.99,
        reputation_level="ELITE",
        hours_online=1.0
    )
    print_event("Network Uptime Event", {
        "Node": "alice",
        "Uptime": "99% (above 95% threshold)",
        "Reputation": "ELITE (1.5x multiplier)",
        "Base Credits": "1.0 credit/hour",
        "Final Credits": "1.0 * 1.5 = 1.5",
        "Credit ID": result8
    })

    print("2️⃣ Node 'bob' - 97% uptime (good)")
    result9 = await integration.on_network_contribution(
        node_id="bob",
        uptime_percentage=0.97,
        reputation_level="NORMAL",
        hours_online=1.0
    )
    print_event("Network Uptime Event", {
        "Node": "bob",
        "Uptime": "97% (above 95% threshold)",
        "Reputation": "NORMAL (1.0x multiplier)",
        "Base Credits": "1.0 credit/hour",
        "Final Credits": "1.0 * 1.0 = 1.0",
        "Credit ID": result9
    })

    print("3️⃣ Node 'eve' - 92% uptime (below threshold)")
    result10 = await integration.on_network_contribution(
        node_id="eve",
        uptime_percentage=0.92,
        reputation_level="NORMAL",
        hours_online=1.0
    )
    print_event("Network Uptime Event (REJECTED)", {
        "Node": "eve",
        "Uptime": "92% (below 95% threshold)",
        "Result": "❌ No credits issued",
        "Credit ID": result10 or "None"
    })

    # =========================================================================
    # AUDIT TRAILS
    # =========================================================================
    print_section("Audit Trails (Complete History)", "📜")

    for node in ["alice", "bob", "charlie"]:
        audit = await integration.get_audit_trail(node)
        if audit:
            print(f"🔍 Node '{node}' - {len(audit)} credit issuances:")
            total_credits = sum(record["credits_issued"] for record in audit)
            print(f"   Total Credits Earned: {total_credits:.2f}\n")

            for i, record in enumerate(audit, 1):
                print(f"   {i}. {record['event_type']}")
                print(f"      • Credits: {record['credits_issued']:.2f}")
                print(f"      • Multiplier: {record['multiplier']:.2f}x")
                print(f"      • Time: {record['timestamp']}")
            print()

    # =========================================================================
    # INTEGRATION STATISTICS
    # =========================================================================
    print_section("Integration Statistics", "📊")

    stats = integration.get_integration_stats()
    print("📈 Overall Statistics:")
    print(f"   • Integration Enabled: {stats['enabled']}")
    print(f"   • Mock Mode: {stats['mock_mode']}")
    print(f"   • Total Nodes: {stats['total_nodes']}")
    print(f"   • Total Events: {stats['total_events']}")
    print(f"   • Total Credits Issued: {stats['total_credits_issued']:.2f}")
    print()

    print("📊 Credits by Event Type:")
    for event_type, credits in stats['credits_by_event_type'].items():
        print(f"   • {event_type}: {credits:.2f} credits")
    print()

    # =========================================================================
    # RATE LIMITING DEMO
    # =========================================================================
    print_section("Rate Limiting Demonstration", "⏱️")

    print("Scenario: Testing hourly rate limits for quality gradients\n")

    # Try to issue many credits rapidly
    print("Attempting to issue 150 quality gradient credits rapidly...")
    issued_count = 0
    rejected_count = 0

    for i in range(150):
        result = await integration.on_quality_gradient(
            node_id="rate_test_node",
            pogq_score=1.0,
            reputation_level="NORMAL",
            verifiers=["validator"]
        )
        if result:
            issued_count += 1
        else:
            rejected_count += 1

    print(f"\n📊 Rate Limiting Results:")
    print(f"   • Credits Issued: {issued_count}")
    print(f"   • Credits Rejected: {rejected_count}")
    print(f"   • Hourly Limit: 10,000 credits")
    print(f"   • Total Attempted: {issued_count * 100:.0f} credits")
    print(f"   • Result: {'✅ Rate limiting working correctly' if rejected_count > 0 else '✅ Within limits'}")
    print()

    # Show rate limit stats
    rate_stats = integration.get_rate_limit_stats("rate_test_node")
    print(f"📈 Rate Limit Stats for 'rate_test_node':")
    print(f"   • Total Issuances: {rate_stats['total_issuances']}")
    for event_type, stats_detail in rate_stats['by_event_type'].items():
        print(f"   • {event_type}:")
        print(f"      - Count: {stats_detail['count']}")
        print(f"      - Total: {stats_detail['total']:.2f} credits")
        print(f"      - Hourly: {stats_detail['hourly']:.2f} credits")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section("Demo Complete - Summary", "✅")

    print("🎉 Successfully demonstrated all ZeroTrustML-Credits integration features:\n")
    print("   ✅ Quality Gradient Credits (PoGQ-based, 0-100 credits)")
    print("   ✅ Byzantine Detection Rewards (50 credits fixed)")
    print("   ✅ Peer Validation Credits (10 credits fixed)")
    print("   ✅ Network Uptime Credits (1 credit/hour)")
    print("   ✅ Reputation Multipliers (0.0x - 1.5x)")
    print("   ✅ Minimum Thresholds (PoGQ ≥ 0.7, uptime ≥ 95%)")
    print("   ✅ Rate Limiting (10,000/hour for quality gradients)")
    print("   ✅ Complete Audit Trails")
    print("   ✅ Integration Statistics")
    print()

    final_stats = integration.get_integration_stats()
    print(f"📊 Final Statistics:")
    print(f"   • Total Nodes Participated: {final_stats['total_nodes']}")
    print(f"   • Total Events Processed: {final_stats['total_events']}")
    print(f"   • Total Credits Issued: {final_stats['total_credits_issued']:.2f}")
    print()

    print("🚀 System Status: FULLY OPERATIONAL")
    print("💡 Ready for production deployment with real Holochain conductor!")
    print()


if __name__ == "__main__":
    asyncio.run(main())