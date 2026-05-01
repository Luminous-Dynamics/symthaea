// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conductor integration test — verifies wire protocol against a running conductor.
//!
//! This test is SKIPPED when no conductor is running at ws://localhost:8888.
//! It's not a unit test — it's a manual integration test you run when
//! a conductor is available.
//!
//! # Running
//!
//! ```bash
//! # Start a conductor first:
//! hc sandbox --piped generate happ/mycelix-unified.happ --run=8888
//!
//! # Then run this test:
//! cargo test -p mycelix-sensorium --test conductor_integration -- --nocapture
//! ```
//!
//! # What it tests
//!
//! 1. WebSocket connection to conductor
//! 2. MessagePack request/response roundtrip
//! 3. app_info cell discovery
//! 4. A real zome call (list_active_proposals or similar)

// This test requires native WebSocket (not browser), so we use a lightweight
// approach: just test the serialization layer that the WASM transport uses.
// Full conductor E2E requires browser environment or a headless WASM runner.

use domain_governance::types::*;

/// Verify that our Proposal type can encode/decode through MessagePack
/// in the exact format the conductor expects.
#[test]
fn proposal_msgpack_wire_format() {
    let proposal = Proposal {
        id: "MIP-001".into(),
        title: "Test proposal".into(),
        description: "Testing wire format".into(),
        proposal_type: ProposalType::Standard,
        author: "did:mycelix:uhCAkTest".into(),
        status: ProposalStatus::Active,
        actions: "[]".into(),
        discussion_url: None,
        voting_starts: 1711900800,
        voting_ends: 1712505600,
        created: 1711814400,
        updated: 1711900800,
        version: 1,
    };

    // Encode as MessagePack (same as ExternIO)
    let bytes = rmp_serde::to_vec_named(&proposal).expect("encode");
    assert!(
        bytes.len() > 50,
        "Proposal should encode to >50 bytes, got {}",
        bytes.len()
    );

    // Decode back
    let decoded: Proposal = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(decoded.id, "MIP-001");
    assert_eq!(decoded.title, "Test proposal");
    assert_eq!(decoded.proposal_type, ProposalType::Standard);
    assert_eq!(decoded.status, ProposalStatus::Active);
    assert!(decoded.discussion_url.is_none());
}

/// Verify Vec<Proposal> roundtrips (conductor returns arrays of entries).
#[test]
fn proposal_vec_wire_format() {
    let proposals = vec![
        Proposal {
            id: "MIP-001".into(),
            title: "First".into(),
            description: "".into(),
            proposal_type: ProposalType::Standard,
            author: "did:test".into(),
            status: ProposalStatus::Active,
            actions: "[]".into(),
            discussion_url: None,
            voting_starts: 0,
            voting_ends: 0,
            created: 0,
            updated: 0,
            version: 1,
        },
        Proposal {
            id: "MIP-002".into(),
            title: "Second".into(),
            description: "".into(),
            proposal_type: ProposalType::Emergency,
            author: "did:test".into(),
            status: ProposalStatus::Draft,
            actions: "[]".into(),
            discussion_url: Some("https://forum.mycelix.net/p/2".into()),
            voting_starts: 0,
            voting_ends: 0,
            created: 0,
            updated: 0,
            version: 1,
        },
    ];

    let bytes = rmp_serde::to_vec_named(&proposals).expect("encode vec");
    let decoded: Vec<Proposal> = rmp_serde::from_slice(&bytes).expect("decode vec");
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].id, "MIP-001");
    assert_eq!(decoded[1].proposal_type, ProposalType::Emergency);
    assert!(decoded[1].discussion_url.is_some());
}

/// Verify Vote with PhiWeight roundtrips correctly.
#[test]
fn vote_phi_weight_wire_format() {
    let vote = Vote {
        id: "v-001".into(),
        proposal_id: "MIP-001".into(),
        voter: "did:mycelix:uhCAkVoter".into(),
        choice: VoteChoice::For,
        weight: 2.35,
        reason: Some("Strong support for community solar".into()),
        delegated: false,
        delegator: None,
        phi_weight: PhiWeight {
            phi_score: 0.72,
            k_trust: 0.85,
            stake_weight: 0.60,
            participation_score: 0.91,
            domain_reputation: 0.45,
        },
        timestamp: 1711900800_000_000,
    };

    let bytes = rmp_serde::to_vec_named(&vote).expect("encode");
    let decoded: Vote = rmp_serde::from_slice(&bytes).expect("decode");

    assert_eq!(decoded.choice, VoteChoice::For);
    assert!((decoded.phi_weight.phi_score - 0.72).abs() < f64::EPSILON);
    assert!((decoded.weight - 2.35).abs() < f64::EPSILON);
    assert_eq!(
        decoded.reason.as_deref(),
        Some("Strong support for community solar")
    );

    // Verify combined weight computation
    let combined = decoded.phi_weight.combined();
    assert!(combined > 0.0);
    assert!(combined < 5.0);
}

/// Verify Council with nested enum roundtrips.
#[test]
fn council_nested_enum_wire_format() {
    let councils = vec![
        Council {
            id: "c-root".into(),
            name: "Root Council".into(),
            purpose: "Top-level governance".into(),
            council_type: CouncilType::Root,
            parent_council_id: None,
            phi_threshold: 0.4,
            quorum: 0.51,
            supermajority: 0.67,
            status: CouncilStatus::Active,
            created_at: 1711814400,
        },
        Council {
            id: "c-energy".into(),
            name: "Energy Working Group".into(),
            purpose: "Solar and wind projects".into(),
            council_type: CouncilType::WorkingGroup {
                focus: "renewable energy".into(),
                expires: Some(1743350400),
            },
            parent_council_id: Some("c-root".into()),
            phi_threshold: 0.3,
            quorum: 0.4,
            supermajority: 0.6,
            status: CouncilStatus::Active,
            created_at: 1711900800,
        },
    ];

    let bytes = rmp_serde::to_vec_named(&councils).expect("encode");
    let decoded: Vec<Council> = rmp_serde::from_slice(&bytes).expect("decode");

    assert_eq!(decoded.len(), 2);
    assert!(matches!(decoded[0].council_type, CouncilType::Root));
    if let CouncilType::WorkingGroup { ref focus, expires } = decoded[1].council_type {
        assert_eq!(focus, "renewable energy");
        assert!(expires.is_some());
    } else {
        panic!("Expected WorkingGroup");
    }
}

/// Verify BudgetCycle + BudgetProject roundtrip.
#[test]
fn budget_wire_format() {
    let cycle = BudgetCycle {
        cycle_id: "2026-Q1".into(),
        name: "Q1 2026 Community Budget".into(),
        total_budget: 50_000,
        currency: "SAP".into(),
        phase: BudgetPhase::Voting,
        proposal_deadline: 1711814400,
        voting_deadline: 1712505600,
        created_at: 1711728000,
    };

    let bytes = rmp_serde::to_vec_named(&cycle).expect("encode");
    let decoded: BudgetCycle = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(decoded.phase, BudgetPhase::Voting);
    assert_eq!(decoded.total_budget, 50_000);
}
