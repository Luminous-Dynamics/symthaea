// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Conductor Integration Tests
//!
//! Tests the end-to-end flow: CLS → governance dispatch → conductor adapter → outcome.
//!
//! ## Running
//!
//! Mock variant (no conductor required):
//! ```bash
//! cargo test --test mycelix_conductor_integration
//! ```
//!
//! Real variant (requires running Holochain conductor with mycelix-unified hApp):
//! ```bash
//! MYCELIX_CONDUCTOR_URL=ws://localhost:8888 \
//! MYCELIX_APP_TOKEN=... \
//! MYCELIX_APP_ID=mycelix-unified \
//! cargo test --test mycelix_conductor_integration -- --ignored
//! ```

use symthaea_mycelix_conductor::{
    ConductorConfig, ConductorTransport, DispatchCommand, DispatchOutcome, GovernanceDispatcher,
    MockTransport,
};

// ============================================================================
// Mock-conductor tests (always runnable)
// ============================================================================

#[tokio::test]
async fn test_mock_proposal_dispatch_roundtrip() {
    let mut dispatcher = GovernanceDispatcher::new(MockTransport);

    let cmd = DispatchCommand::SubmitProposal {
        correlation_id: 1,
        description: "Fund community garden project".into(),
        proposer_did: "did:mycelix:alice".into(),
        consciousness_phi: 0.85,
        meta_awareness: 0.6,
        coherence: 0.7,
        care_activation: 0.5,
        alignment_score: 0.92,
    };

    let outcome = dispatcher.dispatch(cmd).await;
    match outcome {
        DispatchOutcome::ProposalAccepted { correlation_id, .. } => {
            assert_eq!(correlation_id, 1);
        }
        other => panic!("Expected ProposalAccepted, got {:?}", other),
    }
}

#[tokio::test]
async fn test_mock_vote_dispatch_roundtrip() {
    let mut dispatcher = GovernanceDispatcher::new(MockTransport);

    let cmd = DispatchCommand::CastVote {
        correlation_id: 2,
        proposal_id: "uhCkk_proposal_123".into(),
        voter_did: "did:mycelix:bob".into(),
        approve: true,
        rationale: "Aligns with ecological wisdom harmony".into(),
        consciousness_phi: 0.8,
        meta_awareness: 0.6,
        coherence: 0.7,
        care_activation: 0.5,
    };

    let outcome = dispatcher.dispatch(cmd).await;
    match outcome {
        DispatchOutcome::VoteAccepted { correlation_id, .. } => {
            assert_eq!(correlation_id, 2);
        }
        other => panic!("Expected VoteAccepted, got {:?}", other),
    }
}

#[tokio::test]
async fn test_mock_dispatch_loop_processes_multiple_commands() {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::sync_channel(10);
    let (outcome_tx, mut outcome_rx) = tokio::sync::mpsc::channel(10);

    let dispatcher = GovernanceDispatcher::new(MockTransport);

    // Send 3 commands
    cmd_tx
        .send(DispatchCommand::SubmitProposal {
            correlation_id: 10,
            description: "Proposal A".into(),
            proposer_did: "did:test:a".into(),
            consciousness_phi: 0.8,
            meta_awareness: 0.5,
            coherence: 0.6,
            care_activation: 0.4,
            alignment_score: 0.9,
        })
        .unwrap();
    cmd_tx
        .send(DispatchCommand::CastVote {
            correlation_id: 11,
            proposal_id: "prop_a".into(),
            voter_did: "did:test:b".into(),
            approve: false,
            rationale: "Insufficient alignment".into(),
            consciousness_phi: 0.7,
            meta_awareness: 0.5,
            coherence: 0.6,
            care_activation: 0.4,
        })
        .unwrap();
    cmd_tx
        .send(DispatchCommand::SubmitProposal {
            correlation_id: 12,
            description: "Proposal B".into(),
            proposer_did: "did:test:c".into(),
            consciousness_phi: 0.9,
            meta_awareness: 0.6,
            coherence: 0.7,
            care_activation: 0.5,
            alignment_score: 0.95,
        })
        .unwrap();

    // Drop sender so the loop will eventually run out of commands
    drop(cmd_tx);

    // Run loop in background
    let handle = tokio::spawn(async move {
        dispatcher.run_dispatch_loop(cmd_rx, outcome_tx).await;
    });

    // Collect 3 outcomes
    let mut outcomes = Vec::new();
    for _ in 0..3 {
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), outcome_rx.recv())
            .await
            .expect("should receive within timeout")
            .expect("channel should not be closed");
        outcomes.push(outcome);
    }

    assert_eq!(outcomes.len(), 3, "Should receive 3 outcomes");

    // Verify correlation IDs match
    let corr_ids: Vec<u64> = outcomes
        .iter()
        .map(|o| match o {
            DispatchOutcome::ProposalAccepted { correlation_id, .. } => *correlation_id,
            DispatchOutcome::VoteAccepted { correlation_id, .. } => *correlation_id,
            other => panic!("Unexpected outcome: {:?}", other),
        })
        .collect();

    assert!(corr_ids.contains(&10));
    assert!(corr_ids.contains(&11));
    assert!(corr_ids.contains(&12));

    handle.abort();
}

#[tokio::test]
async fn test_failing_transport_returns_rejection() {
    struct FailTransport;

    #[async_trait::async_trait]
    impl ConductorTransport for FailTransport {
        async fn call_zome(
            &mut self,
            _: &str,
            _: &str,
            _: &str,
            _: Vec<u8>,
        ) -> Result<Vec<u8>, String> {
            Err("conductor offline".into())
        }
        fn is_connected(&self) -> bool {
            false
        }
    }

    let mut dispatcher = GovernanceDispatcher::new(FailTransport);

    let cmd = DispatchCommand::SubmitProposal {
        correlation_id: 99,
        description: "Should fail".into(),
        proposer_did: "did:test".into(),
        consciousness_phi: 0.5,
        meta_awareness: 0.4,
        coherence: 0.3,
        care_activation: 0.4,
        alignment_score: 0.5,
    };

    let outcome = dispatcher.dispatch(cmd).await;
    match outcome {
        DispatchOutcome::ProposalRejected {
            correlation_id,
            reason,
        } => {
            assert_eq!(correlation_id, 99);
            assert!(reason.contains("offline"));
        }
        other => panic!("Expected ProposalRejected, got {:?}", other),
    }
}

// ============================================================================
// Real-conductor tests (require running Holochain)
// ============================================================================

#[tokio::test]
#[ignore = "requires running Holochain conductor with mycelix-unified hApp"]
async fn test_real_conductor_proposal_submission() {
    let config = ConductorConfig::from_env()
        .expect("Set MYCELIX_CONDUCTOR_URL, MYCELIX_APP_TOKEN, MYCELIX_APP_ID env vars");

    // This test would use a real transport implementation.
    // For now, just verify config parsing works.
    assert!(!config.url.is_empty());
    assert!(!config.app_id.is_empty());
}