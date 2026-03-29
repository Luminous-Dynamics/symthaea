// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Attestation E2E Sweettest Integration Tests
//!
//! Tests the full consciousness attestation flow:
//!   1. Record a ConsciousnessAttestation via governance bridge
//!   2. Verify consciousness gate using attested consciousness level (v2 API)
//!   3. Validate provenance tracking (Attested vs Unavailable)
//!   4. Cross-cluster: personal bridge submits attestation to governance
//!
//! ## Prerequisites
//!
//! ```bash
//! # Build governance WASM zomes
//! cd mycelix-governance && cargo build --release --target wasm32-unknown-unknown
//! hc dna pack mycelix-governance/dna/
//!
//! # Build personal WASM zomes
//! cd mycelix-personal && cargo build --release --target wasm32-unknown-unknown
//! hc dna pack mycelix-personal/dna/
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo test --release -p mycelix-sweettest --test phi_attestation_e2e -- --ignored --test-threads=1
//! ```

mod harness;

use harness::*;
use holochain::prelude::*;
use holochain::sweettest::*;
use serial_test::serial;

// ============================================================================
// Mirror types — avoid WASM symbol conflicts by re-defining structs
// ============================================================================

/// Input for recording a consciousness attestation (mirrors governance_bridge coordinator).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RecordConsciousnessAttestationInput {
    consciousness_level: f64,
    cycle_id: u64,
    captured_at_us: u64,
    signature: Vec<u8>,
}

/// Input for verifying a consciousness gate (v2 API).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct VerifyGateInput {
    action_type: GovernanceActionType,
}

/// Governance action types requiring different Phi thresholds.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum GovernanceActionType {
    Basic,
    ProposalSubmission,
    Voting,
    Constitutional,
}

/// Gate verification result with provenance tracking (v2).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct GateVerificationResultV2 {
    passed: bool,
    consciousness_level: Option<f64>,
    required_consciousness: f64,
    provenance: ConsciousnessProvenance,
    action_type: GovernanceActionType,
    failure_reason: Option<String>,
}

/// How the consciousness value was obtained.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
enum ConsciousnessProvenance {
    Attested,
    Snapshot,
    Unavailable,
}

// ============================================================================
// Single-DNA governance setup
// ============================================================================

/// Setup a single agent with governance DNA for direct bridge testing.
async fn setup_governance_agent() -> TestAgent {
    let agents = setup_test_agents(&DnaPaths::governance(), "phi-attestation-test", 1).await;
    agents.into_iter().next().unwrap()
}

// ============================================================================
// Dual-DNA setup (personal + governance) for cross-cluster tests
// ============================================================================

struct ConsciousnessTestAgent {
    conductor: SweetConductor,
    personal_cell: SweetCell,
    governance_cell: SweetCell,
}

impl ConsciousnessTestAgent {
    async fn call_personal<I, O>(&self, zome_name: &str, fn_name: &str, input: I) -> O
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let zome = self.personal_cell.zome(zome_name);
        self.conductor.call(&zome, fn_name, input).await
    }

    async fn call_governance<I, O>(&self, zome_name: &str, fn_name: &str, input: I) -> O
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let zome = self.governance_cell.zome(zome_name);
        self.conductor.call(&zome, fn_name, input).await
    }
}

async fn setup_consciousness_test_conductor() -> ConsciousnessTestAgent {
    let personal_dna = SweetDnaFile::from_bundle(&DnaPaths::personal())
        .await
        .expect("Personal DNA required — run: hc dna pack mycelix-personal/dna/");

    let governance_dna = SweetDnaFile::from_bundle(&DnaPaths::governance())
        .await
        .expect("Governance DNA required — run: hc dna pack mycelix-governance/dna/");

    let mut conductor = SweetConductor::from_standard_config().await;

    // Install both DNAs so OtherRole("governance") dispatch works from personal
    let app = conductor
        .setup_app("phi-attestation-e2e", &[personal_dna, governance_dna])
        .await
        .unwrap();

    let cells = app.into_cells();

    ConsciousnessTestAgent {
        conductor,
        personal_cell: cells[0].clone(),
        governance_cell: cells[1].clone(),
    }
}

// ============================================================================
// Tests — Governance Bridge Direct
// ============================================================================

/// Test: Record a Phi attestation and verify it was stored.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_record_consciousness_attestation() {
    let agent = setup_governance_agent().await;

    // Build attestation message matching the canonical format
    let agent_did = format!("did:mycelix:{}", agent.agent_pubkey);
    let phi = 0.75f64;
    let cycle_id = 42u64;
    let captured_at_us = 1708363200000000u64;

    let message = format!(
        "symthaea-consciousness-attestation:v1:{}:{:.6}:{}:{}",
        agent_did, phi, cycle_id, captured_at_us,
    );

    // Sign with agent's Holochain key
    let signature: Signature = agent
        .conductor
        .call(
            &agent.zome("governance_bridge"),
            "sign_data",
            message.clone().into_bytes(),
        )
        .await;

    let input = RecordConsciousnessAttestationInput {
        consciousness_level: phi,
        cycle_id,
        captured_at_us,
        signature: signature.0.to_vec(),
    };

    let result: Record = agent
        .call_zome_fn("governance_bridge", "record_consciousness_attestation", input)
        .await;

    assert!(
        !result.action_hashed().hash.as_ref().is_empty(),
        "ConsciousnessAttestation record should have valid action hash"
    );
}

/// Test: Verify consciousness gate with attested Phi (should pass for Voting threshold).
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_verify_gate_with_attestation() {
    let agent = setup_governance_agent().await;

    // First record an attestation with Phi = 0.75
    let agent_did = format!("did:mycelix:{}", agent.agent_pubkey);
    let phi = 0.75f64;
    let cycle_id = 100u64;
    let captured_at_us = 1708363200000000u64;

    let message = format!(
        "symthaea-consciousness-attestation:v1:{}:{:.6}:{}:{}",
        agent_did, phi, cycle_id, captured_at_us,
    );

    let signature: Signature = agent
        .conductor
        .call(
            &agent.zome("governance_bridge"),
            "sign_data",
            message.into_bytes(),
        )
        .await;

    let _: Record = agent
        .call_zome_fn(
            "governance_bridge",
            "record_consciousness_attestation",
            RecordConsciousnessAttestationInput {
                consciousness_level: phi,
                cycle_id,
                captured_at_us,
                signature: signature.0.to_vec(),
            },
        )
        .await;

    wait_for_dht_sync().await;

    // Now verify consciousness gate for Voting (threshold = 0.4)
    let gate_result: GateVerificationResultV2 = agent
        .call_zome_fn(
            "governance_bridge",
            "verify_consciousness_gate_v2",
            VerifyGateInput {
                action_type: GovernanceActionType::Voting,
            },
        )
        .await;

    assert!(gate_result.passed, "Phi 0.75 should pass Voting threshold 0.4");
    assert_eq!(gate_result.consciousness_level, Some(phi));
    assert_eq!(gate_result.provenance, ConsciousnessProvenance::Attested);
    assert!(gate_result.failure_reason.is_none());
}

/// Test: Verify consciousness gate without any attestation returns Unavailable.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_verify_gate_no_attestation() {
    let agent = setup_governance_agent().await;

    // No attestation recorded — gate should report Unavailable
    let gate_result: GateVerificationResultV2 = agent
        .call_zome_fn(
            "governance_bridge",
            "verify_consciousness_gate_v2",
            VerifyGateInput {
                action_type: GovernanceActionType::Voting,
            },
        )
        .await;

    assert!(!gate_result.passed, "Should fail without attestation data");
    assert_eq!(gate_result.consciousness_level, None);
    assert_eq!(gate_result.provenance, ConsciousnessProvenance::Unavailable);
    assert!(gate_result.failure_reason.is_some());
}

/// Test: Phi below threshold fails the gate even when attested.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_verify_gate_insufficient_consciousness() {
    let agent = setup_governance_agent().await;

    // Record attestation with low Phi = 0.15
    let agent_did = format!("did:mycelix:{}", agent.agent_pubkey);
    let phi = 0.15f64;
    let cycle_id = 200u64;
    let captured_at_us = 1708363200000000u64;

    let message = format!(
        "symthaea-consciousness-attestation:v1:{}:{:.6}:{}:{}",
        agent_did, phi, cycle_id, captured_at_us,
    );

    let signature: Signature = agent
        .conductor
        .call(
            &agent.zome("governance_bridge"),
            "sign_data",
            message.into_bytes(),
        )
        .await;

    let _: Record = agent
        .call_zome_fn(
            "governance_bridge",
            "record_consciousness_attestation",
            RecordConsciousnessAttestationInput {
                consciousness_level: phi,
                cycle_id,
                captured_at_us,
                signature: signature.0.to_vec(),
            },
        )
        .await;

    wait_for_dht_sync().await;

    // Voting requires 0.4 — should fail
    let gate_result: GateVerificationResultV2 = agent
        .call_zome_fn(
            "governance_bridge",
            "verify_consciousness_gate_v2",
            VerifyGateInput {
                action_type: GovernanceActionType::Voting,
            },
        )
        .await;

    assert!(!gate_result.passed, "Phi 0.15 should fail Voting threshold 0.4");
    assert_eq!(gate_result.consciousness_level, Some(phi));
    assert_eq!(gate_result.provenance, ConsciousnessProvenance::Attested);
    assert!(gate_result.failure_reason.is_some());
}

// ============================================================================
// Tests — Cross-Cluster (Personal → Governance)
// ============================================================================

/// Test: Submit Phi attestation from personal bridge to governance.
/// This exercises the full cross-cluster dispatch path.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires both DNA bundles
async fn test_cross_cluster_consciousness_attestation() {
    let agent = setup_consciousness_test_conductor().await;

    // Input for submit_consciousness_attestation on personal bridge
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct SubmitConsciousnessAttestationInput {
        consciousness_level: f64,
        cycle_id: u64,
    }

    // Submit via personal bridge (which signs and dispatches to governance)
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct DispatchResult {
        success: bool,
        response: Option<Vec<u8>>,
        error: Option<String>,
    }

    let input = SubmitConsciousnessAttestationInput {
        consciousness_level: 0.82,
        cycle_id: 500,
    };

    let result: DispatchResult = agent
        .call_personal("personal_bridge", "submit_consciousness_attestation", input)
        .await;

    assert!(
        result.success,
        "Cross-cluster attestation should succeed: {:?}",
        result.error
    );

    wait_for_dht_sync().await;

    // Verify the attestation landed in governance
    let gate_result: GateVerificationResultV2 = agent
        .call_governance(
            "governance_bridge",
            "verify_consciousness_gate_v2",
            VerifyGateInput {
                action_type: GovernanceActionType::Constitutional,
            },
        )
        .await;

    assert!(
        gate_result.passed,
        "Phi 0.82 should pass Constitutional threshold 0.6"
    );
    assert_eq!(gate_result.provenance, ConsciousnessProvenance::Attested);
}

/// Test: Multi-agent attestation — two agents, different Phi values.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_multi_agent_attestation() {
    let governance_dna = SweetDnaFile::from_bundle(&DnaPaths::governance())
        .await
        .expect("Governance DNA required");

    let mut conductor1 = SweetConductor::from_standard_config().await;
    let mut conductor2 = SweetConductor::from_standard_config().await;

    let app1 = conductor1
        .setup_app("phi-multi-1", &[governance_dna.clone()])
        .await
        .unwrap();
    let app2 = conductor2
        .setup_app("phi-multi-2", &[governance_dna])
        .await
        .unwrap();

    let cell1 = app1.cells()[0].clone();
    let cell2 = app2.cells()[0].clone();

    // Connect for DHT gossip
    SweetConductor::exchange_peer_info(vec![&conductor1, &conductor2]).await;

    // Agent 1: high Phi
    let agent1_did = format!("did:mycelix:{}", cell1.agent_pubkey());
    let msg1 = format!(
        "symthaea-consciousness-attestation:v1:{}:{:.6}:{}:{}",
        agent1_did, 0.9f64, 1000u64, 1708363200000000u64,
    );
    let sig1: Signature = conductor1
        .call(&cell1.zome("governance_bridge"), "sign_data", msg1.into_bytes())
        .await;

    let _: Record = conductor1
        .call(
            &cell1.zome("governance_bridge"),
            "record_consciousness_attestation",
            RecordConsciousnessAttestationInput {
                consciousness_level: 0.9,
                cycle_id: 1000,
                captured_at_us: 1708363200000000,
                signature: sig1.0.to_vec(),
            },
        )
        .await;

    // Agent 2: low consciousness
    let agent2_did = format!("did:mycelix:{}", cell2.agent_pubkey());
    let msg2 = format!(
        "symthaea-consciousness-attestation:v1:{}:{:.6}:{}:{}",
        agent2_did, 0.1f64, 1001u64, 1708363200000000u64,
    );
    let sig2: Signature = conductor2
        .call(&cell2.zome("governance_bridge"), "sign_data", msg2.into_bytes())
        .await;

    let _: Record = conductor2
        .call(
            &cell2.zome("governance_bridge"),
            "record_consciousness_attestation",
            RecordConsciousnessAttestationInput {
                consciousness_level: 0.1,
                cycle_id: 1001,
                captured_at_us: 1708363200000000,
                signature: sig2.0.to_vec(),
            },
        )
        .await;

    wait_for_dht_sync().await;

    // Agent 1 passes Constitutional gate
    let gate1: GateVerificationResultV2 = conductor1
        .call(
            &cell1.zome("governance_bridge"),
            "verify_consciousness_gate_v2",
            VerifyGateInput {
                action_type: GovernanceActionType::Constitutional,
            },
        )
        .await;
    assert!(gate1.passed, "Agent 1 (Phi=0.9) should pass Constitutional");

    // Agent 2 fails even Basic gate
    let gate2: GateVerificationResultV2 = conductor2
        .call(
            &cell2.zome("governance_bridge"),
            "verify_consciousness_gate_v2",
            VerifyGateInput {
                action_type: GovernanceActionType::Basic,
            },
        )
        .await;
    assert!(!gate2.passed, "Agent 2 (Phi=0.1) should fail Basic threshold 0.2");
}
