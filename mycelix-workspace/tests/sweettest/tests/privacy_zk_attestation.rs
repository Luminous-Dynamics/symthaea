//! Sweettest integration tests for LUCID Privacy Zome ZK Attestations
//!
//! Tests the zero-knowledge proof attestation functions:
//! - submit_proof_attestation
//! - get_subject_attestations
//! - get_attestations_by_type
//! - has_valid_attestation
//!
//! Updated for Holochain 0.6 sweettest API.

mod harness;

use harness::{setup_test_agents, wait_for_dht_sync, DnaPaths};
use holochain::prelude::*;
use serde::Serialize;
use serial_test::serial;

/// Input for submitting a ZK proof attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitAttestationInput {
    pub proof_type: String,
    pub proof_cid: String,
    pub public_inputs_hash: String,
    pub subject_hash: String,
    pub verified: bool,
    pub expires_at: Option<Timestamp>,
    pub metadata: Option<String>,
}

/// Input for checking valid attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidAttestationInput {
    pub subject_hash: String,
    pub proof_type: Option<String>,
}

// ZK Proof Attestation entry structure - used for deserialization when needed
// Note: Full deserialization tests omitted as entry bytes format may vary by hApp version

impl DnaPaths {
    pub fn lucid() -> std::path::PathBuf {
        Self::workspace_root().join("happs/lucid/lucid.dna")
    }
}

/// Test submitting a ZK proof attestation
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Run with: cargo test -- --ignored
async fn test_submit_proof_attestation() {
    let dna_path = DnaPaths::lucid();
    let agents = setup_test_agents(&dna_path, "lucid", 1).await;

    let agent = &agents[0];

    let input = SubmitAttestationInput {
        proof_type: "anonymous_belief".to_string(),
        proof_cid: "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".to_string(),
        public_inputs_hash: "abc123def456".to_string(),
        subject_hash: "belief_hash_001".to_string(),
        verified: true,
        expires_at: None,
        metadata: Some(r#"{"version":"1.0"}"#.to_string()),
    };

    let record: Record = agent
        .call_zome_fn("privacy", "submit_proof_attestation", input)
        .await;

    // Verify the attestation was created
    assert!(record.action().action_type() == ActionType::Create);

    // Verify the entry exists
    assert!(record.entry().as_option().is_some(), "Entry should exist");
}

/// Test retrieving attestations by subject
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_get_subject_attestations() {
    let dna_path = DnaPaths::lucid();
    let agents = setup_test_agents(&dna_path, "lucid", 1).await;

    let agent = &agents[0];
    let subject_hash = "test_subject_123".to_string();

    // Submit multiple attestations for the same subject
    for i in 0..3 {
        let input = SubmitAttestationInput {
            proof_type: "reputation_range".to_string(),
            proof_cid: format!("bafytest{}", i),
            public_inputs_hash: format!("hash{}", i),
            subject_hash: subject_hash.clone(),
            verified: true,
            expires_at: None,
            metadata: None,
        };

        let _: Record = agent
            .call_zome_fn("privacy", "submit_proof_attestation", input)
            .await;
    }

    wait_for_dht_sync().await;

    // Retrieve attestations
    let attestations: Vec<Record> = agent
        .call_zome_fn("privacy", "get_subject_attestations", subject_hash.clone())
        .await;

    assert_eq!(attestations.len(), 3);
}

/// Test retrieving attestations by proof type
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_get_attestations_by_type() {
    let dna_path = DnaPaths::lucid();
    let agents = setup_test_agents(&dna_path, "lucid", 1).await;

    let agent = &agents[0];

    // Submit attestations of different types
    let types = ["anonymous_belief", "reputation_range", "vote_eligibility"];
    for (i, proof_type) in types.iter().enumerate() {
        let input = SubmitAttestationInput {
            proof_type: proof_type.to_string(),
            proof_cid: format!("bafytype{}", i),
            public_inputs_hash: format!("typehash{}", i),
            subject_hash: format!("subject{}", i),
            verified: true,
            expires_at: None,
            metadata: None,
        };

        let _: Record = agent
            .call_zome_fn("privacy", "submit_proof_attestation", input)
            .await;
    }

    wait_for_dht_sync().await;

    // Get attestations by type
    let anon_attestations: Vec<Record> = agent
        .call_zome_fn(
            "privacy",
            "get_attestations_by_type",
            "anonymous_belief".to_string(),
        )
        .await;

    assert!(!anon_attestations.is_empty());
}

/// Test checking for valid attestation
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_has_valid_attestation() {
    let dna_path = DnaPaths::lucid();
    let agents = setup_test_agents(&dna_path, "lucid", 1).await;

    let agent = &agents[0];
    let subject_hash = "valid_subject_456".to_string();

    // Initially should have no valid attestation
    let input = ValidAttestationInput {
        subject_hash: subject_hash.clone(),
        proof_type: Some("anonymous_belief".to_string()),
    };

    let has_valid: bool = agent
        .call_zome_fn("privacy", "has_valid_attestation", input.clone())
        .await;

    assert!(!has_valid);

    // Submit a valid attestation
    let submit_input = SubmitAttestationInput {
        proof_type: "anonymous_belief".to_string(),
        proof_cid: "bafyvalidproof".to_string(),
        public_inputs_hash: "validhash".to_string(),
        subject_hash: subject_hash.clone(),
        verified: true,
        expires_at: None,
        metadata: None,
    };

    let _: Record = agent
        .call_zome_fn("privacy", "submit_proof_attestation", submit_input)
        .await;

    wait_for_dht_sync().await;

    // Now should have a valid attestation
    let has_valid: bool = agent
        .call_zome_fn("privacy", "has_valid_attestation", input)
        .await;

    assert!(has_valid);
}

/// Test attestation with expiration
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_attestation_expiration() {
    let dna_path = DnaPaths::lucid();
    let agents = setup_test_agents(&dna_path, "lucid", 1).await;

    let agent = &agents[0];
    let subject_hash = "expiring_subject".to_string();

    // Submit an already-expired attestation (timestamp in the past)
    let past_timestamp = Timestamp::from_micros(1000000); // Very old timestamp

    let input = SubmitAttestationInput {
        proof_type: "reputation_range".to_string(),
        proof_cid: "bafyexpired".to_string(),
        public_inputs_hash: "expiredhash".to_string(),
        subject_hash: subject_hash.clone(),
        verified: true,
        expires_at: Some(past_timestamp),
        metadata: None,
    };

    let _: Record = agent
        .call_zome_fn("privacy", "submit_proof_attestation", input)
        .await;

    wait_for_dht_sync().await;

    // Should not count as valid due to expiration
    let check_input = ValidAttestationInput {
        subject_hash: subject_hash.clone(),
        proof_type: Some("reputation_range".to_string()),
    };

    let has_valid: bool = agent
        .call_zome_fn("privacy", "has_valid_attestation", check_input)
        .await;

    assert!(!has_valid, "Expired attestation should not be valid");
}

/// Test multi-agent attestation verification
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_multi_agent_attestation() {
    let dna_path = DnaPaths::lucid();
    let agents = setup_test_agents(&dna_path, "lucid", 2).await;

    let subject_hash = "shared_subject".to_string();

    // Agent 1 submits an attestation
    let input = SubmitAttestationInput {
        proof_type: "anonymous_belief".to_string(),
        proof_cid: "bafyagent1proof".to_string(),
        public_inputs_hash: "agent1hash".to_string(),
        subject_hash: subject_hash.clone(),
        verified: true,
        expires_at: None,
        metadata: None,
    };

    let _: Record = agents[0]
        .call_zome_fn("privacy", "submit_proof_attestation", input)
        .await;

    wait_for_dht_sync().await;

    // Agent 2 should be able to retrieve the attestation
    let attestations: Vec<Record> = agents[1]
        .call_zome_fn("privacy", "get_subject_attestations", subject_hash.clone())
        .await;

    assert!(!attestations.is_empty(), "Agent 2 should see Agent 1's attestation");

    // Verify the record exists with an entry
    assert!(
        attestations[0].entry().as_option().is_some(),
        "Attestation should have entry content"
    );
}
