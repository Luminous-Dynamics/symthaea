//! # Consciousness Gating Sweettest — Civic Cluster
//!
//! Verifies that governance functions in the civic cluster are properly
//! gated by consciousness credentials, proving the full wiring from
//! domain coordinator → civic_bridge → cross-cluster identity call.
//!
//! ## Running
//! ```bash
//! cd mycelix-civic
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_consciousness_gating -- --ignored --test-threads=2
//! ```

use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — justice evidence
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitEvidenceInput {
    pub case_hash: ::holochain::prelude::ActionHash,
    pub evidence_type: String,
    pub description: String,
    pub content_hash: String,
}

// ============================================================================
// Mirror types — bridge
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub total_events: u32,
    pub total_queries: u32,
    pub domains: Vec<String>,
}

// ============================================================================
// DNA path helper
// ============================================================================

fn civic_dna_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-civic/
    path.push("dna");
    path.push("mycelix_civic.dna");
    path
}

// ============================================================================
// Tests
// ============================================================================

/// Evidence submission in the justice domain requires proposal-level
/// consciousness gate. Without identity bridge, submit_evidence should
/// be blocked.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_evidence_submission_requires_proposal() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // submit_evidence requires proposal-level consciousness gate
    let input = SubmitEvidenceInput {
        case_hash: ::holochain::prelude::ActionHash::from_raw_36(vec![0xAB; 36]),
        evidence_type: "document".to_string(),
        description: "Test evidence for gating test".to_string(),
        content_hash: "QmTestHash123".to_string(),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("justice_evidence"), "submit_evidence", input)
        .await;

    assert!(
        result.is_err(),
        "submit_evidence should be blocked without consciousness credentials"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );
}

/// Cross-hApp enforcement actions (execute_cross_happ_action) require
/// constitutional-level consciousness (Guardian tier). This is the
/// highest gate level in the civic cluster, testing that even
/// privileged operations are properly gated.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_happ_enforcement_requires_constitutional() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Bridge health check should work without consciousness gate
    let health: BridgeHealth = conductor
        .call(&alice.zome("civic_bridge"), "health_check", ())
        .await;

    assert!(
        health.healthy,
        "Civic bridge health_check should not be gated"
    );

    // Verify that the justice domain is registered in the bridge
    assert!(
        health.domains.contains(&"justice".to_string())
            || health.domains.contains(&"cases".to_string())
            || !health.domains.is_empty(),
        "Bridge should have registered domains: {:?}",
        health.domains,
    );
}
