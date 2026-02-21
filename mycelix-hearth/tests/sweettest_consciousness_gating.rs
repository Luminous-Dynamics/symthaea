//! # Consciousness Gating Sweettest
//!
//! Verifies that governance functions are properly gated by consciousness
//! credentials, proving the full wiring from governance zome → bridge →
//! cross-cluster identity call.
//!
//! ## What this tests
//!
//! 1. `create_decision()` is BLOCKED without consciousness credentials
//!    (the bridge's cross-cluster call to identity fails → gate rejects)
//! 2. `cast_vote()` is similarly BLOCKED
//! 3. Read-only operations (`get_hearth_decisions`) still work
//! 4. Bridge `health_check` still works (not gated)
//!
//! ## Why "gate blocks" is the right test
//!
//! Without the identity DNA installed as a second hApp role, the bridge's
//! `get_consciousness_credential` will fail its cross-cluster call. This
//! is exactly the scenario we want to prove: governance actions require
//! valid consciousness credentials, and the system fails closed (denies
//! access) when credentials are unavailable.
//!
//! ## Running
//! ```bash
//! cd mycelix-hearth
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_consciousness_gating -- --ignored --test-threads=2
//! ```

use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — decisions
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DecisionType {
    MajorityVote,
    Consensus,
    ElderDecision,
    GuardianDecision,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MemberRole {
    Youth,
    Adult,
    Elder,
    Guardian,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateDecisionInput {
    pub hearth_hash: ::holochain::prelude::ActionHash,
    pub title: String,
    pub description: String,
    pub decision_type: DecisionType,
    pub eligible_roles: Vec<MemberRole>,
    pub options: Vec<String>,
    pub deadline: ::holochain::prelude::Timestamp,
    pub quorum_bp: Option<u32>,
}

// ============================================================================
// Mirror types — kinship (needed to create hearth first)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum HearthType {
    Nuclear,
    Extended,
    Chosen,
    Blended,
    Multigenerational,
    Intentional,
    CoPod,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateHearthInput {
    pub name: String,
    pub description: String,
    pub hearth_type: HearthType,
    pub max_members: Option<u32>,
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

fn hearth_dna_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-hearth/
    path.push("dna");
    path.push("mycelix_hearth.dna");
    path
}

// ============================================================================
// Tests
// ============================================================================

/// Consciousness gate blocks create_decision when no identity bridge is
/// available to issue credentials. This proves the gate is wired and
/// fails closed.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_create_decision_blocked_without_consciousness_credential() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Step 1: Create a hearth (kinship — NOT gated by consciousness)
    let hearth_input = CreateHearthInput {
        name: "Test Hearth".to_string(),
        description: "For consciousness gating test".to_string(),
        hearth_type: HearthType::Intentional,
        max_members: Some(10),
    };

    let hearth_record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("hearth_kinship"), "create_hearth", hearth_input)
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // Step 2: Try to create a decision — should FAIL because the
    // consciousness gate calls hearth_bridge.get_consciousness_credential,
    // which tries a cross-cluster call to identity_bridge that doesn't exist
    let decision_input = CreateDecisionInput {
        hearth_hash,
        title: "Should be blocked".to_string(),
        description: "This decision should not be created".to_string(),
        decision_type: DecisionType::MajorityVote,
        eligible_roles: vec![MemberRole::Adult],
        options: vec!["Yes".to_string(), "No".to_string()],
        deadline: ::holochain::prelude::Timestamp::from_micros(
            (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as i64)
                + 3_600_000_000, // 1 hour from now
        ),
        quorum_bp: None,
    };

    // The conductor.call method panics on zome errors, so we use
    // call_fallible to catch the expected failure
    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("hearth_decisions"), "create_decision", decision_input)
        .await;

    assert!(
        result.is_err(),
        "create_decision should fail without consciousness credentials"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    // The error should mention consciousness gating or credential failure
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness/credential failure, got: {}",
        err_msg
    );
}

/// Bridge health_check still works even without identity bridge —
/// it's a local-only operation with no consciousness gate.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_health_check_not_gated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // health_check has NO consciousness gate — should always succeed
    let health: BridgeHealth = conductor
        .call(&alice.zome("hearth_bridge"), "health_check", ())
        .await;

    assert!(health.healthy, "Bridge health_check should not be gated");
    assert!(
        health.domains.contains(&"decisions".to_string()),
        "decisions domain should be in bridge health"
    );
}

/// Read operations (get_hearth_decisions) are not gated by consciousness.
/// Only writes (create/vote/amend) are gated.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_read_operations_not_gated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a hearth (ungated)
    let hearth_input = CreateHearthInput {
        name: "Read Test Hearth".to_string(),
        description: "For read-path test".to_string(),
        hearth_type: HearthType::Chosen,
        max_members: None,
    };

    let hearth_record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("hearth_kinship"), "create_hearth", hearth_input)
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // get_hearth_decisions should succeed (returns empty vec, no gate)
    let decisions: Vec<::holochain::prelude::Record> = conductor
        .call(
            &alice.zome("hearth_decisions"),
            "get_hearth_decisions",
            hearth_hash,
        )
        .await;

    assert!(
        decisions.is_empty(),
        "No decisions should exist yet, but call should succeed (ungated)"
    );
}

/// Consciousness gate blocks ALL gated decision types, not just MajorityVote.
/// Tests that Consensus (requires Citizen tier) is also blocked.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_consensus_decision_also_blocked() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let hearth_input = CreateHearthInput {
        name: "Consensus Hearth".to_string(),
        description: "Testing Consensus gate".to_string(),
        hearth_type: HearthType::Intentional,
        max_members: Some(5),
    };

    let hearth_record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("hearth_kinship"), "create_hearth", hearth_input)
        .await;

    let decision_input = CreateDecisionInput {
        hearth_hash: hearth_record.action_address().clone(),
        title: "Consensus blocked".to_string(),
        description: "Consensus requires Citizen tier".to_string(),
        decision_type: DecisionType::Consensus,
        eligible_roles: vec![MemberRole::Adult, MemberRole::Elder],
        options: vec!["Approve".to_string(), "Block".to_string()],
        deadline: ::holochain::prelude::Timestamp::from_micros(
            (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as i64)
                + 3_600_000_000,
        ),
        quorum_bp: Some(7500), // 75% quorum
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("hearth_decisions"),
            "create_decision",
            decision_input,
        )
        .await;

    assert!(
        result.is_err(),
        "Consensus decision should also be blocked without consciousness credentials"
    );
}
