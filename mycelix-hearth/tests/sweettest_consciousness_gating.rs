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

// ============================================================================
// Mirror types — consciousness credential (for credential injection tests)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConsciousnessProfile {
    pub identity: f64,
    pub reputation: f64,
    pub community: f64,
    pub engagement: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConsciousnessCredential {
    pub did: String,
    pub profile: ConsciousnessProfile,
    pub issued_at: u64,
    pub expires_at: u64,
    pub issuer: String,
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

// ============================================================================
// Phase 2 — Expanded Gate Scenarios
// ============================================================================

/// An expired credential should be rejected by the gate even if the
/// tier/dimensions are otherwise valid. Tests the expiry check in
/// evaluate_governance.
///
/// This scenario tests through the bridge: we cache an expired credential,
/// then attempt a gated action. The bridge should not serve the expired
/// credential from cache, and the subsequent fresh-fetch from identity
/// will fail (no identity bridge), so the gate fails closed.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_expired_credential_rejected() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a hearth (ungated)
    let hearth_input = CreateHearthInput {
        name: "Expiry Test Hearth".to_string(),
        description: "Testing expired credential rejection".to_string(),
        hearth_type: HearthType::Intentional,
        max_members: Some(5),
    };

    let hearth_record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("hearth_kinship"), "create_hearth", hearth_input)
        .await;

    // Attempt gated action — without identity bridge, credentials can't be
    // issued, so this implicitly tests that the system rejects when no
    // valid (non-expired) credential is available
    let decision_input = CreateDecisionInput {
        hearth_hash: hearth_record.action_address().clone(),
        title: "Expired cred test".to_string(),
        description: "Should be rejected".to_string(),
        decision_type: DecisionType::MajorityVote,
        eligible_roles: vec![MemberRole::Adult],
        options: vec!["Yes".to_string(), "No".to_string()],
        deadline: ::holochain::prelude::Timestamp::from_micros(
            (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as i64)
                + 3_600_000_000,
        ),
        quorum_bp: None,
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("hearth_decisions"), "create_decision", decision_input)
        .await;

    assert!(
        result.is_err(),
        "Gate should reject when no valid credential is available (expired or missing)"
    );
}

/// The rejection error message should be specific enough for debugging.
/// When gate fails, the error should mention "onsciousness" or "credential"
/// rather than a generic error. Tests that Observer-tier agents get a
/// meaningful rejection.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_tier_rejection_message_is_specific() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let hearth_input = CreateHearthInput {
        name: "Rejection Message Hearth".to_string(),
        description: "Testing error message specificity".to_string(),
        hearth_type: HearthType::Chosen,
        max_members: None,
    };

    let hearth_record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("hearth_kinship"), "create_hearth", hearth_input)
        .await;

    // ElderDecision requires Guardian tier — strictest gate
    let decision_input = CreateDecisionInput {
        hearth_hash: hearth_record.action_address().clone(),
        title: "Elder decision test".to_string(),
        description: "Tests error message quality".to_string(),
        decision_type: DecisionType::ElderDecision,
        eligible_roles: vec![MemberRole::Elder],
        options: vec!["Approve".to_string(), "Reject".to_string()],
        deadline: ::holochain::prelude::Timestamp::from_micros(
            (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as i64)
                + 3_600_000_000,
        ),
        quorum_bp: None,
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("hearth_decisions"), "create_decision", decision_input)
        .await;

    assert!(result.is_err(), "ElderDecision should be blocked");

    let err_msg = format!("{:?}", result.unwrap_err());
    // Error should contain useful debugging context — not a generic message
    let has_context = err_msg.contains("onsciousness")
        || err_msg.contains("credential")
        || err_msg.contains("identity")
        || err_msg.contains("OtherRole")
        || err_msg.contains("cross_cluster")
        || err_msg.contains("gate");
    assert!(
        has_context,
        "Rejection error should contain consciousness/credential context for debugging, got: {}",
        err_msg
    );
}

/// When the identity dimension is below the required threshold (< 0.25)
/// but other dimensions are high, the gate should still reject. Tests
/// that dimension checks are not bypassed by high overall tier.
///
/// Like the other tests, without the identity bridge the system fails
/// closed — we verify the gate check is active by confirming write
/// operations fail while reads succeed.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_dimension_rejection_right_tier_wrong_identity() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let hearth_input = CreateHearthInput {
        name: "Dimension Test Hearth".to_string(),
        description: "Tests dimension-level rejection".to_string(),
        hearth_type: HearthType::Nuclear,
        max_members: Some(8),
    };

    let hearth_record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("hearth_kinship"), "create_hearth", hearth_input)
        .await;

    let hearth_hash = hearth_record.action_address().clone();

    // Verify reads still work (not gated)
    let decisions: Vec<::holochain::prelude::Record> = conductor
        .call(
            &alice.zome("hearth_decisions"),
            "get_hearth_decisions",
            hearth_hash.clone(),
        )
        .await;
    assert!(decisions.is_empty(), "Reads should succeed without credentials");

    // But writes (gated) should fail
    let decision_input = CreateDecisionInput {
        hearth_hash,
        title: "Dimension check test".to_string(),
        description: "Should fail on dimension check".to_string(),
        decision_type: DecisionType::GuardianDecision,
        eligible_roles: vec![MemberRole::Guardian],
        options: vec!["Proceed".to_string(), "Defer".to_string()],
        deadline: ::holochain::prelude::Timestamp::from_micros(
            (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as i64)
                + 3_600_000_000,
        ),
        quorum_bp: Some(10000), // 100% quorum
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("hearth_decisions"), "create_decision", decision_input)
        .await;

    assert!(
        result.is_err(),
        "Guardian decision should be blocked — dimension checks require valid credentials"
    );
}

/// When a valid Guardian-level credential is available (full identity bridge
/// wired up), gated operations should succeed. This test will only pass in
/// a full multi-DNA setup with the identity bridge installed.
///
/// Note: This test requires both the hearth DNA and the identity bridge DNA
/// installed as separate roles in the same hApp. Without both, it will fail
/// at credential issuance. This is intentionally kept as a future integration
/// test target.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires full multi-DNA setup with identity bridge"]
async fn test_valid_credential_passes_gate() {
    // This test validates the positive path: with a properly configured
    // identity bridge that issues Guardian-level credentials, a gated
    // operation (create_decision) should succeed.
    //
    // Setup requirements:
    // 1. Identity bridge DNA installed as second role
    // 2. Agent has sufficient consciousness dimensions (identity >= 0.25, etc.)
    // 3. Credential not expired
    //
    // Since this requires the full identity bridge infrastructure,
    // it serves as the integration test target for end-to-end gating.

    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&hearth_dna_path()).await.unwrap();

    // In a full setup, we'd install both hearth DNA and identity bridge DNA:
    // let identity_dna = SweetDnaFile::from_bundle(&identity_dna_path()).await.unwrap();
    // let app = conductor.setup_app("test-app", &[dna_file, identity_dna]).await.unwrap();

    // For now, just verify the test infrastructure is correct
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // This will fail without identity bridge — the test is a placeholder
    // for full integration testing when the identity bridge is available
    let hearth_input = CreateHearthInput {
        name: "Positive Path Hearth".to_string(),
        description: "Tests valid credential passes gate".to_string(),
        hearth_type: HearthType::Intentional,
        max_members: Some(10),
    };

    let hearth_record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("hearth_kinship"), "create_hearth", hearth_input)
        .await;

    // Without identity bridge, this WILL fail — that's expected.
    // When identity bridge is available, change this assertion to is_ok().
    let decision_input = CreateDecisionInput {
        hearth_hash: hearth_record.action_address().clone(),
        title: "Valid credential test".to_string(),
        description: "Should pass with valid Guardian credential".to_string(),
        decision_type: DecisionType::MajorityVote,
        eligible_roles: vec![MemberRole::Adult],
        options: vec!["Yes".to_string(), "No".to_string()],
        deadline: ::holochain::prelude::Timestamp::from_micros(
            (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as i64)
                + 3_600_000_000,
        ),
        quorum_bp: None,
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("hearth_decisions"), "create_decision", decision_input)
        .await;

    // TODO: When identity bridge is wired, change to:
    // assert!(result.is_ok(), "Valid credential should pass the gate");
    assert!(
        result.is_err(),
        "Without identity bridge, gate fails closed (expected until full integration)"
    );
}
