//! # Consciousness Gating Sweettest — Finance Cluster
//!
//! Verifies that financial operations in the finance cluster are properly
//! gated by DID identity verification and MYCEL-tier consciousness checks,
//! proving the full wiring from domain coordinator → finance_bridge →
//! cross-zome/cross-cluster identity validation.
//!
//! ## What this tests
//!
//! 1. `process_payment` is BLOCKED when caller DID doesn't match (identity gate)
//! 2. `deposit_collateral` is BLOCKED when caller DID doesn't match
//! 3. `register_collateral` is BLOCKED when caller DID doesn't match
//! 4. `governance_mint_sap` requires governance authorization + MYCEL tier gate
//! 5. Read-only operations (`get_payment_history`, `query_sap_balance`,
//!    `get_member_fee_tier`, `health_check`) succeed without credentials
//! 6. Bridge health remains stable after gate rejections
//! 7. Cache consistency — repeated failures don't produce false success
//!
//! ## Why DID verification is the consciousness gate for finance
//!
//! The finance cluster gates write operations through `verify_caller_is_did()`,
//! which ensures the calling agent matches the claimed DID. This prevents
//! identity spoofing in financial transactions. Higher-tier operations
//! (SAP minting) additionally check MYCEL score via the finance bridge,
//! implementing progressive consciousness gating.
//!
//! ## Running
//! ```bash
//! cd mycelix-finance
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_consciousness_gating -- --ignored --test-threads=2
//! ```

use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — bridge
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FinanceBridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub zomes: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProcessPaymentInput {
    pub source_happ: String,
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub currency: String,
    pub reference: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DepositCollateralInput {
    pub depositor_did: String,
    pub collateral_type: String,
    pub collateral_amount: u64,
    pub oracle_rate: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterCollateralInput {
    pub owner_did: String,
    pub source_happ: String,
    pub asset_type: AssetType,
    pub asset_id: String,
    pub value_estimate: u64,
    pub currency: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AssetType {
    RealEstate,
    Vehicle,
    Cryptocurrency,
    EnergyAsset,
    Equipment,
    Other(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetPaymentHistoryInput {
    pub did: String,
    pub limit: Option<usize>,
}

// ============================================================================
// Mirror types — payments
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MintSapFromGovernanceInput {
    pub recipient_did: String,
    pub amount: u64,
    pub proposal_id: String,
}

// ============================================================================
// Mirror types — fee tier
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FeeTierResponse {
    pub member_did: String,
    pub mycel_score: f64,
    pub tier_name: String,
    pub base_fee_rate: f64,
}

// ============================================================================
// Mirror types — balance
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BalanceResponse {
    pub member_did: String,
    pub currency: String,
    pub balance: u64,
    pub available: bool,
}

// ============================================================================
// DNA path helper
// ============================================================================

fn finance_dna_path() -> PathBuf {
    if let Ok(custom) = std::env::var("FINANCE_DNA_PATH") {
        return PathBuf::from(custom);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-finance/
    path.push("dna");
    path.push("mycelix_finance.dna");
    path
}

// ============================================================================
// Tests — DID Identity Gate (verify_caller_is_did)
// ============================================================================

/// Cross-hApp payment requires DID identity verification.
/// When the claimed DID doesn't match the calling agent's DID,
/// process_payment should be rejected with a DID mismatch error.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_process_payment_requires_did_match() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&finance_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Use a spoofed DID that won't match Alice's agent key
    let input = ProcessPaymentInput {
        source_happ: "test-happ".to_string(),
        from_did: "did:mycelix:spoofed_identity_abc123".to_string(),
        to_did: "did:mycelix:recipient_xyz789".to_string(),
        amount: 1000,
        currency: "SAP".to_string(),
        reference: "test-payment-gating".to_string(),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("finance_bridge"), "process_payment", input)
        .await;

    assert!(
        result.is_err(),
        "process_payment should be blocked when caller DID doesn't match"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("DID")
            || err_msg.contains("mismatch")
            || err_msg.contains("Caller")
            || err_msg.contains("identity"),
        "Error should relate to DID identity verification, got: {}",
        err_msg,
    );
}

/// Collateral deposit requires DID identity verification.
/// A spoofed depositor_did should be rejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_deposit_collateral_requires_did_match() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&finance_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = DepositCollateralInput {
        depositor_did: "did:mycelix:spoofed_depositor".to_string(),
        collateral_type: "ETH".to_string(),
        collateral_amount: 1000,
        oracle_rate: 2500.0,
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("finance_bridge"), "deposit_collateral", input)
        .await;

    assert!(
        result.is_err(),
        "deposit_collateral should be blocked when caller DID doesn't match"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("DID")
            || err_msg.contains("mismatch")
            || err_msg.contains("Caller")
            || err_msg.contains("identity"),
        "Error should relate to DID identity verification, got: {}",
        err_msg,
    );
}

/// Collateral registration requires DID identity verification.
/// A spoofed owner_did should be rejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_register_collateral_requires_did_match() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&finance_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = RegisterCollateralInput {
        owner_did: "did:mycelix:spoofed_owner".to_string(),
        source_happ: "test-happ".to_string(),
        asset_type: AssetType::Cryptocurrency,
        asset_id: "asset-001".to_string(),
        value_estimate: 50000,
        currency: "SAP".to_string(),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("finance_bridge"),
            "register_collateral",
            input,
        )
        .await;

    assert!(
        result.is_err(),
        "register_collateral should be blocked when caller DID doesn't match"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("DID")
            || err_msg.contains("mismatch")
            || err_msg.contains("Caller")
            || err_msg.contains("identity"),
        "Error should relate to DID identity verification, got: {}",
        err_msg,
    );
}

// ============================================================================
// Tests — MYCEL Tier Consciousness Gate (governance_mint_sap)
// ============================================================================

/// SAP governance minting requires governance agent authorization.
/// Without being a registered governance agent, mint_sap_governance
/// should be rejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_governance_mint_requires_authorization() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&finance_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // First register Alice as a governance agent (bootstrap allows it)
    let _alice_key: ::holochain::prelude::ActionHash = conductor
        .call(&alice.zome("tend"), "register_governance_agent", alice.agent_pubkey().clone())
        .await;

    // Now try to mint SAP with a spoofed recipient DID — the MYCEL tier
    // check via bridge should block Newcomers from receiving minted SAP
    let input = MintSapFromGovernanceInput {
        recipient_did: "did:mycelix:newcomer_no_mycel".to_string(),
        amount: 1_000_000, // 1 SAP in micro-SAP
        proposal_id: "prop-test-001".to_string(),
    };

    // mint_sap_from_governance checks:
    // 1. governance authorization (via tend zome)
    // 2. MYCEL tier via bridge (Newcomer blocked from receiving minted SAP)
    // 3. constitutional caps
    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("payments"),
            "mint_sap_from_governance",
            input,
        )
        .await;

    // The mint should fail — Newcomer tier check blocks newcomers
    // from receiving governance-minted SAP to prevent bootstrap attacks.
    // Note: In standalone mode, the bridge may fallback to MYCEL=0.0 (Newcomer)
    // which triggers the gate.
    assert!(
        result.is_err(),
        "mint_sap_from_governance should be blocked for newcomer recipients"
    );
}

// ============================================================================
// Tests — Read Operations NOT Gated
// ============================================================================

/// Read-only operations should NOT be gated by DID identity.
/// get_payment_history, query_sap_balance, get_member_fee_tier
/// should all succeed without matching DID credentials.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_read_operations_not_gated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&finance_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // get_payment_history — no DID gate, returns empty vec for unknown DID
    let history_input = GetPaymentHistoryInput {
        did: "did:mycelix:anyone".to_string(),
        limit: Some(10),
    };
    let history: Vec<::holochain::prelude::Record> = conductor
        .call(
            &alice.zome("finance_bridge"),
            "get_payment_history",
            history_input,
        )
        .await;
    assert!(
        history.is_empty(),
        "No payments should exist yet, but call should succeed (ungated)"
    );

    // get_member_fee_tier — no DID gate, returns tier for any DID
    let fee_tier: FeeTierResponse = conductor
        .call(
            &alice.zome("finance_bridge"),
            "get_member_fee_tier",
            "did:mycelix:anyone".to_string(),
        )
        .await;
    assert_eq!(
        fee_tier.tier_name, "Newcomer",
        "Unknown DID should default to Newcomer tier"
    );
    assert!(
        fee_tier.mycel_score == 0.0,
        "Unknown DID should have MYCEL score 0.0"
    );

    // query_sap_balance — no DID gate, returns zero balance
    let balance: BalanceResponse = conductor
        .call(
            &alice.zome("finance_bridge"),
            "query_sap_balance",
            "did:mycelix:anyone".to_string(),
        )
        .await;
    assert_eq!(balance.balance, 0, "Unknown DID should have zero balance");
}

// ============================================================================
// Tests — Bridge Health
// ============================================================================

/// Bridge health_check should always succeed (not gated).
/// Verifies the bridge reports healthy status and all expected zomes.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_health_not_gated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&finance_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let health: FinanceBridgeHealth = conductor
        .call(&alice.zome("finance_bridge"), "health_check", ())
        .await;

    assert!(health.healthy, "Bridge should be healthy");
    assert!(
        health.agent.starts_with("did:mycelix:"),
        "Agent should be a valid DID"
    );
    assert!(
        health.zomes.contains(&"payments".to_string()),
        "Bridge should list payments zome"
    );
    assert!(
        health.zomes.contains(&"treasury".to_string()),
        "Bridge should list treasury zome"
    );
    assert!(
        health.zomes.contains(&"tend".to_string()),
        "Bridge should list tend zome"
    );
}

/// Bridge health remains stable after gate rejections.
/// After a failed DID verification, bridge should still report healthy.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_health_after_gate_rejection() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&finance_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Check health before
    let health_before: FinanceBridgeHealth = conductor
        .call(&alice.zome("finance_bridge"), "health_check", ())
        .await;
    assert!(health_before.healthy, "Bridge should be healthy initially");

    // Trigger a gate rejection (spoofed DID)
    let input = ProcessPaymentInput {
        source_happ: "test".to_string(),
        from_did: "did:mycelix:spoofed".to_string(),
        to_did: "did:mycelix:other".to_string(),
        amount: 100,
        currency: "SAP".to_string(),
        reference: "health-test".to_string(),
    };
    let _result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("finance_bridge"), "process_payment", input)
        .await;

    // Health should remain stable after failure
    let health_after: FinanceBridgeHealth = conductor
        .call(&alice.zome("finance_bridge"), "health_check", ())
        .await;
    assert!(
        health_after.healthy,
        "Bridge should remain healthy after gate rejection"
    );
}

// ============================================================================
// Tests — Cache Consistency
// ============================================================================

/// Repeated DID gate failures should not produce false success.
/// Verifies that there is no credential caching that could serve
/// stale or fabricated identities.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cache_consistency_repeated_failures() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&finance_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // First attempt: spoofed DID should fail
    let input_1 = ProcessPaymentInput {
        source_happ: "test".to_string(),
        from_did: "did:mycelix:spoofed_first".to_string(),
        to_did: "did:mycelix:recipient".to_string(),
        amount: 500,
        currency: "SAP".to_string(),
        reference: "cache-test-1".to_string(),
    };
    let result_1: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("finance_bridge"), "process_payment", input_1)
        .await;
    assert!(result_1.is_err(), "First spoofed DID attempt should fail");

    // Second attempt: different spoofed DID should also fail
    let input_2 = ProcessPaymentInput {
        source_happ: "test".to_string(),
        from_did: "did:mycelix:spoofed_second".to_string(),
        to_did: "did:mycelix:recipient".to_string(),
        amount: 500,
        currency: "SAP".to_string(),
        reference: "cache-test-2".to_string(),
    };
    let result_2: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("finance_bridge"), "process_payment", input_2)
        .await;
    assert!(
        result_2.is_err(),
        "Second spoofed DID attempt should also fail — no stale credential caching"
    );

    // Third attempt: same spoofed DID as first — should still fail
    let input_3 = ProcessPaymentInput {
        source_happ: "test".to_string(),
        from_did: "did:mycelix:spoofed_first".to_string(),
        to_did: "did:mycelix:other".to_string(),
        amount: 200,
        currency: "SAP".to_string(),
        reference: "cache-test-3".to_string(),
    };
    let result_3: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("finance_bridge"), "process_payment", input_3)
        .await;
    assert!(
        result_3.is_err(),
        "Repeated spoofed DID should still fail — cache must not serve fabricated credentials"
    );
}

// ============================================================================
// Tests — Cross-Zome Gate Selectivity
// ============================================================================

/// Write operations are gated while reads are not.
/// deposit_collateral (DID gate) should fail, but
/// get_payment_history (no gate) should succeed.
/// Verifies the gate is selectively applied per-function.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_zome_gate_write_blocked_read_allowed() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&finance_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Write: deposit_collateral with spoofed DID — should fail
    let deposit_input = DepositCollateralInput {
        depositor_did: "did:mycelix:spoofed_write_test".to_string(),
        collateral_type: "USDC".to_string(),
        collateral_amount: 5000,
        oracle_rate: 1.0,
    };
    let write_result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("finance_bridge"),
            "deposit_collateral",
            deposit_input,
        )
        .await;
    assert!(
        write_result.is_err(),
        "deposit_collateral should be blocked with spoofed DID"
    );

    let err_msg = format!("{:?}", write_result.unwrap_err());
    assert!(
        err_msg.contains("DID")
            || err_msg.contains("mismatch")
            || err_msg.contains("Caller")
            || err_msg.contains("identity"),
        "Error should relate to DID verification, got: {}",
        err_msg,
    );

    // Read: get_payment_history — should succeed (no DID gate)
    let read_input = GetPaymentHistoryInput {
        did: "did:mycelix:spoofed_write_test".to_string(),
        limit: Some(5),
    };
    let read_result: Result<Vec<::holochain::prelude::Record>, _> = conductor
        .call_fallible(
            &alice.zome("finance_bridge"),
            "get_payment_history",
            read_input,
        )
        .await;
    assert!(
        read_result.is_ok(),
        "get_payment_history should succeed without DID verification (read-only, ungated)"
    );
}

/// The rejection error from DID gate should contain useful debugging context.
/// Error messages should reference DID, caller, mismatch, or identity —
/// not be a generic opaque error.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_rejection_message_is_specific() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&finance_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = RegisterCollateralInput {
        owner_did: "did:mycelix:spoofed_error_test".to_string(),
        source_happ: "test".to_string(),
        asset_type: AssetType::Equipment,
        asset_id: "asset-error-test".to_string(),
        value_estimate: 10000,
        currency: "SAP".to_string(),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("finance_bridge"),
            "register_collateral",
            input,
        )
        .await;

    assert!(result.is_err(), "register_collateral should be blocked");

    let err_msg = format!("{:?}", result.unwrap_err());
    let has_context = err_msg.contains("DID")
        || err_msg.contains("mismatch")
        || err_msg.contains("Caller")
        || err_msg.contains("identity")
        || err_msg.contains("claimed")
        || err_msg.contains("agent");
    assert!(
        has_context,
        "Rejection error should contain DID/identity context for debugging, got: {}",
        err_msg
    );
}

// ============================================================================
// Tests — Fee Tier Consciousness Level
// ============================================================================

/// Fee tier reflects consciousness level based on MYCEL score.
/// Without any MYCEL recognition history, a member should be at
/// Newcomer tier with the highest fee rate.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_fee_tier_reflects_consciousness_level() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&finance_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Query fee tier for an unknown member — should default to Newcomer
    let tier: FeeTierResponse = conductor
        .call(
            &alice.zome("finance_bridge"),
            "get_member_fee_tier",
            "did:mycelix:unknown_member".to_string(),
        )
        .await;

    assert_eq!(
        tier.tier_name, "Newcomer",
        "Member without MYCEL history should be Newcomer tier"
    );
    assert!(
        (tier.base_fee_rate - 0.001).abs() < f64::EPSILON,
        "Newcomer fee rate should be 0.1% (0.001)"
    );
    assert_eq!(
        tier.mycel_score, 0.0,
        "Unknown member should have zero MYCEL score"
    );

    // Query fee tier for Alice (agent has no recognition either)
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey());
    let alice_tier: FeeTierResponse = conductor
        .call(
            &alice.zome("finance_bridge"),
            "get_member_fee_tier",
            alice_did,
        )
        .await;

    assert_eq!(
        alice_tier.tier_name, "Newcomer",
        "Alice without recognition should also be Newcomer tier"
    );
}
