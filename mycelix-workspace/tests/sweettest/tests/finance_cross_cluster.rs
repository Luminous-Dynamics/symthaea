//! Finance Cross-Cluster Sweettest Integration Tests
//!
//! Tests cross-cluster finance dispatch in the unified hApp:
//! - Finance bridge health check
//! - SAP balance query via bridge
//! - Fee tier lookup (MYCEL-based)
//! - TEND limit lookup (vitality-based)
//! - Unified finance summary
//! - Cross-cluster dispatch from personal and civic clusters
//!
//! ## Prerequisites
//!
//! ```bash
//! # Build finance WASM
//! cd mycelix-finance && cargo build --release --target wasm32-unknown-unknown
//! hc dna pack mycelix-finance/dna/
//!
//! # For cross-cluster tests, also build personal + civic + pack unified hApp
//! cd mycelix-personal && cargo build --release --target wasm32-unknown-unknown
//! cd mycelix-civic    && cargo build --release --target wasm32-unknown-unknown
//! hc dna pack mycelix-personal/dna/
//! hc dna pack mycelix-civic/dna/
//! hc app pack mycelix-workspace/tests/sweettest/
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo test --release -p mycelix-sweettest --test finance_cross_cluster -- --ignored
//! ```

mod harness;

use harness::*;
use holochain::prelude::*;
use holochain::sweettest::*;
use serial_test::serial;

// ============================================================================
// Mirror types — must match actual zome coordinator struct layout
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct FinanceBridgeHealth {
    healthy: bool,
    agent: String,
    zomes: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct BalanceResponse {
    member_did: String,
    currency: String,
    balance: u64,
    available: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct FeeTierResponse {
    member_did: String,
    mycel_score: f64,
    tier_name: String,
    base_fee_rate: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TendLimitResponse {
    member_did: String,
    vitality: u32,
    tier_name: String,
    effective_limit: i32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct FinanceSummaryResponse {
    member_did: String,
    sap_balance: u64,
    tend_balance: i32,
    mycel_score: f64,
    fee_tier: String,
    fee_rate: f64,
    tend_limit: i32,
    tend_tier: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct DispatchResult {
    success: bool,
    response: Option<Vec<u8>>,
    error: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CrossClusterDispatchInput {
    role: String,
    zome: String,
    fn_name: String,
    payload: Vec<u8>,
}

// ============================================================================
// Tests — Single Finance DNA
// ============================================================================

/// Test: Finance bridge health_check returns healthy=true and lists all zomes.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled finance WASM + conductor"]
async fn test_finance_bridge_health_check() {
    let dna_path = DnaPaths::finance();
    if !dna_path.exists() {
        eprintln!("Skipping: finance DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "finance", 1).await;
    let alice = &agents[0];

    let health: FinanceBridgeHealth = alice
        .call_zome_fn("finance_bridge", "health_check", ())
        .await;

    assert!(health.healthy, "Finance bridge should report healthy");
    assert!(
        health.agent.starts_with("did:mycelix:"),
        "Agent should be a DID, got: {}",
        health.agent
    );

    let expected_zomes = vec![
        "payments",
        "treasury",
        "tend",
        "staking",
        "recognition",
        "currency_mint",
    ];
    for zome in &expected_zomes {
        assert!(
            health.zomes.contains(&zome.to_string()),
            "Health check should list '{}' zome, got: {:?}",
            zome,
            health.zomes
        );
    }
    assert_eq!(
        health.zomes.len(),
        expected_zomes.len(),
        "Should have exactly {} zomes listed",
        expected_zomes.len()
    );
}

/// Test: Initialize SAP balance via payments zome, then query via bridge.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled finance WASM + conductor"]
async fn test_finance_query_sap_balance() {
    let dna_path = DnaPaths::finance();
    if !dna_path.exists() {
        eprintln!("Skipping: finance DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "finance", 1).await;
    let alice = &agents[0];
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    // Initialize a SAP balance for alice
    let _record: Record = alice
        .call_zome_fn("payments", "initialize_sap_balance", alice_did.clone())
        .await;

    // Query the balance via the bridge
    let balance: BalanceResponse = alice
        .call_zome_fn("finance_bridge", "query_sap_balance", alice_did.clone())
        .await;

    assert_eq!(balance.member_did, alice_did, "DID should match");
    assert_eq!(balance.currency, "SAP", "Currency should be SAP");
    assert_eq!(balance.balance, 0, "Fresh balance should be zero");
    assert!(balance.available, "Balance should be available after init");
}

/// Test: Get fee tier for a member — should default to Newcomer (MYCEL=0.0).
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled finance WASM + conductor"]
async fn test_finance_get_member_fee_tier() {
    let dna_path = DnaPaths::finance();
    if !dna_path.exists() {
        eprintln!("Skipping: finance DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "finance", 1).await;
    let alice = &agents[0];
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    let fee_tier: FeeTierResponse = alice
        .call_zome_fn("finance_bridge", "get_member_fee_tier", alice_did.clone())
        .await;

    assert_eq!(fee_tier.member_did, alice_did, "DID should match");
    assert_eq!(
        fee_tier.mycel_score, 0.0,
        "Default MYCEL score should be 0.0 (recognition zome fallback)"
    );
    assert_eq!(
        fee_tier.tier_name, "Newcomer",
        "Default tier should be Newcomer"
    );
    assert!(
        fee_tier.base_fee_rate > 0.0,
        "Newcomer fee rate should be positive, got: {}",
        fee_tier.base_fee_rate
    );
}

/// Test: Get TEND limit for a member — should default to Normal (vitality=50).
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled finance WASM + conductor"]
async fn test_finance_get_member_tend_limit() {
    let dna_path = DnaPaths::finance();
    if !dna_path.exists() {
        eprintln!("Skipping: finance DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "finance", 1).await;
    let alice = &agents[0];
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    let tend_limit: TendLimitResponse = alice
        .call_zome_fn("finance_bridge", "get_member_tend_limit", alice_did.clone())
        .await;

    assert_eq!(tend_limit.member_did, alice_did, "DID should match");
    assert_eq!(
        tend_limit.vitality, 50,
        "Default vitality should be 50 (tend oracle fallback)"
    );
    assert_eq!(
        tend_limit.tier_name, "Normal",
        "Default tier should be Normal"
    );
    assert!(
        tend_limit.effective_limit != 0,
        "Normal tier should have a non-zero TEND limit"
    );
}

/// Test: Get unified finance summary combining SAP, TEND, fee tier, and limits.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled finance WASM + conductor"]
async fn test_finance_get_finance_summary() {
    let dna_path = DnaPaths::finance();
    if !dna_path.exists() {
        eprintln!("Skipping: finance DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "finance", 1).await;
    let alice = &agents[0];
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    let summary: FinanceSummaryResponse = alice
        .call_zome_fn("finance_bridge", "get_finance_summary", alice_did.clone())
        .await;

    assert_eq!(summary.member_did, alice_did, "DID should match");
    assert_eq!(summary.sap_balance, 0, "Default SAP balance should be 0");
    assert_eq!(summary.tend_balance, 0, "Default TEND balance should be 0");
    assert_eq!(
        summary.mycel_score, 0.0,
        "Default MYCEL score should be 0.0"
    );
    assert_eq!(
        summary.fee_tier, "Newcomer",
        "Default fee tier should be Newcomer"
    );
    assert!(
        summary.fee_rate > 0.0,
        "Fee rate should be positive for Newcomer"
    );
    assert_eq!(
        summary.tend_tier, "Normal",
        "Default TEND tier should be Normal"
    );
    assert!(
        summary.tend_limit != 0,
        "Normal tier should have a non-zero TEND limit"
    );
}

// ============================================================================
// Tests — Cross-Cluster (Unified hApp)
// ============================================================================

/// Test: Personal bridge dispatches to finance bridge health_check via OtherRole.
///
/// This verifies that the personal cluster can reach the finance cluster
/// through the unified hApp's role-based routing.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires unified hApp bundle"]
async fn test_finance_cross_cluster_from_personal() {
    let happ_path = DnaPaths::unified_happ();
    if !happ_path.exists() {
        eprintln!(
            "Skipping: unified hApp manifest not found at {:?}",
            happ_path
        );
        return;
    }

    let agents = setup_test_agents_from_happ(&happ_path, 1).await;
    let alice = &agents[0];

    // Verify both roles are available
    assert!(
        alice.role_cells.contains_key("personal"),
        "Unified hApp should have personal role. Available: {:?}",
        alice.role_cells.keys().collect::<Vec<_>>()
    );
    assert!(
        alice.role_cells.contains_key("finance"),
        "Unified hApp should have finance role. Available: {:?}",
        alice.role_cells.keys().collect::<Vec<_>>()
    );

    // Dispatch from personal bridge to finance bridge health_check via cross-cluster
    let dispatch_input = CrossClusterDispatchInput {
        role: "finance".into(),
        zome: "finance_bridge".into(),
        fn_name: "health_check".into(),
        payload: serde_json::to_vec(&()).expect("serialize unit"),
    };

    let result: DispatchResult = alice
        .call_zome_fn_on_role(
            "personal",
            "personal_bridge",
            "cross_cluster_dispatch",
            dispatch_input,
        )
        .await;

    assert!(
        result.success,
        "Cross-cluster dispatch personal->finance should succeed. Error: {:?}",
        result.error
    );
    assert!(
        result.response.is_some(),
        "Should have response bytes from finance health_check"
    );

    // Deserialize the health response from the dispatch result
    let health: FinanceBridgeHealth =
        serde_json::from_slice(&result.response.unwrap()).expect("deserialize health response");
    assert!(
        health.healthy,
        "Finance bridge should report healthy via cross-cluster dispatch"
    );
}

/// Test: Civic bridge dispatches to finance bridge query_sap_balance via OtherRole.
///
/// This verifies that the civic cluster can query financial data from the
/// finance cluster through cross-cluster dispatch in the unified hApp.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires unified hApp bundle"]
async fn test_finance_cross_cluster_from_civic() {
    let happ_path = DnaPaths::unified_happ();
    if !happ_path.exists() {
        eprintln!(
            "Skipping: unified hApp manifest not found at {:?}",
            happ_path
        );
        return;
    }

    let agents = setup_test_agents_from_happ(&happ_path, 1).await;
    let alice = &agents[0];

    // Verify both roles are available
    assert!(
        alice.role_cells.contains_key("civic"),
        "Unified hApp should have civic role. Available: {:?}",
        alice.role_cells.keys().collect::<Vec<_>>()
    );
    assert!(
        alice.role_cells.contains_key("finance"),
        "Unified hApp should have finance role. Available: {:?}",
        alice.role_cells.keys().collect::<Vec<_>>()
    );

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    // Dispatch from civic bridge to finance bridge query_sap_balance via cross-cluster
    let dispatch_input = CrossClusterDispatchInput {
        role: "finance".into(),
        zome: "finance_bridge".into(),
        fn_name: "query_sap_balance".into(),
        payload: serde_json::to_vec(&alice_did).expect("serialize DID"),
    };

    let result: DispatchResult = alice
        .call_zome_fn_on_role(
            "civic",
            "civic_bridge",
            "cross_cluster_dispatch",
            dispatch_input,
        )
        .await;

    assert!(
        result.success,
        "Cross-cluster dispatch civic->finance should succeed. Error: {:?}",
        result.error
    );
    assert!(
        result.response.is_some(),
        "Should have response bytes from finance query_sap_balance"
    );

    // Deserialize the balance response from the dispatch result
    let balance: BalanceResponse =
        serde_json::from_slice(&result.response.unwrap()).expect("deserialize balance response");
    assert_eq!(
        balance.member_did, alice_did,
        "Balance DID should match queried DID"
    );
    assert_eq!(balance.currency, "SAP", "Currency should be SAP");
}
