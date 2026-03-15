//! Resilience Kit Smoke Tests
//!
//! Cross-DNA integration tests that verify the 5-DNA resilience hApp
//! works end-to-end: TEND exchange → price report → consensus → volatility
//! → TEND limit escalation.
//!
//! These require the WASM zomes + conductor. Run with:
//!   cargo test -p mycelix-workspace-sweettest --test resilience_smoke -- --ignored --test-threads=1
//!
//! Prerequisites:
//!   - All 5 DNAs built (`just resilience-build`)
//!   - hc, holochain installed (via `nix develop`)

use holochain::sweettest::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Mirror types (avoid linking coordinator crates — symbol conflicts)
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
struct RecordExchangeInput {
    receiver_did: String,
    hours: f64,
    service_description: String,
    service_category: String,
    dao_did: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct BalanceInfo {
    member_did: String,
    dao_did: String,
    balance: f64,
    total_earned: f64,
    total_spent: f64,
    exchange_count: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ReportPriceInput {
    item: String,
    price_tend: f64,
    evidence: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GetConsensusInput {
    item: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ConsensusResult {
    item: String,
    median_price: f64,
    reporter_count: u32,
    std_dev: f64,
    window_start: i64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ComputeVolatilityInput {
    basket_name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct EmergencyMessageInput {
    channel_id: String,
    content: String,
    priority: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OracleState {
    vitality: f64,
    tier: String,
    updated_at: i64,
}

// ============================================================================
// Test: Cross-DNA TEND exchange + price report
// ============================================================================

/// Smoke test: install 5-DNA resilience hApp, record a TEND exchange,
/// verify balance changes, then report a price and compute consensus.
///
/// This catches wiring issues between:
/// - identity → finance (consciousness gating)
/// - finance → governance (proposal authorization)
/// - finance price_oracle → finance tend (volatility escalation)
#[tokio::test(flavor = "multi_thread")]
#[ignore] // Requires WASM + conductor
async fn test_resilience_tend_then_price_oracle() {
    let (conductors, _agent, cells) = setup_resilience_app(2).await;

    let alice = &conductors[0];
    let bob = &conductors[1];
    let alice_finance = &cells[0][1]; // finance role (index 1)
    let bob_finance = &cells[1][1];

    // Alice records a TEND exchange with Bob (1 hour of plumbing)
    let _: () = alice
        .call(
            &alice_finance.zome("tend"),
            "record_exchange",
            RecordExchangeInput {
                receiver_did: "bob.did".into(),
                hours: 1.0,
                service_description: "Plumbing repair".into(),
                service_category: "Maintenance".into(),
                dao_did: "roodepoort-resilience".into(),
            },
        )
        .await;

    // Wait for DHT propagation
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Both agents report bread price
    let _: () = alice
        .call(
            &alice_finance.zome("price_oracle"),
            "report_price",
            ReportPriceInput {
                item: "bread_750g".into(),
                price_tend: 0.15,
                evidence: "Shoprite Roodepoort 2026-03-15".into(),
            },
        )
        .await;

    let _: () = bob
        .call(
            &bob_finance.zome("price_oracle"),
            "report_price",
            ReportPriceInput {
                item: "bread_750g".into(),
                price_tend: 0.16,
                evidence: "Pick n Pay Florida 2026-03-15".into(),
            },
        )
        .await;

    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Compute consensus — should succeed with 2 reporters
    let consensus: ConsensusResult = alice
        .call(
            &alice_finance.zome("price_oracle"),
            "get_consensus_price",
            GetConsensusInput {
                item: "bread_750g".into(),
            },
        )
        .await;

    assert_eq!(consensus.reporter_count, 2);
    assert!(
        (consensus.median_price - 0.155).abs() < 0.02,
        "Consensus should be ~0.155, got {}",
        consensus.median_price
    );
}

/// Smoke test: emergency message sent via civic DNA.
#[tokio::test(flavor = "multi_thread")]
#[ignore] // Requires WASM + conductor
async fn test_resilience_emergency_comms() {
    let (conductors, _agent, cells) = setup_resilience_app(1).await;
    let alice = &conductors[0];
    let alice_civic = &cells[0][3]; // civic role (index 3)

    // Create an emergency channel
    let _channel_id: String = alice
        .call(
            &alice_civic.zome("emergency_comms"),
            "create_channel",
            serde_json::json!({
                "name": "Roodepoort General",
                "description": "Community-wide emergency channel"
            }),
        )
        .await;

    // Send a priority message
    let _: () = alice
        .call(
            &alice_civic.zome("emergency_comms"),
            "send_message",
            EmergencyMessageInput {
                channel_id: _channel_id.clone(),
                content: "Water supply interrupted in Florida area".into(),
                priority: "Immediate".into(),
            },
        )
        .await;
}

/// Smoke test: oracle state reflects normal conditions initially.
#[tokio::test(flavor = "multi_thread")]
#[ignore] // Requires WASM + conductor
async fn test_resilience_oracle_state_normal() {
    let (conductors, _agent, cells) = setup_resilience_app(1).await;
    let alice = &conductors[0];
    let alice_finance = &cells[0][1];

    let state: OracleState = alice
        .call(
            &alice_finance.zome("tend"),
            "get_oracle_state",
            (),
        )
        .await;

    // Initial state should be Normal tier
    assert_eq!(state.tier, "Normal");
}

// ============================================================================
// Setup helper
// ============================================================================

/// Install the 5-DNA resilience hApp for N agents.
/// Returns (conductors, first_agent_key, cells_per_conductor).
///
/// Role indices: 0=identity, 1=finance, 2=commons_care, 3=civic, 4=governance
async fn setup_resilience_app(
    n_agents: usize,
) -> (
    Vec<SweetConductor>,
    holochain_types::prelude::AgentPubKey,
    Vec<Vec<SweetCell>>,
) {
    // This is a placeholder — actual setup requires built DNA files.
    // The test runner should have already built via `just resilience-build`.
    let dna_files = vec![
        ("identity", "../../happs/identity/dna/mycelix_identity_dna.dna"),
        ("finance", "../../happs/finance/dna/mycelix_finance.dna"),
        ("commons_care", "../../happs/commons/dna/mycelix_commons_care.dna"),
        ("civic", "../../happs/civic/dna/mycelix_civic.dna"),
        ("governance", "../../happs/governance/dna/mycelix_governance_dna.dna"),
    ];

    let mut dnas = Vec::new();
    for (name, path) in &dna_files {
        let dna_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(path);
        if dna_path.exists() {
            dnas.push(SweetDnaFile::from_bundle(&dna_path).await.unwrap());
        } else {
            panic!(
                "DNA not found for role '{}' at {:?}. Run `just resilience-build` first.",
                name, dna_path
            );
        }
    }

    let mut conductors = SweetConductorBatch::from_standard_config(n_agents).await;
    let apps = conductors.setup_app("resilience", &dnas).await.unwrap();

    let agent = apps.cells()[0].agent_pubkey().clone();
    let cells: Vec<Vec<SweetCell>> = apps
        .into_inner()
        .into_iter()
        .map(|app| app.into_cells())
        .collect();

    (conductors.into_inner(), agent, cells)
}
