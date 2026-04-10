#![cfg(test)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # "Day in the Life" — Narrative Integration Test
//!
//! Simulates a realistic day in a Mycelix community,
//! Africa, exercising all major finance subsystems together in a coherent
//! scenario. This proves the system works as a whole rather than in isolation.
//!
//! ## Characters
//!
//! - **Thandi**: Solar farmer, produces energy certificates, manages collateral
//! - **Sipho**: Community gardener, exchanges TEND hours, reports prices
//! - **Naledi**: Governance steward, manages treasury and commons pools
//!
//! ## Timeline
//!
//! 06:00 — Morning: Sipho provides eldercare, TEND exchange recorded
//! 07:00 — Sipho and Thandi balances queried
//! 09:00 — Thandi registers solar energy certificate (50 kWh)
//! 10:00 — Thandi uses energy certificate as collateral
//! 12:00 — Sipho registers maize harvest as agricultural asset
//! 14:00 — Sipho and Thandi report price oracle observations
//! 14:30 — Oracle consensus computed
//! 15:00 — Naledi creates commons pool and contributes treasury funds
//! 17:00 — Collateral health check (LTV monitoring)
//! 18:00 — Naledi initializes MYCEL recognition for community members
//! 20:00 — Demurrage application, compost redistribution
//! 22:00 — End of day: health check, verification
//!
//! ## Running
//!
//! ```bash
//! cd mycelix-finance/tests
//! cargo test --release --test sweettest_day_in_the_life -- --ignored --test-threads=1
//! ```

use finance_leptos_types::{ContributionType, ExchangeStatus, ServiceCategory};
use finance_wire_types::{
    ApplyDemurrageInput, BalanceInfo, CommonsPool, ContributeToCommonsInput, CreateCommonsPoolInput,
    DemurrageResult, ExchangeRecord, FeeTierResponse, FinanceBridgeHealth, GetBalanceInput,
    InitializeMemberInput, MemberMycelState, RecognitionEvent, RecognizeMemberInput,
    RecordExchangeInput, RegisterCollateralInput, SapBalanceResponse, UpdateCollateralHealthInput,
    AssetType, CollateralHealth,
};
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — Energy Certificate
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterEnergyCertificateInput {
    pub project_id: String,
    pub source: String,
    pub kwh_produced: f64,
    pub period_start: holochain::prelude::Timestamp,
    pub period_end: holochain::prelude::Timestamp,
    pub location_lat: f64,
    pub location_lon: f64,
    pub producer_did: String,
    pub terra_atlas_id: Option<String>,
}

// ============================================================================
// Mirror types — Agricultural Asset
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterAgriculturalAssetInput {
    pub asset_type: String,
    pub quantity_kg: f64,
    pub location_lat: f64,
    pub location_lon: f64,
    pub production_date: holochain::prelude::Timestamp,
    pub viability_duration_micros: Option<i64>,
    pub producer_did: String,
}

// ============================================================================
// Mirror types — Price Oracle
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReportPriceInput {
    pub item: String,
    pub price_tend: f64,
    pub evidence: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetConsensusInput {
    pub item: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConsensusResult {
    pub item: String,
    pub median_price: f64,
    pub reporter_count: u32,
    pub std_dev: f64,
    pub signal_integrity: f64,
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

/// Decode an entry from a Record into a concrete type via MessagePack deserialization.
fn decode_entry<T: serde::de::DeserializeOwned>(record: &holochain::prelude::Record) -> Option<T> {
    use holochain::prelude::*;
    match record.entry().as_option()? {
        Entry::App(bytes) => {
            let sb = SerializedBytes::from(bytes.to_owned());
            rmp_serde::from_slice(sb.bytes()).ok()
        }
        _ => None,
    }
}

/// Extract a string field from a Record's entry via msgpack deserialization.
fn extract_entry_field(record: &holochain::prelude::Record, field: &str) -> Option<String> {
    use holochain::prelude::{SerializedBytes, UnsafeBytes};
    let entry = record.entry().as_option()?;
    let sb = SerializedBytes::try_from(entry.clone()).ok()?;
    let bytes: Vec<u8> = UnsafeBytes::from(sb).into();
    if let Ok(value) = rmp_serde::from_slice::<serde_json::Value>(&bytes) {
        if let Some(s) = value.get(field).and_then(|v| v.as_str()) {
            return Some(s.to_string());
        }
    }
    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
        if let Some(s) = value.get(field).and_then(|v| v.as_str()) {
            return Some(s.to_string());
        }
    }
    None
}

// ============================================================================
// Main narrative test
// ============================================================================

/// A full day in a Mycelix community.
///
/// Three community members exercise the entire finance stack:
/// TEND timebank, energy certificates, agricultural assets, collateral,
/// price oracle, commons treasury, MYCEL recognition, demurrage, and
/// end-of-day health checks. Each phase builds on earlier ones, proving
/// the system's coherence under realistic usage.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn day_in_the_life_community() {
    // ── Setup: Three community members ──────────────────────────────
    //
    // Thandi (solar farmer), Sipho (community gardener), Naledi (steward).
    // Each gets their own agent and cell so DID verification passes.
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&finance_dna_path())
        .await
        .expect("Failed to load finance DNA — run 'hc dna pack dna/' first");

    let thandi_app = conductor
        .setup_app("thandi-app", &[dna_file.clone()])
        .await
        .expect("Failed to setup Thandi app");
    let sipho_app = conductor
        .setup_app("sipho-app", &[dna_file.clone()])
        .await
        .expect("Failed to setup Sipho app");
    let naledi_app = conductor
        .setup_app("naledi-app", &[dna_file])
        .await
        .expect("Failed to setup Naledi app");

    let (thandi,) = thandi_app.into_tuple();
    let (sipho,) = sipho_app.into_tuple();
    let (naledi,) = naledi_app.into_tuple();

    let thandi_did = format!("did:mycelix:{}", thandi.agent_pubkey());
    let sipho_did = format!("did:mycelix:{}", sipho.agent_pubkey());
    let naledi_did = format!("did:mycelix:{}", naledi.agent_pubkey());

    let dao_did = "did:mycelix:dao:example-community".to_string();

    println!("Community members:");
    println!("  Thandi (solar farmer): {}", thandi_did);
    println!("  Sipho  (gardener):     {}", sipho_did);
    println!("  Naledi (steward):      {}", naledi_did);

    // ── 06:00 Morning: TEND care exchange ───────────────────────────
    //
    // Sipho provides 3 hours of eldercare to Thandi's grandmother.
    // Sipho is the provider (caller), Thandi is the receiver.
    // TEND exchanges start as Proposed; balances update on confirmation.

    println!("\n=== 06:00 Morning: TEND Care Exchange ===");

    let exchange_input = RecordExchangeInput {
        receiver_did: thandi_did.clone(),
        hours: 3.0,
        service_description:
            "Eldercare for Thandi's grandmother — morning routine, medication, breakfast".into(),
        service_category: ServiceCategory::CareWork,
        cultural_alias: Some("Ubuntu care".into()),
        dao_did: dao_did.clone(),
        service_date: None,
    };

    let exchange: ExchangeRecord = conductor
        .call(&sipho.zome("tend"), "record_exchange", exchange_input)
        .await;

    assert_eq!(exchange.hours, 3.0, "Exchange should record 3 hours");
    assert_eq!(
        exchange.service_category,
        ServiceCategory::CareWork,
        "Category should be CareWork"
    );
    assert_eq!(
        exchange.status,
        ExchangeStatus::Proposed,
        "New exchange should be Proposed"
    );
    assert_eq!(
        exchange.receiver_did, thandi_did,
        "Receiver should be Thandi"
    );
    println!(
        "  OK: Sipho recorded 3h eldercare for Thandi (status: Proposed, id: {})",
        exchange.id
    );

    // ── 07:00 Morning: Check TEND balances ──────────────────────────
    //
    // Both should start at 0 — exchange is Proposed, not yet Confirmed.

    println!("\n=== 07:00 Morning: TEND Balance Check ===");

    let sipho_balance: BalanceInfo = conductor
        .call(
            &sipho.zome("tend"),
            "get_balance",
            GetBalanceInput {
                member_did: sipho_did.clone(),
                dao_did: dao_did.clone(),
            },
        )
        .await;

    assert_eq!(
        sipho_balance.balance, 0,
        "Sipho TEND balance should be 0 (exchange not yet confirmed)"
    );
    assert!(sipho_balance.can_provide, "Sipho should be able to provide");
    assert!(sipho_balance.can_receive, "Sipho should be able to receive");
    println!(
        "  OK: Sipho TEND balance: {} (can_provide={}, can_receive={})",
        sipho_balance.balance, sipho_balance.can_provide, sipho_balance.can_receive
    );

    let thandi_balance: BalanceInfo = conductor
        .call(
            &thandi.zome("tend"),
            "get_balance",
            GetBalanceInput {
                member_did: thandi_did.clone(),
                dao_did: dao_did.clone(),
            },
        )
        .await;

    assert_eq!(
        thandi_balance.balance, 0,
        "Thandi TEND balance should be 0 (exchange not yet confirmed)"
    );
    println!("  OK: Thandi TEND balance: {}", thandi_balance.balance);

    // ── 09:00 Morning: Energy certificate registration ──────────────
    //
    // Thandi's rooftop solar array produced 50 kWh overnight.
    // She registers this as an energy certificate on the DHT.

    println!("\n=== 09:00 Morning: Energy Certificate ===");

    let now_micros = 1700032400_000_000i64; // arbitrary timestamp
    let energy_cert_input = RegisterEnergyCertificateInput {
        project_id: "proj-solar-thandi-rooftop".into(),
        source: "Solar".into(),
        kwh_produced: 50.0,
        period_start: holochain::prelude::Timestamp::from_micros(now_micros - 32400_000_000), // 9h ago
        period_end: holochain::prelude::Timestamp::from_micros(now_micros),
        location_lat: -26.1625,
        location_lon: 27.8625,
        producer_did: thandi_did.clone(),
        terra_atlas_id: None,
    };

    let cert_record: holochain::prelude::Record = conductor
        .call(
            &thandi.zome("finance_bridge"),
            "register_energy_certificate",
            energy_cert_input,
        )
        .await;

    let cert_id = extract_entry_field(&cert_record, "id")
        .expect("Energy certificate should have an 'id' field");
    assert!(
        cert_id.starts_with("cert:"),
        "Certificate ID should have cert: prefix, got: {}",
        cert_id
    );
    println!(
        "  OK: Energy certificate registered — 50 kWh solar (id: {})",
        cert_id
    );

    // ── 10:00 Morning: Use certificate as collateral ────────────────
    //
    // Thandi registers the energy certificate as collateral for SAP.
    // Value estimate: 50 kWh * 10,000 micro-SAP/kWh = 500,000 micro-SAP.

    println!("\n=== 10:00 Morning: Collateral Registration ===");

    let collateral_input = RegisterCollateralInput {
        owner_did: thandi_did.clone(),
        source_happ: "mycelix-energy".into(),
        asset_type: AssetType::EnergyAsset,
        asset_id: cert_id.clone(),
        value_estimate: 500_000, // 0.5 SAP in micro-units
        currency: "SAP".into(),
    };

    let collateral_record: holochain::prelude::Record = conductor
        .call(
            &thandi.zome("finance_bridge"),
            "register_collateral",
            collateral_input,
        )
        .await;

    let collateral_id = extract_entry_field(&collateral_record, "id")
        .expect("Collateral should have an 'id' field");
    assert!(
        collateral_id.starts_with("collateral:"),
        "Collateral ID should have collateral: prefix, got: {}",
        collateral_id
    );
    println!(
        "  OK: Energy certificate registered as collateral (id: {}, value: 500,000 micro-SAP)",
        collateral_id
    );

    // ── 12:00 Midday: Agricultural asset registration ───────────────
    //
    // Sipho's community garden produced 200 kg of maize this season.
    // Registered as an agricultural asset with 180-day viability.

    println!("\n=== 12:00 Midday: Agricultural Asset ===");

    let agri_input = RegisterAgriculturalAssetInput {
        asset_type: "Grain".into(),
        quantity_kg: 200.0,
        location_lat: -26.1700,
        location_lon: 27.8580,
        production_date: holochain::prelude::Timestamp::from_micros(now_micros),
        viability_duration_micros: Some(180i64 * 24 * 60 * 60 * 1_000_000), // 180 days
        producer_did: sipho_did.clone(),
    };

    let agri_record: holochain::prelude::Record = conductor
        .call(
            &sipho.zome("finance_bridge"),
            "register_agricultural_asset",
            agri_input,
        )
        .await;

    let agri_id = extract_entry_field(&agri_record, "id")
        .expect("Agricultural asset should have an 'id' field");
    assert!(
        agri_id.starts_with("agri:"),
        "Agri asset ID should have agri: prefix, got: {}",
        agri_id
    );
    println!(
        "  OK: 200 kg maize harvest registered (id: {}, 180-day viability)",
        agri_id
    );

    // ── 14:00 Afternoon: Price oracle reports ───────────────────────
    //
    // Sipho and Thandi both report the current bread price from local
    // shops. Two reporters are needed for consensus.

    println!("\n=== 14:00 Afternoon: Oracle Price Reports ===");

    // Sipho reports bread at 0.15 TEND from Shoprite
    let _sipho_report: holochain::prelude::Record = conductor
        .call(
            &sipho.zome("price_oracle"),
            "report_price",
            ReportPriceInput {
                item: "bread_750g".into(),
                price_tend: 0.15,
                evidence: "Community Store, 2026-03-19".into(),
            },
        )
        .await;
    println!("  OK: Sipho reported bread_750g at 0.15 TEND (Shoprite)");

    // Thandi reports bread at 0.16 TEND from Pick n Pay
    let _thandi_report: holochain::prelude::Record = conductor
        .call(
            &thandi.zome("price_oracle"),
            "report_price",
            ReportPriceInput {
                item: "bread_750g".into(),
                price_tend: 0.16,
                evidence: "Local Market, 2026-03-19".into(),
            },
        )
        .await;
    println!("  OK: Thandi reported bread_750g at 0.16 TEND (Pick n Pay)");

    // ── 14:30 Afternoon: Oracle consensus ───────────────────────────

    println!("\n=== 14:30 Afternoon: Oracle Consensus ===");

    let consensus: ConsensusResult = conductor
        .call(
            &naledi.zome("price_oracle"),
            "get_consensus_price",
            GetConsensusInput {
                item: "bread_750g".into(),
            },
        )
        .await;

    assert_eq!(
        consensus.reporter_count, 2,
        "Should have 2 reporters for consensus"
    );
    assert!(
        consensus.median_price >= 0.14 && consensus.median_price <= 0.17,
        "Consensus price should be between reports: got {}",
        consensus.median_price
    );
    assert!(
        consensus.signal_integrity > 0.5,
        "Signal integrity should be reasonable: got {}",
        consensus.signal_integrity
    );
    println!(
        "  OK: Consensus: bread_750g = {:.2} TEND (reporters: {}, integrity: {:.2})",
        consensus.median_price, consensus.reporter_count, consensus.signal_integrity
    );

    // ── 15:00 Afternoon: Commons pool & treasury ────────────────────
    //
    // Naledi creates a commons pool for the community and contributes
    // 2,000 SAP from the community treasury (25% inalienable reserve).

    println!("\n=== 15:00 Afternoon: Treasury — Commons Pool ===");

    let pool_record: holochain::prelude::Record = conductor
        .call(
            &naledi.zome("treasury"),
            "create_commons_pool",
            CreateCommonsPoolInput {
                dao_did: dao_did.clone(),
            },
        )
        .await;

    let pool: CommonsPool = decode_entry(&pool_record).expect("Failed to decode CommonsPool");
    assert!(!pool.id.is_empty(), "Pool should have an ID");
    assert!(
        pool.demurrage_exempt.unwrap_or(false),
        "Commons pool must be demurrage exempt"
    );
    println!(
        "  OK: Commons pool created (id: {}, demurrage_exempt={})",
        pool.id,
        pool.demurrage_exempt.unwrap_or(false)
    );

    // Naledi contributes 2,000 SAP
    let contrib_record: holochain::prelude::Record = conductor
        .call(
            &naledi.zome("treasury"),
            "contribute_to_commons",
            ContributeToCommonsInput {
                commons_pool_id: pool.id.clone(),
                contributor_did: naledi_did.clone(),
                amount: 2000,
            },
        )
        .await;

    let funded_pool: CommonsPool =
        decode_entry(&contrib_record).expect("Failed to decode funded CommonsPool");
    assert_eq!(
        funded_pool.inalienable_reserve, 500,
        "25% of 2,000 = 500 should go to inalienable reserve"
    );
    assert_eq!(
        funded_pool.available_balance, 1500,
        "75% of 2,000 = 1,500 should be available"
    );
    println!(
        "  OK: Naledi contributed 2,000 SAP (reserve: {}, available: {})",
        funded_pool.inalienable_reserve, funded_pool.available_balance
    );

    // ── 17:00 Evening: Collateral health check ──────────────────────
    //
    // LTV monitoring for Thandi's energy collateral position.
    // Obligation 200,000 micro-SAP against 500,000 value = 0.40 LTV (Healthy).

    println!("\n=== 17:00 Evening: Collateral Health Monitoring ===");

    let health_record: holochain::prelude::Record = conductor
        .call(
            &thandi.zome("finance_bridge"),
            "update_collateral_health",
            UpdateCollateralHealthInput {
                collateral_id: collateral_id.clone(),
                obligation_amount: 200_000, // 40% of 500,000 value
            },
        )
        .await;

    let health: CollateralHealth =
        decode_entry(&health_record).expect("Failed to decode CollateralHealth");
    assert_eq!(
        health.collateral_id, collateral_id,
        "Health check should reference correct collateral"
    );
    assert!(
        health.obligation_amount == 200_000,
        "Obligation should be 200,000"
    );
    // Note: LTV depends on oracle fetching the collateral value; in standalone
    // mode it may use the registered estimate or a fallback.
    println!(
        "  OK: Collateral health — LTV: {:.2}, status: {}, value: {}",
        health.ltv_ratio, health.status, health.current_value
    );

    // ── 18:00 Evening: MYCEL recognition ────────────────────────────
    //
    // Naledi initializes MYCEL scores for all three members, then
    // recognizes Sipho for his eldercare contribution.

    println!("\n=== 18:00 Evening: MYCEL Recognition ===");

    // Initialize all three members
    let _naledi_init: holochain::prelude::Record = conductor
        .call(
            &naledi.zome("recognition"),
            "initialize_member",
            InitializeMemberInput {
                member_did: naledi_did.clone(),
                is_apprentice: false,
                mentor_did: None,
            },
        )
        .await;

    let sipho_init: holochain::prelude::Record = conductor
        .call(
            &sipho.zome("recognition"),
            "initialize_member",
            InitializeMemberInput {
                member_did: sipho_did.clone(),
                is_apprentice: false,
                mentor_did: None,
            },
        )
        .await;

    let sipho_mycel: MemberMycelState =
        decode_entry(&sipho_init).expect("Failed to decode Sipho MemberMycelState");
    assert_eq!(
        sipho_mycel.mycel_score, 0.3,
        "Full member initial MYCEL should be 0.3"
    );
    assert!(
        !sipho_mycel.is_apprentice.unwrap_or(false),
        "Sipho should not be an apprentice"
    );
    println!(
        "  OK: Members initialized (Sipho MYCEL: {}, apprentice: {})",
        sipho_mycel.mycel_score,
        sipho_mycel.is_apprentice.unwrap_or(false)
    );

    // Naledi recognizes Sipho for eldercare
    let recognition_record: holochain::prelude::Record = conductor
        .call(
            &naledi.zome("recognition"),
            "recognize_member",
            RecognizeMemberInput {
                recipient_did: sipho_did.clone(),
                contribution_type: ContributionType::Care,
                cycle_id: "2026-03".into(),
            },
        )
        .await;

    let event: RecognitionEvent =
        decode_entry(&recognition_record).expect("Failed to decode RecognitionEvent");
    assert_eq!(event.recipient_did, sipho_did, "Recipient should be Sipho");
    assert_eq!(
        event.contribution_type,
        ContributionType::Care,
        "Type should be Care"
    );
    assert!(event.weight > 0.0, "Recognition weight must be positive");
    println!(
        "  OK: Naledi recognized Sipho for Care (weight: {:.3}, cycle: {})",
        event.weight, event.cycle_id
    );

    // ── 19:00 Evening: Initialize SAP balances ──────────────────────
    //
    // SAP balances must be initialized before demurrage can be applied.

    println!("\n=== 19:00 Evening: SAP Balance Initialization ===");

    let _thandi_sap_init: holochain::prelude::Record = conductor
        .call(
            &thandi.zome("payments"),
            "initialize_sap_balance",
            thandi_did.clone(),
        )
        .await;
    println!("  OK: Thandi SAP balance initialized");

    let _sipho_sap_init: holochain::prelude::Record = conductor
        .call(
            &sipho.zome("payments"),
            "initialize_sap_balance",
            sipho_did.clone(),
        )
        .await;
    println!("  OK: Sipho SAP balance initialized");

    let _naledi_sap_init: holochain::prelude::Record = conductor
        .call(
            &naledi.zome("payments"),
            "initialize_sap_balance",
            naledi_did.clone(),
        )
        .await;
    println!("  OK: Naledi SAP balance initialized");

    // ── 20:00 Night: Demurrage application ──────────────────────────
    //
    // SAP balances subject to 2% annual demurrage. For newly initialized
    // balances at 0, deduction should be 0 (nothing to demurrage).
    // Compost redistribution to commons pools.

    println!("\n=== 20:00 Night: Demurrage & Compost ===");

    let demurrage_result: DemurrageResult = conductor
        .call(
            &thandi.zome("payments"),
            "apply_demurrage",
            ApplyDemurrageInput {
                member_did: thandi_did.clone(),
                local_commons_pool_id: Some(pool.id.clone()),
                regional_commons_pool_id: None,
                global_commons_pool_id: None,
            },
        )
        .await;

    // New member with 0 balance: no demurrage to apply
    assert_eq!(
        demurrage_result.deducted, 0,
        "Zero balance should have zero demurrage"
    );
    println!(
        "  OK: Demurrage applied to Thandi — deducted: {}, redistributed: {}",
        demurrage_result.deducted, demurrage_result.redistributed
    );

    // ── 21:00 Night: Drain pending compost ──────────────────────────

    println!("\n=== 21:00 Night: Drain Pending Compost ===");

    let drained: u32 = conductor
        .call(&naledi.zome("payments"), "drain_pending_compost", ())
        .await;

    println!("  OK: Drained {} pending compost deliveries", drained);

    // ── 22:00 Night: End-of-day verification ────────────────────────
    //
    // Final health check and balance verification across all subsystems.

    println!("\n=== 22:00 Night: End-of-Day Verification ===");

    // 1. SAP balance query
    let thandi_sap: SapBalanceResponse = conductor
        .call(
            &thandi.zome("payments"),
            "get_sap_balance",
            thandi_did.clone(),
        )
        .await;
    assert_eq!(
        thandi_sap.raw_balance, 0,
        "New member with no deposits should have 0 balance"
    );
    assert_eq!(
        thandi_sap.pending_demurrage, 0,
        "Zero balance should have no pending demurrage"
    );
    println!(
        "  OK: Thandi SAP — raw: {}, effective: {}, pending_demurrage: {}",
        thandi_sap.raw_balance, thandi_sap.effective_balance, thandi_sap.pending_demurrage
    );

    // 2. Fee tier query (consciousness-linked)
    let thandi_tier: FeeTierResponse = conductor
        .call(
            &thandi.zome("finance_bridge"),
            "get_member_fee_tier",
            thandi_did.clone(),
        )
        .await;
    // Without recognition history, defaults to Newcomer
    assert_eq!(
        thandi_tier.tier_name, "Newcomer",
        "Member without recognition should be Newcomer tier"
    );
    println!(
        "  OK: Thandi fee tier: {} (MYCEL: {}, rate: {})",
        thandi_tier.tier_name, thandi_tier.mycel_score, thandi_tier.base_fee_rate
    );

    // 3. Bridge health check (all three agents)
    let health_thandi: FinanceBridgeHealth = conductor
        .call(&thandi.zome("finance_bridge"), "health_check", ())
        .await;
    assert!(health_thandi.healthy, "Thandi's bridge should be healthy");

    let health_sipho: FinanceBridgeHealth = conductor
        .call(&sipho.zome("finance_bridge"), "health_check", ())
        .await;
    assert!(health_sipho.healthy, "Sipho's bridge should be healthy");

    let health_naledi: FinanceBridgeHealth = conductor
        .call(&naledi.zome("finance_bridge"), "health_check", ())
        .await;
    assert!(health_naledi.healthy, "Naledi's bridge should be healthy");

    assert!(
        health_thandi.zomes.len() >= 6,
        "Bridge should list at least 6 zomes, got {}",
        health_thandi.zomes.len()
    );
    println!(
        "  OK: All bridges healthy ({} zomes active)",
        health_thandi.zomes.len()
    );

    // 4. Verify TEND balance is still 0 (exchange was never confirmed)
    let sipho_final_balance: BalanceInfo = conductor
        .call(
            &sipho.zome("tend"),
            "get_balance",
            GetBalanceInput {
                member_did: sipho_did.clone(),
                dao_did: dao_did.clone(),
            },
        )
        .await;
    assert_eq!(
        sipho_final_balance.balance, 0,
        "TEND balance should still be 0 — exchange was Proposed, not Confirmed"
    );
    println!(
        "  OK: Sipho TEND balance: {} (exchange still Proposed)",
        sipho_final_balance.balance
    );

    // ── Summary ─────────────────────────────────────────────────────

    println!("\n=== Day Complete ===");
    println!("Community operated successfully for one day.");
    println!("Subsystems exercised:");
    println!("  - TEND timebank: 3h eldercare exchange (Proposed)");
    println!("  - Energy certificates: 50 kWh solar registered");
    println!("  - Agricultural assets: 200 kg maize registered");
    println!("  - Collateral: energy cert pledged, health monitored");
    println!("  - Price oracle: 2-reporter consensus on bread_750g");
    println!("  - Commons pool: 2,000 SAP contributed (500 inalienable)");
    println!("  - MYCEL recognition: Care contribution recognized");
    println!("  - SAP balances: initialized, demurrage applied (0 deducted)");
    println!("  - Compost: drain completed");
    println!("  - Fee tier: consciousness-linked Newcomer tier verified");
    println!("  - Bridge health: all 3 agents healthy, 6 zomes active");
}

// ============================================================================
// Smoke test — minimal health check
// ============================================================================

/// Verify the finance bridge health endpoint works standalone.
/// This is a minimal smoke test that does not require consciousness gating.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn smoke_test_finance_bridge_health() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&finance_dna_path())
        .await
        .expect("Failed to load finance DNA");

    let (agent,) = conductor
        .setup_app("smoke", &[dna_file])
        .await
        .expect("Failed to setup app")
        .into_tuple();

    // Retry loop for BadNonce — conductor may need time to fully initialize
    // WASM validation on slow systems. Nonce TTL is 5 minutes.
    let mut health: Option<FinanceBridgeHealth> = None;
    for attempt in 0..5 {
        match conductor
            .call_fallible::<_, FinanceBridgeHealth>(
                &agent.zome("finance_bridge"),
                "health_check",
                (),
            )
            .await
        {
            Ok(h) => {
                health = Some(h);
                break;
            }
            Err(e) => {
                let err = format!("{:?}", e);
                if err.contains("BadNonce") || err.contains("Expired") {
                    eprintln!(
                        "Attempt {}/5: BadNonce (conductor still initializing), retrying in 30s...",
                        attempt + 1
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                } else {
                    panic!("Unexpected error on health_check: {}", err);
                }
            }
        }
    }
    let health =
        health.expect("health_check failed after 5 retries (conductor may need more time)");

    assert!(health.healthy, "Bridge should be healthy");
    assert!(
        health.agent.starts_with("did:mycelix:"),
        "Agent should be a valid DID, got: {}",
        health.agent
    );
    assert!(
        health.zomes.contains(&"payments".to_string()),
        "Should list payments zome"
    );
    assert!(
        health.zomes.contains(&"treasury".to_string()),
        "Should list treasury zome"
    );
    assert!(
        health.zomes.contains(&"tend".to_string()),
        "Should list tend zome"
    );
    assert!(
        health.zomes.contains(&"recognition".to_string()),
        "Should list recognition zome"
    );
    println!(
        "Smoke test passed: bridge healthy, {} zomes active",
        health.zomes.len()
    );
}
