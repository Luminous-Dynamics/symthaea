// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix Water - Sweettest Integration Tests
//!
//! Uses pre-built .dna bundle instead of linking zome crates directly,
//! avoiding duplicate `__num_entry_types`/`__num_link_types` symbol collisions
//! from multiple integrity zomes.
//!
//! ## Running Tests
//!
//! ```bash
//! # Build the DNA first
//! cd mycelix-water && hc dna pack dna/
//!
//! # Run tests (require DNA bundle)
//! cargo test --test sweettest_integration -- --ignored
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types for deserialization (avoids importing zome crates)
// ============================================================================

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WaterSourceType {
    Municipal,
    Well,
    Spring,
    Rainwater,
    Aquifer,
    River,
    Lake,
    Recycled,
    Desalinated,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SourceStatus {
    Active,
    Seasonal,
    Depleted,
    Contaminated,
    UnderMaintenance,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WaterSource {
    pub id: String,
    pub name: String,
    pub source_type: WaterSourceType,
    pub max_capacity_liters: u64,
    pub recharge_rate_liters_per_day: u64,
    pub location_lat: f64,
    pub location_lon: f64,
    pub steward: AgentPubKey,
    pub status: SourceStatus,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AllocationType {
    Fixed,
    Proportional,
    Priority,
    Emergency,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WaterClassification {
    Potable,
    Cooking,
    Hygiene,
    Irrigation,
    Industrial,
    Recreation,
    Greywater,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WaterShare {
    pub source_hash: ActionHash,
    pub holder: AgentPubKey,
    pub allocation_type: AllocationType,
    pub volume_per_period_liters: u64,
    pub period_days: u32,
    pub priority: u8,
    pub usage_category: WaterClassification,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct H2OCredit {
    pub holder: AgentPubKey,
    pub balance_liters: i64,
    pub total_earned: u64,
    pub total_spent: u64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TransactionType {
    Allocation,
    Transfer,
    Purchase,
    Donation,
    Emergency,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WaterTransaction {
    pub from_agent: AgentPubKey,
    pub to_agent: AgentPubKey,
    pub liters: u64,
    pub credit_type: TransactionType,
    pub timestamp: Timestamp,
    pub source_hash: Option<ActionHash>,
}

// --- Input types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateSourceStatusInput {
    pub source_hash: ActionHash,
    pub new_status: SourceStatus,
}

// ============================================================================
// Test Utilities
// ============================================================================

fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_water_dna.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load water DNA bundle")
}

fn decode_entry<T: serde::de::DeserializeOwned>(record: &Record) -> Option<T> {
    match record.entry().as_option()? {
        Entry::App(bytes) => {
            let sb = SerializedBytes::from(bytes.to_owned());
            rmp_serde::from_slice(sb.bytes()).ok()
        }
        _ => None,
    }
}

fn make_water_source(id: &str, steward: &AgentPubKey) -> WaterSource {
    WaterSource {
        id: id.to_string(),
        name: format!("Community Well {}", id),
        source_type: WaterSourceType::Well,
        max_capacity_liters: 1_000_000,
        recharge_rate_liters_per_day: 5000,
        location_lat: 32.9483,
        location_lon: -96.7299,
        steward: steward.clone(),
        status: SourceStatus::Active,
    }
}

fn make_water_share(source_hash: &ActionHash, holder: &AgentPubKey) -> WaterShare {
    WaterShare {
        source_hash: source_hash.clone(),
        holder: holder.clone(),
        allocation_type: AllocationType::Fixed,
        volume_per_period_liters: 1000,
        period_days: 30,
        priority: 1,
        usage_category: WaterClassification::Potable,
    }
}

// ============================================================================
// Flow Tests
// ============================================================================

#[cfg(test)]
mod flow_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_register_and_get_source() {
        println!("=== test_register_and_get_source ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-water", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let source = make_water_source("WS-001", &agent);

        let source_record: Record = conductor
            .call(&cell.zome("flow"), "register_source", source.clone())
            .await;

        let created: WaterSource = decode_entry(&source_record).expect("No entry found");

        assert_eq!(created.id, "WS-001");
        assert_eq!(created.name, "Community Well WS-001");
        assert_eq!(created.status, SourceStatus::Active);
        assert_eq!(created.steward, agent);

        // Retrieve by hash
        let retrieved: Option<Record> = conductor
            .call(
                &cell.zome("flow"),
                "get_source",
                source_record.action_address().clone(),
            )
            .await;

        assert!(retrieved.is_some(), "Source should be retrievable by hash");

        println!("=== test_register_and_get_source PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_all_sources() {
        println!("=== test_get_all_sources ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-water", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        for i in 0..3 {
            let source = make_water_source(&format!("WS-ALL-{}", i), &agent);
            let _: Record = conductor
                .call(&cell.zome("flow"), "register_source", source)
                .await;
        }

        let all_sources: Vec<Record> = conductor
            .call(&cell.zome("flow"), "get_all_sources", ())
            .await;

        assert!(all_sources.len() >= 3, "Should have at least 3 sources");

        println!("=== test_get_all_sources PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_update_source_status() {
        println!("=== test_update_source_status ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-water", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let source = make_water_source("WS-UPD", &agent);
        let source_record: Record = conductor
            .call(&cell.zome("flow"), "register_source", source)
            .await;
        let source_hash = source_record.action_address().clone();

        // Update to UnderMaintenance
        let update_input = UpdateSourceStatusInput {
            source_hash: source_hash.clone(),
            new_status: SourceStatus::UnderMaintenance,
        };

        let updated_record: Record = conductor
            .call(&cell.zome("flow"), "update_source_status", update_input)
            .await;

        let updated: WaterSource = decode_entry(&updated_record).expect("No entry found");
        assert_eq!(updated.status, SourceStatus::UnderMaintenance);
        assert_eq!(updated.id, "WS-UPD");

        println!("=== test_update_source_status PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_allocate_shares() {
        println!("=== test_allocate_shares ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app_steward = conductor
            .setup_app("app-steward", &[dna.clone()])
            .await
            .unwrap();
        let app_holder = conductor
            .setup_app("app-holder", &[dna.clone()])
            .await
            .unwrap();
        let steward_cell = app_steward.cells()[0].clone();
        let steward = app_steward.agent().clone();
        let holder = app_holder.agent().clone();

        // Register source
        let source = make_water_source("WS-ALLOC", &steward);
        let source_record: Record = conductor
            .call(&steward_cell.zome("flow"), "register_source", source)
            .await;
        let source_hash = source_record.action_address().clone();

        // Allocate share to holder
        let share = make_water_share(&source_hash, &holder);
        let share_record: Record = conductor
            .call(&steward_cell.zome("flow"), "allocate_shares", share)
            .await;

        let created_share: WaterShare = decode_entry(&share_record).expect("No entry found");
        assert_eq!(created_share.holder, holder);
        assert_eq!(created_share.volume_per_period_liters, 1000);

        // Check source allocations
        let allocations: Vec<Record> = conductor
            .call(
                &steward_cell.zome("flow"),
                "get_source_allocations",
                source_hash,
            )
            .await;

        assert!(allocations.len() >= 1, "Should have at least 1 allocation");

        println!("=== test_allocate_shares PASSED ===\n");
    }
}

// ============================================================================
// Integration Flow Tests
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_complete_water_management_flow() {
        println!("=== test_complete_water_management_flow ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app_steward = conductor
            .setup_app("app-steward-flow", &[dna.clone()])
            .await
            .unwrap();
        let app_user = conductor
            .setup_app("app-user-flow", &[dna.clone()])
            .await
            .unwrap();
        let steward_cell = app_steward.cells()[0].clone();
        let user_cell = app_user.cells()[0].clone();
        let steward = app_steward.agent().clone();
        let user = app_user.agent().clone();

        // 1. Register water source
        let source = make_water_source("WS-FLOW", &steward);
        let source_record: Record = conductor
            .call(&steward_cell.zome("flow"), "register_source", source)
            .await;
        let source_hash = source_record.action_address().clone();
        println!("  1. Water source registered: WS-FLOW");

        // 2. Allocate share to user
        let share = make_water_share(&source_hash, &user);
        let _: Record = conductor
            .call(&steward_cell.zome("flow"), "allocate_shares", share)
            .await;
        println!("  2. Share allocated to user: 1000L/30days");

        // 3. Check user's initial credit balance
        let balance: H2OCredit = conductor
            .call(&user_cell.zome("flow"), "get_my_balance", ())
            .await;
        assert_eq!(balance.balance_liters, 0, "Initial balance should be 0");
        println!(
            "  3. User initial balance: {} liters",
            balance.balance_liters
        );

        // 4. Verify all sources list
        let all_sources: Vec<Record> = conductor
            .call(&steward_cell.zome("flow"), "get_all_sources", ())
            .await;
        assert!(all_sources.len() >= 1, "Should have at least 1 source");
        println!("  4. Total sources registered: {}", all_sources.len());

        // 5. Update source to maintenance
        let update_input = UpdateSourceStatusInput {
            source_hash: source_hash.clone(),
            new_status: SourceStatus::UnderMaintenance,
        };
        let updated: Record = conductor
            .call(
                &steward_cell.zome("flow"),
                "update_source_status",
                update_input,
            )
            .await;
        let final_source: WaterSource = decode_entry(&updated).expect("decode");
        assert_eq!(final_source.status, SourceStatus::UnderMaintenance);
        println!("  5. Source moved to UnderMaintenance");

        println!("=== test_complete_water_management_flow PASSED ===\n");
    }
}

// ============================================================================
// Unit Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_water_source_type_serialization() {
        let types = vec![
            WaterSourceType::Municipal,
            WaterSourceType::Well,
            WaterSourceType::Desalinated,
        ];

        for wst in types {
            let json = serde_json::to_string(&wst).expect("serialize");
            let deserialized: WaterSourceType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(wst, deserialized);
        }
    }

    #[test]
    fn test_source_status_serialization() {
        let statuses = vec![
            SourceStatus::Active,
            SourceStatus::Depleted,
            SourceStatus::Contaminated,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("serialize");
            let deserialized: SourceStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(status, deserialized);
        }
    }

    #[test]
    fn test_allocation_type_serialization() {
        let types = vec![
            AllocationType::Fixed,
            AllocationType::Proportional,
            AllocationType::Emergency,
        ];

        for at in types {
            let json = serde_json::to_string(&at).expect("serialize");
            let deserialized: AllocationType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(at, deserialized);
        }
    }
}
