// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix Housing - Sweettest Integration Tests
//!
//! Uses pre-built .dna bundle instead of linking zome crates directly,
//! avoiding duplicate `__num_entry_types`/`__num_link_types` symbol collisions
//! from multiple integrity zomes.
//!
//! ## Running Tests
//!
//! ```bash
//! # Build the DNA first
//! cd mycelix-housing && hc dna pack dna/
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
pub enum BuildingType {
    Apartment,
    Townhouse,
    SingleFamily,
    Duplex,
    CoHousing,
    MixedUse,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Building {
    pub id: String,
    pub name: String,
    pub address: String,
    pub location_lat: f64,
    pub location_lon: f64,
    pub total_units: u16,
    pub year_built: Option<u16>,
    pub building_type: BuildingType,
    pub cooperative_hash: Option<ActionHash>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum UnitType {
    Studio,
    OneBedroom,
    TwoBedroom,
    ThreeBedroom,
    FourPlus,
    Accessible,
    Family,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AccessFeature {
    WheelchairAccessible,
    Elevator,
    GrabBars,
    WideDoorways,
    LowCounters,
    VisualAlerts,
    HearingLoop,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum UnitStatus {
    Available,
    Occupied,
    UnderMaintenance,
    Reserved,
    Renovation,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Unit {
    pub building_hash: ActionHash,
    pub unit_number: String,
    pub unit_type: UnitType,
    pub square_meters: u32,
    pub floor: u8,
    pub bedrooms: u8,
    pub bathrooms: u8,
    pub accessibility_features: Vec<AccessFeature>,
    pub current_occupant: Option<AgentPubKey>,
    pub status: UnitStatus,
}

// --- Input types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateUnitStatusInput {
    pub unit_action_hash: ActionHash,
    pub new_status: UnitStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AssignOccupantInput {
    pub unit_action_hash: ActionHash,
    pub occupant: AgentPubKey,
}

// ============================================================================
// Test Utilities
// ============================================================================

fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_housing_dna.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load housing DNA bundle")
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

fn make_building(id: &str) -> Building {
    Building {
        id: id.to_string(),
        name: format!("Cooperative Building {}", id),
        address: "123 Community Lane, Richardson, TX 75080".to_string(),
        location_lat: 32.9483,
        location_lon: -96.7299,
        total_units: 24,
        year_built: Some(2020),
        building_type: BuildingType::Apartment,
        cooperative_hash: None,
    }
}

fn make_unit(building_hash: &ActionHash, unit_number: &str) -> Unit {
    Unit {
        building_hash: building_hash.clone(),
        unit_number: unit_number.to_string(),
        unit_type: UnitType::TwoBedroom,
        square_meters: 75,
        floor: 2,
        bedrooms: 2,
        bathrooms: 1,
        accessibility_features: vec![],
        current_occupant: None,
        status: UnitStatus::Available,
    }
}

// ============================================================================
// Units Tests
// ============================================================================

#[cfg(test)]
mod units_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_register_building_and_unit() {
        println!("=== test_register_building_and_unit ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-housing", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Register building
        let building = make_building("BLD-001");
        let building_record: Record = conductor
            .call(&cell.zome("units"), "register_building", building.clone())
            .await;

        let created_building: Building = decode_entry(&building_record).expect("No entry found");
        assert_eq!(created_building.id, "BLD-001");
        assert_eq!(created_building.total_units, 24);

        // Register unit in building
        let building_hash = building_record.action_address().clone();
        let unit = make_unit(&building_hash, "2A");

        let unit_record: Record = conductor
            .call(&cell.zome("units"), "register_unit", unit.clone())
            .await;

        let created_unit: Unit = decode_entry(&unit_record).expect("No entry found");
        assert_eq!(created_unit.unit_number, "2A");
        assert_eq!(created_unit.status, UnitStatus::Available);
        assert_eq!(created_unit.bedrooms, 2);

        println!("=== test_register_building_and_unit PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_building_units() {
        println!("=== test_get_building_units ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-housing", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Register building
        let building = make_building("BLD-LIST");
        let building_record: Record = conductor
            .call(&cell.zome("units"), "register_building", building)
            .await;
        let building_hash = building_record.action_address().clone();

        // Register 3 units
        for i in 1..=3 {
            let unit = make_unit(&building_hash, &format!("{}A", i));
            let _: Record = conductor
                .call(&cell.zome("units"), "register_unit", unit)
                .await;
        }

        let units: Vec<Record> = conductor
            .call(&cell.zome("units"), "get_building_units", building_hash)
            .await;

        assert!(units.len() >= 3, "Should have at least 3 units");

        println!("=== test_get_building_units PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_update_unit_status() {
        println!("=== test_update_unit_status ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-housing", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Register building and unit
        let building = make_building("BLD-UPD");
        let building_record: Record = conductor
            .call(&cell.zome("units"), "register_building", building)
            .await;
        let building_hash = building_record.action_address().clone();

        let unit = make_unit(&building_hash, "3B");
        let unit_record: Record = conductor
            .call(&cell.zome("units"), "register_unit", unit)
            .await;
        let unit_hash = unit_record.action_address().clone();

        // Update status to UnderMaintenance
        let update_input = UpdateUnitStatusInput {
            unit_action_hash: unit_hash.clone(),
            new_status: UnitStatus::UnderMaintenance,
        };

        let updated_record: Record = conductor
            .call(&cell.zome("units"), "update_unit_status", update_input)
            .await;

        let updated: Unit = decode_entry(&updated_record).expect("No entry found");
        assert_eq!(updated.status, UnitStatus::UnderMaintenance);
        assert_eq!(updated.unit_number, "3B");

        println!("=== test_update_unit_status PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_available_units() {
        println!("=== test_get_available_units ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-housing", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        let building = make_building("BLD-AVAIL");
        let building_record: Record = conductor
            .call(&cell.zome("units"), "register_building", building)
            .await;
        let building_hash = building_record.action_address().clone();

        // Register 2 available units
        for i in 1..=2 {
            let unit = make_unit(&building_hash, &format!("{}C", i));
            let _: Record = conductor
                .call(&cell.zome("units"), "register_unit", unit)
                .await;
        }

        let available: Vec<Record> = conductor
            .call(&cell.zome("units"), "get_available_units", ())
            .await;

        assert!(
            available.len() >= 2,
            "Should have at least 2 available units"
        );

        println!("=== test_get_available_units PASSED ===\n");
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
    async fn test_complete_housing_occupancy_flow() {
        println!("=== test_complete_housing_occupancy_flow ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app_admin = conductor
            .setup_app("app-admin", &[dna.clone()])
            .await
            .unwrap();
        let app_tenant = conductor
            .setup_app("app-tenant", &[dna.clone()])
            .await
            .unwrap();
        let admin_cell = app_admin.cells()[0].clone();
        let tenant = app_tenant.agent().clone();

        // 1. Register building
        let building = make_building("BLD-FLOW");
        let building_record: Record = conductor
            .call(&admin_cell.zome("units"), "register_building", building)
            .await;
        let building_hash = building_record.action_address().clone();
        println!("  1. Building registered: BLD-FLOW");

        // 2. Register unit
        let unit = make_unit(&building_hash, "4A");
        let unit_record: Record = conductor
            .call(&admin_cell.zome("units"), "register_unit", unit)
            .await;
        let unit_hash = unit_record.action_address().clone();
        println!("  2. Unit 4A registered (Available)");

        // 3. Verify unit appears in available list
        let available: Vec<Record> = conductor
            .call(&admin_cell.zome("units"), "get_available_units", ())
            .await;
        assert!(available.len() >= 1, "Should have available units");
        println!("  3. Available units: {}", available.len());

        // 4. Assign occupant
        let assign_input = AssignOccupantInput {
            unit_action_hash: unit_hash.clone(),
            occupant: tenant.clone(),
        };
        let assigned_record: Record = conductor
            .call(&admin_cell.zome("units"), "assign_occupant", assign_input)
            .await;
        let assigned: Unit = decode_entry(&assigned_record).expect("decode");
        assert_eq!(assigned.status, UnitStatus::Occupied);
        assert_eq!(assigned.current_occupant, Some(tenant.clone()));
        let occupied_hash = assigned_record.action_address().clone();
        println!("  4. Tenant assigned to unit 4A (Occupied)");

        // 5. Vacate unit
        let vacated_record: Record = conductor
            .call(&admin_cell.zome("units"), "vacate_unit", occupied_hash)
            .await;
        let vacated: Unit = decode_entry(&vacated_record).expect("decode");
        assert_eq!(vacated.status, UnitStatus::Available);
        assert_eq!(vacated.current_occupant, None);
        println!("  5. Unit 4A vacated (Available again)");

        println!("=== test_complete_housing_occupancy_flow PASSED ===\n");
    }
}

// ============================================================================
// Unit Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_building_type_serialization() {
        let types = vec![
            BuildingType::Apartment,
            BuildingType::CoHousing,
            BuildingType::MixedUse,
        ];

        for bt in types {
            let json = serde_json::to_string(&bt).expect("serialize");
            let deserialized: BuildingType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(bt, deserialized);
        }
    }

    #[test]
    fn test_unit_status_serialization() {
        let statuses = vec![
            UnitStatus::Available,
            UnitStatus::Occupied,
            UnitStatus::UnderMaintenance,
            UnitStatus::Reserved,
            UnitStatus::Renovation,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("serialize");
            let deserialized: UnitStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(status, deserialized);
        }
    }

    #[test]
    fn test_unit_type_serialization() {
        let types = vec![UnitType::Studio, UnitType::TwoBedroom, UnitType::Accessible];

        for ut in types {
            let json = serde_json::to_string(&ut).expect("serialize");
            let deserialized: UnitType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(ut, deserialized);
        }
    }

    #[test]
    fn test_access_feature_serialization() {
        let features = vec![
            AccessFeature::WheelchairAccessible,
            AccessFeature::Elevator,
            AccessFeature::HearingLoop,
        ];

        for af in features {
            let json = serde_json::to_string(&af).expect("serialize");
            let deserialized: AccessFeature = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(af, deserialized);
        }
    }
}
