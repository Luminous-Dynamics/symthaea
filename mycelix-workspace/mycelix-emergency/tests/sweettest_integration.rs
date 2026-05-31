// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix Emergency - Sweettest Integration Tests
//!
//! Uses pre-built .dna bundle instead of linking zome crates directly,
//! avoiding duplicate `__num_entry_types`/`__num_link_types` symbol collisions
//! from multiple integrity zomes.
//!
//! ## Running Tests
//!
//! ```bash
//! # Build the DNA first
//! cd mycelix-emergency && hc dna pack dna/
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
pub enum DisasterType {
    Hurricane,
    Earthquake,
    Wildfire,
    Flood,
    Tornado,
    Pandemic,
    Industrial,
    MassCasualty,
    CyberAttack,
    Infrastructure,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SeverityLevel {
    Level1,
    Level2,
    Level3,
    Level4,
    Level5,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DisasterStatus {
    Declared,
    Active,
    Recovery,
    Closed,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AffectedArea {
    pub center_lat: f64,
    pub center_lon: f64,
    pub radius_km: f32,
    pub boundary: Option<Vec<(f64, f64)>>,
    pub zones: Vec<OperationalZone>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OperationalZone {
    pub id: String,
    pub name: String,
    pub boundary: Vec<(f64, f64)>,
    pub priority: ZonePriority,
    pub status: ZoneStatus,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ZonePriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ZoneStatus {
    Unassessed,
    Active,
    Cleared,
    Hazardous,
    Evacuated,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Disaster {
    pub id: String,
    pub disaster_type: DisasterType,
    pub title: String,
    pub description: String,
    pub severity: SeverityLevel,
    pub declared_by: AgentPubKey,
    pub declared_at: Timestamp,
    pub affected_area: AffectedArea,
    pub status: DisasterStatus,
    pub estimated_affected: u32,
    pub coordination_lead: Option<AgentPubKey>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum UpdateType {
    StatusChange,
    SeverityChange,
    AreaExpansion,
    AreaContraction,
    CasualtyReport,
    ResourceUpdate,
    WeatherUpdate,
    InfrastructureUpdate,
    EvacuationOrder,
    AllClear,
    General,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IncidentUpdate {
    pub disaster_hash: ActionHash,
    pub author: AgentPubKey,
    pub timestamp: Timestamp,
    pub update_type: UpdateType,
    pub content: String,
}

// --- Input types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeclareDisasterInput {
    pub id: String,
    pub disaster_type: DisasterType,
    pub title: String,
    pub description: String,
    pub severity: SeverityLevel,
    pub affected_area: AffectedArea,
    pub estimated_affected: u32,
    pub coordination_lead: Option<AgentPubKey>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateDisasterStatusInput {
    pub disaster_hash: ActionHash,
    pub new_status: DisasterStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AddIncidentUpdateInput {
    pub disaster_hash: ActionHash,
    pub update_type: UpdateType,
    pub content: String,
}

// ============================================================================
// Test Utilities
// ============================================================================

fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_emergency_dna.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load emergency DNA bundle")
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

fn make_affected_area() -> AffectedArea {
    AffectedArea {
        center_lat: 32.9483,
        center_lon: -96.7299,
        radius_km: 25.0,
        boundary: None,
        zones: vec![OperationalZone {
            id: "zone-1".to_string(),
            name: "Downtown".to_string(),
            boundary: vec![(32.95, -96.73), (32.94, -96.72), (32.95, -96.72)],
            priority: ZonePriority::Critical,
            status: ZoneStatus::Active,
        }],
    }
}

fn make_disaster_input(id: &str, disaster_type: DisasterType) -> DeclareDisasterInput {
    DeclareDisasterInput {
        id: id.to_string(),
        disaster_type,
        title: format!("Test Disaster: {}", id),
        description: "A test disaster declaration for integration testing.".to_string(),
        severity: SeverityLevel::Level3,
        affected_area: make_affected_area(),
        estimated_affected: 50000,
        coordination_lead: None,
    }
}

// ============================================================================
// Incident Tests
// ============================================================================

#[cfg(test)]
mod incident_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_declare_and_retrieve_disaster() {
        println!("=== test_declare_and_retrieve_disaster ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-emergency", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        let input = make_disaster_input("DIS-001", DisasterType::Flood);

        let disaster_record: Record = conductor
            .call(&cell.zome("incidents"), "declare_disaster", input)
            .await;

        let created: Disaster = decode_entry(&disaster_record).expect("No entry found");

        assert_eq!(created.id, "DIS-001");
        assert_eq!(created.disaster_type, DisasterType::Flood);
        assert_eq!(created.status, DisasterStatus::Declared);
        assert_eq!(created.estimated_affected, 50000);

        println!("=== test_declare_and_retrieve_disaster PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_active_disasters() {
        println!("=== test_get_active_disasters ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-emergency", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Declare multiple disasters
        for i in 0..3 {
            let input = make_disaster_input(&format!("DIS-A{}", i), DisasterType::Wildfire);
            let _: Record = conductor
                .call(&cell.zome("incidents"), "declare_disaster", input)
                .await;
        }

        let active: Vec<Record> = conductor
            .call(&cell.zome("incidents"), "get_active_disasters", ())
            .await;

        assert!(active.len() >= 3, "Should have at least 3 active disasters");

        println!("=== test_get_active_disasters PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_update_disaster_status() {
        println!("=== test_update_disaster_status ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-emergency", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        let input = make_disaster_input("DIS-UPD", DisasterType::Earthquake);
        let disaster_record: Record = conductor
            .call(&cell.zome("incidents"), "declare_disaster", input)
            .await;
        let disaster_hash = disaster_record.action_address().clone();

        // Update status to Recovery
        let update_input = UpdateDisasterStatusInput {
            disaster_hash: disaster_hash.clone(),
            new_status: DisasterStatus::Recovery,
        };

        let updated_record: Record = conductor
            .call(
                &cell.zome("incidents"),
                "update_disaster_status",
                update_input,
            )
            .await;

        let updated: Disaster = decode_entry(&updated_record).expect("No entry found");
        assert_eq!(updated.status, DisasterStatus::Recovery);
        assert_eq!(updated.id, "DIS-UPD");

        println!("=== test_update_disaster_status PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_add_incident_update_and_timeline() {
        println!("=== test_add_incident_update_and_timeline ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-emergency", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Declare disaster
        let input = make_disaster_input("DIS-TL", DisasterType::Hurricane);
        let disaster_record: Record = conductor
            .call(&cell.zome("incidents"), "declare_disaster", input)
            .await;
        let disaster_hash = disaster_record.action_address().clone();

        // Add updates
        for i in 0..3 {
            let update_input = AddIncidentUpdateInput {
                disaster_hash: disaster_hash.clone(),
                update_type: UpdateType::General,
                content: format!("Situation update #{}", i + 1),
            };
            let _: Record = conductor
                .call(&cell.zome("incidents"), "add_incident_update", update_input)
                .await;
        }

        // Get timeline
        let timeline: Vec<Record> = conductor
            .call(
                &cell.zome("incidents"),
                "get_disaster_timeline",
                disaster_hash,
            )
            .await;

        assert!(timeline.len() >= 3, "Should have at least 3 updates");

        println!("=== test_add_incident_update_and_timeline PASSED ===\n");
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
    async fn test_multi_agent_disaster_response_flow() {
        println!("=== test_multi_agent_disaster_response_flow ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app1 = conductor
            .setup_app("app-coordinator", &[dna.clone()])
            .await
            .unwrap();
        let app2 = conductor
            .setup_app("app-responder", &[dna.clone()])
            .await
            .unwrap();
        let coord_cell = app1.cells()[0].clone();
        let resp_cell = app2.cells()[0].clone();

        // 1. Coordinator declares a disaster
        let input = make_disaster_input("DIS-FLOW", DisasterType::Tornado);
        let disaster_record: Record = conductor
            .call(&coord_cell.zome("incidents"), "declare_disaster", input)
            .await;
        let disaster_hash = disaster_record.action_address().clone();
        println!("  1. Disaster declared: DIS-FLOW (Tornado)");

        // 2. Responder adds field updates
        let update_input = AddIncidentUpdateInput {
            disaster_hash: disaster_hash.clone(),
            update_type: UpdateType::CasualtyReport,
            content: "Confirmed 12 injuries in zone-1, no fatalities".to_string(),
        };
        let _: Record = conductor
            .call(
                &resp_cell.zome("incidents"),
                "add_incident_update",
                update_input,
            )
            .await;
        println!("  2. Responder added casualty report");

        // 3. Coordinator adds evacuation order
        let evac_input = AddIncidentUpdateInput {
            disaster_hash: disaster_hash.clone(),
            update_type: UpdateType::EvacuationOrder,
            content: "Mandatory evacuation for zone-1".to_string(),
        };
        let _: Record = conductor
            .call(
                &coord_cell.zome("incidents"),
                "add_incident_update",
                evac_input,
            )
            .await;
        println!("  3. Coordinator issued evacuation order");

        // 4. Verify timeline
        let timeline: Vec<Record> = conductor
            .call(
                &coord_cell.zome("incidents"),
                "get_disaster_timeline",
                disaster_hash.clone(),
            )
            .await;
        assert!(
            timeline.len() >= 2,
            "Should have at least 2 timeline entries"
        );
        println!("  4. Timeline has {} entries", timeline.len());

        // 5. Move to recovery
        let close_input = UpdateDisasterStatusInput {
            disaster_hash: disaster_hash.clone(),
            new_status: DisasterStatus::Recovery,
        };
        let updated: Record = conductor
            .call(
                &coord_cell.zome("incidents"),
                "update_disaster_status",
                close_input,
            )
            .await;
        let final_disaster: Disaster = decode_entry(&updated).expect("decode");
        assert_eq!(final_disaster.status, DisasterStatus::Recovery);
        println!("  5. Disaster moved to Recovery status");

        println!("=== test_multi_agent_disaster_response_flow PASSED ===\n");
    }
}

// ============================================================================
// Unit Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_disaster_type_serialization() {
        let types = vec![
            DisasterType::Hurricane,
            DisasterType::Earthquake,
            DisasterType::Other("Chemical Spill".to_string()),
        ];

        for dt in types {
            let json = serde_json::to_string(&dt).expect("serialize");
            let deserialized: DisasterType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(dt, deserialized);
        }
    }

    #[test]
    fn test_severity_level_serialization() {
        let levels = vec![
            SeverityLevel::Level1,
            SeverityLevel::Level3,
            SeverityLevel::Level5,
        ];

        for level in levels {
            let json = serde_json::to_string(&level).expect("serialize");
            let deserialized: SeverityLevel = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(level, deserialized);
        }
    }

    #[test]
    fn test_disaster_status_serialization() {
        let statuses = vec![
            DisasterStatus::Declared,
            DisasterStatus::Active,
            DisasterStatus::Recovery,
            DisasterStatus::Closed,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("serialize");
            let deserialized: DisasterStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(status, deserialized);
        }
    }
}
