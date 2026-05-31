// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix Space - Sweettest Integration Tests
//!
//! Comprehensive integration tests using Holochain's sweettest framework.
//! Tests cover orbital objects, observations, conjunctions, debris bounties,
//! and traffic control negotiation.
//!
//! ## Running Tests
//!
//! ```bash
//! # DNA bundle exists at workdir/dna/mycelix_space.dna
//!
//! # Run tests (requires Holochain conductor via nix develop)
//! cargo test --test sweettest_integration -- --ignored
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types (avoids importing zome crates / duplicate symbols)
// ============================================================================

// --- shared types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SpaceTimestamp {
    pub micros: i64,
}

impl SpaceTimestamp {
    pub fn now() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        Self {
            micros: now.as_micros() as i64,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct QualityScore {
    pub completeness: f64,
    pub accuracy: f64,
    pub timeliness: f64,
}

// --- orbital_objects types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OrbitalObject {
    pub norad_id: u32,
    pub intl_designator: String,
    pub name: String,
    pub object_type: ObjectType,
    pub country: Option<String>,
    pub launch_date: Option<SpaceTimestamp>,
    pub decay_date: Option<SpaceTimestamp>,
    pub status: OperationalStatus,
    pub created_at: SpaceTimestamp,
    pub created_by: AgentPubKey,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ObjectType {
    Payload,
    RocketBody,
    Debris,
    Unknown,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum OperationalStatus {
    Active,
    Inactive,
    Decayed,
    Unknown,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterObjectInput {
    pub norad_id: u32,
    pub intl_designator: String,
    pub name: String,
    pub object_type: ObjectType,
    pub country: Option<String>,
    pub launch_date: Option<SpaceTimestamp>,
    pub status: Option<OperationalStatus>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DataSourceType {
    SpaceTrack,
    CelesTrak,
    Operator,
    GroundStation,
    Crowdsourced,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitTleInput {
    pub norad_id: u32,
    pub line1: String,
    pub line2: String,
    pub source: Option<DataSourceType>,
    pub quality: Option<QualityScore>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClaimOperatorInput {
    pub norad_id: u32,
    pub organization: String,
    pub contact: Option<String>,
    pub verification_hash: Option<[u8; 32]>,
}

// --- conjunctions types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConjunctionEvent {
    pub event_id: String,
    pub primary_norad_id: u32,
    pub secondary_norad_id: u32,
    pub tca: SpaceTimestamp,
    pub miss_distance_km: f64,
    pub max_pc: f64,
    pub risk_level: RiskLevel,
    pub status: EventStatus,
    pub created_at: SpaceTimestamp,
    pub updated_at: SpaceTimestamp,
}

#[derive(Clone, Debug, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub enum RiskLevel {
    Negligible,
    Low,
    Medium,
    High,
    Emergency,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EventStatus {
    Screening,
    Active,
    Resolved,
    Expired,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateEventInput {
    pub event_id: String,
    pub primary_norad_id: u32,
    pub secondary_norad_id: u32,
    pub tca: SpaceTimestamp,
    pub miss_distance_km: f64,
    pub max_pc: f64,
    pub risk_level: RiskLevel,
    #[serde(default)]
    pub compute_details: bool,
    pub primary_tle: Option<(String, String)>,
    pub secondary_tle: Option<(String, String)>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AnnounceManeuverInput {
    pub event_id: String,
    pub norad_id: u32,
    pub burn_time: SpaceTimestamp,
    pub delta_v_ms: f64,
    pub direction: [f64; 3],
}

// --- observations types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ObservationType {
    Optical,
    Radar,
    LaserRanging,
    RadioFrequency,
    Passive,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GroundLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude_m: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Measurement {
    pub azimuth_deg: Option<f64>,
    pub elevation_deg: Option<f64>,
    pub range_km: Option<f64>,
    pub range_rate_kms: Option<f64>,
    pub visual_magnitude: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitObservationInput {
    pub norad_id: Option<u32>,
    pub observation_time: SpaceTimestamp,
    pub observer_location: Option<GroundLocation>,
    pub observation_type: ObservationType,
    pub measurement: Measurement,
    pub quality: Option<QualityScore>,
    pub sensor_id: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SensorCapabilities {
    pub min_elevation_deg: f64,
    pub max_range_km: f64,
    pub accuracy_arcsec: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterSensorInput {
    pub sensor_id: String,
    pub name: String,
    pub sensor_type: ObservationType,
    pub location: Option<GroundLocation>,
    pub capabilities: SensorCapabilities,
}

// --- debris bounties types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DebrisBounty {
    pub bounty_id: String,
    pub debris_norad_id: u32,
    pub justification: String,
    pub amount: u64,
    pub currency: String,
    pub expires_at: Option<SpaceTimestamp>,
    pub status: BountyStatus,
    pub creator: AgentPubKey,
    pub created_at: SpaceTimestamp,
    pub requirements: RemovalRequirements,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum BountyStatus {
    Open,
    Claimed,
    InProgress,
    Completed,
    Expired,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RemovalRequirements {
    pub method_preference: Option<String>,
    pub deadline_days: Option<u32>,
    pub minimum_verification: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateBountyInput {
    pub bounty_id: String,
    pub debris_norad_id: u32,
    pub justification: String,
    pub amount: u64,
    pub currency: String,
    pub expires_at: Option<SpaceTimestamp>,
    pub requirements: RemovalRequirements,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ContributeInput {
    pub bounty_id: String,
    pub amount: u64,
    pub currency: String,
    pub message: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum RemovalMethod {
    ActiveDebrisRemoval,
    DeOrbitBurn,
    LaserAblation,
    NetCapture,
    Harpoon,
    Other(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClaimBountyInput {
    pub bounty_id: String,
    pub organization: String,
    pub method: RemovalMethod,
    pub estimated_completion: SpaceTimestamp,
    pub mission_plan: String,
}

// --- traffic control types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InitiateNegotiationInput {
    pub session_id: String,
    pub conjunction_id: String,
    pub primary_operator: AgentPubKey,
    pub secondary_operator: AgentPubKey,
    pub primary_norad_id: u32,
    pub secondary_norad_id: u32,
    pub tca: SpaceTimestamp,
    pub deadline: SpaceTimestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ManeuverCapability {
    pub max_delta_v_ms: f64,
    pub min_lead_time_hours: f64,
    pub fuel_remaining_pct: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OperatorPreferences {
    pub willing_to_maneuver: bool,
    pub max_cost_usd: Option<f64>,
    pub preferred_direction: Option<[f64; 3]>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitPositionInput {
    pub session_id: String,
    pub norad_id: u32,
    pub maneuver_capability: ManeuverCapability,
    pub preferences: OperatorPreferences,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CostEstimate {
    pub fuel_cost_usd: f64,
    pub operational_cost_usd: f64,
    pub opportunity_cost_usd: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitProposalInput {
    pub session_id: String,
    pub maneuvering_object: u32,
    pub burn_time: SpaceTimestamp,
    pub delta_v_ms: f64,
    pub direction: [f64; 3],
    pub resulting_miss_km: f64,
    pub resulting_pc: f64,
    pub cost_estimate: Option<CostEstimate>,
}

// ============================================================================
// Test Utilities
// ============================================================================

fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("workdir")
        .join("dna")
        .join("mycelix_space.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load DNA bundle - run 'hc dna pack workdir/dna/' first")
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

// ============================================================================
// Orbital Objects Tests
// ============================================================================

#[cfg(test)]
mod orbital_objects_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_register_object() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = RegisterObjectInput {
            norad_id: 25544,
            intl_designator: "1998-067A".to_string(),
            name: "ISS (ZARYA)".to_string(),
            object_type: ObjectType::Payload,
            country: Some("ISS".to_string()),
            launch_date: None,
            status: Some(OperationalStatus::Active),
        };

        let hash: ActionHash = conductor
            .call(
                &cell.zome("orbital_objects_coordinator"),
                "register_object",
                input,
            )
            .await;

        // Verify record exists
        let record = get_record(&conductor, hash.clone()).await;
        assert!(record.is_some(), "Should create orbital object record");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_submit_tle() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // ISS TLE (example data)
        let input = SubmitTleInput {
            norad_id: 25544,
            line1: "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9990"
                .to_string(),
            line2: "2 25544  51.6400 208.9163 0006703 306.0500  54.0150 15.49560532999999"
                .to_string(),
            source: Some(DataSourceType::CelesTrak),
            quality: None,
        };

        let hash: ActionHash = conductor
            .call(
                &cell.zome("orbital_objects_coordinator"),
                "submit_tle",
                input,
            )
            .await;

        let record = get_record(&conductor, hash).await;
        assert!(record.is_some(), "Should create TLE record");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_claim_operator() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = ClaimOperatorInput {
            norad_id: 25544,
            organization: "NASA/Roscosmos".to_string(),
            contact: Some("ops@iss.example.com".to_string()),
            verification_hash: None,
        };

        let hash: ActionHash = conductor
            .call(
                &cell.zome("orbital_objects_coordinator"),
                "claim_operator",
                input,
            )
            .await;

        let record = get_record(&conductor, hash).await;
        assert!(record.is_some(), "Should create operator claim");
    }
}

// Helper to get record by hash
async fn get_record(conductor: &SweetConductor, hash: ActionHash) -> Option<Record> {
    conductor
        .get_from_stores(hash.into(), GetOptions::default())
        .await
        .ok()
        .flatten()
}

// ============================================================================
// Conjunctions Tests
// ============================================================================

#[cfg(test)]
mod conjunctions_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_conjunction_event() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = CreateEventInput {
            event_id: "conj:25544:40075:2026-02-01".to_string(),
            primary_norad_id: 25544,
            secondary_norad_id: 40075,
            tca: SpaceTimestamp::now(),
            miss_distance_km: 0.5,
            max_pc: 0.001,
            risk_level: RiskLevel::Medium,
            compute_details: false,
            primary_tle: None,
            secondary_tle: None,
        };

        let hash: ActionHash = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "create_conjunction_event",
                input,
            )
            .await;

        let record = get_record(&conductor, hash).await;
        assert!(record.is_some(), "Should create conjunction event");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_announce_maneuver() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = AnnounceManeuverInput {
            event_id: "conj:test-maneuver".to_string(),
            norad_id: 25544,
            burn_time: SpaceTimestamp::now(),
            delta_v_ms: 0.5,
            direction: [0.0, 0.0, 1.0],
        };

        let hash: ActionHash = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "announce_maneuver",
                input,
            )
            .await;

        let record = get_record(&conductor, hash).await;
        assert!(record.is_some(), "Should create maneuver record");
    }
}

// ============================================================================
// Observations Tests
// ============================================================================

#[cfg(test)]
mod observations_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_submit_observation() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = SubmitObservationInput {
            norad_id: Some(25544),
            observation_time: SpaceTimestamp::now(),
            observer_location: Some(GroundLocation {
                latitude: 32.95,
                longitude: -96.73,
                altitude_m: 200.0,
            }),
            observation_type: ObservationType::Optical,
            measurement: Measurement {
                azimuth_deg: Some(180.0),
                elevation_deg: Some(45.0),
                range_km: None,
                range_rate_kms: None,
                visual_magnitude: Some(-2.5),
            },
            quality: None,
            sensor_id: "sensor:texas-optical-01".to_string(),
        };

        let hash: ActionHash = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "submit_observation",
                input,
            )
            .await;

        let record = get_record(&conductor, hash).await;
        assert!(record.is_some(), "Should create observation record");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_register_sensor() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = RegisterSensorInput {
            sensor_id: "sensor:test-radar-01".to_string(),
            name: "Test Radar Station".to_string(),
            sensor_type: ObservationType::Radar,
            location: Some(GroundLocation {
                latitude: 34.0,
                longitude: -118.0,
                altitude_m: 500.0,
            }),
            capabilities: SensorCapabilities {
                min_elevation_deg: 5.0,
                max_range_km: 40000.0,
                accuracy_arcsec: 10.0,
            },
        };

        let hash: ActionHash = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "register_sensor",
                input,
            )
            .await;

        let record = get_record(&conductor, hash).await;
        assert!(record.is_some(), "Should create sensor record");
    }
}

// ============================================================================
// Debris Bounties Tests
// ============================================================================

#[cfg(test)]
mod debris_bounties_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_and_claim_bounty() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create bounty
        let bounty_input = CreateBountyInput {
            bounty_id: "bounty:debris-99999".to_string(),
            debris_norad_id: 99999,
            justification: "High-risk debris in congested orbit".to_string(),
            amount: 500000,
            currency: "USD".to_string(),
            expires_at: None,
            requirements: RemovalRequirements {
                method_preference: Some("ActiveDebrisRemoval".to_string()),
                deadline_days: Some(365),
                minimum_verification: 3,
            },
        };

        let bounty_hash: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "create_bounty",
                bounty_input,
            )
            .await;

        let bounty_record = get_record(&conductor, bounty_hash).await;
        assert!(bounty_record.is_some(), "Should create bounty");

        // Contribute to bounty
        let contrib_input = ContributeInput {
            bounty_id: "bounty:debris-99999".to_string(),
            amount: 100000,
            currency: "USD".to_string(),
            message: Some("Matching contribution from space agency".to_string()),
        };

        let contrib_hash: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "contribute_to_bounty",
                contrib_input,
            )
            .await;

        let contrib_record = get_record(&conductor, contrib_hash).await;
        assert!(contrib_record.is_some(), "Should create contribution");

        // Claim bounty
        let claim_input = ClaimBountyInput {
            bounty_id: "bounty:debris-99999".to_string(),
            organization: "Astroscale".to_string(),
            method: RemovalMethod::NetCapture,
            estimated_completion: SpaceTimestamp::now(),
            mission_plan: "Deploy ADRAS-J successor mission".to_string(),
        };

        let claim_hash: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "claim_bounty",
                claim_input,
            )
            .await;

        let claim_record = get_record(&conductor, claim_hash).await;
        assert!(claim_record.is_some(), "Should create removal claim");
    }
}

// ============================================================================
// Traffic Control Tests
// ============================================================================

#[cfg(test)]
mod traffic_control_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_negotiation_flow() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Set up a second agent for the negotiation
        let app2 = conductor
            .setup_app("test-app-2", &[dna.clone()])
            .await
            .unwrap();
        let agent2 = app2.agent().clone();

        // Initiate negotiation
        let neg_input = InitiateNegotiationInput {
            session_id: "session:conj-001".to_string(),
            conjunction_id: "conj:25544:40075".to_string(),
            primary_operator: agent.clone(),
            secondary_operator: agent2.clone(),
            primary_norad_id: 25544,
            secondary_norad_id: 40075,
            tca: SpaceTimestamp::now(),
            deadline: SpaceTimestamp::now(),
        };

        let neg_hash: ActionHash = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "initiate_negotiation",
                neg_input,
            )
            .await;

        let neg_record = get_record(&conductor, neg_hash).await;
        assert!(neg_record.is_some(), "Should create negotiation session");

        // Submit position
        let pos_input = SubmitPositionInput {
            session_id: "session:conj-001".to_string(),
            norad_id: 25544,
            maneuver_capability: ManeuverCapability {
                max_delta_v_ms: 5.0,
                min_lead_time_hours: 24.0,
                fuel_remaining_pct: 80.0,
            },
            preferences: OperatorPreferences {
                willing_to_maneuver: true,
                max_cost_usd: Some(50000.0),
                preferred_direction: None,
            },
        };

        let pos_hash: ActionHash = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "submit_position",
                pos_input,
            )
            .await;

        let pos_record = get_record(&conductor, pos_hash).await;
        assert!(pos_record.is_some(), "Should create negotiation position");

        // Submit proposal
        let prop_input = SubmitProposalInput {
            session_id: "session:conj-001".to_string(),
            maneuvering_object: 25544,
            burn_time: SpaceTimestamp::now(),
            delta_v_ms: 0.3,
            direction: [0.0, 1.0, 0.0],
            resulting_miss_km: 5.0,
            resulting_pc: 1e-7,
            cost_estimate: Some(CostEstimate {
                fuel_cost_usd: 5000.0,
                operational_cost_usd: 10000.0,
                opportunity_cost_usd: 2000.0,
            }),
        };

        let prop_hash: ActionHash = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "submit_proposal",
                prop_input,
            )
            .await;

        let prop_record = get_record(&conductor, prop_hash).await;
        assert!(prop_record.is_some(), "Should create maneuver proposal");
    }
}

// ============================================================================
// Full Lifecycle Test
// ============================================================================

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_object_observation_conjunction_bounty_lifecycle() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // 1. Register orbital objects
        let obj1 = RegisterObjectInput {
            norad_id: 50001,
            intl_designator: "2024-001A".to_string(),
            name: "Test Satellite Alpha".to_string(),
            object_type: ObjectType::Payload,
            country: Some("US".to_string()),
            launch_date: None,
            status: Some(OperationalStatus::Active),
        };

        let _: ActionHash = conductor
            .call(
                &cell.zome("orbital_objects_coordinator"),
                "register_object",
                obj1,
            )
            .await;

        let obj2 = RegisterObjectInput {
            norad_id: 50002,
            intl_designator: "2020-999Z".to_string(),
            name: "Derelict Debris".to_string(),
            object_type: ObjectType::Debris,
            country: None,
            launch_date: None,
            status: Some(OperationalStatus::Inactive),
        };

        let _: ActionHash = conductor
            .call(
                &cell.zome("orbital_objects_coordinator"),
                "register_object",
                obj2,
            )
            .await;

        // 2. Submit observation
        let obs_input = SubmitObservationInput {
            norad_id: Some(50001),
            observation_time: SpaceTimestamp::now(),
            observer_location: Some(GroundLocation {
                latitude: 28.5,
                longitude: -80.6,
                altitude_m: 10.0,
            }),
            observation_type: ObservationType::Radar,
            measurement: Measurement {
                azimuth_deg: Some(270.0),
                elevation_deg: Some(30.0),
                range_km: Some(800.0),
                range_rate_kms: Some(-2.0),
                visual_magnitude: None,
            },
            quality: None,
            sensor_id: "sensor:cape-canaveral".to_string(),
        };

        let _: ActionHash = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "submit_observation",
                obs_input,
            )
            .await;

        // 3. Create conjunction event
        let conj_input = CreateEventInput {
            event_id: "conj:50001:50002:lifecycle".to_string(),
            primary_norad_id: 50001,
            secondary_norad_id: 50002,
            tca: SpaceTimestamp::now(),
            miss_distance_km: 0.2,
            max_pc: 0.01,
            risk_level: RiskLevel::High,
            compute_details: false,
            primary_tle: None,
            secondary_tle: None,
        };

        let _: ActionHash = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "create_conjunction_event",
                conj_input,
            )
            .await;

        // 4. Create debris bounty for the problematic debris
        let bounty_input = CreateBountyInput {
            bounty_id: "bounty:50002-lifecycle".to_string(),
            debris_norad_id: 50002,
            justification: "Repeated close approaches with active satellites".to_string(),
            amount: 250000,
            currency: "USD".to_string(),
            expires_at: None,
            requirements: RemovalRequirements {
                method_preference: None,
                deadline_days: Some(730),
                minimum_verification: 2,
            },
        };

        let bounty_hash: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "create_bounty",
                bounty_input,
            )
            .await;

        let bounty_record = get_record(&conductor, bounty_hash).await;
        assert!(bounty_record.is_some(), "Lifecycle should produce bounty");
    }
}
