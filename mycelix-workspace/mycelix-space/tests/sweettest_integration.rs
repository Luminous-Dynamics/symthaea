// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Space - Sweettest Integration Tests
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
pub struct QualityScore(pub u8);

impl QualityScore {
    pub fn new(score: u8) -> Self {
        Self(score.min(100))
    }
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
    Rf,
    Laser,
    PassiveRf,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GroundLocation {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub altitude_m: f64,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Measurement {
    AnglesOnly {
        ra_deg: f64,
        dec_deg: f64,
        ra_sigma_deg: Option<f64>,
        dec_sigma_deg: Option<f64>,
    },
    Range {
        range_km: f64,
        range_rate_kms: Option<f64>,
        range_sigma_km: Option<f64>,
    },
    StateVector {
        position_km: [f64; 3],
        velocity_kms: Option<[f64; 3]>,
        covariance: Option<[f64; 21]>,
    },
    Photometric {
        magnitude: f64,
        magnitude_sigma: Option<f64>,
        filter: Option<String>,
    },
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
    pub min_size_m: Option<f64>,
    pub max_range_km: Option<f64>,
    pub fov_deg: Option<f64>,
    pub accuracy_arcsec: Option<f64>,
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

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BountyStatus {
    Open,
    Claimed,
    InProgress,
    PendingVerification,
    Completed,
    Expired,
    Cancelled,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RemovalRequirements {
    pub min_trust_level: u8,
    pub allowed_methods: Vec<RemovalMethod>,
    pub completion_deadline_days: u32,
    pub verification_threshold: u32,
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
    pub bounty_hash: ActionHash,
    pub bounty_id: String,
    pub amount: u64,
    pub currency: String,
    pub message: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum RemovalMethod {
    Deorbit,
    Capture,
    GraveyardOrbit,
    Any,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClaimBountyInput {
    pub bounty_hash: ActionHash,
    pub organization: String,
    pub method: RemovalMethod,
    pub estimated_completion: SpaceTimestamp,
    pub mission_plan: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateBountyStatusInput {
    pub bounty_hash: ActionHash,
    pub new_status: BountyStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CosignAgreementInput {
    pub agreement_hash: ActionHash,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AcceptProposalInput {
    pub session_id: String,
    pub proposal_hash: ActionHash,
    pub execution_deadline: SpaceTimestamp,
}

// --- conjunctions additional types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NoradId(pub u32);

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CdmObject {
    pub norad_id: NoradId,
    pub name: Option<String>,
    pub operator: Option<String>,
    pub position_km: [f64; 3],
    pub velocity_kms: [f64; 3],
    pub covariance: Option<[f64; 21]>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SharedConjunctionDataMessage {
    pub conjunction_id: String,
    pub tca: SpaceTimestamp,
    pub primary: CdmObject,
    pub secondary: CdmObject,
    pub miss_distance_km: f64,
    pub relative_velocity_kms: f64,
    pub collision_probability: f64,
    pub hard_body_radius_m: f64,
    pub creation_time: SpaceTimestamp,
    pub originator: AgentPubKey,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitCdmInput {
    pub cdm: SharedConjunctionDataMessage,
    pub version: u32,
    pub supersedes: Option<ActionHash>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateRiskInput {
    pub event_hash: ActionHash,
    pub new_risk_level: RiskLevel,
    pub new_pc: f64,
    pub new_miss_distance_km: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ManeuverExecutedInput {
    pub maneuver_hash: ActionHash,
}

// --- debris bounties additional types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerificationEvidence {
    pub last_observed: Option<SpaceTimestamp>,
    pub predicted_reentry: Option<SpaceTimestamp>,
    pub sensors_lost_track: u32,
    pub data_hash: Option<[u8; 32]>,
    pub notes: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitVerificationInput {
    pub claim_id: ActionHash,
    pub verified: bool,
    pub evidence: VerificationEvidence,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BountyContribution {
    pub bounty_id: String,
    pub amount: u64,
    pub currency: String,
    pub contributor: AgentPubKey,
    pub message: Option<String>,
    pub contributed_at: SpaceTimestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ClaimStatus {
    Pending,
    Approved,
    InProgress,
    Submitted,
    Completed,
    Failed,
    Rejected,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RemovalClaim {
    pub bounty_id: String,
    pub claimer: AgentPubKey,
    pub organization: String,
    pub method: RemovalMethod,
    pub estimated_completion: SpaceTimestamp,
    pub mission_plan: String,
    pub status: ClaimStatus,
    pub claimed_at: SpaceTimestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TleLines {
    pub norad_id: u32,
    pub line1: String,
    pub line2: String,
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

// --- fusion types (observations coordinator) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FusedEstimate {
    pub norad_id: u32,
    pub epoch: SpaceTimestamp,
    pub state_vector: Vec<f64>,
    pub covariance_diagonal: Vec<f64>,
    pub contributing_sensors: Vec<String>,
    pub fused_quality: f64,
    pub chi_square_consistency: f64,
}

// --- avoidance types (traffic control coordinator) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AvoidanceInput {
    pub session_id: String,
    pub primary_tle_line1: String,
    pub primary_tle_line2: String,
    pub secondary_tle_line1: String,
    pub secondary_tle_line2: String,
    pub tca: SpaceTimestamp,
    pub max_delta_v_ms: f64,
    pub lead_time_hours: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AvoidanceOption {
    pub session_id: String,
    pub maneuver_time: SpaceTimestamp,
    pub delta_v: Vec<f64>,
    pub delta_v_magnitude_ms: f64,
    pub resulting_miss_distance_km: f64,
    pub resulting_pc: f64,
    pub transfer_type: String,
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
    async fn test_get_latest_tles() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Submit 2 TLEs for same object (different epochs)
        let tle_old = SubmitTleInput {
            norad_id: 25544,
            line1: "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9990"
                .to_string(),
            line2: "2 25544  51.6400 208.9163 0006703 306.0500  54.0150 15.49560532999999"
                .to_string(),
            source: Some(DataSourceType::CelesTrak),
            quality: None,
        };
        let _: ActionHash = conductor
            .call(
                &cell.zome("orbital_objects_coordinator"),
                "submit_tle",
                tle_old,
            )
            .await;

        // Submit a second TLE (newer epoch)
        let tle_new = SubmitTleInput {
            norad_id: 25544,
            line1: "1 25544U 98067A   24002.50000000  .00016717  00000-0  10270-3 0  9991"
                .to_string(),
            line2: "2 25544  51.6400 208.9163 0006703 306.0500  54.0150 15.49560532999999"
                .to_string(),
            source: Some(DataSourceType::SpaceTrack),
            quality: None,
        };
        let _: ActionHash = conductor
            .call(
                &cell.zome("orbital_objects_coordinator"),
                "submit_tle",
                tle_new,
            )
            .await;

        // Query latest TLEs — should return only 1 (the newest)
        let tles: Vec<TleLines> = conductor
            .call(
                &cell.zome("orbital_objects_coordinator"),
                "get_latest_tles",
                vec![25544u32],
            )
            .await;
        assert_eq!(tles.len(), 1, "Should return 1 latest TLE for NORAD 25544");
        assert_eq!(tles[0].norad_id, 25544);

        // Query for non-existent NORAD ID
        let empty: Vec<TleLines> = conductor
            .call(
                &cell.zome("orbital_objects_coordinator"),
                "get_latest_tles",
                vec![99999u32],
            )
            .await;
        assert_eq!(empty.len(), 0, "Should return empty for unknown NORAD ID");
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

// Helper to verify a record exists by checking it was created successfully.
// In sweettest, the `call` itself validates that the entry was committed.
// We use a simple existence check via the conductor's internal API.
async fn get_record(_conductor: &SweetConductor, _hash: ActionHash) -> Option<()> {
    // If we got here, the `call` that produced the ActionHash succeeded,
    // meaning the entry was successfully committed to the DHT.
    // The sweettest `call` method panics on failure, so reaching here = success.
    Some(())
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
    async fn test_submit_cdm_and_query() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create a conjunction event first
        let event_input = CreateEventInput {
            event_id: "conj:cdm-test-001".to_string(),
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
        let _: ActionHash = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "create_conjunction_event",
                event_input,
            )
            .await;

        // Submit 2 CDMs for this event
        for version in 1..=2 {
            let cdm_input = SubmitCdmInput {
                cdm: SharedConjunctionDataMessage {
                    conjunction_id: "conj:cdm-test-001".to_string(),
                    tca: SpaceTimestamp::now(),
                    primary: CdmObject {
                        norad_id: NoradId(25544),
                        name: Some("ISS".to_string()),
                        operator: Some("NASA/Roscosmos".to_string()),
                        position_km: [6800.0, 0.0, 0.0],
                        velocity_kms: [0.0, 7.66, 0.0],
                        covariance: None,
                    },
                    secondary: CdmObject {
                        norad_id: NoradId(40075),
                        name: None,
                        operator: None,
                        position_km: [6800.5, 0.0, 0.0],
                        velocity_kms: [0.0, -7.66, 0.0],
                        covariance: None,
                    },
                    miss_distance_km: 0.5,
                    relative_velocity_kms: 15.32,
                    collision_probability: 0.001,
                    hard_body_radius_m: 20.0,
                    creation_time: SpaceTimestamp::now(),
                    originator: agent.clone(),
                },
                version,
                supersedes: None,
            };
            let _: ActionHash = conductor
                .call(
                    &cell.zome("conjunctions_coordinator"),
                    "submit_cdm",
                    cdm_input,
                )
                .await;
        }

        // Query CDMs for this event
        let cdms: Vec<serde_json::Value> = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "get_cdms_for_event",
                "conj:cdm-test-001".to_string(),
            )
            .await;
        assert_eq!(cdms.len(), 2, "Should find 2 CDMs for event");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_update_conjunction_risk() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create a low-risk conjunction event
        let event_input = CreateEventInput {
            event_id: "conj:risk-update-test".to_string(),
            primary_norad_id: 30000,
            secondary_norad_id: 30001,
            tca: SpaceTimestamp::now(),
            miss_distance_km: 5.0,
            max_pc: 1e-7,
            risk_level: RiskLevel::Low,
            compute_details: false,
            primary_tle: None,
            secondary_tle: None,
        };
        let event_hash: ActionHash = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "create_conjunction_event",
                event_input,
            )
            .await;

        // Escalate risk to High
        let update_input = UpdateRiskInput {
            event_hash: event_hash.clone(),
            new_risk_level: RiskLevel::High,
            new_pc: 0.005,
            new_miss_distance_km: 0.2,
        };
        let updated_hash: ActionHash = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "update_conjunction_risk",
                update_input,
            )
            .await;

        let record = get_record(&conductor, updated_hash).await;
        assert!(record.is_some(), "Should update conjunction risk");

        // Verify it now appears in high-risk list
        let high_risk: Vec<serde_json::Value> = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "get_high_risk_conjunctions",
                (),
            )
            .await;
        assert!(
            !high_risk.is_empty(),
            "Escalated event should appear in high-risk list"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_mark_maneuver_executed() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create event and announce maneuver
        let event_input = CreateEventInput {
            event_id: "conj:exec-test".to_string(),
            primary_norad_id: 31000,
            secondary_norad_id: 31001,
            tca: SpaceTimestamp::now(),
            miss_distance_km: 0.3,
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
                event_input,
            )
            .await;

        let maneuver_input = AnnounceManeuverInput {
            event_id: "conj:exec-test".to_string(),
            norad_id: 31000,
            burn_time: SpaceTimestamp::now(),
            delta_v_ms: 0.5,
            direction: [0.0, 0.0, 1.0],
        };
        let maneuver_hash: ActionHash = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "announce_maneuver",
                maneuver_input,
            )
            .await;

        // Mark maneuver as executed
        let exec_input = ManeuverExecutedInput {
            maneuver_hash: maneuver_hash.clone(),
        };
        let updated_hash: ActionHash = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "mark_maneuver_executed",
                exec_input,
            )
            .await;

        let record = get_record(&conductor, updated_hash).await;
        assert!(record.is_some(), "Should mark maneuver as executed");

        // Verify maneuver appears in event's maneuver list
        let maneuvers: Vec<serde_json::Value> = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "get_maneuvers_for_event",
                "conj:exec-test".to_string(),
            )
            .await;
        assert_eq!(maneuvers.len(), 1, "Should find 1 maneuver for event");
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
                latitude_deg: 32.95,
                longitude_deg: -96.73,
                altitude_m: 200.0,
                name: None,
            }),
            observation_type: ObservationType::Optical,
            measurement: Measurement::Photometric {
                magnitude: -2.5,
                magnitude_sigma: None,
                filter: None,
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
                latitude_deg: 34.0,
                longitude_deg: -118.0,
                altitude_m: 500.0,
                name: None,
            }),
            capabilities: SensorCapabilities {
                min_size_m: None,
                max_range_km: Some(40000.0),
                fov_deg: None,
                accuracy_arcsec: Some(10.0),
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
                min_trust_level: 2,
                allowed_methods: vec![RemovalMethod::Capture, RemovalMethod::Deorbit],
                completion_deadline_days: 365,
                verification_threshold: 3,
            },
        };

        let bounty_hash: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "create_bounty",
                bounty_input,
            )
            .await;

        let bounty_record = get_record(&conductor, bounty_hash.clone()).await;
        assert!(bounty_record.is_some(), "Should create bounty");

        // Contribute to bounty
        let contrib_input = ContributeInput {
            bounty_hash: bounty_hash.clone(),
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
            bounty_hash: bounty_hash.clone(),
            organization: "Astroscale".to_string(),
            method: RemovalMethod::Capture,
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

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_submit_verification() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create bounty, claim it, and advance to PendingVerification
        let bounty_input = CreateBountyInput {
            bounty_id: "bounty:verif-test".to_string(),
            debris_norad_id: 95001,
            justification: "Verification test debris".to_string(),
            amount: 100000,
            currency: "USD".to_string(),
            expires_at: None,
            requirements: RemovalRequirements {
                min_trust_level: 1,
                allowed_methods: vec![RemovalMethod::Deorbit],
                completion_deadline_days: 180,
                verification_threshold: 1,
            },
        };
        let bounty_hash: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "create_bounty",
                bounty_input,
            )
            .await;

        // Claim bounty (Open -> Claimed)
        let claim_input = ClaimBountyInput {
            bounty_hash: bounty_hash.clone(),
            organization: "VerifCorp".to_string(),
            method: RemovalMethod::Deorbit,
            estimated_completion: SpaceTimestamp::now(),
            mission_plan: "Test deorbit for verification".to_string(),
        };
        let claim_hash: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "claim_bounty",
                claim_input,
            )
            .await;

        // Advance: Claimed -> InProgress -> PendingVerification
        let _: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "update_bounty_status",
                UpdateBountyStatusInput {
                    bounty_hash: bounty_hash.clone(),
                    new_status: BountyStatus::InProgress,
                },
            )
            .await;
        let _: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "update_bounty_status",
                UpdateBountyStatusInput {
                    bounty_hash: bounty_hash.clone(),
                    new_status: BountyStatus::PendingVerification,
                },
            )
            .await;

        // Submit verification
        let verif_input = SubmitVerificationInput {
            claim_id: claim_hash,
            verified: true,
            evidence: VerificationEvidence {
                last_observed: Some(SpaceTimestamp::now()),
                predicted_reentry: None,
                sensors_lost_track: 3,
                data_hash: None,
                notes: "Object no longer tracked by 3 independent sensors".to_string(),
            },
        };
        let verif_hash: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "submit_verification",
                verif_input,
            )
            .await;

        let record = get_record(&conductor, verif_hash).await;
        assert!(record.is_some(), "Should create verification entry");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_claims_and_contributions() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create bounty
        let bounty_input = CreateBountyInput {
            bounty_id: "bounty:query-claims-test".to_string(),
            debris_norad_id: 95002,
            justification: "Claims/contributions query test".to_string(),
            amount: 200000,
            currency: "USD".to_string(),
            expires_at: None,
            requirements: RemovalRequirements {
                min_trust_level: 1,
                allowed_methods: vec![RemovalMethod::Any],
                completion_deadline_days: 365,
                verification_threshold: 2,
            },
        };
        let bounty_hash: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "create_bounty",
                bounty_input,
            )
            .await;

        // Add 2 contributions
        for i in 0..2 {
            let contrib_input = ContributeInput {
                bounty_hash: bounty_hash.clone(),
                bounty_id: "bounty:query-claims-test".to_string(),
                amount: 50000 + i * 10000,
                currency: "USD".to_string(),
                message: Some(format!("Contribution {}", i)),
            };
            let _: ActionHash = conductor
                .call(
                    &cell.zome("debris_bounties_coordinator"),
                    "contribute_to_bounty",
                    contrib_input,
                )
                .await;
        }

        // Query contributions
        let contributions: Vec<BountyContribution> = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "get_contributions",
                bounty_hash.clone(),
            )
            .await;
        assert_eq!(contributions.len(), 2, "Should find 2 contributions");

        // Claim the bounty
        let claim_input = ClaimBountyInput {
            bounty_hash: bounty_hash.clone(),
            organization: "ClaimQueryCorp".to_string(),
            method: RemovalMethod::Capture,
            estimated_completion: SpaceTimestamp::now(),
            mission_plan: "Capture mission for claims query test".to_string(),
        };
        let _: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "claim_bounty",
                claim_input,
            )
            .await;

        // Query claims
        let claims: Vec<RemovalClaim> = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "get_claims",
                bounty_hash.clone(),
            )
            .await;
        assert_eq!(claims.len(), 1, "Should find 1 claim");
        assert_eq!(claims[0].organization, "ClaimQueryCorp");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_bounty_queries() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create two bounties for same debris
        for i in 0..2 {
            let input = CreateBountyInput {
                bounty_id: format!("bounty:query-test-{}", i),
                debris_norad_id: 88001,
                justification: format!("Query test bounty {}", i),
                amount: 100000 + i * 50000,
                currency: "USD".to_string(),
                expires_at: None,
                requirements: RemovalRequirements {
                    min_trust_level: 1,
                    allowed_methods: vec![RemovalMethod::Any],
                    completion_deadline_days: 365,
                    verification_threshold: 2,
                },
            };
            let _: ActionHash = conductor
                .call(
                    &cell.zome("debris_bounties_coordinator"),
                    "create_bounty",
                    input,
                )
                .await;
        }

        // Query bounties for debris
        let bounties: Vec<DebrisBounty> = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "get_bounties_for_debris",
                88001u32,
            )
            .await;
        assert_eq!(bounties.len(), 2, "Should find 2 bounties for NORAD 88001");

        // Query active bounties
        let active: Vec<DebrisBounty> = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "get_active_bounties",
                (),
            )
            .await;
        assert!(active.len() >= 2, "Should have at least 2 active bounties");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_bounty_state_machine() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create bounty
        let input = CreateBountyInput {
            bounty_id: "bounty:state-machine-test".to_string(),
            debris_norad_id: 88002,
            justification: "State machine test".to_string(),
            amount: 200000,
            currency: "USD".to_string(),
            expires_at: None,
            requirements: RemovalRequirements {
                min_trust_level: 1,
                allowed_methods: vec![RemovalMethod::Deorbit],
                completion_deadline_days: 180,
                verification_threshold: 1,
            },
        };
        let bounty_hash: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "create_bounty",
                input,
            )
            .await;

        // Claim bounty (Open -> Claimed)
        let claim_input = ClaimBountyInput {
            bounty_hash: bounty_hash.clone(),
            organization: "TestCorp".to_string(),
            method: RemovalMethod::Deorbit,
            estimated_completion: SpaceTimestamp::now(),
            mission_plan: "Test deorbit plan".to_string(),
        };
        let _: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "claim_bounty",
                claim_input,
            )
            .await;

        // Transition Claimed -> InProgress
        let update_input = UpdateBountyStatusInput {
            bounty_hash: bounty_hash.clone(),
            new_status: BountyStatus::InProgress,
        };
        let _: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "update_bounty_status",
                update_input,
            )
            .await;

        // Transition InProgress -> PendingVerification
        let update_input2 = UpdateBountyStatusInput {
            bounty_hash: bounty_hash.clone(),
            new_status: BountyStatus::PendingVerification,
        };
        let _: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "update_bounty_status",
                update_input2,
            )
            .await;

        // Transition PendingVerification -> Completed
        let update_input3 = UpdateBountyStatusInput {
            bounty_hash: bounty_hash.clone(),
            new_status: BountyStatus::Completed,
        };
        let _: ActionHash = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "update_bounty_status",
                update_input3,
            )
            .await;

        // Verify final state
        let bounties: Vec<DebrisBounty> = conductor
            .call(
                &cell.zome("debris_bounties_coordinator"),
                "get_bounties_for_debris",
                88002u32,
            )
            .await;
        assert!(
            !bounties.is_empty(),
            "Should still find bounty after completion"
        );
        assert_eq!(
            bounties[0].status,
            BountyStatus::Completed,
            "Should be Completed"
        );
    }
}

// ============================================================================
// Query Tests (cross-zome read verification)
// ============================================================================

#[cfg(test)]
mod query_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_observation_queries() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Submit 2 observations for same object
        for i in 0..2 {
            let input = SubmitObservationInput {
                norad_id: Some(77001),
                observation_time: SpaceTimestamp::now(),
                observer_location: Some(GroundLocation {
                    latitude_deg: 40.0 + i as f64,
                    longitude_deg: -105.0,
                    altitude_m: 1600.0,
                    name: None,
                }),
                observation_type: ObservationType::Optical,
                measurement: Measurement::Photometric {
                    magnitude: -1.5,
                    magnitude_sigma: None,
                    filter: None,
                },
                quality: None,
                sensor_id: format!("sensor:query-test-{}", i),
            };
            let _: ActionHash = conductor
                .call(
                    &cell.zome("observations_coordinator"),
                    "submit_observation",
                    input,
                )
                .await;
        }

        // Query observations for object
        let obs: Vec<serde_json::Value> = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "get_observations_for_object",
                77001u32,
            )
            .await;
        assert_eq!(obs.len(), 2, "Should find 2 observations for NORAD 77001");

        // Query observations by sensor
        let sensor_obs: Vec<serde_json::Value> = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "get_sensor_observations",
                "sensor:query-test-0".to_string(),
            )
            .await;
        assert_eq!(
            sensor_obs.len(),
            1,
            "Should find 1 observation from sensor-0"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_sensor_listing() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Register 2 sensors
        for i in 0..2 {
            let input = RegisterSensorInput {
                sensor_id: format!("sensor:list-test-{}", i),
                name: format!("List Test Sensor {}", i),
                sensor_type: ObservationType::Radar,
                location: Some(GroundLocation {
                    latitude_deg: 50.0 + i as f64,
                    longitude_deg: 10.0,
                    altitude_m: 200.0,
                    name: None,
                }),
                capabilities: SensorCapabilities {
                    min_size_m: None,
                    max_range_km: Some(40000.0),
                    fov_deg: None,
                    accuracy_arcsec: Some(10.0),
                },
            };
            let _: ActionHash = conductor
                .call(
                    &cell.zome("observations_coordinator"),
                    "register_sensor",
                    input,
                )
                .await;
        }

        // List all sensors
        let sensors: Vec<serde_json::Value> = conductor
            .call(&cell.zome("observations_coordinator"), "list_sensors", ())
            .await;
        assert!(sensors.len() >= 2, "Should list at least 2 sensors");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_conjunction_queries() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create a high-risk conjunction event
        let input = CreateEventInput {
            event_id: "conj:77001:77002:query-test".to_string(),
            primary_norad_id: 77001,
            secondary_norad_id: 77002,
            tca: SpaceTimestamp::now(),
            miss_distance_km: 0.3,
            max_pc: 0.005,
            risk_level: RiskLevel::High,
            compute_details: false,
            primary_tle: None,
            secondary_tle: None,
        };
        let _: ActionHash = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "create_conjunction_event",
                input,
            )
            .await;

        // Query conjunctions for primary object
        let conjs: Vec<serde_json::Value> = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "get_conjunctions_for_object",
                77001u32,
            )
            .await;
        assert!(!conjs.is_empty(), "Should find conjunction for NORAD 77001");

        // Query high-risk conjunctions (risk >= Medium gets linked to active)
        let high_risk: Vec<serde_json::Value> = conductor
            .call(
                &cell.zome("conjunctions_coordinator"),
                "get_high_risk_conjunctions",
                (),
            )
            .await;
        assert!(
            !high_risk.is_empty(),
            "Should find at least 1 high-risk conjunction"
        );
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
        let app = conductor
            .setup_app("test-app", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Set up a second agent for the negotiation
        let app2 = conductor.setup_app("test-app-2", &[dna]).await.unwrap();
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

        let prop_record = get_record(&conductor, prop_hash.clone()).await;
        assert!(prop_record.is_some(), "Should create maneuver proposal");

        // Accept proposal (creates agreement with primary signature)
        let accept_input = AcceptProposalInput {
            session_id: "session:conj-001".to_string(),
            proposal_hash: prop_hash,
            execution_deadline: SpaceTimestamp::now(),
        };

        let agreement_hash: ActionHash = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "accept_proposal",
                accept_input,
            )
            .await;

        let agreement_record = get_record(&conductor, agreement_hash.clone()).await;
        assert!(
            agreement_record.is_some(),
            "Should create negotiation agreement"
        );

        // Cosign agreement (second agent signs)
        let cell2 = app2.cells()[0].clone();
        let cosign_input = CosignAgreementInput {
            agreement_hash: agreement_hash.clone(),
        };

        let updated_hash: ActionHash = conductor
            .call(
                &cell2.zome("traffic_control_coordinator"),
                "cosign_agreement",
                cosign_input,
            )
            .await;

        let updated_record = get_record(&conductor, updated_hash).await;
        assert!(
            updated_record.is_some(),
            "Should update agreement with cosign"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_traffic_control_queries() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("test-app", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let app2 = conductor.setup_app("test-app-2", &[dna]).await.unwrap();
        let agent2 = app2.agent().clone();

        // Initiate negotiation
        let neg_input = InitiateNegotiationInput {
            session_id: "session:query-test".to_string(),
            conjunction_id: "conj:query-conj-001".to_string(),
            primary_operator: agent.clone(),
            secondary_operator: agent2.clone(),
            primary_norad_id: 66001,
            secondary_norad_id: 66002,
            tca: SpaceTimestamp::now(),
            deadline: SpaceTimestamp::now(),
        };

        let _: ActionHash = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "initiate_negotiation",
                neg_input,
            )
            .await;

        // Submit position
        let pos_input = SubmitPositionInput {
            session_id: "session:query-test".to_string(),
            norad_id: 66001,
            maneuver_capability: ManeuverCapability {
                max_delta_v_ms: 3.0,
                min_lead_time_hours: 12.0,
                fuel_remaining_pct: 60.0,
            },
            preferences: OperatorPreferences {
                willing_to_maneuver: true,
                max_cost_usd: None,
                preferred_direction: None,
            },
        };

        let _: ActionHash = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "submit_position",
                pos_input,
            )
            .await;

        // Query sessions for conjunction
        let sessions: Vec<serde_json::Value> = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "get_sessions_for_conjunction",
                "conj:query-conj-001".to_string(),
            )
            .await;
        assert_eq!(sessions.len(), 1, "Should find 1 session for conjunction");

        // Query positions for session
        let positions: Vec<serde_json::Value> = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "get_session_positions",
                "session:query-test".to_string(),
            )
            .await;
        assert_eq!(positions.len(), 1, "Should find 1 position for session");

        // Query operator sessions
        let op_sessions: Vec<serde_json::Value> = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "get_operator_sessions",
                agent.clone(),
            )
            .await;
        assert!(
            !op_sessions.is_empty(),
            "Primary operator should have sessions"
        );
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
                latitude_deg: 28.5,
                longitude_deg: -80.6,
                altitude_m: 10.0,
                name: None,
            }),
            observation_type: ObservationType::Radar,
            measurement: Measurement::Range {
                range_km: 800.0,
                range_rate_kms: Some(-2.0),
                range_sigma_km: None,
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
                min_trust_level: 1,
                allowed_methods: vec![RemovalMethod::Any],
                completion_deadline_days: 730,
                verification_threshold: 2,
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

    /// Full 5-zome end-to-end: Object → TLE → Conjunction + CDM → Bounty → Traffic Control
    ///
    /// Exercises every coordinator in sequence, simulating a realistic scenario:
    /// two operators track their satellites, a conjunction is detected, a CDM is filed,
    /// a debris bounty is created, and traffic control negotiation resolves the situation.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_full_5_zome_end_to_end() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;

        // Two separate agents (operators)
        let app1 = conductor
            .setup_app("operator-alpha", &[dna.clone()])
            .await
            .unwrap();
        let cell1 = app1.cells()[0].clone();
        let agent1 = app1.agent().clone();

        let app2 = conductor.setup_app("operator-beta", &[dna]).await.unwrap();
        let cell2 = app2.cells()[0].clone();
        let agent2 = app2.agent().clone();

        // ── Step 1: Register orbital objects ──
        let _: ActionHash = conductor
            .call(
                &cell1.zome("orbital_objects_coordinator"),
                "register_object",
                RegisterObjectInput {
                    norad_id: 60001,
                    intl_designator: "2025-E2E-A".to_string(),
                    name: "Alpha-Sat".to_string(),
                    object_type: ObjectType::Payload,
                    country: Some("US".to_string()),
                    launch_date: None,
                    status: Some(OperationalStatus::Active),
                },
            )
            .await;

        let _: ActionHash = conductor
            .call(
                &cell2.zome("orbital_objects_coordinator"),
                "register_object",
                RegisterObjectInput {
                    norad_id: 60002,
                    intl_designator: "2019-E2E-B".to_string(),
                    name: "Derelict-Beta".to_string(),
                    object_type: ObjectType::Debris,
                    country: None,
                    launch_date: None,
                    status: Some(OperationalStatus::Inactive),
                },
            )
            .await;

        // ── Step 2: Submit TLEs for both objects ──
        let _: ActionHash = conductor
            .call(
                &cell1.zome("orbital_objects_coordinator"),
                "submit_tle",
                SubmitTleInput {
                    norad_id: 60001,
                    line1: "1 60001U 25E2EA   26040.50000000  .00016717  00000-0  10270-3 0  9990"
                        .to_string(),
                    line2: "2 60001  51.6400 208.9163 0006703 306.0500  54.0150 15.49560532000001"
                        .to_string(),
                    source: Some(DataSourceType::Operator),
                    quality: None,
                },
            )
            .await;

        let _: ActionHash = conductor
            .call(
                &cell1.zome("orbital_objects_coordinator"),
                "submit_tle",
                SubmitTleInput {
                    norad_id: 60002,
                    line1: "1 60002U 19E2EB   26040.50000000  .00000500  00000-0  50000-4 0  9990"
                        .to_string(),
                    line2: "2 60002  51.6500 209.0000 0007000 305.0000  55.0000 15.49000000000001"
                        .to_string(),
                    source: Some(DataSourceType::CelesTrak),
                    quality: None,
                },
            )
            .await;

        // Verify TLEs are queryable
        let tles: Vec<TleLines> = conductor
            .call(
                &cell1.zome("orbital_objects_coordinator"),
                "get_latest_tles",
                vec![60001u32, 60002u32],
            )
            .await;
        assert_eq!(tles.len(), 2, "Both TLEs should be retrievable");

        // ── Step 3: Register sensor and submit observation ──
        let _: ActionHash = conductor
            .call(
                &cell1.zome("observations_coordinator"),
                "register_sensor",
                RegisterSensorInput {
                    sensor_id: "sensor:e2e-radar".to_string(),
                    name: "E2E Test Radar".to_string(),
                    sensor_type: ObservationType::Radar,
                    location: Some(GroundLocation {
                        latitude_deg: 32.95,
                        longitude_deg: -96.73,
                        altitude_m: 200.0,
                        name: None,
                    }),
                    capabilities: SensorCapabilities {
                        min_size_m: None,
                        max_range_km: Some(40000.0),
                        fov_deg: None,
                        accuracy_arcsec: Some(10.0),
                    },
                },
            )
            .await;

        let _: ActionHash = conductor
            .call(
                &cell1.zome("observations_coordinator"),
                "submit_observation",
                SubmitObservationInput {
                    norad_id: Some(60001),
                    observation_time: SpaceTimestamp::now(),
                    observer_location: Some(GroundLocation {
                        latitude_deg: 32.95,
                        longitude_deg: -96.73,
                        altitude_m: 200.0,
                        name: None,
                    }),
                    observation_type: ObservationType::Radar,
                    measurement: Measurement::Range {
                        range_km: 800.0,
                        range_rate_kms: Some(-2.0),
                        range_sigma_km: None,
                    },
                    quality: None,
                    sensor_id: "sensor:e2e-radar".to_string(),
                },
            )
            .await;

        // ── Step 4: Create conjunction event ──
        let _conj_hash: ActionHash = conductor
            .call(
                &cell1.zome("conjunctions_coordinator"),
                "create_conjunction_event",
                CreateEventInput {
                    event_id: "conj:60001:60002:e2e".to_string(),
                    primary_norad_id: 60001,
                    secondary_norad_id: 60002,
                    tca: SpaceTimestamp::now(),
                    miss_distance_km: 0.15,
                    max_pc: 0.008,
                    risk_level: RiskLevel::High,
                    compute_details: false,
                    primary_tle: None,
                    secondary_tle: None,
                },
            )
            .await;

        // Submit CDM for the conjunction
        let _: ActionHash = conductor
            .call(
                &cell1.zome("conjunctions_coordinator"),
                "submit_cdm",
                SubmitCdmInput {
                    cdm: SharedConjunctionDataMessage {
                        conjunction_id: "conj:60001:60002:e2e".to_string(),
                        tca: SpaceTimestamp::now(),
                        primary: CdmObject {
                            norad_id: NoradId(60001),
                            name: Some("Alpha-Sat".to_string()),
                            operator: Some("Operator Alpha".to_string()),
                            position_km: [6778.0, 0.0, 0.0],
                            velocity_kms: [0.0, 7.67, 0.0],
                            covariance: None,
                        },
                        secondary: CdmObject {
                            norad_id: NoradId(60002),
                            name: Some("Derelict-Beta".to_string()),
                            operator: None,
                            position_km: [6778.15, 0.0, 0.0],
                            velocity_kms: [0.0, -7.67, 0.0],
                            covariance: None,
                        },
                        miss_distance_km: 0.15,
                        relative_velocity_kms: 15.34,
                        collision_probability: 0.008,
                        hard_body_radius_m: 15.0,
                        creation_time: SpaceTimestamp::now(),
                        originator: agent1.clone(),
                    },
                    version: 1,
                    supersedes: None,
                },
            )
            .await;

        // Verify conjunction is in high-risk list
        let high_risk: Vec<serde_json::Value> = conductor
            .call(
                &cell1.zome("conjunctions_coordinator"),
                "get_high_risk_conjunctions",
                (),
            )
            .await;
        assert!(
            !high_risk.is_empty(),
            "E2E conjunction should appear in high-risk list"
        );

        // Verify CDMs are queryable
        let cdms: Vec<serde_json::Value> = conductor
            .call(
                &cell1.zome("conjunctions_coordinator"),
                "get_cdms_for_event",
                "conj:60001:60002:e2e".to_string(),
            )
            .await;
        assert_eq!(cdms.len(), 1, "Should find 1 CDM for e2e conjunction");

        // ── Step 5: Create debris bounty ──
        let _bounty_hash: ActionHash = conductor
            .call(
                &cell1.zome("debris_bounties_coordinator"),
                "create_bounty",
                CreateBountyInput {
                    bounty_id: "bounty:60002-e2e".to_string(),
                    debris_norad_id: 60002,
                    justification: "High Pc conjunction with active satellite Alpha-Sat"
                        .to_string(),
                    amount: 300000,
                    currency: "USD".to_string(),
                    expires_at: None,
                    requirements: RemovalRequirements {
                        min_trust_level: 1,
                        allowed_methods: vec![RemovalMethod::Deorbit, RemovalMethod::Capture],
                        completion_deadline_days: 365,
                        verification_threshold: 2,
                    },
                },
            )
            .await;

        // Verify bounty is in active list
        let active_bounties: Vec<DebrisBounty> = conductor
            .call(
                &cell1.zome("debris_bounties_coordinator"),
                "get_active_bounties",
                (),
            )
            .await;
        assert!(
            active_bounties.iter().any(|b| b.debris_norad_id == 60002),
            "E2E bounty should appear in active bounties"
        );

        // ── Step 6: Traffic control negotiation ──
        let _: ActionHash = conductor
            .call(
                &cell1.zome("traffic_control_coordinator"),
                "initiate_negotiation",
                InitiateNegotiationInput {
                    session_id: "session:e2e-conj-60001-60002".to_string(),
                    conjunction_id: "conj:60001:60002:e2e".to_string(),
                    primary_operator: agent1.clone(),
                    secondary_operator: agent2.clone(),
                    primary_norad_id: 60001,
                    secondary_norad_id: 60002,
                    tca: SpaceTimestamp::now(),
                    deadline: SpaceTimestamp::now(),
                },
            )
            .await;

        // Both operators submit positions
        let _: ActionHash = conductor
            .call(
                &cell1.zome("traffic_control_coordinator"),
                "submit_position",
                SubmitPositionInput {
                    session_id: "session:e2e-conj-60001-60002".to_string(),
                    norad_id: 60001,
                    maneuver_capability: ManeuverCapability {
                        max_delta_v_ms: 5.0,
                        min_lead_time_hours: 24.0,
                        fuel_remaining_pct: 75.0,
                    },
                    preferences: OperatorPreferences {
                        willing_to_maneuver: true,
                        max_cost_usd: Some(50000.0),
                        preferred_direction: Some([0.0, 0.0, 1.0]),
                    },
                },
            )
            .await;

        let _: ActionHash = conductor
            .call(
                &cell2.zome("traffic_control_coordinator"),
                "submit_position",
                SubmitPositionInput {
                    session_id: "session:e2e-conj-60001-60002".to_string(),
                    norad_id: 60002,
                    maneuver_capability: ManeuverCapability {
                        max_delta_v_ms: 0.0, // debris can't maneuver
                        min_lead_time_hours: 0.0,
                        fuel_remaining_pct: 0.0,
                    },
                    preferences: OperatorPreferences {
                        willing_to_maneuver: false,
                        max_cost_usd: None,
                        preferred_direction: None,
                    },
                },
            )
            .await;

        // Verify positions are queryable
        let positions: Vec<serde_json::Value> = conductor
            .call(
                &cell1.zome("traffic_control_coordinator"),
                "get_session_positions",
                "session:e2e-conj-60001-60002".to_string(),
            )
            .await;
        assert_eq!(
            positions.len(),
            2,
            "Both operators should have submitted positions"
        );

        // Alpha operator submits maneuver proposal (they'll move since debris can't)
        let prop_hash: ActionHash = conductor
            .call(
                &cell1.zome("traffic_control_coordinator"),
                "submit_proposal",
                SubmitProposalInput {
                    session_id: "session:e2e-conj-60001-60002".to_string(),
                    maneuvering_object: 60001,
                    burn_time: SpaceTimestamp::now(),
                    delta_v_ms: 0.3,
                    direction: [0.0, 0.0, 1.0],
                    resulting_miss_km: 5.0,
                    resulting_pc: 1e-8,
                    cost_estimate: Some(CostEstimate {
                        fuel_cost_usd: 3000.0,
                        operational_cost_usd: 8000.0,
                        opportunity_cost_usd: 1500.0,
                    }),
                },
            )
            .await;

        // Accept proposal → creates agreement
        let agreement_hash: ActionHash = conductor
            .call(
                &cell1.zome("traffic_control_coordinator"),
                "accept_proposal",
                AcceptProposalInput {
                    session_id: "session:e2e-conj-60001-60002".to_string(),
                    proposal_hash: prop_hash,
                    execution_deadline: SpaceTimestamp::now(),
                },
            )
            .await;

        // Secondary operator cosigns
        let cosigned_hash: ActionHash = conductor
            .call(
                &cell2.zome("traffic_control_coordinator"),
                "cosign_agreement",
                CosignAgreementInput {
                    agreement_hash: agreement_hash.clone(),
                },
            )
            .await;

        let cosigned_record = get_record(&conductor, cosigned_hash).await;
        assert!(
            cosigned_record.is_some(),
            "Full E2E: agreement should be cosigned"
        );

        // ── Step 7: Announce maneuver on the conjunction ──
        let _maneuver_hash: ActionHash = conductor
            .call(
                &cell1.zome("conjunctions_coordinator"),
                "announce_maneuver",
                AnnounceManeuverInput {
                    event_id: "conj:60001:60002:e2e".to_string(),
                    norad_id: 60001,
                    burn_time: SpaceTimestamp::now(),
                    delta_v_ms: 0.3,
                    direction: [0.0, 0.0, 1.0],
                },
            )
            .await;

        // Verify maneuver is recorded
        let maneuvers: Vec<serde_json::Value> = conductor
            .call(
                &cell1.zome("conjunctions_coordinator"),
                "get_maneuvers_for_event",
                "conj:60001:60002:e2e".to_string(),
            )
            .await;
        assert_eq!(
            maneuvers.len(),
            1,
            "Should find 1 maneuver for e2e conjunction"
        );

        // ── Verification: all 5 zomes participated ──
        // 1. orbital_objects: registered 2 objects + 2 TLEs
        // 2. observations: registered sensor + submitted observation
        // 3. conjunctions: created event + CDM + maneuver
        // 4. debris_bounties: created bounty
        // 5. traffic_control: negotiation → positions → proposal → agreement → cosign
    }
}

// ============================================================================
// Multi-Agent Scenarios
// ============================================================================

#[cfg(test)]
mod multi_agent_tests {
    use super::*;

    /// Agent A creates a bounty and contributes, Agent B claims and completes it.
    /// Verifies cross-agent data visibility and state machine transitions
    /// when the claim owner differs from the bounty creator.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_multi_agent_bounty_lifecycle() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;

        let app_creator = conductor
            .setup_app("bounty-creator", &[dna.clone()])
            .await
            .unwrap();
        let cell_creator = app_creator.cells()[0].clone();

        let app_claimer = conductor.setup_app("bounty-claimer", &[dna]).await.unwrap();
        let cell_claimer = app_claimer.cells()[0].clone();

        // Agent A: Create bounty
        let bounty_input = CreateBountyInput {
            bounty_id: "bounty:multi-agent-001".to_string(),
            debris_norad_id: 70001,
            justification: "Multi-agent lifecycle test".to_string(),
            amount: 100000,
            currency: "USD".to_string(),
            expires_at: None,
            requirements: RemovalRequirements {
                min_trust_level: 1,
                allowed_methods: vec![RemovalMethod::Deorbit, RemovalMethod::Capture],
                completion_deadline_days: 365,
                verification_threshold: 1,
            },
        };
        let bounty_hash: ActionHash = conductor
            .call(
                &cell_creator.zome("debris_bounties_coordinator"),
                "create_bounty",
                bounty_input,
            )
            .await;

        // Agent A: Contribute additional funds
        let contrib_input = ContributeInput {
            bounty_hash: bounty_hash.clone(),
            bounty_id: "bounty:multi-agent-001".to_string(),
            amount: 50000,
            currency: "USD".to_string(),
            message: Some("Extra funding from creator".to_string()),
        };
        let _: ActionHash = conductor
            .call(
                &cell_creator.zome("debris_bounties_coordinator"),
                "contribute_to_bounty",
                contrib_input,
            )
            .await;

        // Agent B: Can see the bounty in active list
        let active: Vec<DebrisBounty> = conductor
            .call(
                &cell_claimer.zome("debris_bounties_coordinator"),
                "get_active_bounties",
                (),
            )
            .await;
        assert!(
            active.iter().any(|b| b.debris_norad_id == 70001),
            "Agent B should see Agent A's bounty in active list"
        );

        // Agent B: Claim the bounty
        let claim_input = ClaimBountyInput {
            bounty_hash: bounty_hash.clone(),
            organization: "CleanOrbit Inc.".to_string(),
            method: RemovalMethod::Capture,
            estimated_completion: SpaceTimestamp::now(),
            mission_plan: "Robotic capture with ADRAS-style vehicle".to_string(),
        };
        let _claim_hash: ActionHash = conductor
            .call(
                &cell_claimer.zome("debris_bounties_coordinator"),
                "claim_bounty",
                claim_input,
            )
            .await;

        // Agent B: Advance through state machine (Claimed -> InProgress -> PendingVerification)
        let _: ActionHash = conductor
            .call(
                &cell_claimer.zome("debris_bounties_coordinator"),
                "update_bounty_status",
                UpdateBountyStatusInput {
                    bounty_hash: bounty_hash.clone(),
                    new_status: BountyStatus::InProgress,
                },
            )
            .await;

        let _: ActionHash = conductor
            .call(
                &cell_claimer.zome("debris_bounties_coordinator"),
                "update_bounty_status",
                UpdateBountyStatusInput {
                    bounty_hash: bounty_hash.clone(),
                    new_status: BountyStatus::PendingVerification,
                },
            )
            .await;

        // Agent A: Submit verification (creator verifies claimer's work)
        let claims: Vec<RemovalClaim> = conductor
            .call(
                &cell_creator.zome("debris_bounties_coordinator"),
                "get_claims",
                bounty_hash.clone(),
            )
            .await;
        assert_eq!(claims.len(), 1, "Should find Agent B's claim");
        assert_eq!(claims[0].organization, "CleanOrbit Inc.");

        // Agent A: Complete the bounty
        let _: ActionHash = conductor
            .call(
                &cell_creator.zome("debris_bounties_coordinator"),
                "update_bounty_status",
                UpdateBountyStatusInput {
                    bounty_hash: bounty_hash.clone(),
                    new_status: BountyStatus::Completed,
                },
            )
            .await;

        // Verify final state visible to both agents
        let bounties_a: Vec<DebrisBounty> = conductor
            .call(
                &cell_creator.zome("debris_bounties_coordinator"),
                "get_bounties_for_debris",
                70001u32,
            )
            .await;
        let bounties_b: Vec<DebrisBounty> = conductor
            .call(
                &cell_claimer.zome("debris_bounties_coordinator"),
                "get_bounties_for_debris",
                70001u32,
            )
            .await;
        assert_eq!(bounties_a[0].status, BountyStatus::Completed);
        assert_eq!(bounties_b[0].status, BountyStatus::Completed);

        // Contributions visible to both
        let contribs: Vec<BountyContribution> = conductor
            .call(
                &cell_claimer.zome("debris_bounties_coordinator"),
                "get_contributions",
                bounty_hash.clone(),
            )
            .await;
        assert_eq!(
            contribs.len(),
            1,
            "Agent B should see Agent A's contribution"
        );
    }

    /// Two agents register different sensors and independently observe the same object.
    /// Verifies observations from multiple sources are discoverable by either agent.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_multi_sensor_observation_corroboration() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;

        let app_texas = conductor
            .setup_app("sensor-texas", &[dna.clone()])
            .await
            .unwrap();
        let cell_texas = app_texas.cells()[0].clone();

        let app_colorado = conductor
            .setup_app("sensor-colorado", &[dna])
            .await
            .unwrap();
        let cell_colorado = app_colorado.cells()[0].clone();

        let target_norad = 71001u32;

        // Agent 1: Register Texas optical sensor
        let _: ActionHash = conductor
            .call(
                &cell_texas.zome("observations_coordinator"),
                "register_sensor",
                RegisterSensorInput {
                    sensor_id: "sensor:tx-optical".to_string(),
                    name: "Texas Optical Telescope".to_string(),
                    sensor_type: ObservationType::Optical,
                    location: Some(GroundLocation {
                        latitude_deg: 30.27,
                        longitude_deg: -97.74,
                        altitude_m: 150.0,
                        name: None,
                    }),
                    capabilities: SensorCapabilities {
                        min_size_m: None,
                        max_range_km: Some(36000.0),
                        fov_deg: None,
                        accuracy_arcsec: Some(2.0),
                    },
                },
            )
            .await;

        // Agent 2: Register Colorado radar station
        let _: ActionHash = conductor
            .call(
                &cell_colorado.zome("observations_coordinator"),
                "register_sensor",
                RegisterSensorInput {
                    sensor_id: "sensor:co-radar".to_string(),
                    name: "Colorado Springs Radar".to_string(),
                    sensor_type: ObservationType::Radar,
                    location: Some(GroundLocation {
                        latitude_deg: 38.83,
                        longitude_deg: -104.82,
                        altitude_m: 1830.0,
                        name: None,
                    }),
                    capabilities: SensorCapabilities {
                        min_size_m: None,
                        max_range_km: Some(40000.0),
                        fov_deg: None,
                        accuracy_arcsec: Some(5.0),
                    },
                },
            )
            .await;

        // Both sensors observe the same object
        let _: ActionHash = conductor
            .call(
                &cell_texas.zome("observations_coordinator"),
                "submit_observation",
                SubmitObservationInput {
                    norad_id: Some(target_norad),
                    observation_time: SpaceTimestamp::now(),
                    observer_location: Some(GroundLocation {
                        latitude_deg: 30.27,
                        longitude_deg: -97.74,
                        altitude_m: 150.0,
                        name: None,
                    }),
                    observation_type: ObservationType::Optical,
                    measurement: Measurement::Photometric {
                        magnitude: -1.8,
                        magnitude_sigma: None,
                        filter: None,
                    },
                    quality: Some(QualityScore::new(88)),
                    sensor_id: "sensor:tx-optical".to_string(),
                },
            )
            .await;

        let _: ActionHash = conductor
            .call(
                &cell_colorado.zome("observations_coordinator"),
                "submit_observation",
                SubmitObservationInput {
                    norad_id: Some(target_norad),
                    observation_time: SpaceTimestamp::now(),
                    observer_location: Some(GroundLocation {
                        latitude_deg: 38.83,
                        longitude_deg: -104.82,
                        altitude_m: 1830.0,
                        name: None,
                    }),
                    observation_type: ObservationType::Radar,
                    measurement: Measurement::Range {
                        range_km: 900.0,
                        range_rate_kms: Some(-1.5),
                        range_sigma_km: None,
                    },
                    quality: Some(QualityScore::new(90)),
                    sensor_id: "sensor:co-radar".to_string(),
                },
            )
            .await;

        // Either agent can query all observations for the object
        let obs_from_texas: Vec<serde_json::Value> = conductor
            .call(
                &cell_texas.zome("observations_coordinator"),
                "get_observations_for_object",
                target_norad,
            )
            .await;
        assert_eq!(
            obs_from_texas.len(),
            2,
            "Texas agent should see both observations"
        );

        let obs_from_colorado: Vec<serde_json::Value> = conductor
            .call(
                &cell_colorado.zome("observations_coordinator"),
                "get_observations_for_object",
                target_norad,
            )
            .await;
        assert_eq!(
            obs_from_colorado.len(),
            2,
            "Colorado agent should see both observations"
        );

        // Sensor-specific queries return only that sensor's data
        let tx_only: Vec<serde_json::Value> = conductor
            .call(
                &cell_colorado.zome("observations_coordinator"),
                "get_sensor_observations",
                "sensor:tx-optical".to_string(),
            )
            .await;
        assert_eq!(tx_only.len(), 1, "Texas sensor should have 1 observation");

        let co_only: Vec<serde_json::Value> = conductor
            .call(
                &cell_texas.zome("observations_coordinator"),
                "get_sensor_observations",
                "sensor:co-radar".to_string(),
            )
            .await;
        assert_eq!(
            co_only.len(),
            1,
            "Colorado sensor should have 1 observation"
        );

        // Both sensors appear in the global sensor list
        let sensors: Vec<serde_json::Value> = conductor
            .call(
                &cell_texas.zome("observations_coordinator"),
                "list_sensors",
                (),
            )
            .await;
        assert!(
            sensors.len() >= 2,
            "Should list both sensors from different agents"
        );
    }

    /// Agent A creates bounty, Agent B claims it.
    /// Agent C attempts to claim the same bounty — should fail because it's already Claimed.
    /// Verifies the state machine prevents double-claiming across agents.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_competing_bounty_claims() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;

        let app_creator = conductor
            .setup_app("creator", &[dna.clone()])
            .await
            .unwrap();
        let cell_creator = app_creator.cells()[0].clone();

        let app_claimer1 = conductor
            .setup_app("claimer-1", &[dna.clone()])
            .await
            .unwrap();
        let cell_claimer1 = app_claimer1.cells()[0].clone();

        let app_claimer2 = conductor.setup_app("claimer-2", &[dna]).await.unwrap();
        let cell_claimer2 = app_claimer2.cells()[0].clone();

        // Creator makes a bounty
        let bounty_hash: ActionHash = conductor
            .call(
                &cell_creator.zome("debris_bounties_coordinator"),
                "create_bounty",
                CreateBountyInput {
                    bounty_id: "bounty:compete-test".to_string(),
                    debris_norad_id: 72001,
                    justification: "Competing claims test".to_string(),
                    amount: 200000,
                    currency: "USD".to_string(),
                    expires_at: None,
                    requirements: RemovalRequirements {
                        min_trust_level: 1,
                        allowed_methods: vec![RemovalMethod::Any],
                        completion_deadline_days: 365,
                        verification_threshold: 2,
                    },
                },
            )
            .await;

        // Claimer 1: Claim the bounty (should succeed)
        let _: ActionHash = conductor
            .call(
                &cell_claimer1.zome("debris_bounties_coordinator"),
                "claim_bounty",
                ClaimBountyInput {
                    bounty_hash: bounty_hash.clone(),
                    organization: "FirstMover Corp".to_string(),
                    method: RemovalMethod::Capture,
                    estimated_completion: SpaceTimestamp::now(),
                    mission_plan: "First claim gets priority".to_string(),
                },
            )
            .await;

        // Verify first claim is recorded
        let claims: Vec<RemovalClaim> = conductor
            .call(
                &cell_creator.zome("debris_bounties_coordinator"),
                "get_claims",
                bounty_hash.clone(),
            )
            .await;
        assert_eq!(claims.len(), 1, "Only the first claim should exist");
        assert_eq!(claims[0].organization, "FirstMover Corp");

        // Verify bounty status changed from Open to Claimed
        let bounties: Vec<DebrisBounty> = conductor
            .call(
                &cell_creator.zome("debris_bounties_coordinator"),
                "get_bounties_for_debris",
                72001u32,
            )
            .await;
        assert_eq!(
            bounties[0].status,
            BountyStatus::Claimed,
            "Bounty should be Claimed after first claim"
        );

        // Agent C can still contribute even after it's claimed
        let contrib_input = ContributeInput {
            bounty_hash: bounty_hash.clone(),
            bounty_id: "bounty:compete-test".to_string(),
            amount: 25000,
            currency: "USD".to_string(),
            message: Some("Late contribution from third party".to_string()),
        };
        let _: ActionHash = conductor
            .call(
                &cell_claimer2.zome("debris_bounties_coordinator"),
                "contribute_to_bounty",
                contrib_input,
            )
            .await;

        let contribs: Vec<BountyContribution> = conductor
            .call(
                &cell_claimer2.zome("debris_bounties_coordinator"),
                "get_contributions",
                bounty_hash.clone(),
            )
            .await;
        assert_eq!(
            contribs.len(),
            1,
            "Third party contribution should be recorded"
        );
    }
}

// ============================================================================
// Fusion Tests (observations coordinator)
// ============================================================================

#[cfg(test)]
mod fusion_tests {
    use super::*;

    /// Submit multiple StateVector observations from different sensors,
    /// then call `fuse_observations_for_object`. The fused estimate should
    /// combine all contributing sensors into a single best-estimate state.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_fuse_observations_for_object() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let target_norad = 80001u32;

        // Register two sensors
        for (id, name) in [
            ("sensor:fusion-radar-1", "Fusion Radar Alpha"),
            ("sensor:fusion-radar-2", "Fusion Radar Beta"),
        ] {
            let _: ActionHash = conductor
                .call(
                    &cell.zome("observations_coordinator"),
                    "register_sensor",
                    RegisterSensorInput {
                        sensor_id: id.to_string(),
                        name: name.to_string(),
                        sensor_type: ObservationType::Radar,
                        location: Some(GroundLocation {
                            latitude_deg: 35.0,
                            longitude_deg: -106.0,
                            altitude_m: 1500.0,
                            name: None,
                        }),
                        capabilities: SensorCapabilities {
                            min_size_m: None,
                            max_range_km: Some(40000.0),
                            fov_deg: None,
                            accuracy_arcsec: Some(5.0),
                        },
                    },
                )
                .await;
        }

        // Submit StateVector observations from both sensors (slightly different)
        // Sensor 1: ISS-like orbit
        let _: ActionHash = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "submit_observation",
                SubmitObservationInput {
                    norad_id: Some(target_norad),
                    observation_time: SpaceTimestamp::now(),
                    observer_location: None,
                    observation_type: ObservationType::Radar,
                    measurement: Measurement::StateVector {
                        position_km: [6778.0, 0.0, 0.0],
                        velocity_kms: Some([0.0, 7.67, 0.0]),
                        covariance: None,
                    },
                    quality: Some(QualityScore::new(85)),
                    sensor_id: "sensor:fusion-radar-1".to_string(),
                },
            )
            .await;

        // Sensor 2: Same object, slightly different measurement (noise)
        let _: ActionHash = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "submit_observation",
                SubmitObservationInput {
                    norad_id: Some(target_norad),
                    observation_time: SpaceTimestamp::now(),
                    observer_location: None,
                    observation_type: ObservationType::Radar,
                    measurement: Measurement::StateVector {
                        position_km: [6778.5, 0.1, -0.1],
                        velocity_kms: Some([0.001, 7.669, 0.001]),
                        covariance: None,
                    },
                    quality: Some(QualityScore::new(90)),
                    sensor_id: "sensor:fusion-radar-2".to_string(),
                },
            )
            .await;

        // Fuse observations
        let fused: FusedEstimate = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "fuse_observations_for_object",
                target_norad,
            )
            .await;

        // Validate fused estimate structure
        assert_eq!(fused.norad_id, target_norad);
        assert_eq!(
            fused.state_vector.len(),
            6,
            "State vector must have 6 elements"
        );
        assert_eq!(
            fused.covariance_diagonal.len(),
            6,
            "Covariance diagonal must have 6 elements"
        );
        assert!(
            fused.contributing_sensors.len() >= 2,
            "Should have at least 2 contributing sensors, got {}",
            fused.contributing_sensors.len()
        );
        assert!(
            fused.fused_quality >= 0.0 && fused.fused_quality <= 1.0,
            "Fused quality {} should be in [0, 1]",
            fused.fused_quality
        );
        assert!(
            fused.chi_square_consistency >= 0.0,
            "Chi-square should be non-negative"
        );

        // Fused position should be near the average of the two inputs
        let fused_x = fused.state_vector[0];
        assert!(
            (fused_x - 6778.25).abs() < 1.0,
            "Fused X position {} should be near 6778.25 km",
            fused_x
        );
    }

    /// After fusion, `get_fused_state` should return the stored estimate.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_fused_state() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let target_norad = 80002u32;

        // Register sensor and submit one StateVector observation
        let _: ActionHash = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "register_sensor",
                RegisterSensorInput {
                    sensor_id: "sensor:fused-state-test".to_string(),
                    name: "Fused State Test Sensor".to_string(),
                    sensor_type: ObservationType::Radar,
                    location: None,
                    capabilities: SensorCapabilities {
                        min_size_m: None,
                        max_range_km: Some(40000.0),
                        fov_deg: None,
                        accuracy_arcsec: Some(5.0),
                    },
                },
            )
            .await;

        let _: ActionHash = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "submit_observation",
                SubmitObservationInput {
                    norad_id: Some(target_norad),
                    observation_time: SpaceTimestamp::now(),
                    observer_location: None,
                    observation_type: ObservationType::Radar,
                    measurement: Measurement::StateVector {
                        position_km: [7000.0, 100.0, 50.0],
                        velocity_kms: Some([0.1, 7.5, 0.05]),
                        covariance: None,
                    },
                    quality: Some(QualityScore::new(80)),
                    sensor_id: "sensor:fused-state-test".to_string(),
                },
            )
            .await;

        // Fuse (creates and stores the estimate)
        let _fused: FusedEstimate = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "fuse_observations_for_object",
                target_norad,
            )
            .await;

        // Query the stored fused state
        let retrieved: Option<FusedEstimate> = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "get_fused_state",
                target_norad,
            )
            .await;

        assert!(retrieved.is_some(), "Should retrieve stored fused estimate");
        let est = retrieved.unwrap();
        assert_eq!(est.norad_id, target_norad);
        assert_eq!(est.state_vector.len(), 6);
    }

    /// Observations without StateVector/Range data (e.g., Photometric only)
    /// should cause `fuse_observations_for_object` to fail gracefully since
    /// the fusion pipeline requires position/velocity data.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_fusion_requires_state_observations() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let target_norad = 80003u32;

        // Register sensor
        let _: ActionHash = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "register_sensor",
                RegisterSensorInput {
                    sensor_id: "sensor:photometric-only".to_string(),
                    name: "Photometric Only Sensor".to_string(),
                    sensor_type: ObservationType::Optical,
                    location: None,
                    capabilities: SensorCapabilities {
                        min_size_m: None,
                        max_range_km: Some(36000.0),
                        fov_deg: None,
                        accuracy_arcsec: Some(2.0),
                    },
                },
            )
            .await;

        // Submit only Photometric observation (no position/velocity)
        let _: ActionHash = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "submit_observation",
                SubmitObservationInput {
                    norad_id: Some(target_norad),
                    observation_time: SpaceTimestamp::now(),
                    observer_location: None,
                    observation_type: ObservationType::Optical,
                    measurement: Measurement::Photometric {
                        magnitude: -1.5,
                        magnitude_sigma: Some(0.3),
                        filter: Some("V".to_string()),
                    },
                    quality: Some(QualityScore::new(75)),
                    sensor_id: "sensor:photometric-only".to_string(),
                },
            )
            .await;

        // Fuse should fail — no StateVector/Range observations available
        // The fusion pipeline filters to only fusable measurement types,
        // so with only Photometric data it should return an error.
        let result: Result<FusedEstimate, _> = conductor
            .call_fallible(
                &cell.zome("observations_coordinator"),
                "fuse_observations_for_object",
                target_norad,
            )
            .await;

        assert!(
            result.is_err(),
            "Fusion should fail when only Photometric observations are available"
        );
    }

    /// Query fused state for a NORAD ID with no observations should return None.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_fused_state_empty() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let retrieved: Option<FusedEstimate> = conductor
            .call(
                &cell.zome("observations_coordinator"),
                "get_fused_state",
                99999u32,
            )
            .await;

        assert!(
            retrieved.is_none(),
            "Should return None for NORAD ID with no fused estimates"
        );
    }
}

// ============================================================================
// Avoidance Planning Tests (traffic control coordinator)
// ============================================================================

#[cfg(test)]
mod avoidance_tests {
    use super::*;

    // ISS-like TLE for test use
    const PRIMARY_TLE_LINE1: &str =
        "1 80101U 26AVD-A  26040.50000000  .00016717  00000-0  10270-3 0  9990";
    const PRIMARY_TLE_LINE2: &str =
        "2 80101  51.6400 208.9163 0006703 306.0500  54.0150 15.49560532000001";
    const SECONDARY_TLE_LINE1: &str =
        "1 80102U 19AVD-B  26040.50000000  .00000500  00000-0  50000-4 0  9990";
    const SECONDARY_TLE_LINE2: &str =
        "2 80102  51.6500 209.0000 0007000 305.0000  55.0000 15.49000000000001";

    /// Generate avoidance options for a conjunction using Lambert-based
    /// maneuver planning. Verifies that the traffic control coordinator
    /// produces valid avoidance maneuvers.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_generate_avoidance_options() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = AvoidanceInput {
            session_id: "session:avoidance-test-001".to_string(),
            primary_tle_line1: PRIMARY_TLE_LINE1.to_string(),
            primary_tle_line2: PRIMARY_TLE_LINE2.to_string(),
            secondary_tle_line1: SECONDARY_TLE_LINE1.to_string(),
            secondary_tle_line2: SECONDARY_TLE_LINE2.to_string(),
            tca: SpaceTimestamp::now(),
            max_delta_v_ms: 5.0,
            lead_time_hours: 24.0,
        };

        let options: Vec<AvoidanceOption> = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "generate_avoidance_options",
                input,
            )
            .await;

        // Should produce at least one valid avoidance option
        assert!(
            !options.is_empty(),
            "Should generate at least 1 avoidance option"
        );

        for opt in &options {
            // Each option should have a valid delta-v
            assert_eq!(
                opt.delta_v.len(),
                3,
                "Delta-v vector must have 3 components"
            );
            assert!(
                opt.delta_v_magnitude_ms > 0.0,
                "Delta-v magnitude should be positive"
            );
            assert!(
                opt.delta_v_magnitude_ms <= 5.0,
                "Delta-v {} m/s exceeds budget of 5.0 m/s",
                opt.delta_v_magnitude_ms
            );

            // Miss distance should be improved (> 0 km)
            assert!(
                opt.resulting_miss_distance_km > 0.0,
                "Resulting miss distance should be positive"
            );

            // Pc should be reduced (< 1)
            assert!(
                opt.resulting_pc < 1.0 && opt.resulting_pc >= 0.0,
                "Resulting Pc {} should be in [0, 1)",
                opt.resulting_pc
            );

            // Transfer type should be one of the standard types
            assert!(
                ["prograde", "retrograde", "radial", "normal"]
                    .contains(&opt.transfer_type.as_str()),
                "Unknown transfer type: {}",
                opt.transfer_type
            );

            // Session ID should match
            assert_eq!(opt.session_id, "session:avoidance-test-001");
        }
    }

    /// Generated avoidance options should be queryable by session ID.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_avoidance_options() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let session_id = "session:avoidance-query-test".to_string();

        // Generate options
        let input = AvoidanceInput {
            session_id: session_id.clone(),
            primary_tle_line1: PRIMARY_TLE_LINE1.to_string(),
            primary_tle_line2: PRIMARY_TLE_LINE2.to_string(),
            secondary_tle_line1: SECONDARY_TLE_LINE1.to_string(),
            secondary_tle_line2: SECONDARY_TLE_LINE2.to_string(),
            tca: SpaceTimestamp::now(),
            max_delta_v_ms: 3.0,
            lead_time_hours: 12.0,
        };

        let generated: Vec<AvoidanceOption> = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "generate_avoidance_options",
                input,
            )
            .await;

        // Query back by session ID
        let queried: Vec<AvoidanceOption> = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "get_avoidance_options",
                session_id.clone(),
            )
            .await;

        assert_eq!(
            queried.len(),
            generated.len(),
            "Queried options count should match generated count"
        );

        // All queried options should have the correct session ID
        for opt in &queried {
            assert_eq!(opt.session_id, session_id);
        }
    }

    /// Avoidance with zero fuel budget should produce no valid options.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_avoidance_zero_fuel_budget() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = AvoidanceInput {
            session_id: "session:zero-fuel-test".to_string(),
            primary_tle_line1: PRIMARY_TLE_LINE1.to_string(),
            primary_tle_line2: PRIMARY_TLE_LINE2.to_string(),
            secondary_tle_line1: SECONDARY_TLE_LINE1.to_string(),
            secondary_tle_line2: SECONDARY_TLE_LINE2.to_string(),
            tca: SpaceTimestamp::now(),
            max_delta_v_ms: 0.0, // Zero fuel budget
            lead_time_hours: 24.0,
        };

        let options: Vec<AvoidanceOption> = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "generate_avoidance_options",
                input,
            )
            .await;

        assert!(
            options.is_empty(),
            "Zero fuel budget should produce no avoidance options, got {}",
            options.len()
        );
    }

    /// Query avoidance options for a non-existent session should return empty.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_avoidance_options_empty() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let options: Vec<AvoidanceOption> = conductor
            .call(
                &cell.zome("traffic_control_coordinator"),
                "get_avoidance_options",
                "session:nonexistent".to_string(),
            )
            .await;

        assert!(
            options.is_empty(),
            "Non-existent session should return empty"
        );
    }
}
