// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared Types and Utilities for Mycelix Space
//!
//! Common definitions used across all DNA zomes:
//! - Orbital object identifiers and entries
//! - TLE and state vector data
//! - Conjunction assessments and risk levels
//! - Observations and sensor data
//! - Maneuver coordination
//! - Debris bounty tracking
//! - Timestamp handling
//! - Validation helpers
//! - Trust and reputation primitives

use chrono::{DateTime, Utc};
use hdi::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// NORAD Catalog Number - unique identifier for tracked space objects
/// Valid range: 1 to 999999 (currently ~60,000 active objects)
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct NoradId(pub u32);

impl NoradId {
    pub fn new(id: u32) -> ExternResult<Self> {
        if id == 0 || id > 999999 {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Invalid NORAD ID: {}. Must be 1-999999",
                id
            ))));
        }
        Ok(Self(id))
    }

    pub fn value(&self) -> u32 {
        self.0
    }
}

/// Epoch timestamp with microsecond precision
/// Wrapper around chrono DateTime for Holochain serialization
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpaceTimestamp {
    /// Microseconds since Unix epoch
    pub micros: i64,
}

impl SpaceTimestamp {
    pub fn now() -> Self {
        let now = Utc::now();
        Self {
            micros: now.timestamp_micros(),
        }
    }

    pub fn from_datetime(dt: DateTime<Utc>) -> Self {
        Self {
            micros: dt.timestamp_micros(),
        }
    }

    pub fn to_datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_micros(self.micros).unwrap_or_else(Utc::now)
    }

    /// Age in seconds from now
    pub fn age_seconds(&self) -> i64 {
        let now = Self::now();
        (now.micros - self.micros) / 1_000_000
    }
}

/// Data quality indicator (0-100)
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct QualityScore(pub u8);

impl QualityScore {
    pub fn new(score: u8) -> Self {
        Self(score.min(100))
    }

    pub fn value(&self) -> u8 {
        self.0
    }

    pub fn is_high(&self) -> bool {
        self.0 >= 80
    }

    pub fn is_acceptable(&self) -> bool {
        self.0 >= 50
    }
}

impl Default for QualityScore {
    fn default() -> Self {
        Self(50)
    }
}

// =============================================================================
// TLE Staleness Detection
// =============================================================================

/// Default staleness threshold: 30 days in seconds.
pub const TLE_STALE_THRESHOLD_SECS: i64 = 30 * 24 * 3600;

/// Configuration for TLE staleness checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StalenessConfig {
    /// Seconds after which a TLE is considered stale (default: 30 days)
    pub stale_threshold_secs: i64,
    /// Quality score below which a TLE is rejected (default: 20)
    pub min_quality: u8,
}

impl Default for StalenessConfig {
    fn default() -> Self {
        Self {
            stale_threshold_secs: TLE_STALE_THRESHOLD_SECS,
            min_quality: 20,
        }
    }
}

/// Staleness assessment result for a TLE.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TleStaleness {
    /// Age of the TLE epoch in seconds
    pub age_secs: i64,
    /// Whether the TLE exceeds the staleness threshold
    pub is_stale: bool,
    /// Original quality score (as submitted)
    pub original_quality: QualityScore,
    /// Effective quality after age-based degradation
    pub effective_quality: QualityScore,
    /// Confidence multiplier for Pc calculations (1.0 = full, 0.0 = no confidence)
    pub confidence: f64,
}

/// Compute age-degraded quality and confidence for a TLE.
///
/// Quality degrades linearly: loses ~1 point per day after 7 days.
/// Confidence drops from 1.0 at epoch to 0.3 at the stale threshold,
/// and continues to 0.1 at 2x the threshold.
pub fn compute_staleness(
    epoch: &SpaceTimestamp,
    quality: &QualityScore,
    config: &StalenessConfig,
) -> TleStaleness {
    let age_secs = epoch.age_seconds();

    // Quality degrades: ~1 point per day after 7 days of age
    let grace_period_secs: i64 = 7 * 24 * 3600;
    let days_past_grace = ((age_secs - grace_period_secs).max(0) as f64) / 86400.0;
    let degraded = (quality.value() as f64 - days_past_grace).max(0.0) as u8;
    let effective_quality = QualityScore::new(degraded);

    // Confidence: 1.0 for fresh TLEs, decays towards 0.1 for very old ones
    let threshold = config.stale_threshold_secs as f64;
    let age_f = age_secs.max(0) as f64;
    let confidence = if threshold <= 0.0 {
        1.0
    } else {
        (1.0 - 0.7 * (age_f / threshold).min(2.0) / 2.0).max(0.1)
    };

    let is_stale = age_secs > config.stale_threshold_secs;

    TleStaleness {
        age_secs,
        is_stale,
        original_quality: *quality,
        effective_quality,
        confidence,
    }
}

/// TLE data with full metadata (epoch, quality, staleness).
/// Used by staleness-aware query endpoints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TleWithMetadata {
    pub norad_id: u32,
    pub line1: String,
    pub line2: String,
    pub epoch: SpaceTimestamp,
    pub quality: QualityScore,
    pub source: DataSourceType,
    pub submitted_at: SpaceTimestamp,
    pub staleness: TleStaleness,
}

/// Source of orbital data
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DataSourceType {
    /// Official NORAD/Space-Track data
    SpaceTrack,

    /// Commercial SSA provider (LeoLabs, ExoAnalytic, etc.)
    Commercial { provider: String },

    /// Ground-based sensor observation
    GroundSensor {
        sensor_id: String,
        location: Option<GroundLocation>,
    },

    /// Space-based observation (from another satellite)
    SpaceSensor { observer_norad_id: NoradId },

    /// Operator-provided ephemeris (highest trust for own assets)
    OperatorEphemeris { operator: AgentPubKey },

    /// Fused from multiple sources in the network
    NetworkFusion { source_count: u32, node_count: u32 },
}

/// Ground sensor location
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GroundLocation {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub altitude_m: f64,
    pub name: Option<String>,
}

/// Trust level for an agent in the network
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum TrustLevel {
    /// New/unknown agent
    #[default]
    Unverified,
    /// Agent has some history but limited track record
    BasicTrust,
    /// Agent has demonstrated consistent good data
    Established,
    /// Agent is a verified operator or organization
    Verified,
    /// Founding member or core infrastructure
    FoundingMember,
}

impl TrustLevel {
    pub fn weight(&self) -> f64 {
        match self {
            TrustLevel::Unverified => 0.1,
            TrustLevel::BasicTrust => 0.3,
            TrustLevel::Established => 0.6,
            TrustLevel::Verified => 0.9,
            TrustLevel::FoundingMember => 1.0,
        }
    }
}

/// Observation metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservationMeta {
    /// When the observation was taken
    pub observation_time: SpaceTimestamp,

    /// When it was submitted to the network
    pub submission_time: SpaceTimestamp,

    /// Source type
    pub source: DataSourceType,

    /// Quality indicator
    pub quality: QualityScore,

    /// Hash of the raw observation data (for verification)
    pub raw_data_hash: Option<[u8; 32]>,
}

/// Maneuver notification (for traffic coordination)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManeuverNotification {
    /// Object being maneuvered
    pub norad_id: NoradId,

    /// Operator announcing the maneuver
    pub operator: AgentPubKey,

    /// Planned maneuver time
    pub planned_time: SpaceTimestamp,

    /// Expected delta-V (m/s)
    pub delta_v_ms: f64,

    /// Direction (unit vector in ECI)
    pub direction: [f64; 3],

    /// Confidence/certainty of the maneuver happening
    pub confidence: f64,

    /// Is this a collision avoidance maneuver?
    pub is_collision_avoidance: bool,

    /// Related conjunction ID (if collision avoidance)
    pub conjunction_id: Option<ActionHash>,
}

/// Input for on-chain conjunction screening via SGP4 propagation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScreenInput {
    /// Primary object NORAD catalog number
    pub primary_norad_id: u32,
    /// Secondary object NORAD catalog numbers (max 100)
    pub secondary_norad_ids: Vec<u32>,
    /// Hours to propagate ahead from current time
    pub hours_ahead: f64,
    /// Minimum collision probability to include in results
    pub pc_threshold: f64,
}

/// Conjunction screening request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScreeningRequest {
    /// Objects to screen (empty = all tracked objects)
    pub protected_objects: Vec<NoradId>,

    /// Time window start
    pub start_time: SpaceTimestamp,

    /// Time window end
    pub end_time: SpaceTimestamp,

    /// Miss distance threshold (km)
    pub threshold_km: f64,

    /// Minimum Pc to report
    pub min_pc: Option<f64>,
}

/// Conjunction Data Message (CDM) - industry standard format
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConjunctionDataMessage {
    /// Unique ID for this conjunction event
    pub conjunction_id: String,

    /// Time of closest approach
    pub tca: SpaceTimestamp,

    /// Primary object info
    pub primary: CdmObject,

    /// Secondary object info
    pub secondary: CdmObject,

    /// Miss distance (km)
    pub miss_distance_km: f64,

    /// Relative velocity (km/s)
    pub relative_velocity_kms: f64,

    /// Probability of collision
    pub collision_probability: f64,

    /// Hard body radius used (m)
    pub hard_body_radius_m: f64,

    /// Message creation time
    pub creation_time: SpaceTimestamp,

    /// Originator of this CDM
    pub originator: AgentPubKey,
}

/// Object info within a CDM
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CdmObject {
    pub norad_id: NoradId,
    pub name: Option<String>,
    pub operator: Option<String>,

    /// State vector at TCA
    pub position_km: [f64; 3],
    pub velocity_kms: [f64; 3],

    /// Covariance (lower triangular, 21 elements)
    pub covariance: Option<[f64; 21]>,
}

// =============================================================================
// Space Object Types
// =============================================================================

/// Type of space object
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObjectType {
    /// Active satellite/spacecraft
    Payload,
    /// Rocket body
    RocketBody,
    /// Debris fragment
    Debris,
    /// Unknown/unclassified
    Unknown,
}

/// Simple data source type (for serialization tests)
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataSourceSimple {
    SpaceTrack,
    CelesTrak,
    Operator,
    Computed,
}

/// Reference frame for state vectors
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReferenceFrame {
    /// True Equator Mean Equinox
    Teme,
    /// J2000 Earth-centered inertial
    J2000,
    /// International Terrestrial Reference Frame
    Itrf,
    /// Geocentric Celestial Reference Frame
    Gcrf,
}

/// Orbital object entry for the catalog
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrbitalObjectEntry {
    pub norad_id: u32,
    pub name: String,
    pub object_type: ObjectType,
    pub launch_date: Option<DateTime<Utc>>,
    pub decay_date: Option<DateTime<Utc>>,
    pub owner_country: Option<String>,
    pub data_source: DataSourceSimple,
    pub metadata: HashMap<String, String>,
}

/// Two-Line Element data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TleData {
    pub norad_id: u32,
    pub line1: String,
    pub line2: String,
    pub epoch: DateTime<Utc>,
    pub source: DataSourceSimple,
}

/// State vector with position and velocity
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateVectorData {
    pub norad_id: u32,
    pub epoch: DateTime<Utc>,
    pub position_km: [f64; 3],
    pub velocity_kms: [f64; 3],
    pub covariance: Option<[f64; 21]>,
    pub reference_frame: ReferenceFrame,
    pub quality: f64,
    pub source: DataSourceSimple,
}

// =============================================================================
// Conjunction Assessment Types
// =============================================================================

/// Risk level for conjunction events
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum RiskLevel {
    /// Pc < 1e-7, routine monitoring only
    Negligible = 0,
    /// Pc 1e-7 to 1e-5, increased monitoring
    Low = 1,
    /// Pc 1e-5 to 1e-4, planning phase
    Medium = 2,
    /// Pc 1e-4 to 1e-3, maneuver recommended
    High = 3,
    /// Pc > 1e-3, immediate action required
    Emergency = 4,
}

impl RiskLevel {
    pub fn from_pc(pc: f64) -> Self {
        if pc > 1e-3 {
            RiskLevel::Emergency
        } else if pc > 1e-4 {
            RiskLevel::High
        } else if pc > 1e-5 {
            RiskLevel::Medium
        } else if pc > 1e-7 {
            RiskLevel::Low
        } else {
            RiskLevel::Negligible
        }
    }
}

/// Probability of collision calculation method
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PcMethod {
    /// Alfano 2D screening
    Alfano2D,
    /// Foster 3D integration
    Foster3D,
    /// Chan Monte Carlo
    ChanMonteCarlo,
    /// Patera maximum
    PateraMax,
}

/// Conjunction assessment entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConjunctionAssessment {
    pub primary_norad_id: u32,
    pub secondary_norad_id: u32,
    pub tca: DateTime<Utc>,
    pub miss_distance_km: f64,
    pub relative_velocity_kms: f64,
    pub collision_probability: f64,
    pub pc_method: PcMethod,
    pub risk_level: RiskLevel,
    pub hard_body_radius_m: f64,
    pub screening_volume_km: f64,
}

/// Staleness-enriched conjunction assessment.
/// Wraps a standard ConjunctionAssessment with TLE quality metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StalenessAwareAssessment {
    /// The conjunction assessment (miss distance, Pc, risk level, etc.)
    pub assessment: ConjunctionAssessment,
    /// Staleness of the primary object's TLE
    pub primary_staleness: TleStaleness,
    /// Staleness of the secondary object's TLE
    pub secondary_staleness: TleStaleness,
    /// Combined confidence (product of both TLE confidences).
    /// Values below 0.5 mean at least one TLE is significantly stale.
    pub combined_confidence: f64,
    /// Whether either TLE is stale (convenience flag)
    pub has_stale_data: bool,
}

// =============================================================================
// Observation Types
// =============================================================================

/// Type of observation
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObservationType {
    /// Optical telescope observation
    Optical,
    /// Radar observation
    Radar,
    /// Laser ranging
    LaserRanging,
    /// Space-based observation
    SpaceBased,
    /// Radio signal tracking
    RadioTracking,
}

/// Observation entry from a sensor
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservationEntry {
    pub norad_id: u32,
    pub timestamp: DateTime<Utc>,
    pub observation_type: ObservationType,
    pub observer_location: GroundLocation,
    pub raw_data: serde_json::Value,
    pub quality_score: f64,
    pub data_source: DataSourceSimple,
}

// =============================================================================
// Maneuver Types
// =============================================================================

/// Type of orbital maneuver
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ManeuverType {
    /// Along velocity vector
    InTrack,
    /// Perpendicular to orbit plane
    CrossTrack,
    /// Toward/away from Earth center
    Radial,
    /// Combined maneuver
    Combined,
}

/// Priority level for maneuvers
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ManeuverPriority {
    Low,
    Normal,
    High,
    Emergency,
}

/// Status of a maneuver request
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ManeuverStatus {
    Proposed,
    Acknowledged,
    Approved,
    Executed,
    Cancelled,
    Rejected,
}

/// Maneuver request for traffic coordination
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManeuverRequest {
    pub requesting_object: u32,
    pub conflicting_object: u32,
    pub conjunction_id: String,
    pub proposed_maneuver: ManeuverType,
    pub delta_v_ms: f64,
    pub execution_time: DateTime<Utc>,
    pub priority: ManeuverPriority,
    pub status: ManeuverStatus,
}

// =============================================================================
// Debris Bounty Types
// =============================================================================

/// Requirements for a debris removal bounty
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BountyRequirements {
    pub min_mass_kg: Option<f64>,
    pub max_altitude_km: Option<f64>,
    pub debris_type: Option<ObjectType>,
    pub removal_method: Option<String>,
}

/// Status of a bounty
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum BountyStatus {
    Active,
    Claimed,
    InProgress,
    Completed,
    Expired,
    Cancelled,
}

/// Bounty entry for debris removal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BountyEntry {
    pub debris_norad_id: u32,
    pub bounty_amount: u64,
    pub currency: String,
    pub sponsor: String,
    pub deadline: Option<DateTime<Utc>>,
    pub requirements: BountyRequirements,
    pub status: BountyStatus,
}

// =============================================================================
// Alert and Signal Types
// =============================================================================

/// Alert types for real-time notifications
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AlertType {
    /// New conjunction detected
    ConjunctionDetected,
    /// Conjunction risk level escalated
    RiskEscalation,
    /// Conjunction risk level de-escalated
    RiskDeescalation,
    /// Maneuver required
    ManeuverRequired,
    /// Maneuver executed successfully
    ManeuverExecuted,
    /// Debris detected in critical orbit
    DebrisAlert,
    /// Bounty claimed
    BountyClaimed,
    /// System alert (network issues, etc.)
    SystemAlert,
}

/// Priority level for alerts
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertPriority {
    /// Informational only
    Info,
    /// Low priority, no immediate action needed
    Low,
    /// Medium priority, review recommended
    Medium,
    /// High priority, action may be needed
    High,
    /// Critical priority, immediate action required
    Critical,
}

impl AlertPriority {
    /// Get priority from risk level
    pub fn from_risk_level(level: RiskLevel) -> Self {
        match level {
            RiskLevel::Negligible => AlertPriority::Info,
            RiskLevel::Low => AlertPriority::Low,
            RiskLevel::Medium => AlertPriority::Medium,
            RiskLevel::High => AlertPriority::High,
            RiskLevel::Emergency => AlertPriority::Critical,
        }
    }
}

/// Conjunction alert payload for signals
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConjunctionAlert {
    /// Alert type
    pub alert_type: AlertType,
    /// Priority level
    pub priority: AlertPriority,
    /// Primary object NORAD ID
    pub primary_norad_id: u32,
    /// Secondary object NORAD ID
    pub secondary_norad_id: u32,
    /// Time of closest approach
    pub tca: DateTime<Utc>,
    /// Current miss distance (km)
    pub miss_distance_km: f64,
    /// Collision probability
    pub collision_probability: f64,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Alert generation time
    pub generated_at: DateTime<Utc>,
    /// Previous risk level (for escalation alerts)
    pub previous_risk_level: Option<RiskLevel>,
    /// Recommended action
    pub recommendation: Option<String>,
}

impl ConjunctionAlert {
    /// Create a new conjunction detected alert
    pub fn new_conjunction(assessment: &ConjunctionAssessment) -> Self {
        let recommendation = match assessment.risk_level {
            RiskLevel::Emergency => {
                Some("IMMEDIATE ACTION REQUIRED: Initiate collision avoidance maneuver".to_string())
            }
            RiskLevel::High => {
                Some("Prepare collision avoidance maneuver, monitor closely".to_string())
            }
            RiskLevel::Medium => {
                Some("Increase monitoring frequency, prepare contingency plans".to_string())
            }
            RiskLevel::Low => Some("Continue monitoring, no immediate action required".to_string()),
            RiskLevel::Negligible => None,
        };

        Self {
            alert_type: AlertType::ConjunctionDetected,
            priority: AlertPriority::from_risk_level(assessment.risk_level),
            primary_norad_id: assessment.primary_norad_id,
            secondary_norad_id: assessment.secondary_norad_id,
            tca: assessment.tca,
            miss_distance_km: assessment.miss_distance_km,
            collision_probability: assessment.collision_probability,
            risk_level: assessment.risk_level,
            generated_at: Utc::now(),
            previous_risk_level: None,
            recommendation,
        }
    }

    /// Create a risk escalation alert
    pub fn risk_escalation(assessment: &ConjunctionAssessment, previous_level: RiskLevel) -> Self {
        let mut alert = Self::new_conjunction(assessment);
        alert.alert_type = AlertType::RiskEscalation;
        alert.previous_risk_level = Some(previous_level);
        alert.recommendation = Some(format!(
            "RISK ESCALATED from {:?} to {:?}. {}",
            previous_level,
            assessment.risk_level,
            alert.recommendation.unwrap_or_default()
        ));
        alert
    }

    /// Check if this alert requires immediate attention
    pub fn is_critical(&self) -> bool {
        self.priority == AlertPriority::Critical || self.priority == AlertPriority::High
    }

    /// Get time until TCA
    pub fn time_to_tca(&self) -> chrono::Duration {
        self.tca.signed_duration_since(Utc::now())
    }
}

/// Maneuver alert payload for signals
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManeuverAlert {
    /// Alert type
    pub alert_type: AlertType,
    /// Priority level
    pub priority: AlertPriority,
    /// Object being maneuvered
    pub norad_id: u32,
    /// Related conjunction (if collision avoidance)
    pub conjunction_primary: Option<u32>,
    pub conjunction_secondary: Option<u32>,
    /// Maneuver details
    pub maneuver_type: ManeuverType,
    pub delta_v_ms: f64,
    pub execution_time: DateTime<Utc>,
    /// Alert generation time
    pub generated_at: DateTime<Utc>,
    /// Additional message
    pub message: Option<String>,
}

/// Debris bounty alert payload
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BountyAlert {
    /// Alert type
    pub alert_type: AlertType,
    /// Priority level
    pub priority: AlertPriority,
    /// Debris NORAD ID
    pub debris_norad_id: u32,
    /// Bounty amount
    pub bounty_amount: u64,
    /// Currency
    pub currency: String,
    /// Sponsor
    pub sponsor: String,
    /// Alert generation time
    pub generated_at: DateTime<Utc>,
    /// Additional message
    pub message: Option<String>,
}

/// Unified signal payload for all alert types
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "signal_type")]
pub enum SpaceSignal {
    /// Conjunction alert
    Conjunction(ConjunctionAlert),
    /// Maneuver notification
    Maneuver(ManeuverAlert),
    /// Bounty notification
    Bounty(BountyAlert),
    /// Generic system alert
    System {
        alert_type: AlertType,
        priority: AlertPriority,
        message: String,
        generated_at: DateTime<Utc>,
    },
}

impl SpaceSignal {
    /// Get the priority of this signal
    pub fn priority(&self) -> AlertPriority {
        match self {
            SpaceSignal::Conjunction(alert) => alert.priority,
            SpaceSignal::Maneuver(alert) => alert.priority,
            SpaceSignal::Bounty(alert) => alert.priority,
            SpaceSignal::System { priority, .. } => *priority,
        }
    }

    /// Check if this signal is critical
    pub fn is_critical(&self) -> bool {
        self.priority() >= AlertPriority::High
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Validate a string field: non-empty and within max length.
pub fn validate_string_field(
    value: &str,
    field_name: &str,
    max_len: usize,
) -> Result<(), SpaceError> {
    if value.is_empty() {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            format!("{} must not be empty", field_name),
        ));
    }
    if value.len() > max_len {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidInput,
            format!(
                "{} exceeds maximum length of {} characters",
                field_name, max_len
            ),
        )
        .with_context(format!("length: {}", value.len())));
    }
    Ok(())
}

/// Validate a geographic latitude (-90 to 90).
pub fn validate_latitude(lat: f64) -> Result<(), SpaceError> {
    if !(-90.0..=90.0).contains(&lat) || lat.is_nan() {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidLocation,
            "Latitude must be between -90 and 90 degrees",
        )
        .with_context(format!("got: {}", lat)));
    }
    Ok(())
}

/// Validate a geographic longitude (-180 to 180).
pub fn validate_longitude(lon: f64) -> Result<(), SpaceError> {
    if !(-180.0..=180.0).contains(&lon) || lon.is_nan() {
        return Err(SpaceError::new(
            SpaceErrorCode::InvalidLocation,
            "Longitude must be between -180 and 180 degrees",
        )
        .with_context(format!("got: {}", lon)));
    }
    Ok(())
}

/// Validate that a vector is a unit vector (within tolerance)
pub fn is_unit_vector(v: &[f64; 3], tolerance: f64) -> bool {
    let magnitude = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (magnitude - 1.0).abs() < tolerance
}

/// Hash data using SHA3-256
pub fn hash_data(data: &[u8]) -> [u8; 32] {
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();
    sha3::Digest::update(&mut hasher, data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norad_id_validation() {
        assert!(NoradId::new(25544).is_ok()); // ISS
        assert!(NoradId::new(1).is_ok());
        assert!(NoradId::new(999999).is_ok());
        assert!(NoradId::new(0).is_err());
        assert!(NoradId::new(1000000).is_err());
    }

    #[test]
    fn test_quality_score() {
        let high = QualityScore::new(90);
        assert!(high.is_high());
        assert!(high.is_acceptable());

        let medium = QualityScore::new(60);
        assert!(!medium.is_high());
        assert!(medium.is_acceptable());

        let low = QualityScore::new(30);
        assert!(!low.is_high());
        assert!(!low.is_acceptable());

        // Clamped to 100
        let over = QualityScore::new(150);
        assert_eq!(over.value(), 100);
    }

    #[test]
    fn test_unit_vector() {
        assert!(is_unit_vector(&[1.0, 0.0, 0.0], 0.01));
        assert!(is_unit_vector(&[0.0, 1.0, 0.0], 0.01));
        assert!(is_unit_vector(&[0.577, 0.577, 0.577], 0.01)); // ~1/sqrt(3)
        assert!(!is_unit_vector(&[1.0, 1.0, 0.0], 0.01)); // magnitude sqrt(2)
    }

    #[test]
    fn test_risk_level_from_pc() {
        assert_eq!(RiskLevel::from_pc(1e-8), RiskLevel::Negligible);
        assert_eq!(RiskLevel::from_pc(1e-6), RiskLevel::Low);
        assert_eq!(RiskLevel::from_pc(5e-5), RiskLevel::Medium);
        assert_eq!(RiskLevel::from_pc(5e-4), RiskLevel::High);
        assert_eq!(RiskLevel::from_pc(1e-2), RiskLevel::Emergency);
    }

    #[test]
    fn test_risk_level_ordering() {
        assert_eq!(RiskLevel::Negligible as u8, 0);
        assert_eq!(RiskLevel::Low as u8, 1);
        assert_eq!(RiskLevel::Medium as u8, 2);
        assert_eq!(RiskLevel::High as u8, 3);
        assert_eq!(RiskLevel::Emergency as u8, 4);

        assert!(RiskLevel::Negligible < RiskLevel::Low);
        assert!(RiskLevel::Low < RiskLevel::Medium);
        assert!(RiskLevel::Medium < RiskLevel::High);
        assert!(RiskLevel::High < RiskLevel::Emergency);
    }

    #[test]
    fn test_object_type_serialization() {
        let types = vec![
            ObjectType::Payload,
            ObjectType::RocketBody,
            ObjectType::Debris,
            ObjectType::Unknown,
        ];

        for obj_type in types {
            let json = serde_json::to_string(&obj_type).expect("Failed to serialize");
            let parsed: ObjectType = serde_json::from_str(&json).expect("Failed to deserialize");
            assert_eq!(parsed, obj_type);
        }
    }

    #[test]
    fn test_orbital_object_entry_serialization() {
        let obj = OrbitalObjectEntry {
            norad_id: 25544,
            name: "ISS (ZARYA)".to_string(),
            object_type: ObjectType::Payload,
            launch_date: Some(Utc::now()),
            decay_date: None,
            owner_country: Some("ISS".to_string()),
            data_source: DataSourceSimple::SpaceTrack,
            metadata: HashMap::new(),
        };

        let json = serde_json::to_string(&obj).expect("Failed to serialize");
        let parsed: OrbitalObjectEntry =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(parsed.norad_id, 25544);
        assert_eq!(parsed.name, "ISS (ZARYA)");
    }

    #[test]
    fn test_tle_data_serialization() {
        let tle = TleData {
            norad_id: 25544,
            line1: "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997"
                .to_string(),
            line2: "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577"
                .to_string(),
            epoch: Utc::now(),
            source: DataSourceSimple::SpaceTrack,
        };

        let json = serde_json::to_string(&tle).expect("Failed to serialize");
        let parsed: TleData = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(parsed.norad_id, 25544);
        assert!(parsed.line1.starts_with("1 25544"));
    }

    #[test]
    fn test_conjunction_assessment_serialization() {
        let assessment = ConjunctionAssessment {
            primary_norad_id: 25544,
            secondary_norad_id: 49863,
            tca: Utc::now(),
            miss_distance_km: 0.5,
            relative_velocity_kms: 14.5,
            collision_probability: 1e-5,
            pc_method: PcMethod::Alfano2D,
            risk_level: RiskLevel::Medium,
            hard_body_radius_m: 20.0,
            screening_volume_km: 5.0,
        };

        let json = serde_json::to_string(&assessment).expect("Failed to serialize");
        let parsed: ConjunctionAssessment =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(parsed.primary_norad_id, 25544);
        assert_eq!(parsed.secondary_norad_id, 49863);
        assert_eq!(parsed.risk_level, RiskLevel::Medium);
    }

    #[test]
    fn test_bounty_entry_serialization() {
        let bounty = BountyEntry {
            debris_norad_id: 49863,
            bounty_amount: 100000,
            currency: "USD".to_string(),
            sponsor: "ESA".to_string(),
            deadline: Some(Utc::now()),
            requirements: BountyRequirements {
                min_mass_kg: Some(100.0),
                max_altitude_km: Some(600.0),
                debris_type: Some(ObjectType::Debris),
                removal_method: None,
            },
            status: BountyStatus::Active,
        };

        let json = serde_json::to_string(&bounty).expect("Failed to serialize");
        let parsed: BountyEntry = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(parsed.debris_norad_id, 49863);
        assert_eq!(parsed.bounty_amount, 100000);
    }
}

// =============================================================================
// Zero-Knowledge Orbit Proofs
// =============================================================================

/// Hash commitment to orbital data.
///
/// An operator commits to their orbital state by publishing a hash of their
/// ephemeris data + a random nonce. Later, they can reveal the data and nonce
/// to prove the commitment, or provide ZK proofs about properties of the
/// committed orbit without revealing the data itself.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OrbitCommitment {
    /// SHA3-256 hash of (ephemeris_data || nonce)
    pub commitment: [u8; 32],
    /// Object this commitment is for
    pub norad_id: u32,
    /// Epoch of the committed orbital state
    pub epoch: SpaceTimestamp,
    /// Agent who created the commitment
    pub committer: AgentPubKey,
    /// When this commitment was published
    pub created_at: SpaceTimestamp,
    /// Optional expiration (commitments become stale)
    pub expires_at: Option<SpaceTimestamp>,
}

/// Proof that an orbit satisfies certain properties without revealing the orbit.
///
/// This is the on-chain attestation structure. The actual ZK proof blob is
/// opaque to Holochain — it gets verified by calling `verify_orbit_proof`
/// which checks the proof against the commitment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrbitProof {
    /// Which commitment this proof refers to
    pub commitment: [u8; 32],
    /// What property is being proven
    pub claim: OrbitClaim,
    /// The opaque proof blob (format depends on `proof_system`)
    pub proof_data: Vec<u8>,
    /// Which proof system generated this proof
    pub proof_system: ProofSystem,
    /// Agent who generated the proof
    pub prover: AgentPubKey,
    /// When the proof was generated
    pub created_at: SpaceTimestamp,
}

/// Claims that can be proven about an orbit without revealing the orbit.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OrbitClaim {
    /// Prove altitude is within a range (km) at a given time.
    /// Does NOT reveal the exact altitude.
    AltitudeRange {
        min_km: f64,
        max_km: f64,
        at_time: SpaceTimestamp,
    },

    /// Prove inclination is within a range (degrees).
    InclinationRange { min_deg: f64, max_deg: f64 },

    /// Prove minimum miss distance from another object exceeds a threshold.
    /// The core use case: "I won't come closer than X km to object Y."
    CollisionFreedom {
        other_norad_id: u32,
        /// Other object's commitment (so we can verify both sides)
        other_commitment: Option<[u8; 32]>,
        min_miss_distance_km: f64,
        window_start: SpaceTimestamp,
        window_end: SpaceTimestamp,
    },

    /// Prove the orbit will decay below a threshold altitude by a deadline.
    /// Used for post-mission disposal compliance.
    DisposalCompliance {
        max_altitude_km: f64,
        by_time: SpaceTimestamp,
    },

    /// Prove the orbit is within a licensed orbital slot (GEO operators).
    SlotCompliance {
        longitude_deg: f64,
        tolerance_deg: f64,
    },

    /// Prove the orbit avoids a protected zone (e.g., ISS corridor).
    ZoneAvoidance {
        zone_id: String,
        min_altitude_km: f64,
        max_altitude_km: f64,
        min_inclination_deg: f64,
        max_inclination_deg: f64,
        window_start: SpaceTimestamp,
        window_end: SpaceTimestamp,
    },
}

/// Supported zero-knowledge proof systems.
///
/// The actual proof generation/verification is off-chain. These tags tell
/// verifiers which library to use for checking the proof blob.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProofSystem {
    /// Hash-based commitment reveal (not truly ZK, but useful for bootstrapping)
    HashReveal,
    /// Bulletproofs range proofs (good for altitude/distance bounds)
    Bulletproofs,
    /// Groth16 SNARKs (succinct, constant-size proofs)
    Groth16,
    /// PLONK (universal trusted setup)
    Plonk,
}

/// Result of verifying an orbit proof.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProofVerification {
    /// Proof is valid — the claim holds
    Valid,
    /// Proof is invalid — either the math doesn't check out or it doesn't
    /// match the commitment
    Invalid { reason: String },
    /// The proof system is not supported by this verifier
    UnsupportedSystem,
    /// The commitment has expired
    CommitmentExpired,
}

/// A collision-freedom certificate: a verified proof that two objects
/// will not collide within a time window.
///
/// This is the "receipt" that traffic coordination can use to skip
/// full conjunction screening between two objects whose operators have
/// both provided valid collision-freedom proofs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollisionFreedomCertificate {
    /// Primary object
    pub primary_norad_id: u32,
    /// Secondary object
    pub secondary_norad_id: u32,
    /// Primary operator's commitment
    pub primary_commitment: [u8; 32],
    /// Secondary operator's commitment
    pub secondary_commitment: [u8; 32],
    /// Minimum guaranteed miss distance (km)
    pub min_miss_distance_km: f64,
    /// Time window this certificate covers
    pub window_start: SpaceTimestamp,
    pub window_end: SpaceTimestamp,
    /// Hash of the proof that was verified
    pub proof_hash: [u8; 32],
    /// Who verified this certificate
    pub verifier: AgentPubKey,
    /// Verification timestamp
    pub verified_at: SpaceTimestamp,
    /// Verification result
    pub status: ProofVerification,
}

/// Input for creating an orbit commitment on-chain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateCommitmentInput {
    /// The orbital data being committed to (will be hashed, NOT stored)
    pub ephemeris_data: Vec<u8>,
    /// Random nonce for the commitment
    pub nonce: [u8; 32],
    /// Object this commitment is for
    pub norad_id: u32,
    /// Epoch of the orbital state
    pub epoch: SpaceTimestamp,
    /// Optional expiration
    pub expires_at: Option<SpaceTimestamp>,
}

impl CreateCommitmentInput {
    /// Compute the commitment hash: SHA3-256(ephemeris_data || nonce)
    pub fn compute_commitment(&self) -> [u8; 32] {
        let mut data = self.ephemeris_data.clone();
        data.extend_from_slice(&self.nonce);
        hash_data(&data)
    }
}

/// Input for submitting an orbit proof on-chain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitProofInput {
    /// The commitment this proof is for
    pub commitment_hash: ActionHash,
    /// What is being claimed
    pub claim: OrbitClaim,
    /// The proof blob
    pub proof_data: Vec<u8>,
    /// Which proof system
    pub proof_system: ProofSystem,
}

/// Input for verifying a collision-freedom proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyCollisionFreedomInput {
    /// Primary object's proof hash
    pub primary_proof_hash: ActionHash,
    /// Secondary object's proof hash
    pub secondary_proof_hash: ActionHash,
}

// =============================================================================
// Structured Error Types
// =============================================================================

/// Machine-readable error codes for mycelix-space coordinator zomes.
///
/// Clients can match on the `code` field of the JSON-serialized error to
/// handle specific failure modes without parsing human-readable messages.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpaceError {
    /// Machine-readable error code (e.g. "INVALID_NORAD_ID")
    pub code: SpaceErrorCode,
    /// Human-readable description
    pub message: String,
    /// Optional context (field name, value, etc.)
    pub context: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SpaceErrorCode {
    // --- General ---
    NotFound,
    Unauthorized,
    InvalidInput,

    // --- Orbital Objects ---
    InvalidNoradId,
    InvalidDesignator,
    TleParseError,
    DuplicateObject,

    // --- Observations ---
    InvalidMeasurement,
    InvalidLocation,
    InvalidSensorConfig,
    SensorNotFound,

    // --- Conjunctions ---
    SelfConjunction,
    InvalidProbability,
    InvalidMissDistance,
    EventNotFound,

    // --- Debris Bounties ---
    InvalidBountyAmount,
    InvalidStateTransition,
    BountyNotFound,
    BountyNotOpen,
    AlreadyClaimed,

    // --- Traffic Control ---
    SessionNotFound,
    InvalidProposal,
    AlreadySigned,
    SameSignerError,
    ConjunctionNotVerified,

    // --- Rate Limiting ---
    RateLimited,
}

impl SpaceError {
    pub fn new(code: SpaceErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            context: None,
        }
    }

    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }

    /// Convert to a Holochain WasmError for use in coordinator externs.
    pub fn into_wasm_error(self) -> WasmError {
        let json = serde_json::to_string(&self).unwrap_or_else(|_| self.message.clone());
        wasm_error!(WasmErrorInner::Guest(json))
    }

    /// Convert to a ValidateCallbackResult::Invalid for use in integrity validation.
    pub fn into_invalid(self) -> ValidateCallbackResult {
        ValidateCallbackResult::Invalid(
            serde_json::to_string(&self).unwrap_or_else(|_| self.message.clone()),
        )
    }
}

impl std::fmt::Display for SpaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.code, self.message)?;
        if let Some(ref ctx) = self.context {
            write!(f, " ({})", ctx)?;
        }
        Ok(())
    }
}

// =============================================================================
// Pagination primitives
// =============================================================================

/// Pagination parameters for link-based queries.
///
/// Holochain `get_links` returns all links at once. Pagination is applied
/// after fetching links but *before* resolving individual entries, so we
/// avoid unnecessary DHT gets for pages we don't need.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PaginationParams {
    /// Maximum number of entries to return (default: 50)
    pub limit: Option<u32>,
    /// Number of links to skip before collecting results
    pub offset: Option<u32>,
}

impl PaginationParams {
    pub fn effective_limit(&self) -> usize {
        self.limit.unwrap_or(50).min(500) as usize
    }

    pub fn effective_offset(&self) -> usize {
        self.offset.unwrap_or(0) as usize
    }
}

/// Paginated response wrapper.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    /// The page of results
    pub items: Vec<T>,
    /// Total number of links available (before pagination)
    pub total: u32,
    /// Offset used for this page
    pub offset: u32,
    /// Limit used for this page
    pub limit: u32,
    /// Whether there are more results after this page
    pub has_more: bool,
}

// =============================================================================
// Universal Trust Fabric — Space-Domain Requirements
// =============================================================================

pub use mycelix_bridge_common::sovereign_gate::{CivicRequirement, CivicTier, SovereignDimension};

/// Observer — anyone can submit TLEs (more data = better catalog)
pub fn requirement_for_tle_submission() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Observer,
        min_dimensions: vec![],
    }
}

/// Observer — low bar for sensor data, chi-square gating handles quality
pub fn requirement_for_observation() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Observer,
        min_dimensions: vec![],
    }
}

/// Participant — conjunction events affect operational decisions
pub fn requirement_for_conjunction_creation() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Participant,
        min_dimensions: vec![(SovereignDimension::EpistemicIntegrity, 0.25)],
    }
}

/// Citizen — risk updates influence maneuver decisions
pub fn requirement_for_risk_update() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Citizen,
        min_dimensions: vec![],
    }
}

/// Citizen — fusion results affect downstream screening
pub fn requirement_for_fusion() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Citizen,
        min_dimensions: vec![],
    }
}

/// Steward — financial commitment requires verified identity
pub fn requirement_for_bounty_creation() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Steward,
        min_dimensions: vec![
            (SovereignDimension::EpistemicIntegrity, 0.5),
            (SovereignDimension::CivicParticipation, 0.3),
        ],
    }
}

/// Citizen — claiming a bounty requires operational capability
pub fn requirement_for_bounty_claim() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Citizen,
        min_dimensions: vec![(SovereignDimension::EpistemicIntegrity, 0.3)],
    }
}

/// Steward — verification evidence integrity is critical
pub fn requirement_for_bounty_verification() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Steward,
        min_dimensions: vec![
            (SovereignDimension::EpistemicIntegrity, 0.5),
            (SovereignDimension::CivicParticipation, 0.2),
        ],
    }
}

/// Citizen — initiating negotiation requires coordination authority
pub fn requirement_for_negotiation() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Citizen,
        min_dimensions: vec![
            (SovereignDimension::EpistemicIntegrity, 0.3),
            (SovereignDimension::CivicParticipation, 0.2),
        ],
    }
}

/// Steward — cosigning is a binding operational agreement
pub fn requirement_for_agreement_signing() -> CivicRequirement {
    CivicRequirement {
        min_tier: CivicTier::Steward,
        min_dimensions: vec![
            (SovereignDimension::EpistemicIntegrity, 0.5),
            (SovereignDimension::CivicParticipation, 0.3),
        ],
    }
}

// =============================================================================
// Space consciousness gate (bootstrap-aware)
// =============================================================================

/// Best-effort civic gate for space operations.
///
/// Attempts a cross-role call to the identity cluster to fetch a
/// `SovereignCredential` (8D). Falls back to bootstrap mode if the identity
/// cluster is unavailable (standalone space hApp or network bootstrap).
pub fn gate_space_operation(
    requirement: &CivicRequirement,
    action_name: &str,
) -> hdk::prelude::ExternResult<()> {
    use hdk::prelude::*;
    use mycelix_bridge_common::sovereign_gate::{DimensionWeights, SovereignCredential};

    let agent = agent_info()?.agent_initial_pubkey;
    let did = format!("did:mycelix:{}", agent);

    // Try cross-role call to identity cluster's bridge zome
    match call(
        CallTargetCell::OtherRole("identity".into()),
        ZomeName::new("identity_bridge"),
        FunctionName::new("issue_sovereign_credential"),
        None,
        did,
    ) {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            match extern_io.decode::<SovereignCredential>() {
                Ok(cred) => {
                    let weights = DimensionWeights::governance();
                    if !cred.profile.meets_requirement(requirement, &weights) {
                        let tier = cred.profile.tier(&weights);
                        return Err(wasm_error!(WasmErrorInner::Guest(format!(
                            "Civic gate denied for action '{}': requires {:?} tier, \
                             agent has {:?}",
                            action_name, requirement.min_tier, tier
                        ))));
                    }
                    Ok(())
                }
                Err(_) => {
                    // Credential decode failed — bootstrap mode
                    debug!(
                        "Space civic gate: credential decode failed, \
                         bootstrap mode for: {}",
                        action_name
                    );
                    Ok(())
                }
            }
        }
        Ok(ZomeCallResponse::NetworkError(e)) => {
            // Identity cluster unreachable — allow in bootstrap mode
            debug!(
                "Space civic gate: identity cluster unreachable ({}), \
                 bootstrap mode for: {}",
                e, action_name
            );
            Ok(())
        }
        Ok(_) => {
            // Unexpected response variant — allow in bootstrap mode
            debug!(
                "Space civic gate: unexpected response, \
                 bootstrap mode for: {}",
                action_name
            );
            Ok(())
        }
        Err(_) => {
            // Zome not found — standalone space hApp without identity cluster
            debug!(
                "Space consciousness gate: no identity cluster, \
                 bootstrap mode for: {}",
                action_name
            );
            Ok(())
        }
    }
}

// =============================================================================
// Trust Fabric Tests
// =============================================================================

#[cfg(test)]
mod trust_fabric_tests {
    use super::*;

    #[test]
    fn test_requirement_tiers_are_ordered() {
        // Verify tier progression: Observer < Participant < Citizen < Steward
        assert!(
            (requirement_for_tle_submission().min_tier as u8)
                <= (requirement_for_conjunction_creation().min_tier as u8)
        );
        assert!(
            (requirement_for_conjunction_creation().min_tier as u8)
                <= (requirement_for_risk_update().min_tier as u8)
        );
        assert!(
            (requirement_for_risk_update().min_tier as u8)
                <= (requirement_for_bounty_creation().min_tier as u8)
        );
    }

    #[test]
    fn test_observer_requirements_have_no_minimums() {
        let req = requirement_for_tle_submission();
        assert_eq!(req.min_tier, CivicTier::Observer);
        assert!(req.min_dimensions.is_empty());

        let req = requirement_for_observation();
        assert_eq!(req.min_tier, CivicTier::Observer);
        assert!(req.min_dimensions.is_empty());
    }

    #[test]
    fn test_steward_requirements_have_dimension_minimums() {
        let req = requirement_for_bounty_creation();
        assert_eq!(req.min_tier, CivicTier::Steward);
        assert_eq!(req.min_dimensions.len(), 2);

        let req = requirement_for_agreement_signing();
        assert_eq!(req.min_tier, CivicTier::Steward);
        assert_eq!(req.min_dimensions.len(), 2);

        let req = requirement_for_bounty_verification();
        assert_eq!(req.min_tier, CivicTier::Steward);
        assert_eq!(req.min_dimensions.len(), 2);
    }

    #[test]
    fn test_citizen_requirements() {
        let req = requirement_for_risk_update();
        assert_eq!(req.min_tier, CivicTier::Citizen);

        let req = requirement_for_fusion();
        assert_eq!(req.min_tier, CivicTier::Citizen);

        let req = requirement_for_bounty_claim();
        assert_eq!(req.min_tier, CivicTier::Citizen);
        assert_eq!(req.min_dimensions.len(), 1);

        let req = requirement_for_negotiation();
        assert_eq!(req.min_tier, CivicTier::Citizen);
        assert_eq!(req.min_dimensions.len(), 2);
    }

    #[test]
    fn test_participant_conjunction_creation() {
        let req = requirement_for_conjunction_creation();
        assert_eq!(req.min_tier, CivicTier::Participant);
        assert_eq!(req.min_dimensions.len(), 1);
        assert_eq!(
            req.min_dimensions[0],
            (SovereignDimension::EpistemicIntegrity, 0.25)
        );
    }

    #[test]
    fn test_bounty_creation_is_stricter_than_claim() {
        let create = requirement_for_bounty_creation();
        let claim = requirement_for_bounty_claim();
        assert!(create.min_tier >= claim.min_tier);
        assert!(create.min_dimensions.len() >= claim.min_dimensions.len());
    }

    #[test]
    fn test_agreement_signing_is_stricter_than_negotiation() {
        let sign = requirement_for_agreement_signing();
        let negotiate = requirement_for_negotiation();
        assert!(sign.min_tier >= negotiate.min_tier);
    }

    #[test]
    fn test_all_dimension_scores_bounded() {
        let all_reqs = vec![
            requirement_for_tle_submission(),
            requirement_for_observation(),
            requirement_for_conjunction_creation(),
            requirement_for_risk_update(),
            requirement_for_fusion(),
            requirement_for_bounty_creation(),
            requirement_for_bounty_claim(),
            requirement_for_bounty_verification(),
            requirement_for_negotiation(),
            requirement_for_agreement_signing(),
        ];
        for req in &all_reqs {
            for &(dim, val) in &req.min_dimensions {
                assert!(val >= 0.0 && val <= 1.0, "{:?} out of bounds: {}", dim, val);
            }
        }
    }
}
