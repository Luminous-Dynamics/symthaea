// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared types for mycelix-position: decentralized cooperative positioning.
//!
//! Defines entry types, error handling, quality scoring, and consciousness
//! gating requirements shared across all positioning zomes.

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// TIMESTAMPS
// ============================================================================

/// Microsecond-precision timestamp.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PositionTimestamp(pub i64);

impl PositionTimestamp {
    pub fn now() -> Self {
        Self(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as i64)
                .unwrap_or(0),
        )
    }
}

// ============================================================================
// QUALITY SCORING
// ============================================================================

/// Position quality score (0-100).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PositionQuality(pub u8);

impl PositionQuality {
    pub fn new(score: u8) -> Self {
        Self(score.min(100))
    }
    pub fn is_high(&self) -> bool {
        self.0 >= 80
    }
    pub fn is_acceptable(&self) -> bool {
        self.0 >= 50
    }
    pub fn as_f64(&self) -> f64 {
        self.0 as f64 / 100.0
    }
}

// ============================================================================
// RANGING METHOD
// ============================================================================

/// Ranging technology used for a measurement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RangingMethod {
    UltraWideband,
    LoRaToA,
    LoRaTDoA,
    RssiPathLoss,
    WifiRtt,
    ManualSurvey,
    MeshtasticHops,
}

// ============================================================================
// SURVEY METHOD
// ============================================================================

/// How an anchor's position was determined.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurveyMethod {
    /// Professional survey with known accuracy.
    ProfessionalSurvey,
    /// GPS fix at time of registration.
    GpsFix,
    /// Self-reported position (lowest trust).
    SelfReported,
    /// Certified by a Steward-tier agent.
    StewardCertified,
    /// Derived from cooperative positioning.
    CooperativeEstimate,
}

// ============================================================================
// ENTRY TYPES (used by integrity zomes)
// ============================================================================

/// An anchor node with a known position.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct AnchorNode {
    pub node_id: String,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub altitude_m: f64,
    pub accuracy_m: f64,
    pub survey_method: SurveyMethod,
    pub certified_by: Option<AgentPubKey>,
    pub registered_by: AgentPubKey,
    pub registered_at: PositionTimestamp,
}

/// Certification of an anchor's accuracy by a Steward.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct AnchorCertification {
    pub anchor_node_id: String,
    pub certifier: AgentPubKey,
    pub verified_accuracy_m: f64,
    pub certification_method: String,
    pub certified_at: PositionTimestamp,
}

/// A range measurement between two nodes.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct RangeMeasurement {
    pub from_node: String,
    pub to_node: String,
    pub range_m: f64,
    pub sigma_m: f64,
    pub method: RangingMethod,
    pub quality: PositionQuality,
    pub measured_by: AgentPubKey,
    pub measured_at: PositionTimestamp,
}

/// A fused position estimate for a node.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct PositionEstimateEntry {
    pub node_id: String,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub altitude_m: f64,
    /// 3×3 covariance matrix (row-major, 9 elements).
    pub covariance: Vec<f64>,
    pub quality: PositionQuality,
    pub contributing_anchors: Vec<String>,
    pub algorithm: String,
    pub computed_by: AgentPubKey,
    pub computed_at: PositionTimestamp,
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

/// Error codes for position operations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PositionErrorCode {
    InvalidInput,
    InsufficientAnchors,
    DegenerateGeometry,
    Unauthorized,
    NotFound,
    ComputationFailed,
}

/// Structured error for position operations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PositionError {
    pub code: PositionErrorCode,
    pub message: String,
    pub context: Option<String>,
}

impl PositionError {
    pub fn new(code: PositionErrorCode, message: impl Into<String>) -> Self {
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

    pub fn into_wasm_error(self) -> WasmError {
        let json = serde_json::to_string(&self).unwrap_or_else(|_| self.message.clone());
        wasm_error!(json)
    }
}

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

/// Validate geodetic coordinates.
pub fn validate_geodetic(lat: f64, lon: f64, alt: f64) -> Result<(), String> {
    if lat < -90.0 || lat > 90.0 {
        return Err(format!("Latitude {} out of range [-90, 90]", lat));
    }
    if lon < -180.0 || lon > 180.0 {
        return Err(format!("Longitude {} out of range [-180, 180]", lon));
    }
    if alt < -12_000.0 || alt > 100_000_000.0 {
        return Err(format!("Altitude {} out of range [-12km, 100,000km]", alt));
    }
    Ok(())
}

/// Validate a range measurement.
pub fn validate_range(range_m: f64, sigma_m: f64) -> Result<(), String> {
    if range_m <= 0.0 {
        return Err(format!("Range must be positive, got {}", range_m));
    }
    if sigma_m <= 0.0 {
        return Err(format!("Sigma must be positive, got {}", sigma_m));
    }
    if range_m > 1_000_000.0 {
        return Err(format!("Range {} exceeds 1000km maximum", range_m));
    }
    Ok(())
}

/// Validate a node ID (non-empty, reasonable length).
pub fn validate_node_id(node_id: &str) -> Result<(), String> {
    if node_id.is_empty() {
        return Err("Node ID cannot be empty".to_string());
    }
    if node_id.len() > 256 {
        return Err(format!(
            "Node ID too long: {} chars (max 256)",
            node_id.len()
        ));
    }
    Ok(())
}

// ============================================================================
// CONSCIOUSNESS GATING REQUIREMENTS
// ============================================================================

/// Consciousness tier requirements for positioning operations.
/// These mirror the mycelix-space pattern but are self-contained
/// (no dependency on mycelix-bridge-common for standalone deployment).

/// Minimum identity score for anchor registration (Participant tier).
pub const ANCHOR_REGISTRATION_IDENTITY: f64 = 0.15;
/// Minimum identity score for anchor certification (Steward tier).
pub const ANCHOR_CERTIFICATION_IDENTITY: f64 = 0.50;
/// Minimum community score for anchor certification.
pub const ANCHOR_CERTIFICATION_COMMUNITY: f64 = 0.30;
/// Minimum identity score for position computation (Citizen tier).
pub const POSITION_COMPUTATION_IDENTITY: f64 = 0.25;
/// Minimum identity score for range submission (Participant tier).
pub const RANGE_SUBMISSION_IDENTITY: f64 = 0.15;

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_score_bounds() {
        let q = PositionQuality::new(150);
        assert_eq!(q.0, 100);
        assert!(q.is_high());
        assert!((q.as_f64() - 1.0).abs() < 0.01);
    }

    #[test]
    fn validate_geodetic_valid() {
        assert!(validate_geodetic(-26.2041, 28.0473, 1753.0).is_ok());
        assert!(validate_geodetic(90.0, 180.0, 0.0).is_ok());
        assert!(validate_geodetic(-90.0, -180.0, -500.0).is_ok());
    }

    #[test]
    fn validate_geodetic_invalid() {
        assert!(validate_geodetic(91.0, 0.0, 0.0).is_err());
        assert!(validate_geodetic(0.0, 181.0, 0.0).is_err());
    }

    #[test]
    fn validate_range_valid() {
        assert!(validate_range(100.0, 5.0).is_ok());
    }

    #[test]
    fn validate_range_invalid() {
        assert!(validate_range(-1.0, 5.0).is_err());
        assert!(validate_range(100.0, 0.0).is_err());
        assert!(validate_range(2_000_000.0, 5.0).is_err());
    }

    #[test]
    fn validate_node_id_valid() {
        assert!(validate_node_id("node-123").is_ok());
    }

    #[test]
    fn validate_node_id_invalid() {
        assert!(validate_node_id("").is_err());
        let long = "x".repeat(300);
        assert!(validate_node_id(&long).is_err());
    }

    #[test]
    fn error_serialization() {
        let err = PositionError::new(PositionErrorCode::InvalidInput, "bad data")
            .with_context("test context");
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("InvalidInput"));
        assert!(json.contains("bad data"));
    }

    // ─── Additional PositionQuality Tests ───────────────────────────

    #[test]
    fn quality_zero_is_not_acceptable() {
        let q = PositionQuality::new(0);
        assert!(!q.is_high());
        assert!(!q.is_acceptable());
        assert!((q.as_f64()).abs() < 0.01);
    }

    #[test]
    fn quality_50_is_acceptable_not_high() {
        let q = PositionQuality::new(50);
        assert!(!q.is_high());
        assert!(q.is_acceptable());
    }

    #[test]
    fn quality_79_is_acceptable_not_high() {
        let q = PositionQuality::new(79);
        assert!(!q.is_high());
        assert!(q.is_acceptable());
    }

    #[test]
    fn quality_80_is_high() {
        let q = PositionQuality::new(80);
        assert!(q.is_high());
        assert!(q.is_acceptable());
        assert!((q.as_f64() - 0.80).abs() < 0.01);
    }

    // ─── Validation Edge Cases ──────────────────────────────────────

    #[test]
    fn validate_geodetic_deep_ocean() {
        // Mariana Trench depth is about -11,000m
        assert!(validate_geodetic(11.35, 142.2, -11_000.0).is_ok());
    }

    #[test]
    fn validate_geodetic_altitude_too_low() {
        assert!(validate_geodetic(0.0, 0.0, -13_000.0).is_err());
    }

    #[test]
    fn validate_geodetic_geostationary_orbit() {
        // GEO orbit ~35,786 km altitude
        assert!(validate_geodetic(0.0, 0.0, 35_786_000.0).is_ok());
    }

    #[test]
    fn validate_range_exactly_max() {
        assert!(validate_range(1_000_000.0, 1.0).is_ok());
    }

    #[test]
    fn validate_range_exceeds_max() {
        assert!(validate_range(1_000_001.0, 1.0).is_err());
    }

    #[test]
    fn validate_node_id_exactly_256() {
        let id = "x".repeat(256);
        assert!(validate_node_id(&id).is_ok());
    }

    #[test]
    fn validate_node_id_257_rejected() {
        let id = "x".repeat(257);
        assert!(validate_node_id(&id).is_err());
    }

    // ─── Ranging Method Serde ───────────────────────────────────────

    #[test]
    fn ranging_method_all_variants_serde() {
        let methods = [
            RangingMethod::UltraWideband,
            RangingMethod::LoRaToA,
            RangingMethod::LoRaTDoA,
            RangingMethod::RssiPathLoss,
            RangingMethod::WifiRtt,
            RangingMethod::ManualSurvey,
            RangingMethod::MeshtasticHops,
        ];
        for m in &methods {
            let json = serde_json::to_string(m).unwrap();
            let restored: RangingMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(&restored, m);
        }
    }

    // ─── Survey Method Serde ────────────────────────────────────────

    #[test]
    fn survey_method_all_variants_serde() {
        let methods = [
            SurveyMethod::ProfessionalSurvey,
            SurveyMethod::GpsFix,
            SurveyMethod::SelfReported,
            SurveyMethod::StewardCertified,
            SurveyMethod::CooperativeEstimate,
        ];
        for m in &methods {
            let json = serde_json::to_string(m).unwrap();
            let restored: SurveyMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(&restored, m);
        }
    }

    // ─── Error Types ────────────────────────────────────────────────

    #[test]
    fn position_error_all_codes_serde() {
        let codes = [
            PositionErrorCode::InvalidInput,
            PositionErrorCode::InsufficientAnchors,
            PositionErrorCode::DegenerateGeometry,
            PositionErrorCode::Unauthorized,
            PositionErrorCode::NotFound,
            PositionErrorCode::ComputationFailed,
        ];
        for code in &codes {
            let err = PositionError::new(code.clone(), "test");
            let json = serde_json::to_string(&err).unwrap();
            assert!(!json.is_empty());
        }
    }

    #[test]
    fn position_error_without_context() {
        let err = PositionError::new(PositionErrorCode::NotFound, "missing");
        assert!(err.context.is_none());
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("NotFound"));
    }

    // ─── Constants ──────────────────────────────────────────────────

    #[test]
    fn consciousness_gating_tier_ordering() {
        // Registration < computation < certification
        assert!(ANCHOR_REGISTRATION_IDENTITY < POSITION_COMPUTATION_IDENTITY);
        assert!(POSITION_COMPUTATION_IDENTITY < ANCHOR_CERTIFICATION_IDENTITY);
    }

    #[test]
    fn range_submission_identity_matches_registration() {
        assert_eq!(RANGE_SUBMISSION_IDENTITY, ANCHOR_REGISTRATION_IDENTITY);
    }
}
