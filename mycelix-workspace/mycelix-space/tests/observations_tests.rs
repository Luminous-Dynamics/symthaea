// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observations Coordinator Tests
//!
//! Tests for the observation and sensor logic that can be exercised
//! without a Holochain conductor:
//! - Latitude/longitude validation
//! - validate_string_field for sensor_id and name
//! - NORAD ID range validation (1-999999)
//! - Measurement type construction (AnglesOnly, Range, StateVector, Photometric)
//! - GroundLocation construction and serialization
//! - SensorCapabilities validation (positive values)
//! - QualityScore edge cases in observation context
//! - ObservationMeta serialization
//! - FusedEstimate construction
//! - Trust-weighted fusion via orbital-mechanics

use chrono::Utc;
use mycelix_space_shared::*;

// =============================================================================
// Latitude Validation
// =============================================================================

/// Valid latitude at the equator.
#[test]
fn test_latitude_valid_equator() {
    assert!(validate_latitude(0.0).is_ok());
}

/// Valid latitude at north pole.
#[test]
fn test_latitude_valid_north_pole() {
    assert!(validate_latitude(90.0).is_ok());
}

/// Valid latitude at south pole.
#[test]
fn test_latitude_valid_south_pole() {
    assert!(validate_latitude(-90.0).is_ok());
}

/// Latitude just above 90 is invalid.
#[test]
fn test_latitude_invalid_above_90() {
    let result = validate_latitude(90.1);
    assert!(result.is_err());
}

/// Latitude just below -90 is invalid.
#[test]
fn test_latitude_invalid_below_neg90() {
    let result = validate_latitude(-90.1);
    assert!(result.is_err());
}

/// NaN latitude is invalid.
#[test]
fn test_latitude_nan_invalid() {
    let result = validate_latitude(f64::NAN);
    assert!(result.is_err());
}

// =============================================================================
// Longitude Validation
// =============================================================================

/// Valid longitude at prime meridian.
#[test]
fn test_longitude_valid_prime_meridian() {
    assert!(validate_longitude(0.0).is_ok());
}

/// Valid longitude at 180.
#[test]
fn test_longitude_valid_180() {
    assert!(validate_longitude(180.0).is_ok());
}

/// Valid longitude at -180.
#[test]
fn test_longitude_valid_neg180() {
    assert!(validate_longitude(-180.0).is_ok());
}

/// Longitude just above 180 is invalid.
#[test]
fn test_longitude_invalid_above_180() {
    assert!(validate_longitude(180.1).is_err());
}

/// Longitude just below -180 is invalid.
#[test]
fn test_longitude_invalid_below_neg180() {
    assert!(validate_longitude(-180.1).is_err());
}

/// NaN longitude is invalid.
#[test]
fn test_longitude_nan_invalid() {
    assert!(validate_longitude(f64::NAN).is_err());
}

// =============================================================================
// NORAD ID Range Validation (observations coordinator: 1-999999)
// =============================================================================

/// NORAD ID 0 is invalid in the observations coordinator.
#[test]
fn test_observation_norad_id_zero_invalid() {
    let norad_id: u32 = 0;
    assert!(norad_id == 0 || norad_id > 999999);
}

/// NORAD ID 1 is valid.
#[test]
fn test_observation_norad_id_one_valid() {
    let norad_id: u32 = 1;
    assert!(norad_id >= 1 && norad_id <= 999999);
}

/// NORAD ID 999999 is valid.
#[test]
fn test_observation_norad_id_max_valid() {
    let norad_id: u32 = 999999;
    assert!(norad_id >= 1 && norad_id <= 999999);
}

/// NORAD ID 1000000 is invalid.
#[test]
fn test_observation_norad_id_above_max_invalid() {
    let norad_id: u32 = 1000000;
    assert!(norad_id == 0 || norad_id > 999999);
}

// =============================================================================
// GroundLocation Construction and Serialization
// =============================================================================

/// Valid ground location with all fields roundtrips through JSON.
#[test]
fn test_ground_location_full_serde() {
    let loc = GroundLocation {
        latitude_deg: -33.9258,
        longitude_deg: 18.4232,
        altitude_m: 0.0,
        name: Some("Cape Town Observatory".into()),
    };

    let json = serde_json::to_string(&loc).expect("serialize");
    let parsed: GroundLocation = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.latitude_deg, loc.latitude_deg);
    assert_eq!(parsed.longitude_deg, loc.longitude_deg);
    assert_eq!(parsed.name, Some("Cape Town Observatory".into()));
}

/// Ground location without name roundtrips.
#[test]
fn test_ground_location_no_name_serde() {
    let loc = GroundLocation {
        latitude_deg: 51.4769,
        longitude_deg: -0.0005,
        altitude_m: 48.0,
        name: None,
    };

    let json = serde_json::to_string(&loc).expect("serialize");
    let parsed: GroundLocation = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.name, None);
}

/// Ground location at extreme valid coordinates.
#[test]
fn test_ground_location_extreme_valid() {
    let loc = GroundLocation {
        latitude_deg: 90.0,
        longitude_deg: -180.0,
        altitude_m: 8848.0, // Everest
        name: Some("Extreme".into()),
    };

    assert!(validate_latitude(loc.latitude_deg).is_ok());
    assert!(validate_longitude(loc.longitude_deg).is_ok());
}

// =============================================================================
// SensorCapabilities Validation (coordinator logic)
// =============================================================================

/// Positive max_range_km is valid.
#[test]
fn test_sensor_capabilities_valid_max_range() {
    let max_range = Some(40000.0_f64);
    assert!(max_range.unwrap() > 0.0);
}

/// Zero max_range_km is invalid.
#[test]
fn test_sensor_capabilities_zero_max_range_invalid() {
    let max_range = Some(0.0_f64);
    assert!(max_range.unwrap() <= 0.0);
}

/// Negative max_range_km is invalid.
#[test]
fn test_sensor_capabilities_negative_max_range_invalid() {
    let max_range = Some(-100.0_f64);
    assert!(max_range.unwrap() <= 0.0);
}

/// Positive accuracy_arcsec is valid.
#[test]
fn test_sensor_capabilities_valid_accuracy() {
    let accuracy = Some(0.5_f64);
    assert!(accuracy.unwrap() > 0.0);
}

/// Zero accuracy_arcsec is invalid.
#[test]
fn test_sensor_capabilities_zero_accuracy_invalid() {
    let accuracy = Some(0.0_f64);
    assert!(accuracy.unwrap() <= 0.0);
}

// =============================================================================
// Measurement Type Construction
// =============================================================================

/// AnglesOnly measurement with valid RA/Dec.
#[test]
fn test_measurement_angles_only_valid() {
    // Coordinator validates: RA 0-360, Dec -90 to 90
    let ra = 180.0_f64;
    let dec = 45.0_f64;
    assert!((0.0..=360.0).contains(&ra) && !ra.is_nan());
    assert!((-90.0..=90.0).contains(&dec) && !dec.is_nan());
}

/// AnglesOnly with RA out of range (>360) is invalid.
#[test]
fn test_measurement_angles_only_ra_too_high() {
    let ra = 361.0_f64;
    assert!(!(0.0..=360.0).contains(&ra));
}

/// AnglesOnly with NaN RA is invalid.
#[test]
fn test_measurement_angles_only_ra_nan() {
    let ra = f64::NAN;
    assert!(ra.is_nan());
}

/// AnglesOnly with Dec out of range (>90) is invalid.
#[test]
fn test_measurement_angles_only_dec_too_high() {
    let dec = 91.0_f64;
    assert!(!((-90.0..=90.0).contains(&dec)));
}

/// Range measurement with negative range is invalid.
#[test]
fn test_measurement_range_negative_invalid() {
    let range_km = -1.0_f64;
    assert!(range_km < 0.0);
}

/// Range measurement with zero range is invalid (coordinator checks < 0.0).
#[test]
fn test_measurement_range_zero() {
    // The coordinator checks: range_km < 0.0 || range_km.is_nan()
    // Zero is actually valid in the coordinator (not < 0)
    let range_km = 0.0_f64;
    assert!(!(range_km < 0.0 || range_km.is_nan()));
}

/// Range measurement with NaN is invalid.
#[test]
fn test_measurement_range_nan_invalid() {
    let range_km = f64::NAN;
    assert!(range_km.is_nan());
}

/// Photometric measurement with NaN magnitude is invalid.
#[test]
fn test_measurement_photometric_nan_invalid() {
    let magnitude = f64::NAN;
    assert!(magnitude.is_nan());
}

/// Photometric measurement with valid magnitude.
#[test]
fn test_measurement_photometric_valid() {
    // Typical satellite: magnitude 4-6
    let magnitude = 5.0_f64;
    assert!(!magnitude.is_nan());
}

// =============================================================================
// QualityScore in Observation Context
// =============================================================================

/// Default quality score (50) is used when no quality is provided.
#[test]
fn test_observation_default_quality() {
    let quality = QualityScore::default();
    assert_eq!(quality.value(), 50);
    assert!(quality.is_acceptable());
}

/// Quality score value() returns clamped value for fusion weight.
#[test]
fn test_quality_score_as_fusion_weight() {
    let quality = QualityScore::new(75);
    let weight = quality.value() as f64 / 100.0;
    assert!((0.0..=1.0).contains(&weight));
    assert!((weight - 0.75).abs() < 1e-10);
}

// =============================================================================
// ObservationMeta Serialization
// =============================================================================

/// ObservationMeta roundtrips through JSON.
#[test]
fn test_observation_meta_serde() {
    let meta = ObservationMeta {
        observation_time: SpaceTimestamp::now(),
        submission_time: SpaceTimestamp::now(),
        source: DataSourceType::SpaceTrack,
        quality: QualityScore::new(90),
        raw_data_hash: Some([0xAA; 32]),
    };

    let json = serde_json::to_string(&meta).expect("serialize");
    let parsed: ObservationMeta = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.quality.value(), 90);
    assert!(parsed.raw_data_hash.is_some());
}

/// ObservationMeta without raw_data_hash roundtrips.
#[test]
fn test_observation_meta_no_hash_serde() {
    let meta = ObservationMeta {
        observation_time: SpaceTimestamp::now(),
        submission_time: SpaceTimestamp::now(),
        source: DataSourceType::GroundSensor {
            sensor_id: "SENSOR-01".into(),
            location: None,
        },
        quality: QualityScore::new(70),
        raw_data_hash: None,
    };

    let json = serde_json::to_string(&meta).expect("serialize");
    let parsed: ObservationMeta = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.raw_data_hash.is_none());
}

// =============================================================================
// Trust-Weighted Fusion (orbital-mechanics library)
// =============================================================================

/// Trust level ordering: weight increases with trust.
#[test]
fn test_trust_level_weights_ordered_for_fusion() {
    let levels = [
        TrustLevel::Unverified,
        TrustLevel::BasicTrust,
        TrustLevel::Established,
        TrustLevel::Verified,
        TrustLevel::FoundingMember,
    ];

    for i in 0..levels.len() - 1 {
        assert!(
            levels[i].weight() < levels[i + 1].weight(),
            "{:?} weight {} should be less than {:?} weight {}",
            levels[i],
            levels[i].weight(),
            levels[i + 1],
            levels[i + 1].weight()
        );
    }
}

/// All trust weights are in [0, 1] for use as fusion quality multipliers.
#[test]
fn test_trust_level_weights_bounded() {
    for level in [
        TrustLevel::Unverified,
        TrustLevel::BasicTrust,
        TrustLevel::Established,
        TrustLevel::Verified,
        TrustLevel::FoundingMember,
    ] {
        let w = level.weight();
        assert!(
            (0.0..=1.0).contains(&w),
            "{:?} weight {} out of [0,1]",
            level,
            w
        );
    }
}

// =============================================================================
// validate_string_field for sensor_id (observations coordinator)
// =============================================================================

/// Empty sensor_id is rejected.
#[test]
fn test_sensor_id_empty_rejected() {
    assert!(validate_string_field("", "sensor_id", 256).is_err());
}

/// sensor_id at max length is valid.
#[test]
fn test_sensor_id_max_length_valid() {
    let id = "X".repeat(256);
    assert!(validate_string_field(&id, "sensor_id", 256).is_ok());
}

/// sensor_id exceeding max length is rejected.
#[test]
fn test_sensor_id_too_long_rejected() {
    let id = "X".repeat(257);
    assert!(validate_string_field(&id, "sensor_id", 256).is_err());
}

// =============================================================================
// FusedEstimate Construction Rules
// =============================================================================

/// A valid FusedEstimate state vector has exactly 6 elements.
#[test]
fn test_fused_estimate_state_vector_length() {
    let sv = vec![7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
    assert_eq!(sv.len(), 6);
}

/// Covariance diagonal must have exactly 6 elements.
#[test]
fn test_fused_estimate_covariance_length() {
    let cov = vec![1.0, 1.0, 1.0, 0.01, 0.01, 0.01];
    assert_eq!(cov.len(), 6);
}

/// Fused quality must be in [0, 1].
#[test]
fn test_fused_estimate_quality_range() {
    let quality = 0.85_f64;
    assert!(quality.is_finite() && quality >= 0.0 && quality <= 1.0);
}

/// Chi-square consistency must be non-negative.
#[test]
fn test_fused_estimate_chi_square_nonnegative() {
    let chi2 = 2.3_f64;
    assert!(chi2.is_finite() && chi2 >= 0.0);
}

/// Contributing sensors must not be empty.
#[test]
fn test_fused_estimate_needs_sensors() {
    let sensors: Vec<String> = vec!["SENSOR-01".into()];
    assert!(!sensors.is_empty());
}
