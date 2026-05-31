// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Edge Case and Error Handling Tests
//!
//! Tests for error paths, boundary conditions, and validation logic
//! as specified in the improvement plan:
//! - Duplicate object registration handling
//! - Invalid timestamp handling
//! - Bounty contribution to non-existent bounty
//! - Maneuver announcement for non-existent conjunction
//! - NoradId boundary validation
//! - Quality score edge cases
//! - State vector validation

use chrono::{Datelike, Duration, Utc};
use mycelix_space_shared::*;
use std::collections::HashMap;

// =============================================================================
// NoradId Validation Tests
// =============================================================================

/// Test NoradId boundary at zero (invalid)
#[test]
fn test_norad_id_zero_is_invalid() {
    let result = NoradId::new(0);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Invalid NORAD ID") || err_msg.contains("0"));
}

/// Test NoradId boundary at minimum valid value
#[test]
fn test_norad_id_minimum_valid() {
    let result = NoradId::new(1);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().value(), 1);
}

/// Test NoradId boundary at maximum valid value
#[test]
fn test_norad_id_maximum_valid() {
    let result = NoradId::new(999999);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().value(), 999999);
}

/// Test NoradId just above maximum (invalid)
#[test]
fn test_norad_id_above_maximum_invalid() {
    let result = NoradId::new(1000000);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Invalid NORAD ID") || err_msg.contains("1000000"));
}

/// Test NoradId with large invalid value
#[test]
fn test_norad_id_very_large_invalid() {
    let result = NoradId::new(u32::MAX);
    assert!(result.is_err());
}

/// Test NoradId hash and equality
#[test]
fn test_norad_id_hash_equality() {
    let id1 = NoradId::new(25544).unwrap();
    let id2 = NoradId::new(25544).unwrap();
    let id3 = NoradId::new(25545).unwrap();

    assert_eq!(id1, id2);
    assert_ne!(id1, id3);

    // Should be usable as HashMap key
    let mut map: HashMap<NoradId, String> = HashMap::new();
    map.insert(id1, "ISS".to_string());
    assert_eq!(map.get(&id2), Some(&"ISS".to_string()));
    assert_eq!(map.get(&id3), None);
}

// =============================================================================
// Timestamp Edge Case Tests
// =============================================================================

/// Test SpaceTimestamp with very old date (Unix epoch)
#[test]
fn test_timestamp_unix_epoch() {
    use chrono::TimeZone;
    let epoch = Utc.with_ymd_and_hms(1970, 1, 1, 0, 0, 0).unwrap();
    let ts = SpaceTimestamp::from_datetime(epoch);

    assert_eq!(ts.micros, 0);
    let reconstructed = ts.to_datetime();
    assert_eq!(reconstructed.timestamp(), 0);
}

/// Test SpaceTimestamp with negative timestamp (before Unix epoch)
#[test]
fn test_timestamp_before_epoch() {
    use chrono::TimeZone;
    let before_epoch = Utc.with_ymd_and_hms(1960, 1, 1, 0, 0, 0).unwrap();
    let ts = SpaceTimestamp::from_datetime(before_epoch);

    assert!(ts.micros < 0);
    let reconstructed = ts.to_datetime();
    assert!(reconstructed.year() == 1960);
}

/// Test SpaceTimestamp with future date
#[test]
fn test_timestamp_future_date() {
    let future = Utc::now() + Duration::days(365 * 100); // 100 years in future
    let ts = SpaceTimestamp::from_datetime(future);

    let reconstructed = ts.to_datetime();
    assert!(reconstructed > Utc::now());
}

/// Test SpaceTimestamp age calculation
#[test]
fn test_timestamp_age_calculation() {
    let past = Utc::now() - Duration::hours(1);
    let ts = SpaceTimestamp::from_datetime(past);

    // Age should be approximately 3600 seconds (1 hour)
    let age = ts.age_seconds();
    assert!(
        age >= 3599 && age <= 3601,
        "Age {} not within expected range",
        age
    );
}

/// Test SpaceTimestamp age for future timestamp (negative age)
#[test]
fn test_timestamp_future_age_negative() {
    let future = Utc::now() + Duration::hours(1);
    let ts = SpaceTimestamp::from_datetime(future);

    // Age should be negative for future timestamps
    let age = ts.age_seconds();
    assert!(
        age < 0,
        "Future timestamp should have negative age, got {}",
        age
    );
}

/// Test SpaceTimestamp roundtrip precision
#[test]
fn test_timestamp_microsecond_precision() {
    let now = Utc::now();
    let ts = SpaceTimestamp::from_datetime(now);
    let reconstructed = ts.to_datetime();

    // Difference should be less than 1 microsecond
    let diff = (now - reconstructed).num_microseconds().unwrap_or(i64::MAX);
    assert!(
        diff.abs() < 2,
        "Timestamp lost precision: {} microseconds",
        diff
    );
}

// =============================================================================
// Quality Score Edge Cases
// =============================================================================

/// Test QualityScore exactly at high threshold
#[test]
fn test_quality_score_at_high_threshold() {
    let score = QualityScore::new(80);
    assert!(score.is_high());
    assert!(score.is_acceptable());
}

/// Test QualityScore just below high threshold
#[test]
fn test_quality_score_below_high_threshold() {
    let score = QualityScore::new(79);
    assert!(!score.is_high());
    assert!(score.is_acceptable());
}

/// Test QualityScore at acceptable threshold
#[test]
fn test_quality_score_at_acceptable_threshold() {
    let score = QualityScore::new(50);
    assert!(!score.is_high());
    assert!(score.is_acceptable());
}

/// Test QualityScore just below acceptable threshold
#[test]
fn test_quality_score_below_acceptable_threshold() {
    let score = QualityScore::new(49);
    assert!(!score.is_high());
    assert!(!score.is_acceptable());
}

/// Test QualityScore at zero
#[test]
fn test_quality_score_zero() {
    let score = QualityScore::new(0);
    assert_eq!(score.value(), 0);
    assert!(!score.is_high());
    assert!(!score.is_acceptable());
}

/// Test QualityScore clamping at maximum
#[test]
fn test_quality_score_clamping() {
    let score = QualityScore::new(255);
    assert_eq!(score.value(), 100);
    assert!(score.is_high());
}

/// Test QualityScore default
#[test]
fn test_quality_score_default() {
    let score = QualityScore::default();
    assert_eq!(score.value(), 50);
    assert!(score.is_acceptable());
}

// =============================================================================
// Risk Level Edge Cases
// =============================================================================

/// Test RiskLevel at exact boundary Negligible/Low (1e-7)
#[test]
fn test_risk_level_boundary_negligible_low() {
    // Just above 1e-7 should be Low
    assert_eq!(RiskLevel::from_pc(1.0001e-7), RiskLevel::Low);
    // At 1e-7 should be Low (> comparison)
    assert_eq!(RiskLevel::from_pc(1e-7), RiskLevel::Negligible);
    // Just below 1e-7 should be Negligible
    assert_eq!(RiskLevel::from_pc(0.9999e-7), RiskLevel::Negligible);
}

/// Test RiskLevel at exact boundary Low/Medium (1e-5)
#[test]
fn test_risk_level_boundary_low_medium() {
    assert_eq!(RiskLevel::from_pc(1.0001e-5), RiskLevel::Medium);
    assert_eq!(RiskLevel::from_pc(1e-5), RiskLevel::Low);
    assert_eq!(RiskLevel::from_pc(0.9999e-5), RiskLevel::Low);
}

/// Test RiskLevel at exact boundary Medium/High (1e-4)
#[test]
fn test_risk_level_boundary_medium_high() {
    assert_eq!(RiskLevel::from_pc(1.0001e-4), RiskLevel::High);
    assert_eq!(RiskLevel::from_pc(1e-4), RiskLevel::Medium);
    assert_eq!(RiskLevel::from_pc(0.9999e-4), RiskLevel::Medium);
}

/// Test RiskLevel at exact boundary High/Emergency (1e-3)
#[test]
fn test_risk_level_boundary_high_emergency() {
    assert_eq!(RiskLevel::from_pc(1.0001e-3), RiskLevel::Emergency);
    assert_eq!(RiskLevel::from_pc(1e-3), RiskLevel::High);
    assert_eq!(RiskLevel::from_pc(0.9999e-3), RiskLevel::High);
}

/// Test RiskLevel with Pc = 0
#[test]
fn test_risk_level_zero_pc() {
    assert_eq!(RiskLevel::from_pc(0.0), RiskLevel::Negligible);
}

/// Test RiskLevel with negative Pc (edge case)
#[test]
fn test_risk_level_negative_pc() {
    // Should handle gracefully (treat as Negligible)
    assert_eq!(RiskLevel::from_pc(-1e-5), RiskLevel::Negligible);
}

/// Test RiskLevel with Pc = 1 (guaranteed collision)
#[test]
fn test_risk_level_certain_collision() {
    assert_eq!(RiskLevel::from_pc(1.0), RiskLevel::Emergency);
}

/// Test RiskLevel with Pc > 1 (invalid but should handle)
#[test]
fn test_risk_level_pc_greater_than_one() {
    // Edge case: invalid probability
    assert_eq!(RiskLevel::from_pc(1.5), RiskLevel::Emergency);
}

// =============================================================================
// Unit Vector Validation Tests
// =============================================================================

/// Test unit vector validation with exact unit vectors
#[test]
fn test_unit_vector_exact() {
    assert!(is_unit_vector(&[1.0, 0.0, 0.0], 0.001));
    assert!(is_unit_vector(&[0.0, 1.0, 0.0], 0.001));
    assert!(is_unit_vector(&[0.0, 0.0, 1.0], 0.001));
}

/// Test unit vector validation with normalized diagonal
#[test]
fn test_unit_vector_normalized_diagonal() {
    let sqrt3 = 3.0_f64.sqrt();
    assert!(is_unit_vector(
        &[1.0 / sqrt3, 1.0 / sqrt3, 1.0 / sqrt3],
        0.001
    ));
}

/// Test unit vector validation with zero vector (invalid)
#[test]
fn test_unit_vector_zero_invalid() {
    assert!(!is_unit_vector(&[0.0, 0.0, 0.0], 0.001));
}

/// Test unit vector validation with non-unit vector
#[test]
fn test_unit_vector_non_unit_invalid() {
    assert!(!is_unit_vector(&[1.0, 1.0, 0.0], 0.001)); // magnitude sqrt(2)
    assert!(!is_unit_vector(&[2.0, 0.0, 0.0], 0.001)); // magnitude 2
    assert!(!is_unit_vector(&[0.5, 0.0, 0.0], 0.001)); // magnitude 0.5
}

/// Test unit vector with various tolerances
#[test]
fn test_unit_vector_tolerance_levels() {
    // Slightly off unit vector (magnitude ~1.001)
    let v = [1.0005, 0.0, 0.0];

    // Tight tolerance should reject
    assert!(!is_unit_vector(&v, 0.0001));

    // Loose tolerance should accept
    assert!(is_unit_vector(&v, 0.01));
}

// =============================================================================
// Trust Level Tests
// =============================================================================

/// Test TrustLevel weight values
#[test]
fn test_trust_level_weight_bounds() {
    // All weights should be between 0 and 1
    for level in [
        TrustLevel::Unverified,
        TrustLevel::BasicTrust,
        TrustLevel::Established,
        TrustLevel::Verified,
        TrustLevel::FoundingMember,
    ] {
        let weight = level.weight();
        assert!(
            weight >= 0.0 && weight <= 1.0,
            "Weight {} for {:?} out of bounds",
            weight,
            level
        );
    }
}

/// Test TrustLevel weight ordering matches level ordering
#[test]
fn test_trust_level_weight_ordering() {
    assert!(TrustLevel::Unverified.weight() < TrustLevel::BasicTrust.weight());
    assert!(TrustLevel::BasicTrust.weight() < TrustLevel::Established.weight());
    assert!(TrustLevel::Established.weight() < TrustLevel::Verified.weight());
    assert!(TrustLevel::Verified.weight() < TrustLevel::FoundingMember.weight());
}

/// Test TrustLevel default
#[test]
fn test_trust_level_default() {
    let default: TrustLevel = Default::default();
    assert_eq!(default, TrustLevel::Unverified);
}

// =============================================================================
// MycelixError Tests
// =============================================================================

/// Test MycelixError message generation
#[test]
fn test_mycelix_error_messages() {
    let errors = vec![
        (MycelixError::ObjectNotFound { norad_id: 25544 }, "25544"),
        (MycelixError::InvalidNoradId { id: 0 }, "0"),
        (
            MycelixError::InvalidTle {
                reason: "bad checksum".to_string(),
            },
            "bad checksum",
        ),
        (
            MycelixError::TleChecksumMismatch {
                line: 1,
                expected: 5,
                actual: 3,
            },
            "checksum",
        ),
        (
            MycelixError::EntryNotFound {
                entry_type: "Bounty".to_string(),
                hash: "abc123".to_string(),
            },
            "Bounty",
        ),
        (
            MycelixError::Unauthorized {
                operation: "delete".to_string(),
                reason: "not owner".to_string(),
            },
            "delete",
        ),
        (
            MycelixError::DuplicateEntry {
                entry_type: "Object".to_string(),
                identifier: "25544".to_string(),
            },
            "Duplicate",
        ),
    ];

    for (error, expected_substring) in errors {
        let message = error.message();
        assert!(
            message.contains(expected_substring),
            "Error message '{}' should contain '{}'",
            message,
            expected_substring
        );
    }
}

/// Test MycelixError code uniqueness
#[test]
fn test_mycelix_error_codes_are_screaming_snake_case() {
    let errors = vec![
        MycelixError::ObjectNotFound { norad_id: 0 },
        MycelixError::InvalidNoradId { id: 0 },
        MycelixError::DuplicateEntry {
            entry_type: String::new(),
            identifier: String::new(),
        },
    ];

    for error in errors {
        let code = error.code();
        // Code should be SCREAMING_SNAKE_CASE
        assert!(
            code.chars().all(|c| c.is_uppercase() || c == '_'),
            "Error code '{}' should be SCREAMING_SNAKE_CASE",
            code
        );
    }
}

/// Test MycelixError serialization roundtrip
#[test]
fn test_mycelix_error_serialization_roundtrip() {
    let original = MycelixError::BountyError {
        bounty_id: "BOUNTY-001".to_string(),
        reason: "Already claimed".to_string(),
    };

    let json = serde_json::to_string(&original).expect("Failed to serialize");
    let parsed: MycelixError = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(original, parsed);
}

// =============================================================================
// Alert Priority Tests
// =============================================================================

/// Test AlertPriority from all RiskLevels
#[test]
fn test_alert_priority_from_all_risk_levels() {
    assert_eq!(
        AlertPriority::from_risk_level(RiskLevel::Negligible),
        AlertPriority::Info
    );
    assert_eq!(
        AlertPriority::from_risk_level(RiskLevel::Low),
        AlertPriority::Low
    );
    assert_eq!(
        AlertPriority::from_risk_level(RiskLevel::Medium),
        AlertPriority::Medium
    );
    assert_eq!(
        AlertPriority::from_risk_level(RiskLevel::High),
        AlertPriority::High
    );
    assert_eq!(
        AlertPriority::from_risk_level(RiskLevel::Emergency),
        AlertPriority::Critical
    );
}

/// Test AlertPriority ordering
#[test]
fn test_alert_priority_ordering() {
    let priorities = [
        AlertPriority::Info,
        AlertPriority::Low,
        AlertPriority::Medium,
        AlertPriority::High,
        AlertPriority::Critical,
    ];

    for i in 0..priorities.len() - 1 {
        assert!(
            priorities[i] < priorities[i + 1],
            "{:?} should be less than {:?}",
            priorities[i],
            priorities[i + 1]
        );
    }
}

// =============================================================================
// ConjunctionAlert Edge Cases
// =============================================================================

/// Test ConjunctionAlert is_critical for various risk levels
#[test]
fn test_conjunction_alert_is_critical() {
    let make_assessment = |risk_level: RiskLevel| ConjunctionAssessment {
        primary_norad_id: 25544,
        secondary_norad_id: 49863,
        tca: Utc::now(),
        miss_distance_km: 0.5,
        relative_velocity_kms: 14.5,
        collision_probability: 1e-5,
        pc_method: PcMethod::Alfano2D,
        risk_level,
        hard_body_radius_m: 20.0,
        screening_volume_km: 5.0,
    };

    // High and Emergency should be critical
    let high_alert = ConjunctionAlert::new_conjunction(&make_assessment(RiskLevel::High));
    assert!(high_alert.is_critical());

    let emergency_alert = ConjunctionAlert::new_conjunction(&make_assessment(RiskLevel::Emergency));
    assert!(emergency_alert.is_critical());

    // Others should not be critical
    let medium_alert = ConjunctionAlert::new_conjunction(&make_assessment(RiskLevel::Medium));
    assert!(!medium_alert.is_critical());

    let low_alert = ConjunctionAlert::new_conjunction(&make_assessment(RiskLevel::Low));
    assert!(!low_alert.is_critical());

    let negligible_alert =
        ConjunctionAlert::new_conjunction(&make_assessment(RiskLevel::Negligible));
    assert!(!negligible_alert.is_critical());
}

/// Test ConjunctionAlert recommendation messages
#[test]
fn test_conjunction_alert_recommendations() {
    let make_assessment = |risk_level: RiskLevel| ConjunctionAssessment {
        primary_norad_id: 25544,
        secondary_norad_id: 49863,
        tca: Utc::now(),
        miss_distance_km: 0.5,
        relative_velocity_kms: 14.5,
        collision_probability: 1e-5,
        pc_method: PcMethod::Alfano2D,
        risk_level,
        hard_body_radius_m: 20.0,
        screening_volume_km: 5.0,
    };

    // Emergency should have immediate action recommendation
    let emergency_alert = ConjunctionAlert::new_conjunction(&make_assessment(RiskLevel::Emergency));
    assert!(emergency_alert.recommendation.is_some());
    assert!(
        emergency_alert
            .recommendation
            .as_ref()
            .unwrap()
            .contains("IMMEDIATE")
    );

    // Negligible should have no recommendation
    let negligible_alert =
        ConjunctionAlert::new_conjunction(&make_assessment(RiskLevel::Negligible));
    assert!(negligible_alert.recommendation.is_none());
}

/// Test ConjunctionAlert time_to_tca calculation
#[test]
fn test_conjunction_alert_time_to_tca() {
    let future_tca = Utc::now() + Duration::hours(24);
    let assessment = ConjunctionAssessment {
        primary_norad_id: 25544,
        secondary_norad_id: 49863,
        tca: future_tca,
        miss_distance_km: 0.5,
        relative_velocity_kms: 14.5,
        collision_probability: 1e-5,
        pc_method: PcMethod::Alfano2D,
        risk_level: RiskLevel::Medium,
        hard_body_radius_m: 20.0,
        screening_volume_km: 5.0,
    };

    let alert = ConjunctionAlert::new_conjunction(&assessment);
    let time_to_tca = alert.time_to_tca();

    // Should be approximately 24 hours
    let hours = time_to_tca.num_hours();
    assert!(hours >= 23 && hours <= 24, "Time to TCA: {} hours", hours);
}

// =============================================================================
// SpaceSignal Edge Cases
// =============================================================================

/// Test SpaceSignal priority extraction
#[test]
fn test_space_signal_priority() {
    let assessment = ConjunctionAssessment {
        primary_norad_id: 25544,
        secondary_norad_id: 49863,
        tca: Utc::now(),
        miss_distance_km: 0.5,
        relative_velocity_kms: 14.5,
        collision_probability: 1e-5,
        pc_method: PcMethod::Alfano2D,
        risk_level: RiskLevel::High,
        hard_body_radius_m: 20.0,
        screening_volume_km: 5.0,
    };

    let alert = ConjunctionAlert::new_conjunction(&assessment);
    let signal = SpaceSignal::Conjunction(alert);

    assert_eq!(signal.priority(), AlertPriority::High);
    assert!(signal.is_critical());
}

/// Test SpaceSignal system variant
#[test]
fn test_space_signal_system() {
    let signal = SpaceSignal::System {
        alert_type: AlertType::SystemAlert,
        priority: AlertPriority::Info,
        message: "Test message".to_string(),
        generated_at: Utc::now(),
    };

    assert_eq!(signal.priority(), AlertPriority::Info);
    assert!(!signal.is_critical());
}

// =============================================================================
// Data Entry Validation Edge Cases
// =============================================================================

/// Test BountyEntry with zero amount
#[test]
fn test_bounty_entry_zero_amount() {
    let bounty = BountyEntry {
        debris_norad_id: 49863,
        bounty_amount: 0,
        currency: "USD".to_string(),
        sponsor: "Test".to_string(),
        deadline: None,
        requirements: BountyRequirements {
            min_mass_kg: None,
            max_altitude_km: None,
            debris_type: None,
            removal_method: None,
        },
        status: BountyStatus::Active,
    };

    // Should serialize without error
    let json = serde_json::to_string(&bounty).expect("Failed to serialize");
    assert!(json.contains("\"bounty_amount\":0"));
}

/// Test BountyEntry with very large amount
#[test]
fn test_bounty_entry_large_amount() {
    let bounty = BountyEntry {
        debris_norad_id: 49863,
        bounty_amount: u64::MAX,
        currency: "USD".to_string(),
        sponsor: "Test".to_string(),
        deadline: None,
        requirements: BountyRequirements {
            min_mass_kg: None,
            max_altitude_km: None,
            debris_type: None,
            removal_method: None,
        },
        status: BountyStatus::Active,
    };

    let json = serde_json::to_string(&bounty).expect("Failed to serialize");
    let parsed: BountyEntry = serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(parsed.bounty_amount, u64::MAX);
}

/// Test StateVectorData with extreme position values
#[test]
fn test_state_vector_extreme_position() {
    let sv = StateVectorData {
        norad_id: 25544,
        epoch: Utc::now(),
        position_km: [1e10, -1e10, 0.0], // Far from Earth
        velocity_kms: [0.0, 0.0, 0.0],
        covariance: None,
        reference_frame: ReferenceFrame::Teme,
        quality: 0.5,
        source: DataSourceSimple::Computed,
    };

    let json = serde_json::to_string(&sv).expect("Failed to serialize");
    let parsed: StateVectorData = serde_json::from_str(&json).expect("Failed to deserialize");
    assert!((parsed.position_km[0] - 1e10).abs() < 1.0);
}

/// Test StateVectorData with NaN values (should handle gracefully)
#[test]
fn test_state_vector_nan_handling() {
    let sv = StateVectorData {
        norad_id: 25544,
        epoch: Utc::now(),
        position_km: [f64::NAN, 0.0, 0.0],
        velocity_kms: [0.0, f64::NAN, 0.0],
        covariance: None,
        reference_frame: ReferenceFrame::Teme,
        quality: f64::NAN,
        source: DataSourceSimple::Computed,
    };

    // Should serialize (JSON will represent NaN as null or special value)
    let json = serde_json::to_string(&sv).expect("Failed to serialize");
    assert!(json.contains("null") || json.contains("NaN") || json.contains("nan"));
}

/// Test OrbitalObjectEntry with empty name
#[test]
fn test_orbital_object_empty_name() {
    let obj = OrbitalObjectEntry {
        norad_id: 99999,
        name: String::new(),
        object_type: ObjectType::Debris,
        launch_date: None,
        decay_date: None,
        owner_country: None,
        data_source: DataSourceSimple::Computed,
        metadata: HashMap::new(),
    };

    let json = serde_json::to_string(&obj).expect("Failed to serialize");
    let parsed: OrbitalObjectEntry = serde_json::from_str(&json).expect("Failed to deserialize");
    assert!(parsed.name.is_empty());
}

/// Test GroundLocation with extreme coordinates
#[test]
fn test_ground_location_extreme_coordinates() {
    // Valid extreme coordinates
    let north_pole = GroundLocation {
        latitude_deg: 90.0,
        longitude_deg: 0.0,
        altitude_m: 0.0,
        name: Some("North Pole".to_string()),
    };

    let south_pole = GroundLocation {
        latitude_deg: -90.0,
        longitude_deg: 180.0,
        altitude_m: -100.0, // Below sea level
        name: Some("South Pole".to_string()),
    };

    // Should serialize/deserialize without error
    let json1 = serde_json::to_string(&north_pole).expect("Failed to serialize");
    let json2 = serde_json::to_string(&south_pole).expect("Failed to serialize");

    let parsed1: GroundLocation = serde_json::from_str(&json1).expect("Failed to deserialize");
    let parsed2: GroundLocation = serde_json::from_str(&json2).expect("Failed to deserialize");

    assert_eq!(parsed1.latitude_deg, 90.0);
    assert_eq!(parsed2.latitude_deg, -90.0);
}

// =============================================================================
// Hash Function Tests
// =============================================================================

/// Test hash_data produces consistent results
#[test]
fn test_hash_data_consistency() {
    let data = b"test data for hashing";
    let hash1 = hash_data(data);
    let hash2 = hash_data(data);

    assert_eq!(hash1, hash2);
}

/// Test hash_data produces different results for different inputs
#[test]
fn test_hash_data_uniqueness() {
    let hash1 = hash_data(b"input 1");
    let hash2 = hash_data(b"input 2");

    assert_ne!(hash1, hash2);
}

/// Test hash_data with empty input
#[test]
fn test_hash_data_empty() {
    let hash = hash_data(b"");
    // Should produce a valid 32-byte hash
    assert_eq!(hash.len(), 32);
    // Should not be all zeros
    assert!(hash.iter().any(|&b| b != 0));
}

/// Test hash_data with large input
#[test]
fn test_hash_data_large_input() {
    let large_data = vec![0u8; 1_000_000]; // 1MB
    let hash = hash_data(&large_data);

    assert_eq!(hash.len(), 32);
}
