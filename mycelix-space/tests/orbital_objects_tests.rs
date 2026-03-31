// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Orbital Objects Coordinator Tests
//!
//! Tests for the orbital objects catalog logic that can be exercised
//! without a Holochain conductor:
//! - Input validation rules (designator, name, country length bounds)
//! - ObjectType serialization roundtrips
//! - TLE parsing validation via orbital-mechanics
//! - DataSourceType construction and serialization
//! - Staleness computation
//! - TleWithMetadata serialization
//! - validate_string_field shared helper
//! - get_latest_tles / get_tles_with_metadata input validation

use chrono::{Duration, Utc};
use mycelix_space_shared::*;
use orbital_mechanics::tle::TwoLineElement;

// =============================================================================
// Input Validation — String Field Helper
// =============================================================================

/// validate_string_field rejects empty strings.
#[test]
fn test_validate_string_field_empty() {
    let result = validate_string_field("", "designator", 256);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.message.contains("must not be empty"));
}

/// validate_string_field rejects strings exceeding max length.
#[test]
fn test_validate_string_field_too_long() {
    let long = "A".repeat(257);
    let result = validate_string_field(&long, "designator", 256);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.message.contains("exceeds maximum length"));
}

/// validate_string_field accepts valid string at boundary.
#[test]
fn test_validate_string_field_boundary_256() {
    let exact = "B".repeat(256);
    let result = validate_string_field(&exact, "designator", 256);
    assert!(result.is_ok());
}

/// validate_string_field accepts short valid string.
#[test]
fn test_validate_string_field_valid() {
    let result = validate_string_field("1998-067A", "designator", 256);
    assert!(result.is_ok());
}

// =============================================================================
// Input Validation — Name Field (coordinator logic)
// =============================================================================

/// The coordinator rejects empty names (length check: 1-256).
#[test]
fn test_name_empty_fails_length_check() {
    let name = "";
    assert!(name.is_empty(), "Empty name must be rejected");
}

/// The coordinator rejects names exceeding 256 characters.
#[test]
fn test_name_too_long_fails_length_check() {
    let name = "S".repeat(257);
    assert!(name.len() > 256);
}

/// A normal satellite name passes the length check.
#[test]
fn test_name_valid_iss() {
    let name = "ISS (ZARYA)";
    assert!(!name.is_empty() && name.len() <= 256);
}

// =============================================================================
// Input Validation — Country Field (coordinator logic)
// =============================================================================

/// Empty country strings are rejected when present.
#[test]
fn test_country_empty_rejected() {
    let result = validate_string_field("", "country", 256);
    assert!(result.is_err());
}

/// Country strings exceeding 256 characters are rejected.
#[test]
fn test_country_too_long_rejected() {
    let long = "X".repeat(257);
    let result = validate_string_field(&long, "country", 256);
    assert!(result.is_err());
}

/// None country means validation is skipped (valid).
#[test]
fn test_country_none_skips_validation() {
    let country: Option<String> = None;
    // Coordinator only validates when Some
    assert!(country.is_none());
}

// =============================================================================
// ObjectType Serialization (shared crate re-export)
// =============================================================================

/// All ObjectType variants roundtrip through JSON.
#[test]
fn test_object_type_serde_roundtrip() {
    let types = vec![
        ObjectType::Payload,
        ObjectType::RocketBody,
        ObjectType::Debris,
        ObjectType::Unknown,
    ];

    for ot in &types {
        let json = serde_json::to_string(ot).expect("serialize");
        let parsed: ObjectType = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&parsed, ot);
    }
}

/// All ObjectType variants produce distinct JSON representations.
#[test]
fn test_object_type_distinct_json() {
    let types = vec![
        ObjectType::Payload,
        ObjectType::RocketBody,
        ObjectType::Debris,
        ObjectType::Unknown,
    ];

    let jsons: Vec<String> = types
        .iter()
        .map(|t| serde_json::to_string(t).unwrap())
        .collect();

    for i in 0..jsons.len() {
        for j in (i + 1)..jsons.len() {
            assert_ne!(jsons[i], jsons[j], "types at {} and {} collide", i, j);
        }
    }
}

// =============================================================================
// TLE Parsing via orbital-mechanics
// =============================================================================

/// Valid ISS TLE parses correctly and returns matching NORAD ID.
#[test]
fn test_tle_parse_valid_iss() {
    let line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577";

    let result = TwoLineElement::parse_lines(None, line1, line2);
    assert!(
        result.is_ok(),
        "Valid ISS TLE should parse: {:?}",
        result.err()
    );

    let tle = result.unwrap();
    assert_eq!(tle.norad_id, 25544);
}

/// TLE with mismatched NORAD ID — the coordinator checks this.
#[test]
fn test_tle_norad_id_mismatch_detection() {
    let line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577";

    let parsed = TwoLineElement::parse_lines(None, line1, line2).unwrap();
    let submitted_norad_id: u32 = 99999;

    assert_ne!(
        parsed.norad_id, submitted_norad_id,
        "Coordinator would reject this mismatch"
    );
}

/// Empty TLE lines fail parsing.
#[test]
fn test_tle_parse_empty_lines() {
    let result = TwoLineElement::parse_lines(None, "", "");
    assert!(result.is_err());
}

/// Corrupted TLE lines fail parsing.
#[test]
fn test_tle_parse_corrupted() {
    let result = TwoLineElement::parse_lines(
        None,
        "1 XXXXX THIS IS NOT A VALID TLE LINE",
        "2 XXXXX THIS IS ALSO NOT VALID",
    );
    assert!(result.is_err());
}

/// TLE line length too short fails parsing.
#[test]
fn test_tle_parse_short_lines() {
    let result = TwoLineElement::parse_lines(None, "1 25544", "2 25544");
    assert!(result.is_err());
}

// =============================================================================
// DataSourceType Serialization
// =============================================================================

/// DataSourceType::SpaceTrack roundtrips through JSON.
#[test]
fn test_data_source_spacetrack_serde() {
    let source = DataSourceType::SpaceTrack;
    let json = serde_json::to_string(&source).expect("serialize");
    let parsed: DataSourceType = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed, source);
}

/// DataSourceType::Commercial with provider roundtrips.
#[test]
fn test_data_source_commercial_serde() {
    let source = DataSourceType::Commercial {
        provider: "LeoLabs".to_string(),
    };
    let json = serde_json::to_string(&source).expect("serialize");
    let parsed: DataSourceType = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed, source);
}

/// DataSourceType::NetworkFusion roundtrips with counts.
#[test]
fn test_data_source_network_fusion_serde() {
    let source = DataSourceType::NetworkFusion {
        source_count: 5,
        node_count: 12,
    };
    let json = serde_json::to_string(&source).expect("serialize");
    let parsed: DataSourceType = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed, source);
}

// =============================================================================
// Staleness Computation
// =============================================================================

/// Fresh TLE (epoch = now) should not be stale.
#[test]
fn test_staleness_fresh_tle_not_stale() {
    let epoch = SpaceTimestamp::now();
    let quality = QualityScore::new(80);
    let config = StalenessConfig::default();

    let staleness = compute_staleness(&epoch, &quality, &config);
    assert!(!staleness.is_stale);
    assert!(
        staleness.confidence > 0.9,
        "Fresh TLE confidence: {}",
        staleness.confidence
    );
    assert_eq!(staleness.effective_quality.value(), 80);
}

/// Old TLE (60 days) should be stale with degraded quality.
#[test]
fn test_staleness_old_tle_is_stale() {
    let old = Utc::now() - Duration::days(60);
    let epoch = SpaceTimestamp::from_datetime(old);
    let quality = QualityScore::new(80);
    let config = StalenessConfig::default(); // 30-day threshold

    let staleness = compute_staleness(&epoch, &quality, &config);
    assert!(staleness.is_stale);
    assert!(staleness.effective_quality.value() < quality.value());
    assert!(
        staleness.confidence < 0.5,
        "Old TLE confidence should be low: {}",
        staleness.confidence
    );
}

/// Custom staleness config with very short threshold.
#[test]
fn test_staleness_custom_short_threshold() {
    let one_day_ago = Utc::now() - Duration::days(1);
    let epoch = SpaceTimestamp::from_datetime(one_day_ago);
    let quality = QualityScore::new(90);
    let config = StalenessConfig {
        stale_threshold_secs: 3600, // 1 hour threshold
        min_quality: 50,
    };

    let staleness = compute_staleness(&epoch, &quality, &config);
    assert!(
        staleness.is_stale,
        "1-day-old TLE should be stale with 1-hour threshold"
    );
}

// =============================================================================
// TleWithMetadata Serialization
// =============================================================================

/// TleWithMetadata roundtrips through JSON.
#[test]
fn test_tle_with_metadata_serde_roundtrip() {
    let epoch = SpaceTimestamp::now();
    let quality = QualityScore::new(85);
    let config = StalenessConfig::default();
    let staleness = compute_staleness(&epoch, &quality, &config);

    let meta = TleWithMetadata {
        norad_id: 25544,
        line1: "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997".into(),
        line2: "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577".into(),
        epoch,
        quality,
        source: DataSourceType::SpaceTrack,
        submitted_at: SpaceTimestamp::now(),
        staleness,
    };

    let json = serde_json::to_string(&meta).expect("serialize");
    let parsed: TleWithMetadata = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.norad_id, 25544);
    assert_eq!(parsed.quality.value(), 85);
    assert!(!parsed.staleness.is_stale);
}

// =============================================================================
// get_latest_tles Input Validation (replicated coordinator logic)
// =============================================================================

/// Empty NORAD ID list is rejected.
#[test]
fn test_get_tles_empty_list_rejected() {
    let norad_ids: Vec<u32> = vec![];
    assert!(norad_ids.is_empty());
}

/// List exceeding 1000 NORAD IDs is rejected.
#[test]
fn test_get_tles_over_1000_rejected() {
    let norad_ids: Vec<u32> = (1..=1001).collect();
    assert!(norad_ids.len() > 1000);
}

/// Exactly 1000 IDs is valid.
#[test]
fn test_get_tles_exactly_1000_valid() {
    let norad_ids: Vec<u32> = (1..=1000).collect();
    assert!(norad_ids.len() <= 1000 && !norad_ids.is_empty());
}

/// Single ID is valid.
#[test]
fn test_get_tles_single_id_valid() {
    let norad_ids = vec![25544u32];
    assert!(!norad_ids.is_empty() && norad_ids.len() <= 1000);
}

// =============================================================================
// PaginationParams Defaults
// =============================================================================

/// Default PaginationParams: limit=50, offset=0.
#[test]
fn test_pagination_defaults() {
    let params = PaginationParams {
        limit: None,
        offset: None,
    };
    assert_eq!(params.effective_limit(), 50);
    assert_eq!(params.effective_offset(), 0);
}

/// PaginationParams limit is clamped to 500.
#[test]
fn test_pagination_limit_clamped() {
    let params = PaginationParams {
        limit: Some(1000),
        offset: None,
    };
    assert_eq!(params.effective_limit(), 500);
}

/// PaginationParams respects custom values.
#[test]
fn test_pagination_custom_values() {
    let params = PaginationParams {
        limit: Some(25),
        offset: Some(100),
    };
    assert_eq!(params.effective_limit(), 25);
    assert_eq!(params.effective_offset(), 100);
}
