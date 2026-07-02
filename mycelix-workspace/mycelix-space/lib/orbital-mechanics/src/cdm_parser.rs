// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CCSDS Conjunction Data Message (CDM) KVN Parser
//!
//! Parses Conjunction Data Messages in CCSDS 508.0-B-1 Keyword-Value Notation
//! (KVN) format into [`ConjunctionDataMessage`] structs.
//!
//! # KVN Format
//! ```text
//! CCSDS_CDM_VERS = 1.0
//! CREATION_DATE = 2024-01-15T12:00:00.000
//! ORIGINATOR = MYCELIX-SPACE
//! ...
//! ```
//!
//! Lines are `KEY = VALUE` pairs. Comments start with `COMMENT`.
//! Object data blocks are distinguished by `OBJECT1_` and `OBJECT2_` prefixes.

use chrono::{DateTime, NaiveDateTime, Utc};

use crate::cdm::{
    CdmCovariance, CdmObjectData, CdmObjectMetadata, CdmRefFrame, CdmStateVector,
    ConjunctionDataMessage, Maneuverable, PcMethod,
};

/// Error type for CDM parsing
#[derive(Debug, thiserror::Error)]
pub enum CdmParseError {
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("Invalid value for {field}: {value}")]
    InvalidValue { field: String, value: String },
    #[error("Date parse error for {field}: {value}")]
    DateParseError { field: String, value: String },
    #[error("Unsupported CDM version: {0}")]
    UnsupportedVersion(String),
    #[error("Empty input")]
    EmptyInput,
    #[error("Incomplete covariance for {object}: CR_R present but missing elements: {missing}")]
    IncompleteCovariance { object: String, missing: String },
    #[error("CREATION_DATE ({creation}) is after TCA ({tca}) — CDM is malformed")]
    CreationAfterTca { creation: String, tca: String },
}

/// Parse a CCSDS CDM in KVN (Keyword-Value Notation) format.
///
/// Handles the standard layout:
/// 1. Header (CCSDS_CDM_VERS, CREATION_DATE, ORIGINATOR, MESSAGE_ID)
/// 2. Relative metadata (TCA, MISS_DISTANCE, etc.)
/// 3. Collision probability
/// 4. Object 1 data (metadata + state vector + optional covariance)
/// 5. Object 2 data (metadata + state vector + optional covariance)
pub fn parse_cdm_kvn(input: &str) -> Result<ConjunctionDataMessage, CdmParseError> {
    if input.trim().is_empty() {
        return Err(CdmParseError::EmptyInput);
    }

    let lines: Vec<(&str, &str)> = input
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with("COMMENT") {
                return None;
            }
            let mut parts = line.splitn(2, '=');
            let key = parts.next()?.trim();
            let value = parts.next()?.trim();
            Some((key, value))
        })
        .collect();

    let get = |key: &str| -> Result<String, CdmParseError> {
        lines
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| strip_units(v).to_string())
            .ok_or_else(|| CdmParseError::MissingField(key.to_string()))
    };

    let get_f64 = |key: &str| -> Result<f64, CdmParseError> {
        let val = get(key)?;
        val.parse::<f64>().map_err(|_| CdmParseError::InvalidValue {
            field: key.to_string(),
            value: val,
        })
    };

    let get_opt_f64 =
        |key: &str| -> Option<f64> { get(key).ok().and_then(|v| v.parse::<f64>().ok()) };

    // --- Header ---
    let version = get("CCSDS_CDM_VERS")?;
    if version != "1.0" {
        return Err(CdmParseError::UnsupportedVersion(version));
    }

    let creation_date = parse_cdm_datetime(&get("CREATION_DATE")?, "CREATION_DATE")?;
    let originator = get("ORIGINATOR")?;
    let message_id = get("MESSAGE_ID")?;

    // --- Relative Metadata ---
    let tca = parse_cdm_datetime(&get("TCA")?, "TCA")?;
    let miss_distance = get_f64("MISS_DISTANCE")?;
    let relative_speed = get_f64("RELATIVE_SPEED")?;

    let relative_position_r = get_f64("RELATIVE_POSITION_R").unwrap_or(0.0);
    let relative_position_t = get_f64("RELATIVE_POSITION_T").unwrap_or(0.0);
    let relative_position_n = get_f64("RELATIVE_POSITION_N").unwrap_or(0.0);
    let relative_velocity_r = get_f64("RELATIVE_VELOCITY_R").unwrap_or(0.0);
    let relative_velocity_t = get_f64("RELATIVE_VELOCITY_T").unwrap_or(0.0);
    let relative_velocity_n = get_f64("RELATIVE_VELOCITY_N").unwrap_or(0.0);

    // --- Freshness validation: CREATION_DATE must be <= TCA ---
    if creation_date > tca {
        return Err(CdmParseError::CreationAfterTca {
            creation: creation_date.to_rfc3339(),
            tca: tca.to_rfc3339(),
        });
    }

    // --- Probability ---
    let collision_probability = get_f64("COLLISION_PROBABILITY")?;
    let collision_probability_method_str = get("COLLISION_PROBABILITY_METHOD")?;
    let collision_probability_method =
        PcMethod::from_str_permissive(&collision_probability_method_str);

    // --- Object Data ---
    let object1 = parse_object_block(&lines, "OBJECT1")?;
    let object2 = parse_object_block(&lines, "OBJECT2")?;

    // --- Optional fields ---
    let hard_body_radius = get_opt_f64("HARD_BODY_RADIUS");

    Ok(ConjunctionDataMessage {
        ccsds_cdm_vers: version,
        creation_date,
        originator,
        message_id,
        tca,
        miss_distance,
        relative_speed,
        relative_position_r,
        relative_position_t,
        relative_position_n,
        relative_velocity_r,
        relative_velocity_t,
        relative_velocity_n,
        collision_probability,
        collision_probability_method,
        object1,
        object2,
        screening_entry_time: None,
        screening_data_source: None,
        hard_body_radius,
    })
}

/// Parse an object data block (metadata + state vector + optional covariance).
fn parse_object_block(
    lines: &[(&str, &str)],
    prefix: &str,
) -> Result<CdmObjectData, CdmParseError> {
    let get = |suffix: &str| -> Result<String, CdmParseError> {
        let key = format!("{}_{}", prefix, suffix);
        lines
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| strip_units(v).to_string())
            .ok_or(CdmParseError::MissingField(key))
    };

    let get_f64 = |suffix: &str| -> Result<f64, CdmParseError> {
        let key = format!("{}_{}", prefix, suffix);
        let val = get(suffix)?;
        val.parse::<f64>().map_err(|_| CdmParseError::InvalidValue {
            field: key,
            value: val,
        })
    };

    let get_opt_f64 =
        |suffix: &str| -> Option<f64> { get(suffix).ok().and_then(|v| v.parse::<f64>().ok()) };

    // Metadata
    let metadata = CdmObjectMetadata {
        object_designator: get("OBJECT_DESIGNATOR")?,
        catalog_name: get("CATALOG_NAME").unwrap_or_else(|_| "SATCAT".to_string()),
        object_name: get("OBJECT_NAME")?,
        international_designator: get("INTERNATIONAL_DESIGNATOR")
            .unwrap_or_else(|_| "UNKNOWN".to_string()),
        object_type: get("OBJECT_TYPE").unwrap_or_else(|_| "UNKNOWN".to_string()),
        operator_organization: get("OPERATOR_ORGANIZATION").ok(),
        operator_phone: get("OPERATOR_PHONE").ok(),
        operator_email: get("OPERATOR_EMAIL").ok(),
        ephemeris_name: get("EPHEMERIS_NAME").unwrap_or_else(|_| "NONE".to_string()),
        covariance_method: get("COVARIANCE_METHOD").unwrap_or_else(|_| "CALCULATED".to_string()),
        maneuverable: parse_maneuverable(&get("MANEUVERABLE").unwrap_or_else(|_| "N/A".into())),
        ref_frame: parse_ref_frame(&get("REF_FRAME").unwrap_or_else(|_| "EME2000".into())),
    };

    // State vector
    let state_vector = CdmStateVector {
        x: get_f64("X")?,
        y: get_f64("Y")?,
        z: get_f64("Z")?,
        x_dot: get_f64("X_DOT")?,
        y_dot: get_f64("Y_DOT")?,
        z_dot: get_f64("Z_DOT")?,
    };

    // Covariance (optional — but if CR_R is present, all 21 elements are required
    // per CCSDS 508.0-B-1 Section 5.2.5)
    let covariance = if get_opt_f64("CR_R").is_some() {
        // All 21 lower-triangular covariance elements
        const COV_ELEMENTS: &[&str] = &[
            "CR_R",
            "CT_R",
            "CT_T",
            "CN_R",
            "CN_T",
            "CN_N",
            "CRDOT_R",
            "CRDOT_T",
            "CRDOT_N",
            "CRDOT_RDOT",
            "CTDOT_R",
            "CTDOT_T",
            "CTDOT_N",
            "CTDOT_RDOT",
            "CTDOT_TDOT",
            "CNDOT_R",
            "CNDOT_T",
            "CNDOT_N",
            "CNDOT_RDOT",
            "CNDOT_TDOT",
            "CNDOT_NDOT",
        ];

        let missing: Vec<&str> = COV_ELEMENTS
            .iter()
            .filter(|elem| get_opt_f64(elem).is_none())
            .copied()
            .collect();

        if !missing.is_empty() {
            return Err(CdmParseError::IncompleteCovariance {
                object: prefix.to_string(),
                missing: missing.join(", "),
            });
        }

        Some(CdmCovariance {
            cr_r: get_opt_f64("CR_R").unwrap(),
            ct_r: get_opt_f64("CT_R").unwrap(),
            ct_t: get_opt_f64("CT_T").unwrap(),
            cn_r: get_opt_f64("CN_R").unwrap(),
            cn_t: get_opt_f64("CN_T").unwrap(),
            cn_n: get_opt_f64("CN_N").unwrap(),
            crdot_r: get_opt_f64("CRDOT_R").unwrap(),
            crdot_t: get_opt_f64("CRDOT_T").unwrap(),
            crdot_n: get_opt_f64("CRDOT_N").unwrap(),
            crdot_rdot: get_opt_f64("CRDOT_RDOT").unwrap(),
            ctdot_r: get_opt_f64("CTDOT_R").unwrap(),
            ctdot_t: get_opt_f64("CTDOT_T").unwrap(),
            ctdot_n: get_opt_f64("CTDOT_N").unwrap(),
            ctdot_rdot: get_opt_f64("CTDOT_RDOT").unwrap(),
            ctdot_tdot: get_opt_f64("CTDOT_TDOT").unwrap(),
            cndot_r: get_opt_f64("CNDOT_R").unwrap(),
            cndot_t: get_opt_f64("CNDOT_T").unwrap(),
            cndot_n: get_opt_f64("CNDOT_N").unwrap(),
            cndot_rdot: get_opt_f64("CNDOT_RDOT").unwrap(),
            cndot_tdot: get_opt_f64("CNDOT_TDOT").unwrap(),
            cndot_ndot: get_opt_f64("CNDOT_NDOT").unwrap(),
        })
    } else {
        None
    };

    Ok(CdmObjectData {
        metadata,
        state_vector,
        covariance,
    })
}

/// Parse a CDM datetime string.
/// Supports ISO 8601 format: `2024-01-15T12:00:00.000`
fn parse_cdm_datetime(s: &str, field: &str) -> Result<DateTime<Utc>, CdmParseError> {
    // Try full ISO 8601 with fractional seconds
    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        return Ok(ndt.and_utc());
    }
    // Try without fractional seconds
    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Ok(ndt.and_utc());
    }
    // Try DOY format: YYYY-DDDTHH:MM:SS.SSS
    if s.len() >= 17 && s.chars().nth(4) == Some('-') && s.chars().nth(8) == Some('T') {
        if let (Ok(year), Ok(doy)) = (s[..4].parse::<i32>(), s[5..8].parse::<u32>()) {
            let time_part = &s[9..];
            let base = chrono::NaiveDate::from_yo_opt(year, doy).and_then(|d| {
                chrono::NaiveTime::parse_from_str(time_part, "%H:%M:%S%.f")
                    .ok()
                    .or_else(|| chrono::NaiveTime::parse_from_str(time_part, "%H:%M:%S").ok())
                    .map(|t| d.and_time(t))
            });
            if let Some(ndt) = base {
                return Ok(ndt.and_utc());
            }
        }
    }
    Err(CdmParseError::DateParseError {
        field: field.to_string(),
        value: s.to_string(),
    })
}

/// Strip unit annotations from KVN values, e.g. `"1.234 [km]"` → `"1.234"`
fn strip_units(value: &str) -> &str {
    if let Some(pos) = value.find('[') {
        value[..pos].trim()
    } else {
        value.trim()
    }
}

fn parse_maneuverable(s: &str) -> Maneuverable {
    match s.to_uppercase().as_str() {
        "YES" => Maneuverable::Yes,
        "NO" => Maneuverable::No,
        _ => Maneuverable::Unknown,
    }
}

fn parse_ref_frame(s: &str) -> CdmRefFrame {
    match s.to_uppercase().as_str() {
        "EME2000" => CdmRefFrame::EME2000,
        "GCRF" => CdmRefFrame::GCRF,
        "ITRF" => CdmRefFrame::ITRF,
        "TEME" => CdmRefFrame::TEME,
        "TOD" => CdmRefFrame::TOD,
        _ => CdmRefFrame::EME2000, // Default per CCSDS standard
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CDM: &str = r#"CCSDS_CDM_VERS = 1.0
CREATION_DATE = 2024-01-15T12:00:00.000
ORIGINATOR = MYCELIX-SPACE
MESSAGE_ID = CDM-25544-99999-20240115T180000

TCA = 2024-01-15T18:00:00.000
MISS_DISTANCE = 0.500000 [km]
RELATIVE_SPEED = 14500.000000 [m/s]
RELATIVE_POSITION_R = 200.000000 [m]
RELATIVE_POSITION_T = 300.000000 [m]
RELATIVE_POSITION_N = 100.000000 [m]
RELATIVE_VELOCITY_R = 5000.000000 [m/s]
RELATIVE_VELOCITY_T = 12000.000000 [m/s]
RELATIVE_VELOCITY_N = 3000.000000 [m/s]

COLLISION_PROBABILITY = 1.000000e-05
COLLISION_PROBABILITY_METHOD = ALFANO-2005

COMMENT Object 1 (Primary)
OBJECT1_OBJECT_DESIGNATOR = 25544
OBJECT1_CATALOG_NAME = SATCAT
OBJECT1_OBJECT_NAME = ISS (ZARYA)
OBJECT1_INTERNATIONAL_DESIGNATOR = 1998-067A
OBJECT1_OBJECT_TYPE = PAYLOAD
OBJECT1_EPHEMERIS_NAME = MYCELIX-SPACE
OBJECT1_COVARIANCE_METHOD = CALCULATED
OBJECT1_MANEUVERABLE = YES
OBJECT1_REF_FRAME = TEME
OBJECT1_X = 6800.000000 [km]
OBJECT1_Y = 100.000000 [km]
OBJECT1_Z = 50.000000 [km]
OBJECT1_X_DOT = 0.100000000 [km/s]
OBJECT1_Y_DOT = 7.660000000 [km/s]
OBJECT1_Z_DOT = 0.050000000 [km/s]

COMMENT Object 2 (Secondary)
OBJECT2_OBJECT_DESIGNATOR = 99999
OBJECT2_CATALOG_NAME = SATCAT
OBJECT2_OBJECT_NAME = COSMOS 1408 DEB
OBJECT2_INTERNATIONAL_DESIGNATOR = 2021-101A
OBJECT2_OBJECT_TYPE = DEBRIS
OBJECT2_EPHEMERIS_NAME = MYCELIX-SPACE
OBJECT2_COVARIANCE_METHOD = CALCULATED
OBJECT2_MANEUVERABLE = NO
OBJECT2_REF_FRAME = TEME
OBJECT2_X = 6800.500000 [km]
OBJECT2_Y = 100.000000 [km]
OBJECT2_Z = 50.000000 [km]
OBJECT2_X_DOT = 0.100000000 [km/s]
OBJECT2_Y_DOT = 7.660000000 [km/s]
OBJECT2_Z_DOT = 0.050000000 [km/s]
"#;

    #[test]
    fn test_parse_reference_cdm() {
        let cdm = parse_cdm_kvn(SAMPLE_CDM).unwrap();

        assert_eq!(cdm.ccsds_cdm_vers, "1.0");
        assert_eq!(cdm.originator, "MYCELIX-SPACE");
        assert_eq!(cdm.message_id, "CDM-25544-99999-20240115T180000");
        assert!((cdm.miss_distance - 0.5).abs() < 0.001);
        assert!((cdm.relative_speed - 14500.0).abs() < 0.1);
        assert!((cdm.collision_probability - 1e-5).abs() < 1e-10);
        assert_eq!(cdm.collision_probability_method, PcMethod::Alfano2005);

        // Object 1
        assert_eq!(cdm.object1.metadata.object_designator, "25544");
        assert_eq!(cdm.object1.metadata.object_name, "ISS (ZARYA)");
        assert_eq!(cdm.object1.metadata.maneuverable, Maneuverable::Yes);
        assert_eq!(cdm.object1.metadata.ref_frame, CdmRefFrame::TEME);
        assert!((cdm.object1.state_vector.x - 6800.0).abs() < 0.001);

        // Object 2
        assert_eq!(cdm.object2.metadata.object_designator, "99999");
        assert_eq!(cdm.object2.metadata.object_name, "COSMOS 1408 DEB");
        assert_eq!(cdm.object2.metadata.maneuverable, Maneuverable::No);
    }

    #[test]
    fn test_roundtrip_generate_parse() {
        // Generate KVN from code, then parse it back
        let cdm_original = parse_cdm_kvn(SAMPLE_CDM).unwrap();
        let kvn_output = cdm_original.to_kvn();
        let cdm_roundtrip = parse_cdm_kvn(&kvn_output).unwrap();

        assert_eq!(cdm_original.originator, cdm_roundtrip.originator);
        assert!((cdm_original.miss_distance - cdm_roundtrip.miss_distance).abs() < 0.001);
        assert!(
            (cdm_original.collision_probability - cdm_roundtrip.collision_probability).abs() < 1e-8
        );
        assert_eq!(
            cdm_original.object1.metadata.object_name,
            cdm_roundtrip.object1.metadata.object_name
        );
        assert_eq!(
            cdm_original.object2.metadata.object_name,
            cdm_roundtrip.object2.metadata.object_name
        );
    }

    #[test]
    fn test_missing_required_field() {
        let incomplete = "CCSDS_CDM_VERS = 1.0\nORIGINATOR = TEST\n";
        let err = parse_cdm_kvn(incomplete);
        assert!(err.is_err());
        match err.unwrap_err() {
            CdmParseError::MissingField(field) => {
                assert_eq!(field, "CREATION_DATE");
            }
            other => panic!("Expected MissingField, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_input() {
        let err = parse_cdm_kvn("").unwrap_err();
        assert!(matches!(err, CdmParseError::EmptyInput));
    }

    #[test]
    fn test_unsupported_version() {
        let bad_version = "CCSDS_CDM_VERS = 2.0\nCREATION_DATE = 2024-01-01T00:00:00\n";
        let err = parse_cdm_kvn(bad_version).unwrap_err();
        assert!(matches!(err, CdmParseError::UnsupportedVersion(_)));
    }

    #[test]
    fn test_strip_units() {
        assert_eq!(strip_units("1.234 [km]"), "1.234");
        assert_eq!(strip_units("1.234e-05 [km**2]"), "1.234e-05");
        assert_eq!(strip_units("ALFANO-2005"), "ALFANO-2005");
        assert_eq!(strip_units("  42.0  "), "42.0");
    }

    #[test]
    fn test_parse_datetime_iso() {
        let dt = parse_cdm_datetime("2024-01-15T18:30:45.123", "test").unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 15);
        assert_eq!(dt.hour(), 18);
    }

    #[test]
    fn test_parse_datetime_doy() {
        // Day-of-year format: 2024-015T12:00:00.000 (Jan 15)
        let dt = parse_cdm_datetime("2024-015T12:00:00.000", "test").unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 15);
    }

    use chrono::{Datelike, Timelike};

    /// Helper: generate all 21 covariance KVN lines for a given object prefix.
    fn full_covariance_kvn(prefix: &str) -> String {
        format!(
            r#"{pfx}_CR_R = 1.0e-04 [km**2]
{pfx}_CT_R = 2.0e-05 [km**2]
{pfx}_CT_T = 5.0e-04 [km**2]
{pfx}_CN_R = 1.0e-06 [km**2]
{pfx}_CN_T = 2.0e-06 [km**2]
{pfx}_CN_N = 3.0e-04 [km**2]
{pfx}_CRDOT_R = 1.0e-07 [km**2/s]
{pfx}_CRDOT_T = 2.0e-07 [km**2/s]
{pfx}_CRDOT_N = 3.0e-07 [km**2/s]
{pfx}_CRDOT_RDOT = 4.0e-08 [(km/s)**2]
{pfx}_CTDOT_R = 5.0e-07 [km**2/s]
{pfx}_CTDOT_T = 6.0e-07 [km**2/s]
{pfx}_CTDOT_N = 7.0e-07 [km**2/s]
{pfx}_CTDOT_RDOT = 8.0e-08 [(km/s)**2]
{pfx}_CTDOT_TDOT = 9.0e-08 [(km/s)**2]
{pfx}_CNDOT_R = 1.0e-07 [km**2/s]
{pfx}_CNDOT_T = 1.1e-07 [km**2/s]
{pfx}_CNDOT_N = 1.2e-07 [km**2/s]
{pfx}_CNDOT_RDOT = 1.3e-08 [(km/s)**2]
{pfx}_CNDOT_TDOT = 1.4e-08 [(km/s)**2]
{pfx}_CNDOT_NDOT = 1.5e-08 [(km/s)**2]
"#,
            pfx = prefix
        )
    }

    #[test]
    fn test_covariance_parsing_complete() {
        let cdm_with_cov = format!(
            r#"CCSDS_CDM_VERS = 1.0
CREATION_DATE = 2024-01-15T12:00:00.000
ORIGINATOR = TEST
MESSAGE_ID = TEST-001

TCA = 2024-01-15T18:00:00.000
MISS_DISTANCE = 1.0 [km]
RELATIVE_SPEED = 10000.0 [m/s]

COLLISION_PROBABILITY = 1.0e-06
COLLISION_PROBABILITY_METHOD = ALFANO-2005

OBJECT1_OBJECT_DESIGNATOR = 11111
OBJECT1_OBJECT_NAME = SAT-A
OBJECT1_REF_FRAME = EME2000
OBJECT1_MANEUVERABLE = YES
OBJECT1_X = 7000.0 [km]
OBJECT1_Y = 0.0 [km]
OBJECT1_Z = 0.0 [km]
OBJECT1_X_DOT = 0.0 [km/s]
OBJECT1_Y_DOT = 7.5 [km/s]
OBJECT1_Z_DOT = 0.0 [km/s]
{cov1}
OBJECT2_OBJECT_DESIGNATOR = 22222
OBJECT2_OBJECT_NAME = DEBRIS-B
OBJECT2_REF_FRAME = EME2000
OBJECT2_MANEUVERABLE = NO
OBJECT2_X = 7000.5 [km]
OBJECT2_Y = 0.0 [km]
OBJECT2_Z = 0.0 [km]
OBJECT2_X_DOT = 0.0 [km/s]
OBJECT2_Y_DOT = -7.5 [km/s]
OBJECT2_Z_DOT = 0.0 [km/s]
"#,
            cov1 = full_covariance_kvn("OBJECT1"),
        );

        let cdm = parse_cdm_kvn(&cdm_with_cov).unwrap();

        // Object 1 should have full covariance
        let cov1 = cdm.object1.covariance.as_ref().unwrap();
        assert!((cov1.cr_r - 1.0e-4).abs() < 1e-10);
        assert!((cov1.ct_r - 2.0e-5).abs() < 1e-10);
        assert!((cov1.ct_t - 5.0e-4).abs() < 1e-10);
        assert!((cov1.cndot_ndot - 1.5e-8).abs() < 1e-14);

        // Object 2 should NOT have covariance (no CR_R)
        assert!(cdm.object2.covariance.is_none());
    }

    #[test]
    fn test_incomplete_covariance_rejected() {
        // Only 3 of 21 covariance elements — must be rejected
        let cdm_partial_cov = r#"CCSDS_CDM_VERS = 1.0
CREATION_DATE = 2024-01-15T12:00:00.000
ORIGINATOR = TEST
MESSAGE_ID = TEST-001

TCA = 2024-01-15T18:00:00.000
MISS_DISTANCE = 1.0 [km]
RELATIVE_SPEED = 10000.0 [m/s]

COLLISION_PROBABILITY = 1.0e-06
COLLISION_PROBABILITY_METHOD = ALFANO-2005

OBJECT1_OBJECT_DESIGNATOR = 11111
OBJECT1_OBJECT_NAME = SAT-A
OBJECT1_REF_FRAME = EME2000
OBJECT1_MANEUVERABLE = YES
OBJECT1_X = 7000.0 [km]
OBJECT1_Y = 0.0 [km]
OBJECT1_Z = 0.0 [km]
OBJECT1_X_DOT = 0.0 [km/s]
OBJECT1_Y_DOT = 7.5 [km/s]
OBJECT1_Z_DOT = 0.0 [km/s]
OBJECT1_CR_R = 1.0e-04 [km**2]
OBJECT1_CT_R = 2.0e-05 [km**2]
OBJECT1_CT_T = 5.0e-04 [km**2]

OBJECT2_OBJECT_DESIGNATOR = 22222
OBJECT2_OBJECT_NAME = DEBRIS-B
OBJECT2_REF_FRAME = EME2000
OBJECT2_MANEUVERABLE = NO
OBJECT2_X = 7000.5 [km]
OBJECT2_Y = 0.0 [km]
OBJECT2_Z = 0.0 [km]
OBJECT2_X_DOT = 0.0 [km/s]
OBJECT2_Y_DOT = -7.5 [km/s]
OBJECT2_Z_DOT = 0.0 [km/s]
"#;

        let err = parse_cdm_kvn(cdm_partial_cov).unwrap_err();
        match err {
            CdmParseError::IncompleteCovariance { object, missing } => {
                assert_eq!(object, "OBJECT1");
                // Should list the 18 missing elements
                assert!(missing.contains("CN_R"));
                assert!(missing.contains("CNDOT_NDOT"));
                assert!(!missing.contains("CR_R")); // CR_R is present
                assert!(!missing.contains("CT_R")); // CT_R is present
                assert!(!missing.contains("CT_T")); // CT_T is present
            }
            other => panic!("Expected IncompleteCovariance, got {:?}", other),
        }
    }

    #[test]
    fn test_creation_after_tca_rejected() {
        // CREATION_DATE is 2024-01-16 but TCA is 2024-01-15 — malformed
        let stale_cdm = r#"CCSDS_CDM_VERS = 1.0
CREATION_DATE = 2024-01-16T12:00:00.000
ORIGINATOR = TEST
MESSAGE_ID = TEST-001

TCA = 2024-01-15T18:00:00.000
MISS_DISTANCE = 1.0 [km]
RELATIVE_SPEED = 10000.0 [m/s]

COLLISION_PROBABILITY = 1.0e-06
COLLISION_PROBABILITY_METHOD = FOSTER-1992

OBJECT1_OBJECT_DESIGNATOR = 11111
OBJECT1_OBJECT_NAME = SAT-A
OBJECT1_REF_FRAME = EME2000
OBJECT1_MANEUVERABLE = YES
OBJECT1_X = 7000.0 [km]
OBJECT1_Y = 0.0 [km]
OBJECT1_Z = 0.0 [km]
OBJECT1_X_DOT = 0.0 [km/s]
OBJECT1_Y_DOT = 7.5 [km/s]
OBJECT1_Z_DOT = 0.0 [km/s]

OBJECT2_OBJECT_DESIGNATOR = 22222
OBJECT2_OBJECT_NAME = DEBRIS-B
OBJECT2_REF_FRAME = EME2000
OBJECT2_MANEUVERABLE = NO
OBJECT2_X = 7000.5 [km]
OBJECT2_Y = 0.0 [km]
OBJECT2_Z = 0.0 [km]
OBJECT2_X_DOT = 0.0 [km/s]
OBJECT2_Y_DOT = -7.5 [km/s]
OBJECT2_Z_DOT = 0.0 [km/s]
"#;

        let err = parse_cdm_kvn(stale_cdm).unwrap_err();
        match err {
            CdmParseError::CreationAfterTca { .. } => {} // expected
            other => panic!("Expected CreationAfterTca, got {:?}", other),
        }
    }

    #[test]
    fn test_pc_method_known_variants() {
        assert_eq!(
            PcMethod::from_str_permissive("ALFANO-2005"),
            PcMethod::Alfano2005
        );
        assert_eq!(
            PcMethod::from_str_permissive("FOSTER-1992"),
            PcMethod::Foster1992
        );
        assert_eq!(
            PcMethod::from_str_permissive("PATERA-2001"),
            PcMethod::Patera2001
        );
        assert_eq!(
            PcMethod::from_str_permissive("MCKINLEY-2006"),
            PcMethod::McKinley2006
        );
        assert_eq!(
            PcMethod::from_str_permissive("CHAN-2008"),
            PcMethod::Chan2008
        );
        assert_eq!(
            PcMethod::from_str_permissive("MONTE-CARLO"),
            PcMethod::MonteCarlo
        );
        assert_eq!(
            PcMethod::from_str_permissive("MonteCarlo"),
            PcMethod::MonteCarlo
        );
    }

    #[test]
    fn test_pc_method_unknown_becomes_other() {
        let m = PcMethod::from_str_permissive("CUSTOM-METHOD-2026");
        assert_eq!(m, PcMethod::Other("CUSTOM-METHOD-2026".to_string()));
        assert_eq!(m.to_string(), "CUSTOM-METHOD-2026");
    }

    #[test]
    fn test_pc_method_parsed_from_cdm() {
        let cdm = parse_cdm_kvn(SAMPLE_CDM).unwrap();
        assert_eq!(cdm.collision_probability_method, PcMethod::Alfano2005);
    }
}
