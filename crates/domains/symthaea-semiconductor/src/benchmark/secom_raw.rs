// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Digest-bound SECOM raw-text ingestion and integrity audit.
//!
//! Sensor missingness is represented as Option::None, never silently changed
//! to zero. Source timestamps remain source-local naive values; this module
//! does not infer a timezone or merge this clock with another source.

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::str;
use thiserror::Error;

pub const EXPECTATIONS_SCHEMA_V1: &str = "luminous.semi-bench.secom-raw-expectations.v1";
pub const REPORT_SCHEMA_V1: &str = "luminous.semi-bench.secom-raw-integrity-report.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomRawExpectationsV1 {
    pub schema: String,
    pub dataset_id: String,
    pub expected_feature_sha256: String,
    pub expected_label_sha256: String,
    pub expected_rows: usize,
    pub expected_features_per_row: usize,
    pub clock_domain_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SecomSensorRowV1 {
    pub values: Vec<Option<f64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SecomLabelRowV1 {
    /// -1 is pass; +1 is fail in the source dataset.
    pub label: i8,
    /// Parsed as a naive source-local timestamp. It is not UTC.
    pub observed_at: NaiveDateTime,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomClassCountsV1 {
    pub pass: usize,
    pub fail: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomRawIntegrityReportV1 {
    pub schema: String,
    pub dataset_id: String,
    pub feature_sha256: String,
    pub label_sha256: String,
    pub rows: usize,
    pub features_per_row: usize,
    pub missing_sensor_cells: usize,
    pub total_sensor_cells: usize,
    pub missing_fraction: f64,
    pub rows_with_any_missing: usize,
    pub class_counts: SecomClassCountsV1,
    pub unique_timestamps: usize,
    pub duplicate_timestamp_rows: usize,
    pub timestamps_non_decreasing: bool,
    pub first_timestamp_source_local: String,
    pub last_timestamp_source_local: String,
    pub clock_domain_id: String,
    pub timezone_interpretation: String,
    pub claim_limits: SecomRawClaimLimitsV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomRawClaimLimitsV1 {
    pub validates_source_file_identity: bool,
    pub preserves_missing_measurements: bool,
    pub preserves_source_local_clock_only: bool,
    pub does_not_establish_feature_semantics_or_units: bool,
    pub does_not_establish_group_or_tool_independence: bool,
    pub does_not_establish_predictive_performance: bool,
    pub does_not_establish_causality_or_yield_improvement: bool,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct SecomRawDataError(pub String);

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_sha256(value: &str, field: &str) -> Result<(), SecomRawDataError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(SecomRawDataError(format!(
            "{field}: expected lowercase SHA-256 hex"
        )));
    }
    Ok(())
}

fn parse_features(bytes: &[u8]) -> Result<Vec<SecomSensorRowV1>, SecomRawDataError> {
    let text = str::from_utf8(bytes)
        .map_err(|error| SecomRawDataError(format!("feature_file: invalid UTF-8: {error}")))?;
    let mut rows = Vec::new();
    for (row_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(SecomRawDataError(format!(
                "feature_file line {}: blank row is not permitted",
                row_index + 1
            )));
        }
        let mut values = Vec::new();
        for (column_index, token) in line.split_whitespace().enumerate() {
            if token.eq_ignore_ascii_case("nan") {
                values.push(None);
                continue;
            }
            let value = token.parse::<f64>().map_err(|_| {
                SecomRawDataError(format!(
                    "feature_file line {}, column {}: invalid numeric token {:?}",
                    row_index + 1,
                    column_index + 1,
                    token
                ))
            })?;
            if !value.is_finite() {
                return Err(SecomRawDataError(format!(
                    "feature_file line {}, column {}: non-finite values must be encoded as NaN",
                    row_index + 1,
                    column_index + 1
                )));
            }
            values.push(Some(value));
        }
        rows.push(SecomSensorRowV1 { values });
    }
    Ok(rows)
}

fn parse_labels(bytes: &[u8]) -> Result<Vec<SecomLabelRowV1>, SecomRawDataError> {
    let text = str::from_utf8(bytes)
        .map_err(|error| SecomRawDataError(format!("label_file: invalid UTF-8: {error}")))?;
    let mut rows = Vec::new();
    for (row_index, line) in text.lines().enumerate() {
        let parts: Vec<&str> = line.split('"').collect();
        if parts.len() != 3 || !parts[2].trim().is_empty() {
            return Err(SecomRawDataError(format!(
                "label_file line {}: expected label followed by one quoted timestamp",
                row_index + 1
            )));
        }
        let label_value = parts[0].trim().parse::<i8>().map_err(|_| {
            SecomRawDataError(format!("label_file line {}: invalid label", row_index + 1))
        })?;
        if label_value != -1 && label_value != 1 {
            return Err(SecomRawDataError(format!(
                "label_file line {}: label must be -1 (pass) or +1 (fail)",
                row_index + 1
            )));
        }
        let observed_at =
            NaiveDateTime::parse_from_str(parts[1], "%d/%m/%Y %H:%M:%S").map_err(|_| {
                SecomRawDataError(format!(
                    "label_file line {}: timestamp must match DD/MM/YYYY HH:MM:SS without timezone",
                    row_index + 1
                ))
            })?;
        rows.push(SecomLabelRowV1 {
            label: label_value,
            observed_at,
        });
    }
    Ok(rows)
}

/// Audits a pair of SECOM raw files against digest-bound expectations.
///
/// No normalization, imputation, feature dropping, time conversion or
/// resampling occurs. NaN sensor cells remain None in the parsed rows.
pub fn audit_secom_raw_v1(
    feature_bytes: &[u8],
    label_bytes: &[u8],
    expectations: &SecomRawExpectationsV1,
) -> Result<SecomRawIntegrityReportV1, SecomRawDataError> {
    if expectations.schema != EXPECTATIONS_SCHEMA_V1 {
        return Err(SecomRawDataError(
            "schema: unsupported or missing SECOM expectations schema".to_string(),
        ));
    }
    if expectations.dataset_id.trim().is_empty() {
        return Err(SecomRawDataError(
            "dataset_id: expected non-empty string".to_string(),
        ));
    }
    if expectations.clock_domain_id.trim().is_empty() {
        return Err(SecomRawDataError(
            "clock_domain_id: expected non-empty string".to_string(),
        ));
    }
    validate_sha256(
        &expectations.expected_feature_sha256,
        "expected_feature_sha256",
    )?;
    validate_sha256(&expectations.expected_label_sha256, "expected_label_sha256")?;
    if expectations.expected_rows == 0 || expectations.expected_features_per_row == 0 {
        return Err(SecomRawDataError(
            "expectations: expected_rows and expected_features_per_row must be positive"
                .to_string(),
        ));
    }

    let feature_sha256 = sha256_hex(feature_bytes);
    let label_sha256 = sha256_hex(label_bytes);
    if feature_sha256 != expectations.expected_feature_sha256 {
        return Err(SecomRawDataError(format!(
            "feature_file SHA-256 mismatch: expected {}, observed {}",
            expectations.expected_feature_sha256, feature_sha256
        )));
    }
    if label_sha256 != expectations.expected_label_sha256 {
        return Err(SecomRawDataError(format!(
            "label_file SHA-256 mismatch: expected {}, observed {}",
            expectations.expected_label_sha256, label_sha256
        )));
    }

    let features = parse_features(feature_bytes)?;
    let labels = parse_labels(label_bytes)?;
    if features.len() != expectations.expected_rows {
        return Err(SecomRawDataError(format!(
            "feature_file row count mismatch: expected {}, observed {}",
            expectations.expected_rows,
            features.len()
        )));
    }
    if labels.len() != expectations.expected_rows {
        return Err(SecomRawDataError(format!(
            "label_file row count mismatch: expected {}, observed {}",
            expectations.expected_rows,
            labels.len()
        )));
    }
    for (row_index, row) in features.iter().enumerate() {
        if row.values.len() != expectations.expected_features_per_row {
            return Err(SecomRawDataError(format!(
                "feature_file row {} column count mismatch: expected {}, observed {}",
                row_index + 1,
                expectations.expected_features_per_row,
                row.values.len()
            )));
        }
    }

    let mut missing_sensor_cells = 0usize;
    let mut rows_with_any_missing = 0usize;
    for row in &features {
        let row_missing = row.values.iter().filter(|value| value.is_none()).count();
        missing_sensor_cells += row_missing;
        if row_missing > 0 {
            rows_with_any_missing += 1;
        }
    }
    let total_sensor_cells = features.len() * expectations.expected_features_per_row;
    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut unique_timestamps = BTreeSet::new();
    let mut timestamps_non_decreasing = true;
    let mut previous = None;
    for row in &labels {
        match row.label {
            -1 => pass += 1,
            1 => fail += 1,
            _ => unreachable!("label parser admits only -1 or +1"),
        }
        unique_timestamps.insert(row.observed_at);
        if previous.is_some_and(|previous| row.observed_at < previous) {
            timestamps_non_decreasing = false;
        }
        previous = Some(row.observed_at);
    }
    if !timestamps_non_decreasing {
        return Err(SecomRawDataError(
            "label_file timestamps are not non-decreasing in source row order".to_string(),
        ));
    }
    let first_timestamp = labels
        .first()
        .ok_or_else(|| SecomRawDataError("label_file: no rows".to_string()))?
        .observed_at;
    let last_timestamp = labels.last().expect("non-empty label file").observed_at;

    Ok(SecomRawIntegrityReportV1 {
        schema: REPORT_SCHEMA_V1.to_string(),
        dataset_id: expectations.dataset_id.clone(),
        feature_sha256,
        label_sha256,
        rows: features.len(),
        features_per_row: expectations.expected_features_per_row,
        missing_sensor_cells,
        total_sensor_cells,
        missing_fraction: missing_sensor_cells as f64 / total_sensor_cells as f64,
        rows_with_any_missing,
        class_counts: SecomClassCountsV1 { pass, fail },
        unique_timestamps: unique_timestamps.len(),
        duplicate_timestamp_rows: labels.len() - unique_timestamps.len(),
        timestamps_non_decreasing,
        first_timestamp_source_local: first_timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
        last_timestamp_source_local: last_timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
        clock_domain_id: expectations.clock_domain_id.clone(),
        timezone_interpretation: "source_local_naive; no timezone or UTC conversion".to_string(),
        claim_limits: SecomRawClaimLimitsV1 {
            validates_source_file_identity: true,
            preserves_missing_measurements: true,
            preserves_source_local_clock_only: true,
            does_not_establish_feature_semantics_or_units: true,
            does_not_establish_group_or_tool_independence: true,
            does_not_establish_predictive_performance: true,
            does_not_establish_causality_or_yield_improvement: true,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FEATURES: &[u8] = b"1.0 NaN -2.5\n4.0 5.5 6.0\n";
    const LABELS: &[u8] = b"-1 \"19/07/2008 11:55:00\"\n1 \"19/07/2008 12:00:00\"\n";

    fn expectations() -> SecomRawExpectationsV1 {
        SecomRawExpectationsV1 {
            schema: EXPECTATIONS_SCHEMA_V1.to_string(),
            dataset_id: "synthetic-secom-raw-fixture-v1".to_string(),
            expected_feature_sha256:
                "06324b50b87e6ae9b97347907a42c928cca9716fd67a4a444c9c25d03872ae59".to_string(),
            expected_label_sha256:
                "1b6643872e828998aac2fdcca7bd6fc87cb90bc9f8ce41bb95e93cea9e655572".to_string(),
            expected_rows: 2,
            expected_features_per_row: 3,
            clock_domain_id: "synthetic-secom-fixture-clock".to_string(),
        }
    }

    #[test]
    fn preserves_nan_and_validates_file_identity_and_support() {
        let report = audit_secom_raw_v1(FEATURES, LABELS, &expectations()).unwrap();
        assert_eq!(report.rows, 2);
        assert_eq!(report.features_per_row, 3);
        assert_eq!(report.missing_sensor_cells, 1);
        assert_eq!(report.total_sensor_cells, 6);
        assert!((report.missing_fraction - (1.0 / 6.0)).abs() < 1e-12);
        assert_eq!(report.rows_with_any_missing, 1);
        assert_eq!(report.class_counts, SecomClassCountsV1 { pass: 1, fail: 1 });
        assert_eq!(report.unique_timestamps, 2);
        assert_eq!(report.duplicate_timestamp_rows, 0);
        assert_eq!(report.first_timestamp_source_local, "2008-07-19T11:55:00");
        assert_eq!(
            report.timezone_interpretation,
            "source_local_naive; no timezone or UTC conversion"
        );
    }

    #[test]
    fn rejects_feature_digest_mismatch_before_parsing() {
        let mut expected = expectations();
        expected.expected_feature_sha256 = "a".repeat(64);
        assert!(
            audit_secom_raw_v1(FEATURES, LABELS, &expected)
                .unwrap_err()
                .to_string()
                .contains("feature_file SHA-256 mismatch")
        );
    }

    #[test]
    fn rejects_row_alignment_mismatch() {
        let mut expected = expectations();
        expected.expected_rows = 3;
        assert!(
            audit_secom_raw_v1(FEATURES, LABELS, &expected)
                .unwrap_err()
                .to_string()
                .contains("row count mismatch")
        );
    }

    #[test]
    fn rejects_non_monotonic_source_clock() {
        let labels = b"-1 \"19/07/2008 12:00:00\"\n1 \"19/07/2008 11:55:00\"\n";
        let mut expected = expectations();
        expected.expected_label_sha256 = format!("{:x}", Sha256::digest(labels));
        assert!(
            audit_secom_raw_v1(FEATURES, labels, &expected)
                .unwrap_err()
                .to_string()
                .contains("not non-decreasing")
        );
    }

    #[test]
    fn rejects_unknown_target_label() {
        let labels = b"0 \"19/07/2008 11:55:00\"\n1 \"19/07/2008 12:00:00\"\n";
        let mut expected = expectations();
        expected.expected_label_sha256 =
            format!("{:x}", Sha256::digest(labels));
        assert!(
            audit_secom_raw_v1(FEATURES, labels, &expected)
                .unwrap_err()
                .to_string()
                .contains("label must be -1 (pass) or +1 (fail)")
        );
    }
}
