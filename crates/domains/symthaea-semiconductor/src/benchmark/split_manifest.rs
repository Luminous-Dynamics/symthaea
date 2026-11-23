// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Split-manifest admission and leakage validation for SEMI-BENCH-001.

use chrono::{DateTime, FixedOffset, NaiveDateTime};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const INPUT_SCHEMA_V1: &str = "luminous.semi-bench.split-manifest.v1";
pub const OUTPUT_SCHEMA_V1: &str = "luminous.semi-bench.split-validation-report.v1";
const SPLITS: [&str; 3] = ["train", "validation", "test"];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplitAssignmentV1 {
    pub sample_id: String,
    pub split: String,
    pub positive: bool,
    #[serde(default)]
    pub group_id: Option<String>,
    #[serde(default)]
    pub observed_at: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplitManifestV1 {
    pub schema: String,
    pub dataset_id: String,
    pub dataset_snapshot_sha256: String,
    pub split_scheme: String,
    #[serde(default)]
    pub time_basis: Option<String>,
    #[serde(default)]
    pub clock_domain_id: Option<String>,
    pub assignments: Vec<SplitAssignmentV1>,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct SplitManifestError(pub String);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplitCountsV1 {
    pub samples: usize,
    pub positive: usize,
    pub negative: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplitGuaranteesV1 {
    pub unique_sample_identity: bool,
    pub all_splits_non_empty: bool,
    pub group_disjoint: bool,
    pub strict_forward_time: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemporalBoundsV1 {
    pub minimum: String,
    pub maximum: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplitClaimLimitsV1 {
    pub group_independence: String,
    pub temporal_independence: String,
    pub stratified_sample_split: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplitValidationReportV1 {
    pub schema: String,
    pub dataset_id: String,
    pub dataset_snapshot_sha256: String,
    pub split_scheme: String,
    pub split_manifest_sha256: String,
    pub split_counts: BTreeMap<String, SplitCountsV1>,
    pub guarantees: SplitGuaranteesV1,
    pub temporal_bounds: Option<BTreeMap<String, TemporalBoundsV1>>,
    pub claim_limits: SplitClaimLimitsV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_basis: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clock_domain_id: Option<String>,
}

#[derive(Clone, Debug)]
enum ParsedTime {
    Offset(DateTime<FixedOffset>),
    Naive(NaiveDateTime),
}

impl ParsedTime {
    fn parse(raw: &str, basis: &str, field: &str) -> Result<Self, SplitManifestError> {
        match basis {
            "offset_aware" => {
                let parsed = DateTime::parse_from_rfc3339(raw).map_err(|_| {
                    SplitManifestError(format!(
                        "{field}: timezone offset is required for offset_aware"
                    ))
                })?;
                Ok(Self::Offset(parsed))
            }
            "source_local_naive" => {
                if raw.ends_with('Z') || has_explicit_offset(raw) {
                    return Err(SplitManifestError(format!(
                        "{field}: must be timezone-free for source_local_naive"
                    )));
                }
                let parsed = parse_naive_datetime(raw).ok_or_else(|| {
                    SplitManifestError(format!("{field}: expected ISO-8601 timestamp"))
                })?;
                Ok(Self::Naive(parsed))
            }
            _ => Err(SplitManifestError(
                "time_basis: expected offset_aware or source_local_naive".to_string(),
            )),
        }
    }

    fn canonical(&self) -> String {
        match self {
            Self::Offset(value) => value.format("%Y-%m-%dT%H:%M:%S%.f%:z").to_string(),
            Self::Naive(value) => value.format("%Y-%m-%dT%H:%M:%S%.f").to_string(),
        }
    }

    fn compare(&self, other: &Self) -> Result<std::cmp::Ordering, SplitManifestError> {
        match (self, other) {
            (Self::Offset(left), Self::Offset(right)) => Ok(left.cmp(right)),
            (Self::Naive(left), Self::Naive(right)) => Ok(left.cmp(right)),
            _ => Err(SplitManifestError(
                "timestamp clock basis mismatch".to_string(),
            )),
        }
    }
}

fn has_explicit_offset(raw: &str) -> bool {
    let Some(time_start) = raw.find('T').or_else(|| raw.find('t')) else {
        return false;
    };
    raw[time_start + 1..].contains('+') || raw[time_start + 1..].rfind('-').is_some()
}

fn parse_naive_datetime(raw: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(raw, "%Y-%m-%dT%H:%M:%S%.f")
        .or_else(|_| NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S%.f"))
        .ok()
}

fn require_text(value: &str, field: &str) -> Result<(), SplitManifestError> {
    if value.trim().is_empty() {
        Err(SplitManifestError(format!(
            "{field}: expected non-empty string"
        )))
    } else {
        Ok(())
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn split_counts() -> BTreeMap<String, SplitCountsV1> {
    SPLITS
        .iter()
        .map(|split| {
            (
                (*split).to_string(),
                SplitCountsV1 {
                    samples: 0,
                    positive: 0,
                    negative: 0,
                },
            )
        })
        .collect()
}

fn canonical_manifest_payload(
    manifest: &SplitManifestV1,
    canonical_rows: &[Value],
    time_basis: Option<&str>,
    clock_domain_id: Option<&str>,
) -> Value {
    let mut payload = Map::new();
    payload.insert("schema".into(), json!(INPUT_SCHEMA_V1));
    payload.insert("dataset_id".into(), json!(manifest.dataset_id));
    payload.insert(
        "dataset_snapshot_sha256".into(),
        json!(manifest.dataset_snapshot_sha256),
    );
    payload.insert("split_scheme".into(), json!(manifest.split_scheme));
    payload.insert("assignments".into(), json!(canonical_rows));
    if manifest.split_scheme == "forward_time" {
        payload.insert("time_basis".into(), json!(time_basis));
        if time_basis == Some("source_local_naive") {
            payload.insert("clock_domain_id".into(), json!(clock_domain_id));
        }
    }
    Value::Object(payload)
}

/// Validates a versioned split manifest and computes its canonical digest.
///
/// Group-disjoint and forward-time guarantees are checked independently. A
/// source-local naive clock can only establish chronology within the declared
/// clock domain; it is never converted to UTC.
pub fn validate_split_manifest_v1(
    manifest: &SplitManifestV1,
) -> Result<SplitValidationReportV1, SplitManifestError> {
    if manifest.schema != INPUT_SCHEMA_V1 {
        return Err(SplitManifestError(
            "schema: unsupported or missing split-manifest schema".to_string(),
        ));
    }
    require_text(&manifest.dataset_id, "dataset_id")?;
    if !is_sha256(&manifest.dataset_snapshot_sha256) {
        return Err(SplitManifestError(
            "dataset_snapshot_sha256: expected lowercase SHA-256 hex".to_string(),
        ));
    }
    if !["group_disjoint", "forward_time", "stratified_sample"]
        .contains(&manifest.split_scheme.as_str())
    {
        return Err(SplitManifestError(
            "split_scheme: unsupported scheme".to_string(),
        ));
    }
    if manifest.assignments.is_empty() {
        return Err(SplitManifestError(
            "assignments: expected non-empty array".to_string(),
        ));
    }

    let mut seen_ids = BTreeSet::new();
    let mut group_splits: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut split_times: BTreeMap<String, Vec<ParsedTime>> = BTreeMap::new();
    let mut counts = split_counts();
    let mut canonical_rows = Vec::with_capacity(manifest.assignments.len());

    let time_basis = if manifest.split_scheme == "forward_time" {
        let basis = manifest.time_basis.as_deref().ok_or_else(|| {
            SplitManifestError(
                "time_basis: expected offset_aware or source_local_naive".to_string(),
            )
        })?;
        if !["offset_aware", "source_local_naive"].contains(&basis) {
            return Err(SplitManifestError(
                "time_basis: expected offset_aware or source_local_naive".to_string(),
            ));
        }
        if basis == "source_local_naive" {
            require_text(
                manifest.clock_domain_id.as_deref().unwrap_or_default(),
                "clock_domain_id",
            )?;
        }
        Some(basis)
    } else {
        None
    };

    for (index, row) in manifest.assignments.iter().enumerate() {
        require_text(&row.sample_id, &format!("assignments[{index}].sample_id"))?;
        if !seen_ids.insert(row.sample_id.clone()) {
            return Err(SplitManifestError(format!(
                "assignments[{index}].sample_id: duplicate {:?}",
                row.sample_id
            )));
        }
        if !SPLITS.contains(&row.split.as_str()) {
            return Err(SplitManifestError(format!(
                "assignments[{index}].split: expected train, validation or test"
            )));
        }

        let count = counts.get_mut(&row.split).expect("validated split key");
        count.samples += 1;
        if row.positive {
            count.positive += 1;
        } else {
            count.negative += 1;
        }

        let mut canonical = Map::new();
        canonical.insert("sample_id".into(), json!(row.sample_id));
        canonical.insert("split".into(), json!(row.split));
        canonical.insert("positive".into(), json!(row.positive));

        if manifest.split_scheme == "group_disjoint" {
            let group_id = row.group_id.as_deref().ok_or_else(|| {
                SplitManifestError(format!(
                    "assignments[{index}].group_id: expected non-empty string"
                ))
            })?;
            require_text(group_id, &format!("assignments[{index}].group_id"))?;
            group_splits
                .entry(group_id.to_string())
                .or_default()
                .insert(row.split.clone());
            canonical.insert("group_id".into(), json!(group_id));
        } else if manifest.split_scheme == "forward_time" {
            let basis = time_basis.expect("forward-time requires basis");
            let raw = row.observed_at.as_deref().ok_or_else(|| {
                SplitManifestError(format!(
                    "assignments[{index}].observed_at: expected non-empty string"
                ))
            })?;
            require_text(raw, &format!("assignments[{index}].observed_at"))?;
            let parsed =
                ParsedTime::parse(raw, basis, &format!("assignments[{index}].observed_at"))?;
            canonical.insert("observed_at".into(), json!(parsed.canonical()));
            split_times
                .entry(row.split.clone())
                .or_default()
                .push(parsed);
        }
        canonical_rows.push(Value::Object(canonical));
    }

    let empty: Vec<&str> = SPLITS
        .iter()
        .copied()
        .filter(|split| counts.get(*split).map(|c| c.samples == 0).unwrap_or(true))
        .collect();
    if !empty.is_empty() {
        return Err(SplitManifestError(format!(
            "assignments: empty required split(s): {}",
            empty.join(", ")
        )));
    }

    let group_leakage: Vec<String> = group_splits
        .iter()
        .filter(|(_, assigned)| assigned.len() > 1)
        .map(|(group, _)| group.clone())
        .collect();
    if manifest.split_scheme == "group_disjoint" && !group_leakage.is_empty() {
        return Err(SplitManifestError(format!(
            "group leakage across splits: {:?}",
            group_leakage[0]
        )));
    }

    let mut temporal_bounds = None;
    if manifest.split_scheme == "forward_time" {
        let mut bounds = BTreeMap::new();
        for split in SPLITS {
            let values = split_times.get(split).ok_or_else(|| {
                SplitManifestError(format!("forward_time: missing timestamps in {split}"))
            })?;
            if values.is_empty() {
                return Err(SplitManifestError(format!(
                    "forward_time: missing timestamps in {split}"
                )));
            }
            let minimum = values
                .iter()
                .min_by(|a, b| a.compare(b).expect("clock basis already validated"))
                .expect("non-empty timestamp list");
            let maximum = values
                .iter()
                .max_by(|a, b| a.compare(b).expect("clock basis already validated"))
                .expect("non-empty timestamp list");
            bounds.insert(
                split.to_string(),
                TemporalBoundsV1 {
                    minimum: minimum.canonical(),
                    maximum: maximum.canonical(),
                },
            );
        }
        let train_max = split_times["train"]
            .iter()
            .max_by(|a, b| a.compare(b).expect("clock basis already validated"))
            .expect("non-empty train timestamps");
        let validation_min = split_times["validation"]
            .iter()
            .min_by(|a, b| a.compare(b).expect("clock basis already validated"))
            .expect("non-empty validation timestamps");
        let validation_max = split_times["validation"]
            .iter()
            .max_by(|a, b| a.compare(b).expect("clock basis already validated"))
            .expect("non-empty validation timestamps");
        let test_min = split_times["test"]
            .iter()
            .min_by(|a, b| a.compare(b).expect("clock basis already validated"))
            .expect("non-empty test timestamps");
        if train_max.compare(validation_min)? != std::cmp::Ordering::Less
            || validation_max.compare(test_min)? != std::cmp::Ordering::Less
        {
            return Err(SplitManifestError(
                "forward_time: train, validation and test windows must be strictly chronological and non-overlapping".to_string(),
            ));
        }
        temporal_bounds = Some(bounds);
    }

    canonical_rows
        .sort_by(|left, right| left["sample_id"].as_str().cmp(&right["sample_id"].as_str()));
    let payload = canonical_manifest_payload(
        manifest,
        &canonical_rows,
        time_basis,
        manifest.clock_domain_id.as_deref(),
    );
    let serialized = serde_json::to_vec(&payload).map_err(|error| {
        SplitManifestError(format!("canonical manifest serialization failed: {error}"))
    })?;
    let split_digest = format!("{:x}", Sha256::digest(serialized));

    let report = SplitValidationReportV1 {
        schema: OUTPUT_SCHEMA_V1.to_string(),
        dataset_id: manifest.dataset_id.clone(),
        dataset_snapshot_sha256: manifest.dataset_snapshot_sha256.clone(),
        split_scheme: manifest.split_scheme.clone(),
        split_manifest_sha256: split_digest,
        split_counts: counts,
        guarantees: SplitGuaranteesV1 {
            unique_sample_identity: true,
            all_splits_non_empty: true,
            group_disjoint: manifest.split_scheme == "group_disjoint",
            strict_forward_time: manifest.split_scheme == "forward_time",
        },
        temporal_bounds,
        claim_limits: SplitClaimLimitsV1 {
            group_independence: if manifest.split_scheme == "group_disjoint" {
                "established-for-this-manifest"
            } else {
                "not-established"
            }
            .to_string(),
            temporal_independence: if manifest.split_scheme == "forward_time"
                && time_basis == Some("offset_aware")
            {
                "established-for-this-manifest"
            } else if manifest.split_scheme == "forward_time"
                && time_basis == Some("source_local_naive")
            {
                "established-within-declared-source-clock-domain-only"
            } else {
                "not-established"
            }
            .to_string(),
            stratified_sample_split: manifest.split_scheme == "stratified_sample",
        },
        time_basis: if manifest.split_scheme == "forward_time" {
            time_basis.map(str::to_string)
        } else {
            None
        },
        clock_domain_id: if manifest.split_scheme == "forward_time"
            && time_basis == Some("source_local_naive")
        {
            manifest.clock_domain_id.clone()
        } else {
            None
        },
    };
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const INPUT: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/split-manifest-known-answer-v1.json"
    );
    const EXPECTED: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/split-manifest-known-answer-v1.expected.json"
    );
    const SECOM_MANIFEST: &str = include_str!(
        "../../../../../docs/research/semi-bench-001/splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json"
    );

    fn known_answer() -> SplitManifestV1 {
        serde_json::from_str(INPUT).expect("split fixture must parse")
    }

    #[test]
    fn rust_split_report_matches_frozen_python_reference() {
        let report = validate_split_manifest_v1(&known_answer()).expect("manifest validates");
        let actual = serde_json::to_value(report).expect("report serializes");
        let expected: Value = serde_json::from_str(EXPECTED).expect("expected parses");
        assert_eq!(actual, expected);
    }

    #[test]
    fn frozen_secom_source_clock_manifest_has_expected_support() {
        let manifest: SplitManifestV1 =
            serde_json::from_str(SECOM_MANIFEST).expect("SECOM chronology manifest parses");
        let report = validate_split_manifest_v1(&manifest).expect("SECOM chronology validates");
        assert_eq!(
            manifest.dataset_snapshot_sha256,
            "26c59ab70a7339d82ab4e4006a5230d1a2ffff48c2df3fc423a458f9ef152281"
        );
        assert_eq!(report.time_basis.as_deref(), Some("source_local_naive"));
        assert_eq!(
            report.clock_domain_id.as_deref(),
            Some("uci-secom-mirror-single-source-clock")
        );
        assert_eq!(
            report.split_counts["train"],
            SplitCountsV1 {
                samples: 940,
                positive: 76,
                negative: 864,
            }
        );
        assert_eq!(
            report.split_counts["validation"],
            SplitCountsV1 {
                samples: 313,
                positive: 11,
                negative: 302,
            }
        );
        assert_eq!(
            report.split_counts["test"],
            SplitCountsV1 {
                samples: 314,
                positive: 17,
                negative: 297,
            }
        );
        assert_eq!(
            report.claim_limits.temporal_independence,
            "established-within-declared-source-clock-domain-only"
        );
    }

    #[test]
    fn group_leakage_is_rejected() {
        let mut manifest = known_answer();
        manifest.assignments[3].group_id = Some("LOT-A".to_string());
        assert!(
            validate_split_manifest_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("group leakage")
        );
    }

    #[test]
    fn duplicate_sample_identity_is_rejected() {
        let mut manifest = known_answer();
        manifest.assignments[1].sample_id = manifest.assignments[0].sample_id.clone();
        assert!(
            validate_split_manifest_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
    }

    #[test]
    fn requires_all_three_splits() {
        let mut manifest = known_answer();
        manifest.assignments.retain(|row| row.split != "test");
        assert!(
            validate_split_manifest_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("empty required split")
        );
    }

    #[test]
    fn rejects_naive_clock_without_declared_domain() {
        let mut manifest = known_answer();
        manifest.split_scheme = "forward_time".to_string();
        manifest.time_basis = Some("source_local_naive".to_string());
        manifest.clock_domain_id = None;
        assert!(
            validate_split_manifest_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("clock_domain_id")
        );
    }
}
