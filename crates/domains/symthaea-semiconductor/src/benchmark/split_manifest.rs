// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Versioned semiconductor benchmark split-manifest validation.
//!
//! The digest binds the canonical source identity and exact split assignment.
//! It does not prove that a dataset is representative or that a model
//! generalizes. Source-local naive clocks establish ordering only within the
//! declared clock domain; they are never interpreted as UTC.

use chrono::{DateTime, FixedOffset, NaiveDateTime, SecondsFormat};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};

pub const INPUT_SCHEMA: &str = "luminous.semi-bench.split-manifest.v1";
pub const OUTPUT_SCHEMA: &str = "luminous.semi-bench.split-validation-report.v1";

const REQUIRED_SPLITS: [&str; 3] = ["train", "validation", "test"];

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SplitAssignmentV1 {
    pub sample_id: String,
    pub split: String,
    pub positive: bool,
    #[serde(default)]
    pub group_id: Option<String>,
    #[serde(default)]
    pub observed_at: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SplitCount {
    pub samples: usize,
    pub positive: usize,
    pub negative: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Guarantees {
    pub unique_sample_identity: bool,
    pub all_splits_non_empty: bool,
    pub group_disjoint: bool,
    pub strict_forward_time: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalBounds {
    pub minimum: String,
    pub maximum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClaimLimits {
    pub group_independence: String,
    pub temporal_independence: String,
    pub stratified_sample_split: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SplitValidationReportV1 {
    pub schema: String,
    pub dataset_id: String,
    pub dataset_snapshot_sha256: String,
    pub split_scheme: String,
    pub split_manifest_sha256: String,
    pub split_counts: BTreeMap<String, SplitCount>,
    pub guarantees: Guarantees,
    pub temporal_bounds: Option<BTreeMap<String, TemporalBounds>>,
    pub claim_limits: ClaimLimits,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_basis: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clock_domain_id: Option<String>,
}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum SplitManifestError {
    #[error("unsupported or missing split-manifest schema: {0}")]
    UnsupportedSchema(String),
    #[error("missing or empty field: {0}")]
    MissingField(&'static str),
    #[error("dataset_snapshot_sha256 must be 64 lowercase hexadecimal characters")]
    InvalidDatasetDigest,
    #[error("unsupported split scheme: {0}")]
    UnsupportedSplitScheme(String),
    #[error("unsupported or missing time_basis")]
    InvalidTimeBasis,
    #[error("source_local_naive time basis requires clock_domain_id")]
    MissingClockDomain,
    #[error("invalid or duplicate sample identity: {0}")]
    DuplicateSampleIdentity(String),
    #[error("invalid split name for sample {sample_id}: {split}")]
    InvalidSplit { sample_id: String, split: String },
    #[error("invalid or missing group_id for sample {0}")]
    MissingGroup(String),
    #[error("group {group_id} appears in more than one split")]
    GroupLeakage { group_id: String },
    #[error("invalid or missing timestamp for sample {0}")]
    InvalidTimestamp(String),
    #[error("timezone offset is required for offset_aware timestamps")]
    MissingTimezoneOffset,
    #[error("source_local_naive timestamps must not include a timezone offset")]
    UnexpectedTimezoneOffset,
    #[error("one or more required splits are empty")]
    EmptyRequiredSplit,
    #[error("train, validation and test windows must be strictly chronological and non-overlapping")]
    NonChronologicalWindows,
    #[error("failed to canonicalize split manifest: {0}")]
    Canonicalization(String),
}

#[derive(Debug, Clone)]
struct ParsedTimestamp {
    order_key: (i64, u32),
    canonical: String,
}

fn parse_timestamp(
    value: &str,
    sample_id: &str,
    time_basis: &str,
) -> Result<ParsedTimestamp, SplitManifestError> {
    match time_basis {
        "offset_aware" => {
            let parsed = DateTime::<FixedOffset>::parse_from_rfc3339(value)
                .map_err(|_| SplitManifestError::InvalidTimestamp(sample_id.to_owned()))?;
            let canonical = parsed.to_rfc3339_opts(SecondsFormat::AutoSi, false);
            Ok(ParsedTimestamp {
                order_key: (parsed.timestamp(), parsed.timestamp_subsec_nanos()),
                canonical,
            })
        }
        "source_local_naive" => {
            let parsed = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f")
                .or_else(|_| NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S"))
                .map_err(|_| SplitManifestError::UnexpectedTimezoneOffset)?;
            // Treat the naive timestamp as a numeric calendar coordinate only
            // for ordering within this declared source clock. This conversion
            // does not establish UTC or cross-domain time comparability.
            let coordinate = parsed.and_utc();
            let canonical = parsed.format("%Y-%m-%dT%H:%M:%S%.f").to_string();
            Ok(ParsedTimestamp {
                order_key: (coordinate.timestamp(), coordinate.timestamp_subsec_nanos()),
                canonical,
            })
        }
        _ => Err(SplitManifestError::InvalidTimeBasis),
    }
}

fn is_nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_manifest_digest(
    manifest: &SplitManifestV1,
    ordered_assignments: &[SplitAssignmentV1],
    canonical_times: &HashMap<String, String>,
) -> Result<String, SplitManifestError> {
    // serde_json::Map is key-sorted unless the preserve_order feature is
    // enabled. The repository workspace uses the default sorted map, matching
    // the independent Python oracle's json.dumps(sort_keys=True).
    let mut root = BTreeMap::<String, Value>::new();
    root.insert("schema".into(), Value::String(INPUT_SCHEMA.to_owned()));
    root.insert(
        "dataset_id".into(),
        Value::String(manifest.dataset_id.clone()),
    );
    root.insert(
        "dataset_snapshot_sha256".into(),
        Value::String(manifest.dataset_snapshot_sha256.clone()),
    );
    root.insert(
        "split_scheme".into(),
        Value::String(manifest.split_scheme.clone()),
    );

    let mut canonical_rows = Vec::with_capacity(ordered_assignments.len());
    for assignment in ordered_assignments {
        let mut row = BTreeMap::<String, Value>::new();
        row.insert(
            "sample_id".into(),
            Value::String(assignment.sample_id.clone()),
        );
        row.insert("split".into(), Value::String(assignment.split.clone()));
        row.insert("positive".into(), Value::Bool(assignment.positive));

        match manifest.split_scheme.as_str() {
            "group_disjoint" => {
                let group_id = assignment
                    .group_id
                    .as_ref()
                    .ok_or_else(|| SplitManifestError::MissingGroup(assignment.sample_id.clone()))?;
                row.insert("group_id".into(), Value::String(group_id.clone()));
            }
            "forward_time" => {
                let timestamp = canonical_times
                    .get(&assignment.sample_id)
                    .ok_or_else(|| {
                        SplitManifestError::InvalidTimestamp(assignment.sample_id.clone())
                    })?;
                row.insert("observed_at".into(), Value::String(timestamp.clone()));
            }
            "stratified_sample" => {}
            other => return Err(SplitManifestError::UnsupportedSplitScheme(other.to_owned())),
        }
        let row_value = serde_json::to_value(row)
            .map_err(|error| SplitManifestError::Canonicalization(error.to_string()))?;
        canonical_rows.push(row_value);
    }
    canonical_rows.sort_by(|left, right| {
        left.get("sample_id")
            .and_then(Value::as_str)
            .cmp(&right.get("sample_id").and_then(Value::as_str))
    });
    root.insert("assignments".into(), Value::Array(canonical_rows));

    if manifest.split_scheme == "forward_time" {
        let time_basis = manifest
            .time_basis
            .as_ref()
            .ok_or(SplitManifestError::InvalidTimeBasis)?;
        root.insert("time_basis".into(), Value::String(time_basis.clone()));
        if time_basis == "source_local_naive" {
            let clock_domain_id = manifest
                .clock_domain_id
                .as_ref()
                .ok_or(SplitManifestError::MissingClockDomain)?;
            root.insert(
                "clock_domain_id".into(),
                Value::String(clock_domain_id.clone()),
            );
        }
    }

    let bytes = serde_json::to_vec(&root)
        .map_err(|error| SplitManifestError::Canonicalization(error.to_string()))?;
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

/// Validate a versioned split manifest and produce a deterministic report.
///
/// The report certifies only properties of the supplied assignment manifest.
/// It does not establish dataset representativeness, process causality or model
/// generalization.
pub fn validate_split_manifest(
    manifest: &SplitManifestV1,
) -> Result<SplitValidationReportV1, SplitManifestError> {
    if manifest.schema != INPUT_SCHEMA {
        return Err(SplitManifestError::UnsupportedSchema(
            manifest.schema.clone(),
        ));
    }
    if !is_nonempty(&manifest.dataset_id) {
        return Err(SplitManifestError::MissingField("dataset_id"));
    }
    if !is_lower_hex_sha256(&manifest.dataset_snapshot_sha256) {
        return Err(SplitManifestError::InvalidDatasetDigest);
    }
    if !matches!(
        manifest.split_scheme.as_str(),
        "group_disjoint" | "forward_time" | "stratified_sample"
    ) {
        return Err(SplitManifestError::UnsupportedSplitScheme(
            manifest.split_scheme.clone(),
        ));
    }

    let time_basis = if manifest.split_scheme == "forward_time" {
        match manifest.time_basis.as_deref() {
            Some("offset_aware") => Some("offset_aware"),
            Some("source_local_naive") => {
                if !manifest
                    .clock_domain_id
                    .as_deref()
                    .is_some_and(is_nonempty)
                {
                    return Err(SplitManifestError::MissingClockDomain);
                }
                Some("source_local_naive")
            }
            _ => return Err(SplitManifestError::InvalidTimeBasis),
        }
    } else {
        None
    };

    if manifest.assignments.is_empty() {
        return Err(SplitManifestError::EmptyRequiredSplit);
    }

    let mut seen = HashMap::<String, ()>::new();
    let mut group_split = HashMap::<String, String>::new();
    let mut counts = BTreeMap::<String, SplitCount>::new();
    for split in REQUIRED_SPLITS {
        counts.insert(split.to_owned(), SplitCount::default());
    }

    let mut timestamps = BTreeMap::<String, Vec<ParsedTimestamp>>::new();
    let mut canonical_times = HashMap::<String, String>::new();

    let mut ordered_assignments = manifest.assignments.clone();
    ordered_assignments.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));

    for assignment in &manifest.assignments {
        if !is_nonempty(&assignment.sample_id) || seen.insert(assignment.sample_id.clone(), ()).is_some() {
            return Err(SplitManifestError::DuplicateSampleIdentity(
                assignment.sample_id.clone(),
            ));
        }
        if !REQUIRED_SPLITS.contains(&assignment.split.as_str()) {
            return Err(SplitManifestError::InvalidSplit {
                sample_id: assignment.sample_id.clone(),
                split: assignment.split.clone(),
            });
        }

        let split_count = counts
            .get_mut(&assignment.split)
            .expect("split was checked against required names");
        split_count.samples += 1;
        if assignment.positive {
            split_count.positive += 1;
        } else {
            split_count.negative += 1;
        }

        if manifest.split_scheme == "group_disjoint" {
            let group_id = assignment
                .group_id
                .as_ref()
                .filter(|value| is_nonempty(value))
                .ok_or_else(|| SplitManifestError::MissingGroup(assignment.sample_id.clone()))?;
            if let Some(previous_split) = group_split.get(group_id) {
                if previous_split != &assignment.split {
                    return Err(SplitManifestError::GroupLeakage {
                        group_id: group_id.clone(),
                    });
                }
            } else {
                group_split.insert(group_id.clone(), assignment.split.clone());
            }
        }

        if let Some(basis) = time_basis {
            let raw = assignment
                .observed_at
                .as_deref()
                .filter(|value| is_nonempty(value))
                .ok_or_else(|| SplitManifestError::InvalidTimestamp(assignment.sample_id.clone()))?;
            let parsed = parse_timestamp(raw, &assignment.sample_id, basis)?;
            canonical_times.insert(assignment.sample_id.clone(), parsed.canonical.clone());
            timestamps
                .entry(assignment.split.clone())
                .or_default()
                .push(parsed);
        }
    }

    if REQUIRED_SPLITS
        .iter()
        .any(|split| counts.get(*split).is_none_or(|count| count.samples == 0))
    {
        return Err(SplitManifestError::EmptyRequiredSplit);
    }

    let temporal_bounds = if let Some(basis) = time_basis {
        let mut bounds = BTreeMap::new();
        let mut split_min = BTreeMap::<String, (i64, u32)>::new();
        let mut split_max = BTreeMap::<String, (i64, u32)>::new();

        for split in REQUIRED_SPLITS {
            let values = timestamps
                .get(split)
                .ok_or(SplitManifestError::EmptyRequiredSplit)?;
            let minimum = values
                .iter()
                .min_by_key(|value| value.order_key)
                .ok_or(SplitManifestError::EmptyRequiredSplit)?;
            let maximum = values
                .iter()
                .max_by_key(|value| value.order_key)
                .ok_or(SplitManifestError::EmptyRequiredSplit)?;
            bounds.insert(
                split.to_owned(),
                TemporalBounds {
                    minimum: minimum.canonical.clone(),
                    maximum: maximum.canonical.clone(),
                },
            );
            split_min.insert(split.to_owned(), minimum.order_key);
            split_max.insert(split.to_owned(), maximum.order_key);
        }

        let train_max = split_max["train"];
        let validation_min = split_min["validation"];
        let validation_max = split_max["validation"];
        let test_min = split_min["test"];
        if !(train_max < validation_min && validation_max < test_min) {
            return Err(SplitManifestError::NonChronologicalWindows);
        }
        let _ = basis;
        Some(bounds)
    } else {
        None
    };

    let split_manifest_sha256 = canonical_manifest_digest(
        manifest,
        &ordered_assignments,
        &canonical_times,
    )?;

    let temporal_independence = match time_basis {
        Some("offset_aware") => "established-for-this-manifest",
        Some("source_local_naive") => {
            "established-within-declared-source-clock-domain-only"
        }
        _ => "not-established",
    };

    Ok(SplitValidationReportV1 {
        schema: OUTPUT_SCHEMA.to_owned(),
        dataset_id: manifest.dataset_id.clone(),
        dataset_snapshot_sha256: manifest.dataset_snapshot_sha256.clone(),
        split_scheme: manifest.split_scheme.clone(),
        split_manifest_sha256,
        split_counts: counts,
        guarantees: Guarantees {
            unique_sample_identity: true,
            all_splits_non_empty: true,
            group_disjoint: manifest.split_scheme == "group_disjoint",
            strict_forward_time: manifest.split_scheme == "forward_time",
        },
        temporal_bounds,
        claim_limits: ClaimLimits {
            group_independence: if manifest.split_scheme == "group_disjoint" {
                "established-for-this-manifest"
            } else {
                "not-established"
            }
            .to_owned(),
            temporal_independence: temporal_independence.to_owned(),
            stratified_sample_split: manifest.split_scheme == "stratified_sample",
        },
        time_basis: time_basis.map(str::to_owned),
        clock_domain_id: if time_basis == Some("source_local_naive") {
            manifest.clock_domain_id.clone()
        } else {
            None
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const GROUP_INPUT: &str =
        include_str!("../../../../../benchmarks/semi-bench-001/reference/fixtures/split-manifest-known-answer-v1.json");
    const GROUP_EXPECTED: &str =
        include_str!("../../../../../benchmarks/semi-bench-001/reference/fixtures/split-manifest-known-answer-v1.expected.json");
    const SECOM_INPUT: &str =
        include_str!("../../../../../docs/research/semi-bench-001/splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json");

    fn read_manifest(raw: &str) -> SplitManifestV1 {
        serde_json::from_str(raw).expect("valid test split manifest")
    }

    #[test]
    fn group_disjoint_report_matches_python_known_answer_exactly() {
        let manifest = read_manifest(GROUP_INPUT);
        let report = validate_split_manifest(&manifest).expect("valid group split");
        let actual = serde_json::to_value(report).expect("serialize report");
        let expected: Value = serde_json::from_str(GROUP_EXPECTED).expect("expected JSON");
        assert_eq!(actual, expected);
    }

    #[test]
    fn frozen_secom_source_clock_manifest_has_exact_support() {
        let manifest = read_manifest(SECOM_INPUT);
        let report = validate_split_manifest(&manifest).expect("valid SECOM source split");

        assert_eq!(manifest.assignments.len(), 1_567);
        assert_eq!(report.time_basis.as_deref(), Some("source_local_naive"));
        assert_eq!(
            report.clock_domain_id.as_deref(),
            Some("uci-secom-mirror-single-source-clock")
        );
        assert_eq!(
            report.split_counts["train"],
            SplitCount {
                samples: 940,
                positive: 76,
                negative: 864
            }
        );
        assert_eq!(
            report.split_counts["validation"],
            SplitCount {
                samples: 313,
                positive: 11,
                negative: 302
            }
        );
        assert_eq!(
            report.split_counts["test"],
            SplitCount {
                samples: 314,
                positive: 17,
                negative: 297
            }
        );
        assert_eq!(
            report.claim_limits.temporal_independence,
            "established-within-declared-source-clock-domain-only"
        );
        assert_eq!(
            report.temporal_bounds.as_ref().unwrap()["train"].minimum,
            "2008-07-19T11:55:00"
        );
        assert_eq!(
            report.temporal_bounds.as_ref().unwrap()["test"].maximum,
            "2008-10-17T06:07:00"
        );
    }

    #[test]
    fn rejects_duplicate_sample_identity() {
        let mut manifest = read_manifest(GROUP_INPUT);
        manifest.assignments[1].sample_id = manifest.assignments[0].sample_id.clone();
        assert!(matches!(
            validate_split_manifest(&manifest),
            Err(SplitManifestError::DuplicateSampleIdentity(_))
        ));
    }

    #[test]
    fn rejects_group_leakage() {
        let mut manifest = read_manifest(GROUP_INPUT);
        let group = manifest.assignments[0].group_id.clone();
        let last = manifest.assignments.len() - 1;
        manifest.assignments[last].group_id = group;
        assert!(matches!(
            validate_split_manifest(&manifest),
            Err(SplitManifestError::GroupLeakage { .. })
        ));
    }

    #[test]
    fn rejects_empty_required_split() {
        let mut manifest = read_manifest(GROUP_INPUT);
        for assignment in &mut manifest.assignments {
            if assignment.split == "test" {
                assignment.split = "train".to_owned();
            }
        }
        assert_eq!(
            validate_split_manifest(&manifest),
            Err(SplitManifestError::EmptyRequiredSplit)
        );
    }

    #[test]
    fn rejects_source_local_timestamp_without_clock_domain() {
        let mut manifest = read_manifest(SECOM_INPUT);
        manifest.clock_domain_id = None;
        assert_eq!(
            validate_split_manifest(&manifest),
            Err(SplitManifestError::MissingClockDomain)
        );
    }

    #[test]
    fn rejects_timezone_offset_in_source_local_clock() {
        let mut manifest = read_manifest(SECOM_INPUT);
        manifest.assignments[0].observed_at = Some("2008-07-19T11:55:00+02:00".to_owned());
        assert!(matches!(
            validate_split_manifest(&manifest),
            Err(SplitManifestError::UnexpectedTimezoneOffset)
        ));
    }

    #[test]
    fn rejects_naive_timestamp_in_offset_aware_mode() {
        let mut value: Value = serde_json::from_str(SECOM_INPUT).expect("SECOM JSON");
        value["time_basis"] = json!("offset_aware");
        value.as_object_mut().unwrap().remove("clock_domain_id");
        let manifest: SplitManifestV1 =
            serde_json::from_value(value).expect("typed manifest without required clock id");
        assert_eq!(
            validate_split_manifest(&manifest),
            Err(SplitManifestError::InvalidTimestamp(
                "secom-mirror-row-0000".to_owned()
            ))
        );
    }
}
