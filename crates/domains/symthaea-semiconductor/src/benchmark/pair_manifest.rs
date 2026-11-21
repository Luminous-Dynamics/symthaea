// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Split-local contrastive-pair provenance validation for SEMI-BENCH-001.

use chrono::{DateTime, FixedOffset, NaiveDateTime};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const INPUT_SCHEMA_V1: &str = "luminous.semi-bench.contrastive-pair-manifest.v1";
pub const OUTPUT_SCHEMA_V1: &str = "luminous.semi-bench.contrastive-pair-validation.v1";
const SPLITS: [&str; 3] = ["train", "validation", "test"];
const BASES: [&str; 4] = ["spatial", "density", "structure", "production_proximity"];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairSampleAssignmentV1 {
    pub sample_id: String,
    pub split: String,
    #[serde(default)]
    pub group_id: Option<String>,
    #[serde(default)]
    pub observed_at: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContrastivePairV1 {
    pub pair_id: String,
    pub left_sample_id: String,
    pub right_sample_id: String,
    pub pair_scope: String,
    pub relation: String,
    pub basis: Vec<String>,
    #[serde(default)]
    pub proximity_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContrastivePairManifestV1 {
    pub schema: String,
    pub dataset_id: String,
    pub split_manifest_sha256: String,
    pub time_basis: String,
    #[serde(default)]
    pub clock_domain_id: Option<String>,
    pub time_embargo_minutes: f64,
    pub sample_assignments: Vec<PairSampleAssignmentV1>,
    pub pairs: Vec<ContrastivePairV1>,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct PairManifestError(pub String);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairGuaranteesV1 {
    pub unique_sample_identity: bool,
    pub unique_pair_identity: bool,
    pub unique_unordered_pair: bool,
    pub no_self_pairs: bool,
    pub all_pair_endpoints_in_declared_scope: bool,
    pub production_proximity_has_supporting_lot_or_time_evidence: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairClaimLimitsV1 {
    pub does_not_establish_causal_production_relationship: bool,
    pub does_not_prove_embedding_quality_or_model_performance: bool,
    pub does_not_prove_pair_semantic_correctness: bool,
    pub pair_manifest_integrity_only: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairValidationReportV1 {
    pub schema: String,
    pub dataset_id: String,
    pub split_manifest_sha256: String,
    pub time_basis: String,
    pub clock_domain_id: Option<String>,
    pub time_embargo_minutes: f64,
    pub sample_count: usize,
    pub pair_count: usize,
    pub pair_counts_by_scope: BTreeMap<String, usize>,
    pub relation_counts: BTreeMap<String, usize>,
    pub basis_counts: BTreeMap<String, usize>,
    pub guarantees: PairGuaranteesV1,
    pub claim_limits: PairClaimLimitsV1,
}

#[derive(Clone, Debug)]
enum PairTime {
    Offset(DateTime<FixedOffset>),
    Naive(NaiveDateTime),
}

impl PairTime {
    fn parse(raw: &str, basis: &str, field: &str) -> Result<Self, PairManifestError> {
        match basis {
            "offset_aware" => DateTime::parse_from_rfc3339(raw)
                .map(Self::Offset)
                .map_err(|_| {
                    PairManifestError(format!("{field}: timezone offset required"))
                }),
            "source_local_naive" => {
                if raw.ends_with('Z') || has_explicit_offset(raw) {
                    return Err(PairManifestError(format!(
                        "{field}: timezone offset forbidden for source_local_naive"
                    )));
                }
                NaiveDateTime::parse_from_str(raw, "%Y-%m-%dT%H:%M:%S%.f")
                    .or_else(|_| NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S%.f"))
                    .map(Self::Naive)
                    .map_err(|_| PairManifestError(format!("{field}: invalid ISO-8601 timestamp")))
            }
            _ => Err(PairManifestError("time_basis: invalid".to_string())),
        }
    }

    fn absolute_minutes(&self, other: &Self) -> Result<f64, PairManifestError> {
        let seconds = match (self, other) {
            (Self::Offset(left), Self::Offset(right)) => {
                (left.timestamp_millis() - right.timestamp_millis()).unsigned_abs() as f64 / 1000.0
            }
            (Self::Naive(left), Self::Naive(right)) => {
                (left.and_utc().timestamp_millis() - right.and_utc().timestamp_millis())
                    .unsigned_abs() as f64 / 1000.0
            }
            _ => return Err(PairManifestError("timestamp clock basis mismatch".to_string())),
        };
        Ok(seconds / 60.0)
    }
}

fn has_explicit_offset(raw: &str) -> bool {
    let Some(time_start) = raw.find('T').or_else(|| raw.find('t')) else {
        return false;
    };
    raw[time_start + 1..].contains('+') || raw[time_start + 1..].rfind('-').is_some()
}

fn require_text(value: &str, field: &str) -> Result<(), PairManifestError> {
    if value.trim().is_empty() {
        Err(PairManifestError(format!(
            "{field}: expected non-empty string"
        )))
    } else {
        Ok(())
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn validate_pair_manifest_v1(
    manifest: &ContrastivePairManifestV1,
) -> Result<PairValidationReportV1, PairManifestError> {
    if manifest.schema != INPUT_SCHEMA_V1 {
        return Err(PairManifestError(
            "schema: unsupported or missing".to_string(),
        ));
    }
    require_text(&manifest.dataset_id, "dataset_id")?;
    require_text(&manifest.split_manifest_sha256, "split_manifest_sha256")?;
    if !valid_sha256(&manifest.split_manifest_sha256) {
        return Err(PairManifestError(
            "split_manifest_sha256: invalid SHA-256".to_string(),
        ));
    }
    if !["offset_aware", "source_local_naive"].contains(&manifest.time_basis.as_str()) {
        return Err(PairManifestError("time_basis: invalid".to_string()));
    }
    let clock_domain = if manifest.time_basis == "source_local_naive" {
        let domain = manifest.clock_domain_id.as_deref().unwrap_or_default();
        require_text(domain, "clock_domain_id")?;
        Some(domain.to_string())
    } else {
        None
    };
    if !manifest.time_embargo_minutes.is_finite() || manifest.time_embargo_minutes < 0.0 {
        return Err(PairManifestError(
            "time_embargo_minutes: expected non-negative number".to_string(),
        ));
    }
    if manifest.sample_assignments.is_empty() {
        return Err(PairManifestError(
            "sample_assignments: expected non-empty array".to_string(),
        ));
    }

    let mut assignments = BTreeMap::new();
    for (index, row) in manifest.sample_assignments.iter().enumerate() {
        require_text(&row.sample_id, &format!("sample_assignments[{index}].sample_id"))?;
        if !SPLITS.contains(&row.split.as_str()) {
            return Err(PairManifestError(format!(
                "sample_assignments[{index}]: invalid split"
            )));
        }
        if assignments.insert(row.sample_id.as_str(), row).is_some() {
            return Err(PairManifestError(format!(
                "sample_assignments[{index}]: duplicate sample identity"
            )));
        }
        if let Some(group) = row.group_id.as_deref() {
            require_text(group, &format!("sample_assignments[{index}].group_id"))?;
        }
    }
    if manifest.pairs.is_empty() {
        return Err(PairManifestError(
            "pairs: expected non-empty array".to_string(),
        ));
    }

    let mut pair_ids = BTreeSet::new();
    let mut edges = BTreeSet::new();
    let mut pair_counts = BTreeMap::from([
        ("test".to_string(), 0usize),
        ("train".to_string(), 0usize),
        ("validation".to_string(), 0usize),
    ]);
    let mut relation_counts = BTreeMap::from([
        ("negative".to_string(), 0usize),
        ("positive".to_string(), 0usize),
    ]);
    let mut basis_counts: BTreeMap<String, usize> = BASES
        .iter()
        .map(|base| ((*base).to_string(), 0usize))
        .collect();

    for (index, pair) in manifest.pairs.iter().enumerate() {
        require_text(&pair.pair_id, &format!("pairs[{index}].pair_id"))?;
        if !pair_ids.insert(pair.pair_id.as_str()) {
            return Err(PairManifestError(format!(
                "pairs[{index}]: duplicate pair identity"
            )));
        }
        require_text(
            &pair.left_sample_id,
            &format!("pairs[{index}].left_sample_id"),
        )?;
        require_text(
            &pair.right_sample_id,
            &format!("pairs[{index}].right_sample_id"),
        )?;
        if pair.left_sample_id == pair.right_sample_id {
            return Err(PairManifestError(format!("pairs[{index}]: self-pair")));
        }
        let left = assignments.get(pair.left_sample_id.as_str()).ok_or_else(|| {
            PairManifestError(format!("pairs[{index}]: orphan sample reference"))
        })?;
        let right = assignments.get(pair.right_sample_id.as_str()).ok_or_else(|| {
            PairManifestError(format!("pairs[{index}]: orphan sample reference"))
        })?;

        let edge = if pair.left_sample_id < pair.right_sample_id {
            (pair.left_sample_id.as_str(), pair.right_sample_id.as_str())
        } else {
            (pair.right_sample_id.as_str(), pair.left_sample_id.as_str())
        };
        if !edges.insert(edge) {
            return Err(PairManifestError(format!(
                "pairs[{index}]: duplicate unordered pair"
            )));
        }
        if !SPLITS.contains(&pair.pair_scope.as_str()) {
            return Err(PairManifestError(format!(
                "pairs[{index}]: invalid pair_scope"
            )));
        }
        if left.split != right.split || left.split != pair.pair_scope {
            return Err(PairManifestError(format!(
                "pairs[{index}]: cross-split pair or pair_scope mismatch"
            )));
        }
        if !["positive", "negative"].contains(&pair.relation.as_str()) {
            return Err(PairManifestError(format!(
                "pairs[{index}]: invalid relation"
            )));
        }
        if pair.basis.is_empty() {
            return Err(PairManifestError(format!("pairs[{index}]: invalid basis")));
        }
        let mut seen_basis = BTreeSet::new();
        for basis in &pair.basis {
            if !BASES.contains(&basis.as_str()) || !seen_basis.insert(basis.as_str()) {
                return Err(PairManifestError(format!("pairs[{index}]: invalid basis")));
            }
        }

        let has_proximity = pair.basis.iter().any(|basis| basis == "production_proximity");
        match (has_proximity, pair.proximity_reason.as_deref()) {
            (true, Some("same_lot")) => {
                if left.group_id.as_deref().unwrap_or_default().is_empty()
                    || left.group_id != right.group_id
                {
                    return Err(PairManifestError(format!(
                        "pairs[{index}]: same_lot requires matching group identity"
                    )));
                }
            }
            (true, Some("time_near")) => {
                let left_time = left.observed_at.as_deref().ok_or_else(|| {
                    PairManifestError(format!("pairs[{index}]: time_near lacks timestamps"))
                })?;
                let right_time = right.observed_at.as_deref().ok_or_else(|| {
                    PairManifestError(format!("pairs[{index}]: time_near lacks timestamps"))
                })?;
                let left_time = PairTime::parse(
                    left_time,
                    &manifest.time_basis,
                    &format!("sample_assignments.{}.observed_at", left.sample_id),
                )?;
                let right_time = PairTime::parse(
                    right_time,
                    &manifest.time_basis,
                    &format!("sample_assignments.{}.observed_at", right.sample_id),
                )?;
                if left_time.absolute_minutes(&right_time)? > manifest.time_embargo_minutes {
                    return Err(PairManifestError(format!(
                        "pairs[{index}]: time_near exceeds embargo"
                    )));
                }
            }
            (true, _) => {
                return Err(PairManifestError(format!(
                    "pairs[{index}]: proximity reason must be same_lot or time_near"
                )));
            }
            (false, Some(_)) => {
                return Err(PairManifestError(format!(
                    "pairs[{index}]: proximity_reason without production_proximity"
                )));
            }
            (false, None) => {}
        }

        *pair_counts.get_mut(&pair.pair_scope).expect("validated scope") += 1;
        *relation_counts
            .get_mut(&pair.relation)
            .expect("validated relation") += 1;
        for basis in &pair.basis {
            *basis_counts.get_mut(basis).expect("validated basis") += 1;
        }
    }

    Ok(PairValidationReportV1 {
        schema: OUTPUT_SCHEMA_V1.to_string(),
        dataset_id: manifest.dataset_id.clone(),
        split_manifest_sha256: manifest.split_manifest_sha256.clone(),
        time_basis: manifest.time_basis.clone(),
        clock_domain_id: clock_domain,
        time_embargo_minutes: manifest.time_embargo_minutes,
        sample_count: assignments.len(),
        pair_count: manifest.pairs.len(),
        pair_counts_by_scope: pair_counts,
        relation_counts,
        basis_counts,
        guarantees: PairGuaranteesV1 {
            unique_sample_identity: true,
            unique_pair_identity: true,
            unique_unordered_pair: true,
            no_self_pairs: true,
            all_pair_endpoints_in_declared_scope: true,
            production_proximity_has_supporting_lot_or_time_evidence: true,
        },
        claim_limits: PairClaimLimitsV1 {
            does_not_establish_causal_production_relationship: true,
            does_not_prove_embedding_quality_or_model_performance: true,
            does_not_prove_pair_semantic_correctness: true,
            pair_manifest_integrity_only: true,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const INPUT: &str = include_str!("../../../../../benchmarks/semi-bench-001/reference/fixtures/contrastive-pair-manifest-known-answer-v1.json");
    const EXPECTED: &str = include_str!("../../../../../benchmarks/semi-bench-001/reference/fixtures/contrastive-pair-manifest-known-answer-v1.expected.json");

    fn known_answer() -> ContrastivePairManifestV1 {
        serde_json::from_str(INPUT).expect("pair fixture must parse")
    }

    #[test]
    fn rust_pair_report_matches_frozen_python_reference() {
        let report = validate_pair_manifest_v1(&known_answer()).expect("pair manifest validates");
        let actual = serde_json::to_value(report).expect("report serializes");
        let expected: Value = serde_json::from_str(EXPECTED).expect("expected parses");
        assert_eq!(actual, expected);
    }

    #[test]
    fn cross_split_pair_is_rejected() {
        let mut manifest = known_answer();
        manifest.pairs[0].right_sample_id = "V01".to_string();
        assert!(
            validate_pair_manifest_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("cross-split pair")
        );
    }

    #[test]
    fn duplicate_unordered_pair_is_rejected() {
        let mut manifest = known_answer();
        let mut duplicate = manifest.pairs[0].clone();
        duplicate.pair_id = "P-TRAIN-02".to_string();
        std::mem::swap(&mut duplicate.left_sample_id, &mut duplicate.right_sample_id);
        manifest.pairs.push(duplicate);
        assert!(
            validate_pair_manifest_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("duplicate unordered pair")
        );
    }

    #[test]
    fn same_lot_requires_matching_group_identity() {
        let mut manifest = known_answer();
        manifest.sample_assignments[1].group_id = Some("LOT-OTHER".to_string());
        assert!(
            validate_pair_manifest_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("same_lot requires matching group identity")
        );
    }

    #[test]
    fn time_near_pair_must_stay_within_embargo() {
        let mut manifest = known_answer();
        manifest.sample_assignments[5].observed_at = Some("2008-07-19T18:00:00".to_string());
        assert!(
            validate_pair_manifest_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("time_near exceeds embargo")
        );
    }

    #[test]
    fn rejects_missing_clock_domain_for_naive_source_time() {
        let mut manifest = known_answer();
        manifest.clock_domain_id = None;
        assert!(
            validate_pair_manifest_v1(&manifest)
                .unwrap_err()
                .to_string()
                .contains("clock_domain_id")
        );
    }
}
