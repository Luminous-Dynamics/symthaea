// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Semantic-anchor and measurement-invariance evidence across cultures and time.
//!
//! An aesthetic dimension can retain the same field name while changing meaning.
//! This module compares independently sampled semantic anchors and fails closed
//! when a metric no longer has enough shared, reliable meaning to support direct
//! comparison.

use crate::{IntegrityError, digest_json, load_json, save_json_atomic};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const SEMANTIC_ANCHOR_SET_SCHEMA_VERSION: u32 = 1;
pub const SEMANTIC_DRIFT_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SemanticContext {
    pub culture: String,
    pub language: String,
    pub population: String,
    pub period_start_unix_ms: u64,
    pub period_end_unix_ms: u64,
}

impl SemanticContext {
    fn validate(&self) -> Result<(), SemanticDriftError> {
        validate_identifier(&self.culture, "culture")?;
        validate_identifier(&self.language, "language")?;
        validate_identifier(&self.population, "population")?;
        if self.period_end_unix_ms <= self.period_start_unix_ms {
            return Err(SemanticDriftError::InvalidPeriod);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticAnchor {
    pub anchor_id: String,
    pub label: String,
    /// Named semantic dimensions in `[-1, 1]`. The names, not vector order,
    /// define the comparison contract.
    pub dimensions: BTreeMap<String, f32>,
    pub sample_count: u64,
    pub reliability: f32,
}

impl SemanticAnchor {
    fn validate(&self) -> Result<(), SemanticDriftError> {
        validate_identifier(&self.anchor_id, "anchor id")?;
        validate_identifier(&self.label, "anchor label")?;
        if self.dimensions.is_empty() {
            return Err(SemanticDriftError::EmptyDimensions(self.anchor_id.clone()));
        }
        if self.sample_count == 0 {
            return Err(SemanticDriftError::ZeroSamples(self.anchor_id.clone()));
        }
        validate_unit(self.reliability, "anchor reliability")?;
        for (dimension, value) in &self.dimensions {
            validate_identifier(dimension, "semantic dimension")?;
            validate_signed_unit(*value, "semantic dimension value")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticAnchorSet {
    pub schema_version: u32,
    pub set_id: String,
    pub metric_id: String,
    pub context: SemanticContext,
    pub anchors: Vec<SemanticAnchor>,
}

impl SemanticAnchorSet {
    pub fn validate(&self) -> Result<(), SemanticDriftError> {
        if self.schema_version == 0 || self.schema_version > SEMANTIC_ANCHOR_SET_SCHEMA_VERSION {
            return Err(SemanticDriftError::UnsupportedAnchorSchema(self.schema_version));
        }
        validate_identifier(&self.set_id, "anchor set id")?;
        validate_identifier(&self.metric_id, "metric id")?;
        self.context.validate()?;
        if self.anchors.len() < 2 {
            return Err(SemanticDriftError::InsufficientAnchors);
        }
        let mut ids = BTreeSet::new();
        for anchor in &self.anchors {
            anchor.validate()?;
            if !ids.insert(anchor.anchor_id.as_str()) {
                return Err(SemanticDriftError::DuplicateAnchor(anchor.anchor_id.clone()));
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, SemanticDriftError> {
        self.validate()?;
        digest_json(self).map_err(SemanticDriftError::Integrity)
    }

    pub fn save(&self, path: &Path) -> Result<(), SemanticDriftError> {
        self.validate()?;
        save_json_atomic::<_, SemanticDriftError>(path, self)
    }

    pub fn load(path: &Path) -> Result<Self, SemanticDriftError> {
        let set: Self = load_json::<_, SemanticDriftError>(path)?;
        set.validate()?;
        Ok(set)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SemanticDriftPolicy {
    pub minimum_shared_anchor_fraction: f32,
    pub minimum_shared_dimension_fraction: f32,
    pub minimum_anchor_reliability: f32,
    pub minimum_samples_per_anchor: u64,
    pub warning_mean_shift: f32,
    pub failure_mean_shift: f32,
    pub warning_maximum_shift: f32,
    pub failure_maximum_shift: f32,
}

impl SemanticDriftPolicy {
    pub const fn production() -> Self {
        Self {
            minimum_shared_anchor_fraction: 0.80,
            minimum_shared_dimension_fraction: 0.80,
            minimum_anchor_reliability: 0.65,
            minimum_samples_per_anchor: 20,
            warning_mean_shift: 0.15,
            failure_mean_shift: 0.30,
            warning_maximum_shift: 0.30,
            failure_maximum_shift: 0.55,
        }
    }

    pub fn validate(&self) -> Result<(), SemanticDriftError> {
        validate_unit(self.minimum_shared_anchor_fraction, "minimum shared anchor fraction")?;
        validate_unit(self.minimum_shared_dimension_fraction, "minimum shared dimension fraction")?;
        validate_unit(self.minimum_anchor_reliability, "minimum anchor reliability")?;
        if self.minimum_samples_per_anchor == 0 {
            return Err(SemanticDriftError::InvalidPolicy(
                "minimum samples per anchor must be positive".to_owned(),
            ));
        }
        validate_nonnegative(self.warning_mean_shift, "warning mean shift")?;
        validate_nonnegative(self.failure_mean_shift, "failure mean shift")?;
        validate_nonnegative(self.warning_maximum_shift, "warning maximum shift")?;
        validate_nonnegative(self.failure_maximum_shift, "failure maximum shift")?;
        if self.warning_mean_shift > self.failure_mean_shift
            || self.warning_maximum_shift > self.failure_maximum_shift
        {
            return Err(SemanticDriftError::InvalidPolicy(
                "warning thresholds must not exceed failure thresholds".to_owned(),
            ));
        }
        Ok(())
    }
}

impl Default for SemanticDriftPolicy {
    fn default() -> Self { Self::production() }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnchorShift {
    pub anchor_id: String,
    pub shared_dimensions: usize,
    pub dimension_union: usize,
    pub mean_absolute_shift: f32,
    pub maximum_absolute_shift: f32,
    pub reliability_weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SemanticDriftFindingCode {
    MetricBinding,
    AnchorCoverage,
    DimensionCoverage,
    SampleSupport,
    Reliability,
    MeanShift,
    MaximumShift,
}

impl SemanticDriftFindingCode {
    pub const ALL: [Self; 7] = [
        Self::MetricBinding,
        Self::AnchorCoverage,
        Self::DimensionCoverage,
        Self::SampleSupport,
        Self::Reliability,
        Self::MeanShift,
        Self::MaximumShift,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticDriftFindingStatus { Pass, Warning, Fail }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticDriftFinding {
    pub code: SemanticDriftFindingCode,
    pub status: SemanticDriftFindingStatus,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticComparability {
    Invariant,
    DriftedButComparable,
    HumanReview,
    NonComparable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticDriftReport {
    pub schema_version: u32,
    pub baseline: SemanticAnchorSet,
    pub candidate: SemanticAnchorSet,
    pub policy: SemanticDriftPolicy,
    pub shared_anchor_fraction: f32,
    pub shared_dimension_fraction: f32,
    pub weighted_mean_shift: f32,
    pub maximum_shift: f32,
    pub shifts: Vec<AnchorShift>,
    pub comparability: SemanticComparability,
    pub findings: Vec<SemanticDriftFinding>,
}

impl SemanticDriftReport {
    pub fn validate(&self) -> Result<(), SemanticDriftError> {
        if self.schema_version == 0 || self.schema_version > SEMANTIC_DRIFT_REPORT_SCHEMA_VERSION {
            return Err(SemanticDriftError::UnsupportedReportSchema(self.schema_version));
        }
        let expected = evaluate_semantic_drift(
            self.baseline.clone(),
            self.candidate.clone(),
            self.policy,
        )?;
        if self != &expected {
            return Err(SemanticDriftError::DerivedReportMismatch);
        }
        Ok(())
    }

    pub fn permits_direct_comparison(&self) -> bool {
        matches!(
            self.comparability,
            SemanticComparability::Invariant | SemanticComparability::DriftedButComparable
        )
    }

    pub fn digest(&self) -> Result<String, SemanticDriftError> {
        self.validate()?;
        digest_json(self).map_err(SemanticDriftError::Integrity)
    }
}

pub fn evaluate_semantic_drift(
    baseline: SemanticAnchorSet,
    candidate: SemanticAnchorSet,
    policy: SemanticDriftPolicy,
) -> Result<SemanticDriftReport, SemanticDriftError> {
    baseline.validate()?;
    candidate.validate()?;
    policy.validate()?;
    if baseline.metric_id != candidate.metric_id {
        return Err(SemanticDriftError::MetricBindingMismatch);
    }
    if baseline.set_id == candidate.set_id {
        return Err(SemanticDriftError::IdenticalAnchorSets);
    }
    let baseline_by_id = baseline
        .anchors
        .iter()
        .map(|anchor| (anchor.anchor_id.as_str(), anchor))
        .collect::<BTreeMap<_, _>>();
    let candidate_by_id = candidate
        .anchors
        .iter()
        .map(|anchor| (anchor.anchor_id.as_str(), anchor))
        .collect::<BTreeMap<_, _>>();
    let union_count = baseline_by_id
        .keys()
        .chain(candidate_by_id.keys())
        .copied()
        .collect::<BTreeSet<_>>()
        .len();
    let shared_ids = baseline_by_id
        .keys()
        .filter(|id| candidate_by_id.contains_key(**id))
        .copied()
        .collect::<Vec<_>>();
    let shared_anchor_fraction = if union_count == 0 { 0.0 } else { shared_ids.len() as f32 / union_count as f32 };
    let mut shifts = Vec::new();
    let mut weighted_shift_sum = 0.0f32;
    let mut total_weight = 0.0f32;
    let mut maximum_shift = 0.0f32;
    let mut shared_dimension_total = 0usize;
    let mut dimension_union_total = 0usize;
    for id in shared_ids {
        let left = baseline_by_id[id];
        let right = candidate_by_id[id];
        let dimension_union = left
            .dimensions
            .keys()
            .chain(right.dimensions.keys())
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let shared = left
            .dimensions
            .keys()
            .filter(|dimension| right.dimensions.contains_key(*dimension))
            .collect::<Vec<_>>();
        let mut sum = 0.0f32;
        let mut local_max = 0.0f32;
        for dimension in &shared {
            let delta = (left.dimensions[*dimension] - right.dimensions[*dimension]).abs();
            sum += delta;
            local_max = local_max.max(delta);
        }
        let mean = if shared.is_empty() { 0.0 } else { sum / shared.len() as f32 };
        let reliability_weight = left.reliability.min(right.reliability)
            * (left.sample_count.min(right.sample_count) as f32).sqrt();
        weighted_shift_sum += mean * reliability_weight;
        total_weight += reliability_weight;
        maximum_shift = maximum_shift.max(local_max);
        shared_dimension_total += shared.len();
        dimension_union_total += dimension_union.len();
        shifts.push(AnchorShift {
            anchor_id: id.to_owned(),
            shared_dimensions: shared.len(),
            dimension_union: dimension_union.len(),
            mean_absolute_shift: mean,
            maximum_absolute_shift: local_max,
            reliability_weight,
        });
    }
    shifts.sort_by(|left, right| left.anchor_id.cmp(&right.anchor_id));
    let shared_dimension_fraction = if dimension_union_total == 0 { 0.0 } else { shared_dimension_total as f32 / dimension_union_total as f32 };
    let weighted_mean_shift = if total_weight <= f32::EPSILON { 0.0 } else { weighted_shift_sum / total_weight };
    let samples_ok = baseline.anchors.iter().chain(candidate.anchors.iter()).all(|anchor| anchor.sample_count >= policy.minimum_samples_per_anchor);
    let reliability_ok = baseline.anchors.iter().chain(candidate.anchors.iter()).all(|anchor| anchor.reliability >= policy.minimum_anchor_reliability);
    let coverage_ok = shared_anchor_fraction >= policy.minimum_shared_anchor_fraction;
    let dimensions_ok = shared_dimension_fraction >= policy.minimum_shared_dimension_fraction;
    let mean_status = threshold_status(weighted_mean_shift, policy.warning_mean_shift, policy.failure_mean_shift);
    let max_status = threshold_status(maximum_shift, policy.warning_maximum_shift, policy.failure_maximum_shift);
    let findings = vec![
        finding(SemanticDriftFindingCode::MetricBinding, SemanticDriftFindingStatus::Pass, format!("metric {} is bound across both anchor sets", baseline.metric_id)),
        finding(SemanticDriftFindingCode::AnchorCoverage, pass_fail(coverage_ok), format!("shared anchor fraction {:.4}", shared_anchor_fraction)),
        finding(SemanticDriftFindingCode::DimensionCoverage, pass_fail(dimensions_ok), format!("shared dimension fraction {:.4}", shared_dimension_fraction)),
        finding(SemanticDriftFindingCode::SampleSupport, pass_fail(samples_ok), format!("minimum samples per anchor is {}", baseline.anchors.iter().chain(candidate.anchors.iter()).map(|anchor| anchor.sample_count).min().unwrap_or(0))),
        finding(SemanticDriftFindingCode::Reliability, pass_fail(reliability_ok), format!("minimum reliability {:.4}", baseline.anchors.iter().chain(candidate.anchors.iter()).map(|anchor| anchor.reliability).fold(1.0f32, f32::min))),
        finding(SemanticDriftFindingCode::MeanShift, mean_status, format!("weighted mean semantic shift {:.4}", weighted_mean_shift)),
        finding(SemanticDriftFindingCode::MaximumShift, max_status, format!("maximum semantic shift {:.4}", maximum_shift)),
    ];
    let fail_count = findings.iter().filter(|finding| finding.status == SemanticDriftFindingStatus::Fail).count();
    let warning_count = findings.iter().filter(|finding| finding.status == SemanticDriftFindingStatus::Warning).count();
    let comparability = if !coverage_ok || !dimensions_ok || fail_count >= 2 {
        SemanticComparability::NonComparable
    } else if fail_count > 0 || !samples_ok || !reliability_ok {
        SemanticComparability::HumanReview
    } else if warning_count > 0 {
        SemanticComparability::DriftedButComparable
    } else {
        SemanticComparability::Invariant
    };
    Ok(SemanticDriftReport {
        schema_version: SEMANTIC_DRIFT_REPORT_SCHEMA_VERSION,
        baseline,
        candidate,
        policy,
        shared_anchor_fraction,
        shared_dimension_fraction,
        weighted_mean_shift,
        maximum_shift,
        shifts,
        comparability,
        findings,
    })
}

fn threshold_status(value: f32, warning: f32, failure: f32) -> SemanticDriftFindingStatus {
    if value >= failure { SemanticDriftFindingStatus::Fail }
    else if value >= warning { SemanticDriftFindingStatus::Warning }
    else { SemanticDriftFindingStatus::Pass }
}
fn pass_fail(pass: bool) -> SemanticDriftFindingStatus {
    if pass { SemanticDriftFindingStatus::Pass } else { SemanticDriftFindingStatus::Fail }
}
fn finding(code: SemanticDriftFindingCode, status: SemanticDriftFindingStatus, detail: String) -> SemanticDriftFinding {
    SemanticDriftFinding { code, status, detail }
}
fn validate_identifier(value: &str, field: &str) -> Result<(), SemanticDriftError> {
    if value.trim().is_empty() { Err(SemanticDriftError::InvalidIdentifier(field.to_owned())) } else { Ok(()) }
}
fn validate_unit(value: f32, field: &str) -> Result<(), SemanticDriftError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) { Err(SemanticDriftError::InvalidNumber(field.to_owned())) } else { Ok(()) }
}
fn validate_signed_unit(value: f32, field: &str) -> Result<(), SemanticDriftError> {
    if !value.is_finite() || !(-1.0..=1.0).contains(&value) { Err(SemanticDriftError::InvalidNumber(field.to_owned())) } else { Ok(()) }
}
fn validate_nonnegative(value: f32, field: &str) -> Result<(), SemanticDriftError> {
    if !value.is_finite() || value < 0.0 { Err(SemanticDriftError::InvalidNumber(field.to_owned())) } else { Ok(()) }
}

#[derive(Debug)]
pub enum SemanticDriftError {
    UnsupportedAnchorSchema(u32),
    UnsupportedReportSchema(u32),
    InvalidIdentifier(String),
    InvalidNumber(String),
    InvalidPolicy(String),
    InvalidPeriod,
    EmptyDimensions(String),
    ZeroSamples(String),
    InsufficientAnchors,
    DuplicateAnchor(String),
    MetricBindingMismatch,
    IdenticalAnchorSets,
    DerivedReportMismatch,
    Integrity(IntegrityError),
    Persistence(String),
}

impl std::fmt::Display for SemanticDriftError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedAnchorSchema(version) => write!(formatter, "unsupported semantic-anchor schema {version}"),
            Self::UnsupportedReportSchema(version) => write!(formatter, "unsupported semantic-drift schema {version}"),
            Self::InvalidIdentifier(field) => write!(formatter, "{field} must not be empty"),
            Self::InvalidNumber(field) => write!(formatter, "{field} is outside its finite range"),
            Self::InvalidPolicy(detail) | Self::Persistence(detail) => formatter.write_str(detail),
            Self::InvalidPeriod => formatter.write_str("semantic context has an invalid period"),
            Self::EmptyDimensions(id) => write!(formatter, "semantic anchor {id} has no dimensions"),
            Self::ZeroSamples(id) => write!(formatter, "semantic anchor {id} has zero samples"),
            Self::InsufficientAnchors => formatter.write_str("semantic anchor set requires at least two anchors"),
            Self::DuplicateAnchor(id) => write!(formatter, "duplicate semantic anchor {id}"),
            Self::MetricBindingMismatch => formatter.write_str("semantic anchor sets refer to different metrics"),
            Self::IdenticalAnchorSets => formatter.write_str("semantic drift requires two distinct anchor sets"),
            Self::DerivedReportMismatch => formatter.write_str("semantic drift report does not match recomputed evidence"),
            Self::Integrity(error) => write!(formatter, "semantic drift integrity failed: {error}"),
        }
    }
}
impl std::error::Error for SemanticDriftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { match self { Self::Integrity(error) => Some(error), _ => None } }
}
impl From<std::io::Error> for SemanticDriftError { fn from(error: std::io::Error) -> Self { Self::Persistence(error.to_string()) } }
impl From<serde_json::Error> for SemanticDriftError { fn from(error: serde_json::Error) -> Self { Self::Persistence(error.to_string()) } }

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor(id: &str, warmth: f32, tension: f32) -> SemanticAnchor {
        SemanticAnchor {
            anchor_id: id.to_owned(), label: id.to_owned(),
            dimensions: BTreeMap::from([("warmth".to_owned(), warmth), ("tension".to_owned(), tension)]),
            sample_count: 100, reliability: 0.9,
        }
    }
    fn set(id: &str, culture: &str, delta: f32) -> SemanticAnchorSet {
        SemanticAnchorSet {
            schema_version: SEMANTIC_ANCHOR_SET_SCHEMA_VERSION,
            set_id: id.to_owned(), metric_id: "metric-harmony".to_owned(),
            context: SemanticContext { culture: culture.to_owned(), language: "en".to_owned(), population: "adult".to_owned(), period_start_unix_ms: 1, period_end_unix_ms: 2 },
            anchors: vec![anchor("calm", 0.4 + delta, -0.4), anchor("radiant", 0.8 + delta, 0.2)],
        }
    }

    #[test]
    fn small_shift_remains_comparable() {
        let report = evaluate_semantic_drift(set("a", "A", 0.0), set("b", "B", 0.05), SemanticDriftPolicy::production()).expect("report");
        assert!(report.permits_direct_comparison());
        report.validate().expect("valid report");
    }

    #[test]
    fn large_shift_blocks_direct_comparison() {
        let report = evaluate_semantic_drift(set("a", "A", 0.0), set("b", "B", -0.8), SemanticDriftPolicy::production()).expect("report");
        assert!(!report.permits_direct_comparison());
    }

    #[test]
    fn substituted_metric_is_rejected() {
        let mut candidate = set("b", "B", 0.0);
        candidate.metric_id = "other".to_owned();
        assert!(matches!(evaluate_semantic_drift(set("a", "A", 0.0), candidate, SemanticDriftPolicy::production()), Err(SemanticDriftError::MetricBindingMismatch)));
    }
}
