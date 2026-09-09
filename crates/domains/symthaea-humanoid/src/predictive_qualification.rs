// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! R4.4 content-addressed predictive qualification evidence.
//!
//! R4.3 answers a narrow modeling question: can one context-bound continuous-
//! time predictor beat persistence for one physical role? R4.4 makes that answer
//! durable and auditable by separating data membership from model fitting and by
//! recording the *measured margin*, not only a boolean test result.
//!
//! The qualified path has an explicit test-set firewall:
//! - a split manifest commits only to sealed case IDs and split/group metadata;
//! - `train_from_split_manifest_v1` accepts **train cases only** and rejects any
//!   validation/test case object passed to the training boundary;
//! - `qualify_test_split_v1` accepts **test cases only** after the checkpoint is
//!   frozen and compares the learned role against the exact R4.0 persistence
//!   baseline.
//!
//! Duplicate case IDs and cross-split group leakage fail closed. Weighted
//! resampling is intentionally not encoded by v1: changing sample multiplicity
//! changes the optimization problem and therefore needs an explicit future
//! evidence schema rather than silent duplicate rows.
//!
//! This module is measurement/qualification evidence only. It grants no control,
//! safety, capability, or motor authority.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fmt::Write as _;

use symthaea_core::hdc::sensorimotor_contingencies::{
    SensorimotorObservationV1, SensorimotorUnitV1,
};

use crate::contextual_cfc::{
    HumanoidContextualCfcErrorV1, HumanoidContextualCfcPredictionV1,
    HumanoidContextualCfcTrainingCaseV1, HumanoidContextualCfcTrainingConfigV1,
    HumanoidRootVelocityXCfcPredictorV1,
};
use crate::continuous_predictor::{
    HumanoidPredictedValueV1, HumanoidPredictorProvenanceV1,
};
use crate::predictive_baseline::{
    evaluate_persistence_baseline_v1, HumanoidPredictionBaselineErrorV1,
};
use crate::predictor_context::{
    HumanoidContextualPredictorInputV1, HumanoidPredictorContextEvidenceV1,
};

const SPLIT_MANIFEST_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.predictive-split-manifest.v1\0";
const EXPERIMENT_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.predictive-qualified-experiment.v1\0";
const QUALIFICATION_REPORT_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.predictive-qualification-report.v1\0";
const SINGLE_SPLIT_GROUPING_POLICY_V1: &str =
    "symthaea.predictive.split-group.single-split.v1";

/// Immutable dataset partition used by qualification evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidPredictiveSplitV1 {
    Train,
    Validation,
    Test,
}

/// One sealed case membership statement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidPredictiveSplitEntryV1 {
    pub case_digest_hex: String,
    pub split: HumanoidPredictiveSplitV1,
    /// Correlated cases may share a group. V1 requires one group to belong to a
    /// single split so a scenario family cannot leak across the partition.
    pub group_id: String,
    /// Human-readable but committed purpose, e.g. `force_interpolation`.
    pub purpose_tag: String,
}

/// Content-addressed train/validation/test membership.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidPredictiveSplitManifestV1 {
    pub schema_id: String,
    pub dataset_profile_id: String,
    pub contextual_input_schema_id: String,
    pub context_schema_id: String,
    pub grouping_policy_id: String,
    /// Canonically ordered by case digest.
    pub entries: Vec<HumanoidPredictiveSplitEntryV1>,
    pub manifest_digest_hex: String,
}

impl HumanoidPredictiveSplitManifestV1 {
    pub const SCHEMA_ID: &'static str =
        "symthaea.humanoid.predictive-split-manifest.v1";

    pub fn new(
        dataset_profile_id: impl Into<String>,
        mut entries: Vec<HumanoidPredictiveSplitEntryV1>,
    ) -> Result<Self, HumanoidPredictiveQualificationErrorV1> {
        entries.sort_by(|left, right| left.case_digest_hex.cmp(&right.case_digest_hex));
        let mut manifest = Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            dataset_profile_id: dataset_profile_id.into(),
            contextual_input_schema_id: HumanoidContextualPredictorInputV1::SCHEMA_ID.to_string(),
            context_schema_id: HumanoidPredictorContextEvidenceV1::SCHEMA_ID.to_string(),
            grouping_policy_id: SINGLE_SPLIT_GROUPING_POLICY_V1.to_string(),
            entries,
            manifest_digest_hex: String::new(),
        };
        manifest.validate_without_digest()?;
        manifest.manifest_digest_hex = manifest.compute_digest_hex()?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), HumanoidPredictiveQualificationErrorV1> {
        self.validate_without_digest()?;
        if self.manifest_digest_hex != self.compute_digest_hex()? {
            return Err(HumanoidPredictiveQualificationErrorV1::ManifestDigestMismatch);
        }
        Ok(())
    }

    fn validate_without_digest(&self) -> Result<(), HumanoidPredictiveQualificationErrorV1> {
        if self.schema_id != Self::SCHEMA_ID
            || self.dataset_profile_id.trim().is_empty()
            || self.contextual_input_schema_id != HumanoidContextualPredictorInputV1::SCHEMA_ID
            || self.context_schema_id != HumanoidPredictorContextEvidenceV1::SCHEMA_ID
            || self.grouping_policy_id != SINGLE_SPLIT_GROUPING_POLICY_V1
            || self.entries.is_empty()
        {
            return Err(HumanoidPredictiveQualificationErrorV1::InvalidManifest);
        }

        let mut previous: Option<&str> = None;
        let mut seen_cases = BTreeSet::new();
        let mut group_split = BTreeMap::new();
        let mut train_count = 0usize;
        let mut test_count = 0usize;

        for entry in &self.entries {
            if !valid_hex_digest(&entry.case_digest_hex)
                || entry.group_id.trim().is_empty()
                || entry.purpose_tag.trim().is_empty()
            {
                return Err(HumanoidPredictiveQualificationErrorV1::InvalidManifestEntry);
            }
            if previous.is_some_and(|prior| prior >= entry.case_digest_hex.as_str()) {
                return Err(HumanoidPredictiveQualificationErrorV1::NonCanonicalManifestOrder);
            }
            previous = Some(&entry.case_digest_hex);
            if !seen_cases.insert(entry.case_digest_hex.as_str()) {
                return Err(HumanoidPredictiveQualificationErrorV1::DuplicateCaseDigest);
            }

            if let Some(existing) = group_split.insert(entry.group_id.as_str(), entry.split)
                && existing != entry.split
            {
                return Err(HumanoidPredictiveQualificationErrorV1::GroupLeakage);
            }

            match entry.split {
                HumanoidPredictiveSplitV1::Train => train_count += 1,
                HumanoidPredictiveSplitV1::Validation => {}
                HumanoidPredictiveSplitV1::Test => test_count += 1,
            }
        }

        if train_count == 0 || test_count == 0 {
            return Err(HumanoidPredictiveQualificationErrorV1::MissingRequiredSplit);
        }
        Ok(())
    }

    pub fn case_digests(&self, split: HumanoidPredictiveSplitV1) -> Vec<String> {
        self.entries
            .iter()
            .filter(|entry| entry.split == split)
            .map(|entry| entry.case_digest_hex.clone())
            .collect()
    }

    fn compute_digest_hex(&self) -> Result<String, HumanoidPredictiveQualificationErrorV1> {
        self.validate_without_digest()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(SPLIT_MANIFEST_DOMAIN_V1);
        feed_str(&mut hasher, &self.schema_id);
        feed_str(&mut hasher, &self.dataset_profile_id);
        feed_str(&mut hasher, &self.contextual_input_schema_id);
        feed_str(&mut hasher, &self.context_schema_id);
        feed_str(&mut hasher, &self.grouping_policy_id);
        hasher.update(&(self.entries.len() as u64).to_le_bytes());
        for entry in &self.entries {
            feed_str(&mut hasher, &entry.case_digest_hex);
            feed_str(&mut hasher, split_token(entry.split));
            feed_str(&mut hasher, &entry.group_id);
            feed_str(&mut hasher, &entry.purpose_tag);
        }
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

/// Frozen R4.3 checkpoint trained through the R4.4 split firewall.
///
/// Test/validation target-bearing case objects are not accepted by the training
/// constructor. Only their committed IDs exist in the manifest at this boundary.
#[derive(Debug, Clone)]
pub struct HumanoidPredictiveExperimentV1 {
    manifest: HumanoidPredictiveSplitManifestV1,
    predictor: HumanoidRootVelocityXCfcPredictorV1,
    train_case_digests: Vec<String>,
    experiment_digest_hex: String,
}

impl HumanoidPredictiveExperimentV1 {
    pub fn manifest(&self) -> &HumanoidPredictiveSplitManifestV1 {
        &self.manifest
    }

    pub fn predictor(&self) -> &HumanoidRootVelocityXCfcPredictorV1 {
        &self.predictor
    }

    pub fn experiment_digest_hex(&self) -> &str {
        &self.experiment_digest_hex
    }

    pub fn train_case_digests(&self) -> &[String] {
        &self.train_case_digests
    }

    pub fn validate(&self) -> Result<(), HumanoidPredictiveQualificationErrorV1> {
        self.manifest.validate()?;
        let expected = self.manifest.case_digests(HumanoidPredictiveSplitV1::Train);
        if self.train_case_digests != expected
            || !valid_hex_digest(&self.experiment_digest_hex)
        {
            return Err(HumanoidPredictiveQualificationErrorV1::ExperimentIdentityMismatch);
        }
        self.predictor.descriptor().validate()?;
        if self.experiment_digest_hex != self.compute_digest_hex()? {
            return Err(HumanoidPredictiveQualificationErrorV1::ExperimentDigestMismatch);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> Result<String, HumanoidPredictiveQualificationErrorV1> {
        self.manifest.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(EXPERIMENT_DOMAIN_V1);
        feed_str(&mut hasher, &self.manifest.manifest_digest_hex);
        feed_str(
            &mut hasher,
            &self.predictor.descriptor().descriptor_digest_hex()?,
        );
        hasher.update(&(self.train_case_digests.len() as u64).to_le_bytes());
        for digest in &self.train_case_digests {
            feed_str(&mut hasher, digest);
        }
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

/// Train the qualified experiment using *only* the manifest's train partition.
pub fn train_from_split_manifest_v1(
    manifest: HumanoidPredictiveSplitManifestV1,
    train_cases: &[HumanoidContextualCfcTrainingCaseV1],
    config: HumanoidContextualCfcTrainingConfigV1,
) -> Result<HumanoidPredictiveExperimentV1, HumanoidPredictiveQualificationErrorV1> {
    manifest.validate()?;
    verify_exact_case_membership(&manifest, HumanoidPredictiveSplitV1::Train, train_cases)?;

    let predictor = HumanoidRootVelocityXCfcPredictorV1::train(train_cases, config)?;
    let train_case_digests = manifest.case_digests(HumanoidPredictiveSplitV1::Train);
    let mut experiment = HumanoidPredictiveExperimentV1 {
        manifest,
        predictor,
        train_case_digests,
        experiment_digest_hex: String::new(),
    };
    experiment.experiment_digest_hex = experiment.compute_digest_hex()?;
    experiment.validate()?;
    Ok(experiment)
}

fn verify_exact_case_membership(
    manifest: &HumanoidPredictiveSplitManifestV1,
    split: HumanoidPredictiveSplitV1,
    cases: &[HumanoidContextualCfcTrainingCaseV1],
) -> Result<(), HumanoidPredictiveQualificationErrorV1> {
    manifest.validate()?;
    let expected: BTreeSet<String> = manifest.case_digests(split).into_iter().collect();
    let mut actual = BTreeSet::new();
    for case in cases {
        case.validate()?;
        if !actual.insert(case.case_digest_hex.clone()) {
            return Err(HumanoidPredictiveQualificationErrorV1::DuplicateProvidedCase);
        }
    }
    if actual != expected {
        return Err(HumanoidPredictiveQualificationErrorV1::SplitMembershipMismatch);
    }
    Ok(())
}

/// Qualification rule is committed to the report so later threshold changes
/// cannot reinterpret old residuals without changing evidence identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidPredictiveQualificationRuleV1 {
    MeanAbsoluteErrorStrictlyBelowPersistence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanoidPredictiveQualificationStateV1 {
    HeldOutEvaluated,
    BeatsPersistence,
}

/// Per-test-case audit trail for one qualified physical role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidPredictiveCaseComparisonV1 {
    pub case_digest_hex: String,
    pub transition_digest_hex: String,
    pub contextual_input_digest_hex: String,
    pub model_prediction_digest_hex: String,
    pub persistence_report_digest_hex: String,
    pub model_predicted_value: f64,
    pub persistence_predicted_value: f64,
    pub actual_value: f64,
    pub model_absolute_error: f64,
    pub persistence_absolute_error: f64,
    pub model_span_normalized_absolute_error: f64,
    pub persistence_span_normalized_absolute_error: f64,
    pub model_outside_contract: bool,
    pub persistence_outside_contract: bool,
    pub actual_outside_contract: bool,
}

/// Frozen, content-addressed model-vs-persistence qualification evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidPredictiveQualificationReportV1 {
    pub schema_id: String,
    pub experiment_digest_hex: String,
    pub predictor_descriptor_digest_hex: String,
    pub split_manifest_digest_hex: String,
    pub output_contract_digest_hex: String,
    pub unit: SensorimotorUnitV1,
    pub contract_min: f64,
    pub contract_max: f64,
    pub contract_bins: u16,
    pub test_case_digests: Vec<String>,
    pub compared_count: usize,
    pub model_mean_absolute_error: f64,
    pub persistence_mean_absolute_error: f64,
    pub model_mean_span_normalized_absolute_error: f64,
    pub persistence_mean_span_normalized_absolute_error: f64,
    /// `persistence_mae - model_mae`; positive is improvement.
    pub absolute_improvement_margin: f64,
    /// `model_mae / persistence_mae`; `< 1` is improvement. `None` when the
    /// persistence denominator is exactly zero.
    pub model_to_persistence_mae_ratio: Option<f64>,
    pub model_worst_absolute_error: f64,
    pub persistence_worst_absolute_error: f64,
    pub model_outside_contract_count: usize,
    pub persistence_outside_contract_count: usize,
    pub actual_outside_contract_count: usize,
    pub rule: HumanoidPredictiveQualificationRuleV1,
    pub state: HumanoidPredictiveQualificationStateV1,
    /// Canonically ordered by case digest.
    pub comparisons: Vec<HumanoidPredictiveCaseComparisonV1>,
    pub report_digest_hex: String,
}

impl HumanoidPredictiveQualificationReportV1 {
    pub const SCHEMA_ID: &'static str =
        "symthaea.humanoid.predictive-qualification-report.v1";

    pub fn validate(&self) -> Result<(), HumanoidPredictiveQualificationErrorV1> {
        if self.schema_id != Self::SCHEMA_ID
            || !valid_hex_digest(&self.experiment_digest_hex)
            || !valid_hex_digest(&self.predictor_descriptor_digest_hex)
            || !valid_hex_digest(&self.split_manifest_digest_hex)
            || !valid_hex_digest(&self.output_contract_digest_hex)
            || self.compared_count == 0
            || self.compared_count != self.comparisons.len()
            || self.test_case_digests.len() != self.comparisons.len()
            || self.contract_max <= self.contract_min
            || !(2..=4096).contains(&self.contract_bins)
        {
            return Err(HumanoidPredictiveQualificationErrorV1::InvalidReport);
        }

        for value in [
            self.model_mean_absolute_error,
            self.persistence_mean_absolute_error,
            self.model_mean_span_normalized_absolute_error,
            self.persistence_mean_span_normalized_absolute_error,
            self.absolute_improvement_margin,
            self.model_worst_absolute_error,
            self.persistence_worst_absolute_error,
        ] {
            if !value.is_finite() {
                return Err(HumanoidPredictiveQualificationErrorV1::NonFiniteMetric);
            }
        }
        if self.model_mean_absolute_error < 0.0
            || self.persistence_mean_absolute_error < 0.0
            || self.model_mean_span_normalized_absolute_error < 0.0
            || self.persistence_mean_span_normalized_absolute_error < 0.0
            || self.model_worst_absolute_error < 0.0
            || self.persistence_worst_absolute_error < 0.0
            || self
                .model_to_persistence_mae_ratio
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(HumanoidPredictiveQualificationErrorV1::NonFiniteMetric);
        }

        let mut previous: Option<&str> = None;
        for (index, comparison) in self.comparisons.iter().enumerate() {
            validate_comparison(comparison, self.contract_min, self.contract_max)?;
            if previous.is_some_and(|prior| prior >= comparison.case_digest_hex.as_str()) {
                return Err(HumanoidPredictiveQualificationErrorV1::NonCanonicalReportOrder);
            }
            previous = Some(&comparison.case_digest_hex);
            if self.test_case_digests[index] != comparison.case_digest_hex {
                return Err(HumanoidPredictiveQualificationErrorV1::ReportSummaryMismatch);
            }
        }

        let count = self.comparisons.len() as f64;
        let model_sum: f64 = self
            .comparisons
            .iter()
            .map(|comparison| comparison.model_absolute_error)
            .sum();
        let persistence_sum: f64 = self
            .comparisons
            .iter()
            .map(|comparison| comparison.persistence_absolute_error)
            .sum();
        let model_span_sum: f64 = self
            .comparisons
            .iter()
            .map(|comparison| comparison.model_span_normalized_absolute_error)
            .sum();
        let persistence_span_sum: f64 = self
            .comparisons
            .iter()
            .map(|comparison| comparison.persistence_span_normalized_absolute_error)
            .sum();
        let model_mae = model_sum / count;
        let persistence_mae = persistence_sum / count;
        let model_span_mae = model_span_sum / count;
        let persistence_span_mae = persistence_span_sum / count;
        let model_worst = self
            .comparisons
            .iter()
            .map(|comparison| comparison.model_absolute_error)
            .fold(0.0_f64, f64::max);
        let persistence_worst = self
            .comparisons
            .iter()
            .map(|comparison| comparison.persistence_absolute_error)
            .fold(0.0_f64, f64::max);
        let margin = persistence_mae - model_mae;
        let ratio = if persistence_mae.to_bits() == 0.0f64.to_bits() {
            None
        } else {
            Some(model_mae / persistence_mae)
        };
        let model_outside = self
            .comparisons
            .iter()
            .filter(|comparison| comparison.model_outside_contract)
            .count();
        let persistence_outside = self
            .comparisons
            .iter()
            .filter(|comparison| comparison.persistence_outside_contract)
            .count();
        let actual_outside = self
            .comparisons
            .iter()
            .filter(|comparison| comparison.actual_outside_contract)
            .count();
        let state = qualification_state(self.rule, model_mae, persistence_mae);

        if self.model_mean_absolute_error.to_bits() != model_mae.to_bits()
            || self.persistence_mean_absolute_error.to_bits() != persistence_mae.to_bits()
            || self.model_mean_span_normalized_absolute_error.to_bits()
                != model_span_mae.to_bits()
            || self.persistence_mean_span_normalized_absolute_error.to_bits()
                != persistence_span_mae.to_bits()
            || self.absolute_improvement_margin.to_bits() != margin.to_bits()
            || !option_f64_bits_equal(self.model_to_persistence_mae_ratio, ratio)
            || self.model_worst_absolute_error.to_bits() != model_worst.to_bits()
            || self.persistence_worst_absolute_error.to_bits() != persistence_worst.to_bits()
            || self.model_outside_contract_count != model_outside
            || self.persistence_outside_contract_count != persistence_outside
            || self.actual_outside_contract_count != actual_outside
            || self.state != state
        {
            return Err(HumanoidPredictiveQualificationErrorV1::ReportSummaryMismatch);
        }

        if self.report_digest_hex != self.compute_digest_hex()? {
            return Err(HumanoidPredictiveQualificationErrorV1::ReportDigestMismatch);
        }
        Ok(())
    }

    pub fn passed(&self) -> bool {
        self.state == HumanoidPredictiveQualificationStateV1::BeatsPersistence
    }

    fn compute_digest_hex(&self) -> Result<String, HumanoidPredictiveQualificationErrorV1> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(QUALIFICATION_REPORT_DOMAIN_V1);
        feed_str(&mut hasher, &self.schema_id);
        feed_str(&mut hasher, &self.experiment_digest_hex);
        feed_str(&mut hasher, &self.predictor_descriptor_digest_hex);
        feed_str(&mut hasher, &self.split_manifest_digest_hex);
        feed_str(&mut hasher, &self.output_contract_digest_hex);
        feed_str(
            &mut hasher,
            &serde_json::to_string(&self.unit)
                .map_err(|_| HumanoidPredictiveQualificationErrorV1::Serialization)?,
        );
        hasher.update(&self.contract_min.to_bits().to_le_bytes());
        hasher.update(&self.contract_max.to_bits().to_le_bytes());
        hasher.update(&self.contract_bins.to_le_bytes());
        hasher.update(&(self.test_case_digests.len() as u64).to_le_bytes());
        for digest in &self.test_case_digests {
            feed_str(&mut hasher, digest);
        }
        hasher.update(&(self.compared_count as u64).to_le_bytes());
        for value in [
            self.model_mean_absolute_error,
            self.persistence_mean_absolute_error,
            self.model_mean_span_normalized_absolute_error,
            self.persistence_mean_span_normalized_absolute_error,
            self.absolute_improvement_margin,
            self.model_worst_absolute_error,
            self.persistence_worst_absolute_error,
        ] {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        feed_option_f64(&mut hasher, self.model_to_persistence_mae_ratio);
        for count in [
            self.model_outside_contract_count,
            self.persistence_outside_contract_count,
            self.actual_outside_contract_count,
        ] {
            hasher.update(&(count as u64).to_le_bytes());
        }
        feed_str(&mut hasher, qualification_rule_token(self.rule));
        feed_str(&mut hasher, qualification_state_token(self.state));
        hasher.update(&(self.comparisons.len() as u64).to_le_bytes());
        for comparison in &self.comparisons {
            feed_comparison(&mut hasher, comparison);
        }
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

/// Evaluate the frozen checkpoint on the manifest's test partition only.
pub fn qualify_test_split_v1(
    experiment: &HumanoidPredictiveExperimentV1,
    test_cases: &[HumanoidContextualCfcTrainingCaseV1],
    rule: HumanoidPredictiveQualificationRuleV1,
) -> Result<HumanoidPredictiveQualificationReportV1, HumanoidPredictiveQualificationErrorV1> {
    experiment.validate()?;
    verify_exact_case_membership(
        &experiment.manifest,
        HumanoidPredictiveSplitV1::Test,
        test_cases,
    )?;

    let descriptor = experiment.predictor.descriptor();
    let output_digest = descriptor.output_role_digest_hex.clone();
    let output_digest_bytes = parse_hex_digest(&output_digest)?;
    let mut comparisons = Vec::with_capacity(test_cases.len());
    let mut contract_identity: Option<(SensorimotorUnitV1, f64, f64, u16)> = None;

    for case in test_cases {
        case.validate()?;
        let model_prediction = experiment.predictor.predict(&case.input)?;
        let model_value = model_predicted_value(&model_prediction, &output_digest_bytes)?;
        let persistence = evaluate_persistence_baseline_v1(&case.transition)?;
        let residual = persistence
            .residuals
            .iter()
            .find(|residual| residual.contract_digest_hex == output_digest)
            .ok_or(HumanoidPredictiveQualificationErrorV1::OutputResidualMissing)?;

        let current_identity = (
            residual.unit.clone(),
            residual.contract_min,
            residual.contract_max,
            residual.contract_bins,
        );
        if let Some((unit, min, max, bins)) = &contract_identity {
            if unit != &current_identity.0
                || min.to_bits() != current_identity.1.to_bits()
                || max.to_bits() != current_identity.2.to_bits()
                || *bins != current_identity.3
            {
                return Err(HumanoidPredictiveQualificationErrorV1::OutputContractChanged);
            }
        } else {
            contract_identity = Some(current_identity.clone());
        }

        let span = residual.contract_span;
        let model_absolute_error = (model_value - residual.actual_value).abs();
        let model_span_normalized_absolute_error = model_absolute_error / span;
        let model_outside_contract =
            model_value < residual.contract_min || model_value > residual.contract_max;

        comparisons.push(HumanoidPredictiveCaseComparisonV1 {
            case_digest_hex: case.case_digest_hex.clone(),
            transition_digest_hex: case.transition.transition_digest_hex.clone(),
            contextual_input_digest_hex: case.input.contextual_input_digest_hex.clone(),
            model_prediction_digest_hex: model_prediction.prediction_digest_hex.clone(),
            persistence_report_digest_hex: persistence.report_digest_hex.clone(),
            model_predicted_value: model_value,
            persistence_predicted_value: residual.predicted_value,
            actual_value: residual.actual_value,
            model_absolute_error,
            persistence_absolute_error: residual.absolute_error,
            model_span_normalized_absolute_error,
            persistence_span_normalized_absolute_error: residual
                .span_normalized_absolute_error,
            model_outside_contract,
            persistence_outside_contract: residual.predicted_outside_contract,
            actual_outside_contract: residual.actual_outside_contract,
        });
    }

    comparisons.sort_by(|left, right| left.case_digest_hex.cmp(&right.case_digest_hex));
    let (unit, contract_min, contract_max, contract_bins) =
        contract_identity.ok_or(HumanoidPredictiveQualificationErrorV1::NoTestCases)?;
    let count = comparisons.len() as f64;
    let model_mean_absolute_error = comparisons
        .iter()
        .map(|comparison| comparison.model_absolute_error)
        .sum::<f64>()
        / count;
    let persistence_mean_absolute_error = comparisons
        .iter()
        .map(|comparison| comparison.persistence_absolute_error)
        .sum::<f64>()
        / count;
    let model_mean_span_normalized_absolute_error = comparisons
        .iter()
        .map(|comparison| comparison.model_span_normalized_absolute_error)
        .sum::<f64>()
        / count;
    let persistence_mean_span_normalized_absolute_error = comparisons
        .iter()
        .map(|comparison| comparison.persistence_span_normalized_absolute_error)
        .sum::<f64>()
        / count;
    let absolute_improvement_margin =
        persistence_mean_absolute_error - model_mean_absolute_error;
    let model_to_persistence_mae_ratio =
        if persistence_mean_absolute_error.to_bits() == 0.0f64.to_bits() {
            None
        } else {
            Some(model_mean_absolute_error / persistence_mean_absolute_error)
        };
    let model_worst_absolute_error = comparisons
        .iter()
        .map(|comparison| comparison.model_absolute_error)
        .fold(0.0_f64, f64::max);
    let persistence_worst_absolute_error = comparisons
        .iter()
        .map(|comparison| comparison.persistence_absolute_error)
        .fold(0.0_f64, f64::max);
    let model_outside_contract_count = comparisons
        .iter()
        .filter(|comparison| comparison.model_outside_contract)
        .count();
    let persistence_outside_contract_count = comparisons
        .iter()
        .filter(|comparison| comparison.persistence_outside_contract)
        .count();
    let actual_outside_contract_count = comparisons
        .iter()
        .filter(|comparison| comparison.actual_outside_contract)
        .count();
    let state = qualification_state(
        rule,
        model_mean_absolute_error,
        persistence_mean_absolute_error,
    );
    let test_case_digests = comparisons
        .iter()
        .map(|comparison| comparison.case_digest_hex.clone())
        .collect();

    let mut report = HumanoidPredictiveQualificationReportV1 {
        schema_id: HumanoidPredictiveQualificationReportV1::SCHEMA_ID.to_string(),
        experiment_digest_hex: experiment.experiment_digest_hex.clone(),
        predictor_descriptor_digest_hex: descriptor.descriptor_digest_hex()?,
        split_manifest_digest_hex: experiment.manifest.manifest_digest_hex.clone(),
        output_contract_digest_hex: output_digest,
        unit,
        contract_min,
        contract_max,
        contract_bins,
        test_case_digests,
        compared_count: comparisons.len(),
        model_mean_absolute_error,
        persistence_mean_absolute_error,
        model_mean_span_normalized_absolute_error,
        persistence_mean_span_normalized_absolute_error,
        absolute_improvement_margin,
        model_to_persistence_mae_ratio,
        model_worst_absolute_error,
        persistence_worst_absolute_error,
        model_outside_contract_count,
        persistence_outside_contract_count,
        actual_outside_contract_count,
        rule,
        state,
        comparisons,
        report_digest_hex: String::new(),
    };
    report.report_digest_hex = report.compute_digest_hex()?;
    report.validate()?;
    Ok(report)
}

fn model_predicted_value(
    prediction: &HumanoidContextualCfcPredictionV1,
    output_digest: &[u8; 32],
) -> Result<f64, HumanoidPredictiveQualificationErrorV1> {
    for value in &prediction.policy_predictions {
        let digest = value
            .address
            .semantic_digest()
            .map_err(HumanoidPredictiveQualificationErrorV1::Sensorimotor)?;
        if &digest == output_digest {
            return match value.prediction {
                HumanoidPredictedValueV1::Predicted(predicted) if predicted.is_finite() => {
                    Ok(predicted)
                }
                _ => Err(HumanoidPredictiveQualificationErrorV1::OutputPredictionUnavailable),
            };
        }
    }
    Err(HumanoidPredictiveQualificationErrorV1::OutputPredictionUnavailable)
}

fn validate_comparison(
    comparison: &HumanoidPredictiveCaseComparisonV1,
    contract_min: f64,
    contract_max: f64,
) -> Result<(), HumanoidPredictiveQualificationErrorV1> {
    if !valid_hex_digest(&comparison.case_digest_hex)
        || !valid_hex_digest(&comparison.transition_digest_hex)
        || !valid_hex_digest(&comparison.contextual_input_digest_hex)
        || !valid_hex_digest(&comparison.model_prediction_digest_hex)
        || !valid_hex_digest(&comparison.persistence_report_digest_hex)
    {
        return Err(HumanoidPredictiveQualificationErrorV1::InvalidComparison);
    }
    for value in [
        comparison.model_predicted_value,
        comparison.persistence_predicted_value,
        comparison.actual_value,
        comparison.model_absolute_error,
        comparison.persistence_absolute_error,
        comparison.model_span_normalized_absolute_error,
        comparison.persistence_span_normalized_absolute_error,
    ] {
        if !value.is_finite() {
            return Err(HumanoidPredictiveQualificationErrorV1::NonFiniteMetric);
        }
    }
    let span = contract_max - contract_min;
    if comparison.model_absolute_error.to_bits()
        != (comparison.model_predicted_value - comparison.actual_value)
            .abs()
            .to_bits()
        || comparison.persistence_absolute_error.to_bits()
            != (comparison.persistence_predicted_value - comparison.actual_value)
                .abs()
                .to_bits()
        || comparison.model_span_normalized_absolute_error.to_bits()
            != (comparison.model_absolute_error / span).to_bits()
        || comparison
            .persistence_span_normalized_absolute_error
            .to_bits()
            != (comparison.persistence_absolute_error / span).to_bits()
        || comparison.model_outside_contract
            != (comparison.model_predicted_value < contract_min
                || comparison.model_predicted_value > contract_max)
        || comparison.persistence_outside_contract
            != (comparison.persistence_predicted_value < contract_min
                || comparison.persistence_predicted_value > contract_max)
        || comparison.actual_outside_contract
            != (comparison.actual_value < contract_min || comparison.actual_value > contract_max)
    {
        return Err(HumanoidPredictiveQualificationErrorV1::ComparisonSummaryMismatch);
    }
    Ok(())
}

fn qualification_state(
    rule: HumanoidPredictiveQualificationRuleV1,
    model_mae: f64,
    persistence_mae: f64,
) -> HumanoidPredictiveQualificationStateV1 {
    match rule {
        HumanoidPredictiveQualificationRuleV1::MeanAbsoluteErrorStrictlyBelowPersistence
            if persistence_mae > 0.0 && model_mae < persistence_mae =>
        {
            HumanoidPredictiveQualificationStateV1::BeatsPersistence
        }
        HumanoidPredictiveQualificationRuleV1::MeanAbsoluteErrorStrictlyBelowPersistence => {
            HumanoidPredictiveQualificationStateV1::HeldOutEvaluated
        }
    }
}

fn feed_comparison(hasher: &mut blake3::Hasher, comparison: &HumanoidPredictiveCaseComparisonV1) {
    feed_str(hasher, &comparison.case_digest_hex);
    feed_str(hasher, &comparison.transition_digest_hex);
    feed_str(hasher, &comparison.contextual_input_digest_hex);
    feed_str(hasher, &comparison.model_prediction_digest_hex);
    feed_str(hasher, &comparison.persistence_report_digest_hex);
    for value in [
        comparison.model_predicted_value,
        comparison.persistence_predicted_value,
        comparison.actual_value,
        comparison.model_absolute_error,
        comparison.persistence_absolute_error,
        comparison.model_span_normalized_absolute_error,
        comparison.persistence_span_normalized_absolute_error,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.update(&[
        u8::from(comparison.model_outside_contract),
        u8::from(comparison.persistence_outside_contract),
        u8::from(comparison.actual_outside_contract),
    ]);
}

fn split_token(split: HumanoidPredictiveSplitV1) -> &'static str {
    match split {
        HumanoidPredictiveSplitV1::Train => "train",
        HumanoidPredictiveSplitV1::Validation => "validation",
        HumanoidPredictiveSplitV1::Test => "test",
    }
}

fn qualification_rule_token(rule: HumanoidPredictiveQualificationRuleV1) -> &'static str {
    match rule {
        HumanoidPredictiveQualificationRuleV1::MeanAbsoluteErrorStrictlyBelowPersistence => {
            "mean_absolute_error_strictly_below_persistence"
        }
    }
}

fn qualification_state_token(state: HumanoidPredictiveQualificationStateV1) -> &'static str {
    match state {
        HumanoidPredictiveQualificationStateV1::HeldOutEvaluated => "held_out_evaluated",
        HumanoidPredictiveQualificationStateV1::BeatsPersistence => "beats_persistence",
    }
}

fn feed_option_f64(hasher: &mut blake3::Hasher, value: Option<f64>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_bits().to_le_bytes());
        }
        None => hasher.update(&[0]),
    }
}

fn option_f64_bits_equal(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left.to_bits() == right.to_bits(),
        (None, None) => true,
        _ => false,
    }
}

fn valid_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn parse_hex_digest(
    value: &str,
) -> Result<[u8; 32], HumanoidPredictiveQualificationErrorV1> {
    if !valid_hex_digest(value) {
        return Err(HumanoidPredictiveQualificationErrorV1::InvalidDigest);
    }
    let mut bytes = [0u8; 32];
    for index in 0..32 {
        bytes[index] = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| HumanoidPredictiveQualificationErrorV1::InvalidDigest)?;
    }
    Ok(bytes)
}

fn feed_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[derive(Debug)]
pub enum HumanoidPredictiveQualificationErrorV1 {
    InvalidManifest,
    InvalidManifestEntry,
    DuplicateCaseDigest,
    NonCanonicalManifestOrder,
    GroupLeakage,
    MissingRequiredSplit,
    ManifestDigestMismatch,
    DuplicateProvidedCase,
    SplitMembershipMismatch,
    ExperimentIdentityMismatch,
    ExperimentDigestMismatch,
    NoTestCases,
    InvalidReport,
    InvalidComparison,
    NonCanonicalReportOrder,
    OutputResidualMissing,
    OutputPredictionUnavailable,
    OutputContractChanged,
    NonFiniteMetric,
    ComparisonSummaryMismatch,
    ReportSummaryMismatch,
    ReportDigestMismatch,
    InvalidDigest,
    Serialization,
    Sensorimotor(&'static str),
    ContextualCfc(HumanoidContextualCfcErrorV1),
    Baseline(HumanoidPredictionBaselineErrorV1),
}

impl From<HumanoidContextualCfcErrorV1> for HumanoidPredictiveQualificationErrorV1 {
    fn from(error: HumanoidContextualCfcErrorV1) -> Self {
        Self::ContextualCfc(error)
    }
}

impl From<HumanoidPredictionBaselineErrorV1> for HumanoidPredictiveQualificationErrorV1 {
    fn from(error: HumanoidPredictionBaselineErrorV1) -> Self {
        Self::Baseline(error)
    }
}

impl fmt::Display for HumanoidPredictiveQualificationErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidManifest => write!(f, "invalid predictive split manifest"),
            Self::InvalidManifestEntry => write!(f, "invalid predictive split entry"),
            Self::DuplicateCaseDigest => write!(f, "duplicate case digest in split manifest"),
            Self::NonCanonicalManifestOrder => write!(f, "split manifest is not canonically ordered"),
            Self::GroupLeakage => write!(f, "correlated case group leaks across splits"),
            Self::MissingRequiredSplit => write!(f, "split manifest requires non-empty train and test partitions"),
            Self::ManifestDigestMismatch => write!(f, "split manifest digest mismatch"),
            Self::DuplicateProvidedCase => write!(f, "duplicate target-bearing case object supplied"),
            Self::SplitMembershipMismatch => write!(f, "provided cases do not exactly match requested split"),
            Self::ExperimentIdentityMismatch => write!(f, "qualified experiment identity mismatch"),
            Self::ExperimentDigestMismatch => write!(f, "qualified experiment digest mismatch"),
            Self::NoTestCases => write!(f, "no test cases supplied"),
            Self::InvalidReport => write!(f, "invalid predictive qualification report"),
            Self::InvalidComparison => write!(f, "invalid predictive case comparison"),
            Self::NonCanonicalReportOrder => write!(f, "qualification comparisons are not canonically ordered"),
            Self::OutputResidualMissing => write!(f, "persistence baseline lacks qualified output residual"),
            Self::OutputPredictionUnavailable => write!(f, "learned output prediction is unavailable"),
            Self::OutputContractChanged => write!(f, "qualified output contract changed across test cases"),
            Self::NonFiniteMetric => write!(f, "qualification metric is non-finite or invalid"),
            Self::ComparisonSummaryMismatch => write!(f, "case comparison residual summary mismatch"),
            Self::ReportSummaryMismatch => write!(f, "qualification report aggregate mismatch"),
            Self::ReportDigestMismatch => write!(f, "qualification report digest mismatch"),
            Self::InvalidDigest => write!(f, "invalid hexadecimal digest"),
            Self::Serialization => write!(f, "qualification evidence serialization failed"),
            Self::Sensorimotor(message) => write!(f, "sensorimotor schema: {message}"),
            Self::ContextualCfc(error) => write!(f, "R4.3 contextual CfC: {error}"),
            Self::Baseline(error) => write!(f, "R4.0 persistence baseline: {error}"),
        }
    }
}

impl std::error::Error for HumanoidPredictiveQualificationErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contextual_cfc::capture_contextual_cfc_training_case_v1;
    use crate::morphology::HumanoidMorphology;
    use crate::predictor_context::ContextInstrumentedSimpleHumanoidSimulator;
    use crate::simulator::HumanoidPhysicsSimulator;
    use crate::types::HumanoidCommand;

    fn zero_command() -> HumanoidCommand {
        HumanoidCommand {
            torques: vec![0.0; HumanoidMorphology::Dmc21.num_actuators()],
        }
    }

    fn case(index: usize, force_x: f64, dt: f64) -> HumanoidContextualCfcTrainingCaseV1 {
        let mut simulator = ContextInstrumentedSimpleHumanoidSimulator::new();
        simulator.reset_with_perturbation(0.0, 41);
        capture_contextual_cfc_training_case_v1(
            &mut simulator,
            &zero_command(),
            dt,
            [force_x, 0.0, 0.0],
            "r4-qualification-clock",
            format!("r4-qualification-case-{index}"),
        )
        .unwrap()
    }

    fn entry(
        case: &HumanoidContextualCfcTrainingCaseV1,
        split: HumanoidPredictiveSplitV1,
        group: impl Into<String>,
    ) -> HumanoidPredictiveSplitEntryV1 {
        HumanoidPredictiveSplitEntryV1 {
            case_digest_hex: case.case_digest_hex.clone(),
            split,
            group_id: group.into(),
            purpose_tag: "force-dt-mechanistic".to_string(),
        }
    }

    fn dataset() -> (
        Vec<HumanoidContextualCfcTrainingCaseV1>,
        Vec<HumanoidContextualCfcTrainingCaseV1>,
        HumanoidPredictiveSplitManifestV1,
    ) {
        let dts = [0.015, 0.025, 0.040];
        let train: Vec<_> = (-12..=12)
            .enumerate()
            .map(|(index, step)| case(index, f64::from(step) * 10.0, dts[index % dts.len()]))
            .collect();
        let held_out = [
            (-87.5, 0.020),
            (-42.5, 0.030),
            (37.5, 0.050),
            (92.5, 0.035),
        ];
        let test: Vec<_> = held_out
            .into_iter()
            .enumerate()
            .map(|(index, (force, dt))| case(100 + index, force, dt))
            .collect();

        let mut entries = Vec::new();
        for (index, case) in train.iter().enumerate() {
            entries.push(entry(
                case,
                HumanoidPredictiveSplitV1::Train,
                format!("train-family-{index}"),
            ));
        }
        for (index, case) in test.iter().enumerate() {
            entries.push(entry(
                case,
                HumanoidPredictiveSplitV1::Test,
                format!("test-family-{index}"),
            ));
        }
        let manifest = HumanoidPredictiveSplitManifestV1::new(
            "symthaea.humanoid.r4.force-dt-mechanistic.v1",
            entries,
        )
        .unwrap();
        (train, test, manifest)
    }

    #[test]
    fn split_manifest_rejects_duplicate_case_membership() {
        let train = case(1, -40.0, 0.025);
        let test = case(2, 40.0, 0.025);
        let mut entries = vec![
            entry(&train, HumanoidPredictiveSplitV1::Train, "train-a"),
            entry(&test, HumanoidPredictiveSplitV1::Test, "test-a"),
        ];
        let mut duplicate = entries[0].clone();
        duplicate.split = HumanoidPredictiveSplitV1::Test;
        duplicate.group_id = "duplicate-test".to_string();
        entries.push(duplicate);
        assert!(matches!(
            HumanoidPredictiveSplitManifestV1::new("duplicate-negative", entries),
            Err(HumanoidPredictiveQualificationErrorV1::NonCanonicalManifestOrder)
                | Err(HumanoidPredictiveQualificationErrorV1::DuplicateCaseDigest)
        ));
    }

    #[test]
    fn split_manifest_rejects_group_leakage() {
        let train = case(3, -30.0, 0.025);
        let test = case(4, 30.0, 0.025);
        let entries = vec![
            entry(&train, HumanoidPredictiveSplitV1::Train, "shared-family"),
            entry(&test, HumanoidPredictiveSplitV1::Test, "shared-family"),
        ];
        assert!(matches!(
            HumanoidPredictiveSplitManifestV1::new("group-negative", entries),
            Err(HumanoidPredictiveQualificationErrorV1::GroupLeakage)
        ));
    }

    #[test]
    fn training_boundary_rejects_test_target_objects() {
        let (train, test, manifest) = dataset();
        let mut contaminated = train.clone();
        contaminated.push(test[0].clone());
        assert!(matches!(
            train_from_split_manifest_v1(
                manifest,
                &contaminated,
                HumanoidContextualCfcTrainingConfigV1::default(),
            ),
            Err(HumanoidPredictiveQualificationErrorV1::SplitMembershipMismatch)
        ));
    }

    #[test]
    fn qualification_report_is_deterministic_and_records_margin() {
        let (train, test, manifest) = dataset();
        let experiment = train_from_split_manifest_v1(
            manifest,
            &train,
            HumanoidContextualCfcTrainingConfigV1::default(),
        )
        .unwrap();
        let rule = HumanoidPredictiveQualificationRuleV1::MeanAbsoluteErrorStrictlyBelowPersistence;
        let left = qualify_test_split_v1(&experiment, &test, rule).unwrap();
        let mut reversed = test.clone();
        reversed.reverse();
        let right = qualify_test_split_v1(&experiment, &reversed, rule).unwrap();

        left.validate().unwrap();
        right.validate().unwrap();
        assert_eq!(left.report_digest_hex, right.report_digest_hex);
        assert_eq!(left.test_case_digests, right.test_case_digests);
        assert!(left.persistence_mean_absolute_error > 0.0);
        assert!(left.model_mean_absolute_error < left.persistence_mean_absolute_error);
        assert!(left.absolute_improvement_margin > 0.0);
        assert!(left.model_to_persistence_mae_ratio.is_some_and(|ratio| ratio < 1.0));
        assert!(left.passed());
    }

    #[test]
    fn report_tampering_fails_closed() {
        let (train, test, manifest) = dataset();
        let experiment = train_from_split_manifest_v1(
            manifest,
            &train,
            HumanoidContextualCfcTrainingConfigV1::default(),
        )
        .unwrap();
        let mut report = qualify_test_split_v1(
            &experiment,
            &test,
            HumanoidPredictiveQualificationRuleV1::MeanAbsoluteErrorStrictlyBelowPersistence,
        )
        .unwrap();
        report.comparisons[0].model_predicted_value += 0.001;
        assert!(report.validate().is_err());
    }
}
