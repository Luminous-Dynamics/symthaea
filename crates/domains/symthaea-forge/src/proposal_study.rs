// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Precommitted study protocol for future Forge proposal-policy estimators.
//!
//! This module freezes the information boundary before any model is fitted. The feature vocabulary
//! contains only proposal-time information; post-decision fields such as selected site, mutation
//! detail, candidate artifact, gate result, benchmark result, terminal outcome, or generation winner
//! are intentionally not representable as features.
//!
//! The study also binds exact training/validation datasets and an identity-only holdout seal,
//! precommits one primary validation metric, and records the estimator implementation/configuration
//! identities without executing an estimator.

use crate::proposal_corpus::{
    ForgeProposalCorpusError, ForgeProposalCorpusManifest, ForgeProposalHoldoutSeal,
    ForgeProposalTrainingSet, ForgeProposalValidationSet,
};
use crate::proposal_endpoints::ForgeProposalEndpoint;
use crate::sequence_stats::{ForgeHistoryOrder, ForgeSequenceStatsError};
use serde::Serialize;
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalStudyError {
    #[error(transparent)]
    Corpus(#[from] ForgeProposalCorpusError),
    #[error(transparent)]
    Sequence(#[from] ForgeSequenceStatsError),
    #[error("proposal study feature schema must contain at least one pre-decision feature")]
    EmptyFeatures,
    #[error("proposal study feature schema contains duplicate features")]
    DuplicateFeature,
    #[error("accepted-history feature requires an explicit history order, and history order is forbidden when that feature is absent")]
    HistoryOrderMismatch,
    #[error("proposal study feature schema identity does not match canonical fields")]
    FeatureSchemaIdentityMismatch,
    #[error("proposal study support thresholds must be non-zero and internally consistent")]
    InvalidSupport,
    #[error("proposal study does not have enough frozen training or validation runs for its support thresholds")]
    InsufficientRuns,
    #[error("proposal study evaluation metrics contain duplicates or repeat the primary metric")]
    InvalidMetrics,
    #[error("proposal study evaluation specification identity does not match canonical fields")]
    EvaluationIdentityMismatch,
    #[error("proposal study estimator identity does not match canonical fields")]
    EstimatorIdentityMismatch,
    #[error("proposal study corpus objects do not share the exact frozen manifest")]
    CorpusScopeMismatch,
    #[error("proposal study identity does not match canonical fields")]
    StudyIdentityMismatch,
}

/// Pre-decision feature vocabulary. The absence of post-decision variants is intentional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeProposalFeature {
    /// Generator-scoped transformation family identity under consideration.
    FamilyIdentity,
    /// Search generation known before a proposal draw is interpreted.
    Generation,
    /// Number of eligible AST sites for this family in the current accepted parent.
    FamilyEligibleSites,
    /// Total eligible `(family, site)` pairs in the current accepted parent.
    TotalEligibleSites,
    /// Bounded suffix of accepted family history entering the generation.
    AcceptedHistorySuffix,
}

impl ForgeProposalFeature {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::FamilyIdentity => b"family-identity",
            Self::Generation => b"generation",
            Self::FamilyEligibleSites => b"family-eligible-sites",
            Self::TotalEligibleSites => b"total-eligible-sites",
            Self::AcceptedHistorySuffix => b"accepted-history-suffix",
        }
    }
}

/// Canonical allowlist of pre-decision features for one study.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalFeatureSchema {
    id: ContentId,
    features: Vec<ForgeProposalFeature>,
    history_order: Option<ForgeHistoryOrder>,
}

impl ForgeProposalFeatureSchema {
    pub fn new(
        mut features: Vec<ForgeProposalFeature>,
        history_order: Option<ForgeHistoryOrder>,
    ) -> Result<Self, ForgeProposalStudyError> {
        if features.is_empty() {
            return Err(ForgeProposalStudyError::EmptyFeatures);
        }
        features.sort();
        if features.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ForgeProposalStudyError::DuplicateFeature);
        }
        let has_history = features.contains(&ForgeProposalFeature::AcceptedHistorySuffix);
        if has_history != history_order.is_some() {
            return Err(ForgeProposalStudyError::HistoryOrderMismatch);
        }
        if let Some(order) = &history_order {
            order.validate()?;
        }
        let id = derive_feature_schema_id(&features, history_order.as_ref());
        Ok(Self {
            id,
            features,
            history_order,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn features(&self) -> &[ForgeProposalFeature] { &self.features }
    pub fn history_order(&self) -> Option<&ForgeHistoryOrder> { self.history_order.as_ref() }

    pub fn validate(&self) -> Result<(), ForgeProposalStudyError> {
        let rebuilt = Self::new(self.features.clone(), self.history_order.clone())?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeProposalStudyError::FeatureSchemaIdentityMismatch)
        }
    }
}

fn derive_feature_schema_id(
    features: &[ForgeProposalFeature],
    history_order: Option<&ForgeHistoryOrder>,
) -> ContentId {
    let count = (features.len() as u64).to_be_bytes();
    let mut parts = vec![count.to_vec()];
    parts.extend(features.iter().map(|feature| feature.tag().to_vec()));
    match history_order {
        Some(order) => {
            parts.push(b"history".to_vec());
            parts.push(order.id().as_str().as_bytes().to_vec());
        }
        None => parts.push(b"no-history".to_vec()),
    }
    ContentId::derive(
        "symthaea.forge-proposal-feature-schema.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Which exact proposal probability is the estimator protocol allowed to reason about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeProposalExposureUnit {
    /// Family selection mass: `family_eligible_sites / total_eligible_sites`.
    FamilySelection,
    /// Exact eligible pair mass: `1 / total_eligible_sites` for the selected site.
    ExactEligiblePair,
}

impl ForgeProposalExposureUnit {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::FamilySelection => b"family-selection",
            Self::ExactEligiblePair => b"exact-eligible-pair",
        }
    }
}

/// Missingness rule for v1 studies. No imputation or negative-label conversion is permitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeProposalMissingnessPolicy {
    /// Only `Observed(true/false)` endpoint records are labels. Censored and counterfactual rows
    /// remain in evidence/exposure accounting but are never relabeled as observed outcomes.
    ObservedLabelsOnlyV1,
}

impl ForgeProposalMissingnessPolicy {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::ObservedLabelsOnlyV1 => b"observed-labels-only-v1",
        }
    }
}

/// Exact implementation/configuration identity for a future estimator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalEstimatorSpec {
    id: ContentId,
    implementation_id: ContentId,
    configuration_id: ContentId,
}

impl ForgeProposalEstimatorSpec {
    pub fn new(implementation_id: ContentId, configuration_id: ContentId) -> Self {
        let id = derive_estimator_id(&implementation_id, &configuration_id);
        Self {
            id,
            implementation_id,
            configuration_id,
        }
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn implementation_id(&self) -> &ContentId { &self.implementation_id }
    pub fn configuration_id(&self) -> &ContentId { &self.configuration_id }

    pub fn validate(&self) -> Result<(), ForgeProposalStudyError> {
        if derive_estimator_id(&self.implementation_id, &self.configuration_id) == self.id {
            Ok(())
        } else {
            Err(ForgeProposalStudyError::EstimatorIdentityMismatch)
        }
    }
}

fn derive_estimator_id(implementation_id: &ContentId, configuration_id: &ContentId) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-estimator-spec.v1",
        [
            implementation_id.as_str().as_bytes(),
            configuration_id.as_str().as_bytes(),
        ],
    )
}

/// Minimum evidence requirements frozen before fitting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalSupportSpec {
    min_training_runs: u64,
    min_validation_runs: u64,
    min_distinct_runs_per_family: u64,
    min_observed_labels_per_family: u64,
    min_positive_labels_per_family: u64,
    min_negative_labels_per_family: u64,
}

impl ForgeProposalSupportSpec {
    pub fn new(
        min_training_runs: u64,
        min_validation_runs: u64,
        min_distinct_runs_per_family: u64,
        min_observed_labels_per_family: u64,
        min_positive_labels_per_family: u64,
        min_negative_labels_per_family: u64,
    ) -> Result<Self, ForgeProposalStudyError> {
        if min_training_runs == 0
            || min_validation_runs == 0
            || min_distinct_runs_per_family == 0
            || min_observed_labels_per_family == 0
            || min_positive_labels_per_family == 0
            || min_negative_labels_per_family == 0
            || min_positive_labels_per_family
                .checked_add(min_negative_labels_per_family)
                .is_none_or(|value| value > min_observed_labels_per_family)
        {
            return Err(ForgeProposalStudyError::InvalidSupport);
        }
        Ok(Self {
            min_training_runs,
            min_validation_runs,
            min_distinct_runs_per_family,
            min_observed_labels_per_family,
            min_positive_labels_per_family,
            min_negative_labels_per_family,
        })
    }

    pub fn min_training_runs(&self) -> u64 { self.min_training_runs }
    pub fn min_validation_runs(&self) -> u64 { self.min_validation_runs }
    pub fn min_distinct_runs_per_family(&self) -> u64 { self.min_distinct_runs_per_family }
    pub fn min_observed_labels_per_family(&self) -> u64 { self.min_observed_labels_per_family }
    pub fn min_positive_labels_per_family(&self) -> u64 { self.min_positive_labels_per_family }
    pub fn min_negative_labels_per_family(&self) -> u64 { self.min_negative_labels_per_family }

    fn parts(&self) -> [[u8; 8]; 6] {
        [
            self.min_training_runs.to_be_bytes(),
            self.min_validation_runs.to_be_bytes(),
            self.min_distinct_runs_per_family.to_be_bytes(),
            self.min_observed_labels_per_family.to_be_bytes(),
            self.min_positive_labels_per_family.to_be_bytes(),
            self.min_negative_labels_per_family.to_be_bytes(),
        ]
    }
}

/// Probabilistic evaluation metric frozen before validation is observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeProposalMetric {
    BrierScore,
    LogLoss,
}

impl ForgeProposalMetric {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::BrierScore => b"brier-score",
            Self::LogLoss => b"log-loss",
        }
    }
}

/// Primary model-selection metric plus optional report-only metrics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalEvaluationSpec {
    id: ContentId,
    primary: ForgeProposalMetric,
    report_only: Vec<ForgeProposalMetric>,
}

impl ForgeProposalEvaluationSpec {
    pub fn new(
        primary: ForgeProposalMetric,
        mut report_only: Vec<ForgeProposalMetric>,
    ) -> Result<Self, ForgeProposalStudyError> {
        report_only.sort();
        if report_only.contains(&primary)
            || report_only.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(ForgeProposalStudyError::InvalidMetrics);
        }
        let id = derive_evaluation_id(primary, &report_only);
        Ok(Self { id, primary, report_only })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn primary(&self) -> ForgeProposalMetric { self.primary }
    pub fn report_only(&self) -> &[ForgeProposalMetric] { &self.report_only }

    pub fn validate(&self) -> Result<(), ForgeProposalStudyError> {
        let rebuilt = Self::new(self.primary, self.report_only.clone())?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeProposalStudyError::EvaluationIdentityMismatch)
        }
    }
}

fn derive_evaluation_id(
    primary: ForgeProposalMetric,
    report_only: &[ForgeProposalMetric],
) -> ContentId {
    let count = (report_only.len() as u64).to_be_bytes();
    let mut parts = vec![primary.tag().to_vec(), count.to_vec()];
    parts.extend(report_only.iter().map(|metric| metric.tag().to_vec()));
    ContentId::derive(
        "symthaea.forge-proposal-evaluation-spec.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Frozen protocol that must exist before any proposal estimator is fitted.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalStudySpec {
    id: ContentId,
    corpus_manifest_id: ContentId,
    training_set_id: ContentId,
    validation_set_id: ContentId,
    holdout_seal_id: ContentId,
    training_run_count: u64,
    validation_run_count: u64,
    endpoint: ForgeProposalEndpoint,
    feature_schema: ForgeProposalFeatureSchema,
    exposure_unit: ForgeProposalExposureUnit,
    missingness_policy: ForgeProposalMissingnessPolicy,
    estimator: ForgeProposalEstimatorSpec,
    training_seed: u64,
    support: ForgeProposalSupportSpec,
    evaluation: ForgeProposalEvaluationSpec,
}

impl ForgeProposalStudySpec {
    #[allow(clippy::too_many_arguments)]
    pub fn freeze(
        manifest: &ForgeProposalCorpusManifest,
        training: &ForgeProposalTrainingSet,
        validation: &ForgeProposalValidationSet,
        holdout: &ForgeProposalHoldoutSeal,
        endpoint: ForgeProposalEndpoint,
        feature_schema: ForgeProposalFeatureSchema,
        exposure_unit: ForgeProposalExposureUnit,
        missingness_policy: ForgeProposalMissingnessPolicy,
        estimator: ForgeProposalEstimatorSpec,
        training_seed: u64,
        support: ForgeProposalSupportSpec,
        evaluation: ForgeProposalEvaluationSpec,
    ) -> Result<Self, ForgeProposalStudyError> {
        manifest.validate()?;
        training.validate_for(manifest)?;
        validation.validate_for(manifest)?;
        holdout.validate_for(manifest)?;
        feature_schema.validate()?;
        estimator.validate()?;
        evaluation.validate()?;

        if training.manifest_id() != manifest.id()
            || validation.manifest_id() != manifest.id()
            || holdout.manifest_id() != manifest.id()
        {
            return Err(ForgeProposalStudyError::CorpusScopeMismatch);
        }

        let training_run_count = u64::try_from(training.tables().len())
            .map_err(|_| ForgeProposalStudyError::InsufficientRuns)?;
        let validation_run_count = u64::try_from(validation.tables().len())
            .map_err(|_| ForgeProposalStudyError::InsufficientRuns)?;
        if training_run_count < support.min_training_runs()
            || validation_run_count < support.min_validation_runs()
            || training_run_count < support.min_distinct_runs_per_family()
        {
            return Err(ForgeProposalStudyError::InsufficientRuns);
        }

        let id = derive_study_id(
            manifest.id(),
            training.id(),
            validation.id(),
            holdout.id(),
            training_run_count,
            validation_run_count,
            endpoint,
            feature_schema.id(),
            exposure_unit,
            missingness_policy,
            estimator.id(),
            training_seed,
            &support,
            evaluation.id(),
        );
        Ok(Self {
            id,
            corpus_manifest_id: manifest.id().clone(),
            training_set_id: training.id().clone(),
            validation_set_id: validation.id().clone(),
            holdout_seal_id: holdout.id().clone(),
            training_run_count,
            validation_run_count,
            endpoint,
            feature_schema,
            exposure_unit,
            missingness_policy,
            estimator,
            training_seed,
            support,
            evaluation,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn corpus_manifest_id(&self) -> &ContentId { &self.corpus_manifest_id }
    pub fn training_set_id(&self) -> &ContentId { &self.training_set_id }
    pub fn validation_set_id(&self) -> &ContentId { &self.validation_set_id }
    pub fn holdout_seal_id(&self) -> &ContentId { &self.holdout_seal_id }
    pub fn endpoint(&self) -> ForgeProposalEndpoint { self.endpoint }
    pub fn feature_schema(&self) -> &ForgeProposalFeatureSchema { &self.feature_schema }
    pub fn exposure_unit(&self) -> ForgeProposalExposureUnit { self.exposure_unit }
    pub fn missingness_policy(&self) -> ForgeProposalMissingnessPolicy { self.missingness_policy }
    pub fn estimator(&self) -> &ForgeProposalEstimatorSpec { &self.estimator }
    pub fn training_seed(&self) -> u64 { self.training_seed }
    pub fn support(&self) -> &ForgeProposalSupportSpec { &self.support }
    pub fn evaluation(&self) -> &ForgeProposalEvaluationSpec { &self.evaluation }

    pub fn validate_for(
        &self,
        manifest: &ForgeProposalCorpusManifest,
        training: &ForgeProposalTrainingSet,
        validation: &ForgeProposalValidationSet,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<(), ForgeProposalStudyError> {
        manifest.validate()?;
        training.validate_for(manifest)?;
        validation.validate_for(manifest)?;
        holdout.validate_for(manifest)?;
        self.feature_schema.validate()?;
        self.estimator.validate()?;
        self.evaluation.validate()?;
        if self.corpus_manifest_id != *manifest.id()
            || self.training_set_id != *training.id()
            || self.validation_set_id != *validation.id()
            || self.holdout_seal_id != *holdout.id()
            || self.training_run_count != training.tables().len() as u64
            || self.validation_run_count != validation.tables().len() as u64
        {
            return Err(ForgeProposalStudyError::CorpusScopeMismatch);
        }
        let expected = derive_study_id(
            &self.corpus_manifest_id,
            &self.training_set_id,
            &self.validation_set_id,
            &self.holdout_seal_id,
            self.training_run_count,
            self.validation_run_count,
            self.endpoint,
            self.feature_schema.id(),
            self.exposure_unit,
            self.missingness_policy,
            self.estimator.id(),
            self.training_seed,
            &self.support,
            self.evaluation.id(),
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalStudyError::StudyIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_study_id(
    manifest_id: &ContentId,
    training_id: &ContentId,
    validation_id: &ContentId,
    holdout_id: &ContentId,
    training_runs: u64,
    validation_runs: u64,
    endpoint: ForgeProposalEndpoint,
    feature_schema_id: &ContentId,
    exposure_unit: ForgeProposalExposureUnit,
    missingness_policy: ForgeProposalMissingnessPolicy,
    estimator_id: &ContentId,
    training_seed: u64,
    support: &ForgeProposalSupportSpec,
    evaluation_id: &ContentId,
) -> ContentId {
    let training_runs = training_runs.to_be_bytes();
    let validation_runs = validation_runs.to_be_bytes();
    let training_seed = training_seed.to_be_bytes();
    let support_parts = support.parts();
    let mut parts = vec![
        manifest_id.as_str().as_bytes().to_vec(),
        training_id.as_str().as_bytes().to_vec(),
        validation_id.as_str().as_bytes().to_vec(),
        holdout_id.as_str().as_bytes().to_vec(),
        training_runs.to_vec(),
        validation_runs.to_vec(),
        endpoint_tag(endpoint).to_vec(),
        feature_schema_id.as_str().as_bytes().to_vec(),
        exposure_unit.tag().to_vec(),
        missingness_policy.tag().to_vec(),
        estimator_id.as_str().as_bytes().to_vec(),
        training_seed.to_vec(),
    ];
    parts.extend(support_parts.iter().map(|part| part.to_vec()));
    parts.push(evaluation_id.as_str().as_bytes().to_vec());
    ContentId::derive(
        "symthaea.forge-proposal-study-spec.v1",
        parts.iter().map(Vec::as_slice),
    )
}

fn endpoint_tag(endpoint: ForgeProposalEndpoint) -> &'static [u8] {
    match endpoint {
        ForgeProposalEndpoint::DistinctCandidate => b"distinct-candidate",
        ForgeProposalEndpoint::CompilePassed => b"compile-passed",
        ForgeProposalEndpoint::CorrectnessPassed => b"correctness-passed",
        ForgeProposalEndpoint::EvaluationValid => b"evaluation-valid",
        ForgeProposalEndpoint::SelectedForContinuation => b"selected-for-continuation",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    #[test]
    fn history_feature_requires_explicit_order() {
        assert!(matches!(
            ForgeProposalFeatureSchema::new(
                vec![ForgeProposalFeature::AcceptedHistorySuffix],
                None,
            ),
            Err(ForgeProposalStudyError::HistoryOrderMismatch)
        ));
        let order = ForgeHistoryOrder::new(2).unwrap();
        let schema = ForgeProposalFeatureSchema::new(
            vec![
                ForgeProposalFeature::FamilyIdentity,
                ForgeProposalFeature::AcceptedHistorySuffix,
            ],
            Some(order),
        )
        .unwrap();
        assert!(schema.validate().is_ok());
    }

    #[test]
    fn history_order_is_forbidden_without_history_feature() {
        assert!(matches!(
            ForgeProposalFeatureSchema::new(
                vec![ForgeProposalFeature::FamilyIdentity],
                Some(ForgeHistoryOrder::new(1).unwrap()),
            ),
            Err(ForgeProposalStudyError::HistoryOrderMismatch)
        ));
    }

    #[test]
    fn evaluation_primary_cannot_be_repeated_as_report_metric() {
        assert!(matches!(
            ForgeProposalEvaluationSpec::new(
                ForgeProposalMetric::BrierScore,
                vec![ForgeProposalMetric::BrierScore],
            ),
            Err(ForgeProposalStudyError::InvalidMetrics)
        ));
    }

    #[test]
    fn support_requires_positive_and_negative_examples_inside_observed_total() {
        assert!(ForgeProposalSupportSpec::new(3, 2, 2, 20, 5, 5).is_ok());
        assert!(ForgeProposalSupportSpec::new(3, 2, 2, 5, 4, 4).is_err());
    }

    #[test]
    fn estimator_identity_binds_code_and_configuration() {
        let a = ForgeProposalEstimatorSpec::new(cid("code", "v1"), cid("config", "a"));
        let b = ForgeProposalEstimatorSpec::new(cid("code", "v1"), cid("config", "b"));
        assert_ne!(a.id(), b.id());
        assert!(a.validate().is_ok());
    }
}
