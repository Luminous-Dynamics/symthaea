// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Observed-label coverage gate for Forge proposal validation.
//!
//! A validation metric is not meaningful merely because a role-isolated validation set exists.
//! The endpoint being scored also needs enough actually observed labels across distinct semantic
//! runs. This module precommits that requirement before scoring, preserves censored and
//! counterfactual rows as explicit evidence, and issues an identity-only score permit only when
//! every frozen transformation family satisfies the declared validation-coverage thresholds.
//!
//! It performs no model scoring and exposes no holdout rows.

use crate::family_learning::{ForgeFamilyLearningError, ForgeTransformationFamilyId};
use crate::proposal_corpus::{
    ForgeProposalCorpusError, ForgeProposalCorpusManifest, ForgeProposalValidationSet,
};
use crate::proposal_endpoints::{
    ForgeProposalEndpointError, ForgeProposalEndpointRecord, ForgeProposalEndpointValue,
};
use crate::proposal_study::ForgeProposalStudySpec;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalValidationCoverageError {
    #[error(transparent)]
    Corpus(#[from] ForgeProposalCorpusError),
    #[error(transparent)]
    Endpoint(#[from] ForgeProposalEndpointError),
    #[error(transparent)]
    Family(#[from] ForgeFamilyLearningError),
    #[error("validation coverage thresholds must be non-zero and internally consistent")]
    InvalidThresholds,
    #[error("validation coverage specification does not bind the supplied study")]
    SpecStudyMismatch,
    #[error("validation coverage specification identity does not match canonical fields")]
    SpecIdentityMismatch,
    #[error("validation coverage accounting overflow")]
    CountOverflow,
    #[error("validation coverage does not cover the exact frozen family set")]
    FamilyCoverageMismatch,
    #[error("validation family coverage identity does not match canonical fields")]
    FamilyIdentityMismatch,
    #[error("validation coverage receipt does not bind the supplied study/validation corpus")]
    ReceiptScopeMismatch,
    #[error("validation coverage receipt identity does not match canonical fields")]
    ReceiptIdentityMismatch,
    #[error("validation scoring is forbidden because frozen coverage requirements are not satisfied")]
    InsufficientCoverage,
    #[error("validation score permit identity or scope does not match the supplied coverage evidence")]
    ScorePermitMismatch,
}

/// Coverage requirements that must be frozen before model validation is scored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalValidationCoverageSpec {
    id: ContentId,
    study_id: ContentId,
    min_distinct_runs_per_family: u64,
    min_observed_labels_per_family: u64,
    min_positive_labels_per_family: u64,
    min_negative_labels_per_family: u64,
}

impl ForgeProposalValidationCoverageSpec {
    pub fn precommit(
        study: &ForgeProposalStudySpec,
        min_distinct_runs_per_family: u64,
        min_observed_labels_per_family: u64,
        min_positive_labels_per_family: u64,
        min_negative_labels_per_family: u64,
    ) -> Result<Self, ForgeProposalValidationCoverageError> {
        if min_distinct_runs_per_family == 0
            || min_observed_labels_per_family == 0
            || min_positive_labels_per_family == 0
            || min_negative_labels_per_family == 0
            || min_positive_labels_per_family
                .checked_add(min_negative_labels_per_family)
                .is_none_or(|value| value > min_observed_labels_per_family)
        {
            return Err(ForgeProposalValidationCoverageError::InvalidThresholds);
        }
        let id = derive_spec_id(
            study.id(),
            min_distinct_runs_per_family,
            min_observed_labels_per_family,
            min_positive_labels_per_family,
            min_negative_labels_per_family,
        );
        Ok(Self {
            id,
            study_id: study.id().clone(),
            min_distinct_runs_per_family,
            min_observed_labels_per_family,
            min_positive_labels_per_family,
            min_negative_labels_per_family,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn study_id(&self) -> &ContentId { &self.study_id }
    pub fn min_distinct_runs_per_family(&self) -> u64 { self.min_distinct_runs_per_family }
    pub fn min_observed_labels_per_family(&self) -> u64 { self.min_observed_labels_per_family }
    pub fn min_positive_labels_per_family(&self) -> u64 { self.min_positive_labels_per_family }
    pub fn min_negative_labels_per_family(&self) -> u64 { self.min_negative_labels_per_family }

    pub fn validate_for(
        &self,
        study: &ForgeProposalStudySpec,
    ) -> Result<(), ForgeProposalValidationCoverageError> {
        if self.study_id != *study.id()
            || self.min_distinct_runs_per_family == 0
            || self.min_observed_labels_per_family == 0
            || self.min_positive_labels_per_family == 0
            || self.min_negative_labels_per_family == 0
            || self
                .min_positive_labels_per_family
                .checked_add(self.min_negative_labels_per_family)
                .is_none_or(|value| value > self.min_observed_labels_per_family)
        {
            return Err(ForgeProposalValidationCoverageError::SpecStudyMismatch);
        }
        let expected = derive_spec_id(
            &self.study_id,
            self.min_distinct_runs_per_family,
            self.min_observed_labels_per_family,
            self.min_positive_labels_per_family,
            self.min_negative_labels_per_family,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalValidationCoverageError::SpecIdentityMismatch)
        }
    }
}

fn derive_spec_id(
    study_id: &ContentId,
    distinct_runs: u64,
    observed: u64,
    positive: u64,
    negative: u64,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-validation-coverage-spec.v1",
        [
            study_id.as_str().as_bytes(),
            distinct_runs.to_be_bytes().as_slice(),
            observed.to_be_bytes().as_slice(),
            positive.to_be_bytes().as_slice(),
            negative.to_be_bytes().as_slice(),
        ],
    )
}

/// Validation coverage observed for one exact generator-scoped transformation family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalValidationFamilyCoverage {
    id: ContentId,
    family_id: ForgeTransformationFamilyId,
    total_rows: u64,
    selected_rows: u64,
    observed_labels: u64,
    positive_labels: u64,
    negative_labels: u64,
    censored_labels: u64,
    counterfactual_unobserved: u64,
    distinct_runs_with_observed_label: u64,
    meets_frozen_coverage: bool,
}

impl ForgeProposalValidationFamilyCoverage {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn family_id(&self) -> &ForgeTransformationFamilyId { &self.family_id }
    pub fn total_rows(&self) -> u64 { self.total_rows }
    pub fn selected_rows(&self) -> u64 { self.selected_rows }
    pub fn observed_labels(&self) -> u64 { self.observed_labels }
    pub fn positive_labels(&self) -> u64 { self.positive_labels }
    pub fn negative_labels(&self) -> u64 { self.negative_labels }
    pub fn censored_labels(&self) -> u64 { self.censored_labels }
    pub fn counterfactual_unobserved(&self) -> u64 { self.counterfactual_unobserved }
    pub fn distinct_runs_with_observed_label(&self) -> u64 {
        self.distinct_runs_with_observed_label
    }
    pub fn meets_frozen_coverage(&self) -> bool { self.meets_frozen_coverage }

    fn validate_for(
        &self,
        spec: &ForgeProposalValidationCoverageSpec,
    ) -> Result<(), ForgeProposalValidationCoverageError> {
        self.family_id.validate()?;
        let selected = self
            .observed_labels
            .checked_add(self.censored_labels)
            .ok_or(ForgeProposalValidationCoverageError::CountOverflow)?;
        let observed_partition = self
            .positive_labels
            .checked_add(self.negative_labels)
            .ok_or(ForgeProposalValidationCoverageError::CountOverflow)?;
        let total = selected
            .checked_add(self.counterfactual_unobserved)
            .ok_or(ForgeProposalValidationCoverageError::CountOverflow)?;
        let meets = family_coverage_satisfied(
            self.distinct_runs_with_observed_label,
            self.observed_labels,
            self.positive_labels,
            self.negative_labels,
            spec,
        );
        if selected != self.selected_rows
            || observed_partition != self.observed_labels
            || total != self.total_rows
            || meets != self.meets_frozen_coverage
        {
            return Err(ForgeProposalValidationCoverageError::FamilyIdentityMismatch);
        }
        let expected = derive_family_coverage_id(
            &self.family_id,
            self.total_rows,
            self.selected_rows,
            self.observed_labels,
            self.positive_labels,
            self.negative_labels,
            self.censored_labels,
            self.counterfactual_unobserved,
            self.distinct_runs_with_observed_label,
            self.meets_frozen_coverage,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalValidationCoverageError::FamilyIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_family_coverage_id(
    family_id: &ForgeTransformationFamilyId,
    total_rows: u64,
    selected_rows: u64,
    observed_labels: u64,
    positive_labels: u64,
    negative_labels: u64,
    censored_labels: u64,
    counterfactual_unobserved: u64,
    distinct_runs_with_observed_label: u64,
    meets: bool,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-validation-family-coverage.v1",
        [
            family_id.as_content_id().as_str().as_bytes(),
            total_rows.to_be_bytes().as_slice(),
            selected_rows.to_be_bytes().as_slice(),
            observed_labels.to_be_bytes().as_slice(),
            positive_labels.to_be_bytes().as_slice(),
            negative_labels.to_be_bytes().as_slice(),
            censored_labels.to_be_bytes().as_slice(),
            counterfactual_unobserved.to_be_bytes().as_slice(),
            distinct_runs_with_observed_label.to_be_bytes().as_slice(),
            if meets { b"covered".as_slice() } else { b"insufficient".as_slice() },
        ],
    )
}

fn family_coverage_satisfied(
    distinct_runs: u64,
    observed: u64,
    positive: u64,
    negative: u64,
    spec: &ForgeProposalValidationCoverageSpec,
) -> bool {
    distinct_runs >= spec.min_distinct_runs_per_family()
        && observed >= spec.min_observed_labels_per_family()
        && positive >= spec.min_positive_labels_per_family()
        && negative >= spec.min_negative_labels_per_family()
}

#[derive(Default)]
struct MutableCoverage {
    total_rows: u64,
    observed_labels: u64,
    positive_labels: u64,
    negative_labels: u64,
    censored_labels: u64,
    counterfactual_unobserved: u64,
    observed_runs: BTreeSet<String>,
}

impl MutableCoverage {
    fn observe(
        &mut self,
        run_id: &ContentId,
        endpoint: &ForgeProposalEndpointRecord,
    ) -> Result<(), ForgeProposalValidationCoverageError> {
        self.total_rows = self
            .total_rows
            .checked_add(1)
            .ok_or(ForgeProposalValidationCoverageError::CountOverflow)?;
        match endpoint.value() {
            ForgeProposalEndpointValue::Observed(value) => {
                self.observed_labels = self
                    .observed_labels
                    .checked_add(1)
                    .ok_or(ForgeProposalValidationCoverageError::CountOverflow)?;
                if value {
                    self.positive_labels = self
                        .positive_labels
                        .checked_add(1)
                        .ok_or(ForgeProposalValidationCoverageError::CountOverflow)?;
                } else {
                    self.negative_labels = self
                        .negative_labels
                        .checked_add(1)
                        .ok_or(ForgeProposalValidationCoverageError::CountOverflow)?;
                }
                self.observed_runs.insert(run_id.as_str().to_string());
            }
            ForgeProposalEndpointValue::Censored => {
                self.censored_labels = self
                    .censored_labels
                    .checked_add(1)
                    .ok_or(ForgeProposalValidationCoverageError::CountOverflow)?;
            }
            ForgeProposalEndpointValue::CounterfactualUnobserved => {
                self.counterfactual_unobserved = self
                    .counterfactual_unobserved
                    .checked_add(1)
                    .ok_or(ForgeProposalValidationCoverageError::CountOverflow)?;
            }
        }
        Ok(())
    }
}

/// Immutable accounting of whether the exact validation set has enough observed endpoint labels.
/// A failing receipt remains useful evidence and cannot mint a score permit.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalValidationCoverageReceipt {
    id: ContentId,
    coverage_spec_id: ContentId,
    study_id: ContentId,
    validation_set_id: ContentId,
    endpoint_tag: &'static str,
    families: Vec<ForgeProposalValidationFamilyCoverage>,
    all_families_covered: bool,
}

impl ForgeProposalValidationCoverageReceipt {
    pub fn evaluate(
        spec: &ForgeProposalValidationCoverageSpec,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        validation: &ForgeProposalValidationSet,
    ) -> Result<Self, ForgeProposalValidationCoverageError> {
        spec.validate_for(study)?;
        manifest.validate()?;
        validation.validate_for(manifest)?;
        if manifest.id() != study.corpus_manifest_id()
            || validation.id() != study.validation_set_id()
            || validation.manifest_id() != manifest.id()
        {
            return Err(ForgeProposalValidationCoverageError::ReceiptScopeMismatch);
        }

        let mut by_family = BTreeMap::<String, MutableCoverage>::new();
        for family in manifest.families() {
            family.validate()?;
            by_family.insert(
                family.as_content_id().as_str().to_string(),
                MutableCoverage::default(),
            );
        }

        for table in validation.tables() {
            table.validate()?;
            for row in table.rows() {
                let key = row.family_id().as_content_id().as_str();
                let coverage = by_family
                    .get_mut(key)
                    .ok_or(ForgeProposalValidationCoverageError::FamilyCoverageMismatch)?;
                let endpoint = ForgeProposalEndpointRecord::from_row(row, study.endpoint())?;
                coverage.observe(table.run_id(), &endpoint)?;
            }
        }

        let mut families = Vec::with_capacity(manifest.families().len());
        for family in manifest.families() {
            let counts = by_family
                .remove(family.as_content_id().as_str())
                .ok_or(ForgeProposalValidationCoverageError::FamilyCoverageMismatch)?;
            let selected_rows = counts
                .observed_labels
                .checked_add(counts.censored_labels)
                .ok_or(ForgeProposalValidationCoverageError::CountOverflow)?;
            let distinct_runs = u64::try_from(counts.observed_runs.len())
                .map_err(|_| ForgeProposalValidationCoverageError::CountOverflow)?;
            let meets = family_coverage_satisfied(
                distinct_runs,
                counts.observed_labels,
                counts.positive_labels,
                counts.negative_labels,
                spec,
            );
            let id = derive_family_coverage_id(
                family,
                counts.total_rows,
                selected_rows,
                counts.observed_labels,
                counts.positive_labels,
                counts.negative_labels,
                counts.censored_labels,
                counts.counterfactual_unobserved,
                distinct_runs,
                meets,
            );
            let row = ForgeProposalValidationFamilyCoverage {
                id,
                family_id: family.clone(),
                total_rows: counts.total_rows,
                selected_rows,
                observed_labels: counts.observed_labels,
                positive_labels: counts.positive_labels,
                negative_labels: counts.negative_labels,
                censored_labels: counts.censored_labels,
                counterfactual_unobserved: counts.counterfactual_unobserved,
                distinct_runs_with_observed_label: distinct_runs,
                meets_frozen_coverage: meets,
            };
            row.validate_for(spec)?;
            families.push(row);
        }
        if !by_family.is_empty() || families.len() != manifest.families().len() {
            return Err(ForgeProposalValidationCoverageError::FamilyCoverageMismatch);
        }

        let all_families_covered = families
            .iter()
            .all(ForgeProposalValidationFamilyCoverage::meets_frozen_coverage);
        let endpoint_tag = endpoint_tag(study.endpoint());
        let id = derive_receipt_id(
            spec.id(),
            study.id(),
            validation.id(),
            endpoint_tag,
            &families,
            all_families_covered,
        );
        let receipt = Self {
            id,
            coverage_spec_id: spec.id().clone(),
            study_id: study.id().clone(),
            validation_set_id: validation.id().clone(),
            endpoint_tag,
            families,
            all_families_covered,
        };
        receipt.validate_for(spec, manifest, study, validation)?;
        Ok(receipt)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn coverage_spec_id(&self) -> &ContentId { &self.coverage_spec_id }
    pub fn study_id(&self) -> &ContentId { &self.study_id }
    pub fn validation_set_id(&self) -> &ContentId { &self.validation_set_id }
    pub fn families(&self) -> &[ForgeProposalValidationFamilyCoverage] { &self.families }
    pub fn all_families_covered(&self) -> bool { self.all_families_covered }
    pub fn uncovered_families(
        &self,
    ) -> impl Iterator<Item = &ForgeProposalValidationFamilyCoverage> {
        self.families.iter().filter(|row| !row.meets_frozen_coverage())
    }

    pub fn validate_for(
        &self,
        spec: &ForgeProposalValidationCoverageSpec,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        validation: &ForgeProposalValidationSet,
    ) -> Result<(), ForgeProposalValidationCoverageError> {
        spec.validate_for(study)?;
        manifest.validate()?;
        validation.validate_for(manifest)?;
        if self.coverage_spec_id != *spec.id()
            || self.study_id != *study.id()
            || self.validation_set_id != *validation.id()
            || self.endpoint_tag != endpoint_tag(study.endpoint())
            || manifest.id() != study.corpus_manifest_id()
            || validation.manifest_id() != manifest.id()
            || self.families.len() != manifest.families().len()
        {
            return Err(ForgeProposalValidationCoverageError::ReceiptScopeMismatch);
        }
        for (row, family) in self.families.iter().zip(manifest.families()) {
            if row.family_id() != family {
                return Err(ForgeProposalValidationCoverageError::FamilyCoverageMismatch);
            }
            row.validate_for(spec)?;
        }
        let all_covered = self
            .families
            .iter()
            .all(ForgeProposalValidationFamilyCoverage::meets_frozen_coverage);
        if all_covered != self.all_families_covered {
            return Err(ForgeProposalValidationCoverageError::ReceiptIdentityMismatch);
        }
        let expected = derive_receipt_id(
            &self.coverage_spec_id,
            &self.study_id,
            &self.validation_set_id,
            self.endpoint_tag,
            &self.families,
            self.all_families_covered,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalValidationCoverageError::ReceiptIdentityMismatch)
        }
    }
}

fn derive_receipt_id(
    spec_id: &ContentId,
    study_id: &ContentId,
    validation_set_id: &ContentId,
    endpoint_tag: &str,
    families: &[ForgeProposalValidationFamilyCoverage],
    all_covered: bool,
) -> ContentId {
    let count = (families.len() as u64).to_be_bytes();
    let mut parts = vec![
        spec_id.as_str().as_bytes().to_vec(),
        study_id.as_str().as_bytes().to_vec(),
        validation_set_id.as_str().as_bytes().to_vec(),
        endpoint_tag.as_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        families
            .iter()
            .map(|row| row.id().as_str().as_bytes().to_vec()),
    );
    parts.push(if all_covered { b"all-covered".to_vec() } else { b"insufficient".to_vec() });
    ContentId::derive(
        "symthaea.forge-proposal-validation-coverage-receipt.v1",
        parts.iter().map(Vec::as_slice),
    )
}

fn endpoint_tag(endpoint: crate::proposal_endpoints::ForgeProposalEndpoint) -> &'static str {
    use crate::proposal_endpoints::ForgeProposalEndpoint::*;
    match endpoint {
        DistinctCandidate => "distinct-candidate",
        CompilePassed => "compile-passed",
        CorrectnessPassed => "correctness-passed",
        EvaluationValid => "evaluation-valid",
        SelectedForContinuation => "selected-for-continuation",
    }
}

/// Identity-only permission to score one exact validation set after coverage has been proven.
/// It exposes no validation rows and performs no scoring.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalValidationScorePermit {
    id: ContentId,
    coverage_spec_id: ContentId,
    study_id: ContentId,
    validation_set_id: ContentId,
    coverage_receipt_id: ContentId,
}

impl ForgeProposalValidationScorePermit {
    pub fn issue(
        spec: &ForgeProposalValidationCoverageSpec,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        validation: &ForgeProposalValidationSet,
        receipt: &ForgeProposalValidationCoverageReceipt,
    ) -> Result<Self, ForgeProposalValidationCoverageError> {
        receipt.validate_for(spec, manifest, study, validation)?;
        if !receipt.all_families_covered() {
            return Err(ForgeProposalValidationCoverageError::InsufficientCoverage);
        }
        let id = derive_score_permit_id(spec.id(), study.id(), validation.id(), receipt.id());
        Ok(Self {
            id,
            coverage_spec_id: spec.id().clone(),
            study_id: study.id().clone(),
            validation_set_id: validation.id().clone(),
            coverage_receipt_id: receipt.id().clone(),
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn coverage_spec_id(&self) -> &ContentId { &self.coverage_spec_id }
    pub fn study_id(&self) -> &ContentId { &self.study_id }
    pub fn validation_set_id(&self) -> &ContentId { &self.validation_set_id }
    pub fn coverage_receipt_id(&self) -> &ContentId { &self.coverage_receipt_id }

    pub fn validate_for(
        &self,
        spec: &ForgeProposalValidationCoverageSpec,
        receipt: &ForgeProposalValidationCoverageReceipt,
    ) -> Result<(), ForgeProposalValidationCoverageError> {
        if !receipt.all_families_covered()
            || self.coverage_spec_id != *spec.id()
            || self.study_id != *spec.study_id()
            || self.validation_set_id != *receipt.validation_set_id()
            || self.coverage_receipt_id != *receipt.id()
            || derive_score_permit_id(
                &self.coverage_spec_id,
                &self.study_id,
                &self.validation_set_id,
                &self.coverage_receipt_id,
            ) != self.id
        {
            return Err(ForgeProposalValidationCoverageError::ScorePermitMismatch);
        }
        Ok(())
    }
}

fn derive_score_permit_id(
    spec_id: &ContentId,
    study_id: &ContentId,
    validation_set_id: &ContentId,
    receipt_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-validation-score-permit.v1",
        [
            spec_id.as_str().as_bytes(),
            study_id.as_str().as_bytes(),
            validation_set_id.as_str().as_bytes(),
            receipt_id.as_str().as_bytes(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_coverage_requires_every_frozen_threshold() {
        let study_id = ContentId::derive("study", [b"study".as_slice()]);
        let spec = ForgeProposalValidationCoverageSpec {
            id: derive_spec_id(&study_id, 2, 20, 5, 5),
            study_id,
            min_distinct_runs_per_family: 2,
            min_observed_labels_per_family: 20,
            min_positive_labels_per_family: 5,
            min_negative_labels_per_family: 5,
        };
        assert!(family_coverage_satisfied(2, 20, 5, 15, &spec));
        assert!(!family_coverage_satisfied(1, 20, 5, 15, &spec));
        assert!(!family_coverage_satisfied(2, 19, 5, 14, &spec));
        assert!(!family_coverage_satisfied(2, 20, 4, 16, &spec));
        assert!(!family_coverage_satisfied(2, 20, 15, 4, &spec));
    }
}
