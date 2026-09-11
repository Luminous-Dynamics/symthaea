// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Training-support accounting and fit-permit boundary for Forge proposal studies.
//!
//! A frozen study declares minimum evidence requirements before fitting. This module checks those
//! requirements against the role-isolated training corpus without imputing censored or
//! counterfactual outcomes. Insufficient support is preserved as evidence; it is not interpreted as
//! poor transformation quality. Only a fully supported receipt can mint a fit permit, and the fit
//! permit also binds the validation-coverage specification that must already exist before fitting.

use crate::family_learning::{ForgeFamilyLearningError, ForgeTransformationFamilyId};
use crate::proposal_corpus::{
    ForgeProposalCorpusError, ForgeProposalCorpusManifest, ForgeProposalHoldoutSeal,
    ForgeProposalTrainingSet, ForgeProposalValidationSet,
};
use crate::proposal_endpoints::{
    ForgeProposalEndpointError, ForgeProposalEndpointRecord, ForgeProposalEndpointValue,
};
use crate::proposal_study::{ForgeProposalStudyError, ForgeProposalStudySpec, ForgeProposalSupportSpec};
use crate::proposal_validation_coverage::{
    ForgeProposalValidationCoverageError, ForgeProposalValidationCoverageSpec,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalSupportError {
    #[error(transparent)]
    Corpus(#[from] ForgeProposalCorpusError),
    #[error(transparent)]
    Study(#[from] ForgeProposalStudyError),
    #[error(transparent)]
    Endpoint(#[from] ForgeProposalEndpointError),
    #[error(transparent)]
    Family(#[from] ForgeFamilyLearningError),
    #[error(transparent)]
    ValidationCoverage(#[from] ForgeProposalValidationCoverageError),
    #[error("proposal support accounting overflow")]
    CountOverflow,
    #[error("proposal support receipt does not cover the study's exact family set")]
    FamilyCoverageMismatch,
    #[error("proposal support row identity does not match canonical fields")]
    RowIdentityMismatch,
    #[error("proposal support receipt identity does not match canonical fields")]
    ReceiptIdentityMismatch,
    #[error("proposal support receipt does not bind the supplied study/training corpus")]
    ReceiptScopeMismatch,
    #[error("proposal model fitting is forbidden because frozen minimum support is not satisfied")]
    InsufficientSupport,
    #[error("proposal fit permit identity or scope does not match the supplied study/support/validation-coverage precommitment")]
    FitPermitMismatch,
}

/// Observed training support for one exact generator-scoped transformation family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalFamilySupport {
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
    meets_frozen_support: bool,
}

impl ForgeProposalFamilySupport {
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
    pub fn meets_frozen_support(&self) -> bool { self.meets_frozen_support }

    fn validate_for(
        &self,
        support: &ForgeProposalSupportSpec,
    ) -> Result<(), ForgeProposalSupportError> {
        self.family_id.validate()?;
        let selected = self
            .observed_labels
            .checked_add(self.censored_labels)
            .ok_or(ForgeProposalSupportError::CountOverflow)?;
        let observed_partition = self
            .positive_labels
            .checked_add(self.negative_labels)
            .ok_or(ForgeProposalSupportError::CountOverflow)?;
        let total = selected
            .checked_add(self.counterfactual_unobserved)
            .ok_or(ForgeProposalSupportError::CountOverflow)?;
        let meets = family_support_satisfied(
            self.distinct_runs_with_observed_label,
            self.observed_labels,
            self.positive_labels,
            self.negative_labels,
            support,
        );
        if selected != self.selected_rows
            || observed_partition != self.observed_labels
            || total != self.total_rows
            || meets != self.meets_frozen_support
        {
            return Err(ForgeProposalSupportError::RowIdentityMismatch);
        }
        let expected = derive_family_support_id(
            &self.family_id,
            self.total_rows,
            self.selected_rows,
            self.observed_labels,
            self.positive_labels,
            self.negative_labels,
            self.censored_labels,
            self.counterfactual_unobserved,
            self.distinct_runs_with_observed_label,
            self.meets_frozen_support,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalSupportError::RowIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_family_support_id(
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
        "symthaea.forge-proposal-family-support.v1",
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
            if meets { b"supported".as_slice() } else { b"insufficient".as_slice() },
        ],
    )
}

fn family_support_satisfied(
    distinct_runs: u64,
    observed: u64,
    positive: u64,
    negative: u64,
    support: &ForgeProposalSupportSpec,
) -> bool {
    distinct_runs >= support.min_distinct_runs_per_family()
        && observed >= support.min_observed_labels_per_family()
        && positive >= support.min_positive_labels_per_family()
        && negative >= support.min_negative_labels_per_family()
}

#[derive(Default)]
struct MutableSupport {
    total_rows: u64,
    observed_labels: u64,
    positive_labels: u64,
    negative_labels: u64,
    censored_labels: u64,
    counterfactual_unobserved: u64,
    observed_runs: BTreeSet<String>,
}

impl MutableSupport {
    fn observe(
        &mut self,
        run_id: &ContentId,
        endpoint: &ForgeProposalEndpointRecord,
    ) -> Result<(), ForgeProposalSupportError> {
        self.total_rows = self
            .total_rows
            .checked_add(1)
            .ok_or(ForgeProposalSupportError::CountOverflow)?;
        match endpoint.value() {
            ForgeProposalEndpointValue::Observed(value) => {
                self.observed_labels = self
                    .observed_labels
                    .checked_add(1)
                    .ok_or(ForgeProposalSupportError::CountOverflow)?;
                if value {
                    self.positive_labels = self
                        .positive_labels
                        .checked_add(1)
                        .ok_or(ForgeProposalSupportError::CountOverflow)?;
                } else {
                    self.negative_labels = self
                        .negative_labels
                        .checked_add(1)
                        .ok_or(ForgeProposalSupportError::CountOverflow)?;
                }
                self.observed_runs.insert(run_id.as_str().to_string());
            }
            ForgeProposalEndpointValue::Censored => {
                self.censored_labels = self
                    .censored_labels
                    .checked_add(1)
                    .ok_or(ForgeProposalSupportError::CountOverflow)?;
            }
            ForgeProposalEndpointValue::CounterfactualUnobserved => {
                self.counterfactual_unobserved = self
                    .counterfactual_unobserved
                    .checked_add(1)
                    .ok_or(ForgeProposalSupportError::CountOverflow)?;
            }
        }
        Ok(())
    }
}

/// Immutable accounting of whether the role-isolated training corpus satisfies the study's frozen
/// support thresholds. A failing receipt is still valid evidence; it simply cannot authorize fit.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalSupportReceipt {
    id: ContentId,
    study_id: ContentId,
    training_set_id: ContentId,
    endpoint_tag: &'static str,
    families: Vec<ForgeProposalFamilySupport>,
    all_families_supported: bool,
}

impl ForgeProposalSupportReceipt {
    pub fn evaluate(
        study: &ForgeProposalStudySpec,
        manifest: &ForgeProposalCorpusManifest,
        training: &ForgeProposalTrainingSet,
        validation: &ForgeProposalValidationSet,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<Self, ForgeProposalSupportError> {
        study.validate_for(manifest, training, validation, holdout)?;

        let mut by_family = BTreeMap::<String, MutableSupport>::new();
        for family in manifest.families() {
            family.validate()?;
            by_family.insert(
                family.as_content_id().as_str().to_string(),
                MutableSupport::default(),
            );
        }

        for table in training.tables() {
            table.validate()?;
            for row in table.rows() {
                let key = row.family_id().as_content_id().as_str();
                let support = by_family
                    .get_mut(key)
                    .ok_or(ForgeProposalSupportError::FamilyCoverageMismatch)?;
                let endpoint = ForgeProposalEndpointRecord::from_row(row, study.endpoint())?;
                support.observe(table.run_id(), &endpoint)?;
            }
        }

        let mut families = Vec::with_capacity(manifest.families().len());
        for family in manifest.families() {
            let counts = by_family
                .remove(family.as_content_id().as_str())
                .ok_or(ForgeProposalSupportError::FamilyCoverageMismatch)?;
            let selected_rows = counts
                .observed_labels
                .checked_add(counts.censored_labels)
                .ok_or(ForgeProposalSupportError::CountOverflow)?;
            let distinct_runs = u64::try_from(counts.observed_runs.len())
                .map_err(|_| ForgeProposalSupportError::CountOverflow)?;
            let meets = family_support_satisfied(
                distinct_runs,
                counts.observed_labels,
                counts.positive_labels,
                counts.negative_labels,
                study.support(),
            );
            let id = derive_family_support_id(
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
            let row = ForgeProposalFamilySupport {
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
                meets_frozen_support: meets,
            };
            row.validate_for(study.support())?;
            families.push(row);
        }
        if !by_family.is_empty() || families.len() != manifest.families().len() {
            return Err(ForgeProposalSupportError::FamilyCoverageMismatch);
        }

        let all_families_supported = families.iter().all(ForgeProposalFamilySupport::meets_frozen_support);
        let endpoint_tag = endpoint_tag(study.endpoint());
        let id = derive_receipt_id(
            study.id(),
            training.id(),
            endpoint_tag,
            &families,
            all_families_supported,
        );
        let receipt = Self {
            id,
            study_id: study.id().clone(),
            training_set_id: training.id().clone(),
            endpoint_tag,
            families,
            all_families_supported,
        };
        receipt.validate_for(study, training)?;
        Ok(receipt)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn study_id(&self) -> &ContentId { &self.study_id }
    pub fn training_set_id(&self) -> &ContentId { &self.training_set_id }
    pub fn families(&self) -> &[ForgeProposalFamilySupport] { &self.families }
    pub fn all_families_supported(&self) -> bool { self.all_families_supported }
    pub fn unsupported_families(&self) -> impl Iterator<Item = &ForgeProposalFamilySupport> {
        self.families.iter().filter(|row| !row.meets_frozen_support())
    }

    pub fn validate_for(
        &self,
        study: &ForgeProposalStudySpec,
        training: &ForgeProposalTrainingSet,
    ) -> Result<(), ForgeProposalSupportError> {
        if self.study_id != *study.id()
            || self.training_set_id != *training.id()
            || self.endpoint_tag != endpoint_tag(study.endpoint())
        {
            return Err(ForgeProposalSupportError::ReceiptScopeMismatch);
        }
        for family in &self.families {
            family.validate_for(study.support())?;
        }
        let all_supported = self
            .families
            .iter()
            .all(ForgeProposalFamilySupport::meets_frozen_support);
        if all_supported != self.all_families_supported {
            return Err(ForgeProposalSupportError::ReceiptIdentityMismatch);
        }
        let expected = derive_receipt_id(
            &self.study_id,
            &self.training_set_id,
            self.endpoint_tag,
            &self.families,
            self.all_families_supported,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalSupportError::ReceiptIdentityMismatch)
        }
    }
}

fn derive_receipt_id(
    study_id: &ContentId,
    training_set_id: &ContentId,
    endpoint_tag: &str,
    families: &[ForgeProposalFamilySupport],
    all_supported: bool,
) -> ContentId {
    let count = (families.len() as u64).to_be_bytes();
    let mut parts = vec![
        study_id.as_str().as_bytes().to_vec(),
        training_set_id.as_str().as_bytes().to_vec(),
        endpoint_tag.as_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        families
            .iter()
            .map(|row| row.id().as_str().as_bytes().to_vec()),
    );
    parts.push(if all_supported { b"all-supported".to_vec() } else { b"insufficient".to_vec() });
    ContentId::derive(
        "symthaea.forge-proposal-support-receipt.v1",
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

/// Identity-only authorization to invoke a future fitter on this exact study/training set under one
/// already-precommitted validation-coverage specification.
///
/// This type contains no training implementation and grants no search/runtime authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalFitPermit {
    id: ContentId,
    study_id: ContentId,
    training_set_id: ContentId,
    support_receipt_id: ContentId,
    validation_coverage_spec_id: ContentId,
}

impl ForgeProposalFitPermit {
    pub fn issue(
        study: &ForgeProposalStudySpec,
        training: &ForgeProposalTrainingSet,
        receipt: &ForgeProposalSupportReceipt,
        validation_coverage_spec: &ForgeProposalValidationCoverageSpec,
    ) -> Result<Self, ForgeProposalSupportError> {
        receipt.validate_for(study, training)?;
        validation_coverage_spec.validate_for(study)?;
        if !receipt.all_families_supported() {
            return Err(ForgeProposalSupportError::InsufficientSupport);
        }
        let id = derive_fit_permit_id(
            study.id(),
            training.id(),
            receipt.id(),
            validation_coverage_spec.id(),
        );
        Ok(Self {
            id,
            study_id: study.id().clone(),
            training_set_id: training.id().clone(),
            support_receipt_id: receipt.id().clone(),
            validation_coverage_spec_id: validation_coverage_spec.id().clone(),
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn study_id(&self) -> &ContentId { &self.study_id }
    pub fn training_set_id(&self) -> &ContentId { &self.training_set_id }
    pub fn support_receipt_id(&self) -> &ContentId { &self.support_receipt_id }
    pub fn validation_coverage_spec_id(&self) -> &ContentId { &self.validation_coverage_spec_id }

    pub fn validate_for(
        &self,
        study: &ForgeProposalStudySpec,
        training: &ForgeProposalTrainingSet,
        receipt: &ForgeProposalSupportReceipt,
        validation_coverage_spec: &ForgeProposalValidationCoverageSpec,
    ) -> Result<(), ForgeProposalSupportError> {
        receipt.validate_for(study, training)?;
        validation_coverage_spec.validate_for(study)?;
        if !receipt.all_families_supported()
            || self.study_id != *study.id()
            || self.training_set_id != *training.id()
            || self.support_receipt_id != *receipt.id()
            || self.validation_coverage_spec_id != *validation_coverage_spec.id()
            || derive_fit_permit_id(
                &self.study_id,
                &self.training_set_id,
                &self.support_receipt_id,
                &self.validation_coverage_spec_id,
            ) != self.id
        {
            return Err(ForgeProposalSupportError::FitPermitMismatch);
        }
        Ok(())
    }
}

fn derive_fit_permit_id(
    study_id: &ContentId,
    training_set_id: &ContentId,
    support_receipt_id: &ContentId,
    validation_coverage_spec_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-fit-permit.v2",
        [
            study_id.as_str().as_bytes(),
            training_set_id.as_str().as_bytes(),
            support_receipt_id.as_str().as_bytes(),
            validation_coverage_spec_id.as_str().as_bytes(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_predicate_requires_every_frozen_threshold() {
        let spec = ForgeProposalSupportSpec::new(3, 2, 2, 20, 5, 5).unwrap();
        assert!(family_support_satisfied(2, 20, 5, 15, &spec));
        assert!(!family_support_satisfied(1, 20, 5, 15, &spec));
        assert!(!family_support_satisfied(2, 19, 5, 14, &spec));
        assert!(!family_support_satisfied(2, 20, 4, 16, &spec));
        assert!(!family_support_satisfied(2, 20, 15, 4, &spec));
    }
}
