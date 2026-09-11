// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Schema-bound, label-blind validation feature extraction.
//!
//! Validation prediction targets identify proposal rows without exposing endpoint labels. This module
//! adds the exact pre-decision feature values permitted by the frozen study, including accepted
//! history reconstructed from the exact sequence batch bound by the corpus member. No endpoint,
//! trial outcome, selected site, candidate artifact, benchmark, or generation winner is emitted.

use crate::corpus_split::ForgeCorpusRole;
use crate::family_learning::ForgeTransformationFamilyId;
use crate::proposal_corpus::{
    ForgeProposalCorpusError, ForgeProposalCorpusManifest, ForgeProposalValidationSet,
};
use crate::proposal_history::{
    ForgeConditionedProposalObservationTable, ForgeProposalHistoryError,
};
use crate::proposal_study::{
    ForgeProposalFeature, ForgeProposalStudyError, ForgeProposalStudySpec,
};
use crate::proposal_validation_blind::{
    ForgeProposalBlindValidationError, ForgeProposalValidationTargetSet,
};
use crate::sequence_learning::ForgeFamilyHistory;
use crate::sequence_stats::{ExactContextForgeSequenceCohort, ForgeSequenceStatsError};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalValidationFeatureError {
    #[error(transparent)]
    Corpus(#[from] ForgeProposalCorpusError),
    #[error(transparent)]
    History(#[from] ForgeProposalHistoryError),
    #[error(transparent)]
    Sequence(#[from] ForgeSequenceStatsError),
    #[error(transparent)]
    Study(#[from] ForgeProposalStudyError),
    #[error(transparent)]
    Blind(#[from] ForgeProposalBlindValidationError),
    #[error("validation feature extraction does not bind the supplied frozen study/corpus/cohort")]
    ScopeMismatch,
    #[error("validation corpus member does not bind the exact sequence batch used for features")]
    SequenceBatchMismatch,
    #[error("validation feature extraction cannot resolve a label-blind target")]
    TargetCoverageMismatch,
    #[error("validation feature schema requests history without a frozen history order")]
    MissingHistoryOrder,
    #[error("validation feature row does not follow the frozen feature schema")]
    FeatureSchemaMismatch,
    #[error("validation feature row identity does not match canonical fields")]
    FeatureRowIdentityMismatch,
    #[error("validation feature table identity does not match canonical fields")]
    FeatureTableIdentityMismatch,
}

/// One exact feature value from the frozen pre-decision v1 vocabulary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "feature", content = "value")]
pub enum ForgeProposalValidationFeatureValue {
    FamilyIdentity(ForgeTransformationFamilyId),
    Generation(u64),
    FamilyEligibleSites(u64),
    TotalEligibleSites(u64),
    AcceptedHistorySuffix(ForgeFamilyHistory),
}

impl ForgeProposalValidationFeatureValue {
    fn kind(&self) -> ForgeProposalFeature {
        match self {
            Self::FamilyIdentity(_) => ForgeProposalFeature::FamilyIdentity,
            Self::Generation(_) => ForgeProposalFeature::Generation,
            Self::FamilyEligibleSites(_) => ForgeProposalFeature::FamilyEligibleSites,
            Self::TotalEligibleSites(_) => ForgeProposalFeature::TotalEligibleSites,
            Self::AcceptedHistorySuffix(_) => ForgeProposalFeature::AcceptedHistorySuffix,
        }
    }

    fn identity_parts(&self) -> Vec<Vec<u8>> {
        match self {
            Self::FamilyIdentity(family) => vec![
                b"family-identity".to_vec(),
                family.as_content_id().as_str().as_bytes().to_vec(),
            ],
            Self::Generation(value) => vec![
                b"generation".to_vec(),
                value.to_be_bytes().to_vec(),
            ],
            Self::FamilyEligibleSites(value) => vec![
                b"family-eligible-sites".to_vec(),
                value.to_be_bytes().to_vec(),
            ],
            Self::TotalEligibleSites(value) => vec![
                b"total-eligible-sites".to_vec(),
                value.to_be_bytes().to_vec(),
            ],
            Self::AcceptedHistorySuffix(history) => vec![
                b"accepted-history-suffix".to_vec(),
                history.id().as_str().as_bytes().to_vec(),
            ],
        }
    }
}

/// Label-blind feature vector for one exact validation target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalValidationFeatureRow {
    id: ContentId,
    target_id: ContentId,
    feature_schema_id: ContentId,
    values: Vec<ForgeProposalValidationFeatureValue>,
}

impl ForgeProposalValidationFeatureRow {
    fn new(
        target_id: ContentId,
        feature_schema_id: ContentId,
        schema: &[ForgeProposalFeature],
        values: Vec<ForgeProposalValidationFeatureValue>,
    ) -> Result<Self, ForgeProposalValidationFeatureError> {
        if values.len() != schema.len()
            || values
                .iter()
                .zip(schema.iter())
                .any(|(value, expected)| value.kind() != *expected)
        {
            return Err(ForgeProposalValidationFeatureError::FeatureSchemaMismatch);
        }
        let id = derive_feature_row_id(&target_id, &feature_schema_id, &values);
        Ok(Self {
            id,
            target_id,
            feature_schema_id,
            values,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn target_id(&self) -> &ContentId { &self.target_id }
    pub fn feature_schema_id(&self) -> &ContentId { &self.feature_schema_id }
    pub fn values(&self) -> &[ForgeProposalValidationFeatureValue] { &self.values }

    fn validate_schema(
        &self,
        schema_id: &ContentId,
        schema: &[ForgeProposalFeature],
    ) -> Result<(), ForgeProposalValidationFeatureError> {
        if &self.feature_schema_id != schema_id
            || self.values.len() != schema.len()
            || self
                .values
                .iter()
                .zip(schema.iter())
                .any(|(value, expected)| value.kind() != *expected)
        {
            return Err(ForgeProposalValidationFeatureError::FeatureSchemaMismatch);
        }
        for value in &self.values {
            match value {
                ForgeProposalValidationFeatureValue::FamilyIdentity(family) => family
                    .validate()
                    .map_err(|_| ForgeProposalValidationFeatureError::FeatureSchemaMismatch)?,
                ForgeProposalValidationFeatureValue::AcceptedHistorySuffix(history) => history
                    .validate()
                    .map_err(|_| ForgeProposalValidationFeatureError::FeatureSchemaMismatch)?,
                _ => {}
            }
        }
        let expected = derive_feature_row_id(
            &self.target_id,
            &self.feature_schema_id,
            &self.values,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalValidationFeatureError::FeatureRowIdentityMismatch)
        }
    }
}

fn derive_feature_row_id(
    target_id: &ContentId,
    feature_schema_id: &ContentId,
    values: &[ForgeProposalValidationFeatureValue],
) -> ContentId {
    let count = (values.len() as u64).to_be_bytes();
    let mut parts = vec![
        target_id.as_str().as_bytes().to_vec(),
        feature_schema_id.as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    for value in values {
        parts.extend(value.identity_parts());
    }
    ContentId::derive(
        "symthaea.forge-proposal-validation-feature-row.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Exact label-blind validation feature table for one frozen study.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalValidationFeatureTable {
    id: ContentId,
    study_id: ContentId,
    manifest_id: ContentId,
    validation_set_id: ContentId,
    sequence_cohort_id: ContentId,
    target_set_id: ContentId,
    feature_schema_id: ContentId,
    rows: Vec<ForgeProposalValidationFeatureRow>,
}

impl ForgeProposalValidationFeatureTable {
    pub fn derive(
        manifest: &ForgeProposalCorpusManifest,
        cohort: &ExactContextForgeSequenceCohort,
        study: &ForgeProposalStudySpec,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
    ) -> Result<Self, ForgeProposalValidationFeatureError> {
        manifest.validate()?;
        cohort.validate()?;
        validation.validate_for(manifest)?;
        study.feature_schema().validate()?;
        targets.validate_for(manifest, validation)?;
        if study.corpus_manifest_id() != manifest.id()
            || study.validation_set_id() != validation.id()
            || study.endpoint() != targets.endpoint()
            || manifest.sequence_cohort_id() != cohort.id()
        {
            return Err(ForgeProposalValidationFeatureError::ScopeMismatch);
        }

        let batches = cohort
            .batches()
            .iter()
            .map(|batch| (batch.run_id().as_str().to_string(), batch))
            .collect::<BTreeMap<_, _>>();
        let members = manifest
            .members_for(ForgeCorpusRole::Validation)
            .map(|member| (member.run_id().as_str().to_string(), member))
            .collect::<BTreeMap<_, _>>();
        let target_ids = targets
            .targets()
            .iter()
            .map(|target| target.id().as_str().to_string())
            .collect::<BTreeSet<_>>();
        if target_ids.len() != targets.targets().len() {
            return Err(ForgeProposalValidationFeatureError::TargetCoverageMismatch);
        }

        let schema = study.feature_schema().features();
        let history_order = study.feature_schema().history_order();
        let mut rows = Vec::with_capacity(targets.targets().len());
        let mut seen_targets = BTreeSet::new();

        for table in validation.tables() {
            let batch = batches
                .get(table.run_id().as_str())
                .ok_or(ForgeProposalValidationFeatureError::SequenceBatchMismatch)?;
            let member = members
                .get(table.run_id().as_str())
                .ok_or(ForgeProposalValidationFeatureError::SequenceBatchMismatch)?;
            if member.sequence_batch_id() != batch.id()
                || member.proposal_table_id() != table.id()
            {
                return Err(ForgeProposalValidationFeatureError::SequenceBatchMismatch);
            }
            let conditioned = ForgeConditionedProposalObservationTable::from_table(
                table,
                batch.sequence(),
            )?;
            for conditioned_row in conditioned.rows() {
                let source = conditioned_row.source();
                let target_id = derive_target_id_for_row(table.run_id(), source, study.endpoint());
                if !target_ids.contains(target_id.as_str())
                    || !seen_targets.insert(target_id.as_str().to_string())
                {
                    return Err(ForgeProposalValidationFeatureError::TargetCoverageMismatch);
                }
                let mut values = Vec::with_capacity(schema.len());
                for feature in schema {
                    let value = match feature {
                        ForgeProposalFeature::FamilyIdentity => {
                            ForgeProposalValidationFeatureValue::FamilyIdentity(
                                source.family_id().clone(),
                            )
                        }
                        ForgeProposalFeature::Generation => {
                            ForgeProposalValidationFeatureValue::Generation(source.generation())
                        }
                        ForgeProposalFeature::FamilyEligibleSites => {
                            ForgeProposalValidationFeatureValue::FamilyEligibleSites(
                                source.eligible_sites(),
                            )
                        }
                        ForgeProposalFeature::TotalEligibleSites => {
                            ForgeProposalValidationFeatureValue::TotalEligibleSites(
                                source.total_eligible_sites(),
                            )
                        }
                        ForgeProposalFeature::AcceptedHistorySuffix => {
                            let order = history_order.ok_or(
                                ForgeProposalValidationFeatureError::MissingHistoryOrder,
                            )?;
                            ForgeProposalValidationFeatureValue::AcceptedHistorySuffix(
                                conditioned_row.history_suffix(order)?,
                            )
                        }
                    };
                    values.push(value);
                }
                let row = ForgeProposalValidationFeatureRow::new(
                    target_id,
                    study.feature_schema().id().clone(),
                    schema,
                    values,
                )?;
                row.validate_schema(study.feature_schema().id(), schema)?;
                rows.push(row);
            }
        }

        if seen_targets != target_ids || rows.len() != targets.targets().len() {
            return Err(ForgeProposalValidationFeatureError::TargetCoverageMismatch);
        }
        rows.sort_by(|left, right| left.target_id().cmp(right.target_id()));
        let id = derive_feature_table_id(
            study.id(),
            manifest.id(),
            validation.id(),
            cohort.id(),
            targets.id(),
            study.feature_schema().id(),
            &rows,
        );
        Ok(Self {
            id,
            study_id: study.id().clone(),
            manifest_id: manifest.id().clone(),
            validation_set_id: validation.id().clone(),
            sequence_cohort_id: cohort.id().clone(),
            target_set_id: targets.id().clone(),
            feature_schema_id: study.feature_schema().id().clone(),
            rows,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn study_id(&self) -> &ContentId { &self.study_id }
    pub fn target_set_id(&self) -> &ContentId { &self.target_set_id }
    pub fn feature_schema_id(&self) -> &ContentId { &self.feature_schema_id }
    pub fn rows(&self) -> &[ForgeProposalValidationFeatureRow] { &self.rows }

    pub fn validate_for(
        &self,
        manifest: &ForgeProposalCorpusManifest,
        cohort: &ExactContextForgeSequenceCohort,
        study: &ForgeProposalStudySpec,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
    ) -> Result<(), ForgeProposalValidationFeatureError> {
        let rebuilt = Self::derive(manifest, cohort, study, validation, targets)?;
        if rebuilt.id == self.id
            && rebuilt.study_id == self.study_id
            && rebuilt.manifest_id == self.manifest_id
            && rebuilt.validation_set_id == self.validation_set_id
            && rebuilt.sequence_cohort_id == self.sequence_cohort_id
            && rebuilt.target_set_id == self.target_set_id
            && rebuilt.feature_schema_id == self.feature_schema_id
            && rebuilt.rows == self.rows
        {
            Ok(())
        } else {
            Err(ForgeProposalValidationFeatureError::FeatureTableIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_target_id_for_row(
    run_id: &ContentId,
    row: &crate::proposal_dataset::ForgeProposalObservationRow,
    endpoint: crate::proposal_endpoints::ForgeProposalEndpoint,
) -> ContentId {
    let endpoint = match endpoint {
        crate::proposal_endpoints::ForgeProposalEndpoint::DistinctCandidate => b"distinct-candidate".as_slice(),
        crate::proposal_endpoints::ForgeProposalEndpoint::CompilePassed => b"compile-passed".as_slice(),
        crate::proposal_endpoints::ForgeProposalEndpoint::CorrectnessPassed => b"correctness-passed".as_slice(),
        crate::proposal_endpoints::ForgeProposalEndpoint::EvaluationValid => b"evaluation-valid".as_slice(),
        crate::proposal_endpoints::ForgeProposalEndpoint::SelectedForContinuation => b"selected-for-continuation".as_slice(),
    };
    ContentId::derive(
        "symthaea.forge-proposal-validation-target.v1",
        [
            run_id.as_str().as_bytes(),
            row.attempt_id().as_content_id().as_str().as_bytes(),
            row.parent_artifact_id().as_str().as_bytes(),
            row.family_id().as_content_id().as_str().as_bytes(),
            row.generation().to_be_bytes().as_slice(),
            row.eligible_sites().to_be_bytes().as_slice(),
            row.total_eligible_sites().to_be_bytes().as_slice(),
            endpoint,
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn derive_feature_table_id(
    study_id: &ContentId,
    manifest_id: &ContentId,
    validation_set_id: &ContentId,
    sequence_cohort_id: &ContentId,
    target_set_id: &ContentId,
    feature_schema_id: &ContentId,
    rows: &[ForgeProposalValidationFeatureRow],
) -> ContentId {
    let count = (rows.len() as u64).to_be_bytes();
    let mut parts = vec![
        study_id.as_str().as_bytes().to_vec(),
        manifest_id.as_str().as_bytes().to_vec(),
        validation_set_id.as_str().as_bytes().to_vec(),
        sequence_cohort_id.as_str().as_bytes().to_vec(),
        target_set_id.as_str().as_bytes().to_vec(),
        feature_schema_id.as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(rows.iter().map(|row| row.id().as_str().as_bytes().to_vec()));
    ContentId::derive(
        "symthaea.forge-proposal-validation-feature-table.v1",
        parts.iter().map(Vec::as_slice),
    )
}
