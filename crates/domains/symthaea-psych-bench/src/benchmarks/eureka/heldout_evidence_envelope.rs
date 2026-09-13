// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical family-neutral paired HeldOut evidence envelope for EUREKA-002S.
//!
//! This module is evidence transport only. It does not construct worlds,
//! execute a target/comparator, fit a model, or decide a scientific result.
//! Both future family runners must commit predictions prospectively and then
//! use this envelope to derive both scores from the same realized transition.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2302>

use super::analysis_plan::{CampaignRowDisposition, EUREKA_002_ANALYSIS_PLAN_V1};
use super::consequence::{
    ConsequenceMetrics, ConsequencePrediction, ConsequenceScore, ConsequenceScoringError,
    PredictionOutcome, score_consequence,
};
use super::cross_family_analysis::{
    AnalysisMetricOutcome, EvidenceFamilyId, PairedAnalysisRow, RawConsequenceCounts,
};
use super::hidden_world::{CorpusPartition, PublicAction, PublicObservation, PublicValue};

pub(super) const HELDOUT_EVIDENCE_ENVELOPE_REVISION: &str =
    "EUREKA.002S.CANONICAL_HELDOUT_EVIDENCE_ENVELOPE.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PredictionRole {
    Target,
    Comparator,
}

impl PredictionRole {
    const fn tag(self) -> u8 {
        match self {
            Self::Target => 1,
            Self::Comparator => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ProspectivePredictionCommitment {
    pub role: PredictionRole,
    pub family: EvidenceFamilyId,
    pub analysis_plan_digest: u64,
    pub authorization_digest: u64,
    pub subject_digest: u64,
    pub profile_index: u32,
    pub seed_identity: u64,
    pub profile_digest: u64,
    pub world_digest: u64,
    pub sealed_action: PublicAction,
    pub sealed_target_action_index: u16,
    pub pre_state: PublicObservation,
    pub prediction: ConsequencePrediction,
    pub ordinal: u64,
    pub replay_digest: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ProspectivePredictionInput {
    pub role: PredictionRole,
    pub family: EvidenceFamilyId,
    pub authorization_digest: u64,
    pub subject_digest: u64,
    pub profile_index: u32,
    pub seed_identity: u64,
    pub profile_digest: u64,
    pub world_digest: u64,
    pub sealed_action: PublicAction,
    pub sealed_target_action_index: u16,
    pub pre_state: PublicObservation,
    pub prediction: ConsequencePrediction,
    pub ordinal: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HeldoutEvidenceEnvelopeError {
    PredictionActionMismatch,
    WrongPredictionRole,
    CommitmentFamilyMismatch,
    AnalysisPlanMismatch,
    AuthorizationMismatch,
    ProfileIndexMismatch,
    SeedIdentityMismatch,
    ProfileDigestMismatch,
    WorldDigestMismatch,
    SealedActionMismatch,
    TargetActionIndexMismatch,
    PreStateMismatch,
    PredictionNotProspective,
    Scoring(ConsequenceScoringError),
    MetricCountOverflow,
    InvalidDisposition,
    InvalidRowContainsMetricEvidence,
    ValidRowMissingMetricEvidence,
}

impl ProspectivePredictionCommitment {
    pub(super) fn freeze(
        input: ProspectivePredictionInput,
    ) -> Result<Self, HeldoutEvidenceEnvelopeError> {
        if input.prediction.action != input.sealed_action {
            return Err(HeldoutEvidenceEnvelopeError::PredictionActionMismatch);
        }
        let mut commitment = Self {
            role: input.role,
            family: input.family,
            analysis_plan_digest: EUREKA_002_ANALYSIS_PLAN_V1.replay_digest(),
            authorization_digest: input.authorization_digest,
            subject_digest: input.subject_digest,
            profile_index: input.profile_index,
            seed_identity: input.seed_identity,
            profile_digest: input.profile_digest,
            world_digest: input.world_digest,
            sealed_action: input.sealed_action,
            sealed_target_action_index: input.sealed_target_action_index,
            pre_state: input.pre_state,
            prediction: input.prediction,
            ordinal: input.ordinal,
            replay_digest: 0,
        };
        commitment.replay_digest = prediction_commitment_digest(&commitment);
        Ok(commitment)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct EnvelopeRawConsequenceCounts {
    pub field_count: u16,
    pub actual_changed: u16,
    pub true_positive_changes: u16,
    pub false_positive_changes: u16,
    pub missed_changes: u16,
    pub correct_changed_values: u16,
    pub correct_unchanged_values: u16,
}

impl EnvelopeRawConsequenceCounts {
    fn from_metrics(metrics: ConsequenceMetrics) -> Result<Self, HeldoutEvidenceEnvelopeError> {
        Ok(Self {
            field_count: u16::try_from(metrics.field_count)
                .map_err(|_| HeldoutEvidenceEnvelopeError::MetricCountOverflow)?,
            actual_changed: u16::try_from(metrics.actual_changed)
                .map_err(|_| HeldoutEvidenceEnvelopeError::MetricCountOverflow)?,
            true_positive_changes: u16::try_from(metrics.true_positive_changes)
                .map_err(|_| HeldoutEvidenceEnvelopeError::MetricCountOverflow)?,
            false_positive_changes: u16::try_from(metrics.false_positive_changes)
                .map_err(|_| HeldoutEvidenceEnvelopeError::MetricCountOverflow)?,
            missed_changes: u16::try_from(metrics.missed_changes)
                .map_err(|_| HeldoutEvidenceEnvelopeError::MetricCountOverflow)?,
            correct_changed_values: u16::try_from(metrics.correct_changed_values)
                .map_err(|_| HeldoutEvidenceEnvelopeError::MetricCountOverflow)?,
            correct_unchanged_values: u16::try_from(metrics.correct_unchanged_values)
                .map_err(|_| HeldoutEvidenceEnvelopeError::MetricCountOverflow)?,
        })
    }

    fn to_analysis(self) -> RawConsequenceCounts {
        RawConsequenceCounts {
            field_count: self.field_count,
            actual_changed: self.actual_changed,
            true_positive_changes: self.true_positive_changes,
            false_positive_changes: self.false_positive_changes,
            missed_changes: self.missed_changes,
            correct_changed_values: self.correct_changed_values,
            correct_unchanged_values: self.correct_unchanged_values,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EnvelopeMetricOutcome {
    Scored(EnvelopeRawConsequenceCounts),
    Abstained,
    OutOfDomain,
}

impl EnvelopeMetricOutcome {
    fn from_score(score: ConsequenceScore) -> Result<Self, HeldoutEvidenceEnvelopeError> {
        match score {
            ConsequenceScore::Scored(metrics) => Ok(Self::Scored(
                EnvelopeRawConsequenceCounts::from_metrics(metrics)?,
            )),
            ConsequenceScore::AbstainedInsufficientEvidence => Ok(Self::Abstained),
            ConsequenceScore::OutOfQualifiedDomain => Ok(Self::OutOfDomain),
        }
    }

    fn to_analysis(self) -> AnalysisMetricOutcome {
        match self {
            Self::Scored(counts) => AnalysisMetricOutcome::Scored(counts.to_analysis()),
            Self::Abstained => AnalysisMetricOutcome::Abstained,
            Self::OutOfDomain => AnalysisMetricOutcome::OutOfDomain,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CompletedHeldoutEvidenceRow {
    pub schema_revision: &'static str,
    pub family: EvidenceFamilyId,
    pub partition: CorpusPartition,
    pub analysis_plan_digest: u64,
    pub authorization_digest: u64,
    pub target_subject_digest: u64,
    pub comparator_subject_digest: u64,
    pub profile_index: u32,
    pub seed_identity: u64,
    pub profile_digest: u64,
    pub world_digest: u64,
    pub sealed_action: PublicAction,
    pub sealed_target_action_index: u16,
    pub pre_state: PublicObservation,
    pub target_prediction_commitment: ProspectivePredictionCommitment,
    pub comparator_prediction_commitment: ProspectivePredictionCommitment,
    pub post_state: PublicObservation,
    pub transition_digest: u64,
    pub transition_ordinal: u64,
    pub target_score: EnvelopeMetricOutcome,
    pub comparator_score: EnvelopeMetricOutcome,
    pub disposition: CampaignRowDisposition,
    pub replay_digest: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CompletedHeldoutEvidenceInput {
    pub family: EvidenceFamilyId,
    pub authorization_digest: u64,
    pub target_subject_digest: u64,
    pub comparator_subject_digest: u64,
    pub profile_index: u32,
    pub seed_identity: u64,
    pub profile_digest: u64,
    pub world_digest: u64,
    pub sealed_action: PublicAction,
    pub sealed_target_action_index: u16,
    pub pre_state: PublicObservation,
    pub target_prediction_commitment: ProspectivePredictionCommitment,
    pub comparator_prediction_commitment: ProspectivePredictionCommitment,
    pub post_state: PublicObservation,
    pub transition_digest: u64,
    pub transition_ordinal: u64,
}

impl CompletedHeldoutEvidenceRow {
    pub(super) fn complete(
        input: CompletedHeldoutEvidenceInput,
    ) -> Result<Self, HeldoutEvidenceEnvelopeError> {
        validate_commitment(
            &input.target_prediction_commitment,
            PredictionRole::Target,
            input.family,
            input.authorization_digest,
            input.target_subject_digest,
            input.profile_index,
            input.seed_identity,
            input.profile_digest,
            input.world_digest,
            input.sealed_action,
            input.sealed_target_action_index,
            &input.pre_state,
            input.transition_ordinal,
        )?;
        validate_commitment(
            &input.comparator_prediction_commitment,
            PredictionRole::Comparator,
            input.family,
            input.authorization_digest,
            input.comparator_subject_digest,
            input.profile_index,
            input.seed_identity,
            input.profile_digest,
            input.world_digest,
            input.sealed_action,
            input.sealed_target_action_index,
            &input.pre_state,
            input.transition_ordinal,
        )?;

        let target_score = score_consequence(
            &input.pre_state,
            &input.target_prediction_commitment.prediction,
            &input.post_state,
        )
        .map_err(HeldoutEvidenceEnvelopeError::Scoring)
        .and_then(EnvelopeMetricOutcome::from_score)?;
        let comparator_score = score_consequence(
            &input.pre_state,
            &input.comparator_prediction_commitment.prediction,
            &input.post_state,
        )
        .map_err(HeldoutEvidenceEnvelopeError::Scoring)
        .and_then(EnvelopeMetricOutcome::from_score)?;

        let disposition = match target_score {
            EnvelopeMetricOutcome::Scored(_) => CampaignRowDisposition::ValidScored,
            EnvelopeMetricOutcome::Abstained => CampaignRowDisposition::ValidAbstained,
            EnvelopeMetricOutcome::OutOfDomain => CampaignRowDisposition::ValidOutOfDomain,
        };

        let mut row = Self {
            schema_revision: HELDOUT_EVIDENCE_ENVELOPE_REVISION,
            family: input.family,
            partition: CorpusPartition::HeldOutEvaluation,
            analysis_plan_digest: EUREKA_002_ANALYSIS_PLAN_V1.replay_digest(),
            authorization_digest: input.authorization_digest,
            target_subject_digest: input.target_subject_digest,
            comparator_subject_digest: input.comparator_subject_digest,
            profile_index: input.profile_index,
            seed_identity: input.seed_identity,
            profile_digest: input.profile_digest,
            world_digest: input.world_digest,
            sealed_action: input.sealed_action,
            sealed_target_action_index: input.sealed_target_action_index,
            pre_state: input.pre_state,
            target_prediction_commitment: input.target_prediction_commitment,
            comparator_prediction_commitment: input.comparator_prediction_commitment,
            post_state: input.post_state,
            transition_digest: input.transition_digest,
            transition_ordinal: input.transition_ordinal,
            target_score,
            comparator_score,
            disposition,
            replay_digest: 0,
        };
        row.replay_digest = completed_row_digest(&row);
        Ok(row)
    }

    pub(super) fn to_cross_family_row(&self) -> PairedAnalysisRow {
        PairedAnalysisRow {
            row_identity: self.replay_digest,
            family: self.family,
            partition: CorpusPartition::HeldOutEvaluation,
            seed_identity: self.seed_identity,
            disposition: self.disposition,
            candidate: self.target_score.to_analysis(),
            comparator: self.comparator_score.to_analysis(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct InvalidHeldoutEvidenceRow {
    pub schema_revision: &'static str,
    pub family: EvidenceFamilyId,
    pub partition: CorpusPartition,
    pub analysis_plan_digest: u64,
    pub authorization_digest: u64,
    pub target_subject_digest: u64,
    pub comparator_subject_digest: u64,
    pub profile_index: u32,
    pub seed_identity: u64,
    pub profile_digest: u64,
    pub world_digest: u64,
    pub sealed_action: PublicAction,
    pub sealed_target_action_index: u16,
    pub pre_state: Option<PublicObservation>,
    pub target_prediction_commitment: Option<ProspectivePredictionCommitment>,
    pub comparator_prediction_commitment: Option<ProspectivePredictionCommitment>,
    pub transition_digest: Option<u64>,
    pub transition_ordinal: Option<u64>,
    pub disposition: CampaignRowDisposition,
    pub replay_digest: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct InvalidHeldoutEvidenceInput {
    pub family: EvidenceFamilyId,
    pub authorization_digest: u64,
    pub target_subject_digest: u64,
    pub comparator_subject_digest: u64,
    pub profile_index: u32,
    pub seed_identity: u64,
    pub profile_digest: u64,
    pub world_digest: u64,
    pub sealed_action: PublicAction,
    pub sealed_target_action_index: u16,
    pub pre_state: Option<PublicObservation>,
    pub target_prediction_commitment: Option<ProspectivePredictionCommitment>,
    pub comparator_prediction_commitment: Option<ProspectivePredictionCommitment>,
    pub transition_digest: Option<u64>,
    pub transition_ordinal: Option<u64>,
    pub disposition: CampaignRowDisposition,
}

impl InvalidHeldoutEvidenceRow {
    pub(super) fn freeze(
        input: InvalidHeldoutEvidenceInput,
    ) -> Result<Self, HeldoutEvidenceEnvelopeError> {
        if !is_invalid_disposition(input.disposition) {
            return Err(HeldoutEvidenceEnvelopeError::InvalidDisposition);
        }
        for commitment in [
            input.target_prediction_commitment.as_ref(),
            input.comparator_prediction_commitment.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if commitment.family != input.family {
                return Err(HeldoutEvidenceEnvelopeError::CommitmentFamilyMismatch);
            }
            if commitment.authorization_digest != input.authorization_digest {
                return Err(HeldoutEvidenceEnvelopeError::AuthorizationMismatch);
            }
            if commitment.profile_index != input.profile_index {
                return Err(HeldoutEvidenceEnvelopeError::ProfileIndexMismatch);
            }
            if commitment.seed_identity != input.seed_identity {
                return Err(HeldoutEvidenceEnvelopeError::SeedIdentityMismatch);
            }
            if commitment.profile_digest != input.profile_digest {
                return Err(HeldoutEvidenceEnvelopeError::ProfileDigestMismatch);
            }
            if commitment.world_digest != input.world_digest {
                return Err(HeldoutEvidenceEnvelopeError::WorldDigestMismatch);
            }
            if commitment.sealed_action != input.sealed_action {
                return Err(HeldoutEvidenceEnvelopeError::SealedActionMismatch);
            }
            if commitment.sealed_target_action_index != input.sealed_target_action_index {
                return Err(HeldoutEvidenceEnvelopeError::TargetActionIndexMismatch);
            }
            if let Some(pre) = input.pre_state.as_ref() {
                if commitment.pre_state != *pre {
                    return Err(HeldoutEvidenceEnvelopeError::PreStateMismatch);
                }
            }
            if let Some(transition_ordinal) = input.transition_ordinal {
                if commitment.ordinal >= transition_ordinal {
                    return Err(HeldoutEvidenceEnvelopeError::PredictionNotProspective);
                }
            }
        }

        let mut row = Self {
            schema_revision: HELDOUT_EVIDENCE_ENVELOPE_REVISION,
            family: input.family,
            partition: CorpusPartition::HeldOutEvaluation,
            analysis_plan_digest: EUREKA_002_ANALYSIS_PLAN_V1.replay_digest(),
            authorization_digest: input.authorization_digest,
            target_subject_digest: input.target_subject_digest,
            comparator_subject_digest: input.comparator_subject_digest,
            profile_index: input.profile_index,
            seed_identity: input.seed_identity,
            profile_digest: input.profile_digest,
            world_digest: input.world_digest,
            sealed_action: input.sealed_action,
            sealed_target_action_index: input.sealed_target_action_index,
            pre_state: input.pre_state,
            target_prediction_commitment: input.target_prediction_commitment,
            comparator_prediction_commitment: input.comparator_prediction_commitment,
            transition_digest: input.transition_digest,
            transition_ordinal: input.transition_ordinal,
            disposition: input.disposition,
            replay_digest: 0,
        };
        row.replay_digest = invalid_row_digest(&row);
        Ok(row)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CanonicalHeldoutEvidenceRow {
    Completed(CompletedHeldoutEvidenceRow),
    Invalid(InvalidHeldoutEvidenceRow),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CanonicalAnalysisMetricOutcome {
    Scored(EnvelopeRawConsequenceCounts),
    Abstained,
    OutOfDomain,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CanonicalAnalysisProjection {
    pub row_identity: u64,
    pub family: EvidenceFamilyId,
    pub partition: CorpusPartition,
    pub seed_identity: u64,
    pub disposition: CampaignRowDisposition,
    pub candidate: Option<CanonicalAnalysisMetricOutcome>,
    pub comparator: Option<CanonicalAnalysisMetricOutcome>,
}

impl CanonicalHeldoutEvidenceRow {
    pub(super) fn replay_digest(&self) -> u64 {
        match self {
            Self::Completed(row) => row.replay_digest,
            Self::Invalid(row) => row.replay_digest,
        }
    }

    pub(super) fn analysis_projection(&self) -> CanonicalAnalysisProjection {
        match self {
            Self::Completed(row) => CanonicalAnalysisProjection {
                row_identity: row.replay_digest,
                family: row.family,
                partition: row.partition,
                seed_identity: row.seed_identity,
                disposition: row.disposition,
                candidate: Some(metric_projection(row.target_score)),
                comparator: Some(metric_projection(row.comparator_score)),
            },
            Self::Invalid(row) => CanonicalAnalysisProjection {
                row_identity: row.replay_digest,
                family: row.family,
                partition: row.partition,
                seed_identity: row.seed_identity,
                disposition: row.disposition,
                candidate: None,
                comparator: None,
            },
        }
    }

    /// #2296 currently accepts metric outcomes directly. This adapter is
    /// intentionally valid-row-only so invalid evidence can never be encoded as
    /// a fake abstention/OOD placeholder. A follow-up input-type evolution will
    /// make #2296 consume `CanonicalAnalysisProjection` directly.
    pub(super) fn valid_cross_family_row(&self) -> Option<PairedAnalysisRow> {
        match self {
            Self::Completed(row) => Some(row.to_cross_family_row()),
            Self::Invalid(_) => None,
        }
    }
}

fn metric_projection(outcome: EnvelopeMetricOutcome) -> CanonicalAnalysisMetricOutcome {
    match outcome {
        EnvelopeMetricOutcome::Scored(counts) => CanonicalAnalysisMetricOutcome::Scored(counts),
        EnvelopeMetricOutcome::Abstained => CanonicalAnalysisMetricOutcome::Abstained,
        EnvelopeMetricOutcome::OutOfDomain => CanonicalAnalysisMetricOutcome::OutOfDomain,
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_commitment(
    commitment: &ProspectivePredictionCommitment,
    role: PredictionRole,
    family: EvidenceFamilyId,
    authorization_digest: u64,
    subject_digest: u64,
    profile_index: u32,
    seed_identity: u64,
    profile_digest: u64,
    world_digest: u64,
    sealed_action: PublicAction,
    sealed_target_action_index: u16,
    pre_state: &PublicObservation,
    transition_ordinal: u64,
) -> Result<(), HeldoutEvidenceEnvelopeError> {
    if commitment.role != role {
        return Err(HeldoutEvidenceEnvelopeError::WrongPredictionRole);
    }
    if commitment.family != family {
        return Err(HeldoutEvidenceEnvelopeError::CommitmentFamilyMismatch);
    }
    if commitment.analysis_plan_digest != EUREKA_002_ANALYSIS_PLAN_V1.replay_digest() {
        return Err(HeldoutEvidenceEnvelopeError::AnalysisPlanMismatch);
    }
    if commitment.authorization_digest != authorization_digest {
        return Err(HeldoutEvidenceEnvelopeError::AuthorizationMismatch);
    }
    if commitment.subject_digest != subject_digest {
        return Err(HeldoutEvidenceEnvelopeError::AuthorizationMismatch);
    }
    if commitment.profile_index != profile_index {
        return Err(HeldoutEvidenceEnvelopeError::ProfileIndexMismatch);
    }
    if commitment.seed_identity != seed_identity {
        return Err(HeldoutEvidenceEnvelopeError::SeedIdentityMismatch);
    }
    if commitment.profile_digest != profile_digest {
        return Err(HeldoutEvidenceEnvelopeError::ProfileDigestMismatch);
    }
    if commitment.world_digest != world_digest {
        return Err(HeldoutEvidenceEnvelopeError::WorldDigestMismatch);
    }
    if commitment.sealed_action != sealed_action {
        return Err(HeldoutEvidenceEnvelopeError::SealedActionMismatch);
    }
    if commitment.sealed_target_action_index != sealed_target_action_index {
        return Err(HeldoutEvidenceEnvelopeError::TargetActionIndexMismatch);
    }
    if commitment.pre_state != *pre_state {
        return Err(HeldoutEvidenceEnvelopeError::PreStateMismatch);
    }
    if commitment.prediction.action != sealed_action {
        return Err(HeldoutEvidenceEnvelopeError::PredictionActionMismatch);
    }
    if commitment.ordinal >= transition_ordinal {
        return Err(HeldoutEvidenceEnvelopeError::PredictionNotProspective);
    }
    if prediction_commitment_digest(commitment) != commitment.replay_digest {
        return Err(HeldoutEvidenceEnvelopeError::AnalysisPlanMismatch);
    }
    Ok(())
}

fn is_invalid_disposition(disposition: CampaignRowDisposition) -> bool {
    matches!(
        disposition,
        CampaignRowDisposition::InvalidLeakage
            | CampaignRowDisposition::InvalidLineageMismatch
            | CampaignRowDisposition::ScorerFailure
            | CampaignRowDisposition::TargetExecutionFailure
            | CampaignRowDisposition::InfrastructureIndeterminate
    )
}

fn prediction_commitment_digest(commitment: &ProspectivePredictionCommitment) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"eureka.002s.prediction-commitment.v1\0");
    bytes.push(commitment.role.tag());
    bytes.push(commitment.family.tag());
    bytes.extend_from_slice(&commitment.analysis_plan_digest.to_le_bytes());
    bytes.extend_from_slice(&commitment.authorization_digest.to_le_bytes());
    bytes.extend_from_slice(&commitment.subject_digest.to_le_bytes());
    bytes.extend_from_slice(&commitment.profile_index.to_le_bytes());
    bytes.extend_from_slice(&commitment.seed_identity.to_le_bytes());
    bytes.extend_from_slice(&commitment.profile_digest.to_le_bytes());
    bytes.extend_from_slice(&commitment.world_digest.to_le_bytes());
    encode_action(&mut bytes, commitment.sealed_action);
    bytes.extend_from_slice(&commitment.sealed_target_action_index.to_le_bytes());
    encode_observation(&mut bytes, &commitment.pre_state);
    encode_prediction(&mut bytes, &commitment.prediction);
    bytes.extend_from_slice(&commitment.ordinal.to_le_bytes());
    fnv1a64(&bytes)
}

fn completed_row_digest(row: &CompletedHeldoutEvidenceRow) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"eureka.002s.completed-row.v1\0");
    encode_str(&mut bytes, row.schema_revision);
    bytes.push(row.family.tag());
    bytes.push(partition_tag(row.partition));
    bytes.extend_from_slice(&row.analysis_plan_digest.to_le_bytes());
    bytes.extend_from_slice(&row.authorization_digest.to_le_bytes());
    bytes.extend_from_slice(&row.target_subject_digest.to_le_bytes());
    bytes.extend_from_slice(&row.comparator_subject_digest.to_le_bytes());
    bytes.extend_from_slice(&row.profile_index.to_le_bytes());
    bytes.extend_from_slice(&row.seed_identity.to_le_bytes());
    bytes.extend_from_slice(&row.profile_digest.to_le_bytes());
    bytes.extend_from_slice(&row.world_digest.to_le_bytes());
    encode_action(&mut bytes, row.sealed_action);
    bytes.extend_from_slice(&row.sealed_target_action_index.to_le_bytes());
    encode_observation(&mut bytes, &row.pre_state);
    bytes.extend_from_slice(&row.target_prediction_commitment.replay_digest.to_le_bytes());
    bytes.extend_from_slice(&row.comparator_prediction_commitment.replay_digest.to_le_bytes());
    encode_observation(&mut bytes, &row.post_state);
    bytes.extend_from_slice(&row.transition_digest.to_le_bytes());
    bytes.extend_from_slice(&row.transition_ordinal.to_le_bytes());
    encode_metric_outcome(&mut bytes, row.target_score);
    encode_metric_outcome(&mut bytes, row.comparator_score);
    bytes.push(disposition_tag(row.disposition));
    fnv1a64(&bytes)
}

fn invalid_row_digest(row: &InvalidHeldoutEvidenceRow) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"eureka.002s.invalid-row.v1\0");
    encode_str(&mut bytes, row.schema_revision);
    bytes.push(row.family.tag());
    bytes.push(partition_tag(row.partition));
    bytes.extend_from_slice(&row.analysis_plan_digest.to_le_bytes());
    bytes.extend_from_slice(&row.authorization_digest.to_le_bytes());
    bytes.extend_from_slice(&row.target_subject_digest.to_le_bytes());
    bytes.extend_from_slice(&row.comparator_subject_digest.to_le_bytes());
    bytes.extend_from_slice(&row.profile_index.to_le_bytes());
    bytes.extend_from_slice(&row.seed_identity.to_le_bytes());
    bytes.extend_from_slice(&row.profile_digest.to_le_bytes());
    bytes.extend_from_slice(&row.world_digest.to_le_bytes());
    encode_action(&mut bytes, row.sealed_action);
    bytes.extend_from_slice(&row.sealed_target_action_index.to_le_bytes());
    encode_optional_observation(&mut bytes, row.pre_state.as_ref());
    encode_optional_commitment(&mut bytes, row.target_prediction_commitment.as_ref());
    encode_optional_commitment(&mut bytes, row.comparator_prediction_commitment.as_ref());
    encode_optional_u64(&mut bytes, row.transition_digest);
    encode_optional_u64(&mut bytes, row.transition_ordinal);
    bytes.push(disposition_tag(row.disposition));
    fnv1a64(&bytes)
}

fn encode_prediction(bytes: &mut Vec<u8>, prediction: &ConsequencePrediction) {
    encode_action(bytes, prediction.action);
    match &prediction.outcome {
        PredictionOutcome::Predicted { fields } => {
            bytes.push(1);
            bytes.extend_from_slice(&(fields.len() as u64).to_le_bytes());
            for value in fields {
                encode_value(bytes, *value);
            }
        }
        PredictionOutcome::AbstainInsufficientEvidence => bytes.push(2),
        PredictionOutcome::OutOfQualifiedDomain => bytes.push(3),
    }
}

fn encode_metric_outcome(bytes: &mut Vec<u8>, outcome: EnvelopeMetricOutcome) {
    match outcome {
        EnvelopeMetricOutcome::Scored(counts) => {
            bytes.push(1);
            bytes.extend_from_slice(&counts.field_count.to_le_bytes());
            bytes.extend_from_slice(&counts.actual_changed.to_le_bytes());
            bytes.extend_from_slice(&counts.true_positive_changes.to_le_bytes());
            bytes.extend_from_slice(&counts.false_positive_changes.to_le_bytes());
            bytes.extend_from_slice(&counts.missed_changes.to_le_bytes());
            bytes.extend_from_slice(&counts.correct_changed_values.to_le_bytes());
            bytes.extend_from_slice(&counts.correct_unchanged_values.to_le_bytes());
        }
        EnvelopeMetricOutcome::Abstained => bytes.push(2),
        EnvelopeMetricOutcome::OutOfDomain => bytes.push(3),
    }
}

fn encode_action(bytes: &mut Vec<u8>, action: PublicAction) {
    match action {
        PublicAction::NoOp => bytes.push(1),
        PublicAction::Pulse { slot } => {
            bytes.push(2);
            bytes.push(slot);
        }
        PublicAction::Transfer { from, to, amount } => {
            bytes.push(3);
            bytes.push(from);
            bytes.push(to);
            bytes.extend_from_slice(&amount.to_le_bytes());
        }
    }
}

fn encode_observation(bytes: &mut Vec<u8>, observation: &PublicObservation) {
    bytes.extend_from_slice(&observation.step.to_le_bytes());
    bytes.extend_from_slice(&(observation.fields.len() as u64).to_le_bytes());
    for value in &observation.fields {
        encode_value(bytes, *value);
    }
}

fn encode_optional_observation(bytes: &mut Vec<u8>, observation: Option<&PublicObservation>) {
    match observation {
        Some(value) => {
            bytes.push(1);
            encode_observation(bytes, value);
        }
        None => bytes.push(0),
    }
}

fn encode_optional_commitment(
    bytes: &mut Vec<u8>,
    commitment: Option<&ProspectivePredictionCommitment>,
) {
    match commitment {
        Some(value) => {
            bytes.push(1);
            bytes.extend_from_slice(&value.replay_digest.to_le_bytes());
        }
        None => bytes.push(0),
    }
}

fn encode_optional_u64(bytes: &mut Vec<u8>, value: Option<u64>) {
    match value {
        Some(value) => {
            bytes.push(1);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        None => bytes.push(0),
    }
}

fn encode_value(bytes: &mut Vec<u8>, value: PublicValue) {
    match value {
        PublicValue::Bit(value) => {
            bytes.push(1);
            bytes.push(u8::from(value));
        }
        PublicValue::Count(value) => {
            bytes.push(2);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
}

fn encode_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn partition_tag(partition: CorpusPartition) -> u8 {
    match partition {
        CorpusPartition::Development => 1,
        CorpusPartition::Calibration => 2,
        CorpusPartition::HeldOutEvaluation => 3,
        CorpusPartition::ExternalReplication => 4,
    }
}

fn disposition_tag(disposition: CampaignRowDisposition) -> u8 {
    match disposition {
        CampaignRowDisposition::ValidScored => 1,
        CampaignRowDisposition::ValidAbstained => 2,
        CampaignRowDisposition::ValidOutOfDomain => 3,
        CampaignRowDisposition::InvalidLeakage => 4,
        CampaignRowDisposition::InvalidLineageMismatch => 5,
        CampaignRowDisposition::ScorerFailure => 6,
        CampaignRowDisposition::TargetExecutionFailure => 7,
        CampaignRowDisposition::InfrastructureIndeterminate => 8,
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(step: u32, values: impl IntoIterator<Item = PublicValue>) -> PublicObservation {
        PublicObservation {
            step,
            fields: values.into_iter().collect(),
        }
    }

    fn pred(action: PublicAction, values: impl IntoIterator<Item = PublicValue>) -> ConsequencePrediction {
        ConsequencePrediction {
            action,
            outcome: PredictionOutcome::Predicted {
                fields: values.into_iter().collect(),
            },
        }
    }

    fn commitment(
        role: PredictionRole,
        subject: u64,
        action: PublicAction,
        prediction: ConsequencePrediction,
        ordinal: u64,
    ) -> ProspectivePredictionCommitment {
        ProspectivePredictionCommitment::freeze(ProspectivePredictionInput {
            role,
            family: EvidenceFamilyId::ResourceFlowV1,
            authorization_digest: 11,
            subject_digest: subject,
            profile_index: 7,
            seed_identity: 77,
            profile_digest: 22,
            world_digest: 33,
            sealed_action: action,
            sealed_target_action_index: 2,
            pre_state: obs(
                0,
                [
                    PublicValue::Count(0),
                    PublicValue::Count(2),
                    PublicValue::Count(8),
                ],
            ),
            prediction,
            ordinal,
        })
        .unwrap()
    }

    fn completed_row() -> CompletedHeldoutEvidenceRow {
        let action = PublicAction::Transfer {
            from: 0,
            to: 1,
            amount: 1,
        };
        let pre = obs(
            0,
            [
                PublicValue::Count(0),
                PublicValue::Count(2),
                PublicValue::Count(8),
            ],
        );
        let post = obs(
            1,
            [
                PublicValue::Count(1),
                PublicValue::Count(1),
                PublicValue::Count(8),
            ],
        );
        let target = commitment(
            PredictionRole::Target,
            101,
            action,
            pred(action, post.fields.clone()),
            10,
        );
        let comparator = commitment(
            PredictionRole::Comparator,
            202,
            action,
            pred(action, pre.fields.clone()),
            11,
        );
        CompletedHeldoutEvidenceRow::complete(CompletedHeldoutEvidenceInput {
            family: EvidenceFamilyId::ResourceFlowV1,
            authorization_digest: 11,
            target_subject_digest: 101,
            comparator_subject_digest: 202,
            profile_index: 7,
            seed_identity: 77,
            profile_digest: 22,
            world_digest: 33,
            sealed_action: action,
            sealed_target_action_index: 2,
            pre_state: pre,
            target_prediction_commitment: target,
            comparator_prediction_commitment: comparator,
            post_state: post,
            transition_digest: 44,
            transition_ordinal: 12,
        })
        .unwrap()
    }

    #[test]
    fn same_completed_row_has_same_identity_and_subject_changes_identity() {
        let a = completed_row();
        let b = completed_row();
        assert_eq!(a.replay_digest, b.replay_digest);

        let mut input = b.clone();
        input.target_subject_digest = 999;
        // Row identity binds the subject digest even if a caller tampers with a
        // detached copy; canonical construction would reject a mismatched
        // commitment before this state could be minted.
        assert_ne!(completed_row_digest(&input), a.replay_digest);
    }

    #[test]
    fn prediction_must_name_the_exact_sealed_action() {
        let action = PublicAction::NoOp;
        let err = ProspectivePredictionCommitment::freeze(ProspectivePredictionInput {
            role: PredictionRole::Target,
            family: EvidenceFamilyId::RelayTriadV1,
            authorization_digest: 1,
            subject_digest: 2,
            profile_index: 0,
            seed_identity: 3,
            profile_digest: 4,
            world_digest: 5,
            sealed_action: action,
            sealed_target_action_index: 0,
            pre_state: obs(0, [PublicValue::Bit(false); 3]),
            prediction: pred(PublicAction::Pulse { slot: 0 }, [PublicValue::Bit(false); 3]),
            ordinal: 1,
        })
        .unwrap_err();
        assert_eq!(err, HeldoutEvidenceEnvelopeError::PredictionActionMismatch);
    }

    #[test]
    fn prediction_ordinals_must_precede_transition_realization() {
        let mut row = completed_row();
        row.target_prediction_commitment.ordinal = row.transition_ordinal;
        assert_eq!(
            validate_commitment(
                &row.target_prediction_commitment,
                PredictionRole::Target,
                row.family,
                row.authorization_digest,
                row.target_subject_digest,
                row.profile_index,
                row.seed_identity,
                row.profile_digest,
                row.world_digest,
                row.sealed_action,
                row.sealed_target_action_index,
                &row.pre_state,
                row.transition_ordinal,
            ),
            Err(HeldoutEvidenceEnvelopeError::PredictionNotProspective)
        );
    }

    #[test]
    fn same_transition_scoring_projects_exact_integer_counts() {
        let row = completed_row();
        let EnvelopeMetricOutcome::Scored(target) = row.target_score else {
            panic!("expected scored target");
        };
        let EnvelopeMetricOutcome::Scored(comparator) = row.comparator_score else {
            panic!("expected scored comparator");
        };
        assert_eq!(target.field_count, 3);
        assert_eq!(target.actual_changed, 2);
        assert_eq!(target.true_positive_changes, 2);
        assert_eq!(target.missed_changes, 0);
        assert_eq!(target.correct_changed_values, 2);
        assert_eq!(comparator.actual_changed, 2);
        assert_eq!(comparator.true_positive_changes, 0);
        assert_eq!(comparator.missed_changes, 2);
        assert_eq!(comparator.correct_changed_values, 0);

        let analysis = row.to_cross_family_row();
        assert_eq!(analysis.row_identity, row.replay_digest);
        assert_eq!(analysis.partition, CorpusPartition::HeldOutEvaluation);
    }

    #[test]
    fn abstention_and_out_of_domain_remain_distinct() {
        let action = PublicAction::NoOp;
        let pre = obs(0, [PublicValue::Bit(false), PublicValue::Bit(true)]);
        let post = obs(1, [PublicValue::Bit(true), PublicValue::Bit(true)]);
        let target = ProspectivePredictionCommitment::freeze(ProspectivePredictionInput {
            role: PredictionRole::Target,
            family: EvidenceFamilyId::RelayTriadV1,
            authorization_digest: 1,
            subject_digest: 2,
            profile_index: 0,
            seed_identity: 3,
            profile_digest: 4,
            world_digest: 5,
            sealed_action: action,
            sealed_target_action_index: 0,
            pre_state: pre.clone(),
            prediction: ConsequencePrediction {
                action,
                outcome: PredictionOutcome::AbstainInsufficientEvidence,
            },
            ordinal: 1,
        })
        .unwrap();
        let comparator = ProspectivePredictionCommitment::freeze(ProspectivePredictionInput {
            role: PredictionRole::Comparator,
            family: EvidenceFamilyId::RelayTriadV1,
            authorization_digest: 1,
            subject_digest: 9,
            profile_index: 0,
            seed_identity: 3,
            profile_digest: 4,
            world_digest: 5,
            sealed_action: action,
            sealed_target_action_index: 0,
            pre_state: pre.clone(),
            prediction: ConsequencePrediction {
                action,
                outcome: PredictionOutcome::OutOfQualifiedDomain,
            },
            ordinal: 2,
        })
        .unwrap();
        let row = CompletedHeldoutEvidenceRow::complete(CompletedHeldoutEvidenceInput {
            family: EvidenceFamilyId::RelayTriadV1,
            authorization_digest: 1,
            target_subject_digest: 2,
            comparator_subject_digest: 9,
            profile_index: 0,
            seed_identity: 3,
            profile_digest: 4,
            world_digest: 5,
            sealed_action: action,
            sealed_target_action_index: 0,
            pre_state: pre,
            target_prediction_commitment: target,
            comparator_prediction_commitment: comparator,
            post_state: post,
            transition_digest: 6,
            transition_ordinal: 3,
        })
        .unwrap();
        assert_eq!(row.target_score, EnvelopeMetricOutcome::Abstained);
        assert_eq!(row.comparator_score, EnvelopeMetricOutcome::OutOfDomain);
        assert_eq!(row.disposition, CampaignRowDisposition::ValidAbstained);
    }

    #[test]
    fn invalid_row_has_no_fake_metric_outcome() {
        let invalid = CanonicalHeldoutEvidenceRow::Invalid(
            InvalidHeldoutEvidenceRow::freeze(InvalidHeldoutEvidenceInput {
                family: EvidenceFamilyId::ResourceFlowV1,
                authorization_digest: 1,
                target_subject_digest: 2,
                comparator_subject_digest: 3,
                profile_index: 0,
                seed_identity: 4,
                profile_digest: 5,
                world_digest: 6,
                sealed_action: PublicAction::NoOp,
                sealed_target_action_index: 0,
                pre_state: None,
                target_prediction_commitment: None,
                comparator_prediction_commitment: None,
                transition_digest: None,
                transition_ordinal: None,
                disposition: CampaignRowDisposition::InfrastructureIndeterminate,
            })
            .unwrap(),
        );
        let projection = invalid.analysis_projection();
        assert_eq!(projection.candidate, None);
        assert_eq!(projection.comparator, None);
        assert_eq!(invalid.valid_cross_family_row(), None);
    }
}
