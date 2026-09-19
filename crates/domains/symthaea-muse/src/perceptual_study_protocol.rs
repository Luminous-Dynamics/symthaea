// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1A: frozen protocol contract for the first human perceptual study.
//!
//! This module deliberately stops before stimulus-pack construction,
//! participant scheduling, response collection, unblinding, or analysis.
//! Acoustic evidence motivates the study but cannot stand in for human
//! perception; the protocol therefore creates a new, separately governed
//! evidence layer.

use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PERCEPTUAL_STUDY_PROTOCOL_VERSION: &str =
    "mel003-perceptual-study-protocol-v1";
pub const MEL003_C6F_BUNDLE_VERSION: &str = "mel003c6f-sonata-multimodal-bundle-v1";
pub const MEL003_FIXED_SEEDS: [u64; 8] = [3, 11, 23, 41, 59, 79, 97, 127];
pub const CHANCE_PROBABILITY: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PerceptualTaskV1 {
    /// Known A/B references plus hidden X; choose whether X is A or B.
    AbxDiscrimination,
    /// Blind forced choice: which presentation contains more distinct local
    /// accompaniment re-attacks/re-articulations in the returning passage?
    DirectionalRearticulation2Afc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualEndpointRoleV1 {
    Primary,
    KeySecondary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptualEndpointV1 {
    pub task: PerceptualTaskV1,
    pub role: PerceptualEndpointRoleV1,
    /// Exact binary chance rate frozen before responses exist.
    pub chance_probability: f64,
    /// Human-readable interpretation kept narrow enough to match the task.
    pub estimand: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mel003AcousticSubjectBindingV1 {
    /// Exact C6F source revision that defines the pre-perceptual evidence bundle.
    pub c6f_source_commit: String,
    /// Canonical runtime identity of the complete C6F bundle. P1A requires this
    /// to exist before a concrete human protocol can validate.
    pub c6f_bundle_sha256: String,
    pub c6f_bundle_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PerceptualStudyItemV1 {
    pub item_id: String,
    pub seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StimulusExtentV1 {
    /// Preserve the complete four-bar rendered subject. No post-hoc excerpt
    /// boundary may be selected after observing human responses.
    WholeFourBarSubject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoudnessMatchingV1 {
    /// Pairwise integrated-loudness matching; attenuate the louder arm only.
    /// This reduces a trivial level cue without claiming perceptual identity.
    PairwiseIntegratedLufsAttenuationOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StimulusPolicyV1 {
    pub extent: StimulusExtentV1,
    pub synchronized_playhead_required: bool,
    pub loudness_matching: LoudnessMatchingV1,
    pub maximum_attenuation_db: f64,
    pub preserve_pair_alignment: bool,
    pub neutral_presentation_labels_required: bool,
    /// Practice material must not reuse any scored MEL-003 item.
    pub disjoint_practice_material_required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindingAndRandomizationPolicyV1 {
    /// Public commitment to a private schedule key. The key itself is not part
    /// of the protocol or participant-facing material.
    pub randomization_commitment_sha256: String,
    pub schedule_builder_version: String,
    pub balance_ab_label_assignment: bool,
    pub balance_abx_hidden_identity: bool,
    pub balance_directional_left_right_assignment: bool,
    pub reveal_correct_answers_during_scored_collection: bool,
    pub arm_labelled_monitoring_during_collection: bool,
    pub investigator_can_modify_schedule_after_first_response: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticipantPolicyV1 {
    pub minimum_age_years: u8,
    pub informed_consent_required: bool,
    pub pseudonymous_participant_tokens_required: bool,
    pub raw_names_or_contact_details_in_study_dataset_allowed: bool,
    pub stereo_playback_check_required: bool,
    pub task_comprehension_practice_required: bool,
    pub practice_feedback_allowed: bool,
    pub scored_trial_feedback_allowed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleSizePlanV1 {
    /// Participant count is frozen by a separately reviewable planning artifact,
    /// rather than guessed from the acoustic proxy magnitude.
    pub planning_artifact_sha256: String,
    pub planned_completed_participants: usize,
    pub maximum_enrolled_participants: usize,
    pub outcome_adaptive_stopping_allowed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingResponsePolicyV1 {
    /// Keep incomplete/raw records for attrition reporting; exclude the
    /// participant from the primary complete-session analysis if any scored
    /// item response is missing. No outcome-dependent imputation is performed.
    RetainRawExcludeIncompleteSessionNoImputation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimaryAnalysisModelV1 {
    /// Logistic GLMM over participant × item binary responses with crossed
    /// participant and item random intercepts.
    CrossedParticipantItemLogisticRandomIntercepts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecondaryMultiplicityPolicyV1 {
    /// P1A has one key secondary endpoint, but freezing a Holm family keeps the
    /// contract safe if later protocol versions add registered secondary tests.
    HolmWithinRegisteredSecondaryFamily,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisPolicyV1 {
    pub primary_model: PrimaryAnalysisModelV1,
    pub participant_grouping_factor_required: bool,
    pub item_grouping_factor_required: bool,
    pub primary_alternative_is_greater_than_chance: bool,
    pub alpha: f64,
    pub confidence_level: f64,
    pub secondary_multiplicity: SecondaryMultiplicityPolicyV1,
    pub report_item_level_outcomes: bool,
    pub report_participant_level_outcomes: bool,
    pub report_random_effect_variance: bool,
    pub missing_response_policy: MissingResponsePolicyV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ForbiddenPerceptualClaimV1 {
    Preference,
    ArtisticQuality,
    EmotionalImpact,
    StyleIdentity,
    CulturalAuthenticity,
    HumanLikePerformance,
    IndependentAcousticReplication,
    CognitionProductAuthority,
    GeneralizationBeyondRegisteredItemsAndEligiblePopulation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalPerceptualPreregistrationV1 {
    pub registry: String,
    pub record_id: String,
    pub frozen_at_utc: String,
    pub record_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenPerceptualStudyProtocolV1 {
    pub protocol_version: String,
    pub acoustic_subject: Mel003AcousticSubjectBindingV1,
    pub external_preregistration: ExternalPerceptualPreregistrationV1,
    /// Exact frozen analysis specification/code identity, separate from this
    /// protocol's human-readable statistical declarations.
    pub analysis_spec_sha256: String,
    pub items: Vec<PerceptualStudyItemV1>,
    pub endpoints: Vec<PerceptualEndpointV1>,
    pub stimulus: StimulusPolicyV1,
    pub blinding: BlindingAndRandomizationPolicyV1,
    pub participants: ParticipantPolicyV1,
    pub sample_size: SampleSizePlanV1,
    pub analysis: AnalysisPolicyV1,
    pub forbidden_claims: Vec<ForbiddenPerceptualClaimV1>,
    /// P1A is protocol-only. These remain false until separate later tranches
    /// bind a generated stimulus pack and private participant schedule.
    pub stimulus_pack_bound: bool,
    pub participant_schedule_bound: bool,
    pub collection_authorized: bool,
    pub responses_present: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualStudyProtocolIssueV1 {
    WrongProtocolVersion,
    InvalidC6fSourceCommit,
    InvalidDigest { field: String },
    WrongC6fBundleVersion,
    MissingPreregistrationField { field: String },
    WrongItemCount { found: usize },
    EmptyItemId { index: usize },
    DuplicateItemId { item_id: String },
    DuplicateSeed { seed: u64 },
    WrongSeedPanel,
    WrongEndpointCount { found: usize },
    DuplicateEndpoint { task: PerceptualTaskV1 },
    MissingPrimaryAbx,
    MissingKeySecondaryDirectional,
    InvalidChanceProbability { task: PerceptualTaskV1 },
    EmptyEstimand { task: PerceptualTaskV1 },
    InvalidStimulusPolicy,
    InvalidMaximumAttenuation,
    InvalidBlindingPolicy,
    InvalidParticipantPolicy,
    InvalidSampleSizePlan,
    OutcomeAdaptiveStoppingAllowed,
    InvalidAlpha,
    InvalidConfidenceLevel,
    InvalidAnalysisPolicy,
    EmptyForbiddenClaimRegistry,
    DuplicateForbiddenClaim { claim: ForbiddenPerceptualClaimV1 },
    MissingRequiredForbiddenClaim { claim: ForbiddenPerceptualClaimV1 },
    PrematureStimulusPackBinding,
    PrematureScheduleBinding,
    PrematureCollectionAuthority,
    ResponsesPresentInProtocol,
}

impl FrozenPerceptualStudyProtocolV1 {
    pub fn validate(&self) -> Vec<PerceptualStudyProtocolIssueV1> {
        let mut issues = Vec::new();
        if self.protocol_version != PERCEPTUAL_STUDY_PROTOCOL_VERSION {
            issues.push(PerceptualStudyProtocolIssueV1::WrongProtocolVersion);
        }
        if !is_git_sha1(&self.acoustic_subject.c6f_source_commit) {
            issues.push(PerceptualStudyProtocolIssueV1::InvalidC6fSourceCommit);
        }
        for (field, value) in [
            (
                "acoustic_subject.c6f_bundle_sha256",
                self.acoustic_subject.c6f_bundle_sha256.as_str(),
            ),
            ("analysis_spec_sha256", self.analysis_spec_sha256.as_str()),
            (
                "external_preregistration.record_sha256",
                self.external_preregistration.record_sha256.as_str(),
            ),
            (
                "blinding.randomization_commitment_sha256",
                self.blinding.randomization_commitment_sha256.as_str(),
            ),
            (
                "sample_size.planning_artifact_sha256",
                self.sample_size.planning_artifact_sha256.as_str(),
            ),
        ] {
            if !is_sha256(value) {
                issues.push(PerceptualStudyProtocolIssueV1::InvalidDigest {
                    field: field.into(),
                });
            }
        }
        if self.acoustic_subject.c6f_bundle_version != MEL003_C6F_BUNDLE_VERSION {
            issues.push(PerceptualStudyProtocolIssueV1::WrongC6fBundleVersion);
        }
        for (field, value) in [
            (
                "external_preregistration.registry",
                self.external_preregistration.registry.as_str(),
            ),
            (
                "external_preregistration.record_id",
                self.external_preregistration.record_id.as_str(),
            ),
            (
                "external_preregistration.frozen_at_utc",
                self.external_preregistration.frozen_at_utc.as_str(),
            ),
            (
                "blinding.schedule_builder_version",
                self.blinding.schedule_builder_version.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                issues.push(PerceptualStudyProtocolIssueV1::MissingPreregistrationField {
                    field: field.into(),
                });
            }
        }

        validate_items(&self.items, &mut issues);
        validate_endpoints(&self.endpoints, &mut issues);

        if !self.stimulus.synchronized_playhead_required
            || !self.stimulus.preserve_pair_alignment
            || !self.stimulus.neutral_presentation_labels_required
            || !self.stimulus.disjoint_practice_material_required
        {
            issues.push(PerceptualStudyProtocolIssueV1::InvalidStimulusPolicy);
        }
        if !self.stimulus.maximum_attenuation_db.is_finite()
            || self.stimulus.maximum_attenuation_db <= 0.0
        {
            issues.push(PerceptualStudyProtocolIssueV1::InvalidMaximumAttenuation);
        }
        if !self.blinding.balance_ab_label_assignment
            || !self.blinding.balance_abx_hidden_identity
            || !self.blinding.balance_directional_left_right_assignment
            || self.blinding.reveal_correct_answers_during_scored_collection
            || self.blinding.arm_labelled_monitoring_during_collection
            || self.blinding.investigator_can_modify_schedule_after_first_response
        {
            issues.push(PerceptualStudyProtocolIssueV1::InvalidBlindingPolicy);
        }
        if self.participants.minimum_age_years < 18
            || !self.participants.informed_consent_required
            || !self.participants.pseudonymous_participant_tokens_required
            || self.participants.raw_names_or_contact_details_in_study_dataset_allowed
            || !self.participants.stereo_playback_check_required
            || !self.participants.task_comprehension_practice_required
            || !self.participants.practice_feedback_allowed
            || self.participants.scored_trial_feedback_allowed
        {
            issues.push(PerceptualStudyProtocolIssueV1::InvalidParticipantPolicy);
        }
        if self.sample_size.planned_completed_participants == 0
            || self.sample_size.maximum_enrolled_participants
                < self.sample_size.planned_completed_participants
        {
            issues.push(PerceptualStudyProtocolIssueV1::InvalidSampleSizePlan);
        }
        if self.sample_size.outcome_adaptive_stopping_allowed {
            issues.push(PerceptualStudyProtocolIssueV1::OutcomeAdaptiveStoppingAllowed);
        }
        if !self.analysis.alpha.is_finite()
            || self.analysis.alpha <= 0.0
            || self.analysis.alpha >= 0.5
        {
            issues.push(PerceptualStudyProtocolIssueV1::InvalidAlpha);
        }
        if !self.analysis.confidence_level.is_finite()
            || self.analysis.confidence_level <= 0.5
            || self.analysis.confidence_level >= 1.0
        {
            issues.push(PerceptualStudyProtocolIssueV1::InvalidConfidenceLevel);
        }
        if !self.analysis.participant_grouping_factor_required
            || !self.analysis.item_grouping_factor_required
            || !self.analysis.primary_alternative_is_greater_than_chance
            || !self.analysis.report_item_level_outcomes
            || !self.analysis.report_participant_level_outcomes
            || !self.analysis.report_random_effect_variance
        {
            issues.push(PerceptualStudyProtocolIssueV1::InvalidAnalysisPolicy);
        }
        validate_forbidden_claims(&self.forbidden_claims, &mut issues);

        if self.stimulus_pack_bound {
            issues.push(PerceptualStudyProtocolIssueV1::PrematureStimulusPackBinding);
        }
        if self.participant_schedule_bound {
            issues.push(PerceptualStudyProtocolIssueV1::PrematureScheduleBinding);
        }
        if self.collection_authorized {
            issues.push(PerceptualStudyProtocolIssueV1::PrematureCollectionAuthority);
        }
        if self.responses_present {
            issues.push(PerceptualStudyProtocolIssueV1::ResponsesPresentInProtocol);
        }
        issues
    }

    pub fn protocol_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }
}

fn validate_items(
    items: &[PerceptualStudyItemV1],
    issues: &mut Vec<PerceptualStudyProtocolIssueV1>,
) {
    if items.len() != MEL003_FIXED_SEEDS.len() {
        issues.push(PerceptualStudyProtocolIssueV1::WrongItemCount {
            found: items.len(),
        });
    }
    let mut item_ids = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    for (index, item) in items.iter().enumerate() {
        if item.item_id.trim().is_empty() {
            issues.push(PerceptualStudyProtocolIssueV1::EmptyItemId { index });
        } else if !item_ids.insert(item.item_id.clone()) {
            issues.push(PerceptualStudyProtocolIssueV1::DuplicateItemId {
                item_id: item.item_id.clone(),
            });
        }
        if !seeds.insert(item.seed) {
            issues.push(PerceptualStudyProtocolIssueV1::DuplicateSeed { seed: item.seed });
        }
    }
    let expected: BTreeSet<_> = MEL003_FIXED_SEEDS.into_iter().collect();
    if seeds != expected {
        issues.push(PerceptualStudyProtocolIssueV1::WrongSeedPanel);
    }
}

fn validate_endpoints(
    endpoints: &[PerceptualEndpointV1],
    issues: &mut Vec<PerceptualStudyProtocolIssueV1>,
) {
    if endpoints.len() != 2 {
        issues.push(PerceptualStudyProtocolIssueV1::WrongEndpointCount {
            found: endpoints.len(),
        });
    }
    let mut seen = BTreeSet::new();
    let mut primary_abx = false;
    let mut secondary_directional = false;
    for endpoint in endpoints {
        if !seen.insert(endpoint.task) {
            issues.push(PerceptualStudyProtocolIssueV1::DuplicateEndpoint {
                task: endpoint.task,
            });
        }
        if endpoint.chance_probability != CHANCE_PROBABILITY {
            issues.push(PerceptualStudyProtocolIssueV1::InvalidChanceProbability {
                task: endpoint.task,
            });
        }
        if endpoint.estimand.trim().is_empty() {
            issues.push(PerceptualStudyProtocolIssueV1::EmptyEstimand {
                task: endpoint.task,
            });
        }
        if endpoint.task == PerceptualTaskV1::AbxDiscrimination
            && endpoint.role == PerceptualEndpointRoleV1::Primary
        {
            primary_abx = true;
        }
        if endpoint.task == PerceptualTaskV1::DirectionalRearticulation2Afc
            && endpoint.role == PerceptualEndpointRoleV1::KeySecondary
        {
            secondary_directional = true;
        }
    }
    if !primary_abx {
        issues.push(PerceptualStudyProtocolIssueV1::MissingPrimaryAbx);
    }
    if !secondary_directional {
        issues.push(PerceptualStudyProtocolIssueV1::MissingKeySecondaryDirectional);
    }
}

fn validate_forbidden_claims(
    claims: &[ForbiddenPerceptualClaimV1],
    issues: &mut Vec<PerceptualStudyProtocolIssueV1>,
) {
    if claims.is_empty() {
        issues.push(PerceptualStudyProtocolIssueV1::EmptyForbiddenClaimRegistry);
        return;
    }
    let mut seen = BTreeSet::new();
    for &claim in claims {
        if !seen.insert(claim) {
            issues.push(PerceptualStudyProtocolIssueV1::DuplicateForbiddenClaim { claim });
        }
    }
    for claim in [
        ForbiddenPerceptualClaimV1::Preference,
        ForbiddenPerceptualClaimV1::ArtisticQuality,
        ForbiddenPerceptualClaimV1::EmotionalImpact,
        ForbiddenPerceptualClaimV1::StyleIdentity,
        ForbiddenPerceptualClaimV1::CulturalAuthenticity,
        ForbiddenPerceptualClaimV1::HumanLikePerformance,
        ForbiddenPerceptualClaimV1::IndependentAcousticReplication,
        ForbiddenPerceptualClaimV1::CognitionProductAuthority,
        ForbiddenPerceptualClaimV1::GeneralizationBeyondRegisteredItemsAndEligiblePopulation,
    ] {
        if !seen.contains(&claim) {
            issues.push(PerceptualStudyProtocolIssueV1::MissingRequiredForbiddenClaim { claim });
        }
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_git_sha1(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const COMMIT: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn protocol() -> FrozenPerceptualStudyProtocolV1 {
        FrozenPerceptualStudyProtocolV1 {
            protocol_version: PERCEPTUAL_STUDY_PROTOCOL_VERSION.into(),
            acoustic_subject: Mel003AcousticSubjectBindingV1 {
                c6f_source_commit: COMMIT.into(),
                c6f_bundle_sha256: DIGEST.into(),
                c6f_bundle_version: MEL003_C6F_BUNDLE_VERSION.into(),
            },
            external_preregistration: ExternalPerceptualPreregistrationV1 {
                registry: "OSF".into(),
                record_id: "mel003-p1".into(),
                frozen_at_utc: "2026-09-19T00:00:00Z".into(),
                record_sha256: DIGEST.into(),
            },
            analysis_spec_sha256: DIGEST.into(),
            items: MEL003_FIXED_SEEDS
                .into_iter()
                .map(|seed| PerceptualStudyItemV1 {
                    item_id: format!("sonata-seed-{seed}"),
                    seed,
                })
                .collect(),
            endpoints: vec![
                PerceptualEndpointV1 {
                    task: PerceptualTaskV1::AbxDiscrimination,
                    role: PerceptualEndpointRoleV1::Primary,
                    chance_probability: 0.5,
                    estimand: "probability of correctly identifying hidden X across registered participants and items".into(),
                },
                PerceptualEndpointV1 {
                    task: PerceptualTaskV1::DirectionalRearticulation2Afc,
                    role: PerceptualEndpointRoleV1::KeySecondary,
                    chance_probability: 0.5,
                    estimand: "probability of selecting the intervention as containing more distinct accompaniment re-articulations in the returning passage".into(),
                },
            ],
            stimulus: StimulusPolicyV1 {
                extent: StimulusExtentV1::WholeFourBarSubject,
                synchronized_playhead_required: true,
                loudness_matching: LoudnessMatchingV1::PairwiseIntegratedLufsAttenuationOnly,
                maximum_attenuation_db: 12.0,
                preserve_pair_alignment: true,
                neutral_presentation_labels_required: true,
                disjoint_practice_material_required: true,
            },
            blinding: BlindingAndRandomizationPolicyV1 {
                randomization_commitment_sha256: DIGEST.into(),
                schedule_builder_version: "mel003-perceptual-schedule-v1".into(),
                balance_ab_label_assignment: true,
                balance_abx_hidden_identity: true,
                balance_directional_left_right_assignment: true,
                reveal_correct_answers_during_scored_collection: false,
                arm_labelled_monitoring_during_collection: false,
                investigator_can_modify_schedule_after_first_response: false,
            },
            participants: ParticipantPolicyV1 {
                minimum_age_years: 18,
                informed_consent_required: true,
                pseudonymous_participant_tokens_required: true,
                raw_names_or_contact_details_in_study_dataset_allowed: false,
                stereo_playback_check_required: true,
                task_comprehension_practice_required: true,
                practice_feedback_allowed: true,
                scored_trial_feedback_allowed: false,
            },
            sample_size: SampleSizePlanV1 {
                planning_artifact_sha256: DIGEST.into(),
                planned_completed_participants: 48,
                maximum_enrolled_participants: 56,
                outcome_adaptive_stopping_allowed: false,
            },
            analysis: AnalysisPolicyV1 {
                primary_model: PrimaryAnalysisModelV1::CrossedParticipantItemLogisticRandomIntercepts,
                participant_grouping_factor_required: true,
                item_grouping_factor_required: true,
                primary_alternative_is_greater_than_chance: true,
                alpha: 0.05,
                confidence_level: 0.95,
                secondary_multiplicity: SecondaryMultiplicityPolicyV1::HolmWithinRegisteredSecondaryFamily,
                report_item_level_outcomes: true,
                report_participant_level_outcomes: true,
                report_random_effect_variance: true,
                missing_response_policy: MissingResponsePolicyV1::RetainRawExcludeIncompleteSessionNoImputation,
            },
            forbidden_claims: vec![
                ForbiddenPerceptualClaimV1::Preference,
                ForbiddenPerceptualClaimV1::ArtisticQuality,
                ForbiddenPerceptualClaimV1::EmotionalImpact,
                ForbiddenPerceptualClaimV1::StyleIdentity,
                ForbiddenPerceptualClaimV1::CulturalAuthenticity,
                ForbiddenPerceptualClaimV1::HumanLikePerformance,
                ForbiddenPerceptualClaimV1::IndependentAcousticReplication,
                ForbiddenPerceptualClaimV1::CognitionProductAuthority,
                ForbiddenPerceptualClaimV1::GeneralizationBeyondRegisteredItemsAndEligiblePopulation,
            ],
            stimulus_pack_bound: false,
            participant_schedule_bound: false,
            collection_authorized: false,
            responses_present: false,
        }
    }

    #[test]
    fn valid_contract_is_protocol_only_and_digestible() {
        let protocol = protocol();
        assert!(protocol.validate().is_empty());
        let digest = protocol.protocol_sha256().unwrap();
        assert_eq!(digest.len(), 64);
        assert!(!protocol.collection_authorized);
        assert!(!protocol.responses_present);
    }

    #[test]
    fn preference_cannot_replace_discriminability_primary() {
        let mut protocol = protocol();
        protocol.endpoints[0].role = PerceptualEndpointRoleV1::KeySecondary;
        let issues = protocol.validate();
        assert!(issues.contains(&PerceptualStudyProtocolIssueV1::MissingPrimaryAbx));
    }

    #[test]
    fn acoustic_seed_panel_cannot_be_expanded_after_the_fact() {
        let mut protocol = protocol();
        protocol.items.push(PerceptualStudyItemV1 {
            item_id: "post-hoc-seed".into(),
            seed: 999,
        });
        let issues = protocol.validate();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            PerceptualStudyProtocolIssueV1::WrongItemCount { .. }
        )));
        assert!(issues.contains(&PerceptualStudyProtocolIssueV1::WrongSeedPanel));
    }

    #[test]
    fn collection_cannot_be_authorized_in_p1a() {
        let mut protocol = protocol();
        protocol.collection_authorized = true;
        protocol.stimulus_pack_bound = true;
        protocol.participant_schedule_bound = true;
        let issues = protocol.validate();
        assert!(issues.contains(&PerceptualStudyProtocolIssueV1::PrematureCollectionAuthority));
        assert!(issues.contains(&PerceptualStudyProtocolIssueV1::PrematureStimulusPackBinding));
        assert!(issues.contains(&PerceptualStudyProtocolIssueV1::PrematureScheduleBinding));
    }

    #[test]
    fn outcome_adaptive_stopping_and_arm_labelled_monitoring_fail_closed() {
        let mut protocol = protocol();
        protocol.sample_size.outcome_adaptive_stopping_allowed = true;
        protocol.blinding.arm_labelled_monitoring_during_collection = true;
        let issues = protocol.validate();
        assert!(issues.contains(&PerceptualStudyProtocolIssueV1::OutcomeAdaptiveStoppingAllowed));
        assert!(issues.contains(&PerceptualStudyProtocolIssueV1::InvalidBlindingPolicy));
    }

    #[test]
    fn required_nonclaims_cannot_be_silently_removed() {
        let mut protocol = protocol();
        protocol
            .forbidden_claims
            .retain(|claim| *claim != ForbiddenPerceptualClaimV1::ArtisticQuality);
        assert!(protocol.validate().contains(
            &PerceptualStudyProtocolIssueV1::MissingRequiredForbiddenClaim {
                claim: ForbiddenPerceptualClaimV1::ArtisticQuality,
            }
        ));
    }
}
