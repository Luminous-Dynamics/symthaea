// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1D: frozen participant scheduling and private randomization audit
//! for the first blinded perceptual study.
//!
//! The schedule is generated for the complete maximum-enrollment slot pool
//! before collection. Attrition therefore leaves unused schedules; it never
//! changes future assignments.
//!
//! Task order is a participant-level factor. The three item-level binary
//! factors (A/B reference mapping, ABX hidden-X label, directional left/right
//! mapping) form eight factorial cells. Because MEL-003 has exactly eight
//! registered items, every participant receives every factorial cell exactly
//! once. A secret-derived participant ranking rotates cells and item positions,
//! keeping every item balanced across the enrollment pool to within one count.

use crate::evidence_digest::{
    canonical_json_sha256, sha256_hex,
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
        StimulusArmV1,
    },
    perceptual_study_protocol::{
        FrozenPerceptualStudyProtocolV1, MEL003_FIXED_SEEDS,
    },
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

pub const PERCEPTUAL_PARTICIPANT_SCHEDULE_VERSION: &str =
    "mel003-perceptual-participant-schedule-v1";
pub const PERCEPTUAL_SCHEDULE_BUILDER_VERSION: &str =
    "mel003-perceptual-schedule-builder-v1";
pub const FACTORIAL_CELL_COUNT: usize = 8;
pub const SECOND_TASK_ITEM_SHIFT: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TaskBlockOrderV1 {
    AbxThenDirectional,
    DirectionalThenAbx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AbxReferenceMappingV1 {
    BaselineIsA,
    InterventionIsA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HiddenXLabelV1 {
    XIsA,
    XIsB,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DirectionalSideV1 {
    InterventionLeft,
    InterventionRight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FactorialAssignmentV1 {
    pub abx_reference_mapping: AbxReferenceMappingV1,
    pub hidden_x_label: HiddenXLabelV1,
    pub directional_side: DirectionalSideV1,
}

impl FactorialAssignmentV1 {
    pub fn from_index(index: usize) -> Option<Self> {
        if index >= FACTORIAL_CELL_COUNT {
            return None;
        }
        Some(Self {
            abx_reference_mapping: if index & 0b001 == 0 {
                AbxReferenceMappingV1::BaselineIsA
            } else {
                AbxReferenceMappingV1::InterventionIsA
            },
            hidden_x_label: if index & 0b010 == 0 {
                HiddenXLabelV1::XIsA
            } else {
                HiddenXLabelV1::XIsB
            },
            directional_side: if index & 0b100 == 0 {
                DirectionalSideV1::InterventionLeft
            } else {
                DirectionalSideV1::InterventionRight
            },
        })
    }

    pub fn index(self) -> usize {
        let a = match self.abx_reference_mapping {
            AbxReferenceMappingV1::BaselineIsA => 0,
            AbxReferenceMappingV1::InterventionIsA => 1,
        };
        let x = match self.hidden_x_label {
            HiddenXLabelV1::XIsA => 0,
            HiddenXLabelV1::XIsB => 2,
        };
        let side = match self.directional_side {
            DirectionalSideV1::InterventionLeft => 0,
            DirectionalSideV1::InterventionRight => 4,
        };
        a + x + side
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptualCohortSlotsV1 {
    pub cohort_id: String,
    /// Pseudonymous enrollment slots only. Names/contact details do not belong
    /// in this artifact.
    pub participant_tokens: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicAbxTrialV1 {
    pub trial_id: String,
    pub item_id: String,
    pub seed: u64,
    pub a_clip_id: String,
    pub b_clip_id: String,
    pub x_clip_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicDirectionalTrialV1 {
    pub trial_id: String,
    pub item_id: String,
    pub seed: u64,
    pub left_clip_id: String,
    pub right_clip_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicParticipantScheduleV1 {
    pub participant_token: String,
    pub task_order: TaskBlockOrderV1,
    pub abx_trials: Vec<PublicAbxTrialV1>,
    pub directional_trials: Vec<PublicDirectionalTrialV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptualParticipantScheduleBookV1 {
    pub schedule_version: String,
    pub builder_version: String,
    pub protocol_sha256: String,
    pub stimulus_pack_sha256: String,
    pub randomization_commitment_sha256: String,
    pub cohort_id: String,
    pub participant_count: usize,
    /// Commitment to the private arm-mapping audit. The audit itself must not
    /// be handed to participants or exposed during blinded collection.
    pub private_audit_sha256: String,
    pub schedules: Vec<PublicParticipantScheduleV1>,
    pub responses_present: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivateTrialMappingV1 {
    pub participant_token: String,
    pub item_id: String,
    pub seed: u64,
    pub factorial_assignment: FactorialAssignmentV1,
    pub abx_position: usize,
    pub directional_position: usize,
    pub baseline_output_sha256: String,
    pub intervention_output_sha256: String,
    pub a_arm: StimulusArmV1,
    pub b_arm: StimulusArmV1,
    pub x_arm: StimulusArmV1,
    pub left_arm: StimulusArmV1,
    pub right_arm: StimulusArmV1,
    pub a_clip_id: String,
    pub b_clip_id: String,
    pub x_clip_id: String,
    pub left_clip_id: String,
    pub right_clip_id: String,
    pub abx_trial_id: String,
    pub directional_trial_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivateParticipantScheduleAuditV1 {
    pub participant_token: String,
    pub secret_rank_index: usize,
    pub task_order: TaskBlockOrderV1,
    pub trials: Vec<PrivateTrialMappingV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptualParticipantScheduleAuditV1 {
    pub schedule_version: String,
    pub builder_version: String,
    pub protocol_sha256: String,
    pub stimulus_pack_sha256: String,
    pub cohort_id: String,
    pub participants: Vec<PrivateParticipantScheduleAuditV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualParticipantScheduleIssueV1 {
    InvalidProtocol,
    InvalidStimulusPack,
    ProtocolSerializationFailed,
    ProtocolDigestMismatch,
    StimulusPackSerializationFailed,
    StimulusPackDigestMismatch,
    WrongScheduleVersion,
    WrongBuilderVersion,
    ProtocolBuilderVersionMismatch,
    RandomizationCommitmentMismatch,
    EmptyCohortId,
    ParticipantCountMismatch { found: usize, required: usize },
    EmptyParticipantToken { index: usize },
    DuplicateParticipantToken { participant_token: String },
    ScheduleCountMismatch { found: usize, expected: usize },
    DuplicateScheduleParticipant { participant_token: String },
    MissingScheduleParticipant { participant_token: String },
    UnexpectedScheduleParticipant { participant_token: String },
    InvalidPrivateAuditDigest,
    PrivateAuditDigestMismatch,
    AuditIdentityMismatch,
    MissingPrivateAuditParticipant { participant_token: String },
    UnexpectedPrivateAuditParticipant { participant_token: String },
    WrongSecretRankIndex { participant_token: String },
    TaskOrderMismatch { participant_token: String },
    TaskOrderImbalance { abx_first: usize, directional_first: usize },
    WrongTrialCount { participant_token: String, task: String, found: usize },
    DuplicateTrialId { trial_id: String },
    DuplicateOpaqueClipId { clip_id: String },
    WrongItemOrder { participant_token: String, task: String, position: usize },
    DuplicateParticipantItem { participant_token: String, task: String, seed: u64 },
    WrongParticipantSeedPanel { participant_token: String, task: String },
    SameItemOrdinalAcrossTaskBlocks { participant_token: String, seed: u64 },
    PublicPrivateTrialMismatch { participant_token: String, seed: u64, task: String },
    FactorialCellNotUniqueWithinParticipant { participant_token: String, cell: usize },
    MissingFactorialCellWithinParticipant { participant_token: String, cell: usize },
    FactorialItemImbalance { seed: u64, minimum: usize, maximum: usize },
    ItemPositionImbalance { seed: u64, task: String, minimum: usize, maximum: usize },
    ResponsesPresent,
}

pub fn build_perceptual_participant_schedule(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    secret_key: [u8; 32],
) -> Result<
    (
        PerceptualParticipantScheduleBookV1,
        PerceptualParticipantScheduleAuditV1,
    ),
    Vec<PerceptualParticipantScheduleIssueV1>,
> {
    let mut issues = validate_inputs(protocol, stimulus_pack, render_binding, cohort);
    if protocol.blinding.schedule_builder_version != PERCEPTUAL_SCHEDULE_BUILDER_VERSION {
        issues.push(PerceptualParticipantScheduleIssueV1::ProtocolBuilderVersionMismatch);
    }
    if sha256_hex(&secret_key) != protocol.blinding.randomization_commitment_sha256 {
        issues.push(PerceptualParticipantScheduleIssueV1::RandomizationCommitmentMismatch);
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let protocol_sha256 = canonical_json_sha256(protocol)
        .expect("validated protocol must serialize canonically");
    let stimulus_pack_sha256 = canonical_json_sha256(stimulus_pack)
        .expect("validated stimulus pack must serialize canonically");

    let mut ranked: Vec<_> = cohort
        .participant_tokens
        .iter()
        .map(|token| (derive_rank(&secret_key, token), token.clone()))
        .collect();
    ranked.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

    let pack_by_seed: BTreeMap<_, _> = stimulus_pack
        .items
        .iter()
        .map(|pair| (pair.seed, pair))
        .collect();

    let mut public_schedules = Vec::with_capacity(ranked.len());
    let mut private_participants = Vec::with_capacity(ranked.len());

    for (rank_index, (_, participant_token)) in ranked.iter().enumerate() {
        let task_order = if rank_index % 2 == 0 {
            TaskBlockOrderV1::AbxThenDirectional
        } else {
            TaskBlockOrderV1::DirectionalThenAbx
        };

        let mut private_trials = Vec::with_capacity(MEL003_FIXED_SEEDS.len());
        let mut audit_by_seed = BTreeMap::new();
        for (item_index, seed) in MEL003_FIXED_SEEDS.iter().copied().enumerate() {
            let pair = pack_by_seed
                .get(&seed)
                .expect("validated pack contains every registered seed");
            let assignment = FactorialAssignmentV1::from_index(
                (rank_index + item_index) % FACTORIAL_CELL_COUNT,
            )
            .expect("modulo factorial cell count is valid");
            let (a_arm, b_arm) = match assignment.abx_reference_mapping {
                AbxReferenceMappingV1::BaselineIsA => {
                    (StimulusArmV1::Baseline, StimulusArmV1::Intervention)
                }
                AbxReferenceMappingV1::InterventionIsA => {
                    (StimulusArmV1::Intervention, StimulusArmV1::Baseline)
                }
            };
            let x_arm = match assignment.hidden_x_label {
                HiddenXLabelV1::XIsA => a_arm,
                HiddenXLabelV1::XIsB => b_arm,
            };
            let (left_arm, right_arm) = match assignment.directional_side {
                DirectionalSideV1::InterventionLeft => {
                    (StimulusArmV1::Intervention, StimulusArmV1::Baseline)
                }
                DirectionalSideV1::InterventionRight => {
                    (StimulusArmV1::Baseline, StimulusArmV1::Intervention)
                }
            };

            let a_clip_id = opaque_id(&secret_key, "abx-a", participant_token, seed);
            let b_clip_id = opaque_id(&secret_key, "abx-b", participant_token, seed);
            let x_clip_id = opaque_id(&secret_key, "abx-x", participant_token, seed);
            let left_clip_id = opaque_id(&secret_key, "direction-left", participant_token, seed);
            let right_clip_id = opaque_id(&secret_key, "direction-right", participant_token, seed);
            let abx_trial_id = opaque_id(&secret_key, "abx-trial", participant_token, seed);
            let directional_trial_id =
                opaque_id(&secret_key, "direction-trial", participant_token, seed);

            let abx_position = (item_index + MEL003_FIXED_SEEDS.len() - rank_index % MEL003_FIXED_SEEDS.len())
                % MEL003_FIXED_SEEDS.len();
            let directional_position = (
                item_index
                    + MEL003_FIXED_SEEDS.len()
                    - (rank_index + SECOND_TASK_ITEM_SHIFT) % MEL003_FIXED_SEEDS.len()
            ) % MEL003_FIXED_SEEDS.len();

            let trial = PrivateTrialMappingV1 {
                participant_token: participant_token.clone(),
                item_id: pair.item_id.clone(),
                seed,
                factorial_assignment: assignment,
                abx_position,
                directional_position,
                baseline_output_sha256: pair.baseline.output_sha256.clone(),
                intervention_output_sha256: pair.intervention.output_sha256.clone(),
                a_arm,
                b_arm,
                x_arm,
                left_arm,
                right_arm,
                a_clip_id,
                b_clip_id,
                x_clip_id,
                left_clip_id,
                right_clip_id,
                abx_trial_id,
                directional_trial_id,
            };
            audit_by_seed.insert(seed, trial.clone());
            private_trials.push(trial);
        }
        private_trials.sort_by_key(|trial| trial.seed);

        let mut abx_trials = Vec::with_capacity(MEL003_FIXED_SEEDS.len());
        let mut directional_trials = Vec::with_capacity(MEL003_FIXED_SEEDS.len());
        for position in 0..MEL003_FIXED_SEEDS.len() {
            let abx_item_index =
                (position + rank_index) % MEL003_FIXED_SEEDS.len();
            let abx_seed = MEL003_FIXED_SEEDS[abx_item_index];
            let abx = audit_by_seed
                .get(&abx_seed)
                .expect("audit contains each seed");
            abx_trials.push(PublicAbxTrialV1 {
                trial_id: abx.abx_trial_id.clone(),
                item_id: abx.item_id.clone(),
                seed: abx.seed,
                a_clip_id: abx.a_clip_id.clone(),
                b_clip_id: abx.b_clip_id.clone(),
                x_clip_id: abx.x_clip_id.clone(),
            });

            let directional_item_index =
                (position + rank_index + SECOND_TASK_ITEM_SHIFT) % MEL003_FIXED_SEEDS.len();
            let directional_seed = MEL003_FIXED_SEEDS[directional_item_index];
            let directional = audit_by_seed
                .get(&directional_seed)
                .expect("audit contains each seed");
            directional_trials.push(PublicDirectionalTrialV1 {
                trial_id: directional.directional_trial_id.clone(),
                item_id: directional.item_id.clone(),
                seed: directional.seed,
                left_clip_id: directional.left_clip_id.clone(),
                right_clip_id: directional.right_clip_id.clone(),
            });
        }

        public_schedules.push(PublicParticipantScheduleV1 {
            participant_token: participant_token.clone(),
            task_order,
            abx_trials,
            directional_trials,
        });
        private_participants.push(PrivateParticipantScheduleAuditV1 {
            participant_token: participant_token.clone(),
            secret_rank_index: rank_index,
            task_order,
            trials: private_trials,
        });
    }

    public_schedules.sort_by(|left, right| left.participant_token.cmp(&right.participant_token));
    private_participants.sort_by(|left, right| left.participant_token.cmp(&right.participant_token));

    let audit = PerceptualParticipantScheduleAuditV1 {
        schedule_version: PERCEPTUAL_PARTICIPANT_SCHEDULE_VERSION.into(),
        builder_version: PERCEPTUAL_SCHEDULE_BUILDER_VERSION.into(),
        protocol_sha256: protocol_sha256.clone(),
        stimulus_pack_sha256: stimulus_pack_sha256.clone(),
        cohort_id: cohort.cohort_id.clone(),
        participants: private_participants,
    };
    let private_audit_sha256 = canonical_json_sha256(&audit)
        .expect("private audit must serialize canonically");
    let book = PerceptualParticipantScheduleBookV1 {
        schedule_version: PERCEPTUAL_PARTICIPANT_SCHEDULE_VERSION.into(),
        builder_version: PERCEPTUAL_SCHEDULE_BUILDER_VERSION.into(),
        protocol_sha256,
        stimulus_pack_sha256,
        randomization_commitment_sha256: protocol
            .blinding
            .randomization_commitment_sha256
            .clone(),
        cohort_id: cohort.cohort_id.clone(),
        participant_count: cohort.participant_tokens.len(),
        private_audit_sha256,
        schedules: public_schedules,
        responses_present: false,
    };

    let validation = validate_perceptual_participant_schedule(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        &book,
        Some(&audit),
    );
    if validation.is_empty() {
        Ok((book, audit))
    } else {
        Err(validation)
    }
}

pub fn validate_perceptual_participant_schedule(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    book: &PerceptualParticipantScheduleBookV1,
    audit: Option<&PerceptualParticipantScheduleAuditV1>,
) -> Vec<PerceptualParticipantScheduleIssueV1> {
    let mut issues = validate_inputs(protocol, stimulus_pack, render_binding, cohort);
    if book.schedule_version != PERCEPTUAL_PARTICIPANT_SCHEDULE_VERSION {
        issues.push(PerceptualParticipantScheduleIssueV1::WrongScheduleVersion);
    }
    if book.builder_version != PERCEPTUAL_SCHEDULE_BUILDER_VERSION {
        issues.push(PerceptualParticipantScheduleIssueV1::WrongBuilderVersion);
    }
    if protocol.blinding.schedule_builder_version != PERCEPTUAL_SCHEDULE_BUILDER_VERSION {
        issues.push(PerceptualParticipantScheduleIssueV1::ProtocolBuilderVersionMismatch);
    }
    match canonical_json_sha256(protocol) {
        Ok(value) if value == book.protocol_sha256 => {}
        Ok(_) => issues.push(PerceptualParticipantScheduleIssueV1::ProtocolDigestMismatch),
        Err(_) => issues.push(PerceptualParticipantScheduleIssueV1::ProtocolSerializationFailed),
    }
    match canonical_json_sha256(stimulus_pack) {
        Ok(value) if value == book.stimulus_pack_sha256 => {}
        Ok(_) => issues.push(PerceptualParticipantScheduleIssueV1::StimulusPackDigestMismatch),
        Err(_) => {
            issues.push(PerceptualParticipantScheduleIssueV1::StimulusPackSerializationFailed)
        }
    }
    if book.randomization_commitment_sha256
        != protocol.blinding.randomization_commitment_sha256
    {
        issues.push(PerceptualParticipantScheduleIssueV1::RandomizationCommitmentMismatch);
    }
    if book.cohort_id != cohort.cohort_id || book.cohort_id.trim().is_empty() {
        issues.push(PerceptualParticipantScheduleIssueV1::EmptyCohortId);
    }
    if book.participant_count != cohort.participant_tokens.len() {
        issues.push(PerceptualParticipantScheduleIssueV1::ParticipantCountMismatch {
            found: book.participant_count,
            required: cohort.participant_tokens.len(),
        });
    }
    if book.schedules.len() != cohort.participant_tokens.len() {
        issues.push(PerceptualParticipantScheduleIssueV1::ScheduleCountMismatch {
            found: book.schedules.len(),
            expected: cohort.participant_tokens.len(),
        });
    }
    if book.responses_present {
        issues.push(PerceptualParticipantScheduleIssueV1::ResponsesPresent);
    }

    let cohort_tokens: BTreeSet<_> = cohort.participant_tokens.iter().cloned().collect();
    let mut scheduled_tokens = BTreeSet::new();
    let mut task_order_counts = [0usize; 2];
    let mut trial_ids = BTreeSet::new();
    let mut clip_ids = BTreeSet::new();
    let mut abx_positions: BTreeMap<u64, [usize; 8]> = BTreeMap::new();
    let mut directional_positions: BTreeMap<u64, [usize; 8]> = BTreeMap::new();

    for schedule in &book.schedules {
        if !scheduled_tokens.insert(schedule.participant_token.clone()) {
            issues.push(PerceptualParticipantScheduleIssueV1::DuplicateScheduleParticipant {
                participant_token: schedule.participant_token.clone(),
            });
        }
        if !cohort_tokens.contains(&schedule.participant_token) {
            issues.push(PerceptualParticipantScheduleIssueV1::UnexpectedScheduleParticipant {
                participant_token: schedule.participant_token.clone(),
            });
        }
        match schedule.task_order {
            TaskBlockOrderV1::AbxThenDirectional => task_order_counts[0] += 1,
            TaskBlockOrderV1::DirectionalThenAbx => task_order_counts[1] += 1,
        }
        validate_public_blocks(
            schedule,
            &mut trial_ids,
            &mut clip_ids,
            &mut abx_positions,
            &mut directional_positions,
            &mut issues,
        );
    }
    for token in cohort_tokens.difference(&scheduled_tokens) {
        issues.push(PerceptualParticipantScheduleIssueV1::MissingScheduleParticipant {
            participant_token: token.clone(),
        });
    }
    if task_order_counts[0].abs_diff(task_order_counts[1]) > 1 {
        issues.push(PerceptualParticipantScheduleIssueV1::TaskOrderImbalance {
            abx_first: task_order_counts[0],
            directional_first: task_order_counts[1],
        });
    }
    validate_position_balance("abx", &abx_positions, &mut issues);
    validate_position_balance("directional", &directional_positions, &mut issues);

    if !is_sha256(&book.private_audit_sha256) {
        issues.push(PerceptualParticipantScheduleIssueV1::InvalidPrivateAuditDigest);
    }
    match audit {
        None => {}
        Some(audit) => {
            match canonical_json_sha256(audit) {
                Ok(value) if value == book.private_audit_sha256 => {}
                _ => issues.push(PerceptualParticipantScheduleIssueV1::PrivateAuditDigestMismatch),
            }
            validate_private_audit(
                protocol,
                stimulus_pack,
                book,
                audit,
                &mut issues,
            );
        }
    }
    issues
}

fn validate_inputs(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
) -> Vec<PerceptualParticipantScheduleIssueV1> {
    let mut issues = Vec::new();
    if !protocol.validate().is_empty() {
        issues.push(PerceptualParticipantScheduleIssueV1::InvalidProtocol);
    }
    if !stimulus_pack.validate(protocol, render_binding).is_empty() {
        issues.push(PerceptualParticipantScheduleIssueV1::InvalidStimulusPack);
    }
    if cohort.cohort_id.trim().is_empty() {
        issues.push(PerceptualParticipantScheduleIssueV1::EmptyCohortId);
    }
    if cohort.participant_tokens.len() != protocol.sample_size.maximum_enrolled_participants {
        issues.push(PerceptualParticipantScheduleIssueV1::ParticipantCountMismatch {
            found: cohort.participant_tokens.len(),
            required: protocol.sample_size.maximum_enrolled_participants,
        });
    }
    let mut tokens = BTreeSet::new();
    for (index, token) in cohort.participant_tokens.iter().enumerate() {
        if token.trim().is_empty() {
            issues.push(PerceptualParticipantScheduleIssueV1::EmptyParticipantToken { index });
        } else if !tokens.insert(token.clone()) {
            issues.push(PerceptualParticipantScheduleIssueV1::DuplicateParticipantToken {
                participant_token: token.clone(),
            });
        }
    }
    issues
}

fn validate_public_blocks(
    schedule: &PublicParticipantScheduleV1,
    trial_ids: &mut BTreeSet<String>,
    clip_ids: &mut BTreeSet<String>,
    abx_positions: &mut BTreeMap<u64, [usize; 8]>,
    directional_positions: &mut BTreeMap<u64, [usize; 8]>,
    issues: &mut Vec<PerceptualParticipantScheduleIssueV1>,
) {
    if schedule.abx_trials.len() != MEL003_FIXED_SEEDS.len() {
        issues.push(PerceptualParticipantScheduleIssueV1::WrongTrialCount {
            participant_token: schedule.participant_token.clone(),
            task: "abx".into(),
            found: schedule.abx_trials.len(),
        });
    }
    if schedule.directional_trials.len() != MEL003_FIXED_SEEDS.len() {
        issues.push(PerceptualParticipantScheduleIssueV1::WrongTrialCount {
            participant_token: schedule.participant_token.clone(),
            task: "directional".into(),
            found: schedule.directional_trials.len(),
        });
    }

    let mut abx_seeds = BTreeSet::new();
    let mut directional_seeds = BTreeSet::new();
    let mut abx_position_by_seed = BTreeMap::new();
    let mut directional_position_by_seed = BTreeMap::new();

    for (position, trial) in schedule.abx_trials.iter().enumerate() {
        if !trial_ids.insert(trial.trial_id.clone()) {
            issues.push(PerceptualParticipantScheduleIssueV1::DuplicateTrialId {
                trial_id: trial.trial_id.clone(),
            });
        }
        for clip_id in [&trial.a_clip_id, &trial.b_clip_id, &trial.x_clip_id] {
            if !clip_ids.insert(clip_id.clone()) {
                issues.push(PerceptualParticipantScheduleIssueV1::DuplicateOpaqueClipId {
                    clip_id: clip_id.clone(),
                });
            }
        }
        if !abx_seeds.insert(trial.seed) {
            issues.push(PerceptualParticipantScheduleIssueV1::DuplicateParticipantItem {
                participant_token: schedule.participant_token.clone(),
                task: "abx".into(),
                seed: trial.seed,
            });
        }
        abx_position_by_seed.insert(trial.seed, position);
        if position < 8 {
            abx_positions.entry(trial.seed).or_insert([0; 8])[position] += 1;
        }
    }
    for (position, trial) in schedule.directional_trials.iter().enumerate() {
        if !trial_ids.insert(trial.trial_id.clone()) {
            issues.push(PerceptualParticipantScheduleIssueV1::DuplicateTrialId {
                trial_id: trial.trial_id.clone(),
            });
        }
        for clip_id in [&trial.left_clip_id, &trial.right_clip_id] {
            if !clip_ids.insert(clip_id.clone()) {
                issues.push(PerceptualParticipantScheduleIssueV1::DuplicateOpaqueClipId {
                    clip_id: clip_id.clone(),
                });
            }
        }
        if !directional_seeds.insert(trial.seed) {
            issues.push(PerceptualParticipantScheduleIssueV1::DuplicateParticipantItem {
                participant_token: schedule.participant_token.clone(),
                task: "directional".into(),
                seed: trial.seed,
            });
        }
        directional_position_by_seed.insert(trial.seed, position);
        if position < 8 {
            directional_positions
                .entry(trial.seed)
                .or_insert([0; 8])[position] += 1;
        }
    }
    let expected: BTreeSet<_> = MEL003_FIXED_SEEDS.into_iter().collect();
    if abx_seeds != expected {
        issues.push(PerceptualParticipantScheduleIssueV1::WrongParticipantSeedPanel {
            participant_token: schedule.participant_token.clone(),
            task: "abx".into(),
        });
    }
    if directional_seeds != expected {
        issues.push(PerceptualParticipantScheduleIssueV1::WrongParticipantSeedPanel {
            participant_token: schedule.participant_token.clone(),
            task: "directional".into(),
        });
    }
    for seed in expected {
        if abx_position_by_seed.get(&seed) == directional_position_by_seed.get(&seed) {
            issues.push(
                PerceptualParticipantScheduleIssueV1::SameItemOrdinalAcrossTaskBlocks {
                    participant_token: schedule.participant_token.clone(),
                    seed,
                },
            );
        }
    }
}

fn validate_position_balance(
    task: &str,
    positions: &BTreeMap<u64, [usize; 8]>,
    issues: &mut Vec<PerceptualParticipantScheduleIssueV1>,
) {
    for seed in MEL003_FIXED_SEEDS {
        let counts = positions.get(&seed).copied().unwrap_or([0; 8]);
        let minimum = counts.into_iter().min().unwrap_or(0);
        let maximum = counts.into_iter().max().unwrap_or(0);
        if maximum.abs_diff(minimum) > 1 {
            issues.push(PerceptualParticipantScheduleIssueV1::ItemPositionImbalance {
                seed,
                task: task.into(),
                minimum,
                maximum,
            });
        }
    }
}

fn validate_private_audit(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    book: &PerceptualParticipantScheduleBookV1,
    audit: &PerceptualParticipantScheduleAuditV1,
    issues: &mut Vec<PerceptualParticipantScheduleIssueV1>,
) {
    if audit.schedule_version != book.schedule_version
        || audit.builder_version != book.builder_version
        || audit.protocol_sha256 != book.protocol_sha256
        || audit.stimulus_pack_sha256 != book.stimulus_pack_sha256
        || audit.cohort_id != book.cohort_id
    {
        issues.push(PerceptualParticipantScheduleIssueV1::AuditIdentityMismatch);
    }

    let schedule_by_token: BTreeMap<_, _> = book
        .schedules
        .iter()
        .map(|schedule| (schedule.participant_token.as_str(), schedule))
        .collect();
    let pack_by_seed: BTreeMap<_, _> = stimulus_pack
        .items
        .iter()
        .map(|pair| (pair.seed, pair))
        .collect();
    let mut audit_tokens = BTreeSet::new();
    let mut item_cell_counts: BTreeMap<u64, [usize; 8]> = BTreeMap::new();

    for participant in &audit.participants {
        audit_tokens.insert(participant.participant_token.clone());
        let Some(public) = schedule_by_token.get(participant.participant_token.as_str()).copied()
        else {
            issues.push(
                PerceptualParticipantScheduleIssueV1::UnexpectedPrivateAuditParticipant {
                    participant_token: participant.participant_token.clone(),
                },
            );
            continue;
        };
        if participant.task_order != public.task_order {
            issues.push(PerceptualParticipantScheduleIssueV1::TaskOrderMismatch {
                participant_token: participant.participant_token.clone(),
            });
        }
        let mut cells = BTreeSet::new();
        for trial in &participant.trials {
            let cell = trial.factorial_assignment.index();
            if !cells.insert(cell) {
                issues.push(
                    PerceptualParticipantScheduleIssueV1::FactorialCellNotUniqueWithinParticipant {
                        participant_token: participant.participant_token.clone(),
                        cell,
                    },
                );
            }
            item_cell_counts.entry(trial.seed).or_insert([0; 8])[cell] += 1;

            let Some(pair) = pack_by_seed.get(&trial.seed).copied() else {
                issues.push(PerceptualParticipantScheduleIssueV1::PublicPrivateTrialMismatch {
                    participant_token: participant.participant_token.clone(),
                    seed: trial.seed,
                    task: "missing-pack-item".into(),
                });
                continue;
            };
            if trial.baseline_output_sha256 != pair.baseline.output_sha256
                || trial.intervention_output_sha256 != pair.intervention.output_sha256
            {
                issues.push(PerceptualParticipantScheduleIssueV1::PublicPrivateTrialMismatch {
                    participant_token: participant.participant_token.clone(),
                    seed: trial.seed,
                    task: "asset-binding".into(),
                });
            }
            validate_public_private_trial(public, trial, issues);
        }
        for cell in 0..FACTORIAL_CELL_COUNT {
            if !cells.contains(&cell) {
                issues.push(
                    PerceptualParticipantScheduleIssueV1::MissingFactorialCellWithinParticipant {
                        participant_token: participant.participant_token.clone(),
                        cell,
                    },
                );
            }
        }
    }
    for schedule in &book.schedules {
        if !audit_tokens.contains(&schedule.participant_token) {
            issues.push(
                PerceptualParticipantScheduleIssueV1::MissingPrivateAuditParticipant {
                    participant_token: schedule.participant_token.clone(),
                },
            );
        }
    }
    for seed in MEL003_FIXED_SEEDS {
        let counts = item_cell_counts.get(&seed).copied().unwrap_or([0; 8]);
        let minimum = counts.into_iter().min().unwrap_or(0);
        let maximum = counts.into_iter().max().unwrap_or(0);
        if maximum.abs_diff(minimum) > 1 {
            issues.push(PerceptualParticipantScheduleIssueV1::FactorialItemImbalance {
                seed,
                minimum,
                maximum,
            });
        }
    }

    // Rank indices must form exactly 0..N-1. Their secret-derived ordering is
    // verified by reproducible rebuild after key reveal; this audit at least
    // prevents duplicate or missing rank slots before collection.
    let mut rank_indices: Vec<_> = audit
        .participants
        .iter()
        .map(|participant| participant.secret_rank_index)
        .collect();
    rank_indices.sort_unstable();
    if rank_indices != (0..audit.participants.len()).collect::<Vec<_>>() {
        for participant in &audit.participants {
            issues.push(PerceptualParticipantScheduleIssueV1::WrongSecretRankIndex {
                participant_token: participant.participant_token.clone(),
            });
        }
    }

    // The public commitment remains load-bearing even though the secret itself
    // is intentionally absent from both schedule and audit artifacts.
    if book.randomization_commitment_sha256
        != protocol.blinding.randomization_commitment_sha256
    {
        issues.push(PerceptualParticipantScheduleIssueV1::RandomizationCommitmentMismatch);
    }
}

fn validate_public_private_trial(
    public: &PublicParticipantScheduleV1,
    audit: &PrivateTrialMappingV1,
    issues: &mut Vec<PerceptualParticipantScheduleIssueV1>,
) {
    let abx = public.abx_trials.iter().find(|trial| trial.seed == audit.seed);
    let directional = public
        .directional_trials
        .iter()
        .find(|trial| trial.seed == audit.seed);
    match abx {
        Some(trial)
            if trial.trial_id == audit.abx_trial_id
                && trial.item_id == audit.item_id
                && trial.a_clip_id == audit.a_clip_id
                && trial.b_clip_id == audit.b_clip_id
                && trial.x_clip_id == audit.x_clip_id => {}
        _ => issues.push(PerceptualParticipantScheduleIssueV1::PublicPrivateTrialMismatch {
            participant_token: public.participant_token.clone(),
            seed: audit.seed,
            task: "abx".into(),
        }),
    }
    match directional {
        Some(trial)
            if trial.trial_id == audit.directional_trial_id
                && trial.item_id == audit.item_id
                && trial.left_clip_id == audit.left_clip_id
                && trial.right_clip_id == audit.right_clip_id => {}
        _ => issues.push(PerceptualParticipantScheduleIssueV1::PublicPrivateTrialMismatch {
            participant_token: public.participant_token.clone(),
            seed: audit.seed,
            task: "directional".into(),
        }),
    }
}

fn derive_rank(secret_key: &[u8; 32], participant_token: &str) -> [u8; 32] {
    domain_hash(secret_key, "participant-rank", &[participant_token.as_bytes()])
}

fn opaque_id(secret_key: &[u8; 32], domain: &str, participant_token: &str, seed: u64) -> String {
    let seed_bytes = seed.to_be_bytes();
    let digest = domain_hash(
        secret_key,
        domain,
        &[participant_token.as_bytes(), &seed_bytes],
    );
    format!("opaque-{}", hex32(&digest))
}

fn domain_hash(secret_key: &[u8; 32], domain: &str, parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"mel003-perceptual-schedule-domain-v1\0");
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain.as_bytes());
    hasher.update(secret_key);
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn hex32(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::perceptual_stimulus_pack::{
        C6fRenderSubjectItemV1, LoudnessMeterIdentityV1,
        StimulusAudioAssetV1, StimulusRendererIdentityV1, StimulusTransformV1,
        C6F_RENDER_SUBJECT_BINDING_VERSION, GAIN_TRANSFORM_PROFILE_V1,
        LOUDNESS_MEASUREMENT_PROFILE_V1, MAX_RESIDUAL_PAIR_DELTA_LU_V1,
        PERCEPTUAL_STIMULUS_PACK_VERSION, REQUIRED_CHANNEL_COUNT,
        REQUIRED_SAMPLE_RATE_HZ,
    };
    use crate::evidence_digest::perceptual_study_protocol::{
        AnalysisPolicyV1, BlindingAndRandomizationPolicyV1,
        ExternalPerceptualPreregistrationV1, ForbiddenPerceptualClaimV1,
        LoudnessMatchingV1, Mel003AcousticSubjectBindingV1,
        MissingResponsePolicyV1, ParticipantPolicyV1,
        PerceptualEndpointRoleV1, PerceptualEndpointV1, PerceptualStudyItemV1,
        PerceptualTaskV1, PrimaryAnalysisModelV1, SampleSizePlanV1,
        SecondaryMultiplicityPolicyV1, StimulusExtentV1, StimulusPolicyV1,
        MEL003_C6F_BUNDLE_VERSION, PERCEPTUAL_STUDY_PROTOCOL_VERSION,
    };
    use crate::evidence_digest::perceptual_stimulus_pack::PerceptualStimulusPairV1;

    const SECRET: [u8; 32] = [7u8; 32];
    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const COMMIT: &str = "cccccccccccccccccccccccccccccccccccccccc";

    fn protocol(maximum_enrolled: usize) -> FrozenPerceptualStudyProtocolV1 {
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
                    estimand: "ABX correctness probability".into(),
                },
                PerceptualEndpointV1 {
                    task: PerceptualTaskV1::DirectionalRearticulation2Afc,
                    role: PerceptualEndpointRoleV1::KeySecondary,
                    chance_probability: 0.5,
                    estimand: "directional re-articulation choice probability".into(),
                },
            ],
            stimulus: StimulusPolicyV1 {
                extent: StimulusExtentV1::WholeFourBarSubject,
                synchronized_playhead_required: true,
                loudness_matching: LoudnessMatchingV1::PairwiseIntegratedLufsAttenuationOnly,
                maximum_attenuation_db: 6.0,
                preserve_pair_alignment: true,
                neutral_presentation_labels_required: true,
                disjoint_practice_material_required: true,
            },
            blinding: BlindingAndRandomizationPolicyV1 {
                randomization_commitment_sha256: sha256_hex(&SECRET),
                schedule_builder_version: PERCEPTUAL_SCHEDULE_BUILDER_VERSION.into(),
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
                planned_completed_participants: maximum_enrolled.saturating_sub(2).max(1),
                maximum_enrolled_participants: maximum_enrolled,
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

    fn binding(protocol: &FrozenPerceptualStudyProtocolV1) -> FrozenC6fRenderSubjectBindingV1 {
        FrozenC6fRenderSubjectBindingV1 {
            binding_version: C6F_RENDER_SUBJECT_BINDING_VERSION.into(),
            protocol_sha256: protocol.protocol_sha256().unwrap(),
            c6f_source_commit: protocol.acoustic_subject.c6f_source_commit.clone(),
            c6f_bundle_sha256: protocol.acoustic_subject.c6f_bundle_sha256.clone(),
            c6f_bundle_version: protocol.acoustic_subject.c6f_bundle_version.clone(),
            renderer: StimulusRendererIdentityV1 {
                source_revision: COMMIT.into(),
                renderer_version: "native-sonata-renderer-v1".into(),
                render_config_sha256: DIGEST.into(),
                environment_sha256: DIGEST.into(),
            },
            items: protocol
                .items
                .iter()
                .enumerate()
                .map(|(index, item)| C6fRenderSubjectItemV1 {
                    item_id: item.item_id.clone(),
                    seed: item.seed,
                    baseline_render_sha256: format!("{:064x}", index + 1),
                    intervention_render_sha256: format!("{:064x}", index + 100),
                    sample_rate_hz: REQUIRED_SAMPLE_RATE_HZ,
                    channel_count: REQUIRED_CHANNEL_COUNT,
                    frame_count: 88_200,
                })
                .collect(),
        }
    }

    fn asset(source: &str, output: &str, initial: f64, final_lufs: f64, gain: f64) -> StimulusAudioAssetV1 {
        StimulusAudioAssetV1 {
            source_sha256: source.into(),
            output_sha256: output.into(),
            sample_rate_hz: REQUIRED_SAMPLE_RATE_HZ,
            channel_count: REQUIRED_CHANNEL_COUNT,
            frame_count: 88_200,
            initial_integrated_lufs: initial,
            final_integrated_lufs: final_lufs,
            applied_gain_db: gain,
        }
    }

    fn stimulus_pack(
        protocol: &FrozenPerceptualStudyProtocolV1,
        binding: &FrozenC6fRenderSubjectBindingV1,
    ) -> FrozenPerceptualStimulusPackV1 {
        FrozenPerceptualStimulusPackV1 {
            pack_version: PERCEPTUAL_STIMULUS_PACK_VERSION.into(),
            protocol_sha256: protocol.protocol_sha256().unwrap(),
            render_subject_binding_sha256: binding.binding_sha256().unwrap(),
            c6f_source_commit: protocol.acoustic_subject.c6f_source_commit.clone(),
            c6f_bundle_sha256: protocol.acoustic_subject.c6f_bundle_sha256.clone(),
            c6f_bundle_version: protocol.acoustic_subject.c6f_bundle_version.clone(),
            extent: StimulusExtentV1::WholeFourBarSubject,
            loudness_matching: LoudnessMatchingV1::PairwiseIntegratedLufsAttenuationOnly,
            transform: StimulusTransformV1::ConstantGainAttenuationOnly,
            gain_transform_profile: GAIN_TRANSFORM_PROFILE_V1.into(),
            max_attenuation_db: protocol.stimulus.maximum_attenuation_db,
            max_residual_pair_delta_lu: MAX_RESIDUAL_PAIR_DELTA_LU_V1,
            renderer: binding.renderer.clone(),
            loudness_meter: LoudnessMeterIdentityV1 {
                measurement_profile: LOUDNESS_MEASUREMENT_PROFILE_V1.into(),
                source_revision: COMMIT.into(),
                implementation_version: "symthaea-measure-lufs-v1".into(),
                environment_sha256: DIGEST.into(),
            },
            items: binding
                .items
                .iter()
                .enumerate()
                .map(|(index, item)| PerceptualStimulusPairV1 {
                    item_id: item.item_id.clone(),
                    seed: item.seed,
                    baseline: asset(
                        &item.baseline_render_sha256,
                        &format!("{:064x}", index + 300),
                        -17.0,
                        -18.0,
                        -1.0,
                    ),
                    intervention: asset(
                        &item.intervention_render_sha256,
                        &item.intervention_render_sha256,
                        -18.0,
                        -18.0,
                        0.0,
                    ),
                    attenuated_arm: Some(StimulusArmV1::Baseline),
                    post_match_pair_delta_lu: 0.0,
                })
                .collect(),
            participant_labels_bound: false,
            participant_schedule_bound: false,
            responses_present: false,
        }
    }

    fn cohort(count: usize) -> PerceptualCohortSlotsV1 {
        PerceptualCohortSlotsV1 {
            cohort_id: "mel003-confirmatory-v1".into(),
            participant_tokens: (0..count)
                .map(|index| format!("slot-{index:03}"))
                .collect(),
        }
    }

    #[test]
    fn generated_schedule_balances_all_item_factors_and_task_order() {
        let protocol = protocol(18);
        let binding = binding(&protocol);
        let pack = stimulus_pack(&protocol, &binding);
        let cohort = cohort(18);
        let (book, audit) = build_perceptual_participant_schedule(
            &protocol,
            &pack,
            &binding,
            &cohort,
            SECRET,
        )
        .unwrap();
        assert!(validate_perceptual_participant_schedule(
            &protocol,
            &pack,
            &binding,
            &cohort,
            &book,
            Some(&audit),
        )
        .is_empty());
        let abx_first = book
            .schedules
            .iter()
            .filter(|schedule| schedule.task_order == TaskBlockOrderV1::AbxThenDirectional)
            .count();
        assert_eq!(abx_first, 9);
        for participant in &audit.participants {
            let cells: BTreeSet<_> = participant
                .trials
                .iter()
                .map(|trial| trial.factorial_assignment.index())
                .collect();
            assert_eq!(cells.len(), FACTORIAL_CELL_COUNT);
        }
    }

    #[test]
    fn schedule_is_deterministic_for_same_secret_and_inputs() {
        let protocol = protocol(10);
        let binding = binding(&protocol);
        let pack = stimulus_pack(&protocol, &binding);
        let cohort = cohort(10);
        let first = build_perceptual_participant_schedule(
            &protocol, &pack, &binding, &cohort, SECRET,
        )
        .unwrap();
        let second = build_perceptual_participant_schedule(
            &protocol, &pack, &binding, &cohort, SECRET,
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn wrong_secret_is_rejected_before_schedule_generation() {
        let protocol = protocol(10);
        let binding = binding(&protocol);
        let pack = stimulus_pack(&protocol, &binding);
        let cohort = cohort(10);
        let issues = build_perceptual_participant_schedule(
            &protocol, &pack, &binding, &cohort, [8u8; 32],
        )
        .unwrap_err();
        assert!(issues.contains(
            &PerceptualParticipantScheduleIssueV1::RandomizationCommitmentMismatch
        ));
    }

    #[test]
    fn participant_pool_must_be_preallocated_to_maximum_enrollment() {
        let protocol = protocol(10);
        let binding = binding(&protocol);
        let pack = stimulus_pack(&protocol, &binding);
        let cohort = cohort(9);
        let issues = build_perceptual_participant_schedule(
            &protocol, &pack, &binding, &cohort, SECRET,
        )
        .unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            PerceptualParticipantScheduleIssueV1::ParticipantCountMismatch { .. }
        )));
    }

    #[test]
    fn private_audit_commitment_is_load_bearing() {
        let protocol = protocol(10);
        let binding = binding(&protocol);
        let pack = stimulus_pack(&protocol, &binding);
        let cohort = cohort(10);
        let (mut book, audit) = build_perceptual_participant_schedule(
            &protocol, &pack, &binding, &cohort, SECRET,
        )
        .unwrap();
        book.private_audit_sha256 = "b".repeat(64);
        assert!(validate_perceptual_participant_schedule(
            &protocol,
            &pack,
            &binding,
            &cohort,
            &book,
            Some(&audit),
        )
        .contains(&PerceptualParticipantScheduleIssueV1::PrivateAuditDigestMismatch));
    }

    #[test]
    fn no_item_occupies_same_ordinal_position_in_both_task_blocks() {
        let protocol = protocol(10);
        let binding = binding(&protocol);
        let pack = stimulus_pack(&protocol, &binding);
        let cohort = cohort(10);
        let (book, audit) = build_perceptual_participant_schedule(
            &protocol, &pack, &binding, &cohort, SECRET,
        )
        .unwrap();
        assert!(!validate_perceptual_participant_schedule(
            &protocol,
            &pack,
            &binding,
            &cohort,
            &book,
            Some(&audit),
        )
        .iter()
        .any(|issue| matches!(
            issue,
            PerceptualParticipantScheduleIssueV1::SameItemOrdinalAcrossTaskBlocks { .. }
        )));
    }
}
