#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Offline symbolic sequence contract for multi-step musical editing.
//!
//! This module validates exact state ancestry, support-local event mutation,
//! per-step hard gates, and sequence-level cost denominators. It does not
//! render audio, authorize a commit, or replace MEL-040 / MEL-041.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const SEQUENCE_REQUEST_SCHEMA_V1: &str = "mel-music-edit-sequence-request-v1";
pub const SEQUENCE_STEP_SCHEMA_V1: &str = "mel-music-edit-sequence-step-v1";
pub const SEQUENCE_STATE_DOMAIN_V1: &[u8] = b"luminous.mel.music-symbolic-state.v1";
pub const MAX_SEQUENCE_LENGTH_V1: u8 = 8;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalEventV1 {
    pub event_id: String,
    pub part_id: String,
    pub onset_ticks: u64,
    pub duration_ticks: u64,
    /// MIDI pitch in milli-semitones relative to MIDI note zero.
    pub pitch_millist: i32,
    pub velocity: u8,
    pub program: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditOperationKindV1 {
    Add,
    Remove,
    Replace,
    Volume,
    Tempo,
    Arrangement,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepGateVerdictV1 {
    Verified,
    Rejected,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceHistoryActionV1 {
    Apply,
    Undo,
    Redo,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepHardGatesV1 {
    pub target_edit: StepGateVerdictV1,
    pub protected_preservation: StepGateVerdictV1,
    pub rights_admission: StepGateVerdictV1,
    pub integrity: StepGateVerdictV1,
    pub source_lineage: StepGateVerdictV1,
    pub realization_evidence: StepGateVerdictV1,
}

impl StepHardGatesV1 {
    pub fn all_verified(&self) -> bool {
        self.target_edit == StepGateVerdictV1::Verified
            && self.protected_preservation == StepGateVerdictV1::Verified
            && self.rights_admission == StepGateVerdictV1::Verified
            && self.integrity == StepGateVerdictV1::Verified
            && self.source_lineage == StepGateVerdictV1::Verified
            && self.realization_evidence == StepGateVerdictV1::Verified
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceRequestV1 {
    pub schema_version: String,
    pub session_id: String,
    pub source_lineage_sha256: String,
    pub initial_artifact_sha256: String,
    pub initial_state_sha256: String,
    pub operation_plan_sha256: String,
    /// Confirmatory lengths are fixed at 1, 2, 4 and 8.
    pub expected_step_count: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceStepObservationV1 {
    pub schema_version: String,
    /// One-based and contiguous.
    pub sequence: u8,
    pub step_id: String,
    pub operation_id: String,
    pub operation_kind: MusicEditOperationKindV1,
    pub history_action: SequenceHistoryActionV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub undo_target_step_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redo_target_step_id: Option<String>,
    pub instruction_sha256: String,
    pub source_lineage_sha256: String,
    pub input_artifact_sha256: String,
    pub output_artifact_sha256: String,
    pub input_state_sha256: String,
    pub output_state_sha256: String,
    /// Candidate-selection receipt from EVAL-042; validated upstream.
    pub selection_receipt_sha256: String,
    /// Capability-qualified realization receipt from EVAL-041; validated upstream.
    pub realization_receipt_sha256: String,
    /// Evidence bundle for the six per-step hard gates.
    pub verification_report_sha256: String,
    /// Sorted, unique stable event IDs / edit support for this operation.
    pub allowed_event_ids: Vec<String>,
    /// Full canonical symbolic state after this operation.
    pub events_after: Vec<MusicalEventV1>,
    pub hard_gates: StepHardGatesV1,
    /// All attempts are counted, including failed and unresolved attempts.
    pub candidate_attempt_count: u16,
    pub failed_candidate_attempt_count: u16,
    pub render_call_count: u16,
    pub retry_count: u16,
    pub manual_intervention_count: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceDispositionV1 {
    Accepted,
    RejectedAtStep,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepEvaluationV1 {
    pub sequence: u8,
    pub step_id: String,
    pub changed_event_count: u32,
    pub changed_event_ids: Vec<String>,
    pub hard_gates_verified: bool,
    pub cumulative_attempt_count: u32,
    pub cumulative_failed_attempt_count: u32,
    pub cumulative_render_call_count: u32,
    pub cumulative_retry_count: u32,
    pub cumulative_manual_intervention_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceEvaluationV1 {
    pub session_id: String,
    pub disposition: SequenceDispositionV1,
    pub expected_step_count: u8,
    pub observed_step_count: u8,
    pub accepted_step_count: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejected_at_step: Option<u8>,
    pub cumulative_event_mutations: u32,
    pub cumulative_attempt_count: u32,
    pub cumulative_failed_candidate_attempt_count: u32,
    pub cumulative_render_call_count: u32,
    pub cumulative_retry_count: u32,
    pub cumulative_manual_intervention_count: u32,
    pub steps: Vec<StepEvaluationV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceErrorV1 {
    InvalidRequest,
    InvalidDigest,
    InvalidEvent,
    NonCanonicalEventOrder,
    DuplicateEventId,
    InvalidSupportOrder,
    DuplicateSupportId,
    StepCountExceeded,
    StepSequenceMismatch,
    StepIdentityMissing,
    SourceLineageMismatch,
    ArtifactAncestryMismatch,
    ArtifactIdentityNoChange,
    StateAncestryMismatch,
    OutputStateDigestMismatch,
    UnauthorizedEventMutation,
    NoTargetMutation,
    InvalidAttemptAccounting,
    MissingReceiptIdentity,
    InvalidHistoryTransition,
    UndoRestoreMismatch,
    RedoRestoreMismatch,
    ContinuationAfterHardGateFailure,
    OperationIdentityMissing,
    PatchSetNotCanonical,
    PatchPreconditionMismatch,
    ReadSetMissingEvent,
    EmptyOperation,
    HistoryReceiptMismatch,
    UndoStateMismatch,
    RedoStateMismatch,
    BranchIdentityInvalid,
    BranchBaseMismatch,
    MergeResolutionSetMismatch,
    MergeResolutionIdentityMismatch,
}

impl fmt::Display for SequenceErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidRequest => "sequence request schema, identity or length is invalid",
            Self::InvalidDigest => "a required SHA-256 digest is malformed",
            Self::InvalidEvent => "a symbolic event has invalid fields",
            Self::NonCanonicalEventOrder => "event snapshots must be sorted by event ID",
            Self::DuplicateEventId => "event IDs must be unique within a state",
            Self::InvalidSupportOrder => "allowed event IDs must be sorted",
            Self::DuplicateSupportId => "allowed event IDs must be unique",
            Self::StepCountExceeded => "observed steps exceed the frozen session length",
            Self::StepSequenceMismatch => "step sequence must be contiguous and one-based",
            Self::StepIdentityMissing => "step and operation identities must be nonempty",
            Self::SourceLineageMismatch => "step source lineage differs from the frozen session",
            Self::ArtifactAncestryMismatch => "step input artifact is not the previous verified output",
            Self::ArtifactIdentityNoChange => "a changed symbolic state cannot retain the same artifact digest",
            Self::StateAncestryMismatch => "step input event state is not the previous verified state",
            Self::OutputStateDigestMismatch => "recomputed output event-state digest differs",
            Self::UnauthorizedEventMutation => "an event outside the declared edit support changed",
            Self::NoTargetMutation => "target edit gate passed but no symbolic event changed",
            Self::InvalidAttemptAccounting => "candidate attempt counters are inconsistent",
            Self::MissingReceiptIdentity => "selection, realization or verification receipt is missing",
            Self::InvalidHistoryTransition => "undo or redo target is missing, stale, or not the active history edge",
            Self::UndoRestoreMismatch => "undo does not restore the exact prior artifact and symbolic state",
            Self::RedoRestoreMismatch => "redo does not restore the exact previously applied artifact and symbolic state",
            Self::ContinuationAfterHardGateFailure => "a later step cannot erase or follow an unresolved hard-gate failure",
            Self::OperationIdentityMissing => "operation identity must be nonempty",
            Self::PatchSetNotCanonical => "event patches or read sets are not sorted and unique",
            Self::PatchPreconditionMismatch => "event patch does not match the exact current event state",
            Self::ReadSetMissingEvent => "an operation read set references a missing event",
            Self::EmptyOperation => "an operation must declare at least one event patch",
            Self::HistoryReceiptMismatch => "edit history receipt commitment is invalid",
            Self::UndoStateMismatch => "undo requires the exact state produced by the operation",
            Self::RedoStateMismatch => "redo requires the exact state restored by undo",
            Self::BranchIdentityInvalid => "branch identities must be nonempty and distinct",
            Self::BranchBaseMismatch => "branch snapshot does not descend from the frozen base state",
            Self::MergeResolutionSetMismatch => "every conflict must have exactly one explicit resolution",
            Self::MergeResolutionIdentityMismatch => "a merge resolution names an unknown or duplicate conflict",
        };
        f.write_str(message)
    }
}
impl std::error::Error for SequenceErrorV1 {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update([0]);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub fn symbolic_state_sha256_v1(events: &[MusicalEventV1]) -> Result<String, SequenceErrorV1> {
    let mut previous: Option<&str> = None;
    for event in events {
        if event.event_id.is_empty()
            || event.part_id.is_empty()
            || event.duration_ticks == 0
            || event.velocity > 127
            || event.program > 127
        {
            return Err(SequenceErrorV1::InvalidEvent);
        }
        if previous.is_some_and(|prior| prior >= event.event_id.as_str()) {
            return Err(if previous == Some(event.event_id.as_str()) {
                SequenceErrorV1::DuplicateEventId
            } else {
                SequenceErrorV1::NonCanonicalEventOrder
            });
        }
        previous = Some(&event.event_id);
    }
    let bytes = serde_json::to_vec(events).map_err(|_| SequenceErrorV1::InvalidEvent)?;
    Ok(digest(SEQUENCE_STATE_DOMAIN_V1, &bytes))
}

fn event_map(events: &[MusicalEventV1]) -> BTreeMap<&str, &MusicalEventV1> {
    events.iter().map(|event| (event.event_id.as_str(), event)).collect()
}

fn changed_event_ids(
    before: &[MusicalEventV1],
    after: &[MusicalEventV1],
) -> Vec<String> {
    let old = event_map(before);
    let new = event_map(after);
    let ids: BTreeSet<&str> = old.keys().chain(new.keys()).copied().collect();
    ids.into_iter()
        .filter(|id| old.get(id) != new.get(id))
        .map(str::to_owned)
        .collect()
}

fn validate_request(request: &SequenceRequestV1) -> Result<(), SequenceErrorV1> {
    if request.schema_version != SEQUENCE_REQUEST_SCHEMA_V1
        || request.session_id.is_empty()
        || ![1, 2, 4, 8].contains(&request.expected_step_count)
    {
        return Err(SequenceErrorV1::InvalidRequest);
    }
    for digest_value in [
        &request.source_lineage_sha256,
        &request.initial_artifact_sha256,
        &request.initial_state_sha256,
        &request.operation_plan_sha256,
    ] {
        if !valid_sha256(digest_value) {
            return Err(SequenceErrorV1::InvalidDigest);
        }
    }
    Ok(())
}

fn validate_support(ids: &[String]) -> Result<(), SequenceErrorV1> {
    let mut previous: Option<&str> = None;
    for id in ids {
        if id.is_empty() {
            return Err(SequenceErrorV1::InvalidEvent);
        }
        if let Some(prior) = previous {
            if prior == id {
                return Err(SequenceErrorV1::DuplicateSupportId);
            }
            if prior > id {
                return Err(SequenceErrorV1::InvalidSupportOrder);
            }
        }
        previous = Some(id);
    }
    Ok(())
}

/// Replay the complete symbolic event state after each edit. A failed/unknown
/// hard gate remains in the returned result; no later step may continue from it.
pub fn evaluate_sequence_v1(
    request: &SequenceRequestV1,
    initial_events: &[MusicalEventV1],
    steps: &[SequenceStepObservationV1],
) -> Result<SequenceEvaluationV1, SequenceErrorV1> {
    validate_request(request)?;
    if steps.len() > usize::from(request.expected_step_count) {
        return Err(SequenceErrorV1::StepCountExceeded);
    }

    let initial_state_sha256 = symbolic_state_sha256_v1(initial_events)?;
    if initial_state_sha256 != request.initial_state_sha256 {
        return Err(SequenceErrorV1::StateAncestryMismatch);
    }
    let mut previous_state_sha256 = initial_state_sha256;
    let mut previous_artifact_sha256 = request.initial_artifact_sha256.clone();
    let mut previous_events = initial_events.to_vec();

    let mut total_attempts = 0_u32;
    let mut total_failed_attempts = 0_u32;
    let mut total_render_calls = 0_u32;
    let mut total_retries = 0_u32;
    let mut total_manual_interventions = 0_u32;
    let mut total_mutations = 0_u32;
    let mut accepted_steps = 0_u8;
    let mut rejected_at_step = None;
    let mut reports = Vec::with_capacity(steps.len());
    let mut step_ids = BTreeSet::new();
    let mut redo_target_index: Option<usize> = None;
    let mut operation_ids = BTreeSet::new();

    for (index, step) in steps.iter().enumerate() {
        let expected_sequence = index as u8 + 1;
        if step.schema_version != SEQUENCE_STEP_SCHEMA_V1 || step.sequence != expected_sequence {
            return Err(SequenceErrorV1::StepSequenceMismatch);
        }
        if step.step_id.is_empty()
            || step.operation_id.is_empty()
            || !step_ids.insert(step.step_id.as_str())
            || !operation_ids.insert(step.operation_id.as_str())
        {
            return Err(SequenceErrorV1::StepIdentityMissing);
        }
        if rejected_at_step.is_some() {
            return Err(SequenceErrorV1::ContinuationAfterHardGateFailure);
        }
        if step.source_lineage_sha256 != request.source_lineage_sha256 {
            return Err(SequenceErrorV1::SourceLineageMismatch);
        }
        for digest_value in [
            &step.instruction_sha256,
            &step.input_artifact_sha256,
            &step.output_artifact_sha256,
            &step.input_state_sha256,
            &step.output_state_sha256,
            &step.selection_receipt_sha256,
            &step.realization_receipt_sha256,
            &step.verification_report_sha256,
        ] {
            if !valid_sha256(digest_value) {
                return Err(SequenceErrorV1::InvalidDigest);
            }
        }
        let receipt_ids = [
            step.selection_receipt_sha256.as_str(),
            step.realization_receipt_sha256.as_str(),
            step.verification_report_sha256.as_str(),
        ];
        let unique_receipt_ids: BTreeSet<&str> = receipt_ids.into_iter().collect();
        if unique_receipt_ids.len() != receipt_ids.len() {
            return Err(SequenceErrorV1::MissingReceiptIdentity);
        }
        if step.input_artifact_sha256 != previous_artifact_sha256 {
            return Err(SequenceErrorV1::ArtifactAncestryMismatch);
        }
        if step.input_state_sha256 != previous_state_sha256 {
            return Err(SequenceErrorV1::StateAncestryMismatch);
        }
        if step.candidate_attempt_count == 0
            || step.failed_candidate_attempt_count > step.candidate_attempt_count
            || step.retry_count > step.candidate_attempt_count
        {
            return Err(SequenceErrorV1::InvalidAttemptAccounting);
        }
        validate_support(&step.allowed_event_ids)?;
        let output_state_sha256 = symbolic_state_sha256_v1(&step.events_after)?;
        if output_state_sha256 != step.output_state_sha256 {
            return Err(SequenceErrorV1::OutputStateDigestMismatch);
        }

        let changed_ids = changed_event_ids(&previous_events, &step.events_after);
        match step.history_action {
            SequenceHistoryActionV1::Apply => {
                if step.undo_target_step_id.is_some() || step.redo_target_step_id.is_some() {
                    return Err(SequenceErrorV1::InvalidHistoryTransition);
                }
                // A new edit after undo creates a new branch and invalidates redo.
                redo_target_index = None;
            }
            SequenceHistoryActionV1::Undo => {
                if step.redo_target_step_id.is_some() {
                    return Err(SequenceErrorV1::InvalidHistoryTransition);
                }
                let Some(target_id) = step.undo_target_step_id.as_deref() else {
                    return Err(SequenceErrorV1::InvalidHistoryTransition);
                };
                let Some(target_index) = index.checked_sub(1) else {
                    return Err(SequenceErrorV1::InvalidHistoryTransition);
                };
                let target = &steps[target_index];
                if target.step_id != target_id
                    || target.history_action != SequenceHistoryActionV1::Apply
                {
                    return Err(SequenceErrorV1::InvalidHistoryTransition);
                }
                if step.output_artifact_sha256 != target.input_artifact_sha256
                    || step.output_state_sha256 != target.input_state_sha256
                {
                    return Err(SequenceErrorV1::UndoRestoreMismatch);
                }
                redo_target_index = Some(target_index);
            }
            SequenceHistoryActionV1::Redo => {
                if step.undo_target_step_id.is_some() {
                    return Err(SequenceErrorV1::InvalidHistoryTransition);
                }
                let Some(target_id) = step.redo_target_step_id.as_deref() else {
                    return Err(SequenceErrorV1::InvalidHistoryTransition);
                };
                let Some(target_index) = redo_target_index else {
                    return Err(SequenceErrorV1::InvalidHistoryTransition);
                };
                let target = &steps[target_index];
                if target.step_id != target_id
                    || index == 0
                    || steps[index - 1].history_action != SequenceHistoryActionV1::Undo
                {
                    return Err(SequenceErrorV1::InvalidHistoryTransition);
                }
                if step.output_artifact_sha256 != target.output_artifact_sha256
                    || step.output_state_sha256 != target.output_state_sha256
                {
                    return Err(SequenceErrorV1::RedoRestoreMismatch);
                }
                redo_target_index = None;
            }
        }
        let allowed: BTreeSet<&str> = step.allowed_event_ids.iter().map(String::as_str).collect();
        if changed_ids.iter().any(|id| !allowed.contains(id.as_str())) {
            return Err(SequenceErrorV1::UnauthorizedEventMutation);
        }
        if step.hard_gates.target_edit == StepGateVerdictV1::Verified && changed_ids.is_empty() {
            return Err(SequenceErrorV1::NoTargetMutation);
        }
        if !changed_ids.is_empty()
            && step.input_artifact_sha256 == step.output_artifact_sha256
        {
            return Err(SequenceErrorV1::ArtifactIdentityNoChange);
        }

        total_attempts += u32::from(step.candidate_attempt_count);
        total_failed_attempts += u32::from(step.failed_candidate_attempt_count);
        total_render_calls += u32::from(step.render_call_count);
        total_retries += u32::from(step.retry_count);
        total_manual_interventions += u32::from(step.manual_intervention_count);
        total_mutations += changed_ids.len() as u32;

        let gates_verified = step.hard_gates.all_verified();
        if gates_verified {
            accepted_steps += 1;
        } else {
            rejected_at_step = Some(step.sequence);
        }

        reports.push(StepEvaluationV1 {
            sequence: step.sequence,
            step_id: step.step_id.clone(),
            changed_event_count: changed_ids.len() as u32,
            changed_event_ids: changed_ids,
            hard_gates_verified: gates_verified,
            cumulative_attempt_count: total_attempts,
            cumulative_failed_attempt_count: total_failed_attempts,
            cumulative_render_call_count: total_render_calls,
            cumulative_retry_count: total_retries,
            cumulative_manual_intervention_count: total_manual_interventions,
        });

        previous_state_sha256 = output_state_sha256;
        previous_artifact_sha256 = step.output_artifact_sha256.clone();
        previous_events = step.events_after.clone();
    }

    let disposition = if rejected_at_step.is_some() {
        SequenceDispositionV1::RejectedAtStep
    } else if steps.len() == usize::from(request.expected_step_count) {
        SequenceDispositionV1::Accepted
    } else {
        SequenceDispositionV1::Incomplete
    };

    Ok(SequenceEvaluationV1 {
        session_id: request.session_id.clone(),
        disposition,
        expected_step_count: request.expected_step_count,
        observed_step_count: steps.len() as u8,
        accepted_step_count: accepted_steps,
        rejected_at_step,
        cumulative_event_mutations: total_mutations,
        cumulative_attempt_count: total_attempts,
        cumulative_failed_candidate_attempt_count: total_failed_attempts,
        cumulative_render_call_count: total_render_calls,
        cumulative_retry_count: total_retries,
        cumulative_manual_intervention_count: total_manual_interventions,
        steps: reports,
    })
}

// ---- Operation algebra, undo/redo, and explicit branch merge (Stage 2) ----

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalEventPatchV1 {
    pub event_id: String,
    /// None means the event must not exist in the input state.
    pub before: Option<MusicalEventV1>,
    /// None means the event is deleted from the output state.
    pub after: Option<MusicalEventV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SymbolicEditOperationV1 {
    pub operation_id: String,
    /// Sorted event IDs whose current state is read by this operation.
    pub read_event_ids: Vec<String>,
    /// Sorted, unique writes with exact before-state preconditions.
    pub patches: Vec<MusicalEventPatchV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditHistoryReceiptV1 {
    pub operation: SymbolicEditOperationV1,
    pub before_state_sha256: String,
    pub after_state_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct EditHistoryReceiptPayloadV1 {
    operation: SymbolicEditOperationV1,
    before_state_sha256: String,
    after_state_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BranchSnapshotV1 {
    pub branch_id: String,
    pub base_state_sha256: String,
    pub events: Vec<MusicalEventV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BranchMergeConflictV1 {
    pub event_id: String,
    pub base: Option<MusicalEventV1>,
    pub left: Option<MusicalEventV1>,
    pub right: Option<MusicalEventV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BranchMergePlanV1 {
    pub left_branch_id: String,
    pub right_branch_id: String,
    pub base_state_sha256: String,
    pub base_events: Vec<MusicalEventV1>,
    pub left_patches: Vec<MusicalEventPatchV1>,
    pub right_patches: Vec<MusicalEventPatchV1>,
    pub conflicts: Vec<BranchMergeConflictV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeChoiceV1 {
    UseLeft,
    UseRight,
    UseCustom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MergeResolutionV1 {
    pub event_id: String,
    pub choice: MergeChoiceV1,
    /// Required for UseCustom; None explicitly means delete the event.
    #[serde(default)]
    pub custom_event: Option<MusicalEventV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BranchMergeResultV1 {
    pub merge_parent_ids: [String; 2],
    pub base_state_sha256: String,
    pub output_state_sha256: String,
    pub resolved_conflict_event_ids: Vec<String>,
    pub events: Vec<MusicalEventV1>,
    pub merge_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct BranchMergeResultPayloadV1 {
    merge_parent_ids: [String; 2],
    base_state_sha256: String,
    output_state_sha256: String,
    resolved_conflict_event_ids: Vec<String>,
    events: Vec<MusicalEventV1>,
}

fn validate_operation_v1(operation: &SymbolicEditOperationV1) -> Result<(), SequenceErrorV1> {
    if operation.operation_id.is_empty() {
        return Err(SequenceErrorV1::OperationIdentityMissing);
    }
    if operation.patches.is_empty() {
        return Err(SequenceErrorV1::EmptyOperation);
    }
    if !operation
        .read_event_ids
        .windows(2)
        .all(|pair| pair[0] < pair[1])
        || !operation
            .patches
            .windows(2)
            .all(|pair| pair[0].event_id < pair[1].event_id)
    {
        return Err(SequenceErrorV1::PatchSetNotCanonical);
    }
    for patch in &operation.patches {
        if patch.event_id.is_empty()
            || patch.before.as_ref().is_some_and(|event| event.event_id != patch.event_id)
            || patch.after.as_ref().is_some_and(|event| event.event_id != patch.event_id)
            || (patch.before.is_none() && patch.after.is_none())
        {
            return Err(SequenceErrorV1::InvalidEvent);
        }
    }
    Ok(())
}

pub fn apply_event_patches_v1(
    events: &[MusicalEventV1],
    patches: &[MusicalEventPatchV1],
) -> Result<Vec<MusicalEventV1>, SequenceErrorV1> {
    symbolic_state_sha256_v1(events)?;
    if !patches
        .windows(2)
        .all(|pair| pair[0].event_id < pair[1].event_id)
    {
        return Err(SequenceErrorV1::PatchSetNotCanonical);
    }
    let mut state: BTreeMap<String, MusicalEventV1> = events
        .iter()
        .cloned()
        .map(|event| (event.event_id.clone(), event))
        .collect();
    for patch in patches {
        if patch.event_id.is_empty()
            || patch.before.as_ref().is_some_and(|event| event.event_id != patch.event_id)
            || patch.after.as_ref().is_some_and(|event| event.event_id != patch.event_id)
            || (patch.before.is_none() && patch.after.is_none())
        {
            return Err(SequenceErrorV1::InvalidEvent);
        }
        let actual = state.get(&patch.event_id).cloned();
        if actual != patch.before {
            return Err(SequenceErrorV1::PatchPreconditionMismatch);
        }
        match &patch.after {
            Some(event) => {
                state.insert(patch.event_id.clone(), event.clone());
            }
            None => {
                state.remove(&patch.event_id);
            }
        }
    }
    let output: Vec<MusicalEventV1> = state.into_values().collect();
    symbolic_state_sha256_v1(&output)?;
    Ok(output)
}

pub fn apply_symbolic_edit_operation_v1(
    events: &[MusicalEventV1],
    operation: &SymbolicEditOperationV1,
) -> Result<Vec<MusicalEventV1>, SequenceErrorV1> {
    validate_operation_v1(operation)?;
    let available: BTreeSet<&str> = events.iter().map(|event| event.event_id.as_str()).collect();
    if operation
        .read_event_ids
        .iter()
        .any(|event_id| !available.contains(event_id.as_str()))
    {
        return Err(SequenceErrorV1::ReadSetMissingEvent);
    }
    apply_event_patches_v1(events, &operation.patches)
}

pub fn apply_symbolic_edit_with_history_v1(
    events: &[MusicalEventV1],
    operation: &SymbolicEditOperationV1,
) -> Result<(Vec<MusicalEventV1>, EditHistoryReceiptV1), SequenceErrorV1> {
    let before_state_sha256 = symbolic_state_sha256_v1(events)?;
    let output = apply_symbolic_edit_operation_v1(events, operation)?;
    let after_state_sha256 = symbolic_state_sha256_v1(&output)?;
    let payload = EditHistoryReceiptPayloadV1 {
        operation: operation.clone(),
        before_state_sha256: before_state_sha256.clone(),
        after_state_sha256: after_state_sha256.clone(),
    };
    let encoded = serde_json::to_vec(&payload)
        .map_err(|_| SequenceErrorV1::InvalidEvent)?;
    let receipt_sha256 = digest(b"luminous.mel.edit-history-receipt.v1", &encoded);
    Ok((
        output,
        EditHistoryReceiptV1 {
            operation: operation.clone(),
            before_state_sha256,
            after_state_sha256,
            receipt_sha256,
        },
    ))
}

fn verify_history_receipt_v1(receipt: &EditHistoryReceiptV1) -> Result<(), SequenceErrorV1> {
    let payload = EditHistoryReceiptPayloadV1 {
        operation: receipt.operation.clone(),
        before_state_sha256: receipt.before_state_sha256.clone(),
        after_state_sha256: receipt.after_state_sha256.clone(),
    };
    let encoded = serde_json::to_vec(&payload)
        .map_err(|_| SequenceErrorV1::HistoryReceiptMismatch)?;
    let expected = digest(b"luminous.mel.edit-history-receipt.v1", &encoded);
    if receipt.receipt_sha256 != expected {
        return Err(SequenceErrorV1::HistoryReceiptMismatch);
    }
    validate_operation_v1(&receipt.operation)
}

pub fn undo_symbolic_edit_v1(
    current_events: &[MusicalEventV1],
    receipt: &EditHistoryReceiptV1,
) -> Result<Vec<MusicalEventV1>, SequenceErrorV1> {
    verify_history_receipt_v1(receipt)?;
    if symbolic_state_sha256_v1(current_events)? != receipt.after_state_sha256 {
        return Err(SequenceErrorV1::UndoStateMismatch);
    }
    let inverse: Vec<MusicalEventPatchV1> = receipt
        .operation
        .patches
        .iter()
        .map(|patch| MusicalEventPatchV1 {
            event_id: patch.event_id.clone(),
            before: patch.after.clone(),
            after: patch.before.clone(),
        })
        .collect();
    let restored = apply_event_patches_v1(current_events, &inverse)?;
    if symbolic_state_sha256_v1(&restored)? != receipt.before_state_sha256 {
        return Err(SequenceErrorV1::UndoStateMismatch);
    }
    Ok(restored)
}

pub fn redo_symbolic_edit_v1(
    current_events: &[MusicalEventV1],
    receipt: &EditHistoryReceiptV1,
) -> Result<Vec<MusicalEventV1>, SequenceErrorV1> {
    verify_history_receipt_v1(receipt)?;
    if symbolic_state_sha256_v1(current_events)? != receipt.before_state_sha256 {
        return Err(SequenceErrorV1::RedoStateMismatch);
    }
    let restored = apply_event_patches_v1(current_events, &receipt.operation.patches)?;
    if symbolic_state_sha256_v1(&restored)? != receipt.after_state_sha256 {
        return Err(SequenceErrorV1::RedoStateMismatch);
    }
    Ok(restored)
}

fn operation_read_set(operation: &SymbolicEditOperationV1) -> BTreeSet<&str> {
    operation.read_event_ids.iter().map(String::as_str).collect()
}

fn operation_write_set(operation: &SymbolicEditOperationV1) -> BTreeSet<&str> {
    operation.patches.iter().map(|patch| patch.event_id.as_str()).collect()
}

/// Return true only when the read/write independence test holds and executing
/// both operations in either order yields the same canonical state.
pub fn symbolic_operations_commute_v1(
    base_events: &[MusicalEventV1],
    left: &SymbolicEditOperationV1,
    right: &SymbolicEditOperationV1,
) -> Result<bool, SequenceErrorV1> {
    validate_operation_v1(left)?;
    validate_operation_v1(right)?;
    let left_reads = operation_read_set(left);
    let right_reads = operation_read_set(right);
    let left_writes = operation_write_set(left);
    let right_writes = operation_write_set(right);
    if !left_writes.is_disjoint(&right_writes)
        || !left_writes.is_disjoint(&right_reads)
        || !right_writes.is_disjoint(&left_reads)
    {
        return Ok(false);
    }
    let left_then_right = apply_symbolic_edit_operation_v1(
        &apply_symbolic_edit_operation_v1(base_events, left)?,
        right,
    )?;
    let right_then_left = apply_symbolic_edit_operation_v1(
        &apply_symbolic_edit_operation_v1(base_events, right)?,
        left,
    )?;
    Ok(symbolic_state_sha256_v1(&left_then_right)?
        == symbolic_state_sha256_v1(&right_then_left)?)
}

fn diff_event_patches_v1(
    base_events: &[MusicalEventV1],
    branch_events: &[MusicalEventV1],
) -> Result<Vec<MusicalEventPatchV1>, SequenceErrorV1> {
    symbolic_state_sha256_v1(base_events)?;
    symbolic_state_sha256_v1(branch_events)?;
    let base = event_map(base_events);
    let branch = event_map(branch_events);
    let ids: BTreeSet<&str> = base.keys().chain(branch.keys()).copied().collect();
    Ok(ids
        .into_iter()
        .filter_map(|event_id| {
            let before = base.get(event_id).map(|event| (**event).clone());
            let after = branch.get(event_id).map(|event| (**event).clone());
            (before != after).then(|| MusicalEventPatchV1 {
                event_id: event_id.to_owned(),
                before,
                after,
            })
        })
        .collect())
}

pub fn plan_branch_merge_v1(
    base_events: &[MusicalEventV1],
    left: &BranchSnapshotV1,
    right: &BranchSnapshotV1,
) -> Result<BranchMergePlanV1, SequenceErrorV1> {
    if left.branch_id.is_empty()
        || right.branch_id.is_empty()
        || left.branch_id == right.branch_id
    {
        return Err(SequenceErrorV1::BranchIdentityInvalid);
    }
    let base_state_sha256 = symbolic_state_sha256_v1(base_events)?;
    if left.base_state_sha256 != base_state_sha256
        || right.base_state_sha256 != base_state_sha256
    {
        return Err(SequenceErrorV1::BranchBaseMismatch);
    }
    let left_patches = diff_event_patches_v1(base_events, &left.events)?;
    let right_patches = diff_event_patches_v1(base_events, &right.events)?;
    let left_by_id: BTreeMap<&str, &MusicalEventPatchV1> = left_patches
        .iter()
        .map(|patch| (patch.event_id.as_str(), patch))
        .collect();
    let right_by_id: BTreeMap<&str, &MusicalEventPatchV1> = right_patches
        .iter()
        .map(|patch| (patch.event_id.as_str(), patch))
        .collect();
    let mut conflicts = Vec::new();
    for (event_id, left_patch) in &left_by_id {
        if let Some(right_patch) = right_by_id.get(event_id)
            && left_patch.after != right_patch.after
        {
            conflicts.push(BranchMergeConflictV1 {
                event_id: (*event_id).to_owned(),
                base: left_patch.before.clone(),
                left: left_patch.after.clone(),
                right: right_patch.after.clone(),
            });
        }
    }
    Ok(BranchMergePlanV1 {
        left_branch_id: left.branch_id.clone(),
        right_branch_id: right.branch_id.clone(),
        base_state_sha256,
        base_events: base_events.to_vec(),
        left_patches,
        right_patches,
        conflicts,
    })
}

pub fn resolve_branch_merge_v1(
    plan: &BranchMergePlanV1,
    resolutions: &[MergeResolutionV1],
) -> Result<BranchMergeResultV1, SequenceErrorV1> {
    if plan.left_branch_id.is_empty()
        || plan.right_branch_id.is_empty()
        || plan.left_branch_id == plan.right_branch_id
        || symbolic_state_sha256_v1(&plan.base_events)? != plan.base_state_sha256
    {
        return Err(SequenceErrorV1::BranchBaseMismatch);
    }
    if !resolutions
        .windows(2)
        .all(|pair| pair[0].event_id < pair[1].event_id)
    {
        return Err(SequenceErrorV1::MergeResolutionIdentityMismatch);
    }
    let conflict_ids: BTreeSet<&str> = plan
        .conflicts
        .iter()
        .map(|conflict| conflict.event_id.as_str())
        .collect();
    let resolution_ids: BTreeSet<&str> = resolutions
        .iter()
        .map(|resolution| resolution.event_id.as_str())
        .collect();
    if conflict_ids.len() != plan.conflicts.len() || conflict_ids != resolution_ids {
        return Err(SequenceErrorV1::MergeResolutionSetMismatch);
    }
    let resolution_map: BTreeMap<&str, &MergeResolutionV1> = resolutions
        .iter()
        .map(|resolution| (resolution.event_id.as_str(), resolution))
        .collect();
    let left: BTreeMap<&str, &MusicalEventPatchV1> = plan
        .left_patches
        .iter()
        .map(|patch| (patch.event_id.as_str(), patch))
        .collect();
    let right: BTreeMap<&str, &MusicalEventPatchV1> = plan
        .right_patches
        .iter()
        .map(|patch| (patch.event_id.as_str(), patch))
        .collect();
    let ids: BTreeSet<&str> = left.keys().chain(right.keys()).copied().collect();
    let mut merged_patches = Vec::new();
    for event_id in ids {
        let left_patch = left.get(event_id).copied();
        let right_patch = right.get(event_id).copied();
        let before = left_patch
            .or(right_patch)
            .map(|patch| patch.before.clone())
            .unwrap_or(None);
        let after = match (left_patch, right_patch) {
            (Some(left_patch), Some(right_patch)) if left_patch.after == right_patch.after => {
                left_patch.after.clone()
            }
            (Some(_), Some(_)) => {
                let resolution = resolution_map
                    .get(event_id)
                    .ok_or(SequenceErrorV1::MergeResolutionSetMismatch)?;
                match resolution.choice {
                    MergeChoiceV1::UseLeft => left_patch.unwrap().after.clone(),
                    MergeChoiceV1::UseRight => right_patch.unwrap().after.clone(),
                    MergeChoiceV1::UseCustom => {
                        if resolution.custom_event.as_ref().is_some_and(|event| event.event_id != event_id) {
                            return Err(SequenceErrorV1::MergeResolutionIdentityMismatch);
                        }
                        resolution.custom_event.clone()
                    }
                }
            }
            (Some(left_patch), None) => left_patch.after.clone(),
            (None, Some(right_patch)) => right_patch.after.clone(),
            (None, None) => continue,
        };
        merged_patches.push(MusicalEventPatchV1 {
            event_id: event_id.to_owned(),
            before,
            after,
        });
    }
    let merged_events = apply_event_patches_v1(&plan.base_events, &merged_patches)?;
    let output_state_sha256 = symbolic_state_sha256_v1(&merged_events)?;
    let resolved_conflict_event_ids: Vec<String> = plan
        .conflicts
        .iter()
        .map(|conflict| conflict.event_id.clone())
        .collect();
    let payload = BranchMergeResultPayloadV1 {
        merge_parent_ids: [
            plan.left_branch_id.clone(),
            plan.right_branch_id.clone(),
        ],
        base_state_sha256: plan.base_state_sha256.clone(),
        output_state_sha256: output_state_sha256.clone(),
        resolved_conflict_event_ids: resolved_conflict_event_ids.clone(),
        events: merged_events.clone(),
    };
    let encoded = serde_json::to_vec(&payload).map_err(|_| SequenceErrorV1::InvalidEvent)?;
    let merge_receipt_sha256 = digest(b"luminous.mel.branch-merge-receipt.v1", &encoded);
    Ok(BranchMergeResultV1 {
        merge_parent_ids: payload.merge_parent_ids,
        base_state_sha256: payload.base_state_sha256,
        output_state_sha256,
        resolved_conflict_event_ids,
        events: merged_events,
        merge_receipt_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(value: u64) -> String { format!("{value:064x}") }

    fn gates(verdict: StepGateVerdictV1) -> StepHardGatesV1 {
        StepHardGatesV1 {
            target_edit: verdict,
            protected_preservation: verdict,
            rights_admission: verdict,
            integrity: verdict,
            source_lineage: verdict,
            realization_evidence: verdict,
        }
    }

    fn event(id: &str, pitch: i32) -> MusicalEventV1 {
        MusicalEventV1 {
            event_id: id.to_owned(),
            part_id: "piano".to_owned(),
            onset_ticks: 0,
            duration_ticks: 480,
            pitch_millist: pitch,
            velocity: 88,
            program: 0,
        }
    }

    fn initial_fixture_events() -> Vec<MusicalEventV1> {
        vec![event("event-a", 60_000), event("event-b", 64_000)]
    }

    fn request(count: u8, initial_events: &[MusicalEventV1]) -> SequenceRequestV1 {
        SequenceRequestV1 {
            schema_version: SEQUENCE_REQUEST_SCHEMA_V1.to_owned(),
            session_id: format!("session-{count}"),
            source_lineage_sha256: h(10),
            initial_artifact_sha256: h(11),
            initial_state_sha256: symbolic_state_sha256_v1(initial_events).unwrap(),
            operation_plan_sha256: h(12),
            expected_step_count: count,
        }
    }

    fn step(
        index: u8,
        input_artifact: String,
        output_artifact: String,
        input_events: &[MusicalEventV1],
        output_events: Vec<MusicalEventV1>,
        changed_id: &str,
        verdict: StepGateVerdictV1,
    ) -> SequenceStepObservationV1 {
        SequenceStepObservationV1 {
            schema_version: SEQUENCE_STEP_SCHEMA_V1.to_owned(),
            sequence: index,
            step_id: format!("step-{index}"),
            operation_id: format!("op-{index}"),
            operation_kind: MusicEditOperationKindV1::Replace,
            history_action: SequenceHistoryActionV1::Apply,
            undo_target_step_id: None,
            redo_target_step_id: None,
            instruction_sha256: h(100 + u64::from(index)),
            source_lineage_sha256: h(10),
            input_artifact_sha256: input_artifact,
            output_artifact_sha256: output_artifact,
            input_state_sha256: symbolic_state_sha256_v1(input_events).unwrap(),
            output_state_sha256: symbolic_state_sha256_v1(&output_events).unwrap(),
            selection_receipt_sha256: h(200 + u64::from(index)),
            realization_receipt_sha256: h(300 + u64::from(index)),
            verification_report_sha256: h(400 + u64::from(index)),
            allowed_event_ids: vec![changed_id.to_owned()],
            events_after: output_events,
            hard_gates: gates(verdict),
            candidate_attempt_count: 2,
            failed_candidate_attempt_count: 1,
            render_call_count: 3,
            retry_count: 1,
            manual_intervention_count: 0,
        }
    }

    #[test]
    fn initial_symbolic_state_must_be_bound_to_frozen_source() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let mut req = request(1, &initial);
        req.initial_state_sha256 = h(999);
        assert_eq!(
            evaluate_sequence_v1(&req, &initial, &[]),
            Err(SequenceErrorV1::StateAncestryMismatch)
        );
    }

    #[test]
    fn changed_artifact_must_have_a_new_artifact_digest() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let output = vec![event("event-a", 61_000), event("event-b", 64_000)];
        let first = step(1, h(11), h(11), &initial, output, "event-a", StepGateVerdictV1::Verified);
        assert_eq!(
            evaluate_sequence_v1(&request(1, &initial), &initial, &[first]),
            Err(SequenceErrorV1::ArtifactIdentityNoChange)
        );
    }

    #[test]
    fn later_hard_gate_is_checked_even_when_earlier_steps_passed() {
        let initial = initial_fixture_events();
        let state_1 = vec![event("event-a", 61_000), event("event-b", 64_000)];
        let state_2 = vec![event("event-a", 61_000), event("event-b", 65_000)];
        let first = step(1, h(11), h(21), &initial, state_1.clone(), "event-a", StepGateVerdictV1::Verified);
        let mut second = step(2, h(21), h(22), &state_1, state_2, "event-b", StepGateVerdictV1::Verified);
        second.hard_gates.integrity = StepGateVerdictV1::Unknown;
        let result = evaluate_sequence_v1(&request(2, &initial), &initial, &[first, second]).unwrap();
        assert_eq!(result.disposition, SequenceDispositionV1::RejectedAtStep);
        assert_eq!(result.accepted_step_count, 1);
        assert_eq!(result.rejected_at_step, Some(2));
    }

    fn patch(event_id: &str, before: i32, after: i32) -> MusicalEventPatchV1 {
        MusicalEventPatchV1 {
            event_id: event_id.to_owned(),
            before: Some(event(event_id, before)),
            after: Some(event(event_id, after)),
        }
    }

    fn operation(id: &str, event_id: &str, before: i32, after: i32) -> SymbolicEditOperationV1 {
        SymbolicEditOperationV1 {
            operation_id: id.to_owned(),
            read_event_ids: vec![event_id.to_owned()],
            patches: vec![patch(event_id, before, after)],
        }
    }

    #[test]
    fn disjoint_event_operations_commute_and_converge() {
        let base = vec![
            event("event-a", 60_000),
            event("event-b", 64_000),
            event("event-c", 67_000),
        ];
        let left = operation("raise-a", "event-a", 60_000, 61_000);
        let right = operation("raise-c", "event-c", 67_000, 68_000);
        assert!(symbolic_operations_commute_v1(&base, &left, &right).unwrap());
        let left_then_right = apply_symbolic_edit_operation_v1(
            &apply_symbolic_edit_operation_v1(&base, &left).unwrap(),
            &right,
        )
        .unwrap();
        let right_then_left = apply_symbolic_edit_operation_v1(
            &apply_symbolic_edit_operation_v1(&base, &right).unwrap(),
            &left,
        )
        .unwrap();
        assert_eq!(
            symbolic_state_sha256_v1(&left_then_right).unwrap(),
            symbolic_state_sha256_v1(&right_then_left).unwrap()
        );
    }

    #[test]
    fn same_event_operations_are_noncommuting_and_need_explicit_conflict_handling() {
        let base = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let left = operation("left-tune", "event-a", 60_000, 61_000);
        let right = operation("right-tune", "event-a", 60_000, 62_000);
        assert!(!symbolic_operations_commute_v1(&base, &left, &right).unwrap());
        assert_eq!(
            apply_symbolic_edit_operation_v1(
                &apply_symbolic_edit_operation_v1(&base, &left).unwrap(),
                &right,
            ),
            Err(SequenceErrorV1::PatchPreconditionMismatch)
        );
    }

    #[test]
    fn undo_and_redo_restore_exact_symbolic_states() {
        let base = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let edit = operation("edit-a", "event-a", 60_000, 63_000);
        let (edited, receipt) = apply_symbolic_edit_with_history_v1(&base, &edit).unwrap();
        let undone = undo_symbolic_edit_v1(&edited, &receipt).unwrap();
        assert_eq!(
            symbolic_state_sha256_v1(&undone).unwrap(),
            symbolic_state_sha256_v1(&base).unwrap()
        );
        let redone = redo_symbolic_edit_v1(&undone, &receipt).unwrap();
        assert_eq!(
            symbolic_state_sha256_v1(&redone).unwrap(),
            symbolic_state_sha256_v1(&edited).unwrap()
        );
    }

    #[test]
    fn undo_rejects_stale_state_and_tampered_history() {
        let base = vec![event("event-a", 60_000)];
        let edit = operation("edit-a", "event-a", 60_000, 62_000);
        let (edited, receipt) = apply_symbolic_edit_with_history_v1(&base, &edit).unwrap();
        assert_eq!(
            undo_symbolic_edit_v1(&base, &receipt),
            Err(SequenceErrorV1::UndoStateMismatch)
        );
        let mut tampered = receipt.clone();
        tampered.operation.patches[0].after.as_mut().unwrap().pitch_millist = 65_000;
        assert_eq!(
            undo_symbolic_edit_v1(&edited, &tampered),
            Err(SequenceErrorV1::HistoryReceiptMismatch)
        );
    }

    #[test]
    fn disjoint_branches_merge_without_losing_parentage() {
        let base = vec![
            event("event-a", 60_000),
            event("event-b", 64_000),
            event("event-c", 67_000),
        ];
        let base_hash = symbolic_state_sha256_v1(&base).unwrap();
        let left_events = vec![
            event("event-a", 61_000),
            event("event-b", 64_000),
            event("event-c", 67_000),
        ];
        let right_events = vec![
            event("event-a", 60_000),
            event("event-b", 65_000),
            event("event-c", 67_000),
        ];
        let left = BranchSnapshotV1 {
            branch_id: "left".to_owned(),
            base_state_sha256: base_hash.clone(),
            events: left_events,
        };
        let right = BranchSnapshotV1 {
            branch_id: "right".to_owned(),
            base_state_sha256: base_hash,
            events: right_events,
        };
        let plan = plan_branch_merge_v1(&base, &left, &right).unwrap();
        assert!(plan.conflicts.is_empty());
        let result = resolve_branch_merge_v1(&plan, &[]).unwrap();
        assert_eq!(result.merge_parent_ids, ["left".to_owned(), "right".to_owned()]);
        assert_eq!(result.events, vec![
            event("event-a", 61_000),
            event("event-b", 65_000),
            event("event-c", 67_000),
        ]);
    }

    #[test]
    fn conflicting_branches_require_one_explicit_resolution_per_event() {
        let base = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let base_hash = symbolic_state_sha256_v1(&base).unwrap();
        let left = BranchSnapshotV1 {
            branch_id: "left".to_owned(),
            base_state_sha256: base_hash.clone(),
            events: vec![event("event-a", 61_000), event("event-b", 64_000)],
        };
        let right = BranchSnapshotV1 {
            branch_id: "right".to_owned(),
            base_state_sha256: base_hash,
            events: vec![event("event-a", 62_000), event("event-b", 64_000)],
        };
        let plan = plan_branch_merge_v1(&base, &left, &right).unwrap();
        assert_eq!(plan.conflicts.len(), 1);
        assert_eq!(
            resolve_branch_merge_v1(&plan, &[]),
            Err(SequenceErrorV1::MergeResolutionSetMismatch)
        );
        let result = resolve_branch_merge_v1(
            &plan,
            &[MergeResolutionV1 {
                event_id: "event-a".to_owned(),
                choice: MergeChoiceV1::UseRight,
                custom_event: None,
            }],
        )
        .unwrap();
        assert_eq!(result.events, vec![event("event-a", 62_000), event("event-b", 64_000)]);
        assert_eq!(result.resolved_conflict_event_ids, vec!["event-a"]);
        assert_eq!(result.merge_parent_ids, ["left".to_owned(), "right".to_owned()]);
    }

    #[test]
    fn undo_restores_exact_prior_artifact_and_symbolic_state() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let edited = vec![event("event-a", 61_000), event("event-b", 64_000)];
        let apply = step(1, h(11), h(21), &initial, edited.clone(), "event-a", StepGateVerdictV1::Verified);
        let mut undo = step(2, h(21), h(11), &edited, initial.clone(), "event-a", StepGateVerdictV1::Verified);
        undo.history_action = SequenceHistoryActionV1::Undo;
        undo.undo_target_step_id = Some("step-1".to_owned());
        let result = evaluate_sequence_v1(&request(2, &initial), &initial, &[apply, undo]).unwrap();
        assert_eq!(result.disposition, SequenceDispositionV1::Accepted);
        assert_eq!(result.observed_step_count, 2);
        assert_eq!(result.cumulative_event_mutations, 2);
    }

    #[test]
    fn undo_rejects_a_nearby_but_not_exact_restore() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let edited = vec![event("event-a", 61_000), event("event-b", 64_000)];
        let almost_initial = vec![event("event-a", 60_001), event("event-b", 64_000)];
        let apply = step(1, h(11), h(21), &initial, edited.clone(), "event-a", StepGateVerdictV1::Verified);
        let mut undo = step(2, h(21), h(11), &edited, almost_initial, "event-a", StepGateVerdictV1::Verified);
        undo.history_action = SequenceHistoryActionV1::Undo;
        undo.undo_target_step_id = Some("step-1".to_owned());
        assert_eq!(
            evaluate_sequence_v1(&request(2, &initial), &initial, &[apply, undo]),
            Err(SequenceErrorV1::UndoRestoreMismatch)
        );
    }

    #[test]
    fn redo_restores_exactly_the_previous_edit_after_undo() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let edited = vec![event("event-a", 61_000), event("event-b", 64_000)];
        let apply = step(1, h(11), h(21), &initial, edited.clone(), "event-a", StepGateVerdictV1::Verified);
        let mut undo = step(2, h(21), h(11), &edited, initial.clone(), "event-a", StepGateVerdictV1::Verified);
        undo.history_action = SequenceHistoryActionV1::Undo;
        undo.undo_target_step_id = Some("step-1".to_owned());
        let mut redo = step(3, h(11), h(21), &initial, edited.clone(), "event-a", StepGateVerdictV1::Verified);
        redo.history_action = SequenceHistoryActionV1::Redo;
        redo.redo_target_step_id = Some("step-1".to_owned());
        let result = evaluate_sequence_v1(&request(4, &initial), &initial, &[apply, undo, redo]).unwrap();
        assert_eq!(result.disposition, SequenceDispositionV1::Incomplete);
        assert_eq!(result.observed_step_count, 3);
        assert_eq!(result.steps[2].changed_event_ids, vec!["event-a".to_owned()]);
    }

    #[test]
    fn redo_without_matching_undo_is_rejected() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let edited = vec![event("event-a", 61_000), event("event-b", 64_000)];
        let mut redo = step(1, h(11), h(21), &initial, edited, "event-a", StepGateVerdictV1::Verified);
        redo.history_action = SequenceHistoryActionV1::Redo;
        redo.redo_target_step_id = Some("missing-step".to_owned());
        assert_eq!(
            evaluate_sequence_v1(&request(1, &initial), &initial, &[redo]),
            Err(SequenceErrorV1::InvalidHistoryTransition)
        );
    }

    #[test]
    fn new_edit_after_undo_invalidates_redo_target() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let edited = vec![event("event-a", 61_000), event("event-b", 64_000)];
        let alternate = vec![event("event-a", 60_000), event("event-b", 65_000)];
        let apply = step(1, h(11), h(21), &initial, edited.clone(), "event-a", StepGateVerdictV1::Verified);
        let mut undo = step(2, h(21), h(11), &edited, initial.clone(), "event-a", StepGateVerdictV1::Verified);
        undo.history_action = SequenceHistoryActionV1::Undo;
        undo.undo_target_step_id = Some("step-1".to_owned());
        let new_edit = step(3, h(11), h(31), &initial, alternate, "event-b", StepGateVerdictV1::Verified);
        let result = evaluate_sequence_v1(&request(4, &initial), &initial, &[apply, undo, new_edit]).unwrap();
        assert_eq!(result.disposition, SequenceDispositionV1::Incomplete);
    }

    #[test]
    fn exact_two_step_chain_is_accepted_and_costs_accumulate() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000), event("event-c", 67_000)];
        let state_1 = vec![event("event-a", 60_000), event("event-b", 65_000), event("event-c", 67_000)];
        let state_2 = vec![event("event-a", 62_000), event("event-b", 65_000), event("event-c", 67_000)];
        let first = step(1, h(11), h(21), &initial, state_1.clone(), "event-b", StepGateVerdictV1::Verified);
        let second = step(2, h(21), h(22), &state_1, state_2, "event-a", StepGateVerdictV1::Verified);
        let result = evaluate_sequence_v1(&request(2, &initial), &initial, &[first, second]).unwrap();
        assert_eq!(result.disposition, SequenceDispositionV1::Accepted);
        assert_eq!(result.accepted_step_count, 2);
        assert_eq!(result.cumulative_event_mutations, 2);
        assert_eq!(result.cumulative_attempt_count, 4);
        assert_eq!(result.cumulative_failed_candidate_attempt_count, 2);
        assert_eq!(result.cumulative_render_call_count, 6);
    }

    #[test]
    fn unauthorized_collateral_event_change_is_rejected() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000), event("event-c", 67_000)];
        let output = vec![event("event-a", 60_000), event("event-b", 65_000), event("event-c", 68_000)];
        let first = step(1, h(11), h(21), &initial, output, "event-b", StepGateVerdictV1::Verified);
        assert_eq!(
            evaluate_sequence_v1(&request(1, &initial), &initial, &[first]),
            Err(SequenceErrorV1::UnauthorizedEventMutation)
        );
    }

    #[test]
    fn stale_artifact_parent_is_rejected() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let first = step(1, h(999), h(21), &initial, vec![event("event-a", 61_000), event("event-b", 64_000)], "event-a", StepGateVerdictV1::Verified);
        assert_eq!(
            evaluate_sequence_v1(&request(1, &initial), &initial, &[first]),
            Err(SequenceErrorV1::ArtifactAncestryMismatch)
        );
    }

    #[test]
    fn stale_symbolic_state_parent_is_rejected() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let mut first = step(1, h(11), h(21), &initial, vec![event("event-a", 61_000), event("event-b", 64_000)], "event-a", StepGateVerdictV1::Verified);
        first.input_state_sha256 = h(777);
        assert_eq!(
            evaluate_sequence_v1(&request(1, &initial), &initial, &[first]),
            Err(SequenceErrorV1::StateAncestryMismatch)
        );
    }

    #[test]
    fn later_step_cannot_launder_an_earlier_hard_gate_failure() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let state_1 = vec![event("event-a", 61_000), event("event-b", 64_000)];
        let state_2 = vec![event("event-a", 62_000), event("event-b", 64_000)];
        let first = step(1, h(11), h(21), &initial, state_1.clone(), "event-a", StepGateVerdictV1::Rejected);
        let second = step(2, h(21), h(22), &state_1, state_2, "event-a", StepGateVerdictV1::Verified);
        assert_eq!(
            evaluate_sequence_v1(&request(2, &initial), &initial, &[first, second]),
            Err(SequenceErrorV1::ContinuationAfterHardGateFailure)
        );
    }

    #[test]
    fn a_hard_gate_failure_is_retained_as_rejected_session_state() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let output = vec![event("event-a", 61_000), event("event-b", 64_000)];
        let first = step(1, h(11), h(21), &initial, output, "event-a", StepGateVerdictV1::Unknown);
        let result = evaluate_sequence_v1(&request(2, &initial), &initial, &[first]).unwrap();
        assert_eq!(result.disposition, SequenceDispositionV1::RejectedAtStep);
        assert_eq!(result.rejected_at_step, Some(1));
        assert_eq!(result.accepted_step_count, 0);
        assert_eq!(result.cumulative_attempt_count, 2);
    }

    #[test]
    fn incomplete_sequence_is_not_accepted() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let output = vec![event("event-a", 61_000), event("event-b", 64_000)];
        let first = step(1, h(11), h(21), &initial, output, "event-a", StepGateVerdictV1::Verified);
        let result = evaluate_sequence_v1(&request(2, &initial), &initial, &[first]).unwrap();
        assert_eq!(result.disposition, SequenceDispositionV1::Incomplete);
    }

    #[test]
    fn tampered_output_state_digest_is_rejected() {
        let initial = vec![event("event-a", 60_000), event("event-b", 64_000)];
        let output = vec![event("event-a", 61_000), event("event-b", 64_000)];
        let mut first = step(1, h(11), h(21), &initial, output, "event-a", StepGateVerdictV1::Verified);
        first.output_state_sha256 = h(999);
        assert_eq!(
            evaluate_sequence_v1(&request(1, &initial), &initial, &[first]),
            Err(SequenceErrorV1::OutputStateDigestMismatch)
        );
    }

    #[test]
    fn noncanonical_and_duplicate_event_ids_are_rejected() {
        let unsorted = vec![event("event-b", 64_000), event("event-a", 60_000)];
        assert_eq!(
            symbolic_state_sha256_v1(&unsorted),
            Err(SequenceErrorV1::NonCanonicalEventOrder)
        );
        let duplicate = vec![event("event-a", 60_000), event("event-a", 61_000)];
        assert_eq!(
            symbolic_state_sha256_v1(&duplicate),
            Err(SequenceErrorV1::DuplicateEventId)
        );
    }

    #[test]
    fn selection_realization_and_verification_receipts_must_be_distinct() {
        let initial = initial_fixture_events();
        let output = vec![event("event-a", 61_000), event("event-b", 64_000)];
        let mut first = step(1, h(11), h(21), &initial, output, "event-a", StepGateVerdictV1::Verified);
        first.verification_report_sha256 = first.selection_receipt_sha256.clone();
        assert_eq!(
            evaluate_sequence_v1(&request(1, &initial), &initial, &[first]),
            Err(SequenceErrorV1::MissingReceiptIdentity)
        );
    }

    #[test]
    fn candidate_attempt_accounting_cannot_drop_failures_or_exceed_attempts() {
        let initial = vec![event("event-a", 60_000)];
        let output = vec![event("event-a", 61_000)];
        let mut first = step(1, h(11), h(21), &initial, output, "event-a", StepGateVerdictV1::Verified);
        first.failed_candidate_attempt_count = 3;
        assert_eq!(
            evaluate_sequence_v1(&request(1, &initial), &initial, &[first]),
            Err(SequenceErrorV1::InvalidAttemptAccounting)
        );
    }

    #[test]
    fn session_length_is_limited_to_preregistered_ladder() {
        let initial = vec![event("event-a", 60_000)];
        assert_eq!(
            evaluate_sequence_v1(&request(3, &initial), &initial, &[]),
            Err(SequenceErrorV1::InvalidRequest)
        );
    }
}
