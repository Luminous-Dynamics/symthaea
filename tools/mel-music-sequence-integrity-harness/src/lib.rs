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
    ContinuationAfterHardGateFailure,
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
            Self::ContinuationAfterHardGateFailure => "a later step cannot erase or follow an unresolved hard-gate failure",
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
