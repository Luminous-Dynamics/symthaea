// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Replay-verified EVAL-017 → EVAL-019 support adapter.
//!
//! This adapter turns one independently replay-verified counterfactual
//! influence report per source event into planner-ready EVAL-019 support.
//! Support intervals are always in rendered PCM output-frame coordinates.
//! Empty intervals remain evidence-bearing: the per-event digest commits to a
//! complete replayed scan that found no window above the calibrated floor.
//!
//! FluidSynth effects-group assignments are read from the verified renderer
//! token, which extracted the exact synth.effects-groups=N setting from the
//! receipt-bound argv. The assignment is carried into routing-domain metadata.
//! Routing is deliberately marked incomplete: effects-group assignment does
//! not prove the full signal graph, master path, plugin graph or output bus.
//! EVAL-019 must therefore expand incomplete routing to global pair scope.

use crate::counterfactual_influence::{
    CounterfactualInfluenceError, CounterfactualInfluenceInput,
    CounterfactualInfluenceReport, EventIntervention, InfluenceDisposition,
    VerifiedRendererAudioBinding, VerifiedRepeatabilityCalibration,
    verify_counterfactual_influence,
};
use crate::fluid_render::{
    FluidSynthEffectsGroupAssignment, build_effects_group_assignment,
    effects_group_for_midi_channel, verify_effects_group_assignment,
};
use crate::interaction_probe_plan::{AcousticSupportInterval, EventInteractionSupport};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const COUNTERFACTUAL_SUPPORT_ADAPTER_VERSION: &str =
    "mel-mr-eval-025-counterfactual-support-adapter-v1";
const EVENT_EVIDENCE_DOMAIN: &[u8] =
    b"luminous-dynamics-mel-mr-eval-025-event-support-evidence-v1\0";
const ADAPTER_REPORT_DOMAIN: &[u8] =
    b"luminous-dynamics-mel-mr-eval-025-support-adapter-report-v1\0";
const MAX_ADAPTER_EVENTS: usize = 100_000;
const MAX_SUPPORT_INTERVALS_PER_EVENT: usize = 100_000;
const ROUTING_INCOMPLETE_REASON: &str =
    "effects-group assignment is known, but complete FluidSynth downstream/master/plugin routing is not independently evidenced";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterfactualEventSupportEvidence {
    pub version: String,
    pub source_event_index: usize,
    pub source_event_id: String,
    pub expected_score_event_count: usize,
    pub intervention: EventIntervention,
    pub influence_report_sha256: String,
    pub source_score_sha256: String,
    pub baseline_event_map_artifact_sha256: String,
    pub baseline_event_map_receipt_sha256: String,
    pub baseline_schedule_sha256: String,
    pub baseline_renderer_receipt_sha256: String,
    pub counterfactual_renderer_receipt_sha256: String,
    pub baseline_audio_sha256: String,
    pub counterfactual_audio_sha256: String,
    pub repeatability_calibration_sha256: String,
    pub renderer_identity_sha256: String,
    pub renderer_config_sha256: String,
    pub effects_group_assignment: FluidSynthEffectsGroupAssignment,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub output_frame_count: u64,
    pub disposition: InfluenceDisposition,
    pub support: EventInteractionSupport,
    /// Domain-separated hash over this evidence with both digest fields blank.
    pub evidence_sha256: String,
}

#[derive(Debug, Clone)]
pub struct VerifiedCounterfactualEventSupport {
    evidence: CounterfactualEventSupportEvidence,
}

impl VerifiedCounterfactualEventSupport {
    pub fn evidence(&self) -> &CounterfactualEventSupportEvidence {
        &self.evidence
    }

    pub fn planner_support(&self) -> &EventInteractionSupport {
        &self.evidence.support
    }
}

/// Aggregate EVAL-017 event scans for one exact source score, baseline render,
/// renderer profile and effects-group assignment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterfactualSupportAdapterReport {
    pub version: String,
    pub source_score_sha256: String,
    pub baseline_event_map_artifact_sha256: String,
    pub baseline_event_map_receipt_sha256: String,
    pub baseline_schedule_sha256: String,
    pub baseline_renderer_receipt_sha256: String,
    pub baseline_audio_sha256: String,
    pub renderer_identity_sha256: String,
    pub renderer_config_sha256: String,
    pub effects_group_assignment: FluidSynthEffectsGroupAssignment,
    pub sample_rate_hz: u32,
    pub channels: usize,
    /// Exact rendered PCM output horizon, not source/input MIDI frames.
    pub output_frame_count: u64,
    pub event_count: usize,
    pub events: Vec<CounterfactualEventSupportEvidence>,
    /// False until the complete downstream/master route graph is verified.
    pub routing_complete: bool,
    pub routing_incomplete_reason: String,
    pub receipt_sha256: String,
}

impl CounterfactualSupportAdapterReport {
    /// The vector can be passed directly to EVAL-019. Its per-event support
    /// digests are mandatory even for measured-empty support.
    pub fn planner_event_supports(&self) -> Vec<EventInteractionSupport> {
        self.events
            .iter()
            .map(|event| event.support.clone())
            .collect()
    }

    /// True means the support manifest is structurally ready for EVAL-019.
    /// It does not mean routing is complete: the planner must conservatively
    /// expand all pairs when routing_complete is false.
    pub fn planner_ready(&self) -> bool {
        self.events.len() == self.event_count
            && self.event_count >= 2
            && !self.routing_complete
            && self.events.iter().enumerate().all(|(index, event)| {
                event.source_event_index == index
                    && event.support.source_event_index == index
                    && event.support.support_evidence_sha256.as_deref()
                        == Some(event.evidence_sha256.as_str())
                    && is_sha256(&event.evidence_sha256)
            })
            && adapter_report_sha256(self)
                .is_ok_and(|digest| self.receipt_sha256 == digest)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CounterfactualSupportAdapterError {
    InfluenceReplay(CounterfactualInfluenceError),
    InvalidEventIdentity,
    MissingEffectsGroupEvidence,
    InvalidEffectsGroupAssignment,
    InvalidSupportIntervals,
    EventCountMismatch,
    DuplicateOrMissingEvent,
    InconsistentSourceOrRendererContext,
    InvalidEvidenceReceipt,
    InvalidAdapterReceipt,
    SerializationFailure,
}

impl std::fmt::Display for CounterfactualSupportAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "counterfactual support adapter failed: {self:?}")
    }
}

impl std::error::Error for CounterfactualSupportAdapterError {}

impl From<CounterfactualInfluenceError> for CounterfactualSupportAdapterError {
    fn from(value: CounterfactualInfluenceError) -> Self {
        Self::InfluenceReplay(value)
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_event_index(value: &str) -> Option<usize> {
    let suffix = value.strip_prefix("score_event_index:")?;
    if suffix.is_empty()
        || (suffix.len() > 1 && suffix.starts_with('0'))
        || !suffix.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    suffix.parse().ok()
}

fn hash_component(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn event_evidence_sha256(
    evidence: &CounterfactualEventSupportEvidence,
) -> Result<String, CounterfactualSupportAdapterError> {
    let mut unsigned = evidence.clone();
    unsigned.evidence_sha256.clear();
    unsigned.support.support_evidence_sha256 = None;
    let bytes = serde_json::to_vec(&unsigned)
        .map_err(|_| CounterfactualSupportAdapterError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(EVENT_EVIDENCE_DOMAIN);
    hash_component(&mut hasher, &bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn adapter_report_sha256(
    report: &CounterfactualSupportAdapterReport,
) -> Result<String, CounterfactualSupportAdapterError> {
    let mut unsigned = report.clone();
    unsigned.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&unsigned)
        .map_err(|_| CounterfactualSupportAdapterError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(ADAPTER_REPORT_DOMAIN);
    hash_component(&mut hasher, &bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn effects_assignment(
    baseline: &VerifiedRendererAudioBinding,
    counterfactual: &VerifiedRendererAudioBinding,
) -> Result<FluidSynthEffectsGroupAssignment, CounterfactualSupportAdapterError> {
    let baseline_groups = baseline
        .fluid_synth_effects_groups()
        .ok_or(CounterfactualSupportAdapterError::MissingEffectsGroupEvidence)?;
    let counterfactual_groups = counterfactual
        .fluid_synth_effects_groups()
        .ok_or(CounterfactualSupportAdapterError::MissingEffectsGroupEvidence)?;
    if baseline_groups != counterfactual_groups {
        return Err(CounterfactualSupportAdapterError::InconsistentSourceOrRendererContext);
    }
    let assignment = build_effects_group_assignment(baseline_groups)
        .ok_or(CounterfactualSupportAdapterError::InvalidEffectsGroupAssignment)?;
    if !verify_effects_group_assignment(&assignment) {
        return Err(CounterfactualSupportAdapterError::InvalidEffectsGroupAssignment);
    }
    Ok(assignment)
}

fn build_event_support(
    report: &CounterfactualInfluenceReport,
    input: &CounterfactualInfluenceInput<'_, '_>,
) -> Result<VerifiedCounterfactualEventSupport, CounterfactualSupportAdapterError> {
    let source_event_index = parse_event_index(input.source_event_id)
        .ok_or(CounterfactualSupportAdapterError::InvalidEventIdentity)?;
    if report.source_event_id != input.source_event_id
        || source_event_index >= input.expected_score_event_count
        || input.baseline_audio.channels == 0
        || report.total_frames as u64
            != input.baseline_audio.interleaved_samples.len() as u64
                / input.baseline_audio.channels as u64
    {
        return Err(CounterfactualSupportAdapterError::InvalidEventIdentity);
    }

    let assignment = effects_assignment(
        input.baseline_renderer_evidence,
        input.counterfactual_renderer_evidence,
    )?;
    let baseline_channels: BTreeSet<u8> = input
        .baseline_renderer_evidence
        .performed_event_lineage()
        .iter()
        .filter(|lineage| lineage.source_event_id == input.source_event_id)
        .map(|lineage| lineage.midi_channel)
        .collect();
    if baseline_channels.is_empty() {
        return Err(CounterfactualSupportAdapterError::InvalidEventIdentity);
    }

    let mut domains = BTreeSet::new();
    for channel in baseline_channels {
        let group = effects_group_for_midi_channel(
            channel,
            assignment.effects_group_count,
        )
        .ok_or(CounterfactualSupportAdapterError::InvalidEffectsGroupAssignment)?;
        domains.insert(format!("fluidsynth:effects-group:{group}"));
    }

    if report.measured_influence_intervals.len() > MAX_SUPPORT_INTERVALS_PER_EVENT {
        return Err(CounterfactualSupportAdapterError::InvalidSupportIntervals);
    }
    let mut acoustic_support = Vec::with_capacity(report.measured_influence_intervals.len());
    for interval in &report.measured_influence_intervals {
        let start_frame = u64::try_from(interval.start_frame)
            .map_err(|_| CounterfactualSupportAdapterError::InvalidSupportIntervals)?;
        let end_frame = u64::try_from(interval.end_frame)
            .map_err(|_| CounterfactualSupportAdapterError::InvalidSupportIntervals)?;
        if start_frame >= end_frame || end_frame > report.total_frames as u64 {
            return Err(CounterfactualSupportAdapterError::InvalidSupportIntervals);
        }
        acoustic_support.push(AcousticSupportInterval { start_frame, end_frame });
    }
    acoustic_support.sort_unstable();
    let mut merged: Vec<AcousticSupportInterval> = Vec::with_capacity(acoustic_support.len());
    for interval in acoustic_support {
        if let Some(previous) = merged.last_mut() {
            if interval.start_frame <= previous.end_frame {
                previous.end_frame = previous.end_frame.max(interval.end_frame);
                continue;
            }
        }
        merged.push(interval);
    }

    let mut evidence = CounterfactualEventSupportEvidence {
        version: COUNTERFACTUAL_SUPPORT_ADAPTER_VERSION.to_owned(),
        source_event_index,
        source_event_id: input.source_event_id.to_owned(),
        expected_score_event_count: input.expected_score_event_count,
        intervention: report.intervention.clone(),
        influence_report_sha256: report.receipt_sha256.clone(),
        source_score_sha256: report.source_score_sha256.clone(),
        baseline_event_map_artifact_sha256: report.performed_event_map_sha256.clone(),
        baseline_event_map_receipt_sha256: report.performed_event_map_receipt_sha256.clone(),
        baseline_schedule_sha256: report.baseline_schedule_sha256.clone(),
        baseline_renderer_receipt_sha256: report.baseline_renderer_receipt_sha256.clone(),
        counterfactual_renderer_receipt_sha256: report.counterfactual_renderer_receipt_sha256.clone(),
        baseline_audio_sha256: report.baseline_audio_sha256.clone(),
        counterfactual_audio_sha256: report.counterfactual_audio_sha256.clone(),
        repeatability_calibration_sha256: report.repeatability_calibration_sha256.clone(),
        renderer_identity_sha256: report.renderer_identity_sha256.clone(),
        renderer_config_sha256: report.renderer_config_sha256.clone(),
        effects_group_assignment: assignment,
        sample_rate_hz: report.sample_rate_hz,
        channels: report.channels,
        output_frame_count: report.total_frames as u64,
        disposition: report.disposition,
        support: EventInteractionSupport {
            source_event_index,
            acoustic_support: merged,
            routing_domains: domains.into_iter().collect(),
            // Partial effects-group topology is not a complete render graph.
            // EVAL-019 will conservatively expand pair scope until this changes.
            routing_complete: false,
            support_evidence_sha256: None,
        },
        evidence_sha256: String::new(),
    };
    let digest = event_evidence_sha256(&evidence)?;
    evidence.evidence_sha256 = digest.clone();
    evidence.support.support_evidence_sha256 = Some(digest);
    Ok(VerifiedCounterfactualEventSupport { evidence })
}

/// Verify one EVAL-017 report against the exact artifacts/PCM/no-op calibration,
/// then bind its measured intervals and EVAL-029 route assignment to one event.
pub fn verify_counterfactual_event_support(
    report: &CounterfactualInfluenceReport,
    input: &CounterfactualInfluenceInput<'_, '_>,
    calibration: &VerifiedRepeatabilityCalibration,
) -> Result<VerifiedCounterfactualEventSupport, CounterfactualSupportAdapterError> {
    verify_counterfactual_influence(report, input, calibration)?;
    build_event_support(report, input)
}

/// Replay a serialized event-support evidence item from its original EVAL-017
/// report/input/calibration. The opaque token is not deserialized from a digest.
pub fn verify_counterfactual_event_support_evidence(
    supplied: &CounterfactualEventSupportEvidence,
    report: &CounterfactualInfluenceReport,
    input: &CounterfactualInfluenceInput<'_, '_>,
    calibration: &VerifiedRepeatabilityCalibration,
) -> Result<bool, CounterfactualSupportAdapterError> {
    if !is_sha256(&supplied.evidence_sha256)
        || supplied.evidence_sha256 != event_evidence_sha256(supplied)?
        || supplied.support.support_evidence_sha256.as_deref()
            != Some(supplied.evidence_sha256.as_str())
    {
        return Ok(false);
    }
    let expected = verify_counterfactual_event_support(report, input, calibration)?;
    Ok(expected.evidence == *supplied)
}

/// Complete original inputs required to replay one per-event evidence record.
pub struct CounterfactualSupportReplayInput<'a, 'view, 'audio> {
    pub report: &'a CounterfactualInfluenceReport,
    pub input: &'a CounterfactualInfluenceInput<'view, 'audio>,
    pub calibration: &'a VerifiedRepeatabilityCalibration,
}

fn assemble_from_verified_events(
    events: &[VerifiedCounterfactualEventSupport],
) -> Result<CounterfactualSupportAdapterReport, CounterfactualSupportAdapterError> {
    if events.len() < 2 || events.len() > MAX_ADAPTER_EVENTS {
        return Err(CounterfactualSupportAdapterError::EventCountMismatch);
    }
    let mut ordered: Vec<&CounterfactualEventSupportEvidence> =
        events.iter().map(|event| &event.evidence).collect();
    ordered.sort_by_key(|event| event.source_event_index);
    let first = ordered
        .first()
        .ok_or(CounterfactualSupportAdapterError::EventCountMismatch)?;
    for (expected_index, event) in ordered.iter().enumerate() {
        if event.source_event_index != expected_index
            || event.support.source_event_index != expected_index
            || event.expected_score_event_count != ordered.len()
            || !is_sha256(&event.evidence_sha256)
            || event.evidence_sha256 != event_evidence_sha256(event)?
            || event.support.support_evidence_sha256.as_deref()
                != Some(event.evidence_sha256.as_str())
            || event.version != COUNTERFACTUAL_SUPPORT_ADAPTER_VERSION
            || event.support.routing_complete
        {
            return Err(CounterfactualSupportAdapterError::DuplicateOrMissingEvent);
        }
        if event.source_score_sha256 != first.source_score_sha256
            || event.baseline_event_map_artifact_sha256
                != first.baseline_event_map_artifact_sha256
            || event.baseline_event_map_receipt_sha256
                != first.baseline_event_map_receipt_sha256
            || event.baseline_schedule_sha256 != first.baseline_schedule_sha256
            || event.baseline_renderer_receipt_sha256
                != first.baseline_renderer_receipt_sha256
            || event.baseline_audio_sha256 != first.baseline_audio_sha256
            || event.renderer_identity_sha256 != first.renderer_identity_sha256
            || event.renderer_config_sha256 != first.renderer_config_sha256
            || event.effects_group_assignment != first.effects_group_assignment
            || event.sample_rate_hz != first.sample_rate_hz
            || event.channels != first.channels
            || event.output_frame_count != first.output_frame_count
        {
            return Err(CounterfactualSupportAdapterError::InconsistentSourceOrRendererContext);
        }
    }

    let mut result = CounterfactualSupportAdapterReport {
        version: COUNTERFACTUAL_SUPPORT_ADAPTER_VERSION.to_owned(),
        source_score_sha256: first.source_score_sha256.clone(),
        baseline_event_map_artifact_sha256: first.baseline_event_map_artifact_sha256.clone(),
        baseline_event_map_receipt_sha256: first.baseline_event_map_receipt_sha256.clone(),
        baseline_schedule_sha256: first.baseline_schedule_sha256.clone(),
        baseline_renderer_receipt_sha256: first.baseline_renderer_receipt_sha256.clone(),
        baseline_audio_sha256: first.baseline_audio_sha256.clone(),
        renderer_identity_sha256: first.renderer_identity_sha256.clone(),
        renderer_config_sha256: first.renderer_config_sha256.clone(),
        effects_group_assignment: first.effects_group_assignment.clone(),
        sample_rate_hz: first.sample_rate_hz,
        channels: first.channels,
        output_frame_count: first.output_frame_count,
        event_count: ordered.len(),
        events: ordered.into_iter().cloned().collect(),
        routing_complete: false,
        routing_incomplete_reason: ROUTING_INCOMPLETE_REASON.to_owned(),
        receipt_sha256: String::new(),
    };
    result.receipt_sha256 = adapter_report_sha256(&result)?;
    Ok(result)
}

/// Assemble one complete, canonical EVAL-019 event-support vector. Every
/// source event must be present exactly once, including measured-empty events.
pub fn assemble_counterfactual_support_adapter_report(
    events: &[VerifiedCounterfactualEventSupport],
) -> Result<CounterfactualSupportAdapterReport, CounterfactualSupportAdapterError> {
    assemble_from_verified_events(events)
}

/// Replay the aggregate adapter report from every original EVAL-017 report,
/// renderer-bound input and verified repeatability calibration.
pub fn verify_counterfactual_support_adapter_report(
    supplied: &CounterfactualSupportAdapterReport,
    replay_inputs: &[CounterfactualSupportReplayInput<'_, '_, '_>],
) -> Result<bool, CounterfactualSupportAdapterError> {
    if supplied.version != COUNTERFACTUAL_SUPPORT_ADAPTER_VERSION
        || !is_sha256(&supplied.receipt_sha256)
        || supplied.receipt_sha256 != adapter_report_sha256(supplied)?
    {
        return Ok(false);
    }
    let mut verified = Vec::with_capacity(replay_inputs.len());
    for replay in replay_inputs {
        verified.push(verify_counterfactual_event_support(
            replay.report,
            replay.input,
            replay.calibration,
        )?);
    }
    let expected = assemble_from_verified_events(&verified)?;
    Ok(expected == *supplied)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_support_evidence_digest_is_mandatory_even_for_empty_measurement() {
        let empty = EventInteractionSupport {
            source_event_index: 0,
            acoustic_support: Vec::new(),
            routing_domains: vec!["fluidsynth:effects-group:0".to_owned()],
            routing_complete: false,
            support_evidence_sha256: None,
        };
        assert!(empty.support_evidence_sha256.is_none());
    }

    #[test]
    fn effect_group_channel_mapping_matches_the_receipt_bound_assignment() {
        let assignment = build_effects_group_assignment(4).unwrap();
        assert!(verify_effects_group_assignment(&assignment));
        assert_eq!(effects_group_for_midi_channel(0, 4), Some(0));
        assert_eq!(effects_group_for_midi_channel(4, 4), Some(0));
        assert_eq!(effects_group_for_midi_channel(7, 4), Some(3));
        assert_eq!(effects_group_for_midi_channel(16, 4), None);
    }
}
