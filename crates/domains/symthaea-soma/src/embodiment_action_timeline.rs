// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Clock-bound, phase-anchored action schedules for polyembodied output.
//!
//! This module describes *when* already-authorized output components may run.
//! It does not grant permission, synthesize media, schedule actuators, or prove
//! that a renderer/device stopped. Policy authorization and physical/output
//! observation remain independent gates.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

pub const ACTION_TIMELINE_SCHEMA_V1: &str = "symthaea.action-timeline.v1";
const ACTION_TIMELINE_DIGEST_CONTEXT: &str = "symthaea.action-timeline.v1";
pub const MAX_ACTION_SEGMENTS_V1: usize = 256;
pub const MAX_ACTION_TIMELINE_DURATION_NS: u64 = 60_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionChannel {
    Speech,
    Viseme,
    Gesture,
    Gaze,
    Expression,
    Posture,
    Locomotion,
    SpatialCue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionPhase {
    Prepare,
    Onset,
    Sustain,
    Release,
    Recovery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    SpeechUtterance,
    Viseme,
    Gesture,
    GazeShift,
    Expression,
    PostureShift,
    Locomotion,
    SpatialCue,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionSegmentV1 {
    pub segment_id: String,
    pub channel: ActionChannel,
    pub phase: ActionPhase,
    pub kind: ActionKind,
    pub starts_at_monotonic_ns: u64,
    pub ends_at_monotonic_ns: u64,
    pub synchronization_group_id: Option<String>,
    /// Digest of an opaque payload reference. Raw private media/text is not
    /// copied into the schedule.
    pub payload_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionTimelineV1 {
    pub schema: String,
    pub plan_id: String,
    pub session_id: String,
    pub event_id: String,
    pub event_sequence: u64,
    pub clock_domain_id: String,
    pub source_timestamp_monotonic_ns: u64,
    pub expires_at_monotonic_ns: u64,
    pub canonical_cue_digest: String,
    pub segments: Vec<ActionSegmentV1>,
    pub timeline_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionCancellationReason {
    UserBargeIn,
    UserStop,
    PolicyRevocation,
    Superseded,
    Expired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionCancellationV1 {
    pub plan_id: String,
    pub clock_domain_id: String,
    pub cancelled_at_monotonic_ns: u64,
    pub cause_event_id: String,
    pub reason: ActionCancellationReason,
    pub replacement_plan_id: Option<String>,
    pub timeline_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionTimelineError {
    WrongSchema,
    InvalidIdentifier,
    InvalidDigest,
    InvalidTimeRange,
    TimelineTooLong,
    InvalidSegmentCount,
    DuplicateSegmentId,
    NonCanonicalOrder,
    WrongChannelForAction,
    OverlappingExclusiveChannel,
    InvalidSynchronizationGroup,
    TimelineDigestMismatch,
    WrongCancellationTarget,
    WrongClockDomain,
    CancellationOutsideLifetime,
    MissingReplacementPlan,
    UnexpectedReplacementPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionExecutionError {
    WrongExecutionClockDomain,
    ExecutionClockRegression,
    ExecutionBeforeTimeline,
    CancellationTimestampInFuture,
    InvalidExecutionTimeline(ActionTimelineError),
}

/// Sorts a schedule deterministically, computes its domain-separated digest,
/// and runs all structural checks before returning it.
pub fn compile_action_timeline_v1(
    mut timeline: ActionTimelineV1,
) -> Result<ActionTimelineV1, ActionTimelineError> {
    timeline.schema = ACTION_TIMELINE_SCHEMA_V1.to_owned();
    timeline.segments.sort_by(segment_order);
    timeline.timeline_digest.clear();
    timeline.timeline_digest = compute_action_timeline_digest(&timeline)?;
    verify_action_timeline_v1(&timeline)?;
    Ok(timeline)
}

pub fn compute_action_timeline_digest(
    timeline: &ActionTimelineV1,
) -> Result<String, ActionTimelineError> {
    let mut material = timeline.clone();
    material.timeline_digest.clear();
    let bytes = serde_json::to_vec(&material).map_err(|_| ActionTimelineError::InvalidIdentifier)?;
    let mut hasher = blake3::Hasher::new_derive_key(ACTION_TIMELINE_DIGEST_CONTEXT);
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

/// Structural verification only. A valid timeline is not an authorization
/// token, a device command, or evidence of emitted/observed output.
pub fn verify_action_timeline_v1(
    timeline: &ActionTimelineV1,
) -> Result<(), ActionTimelineError> {
    if timeline.schema != ACTION_TIMELINE_SCHEMA_V1 {
        return Err(ActionTimelineError::WrongSchema);
    }
    for value in [&timeline.plan_id, &timeline.session_id, &timeline.event_id, &timeline.clock_domain_id] {
        validate_identifier(value)?;
    }
    validate_digest(&timeline.canonical_cue_digest)?;
    validate_digest(&timeline.timeline_digest)?;
    if timeline.expires_at_monotonic_ns <= timeline.source_timestamp_monotonic_ns {
        return Err(ActionTimelineError::InvalidTimeRange);
    }
    let duration = timeline.expires_at_monotonic_ns - timeline.source_timestamp_monotonic_ns;
    if duration > MAX_ACTION_TIMELINE_DURATION_NS {
        return Err(ActionTimelineError::TimelineTooLong);
    }
    if timeline.segments.is_empty() || timeline.segments.len() > MAX_ACTION_SEGMENTS_V1 {
        return Err(ActionTimelineError::InvalidSegmentCount);
    }

    let mut ids = HashSet::with_capacity(timeline.segments.len());
    let mut groups: BTreeMap<&str, (usize, HashSet<ActionChannel>)> = BTreeMap::new();
    let mut last_phase_by_group_channel: BTreeMap<(&str, ActionChannel), (u64, ActionPhase)> = BTreeMap::new();
    let mut last_end_by_channel: BTreeMap<ActionChannel, u64> = BTreeMap::new();
    let mut previous: Option<&ActionSegmentV1> = None;

    for segment in &timeline.segments {
        validate_identifier(&segment.segment_id)?;
        validate_digest(&segment.payload_digest)?;
        if !ids.insert(segment.segment_id.as_str()) {
            return Err(ActionTimelineError::DuplicateSegmentId);
        }
        if segment.starts_at_monotonic_ns < timeline.source_timestamp_monotonic_ns
            || segment.ends_at_monotonic_ns <= segment.starts_at_monotonic_ns
            || segment.ends_at_monotonic_ns > timeline.expires_at_monotonic_ns
        {
            return Err(ActionTimelineError::InvalidTimeRange);
        }
        if !action_matches_channel(segment.kind, segment.channel) {
            return Err(ActionTimelineError::WrongChannelForAction);
        }
        if let Some(previous) = previous {
            if segment_order(previous, segment).is_gt() {
                return Err(ActionTimelineError::NonCanonicalOrder);
            }
        }
        previous = Some(segment);

        if let Some(last_end) = last_end_by_channel.get(&segment.channel) {
            if segment.starts_at_monotonic_ns < *last_end {
                return Err(ActionTimelineError::OverlappingExclusiveChannel);
            }
        }
        last_end_by_channel.insert(segment.channel, segment.ends_at_monotonic_ns);

        if let Some(group_id) = segment.synchronization_group_id.as_deref() {
            validate_identifier(group_id)?;
            let entry = groups.entry(group_id).or_insert_with(|| (0, HashSet::new()));
            entry.0 += 1;
            entry.1.insert(segment.channel);
            let key = (group_id, segment.channel);
            if let Some((last_start, last_phase)) = last_phase_by_group_channel.get(&key) {
                if segment.starts_at_monotonic_ns > *last_start && segment.phase < *last_phase {
                    return Err(ActionTimelineError::InvalidSynchronizationGroup);
                }
            }
            last_phase_by_group_channel.insert(key, (segment.starts_at_monotonic_ns, segment.phase));
        }
    }

    if groups.values().any(|(count, channels)| *count < 2 || channels.len() < 2) {
        return Err(ActionTimelineError::InvalidSynchronizationGroup);
    }

    let expected = compute_action_timeline_digest(timeline)?;
    if expected != timeline.timeline_digest {
        return Err(ActionTimelineError::TimelineDigestMismatch);
    }
    Ok(())
}

pub fn verify_action_cancellation_v1(
    timeline: &ActionTimelineV1,
    cancellation: &ActionCancellationV1,
) -> Result<(), ActionTimelineError> {
    verify_action_timeline_v1(timeline)?;
    if cancellation.plan_id != timeline.plan_id || cancellation.timeline_digest != timeline.timeline_digest {
        return Err(ActionTimelineError::WrongCancellationTarget);
    }
    if cancellation.clock_domain_id != timeline.clock_domain_id {
        return Err(ActionTimelineError::WrongClockDomain);
    }
    validate_identifier(&cancellation.cause_event_id)?;
    if cancellation.cancelled_at_monotonic_ns < timeline.source_timestamp_monotonic_ns {
        return Err(ActionTimelineError::CancellationOutsideLifetime);
    }
    match cancellation.reason {
        ActionCancellationReason::Expired => {
            if cancellation.cancelled_at_monotonic_ns < timeline.expires_at_monotonic_ns {
                return Err(ActionTimelineError::CancellationOutsideLifetime);
            }
        }
        _ => {
            if cancellation.cancelled_at_monotonic_ns >= timeline.expires_at_monotonic_ns {
                return Err(ActionTimelineError::CancellationOutsideLifetime);
            }
        }
    }
    match (cancellation.reason, cancellation.replacement_plan_id.as_deref()) {
        (ActionCancellationReason::Superseded, Some(id)) => validate_identifier(id)?,
        (ActionCancellationReason::Superseded, None) => return Err(ActionTimelineError::MissingReplacementPlan),
        (_, Some(_)) => return Err(ActionTimelineError::UnexpectedReplacementPlan),
        _ => {}
    }
    Ok(())
}


/// One deterministic scheduling decision. `stop_required_segment_ids` means
/// the caller must request an actuator stop; it does not prove that the
/// physical/audio output has stopped. Only a device observer can establish
/// that fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionExecutionTick {
    pub dispatch_segment_ids: Vec<String>,
    pub missed_segment_ids: Vec<String>,
    /// Segments whose scheduled end was reached; the adapter must issue a stop.
    /// This is not physical completion proof.
    pub scheduled_stop_segment_ids: Vec<String>,
    /// Active segments whose actuators should be stopped immediately.
    pub stop_required_segment_ids: Vec<String>,
    pub cancelled: bool,
    pub expired: bool,
}

/// Deterministic runtime gate over a previously compiled and verified plan.
///
/// It does not send commands to an output device. It makes one-shot dispatch,
/// stale-segment suppression, monotonic-clock checks, and cancellation/expiry
/// decisions that an audio/visual/haptic adapter can consume.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionExecutionGate {
    timeline: ActionTimelineV1,
    clock_domain_id: String,
    last_now_monotonic_ns: Option<u64>,
    seen_segment_ids: HashSet<String>,
    active_segment_ids: HashSet<String>,
    stop_command_accepted_ids: HashSet<String>,
    cancelled: bool,
    expired: bool,
}

impl ActionExecutionGate {
    pub fn new(
        timeline: ActionTimelineV1,
        clock_domain_id: impl Into<String>,
    ) -> Result<Self, ActionExecutionError> {
        verify_action_timeline_v1(&timeline)
            .map_err(ActionExecutionError::InvalidExecutionTimeline)?;
        let clock_domain_id = clock_domain_id.into();
        validate_identifier(&clock_domain_id)
            .map_err(|_| ActionExecutionError::WrongExecutionClockDomain)?;
        if timeline.clock_domain_id != clock_domain_id {
            return Err(ActionExecutionError::WrongExecutionClockDomain);
        }
        Ok(Self {
            timeline,
            clock_domain_id,
            last_now_monotonic_ns: None,
            seen_segment_ids: HashSet::new(),
            active_segment_ids: HashSet::new(),
            stop_command_accepted_ids: HashSet::new(),
            cancelled: false,
            expired: false,
        })
    }

    pub fn timeline_digest(&self) -> &str {
        &self.timeline.timeline_digest
    }

    pub fn is_cancelled(&self) -> bool { self.cancelled }
    pub fn is_expired(&self) -> bool { self.expired }

    /// Record that the output adapter accepted a stop command for a dispatched
    /// segment. This is not independent evidence that output physically ceased.
    /// Returns false for unknown or already-acknowledged segments.
    pub fn acknowledge_stop_command_accepted(&mut self, segment_id: &str) -> bool {
        if self.active_segment_ids.remove(segment_id) {
            self.stop_command_accepted_ids.insert(segment_id.to_owned());
            true
        } else {
            false
        }
    }

    /// Advance the schedule at a timestamp from the timeline's source clock.
    ///
    /// A segment is dispatched at most once and only while its scheduled
    /// interval is live. A segment first noticed after its end is classified
    /// as missed and is never sent late. Cancellation and expiry suppress all
    /// future dispatch; every currently active segment is returned as
    /// stop-required exactly once.
    pub fn advance(
        &mut self,
        now_monotonic_ns: u64,
        clock_domain_id: &str,
        cancellation: Option<&ActionCancellationV1>,
    ) -> Result<ActionExecutionTick, ActionExecutionError> {
        if clock_domain_id != self.clock_domain_id {
            return Err(ActionExecutionError::WrongExecutionClockDomain);
        }
        if let Some(previous) = self.last_now_monotonic_ns {
            if now_monotonic_ns < previous {
                return Err(ActionExecutionError::ExecutionClockRegression);
            }
        }
        if now_monotonic_ns < self.timeline.source_timestamp_monotonic_ns {
            return Err(ActionExecutionError::ExecutionBeforeTimeline);
        }
        self.last_now_monotonic_ns = Some(now_monotonic_ns);

        if self.cancelled || self.expired {
            let mut stop_required_segment_ids: Vec<_> =
                self.active_segment_ids.iter().cloned().collect();
            stop_required_segment_ids.sort();
            let mut tick = empty_execution_tick(self.cancelled, self.expired);
            tick.stop_required_segment_ids = stop_required_segment_ids;
            return Ok(tick);
        }

        if now_monotonic_ns >= self.timeline.expires_at_monotonic_ns {
            self.expired = true;
            let mut stop_required_segment_ids: Vec<_> =
                self.active_segment_ids.iter().cloned().collect();
            stop_required_segment_ids.sort();
            return Ok(ActionExecutionTick {
                dispatch_segment_ids: Vec::new(),
                missed_segment_ids: Vec::new(),
                scheduled_stop_segment_ids: Vec::new(),
                stop_required_segment_ids,
                cancelled: false,
                expired: true,
            });
        }

        if let Some(cancellation) = cancellation {
            verify_action_cancellation_v1(&self.timeline, cancellation)
                .map_err(ActionExecutionError::InvalidExecutionTimeline)?;
            if cancellation.cancelled_at_monotonic_ns > now_monotonic_ns {
                return Err(ActionExecutionError::CancellationTimestampInFuture);
            }
            self.cancelled = true;
            let mut stop_required_segment_ids: Vec<_> =
                self.active_segment_ids.iter().cloned().collect();
            stop_required_segment_ids.sort();
            return Ok(ActionExecutionTick {
                dispatch_segment_ids: Vec::new(),
                missed_segment_ids: Vec::new(),
                scheduled_stop_segment_ids: Vec::new(),
                stop_required_segment_ids,
                cancelled: true,
                expired: false,
            });
        }

        let mut scheduled_stop_segment_ids = Vec::new();
        for id in &self.active_segment_ids {
            let scheduled_end_reached = self.timeline.segments.iter().any(|segment| {
                segment.segment_id == *id
                    && segment.ends_at_monotonic_ns <= now_monotonic_ns
            });
            if scheduled_end_reached && !self.stop_command_accepted_ids.contains(id) {
                scheduled_stop_segment_ids.push(id.clone());
            }
        }

        let mut dispatch_segment_ids = Vec::new();
        let mut missed_segment_ids = Vec::new();
        for segment in &self.timeline.segments {
            if self.seen_segment_ids.contains(&segment.segment_id) {
                continue;
            }
            if segment.starts_at_monotonic_ns <= now_monotonic_ns
                && now_monotonic_ns < segment.ends_at_monotonic_ns
            {
                self.seen_segment_ids.insert(segment.segment_id.clone());
                self.active_segment_ids.insert(segment.segment_id.clone());
                dispatch_segment_ids.push(segment.segment_id.clone());
            } else if segment.ends_at_monotonic_ns <= now_monotonic_ns {
                self.seen_segment_ids.insert(segment.segment_id.clone());
                missed_segment_ids.push(segment.segment_id.clone());
            }
        }
        dispatch_segment_ids.sort();
        missed_segment_ids.sort();
        scheduled_stop_segment_ids.sort();

        Ok(ActionExecutionTick {
            dispatch_segment_ids,
            missed_segment_ids,
            scheduled_stop_segment_ids,
            stop_required_segment_ids: Vec::new(),
            cancelled: false,
            expired: false,
        })
    }
}

fn empty_execution_tick(cancelled: bool, expired: bool) -> ActionExecutionTick {
    ActionExecutionTick {
        dispatch_segment_ids: Vec::new(),
        missed_segment_ids: Vec::new(),
        scheduled_stop_segment_ids: Vec::new(),
        stop_required_segment_ids: Vec::new(),
        cancelled,
        expired,
    }
}

fn action_matches_channel(kind: ActionKind, channel: ActionChannel) -> bool {
    matches!(
        (kind, channel),
        (ActionKind::SpeechUtterance, ActionChannel::Speech)
            | (ActionKind::Viseme, ActionChannel::Viseme)
            | (ActionKind::Gesture, ActionChannel::Gesture)
            | (ActionKind::GazeShift, ActionChannel::Gaze)
            | (ActionKind::Expression, ActionChannel::Expression)
            | (ActionKind::PostureShift, ActionChannel::Posture)
            | (ActionKind::Locomotion, ActionChannel::Locomotion)
            | (ActionKind::SpatialCue, ActionChannel::SpatialCue)
    )
}

fn segment_order(left: &ActionSegmentV1, right: &ActionSegmentV1) -> std::cmp::Ordering {
    left.starts_at_monotonic_ns
        .cmp(&right.starts_at_monotonic_ns)
        .then_with(|| left.channel.cmp(&right.channel))
        .then_with(|| left.phase.cmp(&right.phase))
        .then_with(|| left.segment_id.cmp(&right.segment_id))
}

fn validate_identifier(value: &str) -> Result<(), ActionTimelineError> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(ActionTimelineError::InvalidIdentifier);
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), ActionTimelineError> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ActionTimelineError::InvalidDigest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: u8) -> String { std::iter::repeat_n(ch as char, 64).collect() }

    fn segment(id: &str, channel: ActionChannel, kind: ActionKind, start: u64, end: u64, group: Option<&str>) -> ActionSegmentV1 {
        ActionSegmentV1 {
            segment_id: id.to_owned(),
            channel,
            phase: ActionPhase::Onset,
            kind,
            starts_at_monotonic_ns: start,
            ends_at_monotonic_ns: end,
            synchronization_group_id: group.map(str::to_owned),
            payload_digest: digest(b'a'),
        }
    }

    fn timeline() -> ActionTimelineV1 {
        ActionTimelineV1 {
            schema: ACTION_TIMELINE_SCHEMA_V1.to_owned(),
            plan_id: "plan-1".to_owned(),
            session_id: "session-1".to_owned(),
            event_id: "event-1".to_owned(),
            event_sequence: 1,
            clock_domain_id: "clock:source-1".to_owned(),
            source_timestamp_monotonic_ns: 1_000,
            expires_at_monotonic_ns: 10_000,
            canonical_cue_digest: digest(b'b'),
            segments: vec![
                segment("gesture-1", ActionChannel::Gesture, ActionKind::Gesture, 2_000, 4_000, Some("sync-1")),
                segment("speech-1", ActionChannel::Speech, ActionKind::SpeechUtterance, 2_000, 5_000, Some("sync-1")),
                segment("viseme-1", ActionChannel::Viseme, ActionKind::Viseme, 2_000, 5_000, Some("sync-1")),
            ],
            timeline_digest: String::new(),
        }
    }

    #[test]
    fn compile_sorts_and_verifies_cross_channel_schedule() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        assert_eq!(compiled.segments[0].segment_id, "speech-1");
        assert_eq!(compiled.segments[1].segment_id, "viseme-1");
        assert_eq!(compiled.segments[2].segment_id, "gesture-1");
        verify_action_timeline_v1(&compiled).unwrap();
    }

    #[test]
    fn action_kind_channel_mismatch_is_rejected() {
        let mut value = timeline();
        value.segments[0].channel = ActionChannel::Gaze;
        assert_eq!(
            compile_action_timeline_v1(value),
            Err(ActionTimelineError::WrongChannelForAction)
        );
    }

    #[test]
    fn a_channel_cannot_overlap_itself() {
        let mut value = timeline();
        value.segments.push(segment("gesture-2", ActionChannel::Gesture, ActionKind::Gesture, 3_000, 6_000, None));
        assert_eq!(
            compile_action_timeline_v1(value),
            Err(ActionTimelineError::OverlappingExclusiveChannel)
        );
    }

    #[test]
    fn synchronization_group_requires_multiple_channels() {
        let mut value = timeline();
        value.segments[0].synchronization_group_id = Some("alone".to_owned());
        assert_eq!(
            compile_action_timeline_v1(value),
            Err(ActionTimelineError::InvalidSynchronizationGroup)
        );
    }

    #[test]
    fn schedule_cannot_outlive_parent_expiry() {
        let mut value = timeline();
        value.segments[0].ends_at_monotonic_ns = 11_000;
        assert_eq!(
            compile_action_timeline_v1(value),
            Err(ActionTimelineError::InvalidTimeRange)
        );
    }

    #[test]
    fn schedule_rejects_stale_start() {
        let mut value = timeline();
        value.segments[0].starts_at_monotonic_ns = 999;
        assert_eq!(
            compile_action_timeline_v1(value),
            Err(ActionTimelineError::InvalidTimeRange)
        );
    }

    #[test]
    fn schedule_digest_detects_mutation() {
        let mut compiled = compile_action_timeline_v1(timeline()).unwrap();
        compiled.segments[0].ends_at_monotonic_ns += 1;
        assert_eq!(verify_action_timeline_v1(&compiled), Err(ActionTimelineError::TimelineDigestMismatch));
    }

    #[test]
    fn cancellation_is_bound_to_plan_digest_and_clock() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let cancellation = ActionCancellationV1 {
            plan_id: compiled.plan_id.clone(),
            clock_domain_id: compiled.clock_domain_id.clone(),
            cancelled_at_monotonic_ns: 3_000,
            cause_event_id: "barge-in-1".to_owned(),
            reason: ActionCancellationReason::UserBargeIn,
            replacement_plan_id: None,
            timeline_digest: compiled.timeline_digest.clone(),
        };
        verify_action_cancellation_v1(&compiled, &cancellation).unwrap();
    }

    #[test]
    fn supersession_requires_an_explicit_replacement() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let cancellation = ActionCancellationV1 {
            plan_id: compiled.plan_id.clone(),
            clock_domain_id: compiled.clock_domain_id.clone(),
            cancelled_at_monotonic_ns: 3_000,
            cause_event_id: "new-event".to_owned(),
            reason: ActionCancellationReason::Superseded,
            replacement_plan_id: None,
            timeline_digest: compiled.timeline_digest.clone(),
        };
        assert_eq!(
            verify_action_cancellation_v1(&compiled, &cancellation),
            Err(ActionTimelineError::MissingReplacementPlan)
        );
    }

    #[test]
    fn cancellation_outside_lifetime_is_rejected() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let cancellation = ActionCancellationV1 {
            plan_id: compiled.plan_id.clone(),
            clock_domain_id: compiled.clock_domain_id.clone(),
            cancelled_at_monotonic_ns: 10_000,
            cause_event_id: "stop-1".to_owned(),
            reason: ActionCancellationReason::UserStop,
            replacement_plan_id: None,
            timeline_digest: compiled.timeline_digest.clone(),
        };
        assert_eq!(
            verify_action_cancellation_v1(&compiled, &cancellation),
            Err(ActionTimelineError::CancellationOutsideLifetime)
        );
    }

    #[test]
    fn synchronization_group_rejects_phase_regression_within_a_channel() {
        let mut value = timeline();
        value.segments[0].phase = ActionPhase::Release;
        value.segments.push(segment("speech-late", ActionChannel::Speech, ActionKind::SpeechUtterance, 6_000, 7_000, Some("sync-1")));
        value.segments.last_mut().unwrap().phase = ActionPhase::Onset;
        assert_eq!(
            compile_action_timeline_v1(value),
            Err(ActionTimelineError::InvalidSynchronizationGroup)
        );
    }


    #[test]
    fn execution_gate_dispatches_each_live_segment_once() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let mut gate = ActionExecutionGate::new(compiled, "clock:source-1").unwrap();
        let early = gate.advance(1_999, "clock:source-1", None).unwrap();
        assert!(early.dispatch_segment_ids.is_empty());
        let due = gate.advance(2_000, "clock:source-1", None).unwrap();
        assert_eq!(due.dispatch_segment_ids, vec!["gesture-1", "speech-1", "viseme-1"]);
        let repeated = gate.advance(2_001, "clock:source-1", None).unwrap();
        assert!(repeated.dispatch_segment_ids.is_empty());
    }

    #[test]
    fn execution_gate_never_dispatches_a_segment_after_its_window() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let mut gate = ActionExecutionGate::new(compiled, "clock:source-1").unwrap();
        let tick = gate.advance(6_000, "clock:source-1", None).unwrap();
        assert!(tick.dispatch_segment_ids.is_empty());
        assert_eq!(tick.missed_segment_ids, vec!["gesture-1", "speech-1", "viseme-1"]);
    }

    #[test]
    fn segment_end_emits_a_scheduled_stop_request() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let mut gate = ActionExecutionGate::new(compiled, "clock:source-1").unwrap();
        gate.advance(2_000, "clock:source-1", None).unwrap();
        let ended = gate.advance(4_000, "clock:source-1", None).unwrap();
        assert_eq!(ended.scheduled_stop_segment_ids, vec!["gesture-1"]);
        assert!(ended.stop_required_segment_ids.is_empty());
    }

    #[test]
    fn cancellation_stops_every_active_channel_and_blocks_future_dispatch() {
        let mut value = timeline();
        value.segments.push(segment(
            "late-gesture",
            ActionChannel::Gesture,
            ActionKind::Gesture,
            6_000,
            7_000,
            None,
        ));
        let compiled = compile_action_timeline_v1(value).unwrap();
        let mut gate = ActionExecutionGate::new(compiled.clone(), "clock:source-1").unwrap();
        let started = gate.advance(2_000, "clock:source-1", None).unwrap();
        assert_eq!(started.dispatch_segment_ids.len(), 3);
        let cancellation = ActionCancellationV1 {
            plan_id: compiled.plan_id.clone(),
            clock_domain_id: compiled.clock_domain_id.clone(),
            cancelled_at_monotonic_ns: 3_000,
            cause_event_id: "barge-in-1".to_owned(),
            reason: ActionCancellationReason::UserBargeIn,
            replacement_plan_id: None,
            timeline_digest: compiled.timeline_digest.clone(),
        };
        let stopped = gate.advance(3_000, "clock:source-1", Some(&cancellation)).unwrap();
        assert_eq!(stopped.stop_required_segment_ids, vec!["gesture-1", "speech-1", "viseme-1"]);
        assert!(stopped.cancelled);
        let after = gate.advance(6_000, "clock:source-1", None).unwrap();
        assert!(after.dispatch_segment_ids.is_empty());
        assert!(after.stop_required_segment_ids.is_empty());
    }

    #[test]
    fn cancellation_cannot_be_applied_before_its_source_timestamp() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let mut gate = ActionExecutionGate::new(compiled.clone(), "clock:source-1").unwrap();
        let cancellation = ActionCancellationV1 {
            plan_id: compiled.plan_id.clone(),
            clock_domain_id: compiled.clock_domain_id.clone(),
            cancelled_at_monotonic_ns: 3_000,
            cause_event_id: "barge-in-2".to_owned(),
            reason: ActionCancellationReason::UserBargeIn,
            replacement_plan_id: None,
            timeline_digest: compiled.timeline_digest.clone(),
        };
        assert_eq!(
            gate.advance(2_500, "clock:source-1", Some(&cancellation)),
            Err(ActionExecutionError::CancellationTimestampInFuture)
        );
    }

    #[test]
    fn expiry_requests_stop_for_active_segments_once() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let mut gate = ActionExecutionGate::new(compiled, "clock:source-1").unwrap();
        gate.advance(2_000, "clock:source-1", None).unwrap();
        let expired = gate.advance(10_000, "clock:source-1", None).unwrap();
        assert_eq!(expired.stop_required_segment_ids, vec!["gesture-1", "speech-1", "viseme-1"]);
        assert!(expired.expired);
        let repeated = gate.advance(10_001, "clock:source-1", None).unwrap();
        assert!(repeated.stop_required_segment_ids.is_empty());
        assert!(repeated.expired);
    }

    #[test]
    fn execution_gate_rejects_clock_regression_and_clock_mismatch() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let mut gate = ActionExecutionGate::new(compiled, "clock:source-1").unwrap();
        gate.advance(2_000, "clock:source-1", None).unwrap();
        assert_eq!(
            gate.advance(1_999, "clock:source-1", None),
            Err(ActionExecutionError::ExecutionClockRegression)
        );
        assert_eq!(
            gate.advance(2_001, "clock:other", None),
            Err(ActionExecutionError::WrongExecutionClockDomain)
        );
    }

    #[test]
    fn execution_gate_requires_timestamp_inside_or_after_source_event() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let mut gate = ActionExecutionGate::new(compiled, "clock:source-1").unwrap();
        assert_eq!(
            gate.advance(999, "clock:source-1", None),
            Err(ActionExecutionError::ExecutionBeforeTimeline)
        );
    }

    #[test]
    fn execution_gate_rejects_timeline_bound_to_another_clock() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        assert_eq!(
            ActionExecutionGate::new(compiled, "clock:other"),
            Err(ActionExecutionError::WrongExecutionClockDomain)
        );
    }


    #[test]
    fn scheduled_stop_is_repeated_until_adapter_accepts_it() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let mut gate = ActionExecutionGate::new(compiled, "clock:source-1").unwrap();
        gate.advance(2_000, "clock:source-1", None).unwrap();
        let first = gate.advance(4_000, "clock:source-1", None).unwrap();
        let second = gate.advance(4_001, "clock:source-1", None).unwrap();
        assert_eq!(first.scheduled_stop_segment_ids, vec!["gesture-1"]);
        assert_eq!(second.scheduled_stop_segment_ids, vec!["gesture-1"]);
        assert!(gate.acknowledge_stop_command_accepted("gesture-1"));
        let acknowledged = gate.advance(4_002, "clock:source-1", None).unwrap();
        assert!(acknowledged.scheduled_stop_segment_ids.is_empty());
    }

    #[test]
    fn cancellation_retries_stop_until_adapter_acknowledgement() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let mut gate = ActionExecutionGate::new(compiled.clone(), "clock:source-1").unwrap();
        gate.advance(2_000, "clock:source-1", None).unwrap();
        let cancellation = ActionCancellationV1 {
            plan_id: compiled.plan_id.clone(),
            clock_domain_id: compiled.clock_domain_id.clone(),
            cancelled_at_monotonic_ns: 3_000,
            cause_event_id: "stop-retry-1".to_owned(),
            reason: ActionCancellationReason::UserStop,
            replacement_plan_id: None,
            timeline_digest: compiled.timeline_digest.clone(),
        };
        let first = gate.advance(3_000, "clock:source-1", Some(&cancellation)).unwrap();
        assert_eq!(first.stop_required_segment_ids.len(), 3);
        let retry = gate.advance(3_001, "clock:source-1", None).unwrap();
        assert_eq!(retry.stop_required_segment_ids.len(), 3);
        assert!(gate.acknowledge_stop_command_accepted("speech-1"));
        let after_ack = gate.advance(3_002, "clock:source-1", None).unwrap();
        assert_eq!(after_ack.stop_required_segment_ids.len(), 2);
        assert!(!gate.acknowledge_stop_command_accepted("unknown"));
    }

    #[test]
    fn phase_order_is_an_explicit_part_of_the_contract() {
        assert!(ActionPhase::Prepare < ActionPhase::Onset);
        assert!(ActionPhase::Onset < ActionPhase::Sustain);
        assert!(ActionPhase::Sustain < ActionPhase::Release);
        assert!(ActionPhase::Release < ActionPhase::Recovery);
    }

    #[test]
    fn expiry_cancellation_is_valid_at_or_after_plan_expiry() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let cancellation = ActionCancellationV1 {
            plan_id: compiled.plan_id.clone(),
            clock_domain_id: compiled.clock_domain_id.clone(),
            cancelled_at_monotonic_ns: compiled.expires_at_monotonic_ns,
            cause_event_id: "expiry-1".to_owned(),
            reason: ActionCancellationReason::Expired,
            replacement_plan_id: None,
            timeline_digest: compiled.timeline_digest.clone(),
        };
        verify_action_cancellation_v1(&compiled, &cancellation).unwrap();
    }

    #[test]
    fn different_clock_domains_cannot_cancel_a_plan() {
        let compiled = compile_action_timeline_v1(timeline()).unwrap();
        let cancellation = ActionCancellationV1 {
            plan_id: compiled.plan_id.clone(),
            clock_domain_id: "clock:other".to_owned(),
            cancelled_at_monotonic_ns: 3_000,
            cause_event_id: "stop-2".to_owned(),
            reason: ActionCancellationReason::UserStop,
            replacement_plan_id: None,
            timeline_digest: compiled.timeline_digest.clone(),
        };
        assert_eq!(
            verify_action_cancellation_v1(&compiled, &cancellation),
            Err(ActionTimelineError::WrongClockDomain)
        );
    }
}
