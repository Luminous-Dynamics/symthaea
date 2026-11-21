// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned, deterministic event traces for embodiment replay and benchmarking.
//!
//! Traces make the semantic control plane testable without a renderer, model,
//! network connection, or live cognitive runtime. Replay is fail-fast: an
//! unsupported schema or invalid event rejects the whole trace.

use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::embodiment::{
    EmbodimentActivity, EmbodimentController, EmbodimentEvent, EmbodimentEventKind,
    EmbodimentEventSequencer, EmbodimentIngestError, EmbodimentSequenceError, EmbodimentSnapshot,
    SignalProvenance, SpeechOutputState,
};

/// Current serialized trace schema.
pub const EMBODIMENT_TRACE_SCHEMA_VERSION: u16 = 1;
/// Bound untrusted trace input before replay.
pub const MAX_EMBODIMENT_TRACE_EVENTS: usize = 65_536;
/// Bound externally supplied session identifiers.
pub const MAX_EMBODIMENT_SESSION_ID_BYTES: usize = 128;

/// Serializable, versioned event corpus for one embodiment session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbodimentTrace {
    pub schema_version: u16,
    pub session_id: String,
    pub events: Vec<EmbodimentEvent>,
}

impl EmbodimentTrace {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            schema_version: EMBODIMENT_TRACE_SCHEMA_VERSION,
            session_id: session_id.into(),
            events: Vec::new(),
        }
    }

    pub fn push(&mut self, event: EmbodimentEvent) -> Result<(), EmbodimentTraceError> {
        if self.events.len() >= MAX_EMBODIMENT_TRACE_EVENTS {
            return Err(EmbodimentTraceError::TooManyEvents {
                received: self.events.len().saturating_add(1),
                maximum: MAX_EMBODIMENT_TRACE_EVENTS,
            });
        }
        self.events.push(event);
        Ok(())
    }

    /// Reduce the complete trace into a final semantic snapshot.
    ///
    /// The method never returns a partial snapshot. A malformed event rejects
    /// the complete trace with its zero-based event index.
    pub fn replay(&self) -> Result<EmbodimentSnapshot, EmbodimentTraceError> {
        self.validate_header()?;
        if self.events.len() > MAX_EMBODIMENT_TRACE_EVENTS {
            return Err(EmbodimentTraceError::TooManyEvents {
                received: self.events.len(),
                maximum: MAX_EMBODIMENT_TRACE_EVENTS,
            });
        }

        let mut controller = EmbodimentController::new();
        for (index, event) in self.events.iter().cloned().enumerate() {
            controller
                .ingest(event)
                .map_err(|source| EmbodimentTraceError::EventRejected { index, source })?;
        }
        Ok(controller.snapshot().clone())
    }

    /// Compute deterministic, renderer-independent state changes and
    /// completed activity/playback intervals from a valid trace.
    ///
    /// The currently active interval is reported separately and is not counted
    /// as completed duration because its end boundary is not known.
    pub fn summarize(&self) -> Result<EmbodimentTraceSummary, EmbodimentTraceError> {
        let final_snapshot = self.replay()?;

        let mut activity_changes = 0usize;
        let mut completed_activity_intervals = 0usize;
        let mut activity_duration_ns = 0u128;
        let mut speech_output_transitions = 0usize;
        let mut completed_playback_intervals = 0usize;
        let mut playback_duration_ns = 0u128;
        let mut viseme_event_count = 0usize;
        let mut uncertainty_observation_count = 0usize;
        let mut previous_activity: Option<(EmbodimentActivity, u64)> = None;
        let mut previous_playback: Option<(SpeechOutputState, u64)> = None;

        for event in &self.events {
            let activity_update = match &event.kind {
                EmbodimentEventKind::ActivityChanged { activity } => Some(*activity),
                EmbodimentEventKind::Reset => Some(EmbodimentActivity::Idle),
                _ => None,
            };

            if let Some(activity) = activity_update {
                if let Some((previous, started_at)) = previous_activity {
                    if previous != activity {
                        activity_changes = activity_changes.saturating_add(1);
                        let duration = event.monotonic_ns.saturating_sub(started_at);
                        activity_duration_ns =
                            activity_duration_ns.saturating_add(u128::from(duration));
                        completed_activity_intervals =
                            completed_activity_intervals.saturating_add(1);
                        previous_activity = Some((activity, event.monotonic_ns));
                    }
                } else {
                    // The first observation establishes the initial state;
                    // it is not a transition from an unknown state.
                    previous_activity = Some((activity, event.monotonic_ns));
                }
            }

            let playback_update = match &event.kind {
                EmbodimentEventKind::SpeechOutputChanged { state } => Some(Some(*state)),
                EmbodimentEventKind::Reset => Some(None),
                _ => None,
            };

            if let Some(next_state) = playback_update {
                match (previous_playback, next_state) {
                    (Some((previous, started_at)), Some(next)) if previous != next => {
                        speech_output_transitions = speech_output_transitions.saturating_add(1);
                        if previous == SpeechOutputState::Playing {
                            completed_playback_intervals =
                                completed_playback_intervals.saturating_add(1);
                            let duration = event.monotonic_ns.saturating_sub(started_at);
                            playback_duration_ns =
                                playback_duration_ns.saturating_add(u128::from(duration));
                        }
                        previous_playback = Some((next, event.monotonic_ns));
                    }
                    (None, Some(next)) => {
                        // Initial observation is not counted as a transition.
                        previous_playback = Some((next, event.monotonic_ns));
                    }
                    (Some((previous, started_at)), None) => {
                        // Reset closes the lifecycle without fabricating another
                        // SpeechOutputState transition.
                        if previous == SpeechOutputState::Playing {
                            completed_playback_intervals =
                                completed_playback_intervals.saturating_add(1);
                            let duration = event.monotonic_ns.saturating_sub(started_at);
                            playback_duration_ns =
                                playback_duration_ns.saturating_add(u128::from(duration));
                        }
                        previous_playback = None;
                    }
                    (Some(_), Some(_)) | (None, None) => {}
                }
            }

            match &event.kind {
                EmbodimentEventKind::VisemeChanged { .. } => {
                    viseme_event_count = viseme_event_count.saturating_add(1);
                }
                EmbodimentEventKind::UncertaintyChanged { .. } => {
                    uncertainty_observation_count = uncertainty_observation_count.saturating_add(1);
                }
                _ => {}
            }
        }

        Ok(EmbodimentTraceSummary {
            session_id: self.session_id.clone(),
            event_count: self.events.len(),
            activity_changes,
            completed_activity_intervals,
            activity_duration_ns,
            speech_output_transitions,
            completed_playback_intervals,
            playback_duration_ns,
            viseme_event_count,
            uncertainty_observation_count,
            open_activity: previous_activity.map(|(activity, started_at_ns)| {
                OpenActivityInterval {
                    activity,
                    started_at_ns,
                }
            }),
            open_playback: previous_playback.and_then(|(state, started_at_ns)| {
                (state == SpeechOutputState::Playing)
                    .then_some(OpenPlaybackInterval { started_at_ns })
            }),
            final_snapshot,
        })
    }

    fn validate_header(&self) -> Result<(), EmbodimentTraceError> {
        if self.schema_version != EMBODIMENT_TRACE_SCHEMA_VERSION {
            return Err(EmbodimentTraceError::UnsupportedSchemaVersion {
                received: self.schema_version,
                supported: EMBODIMENT_TRACE_SCHEMA_VERSION,
            });
        }
        if self.session_id.trim().is_empty()
            || self.session_id.len() > MAX_EMBODIMENT_SESSION_ID_BYTES
            || self.session_id.chars().any(char::is_control)
        {
            return Err(EmbodimentTraceError::InvalidSessionId);
        }
        Ok(())
    }
}

/// End-to-end semantic embodiment session.
///
/// A single publish operation sequences, validates, reduces, and records an
/// event. Renderer code only reads the snapshot; runtime producers publish
/// through this session or a shared event sequencer.
pub struct EmbodimentSession {
    sequencer: EmbodimentEventSequencer,
    controller: EmbodimentController,
    trace: EmbodimentTrace,
}

impl EmbodimentSession {
    pub fn new(session_id: impl Into<String>) -> Result<Self, EmbodimentTraceError> {
        let trace = EmbodimentTrace::new(session_id);
        // Validate session metadata before accepting any runtime events.
        trace.replay()?;
        Ok(Self {
            sequencer: EmbodimentEventSequencer::new(),
            controller: EmbodimentController::new(),
            trace,
        })
    }

    pub fn snapshot(&self) -> &EmbodimentSnapshot {
        self.controller.snapshot()
    }

    pub fn trace(&self) -> &EmbodimentTrace {
        &self.trace
    }

    pub fn into_trace(self) -> EmbodimentTrace {
        self.trace
    }

    pub fn publish(
        &mut self,
        provenance: SignalProvenance,
        kind: EmbodimentEventKind,
    ) -> Result<EmbodimentEvent, EmbodimentSessionError> {
        self.publish_at(Instant::now(), provenance, kind)
    }

    pub fn publish_at(
        &mut self,
        observed_at: Instant,
        provenance: SignalProvenance,
        kind: EmbodimentEventKind,
    ) -> Result<EmbodimentEvent, EmbodimentSessionError> {
        if self.trace.events.len() >= MAX_EMBODIMENT_TRACE_EVENTS {
            return Err(EmbodimentTraceError::TooManyEvents {
                received: self.trace.events.len().saturating_add(1),
                maximum: MAX_EMBODIMENT_TRACE_EVENTS,
            }
            .into());
        }

        let event = self.sequencer.emit_at(observed_at, provenance, kind)?;
        self.controller.ingest(event.clone())?;
        self.trace.push(event.clone())?;
        Ok(event)
    }
}

/// Failure while publishing an event into an embodiment session.
///
/// A validation failure may consume a sequence number and timestamp in the
/// producer, but never mutates the authoritative snapshot or recorded trace.
/// Sequence gaps are explicitly permitted by the trace format.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbodimentSessionError {
    Sequence(EmbodimentSequenceError),
    Ingest(EmbodimentIngestError),
    Trace(EmbodimentTraceError),
}

impl fmt::Display for EmbodimentSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sequence(error) => write!(f, "event sequencing failed: {error}"),
            Self::Ingest(error) => write!(f, "event validation failed: {error}"),
            Self::Trace(error) => write!(f, "event trace append failed: {error}"),
        }
    }
}

impl std::error::Error for EmbodimentSessionError {}

impl From<EmbodimentSequenceError> for EmbodimentSessionError {
    fn from(value: EmbodimentSequenceError) -> Self {
        Self::Sequence(value)
    }
}

impl From<EmbodimentIngestError> for EmbodimentSessionError {
    fn from(value: EmbodimentIngestError) -> Self {
        Self::Ingest(value)
    }
}

impl From<EmbodimentTraceError> for EmbodimentSessionError {
    fn from(value: EmbodimentTraceError) -> Self {
        Self::Trace(value)
    }
}

/// An activity interval still open at the end of a trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenActivityInterval {
    pub activity: EmbodimentActivity,
    pub started_at_ns: u64,
}

/// An audio playback interval still open at the end of a trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenPlaybackInterval {
    pub started_at_ns: u64,
}

/// Stable summary suitable for benchmark output and trace comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbodimentTraceSummary {
    pub session_id: String,
    pub event_count: usize,
    pub activity_changes: usize,
    pub completed_activity_intervals: usize,
    pub activity_duration_ns: u128,
    pub speech_output_transitions: usize,
    pub completed_playback_intervals: usize,
    pub playback_duration_ns: u128,
    pub viseme_event_count: usize,
    pub uncertainty_observation_count: usize,
    pub open_activity: Option<OpenActivityInterval>,
    pub open_playback: Option<OpenPlaybackInterval>,
    pub final_snapshot: EmbodimentSnapshot,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EmbodimentTraceError {
    UnsupportedSchemaVersion {
        received: u16,
        supported: u16,
    },
    InvalidSessionId,
    TooManyEvents {
        received: usize,
        maximum: usize,
    },
    EventRejected {
        index: usize,
        source: EmbodimentIngestError,
    },
}

impl fmt::Display for EmbodimentTraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion {
                received,
                supported,
            } => write!(
                f,
                "unsupported embodiment trace schema {received}; supported version is {supported}"
            ),
            Self::InvalidSessionId => write!(f, "invalid embodiment trace session identifier"),
            Self::TooManyEvents { received, maximum } => {
                write!(f, "trace has {received} events; maximum is {maximum}")
            }
            Self::EventRejected { index, source } => {
                write!(f, "trace event at index {index} was rejected: {source}")
            }
        }
    }
}

impl std::error::Error for EmbodimentTraceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment::{SignalProvenance, Viseme};

    fn event(sequence: u64, monotonic_ns: u64, kind: EmbodimentEventKind) -> EmbodimentEvent {
        EmbodimentEvent {
            sequence,
            monotonic_ns,
            provenance: SignalProvenance::RuntimeEvent,
            kind,
        }
    }

    fn sample_trace() -> EmbodimentTrace {
        let mut trace = EmbodimentTrace::new("session-test-01");
        trace
            .push(event(
                1,
                50,
                EmbodimentEventKind::ActivityChanged {
                    activity: EmbodimentActivity::Listening,
                },
            ))
            .unwrap();
        trace
            .push(event(
                2,
                100,
                EmbodimentEventKind::SpeechOutputChanged {
                    state: SpeechOutputState::Synthesizing,
                },
            ))
            .unwrap();
        trace
            .push(event(
                3,
                110,
                EmbodimentEventKind::SpeechOutputChanged {
                    state: SpeechOutputState::Ready,
                },
            ))
            .unwrap();
        trace
            .push(event(
                4,
                120,
                EmbodimentEventKind::SpeechOutputChanged {
                    state: SpeechOutputState::Queued,
                },
            ))
            .unwrap();
        trace
            .push(event(
                5,
                130,
                EmbodimentEventKind::SpeechOutputChanged {
                    state: SpeechOutputState::Playing,
                },
            ))
            .unwrap();
        trace
            .push(event(
                6,
                150,
                EmbodimentEventKind::VisemeChanged {
                    viseme: Viseme::Aa,
                    weight: 0.8,
                },
            ))
            .unwrap();
        trace
            .push(event(
                7,
                200,
                EmbodimentEventKind::SpeechOutputChanged {
                    state: SpeechOutputState::Completed,
                },
            ))
            .unwrap();
        trace
            .push(event(
                8,
                210,
                EmbodimentEventKind::ActivityChanged {
                    activity: EmbodimentActivity::Idle,
                },
            ))
            .unwrap();
        trace
            .push(event(
                9,
                230,
                EmbodimentEventKind::ActivityChanged {
                    activity: EmbodimentActivity::Listening,
                },
            ))
            .unwrap();
        trace
    }

    #[test]
    fn session_publish_updates_snapshot_and_trace_together() {
        let mut session = EmbodimentSession::new("session-facade-test").unwrap();
        let started = Instant::now();
        let at1 = started + std::time::Duration::from_millis(1);
        let at2 = started + std::time::Duration::from_millis(2);

        let first = session
            .publish_at(
                at1,
                SignalProvenance::RuntimeEvent,
                EmbodimentEventKind::ActivityChanged {
                    activity: EmbodimentActivity::Associating,
                },
            )
            .unwrap();
        let second = session
            .publish_at(
                at2,
                SignalProvenance::RuntimeEvent,
                EmbodimentEventKind::SpeechOutputChanged {
                    state: SpeechOutputState::Playing,
                },
            )
            .unwrap();

        assert_eq!(first.sequence, 1);
        assert_eq!(second.sequence, 2);
        assert_eq!(
            session.snapshot().current_activity(),
            Some(EmbodimentActivity::Associating)
        );
        assert_eq!(
            session.snapshot().current_speech_output(),
            Some(SpeechOutputState::Playing)
        );
        assert_eq!(session.trace().events.len(), 2);
        assert_eq!(session.trace().replay().unwrap(), *session.snapshot());
    }

    #[test]
    fn rejected_session_event_is_not_recorded_and_next_sequence_can_have_a_gap() {
        let mut session = EmbodimentSession::new("session-rejected-event").unwrap();
        let started = Instant::now();

        let rejected = session.publish_at(
            started + std::time::Duration::from_millis(1),
            SignalProvenance::RuntimeDerived,
            EmbodimentEventKind::SpeechOutputChanged {
                state: SpeechOutputState::Playing,
            },
        );
        assert!(matches!(
            rejected,
            Err(EmbodimentSessionError::Ingest(
                EmbodimentIngestError::SignalProvenanceNotAllowed { .. }
            ))
        ));
        assert!(session.trace().events.is_empty());
        assert_eq!(session.snapshot().last_event_sequence, None);

        let accepted = session
            .publish_at(
                started + std::time::Duration::from_millis(2),
                SignalProvenance::RuntimeEvent,
                EmbodimentEventKind::ActivityChanged {
                    activity: EmbodimentActivity::Listening,
                },
            )
            .unwrap();
        assert_eq!(accepted.sequence, 2);
        assert_eq!(session.trace().events.len(), 1);
        assert_eq!(session.trace().events[0].sequence, 2);
        assert_eq!(session.trace().replay().unwrap(), *session.snapshot());
    }

    #[test]
    fn session_rejects_invalid_identifier_at_creation() {
        assert!(matches!(
            EmbodimentSession::new(" \n "),
            Err(EmbodimentTraceError::InvalidSessionId)
        ));
    }

    #[test]
    fn sefb_v1_fixture_replays_and_summarizes() {
        let encoded = include_str!("../tests/fixtures/embodiment_trace_v1.json");
        let trace: EmbodimentTrace = serde_json::from_str(encoded).unwrap();
        let summary = trace.summarize().unwrap();

        assert_eq!(summary.session_id, "sefb-fixture-v1");
        assert_eq!(summary.event_count, 11);
        assert_eq!(summary.activity_changes, 1);
        assert_eq!(summary.completed_activity_intervals, 1);
        assert_eq!(summary.activity_duration_ns, 2_600_000);
        assert_eq!(summary.speech_output_transitions, 4);
        assert_eq!(summary.completed_playback_intervals, 1);
        assert_eq!(summary.playback_duration_ns, 1_700_000);
        assert_eq!(summary.open_playback, None);
        assert_eq!(summary.viseme_event_count, 2);
        assert_eq!(summary.uncertainty_observation_count, 1);
        assert_eq!(
            summary.final_snapshot.current_activity(),
            Some(EmbodimentActivity::Idle)
        );
        assert!(summary.final_snapshot.viseme.is_none());
        assert!(summary.final_snapshot.uncertainty.is_none());
        assert_eq!(summary.final_snapshot.current_speech_output(), None);
    }

    #[test]
    fn replay_is_deterministic() {
        let trace = sample_trace();
        let first = trace.replay().unwrap();
        let second = trace.replay().unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.current_activity(),
            Some(EmbodimentActivity::Listening)
        );
        assert_eq!(
            first.current_speech_output(),
            Some(SpeechOutputState::Completed)
        );
    }

    #[test]
    fn summary_counts_activity_and_playback_intervals_separately() {
        let summary = sample_trace().summarize().unwrap();
        assert_eq!(summary.event_count, 9);
        assert_eq!(summary.activity_changes, 2);
        assert_eq!(summary.completed_activity_intervals, 2);
        assert_eq!(summary.activity_duration_ns, 180);
        assert_eq!(summary.speech_output_transitions, 4);
        assert_eq!(summary.completed_playback_intervals, 1);
        assert_eq!(summary.playback_duration_ns, 70);
        assert_eq!(summary.open_playback, None);
        assert_eq!(summary.viseme_event_count, 1);
        assert_eq!(
            summary.open_activity,
            Some(OpenActivityInterval {
                activity: EmbodimentActivity::Listening,
                started_at_ns: 230,
            })
        );
    }

    #[test]
    fn reset_closes_playback_interval_without_counting_a_fake_state_transition() {
        let mut trace = EmbodimentTrace::new("reset-closes-playback");
        trace
            .push(event(
                1,
                10,
                EmbodimentEventKind::SpeechOutputChanged {
                    state: SpeechOutputState::Playing,
                },
            ))
            .unwrap();
        trace
            .push(event(2, 110, EmbodimentEventKind::Reset))
            .unwrap();

        let summary = trace.summarize().unwrap();
        assert_eq!(summary.speech_output_transitions, 0);
        assert_eq!(summary.completed_playback_intervals, 1);
        assert_eq!(summary.playback_duration_ns, 100);
        assert_eq!(summary.open_playback, None);
    }

    #[test]
    fn trace_round_trips_through_json_and_replays_identically() {
        let trace = sample_trace();
        let encoded = serde_json::to_string(&trace).unwrap();
        let decoded: EmbodimentTrace = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, trace);
        assert_eq!(decoded.replay().unwrap(), trace.replay().unwrap());
    }

    #[test]
    fn unsupported_schema_fails_before_replay() {
        let mut trace = sample_trace();
        trace.schema_version = EMBODIMENT_TRACE_SCHEMA_VERSION + 1;
        assert_eq!(
            trace.replay(),
            Err(EmbodimentTraceError::UnsupportedSchemaVersion {
                received: EMBODIMENT_TRACE_SCHEMA_VERSION + 1,
                supported: EMBODIMENT_TRACE_SCHEMA_VERSION,
            })
        );
    }

    #[test]
    fn blank_session_identifier_is_rejected() {
        let mut trace = sample_trace();
        trace.session_id = "   ".to_string();
        assert_eq!(trace.replay(), Err(EmbodimentTraceError::InvalidSessionId));
    }

    #[test]
    fn malformed_event_fails_trace_at_exact_index() {
        let mut trace = sample_trace();
        trace.events.push(event(
            9,
            240,
            EmbodimentEventKind::ActivityChanged {
                activity: EmbodimentActivity::Fault,
            },
        ));
        assert!(matches!(
            trace.replay(),
            Err(EmbodimentTraceError::EventRejected { index: 9, .. })
        ));
    }

    #[test]
    fn trace_capacity_is_enforced_on_append() {
        let mut trace = EmbodimentTrace::new("bounded-session");
        for sequence in 0..MAX_EMBODIMENT_TRACE_EVENTS as u64 {
            trace
                .push(event(
                    sequence + 1,
                    sequence + 1,
                    EmbodimentEventKind::ActivityChanged {
                        activity: EmbodimentActivity::Idle,
                    },
                ))
                .unwrap();
        }
        let result = trace.push(event(
            MAX_EMBODIMENT_TRACE_EVENTS as u64 + 1,
            MAX_EMBODIMENT_TRACE_EVENTS as u64 + 1,
            EmbodimentEventKind::ActivityChanged {
                activity: EmbodimentActivity::Idle,
            },
        ));
        assert_eq!(
            result,
            Err(EmbodimentTraceError::TooManyEvents {
                received: MAX_EMBODIMENT_TRACE_EVENTS + 1,
                maximum: MAX_EMBODIMENT_TRACE_EVENTS,
            })
        );
    }
}
