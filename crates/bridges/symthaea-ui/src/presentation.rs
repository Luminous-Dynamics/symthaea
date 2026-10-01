// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Semantic presentation types for the Symthaea UI.
//!
//! This crate-local module deliberately contains only small, WASM-friendly
//! presentation semantics. It is not a mirror of the cognitive engine and
//! makes no claim that a display label is itself a scientific measurement.

/// Monotonic identity for one live telemetry connection.
///
/// A reconnect or gateway change creates a new generation. Callbacks retain
/// their generation and must not mutate current UI state after they become
/// stale. This is intentionally separate from event sequence numbers: event
/// sequence identifies presentation items, while this identifies the
/// authority allowed to produce them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TelemetrySessionId(u64);

/// Starts a new telemetry session without wrapping the generation counter.
/// Saturation is preferable to reusing an old identity after u64::MAX.
pub fn next_telemetry_session(current: &mut u64) -> Option<TelemetrySessionId> {
    let next = current.checked_add(1)?;
    *current = next;
    Some(TelemetrySessionId(next))
}

/// Allocates a globally unique event sequence without wrapping or reusing IDs.
/// `None` means the identity space is exhausted and callers must suppress new
/// events rather than publish a colliding identity.
pub fn next_event_sequence(current: &mut u64) -> Option<u64> {
    let next = current.checked_add(1)?;
    *current = next;
    Some(next)
}

/// Returns whether a callback still belongs to the currently authoritative
/// telemetry session.
pub const fn telemetry_session_is_current(
    current: u64,
    session: TelemetrySessionId,
) -> bool {
    current == session.0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresenceState {
    Disconnected,
    Available,
    Processing,
    Recovering,
    Degraded,
}

impl PresenceState {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Disconnected => "disconnected",
            Self::Available => "available",
            Self::Processing => "processing",
            Self::Recovering => "recovering",
            Self::Degraded => "degraded",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitiveMode {
    Resting,
    Exploring,
    Integrating,
    Responding,
    Uncertain,
}

impl CognitiveMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Resting => "resting",
            Self::Exploring => "exploring",
            Self::Integrating => "integrating",
            Self::Responding => "responding",
            Self::Uncertain => "uncertain",
        }
    }
}

/// Human-facing semantic state derived only from observable telemetry.
///
/// Thresholds here are intentionally conservative and descriptive. They are
/// presentation heuristics, not a new cognitive measurement or ontology.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CognitiveState {
    pub presence: PresenceState,
    pub mode: CognitiveMode,
    pub coherence: f64,
    pub thermodynamic_load: f64,
    pub confidence: f64,
    pub prediction_error: f64,
}

impl CognitiveState {
    fn finite_unit(value: f64) -> f64 {
        if value.is_finite() {
            value.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub fn from_observation(
        connected: bool,
        processing: bool,
        coherence: f64,
        thermodynamic_load: f64,
        confidence: f64,
        prediction_error: f64,
    ) -> Self {
        let telemetry_valid = coherence.is_finite()
            && thermodynamic_load.is_finite()
            && confidence.is_finite()
            && prediction_error.is_finite();
        let presence = if !connected {
            PresenceState::Disconnected
        } else if !telemetry_valid {
            // Invalid measurements remain visible as degraded even during a
            // request; processing must not mask broken telemetry.
            PresenceState::Degraded
        } else if processing {
            PresenceState::Processing
        } else {
            PresenceState::Available
        };

        let mode = if matches!(
            presence,
            PresenceState::Disconnected | PresenceState::Degraded
        ) {
            CognitiveMode::Uncertain
        } else if processing {
            CognitiveMode::Responding
        } else if confidence < 0.5 {
            CognitiveMode::Uncertain
        } else if prediction_error > 0.5 {
            CognitiveMode::Exploring
        } else if thermodynamic_load < 0.12 {
            CognitiveMode::Resting
        } else if coherence > 0.70 && thermodynamic_load > 0.45 {
            CognitiveMode::Integrating
        } else {
            CognitiveMode::Uncertain
        };

        Self {
            presence,
            mode,
            coherence: Self::finite_unit(coherence),
            thermodynamic_load: Self::finite_unit(thermodynamic_load),
            confidence: Self::finite_unit(confidence),
            prediction_error: Self::finite_unit(prediction_error),
        }
    }
}

/// Semantic events are deliberately coarser than the 10 Hz telemetry stream.
/// They describe observable transitions or explicit engine signals, not
/// hidden reasoning or chain-of-thought.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitiveEventKind {
    Connected,
    Disconnected,
    ProcessingStarted,
    ProcessingCompleted,
    SurpriseDetected,
    WorkspaceBroadcast,
    EnteredRest,
    ExitedRest,
    StateChanged,
}

impl CognitiveEventKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Connected => "connected",
            Self::Disconnected => "disconnected",
            Self::ProcessingStarted => "processing started",
            Self::ProcessingCompleted => "processing completed",
            Self::SurpriseDetected => "surprise detected",
            Self::WorkspaceBroadcast => "workspace broadcast",
            Self::EnteredRest => "entered rest",
            Self::ExitedRest => "exited rest",
            Self::StateChanged => "state changed",
        }
    }
}

/// Observable basis for presenting an event. This is provenance for the UI
/// heuristic/signal, not a claim about hidden cognitive causality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventEvidenceBasis {
    Lifecycle,
    PresenceTransition,
    ModeTransition,
    StateTransition,
    ExplicitSurpriseSignal,
    ExplicitWorkspaceSignal,
}

impl EventEvidenceBasis {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Lifecycle => "lifecycle marker",
            Self::PresenceTransition => "presence transition",
            Self::ModeTransition => "mode transition",
            Self::StateTransition => "state transition",
            Self::ExplicitSurpriseSignal => "explicit surprise signal",
            Self::ExplicitWorkspaceSignal => "explicit workspace signal",
        }
    }

    const fn for_kind(kind: CognitiveEventKind) -> Self {
        match kind {
            CognitiveEventKind::Connected | CognitiveEventKind::Disconnected => Self::Lifecycle,
            CognitiveEventKind::ProcessingStarted | CognitiveEventKind::ProcessingCompleted => {
                Self::PresenceTransition
            }
            CognitiveEventKind::EnteredRest | CognitiveEventKind::ExitedRest => Self::ModeTransition,
            CognitiveEventKind::StateChanged => Self::StateTransition,
            CognitiveEventKind::SurpriseDetected => Self::ExplicitSurpriseSignal,
            CognitiveEventKind::WorkspaceBroadcast => Self::ExplicitWorkspaceSignal,
        }
    }
}

/// A timeline event carries the measurements that justified its presentation.
/// It never contains private chain-of-thought.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CognitiveEvent {
    /// Monotonic presentation identity, independent of daemon cycle numbers.
    /// This keeps reconnects and repeated cycle values from colliding in the UI.
    pub sequence: u64,
    pub kind: CognitiveEventKind,
    pub evidence_basis: EventEvidenceBasis,
    /// Daemon-local sample label. It is descriptive metadata only: it must
    /// never determine temporal ordering, identity, or span reconstruction.
    /// The stable presentation sequence above is authoritative for ordering.
    pub cycle: u64,
    pub coherence: f64,
    pub thermodynamic_load: f64,
    pub prediction_error: f64,
}

impl CognitiveEvent {
    pub fn lifecycle(sequence: u64, kind: CognitiveEventKind, cycle: u64) -> Self {
        Self {
            sequence,
            kind,
            evidence_basis: EventEvidenceBasis::for_kind(kind),
            cycle,
            coherence: 0.0,
            thermodynamic_load: 0.0,
            prediction_error: 0.0,
        }
    }

    pub fn from_state(
        sequence: u64,
        kind: CognitiveEventKind,
        cycle: u64,
        state: CognitiveState,
    ) -> Self {
        Self::from_state_with_basis(
            sequence,
            kind,
            cycle,
            state,
            EventEvidenceBasis::for_kind(kind),
        )
    }

    fn from_state_with_basis(
        sequence: u64,
        kind: CognitiveEventKind,
        cycle: u64,
        state: CognitiveState,
        evidence_basis: EventEvidenceBasis,
    ) -> Self {
        Self {
            sequence,
            kind,
            evidence_basis,
            cycle,
            coherence: state.coherence,
            thermodynamic_load: state.thermodynamic_load,
            prediction_error: state.prediction_error,
        }
    }
}

/// Returns at most one high-signal event for a telemetry transition.
///
/// Priority is intentional: state transitions are more meaningful than a
/// simultaneous broadcast/surprise flag, preventing a noisy timeline.
pub fn event_between(
    previous: Option<CognitiveState>,
    current: CognitiveState,
    sequence: u64,
    cycle: u64,
    surprise: bool,
    gwt: bool,
) -> Option<CognitiveEvent> {
    let previous = previous?;

    let (kind, evidence_basis) = if previous.presence != current.presence {
        let kind = match (previous.presence, current.presence) {
            (_, PresenceState::Processing) => CognitiveEventKind::ProcessingStarted,
            (PresenceState::Processing, PresenceState::Available) => {
                CognitiveEventKind::ProcessingCompleted
            }
            (PresenceState::Disconnected, _) => CognitiveEventKind::Connected,
            (_, PresenceState::Disconnected) => CognitiveEventKind::Disconnected,
            _ => CognitiveEventKind::StateChanged,
        };
        (kind, EventEvidenceBasis::PresenceTransition)
    } else if previous.mode != current.mode {
        let kind = match (previous.mode, current.mode) {
            (CognitiveMode::Resting, _) => CognitiveEventKind::ExitedRest,
            (_, CognitiveMode::Resting) => CognitiveEventKind::EnteredRest,
            _ => CognitiveEventKind::StateChanged,
        };
        (kind, EventEvidenceBasis::ModeTransition)
    } else if surprise {
        (
            CognitiveEventKind::SurpriseDetected,
            EventEvidenceBasis::ExplicitSurpriseSignal,
        )
    } else if gwt {
        (
            CognitiveEventKind::WorkspaceBroadcast,
            EventEvidenceBasis::ExplicitWorkspaceSignal,
        )
    } else {
        return None;
    };

    Some(CognitiveEvent::from_state_with_basis(
        sequence,
        kind,
        cycle,
        current,
        evidence_basis,
    ))
}


/// A bounded temporal span reconstructed from paired semantic events.
///
/// The UI may show these as cycle spans, but must not turn them into elapsed
/// seconds: the presentation layer has daemon cycle identifiers, not an
/// authoritative wall clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitiveSpanKind {
    Processing,
    Resting,
}

impl CognitiveSpanKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Processing => "processing",
            Self::Resting => "resting",
        }
    }
}

/// A semantic interval with explicit entry/exit identities.
///
/// `end_*` is `None` while the interval remains open in the observed event
/// window. An open interval is not treated as proof that the underlying state
/// continues beyond the retained event history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CognitiveSpan {
    pub kind: CognitiveSpanKind,
    pub start_sequence: u64,
    pub start_cycle: u64,
    pub end_sequence: Option<u64>,
    pub end_cycle: Option<u64>,
}

impl CognitiveSpan {
    pub const fn is_open(self) -> bool {
        self.end_sequence.is_none()
    }
}

/// Appends one event to the presentation window and enforces its bounded size.
///
/// This is intentionally the single mutation primitive for the timeline: callers
/// cannot accidentally forget the retention limit when adding lifecycle or
/// telemetry-derived events.
pub const COGNITIVE_EVENT_WINDOW: usize = 32;

pub fn push_cognitive_event(events: &mut Vec<CognitiveEvent>, event: CognitiveEvent) {
    events.push(event);
    if events.len() > COGNITIVE_EVENT_WINDOW {
        let overflow = events.len() - COGNITIVE_EVENT_WINDOW;
        events.drain(..overflow);
    }
}

/// Reconstructs only spans whose entry and exit are explicitly represented by
/// semantic events. Unmatched exits are ignored rather than guessed.
pub fn cognitive_spans(events: &[CognitiveEvent]) -> Vec<CognitiveSpan> {
    let mut spans = Vec::new();
    let mut open_processing: Option<CognitiveSpan> = None;
    let mut open_resting: Option<CognitiveSpan> = None;

    let mut ordered = events.to_vec();
    ordered.sort_by_key(|event| event.sequence);

    for event in &ordered {
        match event.kind {
            CognitiveEventKind::Disconnected => {
                open_processing = None;
                open_resting = None;
            }
            CognitiveEventKind::ProcessingStarted => {
                if open_processing.is_none() {
                    open_processing = Some(CognitiveSpan {
                        kind: CognitiveSpanKind::Processing,
                        start_sequence: event.sequence,
                        start_cycle: event.cycle,
                        end_sequence: None,
                        end_cycle: None,
                    });
                }
            }
            CognitiveEventKind::ProcessingCompleted => {
                if let Some(mut span) = open_processing.take() {
                    span.end_sequence = Some(event.sequence);
                    span.end_cycle = Some(event.cycle);
                    spans.push(span);
                }
            }
            CognitiveEventKind::EnteredRest => {
                if open_resting.is_none() {
                    open_resting = Some(CognitiveSpan {
                        kind: CognitiveSpanKind::Resting,
                        start_sequence: event.sequence,
                        start_cycle: event.cycle,
                        end_sequence: None,
                        end_cycle: None,
                    });
                }
            }
            CognitiveEventKind::ExitedRest => {
                if let Some(mut span) = open_resting.take() {
                    span.end_sequence = Some(event.sequence);
                    span.end_cycle = Some(event.cycle);
                    spans.push(span);
                }
            }
            _ => {}
        }
    }

    if let Some(span) = open_processing {
        spans.push(span);
    }
    if let Some(span) = open_resting {
        spans.push(span);
    }

    spans.sort_by_key(|span| span.start_sequence);
    spans
}

#[cfg(test)]
mod telemetry_session_tests {
    use super::*;

    #[test]
    fn new_session_invalidates_previous_generation() {
        let mut generation = 0;
        let first = next_telemetry_session(&mut generation).unwrap();
        let second = next_telemetry_session(&mut generation).unwrap();
        assert_ne!(first, second);
        assert!(!telemetry_session_is_current(generation, first));
        assert!(telemetry_session_is_current(generation, second));
    }

    #[test]
    fn generation_exhaustion_fails_closed_without_reusing_identity() {
        let mut generation = u64::MAX - 1;
        let final_session = next_telemetry_session(&mut generation).unwrap();
        assert_eq!(generation, u64::MAX);
        assert!(telemetry_session_is_current(generation, final_session));
        assert!(next_telemetry_session(&mut generation).is_none());
        assert_eq!(generation, u64::MAX);
        assert!(telemetry_session_is_current(generation, final_session));
    }

    #[test]
    fn stale_callbacks_are_rejected() {
        let mut generation = 0;
        let stale = next_telemetry_session(&mut generation).unwrap();
        let _current = next_telemetry_session(&mut generation).unwrap();
        assert!(!telemetry_session_is_current(generation, stale));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(
        connected: bool,
        processing: bool,
        coherence: f64,
        load: f64,
        prediction_error: f64,
    ) -> CognitiveState {
        CognitiveState::from_observation(
            connected,
            processing,
            coherence,
            load,
            0.8,
            prediction_error,
        )
    }

    #[test]
    fn disconnected_state_is_explicit() {
        let value = state(false, false, 0.8, 0.2, 0.1);
        assert_eq!(value.presence, PresenceState::Disconnected);
        assert_eq!(value.mode, CognitiveMode::Uncertain);
    }

    #[test]
    fn low_confidence_does_not_invent_exploration() {
        let value = CognitiveState::from_observation(true, false, 0.6, 0.3, 0.4, 0.2);
        assert_eq!(value.presence, PresenceState::Available);
        assert_eq!(value.mode, CognitiveMode::Uncertain);
    }

    #[test]
    fn stable_low_load_maps_to_resting() {
        let value = state(true, false, 0.8, 0.1, 0.0);
        assert_eq!(value.mode, CognitiveMode::Resting);
    }

    #[test]
    fn high_prediction_error_maps_to_exploration() {
        let value = state(true, false, 0.6, 0.3, 0.6);
        assert_eq!(value.mode, CognitiveMode::Exploring);
    }

    #[test]
    fn low_confidence_suppresses_strong_mode_inference() {
        let value = CognitiveState::from_observation(true, false, 0.8, 0.5, 0.4, 0.9);
        assert_eq!(value.mode, CognitiveMode::Uncertain);
    }

    #[test]
    fn values_are_clamped_for_display() {
        let value = CognitiveState::from_observation(true, false, 2.0, 4.0, 3.0, 2.0);
        assert_eq!(value.coherence, 1.0);
        assert_eq!(value.thermodynamic_load, 1.0);
        assert_eq!(value.confidence, 1.0);
        assert_eq!(value.prediction_error, 1.0);
    }

    #[test]
    fn non_finite_values_are_safely_normalized() {
        let value = CognitiveState::from_observation(
            true,
            false,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        );
        assert_eq!(value.presence, PresenceState::Degraded);
        assert_eq!(value.coherence, 0.0);
        assert_eq!(value.thermodynamic_load, 0.0);
        assert_eq!(value.confidence, 0.0);
        assert_eq!(value.prediction_error, 0.0);
        assert!(value.coherence.is_finite());
        assert!(value.thermodynamic_load.is_finite());
        assert!(value.confidence.is_finite());
        assert!(value.prediction_error.is_finite());
    }

    #[test]
    fn processing_transition_has_priority_over_broadcast() {
        let previous = state(true, false, 0.8, 0.3, 0.1);
        let current = state(true, true, 0.8, 0.3, 0.1);
        let event = event_between(Some(previous), current, 1, 42, true, true).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::ProcessingStarted);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::PresenceTransition);
    }

    #[test]
    fn rest_transition_is_explicit() {
        let previous = state(true, false, 0.8, 0.3, 0.1);
        let current = state(true, false, 0.8, 0.1, 0.1);
        let event = event_between(Some(previous), current, 2, 42, false, false).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::EnteredRest);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::ModeTransition);
    }

    #[test]
    fn explicit_surprise_creates_event_without_state_change() {
        let previous = state(true, false, 0.8, 0.3, 0.1);
        let current = state(true, false, 0.8, 0.3, 0.1);
        let event = event_between(Some(previous), current, 3, 42, true, false).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::SurpriseDetected);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::ExplicitSurpriseSignal);
    }


    #[test]
    fn processing_to_degraded_is_not_called_completion() {
        let previous = state(true, true, 0.8, 0.5, 0.1);
        let current = CognitiveState::from_observation(true, false, 0.8, 0.5, 0.8, 0.1);
        assert_eq!(current.presence, PresenceState::Available);
        let degraded = CognitiveState {
            presence: PresenceState::Degraded,
            ..current
        };
        let event = event_between(Some(previous), degraded, 4, 43, false, false).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::StateChanged);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::PresenceTransition);
    }

    #[test]
    fn processing_to_recovering_is_not_called_completion() {
        let previous = state(true, true, 0.8, 0.5, 0.1);
        let recovering = CognitiveState {
            presence: PresenceState::Recovering,
            mode: CognitiveMode::Uncertain,
            ..state(true, false, 0.8, 0.5, 0.1)
        };
        let event = event_between(Some(previous), recovering, 5, 44, false, false).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::StateChanged);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::PresenceTransition);
    }

    #[test]
    fn responding_mode_transition_is_never_processing_lifecycle() {
        let previous = CognitiveState {
            presence: PresenceState::Available,
            mode: CognitiveMode::Exploring,
            ..state(true, false, 0.8, 0.3, 0.6)
        };
        let current = CognitiveState {
            presence: PresenceState::Available,
            mode: CognitiveMode::Responding,
            ..state(true, false, 0.8, 0.3, 0.1)
        };
        let event = event_between(Some(previous), current, 6, 45, false, false).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::StateChanged);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::ModeTransition);
    }

    #[test]
    fn processing_to_available_is_the_only_implicit_completion() {
        let previous = state(true, true, 0.8, 0.5, 0.1);
        let current = state(true, false, 0.8, 0.3, 0.1);
        let event = event_between(Some(previous), current, 6, 45, false, false).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::ProcessingCompleted);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::PresenceTransition);
    }

    #[test]
    fn processing_completion_has_presence_transition_basis() {
        let previous = state(true, true, 0.8, 0.5, 0.1);
        let current = state(true, false, 0.8, 0.3, 0.1);
        let event = event_between(Some(previous), current, 4, 43, false, false).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::ProcessingCompleted);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::PresenceTransition);
    }

    #[test]
    fn processing_start_has_presence_transition_basis() {
        let previous = state(true, false, 0.8, 0.3, 0.1);
        let current = state(true, true, 0.8, 0.5, 0.1);
        let event = event_between(Some(previous), current, 5, 44, false, false).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::ProcessingStarted);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::PresenceTransition);
    }

    #[test]
    fn resting_exit_is_not_mislabeled_as_processing() {
        let previous = state(true, false, 0.8, 0.1, 0.1);
        let current = state(true, false, 0.8, 0.3, 0.1);
        let event = event_between(Some(previous), current, 6, 45, false, false).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::ExitedRest);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::ModeTransition);
    }

    #[test]
    fn workspace_broadcast_is_used_when_no_stronger_transition_exists() {
        let previous = state(true, false, 0.8, 0.3, 0.1);
        let current = state(true, false, 0.8, 0.3, 0.1);
        let event = event_between(Some(previous), current, 7, 46, false, true).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::WorkspaceBroadcast);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::ExplicitWorkspaceSignal);
    }

    #[test]
    fn state_change_is_reserved_for_non_specific_mode_transitions() {
        let previous = CognitiveState::from_observation(true, false, 0.8, 0.3, 0.8, 0.1);
        let current = CognitiveState::from_observation(true, false, 0.8, 0.3, 0.4, 0.1);
        let event = event_between(Some(previous), current, 8, 47, false, false).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::StateChanged);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::ModeTransition);
    }

    #[test]
    fn processing_span_uses_cycles_not_wall_clock() {
        let events = vec![
            CognitiveEvent::lifecycle(1, CognitiveEventKind::ProcessingStarted, 40),
            CognitiveEvent::lifecycle(2, CognitiveEventKind::ProcessingCompleted, 44),
        ];
        let spans = cognitive_spans(&events);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].kind, CognitiveSpanKind::Processing);
        assert_eq!(spans[0].start_cycle, 40);
        assert_eq!(spans[0].end_cycle, Some(44));
        assert_eq!(spans[0].start_sequence, 1);
        assert_eq!(spans[0].end_sequence, Some(2));
        assert!(!spans[0].is_open());
    }

    #[test]
    fn disconnect_breaks_span_continuity() {
        let events = vec![
            CognitiveEvent::lifecycle(4, CognitiveEventKind::ProcessingStarted, 40),
            CognitiveEvent::lifecycle(5, CognitiveEventKind::Disconnected, 41),
            CognitiveEvent::lifecycle(6, CognitiveEventKind::Connected, 0),
            CognitiveEvent::lifecycle(7, CognitiveEventKind::ProcessingCompleted, 44),
        ];
        assert!(cognitive_spans(&events).is_empty());
    }

    #[test]
    fn span_reconstruction_uses_sequence_order() {
        let events = vec![
            CognitiveEvent::lifecycle(8, CognitiveEventKind::ProcessingCompleted, 44),
            CognitiveEvent::lifecycle(7, CognitiveEventKind::ProcessingStarted, 40),
        ];
        let spans = cognitive_spans(&events);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start_sequence, 7);
        assert_eq!(spans[0].end_sequence, Some(8));
    }

    #[test]
    fn unmatched_exit_does_not_fabricate_a_span() {
        let events = vec![
            CognitiveEvent::lifecycle(1, CognitiveEventKind::ProcessingCompleted, 44),
        ];
        assert!(cognitive_spans(&events).is_empty());
    }

    #[test]
    fn duplicate_start_does_not_replace_original_span() {
        let events = vec![
            CognitiveEvent::lifecycle(1, CognitiveEventKind::ProcessingStarted, 40),
            CognitiveEvent::lifecycle(2, CognitiveEventKind::ProcessingStarted, 41),
            CognitiveEvent::lifecycle(3, CognitiveEventKind::ProcessingCompleted, 44),
        ];
        let spans = cognitive_spans(&events);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start_sequence, 1);
        assert_eq!(spans[0].start_cycle, 40);
        assert_eq!(spans[0].end_sequence, Some(3));
    }

    #[test]
    fn duplicate_completion_does_not_fabricate_a_second_span() {
        let events = vec![
            CognitiveEvent::lifecycle(1, CognitiveEventKind::ProcessingStarted, 40),
            CognitiveEvent::lifecycle(2, CognitiveEventKind::ProcessingCompleted, 44),
            CognitiveEvent::lifecycle(3, CognitiveEventKind::ProcessingCompleted, 45),
        ];
        let spans = cognitive_spans(&events);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].end_sequence, Some(2));
    }

    #[test]
    fn retained_window_with_evicted_start_does_not_fabricate_span() {
        let events = vec![
            CognitiveEvent::lifecycle(32, CognitiveEventKind::StateChanged, 70),
            CognitiveEvent::lifecycle(33, CognitiveEventKind::ProcessingCompleted, 71),
        ];
        assert!(cognitive_spans(&events).is_empty());
    }

    #[test]
    fn retained_window_with_start_but_no_exit_keeps_span_open() {
        let events = vec![
            CognitiveEvent::lifecycle(32, CognitiveEventKind::ProcessingStarted, 70),
            CognitiveEvent::lifecycle(33, CognitiveEventKind::StateChanged, 71),
        ];
        let spans = cognitive_spans(&events);
        assert_eq!(spans.len(), 1);
        assert!(spans[0].is_open());
        assert_eq!(spans[0].start_sequence, 32);
    }

    #[test]
    fn reconnect_with_repeated_cycle_numbers_stays_disconnected() {
        let events = vec![
            CognitiveEvent::lifecycle(1, CognitiveEventKind::ProcessingStarted, 40),
            CognitiveEvent::lifecycle(2, CognitiveEventKind::Disconnected, 40),
            CognitiveEvent::lifecycle(3, CognitiveEventKind::Connected, 0),
            CognitiveEvent::lifecycle(4, CognitiveEventKind::ProcessingCompleted, 40),
        ];
        assert!(cognitive_spans(&events).is_empty());
    }

    #[test]
    fn open_span_remains_explicitly_open() {
        let events = vec![
            CognitiveEvent::lifecycle(3, CognitiveEventKind::EnteredRest, 90),
        ];
        let spans = cognitive_spans(&events);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].kind, CognitiveSpanKind::Resting);
        assert_eq!(spans[0].start_cycle, 90);
        assert_eq!(spans[0].end_cycle, None);
        assert!(spans[0].is_open());
    }

    #[test]
    fn span_order_follows_entry_sequence() {
        let events = vec![
            CognitiveEvent::lifecycle(5, CognitiveEventKind::EnteredRest, 50),
            CognitiveEvent::lifecycle(6, CognitiveEventKind::ExitedRest, 52),
            CognitiveEvent::lifecycle(7, CognitiveEventKind::ProcessingStarted, 53),
            CognitiveEvent::lifecycle(8, CognitiveEventKind::ProcessingCompleted, 57),
        ];
        let spans = cognitive_spans(&events);
        assert_eq!(
            spans.iter().map(|span| span.kind).collect::<Vec<_>>(),
            vec![CognitiveSpanKind::Resting, CognitiveSpanKind::Processing]
        );
    }

    #[test]
    fn event_window_is_exactly_bounded() {
        let mut events = Vec::new();
        for sequence in 1..=COGNITIVE_EVENT_WINDOW as u64 + 5 {
            push_cognitive_event(
                &mut events,
                CognitiveEvent::lifecycle(sequence, CognitiveEventKind::StateChanged, sequence),
            );
        }
        assert_eq!(events.len(), COGNITIVE_EVENT_WINDOW);
        assert_eq!(events.first().unwrap().sequence, 6);
        assert_eq!(events.last().unwrap().sequence, 37);
    }

    #[test]
    fn event_window_preserves_order_after_multiple_evictions() {
        let mut events = Vec::new();
        for sequence in 1..=64 {
            push_cognitive_event(
                &mut events,
                CognitiveEvent::lifecycle(sequence, CognitiveEventKind::StateChanged, sequence),
            );
        }
        assert!(events.windows(2).all(|pair| pair[0].sequence < pair[1].sequence));
        assert_eq!(events.first().unwrap().sequence, 33);
        assert_eq!(events.last().unwrap().sequence, 64);
    }

    #[test]
    #[test]
    fn disconnected_to_degraded_is_connection_recovery_not_cognitive_processing() {
        let disconnected = CognitiveState {
            presence: PresenceState::Disconnected,
            mode: CognitiveMode::Uncertain,
            ..state(false, false, 0.0, 0.0, 0.1)
        };
        let degraded = CognitiveState {
            presence: PresenceState::Degraded,
            mode: CognitiveMode::Uncertain,
            ..state(true, false, 0.8, 0.3, 0.1)
        };

        let event = event_between(Some(disconnected), degraded, 1, 7, true, true)
            .expect("connection recovery should be represented");

        assert_eq!(event.kind, CognitiveEventKind::Connected);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::PresenceTransition);
    }

    #[test]
    fn disconnected_to_processing_is_processing_start_with_explicit_observed_presence() {
        let disconnected = CognitiveState {
            presence: PresenceState::Disconnected,
            mode: CognitiveMode::Uncertain,
            ..state(false, false, 0.0, 0.0, 0.1)
        };
        let processing = state(true, true, 0.8, 0.5, 0.1);

        let event = event_between(Some(disconnected), processing, 2, 8, true, true)
            .expect("observed processing should be represented");

        assert_eq!(event.kind, CognitiveEventKind::ProcessingStarted);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::PresenceTransition);
    }

    #[test]
    fn degraded_to_processing_is_processing_start_when_presence_becomes_processing() {
        let degraded = CognitiveState {
            presence: PresenceState::Degraded,
            mode: CognitiveMode::Uncertain,
            ..state(true, false, 0.8, 0.3, 0.1)
        };
        let processing = state(true, true, 0.8, 0.5, 0.1);

        let event = event_between(Some(degraded), processing, 3, 9, false, false)
            .expect("observed processing should be represented");

        assert_eq!(event.kind, CognitiveEventKind::ProcessingStarted);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::PresenceTransition);
    }

    fn presence_transition_contract_is_conservative() {
        let available = state(true, false, 0.8, 0.3, 0.1);
        let processing = state(true, true, 0.8, 0.5, 0.1);
        let disconnected = state(false, false, 0.8, 0.3, 0.1);
        let degraded = CognitiveState {
            presence: PresenceState::Degraded,
            mode: CognitiveMode::Uncertain,
            ..available
        };
        let recovering = CognitiveState {
            presence: PresenceState::Recovering,
            mode: CognitiveMode::Uncertain,
            ..available
        };

        let cases = [
            (
                available,
                processing,
                CognitiveEventKind::ProcessingStarted,
                EventEvidenceBasis::PresenceTransition,
            ),
            (
                processing,
                available,
                CognitiveEventKind::ProcessingCompleted,
                EventEvidenceBasis::PresenceTransition,
            ),
            (
                processing,
                degraded,
                CognitiveEventKind::StateChanged,
                EventEvidenceBasis::PresenceTransition,
            ),
            (
                processing,
                recovering,
                CognitiveEventKind::StateChanged,
                EventEvidenceBasis::PresenceTransition,
            ),
            (
                processing,
                disconnected,
                CognitiveEventKind::Disconnected,
                EventEvidenceBasis::PresenceTransition,
            ),
            (
                disconnected,
                available,
                CognitiveEventKind::Connected,
                EventEvidenceBasis::PresenceTransition,
            ),
            (
                disconnected,
                processing,
                CognitiveEventKind::ProcessingStarted,
                EventEvidenceBasis::PresenceTransition,
            ),
            (
                disconnected,
                degraded,
                CognitiveEventKind::Connected,
                EventEvidenceBasis::PresenceTransition,
            ),
            (
                degraded,
                available,
                CognitiveEventKind::StateChanged,
                EventEvidenceBasis::PresenceTransition,
            ),
            (
                recovering,
                available,
                CognitiveEventKind::StateChanged,
                EventEvidenceBasis::PresenceTransition,
            ),
        ];

        for (index, (previous, current, expected_kind, expected_basis)) in cases.into_iter().enumerate() {
            let event = event_between(Some(previous), current, index as u64 + 1, 100 + index as u64, false, false)
                .expect("every presence transition should produce one event");
            assert_eq!(event.kind, expected_kind, "case {index}");
            assert_eq!(event.evidence_basis, expected_basis, "case {index}");
        }
    }

    #[test]
    fn mode_transition_contract_never_promotes_generic_modes_to_lifecycle() {
        let modes = [
            CognitiveMode::Resting,
            CognitiveMode::Exploring,
            CognitiveMode::Integrating,
            CognitiveMode::Responding,
            CognitiveMode::Uncertain,
        ];

        for previous_mode in modes {
            for current_mode in modes {
                if previous_mode == current_mode {
                    continue;
                }
                let previous = CognitiveState {
                    presence: PresenceState::Available,
                    mode: previous_mode,
                    ..state(true, false, 0.8, 0.3, 0.1)
                };
                let current = CognitiveState {
                    presence: PresenceState::Available,
                    mode: current_mode,
                    ..state(true, false, 0.8, 0.3, 0.1)
                };
                let event = event_between(Some(previous), current, 1, 100, false, false)
                    .expect("mode transitions should produce one event");

                assert_ne!(event.kind, CognitiveEventKind::ProcessingStarted);
                assert_ne!(event.kind, CognitiveEventKind::ProcessingCompleted);
                assert_eq!(event.evidence_basis, EventEvidenceBasis::ModeTransition);

                match (previous_mode, current_mode) {
                    (CognitiveMode::Resting, _) => {
                        assert_eq!(event.kind, CognitiveEventKind::ExitedRest)
                    }
                    (_, CognitiveMode::Resting) => {
                        assert_eq!(event.kind, CognitiveEventKind::EnteredRest)
                    }
                    _ => assert_eq!(event.kind, CognitiveEventKind::StateChanged),
                }
            }
        }
    }

    #[test]
    fn invalid_processing_observation_stays_degraded_and_uncertain() {
        let value = CognitiveState::from_observation(true, true, f64::NAN, 0.5, 0.8, 0.1);
        assert_eq!(value.presence, PresenceState::Degraded);
        assert_eq!(value.mode, CognitiveMode::Uncertain);
    }

    #[test]
    fn transition_priority_coalesces_explicit_signals_deterministically() {
        let previous = state(true, false, 0.8, 0.3, 0.1);
        let current = state(true, true, 0.8, 0.5, 0.1);
        let event = event_between(Some(previous), current, 1, 42, true, true).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::ProcessingStarted);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::PresenceTransition);
    }

    #[test]
    fn explicit_signal_priority_is_surprise_before_workspace_broadcast() {
        let state = state(true, false, 0.8, 0.3, 0.1);
        let event = event_between(Some(state), state, 1, 42, true, true).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::SurpriseDetected);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::ExplicitSurpriseSignal);
    }

    #[test]
    fn workspace_broadcast_is_reached_only_when_no_higher_signal_exists() {
        let state = state(true, false, 0.8, 0.3, 0.1);
        let event = event_between(Some(state), state, 1, 42, false, true).unwrap();
        assert_eq!(event.kind, CognitiveEventKind::WorkspaceBroadcast);
        assert_eq!(event.evidence_basis, EventEvidenceBasis::ExplicitWorkspaceSignal);
    }

    #[test]
    fn quiet_cycle_creates_no_event() {
        let previous = state(true, false, 0.8, 0.3, 0.1);
        let current = state(true, false, 0.8, 0.3, 0.1);
        assert!(event_between(Some(previous), current, 4, 42, false, false).is_none());
    }

    #[test]
    fn confidence_boundary_is_exactly_half() {
        let below = CognitiveState::from_observation(true, false, 0.8, 0.3, 0.499_999, 0.1);
        let exact = CognitiveState::from_observation(true, false, 0.8, 0.3, 0.5, 0.1);
        let above = CognitiveState::from_observation(true, false, 0.8, 0.3, 0.500_001, 0.1);

        assert_eq!(below.mode, CognitiveMode::Uncertain);
        assert_eq!(exact.mode, CognitiveMode::Resting);
        assert_eq!(above.mode, CognitiveMode::Resting);
    }

    #[test]
    fn prediction_error_boundary_is_strictly_greater_than_half() {
        let below = state(true, false, 0.8, 0.3, 0.499_999);
        let exact = state(true, false, 0.8, 0.3, 0.5);
        let above = state(true, false, 0.8, 0.3, 0.500_001);

        assert_eq!(below.mode, CognitiveMode::Resting);
        assert_eq!(exact.mode, CognitiveMode::Resting);
        assert_eq!(above.mode, CognitiveMode::Exploring);
    }

    #[test]
    fn resting_load_boundary_is_strictly_less_than_point_twelve() {
        let below = state(true, false, 0.8, 0.119_999, 0.1);
        let exact = state(true, false, 0.8, 0.12, 0.1);
        let above = state(true, false, 0.8, 0.120_001, 0.1);

        assert_eq!(below.mode, CognitiveMode::Resting);
        assert_ne!(exact.mode, CognitiveMode::Resting);
        assert_ne!(above.mode, CognitiveMode::Resting);
    }

    #[test]
    fn integrating_coherence_boundary_is_strictly_greater_than_point_seven() {
        let below = state(true, false, 0.699_999, 0.5, 0.1);
        let exact = state(true, false, 0.70, 0.5, 0.1);
        let above = state(true, false, 0.700_001, 0.5, 0.1);

        assert_ne!(below.mode, CognitiveMode::Integrating);
        assert_ne!(exact.mode, CognitiveMode::Integrating);
        assert_eq!(above.mode, CognitiveMode::Integrating);
    }

    #[test]
    fn integrating_load_boundary_is_strictly_greater_than_point_four_five() {
        let below = state(true, false, 0.8, 0.449_999, 0.1);
        let exact = state(true, false, 0.8, 0.45, 0.1);
        let above = state(true, false, 0.8, 0.450_001, 0.1);

        assert_ne!(below.mode, CognitiveMode::Integrating);
        assert_ne!(exact.mode, CognitiveMode::Integrating);
        assert_eq!(above.mode, CognitiveMode::Integrating);
    }

    #[test]
    fn disconnected_presence_overrides_measurements_and_processing() {
        let value = CognitiveState::from_observation(false, true, 1.0, 1.0, 1.0, 1.0);

        assert_eq!(value.presence, PresenceState::Disconnected);
        assert_eq!(value.mode, CognitiveMode::Uncertain);
    }

    #[test]
    fn processing_presence_overrides_mode_thresholds_when_valid() {
        let value = CognitiveState::from_observation(true, true, 1.0, 0.1, 1.0, 1.0);

        assert_eq!(value.presence, PresenceState::Processing);
        assert_eq!(value.mode, CognitiveMode::Responding);
    }

    #[test]
    fn invalid_telemetry_overrides_processing_presence() {
        let value = CognitiveState::from_observation(true, true, f64::NAN, 0.1, 1.0, 1.0);

        assert_eq!(value.presence, PresenceState::Degraded);
        assert_eq!(value.mode, CognitiveMode::Uncertain);
    }

    #[test]
    fn confidence_gate_precedes_prediction_error_and_load_thresholds() {
        let value = CognitiveState::from_observation(true, false, 1.0, 0.5, 0.499_999, 1.0);

        assert_eq!(value.presence, PresenceState::Available);
        assert_eq!(value.mode, CognitiveMode::Uncertain);
    }

    #[test]
    fn prediction_error_gate_precedes_lower_load_thresholds() {
        let value = state(true, false, 0.8, 0.1, 0.500_001);

        assert_eq!(value.mode, CognitiveMode::Exploring);
    }

    #[test]
    fn integrating_requires_both_strict_upper_boundaries() {
        let value = state(true, false, 0.700_001, 0.450_001, 0.1);

        assert_eq!(value.mode, CognitiveMode::Integrating);
    }

}


#[cfg(test)]
mod identity_tests {
    use super::*;

    #[test]
    fn event_evidence_basis_is_explicit_and_kind_aligned() {
        assert_eq!(
            CognitiveEvent::lifecycle(1, CognitiveEventKind::Connected, 0).evidence_basis,
            EventEvidenceBasis::Lifecycle
        );
        let state = CognitiveState::from_observation(true, false, 0.8, 0.3, 0.8, 0.1);
        let surprise =
            CognitiveEvent::from_state(2, CognitiveEventKind::SurpriseDetected, 9, state);
        assert_eq!(surprise.evidence_basis, EventEvidenceBasis::ExplicitSurpriseSignal);
        assert_eq!(surprise.evidence_basis.label(), "explicit surprise signal");
        let changed = CognitiveEvent::from_state(3, CognitiveEventKind::StateChanged, 10, state);
        assert_eq!(changed.evidence_basis, EventEvidenceBasis::StateTransition);
    }

    #[test]
    fn event_identity_is_independent_of_cycle() {
        let state = CognitiveState::from_observation(true, false, 0.8, 0.3, 0.8, 0.1);
        let first = CognitiveEvent::from_state(7, CognitiveEventKind::StateChanged, 12, state);
        let second = CognitiveEvent::from_state(8, CognitiveEventKind::StateChanged, 12, state);
        assert_ne!(first.sequence, second.sequence);
        assert_eq!(first.cycle, second.cycle);
    }

    #[test]
    fn lifecycle_event_retains_actual_cycle() {
        let event = CognitiveEvent::lifecycle(9, CognitiveEventKind::Disconnected, 1234);
        assert_eq!(event.sequence, 9);
        assert_eq!(event.cycle, 1234);
    }

    #[test]
    fn regressive_cycle_values_do_not_change_event_order() {
        let state = CognitiveState::from_observation(true, false, 0.8, 0.3, 0.8, 0.1);
        let earlier = CognitiveEvent::from_state(
            10,
            CognitiveEventKind::ProcessingStarted,
            100,
            state,
        );
        let later = CognitiveEvent::from_state(
            11,
            CognitiveEventKind::ProcessingCompleted,
            1,
            state,
        );

        assert!(earlier.sequence < later.sequence);
        assert!(earlier.cycle > later.cycle);
    }

    #[test]
    fn regressive_cycle_values_do_not_reverse_span_reconstruction() {
        let events = vec![
            CognitiveEvent::lifecycle(20, CognitiveEventKind::ProcessingStarted, 100),
            CognitiveEvent::lifecycle(21, CognitiveEventKind::ProcessingCompleted, 1),
        ];

        let spans = cognitive_spans(&events);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start_sequence, 20);
        assert_eq!(spans[0].end_sequence, Some(21));
        assert_eq!(spans[0].start_cycle, 100);
        assert_eq!(spans[0].end_cycle, Some(1));
    }
}
