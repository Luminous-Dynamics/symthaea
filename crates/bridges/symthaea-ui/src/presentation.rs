// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Semantic presentation types for the Symthaea UI.
//!
//! This crate-local module deliberately contains only small, WASM-friendly
//! presentation semantics. It is not a mirror of the cognitive engine and
//! makes no claim that a display label is itself a scientific measurement.

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
    ExplicitSurpriseSignal,
    ExplicitWorkspaceSignal,
}

impl EventEvidenceBasis {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Lifecycle => "lifecycle marker",
            Self::PresenceTransition => "presence transition",
            Self::ModeTransition => "mode transition",
            Self::ExplicitSurpriseSignal => "explicit surprise signal",
            Self::ExplicitWorkspaceSignal => "explicit workspace signal",
        }
    }

    const fn for_kind(kind: CognitiveEventKind) -> Self {
        match kind {
            CognitiveEventKind::Connected | CognitiveEventKind::Disconnected => Self::Lifecycle,
            CognitiveEventKind::ProcessingStarted | CognitiveEventKind::ProcessingCompleted => Self::PresenceTransition,
            CognitiveEventKind::EnteredRest | CognitiveEventKind::ExitedRest | CognitiveEventKind::StateChanged => Self::ModeTransition,
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
            (PresenceState::Processing, _) => CognitiveEventKind::ProcessingCompleted,
            (PresenceState::Disconnected, _) => CognitiveEventKind::Connected,
            (_, PresenceState::Disconnected) => CognitiveEventKind::Disconnected,
            _ => CognitiveEventKind::StateChanged,
        };
        (kind, EventEvidenceBasis::PresenceTransition)
    } else if previous.mode != current.mode {
        let kind = match (previous.mode, current.mode) {
            (CognitiveMode::Resting, _) => CognitiveEventKind::ExitedRest,
            (_, CognitiveMode::Resting) => CognitiveEventKind::EnteredRest,
            (CognitiveMode::Responding, _) => CognitiveEventKind::ProcessingCompleted,
            (_, CognitiveMode::Responding) => CognitiveEventKind::ProcessingStarted,
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

/// Reconstructs only spans whose entry and exit are explicitly represented by
/// semantic events. Unmatched exits are ignored rather than guessed.
pub fn cognitive_spans(events: &[CognitiveEvent]) -> Vec<CognitiveSpan> {
    let mut spans = Vec::new();
    let mut open_processing: Option<CognitiveSpan> = None;
    let mut open_resting: Option<CognitiveSpan> = None;

    for event in events {
        match event.kind {
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
        assert_eq!(
            event_between(Some(previous), current, 1, 42, true, true)
                .map(|event| event.kind),
            Some(CognitiveEventKind::ProcessingStarted)
        );
    }

    #[test]
    fn rest_transition_is_explicit() {
        let previous = state(true, false, 0.8, 0.3, 0.1);
        let current = state(true, false, 0.8, 0.1, 0.1);
        assert_eq!(
            event_between(Some(previous), current, 2, 42, false, false)
                .map(|event| event.kind),
            Some(CognitiveEventKind::EnteredRest)
        );
    }

    #[test]
    fn explicit_surprise_creates_event_without_state_change() {
        let previous = state(true, false, 0.8, 0.3, 0.1);
        let current = state(true, false, 0.8, 0.3, 0.1);
        assert_eq!(
            event_between(Some(previous), current, 3, 42, true, false)
                .map(|event| event.kind),
            Some(CognitiveEventKind::SurpriseDetected)
        );
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
    fn unmatched_exit_does_not_fabricate_a_span() {
        let events = vec![
            CognitiveEvent::lifecycle(1, CognitiveEventKind::ProcessingCompleted, 44),
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
    fn quiet_cycle_creates_no_event() {
        let previous = state(true, false, 0.8, 0.3, 0.1);
        let current = state(true, false, 0.8, 0.3, 0.1);
        assert!(event_between(Some(previous), current, 4, 42, false, false).is_none());
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
        let surprise = CognitiveEvent::from_state(2, CognitiveEventKind::SurpriseDetected, 9, state);
        assert_eq!(surprise.evidence_basis, EventEvidenceBasis::ExplicitSurpriseSignal);
        assert_eq!(surprise.evidence_basis.label(), "explicit surprise signal");
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
}
