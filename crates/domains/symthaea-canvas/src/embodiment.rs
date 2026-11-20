// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-aware, renderer-neutral semantic state for Symthaea embodiments.
//!
//! This module deliberately does not render an avatar or infer emotional states.
//! Runtime adapters publish explicit, ordered events; renderers translate the
//! resulting snapshot into an abstract presence, a conversational puppet, or a
//! cognitive visualization. Every displayed signal retains its provenance.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Runtime cognitive activity that an embodiment may communicate.
///
/// Speech output is modeled independently by [SpeechOutputState] so retrieval,
/// planning, and other cognitive activity can overlap with audio playback. These
/// are operational states, not claims about subjective experience.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbodimentActivity {
    Idle,
    Listening,
    Perceiving,
    Retrieving,
    Associating,
    Predicting,
    Planning,
    ExecutingTool,
    AwaitingTool,
    Interrupted,
    Fault,
}


/// Lifecycle of generated speech output. Only Playing authorizes viseme motion.
///
/// Synthesized or queued audio is not evidence that sound is currently being
/// played. The playback backend must emit Playing after it actually starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechOutputState {
    Synthesizing,
    Ready,
    Queued,
    Playing,
    Completed,
    Interrupted,
    Unavailable,
    Failed,
}

/// Origin of a signal supplied to an embodiment renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalProvenance {
    /// An explicit event emitted by a runtime subsystem.
    RuntimeEvent,
    /// A value computed from runtime telemetry or a model output.
    RuntimeDerived,
    /// An explicit choice made by the user.
    UserControlled,
    /// A renderer-local or decorative effect; never evidence of cognition.
    RendererLocal,
}

/// The five vowel poses commonly exposed by humanoid avatar formats, plus rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Viseme {
    Rest,
    Aa,
    Ih,
    Ou,
    Ee,
    Oh,
}

/// Semantic gaze destination. Renderers may map these to different rig controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GazeTarget {
    ConversationPartner,
    Content,
    Offscreen,
    Neutral,
}

/// An observed value with enough metadata to audit where and when it originated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observed<T> {
    pub value: T,
    pub provenance: SignalProvenance,
    pub sequence: u64,
    /// Monotonic nanoseconds relative to the current embodiment session.
    pub monotonic_ns: u64,
}

impl<T> Observed<T> {
    fn new(
        value: T,
        provenance: SignalProvenance,
        sequence: u64,
        monotonic_ns: u64,
    ) -> Self {
        Self {
            value,
            provenance,
            sequence,
            monotonic_ns,
        }
    }
}

fn make_observed<T>(
    value: T,
    provenance: SignalProvenance,
    sequence: u64,
    monotonic_ns: u64,
) -> Observed<T> {
    Observed::new(value, provenance, sequence, monotonic_ns)
}

/// A typed event emitted by a runtime adapter or explicit user control.
///
/// Sequence numbers must strictly increase within one controller/session.
/// Timestamps must be monotonic non-decreasing. Gaps are allowed so a dropped
/// event does not permanently block later events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbodimentEvent {
    pub sequence: u64,
    pub monotonic_ns: u64,
    pub provenance: SignalProvenance,
    pub kind: EmbodimentEventKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmbodimentEventKind {
    ActivityChanged {
        activity: EmbodimentActivity,
    },
    SpeechOutputChanged {
        state: SpeechOutputState,
    },
    VisemeChanged {
        viseme: Viseme,
        weight: f32,
    },
    GazeChanged {
        target: GazeTarget,
    },
    /// Explicit normalized uncertainty supplied by a calibrated runtime source.
    /// This module does not estimate or calibrate the value.
    UncertaintyChanged {
        level: f32,
    },
    /// Clear short-lived visual signals without changing the activity.
    ClearTransient,
    /// End the current interaction and clear all transient signals.
    Reset,
}

/// Renderer-facing snapshot. Missing fields mean "no signal supplied", not zero.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbodimentSnapshot {
    pub activity: Option<Observed<EmbodimentActivity>>,
    pub speech_output: Option<Observed<SpeechOutputState>>,
    pub viseme: Option<Observed<VisemeSignal>>,
    pub gaze: Option<Observed<GazeTarget>>,
    pub uncertainty: Option<Observed<f32>>,
    pub last_event_sequence: Option<u64>,
    pub last_event_ns: Option<u64>,
}

impl EmbodimentSnapshot {
    pub fn current_activity(&self) -> Option<EmbodimentActivity> {
        self.activity.as_ref().map(|observed| observed.value)
    }

    pub fn current_speech_output(&self) -> Option<SpeechOutputState> {
        self.speech_output.as_ref().map(|observed| observed.value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VisemeSignal {
    pub viseme: Viseme,
    pub weight: f32,
}

/// Stateful, deterministic event reducer for a single embodiment session.
#[derive(Debug, Clone, Default)]
pub struct EmbodimentController {
    snapshot: EmbodimentSnapshot,
}

impl EmbodimentController {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> &EmbodimentSnapshot {
        &self.snapshot
    }

    /// Apply one event, rejecting stale, reordered, or invalid signals.
    ///
    /// Rejected events leave the snapshot and sequence cursor unchanged.
    pub fn ingest(&mut self, event: EmbodimentEvent) -> Result<(), EmbodimentIngestError> {
        if let Some(last) = self.snapshot.last_event_sequence {
            if event.sequence <= last {
                return Err(EmbodimentIngestError::NonIncreasingSequence {
                    last,
                    received: event.sequence,
                });
            }
        }
        if let Some(last) = self.snapshot.last_event_ns {
            if event.monotonic_ns < last {
                return Err(EmbodimentIngestError::TimestampRegression {
                    last,
                    received: event.monotonic_ns,
                });
            }
        }

        match &event.kind {
            EmbodimentEventKind::ActivityChanged { .. } => {
                validate_runtime_provenance("activity", event.provenance)?;
            }
            EmbodimentEventKind::SpeechOutputChanged { .. } => {
                validate_runtime_event_provenance("speech output", event.provenance)?;
            }
            EmbodimentEventKind::Reset => {
                validate_runtime_provenance("reset", event.provenance)?;
            }
            EmbodimentEventKind::VisemeChanged { weight, .. } => {
                validate_runtime_provenance("viseme", event.provenance)?;
                validate_unit_interval("viseme weight", *weight)?;
                if self.snapshot.current_speech_output() != Some(SpeechOutputState::Playing) {
                    return Err(EmbodimentIngestError::VisemeWithoutPlayback);
                }
            }
            EmbodimentEventKind::UncertaintyChanged { level } => {
                validate_runtime_provenance("uncertainty", event.provenance)?;
                validate_unit_interval("uncertainty", *level)?;
            }
            _ => {}
        }

        let provenance = event.provenance;
        let sequence = event.sequence;
        let monotonic_ns = event.monotonic_ns;
        match event.kind {
            EmbodimentEventKind::ActivityChanged { activity } => {
                self.snapshot.activity = Some(make_observed(
                    activity,
                    provenance,
                    sequence,
                    monotonic_ns,
                ));
            }
            EmbodimentEventKind::SpeechOutputChanged { state } => {
                self.snapshot.speech_output = Some(make_observed(
                    state,
                    provenance,
                    sequence,
                    monotonic_ns,
                ));
                if state != SpeechOutputState::Playing {
                    self.snapshot.viseme = None;
                }
            }
            EmbodimentEventKind::VisemeChanged { viseme, weight } => {
                self.snapshot.viseme = Some(make_observed(
                    VisemeSignal { viseme, weight },
                    provenance,
                    sequence,
                    monotonic_ns,
                ));
            }
            EmbodimentEventKind::GazeChanged { target } => {
                self.snapshot.gaze = Some(make_observed(
                    target,
                    provenance,
                    sequence,
                    monotonic_ns,
                ));
            }
            EmbodimentEventKind::UncertaintyChanged { level } => {
                self.snapshot.uncertainty = Some(make_observed(
                    level,
                    provenance,
                    sequence,
                    monotonic_ns,
                ));
            }
            EmbodimentEventKind::ClearTransient => {
                self.snapshot.viseme = None;
                self.snapshot.gaze = None;
                self.snapshot.uncertainty = None;
            }
            EmbodimentEventKind::Reset => {
                self.snapshot.activity = Some(make_observed(
                    EmbodimentActivity::Idle,
                    provenance,
                    sequence,
                    monotonic_ns,
                ));
                self.snapshot.speech_output = None;
                self.snapshot.viseme = None;
                self.snapshot.gaze = None;
                self.snapshot.uncertainty = None;
            }
        }

        self.snapshot.last_event_sequence = Some(sequence);
        self.snapshot.last_event_ns = Some(monotonic_ns);
        Ok(())
    }
}

fn validate_runtime_provenance(
    field: &'static str,
    provenance: SignalProvenance,
) -> Result<(), EmbodimentIngestError> {
    match provenance {
        SignalProvenance::RuntimeEvent | SignalProvenance::RuntimeDerived => Ok(()),
        SignalProvenance::UserControlled | SignalProvenance::RendererLocal => {
            Err(EmbodimentIngestError::SignalProvenanceNotAllowed { field, provenance })
        }
    }
}

fn validate_runtime_event_provenance(
    field: &'static str,
    provenance: SignalProvenance,
) -> Result<(), EmbodimentIngestError> {
    if provenance == SignalProvenance::RuntimeEvent {
        Ok(())
    } else {
        Err(EmbodimentIngestError::SignalProvenanceNotAllowed { field, provenance })
    }
}

fn validate_unit_interval(field: &'static str, value: f32) -> Result<(), EmbodimentIngestError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(EmbodimentIngestError::InvalidUnitInterval { field, value })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EmbodimentIngestError {
    NonIncreasingSequence { last: u64, received: u64 },
    TimestampRegression { last: u64, received: u64 },
    InvalidUnitInterval { field: &'static str, value: f32 },
    VisemeWithoutPlayback,
    SignalProvenanceNotAllowed {
        field: &'static str,
        provenance: SignalProvenance,
    },
}

impl fmt::Display for EmbodimentIngestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonIncreasingSequence { last, received } => {
                write!(f, "event sequence {received} is not greater than {last}")
            }
            Self::TimestampRegression { last, received } => {
                write!(f, "event timestamp {received} regresses behind {last}")
            }
            Self::InvalidUnitInterval { field, value } => {
                write!(f, "{field} must be finite and within [0, 1], got {value}")
            }
            Self::VisemeWithoutPlayback => {
                write!(f, "viseme updates require confirmed speech playback")
            }
            Self::SignalProvenanceNotAllowed { field, provenance } => {
                write!(f, "{field} cannot use {provenance:?} provenance")
            }
        }
    }
}

impl std::error::Error for EmbodimentIngestError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(sequence: u64, monotonic_ns: u64, kind: EmbodimentEventKind) -> EmbodimentEvent {
        EmbodimentEvent {
            sequence,
            monotonic_ns,
            provenance: SignalProvenance::RuntimeEvent,
            kind,
        }
    }

    #[test]
    fn explicit_runtime_activity_is_preserved_with_provenance() {
        let mut controller = EmbodimentController::new();
        controller
            .ingest(event(
                1,
                10,
                EmbodimentEventKind::ActivityChanged {
                    activity: EmbodimentActivity::Associating,
                },
            ))
            .unwrap();

        let activity = controller.snapshot().activity.as_ref().unwrap();
        assert_eq!(activity.value, EmbodimentActivity::Associating);
        assert_eq!(activity.provenance, SignalProvenance::RuntimeEvent);
        assert_eq!(activity.sequence, 1);
    }

    #[test]
    fn viseme_requires_confirmed_playback_state() {
        let mut controller = EmbodimentController::new();
        let result = controller.ingest(event(
            1,
            10,
            EmbodimentEventKind::VisemeChanged {
                viseme: Viseme::Aa,
                weight: 0.8,
            },
        ));
        assert_eq!(result, Err(EmbodimentIngestError::VisemeWithoutPlayback));
        assert_eq!(controller.snapshot().last_event_sequence, None);
    }

    #[test]
    fn leaving_playback_clears_stale_viseme() {
        let mut controller = EmbodimentController::new();
        controller
            .ingest(event(
                1,
                10,
                EmbodimentEventKind::SpeechOutputChanged {
                    state: SpeechOutputState::Playing,
                },
            ))
            .unwrap();
        controller
            .ingest(event(
                2,
                20,
                EmbodimentEventKind::VisemeChanged {
                    viseme: Viseme::Ee,
                    weight: 0.7,
                },
            ))
            .unwrap();
        assert!(controller.snapshot().viseme.is_some());

        controller
            .ingest(event(
                3,
                30,
                EmbodimentEventKind::SpeechOutputChanged {
                    state: SpeechOutputState::Completed,
                },
            ))
            .unwrap();
        assert!(controller.snapshot().viseme.is_none());
    }

    #[test]
    fn stale_sequence_is_rejected_without_mutation() {
        let mut controller = EmbodimentController::new();
        controller
            .ingest(event(
                4,
                40,
                EmbodimentEventKind::ActivityChanged {
                    activity: EmbodimentActivity::Listening,
                },
            ))
            .unwrap();
        let before = controller.snapshot().clone();

        let result = controller.ingest(event(
            4,
            50,
            EmbodimentEventKind::ActivityChanged {
                activity: EmbodimentActivity::Fault,
            },
        ));
        assert_eq!(
            result,
            Err(EmbodimentIngestError::NonIncreasingSequence {
                last: 4,
                received: 4
            })
        );
        assert_eq!(*controller.snapshot(), before);
    }

    #[test]
    fn timestamp_regression_is_rejected_without_mutation() {
        let mut controller = EmbodimentController::new();
        controller
            .ingest(event(
                1,
                100,
                EmbodimentEventKind::ActivityChanged {
                    activity: EmbodimentActivity::Listening,
                },
            ))
            .unwrap();
        let before = controller.snapshot().clone();

        let result = controller.ingest(event(
            2,
            99,
            EmbodimentEventKind::ActivityChanged {
                activity: EmbodimentActivity::Predicting,
            },
        ));
        assert_eq!(
            result,
            Err(EmbodimentIngestError::TimestampRegression {
                last: 100,
                received: 99
            })
        );
        assert_eq!(*controller.snapshot(), before);
    }

    #[test]
    fn invalid_numeric_signal_is_rejected() {
        let mut controller = EmbodimentController::new();
        controller
            .ingest(event(
                1,
                10,
                EmbodimentEventKind::ActivityChanged {
                    activity: EmbodimentActivity::Speaking,
                },
            ))
            .unwrap();

        let result = controller.ingest(event(
            2,
            20,
            EmbodimentEventKind::VisemeChanged {
                viseme: Viseme::Oh,
                weight: f32::NAN,
            },
        ));
        assert!(matches!(
            result,
            Err(EmbodimentIngestError::InvalidUnitInterval {
                field: "viseme weight",
                ..
            })
        ));
        assert_eq!(controller.snapshot().last_event_sequence, Some(1));
    }

    #[test]
    fn reset_clears_transient_signals_and_returns_to_idle() {
        let mut controller = EmbodimentController::new();
        controller
            .ingest(event(
                1,
                10,
                EmbodimentEventKind::ActivityChanged {
                    activity: EmbodimentActivity::Speaking,
                },
            ))
            .unwrap();
        controller
            .ingest(event(
                2,
                20,
                EmbodimentEventKind::VisemeChanged {
                    viseme: Viseme::Ou,
                    weight: 0.5,
                },
            ))
            .unwrap();
        controller
            .ingest(event(
                3,
                30,
                EmbodimentEventKind::UncertaintyChanged { level: 0.3 },
            ))
            .unwrap();
        controller
            .ingest(event(4, 40, EmbodimentEventKind::Reset))
            .unwrap();

        assert_eq!(
            controller.snapshot().current_activity(),
            Some(EmbodimentActivity::Idle)
        );
        assert!(controller.snapshot().viseme.is_none());
        assert!(controller.snapshot().uncertainty.is_none());
    }

    #[test]
    fn equal_timestamps_are_allowed_when_sequence_increases() {
        let mut controller = EmbodimentController::new();
        for sequence in 1..=2 {
            controller
                .ingest(event(
                    sequence,
                    100,
                    EmbodimentEventKind::ActivityChanged {
                        activity: EmbodimentActivity::Listening,
                    },
                ))
                .unwrap();
        }
        assert_eq!(controller.snapshot().last_event_sequence, Some(2));
    }


    #[test]
    fn renderer_local_effect_cannot_claim_runtime_activity() {
        let mut controller = EmbodimentController::new();
        let mut event = event(
            1,
            10,
            EmbodimentEventKind::ActivityChanged {
                activity: EmbodimentActivity::Predicting,
            },
        );
        event.provenance = SignalProvenance::RendererLocal;

        assert!(matches!(
            controller.ingest(event),
            Err(EmbodimentIngestError::SignalProvenanceNotAllowed {
                field: "activity",
                provenance: SignalProvenance::RendererLocal
            })
        ));
        assert_eq!(controller.snapshot().last_event_sequence, None);
    }


    #[test]
    fn renderer_local_reset_cannot_manufacture_idle_runtime_state() {
        let mut controller = EmbodimentController::new();
        controller
            .ingest(event(
                1,
                10,
                EmbodimentEventKind::ActivityChanged {
                    activity: EmbodimentActivity::Predicting,
                },
            ))
            .unwrap();
        let before = controller.snapshot().clone();

        let mut reset = event(2, 20, EmbodimentEventKind::Reset);
        reset.provenance = SignalProvenance::RendererLocal;
        assert!(matches!(
            controller.ingest(reset),
            Err(EmbodimentIngestError::SignalProvenanceNotAllowed {
                field: "reset",
                provenance: SignalProvenance::RendererLocal
            })
        ));
        assert_eq!(*controller.snapshot(), before);
    }

    #[test]
    fn snapshot_round_trips_through_json() {
        let mut controller = EmbodimentController::new();
        controller
            .ingest(event(
                1,
                10,
                EmbodimentEventKind::ActivityChanged {
                    activity: EmbodimentActivity::Predicting,
                },
            ))
            .unwrap();
        let encoded = serde_json::to_string(controller.snapshot()).unwrap();
        let decoded: EmbodimentSnapshot = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, *controller.snapshot());
    }
}
