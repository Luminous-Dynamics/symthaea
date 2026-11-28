// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Canonical semantic cues and deterministic replay for Symthaea embodiments.
//!
//! This module deliberately stops before animation or rendering. Every form gets
//! the same typed semantic cue; a renderer may choose its own visual grammar but
//! may not silently alter the cue's meaning or epistemic provenance.

use super::embodiment_event::{
    BackchannelSignal, EmbodimentEvent, EmbodimentStreamValidator, EpistemicStatus,
    EventKind, EventPayload, EventSource, EventValidationError, EvidenceState,
    ExpressionChannel, ExpressionIntentKind, PrivacyClass, TurnState, UserControlAction,
};
use serde::{Deserialize, Serialize};

/// Presentation form identifiers. They describe adapters, not separate agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbodimentForm {
    LivingConstellation,
    PuppetTheatre,
    MycelialBeing,
    Swarm,
    LivingInstrument,
    HearthCompanion,
    CognitiveMandala,
}

impl EmbodimentForm {
    pub const ALL: [Self; 7] = [
        Self::LivingConstellation,
        Self::PuppetTheatre,
        Self::MycelialBeing,
        Self::Swarm,
        Self::LivingInstrument,
        Self::HearthCompanion,
        Self::CognitiveMandala,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::LivingConstellation => "living_constellation",
            Self::PuppetTheatre => "puppet_theatre",
            Self::MycelialBeing => "mycelial_being",
            Self::Swarm => "swarm",
            Self::LivingInstrument => "living_instrument",
            Self::HearthCompanion => "hearth_companion",
            Self::CognitiveMandala => "cognitive_mandala",
        }
    }
}

/// Semantic content shared by every embodiment renderer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cue", rename_all = "snake_case")]
pub enum SemanticCue {
    Listening {
        active: bool,
    },
    Speaking {
        active: bool,
        segment_id: String,
    },
    TurnState {
        state: TurnState,
        target: Option<String>,
    },
    Backchannel {
        signal: BackchannelSignal,
        intensity: f32,
    },
    AttentionFocus {
        target: String,
        strength: Option<f32>,
    },
    ExpressionIntent {
        intent: ExpressionIntentKind,
        intensity: f32,
        channel: Option<ExpressionChannel>,
    },
    Uncertainty {
        subject: String,
        level: f32,
        calibration_ref: Option<String>,
    },
    EvidenceStatus {
        subject_id: String,
        status: EvidenceState,
        verifier_receipt_ref: Option<String>,
    },
    UserControl {
        action: UserControlAction,
        form_id: Option<String>,
        target_event_id: Option<String>,
        enabled: Option<bool>,
        intensity: Option<f32>,
    },
}

impl SemanticCue {
    /// Stable semantic key for renderer mappings and benchmark aggregation.
    pub fn semantic_key(&self) -> &'static str {
        match self {
            Self::Listening { active: true } => "conversation.listening.active",
            Self::Listening { active: false } => "conversation.listening.inactive",
            Self::Speaking { active: true, .. } => "conversation.speaking.active",
            Self::Speaking { active: false, .. } => "conversation.speaking.inactive",
            Self::TurnState { state: TurnState::Wait, .. } => "conversation.turn.wait",
            Self::TurnState { state: TurnState::Respond, .. } => "conversation.turn.respond",
            Self::TurnState { state: TurnState::Continue, .. } => "conversation.turn.continue",
            Self::TurnState { state: TurnState::Yield, .. } => "conversation.turn.yield",
            Self::Backchannel { .. } => "conversation.backchannel",
            Self::AttentionFocus { .. } => "attention.focus",
            Self::ExpressionIntent { .. } => "expression.intent",
            Self::Uncertainty { .. } => "cognition.uncertainty",
            Self::EvidenceStatus { .. } => "evidence.status",
            Self::UserControl { .. } => "user.control",
        }
    }

    /// Text alternative describes the reported event, not an inferred emotion
    /// or a claim that a downstream action has completed.
    fn accessible_description(&self) -> String {
        match self {
            Self::Listening { active: true } => "Listening state reported active.".to_owned(),
            Self::Listening { active: false } => "Listening state reported inactive.".to_owned(),
            Self::Speaking { active: true, .. } => "Speech segment reported active.".to_owned(),
            Self::Speaking { active: false, .. } => "Speech segment reported inactive.".to_owned(),
            Self::TurnState { state, .. } => match state {
                TurnState::Wait => "Conversation action: wait.".to_owned(),
                TurnState::Respond => "Conversation action: respond.".to_owned(),
                TurnState::Continue => "Conversation action: continue.".to_owned(),
                TurnState::Yield => "Conversation action: yield the floor.".to_owned(),
            },
            Self::Backchannel { signal, .. } => {
                format!("Backchannel signal requested: {signal:?}.")
            }
            Self::AttentionFocus { target, .. } => {
                format!("Reported attention target: {target}.")
            }
            Self::ExpressionIntent { intent, .. } => {
                format!("Presentation intent reported: {intent:?}.")
            }
            Self::Uncertainty { subject, level, .. } => {
                format!("Reported uncertainty indicator for {subject}: {level:.3}.")
            }
            Self::EvidenceStatus { subject_id, status, verifier_receipt_ref } => {
                let receipt = if verifier_receipt_ref.is_some() {
                    " A verifier receipt reference is present."
                } else {
                    ""
                };
                format!("Evidence status reported for {subject_id}: {status:?}.{receipt}")
            }
            Self::UserControl { action, .. } => {
                format!("User control requested: {action:?}.")
            }
        }
    }
}

/// Form-independent cue with the full provenance needed by a renderer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalCue {
    pub session_id: String,
    pub clock_domain_id: String,
    pub event_id: String,
    pub sequence: u64,
    pub event_kind: EventKind,
    pub semantic: SemanticCue,
    pub source: EventSource,
    pub observed_at_monotonic_ns: String,
    pub expires_at_monotonic_ns: String,
    pub epistemic_status: EpistemicStatus,
    pub confidence: Option<f64>,
    pub evidence_refs: Vec<String>,
    pub privacy_class: PrivacyClass,
}

impl CanonicalCue {
    /// Privacy-aware text alternative for generic renderers and assistive
    /// technology. Sensitive payload values are not copied into this label.
    pub fn accessible_description(&self) -> String {
        match self.privacy_class {
            PrivacyClass::Public => self.semantic.accessible_description(),
            PrivacyClass::LocalSensitive => {
                "Sensitive interaction cue; details omitted from generic presentation.".to_owned()
            }
            PrivacyClass::Restricted => {
                "Restricted interaction cue; details omitted.".to_owned()
            }
        }
    }

    pub fn from_event(event: &EmbodimentEvent) -> Result<Self, EventValidationError> {
        event.validate()?;
        let semantic = match &event.payload {
            EventPayload::Listening(value) => SemanticCue::Listening {
                active: value.active,
            },
            EventPayload::Speaking(value) => SemanticCue::Speaking {
                active: value.active,
                segment_id: value.segment_id.clone(),
            },
            EventPayload::TurnState(value) => SemanticCue::TurnState {
                state: value.state,
                target: value.target.clone(),
            },
            EventPayload::Backchannel(value) => SemanticCue::Backchannel {
                signal: value.signal,
                intensity: value.intensity,
            },
            EventPayload::AttentionFocus(value) => SemanticCue::AttentionFocus {
                target: value.target.clone(),
                strength: value.strength,
            },
            EventPayload::ExpressionIntent(value) => SemanticCue::ExpressionIntent {
                intent: value.intent,
                intensity: value.intensity,
                channel: value.channel,
            },
            EventPayload::Uncertainty(value) => SemanticCue::Uncertainty {
                subject: value.subject.clone(),
                level: value.level,
                calibration_ref: value.calibration_ref.clone(),
            },
            EventPayload::EvidenceStatus(value) => SemanticCue::EvidenceStatus {
                subject_id: value.subject_id.clone(),
                status: value.status,
                verifier_receipt_ref: value.verifier_receipt_ref.clone(),
            },
            EventPayload::UserControl(value) => SemanticCue::UserControl {
                action: value.action,
                form_id: value.form_id.clone(),
                target_event_id: value.target_event_id.clone(),
                enabled: value.enabled,
                intensity: value.intensity,
            },
        };

        Ok(Self {
            session_id: event.session_id.clone(),
            clock_domain_id: event.clock_domain_id.clone(),
            event_id: event.event_id.clone(),
            sequence: event.sequence,
            event_kind: event.event_kind,
            semantic,
            source: event.source.clone(),
            observed_at_monotonic_ns: event.observed_at_monotonic_ns.clone(),
            expires_at_monotonic_ns: event.expires_at_monotonic_ns.clone(),
            epistemic_status: event.epistemic_status,
            confidence: event.confidence,
            evidence_refs: event.evidence_refs.clone(),
            privacy_class: event.privacy_class,
        })
    }
}

/// Renderer-facing envelope. The semantic cue is identical across forms; only
/// the selected form identifier changes. No animation or physical output occurs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormCue {
    pub form: EmbodimentForm,
    pub cue: CanonicalCue,
}

pub fn project_for_form(cue: &CanonicalCue, form: EmbodimentForm) -> FormCue {
    FormCue {
        form,
        cue: cue.clone(),
    }
}

/// One recorded event and the monotonic "now" sampled from that event's clock
/// domain. Cross-device traces require explicit clock synchronization before
/// they can be represented in this format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayStep {
    pub event: EmbodimentEvent,
    pub now_monotonic_ns: String,
    pub now_clock_domain_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayError {
    pub step_index: usize,
    pub cause: EventValidationError,
}

/// Validate and replay a trace without a renderer, GPU, audio device, or network.
/// The returned cues are deterministic for byte-identical input events and
/// preserve the source event order.
pub fn replay_trace(
    session_id: impl Into<String>,
    steps: &[ReplayStep],
) -> Result<Vec<CanonicalCue>, ReplayError> {
    let mut validator = EmbodimentStreamValidator::new(session_id)
        .map_err(|cause| ReplayError { step_index: 0, cause })?;
    let mut cues = Vec::with_capacity(steps.len());

    for (step_index, step) in steps.iter().enumerate() {
        validator
            .accept(
                &step.event,
                &step.now_monotonic_ns,
                &step.now_clock_domain_id,
            )
            .map_err(|cause| ReplayError { step_index, cause })?;
        let cue = CanonicalCue::from_event(&step.event)
            .map_err(|cause| ReplayError { step_index, cause })?;
        cues.push(cue);
    }

    Ok(cues)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment_event::{
        EmbodimentEvent, EventPayload, EventSource, ListeningPayload,
    };

    fn event(sequence: u64, event_id: &str) -> EmbodimentEvent {
        EmbodimentEvent {
            schema: "symthaea.embodiment-event.v1".to_owned(),
            session_id: "session-1".to_owned(),
            clock_domain_id: "clock:dialogue-epoch-1".to_owned(),
            event_id: event_id.to_owned(),
            sequence,
            observed_at_monotonic_ns: (100 + sequence).to_string(),
            expires_at_monotonic_ns: "1000".to_owned(),
            event_kind: EventKind::ConversationListening,
            payload: EventPayload::Listening(ListeningPayload { active: true }),
            source: EventSource {
                component: "dialogue".to_owned(),
                instance_id: "dialogue-1".to_owned(),
                source_event_id: None,
                build_id: None,
            },
            epistemic_status: EpistemicStatus::Measured,
            confidence: None,
            evidence_refs: Vec::new(),
            privacy_class: PrivacyClass::Public,
        }
    }

    fn steps() -> Vec<ReplayStep> {
        vec![
            ReplayStep {
                event: event(0, "event-0"),
                now_monotonic_ns: "200".to_owned(),
                now_clock_domain_id: "clock:dialogue-epoch-1".to_owned(),
            },
            ReplayStep {
                event: event(1, "event-1"),
                now_monotonic_ns: "300".to_owned(),
                now_clock_domain_id: "clock:dialogue-epoch-1".to_owned(),
            },
        ]
    }

    #[test]
    fn replay_is_deterministic_for_identical_traces() {
        let trace = steps();
        assert_eq!(
            replay_trace("session-1", &trace).unwrap(),
            replay_trace("session-1", &trace).unwrap()
        );
    }

    #[test]
    fn replay_rejects_a_clock_reading_from_another_domain() {
        let mut trace = steps();
        trace[0].now_clock_domain_id = "clock:other-process-epoch".to_owned();
        let error = replay_trace("session-1", &trace).unwrap_err();
        assert_eq!(error.step_index, 0);
        assert_eq!(error.cause, EventValidationError::ClockDomainMismatch);
    }

    #[test]
    fn replay_preserves_source_sequence_and_identity() {
        let cues = replay_trace("session-1", &steps()).unwrap();
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].sequence, 0);
        assert_eq!(cues[0].event_id, "event-0");
        assert_eq!(cues[1].sequence, 1);
        assert_eq!(cues[1].event_id, "event-1");
    }

    #[test]
    fn replay_rejects_invalid_second_step_without_returning_partial_trace() {
        let mut trace = steps();
        trace[1].event.sequence = 3;
        let error = replay_trace("session-1", &trace).unwrap_err();
        assert_eq!(error.step_index, 1);
        assert_eq!(error.cause, EventValidationError::SequenceGap);
    }

    #[test]
    fn every_form_receives_the_same_canonical_semantics() {
        let cue = CanonicalCue::from_event(&event(0, "event-0")).unwrap();
        let projections = EmbodimentForm::ALL.map(|form| project_for_form(&cue, form));
        for projection in &projections {
            assert_eq!(projection.cue, cue);
            assert_eq!(projection.cue.semantic.semantic_key(), "conversation.listening.active");
        }
        assert_eq!(
            projections.iter().map(|p| p.form.id()).collect::<Vec<_>>(),
            EmbodimentForm::ALL.map(EmbodimentForm::id).to_vec()
        );
    }

    #[test]
    fn maps_all_event_kinds_to_stable_semantic_keys() {
        use crate::embodiment_event::{
            AttentionFocusPayload, BackchannelPayload, EvidenceStatusPayload,
            ExpressionIntentPayload, ListeningPayload, SpeakingPayload,
            TurnStatePayload, UncertaintyPayload, UserControlPayload,
        };

        let cases = vec![
            (
                EventKind::ConversationListening,
                EventPayload::Listening(ListeningPayload { active: true }),
                "conversation.listening.active",
            ),
            (
                EventKind::ConversationSpeaking,
                EventPayload::Speaking(SpeakingPayload {
                    active: true,
                    segment_id: "segment-1".to_owned(),
                    text_ref: None,
                }),
                "conversation.speaking.active",
            ),
            (
                EventKind::ConversationTurnState,
                EventPayload::TurnState(TurnStatePayload {
                    state: TurnState::Yield,
                    target: None,
                }),
                "conversation.turn.yield",
            ),
            (
                EventKind::ConversationBackchannel,
                EventPayload::Backchannel(BackchannelPayload {
                    signal: BackchannelSignal::VisualAck,
                    intensity: 0.5,
                }),
                "conversation.backchannel",
            ),
            (
                EventKind::AttentionFocus,
                EventPayload::AttentionFocus(AttentionFocusPayload {
                    target: "document-1".to_owned(),
                    strength: Some(0.7),
                }),
                "attention.focus",
            ),
            (
                EventKind::ExpressionIntent,
                EventPayload::ExpressionIntent(ExpressionIntentPayload {
                    intent: ExpressionIntentKind::Question,
                    intensity: 0.4,
                    channel: Some(ExpressionChannel::Gesture),
                }),
                "expression.intent",
            ),
            (
                EventKind::CognitionUncertainty,
                EventPayload::Uncertainty(UncertaintyPayload {
                    subject: "hypothesis-1".to_owned(),
                    level: 0.4,
                    calibration_ref: None,
                }),
                "cognition.uncertainty",
            ),
            (
                EventKind::EvidenceStatus,
                EventPayload::EvidenceStatus(EvidenceStatusPayload {
                    subject_id: "claim-1".to_owned(),
                    status: EvidenceState::Unverified,
                    verifier_receipt_ref: None,
                }),
                "evidence.status",
            ),
            (
                EventKind::UserControl,
                EventPayload::UserControl(UserControlPayload {
                    action: UserControlAction::Pause,
                    form_id: None,
                    target_event_id: None,
                    enabled: None,
                    intensity: None,
                }),
                "user.control",
            ),
        ];

        for (kind, payload, expected_key) in cases {
            let mut source = event(0, "event-0");
            source.event_kind = kind;
            source.payload = payload;
            let cue = CanonicalCue::from_event(&source).unwrap();
            assert_eq!(cue.semantic.semantic_key(), expected_key);
        }
    }

    #[test]
    fn supported_evidence_accessibility_does_not_upgrade_status_to_verified() {
        let mut source = event(0, "event-0");
        source.event_kind = EventKind::EvidenceStatus;
        source.payload = EventPayload::EvidenceStatus(
            crate::embodiment_event::EvidenceStatusPayload {
                subject_id: "claim-1".to_owned(),
                status: EvidenceState::Supported,
                verifier_receipt_ref: Some("receipt:verifier-1".to_owned()),
            },
        );
        source.evidence_refs = vec!["sha256:abc123".to_owned()];
        let cue = CanonicalCue::from_event(&source).unwrap();
        let description = cue.accessible_description();
        assert!(description.contains("Supported"));
        assert!(!description.to_lowercase().contains("verified"));
    }

    #[test]
    fn restricted_cues_do_not_leak_subjects_through_accessible_text() {
        let mut restricted = event(0, "event-0");
        restricted.event_kind = EventKind::CognitionUncertainty;
        restricted.payload = EventPayload::Uncertainty(crate::embodiment_event::UncertaintyPayload {
            subject: "secret-project-name".to_owned(),
            level: 0.4,
            calibration_ref: None,
        });
        restricted.privacy_class = PrivacyClass::Restricted;
        let cue = CanonicalCue::from_event(&restricted).unwrap();
        let description = cue.accessible_description();
        assert!(!description.contains("secret-project-name"));
        assert!(description.contains("Restricted"));
    }

    #[test]
    fn local_sensitive_cues_are_redacted_from_generic_accessible_text() {
        let mut sensitive = event(0, "event-0");
        sensitive.event_kind = EventKind::AttentionFocus;
        sensitive.payload = EventPayload::AttentionFocus(
            crate::embodiment_event::AttentionFocusPayload {
                target: "private-personal-context".to_owned(),
                strength: None,
            },
        );
        sensitive.privacy_class = PrivacyClass::LocalSensitive;
        let cue = CanonicalCue::from_event(&sensitive).unwrap();
        assert!(!cue.accessible_description().contains("private-personal-context"));
    }

    #[test]
    fn accessible_description_does_not_claim_an_emotion() {
        let cue = CanonicalCue::from_event(&event(0, "event-0")).unwrap();
        assert_eq!(
            cue.accessible_description(),
            "Listening state reported active."
        );
    }
}
