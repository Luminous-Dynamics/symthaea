// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Renderer-neutral, provenance-bearing events for Symthaea embodiments.
//!
//! This module intentionally has no renderer, GPU, network, microphone, camera,
//! or actuator dependency. It defines the boundary between cognitive/communication
//! events and their visual, audio, haptic, or physical presentation.
//!
//! It does not infer consciousness or emotion. It validates event structure,
//! local stream ordering, freshness, and a small set of epistemic invariants.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::str::FromStr;

/// Version identifier shared with the JSON Schema document.
pub const EMBODIMENT_EVENT_SCHEMA_V1: &str = "symthaea.embodiment-event.v1";

/// JSON-safe upper bound for sequence numbers.
pub const MAX_SAFE_SEQUENCE: u64 = 9_007_199_254_740_991;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicStatus {
    Measured,
    Derived,
    Inferred,
    Proposed,
    PresentationOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyClass {
    Public,
    LocalSensitive,
    Restricted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    #[serde(rename = "conversation.listening")]
    ConversationListening,
    #[serde(rename = "conversation.speaking")]
    ConversationSpeaking,
    #[serde(rename = "conversation.turn_state")]
    ConversationTurnState,
    #[serde(rename = "conversation.backchannel")]
    ConversationBackchannel,
    #[serde(rename = "attention.focus")]
    AttentionFocus,
    #[serde(rename = "expression.intent")]
    ExpressionIntent,
    #[serde(rename = "cognition.uncertainty")]
    CognitionUncertainty,
    #[serde(rename = "evidence.status")]
    EvidenceStatus,
    #[serde(rename = "user.control")]
    UserControl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnState {
    Wait,
    Respond,
    Continue,
    Yield,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackchannelSignal {
    VocalAck,
    VisualAck,
    GestureAck,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpressionIntentKind {
    Explain,
    Invite,
    Caution,
    Celebrate,
    Reflect,
    Question,
    Neutral,
    Transition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpressionChannel {
    Light,
    Particle,
    Gesture,
    Sound,
    Haptic,
    Spatial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceState {
    Unverified,
    Supported,
    Stale,
    Contradicted,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserControlAction {
    Pause,
    Cancel,
    ChooseForm,
    SetReducedMotion,
    SetIntensity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ListeningPayload {
    pub active: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeakingPayload {
    pub active: bool,
    pub segment_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TurnStatePayload {
    pub state: TurnState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackchannelPayload {
    pub signal: BackchannelSignal,
    pub intensity: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionFocusPayload {
    pub target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpressionIntentPayload {
    pub intent: ExpressionIntentKind,
    pub intensity: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<ExpressionChannel>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UncertaintyPayload {
    pub subject: String,
    pub level: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceStatusPayload {
    pub subject_id: String,
    pub status: EvidenceState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verifier_receipt_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UserControlPayload {
    pub action: UserControlAction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub form_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_event_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intensity: Option<f32>,
}

/// Untagged payloads keep the wire format aligned with the event-kind field:
/// payload is the event-specific object, without a second embedded tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EventPayload {
    Listening(ListeningPayload),
    Speaking(SpeakingPayload),
    TurnState(TurnStatePayload),
    Backchannel(BackchannelPayload),
    AttentionFocus(AttentionFocusPayload),
    ExpressionIntent(ExpressionIntentPayload),
    Uncertainty(UncertaintyPayload),
    EvidenceStatus(EvidenceStatusPayload),
    UserControl(UserControlPayload),
}

impl EventPayload {
    fn kind(&self) -> EventKind {
        match self {
            Self::Listening(_) => EventKind::ConversationListening,
            Self::Speaking(_) => EventKind::ConversationSpeaking,
            Self::TurnState(_) => EventKind::ConversationTurnState,
            Self::Backchannel(_) => EventKind::ConversationBackchannel,
            Self::AttentionFocus(_) => EventKind::AttentionFocus,
            Self::ExpressionIntent(_) => EventKind::ExpressionIntent,
            Self::Uncertainty(_) => EventKind::CognitionUncertainty,
            Self::EvidenceStatus(_) => EventKind::EvidenceStatus,
            Self::UserControl(_) => EventKind::UserControl,
        }
    }

    fn validate(&self, evidence_refs: &[String]) -> Result<(), EventValidationError> {
        match self {
            Self::Listening(_) => Ok(()),
            Self::TurnState(value) => {
                if let Some(target) = value.target.as_deref() {
                    validate_identifier("turn target", target)?;
                }
                Ok(())
            },
            Self::Speaking(value) => {
                validate_identifier("segment_id", &value.segment_id)?;
                if let Some(text_ref) = value.text_ref.as_deref() {
                    validate_identifier_max("text_ref", text_ref, 256)?;
                }
                Ok(())
            }
            Self::Backchannel(value) => validate_unit_interval(value.intensity),
            Self::AttentionFocus(value) => {
                validate_identifier_max("attention target", &value.target, 256)?;
                if let Some(strength) = value.strength {
                    validate_unit_interval(strength)?;
                }
                Ok(())
            }
            Self::ExpressionIntent(value) => validate_unit_interval(value.intensity),
            Self::Uncertainty(value) => {
                validate_identifier_max("uncertainty subject", &value.subject, 256)?;
                validate_unit_interval(value.level)
            }
            Self::EvidenceStatus(value) => {
                validate_identifier_max("evidence subject", &value.subject_id, 256)?;
                if let Some(receipt_ref) = value.verifier_receipt_ref.as_deref() {
                    validate_identifier_max("verifier_receipt_ref", receipt_ref, 256)?;
                }
                if value.status == EvidenceState::Supported {
                    if value.verifier_receipt_ref.as_deref().map_or(true, str::is_empty)
                        || evidence_refs.is_empty()
                    {
                        return Err(EventValidationError::MissingEvidenceReceipt);
                    }
                }
                Ok(())
            }
            Self::UserControl(value) => {
                match value.action {
                    UserControlAction::Pause => Ok(()),
                    UserControlAction::Cancel => value
                        .target_event_id
                        .as_deref()
                        .ok_or(EventValidationError::MissingCancelTarget)
                        .and_then(|id| validate_identifier("target_event_id", id)),
                    UserControlAction::ChooseForm => value
                        .form_id
                        .as_deref()
                        .ok_or(EventValidationError::MissingFormId)
                        .and_then(|id| validate_identifier("form_id", id)),
                    UserControlAction::SetReducedMotion => value
                        .enabled
                        .ok_or(EventValidationError::MissingReducedMotionValue)
                        .map(|_| ()),
                    UserControlAction::SetIntensity => value
                        .intensity
                        .ok_or(EventValidationError::MissingIntensity)
                        .and_then(validate_unit_interval),
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventSource {
    pub component: String,
    pub instance_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_event_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_id: Option<String>,
}

/// A single semantic event emitted by Symthaea or an explicitly identified
/// interaction boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbodimentEvent {
    pub schema: String,
    pub session_id: String,
    pub event_id: String,
    pub sequence: u64,
    /// Decimal string in monotonic nanoseconds; string encoding avoids JSON
    /// consumer precision loss for 64-bit clocks.
    pub observed_at_monotonic_ns: String,
    pub expires_at_monotonic_ns: String,
    pub event_kind: EventKind,
    pub payload: EventPayload,
    pub source: EventSource,
    pub epistemic_status: EpistemicStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_refs: Vec<String>,
    pub privacy_class: PrivacyClass,
}

impl EmbodimentEvent {
    /// Validate the event in isolation. Stream order and current freshness are
    /// checked by EmbodimentStreamValidator.
    pub fn validate(&self) -> Result<(), EventValidationError> {
        if self.schema != EMBODIMENT_EVENT_SCHEMA_V1 {
            return Err(EventValidationError::WrongSchema);
        }
        validate_identifier("session_id", &self.session_id)?;
        validate_identifier("event_id", &self.event_id)?;
        validate_identifier("source.component", &self.source.component)?;
        validate_identifier("source.instance_id", &self.source.instance_id)?;
        if let Some(id) = self.source.source_event_id.as_deref() {
            validate_identifier("source.source_event_id", id)?;
        }
        if let Some(id) = self.source.build_id.as_deref() {
            validate_identifier("source.build_id", id)?;
        }
        if self.sequence > MAX_SAFE_SEQUENCE {
            return Err(EventValidationError::SequenceBeyondSafeRange);
        }

        let observed = parse_monotonic_ns(&self.observed_at_monotonic_ns)?;
        let expires = parse_monotonic_ns(&self.expires_at_monotonic_ns)?;
        if expires <= observed {
            return Err(EventValidationError::ExpiryNotAfterObservation);
        }

        if self.event_kind != self.payload.kind() {
            return Err(EventValidationError::KindPayloadMismatch);
        }

        match self.epistemic_status {
            EpistemicStatus::Inferred if self.confidence.is_none() => {
                return Err(EventValidationError::MissingInferenceConfidence);
            }
            _ => {}
        }
        if let Some(confidence) = self.confidence {
            validate_unit_interval_f64(confidence)?;
        }

        if self.evidence_refs.len() > 64 {
            return Err(EventValidationError::TooManyEvidenceReferences);
        }
        for reference in &self.evidence_refs {
            validate_identifier_max("evidence_ref", reference, 256)?;
        }

        self.payload.validate(&self.evidence_refs)?;
        Ok(())
    }

    pub fn observed_at_ns(&self) -> Result<u64, EventValidationError> {
        parse_monotonic_ns(&self.observed_at_monotonic_ns)
    }

    pub fn expires_at_ns(&self) -> Result<u64, EventValidationError> {
        parse_monotonic_ns(&self.expires_at_monotonic_ns)
    }
}

/// Stateful validator for one event stream. It rejects gaps rather than
/// silently interpreting missing events as neutral behavior or WAIT.
#[derive(Debug, Clone)]
pub struct EmbodimentStreamValidator {
    session_id: String,
    last_sequence: Option<u64>,
    last_observed_at_ns: Option<u64>,
    seen_event_ids: HashSet<String>,
}

impl EmbodimentStreamValidator {
    pub fn new(session_id: impl Into<String>) -> Result<Self, EventValidationError> {
        let session_id = session_id.into();
        validate_identifier("session_id", &session_id)?;
        Ok(Self {
            session_id,
            last_sequence: None,
            last_observed_at_ns: None,
            seen_event_ids: HashSet::new(),
        })
    }

    /// Accept an event against the caller-supplied monotonic clock reading.
    /// This function performs no I/O and does not render or execute actions.
    pub fn accept(
        &mut self,
        event: &EmbodimentEvent,
        now_monotonic_ns: &str,
    ) -> Result<(), EventValidationError> {
        event.validate()?;
        let now = parse_monotonic_ns(now_monotonic_ns)?;
        let observed = event.observed_at_ns()?;
        let expires = event.expires_at_ns()?;

        if event.session_id != self.session_id {
            return Err(EventValidationError::SessionMismatch);
        }
        if observed > now {
            return Err(EventValidationError::EventObservedInFuture);
        }
        if expires <= now {
            return Err(EventValidationError::EventExpired);
        }
        if self.seen_event_ids.contains(&event.event_id) {
            return Err(EventValidationError::DuplicateEventId);
        }
        if let EventPayload::UserControl(UserControlPayload {
            action: UserControlAction::Cancel,
            target_event_id: Some(target_event_id),
            ..
        }) = &event.payload
        {
            if !self.seen_event_ids.contains(target_event_id) {
                return Err(EventValidationError::UnknownCancelTarget);
            }
        }

        match self.last_sequence {
            None if event.sequence != 0 => {
                return Err(EventValidationError::SequenceMustStartAtZero);
            }
            Some(previous) if event.sequence <= previous => {
                return Err(EventValidationError::SequenceRegression);
            }
            Some(previous) if event.sequence != previous + 1 => {
                return Err(EventValidationError::SequenceGap);
            }
            _ => {}
        }

        if let Some(previous_time) = self.last_observed_at_ns {
            if observed < previous_time {
                return Err(EventValidationError::ObservedClockRegression);
            }
        }

        // Mutate stream state only after every validation has succeeded.
        self.seen_event_ids.insert(event.event_id.clone());
        self.last_sequence = Some(event.sequence);
        self.last_observed_at_ns = Some(observed);
        Ok(())
    }

    pub fn last_sequence(&self) -> Option<u64> {
        self.last_sequence
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventValidationError {
    WrongSchema,
    InvalidIdentifier(&'static str),
    SequenceBeyondSafeRange,
    InvalidMonotonicTimestamp(&'static str),
    ExpiryNotAfterObservation,
    KindPayloadMismatch,
    MissingInferenceConfidence,
    InvalidUnitInterval,
    MissingEvidenceReceipt,
    TooManyEvidenceReferences,
    MissingCancelTarget,
    UnknownCancelTarget,
    MissingFormId,
    MissingReducedMotionValue,
    MissingIntensity,
    SessionMismatch,
    DuplicateEventId,
    SequenceMustStartAtZero,
    SequenceRegression,
    SequenceGap,
    ObservedClockRegression,
    EventExpired,
    EventObservedInFuture,
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), EventValidationError> {
    validate_identifier_max(field, value, 128)
}

fn validate_identifier_max(
    field: &'static str,
    value: &str,
    max_len: usize,
) -> Result<(), EventValidationError> {
    if value.is_empty()
        || value.len() > max_len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(EventValidationError::InvalidIdentifier(field));
    }
    Ok(())
}

fn parse_monotonic_ns(value: &str) -> Result<u64, EventValidationError> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(EventValidationError::InvalidMonotonicTimestamp(
            "must be an unsigned decimal string without leading zeroes",
        ));
    }
    u64::from_str(value).map_err(|_| {
        EventValidationError::InvalidMonotonicTimestamp("value exceeds u64 nanosecond range")
    })
}

fn validate_unit_interval(value: f32) -> Result<(), EventValidationError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(EventValidationError::InvalidUnitInterval);
    }
    Ok(())
}

fn validate_unit_interval_f64(value: f64) -> Result<(), EventValidationError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(EventValidationError::InvalidUnitInterval);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listening_event(sequence: u64, event_id: &str) -> EmbodimentEvent {
        EmbodimentEvent {
            schema: EMBODIMENT_EVENT_SCHEMA_V1.to_owned(),
            session_id: "session-1".to_owned(),
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

    #[test]
    fn accepts_first_well_formed_event() {
        let mut validator = EmbodimentStreamValidator::new("session-1").unwrap();
        assert_eq!(validator.accept(&listening_event(0, "event-0"), "200"), Ok(()));
        assert_eq!(validator.last_sequence(), Some(0));
    }

    #[test]
    fn rejects_sequence_gap_without_mutating_state() {
        let mut validator = EmbodimentStreamValidator::new("session-1").unwrap();
        validator.accept(&listening_event(0, "event-0"), "200").unwrap();
        assert_eq!(
            validator.accept(&listening_event(2, "event-2"), "300"),
            Err(EventValidationError::SequenceGap)
        );
        assert_eq!(validator.last_sequence(), Some(0));
    }

    #[test]
    fn rejects_duplicate_event_identity() {
        let mut validator = EmbodimentStreamValidator::new("session-1").unwrap();
        validator.accept(&listening_event(0, "event-0"), "200").unwrap();
        assert_eq!(
            validator.accept(&listening_event(1, "event-0"), "300"),
            Err(EventValidationError::DuplicateEventId)
        );
        assert_eq!(validator.last_sequence(), Some(0));
    }

    #[test]
    fn rejects_wrong_session() {
        let mut validator = EmbodimentStreamValidator::new("session-2").unwrap();
        assert_eq!(
            validator.accept(&listening_event(0, "event-0"), "200"),
            Err(EventValidationError::SessionMismatch)
        );
    }

    #[test]
    fn rejects_expired_event() {
        let mut validator = EmbodimentStreamValidator::new("session-1").unwrap();
        assert_eq!(
            validator.accept(&listening_event(0, "event-0"), "1000"),
            Err(EventValidationError::EventExpired)
        );
    }

    #[test]
    fn rejects_future_observation() {
        let mut validator = EmbodimentStreamValidator::new("session-1").unwrap();
        assert_eq!(
            validator.accept(&listening_event(0, "event-0"), "99"),
            Err(EventValidationError::EventObservedInFuture)
        );
    }

    #[test]
    fn rejects_event_kind_payload_mismatch() {
        let mut event = listening_event(0, "event-0");
        event.event_kind = EventKind::ConversationTurnState;
        assert_eq!(event.validate(), Err(EventValidationError::KindPayloadMismatch));
    }

    #[test]
    fn inferred_event_requires_confidence() {
        let mut event = listening_event(0, "event-0");
        event.epistemic_status = EpistemicStatus::Inferred;
        assert_eq!(event.validate(), Err(EventValidationError::MissingInferenceConfidence));
    }

    #[test]
    fn inferred_confidence_must_be_finite_and_bounded() {
        let mut event = listening_event(0, "event-0");
        event.epistemic_status = EpistemicStatus::Inferred;
        event.confidence = Some(f64::NAN);
        assert_eq!(event.validate(), Err(EventValidationError::InvalidUnitInterval));
    }

    #[test]
    fn expiry_must_follow_observation() {
        let mut event = listening_event(0, "event-0");
        event.expires_at_monotonic_ns = event.observed_at_monotonic_ns.clone();
        assert_eq!(event.validate(), Err(EventValidationError::ExpiryNotAfterObservation));
    }

    #[test]
    fn supported_evidence_requires_a_receipt_and_reference() {
        let mut event = listening_event(0, "event-0");
        event.event_kind = EventKind::EvidenceStatus;
        event.payload = EventPayload::EvidenceStatus(EvidenceStatusPayload {
            subject_id: "claim-1".to_owned(),
            status: EvidenceState::Supported,
            verifier_receipt_ref: None,
        });
        assert_eq!(event.validate(), Err(EventValidationError::MissingEvidenceReceipt));
    }

    #[test]
    fn cancel_control_requires_target_identity() {
        let mut event = listening_event(0, "event-0");
        event.event_kind = EventKind::UserControl;
        event.payload = EventPayload::UserControl(UserControlPayload {
            action: UserControlAction::Cancel,
            form_id: None,
            target_event_id: None,
            enabled: None,
            intensity: None,
        });
        assert_eq!(event.validate(), Err(EventValidationError::MissingCancelTarget));
    }

    #[test]
    fn cancel_control_must_reference_an_event_in_the_same_stream() {
        let mut validator = EmbodimentStreamValidator::new("session-1").unwrap();
        validator.accept(&listening_event(0, "event-0"), "200").unwrap();
        let mut cancel = listening_event(1, "event-1");
        cancel.event_kind = EventKind::UserControl;
        cancel.payload = EventPayload::UserControl(UserControlPayload {
            action: UserControlAction::Cancel,
            form_id: None,
            target_event_id: Some("missing-event".to_owned()),
            enabled: None,
            intensity: None,
        });
        assert_eq!(
            validator.accept(&cancel, "300"),
            Err(EventValidationError::UnknownCancelTarget)
        );
        assert_eq!(validator.last_sequence(), Some(0));
    }

    #[test]
    fn supported_evidence_receipt_must_be_a_valid_reference() {
        let mut event = listening_event(0, "event-0");
        event.event_kind = EventKind::EvidenceStatus;
        event.payload = EventPayload::EvidenceStatus(EvidenceStatusPayload {
            subject_id: "claim-1".to_owned(),
            status: EvidenceState::Supported,
            verifier_receipt_ref: Some("invalid receipt".to_owned()),
        });
        event.evidence_refs = vec!["sha256:abc123".to_owned()];
        assert_eq!(
            event.validate(),
            Err(EventValidationError::InvalidIdentifier("verifier_receipt_ref"))
        );
    }

    #[test]
    fn event_json_roundtrips_with_versioned_wire_names() {
        let event = listening_event(0, "event-0");
        let encoded = serde_json::to_string(&event).unwrap();
        assert!(encoded.contains("\"conversation.listening\""));
        assert!(encoded.contains("\"observed_at_monotonic_ns\":\"100\""));
        let decoded: EmbodimentEvent = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, event);
    }

    #[test]
    fn monotonic_clock_requires_canonical_decimal_encoding() {
        assert_eq!(
            parse_monotonic_ns("001"),
            Err(EventValidationError::InvalidMonotonicTimestamp(
                "must be an unsigned decimal string without leading zeroes"
            ))
        );
        assert_eq!(parse_monotonic_ns("18446744073709551616"), Err(
            EventValidationError::InvalidMonotonicTimestamp(
                "value exceeds u64 nanosecond range"
            )
        ));
    }
}
