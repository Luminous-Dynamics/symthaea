// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport-neutral interface types for a running Symthaea.
//!
//! This crate intentionally contains no runtime, transport, cognitive, UI, audio,
//! graphics, or action-execution implementation. It defines only the small semantic
//! vocabulary shared by independently replaceable interfaces.
//!
//! Governing invariants:
//!
//! ```text
//! transport encoding != semantic authority
//! presentation       != cognitive state authority
//! cognitive telemetry != execution authority
//! ```
//!
//! Ordered semantic events use a runtime-local monotonic [`EventSeq`]. Wall-clock
//! timestamps may be carried by higher layers as metadata, but are not the event
//! ordering primitive defined here.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// V1 interface semantic protocol version.
pub const INTERFACE_PROTOCOL_VERSION: u16 = 1;

/// Maximum UTF-8 bytes allowed in an incremental text event.
///
/// Bulk artifacts and observations belong on their own planes and should be
/// referenced from semantic events rather than embedded without bound.
pub const MAX_EVENT_TEXT_BYTES: usize = 64 * 1024;

/// Maximum bytes in an opaque interface identity token.
pub const MAX_INTERFACE_ID_BYTES: usize = 128;

/// Validation failure for an interface identity token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdError {
    Empty,
    TooLong { len: usize, max: usize },
    InvalidByte { index: usize, byte: u8 },
}

impl fmt::Display for IdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "interface identity must not be empty"),
            Self::TooLong { len, max } => {
                write!(f, "interface identity is {len} bytes; maximum is {max}")
            }
            Self::InvalidByte { index, byte } => write!(
                f,
                "interface identity contains invalid byte 0x{byte:02x} at index {index}"
            ),
        }
    }
}

impl std::error::Error for IdError {}

fn validate_interface_id(value: &str) -> Result<(), IdError> {
    if value.is_empty() {
        return Err(IdError::Empty);
    }
    if value.len() > MAX_INTERFACE_ID_BYTES {
        return Err(IdError::TooLong {
            len: value.len(),
            max: MAX_INTERFACE_ID_BYTES,
        });
    }
    for (index, byte) in value.bytes().enumerate() {
        if !(byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')) {
            return Err(IdError::InvalidByte { index, byte });
        }
    }
    Ok(())
}

macro_rules! interface_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, IdError> {
                let value = value.into();
                validate_interface_id(&value)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_inner(self) -> String {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl TryFrom<String> for $name {
            type Error = IdError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = IdError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::new(value).map_err(D::Error::custom)
            }
        }
    };
}

interface_id!(
    /// Identity of one running Symthaea runtime instance.
    RuntimeId
);
interface_id!(
    /// Identity of one interface client.
    ClientId
);
interface_id!(
    /// Identity of one client session attached to a runtime.
    SessionId
);
interface_id!(
    /// Identity of one conversational or task turn.
    TurnId
);
interface_id!(
    /// Identity shared by the acoustic and semantic representations of one utterance.
    UtteranceId
);
interface_id!(
    /// Opaque reference to a canonical modality-neutral observation.
    ObservationId
);
interface_id!(
    /// Identity of an action proposal. This does not itself grant action authority.
    ActionProposalId
);
interface_id!(
    /// Identity of an authority decision about an action proposal.
    ActionDecisionId
);
interface_id!(
    /// Identity of an observed action result.
    ActionResultId
);
interface_id!(
    /// Identity of an artifact produced or referenced by the runtime.
    ArtifactId
);
interface_id!(
    /// Stable machine-readable error code for a semantic runtime event.
    ErrorCode
);

/// Non-zero runtime-local semantic event sequence number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct EventSeq(u64);

impl EventSeq {
    /// Construct a sequence number. Zero is reserved to mean "no event observed".
    pub fn new(value: u64) -> Option<Self> {
        (value != 0).then_some(Self(value))
    }

    pub fn get(self) -> u64 {
        self.0
    }

    pub fn checked_next(self) -> Option<Self> {
        self.0.checked_add(1).and_then(Self::new)
    }
}

impl fmt::Display for EventSeq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<'de> Deserialize<'de> for EventSeq {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| D::Error::custom("event sequence must be non-zero"))
    }
}

/// Exact semantic position in one runtime's ordered event stream.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuntimeCursor {
    runtime_id: RuntimeId,
    seq: EventSeq,
}

impl RuntimeCursor {
    pub fn new(runtime_id: RuntimeId, seq: EventSeq) -> Self {
        Self { runtime_id, seq }
    }

    pub fn runtime_id(&self) -> &RuntimeId {
        &self.runtime_id
    }

    pub fn seq(&self) -> EventSeq {
        self.seq
    }

    /// Whether this cursor is the immediate semantic successor of `previous`.
    pub fn is_immediate_successor_of(&self, previous: &Self) -> bool {
        self.runtime_id == previous.runtime_id && previous.seq.checked_next() == Some(self.seq)
    }
}

/// Provenance/freshness classification for externally rendered runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateProvenance {
    Live,
    Stale,
    Replay,
    Simulation,
    Unknown,
}

/// A runtime state value whose provenance is structurally explicit.
///
/// `Unknown` carries no value, so a disconnected client cannot represent
/// plausible placeholder metrics as merely "unknown live state" by accident.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "provenance", rename_all = "snake_case")]
pub enum RuntimeState<T> {
    Live {
        cursor: RuntimeCursor,
        value: T,
    },
    Stale {
        cursor: RuntimeCursor,
        value: T,
    },
    Replay {
        cursor: RuntimeCursor,
        value: T,
    },
    Simulation {
        cursor: Option<RuntimeCursor>,
        value: T,
    },
    Unknown,
}

impl<T> RuntimeState<T> {
    pub fn provenance(&self) -> StateProvenance {
        match self {
            Self::Live { .. } => StateProvenance::Live,
            Self::Stale { .. } => StateProvenance::Stale,
            Self::Replay { .. } => StateProvenance::Replay,
            Self::Simulation { .. } => StateProvenance::Simulation,
            Self::Unknown => StateProvenance::Unknown,
        }
    }

    pub fn cursor(&self) -> Option<&RuntimeCursor> {
        match self {
            Self::Live { cursor, .. }
            | Self::Stale { cursor, .. }
            | Self::Replay { cursor, .. } => Some(cursor),
            Self::Simulation { cursor, .. } => cursor.as_ref(),
            Self::Unknown => None,
        }
    }

    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Live { value, .. }
            | Self::Stale { value, .. }
            | Self::Replay { value, .. }
            | Self::Simulation { value, .. } => Some(value),
            Self::Unknown => None,
        }
    }

    pub fn is_live(&self) -> bool {
        matches!(self, Self::Live { .. })
    }
}

/// Logical runtime plane. Transport implementations may encode each plane differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimePlane {
    State,
    SemanticEvents,
    Sensory,
    EvidenceAudit,
}

/// Capability advertised by an interface client or accepted by a runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterfaceCapability {
    StateStream,
    SemanticEvents,
    TextInput,
    AudioInput,
    VisionInput,
    VoiceOutput,
    ActionProposals,
    Artifacts,
    ResearchTelemetry,
}

/// Validation failure for the semantic interface protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    UnsupportedVersion { found: u16, supported: u16 },
    DuplicateCapability(InterfaceCapability),
    EmptyEventText,
    EventTextTooLong { len: usize, max: usize },
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion { found, supported } => write!(
                f,
                "unsupported interface protocol version {found}; supported version is {supported}"
            ),
            Self::DuplicateCapability(capability) => {
                write!(f, "duplicate interface capability: {capability:?}")
            }
            Self::EmptyEventText => write!(f, "incremental event text must not be empty"),
            Self::EventTextTooLong { len, max } => {
                write!(f, "incremental event text is {len} bytes; maximum is {max}")
            }
        }
    }
}

impl std::error::Error for ProtocolError {}

fn canonicalize_capabilities(
    mut capabilities: Vec<InterfaceCapability>,
) -> Result<Vec<InterfaceCapability>, ProtocolError> {
    let mut seen = BTreeSet::new();
    for capability in &capabilities {
        if !seen.insert(*capability) {
            return Err(ProtocolError::DuplicateCapability(*capability));
        }
    }
    capabilities.sort_unstable();
    Ok(capabilities)
}

/// Client side of the transport-neutral capability handshake.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClientHello {
    protocol_version: u16,
    client_id: ClientId,
    capabilities: Vec<InterfaceCapability>,
}

impl ClientHello {
    pub fn new(
        client_id: ClientId,
        capabilities: Vec<InterfaceCapability>,
    ) -> Result<Self, ProtocolError> {
        Ok(Self {
            protocol_version: INTERFACE_PROTOCOL_VERSION,
            client_id,
            capabilities: canonicalize_capabilities(capabilities)?,
        })
    }

    pub fn protocol_version(&self) -> u16 {
        self.protocol_version
    }

    pub fn client_id(&self) -> &ClientId {
        &self.client_id
    }

    pub fn capabilities(&self) -> &[InterfaceCapability] {
        &self.capabilities
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClientHelloWire {
    protocol_version: u16,
    client_id: ClientId,
    capabilities: Vec<InterfaceCapability>,
}

impl<'de> Deserialize<'de> for ClientHello {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ClientHelloWire::deserialize(deserializer)?;
        if wire.protocol_version != INTERFACE_PROTOCOL_VERSION {
            return Err(D::Error::custom(ProtocolError::UnsupportedVersion {
                found: wire.protocol_version,
                supported: INTERFACE_PROTOCOL_VERSION,
            }));
        }
        Self::new(wire.client_id, wire.capabilities).map_err(D::Error::custom)
    }
}

/// Runtime side of the transport-neutral capability handshake.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeHello {
    protocol_version: u16,
    runtime_id: RuntimeId,
    session_id: SessionId,
    accepted_capabilities: Vec<InterfaceCapability>,
}

impl RuntimeHello {
    pub fn new(
        runtime_id: RuntimeId,
        session_id: SessionId,
        accepted_capabilities: Vec<InterfaceCapability>,
    ) -> Result<Self, ProtocolError> {
        Ok(Self {
            protocol_version: INTERFACE_PROTOCOL_VERSION,
            runtime_id,
            session_id,
            accepted_capabilities: canonicalize_capabilities(accepted_capabilities)?,
        })
    }

    pub fn protocol_version(&self) -> u16 {
        self.protocol_version
    }

    pub fn runtime_id(&self) -> &RuntimeId {
        &self.runtime_id
    }

    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    pub fn accepted_capabilities(&self) -> &[InterfaceCapability] {
        &self.accepted_capabilities
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeHelloWire {
    protocol_version: u16,
    runtime_id: RuntimeId,
    session_id: SessionId,
    accepted_capabilities: Vec<InterfaceCapability>,
}

impl<'de> Deserialize<'de> for RuntimeHello {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RuntimeHelloWire::deserialize(deserializer)?;
        if wire.protocol_version != INTERFACE_PROTOCOL_VERSION {
            return Err(D::Error::custom(ProtocolError::UnsupportedVersion {
                found: wire.protocol_version,
                supported: INTERFACE_PROTOCOL_VERSION,
            }));
        }
        Self::new(
            wire.runtime_id,
            wire.session_id,
            wire.accepted_capabilities,
        )
        .map_err(D::Error::custom)
    }
}

/// Descriptive result of the actual action-authority layer.
///
/// This enum mirrors a decision for interfaces. Constructing or deserializing it
/// grants no capability and authorizes no effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionDisposition {
    Authorized,
    Denied,
    NeedsConfirmation,
    Cancelled,
}

/// Compact semantic events shared by interface transports.
///
/// Large sensory payloads, evidence records, and artifacts are referenced by
/// identity instead of being embedded into this stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeEventKind {
    ObservationAccepted {
        observation_id: ObservationId,
    },
    UtterancePartial {
        utterance_id: UtteranceId,
        text: String,
    },
    UtteranceFinal {
        utterance_id: UtteranceId,
        observation_id: ObservationId,
    },
    ResponseStarted {
        turn_id: TurnId,
    },
    ResponseDelta {
        turn_id: TurnId,
        text: String,
    },
    ResponseFinished {
        turn_id: TurnId,
    },
    ActionProposed {
        proposal_id: ActionProposalId,
    },
    ActionDecision {
        proposal_id: ActionProposalId,
        decision_id: ActionDecisionId,
        disposition: ActionDisposition,
    },
    ActionResult {
        proposal_id: ActionProposalId,
        result_id: ActionResultId,
        success: bool,
    },
    ArtifactProduced {
        artifact_id: ArtifactId,
    },
    VoiceInterrupted {
        turn_id: TurnId,
    },
    Error {
        code: ErrorCode,
    },
}

impl RuntimeEventKind {
    fn validate(&self) -> Result<(), ProtocolError> {
        let text = match self {
            Self::UtterancePartial { text, .. } | Self::ResponseDelta { text, .. } => Some(text),
            _ => None,
        };
        if let Some(text) = text {
            if text.is_empty() {
                return Err(ProtocolError::EmptyEventText);
            }
            if text.len() > MAX_EVENT_TEXT_BYTES {
                return Err(ProtocolError::EventTextTooLong {
                    len: text.len(),
                    max: MAX_EVENT_TEXT_BYTES,
                });
            }
        }
        Ok(())
    }
}

/// One ordered semantic event emitted by one runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeEvent {
    protocol_version: u16,
    cursor: RuntimeCursor,
    session_id: Option<SessionId>,
    kind: RuntimeEventKind,
}

impl RuntimeEvent {
    pub fn new(
        cursor: RuntimeCursor,
        session_id: Option<SessionId>,
        kind: RuntimeEventKind,
    ) -> Result<Self, ProtocolError> {
        kind.validate()?;
        Ok(Self {
            protocol_version: INTERFACE_PROTOCOL_VERSION,
            cursor,
            session_id,
            kind,
        })
    }

    pub fn protocol_version(&self) -> u16 {
        self.protocol_version
    }

    pub fn cursor(&self) -> &RuntimeCursor {
        &self.cursor
    }

    pub fn session_id(&self) -> Option<&SessionId> {
        self.session_id.as_ref()
    }

    pub fn kind(&self) -> &RuntimeEventKind {
        &self.kind
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeEventWire {
    protocol_version: u16,
    cursor: RuntimeCursor,
    session_id: Option<SessionId>,
    kind: RuntimeEventKind,
}

impl<'de> Deserialize<'de> for RuntimeEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RuntimeEventWire::deserialize(deserializer)?;
        if wire.protocol_version != INTERFACE_PROTOCOL_VERSION {
            return Err(D::Error::custom(ProtocolError::UnsupportedVersion {
                found: wire.protocol_version,
                supported: INTERFACE_PROTOCOL_VERSION,
            }));
        }
        Self::new(wire.cursor, wire.session_id, wire.kind).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn runtime_id() -> RuntimeId {
        RuntimeId::new("runtime-1").unwrap()
    }

    fn session_id() -> SessionId {
        SessionId::new("session-1").unwrap()
    }

    fn cursor(seq: u64) -> RuntimeCursor {
        RuntimeCursor::new(runtime_id(), EventSeq::new(seq).unwrap())
    }

    #[test]
    fn interface_ids_are_bounded_and_wire_validated() {
        assert_eq!(RuntimeId::new(""), Err(IdError::Empty));
        assert!(RuntimeId::new("has whitespace").is_err());
        assert!(RuntimeId::new("runtime:abc_123.test").is_ok());

        let invalid: Result<RuntimeId, _> = serde_json::from_str("\"bad id\"");
        assert!(invalid.is_err());
    }

    #[test]
    fn event_sequence_zero_fails_closed_and_overflow_has_no_successor() {
        assert!(EventSeq::new(0).is_none());
        let zero: Result<EventSeq, _> = serde_json::from_str("0");
        assert!(zero.is_err());
        assert_eq!(EventSeq::new(u64::MAX).unwrap().checked_next(), None);
    }

    #[test]
    fn runtime_cursor_detects_continuity_and_gaps() {
        let one = cursor(1);
        let two = cursor(2);
        let four = cursor(4);
        assert!(two.is_immediate_successor_of(&one));
        assert!(!four.is_immediate_successor_of(&two));

        let other_runtime = RuntimeCursor::new(
            RuntimeId::new("runtime-2").unwrap(),
            EventSeq::new(2).unwrap(),
        );
        assert!(!other_runtime.is_immediate_successor_of(&one));
    }

    #[test]
    fn unknown_state_cannot_carry_plausible_value() {
        let state: RuntimeState<u64> = RuntimeState::Unknown;
        assert_eq!(state.provenance(), StateProvenance::Unknown);
        assert_eq!(state.cursor(), None);
        assert_eq!(state.value(), None);
        assert!(!state.is_live());

        let encoded = serde_json::to_string(&state).unwrap();
        let decoded: RuntimeState<u64> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, RuntimeState::Unknown);
    }

    #[test]
    fn simulation_is_never_mislabeled_live() {
        let state = RuntimeState::Simulation {
            cursor: Some(cursor(7)),
            value: 0.87_f64,
        };
        assert_eq!(state.provenance(), StateProvenance::Simulation);
        assert!(!state.is_live());
        assert_eq!(state.value(), Some(&0.87));
    }

    #[test]
    fn live_state_requires_a_runtime_cursor_by_type() {
        let state = RuntimeState::Live {
            cursor: cursor(9),
            value: "focused".to_string(),
        };
        assert!(state.is_live());
        assert_eq!(state.cursor().unwrap().seq(), EventSeq::new(9).unwrap());
    }

    #[test]
    fn capability_handshake_is_canonical_and_rejects_duplicates() {
        let hello = ClientHello::new(
            ClientId::new("tui-1").unwrap(),
            vec![
                InterfaceCapability::VoiceOutput,
                InterfaceCapability::StateStream,
            ],
        )
        .unwrap();
        assert_eq!(
            hello.capabilities(),
            &[
                InterfaceCapability::StateStream,
                InterfaceCapability::VoiceOutput
            ]
        );

        let duplicate = ClientHello::new(
            ClientId::new("tui-1").unwrap(),
            vec![
                InterfaceCapability::StateStream,
                InterfaceCapability::StateStream,
            ],
        );
        assert_eq!(
            duplicate,
            Err(ProtocolError::DuplicateCapability(
                InterfaceCapability::StateStream
            ))
        );
    }

    #[test]
    fn handshake_wire_rejects_wrong_version_and_unknown_fields() {
        let wrong_version = json!({
            "protocol_version": 2,
            "client_id": "tui-1",
            "capabilities": ["state_stream"]
        });
        assert!(serde_json::from_value::<ClientHello>(wrong_version).is_err());

        let unknown_field = json!({
            "protocol_version": INTERFACE_PROTOCOL_VERSION,
            "client_id": "tui-1",
            "capabilities": ["state_stream"],
            "authority": "root"
        });
        assert!(serde_json::from_value::<ClientHello>(unknown_field).is_err());
    }

    #[test]
    fn runtime_hello_roundtrips_with_explicit_runtime_and_session_identity() {
        let hello = RuntimeHello::new(
            runtime_id(),
            session_id(),
            vec![
                InterfaceCapability::SemanticEvents,
                InterfaceCapability::StateStream,
            ],
        )
        .unwrap();
        let encoded = serde_json::to_string(&hello).unwrap();
        let decoded: RuntimeHello = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, hello);
    }

    #[test]
    fn semantic_event_roundtrips_and_carries_ordering_cursor() {
        let event = RuntimeEvent::new(
            cursor(11),
            Some(session_id()),
            RuntimeEventKind::ResponseDelta {
                turn_id: TurnId::new("turn-1").unwrap(),
                text: "hello".to_string(),
            },
        )
        .unwrap();
        let encoded = serde_json::to_string(&event).unwrap();
        let decoded: RuntimeEvent = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, event);
        assert_eq!(decoded.cursor().seq(), EventSeq::new(11).unwrap());
    }

    #[test]
    fn semantic_event_wire_rejects_wrong_version() {
        let event = RuntimeEvent::new(
            cursor(12),
            None,
            RuntimeEventKind::ResponseFinished {
                turn_id: TurnId::new("turn-1").unwrap(),
            },
        )
        .unwrap();
        let mut value: Value = serde_json::to_value(event).unwrap();
        value["protocol_version"] = json!(999);
        assert!(serde_json::from_value::<RuntimeEvent>(value).is_err());
    }

    #[test]
    fn incremental_text_events_are_bounded() {
        let empty = RuntimeEvent::new(
            cursor(13),
            None,
            RuntimeEventKind::ResponseDelta {
                turn_id: TurnId::new("turn-1").unwrap(),
                text: String::new(),
            },
        );
        assert_eq!(empty, Err(ProtocolError::EmptyEventText));

        let oversized = "x".repeat(MAX_EVENT_TEXT_BYTES + 1);
        let too_large = RuntimeEvent::new(
            cursor(14),
            None,
            RuntimeEventKind::UtterancePartial {
                utterance_id: UtteranceId::new("utterance-1").unwrap(),
                text: oversized,
            },
        );
        assert!(matches!(
            too_large,
            Err(ProtocolError::EventTextTooLong { .. })
        ));
    }

    #[test]
    fn action_decision_event_is_descriptive_and_identity_bound() {
        let event = RuntimeEvent::new(
            cursor(15),
            Some(session_id()),
            RuntimeEventKind::ActionDecision {
                proposal_id: ActionProposalId::new("proposal-7").unwrap(),
                decision_id: ActionDecisionId::new("decision-9").unwrap(),
                disposition: ActionDisposition::NeedsConfirmation,
            },
        )
        .unwrap();

        match event.kind() {
            RuntimeEventKind::ActionDecision {
                proposal_id,
                decision_id,
                disposition,
            } => {
                assert_eq!(proposal_id.as_str(), "proposal-7");
                assert_eq!(decision_id.as_str(), "decision-9");
                assert_eq!(*disposition, ActionDisposition::NeedsConfirmation);
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
}
