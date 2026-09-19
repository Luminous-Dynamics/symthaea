// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Owner-side semantic event sequencing authority for the service runtime.
//!
//! The bounded event plane validates ordered retention, but it deliberately does
//! not decide who may mint semantic cursors. This crate supplies that missing
//! capability boundary: one shared sequencer owns `RuntimeId + next EventSeq`.
//! The general emitter remains non-cloneable, while voice receives only a narrow
//! cloneable capability that can publish `VoiceInterrupted`.

use std::fmt;
use std::sync::{Arc, Mutex};

use symthaea_interface_events::{
    EventPlaneError, SemanticEventPublisher, SemanticEventSubscriber, SubscribeFrom,
    semantic_event_channel,
};
use symthaea_interface_types::{
    EventSeq, IdError, ProtocolError, RuntimeCursor, RuntimeEvent, RuntimeEventKind, RuntimeId,
    SessionId, TurnId,
};
use symthaea_runtime_owner::OwnerCommandContext;

/// Failure while constructing or publishing an owner-authored semantic event.
#[derive(Debug)]
pub enum ServiceEventError {
    Plane(EventPlaneError),
    Protocol(ProtocolError),
    Identity(IdError),
    SequenceExhausted,
    SequencerPoisoned,
}

impl fmt::Display for ServiceEventError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plane(error) => write!(f, "semantic event plane failed: {error}"),
            Self::Protocol(error) => write!(f, "semantic event protocol rejected event: {error}"),
            Self::Identity(error) => write!(f, "semantic event identity is invalid: {error}"),
            Self::SequenceExhausted => write!(f, "semantic runtime event sequence exhausted"),
            Self::SequencerPoisoned => write!(f, "semantic event sequencer is poisoned"),
        }
    }
}

impl std::error::Error for ServiceEventError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Plane(error) => Some(error),
            Self::Protocol(error) => Some(error),
            Self::Identity(error) => Some(error),
            Self::SequenceExhausted | Self::SequencerPoisoned => None,
        }
    }
}

impl From<EventPlaneError> for ServiceEventError {
    fn from(value: EventPlaneError) -> Self {
        Self::Plane(value)
    }
}

impl From<ProtocolError> for ServiceEventError {
    fn from(value: ProtocolError) -> Self {
        Self::Protocol(value)
    }
}

impl From<IdError> for ServiceEventError {
    fn from(value: IdError) -> Self {
        Self::Identity(value)
    }
}

#[derive(Debug)]
struct EventSequencer {
    next_seq: Option<EventSeq>,
    publisher: SemanticEventPublisher,
}

type SharedEventSequencer = Arc<Mutex<EventSequencer>>;

fn emit_shared(
    runtime_id: &RuntimeId,
    shared: &SharedEventSequencer,
    session_id: Option<SessionId>,
    kind: RuntimeEventKind,
) -> Result<RuntimeCursor, ServiceEventError> {
    let mut sequencer = shared
        .lock()
        .map_err(|_| ServiceEventError::SequencerPoisoned)?;
    let seq = sequencer
        .next_seq
        .ok_or(ServiceEventError::SequenceExhausted)?;
    let cursor = RuntimeCursor::new(runtime_id.clone(), seq);
    let event = RuntimeEvent::new(cursor.clone(), session_id, kind)?;
    sequencer.publisher.publish(event)?;
    sequencer.next_seq = seq.checked_next();
    Ok(cursor)
}

/// General semantic-event write capability for one runtime identity.
///
/// This type intentionally does not implement `Clone`. Keep it inside the runtime
/// owner. Specialized cloneable capabilities below expose only narrowly typed event
/// mutations while sharing the same contiguous sequencer.
pub struct ServiceEventEmitter {
    runtime_id: RuntimeId,
    sequencer: SharedEventSequencer,
}

impl ServiceEventEmitter {
    pub fn runtime_id(&self) -> &RuntimeId {
        &self.runtime_id
    }

    pub fn next_sequence(&self) -> Option<EventSeq> {
        self.sequencer.lock().ok().and_then(|sequencer| sequencer.next_seq)
    }

    /// Mint and publish one authoritative semantic event.
    ///
    /// Sequence advances only after the event plane accepts the event. A failed
    /// publication therefore cannot create an artificial semantic gap.
    pub fn emit(
        &mut self,
        session_id: Option<SessionId>,
        kind: RuntimeEventKind,
    ) -> Result<RuntimeCursor, ServiceEventError> {
        emit_shared(&self.runtime_id, &self.sequencer, session_id, kind)
    }

    pub fn response_started(
        &mut self,
        context: OwnerCommandContext,
        session_id: Option<SessionId>,
    ) -> Result<(TurnId, RuntimeCursor), ServiceEventError> {
        let turn_id = turn_id_for_owner_context(context)?;
        let cursor = self.emit(
            session_id,
            RuntimeEventKind::ResponseStarted {
                turn_id: turn_id.clone(),
            },
        )?;
        Ok((turn_id, cursor))
    }

    pub fn response_finished(
        &mut self,
        turn_id: TurnId,
        session_id: Option<SessionId>,
    ) -> Result<RuntimeCursor, ServiceEventError> {
        self.emit(session_id, RuntimeEventKind::ResponseFinished { turn_id })
    }
}

/// Narrow fast-path semantic authority for voice barge-in.
///
/// Cloning this handle does not grant generic event minting. Its only mutation is
/// one typed `VoiceInterrupted` event that shares the same runtime-local `EventSeq`
/// source as query lifecycle events.
#[derive(Clone)]
pub struct ServiceVoiceEventEmitter {
    runtime_id: RuntimeId,
    sequencer: SharedEventSequencer,
}

impl ServiceVoiceEventEmitter {
    pub fn runtime_id(&self) -> &RuntimeId {
        &self.runtime_id
    }

    pub fn voice_interrupted(
        &self,
        session_id: Option<SessionId>,
        turn_id: TurnId,
    ) -> Result<RuntimeCursor, ServiceEventError> {
        emit_shared(
            &self.runtime_id,
            &self.sequencer,
            session_id,
            RuntimeEventKind::VoiceInterrupted { turn_id },
        )
    }
}

/// Read-only subscription capability. The internal publisher clone is private and
/// used only to construct independent retained-ring subscribers.
#[derive(Clone)]
pub struct ServiceEventHub {
    publisher: SemanticEventPublisher,
}

impl ServiceEventHub {
    pub fn subscribe(
        &self,
        start: SubscribeFrom,
    ) -> Result<SemanticEventSubscriber, EventPlaneError> {
        self.publisher.subscribe(start)
    }

    pub fn newest_cursor(&self) -> Result<Option<RuntimeCursor>, EventPlaneError> {
        self.publisher.newest_cursor()
    }
}

/// Construct one semantic runtime lineage with a narrow voice-control authority.
/// Runtime identity is supplied explicitly; this layer never invents a process
/// identity from wall-clock time or transport.
pub fn service_event_plane_with_voice_control(
    runtime_id: RuntimeId,
    retention_capacity: usize,
) -> Result<
    (ServiceEventEmitter, ServiceVoiceEventEmitter, ServiceEventHub),
    EventPlaneError,
> {
    let (publisher, _initial_tail) = semantic_event_channel(retention_capacity)?;
    let hub = ServiceEventHub {
        publisher: publisher.clone(),
    };
    let sequencer = Arc::new(Mutex::new(EventSequencer {
        next_seq: EventSeq::new(1),
        publisher,
    }));
    let emitter = ServiceEventEmitter {
        runtime_id: runtime_id.clone(),
        sequencer: Arc::clone(&sequencer),
    };
    let voice = ServiceVoiceEventEmitter {
        runtime_id,
        sequencer,
    };
    Ok((emitter, voice, hub))
}

/// Backward-compatible constructor for runtimes that do not need voice control.
pub fn service_event_plane(
    runtime_id: RuntimeId,
    retention_capacity: usize,
) -> Result<(ServiceEventEmitter, ServiceEventHub), EventPlaneError> {
    let (emitter, _voice, hub) =
        service_event_plane_with_voice_control(runtime_id, retention_capacity)?;
    Ok((emitter, hub))
}

/// Deterministic correlation identity for one owner-admitted command.
///
/// Owner-command sequence and semantic event sequence remain distinct types and
/// counters. This helper only derives a `TurnId` label for correlation inside one
/// `RuntimeId` lineage.
pub fn turn_id_for_owner_context(
    context: OwnerCommandContext,
) -> Result<TurnId, IdError> {
    TurnId::new(format!("turn:{}", context.sequence().get()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_interface_events::EventRead;

    fn runtime_id() -> RuntimeId {
        RuntimeId::new("runtime:test").unwrap()
    }

    #[test]
    fn emitter_mints_one_contiguous_runtime_lineage() {
        let (mut emitter, hub) = service_event_plane(runtime_id(), 8).unwrap();
        let mut subscriber = hub.subscribe(SubscribeFrom::OldestRetained).unwrap();
        let turn = TurnId::new("turn:1").unwrap();

        let first = emitter
            .emit(
                None,
                RuntimeEventKind::ResponseStarted {
                    turn_id: turn.clone(),
                },
            )
            .unwrap();
        let second = emitter.response_finished(turn, None).unwrap();

        assert_eq!(first.runtime_id(), emitter.runtime_id());
        assert_eq!(first.seq().get(), 1);
        assert_eq!(second.seq().get(), 2);
        assert_eq!(emitter.next_sequence().unwrap().get(), 3);

        let read = subscriber.read_available(8).unwrap();
        let EventRead::Events(batch) = read else {
            panic!("expected retained semantic events");
        };
        assert_eq!(batch.events.len(), 2);
        assert_eq!(batch.events[0].cursor().seq().get(), 1);
        assert_eq!(batch.events[1].cursor().seq().get(), 2);
    }

    #[test]
    fn voice_interrupt_shares_query_event_sequence_without_generic_authority() {
        let (mut emitter, voice, hub) =
            service_event_plane_with_voice_control(runtime_id(), 8).unwrap();
        let mut subscriber = hub.subscribe(SubscribeFrom::OldestRetained).unwrap();
        let turn = TurnId::new("turn:7").unwrap();
        let session = SessionId::new("voice-session:test").unwrap();

        let started = emitter
            .emit(
                Some(session.clone()),
                RuntimeEventKind::ResponseStarted {
                    turn_id: turn.clone(),
                },
            )
            .unwrap();
        let interrupted = voice
            .voice_interrupted(Some(session.clone()), turn.clone())
            .unwrap();
        let finished = emitter.response_finished(turn, Some(session)).unwrap();

        assert_eq!(started.seq().get(), 1);
        assert_eq!(interrupted.seq().get(), 2);
        assert_eq!(finished.seq().get(), 3);

        let read = subscriber.read_available(8).unwrap();
        let EventRead::Events(batch) = read else {
            panic!("expected retained semantic events");
        };
        assert!(matches!(
            batch.events[1].kind(),
            RuntimeEventKind::VoiceInterrupted { .. }
        ));
    }

    #[test]
    fn rejected_event_does_not_consume_semantic_sequence() {
        let (mut emitter, _hub) = service_event_plane(runtime_id(), 8).unwrap();
        let turn = TurnId::new("turn:1").unwrap();

        let error = emitter
            .emit(
                None,
                RuntimeEventKind::ResponseDelta {
                    turn_id: turn.clone(),
                    text: String::new(),
                },
            )
            .unwrap_err();
        assert!(matches!(error, ServiceEventError::Protocol(ProtocolError::EmptyEventText)));
        assert_eq!(emitter.next_sequence().unwrap().get(), 1);

        let cursor = emitter
            .emit(None, RuntimeEventKind::ResponseStarted { turn_id: turn })
            .unwrap();
        assert_eq!(cursor.seq().get(), 1);
        assert_eq!(emitter.next_sequence().unwrap().get(), 2);
    }

    #[test]
    fn zero_retention_fails_before_emitter_exists() {
        assert!(matches!(
            service_event_plane(runtime_id(), 0),
            Err(EventPlaneError::ZeroCapacity)
        ));
    }
}
