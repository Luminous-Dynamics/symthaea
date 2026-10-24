// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Transport-neutral v1 envelope constraints.
//!
//! The planned Unix-datagram transport can use this module without letting the
//! protocol become an unbounded log transport. Every message is also bound to
//! an explicit observation lineage so a delayed datagram from an older observer
//! cannot poison sequence ordering in the current reducer.

use crate::state::BootStateReducer;
use crate::{BootEvent, BootSnapshot, PROTOCOL_VERSION, ProtocolError};
use serde::{Deserialize, Serialize};

pub const MAX_WIRE_BYTES: usize = 4096;

/// Opaque identifier for one authoritative observation lineage.
///
/// The protocol deliberately does not generate this identifier. A future Linux
/// observer may derive it from the kernel boot ID plus an observer-instance
/// nonce. Receivers compare it only for equality; it carries no authority or
/// ordering by itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObservationId([u8; 16]);

impl ObservationId {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "message", rename_all = "kebab-case")]
pub enum WireMessage {
    Event {
        version: u16,
        observation: ObservationId,
        event: BootEvent,
    },
    Snapshot {
        version: u16,
        observation: ObservationId,
        snapshot: BootSnapshot,
    },
}

impl WireMessage {
    pub fn event(observation: ObservationId, event: BootEvent) -> Self {
        Self::Event {
            version: PROTOCOL_VERSION,
            observation,
            event,
        }
    }

    pub fn snapshot(observation: ObservationId, snapshot: BootSnapshot) -> Self {
        Self::Snapshot {
            version: PROTOCOL_VERSION,
            observation,
            snapshot,
        }
    }

    pub const fn observation(&self) -> ObservationId {
        match self {
            Self::Event { observation, .. } | Self::Snapshot { observation, .. } => *observation,
        }
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        match self {
            Self::Event { version, event, .. } => {
                validate_version(*version)?;
                event.validate()
            }
            Self::Snapshot {
                version, snapshot, ..
            } => {
                validate_version(*version)?;
                snapshot.validate()
            }
        }
    }
}

/// Result of reducing one validated wire message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireApply {
    /// A current-lineage event or snapshot changed reducer state.
    Applied,
    /// Sequence ordering made the message duplicate/stale within this lineage.
    IgnoredStale,
    /// The reducer has no lineage yet and requires an initial snapshot.
    AwaitingSnapshot,
    /// The message belongs to another lineage and cannot implicitly reset state.
    ForeignObservation,
}

/// Lineage-aware presentation reducer for a datagram consumer.
///
/// A foreign observation can never silently replace current state. The process
/// that owns the transport must make lineage transition explicit by calling
/// `reset_from_snapshot` with a validated snapshot from the newly authoritative
/// observer. This prevents late snapshots/events from older socket queues from
/// resetting or sequence-poisoning the current presentation.
#[derive(Debug, Default, Clone)]
pub struct WireStateReducer {
    observation: Option<ObservationId>,
    state: BootStateReducer,
}

impl WireStateReducer {
    pub const fn observation(&self) -> Option<ObservationId> {
        self.observation
    }

    pub fn snapshot(&self) -> Option<BootSnapshot> {
        self.observation.map(|_| self.state.snapshot())
    }

    pub fn apply(&mut self, message: &WireMessage) -> Result<WireApply, ProtocolError> {
        message.validate()?;

        let observation = message.observation();
        match self.observation {
            None => match message {
                WireMessage::Snapshot { snapshot, .. } => {
                    let mut state = BootStateReducer::default();
                    state.try_replace(snapshot.clone())?;
                    self.observation = Some(observation);
                    self.state = state;
                    Ok(WireApply::Applied)
                }
                WireMessage::Event { .. } => Ok(WireApply::AwaitingSnapshot),
            },
            Some(current) if current != observation => Ok(WireApply::ForeignObservation),
            Some(_) => match message {
                WireMessage::Event { event, .. } => Ok(if self.state.try_apply(event)? {
                    WireApply::Applied
                } else {
                    WireApply::IgnoredStale
                }),
                WireMessage::Snapshot { snapshot, .. } => {
                    Ok(if self.state.try_replace(snapshot.clone())? {
                        WireApply::Applied
                    } else {
                        WireApply::IgnoredStale
                    })
                }
            },
        }
    }

    /// Explicitly start a different authoritative observation lineage.
    ///
    /// Only snapshots are accepted for resets. Transport code should call this
    /// when it has independently established that a new observer/boot lineage is
    /// authoritative (for example, after recreating the root-owned socket).
    pub fn reset_from_snapshot(&mut self, message: &WireMessage) -> Result<(), ProtocolError> {
        message.validate()?;
        let WireMessage::Snapshot {
            observation,
            snapshot,
            ..
        } = message
        else {
            return Err(ProtocolError::SnapshotRequiredForObservationReset);
        };

        let mut state = BootStateReducer::default();
        state.try_replace(snapshot.clone())?;
        self.observation = Some(*observation);
        self.state = state;
        Ok(())
    }
}

/// Validate a transport datagram before deserialization.
///
/// The observer/consumer transport MUST call this on the received byte slice
/// before attempting to decode it. This gives the 4096-byte application ceiling
/// an executable contract rather than leaving it as documentation only.
pub fn validate_datagram_size(bytes: &[u8]) -> Result<(), ProtocolError> {
    if bytes.len() > MAX_WIRE_BYTES {
        return Err(ProtocolError::WireTooLarge {
            bytes: bytes.len(),
            max: MAX_WIRE_BYTES,
        });
    }
    Ok(())
}

fn validate_version(version: u16) -> Result<(), ProtocolError> {
    if version != PROTOCOL_VERSION {
        return Err(ProtocolError::UnsupportedVersion(version));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BootDomain, BootEvent, BootPhase};
    use std::time::Duration;

    const OBS_A: ObservationId = ObservationId::from_bytes([0xAA; 16]);
    const OBS_B: ObservationId = ObservationId::from_bytes([0xBB; 16]);

    fn initial_snapshot(observation: ObservationId) -> WireMessage {
        WireMessage::snapshot(
            observation,
            BootSnapshot::new(1, Duration::from_millis(100), BootPhase::Kernel),
        )
    }

    #[test]
    fn wire_event_round_trip_stays_bounded() {
        let message = WireMessage::event(
            OBS_A,
            BootEvent::DomainReady {
                sequence: 4,
                elapsed_ms: 850,
                domain: BootDomain::Storage,
            },
        );

        let bytes = serde_json::to_vec(&message).unwrap();
        validate_datagram_size(&bytes).unwrap();
        let decoded: WireMessage = serde_json::from_slice(&bytes).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn reducer_requires_snapshot_before_first_event() {
        let mut reducer = WireStateReducer::default();
        let event = WireMessage::event(
            OBS_A,
            BootEvent::DomainReady {
                sequence: 2,
                elapsed_ms: 110,
                domain: BootDomain::Kernel,
            },
        );
        assert_eq!(reducer.apply(&event).unwrap(), WireApply::AwaitingSnapshot);
        assert_eq!(reducer.observation(), None);
    }

    #[test]
    fn foreign_high_sequence_event_cannot_poison_current_lineage() {
        let mut reducer = WireStateReducer::default();
        reducer.apply(&initial_snapshot(OBS_A)).unwrap();

        let foreign = WireMessage::event(
            OBS_B,
            BootEvent::DomainReady {
                sequence: u64::MAX - 1,
                elapsed_ms: 101,
                domain: BootDomain::Storage,
            },
        );
        assert_eq!(
            reducer.apply(&foreign).unwrap(),
            WireApply::ForeignObservation
        );

        let current = WireMessage::event(
            OBS_A,
            BootEvent::DomainReady {
                sequence: 2,
                elapsed_ms: 110,
                domain: BootDomain::Kernel,
            },
        );
        assert_eq!(reducer.apply(&current).unwrap(), WireApply::Applied);
        assert_eq!(reducer.snapshot().unwrap().sequence, 2);
    }

    #[test]
    fn foreign_snapshot_requires_explicit_authority_reset() {
        let mut reducer = WireStateReducer::default();
        reducer.apply(&initial_snapshot(OBS_A)).unwrap();
        let next = WireMessage::snapshot(
            OBS_B,
            BootSnapshot::new(1, Duration::from_millis(5), BootPhase::Kernel),
        );

        assert_eq!(
            reducer.apply(&next).unwrap(),
            WireApply::ForeignObservation
        );
        assert_eq!(reducer.observation(), Some(OBS_A));

        reducer.reset_from_snapshot(&next).unwrap();
        assert_eq!(reducer.observation(), Some(OBS_B));
        assert_eq!(reducer.snapshot().unwrap().elapsed_ms, 5);
    }

    #[test]
    fn reset_rejects_event_even_if_it_has_a_new_observation() {
        let mut reducer = WireStateReducer::default();
        let event = WireMessage::event(
            OBS_B,
            BootEvent::DomainReady {
                sequence: 1,
                elapsed_ms: 1,
                domain: BootDomain::Kernel,
            },
        );
        assert_eq!(
            reducer.reset_from_snapshot(&event),
            Err(ProtocolError::SnapshotRequiredForObservationReset)
        );
    }

    #[test]
    fn oversized_datagram_is_rejected_before_decode() {
        let bytes = vec![b'x'; MAX_WIRE_BYTES + 1];
        assert_eq!(
            validate_datagram_size(&bytes),
            Err(ProtocolError::WireTooLarge {
                bytes: MAX_WIRE_BYTES + 1,
                max: MAX_WIRE_BYTES,
            })
        );
    }

    #[test]
    fn exact_wire_budget_is_accepted() {
        let bytes = vec![0_u8; MAX_WIRE_BYTES];
        validate_datagram_size(&bytes).unwrap();
    }

    #[test]
    fn envelope_rejects_unknown_version() {
        let message = WireMessage::Event {
            version: PROTOCOL_VERSION + 1,
            observation: OBS_A,
            event: BootEvent::DomainReady {
                sequence: 1,
                elapsed_ms: 1,
                domain: BootDomain::Kernel,
            },
        };
        assert_eq!(
            message.validate(),
            Err(ProtocolError::UnsupportedVersion(PROTOCOL_VERSION + 1))
        );
    }
}
