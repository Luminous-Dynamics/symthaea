// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Transport-neutral v1 envelope constraints.
//!
//! The planned Unix-datagram transport can use this module without letting the
//! protocol become an unbounded log transport.

use crate::{BootEvent, BootSnapshot, PROTOCOL_VERSION, ProtocolError};
use serde::{Deserialize, Serialize};

pub const MAX_WIRE_BYTES: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "message", rename_all = "kebab-case")]
pub enum WireMessage {
    Event { version: u16, event: BootEvent },
    Snapshot { version: u16, snapshot: BootSnapshot },
}

impl WireMessage {
    pub fn event(event: BootEvent) -> Self {
        Self::Event {
            version: PROTOCOL_VERSION,
            event,
        }
    }

    pub fn snapshot(snapshot: BootSnapshot) -> Self {
        Self::Snapshot {
            version: PROTOCOL_VERSION,
            snapshot,
        }
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        match self {
            Self::Event { version, event } => {
                validate_version(*version)?;
                event.validate()
            }
            Self::Snapshot { version, snapshot } => {
                validate_version(*version)?;
                snapshot.validate()
            }
        }
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
    use crate::{BootDomain, BootEvent};

    #[test]
    fn wire_event_round_trip_stays_bounded() {
        let message = WireMessage::event(BootEvent::DomainReady {
            sequence: 4,
            elapsed_ms: 850,
            domain: BootDomain::Storage,
        });

        let bytes = serde_json::to_vec(&message).unwrap();
        validate_datagram_size(&bytes).unwrap();
        let decoded: WireMessage = serde_json::from_slice(&bytes).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded, message);
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
