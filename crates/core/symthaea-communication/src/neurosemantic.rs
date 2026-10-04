//! Neurosemantic communication protocol primitives.
//!
//! Protocol infrastructure for exchanging derived cognitive representations.
//! This is not a claim that arbitrary thoughts can be decoded or written.

use crate::{content_hash, RepresentationFamily};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Representation channel used for routing and authorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CognitiveChannel {
    Semantic,
    Affective,
    Spatial,
    Temporal,
    Procedural,
    Sensory,
}

/// Direction of information flow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelDirection {
    Read,
    Write,
}

/// Purpose binding prevents ambient authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommunicationPurpose {
    AssistiveCommunication,
    HumanCollaboration,
    AgentCoordination,
    Research,
}

/// Derived representations only. Raw neural samples are intentionally absent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NeurosemanticPayload {
    SemanticGraph(Vec<u8>),
    Hypervector(Vec<i8>),
    StructuredRepresentation(Vec<u8>),
    DerivedNeuralFeature(Vec<f32>),
}

/// Sensitivity class for policy and data minimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveSensitivity {
    Public,
    Contextual,
    Private,
    HighlyPrivate,
}

/// Explicit, peer-specific, purpose-bound, time-bounded consent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CognitiveConsentLease {
    pub lease_id: String,
    pub subject_id: String,
    pub peer_id: String,
    pub purpose: CommunicationPurpose,
    pub read_scopes: BTreeSet<CognitiveChannel>,
    pub write_scopes: BTreeSet<CognitiveChannel>,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub consent_epoch: u64,
    pub revoked: bool,
}

impl CognitiveConsentLease {
    pub fn validate(&self) -> Result<(), String> {
        if self.lease_id.trim().is_empty()
            || self.subject_id.trim().is_empty()
            || self.peer_id.trim().is_empty()
            || self.issued_at_unix_s >= self.expires_at_unix_s
        {
            return Err("consent lease identity or time bounds are invalid".into());
        }
        Ok(())
    }

    pub fn authorizes(
        &self,
        channel: CognitiveChannel,
        direction: ChannelDirection,
        purpose: CommunicationPurpose,
        peer_id: &str,
        now_unix_s: u64,
    ) -> bool {
        if self.revoked
            || self.peer_id != peer_id
            || self.purpose != purpose
            || now_unix_s < self.issued_at_unix_s
            || now_unix_s >= self.expires_at_unix_s
        {
            return false;
        }

        match direction {
            ChannelDirection::Read => self.read_scopes.contains(&channel),
            ChannelDirection::Write => self.write_scopes.contains(&channel),
        }
    }
}

/// Content-addressed packet for derived cognitive representations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeurosemanticPacket {
    pub protocol_version: u16,
    pub sequence: u64,
    pub sender_id: String,
    pub recipient_id: String,
    pub purpose: CommunicationPurpose,
    pub channel: CognitiveChannel,
    pub direction: ChannelDirection,
    pub representation: RepresentationFamily,
    pub sensitivity: CognitiveSensitivity,
    pub confidence: f32,
    pub payload: NeurosemanticPayload,
    pub payload_hash: String,
    pub packet_hash: String,
}

impl NeurosemanticPacket {
    pub fn new(
        sequence: u64,
        sender_id: impl Into<String>,
        recipient_id: impl Into<String>,
        purpose: CommunicationPurpose,
        channel: CognitiveChannel,
        direction: ChannelDirection,
        representation: RepresentationFamily,
        sensitivity: CognitiveSensitivity,
        confidence: f32,
        payload: NeurosemanticPayload,
    ) -> Result<Self, String> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err("confidence must be finite and in [0, 1]".into());
        }

        let mut packet = Self {
            protocol_version: 1,
            sequence,
            sender_id: sender_id.into(),
            recipient_id: recipient_id.into(),
            purpose,
            channel,
            direction,
            representation,
            sensitivity,
            confidence,
            payload,
            payload_hash: String::new(),
            packet_hash: String::new(),
        };
        packet.refresh_hashes()?;
        Ok(packet)
    }

    pub fn validate_integrity(&self) -> Result<(), String> {
        if self.protocol_version == 0
            || self.sender_id.trim().is_empty()
            || self.recipient_id.trim().is_empty()
        {
            return Err("packet identity or protocol version is invalid".into());
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err("packet confidence must be finite and in [0, 1]".into());
        }

        validate_payload(&self.payload)?;\n        let expected_payload_hash = payload_hash(&self.payload)?;
        if self.payload_hash != expected_payload_hash {
            return Err("payload hash mismatch".into());
        }

        let mut canonical = self.clone();
        canonical.packet_hash.clear();
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| format!("packet serialization: {error}"))?;
        if self.packet_hash != content_hash(&bytes) {
            return Err("packet hash mismatch".into());
        }
        Ok(())
    }

    pub fn refresh_hashes(&mut self) -> Result<(), String> {
        self.payload_hash = payload_hash(&self.payload)?;
        let mut canonical = self.clone();
        canonical.packet_hash.clear();
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| format!("packet serialization: {error}"))?;
        self.packet_hash = content_hash(&bytes);
        Ok(())
    }
}

/// A packet plus the exact consent epoch used to authorize it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuthorizedNeurosemanticMessage {
    pub packet: NeurosemanticPacket,
    pub consent_epoch: u64,
    pub lease_id: String,
}

impl AuthorizedNeurosemanticMessage {
    pub fn validate(
        &self,
        lease: &CognitiveConsentLease,
        now_unix_s: u64,
    ) -> Result<(), String> {
        self.packet.validate_integrity()?;
        lease.validate()?;

        if self.lease_id != lease.lease_id || self.consent_epoch != lease.consent_epoch {
            return Err("authorization epoch or lease id mismatch".into());
        }

        let (expected_sender, expected_recipient) = match self.packet.direction {
            ChannelDirection::Read => (lease.subject_id.as_str(), lease.peer_id.as_str()),
            ChannelDirection::Write => (lease.peer_id.as_str(), lease.subject_id.as_str()),
        };

        if self.packet.sender_id != expected_sender || self.packet.recipient_id != expected_recipient {
            return Err("packet endpoints do not match the consent direction".into());
        }

        if !lease.authorizes(
            self.packet.channel,
            self.packet.direction,
            self.packet.purpose,
            &lease.peer_id,
            now_unix_s,
        ) {
            return Err("communication is not authorized by the active consent lease".into());
        }

        Ok(())
    }
}

/// Persistent per-lease sequence guard for replay/collision detection.
///
/// The tracker deliberately lives above packet integrity: integrity answers
/// "was this packet altered?", while this tracker answers "have we already
/// accepted this packet sequence in this consent epoch?".
pub const NEUROSEMANTIC_PROTOCOL_VERSION: u16 = 1;
pub const MAX_NEUROSEMANTIC_PAYLOAD_BYTES: usize = 1_048_576;
pub const MAX_TRACKED_NEUROSEMANTIC_SESSIONS: usize = 4096;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticReplayTracker {
    latest: std::collections::BTreeMap<(String, String, String, u64), ReplayState>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ReplayState {
    sequence: u64,
    packet_hash: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayDecision {
    Accept,
    Duplicate,
}

impl NeurosemanticReplayTracker {
    pub fn observe(
        &mut self,
        message: &AuthorizedNeurosemanticMessage,
    ) -> Result<ReplayDecision, String> {
        message.packet.validate_integrity()?;
        if message.packet.sequence == 0 {
            return Err("packet sequence must be non-zero".into());
        }

        let key = (
            message.packet.sender_id.clone(),
            message.packet.recipient_id.clone(),
            message.lease_id.clone(),
            message.consent_epoch,
        );

        match self.latest.get(&key) {
            None => {
                if self.latest.len() >= MAX_TRACKED_NEUROSEMANTIC_SESSIONS {
                    return Err("replay tracker capacity exceeded".into());
                }
                self.latest.insert(
                    key,
                    ReplayState {
                        sequence: message.packet.sequence,
                        packet_hash: message.packet.packet_hash.clone(),
                    },
                );
                Ok(ReplayDecision::Accept)
            }
            Some(state) if message.packet.sequence > state.sequence => {
                self.latest.insert(
                    key,
                    ReplayState {
                        sequence: message.packet.sequence,
                        packet_hash: message.packet.packet_hash.clone(),
                    },
                );
                Ok(ReplayDecision::Accept)
            }
            Some(state)
                if message.packet.sequence == state.sequence
                    && message.packet.packet_hash == state.packet_hash =>
            {
                Ok(ReplayDecision::Duplicate)
            }
            Some(state) => Err(format!(
                "replay or sequence collision: latest={}, proposed={}",
                state.sequence, message.packet.sequence
            )),
        }
    }
}

fn validate_payload(payload: &NeurosemanticPayload) -> Result<(), String> {
    match payload {
        NeurosemanticPayload::DerivedNeuralFeature(values)
            if values.iter().any(|value| !value.is_finite()) =>
        {
            Err("derived neural features must be finite".into())
        }
        _ => {
            let bytes = serde_json::to_vec(payload)
                .map_err(|error| format!("payload serialization: {error}"))?;
            if bytes.len() > MAX_NEUROSEMANTIC_PAYLOAD_BYTES {
                return Err(format!(
                    "neurosemantic payload exceeds {} bytes",
                    MAX_NEUROSEMANTIC_PAYLOAD_BYTES
                ));
            }
            Ok(())
        }
    }
}

fn payload_hash(payload: &NeurosemanticPayload) -> Result<String, String> {
    let bytes = serde_json::to_vec(payload)
        .map_err(|error| format!("payload serialization: {error}"))?;
    Ok(content_hash(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lease() -> CognitiveConsentLease {
        CognitiveConsentLease {
            lease_id: "lease-1".into(),
            subject_id: "subject".into(),
            peer_id: "peer".into(),
            purpose: CommunicationPurpose::HumanCollaboration,
            read_scopes: BTreeSet::from([CognitiveChannel::Semantic]),
            write_scopes: BTreeSet::from([CognitiveChannel::Semantic]),
            issued_at_unix_s: 100,
            expires_at_unix_s: 200,
            consent_epoch: 7,
            revoked: false,
        }
    }

    #[test]
    fn consent_is_deny_by_default() {
        let l = lease();
        assert!(!l.authorizes(
            CognitiveChannel::Affective,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "peer",
            150
        ));
        assert!(!l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::Research,
            "peer",
            150
        ));
        assert!(!l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "other-peer",
            150
        ));
    }

    #[test]
    fn expiry_and_revocation_block_authority() {
        let mut l = lease();
        assert!(l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "peer",
            150
        ));
        assert!(!l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "peer",
            200
        ));
        l.revoked = true;
        assert!(!l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "peer",
            150
        ));
    }

    #[test]
    fn packet_hashes_detect_tampering() {
        let mut packet = NeurosemanticPacket::new(
            1,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.91,
            NeurosemanticPayload::Hypervector(vec![1, -1, 1]),
        )
        .unwrap();

        assert!(packet.validate_integrity().is_ok());
        if let NeurosemanticPayload::Hypervector(values) = &mut packet.payload {
            values[0] = -values[0];
        }
        assert!(packet.validate_integrity().is_err());
    }

    #[test]
    fn authorized_message_requires_scope_epoch_and_subject_binding() {
        let packet = NeurosemanticPacket::new(
            3,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.88,
            NeurosemanticPayload::StructuredRepresentation(b"hello".to_vec()),
        )
        .unwrap();

        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: 7,
            lease_id: "lease-1".into(),
        };
        assert!(message.validate(&lease(), 150).is_ok());

        let mut wrong_epoch = message.clone();
        wrong_epoch.consent_epoch = 8;
        assert!(wrong_epoch.validate(&lease(), 150).is_err());
    }

    #[test]
    fn read_direction_flows_from_subject_to_peer() {
        let base = lease();
        let packet = NeurosemanticPacket::new(
            9,
            "subject",
            "peer",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.77,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();

        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: base.consent_epoch,
            lease_id: base.lease_id,
        };

        assert!(message.validate(&lease(), 150).is_ok());
    }

    #[test]
    fn read_direction_rejects_peer_to_subject_packet() {
        let base = lease();
        let packet = NeurosemanticPacket::new(
            10,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.77,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();

        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: base.consent_epoch,
            lease_id: base.lease_id,
        };

        assert!(message.validate(&lease(), 150).is_err());
    }

    #[test]
    #[test]
    fn nonfinite_derived_features_are_rejected() {
        let result = NeurosemanticPacket::new(
            11,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.5,
            NeurosemanticPayload::DerivedNeuralFeature(vec![0.1, f32::NAN]),
        );
        assert!(result.is_err());
    }

    #[test]
    fn unsupported_protocol_versions_are_rejected() {
        let mut packet = NeurosemanticPacket::new(
            12,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        packet.protocol_version += 1;
        assert!(packet.validate_integrity().is_err());
    }

    #[test]
    fn payload_size_is_bounded() {
        let values = vec![1_i8; MAX_NEUROSEMANTIC_PAYLOAD_BYTES];
        let result = NeurosemanticPacket::new(
            13,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.5,
            NeurosemanticPayload::Hypervector(values),
        );
        assert!(result.is_err());
    }

    #[test]
    fn raw_neural_samples_are_not_a_payload_variant() {
        let encoded = serde_json::to_string(
            &NeurosemanticPayload::DerivedNeuralFeature(vec![0.1, 0.2]),
        )
        .unwrap();
        assert!(encoded.contains("DerivedNeuralFeature"));
        assert!(!encoded.contains("RawNeural"));
    }
}
