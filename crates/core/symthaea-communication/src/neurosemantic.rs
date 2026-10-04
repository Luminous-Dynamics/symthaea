//! Neurosemantic communication protocol primitives.
//!
//! Protocol infrastructure for exchanging derived cognitive representations.
//! This is not a claim that arbitrary thoughts can be decoded or written.

use crate::{content_hash, RepresentationFamily};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Stable neurosemantic protocol version.
pub const NEUROSEMANTIC_PROTOCOL_VERSION: u16 = 1;

/// Maximum encoded payload size before content hashing.
pub const MAX_NEUROSEMANTIC_PAYLOAD_BYTES: usize = 1_048_576;

/// Maximum number of replay-tracker keys retained in memory.
pub const MAX_TRACKED_NEUROSEMANTIC_SESSIONS: usize = 4096;

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

/// Direction of information flow from the subject's perspective.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelDirection {
    /// Subject -> peer.
    Read,
    /// Peer -> subject.
    Write,
}

/// Purpose binding prevents ambient authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CommunicationPurpose {
    AssistiveCommunication,
    HumanCollaboration,
    AgentCoordination,
    Research,
}

/// Machine-readable class of the cognitive data being transported.
/// Unknown is the conservative legacy/default state and cannot authorize
/// neurosemantic access.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticDataClass {
    #[default]
    Unknown,
    RawNeuralRecording,
    DerivedNeuralFeature,
    SemanticRepresentation,
    DecodedClaim,
    PersonalizedDecoderModel,
}

/// Machine-readable inference classes a data product may expose or enable.
/// These remain separate from data class and transport sensitivity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticInferenceClass {
    Unknown,
    SignalPattern,
    UnitPattern,
    LinguisticContent,
    SemanticContent,
    AffectiveState,
    Intent,
    Identity,
}

pub const NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticDataPolicy {
    pub schema_version: u16,
    pub data_class: NeurosemanticDataClass,
    #[serde(default)]
    pub inference_classes: BTreeSet<NeurosemanticInferenceClass>,
    #[serde(default)]
    pub permitted_purposes: BTreeSet<CommunicationPurpose>,
}

impl Default for NeurosemanticDataPolicy {
    fn default() -> Self {
        Self {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            data_class: NeurosemanticDataClass::Unknown,
            inference_classes: BTreeSet::new(),
            permitted_purposes: BTreeSet::new(),
        }
    }
}

impl NeurosemanticDataPolicy {
    pub fn validates(&self) -> bool {
        self.schema_version == NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION
            && self.data_class != NeurosemanticDataClass::Unknown
            && !self.inference_classes.is_empty()
            && !self.inference_classes.contains(&NeurosemanticInferenceClass::Unknown)
            && !self.permitted_purposes.is_empty()
    }

    pub fn allows_purpose(&self, purpose: CommunicationPurpose) -> bool {
        self.validates() && self.permitted_purposes.contains(&purpose)
    }
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
    #[serde(default = "default_public_sensitivity")]
    pub max_read_sensitivity: CognitiveSensitivity,
    #[serde(default = "default_public_sensitivity")]
    pub max_write_sensitivity: CognitiveSensitivity,
    #[serde(default)]
    pub read_data_classes: BTreeSet<NeurosemanticDataClass>,
    #[serde(default)]
    pub write_data_classes: BTreeSet<NeurosemanticDataClass>,
    #[serde(default)]
    pub read_inference_classes: BTreeSet<NeurosemanticInferenceClass>,
    #[serde(default)]
    pub write_inference_classes: BTreeSet<NeurosemanticInferenceClass>,
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

    pub fn authorizes_sensitivity(
        &self,
        direction: ChannelDirection,
        sensitivity: CognitiveSensitivity,
    ) -> bool {
        let maximum = match direction {
            ChannelDirection::Read => self.max_read_sensitivity,
            ChannelDirection::Write => self.max_write_sensitivity,
        };
        sensitivity <= maximum
    }

    pub fn authorizes_data_policy(
        &self,
        direction: ChannelDirection,
        policy: &NeurosemanticDataPolicy,
    ) -> bool {
        if !policy.validates() {
            return false;
        }

        let (allowed_data_classes, allowed_inference_classes) = match direction {
            ChannelDirection::Read => (&self.read_data_classes, &self.read_inference_classes),
            ChannelDirection::Write => (&self.write_data_classes, &self.write_inference_classes),
        };

        allowed_data_classes.contains(&policy.data_class)
            && policy
                .inference_classes
                .iter()
                .all(|inference| allowed_inference_classes.contains(inference))
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
    #[serde(default)]
    pub data_policy: NeurosemanticDataPolicy,
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
        if sequence == 0 {
            return Err("packet sequence must be non-zero".into());
        }
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err("confidence must be finite and in [0, 1]".into());
        }
        validate_payload(&payload)?;

        let mut packet = Self {
            protocol_version: NEUROSEMANTIC_PROTOCOL_VERSION,
            sequence,
            sender_id: sender_id.into(),
            recipient_id: recipient_id.into(),
            purpose,
            channel,
            direction,
            representation,
            sensitivity,
            data_policy: NeurosemanticDataPolicy::default(),
            confidence,
            payload,
            payload_hash: String::new(),
            packet_hash: String::new(),
        };
        packet.refresh_hashes()?;
        Ok(packet)
    }

    pub fn new_with_policy(
        sequence: u64,
        sender_id: impl Into<String>,
        recipient_id: impl Into<String>,
        purpose: CommunicationPurpose,
        channel: CognitiveChannel,
        direction: ChannelDirection,
        representation: RepresentationFamily,
        sensitivity: CognitiveSensitivity,
        data_policy: NeurosemanticDataPolicy,
        confidence: f32,
        payload: NeurosemanticPayload,
    ) -> Result<Self, String> {
        if !data_policy.validates() {
            return Err("neurosemantic data policy is invalid or deny-by-default".into());
        }
        let mut packet = Self::new(
            sequence,
            sender_id,
            recipient_id,
            purpose,
            channel,
            direction,
            representation,
            sensitivity,
            confidence,
            payload,
        )?;
        packet.data_policy = data_policy;
        packet.refresh_hashes()?;
        Ok(packet)
    }

    pub fn validate_integrity(&self) -> Result<(), String> {
        if self.protocol_version != NEUROSEMANTIC_PROTOCOL_VERSION
            || self.sender_id.trim().is_empty()
            || self.recipient_id.trim().is_empty()
        {
            return Err("packet identity or protocol version is invalid".into());
        }
        if self.sequence == 0 {
            return Err("packet sequence must be non-zero".into());
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err("packet confidence must be finite and in [0, 1]".into());
        }

        if self.data_policy.schema_version != NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION {
            return Err("unsupported neurosemantic data policy schema version".into());
        }
        validate_payload(&self.payload)?;
        let expected_payload_hash = payload_hash(&self.payload)?;
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
        validate_payload(&self.payload)?;
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

        if self.packet.sender_id != expected_sender || self.packet.recipient_id != expected_recipient
        {
            return Err("packet endpoints do not match the consent direction".into());
        }

        if !self.packet.data_policy.allows_purpose(self.packet.purpose) {
            return Err("packet data policy does not permit the requested purpose".into());
        }
        if !lease.authorizes_data_policy(self.packet.direction, &self.packet.data_policy) {
            return Err("packet data class or inference class is not authorized by the consent lease".into());
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
        if !lease.authorizes_sensitivity(self.packet.direction, self.packet.sensitivity) {
            return Err("packet sensitivity exceeds the consent lease ceiling".into());
        }

        Ok(())
    }
}

/// Persistent per-lease sequence guard for replay/collision detection.
///
/// The tracker deliberately lives above packet integrity: integrity answers
/// "was this packet altered?", while this tracker answers "have we already
/// accepted this packet sequence in this consent epoch?".
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticReplayTracker {
    latest: BTreeMap<(String, String, String, u64), ReplayState>,
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
    /// Authorize first, then apply replay/collision protection.
    pub fn observe_authorized(
        &mut self,
        message: &AuthorizedNeurosemanticMessage,
        lease: &CognitiveConsentLease,
        now_unix_s: u64,
    ) -> Result<ReplayDecision, String> {
        message.validate(lease, now_unix_s)?;
        self.observe(message)
    }

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

fn default_public_sensitivity() -> CognitiveSensitivity {
    CognitiveSensitivity::Public
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

    fn semantic_policy() -> NeurosemanticDataPolicy {
        NeurosemanticDataPolicy {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            data_class: NeurosemanticDataClass::SemanticRepresentation,
            inference_classes: BTreeSet::from([NeurosemanticInferenceClass::SemanticContent]),
            permitted_purposes: BTreeSet::from([CommunicationPurpose::HumanCollaboration]),
        }
    }

    fn lease() -> CognitiveConsentLease {
        CognitiveConsentLease {
            lease_id: "lease-1".into(),
            subject_id: "subject".into(),
            peer_id: "peer".into(),
            purpose: CommunicationPurpose::HumanCollaboration,
            read_scopes: BTreeSet::from([CognitiveChannel::Semantic]),
            write_scopes: BTreeSet::from([CognitiveChannel::Semantic]),
            max_read_sensitivity: CognitiveSensitivity::Private,
            max_write_sensitivity: CognitiveSensitivity::Private,
            read_data_classes: BTreeSet::from([NeurosemanticDataClass::SemanticRepresentation]),
            write_data_classes: BTreeSet::from([NeurosemanticDataClass::SemanticRepresentation]),
            read_inference_classes: BTreeSet::from([NeurosemanticInferenceClass::SemanticContent]),
            write_inference_classes: BTreeSet::from([NeurosemanticInferenceClass::SemanticContent]),
            issued_at_unix_s: 100,
            expires_at_unix_s: 200,
            consent_epoch: 7,
            revoked: false,
        }
    }

    #[test]
    fn data_policy_is_deny_by_default() {
        let policy = NeurosemanticDataPolicy::default();
        assert!(!policy.validates());
        assert!(!policy.allows_purpose(CommunicationPurpose::HumanCollaboration));
    }

    #[test]
    fn data_policy_requires_explicit_inference_and_purpose() {
        let mut policy = semantic_policy();
        assert!(policy.validates());
        policy.inference_classes.clear();
        assert!(!policy.validates());
        let mut policy = semantic_policy();
        policy.permitted_purposes.clear();
        assert!(!policy.validates());
    }

    #[test]
    fn legacy_lease_data_permissions_default_to_empty_and_deny() {
        let lease = lease();
        let mut value = serde_json::to_value(&lease).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("read_data_classes");
        object.remove("write_data_classes");
        object.remove("read_inference_classes");
        object.remove("write_inference_classes");
        let restored: CognitiveConsentLease = serde_json::from_value(value).unwrap();
        assert!(restored.read_data_classes.is_empty());
        assert!(restored.write_data_classes.is_empty());
        assert!(!restored.authorizes_data_policy(ChannelDirection::Read, &semantic_policy()));
    }

    #[test]
    fn data_class_and_inference_authorization_are_independent() {
        let lease = lease();
        let policy = semantic_policy();
        assert!(lease.authorizes_data_policy(ChannelDirection::Read, &policy));
        let mut identity = policy.clone();
        identity.inference_classes = BTreeSet::from([NeurosemanticInferenceClass::Identity]);
        assert!(!lease.authorizes_data_policy(ChannelDirection::Read, &identity));
        let mut decoded_claim = policy;
        decoded_claim.data_class = NeurosemanticDataClass::DecodedClaim;
        assert!(!lease.authorizes_data_policy(ChannelDirection::Read, &decoded_claim));
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
        let packet = NeurosemanticPacket::new_with_policy(
            3,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
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
        let packet = NeurosemanticPacket::new_with_policy(
            9,
            "subject",
            "peer",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
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
        let packet = NeurosemanticPacket::new_with_policy(
            10,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
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
            semantic_policy(),
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
    fn zero_sequence_is_rejected_at_construction() {
        let result = NeurosemanticPacket::new(
            0,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        );
        assert!(result.is_err());
    }

    #[test]
    fn authorized_replay_path_requires_active_consent() {
        let base = lease();
        let packet = NeurosemanticPacket::new_with_policy(
            14,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: base.consent_epoch,
            lease_id: base.lease_id.clone(),
        };
        let mut tracker = NeurosemanticReplayTracker::default();
        assert_eq!(
            tracker.observe_authorized(&message, &base, 150).unwrap(),
            ReplayDecision::Accept
        );
        let mut revoked = base.clone();
        revoked.revoked = true;
        assert!(tracker
            .observe_authorized(&message, &revoked, 150)
            .is_err());
    }

    #[test]
    fn sensitivity_ceiling_is_enforced() {
        let base = lease();
        let packet = NeurosemanticPacket::new_with_policy(
            15,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::HighlyPrivate,
            semantic_policy(),
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: base.consent_epoch,
            lease_id: base.lease_id.clone(),
        };
        assert!(message.validate(&base, 150).is_err());
        let mut elevated = base.clone();
        elevated.max_write_sensitivity = CognitiveSensitivity::HighlyPrivate;
        assert!(message.validate(&elevated, 150).is_ok());
    }

    #[test]
    fn legacy_leases_default_to_public_sensitivity() {
        let lease = lease();
        let mut value = serde_json::to_value(&lease).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("max_read_sensitivity");
        object.remove("max_write_sensitivity");
        let restored: CognitiveConsentLease = serde_json::from_value(value).unwrap();
        assert_eq!(restored.max_read_sensitivity, CognitiveSensitivity::Public);
        assert_eq!(restored.max_write_sensitivity, CognitiveSensitivity::Public);
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
