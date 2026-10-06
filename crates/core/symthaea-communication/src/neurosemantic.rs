//! Neurosemantic communication protocol primitives.
//!
//! Protocol infrastructure for exchanging derived cognitive representations.
//! This is not a claim that arbitrary thoughts can be decoded or written.

use crate::{content_hash, RepresentationFamily};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Stable neurosemantic protocol version.
pub const NEUROSEMANTIC_PROTOCOL_VERSION: u16 = 1;

/// Maximum encoded payload size before content hashing.
pub const MAX_NEUROSEMANTIC_PAYLOAD_BYTES: usize = 1_048_576;

/// Maximum number of replay-tracker keys retained in memory.
pub const MAX_TRACKED_NEUROSEMANTIC_SESSIONS: usize = 4096;

/// Maximum identifier size accepted by protocol constructors and validators.
pub const MAX_NEUROSEMANTIC_ID_BYTES: usize = 4096;

/// Maximum serialized packet/authorization artifact size before JSON materialization.
pub const MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES: usize = MAX_NEUROSEMANTIC_PAYLOAD_BYTES;
pub const MAX_NEUROSEMANTIC_JURISDICTION_ID_BYTES: usize = 64;
pub const MAX_NEUROSEMANTIC_SECONDARY_USE_CLASSES: usize = 16;
pub const MAX_NEUROSEMANTIC_DESTINATION_JURISDICTIONS: usize = 16;

const NEUROSEMANTIC_POLICY_PROVENANCE_DOMAIN: &[u8] =
    b"symthaea-neurosemantic-policy-provenance-v1\0";
const NEUROSEMANTIC_POLICY_ATTESTATION_DOMAIN: &[u8] =
    b"symthaea-neurosemantic-policy-attestation-v1\0";
const MAX_NEUROSEMANTIC_AUTHORITY_REF_BYTES: usize = 4096;
const MAX_NEUROSEMANTIC_AUTHORITY_KEY_REF_BYTES: usize = 4096;
const MAX_NEUROSEMANTIC_AUTHORITY_SIGNATURE_BYTES: usize = 64;

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

pub const NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION: u16 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NeurosemanticSecondaryUse {
    Research,
    ModelTraining,
    ProductDevelopment,
    CommercialAnalytics,
    BehavioralProfiling,
    AffectiveInference,
    IdentityInference,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeurosemanticRetentionPolicy {
    Ephemeral,
    UntilUnixS(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeurosemanticHandlingAction {
    Transmit,
    Persist,
    SecondaryUse(NeurosemanticSecondaryUse),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticHandlingPolicy {
    pub schema_version: u16,
    /// Opaque reference to the externally authoritative policy/consent record.
    /// Symthaea binds this reference into packet integrity but does not authenticate
    /// the authority; the deployment policy layer remains responsible for verification.
    pub policy_provenance_ref: String,
    /// BLAKE3-256 digest of the exact externally authoritative policy/consent record bytes
    /// (or its separately specified canonical form). This is a binding, not an authority signature.
    pub policy_provenance_hash: String,
    /// Jurisdiction identifier asserted for the originating data/controller context.
    /// This is an interoperable policy identifier, not a legal determination.
    pub origin_jurisdiction: String,
    /// Explicit destination allow-list. Empty means deny-all.
    #[serde(default)]
    pub permitted_destination_jurisdictions: BTreeSet<String>,
    /// Explicitly authorized downstream uses beyond the packet's primary purpose.
    /// Empty means no secondary use.
    #[serde(default)]
    pub permitted_secondary_uses: BTreeSet<NeurosemanticSecondaryUse>,
    pub retention: NeurosemanticRetentionPolicy,
}

impl Default for NeurosemanticHandlingPolicy {
    fn default() -> Self {
        Self {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            policy_provenance_ref: String::new(),
            policy_provenance_hash: String::new(),
            origin_jurisdiction: String::new(),
            permitted_destination_jurisdictions: BTreeSet::new(),
            permitted_secondary_uses: BTreeSet::new(),
            retention: NeurosemanticRetentionPolicy::Ephemeral,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticPolicyAuthorityAttestation {
    pub schema_version: u16,
    /// External authority identifier. Symthaea does not establish trust in this
    /// identifier; the deployment trust layer must map it to an authorized issuer.
    pub authority_ref: String,
    /// External key identifier used by the authority's trust registry.
    pub key_ref: String,
    /// Exact policy fingerprint that the authority attests.
    pub handling_policy_fingerprint: String,
    /// Exact provenance digest that the authority attests.
    pub policy_provenance_hash: String,
    /// Inclusive validity start for the attestation.
    pub issued_at_unix_s: u64,
    /// Exclusive validity end for the attestation.
    pub expires_at_unix_s: u64,
    /// Ed25519 signature over the domain-separated attestation message.
    pub signature: Vec<u8>,
}

pub const NEUROSEMANTIC_POLICY_ATTESTATION_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NeurosemanticPolicyProvenanceBinding {
    policy_provenance_ref: String,
    policy_provenance_hash: String,
    handling_policy_fingerprint: String,
    authority_ref: String,
    key_ref: String,
    attestation_fingerprint: String,
    attestation_expires_at_unix_s: u64,
}

impl NeurosemanticPolicyAuthorityAttestation {
    /// Deserialize an untrusted authority attestation only after enforcing the
    /// serialized byte ceiling and structural bounds.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic authority attestation JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let attestation: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic authority attestation JSON: {error}"))?;
        attestation.validate()?;
        Ok(attestation)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NEUROSEMANTIC_POLICY_ATTESTATION_SCHEMA_VERSION
            || !valid_identifier(&self.authority_ref)
            || !valid_identifier(&self.key_ref)
            || self.authority_ref.len() > MAX_NEUROSEMANTIC_AUTHORITY_REF_BYTES
            || self.key_ref.len() > MAX_NEUROSEMANTIC_AUTHORITY_KEY_REF_BYTES
            || !valid_blake3_digest(&self.handling_policy_fingerprint)
            || !valid_blake3_digest(&self.policy_provenance_hash)
            || self.issued_at_unix_s >= self.expires_at_unix_s
            || self.signature.len() != MAX_NEUROSEMANTIC_AUTHORITY_SIGNATURE_BYTES
        {
            return Err("neurosemantic authority attestation fields are invalid".into());
        }
        Ok(())
    }

    pub fn message_bytes(
        authority_ref: &str,
        key_ref: &str,
        handling_policy_fingerprint: &str,
        policy_provenance_hash: &str,
        issued_at_unix_s: u64,
        expires_at_unix_s: u64,
    ) -> Result<Vec<u8>, String> {
        if !valid_identifier(authority_ref)
            || authority_ref.len() > MAX_NEUROSEMANTIC_AUTHORITY_REF_BYTES
            || !valid_identifier(key_ref)
            || key_ref.len() > MAX_NEUROSEMANTIC_AUTHORITY_KEY_REF_BYTES
            || !valid_blake3_digest(handling_policy_fingerprint)
            || !valid_blake3_digest(policy_provenance_hash)
            || issued_at_unix_s >= expires_at_unix_s
        {
            return Err("neurosemantic authority attestation fields are invalid".into());
        }
        let mut bytes = Vec::with_capacity(256);
        bytes.extend_from_slice(NEUROSEMANTIC_POLICY_ATTESTATION_DOMAIN);
        for value in [authority_ref.as_bytes(), key_ref.as_bytes(), handling_policy_fingerprint.as_bytes(), policy_provenance_hash.as_bytes()] {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value);
        }
        bytes.extend_from_slice(&issued_at_unix_s.to_le_bytes());
        bytes.extend_from_slice(&expires_at_unix_s.to_le_bytes());
        Ok(bytes)
    }

    pub fn verify(
        &self,
        expected_policy_fingerprint: &str,
        expected_policy_provenance_hash: &str,
        verifying_key: &VerifyingKey,
        now_unix_s: u64,
    ) -> Result<(), String> {
        self.validate()?;
        if self.handling_policy_fingerprint != expected_policy_fingerprint
            || self.policy_provenance_hash != expected_policy_provenance_hash
            || now_unix_s < self.issued_at_unix_s
            || now_unix_s >= self.expires_at_unix_s
        {
            return Err("neurosemantic authority attestation is invalid or outside its validity interval".into());
        }
        let message = Self::message_bytes(
            &self.authority_ref,
            &self.key_ref,
            &self.handling_policy_fingerprint,
            &self.policy_provenance_hash,
            self.issued_at_unix_s,
            self.expires_at_unix_s,
        )?;
        let signature_bytes: [u8; 64] = self.signature.as_slice()
            .try_into()
            .map_err(|_| "neurosemantic authority signature has invalid length".to_string())?;
        let signature = Signature::from_bytes(&signature_bytes);
        verifying_key
            .verify(&message, &signature)
            .map_err(|_| "neurosemantic authority signature verification failed".to_string())
    }

    /// Fingerprint the exact serialized authority proof artifact, including its signature.
    pub fn fingerprint_for_attestation(&self) -> Result<String, String> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("authority attestation serialization: {error}"))?;
        Ok(content_hash(&bytes))
    }
}

impl NeurosemanticPolicyProvenanceBinding {
    pub fn reference(&self) -> &str {
        &self.policy_provenance_ref
    }

    pub fn digest(&self) -> &str {
        &self.policy_provenance_hash
    }

    pub fn handling_policy_fingerprint(&self) -> &str {
        &self.handling_policy_fingerprint
    }

    pub fn authority_ref(&self) -> &str {
        &self.authority_ref
    }

    pub fn key_ref(&self) -> &str {
        &self.key_ref
    }

    pub fn attestation_fingerprint(&self) -> &str {
        &self.attestation_fingerprint
    }

    pub fn attestation_expires_at_unix_s(&self) -> u64 {
        self.attestation_expires_at_unix_s
    }
}

impl NeurosemanticHandlingPolicy {
    pub fn fingerprint_for_attestation(&self) -> Result<String, String> {
        let mut canonical = self.clone();
        canonical.policy_provenance_hash.clear();
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| format!("handling policy serialization: {error}"))?;
        Ok(content_hash(&bytes))
    }

    pub fn validates(&self) -> bool {
        self.schema_version == NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION
            && valid_identifier(&self.policy_provenance_ref)
            && valid_blake3_digest(&self.policy_provenance_hash)
            && valid_jurisdiction_id(&self.origin_jurisdiction)
            && !self.permitted_destination_jurisdictions.is_empty()
            && self.permitted_destination_jurisdictions.len() <= MAX_NEUROSEMANTIC_DESTINATION_JURISDICTIONS
            && self
                .permitted_destination_jurisdictions
                .iter()
                .all(|jurisdiction| valid_jurisdiction_id(jurisdiction))
            && self.permitted_destination_jurisdictions.contains(&self.origin_jurisdiction)
            && self.permitted_secondary_uses.len() <= MAX_NEUROSEMANTIC_SECONDARY_USE_CLASSES
            && matches!(
                self.retention,
                NeurosemanticRetentionPolicy::Ephemeral
                    | NeurosemanticRetentionPolicy::UntilUnixS(_)
            )
    }

    /// Verify that the exact external policy/consent record is bound to the
    /// declared provenance reference and digest. This does not produce a
    /// handling capability or authenticate the issuing authority.
    pub fn verify_policy_record_binding_bytes(&self, record_bytes: &[u8]) -> bool {
        self.validates()
            && record_bytes.len() <= MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            && compute_policy_provenance_hash(&self.policy_provenance_ref, record_bytes)
                == self.policy_provenance_hash
    }

    /// Produce a handling capability only after both external-record binding and
    /// authority attestation verification succeed.
    pub fn bind_policy_provenance_with_attestation(
        &self,
        record_bytes: &[u8],
        attestation: &NeurosemanticPolicyAuthorityAttestation,
        verifying_key: &VerifyingKey,
        now_unix_s: u64,
    ) -> Result<NeurosemanticPolicyProvenanceBinding, String> {
        if !self.validates() {
            return Err("neurosemantic policy provenance state is invalid".into());
        }
        if record_bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err("neurosemantic policy provenance record exceeds the serialized artifact limit".into());
        }
        let policy_fingerprint = self.fingerprint_for_attestation()?;
        if compute_policy_provenance_hash(&self.policy_provenance_ref, record_bytes)
            != self.policy_provenance_hash
        {
            return Err("neurosemantic policy provenance reference/record binding mismatch".into());
        }
        attestation.verify(&policy_fingerprint, &self.policy_provenance_hash, verifying_key, now_unix_s)?;
        Ok(NeurosemanticPolicyProvenanceBinding {
            policy_provenance_ref: self.policy_provenance_ref.clone(),
            policy_provenance_hash: self.policy_provenance_hash.clone(),
            handling_policy_fingerprint: policy_fingerprint,
            authority_ref: attestation.authority_ref.clone(),
            key_ref: attestation.key_ref.clone(),
            attestation_fingerprint: attestation.fingerprint_for_attestation()?,
            attestation_expires_at_unix_s: attestation.expires_at_unix_s,
        })
    }

    pub fn allows_destination(&self, destination_jurisdiction: &str) -> bool {
        self.validates()
            && valid_jurisdiction_id(destination_jurisdiction)
            && self
                .permitted_destination_jurisdictions
                .contains(destination_jurisdiction)
    }

    pub fn allows_action(&self, action: NeurosemanticHandlingAction, now_unix_s: u64) -> bool {
        if !self.validates() {
            return false;
        }

        match action {
            NeurosemanticHandlingAction::Transmit => true,
            NeurosemanticHandlingAction::Persist => self.retention_allows_persistence(now_unix_s),
            NeurosemanticHandlingAction::SecondaryUse(secondary_use) => {
                self.retention_allows_persistence(now_unix_s)
                    && self.permitted_secondary_uses.contains(&secondary_use)
            }
        }
    }

    pub fn retention_allows_persistence(&self, now_unix_s: u64) -> bool {
        match self.retention {
            NeurosemanticRetentionPolicy::Ephemeral => false,
            NeurosemanticRetentionPolicy::UntilUnixS(expires_at) => now_unix_s < expires_at,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeurosemanticDataPolicy {
    pub schema_version: u16,
    pub data_class: NeurosemanticDataClass,
    #[serde(default)]
    pub inference_classes: BTreeSet<NeurosemanticInferenceClass>,
    #[serde(default)]
    pub permitted_purposes: BTreeSet<CommunicationPurpose>,
    #[serde(default)]
    pub handling: NeurosemanticHandlingPolicy,
}

impl Default for NeurosemanticDataPolicy {
    fn default() -> Self {
        Self {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            data_class: NeurosemanticDataClass::Unknown,
            inference_classes: BTreeSet::new(),
            permitted_purposes: BTreeSet::new(),
            handling: NeurosemanticHandlingPolicy::default(),
        }
    }
}

impl NeurosemanticDataPolicy {
    pub fn transportable(&self) -> bool {
        matches!(
            self.data_class,
            NeurosemanticDataClass::DerivedNeuralFeature
                | NeurosemanticDataClass::SemanticRepresentation
                | NeurosemanticDataClass::DecodedClaim
        )
    }

    pub fn validates(&self) -> bool {
        self.schema_version == NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION
            && self.data_class != NeurosemanticDataClass::Unknown
            && !self.inference_classes.is_empty()
            && !self.inference_classes.contains(&NeurosemanticInferenceClass::Unknown)
            && !self.permitted_purposes.is_empty()
            && self.handling.validates()
    }

    pub fn allows_purpose(&self, purpose: CommunicationPurpose) -> bool {
        self.validates() && self.permitted_purposes.contains(&purpose)
    }

    /// Secondary uses that explicitly infer affective state or identity must also
    /// be represented in the packet's declared inference classes. This prevents a
    /// downstream-use flag from escalating the inference capability beyond the
    /// data product's declared boundary.
    fn allows_secondary_inference(&self, secondary_use: NeurosemanticSecondaryUse) -> bool {
        match secondary_use {
            NeurosemanticSecondaryUse::AffectiveInference => {
                self.inference_classes
                    .contains(&NeurosemanticInferenceClass::AffectiveState)
            }
            NeurosemanticSecondaryUse::IdentityInference => {
                self.inference_classes
                    .contains(&NeurosemanticInferenceClass::Identity)
            }
            _ => true,
        }
    }

    pub fn allows_handling(
        &self,
        destination_jurisdiction: &str,
        action: NeurosemanticHandlingAction,
        now_unix_s: u64,
    ) -> bool {
        self.validates()
            && match action {
                NeurosemanticHandlingAction::SecondaryUse(secondary_use) => {
                    self.allows_secondary_inference(secondary_use)
                }
                _ => true,
            }
            && self.handling.allows_destination(destination_jurisdiction)
            && self.handling.allows_action(action, now_unix_s)
    }
}

/// Derived representations only. Raw neural samples are intentionally absent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NeurosemanticPayload {
    SemanticGraph(Vec<u8>),
    Hypervector(Vec<i8>),
    /// A representation that has already crossed an explicit decode boundary.
    DecodedClaim(Vec<u8>),
    /// Legacy opaque representation retained for deserialization compatibility.
    /// It cannot be authorized under a declared v1 data policy because its
    /// semantic data class is not machine-verifiable from the enum variant.
    StructuredRepresentation(Vec<u8>),
    DerivedNeuralFeature(Vec<f32>),
}

impl NeurosemanticPayload {
    /// Return the data class that this concrete payload variant can authorize
    /// without inspecting opaque user-defined bytes.
    pub fn intrinsic_data_class(&self) -> Option<NeurosemanticDataClass> {
        match self {
            Self::SemanticGraph(_) | Self::Hypervector(_) => {
                Some(NeurosemanticDataClass::SemanticRepresentation)
            }
            Self::DecodedClaim(_) => Some(NeurosemanticDataClass::DecodedClaim),
            Self::StructuredRepresentation(_) => None,
            Self::DerivedNeuralFeature(_) => Some(NeurosemanticDataClass::DerivedNeuralFeature),
        }
    }
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
    /// Effective timestamp for a revocation record. A revoked lease must carry this value.
    #[serde(default)]
    pub revoked_at_unix_s: Option<u64>,
}

impl CognitiveConsentLease {
    /// Deserialize a persisted lease only after enforcing the serialized size ceiling.
    /// Untrusted callers should use this entry point rather than unbounded `serde_json`
    /// deserialization.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic consent lease JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let lease: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic consent lease JSON: {error}"))?;
        lease.validate()?;
        Ok(lease)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !valid_identifier(&self.lease_id)
            || !valid_identifier(&self.subject_id)
            || !valid_identifier(&self.peer_id)
            || self.issued_at_unix_s >= self.expires_at_unix_s
        {
            return Err("consent lease identity or time bounds are invalid".into());
        }
        if self.revoked && self.revoked_at_unix_s.is_none() {
            return Err("revoked consent leases require an explicit effective timestamp".into());
        }
        if let Some(revoked_at) = self.revoked_at_unix_s {
            if revoked_at < self.issued_at_unix_s {
                return Err("consent revocation effective time cannot precede lease issuance".into());
            }
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
            || self
                .revoked_at_unix_s
                .is_some_and(|revoked_at| now_unix_s >= revoked_at)
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
    /// Deserialize an untrusted packet only after enforcing the raw byte ceiling.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "neurosemantic packet JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let packet: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("neurosemantic packet JSON: {error}"))?;
        packet.validate_integrity()?;
        Ok(packet)
    }

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
        let sender_id = sender_id.into();
        let recipient_id = recipient_id.into();
        if !valid_identifier(&sender_id) || !valid_identifier(&recipient_id) {
            return Err("packet sender and recipient identifiers are invalid or oversized".into());
        }
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err("confidence must be finite and in [0, 1]".into());
        }
        validate_payload(&payload)?;

        let mut packet = Self {
            protocol_version: NEUROSEMANTIC_PROTOCOL_VERSION,
            sequence,
            sender_id,
            recipient_id,
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
        if !data_policy.validates() || !data_policy.transportable() {
            return Err("neurosemantic data policy is invalid or not transportable in protocol v1".into());
        }
        if payload.intrinsic_data_class() != Some(data_policy.data_class) {
            return Err("neurosemantic payload type does not match its declared data class".into());
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
            || !valid_identifier(&self.sender_id)
            || !valid_identifier(&self.recipient_id)
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
        if self.data_policy.validates()
            && self.payload.intrinsic_data_class() != Some(self.data_policy.data_class)
        {
            return Err("neurosemantic payload type does not match its declared data class".into());
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
    /// Deserialize an authorized message only after enforcing the raw byte ceiling
    /// and re-running packet + consent validation.
    pub fn from_json_bytes(
        bytes: &[u8],
        lease: &CognitiveConsentLease,
        now_unix_s: u64,
    ) -> Result<Self, String> {
        if bytes.len() > MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "authorized neurosemantic message JSON exceeds {} bytes",
                MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let message: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("authorized neurosemantic message JSON: {error}"))?;
        message.validate(lease, now_unix_s)?;
        Ok(message)
    }

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

        if !self.packet.data_policy.transportable() {
            return Err("packet data class is not transportable in neurosemantic protocol v1".into());
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

    /// Apply the declared handling policy to a concrete downstream request.
    /// Mycelix/another policy authority must supply the deployment context and
    /// independently authenticate the policy provenance; this method does not
    /// establish legal compliance or signed authorization by itself.
    pub fn validate_for_handling(
        &self,
        lease: &CognitiveConsentLease,
        provenance: &NeurosemanticPolicyProvenanceBinding,
        destination_jurisdiction: &str,
        action: NeurosemanticHandlingAction,
        now_unix_s: u64,
    ) -> Result<(), String> {
        if provenance.policy_provenance_ref
            != self.packet.data_policy.handling.policy_provenance_ref
            || provenance.policy_provenance_hash
                != self.packet.data_policy.handling.policy_provenance_hash
            || provenance.handling_policy_fingerprint
                != self.packet.data_policy.handling.fingerprint()?
            || provenance.authority_ref.is_empty()
            || provenance.key_ref.is_empty()
            || now_unix_s >= provenance.attestation_expires_at_unix_s
            || provenance.attestation_fingerprint.is_empty()
        {
            return Err("policy provenance capability does not match the current policy or is expired".into());
        }
        self.validate(lease, now_unix_s)?;
        if !self.packet.data_policy.allows_handling(
            destination_jurisdiction,
            action,
            now_unix_s,
        ) {
            return Err(
                "neurosemantic handling request exceeds destination, retention, or secondary-use policy"
                    .into(),
            );
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
    expires_at_unix_s: u64,
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
        self.prune_expired(now_unix_s);
        self.observe_with_expiry(message, lease.expires_at_unix_s)
    }

    /// Explicitly reclaim replay state whose authorization lease has expired.
    /// This makes the bounded tracker a reclaimable resource rather than a
    /// permanent accumulation vector across short-lived leases.
    pub fn prune_expired(&mut self, now_unix_s: u64) -> usize {
        let before = self.latest.len();
        self.latest
            .retain(|_, state| now_unix_s < state.expires_at_unix_s);
        before - self.latest.len()
    }

    fn observe_with_expiry(
        &mut self,
        message: &AuthorizedNeurosemanticMessage,
        expires_at_unix_s: u64,
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
                        expires_at_unix_s,
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
                        expires_at_unix_s,
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

pub fn compute_policy_provenance_hash(provenance_ref: &str, record_bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(NEUROSEMANTIC_POLICY_PROVENANCE_DOMAIN);
    hasher.update(&(provenance_ref.len() as u64).to_le_bytes());
    hasher.update(provenance_ref.as_bytes());
    hasher.update(&(record_bytes.len() as u64).to_le_bytes());
    hasher.update(record_bytes);
    hasher.finalize().to_hex().to_string()
}

fn valid_identifier(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= MAX_NEUROSEMANTIC_ID_BYTES
}

fn valid_blake3_digest(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        && value.bytes().any(|byte| byte != b'0')
}

fn valid_jurisdiction_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 2
        && bytes[0].is_ascii_uppercase()
        && bytes[1].is_ascii_uppercase()
        && bytes.len() <= MAX_NEUROSEMANTIC_JURISDICTION_ID_BYTES
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
            handling: NeurosemanticHandlingPolicy {
                schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
                policy_provenance_ref: "synthetic-policy-record-1".into(),
            policy_provenance_hash: compute_policy_provenance_hash(
                "synthetic-policy-record-1",
                b"synthetic-policy-record-1",
            ),
            origin_jurisdiction: "ZA".into(),
                permitted_destination_jurisdictions: BTreeSet::from(["ZA".into()]),
                permitted_secondary_uses: BTreeSet::new(),
                retention: NeurosemanticRetentionPolicy::UntilUnixS(200),
            },
        }
    }

    fn policy_provenance_binding() -> NeurosemanticPolicyProvenanceBinding {
        let policy = semantic_policy();
        let (attestation, verifying_key) = authority_attestation(&policy);
        policy
            .handling
            .bind_policy_provenance_with_attestation(
                b"synthetic-policy-record-1",
                &attestation,
                &verifying_key,
                150,
            )
            .unwrap()
    }

    fn authority_attestation(
        policy: &NeurosemanticDataPolicy,
    ) -> (NeurosemanticPolicyAuthorityAttestation, VerifyingKey) {
        use ed25519_dalek::{Signer, SigningKey};
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let fingerprint = policy.handling.fingerprint_for_attestation().unwrap();
        let hash = policy.handling.policy_provenance_hash.clone();
        let message = NeurosemanticPolicyAuthorityAttestation::message_bytes(
            "mycelix-policy-authority",
            "test-key-1",
            &fingerprint,
            &hash,
            100,
            2_000,
        )
        .unwrap();
        let signature = signing_key.sign(&message).to_bytes().to_vec();
        (
            NeurosemanticPolicyAuthorityAttestation {
                schema_version: NEUROSEMANTIC_POLICY_ATTESTATION_SCHEMA_VERSION,
                authority_ref: "mycelix-policy-authority".into(),
                key_ref: "test-key-1".into(),
                handling_policy_fingerprint: fingerprint,
                policy_provenance_hash: hash,
                issued_at_unix_s: 100,
                expires_at_unix_s: 2_000,
                signature,
            },
            signing_key.verifying_key(),
        )
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
            revoked_at_unix_s: None,
        }
    }

    #[test]
    fn v2_transport_rejects_raw_and_personalized_data_classes() {
        let mut policy = semantic_policy();
        policy.data_class = NeurosemanticDataClass::RawNeuralRecording;
        assert!(!policy.transportable());
        let mut policy = semantic_policy();
        policy.data_class = NeurosemanticDataClass::PersonalizedDecoderModel;
        assert!(!policy.transportable());
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
    fn legacy_data_policy_schema_is_rejected_by_v3_validator() {
        let policy = semantic_policy();
        let mut value = serde_json::to_value(&policy).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert(
            "schema_version".into(),
            serde_json::json!(NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION - 1),
        );
        let restored: NeurosemanticDataPolicy = serde_json::from_value(value).unwrap();
        assert!(!restored.validates());
        assert!(!restored.handling.validates());
    }

    #[test]
    fn data_class_schema_is_stable() {
        for (class, expected) in [
            (NeurosemanticDataClass::Unknown, "Unknown"),
            (NeurosemanticDataClass::RawNeuralRecording, "RawNeuralRecording"),
            (NeurosemanticDataClass::DerivedNeuralFeature, "DerivedNeuralFeature"),
            (NeurosemanticDataClass::SemanticRepresentation, "SemanticRepresentation"),
            (NeurosemanticDataClass::DecodedClaim, "DecodedClaim"),
            (NeurosemanticDataClass::PersonalizedDecoderModel, "PersonalizedDecoderModel"),
        ] {
            assert_eq!(serde_json::to_string(&class).unwrap(), format!("\"{expected}\""));
        }
    }

    #[test]
    fn inference_class_schema_is_stable() {
        for (class, expected) in [
            (NeurosemanticInferenceClass::Unknown, "Unknown"),
            (NeurosemanticInferenceClass::SignalPattern, "SignalPattern"),
            (NeurosemanticInferenceClass::UnitPattern, "UnitPattern"),
            (NeurosemanticInferenceClass::LinguisticContent, "LinguisticContent"),
            (NeurosemanticInferenceClass::SemanticContent, "SemanticContent"),
            (NeurosemanticInferenceClass::AffectiveState, "AffectiveState"),
            (NeurosemanticInferenceClass::Intent, "Intent"),
            (NeurosemanticInferenceClass::Identity, "Identity"),
        ] {
            assert_eq!(serde_json::to_string(&class).unwrap(), format!("\"{expected}\""));
        }
    }

    #[test]
    fn authority_attestation_rejects_tampering_wrong_key_and_policy() {
        let policy = semantic_policy();
        let (mut attestation, verifying_key) = authority_attestation(&policy);
        attestation.signature[0] ^= 0x01;
        assert!(attestation
            .verify(
                &policy.handling.fingerprint_for_attestation().unwrap(),
                &policy.handling.policy_provenance_hash,
                &verifying_key,
                150
            )
            .is_err());

        let (good_attestation, _) = authority_attestation(&policy);
        let wrong_key = ed25519_dalek::SigningKey::from_bytes(&[8u8; 32]);
        assert!(good_attestation
            .verify(
                &policy.handling.fingerprint_for_attestation().unwrap(),
                &policy.handling.policy_provenance_hash,
                &wrong_key.verifying_key(),
                150
            )
            .is_err());

        let mut changed_policy = policy.clone();
        changed_policy.handling.retention = NeurosemanticRetentionPolicy::Ephemeral;
        assert!(good_attestation
            .verify(
                &changed_policy.handling.fingerprint_for_attestation().unwrap(),
                &changed_policy.handling.policy_provenance_hash,
                &verifying_key,
                150
            )
            .is_err());
    }

    #[test]
    fn authority_attestation_bounded_json_parser_fails_closed() {
        let policy = semantic_policy();
        let (attestation, _) = authority_attestation(&policy);
        let encoded = serde_json::to_vec(&attestation).unwrap();
        assert_eq!(
            NeurosemanticPolicyAuthorityAttestation::from_json_bytes(&encoded).unwrap(),
            attestation
        );

        let mut malformed = attestation.clone();
        malformed.signature = vec![0u8; MAX_NEUROSEMANTIC_AUTHORITY_SIGNATURE_BYTES - 1];
        let malformed_bytes = serde_json::to_vec(&malformed).unwrap();
        assert!(NeurosemanticPolicyAuthorityAttestation::from_json_bytes(&malformed_bytes).is_err());

        let oversized = vec![b' '; MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES + 1];
        assert!(NeurosemanticPolicyAuthorityAttestation::from_json_bytes(&oversized).is_err());
    }

    #[test]
    fn authority_attestation_fingerprint_binds_exact_signature() {
        let policy = semantic_policy();
        let (attestation, verifying_key) = authority_attestation(&policy);
        let fingerprint = attestation.fingerprint_for_attestation().unwrap();
        let mut tampered = attestation.clone();
        tampered.signature[0] ^= 0x01;
        assert_ne!(
            tampered.fingerprint_for_attestation().unwrap(),
            fingerprint
        );
        assert!(tampered
            .verify(
                &policy.handling.fingerprint_for_attestation().unwrap(),
                &policy.handling.policy_provenance_hash,
                &verifying_key,
                150
            )
            .is_err());
    }

    #[test]
    fn authority_attestation_verifies_exact_policy_and_provenance() {
        let policy = semantic_policy();
        let (attestation, verifying_key) = authority_attestation(&policy);
        assert!(attestation
            .verify(
                &policy.handling.fingerprint_for_attestation().unwrap(),
                &policy.handling.policy_provenance_hash,
                &verifying_key,
                150
            )
            .is_ok());

        let mut changed = attestation.clone();
        changed.policy_provenance_hash = compute_policy_provenance_hash(
            "synthetic-policy-record-1",
            b"synthetic-policy-record-2",
        );
        assert!(changed
            .verify(
                &policy.handling.fingerprint_for_attestation().unwrap(),
                &policy.handling.policy_provenance_hash,
                &verifying_key,
                150
            )
            .is_err());
    }

    #[test]
    fn authority_attestation_expires_and_invalidates_binding() {
        let policy = semantic_policy();
        let (attestation, verifying_key) = authority_attestation(&policy);
        let binding = policy
            .handling
            .bind_policy_provenance_with_attestation(
                b"synthetic-policy-record-1",
                &attestation,
                &verifying_key,
                150,
            )
            .unwrap();
        assert_eq!(binding.authority_ref(), "mycelix-policy-authority");
        assert!(policy
            .handling
            .bind_policy_provenance_with_attestation(
                b"synthetic-policy-record-1",
                &attestation,
                &verifying_key,
                2_000,
            )
            .is_err());
    }

    #[test]
    fn handling_rejects_stale_or_unattested_provenance_capability() {
        let policy = semantic_policy();
        let (attestation, verifying_key) = authority_attestation(&policy);
        let original_binding = policy
            .handling
            .bind_policy_provenance_with_attestation(
                b"synthetic-policy-record-1",
                &attestation,
                &verifying_key,
                150,
            )
            .unwrap();

        let packet = NeurosemanticPacket::new_with_policy(
            19,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            policy,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: lease().consent_epoch,
            lease_id: lease().lease_id,
        };
        assert!(message
            .validate_for_handling(
                &lease(),
                &original_binding,
                "ZA",
                NeurosemanticHandlingAction::Transmit,
                150,
            )
            .is_ok());

        let mut expired = original_binding.clone();
        expired.attestation_expires_at_unix_s = 150;
        assert!(message
            .validate_for_handling(
                &lease(),
                &expired,
                "ZA",
                NeurosemanticHandlingAction::Transmit,
                150,
            )
            .is_err());
    }

    #[test]
    fn handling_policy_defaults_to_deny() {
        let policy = NeurosemanticHandlingPolicy::default();
        assert!(!policy.validates());
        assert!(!policy.allows_destination("ZA"));
        assert!(!policy.allows_action(NeurosemanticHandlingAction::Persist, 1));
        assert!(!policy.allows_action(
            NeurosemanticHandlingAction::SecondaryUse(NeurosemanticSecondaryUse::Research),
            1
        ));
    }

    #[test]
    fn handling_policy_provenance_reference_is_bounded() {
        let mut policy = NeurosemanticHandlingPolicy {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            policy_provenance_ref: "x".repeat(MAX_NEUROSEMANTIC_ID_BYTES + 1),
            policy_provenance_hash: compute_policy_provenance_hash(
                "synthetic-policy-record-1",
                b"synthetic-policy-record-1",
            ),
            origin_jurisdiction: "ZA".into(),
            permitted_destination_jurisdictions: BTreeSet::from(["ZA".into()]),
            permitted_secondary_uses: BTreeSet::new(),
            retention: NeurosemanticRetentionPolicy::Ephemeral,
        };
        assert!(!policy.validates());
        policy.policy_provenance_ref = "synthetic-policy-record-1".into();
        assert!(policy.validates());
    }

    #[test]
    fn handling_policy_requires_machine_verifiable_provenance_hash() {
        let mut policy = NeurosemanticHandlingPolicy {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            policy_provenance_ref: "synthetic-policy-record-1".into(),
            policy_provenance_hash: String::new(),
            origin_jurisdiction: "ZA".into(),
            permitted_destination_jurisdictions: BTreeSet::from(["ZA".into()]),
            permitted_secondary_uses: BTreeSet::new(),
            retention: NeurosemanticRetentionPolicy::Ephemeral,
        };
        assert!(!policy.validates());
        policy.policy_provenance_hash = compute_policy_provenance_hash(
            "synthetic-policy-record-1",
            b"synthetic-policy-record-1",
        );
        assert!(policy.validates());
        policy.policy_provenance_hash = "not-a-blake3-digest".into();
        assert!(!policy.validates());
        policy.policy_provenance_hash = "A".repeat(64);
        assert!(!policy.validates());
    }

    #[test]
    fn handling_policy_provenance_digest_verifies_exact_record_and_reference() {
        let policy = semantic_policy().handling;
        assert!(policy.verify_policy_record_binding_bytes(b"synthetic-policy-record-1"));
        assert!(!policy.verify_policy_record_binding_bytes(b"synthetic-policy-record-2"));
        assert!(!policy.verify_policy_record_binding_bytes(
            &vec![b'x'; MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES + 1]
        ));
    }

    #[test]
    fn handling_policy_provenance_reference_cannot_be_swapped_under_existing_digest() {
        let mut policy = semantic_policy().handling;
        assert!(policy.verify_policy_record_binding_bytes(b"synthetic-policy-record-1"));
        policy.policy_provenance_ref = "synthetic-policy-record-2".into();
        assert!(!policy.verify_policy_record_binding_bytes(b"synthetic-policy-record-1"));
    }

    #[test]
    fn handling_policy_provenance_binding_becomes_stale_after_policy_mutation() {
        let policy = semantic_policy();
        let original_binding = {
            let (attestation, verifying_key) = authority_attestation(&policy);
            policy
                .handling
                .bind_policy_provenance_with_attestation(
                    b"synthetic-policy-record-1",
                    &attestation,
                    &verifying_key,
                    150,
                )
                .unwrap()
        };

        let mut mutated = policy.clone();
        mutated
            .handling
            .permitted_secondary_uses
            .insert(NeurosemanticSecondaryUse::Research);

        let (fresh_attestation, fresh_key) = authority_attestation(&mutated);
        let fresh_binding = mutated
            .handling
            .bind_policy_provenance_with_attestation(
                b"synthetic-policy-record-1",
                &fresh_attestation,
                &fresh_key,
                150,
            )
            .unwrap();

        let mut packet = NeurosemanticPacket::new_with_policy(
            18,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            NeurosemanticDataPolicy {
                handling: mutated,
                ..semantic_policy()
            },
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();

        packet.refresh_hashes().unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: lease().consent_epoch,
            lease_id: lease().lease_id,
        };
        assert!(message
            .validate_for_handling(
                &lease(),
                &fresh_binding,
                "ZA",
                NeurosemanticHandlingAction::SecondaryUse(NeurosemanticSecondaryUse::Research),
                150,
            )
            .is_ok());
        assert!(message
            .validate_for_handling(
                &lease(),
                &original_binding,
                "ZA",
                NeurosemanticHandlingAction::SecondaryUse(NeurosemanticSecondaryUse::Research),
                150,
            )
            .is_err());
    }

    #[test]
    fn handling_policy_requires_provenance_reference() {
        let mut policy = NeurosemanticHandlingPolicy {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            policy_provenance_ref: String::new(),
            policy_provenance_hash: String::new(),
            origin_jurisdiction: "ZA".into(),
            permitted_destination_jurisdictions: BTreeSet::from(["ZA".into()]),
            permitted_secondary_uses: BTreeSet::new(),
            retention: NeurosemanticRetentionPolicy::Ephemeral,
        };
        assert!(!policy.validates());
        policy.policy_provenance_ref = "synthetic-policy-record-1".into();
        assert!(policy.validates());
    }

    #[test]
    fn handling_policy_requires_explicit_destination_and_retention() {
        let policy = NeurosemanticHandlingPolicy {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            policy_provenance_ref: "synthetic-policy-record-1".into(),
            policy_provenance_hash: compute_policy_provenance_hash(
                "synthetic-policy-record-1",
                b"synthetic-policy-record-1",
            ),
            origin_jurisdiction: "ZA".into(),
            permitted_destination_jurisdictions: BTreeSet::from(["ZA".into(), "GB".into()]),
            permitted_secondary_uses: BTreeSet::from([NeurosemanticSecondaryUse::Research]),
            retention: NeurosemanticRetentionPolicy::UntilUnixS(200),
        };
        assert!(policy.validates());
        assert!(policy.allows_destination("GB"));
        assert!(!policy.allows_destination("US"));
        assert!(policy.allows_action(NeurosemanticHandlingAction::Persist, 150));
        assert!(!policy.allows_action(NeurosemanticHandlingAction::Persist, 200));
        assert!(policy.allows_action(
            NeurosemanticHandlingAction::SecondaryUse(NeurosemanticSecondaryUse::Research),
            150
        ));
        assert!(!policy.allows_action(
            NeurosemanticHandlingAction::SecondaryUse(NeurosemanticSecondaryUse::ModelTraining),
            150
        ));
    }

    #[test]
    fn handling_policy_rejects_noncanonical_jurisdictions() {
        let mut policy = NeurosemanticHandlingPolicy {
            schema_version: NEUROSEMANTIC_DATA_POLICY_SCHEMA_VERSION,
            policy_provenance_ref: "synthetic-policy-record-1".into(),
            policy_provenance_hash: compute_policy_provenance_hash(
                "synthetic-policy-record-1",
                b"synthetic-policy-record-1",
            ),
            origin_jurisdiction: "za".into(),
            permitted_destination_jurisdictions: BTreeSet::from(["za".into()]),
            permitted_secondary_uses: BTreeSet::new(),
            retention: NeurosemanticRetentionPolicy::Ephemeral,
        };
        assert!(!policy.validates());
        policy.origin_jurisdiction = "ZA".into();
        policy.permitted_destination_jurisdictions = BTreeSet::from(["ZA".into()]);
        assert!(policy.validates());
    }

    #[test]
    fn inference_sensitive_secondary_uses_cannot_escalate_declared_inference_classes() {
        let lease = lease();
        let mut policy = semantic_policy();
        policy.handling.permitted_secondary_uses =
            BTreeSet::from([NeurosemanticSecondaryUse::AffectiveInference]);
        let packet = NeurosemanticPacket::new_with_policy(
            109,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            policy.clone(),
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: lease.consent_epoch,
            lease_id: lease.lease_id.clone(),
        };
        assert!(message
            .validate_for_handling(
                &lease,
                &policy_provenance_binding(),
                "ZA",
                NeurosemanticHandlingAction::SecondaryUse(
                    NeurosemanticSecondaryUse::AffectiveInference
                ),
                150
            )
            .is_err());

        policy.inference_classes =
            BTreeSet::from([NeurosemanticInferenceClass::AffectiveState]);
        policy.permitted_secondary_uses =
            BTreeSet::from([NeurosemanticSecondaryUse::AffectiveInference]);
        let packet = NeurosemanticPacket::new_with_policy(
            110,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            policy,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: lease.consent_epoch,
            lease_id: lease.lease_id,
        };
        assert!(message
            .validate_for_handling(
                &lease,
                &policy_provenance_binding(),
                "ZA",
                NeurosemanticHandlingAction::SecondaryUse(
                    NeurosemanticSecondaryUse::AffectiveInference
                ),
                150
            )
            .is_ok());
    }

    #[test]
    fn authorized_handling_respects_secondary_use_jurisdiction_and_retention() {
        let lease = lease();
        let packet = NeurosemanticPacket::new_with_policy(
            107,
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
            consent_epoch: lease.consent_epoch,
            lease_id: lease.lease_id.clone(),
        };

        assert!(message
            .validate_for_handling(
                &lease,
                &policy_provenance_binding(),
                "ZA",
                NeurosemanticHandlingAction::Transmit,
                150
            )
            .is_ok());
        assert!(message
            .validate_for_handling(
                &lease,
                &policy_provenance_binding(),
                "GB",
                NeurosemanticHandlingAction::Transmit,
                150
            )
            .is_err());
        assert!(message
            .validate_for_handling(
                &lease,
                &policy_provenance_binding(),
                "ZA",
                NeurosemanticHandlingAction::Persist,
                200
            )
            .is_err());

        let mut secondary_policy = semantic_policy();
        secondary_policy.handling.permitted_secondary_uses =
            BTreeSet::from([NeurosemanticSecondaryUse::Research]);
        let packet = NeurosemanticPacket::new_with_policy(
            108,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            secondary_policy,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: lease.consent_epoch,
            lease_id: lease.lease_id,
        };
        assert!(message
            .validate_for_handling(
                &lease,
                &policy_provenance_binding(),
                "ZA",
                NeurosemanticHandlingAction::SecondaryUse(
                    NeurosemanticSecondaryUse::Research
                ),
                150
            )
            .is_ok());
    }

    #[test]
    fn data_policy_serialization_roundtrips() {
        let policy = semantic_policy();
        let encoded = serde_json::to_vec(&policy).unwrap();
        let decoded: NeurosemanticDataPolicy = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, policy);
    }

    #[test]
    fn bounded_json_parsers_reject_oversized_artifacts() {
        let oversized = vec![b' '; MAX_NEUROSEMANTIC_SERIALIZED_ARTIFACT_BYTES + 1];
        assert!(NeurosemanticPacket::from_json_bytes(&oversized).is_err());
        assert!(CognitiveConsentLease::from_json_bytes(&oversized,).is_err());
        let lease = lease();
        assert!(AuthorizedNeurosemanticMessage::from_json_bytes(&oversized, &lease, 150).is_err());
    }

    #[test]
    fn identifiers_are_bounded_and_nonempty() {
        let oversized = "x".repeat(MAX_NEUROSEMANTIC_ID_BYTES + 1);
        let packet = NeurosemanticPacket::new(
            105,
            oversized,
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        );
        assert!(packet.is_err());

        let mut l = lease();
        l.peer_id.clear();
        assert!(l.validate().is_err());
        l.peer_id = "x".repeat(MAX_NEUROSEMANTIC_ID_BYTES + 1);
        assert!(l.validate().is_err());
    }

    #[test]
    fn bounded_json_parsers_roundtrip_valid_artifacts() {
        let lease = lease();
        let encoded_lease = serde_json::to_vec(&lease).unwrap();
        assert_eq!(CognitiveConsentLease::from_json_bytes(&encoded_lease).unwrap(), lease);

        let packet = NeurosemanticPacket::new_with_policy(
            106,
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
        ).unwrap();
        let encoded_packet = serde_json::to_vec(&packet).unwrap();
        assert_eq!(NeurosemanticPacket::from_json_bytes(&encoded_packet).unwrap(), packet);

        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: lease.consent_epoch,
            lease_id: lease.lease_id.clone(),
        };
        let encoded_message = serde_json::to_vec(&message).unwrap();
        assert_eq!(AuthorizedNeurosemanticMessage::from_json_bytes(&encoded_message, &lease, 150).unwrap(), message);
    }
    #[test]
    fn policy_binds_payload_to_declared_data_class() {
        let result = NeurosemanticPacket::new_with_policy(
            101,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            NeurosemanticDataPolicy {
                data_class: NeurosemanticDataClass::DerivedNeuralFeature,
                ..semantic_policy()
            },
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        );
        assert!(result.is_err());

        let mut packet = NeurosemanticPacket::new_with_policy(
            102,
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
        packet.data_policy.data_class = NeurosemanticDataClass::DecodedClaim;
        packet.refresh_hashes().unwrap();
        assert!(packet.validate_integrity().is_err());
    }

    #[test]
    fn opaque_legacy_structured_representation_cannot_cross_policy_boundary() {
        assert_eq!(
            NeurosemanticPayload::StructuredRepresentation(b"hello".to_vec())
                .intrinsic_data_class(),
            None
        );
        assert!(NeurosemanticPacket::new_with_policy(
            103,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            semantic_policy(),
            0.5,
            NeurosemanticPayload::StructuredRepresentation(b"hello".to_vec()),
        )
        .is_err());
    }

    #[test]
    fn decoded_claim_is_a_typed_transport_payload() {
        let policy = NeurosemanticDataPolicy {
            data_class: NeurosemanticDataClass::DecodedClaim,
            ..semantic_policy()
        };
        let packet = NeurosemanticPacket::new_with_policy(
            104,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            policy,
            0.5,
            NeurosemanticPayload::DecodedClaim(b"claim".to_vec()),
        )
        .unwrap();
        assert!(packet.validate_integrity().is_ok());
    }

    #[test]
    fn legacy_packet_data_policy_defaults_to_unknown_and_cannot_authorize() {
        let packet = NeurosemanticPacket::new(
            1,
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
        assert_eq!(packet.data_policy.data_class, NeurosemanticDataClass::Unknown);
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: 7,
            lease_id: "lease-1".into(),
        };
        assert!(message.validate(&lease(), 150).is_err());
    }

    #[test]
    fn packet_policy_purpose_must_match_packet_purpose() {
        let mut policy = semantic_policy();
        policy.permitted_purposes = BTreeSet::from([CommunicationPurpose::Research]);
        let packet = NeurosemanticPacket::new_with_policy(
            2,
            "peer",
            "subject",
            CommunicationPurpose::HumanCollaboration,
            CognitiveChannel::Semantic,
            ChannelDirection::Write,
            RepresentationFamily::Hdc,
            CognitiveSensitivity::Private,
            policy,
            0.5,
            NeurosemanticPayload::Hypervector(vec![1, -1]),
        )
        .unwrap();
        let message = AuthorizedNeurosemanticMessage {
            packet,
            consent_epoch: 7,
            lease_id: "lease-1".into(),
        };
        assert!(message.validate(&lease(), 150).is_err());
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
    fn policy_provenance_ref_is_bound_to_packet_integrity() {
        let mut packet = NeurosemanticPacket::new_with_policy(
            17,
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
        assert!(packet.validate_integrity().is_ok());
        packet.data_policy.handling.policy_provenance_ref = "synthetic-policy-record-2".into();
        assert!(packet.validate_integrity().is_err());
    }

    #[test]
    fn policy_provenance_hash_is_bound_to_packet_integrity() {
        let mut packet = NeurosemanticPacket::new_with_policy(
            17,
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
        assert!(packet.validate_integrity().is_ok());
        packet.data_policy.handling.policy_provenance_hash = compute_policy_provenance_hash(
            "synthetic-policy-record-1",
            b"synthetic-policy-record-2",
        );
        assert!(packet.validate_integrity().is_err());
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
            NeurosemanticPayload::Hypervector(vec![1, -1, 1]),
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
        revoked.revoked_at_unix_s = Some(150);
        assert!(tracker
            .observe_authorized(&message, &revoked, 150)
            .is_err());

        let mut malformed_revoked = base.clone();
        malformed_revoked.revoked = true;
        malformed_revoked.revoked_at_unix_s = None;
        assert!(malformed_revoked.validate().is_err());
    }

    #[test]
    fn replay_state_is_reclaimed_after_lease_expiry() {
        let base = lease();
        let packet = NeurosemanticPacket::new_with_policy(
            16,
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
        assert_eq!(tracker.prune_expired(199), 0);
        assert_eq!(tracker.prune_expired(200), 1);
        assert_eq!(tracker.prune_expired(200), 0);
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
    fn revoked_leases_require_effective_timestamp() {
        let mut l = lease();
        l.revoked = true;
        assert!(l.validate().is_err());
        l.revoked_at_unix_s = Some(150);
        assert!(l.validate().is_ok());
        assert!(l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "peer",
            149
        ));
        assert!(!l.authorizes(
            CognitiveChannel::Semantic,
            ChannelDirection::Read,
            CommunicationPurpose::HumanCollaboration,
            "peer",
            150
        ));
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
