// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Relying-party verification for exact Xenia-authenticated physical-effect payloads.
//!
//! This crate intentionally does **not** depend on `xenia-peer-core`. Symthaea's
//! supply-chain policy disallows ad-hoc Git dependencies, and an opaque in-process
//! Xenia token is not portable across a process/repository boundary anyway.
//!
//! Instead this crate implements the stable signed-receipt wire contract emitted by
//! Xenia and applies independent relying-party policy:
//!
//! ```text
//! raw Xenia receipt bytes + raw physical-effect payload bytes
//!   -> canonical receipt decode
//!   -> current anti-rollback transport-attestor trust
//!   -> exact peer/capability policy
//!   -> Ed25519 verification AND ML-DSA-65 verification
//!   -> exact raw-payload digest/length binding
//!   -> canonical PhysicalEffectEnvelopeV1 decode/re-encode equality
//!   -> send-deadline check
//!   -> VerifiedTransportEnvelope  (opaque; non-authorizing by itself)
//! ```
//!
//! The concrete ML-DSA implementation is deliberately behind
//! [`HybridReceiptSignatureVerifier`] in v0.1. Symthaea does not yet standardize on
//! Xenia's exact RustCrypto `ml-dsa` package in its workspace dependency policy. A
//! production provider must verify both signatures over the same digest; this crate
//! calls the two verification methods independently and requires both to succeed.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use symthaea_authority::Digest32;
use symthaea_iot_device_protocol::PhysicalEffectEnvelopeV1;
use thiserror::Error;

/// Receipt schema shared with Xenia PR #238.
pub const XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA: &str =
    "xenia-authenticated-payload-receipt-v1";
/// Domain separator shared with the Xenia receipt signer.
pub const XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_DOMAIN: &[u8] =
    b"xenia-authenticated-payload-receipt-v1\0";
/// Xenia's current mandatory hybrid receipt/transcript signature suite.
pub const XENIA_HYBRID_SIGNATURE_SUITE: &str = "ed25519-rfc8032+ml-dsa-65-fips204";
/// Application payload type reserved by this integration for physical-effect envelopes.
pub const XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE: u8 = 0x70;
/// Xenia ML-DSA-65 public-key length.
pub const XENIA_ML_DSA_65_PUBLIC_KEY_LEN: usize = 1_952;
/// Xenia ML-DSA-65 signature length.
pub const XENIA_ML_DSA_65_SIGNATURE_LEN: usize = 3_309;
/// Xenia Ed25519 signature length.
pub const XENIA_ED25519_SIGNATURE_LEN: usize = 64;
/// Maximum signed-receipt lifetime accepted by Xenia and this relying party.
pub const MAX_XENIA_RECEIPT_LIFETIME_MS: u64 = 5_000;
/// Maximum physical-effect payload size accepted before decoding.
pub const MAX_XENIA_PHYSICAL_EFFECT_PAYLOAD_BYTES: usize = 64 * 1024;
/// Maximum serialized receipt size accepted before decoding.
pub const MAX_XENIA_RECEIPT_BYTES: usize = 16 * 1024;
/// Maximum configured Xenia transport-attestor entries per trust snapshot.
pub const MAX_TRANSPORT_ATTESTOR_KEYS: usize = 512;
/// Current trust-snapshot schema.
pub const TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION: u16 = 1;
/// Domain separator for one trusted attestor-key record.
pub const TRANSPORT_ATTESTOR_KEY_DOMAIN: &[u8] = b"symthaea-xenia-transport-key-v1\0";
/// Domain separator for anti-rollback transport-trust snapshots.
pub const TRANSPORT_TRUST_SNAPSHOT_DOMAIN: &[u8] = b"symthaea-xenia-transport-trust-v1\0";

/// Remote Xenia role carried by a portable signed receipt.
///
/// Variant order intentionally matches Xenia's `ReceiptPeerRoleV1` bincode layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum XeniaReceiptPeerRoleV1 {
    /// Authenticated remote peer is the controlled/serving host.
    Host,
    /// Authenticated remote peer is the viewer/operator side.
    Viewer,
}

/// Exact Xenia receipt body. Field order and types are part of the v1 bincode contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaAuthenticatedPayloadReceiptBodyV1 {
    /// Stable schema label.
    pub schema: String,
    /// Deployment/operator identifier for the Xenia receipt signer.
    pub attestor_id: String,
    /// Key lifecycle identifier.
    pub key_id: String,
    /// Mandatory hybrid signature-suite label.
    pub signature_algorithm: String,
    /// Commitment to Xenia's opaque authenticated-session evidence.
    pub session_evidence_digest: [u8; 32],
    /// Authenticated remote role.
    pub peer_role: XeniaReceiptPeerRoleV1,
    /// Authenticated hybrid peer-identity fingerprint.
    pub peer_identity_fingerprint: [u8; 32],
    /// Canonical handshake transcript hash.
    pub transcript_hash: [u8; 32],
    /// Exact capability-authenticated Xenia session context.
    pub session_context_hash: [u8; 32],
    /// Authenticated telemetry capability bit.
    pub telemetry_enabled: bool,
    /// Authenticated input/control capability bit.
    pub input_control_enabled: bool,
    /// Exact Xenia application payload type.
    pub payload_type: u8,
    /// Exact opened plaintext length.
    pub payload_len: u32,
    /// BLAKE3-256 of the exact opened plaintext.
    pub payload_digest: [u8; 32],
    /// BLAKE3-256 of the exact sealed Xenia envelope admitted by AEAD/replay checks.
    pub sealed_envelope_digest: [u8; 32],
    /// Receiver-local AEAD/replay acceptance time in Unix milliseconds.
    pub opened_at_unix_ms: u64,
    /// Exclusive receipt expiry in Unix milliseconds.
    pub expires_at_unix_ms: u64,
}

impl XeniaAuthenticatedPayloadReceiptBodyV1 {
    /// Validate the shared Xenia receipt schema independent of key trust.
    pub fn validate_structure(&self) -> Result<(), TransportReceiptError> {
        if self.schema != XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA {
            return Err(TransportReceiptError::UnsupportedReceiptSchema);
        }
        if self.attestor_id.is_empty()
            || self.key_id.is_empty()
            || self.attestor_id.len() > 128
            || self.key_id.len() > 128
            || self.attestor_id.trim() != self.attestor_id
            || self.key_id.trim() != self.key_id
        {
            return Err(TransportReceiptError::InvalidAttestorIdentity);
        }
        if self.signature_algorithm != XENIA_HYBRID_SIGNATURE_SUITE {
            return Err(TransportReceiptError::SignatureSuiteMismatch);
        }
        if self.payload_type != XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE {
            return Err(TransportReceiptError::UnexpectedPayloadType(self.payload_type));
        }
        if self.payload_len == 0
            || self.payload_len as usize > MAX_XENIA_PHYSICAL_EFFECT_PAYLOAD_BYTES
        {
            return Err(TransportReceiptError::InvalidPayloadLength);
        }
        if [
            self.session_evidence_digest,
            self.peer_identity_fingerprint,
            self.transcript_hash,
            self.session_context_hash,
            self.payload_digest,
            self.sealed_envelope_digest,
        ]
        .contains(&[0; 32])
        {
            return Err(TransportReceiptError::ZeroSecurityDigest);
        }
        let lifetime = self
            .expires_at_unix_ms
            .checked_sub(self.opened_at_unix_ms)
            .ok_or(TransportReceiptError::InvalidReceiptLifetime)?;
        if lifetime == 0 || lifetime > MAX_XENIA_RECEIPT_LIFETIME_MS {
            return Err(TransportReceiptError::InvalidReceiptLifetime);
        }
        Ok(())
    }

    /// Canonical bincode-v1 receipt body bytes, matching Xenia's signer.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, TransportReceiptError> {
        self.validate_structure()?;
        bincode::serialize(self).map_err(TransportReceiptError::Encoding)
    }

    /// Exact domain-separated digest covered by both Xenia signatures.
    pub fn signing_digest(&self) -> Result<[u8; 32], TransportReceiptError> {
        let bytes = self.canonical_bytes()?;
        let mut h = blake3::Hasher::new();
        h.update(XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_DOMAIN);
        h.update(&bytes);
        Ok(*h.finalize().as_bytes())
    }
}

/// Exact outer Xenia portable receipt wire shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeniaAuthenticatedPayloadReceiptV1 {
    /// Signed receipt body.
    pub body: XeniaAuthenticatedPayloadReceiptBodyV1,
    /// Mandatory Ed25519 signature over the body signing digest.
    #[serde(with = "BigArray")]
    pub ed25519_signature: [u8; XENIA_ED25519_SIGNATURE_LEN],
    /// Mandatory ML-DSA-65 signature over the identical digest.
    #[serde(with = "BigArray")]
    pub ml_dsa_signature: [u8; XENIA_ML_DSA_65_SIGNATURE_LEN],
}

/// Lifecycle state for one trusted Xenia transport-attestor key identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransportAttestorStatus {
    /// Key may authenticate receipts during its validity window.
    Active,
    /// Key is retired and may never become active again.
    Retired,
    /// Key is revoked; revocation is terminal.
    Revoked,
}

impl TransportAttestorStatus {
    fn tag(self) -> u8 {
        match self {
            Self::Active => 0,
            Self::Retired => 1,
            Self::Revoked => 2,
        }
    }

    fn transition_allowed(self, next: Self) -> bool {
        match self {
            Self::Active => true,
            Self::Retired => matches!(next, Self::Retired | Self::Revoked),
            Self::Revoked => next == Self::Revoked,
        }
    }
}

/// Trusted Xenia transport-attestor identity and relying-party peer policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportAttestorKeyV1 {
    /// Exact attestor service identity expected in a receipt.
    pub attestor_id: String,
    /// Exact key lifecycle identity expected in a receipt.
    pub key_id: String,
    /// Ed25519 verifying-key bytes.
    pub ed25519_public_key: [u8; 32],
    /// Exact ML-DSA-65 verifying-key encoding used by Xenia.
    pub ml_dsa_public_key: Vec<u8>,
    /// Monotonic key lifecycle state.
    pub status: TransportAttestorStatus,
    /// Earliest Unix millisecond at which this key may authenticate a receipt.
    pub not_before_unix_ms: u64,
    /// Exclusive Unix-millisecond expiry for this key.
    pub not_after_unix_ms: u64,
    /// Maximum receipt lifetime permitted under this key.
    pub max_receipt_lifetime_ms: u64,
    /// Authenticated remote role permitted for consequential receipts.
    pub required_peer_role: XeniaReceiptPeerRoleV1,
    /// Exact authenticated peer fingerprints allowed under this key policy.
    pub allowed_peer_fingerprints: BTreeSet<[u8; 32]>,
    /// Whether the sealed Xenia capability surface must permit input/control.
    pub require_input_control: bool,
}

impl TransportAttestorKeyV1 {
    /// Validate one trust record independent of current time.
    pub fn validate(&self) -> Result<(), TransportReceiptError> {
        if self.attestor_id.is_empty()
            || self.key_id.is_empty()
            || self.attestor_id.trim() != self.attestor_id
            || self.key_id.trim() != self.key_id
            || self.attestor_id.len() > 128
            || self.key_id.len() > 128
        {
            return Err(TransportReceiptError::InvalidAttestorIdentity);
        }
        if self.ed25519_public_key == [0; 32]
            || self.ml_dsa_public_key.len() != XENIA_ML_DSA_65_PUBLIC_KEY_LEN
            || self.ml_dsa_public_key.iter().all(|byte| *byte == 0)
        {
            return Err(TransportReceiptError::MalformedTrustedKey);
        }
        if self.not_before_unix_ms >= self.not_after_unix_ms {
            return Err(TransportReceiptError::InvalidTrustedKeyWindow);
        }
        if self.max_receipt_lifetime_ms == 0
            || self.max_receipt_lifetime_ms > MAX_XENIA_RECEIPT_LIFETIME_MS
        {
            return Err(TransportReceiptError::InvalidReceiptLifetime);
        }
        if self.allowed_peer_fingerprints.is_empty()
            || self
                .allowed_peer_fingerprints
                .iter()
                .any(|fingerprint| *fingerprint == [0; 32])
        {
            return Err(TransportReceiptError::InvalidPeerPolicy);
        }
        Ok(())
    }

    /// Domain-separated commitment to immutable key and peer-policy bytes plus lifecycle.
    pub fn digest(&self) -> Result<Digest32, TransportReceiptError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(TRANSPORT_ATTESTOR_KEY_DOMAIN);
        update_string(&mut h, &self.attestor_id);
        update_string(&mut h, &self.key_id);
        h.update(&self.ed25519_public_key);
        h.update(&(self.ml_dsa_public_key.len() as u32).to_be_bytes());
        h.update(&self.ml_dsa_public_key);
        h.update(&[self.status.tag()]);
        h.update(&self.not_before_unix_ms.to_be_bytes());
        h.update(&self.not_after_unix_ms.to_be_bytes());
        h.update(&self.max_receipt_lifetime_ms.to_be_bytes());
        h.update(&[match self.required_peer_role {
            XeniaReceiptPeerRoleV1::Host => 0,
            XeniaReceiptPeerRoleV1::Viewer => 1,
        }]);
        h.update(&[u8::from(self.require_input_control)]);
        h.update(&(self.allowed_peer_fingerprints.len() as u32).to_be_bytes());
        for fingerprint in &self.allowed_peer_fingerprints {
            h.update(fingerprint);
        }
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    fn immutable_policy_eq(&self, other: &Self) -> bool {
        self.attestor_id == other.attestor_id
            && self.key_id == other.key_id
            && self.ed25519_public_key == other.ed25519_public_key
            && self.ml_dsa_public_key == other.ml_dsa_public_key
            && self.not_before_unix_ms == other.not_before_unix_ms
            && self.max_receipt_lifetime_ms == other.max_receipt_lifetime_ms
            && self.required_peer_role == other.required_peer_role
            && self.allowed_peer_fingerprints == other.allowed_peer_fingerprints
            && self.require_input_control == other.require_input_control
    }

    fn active_at(&self, now_unix_ms: u64) -> bool {
        self.status == TransportAttestorStatus::Active
            && now_unix_ms >= self.not_before_unix_ms
            && now_unix_ms < self.not_after_unix_ms
    }
}

/// Public anti-rollback trust snapshot for Xenia receipt attestors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportTrustSnapshotV1 {
    /// Fail-closed schema version.
    pub schema_version: u16,
    /// Monotonic snapshot generation; zero is invalid.
    pub sequence: u64,
    /// Trusted snapshot issue time in Unix milliseconds.
    pub issued_at_unix_ms: u64,
    /// Exclusive snapshot expiry in Unix milliseconds.
    pub expires_at_unix_ms: u64,
    /// Previous snapshot commitment; absent only for generation one.
    pub previous_snapshot_digest: Option<Digest32>,
    /// Complete persistent lifecycle table for attestor/key identities.
    pub keys: Vec<TransportAttestorKeyV1>,
}

impl TransportTrustSnapshotV1 {
    /// Validate snapshot and key structure.
    pub fn validate(&self) -> Result<(), TransportReceiptError> {
        if self.schema_version != TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION {
            return Err(TransportReceiptError::UnsupportedTrustSnapshotSchema);
        }
        if self.sequence == 0 {
            return Err(TransportReceiptError::TrustSequenceZero);
        }
        if self.issued_at_unix_ms >= self.expires_at_unix_ms {
            return Err(TransportReceiptError::InvalidTrustSnapshotWindow);
        }
        if self.keys.is_empty() || self.keys.len() > MAX_TRANSPORT_ATTESTOR_KEYS {
            return Err(TransportReceiptError::InvalidTrustKeyCount);
        }
        if self.sequence == 1 && self.previous_snapshot_digest.is_some() {
            return Err(TransportReceiptError::TrustGenesisHasPredecessor);
        }
        if self.sequence > 1 && self.previous_snapshot_digest.is_none() {
            return Err(TransportReceiptError::TrustSuccessorMissingPredecessor);
        }

        let mut identities = BTreeSet::new();
        for key in &self.keys {
            key.validate()?;
            if !identities.insert((key.attestor_id.clone(), key.key_id.clone())) {
                return Err(TransportReceiptError::DuplicateTrustedKeyIdentity);
            }
        }
        Ok(())
    }

    /// Canonical snapshot commitment independent of input key-vector order.
    pub fn digest(&self) -> Result<Digest32, TransportReceiptError> {
        self.validate()?;
        let mut keys = self.keys.iter().collect::<Vec<_>>();
        keys.sort_by(|left, right| {
            (&left.attestor_id, &left.key_id).cmp(&(&right.attestor_id, &right.key_id))
        });
        let mut h = blake3::Hasher::new();
        h.update(TRANSPORT_TRUST_SNAPSHOT_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.sequence.to_be_bytes());
        h.update(&self.issued_at_unix_ms.to_be_bytes());
        h.update(&self.expires_at_unix_ms.to_be_bytes());
        match self.previous_snapshot_digest {
            Some(Digest32(bytes)) => {
                h.update(&[1]);
                h.update(&bytes);
            }
            None => h.update(&[0]),
        };
        h.update(&(keys.len() as u32).to_be_bytes());
        for key in keys {
            let Digest32(bytes) = key.digest()?;
            h.update(&bytes);
        }
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    fn key_map(&self) -> BTreeMap<(&str, &str), &TransportAttestorKeyV1> {
        self.keys
            .iter()
            .map(|key| ((key.attestor_id.as_str(), key.key_id.as_str()), key))
            .collect()
    }
}

/// Externally retainable anti-rollback anchor for transport trust.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportTrustHead {
    /// Current snapshot generation.
    pub sequence: u64,
    /// Commitment to the exact trust snapshot.
    pub digest: Digest32,
}

/// In-process verified transport-attestor trust state.
///
/// Persist the public snapshot plus [`TransportTrustHead`] separately; this type is
/// intentionally not serializable/deserializable.
#[derive(Debug)]
pub struct TransportTrustRegistry {
    snapshot: TransportTrustSnapshotV1,
    head: TransportTrustHead,
}

impl TransportTrustRegistry {
    /// Create generation one of a transport-trust lineage.
    pub fn genesis(snapshot: TransportTrustSnapshotV1) -> Result<Self, TransportReceiptError> {
        snapshot.validate()?;
        if snapshot.sequence != 1 || snapshot.previous_snapshot_digest.is_some() {
            return Err(TransportReceiptError::NotTrustGenesis);
        }
        let head = TransportTrustHead {
            sequence: 1,
            digest: snapshot.digest()?,
        };
        Ok(Self { snapshot, head })
    }

    /// Verify and accept the immediate successor trust generation.
    pub fn successor(
        &self,
        snapshot: TransportTrustSnapshotV1,
    ) -> Result<Self, TransportReceiptError> {
        snapshot.validate()?;
        let expected = self
            .head
            .sequence
            .checked_add(1)
            .ok_or(TransportReceiptError::TrustSequenceOverflow)?;
        if snapshot.sequence != expected {
            return Err(TransportReceiptError::TrustSequenceNotNext {
                expected,
                proposed: snapshot.sequence,
            });
        }
        if snapshot.previous_snapshot_digest != Some(self.head.digest) {
            return Err(TransportReceiptError::TrustPredecessorMismatch);
        }
        if snapshot.issued_at_unix_ms < self.snapshot.issued_at_unix_ms {
            return Err(TransportReceiptError::TrustIssuedAtRegressed);
        }
        validate_trust_successor(&self.snapshot, &snapshot)?;
        let head = TransportTrustHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        Ok(Self { snapshot, head })
    }

    /// Restore a persisted snapshot only against its separately retained trusted head.
    pub fn restore(
        snapshot: TransportTrustSnapshotV1,
        trusted_head: TransportTrustHead,
    ) -> Result<Self, TransportReceiptError> {
        snapshot.validate()?;
        let head = TransportTrustHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        if head != trusted_head {
            return Err(TransportReceiptError::TrustedTransportHeadMismatch);
        }
        Ok(Self { snapshot, head })
    }

    /// Current externally retainable trust head.
    pub const fn head(&self) -> TransportTrustHead {
        self.head
    }

    /// Read-only trust snapshot for persistence/audit.
    pub fn snapshot(&self) -> &TransportTrustSnapshotV1 {
        &self.snapshot
    }

    fn trusted_key(
        &self,
        attestor_id: &str,
        key_id: &str,
        opened_at_unix_ms: u64,
        now_unix_ms: u64,
    ) -> Result<&TransportAttestorKeyV1, TransportReceiptError> {
        if now_unix_ms < self.snapshot.issued_at_unix_ms
            || now_unix_ms >= self.snapshot.expires_at_unix_ms
        {
            return Err(TransportReceiptError::TransportTrustSnapshotNotFresh);
        }
        let key = self
            .snapshot
            .keys
            .iter()
            .find(|key| key.attestor_id == attestor_id && key.key_id == key_id)
            .ok_or(TransportReceiptError::UnknownTransportAttestorKey)?;
        if !key.active_at(opened_at_unix_ms) || !key.active_at(now_unix_ms) {
            return Err(TransportReceiptError::TransportAttestorKeyNotActive);
        }
        Ok(key)
    }
}

fn validate_trust_successor(
    previous: &TransportTrustSnapshotV1,
    next: &TransportTrustSnapshotV1,
) -> Result<(), TransportReceiptError> {
    let previous_map = previous.key_map();
    let next_map = next.key_map();
    for (identity, old) in previous_map {
        let new = next_map
            .get(&identity)
            .ok_or(TransportReceiptError::TrustedKeyIdentityDeleted)?;
        if !old.immutable_policy_eq(new) {
            return Err(TransportReceiptError::TrustedKeyPolicyMutated);
        }
        if !old.status.transition_allowed(new.status) {
            return Err(TransportReceiptError::TrustedKeyLifecycleReactivated);
        }
        if new.not_after_unix_ms > old.not_after_unix_ms {
            return Err(TransportReceiptError::TrustedKeyExpiryExtended);
        }
    }
    Ok(())
}

/// Concrete signature provider contract used by the relying party.
///
/// A production provider must implement Xenia's exact Ed25519 and ML-DSA-65 wire
/// encodings. The two methods are called independently and both must return true.
pub trait HybridReceiptSignatureVerifier {
    /// Verify the Ed25519 signature over the exact receipt digest.
    fn verify_ed25519(
        &self,
        public_key: &[u8; 32],
        digest: &[u8; 32],
        signature: &[u8; XENIA_ED25519_SIGNATURE_LEN],
    ) -> bool;

    /// Verify the ML-DSA-65 signature over the exact same receipt digest.
    fn verify_ml_dsa_65(
        &self,
        public_key: &[u8],
        digest: &[u8; 32],
        signature: &[u8; XENIA_ML_DSA_65_SIGNATURE_LEN],
    ) -> bool;
}

/// Opaque transport proof binding one exact canonical physical-effect envelope to a
/// currently trusted Xenia authenticated-session receipt.
///
/// This is still not physical authority by itself. Device-local semantic acceptance
/// and physical interlock state remain separate requirements.
#[derive(Debug)]
pub struct VerifiedTransportEnvelope {
    envelope: PhysicalEffectEnvelopeV1,
    envelope_digest: Digest32,
    payload_digest: Digest32,
    receipt_digest: Digest32,
    trust_head: TransportTrustHead,
    peer_identity_fingerprint: [u8; 32],
    session_evidence_digest: [u8; 32],
    opened_at_unix_ms: u64,
}

impl VerifiedTransportEnvelope {
    /// Canonically decoded physical-effect envelope.
    pub fn envelope(&self) -> &PhysicalEffectEnvelopeV1 {
        &self.envelope
    }

    /// Semantic domain-separated physical-envelope commitment.
    pub const fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    /// BLAKE3-256 commitment to the exact authenticated raw payload bytes.
    pub const fn payload_digest(&self) -> Digest32 {
        self.payload_digest
    }

    /// Commitment to the exact signed receipt body.
    pub const fn receipt_digest(&self) -> Digest32 {
        self.receipt_digest
    }

    /// Anti-rollback transport-trust generation used for verification.
    pub const fn trust_head(&self) -> TransportTrustHead {
        self.trust_head
    }

    /// Authenticated Xenia peer identity fingerprint.
    pub const fn peer_identity_fingerprint(&self) -> [u8; 32] {
        self.peer_identity_fingerprint
    }

    /// Commitment to Xenia's opaque authenticated session evidence.
    pub const fn session_evidence_digest(&self) -> [u8; 32] {
        self.session_evidence_digest
    }

    /// Receiver-local Xenia acceptance time.
    pub const fn opened_at_unix_ms(&self) -> u64 {
        self.opened_at_unix_ms
    }
}

/// Verify raw Xenia receipt bytes and exact raw physical-effect payload bytes.
pub fn verify_xenia_transport_receipt(
    registry: &TransportTrustRegistry,
    raw_receipt: &[u8],
    raw_payload: &[u8],
    now_unix_ms: u64,
    verifier: &impl HybridReceiptSignatureVerifier,
) -> Result<VerifiedTransportEnvelope, TransportReceiptError> {
    if raw_receipt.is_empty() || raw_receipt.len() > MAX_XENIA_RECEIPT_BYTES {
        return Err(TransportReceiptError::ReceiptSizeOutOfBounds);
    }
    if raw_payload.is_empty() || raw_payload.len() > MAX_XENIA_PHYSICAL_EFFECT_PAYLOAD_BYTES {
        return Err(TransportReceiptError::PayloadSizeOutOfBounds);
    }

    let receipt: XeniaAuthenticatedPayloadReceiptV1 =
        bincode::deserialize(raw_receipt).map_err(TransportReceiptError::Decoding)?;
    let canonical_receipt = bincode::serialize(&receipt).map_err(TransportReceiptError::Encoding)?;
    if canonical_receipt != raw_receipt {
        return Err(TransportReceiptError::NonCanonicalReceiptEncoding);
    }
    receipt.body.validate_structure()?;
    if now_unix_ms < receipt.body.opened_at_unix_ms
        || now_unix_ms >= receipt.body.expires_at_unix_ms
    {
        return Err(TransportReceiptError::ReceiptNotFresh);
    }

    let key = registry.trusted_key(
        &receipt.body.attestor_id,
        &receipt.body.key_id,
        receipt.body.opened_at_unix_ms,
        now_unix_ms,
    )?;
    let receipt_lifetime = receipt.body.expires_at_unix_ms - receipt.body.opened_at_unix_ms;
    if receipt_lifetime > key.max_receipt_lifetime_ms {
        return Err(TransportReceiptError::ReceiptExceedsTrustedKeyLifetime);
    }
    if receipt.body.peer_role != key.required_peer_role {
        return Err(TransportReceiptError::PeerRoleDenied);
    }
    if !key
        .allowed_peer_fingerprints
        .contains(&receipt.body.peer_identity_fingerprint)
    {
        return Err(TransportReceiptError::PeerIdentityDenied);
    }
    if key.require_input_control && !receipt.body.input_control_enabled {
        return Err(TransportReceiptError::RequiredInputControlCapabilityMissing);
    }

    let signing_digest = receipt.body.signing_digest()?;
    if !verifier.verify_ed25519(
        &key.ed25519_public_key,
        &signing_digest,
        &receipt.ed25519_signature,
    ) {
        return Err(TransportReceiptError::Ed25519SignatureInvalid);
    }
    if !verifier.verify_ml_dsa_65(
        &key.ml_dsa_public_key,
        &signing_digest,
        &receipt.ml_dsa_signature,
    ) {
        return Err(TransportReceiptError::MlDsaSignatureInvalid);
    }

    if receipt.body.payload_len as usize != raw_payload.len() {
        return Err(TransportReceiptError::PayloadLengthMismatch);
    }
    let payload_digest_bytes = *blake3::hash(raw_payload).as_bytes();
    if receipt.body.payload_digest != payload_digest_bytes {
        return Err(TransportReceiptError::PayloadDigestMismatch);
    }

    let envelope: PhysicalEffectEnvelopeV1 =
        bincode::deserialize(raw_payload).map_err(TransportReceiptError::Decoding)?;
    envelope
        .validate_structure()
        .map_err(|_| TransportReceiptError::InvalidPhysicalEnvelope)?;
    let canonical_payload = bincode::serialize(&envelope).map_err(TransportReceiptError::Encoding)?;
    if canonical_payload != raw_payload {
        return Err(TransportReceiptError::NonCanonicalPayloadEncoding);
    }

    let send_not_after_ms = envelope
        .send_not_after_unix_s
        .checked_mul(1_000)
        .ok_or(TransportReceiptError::PhysicalEnvelopeTimeOverflow)?;
    if receipt.body.opened_at_unix_ms > send_not_after_ms || now_unix_ms > send_not_after_ms {
        return Err(TransportReceiptError::PhysicalEnvelopeSendDeadlineElapsed);
    }

    let envelope_digest = envelope
        .digest()
        .map_err(|_| TransportReceiptError::InvalidPhysicalEnvelope)?;
    Ok(VerifiedTransportEnvelope {
        envelope,
        envelope_digest,
        payload_digest: Digest32(payload_digest_bytes),
        receipt_digest: Digest32(signing_digest),
        trust_head: registry.head(),
        peer_identity_fingerprint: receipt.body.peer_identity_fingerprint,
        session_evidence_digest: receipt.body.session_evidence_digest,
        opened_at_unix_ms: receipt.body.opened_at_unix_ms,
    })
}

fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u32).to_be_bytes());
    h.update(value.as_bytes());
}

/// Fail-closed receipt/trust verification errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TransportReceiptError {
    #[error("unsupported Xenia authenticated-payload receipt schema")]
    UnsupportedReceiptSchema,
    #[error("invalid Xenia transport-attestor identity")]
    InvalidAttestorIdentity,
    #[error("Xenia receipt signature suite is not the required hybrid profile")]
    SignatureSuiteMismatch,
    #[error("unexpected Xenia application payload type {0:#04x}")]
    UnexpectedPayloadType(u8),
    #[error("invalid Xenia receipt payload length")]
    InvalidPayloadLength,
    #[error("zero security commitment in Xenia receipt")]
    ZeroSecurityDigest,
    #[error("invalid Xenia receipt lifetime")]
    InvalidReceiptLifetime,
    #[error("trusted Xenia transport key is malformed")]
    MalformedTrustedKey,
    #[error("invalid trusted transport-key validity window")]
    InvalidTrustedKeyWindow,
    #[error("invalid trusted Xenia peer policy")]
    InvalidPeerPolicy,
    #[error("unsupported transport-trust snapshot schema")]
    UnsupportedTrustSnapshotSchema,
    #[error("transport-trust sequence zero is invalid")]
    TrustSequenceZero,
    #[error("invalid transport-trust snapshot validity window")]
    InvalidTrustSnapshotWindow,
    #[error("invalid number of transport-attestor keys")]
    InvalidTrustKeyCount,
    #[error("transport-trust genesis must not have a predecessor")]
    TrustGenesisHasPredecessor,
    #[error("transport-trust successor is missing predecessor commitment")]
    TrustSuccessorMissingPredecessor,
    #[error("duplicate transport-attestor/key identity")]
    DuplicateTrustedKeyIdentity,
    #[error("snapshot is not transport-trust genesis")]
    NotTrustGenesis,
    #[error("transport-trust sequence overflow")]
    TrustSequenceOverflow,
    #[error("transport-trust sequence is not next: expected {expected}, proposed {proposed}")]
    TrustSequenceNotNext { expected: u64, proposed: u64 },
    #[error("transport-trust predecessor mismatch")]
    TrustPredecessorMismatch,
    #[error("transport-trust issue time regressed")]
    TrustIssuedAtRegressed,
    #[error("transport trusted-head mismatch")]
    TrustedTransportHeadMismatch,
    #[error("existing trusted key identity was deleted")]
    TrustedKeyIdentityDeleted,
    #[error("existing trusted key material or peer policy mutated under the same key id")]
    TrustedKeyPolicyMutated,
    #[error("trusted key lifecycle attempted reactivation")]
    TrustedKeyLifecycleReactivated,
    #[error("trusted key expiry was extended under the same key id")]
    TrustedKeyExpiryExtended,
    #[error("transport-trust snapshot is not fresh")]
    TransportTrustSnapshotNotFresh,
    #[error("Xenia transport-attestor/key is not trusted")]
    UnknownTransportAttestorKey,
    #[error("Xenia transport-attestor key is not active at receipt and relying-party time")]
    TransportAttestorKeyNotActive,
    #[error("receipt lifetime exceeds trusted key policy")]
    ReceiptExceedsTrustedKeyLifetime,
    #[error("authenticated Xenia peer role is denied")]
    PeerRoleDenied,
    #[error("authenticated Xenia peer identity is denied")]
    PeerIdentityDenied,
    #[error("authenticated Xenia session lacks required input/control capability")]
    RequiredInputControlCapabilityMissing,
    #[error("Ed25519 receipt signature failed verification")]
    Ed25519SignatureInvalid,
    #[error("ML-DSA-65 receipt signature failed verification")]
    MlDsaSignatureInvalid,
    #[error("serialized receipt size is outside the accepted bound")]
    ReceiptSizeOutOfBounds,
    #[error("physical-effect payload size is outside the accepted bound")]
    PayloadSizeOutOfBounds,
    #[error("receipt is not fresh at relying-party time")]
    ReceiptNotFresh,
    #[error("receipt payload length does not match exact raw payload")]
    PayloadLengthMismatch,
    #[error("receipt payload digest does not match exact raw payload")]
    PayloadDigestMismatch,
    #[error("receipt bincode representation is not canonical for v1")]
    NonCanonicalReceiptEncoding,
    #[error("physical-effect payload bincode representation is not canonical for v1")]
    NonCanonicalPayloadEncoding,
    #[error("physical-effect envelope failed structural validation")]
    InvalidPhysicalEnvelope,
    #[error("physical-effect send deadline elapsed before receipt/relying-party acceptance")]
    PhysicalEnvelopeSendDeadlineElapsed,
    #[error("physical-effect send deadline overflowed milliseconds")]
    PhysicalEnvelopeTimeOverflow,
    #[error("bincode decode failed: {0}")]
    Decoding(bincode::Error),
    #[error("bincode encode failed: {0}")]
    Encoding(bincode::Error),
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use symthaea_action_checkpoint::CheckpointHead;
    use symthaea_authority::{Operation, PrincipalId, ResourceRef, TaskId};
    use symthaea_iot_authority::{DEVICE_COMMAND_SCHEMA_VERSION, DeviceCommand};
    use symthaea_iot_durable_runtime::DurableIoTHead;
    use symthaea_iot_policy::ActuationPolicyHead;
    use symthaea_iot_posture::VerifierTrustHead;

    struct TestHybridVerifier;

    impl HybridReceiptSignatureVerifier for TestHybridVerifier {
        fn verify_ed25519(
            &self,
            _public_key: &[u8; 32],
            digest: &[u8; 32],
            signature: &[u8; XENIA_ED25519_SIGNATURE_LEN],
        ) -> bool {
            signature[..32] == digest[..]
        }

        fn verify_ml_dsa_65(
            &self,
            _public_key: &[u8],
            digest: &[u8; 32],
            signature: &[u8; XENIA_ML_DSA_65_SIGNATURE_LEN],
        ) -> bool {
            signature[..32] == digest[..]
        }
    }

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn envelope() -> PhysicalEffectEnvelopeV1 {
        PhysicalEffectEnvelopeV1 {
            schema_version: 1,
            command: DeviceCommand {
                schema_version: DEVICE_COMMAND_SCHEMA_VERSION,
                command_id: "cmd-7".into(),
                actor: PrincipalId("agent:irrigation".into()),
                executor: PrincipalId("gateway:field-a".into()),
                task: Some(TaskId("irrigate:zone-7".into())),
                device: ResourceRef("iot:valve:72".into()),
                operation: Operation("valve.open".into()),
                expected_firmware: d(7),
                sequence: 7,
                issued_at_unix_s: 100,
                expires_at_unix_s: 120,
                parameters: BTreeMap::new(),
            },
            proposal_digest: d(2),
            policy_digest: d(3),
            policy_registry_head: ActuationPolicyHead {
                sequence: 1,
                digest: d(4),
            },
            durable_host_head: DurableIoTHead {
                action_head: CheckpointHead {
                    sequence: 0,
                    digest: d(5),
                },
                digest: d(6),
            },
            posture_result_digest: d(8),
            posture_evidence_digest: d(9),
            posture_reference_values_digest: d(10),
            posture_appraisal_policy_digest: d(11),
            posture_challenge_digest: d(12),
            posture_verifier_trust_head: VerifierTrustHead {
                sequence: 1,
                digest: d(13),
            },
            posture_expires_at_unix_s: 120,
            host_preflight_at_unix_s: 110,
            send_not_after_unix_s: 115,
        }
    }

    fn trusted_key() -> TransportAttestorKeyV1 {
        TransportAttestorKeyV1 {
            attestor_id: "xenia-gateway-a".into(),
            key_id: "transport-key-1".into(),
            ed25519_public_key: [0x21; 32],
            ml_dsa_public_key: vec![0x22; XENIA_ML_DSA_65_PUBLIC_KEY_LEN],
            status: TransportAttestorStatus::Active,
            not_before_unix_ms: 90_000,
            not_after_unix_ms: 130_000,
            max_receipt_lifetime_ms: 2_000,
            required_peer_role: XeniaReceiptPeerRoleV1::Viewer,
            allowed_peer_fingerprints: BTreeSet::from([[0x44; 32]]),
            require_input_control: true,
        }
    }

    fn registry() -> TransportTrustRegistry {
        TransportTrustRegistry::genesis(TransportTrustSnapshotV1 {
            schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_ms: 90_000,
            expires_at_unix_ms: 130_000,
            previous_snapshot_digest: None,
            keys: vec![trusted_key()],
        })
        .unwrap()
    }

    fn receipt(payload: &[u8]) -> XeniaAuthenticatedPayloadReceiptV1 {
        let body = XeniaAuthenticatedPayloadReceiptBodyV1 {
            schema: XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA.into(),
            attestor_id: "xenia-gateway-a".into(),
            key_id: "transport-key-1".into(),
            signature_algorithm: XENIA_HYBRID_SIGNATURE_SUITE.into(),
            session_evidence_digest: [0x31; 32],
            peer_role: XeniaReceiptPeerRoleV1::Viewer,
            peer_identity_fingerprint: [0x44; 32],
            transcript_hash: [0x45; 32],
            session_context_hash: [0x46; 32],
            telemetry_enabled: false,
            input_control_enabled: true,
            payload_type: XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE,
            payload_len: payload.len() as u32,
            payload_digest: *blake3::hash(payload).as_bytes(),
            sealed_envelope_digest: [0x47; 32],
            opened_at_unix_ms: 112_000,
            expires_at_unix_ms: 114_000,
        };
        let digest = body.signing_digest().unwrap();
        let mut ed = [0u8; XENIA_ED25519_SIGNATURE_LEN];
        ed[..32].copy_from_slice(&digest);
        let mut pq = [0u8; XENIA_ML_DSA_65_SIGNATURE_LEN];
        pq[..32].copy_from_slice(&digest);
        XeniaAuthenticatedPayloadReceiptV1 {
            body,
            ed25519_signature: ed,
            ml_dsa_signature: pq,
        }
    }

    #[test]
    fn exact_transport_receipt_mints_opaque_verified_envelope() {
        let payload = bincode::serialize(&envelope()).unwrap();
        let raw_receipt = bincode::serialize(&receipt(&payload)).unwrap();
        let verified = verify_xenia_transport_receipt(
            &registry(),
            &raw_receipt,
            &payload,
            113_000,
            &TestHybridVerifier,
        )
        .unwrap();
        assert_eq!(verified.envelope().command.sequence, 7);
        assert_eq!(verified.peer_identity_fingerprint(), [0x44; 32]);
        assert_eq!(verified.payload_digest(), Digest32(*blake3::hash(&payload).as_bytes()));
    }

    #[test]
    fn either_signature_failure_denies_transport_evidence() {
        let payload = bincode::serialize(&envelope()).unwrap();
        let mut bad = receipt(&payload);
        bad.ml_dsa_signature[0] ^= 1;
        let raw_receipt = bincode::serialize(&bad).unwrap();
        assert!(matches!(
            verify_xenia_transport_receipt(
                &registry(),
                &raw_receipt,
                &payload,
                113_000,
                &TestHybridVerifier,
            ),
            Err(TransportReceiptError::MlDsaSignatureInvalid)
        ));
    }

    #[test]
    fn authenticated_bytes_cannot_be_substituted_after_receipt() {
        let payload = bincode::serialize(&envelope()).unwrap();
        let raw_receipt = bincode::serialize(&receipt(&payload)).unwrap();
        let mut altered = payload.clone();
        *altered.last_mut().unwrap() ^= 1;
        assert!(matches!(
            verify_xenia_transport_receipt(
                &registry(),
                &raw_receipt,
                &altered,
                113_000,
                &TestHybridVerifier,
            ),
            Err(TransportReceiptError::PayloadDigestMismatch)
        ));
    }

    #[test]
    fn trust_lifecycle_cannot_reactivate_or_mutate_same_key_identity() {
        let base = registry();
        let mut retired = trusted_key();
        retired.status = TransportAttestorStatus::Retired;
        retired.not_after_unix_ms = 125_000;
        let snapshot2 = TransportTrustSnapshotV1 {
            schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 2,
            issued_at_unix_ms: 91_000,
            expires_at_unix_ms: 130_000,
            previous_snapshot_digest: Some(base.head().digest),
            keys: vec![retired.clone()],
        };
        let retired_registry = base.successor(snapshot2).unwrap();

        let mut reactivated = retired;
        reactivated.status = TransportAttestorStatus::Active;
        let snapshot3 = TransportTrustSnapshotV1 {
            schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 3,
            issued_at_unix_ms: 92_000,
            expires_at_unix_ms: 130_000,
            previous_snapshot_digest: Some(retired_registry.head().digest),
            keys: vec![reactivated],
        };
        assert!(matches!(
            retired_registry.successor(snapshot3),
            Err(TransportReceiptError::TrustedKeyLifecycleReactivated)
        ));
    }
}
