// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ephemeral exact-byte provenance for Xenia-authenticated physical-effect transport.
//!
//! `VerifiedTransportEnvelope` proves that one receipt/payload pair passed the configured
//! transport verifier at an earlier boundary, but intentionally retains only interpreted
//! commitments. This crate consumes that opaque proof while binding the exact canonical receipt
//! and payload bytes back to it, so a later privileged stage can re-run current-trust cryptography
//! over the identical evidence rather than over caller-supplied replacement bytes.
//!
//! This capsule is not transport revalidation and is not physical authority. It performs no
//! signature verification, accepts no trust registry or clock, and exposes no final/JIT/HAL
//! surface. The next fixed verifier must consume this exact evidence under current trust.

#![deny(unsafe_code)]

use symthaea_authority::Digest32;
use symthaea_iot_device_protocol::PhysicalEffectEnvelopeV1;
use symthaea_iot_transport_receipt::{
    MAX_XENIA_PHYSICAL_EFFECT_PAYLOAD_BYTES, MAX_XENIA_RECEIPT_BYTES, TransportTrustHead,
    VerifiedTransportEnvelope, XeniaAuthenticatedPayloadReceiptV1,
};
use thiserror::Error;

const EXACT_XENIA_EVIDENCE_DOMAIN: &[u8] = b"symthaea-iot-exact-xenia-transport-evidence-v1\0";

/// Opaque in-memory capsule retaining the exact canonical Xenia receipt and physical payload
/// represented by one consumed `VerifiedTransportEnvelope`.
///
/// It is deliberately neither `Clone` nor serializable. A process restart therefore destroys
/// this continuation evidence; the already-burned command must be resubmitted instead of
/// reconstructing a pre-crash actuation attempt from portable bytes. Construction consumes the
/// prior opaque transport proof so normal callers cannot mint multiple continuation capsules
/// from the same in-process proof.
#[derive(Debug)]
pub struct ExactXeniaTransportEvidence {
    raw_receipt: Vec<u8>,
    raw_payload: Vec<u8>,
    exact_evidence_digest: Digest32,
    receipt_digest: Digest32,
    payload_digest: Digest32,
    envelope_digest: Digest32,
    transport_trust_head: TransportTrustHead,
    attestor_id: String,
    key_id: String,
    peer_identity_fingerprint: [u8; 32],
    session_evidence_digest: [u8; 32],
    opened_at_unix_ms: u64,
    receipt_expires_at_unix_ms: u64,
}

impl ExactXeniaTransportEvidence {
    /// Exact canonical outer receipt bytes accepted at the original transport boundary.
    pub fn canonical_receipt_bytes(&self) -> &[u8] {
        &self.raw_receipt
    }

    /// Exact canonical physical-effect payload bytes authenticated by that receipt.
    pub fn canonical_payload_bytes(&self) -> &[u8] {
        &self.raw_payload
    }

    /// Domain-separated commitment to both exact byte strings plus the original trust head.
    pub const fn exact_evidence_digest(&self) -> Digest32 {
        self.exact_evidence_digest
    }

    pub const fn receipt_digest(&self) -> Digest32 {
        self.receipt_digest
    }

    pub const fn payload_digest(&self) -> Digest32 {
        self.payload_digest
    }

    pub const fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    pub const fn transport_trust_head(&self) -> TransportTrustHead {
        self.transport_trust_head
    }

    pub fn attestor_id(&self) -> &str {
        &self.attestor_id
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub const fn peer_identity_fingerprint(&self) -> [u8; 32] {
        self.peer_identity_fingerprint
    }

    pub const fn session_evidence_digest(&self) -> [u8; 32] {
        self.session_evidence_digest
    }

    pub const fn opened_at_unix_ms(&self) -> u64 {
        self.opened_at_unix_ms
    }

    pub const fn receipt_expires_at_unix_ms(&self) -> u64 {
        self.receipt_expires_at_unix_ms
    }
}

/// Bind exact canonical receipt/payload bytes by consuming one already verified opaque proof.
///
/// This function deliberately performs no cryptographic verification. Its only claim is that
/// the retained bytes reproduce every exact transport commitment available from `transport`.
/// Taking `transport` by value makes this an affine continuation boundary: the same opaque proof
/// cannot normally be reused to mint multiple exact-evidence capsules. A later fixed current-
/// trust verifier must re-run both receipt signatures over these bytes.
pub fn bind_exact_xenia_transport_evidence(
    transport: VerifiedTransportEnvelope,
    raw_receipt: &[u8],
    raw_payload: &[u8],
) -> Result<ExactXeniaTransportEvidence, ExactTransportEvidenceError> {
    if raw_receipt.is_empty() || raw_receipt.len() > MAX_XENIA_RECEIPT_BYTES {
        return Err(ExactTransportEvidenceError::ReceiptSizeOutOfBounds);
    }
    if raw_payload.is_empty() || raw_payload.len() > MAX_XENIA_PHYSICAL_EFFECT_PAYLOAD_BYTES {
        return Err(ExactTransportEvidenceError::PayloadSizeOutOfBounds);
    }

    let receipt: XeniaAuthenticatedPayloadReceiptV1 = bincode::deserialize(raw_receipt)
        .map_err(|_| ExactTransportEvidenceError::ReceiptEncoding)?;
    receipt
        .body
        .validate_structure()
        .map_err(|_| ExactTransportEvidenceError::InvalidReceiptStructure)?;
    let canonical_receipt = bincode::serialize(&receipt)
        .map_err(|_| ExactTransportEvidenceError::ReceiptEncoding)?;
    if canonical_receipt != raw_receipt {
        return Err(ExactTransportEvidenceError::NonCanonicalReceiptEncoding);
    }

    let signing_digest = Digest32(
        receipt
            .body
            .signing_digest()
            .map_err(|_| ExactTransportEvidenceError::InvalidReceiptStructure)?,
    );
    if signing_digest != transport.receipt_digest() {
        return Err(ExactTransportEvidenceError::ReceiptCommitmentMismatch);
    }
    if receipt.body.opened_at_unix_ms != transport.opened_at_unix_ms() {
        return Err(ExactTransportEvidenceError::OpenedAtMismatch);
    }
    if receipt.body.peer_identity_fingerprint != transport.peer_identity_fingerprint() {
        return Err(ExactTransportEvidenceError::PeerIdentityMismatch);
    }
    if receipt.body.session_evidence_digest != transport.session_evidence_digest() {
        return Err(ExactTransportEvidenceError::SessionEvidenceMismatch);
    }

    let payload_hash = *blake3::hash(raw_payload).as_bytes();
    let payload_digest = Digest32(payload_hash);
    if payload_digest != transport.payload_digest() {
        return Err(ExactTransportEvidenceError::PayloadCommitmentMismatch);
    }
    if receipt.body.payload_len as usize != raw_payload.len()
        || receipt.body.payload_digest != payload_hash
    {
        return Err(ExactTransportEvidenceError::ReceiptPayloadBindingMismatch);
    }

    let envelope: PhysicalEffectEnvelopeV1 = bincode::deserialize(raw_payload)
        .map_err(|_| ExactTransportEvidenceError::PayloadEncoding)?;
    envelope
        .validate_structure()
        .map_err(|_| ExactTransportEvidenceError::InvalidPhysicalEnvelope)?;
    let canonical_payload = bincode::serialize(&envelope)
        .map_err(|_| ExactTransportEvidenceError::PayloadEncoding)?;
    if canonical_payload != raw_payload {
        return Err(ExactTransportEvidenceError::NonCanonicalPayloadEncoding);
    }
    if &envelope != transport.envelope() {
        return Err(ExactTransportEvidenceError::EnvelopeObjectMismatch);
    }
    let envelope_digest = envelope
        .digest()
        .map_err(|_| ExactTransportEvidenceError::InvalidPhysicalEnvelope)?;
    if envelope_digest != transport.envelope_digest() {
        return Err(ExactTransportEvidenceError::EnvelopeCommitmentMismatch);
    }

    let trust_head = transport.trust_head();
    let exact_evidence_digest = digest_exact_evidence(raw_receipt, raw_payload, trust_head);

    Ok(ExactXeniaTransportEvidence {
        raw_receipt: raw_receipt.to_vec(),
        raw_payload: raw_payload.to_vec(),
        exact_evidence_digest,
        receipt_digest: signing_digest,
        payload_digest,
        envelope_digest,
        transport_trust_head: trust_head,
        attestor_id: receipt.body.attestor_id,
        key_id: receipt.body.key_id,
        peer_identity_fingerprint: receipt.body.peer_identity_fingerprint,
        session_evidence_digest: receipt.body.session_evidence_digest,
        opened_at_unix_ms: receipt.body.opened_at_unix_ms,
        receipt_expires_at_unix_ms: receipt.body.expires_at_unix_ms,
    })
}

fn digest_exact_evidence(
    raw_receipt: &[u8],
    raw_payload: &[u8],
    trust_head: TransportTrustHead,
) -> Digest32 {
    let mut h = blake3::Hasher::new();
    h.update(EXACT_XENIA_EVIDENCE_DOMAIN);
    h.update(&(raw_receipt.len() as u64).to_be_bytes());
    h.update(raw_receipt);
    h.update(&(raw_payload.len() as u64).to_be_bytes());
    h.update(raw_payload);
    h.update(&trust_head.sequence.to_be_bytes());
    let Digest32(trust_digest) = trust_head.digest;
    h.update(&trust_digest);
    Digest32(*h.finalize().as_bytes())
}

#[derive(Debug, Error)]
pub enum ExactTransportEvidenceError {
    #[error("exact Xenia receipt size is outside accepted bounds")]
    ReceiptSizeOutOfBounds,
    #[error("exact physical-effect payload size is outside accepted bounds")]
    PayloadSizeOutOfBounds,
    #[error("exact Xenia receipt encoding is invalid")]
    ReceiptEncoding,
    #[error("exact physical-effect payload encoding is invalid")]
    PayloadEncoding,
    #[error("Xenia receipt fails its structural contract")]
    InvalidReceiptStructure,
    #[error("Xenia receipt is not canonically encoded")]
    NonCanonicalReceiptEncoding,
    #[error("physical-effect payload is not canonically encoded")]
    NonCanonicalPayloadEncoding,
    #[error("physical-effect envelope fails its structural contract")]
    InvalidPhysicalEnvelope,
    #[error("exact receipt does not reproduce the opaque transport receipt commitment")]
    ReceiptCommitmentMismatch,
    #[error("exact payload does not reproduce the opaque transport payload commitment")]
    PayloadCommitmentMismatch,
    #[error("receipt payload length/digest does not bind the exact retained payload")]
    ReceiptPayloadBindingMismatch,
    #[error("exact payload decodes to another physical-effect envelope")]
    EnvelopeObjectMismatch,
    #[error("exact payload does not reproduce the opaque semantic envelope commitment")]
    EnvelopeCommitmentMismatch,
    #[error("exact receipt names another authenticated Xenia peer")]
    PeerIdentityMismatch,
    #[error("exact receipt names another authenticated-session evidence commitment")]
    SessionEvidenceMismatch,
    #[error("exact receipt has another receiver-local opening time")]
    OpenedAtMismatch,
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use symthaea_action_checkpoint::CheckpointHead;
    use symthaea_authority::{Operation, PrincipalId, ResourceRef, TaskId};
    use symthaea_iot_authority::{DEVICE_COMMAND_SCHEMA_VERSION, DeviceCommand};
    use symthaea_iot_device_protocol::{
        PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION, PhysicalEffectEnvelopeV1,
    };
    use symthaea_iot_durable_runtime::DurableIoTHead;
    use symthaea_iot_policy::ActuationPolicyHead;
    use symthaea_iot_posture::VerifierTrustHead;
    use symthaea_iot_transport_receipt::{
        HybridReceiptSignatureVerifier, TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
        TransportAttestorKeyV1, TransportAttestorStatus, TransportTrustRegistry,
        TransportTrustSnapshotV1, XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA,
        XENIA_ED25519_SIGNATURE_LEN, XENIA_HYBRID_SIGNATURE_SUITE,
        XENIA_ML_DSA_65_PUBLIC_KEY_LEN, XENIA_ML_DSA_65_SIGNATURE_LEN,
        XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE, XeniaAuthenticatedPayloadReceiptBodyV1,
        XeniaReceiptPeerRoleV1, verify_xenia_transport_receipt,
    };

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

    fn envelope(sequence: u64) -> PhysicalEffectEnvelopeV1 {
        PhysicalEffectEnvelopeV1 {
            schema_version: PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION,
            command: DeviceCommand {
                schema_version: DEVICE_COMMAND_SCHEMA_VERSION,
                command_id: format!("cmd-{sequence}"),
                actor: PrincipalId("agent:irrigation".into()),
                executor: PrincipalId("gateway:field-a".into()),
                task: Some(TaskId("irrigate:zone-7".into())),
                device: ResourceRef("iot:valve:72".into()),
                operation: Operation("valve.open".into()),
                expected_firmware: d(7),
                sequence,
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

    fn registry() -> TransportTrustRegistry {
        TransportTrustRegistry::genesis(TransportTrustSnapshotV1 {
            schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_ms: 90_000,
            expires_at_unix_ms: 130_000,
            previous_snapshot_digest: None,
            keys: vec![TransportAttestorKeyV1 {
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
            }],
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

    fn verified_for(payload: &[u8], raw_receipt: &[u8]) -> VerifiedTransportEnvelope {
        verify_xenia_transport_receipt(
            &registry(),
            raw_receipt,
            payload,
            113_000,
            &TestHybridVerifier,
        )
        .unwrap()
    }

    #[test]
    fn exact_evidence_token_owns_original_canonical_bytes() {
        let mut payload = bincode::serialize(&envelope(7)).unwrap();
        let mut raw_receipt = bincode::serialize(&receipt(&payload)).unwrap();
        let original_payload = payload.clone();
        let original_receipt = raw_receipt.clone();
        let verified = verified_for(&payload, &raw_receipt);
        let expected_receipt_digest = verified.receipt_digest();
        let expected_payload_digest = verified.payload_digest();
        let expected_envelope_digest = verified.envelope_digest();
        let expected_trust_head = verified.trust_head();

        let exact = bind_exact_xenia_transport_evidence(verified, &raw_receipt, &payload).unwrap();
        raw_receipt[0] ^= 1;
        payload[0] ^= 1;

        assert_eq!(exact.canonical_receipt_bytes(), original_receipt.as_slice());
        assert_eq!(exact.canonical_payload_bytes(), original_payload.as_slice());
        assert_eq!(exact.receipt_digest(), expected_receipt_digest);
        assert_eq!(exact.payload_digest(), expected_payload_digest);
        assert_eq!(exact.envelope_digest(), expected_envelope_digest);
        assert_eq!(exact.transport_trust_head(), expected_trust_head);
        assert_eq!(exact.attestor_id(), "xenia-gateway-a");
        assert_eq!(exact.key_id(), "transport-key-1");
        assert_ne!(exact.exact_evidence_digest(), Digest32([0; 32]));
    }

    #[test]
    fn another_canonical_receipt_or_payload_cannot_bind_existing_transport_proof() {
        let payload = bincode::serialize(&envelope(7)).unwrap();
        let raw_receipt = bincode::serialize(&receipt(&payload)).unwrap();
        let other_payload = bincode::serialize(&envelope(8)).unwrap();
        let other_receipt = bincode::serialize(&receipt(&other_payload)).unwrap();

        assert!(matches!(
            bind_exact_xenia_transport_evidence(
                verified_for(&payload, &raw_receipt),
                &other_receipt,
                &payload,
            ),
            Err(ExactTransportEvidenceError::ReceiptCommitmentMismatch)
        ));
        assert!(matches!(
            bind_exact_xenia_transport_evidence(
                verified_for(&payload, &raw_receipt),
                &raw_receipt,
                &other_payload,
            ),
            Err(ExactTransportEvidenceError::PayloadCommitmentMismatch)
        ));
    }

    #[test]
    fn trailing_bytes_cannot_be_retained_as_exact_evidence() {
        let payload = bincode::serialize(&envelope(7)).unwrap();
        let raw_receipt = bincode::serialize(&receipt(&payload)).unwrap();

        let mut receipt_with_trailing = raw_receipt.clone();
        receipt_with_trailing.push(0);
        assert!(matches!(
            bind_exact_xenia_transport_evidence(
                verified_for(&payload, &raw_receipt),
                &receipt_with_trailing,
                &payload,
            ),
            Err(ExactTransportEvidenceError::NonCanonicalReceiptEncoding)
                | Err(ExactTransportEvidenceError::ReceiptEncoding)
        ));

        let mut payload_with_trailing = payload.clone();
        payload_with_trailing.push(0);
        assert!(matches!(
            bind_exact_xenia_transport_evidence(
                verified_for(&payload, &raw_receipt),
                &raw_receipt,
                &payload_with_trailing,
            ),
            Err(ExactTransportEvidenceError::PayloadCommitmentMismatch)
                | Err(ExactTransportEvidenceError::NonCanonicalPayloadEncoding)
                | Err(ExactTransportEvidenceError::PayloadEncoding)
        ));
    }
}
