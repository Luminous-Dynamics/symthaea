// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-shot current-trust revalidation for exact Xenia physical-effect transport evidence.
//!
//! The prior exact-evidence stage retains the canonical Xenia receipt/payload bytes represented
//! by an older opaque transport proof. This crate consumes that evidence together with a
//! guard-owned `TransportTrustRegistry` whose head has been independently anchored, obtains
//! relying-party time internally, and re-runs the existing fixed Xenia Ed25519 + ML-DSA-65
//! verification path over those exact bytes.
//!
//! Success remains non-authorizing. The resulting proof additionally preserves the exact selected
//! attestor/key identity, key-record commitment, and the earliest natural expiry among receipt,
//! key and trust snapshot so a later JIT fence cannot confuse unchanged head identity with
//! unexpired trust.

#![deny(unsafe_code)]

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use symthaea_authority::Digest32;
use symthaea_iot_device_protocol::PhysicalEffectEnvelopeV1;
use symthaea_iot_transport_exact_evidence::ExactXeniaTransportEvidence;
use symthaea_iot_transport_receipt::{
    TransportReceiptError, TransportTrustHead, TransportTrustRegistry, VerifiedTransportEnvelope,
};
use symthaea_iot_xenia_hybrid_verifier::verify_xenia_physical_effect_receipt;
use thiserror::Error;

/// One-shot holder of independently anchored current Xenia transport trust.
pub struct CurrentXeniaTransportRevalidator {
    registry: TransportTrustRegistry,
    anchored_current_head: TransportTrustHead,
}

impl fmt::Debug for CurrentXeniaTransportRevalidator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CurrentXeniaTransportRevalidator")
            .field("anchored_current_head", &self.anchored_current_head)
            .finish_non_exhaustive()
    }
}

impl CurrentXeniaTransportRevalidator {
    /// Bind one verified current registry to its independently retained anti-rollback head.
    pub fn new(
        registry: TransportTrustRegistry,
        anchored_current_head: TransportTrustHead,
    ) -> Result<Self, CurrentTransportRevalidationError> {
        if registry.head() != anchored_current_head {
            return Err(CurrentTransportRevalidationError::RegistryHeadNotAnchored);
        }
        Ok(Self {
            registry,
            anchored_current_head,
        })
    }

    /// Consume exact evidence and revalidate it using guard-local current time and fixed crypto.
    pub fn revalidate(
        self,
        exact: ExactXeniaTransportEvidence,
    ) -> Result<RevalidatedXeniaTransport, CurrentTransportRevalidationError> {
        self.revalidate_at(exact, system_unix_ms()?)
    }

    fn revalidate_at(
        self,
        exact: ExactXeniaTransportEvidence,
        now_unix_ms: u64,
    ) -> Result<RevalidatedXeniaTransport, CurrentTransportRevalidationError> {
        if self.registry.head() != self.anchored_current_head {
            return Err(CurrentTransportRevalidationError::RegistryHeadNotAnchored);
        }
        if exact.transport_trust_head() != self.anchored_current_head {
            return Err(CurrentTransportRevalidationError::OriginalTransportGenerationNotCurrent);
        }

        let expected_exact_evidence_digest = exact.exact_evidence_digest();
        let expected_receipt_digest = exact.receipt_digest();
        let expected_payload_digest = exact.payload_digest();
        let expected_envelope_digest = exact.envelope_digest();
        let expected_peer_identity = exact.peer_identity_fingerprint();
        let expected_session_evidence = exact.session_evidence_digest();
        let expected_opened_at = exact.opened_at_unix_ms();
        let receipt_expires_at_unix_ms = exact.receipt_expires_at_unix_ms();
        let attestor_id = exact.attestor_id().to_owned();
        let key_id = exact.key_id().to_owned();

        let verified = verify_xenia_physical_effect_receipt(
            &self.registry,
            exact.canonical_receipt_bytes(),
            exact.canonical_payload_bytes(),
            now_unix_ms,
        )?;

        if verified.trust_head() != self.anchored_current_head {
            return Err(CurrentTransportRevalidationError::CurrentVerificationHeadMismatch);
        }
        if verified.receipt_digest() != expected_receipt_digest {
            return Err(CurrentTransportRevalidationError::ReceiptCommitmentMismatch);
        }
        if verified.payload_digest() != expected_payload_digest {
            return Err(CurrentTransportRevalidationError::PayloadCommitmentMismatch);
        }
        if verified.envelope_digest() != expected_envelope_digest {
            return Err(CurrentTransportRevalidationError::EnvelopeCommitmentMismatch);
        }
        if verified.peer_identity_fingerprint() != expected_peer_identity {
            return Err(CurrentTransportRevalidationError::PeerIdentityMismatch);
        }
        if verified.session_evidence_digest() != expected_session_evidence {
            return Err(CurrentTransportRevalidationError::SessionEvidenceMismatch);
        }
        if verified.opened_at_unix_ms() != expected_opened_at {
            return Err(CurrentTransportRevalidationError::OpenedAtMismatch);
        }

        // Fixed verification proved that this exact signed receipt selected an active current key.
        // Preserve that immutable key record and every natural-expiry ceiling before the one-shot
        // registry is destroyed. A later JIT fence can then require current key identity/digest
        // equality rather than treating unchanged trust-head identity as sufficient freshness.
        let snapshot = self.registry.snapshot();
        let current_key = snapshot
            .keys
            .iter()
            .find(|key| key.attestor_id == attestor_id && key.key_id == key_id)
            .ok_or(CurrentTransportRevalidationError::CurrentTransportKeyMissing)?;
        let transport_key_digest = current_key.digest()?;
        let transport_key_not_after_unix_ms = current_key.not_after_unix_ms;
        let trust_snapshot_expires_at_unix_ms = snapshot.expires_at_unix_ms;
        let valid_until_unix_ms = receipt_expires_at_unix_ms
            .min(transport_key_not_after_unix_ms)
            .min(trust_snapshot_expires_at_unix_ms);
        if now_unix_ms >= valid_until_unix_ms {
            return Err(CurrentTransportRevalidationError::CurrentTransportWindowElapsed);
        }

        Ok(RevalidatedXeniaTransport {
            verified,
            exact_evidence_digest: expected_exact_evidence_digest,
            attestor_id,
            key_id,
            transport_key_digest,
            revalidated_at_unix_ms: now_unix_ms,
            receipt_expires_at_unix_ms,
            transport_key_not_after_unix_ms,
            trust_snapshot_expires_at_unix_ms,
            valid_until_unix_ms,
        })
    }
}

fn system_unix_ms() -> Result<u64, CurrentTransportRevalidationError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| CurrentTransportRevalidationError::ClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis())
        .map_err(|_| CurrentTransportRevalidationError::ClockOverflow)
}

/// Opaque proof that the exact Xenia evidence passed fixed current-trust verification.
///
/// This type is neither `Clone` nor serializable and retains no portable raw receipt/payload.
/// `valid_until_unix_ms` is the earliest exclusive natural expiry of the receipt, selected key,
/// and current trust snapshot. It is evidence for a later fence, not authority or a lease.
pub struct RevalidatedXeniaTransport {
    verified: VerifiedTransportEnvelope,
    exact_evidence_digest: Digest32,
    attestor_id: String,
    key_id: String,
    transport_key_digest: Digest32,
    revalidated_at_unix_ms: u64,
    receipt_expires_at_unix_ms: u64,
    transport_key_not_after_unix_ms: u64,
    trust_snapshot_expires_at_unix_ms: u64,
    valid_until_unix_ms: u64,
}

impl fmt::Debug for RevalidatedXeniaTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RevalidatedXeniaTransport")
            .field("exact_evidence_digest", &self.exact_evidence_digest)
            .field("receipt_digest", &self.verified.receipt_digest())
            .field("payload_digest", &self.verified.payload_digest())
            .field("envelope_digest", &self.verified.envelope_digest())
            .field("transport_trust_head", &self.verified.trust_head())
            .field("attestor_id", &self.attestor_id)
            .field("key_id", &self.key_id)
            .field("transport_key_digest", &self.transport_key_digest)
            .field("revalidated_at_unix_ms", &self.revalidated_at_unix_ms)
            .field("receipt_expires_at_unix_ms", &self.receipt_expires_at_unix_ms)
            .field(
                "transport_key_not_after_unix_ms",
                &self.transport_key_not_after_unix_ms,
            )
            .field(
                "trust_snapshot_expires_at_unix_ms",
                &self.trust_snapshot_expires_at_unix_ms,
            )
            .field("valid_until_unix_ms", &self.valid_until_unix_ms)
            .finish_non_exhaustive()
    }
}

impl RevalidatedXeniaTransport {
    pub fn envelope(&self) -> &PhysicalEffectEnvelopeV1 {
        self.verified.envelope()
    }

    pub const fn exact_evidence_digest(&self) -> Digest32 {
        self.exact_evidence_digest
    }

    pub const fn receipt_digest(&self) -> Digest32 {
        self.verified.receipt_digest()
    }

    pub const fn payload_digest(&self) -> Digest32 {
        self.verified.payload_digest()
    }

    pub const fn envelope_digest(&self) -> Digest32 {
        self.verified.envelope_digest()
    }

    pub const fn transport_trust_head(&self) -> TransportTrustHead {
        self.verified.trust_head()
    }

    pub const fn peer_identity_fingerprint(&self) -> [u8; 32] {
        self.verified.peer_identity_fingerprint()
    }

    pub const fn session_evidence_digest(&self) -> [u8; 32] {
        self.verified.session_evidence_digest()
    }

    pub const fn opened_at_unix_ms(&self) -> u64 {
        self.verified.opened_at_unix_ms()
    }

    pub fn attestor_id(&self) -> &str {
        &self.attestor_id
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub const fn transport_key_digest(&self) -> Digest32 {
        self.transport_key_digest
    }

    pub const fn revalidated_at_unix_ms(&self) -> u64 {
        self.revalidated_at_unix_ms
    }

    pub const fn receipt_expires_at_unix_ms(&self) -> u64 {
        self.receipt_expires_at_unix_ms
    }

    pub const fn transport_key_not_after_unix_ms(&self) -> u64 {
        self.transport_key_not_after_unix_ms
    }

    pub const fn trust_snapshot_expires_at_unix_ms(&self) -> u64 {
        self.trust_snapshot_expires_at_unix_ms
    }

    /// Exclusive earliest transport-validity deadline. This is not itself authority.
    pub const fn valid_until_unix_ms(&self) -> u64 {
        self.valid_until_unix_ms
    }
}

/// Fail-closed error at the current Xenia transport revalidation boundary.
#[derive(Debug, Error)]
pub enum CurrentTransportRevalidationError {
    #[error("transport trust registry does not match independently anchored current head")]
    RegistryHeadNotAnchored,
    #[error("original physical-effect transport generation is no longer current")]
    OriginalTransportGenerationNotCurrent,
    #[error("fixed current verification returned a different transport-trust head")]
    CurrentVerificationHeadMismatch,
    #[error("current verification changed the signed receipt commitment")]
    ReceiptCommitmentMismatch,
    #[error("current verification changed the exact payload commitment")]
    PayloadCommitmentMismatch,
    #[error("current verification changed the semantic physical-envelope commitment")]
    EnvelopeCommitmentMismatch,
    #[error("current verification changed the authenticated Xenia peer")]
    PeerIdentityMismatch,
    #[error("current verification changed the authenticated-session evidence commitment")]
    SessionEvidenceMismatch,
    #[error("current verification changed the receiver-local opening time")]
    OpenedAtMismatch,
    #[error("fixed current verification succeeded but the exact current key record is missing")]
    CurrentTransportKeyMissing,
    #[error("the earliest current transport validity deadline has already elapsed")]
    CurrentTransportWindowElapsed,
    #[error("system clock is before the Unix epoch")]
    ClockBeforeUnixEpoch,
    #[error("system clock cannot be represented as Unix milliseconds")]
    ClockOverflow,
    #[error("fixed Xenia transport verification failed: {0}")]
    Transport(#[from] TransportReceiptError),
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use ed25519_dalek::{Signer as Ed25519Signer, SigningKey};
    use fips204::{
        ml_dsa_65,
        traits::{KeyGen, SerDes, Signer as MlDsaSigner},
    };
    use symthaea_action_checkpoint::CheckpointHead;
    use symthaea_authority::{Operation, PrincipalId, ResourceRef, TaskId};
    use symthaea_iot_authority::{DEVICE_COMMAND_SCHEMA_VERSION, DeviceCommand};
    use symthaea_iot_device_protocol::PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION;
    use symthaea_iot_durable_runtime::DurableIoTHead;
    use symthaea_iot_policy::ActuationPolicyHead;
    use symthaea_iot_posture::VerifierTrustHead;
    use symthaea_iot_transport_exact_evidence::bind_exact_xenia_transport_evidence;
    use symthaea_iot_transport_receipt::{
        HybridReceiptSignatureVerifier, TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
        TransportAttestorKeyV1, TransportAttestorStatus, TransportTrustSnapshotV1,
        XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA, XENIA_ED25519_SIGNATURE_LEN,
        XENIA_HYBRID_SIGNATURE_SUITE, XENIA_ML_DSA_65_SIGNATURE_LEN,
        XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE, XeniaAuthenticatedPayloadReceiptBodyV1,
        XeniaAuthenticatedPayloadReceiptV1, XeniaReceiptPeerRoleV1,
        verify_xenia_transport_receipt,
    };

    use super::*;

    struct PermissiveHistoricalVerifier;

    impl HybridReceiptSignatureVerifier for PermissiveHistoricalVerifier {
        fn verify_ed25519(
            &self,
            _public_key: &[u8; 32],
            _digest: &[u8; 32],
            _signature: &[u8; XENIA_ED25519_SIGNATURE_LEN],
        ) -> bool {
            true
        }

        fn verify_ml_dsa_65(
            &self,
            _public_key: &[u8],
            _digest: &[u8; 32],
            _signature: &[u8; XENIA_ML_DSA_65_SIGNATURE_LEN],
        ) -> bool {
            true
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

    fn trusted_key_with_not_after(not_after_unix_ms: u64) -> TransportAttestorKeyV1 {
        let ed25519 = SigningKey::from_bytes(&[0x66; 32]);
        let (ml_dsa_public, _) = ml_dsa_65::KG::keygen_from_seed(&[0x77; 32]);
        TransportAttestorKeyV1 {
            attestor_id: "xenia-gateway-a".into(),
            key_id: "transport-key-1".into(),
            ed25519_public_key: ed25519.verifying_key().to_bytes(),
            ml_dsa_public_key: ml_dsa_public.into_bytes().to_vec(),
            status: TransportAttestorStatus::Active,
            not_before_unix_ms: 90_000,
            not_after_unix_ms,
            max_receipt_lifetime_ms: 2_000,
            required_peer_role: XeniaReceiptPeerRoleV1::Viewer,
            allowed_peer_fingerprints: BTreeSet::from([[0x44; 32]]),
            require_input_control: true,
        }
    }

    fn trusted_key() -> TransportAttestorKeyV1 {
        trusted_key_with_not_after(130_000)
    }

    fn registry_with_windows(
        key_not_after_unix_ms: u64,
        snapshot_expires_at_unix_ms: u64,
    ) -> TransportTrustRegistry {
        TransportTrustRegistry::genesis(TransportTrustSnapshotV1 {
            schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_ms: 90_000,
            expires_at_unix_ms: snapshot_expires_at_unix_ms,
            previous_snapshot_digest: None,
            keys: vec![trusted_key_with_not_after(key_not_after_unix_ms)],
        })
        .unwrap()
    }

    fn registry() -> TransportTrustRegistry {
        registry_with_windows(130_000, 130_000)
    }

    fn signed_receipt(payload: &[u8]) -> XeniaAuthenticatedPayloadReceiptV1 {
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
        let ed25519 = SigningKey::from_bytes(&[0x66; 32]);
        let (_, ml_dsa_private) = ml_dsa_65::KG::keygen_from_seed(&[0x77; 32]);
        let ml_dsa_signature = ml_dsa_private
            .try_sign_with_seed(&[0x88; 32], &digest, &[])
            .expect("deterministic ML-DSA-65 test signature");
        XeniaAuthenticatedPayloadReceiptV1 {
            body,
            ed25519_signature: ed25519.sign(&digest).to_bytes(),
            ml_dsa_signature,
        }
    }

    fn exact_fixture_with_windows(
        key_not_after_unix_ms: u64,
        snapshot_expires_at_unix_ms: u64,
    ) -> (
        TransportTrustRegistry,
        TransportTrustHead,
        ExactXeniaTransportEvidence,
    ) {
        let registry =
            registry_with_windows(key_not_after_unix_ms, snapshot_expires_at_unix_ms);
        let head = registry.head();
        let payload = bincode::serialize(&envelope(7)).unwrap();
        let raw_receipt = bincode::serialize(&signed_receipt(&payload)).unwrap();
        let verified = verify_xenia_physical_effect_receipt(
            &registry,
            &raw_receipt,
            &payload,
            113_000,
        )
        .unwrap();
        let exact = bind_exact_xenia_transport_evidence(verified, &raw_receipt, &payload).unwrap();
        (registry, head, exact)
    }

    fn exact_fixture() -> (
        TransportTrustRegistry,
        TransportTrustHead,
        ExactXeniaTransportEvidence,
    ) {
        exact_fixture_with_windows(130_000, 130_000)
    }

    #[test]
    fn exact_evidence_revalidates_under_same_current_generation() {
        let (registry, head, exact) = exact_fixture();
        let exact_digest = exact.exact_evidence_digest();
        let guard = CurrentXeniaTransportRevalidator::new(registry, head).unwrap();

        let current = guard.revalidate_at(exact, 113_500).unwrap();

        assert_eq!(current.envelope().command.sequence, 7);
        assert_eq!(current.transport_trust_head(), head);
        assert_eq!(current.exact_evidence_digest(), exact_digest);
        assert_eq!(current.peer_identity_fingerprint(), [0x44; 32]);
        assert_eq!(current.attestor_id(), "xenia-gateway-a");
        assert_eq!(current.key_id(), "transport-key-1");
        assert_eq!(current.transport_key_digest(), trusted_key().digest().unwrap());
        assert_eq!(current.opened_at_unix_ms(), 112_000);
        assert_eq!(current.revalidated_at_unix_ms(), 113_500);
        assert_eq!(current.receipt_expires_at_unix_ms(), 114_000);
        assert_eq!(current.transport_key_not_after_unix_ms(), 130_000);
        assert_eq!(current.trust_snapshot_expires_at_unix_ms(), 130_000);
        assert_eq!(current.valid_until_unix_ms(), 114_000);
    }

    #[test]
    fn independently_anchored_head_must_match_registry() {
        let registry = registry();
        let wrong = TransportTrustHead {
            sequence: registry.head().sequence,
            digest: Digest32([0xFF; 32]),
        };
        assert!(matches!(
            CurrentXeniaTransportRevalidator::new(registry, wrong),
            Err(CurrentTransportRevalidationError::RegistryHeadNotAnchored)
        ));
    }

    #[test]
    fn successor_generation_forces_fresh_xenia_submission_even_if_key_stays_active() {
        let (registry, original_head, exact) = exact_fixture();
        let successor = registry
            .successor(TransportTrustSnapshotV1 {
                schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
                sequence: 2,
                issued_at_unix_ms: 113_100,
                expires_at_unix_ms: 130_000,
                previous_snapshot_digest: Some(original_head.digest),
                keys: vec![trusted_key()],
            })
            .unwrap();
        let current_head = successor.head();
        let guard = CurrentXeniaTransportRevalidator::new(successor, current_head).unwrap();

        assert!(matches!(
            guard.revalidate_at(exact, 113_500),
            Err(CurrentTransportRevalidationError::OriginalTransportGenerationNotCurrent)
        ));
    }

    #[test]
    fn receipt_expiry_is_rechecked_at_current_boundary() {
        let (registry, head, exact) = exact_fixture();
        let guard = CurrentXeniaTransportRevalidator::new(registry, head).unwrap();

        assert!(matches!(
            guard.revalidate_at(exact, 114_000),
            Err(CurrentTransportRevalidationError::Transport(
                TransportReceiptError::ReceiptNotFresh
            ))
        ));
    }

    #[test]
    fn natural_key_and_snapshot_expiry_cap_revalidated_validity() {
        let (registry, head, exact) = exact_fixture_with_windows(113_650, 113_900);
        let guard = CurrentXeniaTransportRevalidator::new(registry, head).unwrap();
        let key_bounded = guard.revalidate_at(exact, 113_500).unwrap();
        assert_eq!(key_bounded.valid_until_unix_ms(), 113_650);
        assert_eq!(key_bounded.transport_key_not_after_unix_ms(), 113_650);

        let (registry, head, exact) = exact_fixture_with_windows(113_900, 113_650);
        let guard = CurrentXeniaTransportRevalidator::new(registry, head).unwrap();
        let snapshot_bounded = guard.revalidate_at(exact, 113_500).unwrap();
        assert_eq!(snapshot_bounded.valid_until_unix_ms(), 113_650);
        assert_eq!(
            snapshot_bounded.trust_snapshot_expires_at_unix_ms(),
            113_650
        );
    }

    #[test]
    fn permissive_historical_verifier_does_not_survive_fixed_current_revalidation() {
        let registry = registry();
        let head = registry.head();
        let payload = bincode::serialize(&envelope(7)).unwrap();
        let mut invalid = signed_receipt(&payload);
        invalid.ed25519_signature = [0; XENIA_ED25519_SIGNATURE_LEN];
        invalid.ml_dsa_signature = [0; XENIA_ML_DSA_65_SIGNATURE_LEN];
        let raw_receipt = bincode::serialize(&invalid).unwrap();

        let historical = verify_xenia_transport_receipt(
            &registry,
            &raw_receipt,
            &payload,
            113_000,
            &PermissiveHistoricalVerifier,
        )
        .unwrap();
        let exact =
            bind_exact_xenia_transport_evidence(historical, &raw_receipt, &payload).unwrap();
        let guard = CurrentXeniaTransportRevalidator::new(registry, head).unwrap();

        assert!(matches!(
            guard.revalidate_at(exact, 113_500),
            Err(CurrentTransportRevalidationError::Transport(
                TransportReceiptError::Ed25519SignatureInvalid
            ))
        ));
    }
}
