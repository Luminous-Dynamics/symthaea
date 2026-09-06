use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signer as Ed25519Signer, SigningKey};
use fips204::{
    ml_dsa_65,
    traits::{KeyGen, SerDes, Signer as MlDsaSigner},
};
use symthaea_action_checkpoint::CheckpointHead;
use symthaea_authority::{Digest32, Operation, PrincipalId, ResourceRef, TaskId};
use symthaea_iot_authority::{DEVICE_COMMAND_SCHEMA_VERSION, DeviceCommand};
use symthaea_iot_device_protocol::{
    PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION, PhysicalEffectEnvelopeV1,
};
use symthaea_iot_durable_runtime::DurableIoTHead;
use symthaea_iot_policy::ActuationPolicyHead;
use symthaea_iot_posture::VerifierTrustHead;
use symthaea_iot_transport_current_fence::{
    CurrentXeniaTransportFenceError, CurrentXeniaTransportFenceGuard,
};
use symthaea_iot_transport_current_revalidation::CurrentXeniaTransportRevalidator;
use symthaea_iot_transport_exact_evidence::bind_exact_xenia_transport_evidence;
use symthaea_iot_transport_receipt::{
    HybridReceiptSignatureVerifier, TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
    TransportAttestorKeyV1, TransportAttestorStatus, TransportTrustRegistry,
    TransportTrustSnapshotV1, XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA,
    XENIA_ED25519_SIGNATURE_LEN, XENIA_HYBRID_SIGNATURE_SUITE,
    XENIA_ML_DSA_65_SIGNATURE_LEN, XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE,
    XeniaAuthenticatedPayloadReceiptBodyV1, XeniaAuthenticatedPayloadReceiptV1,
    XeniaReceiptPeerRoleV1, verify_xenia_transport_receipt,
};

struct HistoricalFixtureVerifier;

impl HybridReceiptSignatureVerifier for HistoricalFixtureVerifier {
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

fn wall_ms() -> u64 {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    )
    .unwrap()
}

fn d(byte: u8) -> Digest32 {
    Digest32([byte; 32])
}

fn envelope(now_ms: u64) -> PhysicalEffectEnvelopeV1 {
    let now_s = now_ms / 1_000;
    PhysicalEffectEnvelopeV1 {
        schema_version: PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION,
        command: DeviceCommand {
            schema_version: DEVICE_COMMAND_SCHEMA_VERSION,
            command_id: "cmd-transport-current-fence".into(),
            actor: PrincipalId("agent:irrigation".into()),
            executor: PrincipalId("gateway:field-a".into()),
            task: Some(TaskId("irrigate:zone-7".into())),
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            expected_firmware: d(7),
            sequence: 1,
            issued_at_unix_s: now_s.saturating_sub(1),
            expires_at_unix_s: now_s + 10,
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
        posture_expires_at_unix_s: now_s + 10,
        host_preflight_at_unix_s: now_s,
        send_not_after_unix_s: now_s + 8,
    }
}

#[test]
fn fixed_current_transport_can_be_borrowed_fenced_and_generation_advance_kills_it() {
    let now = wall_ms();
    let ed25519 = SigningKey::from_bytes(&[0x66; 32]);
    let (ml_dsa_public, ml_dsa_private) = ml_dsa_65::KG::keygen_from_seed(&[0x77; 32]);

    let key_not_after = now + 4_800;
    let snapshot_expires = now + 5_000;
    let receipt_expires = now + 4_500;
    let key = TransportAttestorKeyV1 {
        attestor_id: "xenia-gateway-a".into(),
        key_id: "transport-key-1".into(),
        ed25519_public_key: ed25519.verifying_key().to_bytes(),
        ml_dsa_public_key: ml_dsa_public.into_bytes().to_vec(),
        status: TransportAttestorStatus::Active,
        not_before_unix_ms: now.saturating_sub(5_000),
        not_after_unix_ms: key_not_after,
        max_receipt_lifetime_ms: 5_000,
        required_peer_role: XeniaReceiptPeerRoleV1::Viewer,
        allowed_peer_fingerprints: BTreeSet::from([[0x44; 32]]),
        require_input_control: true,
    };
    let snapshot = TransportTrustSnapshotV1 {
        schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
        sequence: 1,
        issued_at_unix_ms: now.saturating_sub(2_000),
        expires_at_unix_ms: snapshot_expires,
        previous_snapshot_digest: None,
        keys: vec![key.clone()],
    };
    let registry = TransportTrustRegistry::genesis(snapshot.clone()).unwrap();
    let head = registry.head();

    let payload = bincode::serialize(&envelope(now)).unwrap();
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
        payload_len: u32::try_from(payload.len()).unwrap(),
        payload_digest: *blake3::hash(&payload).as_bytes(),
        sealed_envelope_digest: [0x47; 32],
        opened_at_unix_ms: now,
        expires_at_unix_ms: receipt_expires,
    };
    let digest = body.signing_digest().unwrap();
    let ml_dsa_signature = ml_dsa_private
        .try_sign_with_seed(&[0x88; 32], &digest, &[])
        .unwrap();
    let receipt = XeniaAuthenticatedPayloadReceiptV1 {
        body,
        ed25519_signature: ed25519.sign(&digest).to_bytes(),
        ml_dsa_signature,
    };
    let raw_receipt = bincode::serialize(&receipt).unwrap();

    let historical = verify_xenia_transport_receipt(
        &registry,
        &raw_receipt,
        &payload,
        now,
        &HistoricalFixtureVerifier,
    )
    .unwrap();
    let exact = bind_exact_xenia_transport_evidence(historical, &raw_receipt, &payload).unwrap();
    let proof = CurrentXeniaTransportRevalidator::new(registry, head)
        .unwrap()
        .revalidate(exact)
        .unwrap();

    assert_eq!(proof.transport_key_digest(), key.digest().unwrap());
    assert_eq!(proof.valid_until_unix_ms(), receipt_expires);

    let current_registry = TransportTrustRegistry::genesis(snapshot.clone()).unwrap();
    let current_head = current_registry.head();
    let current = CurrentXeniaTransportFenceGuard::new(current_registry, current_head).unwrap();
    let fence = current.fence_current(&proof).unwrap();
    assert_eq!(fence.proof().receipt_digest(), proof.receipt_digest());
    assert_eq!(fence.valid_until_unix_ms(), receipt_expires);

    let base = TransportTrustRegistry::genesis(snapshot.clone()).unwrap();
    let successor = base
        .successor(TransportTrustSnapshotV1 {
            schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 2,
            issued_at_unix_ms: snapshot.issued_at_unix_ms,
            expires_at_unix_ms: snapshot.expires_at_unix_ms,
            previous_snapshot_digest: Some(base.head().digest),
            keys: snapshot.keys.clone(),
        })
        .unwrap();
    let successor_head = successor.head();
    let advanced = CurrentXeniaTransportFenceGuard::new(successor, successor_head).unwrap();
    assert!(matches!(
        advanced.fence_current(&proof),
        Err(CurrentXeniaTransportFenceError::ProofTrustGenerationNotCurrent)
    ));
}
