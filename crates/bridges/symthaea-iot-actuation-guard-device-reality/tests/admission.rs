use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signer, SigningKey};
use symthaea_action_checkpoint::CheckpointHead;
use symthaea_authority::{Digest32, Operation, PrincipalId, ResourceRef};
use symthaea_iot_actuation_guard_admission_challenge::{
    AdmissionDeviceRealityResponseV1, AdmissionRealityChallengeV1,
    ADMISSION_DEVICE_REALITY_RESPONSE_SCHEMA_VERSION, decode_admission_device_reality_response,
};
use symthaea_iot_actuation_guard_admission_reservation::DurableAdmissionReservationStore;
use symthaea_iot_actuation_guard_device_reality::{
    DEVICE_REALITY_ED25519_ALGORITHM, DEVICE_REALITY_POLICY_SCHEMA_VERSION,
    DEVICE_REALITY_TRUST_SCHEMA_VERSION, DeviceRealityPolicyV1, DeviceRealityTrustRegistry,
    DeviceRealityTrustSnapshotV1, DeviceRealityVerifierKeyStatus, DeviceRealityVerifierKeyV1,
    GuardAdmissionDeviceRealityState,
};
use symthaea_iot_authority::{
    DEVICE_COMMAND_SCHEMA_VERSION, DeviceCommand, InclusiveRangeI64,
    SAFETY_ENVELOPE_SCHEMA_VERSION, SafetyEnvelope,
};
use symthaea_iot_device_protocol::{
    DEVICE_ENFORCEMENT_CONFIG_SCHEMA_VERSION, DeviceEnforcementConfigV1,
    PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION, PhysicalEffectEnvelopeV1,
};
use symthaea_iot_durable_runtime::DurableIoTHead;
use symthaea_iot_policy::ActuationPolicyHead;
use symthaea_iot_posture::{
    DEVICE_ATTESTATION_RESULT_SCHEMA_VERSION, DeviceAttestationResultBodyV1,
    DeviceAttestationResultV1, VerifierTrustHead,
};
use symthaea_iot_transport_receipt::{
    HybridReceiptSignatureVerifier, TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
    TransportAttestorKeyV1, TransportAttestorStatus, TransportTrustRegistry,
    TransportTrustSnapshotV1, XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA,
    XENIA_ED25519_SIGNATURE_LEN, XENIA_HYBRID_SIGNATURE_SUITE,
    XENIA_ML_DSA_65_PUBLIC_KEY_LEN, XENIA_ML_DSA_65_SIGNATURE_LEN,
    XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE, XeniaAuthenticatedPayloadReceiptBodyV1,
    XeniaAuthenticatedPayloadReceiptV1, XeniaReceiptPeerRoleV1,
    verify_xenia_transport_receipt,
};

struct AcceptTransportFixtureSignatures;

impl HybridReceiptSignatureVerifier for AcceptTransportFixtureSignatures {
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

fn wall_ms() -> u64 {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    )
    .unwrap()
}

fn temp_root() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "symthaea-admission-device-reality-{}-{nanos}",
        std::process::id()
    ))
}

fn config() -> DeviceEnforcementConfigV1 {
    DeviceEnforcementConfigV1 {
        schema_version: DEVICE_ENFORCEMENT_CONFIG_SCHEMA_VERSION,
        device: ResourceRef("iot:valve:72".into()),
        operation: Operation("valve.open".into()),
        exact_policy_digest: d(20),
        minimum_policy_registry_sequence: 5,
        safety: SafetyEnvelope {
            schema_version: SAFETY_ENVELOPE_SCHEMA_VERSION,
            policy_id: "device-local-safe-open".into(),
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            allowed_firmware: BTreeSet::from([d(7)]),
            parameter_ranges: BTreeMap::from([(
                "duration_ms".into(),
                InclusiveRangeI64 {
                    min: 1_000,
                    max: 120_000,
                },
            )]),
            required_observations: BTreeMap::from([(
                "pressure_x100".into(),
                InclusiveRangeI64 {
                    min: 100,
                    max: 350_000,
                },
            )]),
        },
        maximum_envelope_lifetime_s: 5,
    }
}

fn envelope(now_ms: u64) -> PhysicalEffectEnvelopeV1 {
    let now_s = now_ms / 1_000;
    PhysicalEffectEnvelopeV1 {
        schema_version: PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION,
        command: DeviceCommand {
            schema_version: DEVICE_COMMAND_SCHEMA_VERSION,
            command_id: "cmd-admission-reality-1".into(),
            actor: PrincipalId("agent:irrigation".into()),
            executor: PrincipalId("gateway:field-a".into()),
            task: None,
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            expected_firmware: d(7),
            sequence: 1,
            issued_at_unix_s: now_s.saturating_sub(1),
            expires_at_unix_s: now_s + 10,
            parameters: BTreeMap::from([("duration_ms".into(), 60_000)]),
        },
        proposal_digest: d(10),
        policy_digest: d(20),
        policy_registry_head: ActuationPolicyHead {
            sequence: 5,
            digest: d(21),
        },
        durable_host_head: DurableIoTHead {
            action_head: CheckpointHead {
                sequence: 9,
                digest: d(30),
            },
            digest: d(31),
        },
        posture_result_digest: d(40),
        posture_evidence_digest: d(41),
        posture_reference_values_digest: d(42),
        posture_appraisal_policy_digest: d(43),
        posture_challenge_digest: d(44),
        posture_verifier_trust_head: VerifierTrustHead {
            sequence: 3,
            digest: d(45),
        },
        posture_expires_at_unix_s: now_s + 10,
        host_preflight_at_unix_s: now_s,
        send_not_after_unix_s: now_s + 5,
    }
}

fn verified_transport(now_ms: u64) -> symthaea_iot_transport_receipt::VerifiedTransportEnvelope {
    let envelope = envelope(now_ms);
    let raw_payload = bincode::serialize(&envelope).unwrap();
    let peer = [0x77; 32];
    let snapshot = TransportTrustSnapshotV1 {
        schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
        sequence: 1,
        issued_at_unix_ms: now_ms.saturating_sub(2_000),
        expires_at_unix_ms: now_ms + 60_000,
        previous_snapshot_digest: None,
        keys: vec![TransportAttestorKeyV1 {
            attestor_id: "attestor:admission-reality".into(),
            key_id: "transport-key-1".into(),
            ed25519_public_key: [0x44; 32],
            ml_dsa_public_key: vec![0x55; XENIA_ML_DSA_65_PUBLIC_KEY_LEN],
            status: TransportAttestorStatus::Active,
            not_before_unix_ms: now_ms.saturating_sub(5_000),
            not_after_unix_ms: now_ms + 60_000,
            max_receipt_lifetime_ms: 5_000,
            required_peer_role: XeniaReceiptPeerRoleV1::Host,
            allowed_peer_fingerprints: BTreeSet::from([peer]),
            require_input_control: true,
        }],
    };
    let registry = TransportTrustRegistry::genesis(snapshot).unwrap();
    let body = XeniaAuthenticatedPayloadReceiptBodyV1 {
        schema: XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA.into(),
        attestor_id: "attestor:admission-reality".into(),
        key_id: "transport-key-1".into(),
        signature_algorithm: XENIA_HYBRID_SIGNATURE_SUITE.into(),
        session_evidence_digest: [0x61; 32],
        peer_role: XeniaReceiptPeerRoleV1::Host,
        peer_identity_fingerprint: peer,
        transcript_hash: [0x62; 32],
        session_context_hash: [0x63; 32],
        telemetry_enabled: true,
        input_control_enabled: true,
        payload_type: XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE,
        payload_len: u32::try_from(raw_payload.len()).unwrap(),
        payload_digest: *blake3::hash(&raw_payload).as_bytes(),
        sealed_envelope_digest: [0x64; 32],
        opened_at_unix_ms: now_ms,
        expires_at_unix_ms: now_ms + 4_000,
    };
    let receipt = XeniaAuthenticatedPayloadReceiptV1 {
        body,
        ed25519_signature: [0x71; XENIA_ED25519_SIGNATURE_LEN],
        ml_dsa_signature: [0x72; XENIA_ML_DSA_65_SIGNATURE_LEN],
    };
    verify_xenia_transport_receipt(
        &registry,
        &bincode::serialize(&receipt).unwrap(),
        &raw_payload,
        now_ms,
        &AcceptTransportFixtureSignatures,
    )
    .unwrap()
}

fn policy() -> DeviceRealityPolicyV1 {
    DeviceRealityPolicyV1 {
        schema_version: DEVICE_REALITY_POLICY_SCHEMA_VERSION,
        device: ResourceRef("iot:valve:72".into()),
        allowed_verifier_ids: BTreeSet::from(["verifier:fleet-a".into()]),
        accepted_reference_values: BTreeSet::from([d(0x32)]),
        exact_appraisal_policy_digest: d(0x33),
        max_result_lifetime_ms: 2_000,
    }
}

fn verifier_state(
    signing_key: &SigningKey,
    issued_before_unix_ms: u64,
    expires_after_unix_ms: u64,
) -> GuardAdmissionDeviceRealityState {
    let policy = policy();
    let policy_digest = policy.digest().unwrap();
    let snapshot = DeviceRealityTrustSnapshotV1 {
        schema_version: DEVICE_REALITY_TRUST_SCHEMA_VERSION,
        sequence: 1,
        issued_at_unix_ms: issued_before_unix_ms,
        expires_at_unix_ms: expires_after_unix_ms,
        previous_snapshot_digest: None,
        keys: vec![DeviceRealityVerifierKeyV1 {
            verifier_id: "verifier:fleet-a".into(),
            key_id: "device-key-1".into(),
            algorithm: DEVICE_REALITY_ED25519_ALGORITHM.into(),
            public_key: signing_key.verifying_key().to_bytes(),
            status: DeviceRealityVerifierKeyStatus::Active,
            not_before_unix_ms: issued_before_unix_ms,
            not_after_unix_ms: expires_after_unix_ms,
            max_result_lifetime_ms: 2_000,
        }],
    };
    let registry = DeviceRealityTrustRegistry::genesis(snapshot).unwrap();
    let head = registry.head();
    GuardAdmissionDeviceRealityState::new(policy, policy_digest, registry, head).unwrap()
}

fn signed_response(
    signing_key: &SigningKey,
    challenge: &AdmissionRealityChallengeV1,
    appraised_at_unix_s: u64,
    expires_at_unix_s: u64,
) -> Vec<u8> {
    let body = DeviceAttestationResultBodyV1 {
        schema_version: DEVICE_ATTESTATION_RESULT_SCHEMA_VERSION,
        verifier_id: "verifier:fleet-a".into(),
        key_id: "device-key-1".into(),
        algorithm: DEVICE_REALITY_ED25519_ALGORITHM.into(),
        device: challenge.device().clone(),
        challenge_digest: challenge.digest().unwrap(),
        appraised_at_unix_s,
        expires_at_unix_s,
        evidence_digest: d(0x31),
        reference_values_digest: d(0x32),
        appraisal_policy_digest: d(0x33),
        running_firmware: d(7),
        last_accepted_sequence: None,
        observations: BTreeMap::from([("pressure_x100".into(), 210_000)]),
    };
    let signature = signing_key.sign(&body.signature_message().unwrap()).to_bytes();
    let result = DeviceAttestationResultV1 {
        body,
        signature: signature.to_vec(),
    };
    AdmissionDeviceRealityResponseV1 {
        schema_version: ADMISSION_DEVICE_REALITY_RESPONSE_SCHEMA_VERSION,
        raw_attestation_result: bincode::serialize(&result).unwrap(),
    }
    .canonical_bytes()
    .unwrap()
}

fn appraisal_window(challenge: &AdmissionRealityChallengeV1) -> (u64, u64) {
    let appraisal_s = challenge.issued_at_unix_ms().div_ceil(1_000);
    let appraisal_ms = appraisal_s * 1_000;
    let current_ms = wall_ms();
    if current_ms < appraisal_ms {
        thread::sleep(Duration::from_millis(appraisal_ms - current_ms + 20));
    }
    let expiry_s = appraisal_s + 2;
    assert!(expiry_s * 1_000 <= challenge.expires_at_unix_ms());
    (appraisal_s, expiry_s)
}

#[test]
fn real_ed25519_device_reality_is_bound_to_durable_admission_reservation() {
    let root = temp_root();
    let now = wall_ms();
    let store = DurableAdmissionReservationStore::open(&root, config()).unwrap();
    let reservation = store
        .reserve_verified_transport(verified_transport(now))
        .unwrap();
    let challenge =
        AdmissionRealityChallengeV1::issue_from_persisted_reservation(&reservation).unwrap();
    let (appraisal_s, expiry_s) = appraisal_window(&challenge);

    let signing_key = SigningKey::from_bytes(&[0x61; 32]);
    let state = verifier_state(
        &signing_key,
        reservation.persisted_at_unix_ms().saturating_sub(1_000),
        challenge.expires_at_unix_ms().saturating_add(5_000),
    );
    let frame = signed_response(&signing_key, &challenge, appraisal_s, expiry_s);
    let decoded = decode_admission_device_reality_response(&frame, &challenge).unwrap();
    let object_digest = decoded.attestation_object_digest();
    let proof = state.verify_admission_evidence(decoded, &challenge).unwrap();

    assert_eq!(proof.reservation_head(), reservation.head());
    assert_eq!(proof.envelope_digest(), reservation.envelope_digest());
    assert_eq!(proof.config_digest(), reservation.checkpoint().config_digest);
    assert_eq!(proof.challenge_digest(), challenge.digest().unwrap());
    assert_eq!(proof.attestation_object_digest(), object_digest);
    assert_eq!(proof.runtime_state().running_firmware, d(7));
    assert_eq!(
        proof.runtime_state().observations.get("pressure_x100"),
        Some(&210_000)
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn mutated_device_signature_is_rejected_after_canonical_correlation() {
    let root = temp_root();
    let now = wall_ms();
    let store = DurableAdmissionReservationStore::open(&root, config()).unwrap();
    let reservation = store
        .reserve_verified_transport(verified_transport(now))
        .unwrap();
    let challenge =
        AdmissionRealityChallengeV1::issue_from_persisted_reservation(&reservation).unwrap();
    let (appraisal_s, expiry_s) = appraisal_window(&challenge);

    let signing_key = SigningKey::from_bytes(&[0x61; 32]);
    let state = verifier_state(
        &signing_key,
        reservation.persisted_at_unix_ms().saturating_sub(1_000),
        challenge.expires_at_unix_ms().saturating_add(5_000),
    );
    let frame = signed_response(&signing_key, &challenge, appraisal_s, expiry_s);
    let mut response: AdmissionDeviceRealityResponseV1 = bincode::deserialize(&frame).unwrap();
    let mut result: DeviceAttestationResultV1 =
        bincode::deserialize(&response.raw_attestation_result).unwrap();
    result.signature[0] ^= 1;
    response.raw_attestation_result = bincode::serialize(&result).unwrap();
    let mutated_frame = response.canonical_bytes().unwrap();
    let decoded = decode_admission_device_reality_response(&mutated_frame, &challenge).unwrap();

    assert!(state.verify_admission_evidence(decoded, &challenge).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
