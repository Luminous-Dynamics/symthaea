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
use symthaea_iot_actuation_guard_admission_reservation::{
    DurableAdmissionReservationStore, PersistedAdmissionReservation,
};
use symthaea_iot_actuation_guard_device_reality::{
    DEVICE_REALITY_ED25519_ALGORITHM, DEVICE_REALITY_POLICY_SCHEMA_VERSION,
    DEVICE_REALITY_TRUST_SCHEMA_VERSION, DeviceRealityPolicyV1, DeviceRealityTrustRegistry,
    DeviceRealityTrustSnapshotV1, DeviceRealityVerifierKeyStatus, DeviceRealityVerifierKeyV1,
    GuardAdmissionDeviceRealityState, VerifiedAdmissionDeviceReality,
};
use symthaea_iot_actuation_guard_semantic_persistence::DurableSemanticAcceptanceStore;
use symthaea_iot_authority::{
    DEVICE_COMMAND_SCHEMA_VERSION, DeviceCommand, InclusiveRangeI64,
    SAFETY_ENVELOPE_SCHEMA_VERSION, SafetyEnvelope,
};
use symthaea_iot_device_protocol::{
    DEVICE_ENFORCEMENT_CONFIG_SCHEMA_VERSION, DeviceEnforcementConfigV1,
    DeviceSemanticCheckpointV1, PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION,
    PhysicalEffectEnvelopeV1,
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

fn temp_root(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "symthaea-semantic-{label}-{}-{nanos}",
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
            command_id: "cmd-semantic-1".into(),
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
            attestor_id: "attestor:semantic".into(),
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
        attestor_id: "attestor:semantic".into(),
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

fn reality_policy() -> DeviceRealityPolicyV1 {
    DeviceRealityPolicyV1 {
        schema_version: DEVICE_REALITY_POLICY_SCHEMA_VERSION,
        device: ResourceRef("iot:valve:72".into()),
        allowed_verifier_ids: BTreeSet::from(["verifier:fleet-a".into()]),
        accepted_reference_values: BTreeSet::from([d(0x32)]),
        exact_appraisal_policy_digest: d(0x33),
        max_result_lifetime_ms: 2_000,
    }
}

fn reality_state(
    signing_key: &SigningKey,
    issued_before_unix_ms: u64,
    expires_after_unix_ms: u64,
) -> GuardAdmissionDeviceRealityState {
    let policy = reality_policy();
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

fn signed_response(
    signing_key: &SigningKey,
    challenge: &AdmissionRealityChallengeV1,
    pressure_x100: i64,
) -> Vec<u8> {
    let (appraisal_s, expiry_s) = appraisal_window(challenge);
    let body = DeviceAttestationResultBodyV1 {
        schema_version: DEVICE_ATTESTATION_RESULT_SCHEMA_VERSION,
        verifier_id: "verifier:fleet-a".into(),
        key_id: "device-key-1".into(),
        algorithm: DEVICE_REALITY_ED25519_ALGORITHM.into(),
        device: challenge.device().clone(),
        challenge_digest: challenge.digest().unwrap(),
        appraised_at_unix_s: appraisal_s,
        expires_at_unix_s: expiry_s,
        evidence_digest: d(0x31),
        reference_values_digest: d(0x32),
        appraisal_policy_digest: d(0x33),
        running_firmware: d(7),
        last_accepted_sequence: None,
        observations: BTreeMap::from([("pressure_x100".into(), pressure_x100)]),
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

fn admission_and_reality(
    admission_root: &PathBuf,
    pressure_x100: i64,
) -> (PersistedAdmissionReservation, VerifiedAdmissionDeviceReality) {
    let now = wall_ms();
    let admission_store = DurableAdmissionReservationStore::open(admission_root, config()).unwrap();
    let reservation = admission_store
        .reserve_verified_transport(verified_transport(now))
        .unwrap();
    let challenge =
        AdmissionRealityChallengeV1::issue_from_persisted_reservation(&reservation).unwrap();
    let signing_key = SigningKey::from_bytes(&[0x61; 32]);
    let state = reality_state(
        &signing_key,
        reservation.persisted_at_unix_ms().saturating_sub(1_000),
        challenge.expires_at_unix_ms().saturating_add(5_000),
    );
    let frame = signed_response(&signing_key, &challenge, pressure_x100);
    let decoded = decode_admission_device_reality_response(&frame, &challenge).unwrap();
    let proof = state.verify_admission_evidence(decoded, &challenge).unwrap();
    (reservation, proof)
}

#[test]
fn semantic_acceptance_is_durable_only_after_authenticated_device_reality() {
    let admission_root = temp_root("admission-ok");
    let semantic_root = temp_root("semantic-ok");
    let cfg = config();
    let genesis = DeviceSemanticCheckpointV1::genesis(&cfg).unwrap();
    let genesis_head = genesis.head().unwrap();
    let (reservation, reality) = admission_and_reality(&admission_root, 210_000);

    let store = DurableSemanticAcceptanceStore::open(&semantic_root, cfg.clone(), genesis_head)
        .unwrap();
    let persisted = store
        .persist_semantic_acceptance(reservation, reality)
        .unwrap();

    assert_eq!(persisted.device_head().generation, 1);
    assert_eq!(persisted.checkpoint().highest_accepted_sequence, Some(1));
    assert_eq!(
        persisted.checkpoint().last_envelope_digest,
        Some(persisted.envelope_digest())
    );
    assert_eq!(
        persisted.semantic_effect().device_head(),
        persisted.device_head()
    );
    assert!(
        persisted.semantic_persisted_at_unix_ms()
            >= persisted.device_reality().verified_at_unix_ms()
    );
    assert_ne!(
        persisted.device_attestation_object_digest(),
        Digest32([0; 32])
    );

    // Reopening against the new independently retained head succeeds; the stale genesis
    // anchor fails closed against the newer disk checkpoint.
    DurableSemanticAcceptanceStore::open(&semantic_root, cfg.clone(), persisted.device_head())
        .unwrap();
    assert!(
        DurableSemanticAcceptanceStore::open(&semantic_root, cfg, genesis_head).is_err()
    );

    std::fs::remove_dir_all(admission_root).unwrap();
    std::fs::remove_dir_all(semantic_root).unwrap();
}

#[test]
fn authenticated_but_unsafe_observation_never_advances_semantic_disk_head() {
    let admission_root = temp_root("admission-unsafe");
    let semantic_root = temp_root("semantic-unsafe");
    let cfg = config();
    let genesis_head = DeviceSemanticCheckpointV1::genesis(&cfg)
        .unwrap()
        .head()
        .unwrap();
    let (reservation, reality) = admission_and_reality(&admission_root, 400_000);

    let store = DurableSemanticAcceptanceStore::open(&semantic_root, cfg.clone(), genesis_head)
        .unwrap();
    assert!(store.persist_semantic_acceptance(reservation, reality).is_err());

    // Semantic policy failed before any successor write, so the genesis anchor still opens.
    DurableSemanticAcceptanceStore::open(&semantic_root, cfg, genesis_head).unwrap();

    std::fs::remove_dir_all(admission_root).unwrap();
    std::fs::remove_dir_all(semantic_root).unwrap();
}
