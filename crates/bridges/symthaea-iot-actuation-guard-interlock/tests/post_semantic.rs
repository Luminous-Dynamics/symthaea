use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signer, SigningKey};
use symthaea_action_checkpoint::CheckpointHead;
use symthaea_authority::{Digest32, Operation, PrincipalId, ResourceRef};
use symthaea_iot_actuation_guard_admission_challenge::{
    ADMISSION_DEVICE_REALITY_RESPONSE_SCHEMA_VERSION, AdmissionDeviceRealityResponseV1,
    AdmissionRealityChallengeV1, decode_admission_device_reality_response,
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
use symthaea_iot_actuation_guard_interlock::{
    GuardInterlockState, PostSemanticGuardInterlockError,
};
use symthaea_iot_actuation_guard_post_semantic_controller::{
    POST_SEMANTIC_CONTROLLER_RESPONSE_SCHEMA_VERSION, PostSemanticControllerChallengeV1,
    PostSemanticControllerResponseV1, decode_post_semantic_controller_response,
};
use symthaea_iot_actuation_guard_semantic_persistence::{
    DurableSemanticAcceptanceStore, PersistedSemanticAcceptance,
};
use symthaea_iot_actuation_guard_two_phase_protocol::{
    POST_RESERVATION_INTERLOCK_STATEMENT_SCHEMA_VERSION, PostReservationInterlockReportV1,
    PostReservationInterlockStatementV1,
};
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
use symthaea_iot_final_gate::{
    PHYSICAL_INTERLOCK_POLICY_SCHEMA_VERSION, PhysicalInterlockPolicyV1,
};
use symthaea_iot_interlock_ed25519::{
    INTERLOCK_ED25519_ALGORITHM, interlock_ed25519_signing_message,
};
use symthaea_iot_interlock_trust::{
    INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION, InterlockControllerKeyStatus,
    InterlockControllerKeyV1, InterlockTrustRegistry, InterlockTrustSnapshotV1,
};
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
        "symthaea-post-semantic-interlock-{label}-{}-{nanos}",
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
            command_id: "cmd-post-semantic-interlock-1".into(),
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
    let registry = TransportTrustRegistry::genesis(TransportTrustSnapshotV1 {
        schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
        sequence: 1,
        issued_at_unix_ms: now_ms.saturating_sub(2_000),
        expires_at_unix_ms: now_ms + 60_000,
        previous_snapshot_digest: None,
        keys: vec![TransportAttestorKeyV1 {
            attestor_id: "attestor:post-semantic-interlock".into(),
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
    })
    .unwrap();
    let body = XeniaAuthenticatedPayloadReceiptBodyV1 {
        schema: XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA.into(),
        attestor_id: "attestor:post-semantic-interlock".into(),
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
    let registry = DeviceRealityTrustRegistry::genesis(DeviceRealityTrustSnapshotV1 {
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
    })
    .unwrap();
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

fn signed_device_response(
    signing_key: &SigningKey,
    challenge: &AdmissionRealityChallengeV1,
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

fn semantic_acceptance(
    admission_root: &PathBuf,
    semantic_root: &PathBuf,
) -> PersistedSemanticAcceptance {
    let now = wall_ms();
    let admission_store = DurableAdmissionReservationStore::open(admission_root, config()).unwrap();
    let reservation: PersistedAdmissionReservation = admission_store
        .reserve_verified_transport(verified_transport(now))
        .unwrap();
    let challenge =
        AdmissionRealityChallengeV1::issue_from_persisted_reservation(&reservation).unwrap();
    let device_key = SigningKey::from_bytes(&[0x61; 32]);
    let reality_state = reality_state(
        &device_key,
        reservation.persisted_at_unix_ms().saturating_sub(1_000),
        challenge.expires_at_unix_ms().saturating_add(5_000),
    );
    let frame = signed_device_response(&device_key, &challenge);
    let decoded = decode_admission_device_reality_response(&frame, &challenge).unwrap();
    let reality: VerifiedAdmissionDeviceReality = reality_state
        .verify_admission_evidence(decoded, &challenge)
        .unwrap();

    let cfg = config();
    let genesis_head = DeviceSemanticCheckpointV1::genesis(&cfg)
        .unwrap()
        .head()
        .unwrap();
    DurableSemanticAcceptanceStore::open(semantic_root, cfg, genesis_head)
        .unwrap()
        .persist_semantic_acceptance(reservation, reality)
        .unwrap()
}

fn physical_policy(required_interlocks: BTreeSet<String>) -> PhysicalInterlockPolicyV1 {
    PhysicalInterlockPolicyV1 {
        schema_version: PHYSICAL_INTERLOCK_POLICY_SCHEMA_VERSION,
        device: ResourceRef("iot:valve:72".into()),
        allowed_controllers: BTreeSet::from(["controller:valve-72".into()]),
        required_interlocks,
        max_report_lifetime_ms: 1_000,
    }
}

fn interlock_state(
    controller_key: &SigningKey,
    challenge: &PostSemanticControllerChallengeV1,
    status: InterlockControllerKeyStatus,
    required_interlocks: BTreeSet<String>,
) -> GuardInterlockState {
    let policy = physical_policy(required_interlocks);
    let policy_digest = policy.digest().unwrap();
    let issued_at = challenge.issued_at_unix_ms().saturating_sub(1_000);
    let expires_at = challenge.expires_at_unix_ms().saturating_add(5_000);
    let registry = InterlockTrustRegistry::genesis(InterlockTrustSnapshotV1 {
        schema_version: INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION,
        sequence: 1,
        issued_at_unix_ms: issued_at,
        expires_at_unix_ms: expires_at,
        previous_snapshot_digest: None,
        keys: vec![InterlockControllerKeyV1 {
            controller_id: "controller:valve-72".into(),
            key_id: "controller-key-1".into(),
            algorithm: INTERLOCK_ED25519_ALGORITHM.into(),
            public_key: controller_key.verifying_key().to_bytes().to_vec(),
            status,
            not_before_unix_ms: issued_at,
            not_after_unix_ms: expires_at,
        }],
    })
    .unwrap();
    let head = registry.head();
    GuardInterlockState::new(policy, policy_digest, registry, head).unwrap()
}

fn controller_frame(
    controller_key: &SigningKey,
    challenge: &PostSemanticControllerChallengeV1,
    asserted_interlocks: BTreeSet<String>,
) -> Vec<u8> {
    let checked_at = wall_ms().max(challenge.issued_at_unix_ms());
    let expires_at = challenge
        .expires_at_unix_ms()
        .min(checked_at.saturating_add(500));
    assert!(expires_at > checked_at);
    let statement = PostReservationInterlockStatementV1 {
        schema_version: POST_RESERVATION_INTERLOCK_STATEMENT_SCHEMA_VERSION,
        challenge_digest: challenge.digest().unwrap(),
        device_attestation_result_digest: challenge.device_attestation_object_digest(),
        controller_id: "controller:valve-72".into(),
        device: challenge.device().clone(),
        envelope_digest: challenge.envelope_digest(),
        semantic_head: challenge.semantic_head(),
        transport_trust_head: challenge.transport_trust_head(),
        asserted_interlocks,
        checked_at_unix_ms: checked_at,
        expires_at_unix_ms: expires_at,
    };
    let statement_digest = statement.digest().unwrap();
    let signature = controller_key
        .sign(&interlock_ed25519_signing_message(statement_digest))
        .to_bytes();
    let report = PostReservationInterlockReportV1 {
        statement,
        evidence_digest: Digest32(*blake3::hash(&signature).as_bytes()),
    };
    PostSemanticControllerResponseV1 {
        schema_version: POST_SEMANTIC_CONTROLLER_RESPONSE_SCHEMA_VERSION,
        raw_interlock_report: bincode::serialize(&report).unwrap(),
        raw_interlock_evidence: signature.to_vec(),
    }
    .canonical_bytes()
    .unwrap()
}

fn exact_interlocks() -> BTreeSet<String> {
    BTreeSet::from([
        "manual-stop-clear".into(),
        "pressure-within-range".into(),
    ])
}

#[test]
fn real_controller_key_and_guard_policy_verify_the_post_semantic_statement() {
    let admission_root = temp_root("admission-ok");
    let semantic_root = temp_root("semantic-ok");
    let semantic = semantic_acceptance(&admission_root, &semantic_root);
    let challenge = PostSemanticControllerChallengeV1::issue_from_persisted_semantic_acceptance(
        &semantic,
    )
    .unwrap();
    let controller_key = SigningKey::from_bytes(&[0x71; 32]);
    let state = interlock_state(
        &controller_key,
        &challenge,
        InterlockControllerKeyStatus::Active,
        exact_interlocks(),
    );
    let frame = controller_frame(&controller_key, &challenge, exact_interlocks());
    let decoded = decode_post_semantic_controller_response(&frame, &challenge).unwrap();
    let expected_statement_digest = decoded.report().statement.digest().unwrap();
    let proof = state
        .verify_post_semantic_controller(decoded, challenge)
        .unwrap();

    assert_eq!(proof.statement_digest(), expected_statement_digest);
    assert_eq!(proof.controller_id(), "controller:valve-72");
    assert_eq!(proof.controller_key_id(), "controller-key-1");
    assert_ne!(proof.controller_key_digest(), Digest32([0; 32]));
    assert_ne!(proof.evidence_digest(), Digest32([0; 32]));

    std::fs::remove_dir_all(admission_root).unwrap();
    std::fs::remove_dir_all(semantic_root).unwrap();
}

#[test]
fn wrong_controller_key_is_rejected_even_when_report_commitment_is_self_consistent() {
    let admission_root = temp_root("admission-wrong-key");
    let semantic_root = temp_root("semantic-wrong-key");
    let semantic = semantic_acceptance(&admission_root, &semantic_root);
    let challenge = PostSemanticControllerChallengeV1::issue_from_persisted_semantic_acceptance(
        &semantic,
    )
    .unwrap();
    let trusted_key = SigningKey::from_bytes(&[0x71; 32]);
    let attacker_key = SigningKey::from_bytes(&[0x72; 32]);
    let state = interlock_state(
        &trusted_key,
        &challenge,
        InterlockControllerKeyStatus::Active,
        exact_interlocks(),
    );
    let frame = controller_frame(&attacker_key, &challenge, exact_interlocks());
    let decoded = decode_post_semantic_controller_response(&frame, &challenge).unwrap();
    assert!(state.verify_post_semantic_controller(decoded, challenge).is_err());

    std::fs::remove_dir_all(admission_root).unwrap();
    std::fs::remove_dir_all(semantic_root).unwrap();
}

#[test]
fn revoked_controller_key_is_rejected() {
    let admission_root = temp_root("admission-revoked");
    let semantic_root = temp_root("semantic-revoked");
    let semantic = semantic_acceptance(&admission_root, &semantic_root);
    let challenge = PostSemanticControllerChallengeV1::issue_from_persisted_semantic_acceptance(
        &semantic,
    )
    .unwrap();
    let controller_key = SigningKey::from_bytes(&[0x71; 32]);
    let state = interlock_state(
        &controller_key,
        &challenge,
        InterlockControllerKeyStatus::Revoked,
        exact_interlocks(),
    );
    let frame = controller_frame(&controller_key, &challenge, exact_interlocks());
    let decoded = decode_post_semantic_controller_response(&frame, &challenge).unwrap();
    assert!(state.verify_post_semantic_controller(decoded, challenge).is_err());

    std::fs::remove_dir_all(admission_root).unwrap();
    std::fs::remove_dir_all(semantic_root).unwrap();
}

#[test]
fn exact_guard_owned_interlock_set_is_required_after_valid_signature() {
    let admission_root = temp_root("admission-policy");
    let semantic_root = temp_root("semantic-policy");
    let semantic = semantic_acceptance(&admission_root, &semantic_root);
    let challenge = PostSemanticControllerChallengeV1::issue_from_persisted_semantic_acceptance(
        &semantic,
    )
    .unwrap();
    let controller_key = SigningKey::from_bytes(&[0x71; 32]);
    let state = interlock_state(
        &controller_key,
        &challenge,
        InterlockControllerKeyStatus::Active,
        exact_interlocks(),
    );
    let incomplete = BTreeSet::from(["manual-stop-clear".into()]);
    let frame = controller_frame(&controller_key, &challenge, incomplete);
    let decoded = decode_post_semantic_controller_response(&frame, &challenge).unwrap();
    assert!(matches!(
        state.verify_post_semantic_controller(decoded, challenge),
        Err(PostSemanticGuardInterlockError::InterlockSetMismatch)
    ));

    std::fs::remove_dir_all(admission_root).unwrap();
    std::fs::remove_dir_all(semantic_root).unwrap();
}
