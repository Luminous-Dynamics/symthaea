use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use symthaea_action_checkpoint::CheckpointHead;
use symthaea_authority::{Digest32, Operation, PrincipalId, ResourceRef};
use symthaea_iot_actuation_guard_admission_reservation::{
    AdmissionReservationError, DurableAdmissionReservationStore,
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
use symthaea_iot_posture::VerifierTrustHead;
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

struct AcceptFixtureSignatures;

impl HybridReceiptSignatureVerifier for AcceptFixtureSignatures {
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
        "symthaea-admission-reservation-{label}-{}-{nanos}",
        std::process::id()
    ))
}

fn safety() -> SafetyEnvelope {
    SafetyEnvelope {
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
    }
}

fn config() -> DeviceEnforcementConfigV1 {
    DeviceEnforcementConfigV1 {
        schema_version: DEVICE_ENFORCEMENT_CONFIG_SCHEMA_VERSION,
        device: ResourceRef("iot:valve:72".into()),
        operation: Operation("valve.open".into()),
        exact_policy_digest: d(20),
        minimum_policy_registry_sequence: 5,
        safety: safety(),
        maximum_envelope_lifetime_s: 5,
    }
}

fn envelope(sequence: u64, now_ms: u64) -> PhysicalEffectEnvelopeV1 {
    let now_s = now_ms / 1_000;
    PhysicalEffectEnvelopeV1 {
        schema_version: PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION,
        command: DeviceCommand {
            schema_version: DEVICE_COMMAND_SCHEMA_VERSION,
            command_id: format!("cmd-{sequence}"),
            actor: PrincipalId("agent:irrigation".into()),
            executor: PrincipalId("gateway:field-a".into()),
            task: None,
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            expected_firmware: d(7),
            sequence,
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

fn verified_transport(
    mut envelope: PhysicalEffectEnvelopeV1,
    now_ms: u64,
) -> symthaea_iot_transport_receipt::VerifiedTransportEnvelope {
    envelope.schema_version = PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION;
    let raw_payload = bincode::serialize(&envelope).unwrap();
    let peer = [0x77; 32];
    let snapshot = TransportTrustSnapshotV1 {
        schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
        sequence: 1,
        issued_at_unix_ms: now_ms.saturating_sub(2_000),
        expires_at_unix_ms: now_ms + 60_000,
        previous_snapshot_digest: None,
        keys: vec![TransportAttestorKeyV1 {
            attestor_id: "attestor:guard-test".into(),
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
        attestor_id: "attestor:guard-test".into(),
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
    let raw_receipt = bincode::serialize(&receipt).unwrap();
    verify_xenia_transport_receipt(
        &registry,
        &raw_receipt,
        &raw_payload,
        now_ms,
        &AcceptFixtureSignatures,
    )
    .unwrap()
}

#[test]
fn reservation_survives_restart_and_rejects_same_sequence() {
    let root = temp_root("restart");
    let cfg = config();
    let now = wall_ms();

    let store = DurableAdmissionReservationStore::open(&root, cfg.clone()).unwrap();
    let reservation = store
        .reserve_verified_transport(verified_transport(envelope(1, now), now))
        .unwrap();
    assert_eq!(reservation.head().generation, 1);
    assert_eq!(reservation.checkpoint().highest_reserved_sequence, Some(1));
    let first_head = reservation.head();
    drop(store);

    let reopened = DurableAdmissionReservationStore::open(&root, cfg).unwrap();
    assert_eq!(reopened.current_head().unwrap(), first_head);
    assert_eq!(
        reopened
            .current_checkpoint()
            .unwrap()
            .highest_reserved_sequence,
        Some(1)
    );

    let replay_now = wall_ms();
    assert!(matches!(
        reopened.reserve_verified_transport(verified_transport(envelope(1, replay_now), replay_now)),
        Err(AdmissionReservationError::SequenceAlreadyReserved {
            proposed: 1,
            highest: 1
        })
    ));
    assert_eq!(reopened.current_head().unwrap(), first_head);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn higher_sequence_advances_exact_hash_chain() {
    let root = temp_root("advance");
    let cfg = config();
    let now = wall_ms();
    let store = DurableAdmissionReservationStore::open(&root, cfg).unwrap();
    let first = store
        .reserve_verified_transport(verified_transport(envelope(3, now), now))
        .unwrap();
    let first_digest = first.head().digest;

    let later = wall_ms();
    let second = store
        .reserve_verified_transport(verified_transport(envelope(4, later), later))
        .unwrap();
    assert_eq!(second.head().generation, 2);
    assert_eq!(second.checkpoint().previous_checkpoint_digest, Some(first_digest));
    assert_eq!(second.checkpoint().highest_reserved_sequence, Some(4));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn static_policy_failure_does_not_mutate_durable_head() {
    let root = temp_root("policy");
    let cfg = config();
    let store = DurableAdmissionReservationStore::open(&root, cfg).unwrap();
    let genesis = store.current_head().unwrap();
    let now = wall_ms();
    let mut bad = envelope(1, now);
    bad.command.expected_firmware = d(0x99);

    assert!(matches!(
        store.reserve_verified_transport(verified_transport(bad, now)),
        Err(AdmissionReservationError::ExpectedFirmwareNotAllowed)
    ));
    assert_eq!(store.current_head().unwrap(), genesis);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn device_local_envelope_lifetime_failure_does_not_mutate_durable_head() {
    let root = temp_root("local-lifetime");
    let mut cfg = config();
    cfg.maximum_envelope_lifetime_s = 1;
    let store = DurableAdmissionReservationStore::open(&root, cfg).unwrap();
    let genesis = store.current_head().unwrap();
    let now = wall_ms();

    assert!(matches!(
        store.reserve_verified_transport(verified_transport(envelope(1, now), now)),
        Err(AdmissionReservationError::EnvelopeLifetimeExceedsDevicePolicy)
    ));
    assert_eq!(store.current_head().unwrap(), genesis);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn state_is_owner_only_and_corruption_fails_closed_on_reopen() {
    let root = temp_root("corruption");
    let cfg = config();
    let now = wall_ms();
    let store = DurableAdmissionReservationStore::open(&root, cfg.clone()).unwrap();
    store
        .reserve_verified_transport(verified_transport(envelope(1, now), now))
        .unwrap();
    drop(store);

    let root_mode = fs::metadata(&root).unwrap().permissions().mode() & 0o777;
    let state_path = root.join("admission-reservation.state");
    let state_mode = fs::metadata(&state_path).unwrap().permissions().mode() & 0o777;
    assert_eq!(root_mode, 0o700);
    assert_eq!(state_mode, 0o600);

    fs::write(&state_path, b"corrupted-state").unwrap();
    assert!(DurableAdmissionReservationStore::open(&root, cfg).is_err());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn symlink_root_is_rejected() {
    let real = temp_root("real");
    let link = temp_root("link");
    fs::create_dir_all(&real).unwrap();
    symlink(&real, &link).unwrap();
    assert!(matches!(
        DurableAdmissionReservationStore::open(&link, config()),
        Err(AdmissionReservationError::InvalidRootDirectory)
    ));
    let _ = fs::remove_file(link);
    let _ = fs::remove_dir_all(real);
}
