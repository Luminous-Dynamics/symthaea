use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use symthaea_action_checkpoint::CheckpointHead;
use symthaea_authority::{Digest32, Operation, PrincipalId, ResourceRef};
use symthaea_iot_actuation_guard_admission_challenge::AdmissionRealityChallengeV1;
use symthaea_iot_actuation_guard_admission_reservation::DurableAdmissionReservationStore;
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

fn temp_root() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "symthaea-admission-challenge-e2e-{}-{nanos}",
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
            command_id: "cmd-e2e-1".into(),
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
            attestor_id: "attestor:challenge-e2e".into(),
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
        attestor_id: "attestor:challenge-e2e".into(),
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
fn challenge_commitments_come_from_real_persisted_reservation() {
    let root = temp_root();
    let cfg = config();
    let now = wall_ms();
    let store = DurableAdmissionReservationStore::open(&root, cfg).unwrap();
    let reservation = store
        .reserve_verified_transport(verified_transport(now))
        .unwrap();
    let effect_deadline = reservation.envelope().send_not_after_unix_s * 1_000;

    let challenge =
        AdmissionRealityChallengeV1::issue_from_persisted_reservation(&reservation).unwrap();

    assert_eq!(challenge.reservation_head(), reservation.head());
    assert_eq!(challenge.envelope_digest(), reservation.envelope_digest());
    assert_eq!(
        challenge.transport_receipt_digest(),
        reservation.transport_receipt_digest()
    );
    assert_eq!(
        challenge.transport_trust_head(),
        reservation.transport_trust_head()
    );
    assert_eq!(challenge.config_digest(), reservation.checkpoint().config_digest);
    assert_eq!(challenge.device(), &reservation.envelope().command.device);
    assert_eq!(
        challenge.reservation_persisted_at_unix_ms(),
        reservation.persisted_at_unix_ms()
    );
    assert!(challenge.issued_at_unix_ms() >= reservation.persisted_at_unix_ms());
    assert_eq!(challenge.effect_deadline_unix_ms(), effect_deadline);
    assert!(challenge.expires_at_unix_ms() <= effect_deadline);
    assert!(challenge.expires_at_unix_ms() > challenge.issued_at_unix_ms());
    assert_ne!(challenge.digest().unwrap(), Digest32([0; 32]));

    std::fs::remove_dir_all(root).unwrap();
}
