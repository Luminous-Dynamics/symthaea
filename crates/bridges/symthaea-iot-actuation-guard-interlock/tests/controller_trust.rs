// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signer as Ed25519Signer, SigningKey};
use fips204::{
    ml_dsa_65,
    traits::{KeyGen, SerDes, Signer as MlDsaSigner},
};
use symthaea_action_checkpoint::CheckpointHead;
use symthaea_authority::{Digest32, Operation, PrincipalId, ResourceRef, TaskId};
use symthaea_iot_actuation_guard::GuardIngressState;
use symthaea_iot_actuation_guard_interlock::{GuardInterlockError, GuardInterlockState};
use symthaea_iot_actuation_guard_protocol::ActuationGuardRequestV1;
use symthaea_iot_authority::{DEVICE_COMMAND_SCHEMA_VERSION, DeviceCommand};
use symthaea_iot_device_protocol::{
    PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION, DeviceSemanticHead, PhysicalEffectEnvelopeV1,
};
use symthaea_iot_durable_runtime::DurableIoTHead;
use symthaea_iot_final_gate::{
    FinalActuatorGateError, PHYSICAL_INTERLOCK_POLICY_SCHEMA_VERSION,
    PHYSICAL_INTERLOCK_REPORT_SCHEMA_VERSION, PhysicalInterlockPolicyV1,
    PhysicalInterlockReportV1,
};
use symthaea_iot_interlock_ed25519::{
    INTERLOCK_ED25519_ALGORITHM, interlock_ed25519_signing_message,
};
use symthaea_iot_interlock_trust::{
    INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION, InterlockControllerKeyStatus,
    InterlockControllerKeyV1, InterlockTrustError, InterlockTrustHead,
    InterlockTrustRegistry, InterlockTrustSnapshotV1,
};
use symthaea_iot_policy::ActuationPolicyHead;
use symthaea_iot_posture::VerifierTrustHead;
use symthaea_iot_transport_receipt::{
    TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION, TransportAttestorKeyV1, TransportAttestorStatus,
    TransportTrustRegistry, TransportTrustSnapshotV1,
    XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA, XENIA_HYBRID_SIGNATURE_SUITE,
    XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE, XeniaAuthenticatedPayloadReceiptBodyV1,
    XeniaAuthenticatedPayloadReceiptV1, XeniaReceiptPeerRoleV1,
};

fn d(byte: u8) -> Digest32 {
    Digest32([byte; 32])
}

fn now_ms() -> u64 {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    )
    .unwrap()
}

#[derive(Debug, Clone, Copy)]
struct FixtureOptions {
    report_checked_ago_ms: u64,
    interlock_trust_issued_ago_ms: u64,
    require_extra_policy_interlock: bool,
    controller_status: InterlockControllerKeyStatus,
}

impl Default for FixtureOptions {
    fn default() -> Self {
        Self {
            report_checked_ago_ms: 0,
            interlock_trust_issued_ago_ms: 1_000,
            require_extra_policy_interlock: false,
            controller_status: InterlockControllerKeyStatus::Active,
        }
    }
}

struct Fixture {
    ingress: GuardIngressState,
    interlock: GuardInterlockState,
    request: ActuationGuardRequestV1,
    policy_digest: Digest32,
    interlock_trust_head: InterlockTrustHead,
}

fn envelope(now_unix_ms: u64) -> PhysicalEffectEnvelopeV1 {
    let now_s = now_unix_ms / 1_000;
    PhysicalEffectEnvelopeV1 {
        schema_version: PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION,
        command: DeviceCommand {
            schema_version: DEVICE_COMMAND_SCHEMA_VERSION,
            command_id: "cmd-controller-trust-7".into(),
            actor: PrincipalId("agent:irrigation".into()),
            executor: PrincipalId("gateway:field-a".into()),
            task: Some(TaskId("irrigate:zone-7".into())),
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            expected_firmware: d(7),
            sequence: 7,
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
        send_not_after_unix_s: now_s + 5,
    }
}

fn fixture(options: FixtureOptions) -> Fixture {
    // Generate both expensive key pairs before starting the deliberately short freshness
    // windows so a slow CI host cannot consume the report lifetime during keygen.
    let xenia_ed25519_signing_key = SigningKey::from_bytes(&[0x66; 32]);
    let (ml_dsa_public_key, ml_dsa_private_key) = ml_dsa_65::KG::keygen_from_seed(&[0x77; 32]);
    let controller_signing_key = SigningKey::from_bytes(&[0x31; 32]);

    let now_unix_ms = now_ms();
    let receipt_opened_at = now_unix_ms.saturating_sub(500);
    let envelope = envelope(now_unix_ms);
    let envelope_digest = envelope.digest().unwrap();
    let raw_payload = bincode::serialize(&envelope).unwrap();

    let peer_fingerprint = [0x44; 32];
    let transport_registry = TransportTrustRegistry::genesis(TransportTrustSnapshotV1 {
        schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
        sequence: 1,
        issued_at_unix_ms: now_unix_ms.saturating_sub(10_000),
        expires_at_unix_ms: now_unix_ms + 60_000,
        previous_snapshot_digest: None,
        keys: vec![TransportAttestorKeyV1 {
            attestor_id: "xenia-gateway-a".into(),
            key_id: "transport-key-1".into(),
            ed25519_public_key: xenia_ed25519_signing_key.verifying_key().to_bytes(),
            ml_dsa_public_key: ml_dsa_public_key.into_bytes().to_vec(),
            status: TransportAttestorStatus::Active,
            not_before_unix_ms: now_unix_ms.saturating_sub(10_000),
            not_after_unix_ms: now_unix_ms + 60_000,
            max_receipt_lifetime_ms: 4_000,
            required_peer_role: XeniaReceiptPeerRoleV1::Viewer,
            allowed_peer_fingerprints: BTreeSet::from([peer_fingerprint]),
            require_input_control: true,
        }],
    })
    .unwrap();
    let transport_head = transport_registry.head();

    let receipt_body = XeniaAuthenticatedPayloadReceiptBodyV1 {
        schema: XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA.into(),
        attestor_id: "xenia-gateway-a".into(),
        key_id: "transport-key-1".into(),
        signature_algorithm: XENIA_HYBRID_SIGNATURE_SUITE.into(),
        session_evidence_digest: [0x41; 32],
        peer_role: XeniaReceiptPeerRoleV1::Viewer,
        peer_identity_fingerprint: peer_fingerprint,
        transcript_hash: [0x42; 32],
        session_context_hash: [0x43; 32],
        telemetry_enabled: false,
        input_control_enabled: true,
        payload_type: XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE,
        payload_len: raw_payload.len() as u32,
        payload_digest: *blake3::hash(&raw_payload).as_bytes(),
        sealed_envelope_digest: [0x45; 32],
        opened_at_unix_ms: receipt_opened_at,
        expires_at_unix_ms: now_unix_ms + 3_000,
    };
    let receipt_digest = receipt_body.signing_digest().unwrap();
    let xenia_ed25519_signature = xenia_ed25519_signing_key.sign(&receipt_digest).to_bytes();
    let ml_dsa_signature = ml_dsa_private_key
        .try_sign_with_seed(&[0x88; 32], &receipt_digest, &[])
        .expect("deterministic ML-DSA-65 receipt signature");
    let raw_receipt = bincode::serialize(&XeniaAuthenticatedPayloadReceiptV1 {
        body: receipt_body,
        ed25519_signature: xenia_ed25519_signature,
        ml_dsa_signature,
    })
    .unwrap();

    let report_checked_at = now_unix_ms.saturating_sub(options.report_checked_ago_ms);
    assert!(report_checked_at >= receipt_opened_at);
    let mut report = PhysicalInterlockReportV1 {
        schema_version: PHYSICAL_INTERLOCK_REPORT_SCHEMA_VERSION,
        controller_id: "safety-plc:field-a".into(),
        device: envelope.command.device.clone(),
        envelope_digest,
        device_head: DeviceSemanticHead {
            generation: 9,
            digest: d(0x51),
        },
        transport_trust_head: transport_head,
        asserted_interlocks: BTreeSet::from(["pressure-safe".into()]),
        checked_at_unix_ms: report_checked_at,
        expires_at_unix_ms: report_checked_at + 1_500,
        evidence_digest: d(0xEE),
    };
    let report_digest = report.digest().unwrap();
    let controller_signature = controller_signing_key
        .sign(&interlock_ed25519_signing_message(report_digest))
        .to_bytes();
    report.evidence_digest = Digest32(*blake3::hash(&controller_signature).as_bytes());
    assert_eq!(report.digest().unwrap(), report_digest);

    let request = ActuationGuardRequestV1 {
        schema_version: 1,
        raw_transport_receipt: raw_receipt,
        raw_physical_effect_payload: raw_payload,
        raw_interlock_report: bincode::serialize(&report).unwrap(),
        raw_interlock_evidence: controller_signature.to_vec(),
    };

    let policy = PhysicalInterlockPolicyV1 {
        schema_version: PHYSICAL_INTERLOCK_POLICY_SCHEMA_VERSION,
        device: ResourceRef("iot:valve:72".into()),
        allowed_controllers: BTreeSet::from(["safety-plc:field-a".into()]),
        required_interlocks: if options.require_extra_policy_interlock {
            BTreeSet::from(["pressure-safe".into(), "manual-stop-ready".into()])
        } else {
            BTreeSet::from(["pressure-safe".into()])
        },
        max_report_lifetime_ms: 1_500,
    };
    let policy_digest = policy.digest().unwrap();

    let interlock_trust_registry = InterlockTrustRegistry::genesis(InterlockTrustSnapshotV1 {
        schema_version: INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION,
        sequence: 1,
        issued_at_unix_ms: now_unix_ms.saturating_sub(options.interlock_trust_issued_ago_ms),
        expires_at_unix_ms: now_unix_ms + 60_000,
        previous_snapshot_digest: None,
        keys: vec![InterlockControllerKeyV1 {
            controller_id: "safety-plc:field-a".into(),
            key_id: "plc-key-1".into(),
            algorithm: INTERLOCK_ED25519_ALGORITHM.into(),
            public_key: controller_signing_key.verifying_key().to_bytes().to_vec(),
            status: options.controller_status,
            not_before_unix_ms: now_unix_ms.saturating_sub(10_000),
            not_after_unix_ms: now_unix_ms + 60_000,
        }],
    })
    .unwrap();
    let interlock_trust_head = interlock_trust_registry.head();

    Fixture {
        ingress: GuardIngressState::new(transport_registry, transport_head).unwrap(),
        interlock: GuardInterlockState::new(
            policy,
            policy_digest,
            interlock_trust_registry,
            interlock_trust_head,
        )
        .unwrap(),
        request,
        policy_digest,
        interlock_trust_head,
    }
}

fn verified_ingress(fixture: &Fixture) -> symthaea_iot_actuation_guard::VerifiedGuardIngress {
    let frame = fixture.request.canonical_bytes().unwrap();
    fixture.ingress.verify_frame(&frame).unwrap()
}

#[test]
fn real_xenia_and_real_controller_signatures_pass_current_guard_trust() {
    let fixture = fixture(FixtureOptions::default());
    let verified = fixture
        .interlock
        .verify_ingress(verified_ingress(&fixture))
        .unwrap();

    assert_eq!(verified.policy_digest(), fixture.policy_digest);
    assert_eq!(verified.interlock_trust_head(), fixture.interlock_trust_head);
    assert_eq!(verified.controller_key_id(), "plc-key-1");
    assert_ne!(verified.controller_key_digest(), Digest32([0; 32]));
    assert_ne!(verified.report_digest(), Digest32([0; 32]));
    assert_ne!(verified.evidence_digest(), Digest32([0; 32]));
}

#[test]
fn corrupted_controller_signature_cannot_hide_behind_updated_evidence_digest() {
    let mut fixture = fixture(FixtureOptions::default());
    fixture.request.raw_interlock_evidence[0] ^= 1;
    let mut report: PhysicalInterlockReportV1 =
        bincode::deserialize(&fixture.request.raw_interlock_report).unwrap();
    report.evidence_digest =
        Digest32(*blake3::hash(&fixture.request.raw_interlock_evidence).as_bytes());
    fixture.request.raw_interlock_report = bincode::serialize(&report).unwrap();

    let ingress = verified_ingress(&fixture);
    assert!(matches!(
        fixture.interlock.verify_ingress(ingress),
        Err(GuardInterlockError::InterlockTrust(
            InterlockTrustError::ControllerEvidenceVerificationFailed
        ))
    ));
}

#[test]
fn report_before_current_trust_generation_is_rejected_even_for_same_key() {
    let fixture = fixture(FixtureOptions {
        report_checked_ago_ms: 400,
        interlock_trust_issued_ago_ms: 200,
        ..FixtureOptions::default()
    });

    assert!(matches!(
        fixture.interlock.verify_ingress(verified_ingress(&fixture)),
        Err(GuardInterlockError::ReportPredatesCurrentTrustGeneration)
    ));
}

#[test]
fn guard_owned_policy_must_match_exact_asserted_interlock_set() {
    let fixture = fixture(FixtureOptions {
        require_extra_policy_interlock: true,
        ..FixtureOptions::default()
    });

    assert!(matches!(
        fixture.interlock.verify_ingress(verified_ingress(&fixture)),
        Err(GuardInterlockError::FinalGate(
            FinalActuatorGateError::InterlockSetMismatch
        ))
    ));
}

#[test]
fn revoked_current_controller_key_fails_closed() {
    let fixture = fixture(FixtureOptions {
        controller_status: InterlockControllerKeyStatus::Revoked,
        ..FixtureOptions::default()
    });

    assert!(matches!(
        fixture.interlock.verify_ingress(verified_ingress(&fixture)),
        Err(GuardInterlockError::InterlockTrust(
            InterlockTrustError::NoActiveControllerKey
        ))
    ));
}
