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
use symthaea_iot_actuation_guard::{GuardIngressError, GuardIngressState};
use symthaea_iot_actuation_guard_protocol::ActuationGuardRequestV1;
use symthaea_iot_authority::{DEVICE_COMMAND_SCHEMA_VERSION, DeviceCommand};
use symthaea_iot_device_protocol::{
    PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION, DeviceSemanticHead, PhysicalEffectEnvelopeV1,
};
use symthaea_iot_durable_runtime::DurableIoTHead;
use symthaea_iot_final_gate::{
    PHYSICAL_INTERLOCK_REPORT_SCHEMA_VERSION, PhysicalInterlockReportV1,
};
use symthaea_iot_policy::ActuationPolicyHead;
use symthaea_iot_posture::VerifierTrustHead;
use symthaea_iot_transport_receipt::{
    TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION, TransportAttestorKeyV1, TransportAttestorStatus,
    TransportTrustHead, TransportTrustRegistry, TransportTrustSnapshotV1,
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

fn envelope(now_unix_ms: u64) -> PhysicalEffectEnvelopeV1 {
    let now_s = now_unix_ms / 1_000;
    PhysicalEffectEnvelopeV1 {
        schema_version: PHYSICAL_EFFECT_ENVELOPE_SCHEMA_VERSION,
        command: DeviceCommand {
            schema_version: DEVICE_COMMAND_SCHEMA_VERSION,
            command_id: "cmd-guard-7".into(),
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

struct Fixture {
    ingress: GuardIngressState,
    request: ActuationGuardRequestV1,
    now_unix_ms: u64,
    opened_at_unix_ms: u64,
    envelope_digest: Digest32,
    trust_head: TransportTrustHead,
}

fn fixture() -> Fixture {
    // Generate cryptographic fixture keys before taking the freshness timestamps so a
    // slow CI host cannot consume the short physical-interlock validity window.
    let ed25519_signing_key = SigningKey::from_bytes(&[0x66; 32]);
    let (ml_dsa_public_key, ml_dsa_private_key) = ml_dsa_65::KG::keygen_from_seed(&[0x77; 32]);

    let now_unix_ms = now_ms();
    let opened_at_unix_ms = now_unix_ms.saturating_sub(100);
    let receipt_expires_at_unix_ms = now_unix_ms + 3_000;

    let envelope = envelope(now_unix_ms);
    let envelope_digest = envelope.digest().unwrap();
    let raw_payload = bincode::serialize(&envelope).unwrap();

    let peer_fingerprint = [0x44; 32];
    let trusted_key = TransportAttestorKeyV1 {
        attestor_id: "xenia-gateway-a".into(),
        key_id: "transport-key-1".into(),
        ed25519_public_key: ed25519_signing_key.verifying_key().to_bytes(),
        ml_dsa_public_key: ml_dsa_public_key.into_bytes().to_vec(),
        status: TransportAttestorStatus::Active,
        not_before_unix_ms: now_unix_ms.saturating_sub(10_000),
        not_after_unix_ms: now_unix_ms + 60_000,
        max_receipt_lifetime_ms: 4_000,
        required_peer_role: XeniaReceiptPeerRoleV1::Viewer,
        allowed_peer_fingerprints: BTreeSet::from([peer_fingerprint]),
        require_input_control: true,
    };
    let registry = TransportTrustRegistry::genesis(TransportTrustSnapshotV1 {
        schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
        sequence: 1,
        issued_at_unix_ms: now_unix_ms.saturating_sub(10_000),
        expires_at_unix_ms: now_unix_ms + 60_000,
        previous_snapshot_digest: None,
        keys: vec![trusted_key],
    })
    .unwrap();
    let trust_head = registry.head();

    let body = XeniaAuthenticatedPayloadReceiptBodyV1 {
        schema: XENIA_AUTHENTICATED_PAYLOAD_RECEIPT_SCHEMA.into(),
        attestor_id: "xenia-gateway-a".into(),
        key_id: "transport-key-1".into(),
        signature_algorithm: XENIA_HYBRID_SIGNATURE_SUITE.into(),
        session_evidence_digest: [0x31; 32],
        peer_role: XeniaReceiptPeerRoleV1::Viewer,
        peer_identity_fingerprint: peer_fingerprint,
        transcript_hash: [0x45; 32],
        session_context_hash: [0x46; 32],
        telemetry_enabled: false,
        input_control_enabled: true,
        payload_type: XENIA_PHYSICAL_EFFECT_PAYLOAD_TYPE,
        payload_len: raw_payload.len() as u32,
        payload_digest: *blake3::hash(&raw_payload).as_bytes(),
        sealed_envelope_digest: [0x47; 32],
        opened_at_unix_ms,
        expires_at_unix_ms: receipt_expires_at_unix_ms,
    };
    let receipt_digest = body.signing_digest().unwrap();
    let ed25519_signature = ed25519_signing_key.sign(&receipt_digest).to_bytes();
    let ml_dsa_signature = ml_dsa_private_key
        .try_sign_with_seed(&[0x88; 32], &receipt_digest, &[])
        .expect("deterministic ML-DSA-65 receipt signature");
    let raw_receipt = bincode::serialize(&XeniaAuthenticatedPayloadReceiptV1 {
        body,
        ed25519_signature,
        ml_dsa_signature,
    })
    .unwrap();

    let raw_interlock_evidence = b"controller-evidence-not-yet-authenticated-at-ingress".to_vec();
    let report = PhysicalInterlockReportV1 {
        schema_version: PHYSICAL_INTERLOCK_REPORT_SCHEMA_VERSION,
        controller_id: "safety-plc:field-a".into(),
        device: envelope.command.device.clone(),
        envelope_digest,
        device_head: DeviceSemanticHead {
            generation: 9,
            digest: d(0x51),
        },
        transport_trust_head: trust_head,
        asserted_interlocks: BTreeSet::from(["pressure-safe".into()]),
        checked_at_unix_ms: now_unix_ms,
        expires_at_unix_ms: now_unix_ms + 1_500,
        evidence_digest: Digest32(*blake3::hash(&raw_interlock_evidence).as_bytes()),
    };

    let request = ActuationGuardRequestV1 {
        schema_version: 1,
        raw_transport_receipt: raw_receipt,
        raw_physical_effect_payload: raw_payload,
        raw_interlock_report: bincode::serialize(&report).unwrap(),
        raw_interlock_evidence,
    };

    Fixture {
        ingress: GuardIngressState::new(registry, trust_head).unwrap(),
        request,
        now_unix_ms,
        opened_at_unix_ms,
        envelope_digest,
        trust_head,
    }
}

fn mutate_report(
    request: &mut ActuationGuardRequestV1,
    mutate: impl FnOnce(&mut PhysicalInterlockReportV1),
) {
    let mut report: PhysicalInterlockReportV1 =
        bincode::deserialize(&request.raw_interlock_report).unwrap();
    mutate(&mut report);
    request.raw_interlock_report = bincode::serialize(&report).unwrap();
}

#[test]
fn real_hybrid_receipt_and_exact_cross_bindings_pass_guard_ingress() {
    let fixture = fixture();
    let frame = fixture.request.canonical_bytes().unwrap();
    let verified = fixture.ingress.verify_frame(&frame).unwrap();

    assert_eq!(verified.envelope_digest(), fixture.envelope_digest);
    assert_eq!(verified.transport_trust_head(), fixture.trust_head);
    assert_ne!(verified.transport_receipt_digest(), Digest32([0; 32]));
    assert_ne!(verified.interlock_report_digest(), Digest32([0; 32]));
}

#[test]
fn interlock_envelope_substitution_fails_after_real_xenia_verification() {
    let mut fixture = fixture();
    mutate_report(&mut fixture.request, |report| report.envelope_digest = d(0x91));
    let frame = fixture.request.canonical_bytes().unwrap();
    assert!(matches!(
        fixture.ingress.verify_frame(&frame),
        Err(GuardIngressError::InterlockEnvelopeMismatch)
    ));
}

#[test]
fn interlock_transport_generation_substitution_fails() {
    let mut fixture = fixture();
    let current_head = fixture.trust_head;
    mutate_report(&mut fixture.request, |report| {
        report.transport_trust_head = TransportTrustHead {
            sequence: current_head.sequence + 1,
            digest: d(0x92),
        };
    });
    let frame = fixture.request.canonical_bytes().unwrap();
    assert!(matches!(
        fixture.ingress.verify_frame(&frame),
        Err(GuardIngressError::InterlockTransportTrustMismatch)
    ));
}

#[test]
fn interlock_device_substitution_fails() {
    let mut fixture = fixture();
    mutate_report(&mut fixture.request, |report| {
        report.device = ResourceRef("iot:valve:other".into());
    });
    let frame = fixture.request.canonical_bytes().unwrap();
    assert!(matches!(
        fixture.ingress.verify_frame(&frame),
        Err(GuardIngressError::InterlockDeviceMismatch)
    ));
}

#[test]
fn interlock_observation_cannot_predate_authenticated_transport() {
    let mut fixture = fixture();
    let opened_at = fixture.opened_at_unix_ms;
    mutate_report(&mut fixture.request, |report| {
        report.checked_at_unix_ms = opened_at - 1;
        report.expires_at_unix_ms = opened_at + 999;
    });
    let frame = fixture.request.canonical_bytes().unwrap();
    assert!(matches!(
        fixture.ingress.verify_frame(&frame),
        Err(GuardIngressError::InterlockPredatesAuthenticatedTransport)
    ));
}

#[test]
fn stale_interlock_report_fails_even_while_xenia_receipt_is_fresh() {
    let mut fixture = fixture();
    let opened_at = fixture.opened_at_unix_ms;
    let fixture_now = fixture.now_unix_ms;
    mutate_report(&mut fixture.request, |report| {
        report.checked_at_unix_ms = opened_at + 1;
        report.expires_at_unix_ms = fixture_now;
    });
    let frame = fixture.request.canonical_bytes().unwrap();
    assert!(matches!(
        fixture.ingress.verify_frame(&frame),
        Err(GuardIngressError::InterlockReportNotFresh)
    ));
}

#[test]
fn independently_anchored_transport_head_is_required_at_guard_construction() {
    let ed25519_signing_key = SigningKey::from_bytes(&[0x66; 32]);
    let (ml_dsa_public_key, _) = ml_dsa_65::KG::keygen_from_seed(&[0x77; 32]);
    let now_unix_ms = now_ms();
    let registry = TransportTrustRegistry::genesis(TransportTrustSnapshotV1 {
        schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
        sequence: 1,
        issued_at_unix_ms: now_unix_ms.saturating_sub(1_000),
        expires_at_unix_ms: now_unix_ms + 60_000,
        previous_snapshot_digest: None,
        keys: vec![TransportAttestorKeyV1 {
            attestor_id: "xenia-gateway-a".into(),
            key_id: "transport-key-1".into(),
            ed25519_public_key: ed25519_signing_key.verifying_key().to_bytes(),
            ml_dsa_public_key: ml_dsa_public_key.into_bytes().to_vec(),
            status: TransportAttestorStatus::Active,
            not_before_unix_ms: now_unix_ms.saturating_sub(1_000),
            not_after_unix_ms: now_unix_ms + 60_000,
            max_receipt_lifetime_ms: 4_000,
            required_peer_role: XeniaReceiptPeerRoleV1::Viewer,
            allowed_peer_fingerprints: BTreeSet::from([[0x44; 32]]),
            require_input_control: true,
        }],
    })
    .unwrap();
    let wrong_head = TransportTrustHead {
        sequence: registry.head().sequence,
        digest: d(0xFE),
    };

    assert!(matches!(
        GuardIngressState::new(registry, wrong_head),
        Err(GuardIngressError::AnchoredTransportHeadMismatch)
    ));
}
