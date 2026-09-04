use std::collections::{BTreeMap, BTreeSet};

use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_actuation_guard_two_phase_protocol::{
    ACTUATION_GUARD_ADMISSION_SCHEMA_VERSION, ACTUATION_GUARD_POST_RESERVATION_SCHEMA_VERSION,
    POST_RESERVATION_INTERLOCK_STATEMENT_SCHEMA_VERSION,
    SEMANTIC_RESERVATION_CHALLENGE_SCHEMA_VERSION, ActuationGuardAdmissionRequestV2,
    ActuationGuardPostReservationResponseV2, PostReservationInterlockReportV1,
    PostReservationInterlockStatementV1, SemanticReservationChallengeV1,
    TwoPhaseGuardProtocolError, decode_canonical_guard_admission_v2,
    decode_canonical_post_reservation_response_v2, device_attestation_result_digest,
};
use symthaea_iot_device_protocol::DeviceSemanticHead;
use symthaea_iot_posture::{
    DEVICE_ATTESTATION_RESULT_SCHEMA_VERSION, DeviceAttestationResultBodyV1,
    DeviceAttestationResultV1,
};
use symthaea_iot_transport_receipt::TransportTrustHead;

fn d(byte: u8) -> Digest32 {
    Digest32([byte; 32])
}

fn admission() -> ActuationGuardAdmissionRequestV2 {
    ActuationGuardAdmissionRequestV2 {
        schema_version: ACTUATION_GUARD_ADMISSION_SCHEMA_VERSION,
        raw_transport_receipt: vec![0x11; 128],
        raw_physical_effect_payload: vec![0x22; 256],
    }
}

fn challenge() -> SemanticReservationChallengeV1 {
    SemanticReservationChallengeV1 {
        schema_version: SEMANTIC_RESERVATION_CHALLENGE_SCHEMA_VERSION,
        nonce: [0xA5; 32],
        admission_request_digest: admission().digest().unwrap(),
        envelope_digest: d(2),
        transport_receipt_digest: d(3),
        device: ResourceRef("iot:valve:72".into()),
        transport_trust_head: TransportTrustHead {
            sequence: 5,
            digest: d(4),
        },
        semantic_head: DeviceSemanticHead {
            generation: 7,
            digest: d(5),
        },
        persisted_at_unix_ms: 10_000,
        expires_at_unix_ms: 15_000,
    }
}

fn attestation(challenge: &SemanticReservationChallengeV1) -> DeviceAttestationResultV1 {
    DeviceAttestationResultV1 {
        body: DeviceAttestationResultBodyV1 {
            schema_version: DEVICE_ATTESTATION_RESULT_SCHEMA_VERSION,
            verifier_id: "verifier:fleet-a".into(),
            key_id: "device-key-1".into(),
            algorithm: "ed25519-rfc8032".into(),
            device: challenge.device.clone(),
            challenge_digest: challenge.digest().unwrap(),
            appraised_at_unix_s: 11,
            expires_at_unix_s: 14,
            evidence_digest: d(0x31),
            reference_values_digest: d(0x32),
            appraisal_policy_digest: d(0x33),
            running_firmware: d(0x34),
            last_accepted_sequence: Some(6),
            observations: BTreeMap::from([("pressure_x100".into(), 20_000)]),
        },
        signature: vec![0x44; 64],
    }
}

fn report(
    challenge: &SemanticReservationChallengeV1,
    attestation: &DeviceAttestationResultV1,
    raw_evidence: &[u8],
) -> PostReservationInterlockReportV1 {
    PostReservationInterlockReportV1 {
        statement: PostReservationInterlockStatementV1 {
            schema_version: POST_RESERVATION_INTERLOCK_STATEMENT_SCHEMA_VERSION,
            challenge_digest: challenge.digest().unwrap(),
            device_attestation_result_digest: device_attestation_result_digest(attestation).unwrap(),
            controller_id: "safety-plc:field-a".into(),
            device: challenge.device.clone(),
            envelope_digest: challenge.envelope_digest,
            semantic_head: challenge.semantic_head,
            transport_trust_head: challenge.transport_trust_head,
            asserted_interlocks: BTreeSet::from(["pressure-safe".into()]),
            checked_at_unix_ms: 12_000,
            expires_at_unix_ms: 13_000,
        },
        evidence_digest: Digest32(*blake3::hash(raw_evidence).as_bytes()),
    }
}

fn response(
    challenge: &SemanticReservationChallengeV1,
    raw_evidence: &[u8],
) -> ActuationGuardPostReservationResponseV2 {
    let attestation = attestation(challenge);
    let report = report(challenge, &attestation, raw_evidence);
    ActuationGuardPostReservationResponseV2 {
        schema_version: ACTUATION_GUARD_POST_RESERVATION_SCHEMA_VERSION,
        raw_device_attestation_result: bincode::serialize(&attestation).unwrap(),
        raw_interlock_report: bincode::serialize(&report).unwrap(),
        raw_interlock_evidence: raw_evidence.to_vec(),
    }
}

#[test]
fn phase_one_contains_transport_evidence_only_and_rejects_trailing_bytes() {
    let request = admission();
    let frame = request.canonical_bytes().unwrap();
    let decoded = decode_canonical_guard_admission_v2(&frame).unwrap();
    assert_eq!(decoded.request_digest(), request.digest().unwrap());
    assert_eq!(
        decoded.raw_transport_receipt(),
        request.raw_transport_receipt.as_slice()
    );
    assert_eq!(
        decoded.raw_physical_effect_payload(),
        request.raw_physical_effect_payload.as_slice()
    );

    let mut smuggled = frame;
    smuggled.extend_from_slice(b"premature-interlock-or-caller-policy");
    assert!(matches!(
        decode_canonical_guard_admission_v2(&smuggled),
        Err(TwoPhaseGuardProtocolError::NonCanonicalAdmissionEncoding)
            | Err(TwoPhaseGuardProtocolError::Decoding(_))
    ));
}

#[test]
fn exact_post_reservation_evidence_binds_one_persisted_semantic_head() {
    let challenge = challenge();
    let raw_evidence = b"controller-signature";
    let response = response(&challenge, raw_evidence);
    let frame = response.canonical_bytes().unwrap();
    let decoded = decode_canonical_post_reservation_response_v2(&frame, &challenge).unwrap();

    assert_eq!(decoded.response_digest(), response.digest().unwrap());
    assert_eq!(
        decoded.interlock_report().statement.semantic_head,
        challenge.semantic_head
    );
    assert_eq!(
        decoded.device_attestation_result().body.challenge_digest,
        challenge.digest().unwrap()
    );
    assert_eq!(
        decoded.interlock_report().statement.device_attestation_result_digest,
        device_attestation_result_digest(decoded.device_attestation_result()).unwrap()
    );
    assert_eq!(decoded.raw_interlock_evidence(), raw_evidence);
}

#[test]
fn controller_statement_cannot_be_observed_before_semantic_persistence() {
    let challenge = challenge();
    let raw_evidence = b"controller-signature";
    let mut response = response(&challenge, raw_evidence);
    let mut report: PostReservationInterlockReportV1 =
        bincode::deserialize(&response.raw_interlock_report).unwrap();
    report.statement.checked_at_unix_ms = challenge.persisted_at_unix_ms - 1;
    report.statement.expires_at_unix_ms = challenge.persisted_at_unix_ms + 999;
    response.raw_interlock_report = bincode::serialize(&report).unwrap();
    let frame = response.canonical_bytes().unwrap();

    assert!(matches!(
        decode_canonical_post_reservation_response_v2(&frame, &challenge),
        Err(TwoPhaseGuardProtocolError::InterlockPredatesSemanticPersistence)
    ));
}

#[test]
fn another_semantic_head_cannot_be_substituted_into_phase_two() {
    let challenge = challenge();
    let raw_evidence = b"controller-signature";
    let mut response = response(&challenge, raw_evidence);
    let mut report: PostReservationInterlockReportV1 =
        bincode::deserialize(&response.raw_interlock_report).unwrap();
    report.statement.semantic_head = DeviceSemanticHead {
        generation: challenge.semantic_head.generation + 1,
        digest: d(0x91),
    };
    response.raw_interlock_report = bincode::serialize(&report).unwrap();
    let frame = response.canonical_bytes().unwrap();

    assert!(matches!(
        decode_canonical_post_reservation_response_v2(&frame, &challenge),
        Err(TwoPhaseGuardProtocolError::ChallengeSemanticHeadMismatch)
    ));
}

#[test]
fn device_attestation_and_controller_report_must_share_exact_challenge() {
    let challenge = challenge();
    let raw_evidence = b"controller-signature";
    let mut response = response(&challenge, raw_evidence);
    let mut result: DeviceAttestationResultV1 =
        bincode::deserialize(&response.raw_device_attestation_result).unwrap();
    result.body.challenge_digest = d(0xA1);
    response.raw_device_attestation_result = bincode::serialize(&result).unwrap();
    let frame = response.canonical_bytes().unwrap();

    assert!(matches!(
        decode_canonical_post_reservation_response_v2(&frame, &challenge),
        Err(TwoPhaseGuardProtocolError::ChallengeBindingMismatch)
    ));
}

#[test]
fn controller_statement_must_bind_exact_device_appraisal_object() {
    let challenge = challenge();
    let raw_evidence = b"controller-signature";
    let mut response = response(&challenge, raw_evidence);
    let mut result: DeviceAttestationResultV1 =
        bincode::deserialize(&response.raw_device_attestation_result).unwrap();
    result.body.observations.insert("pressure_x100".into(), 20_001);
    response.raw_device_attestation_result = bincode::serialize(&result).unwrap();
    let frame = response.canonical_bytes().unwrap();

    assert!(matches!(
        decode_canonical_post_reservation_response_v2(&frame, &challenge),
        Err(TwoPhaseGuardProtocolError::InterlockDeviceAttestationDigestMismatch)
    ));
}

#[test]
fn controller_statement_binds_exact_device_attestation_signature_bytes_too() {
    let challenge = challenge();
    let raw_evidence = b"controller-signature";
    let mut response = response(&challenge, raw_evidence);
    let mut result: DeviceAttestationResultV1 =
        bincode::deserialize(&response.raw_device_attestation_result).unwrap();
    result.signature[0] ^= 1;
    response.raw_device_attestation_result = bincode::serialize(&result).unwrap();
    let frame = response.canonical_bytes().unwrap();

    assert!(matches!(
        decode_canonical_post_reservation_response_v2(&frame, &challenge),
        Err(TwoPhaseGuardProtocolError::InterlockDeviceAttestationDigestMismatch)
    ));
}

#[test]
fn raw_controller_evidence_substitution_fails_before_signature_trust() {
    let challenge = challenge();
    let committed = b"controller-signature";
    let mut response = response(&challenge, committed);
    response.raw_interlock_evidence = b"substituted-signature".to_vec();
    let frame = response.canonical_bytes().unwrap();

    assert!(matches!(
        decode_canonical_post_reservation_response_v2(&frame, &challenge),
        Err(TwoPhaseGuardProtocolError::InterlockEvidenceDigestMismatch)
    ));
}

#[test]
fn signed_statement_digest_is_structurally_independent_of_controller_signature_commitment() {
    let challenge = challenge();
    let attestation = attestation(&challenge);
    let statement = report(&challenge, &attestation, b"signature-a").statement;
    let statement_digest = statement.digest().unwrap();
    let report_a = PostReservationInterlockReportV1 {
        statement: statement.clone(),
        evidence_digest: d(0xA2),
    };
    let report_b = PostReservationInterlockReportV1 {
        statement,
        evidence_digest: d(0xA3),
    };

    assert_eq!(report_a.statement.digest().unwrap(), statement_digest);
    assert_eq!(report_b.statement.digest().unwrap(), statement_digest);
    assert_ne!(report_a.full_digest().unwrap(), report_b.full_digest().unwrap());
}
