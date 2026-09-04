use std::collections::{BTreeMap, BTreeSet};

use ed25519_dalek::{Signer, SigningKey};
use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_actuation_guard_two_phase_protocol::{
    POST_RESERVATION_INTERLOCK_STATEMENT_SCHEMA_VERSION,
    SEMANTIC_RESERVATION_CHALLENGE_SCHEMA_VERSION, PostReservationInterlockReportV1,
    PostReservationInterlockStatementV1, SemanticReservationChallengeV1,
    device_attestation_result_digest,
};
use symthaea_iot_device_protocol::DeviceSemanticHead;
use symthaea_iot_interlock_ed25519::{
    INTERLOCK_ED25519_ALGORITHM, Ed25519Rfc8032InterlockVerifier,
    interlock_ed25519_signing_message,
};
use symthaea_iot_interlock_trust::InterlockControllerEvidenceVerifier;
use symthaea_iot_posture::{
    DEVICE_ATTESTATION_RESULT_SCHEMA_VERSION, DeviceAttestationResultBodyV1,
    DeviceAttestationResultV1,
};
use symthaea_iot_transport_receipt::TransportTrustHead;

fn d(byte: u8) -> Digest32 {
    Digest32([byte; 32])
}

#[test]
fn controller_can_sign_statement_before_signature_commitment_exists() {
    let challenge = SemanticReservationChallengeV1 {
        schema_version: SEMANTIC_RESERVATION_CHALLENGE_SCHEMA_VERSION,
        nonce: [0xA5; 32],
        admission_request_digest: d(1),
        envelope_digest: d(2),
        transport_receipt_digest: d(3),
        device: ResourceRef("iot:valve:72".into()),
        transport_trust_head: TransportTrustHead {
            sequence: 4,
            digest: d(4),
        },
        semantic_head: DeviceSemanticHead {
            generation: 8,
            digest: d(5),
        },
        persisted_at_unix_ms: 10_000,
        expires_at_unix_ms: 15_000,
    };
    let attestation = DeviceAttestationResultV1 {
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
            last_accepted_sequence: Some(7),
            observations: BTreeMap::from([("pressure_x100".into(), 20_000)]),
        },
        signature: vec![0x55; 64],
    };
    let statement = PostReservationInterlockStatementV1 {
        schema_version: POST_RESERVATION_INTERLOCK_STATEMENT_SCHEMA_VERSION,
        challenge_digest: challenge.digest().unwrap(),
        device_attestation_result_digest: device_attestation_result_digest(&attestation).unwrap(),
        controller_id: "safety-plc:field-a".into(),
        device: challenge.device.clone(),
        envelope_digest: challenge.envelope_digest,
        semantic_head: challenge.semantic_head,
        transport_trust_head: challenge.transport_trust_head,
        asserted_interlocks: BTreeSet::from(["pressure-safe".into()]),
        checked_at_unix_ms: 12_000,
        expires_at_unix_ms: 13_000,
    };

    let statement_digest = statement.digest().unwrap();
    let signing_key = SigningKey::from_bytes(&[0x31; 32]);
    let signature = signing_key
        .sign(&interlock_ed25519_signing_message(statement_digest))
        .to_bytes();
    let report = PostReservationInterlockReportV1 {
        statement,
        evidence_digest: Digest32(*blake3::hash(&signature).as_bytes()),
    };

    assert_ne!(report.full_digest().unwrap(), Digest32([0; 32]));
    assert!(Ed25519Rfc8032InterlockVerifier.verify_controller_evidence(
        "safety-plc:field-a",
        "plc-key-1",
        INTERLOCK_ED25519_ALGORITHM,
        signing_key.verifying_key().as_bytes(),
        statement_digest,
        &signature,
    ));
}

#[test]
fn changing_the_signed_statement_breaks_the_existing_fixed_profile() {
    let signing_key = SigningKey::from_bytes(&[0x31; 32]);
    let original_digest = d(7);
    let signature = signing_key
        .sign(&interlock_ed25519_signing_message(original_digest))
        .to_bytes();

    assert!(!Ed25519Rfc8032InterlockVerifier.verify_controller_evidence(
        "safety-plc:field-a",
        "plc-key-1",
        INTERLOCK_ED25519_ALGORITHM,
        signing_key.verifying_key().as_bytes(),
        d(8),
        &signature,
    ));
}
