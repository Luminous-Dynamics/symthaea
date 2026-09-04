// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use ed25519_dalek::{Signer, SigningKey};
use symthaea_authority::Digest32;
use symthaea_iot_actuation_guard_two_phase_protocol::{
    POST_RESERVATION_INTERLOCK_STATEMENT_SCHEMA_VERSION, PostReservationInterlockReportV1,
    PostReservationInterlockStatementV1,
};
use symthaea_iot_device_protocol::DeviceSemanticHead;

use crate::{
    POST_SEMANTIC_CONTROLLER_RESPONSE_SCHEMA_VERSION, PostSemanticControllerChallengeV1,
    PostSemanticControllerError, PostSemanticControllerResponseV1,
    decode_post_semantic_controller_response,
};

fn controller_frame(
    challenge: &PostSemanticControllerChallengeV1,
    semantic_head: DeviceSemanticHead,
    checked_at_unix_ms: u64,
) -> Vec<u8> {
    let statement = PostReservationInterlockStatementV1 {
        schema_version: POST_RESERVATION_INTERLOCK_STATEMENT_SCHEMA_VERSION,
        challenge_digest: challenge.digest().unwrap(),
        device_attestation_result_digest: challenge.device_attestation_object_digest(),
        controller_id: "controller:valve-72".into(),
        device: challenge.device().clone(),
        envelope_digest: challenge.envelope_digest(),
        semantic_head,
        transport_trust_head: challenge.transport_trust_head(),
        asserted_interlocks: BTreeSet::from([
            "pressure-within-range".into(),
            "manual-stop-clear".into(),
        ]),
        checked_at_unix_ms,
        expires_at_unix_ms: checked_at_unix_ms + 500,
    };

    // Construct the exact controller-signature flow: statement first, signature second,
    // evidence commitment third. The parser does not trust this key; fixed trust is later.
    let signing_key = SigningKey::from_bytes(&[0x71; 32]);
    let signature = signing_key.sign(&statement.digest().unwrap().0).to_bytes();
    let report = PostReservationInterlockReportV1 {
        statement,
        evidence_digest: Digest32(*blake3::hash(&signature).as_bytes()),
    };
    let response = PostSemanticControllerResponseV1 {
        schema_version: POST_SEMANTIC_CONTROLLER_RESPONSE_SCHEMA_VERSION,
        raw_interlock_report: bincode::serialize(&report).unwrap(),
        raw_interlock_evidence: signature.to_vec(),
    };
    response.canonical_bytes().unwrap()
}

#[test]
fn controller_signature_is_constructible_only_over_post_semantic_statement_shape() {
    let challenge = PostSemanticControllerChallengeV1::fixture();
    let frame = controller_frame(
        &challenge,
        challenge.semantic_head(),
        challenge.issued_at_unix_ms() + 100,
    );
    let decoded = decode_post_semantic_controller_response(&frame, &challenge).unwrap();
    assert_eq!(decoded.report().statement.semantic_head, challenge.semantic_head());
    assert_eq!(
        decoded.report().statement.device_attestation_result_digest,
        challenge.device_attestation_object_digest()
    );
    assert_ne!(decoded.response_digest(), Digest32([0; 32]));
}

#[test]
fn another_semantic_head_is_rejected_even_with_self_consistent_controller_evidence() {
    let challenge = PostSemanticControllerChallengeV1::fixture();
    let other_head = DeviceSemanticHead {
        generation: challenge.semantic_head().generation + 1,
        digest: Digest32([0xCC; 32]),
    };
    let frame = controller_frame(
        &challenge,
        other_head,
        challenge.issued_at_unix_ms() + 100,
    );
    assert!(matches!(
        decode_post_semantic_controller_response(&frame, &challenge),
        Err(PostSemanticControllerError::SemanticHeadMismatch)
    ));
}

#[test]
fn controller_observation_must_follow_challenge_issuance_not_only_semantic_persistence() {
    let challenge = PostSemanticControllerChallengeV1::fixture();
    let frame = controller_frame(
        &challenge,
        challenge.semantic_head(),
        challenge.issued_at_unix_ms() - 1,
    );
    assert!(matches!(
        decode_post_semantic_controller_response(&frame, &challenge),
        Err(PostSemanticControllerError::ControllerObservationPredatesChallenge)
    ));
}

#[test]
fn raw_evidence_must_match_the_report_commitment() {
    let challenge = PostSemanticControllerChallengeV1::fixture();
    let frame = controller_frame(
        &challenge,
        challenge.semantic_head(),
        challenge.issued_at_unix_ms() + 100,
    );
    let mut response: PostSemanticControllerResponseV1 = bincode::deserialize(&frame).unwrap();
    response.raw_interlock_evidence[0] ^= 1;
    let mutated = response.canonical_bytes().unwrap();
    assert!(matches!(
        decode_post_semantic_controller_response(&mutated, &challenge),
        Err(PostSemanticControllerError::EvidenceDigestMismatch)
    ));
}

#[test]
fn trailing_data_cannot_smuggle_controller_policy_or_authority() {
    let challenge = PostSemanticControllerChallengeV1::fixture();
    let mut frame = controller_frame(
        &challenge,
        challenge.semantic_head(),
        challenge.issued_at_unix_ms() + 100,
    );
    frame.extend_from_slice(b"caller-owned-controller-policy");
    assert!(decode_post_semantic_controller_response(&frame, &challenge).is_err());
}

#[test]
fn changing_semantic_head_changes_the_privileged_challenge_digest() {
    let a = PostSemanticControllerChallengeV1::fixture();
    let mut b = a.clone();
    b.test_set_semantic_head(DeviceSemanticHead {
        generation: a.semantic_head().generation + 1,
        digest: Digest32([0xDD; 32]),
    });
    assert_ne!(a.digest().unwrap(), b.digest().unwrap());
}
