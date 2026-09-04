// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use ed25519_dalek::{Signer, SigningKey};
use symthaea_authority_time::{
    AUTHORITY_TIME_SCHEMA_VERSION, AuthorityTimeStatementV1, PendingAuthorityTimeChallenge,
    TimeAuthorityId, TrustedTimeAuthorityV1, TrustedTimePolicyV1, VerifiedAuthorityTime,
    verify_authority_time_v1,
};
use symthaea_xenia_authority::{
    ED25519_SIGNATURE_ALGORITHM, XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION,
    XeniaSignatureEnvelopeV1, XeniaSignedWitnessFrontierAnchorV1,
    XeniaSignedWitnessFrontierObservationV1, XeniaWitnessFrontierAnchorSummaryV1,
    XeniaWitnessFrontierAnchorTargetV1, XeniaWitnessFrontierExpectationV1,
    XeniaWitnessFrontierFreshnessPolicyV1, XeniaWitnessFrontierVerificationError,
    derive_xenia_witness_frontier_source_id, verify_xenia_witness_frontier_v1,
    witness_frontier_statement_digest, xenia_witness_frontier_time_subject_digest_v1,
};

const POLICY: [u8; 32] = [0x33; 32];
const WITNESS: [u8; 16] = [0x44; 16];
const RESERVATION_HEAD: [u8; 32] = [0x55; 32];
const CHALLENGE: [u8; 32] = [0x66; 32];

fn xenia_key() -> SigningKey {
    SigningKey::from_bytes(&[7; 32])
}

fn envelope(signature: [u8; 64]) -> XeniaSignatureEnvelopeV1 {
    XeniaSignatureEnvelopeV1 {
        algorithm: ED25519_SIGNATURE_ALGORITHM.to_string(),
        signature: signature.to_vec(),
    }
}

fn expectation() -> XeniaWitnessFrontierExpectationV1 {
    XeniaWitnessFrontierExpectationV1 {
        trusted_ledger_public_key: xenia_key().verifying_key().to_bytes(),
        source_epoch: 3,
        anchor_policy_digest: POLICY,
        witness_id: WITNESS,
        challenge: CHALLENGE,
    }
}

fn signed_anchor() -> XeniaSignedWitnessFrontierAnchorV1 {
    let key = xenia_key();
    let source_id = derive_xenia_witness_frontier_source_id(
        key.verifying_key().to_bytes(),
        POLICY,
    )
    .unwrap();
    let frontier_statement_digest =
        witness_frontier_statement_digest(WITNESS, 9, RESERVATION_HEAD);
    let mut target = XeniaWitnessFrontierAnchorTargetV1 {
        schema_version: XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION,
        operation_id: [0; 32],
        source_id,
        source_epoch: 3,
        anchor_policy_digest: POLICY,
        witness_id: WITNESS,
        high_watermark: 9,
        reservation_head: RESERVATION_HEAD,
        frontier_statement_digest,
    };
    target.operation_id = target.recompute_operation_id();

    let mut anchor = XeniaSignedWitnessFrontierAnchorV1 {
        schema_version: XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION,
        target,
        anchor_sequence: 1,
        previous_anchor_fingerprint: [0; 32],
        ledger_entry_count: 12,
        ledger_head_hash: [0x77; 32],
        ledger_public_key: key.verifying_key().to_bytes(),
        issued_at_unix_s: 1_000,
        signature: envelope([0; 64]),
    };
    anchor.signature = envelope(key.sign(&anchor.canonical_message().unwrap()).to_bytes());
    anchor
}

fn signed_observation(
    anchor: &XeniaSignedWitnessFrontierAnchorV1,
) -> XeniaSignedWitnessFrontierObservationV1 {
    let key = xenia_key();
    let mut observation = XeniaSignedWitnessFrontierObservationV1 {
        schema_version: XENIA_WITNESS_FRONTIER_ANCHOR_SCHEMA_VERSION,
        source_id: anchor.target.source_id,
        source_epoch: anchor.target.source_epoch,
        anchor_policy_digest: anchor.target.anchor_policy_digest,
        witness_id: anchor.target.witness_id,
        challenge: CHALLENGE,
        observed_at_unix_s: 1_010,
        current: Some(XeniaWitnessFrontierAnchorSummaryV1 {
            anchor_sequence: anchor.anchor_sequence,
            anchor_fingerprint: anchor.fingerprint().unwrap(),
            operation_id: anchor.target.operation_id,
            high_watermark: anchor.target.high_watermark,
            reservation_head: anchor.target.reservation_head,
            frontier_statement_digest: anchor.target.frontier_statement_digest,
        }),
        ledger_entry_count: 13,
        ledger_head_hash: [0x88; 32],
        ledger_public_key: key.verifying_key().to_bytes(),
        signature: envelope([0; 64]),
    };
    observation.signature =
        envelope(key.sign(&observation.canonical_message().unwrap()).to_bytes());
    observation
}

fn time_policy() -> (TrustedTimePolicyV1, SigningKey, SigningKey) {
    let key_a = SigningKey::from_bytes(&[31; 32]);
    let key_b = SigningKey::from_bytes(&[32; 32]);
    (
        TrustedTimePolicyV1 {
            schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
            policy_id: [33; 16],
            authorities: vec![
                TrustedTimeAuthorityV1 {
                    authority_id: TimeAuthorityId([1; 16]),
                    verifying_key: key_a.verifying_key().to_bytes(),
                    organization_binding: [41; 32],
                    service_binding: [51; 32],
                },
                TrustedTimeAuthorityV1 {
                    authority_id: TimeAuthorityId([2; 16]),
                    verifying_key: key_b.verifying_key().to_bytes(),
                    organization_binding: [42; 32],
                    service_binding: [52; 32],
                },
            ],
            threshold: 2,
            minimum_organizations: 2,
            maximum_uncertainty_s: 1,
            maximum_challenge_age_ns: 5_000_000_000,
            maximum_post_verification_age_ns: 5_000_000_000,
        },
        key_a,
        key_b,
    )
}

fn verified_time(subject: [u8; 32], witnessed: u64) -> VerifiedAuthorityTime {
    let (policy, key_a, key_b) = time_policy();
    let pending = PendingAuthorityTimeChallenge::new(&policy, subject).unwrap();
    let challenge = pending.wire();
    let sign = |id: TimeAuthorityId, key: &SigningKey| {
        let mut statement = AuthorityTimeStatementV1 {
            schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
            authority_id: id,
            policy_digest: challenge.policy_digest,
            subject_digest: challenge.subject_digest,
            challenge_nonce: challenge.nonce,
            witnessed_unix_s: witnessed,
            uncertainty_s: 1,
            signature: Vec::new(),
        };
        statement.signature = key
            .sign(&statement.canonical_message().unwrap())
            .to_bytes()
            .to_vec();
        statement
    };
    verify_authority_time_v1(
        &policy,
        pending,
        &[
            sign(TimeAuthorityId([1; 16]), &key_a),
            sign(TimeAuthorityId([2; 16]), &key_b),
        ],
    )
    .unwrap()
}

#[test]
fn public_verifier_requires_subject_bound_verified_time() {
    let anchor = signed_anchor();
    let observation = signed_observation(&anchor);
    let expected = expectation();
    let subject = xenia_witness_frontier_time_subject_digest_v1(&anchor, expected).unwrap();
    let time = verified_time(subject, 1_010);

    let verified = verify_xenia_witness_frontier_v1(
        &anchor,
        &observation,
        expected,
        &time,
        XeniaWitnessFrontierFreshnessPolicyV1::strict(30, 2),
    )
    .unwrap();
    assert_eq!(verified.witness_id(), WITNESS);
    assert_eq!(verified.high_watermark(), 9);
}

#[test]
fn valid_time_for_another_subject_cannot_establish_currentness() {
    let anchor = signed_anchor();
    let observation = signed_observation(&anchor);
    let wrong_time = verified_time([0x99; 32], 1_010);

    assert!(matches!(
        verify_xenia_witness_frontier_v1(
            &anchor,
            &observation,
            expectation(),
            &wrong_time,
            XeniaWitnessFrontierFreshnessPolicyV1::strict(30, 2),
        ),
        Err(XeniaWitnessFrontierVerificationError::AuthorityTimeRejected)
    ));
}
