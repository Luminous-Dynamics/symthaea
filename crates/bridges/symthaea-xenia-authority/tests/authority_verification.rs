// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use ed25519_dalek::{Signer, SigningKey};
use symthaea_action_checkpoint::CheckpointHead;
use symthaea_authority::{AuthorityEpoch, CapabilityGrant, Digest32, PrincipalId};
use symthaea_authority_time::{
    AUTHORITY_TIME_SCHEMA_VERSION, AuthorityTimeStatementV1, PendingAuthorityTimeChallenge,
    TimeAuthorityId, TrustedTimeAuthorityV1, TrustedTimePolicyV1, VerifiedAuthorityTime,
    verify_authority_time_v1,
};
use symthaea_xenia_authority::{
    AGENT_CAPABILITY_ATTESTATION_SCHEMA, AGENT_CAPABILITY_AUTHORIZATION_SCHEMA_VERSION,
    ED25519_SIGNATURE_ALGORITHM, ExecutorWorkloadV1, TranscriptSignatureSuiteV1,
    XENIA_LEDGER_CHECKPOINT_SCHEMA, XeniaAgentAuthorizationV1,
    XeniaAgentCapabilityAttestationV1, XeniaAuthorityError, XeniaCheckpointAnchorV1,
    XeniaFreshnessPolicyV1, XeniaLedgerCheckpointV1, XeniaSessionExpectationV1,
    XeniaSignatureEnvelopeV1, verify_xenia_capability_v1,
};

const VECTOR_PUBLIC_KEY: [u8; 32] = [
    0xed, 0x49, 0x28, 0xc6, 0x28, 0xd1, 0xc2, 0xc6, 0xea, 0xe9, 0x03, 0x38, 0x90, 0x59,
    0x95, 0x61, 0x29, 0x59, 0x27, 0x3a, 0x5c, 0x63, 0xf9, 0x36, 0x36, 0xc1, 0x46, 0x14,
    0xac, 0x87, 0x37, 0xd1,
];
const VECTOR_SIGNATURE: [u8; 64] = [
    0xf3, 0x42, 0x66, 0xc5, 0x84, 0xae, 0xa2, 0x6f, 0x84, 0x94, 0xf5, 0x05, 0xe3, 0xfa,
    0xba, 0xc4, 0x90, 0xce, 0xd1, 0x92, 0xc6, 0x04, 0xb0, 0x4c, 0x97, 0x63, 0xe2, 0xd1,
    0x2d, 0xcb, 0xce, 0xa9, 0xf6, 0x65, 0x24, 0x9f, 0xaa, 0xd3, 0x7d, 0x1e, 0xae, 0xf1,
    0x7b, 0x7b, 0x00, 0x11, 0x8a, 0xc3, 0xd2, 0x3d, 0x47, 0xd1, 0x6c, 0x22, 0x66, 0x36,
    0xdb, 0xc7, 0xd2, 0x0a, 0x05, 0x71, 0x7c, 0x01,
];

fn vector_authorization() -> XeniaAgentAuthorizationV1 {
    XeniaAgentAuthorizationV1 {
        schema_version: AGENT_CAPABILITY_AUTHORIZATION_SCHEMA_VERSION,
        authorization_id: [1; 16],
        session_id: [2; 16],
        session_transcript_hash: [3; 32],
        session_signature_suite: TranscriptSignatureSuiteV1::Ed25519Rfc8032,
        capability_digest: [4; 32],
        executor_workload_digest: [5; 32],
        authority_epoch: 7,
        issued_at_unix_s: 100,
        expires_at_unix_s: 160,
        nonce: [6; 16],
        ledger_entry_count: 12,
        ledger_head_hash: [7; 32],
        prior_checkpoint: Some(XeniaCheckpointAnchorV1 {
            sequence: 9,
            digest: [8; 32],
        }),
    }
}

#[test]
fn reproduces_frozen_xenia_cross_repo_signature_vector() {
    let message = vector_authorization().canonical_message().unwrap();
    assert_eq!(message.len(), 292);
    let key = SigningKey::from_bytes(&[3; 32]);
    assert_eq!(key.verifying_key().to_bytes(), VECTOR_PUBLIC_KEY);
    assert_eq!(key.sign(&message).to_bytes(), VECTOR_SIGNATURE);
}

fn grant_and_workload() -> (CapabilityGrant, ExecutorWorkloadV1, CheckpointHead) {
    let executor = PrincipalId("spiffe://luminous.local/symthaea/system-broker".into());
    let mut grant = CapabilityGrant::new(
        "grant-systemd-1",
        PrincipalId("xenia://operator/alice".into()),
        PrincipalId("symthaea://agent/system-recovery".into()),
        AuthorityEpoch(7),
    );
    grant.audience = Some(executor.clone());
    grant.expires_at_unix_s = Some(160);
    grant.max_uses = 1;

    let workload = ExecutorWorkloadV1 {
        executor,
        artifact_digest: Digest32([11; 32]),
        configuration_digest: Digest32([12; 32]),
        host_identity_digest: Digest32([13; 32]),
    };
    let head = CheckpointHead {
        sequence: 9,
        digest: Digest32([8; 32]),
    };
    (grant, workload, head)
}

fn signed_fixture(
    grant: &CapabilityGrant,
    workload: &ExecutorWorkloadV1,
    head: CheckpointHead,
) -> (
    XeniaAgentCapabilityAttestationV1,
    XeniaLedgerCheckpointV1,
    [u8; 32],
    XeniaSessionExpectationV1,
) {
    let signing_key = SigningKey::from_bytes(&[3; 32]);
    let public_key = signing_key.verifying_key().to_bytes();
    let authorization = XeniaAgentAuthorizationV1 {
        schema_version: 1,
        authorization_id: [21; 16],
        session_id: [22; 16],
        session_transcript_hash: [23; 32],
        session_signature_suite: TranscriptSignatureSuiteV1::Ed25519Rfc8032,
        capability_digest: grant.digest().0,
        executor_workload_digest: workload.digest().unwrap().0,
        authority_epoch: grant.authority_epoch.0,
        issued_at_unix_s: 100,
        expires_at_unix_s: 150,
        nonce: [24; 16],
        ledger_entry_count: 12,
        ledger_head_hash: [25; 32],
        prior_checkpoint: Some(XeniaCheckpointAnchorV1 {
            sequence: head.sequence,
            digest: head.digest.0,
        }),
    };
    let signature = signing_key.sign(&authorization.canonical_message().unwrap());
    let attestation = XeniaAgentCapabilityAttestationV1 {
        schema: AGENT_CAPABILITY_ATTESTATION_SCHEMA.into(),
        authorization,
        ledger_public_key_fingerprint: *blake3::hash(&public_key).as_bytes(),
        signature: XeniaSignatureEnvelopeV1 {
            algorithm: ED25519_SIGNATURE_ALGORITHM.into(),
            signature: signature.to_bytes().to_vec(),
        },
    };

    let mut checkpoint = XeniaLedgerCheckpointV1 {
        schema: XENIA_LEDGER_CHECKPOINT_SCHEMA.into(),
        entry_count: 12,
        head_hash: [25; 32],
        ledger_public_key: public_key,
        timestamp_unix_secs: 120,
        signature: Vec::new(),
    };
    checkpoint.signature = signing_key
        .sign(&checkpoint.signature_message().unwrap())
        .to_bytes()
        .to_vec();

    let session = XeniaSessionExpectationV1 {
        session_id: [22; 16],
        transcript_hash: [23; 32],
        transcript_signature_suite: TranscriptSignatureSuiteV1::Ed25519Rfc8032,
    };
    (attestation, checkpoint, public_key, session)
}

fn verified_time(grant: &CapabilityGrant, witnessed_unix_s: u64) -> VerifiedAuthorityTime {
    let key_a = SigningKey::from_bytes(&[31; 32]);
    let key_b = SigningKey::from_bytes(&[32; 32]);
    let policy = TrustedTimePolicyV1 {
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
    };
    let pending = PendingAuthorityTimeChallenge::new(&policy, grant.digest().0).unwrap();
    let challenge = pending.wire();

    let sign = |authority_id: TimeAuthorityId, key: &SigningKey| {
        let mut statement = AuthorityTimeStatementV1 {
            schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
            authority_id,
            policy_digest: challenge.policy_digest,
            subject_digest: challenge.subject_digest,
            challenge_nonce: challenge.nonce,
            witnessed_unix_s,
            uncertainty_s: 1,
            signature: [0; 64],
        };
        statement.signature = key.sign(&statement.canonical_message().unwrap()).to_bytes();
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
fn exact_grant_workload_checkpoint_and_fresh_frontier_verify() {
    let (grant, workload, head) = grant_and_workload();
    let (attestation, checkpoint, public_key, session) = signed_fixture(&grant, &workload, head);
    let time = verified_time(&grant, 125);
    let verified = verify_xenia_capability_v1(
        &attestation,
        &checkpoint,
        public_key,
        &grant,
        &workload,
        session,
        head,
        &time,
        XeniaFreshnessPolicyV1::strict(30, 5),
    )
    .unwrap();
    assert_eq!(verified.grant_digest(), grant.digest());
    assert_eq!(verified.prior_checkpoint(), head);
}

#[test]
fn valid_old_attestation_fails_after_fresh_xenia_frontier_advances() {
    let (grant, workload, head) = grant_and_workload();
    let (attestation, mut checkpoint, public_key, session) =
        signed_fixture(&grant, &workload, head);
    let signing_key = SigningKey::from_bytes(&[3; 32]);
    checkpoint.entry_count = 13;
    checkpoint.head_hash = [99; 32];
    checkpoint.timestamp_unix_secs = 124;
    checkpoint.signature = signing_key
        .sign(&checkpoint.signature_message().unwrap())
        .to_bytes()
        .to_vec();
    let time = verified_time(&grant, 125);

    assert!(matches!(
        verify_xenia_capability_v1(
            &attestation,
            &checkpoint,
            public_key,
            &grant,
            &workload,
            session,
            head,
            &time,
            XeniaFreshnessPolicyV1::strict(30, 5),
        ),
        Err(XeniaAuthorityError::AuthorizationFrontierStale)
    ));
}

#[test]
fn stale_freshness_checkpoint_is_rejected_even_when_signatures_are_valid() {
    let (grant, workload, head) = grant_and_workload();
    let (attestation, checkpoint, public_key, session) = signed_fixture(&grant, &workload, head);
    let time = verified_time(&grant, 200);
    assert!(matches!(
        verify_xenia_capability_v1(
            &attestation,
            &checkpoint,
            public_key,
            &grant,
            &workload,
            session,
            head,
            &time,
            XeniaFreshnessPolicyV1::strict(30, 5),
        ),
        Err(XeniaAuthorityError::LedgerCheckpointStale)
    ));
}

#[test]
fn workload_or_agent_checkpoint_substitution_fails() {
    let (grant, workload, head) = grant_and_workload();
    let (attestation, checkpoint, public_key, session) = signed_fixture(&grant, &workload, head);
    let time = verified_time(&grant, 125);

    let mut wrong_workload = workload.clone();
    wrong_workload.artifact_digest = Digest32([77; 32]);
    assert!(matches!(
        verify_xenia_capability_v1(
            &attestation,
            &checkpoint,
            public_key,
            &grant,
            &wrong_workload,
            session,
            head,
            &time,
            XeniaFreshnessPolicyV1::strict(30, 5),
        ),
        Err(XeniaAuthorityError::WorkloadDigestMismatch)
    ));

    let wrong_head = CheckpointHead {
        sequence: head.sequence + 1,
        digest: Digest32([88; 32]),
    };
    assert!(matches!(
        verify_xenia_capability_v1(
            &attestation,
            &checkpoint,
            public_key,
            &grant,
            &workload,
            session,
            wrong_head,
            &time,
            XeniaFreshnessPolicyV1::strict(30, 5),
        ),
        Err(XeniaAuthorityError::AgentCheckpointMismatch)
    ));
}

#[test]
fn time_for_another_grant_cannot_validate_this_grant() {
    let (grant, workload, head) = grant_and_workload();
    let (attestation, checkpoint, public_key, session) = signed_fixture(&grant, &workload, head);
    let mut other_grant = grant.clone();
    other_grant.grant_id = "other-grant".into();
    let wrong_time = verified_time(&other_grant, 125);

    assert!(matches!(
        verify_xenia_capability_v1(
            &attestation,
            &checkpoint,
            public_key,
            &grant,
            &workload,
            session,
            head,
            &wrong_time,
            XeniaFreshnessPolicyV1::strict(30, 5),
        ),
        Err(XeniaAuthorityError::AuthorityTime(_))
    ));
}
