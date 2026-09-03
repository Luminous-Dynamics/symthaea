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
    AGENT_CAPABILITY_ATTESTATION_SCHEMA, ED25519_SIGNATURE_ALGORITHM, ExecutorWorkloadV1,
    TranscriptSignatureSuiteV1, XENIA_LEDGER_CHECKPOINT_SCHEMA, XeniaAgentAuthorizationV1,
    XeniaAgentCapabilityAttestationV1, XeniaAuthorityError, XeniaCheckpointAnchorV1,
    XeniaFreshnessPolicyV1, XeniaLedgerCheckpointV1, XeniaSessionExpectationV1,
    XeniaSignatureEnvelopeV1, verify_xenia_capability_v1,
};

fn grant_workload_head() -> (CapabilityGrant, ExecutorWorkloadV1, CheckpointHead) {
    let executor = PrincipalId("spiffe://luminous.local/symthaea/system-broker".into());
    let mut grant = CapabilityGrant::new(
        "interval-test-grant",
        PrincipalId("xenia://operator/alice".into()),
        PrincipalId("symthaea://agent/system-recovery".into()),
        AuthorityEpoch(7),
    );
    grant.audience = Some(executor.clone());
    grant.expires_at_unix_s = Some(200);
    grant.max_uses = 1;
    let workload = ExecutorWorkloadV1 {
        executor,
        artifact_digest: Digest32([11; 32]),
        configuration_digest: Digest32([12; 32]),
        host_identity_digest: Digest32([13; 32]),
    };
    let head = CheckpointHead {
        sequence: 9,
        digest: Digest32([14; 32]),
    };
    (grant, workload, head)
}

fn verified_time(grant: &CapabilityGrant, witnessed_unix_s: u64, uncertainty_s: u64) -> VerifiedAuthorityTime {
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
        maximum_uncertainty_s: 10,
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
            uncertainty_s,
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

#[allow(clippy::too_many_arguments)]
fn signed_xenia_fixture(
    grant: &CapabilityGrant,
    workload: &ExecutorWorkloadV1,
    head: CheckpointHead,
    issued_at_unix_s: u64,
    expires_at_unix_s: u64,
    checkpoint_timestamp_unix_s: u64,
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
        issued_at_unix_s,
        expires_at_unix_s,
        nonce: [24; 16],
        ledger_entry_count: 12,
        ledger_head_hash: [25; 32],
        prior_checkpoint: Some(XeniaCheckpointAnchorV1 {
            sequence: head.sequence,
            digest: head.digest.0,
        }),
    };
    let attestation = XeniaAgentCapabilityAttestationV1 {
        schema: AGENT_CAPABILITY_ATTESTATION_SCHEMA.into(),
        ledger_public_key_fingerprint: *blake3::hash(&public_key).as_bytes(),
        signature: XeniaSignatureEnvelopeV1 {
            algorithm: ED25519_SIGNATURE_ALGORITHM.into(),
            signature: signing_key
                .sign(&authorization.canonical_message().unwrap())
                .to_bytes()
                .to_vec(),
        },
        authorization,
    };
    let mut checkpoint = XeniaLedgerCheckpointV1 {
        schema: XENIA_LEDGER_CHECKPOINT_SCHEMA.into(),
        entry_count: 12,
        head_hash: [25; 32],
        ledger_public_key: public_key,
        timestamp_unix_secs: checkpoint_timestamp_unix_s,
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

#[test]
fn earliest_plausible_time_blocks_authorization_that_may_not_be_valid_yet() {
    let (grant, workload, head) = grant_workload_head();
    // Signed time says roughly 125 +/- 2. The earliest plausible instant is 123,
    // so an authorization issued at 124 is not proven live yet even though the
    // latest plausible instant is later than issuance.
    let time = verified_time(&grant, 125, 2);
    let (attestation, checkpoint, public_key, session) =
        signed_xenia_fixture(&grant, &workload, head, 124, 180, 124);

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
        Err(XeniaAuthorityError::AuthorizationNotYetValid)
    ));
}

#[test]
fn earliest_plausible_time_blocks_future_checkpoint_even_if_upper_bound_would_allow_it() {
    let (grant, workload, head) = grant_workload_head();
    // The earliest plausible time is 123. With five seconds allowed skew, a
    // checkpoint at 130 is still too far in the future (130 > 128). Using the
    // later upper bound here would incorrectly weaken this check.
    let time = verified_time(&grant, 125, 2);
    let (attestation, checkpoint, public_key, session) =
        signed_xenia_fixture(&grant, &workload, head, 100, 180, 130);

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
        Err(XeniaAuthorityError::LedgerCheckpointFromFuture)
    ));
}
