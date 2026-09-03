// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use ed25519_dalek::{Signer, SigningKey};
use symthaea_action_checkpoint::CheckpointHead;
use symthaea_authority::{
    AuthorityEpoch, CapabilityGrant, Digest32, NegativeAuthorityFact, PrincipalId,
};
use symthaea_authority_state::{
    AUTHORITY_STATE_SCHEMA_VERSION, AuthorityStatePolicyV1, AuthorityStateStatementV1,
    AuthorityStateWitnessId, PendingAuthorityStateChallenge, TrustedAuthorityStateWitnessV1,
    VerifiedAuthorityState, verify_authority_state_v1,
};
use symthaea_authority_time::{
    AUTHORITY_TIME_SCHEMA_VERSION, AuthorityTimeStatementV1, PendingAuthorityTimeChallenge,
    TimeAuthorityId, TrustedTimeAuthorityV1, TrustedTimePolicyV1, VerifiedAuthorityTime,
    verify_authority_time_v1,
};
use symthaea_executor_workload::{
    EXECUTOR_WORKLOAD_SCHEMA_VERSION, ExecutorWorkloadV1, PendingWorkloadChallenge,
    TrustedWorkloadWitnessV1, VerifiedExecutorWorkload, WorkloadMeasurementStatementV1,
    WorkloadWitnessId, WorkloadWitnessPolicyV1, measure_linux_process_instance,
    verify_executor_workload_v1,
};
use symthaea_xenia_authority::{
    AGENT_CAPABILITY_ATTESTATION_SCHEMA, AGENT_CAPABILITY_AUTHORIZATION_SCHEMA_VERSION,
    ED25519_SIGNATURE_ALGORITHM, TranscriptSignatureSuiteV1, XENIA_LEDGER_CHECKPOINT_SCHEMA,
    XeniaAgentAuthorizationV1, XeniaAgentCapabilityAttestationV1, XeniaAuthorityError,
    XeniaCheckpointAnchorV1, XeniaFreshnessPolicyV1, XeniaLedgerCheckpointV1,
    XeniaSessionExpectationV1, XeniaSignatureEnvelopeV1, verify_xenia_capability_v1,
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

fn grant() -> CapabilityGrant {
    let mut grant = CapabilityGrant::new(
        "grant-systemd-1",
        PrincipalId("xenia://operator/alice".into()),
        PrincipalId("symthaea://agent/system-recovery".into()),
        AuthorityEpoch(7),
    );
    grant.audience = Some(PrincipalId(
        "spiffe://luminous.local/symthaea/system-broker".into(),
    ));
    grant.expires_at_unix_s = Some(160);
    grant.max_uses = 1;
    grant
}

fn head() -> CheckpointHead {
    CheckpointHead {
        sequence: 9,
        digest: Digest32([8; 32]),
    }
}

fn verified_time(grant: &CapabilityGrant, witnessed: u64) -> VerifiedAuthorityTime {
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

fn verified_state(
    grant: &CapabilityGrant,
    time: &VerifiedAuthorityTime,
    frontier_sequence: u64,
    frontier_digest: Digest32,
    facts: Vec<NegativeAuthorityFact>,
) -> VerifiedAuthorityState {
    let key_a = SigningKey::from_bytes(&[71; 32]);
    let key_b = SigningKey::from_bytes(&[72; 32]);
    let policy = AuthorityStatePolicyV1 {
        schema_version: AUTHORITY_STATE_SCHEMA_VERSION,
        policy_id: [73; 16],
        witnesses: vec![
            TrustedAuthorityStateWitnessV1 {
                witness_id: AuthorityStateWitnessId([1; 16]),
                verifying_key: key_a.verifying_key().to_bytes(),
                organization_binding: [81; 32],
                service_binding: [91; 32],
            },
            TrustedAuthorityStateWitnessV1 {
                witness_id: AuthorityStateWitnessId([2; 16]),
                verifying_key: key_b.verifying_key().to_bytes(),
                organization_binding: [82; 32],
                service_binding: [92; 32],
            },
        ],
        threshold: 2,
        minimum_organizations: 2,
        maximum_challenge_age_s: 60,
        maximum_post_verification_age_s: 60,
    };
    let pending = PendingAuthorityStateChallenge::new(&policy, grant, time).unwrap();
    let challenge = pending.wire();
    let sign = |id: AuthorityStateWitnessId, key: &SigningKey, generation: u64| {
        let mut statement = AuthorityStateStatementV1 {
            schema_version: AUTHORITY_STATE_SCHEMA_VERSION,
            witness_id: id,
            challenge_nonce: challenge.nonce,
            grant_digest: challenge.grant_digest,
            state_policy_digest: challenge.state_policy_digest,
            time_policy_digest: challenge.time_policy_digest,
            source_frontier_sequence: frontier_sequence,
            source_frontier_digest: frontier_digest,
            state_sequence: frontier_sequence,
            authority_epoch: grant.authority_epoch,
            negative_facts: facts.clone(),
            witness_generation: generation,
            signature: Vec::new(),
        };
        statement.signature = key
            .sign(&statement.canonical_message().unwrap())
            .to_bytes()
            .to_vec();
        statement
    };
    verify_authority_state_v1(
        &policy,
        grant,
        pending,
        time,
        &[
            sign(AuthorityStateWitnessId([1; 16]), &key_a, 1),
            sign(AuthorityStateWitnessId([2; 16]), &key_b, 2),
        ],
    )
    .unwrap()
}

fn verified_workload(
    grant: &CapabilityGrant,
    time: &VerifiedAuthorityTime,
) -> (VerifiedExecutorWorkload, ExecutorWorkloadV1) {
    let direct = measure_linux_process_instance(std::process::id()).unwrap();
    let workload = ExecutorWorkloadV1 {
        executor: grant.audience.clone().unwrap(),
        artifact_digest: direct.artifact_digest,
        configuration_digest: Digest32([11; 32]),
        host_identity_digest: direct.host_identity_digest,
    };
    let key_a = SigningKey::from_bytes(&[101; 32]);
    let key_b = SigningKey::from_bytes(&[102; 32]);
    let policy = WorkloadWitnessPolicyV1 {
        schema_version: EXECUTOR_WORKLOAD_SCHEMA_VERSION,
        policy_id: [103; 16],
        witnesses: vec![
            TrustedWorkloadWitnessV1 {
                witness_id: WorkloadWitnessId([1; 16]),
                verifying_key: key_a.verifying_key().to_bytes(),
                organization_binding: [111; 32],
                service_binding: [121; 32],
            },
            TrustedWorkloadWitnessV1 {
                witness_id: WorkloadWitnessId([2; 16]),
                verifying_key: key_b.verifying_key().to_bytes(),
                organization_binding: [112; 32],
                service_binding: [122; 32],
            },
        ],
        threshold: 2,
        minimum_organizations: 2,
        maximum_challenge_age_s: 10,
        maximum_post_verification_age_s: 10,
        require_nix_store_executable: false,
    };
    let pending = PendingWorkloadChallenge::new(&policy, grant, time).unwrap();
    let challenge = pending.wire();
    let sign = |id: WorkloadWitnessId, key: &SigningKey, generation: u64| {
        let mut statement = WorkloadMeasurementStatementV1 {
            schema_version: EXECUTOR_WORKLOAD_SCHEMA_VERSION,
            witness_id: id,
            challenge_nonce: challenge.nonce,
            grant_digest: challenge.grant_digest,
            executor: challenge.executor.clone(),
            workload_policy_digest: challenge.workload_policy_digest,
            time_policy_digest: challenge.time_policy_digest,
            workload: workload.clone(),
            process: direct.process,
            witness_generation: generation,
            executable_in_nix_store: direct.executable_in_nix_store,
            signature: Vec::new(),
        };
        statement.signature = key
            .sign(&statement.canonical_message().unwrap())
            .to_bytes()
            .to_vec();
        statement
    };
    let verified = verify_executor_workload_v1(
        &policy,
        grant,
        pending,
        time,
        &[
            sign(WorkloadWitnessId([1; 16]), &key_a, 1),
            sign(WorkloadWitnessId([2; 16]), &key_b, 2),
        ],
    )
    .unwrap();
    (verified, workload)
}

fn signed_xenia_fixture(
    grant: &CapabilityGrant,
    workload: &ExecutorWorkloadV1,
    agent_head: CheckpointHead,
) -> (
    XeniaAgentCapabilityAttestationV1,
    XeniaLedgerCheckpointV1,
    [u8; 32],
    XeniaSessionExpectationV1,
) {
    let key = SigningKey::from_bytes(&[3; 32]);
    let public_key = key.verifying_key().to_bytes();
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
            sequence: agent_head.sequence,
            digest: agent_head.digest.0,
        }),
    };
    let attestation = XeniaAgentCapabilityAttestationV1 {
        schema: AGENT_CAPABILITY_ATTESTATION_SCHEMA.into(),
        ledger_public_key_fingerprint: *blake3::hash(&public_key).as_bytes(),
        signature: XeniaSignatureEnvelopeV1 {
            algorithm: ED25519_SIGNATURE_ALGORITHM.into(),
            signature: key
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
        timestamp_unix_secs: 120,
        signature: Vec::new(),
    };
    checkpoint.signature = key
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
fn xenia_verification_consumes_measured_workload_and_authority_state() {
    let grant = grant();
    let time = verified_time(&grant, 125);
    let (workload_proof, raw_workload) = verified_workload(&grant, &time);
    let state = verified_state(&grant, &time, 12, Digest32([25; 32]), Vec::new());
    let (attestation, checkpoint, public_key, session) =
        signed_xenia_fixture(&grant, &raw_workload, head());

    let verified = verify_xenia_capability_v1(
        &attestation,
        &checkpoint,
        public_key,
        &grant,
        workload_proof,
        session,
        head(),
        &time,
        state,
        XeniaFreshnessPolicyV1::strict(30, 5),
    )
    .unwrap();

    assert_eq!(verified.workload_digest(), raw_workload.digest().unwrap());
    verified.executor_workload().require_current_process().unwrap();
}

#[test]
fn fresh_frontier_change_still_invalidates_signature_valid_authorization() {
    let grant = grant();
    let time = verified_time(&grant, 125);
    let (workload_proof, raw_workload) = verified_workload(&grant, &time);
    let state = verified_state(&grant, &time, 13, Digest32([99; 32]), Vec::new());
    let (attestation, mut checkpoint, public_key, session) =
        signed_xenia_fixture(&grant, &raw_workload, head());
    let key = SigningKey::from_bytes(&[3; 32]);
    checkpoint.entry_count = 13;
    checkpoint.head_hash = [99; 32];
    checkpoint.timestamp_unix_secs = 124;
    checkpoint.signature = key
        .sign(&checkpoint.signature_message().unwrap())
        .to_bytes()
        .to_vec();

    assert!(matches!(
        verify_xenia_capability_v1(
            &attestation,
            &checkpoint,
            public_key,
            &grant,
            workload_proof,
            session,
            head(),
            &time,
            state,
            XeniaFreshnessPolicyV1::strict(30, 5),
        ),
        Err(XeniaAuthorityError::AuthorizationFrontierStale)
    ));
}

#[test]
fn authenticated_revocation_is_preserved_inside_xenia_proof() {
    let grant = grant();
    let time = verified_time(&grant, 125);
    let (workload_proof, raw_workload) = verified_workload(&grant, &time);
    let revocation = NegativeAuthorityFact::RevokeGrant {
        grant_digest: grant.digest(),
    };
    let state = verified_state(
        &grant,
        &time,
        12,
        Digest32([25; 32]),
        vec![revocation.clone()],
    );
    let (attestation, checkpoint, public_key, session) =
        signed_xenia_fixture(&grant, &raw_workload, head());
    let verified = verify_xenia_capability_v1(
        &attestation,
        &checkpoint,
        public_key,
        &grant,
        workload_proof,
        session,
        head(),
        &time,
        state,
        XeniaFreshnessPolicyV1::strict(30, 5),
    )
    .unwrap();

    assert_eq!(verified.authority_state().negative_facts(), &[revocation]);
}
