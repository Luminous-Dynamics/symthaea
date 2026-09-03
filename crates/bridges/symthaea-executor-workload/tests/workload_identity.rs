// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use ed25519_dalek::{Signer, SigningKey};
use symthaea_authority::{AuthorityEpoch, CapabilityGrant, Digest32, PrincipalId};
use symthaea_authority_time::{
    AUTHORITY_TIME_SCHEMA_VERSION, AuthorityTimeStatementV1, PendingAuthorityTimeChallenge,
    TimeAuthorityId, TrustedTimeAuthorityV1, TrustedTimePolicyV1, VerifiedAuthorityTime,
    verify_authority_time_v1,
};
use symthaea_executor_workload::{
    EXECUTOR_WORKLOAD_SCHEMA_VERSION, ExecutorWorkloadV1, PendingWorkloadChallenge,
    TrustedWorkloadWitnessV1, WorkloadIdentityError, WorkloadMeasurementStatementV1,
    WorkloadWitnessId, WorkloadWitnessPolicyV1, measure_linux_process_instance,
    verify_executor_workload_v1,
};

fn grant() -> CapabilityGrant {
    let mut grant = CapabilityGrant::new(
        "measured-workload-grant",
        PrincipalId("xenia://operator/alice".into()),
        PrincipalId("symthaea://agent/recovery".into()),
        AuthorityEpoch(7),
    );
    grant.audience = Some(PrincipalId(
        "spiffe://luminous.local/symthaea/system-broker".into(),
    ));
    grant
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

fn witness_policy() -> (WorkloadWitnessPolicyV1, Vec<SigningKey>) {
    let key_a = SigningKey::from_bytes(&[61; 32]);
    let key_b = SigningKey::from_bytes(&[62; 32]);
    (
        WorkloadWitnessPolicyV1 {
            schema_version: EXECUTOR_WORKLOAD_SCHEMA_VERSION,
            policy_id: [63; 16],
            witnesses: vec![
                TrustedWorkloadWitnessV1 {
                    witness_id: WorkloadWitnessId([1; 16]),
                    verifying_key: key_a.verifying_key().to_bytes(),
                    organization_binding: [71; 32],
                    service_binding: [81; 32],
                },
                TrustedWorkloadWitnessV1 {
                    witness_id: WorkloadWitnessId([2; 16]),
                    verifying_key: key_b.verifying_key().to_bytes(),
                    organization_binding: [72; 32],
                    service_binding: [82; 32],
                },
            ],
            threshold: 2,
            minimum_organizations: 2,
            maximum_challenge_age_s: 10,
            maximum_post_verification_age_s: 10,
            // CI/test binaries are not expected to execute directly from a Nix store path.
            require_nix_store_executable: false,
        },
        vec![key_a, key_b],
    )
}

fn statement(
    challenge: symthaea_executor_workload::WorkloadChallengeV1,
    key: &SigningKey,
    id: WorkloadWitnessId,
    generation: u64,
    workload: ExecutorWorkloadV1,
    process: symthaea_executor_workload::LinuxProcessInstanceV1,
    in_nix_store: bool,
) -> WorkloadMeasurementStatementV1 {
    let mut statement = WorkloadMeasurementStatementV1 {
        schema_version: EXECUTOR_WORKLOAD_SCHEMA_VERSION,
        witness_id: id,
        challenge_nonce: challenge.nonce,
        grant_digest: challenge.grant_digest,
        executor: challenge.executor,
        workload_policy_digest: challenge.workload_policy_digest,
        time_policy_digest: challenge.time_policy_digest,
        workload,
        process,
        witness_generation: generation,
        executable_in_nix_store: in_nix_store,
        signature: Vec::new(),
    };
    statement.signature = key
        .sign(&statement.canonical_message().unwrap())
        .to_bytes()
        .to_vec();
    statement
}

#[test]
fn fresh_witnesses_bind_exact_current_process_and_artifact() {
    let grant = grant();
    let time = verified_time(&grant, 1_000);
    let direct = measure_linux_process_instance(std::process::id()).unwrap();
    let workload = ExecutorWorkloadV1 {
        executor: grant.audience.clone().unwrap(),
        artifact_digest: direct.artifact_digest,
        configuration_digest: Digest32([90; 32]),
        host_identity_digest: direct.host_identity_digest,
    };
    let (policy, keys) = witness_policy();
    let pending = PendingWorkloadChallenge::new(&policy, &grant, &time).unwrap();
    let challenge = pending.wire();
    let verified = verify_executor_workload_v1(
        &policy,
        &grant,
        pending,
        &time,
        &[
            statement(
                challenge.clone(),
                &keys[0],
                WorkloadWitnessId([1; 16]),
                1,
                workload.clone(),
                direct.process,
                direct.executable_in_nix_store,
            ),
            statement(
                challenge,
                &keys[1],
                WorkloadWitnessId([2; 16]),
                2,
                workload.clone(),
                direct.process,
                direct.executable_in_nix_store,
            ),
        ],
    )
    .unwrap();

    assert_eq!(verified.workload_digest().unwrap(), workload.digest().unwrap());
    verified.require_current_process().unwrap();
}

#[test]
fn witness_disagreement_on_process_instance_fails_closed() {
    let grant = grant();
    let time = verified_time(&grant, 1_000);
    let direct = measure_linux_process_instance(std::process::id()).unwrap();
    let workload = ExecutorWorkloadV1 {
        executor: grant.audience.clone().unwrap(),
        artifact_digest: direct.artifact_digest,
        configuration_digest: Digest32([90; 32]),
        host_identity_digest: direct.host_identity_digest,
    };
    let (policy, keys) = witness_policy();
    let pending = PendingWorkloadChallenge::new(&policy, &grant, &time).unwrap();
    let challenge = pending.wire();
    let mut other_process = direct.process;
    other_process.start_time_ticks += 1;

    assert!(matches!(
        verify_executor_workload_v1(
            &policy,
            &grant,
            pending,
            &time,
            &[
                statement(
                    challenge.clone(),
                    &keys[0],
                    WorkloadWitnessId([1; 16]),
                    1,
                    workload.clone(),
                    direct.process,
                    direct.executable_in_nix_store,
                ),
                statement(
                    challenge,
                    &keys[1],
                    WorkloadWitnessId([2; 16]),
                    2,
                    workload,
                    other_process,
                    direct.executable_in_nix_store,
                ),
            ],
        ),
        Err(WorkloadIdentityError::WitnessDisagreement)
    ));
}

#[test]
fn proof_for_different_process_instance_cannot_pass_point_of_use_check() {
    let grant = grant();
    let time = verified_time(&grant, 1_000);
    let direct = measure_linux_process_instance(std::process::id()).unwrap();
    let workload = ExecutorWorkloadV1 {
        executor: grant.audience.clone().unwrap(),
        artifact_digest: direct.artifact_digest,
        configuration_digest: Digest32([90; 32]),
        host_identity_digest: direct.host_identity_digest,
    };
    let (policy, keys) = witness_policy();
    let pending = PendingWorkloadChallenge::new(&policy, &grant, &time).unwrap();
    let challenge = pending.wire();
    let mut claimed = direct.process;
    claimed.start_time_ticks += 1;
    let verified = verify_executor_workload_v1(
        &policy,
        &grant,
        pending,
        &time,
        &[
            statement(
                challenge.clone(),
                &keys[0],
                WorkloadWitnessId([1; 16]),
                1,
                workload.clone(),
                claimed,
                direct.executable_in_nix_store,
            ),
            statement(
                challenge,
                &keys[1],
                WorkloadWitnessId([2; 16]),
                2,
                workload,
                claimed,
                direct.executable_in_nix_store,
            ),
        ],
    )
    .unwrap();

    assert!(matches!(
        verified.require_current_process(),
        Err(WorkloadIdentityError::CurrentProcessMismatch)
    ));
}

#[test]
fn old_signed_measurement_cannot_cross_new_challenge_nonce() {
    let grant = grant();
    let time = verified_time(&grant, 1_000);
    let direct = measure_linux_process_instance(std::process::id()).unwrap();
    let workload = ExecutorWorkloadV1 {
        executor: grant.audience.clone().unwrap(),
        artifact_digest: direct.artifact_digest,
        configuration_digest: Digest32([90; 32]),
        host_identity_digest: direct.host_identity_digest,
    };
    let (policy, keys) = witness_policy();
    let old_pending = PendingWorkloadChallenge::new(&policy, &grant, &time).unwrap();
    let old = old_pending.wire();
    let old_statements = vec![
        statement(
            old.clone(),
            &keys[0],
            WorkloadWitnessId([1; 16]),
            1,
            workload.clone(),
            direct.process,
            direct.executable_in_nix_store,
        ),
        statement(
            old,
            &keys[1],
            WorkloadWitnessId([2; 16]),
            2,
            workload,
            direct.process,
            direct.executable_in_nix_store,
        ),
    ];
    let fresh_pending = PendingWorkloadChallenge::new(&policy, &grant, &time).unwrap();

    assert!(matches!(
        verify_executor_workload_v1(&policy, &grant, fresh_pending, &time, &old_statements),
        Err(WorkloadIdentityError::InvalidStatement)
    ));
}
