// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ed25519_dalek::{Signer, SigningKey};
use symthaea_action_checkpoint::{CheckpointHead, GrantAccountCheckpoint};
use symthaea_action_runtime::{ExecutionId, GrantAccount, ReservationId, ReservationState};
use symthaea_authority::{AuthorityEpoch, CapabilityGrant, Digest32, PrincipalId, RiskBudget};
use symthaea_authority_frontier_sqlite::SqliteCheckpointCasStore;
use symthaea_authority_time::{
    AUTHORITY_TIME_SCHEMA_VERSION, AuthorityTimeStatementV1, PendingAuthorityTimeChallenge,
    TimeAuthorityId, TrustedTimeAuthorityV1, TrustedTimePolicyV1, VerifiedAuthorityTime,
    verify_authority_time_v1,
};
use symthaea_checkpoint_witness::{
    CHECKPOINT_WITNESS_SCHEMA_VERSION, CheckpointWitnessChallengeV1, CheckpointWitnessId,
    CheckpointWitnessPolicyV1, CheckpointWitnessStatementV1, PendingCheckpointWitnessChallenge,
    TrustedCheckpointWitnessV1, VerifiedCheckpointHead, verify_checkpoint_witnesses_v1,
};
use symthaea_system_attempt_evidence::SqliteAttemptEvidenceJournal;
use symthaea_system_attempt_recovery_index::SqliteAttemptRecoveryIndex;
use symthaea_witnessed_system_crash_recovery::{
    WitnessedRecoveryError, recover_witnessed_to_quiescent,
};

static NEXT_DB: AtomicU64 = AtomicU64::new(0);

fn temp_path(label: &str) -> PathBuf {
    let id = NEXT_DB.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "symthaea-witnessed-recovery-{label}-{}-{id}.sqlite",
        std::process::id()
    ))
}

fn cleanup(path: &Path) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(format!("{}-wal", path.display()));
    let _ = fs::remove_file(format!("{}-shm", path.display()));
}

fn grant() -> CapabilityGrant {
    let mut grant = CapabilityGrant::new(
        "witnessed-recovery-test",
        PrincipalId("issuer".into()),
        PrincipalId("actor".into()),
        AuthorityEpoch(7),
    );
    grant.plan_digest = Some(Digest32([21; 32]));
    grant.world_digest = Some(Digest32([22; 32]));
    grant.max_uses = 1;
    grant.risk_budget = RiskBudget {
        mutation_units: 1,
        ..RiskBudget::default()
    };
    grant
}

fn reserved_frontier(
    path: &Path,
    grant: &CapabilityGrant,
    reservation_id: ReservationId,
) -> (SqliteCheckpointCasStore, CheckpointHead) {
    use symthaea_authority_frontier::CheckpointCasStore;

    let mut store = SqliteCheckpointCasStore::open(path).unwrap();
    let mut account = GrantAccount::new(grant);
    let genesis = GrantAccountCheckpoint::first(grant, account.snapshot()).unwrap();
    let genesis_head = store.compare_and_swap(None, &genesis).unwrap();
    account
        .reserve_execution(
            reservation_id,
            ExecutionId("exec-1".into()),
            RiskBudget {
                mutation_units: 1,
                ..RiskBudget::default()
            },
        )
        .unwrap();
    let reserved = GrantAccountCheckpoint::successor(&genesis, grant, account.snapshot()).unwrap();
    let reserved_head = store
        .compare_and_swap(Some(genesis_head), &reserved)
        .unwrap();
    (store, reserved_head)
}

fn trusted_time(grant_digest: Digest32, witnessed: u64) -> VerifiedAuthorityTime {
    let key_a = SigningKey::from_bytes(&[61; 32]);
    let key_b = SigningKey::from_bytes(&[62; 32]);
    let policy = TrustedTimePolicyV1 {
        schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
        policy_id: [63; 16],
        authorities: vec![
            TrustedTimeAuthorityV1 {
                authority_id: TimeAuthorityId([1; 16]),
                verifying_key: key_a.verifying_key().to_bytes(),
                organization_binding: [71; 32],
                service_binding: [81; 32],
            },
            TrustedTimeAuthorityV1 {
                authority_id: TimeAuthorityId([2; 16]),
                verifying_key: key_b.verifying_key().to_bytes(),
                organization_binding: [72; 32],
                service_binding: [82; 32],
            },
        ],
        threshold: 2,
        minimum_organizations: 2,
        maximum_uncertainty_s: 1,
        maximum_challenge_age_ns: 5_000_000_000,
        maximum_post_verification_age_ns: 5_000_000_000,
    };
    let pending = PendingAuthorityTimeChallenge::new(&policy, grant_digest.0).unwrap();
    let challenge = pending.wire();
    let sign = |authority_id: TimeAuthorityId, key: &SigningKey| {
        let mut statement = AuthorityTimeStatementV1 {
            schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
            authority_id,
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

fn checkpoint_policy() -> (CheckpointWitnessPolicyV1, Vec<SigningKey>) {
    let key_a = SigningKey::from_bytes(&[11; 32]);
    let key_b = SigningKey::from_bytes(&[12; 32]);
    (
        CheckpointWitnessPolicyV1 {
            schema_version: CHECKPOINT_WITNESS_SCHEMA_VERSION,
            policy_id: [13; 16],
            witnesses: vec![
                TrustedCheckpointWitnessV1 {
                    witness_id: CheckpointWitnessId([1; 16]),
                    verifying_key: key_a.verifying_key().to_bytes(),
                    organization_binding: [31; 32],
                    service_binding: [41; 32],
                },
                TrustedCheckpointWitnessV1 {
                    witness_id: CheckpointWitnessId([2; 16]),
                    verifying_key: key_b.verifying_key().to_bytes(),
                    organization_binding: [32; 32],
                    service_binding: [42; 32],
                },
            ],
            threshold: 2,
            minimum_organizations: 2,
            maximum_challenge_age_s: 10,
        },
        vec![key_a, key_b],
    )
}

fn signed_statement(
    challenge: CheckpointWitnessChallengeV1,
    key: &SigningKey,
    witness_id: CheckpointWitnessId,
    head: CheckpointHead,
    generation: u64,
) -> CheckpointWitnessStatementV1 {
    let mut statement = CheckpointWitnessStatementV1 {
        schema_version: CHECKPOINT_WITNESS_SCHEMA_VERSION,
        witness_id,
        challenge_nonce: challenge.nonce,
        grant_digest: challenge.grant_digest,
        witness_policy_digest: challenge.witness_policy_digest,
        time_policy_digest: challenge.time_policy_digest,
        checkpoint_sequence: head.sequence,
        checkpoint_digest: head.digest,
        witness_generation: generation,
        signature: Vec::new(),
    };
    statement.signature = key
        .sign(&statement.canonical_message().unwrap())
        .to_bytes()
        .to_vec();
    statement
}

fn witnessed_head(
    grant: &CapabilityGrant,
    head: CheckpointHead,
    time: &VerifiedAuthorityTime,
) -> VerifiedCheckpointHead {
    let (policy, keys) = checkpoint_policy();
    let pending = PendingCheckpointWitnessChallenge::new(&policy, grant.digest(), time).unwrap();
    let challenge = pending.wire();
    verify_checkpoint_witnesses_v1(
        &policy,
        pending,
        time,
        &[
            signed_statement(challenge, &keys[0], CheckpointWitnessId([1; 16]), head, 9),
            signed_statement(challenge, &keys[1], CheckpointWitnessId([2; 16]), head, 14),
        ],
    )
    .unwrap()
}

#[test]
fn fresh_external_head_gates_then_preserves_monotone_recovery() {
    let frontier_path = temp_path("frontier");
    let attempts_path = temp_path("attempts");
    let grant = grant();
    let reservation = ReservationId("reservation-a".into());
    let (mut frontier, reserved_head) =
        reserved_frontier(&frontier_path, &grant, reservation.clone());
    {
        let _journal = SqliteAttemptEvidenceJournal::open(&attempts_path).unwrap();
    }
    let attempts = SqliteAttemptRecoveryIndex::open_read_only(&attempts_path).unwrap();
    let time = trusted_time(grant.digest(), 1_000);
    let witness = witnessed_head(&grant, reserved_head, &time);

    let recovered = recover_witnessed_to_quiescent(&grant, witness, &mut frontier, &attempts)
        .unwrap();
    assert_eq!(recovered.witness_count, 2);
    assert_eq!(recovered.recovery.original_head, reserved_head);
    assert!(recovered.recovery.external_anchor_update_required);
    assert_eq!(recovered.recovery.normalized_reservations, vec![reservation.clone()]);

    let (checkpoint, final_head) = frontier.load_frontier().unwrap().unwrap();
    assert_eq!(final_head, recovered.recovery.final_head);
    assert_eq!(
        checkpoint.snapshot.reservations[&reservation].state,
        ReservationState::OutcomeUnknown
    );
    cleanup(&frontier_path);
    cleanup(&attempts_path);
}

#[test]
fn fresh_witness_for_old_head_cannot_bless_newer_local_frontier() {
    let frontier_path = temp_path("wrong-head");
    let attempts_path = temp_path("wrong-head-attempts");
    let grant = grant();
    let reservation = ReservationId("reservation-b".into());
    let (mut frontier, reserved_head) =
        reserved_frontier(&frontier_path, &grant, reservation);
    {
        let _journal = SqliteAttemptEvidenceJournal::open(&attempts_path).unwrap();
    }
    let attempts = SqliteAttemptRecoveryIndex::open_read_only(&attempts_path).unwrap();
    let time = trusted_time(grant.digest(), 1_000);
    let old_head = CheckpointHead {
        sequence: reserved_head.sequence.saturating_sub(1),
        digest: Digest32([99; 32]),
    };
    let witness = witnessed_head(&grant, old_head, &time);

    assert!(matches!(
        recover_witnessed_to_quiescent(&grant, witness, &mut frontier, &attempts),
        Err(WitnessedRecoveryError::Recovery(_))
    ));
    assert_eq!(frontier.load_frontier().unwrap().unwrap().1, reserved_head);
    cleanup(&frontier_path);
    cleanup(&attempts_path);
}
