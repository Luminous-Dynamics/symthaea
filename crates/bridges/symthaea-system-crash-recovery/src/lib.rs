// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed crash recovery for the durable system-agency vertical slice.
//!
//! Recovery is deliberately not an execution mode. This crate authenticates an
//! exact persisted Agency Kernel frontier against an externally trusted head,
//! validates locally discovered attempt evidence, conservatively converts every
//! stranded `Reserved` execution to `OutcomeUnknown`, and stops in a
//! `QuiescentNoAuthority` state.
//!
//! It has no service backend and no capability-verification API. A caller must
//! publish the returned frontier head into its independent trust domain and then
//! obtain fresh authority (for example a fresh Xenia proof) before constructing
//! any executor.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use symthaea_action_checkpoint::{CheckpointError, CheckpointHead, GrantAccountCheckpoint};
use symthaea_action_runtime::{
    GrantAccount, ReservationId, ReservationState, RuntimeAccountingError,
};
use symthaea_authority::{CapabilityGrant, Digest32};
use symthaea_authority_frontier::CheckpointCasStore;
use symthaea_authority_frontier_sqlite::{SqliteCheckpointCasStore, SqliteFrontierError};
use symthaea_system_attempt_evidence::AttemptEvidenceState;
use symthaea_system_attempt_recovery_index::{
    reservation_id_digest_v1, DiscoveredAttempt, LocalEvidenceDisposition, RecoveryIndexError,
    SqliteAttemptRecoveryIndex,
};
use thiserror::Error;

/// Externally authenticated anti-rollback input.
///
/// This crate does not decide how the head becomes trustworthy. Xenia, a TPM,
/// remote witness, signed supervisor state, or another independently reviewed
/// mechanism may supply it. The important property is that it is not learned
/// from the same SQLite frontier being authenticated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrustedRecoveryAnchor {
    pub checkpoint_head: CheckpointHead,
}

/// Marker proving only that recovery reached a fail-closed quiescent state.
///
/// It is intentionally not a capability and cannot be converted into one by
/// this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuiescentNoAuthority;

/// Minimal attempt reference retained in the recovery result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryAttemptReference {
    pub attempt_key: Digest32,
    pub reservation_digest: Digest32,
    pub last_state: AttemptEvidenceState,
    pub local_disposition: LocalEvidenceDisposition,
}

/// Successful crash recovery result.
///
/// `external_anchor_update_required` is true whenever recovery advanced the
/// local CAS frontier. No fresh authority should be accepted until the caller's
/// independent trust domain durably acknowledges `final_head`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuiescentRecoveryState {
    pub authority: QuiescentNoAuthority,
    pub original_head: CheckpointHead,
    pub final_head: CheckpointHead,
    pub normalized_reservations: Vec<ReservationId>,
    pub incomplete_attempts: Vec<RecoveryAttemptReference>,
    pub external_anchor_update_required: bool,
}

/// Recover the current durable grant state to a conservative, non-executing
/// quiescent point.
///
/// The operation is intentionally ordered so every evidence/identity check
/// happens before a local CAS mutation. The only mutation this function can
/// perform is `Reserved -> OutcomeUnknown`, which preserves the full use/risk
/// charge and therefore cannot increase remaining authority.
pub fn recover_to_quiescent(
    grant: &CapabilityGrant,
    trusted_anchor: TrustedRecoveryAnchor,
    frontier: &mut SqliteCheckpointCasStore,
    attempt_index: &SqliteAttemptRecoveryIndex,
) -> Result<QuiescentRecoveryState, CrashRecoveryError> {
    let (checkpoint, current_head) = frontier
        .load_frontier()?
        .ok_or(CrashRecoveryError::MissingFrontier)?;
    if current_head != trusted_anchor.checkpoint_head {
        return Err(CrashRecoveryError::TrustedHeadMismatch {
            trusted: trusted_anchor.checkpoint_head,
            local: current_head,
        });
    }

    let plan_digest = grant
        .plan_digest
        .ok_or(CrashRecoveryError::GrantMissingPlanBinding)?;
    let world_digest = grant
        .world_digest
        .ok_or(CrashRecoveryError::GrantMissingWorldBinding)?;
    let mut account = checkpoint.verify_payload(grant)?;

    // Evidence is advisory for consequence reconstruction but never an
    // authority source. Validate all incomplete chains and their exact binding
    // to this grant before changing the durable account.
    let incomplete = attempt_index.scan_incomplete()?;
    validate_attempt_bindings(
        grant,
        plan_digest,
        world_digest,
        current_head,
        &account,
        &incomplete,
    )?;

    let mut normalized = Vec::new();
    let snapshot = account.snapshot();
    for (reservation_id, reservation) in &snapshot.reservations {
        if reservation.state == ReservationState::Reserved {
            account.mark_outcome_unknown(reservation_id)?;
            normalized.push(reservation_id.clone());
        }
    }

    let final_head = if normalized.is_empty() {
        current_head
    } else {
        let successor = GrantAccountCheckpoint::successor(&checkpoint, grant, account.snapshot())?;
        frontier.compare_and_swap(Some(current_head), &successor)?
    };

    // Independent post-CAS readback. `SqliteCheckpointCasStore` already performs
    // its own readback; repeating it here makes the coordinator's output depend
    // only on bytes it can freshly reconstruct after the transition.
    let (durable_checkpoint, durable_head) = frontier
        .load_frontier()?
        .ok_or(CrashRecoveryError::MissingFrontierAfterRecovery)?;
    if durable_head != final_head {
        return Err(CrashRecoveryError::FinalReadbackMismatch {
            expected: final_head,
            actual: durable_head,
        });
    }
    durable_checkpoint.verify_payload(grant)?;

    Ok(QuiescentRecoveryState {
        authority: QuiescentNoAuthority,
        original_head: current_head,
        final_head,
        normalized_reservations: normalized,
        incomplete_attempts: incomplete
            .into_iter()
            .map(|attempt| RecoveryAttemptReference {
                attempt_key: attempt.attempt_key,
                reservation_digest: attempt.context.reservation_digest,
                last_state: attempt.last_state,
                local_disposition: attempt.disposition,
            })
            .collect(),
        external_anchor_update_required: final_head != current_head,
    })
}

fn validate_attempt_bindings(
    grant: &CapabilityGrant,
    plan_digest: Digest32,
    world_digest: Digest32,
    current_head: CheckpointHead,
    account: &GrantAccount,
    attempts: &[DiscoveredAttempt],
) -> Result<(), CrashRecoveryError> {
    let snapshot = account.snapshot();
    let mut reservations_by_digest: BTreeMap<Digest32, (&ReservationId, ReservationState)> =
        BTreeMap::new();
    for (reservation_id, reservation) in &snapshot.reservations {
        let digest = reservation_id_digest_v1(reservation_id);
        if reservations_by_digest
            .insert(digest, (reservation_id, reservation.state))
            .is_some()
        {
            return Err(CrashRecoveryError::ReservationDigestCollision);
        }
    }

    let mut incomplete_reservations = BTreeSet::new();
    for attempt in attempts {
        if attempt.context.grant_digest != grant.digest() {
            return Err(CrashRecoveryError::AttemptGrantMismatch {
                attempt_key: attempt.attempt_key,
            });
        }
        if attempt.context.plan_digest != plan_digest {
            return Err(CrashRecoveryError::AttemptPlanMismatch {
                attempt_key: attempt.attempt_key,
            });
        }
        if attempt.context.before_world_digest != world_digest {
            return Err(CrashRecoveryError::AttemptWorldMismatch {
                attempt_key: attempt.attempt_key,
            });
        }
        if attempt.reservation_checkpoint_head.sequence > current_head.sequence
            || attempt.latest_checkpoint_head.sequence > current_head.sequence
        {
            return Err(CrashRecoveryError::AttemptReferencesFutureCheckpoint {
                attempt_key: attempt.attempt_key,
            });
        }

        let Some((_reservation_id, reservation_state)) =
            reservations_by_digest.get(&attempt.context.reservation_digest)
        else {
            return Err(CrashRecoveryError::AttemptReservationMissing {
                attempt_key: attempt.attempt_key,
            });
        };
        if !incomplete_reservations.insert(attempt.context.reservation_digest) {
            return Err(CrashRecoveryError::MultipleIncompleteAttemptsForReservation);
        }
        validate_crash_cut(*reservation_state, attempt)?;
    }
    Ok(())
}

/// Reject only combinations that contradict the current #305/#326 write
/// ordering. Ordinary crash cuts are intentionally accepted.
fn validate_crash_cut(
    reservation_state: ReservationState,
    attempt: &DiscoveredAttempt,
) -> Result<(), CrashRecoveryError> {
    let contradictory = match reservation_state {
        ReservationState::Reserved => false,
        ReservationState::OutcomeUnknown => {
            attempt.last_state == AttemptEvidenceState::ProvenNotDispatched
        }
        ReservationState::Committed => {
            attempt.last_state == AttemptEvidenceState::ProvenNotDispatched
        }
        ReservationState::Released => {
            attempt.last_state != AttemptEvidenceState::ProvenNotDispatched
        }
    };
    if contradictory {
        Err(CrashRecoveryError::ContradictoryCrashCut {
            reservation_state,
            evidence_state: attempt.last_state,
            attempt_key: attempt.attempt_key,
        })
    } else {
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum CrashRecoveryError {
    #[error("SQLite authority frontier failed: {0}")]
    Frontier(#[from] SqliteFrontierError),
    #[error("grant checkpoint validation failed: {0}")]
    Checkpoint(#[from] CheckpointError),
    #[error("runtime accounting recovery failed: {0}")]
    Runtime(#[from] RuntimeAccountingError),
    #[error("attempt recovery index failed: {0}")]
    AttemptIndex(#[from] RecoveryIndexError),
    #[error("durable Agency Kernel frontier is missing")]
    MissingFrontier,
    #[error("durable Agency Kernel frontier disappeared after recovery")]
    MissingFrontierAfterRecovery,
    #[error("trusted recovery head {trusted:?} does not equal local durable head {local:?}")]
    TrustedHeadMismatch {
        trusted: CheckpointHead,
        local: CheckpointHead,
    },
    #[error("system recovery grant lacks an exact plan binding")]
    GrantMissingPlanBinding,
    #[error("system recovery grant lacks an exact world binding")]
    GrantMissingWorldBinding,
    #[error("two runtime reservation identifiers collide under the v1 evidence commitment")]
    ReservationDigestCollision,
    #[error("attempt {attempt_key:?} is bound to a different capability grant")]
    AttemptGrantMismatch { attempt_key: Digest32 },
    #[error("attempt {attempt_key:?} is bound to a different plan")]
    AttemptPlanMismatch { attempt_key: Digest32 },
    #[error("attempt {attempt_key:?} is bound to a different observed world")]
    AttemptWorldMismatch { attempt_key: Digest32 },
    #[error("attempt {attempt_key:?} references a checkpoint sequence newer than the trusted frontier")]
    AttemptReferencesFutureCheckpoint { attempt_key: Digest32 },
    #[error("attempt {attempt_key:?} does not match any reservation in the authenticated checkpoint")]
    AttemptReservationMissing { attempt_key: Digest32 },
    #[error("multiple incomplete attempts claim the same runtime reservation")]
    MultipleIncompleteAttemptsForReservation,
    #[error("attempt {attempt_key:?} contradicts durable crash cut: reservation {reservation_state:?}, evidence {evidence_state:?}")]
    ContradictoryCrashCut {
        reservation_state: ReservationState,
        evidence_state: AttemptEvidenceState,
        attempt_key: Digest32,
    },
    #[error("post-recovery frontier readback mismatch: expected {expected:?}, actual {actual:?}")]
    FinalReadbackMismatch {
        expected: CheckpointHead,
        actual: CheckpointHead,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};
    use symthaea_action_runtime::{ExecutionId, GrantAccount};
    use symthaea_authority::{AuthorityEpoch, PrincipalId, RiskBudget};
    use symthaea_system_attempt_evidence::{
        AttemptEvidenceContext, AttemptEvidenceJournal, AttemptEvidenceRecord,
        SqliteAttemptEvidenceJournal, ATTEMPT_EVIDENCE_SCHEMA_VERSION,
    };

    fn temp_path(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "symthaea-crash-recovery-{label}-{}-{nonce}.sqlite",
            std::process::id()
        ))
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
        let base = path.to_string_lossy();
        let _ = fs::remove_file(format!("{base}-wal"));
        let _ = fs::remove_file(format!("{base}-shm"));
    }

    fn grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "crash-recovery-test",
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

    fn durable_reserved_frontier(
        path: &Path,
        grant: &CapabilityGrant,
        reservation_id: ReservationId,
    ) -> (SqliteCheckpointCasStore, CheckpointHead) {
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

    fn empty_attempt_index(path: &Path) -> SqliteAttemptRecoveryIndex {
        {
            let _journal = SqliteAttemptEvidenceJournal::open(path).unwrap();
        }
        SqliteAttemptRecoveryIndex::open_read_only(path).unwrap()
    }

    fn armed_record(
        grant: &CapabilityGrant,
        reservation_id: &ReservationId,
        checkpoint_head: CheckpointHead,
    ) -> AttemptEvidenceRecord {
        let context = AttemptEvidenceContext::new(
            &ExecutionId("exec-1".into()),
            reservation_id,
            grant.digest(),
            grant.plan_digest.unwrap(),
            grant.world_digest.unwrap(),
            Some(Digest32([23; 32])),
        );
        AttemptEvidenceRecord {
            schema_version: ATTEMPT_EVIDENCE_SCHEMA_VERSION,
            context,
            sequence: 0,
            previous_evidence_digest: None,
            checkpoint_head,
            state: AttemptEvidenceState::DispatchArmed,
            diagnostic_digest: None,
            after_world_digest: None,
            recovery_outcome: None,
            verification: None,
        }
    }

    #[test]
    fn missing_attempt_evidence_never_turns_reserved_into_reusable_capacity() {
        let frontier_path = temp_path("frontier-no-evidence");
        let attempt_path = temp_path("attempt-no-evidence");
        let grant = grant();
        let reservation_id = ReservationId("reservation-a".into());
        let (mut frontier, trusted_head) =
            durable_reserved_frontier(&frontier_path, &grant, reservation_id.clone());
        let index = empty_attempt_index(&attempt_path);

        let recovered = recover_to_quiescent(
            &grant,
            TrustedRecoveryAnchor {
                checkpoint_head: trusted_head,
            },
            &mut frontier,
            &index,
        )
        .unwrap();

        assert_eq!(recovered.normalized_reservations, vec![reservation_id.clone()]);
        assert!(recovered.external_anchor_update_required);
        assert_ne!(recovered.final_head, trusted_head);
        let (checkpoint, head) = frontier.load_frontier().unwrap().unwrap();
        assert_eq!(head, recovered.final_head);
        assert_eq!(
            checkpoint.snapshot.reservations[&reservation_id].state,
            ReservationState::OutcomeUnknown
        );
        cleanup(&frontier_path);
        cleanup(&attempt_path);
    }

    #[test]
    fn discovered_dispatch_armed_is_bound_then_normalized_without_redispatch() {
        let frontier_path = temp_path("frontier-armed");
        let attempt_path = temp_path("attempt-armed");
        let grant = grant();
        let reservation_id = ReservationId("reservation-b".into());
        let (mut frontier, trusted_head) =
            durable_reserved_frontier(&frontier_path, &grant, reservation_id.clone());
        let record = armed_record(&grant, &reservation_id, trusted_head);
        {
            let mut journal = SqliteAttemptEvidenceJournal::open(&attempt_path).unwrap();
            journal.append(&record).unwrap();
        }
        let index = SqliteAttemptRecoveryIndex::open_read_only(&attempt_path).unwrap();

        let recovered = recover_to_quiescent(
            &grant,
            TrustedRecoveryAnchor {
                checkpoint_head: trusted_head,
            },
            &mut frontier,
            &index,
        )
        .unwrap();
        assert_eq!(recovered.incomplete_attempts.len(), 1);
        assert_eq!(
            recovered.incomplete_attempts[0].last_state,
            AttemptEvidenceState::DispatchArmed
        );
        assert_eq!(recovered.normalized_reservations, vec![reservation_id]);
        cleanup(&frontier_path);
        cleanup(&attempt_path);
    }

    #[test]
    fn wrong_external_head_fails_before_mutating_local_frontier() {
        let frontier_path = temp_path("frontier-wrong-head");
        let attempt_path = temp_path("attempt-wrong-head");
        let grant = grant();
        let reservation_id = ReservationId("reservation-c".into());
        let (mut frontier, actual_head) =
            durable_reserved_frontier(&frontier_path, &grant, reservation_id);
        let index = empty_attempt_index(&attempt_path);
        let wrong = CheckpointHead {
            sequence: actual_head.sequence,
            digest: Digest32([0xEE; 32]),
        };

        assert!(matches!(
            recover_to_quiescent(
                &grant,
                TrustedRecoveryAnchor {
                    checkpoint_head: wrong,
                },
                &mut frontier,
                &index,
            )
            .unwrap_err(),
            CrashRecoveryError::TrustedHeadMismatch { .. }
        ));
        assert_eq!(frontier.load_frontier().unwrap().unwrap().1, actual_head);
        cleanup(&frontier_path);
        cleanup(&attempt_path);
    }

    #[test]
    fn mismatched_attempt_plan_contains_before_frontier_change() {
        let frontier_path = temp_path("frontier-plan-mismatch");
        let attempt_path = temp_path("attempt-plan-mismatch");
        let grant = grant();
        let reservation_id = ReservationId("reservation-d".into());
        let (mut frontier, trusted_head) =
            durable_reserved_frontier(&frontier_path, &grant, reservation_id.clone());
        let mut record = armed_record(&grant, &reservation_id, trusted_head);
        record.context.plan_digest = Digest32([99; 32]);
        {
            let mut journal = SqliteAttemptEvidenceJournal::open(&attempt_path).unwrap();
            journal.append(&record).unwrap();
        }
        let index = SqliteAttemptRecoveryIndex::open_read_only(&attempt_path).unwrap();

        assert!(matches!(
            recover_to_quiescent(
                &grant,
                TrustedRecoveryAnchor {
                    checkpoint_head: trusted_head,
                },
                &mut frontier,
                &index,
            )
            .unwrap_err(),
            CrashRecoveryError::AttemptPlanMismatch { .. }
        ));
        assert_eq!(frontier.load_frontier().unwrap().unwrap().1, trusted_head);
        cleanup(&frontier_path);
        cleanup(&attempt_path);
    }

    #[test]
    fn second_recovery_requires_the_new_external_anchor_and_is_idempotent() {
        let frontier_path = temp_path("frontier-idempotent");
        let attempt_path = temp_path("attempt-idempotent");
        let grant = grant();
        let reservation_id = ReservationId("reservation-e".into());
        let (mut frontier, original_head) =
            durable_reserved_frontier(&frontier_path, &grant, reservation_id);
        let index = empty_attempt_index(&attempt_path);

        let first = recover_to_quiescent(
            &grant,
            TrustedRecoveryAnchor {
                checkpoint_head: original_head,
            },
            &mut frontier,
            &index,
        )
        .unwrap();
        assert!(first.external_anchor_update_required);

        // Reusing the pre-recovery trusted anchor cannot bless the locally
        // advanced frontier after a crash or lost external-head publication.
        assert!(matches!(
            recover_to_quiescent(
                &grant,
                TrustedRecoveryAnchor {
                    checkpoint_head: original_head,
                },
                &mut frontier,
                &index,
            )
            .unwrap_err(),
            CrashRecoveryError::TrustedHeadMismatch { .. }
        ));

        // Once the independent trust domain acknowledges the exact new head,
        // recovery is idempotent and performs no further mutation.
        let second = recover_to_quiescent(
            &grant,
            TrustedRecoveryAnchor {
                checkpoint_head: first.final_head,
            },
            &mut frontier,
            &index,
        )
        .unwrap();
        assert_eq!(second.original_head, first.final_head);
        assert_eq!(second.final_head, first.final_head);
        assert!(second.normalized_reservations.is_empty());
        assert!(!second.external_anchor_update_required);
        cleanup(&frontier_path);
        cleanup(&attempt_path);
    }
}
