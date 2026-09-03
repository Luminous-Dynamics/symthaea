// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Production crash-recovery profile requiring a fresh external checkpoint witness.
//!
//! `symthaea-system-crash-recovery` contains the deterministic recovery state
//! machine and retains an explicit raw-head seam for low-level composition and
//! testing. This crate is the authenticated production boundary: callers cannot
//! provide a bare `CheckpointHead`; they must provide an opaque
//! `VerifiedCheckpointHead` created by the fresh challenge protocol in
//! `symthaea-checkpoint-witness`.
//!
//! A successful result is still `QuiescentNoAuthority`. Witnessing authenticates
//! the anti-rollback input; it does not grant permission to dispatch, retry, or
//! reuse the recovered capability.

#![deny(unsafe_code)]

use symthaea_authority::CapabilityGrant;
use symthaea_authority_frontier_sqlite::SqliteCheckpointCasStore;
use symthaea_checkpoint_witness::{CheckpointWitnessError, VerifiedCheckpointHead};
use symthaea_system_attempt_recovery_index::SqliteAttemptRecoveryIndex;
use symthaea_system_crash_recovery::{
    recover_to_quiescent, CrashRecoveryError, QuiescentRecoveryState, TrustedRecoveryAnchor,
};
use thiserror::Error;

#[derive(Debug)]
pub struct WitnessedQuiescentRecoveryState {
    pub witness_policy_digest: [u8; 32],
    pub time_policy_digest: [u8; 32],
    pub witness_count: u16,
    pub witness_organization_count: u16,
    pub recovery: QuiescentRecoveryState,
}

/// Recover one exact grant lineage using only a freshly verified external head.
///
/// The witness is consumed by value so it cannot accidentally be reused by this
/// profile after the local frontier changes. Durable anti-replay is still the
/// external witness domain's monotonic retained state plus the local SQLite CAS.
pub fn recover_witnessed_to_quiescent(
    grant: &CapabilityGrant,
    verified_head: VerifiedCheckpointHead,
    frontier: &mut SqliteCheckpointCasStore,
    attempt_index: &SqliteAttemptRecoveryIndex,
) -> Result<WitnessedQuiescentRecoveryState, WitnessedRecoveryError> {
    let grant_digest = grant.digest();
    verified_head.require_grant(grant_digest)?;

    let witness_policy_digest = verified_head.witness_policy_digest();
    let time_policy_digest = verified_head.time_policy_digest();
    let witness_count = verified_head.witness_count();
    let witness_organization_count = verified_head.organization_count();
    let checkpoint_head = verified_head.head();

    // Consume the witness before entering the low-level recovery state machine.
    // It carries evidence, not authority, and there is no reason to retain a
    // reusable proof after binding its exact head.
    drop(verified_head);

    let recovery = recover_to_quiescent(
        grant,
        TrustedRecoveryAnchor { checkpoint_head },
        frontier,
        attempt_index,
    )?;

    Ok(WitnessedQuiescentRecoveryState {
        witness_policy_digest,
        time_policy_digest,
        witness_count,
        witness_organization_count,
        recovery,
    })
}

#[derive(Debug, Error)]
pub enum WitnessedRecoveryError {
    #[error("checkpoint witness verification/binding failed: {0}")]
    Witness(#[from] CheckpointWitnessError),
    #[error("low-level crash recovery failed: {0}")]
    Recovery(#[from] CrashRecoveryError),
}
