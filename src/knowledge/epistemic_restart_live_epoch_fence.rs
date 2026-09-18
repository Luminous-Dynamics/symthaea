// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only live epistemic epoch fence for restart activation review.
//!
//! EKM-067 can retain a verified support store inside a sealed sandbox, but a
//! later activation attempt must also prove that the currently running EKM state
//! did not change while the candidate was being reviewed. This module establishes
//! that baseline without introducing swap or checkpoint-commit authority.
//!
//! The baseline is a complete typed [`EpistemicRestartCapsuleV2`] that is first
//! verified against the *actual live* ledger/support/revision/schema objects at
//! one atomic capture cycle. Future rechecks use the existing source-live
//! validators rather than comparing capture-cycle-sensitive restart digests.

use crate::knowledge::belief_mutation_firewall::EpistemicSupportStore;
use crate::knowledge::belief_revision_receipt::BeliefRevisionHistory;
use crate::knowledge::belief_revision_schema_history::BeliefRevisionSchemaHistoryV1;
use crate::knowledge::belief_revision_schema_persistence::BeliefRevisionSchemaPersistenceError;
use crate::knowledge::claim_evidence::EpistemicLedger;
use crate::knowledge::epistemic_restart_capsule::EpistemicRestartCapsuleError;
use crate::knowledge::epistemic_restart_capsule_v2::{
    EpistemicRestartCapsuleV2, EpistemicRestartCapsuleV2Error,
};
use crate::knowledge::epistemic_restart_manifest::EpistemicLedgerInventoryV1;
use crate::knowledge::epistemic_restart_split_state_sandbox::{
    SealedSplitStateHydrationSandboxV1, SplitStateHydrationSandboxError,
};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveEpistemicEpochFenceVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LiveEpistemicEpochObservationDigestV1([u8; 32]);

impl LiveEpistemicEpochObservationDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LiveEpistemicEpochFenceDigestV1([u8; 32]);

impl LiveEpistemicEpochFenceDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

#[derive(Debug)]
pub struct LiveEpistemicEpochObservationV1 {
    version: LiveEpistemicEpochFenceVersion,
    observed_at_cycle: u64,
    source_capture_cycle: u64,
    live_v2_digest: [u8; 32],
    live_schema_digest: [u8; 32],
    live_manifest_digest: [u8; 32],
    claim_count: usize,
    evidence_count: usize,
    provenance_count: usize,
    support_state_count: usize,
    mutation_count: usize,
    revision_count: usize,
    source_capsule: EpistemicRestartCapsuleV2,
    live_objects_retained: bool,
    mutation_authority: bool,
    activation_authorized: bool,
    trusted_checkpoint_commit_authorized: bool,
    observation_digest: LiveEpistemicEpochObservationDigestV1,
}

impl LiveEpistemicEpochObservationV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        source_capsule: EpistemicRestartCapsuleV2,
        ledger: &EpistemicLedger,
        inventory: &EpistemicLedgerInventoryV1,
        store: &EpistemicSupportStore,
        history: &BeliefRevisionHistory,
        schema_history: &BeliefRevisionSchemaHistoryV1,
        observed_at_cycle: u64,
    ) -> Result<Self, LiveEpistemicEpochFenceError> {
        source_capsule
            .verify()
            .map_err(LiveEpistemicEpochFenceError::LiveCapsuleRejected)?;
        if source_capsule.captured_at_cycle() != observed_at_cycle {
            return Err(LiveEpistemicEpochFenceError::ObservationCycleMismatch {
                capsule_cycle: source_capsule.captured_at_cycle(),
                observed_at_cycle,
            });
        }

        let base = source_capsule.base_v1();
        base.verify_source_live(ledger, inventory, store, history, observed_at_cycle)
            .map_err(LiveEpistemicEpochFenceError::LiveSourceDiverged)?;
        source_capsule
            .revision_schema_capsule()
            .validate_live(
                schema_history,
                history,
                base.revision_capsule(),
                observed_at_cycle,
            )
            .map_err(LiveEpistemicEpochFenceError::LiveSchemaDiverged)?;

        let lineage = base.manifest().ledger_lineage();
        let mut out = Self {
            version: LiveEpistemicEpochFenceVersion::V1,
            observed_at_cycle,
            source_capture_cycle: source_capsule.captured_at_cycle(),
            live_v2_digest: source_capsule.capsule_digest().as_bytes(),
            live_schema_digest: source_capsule.schema_digest().as_bytes(),
            live_manifest_digest: base.manifest_digest().as_bytes(),
            claim_count: lineage.claim_count,
            evidence_count: lineage.evidence_count,
            provenance_count: lineage.provenance_count,
            support_state_count: base.mutation_capsule().states().len(),
            mutation_count: base.mutation_capsule().mutations().len(),
            revision_count: base.revision_capsule().receipts().len(),
            source_capsule,
            live_objects_retained: false,
            mutation_authority: false,
            activation_authorized: false,
            trusted_checkpoint_commit_authorized: false,
            observation_digest: LiveEpistemicEpochObservationDigestV1([0; 32]),
        };
        out.observation_digest = digest_observation(&out)?;
        Ok(out)
    }

    pub fn version(&self) -> LiveEpistemicEpochFenceVersion {
        self.version
    }
    pub fn observed_at_cycle(&self) -> u64 {
        self.observed_at_cycle
    }
    pub fn source_capture_cycle(&self) -> u64 {
        self.source_capture_cycle
    }
    pub fn live_v2_digest(&self) -> [u8; 32] {
        self.live_v2_digest
    }
    pub fn claim_count(&self) -> usize {
        self.claim_count
    }
    pub fn evidence_count(&self) -> usize {
        self.evidence_count
    }
    pub fn provenance_count(&self) -> usize {
        self.provenance_count
    }
    pub fn support_state_count(&self) -> usize {
        self.support_state_count
    }
    pub fn mutation_count(&self) -> usize {
        self.mutation_count
    }
    pub fn revision_count(&self) -> usize {
        self.revision_count
    }
    pub fn live_objects_retained(&self) -> bool {
        self.live_objects_retained
    }
    pub fn mutation_authority(&self) -> bool {
        self.mutation_authority
    }
    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }
    pub fn trusted_checkpoint_commit_authorized(&self) -> bool {
        self.trusted_checkpoint_commit_authorized
    }
    pub fn observation_digest(&self) -> LiveEpistemicEpochObservationDigestV1 {
        self.observation_digest
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_live_unchanged(
        &self,
        ledger: &EpistemicLedger,
        inventory: &EpistemicLedgerInventoryV1,
        store: &EpistemicSupportStore,
        history: &BeliefRevisionHistory,
        schema_history: &BeliefRevisionSchemaHistoryV1,
        rechecked_at_cycle: u64,
    ) -> Result<LiveEpistemicEpochContinuityReceiptV1, LiveEpistemicEpochFenceError> {
        if rechecked_at_cycle < self.observed_at_cycle {
            return Err(LiveEpistemicEpochFenceError::RecheckPredatesObservation {
                rechecked_at_cycle,
                observed_at_cycle: self.observed_at_cycle,
            });
        }
        if digest_observation(self)? != self.observation_digest {
            return Err(LiveEpistemicEpochFenceError::ObservationDigestMismatch);
        }
        let base = self.source_capsule.base_v1();
        base.verify_source_live(
            ledger,
            inventory,
            store,
            history,
            rechecked_at_cycle,
        )
        .map_err(LiveEpistemicEpochFenceError::LiveSourceDiverged)?;
        self.source_capsule
            .revision_schema_capsule()
            .validate_live(
                schema_history,
                history,
                base.revision_capsule(),
                rechecked_at_cycle,
            )
            .map_err(LiveEpistemicEpochFenceError::LiveSchemaDiverged)?;
        LiveEpistemicEpochContinuityReceiptV1::new(self, rechecked_at_cycle)
    }
}

#[derive(Debug)]
pub struct LiveEpistemicEpochFenceV1 {
    version: LiveEpistemicEpochFenceVersion,
    established_at_cycle: u64,
    sandbox_digest: [u8; 32],
    observation: LiveEpistemicEpochObservationV1,
    live_epoch_fence_established: bool,
    activation_preflight_authorized: bool,
    activation_authorized: bool,
    trusted_checkpoint_commit_authorized: bool,
    fence_digest: LiveEpistemicEpochFenceDigestV1,
}

impl LiveEpistemicEpochFenceV1 {
    pub fn establish(
        sandbox: &SealedSplitStateHydrationSandboxV1,
        observation: LiveEpistemicEpochObservationV1,
        established_at_cycle: u64,
    ) -> Result<Self, LiveEpistemicEpochFenceError> {
        sandbox
            .verify()
            .map_err(LiveEpistemicEpochFenceError::SandboxRejected)?;
        if established_at_cycle != observation.observed_at_cycle {
            return Err(LiveEpistemicEpochFenceError::FenceObservationCycleMismatch {
                established_at_cycle,
                observation_cycle: observation.observed_at_cycle,
            });
        }
        if established_at_cycle < sandbox.sandboxed_at_cycle() {
            return Err(LiveEpistemicEpochFenceError::FencePredatesSandbox {
                established_at_cycle,
                sandboxed_at_cycle: sandbox.sandboxed_at_cycle(),
            });
        }
        if observation.live_objects_retained
            || observation.mutation_authority
            || observation.activation_authorized
            || observation.trusted_checkpoint_commit_authorized
        {
            return Err(LiveEpistemicEpochFenceError::UnexpectedObservationAuthority);
        }

        let mut out = Self {
            version: LiveEpistemicEpochFenceVersion::V1,
            established_at_cycle,
            sandbox_digest: sandbox.sandbox_digest().as_bytes(),
            observation,
            live_epoch_fence_established: true,
            activation_preflight_authorized: false,
            activation_authorized: false,
            trusted_checkpoint_commit_authorized: false,
            fence_digest: LiveEpistemicEpochFenceDigestV1([0; 32]),
        };
        out.fence_digest = digest_fence(&out)?;
        Ok(out)
    }

    pub fn version(&self) -> LiveEpistemicEpochFenceVersion {
        self.version
    }
    pub fn established_at_cycle(&self) -> u64 {
        self.established_at_cycle
    }
    pub fn sandbox_digest(&self) -> [u8; 32] {
        self.sandbox_digest
    }
    pub fn observation_digest(&self) -> LiveEpistemicEpochObservationDigestV1 {
        self.observation.observation_digest()
    }
    pub fn live_epoch_fence_established(&self) -> bool {
        self.live_epoch_fence_established
    }
    pub fn activation_preflight_authorized(&self) -> bool {
        self.activation_preflight_authorized
    }
    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }
    pub fn trusted_checkpoint_commit_authorized(&self) -> bool {
        self.trusted_checkpoint_commit_authorized
    }
    pub fn fence_digest(&self) -> LiveEpistemicEpochFenceDigestV1 {
        self.fence_digest
    }

    #[allow(clippy::too_many_arguments)]
    pub fn recheck_live_unchanged(
        &self,
        sandbox: &SealedSplitStateHydrationSandboxV1,
        ledger: &EpistemicLedger,
        inventory: &EpistemicLedgerInventoryV1,
        store: &EpistemicSupportStore,
        history: &BeliefRevisionHistory,
        schema_history: &BeliefRevisionSchemaHistoryV1,
        rechecked_at_cycle: u64,
    ) -> Result<LiveEpistemicEpochContinuityReceiptV1, LiveEpistemicEpochFenceError> {
        sandbox
            .verify()
            .map_err(LiveEpistemicEpochFenceError::SandboxRejected)?;
        if sandbox.sandbox_digest().as_bytes() != self.sandbox_digest {
            return Err(LiveEpistemicEpochFenceError::SandboxDigestMismatch);
        }
        if !self.live_epoch_fence_established
            || self.activation_preflight_authorized
            || self.activation_authorized
            || self.trusted_checkpoint_commit_authorized
        {
            return Err(LiveEpistemicEpochFenceError::UnexpectedFenceAuthority);
        }
        if digest_fence(self)? != self.fence_digest {
            return Err(LiveEpistemicEpochFenceError::FenceDigestMismatch);
        }
        self.observation.verify_live_unchanged(
            ledger,
            inventory,
            store,
            history,
            schema_history,
            rechecked_at_cycle,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveEpistemicEpochContinuityReceiptV1 {
    version: LiveEpistemicEpochFenceVersion,
    baseline_observed_at_cycle: u64,
    rechecked_at_cycle: u64,
    observation_digest: [u8; 32],
    live_v2_digest: [u8; 32],
    live_state_unchanged: bool,
    activation_preflight_authorized: bool,
    activation_authorized: bool,
    trusted_checkpoint_commit_authorized: bool,
    receipt_digest: LiveEpistemicEpochFenceDigestV1,
}

impl LiveEpistemicEpochContinuityReceiptV1 {
    fn new(
        observation: &LiveEpistemicEpochObservationV1,
        rechecked_at_cycle: u64,
    ) -> Result<Self, LiveEpistemicEpochFenceError> {
        let mut out = Self {
            version: LiveEpistemicEpochFenceVersion::V1,
            baseline_observed_at_cycle: observation.observed_at_cycle,
            rechecked_at_cycle,
            observation_digest: observation.observation_digest.as_bytes(),
            live_v2_digest: observation.live_v2_digest,
            live_state_unchanged: true,
            activation_preflight_authorized: false,
            activation_authorized: false,
            trusted_checkpoint_commit_authorized: false,
            receipt_digest: LiveEpistemicEpochFenceDigestV1([0; 32]),
        };
        out.receipt_digest = digest_continuity_receipt(&out)?;
        Ok(out)
    }

    pub fn version(&self) -> LiveEpistemicEpochFenceVersion {
        self.version
    }
    pub fn baseline_observed_at_cycle(&self) -> u64 {
        self.baseline_observed_at_cycle
    }
    pub fn rechecked_at_cycle(&self) -> u64 {
        self.rechecked_at_cycle
    }
    pub fn live_state_unchanged(&self) -> bool {
        self.live_state_unchanged
    }
    pub fn activation_preflight_authorized(&self) -> bool {
        self.activation_preflight_authorized
    }
    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }
    pub fn trusted_checkpoint_commit_authorized(&self) -> bool {
        self.trusted_checkpoint_commit_authorized
    }
    pub fn receipt_digest(&self) -> LiveEpistemicEpochFenceDigestV1 {
        self.receipt_digest
    }
}

fn digest_observation(
    observation: &LiveEpistemicEpochObservationV1,
) -> Result<LiveEpistemicEpochObservationDigestV1, LiveEpistemicEpochFenceError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-live-epistemic-epoch-observation-v1");
    hasher.update(&[1]);
    hasher.update(&observation.observed_at_cycle.to_le_bytes());
    hasher.update(&observation.source_capture_cycle.to_le_bytes());
    hasher.update(&observation.live_v2_digest);
    hasher.update(&observation.live_schema_digest);
    hasher.update(&observation.live_manifest_digest);
    hash_usize(&mut hasher, observation.claim_count)?;
    hash_usize(&mut hasher, observation.evidence_count)?;
    hash_usize(&mut hasher, observation.provenance_count)?;
    hash_usize(&mut hasher, observation.support_state_count)?;
    hash_usize(&mut hasher, observation.mutation_count)?;
    hash_usize(&mut hasher, observation.revision_count)?;
    hasher.update(&[u8::from(observation.live_objects_retained)]);
    hasher.update(&[u8::from(observation.mutation_authority)]);
    hasher.update(&[u8::from(observation.activation_authorized)]);
    hasher.update(&[u8::from(
        observation.trusted_checkpoint_commit_authorized,
    )]);
    Ok(LiveEpistemicEpochObservationDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn digest_fence(
    fence: &LiveEpistemicEpochFenceV1,
) -> Result<LiveEpistemicEpochFenceDigestV1, LiveEpistemicEpochFenceError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-live-epistemic-epoch-fence-v1");
    hasher.update(&[1]);
    hasher.update(&fence.established_at_cycle.to_le_bytes());
    hasher.update(&fence.sandbox_digest);
    hasher.update(&fence.observation.observation_digest.as_bytes());
    hasher.update(&[u8::from(fence.live_epoch_fence_established)]);
    hasher.update(&[u8::from(fence.activation_preflight_authorized)]);
    hasher.update(&[u8::from(fence.activation_authorized)]);
    hasher.update(&[u8::from(fence.trusted_checkpoint_commit_authorized)]);
    Ok(LiveEpistemicEpochFenceDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn digest_continuity_receipt(
    receipt: &LiveEpistemicEpochContinuityReceiptV1,
) -> Result<LiveEpistemicEpochFenceDigestV1, LiveEpistemicEpochFenceError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-live-epistemic-epoch-continuity-v1");
    hasher.update(&[1]);
    hasher.update(&receipt.baseline_observed_at_cycle.to_le_bytes());
    hasher.update(&receipt.rechecked_at_cycle.to_le_bytes());
    hasher.update(&receipt.observation_digest);
    hasher.update(&receipt.live_v2_digest);
    hasher.update(&[u8::from(receipt.live_state_unchanged)]);
    hasher.update(&[u8::from(receipt.activation_preflight_authorized)]);
    hasher.update(&[u8::from(receipt.activation_authorized)]);
    hasher.update(&[u8::from(
        receipt.trusted_checkpoint_commit_authorized,
    )]);
    Ok(LiveEpistemicEpochFenceDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn hash_usize(
    hasher: &mut blake3::Hasher,
    value: usize,
) -> Result<(), LiveEpistemicEpochFenceError> {
    let value = u64::try_from(value).map_err(|_| LiveEpistemicEpochFenceError::LengthOverflow)?;
    hasher.update(&value.to_le_bytes());
    Ok(())
}

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[derive(Debug)]
pub enum LiveEpistemicEpochFenceError {
    LiveCapsuleRejected(EpistemicRestartCapsuleV2Error),
    LiveSourceDiverged(EpistemicRestartCapsuleError),
    LiveSchemaDiverged(BeliefRevisionSchemaPersistenceError),
    SandboxRejected(SplitStateHydrationSandboxError),
    ObservationCycleMismatch {
        capsule_cycle: u64,
        observed_at_cycle: u64,
    },
    RecheckPredatesObservation {
        rechecked_at_cycle: u64,
        observed_at_cycle: u64,
    },
    FenceObservationCycleMismatch {
        established_at_cycle: u64,
        observation_cycle: u64,
    },
    FencePredatesSandbox {
        established_at_cycle: u64,
        sandboxed_at_cycle: u64,
    },
    UnexpectedObservationAuthority,
    UnexpectedFenceAuthority,
    SandboxDigestMismatch,
    ObservationDigestMismatch,
    FenceDigestMismatch,
    LengthOverflow,
}

impl fmt::Display for LiveEpistemicEpochFenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "live epistemic epoch fence rejected: {self:?}")
    }
}

impl Error for LiveEpistemicEpochFenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistoryCapsuleV1,
        BeliefRevisionSchemaHistoryCapsuleV1, ClaimKind, EpistemicRestartCapsuleV1,
    };

    fn empty_live_fixture(cycle: u64) -> (
        EpistemicRestartCapsuleV2,
        EpistemicLedger,
        EpistemicLedgerInventoryV1,
        EpistemicSupportStore,
        BeliefRevisionHistory,
        BeliefRevisionSchemaHistoryV1,
    ) {
        let ledger = EpistemicLedger::new();
        let inventory = EpistemicLedgerInventoryV1::new(vec![], vec![], vec![]).unwrap();
        let store = EpistemicSupportStore::new();
        let mutations = BeliefMutationPersistenceCapsuleV1::capture(&store, &[], cycle).unwrap();
        let history = BeliefRevisionHistory::new();
        let revisions = BeliefRevisionHistoryCapsuleV1::capture(&history, &mutations, cycle).unwrap();
        let schemas = BeliefRevisionSchemaHistoryV1::new();
        let schema_capsule = BeliefRevisionSchemaHistoryCapsuleV1::capture(
            &schemas,
            &history,
            &revisions,
            cycle,
        )
        .unwrap();
        let base = EpistemicRestartCapsuleV1::capture(
            &ledger,
            &inventory,
            &mutations,
            &revisions,
            cycle,
        )
        .unwrap();
        let v2 = EpistemicRestartCapsuleV2::capture(&base, &schema_capsule).unwrap();
        (v2, ledger, inventory, store, history, schemas)
    }

    #[test]
    fn live_observation_rechecks_unchanged_state_without_retaining_live_handles() {
        let cycle = 7;
        let (capsule, ledger, inventory, store, history, schemas) = empty_live_fixture(cycle);
        let observation = LiveEpistemicEpochObservationV1::capture(
            capsule,
            &ledger,
            &inventory,
            &store,
            &history,
            &schemas,
            cycle,
        )
        .unwrap();
        assert!(!observation.live_objects_retained());
        assert!(!observation.activation_authorized());
        let receipt = observation
            .verify_live_unchanged(
                &ledger,
                &inventory,
                &store,
                &history,
                &schemas,
                cycle + 1,
            )
            .unwrap();
        assert!(receipt.live_state_unchanged());
        assert!(!receipt.activation_preflight_authorized());
        assert!(!receipt.activation_authorized());
    }

    #[test]
    fn live_observation_detects_ledger_divergence() {
        let cycle = 7;
        let (capsule, mut ledger, inventory, store, history, schemas) = empty_live_fixture(cycle);
        let observation = LiveEpistemicEpochObservationV1::capture(
            capsule,
            &ledger,
            &inventory,
            &store,
            &history,
            &schemas,
            cycle,
        )
        .unwrap();
        ledger.add_claim("later", ClaimKind::Descriptive, None, None, cycle + 1);
        assert!(observation
            .verify_live_unchanged(
                &ledger,
                &inventory,
                &store,
                &history,
                &schemas,
                cycle + 1,
            )
            .is_err());
    }
}
