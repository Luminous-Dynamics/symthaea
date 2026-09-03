// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Two-phase admission for operational checkpoint restore.
//!
//! Restore is not deserialization. A checkpoint is first compared with the
//! current live authority/evidence/physical context without mutating runtime
//! state. The resulting prepared plan is affine, bound to that exact context,
//! and can be committed only while every generation fence still matches.
//!
//! This module intentionally does **not** mutate `SubterraneanEmbodiment` yet.
//! It is the pure transaction primitive that a later owner-bound integration
//! can use after domain-specific RA-17 restore semantics are implemented.

use super::restore_semantics::{OPERATIONAL_RESTORE_CONTRACTS, RestoreDomain};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RestoreDigest([u8; 32]);

impl RestoreDigest {
    pub(crate) const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub(crate) const fn bytes(self) -> [u8; 32] {
        self.0
    }

    const fn is_valid(self) -> bool {
        let mut index = 0;
        while index < self.0.len() {
            if self.0[index] != 0 {
                return true;
            }
            index += 1;
        }
        false
    }
}

/// Exact live context against which a restore decision was prepared.
///
/// Generation counters are equality fences, not clocks and not authority by
/// themselves. The trusted owner is responsible for advancing them whenever
/// the corresponding live truth changes in a restore-relevant way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RestoreGenerationFence {
    pub(crate) boot_epoch: u64,
    pub(crate) control_plane_generation: u64,
    pub(crate) authority_generation: u64,
    pub(crate) evidence_generation: u64,
    pub(crate) physical_state_generation: u64,
    pub(crate) live_snapshot_digest: RestoreDigest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RestorePreparationContext {
    pub(crate) checkpoint_digest: RestoreDigest,
    pub(crate) fence: RestoreGenerationFence,
}

/// Result of one domain-specific restore comparison.
///
/// `ConservativeRequalification` and `ReconciliationRequired` are admissible
/// only because they preserve an explicit restrictive/reconciliation action in
/// the committed plan. They are never silently promoted to historical replace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RestoreAdmissionVerdict {
    ProvenNonWidening,
    ConservativeRequalification,
    ReconciliationRequired,
    Widening,
    NotProvable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RestoreDomainDecision {
    pub(crate) domain: RestoreDomain,
    pub(crate) verdict: RestoreAdmissionVerdict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RestorePrepareError {
    InvalidCheckpointDigest,
    InvalidLiveSnapshotDigest,
    MissingDomain(RestoreDomain),
    DuplicateDomain(RestoreDomain),
    Widening(RestoreDomain),
    NotProvable(RestoreDomain),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RestoreFenceField {
    BootEpoch,
    ControlPlaneGeneration,
    AuthorityGeneration,
    EvidenceGeneration,
    PhysicalStateGeneration,
    LiveSnapshotDigest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RestoreCommitError {
    pub(crate) changed: RestoreFenceField,
}

/// Affine prepared restore authority.
///
/// Deliberately not `Clone`, `Copy`, `Serialize` or `Deserialize`. Raw portable
/// checkpoint/evidence data must never deserialize into this live transaction.
pub(crate) struct PreparedOperationalRestore {
    checkpoint_digest: RestoreDigest,
    fence: RestoreGenerationFence,
    decisions: Vec<RestoreDomainDecision>,
}

impl PreparedOperationalRestore {
    pub(crate) fn checkpoint_digest(&self) -> RestoreDigest {
        self.checkpoint_digest
    }

    pub(crate) fn decisions(&self) -> &[RestoreDomainDecision] {
        &self.decisions
    }
}

/// Single-use committed restore plan.
///
/// This is still not permission to mutate arbitrary runtime state. A later
/// owner-bound integration must interpret each decision according to the RA-17
/// restore registry and perform the actual atomic mutation/requalification.
pub(crate) struct CommittedOperationalRestore {
    checkpoint_digest: RestoreDigest,
    decisions: Vec<RestoreDomainDecision>,
}

impl CommittedOperationalRestore {
    pub(crate) fn checkpoint_digest(&self) -> RestoreDigest {
        self.checkpoint_digest
    }

    pub(crate) fn decisions(&self) -> &[RestoreDomainDecision] {
        &self.decisions
    }
}

/// Prepare a complete, canonical restore plan without mutating live state.
///
/// Every current `RestoreDomain` must appear exactly once. Widening and
/// unprovable transitions fail closed. Successful decisions are reordered to
/// the canonical checkpoint-registry order so later commit/apply logic cannot
/// depend on caller ordering.
pub(crate) fn prepare_operational_restore(
    context: RestorePreparationContext,
    decisions: Vec<RestoreDomainDecision>,
) -> Result<PreparedOperationalRestore, RestorePrepareError> {
    if !context.checkpoint_digest.is_valid() {
        return Err(RestorePrepareError::InvalidCheckpointDigest);
    }
    if !context.fence.live_snapshot_digest.is_valid() {
        return Err(RestorePrepareError::InvalidLiveSnapshotDigest);
    }

    let mut canonical = Vec::with_capacity(OPERATIONAL_RESTORE_CONTRACTS.len());
    for contract in OPERATIONAL_RESTORE_CONTRACTS {
        let mut matches = decisions
            .iter()
            .copied()
            .filter(|decision| decision.domain == contract.domain);
        let Some(decision) = matches.next() else {
            return Err(RestorePrepareError::MissingDomain(contract.domain));
        };
        if matches.next().is_some() {
            return Err(RestorePrepareError::DuplicateDomain(contract.domain));
        }
        match decision.verdict {
            RestoreAdmissionVerdict::Widening => {
                return Err(RestorePrepareError::Widening(decision.domain));
            }
            RestoreAdmissionVerdict::NotProvable => {
                return Err(RestorePrepareError::NotProvable(decision.domain));
            }
            RestoreAdmissionVerdict::ProvenNonWidening
            | RestoreAdmissionVerdict::ConservativeRequalification
            | RestoreAdmissionVerdict::ReconciliationRequired => {}
        }
        canonical.push(decision);
    }

    // Exact coverage above also rejects extra/unknown duplicate entries. The
    // enum currently contains exactly the registered domains, so a longer input
    // can only be produced by duplicating a registered domain.
    if decisions.len() != OPERATIONAL_RESTORE_CONTRACTS.len() {
        for contract in OPERATIONAL_RESTORE_CONTRACTS {
            let count = decisions
                .iter()
                .filter(|decision| decision.domain == contract.domain)
                .count();
            if count > 1 {
                return Err(RestorePrepareError::DuplicateDomain(contract.domain));
            }
        }
        // Defensive fallback: if RestoreDomain later gains a variant before the
        // registry is updated, preparation still fails closed rather than
        // accepting an unclassified decision set.
        return Err(RestorePrepareError::MissingDomain(
            OPERATIONAL_RESTORE_CONTRACTS[0].domain,
        ));
    }

    Ok(PreparedOperationalRestore {
        checkpoint_digest: context.checkpoint_digest,
        fence: context.fence,
        decisions: canonical,
    })
}

/// Commit consumes the prepared token and rechecks the complete live fence.
///
/// A restriction/evidence/physical change after preparation invalidates the
/// transaction. The caller must prepare again against the new world state.
pub(crate) fn commit_operational_restore(
    prepared: PreparedOperationalRestore,
    current: RestoreGenerationFence,
) -> Result<CommittedOperationalRestore, RestoreCommitError> {
    let expected = prepared.fence;
    let changed = if current.boot_epoch != expected.boot_epoch {
        Some(RestoreFenceField::BootEpoch)
    } else if current.control_plane_generation != expected.control_plane_generation {
        Some(RestoreFenceField::ControlPlaneGeneration)
    } else if current.authority_generation != expected.authority_generation {
        Some(RestoreFenceField::AuthorityGeneration)
    } else if current.evidence_generation != expected.evidence_generation {
        Some(RestoreFenceField::EvidenceGeneration)
    } else if current.physical_state_generation != expected.physical_state_generation {
        Some(RestoreFenceField::PhysicalStateGeneration)
    } else if current.live_snapshot_digest != expected.live_snapshot_digest {
        Some(RestoreFenceField::LiveSnapshotDigest)
    } else {
        None
    };

    if let Some(changed) = changed {
        return Err(RestoreCommitError { changed });
    }

    Ok(CommittedOperationalRestore {
        checkpoint_digest: prepared.checkpoint_digest,
        decisions: prepared.decisions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> RestoreDigest {
        RestoreDigest::new([byte; 32])
    }

    fn fence() -> RestoreGenerationFence {
        RestoreGenerationFence {
            boot_epoch: 7,
            control_plane_generation: 11,
            authority_generation: 13,
            evidence_generation: 17,
            physical_state_generation: 19,
            live_snapshot_digest: digest(23),
        }
    }

    fn context() -> RestorePreparationContext {
        RestorePreparationContext {
            checkpoint_digest: digest(29),
            fence: fence(),
        }
    }

    fn nominal_decisions() -> Vec<RestoreDomainDecision> {
        OPERATIONAL_RESTORE_CONTRACTS
            .iter()
            .map(|contract| RestoreDomainDecision {
                domain: contract.domain,
                verdict: RestoreAdmissionVerdict::ProvenNonWidening,
            })
            .collect()
    }

    fn prepared() -> PreparedOperationalRestore {
        prepare_operational_restore(context(), nominal_decisions()).expect("valid preparation")
    }

    #[test]
    fn complete_non_widening_restore_prepares_and_commits() {
        let prepared = prepared();
        assert_eq!(prepared.checkpoint_digest().bytes(), [29; 32]);
        assert_eq!(
            prepared.decisions().len(),
            OPERATIONAL_RESTORE_CONTRACTS.len()
        );
        let committed = commit_operational_restore(prepared, fence()).expect("fence unchanged");
        assert_eq!(committed.checkpoint_digest().bytes(), [29; 32]);
        assert_eq!(
            committed.decisions().len(),
            OPERATIONAL_RESTORE_CONTRACTS.len()
        );
    }

    #[test]
    fn missing_domain_fails_preparation() {
        let mut decisions = nominal_decisions();
        let missing = decisions.pop().expect("non-empty registry").domain;
        assert_eq!(
            prepare_operational_restore(context(), decisions).err(),
            Some(RestorePrepareError::MissingDomain(missing))
        );
    }

    #[test]
    fn duplicate_domain_fails_preparation() {
        let mut decisions = nominal_decisions();
        let duplicate = decisions[0];
        decisions.push(duplicate);
        assert_eq!(
            prepare_operational_restore(context(), decisions).err(),
            Some(RestorePrepareError::DuplicateDomain(duplicate.domain))
        );
    }

    #[test]
    fn widening_fails_closed() {
        let mut decisions = nominal_decisions();
        let domain = decisions[2].domain;
        decisions[2].verdict = RestoreAdmissionVerdict::Widening;
        assert_eq!(
            prepare_operational_restore(context(), decisions).err(),
            Some(RestorePrepareError::Widening(domain))
        );
    }

    #[test]
    fn not_provable_fails_closed() {
        let mut decisions = nominal_decisions();
        let domain = decisions[4].domain;
        decisions[4].verdict = RestoreAdmissionVerdict::NotProvable;
        assert_eq!(
            prepare_operational_restore(context(), decisions).err(),
            Some(RestorePrepareError::NotProvable(domain))
        );
    }

    #[test]
    fn conservative_and_reconciliation_actions_survive_commit_explicitly() {
        let mut decisions = nominal_decisions();
        decisions[3].verdict = RestoreAdmissionVerdict::ConservativeRequalification;
        decisions[4].verdict = RestoreAdmissionVerdict::ReconciliationRequired;
        let conservative_domain = decisions[3].domain;
        let reconciliation_domain = decisions[4].domain;
        let prepared =
            prepare_operational_restore(context(), decisions).expect("safe explicit actions");
        let committed = commit_operational_restore(prepared, fence()).expect("unchanged fence");
        assert!(committed.decisions().iter().any(|decision| {
            decision.domain == conservative_domain
                && decision.verdict == RestoreAdmissionVerdict::ConservativeRequalification
        }));
        assert!(committed.decisions().iter().any(|decision| {
            decision.domain == reconciliation_domain
                && decision.verdict == RestoreAdmissionVerdict::ReconciliationRequired
        }));
    }

    #[test]
    fn any_generation_or_snapshot_change_invalidates_prepared_restore() {
        let base = fence();
        let mut changed_fences = Vec::new();

        let mut changed = base;
        changed.boot_epoch += 1;
        changed_fences.push((RestoreFenceField::BootEpoch, changed));

        let mut changed = base;
        changed.control_plane_generation += 1;
        changed_fences.push((RestoreFenceField::ControlPlaneGeneration, changed));

        let mut changed = base;
        changed.authority_generation += 1;
        changed_fences.push((RestoreFenceField::AuthorityGeneration, changed));

        let mut changed = base;
        changed.evidence_generation += 1;
        changed_fences.push((RestoreFenceField::EvidenceGeneration, changed));

        let mut changed = base;
        changed.physical_state_generation += 1;
        changed_fences.push((RestoreFenceField::PhysicalStateGeneration, changed));

        let mut changed = base;
        changed.live_snapshot_digest = digest(31);
        changed_fences.push((RestoreFenceField::LiveSnapshotDigest, changed));

        for (expected, changed) in changed_fences {
            let error = commit_operational_restore(prepared(), changed)
                .err()
                .expect("stale prepared restore must fail");
            assert_eq!(error.changed, expected);
        }
    }

    #[test]
    fn invalid_digests_fail_before_preparation() {
        let mut invalid_checkpoint = context();
        invalid_checkpoint.checkpoint_digest = RestoreDigest::new([0; 32]);
        assert_eq!(
            prepare_operational_restore(invalid_checkpoint, nominal_decisions()).err(),
            Some(RestorePrepareError::InvalidCheckpointDigest)
        );

        let mut invalid_live = context();
        invalid_live.fence.live_snapshot_digest = RestoreDigest::new([0; 32]);
        assert_eq!(
            prepare_operational_restore(invalid_live, nominal_decisions()).err(),
            Some(RestorePrepareError::InvalidLiveSnapshotDigest)
        );
    }

    #[test]
    fn checkpoint_identity_is_preserved_across_prepare_and_commit() {
        let prepared = prepared();
        let expected = prepared.checkpoint_digest();
        let committed = commit_operational_restore(prepared, fence()).expect("unchanged fence");
        assert_eq!(committed.checkpoint_digest(), expected);
    }
}
