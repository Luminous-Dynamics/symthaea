// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative evidence/configuration merge algebra for operational restore.
//!
//! This module intentionally contains only small deterministic value operations.
//! Domain owners must first classify evidence polarity through RA-20 before they
//! may select one of these primitives. There is deliberately no generic
//! `merge_any_evidence` operation.

use super::restore_actions::EvidenceRestorePolicy;

/// Executable merge primitive licensed by one audited evidence polarity.
///
/// Unsupported means the evidence class has not yet received an executable
/// conservative algebra in this module; callers must fail closed rather than
/// substitute another primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MergePrimitive {
    ReplayBarrierJoin,
    RestrictionCounterJoin,
    FreshOnlyReset,
    Unsupported,
}

pub(super) const fn primitive_for_policy(policy: EvidenceRestorePolicy) -> MergePrimitive {
    match policy {
        EvidenceRestorePolicy::ReplayBarrier => MergePrimitive::ReplayBarrierJoin,
        EvidenceRestorePolicy::RestrictionSupporting => MergePrimitive::RestrictionCounterJoin,
        EvidenceRestorePolicy::RecoverySupportingFreshOnly => MergePrimitive::FreshOnlyReset,
        EvidenceRestorePolicy::CounterexamplePreserving | EvidenceRestorePolicy::NeutralHistory => {
            MergePrimitive::Unsupported
        }
    }
}

/// Replay barrier for one principal/source stream.
///
/// Ordering is lexicographic: a newer epoch dominates every sequence from an
/// older epoch; within one epoch, the highest consumed sequence wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct ReplayBarrier {
    epoch: u64,
    sequence: u64,
}

impl ReplayBarrier {
    pub(super) const fn new(epoch: u64, sequence: u64) -> Self {
        Self { epoch, sequence }
    }

    pub(super) const fn epoch(self) -> u64 {
        self.epoch
    }

    pub(super) const fn sequence(self) -> u64 {
        self.sequence
    }

    /// Monotone join for already-consumed replay evidence.
    pub(super) const fn merge(self, other: Self) -> Self {
        if other.epoch > self.epoch
            || (other.epoch == self.epoch && other.sequence > self.sequence)
        {
            other
        } else {
            self
        }
    }
}

/// Restriction-supporting evidence counter.
///
/// This is appropriate only after a domain audit proves that a larger value can
/// justify equal or narrower authority. Examples include consecutive watchdog
/// failures before a restrictive latch. It must not be used for healthy/recovery
/// dwell or other widening-supporting evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct RestrictionEvidenceCounter(u64);

impl RestrictionEvidenceCounter {
    pub(super) const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(super) const fn value(self) -> u64 {
        self.0
    }

    pub(super) const fn merge(self, other: Self) -> Self {
        if other.0 > self.0 { other } else { self }
    }
}

/// Recovery/widening-supporting progress.
///
/// Persisted progress is evidence about a historical recovery attempt, not
/// authority to continue widening after restore. The safe restore value is zero
/// and must be re-earned from fresh post-restore evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FreshRecoveryCredit(u64);

impl FreshRecoveryCredit {
    pub(super) const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(super) const fn value(self) -> u64 {
        self.0
    }

    /// Restore boundary: discard both current/persisted recovery credit rather
    /// than guessing which portion remains fresh enough to widen authority.
    pub(super) const fn after_restore(_current: Self, _checkpoint: Self) -> Self {
        Self(0)
    }
}

/// Content identity for safety-relevant policy/configuration.
///
/// This digest is only an equality descriptor in this pure algebra. A later
/// trusted owner must bind it to verified deployment/configuration identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PolicyDigest([u8; 32]);

impl PolicyDigest {
    pub(super) const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub(super) const fn is_valid(self) -> bool {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PolicyReconciliation {
    /// Candidate policy is exactly the currently trusted policy identity.
    ExactCurrentPolicy,
    /// Candidate differs; restore may preserve historical state but productive
    /// activation remains blocked until explicit policy/config reconciliation.
    ReconciliationRequired,
    /// Zero/unset digest is never accepted as proof of equivalence.
    InvalidIdentity,
}

pub(super) const fn reconcile_policy(
    current: PolicyDigest,
    checkpoint: PolicyDigest,
) -> PolicyReconciliation {
    if !current.is_valid() || !checkpoint.is_valid() {
        PolicyReconciliation::InvalidIdentity
    } else if current.0 == checkpoint.0 {
        PolicyReconciliation::ExactCurrentPolicy
    } else {
        PolicyReconciliation::ReconciliationRequired
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_polarity_maps_only_to_its_licensed_primitive() {
        assert_eq!(
            primitive_for_policy(EvidenceRestorePolicy::ReplayBarrier),
            MergePrimitive::ReplayBarrierJoin
        );
        assert_eq!(
            primitive_for_policy(EvidenceRestorePolicy::RestrictionSupporting),
            MergePrimitive::RestrictionCounterJoin
        );
        assert_eq!(
            primitive_for_policy(EvidenceRestorePolicy::RecoverySupportingFreshOnly),
            MergePrimitive::FreshOnlyReset
        );
        assert_eq!(
            primitive_for_policy(EvidenceRestorePolicy::CounterexamplePreserving),
            MergePrimitive::Unsupported
        );
        assert_eq!(
            primitive_for_policy(EvidenceRestorePolicy::NeutralHistory),
            MergePrimitive::Unsupported
        );
    }

    fn barriers() -> [ReplayBarrier; 6] {
        [
            ReplayBarrier::new(0, 0),
            ReplayBarrier::new(1, 0),
            ReplayBarrier::new(1, 1),
            ReplayBarrier::new(1, 99),
            ReplayBarrier::new(2, 0),
            ReplayBarrier::new(2, 7),
        ]
    }

    #[test]
    fn replay_barrier_epoch_dominates_sequence() {
        let older_high_sequence = ReplayBarrier::new(4, u64::MAX);
        let newer_epoch = ReplayBarrier::new(5, 0);
        assert_eq!(older_high_sequence.merge(newer_epoch), newer_epoch);
        assert_eq!(newer_epoch.merge(older_high_sequence), newer_epoch);
    }

    #[test]
    fn replay_merge_is_inflationary_commutative_and_idempotent() {
        for left in barriers() {
            assert_eq!(left.merge(left), left);
            for right in barriers() {
                let merged = left.merge(right);
                assert!(merged >= left);
                assert!(merged >= right);
                assert_eq!(left.merge(right), right.merge(left));
            }
        }
    }

    #[test]
    fn replay_merge_is_associative() {
        for a in barriers() {
            for b in barriers() {
                for c in barriers() {
                    assert_eq!(a.merge(b).merge(c), a.merge(b.merge(c)));
                }
            }
        }
    }

    #[test]
    fn replay_barrier_accessors_preserve_exact_values() {
        let barrier = ReplayBarrier::new(17, 23);
        assert_eq!(barrier.epoch(), 17);
        assert_eq!(barrier.sequence(), 23);
    }

    fn counters() -> [RestrictionEvidenceCounter; 6] {
        [0, 1, 2, 7, 99, u32::MAX as u64]
            .map(RestrictionEvidenceCounter::new)
    }

    #[test]
    fn restriction_counter_merge_is_inflationary_commutative_idempotent() {
        for left in counters() {
            assert_eq!(left.merge(left), left);
            for right in counters() {
                let merged = left.merge(right);
                assert!(merged >= left);
                assert!(merged >= right);
                assert_eq!(left.merge(right), right.merge(left));
            }
        }
    }

    #[test]
    fn restriction_counter_merge_is_associative() {
        for a in counters() {
            for b in counters() {
                for c in counters() {
                    assert_eq!(a.merge(b).merge(c), a.merge(b.merge(c)));
                }
            }
        }
    }

    #[test]
    fn restriction_counter_never_moves_adverse_evidence_backward() {
        let current = RestrictionEvidenceCounter::new(9);
        let stale_checkpoint = RestrictionEvidenceCounter::new(3);
        assert_eq!(current.merge(stale_checkpoint).value(), 9);

        let checkpoint_with_unseen_adverse_history = RestrictionEvidenceCounter::new(12);
        assert_eq!(current.merge(checkpoint_with_unseen_adverse_history).value(), 12);
    }

    #[test]
    fn recovery_credit_is_always_reearned_after_restore() {
        for current in [0, 1, 50, 200] {
            for checkpoint in [0, 1, 50, 200] {
                let restored = FreshRecoveryCredit::after_restore(
                    FreshRecoveryCredit::new(current),
                    FreshRecoveryCredit::new(checkpoint),
                );
                assert_eq!(restored.value(), 0);
            }
        }
    }

    #[test]
    fn stale_checkpoint_cannot_resurrect_recovery_credit() {
        let current = FreshRecoveryCredit::new(0);
        let historical_near_quorum = FreshRecoveryCredit::new(199);
        assert_eq!(
            FreshRecoveryCredit::after_restore(current, historical_near_quorum).value(),
            0
        );
    }

    #[test]
    fn exact_policy_identity_can_reconcile() {
        let current = PolicyDigest::new([7; 32]);
        assert_eq!(
            reconcile_policy(current, current),
            PolicyReconciliation::ExactCurrentPolicy
        );
    }

    #[test]
    fn stale_policy_identity_blocks_activation() {
        assert_eq!(
            reconcile_policy(PolicyDigest::new([7; 32]), PolicyDigest::new([8; 32])),
            PolicyReconciliation::ReconciliationRequired
        );
    }

    #[test]
    fn missing_policy_identity_never_means_equivalent() {
        assert_eq!(
            reconcile_policy(PolicyDigest::new([0; 32]), PolicyDigest::new([7; 32])),
            PolicyReconciliation::InvalidIdentity
        );
        assert_eq!(
            reconcile_policy(PolicyDigest::new([7; 32]), PolicyDigest::new([0; 32])),
            PolicyReconciliation::InvalidIdentity
        );
    }
}
