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
    ScopedRestrictionCounterJoin,
    FreshOnlyReset,
    Unsupported,
}

pub(super) const fn primitive_for_policy(policy: EvidenceRestorePolicy) -> MergePrimitive {
    match policy {
        EvidenceRestorePolicy::ReplayBarrier => MergePrimitive::ReplayBarrierJoin,
        EvidenceRestorePolicy::RestrictionSupporting => {
            MergePrimitive::ScopedRestrictionCounterJoin
        }
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

/// Crate-internal adapter for domain owners whose private replay ledger is stored
/// as raw `(epoch, sequence)` tuples. The ordering rule remains centralized in
/// RA-21 without exposing the `ReplayBarrier` type downstream.
pub(crate) const fn merge_replay_barrier_values(
    current: (u64, u64),
    checkpoint: (u64, u64),
) -> (u64, u64) {
    let merged = ReplayBarrier::new(current.0, current.1)
        .merge(ReplayBarrier::new(checkpoint.0, checkpoint.1));
    (merged.epoch(), merged.sequence())
}

/// Identity of the evidence window in which a restriction-supporting counter is
/// meaningful.
///
/// A numeric counter is not portable across arbitrary time/boot windows. The
/// trusted owner must derive these values from its actual boot/evidence timeline;
/// they are equality fences, not self-authenticating authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct EvidenceScope {
    boot_epoch: u64,
    window_id: u64,
}

impl EvidenceScope {
    pub(super) const fn new(boot_epoch: u64, window_id: u64) -> Self {
        Self {
            boot_epoch,
            window_id,
        }
    }
}

/// Restriction-supporting evidence within one exact evidence scope.
///
/// A larger value is safe to preserve only after the domain audit has proved
/// that the counter is adverse/restriction-supporting *and* both values refer to
/// the same boot/window semantics. A consecutive watchdog-failure streak from a
/// historical window must not be numerically joined into a different current
/// streak merely because both are integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ScopedRestrictionEvidence {
    scope: EvidenceScope,
    value: u64,
}

impl ScopedRestrictionEvidence {
    pub(super) const fn new(scope: EvidenceScope, value: u64) -> Self {
        Self { scope, value }
    }

    pub(super) const fn scope(self) -> EvidenceScope {
        self.scope
    }

    pub(super) const fn value(self) -> u64 {
        self.value
    }

    pub(super) const fn merge(self, other: Self) -> RestrictionEvidenceMerge {
        if self.scope != other.scope {
            RestrictionEvidenceMerge::ReconciliationRequired {
                current_scope: self.scope,
                checkpoint_scope: other.scope,
            }
        } else if other.value > self.value {
            RestrictionEvidenceMerge::Merged(other)
        } else {
            RestrictionEvidenceMerge::Merged(self)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestrictionEvidenceMerge {
    Merged(ScopedRestrictionEvidence),
    ReconciliationRequired {
        current_scope: EvidenceScope,
        checkpoint_scope: EvidenceScope,
    },
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
            MergePrimitive::ScopedRestrictionCounterJoin
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
    fn raw_tuple_adapter_uses_exact_replay_algebra() {
        assert_eq!(
            merge_replay_barrier_values((4, u64::MAX), (5, 0)),
            (5, 0)
        );
        assert_eq!(merge_replay_barrier_values((5, 3), (5, 9)), (5, 9));
        assert_eq!(merge_replay_barrier_values((5, 9), (5, 3)), (5, 9));
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

    fn scoped(value: u64) -> ScopedRestrictionEvidence {
        ScopedRestrictionEvidence::new(EvidenceScope::new(9, 17), value)
    }

    #[test]
    fn same_scope_restriction_merge_preserves_greater_adverse_evidence() {
        for left in [0, 1, 7, 99] {
            for right in [0, 1, 7, 99] {
                let expected = left.max(right);
                assert_eq!(
                    scoped(left).merge(scoped(right)),
                    RestrictionEvidenceMerge::Merged(scoped(expected))
                );
                assert_eq!(
                    scoped(left).merge(scoped(right)),
                    scoped(right).merge(scoped(left))
                );
            }
        }
    }

    #[test]
    fn same_scope_restriction_merge_is_associative() {
        for a in [0, 1, 7] {
            for b in [0, 1, 7] {
                for c in [0, 1, 7] {
                    let ab = match scoped(a).merge(scoped(b)) {
                        RestrictionEvidenceMerge::Merged(value) => value,
                        _ => unreachable!(),
                    };
                    let bc = match scoped(b).merge(scoped(c)) {
                        RestrictionEvidenceMerge::Merged(value) => value,
                        _ => unreachable!(),
                    };
                    assert_eq!(ab.merge(scoped(c)), scoped(a).merge(bc));
                }
            }
        }
    }

    #[test]
    fn mismatched_restriction_scope_requires_reconciliation_not_max() {
        let current = ScopedRestrictionEvidence::new(EvidenceScope::new(4, 10), 1);
        let historical = ScopedRestrictionEvidence::new(EvidenceScope::new(3, 99), 1000);
        assert_eq!(
            current.merge(historical),
            RestrictionEvidenceMerge::ReconciliationRequired {
                current_scope: current.scope(),
                checkpoint_scope: historical.scope(),
            }
        );
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
