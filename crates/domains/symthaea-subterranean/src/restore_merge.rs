// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal conservative merge algebra required by executable restore actions.
//!
//! This tranche intentionally ports only the already-audited replay-barrier
//! join needed by operator restore. Other evidence classes remain separately
//! qualified under RA-27 rather than inheriting a generic merge rule.

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

/// Crate-internal adapter for owner-local replay ledgers represented as raw
/// `(epoch, sequence)` tuples.
pub(crate) const fn merge_replay_barrier_values(
    current: (u64, u64),
    checkpoint: (u64, u64),
) -> (u64, u64) {
    let merged = ReplayBarrier::new(current.0, current.1)
        .merge(ReplayBarrier::new(checkpoint.0, checkpoint.1));
    (merged.epoch(), merged.sequence())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn newer_epoch_dominates_older_high_sequence() {
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
    fn replay_join_is_inflationary_commutative_idempotent_and_associative() {
        for a in barriers() {
            assert_eq!(a.merge(a), a);
            for b in barriers() {
                let ab = a.merge(b);
                assert!(ab >= a);
                assert!(ab >= b);
                assert_eq!(ab, b.merge(a));
                for c in barriers() {
                    assert_eq!(a.merge(b).merge(c), a.merge(b.merge(c)));
                }
            }
        }
    }
}
