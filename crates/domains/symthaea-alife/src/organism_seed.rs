// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, nonzero construction-seed allocation for ALife organisms.
//!
//! `Population` historically derived founder/newborn seeds with `wrapping_add(...).max(1)`.
//! That had two bad boundary behaviors: `seed_base == 0` assigned seed 1 to both of the first
//! two founders, and arithmetic exhaustion wrapped back to previously used seeds. This allocator
//! preserves the historical sequence for ordinary nonzero runs while making zero-base handling
//! distinct and exhaustion fail closed.
//!
//! A persisted cursor is not sufficient resume authority on its own. Production creates exactly
//! one organism-construction seed for every persistent organism identity, so restore is bound to a
//! canonically validated lifecycle ledger and requires the snapshot's issued count to equal the
//! complete historical organism count.

use serde::{Deserialize, Serialize};

use crate::LifecycleLedgerV1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrganismSeedAllocatorSnapshotV1 {
    start_seed: u64,
    issued: u64,
    next: Option<u64>,
}

impl OrganismSeedAllocatorSnapshotV1 {
    pub fn start_seed(self) -> u64 {
        self.start_seed
    }

    pub fn issued_count(self) -> u64 {
        self.issued
    }

    /// Next seed that would be allocated, or `None` once the finite nonzero seed space is exhausted.
    pub fn next_seed(self) -> Option<u64> {
        self.next
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrganismSeedAllocatorErrorV1 {
    Exhausted,
    ZeroStartSeed,
    ZeroNextSeed,
    IssuedCountExceedsSeedSpace {
        start_seed: u64,
        issued: u64,
    },
    CursorMismatch {
        expected: Option<u64>,
        observed: Option<u64>,
    },
    LifecycleCountOverflow,
    LifecycleIssuedCountMismatch {
        expected: u64,
        observed: u64,
    },
}

/// Monotonic allocator over the complete nonzero `u64` seed space.
///
/// Fresh runs start at `max(seed_base, 1)`. Thus every ordinary nonzero seed base keeps its legacy
/// sequence exactly, while `seed_base == 0` deterministically becomes `1, 2, 3, ...` instead of
/// the historical duplicate `1, 1, 2, ...` behavior. `u64::MAX` is a valid final seed; after it is
/// consumed, the allocator becomes explicitly exhausted rather than wrapping to 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrganismSeedAllocatorV1 {
    start_seed: u64,
    issued: u64,
    next: Option<u64>,
}

impl OrganismSeedAllocatorV1 {
    pub fn new(seed_base: u64) -> Self {
        let start_seed = seed_base.max(1);
        Self {
            start_seed,
            issued: 0,
            next: Some(start_seed),
        }
    }

    /// Restore executable seed-allocation authority only when both the serialized state and the
    /// complete validated lifecycle agree about how many organisms have historically existed.
    ///
    /// The snapshot proves internal cursor arithmetic from `start_seed`; the lifecycle supplies an
    /// independent count of founder + birth identities. This catches accidental/forged cursor
    /// rollback that would otherwise reuse a seed for a later organism. It is semantic validation,
    /// not cryptographic authentication of `start_seed` itself.
    pub fn from_snapshot_for_lifecycle(
        snapshot: OrganismSeedAllocatorSnapshotV1,
        lifecycle: &LifecycleLedgerV1,
    ) -> Result<Self, OrganismSeedAllocatorErrorV1> {
        validate_snapshot(snapshot)?;

        let expected = u64::try_from(lifecycle.records().len())
            .map_err(|_| OrganismSeedAllocatorErrorV1::LifecycleCountOverflow)?;
        if snapshot.issued != expected {
            return Err(
                OrganismSeedAllocatorErrorV1::LifecycleIssuedCountMismatch {
                    expected,
                    observed: snapshot.issued,
                },
            );
        }

        Ok(Self {
            start_seed: snapshot.start_seed,
            issued: snapshot.issued,
            next: snapshot.next,
        })
    }

    pub fn snapshot(&self) -> OrganismSeedAllocatorSnapshotV1 {
        OrganismSeedAllocatorSnapshotV1 {
            start_seed: self.start_seed,
            issued: self.issued,
            next: self.next,
        }
    }

    /// Allocate one never-reused, nonzero organism-construction seed.
    pub fn allocate(&mut self) -> Result<u64, OrganismSeedAllocatorErrorV1> {
        let seed = self.next.ok_or(OrganismSeedAllocatorErrorV1::Exhausted)?;
        debug_assert_ne!(seed, 0);
        let issued = self
            .issued
            .checked_add(1)
            .ok_or(OrganismSeedAllocatorErrorV1::IssuedCountExceedsSeedSpace {
                start_seed: self.start_seed,
                issued: self.issued,
            })?;
        self.next = seed.checked_add(1);
        self.issued = issued;
        Ok(seed)
    }
}

fn validate_snapshot(
    snapshot: OrganismSeedAllocatorSnapshotV1,
) -> Result<(), OrganismSeedAllocatorErrorV1> {
    if snapshot.start_seed == 0 {
        return Err(OrganismSeedAllocatorErrorV1::ZeroStartSeed);
    }
    if snapshot.next == Some(0) {
        return Err(OrganismSeedAllocatorErrorV1::ZeroNextSeed);
    }

    let expected = expected_next(snapshot.start_seed, snapshot.issued)?;
    if snapshot.next != expected {
        return Err(OrganismSeedAllocatorErrorV1::CursorMismatch {
            expected,
            observed: snapshot.next,
        });
    }
    Ok(())
}

fn expected_next(
    start_seed: u64,
    issued: u64,
) -> Result<Option<u64>, OrganismSeedAllocatorErrorV1> {
    if start_seed == 0 {
        return Err(OrganismSeedAllocatorErrorV1::ZeroStartSeed);
    }
    if issued == 0 {
        return Ok(Some(start_seed));
    }

    let last_offset = issued - 1;
    let last = start_seed.checked_add(last_offset).ok_or(
        OrganismSeedAllocatorErrorV1::IssuedCountExceedsSeedSpace {
            start_seed,
            issued,
        },
    )?;
    Ok(last.checked_add(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AgentIdAllocator, Genome, GenomeEvidenceV1, LifecycleEventV1, LifecycleTransitionV1,
        OrganismConfig, analyze_lifecycle_events,
    };

    fn founder_events(count: usize) -> Vec<LifecycleEventV1> {
        let mut ids = AgentIdAllocator::new();
        let genome = GenomeEvidenceV1::from_genome(Genome::from_config(&OrganismConfig::default()));
        (0..count)
            .map(|sequence| {
                let id = ids.allocate();
                LifecycleEventV1 {
                    sequence: sequence as u64,
                    tick: 0,
                    transition: LifecycleTransitionV1::Founder {
                        agent_id: id,
                        lineage_id: id,
                        generation: 0,
                        genome,
                        initial_energy_bits: 0.8f64.to_bits(),
                    },
                }
            })
            .collect()
    }

    #[test]
    fn ordinary_nonzero_sequence_matches_historical_population_seeds() {
        let mut seeds = OrganismSeedAllocatorV1::new(42);
        assert_eq!(seeds.allocate(), Ok(42));
        assert_eq!(seeds.allocate(), Ok(43));
        assert_eq!(seeds.allocate(), Ok(44));
        let snapshot = seeds.snapshot();
        assert_eq!(snapshot.start_seed(), 42);
        assert_eq!(snapshot.issued_count(), 3);
        assert_eq!(snapshot.next_seed(), Some(45));
    }

    #[test]
    fn zero_seed_base_no_longer_duplicates_the_first_founder_seed() {
        let mut seeds = OrganismSeedAllocatorV1::new(0);
        assert_eq!(seeds.allocate(), Ok(1));
        assert_eq!(seeds.allocate(), Ok(2));
        assert_eq!(seeds.allocate(), Ok(3));
    }

    #[test]
    fn maximum_seed_is_used_once_then_exhaustion_fails_closed() {
        let mut seeds = OrganismSeedAllocatorV1::new(u64::MAX);
        assert_eq!(seeds.allocate(), Ok(u64::MAX));
        assert_eq!(seeds.snapshot().next_seed(), None);
        assert_eq!(seeds.allocate(), Err(OrganismSeedAllocatorErrorV1::Exhausted));
        assert_eq!(seeds.allocate(), Err(OrganismSeedAllocatorErrorV1::Exhausted));
    }

    #[test]
    fn lifecycle_bound_serialized_snapshot_restores_exact_future_seed_sequence() {
        let lifecycle = analyze_lifecycle_events(&founder_events(2)).expect("complete lifecycle");
        let mut uninterrupted = OrganismSeedAllocatorV1::new(9);
        assert_eq!(uninterrupted.allocate(), Ok(9));
        assert_eq!(uninterrupted.allocate(), Ok(10));

        let encoded = serde_json::to_string(&uninterrupted.snapshot()).expect("serialize");
        let decoded: OrganismSeedAllocatorSnapshotV1 =
            serde_json::from_str(&encoded).expect("deserialize");
        let mut restored =
            OrganismSeedAllocatorV1::from_snapshot_for_lifecycle(decoded, &lifecycle)
                .expect("lifecycle-bound restore");

        for _ in 0..64 {
            assert_eq!(restored.allocate(), uninterrupted.allocate());
        }
        assert_eq!(restored.snapshot(), uninterrupted.snapshot());
    }

    #[test]
    fn rolled_back_seed_issue_count_cannot_authorize_restore() {
        let lifecycle = analyze_lifecycle_events(&founder_events(2)).expect("complete lifecycle");
        let forged = OrganismSeedAllocatorSnapshotV1 {
            start_seed: 9,
            issued: 1,
            next: Some(10),
        };
        assert_eq!(
            OrganismSeedAllocatorV1::from_snapshot_for_lifecycle(forged, &lifecycle),
            Err(
                OrganismSeedAllocatorErrorV1::LifecycleIssuedCountMismatch {
                    expected: 2,
                    observed: 1,
                }
            )
        );
    }

    #[test]
    fn internally_inconsistent_cursor_is_rejected() {
        let lifecycle = analyze_lifecycle_events(&founder_events(2)).expect("complete lifecycle");
        let forged = OrganismSeedAllocatorSnapshotV1 {
            start_seed: 9,
            issued: 2,
            next: Some(99),
        };
        assert_eq!(
            OrganismSeedAllocatorV1::from_snapshot_for_lifecycle(forged, &lifecycle),
            Err(OrganismSeedAllocatorErrorV1::CursorMismatch {
                expected: Some(11),
                observed: Some(99),
            })
        );
    }

    #[test]
    fn zero_persisted_cursor_is_rejected() {
        let lifecycle = analyze_lifecycle_events(&[]).expect("empty lifecycle");
        let snapshot = OrganismSeedAllocatorSnapshotV1 {
            start_seed: 1,
            issued: 0,
            next: Some(0),
        };
        assert_eq!(
            OrganismSeedAllocatorV1::from_snapshot_for_lifecycle(snapshot, &lifecycle),
            Err(OrganismSeedAllocatorErrorV1::ZeroNextSeed)
        );
    }
}
