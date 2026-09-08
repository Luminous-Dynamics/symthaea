// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent per-organism identity, per `ALIFE_MULTIAGENT_GENESIS_PLAN_2026-07-25.md` (G0a).
//!
//! An [`AgentId`] is allocated once by [`AgentIdAllocator`] and travels with an [`crate::Organism`]
//! through ticks, reproduction, and death — it is never derived from `Population`'s backing `Vec`
//! index, since that index shifts on every birth/death (`Vec::remove`/`extend`). This is the Stage
//! 0 "Identity" invariant: an `AgentId` must resolve to the same conceptual individual regardless
//! of population reordering, never silently alias to a different one.

use serde::{Deserialize, Serialize};

use crate::lifecycle::LifecycleLedgerV1;

/// A persistent, never-reused organism identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AgentId(u64);

impl AgentId {
    /// Placeholder for call sites that don't participate in a [`crate::Population`]'s allocator
    /// (e.g. `Organism::new` used directly in single-organism Phase 0-7 tests, where identity is
    /// irrelevant). Never emitted by [`AgentIdAllocator::allocate`], so it can never collide with
    /// a real allocated id.
    pub const UNALLOCATED: AgentId = AgentId(u64::MAX);

    /// The raw numeric id, for logging/serialization.
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// Persistable cursor for the monotonic identity allocator.
///
/// This value is deliberately not sufficient authority to resume allocation by itself. A forged
/// lower cursor could reuse an identity that already appears in the complete lifecycle history.
/// Restore therefore requires [`AgentIdAllocator::from_snapshot_for_lifecycle`], which binds the
/// cursor to a canonically validated [`LifecycleLedgerV1`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentIdAllocatorSnapshotV1 {
    next: u64,
}

impl AgentIdAllocatorSnapshotV1 {
    pub fn next_raw(self) -> u64 {
        self.next
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentIdAllocatorRestoreErrorV1 {
    /// The supposedly complete lifecycle contains the reserved non-population identity sentinel.
    HistoricalReservedIdentity,
    /// Production allocation is contiguous from zero. A gap/reorder means the history cannot be
    /// the complete output of this allocator and therefore cannot authorize allocator restoration.
    HistoricalIdentitySequenceMismatch { expected: u64, observed: u64 },
    /// The persisted allocator cursor must be exactly one past every historically allocated ID.
    NextIdentityMismatch { expected: u64, observed: u64 },
}

/// Monotonic allocator — every successful call to [`AgentIdAllocator::allocate`] returns a fresh
/// id, never reused, so a later death/birth can never cause one individual's id to be handed to
/// another.
///
/// `u64::MAX` is permanently reserved for [`AgentId::UNALLOCATED`]. Once every allocatable value
/// `0..u64::MAX` has been consumed, allocation fails closed by panicking before changing allocator
/// state. This boundary is astronomically remote in practice, but making it explicit prevents a
/// release-build integer wrap from violating the crate's identity theorem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentIdAllocator {
    next: u64,
}

impl AgentIdAllocator {
    pub fn new() -> Self {
        Self { next: 0 }
    }

    /// Restore the allocator only when its persisted cursor is exactly implied by a complete,
    /// canonically validated lifecycle ledger.
    ///
    /// This proves more than `snapshot.next > max_seen_id`: the ledger must contain the exact
    /// contiguous allocation prefix `0, 1, ..., next-1`. That catches both rollback/reuse and a
    /// supposedly complete history with skipped identities. The latter cannot be produced by this
    /// allocator at a stable simulation boundary.
    pub fn from_snapshot_for_lifecycle(
        snapshot: AgentIdAllocatorSnapshotV1,
        lifecycle: &LifecycleLedgerV1,
    ) -> Result<Self, AgentIdAllocatorRestoreErrorV1> {
        let mut expected = 0u64;
        for agent_id in lifecycle.records().keys().copied() {
            if agent_id == AgentId::UNALLOCATED {
                return Err(AgentIdAllocatorRestoreErrorV1::HistoricalReservedIdentity);
            }
            if agent_id.raw() != expected {
                return Err(
                    AgentIdAllocatorRestoreErrorV1::HistoricalIdentitySequenceMismatch {
                        expected,
                        observed: agent_id.raw(),
                    },
                );
            }
            // `UNALLOCATED == u64::MAX` was rejected above, so a valid historical ID is at most
            // u64::MAX - 1 and incrementing the expected cursor cannot overflow.
            expected += 1;
        }

        if snapshot.next != expected {
            return Err(AgentIdAllocatorRestoreErrorV1::NextIdentityMismatch {
                expected,
                observed: snapshot.next,
            });
        }

        Ok(Self {
            next: snapshot.next,
        })
    }

    pub fn snapshot(&self) -> AgentIdAllocatorSnapshotV1 {
        AgentIdAllocatorSnapshotV1 { next: self.next }
    }

    /// Allocate a fresh id.
    ///
    /// Deliberately *not* named `next`: this is an infallible monotonic allocator, not an
    /// [`Iterator`]. It returns `AgentId`, never `Option<AgentId>`, and cannot silently wrap or
    /// emit [`AgentId::UNALLOCATED`]. Exhausting the finite 64-bit identity space is treated as a
    /// fatal invariant boundary and panics without mutating allocator state.
    pub fn allocate(&mut self) -> AgentId {
        assert_ne!(
            self.next,
            AgentId::UNALLOCATED.0,
            "AgentIdAllocator exhausted: refusing to emit the reserved UNALLOCATED sentinel or reuse an identity"
        );
        let id = AgentId(self.next);
        // The assertion above proves `self.next < u64::MAX`, so this increment cannot overflow.
        self.next += 1;
        id
    }
}

impl Default for AgentIdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::analyze_lifecycle_events;
    use crate::{
        Genome, GenomeEvidenceV1, LifecycleEventV1, LifecycleTransitionV1, OrganismConfig,
    };

    fn founder_event(sequence: u64, raw_id: u64) -> LifecycleEventV1 {
        let id = AgentId(raw_id);
        LifecycleEventV1 {
            sequence,
            tick: 0,
            transition: LifecycleTransitionV1::Founder {
                agent_id: id,
                lineage_id: id,
                generation: 0,
                genome: GenomeEvidenceV1::from_genome(Genome::from_config(
                    &OrganismConfig::default(),
                )),
                initial_energy_bits: 0.8f64.to_bits(),
            },
        }
    }

    #[test]
    fn allocator_never_repeats_an_id() {
        let mut alloc = AgentIdAllocator::new();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..10_000 {
            let id = alloc.allocate();
            assert!(
                seen.insert(id),
                "AgentIdAllocator emitted a duplicate id: {id:?}"
            );
        }
    }

    #[test]
    fn unallocated_sentinel_never_collides_with_a_real_allocation() {
        let mut alloc = AgentIdAllocator::new();
        for _ in 0..10_000 {
            assert_ne!(alloc.allocate(), AgentId::UNALLOCATED);
        }
    }

    #[test]
    fn exhaustion_fails_closed_without_emitting_sentinel_or_wrapping() {
        let mut alloc = AgentIdAllocator {
            next: u64::MAX - 1,
        };

        let last = alloc.allocate();
        assert_eq!(last.raw(), u64::MAX - 1);
        assert_eq!(alloc.next, u64::MAX);
        assert_ne!(last, AgentId::UNALLOCATED);

        let before = alloc.next;
        let exhausted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = alloc.allocate();
        }));
        assert!(exhausted.is_err(), "exhausted allocator must fail closed");
        assert_eq!(
            alloc.next, before,
            "failed allocation must not wrap or consume additional identity state"
        );
    }

    #[test]
    fn lifecycle_bound_restore_preserves_the_exact_next_identity() {
        let lifecycle = analyze_lifecycle_events(&[founder_event(0, 0), founder_event(1, 1)])
            .expect("valid complete lifecycle");

        let mut original = AgentIdAllocator::new();
        assert_eq!(original.allocate().raw(), 0);
        assert_eq!(original.allocate().raw(), 1);
        let encoded = serde_json::to_string(&original.snapshot()).expect("serialize cursor");
        let decoded: AgentIdAllocatorSnapshotV1 =
            serde_json::from_str(&encoded).expect("deserialize cursor");

        let mut restored = AgentIdAllocator::from_snapshot_for_lifecycle(decoded, &lifecycle)
            .expect("lifecycle-bound allocator restore");
        assert_eq!(restored.allocate().raw(), 2);
    }

    #[test]
    fn rolled_back_cursor_cannot_reuse_an_existing_lifecycle_identity() {
        let lifecycle = analyze_lifecycle_events(&[founder_event(0, 0), founder_event(1, 1)])
            .expect("valid complete lifecycle");
        let forged = AgentIdAllocatorSnapshotV1 { next: 1 };

        assert_eq!(
            AgentIdAllocator::from_snapshot_for_lifecycle(forged, &lifecycle),
            Err(AgentIdAllocatorRestoreErrorV1::NextIdentityMismatch {
                expected: 2,
                observed: 1,
            })
        );
    }

    #[test]
    fn noncontiguous_complete_history_cannot_authorize_allocator_restore() {
        // The lifecycle validator intentionally reasons about lifecycle semantics, not allocator
        // provenance, so a standalone founder ID of 1 is otherwise a valid lifecycle. The restore
        // boundary adds the stronger allocator-specific theorem that production IDs are 0..N-1.
        let lifecycle = analyze_lifecycle_events(&[founder_event(0, 1)])
            .expect("lifecycle semantics alone allow arbitrary concrete identity");
        let snapshot = AgentIdAllocatorSnapshotV1 { next: 2 };

        assert_eq!(
            AgentIdAllocator::from_snapshot_for_lifecycle(snapshot, &lifecycle),
            Err(
                AgentIdAllocatorRestoreErrorV1::HistoricalIdentitySequenceMismatch {
                    expected: 0,
                    observed: 1,
                }
            )
        );
    }

    #[test]
    fn reserved_identity_in_history_cannot_authorize_allocator_restore() {
        let lifecycle = analyze_lifecycle_events(&[founder_event(0, u64::MAX)])
            .expect("lifecycle semantics alone do not own allocator sentinel policy");
        let snapshot = AgentIdAllocatorSnapshotV1 { next: u64::MAX };

        assert_eq!(
            AgentIdAllocator::from_snapshot_for_lifecycle(snapshot, &lifecycle),
            Err(AgentIdAllocatorRestoreErrorV1::HistoricalReservedIdentity)
        );
    }
}
