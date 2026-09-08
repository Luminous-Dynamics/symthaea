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

/// Monotonic allocator — every successful call to [`AgentIdAllocator::allocate`] returns a fresh
/// id, never reused, so a later death/birth can never cause one individual's id to be handed to
/// another.
///
/// `u64::MAX` is permanently reserved for [`AgentId::UNALLOCATED`]. Once every allocatable value
/// `0..u64::MAX` has been consumed, allocation fails closed by panicking before changing allocator
/// state. This boundary is astronomically remote in practice, but making it explicit prevents a
/// release-build integer wrap from violating the crate's identity theorem.
#[derive(Debug, Clone)]
pub struct AgentIdAllocator {
    next: u64,
}

impl AgentIdAllocator {
    pub fn new() -> Self {
        Self { next: 0 }
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
}
