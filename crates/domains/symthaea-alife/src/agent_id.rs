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

/// Monotonic allocator — every call to [`AgentIdAllocator::allocate`] returns a fresh id, never
/// reused, so a later death/birth can never cause one individual's id to be handed to another.
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
    /// [`Iterator`]. It returns `AgentId`, never `Option<AgentId>`, and has no end — reading it
    /// as `Iterator::next` would wrongly suggest it can be exhausted or driven by `for`/adapters.
    pub fn allocate(&mut self) -> AgentId {
        let id = AgentId(self.next);
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
}
