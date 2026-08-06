// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Non-spatial encounter scheduling, per `ALIFE_MULTIAGENT_GENESIS_PLAN_2026-07-25.md` (G0b).
//!
//! Genesis v0 deliberately has no spatial substrate — the encounter scheduler *is* the
//! environment for who-interacts-with-whom: each tick it decides pairings using a
//! seed-deterministic rule, with no notion of position, movement, or proximity. Real space
//! (positions, movement, `approach`/`avoid` actions) is explicitly deferred to a later
//! milestone (see the plan doc's "Explicitly deferred to v1+") so that identity, memory, and
//! transfer can be evaluated as their own capability first.

use std::collections::{HashMap, HashSet};

use crate::agent_id::AgentId;

/// How partners are chosen each tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairingMode {
    /// Fresh random pairing every tick — no persistent partner identity across ticks. The
    /// Genesis plan's anonymous / one-shot-like condition.
    Random,
    /// Partners are assigned once (lazily, as agents first appear) and reused every tick — the
    /// "repeated dependence" condition. If an agent's assigned partner dies, it goes unpaired
    /// that one tick and becomes eligible for rematching from then on (Genesis v0.1 audit Gate
    /// 2, 2026-07-26: an earlier version of this scheduler left a bereaved agent permanently
    /// unpaired for the rest of the run, which confounded a since-retracted exploratory finding
    /// — see `ALIFE_MULTIAGENT_GENESIS_PLAN_2026-07-25.md`). Newborns are assigned a partner
    /// from the pool of other currently-unassigned-or-widowed agents the first tick they appear.
    FixedPartners,
}

/// Decides which agents interact each tick. Deterministic given `(mode, seed)` and the exact
/// sequence of `living` slices passed to [`EncounterScheduler::pair`] — the Stage 0 "pairing
/// determinism" invariant.
pub struct EncounterScheduler {
    mode: PairingMode,
    fixed_partners: HashMap<AgentId, AgentId>,
    rng_state: u64,
}

impl EncounterScheduler {
    pub fn new(mode: PairingMode, seed: u64) -> Self {
        Self {
            mode,
            fixed_partners: HashMap::new(),
            rng_state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_unit(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// A deterministic in-place Fisher-Yates shuffle using this scheduler's own RNG stream.
    fn shuffle(&mut self, items: &mut [AgentId]) {
        for i in (1..items.len()).rev() {
            let j = ((self.next_unit() * (i + 1) as f64) as usize).min(i);
            items.swap(i, j);
        }
    }

    /// Compute this tick's pairing given the currently-living agent ids (population order at
    /// the start of the tick). Returns each pair exactly once, `a < b` by [`AgentId`]'s `Ord` so
    /// callers never see both `(a, b)` and `(b, a)`. An odd-sized `living` slice (or a fixed
    /// partner no longer present) leaves exactly one agent unpaired that tick.
    pub fn pair(&mut self, living: &[AgentId]) -> Vec<(AgentId, AgentId)> {
        match self.mode {
            PairingMode::Random => self.pair_random(living),
            PairingMode::FixedPartners => self.pair_fixed(living),
        }
    }

    fn pair_random(&mut self, living: &[AgentId]) -> Vec<(AgentId, AgentId)> {
        let mut pool: Vec<AgentId> = living.to_vec();
        self.shuffle(&mut pool);
        pool.chunks_exact(2)
            .map(|c| {
                if c[0] < c[1] {
                    (c[0], c[1])
                } else {
                    (c[1], c[0])
                }
            })
            .collect()
    }

    fn pair_fixed(&mut self, living: &[AgentId]) -> Vec<(AgentId, AgentId)> {
        let living_set: HashSet<AgentId> = living.iter().copied().collect();

        // Lazily assign fixed partners to any currently-unassigned OR newly-widowed living agent
        // (covers the very first tick, newborns appearing later, AND rematching after a
        // partner's death), pairing them off deterministically. An agent is eligible for
        // (re)matching if it has never been assigned a partner, or its assigned partner is no
        // longer alive -- checking only `contains_key` here would leave a bereaved agent
        // permanently orphaned (its stale key never becomes absent), which is exactly the bug
        // Gate 2 of the Genesis v0.1 audit found and fixed.
        let mut unassigned: Vec<AgentId> = living
            .iter()
            .copied()
            .filter(|id| match self.fixed_partners.get(id) {
                None => true,
                Some(partner) => !living_set.contains(partner),
            })
            .collect();
        self.shuffle(&mut unassigned);
        for pair in unassigned.chunks_exact(2) {
            self.fixed_partners.insert(pair[0], pair[1]);
            self.fixed_partners.insert(pair[1], pair[0]);
        }

        living
            .iter()
            .filter_map(|&id| {
                self.fixed_partners.get(&id).copied().and_then(|partner| {
                    if living_set.contains(&partner) && id < partner {
                        Some((id, partner))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: u64) -> Vec<AgentId> {
        let mut alloc = crate::agent_id::AgentIdAllocator::new();
        (0..n).map(|_| alloc.allocate()).collect()
    }

    #[test]
    fn random_pairing_covers_every_agent_at_most_once_and_leaves_at_most_one_unpaired() {
        let living = ids(17); // odd on purpose
        let mut sched = EncounterScheduler::new(PairingMode::Random, 42);
        let pairs = sched.pair(&living);
        let mut seen = HashSet::new();
        for (a, b) in &pairs {
            assert!(seen.insert(*a));
            assert!(seen.insert(*b));
        }
        assert!(living.len() - seen.len() <= 1);
    }

    #[test]
    fn same_seed_and_inputs_reproduce_the_exact_same_pairing() {
        let living = ids(20);
        let mut a = EncounterScheduler::new(PairingMode::Random, 123);
        let mut b = EncounterScheduler::new(PairingMode::Random, 123);
        for _ in 0..10 {
            assert_eq!(a.pair(&living), b.pair(&living));
        }
    }

    #[test]
    fn different_seeds_produce_different_pairings() {
        let living = ids(20);
        let mut a = EncounterScheduler::new(PairingMode::Random, 1);
        let mut b = EncounterScheduler::new(PairingMode::Random, 2);
        assert_ne!(a.pair(&living), b.pair(&living));
    }

    #[test]
    fn fixed_partners_mode_repeats_the_same_pair_across_many_ticks() {
        let living = ids(10);
        let mut sched = EncounterScheduler::new(PairingMode::FixedPartners, 7);
        let first = sched.pair(&living);
        for _ in 0..50 {
            assert_eq!(
                sched.pair(&living),
                first,
                "FixedPartners must repeat the same pairing"
            );
        }
    }

    #[test]
    fn fixed_partners_assigns_newborns_without_disturbing_existing_pairs() {
        let mut sched = EncounterScheduler::new(PairingMode::FixedPartners, 99);
        let original = ids(6);
        let first = sched.pair(&original);

        let mut alloc = crate::agent_id::AgentIdAllocator::new();
        // Re-derive ids identically then add two more "newborns" with fresh ids beyond the
        // original set (simulating population growth between ticks).
        let mut with_newborns = original.clone();
        for _ in 0..8 {
            alloc.allocate(); // burn ids already used by `original` in this test's own allocator
        }
        with_newborns.push(alloc.allocate());
        with_newborns.push(alloc.allocate());

        let second = sched.pair(&with_newborns);
        for pair in &first {
            assert!(
                second.contains(pair),
                "existing fixed pairing {pair:?} must survive a newborn appearing"
            );
        }
        assert_eq!(
            second.len(),
            first.len() + 1,
            "the two newborns should pair with each other"
        );
    }

    #[test]
    fn fixed_partners_leaves_agent_unpaired_the_tick_its_partner_dies() {
        let living = ids(4);
        let mut sched = EncounterScheduler::new(PairingMode::FixedPartners, 5);
        let first = sched.pair(&living);
        assert_eq!(first.len(), 2);

        // Remove one agent (simulating a death) and re-pair, same tick.
        let survivors: Vec<AgentId> = living[..3].to_vec();
        let second = sched.pair(&survivors);
        // Exactly one pair should survive (whichever didn't include the removed agent), and
        // the third survivor is unpaired that tick -- not silently re-paired with a stranger
        // out of the blue on the very tick its partner disappears.
        assert_eq!(second.len(), 1);
    }

    #[test]
    fn fixed_partners_rematches_a_bereaved_agent_instead_of_orphaning_it_forever() {
        // Genesis v0.1 audit Gate 2: an earlier version of this scheduler left a bereaved agent
        // permanently unpaired for the rest of the run (its stale key in `fixed_partners` never
        // became absent, so it never reappeared in the rematch pool). Two independent original
        // pairs, each loses one member; the two survivors must find each other on a later tick.
        let living = ids(4);
        let mut sched = EncounterScheduler::new(PairingMode::FixedPartners, 11);
        let first = sched.pair(&living);
        assert_eq!(first.len(), 2, "4 agents should form exactly 2 fixed pairs");

        // Keep exactly one member of each original pair (whichever the RNG actually produced --
        // `first`'s pairing is deterministic but not hardcoded here), killing the other member of
        // each pair. Both survivors are then genuinely bereaved, regardless of which specific ids
        // the scheduler happened to pair on the first tick.
        let survivors: Vec<AgentId> = first.iter().map(|&(a, _b)| a).collect();

        // Both survivors become simultaneously eligible for rematching the moment their old
        // partners are gone -- since exactly 2 agents are eligible at once here, they pair with
        // each other immediately (no artificial delay). The property this test actually locks in
        // is that this rematch happens *at all* and *persists*, not a specific tick count before
        // it does (the old bug was "never," not "one tick late").
        let rematched = sched.pair(&survivors);
        assert_eq!(
            rematched.len(),
            1,
            "the two bereaved survivors must be rematched to each other, not left unpaired forever"
        );
        let (a, b) = rematched[0];
        let expected = if survivors[0] < survivors[1] {
            (survivors[0], survivors[1])
        } else {
            (survivors[1], survivors[0])
        };
        assert_eq!((a, b), expected);

        // And it must actually persist as a real fixed pairing across further ticks.
        for _ in 0..20 {
            assert_eq!(sched.pair(&survivors), rematched);
        }
    }
}
