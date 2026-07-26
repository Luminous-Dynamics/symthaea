// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Full working memory backend wrapping Symthaea's `ContinuousMind`.
//!
//! Provides the same API as the lightweight backend but delegates to the
//! real cognitive pipeline: FIFO eviction, dream consolidation (merges
//! similar items > 0.8 similarity), social coherence, and the full
//! tick pipeline.
//!
//! Enabled via `--features symthaea-backend`.

use std::collections::{HashMap, HashSet};

use symthaea::mind::{ContinuousMind, MindConfig};
use symthaea_core::hdc::ContinuousHV;

/// Stable per-item identity within one `working_memory_ticks_slice()`
/// snapshot: (arrival tick, rank among items sharing that tick).
///
/// Arrival tick alone isn't guaranteed unique — `ContinuousMind::process_inputs`
/// (`src/mind/tick.rs:236`) drains its *entire* input queue within one
/// `tick()` call, stamping every item processed in that call with the same
/// `self.state.tick` value. This wrapper's own `perceive()` only ever queues
/// one item before ticking, so a collision can't arise from this type's own
/// call pattern today — but nothing here should rely on that staying true
/// (another concurrent enqueuer, e.g. `process_social`/`process_federated`,
/// could in principle add to the same queue before a tick fires). `rank`
/// disambiguates: eviction always removes from the front and insertion
/// always appends at the back — true both for plain FIFO eviction and for
/// dream consolidation's "keep the earlier tick" merge rule
/// (`src/mind/tick.rs:164-168`) — so relative order among same-tick
/// siblings is preserved by every mutation this type observes.
/// Recomputing rank fresh from the current tick slice therefore yields a
/// stable key for as long as an item survives, with no separate
/// bookkeeping needed.
type ItemKey = (u64, usize);

fn item_keys(ticks: &[u64]) -> Vec<ItemKey> {
    let mut seen: HashMap<u64, usize> = HashMap::new();
    ticks
        .iter()
        .map(|&t| {
            let rank = seen.entry(t).or_insert(0);
            let key = (t, *rank);
            *rank += 1;
            key
        })
        .collect()
}

/// Default activation decay rate per tick.
const DEFAULT_DECAY_RATE: f32 = 0.95;

/// Configuration for the full working memory backend.
#[derive(Debug, Clone)]
pub struct WmConfig {
    /// HDC dimension.
    pub dimension: usize,
    /// Maximum items in working memory.
    pub capacity: usize,
    /// Activation decay rate per tick (default 0.95).
    /// Used for `activation_weighted_similarity()` computation.
    pub decay_rate: f32,
}

impl Default for WmConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            capacity: 7,
            decay_rate: DEFAULT_DECAY_RATE,
        }
    }
}

/// Full working memory backed by Symthaea's `ContinuousMind`.
///
/// Delegates `perceive()` and `tick()` to the real cognitive pipeline,
/// which includes dream consolidation and optional social coherence.
/// Activation is computed from arrival ticks using exponential decay.
pub struct WorkingMemory {
    mind: ContinuousMind,
    decay_rate: f32,
    /// Per-item rehearsal-boost multiplier, keyed by `ItemKey` (arrival
    /// tick + rank among same-tick items) rather than position.
    ///
    /// A positional `VecDeque` was tried first and found broken: at-capacity
    /// FIFO eviction leaves `working_memory_ticks_slice().len()` unchanged
    /// (one item evicted, one inserted, same total length), so a
    /// length-only resync never fires and every boost value silently drifts
    /// onto the wrong item as soon as WM fills up — caught by
    /// `test_boost_survives_fifo_index_shift` before this was fixed. Keying
    /// by `ItemKey` ties a boost to the actual item rather than its
    /// container slot, so it's correct across both plain FIFO eviction and
    /// dream consolidation's merge (which keeps the earlier arrival tick as
    /// the surviving item's identity, so a boosted item's entry survives a
    /// merge for free — see `ItemKey`'s doc comment). Pruned via
    /// `prune_stale_boosts()` after every state-changing call; a missing
    /// entry means "never boosted," read as the neutral value 1.0.
    boost: HashMap<ItemKey, f32>,
}

impl WorkingMemory {
    /// Create a new working memory backed by `ContinuousMind`.
    pub fn new(config: WmConfig) -> Self {
        let mind_config = MindConfig {
            dimension: config.dimension,
            working_memory_capacity: config.capacity,
            ..Default::default()
        };
        let mut mind = ContinuousMind::new(mind_config);
        mind.activate();
        Self {
            mind,
            decay_rate: config.decay_rate,
            boost: HashMap::new(),
        }
    }

    /// Drop any `boost` entry whose `ItemKey` is no longer present in
    /// current working memory — the item it belonged to was evicted or
    /// consolidated away.
    fn prune_stale_boosts(&mut self) {
        let ticks = self.mind.working_memory_ticks_slice();
        let live: HashSet<ItemKey> = item_keys(ticks).into_iter().collect();
        self.boost.retain(|k, _| live.contains(k));
    }

    /// Add an item to working memory via `ContinuousMind::perceive()`.
    ///
    /// Auto-ticks after perceive so items are immediately available in
    /// `contents()`, matching the lightweight backend's API contract.
    pub fn perceive(&mut self, hv: ContinuousHV) {
        self.mind.perceive(hv);
        let _ = self.mind.tick();
        self.prune_stale_boosts();
    }

    /// Advance one tick via `ContinuousMind::tick()`.
    ///
    /// This runs the full cognitive pipeline including:
    /// - Input processing with FIFO eviction
    /// - Dream consolidation (merges similar items > 0.8 similarity)
    /// - Social coherence processing (if enabled)
    pub fn tick(&mut self) {
        let _ = self.mind.tick();
        self.prune_stale_boosts();
    }

    /// Get current working memory contents.
    pub fn contents(&self) -> &[ContinuousHV] {
        self.mind.working_memory()
    }

    /// Get activation levels for each item, computed from arrival ticks
    /// and any accumulated rehearsal boost.
    ///
    /// `activation = decay_rate^(current_tick - arrival_tick) * boost`
    pub fn activations(&self) -> Vec<f32> {
        let current = self.mind.state().tick;
        let ticks = self.mind.working_memory_ticks_slice();
        let keys = item_keys(ticks);
        ticks
            .iter()
            .zip(keys.iter())
            .map(|(&arrival, key)| {
                let boost = self.boost.get(key).copied().unwrap_or(1.0);
                let age = current.saturating_sub(arrival) as f32;
                self.decay_rate.powf(age) * boost
            })
            .collect()
    }

    /// Boost activation of the item at `index` by a multiplicative factor.
    ///
    /// Models covert rehearsal: items rehearsed more frequently maintain
    /// higher activation (Rundus, 1971). Factor > 1.0 boosts, < 1.0
    /// attenuates. Boost is floored at 0.0 but not upper-clamped, allowing
    /// rehearsal to build supra-threshold traces that survive subsequent
    /// decay. Mirrors `wm::lightweight::WorkingMemory::boost_activation`.
    pub fn boost_activation(&mut self, index: usize, factor: f32) {
        self.prune_stale_boosts();
        let ticks = self.mind.working_memory_ticks_slice();
        let Some(&key) = item_keys(ticks).get(index) else {
            return;
        };
        let b = self.boost.entry(key).or_insert(1.0);
        *b = (*b * factor).max(0.0);
    }

    /// Compute the maximum activation-weighted similarity to `probe`.
    pub fn activation_weighted_similarity(&self, probe: &ContinuousHV) -> f32 {
        let activations = self.activations();
        self.mind
            .working_memory()
            .iter()
            .zip(&activations)
            .map(|(item, &act)| item.similarity(probe) * act)
            .fold(0.0f32, f32::max)
    }

    /// Current tick count.
    pub fn current_tick(&self) -> u64 {
        self.mind.state().tick
    }

    /// Drain evicted items since last call.
    pub fn take_evicted(&mut self) -> Vec<(ContinuousHV, u64)> {
        self.mind
            .take_evicted()
            .into_iter()
            .map(|(hv, tick, _source, _consolidated)| (hv, tick))
            .collect()
    }

    /// Number of items currently in WM.
    pub fn len(&self) -> usize {
        self.mind.working_memory().len()
    }

    /// Whether WM is empty.
    pub fn is_empty(&self) -> bool {
        self.mind.working_memory().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a WM with decay disabled (`decay_rate: 1.0`) so
    /// `activations()[i] == boost[i]` exactly — lets boost tests assert
    /// against the public API instead of reaching into the private
    /// `HashMap`, and removes decay arithmetic as a confound.
    fn no_decay_wm(dimension: usize, capacity: usize) -> WorkingMemory {
        WorkingMemory::new(WmConfig {
            dimension,
            capacity,
            decay_rate: 1.0,
        })
    }

    #[test]
    fn test_full_wm_basic() {
        let mut wm = WorkingMemory::new(WmConfig {
            dimension: 64,
            capacity: 3,
            ..Default::default()
        });

        for i in 0..5 {
            wm.perceive(ContinuousHV::random(64, i + 1));
            wm.tick();
        }

        assert!(wm.len() <= 3);
    }

    // ── ItemKey scheme (pure, no ContinuousMind needed) ─────────────────

    #[test]
    fn test_item_keys_disambiguates_duplicate_ticks_by_rank() {
        let ticks = vec![5u64, 5u64, 5u64, 6u64];
        assert_eq!(item_keys(&ticks), vec![(5, 0), (5, 1), (5, 2), (6, 0)]);
    }

    /// The invariant dream consolidation's merge relies on: it always keeps
    /// the *earlier* arrival tick as the surviving item's identity
    /// (`src/mind/tick.rs:164-168`) when collapsing two adjacent items. A
    /// boost keyed on an earlier, still-present tick must resolve
    /// identically whether or not a later item disappears from the slice —
    /// this is what lets a boosted item's entry survive a consolidation
    /// merge without `WorkingMemory` needing to know a merge happened at
    /// all.
    #[test]
    fn test_item_keys_stable_for_surviving_earlier_tick_after_a_later_removal() {
        let before = vec![10u64, 20u64, 30u64];
        let key_of_first = item_keys(&before)[0];

        let after_later_removed = vec![10u64, 30u64]; // the 20 is gone
        let key_of_first_after = item_keys(&after_later_removed)[0];

        assert_eq!(key_of_first, key_of_first_after);
        assert_eq!(key_of_first, (10, 0));
    }

    // ── boost_activation / activations() through the public API ────────

    #[test]
    fn test_unboosted_items_default_to_neutral_activation() {
        let mut wm = no_decay_wm(64, 3);
        for seed in 1..=3 {
            wm.perceive(ContinuousHV::random(64, seed));
        }
        assert!(
            wm.boost.is_empty(),
            "no boost_activation call yet — map should be empty"
        );
        assert_eq!(wm.activations(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_boost_isolated_to_targeted_item_only() {
        let mut wm = no_decay_wm(512, 3);
        for seed in 1..=3 {
            wm.perceive(ContinuousHV::random(512, seed));
        }
        wm.boost_activation(1, 9.0);
        let acts = wm.activations();
        assert_eq!(acts[0], 1.0, "un-boosted neighbor must be unaffected");
        assert_eq!(acts[1], 9.0);
        assert_eq!(acts[2], 1.0, "un-boosted neighbor must be unaffected");
    }

    /// The bug this whole test module was written to catch: boosting an
    /// item, then evicting an older item ahead of it via plain capacity
    /// overflow, must carry the boost along to the shifted index. A prior
    /// positional `VecDeque` implementation failed this exact test —
    /// at-capacity FIFO eviction leaves WM length unchanged, so its
    /// length-only resync never fired and the boost stayed attached to the
    /// wrong item. Uses dimension 512 (cosine similarity between
    /// independent random vectors is negligible at that width) and drives
    /// all state changes through `perceive()`, whose internal auto-tick
    /// always has a non-empty input queue — per `ItemKey`'s doc comment,
    /// that keeps `ContinuousMind` out of its dream/consolidation branch
    /// regardless of wall-clock time, so this test exercises plain FIFO
    /// eviction only, deterministically.
    #[test]
    fn test_boost_survives_fifo_index_shift() {
        let mut wm = no_decay_wm(512, 3);
        for seed in 1..=3 {
            wm.perceive(ContinuousHV::random(512, seed));
        }
        assert_eq!(wm.len(), 3);

        wm.boost_activation(1, 3.0);
        assert_eq!(wm.activations(), vec![1.0, 3.0, 1.0]);

        // A 4th perceive overflows capacity 3, evicting the oldest (index 0)
        // via plain FIFO — every surviving item, including the boosted one,
        // shifts left by one.
        wm.perceive(ContinuousHV::random(512, 4));
        assert_eq!(wm.len(), 3, "capacity should still be respected");
        assert_eq!(
            wm.activations(),
            vec![3.0, 1.0, 1.0],
            "the boosted item shifted from index 1 to index 0 and must keep its boost"
        );
    }

    /// Several evictions in a row must keep shifting the tracked boost with
    /// its item, not just survive a single eviction.
    #[test]
    fn test_boost_survives_multiple_successive_evictions() {
        let mut wm = no_decay_wm(512, 3);
        for seed in 1..=3 {
            wm.perceive(ContinuousHV::random(512, seed));
        }
        // Boost the newest of the initial three (index 2).
        wm.boost_activation(2, 2.0);

        // Two more perceives evict the two items ahead of it in turn.
        wm.perceive(ContinuousHV::random(512, 4));
        wm.perceive(ContinuousHV::random(512, 5));

        assert_eq!(wm.len(), 3);
        assert_eq!(
            wm.activations(),
            vec![2.0, 1.0, 1.0],
            "boost must survive two successive evictions, tracking its item to index 0"
        );
    }

    /// Boosting the item that's about to be evicted must not leave a stale
    /// entry behind once it's actually gone — `prune_stale_boosts()` is the
    /// mechanism responsible, and a leaked entry would be a slow memory
    /// leak over a long-running WM.
    #[test]
    fn test_evicted_items_boost_entry_is_removed() {
        let mut wm = no_decay_wm(512, 3);
        for seed in 1..=3 {
            wm.perceive(ContinuousHV::random(512, seed));
        }
        wm.boost_activation(0, 5.0); // boost the oldest, soon to be evicted
        assert_eq!(wm.boost.len(), 1);

        wm.perceive(ContinuousHV::random(512, 4)); // evicts it
        assert_eq!(
            wm.boost.len(),
            0,
            "the evicted item's boost entry must be pruned, not linger"
        );
        assert_eq!(wm.activations(), vec![1.0, 1.0, 1.0]);
    }

    /// `boost_activation` multiplies the existing boost rather than
    /// overwriting it, so repeated rehearsal compounds (Rundus 1971,
    /// referenced in the method's own doc comment) — verify it isn't
    /// silently reset on each call.
    #[test]
    fn test_boost_activation_compounds_multiplicatively() {
        let mut wm = no_decay_wm(64, 3);
        wm.perceive(ContinuousHV::random(64, 1));
        wm.boost_activation(0, 1.5);
        wm.boost_activation(0, 1.5);
        let acts = wm.activations();
        assert!(
            (acts[0] - 2.25).abs() < 1e-6,
            "expected 1.0 * 1.5 * 1.5 = 2.25, got {}",
            acts[0]
        );
    }

    /// The method's own doc comment says boost is "floored at 0.0 but not
    /// upper-clamped" — verify the floor actually holds for a
    /// negative/attenuating factor, rather than going negative.
    #[test]
    fn test_boost_activation_floored_at_zero() {
        let mut wm = no_decay_wm(64, 3);
        wm.perceive(ContinuousHV::random(64, 1));
        wm.boost_activation(0, -5.0);
        assert_eq!(
            wm.activations()[0],
            0.0,
            "boost must floor at 0.0, not go negative"
        );
    }

    /// An out-of-range index must be a no-op, not a panic.
    #[test]
    fn test_boost_activation_out_of_range_index_is_noop() {
        let mut wm = no_decay_wm(64, 3);
        wm.perceive(ContinuousHV::random(64, 1));
        let before = wm.boost.clone();
        wm.boost_activation(99, 3.0);
        assert_eq!(
            wm.boost, before,
            "out-of-range boost_activation must not mutate state"
        );
    }

    #[test]
    fn test_full_wm_activation() {
        let mut wm = WorkingMemory::new(WmConfig {
            dimension: 64,
            capacity: 7,
            ..Default::default()
        });

        wm.perceive(ContinuousHV::random(64, 1));
        // ContinuousMind processes input during tick, so after 1 tick
        // the item is in WM. After several more ticks, activation decays.
        for _ in 0..5 {
            wm.tick();
        }

        let activations = wm.activations();
        if !activations.is_empty() {
            // After multiple ticks, activation should be below 1.0
            assert!(
                activations[0] < 1.0,
                "activation should decay: got {}",
                activations[0]
            );
            assert!(activations[0] > 0.0);
        }
    }
}
