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

/// Per-item identity within one `working_memory_ticks_slice()` snapshot:
/// (arrival tick, rank among items sharing that tick).
///
/// Arrival tick alone isn't guaranteed unique — `ContinuousMind::process_inputs`
/// (`src/mind/tick.rs:236`) drains its *entire* input queue within one
/// `tick()` call, stamping every item processed in that call with the same
/// `self.state.tick` value. This wrapper's own `perceive()` only ever queues
/// one item before ticking, so a collision can't arise from this type's own
/// call pattern today — but nothing here should rely on that staying true
/// (another concurrent enqueuer, e.g. `process_social`/`process_federated`,
/// could in principle add to the same queue before a tick fires).
///
/// **Known gap, mitigated but not fixed: `rank` is NOT a stable identity
/// across a mutation for duplicate-tick items — it's an occurrence index,
/// recomputed fresh from whatever the current slice happens to contain.**
/// If item A (tick 10, rank 0) is evicted from in front of item B (tick 10,
/// rank 1), recomputing keys on the post-eviction slice gives B rank 0 —
/// its key *changes*, even though B itself didn't move except in this
/// derived numbering. Any `boost` entry still stored under B's old key
/// `(10, 1)` isn't found live-or-not by identity; it's found by whether
/// `(10, 1)` happens to still be *some* live key (e.g. a surviving item C's
/// new key) after eviction — so B's boost could silently misattribute to
/// whichever item recomputation happens to assign that same key to next.
/// See `test_duplicate_tick_rank_reassignment_can_misattribute_boost` for a
/// concrete demonstration of the key-computation property itself.
///
/// **Mitigation**: `reconcile_boosts_after_state_change` fails closed —
/// whenever either tick snapshot it's given contains a duplicate tick, it
/// clears the whole `boost` map rather than attempting any cross-item
/// transfer, since identity can't be trusted in that state. This turns the
/// silent-misattribution risk above into a conservative "lose the boosts,
/// don't miscredit them" outcome. See
/// `test_duplicate_tick_identity_ambiguity_fails_closed`.
///
/// Not fixed at the root here: a real fix needs a per-item identity that
/// survives eviction/consolidation independent of position or arrival tick
/// (e.g. a monotonic item ID from `ContinuousMind` itself), which doesn't
/// exist in its public API today and would be a `symthaea` (main crate)
/// change, out of this crate's scope. In practice this gap requires
/// duplicate ticks to actually occur, which — per the collision note above —
/// cannot happen through this wrapper's own `perceive()`/`tick()` calls
/// alone; the fail-closed guard exists for defense in depth against that
/// invariant being violated by a future caller or an unforeseen
/// `ContinuousMind` change, not because it's expected to fire in practice.
type ItemKey = (u64, usize);

/// Whether `ticks` contains any value more than once. Used to detect the
/// precondition under which `ItemKey`'s `(arrival_tick, rank)` scheme loses
/// stable identity (see `ItemKey`'s doc comment) — a duplicate anywhere in
/// either the before- or after-snapshot means rank recomputation could
/// reassign a surviving item's key, so boost tracking should not trust any
/// key correspondence in that state.
fn has_duplicate_ticks(ticks: &[u64]) -> bool {
    let mut seen = HashSet::new();
    ticks.iter().any(|tick| !seen.insert(*tick))
}

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

/// Detect a dream-consolidation merge between two tick-slice snapshots
/// taken immediately before and after one state-changing call. Returns
/// `Some((removed_tick, survivor_tick))` **only** when removing exactly one
/// element from `before` reproduces `after` *exactly* — i.e. some tick
/// vanished while every other tick in the sequence, before and after that
/// point, survived completely unchanged. This is deliberately a strict
/// verify-the-whole-remainder check rather than "stop at the first
/// mismatch and assume the rest matches": an earlier version did the
/// latter and could misidentify a merge when the input actually reflected
/// more than one change (e.g. a merge plus an unrelated insertion in the
/// same window) — `before=[10,20,30]`/`after=[10,40]` isn't a single
/// removal of `20` (that would leave `[10,30]`, not `[10,40]`), and this
/// function now correctly returns `None` for it rather than a false
/// positive. Only a genuine single-element interior removal — the exact
/// shape of `process_dream`'s merge (`src/mind/tick.rs:155-212`, which
/// bundles two adjacent items and removes the later one's slot, keeping
/// the earlier one as the surviving identity) — returns `Some`. Plain
/// front-eviction (`after == before[1..]`) returns `None`, as does anything
/// that isn't explained by a single interior removal (batched changes,
/// growth).
///
/// Pure and independent of `ContinuousMind` — testable directly against
/// hand-constructed tick sequences without needing to trigger a real dream
/// phase (which requires actual Night circadian phase + idle load + an
/// empty input queue, not reliably reachable from this crate's tests).
fn detect_consolidation_merge(before: &[u64], after: &[u64]) -> Option<(u64, u64)> {
    if before.len() != after.len() + 1 {
        return None;
    }
    if after == &before[1..] {
        return None; // plain front-eviction, not a merge
    }
    for bi in 1..before.len() {
        let matches_before_removal = before[..bi] == after[..bi] && before[bi + 1..] == after[bi..];
        if matches_before_removal {
            return Some((before[bi], before[bi - 1]));
        }
    }
    None
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
    /// container slot, so it's correct across plain FIFO eviction.
    ///
    /// Dream consolidation's merge is handled explicitly, not just "for
    /// free": when `detect_consolidation_merge` (called from
    /// `reconcile_boosts_after_state_change`) finds that two items bundled
    /// into one, the surviving key's boost becomes the **maximum** of the
    /// two merged items' boosts (not their sum, which could let repeated
    /// consolidation events inflate a boost without bound, and not just the
    /// survivor's own prior value, which would silently discard rehearsal
    /// evidence the removed item carried). See
    /// `test_consolidation_merge_takes_max_boost`.
    ///
    /// Reconciled via `reconcile_boosts_after_state_change()` after every
    /// state-changing call; a missing entry means "never boosted," read as
    /// the neutral value 1.0.
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

    /// Reconcile `boost` with working memory's current contents after a
    /// state-changing call, given the tick slice as it stood immediately
    /// before that call.
    ///
    /// If `detect_consolidation_merge` finds that this call bundled two
    /// items into one (dream consolidation), the surviving key's boost
    /// becomes `max(survivor_boost, removed_boost)` -- see the `boost`
    /// field doc comment for why max rather than sum or keep-survivor's-own.
    /// Any entry whose key isn't accounted for by a detected merge and is no
    /// longer live (plain eviction, or a merge this heuristic didn't
    /// recognize) is simply dropped.
    ///
    /// **Fails closed on duplicate-tick identity ambiguity**: if either
    /// snapshot contains a repeated tick, `ItemKey`'s `(arrival_tick, rank)`
    /// scheme cannot be trusted to identify the same item across the
    /// mutation (see `ItemKey`'s doc comment), so this clears the entire
    /// `boost` map rather than attempting any merge/eviction reconciliation
    /// that could silently misattribute rehearsal evidence to the wrong
    /// item. See `test_duplicate_tick_identity_ambiguity_fails_closed`.
    fn reconcile_boosts_after_state_change(&mut self, before_ticks: &[u64]) {
        let after_ticks = self.mind.working_memory_ticks_slice().to_vec();

        if has_duplicate_ticks(before_ticks) || has_duplicate_ticks(&after_ticks) {
            self.boost.clear();
            return;
        }

        if let Some((removed_tick, survivor_tick)) =
            detect_consolidation_merge(before_ticks, &after_ticks)
        {
            let removed_keys: Vec<ItemKey> = self
                .boost
                .keys()
                .copied()
                .filter(|&(t, _)| t == removed_tick)
                .collect();
            for removed_key in removed_keys {
                let removed_boost = self.boost.remove(&removed_key).unwrap_or(1.0);
                let survivor_key = (survivor_tick, removed_key.1);
                let survivor_boost = self.boost.get(&survivor_key).copied().unwrap_or(1.0);
                let merged = removed_boost.max(survivor_boost);
                if merged != 1.0 {
                    self.boost.insert(survivor_key, merged);
                } else {
                    self.boost.remove(&survivor_key);
                }
            }
        }

        let live: HashSet<ItemKey> = item_keys(&after_ticks).into_iter().collect();
        self.boost.retain(|k, _| live.contains(k));
    }

    /// Add an item to working memory via `ContinuousMind::perceive()`.
    ///
    /// Auto-ticks after perceive so items are immediately available in
    /// `contents()`, matching the lightweight backend's API contract.
    pub fn perceive(&mut self, hv: ContinuousHV) {
        let before_ticks = self.mind.working_memory_ticks_slice().to_vec();
        self.mind.perceive(hv);
        let _ = self.mind.tick();
        self.reconcile_boosts_after_state_change(&before_ticks);
    }

    /// Advance one tick via `ContinuousMind::tick()`.
    ///
    /// This runs the full cognitive pipeline including:
    /// - Input processing with FIFO eviction
    /// - Dream consolidation (merges similar items > 0.8 similarity)
    /// - Social coherence processing (if enabled)
    pub fn tick(&mut self) {
        let before_ticks = self.mind.working_memory_ticks_slice().to_vec();
        let _ = self.mind.tick();
        self.reconcile_boosts_after_state_change(&before_ticks);
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
        // Defensive: no mind-state change happens between the last
        // perceive()/tick() (which already reconciled) and this call, so
        // this should be a no-op in practice -- kept as a safety net, not
        // merge-aware since nothing could have merged since then.
        let ticks = self.mind.working_memory_ticks_slice().to_vec();
        let live: HashSet<ItemKey> = item_keys(&ticks).into_iter().collect();
        self.boost.retain(|k, _| live.contains(k));

        let Some(&key) = item_keys(&ticks).get(index) else {
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

    /// Concrete demonstration of the disclosed gap in `ItemKey`'s doc
    /// comment: `rank` is an occurrence index, not a stable identity, for
    /// duplicate-tick items. If the earliest same-tick item (rank 0) is
    /// removed, every surviving same-tick sibling's rank shifts down by
    /// one -- its key *changes* even though the item itself didn't move
    /// except in this derived numbering. A `boost` entry stored under the
    /// old key would silently resolve to whatever item now holds that key,
    /// not to the item it was originally about. Not reachable through this
    /// wrapper's own `perceive()`/`tick()` calls (which never produce
    /// duplicate ticks — see `ItemKey`'s doc comment), so this tests the
    /// pure key-computation property directly rather than fabricating an
    /// unreachable live scenario.
    #[test]
    fn test_duplicate_tick_rank_reassignment_can_misattribute_boost() {
        // Three same-tick items: A=(10,0), B=(10,1), C=(10,2).
        let before = item_keys(&[10, 10, 10]);
        assert_eq!(before, vec![(10, 0), (10, 1), (10, 2)]);
        let b_key_before = before[1]; // (10, 1)

        // A (rank 0) is removed -- only B and C survive.
        let after = item_keys(&[10, 10]);
        let b_key_after = after[0]; // B is now rank 0

        assert_ne!(
            b_key_before, b_key_after,
            "B's key changed purely because an earlier same-tick sibling \
             was removed -- this is the disclosed gap, not a false alarm"
        );
        assert_eq!(
            b_key_after,
            (10, 0),
            "B's new key collides with what would have been A's key"
        );
    }

    /// The enforced regression guard for the gap the previous test merely
    /// demonstrates: `reconcile_boosts_after_state_change` must fail closed
    /// (clear all boosts) rather than silently transfer any boost when
    /// either snapshot contains a duplicate tick, since key correspondence
    /// can't be trusted in that state. This is what actually protects
    /// production behavior -- the key-computation test above shows why the
    /// guard is necessary, this test shows the guard works.
    #[test]
    fn test_duplicate_tick_identity_ambiguity_fails_closed() {
        let mut wm = no_decay_wm(64, 3);
        wm.perceive(ContinuousHV::random(64, 1));
        let real_tick = wm.mind.working_memory_ticks_slice()[0];

        // Give the (only) real item a real boost.
        wm.boost.insert((real_tick, 0), 7.0);
        assert_eq!(wm.boost.len(), 1);

        // Feed a synthetic before-snapshot containing a duplicate tick --
        // this can't arise from this wrapper's own calls (see ItemKey's doc
        // comment) but exercises the guard directly, the same way the
        // existing consolidation-merge tests synthesize snapshots to
        // exercise that code path without needing a real dream phase.
        wm.reconcile_boosts_after_state_change(&[real_tick, real_tick, real_tick + 1]);

        assert!(
            wm.boost.is_empty(),
            "ambiguous duplicate-tick identity must clear all boosts, not \
             attempt a merge/eviction reconciliation that could misattribute \
             the real item's boost"
        );
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
    /// entry behind once it's actually gone —
    /// `reconcile_boosts_after_state_change()` is the mechanism responsible,
    /// and a leaked entry would be a slow memory leak over a long-running WM.
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

    // ── Dream consolidation merge detection (pure, no ContinuousMind) ───

    #[test]
    fn test_detect_consolidation_merge_none_on_plain_front_eviction() {
        // after == before[1..] -- the ordinary at-capacity FIFO case.
        assert_eq!(detect_consolidation_merge(&[10, 20, 30], &[20, 30]), None);
    }

    #[test]
    fn test_detect_consolidation_merge_none_on_growth_or_batched_change() {
        assert_eq!(detect_consolidation_merge(&[10, 20], &[10, 20, 30]), None);
        // Two items removed at once isn't a single merge this heuristic
        // claims to identify.
        assert_eq!(detect_consolidation_merge(&[10, 20, 30], &[30]), None);
    }

    #[test]
    fn test_detect_consolidation_merge_none_when_front_item_itself_vanishes_alone() {
        // The front tick disappearing with nothing surviving before it
        // isn't distinguishable from eviction by this heuristic, and has no
        // earlier survivor to merge into -- correctly returns None rather
        // than guessing.
        assert_eq!(detect_consolidation_merge(&[10], &[]), None);
    }

    #[test]
    fn test_detect_consolidation_merge_identifies_removed_and_survivor() {
        // [10, 20, 30] -> [10, 30]: 20 disappeared while 10 (earlier) and 30
        // (later) both survived unchanged -- exactly process_dream's shape
        // (bundle position i into i, remove i+1).
        assert_eq!(
            detect_consolidation_merge(&[10, 20, 30], &[10, 30]),
            Some((20, 10))
        );
    }

    #[test]
    fn test_detect_consolidation_merge_at_front_pair() {
        // The merged pair can be the first two items in the sequence.
        assert_eq!(
            detect_consolidation_merge(&[10, 20, 30, 40], &[10, 30, 40]),
            Some((20, 10))
        );
    }

    /// A single-mismatch "walk and stop" detector could misidentify this as
    /// "20 was removed" (since after[..1] == before[..1] and neither sees
    /// 20 next) without checking that the REST of the sequence also lines
    /// up -- but removing 20 from before would give [10, 30], not [10, 40].
    /// This isn't a single clean removal at all (30 also changed, which a
    /// real merge/eviction never does to a surviving tick), so the correct
    /// answer is `None`, not a false-positive merge.
    #[test]
    fn test_detect_consolidation_merge_rejects_false_positive_on_compound_change() {
        assert_eq!(detect_consolidation_merge(&[10, 20, 30], &[10, 40]), None);
    }

    // ── Consolidation merge applied to boost (max, not sum or drop) ─────

    /// Direct test of the merge rule via the real `WorkingMemory` API: a
    /// consolidation merge's surviving key must inherit
    /// `max(removed_boost, survivor_boost)`. Simulates the merge by handing
    /// `reconcile_boosts_after_state_change` a `before_ticks` snapshot with
    /// one synthetic extra (later) tick that isn't present in the real
    /// current state — real dream consolidation can't be triggered
    /// deterministically from this crate's public API (needs actual Night
    /// circadian phase + idle load + empty queue), but the reconciliation
    /// method itself only ever reads the *current* tick slice from the real
    /// `ContinuousMind` and takes the *before* slice as a plain parameter,
    /// so this exercises the real merge-application code path, not a mock.
    #[test]
    fn test_consolidation_merge_takes_max_boost_removed_higher() {
        let mut wm = no_decay_wm(64, 3);
        wm.perceive(ContinuousHV::random(64, 1));
        let survivor_tick = wm.mind.working_memory_ticks_slice()[0];

        // The survivor's own boost is low...
        wm.boost.insert((survivor_tick, 0), 1.5);
        // ...but the (synthetic) removed item's boost was higher.
        let removed_tick = survivor_tick + 1;
        wm.boost.insert((removed_tick, 0), 5.0);

        wm.reconcile_boosts_after_state_change(&[survivor_tick, removed_tick]);

        assert_eq!(
            wm.boost.get(&(survivor_tick, 0)),
            Some(&5.0),
            "survivor must inherit the higher (removed item's) boost"
        );
        assert!(
            !wm.boost.contains_key(&(removed_tick, 0)),
            "the removed item's own key must not linger"
        );
    }

    #[test]
    fn test_consolidation_merge_keeps_survivor_higher_boost() {
        let mut wm = no_decay_wm(64, 3);
        wm.perceive(ContinuousHV::random(64, 1));
        let survivor_tick = wm.mind.working_memory_ticks_slice()[0];

        wm.boost.insert((survivor_tick, 0), 4.0);
        let removed_tick = survivor_tick + 1;
        wm.boost.insert((removed_tick, 0), 1.2);

        wm.reconcile_boosts_after_state_change(&[survivor_tick, removed_tick]);

        assert_eq!(
            wm.boost.get(&(survivor_tick, 0)),
            Some(&4.0),
            "survivor must keep its own higher boost, not be pulled down"
        );
    }

    #[test]
    fn test_consolidation_merge_neither_boosted_leaves_no_entry() {
        // If neither merged item was ever boosted, the survivor shouldn't
        // gain a pointless neutral (1.0) entry.
        let mut wm = no_decay_wm(64, 3);
        wm.perceive(ContinuousHV::random(64, 1));
        let survivor_tick = wm.mind.working_memory_ticks_slice()[0];
        let removed_tick = survivor_tick + 1;

        wm.reconcile_boosts_after_state_change(&[survivor_tick, removed_tick]);

        assert!(wm.boost.is_empty());
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
