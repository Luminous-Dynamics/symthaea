// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001 — Partner-Conditioned Social Learning, per
//! `ALIFE_MA001_PARTNER_CONDITIONED_POLICY_PLAN_2026-07-26.md`.
//!
//! Deliberately separate from `population.rs`/`organism.rs`, per the plan's design boundary
//! (§2): this driver never calls `Population::step_social` (no experiment-specific branches in
//! production machinery). It owns its own tick loop, calling `Organism::act_social`/
//! `learn_from_realized_outcome` and `resolve_pair_transfer` directly, and manipulates the
//! public `Organism::ledger` field to implement the three conditions. Everything in this module
//! is MA-001-specific; nothing here is reused by, or changes the behavior of, Genesis's own
//! `Population`/`EncounterScheduler`.

use std::collections::HashMap;

use symthaea_fep::HiddenState;

use crate::agent_id::{AgentId, AgentIdAllocator};
use crate::ledger::{InteractionRecord, compress_for_observation};
use crate::ma001l::{DeltaRuleConfig, DeltaRuleLearner};
use crate::organism::{Organism, OrganismConfig};
use crate::population::resolve_pair_transfer;

/// Which observation an organism receives for its current-tick partner (plan §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    /// The real, correctly-keyed ledger record.
    Bound,
    /// The real record, but the key→value mapping is periodically permuted across an agent's
    /// own partners at `Ma001Config::shuffle_epoch_ticks` boundaries.
    Shuffled,
    /// Always `InteractionRecord::default()`, regardless of the real ledger (which keeps
    /// accumulating in the background for audit purposes only).
    NoHistory,
}

/// The group-of-4 circle-method round-robin schedule (plan §3): local indices 0..4 within a
/// group, `round = tick % 3`. Deterministic and mutual by construction — no RNG needed for
/// pairing at all.
pub fn round_robin_pairs(round: usize) -> [(usize, usize); 2] {
    match round % 3 {
        0 => [(0, 1), (2, 3)],
        1 => [(0, 2), (1, 3)],
        _ => [(0, 3), (1, 2)],
    }
}

/// Frozen run parameters, per the plan's §3/§5.
#[derive(Debug, Clone, Copy)]
pub struct Ma001Config {
    /// Number of 4-organism groups. Population = `groups * 4`.
    pub groups: usize,
    pub total_ticks: u64,
    pub burn_in_ticks: u64,
    pub shuffle_epoch_ticks: u64,
    pub transfer_quantum: f64,
    pub forage_efficiency: f64,
    /// Symmetric Dirichlet smoothing constant for the divergence metric.
    pub dirichlet_alpha: f64,
    /// Minimum encounters with a partner for that partner to count toward an agent's divergence
    /// score.
    pub min_encounters_per_partner: u32,
}

impl Default for Ma001Config {
    fn default() -> Self {
        Self {
            groups: 25,
            total_ticks: 1200,
            burn_in_ticks: 300,
            shuffle_epoch_ticks: 100,
            transfer_quantum: 0.05,
            forage_efficiency: 0.6,
            dirichlet_alpha: 0.5,
            min_encounters_per_partner: 50,
        }
    }
}

impl Ma001Config {
    pub fn population(&self) -> usize {
        self.groups * 4
    }

    fn organism_cfg(&self) -> OrganismConfig {
        OrganismConfig {
            social_enabled: true,
            transfer_quantum: self.transfer_quantum,
            forage_efficiency: self.forage_efficiency,
            ..OrganismConfig::default()
        }
    }
}

fn xorshift64_next_unit(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

/// Fisher-Yates shuffle of `items` using a plain xorshift64 stream — same algorithm this crate
/// already uses elsewhere (`encounter.rs`, `predator_prey.rs`), kept as an independent stream
/// here (never the same `state` as any `Organism`'s own RNG or any other MA-001 stream).
fn shuffle_in_place<T>(items: &mut [T], state: &mut u64) {
    for i in (1..items.len()).rev() {
        let j = ((xorshift64_next_unit(state) * (i + 1) as f64) as usize).min(i);
        items.swap(i, j);
    }
}

/// Symmetric Dirichlet-smoothed action-probability vector from raw per-action counts (plan §5).
pub fn dirichlet_smooth(counts: [u32; 3], alpha: f64) -> [f64; 3] {
    let total = counts.iter().sum::<u32>() as f64 + 3.0 * alpha;
    [
        (counts[0] as f64 + alpha) / total,
        (counts[1] as f64 + alpha) / total,
        (counts[2] as f64 + alpha) / total,
    ]
}

fn kl_divergence_base2(p: &[f64; 3], q: &[f64; 3]) -> f64 {
    p.iter()
        .zip(q.iter())
        .map(|(pi, qi)| {
            if *pi > 0.0 {
                pi * (pi / qi).log2()
            } else {
                0.0
            }
        })
        .sum()
}

/// Jensen-Shannon divergence (base-2 log, bounded `[0, 1]`) between two smoothed action-
/// probability distributions (plan §5).
pub fn jensen_shannon_divergence(p: &[f64; 3], q: &[f64; 3]) -> f64 {
    let m = [
        (p[0] + q[0]) / 2.0,
        (p[1] + q[1]) / 2.0,
        (p[2] + q[2]) / 2.0,
    ];
    0.5 * kl_divergence_base2(p, &m) + 0.5 * kl_divergence_base2(q, &m)
}

/// One agent's mean pairwise JS divergence across its eligible partners (plan §5) — `None` if
/// fewer than 2 partners clear `min_encounters_per_partner`.
pub fn agent_divergence_score(
    partner_counts: &HashMap<AgentId, [u32; 3]>,
    alpha: f64,
    min_encounters_per_partner: u32,
) -> Option<f64> {
    let eligible: Vec<[f64; 3]> = partner_counts
        .values()
        .filter(|c| c.iter().sum::<u32>() >= min_encounters_per_partner)
        .map(|c| dirichlet_smooth(*c, alpha))
        .collect();
    if eligible.len() < 2 {
        return None;
    }
    let mut total = 0.0;
    let mut pairs = 0u32;
    for i in 0..eligible.len() {
        for j in (i + 1)..eligible.len() {
            total += jensen_shannon_divergence(&eligible[i], &eligible[j]);
            pairs += 1;
        }
    }
    Some(total / pairs as f64)
}

/// One MA-001 run: a fixed population under one `Condition`, for one seed, optionally with a
/// history-only swap intervention at a preregistered tick (plan §6).
pub struct Ma001Run {
    pub organisms: Vec<Organism>,
    condition: Condition,
    config: Ma001Config,
    shuffle_rng: u64,
    /// Per-organism, per-partner raw action counts accumulated during the primary analysis
    /// window (`burn_in_ticks..total_ticks`).
    pub analysis_counts: Vec<HashMap<AgentId, [u32; 3]>>,
    /// Set only when a swap intervention is configured: `(tick, pre_window_ticks,
    /// post_window_ticks)`.
    swap_at_tick: Option<u64>,
    swap_window_ticks: u64,
    /// Per-organism: the two swapped partner ids, once the swap has happened.
    pub swapped_partners: Vec<Option<(AgentId, AgentId)>>,
    /// Per-organism, per-partner action counts in the pre-swap window (only populated for the
    /// two swapped partners, once known).
    pub pre_swap_counts: Vec<HashMap<AgentId, [u32; 3]>>,
    /// Per-organism, per-partner action counts in the post-swap window.
    pub post_swap_counts: Vec<HashMap<AgentId, [u32; 3]>>,
}

impl Ma001Run {
    /// `swap_at_tick`/`swap_window_ticks`: `Some((tick, window))` runs the swap sub-experiment
    /// (plan §6) instead of the primary comparison — `window` ticks are compared immediately
    /// before `tick` (pre-swap) and the last `window` ticks of the run (post-swap). `None` runs
    /// the primary A/B/C comparison only.
    pub fn new(
        condition: Condition,
        seed: u64,
        config: Ma001Config,
        swap: Option<(u64, u64)>,
    ) -> Self {
        let n = config.population();
        let mut id_alloc = AgentIdAllocator::new();
        let organisms: Vec<Organism> = (0..n)
            .map(|i| {
                Organism::new(config.organism_cfg(), seed.wrapping_add(i as u64).max(1))
                    .with_id(id_alloc.next())
            })
            .collect();
        Self {
            organisms,
            condition,
            config,
            shuffle_rng: seed.wrapping_add(1_000_003).max(1),
            analysis_counts: vec![HashMap::new(); n],
            swap_at_tick: swap.map(|(t, _)| t),
            swap_window_ticks: swap.map(|(_, w)| w).unwrap_or(0),
            swapped_partners: vec![None; n],
            pre_swap_counts: vec![HashMap::new(); n],
            post_swap_counts: vec![HashMap::new(); n],
        }
    }

    /// Real partner id for organism `global_index` this tick, if its group-round-robin schedule
    /// pairs it with a (living) groupmate.
    fn partner_of(&self, global_index: usize, tick: u64) -> Option<usize> {
        let group = global_index / 4;
        let local = global_index % 4;
        let round = (tick % 3) as usize;
        for (a, b) in round_robin_pairs(round) {
            if a == local {
                return Some(group * 4 + b);
            }
            if b == local {
                return Some(group * 4 + a);
            }
        }
        None
    }

    /// What the organism at `idx` is fed as its partner observation this tick, per `condition`.
    /// `real_partner_idx` is `None` if its groupmate is dead (broken pair -- it acts unpaired
    /// this tick, per the plan's driver notes).
    fn observed_partner_ctx(
        &self,
        idx: usize,
        real_partner_idx: Option<usize>,
    ) -> Option<(AgentId, InteractionRecord)> {
        let partner_idx = real_partner_idx?;
        if self.organisms[partner_idx].is_dead() {
            return None;
        }
        let partner_id = self.organisms[partner_idx].id;
        match self.condition {
            Condition::NoHistory => Some((partner_id, InteractionRecord::default())),
            Condition::Bound | Condition::Shuffled => {
                let record = self.organisms[idx]
                    .ledger
                    .get(&partner_id)
                    .copied()
                    .unwrap_or_default();
                Some((partner_id, record))
            }
        }
    }

    /// Read-only, side-effect-free: exactly what organism `idx` would be fed as its partner
    /// observation if `step(tick, ...)` ran right now, without actually running it. Exposed
    /// `pub` purely for manipulation-check testing (plan §7, checks 3 and 6) — lets a test
    /// verify e.g. "`NoHistory` agents truly receive zeros" or "this tick's observation matches
    /// what was actually used" without needing crate-internal access from `tests/*.rs`.
    pub fn peek_observed_partner_ctx(
        &self,
        idx: usize,
        tick: u64,
    ) -> Option<(AgentId, InteractionRecord)> {
        let real_partner_idx = self.partner_of(idx, tick);
        self.observed_partner_ctx(idx, real_partner_idx)
    }

    /// Run the full `config.total_ticks`, accumulating analysis-window counts (and swap-window
    /// counts, if configured) as it goes.
    pub fn run(&mut self, mut resource_for: impl FnMut(usize) -> f64) {
        for tick in 0..self.config.total_ticks {
            self.step(tick, &mut resource_for);
        }
    }

    /// One tick, exposed `pub` (rather than only via [`Self::run`]) so manipulation-check tests
    /// (plan §7) can single-step and snapshot state between ticks -- e.g. around a shuffle or
    /// swap boundary.
    pub fn step(&mut self, tick: u64, resource_for: &mut impl FnMut(usize) -> f64) {
        let n = self.organisms.len();
        let resource = resource_for(n);

        // Real (pre-swap-agnostic) partner-of-record for this tick, from the fixed group
        // schedule -- used for both the act phase and, after resolve_pair_transfer, the learn
        // phase and count accumulation.
        let real_partner: Vec<Option<usize>> = (0..n).map(|i| self.partner_of(i, tick)).collect();

        // Phase A: act, using each organism's own pre-tick observation per `condition`.
        let mut pendings = Vec::with_capacity(n);
        let mut requested = vec![0.0f64; n];
        let mut actions = vec![0usize; n];
        for i in 0..n {
            if self.organisms[i].is_dead() {
                pendings.push(None);
                continue;
            }
            let ctx = self.observed_partner_ctx(i, real_partner[i]);
            let (t, pending) = self.organisms[i].act_social(resource, None, ctx);
            requested[i] = t.transfer_given;
            actions[i] = t.action;
            pendings.push(Some(pending));
        }

        // Phase B: resolve transfers, once per unique living pair this tick.
        for i in 0..n {
            let Some(j) = real_partner[i] else { continue };
            if j <= i {
                continue; // each pair processed from its lower index only
            }
            if self.organisms[i].is_dead() || self.organisms[j].is_dead() {
                continue;
            }
            let mut e_i = self.organisms[i].energy;
            let mut e_j = self.organisms[j].energy;
            let (actual_i, actual_j) =
                resolve_pair_transfer(&mut e_i, &mut e_j, requested[i], requested[j]);
            self.organisms[i].energy = e_i;
            self.organisms[j].energy = e_j;

            let id_i = self.organisms[i].id;
            let id_j = self.organisms[j].id;
            {
                let entry = self.organisms[i].ledger.entry(id_j).or_default();
                entry.encounter_count += 1;
                entry.given_to_partner += actual_i;
                entry.received_from_partner += actual_j;
            }
            {
                let entry = self.organisms[j].ledger.entry(id_i).or_default();
                entry.encounter_count += 1;
                entry.given_to_partner += actual_j;
                entry.received_from_partner += actual_i;
            }
        }

        // Phase C: learn from the realized outcome, using each organism's post-transfer
        // observation per `condition` (identical selection logic to phase A, re-evaluated after
        // the ledger update).
        for i in 0..n {
            let Some(pending) = pendings[i].take() else {
                continue;
            };
            let ctx = self.observed_partner_ctx(i, real_partner[i]);
            self.organisms[i].learn_from_realized_outcome(pending, ctx);
        }

        // Shuffle epoch (Shuffled condition only): permute each living organism's own ledger
        // values across its own partner keys.
        if self.condition == Condition::Shuffled
            && tick > 0
            && tick % self.config.shuffle_epoch_ticks == 0
        {
            for i in 0..n {
                if self.organisms[i].is_dead() {
                    continue;
                }
                self.shuffle_one_ledger(i);
            }
        }

        // Swap intervention, exactly once, history-only (plan §6).
        if self.swap_at_tick == Some(tick) {
            for i in 0..n {
                if self.organisms[i].is_dead() {
                    continue;
                }
                self.swap_extremes_for(i);
            }
        }

        // Count accumulation.
        for i in 0..n {
            let Some(j) = real_partner[i] else { continue };
            if self.organisms[i].is_dead() || self.organisms[j].is_dead() {
                continue;
            }
            let partner_id = self.organisms[j].id;
            if tick >= self.config.burn_in_ticks {
                let entry = self.analysis_counts[i].entry(partner_id).or_insert([0; 3]);
                entry[actions[i]] += 1;
            }
            if let Some(swap_tick) = self.swap_at_tick {
                // Pre-swap window: the `swap_window_ticks` immediately before the swap.
                let pre_start = swap_tick.saturating_sub(self.swap_window_ticks);
                if tick >= pre_start && tick < swap_tick {
                    let entry = self.pre_swap_counts[i].entry(partner_id).or_insert([0; 3]);
                    entry[actions[i]] += 1;
                }
                // Post-swap window: the last `swap_window_ticks` of the run (lets behavior
                // re-equilibrate rather than measuring the immediate transient).
                let post_start = self
                    .config
                    .total_ticks
                    .saturating_sub(self.swap_window_ticks);
                if tick >= post_start {
                    let entry = self.post_swap_counts[i].entry(partner_id).or_insert([0; 3]);
                    entry[actions[i]] += 1;
                }
            }
        }
    }

    /// MA-001A-delta (`ALIFE_MA001A_DELTA_RERUN_PLAN_2026-07-26.md` §3): construct a run whose
    /// organisms use the validated raw-observation delta rule instead of their default
    /// Hebbian+TD pathway. Returns the run plus one [`DeltaRuleLearner`] per organism (kept
    /// external, not a field on `Self`, matching `Ma001rProbe`'s own pattern — `Self`'s struct
    /// shape is otherwise unchanged). Each organism's own model-learning is disabled
    /// (`enable_model_learning: false`, `td_learner: None`) at construction, matching
    /// `Ma001rProbe::set_learning_pathway(false, false)`'s exact effect.
    pub fn new_with_delta_rule(
        condition: Condition,
        seed: u64,
        config: Ma001Config,
        swap: Option<(u64, u64)>,
        delta_cfg: DeltaRuleConfig,
    ) -> (Self, Vec<DeltaRuleLearner>) {
        let mut run = Self::new(condition, seed, config, swap);
        let mut delta_rules = Vec::with_capacity(run.organisms.len());
        for organism in run.organisms.iter_mut() {
            organism.agent.config.enable_model_learning = false;
            organism.agent.td_learner = None;
            delta_rules.push(DeltaRuleLearner::new(delta_cfg, &organism.agent.model));
        }
        (run, delta_rules)
    }

    /// The realized post-tick state a [`DeltaRuleLearner`] learns from (plan §3 step 4) — the
    /// counterpart to `old_state`, built from the *actual* resource/energy/partner-context values
    /// after phase B's real transfer resolution, not a scripted outcome. Mirrors
    /// `Organism::social_channels`'s own convention (all-zero social channels when unpaired) since
    /// that method is private to `organism.rs` — reimplemented locally rather than reached into,
    /// matching `ma001r.rs`'s own precedent for `realized_state`/`variant_state`.
    fn realized_state_for(
        resource: f64,
        energy: f64,
        ctx: &Option<(AgentId, InteractionRecord)>,
    ) -> HiddenState {
        let (partner_present, given_c, received_c, count_c) = match ctx {
            Some((_, record)) => (
                1.0,
                compress_for_observation(record.given_to_partner),
                compress_for_observation(record.received_from_partner),
                compress_for_observation(record.encounter_count as f64),
            ),
            None => (0.0, 0.0, 0.0, 0.0),
        };
        HiddenState {
            mean: vec![
                resource,
                energy,
                partner_present,
                given_c,
                received_c,
                count_c,
            ],
            precision: vec![1.0; 6],
            mode_probs: vec![1.0],
            current_mode: 0,
        }
    }

    /// `run`'s delta-rule counterpart (plan §3). `delta_rules[i]` must correspond to
    /// `self.organisms[i]` (as returned by [`Self::new_with_delta_rule`]).
    pub fn run_with_delta_rule(
        &mut self,
        mut resource_for: impl FnMut(usize) -> f64,
        delta_rules: &[DeltaRuleLearner],
    ) {
        for tick in 0..self.config.total_ticks {
            self.step_with_delta_rule(tick, &mut resource_for, delta_rules);
        }
    }

    /// `step`'s delta-rule counterpart (plan §3): identical three-phase structure (act / resolve
    /// transfers / learn), except phase A additionally captures each living organism's
    /// `tick.raw_observation` as `old_state`, and phase C additionally calls
    /// `delta_rules[i].update(...)` using the realized post-transfer state — instead of relying on
    /// `learn_from_realized_outcome`'s own (disabled) model-learning. `forced_action` stays `None`
    /// throughout: organisms still freely choose via real `select_action()`, since this rerun
    /// tests population-scale policy differentiation, not a scripted action.
    pub fn step_with_delta_rule(
        &mut self,
        tick: u64,
        resource_for: &mut impl FnMut(usize) -> f64,
        delta_rules: &[DeltaRuleLearner],
    ) {
        let n = self.organisms.len();
        let resource = resource_for(n);

        let real_partner: Vec<Option<usize>> = (0..n).map(|i| self.partner_of(i, tick)).collect();

        // Phase A: act, capturing raw_observation as old_state for the delta rule.
        let mut pendings = Vec::with_capacity(n);
        let mut requested = vec![0.0f64; n];
        let mut actions = vec![0usize; n];
        let mut old_states: Vec<Option<HiddenState>> = vec![None; n];
        for i in 0..n {
            if self.organisms[i].is_dead() {
                pendings.push(None);
                continue;
            }
            let ctx = self.observed_partner_ctx(i, real_partner[i]);
            let (t, pending) = self.organisms[i].act_social(resource, None, ctx);
            requested[i] = t.transfer_given;
            actions[i] = t.action;
            old_states[i] = Some(HiddenState {
                mean: t.raw_observation,
                precision: vec![1.0; 6],
                mode_probs: vec![1.0],
                current_mode: 0,
            });
            pendings.push(Some(pending));
        }

        // Phase B: resolve transfers, once per unique living pair this tick -- identical to
        // `step()`'s own phase B.
        for i in 0..n {
            let Some(j) = real_partner[i] else { continue };
            if j <= i {
                continue;
            }
            if self.organisms[i].is_dead() || self.organisms[j].is_dead() {
                continue;
            }
            let mut e_i = self.organisms[i].energy;
            let mut e_j = self.organisms[j].energy;
            let (actual_i, actual_j) =
                resolve_pair_transfer(&mut e_i, &mut e_j, requested[i], requested[j]);
            self.organisms[i].energy = e_i;
            self.organisms[j].energy = e_j;

            let id_i = self.organisms[i].id;
            let id_j = self.organisms[j].id;
            {
                let entry = self.organisms[i].ledger.entry(id_j).or_default();
                entry.encounter_count += 1;
                entry.given_to_partner += actual_i;
                entry.received_from_partner += actual_j;
            }
            {
                let entry = self.organisms[j].ledger.entry(id_i).or_default();
                entry.encounter_count += 1;
                entry.given_to_partner += actual_j;
                entry.received_from_partner += actual_i;
            }
        }

        // Phase C: learn from the realized outcome (belief-tracking perceive() only, since
        // model-learning is disabled) + apply the delta rule using the realized post-transfer
        // state.
        for i in 0..n {
            let Some(pending) = pendings[i].take() else {
                continue;
            };
            let ctx = self.observed_partner_ctx(i, real_partner[i]);
            self.organisms[i].learn_from_realized_outcome(pending, ctx);
            if let Some(old_state) = old_states[i].take() {
                let realized = Self::realized_state_for(resource, self.organisms[i].energy, &ctx);
                delta_rules[i].update(
                    &mut self.organisms[i].agent.model,
                    &old_state,
                    actions[i],
                    &realized,
                );
            }
        }

        // Shuffle epoch (Shuffled condition only) -- identical to `step()`'s own logic.
        if self.condition == Condition::Shuffled
            && tick > 0
            && tick % self.config.shuffle_epoch_ticks == 0
        {
            for i in 0..n {
                if self.organisms[i].is_dead() {
                    continue;
                }
                self.shuffle_one_ledger(i);
            }
        }

        // Swap intervention, exactly once, history-only -- identical to `step()`'s own logic.
        if self.swap_at_tick == Some(tick) {
            for i in 0..n {
                if self.organisms[i].is_dead() {
                    continue;
                }
                self.swap_extremes_for(i);
            }
        }

        // Count accumulation -- identical to `step()`'s own logic.
        for i in 0..n {
            let Some(j) = real_partner[i] else { continue };
            if self.organisms[i].is_dead() || self.organisms[j].is_dead() {
                continue;
            }
            let partner_id = self.organisms[j].id;
            if tick >= self.config.burn_in_ticks {
                let entry = self.analysis_counts[i].entry(partner_id).or_insert([0; 3]);
                entry[actions[i]] += 1;
            }
            if let Some(swap_tick) = self.swap_at_tick {
                let pre_start = swap_tick.saturating_sub(self.swap_window_ticks);
                if tick >= pre_start && tick < swap_tick {
                    let entry = self.pre_swap_counts[i].entry(partner_id).or_insert([0; 3]);
                    entry[actions[i]] += 1;
                }
                let post_start = self
                    .config
                    .total_ticks
                    .saturating_sub(self.swap_window_ticks);
                if tick >= post_start {
                    let entry = self.post_swap_counts[i].entry(partner_id).or_insert([0; 3]);
                    entry[actions[i]] += 1;
                }
            }
        }
    }

    /// `pub` wrapper around [`Self::shuffle_one_ledger`], purely for manipulation-check testing
    /// (plan §7, check 2) — lets a test inject an arbitrary ledger and verify the shuffle's
    /// multiset-preserving property in isolation, decoupled from a real tick's own encounter
    /// update happening in the same step.
    pub fn shuffle_ledger_for_testing(&mut self, idx: usize) {
        self.shuffle_one_ledger(idx)
    }

    /// Permute organism `idx`'s own ledger values across its own partner keys (Shuffled
    /// condition). Keys collected then sorted before any RNG use, so the operation never depends
    /// on `HashMap`'s non-deterministic iteration order.
    fn shuffle_one_ledger(&mut self, idx: usize) {
        let mut partners: Vec<AgentId> = self.organisms[idx].ledger.keys().copied().collect();
        partners.sort();
        if partners.len() < 2 {
            return;
        }
        let mut values: Vec<InteractionRecord> = partners
            .iter()
            .map(|p| self.organisms[idx].ledger[p])
            .collect();
        shuffle_in_place(&mut values, &mut self.shuffle_rng);
        for (partner, value) in partners.into_iter().zip(values.into_iter()) {
            self.organisms[idx].ledger.insert(partner, value);
        }
    }

    /// `pub` wrapper around [`Self::swap_extremes_for`], purely for manipulation-check testing
    /// (plan §7, check 5) — lets a test inject known ledger values and verify the swap changes
    /// exactly the two intended records, in isolation from needing many ticks of real history.
    pub fn swap_extremes_for_testing(&mut self, idx: usize) {
        self.swap_extremes_for(idx)
    }

    /// History-only swap: identify organism `idx`'s highest- and lowest-`net_balance()` partners
    /// (ties broken by lower `AgentId`) and swap their complete `InteractionRecord`s. Never reads
    /// behavior/action data -- only the already-accumulated ledger.
    fn swap_extremes_for(&mut self, idx: usize) {
        let mut partners: Vec<AgentId> = self.organisms[idx].ledger.keys().copied().collect();
        partners.sort();
        if partners.len() < 2 {
            return;
        }
        let mut highest = partners[0];
        let mut lowest = partners[0];
        for &p in &partners[1..] {
            let bal = self.organisms[idx].ledger[&p].net_balance();
            let hi_bal = self.organisms[idx].ledger[&highest].net_balance();
            let lo_bal = self.organisms[idx].ledger[&lowest].net_balance();
            if bal > hi_bal || (bal == hi_bal && p < highest) {
                highest = p;
            }
            if bal < lo_bal || (bal == lo_bal && p < lowest) {
                lowest = p;
            }
        }
        if highest == lowest {
            return;
        }
        let hi_record = self.organisms[idx].ledger[&highest];
        let lo_record = self.organisms[idx].ledger[&lowest];
        self.organisms[idx].ledger.insert(highest, lo_record);
        self.organisms[idx].ledger.insert(lowest, hi_record);
        self.swapped_partners[idx] = Some((highest, lowest));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_covers_all_three_partners_exactly_once_per_cycle() {
        let mut seen = std::collections::HashSet::new();
        for round in 0..3 {
            for (a, b) in round_robin_pairs(round) {
                seen.insert((a.min(b), a.max(b)));
            }
        }
        // C(4,2) = 6 distinct pairs, all covered across 3 rounds.
        assert_eq!(seen.len(), 6);
    }

    #[test]
    fn round_robin_is_mutual_and_covers_every_local_index_every_round() {
        for round in 0..3 {
            let pairs = round_robin_pairs(round);
            let mut covered = std::collections::HashSet::new();
            for (a, b) in pairs {
                assert!(covered.insert(a));
                assert!(covered.insert(b));
            }
            assert_eq!(covered, std::collections::HashSet::from([0, 1, 2, 3]));
        }
    }

    #[test]
    fn dirichlet_smooth_sums_to_one() {
        let p = dirichlet_smooth([10, 0, 5], 0.5);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn identical_distributions_have_zero_divergence() {
        let p = dirichlet_smooth([10, 5, 5], 0.5);
        assert!(jensen_shannon_divergence(&p, &p) < 1e-12);
    }

    #[test]
    fn maximally_different_distributions_approach_the_upper_bound() {
        let p = dirichlet_smooth([1_000_000, 0, 0], 0.01);
        let q = dirichlet_smooth([0, 0, 1_000_000], 0.01);
        assert!(jensen_shannon_divergence(&p, &q) > 0.9);
    }

    #[test]
    fn agent_divergence_score_needs_at_least_two_eligible_partners() {
        let mut counts = HashMap::new();
        let mut alloc = AgentIdAllocator::new();
        counts.insert(alloc.next(), [100, 0, 0]);
        assert_eq!(agent_divergence_score(&counts, 0.5, 50), None);
        counts.insert(alloc.next(), [0, 100, 0]);
        assert!(agent_divergence_score(&counts, 0.5, 50).is_some());
    }

    #[test]
    fn delta_rule_run_completes_without_panicking_and_moves_some_coefficients() {
        let mut small_cfg = Ma001Config::default();
        small_cfg.groups = 2; // 8 organisms, small+fast smoke test
        small_cfg.total_ticks = 60;
        let (mut run, delta_rules) = Ma001Run::new_with_delta_rule(
            Condition::Bound,
            1,
            small_cfg,
            None,
            DeltaRuleConfig::default(),
        );
        assert_eq!(delta_rules.len(), run.organisms.len());
        let before: Vec<_> = run
            .organisms
            .iter()
            .map(|o| o.agent.model.transition_matrices.clone())
            .collect();
        run.run_with_delta_rule(|_n| 0.25, &delta_rules);
        let alive = run.organisms.iter().filter(|o| !o.is_dead()).count();
        assert!(
            alive > 0,
            "expected at least one organism to survive 60 ticks"
        );
        let any_moved = run
            .organisms
            .iter()
            .zip(before.iter())
            .any(|(o, b)| &o.agent.model.transition_matrices != b);
        assert!(
            any_moved,
            "expected at least one organism's transition matrices to move under the delta rule"
        );
    }

    #[test]
    fn delta_rule_organisms_have_model_learning_disabled() {
        let (run, _delta_rules) = Ma001Run::new_with_delta_rule(
            Condition::Bound,
            1,
            Ma001Config::default(),
            None,
            DeltaRuleConfig::default(),
        );
        for organism in &run.organisms {
            assert!(!organism.agent.config.enable_model_learning);
            assert!(organism.agent.td_learner.is_none());
        }
    }
}
