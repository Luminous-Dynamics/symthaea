// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A population of [`Organism`]s sharing a resource pool, per `ALIFE_PLAN_2026-07-08.md` Phase 1.
//!
//! `Population` is deliberately content-agnostic about *what* the resource is (plants, prey,
//! anything else): the caller supplies a `resource_for(current_count) -> f64` closure each tick,
//! so density-dependence (a smaller per-capita share as population grows) is whatever the caller
//! wires in. This lets the same type drive both Phase 1a (one species, a shared plant pool) and
//! Phase 1b (two coupled species, `crate::predator_prey`) without duplicating birth/death logic.
//!
//! Reproduction and death are driven entirely by each [`Organism`]'s own real energy trajectory
//! (itself the product of its own `select_action`/`perceive` loop) — this module adds no FEP
//! machinery of its own, only the population-level bookkeeping around it.

use std::collections::HashMap;

use crate::agent_id::{AgentId, AgentIdAllocator};
use crate::encounter::EncounterScheduler;
use crate::events::GenesisEvent;
use crate::genome::Genome;
use crate::organism::{Action, Organism, OrganismConfig};

/// How an offspring's genome is chosen at reproduction, per `ALIFE_PLAN_2026-07-08.md` Phase 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InheritanceMode {
    /// Offspring inherit (mutated) from the parent that actually reproduced -- real selection:
    /// a genome only propagates by way of an organism that stayed alive long enough to reach
    /// `reproduction_energy_threshold`.
    #[default]
    FromParent,
    /// Offspring inherit (mutated) from a uniformly-random *current* population member,
    /// independent of who actually reproduced -- deliberately breaks the fitness→inheritance
    /// link. Exists only as Phase 4c's falsifiable control, never a "better" mode: real
    /// selection pressure (death, reproduction thresholds) still applies, but a successful
    /// reproduction event no longer preferentially propagates *that individual's* genome.
    RandomPeer,
}

/// Birth/death thresholds and the per-organism config newborns inherit.
///
/// `Default` exists so existing call sites can add `..Default::default()` to pick up Phase 4's
/// new `mutation_rate`/`mutation_std`/`inheritance` fields without re-specifying the rest --
/// **do not rely on it for the threshold fields themselves** (they default to `0.0`, which
/// would make every organism reproduce on its very first tick).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PopulationConfig {
    /// An organism with energy at or below this dies and is removed.
    pub death_energy_threshold: f64,
    /// An organism with energy at or above this reproduces this tick.
    pub reproduction_energy_threshold: f64,
    /// Energy the parent is left with (and the offspring starts with) after reproducing — a
    /// real cost, not a free clone, so reproduction can't happen every tick indefinitely.
    pub reproduction_energy_cost: f64,
    /// Config the *initial* population is seeded with. Offspring no longer simply clone this
    /// (Phase 4) — they inherit a mutated `Genome` from a parent chosen per `inheritance`; only
    /// the non-heritable fields (costs, thermodynamic constants, thresholds) come from here.
    pub organism_cfg: OrganismConfig,
    /// Probability each heritable trait mutates at a birth event. `0.0` (Phase 0-3's implicit
    /// behavior) means offspring are exact genome copies of their chosen parent.
    pub mutation_rate: f64,
    /// Magnitude of a mutation, when one occurs (`Genome::mutate`'s `mutation_std`).
    pub mutation_std: f64,
    pub inheritance: InheritanceMode,
}

/// What happened during one [`Population::step`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepSummary {
    /// Population size after this tick's births/deaths.
    pub population: usize,
    /// How many organisms actually chose to forage this tick (their own `select_action()`
    /// result, not a scripted rate) — exposed so callers like `predator_prey` can turn real
    /// foraging decisions into a real consequence for a coupled species.
    pub forage_count: usize,
    pub births_this_tick: u64,
    pub deaths_this_tick: u64,
}

/// Resolve one paired encounter's mutual transfer requests into what's actually delivered,
/// conservatively (Genesis v0.1 audit Gate 3, 2026-07-25/26).
///
/// **Contract**: `a_energy`/`b_energy` must already reflect each side's own giver-side debit for
/// `a_requested`/`b_requested` -- i.e. `Organism::act_phase` has already subtracted the
/// requested amount from its own energy before this function ever runs (that's how
/// `Organism::act_phase` computes `transfer_given`/`_requested` in the first place: it's already
/// headroom-limited and already debited). This function's only remaining job is to cap *delivery*
/// by the *receiver's* remaining capacity (`1.0 - energy`) and refund any undelivered remainder
/// back to the giver, so that energy is never silently created or destroyed by a transfer.
/// Extracted as a free, `Organism`-independent function (plain `&mut f64`s) so it's directly
/// unit-testable without needing a full `Population`/`Organism` fixture. Returns `(a_actual,
/// b_actual)` -- the amount each side actually delivered, which is also what belongs in both the
/// ledger and the event log (never the nominal `_requested` amount, which can overstate reality
/// once a receiver saturates).
///
/// `pub` (2026-07-26, MA-001 preregistration): exposed so an experiment driver that doesn't go
/// through `Population::step_social` (e.g. because it needs a custom pairing schedule or to
/// substitute observations) can reuse the exact same conservative-transfer physics rather than
/// reimplementing it and risking drift from the substrate's real semantics.
pub fn resolve_pair_transfer(
    a_energy: &mut f64,
    b_energy: &mut f64,
    a_requested: f64,
    b_requested: f64,
) -> (f64, f64) {
    let a_actual = if a_requested > 0.0 {
        let capacity = (1.0 - *b_energy).max(0.0);
        let actual = a_requested.min(capacity);
        let refund = a_requested - actual;
        if actual > 0.0 {
            *b_energy = (*b_energy + actual).clamp(0.0, 1.0);
        }
        if refund > 0.0 {
            *a_energy = (*a_energy + refund).clamp(0.0, 1.0);
        }
        actual
    } else {
        0.0
    };
    let b_actual = if b_requested > 0.0 {
        let capacity = (1.0 - *a_energy).max(0.0);
        let actual = b_requested.min(capacity);
        let refund = b_requested - actual;
        if actual > 0.0 {
            *a_energy = (*a_energy + actual).clamp(0.0, 1.0);
        }
        if refund > 0.0 {
            *b_energy = (*b_energy + refund).clamp(0.0, 1.0);
        }
        actual
    } else {
        0.0
    };
    (a_actual, b_actual)
}

pub struct Population {
    pub organisms: Vec<Organism>,
    pub cfg: PopulationConfig,
    next_seed: u64,
    /// Stream for mutation and (in `InheritanceMode::RandomPeer`) parent-selection randomness --
    /// deliberately independent of any `Organism`'s own RNG, same precedent as
    /// `predator_prey.rs`'s `PredatorPreySim::rng_state`.
    rng_state: u64,
    pub total_births: u64,
    pub total_deaths: u64,
    /// Genesis v0 (G0a): assigns every organism's persistent [`AgentId`] -- initial population at
    /// construction, offspring at birth in `step`/`step_social`. Never reused, per
    /// `ALIFE_MULTIAGENT_GENESIS_PLAN_2026-07-25.md`'s "Identity" Stage 0 invariant.
    id_allocator: AgentIdAllocator,
    /// Genesis v0 event-stream log, appended to by `step_social` only (`step` never touches
    /// this). Grows one [`GenesisEvent`] per organism per tick -- callers running a long session
    /// should periodically [`Self::drain_event_log`] rather than let it grow unbounded (the same
    /// class of mistake that caused a real runaway-memory incident during this plan's own Stage
    /// 0 test development, from an unrelated unbounded-population bug -- worth guarding against
    /// deliberately here too).
    event_log: Vec<GenesisEvent>,
    /// Ticks `step_social` has been called, for `GenesisEvent::tick`. `step` never advances this
    /// -- it doesn't emit events, and mixing the two step methods on one `Population` isn't a
    /// supported usage this plan needed to define.
    current_tick: u64,
}

impl Population {
    pub fn new(cfg: PopulationConfig, initial_count: usize, seed_base: u64) -> Self {
        let mut id_allocator = AgentIdAllocator::new();
        let organisms = (0..initial_count)
            .map(|i| {
                Organism::new(cfg.organism_cfg, seed_base.wrapping_add(i as u64).max(1))
                    .with_id(id_allocator.allocate())
            })
            .collect();
        Self {
            organisms,
            cfg,
            next_seed: seed_base.wrapping_add(initial_count as u64).max(1),
            rng_state: seed_base.wrapping_add(1_000_011).max(1),
            total_births: 0,
            total_deaths: 0,
            id_allocator,
            event_log: Vec::new(),
            current_tick: 0,
        }
    }

    /// Take ownership of every event logged so far, leaving the log empty. Callers running a
    /// long Genesis session should call this periodically (e.g. every N ticks) and persist the
    /// result, rather than let `step_social` grow this vector unbounded for the whole run.
    pub fn drain_event_log(&mut self) -> Vec<GenesisEvent> {
        std::mem::take(&mut self.event_log)
    }

    fn next_unit(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// Remove up to `n` of the lowest-energy organisms (predation picks off the weak first —
    /// a real, if simplified, ecological detail). Returns how many were actually removed,
    /// capped by the current population size.
    pub fn cull_weakest(&mut self, n: usize) -> usize {
        let n = n.min(self.organisms.len());
        if n == 0 {
            return 0;
        }
        self.organisms
            .sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
        self.organisms.drain(0..n);
        self.total_deaths += n as u64;
        n
    }

    /// One tick for every living organism: real `perceive`/`select_action`/`act`/
    /// `learn_from_outcome`, then birth/death from the resulting energy.
    ///
    /// `resource_for(n)` is called once with the population size *before* this tick's births/
    /// deaths, and its result is given to every organism as their `resource_level` observation
    /// this tick — the shared-pool coupling that produces density-dependence.
    ///
    /// `FnMut`, not `Fn` -- lets a caller drive a genuinely stateful resource source (e.g.
    /// Phase 5's `EarthForcedEnvironment`, which integrates real physics forward one tick per
    /// call) rather than requiring every resource source to be a pure function of `n`. Every
    /// existing caller passes a pure closure, which trivially also satisfies `FnMut`, so this is
    /// a non-breaking relaxation.
    pub fn step(&mut self, mut resource_for: impl FnMut(usize) -> f64) -> StepSummary {
        let n = self.organisms.len();
        let resource = resource_for(n);

        let mut forage_count = 0u64;
        let mut births_this_tick = 0u64;
        let mut deaths_this_tick = 0u64;
        let mut newborns = Vec::new();

        let mut i = 0;
        while i < self.organisms.len() {
            let tick = self.organisms[i].tick(resource, None);
            if tick.action == Action::Forage.index() {
                forage_count += 1;
            }

            if tick.energy <= self.cfg.death_energy_threshold {
                self.organisms.remove(i);
                self.total_deaths += 1;
                deaths_this_tick += 1;
                continue; // next element has shifted into position i
            }

            if tick.energy >= self.cfg.reproduction_energy_threshold {
                self.organisms[i].energy = self.cfg.reproduction_energy_cost;

                // Phase 4: the offspring's genome comes from a parent chosen per
                // `self.cfg.inheritance`, mutated, then reapplied to the population's shared
                // non-heritable constants (costs, thermodynamic constants, thresholds -- see
                // `Genome`'s module docs for exactly which fields are heritable).
                let parent_genome = match self.cfg.inheritance {
                    InheritanceMode::FromParent => Genome::from_config(&self.organisms[i].cfg),
                    InheritanceMode::RandomPeer => {
                        let r = self.next_unit();
                        let peer_idx = ((r * self.organisms.len() as f64) as usize)
                            .min(self.organisms.len() - 1);
                        Genome::from_config(&self.organisms[peer_idx].cfg)
                    }
                };
                let offspring_genome = parent_genome.mutate(
                    &mut self.rng_state,
                    self.cfg.mutation_rate,
                    self.cfg.mutation_std,
                );
                let offspring_cfg = offspring_genome.apply_to(self.cfg.organism_cfg);

                // Lineage tracks the physically-reproducing organism (`self.organisms[i]`)
                // regardless of `InheritanceMode` -- `RandomPeer` only randomizes which genome
                // gets copied, not who gave birth.
                let parent_lineage_id = self.organisms[i].lineage_id;
                let parent_generation = self.organisms[i].generation;
                let mut offspring = Organism::new(offspring_cfg, self.next_seed)
                    .with_id(self.id_allocator.allocate())
                    .with_lineage(parent_lineage_id, parent_generation + 1);
                self.next_seed = self.next_seed.wrapping_add(1).max(1);
                offspring.energy = self.cfg.reproduction_energy_cost;
                newborns.push(offspring);
                self.total_births += 1;
                births_this_tick += 1;
            }

            i += 1;
        }

        self.organisms.extend(newborns);

        StepSummary {
            population: self.organisms.len(),
            forage_count: forage_count as usize,
            births_this_tick,
            deaths_this_tick,
        }
    }

    /// Genesis v0 (G0f/g) social-aware step: identical birth/death bookkeeping to [`Self::step`],
    /// but each organism ticks via [`Organism::act_social`]/[`Organism::learn_from_realized_outcome`],
    /// paired each tick by `scheduler` (`ALIFE_MULTIAGENT_GENESIS_PLAN_2026-07-25.md`, G0b).
    /// `step()` itself is untouched by this method's existence -- calling `step` never invokes any
    /// of this, which is what makes the Stage 0 "baseline equivalence" invariant hold by
    /// construction rather than by convention.
    ///
    /// Four phases per tick, in order:
    /// 1. Pair this tick's living agents (pre-tick population order) via `scheduler.pair`.
    /// 2. Act every organism independently, each fed its own *pre-tick* ledger snapshot for its
    ///    current partner (never this tick's not-yet-decided transfer) -- the "observation
    ///    causality" invariant. Learning is deferred (see 3.5).
    /// 3. For each pair (each unordered pair processed exactly once, not once per side -- doing
    ///    it per-side would double-count `encounter_count`), credit any realized transfer to the
    ///    partner's energy and update both organisms' ledgers with the raw counters from this
    ///    single encounter (conservatively -- see `resolve_pair_transfer`).
    ///
    /// 3.5. Genesis v0.1 audit Gate 4: complete each organism's learning step using the *realized*
    /// post-Phase-3 partner ledger snapshot, not the pre-tick one Phase 2 used -- so an
    /// organism's own `Transfer` this tick is now visible to its own learning update this same
    /// tick, rather than only appearing one full `step_social` call late.
    ///
    /// Death/birth bookkeeping then proceeds exactly as in `step`, including assigning newborns a
    /// fresh, never-reused `AgentId` from the same allocator.
    pub fn step_social(
        &mut self,
        mut resource_for: impl FnMut(usize) -> f64,
        scheduler: &mut EncounterScheduler,
    ) -> StepSummary {
        let n = self.organisms.len();
        let resource = resource_for(n);
        let tick_number = self.current_tick;
        self.current_tick += 1;
        let energy_before: Vec<f64> = self.organisms.iter().map(|o| o.energy).collect();

        // Phase 1: pairing, from pre-tick population order.
        let living_ids: Vec<AgentId> = self.organisms.iter().map(|o| o.id).collect();
        let pairs = scheduler.pair(&living_ids);
        let id_to_idx: HashMap<AgentId, usize> = living_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        let mut partner_of: Vec<Option<AgentId>> = vec![None; n];
        for (a, b) in &pairs {
            if let (Some(&ia), Some(&ib)) = (id_to_idx.get(a), id_to_idx.get(b)) {
                partner_of[ia] = Some(*b);
                partner_of[ib] = Some(*a);
            }
        }

        // Phase 2: act every organism independently, using pre-tick ledger snapshots only.
        // Learning is deferred to Phase 3.5, once the realized outcome is known.
        let mut forage_count = 0u64;
        let mut transfer_given_by = vec![0.0f64; n];
        let mut actions = vec![0usize; n];
        let mut pending_by = Vec::with_capacity(n);
        for i in 0..n {
            let partner_ctx = partner_of[i].map(|pid| {
                let record = self.organisms[i]
                    .ledger
                    .get(&pid)
                    .copied()
                    .unwrap_or_default();
                (pid, record)
            });
            let (tick, pending) = self.organisms[i].act_social(resource, None, partner_ctx);
            actions[i] = tick.action;
            transfer_given_by[i] = tick.transfer_given;
            pending_by.push(pending);
            if tick.action == Action::Forage.index() {
                forage_count += 1;
            }
        }

        // Phase 3: apply cross-organism transfer credit + ledger updates, once per unique pair
        // (not once per side -- see this method's doc comment on why that would double-count
        // `encounter_count`).
        //
        // Conservative accounting (Genesis v0.1 audit Gate 3, 2026-07-26): `tick_social` already
        // limited each organism's *requested* transfer to its own headroom, but the earlier
        // version of this method credited the receiver with the full requested amount even when
        // the receiver had no capacity left (silently lost to the `[0,1]` clamp) while the ledger
        // still recorded the requested amount as fully given *and* fully received -- overstating
        // physical reality. Now: the actually-deliverable amount is capped by the receiver's real
        // headroom, any undelivered remainder is refunded to the giver (energy is never created
        // or destroyed by a transfer), and both ledger fields record only the amount that was
        // actually delivered. `transfer_given_by` is updated in place to this actual amount so
        // `GenesisEvent::transfer_amount` (built below) reports reality too, not the request.
        for (a, b) in &pairs {
            let (Some(&ia), Some(&ib)) = (id_to_idx.get(a), id_to_idx.get(b)) else {
                continue;
            };
            let mut a_energy = self.organisms[ia].energy;
            let mut b_energy = self.organisms[ib].energy;
            let (a_actual, b_actual) = resolve_pair_transfer(
                &mut a_energy,
                &mut b_energy,
                transfer_given_by[ia],
                transfer_given_by[ib],
            );
            self.organisms[ia].energy = a_energy;
            self.organisms[ib].energy = b_energy;

            transfer_given_by[ia] = a_actual;
            transfer_given_by[ib] = b_actual;

            {
                let entry = self.organisms[ia].ledger.entry(*b).or_default();
                entry.encounter_count += 1;
                entry.given_to_partner += a_actual;
                entry.received_from_partner += b_actual;
            }
            {
                let entry = self.organisms[ib].ledger.entry(*a).or_default();
                entry.encounter_count += 1;
                entry.given_to_partner += b_actual;
                entry.received_from_partner += a_actual;
            }
        }

        // Phase 3.5 (Genesis v0.1 audit Gate 4): complete each organism's learning step using the
        // *realized* post-Phase-3 ledger snapshot for its partner -- reflects this tick's own
        // transfer, not just history from before it. Consumes `pending_by` in order.
        for (i, pending) in pending_by.into_iter().enumerate() {
            let realized_partner = partner_of[i].map(|pid| {
                let record = self.organisms[i]
                    .ledger
                    .get(&pid)
                    .copied()
                    .unwrap_or_default();
                (pid, record)
            });
            self.organisms[i].learn_from_realized_outcome(pending, realized_partner);
        }

        // Event log: one GenesisEvent per organism, using this tick's pre-tick energy
        // (`energy_before`) and post-Phase-3 energy (`self.organisms[i].energy`, which already
        // includes any transfer credit this organism received this tick) -- must run before
        // Phase 4 removes any organism, while indices are still valid.
        for i in 0..n {
            self.event_log.push(GenesisEvent {
                tick: tick_number,
                agent_id: self.organisms[i].id,
                partner_id: partner_of[i],
                action: actions[i],
                resource_before: energy_before[i],
                resource_after: self.organisms[i].energy,
                transfer_amount: transfer_given_by[i],
                generation: self.organisms[i].generation,
                lineage_id: self.organisms[i].lineage_id,
            });
        }

        // Phase 4: death/birth bookkeeping -- same logic and thresholds as `step`.
        let mut births_this_tick = 0u64;
        let mut deaths_this_tick = 0u64;
        let mut newborns = Vec::new();
        let mut i = 0;
        while i < self.organisms.len() {
            let energy = self.organisms[i].energy;
            if energy <= self.cfg.death_energy_threshold {
                self.organisms.remove(i);
                self.total_deaths += 1;
                deaths_this_tick += 1;
                continue;
            }

            if energy >= self.cfg.reproduction_energy_threshold {
                self.organisms[i].energy = self.cfg.reproduction_energy_cost;

                let parent_genome = match self.cfg.inheritance {
                    InheritanceMode::FromParent => Genome::from_config(&self.organisms[i].cfg),
                    InheritanceMode::RandomPeer => {
                        let r = self.next_unit();
                        let peer_idx = ((r * self.organisms.len() as f64) as usize)
                            .min(self.organisms.len() - 1);
                        Genome::from_config(&self.organisms[peer_idx].cfg)
                    }
                };
                let offspring_genome = parent_genome.mutate(
                    &mut self.rng_state,
                    self.cfg.mutation_rate,
                    self.cfg.mutation_std,
                );
                let offspring_cfg = offspring_genome.apply_to(self.cfg.organism_cfg);

                // Lineage tracks the physically-reproducing organism (`self.organisms[i]`)
                // regardless of `InheritanceMode` -- `RandomPeer` only randomizes which genome
                // gets copied, not who gave birth.
                let parent_lineage_id = self.organisms[i].lineage_id;
                let parent_generation = self.organisms[i].generation;
                let mut offspring = Organism::new(offspring_cfg, self.next_seed)
                    .with_id(self.id_allocator.allocate())
                    .with_lineage(parent_lineage_id, parent_generation + 1);
                self.next_seed = self.next_seed.wrapping_add(1).max(1);
                offspring.energy = self.cfg.reproduction_energy_cost;
                newborns.push(offspring);
                self.total_births += 1;
                births_this_tick += 1;
            }

            i += 1;
        }

        self.organisms.extend(newborns);

        StepSummary {
            population: self.organisms.len(),
            forage_count: forage_count as usize,
            births_this_tick,
            deaths_this_tick,
        }
    }

    pub fn len(&self) -> usize {
        self.organisms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.organisms.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig::default(),
            ..Default::default()
        }
    }

    #[test]
    fn resolve_pair_transfer_delivers_the_full_request_when_receiver_has_headroom() {
        let mut a_energy = 0.5;
        let mut b_energy = 0.3;
        let (a_actual, b_actual) = resolve_pair_transfer(&mut a_energy, &mut b_energy, 0.1, 0.0);
        assert_eq!(a_actual, 0.1);
        assert_eq!(b_actual, 0.0);
        assert!(
            (a_energy - 0.5).abs() < 1e-12,
            "no refund needed, giver unchanged"
        );
        assert!((b_energy - 0.4).abs() < 1e-12, "receiver gets the full 0.1");
    }

    #[test]
    fn resolve_pair_transfer_caps_delivery_and_refunds_the_giver_when_receiver_is_saturated() {
        let mut a_energy = 0.5;
        let mut b_energy = 0.95; // only 0.05 of headroom left
        let (a_actual, b_actual) = resolve_pair_transfer(&mut a_energy, &mut b_energy, 0.2, 0.0);
        assert!(
            (a_actual - 0.05).abs() < 1e-12,
            "actual delivered must be capped by receiver's real headroom, got {a_actual}"
        );
        assert_eq!(b_actual, 0.0);
        assert!(
            (b_energy - 1.0).abs() < 1e-12,
            "receiver exactly saturates, never exceeds 1.0"
        );
        assert!(
            (a_energy - (0.5 + (0.2 - 0.05))).abs() < 1e-12,
            "giver must be refunded exactly the undelivered 0.15, not lose it: got {a_energy}"
        );
    }

    #[test]
    fn resolve_pair_transfer_is_exactly_conservative_with_headroom_on_both_sides() {
        // `resolve_pair_transfer`'s real contract: `a_energy`/`b_energy` are each *already*
        // reduced by their own `_requested` amount when this runs (the giver's own act_phase
        // debit already happened before Phase 3 calls this) -- this function only resolves the
        // receiver-capacity cap and any refund. So a whole-operation conservation check needs
        // the pre-debit totals, i.e. `energy_passed_in + requested` for each side.
        let a_requested = 0.05;
        let b_requested = 0.02;
        let mut a_energy = 0.4 - a_requested;
        let mut b_energy = 0.3 - b_requested;
        let before_total = (a_energy + a_requested) + (b_energy + b_requested);
        let (a_actual, b_actual) =
            resolve_pair_transfer(&mut a_energy, &mut b_energy, a_requested, b_requested);
        assert_eq!(a_actual, a_requested);
        assert_eq!(b_actual, b_requested);
        let after_total = a_energy + b_energy;
        assert!(
            (before_total - after_total).abs() < 1e-12,
            "with headroom on both sides, total energy across the whole pair operation must be \
             exactly conserved: before={before_total}, after={after_total}"
        );
    }

    /// A traced diagnostic run (see `ALIFE_PLAN_2026-07-08.md` Phase 1 dev notes / commit history)
    /// found `OrganismConfig::default()`'s `forage_efficiency: 0.15` puts an organism's
    /// break-even resource share close to 0.9 -- because the untrained/lightly-trained policy
    /// forages only ~50% of the time, not ~100% as a naive mean-field estimate assumes. That's
    /// fine for Phase 0 (no reproduction/death was in play), but leaves Phase 1 populations on a
    /// knife-edge where even a flat, undivided resource of 0.5 was sub-breakeven and the
    /// population went extinct. Phase 1 tests use this more sustainable efficiency instead, so
    /// growth is a robust, unambiguous outcome under genuinely abundant resources rather than a
    /// coin flip -- Phase 0's already-verified tests are untouched (they use `cfg()` above).
    fn sustainable_cfg() -> PopulationConfig {
        PopulationConfig {
            organism_cfg: OrganismConfig {
                forage_efficiency: 0.6,
                ..OrganismConfig::default()
            },
            ..cfg()
        }
    }

    #[test]
    fn population_never_goes_negative_and_reports_consistent_counts() {
        let mut pop = Population::new(cfg(), 3, 1);
        for _ in 0..500u64 {
            let summary = pop.step(|n| 0.3 / (n.max(1) as f64));
            assert_eq!(summary.population, pop.len());
            assert!(summary.forage_count <= summary.population + summary.deaths_this_tick as usize);
        }
    }

    #[test]
    fn cull_weakest_never_removes_more_than_present() {
        let mut pop = Population::new(cfg(), 4, 7);
        let removed = pop.cull_weakest(100);
        assert_eq!(removed, 4);
        assert!(pop.is_empty());
        assert_eq!(pop.cull_weakest(5), 0);
    }

    #[test]
    fn abundant_resources_grow_the_population() {
        // A generous, constant (NOT shared/divided by population size) per-capita resource is
        // deliberately unbounded growth -- and it compounds fast (~15-20 ticks per doubling, so
        // even a few hundred ticks means millions of organisms and a test that never finishes).
        // A couple of doubling times is enough to prove births can outpace deaths at all;
        // density-dependence is exactly what Phase 1a's shared-pool test covers instead.
        let mut pop = Population::new(sustainable_cfg(), 2, 42);
        for _ in 0..60u64 {
            pop.step(|_n| 0.5); // fixed generous per-capita resource, not shared/divided
        }
        assert!(
            pop.len() > 2,
            "population should have grown, got {}",
            pop.len()
        );
    }
}
