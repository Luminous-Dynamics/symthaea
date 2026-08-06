// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001R — Social→Physical Coupling Probe driver, per
//! `ALIFE_MA001R_SOCIAL_PHYSICAL_COUPLING_PLAN_2026-07-26.md`. Deliberately does not reuse
//! `Ma001Run`/`Population`: one focal `Organism`, two scripted (never-real) partner contexts, and
//! a directly-overwritten physical outcome, removing every population-scale confound (ecology,
//! pairing schedule, resource sharing) MA-001A's null could not rule out. See that plan's §4 for
//! the frozen exposure protocol this module implements exactly.

use symthaea_fep::{HiddenState, TemporalDifferenceLearner};

use crate::agent_id::{AgentId, AgentIdAllocator};
use crate::ledger::{InteractionRecord, compress_for_observation};
use crate::ma001l::DeltaRuleLearner;
use crate::organism::{Action, Organism, OrganismConfig};

/// Context A, per plan §4: a rich shared history (this pair have exchanged resources 20 times).
pub fn context_a() -> InteractionRecord {
    InteractionRecord {
        given_to_partner: 2.0,
        received_from_partner: 2.0,
        encounter_count: 20,
    }
}

/// Context B, per plan §4: no shared history at all.
pub fn context_b() -> InteractionRecord {
    InteractionRecord::default()
}

/// Frozen exposure-protocol parameters (plan §4/§6). `Default` matches the plan's frozen numbers
/// exactly.
#[derive(Debug, Clone, Copy)]
pub struct Ma001rConfig {
    /// Constant resource level fed to every tick — plan §4: "the only thing that differs between
    /// contexts is the social observation."
    pub resource_level: f64,
    /// Scripted energy outcome following Context A, in the (default) bound schedule.
    pub outcome_a: f64,
    /// Scripted energy outcome following Context B, in the (default) bound schedule.
    pub outcome_b: f64,
    /// Scripted energy outcome for the equal-outcome control (plan §6) — same regardless of
    /// context.
    pub equal_outcome: f64,
    /// Main-run training length (plan §4: "1,000 exposures to each context").
    pub training_ticks: u64,
    /// Held-out ticks measured after training, not used for any update (plan §5).
    pub held_out_ticks: u64,
    /// Reversal-condition length after the main run (plan §6).
    pub reversal_ticks: u64,
}

impl Default for Ma001rConfig {
    fn default() -> Self {
        Self {
            resource_level: 0.5,
            outcome_a: 0.9,
            outcome_b: 0.2,
            equal_outcome: 0.6,
            training_ticks: 2000,
            held_out_ticks: 200,
            reversal_ticks: 2000,
        }
    }
}

/// Which context↔outcome correspondence a tick follows — shared by the main run and every control
/// in plan §6, so they all run through identical step mechanics rather than duplicated loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    /// Main run + ablations (plan §4): Context A always pairs with `outcome_a`, Context B with
    /// `outcome_b`.
    Bound,
    /// Reversal control (plan §6): the correspondence is swapped — Context A now pairs with
    /// `outcome_b`, Context B with `outcome_a`.
    Reversed,
    /// Equal-outcome control (plan §6): the scripted outcome is `cfg.equal_outcome` regardless of
    /// which context is presented — a direct false-positive check on the measurement pipeline.
    EqualOutcome,
}

/// Primary metric result (plan §5): counterfactual sensitivity of the generative model's own
/// predicted physical outcome to the social context alone, holding the organism's current
/// resource/energy belief fixed across both variants.
#[derive(Debug, Clone, Copy, Default)]
pub struct CounterfactualReading {
    pub predicted_resource_a: f64,
    pub predicted_resource_b: f64,
    pub predicted_energy_a: f64,
    pub predicted_energy_b: f64,
    /// `|predicted_resource_a - predicted_resource_b| + |predicted_energy_a - predicted_energy_b|`.
    pub delta_predicted: f64,
}

/// One focal organism plus the two scripted, never-real partner identities MA-001R presents to
/// it. Deliberately holds no `Population`/`EncounterScheduler` — see module docs.
pub struct Ma001rProbe {
    pub organism: Organism,
    pub partner_a: AgentId,
    pub partner_b: AgentId,
    pub cfg: Ma001rConfig,
}

impl Ma001rProbe {
    pub fn new(organism_cfg: OrganismConfig, seed: u64, cfg: Ma001rConfig) -> Self {
        let mut alloc = AgentIdAllocator::new();
        let partner_a = alloc.allocate();
        let partner_b = alloc.allocate();
        Self {
            organism: Organism::new(organism_cfg, seed),
            partner_a,
            partner_b,
            cfg,
        }
    }

    /// Learning-pathway ablation setter (plan §7, corrected 2026-07-26): `Organism::new` has no
    /// config knob for either flag, since no call site needed one before this plan. Directly
    /// toggles the constructed agent's `enable_model_learning` and `td_learner` (`Some`/`None`)
    /// rather than trusting `ActiveInferenceAgentConfig::enable_td_learning` alone — per the §2
    /// addendum, `learn_from_outcome`'s own direct Hebbian call is gated only on
    /// `enable_model_learning`, never on `td_learner`, so genuine TD-only isolation requires
    /// `enable_model_learning: false` while `td_learner` is `Some`.
    pub fn set_learning_pathway(&mut self, enable_model_learning: bool, enable_td_learning: bool) {
        let agent = &mut self.organism.agent;
        agent.config.enable_model_learning = enable_model_learning;
        agent.config.enable_td_learning = enable_td_learning;
        if enable_td_learning {
            if agent.td_learner.is_none() {
                agent.td_learner = Some(TemporalDifferenceLearner::new(
                    agent.config.td_config.clone(),
                    agent.config.num_actions,
                    agent.config.state_dim,
                    agent.config.obs_dim,
                ));
            }
        } else {
            agent.td_learner = None;
        }
    }

    /// Context presented on tick `t` under the always-deterministic tick-parity schedule shared by
    /// every control except Shuffled (plan §4: "no RNG needed for context selection").
    fn context_for_tick(&self, t: u64) -> (AgentId, InteractionRecord) {
        if t.is_multiple_of(2) {
            (self.partner_a, context_a())
        } else {
            (self.partner_b, context_b())
        }
    }

    /// Scripted physical outcome for tick `t` under `schedule`.
    fn outcome_for_tick(&self, t: u64, schedule: Schedule) -> f64 {
        match schedule {
            Schedule::Bound => {
                if t.is_multiple_of(2) {
                    self.cfg.outcome_a
                } else {
                    self.cfg.outcome_b
                }
            }
            Schedule::Reversed => {
                if t.is_multiple_of(2) {
                    self.cfg.outcome_b
                } else {
                    self.cfg.outcome_a
                }
            }
            Schedule::EqualOutcome => self.cfg.equal_outcome,
        }
    }

    /// One protocol tick (plan §4): perceive + forced `Transfer`, then overwrite `organism.energy`
    /// to the scripted outcome (discarding whatever the organism's own real metabolic/transfer
    /// dynamics computed that tick), then learn from that realized (scripted) outcome.
    fn step_with(&mut self, partner: (AgentId, InteractionRecord), outcome: f64) {
        let (_tick, pending) = self.organism.act_social(
            self.cfg.resource_level,
            Some(Action::Transfer.index()),
            Some(partner),
        );
        self.organism.energy = outcome;
        self.organism
            .learn_from_realized_outcome(pending, Some(partner));
    }

    /// Run `ticks` protocol ticks starting at `start_tick` under `schedule`, context always
    /// following tick parity (plan §4/§6 — every control except Shuffled keeps this).
    pub fn run(&mut self, start_tick: u64, ticks: u64, schedule: Schedule) {
        for t in start_tick..(start_tick + ticks) {
            let partner = self.context_for_tick(t);
            let outcome = self.outcome_for_tick(t, schedule);
            self.step_with(partner, outcome);
        }
    }

    /// Shuffled-context control (plan §6): the outcome schedule stays tied to tick parity
    /// (`Schedule::Bound`'s mapping), but which context is *presented* is independently
    /// re-randomized each tick via a simple xorshift (matches `encounter.rs`'s own
    /// `rng_state` style) — breaking any temporal correspondence between context and outcome.
    pub fn run_shuffled(&mut self, start_tick: u64, ticks: u64, rng_state: &mut u64) {
        for t in start_tick..(start_tick + ticks) {
            let mut x = *rng_state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *rng_state = x;
            let partner = if x.is_multiple_of(2) {
                (self.partner_a, context_a())
            } else {
                (self.partner_b, context_b())
            };
            let outcome = self.outcome_for_tick(t, Schedule::Bound);
            self.step_with(partner, outcome);
        }
    }

    /// Held-out check (plan §5): run `ticks` further protocol ticks under the bound schedule,
    /// measuring each tick's physical prediction error on the energy dimension *before* applying
    /// that tick's scripted outcome — and deliberately never calling
    /// `learn_from_realized_outcome`, so the model/`td_learner` stay exactly as already trained.
    /// Returns `(mean_abs_error_context_a, mean_abs_error_context_b)`.
    pub fn held_out_check(&mut self, start_tick: u64, ticks: u64) -> (f64, f64) {
        let mut errors_a = Vec::new();
        let mut errors_b = Vec::new();
        for t in start_tick..(start_tick + ticks) {
            let (partner_id, context) = self.context_for_tick(t);
            let outcome = self.outcome_for_tick(t, Schedule::Bound);
            let predicted_energy = self.predicted_energy_for_context(&context);
            let (_tick, _pending) = self.organism.act_social(
                self.cfg.resource_level,
                Some(Action::Transfer.index()),
                Some((partner_id, context)),
            );
            self.organism.energy = outcome;
            // Deliberately no `learn_from_realized_outcome` call here — the whole point of the
            // held-out phase is that the model/td_learner never update from it (plan §5).
            let error = (predicted_energy - outcome).abs();
            if partner_id == self.partner_a {
                errors_a.push(error);
            } else {
                errors_b.push(error);
            }
        }
        let mean = |v: &[f64]| -> f64 {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };
        (mean(&errors_a), mean(&errors_b))
    }

    /// The counterfactual-sensitivity variant state for `context` (plan §5): dims 0-1 held at the
    /// organism's *current* belief (so the comparison isolates the social dims specifically), dims
    /// 2-5 set to `compress_for_observation` of `context`'s fields with `partner_present = 1.0`.
    fn variant_state(&self, context: &InteractionRecord) -> HiddenState {
        let belief = &self.organism.agent.belief;
        HiddenState {
            mean: vec![
                belief.mean[0],
                belief.mean[1],
                1.0,
                compress_for_observation(context.given_to_partner),
                compress_for_observation(context.received_from_partner),
                compress_for_observation(context.encounter_count as f64),
            ],
            precision: belief.precision.clone(),
            mode_probs: belief.mode_probs.clone(),
            current_mode: belief.current_mode,
        }
    }

    /// The generative model's own predicted energy (dim 1) for `context`, under a forced
    /// `Transfer`, from the organism's current belief — the quantity the held-out check (plan §5)
    /// compares against the scripted actual outcome.
    fn predicted_energy_for_context(&self, context: &InteractionRecord) -> f64 {
        let variant = self.variant_state(context);
        let next = self
            .organism
            .agent
            .model
            .predict_next_state(&variant, Action::Transfer.index());
        let predicted = self.organism.agent.model.predict_observation(&next);
        predicted.get(1).copied().unwrap_or(0.5)
    }

    /// Primary metric (plan §5).
    pub fn counterfactual_reading(&self) -> CounterfactualReading {
        let variant_a = self.variant_state(&context_a());
        let variant_b = self.variant_state(&context_b());
        let next_a = self
            .organism
            .agent
            .model
            .predict_next_state(&variant_a, Action::Transfer.index());
        let next_b = self
            .organism
            .agent
            .model
            .predict_next_state(&variant_b, Action::Transfer.index());
        let pred_a = self.organism.agent.model.predict_observation(&next_a);
        let pred_b = self.organism.agent.model.predict_observation(&next_b);
        let predicted_resource_a = pred_a.first().copied().unwrap_or(0.5);
        let predicted_resource_b = pred_b.first().copied().unwrap_or(0.5);
        let predicted_energy_a = pred_a.get(1).copied().unwrap_or(0.5);
        let predicted_energy_b = pred_b.get(1).copied().unwrap_or(0.5);
        CounterfactualReading {
            predicted_resource_a,
            predicted_resource_b,
            predicted_energy_a,
            predicted_energy_b,
            delta_predicted: (predicted_resource_a - predicted_resource_b).abs()
                + (predicted_energy_a - predicted_energy_b).abs(),
        }
    }

    /// Secondary, more granular measurement (plan §5): the raw learned coefficients
    /// `transition_matrices[Transfer][j][i]` for `j` in the four social dims (2-5, indexed 0-3
    /// here), `i` in the two physical dims (0-1, resource/energy).
    pub fn raw_social_to_physical_coefficients(&self) -> [[f64; 2]; 4] {
        let matrix = &self.organism.agent.model.transition_matrices[Action::Transfer.index()];
        let mut out = [[0.0; 2]; 4];
        for (row, j) in (2..6).enumerate() {
            for i in 0..2 {
                out[row][i] = matrix[j][i];
            }
        }
        out
    }

    /// MA-001L integration (`ALIFE_MA001L_CONTEXTUAL_TRANSITION_LEARNING_PLAN_2026-07-26.md` §9
    /// step 5): the realized post-tick state `DeltaRuleLearner::update` learns from -- the
    /// counterpart to `variant_state` (which holds belief fixed and varies context for the
    /// counterfactual *query*). Matches what `Organism::learn_phase`'s own `actual_obs` would
    /// construct: the true environment resource level, the just-overwritten energy outcome, and
    /// `context`'s compressed social fields with `partner_present = 1.0`.
    fn realized_state(&self, context: &InteractionRecord, outcome: f64) -> HiddenState {
        HiddenState {
            mean: vec![
                self.cfg.resource_level,
                outcome,
                1.0,
                compress_for_observation(context.given_to_partner),
                compress_for_observation(context.received_from_partner),
                compress_for_observation(context.encounter_count as f64),
            ],
            precision: vec![1.0; 6],
            mode_probs: vec![1.0],
            current_mode: 0,
        }
    }

    /// One protocol tick, learning via `delta_rule` instead of the organism's own Hebbian/TD
    /// pathway. **Caller must already have called `set_learning_pathway(false, false)`** so the
    /// organism's own model-learning is fully disabled (not re-checked here) -- only
    /// `delta_rule.update` then touches `transition_matrices`. `old_state` is the organism's
    /// *real, evolving* belief just before this tick's own perception runs -- unlike MA-001L's
    /// fixed synthetic placeholder, this reflects however the organism's belief has actually
    /// evolved from real perception so far, the central difference this integration exists to
    /// test.
    fn step_with_delta_rule(
        &mut self,
        delta_rule: &DeltaRuleLearner,
        partner: (AgentId, InteractionRecord),
        outcome: f64,
    ) {
        let old_state = self.organism.agent.belief.clone();
        let (_tick, pending) = self.organism.act_social(
            self.cfg.resource_level,
            Some(Action::Transfer.index()),
            Some(partner),
        );
        self.organism.energy = outcome;
        // Still runs learn_from_realized_outcome for its belief-tracking perceive() -- the
        // organism's own model-learning is disabled by the caller, so this does not touch
        // transition_matrices; only delta_rule.update below does.
        self.organism
            .learn_from_realized_outcome(pending, Some(partner));
        let realized = self.realized_state(&partner.1, outcome);
        delta_rule.update(
            &mut self.organism.agent.model,
            &old_state,
            Action::Transfer.index(),
            &realized,
        );
    }

    /// `run`'s delta-rule counterpart.
    pub fn run_with_delta_rule(
        &mut self,
        delta_rule: &DeltaRuleLearner,
        start_tick: u64,
        ticks: u64,
        schedule: Schedule,
    ) {
        for t in start_tick..(start_tick + ticks) {
            let partner = self.context_for_tick(t);
            let outcome = self.outcome_for_tick(t, schedule);
            self.step_with_delta_rule(delta_rule, partner, outcome);
        }
    }

    /// `run_shuffled`'s delta-rule counterpart.
    pub fn run_shuffled_with_delta_rule(
        &mut self,
        delta_rule: &DeltaRuleLearner,
        start_tick: u64,
        ticks: u64,
        rng_state: &mut u64,
    ) {
        for t in start_tick..(start_tick + ticks) {
            let mut x = *rng_state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *rng_state = x;
            let partner = if x.is_multiple_of(2) {
                (self.partner_a, context_a())
            } else {
                (self.partner_b, context_b())
            };
            let outcome = self.outcome_for_tick(t, Schedule::Bound);
            self.step_with_delta_rule(delta_rule, partner, outcome);
        }
    }

    /// MA-001L §12's diagnosed fix, second pass: identical to [`Self::step_with_delta_rule`]
    /// except `old_state` is built from `tick.gated_observation` (the exact, single-tick,
    /// instantaneous observation `perceive()` was actually driven by this tick) instead of
    /// `self.organism.agent.belief` (a running, inertial estimate blended across many ticks,
    /// diagnosed as the cause of §12's sign-flip failure). Kept as a separate method rather than
    /// replacing `step_with_delta_rule` so both mechanisms can be compared side by side, not just
    /// described in prose.
    fn step_with_delta_rule_from_observation(
        &mut self,
        delta_rule: &DeltaRuleLearner,
        partner: (AgentId, InteractionRecord),
        outcome: f64,
    ) {
        let (tick, pending) = self.organism.act_social(
            self.cfg.resource_level,
            Some(Action::Transfer.index()),
            Some(partner),
        );
        let old_state = HiddenState {
            mean: tick.gated_observation,
            precision: vec![1.0; 6],
            mode_probs: vec![1.0],
            current_mode: 0,
        };
        self.organism.energy = outcome;
        self.organism
            .learn_from_realized_outcome(pending, Some(partner));
        let realized = self.realized_state(&partner.1, outcome);
        delta_rule.update(
            &mut self.organism.agent.model,
            &old_state,
            Action::Transfer.index(),
            &realized,
        );
    }

    /// `run_with_delta_rule`'s observation-based counterpart (plan §12's diagnosed fix).
    pub fn run_with_delta_rule_from_observation(
        &mut self,
        delta_rule: &DeltaRuleLearner,
        start_tick: u64,
        ticks: u64,
        schedule: Schedule,
    ) {
        for t in start_tick..(start_tick + ticks) {
            let partner = self.context_for_tick(t);
            let outcome = self.outcome_for_tick(t, schedule);
            self.step_with_delta_rule_from_observation(delta_rule, partner, outcome);
        }
    }

    /// `run_shuffled_with_delta_rule`'s observation-based counterpart.
    pub fn run_shuffled_with_delta_rule_from_observation(
        &mut self,
        delta_rule: &DeltaRuleLearner,
        start_tick: u64,
        ticks: u64,
        rng_state: &mut u64,
    ) {
        for t in start_tick..(start_tick + ticks) {
            let mut x = *rng_state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *rng_state = x;
            let partner = if x.is_multiple_of(2) {
                (self.partner_a, context_a())
            } else {
                (self.partner_b, context_b())
            };
            let outcome = self.outcome_for_tick(t, Schedule::Bound);
            self.step_with_delta_rule_from_observation(delta_rule, partner, outcome);
        }
    }

    /// MA-001L §13's disclosed, untested hypothesis: identical to
    /// [`Self::step_with_delta_rule_from_observation`] except `old_state` is built from
    /// `tick.raw_observation` (the observation *before* `self.boundary.gate_observation`'s
    /// permeability-based attenuation) instead of `tick.gated_observation`. Tests whether
    /// blanket-permeability attenuation (a function of the organism's physiological deficit --
    /// a real mechanism MA-001L's idealized synthetic tuples never had) rather than belief
    /// inertia is what's limiting the observation-based fix's effect size. Kept as a separate
    /// method rather than replacing `step_with_delta_rule_from_observation` so all three
    /// mechanisms (belief, gated observation, raw observation) can be compared side by side.
    fn step_with_delta_rule_from_raw_observation(
        &mut self,
        delta_rule: &DeltaRuleLearner,
        partner: (AgentId, InteractionRecord),
        outcome: f64,
    ) {
        let (tick, pending) = self.organism.act_social(
            self.cfg.resource_level,
            Some(Action::Transfer.index()),
            Some(partner),
        );
        let old_state = HiddenState {
            mean: tick.raw_observation,
            precision: vec![1.0; 6],
            mode_probs: vec![1.0],
            current_mode: 0,
        };
        self.organism.energy = outcome;
        self.organism
            .learn_from_realized_outcome(pending, Some(partner));
        let realized = self.realized_state(&partner.1, outcome);
        delta_rule.update(
            &mut self.organism.agent.model,
            &old_state,
            Action::Transfer.index(),
            &realized,
        );
    }

    /// `run_with_delta_rule_from_observation`'s raw-observation counterpart (plan §13's disclosed
    /// hypothesis).
    pub fn run_with_delta_rule_from_raw_observation(
        &mut self,
        delta_rule: &DeltaRuleLearner,
        start_tick: u64,
        ticks: u64,
        schedule: Schedule,
    ) {
        for t in start_tick..(start_tick + ticks) {
            let partner = self.context_for_tick(t);
            let outcome = self.outcome_for_tick(t, schedule);
            self.step_with_delta_rule_from_raw_observation(delta_rule, partner, outcome);
        }
    }

    /// `run_shuffled_with_delta_rule_from_observation`'s raw-observation counterpart.
    pub fn run_shuffled_with_delta_rule_from_raw_observation(
        &mut self,
        delta_rule: &DeltaRuleLearner,
        start_tick: u64,
        ticks: u64,
        rng_state: &mut u64,
    ) {
        for t in start_tick..(start_tick + ticks) {
            let mut x = *rng_state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *rng_state = x;
            let partner = if x.is_multiple_of(2) {
                (self.partner_a, context_a())
            } else {
                (self.partner_b, context_b())
            };
            let outcome = self.outcome_for_tick(t, Schedule::Bound);
            self.step_with_delta_rule_from_raw_observation(delta_rule, partner, outcome);
        }
    }

    /// Methodology-check control (2026-07-26 follow-up): tests whether `shuffled_collapses`'s own
    /// comparison baseline (`equal_post.delta_predicted`) is a fair "no coupling" reference for
    /// judging the shuffled-context control. Equal-outcome's target is *constant* (0.6 every
    /// tick), which the model converges to trivially with near-zero residual coefficient movement.
    /// Shuffled's target *alternates* (0.9/0.2 by tick parity, exactly like Bound) -- a
    /// structurally harder, higher-variance target regardless of whether context correlates with
    /// it at all. This control isolates that variable: outcome alternates by tick parity exactly
    /// like Bound/Shuffled, but context alternates on a **period-4 block schedule** (A,A,B,B,
    /// repeat -- switches every 2 ticks) instead of tick parity or per-tick randomization. Over
    /// each 4-tick cycle this gives both contexts an *exactly balanced* mean outcome (each
    /// sees {0.9, 0.2}, mean 0.55) -- a genuinely decorrelated-but-fully-varying exposure, unlike
    /// an earlier draft of this control (fixed always-Context-B) which was mechanistically
    /// degenerate: Context B's own social fields are all zero except `partner_present`, so
    /// `given`/`received`/`count`'s gradient (`error * old_state[j]`) was exactly zero on every
    /// tick, and the one coefficient that *could* move (`partner_present`, identical across both
    /// counterfactual queries in `variant_state`) cancels out symmetrically in `delta_predicted`
    /// by construction -- that version tested nothing. If this corrected control's own
    /// post-training `delta_predicted` is large (comparable to Shuffled's), that shows
    /// `shuffled_collapses` was failing due to an unfair baseline mismatch (alternating-target
    /// difficulty alone), not genuine residual coupling.
    pub fn run_with_delta_rule_from_raw_observation_balanced_decorrelated(
        &mut self,
        delta_rule: &DeltaRuleLearner,
        start_tick: u64,
        ticks: u64,
    ) {
        for t in start_tick..(start_tick + ticks) {
            let outcome = self.outcome_for_tick(t, Schedule::Bound);
            let partner = if (t / 2) % 2 == 0 {
                (self.partner_a, context_a())
            } else {
                (self.partner_b, context_b())
            };
            self.step_with_delta_rule_from_raw_observation(delta_rule, partner, outcome);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn social_cfg() -> OrganismConfig {
        OrganismConfig {
            social_enabled: true,
            ..OrganismConfig::default()
        }
    }

    #[test]
    fn context_for_tick_alternates_deterministically() {
        let probe = Ma001rProbe::new(social_cfg(), 1, Ma001rConfig::default());
        assert_eq!(probe.context_for_tick(0).1, context_a());
        assert_eq!(probe.context_for_tick(1).1, context_b());
        assert_eq!(probe.context_for_tick(2).1, context_a());
        assert_ne!(probe.context_for_tick(0).0, probe.context_for_tick(1).0);
    }

    #[test]
    fn bound_and_reversed_schedules_swap_outcome_mapping() {
        let probe = Ma001rProbe::new(social_cfg(), 1, Ma001rConfig::default());
        assert_eq!(probe.outcome_for_tick(0, Schedule::Bound), 0.9);
        assert_eq!(probe.outcome_for_tick(1, Schedule::Bound), 0.2);
        assert_eq!(probe.outcome_for_tick(0, Schedule::Reversed), 0.2);
        assert_eq!(probe.outcome_for_tick(1, Schedule::Reversed), 0.9);
        assert_eq!(probe.outcome_for_tick(0, Schedule::EqualOutcome), 0.6);
        assert_eq!(probe.outcome_for_tick(1, Schedule::EqualOutcome), 0.6);
    }

    #[test]
    fn set_learning_pathway_toggles_td_learner_presence() {
        let mut probe = Ma001rProbe::new(social_cfg(), 1, Ma001rConfig::default());
        assert!(
            probe.organism.agent.td_learner.is_some(),
            "default has TD active"
        );
        probe.set_learning_pathway(false, false);
        assert!(probe.organism.agent.td_learner.is_none());
        assert!(!probe.organism.agent.config.enable_model_learning);
        probe.set_learning_pathway(false, true);
        assert!(probe.organism.agent.td_learner.is_some());
        assert!(!probe.organism.agent.config.enable_model_learning);
    }

    #[test]
    fn a_short_run_does_not_panic_and_moves_energy_to_scripted_values() {
        let mut probe = Ma001rProbe::new(social_cfg(), 1, Ma001rConfig::default());
        probe.run(0, 10, Schedule::Bound);
        // After an even number of ticks ending on an odd final tick index (t=9, Context B),
        // energy should sit at outcome_b.
        assert!((probe.organism.energy - 0.2).abs() < 1e-9);
    }

    #[test]
    fn delta_rule_integration_runs_without_panicking_and_moves_a_coefficient() {
        use crate::ma001l::{DeltaRuleConfig, DeltaRuleLearner};
        let mut probe = Ma001rProbe::new(social_cfg(), 1, Ma001rConfig::default());
        probe.set_learning_pathway(false, false);
        let delta_rule =
            DeltaRuleLearner::new(DeltaRuleConfig::default(), &probe.organism.agent.model);
        let before = probe.raw_social_to_physical_coefficients();
        probe.run_with_delta_rule(&delta_rule, 0, 200, Schedule::Bound);
        let after = probe.raw_social_to_physical_coefficients();
        assert_ne!(
            before, after,
            "delta rule should move at least one social->physical coefficient over 200 real-organism ticks"
        );
        let reading = probe.counterfactual_reading();
        assert!(reading.delta_predicted.is_finite());
    }

    #[test]
    fn counterfactual_reading_is_computable_before_any_training() {
        let probe = Ma001rProbe::new(social_cfg(), 1, Ma001rConfig::default());
        let reading = probe.counterfactual_reading();
        assert!(reading.delta_predicted.is_finite());
    }

    #[test]
    fn raw_coefficients_start_at_generic_initialization_and_shape_is_4x2() {
        let probe = Ma001rProbe::new(social_cfg(), 1, Ma001rConfig::default());
        let coeffs = probe.raw_social_to_physical_coefficients();
        assert_eq!(coeffs.len(), 4);
        for row in coeffs.iter() {
            assert_eq!(row.len(), 2);
        }
    }
}
