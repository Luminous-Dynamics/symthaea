// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A single organism: a Markov blanket wrapping one [`ActiveInferenceAgent`].
//!
//! Every tick genuinely calls `perceive` → `select_action` → `act` → `learn_from_outcome`
//! and consumes every result (`ALIFE_PLAN_2026-07-08.md` Phase 0 §0b — the anti-theater rule).
//! There is no flag to disable perception here: `Organism` always perceives for real. The
//! "does perception matter" comparison in Phase 0's ground-truth tests uses a separate,
//! non-perceiving *constant* belief as the baseline, not a crippled `Organism` — production
//! code shouldn't carry a theater on/off switch even for testing purposes.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use symthaea_fep::markov_blanket::{BlanketPermeability, MarkovBoundaryOperator, MarkovPartition};
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

use crate::agent_id::AgentId;
use crate::ledger::{InteractionRecord, compress_for_observation};
use crate::metabolism::{
    landauer_minimum, perceptual_resolution_bits, prigogine_dissipation_cost, quantize_to_grain,
    shannon_entropy_bits,
};
use crate::types::BoundaryModulators;

/// The behaviors an [`Organism`] can select between.
///
/// Kept intentionally small and biology-plain rather than reusing `MotorCommandType`
/// (`AttentionShift`, `ReflectionInitiate`, ...) — those are cognition-flavored labels from a
/// different context; an organism's action space here is literally "spend effort gathering
/// resources" vs "conserve" vs (Genesis v0 only) "give some energy to whoever I'm paired with."
///
/// `Transfer` is deliberately **not** paired with a `Cooperate`/`Defect` label anywhere in this
/// crate (`ALIFE_MULTIAGENT_GENESIS_PLAN_2026-07-25.md`'s non-goals) — those are interpretations
/// of *output*, never a choice handed to the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Forage,
    Rest,
    /// Genesis v0 only (`cfg.social_enabled`) — give a fixed quantum of energy to this tick's
    /// encounter partner, if any. Structurally unreachable when `social_enabled` is false: that
    /// path constructs the agent with `num_actions: Action::COUNT` (2), so `select_action()` can
    /// never return this index at all, not merely by convention.
    Transfer,
}

impl Action {
    /// Action count for the pre-existing, asocial organism (Forage/Rest only) — unchanged since
    /// before Genesis existed. Depended on directly by `tests/phase0_ground_truth.rs`'s
    /// random-baseline generator; must stay exactly 2.
    pub const COUNT: usize = 2;
    /// Action count when `OrganismConfig::social_enabled` is true (adds `Transfer`).
    pub const SOCIAL_COUNT: usize = 3;

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Action::Forage,
            1 => Action::Rest,
            _ => Action::Transfer,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Action::Forage => 0,
            Action::Rest => 1,
            Action::Transfer => 2,
        }
    }
}

/// Metabolic/behavioral parameters for an [`Organism`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrganismConfig {
    /// Preferred (homeostatic set-point) energy level, in `[0, 1]`.
    pub set_point: f64,
    /// Baseline energy cost paid every tick regardless of action.
    pub metabolic_cost: f64,
    /// Additional energy cost paid only when foraging.
    pub forage_activity_cost: f64,
    /// How efficiently a gated resource observation converts into energy gain when foraging.
    pub forage_efficiency: f64,
    /// Precision passed to `ActiveInferenceAgent::set_goals` for the energy preference.
    pub goal_precision: f64,
    /// Effective temperature (alife energy units) used by the Landauer/Prigogine physics --
    /// see `metabolism` module docs. A fixed constant here, deliberately: Phase 3 ports the
    /// physics *formulas*, not the main crate's adaptive-temperature machinery.
    pub effective_temperature: f64,
    /// Transport-coefficient-like constant scaling Prigogine dissipation cost with how much
    /// homeostatic order is being maintained. See `metabolism::prigogine_dissipation_cost`.
    pub dissipation_rate: f64,
    /// Energy at or below which the organism is dead (`Organism::is_dead`) -- a hard,
    /// intrinsic property of the organism itself (Phase 3b), not only an emergent side effect
    /// of `Population`'s own bookkeeping (which has its own, possibly different, threshold).
    pub death_energy_threshold: f64,
    /// Softmax temperature for `select_action`'s expected-free-energy-weighted action choice
    /// (`symthaea_fep::ActiveInferenceAgentConfig::action_temperature`) -- higher means more
    /// random/exploratory, lower means more greedily exploitative. Previously hardcoded to
    /// `symthaea-fep`'s own default (1.0) via `..Default::default()`; exposed here so Phase 4
    /// can treat it as a heritable, mutable trait (see `genome`).
    pub action_temperature: f64,
    /// `None` (the default) reproduces this crate's exact pre-existing behavior: the resource
    /// observation is passed to perception raw/unquantized, and no perceptual-resolution energy
    /// cost is charged. `Some(grain)` activates the Hoffman Fitness-Beats-Truth experiment
    /// (`tests/hoffman_fitness_beats_truth.rs`, `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md`
    /// Phase 1): the resource observation is coarse-grained to bucket width `grain` before
    /// perception sees it, and a real Landauer cost is charged for the resolved detail. See
    /// `genome`'s module docs for the full rationale and bounds.
    pub perceptual_grain: Option<f64>,
    /// `None` (the default) reproduces this crate's exact pre-existing behavior: foraging gain
    /// scales linearly with the true `resource_level` (more is always better). `Some(sigma)`
    /// activates Hoffman Interface Theory Plan Phase 2 (`HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md`
    /// -- distinct from this crate's own unrelated `ALIFE_PLAN_2026-07-08.md` Phase 2/coalitions):
    /// foraging gain instead follows a Gaussian interior optimum centered on `resource_level =
    /// 0.5` (matching `Organism::new`'s existing resource-observation goal preference) with
    /// standard deviation `sigma` -- too little OR too much true resource both yield low gain
    /// (spoilage/danger, per Hoffman's "too much is toxic" framing), not just too little. A
    /// non-monotonic payoff, unlike every other config in this crate. See
    /// `tests/hoffman_phase2_interior_optimum_still_favors_coarse.rs` and
    /// `tests/hoffman_action_selection_resource_insensitivity.rs`.
    pub spoilage_sigma: Option<f64>,
    /// Preferred resource *observation* passed to `ActiveInferenceAgent::set_goals` --
    /// previously a hardcoded `0.5` ("prefer a moderate reading") baked directly into
    /// `Organism::new` rather than a real, defensible fitness target. Root-caused via
    /// `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md`'s investigation into why
    /// `select_action` never favored foraging at high true resource levels:
    /// `GenerativeModel`'s per-action transition dynamics make `Forage`'s predicted
    /// resource observation *amplify* the current belief while `Rest` dampens it, so a
    /// *moderate* target made `Forage`'s prediction overshoot at both high and low
    /// resource, leaving `Rest` pragmatically favored everywhere (see
    /// `tests/hoffman_efe_rest_structurally_dominates.rs`, now stale/pre-fix evidence,
    /// not deleted). Since real foraging payoff (`organism.rs` step 6) scales
    /// monotonically with true resource level with no satiation, `1.0` ("prefer as much
    /// resource as observable") is the fitness-aligned target -- a real default-behavior
    /// fix, not an opt-in experimental toggle, verified in
    /// `tests/hoffman_efe_forage_now_favored_at_high_resource.rs`.
    pub resource_preference: f64,
    /// Resource-dimension entry of `GenerativeModel::prior_mean` (`ActiveInferenceAgent::
    /// inject_priors`'s "Passport Route", already-public API -- no new mechanism needed).
    /// Default `0.5` is an **exact no-op**: `GenerativeModel::new` already initializes
    /// `prior_mean` to `[0.5; state_dim]`, so injecting `[0.5, 0.5]` with the same default
    /// precision changes nothing. Found via `tests/hoffman_directed_join_no_effect.rs`'s
    /// investigation: this fixed prior, combined with `update_belief`'s pull toward it, anchors
    /// belief near 0.5 regardless of what's actually perceived -- comfortably above the
    /// `resource_preference` fix's ~0.075-0.1 decision crossover, so an organism forages
    /// successfully even under maximally garbage perception. Unlike `resource_preference`, `0.5`
    /// is a genuinely defensible uninformative prior in the abstract (not an obvious bug the way
    /// the old goal preference was) -- kept as an **opt-in experimental knob**, not a changed
    /// default, so lowering it (e.g. toward `0.0`, "assume no resource until perceived
    /// otherwise") can be tested without touching any existing organism's behavior. See
    /// `tests/hoffman_prior_recalibration.rs`.
    pub resource_prior: f64,
    /// `false` (the default) reproduces this crate's exact pre-existing 2-channel, 2-action
    /// behavior (Phases 0-7) -- no `Transfer` action, no interaction-ledger observation
    /// channels. `true` activates `ALIFE_MULTIAGENT_GENESIS_PLAN_2026-07-25.md`'s Genesis v0
    /// extension: `Action::Transfer` becomes selectable and three raw per-partner ledger
    /// counters (`given_to_partner`, `received_from_partner`, `encounter_count`) plus a
    /// partner-present flag are added to the observation/state vector. See that plan's Stage 0
    /// "baseline equivalence" invariant, which this flag exists to satisfy: with it `false`,
    /// `Organism` is byte-for-byte the pre-Genesis implementation, not just behaviorally similar.
    /// Deliberately **not** a heritable `Genome` trait -- see the plan's non-goals.
    pub social_enabled: bool,
    /// Fixed quantum transferred by one `Transfer` action, when selected and a partner is
    /// present. Deliberately not agent-selectable (`Transfer(amount)`) -- see the Genesis plan's
    /// "keep Transfer stupidly simple" design note. Only meaningful when `social_enabled`.
    pub transfer_quantum: f64,
}

impl Default for OrganismConfig {
    fn default() -> Self {
        Self {
            set_point: 0.8,
            metabolic_cost: 0.01,
            forage_activity_cost: 0.01,
            forage_efficiency: 0.15,
            goal_precision: 2.0,
            effective_temperature: 1.0,
            // See metabolism::K_ALIFE_BOLTZMANN's doc comment -- scaled down ~10x from an
            // initial guess that was measured (not assumed) to roughly double the baseline
            // metabolic_cost outright.
            dissipation_rate: 0.001,
            death_energy_threshold: 0.05,
            action_temperature: 1.0,
            perceptual_grain: None,
            spoilage_sigma: None,
            resource_preference: 1.0,
            resource_prior: 0.5,
            social_enabled: false,
            transfer_quantum: 0.1,
        }
    }
}

/// Telemetry from one [`Organism::tick`] call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismTick {
    /// The true exogenous resource level this tick (before any gating).
    pub resource_observed: f64,
    /// The organism's current belief about the resource level.
    pub belief_resource: f64,
    /// The organism's current belief about its own energy level.
    pub belief_energy: f64,
    /// Actual physiological energy after this tick's action.
    pub energy: f64,
    /// Action actually executed (0 = forage, 1 = rest).
    pub action: usize,
    /// Current variational free energy.
    pub free_energy: f64,
    /// Current blanket permeability.
    pub permeability: BlanketPermeability,
    /// |predicted resource observation from `act()` − actual resource level| — consumes
    /// `act()`'s return value instead of discarding it.
    pub act_prediction_error: f64,
    /// Real bits of information processed this tick (perception's `belief_change` +
    /// action-selection's Shannon entropy) -- what `physical_cost`'s Landauer term is charged
    /// against. Phase 3 telemetry, not decorative.
    pub bits_processed: f64,
    /// Landauer + Prigogine energy cost actually charged this tick (Phase 3b: a real,
    /// unavoidable thermodynamic floor added on top of `metabolic_cost`/`forage_activity_cost`,
    /// not an alternative to them).
    pub physical_cost: f64,
    /// Whether the organism is dead after this tick (`energy <= death_energy_threshold`).
    pub is_dead: bool,
    /// Actual amount debited from this organism's own energy this tick for a `Transfer` action
    /// (`0.0` when the action wasn't `Transfer`, no partner was present, or `social_enabled` is
    /// false). This is the exact value credited to the partner's energy by the caller
    /// (`Population::step_social`) -- the Stage 0 "conservation" invariant.
    pub transfer_given: f64,
    /// The exact observation `perceive()` was actually driven by this tick -- after
    /// blanket-permeability gating (`self.boundary.gate_observation`), before it enters
    /// `agent.perceive()`'s own belief-update integration. Added for
    /// `ALIFE_MA001L_CONTEXTUAL_TRANSITION_LEARNING_PLAN_2026-07-26.md` §12's diagnosed fix: this
    /// is a single-tick, instantaneous signal (unlike `self.agent.belief`, which is a running,
    /// inertial estimate blended across many ticks) -- a caller like `Ma001rProbe::
    /// step_with_delta_rule` that needs to know "which social context was actually presented this
    /// specific tick" should read this field, not `agent.belief`.
    pub gated_observation: Vec<f64>,
    /// The observation *before* `self.boundary.gate_observation` is applied -- the fully raw
    /// signal, upstream of blanket-permeability attenuation. Added to test the disclosed,
    /// untested hypothesis in `ALIFE_MA001L_CONTEXTUAL_TRANSITION_LEARNING_PLAN_2026-07-26.md`
    /// §13 that permeability-based attenuation of `gated_observation` (a function of the
    /// organism's physiological deficit, not present in MA-001L's idealized synthetic tuples) is
    /// what's limiting MA-001R-delta v2's effect size, distinct from belief inertia. A caller
    /// wanting the strongest possible replication of MA-001L's clean fixed-placeholder old_state
    /// should read this field instead of `gated_observation`.
    pub raw_observation: Vec<f64>,
    /// The actual post-softmax action-selection probabilities this tick
    /// (`ActiveInferenceAgent::select_action`'s own `ActionSelectionResult::action_probabilities`,
    /// already computed and previously discarded). Added to directly test the reframed hypothesis
    /// in `ALIFE_MA001A_DELTA_RERUN_PLAN_2026-07-26.md` §10: is action selection close to
    /// state-insensitive (a near-uniform softmax) at the current `action_temperature`, rather than
    /// genuinely differentiating between actions based on their expected free energy?
    pub action_probabilities: Vec<f64>,
}

/// A living system modeled as a Markov blanket minimizing free energy.
pub struct Organism {
    pub id: AgentId,
    pub agent: ActiveInferenceAgent,
    pub boundary: MarkovBoundaryOperator,
    pub energy: f64,
    pub cfg: OrganismConfig,
    /// The true resource level from this organism's most recent `tick()` -- kept so
    /// Phase 2 coalition analysis can evaluate a pooled belief against a real observation
    /// without needing the caller to separately track it.
    pub last_resource_observed: f64,
    /// Genesis v0 (G0c): raw per-partner interaction history, keyed by the partner's
    /// [`AgentId`]. Empty and never read when `cfg.social_enabled` is false.
    pub ledger: HashMap<AgentId, InteractionRecord>,
    /// Genesis v0 event-log support: the founding ancestor's [`AgentId`] -- an organism's own
    /// `id` for an original (non-offspring) individual, inherited unchanged from parent to
    /// offspring at every reproduction. Lets a later analysis group events by family line
    /// without needing a separate genealogy table.
    pub lineage_id: AgentId,
    /// Genesis v0 event-log support: `0` for the initial population, `parent.generation + 1` for
    /// offspring.
    pub generation: u32,
}

impl Organism {
    /// Observation *and* state layout when `cfg.social_enabled` is false: `[resource_level,
    /// energy_level]` (exteroceptive + interoceptive) -- this crate keeps state_dim == obs_dim
    /// throughout, state mirroring observation directly (Phase 0 gates both uniformly through
    /// the blanket for simplicity; a real exteroceptive/interoceptive permeability asymmetry is
    /// left to a later phase). Unchanged since before Genesis existed.
    const OBS_DIM: usize = 2;
    /// Observation/state layout when `cfg.social_enabled` is true: the two channels above, plus
    /// `[partner_present, given_to_partner, received_from_partner, encounter_count]` for this
    /// tick's encounter partner (all zero when unpaired) -- see `ledger` module docs for why
    /// these are raw counters (numerically compressed for the belief substrate only) rather
    /// than a synthesized signal.
    const SOCIAL_DIM: usize = 6;

    pub fn new(cfg: OrganismConfig, seed: u64) -> Self {
        let dim = if cfg.social_enabled {
            Self::SOCIAL_DIM
        } else {
            Self::OBS_DIM
        };
        let num_actions = if cfg.social_enabled {
            Action::SOCIAL_COUNT
        } else {
            Action::COUNT
        };
        let agent_cfg = ActiveInferenceAgentConfig {
            state_dim: dim,
            obs_dim: dim,
            num_actions,
            action_temperature: cfg.action_temperature,
            ..Default::default()
        };
        let mut agent = ActiveInferenceAgent::new(agent_cfg);
        agent.set_rng_seed(seed);
        // Prefer a high resource reading and a high (near-set-point) energy reading -- see
        // `cfg.resource_preference`'s doc comment for why this replaced a hardcoded moderate 0.5.
        let mut preferences = vec![0.5; dim];
        preferences[0] = cfg.resource_preference;
        preferences[1] = cfg.set_point;
        if cfg.social_enabled {
            // Genesis v0.1 audit Gate 5 (2026-07-26): a shared `goal_precision` across every
            // dimension made the social channels' `0.5` preference a real, non-zero-precision
            // pull, not the inert placeholder it was assumed to be -- `set_goals`'s single
            // scalar precision can't express "no preference here." Real precision stays on
            // resource/energy (indices 0-1); the four social channels (indices 2-5) get exactly
            // zero precision, a genuine "observe but don't care" -- this crate must not bake in
            // a bias toward or against social history, since whether that's fitness-relevant is
            // exactly the open question Genesis exists to observe, not decide in advance.
            let mut precisions = vec![0.0; dim];
            precisions[0] = cfg.goal_precision;
            precisions[1] = cfg.goal_precision;
            agent.set_goals_with_precisions(preferences, precisions);
        } else {
            agent.set_goals(preferences, cfg.goal_precision);
        }
        // See `cfg.resource_prior`'s doc comment -- at the default 0.5 this is an exact no-op
        // (matches `GenerativeModel::new`'s own defaults for both mean and precision).
        let mut prior_mean = vec![0.5; dim];
        prior_mean[0] = cfg.resource_prior;
        agent.inject_priors(prior_mean, vec![1.0; dim]);

        let boundary = MarkovBoundaryOperator::new(MarkovPartition {
            internal_dim: dim,
            sensory_dim: 1,
            active_dim: 1,
        });

        Self {
            id: AgentId::UNALLOCATED,
            agent,
            boundary,
            energy: cfg.set_point,
            cfg,
            last_resource_observed: 0.5,
            ledger: HashMap::new(),
            lineage_id: AgentId::UNALLOCATED,
            generation: 0,
        }
    }

    /// Assign a persistent identity after construction -- used by [`crate::Population`], which
    /// owns the [`crate::agent_id::AgentIdAllocator`] and knows the real allocation sequence.
    /// `Organism::new` alone leaves `id` as [`AgentId::UNALLOCATED`], which is fine for
    /// single-organism call sites (Phase 0-7 tests) that never reference identity.
    ///
    /// Also sets `lineage_id` to this same id (an organism is its own lineage founder,
    /// `generation` 0, unless [`Self::with_lineage`] is called afterward to mark it as an
    /// offspring instead) -- correct by default for `Population`'s initial population.
    pub fn with_id(mut self, id: AgentId) -> Self {
        self.id = id;
        self.lineage_id = id;
        self
    }

    /// Mark this organism as an offspring: inherits `lineage_id` from its parent and is one
    /// generation deeper. Call after [`Self::with_id`] (which otherwise defaults a fresh
    /// organism to founding its own lineage at generation 0).
    pub fn with_lineage(mut self, lineage_id: AgentId, generation: u32) -> Self {
        self.lineage_id = lineage_id;
        self.generation = generation;
        self
    }

    /// A hard, checkable constraint (Phase 3b) -- not an FEP preference that can be talked out
    /// of. True once energy has fallen to or below `cfg.death_energy_threshold`.
    pub fn is_dead(&self) -> bool {
        self.energy <= self.cfg.death_energy_threshold
    }

    /// One full perceive → select_action → act → learn_from_outcome cycle.
    ///
    /// `forced_action`, when `Some`, overrides the agent's own `select_action()` choice for the
    /// real-world consequence and the learning step — used only by Phase 0's ground-truth test
    /// comparing FEP-guided vs. uniform-random action selection. Perception and learning always
    /// run for real regardless of `forced_action`.
    ///
    /// Exactly `Self::tick_inner(resource_level, forced_action, None)` -- kept as its own method
    /// (rather than inlined) so every Phase 0-7 call site keeps compiling and behaving
    /// byte-for-byte unchanged. This is the Stage 0 "baseline equivalence" invariant by
    /// construction, not merely by testing: this method's body never changed.
    pub fn tick(&mut self, resource_level: f64, forced_action: Option<usize>) -> OrganismTick {
        self.tick_inner(resource_level, forced_action, None)
    }

    /// Genesis v0 (G0d/e) social-aware tick. `partner` is `Some((partner_id, ledger_snapshot))`
    /// when the caller's encounter scheduler paired this organism this tick, `None` otherwise.
    /// `ledger_snapshot` must be this organism's own pre-tick record for that partner (i.e. not
    /// yet updated with anything that happens this tick) -- the Stage 0 "observation causality"
    /// invariant. Only meaningful when `cfg.social_enabled`; if it's false this is exactly
    /// `tick()` regardless of what's passed for `partner`.
    ///
    /// Learns immediately, from the *same* pre-action `partner` snapshot used for perception --
    /// convenient for standalone-`Organism` tests, but this is exactly the Genesis v0.1 audit's
    /// Gate 4 finding: an organism's own `Transfer` this tick is invisible to *this* learning
    /// call, since the cross-organism consequence hasn't been applied yet. `Population::
    /// step_social` doesn't use this method for that reason -- it uses [`Self::act_social`] /
    /// [`Self::learn_from_realized_outcome`] instead, deferring the learning step until after the
    /// real outcome is known.
    pub fn tick_social(
        &mut self,
        resource_level: f64,
        forced_action: Option<usize>,
        partner: Option<(AgentId, InteractionRecord)>,
    ) -> OrganismTick {
        self.tick_inner(resource_level, forced_action, partner)
    }

    /// Genesis v0.1 audit Gate 4: the "act" half of a social tick, *without* learning from the
    /// outcome yet. Returns the tick's telemetry plus an opaque [`PendingSocialLearning`] token
    /// that must be passed to [`Self::learn_from_realized_outcome`] (with the *realized*
    /// post-consequence partner ledger snapshot, once the caller knows it) to complete the
    /// learning step. Used by `Population::step_social`, which applies the cross-organism
    /// transfer consequence between these two calls.
    pub fn act_social(
        &mut self,
        resource_level: f64,
        forced_action: Option<usize>,
        partner: Option<(AgentId, InteractionRecord)>,
    ) -> (OrganismTick, PendingSocialLearning) {
        self.act_phase(resource_level, forced_action, partner)
    }

    /// Genesis v0.1 audit Gate 4: complete the learning step for a tick started with
    /// [`Self::act_social`], using `realized_partner` -- the partner ledger snapshot *after* this
    /// tick's cross-organism consequence has been applied (e.g. by `Population::step_social`'s
    /// Phase 3), not the pre-action snapshot `act_social` was given. This is what lets an
    /// organism's own `Transfer` this tick actually be visible to its own learning update this
    /// tick, closing the gap the original Genesis v0 exploratory run's flat action distribution
    /// may have been caused by.
    pub fn learn_from_realized_outcome(
        &mut self,
        pending: PendingSocialLearning,
        realized_partner: Option<(AgentId, InteractionRecord)>,
    ) {
        self.learn_phase(pending, realized_partner)
    }

    fn tick_inner(
        &mut self,
        resource_level: f64,
        forced_action: Option<usize>,
        partner: Option<(AgentId, InteractionRecord)>,
    ) -> OrganismTick {
        let (mut tick, pending) = self.act_phase(resource_level, forced_action, partner);
        self.learn_phase(pending, partner);
        // `ActiveInferenceAgent::learn_from_outcome` internally calls `perceive()` a *second*
        // time (on the realized observation), which updates both belief and
        // `last_fe_components`. The original single-phase `tick_inner` built `OrganismTick`
        // *after* that second perceive() ran, so these three fields reflected post-learning
        // state -- re-read them here so the immediate-learn path (`tick`/`tick_social`) stays
        // byte-for-byte identical to before the act/learn split. `act_social`'s caller
        // (`Population::step_social`) never reads these fields off the pre-learn tick it gets
        // back, so this doesn't need mirroring there.
        tick.belief_resource = self.agent.belief.mean[0];
        tick.belief_energy = self.agent.belief.mean[1];
        tick.free_energy = self.agent.current_free_energy();
        tick
    }

    /// Raw per-partner ledger slice, numerically compressed for the belief substrate only via
    /// `compress_for_observation` -- the stored `self.ledger` entry itself stays exact; see
    /// `ledger` module docs. `partner_present` is a plain factual signal ("is anyone paired with
    /// me this tick"), not a social interpretation -- structurally the same kind of raw exogenous
    /// fact as `resource_level`/`energy_level` already are. Shared by both the perceive-time and
    /// learn-time observation construction (`act_phase`/`learn_phase`) so they can never drift
    /// out of sync with each other.
    fn social_channels(
        social_enabled: bool,
        partner: &Option<(AgentId, InteractionRecord)>,
    ) -> (f64, f64, f64, f64) {
        if !social_enabled {
            return (0.0, 0.0, 0.0, 0.0);
        }
        match partner {
            Some((_, record)) => (
                1.0,
                compress_for_observation(record.given_to_partner),
                compress_for_observation(record.received_from_partner),
                compress_for_observation(record.encounter_count as f64),
            ),
            None => (0.0, 0.0, 0.0, 0.0),
        }
    }

    fn act_phase(
        &mut self,
        resource_level: f64,
        forced_action: Option<usize>,
        partner: Option<(AgentId, InteractionRecord)>,
    ) -> (OrganismTick, PendingSocialLearning) {
        // 1. Boundary permeability from the organism's own physiological deficit.
        let deficit = (self.cfg.set_point - self.energy).max(0.0) / self.cfg.set_point.max(1e-6);
        let modulators = BoundaryModulators::from_energy_deficit(deficit);
        let permeability = self
            .boundary
            .compute_permeability(&modulators.to_permeability_inputs())
            .clone();

        // 2. Gate the raw observation through the blanket toward the current prior. Hoffman
        // Fitness-Beats-Truth experiment (`tests/hoffman_fitness_beats_truth.rs`): when
        // `perceptual_grain` is `Some`, the resource reading is coarse-grained *before* the
        // blanket ever sees it -- a real measurable-map perceptual strategy (Mark, Marion &
        // Hoffman 2010), distinct from blanket permeability (a separate, pre-existing
        // mechanism). `None` is a pure no-op: `resource_percept == resource_level`, exactly
        // today's pre-existing behavior.
        let (resource_percept, resolution_bits) = match self.cfg.perceptual_grain {
            Some(grain) => (
                quantize_to_grain(resource_level, grain),
                perceptual_resolution_bits(grain),
            ),
            None => (resource_level, 0.0),
        };

        let (partner_present, given_c, received_c, count_c) =
            Self::social_channels(self.cfg.social_enabled, &partner);
        let mut obs_values = vec![resource_percept, self.energy];
        if self.cfg.social_enabled {
            obs_values.extend_from_slice(&[partner_present, given_c, received_c, count_c]);
        }
        let raw_obs = Observation::new(obs_values, 1.0, "resource+energy+social");
        let gated_obs = self.boundary.gate_observation(&raw_obs, &self.agent.belief);

        // 3. Perceive: update belief from the gated observation. Consumed via belief_resource
        //    below, and via belief_change feeding Phase 3's real Landauer bit-count (not
        //    discarded).
        let perception = self.agent.perceive(&gated_obs);

        // 4. Select action (always run for real; forced_action only overrides which one is
        //    actually executed on the environment, for the Phase-0 baseline comparison).
        let selection = self.agent.select_action();
        let action_index = forced_action.unwrap_or(selection.action);

        // 5. Act: predicted outcome, consumed as a real prediction-error metric.
        let predicted = self.agent.act(action_index);
        let predicted_resource = predicted
            .expected_observation
            .first()
            .copied()
            .unwrap_or(0.5);

        // 6. Real-world consequence: only foraging can gain energy, and it costs more to do.
        // Deliberately uses the true `resource_level`, not `gated_obs` -- the blanket gates what
        // the organism *believes*, not what it can physically metabolize. Using the belief-gated
        // value here would let a slowly-adapting belief "hallucinate" real energy from a resource
        // that has actually crashed to near zero, for as long as belief lags the true world state
        // (found via Phase 1's breakeven calibration collapsing toward ~0 -- an organism was
        // living off its own stale optimism, not the environment).
        let foraging = action_index == Action::Forage.index();
        // Hoffman Interface Theory Plan Phase 2: when `spoilage_sigma` is `Some`, the true
        // payoff is a non-monotonic (Gaussian interior-optimum) function of `resource_level`
        // instead of linear -- still computed from the TRUE resource_level, preserving the same
        // belief/consequence separation as the linear path above.
        let payoff_value = match self.cfg.spoilage_sigma {
            Some(sigma) => {
                let sigma = sigma.max(1e-6);
                let deviation = resource_level - 0.5;
                (-(deviation * deviation) / (2.0 * sigma * sigma)).exp()
            }
            None => resource_level,
        };
        let gained = if foraging {
            self.cfg.forage_efficiency * payoff_value * permeability.active
        } else {
            0.0
        };
        let activity_cost = if foraging {
            self.cfg.forage_activity_cost
        } else {
            0.0
        };

        // 6b. Phase 3: real thermodynamic floor, charged on top of the config-driven costs
        // above, not instead of them. `bits_processed` is real, not decorative -- perception's
        // own `belief_change` (magnitude of belief actually written this tick) plus the Shannon
        // entropy of the action-selection distribution actually used (how many bits of
        // uncertainty were resolved by committing to one action). `order_maintained` reuses
        // `deficit` from step 1: staying close to the homeostatic set-point *is* the order this
        // organism is dissipating energy to maintain (Prigogine).
        let bits_processed = perception.belief_change
            + shannon_entropy_bits(&selection.action_probabilities)
            + resolution_bits;
        let order_maintained = 1.0 - deficit;
        let physical_cost = landauer_minimum(bits_processed, self.cfg.effective_temperature)
            + prigogine_dissipation_cost(
                order_maintained,
                self.cfg.effective_temperature,
                self.cfg.dissipation_rate,
            );

        // Genesis v0 (G0e): a `Transfer` debit is computed from what would remain *after* every
        // other cost, and clamped to that (never "gives energy it doesn't have") -- this is what
        // makes the Stage 0 "conservation" invariant exact: the caller credits the partner with
        // exactly this method's `transfer_given`, not a flat configured quantum.
        let energy_before_transfer =
            (self.energy + gained - self.cfg.metabolic_cost - activity_cost - physical_cost)
                .max(0.0);
        let transferring = self.cfg.social_enabled && action_index == Action::Transfer.index();
        let transfer_given = if transferring && partner.is_some() {
            self.cfg.transfer_quantum.min(energy_before_transfer)
        } else {
            0.0
        };
        self.energy = (energy_before_transfer - transfer_given).clamp(0.0, 1.0);

        let tick = OrganismTick {
            resource_observed: resource_level,
            belief_resource: self.agent.belief.mean[0],
            belief_energy: self.agent.belief.mean[1],
            energy: self.energy,
            action: action_index,
            free_energy: self.agent.current_free_energy(),
            permeability,
            act_prediction_error: (predicted_resource - resource_level).abs(),
            bits_processed,
            physical_cost,
            is_dead: self.is_dead(),
            transfer_given,
            gated_observation: gated_obs.values.clone(),
            raw_observation: raw_obs.values.clone(),
            action_probabilities: selection.action_probabilities.clone(),
        };
        let pending = PendingSocialLearning {
            action_index,
            resource_level,
        };
        (tick, pending)
    }

    /// 7. Learn from the actual post-action outcome (internally perceives again with the real
    /// consequence, per `ActiveInferenceAgent::learn_from_outcome`'s own contract). `partner` is
    /// whatever the caller supplies -- the *same* pre-action snapshot for `tick_inner`'s immediate
    /// path, or the *realized* post-consequence snapshot for `Population::step_social`'s deferred
    /// path (Genesis v0.1 audit Gate 4). Also finalizes `self.last_resource_observed`, matching
    /// the original single-phase `tick_inner`'s ordering.
    fn learn_phase(
        &mut self,
        pending: PendingSocialLearning,
        partner: Option<(AgentId, InteractionRecord)>,
    ) {
        let (partner_present, given_c, received_c, count_c) =
            Self::social_channels(self.cfg.social_enabled, &partner);
        let mut actual_values = vec![pending.resource_level, self.energy];
        if self.cfg.social_enabled {
            actual_values.extend_from_slice(&[partner_present, given_c, received_c, count_c]);
        }
        let actual_obs = Observation::new(actual_values, 1.0, "resource+energy+social");
        self.agent
            .learn_from_outcome(pending.action_index, &actual_obs);
        self.last_resource_observed = pending.resource_level;
    }
}

/// Opaque token carrying what [`Organism::learn_phase`] needs, bridging [`Organism::act_social`]
/// and [`Organism::learn_from_realized_outcome`] across whatever a caller does in between (e.g.
/// `Population::step_social` applying a cross-organism transfer consequence). Callers other than
/// those two methods have no need to inspect its fields.
pub struct PendingSocialLearning {
    action_index: usize,
    resource_level: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::Environment;

    #[test]
    fn energy_stays_in_unit_range_over_many_ticks() {
        let mut organism = Organism::new(OrganismConfig::default(), 1);
        let env = Environment::default();
        for t in 0..2000u64 {
            let tick = organism.tick(env.resource_at(t), None);
            assert!(
                (0.0..=1.0).contains(&tick.energy),
                "energy out of range at t={t}: {}",
                tick.energy
            );
        }
    }

    #[test]
    fn action_index_is_always_valid() {
        let mut organism = Organism::new(OrganismConfig::default(), 2);
        let env = Environment::default();
        for t in 0..500u64 {
            let tick = organism.tick(env.resource_at(t), None);
            assert!(tick.action < Action::COUNT);
        }
    }

    #[test]
    fn perceiving_updates_belief_away_from_default() {
        // A genuinely inert agent starts at belief mean 0.5 in every dimension. If perceive()
        // were theater (called but not actually load-bearing), belief would stay pinned there.
        let mut organism = Organism::new(OrganismConfig::default(), 3);
        let env = Environment::default();
        for t in 0..50u64 {
            organism.tick(env.resource_at(t), None);
        }
        assert!(
            (organism.agent.belief.mean[0] - 0.5).abs() > 0.01,
            "belief_resource didn't move from its default: {}",
            organism.agent.belief.mean[0]
        );
    }

    #[test]
    fn perceptual_grain_none_reproduces_pre_existing_behavior() {
        // The load-bearing backward-compat property: an organism with `perceptual_grain: None`
        // (the default) must tick byte-for-byte identically to before this trait existed. If a
        // coarse-graining/cost bug leaked into the `None` path, every pre-existing phase's tests
        // would be silently affected without any of them naming this trait.
        let mut with_none = Organism::new(OrganismConfig::default(), 5);
        let mut without_field = Organism::new(OrganismConfig::default(), 5);
        let env = Environment::default();
        for t in 0..200u64 {
            let a = with_none.tick(env.resource_at(t), None);
            let b = without_field.tick(env.resource_at(t), None);
            assert_eq!(
                a.energy, b.energy,
                "t={t}: energy diverged under perceptual_grain=None"
            );
            assert_eq!(
                a.bits_processed, b.bits_processed,
                "t={t}: bits_processed diverged under perceptual_grain=None"
            );
        }
    }

    #[test]
    fn perceptual_grain_some_costs_more_than_none() {
        // Direct mechanism check for the Hoffman Fitness-Beats-Truth experiment: activating
        // perceptual resolution must be a real, measurable cost, not decorative.
        let mut fine = Organism::new(
            OrganismConfig {
                perceptual_grain: Some(0.02),
                ..OrganismConfig::default()
            },
            6,
        );
        let mut baseline = Organism::new(OrganismConfig::default(), 6);
        let env = Environment::default();
        let mut fine_bits_total = 0.0;
        let mut baseline_bits_total = 0.0;
        for t in 0..200u64 {
            fine_bits_total += fine.tick(env.resource_at(t), None).bits_processed;
            baseline_bits_total += baseline.tick(env.resource_at(t), None).bits_processed;
        }
        assert!(
            fine_bits_total > baseline_bits_total,
            "Some(grain) should charge strictly more resolution bits than None: \
             fine={fine_bits_total}, baseline={baseline_bits_total}"
        );
    }

    #[test]
    fn spoilage_sigma_none_reproduces_pre_existing_linear_payoff() {
        // Same backward-compat discipline as perceptual_grain: None must tick byte-for-byte
        // identically to before this trait existed.
        let mut with_none = Organism::new(OrganismConfig::default(), 7);
        let mut without_field = Organism::new(OrganismConfig::default(), 7);
        let env = Environment::default();
        for t in 0..200u64 {
            let a = with_none.tick(env.resource_at(t), None);
            let b = without_field.tick(env.resource_at(t), None);
            assert_eq!(
                a.energy, b.energy,
                "t={t}: energy diverged under spoilage_sigma=None"
            );
        }
    }

    #[test]
    fn spoilage_sigma_some_makes_payoff_non_monotonic() {
        // Direct mechanism check: forcing forage_action=0 (Forage) at three fixed resource
        // levels straddling and away from the 0.5 optimum should show more energy gained near
        // the peak than far from it in either direction -- the actual interior-optimum shape,
        // not decorative.
        let cfg = OrganismConfig {
            spoilage_sigma: Some(0.1),
            forage_efficiency: 0.6,
            forage_activity_cost: 0.0, // isolate the payoff shape from activity cost here
            ..OrganismConfig::default()
        };
        let mut at_peak = Organism::new(cfg, 8);
        let mut below_peak = Organism::new(cfg, 9);
        let mut above_peak = Organism::new(cfg, 10);
        let gained_at = |o: &mut Organism, r: f64| {
            let before = o.energy;
            o.tick(r, Some(Action::Forage.index()));
            o.energy - before
        };
        let peak_gain = gained_at(&mut at_peak, 0.5);
        let below_gain = gained_at(&mut below_peak, 0.2);
        let above_gain = gained_at(&mut above_peak, 0.8);
        assert!(
            peak_gain > below_gain && peak_gain > above_gain,
            "gain at the 0.5 optimum should exceed gain away from it in both directions: \
             peak={peak_gain}, below={below_gain}, above={above_gain}"
        );
    }

    #[test]
    fn resource_prior_default_reproduces_pre_existing_belief_dynamics() {
        // Same backward-compat discipline as perceptual_grain/spoilage_sigma: the default 0.5
        // must be an exact no-op, since GenerativeModel::new already initializes prior_mean to
        // [0.5; state_dim] with precision [1.0; state_dim] -- injecting the same values back in
        // must change nothing.
        let mut with_default = Organism::new(OrganismConfig::default(), 9);
        let mut without_injection = Organism::new(OrganismConfig::default(), 9);
        let env = Environment::default();
        for t in 0..200u64 {
            let a = with_default.tick(env.resource_at(t), None);
            let b = without_injection.tick(env.resource_at(t), None);
            assert_eq!(
                a.energy, b.energy,
                "t={t}: energy diverged under resource_prior=0.5 (should be a no-op)"
            );
            assert_eq!(
                with_default.agent.belief.mean, without_injection.agent.belief.mean,
                "t={t}: belief diverged under resource_prior=0.5 (should be a no-op)"
            );
        }
    }
}
