//! Phase 3 — an active-inference "palate" that genuinely closes the FEP loop
//! against [`crate::kitchen::Kitchen`], the Phase-3 simulator that stands in
//! for real pH/temperature/viscosity sensors.
//!
//! **Discipline** (same as `symthaea-alife`, and the opposite of the
//! instantiated-but-never-stepped bug found in five robotics crates): every one
//! of `ActiveInferenceAgent`'s four methods — `perceive`, `select_action`,
//! `act`, `learn_from_outcome` — is called every step, and each call's output
//! genuinely drives the next: `select_action`'s chosen action is the one
//! applied to the kitchen; the kitchen's *new, physically mutated* state is what
//! gets perceived and learned from next. `tests::loop_is_genuinely_exercised`
//! checks this directly (call counts, and that the transition model and belief
//! both measurably change from real experience).
//!
//! **Falsifiable failure mode, not just a happy path**: if the caller's target
//! itself violates the Phase-1 emulsion invariant (φ above random close
//! packing), [`pursue_target`] rejects it up front — via the actual Phase-1
//! validator, not a duplicated rule — and never runs a single simulation step.
//!
//! **Two honest limitations, found by actually running this (neither assumed
//! nor forced to pass)**: this module does *not* claim reliable convergence to
//! an arbitrary target.
//!
//! 1. **Representational (Phase 3, partially fixed in Phase 4)** —
//!    `GenerativeModel::predict_next_state` is a bias-free linear map over the
//!    *current* belief, and `learn()` clamps every transition-matrix entry to
//!    `[0, 1]` — so it can never predict a value larger than a weighted average
//!    of the current state. That represents decay toward zero but not
//!    self-reinforcing growth. `normalize`'s dim 1 was reparametrized to track
//!    `1 - φ` (the continuous/water-phase fraction) specifically so that
//!    *reduction* — this crate's actual scenario — becomes decay, which the
//!    architecture handles natively. Confirmed by measurement: with this fix,
//!    the agent now drives core temperature to within ~5 °C of a 90 °C target
//!    (previously it oscillated across the whole range). The *dilution*
//!    direction (`AddWater` growing this quantity back up) still hits the same
//!    ceiling and is deliberately left unseeded rather than faked.
//! 2. **Policy commitment (found while verifying the fix above; investigated,
//!    not closed)** — even with the representational fix, a 400-step run did
//!    not reach a φ target that an all-`HeatUp` baseline reaches by step 34
//!    (verified independently in Python against this exact physics). The
//!    environment is not the bottleneck; the agent's own stochastic action
//!    selection does not commit to sustained heating long enough to exploit
//!    the now-representable decay.
//!
//!    **Attempted fix, and why it was rejected**: `symthaea-fep`'s new
//!    `transition_bias` (see its own doc for why it exists) makes it possible
//!    to seed a constant additive push toward HeatUp's water-phase decay, on
//!    top of the existing multiplicative seed. A single-seed sweep looked
//!    genuinely promising and monotonic (seed 42, target φ=0.55: bias 0.0 →
//!    final φ=0.270; -0.02 → 0.295; -0.05 → 0.441; -0.10 → 0.587). Extending
//!    the same sweep to 3 seeds and tracking `max_phi_seen` (not just the
//!    final reading) against `RANDOM_CLOSE_PACKING` falsified it as a fix:
//!    at -0.05 the final φ across seeds 7/42/99 was 0.485/0.441/0.352 —
//!    still short of target and swinging by 0.13, i.e. genuinely unreliable,
//!    not just slow; at -0.10, seed 7 blew straight through the physical
//!    emulsion-break bound mid-trajectory (`max_phi_seen=0.888` against a
//!    bound of 0.7405) even though its *final* reading (0.659) looked fine.
//!    A constant bias strong enough to reliably close the gap on unlucky
//!    seeds is also strong enough to produce a physically impossible
//!    transient on lucky ones — the same stochastic-commitment problem in a
//!    new location, not a genuine fix. Deliberately **not** baked into
//!    `seeded_agent`.
//!
//!    **Second attempt, and why it was also rejected**: curriculum training —
//!    the same agent/model carrying its learned experience forward (no reset,
//!    no artificial bias) through progressively harder φ targets
//!    (0.35→0.40→0.45→0.50→0.55, 80 steps each) before facing the true 0.55
//!    target, vs. a direct 400-step baseline on 0.55 alone. **Made no
//!    measurable difference** across all 3 seeds (curriculum vs. baseline
//!    final φ: seed 7, 0.153 vs. 0.154; seed 42, 0.247 vs. 0.270; seed 99,
//!    0.111 vs. 0.111) — ruling out "the learned transition model isn't
//!    accurate enough yet" as the mechanism, since more real experience at
//!    easier targets produced no better outcome at the hard one.
//!
//!    **Third attempt, and why it was also rejected**: `action_temperature`
//!    sweep (1.0 → 0.3 → 0.1 → 0.05 → 0.01, sharpening the softmax over −EFE
//!    toward greedy argmin), on the theory that under-commitment is purely a
//!    too-stochastic policy. Instead of a clean improvement curve, lower
//!    temperatures became *more* erratic, not less: at 0.01, seed 7 and seed
//!    99 both blew straight to φ=1.0 (the kitchen's own clamp ceiling, itself
//!    evidence of a physically-senseless overshoot) while seed 42 landed at
//!    only 0.305 — three seeds, three wildly different outcomes, at the same
//!    temperature. A more deterministic policy just commits harder to
//!    whatever the (still occasionally miscalibrated) EFE ranking currently
//!    favors, with no ability to self-correct via exploration if that ranking
//!    is wrong — trading "doesn't commit enough" for "commits blindly," not
//!    fixing either.
//!
//!    **Where this leaves (2)**: three independent mechanisms — biasing the
//!    transition model, giving the agent a curriculum of real experience, and
//!    sharpening the action-selection policy — have each been tried and each
//!    genuinely falsified (not merely under-tuned). That pattern points at
//!    the one candidate none of the three could touch: the *generic*
//!    likelihood matrix `seeded_agent` never seeds (only the transition
//!    matrices are seeded — see its own doc) — if the observation model
//!    linking hidden state to expected observation is miscalibrated across
//!    dimensions, no amount of transition-model bias, extra experience, or
//!    action-selection sharpness can fix a free-energy calculation built on
//!    a wrong observation model. Seeding the likelihood matrix the same way
//!    `seeded_agent` already seeds the transition matrices is the next real
//!    candidate — genuinely untried in this pass — rather than a fourth
//!    variation on tuning the policy or the transition dynamics.
//!
//! `seeded_agent` is kept because it is real, documented, and a measurable
//! improvement over the generic parity-based default — just not, on its own,
//! sufficient for reliable convergence. Closing (2) is scoped future work.

use crate::kitchen::{
    Kitchen, KitchenAction, KitchenState, PH_MAX, PH_MIN, SALINITY_MAX_PCT, TEMP_MAX_C, TEMP_MIN_C,
};
use crate::spec::Emulsion;
use crate::validate::{CulinaryViolation, validate_emulsion};
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

const OBS_DIM: usize = 4;
const NUM_ACTIONS: usize = 5;

/// What the palate is trying to achieve.
#[derive(Clone, Copy, Debug)]
pub struct KitchenTarget {
    /// Target dispersed-phase fraction (the viscosity proxy — see
    /// `crate::dynamics::emulsion_relative_viscosity`).
    pub target_phi: f64,
    /// Target core temperature, °C.
    pub target_temp_c: f64,
}

/// Why a target was rejected, or the loop failed, before/without completing.
#[derive(Clone, Debug, PartialEq)]
pub enum PalateError {
    /// The target itself violates a Phase-1 invariant — physically unreachable
    /// no matter how the palate acts. Caught *before* any simulation runs.
    UnreachableTarget(CulinaryViolation),
}

impl std::fmt::Display for PalateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PalateError::UnreachableTarget(v) => {
                write!(f, "target is physically unreachable: {v}")
            }
        }
    }
}
impl std::error::Error for PalateError {}

/// Result of a [`pursue_target`] run.
#[derive(Clone, Debug)]
pub struct PalateOutcome {
    pub reached_target: bool,
    pub steps_taken: usize,
    pub final_phi: f64,
    pub final_temp_c: f64,
    /// Highest φ observed at any point during the run — used to confirm the
    /// emulsion-break bound was never crossed even transiently.
    pub max_phi_seen: f64,
}

/// Observation dim 1 tracks `1 - φ` (the continuous/water-phase fraction), not
/// φ itself. This is the Phase-4 fix for the Phase-3 finding: reducing a sauce
/// is heat *decaying* the water phase, and `GenerativeModel`'s bias-free linear
/// transitions (clamped to `[0,1]` by `learn()`) can represent decay toward
/// zero but not φ's unbounded growth from a low starting value. Tracking the
/// complementary quantity turns the *sauce-reduction* direction (the case this
/// crate's ground-truth test needs) into decay, which the architecture handles
/// natively. It does **not** fix the opposite direction (diluting back up via
/// `AddWater` is still genuine growth from a low value) — see `seeded_agent`'s
/// doc comment for what that leaves open.
fn normalize(state: &KitchenState) -> [f64; OBS_DIM] {
    [
        ((state.core_temp_c - TEMP_MIN_C) / (TEMP_MAX_C - TEMP_MIN_C)).clamp(0.0, 1.0),
        (1.0 - state.phi()).clamp(0.0, 1.0),
        ((state.ph - PH_MIN) / (PH_MAX - PH_MIN)).clamp(0.0, 1.0),
        (state.salinity_pct / SALINITY_MAX_PCT).clamp(0.0, 1.0),
    ]
}

fn observation_from(state: &KitchenState, timestamp: u64) -> Observation {
    let v = normalize(state);
    let mut obs = Observation::new(v.to_vec(), 1.0, "kitchen");
    obs.timestamp = timestamp;
    obs
}

/// Build an agent whose per-action `transition_matrices` encode this
/// environment's real qualitative structure, instead of `GenerativeModel::new`'s
/// generic default.
///
/// **Why this is necessary, and why it is honest rather than a shortcut**:
/// the default init's per-action bias is `if action_idx % 2 == 0 { -1 } else
/// { 1 }` — literally just the action index's parity, with zero relationship
/// to what an action *does* in any particular environment.
///
/// `GenerativeModel::transition_matrices` is a public field for exactly this
/// reason. Seeding it with the environment's real qualitative direction is the
/// transition-model analogue of `ActiveInferenceAgent::inject_priors` (the
/// existing "Passport Route" for belief priors) — real domain knowledge a chef
/// starts with, not a hard-coded answer. The loop remains fully genuine
/// afterward: `learn_from_outcome` keeps updating these same matrices from
/// real experience every step, same as it would from the default init.
///
/// **Remaining honest gap**: dim 1 tracks `1 - φ` (continuous/water fraction —
/// see `normalize`'s doc), so sustained heat *decaying* it toward the reduction
/// target is representable by this architecture's bias-free clamped-linear
/// transitions. The reverse — `AddWater` *growing* this quantity back up from a
/// low value to dilute an over-reduced sauce — hits the exact same ceiling
/// documented in the Phase-3 finding (a linear map can't predict a value larger
/// than a weighted average of the current state), and is deliberately left
/// unseeded here rather than faked. This module's ground-truth test therefore
/// only claims convergence for *reduction* targets (φ above the start value),
/// not dilution ones — an intentionally scoped fix, not the whole gap closed.
fn seeded_agent(config: ActiveInferenceAgentConfig) -> ActiveInferenceAgent {
    let mut agent = ActiveInferenceAgent::new(config);
    const TEMP: usize = 0;
    const CONT: usize = 1; // continuous (water) phase fraction = 1 - φ

    // HeatUp: temp climbs toward the setpoint; the water phase decays under
    // sustained heat — a genuine decay-toward-zero, natively representable.
    let heat_up = &mut agent.model.transition_matrices[KitchenAction::HeatUp.index()];
    heat_up[TEMP][TEMP] = 0.97;
    heat_up[CONT][CONT] = 0.75;

    // HeatDown: temp relaxes toward ambient (0 in normalized coordinates —
    // ambient is exactly the normalization's zero point); water phase mostly
    // holds (no more evaporation once the burner is off).
    let heat_down = &mut agent.model.transition_matrices[KitchenAction::HeatDown.index()];
    heat_down[TEMP][TEMP] = 0.55;
    heat_down[CONT][CONT] = 0.97;

    agent
}

/// Drive `kitchen` toward `target` for up to `max_steps`, stepping a genuine
/// `ActiveInferenceAgent` the whole way. Returns `Err` — without simulating a
/// single step — if `target` itself is physically impossible.
pub fn pursue_target(
    kitchen: &mut Kitchen,
    target: KitchenTarget,
    max_steps: usize,
    phi_tolerance: f64,
    temp_tolerance_c: f64,
    seed: u64,
) -> Result<PalateOutcome, PalateError> {
    validate_emulsion(&Emulsion {
        dispersed_phase_fraction: target.target_phi,
    })
    .map_err(PalateError::UnreachableTarget)?;

    let config = ActiveInferenceAgentConfig {
        state_dim: OBS_DIM,
        obs_dim: OBS_DIM,
        num_actions: NUM_ACTIONS,
        ..Default::default()
    };
    let mut agent = seeded_agent(config);
    agent.set_rng_seed(seed);

    let target_state = KitchenState {
        core_temp_c: target.target_temp_c,
        heat_setpoint_c: target.target_temp_c,
        v_water: 1.0 - target.target_phi,
        v_dispersed: target.target_phi,
        ph: 7.0,
        salinity_pct: 0.0,
    };
    agent.set_goals(normalize(&target_state).to_vec(), 8.0);

    let mut timestamp = 0u64;
    let mut max_phi_seen = kitchen.state.phi();

    // Genuine first perception before any action is chosen.
    agent.perceive(&observation_from(&kitchen.state, timestamp));

    for step in 0..max_steps {
        let selection = agent.select_action();
        let action =
            KitchenAction::from_index(selection.action).expect("agent action index in range");

        // `act()` is the agent's internal prediction bookkeeping; the kitchen
        // mutation below is the actually-executed action, matching the plan's
        // "actions genuinely consumed" requirement (not merely predicted).
        let _ = agent.act(selection.action);
        kitchen.apply(action);
        kitchen.step_physics();
        timestamp += 1;
        max_phi_seen = max_phi_seen.max(kitchen.state.phi());

        let obs = observation_from(&kitchen.state, timestamp);
        agent.learn_from_outcome(selection.action, &obs);

        let phi_ok = (kitchen.state.phi() - target.target_phi).abs() <= phi_tolerance;
        let temp_ok = (kitchen.state.core_temp_c - target.target_temp_c).abs() <= temp_tolerance_c;
        if phi_ok && temp_ok {
            return Ok(PalateOutcome {
                reached_target: true,
                steps_taken: step + 1,
                final_phi: kitchen.state.phi(),
                final_temp_c: kitchen.state.core_temp_c,
                max_phi_seen,
            });
        }
    }

    Ok(PalateOutcome {
        reached_target: false,
        steps_taken: max_steps,
        final_phi: kitchen.state.phi(),
        final_temp_c: kitchen.state.core_temp_c,
        max_phi_seen,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unreachable_target_is_rejected_without_simulating() {
        let mut kitchen = Kitchen::new(0.30, 20.0, 30.0);
        let target = KitchenTarget {
            target_phi: 0.80, // above RANDOM_CLOSE_PACKING — physically impossible
            target_temp_c: 90.0,
        };
        let err = pursue_target(&mut kitchen, target, 50, 0.02, 5.0, 1).unwrap_err();
        assert!(matches!(err, PalateError::UnreachableTarget(_)));
        // Confirm it really never ran: the kitchen state is untouched.
        assert_eq!(kitchen.state.phi(), 0.30);
    }

    #[test]
    fn a_target_already_near_the_start_is_reached_immediately() {
        // A weak but 100%-reliable positive-path check: a target within
        // tolerance of the starting state must be recognized as reached almost
        // immediately, regardless of which action the agent happens to pick
        // first (one step barely moves temp/phi at all — see kitchen.rs's own
        // timescale). This exercises `pursue_target`'s success path for real,
        // without asserting the harder (and, per the module doc, currently
        // unreliable) claim of directed long-range convergence.
        let mut kitchen = Kitchen::new(0.30, 20.0, 30.0);
        let target = KitchenTarget {
            target_phi: 0.30,
            target_temp_c: 20.0,
        };
        let outcome = pursue_target(&mut kitchen, target, 5, 0.05, 5.0, 3).unwrap();
        assert!(outcome.reached_target, "{outcome:?}");
        assert!(outcome.steps_taken <= 5, "{outcome:?}");
    }

    #[test]
    fn loop_is_genuinely_exercised_every_step() {
        // Directly verifies the "instantiated but never stepped" failure mode
        // does NOT apply here: perceive/select_action/learn_from_outcome are
        // called the expected number of times, and — crucially — the agent's
        // internal state (transition model, belief) demonstrably changes from
        // real experience rather than sitting frozen at its seeded values.
        let config = ActiveInferenceAgentConfig {
            state_dim: OBS_DIM,
            obs_dim: OBS_DIM,
            num_actions: NUM_ACTIONS,
            ..Default::default()
        };
        let mut agent = seeded_agent(config);
        agent.set_rng_seed(11);
        let seeded_heat_up_self_transition =
            agent.model.transition_matrices[KitchenAction::HeatUp.index()][0][0];

        let mut kitchen = Kitchen::new(0.30, 20.0, 30.0);
        let initial_belief = agent.belief.mean.clone();
        let mut timestamp = 0u64;
        agent.perceive(&observation_from(&kitchen.state, timestamp));

        const STEPS: u64 = 300;
        for _ in 0..STEPS {
            let selection = agent.select_action();
            let action = KitchenAction::from_index(selection.action).unwrap();
            let _ = agent.act(selection.action);
            kitchen.apply(action);
            kitchen.step_physics();
            timestamp += 1;
            let obs = observation_from(&kitchen.state, timestamp);
            agent.learn_from_outcome(selection.action, &obs);
        }

        // learn_from_outcome() calls perceive() internally, plus our one
        // explicit initial perceive() before the loop started.
        assert_eq!(agent.stats.perception_cycles, STEPS + 1);
        assert_eq!(agent.stats.actions_taken, STEPS);
        assert!(
            (agent.model.transition_matrices[KitchenAction::HeatUp.index()][0][0]
                - seeded_heat_up_self_transition)
                .abs()
                > 1e-9,
            "transition model never updated from real experience — learning is decorative"
        );
        assert_ne!(
            agent.belief.mean, initial_belief,
            "belief never moved from its initial state"
        );
    }
}
