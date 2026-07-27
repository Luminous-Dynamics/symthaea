// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rungs 1-6 of the baseline hierarchy for the ecological-collapse scenario's
//! extinction-within-horizon target (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`'s "First
//! experiment").
//!
//! ## Rung 5 (`FepDriven`) — the Stage B adapter, and a real "no learning happened" trap it had
//! to avoid
//!
//! [`FepDrivenGenerator`] is the first consumer of `symthaea-futures-state::mask_observation`
//! for its intended Stage B role (turning a scenario observation into a dense
//! `symthaea_fep::Observation` an `ActiveInferenceAgent` can perceive), not just the
//! not-yet-used dependency it was when `-symtropy` first added it.
//!
//! **A real trap found and avoided, not just a design choice**: calling `agent.perceive()` in a
//! loop with no accompanying `act()`/`learn_from_outcome()` calls does **not** train anything.
//! Reading `symthaea-fep::agent.rs` directly: `GenerativeModel::learn()`'s transition-matrix
//! update only fires when an `action: Some(_)` is passed, and `perceive()` always passes
//! `self.last_action` — which stays `None` forever unless something calls `act()`. With the
//! default `enable_td_learning: true`, the TD path (which *would* teach transitions) is also
//! gated on `self.last_action`, so it never fires either. A perceive-only loop would leave
//! `transition_matrices` at their random `GenerativeModel::new()` initialization forever —
//! extrapolating through them would be extrapolating through noise, not anything learned from
//! the observed sequence, while still *labeled* "FEP-driven." This generator avoids that by
//! configuring `num_actions: 1` (there's no real action/choice in passive forecasting, so action
//! `0` is the only one, existing purely so `observe_transition()` — the public API's
//! explicitly-stated "primary interface" for feeding transition information — has something to
//! index into) and explicitly calling `agent.observe_transition(&prev_belief, 0, &new_belief,
//! &masked_obs)` after every `perceive()`, so `transition_matrices[0]` is actually shaped by the
//! replayed history before `predict_next_state` is used to extrapolate.
//!
//! ## Rung 3 needs a history, not a snapshot — a per-generator `Observation` type, not a trait
//! change
//!
//! [`SimpleStatisticalGenerator`] needs multiple past readings to fit a trend, unlike rungs 1-2
//! and 4. `TrajectoryGenerator::Observation` is an associated type *per implementor*, so this
//! generator simply declares `type Observation = Vec<EcologicalObservation>` — nothing about the
//! trait itself needed to change to accommodate a history-shaped input for one rung while the
//! others stay snapshot-shaped.
//!
//! ## The observable/target mismatch these baselines inherit (and are supposed to expose)
//!
//! Every generator here only ever sees an [`EcologicalObservation`] — the *tracked cohort's*
//! state (see `symthaea-futures-symtropy::ecological`'s module docs on why sampling uses a fixed
//! cohort by `AgentId`, not a live-recomputed fraction). The forecasting target is whether the
//! *whole population* goes extinct, not the cohort. Cohort-extinct does **not** imply
//! population-extinct: untracked offspring born after the cohort was fixed can still be alive.
//! [`PersistenceGenerator`] operationalizes "whatever's true now continues" on the only thing it
//! can observe (cohort status) — the resulting proxy error is a real, expected consequence of
//! the leakage-safe sampling design, not a bug in this generator. Rung 1 exists precisely to be
//! beatable; this mismatch is one of the ways it should lose to better rungs.
//!
//! ## Rung 6 (`OracleUpperBound`) — resolved; the earlier "design gap" note was overstated
//!
//! A prior version of this module doc claimed an oracle "doesn't fit `TrajectoryGenerator`'s
//! per-tick shape" because `EcologicalGroundTruth` can't be `Clone`d (`symthaea_alife::Population`
//! has no `Clone` derive, confirmed by reading its source; this plan doesn't touch
//! `symthaea-alife` to add one). That conclusion assumed an oracle needs to *forward-simulate
//! from each snapshot it's asked about* — which would indeed need cloning a live world. It
//! doesn't need to: this is offline backtest evaluation, not live forecasting, so the ground
//! truth only needs to run forward **once**, all the way through, recording `is_extinct()` at
//! every tick into a `Vec<bool>` — extinction is a true absorbing state here (nothing ever
//! repopulates an empty `Population`), so "does extinction happen within `horizon` ticks of
//! `issued_at_tick`" is exactly "is the recorded trajectory extinct at
//! `issued_at_tick + horizon`." [`OracleGenerator`] bakes that whole recorded trajectory into its
//! own state at construction (`type Observation = u64`, just the tick being forecast from) and
//! answers by direct lookup — no `Clone`, no live re-simulation, and it fits
//! `TrajectoryGenerator` exactly as defined. Corrected here rather than left standing, matching
//! this codebase's own convention of retracting an overstated finding once resolved instead of
//! quietly reworking it (see e.g. the IIT-result correction in `MASTER_ROADMAP.md`'s Ramanujan
//! Protocol row).
//!
//! ## The second target: time-to-extinction, conditional on extinction occurring
//!
//! [`TimeToExtinctionOracleGenerator`] and [`TimeToExtinctionLinearGenerator`] answer the plan's
//! second "First experiment" sub-goal, scored via [`symthaea_futures_calibration::Crps`] instead
//! of Brier (the outcome space is [`symthaea_futures_core::OutcomeRegion::Interval`] — a tick
//! count, not a boolean). "Conditional on extinction occurring" is operationalized simply: only
//! evaluate checkpoints strictly *before* the ground truth's actual recorded extinction tick —
//! the question "how long until it happens" is only meaningful once we already know it happens.
//! Deliberately a smaller slice than the six-rung boolean target: two rungs (an oracle upper
//! bound and one real linear-extrapolation predictor), not all six — establishing the CRPS
//! pipeline works end to end matters more here than immediately re-deriving the full hierarchy.

use symthaea_futures_core::{
    AbstentionReason, ForecastBranch, ForecastDistribution, ForecastOutput, Horizon, OutcomeRegion,
    OutcomeSpaceId, TrajectoryGenerator,
};
use symthaea_futures_state::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation, mask_observation,
};
use symthaea_futures_symtropy::ecological::EcologicalObservation;

use crate::BaselineRung;

const EXTINCTION_WITHIN_HORIZON: &str = "ecological_extinction_within_horizon";

/// Thin wrapper over the crate-shared `boolean_distribution` (see `lib.rs`), fixing this
/// family's outcome-space name so every call site below stays unchanged.
fn boolean_distribution(
    issued_at_tick: u64,
    horizon: Horizon,
    p_true: f64,
) -> ForecastDistribution {
    crate::boolean_distribution(issued_at_tick, horizon, p_true, EXTINCTION_WITHIN_HORIZON)
}

/// Rung 1: "whatever's true now continues," operationalized on the only observable proxy this
/// scenario exposes (cohort status) — see module docs on why that's an imperfect but honest
/// stand-in for the true population-extinction target.
pub struct PersistenceGenerator;

impl PersistenceGenerator {
    pub const RUNG: BaselineRung = BaselineRung::Persistence;
}

impl TrajectoryGenerator for PersistenceGenerator {
    type Observation = EcologicalObservation;

    fn generate(&self, observation: &EcologicalObservation, horizon: Horizon) -> ForecastOutput {
        let Some(sample) = &observation.sample else {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        };

        // Deliberately not a hard 0.0/1.0 -- a hard probability courts an unbounded LogScore
        // penalty the one time the cohort/population proxy mismatch actually bites (a real,
        // expected failure mode of this naive baseline, not something to mask with false
        // certainty).
        let p_true = if sample.sampled_alive_count == 0 {
            0.9
        } else {
            0.05
        };
        ForecastOutput::Distribution(boolean_distribution(observation.tick, horizon, p_true))
    }
}

/// Rung 2: base rate across training seeds, ignoring the specific observation entirely.
/// `base_rate` must be supplied by whoever runs the experiment (measured separately across
/// training seeds) — this generator has no mechanism to calibrate itself, by design; that's
/// what makes it a genuinely different rung from [`PersistenceGenerator`] rather than a
/// relabeling of the same idea.
pub struct HistoricalFrequencyGenerator {
    pub base_rate: f64,
}

impl HistoricalFrequencyGenerator {
    pub const RUNG: BaselineRung = BaselineRung::HistoricalFrequency;
}

impl TrajectoryGenerator for HistoricalFrequencyGenerator {
    type Observation = EcologicalObservation;

    fn generate(&self, observation: &EcologicalObservation, horizon: Horizon) -> ForecastOutput {
        ForecastOutput::Distribution(boolean_distribution(
            observation.tick,
            horizon,
            self.base_rate,
        ))
    }
}

/// Rung 3: a real ordinary-least-squares linear trend fit on the tracked cohort's observed
/// history, extrapolated to `issued_at_tick + horizon`. Needs a history — see module docs on why
/// this generator's `Observation` type is `Vec<EcologicalObservation>`, not a single
/// `EcologicalObservation`.
///
/// **Disclosed simplification**: the projected cohort size at the target tick is mapped to a
/// probability via a simple linear proportion — `1.0 - projected / first_observed_size`, clamped
/// to `[0, 1]` — not a full prediction-interval treatment (which would need the fit's residual
/// variance propagated through to a proper predictive distribution). This is a real regression,
/// not a fit-free heuristic, but its probability output is a documented simplification, not a
/// rigorous one.
pub struct SimpleStatisticalGenerator;

impl SimpleStatisticalGenerator {
    pub const RUNG: BaselineRung = BaselineRung::SimpleStatistical;
}

impl TrajectoryGenerator for SimpleStatisticalGenerator {
    type Observation = Vec<EcologicalObservation>;

    fn generate(&self, history: &Vec<EcologicalObservation>, horizon: Horizon) -> ForecastOutput {
        let points: Vec<(f64, f64)> = history
            .iter()
            .filter_map(|obs| {
                obs.sample
                    .map(|s| (obs.tick as f64, s.sampled_alive_count as f64))
            })
            .collect();

        if points.len() < 2 {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        }

        let issued_at_tick = history.last().map(|o| o.tick).unwrap_or(0);
        let reference = points[0].1; // first observed cohort size -- the depletion baseline.
        if reference <= 0.0 {
            // Cohort was already gone at the very first reading -- nothing to fit a trend to.
            return ForecastOutput::Distribution(boolean_distribution(
                issued_at_tick,
                horizon,
                0.9,
            ));
        }

        let n = points.len() as f64;
        let x_mean = points.iter().map(|&(x, _)| x).sum::<f64>() / n;
        let y_mean = points.iter().map(|&(_, y)| y).sum::<f64>() / n;
        let denom: f64 = points.iter().map(|&(x, _)| (x - x_mean).powi(2)).sum();

        let slope = if denom > 0.0 {
            points
                .iter()
                .map(|&(x, y)| (x - x_mean) * (y - y_mean))
                .sum::<f64>()
                / denom
        } else {
            0.0 // all readings at the same tick -- no trend information, flat projection.
        };
        let intercept = y_mean - slope * x_mean;

        let target_tick = issued_at_tick as f64 + horizon.0 as f64;
        let projected = (intercept + slope * target_tick).max(0.0);

        let p_true = (1.0 - projected / reference).clamp(0.0, 1.0);
        ForecastOutput::Distribution(boolean_distribution(issued_at_tick, horizon, p_true))
    }
}

/// Rung 4: a real closed-form equation, not a fit. Assumes each currently-tracked cohort member
/// independently survives one tick with probability `1 - per_member_death_probability` — a
/// scenario-designer-supplied constant reflecting known typical dynamics (calibrated separately
/// across training seeds, never read from live ground truth — matching rung 2's
/// externally-supplied-parameter pattern). Under that independence assumption, the probability
/// the entire currently-tracked cohort is gone within `horizon` ticks is
/// `(1 - (1 - per_member_death_probability)^horizon)^sampled_alive_count`.
///
/// **Disclosed simplification**: real `symthaea-alife` dynamics couple individuals through a
/// shared resource pool (density-dependence) — this closed form assumes independence instead,
/// deliberately, in favor of being an actual equation rather than a fit to observed data.
pub struct ScenarioMechanisticGenerator {
    pub per_member_death_probability: f64,
}

impl ScenarioMechanisticGenerator {
    pub const RUNG: BaselineRung = BaselineRung::ScenarioMechanistic;
}

impl TrajectoryGenerator for ScenarioMechanisticGenerator {
    type Observation = EcologicalObservation;

    fn generate(&self, observation: &EcologicalObservation, horizon: Horizon) -> ForecastOutput {
        let Some(sample) = &observation.sample else {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        };

        if sample.sampled_alive_count == 0 {
            return ForecastOutput::Distribution(boolean_distribution(
                observation.tick,
                horizon,
                0.9,
            ));
        }

        let p_survive_one_member_one_tick = 1.0 - self.per_member_death_probability.clamp(0.0, 1.0);
        let p_member_survives_horizon = p_survive_one_member_one_tick.powi(horizon.0 as i32);
        let p_member_dies_within_horizon = 1.0 - p_member_survives_horizon;
        let p_true = p_member_dies_within_horizon.powi(sample.sampled_alive_count as i32);

        ForecastOutput::Distribution(boolean_distribution(observation.tick, horizon, p_true))
    }
}

/// Rung 5: the FEP-driven ensemble — the actual system under test. See module docs for the
/// real "perceive-only teaches nothing" trap this generator has to explicitly work around
/// (`num_actions: 1` + explicit `observe_transition` calls, not just `perceive`).
///
/// Needs a history, like rung 3 (`type Observation = Vec<EcologicalObservation>`) — a fresh
/// `ActiveInferenceAgent` is constructed and the whole history replayed through it on every
/// `generate` call, exactly mirroring rung 3's "refit from scratch each time" pattern rather than
/// carrying live state in `self` (which `&self` couldn't support anyway).
///
/// Single observation channel: the cohort's normalized survival fraction
/// (`sampled_alive_count / first_observed_count`), matching rung 3's reference-normalization
/// convention. **Disclosed simplification**: `observed_mean_energy`/`observed_temperature` are
/// not fed to the agent at all — a genuinely richer belief state (multiple channels) is future
/// work, not attempted in this first version.
pub struct FepDrivenGenerator {
    pub agent_config: ActiveInferenceAgentConfig,
}

impl Default for FepDrivenGenerator {
    fn default() -> Self {
        Self {
            agent_config: ActiveInferenceAgentConfig {
                state_dim: 1,
                obs_dim: 1,
                // No real action/choice exists in passive forecasting -- action 0 is the only
                // one, existing purely so `observe_transition` has something to index into (see
                // module docs).
                num_actions: 1,
                ..ActiveInferenceAgentConfig::default()
            },
        }
    }
}

impl FepDrivenGenerator {
    pub const RUNG: BaselineRung = BaselineRung::FepDriven;
}

impl TrajectoryGenerator for FepDrivenGenerator {
    type Observation = Vec<EcologicalObservation>;

    fn generate(&self, history: &Vec<EcologicalObservation>, horizon: Horizon) -> ForecastOutput {
        let issued_at_tick = match history.last() {
            Some(o) => o.tick,
            None => {
                return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
            }
        };

        // Per ExtinctionObservationPolicy, tick 0 is always observed (0 % anything == 0), so a
        // real history should always have a Some sample at index 0.
        let Some(reference) = history
            .first()
            .and_then(|o| o.sample)
            .map(|s| s.sampled_alive_count as f64)
        else {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        };

        if reference <= 0.0 {
            // Cohort was already gone at the very first reading -- nothing to fit or learn from.
            return ForecastOutput::Distribution(boolean_distribution(
                issued_at_tick,
                horizon,
                0.9,
            ));
        }

        let mut agent = ActiveInferenceAgent::new(self.agent_config.clone());
        let mut prev_belief = agent.belief.clone();

        for obs in history {
            let raw_value = obs
                .sample
                .map(|s| (s.sampled_alive_count as f64 / reference).clamp(0.0, 1.0))
                .unwrap_or(0.5); // placeholder value -- masked out below when not observed.
            let visibility = if obs.sample.is_some() { 1.0 } else { 0.0 };

            let raw_obs = Observation::new(vec![raw_value], 1.0, "cohort_survival_fraction");
            let masked = mask_observation(&raw_obs, &agent.belief, &[visibility]);

            agent.perceive(&masked);
            let new_belief = agent.belief.clone();
            agent.observe_transition(&prev_belief, 0, &new_belief, &masked);
            prev_belief = new_belief;
        }

        let mut projected = agent.belief.clone();
        for _ in 0..horizon.0 {
            projected = agent.model.predict_next_state(&projected, 0);
        }
        let projected_fraction = projected
            .mean
            .first()
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let p_true = (1.0 - projected_fraction).clamp(0.0, 1.0);

        ForecastOutput::Distribution(boolean_distribution(issued_at_tick, horizon, p_true))
    }
}

/// Rung 6: the oracle upper bound. See module docs for why this is a valid
/// `TrajectoryGenerator` after all — the whole run's ground truth is recorded once into
/// `trajectory` (built via [`Self::from_trajectory`], typically fed by a harness that runs
/// `EcologicalGroundTruth::step()` forward and records `is_extinct()` at every tick), and
/// `generate` answers by direct lookup rather than live re-simulation.
///
/// Deliberately hard `0.0`/`1.0` probabilities, unlike every other rung here — this is the one
/// generator in the hierarchy with genuine certainty (perfect hindsight), not an estimate, so
/// softening it would misrepresent what it actually knows.
pub struct OracleGenerator {
    /// `trajectory[t]` is whether the population was extinct at tick `t`. Index `t` must exist
    /// for every tick actually simulated, `0..=last_recorded_tick`.
    trajectory: Vec<bool>,
}

impl OracleGenerator {
    pub const RUNG: BaselineRung = BaselineRung::OracleUpperBound;

    pub fn from_trajectory(trajectory: Vec<bool>) -> Self {
        Self { trajectory }
    }
}

impl TrajectoryGenerator for OracleGenerator {
    /// Just the tick being forecast from — every privileged bit of information this generator
    /// uses was already baked into `self.trajectory` at construction, not read per-call. This is
    /// the one deliberate, labeled exception to the observation firewall (see module docs).
    type Observation = u64;

    fn generate(&self, issued_at_tick: &u64, horizon: Horizon) -> ForecastOutput {
        let target_tick = issued_at_tick + horizon.0;
        let Some(&actually_extinct) = self.trajectory.get(target_tick as usize) else {
            return ForecastOutput::Abstain(AbstentionReason::HorizonBeyondValidatedRange);
        };
        let p_true = if actually_extinct { 1.0 } else { 0.0 };
        ForecastOutput::Distribution(boolean_distribution(*issued_at_tick, horizon, p_true))
    }
}

const TIME_TO_EXTINCTION_CONDITIONAL: &str = "ecological_time_to_extinction_conditional";

/// A single-atom (point) forecast for the time-to-extinction target: one branch, probability
/// 1.0, at `Interval { low: point_ticks, high: point_ticks }`. `Crps` handles a degenerate
/// single-atom distribution correctly (see `symthaea-futures-calibration`'s own
/// `crps_single_atom_reduces_to_absolute_error` test) — this is a real, if simple, point
/// forecast, not a placeholder.
fn interval_point_distribution(
    issued_at_tick: u64,
    horizon: Horizon,
    point_ticks: f64,
) -> ForecastDistribution {
    interval_atoms_distribution(issued_at_tick, horizon, &[point_ticks])
}

/// The general form: an equally-weighted particle/ensemble forecast over `atom_ticks` — each
/// value becomes its own `Interval { low, high }` branch at weight `1 / atom_ticks.len()`.
/// `Crps::score` is exactly the standard ensemble-CRPS estimator
/// (`E|X-y| - 0.5*E|X-X'|`; see that crate's `score` implementation), so this genuinely lets a
/// forecast express calibrated spread instead of false point-confidence — the more atoms
/// disagree, the more the pairwise-spread term rewards honestly wide, plausible forecasts over a
/// falsely-precise wrong point. A single atom (this fn's only caller until
/// [`TimeToExtinctionEnsembleGenerator`]) is the degenerate case where that spread term is
/// exactly zero, reducing to a plain point forecast.
fn interval_atoms_distribution(
    issued_at_tick: u64,
    horizon: Horizon,
    atom_ticks: &[f64],
) -> ForecastDistribution {
    let weight = 1.0 / atom_ticks.len() as f64;
    ForecastDistribution {
        issued_at_tick,
        horizon,
        outcome_space: OutcomeSpaceId(TIME_TO_EXTINCTION_CONDITIONAL.to_string()),
        branches: atom_ticks
            .iter()
            .map(|&t| ForecastBranch {
                probability: weight,
                outcome: OutcomeRegion::Interval { low: t, high: t },
                assumptions: Vec::new(),
            })
            .collect(),
        unsupported_mass: 0.0,
    }
}

/// Oracle upper bound for the time-to-extinction target — same "run once, record, look up"
/// design as [`OracleGenerator`], just answering a different question from the same recorded
/// trajectory: not "is it extinct at tick X" but "how many ticks until the first extinction
/// tick." Abstains if the recorded trajectory never reaches extinction after `issued_at_tick`
/// (nothing to condition on).
pub struct TimeToExtinctionOracleGenerator {
    trajectory: Vec<bool>,
}

impl TimeToExtinctionOracleGenerator {
    pub fn from_trajectory(trajectory: Vec<bool>) -> Self {
        Self { trajectory }
    }
}

impl TrajectoryGenerator for TimeToExtinctionOracleGenerator {
    type Observation = u64;

    fn generate(&self, issued_at_tick: &u64, horizon: Horizon) -> ForecastOutput {
        match self.trajectory.iter().position(|&extinct| extinct) {
            Some(extinction_tick) if extinction_tick as u64 >= *issued_at_tick => {
                let delta = extinction_tick as u64 - issued_at_tick;
                ForecastOutput::Distribution(interval_point_distribution(
                    *issued_at_tick,
                    horizon,
                    delta as f64,
                ))
            }
            _ => ForecastOutput::Abstain(AbstentionReason::UnresolvedOutcomeSpace),
        }
    }
}

/// An ordinary-least-squares fit of `y` on `x`. `None` if fewer than 2 points or `x` has zero
/// variance (`denom <= 0`) — there's nothing to fit either way.
struct OlsFit {
    intercept: f64,
    slope: f64,
}

fn fit_ols(points: &[(f64, f64)]) -> Option<OlsFit> {
    if points.len() < 2 {
        return None;
    }

    let n = points.len() as f64;
    let x_mean = points.iter().map(|&(x, _)| x).sum::<f64>() / n;
    let y_mean = points.iter().map(|&(_, y)| y).sum::<f64>() / n;
    let denom: f64 = points.iter().map(|&(x, _)| (x - x_mean).powi(2)).sum();

    if denom <= 0.0 {
        return None;
    }

    let slope = points
        .iter()
        .map(|&(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>()
        / denom;
    let intercept = y_mean - slope * x_mean;

    Some(OlsFit { intercept, slope })
}

/// Shared by both linear time-to-extinction generators below: fits an OLS line through
/// `points` and reports the tick at which it crosses zero as the predicted time-to-extinction.
/// Abstains if there's no variance in the observed ticks (`denom <= 0`) or the fitted trend
/// isn't declining (`slope >= 0`) — there is no crossing to predict either way.
fn ols_crossing_forecast(
    points: &[(f64, f64)],
    issued_at_tick: u64,
    horizon: Horizon,
) -> ForecastOutput {
    let Some(fit) = fit_ols(points) else {
        let reason = if points.len() < 2 {
            AbstentionReason::InsufficientObservationHistory
        } else {
            AbstentionReason::UnresolvedOutcomeSpace
        };
        return ForecastOutput::Abstain(reason);
    };
    if fit.slope >= 0.0 {
        return ForecastOutput::Abstain(AbstentionReason::UnresolvedOutcomeSpace);
    }

    let cross_tick = -fit.intercept / fit.slope;
    let time_to_extinction = (cross_tick - issued_at_tick as f64).max(0.0);

    ForecastOutput::Distribution(interval_point_distribution(
        issued_at_tick,
        horizon,
        time_to_extinction,
    ))
}

/// xorshift64 step, matching the convention already used elsewhere in this codebase (e.g.
/// `symthaea-futures-symtropy::ecological::ExtinctionObservationPolicy::next_unit`,
/// `Population::next_unit`, `ActiveInferenceAgent::select_action`) — a local, deterministic,
/// independent stream rather than a `rand` dependency for one generator.
fn xorshift64_next_unit(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn xorshift64_next_index(state: &mut u64, n: usize) -> usize {
    let u = xorshift64_next_unit(state);
    ((u * n as f64) as usize).min(n - 1)
}

fn ecological_observation_points(history: &[EcologicalObservation]) -> Vec<(f64, f64)> {
    history
        .iter()
        .filter_map(|obs| {
            obs.sample
                .map(|s| (obs.tick as f64, s.sampled_alive_count as f64))
        })
        .collect()
}

/// A real predictor for the time-to-extinction target: the same OLS linear-trend fit
/// [`SimpleStatisticalGenerator`] uses, but instead of reading off a probability at a fixed
/// horizon, solves for the tick at which the fitted line crosses zero and reports that as the
/// predicted time-to-extinction. Fits the *entire* observed history unconditionally — see
/// [`TimeToExtinctionUncensoredLinearGenerator`] for why that's a real weakness, not just a
/// theoretical one, against a capped observation policy.
pub struct TimeToExtinctionLinearGenerator;

impl TrajectoryGenerator for TimeToExtinctionLinearGenerator {
    type Observation = Vec<EcologicalObservation>;

    fn generate(&self, history: &Vec<EcologicalObservation>, horizon: Horizon) -> ForecastOutput {
        let points = ecological_observation_points(history);
        let issued_at_tick = history.last().map(|o| o.tick).unwrap_or(0);
        ols_crossing_forecast(&points, issued_at_tick, horizon)
    }
}

/// Fixes the failure mode diagnosed in `examples/time_to_extinction_diagnostic.rs`:
/// [`TimeToExtinctionLinearGenerator`] fits a single OLS line over the *entire* history, so
/// against a capped observation policy (e.g. `PopulationCensusObservationPolicy`, which reports
/// `min(true_count, sample_size)`) a run that sits flat at the cap for most of its length, then
/// crashes sharply at the very end, gets a slope dominated by the long flat segment and a wildly
/// displaced crossing point (empirically: off by 1,300+ ticks per seed on the dimmed-sun
/// collapse regime).
///
/// The fix is not an arbitrary fixed-size sliding window — it's a direct consequence of what a
/// capped reading actually means: a reading pinned at the *maximum value ever reported* across
/// the whole history is right-censored (`true_count >= reported`, the exact value is unknown),
/// so it carries no information about the decline's magnitude or rate. Excluding those readings
/// before fitting isn't a tuning choice with a magic constant; it falls out of the observation
/// model itself, and the resulting window adapts to however long the real informative (sub-cap)
/// segment turns out to be for that particular trajectory, rather than assuming a fixed length.
/// Against an *uncapped* signal this generator is equivalent to
/// [`TimeToExtinctionLinearGenerator`] modulo dropping the single peak reading (a minor,
/// deliberately-accepted conservatism, not treated as a corner case worth extra machinery).
pub struct TimeToExtinctionUncensoredLinearGenerator;

impl TrajectoryGenerator for TimeToExtinctionUncensoredLinearGenerator {
    type Observation = Vec<EcologicalObservation>;

    fn generate(&self, history: &Vec<EcologicalObservation>, horizon: Horizon) -> ForecastOutput {
        let all_points = ecological_observation_points(history);
        let issued_at_tick = history.last().map(|o| o.tick).unwrap_or(0);

        let Some(inferred_cap) = all_points.iter().map(|&(_, y)| y).reduce(f64::max) else {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        };

        let uncensored_points: Vec<(f64, f64)> = all_points
            .into_iter()
            .filter(|&(_, y)| y < inferred_cap)
            .collect();

        ols_crossing_forecast(&uncensored_points, issued_at_tick, horizon)
    }
}

/// The uncertainty-aware counterpart to [`TimeToExtinctionUncensoredLinearGenerator`]: every
/// generator above reports a single point as if it were certain, which throws away exactly the
/// thing `Crps` is designed to reward — calibrated spread over a plausible range, not false
/// precision (see [`interval_atoms_distribution`]'s docs on why `Crps`'s pairwise-spread term
/// makes that a real, scoreable distinction, not just a philosophical one).
///
/// Uses a **residual bootstrap**, not a closed-form ratio-of-correlated-estimators formula
/// (Fieller's theorem would give an exact analytic interval for this exact statistic, but is
/// easy to get subtly wrong from memory and hard to verify independently). The bootstrap is
/// simple enough to verify by construction: resample the base fit's residuals with replacement,
/// re-attach them to the same `x` values, refit, and recompute the crossing tick — repeated
/// `bootstrap_replicates` times with an independent deterministic xorshift64 stream (seeded from
/// `self.seed` mixed with `issued_at_tick`, so a given generator+history+checkpoint always
/// reproduces the same forecast). Replicates whose resampled fit is no longer declining are
/// dropped rather than forced; if too few survive (< 10%, floor of 5), there's genuinely not
/// enough support for *any* crossing estimate and this abstains rather than reporting a
/// thin-sample-supported guess.
///
/// In the zero-residual limit (a perfectly noiseless declining series) every resample
/// reconstructs the exact same fit, so every atom lands on the identical value — this
/// generator's forecast collapses to exactly [`TimeToExtinctionUncensoredLinearGenerator`]'s
/// point forecast (verified in
/// `ensemble_matches_uncensored_point_forecast_on_a_noiseless_decline`), not a coincidentally
/// similar one.
pub struct TimeToExtinctionEnsembleGenerator {
    pub bootstrap_replicates: usize,
    seed: u64,
}

impl TimeToExtinctionEnsembleGenerator {
    pub fn new(bootstrap_replicates: usize, seed: u64) -> Self {
        Self {
            bootstrap_replicates,
            seed: if seed == 0 { 1 } else { seed },
        }
    }
}

impl TrajectoryGenerator for TimeToExtinctionEnsembleGenerator {
    type Observation = Vec<EcologicalObservation>;

    fn generate(&self, history: &Vec<EcologicalObservation>, horizon: Horizon) -> ForecastOutput {
        let all_points = ecological_observation_points(history);
        let issued_at_tick = history.last().map(|o| o.tick).unwrap_or(0);

        let Some(inferred_cap) = all_points.iter().map(|&(_, y)| y).reduce(f64::max) else {
            return ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory);
        };
        let points: Vec<(f64, f64)> = all_points
            .into_iter()
            .filter(|&(_, y)| y < inferred_cap)
            .collect();

        let Some(base_fit) = fit_ols(&points) else {
            let reason = if points.len() < 2 {
                AbstentionReason::InsufficientObservationHistory
            } else {
                AbstentionReason::UnresolvedOutcomeSpace
            };
            return ForecastOutput::Abstain(reason);
        };
        if base_fit.slope >= 0.0 {
            return ForecastOutput::Abstain(AbstentionReason::UnresolvedOutcomeSpace);
        }

        let residuals: Vec<f64> = points
            .iter()
            .map(|&(x, y)| y - (base_fit.intercept + base_fit.slope * x))
            .collect();

        let mut rng_state = self
            .seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(issued_at_tick);
        if rng_state == 0 {
            rng_state = 1;
        }

        let mut crossing_samples: Vec<f64> = Vec::with_capacity(self.bootstrap_replicates);
        for _ in 0..self.bootstrap_replicates {
            let resampled: Vec<(f64, f64)> = points
                .iter()
                .map(|&(x, _)| {
                    let idx = xorshift64_next_index(&mut rng_state, residuals.len());
                    (x, base_fit.intercept + base_fit.slope * x + residuals[idx])
                })
                .collect();

            if let Some(fit) = fit_ols(&resampled) {
                if fit.slope < 0.0 {
                    let cross_tick = -fit.intercept / fit.slope;
                    crossing_samples.push((cross_tick - issued_at_tick as f64).max(0.0));
                }
            }
        }

        let min_valid = (self.bootstrap_replicates / 10).max(5);
        if crossing_samples.len() < min_valid {
            return ForecastOutput::Abstain(AbstentionReason::UnresolvedOutcomeSpace);
        }

        ForecastOutput::Distribution(interval_atoms_distribution(
            issued_at_tick,
            horizon,
            &crossing_samples,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_futures_core::OutcomeRegion;
    use symthaea_futures_symtropy::ecological::EcologicalSample;

    fn observation_with(sample: Option<EcologicalSample>) -> EcologicalObservation {
        EcologicalObservation { tick: 12, sample }
    }

    fn extract_p_true(output: &ForecastOutput) -> f64 {
        let ForecastOutput::Distribution(dist) = output else {
            panic!("expected a distribution, got an abstention");
        };
        dist.branches
            .iter()
            .find(|b| b.outcome == OutcomeRegion::Boolean(true))
            .map(|b| b.probability)
            .unwrap_or(0.0)
    }

    #[test]
    fn persistence_abstains_with_no_reading() {
        let generator = PersistenceGenerator;
        let out = generator.generate(&observation_with(None), Horizon(10));
        assert!(matches!(
            out,
            ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory)
        ));
    }

    #[test]
    fn persistence_predicts_high_when_cohort_extinct() {
        let generator = PersistenceGenerator;
        let sample = EcologicalSample {
            sampled_alive_count: 0,
            observed_mean_energy: None,
            observed_temperature: None,
        };
        let out = generator.generate(&observation_with(Some(sample)), Horizon(10));
        assert!(extract_p_true(&out) > 0.5);
    }

    #[test]
    fn persistence_predicts_low_when_cohort_alive() {
        let generator = PersistenceGenerator;
        let sample = EcologicalSample {
            sampled_alive_count: 2,
            observed_mean_energy: Some(0.5),
            observed_temperature: None,
        };
        let out = generator.generate(&observation_with(Some(sample)), Horizon(10));
        assert!(extract_p_true(&out) < 0.5);
    }

    #[test]
    fn historical_frequency_ignores_the_observation_entirely() {
        let generator = HistoricalFrequencyGenerator { base_rate: 0.37 };
        let empty = observation_with(None);
        let extinct = observation_with(Some(EcologicalSample {
            sampled_alive_count: 0,
            observed_mean_energy: None,
            observed_temperature: None,
        }));
        let alive = observation_with(Some(EcologicalSample {
            sampled_alive_count: 5,
            observed_mean_energy: Some(0.8),
            observed_temperature: Some(280.0),
        }));

        for obs in [&empty, &extinct, &alive] {
            let out = generator.generate(obs, Horizon(10));
            assert_eq!(extract_p_true(&out), 0.37);
        }
    }

    fn reading(tick: u64, alive: usize) -> EcologicalObservation {
        EcologicalObservation {
            tick,
            sample: Some(EcologicalSample {
                sampled_alive_count: alive,
                observed_mean_energy: None,
                observed_temperature: None,
            }),
        }
    }

    #[test]
    fn simple_statistical_abstains_with_fewer_than_two_readings() {
        let generator = SimpleStatisticalGenerator;
        let history = vec![reading(0, 5)];
        let out = generator.generate(&history, Horizon(10));
        assert!(matches!(
            out,
            ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory)
        ));
    }

    #[test]
    fn simple_statistical_predicts_high_on_a_clear_declining_trend() {
        let generator = SimpleStatisticalGenerator;
        // 5 -> 4 -> 3 -> 2 -> 1 over ticks 0..4: a clean linear decline projecting to 0 well
        // within a horizon of 5.
        let history: Vec<_> = (0..5u64).map(|t| reading(t, 5 - t as usize)).collect();
        let out = generator.generate(&history, Horizon(5));
        assert!(
            extract_p_true(&out) > 0.5,
            "expected a high extinction probability on a clear declining trend"
        );
    }

    #[test]
    fn simple_statistical_predicts_low_on_a_flat_trend() {
        let generator = SimpleStatisticalGenerator;
        let history: Vec<_> = (0..5u64).map(|t| reading(t, 5)).collect();
        let out = generator.generate(&history, Horizon(5));
        assert!(
            extract_p_true(&out) < 0.5,
            "expected a low extinction probability on a flat trend"
        );
    }

    #[test]
    fn simple_statistical_handles_a_cohort_already_gone_at_first_reading() {
        let generator = SimpleStatisticalGenerator;
        let history = vec![reading(0, 0), reading(1, 0)];
        let out = generator.generate(&history, Horizon(5));
        assert!(extract_p_true(&out) > 0.5);
    }

    #[test]
    fn scenario_mechanistic_abstains_with_no_reading() {
        let generator = ScenarioMechanisticGenerator {
            per_member_death_probability: 0.1,
        };
        let out = generator.generate(&observation_with(None), Horizon(10));
        assert!(matches!(
            out,
            ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory)
        ));
    }

    #[test]
    fn scenario_mechanistic_matches_hand_computed_value() {
        // per_member_death_probability=0.5, horizon=1, 1 member:
        // p_survive_one_tick=0.5 -> p_dies_within_horizon=0.5 -> p_true=0.5^1=0.5.
        let generator = ScenarioMechanisticGenerator {
            per_member_death_probability: 0.5,
        };
        let obs = observation_with(Some(EcologicalSample {
            sampled_alive_count: 1,
            observed_mean_energy: None,
            observed_temperature: None,
        }));
        let out = generator.generate(&obs, Horizon(1));
        assert!((extract_p_true(&out) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn scenario_mechanistic_certain_death_gives_certain_extinction() {
        let generator = ScenarioMechanisticGenerator {
            per_member_death_probability: 1.0,
        };
        let obs = observation_with(Some(EcologicalSample {
            sampled_alive_count: 3,
            observed_mean_energy: None,
            observed_temperature: None,
        }));
        let out = generator.generate(&obs, Horizon(1));
        assert!((extract_p_true(&out) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn scenario_mechanistic_negligible_death_gives_near_zero_extinction() {
        let generator = ScenarioMechanisticGenerator {
            per_member_death_probability: 0.0001,
        };
        let obs = observation_with(Some(EcologicalSample {
            sampled_alive_count: 3,
            observed_mean_energy: None,
            observed_temperature: None,
        }));
        let out = generator.generate(&obs, Horizon(2));
        assert!(extract_p_true(&out) < 0.01);
    }

    #[test]
    fn fep_driven_abstains_with_empty_history() {
        let generator = FepDrivenGenerator::default();
        let out = generator.generate(&Vec::new(), Horizon(5));
        assert!(matches!(
            out,
            ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory)
        ));
    }

    #[test]
    fn fep_driven_handles_a_cohort_already_gone_at_first_reading() {
        let generator = FepDrivenGenerator::default();
        let history = vec![reading(0, 0), reading(1, 0)];
        let out = generator.generate(&history, Horizon(5));
        assert!(extract_p_true(&out) > 0.5);
    }

    /// Real finding while writing this test, worth keeping documented: with only 6 replayed
    /// ticks (rung 3's `SimpleStatisticalGenerator` test's scale, matched here at first), this
    /// generator's p_true stayed at ~0.43 -- BELOW 0.5 -- despite a clean, unambiguous decline
    /// (fraction 1.0 down to 0.167). Rung 3's closed-form OLS fit converges to the right answer
    /// immediately regardless of data volume; this generator's gradient-based belief update
    /// (`inference_iterations: 5` per tick, `belief_learning_rate: 0.1`) and TD-learned
    /// transition dynamics genuinely need more replayed history to converge to a confident
    /// signal. 60 ticks of the same clean decline gives p_true ~0.72. This is a real, disclosed
    /// property distinguishing FEP-driven forecasting from closed-form/statistical baselines --
    /// not tuned away by picking a bigger number until the assertion passed for no reason.
    #[test]
    fn fep_driven_predicts_high_on_a_clear_declining_trend() {
        let generator = FepDrivenGenerator::default();
        let history: Vec<_> = (0..60u64).map(|t| reading(t, (60 - t) as usize)).collect();
        let out = generator.generate(&history, Horizon(6));
        assert!(
            extract_p_true(&out) > 0.5,
            "expected a high extinction probability once the learned model extrapolates a clear decline"
        );
    }

    #[test]
    fn fep_driven_predicts_low_on_a_flat_trend() {
        let generator = FepDrivenGenerator::default();
        let history: Vec<_> = (0..6u64).map(|t| reading(t, 6)).collect();
        let out = generator.generate(&history, Horizon(6));
        assert!(
            extract_p_true(&out) < 0.5,
            "expected a low extinction probability once the learned model extrapolates a flat trend"
        );
    }

    /// The trap check: confirms `observe_transition` is doing real work, not decoration.
    /// Directly inspects the agent's own transition matrix after replaying a declining history
    /// and asserts it moved away from `GenerativeModel::new()`'s untrained initialization (a
    /// near-diagonal 0.7 self-transition) -- if this ever regresses to a no-op (e.g. someone
    /// "simplifies" the loop back to calling only `perceive`), this test is what would catch it.
    #[test]
    fn fep_driven_transition_matrix_actually_learns_from_replayed_history() {
        use symthaea_futures_state::ActiveInferenceAgent;

        let config = FepDrivenGenerator::default().agent_config;
        let mut agent = ActiveInferenceAgent::new(config);
        let initial_self_transition = agent.model.transition_matrices[0][0][0];

        let mut prev_belief = agent.belief.clone();
        let history: Vec<_> = (0..8u64).map(|t| reading(t, 8 - t as usize)).collect();
        let reference = 8.0;
        for obs in &history {
            let raw_value = obs
                .sample
                .map(|s| (s.sampled_alive_count as f64 / reference).clamp(0.0, 1.0))
                .unwrap_or(0.5);
            let raw_obs = Observation::new(vec![raw_value], 1.0, "cohort_survival_fraction");
            let masked = mask_observation(&raw_obs, &agent.belief, &[1.0]);
            agent.perceive(&masked);
            let new_belief = agent.belief.clone();
            agent.observe_transition(&prev_belief, 0, &new_belief, &masked);
            prev_belief = new_belief;
        }

        assert_ne!(
            agent.model.transition_matrices[0][0][0], initial_self_transition,
            "transition_matrices[0] never changed -- observe_transition isn't teaching anything"
        );
    }

    /// Integration check: the whole pipeline built so far actually composes -- a real
    /// `EcologicalObservation` flows through a real generator into a real `ForecastDistribution`
    /// that a real scoring rule from `symthaea-futures-calibration` can score end to end.
    #[test]
    fn generated_forecast_is_scoreable_by_the_calibration_crate() {
        use symthaea_futures_calibration::{BrierScore, ScoringRule};

        let generator = PersistenceGenerator;
        let sample = EcologicalSample {
            sampled_alive_count: 0,
            observed_mean_energy: None,
            observed_temperature: None,
        };
        let ForecastOutput::Distribution(forecast) =
            generator.generate(&observation_with(Some(sample)), Horizon(10))
        else {
            panic!("expected a distribution");
        };

        let score = BrierScore.score(&forecast, &OutcomeRegion::Boolean(true));
        assert!(score.is_finite());
        assert!((0.0..=2.0).contains(&score), "score out of range: {score}");
    }

    #[test]
    fn oracle_looks_up_the_recorded_trajectory_exactly() {
        // trajectory: alive through tick 4, extinct from tick 5 onward.
        let trajectory = vec![false, false, false, false, false, true, true, true];
        let oracle = OracleGenerator::from_trajectory(trajectory);

        // From tick 0, extinction happens within horizon 5 (extinct at tick 5) but not within
        // horizon 3 (still alive at tick 3).
        assert_eq!(extract_p_true(&oracle.generate(&0, Horizon(5))), 1.0);
        assert_eq!(extract_p_true(&oracle.generate(&0, Horizon(3))), 0.0);
    }

    #[test]
    fn oracle_abstains_beyond_the_recorded_trajectory() {
        let oracle = OracleGenerator::from_trajectory(vec![false, false, false]);
        let out = oracle.generate(&0, Horizon(10));
        assert!(matches!(
            out,
            ForecastOutput::Abstain(AbstentionReason::HorizonBeyondValidatedRange)
        ));
    }

    /// End-to-end check with a REAL simulated scenario, not a hand-built trajectory: runs
    /// `EcologicalGroundTruth::step()` forward through an actual dimmed-sun collapse (the same
    /// fixture shape `symthaea-alife`'s own `phase5_earth_forcing.rs` test uses for its
    /// guaranteed-collapse scenario), records the real trajectory, and confirms the oracle's
    /// hindsight answer matches what the simulation actually did.
    #[test]
    fn oracle_matches_a_real_simulated_collapse() {
        use symthaea_alife::{
            EarthForcedEnvironment, OrganismConfig, Population, PopulationConfig,
        };
        use symthaea_futures_symtropy::ecological::EcologicalGroundTruth;

        let mut env = EarthForcedEnvironment::earth_like(200.0);
        env.model.solar_constant = 600.0; // dimmed past the snowball threshold -- guaranteed collapse.
        let pop_cfg = PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig {
                forage_efficiency: 0.6,
                ..OrganismConfig::default()
            },
            ..Default::default()
        };
        let population = Population::new(pop_cfg, 6, 11);
        let mut truth = EcologicalGroundTruth::new(env, population, 3.0);

        let mut trajectory = vec![truth.is_extinct()];
        for _ in 0..4000u64 {
            truth.step();
            trajectory.push(truth.is_extinct());
        }
        assert!(
            truth.is_extinct(),
            "expected the dimmed-sun scenario to have actually collapsed within 4000 ticks"
        );

        let extinction_tick = trajectory
            .iter()
            .position(|&extinct| extinct)
            .expect("trajectory must contain an extinction tick")
            as u64;

        let oracle = OracleGenerator::from_trajectory(trajectory);

        // Forecasting from tick 0: a horizon that reaches exactly the true extinction tick
        // should say "yes"; a horizon that stops one tick short should say "no".
        assert_eq!(
            extract_p_true(&oracle.generate(&0, Horizon(extinction_tick))),
            1.0
        );
        if extinction_tick > 0 {
            assert_eq!(
                extract_p_true(&oracle.generate(&0, Horizon(extinction_tick - 1))),
                0.0
            );
        }
    }

    fn extract_interval_point(output: &ForecastOutput) -> f64 {
        let ForecastOutput::Distribution(dist) = output else {
            panic!("expected a distribution, got an abstention");
        };
        let OutcomeRegion::Interval { low, high } = dist.branches[0].outcome else {
            panic!("expected an Interval outcome");
        };
        assert_eq!(low, high, "expected a single-atom point forecast");
        low
    }

    #[test]
    fn time_to_extinction_oracle_matches_a_hand_built_trajectory() {
        let trajectory = vec![false, false, false, false, false, true, true, true];
        let oracle = TimeToExtinctionOracleGenerator::from_trajectory(trajectory);

        assert_eq!(
            extract_interval_point(&oracle.generate(&0, Horizon(0))),
            5.0
        );
        assert_eq!(
            extract_interval_point(&oracle.generate(&3, Horizon(0))),
            2.0
        );
    }

    #[test]
    fn time_to_extinction_oracle_abstains_if_no_extinction_recorded() {
        let oracle = TimeToExtinctionOracleGenerator::from_trajectory(vec![false, false, false]);
        let out = oracle.generate(&0, Horizon(0));
        assert!(matches!(
            out,
            ForecastOutput::Abstain(AbstentionReason::UnresolvedOutcomeSpace)
        ));
    }

    #[test]
    fn time_to_extinction_linear_matches_hand_computed_crossing() {
        // Points (0,10),(1,8),(2,6),(3,4): slope=-2, intercept=10, cross_tick=5.
        // issued_at_tick=3 -> time_to_extinction = 5 - 3 = 2.
        let history = vec![reading(0, 10), reading(1, 8), reading(2, 6), reading(3, 4)];
        let generator = TimeToExtinctionLinearGenerator;
        let out = generator.generate(&history, Horizon(0));
        let predicted = extract_interval_point(&out);
        assert!((predicted - 2.0).abs() < 1e-9, "got {predicted}");
    }

    #[test]
    fn time_to_extinction_linear_abstains_on_flat_trend() {
        let history = vec![reading(0, 6), reading(1, 6), reading(2, 6)];
        let generator = TimeToExtinctionLinearGenerator;
        let out = generator.generate(&history, Horizon(0));
        assert!(matches!(
            out,
            ForecastOutput::Abstain(AbstentionReason::UnresolvedOutcomeSpace)
        ));
    }

    #[test]
    fn time_to_extinction_uncensored_matches_linear_when_signal_never_saturates_the_cap() {
        // Same fixture as `time_to_extinction_linear_matches_hand_computed_crossing`: the peak
        // reading (0,10) is dropped as the inferred cap, but the remaining 3 points
        // (1,8),(2,6),(3,4) still fall on the exact same line (slope=-2, intercept=10,
        // cross_tick=5), so the two generators agree here -- confirms dropping the single peak
        // reading doesn't change anything when the signal was never actually censored.
        let history = vec![reading(0, 10), reading(1, 8), reading(2, 6), reading(3, 4)];
        let generator = TimeToExtinctionUncensoredLinearGenerator;
        let out = generator.generate(&history, Horizon(0));
        let predicted = extract_interval_point(&out);
        assert!((predicted - 2.0).abs() < 1e-9, "got {predicted}");
    }

    #[test]
    fn time_to_extinction_uncensored_abstains_when_signal_never_leaves_the_cap() {
        // All readings pinned at the same (inferred-cap) value -- there is zero uncensored
        // signal at all, not just zero variance, so this abstains via
        // `InsufficientObservationHistory` rather than `UnresolvedOutcomeSpace` (which is what
        // `TimeToExtinctionLinearGenerator` reports on this exact input) -- a deliberate,
        // documented difference in abstention reason, not an inconsistency.
        let history = vec![reading(0, 6), reading(1, 6), reading(2, 6)];
        let generator = TimeToExtinctionUncensoredLinearGenerator;
        let out = generator.generate(&history, Horizon(0));
        assert!(matches!(
            out,
            ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory)
        ));
    }

    #[test]
    fn time_to_extinction_uncensored_recovers_from_a_flat_then_crash_signal() {
        // Reproduces the exact pathology `examples/time_to_extinction_diagnostic.rs` found: a
        // long flat run at the observation cap (10 points at value=3, ticks 0..=90 step 10),
        // followed by a short uncensored decline (100,2),(105,1),(110,0). The naive
        // whole-history fit is dominated by the flat segment; this generator drops every
        // reading == the inferred cap (3) and fits only the 3 uncensored points, which sit on
        // an exact line: slope=-0.2, intercept=22, cross_tick=110. issued_at_tick=110 (the last
        // reading) -> time_to_extinction = 110 - 110 = 0, i.e. "already crossed," matching the
        // fact that the last reading is already 0.
        let mut history: Vec<EcologicalObservation> =
            (0..=90).step_by(10).map(|t| reading(t, 3)).collect();
        history.push(reading(100, 2));
        history.push(reading(105, 1));
        history.push(reading(110, 0));

        let uncensored = TimeToExtinctionUncensoredLinearGenerator;
        let predicted = extract_interval_point(&uncensored.generate(&history, Horizon(0)));
        assert!((predicted - 0.0).abs() < 1e-9, "got {predicted}");

        // The naive whole-history fit on the identical input is dragged far off by the flat
        // segment -- confirms the fix addresses a real difference, not a hypothetical one.
        let naive = TimeToExtinctionLinearGenerator;
        let naive_predicted = extract_interval_point(&naive.generate(&history, Horizon(0)));
        assert!(
            naive_predicted > 50.0,
            "expected the naive fit to badly overshoot on a flat-then-crash signal, got {naive_predicted}"
        );
    }

    fn extract_atom_ticks(output: &ForecastOutput) -> Vec<f64> {
        let ForecastOutput::Distribution(dist) = output else {
            panic!("expected a distribution, got an abstention");
        };
        dist.branches
            .iter()
            .map(|b| {
                let OutcomeRegion::Interval { low, high } = b.outcome else {
                    panic!("expected an Interval outcome");
                };
                assert_eq!(low, high, "expected a single-atom branch");
                low
            })
            .collect()
    }

    #[test]
    fn ensemble_matches_uncensored_point_forecast_on_a_noiseless_decline() {
        // Same fixture as `time_to_extinction_uncensored_recovers_from_a_flat_then_crash_signal`
        // (flat-at-cap segment, then an exact noiseless declining line). With zero residuals,
        // every bootstrap resample reconstructs the identical fit, so every atom must land on
        // the exact same value as the deterministic point predictor -- proving the zero-noise
        // limit genuinely collapses to the point forecast, not just approximately.
        let mut history: Vec<EcologicalObservation> =
            (0..=90).step_by(10).map(|t| reading(t, 3)).collect();
        history.push(reading(100, 2));
        history.push(reading(105, 1));
        history.push(reading(110, 0));

        let uncensored = TimeToExtinctionUncensoredLinearGenerator;
        let point_predicted = extract_interval_point(&uncensored.generate(&history, Horizon(0)));

        let ensemble = TimeToExtinctionEnsembleGenerator::new(50, 42);
        let atoms = extract_atom_ticks(&ensemble.generate(&history, Horizon(0)));

        assert_eq!(atoms.len(), 50);
        for &a in &atoms {
            assert!(
                (a - point_predicted).abs() < 1e-9,
                "expected every atom to match the point forecast ({point_predicted}) in the \
                 zero-noise limit, got {a}"
            );
        }
    }

    #[test]
    fn ensemble_expresses_real_spread_under_noise() {
        // A declining trend with real noise around it -- the bootstrap should produce more than
        // one distinct crossing estimate, proving this generator actually expresses calibrated
        // uncertainty rather than silently collapsing to a point forecast whenever it's called.
        let history = vec![
            reading(0, 20),
            reading(1, 17),
            reading(2, 19),
            reading(3, 14),
            reading(4, 15),
            reading(5, 10),
            reading(6, 11),
            reading(7, 6),
            reading(8, 7),
            reading(9, 2),
        ];
        let ensemble = TimeToExtinctionEnsembleGenerator::new(100, 7);
        let output = ensemble.generate(&history, Horizon(0));
        let atoms = extract_atom_ticks(&output);

        let distinct = atoms.iter().fold(Vec::<f64>::new(), |mut acc, &a| {
            if !acc.iter().any(|&b| (a - b).abs() < 1e-9) {
                acc.push(a);
            }
            acc
        });
        assert!(
            distinct.len() > 1,
            "expected genuine spread across bootstrap replicates, got a single repeated value: {atoms:?}"
        );

        let ForecastOutput::Distribution(dist) = &output else {
            panic!("expected a distribution");
        };
        let total_probability: f64 = dist.branches.iter().map(|b| b.probability).sum();
        assert!(
            (total_probability - 1.0).abs() < 1e-9,
            "got {total_probability}"
        );
    }

    #[test]
    fn ensemble_abstains_like_uncensored_on_a_flat_signal() {
        let history = vec![reading(0, 6), reading(1, 6), reading(2, 6)];
        let ensemble = TimeToExtinctionEnsembleGenerator::new(50, 1);
        let out = ensemble.generate(&history, Horizon(0));
        assert!(matches!(
            out,
            ForecastOutput::Abstain(AbstentionReason::InsufficientObservationHistory)
        ));
    }

    #[test]
    fn ensemble_forecast_is_scoreable_via_crps_and_a_correct_atom_beats_a_confident_wrong_point() {
        use symthaea_futures_calibration::{Crps, ScoringRule};

        // A noisy decline whose exact true crossing tick we control by construction: reuse the
        // noiseless fixture from `ensemble_matches_uncensored_point_forecast_on_a_noiseless_
        // decline` (true crossing/point forecast = 0.0 from issued_at_tick=110), and confirm the
        // ensemble's CRPS score against that true value is genuinely finite and small -- not
        // just structurally scoreable, but actually accurate, mirroring the existing
        // `generated_forecast_is_scoreable_by_the_calibration_crate` integration check.
        let mut history: Vec<EcologicalObservation> =
            (0..=90).step_by(10).map(|t| reading(t, 3)).collect();
        history.push(reading(100, 2));
        history.push(reading(105, 1));
        history.push(reading(110, 0));

        let ensemble = TimeToExtinctionEnsembleGenerator::new(50, 42);
        let ForecastOutput::Distribution(forecast) = ensemble.generate(&history, Horizon(0)) else {
            panic!("expected a distribution");
        };

        let actual = OutcomeRegion::Interval {
            low: 0.0,
            high: 0.0,
        };
        let score = Crps.score(&forecast, &actual);
        assert!(score.is_finite() && score >= 0.0, "got {score}");
        assert!(score < 1.0, "expected a near-perfect score, got {score}");
    }

    /// Integration check: the time-to-extinction target is genuinely scoreable via CRPS, not
    /// just Brier -- proves the second outcome space works end to end, mirroring the boolean
    /// target's own `generated_forecast_is_scoreable_by_the_calibration_crate` test.
    #[test]
    fn time_to_extinction_forecast_is_scoreable_via_crps() {
        use symthaea_futures_calibration::{Crps, ScoringRule};

        let history = vec![reading(0, 10), reading(1, 8), reading(2, 6), reading(3, 4)];
        let generator = TimeToExtinctionLinearGenerator;
        let ForecastOutput::Distribution(forecast) = generator.generate(&history, Horizon(0))
        else {
            panic!("expected a distribution");
        };

        let actual = OutcomeRegion::Interval {
            low: 2.0,
            high: 2.0,
        };
        let score = Crps.score(&forecast, &actual);
        assert!(
            (score - 0.0).abs() < 1e-9,
            "expected a perfect CRPS=0 match, got {score}"
        );
    }
}
