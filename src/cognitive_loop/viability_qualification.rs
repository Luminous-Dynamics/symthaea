// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Experimental qualification bridge between the live FEP world-model field and the
//! deterministic synthetic organism micro-world.
//!
//! This is deliberately a qualification layer, not a second planner:
//! - the existing FEP ODE planner remains untouched;
//! - the existing FepModule::world_model becomes the predictor under test;
//! - the deterministic micro-world remains the sole source of ground-truth consequences;
//! - held-out evaluation freezes learning before scoring;
//! - confidence calibration is measured explicitly instead of inferred from sample count.
//!
//! Scientific boundary: these metrics qualify prediction, calibration, survival, recovery,
//! and transfer. They do not establish consciousness, sentience, subjective experience,
//! biological life, or agency.

use serde::{Deserialize, Serialize};

use super::fep_module::FepModule;
use super::viability_micro_world::{
    benchmark_scenarios, run_homeostatic_agent_horizon_scenario, MicroAction, MicroWorld,
    transition, MicroWorldObservation, MicroWorldPredictor, MicroWorldScenario,
    PersistencePredictor,
};

use crate::dynamics::ode_solvers::{
    OdeConfig, OdeResult, OdeSolver, OdeSolverEngine, OdeSystem,
};
use symthaea_fep::{GenerativeModel, HiddenState};

/// Common action-conditioned transition contract used by both learned and generative
/// transition models.
///
/// The interface is deliberately representation-neutral: callers provide a continuous
/// state vector and receive the model's one-step expected state. A model may optionally
/// expose action-level confidence when it has an evidence-bearing confidence signal.
pub trait ActionConditionedTransitionModel {
    fn state_dimension(&self) -> usize;
    fn action_count(&self) -> usize;

    fn predict_next_state(&self, state: &[f64], action: usize) -> Option<Vec<f64>>;

    fn action_confidence(&self, _action: usize) -> Option<f64> {
        None
    }
}

impl ActionConditionedTransitionModel for super::goal_world::WorldModelBridge {
    fn state_dimension(&self) -> usize {
        super::goal_world::WorldModelBridge::state_dimension(self)
    }

    fn action_count(&self) -> usize {
        super::goal_world::WorldModelBridge::action_count(self)
    }

    fn predict_next_state(&self, state: &[f64], action: usize) -> Option<Vec<f64>> {
        if state.len() != self.state_dimension() {
            return None;
        }

        let input: Vec<f32> = state.iter().map(|&value| value as f32).collect();
        self.predict_action(action, &input)
            .map(|predicted| predicted.into_iter().map(|value| value as f64).collect())
    }

    fn action_confidence(&self, action: usize) -> Option<f64> {
        super::goal_world::WorldModelBridge::action_confidence(self, action)
            .map(|value| value as f64)
    }
}

impl ActionConditionedTransitionModel for GenerativeModel {
    fn state_dimension(&self) -> usize {
        self.state_dim
    }

    fn action_count(&self) -> usize {
        self.num_actions
    }

    fn predict_next_state(&self, state: &[f64], action: usize) -> Option<Vec<f64>> {
        if state.len() != self.state_dim || action >= self.num_actions {
            return None;
        }

        let hidden = HiddenState {
            mean: state.to_vec(),
            precision: vec![1.0; self.state_dim],
            mode_probs: vec![1.0],
            current_mode: 0,
        };

        Some(symthaea_fep::GenerativeModel::predict_next_state(
            self, &hidden, action,
        )
        .mean)
    }
}

/// Continuous extension of an action-conditioned discrete transition model.
///
/// The field is explicitly defined as:
///
/// ds/dt = (F(s,a) - s) / tau
///
/// For a delta model, F(s,a)=s+delta, so the field becomes a constant action-specific
/// velocity. This is a qualification adapter, not an assertion that this extension is
/// the only scientifically valid continuous-time realization.
pub struct ActionConditionedTransitionOde<'a, M: ActionConditionedTransitionModel + ?Sized> {
    pub model: &'a M,
    pub action: usize,
    pub tau: f64,
    pub dim: usize,
}

impl<'a, M: ActionConditionedTransitionModel + ?Sized>
    ActionConditionedTransitionOde<'a, M>
{
    pub fn new(model: &'a M, action: usize, tau: f64) -> Option<Self> {
        let dim = model.state_dimension();
        if dim == 0
            || action >= model.action_count()
            || !tau.is_finite()
            || tau <= 0.0
        {
            return None;
        }

        Some(Self {
            model,
            action,
            tau,
            dim,
        })
    }
}

impl<M: ActionConditionedTransitionModel + ?Sized> OdeSystem
    for ActionConditionedTransitionOde<'_, M>
{
    fn dimension(&self) -> usize {
        self.dim
    }

    fn evaluate(&self, _t: f64, state: &[f64], derivative: &mut [f64]) {
        let Some(next) = self.model.predict_next_state(state, self.action) else {
            derivative.fill(0.0);
            return;
        };

        if next.len() != self.dim {
            derivative.fill(0.0);
            return;
        }

        for i in 0..self.dim {
            derivative[i] = (next[i] - state[i]) / self.tau;
        }
    }
}

/// Result of a continuous trajectory rollout through a shared transition model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContinuousTransitionRollout {
    pub action: usize,
    pub horizon_seconds: f64,
    pub tau: f64,
    pub ode_steps: usize,
    pub initial_state: Vec<f64>,
    pub one_step_prediction: Vec<f64>,
    pub terminal_state: Vec<f64>,
    pub action_confidence: Option<f64>,
}

/// Roll a single action through the shared transition interface using the existing
/// Dormand-Prince ODE engine.
///
/// This function is intentionally isolated from runtime policy. Its purpose is to test
/// whether a model that predicts one-step consequences also supports coherent continuous
/// extrapolation.
pub fn roll_transition_model_trajectory<M: ActionConditionedTransitionModel + ?Sized>(
    model: &M,
    state: &[f64],
    action: usize,
    horizon_seconds: f64,
    tau: f64,
    max_steps: usize,
) -> Option<ContinuousTransitionRollout> {
    if state.len() != model.state_dimension()
        || !horizon_seconds.is_finite()
        || horizon_seconds <= 0.0
        || max_steps == 0
    {
        return None;
    }

    let one_step_prediction = model.predict_next_state(state, action)?;
    if one_step_prediction.len() != state.len()
        || one_step_prediction.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let ode_config = OdeConfig {
        solver: OdeSolver::DormandPrince,
        dt: 0.01,
        tolerance: 1e-4,
        max_step: (horizon_seconds / 5.0).max(1e-6),
        min_step: 1e-8,
        max_iterations: max_steps,
    };
    let solver = OdeSolverEngine::new(ode_config, state.len());
    let ode_system = ActionConditionedTransitionOde::new(model, action, tau)?;

    let result: OdeResult =
        solver.solve(&ode_system, state, (0.0, horizon_seconds));

    let terminal_state = result
        .states
        .last()
        .cloned()
        .filter(|values| values.len() == state.len())?;

    Some(ContinuousTransitionRollout {
        action,
        horizon_seconds,
        tau,
        ode_steps: result.times.len(),
        initial_state: state.to_vec(),
        one_step_prediction,
        terminal_state,
        action_confidence: model.action_confidence(action),
    })
}

/// Confidence calibration statistics for continuous one-step state forecasts.
///
/// realized_accuracy = 1 - MAE is valid here because every micro-world state channel
/// is bounded to [0, 1]. This is an operational forecast score, not a claim that the
/// model's confidence is a calibrated probability of a discrete event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceCalibration {
    pub sample_count: u64,
    pub expected_calibration_error: f64,
    pub confidence_accuracy_mse: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct CalibrationBin {
    count: u64,
    confidence_sum: f64,
    accuracy_sum: f64,
}

#[derive(Debug, Clone)]
struct CalibrationAccumulator {
    bins: [CalibrationBin; 10],
    squared_error_sum: f64,
    count: u64,
}

impl Default for CalibrationAccumulator {
    fn default() -> Self {
        Self {
            bins: [CalibrationBin::default(); 10],
            squared_error_sum: 0.0,
            count: 0,
        }
    }
}

impl CalibrationAccumulator {
    fn record(&mut self, confidence: f64, mae: f64) {
        let confidence = confidence.clamp(0.0, 1.0);
        let realized_accuracy = (1.0 - mae.clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let bin_index = ((confidence * 10.0).floor() as usize).min(9);
        let bin = &mut self.bins[bin_index];

        bin.count = bin.count.saturating_add(1);
        bin.confidence_sum += confidence;
        bin.accuracy_sum += realized_accuracy;

        let squared_error = (confidence - realized_accuracy).powi(2);
        self.squared_error_sum += squared_error;
        self.count = self.count.saturating_add(1);
    }

    fn finish(&self) -> ConfidenceCalibration {
        if self.count == 0 {
            return ConfidenceCalibration {
                sample_count: 0,
                expected_calibration_error: 0.0,
                confidence_accuracy_mse: 0.0,
            };
        }

        let total = self.count as f64;
        let expected_calibration_error = self
            .bins
            .iter()
            .filter(|bin| bin.count > 0)
            .map(|bin| {
                let n = bin.count as f64;
                let mean_confidence = bin.confidence_sum / n;
                let mean_accuracy = bin.accuracy_sum / n;
                (n / total) * (mean_confidence - mean_accuracy).abs()
            })
            .sum();

        ConfidenceCalibration {
            sample_count: self.count,
            expected_calibration_error,
            confidence_accuracy_mse: self.squared_error_sum / total,
        }
    }
}

/// Adapter exposing the live FepModule world-model through the deterministic
/// micro-world predictor contract.
struct FepWorldModelPredictor<'a> {
    bridge: &'a mut super::goal_world::WorldModelBridge,
}

impl MicroWorldPredictor for FepWorldModelPredictor<'_> {
    fn predict(
        &self,
        state: MicroWorldObservation,
        action: MicroAction,
    ) -> MicroWorldObservation {
        let mut encoded = vec![0.0f32; 64];
        encoded[0] = state.energy as f32;
        encoded[1] = state.integrity as f32;
        encoded[2] = state.knowledge as f32;
        encoded[3] = state.threat as f32;
        encoded[4] = state.progress as f32;

        let Some(predicted) = self.bridge.predict_action(action.index(), &encoded) else {
            return state;
        };

        let get = |index: usize| predicted.get(index).copied().unwrap_or(f64::NAN);
        MicroWorldObservation {
            cycle: state.cycle.saturating_add(1),
            energy: get(0),
            integrity: get(1),
            knowledge: get(2),
            threat: get(3),
            progress: get(4),
        }
    }

    fn prediction_confidence(&self, action: MicroAction) -> f64 {
        self.bridge
            .action_confidence(action.index())
            .unwrap_or(0.0) as f64
    }

    fn observe_transition(
        &mut self,
        before: MicroWorldObservation,
        action: MicroAction,
        after: MicroWorldObservation,
    ) {
        let mut before_encoded = vec![0.0f32; 64];
        before_encoded[0] = before.energy as f32;
        before_encoded[1] = before.integrity as f32;
        before_encoded[2] = before.knowledge as f32;
        before_encoded[3] = before.threat as f32;
        before_encoded[4] = before.progress as f32;

        let mut after_encoded = vec![0.0f32; 64];
        after_encoded[0] = after.energy as f32;
        after_encoded[1] = after.integrity as f32;
        after_encoded[2] = after.knowledge as f32;
        after_encoded[3] = after.threat as f32;
        after_encoded[4] = after.progress as f32;

        let _ = self.bridge.observe_action_transition(
            action.index(),
            &before_encoded,
            &after_encoded,
        );
    }
}

/// Result of the grounded world-model qualification experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundedWorldModelQualificationReport {
    pub training_scenario: &'static str,
    pub held_out_scenario: &'static str,

    pub train_steps: u64,
    pub train_predictor_mae: f64,

    pub held_out_steps: u64,
    pub held_out_baseline_mae: f64,
    pub held_out_predictor_mae: f64,
    pub held_out_improvement_over_baseline: f64,
    pub held_out_confidence_calibration: ConfidenceCalibration,
    pub held_out_continuous_rollout_steps: u64,
    pub held_out_continuous_rollout_mae: f64,
    pub held_out_multi_horizon: Vec<HorizonQualificationPoint>,
    pub held_out_survived_fixed_schedule: bool,

    /// Frozen-model policy evaluation on states induced by the model's own actions.
    pub policy_induced_shift: PolicyInducedShiftReport,

    pub persistence_closed_loop_survived: bool,
    pub persistence_closed_loop_mean_oracle_horizon_regret: f64,
    pub persistence_recovery_rate: f64,
    pub persistence_mean_recovery_steps: f64,

    pub closed_loop_survived: bool,
    pub closed_loop_steps: u64,
    pub closed_loop_prediction_mae: f64,
    pub closed_loop_mean_oracle_horizon_regret: f64,
    pub closed_loop_min_actual_viability_margin: f64,
    pub closed_loop_execution_failures: usize,
    pub closed_loop_terminated_on_execution_failure: bool,
    pub perturbations_applied: usize,

    /// Time-to-recovery is measured as cycles required to regain the exact
    /// pre-perturbation viability margin. None means no recovery occurred.
    pub perturbation_recovery_steps: Vec<Option<u64>>,
    pub recovery_rate: f64,
    pub mean_recovery_steps: f64,
}

/// Accuracy profile for a frozen model as the temporal prediction horizon increases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HorizonQualificationPoint {
    pub horizon_steps: usize,
    pub samples: u64,
    pub mean_one_step_mae: f64,
    /// Error of the continuous ODE rollout against the deterministic oracle.
    pub mean_terminal_mae: f64,
    /// Error of repeatedly applying the frozen discrete predictor against the same oracle.
    pub mean_discrete_terminal_mae: f64,
    pub terminal_to_one_step_error_ratio: f64,
}

impl HorizonQualificationPoint {
    pub fn has_temporal_degradation(&self) -> bool {
        self.mean_terminal_mae > self.mean_one_step_mae
    }
}

/// Frozen policy evaluation that deliberately lets the model induce the visited-state distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyInducedShiftReport {
    pub steps: u64,
    pub baseline_mae: f64,
    pub predictor_mae: f64,
    pub improvement_over_baseline: f64,
    pub survival: bool,
    pub min_actual_viability_margin: f64,
    pub execution_failures: usize,
    pub terminated_on_execution_failure: bool,
    /// Ratio against the frozen fixed-schedule held-out predictor MAE.
    pub shift_error_ratio: f64,
}

impl GroundedWorldModelQualificationReport {
    pub fn transfer_beats_persistence(&self) -> bool {
        self.held_out_predictor_mae < self.held_out_baseline_mae
    }

    pub fn all_perturbations_recovered(&self) -> bool {
        self.recovery_rate >= 1.0
    }

    pub fn policy_regret_beats_persistence(&self) -> bool {
        self.closed_loop_mean_oracle_horizon_regret
            < self.persistence_closed_loop_mean_oracle_horizon_regret
    }
}

fn evaluate_frozen_scenario<P: MicroWorldPredictor>(
    predictor: &P,
    scenario: &MicroWorldScenario,
    max_cycles: u64,
) -> (
    u64,
    f64,
    f64,
    f64,
    ConfidenceCalibration,
    bool,
) {
    let mut world = MicroWorld::new(scenario.initial, max_cycles);
    let mut predictor_error = 0.0;
    let mut baseline_error = 0.0;
    let mut steps = 0u64;
    let mut calibration = CalibrationAccumulator::default();

    while !world.done() && steps < max_cycles {
        for (cycle, perturbation) in scenario.perturbations {
            if *cycle == steps {
                world.perturb(*perturbation);
            }
        }

        let before = world.observe();
        let action = scenario.schedule[steps as usize % scenario.schedule.len()];
        let predicted = predictor.predict(before, action);
        let confidence = predictor.prediction_confidence(action);
        let after = world.step(action);

        let mae = predicted.mean_absolute_delta(after);
        let baseline = PersistencePredictor::default()
            .predict(before, action)
            .mean_absolute_delta(after);

        predictor_error += mae;
        baseline_error += baseline;
        calibration.record(confidence, mae);
        steps = steps.saturating_add(1);
    }

    let denom = steps.max(1) as f64;
    let baseline_mae = baseline_error / denom;
    let predictor_mae = predictor_error / denom;
    let improvement = if baseline_mae <= f64::EPSILON {
        0.0
    } else {
        (baseline_mae - predictor_mae) / baseline_mae
    };

    (
        steps,
        baseline_mae,
        predictor_mae,
        improvement,
        calibration.finish(),
        world.observe().is_viable(),
    )
}

const QUALIFICATION_HORIZONS: [usize; 4] = [1, 2, 4, 8];
const QUALIFICATION_TAU: f64 = 0.1;
const QUALIFICATION_MAX_ODE_STEPS: usize = 200;

/// Frozen temporal-composition profile. No learning is permitted while scoring these points.
fn evaluate_multi_horizon<M: ActionConditionedTransitionModel + ?Sized>(
    model: &M,
    scenario: &MicroWorldScenario,
    max_cycles: u64,
) -> Vec<HorizonQualificationPoint> {
    let mut points = Vec::with_capacity(QUALIFICATION_HORIZONS.len());

    for &horizon_steps in &QUALIFICATION_HORIZONS {
        let horizon_seconds = horizon_steps as f64 * QUALIFICATION_TAU;
        let mut world = MicroWorld::new(scenario.initial, max_cycles);
        let mut one_step_error = 0.0;
        let mut continuous_terminal_error = 0.0;
        let mut discrete_terminal_error = 0.0;
        let mut samples = 0u64;
        let mut steps = 0u64;

        while !world.done() && steps < max_cycles {
            for (cycle, perturbation) in scenario.perturbations {
                if *cycle == steps {
                    world.perturb(*perturbation);
                }
            }

            let before = world.observe();
            let action = scenario.schedule[steps as usize % scenario.schedule.len()];
            let encoded = encode_micro_world_state(before);

            if let Some(rollout) = roll_transition_model_trajectory(
                model,
                &encoded,
                action.index(),
                horizon_seconds,
                QUALIFICATION_TAU,
                QUALIFICATION_MAX_ODE_STEPS,
            ) {
                let one_step_actual = transition(before, action);
                one_step_error +=
                    rollout.one_step_prediction[0..5]
                        .iter()
                        .zip([
                            one_step_actual.energy,
                            one_step_actual.integrity,
                            one_step_actual.knowledge,
                            one_step_actual.threat,
                            one_step_actual.progress,
                        ])
                        .map(|(predicted, actual)| (predicted - actual).abs())
                        .sum::<f64>()
                        / 5.0;

                let mut actual_terminal = before;
                for _ in 0..horizon_steps {
                    actual_terminal = transition(actual_terminal, action);
                }

                let mut discrete_predicted = encoded.clone();
                let discrete_valid = (0..horizon_steps).all(|_| {
                    let Some(next) =
                        model.predict_next_state(&discrete_predicted, action.index())
                    else {
                        return false;
                    };
                    discrete_predicted = next;
                    true
                });

                if discrete_valid && discrete_predicted.len() >= 5 {
                    let discrete_mae = [
                        (discrete_predicted[0] - actual_terminal.energy).abs(),
                        (discrete_predicted[1] - actual_terminal.integrity).abs(),
                        (discrete_predicted[2] - actual_terminal.knowledge).abs(),
                        (discrete_predicted[3] - actual_terminal.threat).abs(),
                        (discrete_predicted[4] - actual_terminal.progress).abs(),
                    ]
                    .iter()
                    .sum::<f64>()
                        / 5.0;
                    discrete_terminal_error += discrete_mae;
                }

                let predicted_terminal = MicroWorldObservation {
                    cycle: actual_terminal.cycle,
                    energy: rollout.terminal_state[0],
                    integrity: rollout.terminal_state[1],
                    knowledge: rollout.terminal_state[2],
                    threat: rollout.terminal_state[3],
                    progress: rollout.terminal_state[4],
                };

                continuous_terminal_error += predicted_terminal.mean_absolute_delta(actual_terminal);
                samples = samples.saturating_add(1);
            }

            world.step(action);
            steps = steps.saturating_add(1);
        }

        let denom = samples.max(1) as f64;
        let mean_one_step_mae = one_step_error / denom;
        let mean_terminal_mae = continuous_terminal_error / denom;
        let mean_discrete_terminal_mae = discrete_terminal_error / denom;
        let ratio = if mean_one_step_mae <= f64::EPSILON {
            if mean_terminal_mae <= f64::EPSILON { 1.0 } else { f64::INFINITY }
        } else {
            mean_terminal_mae / mean_one_step_mae
        };

        points.push(HorizonQualificationPoint {
            horizon_steps,
            samples,
            mean_one_step_mae,
            mean_terminal_mae,
            mean_discrete_terminal_mae,
            terminal_to_one_step_error_ratio: ratio,
        });
    }

    points
}

/// Evaluate the frozen model on states induced by its own horizon-aware policy.
fn evaluate_policy_induced_shift<P: MicroWorldPredictor>(
    predictor: &P,
    scenario: &MicroWorldScenario,
    max_cycles: u64,
    policy_horizon: usize,
    policy_discount: f64,
    fixed_schedule_mae: f64,
) -> PolicyInducedShiftReport {
    let mut world = MicroWorld::new(scenario.initial, max_cycles);
    let policy = super::viability_micro_world::HomeostaticPolicy;
    let persistence = PersistencePredictor::default();
    let mut predictor_error = 0.0;
    let mut baseline_error = 0.0;
    let mut steps = 0u64;
    let mut min_actual_viability_margin = f64::INFINITY;
    let mut execution_failures = 0usize;
    let mut terminated_on_execution_failure = false;

    while !world.done() && steps < max_cycles {
        for (cycle, perturbation) in scenario.perturbations {
            if *cycle == steps {
                world.perturb(*perturbation);
            }
        }

        let before = world.observe();
        let (action, _, _) =
            policy.choose_horizon(predictor, before, policy_horizon, policy_discount);

        let predicted = predictor.predict(before, action);
        let baseline = persistence.predict(before, action);

        let after = match world.try_step(action) {
            Ok(after) => after,
            Err(_) => {
                execution_failures += 1;
                terminated_on_execution_failure = true;
                break;
            }
        };

        predictor_error += predicted.mean_absolute_delta(after);
        baseline_error += baseline.mean_absolute_delta(after);
        min_actual_viability_margin = min_actual_viability_margin.min(
            after.energy.min(after.integrity) - 0.08,
        );
        steps = steps.saturating_add(1);
    }

    let denom = steps.max(1) as f64;
    let baseline_mae = baseline_error / denom;
    let predictor_mae = predictor_error / denom;
    let improvement = if baseline_mae <= f64::EPSILON {
        0.0
    } else {
        (baseline_mae - predictor_mae) / baseline_mae
    };
    let shift_ratio = if fixed_schedule_mae <= f64::EPSILON {
        if predictor_mae <= f64::EPSILON { 1.0 } else { f64::INFINITY }
    } else {
        predictor_mae / fixed_schedule_mae
    };

    PolicyInducedShiftReport {
        steps,
        baseline_mae,
        predictor_mae,
        improvement_over_baseline: improvement,
        survival: world.observe().is_viable() && !terminated_on_execution_failure,
        min_actual_viability_margin: if min_actual_viability_margin.is_finite() {
            min_actual_viability_margin
        } else {
            world.observe().energy.min(world.observe().integrity) - 0.08
        },
        execution_failures,
        terminated_on_execution_failure,
        shift_error_ratio: shift_ratio,
    }
}

fn encode_micro_world_state(state: MicroWorldObservation) -> Vec<f64> {
    let mut encoded = vec![0.0f64; 64];
    encoded[0] = state.energy;
    encoded[1] = state.integrity;
    encoded[2] = state.knowledge;
    encoded[3] = state.threat;
    encoded[4] = state.progress;
    encoded
}

/// Evaluate a shared transition model's continuous extrapolation against repeated
/// deterministic oracle transitions from the same starting state.
fn evaluate_frozen_continuous_rollout<M: ActionConditionedTransitionModel + ?Sized>(
    model: &M,
    scenario: &MicroWorldScenario,
    max_cycles: u64,
    horizon_seconds: f64,
    tau: f64,
    max_steps: usize,
) -> (u64, f64) {
    if !horizon_seconds.is_finite() || horizon_seconds <= 0.0 || !tau.is_finite() || tau <= 0.0 {
        return (0, 0.0);
    }

    let repeated_steps = (horizon_seconds / tau).round().max(1.0) as usize;
    let mut world = MicroWorld::new(scenario.initial, max_cycles);
    let mut total_error = 0.0;
    let mut samples = 0u64;
    let mut steps = 0u64;

    while !world.done() && steps < max_cycles {
        for (cycle, perturbation) in scenario.perturbations {
            if *cycle == steps {
                world.perturb(*perturbation);
            }
        }

        let before = world.observe();
        let action = scenario.schedule[steps as usize % scenario.schedule.len()];
        let encoded = encode_micro_world_state(before);

        if let Some(rollout) = roll_transition_model_trajectory(
            model,
            &encoded,
            action.index(),
            horizon_seconds,
            tau,
            max_steps,
        ) {
            let mut actual = before;
            for _ in 0..repeated_steps {
                actual = transition(actual, action);
            }

            let predicted = MicroWorldObservation {
                cycle: actual.cycle,
                energy: rollout.terminal_state[0],
                integrity: rollout.terminal_state[1],
                knowledge: rollout.terminal_state[2],
                threat: rollout.terminal_state[3],
                progress: rollout.terminal_state[4],
            };

            total_error += predicted.mean_absolute_delta(actual);
            samples = samples.saturating_add(1);
        }

        world.step(action);
        steps = steps.saturating_add(1);
    }

    (samples, total_error / samples.max(1) as f64)
}

/// Replay the already selected closed-loop actions through the deterministic oracle
/// to measure recovery against the exact pre-perturbation viability margin.
fn measure_recovery(
    scenario: &MicroWorldScenario,
    actions: &[MicroAction],
    max_cycles: u64,
) -> (Vec<Option<u64>>, f64, f64) {
    let mut world = MicroWorld::new(scenario.initial, max_cycles);
    let mut recovery_targets: Vec<(u64, f64, Option<u64>)> = Vec::new();

    for (step_index, action) in actions.iter().copied().enumerate() {
        let step = step_index as u64;
        if world.done() || step >= max_cycles {
            break;
        }

        for (cycle, perturbation) in scenario.perturbations {
            if *cycle == step {
                let pre_margin = {
                    let state = world.observe();
                    state.energy.min(state.integrity) - 0.08
                };
                world.perturb(*perturbation);
                recovery_targets.push((step, pre_margin, None));
            }
        }

        let after = world.step(action);
        let margin = after.energy.min(after.integrity) - 0.08;

        for target in &mut recovery_targets {
            if target.2.is_none() && step > target.0 && margin >= target.1 {
                target.2 = Some(step.saturating_sub(target.0));
            }
        }
    }

    let recovered: Vec<Option<u64>> = recovery_targets
        .iter()
        .map(|(_, _, recovery)| *recovery)
        .collect();

    let recovery_rate = if recovered.is_empty() {
        1.0
    } else {
        recovered.iter().filter(|value| value.is_some()).count() as f64
            / recovered.len() as f64
    };

    let mean_recovery_steps = {
        let values: Vec<u64> = recovered.iter().filter_map(|value| *value).collect();
        if values.is_empty() {
            0.0
        } else {
            values.iter().map(|&value| value as f64).sum::<f64>() / values.len() as f64
        }
    };

    (recovered, recovery_rate, mean_recovery_steps)
}

impl FepModule {
    /// Roll the current grounded world model through the same ODE engine used by
    /// trajectory planning.
    ///
    /// This is an observational planning probe: it does not select an action,
    /// update model parameters, or alter runtime policy.
    pub fn rollout_current_world_model_trajectory(
        &self,
        state: &[f64],
        action: usize,
        horizon_seconds: f64,
        tau: f64,
        max_steps: usize,
    ) -> Option<ContinuousTransitionRollout> {
        roll_transition_model_trajectory(
            &self.world_model,
            state,
            action,
            horizon_seconds,
            tau,
            max_steps,
        )
    }

    /// Experimentally qualify the live FEP WorldModelBridge against deterministic
    /// synthetic-organism dynamics.
    ///
    /// The phases are intentionally ordered:
    /// 1. train on one scenario;
    /// 2. freeze learning and score a perturbed held-out scenario;
    /// 3. permit online adaptation only for the separate closed-loop survival/recovery run.
    ///
    /// This makes transfer/calibration evidence independent from the later policy run.
    pub fn qualify_world_model_against_micro_world(
        &mut self,
        train_cycles: u64,
        held_out_cycles: u64,
        policy_horizon: usize,
        policy_discount: f64,
    ) -> GroundedWorldModelQualificationReport {
        let scenarios = benchmark_scenarios();
        let training = &scenarios[0];
        let held_out = &scenarios[1];

        let mut persistence = PersistencePredictor::default();
        let persistence_closed_loop = run_homeostatic_agent_horizon_scenario(
            &mut persistence,
            held_out,
            held_out_cycles,
            policy_horizon,
            policy_discount,
        );
        let (_, persistence_recovery_rate, persistence_mean_recovery_steps) =
            measure_recovery(held_out, &persistence_closed_loop.actions, held_out_cycles);

        // Qualification must never reset or retrain the production world model in place.
        // Clone the exact current model so the experiment is isolated from runtime state.
        let mut qualification_world_model = self.world_model.clone();
        let mut predictor = FepWorldModelPredictor {
            bridge: &mut qualification_world_model,
        };

        let mut train_world = MicroWorld::new(training.initial, train_cycles);
        let mut train_error = 0.0;
        let mut train_steps = 0u64;

        while !train_world.done() && train_steps < train_cycles {
            for (cycle, perturbation) in training.perturbations {
                if *cycle == train_steps {
                    train_world.perturb(*perturbation);
                }
            }

            let before = train_world.observe();
            let action =
                training.schedule[train_steps as usize % training.schedule.len()];
            let predicted = predictor.predict(before, action);
            let after = train_world.step(action);
            train_error += predicted.mean_absolute_delta(after);
            predictor.observe_transition(before, action, after);
            train_steps = train_steps.saturating_add(1);
        }

        let (held_out_steps, held_out_baseline_mae, held_out_predictor_mae,
            held_out_improvement, held_out_calibration, held_out_survived) =
            evaluate_frozen_scenario(&predictor, held_out, held_out_cycles);

        let (held_out_continuous_rollout_steps, held_out_continuous_rollout_mae) =
            evaluate_frozen_continuous_rollout(
                predictor.bridge,
                held_out,
                held_out_cycles,
                0.5,
                0.1,
                200,
            );

        let held_out_multi_horizon =
            evaluate_multi_horizon(predictor.bridge, held_out, held_out_cycles);

        let policy_induced_shift = evaluate_policy_induced_shift(
            &predictor,
            held_out,
            held_out_cycles,
            policy_horizon,
            policy_discount,
            held_out_predictor_mae,
        );

        let closed_loop = run_homeostatic_agent_horizon_scenario(
            &mut predictor,
            held_out,
            held_out_cycles,
            policy_horizon,
            policy_discount,
        );

        let (recovery_steps, recovery_rate, mean_recovery_steps) =
            measure_recovery(held_out, &closed_loop.actions, held_out_cycles);

        GroundedWorldModelQualificationReport {
            training_scenario: training.name,
            held_out_scenario: held_out.name,
            train_steps,
            train_predictor_mae: train_error / train_steps.max(1) as f64,
            held_out_steps,
            held_out_baseline_mae,
            held_out_predictor_mae,
            held_out_improvement_over_baseline: held_out_improvement,
            held_out_confidence_calibration: held_out_calibration,
            held_out_continuous_rollout_steps,
            held_out_continuous_rollout_mae,
            held_out_multi_horizon,
            held_out_survived_fixed_schedule: held_out_survived,
            policy_induced_shift,
            persistence_closed_loop_survived: persistence_closed_loop.survived,
            persistence_closed_loop_mean_oracle_horizon_regret:
                persistence_closed_loop.mean_oracle_horizon_regret,
            persistence_recovery_rate,
            persistence_mean_recovery_steps,
            closed_loop_survived: closed_loop.survived,
            closed_loop_steps: closed_loop.steps,
            closed_loop_prediction_mae: if closed_loop.steps == 0 {
                0.0
            } else {
                closed_loop.cumulative_prediction_error / closed_loop.steps as f64
            },
            closed_loop_mean_oracle_horizon_regret: closed_loop.mean_oracle_horizon_regret,
            closed_loop_min_actual_viability_margin: closed_loop.min_actual_viability_margin,
            closed_loop_execution_failures: closed_loop.execution_failures,
            closed_loop_terminated_on_execution_failure: closed_loop.terminated_on_execution_failure,
            perturbations_applied: closed_loop.perturbations_applied,
            perturbation_recovery_steps: recovery_steps,
            recovery_rate,
            mean_recovery_steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default)]
    struct OraclePredictor;

    impl MicroWorldPredictor for OraclePredictor {
        fn predict(
            &self,
            state: MicroWorldObservation,
            action: MicroAction,
        ) -> MicroWorldObservation {
            super::super::viability_micro_world::transition(state, action)
        }

        fn prediction_confidence(&self, _action: MicroAction) -> f64 {
            1.0
        }
    }

    #[test]
    fn shared_transition_adapter_rejects_invalid_action() {
        let model = super::goal_world::WorldModelBridge::with_actions(2);
        assert!(ActionConditionedTransitionOde::new(&model, 2, 0.1).is_none());
        assert!(roll_transition_model_trajectory(&model, &[0.0; 64], 2, 0.1, 0.1, 32).is_none());
    }

    #[test]
    fn continuous_rollout_matches_relaxation_solution_at_tau() {
        let mut model = super::goal_world::WorldModelBridge::with_actions(2);
        let before = vec![0.0f32; 64];
        let mut after = before.clone();
        after[0] = 0.25;
        after[1] = -0.10;

        for _ in 0..20 {
            model
                .observe_action_transition(0, &before, &after)
                .expect("valid action/state dimensions");
        }

        let state = vec![0.0f64; 64];
        let rollout = roll_transition_model_trajectory(
            &model,
            &state,
            0,
            0.1,
            0.1,
            64,
        )
        .expect("valid continuous rollout");

        let relaxation_fraction = 1.0 - (-1.0f64).exp();
        for index in [0usize, 1usize] {
            let predicted = rollout.one_step_prediction[index];
            let expected = predicted * relaxation_fraction;
            assert!((rollout.terminal_state[index] - expected).abs() < 2e-3);
        }
    }

    #[test]
    fn horizon_confidence_decays_with_depth() {
        let predictor = OraclePredictor;
        let policy = HomeostaticPolicy;
        let current = benchmark_scenarios()[0].initial;
        let (_, _, rollout) = policy.choose_horizon(&predictor, current, 4, 0.8);

        assert!(rollout.min_confidence < 1.0);
        let expected = 0.85_f64.powi(3);
        assert!((rollout.min_confidence - expected).abs() < 1e-12);
    }

    #[test]
    fn multi_horizon_profile_has_all_requested_horizons() {
        let mut model = super::goal_world::WorldModelBridge::with_actions(6);
        let before = vec![0.0f32; 64];
        let mut after = before.clone();
        after[0] = 0.20;
        after[1] = -0.05;

        for action in 0..6 {
            for _ in 0..24 {
                model
                    .observe_action_transition(action, &before, &after)
                    .expect("valid action/state dimensions");
            }
        }

        let points = evaluate_multi_horizon(
            &model,
            &benchmark_scenarios()[0],
            8,
        );

        assert_eq!(points.len(), QUALIFICATION_HORIZONS.len());
        assert_eq!(
            points.iter().map(|point| point.horizon_steps).collect::<Vec<_>>(),
            QUALIFICATION_HORIZONS.to_vec()
        );
        assert!(points.iter().all(|point| point.samples > 0));
    }

    #[test]
    fn frozen_policy_induced_shift_has_no_learning_side_effects() {
        let predictor = OraclePredictor;
        let scenario = benchmark_scenarios()[0];
        let report = evaluate_policy_induced_shift(
            &predictor,
            &scenario,
            8,
            4,
            0.8,
            0.05,
        );

        assert!(report.steps > 0);
        assert_eq!(report.execution_failures, 0);
        assert!(!report.terminated_on_execution_failure);
        assert!(report.predictor_mae < 1e-12);
        assert!(report.survival);
    }

    #[test]
    fn deliberately_overconfident_bad_predictions_are_miscalibrated() {
        #[derive(Debug, Default)]
        struct OverconfidentPersistence;

        impl MicroWorldPredictor for OverconfidentPersistence {
            fn predict(
                &self,
                state: MicroWorldObservation,
                action: MicroAction,
            ) -> MicroWorldObservation {
                PersistencePredictor::default().predict(state, action)
            }

            fn prediction_confidence(&self, _action: MicroAction) -> f64 {
                1.0
            }
        }

        let scenarios = benchmark_scenarios();
        let (_, _, _, _, calibration, _) =
            evaluate_frozen_scenario(&OverconfidentPersistence, &scenarios[0], 8);

        assert!(calibration.sample_count > 0);
        assert!(calibration.expected_calibration_error > 0.0);
        assert!(calibration.confidence_accuracy_mse > 0.0);
    }

    #[test]
    fn perfect_oracle_is_perfectly_calibrated() {
        let scenarios = benchmark_scenarios();
        let (_, baseline, predictor, improvement, calibration, survived) =
            evaluate_frozen_scenario(&OraclePredictor, &scenarios[0], 8);

        assert!(predictor.abs() < f64::EPSILON);
        assert!(improvement > 0.0);
        assert!(calibration.sample_count > 0);
        assert!(calibration.expected_calibration_error < f64::EPSILON);
        assert!(calibration.confidence_accuracy_mse < f64::EPSILON);
        assert!(survived);
        assert!(baseline > 0.0);
    }
}