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
    MicroWorldObservation, MicroWorldPredictor, MicroWorldScenario, PersistencePredictor,
};

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

        let get = |index: usize| predicted.get(index).copied().unwrap_or(0.0) as f64;
        MicroWorldObservation {
            cycle: state.cycle.saturating_add(1),
            energy: get(0),
            integrity: get(1),
            knowledge: get(2),
            threat: get(3),
            progress: get(4),
        }
        .clamp_for_qualification()
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

/// Local clamp helper for the qualification adapter.
///
/// The benchmark's public observation clamp is intentionally private to its module;
/// the qualification boundary therefore repeats only the fixed [0,1] contract.
trait QualificationClamp {
    fn clamp_for_qualification(self) -> Self;
}

impl QualificationClamp for MicroWorldObservation {
    fn clamp_for_qualification(mut self) -> Self {
        self.energy = self.energy.clamp(0.0, 1.0);
        self.integrity = self.integrity.clamp(0.0, 1.0);
        self.knowledge = self.knowledge.clamp(0.0, 1.0);
        self.threat = self.threat.clamp(0.0, 1.0);
        self.progress = self.progress.clamp(0.0, 1.0);
        self
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
    pub held_out_survived_fixed_schedule: bool,

    pub closed_loop_survived: bool,
    pub closed_loop_steps: u64,
    pub closed_loop_prediction_mae: f64,
    pub closed_loop_mean_oracle_horizon_regret: f64,
    pub closed_loop_min_actual_viability_margin: f64,
    pub perturbations_applied: usize,

    /// Time-to-recovery is measured as cycles required to regain the exact
    /// pre-perturbation viability margin. None means no recovery occurred.
    pub perturbation_recovery_steps: Vec<Option<u64>>,
    pub recovery_rate: f64,
    pub mean_recovery_steps: f64,
}

impl GroundedWorldModelQualificationReport {
    pub fn transfer_passes(&self) -> bool {
        self.held_out_predictor_mae < self.held_out_baseline_mae
    }

    pub fn recovery_passes(&self) -> bool {
        self.recovery_rate >= 1.0
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

        self.world_model.reset();

        let mut predictor = FepWorldModelPredictor {
            bridge: &mut self.world_model,
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
            held_out_survived_fixed_schedule: held_out_survived,
            closed_loop_survived: closed_loop.survived,
            closed_loop_steps: closed_loop.steps,
            closed_loop_prediction_mae: if closed_loop.steps == 0 {
                0.0
            } else {
                closed_loop.cumulative_prediction_error / closed_loop.steps as f64
            },
            closed_loop_mean_oracle_horizon_regret: closed_loop.mean_oracle_horizon_regret,
            closed_loop_min_actual_viability_margin: closed_loop.min_actual_viability_margin,
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
