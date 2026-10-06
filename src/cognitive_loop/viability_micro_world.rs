// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic micro-world for qualifying organism-like closed-loop behavior.
//!
//! This is deliberately a small benchmark environment, not a claim that Symthaea is
//! already embodied. It supplies:
//! - observable state;
//! - deterministic action consequences;
//! - a ground-truth transition oracle;
//! - a weak persistence baseline;
//! - an episode evaluator that can later accept real world/self-model predictors.
//!
//! The benchmark is useful because it makes "prediction -> action -> consequence" a
//! falsifiable interface before any physical robot or external model is involved.

use super::goal_world::WorldModelBridge;
use super::viability_fabric::{
    ActionOutcome, ActionPrediction, PredictionErrorLedger, ViabilityDelta, ViabilityFabric,
};
use serde::{Deserialize, Serialize};

/// Small action space designed to exercise trade-offs between energy, integrity,
/// knowledge, threat, and progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MicroAction {
    Observe,
    Explore,
    Harvest,
    Repair,
    Rest,
    Retreat,
}

impl MicroAction {
    pub const ALL: [Self; 6] = [
        Self::Observe,
        Self::Explore,
        Self::Harvest,
        Self::Repair,
        Self::Rest,
        Self::Retreat,
    ];

    pub fn index(self) -> usize {
        match self {
            Self::Observe => 0,
            Self::Explore => 1,
            Self::Harvest => 2,
            Self::Repair => 3,
            Self::Rest => 4,
            Self::Retreat => 5,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Observe => "observe",
            Self::Explore => "explore",
            Self::Harvest => "harvest",
            Self::Repair => "repair",
            Self::Rest => "rest",
            Self::Retreat => "retreat",
        }
    }
}

/// The externally observable state of the benchmark organism/environment.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MicroWorldObservation {
    pub cycle: u64,
    pub energy: f64,
    pub integrity: f64,
    pub knowledge: f64,
    pub threat: f64,
    pub progress: f64,
}

impl MicroWorldObservation {
    fn clamp(self) -> Self {
        Self {
            cycle: self.cycle,
            energy: self.energy.clamp(0.0, 1.0),
            integrity: self.integrity.clamp(0.0, 1.0),
            knowledge: self.knowledge.clamp(0.0, 1.0),
            threat: self.threat.clamp(0.0, 1.0),
            progress: self.progress.clamp(0.0, 1.0),
        }
    }

    pub fn digest(self) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for value in [
            self.cycle as f64,
            self.energy,
            self.integrity,
            self.knowledge,
            self.threat,
            self.progress,
        ] {
            for byte in value.to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    pub fn is_viable(self) -> bool {
        self.energy > 0.08 && self.integrity > 0.08
    }

    pub fn mean_absolute_delta(self, other: Self) -> f64 {
        let values_a = [self.energy, self.integrity, self.knowledge, self.threat, self.progress];
        let values_b = [other.energy, other.integrity, other.knowledge, other.threat, other.progress];
        values_a
            .iter()
            .zip(values_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / values_a.len() as f64
    }
}

/// Deterministic environment with no randomness and no external I/O.
#[derive(Debug, Clone)]
pub struct MicroWorld {
    observation: MicroWorldObservation,
    initial: MicroWorldObservation,
    max_cycles: u64,
}

impl Default for MicroWorld {
    fn default() -> Self {
        Self::new(
            MicroWorldObservation {
                cycle: 0,
                energy: 0.65,
                integrity: 0.80,
                knowledge: 0.15,
                threat: 0.20,
                progress: 0.0,
            },
            64,
        )
    }
}

impl MicroWorld {
    pub fn new(initial: MicroWorldObservation, max_cycles: u64) -> Self {
        Self {
            observation: initial.clamp(),
            initial: initial.clamp(),
            max_cycles,
        }
    }

    pub fn reset(&mut self) {
        self.observation = self.initial;
    }

    pub fn observe(&self) -> MicroWorldObservation {
        self.observation
    }

    pub fn done(&self) -> bool {
        self.observation.cycle >= self.max_cycles || !self.observation.is_viable()
    }

    pub fn step(&mut self, action: MicroAction) -> MicroWorldObservation {
        self.observation = transition(self.observation, action);
        self.observation
    }
}

/// Ground-truth deterministic transition function.
///
/// The action effects are intentionally simple enough that a learned predictor can
/// discover them, while the threat-dependent explore/harvest effects prevent a naive
/// constant-delta model from being perfect.
pub fn transition(state: MicroWorldObservation, action: MicroAction) -> MicroWorldObservation {
    let mut next = state;
    next.cycle = state.cycle.saturating_add(1);

    match action {
        MicroAction::Observe => {
            next.energy -= 0.025;
            next.knowledge += 0.045 * (1.0 - state.knowledge);
            next.threat -= 0.012;
        }
        MicroAction::Explore => {
            let pulse = (((state.cycle.wrapping_mul(17) + 5) % 11) as f64) / 10.0;
            next.energy -= 0.105 + 0.025 * pulse;
            next.knowledge += 0.13 * (1.0 - state.knowledge);
            next.threat += 0.05 + 0.10 * pulse * (1.0 - state.threat);
            next.integrity -= 0.02 + 0.04 * pulse * state.threat;
            next.progress += 0.075 * (1.0 - state.progress);
        }
        MicroAction::Harvest => {
            let efficiency = 0.55 + 0.45 * (1.0 - state.threat);
            next.energy += 0.17 * efficiency;
            next.threat += 0.02 * state.threat;
            next.integrity -= 0.015 + 0.025 * state.threat;
            next.progress += 0.03 * efficiency * (1.0 - state.progress);
        }
        MicroAction::Repair => {
            next.energy -= 0.09;
            next.integrity += 0.20 * (1.0 - state.integrity);
            next.threat -= 0.015;
        }
        MicroAction::Rest => {
            next.energy += 0.16 * (1.0 - state.energy);
            next.threat -= 0.08 * state.threat;
            next.integrity += 0.035 * (1.0 - state.integrity);
        }
        MicroAction::Retreat => {
            next.energy -= 0.045;
            next.threat -= 0.20 * state.threat;
            next.progress -= 0.015 * state.progress;
            next.integrity += 0.015 * (1.0 - state.integrity);
        }
    }

    next.clamp()
}

/// A deterministic benchmark scenario with a distinct initial state and action schedule.
#[derive(Debug, Clone)]
pub struct MicroWorldScenario {
    pub name: &'static str,
    pub initial: MicroWorldObservation,
    pub schedule: &'static [MicroAction],
}

static NOMINAL_SCHEDULE: [MicroAction; 6] = [
    MicroAction::Observe,
    MicroAction::Explore,
    MicroAction::Harvest,
    MicroAction::Repair,
    MicroAction::Rest,
    MicroAction::Retreat,
];

static STRESSED_SCHEDULE: [MicroAction; 6] = [
    MicroAction::Harvest,
    MicroAction::Rest,
    MicroAction::Repair,
    MicroAction::Retreat,
    MicroAction::Observe,
    MicroAction::Harvest,
];

static DAMAGED_SCHEDULE: [MicroAction; 6] = [
    MicroAction::Repair,
    MicroAction::Rest,
    MicroAction::Harvest,
    MicroAction::Explore,
    MicroAction::Observe,
    MicroAction::Retreat,
];

static KNOWLEDGE_RICH_SCHEDULE: [MicroAction; 6] = [
    MicroAction::Explore,
    MicroAction::Explore,
    MicroAction::Harvest,
    MicroAction::Rest,
    MicroAction::Observe,
    MicroAction::Retreat,
];

pub fn benchmark_scenarios() -> Vec<MicroWorldScenario> {
    vec![
        MicroWorldScenario {
            name: "nominal",
            initial: MicroWorld::default().observe(),
            schedule: &NOMINAL_SCHEDULE,
        },
        MicroWorldScenario {
            name: "stressed",
            initial: MicroWorldObservation {
                cycle: 0,
                energy: 0.30,
                integrity: 0.35,
                knowledge: 0.25,
                threat: 0.65,
                progress: 0.05,
            },
            schedule: &STRESSED_SCHEDULE,
        },
        MicroWorldScenario {
            name: "damaged",
            initial: MicroWorldObservation {
                cycle: 0,
                energy: 0.50,
                integrity: 0.20,
                knowledge: 0.40,
                threat: 0.35,
                progress: 0.10,
            },
            schedule: &DAMAGED_SCHEDULE,
        },
        MicroWorldScenario {
            name: "knowledge_rich",
            initial: MicroWorldObservation {
                cycle: 0,
                energy: 0.80,
                integrity: 0.90,
                knowledge: 0.80,
                threat: 0.10,
                progress: 0.40,
            },
            schedule: &KNOWLEDGE_RICH_SCHEDULE,
        },
    ]
}

/// Predictor interface for the benchmark harness.
///
/// A production predictor can later be backed by WorldModelBridge, a learned latent
/// model, or a specialist decision/world model. The environment never trusts the
/// predictor; it only scores it.
pub trait MicroWorldPredictor {
    fn predict(&mut self, state: MicroWorldObservation, action: MicroAction)
        -> MicroWorldObservation;

    /// Confidence in the forecast. Zero means the forecast should not materially
    /// override the currently observed state.
    fn prediction_confidence(&self, _action: MicroAction) -> f64 {
        0.0
    }

    /// Optional online learning hook. The default is a fixed predictor.
    fn observe_transition(
        &mut self,
        _before: MicroWorldObservation,
        _action: MicroAction,
        _after: MicroWorldObservation,
    ) {
    }
}

#[derive(Debug)]
pub struct WorldModelBridgePredictor {
    bridge: WorldModelBridge,
}

impl Default for WorldModelBridgePredictor {
    fn default() -> Self {
        Self {
            bridge: WorldModelBridge::with_actions(MicroAction::ALL.len()),
        }
    }
}

impl WorldModelBridgePredictor {
    pub fn model(&self) -> &WorldModelBridge {
        &self.bridge
    }
}

impl MicroWorldPredictor for WorldModelBridgePredictor {
    fn predict(
        &mut self,
        state: MicroWorldObservation,
        action: MicroAction,
    ) -> MicroWorldObservation {
        let encoded = encode_observation(state);
        let Some(predicted) = self.bridge.predict_action(action.index(), &encoded) else {
            return state;
        };
        decode_observation(state.cycle.saturating_add(1), &predicted)
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
        let before_encoded = encode_observation(before);
        let after_encoded = encode_observation(after);
        let _ = self
            .bridge
            .observe_action_transition(action.index(), &before_encoded, &after_encoded);
    }
}

fn encode_observation(state: MicroWorldObservation) -> Vec<f32> {
    let mut encoded = vec![0.0f32; 64];
    encoded[0] = state.energy as f32;
    encoded[1] = state.integrity as f32;
    encoded[2] = state.knowledge as f32;
    encoded[3] = state.threat as f32;
    encoded[4] = state.progress as f32;
    encoded
}

fn decode_observation(cycle: u64, encoded: &[f32]) -> MicroWorldObservation {
    let get = |index: usize| encoded.get(index).copied().unwrap_or(0.0) as f64;
    MicroWorldObservation {
        cycle,
        energy: get(0),
        integrity: get(1),
        knowledge: get(2),
        threat: get(3),
        progress: get(4),
    }
    .clamp()
}

/// Weak baseline: assumes the world remains unchanged after every action.
#[derive(Debug, Default)]
pub struct PersistencePredictor;

impl MicroWorldPredictor for PersistencePredictor {
    fn predict(
        &mut self,
        state: MicroWorldObservation,
        _action: MicroAction,
    ) -> MicroWorldObservation {
        MicroWorldObservation {
            cycle: state.cycle.saturating_add(1),
            ..state
        }
    }
}

/// Minimal homeostatic policy used to qualify whether a predictor can support
/// survival-aware action selection.
///
/// This policy is intentionally simple and inspectable. The research variable is the
/// predictor: replace it with Symthaea's world-model path and measure what changes.
#[derive(Debug, Default)]
pub struct HomeostaticPolicy;

impl HomeostaticPolicy {
    fn score(predicted: MicroWorldObservation, current: MicroWorldObservation, action: MicroAction) -> f64 {
        let viability_pressure = (0.30 - predicted.energy).max(0.0)
            + (0.30 - predicted.integrity).max(0.0)
            + (predicted.threat - 0.60).max(0.0);

        let mut score = predicted.progress * 1.50
            + predicted.knowledge * 0.35
            + predicted.energy * 0.50
            + predicted.integrity * 0.70
            - predicted.threat * 1.20
            - viability_pressure * 2.0;

        // Hysteretic-looking policy biases make the survival objective explicit without
        // allowing the action to bypass the predictor.
        if current.energy < 0.25 {
            if action == MicroAction::Rest {
                score += 0.80;
            }
            if action == MicroAction::Harvest {
                score += 0.50;
            }
        }
        if current.integrity < 0.35 && action == MicroAction::Repair {
            score += 1.10;
        }
        if current.threat > 0.65 {
            if action == MicroAction::Retreat {
                score += 1.10;
            }
            if action == MicroAction::Observe {
                score += 0.20;
            }
        }

        score
    }

    pub fn choose<P: MicroWorldPredictor>(
        &self,
        predictor: &mut P,
        current: MicroWorldObservation,
    ) -> (MicroAction, MicroWorldObservation) {
        let mut best_action = MicroAction::Observe;
        let mut best_prediction = current;
        let mut best_score = f64::NEG_INFINITY;

        for action in MicroAction::ALL {
            let predicted_raw = predictor.predict(current, action);
            let confidence = predictor.prediction_confidence(action).clamp(0.0, 1.0);
            // Low-confidence predictions are blended toward the observed state rather than
            // being allowed to dictate policy as if they were established.
            let predicted = MicroWorldObservation {
                cycle: predicted_raw.cycle,
                energy: current.energy + confidence * (predicted_raw.energy - current.energy),
                integrity: current.integrity
                    + confidence * (predicted_raw.integrity - current.integrity),
                knowledge: current.knowledge
                    + confidence * (predicted_raw.knowledge - current.knowledge),
                threat: current.threat
                    + confidence * (predicted_raw.threat - current.threat),
                progress: current.progress
                    + confidence * (predicted_raw.progress - current.progress),
            };
            let score = Self::score(predicted, current, action);
            if score > best_score {
                best_score = score;
                best_action = action;
                best_prediction = predicted;
            }
        }

        (best_action, best_prediction)
    }
}

/// Report from a policy-driven closed-loop run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HomeostaticRunReport {
    pub steps: u64,
    pub survived: bool,
    pub final_energy: f64,
    pub final_integrity: f64,
    pub final_knowledge: f64,
    pub final_threat: f64,
    pub final_progress: f64,
    pub cumulative_prediction_error: f64,
    pub actions: Vec<MicroAction>,
}

/// Execute perception -> prediction -> selection -> action -> observation for a
/// deterministic environment.
///
/// Every selected action is recorded in ViabilityFabric only after a prediction for
/// that action was already inserted, preserving the no-post-hoc-prediction invariant.
pub fn run_homeostatic_agent<P: MicroWorldPredictor>(
    predictor: &mut P,
    max_cycles: u64,
) -> HomeostaticRunReport {
    let mut world = MicroWorld::default();
    let policy = HomeostaticPolicy;
    let mut fabric = ViabilityFabric::new(max_cycles as usize + 1);
    let mut cumulative_error = 0.0;
    let mut actions = Vec::with_capacity(max_cycles as usize);
    let mut steps = 0u64;

    while !world.done() && steps < max_cycles {
        let before = world.observe();
        let (action, predicted) = policy.choose(predictor, before);
        let action_id = steps + 1;

        fabric.begin_cycle(before.cycle);
        fabric
            .predict_action(ActionPrediction {
                action_id,
                action_label: action.label().to_string(),
                cycle: before.cycle,
                predicted_world_delta: Some(signed_delta_with_confidence(
                    before,
                    predicted,
                    predictor.prediction_confidence(action),
                )),
                predicted_self_delta: None,
                predicted_goal_delta: None,
                authority_granted: true,
            })
            .expect("policy action id must be unique");

        let after = world.step(action);
        let error = predicted.mean_absolute_delta(after);
        predictor.observe_transition(before, action, after);
        cumulative_error += error;
        steps += 1;
        actions.push(action);

        fabric
            .observe_action(ActionOutcome {
                action_id,
                action_label: action.label().to_string(),
                cycle: after.cycle,
                pre_state_digest: before.digest(),
                post_state_digest: after.digest(),
                authority_granted: true,
                safety_gate_passed: true,
                prediction: None,
                observed_effect: Some(
                    super::viability_fabric::ViabilitySignal::new(
                        (signed_delta(before, after).value + 1.0) * 0.5,
                        1.0,
                        after.cycle,
                        "viability-micro-world",
                    ),
                ),
                prediction_error: PredictionErrorLedger {
                    world: error.clamp(0.0, 1.0),
                    ..Default::default()
                },
                evidence_refs: vec![format!(
                    "sim://viability-micro-world/policy-step/{}",
                    after.digest()
                )],
            })
            .expect("closed-loop action must have a pre-action prediction");
    }

    let final_state = world.observe();
    HomeostaticRunReport {
        steps,
        survived: final_state.is_viable(),
        final_energy: final_state.energy,
        final_integrity: final_state.integrity,
        final_knowledge: final_state.knowledge,
        final_threat: final_state.threat,
        final_progress: final_state.progress,
        cumulative_prediction_error: cumulative_error,
        actions,
    }
}

/// Report from a deterministic benchmark episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicroWorldReport {
    pub steps: u64,
    pub baseline_mae: f64,
    pub predictor_mae: f64,
    pub survival_ratio: f64,
    pub final_energy: f64,
    pub final_integrity: f64,
    pub final_progress: f64,
    pub ledger_outcomes: usize,
}

impl MicroWorldReport {
    pub fn improvement_over_baseline(&self) -> f64 {
        if self.baseline_mae <= f64::EPSILON {
            0.0
        } else {
            (self.baseline_mae - self.predictor_mae) / self.baseline_mae
        }
    }
}

/// Run a fixed action schedule and score predictions against ground truth.
///
/// The same schedule is replayed for every predictor, which makes regressions
/// comparable across implementations and machines.
pub fn evaluate_predictor<P: MicroWorldPredictor>(
    predictor: &mut P,
    max_cycles: u64,
) -> MicroWorldReport {
    evaluate_predictor_scenario(
        predictor,
        &MicroWorldScenario {
            name: "nominal",
            initial: MicroWorld::default().observe(),
            schedule: &NOMINAL_SCHEDULE,
        },
        max_cycles,
    )
}

fn evaluate_predictor_scenario<P: MicroWorldPredictor>(
    predictor: &mut P,
    scenario: &MicroWorldScenario,
    max_cycles: u64,
) -> MicroWorldReport {
    let mut world = MicroWorld::new(scenario.initial, max_cycles);
    let mut fabric = ViabilityFabric::new(max_cycles as usize + 1);
    let mut baseline_error = 0.0;
    let mut predictor_error = 0.0;
    let mut steps = 0u64;

    while !world.done() && steps < max_cycles {
        let before = world.observe();
        let action = scenario.schedule[steps as usize % scenario.schedule.len()];

        let predicted = predictor.predict(before, action);
        let predicted_world_delta = signed_delta_with_confidence(
            before,
            predicted,
            predictor.prediction_confidence(action),
        );
        let action_id = steps + 1;

        fabric.begin_cycle(before.cycle);
        fabric
            .predict_action(ActionPrediction {
                action_id,
                action_label: action.label().to_string(),
                cycle: before.cycle,
                predicted_world_delta: Some(predicted_world_delta),
                predicted_self_delta: None,
                predicted_goal_delta: None,
                authority_granted: true,
            })
            .expect("deterministic benchmark must insert unique prediction");

        let after = world.step(action);
        let actual_delta = signed_delta(before, after);
        let mae = predicted.mean_absolute_delta(after);
        predictor.observe_transition(before, action, after);
        let baseline = PersistencePredictor::default()
            .predict(before, action)
            .mean_absolute_delta(after);

        predictor_error += mae;
        baseline_error += baseline;
        steps += 1;

        fabric
            .observe_action(ActionOutcome {
                action_id,
                action_label: action.label().to_string(),
                cycle: after.cycle,
                pre_state_digest: before.digest(),
                post_state_digest: after.digest(),
                authority_granted: true,
                safety_gate_passed: true,
                prediction: None,
                observed_effect: Some(
                    super::viability_fabric::ViabilitySignal::new(
                        (actual_delta.value + 1.0) * 0.5,
                        actual_delta.confidence,
                        after.cycle,
                        "viability-micro-world",
                    ),
                ),
                prediction_error: PredictionErrorLedger {
                    world: mae.clamp(0.0, 1.0),
                    ..Default::default()
                },
                evidence_refs: vec![format!(
                    "sim://viability-micro-world/episode/{}/step/{}",
                    before.digest(),
                    steps
                )],
            })
            .expect("benchmark outcome must close pre-existing prediction");
    }

    let final_state = world.observe();
    let denom = steps.max(1) as f64;

    MicroWorldReport {
        steps,
        baseline_mae: baseline_error / denom,
        predictor_mae: predictor_error / denom,
        survival_ratio: if final_state.is_viable() { 1.0 } else { 0.0 },
        final_energy: final_state.energy,
        final_integrity: final_state.integrity,
        final_progress: final_state.progress,
        ledger_outcomes: fabric.outcomes().len(),
    }
}

/// Aggregate report across several deterministic worlds.
///
/// The suite prevents a future predictor from qualifying by memorizing one initial
/// trajectory. Every predictor sees the same scenarios and schedules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicroWorldSuiteReport {
    pub episodes: usize,
    pub total_steps: u64,
    pub mean_baseline_mae: f64,
    pub mean_predictor_mae: f64,
    pub mean_improvement: f64,
    pub survival_rate: f64,
}

impl MicroWorldSuiteReport {
    pub fn suite_improvement(&self) -> f64 {
        if self.mean_baseline_mae <= f64::EPSILON {
            0.0
        } else {
            (self.mean_baseline_mae - self.mean_predictor_mae) / self.mean_baseline_mae
        }
    }
}

pub fn evaluate_predictor_suite<P: MicroWorldPredictor>(
    predictor: &mut P,
    max_cycles: u64,
) -> MicroWorldSuiteReport {
    let scenarios = benchmark_scenarios();
    let mut reports = Vec::with_capacity(scenarios.len());

    for scenario in &scenarios {
        reports.push(evaluate_predictor_scenario(predictor, scenario, max_cycles));
    }

    let episodes = reports.len();
    let denom = episodes.max(1) as f64;
    MicroWorldSuiteReport {
        episodes,
        total_steps: reports.iter().map(|r| r.steps).sum(),
        mean_baseline_mae: reports.iter().map(|r| r.baseline_mae).sum::<f64>() / denom,
        mean_predictor_mae: reports.iter().map(|r| r.predictor_mae).sum::<f64>() / denom,
        mean_improvement: reports
            .iter()
            .map(MicroWorldReport::improvement_over_baseline)
            .sum::<f64>()
            / denom,
        survival_rate: reports.iter().map(|r| r.survival_ratio).sum::<f64>() / denom,
    }
}

fn signed_delta(before: MicroWorldObservation, after: MicroWorldObservation) -> ViabilityDelta {
    signed_delta_with_confidence(before, after, 1.0)
}

fn signed_delta_with_confidence(
    before: MicroWorldObservation,
    after: MicroWorldObservation,
    confidence: f64,
) -> ViabilityDelta {
    ViabilityDelta::new(
        after.energy - before.energy
            + (after.integrity - before.integrity)
            + (after.knowledge - before.knowledge)
            + (after.threat - before.threat)
            + (after.progress - before.progress),
        confidence.clamp(0.0, 1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_is_deterministic() {
        let state = MicroWorld::default().observe();
        assert_eq!(
            transition(state, MicroAction::Explore),
            transition(state, MicroAction::Explore)
        );
    }

    #[test]
    fn negative_effects_survive_world_transition() {
        let state = MicroWorld::default().observe();
        let next = transition(state, MicroAction::Explore);
        assert!(next.energy < state.energy);
        assert!(next.integrity <= state.integrity);
    }

    #[test]
    fn bridge_predictor_learns_from_repeated_transitions() {
        let mut predictor = WorldModelBridgePredictor::default();
        let before = MicroWorld::default().observe();
        let after = transition(before, MicroAction::Explore);

        let first = predictor.predict(before, MicroAction::Explore)
            .mean_absolute_delta(after);

        for _ in 0..32 {
            predictor.observe_transition(before, MicroAction::Explore, after);
        }

        let learned = predictor.predict(before, MicroAction::Explore)
            .mean_absolute_delta(after);

        assert!(learned < first);
        assert!(predictor.model().action_samples(MicroAction::Explore.index()) > 0);
    }

    #[test]
    fn persistence_baseline_is_nonzero() {
        let mut predictor = PersistencePredictor;
        let report = evaluate_predictor(&mut predictor, 12);
        assert!(report.baseline_mae > 0.0);
        assert!(report.predictor_mae > 0.0);
        assert_eq!(report.ledger_outcomes, report.steps as usize);
    }

    #[test]
    fn oracle_can_be_zero_error() {
        struct Oracle;
        impl MicroWorldPredictor for Oracle {
            fn predict(
                &mut self,
                state: MicroWorldObservation,
                action: MicroAction,
            ) -> MicroWorldObservation {
                transition(state, action)
            }

            fn prediction_confidence(&self, _action: MicroAction) -> f64 {
                1.0
            }
        }

        let mut predictor = Oracle;
        let report = evaluate_predictor(&mut predictor, 12);
        assert!(report.predictor_mae.abs() < 1e-12);
        assert!(report.improvement_over_baseline() > 0.99);
    }

    #[test]
    fn oracle_policy_survives_and_makes_progress() {
        struct Oracle;
        impl MicroWorldPredictor for Oracle {
            fn predict(&mut self, state: MicroWorldObservation, action: MicroAction) -> MicroWorldObservation {
                transition(state, action)
            }

            fn prediction_confidence(&self, _action: MicroAction) -> f64 {
                1.0
            }
        }

        let mut predictor = Oracle;
        let report = run_homeostatic_agent(&mut predictor, 64);
        assert!(report.survived);
        assert!(report.final_progress > 0.2);
        assert_eq!(report.actions.len(), report.steps as usize);
    }

    #[test]
    fn homeostatic_run_is_replay_stable() {
        let mut a = PersistencePredictor;
        let mut b = PersistencePredictor;
        assert_eq!(run_homeostatic_agent(&mut a, 32), run_homeostatic_agent(&mut b, 32));
    }

    #[test]
    fn suite_has_multiple_distinct_scenarios() {
        let scenarios = benchmark_scenarios();
        assert!(scenarios.len() >= 4);
        assert!(scenarios.windows(2).any(|w| w[0].initial != w[1].initial));
    }

    #[test]
    fn oracle_suite_is_zero_error() {
        struct Oracle;
        impl MicroWorldPredictor for Oracle {
            fn predict(&mut self, state: MicroWorldObservation, action: MicroAction) -> MicroWorldObservation {
                transition(state, action)
            }

            fn prediction_confidence(&self, _action: MicroAction) -> f64 {
                1.0
            }
        }

        let mut predictor = Oracle;
        let report = evaluate_predictor_suite(&mut predictor, 24);
        assert_eq!(report.episodes, 4);
        assert!(report.mean_predictor_mae.abs() < 1e-12);
        assert!((report.survival_rate - 1.0).abs() < 1e-12);
    }

    #[test]
    fn report_is_replay_stable() {
        let mut a = PersistencePredictor;
        let mut b = PersistencePredictor;
        assert_eq!(evaluate_predictor(&mut a, 20), evaluate_predictor(&mut b, 20));
    }
}
