// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Viability Fabric: typed causal state for organism-like closed-loop cognition.
//!
//! This module deliberately contains orchestration/data contracts rather than a second
//! cognitive architecture. It connects existing perception, interoception, prediction,
//! action, self-model, and homeostatic subsystems through explicit observations and
//! prediction errors.
//!
//! Scientific boundary: these types measure engineering properties such as persistence,
//! prediction, regulation, and recovery. They are not a consciousness or life detector.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Lifecycle phases that can change resource allocation and consolidation behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecyclePhase {
    Active,
    Recovery,
    Consolidation,
}

impl Default for LifecyclePhase {
    fn default() -> Self {
        Self::Active
    }
}

/// A bounded scalar with explicit provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViabilitySignal {
    pub value: f64,
    pub confidence: f64,
    pub cycle: u64,
    pub producer: String,
}

impl ViabilitySignal {
    pub fn new(value: f64, confidence: f64, cycle: u64, producer: impl Into<String>) -> Self {
        debug_assert!(value.is_finite(), "ViabilitySignal value must be finite");
        debug_assert!(confidence.is_finite(), "ViabilitySignal confidence must be finite");
        Self {
            value: value.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            cycle,
            producer: producer.into(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.value.is_finite()
            && self.confidence.is_finite()
            && self.confidence >= 0.0
            && self.confidence <= 1.0
            && !self.producer.is_empty()
    }
}

/// A signed consequence of an action. Unlike a viability signal, this is intentionally
/// not clamped to [0,1]: negative outcomes must remain observable.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViabilityDelta {
    pub value: f64,
    pub confidence: f64,
}

impl ViabilityDelta {
    pub fn new(value: f64, confidence: f64) -> Self {
        debug_assert!(value.is_finite(), "ViabilityDelta value must be finite");
        debug_assert!(confidence.is_finite(), "ViabilityDelta confidence must be finite");
        Self {
            value,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.value.is_finite()
            && self.confidence.is_finite()
            && self.confidence >= 0.0
            && self.confidence <= 1.0
    }
}

/// Preferred, tolerated, and critical operating bands for an internal variable.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViabilityBand {
    pub preferred: (f64, f64),
    pub tolerated: (f64, f64),
    pub critical: (f64, f64),
}

impl ViabilityBand {
    pub fn contains_preferred(&self, value: f64) -> bool {
        value >= self.preferred.0 && value <= self.preferred.1
    }

    pub fn contains_tolerated(&self, value: f64) -> bool {
        value >= self.tolerated.0 && value <= self.tolerated.1
    }

    pub fn is_critical(&self, value: f64) -> bool {
        value < self.critical.0 || value > self.critical.1
    }

    pub fn validate(&self) -> bool {
        self.critical.0.is_finite()
            && self.critical.1.is_finite()
            && self.tolerated.0.is_finite()
            && self.tolerated.1.is_finite()
            && self.preferred.0.is_finite()
            && self.preferred.1.is_finite()
            && (0.0..=1.0).contains(&self.critical.0)
            && (0.0..=1.0).contains(&self.critical.1)
            && (0.0..=1.0).contains(&self.tolerated.0)
            && (0.0..=1.0).contains(&self.tolerated.1)
            && (0.0..=1.0).contains(&self.preferred.0)
            && (0.0..=1.0).contains(&self.preferred.1)
            && self.critical.0 <= self.critical.1
            && self.critical.0 <= self.tolerated.0
            && self.tolerated.0 <= self.tolerated.1
            && self.tolerated.0 <= self.preferred.0
            && self.preferred.0 <= self.preferred.1
            && self.preferred.1 <= self.tolerated.1
            && self.tolerated.1 <= self.critical.1
    }
}

/// Named internal variable with an observed value, operating band, and prediction error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViabilityVariable {
    pub observation: ViabilitySignal,
    pub band: ViabilityBand,
    pub rate_of_change: f64,
    pub prediction: Option<ViabilitySignal>,
    pub prediction_error: Option<f64>,
}

impl ViabilityVariable {
    pub fn normalized_pressure(&self) -> f64 {
        let value = self.observation.value;
        if !value.is_finite() || !self.band.validate() {
            return 1.0;
        }
        if self.band.contains_preferred(value) {
            return 0.0;
        }

        let (lo, hi) = self.band.preferred;
        if value < lo {
            let scale = (lo - self.band.tolerated.0).max(f64::EPSILON);
            return ((lo - value) / scale).clamp(0.0, 1.0);
        }

        let scale = (self.band.tolerated.1 - hi).max(f64::EPSILON);
        ((value - hi) / scale).clamp(0.0, 1.0)
    }
}

/// Prediction channels are kept distinct so world/self/interoceptive failures cannot hide
/// inside one aggregate scalar.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PredictionErrorLedger {
    pub world: f64,
    pub self_model: f64,
    pub interoceptive: f64,
    pub goal: f64,
    pub model_confidence: f64,
    pub execution: f64,
}

impl Default for PredictionErrorLedger {
    fn default() -> Self {
        Self {
            world: 0.0,
            self_model: 0.0,
            interoceptive: 0.0,
            goal: 0.0,
            model_confidence: 0.0,
            execution: 0.0,
        }
    }
}

impl PredictionErrorLedger {
    pub fn bounded(self) -> Self {
        Self {
            world: self.world.clamp(0.0, 1.0),
            self_model: self.self_model.clamp(0.0, 1.0),
            interoceptive: self.interoceptive.clamp(0.0, 1.0),
            goal: self.goal.clamp(0.0, 1.0),
            model_confidence: self.model_confidence.clamp(0.0, 1.0),
            execution: self.execution.clamp(0.0, 1.0),
        }
    }
}

/// Pre-action prediction. The record is created before execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionPrediction {
    pub action_id: u64,
    pub action_label: String,
    pub cycle: u64,
    pub predicted_world_delta: Option<ViabilityDelta>,
    pub predicted_self_delta: Option<ViabilityDelta>,
    pub predicted_goal_delta: Option<ViabilityDelta>,
    pub authority_granted: bool,
}

/// Explicit disposition for a prediction that did not result in an observed action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionCancellation {
    pub action_id: u64,
    pub prediction_cycle: u64,
    pub cancellation_cycle: u64,
    pub reason: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

/// Post-action observation. A prediction is never synthesized after the fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionOutcome {
    pub action_id: u64,
    pub action_label: String,
    pub cycle: u64,
    pub pre_state_digest: u64,
    pub post_state_digest: u64,
    pub authority_granted: bool,
    pub safety_gate_passed: bool,
    pub prediction: Option<ActionPrediction>,
    pub observed_effect: Option<ViabilitySignal>,
    pub prediction_error: PredictionErrorLedger,
    /// Evidence/provenance references supporting the observed outcome.
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

/// State snapshot used by the viability fabric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViabilityState {
    pub cycle: u64,
    pub lifecycle: LifecyclePhase,
    pub identity_coherence: Option<ViabilitySignal>,
    pub epistemic_confidence: Option<ViabilitySignal>,
    pub world_integrity: Option<ViabilitySignal>,
    pub self_integrity: Option<ViabilitySignal>,
    pub resource_pressure: BTreeMap<String, ViabilityVariable>,
    pub prediction_errors: PredictionErrorLedger,
}

impl Default for ViabilityState {
    fn default() -> Self {
        Self {
            cycle: 0,
            lifecycle: LifecyclePhase::Active,
            identity_coherence: None,
            epistemic_confidence: None,
            world_integrity: None,
            self_integrity: None,
            resource_pressure: BTreeMap::new(),
            prediction_errors: PredictionErrorLedger::default(),
        }
    }
}

impl ViabilityState {
    pub fn aggregate_pressure(&self) -> f64 {
        if self.resource_pressure.is_empty() {
            0.0
        } else {
            self.resource_pressure
                .values()
                .map(ViabilityVariable::normalized_pressure)
                .sum::<f64>()
                / self.resource_pressure.len() as f64
        }
    }

    /// Derive a bounded cognitive resource decision from current viability pressure.
    pub fn regulation_decision(&self, thresholds: RegulationThresholds) -> RegulationDecision {
        let thresholds = if thresholds.validate() {
            thresholds
        } else {
            RegulationThresholds::default()
        };
        RegulationDecision::from_pressure(self.regulation_pressure(), thresholds)
    }

    /// Conservative pressure signal: prediction error or resource pressure can only increase
    /// regulation pressure. Missing optional signals are not treated as healthy evidence.
    pub fn regulation_pressure(&self) -> f64 {
        let resource = self.aggregate_pressure();
        let prediction = {
            let p = self.prediction_errors.bounded();
            [
                p.world,
                p.self_model,
                p.interoceptive,
                p.goal,
                p.model_confidence,
                p.execution,
            ]
            .into_iter()
            .fold(0.0_f64, f64::max)
        };
        resource.max(prediction)
    }
}

/// Coarse cognitive resource modes. These are control states, not consciousness levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveResourceMode {
    Full,
    Focused,
    Recovery,
    Survival,
}

/// Thresholds for mapping viability pressure into bounded cognitive resource allocation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegulationThresholds {
    pub focused: f64,
    pub recovery: f64,
    pub survival: f64,
}

impl Default for RegulationThresholds {
    fn default() -> Self {
        Self {
            focused: 0.20,
            recovery: 0.45,
            survival: 0.75,
        }
    }
}

impl RegulationThresholds {
    pub fn validate(&self) -> bool {
        (0.0..=1.0).contains(&self.focused)
            && (0.0..=1.0).contains(&self.recovery)
            && (0.0..=1.0).contains(&self.survival)
            && self.focused <= self.recovery
            && self.recovery <= self.survival
    }
}

/// Bounded cognitive response to viability pressure.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegulationDecision {
    pub mode: CognitiveResourceMode,
    pub pressure: f64,
    pub planning_horizon_scale: f64,
    pub exploration_scale: f64,
    pub consolidation_priority: f64,
    pub low_priority_cognition_scale: f64,
}

impl RegulationDecision {
    pub fn from_pressure(pressure: f64, thresholds: RegulationThresholds) -> Self {
        let p = pressure.clamp(0.0, 1.0);
        let mode = if p >= thresholds.survival {
            CognitiveResourceMode::Survival
        } else if p >= thresholds.recovery {
            CognitiveResourceMode::Recovery
        } else if p >= thresholds.focused {
            CognitiveResourceMode::Focused
        } else {
            CognitiveResourceMode::Full
        };

        // Pressure contracts planning/exploration and reallocates computation toward
        // stabilization. These mappings are deliberately simple and inspectable.
        let planning_horizon_scale = (1.0 - 0.65 * p).clamp(0.25, 1.0);
        let exploration_scale = (1.0 - p).clamp(0.0, 1.0);
        let consolidation_priority = (0.15 + 0.85 * p).clamp(0.0, 1.0);
        let low_priority_cognition_scale = (1.0 - 0.80 * p).clamp(0.10, 1.0);

        Self {
            mode,
            pressure: p,
            planning_horizon_scale,
            exploration_scale,
            consolidation_priority,
            low_priority_cognition_scale,
        }
    }
}

/// Minimal orchestration container. Existing organs own their domain logic; this fabric
/// only records typed observations and action/prediction relationships.
#[derive(Debug)]
pub struct ViabilityFabric {
    state: ViabilityState,
    pending_predictions: BTreeMap<u64, ActionPrediction>,
    outcomes: Vec<ActionOutcome>,
    cancellations: Vec<PredictionCancellation>,
    max_outcomes: usize,
    max_pending_predictions: usize,
    highest_action_id: u64,
}

impl Default for ViabilityFabric {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl ViabilityFabric {
    pub fn new(max_outcomes: usize) -> Self {
        Self {
            state: ViabilityState::default(),
            pending_predictions: BTreeMap::new(),
            outcomes: Vec::with_capacity(max_outcomes.min(1024)),
            cancellations: Vec::with_capacity(max_outcomes.min(1024)),
            max_outcomes,
            max_pending_predictions: max_outcomes.max(1).min(4096),
            highest_action_id: 0,
        }
    }

    pub fn state(&self) -> &ViabilityState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut ViabilityState {
        &mut self.state
    }

    pub fn begin_cycle(&mut self, cycle: u64) {
        self.state.cycle = cycle;
    }

    pub fn set_lifecycle(&mut self, lifecycle: LifecyclePhase) {
        self.state.lifecycle = lifecycle;
    }

    pub fn observe_variable(
        &mut self,
        name: impl Into<String>,
        observation: ViabilitySignal,
        band: ViabilityBand,
        prediction: Option<ViabilitySignal>,
    ) {
        let prediction_error = prediction
            .as_ref()
            .map(|predicted| (predicted.value - observation.value).abs().clamp(0.0, 1.0));

        self.state.resource_pressure.insert(
            name.into(),
            ViabilityVariable {
                observation: observation.clone(),
                band,
                rate_of_change: 0.0,
                prediction,
                prediction_error,
            },
        );
    }

    /// Record a prediction before action execution.
    pub fn predict_action(&mut self, prediction: ActionPrediction) -> Result<(), &'static str> {
        if self.pending_predictions.contains_key(&prediction.action_id) {
            return Err("duplicate action prediction");
        }
        if prediction.action_id <= self.highest_action_id {
            return Err("action id is not monotonic");
        }
        if self.pending_predictions.len() >= self.max_pending_predictions {
            return Err("pending prediction capacity exhausted");
        }
        if prediction.cycle != self.state.cycle {
            return Err("prediction cycle does not match current cycle");
        }
        if prediction.action_label.trim().is_empty() {
            return Err("empty action label");
        }
        if prediction
            .predicted_world_delta
            .as_ref()
            .is_some_and(|delta| !delta.is_valid())
            || prediction
                .predicted_self_delta
                .as_ref()
                .is_some_and(|delta| !delta.is_valid())
            || prediction
                .predicted_goal_delta
                .as_ref()
                .is_some_and(|delta| !delta.is_valid())
        {
            return Err("invalid action prediction");
        }
        self.highest_action_id = prediction.action_id;
        self.pending_predictions.insert(prediction.action_id, prediction);
        Ok(())
    }

    /// Close a pre-existing action prediction with observed evidence.
    ///
    /// The prediction is removed from the pending set and attached to the outcome.
    /// A caller cannot inject a prediction at observation time: this is deliberately
    /// fail-closed against post-hoc rationalization.
    pub fn observe_action(&mut self, mut outcome: ActionOutcome) -> Result<(), &'static str> {
        // Validate caller-supplied outcome metadata before consuming the pending prediction.
        // This preserves the pending record after rejected/tampered observations.
        if outcome.prediction.is_some() {
            return Err("outcome already contains a prediction");
        }
        if outcome.evidence_refs.is_empty() {
            return Err("missing evidence reference");
        }
        if !outcome.prediction_error.world.is_finite()
            || !outcome.prediction_error.self_model.is_finite()
            || !outcome.prediction_error.interoceptive.is_finite()
            || !outcome.prediction_error.goal.is_finite()
            || !outcome.prediction_error.model_confidence.is_finite()
            || !outcome.prediction_error.execution.is_finite()
        {
            return Err("non-finite prediction error");
        }

        let Some(prediction) = self.pending_predictions.get(&outcome.action_id) else {
            return Err("missing pre-action prediction");
        };

        if prediction.action_label != outcome.action_label {
            return Err("action label mismatch");
        }
        if prediction.authority_granted != outcome.authority_granted {
            return Err("authority mismatch");
        }
        if outcome.cycle < prediction.cycle {
            return Err("outcome predates prediction");
        }
        if !outcome.evidence_refs.iter().all(|r| !r.trim().is_empty()) {
            return Err("invalid evidence reference");
        }
        if let Some(effect) = &outcome.observed_effect {
            if !effect.is_valid() {
                return Err("invalid observed effect");
            }
        }
        if prediction
            .predicted_world_delta
            .as_ref()
            .is_some_and(|delta| !delta.is_valid())
            || prediction
                .predicted_self_delta
                .as_ref()
                .is_some_and(|delta| !delta.is_valid())
            || prediction
                .predicted_goal_delta
                .as_ref()
                .is_some_and(|delta| !delta.is_valid())
        {
            return Err("invalid action prediction");
        }

        let prediction = self
            .pending_predictions
            .remove(&outcome.action_id)
            .expect("pending prediction validated immediately before removal");

        outcome.prediction = Some(prediction);
        outcome.prediction_error = outcome.prediction_error.bounded();

        if self.max_outcomes > 0 && self.outcomes.len() >= self.max_outcomes {
            self.outcomes.remove(0);
        }
        self.outcomes.push(outcome);
        Ok(())
    }

    pub fn outcomes(&self) -> &[ActionOutcome] {
        &self.outcomes
    }

    pub fn cancellations(&self) -> &[PredictionCancellation] {
        &self.cancellations
    }

    /// Explicitly close a prediction when the action is cancelled or could not execute.
    ///
    /// This is preferable to silently dropping a pending prediction: long-lived
    /// traces must distinguish "not observed" from "never happened."
    pub fn cancel_prediction(
        &mut self,
        action_id: u64,
        cancellation_cycle: u64,
        reason: impl Into<String>,
        evidence_refs: Vec<String>,
    ) -> Result<(), &'static str> {
        let Some(prediction) = self.pending_predictions.get(&action_id) else {
            return Err("missing pre-action prediction");
        };
        if cancellation_cycle < prediction.cycle {
            return Err("cancellation predates prediction");
        }
        if evidence_refs.is_empty() {
            return Err("missing evidence reference");
        }
        if !evidence_refs.iter().all(|r| !r.trim().is_empty()) {
            return Err("invalid evidence reference");
        }

        let prediction_cycle = prediction.cycle;
        self.pending_predictions.remove(&action_id);

        if self.max_outcomes > 0 && self.cancellations.len() >= self.max_outcomes {
            self.cancellations.remove(0);
        }
        self.cancellations.push(PredictionCancellation {
            action_id,
            prediction_cycle,
            cancellation_cycle,
            reason: reason.into(),
            evidence_refs,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band() -> ViabilityBand {
        ViabilityBand {
            preferred: (0.2, 0.7),
            tolerated: (0.1, 0.85),
            critical: (0.0, 0.95),
        }
    }

    #[test]
    fn band_is_ordered() {
        assert!(band().validate());
    }

    #[test]
    fn pressure_is_zero_inside_preferred_band() {
        let variable = ViabilityVariable {
            observation: ViabilitySignal::new(0.5, 1.0, 1, "test"),
            band: band(),
            rate_of_change: 0.0,
            prediction: None,
            prediction_error: None,
        };
        assert_eq!(variable.normalized_pressure(), 0.0);
    }

    #[test]
    fn pressure_rises_outside_preferred_band() {
        let variable = ViabilityVariable {
            observation: ViabilitySignal::new(1.0, 1.0, 1, "test"),
            band: band(),
            rate_of_change: 0.0,
            prediction: None,
            prediction_error: None,
        };
        assert!(variable.normalized_pressure() > 0.0);
    }

    #[test]
    fn prediction_must_precede_action_outcome() {
        let mut fabric = ViabilityFabric::new(4);
        fabric.begin_cycle(7);

        let outcome = ActionOutcome {
            action_id: 42,
            action_label: "test".to_string(),
            cycle: 7,
            pre_state_digest: 1,
            post_state_digest: 2,
            authority_granted: true,
            safety_gate_passed: true,
            prediction: None,
            observed_effect: None,
            prediction_error: PredictionErrorLedger::default(),
            evidence_refs: Vec::new(),
        };

        assert_eq!(fabric.observe_action(outcome), Err("missing pre-action prediction"));
    }

    #[test]
    fn signed_delta_preserves_negative_consequences() {
        let delta = ViabilityDelta::new(-0.4, 0.9);
        assert_eq!(delta.value, -0.4);
        assert!(delta.is_valid());
    }

    #[test]
    fn invalid_band_is_rejected() {
        let invalid = ViabilityBand {
            preferred: (0.8, 0.2),
            tolerated: (0.1, 0.9),
            critical: (0.0, 1.0),
        };
        assert!(!invalid.validate());
    }

    #[test]
    fn post_hoc_prediction_in_outcome_is_rejected() {
        let mut fabric = ViabilityFabric::new(4);
        fabric.begin_cycle(7);

        let injected_prediction = ActionPrediction {
            action_id: 42,
            action_label: "test".to_string(),
            cycle: 7,
            predicted_world_delta: None,
            predicted_self_delta: None,
            predicted_goal_delta: None,
            authority_granted: true,
        };

        let outcome = ActionOutcome {
            action_id: 42,
            action_label: "test".to_string(),
            cycle: 7,
            pre_state_digest: 1,
            post_state_digest: 2,
            authority_granted: true,
            safety_gate_passed: true,
            prediction: Some(injected_prediction),
            observed_effect: None,
            prediction_error: PredictionErrorLedger::default(),
            evidence_refs: Vec::new(),
        };

        assert_eq!(fabric.observe_action(outcome), Err("missing pre-action prediction"));
        assert!(fabric.outcomes().is_empty());
    }

    #[test]
    fn rejected_tampered_outcome_does_not_consume_prediction() {
        let mut fabric = ViabilityFabric::new(4);
        fabric.begin_cycle(9);

        let prediction = ActionPrediction {
            action_id: 10,
            action_label: "test".to_string(),
            cycle: 9,
            predicted_world_delta: Some(ViabilityDelta::new(-0.2, 1.0)),
            predicted_self_delta: None,
            predicted_goal_delta: None,
            authority_granted: true,
        };
        fabric.predict_action(prediction).unwrap();

        let tampered = ActionOutcome {
            action_id: 10,
            action_label: "tampered".to_string(),
            cycle: 9,
            pre_state_digest: 1,
            post_state_digest: 2,
            authority_granted: true,
            safety_gate_passed: true,
            prediction: None,
            observed_effect: None,
            prediction_error: PredictionErrorLedger::default(),
            evidence_refs: vec![],
        };

        assert_eq!(fabric.observe_action(tampered), Err("action label mismatch"));

        let good = ActionOutcome {
            action_id: 10,
            action_label: "test".to_string(),
            cycle: 9,
            pre_state_digest: 1,
            post_state_digest: 2,
            authority_granted: true,
            safety_gate_passed: true,
            prediction: None,
            observed_effect: None,
            prediction_error: PredictionErrorLedger {
                world: 0.2,
                ..Default::default()
            },
            evidence_refs: vec!["sim://micro-world/episode-1".to_string()],
        };

        assert!(fabric.observe_action(good).is_ok());
    }

    #[test]
    fn regulation_modes_follow_pressure() {
        let full = RegulationDecision::from_pressure(0.1, RegulationThresholds::default());
        let focused = RegulationDecision::from_pressure(0.3, RegulationThresholds::default());
        let recovery = RegulationDecision::from_pressure(0.6, RegulationThresholds::default());
        let survival = RegulationDecision::from_pressure(0.9, RegulationThresholds::default());

        assert_eq!(full.mode, CognitiveResourceMode::Full);
        assert_eq!(focused.mode, CognitiveResourceMode::Focused);
        assert_eq!(recovery.mode, CognitiveResourceMode::Recovery);
        assert_eq!(survival.mode, CognitiveResourceMode::Survival);
        assert!(survival.exploration_scale < full.exploration_scale);
        assert!(survival.consolidation_priority > full.consolidation_priority);
    }

    #[test]
    fn invalid_thresholds_fail_to_safe_defaults() {
        let thresholds = RegulationThresholds {
            focused: 0.8,
            recovery: 0.2,
            survival: 0.1,
        };
        let decision = RegulationDecision::from_pressure(0.5, thresholds);
        assert_eq!(decision.mode, CognitiveResourceMode::Recovery);
    }

    #[test]
    fn default_fabric_has_working_prediction_capacity() {
        let mut fabric = ViabilityFabric::default();
        fabric.begin_cycle(1);
        let prediction = ActionPrediction {
            action_id: 1,
            action_label: "default".to_string(),
            cycle: 1,
            predicted_world_delta: None,
            predicted_self_delta: None,
            predicted_goal_delta: None,
            authority_granted: false,
        };
        assert!(fabric.predict_action(prediction).is_ok());
    }

    #[test]
    fn cancellation_closes_pending_prediction_explicitly() {
        let mut fabric = ViabilityFabric::new(4);
        fabric.begin_cycle(2);
        fabric.predict_action(ActionPrediction {
            action_id: 1,
            action_label: "test".to_string(),
            cycle: 2,
            predicted_world_delta: None,
            predicted_self_delta: None,
            predicted_goal_delta: None,
            authority_granted: false,
        }).unwrap();

        assert!(fabric
            .cancel_prediction(1, 3, "actuator unavailable", vec!["sim://cancel/1".to_string()])
            .is_ok());
        assert_eq!(fabric.cancellations().len(), 1);
        assert!(fabric.outcomes().is_empty());

        let outcome = ActionOutcome {
            action_id: 1,
            action_label: "test".to_string(),
            cycle: 3,
            pre_state_digest: 1,
            post_state_digest: 2,
            authority_granted: false,
            safety_gate_passed: false,
            prediction: None,
            observed_effect: None,
            prediction_error: PredictionErrorLedger::default(),
            evidence_refs: vec!["sim://cancel/1".to_string()],
        };
        assert_eq!(fabric.observe_action(outcome), Err("missing pre-action prediction"));
    }

    #[test]
    fn action_ids_must_be_monotonic() {
        let mut fabric = ViabilityFabric::new(4);
        fabric.begin_cycle(1);

        let prediction = ActionPrediction {
            action_id: 2,
            action_label: "test".to_string(),
            cycle: 1,
            predicted_world_delta: None,
            predicted_self_delta: None,
            predicted_goal_delta: None,
            authority_granted: true,
        };

        fabric.predict_action(prediction.clone()).unwrap();
        assert_eq!(
            fabric.predict_action(prediction),
            Err("duplicate action prediction")
        );

        let prediction_1 = ActionPrediction {
            action_id: 1,
            action_label: "test".to_string(),
            cycle: 1,
            predicted_world_delta: None,
            predicted_self_delta: None,
            predicted_goal_delta: None,
            authority_granted: true,
        };
        assert_eq!(fabric.predict_action(prediction_1), Err("action id is not monotonic"));
    }

    #[test]
    fn pending_prediction_capacity_is_bounded() {
        let mut fabric = ViabilityFabric::new(2);
        fabric.begin_cycle(1);

        for action_id in [1, 2] {
            fabric.predict_action(ActionPrediction {
                action_id,
                action_label: "test".to_string(),
                cycle: 1,
                predicted_world_delta: None,
                predicted_self_delta: None,
                predicted_goal_delta: None,
                authority_granted: true,
            }).unwrap();
        }

        assert_eq!(
            fabric.predict_action(ActionPrediction {
                action_id: 3,
                action_label: "test".to_string(),
                cycle: 1,
                predicted_world_delta: None,
                predicted_self_delta: None,
                predicted_goal_delta: None,
                authority_granted: true,
            }),
            Err("pending prediction capacity exhausted")
        );
    }

    #[test]
    fn duplicate_prediction_fails_closed() {
        let mut fabric = ViabilityFabric::new(4);
        fabric.begin_cycle(3);

        let p = ActionPrediction {
            action_id: 9,
            action_label: "test".to_string(),
            cycle: 3,
            predicted_world_delta: None,
            predicted_self_delta: None,
            predicted_goal_delta: None,
            authority_granted: true,
        };

        fabric.predict_action(p.clone()).unwrap();
        assert_eq!(fabric.predict_action(p), Err("duplicate action prediction"));
    }
}
