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
use std::collections::{BTreeMap, VecDeque};

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
    /// Anticipatory pressure from movement toward/through a preferred-band boundary.
    ///
    /// This is deliberately bounded and conservative: worsening motion only adds pressure;
    /// improving motion does not erase pressure caused by the current absolute state.
    pub fn anticipatory_pressure(&self) -> f64 {
        let value = self.observation.value;
        let rate = self.rate_of_change;
        if !value.is_finite() || !rate.is_finite() || !self.band.validate() {
            return 1.0;
        }

        let (lo, hi) = self.band.preferred;
        let tolerance_width = (self.band.tolerated.1 - self.band.tolerated.0).max(f64::EPSILON);
        let distance = if rate > 0.0 && value >= lo {
            (hi - value).max(0.0)
        } else if rate < 0.0 && value <= hi {
            (value - lo).max(0.0)
        } else {
            0.0
        };

        if distance <= 0.0 {
            return 1.0_f64.min(rate.abs() / tolerance_width);
        }

        (rate.abs() / (distance + tolerance_width)).clamp(0.0, 1.0)
    }

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
    /// Digest of the exact pre-action state used to produce this prediction.
    pub pre_state_digest: u64,
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

/// Compact tamper-evident receipt for one accepted viability event.
///
/// The event digest binds the accepted event payload. The chain digest then binds the
/// event to the prior receipt. This is local tamper evidence, not an external signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViabilityTraceReceipt {
    pub sequence: u64,
    pub event_kind: String,
    pub action_id: u64,
    pub cycle: u64,
    pub event_digest: [u8; 32],
    pub previous_digest: [u8; 32],
    pub chain_digest: [u8; 32],
}

const VIABILITY_TRACE_DOMAIN: &[u8] = b"symthaea-viability-trace:v1";

fn trace_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn trace_delta(hasher: &mut blake3::Hasher, delta: &Option<ViabilityDelta>) {
    match delta {
        Some(delta) => {
            hasher.update(&[1]);
            hasher.update(&delta.value.to_bits().to_le_bytes());
            hasher.update(&delta.confidence.to_bits().to_le_bytes());
        }
        None => hasher.update(&[0]),
    }
}

fn trace_error(hasher: &mut blake3::Hasher, error: &PredictionErrorLedger) {
    for value in [
        error.world,
        error.self_model,
        error.interoceptive,
        error.goal,
        error.model_confidence,
        error.execution,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn trace_effect(hasher: &mut blake3::Hasher, effect: &Option<ViabilitySignal>) {
    match effect {
        Some(effect) => {
            hasher.update(&[1]);
            hasher.update(&effect.value.to_bits().to_le_bytes());
            hasher.update(&effect.confidence.to_bits().to_le_bytes());
            hasher.update(&effect.cycle.to_le_bytes());
            trace_string(hasher, &effect.producer);
        }
        None => hasher.update(&[0]),
    }
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
                .map(|variable| {
                    variable
                        .normalized_pressure()
                        .max(variable.anticipatory_pressure())
                })
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
    trace: VecDeque<ViabilityTraceReceipt>,
    max_trace: usize,
    trace_head: [u8; 32],
    trace_sequence: u64,
}

/// Cycle-level telemetry view of the viability fabric.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ViabilityTelemetry {
    pub viability_cycle: u64,
    pub viability_lifecycle: String,
    pub viability_resource_mode: String,
    pub viability_pressure: f64,
    pub viability_resource_pressure: f64,
    pub viability_world_prediction_error: f64,
    pub viability_self_prediction_error: f64,
    pub viability_interoceptive_prediction_error: f64,
    pub viability_goal_prediction_error: f64,
    pub viability_model_uncertainty: f64,
    pub viability_execution_prediction_error: f64,
    /// Remaining fraction of the canonical FEP thermodynamic ledger capacity.
    pub viability_energy_reserve: f64,
    /// BLAKE3 head of the locally chained viability evidence trace.
    pub viability_trace_digest: [u8; 32],
    /// Exact multiplier applied to the existing temporal planning-depth factor.
    /// 1.0 means no viability control influence.
    pub viability_planning_horizon_scale: f64,
}

impl CognitiveResourceMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "Full",
            Self::Focused => "Focused",
            Self::Recovery => "Recovery",
            Self::Survival => "Survival",
        }
    }
}

impl LifecyclePhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "Active",
            Self::Recovery => "Recovery",
            Self::Consolidation => "Consolidation",
        }
    }
}

impl ViabilityState {
    pub fn telemetry(&self) -> ViabilityTelemetry {
        let decision = self.regulation_decision(RegulationThresholds::default());
        ViabilityTelemetry {
            viability_cycle: self.cycle,
            viability_lifecycle: self.lifecycle.as_str().to_string(),
            viability_resource_mode: decision.mode.as_str().to_string(),
            viability_pressure: decision.pressure,
            viability_resource_pressure: self.aggregate_pressure(),
            viability_world_prediction_error: self.prediction_errors.world,
            viability_self_prediction_error: self.prediction_errors.self_model,
            viability_interoceptive_prediction_error: self.prediction_errors.interoceptive,
            viability_goal_prediction_error: self.prediction_errors.goal,
            viability_model_uncertainty: self.prediction_errors.model_confidence,
            viability_execution_prediction_error: self.prediction_errors.execution,
            viability_energy_reserve: self
                .resource_pressure
                .get("thermodynamic_energy_reserve")
                .map(|v| v.observation.value)
                .unwrap_or(0.0),
            viability_trace_digest: [0u8; 32],
            viability_planning_horizon_scale: 1.0,
        }
    }
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
            trace: VecDeque::with_capacity(max_outcomes.min(1024)),
            max_trace: max_outcomes,
            trace_head: *blake3::hash(VIABILITY_TRACE_DOMAIN).as_bytes(),
            trace_sequence: 0,
        }
    }

    pub fn state(&self) -> &ViabilityState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut ViabilityState {
        &mut self.state
    }

    /// Snapshot the retained local evidence chain.
    pub fn trace(&self) -> Vec<ViabilityTraceReceipt> {
        self.trace.iter().cloned().collect()
    }

    pub fn latest_trace_digest(&self) -> [u8; 32] {
        self.trace_head
    }

    /// Verify sequence, predecessor links, and chain digests for the retained trace.
    ///
    /// When the ring has evicted older receipts, verification intentionally starts
    /// from the first retained predecessor instead of falsely requiring the genesis hash.
    pub fn verify_trace(&self) -> bool {
        let genesis = *blake3::hash(VIABILITY_TRACE_DOMAIN).as_bytes();
        if self.trace.is_empty() {
            return self.trace_head == genesis;
        }

        let mut previous_digest = None;
        let mut previous_sequence = None;

        for receipt in &self.trace {
            if let Some(previous_sequence) = previous_sequence {
                if receipt.sequence != previous_sequence.saturating_add(1) {
                    return false;
                }
                if receipt.previous_digest != previous_digest.expect("previous receipt exists") {
                    return false;
                }
            }

            let mut hasher = blake3::Hasher::new();
            hasher.update(VIABILITY_TRACE_DOMAIN);
            hasher.update(&receipt.sequence.to_le_bytes());
            trace_string(&mut hasher, &receipt.event_kind);
            hasher.update(&receipt.action_id.to_le_bytes());
            hasher.update(&receipt.cycle.to_le_bytes());
            hasher.update(&receipt.event_digest);
            if *hasher.finalize().as_bytes() != receipt.chain_digest {
                return false;
            }

            previous_sequence = Some(receipt.sequence);
            previous_digest = Some(receipt.chain_digest);
        }

        previous_digest == Some(self.trace_head)
    }

    fn append_trace_receipt(
        &mut self,
        event_kind: &str,
        action_id: u64,
        cycle: u64,
        event_digest: [u8; 32],
    ) {
        if self.max_trace == 0 {
            return;
        }

        let sequence = self.trace_sequence;
        let previous_digest = self.trace_head;
        let mut hasher = blake3::Hasher::new();
        hasher.update(VIABILITY_TRACE_DOMAIN);
        hasher.update(&sequence.to_le_bytes());
        trace_string(&mut hasher, event_kind);
        hasher.update(&action_id.to_le_bytes());
        hasher.update(&cycle.to_le_bytes());
        hasher.update(&event_digest);
        let chain_digest = *hasher.finalize().as_bytes();

        if self.trace.len() >= self.max_trace {
            self.trace.pop_front();
        }
        self.trace.push_back(ViabilityTraceReceipt {
            sequence,
            event_kind: event_kind.to_string(),
            action_id,
            cycle,
            event_digest,
            previous_digest,
            chain_digest,
        });
        self.trace_head = chain_digest;
        self.trace_sequence = self.trace_sequence.saturating_add(1);
    }

    fn append_trace_outcome(&mut self, outcome: &ActionOutcome) {
        if self.max_trace == 0 {
            return;
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(VIABILITY_TRACE_DOMAIN);
        trace_string(&mut hasher, "outcome");
        hasher.update(&outcome.action_id.to_le_bytes());
        hasher.update(&outcome.cycle.to_le_bytes());
        trace_string(&mut hasher, &outcome.action_label);
        hasher.update(&outcome.pre_state_digest.to_le_bytes());
        hasher.update(&outcome.post_state_digest.to_le_bytes());
        hasher.update(&[outcome.authority_granted as u8, outcome.safety_gate_passed as u8]);
        trace_error(&mut hasher, &outcome.prediction_error);
        trace_effect(&mut hasher, &outcome.observed_effect);

        match &outcome.prediction {
            Some(prediction) => {
                hasher.update(&[1]);
                hasher.update(&prediction.action_id.to_le_bytes());
                hasher.update(&prediction.pre_state_digest.to_le_bytes());
                hasher.update(&prediction.cycle.to_le_bytes());
                trace_string(&mut hasher, &prediction.action_label);
                trace_delta(&mut hasher, &prediction.predicted_world_delta);
                trace_delta(&mut hasher, &prediction.predicted_self_delta);
                trace_delta(&mut hasher, &prediction.predicted_goal_delta);
                hasher.update(&[prediction.authority_granted as u8]);
            }
            None => hasher.update(&[0]),
        }

        for evidence in &outcome.evidence_refs {
            trace_string(&mut hasher, evidence);
        }

        self.append_trace_receipt(
            "outcome",
            outcome.action_id,
            outcome.cycle,
            *hasher.finalize().as_bytes(),
        );
    }

    fn append_trace_cancellation(&mut self, cancellation: &PredictionCancellation) {
        if self.max_trace == 0 {
            return;
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(VIABILITY_TRACE_DOMAIN);
        trace_string(&mut hasher, "cancellation");
        hasher.update(&cancellation.action_id.to_le_bytes());
        hasher.update(&cancellation.prediction_cycle.to_le_bytes());
        hasher.update(&cancellation.cancellation_cycle.to_le_bytes());
        trace_string(&mut hasher, &cancellation.reason);
        for evidence in &cancellation.evidence_refs {
            trace_string(&mut hasher, evidence);
        }

        self.append_trace_receipt(
            "cancellation",
            cancellation.action_id,
            cancellation.cancellation_cycle,
            *hasher.finalize().as_bytes(),
        );
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
        let key = name.into();
        let previous = self.state.resource_pressure.get(&key);
        let rate_of_change = previous
            .map(|prev| {
                let delta_cycle = observation.cycle.saturating_sub(prev.observation.cycle);
                if delta_cycle == 0 {
                    // Telemetry may refresh the same variable more than once within a cycle.
                    // Preserve the already-derived trend rather than erasing it.
                    prev.rate_of_change
                } else {
                    (observation.value - prev.observation.value) / delta_cycle as f64
                }
            })
            .unwrap_or(0.0);

        let prediction_error = prediction
            .as_ref()
            .map(|predicted| (predicted.value - observation.value).abs().clamp(0.0, 1.0));

        self.state.resource_pressure.insert(
            key,
            ViabilityVariable {
                observation: observation.clone(),
                band,
                rate_of_change,
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
        // Zero is a valid digest value; identity is established by exact equality
        // with the later observed outcome, not by treating the hash as a nonce.
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
        if prediction.pre_state_digest != outcome.pre_state_digest {
            return Err("pre-action state digest mismatch");
        }
        if outcome.cycle < prediction.cycle {
            return Err("outcome predates prediction");
        }
        if outcome.evidence_refs.is_empty() {
            return Err("missing evidence reference");
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
        self.outcomes.push(outcome.clone());
        self.append_trace_outcome(&outcome);
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
        let cancellation = PredictionCancellation {
            action_id,
            prediction_cycle,
            cancellation_cycle,
            reason: reason.into(),
            evidence_refs,
        };
        self.cancellations.push(cancellation.clone());
        self.append_trace_cancellation(&cancellation);
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
    fn rate_of_change_is_derived_from_previous_observation() {
        let mut fabric = ViabilityFabric::new(4);
        let band = band();
        fabric.begin_cycle(1);
        fabric.observe_variable(
            "load",
            ViabilitySignal::new(0.4, 1.0, 1, "test"),
            band,
            None,
        );
        fabric.begin_cycle(2);
        fabric.observe_variable(
            "load",
            ViabilitySignal::new(0.6, 1.0, 2, "test"),
            band,
            None,
        );

        let variable = fabric
            .state()
            .resource_pressure
            .get("load")
            .expect("load exists");
        assert!((variable.rate_of_change - 0.2).abs() < 1e-12);
        assert!(variable.anticipatory_pressure() > 0.0);
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
    fn same_cycle_observation_preserves_derived_rate() {
        let mut fabric = ViabilityFabric::new(4);
        let band = band();
        fabric.begin_cycle(1);
        fabric.observe_variable(
            "load",
            ViabilitySignal::new(0.4, 1.0, 1, "test"),
            band,
            None,
        );
        fabric.begin_cycle(2);
        fabric.observe_variable(
            "load",
            ViabilitySignal::new(0.6, 1.0, 2, "test"),
            band,
            None,
        );

        let before = fabric
            .state()
            .resource_pressure
            .get("load")
            .expect("load exists")
            .rate_of_change;

        fabric.observe_variable(
            "load",
            ViabilitySignal::new(0.6, 1.0, 2, "test"),
            band,
            None,
        );

        let after = fabric
            .state()
            .resource_pressure
            .get("load")
            .expect("load exists")
            .rate_of_change;

        assert_eq!(before, after);
    }

    #[test]
    fn prediction_must_precede_action_outcome() {
        let mut fabric = ViabilityFabric::new(4);
        fabric.begin_cycle(7);

        let outcome = ActionOutcome {
            action_id: 42,
            pre_state_digest: 1,
            action_label: "test".to_string(),
            cycle: 7,
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
    fn trace_chain_is_verifiable_and_tamper_evident() {
        let mut fabric = ViabilityFabric::new(8);

        fabric.begin_cycle(1);
        fabric
            .predict_action(ActionPrediction {
                action_id: 1,
                pre_state_digest: 11,
                action_label: "observe".to_string(),
                cycle: 1,
                predicted_world_delta: Some(ViabilityDelta::new(-0.1, 1.0)),
                predicted_self_delta: None,
                predicted_goal_delta: None,
                authority_granted: true,
            })
            .unwrap();

        fabric
            .observe_action(ActionOutcome {
                action_id: 1,
                action_label: "observe".to_string(),
                cycle: 2,
                pre_state_digest: 11,
                post_state_digest: 12,
                authority_granted: true,
                safety_gate_passed: true,
                prediction: None,
                observed_effect: Some(ViabilitySignal::new(
                    0.45,
                    1.0,
                    2,
                    "trace-test",
                )),
                prediction_error: PredictionErrorLedger::default(),
                evidence_refs: vec!["sim://trace/outcome/1".to_string()],
            })
            .unwrap();

        fabric.begin_cycle(3);
        fabric
            .predict_action(ActionPrediction {
                action_id: 2,
                pre_state_digest: 13,
                action_label: "explore".to_string(),
                cycle: 3,
                predicted_world_delta: Some(ViabilityDelta::new(0.2, 0.9)),
                predicted_self_delta: None,
                predicted_goal_delta: None,
                authority_granted: true,
            })
            .unwrap();

        fabric
            .cancel_prediction(
                2,
                4,
                "synthetic actuator unavailable",
                vec!["sim://trace/cancel/2".to_string()],
            )
            .unwrap();

        let trace = fabric.trace();
        assert_eq!(trace.len(), 2);
        assert_eq!(trace[1].previous_digest, trace[0].chain_digest);
        assert_eq!(fabric.latest_trace_digest(), trace[1].chain_digest);
        assert!(fabric.verify_trace());

        fabric.trace.make_contiguous()[1].chain_digest[0] ^= 0x01;
        assert!(!fabric.verify_trace());
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
            pre_state_digest: 1,
            action_label: "test".to_string(),
            cycle: 7,
            predicted_world_delta: None,
            predicted_self_delta: None,
            predicted_goal_delta: None,
            authority_granted: true,
        };

        let outcome = ActionOutcome {
            action_id: 42,
            pre_state_digest: 1,
            action_label: "test".to_string(),
            cycle: 7,
            post_state_digest: 2,
            authority_granted: true,
            safety_gate_passed: true,
            prediction: Some(injected_prediction),
            observed_effect: None,
            prediction_error: PredictionErrorLedger::default(),
            evidence_refs: Vec::new(),
        };

        assert_eq!(
            fabric.observe_action(outcome),
            Err("outcome already contains a prediction")
        );
        assert!(fabric.outcomes().is_empty());
    }

    #[test]
    fn mismatched_pre_state_digest_does_not_consume_prediction() {
        let mut fabric = ViabilityFabric::new(4);
        fabric.begin_cycle(9);
        fabric.predict_action(ActionPrediction {
            action_id: 11,
            pre_state_digest: 100,
            action_label: "test".to_string(),
            cycle: 9,
            predicted_world_delta: None,
            predicted_self_delta: None,
            predicted_goal_delta: None,
            authority_granted: true,
        }).unwrap();

        let outcome = ActionOutcome {
            action_id: 11,
            action_label: "test".to_string(),
            cycle: 9,
            pre_state_digest: 101,
            post_state_digest: 102,
            authority_granted: true,
            safety_gate_passed: true,
            prediction: None,
            observed_effect: None,
            prediction_error: PredictionErrorLedger::default(),
            evidence_refs: vec!["sim://digest-mismatch".to_string()],
        };

        assert_eq!(
            fabric.observe_action(outcome),
            Err("pre-action state digest mismatch")
        );
        assert_eq!(fabric.outcomes().len(), 0);

        let matching = ActionOutcome {
            action_id: 11,
            action_label: "test".to_string(),
            cycle: 9,
            pre_state_digest: 100,
            post_state_digest: 102,
            authority_granted: true,
            safety_gate_passed: true,
            prediction: None,
            observed_effect: None,
            prediction_error: PredictionErrorLedger::default(),
            evidence_refs: vec!["sim://digest-match".to_string()],
        };
        assert!(fabric.observe_action(matching).is_ok());
    }

    #[test]
    fn rejected_tampered_outcome_does_not_consume_prediction() {
        let mut fabric = ViabilityFabric::new(4);
        fabric.begin_cycle(9);

        let prediction = ActionPrediction {
            action_id: 10,
            pre_state_digest: 1,
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
            pre_state_digest: 1,
            action_label: "tampered".to_string(),
            cycle: 9,
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
            pre_state_digest: 1,
            action_label: "test".to_string(),
            cycle: 9,
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
            pre_state_digest: 1,
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
            pre_state_digest: 1,
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
            pre_state_digest: 1,
            action_label: "test".to_string(),
            cycle: 3,
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
            pre_state_digest: 1,
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
            pre_state_digest: 1,
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
                pre_state_digest: 1,
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
            pre_state_digest: 1,
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
            pre_state_digest: 1,
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
