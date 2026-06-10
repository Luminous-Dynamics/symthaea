// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MAGI Loop Integration
//!
//! This module integrates the World-Grounded Prediction system with the existing
//! SelfModel to create a complete MAGI Loop implementation.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         WorldGroundedSelfModel                              │
//! │                                                                             │
//! │  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐            │
//! │  │    SelfModel    │  │ BrierScoreTracker│  │ ConstraintGate  │            │
//! │  │ (Self-prediction│  │ (World calibrat- │  │ (Execution mode │            │
//! │  │  & capabilities)│  │  ion & ECE)      │  │  control)       │            │
//! │  └────────┬────────┘  └────────┬─────────┘  └────────┬────────┘            │
//! │           │                    │                      │                     │
//! │           └────────────────────┴──────────────────────┘                     │
//! │                                │                                            │
//! │                    ┌───────────┴───────────┐                                │
//! │                    │  ContractRegistry     │                                │
//! │                    │  (Resolution rules)   │                                │
//! │                    └───────────────────────┘                                │
//! │                                                                             │
//! │  World Predictions ─────────────────────────────────────────────────────────│
//! │  (VecDeque<WorldPrediction>: pending predictions awaiting resolution)       │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## The MAGI Loop Flow
//!
//! 1. **PREDICT**: Create WorldPrediction with confidence and ResolutionContract
//! 2. **SELECT**: ConstraintGate determines ExecutionMode
//! 3. **EXECUTE**: Action runs (autonomous, dry-run, or supervised)
//! 4. **OBSERVE**: ResolutionAuthority resolves outcome
//! 5. **ATTRIBUTE**: Causal attribution for errors
//! 6. **UPDATE**: BrierScoreTracker updates calibration, SelfModel updates

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use super::calibration::{BrierScoreTracker, CalibrationConfig, CalibrationSummary};
use super::constraint_gate::{
    ConstraintGate, ConstraintGateConfig, ExecutionMode, GateDecision, GateStatistics,
};
use super::types::instant_now;
use super::world_prediction::{
    ContractRegistry, OutcomeCategory, PredictionDomain, Resolution, ResolutionContract, RiskTier,
    WorldActionContext, WorldPrediction,
};
use crate::consciousness::compositionality::CompositionalityEngine;

// ═══════════════════════════════════════════════════════════════════════════════
// SELF-MODEL STUB TYPES (Standalone for MAGI Loop)
// When full_consciousness feature is enabled, these can be replaced with the
// full self_model implementations
// ═══════════════════════════════════════════════════════════════════════════════

/// Domain of capability that can be modeled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityDomain {
    /// Logical and causal reasoning
    Reasoning,
    /// Information storage and retrieval
    Memory,
    /// Input processing and pattern recognition
    Perception,
    /// Communication and understanding
    Language,
    /// Adaptation and improvement
    Learning,
    /// Novel generation and exploration
    Creativity,
    /// Φ and information binding
    Integration,
    /// Self-reflection and monitoring
    Metacognition,
}

impl CapabilityDomain {
    /// Get all capability domains
    pub fn all() -> Vec<Self> {
        vec![
            Self::Reasoning,
            Self::Memory,
            Self::Perception,
            Self::Language,
            Self::Learning,
            Self::Creativity,
            Self::Integration,
            Self::Metacognition,
        ]
    }
}

/// Configuration for the self-model (stub)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModelConfig {
    /// Minimum predictions for calibration
    pub min_predictions_for_calibration: usize,
    /// Confidence decay rate
    pub confidence_decay_rate: f64,
    /// Initial capability estimate
    pub initial_capability: f64,
}

impl Default for SelfModelConfig {
    fn default() -> Self {
        Self {
            min_predictions_for_calibration: 10,
            confidence_decay_rate: 0.95,
            initial_capability: 0.5,
        }
    }
}

/// Prediction about system's own behavior (stub)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorPrediction {
    /// What behavior is predicted
    pub prediction: String,
    /// Confidence in this prediction
    pub confidence: f64,
    /// Domain of the behavior
    pub domain: CapabilityDomain,
}

/// Minimal self-model for MAGI Loop (stub implementation)
pub struct SelfModel {
    /// Current capability estimates per domain
    capabilities: HashMap<CapabilityDomain, f64>,
    /// Configuration
    _config: SelfModelConfig,
    /// Behavior predictions history
    behavior_history: VecDeque<BehaviorPrediction>,
}

impl SelfModel {
    /// Create a new self-model
    pub fn new(config: SelfModelConfig) -> Self {
        let initial = config.initial_capability;
        let mut capabilities = HashMap::new();
        for domain in CapabilityDomain::all() {
            capabilities.insert(domain, initial);
        }
        Self {
            capabilities,
            _config: config,
            behavior_history: VecDeque::new(),
        }
    }

    /// Get capability estimate for a domain
    pub fn get_capability(&self, domain: CapabilityDomain) -> f64 {
        *self.capabilities.get(&domain).unwrap_or(&0.5)
    }

    /// Update capability after observation
    pub fn update_capability(
        &mut self,
        domain: CapabilityDomain,
        observed: f64,
        learning_rate: f64,
    ) {
        let current = self.get_capability(domain);
        let updated = current * (1.0 - learning_rate) + observed * learning_rate;
        self.capabilities.insert(domain, updated);
    }

    /// Predict behavior in a domain
    pub fn predict_behavior(
        &mut self,
        domain: CapabilityDomain,
        description: impl Into<String>,
    ) -> BehaviorPrediction {
        let prediction = BehaviorPrediction {
            prediction: description.into(),
            confidence: self.get_capability(domain),
            domain,
        };
        self.behavior_history.push_back(prediction.clone());
        while self.behavior_history.len() > 100 {
            self.behavior_history.pop_front();
        }
        prediction
    }

    /// Predict behavior across multiple domains (average confidence)
    pub fn predict_behavior_multi(&self, domains: &[CapabilityDomain]) -> BehaviorPrediction {
        let avg_confidence = if domains.is_empty() {
            0.5
        } else {
            domains.iter().map(|d| self.get_capability(*d)).sum::<f64>() / domains.len() as f64
        };
        let primary_domain = domains
            .first()
            .copied()
            .unwrap_or(CapabilityDomain::Reasoning);
        BehaviorPrediction {
            prediction: format!("Multi-domain prediction across {:?}", domains),
            confidence: avg_confidence,
            domain: primary_domain,
        }
    }

    /// Get calibration statistics (stub)
    pub fn calibration_stats(&self) -> SelfModelCalibrationStats {
        let total_caps: f64 = self.capabilities.values().sum();
        let count = self.capabilities.len() as f64;
        SelfModelCalibrationStats {
            model_confidence: if count > 0.0 { total_caps / count } else { 0.5 },
            prediction_count: self.behavior_history.len(),
        }
    }
}

/// Statistics from self-model calibration (stub)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModelCalibrationStats {
    /// Overall model confidence
    pub model_confidence: f64,
    /// Number of predictions made
    pub prediction_count: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the world-grounded self model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldGroundedConfig {
    /// Self-model configuration
    pub self_model: SelfModelConfig,
    /// Calibration configuration
    pub calibration: CalibrationConfig,
    /// Constraint gate configuration
    pub constraint_gate: ConstraintGateConfig,
    /// Maximum pending predictions
    pub max_pending_predictions: usize,
    /// Auto-resolve predictions after timeout
    pub auto_resolve_timeout: bool,
}

impl Default for WorldGroundedConfig {
    fn default() -> Self {
        Self {
            self_model: SelfModelConfig::default(),
            calibration: CalibrationConfig::default(),
            constraint_gate: ConstraintGateConfig::default(),
            max_pending_predictions: 1000,
            auto_resolve_timeout: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CAUSAL ATTRIBUTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Causal attribution for prediction errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalAttribution {
    /// The prediction that was wrong
    pub prediction_id: String,

    /// The specific failure mode
    pub failure_mode: String,

    /// Missing information that would have prevented error
    pub missing_information: Vec<String>,

    /// Capability domains responsible
    pub responsible_domains: Vec<CapabilityDomain>,

    /// Testable prediction about when this will recur
    pub recurrence_prediction: Option<String>,

    /// Confidence in this attribution
    pub confidence: f64,

    /// When this attribution was created
    #[serde(skip, default = "instant_now")]
    pub created_at: Instant,
}

impl CausalAttribution {
    /// Create a new causal attribution
    pub fn new(prediction_id: impl Into<String>, failure_mode: impl Into<String>) -> Self {
        Self {
            prediction_id: prediction_id.into(),
            failure_mode: failure_mode.into(),
            missing_information: Vec::new(),
            responsible_domains: Vec::new(),
            recurrence_prediction: None,
            confidence: 0.5,
            created_at: Instant::now(),
        }
    }

    /// Add missing information
    pub fn with_missing_info(mut self, info: impl Into<String>) -> Self {
        self.missing_information.push(info.into());
        self
    }

    /// Add responsible domain
    pub fn with_domain(mut self, domain: CapabilityDomain) -> Self {
        self.responsible_domains.push(domain);
        self
    }

    /// Add recurrence prediction
    pub fn with_recurrence(mut self, prediction: impl Into<String>) -> Self {
        self.recurrence_prediction = Some(prediction.into());
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAGI LOOP STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Current state of the MAGI loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagiLoopState {
    /// How many complete loop iterations
    pub loop_iterations: usize,
    /// Total predictions made
    pub predictions_made: usize,
    /// Total predictions resolved
    pub predictions_resolved: usize,
    /// Total attributions generated
    pub attributions_generated: usize,
    /// Is the loop currently active?
    pub is_active: bool,
    /// Current calibration quality
    pub calibration_quality: CalibrationQuality,
}

/// Quality of calibration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationQuality {
    /// Not enough data
    Insufficient,
    /// Poorly calibrated (ECE > 0.20)
    Poor,
    /// Moderately calibrated (ECE 0.10-0.20)
    Moderate,
    /// Well calibrated (ECE < 0.10)
    Good,
    /// Excellently calibrated (ECE < 0.05)
    Excellent,
}

impl CalibrationQuality {
    fn from_ece(ece: f64, min_predictions: usize, actual_predictions: usize) -> Self {
        if actual_predictions < min_predictions {
            Self::Insufficient
        } else if ece > 0.20 {
            Self::Poor
        } else if ece > 0.10 {
            Self::Moderate
        } else if ece > 0.05 {
            Self::Good
        } else {
            Self::Excellent
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WORLD GROUNDED SELF MODEL
// ═══════════════════════════════════════════════════════════════════════════════

/// Self-model with world-grounded prediction and calibration
pub struct WorldGroundedSelfModel {
    /// Configuration
    config: WorldGroundedConfig,

    /// Core self-model for capabilities and self-prediction
    self_model: SelfModel,

    /// Calibration tracker for world predictions
    calibration: BrierScoreTracker,

    /// Constraint gate for execution mode control
    gate: ConstraintGate,

    /// Registry of resolution contracts
    contracts: ContractRegistry,

    /// Pending world predictions awaiting resolution
    pending_predictions: VecDeque<WorldPrediction>,

    /// Recent causal attributions
    attributions: VecDeque<CausalAttribution>,

    /// Loop state tracking
    loop_state: MagiLoopState,

    /// When this model was created
    _created_at: Instant,

    /// Optional compositionality engine for composing primitives during improvement.
    /// When set, the update step can create composed primitives (e.g. fallback
    /// compositions for weak capability domains) as part of self-improvement.
    compositionality_engine: Option<Arc<std::sync::Mutex<CompositionalityEngine>>>,
}

impl WorldGroundedSelfModel {
    /// Create a new world-grounded self model
    pub fn new(config: WorldGroundedConfig) -> Self {
        Self {
            self_model: SelfModel::new(config.self_model.clone()),
            calibration: BrierScoreTracker::new(config.calibration.clone()),
            gate: ConstraintGate::new(config.constraint_gate.clone()),
            contracts: ContractRegistry::with_defaults(),
            pending_predictions: VecDeque::new(),
            attributions: VecDeque::new(),
            loop_state: MagiLoopState {
                loop_iterations: 0,
                predictions_made: 0,
                predictions_resolved: 0,
                attributions_generated: 0,
                is_active: false,
                calibration_quality: CalibrationQuality::Insufficient,
            },
            config,
            _created_at: Instant::now(),
            compositionality_engine: None,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(WorldGroundedConfig::default())
    }

    /// Attach a compositionality engine for primitive composition during improvement.
    ///
    /// When set, the MAGI update step can compose primitives (e.g. creating
    /// fallback or parallel compositions for weak domains). This is opt-in:
    /// if never called, behaviour is unchanged.
    pub fn set_compositionality_engine(
        &mut self,
        engine: Arc<std::sync::Mutex<CompositionalityEngine>>,
    ) {
        self.compositionality_engine = Some(engine);
    }

    /// Get a reference to the compositionality engine, if attached.
    pub fn compositionality_engine(
        &self,
    ) -> Option<&Arc<std::sync::Mutex<CompositionalityEngine>>> {
        self.compositionality_engine.as_ref()
    }

    /// Restore from a persisted snapshot (WARM START)
    ///
    /// This is the key method for resuming a session with prior knowledge.
    /// It reconstructs the entire model state from a persistence snapshot,
    /// allowing the system to continue learning from where it left off.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut persistence = PersistenceManager::with_defaults();
    /// let startup = persistence.initialize()?;
    ///
    /// let model = match startup {
    ///     StartupMode::WarmStart { .. } => {
    ///         WorldGroundedSelfModel::from_snapshot(persistence.current())
    ///     }
    ///     _ => WorldGroundedSelfModel::with_defaults(),
    /// };
    /// ```
    pub fn from_snapshot(snapshot: &super::persistence::MagiStateSnapshot) -> Self {
        // Reconstruct configuration (use snapshot's gate config)
        let config = WorldGroundedConfig {
            constraint_gate: snapshot.gate_config.clone(),
            ..Default::default()
        };

        // Reconstruct calibration tracker from persisted data
        let calibration = BrierScoreTracker::from_persisted(
            config.calibration.clone(),
            &snapshot.calibration,
            &snapshot.global_stats,
        );

        // Reconstruct self model with persisted capability estimates
        let mut self_model = SelfModel::new(config.self_model.clone());
        for (domain, &estimate) in &snapshot.capability_estimates {
            self_model.update_capability(*domain, estimate, 1.0);
        }

        // Reconstruct loop state
        let loop_state = MagiLoopState {
            loop_iterations: snapshot.loop_state.loop_iterations,
            predictions_made: snapshot.loop_state.predictions_made,
            predictions_resolved: snapshot.loop_state.predictions_resolved,
            attributions_generated: snapshot.loop_state.attributions_generated,
            is_active: true, // Active since we're resuming
            calibration_quality: snapshot.loop_state.calibration_quality,
        };

        // Note: We don't restore pending_predictions or attributions
        // - Pending predictions from a previous session are stale
        // - Attributions are stored in snapshot.attribution_history but
        //   converting back to CausalAttribution would require Instant recreation

        Self {
            self_model,
            calibration,
            gate: ConstraintGate::new(snapshot.gate_config.clone()),
            contracts: ContractRegistry::with_defaults(),
            pending_predictions: VecDeque::new(),
            attributions: VecDeque::new(),
            loop_state,
            config,
            _created_at: Instant::now(),
            compositionality_engine: None,
        }
    }

    /// Check if this model was restored from a snapshot
    pub fn is_warm_start(&self) -> bool {
        self.loop_state.loop_iterations > 0 && self.loop_state.is_active
    }

    /// Get lifetime statistics summary
    pub fn lifetime_summary(&self) -> String {
        format!(
            "Lifetime: {} iterations, {} predictions, {} resolved, Brier={:.4}, ECE={:.4}",
            self.loop_state.loop_iterations,
            self.loop_state.predictions_made,
            self.loop_state.predictions_resolved,
            self.calibration.brier_score(),
            self.calibration.expected_calibration_error()
        )
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 1: PREDICT (Make world-grounded prediction)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Make a world-grounded prediction about an action's outcome
    pub fn predict(
        &mut self,
        claim: impl Into<String>,
        expected_outcome: OutcomeCategory,
        raw_confidence: f64,
        action: WorldActionContext,
    ) -> WorldPrediction {
        // Convert claim once
        let claim_str = claim.into();

        // Get or create resolution contract for this action type
        let contract = self.contracts.get_or_default(&action.action_type);

        // Adjust confidence based on domain calibration
        let adjusted_confidence = self.calibration.adjust_confidence(
            WorldPrediction::new(
                &claim_str,
                expected_outcome,
                raw_confidence,
                action.clone(),
                contract.clone(),
            )
            .domain,
            raw_confidence,
        );

        // Create the prediction
        let prediction = WorldPrediction::new(
            &claim_str,
            expected_outcome,
            adjusted_confidence,
            action,
            contract,
        );

        // Add to pending predictions
        self.pending_predictions.push_back(prediction.clone());
        self.loop_state.predictions_made += 1;

        // Trim pending if over limit
        while self.pending_predictions.len() > self.config.max_pending_predictions {
            self.pending_predictions.pop_front();
        }

        prediction
    }

    /// Make a prediction using a custom resolution contract
    pub fn predict_with_contract(
        &mut self,
        claim: impl Into<String>,
        expected_outcome: OutcomeCategory,
        raw_confidence: f64,
        action: WorldActionContext,
        contract: ResolutionContract,
    ) -> WorldPrediction {
        let adjusted_confidence = self.calibration.adjust_confidence(
            PredictionDomain::Factual, // Will be inferred
            raw_confidence,
        );

        let prediction = WorldPrediction::new(
            claim,
            expected_outcome,
            adjusted_confidence,
            action,
            contract,
        );

        self.pending_predictions.push_back(prediction.clone());
        self.loop_state.predictions_made += 1;

        prediction
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 3: SELECT (Determine execution mode)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Check if an action can proceed and in what mode
    pub fn check_execution_mode(&mut self, action: &WorldActionContext) -> GateDecision {
        self.gate.check(action, &self.calibration)
    }

    /// Get execution mode without modifying state
    pub fn peek_execution_mode(&self, action: &WorldActionContext) -> ExecutionMode {
        // Create temporary gate to not modify statistics
        let mut temp_gate = ConstraintGate::new(self.config.constraint_gate.clone());
        temp_gate.check(action, &self.calibration).mode
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 4: OBSERVE (Record outcomes)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Resolve a prediction with an observed outcome
    pub fn resolve_prediction(
        &mut self,
        prediction_id: &str,
        observed_outcome: OutcomeCategory,
        resolution_confidence: f64,
    ) -> Option<bool> {
        // Find the prediction index
        let idx = self
            .pending_predictions
            .iter()
            .position(|p| p.id == prediction_id)?;

        // Remove from pending (take ownership)
        let mut prediction = self.pending_predictions.remove(idx)?;

        // Determine if prediction was correct
        let was_correct = prediction.predicted_outcome == observed_outcome
            || (prediction.predicted_outcome.is_positive() == observed_outcome.is_positive());

        // Resolve the prediction
        if was_correct {
            prediction.resolve_true(observed_outcome, resolution_confidence);
        } else {
            prediction.resolve_false(observed_outcome, resolution_confidence);
        }

        // Record in calibration tracker
        self.calibration.record_prediction(&prediction);

        // Update loop state
        self.loop_state.predictions_resolved += 1;
        self.update_calibration_quality();

        // If not correct, generate causal attribution
        if !was_correct {
            self.generate_attribution(&prediction);
        }

        // Check for loop completion
        self.check_loop_completion();

        Some(was_correct)
    }

    /// Process expired predictions
    pub fn process_expired_predictions(&mut self) {
        if !self.config.auto_resolve_timeout {
            return;
        }

        let mut resolved_ids = Vec::new();

        for prediction in self.pending_predictions.iter_mut() {
            if prediction.is_pending() && prediction.is_expired() {
                prediction.resolve_timeout();
                resolved_ids.push(prediction.id.clone());
            }
        }

        // Clean up resolved predictions
        self.pending_predictions.retain(|p| !p.is_pending());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 5: ATTRIBUTE (Generate causal explanations)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Generate causal attribution for a failed prediction
    fn generate_attribution(&mut self, prediction: &WorldPrediction) {
        // Infer failure mode based on prediction and outcome
        let failure_mode = match &prediction.resolution {
            Resolution::False {
                observed_outcome,
                predicted_outcome,
                ..
            } => {
                format!(
                    "Predicted {:?} but observed {:?}",
                    predicted_outcome, observed_outcome
                )
            }
            Resolution::TimedOut { .. } => "Prediction timed out before resolution".to_string(),
            Resolution::Unclear { reason, .. } => format!("Resolution unclear: {}", reason),
            _ => return, // No attribution needed for correct predictions
        };

        // Infer responsible domains
        let responsible_domains = self.infer_responsible_domains(prediction);

        // Infer missing information
        let missing_info = self.infer_missing_information(prediction);

        let mut attribution =
            CausalAttribution::new(&prediction.id, failure_mode).with_recurrence(format!(
                "Similar failures likely when {} in domain {:?}",
                if self.calibration.is_domain_overconfident(prediction.domain) {
                    "overconfident"
                } else {
                    "uncertain"
                },
                prediction.domain
            ));

        for domain in responsible_domains {
            attribution = attribution.with_domain(domain);
        }

        for info in missing_info {
            attribution = attribution.with_missing_info(info);
        }

        self.attributions.push_back(attribution);
        self.loop_state.attributions_generated += 1;

        // Keep attributions bounded
        while self.attributions.len() > 100 {
            self.attributions.pop_front();
        }
    }

    /// Infer which capability domains were responsible for a failure
    fn infer_responsible_domains(&self, prediction: &WorldPrediction) -> Vec<CapabilityDomain> {
        let mut domains = Vec::new();

        // Map prediction domain to capability domains
        match prediction.domain {
            PredictionDomain::CodeExecution => {
                domains.push(CapabilityDomain::Reasoning);
                domains.push(CapabilityDomain::Language);
            }
            PredictionDomain::ToolUse => {
                domains.push(CapabilityDomain::Perception);
                domains.push(CapabilityDomain::Memory);
            }
            PredictionDomain::UserBehavior => {
                domains.push(CapabilityDomain::Reasoning);
                domains.push(CapabilityDomain::Memory);
            }
            PredictionDomain::SystemState => {
                domains.push(CapabilityDomain::Perception);
            }
            PredictionDomain::Factual => {
                domains.push(CapabilityDomain::Memory);
                domains.push(CapabilityDomain::Reasoning);
            }
        }

        // Always include metacognition for calibration failures
        if prediction.confidence > 0.7 {
            domains.push(CapabilityDomain::Metacognition);
        }

        domains
    }

    /// Infer what information was missing for correct prediction
    fn infer_missing_information(&self, prediction: &WorldPrediction) -> Vec<String> {
        let mut missing = Vec::new();

        // Check preconditions
        if !prediction.action_context.preconditions.is_empty() {
            missing.push(format!(
                "Preconditions may not have been verified: {:?}",
                prediction.action_context.preconditions
            ));
        }

        // Check if domain has poor calibration
        if self.calibration.is_domain_overconfident(prediction.domain) {
            missing.push(format!(
                "Domain {:?} has history of overconfidence",
                prediction.domain
            ));
        }

        // Check calibration data sufficiency
        let domain_cal = self.calibration.domain_calibration(prediction.domain);
        if let Some(cal) = domain_cal {
            if cal.prediction_count < 20 {
                missing.push(format!(
                    "Insufficient calibration data for domain (only {} predictions)",
                    cal.prediction_count
                ));
            }
        }

        missing
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 6: UPDATE (Improve from experience)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Update capability estimates based on attribution.
    ///
    /// When a compositionality engine is attached, this also attempts to compose
    /// primitives for the weak domains -- creating fallback or parallel
    /// compositions that can strengthen future reasoning.
    pub fn update_from_attribution(&mut self, attribution: &CausalAttribution) {
        // Decrease capability estimates for responsible domains
        for domain in &attribution.responsible_domains {
            self.self_model.update_capability(
                *domain,
                0.4, // Below average performance
                attribution.confidence,
            );
        }

        // If compositionality engine is available, try to compose improvement
        // primitives for the weak domains.
        if let Some(engine_arc) = &self.compositionality_engine {
            if let Ok(mut engine) = engine_arc.lock() {
                self.compose_improvements_for_attribution(&mut engine, attribution);
            }
        }
    }

    /// Use the compositionality engine to create composed primitives that may
    /// help with domains identified as weak by a causal attribution.
    ///
    /// Strategy: for each pair of responsible domains, create a parallel
    /// composition so that future reasoning can cross-pollinate.  For single
    /// weak domains, create a fixed-point (iterative refinement) composition.
    fn compose_improvements_for_attribution(
        &self,
        engine: &mut CompositionalityEngine,
        attribution: &CausalAttribution,
    ) {
        let domains: Vec<String> = attribution
            .responsible_domains
            .iter()
            .map(|d| format!("{:?}", d).to_lowercase())
            .collect();

        // Parallel compositions for each pair of weak domains
        if domains.len() >= 2 {
            for i in 0..domains.len() {
                for j in (i + 1)..domains.len() {
                    // Best-effort: ignore errors (primitive may not exist, etc.)
                    let _ = engine.compose_parallel(&domains[i], &domains[j]);
                }
            }
        }

        // Fixed-point refinement for each weak domain individually
        for domain_id in &domains {
            let _ = engine.compose_fixed_point(domain_id, Some(20), Some(0.95));
        }

        // If there is a strongest and weakest domain, compose a fallback
        // so the strong domain backs up the weak one.
        if let (Some(first), Some(last)) = (domains.first(), domains.last()) {
            if domains.len() >= 2 {
                let _ = engine.compose_fallback(first, last, 0.6);
            }
        }
    }

    /// Update calibration quality assessment
    fn update_calibration_quality(&mut self) {
        let ece = self.calibration.expected_calibration_error();
        let total = self.calibration.total_predictions();
        let min_required = self.config.calibration.min_predictions_for_ece;

        self.loop_state.calibration_quality =
            CalibrationQuality::from_ece(ece, min_required, total);
    }

    /// Check if a complete MAGI loop iteration occurred
    fn check_loop_completion(&mut self) {
        // A loop is complete when:
        // 1. A prediction was made
        // 2. It was resolved
        // 3. Calibration was updated
        // 4. (Optionally) Attribution was generated

        // Simple heuristic: every 10 resolved predictions counts as a loop
        if self.loop_state.predictions_resolved > 0
            && self.loop_state.predictions_resolved % 10 == 0
        {
            self.loop_state.loop_iterations += 1;
            self.loop_state.is_active = true;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get the underlying self model
    pub fn self_model(&self) -> &SelfModel {
        &self.self_model
    }

    /// Get mutable self model
    pub fn self_model_mut(&mut self) -> &mut SelfModel {
        &mut self.self_model
    }

    /// Get calibration tracker
    pub fn calibration(&self) -> &BrierScoreTracker {
        &self.calibration
    }

    /// Get constraint gate
    pub fn gate(&self) -> &ConstraintGate {
        &self.gate
    }

    /// Get contract registry
    pub fn contracts(&self) -> &ContractRegistry {
        &self.contracts
    }

    /// Get mutable contract registry
    pub fn contracts_mut(&mut self) -> &mut ContractRegistry {
        &mut self.contracts
    }

    /// Get pending predictions
    pub fn pending_predictions(&self) -> &VecDeque<WorldPrediction> {
        &self.pending_predictions
    }

    /// Get recent attributions
    pub fn recent_attributions(&self) -> &VecDeque<CausalAttribution> {
        &self.attributions
    }

    /// Get MAGI loop state
    pub fn loop_state(&self) -> &MagiLoopState {
        &self.loop_state
    }

    /// Get calibration summary
    pub fn calibration_summary(&self) -> CalibrationSummary {
        self.calibration.calibration_summary()
    }

    /// Get gate statistics
    pub fn gate_statistics(&self) -> GateStatistics {
        self.gate.statistics()
    }

    /// Predict behavior (delegates to self model)
    pub fn predict_behavior(&self, domains: &[CapabilityDomain]) -> BehaviorPrediction {
        self.self_model.predict_behavior_multi(domains)
    }

    /// Get summary of the world-grounded self model
    pub fn summary(&self) -> String {
        let mut s = String::from("=== World-Grounded Self Model ===\n\n");

        // MAGI Loop State
        s.push_str(&format!(
            "MAGI Loop Iterations: {}\n",
            self.loop_state.loop_iterations
        ));
        s.push_str(&format!(
            "Calibration Quality: {:?}\n\n",
            self.loop_state.calibration_quality
        ));

        // Calibration Stats
        let cal = self.calibration_summary();
        s.push_str(&format!("Brier Score: {:.4}\n", cal.global_brier));
        s.push_str(&format!("ECE: {:.4}\n", cal.global_ece));
        s.push_str(&format!("Accuracy: {:.1}%\n", cal.global_accuracy * 100.0));
        s.push_str(&format!("Total Predictions: {}\n\n", cal.total_predictions));

        // Gate Stats
        let gate = self.gate_statistics();
        s.push_str(&format!("Actions Checked: {}\n", gate.total_checked));
        s.push_str(&format!(
            "Autonomy Rate: {:.1}%\n",
            gate.autonomy_rate * 100.0
        ));

        // Pending Predictions
        let pending_count = self
            .pending_predictions
            .iter()
            .filter(|p| p.is_pending())
            .count();
        s.push_str(&format!("\nPending Predictions: {}\n", pending_count));

        // Self Model Summary (abbreviated)
        s.push_str(&format!(
            "\nModel Confidence: {:.1}%\n",
            self.self_model.calibration_stats().model_confidence * 100.0
        ));

        s
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ACTIVE INFERENCE INTEGRATION (Phase 3)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Compute Expected Free Energy (EFE) contribution from world calibration
    ///
    /// This integrates MAGI Loop calibration into the Active Inference framework.
    /// The EFE contribution captures:
    /// - **Pragmatic value**: How likely is success in this domain?
    /// - **Epistemic value**: How much uncertainty exists in this domain?
    /// - **Novelty penalty**: Should we avoid poorly-calibrated domains?
    ///
    /// Returns an `EfeContribution` that can be added to the base EFE calculation.
    pub fn compute_efe_contribution(&self, action: &WorldActionContext) -> EfeContribution {
        // Determine prediction domain from action type
        let domain = PredictionDomain::from_action_type(&action.action_type);

        // Get domain calibration stats
        let domain_cal = self.calibration.domain_calibration(domain);

        let (domain_accuracy, domain_ece, domain_count) = match domain_cal {
            Some(cal) => (cal.accuracy(), cal.ece, cal.prediction_count),
            None => (0.5, 0.25, 0), // Uninformative priors for unknown domains
        };

        // Pragmatic value: Based on historical success rate in this domain
        // Higher accuracy in domain → higher pragmatic value
        // Scale to [-1, 1] range where 0.5 accuracy = 0
        let pragmatic = (domain_accuracy - 0.5) * 2.0;

        // Epistemic value: Based on calibration error (uncertainty about predictions)
        // Higher ECE means higher uncertainty → higher epistemic value (need to explore)
        // Also factor in data sufficiency
        let data_sufficiency = (domain_count as f64 / 50.0).min(1.0);
        let epistemic = domain_ece * (1.0 + (1.0 - data_sufficiency));

        // Novelty: Penalize domains that are overconfident (miscalibrated)
        // Overconfidence is dangerous → reduce novelty bonus
        let is_overconfident = self.calibration.is_domain_overconfident(domain);
        let novelty = if is_overconfident {
            -0.1 // Penalty for overconfident domains
        } else if domain_count < 10 {
            0.2 // Bonus for exploring under-sampled domains
        } else {
            0.0
        };

        // Risk modifier: High-risk actions reduce pragmatic value
        let risk_modifier = match action.risk_tier {
            RiskTier::Observation => 1.0,
            RiskTier::Reversible => 0.9,
            RiskTier::StateModifying => 0.7,
            RiskTier::Destructive => 0.4,
            RiskTier::Critical => 0.1,
        };

        EfeContribution {
            pragmatic: pragmatic * risk_modifier,
            epistemic,
            novelty,
            domain,
            calibration_quality: self.loop_state.calibration_quality,
            action_risk: action.risk_tier,
            data_points: domain_count,
        }
    }

    /// Compute total EFE for a proposed action, combining calibration with base EFE
    ///
    /// This is the main integration point between MAGI Loop and Active Inference.
    ///
    /// # Arguments
    /// * `action` - The proposed action
    /// * `base_efe` - The base EFE from the Active Inference router (without calibration)
    /// * `weights` - Weights for pragmatic, epistemic, and novelty components
    ///
    /// # Returns
    /// A calibration-adjusted EFE value (lower is better for action selection)
    pub fn compute_calibrated_efe(
        &self,
        action: &WorldActionContext,
        base_efe: f64,
        weights: &EfeWeights,
    ) -> CalibratedEfe {
        let contribution = self.compute_efe_contribution(action);

        // Combine base EFE with calibration contribution
        let calibrated = base_efe - (weights.pragmatic * contribution.pragmatic)
            + (weights.epistemic * contribution.epistemic)
            - (weights.novelty * contribution.novelty);

        // Apply constraint gate check
        let execution_mode = self.peek_execution_mode(action);
        let gate_penalty = match execution_mode {
            ExecutionMode::Autonomous => 0.0,
            ExecutionMode::DryRun { .. } => 0.5, // Mild penalty for dry-run
            ExecutionMode::Supervised { .. } => 1.0, // Larger penalty for supervision
        };

        CalibratedEfe {
            base_efe,
            calibration_contribution: contribution,
            combined_efe: calibrated + gate_penalty,
            execution_mode,
            recommended: calibrated + gate_penalty < 2.0, // Threshold for recommendation
        }
    }

    /// Adjust an action's confidence based on domain calibration
    ///
    /// This should be called when making predictions about actions.
    pub fn adjust_action_confidence(
        &self,
        action: &WorldActionContext,
        raw_confidence: f64,
    ) -> f64 {
        let domain = PredictionDomain::from_action_type(&action.action_type);
        self.calibration.adjust_confidence(domain, raw_confidence)
    }

    /// Get the epistemic horizon for a domain (how much we can learn)
    ///
    /// Returns a value 0-1 indicating learning potential:
    /// - 0: We've learned all we can (low ECE, high sample count)
    /// - 1: High learning potential (high ECE or low sample count)
    pub fn epistemic_horizon(&self, domain: PredictionDomain) -> f64 {
        match self.calibration.domain_calibration(domain) {
            Some(cal) => {
                let data_factor = 1.0 - (cal.prediction_count as f64 / 100.0).min(1.0);
                let ece_factor = cal.ece.min(0.5) * 2.0;
                (data_factor + ece_factor) / 2.0
            }
            None => 1.0, // Unknown domain has maximum learning potential
        }
    }

    /// Should we explore this domain based on MAGI Loop state?
    ///
    /// Uses active inference principles:
    /// - High epistemic horizon → explore
    /// - Good calibration → exploit
    pub fn should_explore_domain(
        &self,
        domain: PredictionDomain,
        exploration_threshold: f64,
    ) -> bool {
        self.epistemic_horizon(domain) > exploration_threshold
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EFE INTEGRATION TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// EFE contribution from world calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfeContribution {
    /// Pragmatic value (expected goal achievement based on domain accuracy)
    pub pragmatic: f64,
    /// Epistemic value (uncertainty about predictions in this domain)
    pub epistemic: f64,
    /// Novelty bonus/penalty (for under/over-explored domains)
    pub novelty: f64,
    /// The prediction domain
    pub domain: PredictionDomain,
    /// Current calibration quality
    pub calibration_quality: CalibrationQuality,
    /// Risk tier of the action
    pub action_risk: RiskTier,
    /// Number of data points in this domain
    pub data_points: usize,
}

impl EfeContribution {
    /// Get the total contribution (for logging/debugging)
    pub fn total(&self, weights: &EfeWeights) -> f64 {
        -(weights.pragmatic * self.pragmatic) + (weights.epistemic * self.epistemic)
            - (weights.novelty * self.novelty)
    }
}

/// Weights for EFE components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfeWeights {
    /// Weight for pragmatic value (goal achievement)
    pub pragmatic: f64,
    /// Weight for epistemic value (uncertainty reduction)
    pub epistemic: f64,
    /// Weight for novelty (exploration bonus)
    pub novelty: f64,
}

impl Default for EfeWeights {
    fn default() -> Self {
        Self {
            pragmatic: 1.0,
            epistemic: 0.5,
            novelty: 0.1,
        }
    }
}

impl EfeWeights {
    /// Create exploration-focused weights
    pub fn exploration() -> Self {
        Self {
            pragmatic: 0.5,
            epistemic: 1.5,
            novelty: 0.3,
        }
    }

    /// Create exploitation-focused weights
    pub fn exploitation() -> Self {
        Self {
            pragmatic: 1.5,
            epistemic: 0.2,
            novelty: 0.0,
        }
    }
}

/// Result of calibration-adjusted EFE computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedEfe {
    /// Original base EFE from Active Inference
    pub base_efe: f64,
    /// Contribution from world calibration
    pub calibration_contribution: EfeContribution,
    /// Final combined EFE
    pub combined_efe: f64,
    /// Execution mode from constraint gate
    pub execution_mode: ExecutionMode,
    /// Is this action recommended?
    pub recommended: bool,
}

impl CalibratedEfe {
    /// Is this EFE value good (low)?
    pub fn is_good(&self, threshold: f64) -> bool {
        self.combined_efe < threshold
    }

    /// Get a human-readable explanation
    pub fn explain(&self) -> String {
        let mut s = format!("EFE: {:.3}\n", self.combined_efe);
        s.push_str(&format!("  Base: {:.3}\n", self.base_efe));
        s.push_str(&format!(
            "  Calibration: pragmatic={:.3}, epistemic={:.3}, novelty={:.3}\n",
            self.calibration_contribution.pragmatic,
            self.calibration_contribution.epistemic,
            self.calibration_contribution.novelty
        ));
        s.push_str(&format!("  Execution Mode: {:?}\n", self.execution_mode));
        s.push_str(&format!(
            "  Domain: {:?} ({} data points)\n",
            self.calibration_contribution.domain, self.calibration_contribution.data_points
        ));
        s.push_str(&format!("  Recommended: {}\n", self.recommended));
        s
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAFE UPDATE PROTOCOL (Phase 6)
// ═══════════════════════════════════════════════════════════════════════════════

/// Snapshot of system state before an update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSnapshot {
    /// Timestamp of snapshot
    #[serde(skip, default = "instant_now")]
    pub timestamp: Instant,
    /// Calibration summary at snapshot time
    pub calibration: CalibrationSummary,
    /// MAGI loop state at snapshot time
    pub loop_state: MagiLoopState,
    /// Number of pending predictions
    pub pending_count: usize,
    /// Custom metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl SystemSnapshot {
    /// Create a new snapshot
    pub fn new(
        calibration: CalibrationSummary,
        loop_state: MagiLoopState,
        pending_count: usize,
    ) -> Self {
        Self {
            timestamp: Instant::now(),
            calibration,
            loop_state,
            pending_count,
            metadata: std::collections::HashMap::new(),
        }
    }
}

/// Type of model update being applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelUpdate {
    /// Update calibration adjustment factors
    CalibrationAdjustment {
        domain: PredictionDomain,
        old_adjustment: f64,
        new_adjustment: f64,
    },
    /// Update constraint gate thresholds
    GateThreshold {
        old_threshold: RiskTier,
        new_threshold: RiskTier,
    },
    /// Update self-model capability estimates
    CapabilityUpdate {
        domain: CapabilityDomain,
        old_capability: f64,
        new_capability: f64,
    },
    /// Update confidence adjustment parameters
    ConfidenceAdjustment { old_factor: f64, new_factor: f64 },
    /// Custom update type
    Custom { name: String, description: String },
}

impl ModelUpdate {
    /// Get a description of the update
    pub fn description(&self) -> String {
        match self {
            Self::CalibrationAdjustment {
                domain,
                old_adjustment,
                new_adjustment,
            } => {
                format!(
                    "Calibration {:?}: {:.3} -> {:.3}",
                    domain, old_adjustment, new_adjustment
                )
            }
            Self::GateThreshold {
                old_threshold,
                new_threshold,
            } => {
                format!("Gate threshold: {:?} -> {:?}", old_threshold, new_threshold)
            }
            Self::CapabilityUpdate {
                domain,
                old_capability,
                new_capability,
            } => {
                format!(
                    "Capability {:?}: {:.3} -> {:.3}",
                    domain, old_capability, new_capability
                )
            }
            Self::ConfidenceAdjustment {
                old_factor,
                new_factor,
            } => {
                format!(
                    "Confidence adjustment: {:.3} -> {:.3}",
                    old_factor, new_factor
                )
            }
            Self::Custom { name, description } => {
                format!("{}: {}", name, description)
            }
        }
    }
}

/// Condition that triggers rollback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackCondition {
    /// Accuracy drops by more than threshold
    AccuracyDrop { threshold: f64 },
    /// Calibration error increases by more than threshold
    CalibrationWorse { threshold: f64 },
    /// Core constraint violated
    ConstraintViolation { constraint: String },
    /// Consecutive failures exceed count
    ConsecutiveFailures { count: usize },
    /// Brier score exceeds threshold
    BrierScoreExceeds { threshold: f64 },
    /// ECE exceeds threshold
    EceExceeds { threshold: f64 },
    /// Time-based rollback (if no improvement within duration)
    NoImprovementWithin { duration_secs: u64 },
    /// Custom condition
    Custom {
        name: String,
        check: String, // Description of what to check
    },
}

impl RollbackCondition {
    /// Check if condition is triggered
    pub fn is_triggered(&self, baseline: &SystemSnapshot, current: &SystemSnapshot) -> bool {
        match self {
            Self::AccuracyDrop { threshold } => {
                let drop =
                    baseline.calibration.global_accuracy - current.calibration.global_accuracy;
                drop > *threshold
            }
            Self::CalibrationWorse { threshold } => {
                let increase = current.calibration.global_ece - baseline.calibration.global_ece;
                increase > *threshold
            }
            Self::ConstraintViolation { .. } => {
                // Would need external constraint checking
                false
            }
            Self::ConsecutiveFailures { count } => {
                // Check recent attributions
                current.loop_state.attributions_generated
                    > baseline.loop_state.attributions_generated + count
            }
            Self::BrierScoreExceeds { threshold } => current.calibration.global_brier > *threshold,
            Self::EceExceeds { threshold } => current.calibration.global_ece > *threshold,
            Self::NoImprovementWithin { .. } => {
                // Time-based check would need external timer
                false
            }
            Self::Custom { .. } => {
                // Custom conditions need external evaluation
                false
            }
        }
    }
}

/// Safe update with rollback capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeUpdate {
    /// ID of this update
    pub id: String,
    /// Snapshot before update
    pub baseline: SystemSnapshot,
    /// The update being applied
    pub update: ModelUpdate,
    /// Conditions that trigger rollback
    pub rollback_triggers: Vec<RollbackCondition>,
    /// When the update was applied
    #[serde(skip, default = "instant_now")]
    pub applied_at: Instant,
    /// Current status
    pub status: UpdateStatus,
}

/// Status of a safe update
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateStatus {
    /// Update is pending application
    Pending,
    /// Update is active and being monitored
    Active,
    /// Update succeeded and was committed
    Committed,
    /// Update failed and was rolled back
    RolledBack,
    /// Update was manually cancelled
    Cancelled,
}

impl SafeUpdate {
    /// Create a new safe update
    pub fn new(
        baseline: SystemSnapshot,
        update: ModelUpdate,
        rollback_triggers: Vec<RollbackCondition>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            baseline,
            update,
            rollback_triggers,
            applied_at: Instant::now(),
            status: UpdateStatus::Pending,
        }
    }

    /// Check if any rollback condition is triggered
    pub fn should_rollback(&self, current: &SystemSnapshot) -> Option<&RollbackCondition> {
        self.rollback_triggers
            .iter()
            .find(|condition| condition.is_triggered(&self.baseline, current))
    }

    /// Mark update as active
    pub fn activate(&mut self) {
        self.status = UpdateStatus::Active;
        self.applied_at = Instant::now();
    }

    /// Mark update as committed (successful)
    pub fn commit(&mut self) {
        self.status = UpdateStatus::Committed;
    }

    /// Mark update as rolled back
    pub fn rollback(&mut self) {
        self.status = UpdateStatus::RolledBack;
    }

    /// Check if update is still active
    pub fn is_active(&self) -> bool {
        self.status == UpdateStatus::Active
    }
}

/// Manager for safe updates with rollback tracking
pub struct SafeUpdateManager {
    /// Active updates being monitored
    active_updates: Vec<SafeUpdate>,
    /// History of completed updates
    history: VecDeque<SafeUpdate>,
    /// Maximum history size
    max_history: usize,
    /// Default rollback conditions
    default_conditions: Vec<RollbackCondition>,
}

impl SafeUpdateManager {
    /// Create a new update manager
    pub fn new() -> Self {
        Self {
            active_updates: Vec::new(),
            history: VecDeque::new(),
            max_history: 100,
            default_conditions: vec![
                RollbackCondition::AccuracyDrop { threshold: 0.10 },
                RollbackCondition::CalibrationWorse { threshold: 0.05 },
                RollbackCondition::BrierScoreExceeds { threshold: 0.35 },
                RollbackCondition::ConsecutiveFailures { count: 5 },
            ],
        }
    }

    /// Create an update with default rollback conditions
    pub fn create_update(&self, baseline: SystemSnapshot, update: ModelUpdate) -> SafeUpdate {
        SafeUpdate::new(baseline, update, self.default_conditions.clone())
    }

    /// Create an update with custom rollback conditions
    pub fn create_update_with_conditions(
        &self,
        baseline: SystemSnapshot,
        update: ModelUpdate,
        conditions: Vec<RollbackCondition>,
    ) -> SafeUpdate {
        SafeUpdate::new(baseline, update, conditions)
    }

    /// Register and activate an update
    pub fn apply_update(&mut self, mut update: SafeUpdate) {
        update.activate();
        self.active_updates.push(update);
    }

    /// Check all active updates against current state
    ///
    /// Returns list of updates that should be rolled back
    pub fn check_updates(&self, current: &SystemSnapshot) -> Vec<&SafeUpdate> {
        self.active_updates
            .iter()
            .filter(|update| update.should_rollback(current).is_some())
            .collect()
    }

    /// Commit an update (mark as successful)
    pub fn commit_update(&mut self, update_id: &str) -> bool {
        if let Some(pos) = self.active_updates.iter().position(|u| u.id == update_id) {
            let mut update = self.active_updates.remove(pos);
            update.commit();
            self.add_to_history(update);
            true
        } else {
            false
        }
    }

    /// Rollback an update
    pub fn rollback_update(&mut self, update_id: &str) -> Option<SafeUpdate> {
        if let Some(pos) = self.active_updates.iter().position(|u| u.id == update_id) {
            let mut update = self.active_updates.remove(pos);
            update.rollback();
            let returned = update.clone();
            self.add_to_history(update);
            Some(returned)
        } else {
            None
        }
    }

    /// Add update to history
    fn add_to_history(&mut self, update: SafeUpdate) {
        self.history.push_back(update);
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Get active updates
    pub fn active_updates(&self) -> &[SafeUpdate] {
        &self.active_updates
    }

    /// Get update history
    pub fn history(&self) -> &VecDeque<SafeUpdate> {
        &self.history
    }

    /// Get statistics about updates
    pub fn statistics(&self) -> UpdateStatistics {
        let committed = self
            .history
            .iter()
            .filter(|u| u.status == UpdateStatus::Committed)
            .count();
        let rolled_back = self
            .history
            .iter()
            .filter(|u| u.status == UpdateStatus::RolledBack)
            .count();
        let total = committed + rolled_back;

        UpdateStatistics {
            active_count: self.active_updates.len(),
            committed_count: committed,
            rolled_back_count: rolled_back,
            success_rate: if total > 0 {
                committed as f64 / total as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for SafeUpdateManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about safe updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStatistics {
    /// Number of active updates being monitored
    pub active_count: usize,
    /// Number of successfully committed updates
    pub committed_count: usize,
    /// Number of rolled back updates
    pub rolled_back_count: usize,
    /// Success rate (committed / total completed)
    pub success_rate: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_grounded_creation() {
        let model = WorldGroundedSelfModel::with_defaults();
        assert_eq!(model.loop_state().loop_iterations, 0);
        assert_eq!(
            model.loop_state().calibration_quality,
            CalibrationQuality::Insufficient
        );
    }

    #[test]
    fn test_predict_and_resolve() {
        let mut model = WorldGroundedSelfModel::with_defaults();

        // Make a prediction
        let action =
            WorldActionContext::new("test", "Test action").with_risk_tier(RiskTier::Observation);

        let prediction = model.predict("Test will pass", OutcomeCategory::Success, 0.8, action);

        assert!(prediction.is_pending());
        assert_eq!(model.pending_predictions().len(), 1);

        // Resolve the prediction
        let was_correct = model.resolve_prediction(&prediction.id, OutcomeCategory::Success, 1.0);

        assert_eq!(was_correct, Some(true));
        assert_eq!(model.loop_state().predictions_resolved, 1);
    }

    #[test]
    fn test_execution_mode_check() {
        let mut model = WorldGroundedSelfModel::with_defaults();

        // Low risk action
        let action =
            WorldActionContext::new("read", "Read file").with_risk_tier(RiskTier::Observation);

        let decision = model.check_execution_mode(&action);
        // With no calibration history, should be dry-run or supervised
        assert!(!decision.mode.is_autonomous() || model.calibration().total_predictions() > 50);
    }

    #[test]
    fn test_causal_attribution() {
        let mut model = WorldGroundedSelfModel::with_defaults();

        let action = WorldActionContext::new("test", "Test").with_risk_tier(RiskTier::Observation);

        let prediction = model.predict("Will succeed", OutcomeCategory::Success, 0.9, action);

        // Resolve as failure
        model.resolve_prediction(&prediction.id, OutcomeCategory::SafeFailure, 1.0);

        // Should have generated an attribution
        assert!(!model.recent_attributions().is_empty());
    }

    #[test]
    fn test_calibration_quality_levels() {
        assert_eq!(
            CalibrationQuality::from_ece(0.03, 50, 100),
            CalibrationQuality::Excellent
        );
        assert_eq!(
            CalibrationQuality::from_ece(0.08, 50, 100),
            CalibrationQuality::Good
        );
        assert_eq!(
            CalibrationQuality::from_ece(0.15, 50, 100),
            CalibrationQuality::Moderate
        );
        assert_eq!(
            CalibrationQuality::from_ece(0.25, 50, 100),
            CalibrationQuality::Poor
        );
        assert_eq!(
            CalibrationQuality::from_ece(0.05, 50, 10),
            CalibrationQuality::Insufficient
        );
    }
}
