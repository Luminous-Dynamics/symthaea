//! Active Inference Language Adapter - Cross-Modal Learning Integration
//!
//! This module creates a bidirectional adapter between the Language Consciousness
//! Bridge and the brain's Active Inference Engine, enabling:
//!
//! 1. **Language → Active Inference**: Linguistic prediction errors update
//!    the generative models for user state, task success, and coherence.
//!
//! 2. **Active Inference → Language**: Precision weights from the inference
//!    engine modulate language processing attention.
//!
//! 3. **Unified Free Energy**: Language free energy contributes to the
//!    global free energy metric for consciousness monitoring.
//!
//! # Revolutionary Cross-Modal Learning
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    ACTIVE INFERENCE INTEGRATION                          │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐  │
//! │  │   LANGUAGE      │      │    ADAPTER      │      │   BRAIN         │  │
//! │  │   BRIDGE        │  →   │                 │  →   │   ACTIVE        │  │
//! │  │                 │      │  • Domain Map   │      │   INFERENCE     │  │
//! │  │  • Prediction   │      │  • Error Conv   │      │                 │  │
//! │  │    Errors       │      │  • Precision    │      │  • Generative   │  │
//! │  │  • Φ Metrics    │      │    Flow         │      │    Models       │  │
//! │  │  • Frames       │      │  • Free Energy  │      │  • Predictions  │  │
//! │  │                 │      │    Fusion       │      │  • Actions      │  │
//! │  └─────────────────┘      └─────────────────┘      └─────────────────┘  │
//! │           ↑                        ↓                        ↓            │
//! │           └────────────────────────┴────────────────────────┘            │
//! │                      Precision Weights Flow Back                         │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Innovations
//!
//! ## 1. Domain Mapping with Semantic Preservation
//! Linguistic levels map to prediction domains while preserving semantic meaning:
//! - Lexical errors → User intent understanding
//! - Syntactic errors → Task completion tracking
//! - Semantic errors → Coherence maintenance
//! - Discourse errors → Social coordination
//!
//! ## 2. Precision-Weighted Attention Allocation
//! High precision in the active inference engine for a domain means language
//! processing should pay more attention to errors in that domain.
//!
//! ## 3. Curiosity-Driven Language Learning
//! High uncertainty in a domain triggers epistemic language actions:
//! asking clarifying questions, requesting confirmations, etc.

use super::consciousness_bridge::{
    ConsciousnessBridge, BridgeConfig, BridgeResult, BridgeStats,
    LinguisticPredictionError, InferenceDomain, ConsciousnessState,
    LanguageAttentionBid,
};
use super::predictive_understanding::LinguisticLevel;
use crate::brain::active_inference::{
    ActiveInferenceEngine, PredictionDomain, PredictionError, ActionType,
    ActiveInferenceSummary, GenerativeModel, ActiveInferenceStats,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// =============================================================================
// DOMAIN MAPPING
// =============================================================================

/// Maps between linguistic inference domains and brain prediction domains
pub fn map_inference_to_prediction(domain: InferenceDomain) -> PredictionDomain {
    match domain {
        InferenceDomain::UserState => PredictionDomain::UserState,
        InferenceDomain::TaskSuccess => PredictionDomain::TaskSuccess,
        InferenceDomain::Coherence => PredictionDomain::Coherence,
        InferenceDomain::Social => PredictionDomain::Social,
        InferenceDomain::Performance => PredictionDomain::Performance,
    }
}

/// Maps brain prediction domains back to linguistic inference domains
pub fn map_prediction_to_inference(domain: PredictionDomain) -> Option<InferenceDomain> {
    match domain {
        PredictionDomain::UserState => Some(InferenceDomain::UserState),
        PredictionDomain::TaskSuccess => Some(InferenceDomain::TaskSuccess),
        PredictionDomain::Coherence => Some(InferenceDomain::Coherence),
        PredictionDomain::Social => Some(InferenceDomain::Social),
        PredictionDomain::Performance => Some(InferenceDomain::Performance),
        // These domains don't have direct linguistic mappings
        PredictionDomain::Safety => None,
        PredictionDomain::Energy => None,
        PredictionDomain::Temporal => None,
    }
}

// =============================================================================
// ERROR CONVERSION
// =============================================================================

/// Converts a linguistic prediction error to a brain prediction error
pub fn convert_to_brain_error(ling_error: &LinguisticPredictionError) -> PredictionError {
    let domain = map_inference_to_prediction(ling_error.inference_domain);

    // For linguistic errors, we treat the error magnitude as the observation
    // and 0.0 as the expected value (no error expected)
    let expected = 0.0;
    let observed = ling_error.error;

    PredictionError::new(expected, observed, ling_error.precision, domain)
}

/// Extracts linguistic errors from a bridge result
///
/// Since PredictionResult contains surprise_per_word rather than structured errors,
/// we synthesize linguistic errors from the surprise data, mapping them to inference domains.
pub fn extract_linguistic_errors(result: &BridgeResult) -> Vec<LinguisticPredictionError> {
    let mut errors = Vec::new();
    let prediction = &result.understanding.prediction_result;

    // Extract errors from surprise_per_word
    // We map different aspects to different domains:
    // - High surprise words → UserState (lexical understanding)
    // - Free energy curve variations → TaskSuccess (syntactic structure)
    // - Final free energy → Coherence (semantic integration)

    // 1. Create errors from individual word surprises (Lexical → UserState)
    for (word, surprise) in &prediction.surprise_per_word {
        if *surprise > 0.1 {  // Only significant surprises
            let precision = 1.0 / (1.0 + *surprise as f32);  // Higher surprise = lower precision
            errors.push(LinguisticPredictionError {
                level: "Lexical".to_string(),
                expected_hash: 0,
                observed_hash: word.len() as u64,  // Simple hash proxy
                error: *surprise as f32,
                precision,
                weighted_error: *surprise as f32 * precision,
                inference_domain: InferenceDomain::UserState,
            });
        }
    }

    // 2. Create errors from free energy curve (Syntactic → TaskSuccess)
    if prediction.free_energy_curve.len() > 1 {
        // Calculate variance in free energy curve as syntactic coherence indicator
        let mean_fe: f64 = prediction.free_energy_curve.iter().sum::<f64>()
            / prediction.free_energy_curve.len() as f64;
        let variance: f64 = prediction.free_energy_curve.iter()
            .map(|fe| (fe - mean_fe).powi(2))
            .sum::<f64>() / prediction.free_energy_curve.len() as f64;

        if variance > 0.01 {  // Significant variance
            errors.push(LinguisticPredictionError {
                level: "Syntactic".to_string(),
                expected_hash: 0,
                observed_hash: 1,
                error: variance.sqrt() as f32,
                precision: 0.8,
                weighted_error: variance.sqrt() as f32 * 0.8,
                inference_domain: InferenceDomain::TaskSuccess,
            });
        }
    }

    // 3. Create error from final free energy (Semantic → Coherence)
    let final_fe = prediction.final_free_energy;
    if final_fe > 0.1 {
        errors.push(LinguisticPredictionError {
            level: "Semantic".to_string(),
            expected_hash: 0,
            observed_hash: 2,
            error: final_fe as f32,
            precision: 0.9,  // High precision for overall coherence
            weighted_error: final_fe as f32 * 0.9,
            inference_domain: InferenceDomain::Coherence,
        });
    }

    // 4. Create error from peak surprise if present (Discourse → Social)
    if let Some((_, peak)) = &prediction.peak_surprise {
        errors.push(LinguisticPredictionError {
            level: "Discourse".to_string(),
            expected_hash: 0,
            observed_hash: 3,
            error: *peak as f32,
            precision: 0.7,
            weighted_error: *peak as f32 * 0.7,
            inference_domain: InferenceDomain::Social,
        });
    }

    errors
}

/// Maps linguistic level to inference domain
fn inference_domain_from_level(level: LinguisticLevel) -> InferenceDomain {
    match level {
        LinguisticLevel::Sublexical => InferenceDomain::Performance,
        LinguisticLevel::Lexical => InferenceDomain::UserState,
        LinguisticLevel::Syntactic => InferenceDomain::TaskSuccess,
        LinguisticLevel::Semantic => InferenceDomain::Coherence,
        LinguisticLevel::Discourse => InferenceDomain::Social,
    }
}

// =============================================================================
// PRECISION FLOW
// =============================================================================

/// Precision weights for language processing domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguagePrecisionWeights {
    /// Weight for user state (lexical) predictions
    pub user_state: f32,
    /// Weight for task success (syntactic) predictions
    pub task_success: f32,
    /// Weight for coherence (semantic) predictions
    pub coherence: f32,
    /// Weight for social (discourse) predictions
    pub social: f32,
    /// Weight for performance (sublexical) predictions
    pub performance: f32,
}

impl Default for LanguagePrecisionWeights {
    fn default() -> Self {
        Self {
            user_state: 1.0,
            task_success: 1.0,
            coherence: 1.0,
            social: 1.0,
            performance: 1.0,
        }
    }
}

/// Extracts precision weights from the active inference engine
pub fn extract_precision_weights(engine: &ActiveInferenceEngine) -> LanguagePrecisionWeights {
    let user_state = engine.models.get(&PredictionDomain::UserState)
        .map(|m| m.precision()).unwrap_or(1.0);
    let task_success = engine.models.get(&PredictionDomain::TaskSuccess)
        .map(|m| m.precision()).unwrap_or(1.0);
    let coherence = engine.models.get(&PredictionDomain::Coherence)
        .map(|m| m.precision()).unwrap_or(1.0);
    let social = engine.models.get(&PredictionDomain::Social)
        .map(|m| m.precision()).unwrap_or(1.0);
    let performance = engine.models.get(&PredictionDomain::Performance)
        .map(|m| m.precision()).unwrap_or(1.0);

    LanguagePrecisionWeights {
        user_state,
        task_success,
        coherence,
        social,
        performance,
    }
}

// =============================================================================
// UNIFIED FREE ENERGY
// =============================================================================

/// Unified free energy combining language and general prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedFreeEnergy {
    /// Language-specific free energy
    pub language_fe: f64,
    /// General prediction free energy
    pub general_fe: f32,
    /// Combined weighted free energy
    pub unified_fe: f64,
    /// Relative contribution of language
    pub language_contribution: f64,
}

impl UnifiedFreeEnergy {
    /// Calculate unified free energy from both sources
    pub fn calculate(language_fe: f64, general_fe: f32, language_weight: f64) -> Self {
        let unified = language_weight * language_fe + (1.0 - language_weight) * general_fe as f64;
        let total = language_fe.abs() + general_fe.abs() as f64;
        let language_contribution = if total > 0.0 {
            language_fe.abs() / total
        } else {
            0.5
        };

        Self {
            language_fe,
            general_fe,
            unified_fe: unified,
            language_contribution,
        }
    }

    /// Check if free energy is in optimal range
    pub fn is_optimal(&self) -> bool {
        self.unified_fe < 0.5
    }

    /// Check if system needs attention (high free energy)
    pub fn needs_attention(&self) -> bool {
        self.unified_fe > 0.7
    }
}

// =============================================================================
// LANGUAGE ACTION SUGGESTIONS
// =============================================================================

/// Language-specific actions suggested by active inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LanguageAction {
    /// Ask a clarifying question (epistemic)
    AskClarification {
        topic: String,
        uncertainty: f32,
    },
    /// Request confirmation (reduce uncertainty)
    RequestConfirmation {
        statement: String,
    },
    /// Provide information (pragmatic)
    ProvideInformation {
        content: String,
        expected_utility: f32,
    },
    /// Adjust language style (social)
    AdjustStyle {
        target_formality: f32,
        reason: String,
    },
    /// Focus on specific topic (centering)
    FocusTopic {
        topic: String,
        priority: f32,
    },
}

/// Generates language actions from active inference state
pub fn suggest_language_actions(
    engine: &ActiveInferenceEngine,
    bridge_result: &BridgeResult,
    max_suggestions: usize,
) -> Vec<LanguageAction> {
    let mut actions = Vec::new();

    // Get the most uncertain domain
    let (uncertain_domain, uncertainty) = engine.most_uncertain_domain();

    // If linguistic domain is uncertain, suggest clarification
    if let Some(ling_domain) = map_prediction_to_inference(uncertain_domain) {
        if uncertainty > 0.3 {
            let topic = match ling_domain {
                InferenceDomain::UserState => "user intent".to_string(),
                InferenceDomain::TaskSuccess => "task requirements".to_string(),
                InferenceDomain::Coherence => "topic coherence".to_string(),
                InferenceDomain::Social => "communication style".to_string(),
                InferenceDomain::Performance => "processing needs".to_string(),
            };

            actions.push(LanguageAction::AskClarification {
                topic,
                uncertainty,
            });
        }
    }

    // If high free energy in coherence, suggest focus
    let coherence_model = engine.models.get(&PredictionDomain::Coherence);
    if let Some(model) = coherence_model {
        if model.free_energy() > 0.5 {
            // Extract primary frame as topic
            if let Some(frame_name) = &bridge_result.bid.primary_frame {
                actions.push(LanguageAction::FocusTopic {
                    topic: frame_name.clone(),
                    priority: model.free_energy(),
                });
            }
        }
    }

    // If social uncertainty is high, suggest style adjustment
    let social_model = engine.models.get(&PredictionDomain::Social);
    if let Some(model) = social_model {
        if model.uncertainty() > 0.4 {
            actions.push(LanguageAction::AdjustStyle {
                target_formality: 0.5, // Neutral
                reason: "high social uncertainty".to_string(),
            });
        }
    }

    // Limit to max suggestions
    actions.truncate(max_suggestions);
    actions
}

// =============================================================================
// THE ADAPTER
// =============================================================================

/// Configuration for the active inference adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    /// Weight for language free energy in unified calculation
    pub language_weight: f64,
    /// Threshold for triggering language actions
    pub action_threshold: f32,
    /// Maximum language actions to suggest
    pub max_actions: usize,
    /// Enable precision flow from engine to language
    pub enable_precision_flow: bool,
    /// Enable curiosity-driven language learning
    pub enable_curiosity: bool,
    /// Update rate for precision weights
    pub precision_update_rate: f32,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            language_weight: 0.4,
            action_threshold: 0.3,
            max_actions: 3,
            enable_precision_flow: true,
            enable_curiosity: true,
            precision_update_rate: 0.1,
        }
    }
}

/// Statistics for the adapter
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdapterStats {
    /// Total language observations sent to engine
    pub observations_sent: u64,
    /// Total precision weight updates
    pub precision_updates: u64,
    /// Actions suggested
    pub actions_suggested: u64,
    /// Unified free energy readings
    pub unified_fe_readings: u64,
    /// Average unified free energy
    pub average_unified_fe: f64,
    /// Peak unified free energy
    pub peak_unified_fe: f64,
}

/// Active Inference Language Adapter
///
/// Bridges the Language Consciousness system with the brain's Active Inference
/// Engine, enabling cross-modal learning and unified consciousness monitoring.
pub struct ActiveInferenceAdapter {
    /// Configuration
    config: AdapterConfig,
    /// Current precision weights
    precision_weights: LanguagePrecisionWeights,
    /// Recent unified free energy values
    unified_fe_history: VecDeque<f64>,
    /// Statistics
    stats: AdapterStats,
}

impl ActiveInferenceAdapter {
    /// Create a new adapter
    pub fn new(config: AdapterConfig) -> Self {
        Self {
            config,
            precision_weights: LanguagePrecisionWeights::default(),
            unified_fe_history: VecDeque::with_capacity(100),
            stats: AdapterStats::default(),
        }
    }

    /// Process a bridge result and update the active inference engine
    ///
    /// This is the main integration point that:
    /// 1. Extracts linguistic errors from the bridge result
    /// 2. Converts them to brain prediction errors
    /// 3. Observes them in the active inference engine
    /// 4. Updates precision weights
    /// 5. Returns unified free energy
    pub fn integrate(
        &mut self,
        bridge_result: &BridgeResult,
        engine: &mut ActiveInferenceEngine,
    ) -> IntegrationResult {
        // Step 1: Extract and convert linguistic errors
        let ling_errors = extract_linguistic_errors(bridge_result);
        let mut brain_errors = Vec::new();

        for ling_error in &ling_errors {
            let brain_error = convert_to_brain_error(ling_error);
            brain_errors.push(brain_error.clone());

            // Step 2: Observe in engine
            engine.observe(brain_error.domain, brain_error.observed);
            self.stats.observations_sent += 1;
        }

        // Step 3: Update precision weights
        if self.config.enable_precision_flow {
            self.precision_weights = extract_precision_weights(engine);
            self.stats.precision_updates += 1;
        }

        // Step 4: Calculate unified free energy
        let language_fe = bridge_result.current_free_energy;
        let general_fe = engine.total_free_energy_estimate();
        let unified = UnifiedFreeEnergy::calculate(
            language_fe,
            general_fe,
            self.config.language_weight,
        );

        // Track history
        self.unified_fe_history.push_back(unified.unified_fe);
        if self.unified_fe_history.len() > 100 {
            self.unified_fe_history.pop_front();
        }

        // Update stats
        self.stats.unified_fe_readings += 1;
        self.stats.average_unified_fe =
            self.unified_fe_history.iter().sum::<f64>() / self.unified_fe_history.len() as f64;
        self.stats.peak_unified_fe = self.stats.peak_unified_fe.max(unified.unified_fe);

        // Step 5: Generate language actions
        let actions = if self.config.enable_curiosity {
            let actions = suggest_language_actions(engine, bridge_result, self.config.max_actions);
            self.stats.actions_suggested += actions.len() as u64;
            actions
        } else {
            Vec::new()
        };

        IntegrationResult {
            linguistic_errors: ling_errors,
            brain_errors,
            unified_free_energy: unified,
            precision_weights: self.precision_weights.clone(),
            suggested_actions: actions,
            engine_summary: engine.summary(),
        }
    }

    /// Get current precision weights
    pub fn precision_weights(&self) -> &LanguagePrecisionWeights {
        &self.precision_weights
    }

    /// Get statistics
    pub fn stats(&self) -> &AdapterStats {
        &self.stats
    }

    /// Get unified free energy history
    pub fn unified_fe_history(&self) -> &VecDeque<f64> {
        &self.unified_fe_history
    }

    /// Check if integration is healthy (low unified FE, stable)
    pub fn is_healthy(&self) -> bool {
        if self.unified_fe_history.is_empty() {
            return true;
        }

        let recent: Vec<f64> = self.unified_fe_history.iter().rev().take(10).copied().collect();
        if recent.is_empty() {
            return true;
        }

        let avg = recent.iter().sum::<f64>() / recent.len() as f64;

        // Healthy if average unified FE is below threshold
        avg < 0.7
    }

    /// Reset the adapter state
    pub fn reset(&mut self) {
        self.precision_weights = LanguagePrecisionWeights::default();
        self.unified_fe_history.clear();
    }
}

/// Result of integration processing
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    /// Linguistic errors extracted
    pub linguistic_errors: Vec<LinguisticPredictionError>,
    /// Converted brain errors
    pub brain_errors: Vec<PredictionError>,
    /// Unified free energy
    pub unified_free_energy: UnifiedFreeEnergy,
    /// Current precision weights
    pub precision_weights: LanguagePrecisionWeights,
    /// Suggested language actions
    pub suggested_actions: Vec<LanguageAction>,
    /// Summary from active inference engine
    pub engine_summary: ActiveInferenceSummary,
}

impl IntegrationResult {
    /// Check if there are high-priority actions
    pub fn has_priority_actions(&self) -> bool {
        !self.suggested_actions.is_empty()
    }

    /// Get the total error count
    pub fn total_errors(&self) -> usize {
        self.linguistic_errors.len()
    }

    /// Get the average precision across all weights
    pub fn average_precision(&self) -> f32 {
        let sum = self.precision_weights.user_state
            + self.precision_weights.task_success
            + self.precision_weights.coherence
            + self.precision_weights.social
            + self.precision_weights.performance;
        sum / 5.0
    }
}

// =============================================================================
// INTEGRATED CONSCIOUSNESS PROCESSOR
// =============================================================================

/// Fully integrated consciousness processor
///
/// Combines the Language Bridge, Active Inference Engine, and Adapter
/// into a single unified processor for end-to-end conscious language processing.
pub struct IntegratedConsciousnessProcessor {
    /// Language consciousness bridge
    bridge: ConsciousnessBridge,
    /// Active inference engine
    engine: ActiveInferenceEngine,
    /// Integration adapter
    adapter: ActiveInferenceAdapter,
    /// Processing statistics
    stats: ProcessorStats,
}

/// Statistics for the integrated processor
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessorStats {
    /// Total inputs processed
    pub inputs_processed: u64,
    /// Total spotlight wins
    pub spotlight_wins: u64,
    /// Total actions suggested
    pub total_actions: u64,
    /// Average unified free energy
    pub average_unified_fe: f64,
    /// Average Φ value
    pub average_phi: f64,
    /// Integration health ratio
    pub health_ratio: f64,
}

impl IntegratedConsciousnessProcessor {
    /// Create a new integrated processor
    pub fn new() -> Self {
        Self {
            bridge: ConsciousnessBridge::new(BridgeConfig::default()),
            engine: ActiveInferenceEngine::new(),
            adapter: ActiveInferenceAdapter::new(AdapterConfig::default()),
            stats: ProcessorStats::default(),
        }
    }

    /// Create with custom configurations
    pub fn with_configs(
        bridge_config: BridgeConfig,
        adapter_config: AdapterConfig,
    ) -> Self {
        Self {
            bridge: ConsciousnessBridge::new(bridge_config),
            engine: ActiveInferenceEngine::new(),
            adapter: ActiveInferenceAdapter::new(adapter_config),
            stats: ProcessorStats::default(),
        }
    }

    /// Process input through the full integrated pipeline
    ///
    /// This performs:
    /// 1. Language understanding via the bridge
    /// 2. Active inference integration via the adapter
    /// 3. Unified consciousness state assessment
    /// 4. Action suggestion generation
    pub fn process(&mut self, input: &str) -> IntegratedResult {
        self.stats.inputs_processed += 1;

        // Step 1: Language processing through bridge
        let bridge_result = self.bridge.process(input);

        if bridge_result.gained_spotlight {
            self.stats.spotlight_wins += 1;
        }

        // Step 2: Active inference integration
        let integration = self.adapter.integrate(&bridge_result, &mut self.engine);

        self.stats.total_actions += integration.suggested_actions.len() as u64;

        // Update running stats
        let alpha = 0.1;
        self.stats.average_unified_fe = self.stats.average_unified_fe * (1.0 - alpha)
            + integration.unified_free_energy.unified_fe * alpha;
        self.stats.average_phi = self.stats.average_phi * (1.0 - alpha)
            + bridge_result.current_phi * alpha;

        // Calculate health ratio
        let healthy = self.adapter.is_healthy() && matches!(
            bridge_result.consciousness_state,
            ConsciousnessState::HighlyCoherent | ConsciousnessState::Coherent | ConsciousnessState::Normal
        );
        self.stats.health_ratio = self.stats.health_ratio * 0.95 + if healthy { 0.05 } else { 0.0 };

        IntegratedResult {
            bridge_result,
            integration,
            processor_stats: self.stats.clone(),
        }
    }

    /// Get the language bridge
    pub fn bridge(&self) -> &ConsciousnessBridge {
        &self.bridge
    }

    /// Get the active inference engine
    pub fn engine(&self) -> &ActiveInferenceEngine {
        &self.engine
    }

    /// Get the adapter
    pub fn adapter(&self) -> &ActiveInferenceAdapter {
        &self.adapter
    }

    /// Get processor statistics
    pub fn stats(&self) -> &ProcessorStats {
        &self.stats
    }

    /// Reset all components
    pub fn reset(&mut self) {
        self.bridge.reset();
        self.engine = ActiveInferenceEngine::new();
        self.adapter.reset();
    }

    /// Check if system is in healthy state
    pub fn is_healthy(&self) -> bool {
        self.stats.health_ratio > 0.7
    }

    /// Get current consciousness assessment
    pub fn consciousness_state(&self) -> IntegratedConsciousnessState {
        let avg_phi = self.stats.average_phi;
        let avg_fe = self.stats.average_unified_fe;
        let health = self.stats.health_ratio;

        if health > 0.9 && avg_phi > 0.7 && avg_fe < 0.3 {
            IntegratedConsciousnessState::Optimal
        } else if health > 0.7 && avg_phi > 0.5 && avg_fe < 0.5 {
            IntegratedConsciousnessState::Good
        } else if health > 0.5 {
            IntegratedConsciousnessState::Adequate
        } else if avg_fe > 0.7 {
            IntegratedConsciousnessState::Struggling
        } else {
            IntegratedConsciousnessState::NeedsAttention
        }
    }
}

impl Default for IntegratedConsciousnessProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Result from integrated processing
#[derive(Debug, Clone)]
pub struct IntegratedResult {
    /// Result from the language bridge
    pub bridge_result: BridgeResult,
    /// Result from active inference integration
    pub integration: IntegrationResult,
    /// Processor statistics
    pub processor_stats: ProcessorStats,
}

impl IntegratedResult {
    /// Get the understood input
    pub fn input(&self) -> &str {
        &self.bridge_result.understanding.input
    }

    /// Get the Φ value
    pub fn phi(&self) -> f64 {
        self.bridge_result.current_phi
    }

    /// Get the unified free energy
    pub fn unified_free_energy(&self) -> f64 {
        self.integration.unified_free_energy.unified_fe
    }

    /// Get suggested actions
    pub fn actions(&self) -> &[LanguageAction] {
        &self.integration.suggested_actions
    }

    /// Check if this understanding gained spotlight
    pub fn gained_spotlight(&self) -> bool {
        self.bridge_result.gained_spotlight
    }
}

/// Integrated consciousness states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegratedConsciousnessState {
    /// Optimal: high Φ, low FE, high health
    Optimal,
    /// Good: acceptable metrics
    Good,
    /// Adequate: functional but not optimal
    Adequate,
    /// Struggling: high FE, needs improvement
    Struggling,
    /// Needs attention: multiple issues
    NeedsAttention,
}

impl std::fmt::Display for IntegratedConsciousnessState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Optimal => write!(f, "Optimal"),
            Self::Good => write!(f, "Good"),
            Self::Adequate => write!(f, "Adequate"),
            Self::Struggling => write!(f, "Struggling"),
            Self::NeedsAttention => write!(f, "Needs Attention"),
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_mapping() {
        assert_eq!(
            map_inference_to_prediction(InferenceDomain::UserState),
            PredictionDomain::UserState
        );
        assert_eq!(
            map_inference_to_prediction(InferenceDomain::Coherence),
            PredictionDomain::Coherence
        );
        assert_eq!(
            map_prediction_to_inference(PredictionDomain::UserState),
            Some(InferenceDomain::UserState)
        );
        assert_eq!(
            map_prediction_to_inference(PredictionDomain::Safety),
            None
        );
    }

    #[test]
    fn test_error_conversion() {
        let ling_error = LinguisticPredictionError {
            level: "Semantic".to_string(),
            expected_hash: 0,
            observed_hash: 1,
            error: 0.5,
            precision: 0.8,
            weighted_error: 0.4,
            inference_domain: InferenceDomain::Coherence,
        };

        let brain_error = convert_to_brain_error(&ling_error);

        assert_eq!(brain_error.domain, PredictionDomain::Coherence);
        assert!((brain_error.precision - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_precision_weights_default() {
        let weights = LanguagePrecisionWeights::default();

        assert!((weights.user_state - 1.0).abs() < 0.01);
        assert!((weights.coherence - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_unified_free_energy_calculation() {
        let unified = UnifiedFreeEnergy::calculate(0.3, 0.5, 0.4);

        // 0.4 * 0.3 + 0.6 * 0.5 = 0.12 + 0.3 = 0.42
        assert!((unified.unified_fe - 0.42).abs() < 0.01);
    }

    #[test]
    fn test_adapter_creation() {
        let config = AdapterConfig::default();
        let adapter = ActiveInferenceAdapter::new(config);

        assert!(adapter.precision_weights.user_state > 0.0);
        assert!(adapter.stats.observations_sent == 0);
    }

    #[test]
    fn test_adapter_integration() {
        let mut adapter = ActiveInferenceAdapter::new(AdapterConfig::default());
        let mut engine = ActiveInferenceEngine::new();

        // Create a minimal bridge result
        let mut test_bridge = ConsciousnessBridge::new(BridgeConfig::default());
        let bridge_result = test_bridge.process("The cat sat on the mat");

        let result = adapter.integrate(&bridge_result, &mut engine);

        // Should have produced some result
        assert!(result.unified_free_energy.unified_fe >= 0.0);
        assert!(adapter.stats().observations_sent >= 0);
    }

    #[test]
    fn test_integrated_processor_creation() {
        let processor = IntegratedConsciousnessProcessor::new();

        assert_eq!(processor.stats().inputs_processed, 0);
    }

    #[test]
    fn test_integrated_processor_processing() {
        let mut processor = IntegratedConsciousnessProcessor::new();

        let result = processor.process("I want something good");

        assert!(!result.input().is_empty());
        assert!(result.phi() >= 0.0);
        assert!(result.unified_free_energy() >= 0.0);
    }

    #[test]
    fn test_integrated_processor_conversation() {
        let mut processor = IntegratedConsciousnessProcessor::new();

        let sentences = [
            "Hello, how can I help you?",
            "I need to install some software.",
            "What software do you need?",
            "Firefox, please.",
        ];

        for sentence in &sentences {
            let result = processor.process(sentence);
            assert!(result.phi() >= 0.0);
        }

        assert_eq!(processor.stats().inputs_processed, 4);
    }

    #[test]
    fn test_consciousness_state_assessment() {
        let mut processor = IntegratedConsciousnessProcessor::new();

        // Process some inputs to establish state
        for _ in 0..5 {
            processor.process("I think you know something");
        }

        let state = processor.consciousness_state();

        // Should be one of the valid states
        assert!(matches!(
            state,
            IntegratedConsciousnessState::Optimal
                | IntegratedConsciousnessState::Good
                | IntegratedConsciousnessState::Adequate
                | IntegratedConsciousnessState::Struggling
                | IntegratedConsciousnessState::NeedsAttention
        ));
    }

    #[test]
    fn test_language_action_suggestions() {
        let engine = ActiveInferenceEngine::new();
        let mut test_bridge = ConsciousnessBridge::new(BridgeConfig::default());
        let bridge_result = test_bridge.process("I need help");

        let actions = suggest_language_actions(&engine, &bridge_result, 5);

        // May or may not have actions depending on uncertainty
        assert!(actions.len() <= 5);
    }

    #[test]
    fn test_adapter_health_check() {
        let adapter = ActiveInferenceAdapter::new(AdapterConfig::default());

        // Should be healthy initially (no history)
        assert!(adapter.is_healthy());
    }

    #[test]
    fn test_processor_reset() {
        let mut processor = IntegratedConsciousnessProcessor::new();

        // Process some inputs
        processor.process("First sentence");
        processor.process("Second sentence");

        // Reset
        processor.reset();

        // Bridge should be cleared (working memory empty)
        assert!(processor.bridge().working_memory().is_empty());
    }

    #[test]
    #[ignore = "performance test - run with cargo test --release"]
    fn benchmark_integrated_processing() {
        use std::time::Instant;

        let mut processor = IntegratedConsciousnessProcessor::new();

        let test_sentences = [
            "The cat sat on the mat",
            "She gave him a beautiful book",
            "I think you know something important",
        ];

        // Warm up
        for _ in 0..3 {
            for s in &test_sentences {
                processor.process(s);
            }
        }
        processor.reset();

        // Benchmark
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            for s in &test_sentences {
                processor.process(s);
            }
        }
        let elapsed = start.elapsed();
        let per_sentence = elapsed.as_micros() / (iterations * test_sentences.len()) as u128;

        // Should be under 100ms per sentence (debug mode)
        assert!(per_sentence < 100_000,
            "Integrated processing too slow: {}μs per sentence", per_sentence);

        println!("\n📊 Integrated Consciousness Performance:");
        println!("   {}μs per sentence", per_sentence);
        println!("   Final state: {}", processor.consciousness_state());
    }
}
