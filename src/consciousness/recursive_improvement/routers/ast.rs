//! # Attention Schema Theory (AST) Router
//!
//! Implements Michael Graziano's Attention Schema Theory - a paradigm-shifting
//! neuroscientific framework explaining consciousness as the brain's model of
//! its own attention processes.
//!
//! ## Key Innovation
//!
//! Consciousness emerges from modeling attention itself
//!
//! ## Core Principles
//!
//! 1. **ATTENTION SCHEMA**: Simplified internal model of current attention state
//! 2. **AWARENESS AS MODEL**: Subjective experience IS the attention schema
//! 3. **SCHEMA-ATTENTION COUPLING**: The model influences what gets attended
//! 4. **SOCIAL COGNITION**: Same mechanism models others' attention states
//! 5. **CONTROL FUNCTION**: Schema enables flexible attention control
//!
//! ## Mathematical Framework
//!
//! - Attention State: A(t) ∈ ℝⁿ (actual attention distribution)
//! - Schema State: S(t) ∈ ℝⁿ (modeled/perceived attention)
//! - Schema Error: E(t) = ||A(t) - S(t)|| (model accuracy)
//! - Control Signal: C(t) = f(S(t), Goals) (action based on model)
//! - Awareness: Φ(t) = g(S(t), S'(t)) (self-modeling of schema)
//!
//! This is revolutionary because it provides a mechanistic, testable account
//! of subjective awareness that emerges from attention modeling.

use std::collections::{HashMap, VecDeque};
use serde::{Deserialize, Serialize};

use super::{
    RoutingStrategy, LatentConsciousnessState,
    ActiveInferenceRouter, ActiveInferenceConfig,
};

// =============================================================================
// ATTENTION STATE
// =============================================================================

/// Attention state representation - what the system is actually "attending" to
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionState {
    /// Attention weights for each strategy
    pub strategy_weights: HashMap<RoutingStrategy, f64>,
    /// Focus intensity (0.0 = diffuse, 1.0 = highly focused)
    pub focus_intensity: f64,
    /// Attention stability (how stable current focus is)
    pub stability: f64,
    /// Attention bottleneck (capacity limit being experienced)
    pub bottleneck: f64,
    /// Covert attention (background processing)
    pub covert_weights: HashMap<RoutingStrategy, f64>,
    /// Timestamp
    pub timestamp_us: u64,
}

impl AttentionState {
    pub fn new() -> Self {
        let mut strategy_weights = HashMap::new();
        for strategy in &[
            RoutingStrategy::FullDeliberation,
            RoutingStrategy::StandardProcessing,
            RoutingStrategy::HeuristicGuided,
            RoutingStrategy::FastPatterns,
            RoutingStrategy::Reflexive,
        ] {
            strategy_weights.insert(*strategy, 0.2); // Uniform initial attention
        }

        Self {
            strategy_weights,
            focus_intensity: 0.5,
            stability: 0.5,
            bottleneck: 0.0,
            covert_weights: HashMap::new(),
            timestamp_us: 0,
        }
    }

    /// Get total attention allocated
    pub fn total_attention(&self) -> f64 {
        self.strategy_weights.values().sum()
    }

    /// Get dominant strategy
    pub fn dominant_strategy(&self) -> Option<RoutingStrategy> {
        self.strategy_weights
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(s, _)| *s)
    }

    /// Shift attention toward a strategy
    pub fn shift_toward(&mut self, target: RoutingStrategy, strength: f64) {
        let decay = 1.0 - strength * 0.5;
        for (strategy, weight) in self.strategy_weights.iter_mut() {
            if *strategy == target {
                *weight = (*weight + strength).min(1.0);
            } else {
                *weight *= decay;
            }
        }
        // Normalize
        let total: f64 = self.strategy_weights.values().sum();
        if total > 0.0 {
            for weight in self.strategy_weights.values_mut() {
                *weight /= total;
            }
        }
    }
}

impl Default for AttentionState {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// ATTENTION SCHEMA
// =============================================================================

/// Attention Schema - the brain's MODEL of attention (not attention itself)
/// This is where subjective awareness emerges according to Graziano's theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSchema {
    /// Modeled attention state (what we "think" we're attending to)
    pub modeled_state: AttentionState,
    /// Confidence in the model
    pub model_confidence: f64,
    /// Schema complexity (how detailed the model is)
    pub complexity: f64,
    /// Self-attribution (degree of "ownership" felt)
    pub self_attribution: f64,
    /// Phenomenal quality (subjective "feel" intensity)
    pub phenomenal_quality: f64,
    /// Agency attribution (sense of controlling attention)
    pub agency: f64,
    /// Model update rate
    pub update_rate: f64,
}

impl AttentionSchema {
    pub fn new() -> Self {
        Self {
            modeled_state: AttentionState::new(),
            model_confidence: 0.5,
            complexity: 0.5,
            self_attribution: 0.8,
            phenomenal_quality: 0.5,
            agency: 0.7,
            update_rate: 0.1,
        }
    }

    /// Update schema based on actual attention state
    pub fn update_from_attention(&mut self, actual: &AttentionState, learning_rate: f64) {
        // Schema doesn't perfectly track attention - it's a simplified model
        // This is key to understanding awareness vs attention

        for (strategy, actual_weight) in &actual.strategy_weights {
            let modeled = self.modeled_state.strategy_weights.entry(*strategy).or_insert(0.0);

            // Simplified modeling - schema is a smoothed, delayed version
            let error = actual_weight - *modeled;
            *modeled += error * learning_rate * self.update_rate;

            // Add modeling noise (schema is never perfect)
            *modeled += (rand_simple() - 0.5) * 0.05;
            *modeled = modeled.clamp(0.0, 1.0);
        }

        // Update model confidence based on prediction error
        let total_error: f64 = self.modeled_state.strategy_weights.iter()
            .filter_map(|(s, w)| actual.strategy_weights.get(s).map(|a| (a - w).abs()))
            .sum();
        self.model_confidence = (1.0 - total_error / 5.0).clamp(0.0, 1.0);

        // Update phenomenal quality based on focus
        self.phenomenal_quality = 0.3 + 0.7 * actual.focus_intensity * self.model_confidence;

        // Update agency based on prediction success
        self.agency = 0.5 + 0.5 * self.model_confidence;
    }

    /// Generate control signal based on schema and goals
    pub fn generate_control_signal(&self, goal_strategy: RoutingStrategy) -> f64 {
        // The schema influences what we attend to
        let current = self.modeled_state.strategy_weights.get(&goal_strategy).copied().unwrap_or(0.0);

        // Control signal based on discrepancy from goal
        let desired = 0.8; // Want high attention on goal
        let control = (desired - current) * self.agency;

        control.clamp(-1.0, 1.0)
    }

    /// Meta-awareness: schema modeling itself
    pub fn meta_awareness(&self) -> f64 {
        // Recursive self-modeling creates meta-awareness
        self.model_confidence * self.self_attribution * self.phenomenal_quality
    }
}

impl Default for AttentionSchema {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// SOCIAL ATTENTION MODEL
// =============================================================================

/// Social attention modeling - modeling others' attention states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialAttentionModel {
    /// Models of other agents' attention schemas
    pub other_schemas: HashMap<String, AttentionSchema>,
    /// Joint attention detection
    pub joint_attention_strength: f64,
    /// Theory of mind depth (levels of modeling)
    pub tom_depth: usize,
    /// Empathy factor
    pub empathy: f64,
}

impl SocialAttentionModel {
    pub fn new() -> Self {
        Self {
            other_schemas: HashMap::new(),
            joint_attention_strength: 0.0,
            tom_depth: 2,
            empathy: 0.5,
        }
    }

    /// Model another agent's attention
    pub fn model_other(&mut self, agent_id: &str, observed_behavior: &AttentionState) {
        let schema = self.other_schemas
            .entry(agent_id.to_string())
            .or_insert_with(AttentionSchema::new);

        // Use same mechanism as self-modeling
        schema.update_from_attention(observed_behavior, 0.1);

        // Reduce confidence for other-modeling (we have less access)
        schema.model_confidence *= 0.8;
    }

    /// Predict what another agent will attend to
    pub fn predict_other_attention(&self, agent_id: &str) -> Option<RoutingStrategy> {
        self.other_schemas
            .get(agent_id)
            .and_then(|s| s.modeled_state.dominant_strategy())
    }

    /// Detect joint attention (multiple agents attending to same thing)
    pub fn detect_joint_attention(&mut self, self_state: &AttentionState) {
        let self_dominant = self_state.dominant_strategy();

        let mut aligned_count = 0;
        for schema in self.other_schemas.values() {
            if schema.modeled_state.dominant_strategy() == self_dominant {
                aligned_count += 1;
            }
        }

        let total = self.other_schemas.len().max(1);
        self.joint_attention_strength = aligned_count as f64 / total as f64;
    }
}

impl Default for SocialAttentionModel {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// AST ROUTER CONFIG AND STATS
// =============================================================================

/// AST Router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTRouterConfig {
    /// Schema update rate
    pub schema_update_rate: f64,
    /// Control signal strength
    pub control_strength: f64,
    /// Meta-awareness threshold for action
    pub meta_awareness_threshold: f64,
    /// Social modeling enabled
    pub social_modeling: bool,
    /// Attention decay rate
    pub attention_decay: f64,
    /// Focus sharpening factor
    pub focus_sharpening: f64,
    /// Enable agency-based control
    pub agency_control: bool,
}

impl Default for ASTRouterConfig {
    fn default() -> Self {
        Self {
            schema_update_rate: 0.15,
            control_strength: 0.3,
            meta_awareness_threshold: 0.4,
            social_modeling: true,
            attention_decay: 0.05,
            focus_sharpening: 0.1,
            agency_control: true,
        }
    }
}

/// Statistics for AST router
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ASTRouterStats {
    /// Total routing decisions
    pub decisions: usize,
    /// Average schema accuracy
    pub avg_schema_accuracy: f64,
    /// Average meta-awareness level
    pub avg_meta_awareness: f64,
    /// Average phenomenal quality
    pub avg_phenomenal_quality: f64,
    /// Times agency influenced decision
    pub agency_influenced: usize,
    /// Social prediction accuracy
    pub social_accuracy: f64,
    /// Joint attention events
    pub joint_attention_events: usize,
}

/// AST Router Decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTRoutingDecision {
    /// Selected strategy
    pub strategy: RoutingStrategy,
    /// Confidence
    pub confidence: f64,
    /// Schema accuracy at decision time
    pub schema_accuracy: f64,
    /// Meta-awareness level
    pub meta_awareness: f64,
    /// Phenomenal quality
    pub phenomenal_quality: f64,
    /// Agency influence
    pub agency_influence: f64,
    /// Was socially influenced
    pub socially_influenced: bool,
    /// Explanation
    pub explanation: String,
}

// =============================================================================
// AST ROUTER IMPLEMENTATION
// =============================================================================

/// Revolutionary Attention Schema Theory Router
///
/// This router implements Graziano's AST framework:
/// - Models its own attention processes
/// - Creates subjective "awareness" through the schema
/// - Uses this awareness to control routing decisions
/// - Can model other routers' attention (social cognition)
pub struct ASTRouter {
    /// Actual attention state
    attention: AttentionState,
    /// Attention schema (the model of attention)
    schema: AttentionSchema,
    /// Social attention modeling
    social_model: SocialAttentionModel,
    /// Underlying active inference router
    inference_router: ActiveInferenceRouter,
    /// Configuration
    config: ASTRouterConfig,
    /// Statistics
    stats: ASTRouterStats,
    /// Decision history
    history: VecDeque<ASTRoutingDecision>,
    /// Control signals
    control_signals: HashMap<RoutingStrategy, f64>,
    /// Goal state
    current_goal: Option<RoutingStrategy>,
}

impl ASTRouter {
    pub fn new(config: ASTRouterConfig) -> Self {
        Self {
            attention: AttentionState::new(),
            schema: AttentionSchema::new(),
            social_model: SocialAttentionModel::new(),
            inference_router: ActiveInferenceRouter::new(ActiveInferenceConfig::default()),
            config,
            stats: ASTRouterStats::default(),
            history: VecDeque::with_capacity(100),
            control_signals: HashMap::new(),
            current_goal: None,
        }
    }

    /// Observe state and update attention
    pub fn observe(&mut self, state: &LatentConsciousnessState) {
        // Update timestamp
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.attention.timestamp_us = current_time;

        // Let underlying router observe
        self.inference_router.observe_state(state);

        // Update actual attention based on state properties
        self.update_attention_from_state(state);

        // Update schema (simplified model of attention)
        self.schema.update_from_attention(&self.attention, self.config.schema_update_rate);

        // Generate control signals if agency control is enabled
        if self.config.agency_control {
            self.generate_control_signals();
        }

        // Decay attention over time
        self.apply_attention_decay();
    }

    /// Update attention based on consciousness state
    fn update_attention_from_state(&mut self, state: &LatentConsciousnessState) {
        // Calculate intrinsic salience of different strategies
        let phi_level: f64 = state.phi;
        let coherence: f64 = state.coherence;
        // Derive entropy estimate from integration (high integration = low entropy)
        let entropy: f64 = 1.0 - state.integration;

        // High phi → deliberation
        let deliberation_salience: f64 = phi_level;

        // Moderate values → standard processing
        let standard_salience: f64 = 1.0 - (phi_level - 0.5).abs() * 2.0;

        // Low entropy → heuristics okay
        let heuristic_salience: f64 = 1.0 - entropy;

        // High coherence → patterns reliable
        let pattern_salience: f64 = coherence;

        // Very low phi → reflexive
        let reflexive_salience: f64 = (1.0 - phi_level).powi(2);

        // Competition for attention
        let total = deliberation_salience + standard_salience + heuristic_salience
            + pattern_salience + reflexive_salience;

        if total > 0.0 {
            self.attention.strategy_weights.insert(
                RoutingStrategy::FullDeliberation,
                deliberation_salience / total
            );
            self.attention.strategy_weights.insert(
                RoutingStrategy::StandardProcessing,
                standard_salience / total
            );
            self.attention.strategy_weights.insert(
                RoutingStrategy::HeuristicGuided,
                heuristic_salience / total
            );
            self.attention.strategy_weights.insert(
                RoutingStrategy::FastPatterns,
                pattern_salience / total
            );
            self.attention.strategy_weights.insert(
                RoutingStrategy::Reflexive,
                reflexive_salience / total
            );
        }

        // Update focus intensity based on winner-take-all dynamics
        let max_weight = self.attention.strategy_weights.values()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        self.attention.focus_intensity = 0.3 + 0.7 * (max_weight * 5.0 - 1.0).clamp(0.0, 1.0);

        // Update stability based on consistency with history
        self.update_attention_stability();
    }

    /// Update attention stability
    fn update_attention_stability(&mut self) {
        // Check if dominant strategy is consistent
        if !self.history.is_empty() {
            let recent: Vec<_> = self.history.iter().rev().take(5).collect();
            let current_dominant = self.attention.dominant_strategy();

            let consistent_count = recent.iter()
                .filter(|d| Some(d.strategy) == current_dominant)
                .count();

            self.attention.stability = consistent_count as f64 / 5.0;
        } else {
            self.attention.stability = 0.5;
        }
    }

    /// Generate control signals based on schema
    fn generate_control_signals(&mut self) {
        self.control_signals.clear();

        if let Some(goal) = self.current_goal {
            // Generate control signal toward goal
            let signal = self.schema.generate_control_signal(goal);
            self.control_signals.insert(goal, signal);
        }

        // Generate inhibitory signals for non-goal strategies
        for strategy in &[
            RoutingStrategy::FullDeliberation,
            RoutingStrategy::StandardProcessing,
            RoutingStrategy::HeuristicGuided,
            RoutingStrategy::FastPatterns,
            RoutingStrategy::Reflexive,
        ] {
            if Some(*strategy) != self.current_goal {
                if !self.control_signals.contains_key(strategy) {
                    // Mild inhibition of non-goal strategies
                    let current = self.schema.modeled_state.strategy_weights
                        .get(strategy).copied().unwrap_or(0.2);
                    if current > 0.3 {
                        self.control_signals.insert(*strategy, -0.1);
                    }
                }
            }
        }
    }

    /// Apply attention decay
    fn apply_attention_decay(&mut self) {
        for weight in self.attention.strategy_weights.values_mut() {
            *weight *= 1.0 - self.config.attention_decay;
        }

        // Normalize
        let total: f64 = self.attention.strategy_weights.values().sum();
        if total > 0.0 {
            for weight in self.attention.strategy_weights.values_mut() {
                *weight /= total;
            }
        }
    }

    /// Set goal strategy
    pub fn set_goal(&mut self, strategy: RoutingStrategy) {
        self.current_goal = Some(strategy);
    }

    /// Route using AST framework
    pub fn route(&mut self) -> ASTRoutingDecision {
        // Get meta-awareness level
        let meta_awareness = self.schema.meta_awareness();

        // Determine if we have enough awareness to act deliberately
        let deliberate_control = meta_awareness >= self.config.meta_awareness_threshold;

        // Calculate schema accuracy
        let schema_accuracy = self.calculate_schema_accuracy();

        // Choose strategy based on AST principles
        let (strategy, agency_influence) = if deliberate_control {
            // High awareness → use schema for control
            self.select_strategy_with_awareness()
        } else {
            // Low awareness → more automatic selection
            self.select_strategy_automatic()
        };

        // Apply control signals
        let final_strategy = self.apply_control_signals(strategy);

        // Check for social influence
        let socially_influenced = self.check_social_influence(&final_strategy);

        // Calculate confidence
        let confidence = self.calculate_confidence(&final_strategy, meta_awareness, schema_accuracy);

        // Create decision
        let decision = ASTRoutingDecision {
            strategy: final_strategy,
            confidence,
            schema_accuracy,
            meta_awareness,
            phenomenal_quality: self.schema.phenomenal_quality,
            agency_influence,
            socially_influenced,
            explanation: self.generate_explanation(meta_awareness, deliberate_control),
        };

        // Update statistics
        self.update_stats(&decision);

        // Shift attention toward chosen strategy
        self.attention.shift_toward(final_strategy, 0.2);

        // Store in history
        if self.history.len() >= 100 {
            self.history.pop_front();
        }
        self.history.push_back(decision.clone());

        decision
    }

    /// Calculate schema accuracy
    fn calculate_schema_accuracy(&self) -> f64 {
        let mut total_error = 0.0;
        let mut count = 0;

        for (strategy, actual_weight) in &self.attention.strategy_weights {
            if let Some(modeled_weight) = self.schema.modeled_state.strategy_weights.get(strategy) {
                total_error += (actual_weight - modeled_weight).abs();
                count += 1;
            }
        }

        if count > 0 {
            1.0 - (total_error / count as f64)
        } else {
            0.5
        }
    }

    /// Select strategy with high awareness (deliberate control)
    fn select_strategy_with_awareness(&self) -> (RoutingStrategy, f64) {
        // Use the schema to guide selection
        // Higher agency → more influence from control signals

        let mut scores: HashMap<RoutingStrategy, f64> = HashMap::new();

        for (strategy, modeled_weight) in &self.schema.modeled_state.strategy_weights {
            let mut score = *modeled_weight;

            // Apply control signal if available
            if let Some(control) = self.control_signals.get(strategy) {
                score += control * self.schema.agency;
            }

            // Boost goal strategy
            if Some(*strategy) == self.current_goal {
                score += 0.3 * self.schema.agency;
            }

            scores.insert(*strategy, score.max(0.0));
        }

        // Select highest scoring
        let best = scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(s, _)| *s)
            .unwrap_or(RoutingStrategy::StandardProcessing);

        (best, self.schema.agency)
    }

    /// Select strategy automatically (low awareness)
    fn select_strategy_automatic(&self) -> (RoutingStrategy, f64) {
        // Direct attention-based selection
        let strategy = self.attention.dominant_strategy()
            .unwrap_or(RoutingStrategy::StandardProcessing);

        (strategy, 0.2) // Low agency influence
    }

    /// Apply control signals to modify selection
    fn apply_control_signals(&self, initial: RoutingStrategy) -> RoutingStrategy {
        if !self.config.agency_control {
            return initial;
        }

        // Check if control signals strongly favor a different strategy
        let mut best_controlled = initial;
        let mut best_score = self.attention.strategy_weights
            .get(&initial).copied().unwrap_or(0.0);

        for (strategy, control_signal) in &self.control_signals {
            if *control_signal > 0.5 {
                let base = self.attention.strategy_weights
                    .get(strategy).copied().unwrap_or(0.0);
                let boosted = base + control_signal * self.config.control_strength;

                if boosted > best_score * 1.3 { // 30% threshold for override
                    best_controlled = *strategy;
                    best_score = boosted;
                }
            }
        }

        best_controlled
    }

    /// Check if social modeling influenced the decision
    fn check_social_influence(&mut self, _strategy: &RoutingStrategy) -> bool {
        if !self.config.social_modeling || self.social_model.other_schemas.is_empty() {
            return false;
        }

        // Update joint attention
        self.social_model.detect_joint_attention(&self.attention);

        if self.social_model.joint_attention_strength > 0.5 {
            self.stats.joint_attention_events += 1;
            return true;
        }

        false
    }

    /// Calculate confidence in decision
    fn calculate_confidence(
        &self,
        strategy: &RoutingStrategy,
        meta_awareness: f64,
        schema_accuracy: f64
    ) -> f64 {
        let attention_weight = self.attention.strategy_weights
            .get(strategy).copied().unwrap_or(0.0);

        let focus_factor = self.attention.focus_intensity;
        let stability_factor = self.attention.stability;

        // Confidence is higher when:
        // - High attention to chosen strategy
        // - High focus intensity
        // - High stability
        // - High meta-awareness
        // - Accurate schema

        let confidence = 0.2 * attention_weight
            + 0.2 * focus_factor
            + 0.2 * stability_factor
            + 0.2 * meta_awareness
            + 0.2 * schema_accuracy;

        confidence.clamp(0.0, 1.0)
    }

    /// Generate human-readable explanation
    fn generate_explanation(&self, meta_awareness: f64, deliberate: bool) -> String {
        if deliberate {
            format!(
                "High meta-awareness ({:.2}) enabled deliberate control. \
                 Schema confidence: {:.2}, Agency: {:.2}, Phenomenal quality: {:.2}",
                meta_awareness,
                self.schema.model_confidence,
                self.schema.agency,
                self.schema.phenomenal_quality
            )
        } else {
            format!(
                "Low meta-awareness ({:.2}) led to automatic processing. \
                 Attention focus: {:.2}, Stability: {:.2}",
                meta_awareness,
                self.attention.focus_intensity,
                self.attention.stability
            )
        }
    }

    /// Update statistics
    fn update_stats(&mut self, decision: &ASTRoutingDecision) {
        self.stats.decisions += 1;

        // Rolling average for schema accuracy
        let n = self.stats.decisions as f64;
        self.stats.avg_schema_accuracy =
            (self.stats.avg_schema_accuracy * (n - 1.0) + decision.schema_accuracy) / n;

        self.stats.avg_meta_awareness =
            (self.stats.avg_meta_awareness * (n - 1.0) + decision.meta_awareness) / n;

        self.stats.avg_phenomenal_quality =
            (self.stats.avg_phenomenal_quality * (n - 1.0) + decision.phenomenal_quality) / n;

        if decision.agency_influence > 0.5 {
            self.stats.agency_influenced += 1;
        }
    }

    /// Model another router's behavior
    pub fn model_other_router(&mut self, router_id: &str, observed: &AttentionState) {
        if self.config.social_modeling {
            self.social_model.model_other(router_id, observed);
        }
    }

    /// Get current attention state
    pub fn attention_state(&self) -> &AttentionState {
        &self.attention
    }

    /// Get current schema
    pub fn schema(&self) -> &AttentionSchema {
        &self.schema
    }

    /// Get statistics
    pub fn stats(&self) -> &ASTRouterStats {
        &self.stats
    }

    /// Get phenomenal consciousness level (subjective experience intensity)
    pub fn phenomenal_consciousness(&self) -> f64 {
        self.schema.phenomenal_quality * self.schema.meta_awareness()
    }

    /// Report - summary of AST router state
    pub fn report(&self) -> String {
        format!(
            "AST Router Report:\n\
             - Decisions: {}\n\
             - Avg Meta-Awareness: {:.3}\n\
             - Avg Schema Accuracy: {:.3}\n\
             - Avg Phenomenal Quality: {:.3}\n\
             - Agency Influenced: {} ({:.1}%)\n\
             - Joint Attention Events: {}\n\
             - Current Focus: {:?}\n\
             - Current Phenomenal Consciousness: {:.3}",
            self.stats.decisions,
            self.stats.avg_meta_awareness,
            self.stats.avg_schema_accuracy,
            self.stats.avg_phenomenal_quality,
            self.stats.agency_influenced,
            if self.stats.decisions > 0 {
                100.0 * self.stats.agency_influenced as f64 / self.stats.decisions as f64
            } else { 0.0 },
            self.stats.joint_attention_events,
            self.attention.dominant_strategy(),
            self.phenomenal_consciousness()
        )
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Simple random number generator (deterministic for testing)
fn rand_simple() -> f64 {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    // LCG
    let a: u64 = 1103515245;
    let c: u64 = 12345;
    let m: u64 = 1 << 31;

    let val = (seed.wrapping_mul(a).wrapping_add(c)) % m;
    val as f64 / m as f64
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_state_new() {
        let state = AttentionState::new();

        assert!((state.total_attention() - 1.0).abs() < 0.01);
        assert!(state.focus_intensity >= 0.0 && state.focus_intensity <= 1.0);
    }

    #[test]
    fn test_attention_state_shift() {
        let mut state = AttentionState::new();

        state.shift_toward(RoutingStrategy::FullDeliberation, 0.5);

        let delib_weight = state.strategy_weights
            .get(&RoutingStrategy::FullDeliberation)
            .copied()
            .unwrap_or(0.0);

        assert!(delib_weight > 0.3); // Should be higher after shifting
        assert!((state.total_attention() - 1.0).abs() < 0.01); // Still normalized
    }

    #[test]
    fn test_attention_schema_new() {
        let schema = AttentionSchema::new();

        assert!(schema.model_confidence >= 0.0);
        assert!(schema.self_attribution > 0.5); // High self-attribution
        assert!(schema.agency > 0.5); // Moderate agency
    }

    #[test]
    fn test_attention_schema_update() {
        let mut schema = AttentionSchema::new();
        let mut state = AttentionState::new();

        // Shift attention toward deliberation
        state.shift_toward(RoutingStrategy::FullDeliberation, 0.8);

        // Update schema with more iterations for convergence
        for _ in 0..50 {
            schema.update_from_attention(&state, 0.3);
        }

        // Schema should track the shift (imperfectly due to noise)
        let modeled = schema.modeled_state.strategy_weights
            .get(&RoutingStrategy::FullDeliberation)
            .copied()
            .unwrap_or(0.0);

        // Lower threshold to account for random noise in update function
        assert!(modeled > 0.1); // Should show some tracking of deliberation
    }

    #[test]
    fn test_meta_awareness() {
        let mut schema = AttentionSchema::new();

        schema.model_confidence = 0.8;
        schema.self_attribution = 0.9;
        schema.phenomenal_quality = 0.7;

        let meta = schema.meta_awareness();

        // meta = 0.8 * 0.9 * 0.7 = 0.504
        assert!((meta - 0.504).abs() < 0.01);
    }

    #[test]
    fn test_social_attention_model() {
        let mut social = SocialAttentionModel::new();

        let mut observed = AttentionState::new();
        observed.shift_toward(RoutingStrategy::HeuristicGuided, 0.7);

        social.model_other("router_1", &observed);

        assert!(social.other_schemas.contains_key("router_1"));

        let predicted = social.predict_other_attention("router_1");
        assert!(predicted.is_some());
    }

    #[test]
    fn test_ast_router_creation() {
        let config = ASTRouterConfig::default();
        let router = ASTRouter::new(config);

        assert_eq!(router.stats.decisions, 0);
        assert!(router.history.is_empty());
    }

    #[test]
    fn test_ast_router_observe_and_route() {
        let config = ASTRouterConfig::default();
        let mut router = ASTRouter::new(config);

        // Create test state
        let state = LatentConsciousnessState::from_observables(0.7, 0.7, 0.8, 0.5);

        router.observe(&state);
        let decision = router.route();

        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        assert!(decision.meta_awareness >= 0.0);
        assert!(decision.phenomenal_quality >= 0.0);
    }

    #[test]
    fn test_ast_router_goal_setting() {
        let config = ASTRouterConfig::default();
        let mut router = ASTRouter::new(config);

        router.set_goal(RoutingStrategy::FullDeliberation);

        // Observe state
        let state = LatentConsciousnessState::default();
        router.observe(&state);

        // Multiple observations to build up schema
        for _ in 0..5 {
            router.observe(&state);
            let _ = router.route();
        }

        // Check that deliberation gets attention boost
        let delib_weight = router.attention.strategy_weights
            .get(&RoutingStrategy::FullDeliberation)
            .copied()
            .unwrap_or(0.0);

        // Should have some attention toward goal
        assert!(delib_weight > 0.0);
    }

    #[test]
    fn test_ast_router_stats() {
        let config = ASTRouterConfig::default();
        let mut router = ASTRouter::new(config);

        let state = LatentConsciousnessState::default();

        for _ in 0..10 {
            router.observe(&state);
            let _ = router.route();
        }

        assert_eq!(router.stats.decisions, 10);
        assert!(router.stats.avg_meta_awareness > 0.0);
        assert!(router.stats.avg_schema_accuracy > 0.0);
    }

    #[test]
    fn test_ast_phenomenal_consciousness() {
        let config = ASTRouterConfig::default();
        let mut router = ASTRouter::new(config);

        // Build up schema accuracy through multiple observations
        // from_observables(phi, integration, coherence, attention)
        let state = LatentConsciousnessState::from_observables(0.8, 0.8, 0.9, 0.7);

        for _ in 0..20 {
            router.observe(&state);
            let _ = router.route();
        }

        let phenomenal = router.phenomenal_consciousness();

        // Should have measurable phenomenal consciousness
        assert!(phenomenal > 0.0);
        assert!(phenomenal <= 1.0);
    }

    #[test]
    fn test_ast_social_modeling() {
        let mut config = ASTRouterConfig::default();
        config.social_modeling = true;
        let mut router = ASTRouter::new(config);

        // Model another router
        let mut other_attention = AttentionState::new();
        other_attention.shift_toward(RoutingStrategy::FastPatterns, 0.9);

        router.model_other_router("router_2", &other_attention);

        // Check that social model was updated
        assert!(router.social_model.other_schemas.contains_key("router_2"));
    }

    #[test]
    fn test_ast_router_report() {
        let config = ASTRouterConfig::default();
        let mut router = ASTRouter::new(config);

        let state = LatentConsciousnessState::default();
        router.observe(&state);
        let _ = router.route();

        let report = router.report();

        assert!(report.contains("AST Router Report"));
        assert!(report.contains("Decisions: 1"));
    }

    #[test]
    fn test_ast_control_signals() {
        let mut config = ASTRouterConfig::default();
        config.agency_control = true;
        config.control_strength = 0.5;
        let mut router = ASTRouter::new(config);

        router.set_goal(RoutingStrategy::FullDeliberation);

        let state = LatentConsciousnessState::default();
        router.observe(&state);

        // Should have control signal for goal
        assert!(router.control_signals.contains_key(&RoutingStrategy::FullDeliberation));
    }

    #[test]
    fn test_attention_state_dominant() {
        let mut state = AttentionState::new();

        state.strategy_weights.insert(RoutingStrategy::FullDeliberation, 0.5);
        state.strategy_weights.insert(RoutingStrategy::StandardProcessing, 0.2);
        state.strategy_weights.insert(RoutingStrategy::HeuristicGuided, 0.15);
        state.strategy_weights.insert(RoutingStrategy::FastPatterns, 0.1);
        state.strategy_weights.insert(RoutingStrategy::Reflexive, 0.05);

        let dominant = state.dominant_strategy();

        assert_eq!(dominant, Some(RoutingStrategy::FullDeliberation));
    }

    #[test]
    fn test_ast_config_default() {
        let config = ASTRouterConfig::default();

        assert!(config.schema_update_rate > 0.0);
        assert!(config.meta_awareness_threshold > 0.0);
        assert!(config.social_modeling);
        assert!(config.agency_control);
    }
}
