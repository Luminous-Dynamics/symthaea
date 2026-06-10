// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Modal Attention Router
//!
//! Dynamically routes attention across different modalities based on:
//! - Salience (how attention-grabbing each modality is)
//! - Φ (consciousness level determines routing capacity)
//! - Context (what's relevant to current goals)
//! - History (what was recently attended)
//!
//! # Architecture
//!
//! ```text
//!                    ┌─────────────────────────────────┐
//!                    │    CROSS-MODAL ATTENTION ROUTER │
//!                    ├─────────────────────────────────┤
//!  Visual ──────────►│                                 │
//!  Auditory ────────►│   Salience    ┌───────────┐    │
//!  Semantic ────────►│   Gating  ───►│ Attention │    │──► Unified
//!  Temporal ────────►│               │ Allocator │    │   Representation
//!  Emotional ───────►│   Φ Gate      └───────────┘    │
//!  Proprioceptive ──►│                                 │
//!                    └─────────────────────────────────┘
//!                            ▲               ▲
//!                            │               │
//!                      Context HV      Goal HV
//! ```
//!
//! # Key Features
//!
//! 1. **Φ-Gated Routing**: Higher consciousness = better cross-modal integration
//! 2. **Salience Competition**: Modalities compete for attention resources
//! 3. **Contextual Biasing**: Current context influences modality weights
//! 4. **Temporal Dynamics**: Attention shifts smoothly over time
//! 5. **Binding Integration**: Routes through CrossModalBinder for fusion

use super::binary_hv::BinaryHV;
use super::cross_modal_binding::{CrossModalBinder, CrossModalConfig, Modality};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// MODALITY INPUT
// ============================================================================

/// Input from a single modality
#[derive(Debug, Clone)]
pub struct ModalityInput {
    /// The modality this input comes from
    pub modality: Modality,
    /// The HDC representation (BinaryHV)
    pub hv: BinaryHV,
    /// Raw salience score (0.0 to 1.0)
    pub salience: f64,
    /// Confidence in this input
    pub confidence: f64,
    /// Timestamp of when this input was received
    pub timestamp: u64,
    /// Optional label for debugging
    pub label: Option<String>,
}

impl ModalityInput {
    pub fn new(modality: Modality, hv: BinaryHV, salience: f64) -> Self {
        Self {
            modality,
            hv,
            salience: salience.clamp(0.0, 1.0),
            confidence: 1.0,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            label: None,
        }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

// ============================================================================
// ROUTING CONFIGURATION
// ============================================================================

/// Configuration for the attention router
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Minimum Φ required for cross-modal binding (below this, only single modality)
    pub phi_threshold: f64,
    /// Temperature for softmax attention allocation
    pub temperature: f64,
    /// Decay rate for historical attention (0.0 = no memory, 1.0 = perfect memory)
    pub history_decay: f64,
    /// Maximum number of modalities to attend simultaneously
    pub max_simultaneous_modalities: usize,
    /// Weight for salience in attention computation
    pub salience_weight: f64,
    /// Weight for context relevance in attention computation
    pub context_weight: f64,
    /// Weight for goal relevance in attention computation
    pub goal_weight: f64,
    /// Enable inhibition of return (reduce attention to recently attended)
    pub inhibition_of_return: bool,
    /// Inhibition of return strength
    pub ior_strength: f64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            phi_threshold: 0.3,
            temperature: 1.0,
            history_decay: 0.9,
            max_simultaneous_modalities: 4,
            salience_weight: 0.4,
            context_weight: 0.3,
            goal_weight: 0.3,
            inhibition_of_return: true,
            ior_strength: 0.3,
        }
    }
}

// ============================================================================
// ATTENTION STATE
// ============================================================================

/// Current attention state for a modality
#[derive(Debug, Clone)]
pub struct ModalityAttention {
    /// Current attention weight (0.0 to 1.0)
    pub weight: f64,
    /// How long this modality has been attended (in steps)
    pub dwell_time: usize,
    /// Last time this modality was strongly attended
    pub last_attended: u64,
    /// Cumulative attention received
    pub cumulative_attention: f64,
}

impl Default for ModalityAttention {
    fn default() -> Self {
        Self {
            weight: 0.0,
            dwell_time: 0,
            last_attended: 0,
            cumulative_attention: 0.0,
        }
    }
}

// ============================================================================
// ROUTING RESULT
// ============================================================================

/// Result of cross-modal attention routing
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Final unified representation
    pub unified_hv: BinaryHV,

    /// Attention weights per modality
    pub attention_weights: HashMap<Modality, f64>,

    /// Which modalities were bound together
    pub bound_modalities: Vec<Modality>,

    /// Effective Φ (consciousness level) used for routing
    pub effective_phi: f64,

    /// Whether cross-modal binding occurred (vs single modality focus)
    pub cross_modal_bound: bool,

    /// Dominant modality (highest attention)
    pub dominant_modality: Option<Modality>,

    /// Routing quality score (0.0 to 1.0)
    pub quality: f64,

    /// Timestamp
    pub timestamp: u64,
}

impl RoutingResult {
    /// Get a summary of which modalities contributed
    pub fn summary(&self) -> String {
        let weights: Vec<String> = self
            .attention_weights
            .iter()
            .filter(|(_, w)| **w > 0.1)
            .map(|(m, w)| format!("{m:?}: {w:.2}"))
            .collect();

        format!(
            "Unified ({} modalities): [{}] Φ={:.2} quality={:.2}",
            self.bound_modalities.len(),
            weights.join(", "),
            self.effective_phi,
            self.quality
        )
    }
}

// ============================================================================
// CROSS-MODAL ATTENTION ROUTER
// ============================================================================

/// The main cross-modal attention router
pub struct CrossModalAttentionRouter {
    /// Configuration
    config: RouterConfig,

    /// Cross-modal binder for actual binding operations
    binder: CrossModalBinder,

    /// Current attention state per modality
    attention_state: HashMap<Modality, ModalityAttention>,

    /// Current context vector (influences attention)
    context_hv: Option<BinaryHV>,

    /// Current goal vector (influences attention)
    goal_hv: Option<BinaryHV>,

    /// History of routing results
    history: Vec<RoutingResult>,

    /// Maximum history length
    max_history: usize,

    /// Step counter
    step: usize,
}

impl CrossModalAttentionRouter {
    pub fn new() -> Self {
        Self::with_config(RouterConfig::default())
    }

    pub fn with_config(config: RouterConfig) -> Self {
        let binder_config = CrossModalConfig {
            dimension: BinaryHV::DIM,
            base_seed: 42,
            learn_alignment: true,
            alignment_lr: 0.01,
            temperature: config.temperature as f32,
            normalize_intermediate: true,
        };

        Self {
            config,
            binder: CrossModalBinder::with_config(binder_config),
            attention_state: HashMap::new(),
            context_hv: None,
            goal_hv: None,
            history: Vec::new(),
            max_history: 100,
            step: 0,
        }
    }

    /// Set current context (influences attention allocation)
    pub fn set_context(&mut self, context: BinaryHV) {
        self.context_hv = Some(context);
    }

    /// Set current goal (influences attention allocation)
    pub fn set_goal(&mut self, goal: BinaryHV) {
        self.goal_hv = Some(goal);
    }

    /// Route attention across modalities and produce unified representation
    ///
    /// # Arguments
    /// * `inputs` - Inputs from various modalities
    /// * `phi` - Current consciousness level (Φ)
    ///
    /// # Returns
    /// * `RoutingResult` - Unified representation with attention weights
    pub fn route(&mut self, inputs: &[ModalityInput], phi: f64) -> RoutingResult {
        self.step += 1;

        if inputs.is_empty() {
            return self.empty_result(phi);
        }

        // Step 1: Compute raw attention scores for each input
        let raw_scores = self.compute_raw_attention_scores(inputs);

        // Step 2: Apply Φ-gating (consciousness determines integration capacity)
        let phi_gated_scores = self.apply_phi_gating(&raw_scores, phi);

        // Step 3: Compute final attention weights (softmax)
        let attention_weights = self.compute_attention_weights(&phi_gated_scores);

        // Step 4: Update attention state (for temporal dynamics)
        self.update_attention_state(&attention_weights);

        // Step 5: Determine which modalities to bind
        let (modalities_to_bind, cross_modal) =
            self.select_modalities_to_bind(&attention_weights, phi);

        // Step 6: Perform binding using CrossModalBinder
        let unified_hv = self.perform_binding(inputs, &attention_weights, &modalities_to_bind);

        // Step 7: Compute quality metrics
        let quality = self.compute_routing_quality(&attention_weights, phi);

        // Step 8: Build result
        let result = RoutingResult {
            unified_hv,
            attention_weights: attention_weights.clone(),
            bound_modalities: modalities_to_bind,
            effective_phi: phi,
            cross_modal_bound: cross_modal,
            dominant_modality: self.find_dominant_modality(&attention_weights),
            quality,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };

        // Store in history
        self.history.push(result.clone());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        result
    }

    /// Compute raw attention scores based on salience, context, and goals
    fn compute_raw_attention_scores(&self, inputs: &[ModalityInput]) -> HashMap<Modality, f64> {
        let mut scores = HashMap::new();

        for input in inputs {
            let mut score = input.salience * self.config.salience_weight;

            // Add context relevance
            if let Some(ref context) = self.context_hv {
                let context_sim = input.hv.similarity(context) as f64;
                score += context_sim * self.config.context_weight;
            }

            // Add goal relevance
            if let Some(ref goal) = self.goal_hv {
                let goal_sim = input.hv.similarity(goal) as f64;
                score += goal_sim * self.config.goal_weight;
            }

            // Apply inhibition of return
            if self.config.inhibition_of_return {
                if let Some(state) = self.attention_state.get(&input.modality) {
                    if state.dwell_time > 3 {
                        // Reduce attention to recently attended modalities
                        let ior_penalty =
                            self.config.ior_strength * (state.dwell_time as f64 / 10.0).min(1.0);
                        score *= 1.0 - ior_penalty;
                    }
                }
            }

            // Weight by confidence
            score *= input.confidence;

            scores.insert(input.modality, score);
        }

        scores
    }

    /// Apply Φ-gating: consciousness level determines integration capacity
    fn apply_phi_gating(
        &self,
        raw_scores: &HashMap<Modality, f64>,
        phi: f64,
    ) -> HashMap<Modality, f64> {
        let mut gated = HashMap::new();

        // Low Φ: only dominant modality gets attention
        // High Φ: all modalities can contribute proportionally
        let integration_capacity = if phi < self.config.phi_threshold {
            0.1 // Minimal integration
        } else {
            // Sigmoid-like scaling
            1.0 / (1.0 + (-5.0 * (phi - self.config.phi_threshold)).exp())
        };

        // Find max score
        let max_score = raw_scores.values().cloned().fold(0.0, f64::max);

        for (modality, score) in raw_scores {
            // Suppress non-dominant modalities when Φ is low
            let relative_score = if max_score > 0.0 {
                score / max_score
            } else {
                0.0
            };
            let suppression = if relative_score < 0.5 {
                1.0 - (1.0 - integration_capacity) * (1.0 - relative_score * 2.0)
            } else {
                1.0
            };

            gated.insert(*modality, score * suppression);
        }

        gated
    }

    /// Convert scores to normalized attention weights using softmax
    fn compute_attention_weights(&self, scores: &HashMap<Modality, f64>) -> HashMap<Modality, f64> {
        let mut weights = HashMap::new();

        // Compute softmax
        let max_score = scores.values().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = scores
            .values()
            .map(|s| ((s - max_score) / self.config.temperature).exp())
            .sum();

        if exp_sum > 0.0 {
            for (modality, score) in scores {
                let exp_val = ((score - max_score) / self.config.temperature).exp();
                weights.insert(*modality, exp_val / exp_sum);
            }
        } else {
            // Fallback: uniform weights
            let n = scores.len() as f64;
            for modality in scores.keys() {
                weights.insert(*modality, 1.0 / n);
            }
        }

        weights
    }

    /// Update attention state for temporal dynamics
    fn update_attention_state(&mut self, weights: &HashMap<Modality, f64>) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Decay existing states
        for state in self.attention_state.values_mut() {
            state.weight *= self.config.history_decay;
        }

        // Update with new weights
        for (modality, weight) in weights {
            let state = self.attention_state.entry(*modality).or_default();

            if *weight > 0.3 {
                state.dwell_time += 1;
                state.last_attended = now;
            } else {
                state.dwell_time = 0;
            }

            state.weight = *weight;
            state.cumulative_attention += weight;
        }
    }

    /// Select which modalities to bind together
    fn select_modalities_to_bind(
        &self,
        weights: &HashMap<Modality, f64>,
        phi: f64,
    ) -> (Vec<Modality>, bool) {
        // Sort modalities by weight
        let mut sorted: Vec<_> = weights.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Low Φ: only top modality
        if phi < self.config.phi_threshold {
            if let Some((m, _)) = sorted.first() {
                return (vec![**m], false);
            }
            return (vec![], false);
        }

        // High Φ: bind modalities above threshold
        let threshold = 0.1;
        let selected: Vec<Modality> = sorted
            .iter()
            .filter(|(_, w)| **w > threshold)
            .take(self.config.max_simultaneous_modalities)
            .map(|(m, _)| **m)
            .collect();

        let cross_modal = selected.len() > 1;
        (selected, cross_modal)
    }

    /// Perform the actual binding using CrossModalBinder
    fn perform_binding(
        &mut self,
        inputs: &[ModalityInput],
        weights: &HashMap<Modality, f64>,
        modalities_to_bind: &[Modality],
    ) -> BinaryHV {
        if modalities_to_bind.is_empty() {
            return BinaryHV::random(42);
        }

        if modalities_to_bind.len() == 1 {
            // Single modality: just return that input
            for input in inputs {
                if input.modality == modalities_to_bind[0] {
                    return input.hv;
                }
            }
            return BinaryHV::random(42);
        }

        // Multiple modalities: use weighted bundling
        // Convert BinaryHV to weighted inputs for binding
        let selected_inputs: Vec<_> = inputs
            .iter()
            .filter(|i| modalities_to_bind.contains(&i.modality))
            .collect();

        if selected_inputs.is_empty() {
            return BinaryHV::random(42);
        }

        // Weighted bundle
        let weighted_hvs: Vec<(BinaryHV, f64)> = selected_inputs
            .iter()
            .map(|i| {
                let weight = *weights.get(&i.modality).unwrap_or(&0.0);
                (i.hv, weight)
            })
            .collect();

        Self::weighted_bundle(&weighted_hvs)
    }

    /// Weighted bundle operation (similar to emotional_depth)
    fn weighted_bundle(weighted_hvs: &[(BinaryHV, f64)]) -> BinaryHV {
        if weighted_hvs.is_empty() {
            return BinaryHV::random(42);
        }

        if weighted_hvs.len() == 1 {
            return weighted_hvs[0].0;
        }

        const DIMENSIONS: usize = 16384;
        let mut accumulator = vec![0.0f64; DIMENSIONS];

        for (hv, weight) in weighted_hvs {
            for (i, byte) in hv.0.iter().enumerate() {
                for bit in 0..8 {
                    let idx = i * 8 + bit;
                    if idx < DIMENSIONS {
                        let bit_val = (byte >> (7 - bit)) & 1;
                        accumulator[idx] += if bit_val == 1 { *weight } else { -*weight };
                    }
                }
            }
        }

        // Threshold at 0
        let mut result = [0u8; 2048];
        for (i, &sum) in accumulator.iter().enumerate() {
            let byte_idx = i / 8;
            let bit_idx = 7 - (i % 8);
            if sum > 0.0 {
                result[byte_idx] |= 1 << bit_idx;
            }
        }

        BinaryHV(result)
    }

    /// Compute routing quality
    fn compute_routing_quality(&self, weights: &HashMap<Modality, f64>, phi: f64) -> f64 {
        // Quality factors:
        // 1. Entropy of attention distribution (higher = more distributed)
        // 2. Strength of dominant modality (higher = more focused)
        // 3. Φ level (higher = better integration)

        let entropy = self.attention_entropy(weights);
        let max_weight = weights.values().cloned().fold(0.0, f64::max);

        // Balance between focus and distribution
        let focus_quality = max_weight;
        let distribution_quality = entropy / (weights.len() as f64).ln().max(1.0);

        // Weight by Φ
        (focus_quality * 0.4 + distribution_quality * 0.3 + phi * 0.3).clamp(0.0, 1.0)
    }

    /// Compute entropy of attention distribution
    fn attention_entropy(&self, weights: &HashMap<Modality, f64>) -> f64 {
        let mut entropy = 0.0;
        for w in weights.values() {
            if *w > 0.0 {
                entropy -= w * w.ln();
            }
        }
        entropy
    }

    /// Find the dominant modality
    fn find_dominant_modality(&self, weights: &HashMap<Modality, f64>) -> Option<Modality> {
        weights
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(m, _)| *m)
    }

    /// Create empty result
    fn empty_result(&self, phi: f64) -> RoutingResult {
        RoutingResult {
            unified_hv: BinaryHV::random(42),
            attention_weights: HashMap::new(),
            bound_modalities: Vec::new(),
            effective_phi: phi,
            cross_modal_bound: false,
            dominant_modality: None,
            quality: 0.0,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Get attention history for a modality
    pub fn modality_history(&self, modality: Modality) -> Vec<f64> {
        self.history
            .iter()
            .filter_map(|r| r.attention_weights.get(&modality).copied())
            .collect()
    }

    /// Get dominant modality over time
    pub fn dominant_modality_sequence(&self) -> Vec<Modality> {
        self.history
            .iter()
            .filter_map(|r| r.dominant_modality)
            .collect()
    }

    /// Check if attention is stable (same dominant modality for N steps)
    pub fn is_attention_stable(&self, n: usize) -> bool {
        if self.history.len() < n {
            return false;
        }

        let recent: Vec<_> = self.history.iter().rev().take(n).collect();
        let first = recent.first().and_then(|r| r.dominant_modality);

        recent.iter().all(|r| r.dominant_modality == first)
    }

    /// Get current attention state
    pub fn attention_state(&self) -> &HashMap<Modality, ModalityAttention> {
        &self.attention_state
    }

    /// Reset router state
    pub fn reset(&mut self) {
        self.attention_state.clear();
        self.context_hv = None;
        self.goal_hv = None;
        self.history.clear();
        self.step = 0;
    }
}

impl Default for CrossModalAttentionRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let router = CrossModalAttentionRouter::new();
        assert!(router.attention_state.is_empty());
    }

    #[test]
    fn test_single_modality_routing() {
        let mut router = CrossModalAttentionRouter::new();

        let input = ModalityInput::new(Modality::Visual, BinaryHV::random(42), 0.8);
        let result = router.route(&[input], 0.5);

        assert!(result.attention_weights.contains_key(&Modality::Visual));
        assert_eq!(result.bound_modalities.len(), 1);
        assert_eq!(result.dominant_modality, Some(Modality::Visual));
    }

    #[test]
    fn test_multi_modality_routing() {
        let mut router = CrossModalAttentionRouter::new();

        let inputs = vec![
            ModalityInput::new(Modality::Visual, BinaryHV::random(1), 0.8),
            ModalityInput::new(Modality::Auditory, BinaryHV::random(2), 0.6),
            ModalityInput::new(Modality::Semantic, BinaryHV::random(3), 0.4),
        ];

        let result = router.route(&inputs, 0.7);

        assert!(result.cross_modal_bound);
        assert!(result.bound_modalities.len() >= 2);
    }

    #[test]
    fn test_low_phi_single_modality_focus() {
        let mut router = CrossModalAttentionRouter::new();

        let inputs = vec![
            ModalityInput::new(Modality::Visual, BinaryHV::random(1), 0.8),
            ModalityInput::new(Modality::Auditory, BinaryHV::random(2), 0.6),
        ];

        // Low Φ should result in single modality focus
        let result = router.route(&inputs, 0.1);

        assert!(!result.cross_modal_bound);
        assert_eq!(result.bound_modalities.len(), 1);
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let mut router = CrossModalAttentionRouter::new();

        let inputs = vec![
            ModalityInput::new(Modality::Visual, BinaryHV::random(1), 0.5),
            ModalityInput::new(Modality::Auditory, BinaryHV::random(2), 0.5),
            ModalityInput::new(Modality::Semantic, BinaryHV::random(3), 0.5),
        ];

        let result = router.route(&inputs, 0.6);

        let sum: f64 = result.attention_weights.values().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Weights should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_context_influences_attention() {
        let mut router = CrossModalAttentionRouter::new();

        // Create a context that's similar to visual input
        let visual_hv = BinaryHV::random(42);
        router.set_context(visual_hv.clone());

        let inputs = vec![
            ModalityInput::new(Modality::Visual, visual_hv, 0.5),
            ModalityInput::new(Modality::Auditory, BinaryHV::random(99), 0.5),
        ];

        let result = router.route(&inputs, 0.6);

        // Visual should have higher attention due to context similarity
        let visual_weight = result
            .attention_weights
            .get(&Modality::Visual)
            .unwrap_or(&0.0);
        let auditory_weight = result
            .attention_weights
            .get(&Modality::Auditory)
            .unwrap_or(&0.0);

        assert!(
            visual_weight > auditory_weight,
            "Visual ({}) should have higher attention than auditory ({}) due to context",
            visual_weight,
            auditory_weight
        );
    }

    #[test]
    fn test_attention_stability() {
        let mut router = CrossModalAttentionRouter::new();

        let input = ModalityInput::new(Modality::Visual, BinaryHV::random(42), 0.9);

        // Route same input multiple times
        for _ in 0..5 {
            router.route(&[input.clone()], 0.5);
        }

        assert!(router.is_attention_stable(3));
    }
}
