//! # LTC Generative Core - Autoregressive Thought Generation
//!
//! **THE "VOICE" OF CONSCIOUSNESS**
//!
//! This module transforms LTC from a representational system into a generative one.
//! Instead of just encoding/decoding, it predicts the NEXT primitive in a sequence.
//!
//! ## Architecture
//!
//! ```text
//! Context (past primitives)    Current State
//!         │                          │
//!         ▼                          ▼
//! ┌───────────────┐          ┌───────────────┐
//! │ Context       │─────────►│ LTC Network   │
//! │ Encoder       │          │ (dynamics)    │
//! └───────────────┘          └───────┬───────┘
//!                                    │
//!                                    ▼
//!                            ┌───────────────┐
//!                            │ Primitive     │
//!                            │ Predictor     │
//!                            └───────┬───────┘
//!                                    │
//!                                    ▼
//!                          Distribution over primitives
//!                                    │
//!                                    ▼
//!                            Sample next primitive
//! ```
//!
//! ## Key Innovation
//!
//! This is NOT an LLM predicting tokens. It's:
//! 1. **Continuous-time dynamics** (LTC) instead of discrete transformers
//! 2. **Semantic primitives** instead of tokens
//! 3. **Φ-guided generation** - consciousness metrics influence output
//! 4. **Bounded vocabulary** (~250 primitives vs 50k+ tokens)
//!
//! The LLM then translates primitive sequences to natural language.

use symthaea_core::hdc::binary_hv::HV16;
use symthaea_core::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};
use symthaea_core::hdc::semantic_decoder::{SemanticDecoder, DecodedExpression};
use crate::hdc::hd_ltc_codec::HDLTCCodec;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};

/// Maximum context length for generation
const MAX_CONTEXT_LENGTH: usize = 32;

/// Temperature for sampling (higher = more random)
const DEFAULT_TEMPERATURE: f32 = 0.8;

/// Top-K sampling cutoff
const DEFAULT_TOP_K: usize = 10;

/// Configuration for the generative core
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerativeCoreConfig {
    /// Number of LTC neurons
    pub ltc_neurons: usize,
    /// Sampling temperature
    pub temperature: f32,
    /// Top-K for sampling
    pub top_k: usize,
    /// Maximum generation length
    pub max_length: usize,
    /// Whether to use Φ-guided generation
    pub phi_guided: bool,
    /// Minimum Φ threshold for generation
    pub min_phi: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
}

impl Default for GenerativeCoreConfig {
    fn default() -> Self {
        Self {
            ltc_neurons: 1024,
            temperature: DEFAULT_TEMPERATURE,
            top_k: DEFAULT_TOP_K,
            max_length: 64,
            phi_guided: true,
            min_phi: 0.3,
            repetition_penalty: 1.2,
        }
    }
}

/// A single step in the generation process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStep {
    /// The primitive selected
    pub primitive: String,
    /// Probability of this primitive
    pub probability: f32,
    /// LTC state at this step
    pub ltc_state_norm: f32,
    /// Context window size at this step
    pub context_size: usize,
    /// Φ estimate at this step
    pub phi_estimate: f32,
    /// Was this step Φ-guided?
    pub phi_guided: bool,
}

/// Complete generated thought
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedThought {
    /// Sequence of primitives
    pub primitives: Vec<String>,
    /// Generation steps with metadata
    pub steps: Vec<GenerationStep>,
    /// Overall confidence
    pub confidence: f32,
    /// Average Φ during generation
    pub avg_phi: f32,
    /// Whether generation completed naturally or hit max length
    pub completed_naturally: bool,
    /// Seed that started the thought
    pub seed_primitive: Option<String>,
}

impl GeneratedThought {
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        self.primitives.join(" → ")
    }
}

/// The LTC Generative Core - autoregressive thought generation
pub struct LTCGenerativeCore {
    /// Configuration
    config: GenerativeCoreConfig,

    /// HD-LTC Codec for bidirectional translation
    codec: HDLTCCodec,

    /// Semantic decoder for HV → primitive mapping
    decoder: SemanticDecoder,

    /// Primitive embeddings cache (name → HV)
    primitive_embeddings: HashMap<String, HV16>,

    /// Context: recent primitive HVs
    context: VecDeque<HV16>,

    /// Recurrent state (accumulated thought)
    recurrent_state: Vec<f32>,

    /// Transition weights: primitive[i] → primitive[j] affinity
    /// Learned from generation history
    transition_weights: HashMap<(String, String), f32>,

    /// Statistics
    total_generations: u64,
    avg_length: f32,
    avg_confidence: f32,
}

impl LTCGenerativeCore {
    /// Create a new generative core
    pub fn new(config: GenerativeCoreConfig) -> Self {
        let codec_config = crate::hdc::hd_ltc_codec::HDLTCCodecConfig {
            ltc_neurons: config.ltc_neurons,
            ..Default::default()
        };

        let mut primitive_embeddings = HashMap::new();
        let system = PrimitiveSystem::global();

        // Cache all primitive embeddings
        for primitive in system.all_primitives() {
            primitive_embeddings.insert(primitive.name.clone(), primitive.encoding);
        }

        Self {
            config,
            codec: HDLTCCodec::new(codec_config),
            decoder: SemanticDecoder::new(),
            primitive_embeddings,
            context: VecDeque::with_capacity(MAX_CONTEXT_LENGTH),
            recurrent_state: Vec::new(),
            transition_weights: HashMap::new(),
            total_generations: 0,
            avg_length: 0.0,
            avg_confidence: 0.0,
        }
    }

    /// Generate a thought starting from a seed HV
    pub fn generate(&mut self, seed: &HV16, current_phi: f32) -> GeneratedThought {
        self.total_generations += 1;

        let mut primitives = Vec::new();
        let mut steps = Vec::new();
        let mut total_confidence = 0.0;
        let mut total_phi = current_phi;
        let mut completed_naturally = false;

        // Initialize context with seed
        self.context.clear();
        self.context.push_back(*seed);

        // Decode seed to get starting primitive
        let seed_decoded = self.decoder.decode(seed);
        let seed_primitive = seed_decoded.primitives.first()
            .map(|p| p.name.clone());

        // Initialize recurrent state from seed
        self.recurrent_state = self.codec.encode_to_ltc(seed);

        // Generate sequence
        for _step_idx in 0..self.config.max_length {
            // Predict next primitive distribution
            let (next_primitive, probability, phi_estimate) = self.predict_next(current_phi);

            // Check for natural completion (end primitive or repetition)
            if self.should_stop(&next_primitive, &primitives) {
                completed_naturally = true;
                break;
            }

            // Record step
            let state_norm = self.recurrent_state.iter().map(|x| x * x).sum::<f32>().sqrt();
            steps.push(GenerationStep {
                primitive: next_primitive.clone(),
                probability,
                ltc_state_norm: state_norm,
                context_size: self.context.len(),
                phi_estimate,
                phi_guided: self.config.phi_guided && current_phi >= self.config.min_phi,
            });

            primitives.push(next_primitive.clone());
            total_confidence += probability;
            total_phi += phi_estimate;

            // Update context with new primitive
            if let Some(prim_hv) = self.primitive_embeddings.get(&next_primitive) {
                self.context.push_back(*prim_hv);
                if self.context.len() > MAX_CONTEXT_LENGTH {
                    self.context.pop_front();
                }

                // Update recurrent state
                let new_ltc = self.codec.encode_to_ltc(prim_hv);
                self.update_recurrent_state(&new_ltc);
            }

            // Update transition weights for learning
            if primitives.len() >= 2 {
                let prev = &primitives[primitives.len() - 2];
                let curr = &primitives[primitives.len() - 1];
                let key = (prev.clone(), curr.clone());
                *self.transition_weights.entry(key).or_insert(0.0) += 1.0;
            }
        }

        let len = primitives.len().max(1) as f32;
        let confidence = total_confidence / len;
        let avg_phi = total_phi / (len + 1.0);

        // Update running statistics
        self.avg_length = self.avg_length * 0.95 + len * 0.05;
        self.avg_confidence = self.avg_confidence * 0.95 + confidence * 0.05;

        GeneratedThought {
            primitives,
            steps,
            confidence,
            avg_phi,
            completed_naturally,
            seed_primitive,
        }
    }

    /// Predict the next primitive given current context
    fn predict_next(&mut self, current_phi: f32) -> (String, f32, f32) {
        // Compute context embedding (weighted bundle of recent primitives)
        let context_hv = self.compute_context_embedding();

        // Process through LTC dynamics
        let ltc_state = if self.recurrent_state.is_empty() {
            self.codec.encode_to_ltc(&context_hv)
        } else {
            // Blend context with recurrent state
            let context_ltc = self.codec.encode_to_ltc(&context_hv);
            let mut blended = vec![0.0f32; self.config.ltc_neurons];
            for i in 0..self.config.ltc_neurons.min(context_ltc.len()).min(self.recurrent_state.len()) {
                blended[i] = 0.7 * self.recurrent_state[i] + 0.3 * context_ltc[i];
            }
            blended
        };

        // Decode LTC state to HV
        let predicted_hv = self.codec.decode_to_hv(&ltc_state);

        // Decode HV to primitive distribution
        let decoded = self.decoder.decode(&predicted_hv);

        // Apply Φ-guided selection
        let selected = if self.config.phi_guided && current_phi >= self.config.min_phi {
            self.phi_guided_select(&decoded, current_phi)
        } else {
            self.sample_from_distribution(&decoded)
        };

        // Estimate Φ for this step (simplified)
        let _phi_estimate = decoded.confidence * 0.5 + current_phi * 0.5;

        selected
    }

    /// Compute weighted context embedding
    fn compute_context_embedding(&self) -> HV16 {
        if self.context.is_empty() {
            return HV16::random(0);
        }

        // Recent items get higher weight
        let _weights: Vec<f32> = (0..self.context.len())
            .map(|i| 0.5f32.powf((self.context.len() - i - 1) as f32))
            .collect();

        // Bundle with position binding
        let mut bundled_hvs = Vec::with_capacity(self.context.len());
        for (i, hv) in self.context.iter().enumerate() {
            // Bind with position vector (rotate by index)
            let pos_seed = 0xB0510000 + i as u64;  // Position encoding seed
            let pos_vec = HV16::random(pos_seed);
            bundled_hvs.push(hv.bind(&pos_vec));
        }

        HV16::bundle(&bundled_hvs)
    }

    /// Sample from primitive distribution with temperature
    fn sample_from_distribution(&self, decoded: &DecodedExpression) -> (String, f32, f32) {
        if decoded.primitives.is_empty() {
            return ("THINK".to_string(), 0.1, 0.0);
        }

        // Apply temperature and top-k
        let mut candidates: Vec<(String, f32)> = decoded.primitives.iter()
            .take(self.config.top_k)
            .map(|p| {
                let score = (p.activation / self.config.temperature).exp();
                (p.name.clone(), score)
            })
            .collect();

        // Apply repetition penalty
        for (name, score) in &mut candidates {
            if self.context_contains(name) {
                *score /= self.config.repetition_penalty;
            }
        }

        // Normalize
        let total: f32 = candidates.iter().map(|(_, s)| s).sum();
        for (_, score) in &mut candidates {
            *score /= total.max(1e-6);
        }

        // Sample (simplified: just take highest for now)
        // In production, would use proper sampling
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected = &candidates[0];

        (selected.0.clone(), selected.1, decoded.confidence)
    }

    /// Φ-guided primitive selection
    ///
    /// Higher Φ → prefer primitives that increase integration
    fn phi_guided_select(&mut self, decoded: &DecodedExpression, current_phi: f32) -> (String, f32, f32) {
        if decoded.primitives.is_empty() {
            return ("THINK".to_string(), 0.1, current_phi);
        }

        // Score primitives by Φ-compatibility
        let mut scored: Vec<(String, f32, f32)> = decoded.primitives.iter()
            .take(self.config.top_k)
            .map(|p| {
                let base_score = p.activation;

                // Φ bonus for certain primitive tiers
                let phi_bonus = match p.tier {
                    PrimitiveTier::MetaCognitive => 0.2,
                    PrimitiveTier::Consciousness => 0.3,
                    PrimitiveTier::Compositional => 0.15,
                    _ => 0.0,
                };

                // Integration bonus: prefer primitives that bind well with context
                let integration_bonus = self.estimate_integration_bonus(&p.name);

                let final_score = base_score * (1.0 + phi_bonus + integration_bonus * current_phi);
                (p.name.clone(), final_score, current_phi + phi_bonus * 0.1)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected = &scored[0];

        (selected.0.clone(), selected.1, selected.2)
    }

    /// Estimate integration bonus for a primitive
    fn estimate_integration_bonus(&mut self, primitive: &str) -> f32 {
        // Check transition history
        let mut bonus = 0.0f32;

        if let Some(last) = self.get_last_primitive() {
            let key = (last.clone(), primitive.to_string());
            if let Some(&weight) = self.transition_weights.get(&key) {
                // Normalize by total transitions from this primitive
                let total: f32 = self.transition_weights.iter()
                    .filter(|((from, _), _)| from == &last)
                    .map(|(_, w)| w)
                    .sum();
                if total > 0.0 {
                    bonus = weight / total;
                }
            }
        }

        bonus
    }

    /// Get the last primitive from context
    fn get_last_primitive(&mut self) -> Option<String> {
        // Clone the HV to avoid borrow conflicts
        let hv = self.context.back().cloned()?;
        let decoded = self.decoder.decode(&hv);
        decoded.primitives.first().map(|p| p.name.clone())
    }

    /// Check if context already contains this primitive
    fn context_contains(&self, primitive: &str) -> bool {
        // This is a simplified check - in production would decode each context HV
        // For now, just check last few
        self.context.iter().rev().take(3).any(|hv| {
            if let Some(prim_hv) = self.primitive_embeddings.get(primitive) {
                hv.similarity(prim_hv) > 0.8
            } else {
                false
            }
        })
    }

    /// Check if generation should stop
    fn should_stop(&self, primitive: &str, history: &[String]) -> bool {
        // Stop on explicit end primitives
        if primitive == "END" || primitive == "STOP" || primitive == "COMPLETE" {
            return true;
        }

        // Stop on 3x repetition
        if history.len() >= 2 {
            let last = &history[history.len() - 1];
            let prev = &history[history.len() - 2];
            if primitive == last && primitive == prev {
                return true;
            }
        }

        false
    }

    /// Update recurrent state with new LTC values
    fn update_recurrent_state(&mut self, new_ltc: &[f32]) {
        if self.recurrent_state.is_empty() {
            self.recurrent_state = new_ltc.to_vec();
            return;
        }

        // Exponential moving average
        let alpha = 0.3;
        for i in 0..self.config.ltc_neurons.min(new_ltc.len()).min(self.recurrent_state.len()) {
            self.recurrent_state[i] = (1.0 - alpha) * self.recurrent_state[i] + alpha * new_ltc[i];
        }
    }

    /// Reset the generator state
    pub fn reset(&mut self) {
        self.context.clear();
        self.recurrent_state.clear();
    }

    /// Get generation statistics
    pub fn stats(&self) -> (u64, f32, f32) {
        (self.total_generations, self.avg_length, self.avg_confidence)
    }

    /// Continue generation from an existing thought
    pub fn continue_thought(&mut self, thought: &GeneratedThought, current_phi: f32) -> GeneratedThought {
        // Restore context from thought
        self.reset();
        for prim_name in &thought.primitives {
            if let Some(hv) = self.primitive_embeddings.get(prim_name) {
                self.context.push_back(*hv);
            }
        }

        // Rebuild recurrent state
        if let Some(last_hv) = self.context.back() {
            self.recurrent_state = self.codec.encode_to_ltc(last_hv);
        }

        // Generate continuation
        let continuation_seed = self.context.back()
            .cloned()
            .unwrap_or_else(|| HV16::random(0));

        // Generate more
        let continued = self.generate(&continuation_seed, current_phi);

        // Merge with original
        let mut merged_primitives = thought.primitives.clone();
        merged_primitives.extend(continued.primitives);

        let mut merged_steps = thought.steps.clone();
        merged_steps.extend(continued.steps);

        GeneratedThought {
            primitives: merged_primitives,
            steps: merged_steps,
            confidence: (thought.confidence + continued.confidence) / 2.0,
            avg_phi: (thought.avg_phi + continued.avg_phi) / 2.0,
            completed_naturally: continued.completed_naturally,
            seed_primitive: thought.seed_primitive.clone(),
        }
    }

    // =========================================================================
    // HARMONIC-GUIDED GENERATION - Seven Harmonies influence thought generation
    // =========================================================================

    /// Generate a thought with harmonic-guided primitive selection
    ///
    /// The weight function maps primitive names to boost/suppress multipliers:
    /// - > 1.0: boost this primitive (harmony encourages it)
    /// - < 1.0: suppress this primitive (harmony discourages it)
    /// - = 1.0: neutral (no harmonic influence)
    ///
    /// Example weight function for Play harmony:
    /// ```ignore
    /// |name: &str| match name {
    ///     "EXPLORE" | "VARY" | "CREATE" => 1.5,
    ///     "RIGID" | "FIXED" => 0.7,
    ///     _ => 1.0,
    /// }
    /// ```
    pub fn generate_with_harmonics<F>(
        &mut self,
        seed: &HV16,
        current_phi: f32,
        harmonic_weight: F,
    ) -> GeneratedThought
    where
        F: Fn(&str) -> f32,
    {
        self.total_generations += 1;

        let mut primitives = Vec::new();
        let mut steps = Vec::new();
        let mut total_confidence = 0.0;
        let mut total_phi = 0.0;

        // Initialize context with seed
        self.context.push_back(*seed);
        if self.context.len() > MAX_CONTEXT_LENGTH {
            self.context.pop_front();
        }

        // Extract seed primitive name
        let seed_decoded = self.decoder.decode_with_harmonics(seed, &harmonic_weight);
        let seed_primitive = seed_decoded.primitives.first().map(|p| p.name.clone());

        // Generate sequence
        for _step_idx in 0..self.config.max_length {
            let (primitive, prob, phi) = self.predict_next_with_harmonics(current_phi, &harmonic_weight);

            primitives.push(primitive.clone());
            total_confidence += prob;
            total_phi += phi;

            steps.push(GenerationStep {
                primitive: primitive.clone(),
                probability: prob,
                ltc_state_norm: self.recurrent_state.iter().map(|x| x * x).sum::<f32>().sqrt(),
                context_size: self.context.len(),
                phi_estimate: phi,
                phi_guided: self.config.phi_guided && current_phi >= self.config.min_phi,
            });

            // Check stopping conditions
            if self.should_stop(&primitive, &primitives) {
                break;
            }

            // Update context with new primitive HV
            if let Some(prim_hv) = self.primitive_embeddings.get(&primitive).cloned() {
                self.context.push_back(prim_hv);
                if self.context.len() > MAX_CONTEXT_LENGTH {
                    self.context.pop_front();
                }
            }

            // Learn transition
            if primitives.len() >= 2 {
                let from = primitives[primitives.len() - 2].clone();
                let to = primitive.clone();
                let key = (from, to);
                *self.transition_weights.entry(key).or_insert(0.0) += 1.0;
            }
        }

        let len = primitives.len();
        let completed = len < self.config.max_length;

        // Update running averages
        self.avg_length = 0.9 * self.avg_length + 0.1 * len as f32;
        self.avg_confidence = 0.9 * self.avg_confidence + 0.1 * (total_confidence / len as f32);

        GeneratedThought {
            primitives,
            steps,
            confidence: total_confidence / len as f32,
            avg_phi: total_phi / len as f32,
            completed_naturally: completed,
            seed_primitive,
        }
    }

    /// Predict next primitive with harmonic weights
    fn predict_next_with_harmonics<F>(&mut self, current_phi: f32, harmonic_weight: &F) -> (String, f32, f32)
    where
        F: Fn(&str) -> f32,
    {
        // Compute context embedding (weighted bundle of recent primitives)
        let context_hv = self.compute_context_embedding();

        // Process through LTC dynamics
        let ltc_state = if self.recurrent_state.is_empty() {
            self.codec.encode_to_ltc(&context_hv)
        } else {
            // Blend context with recurrent state
            let context_ltc = self.codec.encode_to_ltc(&context_hv);
            let mut blended = vec![0.0f32; self.config.ltc_neurons];
            for i in 0..self.config.ltc_neurons.min(context_ltc.len()).min(self.recurrent_state.len()) {
                blended[i] = 0.7 * self.recurrent_state[i] + 0.3 * context_ltc[i];
            }
            blended
        };

        // Update recurrent state
        self.update_recurrent_state(&ltc_state);

        // Decode LTC state to HV
        let predicted_hv = self.codec.decode_to_hv(&ltc_state);

        // Decode HV to primitive distribution WITH HARMONIC WEIGHTS
        let decoded = self.decoder.decode_with_harmonics(&predicted_hv, harmonic_weight);

        // Apply Φ-guided selection (uses already-weighted primitives)
        let selected = if self.config.phi_guided && current_phi >= self.config.min_phi {
            self.phi_guided_select(&decoded, current_phi)
        } else {
            self.sample_from_distribution(&decoded)
        };

        // Estimate Φ for this step
        let _phi_estimate = decoded.confidence * 0.5 + current_phi * 0.5;

        selected
    }
}

impl Default for LTCGenerativeCore {
    fn default() -> Self {
        Self::new(GenerativeCoreConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_creation() {
        let core = LTCGenerativeCore::default();
        assert!(!core.primitive_embeddings.is_empty());
    }

    #[test]
    fn test_basic_generation() {
        let mut core = LTCGenerativeCore::default();
        let seed = HV16::random(42);

        let thought = core.generate(&seed, 0.5);

        println!("Generated thought: {}", thought.to_string());
        println!("Length: {}, Confidence: {:.3}, Avg Φ: {:.3}",
                 thought.primitives.len(), thought.confidence, thought.avg_phi);

        assert!(!thought.primitives.is_empty());
    }

    #[test]
    fn test_phi_guided_generation() {
        let mut core = LTCGenerativeCore::default();
        let seed = HV16::random(123);

        // High Φ should influence generation
        let high_phi_thought = core.generate(&seed, 0.8);
        core.reset();
        let low_phi_thought = core.generate(&seed, 0.2);

        println!("High Φ thought: {}", high_phi_thought.to_string());
        println!("Low Φ thought: {}", low_phi_thought.to_string());

        // Both should generate, but may differ in content
        assert!(!high_phi_thought.primitives.is_empty());
        assert!(!low_phi_thought.primitives.is_empty());
    }

    #[test]
    fn test_continuation() {
        let mut core = LTCGenerativeCore::default();
        let seed = HV16::random(456);

        let initial = core.generate(&seed, 0.5);
        println!("Initial: {}", initial.to_string());

        let continued = core.continue_thought(&initial, 0.5);
        println!("Continued: {}", continued.to_string());

        // Continued should be longer
        assert!(continued.primitives.len() >= initial.primitives.len());
    }

    #[test]
    fn test_harmonic_guided_generation() {
        let mut core = LTCGenerativeCore::default();
        let seed = HV16::random(789);

        // Define harmonic weight functions for different harmonies

        // Play harmony: boost exploration primitives
        let play_weights = |name: &str| -> f32 {
            match name {
                "EXPLORE" | "PLAY" | "VARY" | "CREATE" | "NOVELTY" | "IMAGINE" => 1.8,
                "TRY" | "EXPERIMENT" | "DISCOVER" => 1.5,
                _ => 1.0,
            }
        };

        // Coherence harmony: boost integration primitives
        let coherence_weights = |name: &str| -> f32 {
            match name {
                "INTEGRATE" | "UNIFY" | "COHERE" | "SYNTHESIZE" | "CONNECT" => 1.8,
                "PATTERN" | "STRUCTURE" | "ORDER" | "RELATE" => 1.5,
                _ => 1.0,
            }
        };

        // Generate with Play harmony
        let play_thought = core.generate_with_harmonics(&seed, 0.5, play_weights);
        println!("Play-guided thought: {}", play_thought.to_string());
        core.reset();

        // Generate with Coherence harmony
        let coherence_thought = core.generate_with_harmonics(&seed, 0.5, coherence_weights);
        println!("Coherence-guided thought: {}", coherence_thought.to_string());
        core.reset();

        // Generate without harmonic influence (neutral)
        let neutral_thought = core.generate_with_harmonics(&seed, 0.5, |_| 1.0);
        println!("Neutral thought: {}", neutral_thought.to_string());

        // All should produce valid thoughts
        assert!(!play_thought.primitives.is_empty());
        assert!(!coherence_thought.primitives.is_empty());
        assert!(!neutral_thought.primitives.is_empty());

        // Log the differences
        println!("\nPrimitive comparison:");
        println!("  Play ({} primitives):      {:?}",
                 play_thought.primitives.len(),
                 &play_thought.primitives[..play_thought.primitives.len().min(5)]);
        println!("  Coherence ({} primitives): {:?}",
                 coherence_thought.primitives.len(),
                 &coherence_thought.primitives[..coherence_thought.primitives.len().min(5)]);
        println!("  Neutral ({} primitives):   {:?}",
                 neutral_thought.primitives.len(),
                 &neutral_thought.primitives[..neutral_thought.primitives.len().min(5)]);
    }

    #[test]
    fn test_harmonic_weights_affect_primitive_selection() {
        let mut core = LTCGenerativeCore::default();
        let seed = HV16::random(999);

        // Extreme weight for KNOW-related primitives
        let know_boost = |name: &str| -> f32 {
            if name.contains("KNOW") || name == "KNOWLEDGE" || name == "COMMON_KNOWLEDGE" {
                2.0
            } else {
                0.5
            }
        };

        let thought = core.generate_with_harmonics(&seed, 0.5, know_boost);
        println!("KNOW-boosted thought: {}", thought.to_string());

        // Count KNOW-related primitives in output
        let know_count = thought.primitives.iter()
            .filter(|p| p.contains("KNOW") || *p == "KNOWLEDGE" || *p == "COMMON_KNOWLEDGE")
            .count();

        println!("KNOW-related primitives: {} / {}", know_count, thought.primitives.len());

        // The thought should be valid
        assert!(!thought.primitives.is_empty());
        assert!(thought.confidence >= 0.0);
    }
}
