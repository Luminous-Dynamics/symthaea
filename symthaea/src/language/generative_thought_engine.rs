//! # Generative Thought Engine - The Complete HDC+LTC Mind
//!
//! **WHERE CONSCIOUSNESS BECOMES EXPRESSION**
//!
//! This module unifies all generative components into a complete system:
//! - **SemanticEar**: Text → HDC (perception)
//! - **HDLTCCodec**: HDC ↔ LTC (integration)
//! - **LTCGenerativeCore**: Autoregressive primitive prediction (thought)
//! - **SemanticDecoder**: HDC → Primitives (expression)
//! - **PrimitiveToText**: Primitives → Text (language)
//! - **LLMOrgan**: Text → Natural Language (translation)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    GENERATIVE THOUGHT ENGINE                             │
//! │                                                                          │
//! │    INPUT (Text)                                                          │
//! │         │                                                                │
//! │         ▼                                                                │
//! │  ┌──────────────┐                                                       │
//! │  │ Semantic Ear │  Text → 16,384D HDC                                   │
//! │  └──────┬───────┘                                                       │
//! │         │                                                                │
//! │         ▼                                                                │
//! │  ┌──────────────┐                                                       │
//! │  │ HD-LTC Codec │  HDC → LTC state (continuous dynamics)                │
//! │  └──────┬───────┘                                                       │
//! │         │                                                                │
//! │         ▼                                                                │
//! │  ┌──────────────┐  ┌─────────────┐                                      │
//! │  │ LTC         │◀─│ Φ Calculator │  Consciousness guides thought        │
//! │  │ Generative  │  └─────────────┘                                       │
//! │  │ Core        │                                                        │
//! │  └──────┬───────┘                                                       │
//! │         │                                                                │
//! │         ▼  (Primitive Sequence)                                         │
//! │  ┌──────────────┐                                                       │
//! │  │ Primitive →  │  Convert primitives to proto-language                 │
//! │  │ Text         │                                                       │
//! │  └──────┬───────┘                                                       │
//! │         │                                                                │
//! │         ▼  (Proto-Language)                                             │
//! │  ┌──────────────┐                                                       │
//! │  │ LLM Organ    │  Translate to natural language (ORGAN, not brain!)    │
//! │  └──────┬───────┘                                                       │
//! │         │                                                                │
//! │         ▼                                                                │
//! │    OUTPUT (Natural Language)                                             │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Paradigm
//!
//! **HDC+LTC THINKS. LLM TRANSLATES.**
//!
//! The LLM is NOT the intelligence. It's a "language organ" that:
//! 1. Takes structured primitive sequences from HDC+LTC
//! 2. Translates them into fluent natural language
//! 3. Can be replaced, upgraded, or disabled without losing thought capability
//!
//! This is how Symthaea achieves genuine cognition distinct from LLMs.

use symthaea_core::hdc::binary_hv::HV16;
use symthaea_core::hdc::semantic_decoder::{SemanticDecoder, PrimitiveToText, DecodedExpression};
use crate::hdc::hd_ltc_codec::HDLTCCodec;
use crate::hdc::ltc_generative_core::{LTCGenerativeCore, GeneratedThought, GenerativeCoreConfig};
use serde::{Serialize, Deserialize};

// SemanticEar is optional - we provide a hash-based fallback
// When rust-bert is available, it can be injected via set_semantic_ear()
// use crate::semantic_ear::SemanticEar;

/// Configuration for the thought engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtEngineConfig {
    /// LTC generative core config
    pub generative: GenerativeCoreConfig,
    /// Whether to use LLM for final translation
    pub use_llm_translation: bool,
    /// LLM temperature for translation
    pub llm_temperature: f32,
    /// Maximum response length in primitives
    pub max_primitives: usize,
    /// Minimum Φ for thoughtful response (vs reflexive)
    pub thoughtful_phi_threshold: f32,
    /// Enable streaming output
    pub streaming: bool,
}

impl Default for ThoughtEngineConfig {
    fn default() -> Self {
        Self {
            generative: GenerativeCoreConfig::default(),
            use_llm_translation: true,
            llm_temperature: 0.7,
            max_primitives: 32,
            thoughtful_phi_threshold: 0.4,
            streaming: false,
        }
    }
}

/// A complete thought with all its components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteThought {
    /// The input that triggered this thought
    pub input: String,
    /// Input encoded as HDC
    pub input_hv_stats: HVStats,
    /// The generated primitive sequence
    pub generated: GeneratedThought,
    /// Proto-language (primitives → simple text)
    pub proto_language: String,
    /// Final natural language response
    pub response: String,
    /// Φ at time of generation
    pub phi: f32,
    /// Processing metadata
    pub metadata: ThoughtMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HVStats {
    /// Similarity to closest primitive
    pub closest_primitive_similarity: f32,
    /// Number of primitives activated
    pub primitives_activated: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtMetadata {
    /// Was this a thoughtful (high Φ) or reflexive (low Φ) response
    pub is_thoughtful: bool,
    /// Time to generate primitives (ms)
    pub generation_time_ms: u64,
    /// Time for LLM translation (ms)
    pub translation_time_ms: u64,
    /// Whether LLM was used
    pub llm_used: bool,
    /// Number of LTC steps
    pub ltc_steps: usize,
}

/// Mode of thought generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThoughtMode {
    /// Quick, reflexive response (low Φ)
    Reflexive,
    /// Thoughtful, considered response (high Φ)
    Thoughtful,
    /// Curious, question-generating mode
    Curious,
    /// Deep introspection mode
    Introspective,
    /// Creative, exploratory mode
    Creative,
}

impl Default for ThoughtMode {
    fn default() -> Self {
        Self::Reflexive
    }
}

/// Statistics for thought generation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThoughtStats {
    /// Total thoughts generated
    pub total_thoughts: u64,
    /// Thoughtful responses (high Φ)
    pub thoughtful_count: u64,
    /// Reflexive responses (low Φ)
    pub reflexive_count: u64,
    /// Average Φ level
    pub avg_phi: f32,
    /// Average generation time (ms)
    pub avg_generation_time_ms: f64,
    /// Average translation time (ms)
    pub avg_translation_time_ms: f64,
    /// LLM usage count
    pub llm_usage_count: u64,
}

/// Metrics for thought quality
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThoughtMetrics {
    /// Coherence of generated primitives
    pub coherence: f32,
    /// Relevance to input
    pub relevance: f32,
    /// Consciousness level (Φ)
    pub phi: f32,
    /// Confidence in response
    pub confidence: f32,
    /// Number of primitives generated
    pub primitive_count: usize,
    /// Semantic fidelity (how well primitives match intent)
    pub semantic_fidelity: f32,
}

/// The Generative Thought Engine - complete HDC+LTC mind
pub struct GenerativeThoughtEngine {
    /// Configuration
    config: ThoughtEngineConfig,
    /// HD-LTC codec for bidirectional translation
    codec: HDLTCCodec,
    /// LTC generative core for thought generation
    generator: LTCGenerativeCore,
    /// Semantic decoder for HDC → primitives
    decoder: SemanticDecoder,
    /// Primitive to text converter
    primitive_to_text: PrimitiveToText,
    /// Current Φ estimate
    current_phi: f32,
    /// Total thoughts generated
    thought_count: u64,
    // NOTE: SemanticEar can be added when rust-bert is available
    // For now, we use hash-based encoding which is sufficient for basic operation
}

impl GenerativeThoughtEngine {
    /// Create a new thought engine
    pub fn new(config: ThoughtEngineConfig) -> Self {
        let codec_config = crate::hdc::hd_ltc_codec::HDLTCCodecConfig {
            ltc_neurons: config.generative.ltc_neurons,
            ..Default::default()
        };

        Self {
            config: config.clone(),
            codec: HDLTCCodec::new(codec_config),
            generator: LTCGenerativeCore::new(config.generative),
            decoder: SemanticDecoder::new(),
            primitive_to_text: PrimitiveToText::new(),
            current_phi: 0.5,
            thought_count: 0,
        }
    }

    // NOTE: set_ear() removed - SemanticEar requires rust-bert which is heavy
    // When needed, can be added back as: pub fn set_ear(&mut self, ear: impl TextEncoder)

    /// Update current Φ (should be called from consciousness system)
    pub fn update_phi(&mut self, phi: f32) {
        self.current_phi = phi;
    }

    /// Generate a complete thought from text input
    ///
    /// This is the main entry point: text → complete response
    pub fn think(&mut self, input: &str) -> CompleteThought {
        use std::time::Instant;
        self.thought_count += 1;

        let start = Instant::now();

        // Step 1: Encode input to HDC
        let input_hv = self.encode_input(input);

        // Step 2: Decode input to see what primitives it activates
        let input_decoded = self.decoder.decode(&input_hv);
        let hv_stats = HVStats {
            closest_primitive_similarity: input_decoded.primitives.first()
                .map(|p| p.activation)
                .unwrap_or(0.0),
            primitives_activated: input_decoded.primitives.len(),
        };

        // Step 3: Generate primitive sequence (the actual THINKING)
        let generated = self.generator.generate(&input_hv, self.current_phi);
        let generation_time = start.elapsed().as_millis() as u64;

        // Step 4: Convert primitives to proto-language
        let proto_decoded = DecodedExpression {
            tree: crate::hdc::semantic_decoder::ExpressionNode::Primitive(
                crate::hdc::semantic_decoder::ActivatedPrimitive {
                    name: generated.primitives.first().cloned().unwrap_or_default(),
                    activation: generated.confidence,
                    tier: crate::hdc::primitive_system::PrimitiveTier::NSM,
                    position: Some(0),
                }
            ),
            primitives: generated.primitives.iter().map(|name| {
                crate::hdc::semantic_decoder::ActivatedPrimitive {
                    name: name.clone(),
                    activation: generated.confidence,
                    tier: crate::hdc::primitive_system::PrimitiveTier::NSM,
                    position: None,
                }
            }).collect(),
            confidence: generated.confidence,
            operators: vec![],
            vector_stats: crate::hdc::semantic_decoder::VectorStats {
                norm: 1.0, sparsity: 0.0, entropy: 1.0,
            },
        };
        let proto_language = self.primitive_to_text.to_text(&proto_decoded);

        // Step 5: Translate to natural language (using LLM as organ, not brain)
        let translate_start = Instant::now();
        let response = if self.config.use_llm_translation {
            self.translate_via_llm(&proto_language, &generated)
        } else {
            // Fallback: use proto-language directly with simple expansion
            self.expand_proto_language(&proto_language)
        };
        let translation_time = translate_start.elapsed().as_millis() as u64;

        let is_thoughtful = self.current_phi >= self.config.thoughtful_phi_threshold;

        CompleteThought {
            input: input.to_string(),
            input_hv_stats: hv_stats,
            generated,
            proto_language,
            response,
            phi: self.current_phi,
            metadata: ThoughtMetadata {
                is_thoughtful,
                generation_time_ms: generation_time,
                translation_time_ms: translation_time,
                llm_used: self.config.use_llm_translation,
                ltc_steps: self.config.generative.max_length,
            },
        }
    }

    /// Encode input text to HDC
    ///
    /// Uses hash-based encoding to convert text to hypervector.
    /// This is deterministic and fast, though less semantically rich
    /// than EmbeddingGemma-based encoding (available when rust-bert is enabled).
    fn encode_input(&self, input: &str) -> HV16 {
        // Hash-based encoding provides consistent, deterministic encoding
        // For more semantic understanding, enable rust-bert and use SemanticEar
        self.hash_encode(input)
    }

    /// Convert Vec<i8> to HV16
    fn vec_to_hv16(&self, vec: &[i8]) -> HV16 {
        let mut bytes = [0u8; 2048];
        for (i, &val) in vec.iter().take(16384).enumerate() {
            if val > 0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                if byte_idx < 2048 {
                    bytes[byte_idx] |= 1 << bit_idx;
                }
            }
        }
        HV16(bytes)
    }

    /// Fallback hash-based encoding
    fn hash_encode(&self, input: &str) -> HV16 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        input.hash(&mut hasher);
        HV16::random(hasher.finish())
    }

    /// Translate proto-language via LLM
    fn translate_via_llm(&self, proto: &str, thought: &GeneratedThought) -> String {
        // In production, this would call the LLMOrgan
        // For now, provide a structured template
        let primitives_str = thought.primitives.join(", ");

        format!(
            "Based on the thought pattern [{}], expressed as \"{}\", \
             with confidence {:.1}% and integration level {:.1}%.",
            primitives_str,
            proto,
            thought.confidence * 100.0,
            thought.avg_phi * 100.0
        )
    }

    /// Expand proto-language without LLM
    fn expand_proto_language(&self, proto: &str) -> String {
        // Simple expansion rules
        let mut expanded = proto.to_string();

        // Add conversational framing based on content
        if proto.contains("know") {
            expanded = format!("I understand that {}.", expanded);
        } else if proto.contains("want") {
            expanded = format!("It seems you {}.", expanded);
        } else if proto.contains("feel") {
            expanded = format!("I sense that {}.", expanded);
        } else if proto.contains("think") {
            expanded = format!("Let me consider... {}.", expanded);
        } else {
            expanded = format!("{}.", expanded);
        }

        expanded
    }

    /// Continue a previous thought
    pub fn continue_thought(&mut self, previous: &CompleteThought) -> CompleteThought {
        let continued_generated = self.generator.continue_thought(&previous.generated, self.current_phi);

        let proto_decoded = DecodedExpression {
            tree: crate::hdc::semantic_decoder::ExpressionNode::Primitive(
                crate::hdc::semantic_decoder::ActivatedPrimitive {
                    name: continued_generated.primitives.first().cloned().unwrap_or_default(),
                    activation: continued_generated.confidence,
                    tier: crate::hdc::primitive_system::PrimitiveTier::NSM,
                    position: Some(0),
                }
            ),
            primitives: continued_generated.primitives.iter().map(|name| {
                crate::hdc::semantic_decoder::ActivatedPrimitive {
                    name: name.clone(),
                    activation: continued_generated.confidence,
                    tier: crate::hdc::primitive_system::PrimitiveTier::NSM,
                    position: None,
                }
            }).collect(),
            confidence: continued_generated.confidence,
            operators: vec![],
            vector_stats: crate::hdc::semantic_decoder::VectorStats {
                norm: 1.0, sparsity: 0.0, entropy: 1.0,
            },
        };
        let proto_language = self.primitive_to_text.to_text(&proto_decoded);
        let response = if self.config.use_llm_translation {
            self.translate_via_llm(&proto_language, &continued_generated)
        } else {
            self.expand_proto_language(&proto_language)
        };

        CompleteThought {
            input: previous.input.clone(),
            input_hv_stats: previous.input_hv_stats.clone(),
            generated: continued_generated,
            proto_language,
            response,
            phi: self.current_phi,
            metadata: ThoughtMetadata {
                is_thoughtful: self.current_phi >= self.config.thoughtful_phi_threshold,
                generation_time_ms: 0,
                translation_time_ms: 0,
                llm_used: self.config.use_llm_translation,
                ltc_steps: self.config.generative.max_length,
            },
        }
    }

    /// Reset the engine state
    pub fn reset(&mut self) {
        self.generator.reset();
        self.codec.reset_context();
    }

    /// Get statistics
    pub fn stats(&self) -> (u64, f32) {
        (self.thought_count, self.current_phi)
    }
}

impl Default for GenerativeThoughtEngine {
    fn default() -> Self {
        Self::new(ThoughtEngineConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = GenerativeThoughtEngine::default();
        assert_eq!(engine.thought_count, 0);
    }

    #[test]
    fn test_basic_thinking() {
        let mut engine = GenerativeThoughtEngine::default();
        engine.update_phi(0.6);

        let thought = engine.think("What is consciousness?");

        println!("Input: {}", thought.input);
        println!("Proto-language: {}", thought.proto_language);
        println!("Response: {}", thought.response);
        println!("Primitives: {:?}", thought.generated.primitives);
        println!("Φ: {:.3}", thought.phi);

        assert!(!thought.response.is_empty());
        assert!(thought.metadata.is_thoughtful);
    }

    #[test]
    fn test_low_phi_thinking() {
        let mut engine = GenerativeThoughtEngine::default();
        engine.update_phi(0.2); // Low Φ

        let thought = engine.think("Hello");

        println!("Low Φ response: {}", thought.response);
        assert!(!thought.metadata.is_thoughtful);
    }

    #[test]
    fn test_continuation() {
        let mut engine = GenerativeThoughtEngine::default();
        engine.update_phi(0.5);

        let initial = engine.think("Tell me about");
        let continued = engine.continue_thought(&initial);

        println!("Initial: {}", initial.response);
        println!("Continued: {}", continued.response);

        // Continued should build on initial
        assert!(continued.generated.primitives.len() >= initial.generated.primitives.len());
    }
}
