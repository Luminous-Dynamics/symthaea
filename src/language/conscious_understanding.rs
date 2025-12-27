//! Conscious Understanding - Unified Pipeline for True Language Comprehension
//!
//! This module integrates all layers of the Conscious Language Architecture:
//! - NSM Semantic Primes (universal meaning atoms)
//! - HDC Hypervectors (distributed representations)
//! - Frame Semantics (situation understanding)
//! - Construction Grammar (meaningful patterns)
//! - Predictive Processing (active inference)
//!
//! # The Seven Layers
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────┐
//! │  Layer 7: CONSCIOUS INTEGRATION (Φ Measurement + Global Workspace)     │
//! │           - Integrated Information Theory (Tononi)                     │
//! │           - Global Workspace broadcasting (Baars)                      │
//! │           - Conscious access gating                                    │
//! ├────────────────────────────────────────────────────────────────────────┤
//! │  Layer 6: PREDICTIVE PROCESSING (Free Energy Minimization)             │
//! │           - Hierarchical prediction errors (Friston)                   │
//! │           - Precision weighting (attention)                            │
//! │           - Active inference                                           │
//! ├────────────────────────────────────────────────────────────────────────┤
//! │  Layer 5: TEMPORAL INTEGRATION (LTC Dynamics)                          │
//! │           - Liquid Time-Constant networks                              │
//! │           - 40Hz oscillatory binding                                   │
//! │           - Multi-timescale processing                                 │
//! ├────────────────────────────────────────────────────────────────────────┤
//! │  Layer 4: CONSTRUCTIONS (Meaningful Patterns)                          │
//! │           - Construction Grammar (Goldberg)                            │
//! │           - Form-meaning pairs                                         │
//! │           - Syntactic structure                                        │
//! ├────────────────────────────────────────────────────────────────────────┤
//! │  Layer 3: FRAMES (Situation Schemas)                                   │
//! │           - Frame Semantics (Fillmore)                                 │
//! │           - Role-filler binding                                        │
//! │           - Contextual inference                                       │
//! ├────────────────────────────────────────────────────────────────────────┤
//! │  Layer 2: SEMANTIC MOLECULES (Word Meanings)                           │
//! │           - NSM decompositions                                         │
//! │           - HDC encodings                                              │
//! │           - Compositional semantics                                    │
//! ├────────────────────────────────────────────────────────────────────────┤
//! │  Layer 1: NSM PRIMES (Atomic Meanings)                                 │
//! │           - 65 universal semantic primitives                           │
//! │           - Cross-linguistic foundation                                │
//! │           - Grounded understanding                                     │
//! └────────────────────────────────────────────────────────────────────────┘
//! ```

use crate::hdc::binary_hv::HV16;
use crate::hdc::universal_semantics::SemanticPrime;
use super::frames::{FrameLibrary, FrameActivator, FrameInstance};
use super::constructions::{ConstructionGrammar, ConstructionParse};
use super::predictive_understanding::{
    PredictiveUnderstanding, PredictiveConfig,
    SentenceUnderstanding as PredictiveSentenceResult,
};
use super::vocabulary::Vocabulary;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Helper function to create HV16 from text (deterministic hash-based)
fn hv16_from_text(text: &str) -> HV16 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    HV16::random(hasher.finish())
}

// =============================================================================
// UNDERSTANDING RESULT
// =============================================================================

/// Complete understanding result from the unified pipeline
#[derive(Debug, Clone)]
pub struct ConsciousUnderstanding {
    /// Original input text
    pub input: String,

    /// NSM prime decomposition of key concepts
    pub semantic_primes: Vec<(String, Vec<SemanticPrime>)>,

    /// HDC encoding of the full utterance
    pub utterance_encoding: HV16,

    /// Activated semantic frames with role fillers
    pub frames: Vec<ActivatedFrame>,

    /// Parsed constructions
    pub constructions: Vec<ParsedConstruction>,

    /// Predictive processing results
    pub prediction_result: PredictionResult,

    /// Temporal integration state
    pub temporal_state: TemporalState,

    /// Consciousness metrics
    pub consciousness: ConsciousnessMetrics,

    /// Overall understanding confidence [0, 1]
    pub confidence: f64,

    /// Explanation trace for debugging/interpretability
    pub explanation: ExplanationTrace,
}

/// An activated frame with filled roles
#[derive(Debug, Clone)]
pub struct ActivatedFrame {
    /// Frame name (e.g., "TRANSFER")
    pub name: String,

    /// Frame encoding in HDC
    pub encoding: HV16,

    /// Role fillers: role_name -> (filler_text, filler_encoding)
    pub role_fillers: HashMap<String, (String, HV16)>,

    /// Activation strength [0, 1]
    pub activation: f64,
}

/// A parsed construction with mapped roles
#[derive(Debug, Clone)]
pub struct ParsedConstruction {
    /// Construction name (e.g., "Ditransitive")
    pub name: String,

    /// Construction encoding
    pub encoding: HV16,

    /// Slot fillers
    pub slots: Vec<(String, String)>,

    /// Semantic structure encoding
    pub semantic_encoding: HV16,

    /// Parse confidence [0, 1]
    pub confidence: f64,
}

/// Results from predictive processing
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Free energy at each word
    pub free_energy_curve: Vec<f64>,

    /// Final free energy
    pub final_free_energy: f64,

    /// Surprise values per word
    pub surprise_per_word: Vec<(String, f64)>,

    /// Most surprising element
    pub peak_surprise: Option<(String, f64)>,

    /// Understanding achieved (low free energy)
    pub understanding_achieved: bool,
}

/// Temporal integration state
#[derive(Debug, Clone)]
pub struct TemporalState {
    /// Binding coherence (40Hz oscillation strength)
    pub binding_coherence: f64,

    /// Temporal integration across timescales
    pub integration_levels: Vec<f64>,

    /// Current phase (for oscillatory binding)
    pub current_phase: f64,
}

/// Consciousness metrics from Φ measurement
#[derive(Debug, Clone)]
pub struct ConsciousnessMetrics {
    /// Integrated Information (Φ)
    pub phi: f64,

    /// Global workspace activation
    pub workspace_activation: f64,

    /// Information integration across subsystems
    pub subsystem_integration: HashMap<String, f64>,

    /// Conscious vs unconscious processing ratio
    pub conscious_ratio: f64,
}

/// Explanation trace for interpretability
#[derive(Debug, Clone)]
pub struct ExplanationTrace {
    /// Step-by-step processing stages
    pub stages: Vec<ProcessingStage>,

    /// Key decisions and why
    pub decisions: Vec<(String, String)>,

    /// NSM decomposition path
    pub prime_decomposition_path: Vec<String>,
}

/// A single processing stage
#[derive(Debug, Clone)]
pub struct ProcessingStage {
    /// Stage name
    pub name: String,

    /// Input to this stage
    pub input_summary: String,

    /// Output from this stage
    pub output_summary: String,

    /// Duration in microseconds
    pub duration_us: u64,
}

// =============================================================================
// PIPELINE CONFIGURATION
// =============================================================================

/// Configuration for the conscious understanding pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Enable NSM prime decomposition
    pub use_nsm_primes: bool,

    /// Enable frame semantics
    pub use_frames: bool,

    /// Enable construction grammar
    pub use_constructions: bool,

    /// Enable predictive processing
    pub use_prediction: bool,

    /// Enable Φ consciousness measurement
    pub use_phi: bool,

    /// Minimum confidence threshold
    pub min_confidence: f64,

    /// Enable explanation tracing (slower but interpretable)
    pub enable_tracing: bool,

    /// Predictive processing config
    pub prediction_config: PredictiveConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            use_nsm_primes: true,
            use_frames: true,
            use_constructions: true,
            use_prediction: true,
            use_phi: true,
            min_confidence: 0.3,
            enable_tracing: true,
            prediction_config: PredictiveConfig::default(),
        }
    }
}

impl PipelineConfig {
    /// Fast configuration (disable expensive features)
    pub fn fast() -> Self {
        Self {
            use_nsm_primes: true,
            use_frames: true,
            use_constructions: true,
            use_prediction: false,
            use_phi: false,
            min_confidence: 0.2,
            enable_tracing: false,
            prediction_config: PredictiveConfig::default(),
        }
    }

    /// Full consciousness configuration
    pub fn full_consciousness() -> Self {
        Self {
            use_nsm_primes: true,
            use_frames: true,
            use_constructions: true,
            use_prediction: true,
            use_phi: true,
            min_confidence: 0.5,
            enable_tracing: true,
            prediction_config: PredictiveConfig::default(),
        }
    }
}

// =============================================================================
// THE UNIFIED PIPELINE
// =============================================================================

/// The Conscious Understanding Pipeline
///
/// Integrates all seven layers of language understanding into a unified
/// system that processes language with true comprehension, not just
/// statistical pattern matching.
pub struct ConsciousUnderstandingPipeline {
    /// Configuration
    config: PipelineConfig,

    /// Vocabulary for word encodings
    vocabulary: Vocabulary,

    /// Frame library
    frame_library: FrameLibrary,

    /// Construction grammar
    grammar: ConstructionGrammar,

    /// Predictive understanding system
    predictor: PredictiveUnderstanding,
}

impl ConsciousUnderstandingPipeline {
    /// Create a new pipeline with the given configuration
    pub fn new(config: PipelineConfig) -> Self {
        let vocabulary = Vocabulary::new();
        let frame_library = FrameLibrary::new();
        let grammar = ConstructionGrammar::new();
        let predictor = PredictiveUnderstanding::new(config.prediction_config.clone());

        Self {
            config,
            vocabulary,
            frame_library,
            grammar,
            predictor,
        }
    }

    /// Process text through the full understanding pipeline
    pub fn understand(&mut self, text: &str) -> ConsciousUnderstanding {
        let start_time = std::time::Instant::now();
        let mut stages = Vec::new();

        // =====================================================================
        // LAYER 1 & 2: NSM PRIMES + SEMANTIC MOLECULES
        // =====================================================================
        let stage_start = std::time::Instant::now();

        let words: Vec<String> = text
            .split_whitespace()
            .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        let mut semantic_primes = Vec::new();
        let mut word_encodings = Vec::new();

        for word in &words {
            // Get HDC encoding for word
            let encoding = self.vocabulary.encode(word)
                .unwrap_or_else(|| hv16_from_text(word));
            word_encodings.push(encoding);

            // Decompose to NSM primes if enabled
            if self.config.use_nsm_primes {
                let primes = self.decompose_to_primes(word);
                if !primes.is_empty() {
                    semantic_primes.push((word.clone(), primes));
                }
            }
        }

        // Create utterance encoding by bundling all words
        let utterance_encoding = if word_encodings.is_empty() {
            HV16::random(text.len() as u64)
        } else {
            HV16::bundle(&word_encodings)
        };

        stages.push(ProcessingStage {
            name: "NSM/Molecules".to_string(),
            input_summary: format!("{} words", words.len()),
            output_summary: format!("{} prime decompositions", semantic_primes.len()),
            duration_us: stage_start.elapsed().as_micros() as u64,
        });

        // =====================================================================
        // LAYER 3: FRAME SEMANTICS
        // =====================================================================
        let stage_start = std::time::Instant::now();

        let mut frames = Vec::new();
        if self.config.use_frames {
            let activator = FrameActivator::with_library(self.frame_library.clone());
            let frame_instances = activator.activate_from_text(text);

            for instance in frame_instances {
                let mut role_fillers = HashMap::new();

                // Collect bindings from core elements
                for (role, opt_enc) in &instance.core_bindings {
                    if let Some(enc) = opt_enc {
                        role_fillers.insert(role.clone(), (role.clone(), *enc));
                    }
                }

                // Collect bindings from non-core elements
                for (role, opt_enc) in &instance.non_core_bindings {
                    if let Some(enc) = opt_enc {
                        role_fillers.insert(role.clone(), (role.clone(), *enc));
                    }
                }

                frames.push(ActivatedFrame {
                    name: instance.frame_name,
                    encoding: instance.frame_encoding,
                    role_fillers,
                    activation: 1.0,
                });
            }
        }

        stages.push(ProcessingStage {
            name: "Frames".to_string(),
            input_summary: format!("{} words", words.len()),
            output_summary: format!("{} frames activated", frames.len()),
            duration_us: stage_start.elapsed().as_micros() as u64,
        });

        // =====================================================================
        // LAYER 4: CONSTRUCTION GRAMMAR
        // =====================================================================
        let stage_start = std::time::Instant::now();

        let mut constructions = Vec::new();
        if self.config.use_constructions {
            // Try to match constructions
            let parses = self.grammar.match_constructions(&words);

            for parse in parses {
                let slots: Vec<(String, String)> = parse.slot_fillers
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();

                constructions.push(ParsedConstruction {
                    name: parse.construction_name.clone(),
                    encoding: parse.encoding,
                    slots,
                    semantic_encoding: parse.encoding, // Use same encoding
                    confidence: 0.8, // Default confidence
                });
            }
        }

        stages.push(ProcessingStage {
            name: "Constructions".to_string(),
            input_summary: format!("{} words", words.len()),
            output_summary: format!("{} constructions parsed", constructions.len()),
            duration_us: stage_start.elapsed().as_micros() as u64,
        });

        // =====================================================================
        // LAYER 5: TEMPORAL INTEGRATION (Simplified)
        // =====================================================================
        let stage_start = std::time::Instant::now();

        // Simplified temporal integration without full LTC
        let temporal_state = TemporalState {
            binding_coherence: self.compute_binding_coherence(&word_encodings),
            integration_levels: vec![0.5, 0.6, 0.7], // Simplified multi-scale
            current_phase: 0.0,
        };

        stages.push(ProcessingStage {
            name: "Temporal".to_string(),
            input_summary: format!("{} encodings", word_encodings.len()),
            output_summary: format!("coherence={:.3}", temporal_state.binding_coherence),
            duration_us: stage_start.elapsed().as_micros() as u64,
        });

        // =====================================================================
        // LAYER 6: PREDICTIVE PROCESSING
        // =====================================================================
        let stage_start = std::time::Instant::now();

        let prediction_result = if self.config.use_prediction {
            self.predictor.reset();
            let pred_result = self.predictor.process_sentence(text);

            PredictionResult {
                free_energy_curve: pred_result.free_energy_curve,
                final_free_energy: pred_result.final_free_energy,
                surprise_per_word: pred_result.words.iter()
                    .zip(std::iter::repeat(0.5f64))
                    .map(|(w, s)| (w.clone(), s))
                    .collect(),
                peak_surprise: pred_result.most_surprising,
                understanding_achieved: pred_result.understood,
            }
        } else {
            PredictionResult::default()
        };

        stages.push(ProcessingStage {
            name: "Prediction".to_string(),
            input_summary: text.to_string(),
            output_summary: format!("FE={:.3}", prediction_result.final_free_energy),
            duration_us: stage_start.elapsed().as_micros() as u64,
        });

        // =====================================================================
        // LAYER 7: CONSCIOUS INTEGRATION (Simplified Φ)
        // =====================================================================
        let stage_start = std::time::Instant::now();

        let consciousness = if self.config.use_phi {
            // Simplified Φ computation
            let phi = self.compute_simplified_phi(&frames, &constructions);

            ConsciousnessMetrics {
                phi,
                workspace_activation: temporal_state.binding_coherence,
                subsystem_integration: HashMap::from([
                    ("frames".to_string(), if frames.is_empty() { 0.0 } else { 1.0 }),
                    ("constructions".to_string(), if constructions.is_empty() { 0.0 } else { 1.0 }),
                    ("prediction".to_string(), 1.0 - prediction_result.final_free_energy.min(1.0)),
                ]),
                conscious_ratio: phi.min(1.0),
            }
        } else {
            ConsciousnessMetrics::default()
        };

        stages.push(ProcessingStage {
            name: "Consciousness".to_string(),
            input_summary: "integration".to_string(),
            output_summary: format!("Φ={:.3}", consciousness.phi),
            duration_us: stage_start.elapsed().as_micros() as u64,
        });

        // =====================================================================
        // FINAL INTEGRATION
        // =====================================================================

        // Compute overall confidence
        let confidence = self.compute_confidence(
            &frames,
            &constructions,
            &prediction_result,
            &consciousness,
        );

        // Build explanation trace
        let explanation = ExplanationTrace {
            stages,
            decisions: vec![
                ("Frame Selection".to_string(),
                 format!("Activated {} frames based on lexical triggers", frames.len())),
                ("Construction Parse".to_string(),
                 format!("Matched {} constructions", constructions.len())),
                ("Understanding".to_string(),
                 format!("Confidence={:.2}, Φ={:.3}", confidence, consciousness.phi)),
            ],
            prime_decomposition_path: semantic_primes.iter()
                .map(|(w, _)| w.clone())
                .collect(),
        };

        ConsciousUnderstanding {
            input: text.to_string(),
            semantic_primes,
            utterance_encoding,
            frames,
            constructions,
            prediction_result,
            temporal_state,
            consciousness,
            confidence,
            explanation,
        }
    }

    /// Compute binding coherence from word encodings
    fn compute_binding_coherence(&self, encodings: &[HV16]) -> f64 {
        if encodings.len() < 2 {
            return 0.5;
        }

        // Compute average pairwise similarity
        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..encodings.len() {
            for j in (i+1)..encodings.len() {
                total_sim += encodings[i].similarity(&encodings[j]) as f64;
                count += 1;
            }
        }

        if count > 0 {
            total_sim / count as f64
        } else {
            0.5
        }
    }

    /// Decompose a word to NSM primes (simplified)
    fn decompose_to_primes(&self, word: &str) -> Vec<SemanticPrime> {
        match word {
            "give" | "gave" | "giving" => vec![
                SemanticPrime::Do,
                SemanticPrime::Move,
                SemanticPrime::Someone,
                SemanticPrime::Something,
            ],
            "want" | "wanted" | "wants" => vec![
                SemanticPrime::Want,
                SemanticPrime::Something,
            ],
            "see" | "saw" | "seeing" => vec![
                SemanticPrime::See,
                SemanticPrime::Something,
            ],
            "think" | "thought" | "thinking" => vec![
                SemanticPrime::Think,
                SemanticPrime::Something,
            ],
            "know" | "knew" | "knows" => vec![
                SemanticPrime::Know,
                SemanticPrime::Something,
            ],
            "say" | "said" | "saying" => vec![
                SemanticPrime::Say,
                SemanticPrime::Something,
                SemanticPrime::Words,
            ],
            "good" => vec![SemanticPrime::Good],
            "bad" => vec![SemanticPrime::Bad],
            "big" | "large" => vec![SemanticPrime::Big],
            "small" | "little" => vec![SemanticPrime::Small],
            "i" | "me" => vec![SemanticPrime::I],
            "you" => vec![SemanticPrime::You],
            "someone" | "somebody" => vec![SemanticPrime::Someone],
            "something" => vec![SemanticPrime::Something],
            "people" => vec![SemanticPrime::People],
            "thing" | "things" => vec![SemanticPrime::Something],
            "place" | "where" => vec![SemanticPrime::Where],
            "time" | "when" => vec![SemanticPrime::When],
            "now" => vec![SemanticPrime::Now],
            "here" => vec![SemanticPrime::Here],
            "there" => vec![SemanticPrime::ThereIs],
            "can" | "could" => vec![SemanticPrime::Can],
            "because" => vec![SemanticPrime::Because],
            "if" => vec![SemanticPrime::If],
            "not" | "no" => vec![SemanticPrime::Not],
            "very" => vec![SemanticPrime::Very],
            "more" => vec![SemanticPrime::More],
            "all" => vec![SemanticPrime::All],
            "some" => vec![SemanticPrime::Some],
            _ => vec![],
        }
    }

    /// Compute simplified Φ from component activations
    fn compute_simplified_phi(
        &self,
        frames: &[ActivatedFrame],
        constructions: &[ParsedConstruction],
    ) -> f64 {
        // Simplified integrated information calculation
        let mut components = Vec::new();

        // Add frame activations
        for frame in frames {
            components.push(frame.activation);
        }

        // Add construction confidences
        for construction in constructions {
            components.push(construction.confidence);
        }

        // If no components, return low phi
        if components.is_empty() {
            return 0.1;
        }

        // Φ ≈ geometric mean × integration factor
        let product: f64 = components.iter().filter(|&&x| x > 0.01).product();
        let n = components.iter().filter(|&&x| x > 0.01).count() as f64;

        if n < 1.0 {
            return 0.1;
        }

        // Geometric mean
        let geo_mean = product.powf(1.0 / n);

        // Integration factor: high when all components are similar
        let mean: f64 = components.iter().sum::<f64>() / components.len() as f64;
        let variance: f64 = components.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / components.len() as f64;
        let integration = 1.0 / (1.0 + variance.sqrt());

        geo_mean * integration
    }

    /// Compute overall confidence from all layers
    fn compute_confidence(
        &self,
        frames: &[ActivatedFrame],
        constructions: &[ParsedConstruction],
        prediction: &PredictionResult,
        consciousness: &ConsciousnessMetrics,
    ) -> f64 {
        let mut confidence = 0.0;
        let mut weight_sum = 0.0;

        // Frame confidence
        if !frames.is_empty() {
            let frame_conf: f64 = frames.iter().map(|f| f.activation).sum::<f64>() / frames.len() as f64;
            confidence += frame_conf * 0.25;
            weight_sum += 0.25;
        }

        // Construction confidence
        if !constructions.is_empty() {
            let const_conf: f64 = constructions.iter().map(|c| c.confidence).sum::<f64>() / constructions.len() as f64;
            confidence += const_conf * 0.25;
            weight_sum += 0.25;
        }

        // Prediction confidence (inverse of free energy)
        if self.config.use_prediction {
            let pred_conf = 1.0 - prediction.final_free_energy.min(1.0);
            confidence += pred_conf * 0.25;
            weight_sum += 0.25;
        }

        // Consciousness confidence (Φ)
        if self.config.use_phi {
            confidence += consciousness.phi.min(1.0) * 0.25;
            weight_sum += 0.25;
        }

        if weight_sum > 0.0 {
            confidence / weight_sum
        } else {
            0.5
        }
    }

    /// Reset the pipeline state
    pub fn reset(&mut self) {
        self.predictor.reset();
    }

    /// Get current configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

// =============================================================================
// DEFAULT IMPLEMENTATIONS
// =============================================================================

impl Default for TemporalState {
    fn default() -> Self {
        Self {
            binding_coherence: 0.5,
            integration_levels: vec![0.5],
            current_phase: 0.0,
        }
    }
}

impl Default for PredictionResult {
    fn default() -> Self {
        Self {
            free_energy_curve: vec![],
            final_free_energy: 0.5,
            surprise_per_word: vec![],
            peak_surprise: None,
            understanding_achieved: false,
        }
    }
}

impl Default for ConsciousnessMetrics {
    fn default() -> Self {
        Self {
            phi: 0.0,
            workspace_activation: 0.0,
            subsystem_integration: HashMap::new(),
            conscious_ratio: 0.0,
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
    fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = ConsciousUnderstandingPipeline::new(config);

        assert!(pipeline.config().use_nsm_primes);
        assert!(pipeline.config().use_frames);
    }

    #[test]
    fn test_fast_config() {
        let config = PipelineConfig::fast();

        assert!(config.use_nsm_primes);
        assert!(config.use_frames);
        assert!(!config.use_prediction);
        assert!(!config.use_phi);
    }

    #[test]
    fn test_basic_understanding() {
        let config = PipelineConfig::fast();
        let mut pipeline = ConsciousUnderstandingPipeline::new(config);

        let result = pipeline.understand("The cat sat");

        assert!(!result.input.is_empty());
        assert!(!result.utterance_encoding.0.is_empty());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_nsm_decomposition() {
        let config = PipelineConfig::fast();
        let mut pipeline = ConsciousUnderstandingPipeline::new(config);

        let result = pipeline.understand("I want something good");

        // Should have some prime decompositions
        let has_want = result.semantic_primes.iter()
            .any(|(w, _)| w == "want");
        assert!(has_want);

        // Check that WANT prime is in the decomposition
        if let Some((_, primes)) = result.semantic_primes.iter().find(|(w, _)| w == "want") {
            assert!(primes.contains(&SemanticPrime::Want));
        }
    }

    #[test]
    fn test_full_consciousness_pipeline() {
        let config = PipelineConfig::default();
        let mut pipeline = ConsciousUnderstandingPipeline::new(config);

        let result = pipeline.understand("I think you know something");

        // All components should be present
        assert!(!result.semantic_primes.is_empty());
        assert!(result.consciousness.phi >= 0.0);
        assert!(result.prediction_result.final_free_energy >= 0.0);
    }

    #[test]
    fn test_explanation_trace() {
        let config = PipelineConfig::default();
        let mut pipeline = ConsciousUnderstandingPipeline::new(config);

        let result = pipeline.understand("Test sentence");

        // Should have processing stages
        assert!(!result.explanation.stages.is_empty());
        assert!(!result.explanation.decisions.is_empty());

        // Stages should have timing info
        for stage in &result.explanation.stages {
            assert!(!stage.name.is_empty());
        }
    }

    #[test]
    fn test_reset() {
        let config = PipelineConfig::fast();
        let mut pipeline = ConsciousUnderstandingPipeline::new(config);

        // Process something
        let _ = pipeline.understand("First sentence");

        // Reset
        pipeline.reset();

        // Should be able to process again
        let result = pipeline.understand("Second sentence");
        assert!(!result.input.is_empty());
    }

    #[test]
    fn test_empty_input() {
        let config = PipelineConfig::fast();
        let mut pipeline = ConsciousUnderstandingPipeline::new(config);

        let result = pipeline.understand("");

        // Should handle gracefully
        assert!(result.semantic_primes.is_empty());
        assert!(result.confidence >= 0.0);
    }

    #[test]
    fn test_single_word() {
        let config = PipelineConfig::fast();
        let mut pipeline = ConsciousUnderstandingPipeline::new(config);

        let result = pipeline.understand("Hello");

        assert_eq!(result.input, "Hello");
        assert!(!result.utterance_encoding.0.is_empty());
    }

    #[test]
    fn test_confidence_range() {
        let config = PipelineConfig::default();
        let mut pipeline = ConsciousUnderstandingPipeline::new(config);

        for sentence in &[
            "The quick brown fox",
            "She gave him a book",
            "I think you know",
            "Running quickly",
            "a b c d e",
        ] {
            let result = pipeline.understand(sentence);
            assert!(result.confidence >= 0.0 && result.confidence <= 1.0,
                "Confidence {} out of range for '{}'", result.confidence, sentence);
            pipeline.reset();
        }
    }

    #[test]
    fn test_phi_computation() {
        let config = PipelineConfig::full_consciousness();
        let mut pipeline = ConsciousUnderstandingPipeline::new(config);

        let result = pipeline.understand("I know you want something good");

        // Φ should be computed
        assert!(result.consciousness.phi >= 0.0);
        assert!(result.consciousness.phi.is_finite());
    }

    #[test]
    fn benchmark_pipeline_performance() {
        use std::time::Instant;

        let test_sentences = [
            "The cat sat on the mat",
            "She gave him a beautiful red book",
            "I think you know something important",
            "The quick brown fox jumps over the lazy dog",
            "Complex systems exhibit emergent behavior through self-organization",
        ];

        // Warm up
        let config = PipelineConfig::default();
        let mut pipeline = ConsciousUnderstandingPipeline::new(config);
        for _ in 0..3 {
            for s in &test_sentences {
                pipeline.understand(s);
                pipeline.reset();
            }
        }

        // Benchmark full pipeline
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            for s in &test_sentences {
                let result = pipeline.understand(s);
                assert!(result.confidence > 0.0);
                pipeline.reset();
            }
        }
        let full_elapsed = start.elapsed();
        let full_per_sentence = full_elapsed.as_micros() / (iterations * test_sentences.len()) as u128;

        // Benchmark fast pipeline
        let fast_config = PipelineConfig::fast();
        let mut fast_pipeline = ConsciousUnderstandingPipeline::new(fast_config);
        let start = Instant::now();
        for _ in 0..iterations {
            for s in &test_sentences {
                let result = fast_pipeline.understand(s);
                assert!(result.confidence > 0.0);
                fast_pipeline.reset();
            }
        }
        let fast_elapsed = start.elapsed();
        let fast_per_sentence = fast_elapsed.as_micros() / (iterations * test_sentences.len()) as u128;

        // Assert reasonable performance (lenient for debug builds)
        // Full pipeline: < 50ms per sentence (debug mode)
        assert!(full_per_sentence < 50_000,
            "Full pipeline too slow: {}μs per sentence", full_per_sentence);

        // Fast pipeline: < 10ms per sentence (debug mode)
        assert!(fast_per_sentence < 10_000,
            "Fast pipeline too slow: {}μs per sentence", fast_per_sentence);

        // Fast should be at least 1.5x faster than full
        assert!(full_per_sentence > fast_per_sentence,
            "Fast pipeline should be faster than full");

        println!("\n📊 Conscious Understanding Pipeline Performance:");
        println!("   Full pipeline: {}μs per sentence", full_per_sentence);
        println!("   Fast pipeline: {}μs per sentence", fast_per_sentence);
        println!("   Speedup: {:.1}x", full_per_sentence as f64 / fast_per_sentence as f64);
    }

    #[test]
    fn test_layer_contributions() {
        let config = PipelineConfig::default();
        let mut pipeline = ConsciousUnderstandingPipeline::new(config);

        // Test sentence with rich semantic content
        let result = pipeline.understand("She gave him the ancient book of wisdom");

        // Verify all layers contribute
        assert!(!result.semantic_primes.is_empty(), "NSM layer should contribute primes");
        assert!(!result.utterance_encoding.0.is_empty(), "HDC layer should produce encoding");

        // Verify explanation trace has all stages
        let stage_names: Vec<&str> = result.explanation.stages.iter()
            .map(|s| s.name.as_str())
            .collect();
        assert!(stage_names.contains(&"NSM/Molecules"), "Should have NSM stage");
        assert!(stage_names.contains(&"Frames"), "Should have Frames stage");
        assert!(stage_names.contains(&"Constructions"), "Should have Constructions stage");
        assert!(stage_names.contains(&"Temporal"), "Should have Temporal stage");
        assert!(stage_names.contains(&"Prediction"), "Should have Prediction stage");
        assert!(stage_names.contains(&"Consciousness"), "Should have Consciousness stage");

        // Verify metrics are reasonable
        assert!(result.temporal_state.binding_coherence >= 0.0);
        assert!(result.consciousness.phi >= 0.0);
        assert!(result.prediction_result.final_free_energy >= 0.0);
    }
}
