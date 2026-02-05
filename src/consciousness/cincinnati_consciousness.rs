//! # Cincinnati-Consciousness Integration
//!
//! Integrates the Cincinnati Algorithm and LTC dynamics with the consciousness
//! pipeline for ethics-aware, grounded consciousness with:
//!
//! - Differential learning that adapts based on ethical feedback
//! - Lateral binding for coherent conscious experience
//! - Predictive autopoiesis for elastic topology based on consciousness needs
//! - PoG grounding to anchor consciousness in physical reality
//!
//! ## Architecture Integration
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                 CONSCIOUSNESS-CINCINNATI BRIDGE                      │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                      │
//! │  ┌─────────────────┐         ┌─────────────────────┐                │
//! │  │  Seven Harmonies │ ───────▶│   Cincinnati        │                │
//! │  │  (Ethics Layer)  │ feedback│   Estimator         │                │
//! │  └─────────────────┘         │   (Differential      │                │
//! │                               │    Learning)         │                │
//! │  ┌─────────────────┐         └─────────┬───────────┘                │
//! │  │ Consciousness   │                   │                            │
//! │  │ Graph           │◀──────────────────┤                            │
//! │  │ (HDC States)    │                   ▼                            │
//! │  └────────┬────────┘         ┌─────────────────────┐                │
//! │           │                  │   Cincinnati-LTC    │                │
//! │           │                  │   Network           │                │
//! │           ▼                  │   (W(t+1) = W(t) ⊕  │                │
//! │  ┌─────────────────┐        │   (ΔC ⊛ τ(t)))      │                │
//! │  │ PoG Grounding   │◀───────└─────────────────────┘                │
//! │  │ (Physical Real) │                                                │
//! │  └─────────────────┘                                                │
//! │                                                                      │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::hdc::cincinnati_ltc::{CincinnatiLtcEngine, PoGMetrics, BuddingEvent};
use symthaea_core::hdc::HDC_DIMENSION;

use serde::{Serialize, Deserialize};

/// A conscious node representation for Cincinnati-Consciousness integration
///
/// This type is specifically designed for the Cincinnati bridge and contains
/// semantic and dynamic components that can be mapped to HDC hypervectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CincinnatiConsciousNode {
    /// Semantic component (first half of HDC representation)
    pub semantic: Vec<f32>,

    /// Dynamic component (second half of HDC representation)
    pub dynamic: Vec<f32>,

    /// Consciousness level (0.0-1.0)
    pub consciousness: f32,

    /// Timestamp of this node's creation/update
    pub timestamp: f64,

    /// Importance weight for this node
    pub importance: f32,
}

impl CincinnatiConsciousNode {
    /// Create a new conscious node with default values
    pub fn new(semantic_dim: usize, dynamic_dim: usize) -> Self {
        Self {
            semantic: vec![0.0; semantic_dim],
            dynamic: vec![0.0; dynamic_dim],
            consciousness: 0.5,
            timestamp: 0.0,
            importance: 1.0,
        }
    }

    /// Create from a hypervector, splitting into semantic and dynamic halves
    pub fn from_hypervector(hv: &ContinuousHV, consciousness: f32, timestamp: f64, importance: f32) -> Self {
        let half = hv.dim() / 2;
        Self {
            semantic: hv.values[..half].to_vec(),
            dynamic: hv.values[half..].to_vec(),
            consciousness,
            timestamp,
            importance,
        }
    }

    /// Convert to a hypervector by concatenating semantic and dynamic components
    pub fn to_hypervector(&self) -> ContinuousHV {
        let mut values = self.semantic.clone();
        values.extend_from_slice(&self.dynamic);
        ContinuousHV::from_values(values)
    }
}

/// Configuration for Cincinnati-Consciousness integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CincinnatiConsciousnessConfig {
    /// Number of initial nodes in Cincinnati engine
    pub initial_nodes: usize,

    /// Learning rate for ethical feedback
    pub ethical_learning_rate: f32,

    /// Threshold for consciousness-triggered budding
    pub consciousness_budding_threshold: f32,

    /// Enable ethics-driven learning
    pub enable_ethical_learning: bool,

    /// Enable consciousness-based topology adaptation
    pub enable_consciousness_budding: bool,

    /// Enable physical grounding
    pub enable_pog: bool,
}

impl Default for CincinnatiConsciousnessConfig {
    fn default() -> Self {
        Self {
            initial_nodes: 7,  // Seven Harmonies alignment
            ethical_learning_rate: 0.05,
            consciousness_budding_threshold: 0.6,
            enable_ethical_learning: true,
            enable_consciousness_budding: true,
            enable_pog: true,
        }
    }
}

/// Cincinnati-Consciousness Bridge
///
/// Integrates Cincinnati Algorithm with consciousness graph for:
/// - Ethics-aware differential learning
/// - Consciousness-driven topology adaptation
/// - Physical reality grounding
pub struct CincinnatiConsciousnessBridge {
    /// Cincinnati-LTC engine for learning
    engine: CincinnatiLtcEngine,

    /// Configuration
    config: CincinnatiConsciousnessConfig,

    /// Current consciousness level
    consciousness_level: f32,

    /// Ethical alignment history (rolling)
    ethical_history: Vec<f32>,

    /// Budding events triggered by consciousness
    consciousness_budding_log: Vec<ConsciousnessBuddingEvent>,

    /// Timestep counter
    timestep: u64,
}

/// A budding event triggered by consciousness dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessBuddingEvent {
    /// Underlying budding event
    pub budding_event: BuddingEvent,

    /// Consciousness level at trigger
    pub consciousness_at_trigger: f32,

    /// Ethical alignment at trigger
    pub ethical_alignment: f32,

    /// Trigger reason
    pub reason: BuddingReason,
}

/// Reason for consciousness-triggered budding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BuddingReason {
    /// High prediction error (standard)
    HighPredictionError,

    /// Low consciousness level needs expansion
    LowConsciousness,

    /// Ethical tension requires new perspective
    EthicalTension,

    /// Complexity growth demands more capacity
    ComplexityGrowth,
}

impl CincinnatiConsciousnessBridge {
    /// Create a new consciousness bridge
    pub fn new(config: CincinnatiConsciousnessConfig) -> Self {
        Self {
            engine: CincinnatiLtcEngine::new(config.initial_nodes),
            config,
            consciousness_level: 0.5,
            ethical_history: Vec::with_capacity(100),
            consciousness_budding_log: Vec::new(),
            timestep: 0,
        }
    }

    /// Process consciousness state with ethical feedback
    ///
    /// This is the main integration point:
    /// 1. Update Cincinnati engine with ethical observation
    /// 2. Apply consciousness-based modulation
    /// 3. Check for consciousness-triggered budding
    /// 4. Update grounding metrics
    pub fn process(
        &mut self,
        input: &ContinuousHV,
        ethical_alignment: f32,
        consciousness_level: f32,
    ) -> ConsciousnessProcessResult {
        self.timestep += 1;
        self.consciousness_level = consciousness_level;

        // Store ethical alignment
        self.ethical_history.push(ethical_alignment);
        if self.ethical_history.len() > 100 {
            self.ethical_history.remove(0);
        }

        // 1. Convert ethical alignment to binary observation for Cincinnati
        let ethical_observation = ethical_alignment > 0.5;

        // 2. Step the Cincinnati-LTC engine
        let output = self.engine.step(ethical_observation, input);

        // 3. Modulate based on consciousness level
        let modulated_output = self.apply_consciousness_modulation(&output, consciousness_level);

        // 4. Check for consciousness-triggered budding
        if self.config.enable_consciousness_budding {
            self.check_consciousness_budding(consciousness_level, ethical_alignment);
        }

        // 5. Update PoG with consciousness metrics
        if self.config.enable_pog {
            self.engine.pog.update_accuracy(ethical_alignment > 0.5);
        }

        // 6. Compute comprehensive result
        let (prediction, confidence) = self.engine.predict();

        ConsciousnessProcessResult {
            output: modulated_output,
            prediction,
            confidence,
            consciousness_level,
            ethical_alignment,
            grounding_score: self.engine.pog.grounding_score(),
            node_count: self.engine.node_count(),
        }
    }

    /// Apply consciousness-level modulation to output
    fn apply_consciousness_modulation(
        &self,
        output: &ContinuousHV,
        consciousness_level: f32,
    ) -> ContinuousHV {
        // Higher consciousness = sharper distinctions (scale up high values, suppress low)
        let values: Vec<f32> = output.values.iter().map(|&x| {
            // Sigmoid sharpening based on consciousness level
            let sharpness = 1.0 + consciousness_level * 2.0;
            x.signum() * (x.abs() * sharpness).tanh()
        }).collect();

        ContinuousHV::from_values(values)
    }

    /// Check for consciousness-triggered budding
    fn check_consciousness_budding(&mut self, consciousness_level: f32, ethical_alignment: f32) {
        let _threshold = self.config.consciousness_budding_threshold;

        // Determine if budding should occur
        let reason = if consciousness_level < 0.3 {
            Some(BuddingReason::LowConsciousness)
        } else if ethical_alignment < 0.3 && self.ethical_history.len() > 10 {
            let avg_ethical: f32 = self.ethical_history.iter().sum::<f32>()
                / self.ethical_history.len() as f32;
            if avg_ethical < 0.4 {
                Some(BuddingReason::EthicalTension)
            } else {
                None
            }
        } else if self.engine.node_count() < self.timestep as usize / 100 {
            Some(BuddingReason::ComplexityGrowth)
        } else {
            None
        };

        // Create budding event if triggered
        if let Some(reason) = reason {
            let state = self.engine.weight().clone();
            let timestamp = self.timestep as f64;

            // Use engine's budding system
            if let Some(budding_event) = self.engine.budding.create_budding_event(
                0,  // Primary node
                timestamp,
                &state,
            ) {
                let event = ConsciousnessBuddingEvent {
                    budding_event,
                    consciousness_at_trigger: consciousness_level,
                    ethical_alignment,
                    reason,
                };
                self.consciousness_budding_log.push(event);
            }
        }
    }

    /// Convert conscious node to Cincinnati input
    pub fn conscious_node_to_input(&self, node: &CincinnatiConsciousNode) -> ContinuousHV {
        // Combine semantic and dynamic components
        let mut values = vec![0.0f32; HDC_DIMENSION];

        // Copy semantic (first half)
        let semantic_len = node.semantic.len().min(HDC_DIMENSION / 2);
        for (i, &v) in node.semantic.iter().take(semantic_len).enumerate() {
            values[i] = v;
        }

        // Copy dynamic (second half)
        let dynamic_start = HDC_DIMENSION / 2;
        let dynamic_len = node.dynamic.len().min(HDC_DIMENSION / 2);
        for (i, &v) in node.dynamic.iter().take(dynamic_len).enumerate() {
            values[dynamic_start + i] = v;
        }

        // Modulate by importance and consciousness
        let scale = node.importance * node.consciousness.sqrt();
        ContinuousHV::from_values(values).scale(scale)
    }

    /// Create conscious node from Cincinnati output
    pub fn output_to_conscious_node(
        &self,
        output: &ContinuousHV,
        timestamp: f64,
    ) -> CincinnatiConsciousNode {
        let half = output.dim() / 2;

        CincinnatiConsciousNode {
            semantic: output.values[..half].to_vec(),
            dynamic: output.values[half..].to_vec(),
            consciousness: self.consciousness_level,
            timestamp,
            importance: self.engine.pog.grounding_score(),
        }
    }

    /// Get ethical learning statistics
    pub fn ethical_stats(&self) -> EthicalLearningStats {
        let avg_ethical = if self.ethical_history.is_empty() {
            0.5
        } else {
            self.ethical_history.iter().sum::<f32>() / self.ethical_history.len() as f32
        };

        let ethical_variance = if self.ethical_history.len() > 1 {
            let mean = avg_ethical;
            self.ethical_history.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / self.ethical_history.len() as f32
        } else {
            0.0
        };

        let (prediction, confidence) = self.engine.predict();

        EthicalLearningStats {
            total_observations: self.timestep,
            average_ethical_alignment: avg_ethical,
            ethical_variance,
            current_prediction: prediction,
            prediction_confidence: confidence,
            consciousness_budding_events: self.consciousness_budding_log.len(),
            grounding_score: self.engine.pog.grounding_score(),
        }
    }

    /// Get consciousness budding log
    pub fn consciousness_budding_log(&self) -> &[ConsciousnessBuddingEvent] {
        &self.consciousness_budding_log
    }

    /// Get PoG metrics
    pub fn pog_metrics(&self) -> &PoGMetrics {
        &self.engine.pog
    }

    /// Get current node count
    pub fn node_count(&self) -> usize {
        self.engine.node_count()
    }

    /// Reset the bridge
    pub fn reset(&mut self) {
        self.engine = CincinnatiLtcEngine::new(self.config.initial_nodes);
        self.consciousness_level = 0.5;
        self.ethical_history.clear();
        self.consciousness_budding_log.clear();
        self.timestep = 0;
    }
}

/// Result of consciousness processing
#[derive(Debug, Clone)]
pub struct ConsciousnessProcessResult {
    /// Processed output hypervector
    pub output: ContinuousHV,

    /// Cincinnati prediction
    pub prediction: bool,

    /// Prediction confidence
    pub confidence: f32,

    /// Current consciousness level
    pub consciousness_level: f32,

    /// Current ethical alignment
    pub ethical_alignment: f32,

    /// Physical grounding score
    pub grounding_score: f32,

    /// Current node count
    pub node_count: usize,
}

/// Statistics for ethical learning
#[derive(Debug, Clone)]
pub struct EthicalLearningStats {
    /// Total observations processed
    pub total_observations: u64,

    /// Average ethical alignment over history
    pub average_ethical_alignment: f32,

    /// Variance in ethical alignment
    pub ethical_variance: f32,

    /// Current Cincinnati prediction
    pub current_prediction: bool,

    /// Prediction confidence
    pub prediction_confidence: f32,

    /// Number of consciousness-triggered budding events
    pub consciousness_budding_events: usize,

    /// Current grounding score
    pub grounding_score: f32,
}

// =============================================================================
// SEVEN HARMONIES INTEGRATION
// =============================================================================

/// Seven Harmonies feedback for Cincinnati learning
#[derive(Debug, Clone)]
pub struct HarmonyFeedback {
    /// Resonant Coherence score
    pub resonant_coherence: f32,

    /// Pan-Sentient Flourishing score
    pub pan_sentient_flourishing: f32,

    /// Integral Wisdom score
    pub integral_wisdom: f32,

    /// Infinite Play score
    pub infinite_play: f32,

    /// Universal Interconnectedness score
    pub universal_interconnectedness: f32,

    /// Sacred Reciprocity score
    pub sacred_reciprocity: f32,

    /// Evolutionary Progression score
    pub evolutionary_progression: f32,
}

impl HarmonyFeedback {
    /// Compute overall ethical alignment from harmonies
    pub fn overall_alignment(&self) -> f32 {
        let sum = self.resonant_coherence
            + self.pan_sentient_flourishing
            + self.integral_wisdom
            + self.infinite_play
            + self.universal_interconnectedness
            + self.sacred_reciprocity
            + self.evolutionary_progression;

        sum / 7.0
    }

    /// Check if any harmony is in tension (negative)
    pub fn has_tension(&self) -> bool {
        self.resonant_coherence < 0.0
            || self.pan_sentient_flourishing < 0.0
            || self.integral_wisdom < 0.0
            || self.infinite_play < 0.0
            || self.universal_interconnectedness < 0.0
            || self.sacred_reciprocity < 0.0
            || self.evolutionary_progression < 0.0
    }

    /// Get the most violated harmony
    pub fn most_violated(&self) -> (&'static str, f32) {
        let harmonies = [
            ("Resonant Coherence", self.resonant_coherence),
            ("Pan-Sentient Flourishing", self.pan_sentient_flourishing),
            ("Integral Wisdom", self.integral_wisdom),
            ("Infinite Play", self.infinite_play),
            ("Universal Interconnectedness", self.universal_interconnectedness),
            ("Sacred Reciprocity", self.sacred_reciprocity),
            ("Evolutionary Progression", self.evolutionary_progression),
        ];

        harmonies.iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(name, score)| (name, score))
            .unwrap_or(("Unknown", 0.0))
    }
}

impl CincinnatiConsciousnessBridge {
    /// Process with Seven Harmonies feedback
    pub fn process_with_harmonies(
        &mut self,
        input: &ContinuousHV,
        harmony_feedback: &HarmonyFeedback,
        consciousness_level: f32,
    ) -> ConsciousnessProcessResult {
        let ethical_alignment = harmony_feedback.overall_alignment();
        self.process(input, ethical_alignment, consciousness_level)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let config = CincinnatiConsciousnessConfig::default();
        let bridge = CincinnatiConsciousnessBridge::new(config);

        assert_eq!(bridge.node_count(), 7);  // Seven Harmonies alignment
        assert_eq!(bridge.timestep, 0);
    }

    #[test]
    fn test_process_basic() {
        let config = CincinnatiConsciousnessConfig::default();
        let mut bridge = CincinnatiConsciousnessBridge::new(config);

        let input = ContinuousHV::random_default(42);

        for i in 0..20 {
            let ethical = 0.3 + 0.4 * (i as f32 / 20.0);  // Improving
            let consciousness = 0.5 + 0.2 * (i as f32 / 20.0).sin();

            let result = bridge.process(&input, ethical, consciousness);

            assert!(result.confidence >= 0.0);
            assert!(result.grounding_score >= 0.0 && result.grounding_score <= 1.0);
        }

        let stats = bridge.ethical_stats();
        assert_eq!(stats.total_observations, 20);
        assert!(stats.average_ethical_alignment > 0.0);
    }

    #[test]
    fn test_harmony_feedback() {
        let feedback = HarmonyFeedback {
            resonant_coherence: 0.8,
            pan_sentient_flourishing: 0.7,
            integral_wisdom: 0.6,
            infinite_play: 0.9,
            universal_interconnectedness: 0.5,
            sacred_reciprocity: 0.8,
            evolutionary_progression: 0.7,
        };

        let alignment = feedback.overall_alignment();
        assert!(alignment > 0.5);
        assert!(!feedback.has_tension());
    }

    #[test]
    fn test_harmony_tension() {
        let feedback = HarmonyFeedback {
            resonant_coherence: 0.8,
            pan_sentient_flourishing: -0.3,  // Tension!
            integral_wisdom: 0.6,
            infinite_play: 0.9,
            universal_interconnectedness: 0.5,
            sacred_reciprocity: 0.8,
            evolutionary_progression: 0.7,
        };

        assert!(feedback.has_tension());

        let (name, score) = feedback.most_violated();
        assert_eq!(name, "Pan-Sentient Flourishing");
        assert!(score < 0.0);
    }

    #[test]
    fn test_consciousness_modulation() {
        let config = CincinnatiConsciousnessConfig::default();
        let bridge = CincinnatiConsciousnessBridge::new(config);

        let input = ContinuousHV::random(1024, 42);

        // Low consciousness = softer output
        let low_mod = bridge.apply_consciousness_modulation(&input, 0.2);

        // High consciousness = sharper output
        let high_mod = bridge.apply_consciousness_modulation(&input, 0.9);

        // High consciousness should have more extreme values
        let low_var: f32 = low_mod.values.iter().map(|x| x.abs()).sum::<f32>() / 1024.0;
        let high_var: f32 = high_mod.values.iter().map(|x| x.abs()).sum::<f32>() / 1024.0;

        // High consciousness should have larger absolute values on average
        assert!(high_var >= low_var * 0.9);  // Allow some tolerance
    }

    #[test]
    fn test_node_conversion() {
        let config = CincinnatiConsciousnessConfig::default();
        let bridge = CincinnatiConsciousnessBridge::new(config);

        let node = CincinnatiConsciousNode {
            semantic: vec![0.5; 100],
            dynamic: vec![0.3; 100],
            consciousness: 0.7,
            timestamp: 1.0,
            importance: 0.8,
        };

        let input = bridge.conscious_node_to_input(&node);
        assert_eq!(input.dim(), HDC_DIMENSION);

        let output = ContinuousHV::random_default(42);
        let result_node = bridge.output_to_conscious_node(&output, 2.0);

        assert_eq!(result_node.timestamp, 2.0);
        assert_eq!(result_node.semantic.len(), HDC_DIMENSION / 2);
    }

    #[test]
    fn test_cincinnati_conscious_node_creation() {
        // Test new() constructor
        let node = CincinnatiConsciousNode::new(512, 512);
        assert_eq!(node.semantic.len(), 512);
        assert_eq!(node.dynamic.len(), 512);
        assert_eq!(node.consciousness, 0.5);
        assert_eq!(node.importance, 1.0);
    }

    #[test]
    fn test_cincinnati_conscious_node_from_hypervector() {
        let hv = ContinuousHV::random_default(42);
        let node = CincinnatiConsciousNode::from_hypervector(&hv, 0.8, 100.0, 0.9);

        assert_eq!(node.semantic.len(), HDC_DIMENSION / 2);
        assert_eq!(node.dynamic.len(), HDC_DIMENSION / 2);
        assert_eq!(node.consciousness, 0.8);
        assert_eq!(node.timestamp, 100.0);
        assert_eq!(node.importance, 0.9);

        // Convert back to hypervector
        let hv_back = node.to_hypervector();
        assert_eq!(hv_back.dim(), HDC_DIMENSION);
    }
}
