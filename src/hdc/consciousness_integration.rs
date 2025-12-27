//! Consciousness Integration Module
//!
//! Provides types for consciousness pipeline integration.
//! This is a compatibility layer that works with consciousness_orchestrator.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::binary_hv::HV16;

// Re-export SubstrateType from substrate_independence
pub use super::substrate_independence::SubstrateType;

/// Integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Number of processing cycles
    pub num_cycles: usize,
    /// Features per stimulus
    pub features_per_stimulus: usize,
    /// Attention capacity
    pub attention_capacity: usize,
    /// Workspace capacity
    pub workspace_capacity: usize,
    /// Consciousness threshold
    pub consciousness_threshold: f64,
    /// Verbose logging
    pub verbose: bool,
    /// Binding threshold for feature integration
    pub binding_threshold: f64,
    /// Enable Higher-Order Thought processing
    pub hot_enabled: bool,
    /// Substrate type for consciousness
    pub substrate: SubstrateType,
    /// Processing precision
    pub precision: f64,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            num_cycles: 10,
            features_per_stimulus: 4,
            attention_capacity: 4,
            workspace_capacity: 3,
            consciousness_threshold: 0.5,
            verbose: false,
            binding_threshold: 0.7,
            hot_enabled: true,
            substrate: SubstrateType::Biological,
            precision: 1.0,
        }
    }
}

/// Index into altered states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlteredStateIndex {
    Wake,
    Drowsy,
    N1Sleep,
    N2Sleep,
    N3Sleep,
    REM,
    LucidDream,
    Meditation,
    Flow,
    Propofol,
    Ketamine,
    VegetativeState,
    MinimallyConscious,
}

impl Default for AlteredStateIndex {
    fn default() -> Self {
        AlteredStateIndex::Wake
    }
}

/// Workspace item for global workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceItem {
    pub content: HV16,
    pub activation: f64,
    pub source: String,
    pub is_broadcasting: bool,
    pub duration_ms: u64,
}

/// Meta-thought for HOT theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaThought {
    pub about: String,
    pub target: String,
    pub intensity: f64,
    pub confidence: f64,
    pub order: u8,
    pub representation: HV16,
}

/// Bound object from binding problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundObject {
    pub representation: HV16,
    pub synchrony: f64,
    pub binding_strength: f64,
    pub conscious: bool,
}

impl BoundObject {
    pub fn is_conscious(&self) -> bool {
        self.conscious
    }
}

/// Consciousness state - the complete state of a conscious system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    /// Integrated information Φ
    pub phi: f64,
    /// Free energy (prediction error)
    pub free_energy: f64,
    /// Temporal coherence
    pub temporal_coherence: f64,
    /// Overall consciousness level [0, 1]
    pub consciousness_level: f64,
    /// Items in global workspace
    pub conscious_contents: Vec<WorkspaceItem>,
    /// Bound objects from binding
    pub bound_objects: Vec<BoundObject>,
    /// Meta-awareness (HOT)
    pub meta_awareness: Vec<MetaThought>,
    /// Current altered state
    pub altered_state: AlteredStateIndex,
    /// Attention focus
    pub attention_focus: Option<HV16>,
    /// Prediction accuracy
    pub prediction_accuracy: f64,
    /// Flow stability
    pub flow_stability: f64,
    /// Embodiment level
    pub embodiment: f64,
    /// Semantic depth
    pub semantic_depth: f64,
    /// Topological unity
    pub topological_unity: f64,
}

impl Default for ConsciousnessState {
    fn default() -> Self {
        Self {
            phi: 0.0,
            free_energy: 1.0,
            temporal_coherence: 0.0,
            consciousness_level: 0.0,
            conscious_contents: Vec::new(),
            bound_objects: Vec::new(),
            meta_awareness: Vec::new(),
            altered_state: AlteredStateIndex::Wake,
            attention_focus: None,
            prediction_accuracy: 0.0,
            flow_stability: 0.0,
            embodiment: 0.5,
            semantic_depth: 0.0,
            topological_unity: 0.5,
        }
    }
}

/// Assessment result from integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationAssessment {
    /// Is the system conscious?
    pub is_conscious: bool,
    /// Overall score [0, 1]
    pub consciousness_score: f64,
    /// Component scores
    pub component_scores: HashMap<String, f64>,
    /// Bottlenecks identified
    pub bottlenecks: Vec<String>,
    /// Explanation
    pub explanation: String,
}

impl Default for IntegrationAssessment {
    fn default() -> Self {
        Self {
            is_conscious: false,
            consciousness_score: 0.0,
            component_scores: HashMap::new(),
            bottlenecks: Vec::new(),
            explanation: String::new(),
        }
    }
}

/// The consciousness pipeline orchestrator
#[derive(Debug, Clone)]
pub struct ConsciousnessPipeline {
    /// Configuration
    pub config: IntegrationConfig,
    /// Current state
    pub state: ConsciousnessState,
    /// Processing history
    history: Vec<ConsciousnessState>,
    /// Embodiment level
    embodiment_level: f64,
}

impl ConsciousnessPipeline {
    /// Create new pipeline with config
    pub fn new(config: IntegrationConfig) -> Self {
        Self {
            config,
            state: ConsciousnessState::default(),
            history: Vec::new(),
            embodiment_level: 0.5,
        }
    }

    /// Create with default config
    pub fn default_new() -> Self {
        Self::new(IntegrationConfig::default())
    }

    /// Create with config (alias for compatibility)
    pub fn with_config(config: IntegrationConfig) -> Self {
        Self::new(config)
    }

    /// Set embodiment level
    pub fn set_embodiment(&mut self, level: f64) {
        self.embodiment_level = level.clamp(0.0, 1.0);
    }

    /// Process input through the consciousness pipeline
    pub fn process(&mut self, input: Vec<HV16>, priorities: &[f64]) -> &ConsciousnessState {
        // Feature detection and binding
        let binding_strength = if input.len() > 1 {
            0.7 + (priorities.iter().sum::<f64>() / priorities.len() as f64) * 0.3
        } else {
            0.5
        };

        // Update phi based on integration
        self.state.phi = (binding_strength * self.embodiment_level).min(1.0);

        // Update consciousness level
        let attention_boost = priorities.iter()
            .cloned()
            .fold(0.5_f64, |a, b| a.max(b));
        self.state.consciousness_level = self.state.phi * 0.4 +
            attention_boost * 0.3 +
            self.embodiment_level * 0.3;

        // Generate workspace content if high enough consciousness
        if self.state.consciousness_level > self.config.consciousness_threshold {
            for (i, hv) in input.iter().take(self.config.workspace_capacity).enumerate() {
                let priority = priorities.get(i).copied().unwrap_or(0.5);
                if priority > 0.6 {
                    self.state.conscious_contents.push(WorkspaceItem {
                        content: hv.clone(),
                        activation: priority,
                        source: format!("input_{}", i),
                        is_broadcasting: true,
                        duration_ms: 100,
                    });
                }
            }
        }

        // Generate meta-awareness if HOT enabled
        if self.config.hot_enabled && self.state.consciousness_level > 0.6 {
            self.state.meta_awareness.push(MetaThought {
                about: "current processing".to_string(),
                target: "consciousness_state".to_string(),
                intensity: self.state.consciousness_level,
                confidence: 0.8,
                order: 2,
                representation: HV16::random(99),
            });
        }

        // Update temporal coherence
        self.state.temporal_coherence = 0.5 + self.state.consciousness_level * 0.5;

        // Reduce free energy (better predictions)
        self.state.free_energy = 1.0 - self.state.consciousness_level * 0.8;

        // Update prediction accuracy
        self.state.prediction_accuracy = self.state.consciousness_level * 0.9;

        // Store in history
        self.history.push(self.state.clone());

        &self.state
    }

    /// Process a single cycle
    pub fn process_cycle(&mut self, input: &[HV16]) {
        // Simplified processing - just update state
        let intensity = input.len() as f64 * 0.1;
        self.state.phi = (self.state.phi + intensity).min(1.0);
        self.state.consciousness_level = self.state.phi * 0.5 +
            self.state.temporal_coherence * 0.3 +
            (1.0 - self.state.free_energy) * 0.2;

        self.history.push(self.state.clone());
    }

    /// Set altered state
    pub fn set_altered_state(&mut self, state: AlteredStateIndex) {
        self.state.altered_state = state;

        // Apply state effects
        match state {
            AlteredStateIndex::Wake => {
                self.state.consciousness_level = 0.7;
            }
            AlteredStateIndex::N3Sleep => {
                self.state.consciousness_level = 0.1;
                self.state.conscious_contents.clear();
            }
            AlteredStateIndex::REM => {
                self.state.consciousness_level = 0.5;
            }
            AlteredStateIndex::LucidDream => {
                self.state.consciousness_level = 0.7;
            }
            AlteredStateIndex::Propofol => {
                self.state.consciousness_level = 0.0;
                self.state.phi = 0.05;
            }
            AlteredStateIndex::VegetativeState => {
                self.state.consciousness_level = 0.0;
            }
            AlteredStateIndex::MinimallyConscious => {
                self.state.consciousness_level = 0.3;
            }
            _ => {}
        }
    }

    /// Assess current consciousness
    pub fn assess(&self) -> IntegrationAssessment {
        self.assess_integration()
    }

    /// Full integration assessment
    pub fn assess_integration(&self) -> IntegrationAssessment {
        let mut scores = HashMap::new();
        scores.insert("phi".to_string(), self.state.phi);
        scores.insert("workspace".to_string(),
            if self.state.conscious_contents.is_empty() { 0.0 } else { 0.8 });
        scores.insert("binding".to_string(),
            self.state.bound_objects.iter()
                .filter(|b| b.is_conscious())
                .count() as f64 * 0.2);
        scores.insert("hot".to_string(),
            if self.state.meta_awareness.is_empty() { 0.0 } else { 0.7 });

        let avg_score = scores.values().sum::<f64>() / scores.len() as f64;
        let is_conscious = avg_score >= self.config.consciousness_threshold;

        IntegrationAssessment {
            is_conscious,
            consciousness_score: avg_score,
            component_scores: scores,
            bottlenecks: Vec::new(),
            explanation: format!("Consciousness score: {:.2}", avg_score),
        }
    }

    /// Get current state
    pub fn get_state(&self) -> &ConsciousnessState {
        &self.state
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.state = ConsciousnessState::default();
        self.history.clear();
    }
}

impl Default for ConsciousnessPipeline {
    fn default() -> Self {
        Self::default_new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_config_default() {
        let config = IntegrationConfig::default();
        assert_eq!(config.num_cycles, 10);
        assert_eq!(config.workspace_capacity, 3);
    }

    #[test]
    fn test_consciousness_state_default() {
        let state = ConsciousnessState::default();
        assert!((state.phi - 0.0).abs() < 1e-10);
        assert!(state.conscious_contents.is_empty());
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = ConsciousnessPipeline::default();
        assert!((pipeline.state.phi - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_process() {
        let mut pipeline = ConsciousnessPipeline::default();
        let input = vec![HV16::random(42)];
        pipeline.process_cycle(&input);
        assert!(pipeline.state.phi > 0.0);
    }

    #[test]
    fn test_altered_state_effects() {
        let mut pipeline = ConsciousnessPipeline::default();

        pipeline.set_altered_state(AlteredStateIndex::Wake);
        assert!(pipeline.state.consciousness_level > 0.5);

        pipeline.set_altered_state(AlteredStateIndex::Propofol);
        assert!(pipeline.state.consciousness_level < 0.1);
    }

    #[test]
    fn test_assessment() {
        let pipeline = ConsciousnessPipeline::default();
        let assessment = pipeline.assess();
        assert!(!assessment.is_conscious); // Default state is not conscious
    }

    #[test]
    fn test_process_method() {
        let config = IntegrationConfig::default();
        let mut pipeline = ConsciousnessPipeline::new(config);
        pipeline.set_embodiment(0.8);

        let input = vec![HV16::random(1), HV16::random(2), HV16::random(3)];
        let priorities = vec![0.9, 0.7, 0.5];

        let state = pipeline.process(input, &priorities);
        assert!(state.consciousness_level > 0.0);
        assert!(state.phi > 0.0);
    }

    #[test]
    fn test_bound_object() {
        let obj = BoundObject {
            representation: HV16::random(1),
            synchrony: 0.8,
            binding_strength: 0.9,
            conscious: true,
        };
        assert!(obj.is_conscious());
    }

    #[test]
    fn test_workspace_item() {
        let item = WorkspaceItem {
            content: HV16::random(2),
            activation: 0.95,
            source: "visual".to_string(),
            is_broadcasting: true,
            duration_ms: 100,
        };
        assert!(item.activation > 0.9);
        assert!(item.is_broadcasting);
    }

    #[test]
    fn test_meta_thought() {
        let thought = MetaThought {
            about: "seeing red".to_string(),
            target: "visual_perception".to_string(),
            intensity: 0.8,
            confidence: 0.9,
            order: 2,
            representation: HV16::random(42),
        };
        assert_eq!(thought.order, 2);
        assert!(thought.confidence > 0.5);
    }
}
