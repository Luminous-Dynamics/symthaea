// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Type definitions for the consciousness integration pipeline.
//!
//! Contains all public structs, enums, and state view types used throughout
//! the consciousness integration system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::super::adaptive_topology::CognitiveMode;
use super::super::binary_hv::BinaryHV;
use super::super::cross_modal_binding::Modality;

// Re-export SubstrateType from substrate_independence
pub use super::super::substrate_independence::SubstrateType;

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

// ==========================================
// PIPELINE CHECKPOINT
// ==========================================

/// Serializable snapshot of pipeline state for save/restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCheckpoint {
    /// Consciousness state at checkpoint time
    pub state: ConsciousnessState,
    /// Pipeline configuration
    pub config: IntegrationConfig,
    /// Processing history
    pub history: Vec<ConsciousnessState>,
    /// Maximum history size setting
    pub max_history_size: usize,
    /// Current processing cycle
    pub current_cycle: u64,
    /// Embodiment level
    pub embodiment_level: f64,
}

// ==========================================
// UNIFIED CONSCIOUSNESS OPTIMIZER TYPES
// ==========================================

/// Unified consciousness metrics report
#[derive(Debug, Clone)]
pub struct ConsciousnessMetricsReport {
    pub phi: f64,
    pub consciousness_level: f64,
    pub embodiment_level: f64,
    pub metacognitive_confidence: f64,
    pub cross_modal_coherence: f64,
    pub temporal_coherence: f64,
    pub topological_unity: f64,
    pub narrative_coherence: f64,
    pub self_awareness_level: f64,
    pub prediction_accuracy: f64,
    pub self_model_confidence: f64,
    pub integrated_systems_active: bool,
    pub phi_optimization_active: bool,
    pub feedback_dynamics_active: bool,
    pub self_awareness_active: bool,
    pub processing_cycles: u64,
    pub bound_objects_count: usize,
    pub workspace_items_count: usize,
}

impl std::fmt::Display for ConsciousnessMetricsReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Consciousness Metrics: Phi={:.4} Level={:.4} Embodiment={:.4} Cycles={}",
            self.phi, self.consciousness_level, self.embodiment_level, self.processing_cycles
        )
    }
}

/// Priority level for optimization recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Single optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub system: String,
    pub priority: RecommendationPriority,
    pub message: String,
    pub suggested_action: Option<String>,
}

impl std::fmt::Display for OptimizationRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.system, self.message)?;
        if let Some(action) = &self.suggested_action {
            write!(f, " -> {action}")?;
        }
        Ok(())
    }
}

/// Summary of an optimization cycle
#[derive(Debug, Clone)]
pub struct OptimizationCycleSummary {
    pub phi_optimized: bool,
    pub feedback_processed: bool,
    pub self_model_updated: bool,
    pub phi_before: f64,
    pub phi_after: f64,
    pub recommendations: Vec<OptimizationRecommendation>,
}

impl std::fmt::Display for OptimizationCycleSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Optimization: Phi {:.4} -> {:.4}",
            self.phi_before, self.phi_after
        )
    }
}

/// Index into altered states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AlteredStateIndex {
    #[default]
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

/// Workspace item for global workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceItem {
    pub content: BinaryHV,
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
    pub representation: BinaryHV,
}

/// Binding level in hierarchical binding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BindingLevel {
    #[default]
    Feature,
    Object,
    Scene,
}

/// Bound object from binding problem (hierarchical + temporal)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundObject {
    pub representation: BinaryHV,
    pub synchrony: f64,
    pub binding_strength: f64,
    pub conscious: bool,
    pub level: BindingLevel,
    pub child_ids: Vec<usize>,
    pub attention_weight: f64,
    pub creation_cycle: u64,
    pub persistence_cycles: u64,
    pub temporal_stability: f64,
}

/// Temporal binding memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalBindingMemory {
    pub representation: BinaryHV,
    pub strength: f64,
    pub cycle: u64,
    pub level: BindingLevel,
}

impl BoundObject {
    pub fn is_conscious(&self) -> bool {
        self.conscious
    }

    pub fn new_feature(
        representation: BinaryHV,
        synchrony: f64,
        binding_strength: f64,
        conscious: bool,
    ) -> Self {
        Self {
            representation,
            synchrony,
            binding_strength,
            conscious,
            level: BindingLevel::Feature,
            child_ids: Vec::new(),
            attention_weight: 1.0,
            creation_cycle: 0,
            persistence_cycles: 0,
            temporal_stability: 1.0,
        }
    }

    pub fn from_features(features: &[&BoundObject], seed: u64) -> Self {
        if features.is_empty() {
            return Self::new_feature(BinaryHV::random(seed), 0.0, 0.0, false);
        }
        let bound_repr = features
            .iter()
            .map(|f| f.representation)
            .reduce(|a, b| a.bind(&b))
            .unwrap_or_else(|| BinaryHV::random(seed));
        let avg_synchrony =
            features.iter().map(|f| f.synchrony).sum::<f64>() / features.len() as f64;
        let avg_strength =
            features.iter().map(|f| f.binding_strength).sum::<f64>() / features.len() as f64;
        let avg_attention =
            features.iter().map(|f| f.attention_weight).sum::<f64>() / features.len() as f64;
        let avg_stability =
            features.iter().map(|f| f.temporal_stability).sum::<f64>() / features.len() as f64;
        Self {
            representation: bound_repr,
            synchrony: avg_synchrony,
            binding_strength: (avg_strength * 1.1).min(1.0),
            conscious: features.iter().any(|f| f.conscious),
            level: BindingLevel::Object,
            child_ids: Vec::new(),
            attention_weight: avg_attention,
            creation_cycle: 0,
            persistence_cycles: 0,
            temporal_stability: avg_stability * 1.05,
        }
    }

    pub fn from_objects(objects: &[&BoundObject], seed: u64) -> Self {
        if objects.is_empty() {
            return Self::new_feature(BinaryHV::random(seed), 0.0, 0.0, false);
        }
        let bound_repr = objects
            .iter()
            .map(|o| o.representation)
            .reduce(|a, b| a.bind(&b))
            .unwrap_or_else(|| BinaryHV::random(seed));
        let avg_synchrony = objects.iter().map(|o| o.synchrony).sum::<f64>() / objects.len() as f64;
        let avg_strength =
            objects.iter().map(|o| o.binding_strength).sum::<f64>() / objects.len() as f64;
        let avg_attention =
            objects.iter().map(|o| o.attention_weight).sum::<f64>() / objects.len() as f64;
        let avg_stability =
            objects.iter().map(|o| o.temporal_stability).sum::<f64>() / objects.len() as f64;
        Self {
            representation: bound_repr,
            synchrony: (avg_synchrony * 1.05).min(1.0),
            binding_strength: (avg_strength * 1.2).min(1.0),
            conscious: objects.iter().all(|o| o.conscious),
            level: BindingLevel::Scene,
            child_ids: Vec::new(),
            attention_weight: avg_attention,
            creation_cycle: 0,
            persistence_cycles: 0,
            temporal_stability: (avg_stability * 1.1).min(1.0),
        }
    }
}

/// Consciousness state - the complete state of a conscious system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    pub phi: f64,
    pub free_energy: f64,
    pub temporal_coherence: f64,
    pub consciousness_level: f64,
    pub conscious_contents: Vec<WorkspaceItem>,
    pub bound_objects: Vec<BoundObject>,
    pub meta_awareness: Vec<MetaThought>,
    pub altered_state: AlteredStateIndex,
    pub attention_focus: Option<BinaryHV>,
    pub prediction_accuracy: f64,
    pub flow_stability: f64,
    pub embodiment: f64,
    pub semantic_depth: f64,
    pub topological_unity: f64,
    pub metacognitive_confidence: f64,
    pub metacognitive_coherence: f64,
    pub predicted_phi: Option<f64>,
    pub phi_trend: f64,
    pub predictive_precision: f64,
    pub surprise_level: f64,
    pub inference_mode: PredictiveMode,
    pub cross_modal_coherence: f64,
    pub active_modalities: Vec<Modality>,
    pub theta_phase: f64,
    pub narrative_coherence: f64,
    pub present_window_length: usize,
    pub self_model_confidence: f64,
    pub self_model_accuracy: f64,
    pub cognitive_mode: CognitiveMode,
    pub mode_appropriateness: f64,
    pub emotional_valence: f64,
    pub emotional_arousal: Option<f64>,
    pub uncertainty: f64,
    pub integration_score: f64,
}

/// Predictive processing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PredictiveMode {
    Exploring,
    Exploiting,
    #[default]
    Balanced,
}

/// State view types for structured access to ConsciousnessState fields
pub struct PhiMetrics {
    pub phi: f64,
    pub free_energy: f64,
    pub topological_unity: f64,
    pub phi_trend: f64,
    pub predicted_phi: Option<f64>,
}
pub struct TemporalStateView {
    pub temporal_coherence: f64,
    pub theta_phase: f64,
    pub narrative_coherence: f64,
    pub present_window_length: usize,
}
pub struct SelfModelStateView {
    pub confidence: f64,
    pub accuracy: f64,
}
pub struct EmotionalStateView {
    pub valence: f64,
    pub arousal: Option<f64>,
    pub uncertainty: f64,
}
pub struct PredictiveStateView {
    pub precision: f64,
    pub surprise: f64,
}
pub struct IntegrationMetricsView {
    pub metacognitive_confidence: f64,
    pub cross_modal_coherence: f64,
    pub integration_score: f64,
}

impl ConsciousnessState {
    pub fn phi_metrics(&self) -> PhiMetrics {
        PhiMetrics {
            phi: self.phi,
            free_energy: self.free_energy,
            topological_unity: self.topological_unity,
            phi_trend: self.phi_trend,
            predicted_phi: self.predicted_phi,
        }
    }
    pub fn temporal_state(&self) -> TemporalStateView {
        TemporalStateView {
            temporal_coherence: self.temporal_coherence,
            theta_phase: self.theta_phase,
            narrative_coherence: self.narrative_coherence,
            present_window_length: self.present_window_length,
        }
    }
    pub fn self_model_state(&self) -> SelfModelStateView {
        SelfModelStateView {
            confidence: self.self_model_confidence,
            accuracy: self.self_model_accuracy,
        }
    }
    pub fn emotional_state(&self) -> EmotionalStateView {
        EmotionalStateView {
            valence: self.emotional_valence,
            arousal: self.emotional_arousal,
            uncertainty: self.uncertainty,
        }
    }
    pub fn predictive_state(&self) -> PredictiveStateView {
        PredictiveStateView {
            precision: self.predictive_precision,
            surprise: self.surprise_level,
        }
    }
    pub fn integration_metrics(&self) -> IntegrationMetricsView {
        IntegrationMetricsView {
            metacognitive_confidence: self.metacognitive_confidence,
            cross_modal_coherence: self.cross_modal_coherence,
            integration_score: self.integration_score,
        }
    }
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
            metacognitive_confidence: 0.5,
            metacognitive_coherence: 0.5,
            predicted_phi: None,
            phi_trend: 0.0,
            predictive_precision: 1.0,
            surprise_level: 0.5,
            inference_mode: PredictiveMode::Balanced,
            cross_modal_coherence: 0.0,
            active_modalities: Vec::new(),
            theta_phase: 0.0,
            narrative_coherence: 0.5,
            present_window_length: 0,
            self_model_confidence: 0.5,
            self_model_accuracy: 0.5,
            cognitive_mode: CognitiveMode::default(),
            mode_appropriateness: 0.5,
            emotional_valence: 0.0,
            emotional_arousal: Some(0.5),
            uncertainty: 0.5,
            integration_score: 0.5,
        }
    }
}

/// Assessment result from integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationAssessment {
    pub is_conscious: bool,
    pub consciousness_score: f64,
    pub component_scores: HashMap<String, f64>,
    pub bottlenecks: Vec<String>,
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

// ==========================================
// SUBSYSTEM CYCLE REPORT
// ==========================================

/// Report from a single subsystem's processing cycle
#[derive(Debug, Clone)]
pub struct SubsystemCycleReport {
    pub name: String,
    pub ran: bool,
    pub duration_us: u64,
    pub phi_delta: f64,
    pub error: Option<super::super::consciousness_subsystem::SubsystemError>,
}
