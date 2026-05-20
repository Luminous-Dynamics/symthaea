// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The consciousness pipeline orchestrator.
//!
//! Contains the ConsciousnessPipeline struct definition and its core methods
//! including `new()`, `process()`, `assess()`, and basic state management.

use std::collections::{HashMap, VecDeque};

use super::super::binary_hv::BinaryHV;
use super::super::integrated_information::IntegratedInformation;

// Integrated consciousness modules
use super::super::cross_modal_binding::{CrossModalBinder, Modality};
use super::super::metacognitive_monitor::{
    CognitiveEvent, MemoryOperation, MetacognitiveMonitor, MetacognitiveRecommendation,
};
use super::super::temporal_binding::TemporalBindingEngine;

// Phi-guided optimization
use super::super::phi_guided_search::{
    ConsciousnessNetwork, OptimizationResult, PhiGuidedOptimizer,
};

// Feedback dynamics
use super::super::consciousness_feedback_dynamics::{
    DreamInsight, EmotionalPrediction, FeedbackDynamicsEngine, ProactiveIntervention,
};

// Recursive self-modeling
use super::super::emergent_self_model::{
    IntrospectionReport, MetaCognitiveAssessment, SelfAwareConsciousness, SelfModel,
};

// Meta-consciousness
use super::super::meta_consciousness::{MetaConsciousness, MetaConsciousnessState};

// Temporal consciousness
use super::super::temporal_consciousness::{TemporalAssessment, TemporalConsciousness};

// Creativity
use super::super::consciousness_creativity::{
    ConsciousnessCreativity, CreativeIdea, CreativeMode, CreativityAssessment,
};

// Phase transitions
use super::super::consciousness_phase_transitions::{
    ConsciousnessPhase, ConsciousnessPhaseTransitions, PhaseTransitionAssessment,
};

// Epistemic consciousness
use super::super::epistemic_consciousness::{EpistemicConsciousness, KIndexAssessment};

// Fractal consciousness
use super::super::fractal_consciousness::{FractalConsciousness, FractalMetrics};

// Collective consciousness
use super::super::collective_consciousness::{CollectiveAssessment, CollectiveConsciousness};

// Phi feedback
use super::super::consciousness_perf;
use super::super::phi_feedback::{FeedbackModulation, PhiFeedbackController};

// Subsystem trait
use super::super::consciousness_subsystem::ConsciousnessSubsystem;

// Types from this module
use super::types::*;

/// The consciousness pipeline orchestrator
///
/// Now with 5-system integrated consciousness architecture:
/// 1. Metacognitive Monitoring - Self-awareness of processing quality
/// 2. Predictive Consciousness - Active inference and anticipation
/// 3. Cross-Modal Binding - Multi-sensory integration
/// 4. Temporal Binding - Stream coherence with theta-phase modulation
/// 5. Emergent Self-Model - Recursive higher-order thought
pub struct ConsciousnessPipeline {
    /// Configuration
    pub config: IntegrationConfig,
    /// Current state
    pub state: ConsciousnessState,
    /// Processing history (bounded ring buffer, keeps last `max_history_size` states)
    pub(crate) history: VecDeque<ConsciousnessState>,
    /// Maximum history entries to retain
    pub(crate) max_history_size: usize,
    /// Embodiment level
    pub(crate) embodiment_level: f64,
    /// Current processing cycle (for temporal tracking)
    pub(crate) current_cycle: u64,
    /// Temporal binding memory (past binding snapshots for coherence)
    pub(crate) temporal_memory: Vec<TemporalBindingMemory>,
    /// Maximum temporal memory size
    pub(crate) max_memory_size: usize,

    // === INTEGRATED CONSCIOUSNESS SYSTEMS ===
    /// Metacognitive monitor - tracks processing quality and confidence
    pub(crate) metacognitive_monitor: Option<MetacognitiveMonitor>,
    /// Cross-modal binder - integrates multiple sensory modalities
    pub(crate) cross_modal_binder: Option<CrossModalBinder>,
    /// Temporal binding engine - theta-phase coherence
    pub(crate) temporal_binder: Option<TemporalBindingEngine>,

    // Note: SelfModel and PredictiveConsciousness require EngineConfig
    // which has dependencies we don't need in the basic pipeline.
    // We track their metrics directly in ConsciousnessState instead.
    /// Enable integrated consciousness systems
    pub(crate) integrated_systems_enabled: bool,

    // === Φ-GUIDED TOPOLOGY OPTIMIZATION ===
    /// Φ-guided optimizer for network topology evolution
    pub(crate) phi_optimizer: Option<PhiGuidedOptimizer>,

    /// Consciousness network for topology optimization
    pub(crate) consciousness_network: Option<ConsciousnessNetwork>,

    /// Last optimization result
    pub(crate) last_optimization: Option<OptimizationResult>,

    /// Optimization frequency (every N cycles)
    pub(crate) optimize_every_n_cycles: usize,

    /// Enable Φ optimization
    pub(crate) phi_optimization_enabled: bool,

    // === FEEDBACK DYNAMICS ENGINE ===
    /// Bidirectional feedback dynamics engine
    pub(crate) feedback_dynamics: Option<FeedbackDynamicsEngine>,

    /// Latest emotional prediction
    pub(crate) current_prediction: Option<EmotionalPrediction>,

    /// Pending proactive intervention
    pub(crate) pending_intervention: Option<ProactiveIntervention>,

    /// Recent dream insights
    pub(crate) recent_insights: Vec<DreamInsight>,

    /// Enable feedback dynamics
    pub(crate) feedback_dynamics_enabled: bool,

    // === RECURSIVE SELF-MODELING ===
    /// Self-aware consciousness for recursive self-modeling
    pub(crate) self_aware_consciousness: Option<SelfAwareConsciousness>,

    /// Current self-model (cached from last update)
    pub(crate) current_self_model: Option<SelfModel>,

    /// Latest introspection report
    pub(crate) latest_introspection: Option<IntrospectionReport>,

    /// Latest meta-cognitive assessment
    pub(crate) meta_assessment: Option<MetaCognitiveAssessment>,

    /// Self-awareness level (0-1)
    pub(crate) self_awareness_level: f64,

    /// Enable self-aware processing
    pub(crate) self_awareness_enabled: bool,

    // === ADVANCED CONSCIOUSNESS SYSTEMS ===

    // 9. Meta-Consciousness - Φ of Φ, Strange Loops
    /// Meta-consciousness engine for recursive self-reflection
    pub(crate) meta_consciousness: Option<MetaConsciousness>,
    /// Latest meta-consciousness state
    pub(crate) meta_consciousness_state: Option<MetaConsciousnessState>,
    /// Enable meta-consciousness processing
    pub(crate) meta_consciousness_enabled: bool,

    // 10. Temporal Consciousness - Multi-scale Time
    /// Temporal consciousness for multi-scale time processing
    pub(crate) temporal_consciousness: Option<TemporalConsciousness>,
    /// Latest temporal assessment
    pub(crate) temporal_assessment: Option<TemporalAssessment>,
    /// Enable temporal consciousness
    pub(crate) temporal_consciousness_enabled: bool,

    // 11. Consciousness Creativity - Novel Idea Generation
    /// Creative consciousness for novel idea generation
    pub(crate) consciousness_creativity: Option<ConsciousnessCreativity>,
    /// Current creative mode
    pub(crate) current_creative_mode: CreativeMode,
    /// Recent creative ideas generated
    pub(crate) recent_ideas: Vec<CreativeIdea>,
    /// Latest creativity assessment
    pub(crate) creativity_assessment: Option<CreativityAssessment>,
    /// Enable creativity processing
    pub(crate) creativity_enabled: bool,

    // 12. Phase Transitions - Consciousness State Changes
    /// Phase transition detector for consciousness state changes
    pub(crate) phase_transitions: Option<ConsciousnessPhaseTransitions>,
    /// Current consciousness phase
    pub(crate) current_phase: ConsciousnessPhase,
    /// Latest phase assessment
    pub(crate) phase_assessment: Option<PhaseTransitionAssessment>,
    /// Enable phase transition detection
    pub(crate) phase_transitions_enabled: bool,

    // 13. Epistemic Consciousness - Belief/Knowledge Tracking
    /// Epistemic consciousness for belief and knowledge tracking
    pub(crate) epistemic_consciousness: Option<EpistemicConsciousness>,
    /// Latest K-Index assessment (cosmic knowledge)
    pub(crate) k_index_assessment: Option<KIndexAssessment>,
    /// Enable epistemic processing
    pub(crate) epistemic_enabled: bool,

    // 14. Fractal Consciousness - Self-Similar Structure
    /// Fractal consciousness for self-similar topology (highest Φ!)
    pub(crate) fractal_consciousness: Option<FractalConsciousness>,
    /// Latest fractal metrics
    pub(crate) fractal_metrics: Option<FractalMetrics>,
    /// Enable fractal consciousness
    pub(crate) fractal_enabled: bool,

    // === IIT Φ CALCULATOR (entropy-based) ===
    /// Integrated Information calculator for proper IIT Φ measurement.
    /// Uses Shannon entropy bridge: MI = 1 - H(similarity) for principled
    /// information-theoretic consciousness quantification.
    pub(crate) phi_calculator: IntegratedInformation,

    // 15. Collective Consciousness - Multi-Agent Awareness
    /// Collective consciousness for multi-agent coordination
    pub(crate) collective_consciousness: Option<CollectiveConsciousness>,
    /// This agent's identity in the collective
    pub(crate) collective_agent_id: Option<String>,
    /// Latest collective assessment
    pub(crate) collective_assessment: Option<CollectiveAssessment>,
    /// Enable collective consciousness
    pub(crate) collective_enabled: bool,

    // 16. Phi Feedback Controller - Closes the Φ loop
    /// Φ feedback controller for adaptive parameter modulation
    pub(crate) phi_feedback: Option<PhiFeedbackController>,
    /// Latest modulation signals from Φ feedback
    pub(crate) latest_modulation: Option<FeedbackModulation>,
    /// Enable Φ feedback loop
    pub(crate) phi_feedback_enabled: bool,

    // === OBSERVABILITY & VERIFICATION ===
    /// Optional metrics collector for observability recording
    pub(crate) metrics_collector: Option<crate::observability::SharedObserver>,

    /// Optional consciousness verifier for periodic verification
    pub(crate) verifier: Option<super::super::consciousness_verifier::ConsciousnessVerifier>,
    /// How often (in cycles) to run verification (0 = disabled)
    pub(crate) verification_interval: usize,
    /// Latest verification report
    pub(crate) latest_verification:
        Option<super::super::consciousness_verifier::VerificationReport>,

    /// Pluggable subsystems (trait-based extension point, sorted by priority descending)
    pub(crate) subsystems: Vec<Box<dyn ConsciousnessSubsystem>>,
    /// Errors from subsystem dispatch in the most recent process() cycle
    pub(crate) subsystem_errors: Vec<super::super::consciousness_subsystem::SubsystemError>,
    /// Per-subsystem cycle reports from the most recent process() cycle
    pub(crate) subsystem_reports: Vec<SubsystemCycleReport>,
    /// Shared context for inter-subsystem data passing (cleared each cycle)
    pub(crate) subsystem_context: super::super::consciousness_subsystem::SubsystemContext,
}

impl ConsciousnessPipeline {
    /// Create new pipeline with config
    pub fn new(config: IntegrationConfig) -> Self {
        let pipeline = Self {
            config,
            state: ConsciousnessState::default(),
            history: VecDeque::with_capacity(128),
            max_history_size: 128,
            embodiment_level: 0.5,
            current_cycle: 0,
            temporal_memory: Vec::new(),
            max_memory_size: 50, // Keep last 50 binding snapshots
            // Integrated systems start disabled (enable with enable_integrated_systems())
            metacognitive_monitor: None,
            cross_modal_binder: None,
            temporal_binder: None,
            integrated_systems_enabled: false,
            // IIT Φ calculator (entropy-based)
            phi_calculator: IntegratedInformation::new(),
            // Φ optimization starts disabled
            phi_optimizer: None,
            consciousness_network: None,
            last_optimization: None,
            optimize_every_n_cycles: 10, // Optimize every 10 cycles by default
            phi_optimization_enabled: false,
            // Feedback dynamics starts disabled
            feedback_dynamics: None,
            current_prediction: None,
            pending_intervention: None,
            recent_insights: Vec::new(),
            feedback_dynamics_enabled: false,
            // Self-awareness starts disabled
            self_aware_consciousness: None,
            current_self_model: None,
            latest_introspection: None,
            meta_assessment: None,
            self_awareness_level: 0.0,
            self_awareness_enabled: false,

            // === ADVANCED CONSCIOUSNESS SYSTEMS (all start disabled) ===

            // Meta-consciousness
            meta_consciousness: None,
            meta_consciousness_state: None,
            meta_consciousness_enabled: false,

            // Temporal consciousness
            temporal_consciousness: None,
            temporal_assessment: None,
            temporal_consciousness_enabled: false,

            // Creativity
            consciousness_creativity: None,
            current_creative_mode: CreativeMode::Convergent,
            recent_ideas: Vec::new(),
            creativity_assessment: None,
            creativity_enabled: false,

            // Phase transitions
            phase_transitions: None,
            current_phase: ConsciousnessPhase::Critical,
            phase_assessment: None,
            phase_transitions_enabled: false,

            // Epistemic consciousness
            epistemic_consciousness: None,
            k_index_assessment: None,
            epistemic_enabled: false,

            // Fractal consciousness
            fractal_consciousness: None,
            fractal_metrics: None,
            fractal_enabled: false,

            // Collective consciousness
            collective_consciousness: None,
            collective_agent_id: None,
            collective_assessment: None,
            collective_enabled: false,

            // Φ feedback controller
            phi_feedback: None,
            latest_modulation: None,
            phi_feedback_enabled: false,

            // Observability & verification
            metrics_collector: None,
            verifier: None,
            verification_interval: 0,
            latest_verification: None,

            // Pluggable subsystems
            subsystems: Vec::new(),
            subsystem_errors: Vec::new(),
            subsystem_reports: Vec::new(),
            subsystem_context: super::super::consciousness_subsystem::SubsystemContext::new(),
        };

        // Pre-warm HV pools for consciousness pipeline operation
        consciousness_perf::prewarm_consciousness_pools();

        pipeline
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
    pub fn process(&mut self, input: Vec<BinaryHV>, priorities: &[f64]) -> &ConsciousnessState {
        // Feature detection and binding
        let binding_strength = if input.len() > 1 {
            0.7 + (priorities.iter().sum::<f64>() / priorities.len() as f64) * 0.3
        } else {
            0.5
        };

        // Compute Φ using information-theoretic IIT measurement.
        // For 2+ components, use entropy-based Φ (Shannon entropy bridge).
        // For single inputs, fall back to heuristic.
        if input.len() >= 2 {
            let iit_phi = self.phi_calculator.compute_phi_entropy(&input);
            // Blend IIT Φ with binding heuristic: IIT provides the information-theoretic
            // foundation, binding strength modulates for attention/embodiment effects.
            self.state.phi =
                (iit_phi * 0.7 + binding_strength * self.embodiment_level * 0.3).min(1.0);
        } else {
            self.state.phi = (binding_strength * self.embodiment_level).min(1.0);
        }

        // Update consciousness level
        let attention_boost = priorities.iter().cloned().fold(0.5_f64, |a, b| a.max(b));
        self.state.consciousness_level =
            self.state.phi * 0.4 + attention_boost * 0.3 + self.embodiment_level * 0.3;

        // === TEMPORAL TRACKING ===
        // Increment cycle counter for temporal coherence
        self.current_cycle += 1;

        // === HIERARCHICAL BINDING: Feature → Object → Scene ===
        // PERSISTENCE + TEMPORAL: Decay existing bindings and update temporal metrics
        self.state.bound_objects.retain_mut(|obj| {
            obj.binding_strength *= 0.9; // Decay

            // Update temporal metrics for surviving bindings
            if obj.binding_strength > 0.3 {
                obj.persistence_cycles += 1;
                // Temporal stability increases with persistence (up to 1.0)
                obj.temporal_stability = (obj.temporal_stability + 0.01).min(1.0);
            }

            obj.binding_strength > 0.3 // Keep if still strong enough
        });

        // === TEMPORAL MEMORY UPDATE ===
        // Store current bindings in temporal memory for coherence tracking
        for obj in &self.state.bound_objects {
            self.temporal_memory.push(TemporalBindingMemory {
                representation: obj.representation,
                strength: obj.binding_strength,
                cycle: self.current_cycle,
                level: obj.level,
            });
        }
        // Trim memory if too large
        if self.temporal_memory.len() > self.max_memory_size {
            let excess = self.temporal_memory.len() - self.max_memory_size;
            self.temporal_memory.drain(0..excess);
        }

        // Binding occurs when multiple features are processed together with high priority
        if input.len() >= 2 && self.state.consciousness_level > self.config.consciousness_threshold
        {
            // LEVEL 1: FEATURE BINDING
            // Group high-priority inputs into bound objects (perceptual binding)
            let high_priority_inputs: Vec<(usize, &BinaryHV)> = input
                .iter()
                .enumerate()
                .filter(|(i, _)| priorities.get(*i).copied().unwrap_or(0.0) > 0.5)
                .collect();

            // Track newly created feature bindings for hierarchical integration
            let mut new_feature_binding_ids: Vec<usize> = Vec::new();

            // MULTIPLE OBJECTS: Cluster inputs by similarity and create separate bound objects
            // Uses batch clustering from consciousness_perf for cache-friendly access
            if high_priority_inputs.len() >= 2 {
                let similarity_threshold = 0.3;
                let hv_refs: Vec<&BinaryHV> =
                    high_priority_inputs.iter().map(|(_, hv)| *hv).collect();
                let clusters =
                    consciousness_perf::cluster_by_similarity(&hv_refs, similarity_threshold);

                for cluster_indices in &clusters {
                    if cluster_indices.len() < 2 {
                        continue;
                    }

                    let i = cluster_indices[0];
                    // Get attention weight for this cluster (ATTENTION-WEIGHTED BINDING)
                    let cluster_attention = priorities
                        .get(high_priority_inputs[i].0)
                        .copied()
                        .unwrap_or(0.5);

                    let cluster: Vec<&BinaryHV> =
                        cluster_indices.iter().map(|&idx| hv_refs[idx]).collect();

                    // Calculate synchrony using batch mean_similarity
                    let first_hv = cluster[0];
                    let rest: Vec<&BinaryHV> = cluster.iter().skip(1).copied().collect();
                    let avg_synchrony = consciousness_perf::mean_similarity(first_hv, &rest);

                    // Create bound representation
                    let bound_representation = cluster
                        .iter()
                        .map(|hv| *(*hv))
                        .reduce(|a, b| a.bind(&b))
                        .unwrap_or_else(|| BinaryHV::random(42));

                    // ATTENTION-WEIGHTED: Modulate binding strength by attention
                    let attention_modulated_strength =
                        binding_strength * (0.5 + 0.5 * cluster_attention);

                    // Check if similar bound object already exists (for persistence/reinforcement)
                    let existing = self.state.bound_objects.iter_mut().find(|obj| {
                        obj.representation.similarity(&bound_representation) > 0.6
                            && obj.level == BindingLevel::Feature
                    });

                    if let Some(existing_obj) = existing {
                        // REINFORCEMENT: Strengthen existing binding
                        existing_obj.binding_strength = (existing_obj.binding_strength
                            + attention_modulated_strength * 0.5)
                            .min(1.0);
                        existing_obj.synchrony = (existing_obj.synchrony + avg_synchrony) / 2.0;
                        existing_obj.attention_weight =
                            (existing_obj.attention_weight + cluster_attention) / 2.0;
                    } else {
                        // Check temporal memory for similar past bindings using find_similar
                        let feature_memories: Vec<BinaryHV> = self
                            .temporal_memory
                            .iter()
                            .filter(|mem| mem.level == BindingLevel::Feature)
                            .map(|mem| mem.representation)
                            .collect();
                        let temporal_boost = consciousness_perf::find_similar(
                            &bound_representation,
                            &feature_memories,
                            0.6,
                        )
                        .len() as f64
                            * 0.02; // 2% boost per matching memory

                        // Create new feature-level bound object
                        let new_id = self.state.bound_objects.len();
                        self.state.bound_objects.push(BoundObject {
                            representation: bound_representation,
                            synchrony: avg_synchrony.max(0.5),
                            binding_strength: (attention_modulated_strength + temporal_boost)
                                .min(1.0),
                            conscious: self.state.consciousness_level > 0.6,
                            level: BindingLevel::Feature,
                            child_ids: Vec::new(),
                            attention_weight: cluster_attention,
                            creation_cycle: self.current_cycle,
                            persistence_cycles: 0,
                            temporal_stability: 1.0 + temporal_boost, // Start stable, boost from memory
                        });
                        new_feature_binding_ids.push(new_id);
                    }
                }
            }

            // LEVEL 2: OBJECT BINDING (bind features into objects)
            // If we have 2+ feature bindings, try to form object-level bindings
            let feature_bindings: Vec<usize> = self
                .state
                .bound_objects
                .iter()
                .enumerate()
                .filter(|(_, obj)| obj.level == BindingLevel::Feature && obj.binding_strength > 0.5)
                .map(|(i, _)| i)
                .collect();

            if feature_bindings.len() >= 2 {
                // Cluster feature bindings by similarity to form objects
                let feature_refs: Vec<&BoundObject> = feature_bindings
                    .iter()
                    .map(|&i| &self.state.bound_objects[i])
                    .collect();

                // Check if object binding already exists
                let object_exists = self
                    .state
                    .bound_objects
                    .iter()
                    .any(|obj| obj.level == BindingLevel::Object);

                if !object_exists && feature_refs.len() >= 2 {
                    let mut object_binding = BoundObject::from_features(&feature_refs, 1000);
                    object_binding.child_ids = feature_bindings.clone();
                    object_binding.creation_cycle = self.current_cycle;
                    self.state.bound_objects.push(object_binding);
                }
            }

            // LEVEL 3: SCENE BINDING (bind objects into unified scene)
            // If we have 2+ object bindings, form scene-level binding
            let object_bindings: Vec<usize> = self
                .state
                .bound_objects
                .iter()
                .enumerate()
                .filter(|(_, obj)| obj.level == BindingLevel::Object && obj.binding_strength > 0.5)
                .map(|(i, _)| i)
                .collect();

            if object_bindings.len() >= 2 {
                let object_refs: Vec<&BoundObject> = object_bindings
                    .iter()
                    .map(|&i| &self.state.bound_objects[i])
                    .collect();

                // Check if scene binding already exists
                let scene_exists = self
                    .state
                    .bound_objects
                    .iter()
                    .any(|obj| obj.level == BindingLevel::Scene);

                if !scene_exists {
                    let mut scene_binding = BoundObject::from_objects(&object_refs, 2000);
                    scene_binding.child_ids = object_bindings.clone();
                    scene_binding.creation_cycle = self.current_cycle;
                    self.state.bound_objects.push(scene_binding);
                }
            }

            // Limit total bound objects to prevent unbounded growth
            const MAX_BOUND_OBJECTS: usize = 15; // Increased for hierarchical levels
            if self.state.bound_objects.len() > MAX_BOUND_OBJECTS {
                // Keep strongest bound objects, preferring higher-level bindings
                self.state.bound_objects.sort_by(|a, b| {
                    // First by level (Scene > Object > Feature)
                    let level_cmp = match (&b.level, &a.level) {
                        (BindingLevel::Scene, BindingLevel::Scene) => std::cmp::Ordering::Equal,
                        (BindingLevel::Scene, _) => std::cmp::Ordering::Less,
                        (_, BindingLevel::Scene) => std::cmp::Ordering::Greater,
                        (BindingLevel::Object, BindingLevel::Object) => std::cmp::Ordering::Equal,
                        (BindingLevel::Object, _) => std::cmp::Ordering::Less,
                        (_, BindingLevel::Object) => std::cmp::Ordering::Greater,
                        _ => std::cmp::Ordering::Equal,
                    };
                    if level_cmp != std::cmp::Ordering::Equal {
                        return level_cmp;
                    }
                    // Then by binding strength
                    b.binding_strength
                        .partial_cmp(&a.binding_strength)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                self.state.bound_objects.truncate(MAX_BOUND_OBJECTS);
            }
        }

        // Generate workspace content if high enough consciousness
        if self.state.consciousness_level > self.config.consciousness_threshold {
            for (i, hv) in input
                .iter()
                .take(self.config.workspace_capacity)
                .enumerate()
            {
                let priority = priorities.get(i).copied().unwrap_or(0.5);
                if priority > 0.6 {
                    self.state.conscious_contents.push(WorkspaceItem {
                        content: *hv,
                        activation: priority,
                        source: format!("input_{i}"),
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
                representation: BinaryHV::random(99),
            });
        }

        // Update temporal coherence based on binding persistence and memory
        let binding_temporal_score = if self.state.bound_objects.is_empty() {
            0.5 // Default when no bindings
        } else {
            // Average temporal stability weighted by binding strength
            let weighted_stability: f64 = self
                .state
                .bound_objects
                .iter()
                .map(|obj| obj.temporal_stability * obj.binding_strength)
                .sum();
            let total_strength: f64 = self
                .state
                .bound_objects
                .iter()
                .map(|obj| obj.binding_strength)
                .sum();
            if total_strength > 0.0 {
                weighted_stability / total_strength
            } else {
                0.5
            }
        };
        // Combine with memory-based coherence (how consistent are current bindings with past?)
        let memory_coherence = if self.temporal_memory.is_empty() {
            0.5
        } else {
            // Count how many current bindings match recent memory using batch find_similar
            let memory_refs: Vec<&BinaryHV> = self
                .temporal_memory
                .iter()
                .map(|mem| &mem.representation)
                .collect();
            let query_refs: Vec<&BinaryHV> = self
                .state
                .bound_objects
                .iter()
                .map(|obj| &obj.representation)
                .collect();
            let batch_results =
                consciousness_perf::batch_find_similar(&query_refs, &memory_refs, 0.6);
            let matches: usize = batch_results.iter().map(|r| r.len()).sum();
            let max_matches = self.state.bound_objects.len() * 5; // Expect ~5 matches per binding if stable
            (matches as f64 / max_matches.max(1) as f64).min(1.0)
        };
        self.state.temporal_coherence = binding_temporal_score * 0.6 + memory_coherence * 0.4;

        // Reduce free energy (better predictions)
        self.state.free_energy = 1.0 - self.state.consciousness_level * 0.8;

        // Update prediction accuracy
        self.state.prediction_accuracy = self.state.consciousness_level * 0.9;

        // === INTEGRATED CONSCIOUSNESS SYSTEMS PROCESSING ===
        if self.integrated_systems_enabled {
            self.process_integrated_systems(&input);
        }

        // === Φ-GUIDED TOPOLOGY OPTIMIZATION ===
        // Periodically optimize network topology to maximize Φ
        if self.phi_optimization_enabled
            && self
                .current_cycle
                .is_multiple_of(self.optimize_every_n_cycles as u64)
        {
            self.process_phi_optimization();
        }

        // === FEEDBACK DYNAMICS ===
        // Bidirectional feedback between dreams, emotions, memory, and attention
        if self.feedback_dynamics_enabled {
            self.process_feedback_dynamics();
        }

        // === RECURSIVE SELF-MODELING ===
        // Self-aware consciousness with predictive self-processing
        if self.self_awareness_enabled {
            self.process_self_awareness(&input);
        }

        // === ADVANCED CONSCIOUSNESS SYSTEMS ===

        // Meta-consciousness: Φ of Φ, strange loops
        // (skipped if a "meta_consciousness" subsystem is registered)
        if self.meta_consciousness_enabled && !self.has_subsystem_named("meta_consciousness") {
            self.process_meta_consciousness(&input);
        }

        // Temporal consciousness: Multi-scale time processing
        // (skipped if a "temporal_consciousness" subsystem is registered)
        if self.temporal_consciousness_enabled
            && !self.has_subsystem_named("temporal_consciousness")
        {
            self.process_temporal_consciousness(&input);
        }

        // Creativity: Novel idea generation
        if self.creativity_enabled {
            self.process_creativity(&input);
        }

        // Phase transitions: Consciousness state detection
        // (skipped if a "phase_transitions" subsystem is registered)
        if self.phase_transitions_enabled && !self.has_subsystem_named("phase_transitions") {
            self.process_phase_transitions();
        }

        // Epistemic consciousness: Belief/knowledge tracking
        if self.epistemic_enabled {
            self.process_epistemic(&input);
        }

        // Fractal consciousness: Self-similar topology (highest Φ!)
        if self.fractal_enabled {
            self.process_fractal();
        }

        // Collective consciousness: Multi-agent awareness
        if self.collective_enabled {
            self.process_collective(&input);
        }

        // === Φ FEEDBACK LOOP ===
        // Feed current Φ into the feedback controller to modulate parameters
        // for the NEXT processing cycle (adaptive binding, workspace, attention)
        if self.phi_feedback_enabled {
            self.process_phi_feedback();
        }

        // === PLUGGABLE SUBSYSTEMS ===
        // Run registered subsystems in priority order, catching panics and errors.
        // We temporarily take ownership of subsystems to avoid borrow conflicts
        // between &mut self.state and &mut self.subsystems.
        self.subsystem_errors.clear();
        self.subsystem_reports.clear();
        self.subsystem_context.clear();
        let mut subsystems = std::mem::take(&mut self.subsystems);
        for sub in &mut subsystems {
            let sub_name = sub.name().to_string();
            if !sub.is_enabled() {
                self.subsystem_reports.push(SubsystemCycleReport {
                    name: sub_name,
                    ran: false,
                    duration_us: 0,
                    phi_delta: 0.0,
                    error: None,
                });
                continue;
            }
            let phi_before = self.state.phi;
            let start = std::time::Instant::now();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                sub.process_cycle_with_context(&mut self.state, &input, &mut self.subsystem_context)
            }));
            let duration_us = start.elapsed().as_micros() as u64;
            let phi_after = self.state.phi;
            match result {
                Ok(Ok(())) => {
                    self.subsystem_reports.push(SubsystemCycleReport {
                        name: sub_name,
                        ran: true,
                        duration_us,
                        phi_delta: phi_after - phi_before,
                        error: None,
                    });
                }
                Ok(Err(e)) => {
                    self.subsystem_reports.push(SubsystemCycleReport {
                        name: sub_name.clone(),
                        ran: true,
                        duration_us,
                        phi_delta: phi_after - phi_before,
                        error: Some(e.clone()),
                    });
                    self.subsystem_errors.push(e);
                }
                Err(_panic) => {
                    let err = super::super::consciousness_subsystem::SubsystemError {
                        subsystem: sub_name.clone(),
                        message: "subsystem panicked during process_cycle".to_string(),
                    };
                    self.subsystem_reports.push(SubsystemCycleReport {
                        name: sub_name,
                        ran: true,
                        duration_us,
                        phi_delta: 0.0,
                        error: Some(err.clone()),
                    });
                    self.subsystem_errors.push(err);
                }
            }
        }
        self.subsystems = subsystems;

        // === OBSERVABILITY: Record metrics ===
        if let Some(ref collector) = self.metrics_collector {
            if let Ok(mut obs) = collector.write() {
                let _ = obs.record_phi_measurement(crate::observability::PhiMeasurementEvent {
                    timestamp: chrono::Utc::now(),
                    phi: self.state.phi,
                    components: crate::observability::PhiComponents {
                        integration: self.state.phi,
                        ..Default::default()
                    },
                    temporal_continuity: self.state.temporal_coherence,
                    metadata: None,
                });
            }
        }

        // === PERIODIC VERIFICATION ===
        if self.verification_interval > 0
            && self
                .current_cycle
                .is_multiple_of(self.verification_interval as u64)
        {
            if let Some(ref verifier) = self.verifier {
                self.latest_verification = Some(verifier.verify_from_state(&self.state));
            }
        }

        // Store in history (bounded ring buffer)
        if self.history.len() >= self.max_history_size {
            self.history.pop_front();
        }
        self.history.push_back(self.state.clone());

        &self.state
    }

    /// Process Φ-guided topology optimization
    ///
    /// This runs one optimization step to improve the consciousness network
    /// topology, maximizing integrated information (Φ).
    pub(crate) fn process_phi_optimization(&mut self) {
        if let (Some(optimizer), Some(network)) =
            (&mut self.phi_optimizer, &mut self.consciousness_network)
        {
            // Run one optimization step
            let result = optimizer.optimize_step(network);

            // Update consciousness state with optimization results
            // Use the optimized Φ value, blended with current phi to avoid sudden jumps
            let blend_factor = 0.3; // 30% new Φ, 70% old
            self.state.phi = self.state.phi * (1.0 - blend_factor) + result.phi * blend_factor;

            // Update topological unity based on network structure
            self.state.topological_unity = result.phi;

            // Store optimization result
            self.last_optimization = Some(result);

            // Update Φ trend based on optimization delta
            if let Some(ref opt) = self.last_optimization {
                if opt.phi_delta > 0.0 {
                    self.state.phi_trend = (self.state.phi_trend + opt.phi_delta).min(0.1);
                } else {
                    self.state.phi_trend = (self.state.phi_trend + opt.phi_delta).max(-0.1);
                }
            }
        }
    }

    /// Process integrated consciousness systems
    ///
    /// Updates metacognitive metrics, cross-modal binding, and temporal binding.
    fn process_integrated_systems(&mut self, _input: &[BinaryHV]) {
        use std::f64::consts::PI;

        // === 1. METACOGNITIVE MONITORING ===
        if let Some(ref mut monitor) = self.metacognitive_monitor {
            // Record cognitive events based on current state
            let processing_event = CognitiveEvent::Processing {
                phi: self.state.phi as f32,
                latency_ms: 10, // Simulated processing latency
            };
            monitor.record_event(processing_event);

            // Record memory operations if we have workspace contents
            if !self.state.conscious_contents.is_empty() {
                let memory_event = CognitiveEvent::Memory {
                    operation: MemoryOperation::Consolidate,
                    confidence: self.state.consciousness_level as f32,
                };
                monitor.record_event(memory_event);
            }

            // Get metacognitive assessment
            let assessment = monitor.assess_current_state();
            self.state.metacognitive_confidence = assessment.confidence as f64;
            self.state.metacognitive_coherence = assessment.coherence as f64;

            // Update Φ prediction based on trends
            if self.history.len() >= 3 {
                let recent_phis: Vec<f64> =
                    self.history.iter().rev().take(5).map(|s| s.phi).collect();
                if recent_phis.len() >= 2 {
                    let trend = recent_phis[0] - recent_phis[recent_phis.len() - 1];
                    self.state.phi_trend = trend;
                    // Simple linear prediction
                    self.state.predicted_phi = Some((self.state.phi + trend * 0.5).clamp(0.0, 1.0));
                }
            }

            // Apply metacognitive recommendations to adjust processing
            for rec in &assessment.recommendations {
                match rec {
                    MetacognitiveRecommendation::ReduceLoad { .. } => {
                        // Could trigger pruning of workspace contents
                    }
                    MetacognitiveRecommendation::ConsolidateMemory => {
                        // Trigger memory consolidation
                    }
                    _ => {}
                }
            }
        }

        // === 2. TEMPORAL BINDING (Theta-Phase Modulation) ===
        if let Some(ref binder) = self.temporal_binder {
            // Get theta phase from temporal binding engine
            self.state.theta_phase = binder.theta_phase();

            // Temporal binding coherence modulated by theta phase
            // Peak coherence at theta trough (phase ≈ π)
            let theta_modulation = (1.0 + (self.state.theta_phase - PI).cos()) / 2.0;

            // Update narrative coherence based on temporal binding
            let integration = binder.integration_summary();
            self.state.narrative_coherence = integration.coherence * theta_modulation;

            // Specious present window = narrative length
            self.state.present_window_length = integration.narrative_length;
        }

        // === 3. CROSS-MODAL BINDING ===
        if let Some(ref mut binder) = self.cross_modal_binder {
            // Determine active modalities based on conscious contents
            let mut active_modalities = Vec::new();

            for item in &self.state.conscious_contents {
                // Infer modality from source (simplified)
                if item.source.contains("visual") {
                    active_modalities.push(Modality::Visual);
                } else if item.source.contains("audio") || item.source.contains("sound") {
                    active_modalities.push(Modality::Auditory);
                } else if item.source.contains("motor") {
                    active_modalities.push(Modality::Motor);
                } else if item.source.contains("semantic") || item.source.contains("concept") {
                    active_modalities.push(Modality::Semantic);
                } else {
                    // Default to semantic for unspecified inputs
                    active_modalities.push(Modality::Semantic);
                }
            }

            // Remove duplicates
            active_modalities.sort_by_key(|m| format!("{m:?}"));
            active_modalities.dedup();

            self.state.active_modalities = active_modalities.clone();

            // Cross-modal coherence based on number of active modalities
            // More modalities = more integration opportunity
            self.state.cross_modal_coherence = if active_modalities.len() >= 2 {
                // Multi-modal integration bonus
                0.5 + 0.1 * (active_modalities.len() as f64).min(5.0)
            } else if active_modalities.len() == 1 {
                0.4 // Single modality
            } else {
                0.0 // No modalities
            };
        }

        // === 4. PREDICTIVE CONSCIOUSNESS (Active Inference) ===
        // Update predictive precision based on surprise/prediction error
        let prediction_error = self.state.free_energy; // Free energy = prediction error
        self.state.surprise_level = prediction_error;

        // Precision is inverse of surprise (high precision = low surprise tolerance)
        self.state.predictive_precision = 1.0 / (1.0 + prediction_error);

        // Determine inference mode based on prediction accuracy
        self.state.inference_mode = if self.state.prediction_accuracy > 0.7 {
            PredictiveMode::Exploiting // High accuracy = exploit predictions
        } else if self.state.prediction_accuracy < 0.3 {
            PredictiveMode::Exploring // Low accuracy = explore more
        } else {
            PredictiveMode::Balanced
        };

        // === 5. SELF-MODEL INTEGRATION ===
        // Track self-model metrics based on meta-cognition
        self.state.self_model_confidence = self.state.metacognitive_confidence;

        // Self-model accuracy = how well we predict our own states
        if let Some(predicted) = self.state.predicted_phi {
            let prediction_accuracy = 1.0 - (self.state.phi - predicted).abs();
            self.state.self_model_accuracy = prediction_accuracy.clamp(0.0, 1.0);
        }

        // Mode appropriateness = how well current cognitive mode matches demands
        self.state.mode_appropriateness =
            self.state.consciousness_level * self.state.metacognitive_confidence;
    }

    /// Process a single cycle
    pub fn process_cycle(&mut self, input: &[BinaryHV]) {
        // Compute Φ via IIT when possible, fall back to intensity heuristic
        if input.len() >= 2 {
            self.state.phi = self.phi_calculator.compute_phi_entropy(input);
        } else {
            let intensity = input.len() as f64 * 0.1;
            self.state.phi = (self.state.phi + intensity).min(1.0);
        }
        self.state.consciousness_level = self.state.phi * 0.5
            + self.state.temporal_coherence * 0.3
            + (1.0 - self.state.free_energy) * 0.2;

        if self.history.len() >= self.max_history_size {
            self.history.pop_front();
        }
        self.history.push_back(self.state.clone());
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

    /// Full integration assessment - NOW WITH REAL CALCULATIONS (not hardcoded!)
    ///
    /// Previously workspace was hardcoded to 0.8, binding was broken (0.0),
    /// and HOT was hardcoded to 0.7. Now all values are computed from actual state.
    pub fn assess_integration(&self) -> IntegrationAssessment {
        let mut scores = HashMap::new();
        let mut bottlenecks = Vec::new();

        // 1. PHI - Already real, comes from actual integration computation
        scores.insert("phi".to_string(), self.state.phi);
        if self.state.phi < 0.3 {
            bottlenecks.push("Low integrated information (Φ)".to_string());
        }

        // 2. WORKSPACE - Calculate REAL activation from conscious_contents
        //    Previously was hardcoded to 0.8 when not empty - now uses actual activations!
        let workspace_score = if self.state.conscious_contents.is_empty() {
            0.0
        } else {
            // Average activation of broadcasting contents, weighted by their number
            let broadcasting_items: Vec<_> = self
                .state
                .conscious_contents
                .iter()
                .filter(|c| c.is_broadcasting)
                .collect();

            if broadcasting_items.is_empty() {
                // Nothing broadcasting - low workspace access
                let avg_activation = self
                    .state
                    .conscious_contents
                    .iter()
                    .map(|c| c.activation)
                    .sum::<f64>()
                    / self.state.conscious_contents.len() as f64;
                avg_activation * 0.5 // Penalty for no broadcast
            } else {
                // Real workspace = average activation of broadcasting items
                let broadcast_activation =
                    broadcasting_items.iter().map(|c| c.activation).sum::<f64>()
                        / broadcasting_items.len() as f64;
                // Scale by proportion of items broadcasting (workspace access)
                let access_ratio =
                    broadcasting_items.len() as f64 / self.state.conscious_contents.len() as f64;
                broadcast_activation * (0.5 + 0.5 * access_ratio)
            }
        };
        scores.insert("workspace".to_string(), workspace_score.clamp(0.0, 1.0));
        if workspace_score < 0.4 {
            bottlenecks.push("Limited workspace access".to_string());
        }

        // 3. BINDING - Calculate REAL binding from synchrony and binding_strength
        //    Previously only counted conscious objects (always 0!) - now uses synchrony!
        let binding_score = if self.state.bound_objects.is_empty() {
            0.0
        } else {
            // Real binding = weighted combination of synchrony and binding_strength
            let total_synchrony: f64 = self
                .state
                .bound_objects
                .iter()
                .map(|b| b.synchrony * b.binding_strength)
                .sum();
            let avg_binding = total_synchrony / self.state.bound_objects.len() as f64;

            // Also factor in how many objects are bound (binding breadth)
            let bound_fraction = (self.state.bound_objects.len() as f64).min(10.0) / 10.0;

            // Combine strength with breadth
            avg_binding * 0.7 + bound_fraction * 0.3
        };
        scores.insert("binding".to_string(), binding_score.clamp(0.0, 1.0));
        if binding_score < 0.3 {
            bottlenecks.push("Weak feature binding".to_string());
        }

        // 4. HOT (Higher-Order Thought) - Calculate REAL meta-awareness
        //    Previously was hardcoded to 0.7 when not empty - now uses actual confidence!
        let hot_score = if self.state.meta_awareness.is_empty() {
            0.0
        } else {
            // Real HOT = weighted average of confidence by order level
            // Higher order thoughts (order > 1) contribute more to meta-awareness
            let weighted_confidence: f64 = self
                .state
                .meta_awareness
                .iter()
                .map(|m| m.confidence * (1.0 + 0.2 * m.order as f64))
                .sum();
            let total_weight: f64 = self
                .state
                .meta_awareness
                .iter()
                .map(|m| 1.0 + 0.2 * m.order as f64)
                .sum();

            // Also factor in depth of meta-cognition (max order reached)
            let max_order = self
                .state
                .meta_awareness
                .iter()
                .map(|m| m.order)
                .max()
                .unwrap_or(0) as f64;
            let depth_bonus = (max_order / 3.0).min(0.2);

            (weighted_confidence / total_weight) + depth_bonus
        };
        scores.insert("hot".to_string(), hot_score.clamp(0.0, 1.0));
        if hot_score < 0.3 {
            bottlenecks.push("Limited higher-order awareness".to_string());
        }

        // 5. Additional REAL metrics from state
        scores.insert(
            "temporal_coherence".to_string(),
            self.state.temporal_coherence,
        );
        scores.insert("embodiment".to_string(), self.state.embodiment);

        let avg_score = scores.values().sum::<f64>() / scores.len() as f64;
        let is_conscious = avg_score >= self.config.consciousness_threshold;

        IntegrationAssessment {
            is_conscious,
            consciousness_score: avg_score,
            component_scores: scores,
            bottlenecks,
            explanation: format!(
                "Consciousness score: {:.2} (phi={:.2}, workspace={:.2}, binding={:.2}, hot={:.2})",
                avg_score, self.state.phi, workspace_score, binding_score, hot_score
            ),
        }
    }

    /// Get current state
    pub fn get_state(&self) -> &ConsciousnessState {
        &self.state
    }

    /// Clear history and call `on_shutdown` on all subsystems.
    pub fn clear(&mut self) {
        for sub in &mut self.subsystems {
            sub.on_shutdown();
        }
        self.state = ConsciousnessState::default();
        self.history.clear();
        self.subsystem_errors.clear();
    }

    /// Set maximum history size (ring buffer capacity).
    pub fn set_max_history(&mut self, max_size: usize) {
        self.max_history_size = max_size.max(4);
        while self.history.len() > self.max_history_size {
            self.history.pop_front();
        }
    }

    /// Save a checkpoint of the pipeline's current state.
    ///
    /// Captures the consciousness state, configuration, history, and cycle count.
    /// Subsystems are NOT serialized (they must be re-registered after restore).
    pub fn save_checkpoint(&self) -> PipelineCheckpoint {
        PipelineCheckpoint {
            state: self.state.clone(),
            config: self.config.clone(),
            history: self.history.iter().cloned().collect(),
            max_history_size: self.max_history_size,
            current_cycle: self.current_cycle,
            embodiment_level: self.embodiment_level,
        }
    }

    /// Restore pipeline state from a checkpoint.
    ///
    /// Replaces the current state, config, and history. Subsystems are NOT affected
    /// — they must be re-registered separately.
    pub fn restore_checkpoint(&mut self, checkpoint: PipelineCheckpoint) {
        self.state = checkpoint.state;
        self.config = checkpoint.config;
        self.history = checkpoint.history.into_iter().collect();
        self.max_history_size = checkpoint.max_history_size;
        self.current_cycle = checkpoint.current_cycle;
        self.embodiment_level = checkpoint.embodiment_level;
    }
}

impl Default for ConsciousnessPipeline {
    fn default() -> Self {
        Self::default_new()
    }
}
