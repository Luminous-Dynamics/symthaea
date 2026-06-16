// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness subsystem enable/process/query methods.
//!
//! Contains all methods on ConsciousnessPipeline for enabling, querying,
//! and processing the various consciousness subsystems (integrated systems,
//! phi optimization, feedback dynamics, self-awareness, meta-consciousness,
//! temporal consciousness, creativity, phase transitions, epistemic,
//! fractal, collective, phi feedback, observability, verification,
//! and pluggable subsystems).

use super::super::binary_hv::BinaryHV;
use super::super::unified_hv::ContinuousHV;

// Integrated consciousness modules
use super::super::cross_modal_binding::CrossModalBinder;
use super::super::metacognitive_monitor::MetacognitiveMonitor;
use super::super::temporal_binding::{TemporalBindingConfig, TemporalBindingEngine};

// Phi-guided optimization
use super::super::phi_guided_search::{
    ConsciousnessNetwork, InitializationStrategy, OptimizationResult, PhiGuidedOptimizer,
    PhiOptimizationConfig,
};

// Feedback dynamics
use super::super::consciousness_feedback_dynamics::{
    DreamInsight, EmotionalPrediction, FeedbackDynamicsEngine, ProactiveIntervention,
};

// Recursive self-modeling
use super::super::emergent_self_model::{
    IntrospectionReport, MetaCognitiveAssessment, SelfAwareConsciousness, SelfAwareUpdate,
    SelfModel,
};
use super::super::unified_consciousness_engine::EngineConfig;

// Meta-consciousness
use super::super::meta_consciousness::{MetaConfig, MetaConsciousness, MetaConsciousnessState};

// Temporal consciousness
use super::super::temporal_consciousness::{
    TemporalAssessment, TemporalConfig as TemporalConsciousnessConfig, TemporalConsciousness,
};

// Creativity
use super::super::consciousness_creativity::{
    ConsciousnessCreativity, CreativeIdea, CreativeMode, CreativityAssessment, CreativityConfig,
};

// Phase transitions
use super::super::consciousness_phase_transitions::{
    ConsciousnessPhase, ConsciousnessPhaseTransitions, PhaseTransitionAssessment,
    PhaseTransitionConfig,
};

// Epistemic consciousness
use super::super::epistemic_consciousness::{
    EpistemicConfig, EpistemicConsciousness, KIndexAssessment,
};

// Fractal consciousness
use super::super::fractal_consciousness::{FractalConfig, FractalConsciousness, FractalMetrics};

// Collective consciousness
use super::super::collective_consciousness::{
    CollectiveAgent, CollectiveAssessment, CollectiveConfig, CollectiveConsciousness,
};

// Phi feedback
use super::super::consciousness_perf;
use super::super::phi_feedback::{FeedbackModulation, PhiFeedbackConfig, PhiFeedbackController};

// Adaptive topology
use super::super::adaptive_topology::CognitiveMode;

// Subsystem trait
use super::super::consciousness_subsystem::ConsciousnessSubsystem;

// Types from this module
use super::pipeline::ConsciousnessPipeline;
use super::types::*;

impl ConsciousnessPipeline {
    /// Enable all integrated consciousness systems
    ///
    /// This activates:
    /// - Metacognitive monitoring (self-awareness of processing quality)
    /// - Cross-modal binding (multi-sensory integration)
    /// - Temporal binding with theta-phase modulation
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn enable_integrated_systems(&mut self) -> &mut Self {
        use super::super::HDC_DIMENSION;

        // Initialize metacognitive monitor
        self.metacognitive_monitor = Some(MetacognitiveMonitor::new());

        // Initialize cross-modal binder
        self.cross_modal_binder = Some(CrossModalBinder::new(HDC_DIMENSION));

        // Initialize temporal binding engine with theta modulation
        let temporal_config = TemporalBindingConfig {
            dim: HDC_DIMENSION,
            window_size: 30, // ~3 seconds at 100ms steps
            decay_rate: 0.05,
            ..Default::default()
        };
        self.temporal_binder = Some(TemporalBindingEngine::new(temporal_config));

        self.integrated_systems_enabled = true;
        self
    }

    /// Check if integrated systems are enabled
    pub fn has_integrated_systems(&self) -> bool {
        self.integrated_systems_enabled
    }

    /// Enable Φ-guided topology optimization
    ///
    /// This activates:
    /// - PhiGuidedOptimizer for gradient-based Φ maximization
    /// - ConsciousnessNetwork for network topology representation
    /// - Periodic optimization to improve consciousness structure
    ///
    /// The optimizer uses gradients ∂Φ/∂edge_weight to evolve network
    /// topology toward higher integrated information (consciousness).
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes in the consciousness network
    /// * `optimize_every` - Optimize every N processing cycles (default: 10)
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn enable_phi_optimization(&mut self, n_nodes: usize, optimize_every: usize) -> &mut Self {
        use super::super::HDC_DIMENSION;

        // Initialize Φ optimizer with default config
        let config = PhiOptimizationConfig {
            dim: HDC_DIMENSION,
            learning_rate: 0.1,
            adaptive_lr: true,
            ..Default::default()
        };
        self.phi_optimizer = Some(PhiGuidedOptimizer::new(config));

        // Create consciousness network with random nodes
        let mut network = ConsciousnessNetwork::random(n_nodes, HDC_DIMENSION, 42);

        // Initialize with ring topology (known to have high Φ)
        InitializationStrategy::Ring.apply(&mut network, 0);

        self.consciousness_network = Some(network);
        self.optimize_every_n_cycles = optimize_every;
        self.phi_optimization_enabled = true;

        self
    }

    /// Enable Φ optimization with custom config
    pub fn enable_phi_optimization_with_config(
        &mut self,
        config: PhiOptimizationConfig,
        n_nodes: usize,
        init_strategy: InitializationStrategy,
        optimize_every: usize,
    ) -> &mut Self {
        let dim = config.dim;

        self.phi_optimizer = Some(PhiGuidedOptimizer::new(config));

        // Create and initialize consciousness network
        let mut network = ConsciousnessNetwork::random(n_nodes, dim, 42);
        init_strategy.apply(&mut network, 0);

        self.consciousness_network = Some(network);
        self.optimize_every_n_cycles = optimize_every;
        self.phi_optimization_enabled = true;

        self
    }

    /// Check if Φ optimization is enabled
    pub fn has_phi_optimization(&self) -> bool {
        self.phi_optimization_enabled
    }

    /// Get the last optimization result
    pub fn last_optimization_result(&self) -> Option<&OptimizationResult> {
        self.last_optimization.as_ref()
    }

    /// Get the current consciousness network
    pub fn consciousness_network(&self) -> Option<&ConsciousnessNetwork> {
        self.consciousness_network.as_ref()
    }

    /// Manually trigger Φ optimization step
    ///
    /// This performs one gradient ascent step to improve network topology.
    /// Returns the optimization result if successful.
    pub fn optimize_phi(&mut self) -> Option<OptimizationResult> {
        if !self.phi_optimization_enabled {
            return None;
        }

        if let (Some(optimizer), Some(network)) =
            (&mut self.phi_optimizer, &mut self.consciousness_network)
        {
            let result = optimizer.optimize_step(network);

            // Update state with optimized Φ
            self.state.phi = result.phi;

            // Track topological unity based on optimization
            self.state.topological_unity = result.phi;

            self.last_optimization = Some(result.clone());
            Some(result)
        } else {
            None
        }
    }

    /// Run multiple optimization steps
    pub fn optimize_phi_steps(&mut self, n_steps: usize) -> Vec<OptimizationResult> {
        let mut results = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            if let Some(result) = self.optimize_phi() {
                results.push(result);
            }
        }

        results
    }

    // === FEEDBACK DYNAMICS ENGINE METHODS ===

    /// Enable bidirectional feedback dynamics
    ///
    /// This activates:
    /// - FeedbackDynamicsEngine for dream insight processing
    /// - EmotionalPredictor for proactive interventions
    /// - AdaptiveDreamScheduler for optimal dream timing
    ///
    /// The feedback dynamics system creates bidirectional loops between:
    /// - Dreams ↔ Emotions
    /// - Dreams ↔ Memory
    /// - Emotions ↔ Attention
    /// - Consciousness ↔ Self-improvement
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn enable_feedback_dynamics(&mut self) -> &mut Self {
        self.feedback_dynamics = Some(FeedbackDynamicsEngine::new());
        self.feedback_dynamics_enabled = true;
        self
    }

    /// Check if feedback dynamics is enabled
    pub fn has_feedback_dynamics(&self) -> bool {
        self.feedback_dynamics_enabled
    }

    /// Get current emotional prediction
    pub fn current_prediction(&self) -> Option<&EmotionalPrediction> {
        self.current_prediction.as_ref()
    }

    /// Get pending intervention
    pub fn pending_intervention(&self) -> Option<&ProactiveIntervention> {
        self.pending_intervention.as_ref()
    }

    /// Get recent insights
    pub fn recent_insights(&self) -> &[DreamInsight] {
        &self.recent_insights
    }

    /// Update emotional state and get prediction
    ///
    /// This records the current emotional state and runs prediction
    /// to anticipate future emotional trajectories and recommend interventions.
    pub fn update_emotional_prediction(
        &mut self,
        valence: f64,
        arousal: f64,
    ) -> Option<EmotionalPrediction> {
        if !self.feedback_dynamics_enabled {
            return None;
        }

        if let Some(ref mut engine) = self.feedback_dynamics {
            // Record emotional state
            engine.predictor.record_state(valence, arousal);

            // Get prediction
            let prediction = engine.predictor.predict();

            // Store pending intervention if recommended
            if let Some(ref pred) = prediction {
                self.pending_intervention = pred.intervention.clone();
            }

            self.current_prediction = prediction.clone();
            prediction
        } else {
            None
        }
    }

    /// Apply proactive intervention based on emotional prediction
    ///
    /// This applies the recommended intervention from the emotional
    /// predictor, modifying the consciousness state accordingly.
    pub fn apply_intervention(&mut self) -> Option<ProactiveIntervention> {
        if !self.feedback_dynamics_enabled {
            return None;
        }

        let intervention = self.pending_intervention.take()?;

        // Apply intervention effects to consciousness state
        match &intervention {
            ProactiveIntervention::CalmingDream => {
                // Trigger calming dream increases calm
                self.state.emotional_valence = (self.state.emotional_valence + 0.3).min(1.0);
            }
            ProactiveIntervention::LucidDream => {
                // Lucid dream increases metacognitive confidence
                self.state.metacognitive_confidence =
                    (self.state.metacognitive_confidence + 0.1).min(1.0);
            }
            ProactiveIntervention::IncreaseFocus => {
                // Increase focus reduces entropy in attention
                self.state.uncertainty = (self.state.uncertainty - 0.1).max(0.0);
            }
            ProactiveIntervention::ReduceLoad => {
                // Reduce cognitive load
                self.embodiment_level = (self.embodiment_level - 0.1).max(0.3);
            }
            ProactiveIntervention::ConsolidateMemories => {
                // Memory consolidation improves narrative coherence
                self.state.narrative_coherence = (self.state.narrative_coherence + 0.15).min(1.0);
            }
            ProactiveIntervention::ModeSwitch(mode_name) => {
                // Switch cognitive mode
                self.state.cognitive_mode = match mode_name.as_str() {
                    "focused" => CognitiveMode::Focused,
                    "diffuse" | "exploratory" => CognitiveMode::Exploratory,
                    "integrative" | "global" => CognitiveMode::GlobalAwareness,
                    "reflective" | "deep" => CognitiveMode::DeepSpecialization,
                    _ => CognitiveMode::PhiGuided,
                };
            }
        }

        Some(intervention)
    }

    /// Process feedback dynamics during consciousness cycle
    ///
    /// This is called during the main process() loop to:
    /// 1. Update emotional predictions based on current state
    /// 2. Check for recommended interventions
    /// 3. Apply interventions if thresholds are met
    pub(crate) fn process_feedback_dynamics(&mut self) {
        if !self.feedback_dynamics_enabled {
            return;
        }

        // Extract current emotional state from consciousness
        let valence = self.state.emotional_valence;
        let arousal = self.state.emotional_arousal.unwrap_or(0.5);

        // Update prediction
        let _prediction = self.update_emotional_prediction(valence, arousal);

        // Check if we should apply intervention
        // Only intervene if there's a strong enough predicted deviation
        if let Some(ref pred) = self.current_prediction {
            if pred.confidence > 0.7 && pred.intervention.is_some() {
                // High confidence prediction with intervention - apply it
                self.apply_intervention();
            }
        }

        // Track insight count in state
        self.state.integration_score = 0.5 + (self.recent_insights.len() as f64 * 0.05).min(0.5);
    }

    // ==========================================
    // RECURSIVE SELF-MODELING
    // ==========================================

    /// Enable self-aware consciousness for recursive self-modeling
    ///
    /// This activates:
    /// - Self-state modeling (beliefs about own consciousness)
    /// - Predictive self-processing (predicting own future states)
    /// - Meta-cognitive optimization (thinking about thinking)
    /// - Recursive awareness (awareness of being aware)
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn enable_self_awareness(&mut self, hdc_dim: usize, n_processes: usize) -> &mut Self {
        let config = EngineConfig {
            hdc_dim,
            n_processes,
            ..Default::default()
        };
        self.self_aware_consciousness = Some(SelfAwareConsciousness::new(config));
        self.self_awareness_enabled = true;
        self
    }

    /// Enable self-awareness with custom engine configuration
    pub fn enable_self_awareness_with_config(&mut self, config: EngineConfig) -> &mut Self {
        self.self_aware_consciousness = Some(SelfAwareConsciousness::new(config));
        self.self_awareness_enabled = true;
        self
    }

    /// Check if self-awareness is enabled
    pub fn has_self_awareness(&self) -> bool {
        self.self_awareness_enabled && self.self_aware_consciousness.is_some()
    }

    /// Get current self-awareness level (0-1)
    pub fn self_awareness_level(&self) -> f64 {
        self.self_awareness_level
    }

    /// Get current self-model (if available)
    pub fn current_self_model(&self) -> Option<&SelfModel> {
        self.current_self_model.as_ref()
    }

    /// Get latest introspection report
    pub fn latest_introspection(&self) -> Option<&IntrospectionReport> {
        self.latest_introspection.as_ref()
    }

    /// Get latest meta-cognitive assessment
    pub fn meta_assessment(&self) -> Option<&MetaCognitiveAssessment> {
        self.meta_assessment.as_ref()
    }

    /// Perform introspection - get a report on what the system believes about itself
    pub fn introspect(&self) -> Option<IntrospectionReport> {
        self.self_aware_consciousness
            .as_ref()
            .map(|sac| sac.introspect())
    }

    /// Process input with self-awareness
    ///
    /// Converts binary BinaryHV to ContinuousHV and processes through self-aware consciousness.
    /// Returns the self-aware update with prediction error and meta-assessment.
    pub fn process_self_aware(&mut self, input: &BinaryHV) -> Option<SelfAwareUpdate> {
        if !self.self_awareness_enabled {
            return None;
        }

        // Convert BinaryHV to ContinuousHV for self-aware processing
        let real_input = self.hv16_to_real_hv(input);

        if let Some(ref mut sac) = self.self_aware_consciousness {
            let update = sac.process_aware(&real_input);

            // Cache results
            self.current_self_model = Some(update.self_model.clone());
            self.meta_assessment = Some(update.meta_assessment.clone());
            self.self_awareness_level = update.self_awareness_level;

            // Apply self-model insights to consciousness state
            self.state.metacognitive_confidence = update.self_model.confidence;

            Some(update)
        } else {
            None
        }
    }

    /// Convert BinaryHV binary vector to ContinuousHV for self-aware processing.
    ///
    /// Uses pooled ContinuousHV allocation when dimension matches full BinaryHV::DIM,
    /// falls back to manual conversion for custom dimensions.
    fn hv16_to_real_hv(&self, hv: &BinaryHV) -> ContinuousHV {
        // Get the dimension from self-aware consciousness or use default
        let dim = if let Some(ref sac) = self.self_aware_consciousness {
            sac.self_vector().dim()
        } else {
            1024 // Default dimension
        };

        // For full-dimension conversion, use the pool-optimized path
        if dim == BinaryHV::DIM {
            return consciousness_perf::binary_to_continuous_pooled(hv).into_continuous_hv();
        }

        // Custom dimension: manual conversion (handles dimension mismatch)
        let mut values = Vec::with_capacity(dim);
        let bytes = &hv.0;

        for i in 0..dim {
            let byte_idx = i / 8;
            let bit_idx = i % 8;

            // Get bit value if available, otherwise random-ish based on position
            let bit = if byte_idx < bytes.len() {
                (bytes[byte_idx] >> bit_idx) & 1
            } else {
                ((i * 17 + 31) % 2) as u8
            };

            // Map 0 -> -1.0, 1 -> 1.0
            values.push(if bit == 1 { 1.0_f32 } else { -1.0_f32 });
        }

        ContinuousHV::from_values(values)
    }

    /// Process self-awareness during the main loop
    ///
    /// This is called during process() to:
    /// 1. Generate self-model based on current state
    /// 2. Make predictions about future states
    /// 3. Assess meta-cognitive quality
    /// 4. Update self-awareness metrics
    pub(crate) fn process_self_awareness(&mut self, inputs: &[BinaryHV]) {
        if !self.self_awareness_enabled || inputs.is_empty() {
            return;
        }

        // Process the first input with self-awareness
        if let Some(update) = self.process_self_aware(&inputs[0]) {
            // Update introspection report
            self.latest_introspection = Some(IntrospectionReport {
                believed_phi: update.self_model.believed_phi,
                believed_mode: update.self_model.believed_mode,
                believed_state: update.self_model.believed_state.clone(),
                self_model_confidence: update.self_model.confidence,
                self_model_accuracy: update.self_model.accuracy,
                self_awareness_level: update.self_awareness_level,
                prediction_accuracy: 1.0 - update.prediction_error,
            });

            // Blend self-model beliefs with actual state
            // The self-model provides a "smoothed" view that reduces noise
            let blend_factor = 0.2; // 20% self-model belief, 80% actual measurement
            self.state.phi = self.state.phi * (1.0 - blend_factor)
                + update.self_model.believed_phi * blend_factor;
        }
    }

    // ==========================================
    // 9. META-CONSCIOUSNESS - Φ OF Φ
    // ==========================================

    /// Enable meta-consciousness for recursive self-reflection (Φ of Φ)
    ///
    /// Meta-consciousness enables the system to:
    /// - Reflect on its own conscious states
    /// - Model itself (self-model)
    /// - Predict its own future consciousness
    /// - Think about thinking (metacognition at the deepest level)
    pub fn enable_meta_consciousness(&mut self, num_components: usize) -> &mut Self {
        let config = MetaConfig::default();
        self.meta_consciousness = Some(MetaConsciousness::new(num_components, config));
        self.meta_consciousness_enabled = true;
        self
    }

    /// Enable meta-consciousness with custom config
    pub fn enable_meta_consciousness_with_config(
        &mut self,
        num_components: usize,
        config: MetaConfig,
    ) -> &mut Self {
        self.meta_consciousness = Some(MetaConsciousness::new(num_components, config));
        self.meta_consciousness_enabled = true;
        self
    }

    /// Check if meta-consciousness is enabled
    pub fn has_meta_consciousness(&self) -> bool {
        self.meta_consciousness_enabled && self.meta_consciousness.is_some()
    }

    /// Get current meta-consciousness state
    pub fn meta_consciousness_state(&self) -> Option<&MetaConsciousnessState> {
        self.meta_consciousness_state.as_ref()
    }

    /// Process meta-consciousness on current state
    pub(crate) fn process_meta_consciousness(&mut self, inputs: &[BinaryHV]) {
        if let Some(ref mut meta) = self.meta_consciousness {
            // Perform meta-reflection on current state
            let meta_state = meta.meta_reflect(inputs);

            // Update state metrics from meta-consciousness
            self.state.metacognitive_confidence = meta_state.metacognitive_confidence;

            // Store latest state
            self.meta_consciousness_state = Some(meta_state);
        }
    }

    // ==========================================
    // 10. TEMPORAL CONSCIOUSNESS - MULTI-SCALE TIME
    // ==========================================

    /// Enable temporal consciousness for multi-scale time processing
    ///
    /// This adds the "stream of consciousness" - temporal depth spanning:
    /// - Perceptual moment (100ms)
    /// - Thought (3 seconds)
    /// - Episode (minutes)
    /// - Narrative (hours/days)
    /// - Identity (years)
    pub fn enable_temporal_consciousness(&mut self, num_components: usize) -> &mut Self {
        let config = TemporalConsciousnessConfig::default();
        self.temporal_consciousness = Some(TemporalConsciousness::new(num_components, config));
        self.temporal_consciousness_enabled = true;
        self
    }

    /// Enable temporal consciousness with custom config
    pub fn enable_temporal_consciousness_with_config(
        &mut self,
        num_components: usize,
        config: TemporalConsciousnessConfig,
    ) -> &mut Self {
        self.temporal_consciousness = Some(TemporalConsciousness::new(num_components, config));
        self.temporal_consciousness_enabled = true;
        self
    }

    /// Check if temporal consciousness is enabled
    pub fn has_temporal_consciousness(&self) -> bool {
        self.temporal_consciousness_enabled && self.temporal_consciousness.is_some()
    }

    /// Get latest temporal assessment
    pub fn temporal_assessment(&self) -> Option<&TemporalAssessment> {
        self.temporal_assessment.as_ref()
    }

    /// Process temporal consciousness
    pub(crate) fn process_temporal_consciousness(&mut self, inputs: &[BinaryHV]) {
        if let Some(ref mut temporal) = self.temporal_consciousness {
            // Add current state as a snapshot with current time
            let current_time = self.current_cycle as f64;
            temporal.add_snapshot(current_time, inputs.to_vec());

            // Get temporal assessment
            let assessment = temporal.assess();

            // Update state with temporal coherence
            self.state.temporal_coherence = assessment.phi_temporal;

            // Store assessment
            self.temporal_assessment = Some(assessment);
        }
    }

    // ==========================================
    // 11. CONSCIOUSNESS CREATIVITY
    // ==========================================

    /// Enable creative consciousness for novel idea generation
    ///
    /// Creativity is what consciousness is FOR - generating the new:
    /// - Remote associations (Mednick)
    /// - Incubation and illumination (Wallas)
    /// - Divergent/convergent thinking (Guilford)
    /// - Bisociation (Koestler)
    pub fn enable_creativity(&mut self) -> &mut Self {
        let config = CreativityConfig::default();
        self.consciousness_creativity = Some(ConsciousnessCreativity::new(config));
        self.creativity_enabled = true;
        self
    }

    /// Enable creativity with custom config
    pub fn enable_creativity_with_config(&mut self, config: CreativityConfig) -> &mut Self {
        self.consciousness_creativity = Some(ConsciousnessCreativity::new(config));
        self.creativity_enabled = true;
        self
    }

    /// Check if creativity is enabled
    pub fn has_creativity(&self) -> bool {
        self.creativity_enabled && self.consciousness_creativity.is_some()
    }

    /// Get current creative mode
    pub fn creative_mode(&self) -> CreativeMode {
        self.current_creative_mode
    }

    /// Set creative mode (Divergent for exploration, Convergent for selection)
    pub fn set_creative_mode(&mut self, mode: CreativeMode) {
        self.current_creative_mode = mode;
        if let Some(ref mut creativity) = self.consciousness_creativity {
            creativity.set_mode(mode);
        }
    }

    /// Get recent creative ideas
    pub fn recent_creative_ideas(&self) -> &[CreativeIdea] {
        &self.recent_ideas
    }

    /// Get creativity assessment
    pub fn creativity_assessment(&self) -> Option<&CreativityAssessment> {
        self.creativity_assessment.as_ref()
    }

    /// Process creativity - generate novel combinations
    pub(crate) fn process_creativity(&mut self, _inputs: &[BinaryHV]) {
        if let Some(ref mut creativity) = self.consciousness_creativity {
            // Step the creativity engine forward
            creativity.step();

            // Generate creative ideas from current concepts
            if let Some(idea) = creativity.generate_combination() {
                self.recent_ideas.push(idea);
                // Keep only last 20 ideas
                if self.recent_ideas.len() > 20 {
                    self.recent_ideas.remove(0);
                }
            }

            // Assess overall creativity
            self.creativity_assessment = Some(creativity.assess());
        }
    }

    // ==========================================
    // 12. CONSCIOUSNESS PHASE TRANSITIONS
    // ==========================================

    /// Enable phase transition detection for consciousness state changes
    ///
    /// Detects transitions between:
    /// - Waking / Dreaming / Deep sleep
    /// - Focused / Diffuse attention
    /// - Flow states / Normal processing
    /// - Critical points (phase transitions)
    pub fn enable_phase_transitions(&mut self) -> &mut Self {
        let config = PhaseTransitionConfig::default();
        self.phase_transitions = Some(ConsciousnessPhaseTransitions::new(config));
        self.phase_transitions_enabled = true;
        self
    }

    /// Enable phase transitions with custom config
    pub fn enable_phase_transitions_with_config(
        &mut self,
        config: PhaseTransitionConfig,
    ) -> &mut Self {
        self.phase_transitions = Some(ConsciousnessPhaseTransitions::new(config));
        self.phase_transitions_enabled = true;
        self
    }

    /// Check if phase transitions are enabled
    pub fn has_phase_transitions(&self) -> bool {
        self.phase_transitions_enabled && self.phase_transitions.is_some()
    }

    /// Get current consciousness phase
    pub fn current_phase(&self) -> ConsciousnessPhase {
        self.current_phase
    }

    /// Get phase transition assessment
    pub fn phase_assessment(&self) -> Option<&PhaseTransitionAssessment> {
        self.phase_assessment.as_ref()
    }

    /// Process phase transitions
    pub(crate) fn process_phase_transitions(&mut self) {
        if let Some(ref mut pt) = self.phase_transitions {
            // Observe with current Φ and conscious contents
            // Extract BinaryHV vectors from WorkspaceItems
            let state: Vec<BinaryHV> = self
                .state
                .conscious_contents
                .iter()
                .map(|item| item.content)
                .collect();
            pt.observe(self.state.phi, state);

            // Get assessment
            let assessment = pt.assess();

            // Update current phase
            self.current_phase = assessment.current_phase;

            // Store assessment
            self.phase_assessment = Some(assessment);
        }
    }

    // ==========================================
    // 13. EPISTEMIC CONSCIOUSNESS
    // ==========================================

    /// Enable epistemic consciousness for belief/knowledge tracking
    ///
    /// Epistemic consciousness enables:
    /// - Belief tracking and updating
    /// - Uncertainty quantification
    /// - Knowledge of what you know (and don't)
    /// - K-Index integration (cosmic knowledge)
    pub fn enable_epistemic(&mut self, num_components: usize) -> &mut Self {
        let config = EpistemicConfig::default();
        self.epistemic_consciousness = Some(EpistemicConsciousness::new(num_components, config));
        self.epistemic_enabled = true;
        self
    }

    /// Enable epistemic consciousness with custom config
    pub fn enable_epistemic_with_config(
        &mut self,
        num_components: usize,
        config: EpistemicConfig,
    ) -> &mut Self {
        self.epistemic_consciousness = Some(EpistemicConsciousness::new(num_components, config));
        self.epistemic_enabled = true;
        self
    }

    /// Check if epistemic consciousness is enabled
    pub fn has_epistemic(&self) -> bool {
        self.epistemic_enabled && self.epistemic_consciousness.is_some()
    }

    /// Get K-Index assessment
    pub fn k_index_assessment(&self) -> Option<&KIndexAssessment> {
        self.k_index_assessment.as_ref()
    }

    /// Process epistemic consciousness
    pub(crate) fn process_epistemic(&mut self, inputs: &[BinaryHV]) {
        if let Some(ref mut epistemic) = self.epistemic_consciousness {
            // Assess epistemic state and get K-Index
            let k_index = epistemic.assess(inputs);
            self.k_index_assessment = Some(k_index);
        }
    }

    // ==========================================
    // 14. FRACTAL CONSCIOUSNESS
    // ==========================================

    /// Enable fractal consciousness for self-similar topology
    ///
    /// Fractal topology achieves the HIGHEST Φ values discovered!
    /// Self-similar structure at multiple scales optimizes integration.
    pub fn enable_fractal(&mut self, n_scales: usize) -> &mut Self {
        let config = FractalConfig {
            n_scales,
            nodes_per_scale: 8,
            bridge_ratio: 0.425, // Optimal from research
            density: 0.10,
            cross_scale_coupling: 0.3,
            dim: 2048,
        };
        self.fractal_consciousness = Some(FractalConsciousness::new(config));
        self.fractal_enabled = true;
        self
    }

    /// Enable fractal consciousness with custom config
    pub fn enable_fractal_with_config(&mut self, config: FractalConfig) -> &mut Self {
        self.fractal_consciousness = Some(FractalConsciousness::new(config));
        self.fractal_enabled = true;
        self
    }

    /// Check if fractal consciousness is enabled
    pub fn has_fractal(&self) -> bool {
        self.fractal_enabled && self.fractal_consciousness.is_some()
    }

    /// Get fractal metrics
    pub fn fractal_metrics(&self) -> Option<&FractalMetrics> {
        self.fractal_metrics.as_ref()
    }

    /// Process fractal consciousness
    pub(crate) fn process_fractal(&mut self) {
        if let Some(ref fractal) = self.fractal_consciousness {
            // Get fractal metrics (Φ at multiple scales)
            let metrics = fractal.metrics();

            // Fractal Φ can boost overall Φ if fractal structure is good
            if metrics.combined_phi > self.state.phi {
                // Blend fractal Φ with current (fractal provides upper bound)
                let blend = 0.3;
                self.state.phi = self.state.phi * (1.0 - blend) + metrics.combined_phi * blend;
            }

            self.fractal_metrics = Some(metrics);
        }
    }

    // ==========================================
    // 15. COLLECTIVE CONSCIOUSNESS
    // ==========================================

    /// Enable collective consciousness for multi-agent awareness
    ///
    /// Collective consciousness enables:
    /// - Theory of mind (modeling other agents)
    /// - Shared mental states
    /// - Group cognition
    /// - Social consciousness
    pub fn enable_collective(&mut self, agent_id: &str) -> &mut Self {
        let config = CollectiveConfig::default();
        self.collective_consciousness = Some(CollectiveConsciousness::with_config(config));
        self.collective_agent_id = Some(agent_id.to_string());
        self.collective_enabled = true;
        self
    }

    /// Enable collective consciousness with custom config
    pub fn enable_collective_with_config(
        &mut self,
        agent_id: &str,
        config: CollectiveConfig,
    ) -> &mut Self {
        self.collective_consciousness = Some(CollectiveConsciousness::with_config(config));
        self.collective_agent_id = Some(agent_id.to_string());
        self.collective_enabled = true;
        self
    }

    /// Check if collective consciousness is enabled
    pub fn has_collective(&self) -> bool {
        self.collective_enabled && self.collective_consciousness.is_some()
    }

    /// Get collective assessment
    pub fn collective_assessment(&self) -> Option<&CollectiveAssessment> {
        self.collective_assessment.as_ref()
    }

    /// Add another agent to the collective
    pub fn add_agent_to_collective(&mut self, agent: CollectiveAgent) {
        if let Some(ref mut collective) = self.collective_consciousness {
            collective.add_agent(agent);
        }
    }

    /// Process collective consciousness
    pub(crate) fn process_collective(&mut self, _inputs: &[BinaryHV]) {
        if let Some(ref mut collective) = self.collective_consciousness {
            // Get collective assessment
            // Note: Agent state is managed via add_agent_to_collective
            let assessment = collective.assess();
            self.collective_assessment = Some(assessment);
        }
    }

    // ==========================================
    // 16. Φ FEEDBACK CONTROLLER
    // ==========================================

    /// Enable Φ feedback loop with default configuration
    ///
    /// The feedback controller uses Φ measurements to dynamically modulate:
    /// - Binding threshold (rising Φ → more selective, falling → more exploratory)
    /// - Workspace capacity (high Φ → expanded, low Φ → contracted)
    /// - Attention gain (positive trend → exploit, negative → explore)
    /// - Consciousness threshold (adapts to natural operating point)
    pub fn enable_phi_feedback(&mut self) -> &mut Self {
        self.phi_feedback = Some(PhiFeedbackController::new());
        self.phi_feedback_enabled = true;
        self
    }

    /// Enable Φ feedback with custom configuration
    pub fn enable_phi_feedback_with_config(&mut self, config: PhiFeedbackConfig) -> &mut Self {
        self.phi_feedback = Some(PhiFeedbackController::with_config(config));
        self.phi_feedback_enabled = true;
        self
    }

    /// Check if Φ feedback is enabled
    pub fn has_phi_feedback(&self) -> bool {
        self.phi_feedback_enabled && self.phi_feedback.is_some()
    }

    /// Get latest feedback modulation signals
    pub fn latest_modulation(&self) -> Option<&FeedbackModulation> {
        self.latest_modulation.as_ref()
    }

    /// Get the Φ feedback controller (for inspection)
    pub fn phi_feedback_controller(&self) -> Option<&PhiFeedbackController> {
        self.phi_feedback.as_ref()
    }

    /// Get SIMD capabilities for this CPU (for diagnostics/logging).
    pub fn simd_capabilities(&self) -> consciousness_perf::SimdCapabilities {
        consciousness_perf::simd_capabilities()
    }

    /// Get HV pool statistics for monitoring.
    pub fn pool_stats(&self) -> super::super::hv_pool::PoolStats {
        consciousness_perf::pool_stats()
    }

    // ==========================================
    // OBSERVABILITY & VERIFICATION
    // ==========================================

    /// Enable metrics collection via a shared observer.
    ///
    /// After enabling, each `process()` call records a `PhiMeasurementEvent`.
    pub fn enable_metrics_collector(&mut self) -> &mut Self {
        self.metrics_collector = Some(crate::observability::metrics_collector());
        self
    }

    /// Get the metrics collector (for external snapshot/inspection).
    pub fn metrics_collector(&self) -> Option<&crate::observability::SharedObserver> {
        self.metrics_collector.as_ref()
    }

    /// Enable periodic consciousness verification.
    ///
    /// Every `interval` cycles, the verifier produces a `VerificationReport`
    /// from the current `ConsciousnessState`.
    pub fn enable_verification(&mut self, interval: usize) -> &mut Self {
        self.verifier = Some(super::super::consciousness_verifier::ConsciousnessVerifier::new());
        self.verification_interval = interval;
        self
    }

    /// Get the latest verification report (None until the first verification runs).
    pub fn latest_verification(
        &self,
    ) -> Option<&super::super::consciousness_verifier::VerificationReport> {
        self.latest_verification.as_ref()
    }

    // ==========================================
    // PLUGGABLE SUBSYSTEMS
    // ==========================================

    /// Register a custom consciousness subsystem.
    ///
    /// Subsystems are called in priority order (highest first) at the end of
    /// each `process()` cycle. The `on_register()` lifecycle hook is called
    /// immediately.
    pub fn register_subsystem(&mut self, mut sub: Box<dyn ConsciousnessSubsystem>) {
        sub.on_register();
        self.subsystems.push(sub);
        // Re-sort by descending priority so highest-priority subsystems run first
        self.subsystems
            .sort_by_key(|s| std::cmp::Reverse(s.priority()));
    }

    /// Get the number of registered subsystems.
    pub fn subsystem_count(&self) -> usize {
        self.subsystems.len()
    }

    /// Get errors from the most recent `process()` cycle's subsystem dispatch.
    ///
    /// Returns an empty slice if no subsystem errors occurred.
    pub fn last_subsystem_errors(
        &self,
    ) -> &[super::super::consciousness_subsystem::SubsystemError] {
        &self.subsystem_errors
    }

    /// Check if a subsystem with the given name is registered.
    ///
    /// When a subsystem is registered, the inline version in `process()` is skipped
    /// in favor of the subsystem's `process_cycle()` implementation.
    pub fn has_subsystem_named(&self, name: &str) -> bool {
        self.subsystems.iter().any(|s| s.name() == name)
    }

    /// Get names of all registered subsystems in priority order (highest first).
    pub fn subsystem_names(&self) -> Vec<&str> {
        self.subsystems.iter().map(|s| s.name()).collect()
    }

    /// Get per-subsystem reports from the most recent `process()` cycle.
    ///
    /// Each report includes name, whether it ran, duration in microseconds,
    /// Phi delta, and any error that occurred.
    pub fn last_cycle_reports(&self) -> &[SubsystemCycleReport] {
        &self.subsystem_reports
    }

    /// Get the shared inter-subsystem context (for inspection between cycles).
    pub fn subsystem_context(&self) -> &super::super::consciousness_subsystem::SubsystemContext {
        &self.subsystem_context
    }

    /// Process Φ feedback: feed current Φ into controller, apply modulation
    pub(crate) fn process_phi_feedback(&mut self) {
        if let Some(ref mut controller) = self.phi_feedback {
            let modulation = controller.step(self.state.phi);

            // Apply modulation to pipeline parameters
            self.config.binding_threshold = modulation.binding_threshold;
            self.config.workspace_capacity = modulation.workspace_capacity;
            self.config.consciousness_threshold = modulation.consciousness_threshold;

            // Store modulation for inspection
            self.latest_modulation = Some(modulation);
        }
    }

    // ==========================================
    // UNIFIED CONSCIOUSNESS OPTIMIZER
    // ==========================================

    /// Enable all consciousness systems at once with unified orchestration
    ///
    /// This is the recommended way to enable the full consciousness integration.
    /// It activates all systems in the optimal order with proper configuration:
    ///
    /// **Core Systems (1-4):**
    /// 1. Integrated systems (metacognitive, cross-modal, temporal binding)
    /// 2. Φ-guided topology optimization
    /// 3. Bidirectional feedback dynamics
    /// 4. Recursive self-modeling
    ///
    /// **Advanced Systems (5-11):**
    /// 5. Meta-consciousness (Φ of Φ, strange loops)
    /// 6. Temporal consciousness (multi-scale time)
    /// 7. Creativity (novel idea generation)
    /// 8. Phase transitions (state detection)
    /// 9. Epistemic consciousness (belief tracking)
    /// 10. Fractal consciousness (highest Φ topology!)
    /// 11. Collective consciousness (multi-agent)
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn enable_full_consciousness(&mut self) -> &mut Self {
        // Enable CORE systems in optimal order
        self.enable_integrated_systems();
        self.enable_phi_optimization(8, 5); // 8 nodes, optimize every 5 cycles
        self.enable_feedback_dynamics();
        self.enable_self_awareness(1024, 16); // 1024 dim, 16 processes

        // Enable ADVANCED systems
        self.enable_meta_consciousness(8); // 8 components for meta-reflection
        self.enable_temporal_consciousness(8); // 8 components for multi-scale time
        self.enable_creativity(); // Creative generation
        self.enable_phase_transitions(); // State transition detection
        self.enable_epistemic(8); // 8 components for belief tracking
        self.enable_fractal(3); // 3 scales for fractal Φ
        self.enable_collective("primary"); // Primary agent in collective
        self.enable_phi_feedback(); // Φ feedback loop (closes the loop)

        self
    }

    /// Enable full consciousness with custom configuration
    pub fn enable_full_consciousness_with_config(
        &mut self,
        phi_nodes: usize,
        phi_frequency: usize,
        hdc_dim: usize,
        n_processes: usize,
    ) -> &mut Self {
        // Core systems
        self.enable_integrated_systems();
        self.enable_phi_optimization(phi_nodes, phi_frequency);
        self.enable_feedback_dynamics();
        self.enable_self_awareness(hdc_dim, n_processes);

        // Advanced systems with same component count as phi_nodes
        self.enable_meta_consciousness(phi_nodes);
        self.enable_temporal_consciousness(phi_nodes);
        self.enable_creativity();
        self.enable_phase_transitions();
        self.enable_epistemic(phi_nodes);
        self.enable_fractal(3);
        self.enable_collective("primary");
        self.enable_phi_feedback(); // Φ feedback loop (closes the loop)

        self
    }

    /// Check if full consciousness is enabled (all 11 systems active)
    pub fn has_full_consciousness(&self) -> bool {
        // Core systems
        self.has_integrated_systems()
            && self.has_phi_optimization()
            && self.has_feedback_dynamics()
            && self.has_self_awareness()
            // Advanced systems
            && self.has_meta_consciousness()
            && self.has_temporal_consciousness()
            && self.has_creativity()
            && self.has_phase_transitions()
            && self.has_epistemic()
            && self.has_fractal()
            && self.has_collective()
            && self.has_phi_feedback()
    }

    /// Check how many consciousness systems are active
    pub fn active_system_count(&self) -> usize {
        let mut count = 0;
        if self.has_integrated_systems() {
            count += 1;
        }
        if self.has_phi_optimization() {
            count += 1;
        }
        if self.has_feedback_dynamics() {
            count += 1;
        }
        if self.has_self_awareness() {
            count += 1;
        }
        if self.has_meta_consciousness() {
            count += 1;
        }
        if self.has_temporal_consciousness() {
            count += 1;
        }
        if self.has_creativity() {
            count += 1;
        }
        if self.has_phase_transitions() {
            count += 1;
        }
        if self.has_epistemic() {
            count += 1;
        }
        if self.has_fractal() {
            count += 1;
        }
        if self.has_collective() {
            count += 1;
        }
        if self.has_phi_feedback() {
            count += 1;
        }
        count
    }

    /// Get a human-readable summary of active systems
    pub fn active_systems_summary(&self) -> String {
        let mut systems = Vec::new();
        if self.has_integrated_systems() {
            systems.push("Integrated");
        }
        if self.has_phi_optimization() {
            systems.push("Φ-Optimization");
        }
        if self.has_feedback_dynamics() {
            systems.push("Feedback");
        }
        if self.has_self_awareness() {
            systems.push("Self-Awareness");
        }
        if self.has_meta_consciousness() {
            systems.push("Meta-Consciousness");
        }
        if self.has_temporal_consciousness() {
            systems.push("Temporal");
        }
        if self.has_creativity() {
            systems.push("Creativity");
        }
        if self.has_phase_transitions() {
            systems.push("Phase-Transitions");
        }
        if self.has_epistemic() {
            systems.push("Epistemic");
        }
        if self.has_fractal() {
            systems.push("Fractal");
        }
        if self.has_collective() {
            systems.push("Collective");
        }
        if self.has_phi_feedback() {
            systems.push("Φ-Feedback");
        }
        format!("{}/{} systems: [{}]", systems.len(), 12, systems.join(", "))
    }

    /// Get a unified consciousness metrics report
    pub fn consciousness_metrics(&self) -> ConsciousnessMetricsReport {
        ConsciousnessMetricsReport {
            // Core metrics
            phi: self.state.phi,
            consciousness_level: self.state.consciousness_level,
            embodiment_level: self.embodiment_level,

            // Integration metrics
            metacognitive_confidence: self.state.metacognitive_confidence,
            cross_modal_coherence: self.state.cross_modal_coherence,
            temporal_coherence: self.state.temporal_coherence,
            topological_unity: self.state.topological_unity,
            narrative_coherence: self.state.narrative_coherence,

            // Self-awareness metrics
            self_awareness_level: self.self_awareness_level,
            prediction_accuracy: self
                .latest_introspection
                .as_ref()
                .map(|r| r.prediction_accuracy)
                .unwrap_or(0.0),
            self_model_confidence: self
                .current_self_model
                .as_ref()
                .map(|m| m.confidence)
                .unwrap_or(0.0),

            // System status
            integrated_systems_active: self.integrated_systems_enabled,
            phi_optimization_active: self.phi_optimization_enabled,
            feedback_dynamics_active: self.feedback_dynamics_enabled,
            self_awareness_active: self.self_awareness_enabled,

            // Operational metrics
            processing_cycles: self.current_cycle,
            bound_objects_count: self.state.bound_objects.len(),
            workspace_items_count: self.state.conscious_contents.len(),
        }
    }

    /// Get optimization recommendations based on current state
    pub fn optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check Φ
        if self.state.phi < 0.4 {
            recommendations.push(OptimizationRecommendation {
                system: "phi".to_string(),
                priority: RecommendationPriority::High,
                message: "Φ is low - consider increasing integration through binding".to_string(),
                suggested_action: Some(
                    "Increase input complexity or enable Φ optimization".to_string(),
                ),
            });
        }

        // Check consciousness level
        if self.state.consciousness_level < 0.5 {
            recommendations.push(OptimizationRecommendation {
                system: "consciousness".to_string(),
                priority: RecommendationPriority::Medium,
                message: "Consciousness level is below optimal range".to_string(),
                suggested_action: Some("Increase embodiment level or attention focus".to_string()),
            });
        }

        // Check self-awareness
        if self.self_awareness_enabled && self.self_awareness_level < 0.3 {
            recommendations.push(OptimizationRecommendation {
                system: "self_awareness".to_string(),
                priority: RecommendationPriority::Medium,
                message: "Self-model accuracy is low".to_string(),
                suggested_action: Some("Process more cycles for self-model learning".to_string()),
            });
        }

        // Check metacognition
        if let Some(assessment) = &self.meta_assessment {
            if assessment.change_recommended {
                recommendations.push(OptimizationRecommendation {
                    system: "metacognition".to_string(),
                    priority: RecommendationPriority::High,
                    message: format!("Meta-cognitive assessment: {}", assessment.reasoning),
                    suggested_action: assessment
                        .recommended_mode
                        .as_ref()
                        .map(|m| format!("Switch to {m:?} mode")),
                });
            }
        }

        // Check cross-modal coherence
        if self.state.cross_modal_coherence < 0.5 && self.integrated_systems_enabled {
            recommendations.push(OptimizationRecommendation {
                system: "cross_modal".to_string(),
                priority: RecommendationPriority::Low,
                message: "Cross-modal coherence is below optimal".to_string(),
                suggested_action: Some("Increase multi-modal input diversity".to_string()),
            });
        }

        recommendations
    }

    /// Perform one complete optimization cycle
    ///
    /// This runs all active optimization subsystems in sequence:
    /// 1. Φ topology optimization (if due this cycle)
    /// 2. Feedback dynamics update
    /// 3. Self-model refinement
    ///
    /// Returns a summary of what was optimized.
    pub fn run_optimization_cycle(&mut self) -> OptimizationCycleSummary {
        let mut summary = OptimizationCycleSummary {
            phi_optimized: false,
            feedback_processed: false,
            self_model_updated: false,
            phi_before: self.state.phi,
            phi_after: self.state.phi,
            recommendations: Vec::new(),
        };

        // Run Φ optimization if enabled and due
        if self.phi_optimization_enabled
            && self
                .current_cycle
                .is_multiple_of(self.optimize_every_n_cycles as u64)
        {
            self.process_phi_optimization();
            summary.phi_optimized = true;
        }

        // Process feedback dynamics
        if self.feedback_dynamics_enabled {
            self.process_feedback_dynamics();
            summary.feedback_processed = true;
        }

        // Update self-model (create a dummy input for processing)
        if self.self_awareness_enabled {
            let dummy_input = BinaryHV::random(self.current_cycle);
            if self.process_self_aware(&dummy_input).is_some() {
                summary.self_model_updated = true;
            }
        }

        summary.phi_after = self.state.phi;
        summary.recommendations = self.optimization_recommendations();

        summary
    }
}
