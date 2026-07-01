// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The Unified Conscious Being
//!
//! Core struct, configuration, statistics, and primary interaction logic.
//! This is the central integration point that ties all consciousness subsystems together.

use super::super::adaptive_topology::CognitiveMode;
use super::super::binary_hv::BinaryHV;
use super::super::causal_mind::{CausalExplanation, CausalMind, CausalPrediction};
use super::super::consciousness_advanced_cognition::AdvancedCognitionEngine;
use super::super::consciousness_cross_integration::ConsciousnessIntegrationBridge;
use super::super::consciousness_feedback_dynamics::FeedbackDynamicsEngine;
use super::super::consciousness_metacognition::MetacognitionEngine;
use super::super::counterfactual_dreams::CounterfactualDreamEngine;
use super::super::cross_modal_attention_router::CrossModalAttentionRouter;
use super::super::cross_modal_attention_router::ModalityInput;
use super::super::cross_modal_binding::Modality;
use super::super::emotional_depth::EmotionalDepthSystem;
use super::super::full_stack_consciousness::{ConsciousComprehension, FullStackConsciousness};
use super::super::math_bridge::{MathResult, MathValue, UnifiedMathEngine};
use super::super::self_improvement_integration::SelfImprovementSystem;
use super::super::unified_cognitive_core::{QueryResult as UCEQueryResult, UnifiedCognitiveCore};
use super::super::unified_understanding::DeepUnderstanding;
use super::dialogue::{
    ConsciousDialogueGenerator, ConsciousResponse, DialogueContext, DialogueStyle,
};
use super::do_calculus::{CounterfactualResult, InterventionResult, StructuralCausalModel};
use crate::physics::simulation_bridge::{PhysicsSimulator, SimulationAnalysis, state_to_binary_hv};
use std::collections::{HashMap, VecDeque};

/// Complete consciousness integration stats
#[derive(Debug, Clone)]
pub struct ConsciousnessStats {
    /// Total inputs processed
    pub inputs_processed: u64,
    /// Total responses generated
    pub responses_generated: u64,
    /// Average Phi level
    pub avg_phi: f64,
    /// Peak Phi achieved
    pub peak_phi: f64,
    /// Memories stored
    pub memories_stored: u64,
    /// Causal edges learned
    pub causal_edges: usize,
    /// Counterfactuals explored
    pub counterfactuals_explored: u64,
}

/// The complete unified conscious being
/// A physics simulation result that has been processed through consciousness.
#[derive(Debug, Clone)]
pub struct PhysicsInsight {
    /// Name of the physical system simulated.
    pub system_name: String,
    /// Phi measured during conscious processing of this simulation.
    pub phi: f64,
    /// Whether the system was classified as chaotic.
    pub is_chaotic: bool,
    /// Energy drift (if applicable).
    pub energy_drift: Option<f64>,
    /// Number of trajectory states processed.
    pub num_states: usize,
}

/// Result of a conscious mathematical computation.
#[derive(Debug, Clone)]
pub struct MathInsight {
    /// The computed value.
    pub result: MathResult,
    /// Phi from the consciousness pipeline processing this computation.
    pub consciousness_phi: f64,
    /// Domain promotions that occurred.
    pub promotions: Vec<String>,
}

pub struct UnifiedConsciousBeing {
    /// Full stack consciousness (understanding + inference + memory + counterfactuals)
    pub(super) full_stack: FullStackConsciousness,
    /// Structural causal model (Pearl do-calculus)
    pub(super) causal_model: StructuralCausalModel,
    /// HDC-native causal reasoning (CausalMind - revolutionary integration)
    pub(super) causal_mind: CausalMind,
    /// Unified Cognitive Core (UCE/UCTS - concept = meaning x causality x temporality)
    pub(super) cognitive_core: UnifiedCognitiveCore,
    /// Emotional depth system (complex blends, compound emotions, trajectories)
    pub(super) emotional_depth: EmotionalDepthSystem,
    /// Cross-modal attention router (Phi-gated modality routing)
    pub(super) attention_router: CrossModalAttentionRouter,
    /// Self-improvement system (metacognitive optimization)
    pub(super) self_improvement: SelfImprovementSystem,
    /// Counterfactual dreams engine
    pub(super) counterfactual_dreams: CounterfactualDreamEngine,
    /// Cross-module integration bridge (emotional->dreams, stress->dreams)
    pub(super) integration_bridge: ConsciousnessIntegrationBridge,
    /// Feedback dynamics engine (bidirectional loops, prediction, collective, scheduling)
    pub(super) feedback_dynamics: FeedbackDynamicsEngine,
    /// Metacognition engine (self-monitoring, temporal, narrative, state machine)
    pub(super) metacognition: MetacognitionEngine,
    /// Advanced cognition (motor imagery, theory of mind, imagination, predictive, memory, drives)
    pub(super) advanced_cognition: AdvancedCognitionEngine,
    /// Current cognitive mode
    pub(super) current_mode: CognitiveMode,
    /// Dialogue generator
    pub(super) dialogue: ConsciousDialogueGenerator,
    /// Current flow state
    pub(super) flow_state: f32,
    /// Phi trend (for pacing)
    pub(super) phi_history: VecDeque<f64>,
    /// Statistics
    pub(super) stats: ConsciousnessStats,
    /// Configuration
    pub(super) config: BeingConfig,
    /// Unified math engine (lazy-initialized via enable_math_bridge)
    pub(super) math_engine: Option<UnifiedMathEngine>,
    /// Physics simulation insights accumulated across interactions
    pub(super) physics_insights: Vec<PhysicsInsight>,
}

#[derive(Debug, Clone)]
pub struct BeingConfig {
    /// Enable voice output
    pub voice_enabled: bool,
    /// Enable counterfactual reasoning
    pub counterfactuals_enabled: bool,
    /// Maximum memory traces
    pub max_memories: usize,
    /// Dialogue style
    pub dialogue_style: DialogueStyle,
}

impl Default for BeingConfig {
    fn default() -> Self {
        Self {
            voice_enabled: false, // Disabled by default (requires Kokoro)
            counterfactuals_enabled: true,
            max_memories: 1000,
            dialogue_style: DialogueStyle::Empathetic,
        }
    }
}

/// Complete interaction result
#[derive(Debug, Clone)]
pub struct InteractionResult {
    /// The conscious comprehension
    pub comprehension: ConsciousComprehension,
    /// The generated response
    pub response: ConsciousResponse,
    /// Do-calculus results (if causal structure detected)
    pub causal_analysis: Option<InterventionResult>,
    /// Pearl counterfactual (if applicable)
    pub pearl_counterfactual: Option<CounterfactualResult>,
    /// Overall interaction quality
    pub quality_score: f64,
}

impl UnifiedConsciousBeing {
    pub fn new() -> Self {
        Self::with_config(BeingConfig::default())
    }

    pub fn with_config(config: BeingConfig) -> Self {
        let dialogue = ConsciousDialogueGenerator::new().with_style(config.dialogue_style);

        Self {
            full_stack: FullStackConsciousness::new()
                .with_counterfactuals(config.counterfactuals_enabled, 2),
            causal_model: StructuralCausalModel::new(),
            causal_mind: CausalMind::new(),
            cognitive_core: UnifiedCognitiveCore::new(),
            emotional_depth: EmotionalDepthSystem::new(),
            attention_router: CrossModalAttentionRouter::new(),
            self_improvement: SelfImprovementSystem::new(),
            counterfactual_dreams: CounterfactualDreamEngine::new(),
            integration_bridge: ConsciousnessIntegrationBridge::new(),
            feedback_dynamics: FeedbackDynamicsEngine::new(),
            metacognition: MetacognitionEngine::new(),
            advanced_cognition: AdvancedCognitionEngine::new(),
            current_mode: CognitiveMode::Balanced,
            dialogue,
            flow_state: 0.5,
            phi_history: VecDeque::with_capacity(50),
            stats: ConsciousnessStats {
                inputs_processed: 0,
                responses_generated: 0,
                avg_phi: 0.0,
                peak_phi: 0.0,
                memories_stored: 0,
                causal_edges: 0,
                counterfactuals_explored: 0,
            },
            config,
            math_engine: None,
            physics_insights: Vec::new(),
        }
    }

    /// Process input and generate conscious response
    pub fn interact(&mut self, input: &str) -> InteractionResult {
        self.stats.inputs_processed += 1;

        // 1. Full stack comprehension
        let comprehension = self.full_stack.comprehend(input);

        // 2. Update phi history and flow state
        self.phi_history.push_back(comprehension.consciousness_phi);
        while self.phi_history.len() > 50 {
            self.phi_history.pop_front();
        }
        self.update_flow_state();

        // 3. Update Pearl structural causal model
        self.causal_model
            .learn_from_understanding(&comprehension.understanding);

        // 4. Update HDC-native CausalMind (revolutionary: causality in the vector!)
        self.causal_mind.learn_from_text(input);

        // 5. Update UnifiedCognitiveCore (UCE: concept = meaning x causality x temporality)
        self.cognitive_core.learn_from_text(input);

        // Aggregate causal edges from all systems
        self.stats.causal_edges =
            self.causal_model.equation_count() + self.causal_mind.link_count();

        // 6. Perform Pearl do-calculus if causal structure present
        let causal_analysis = self.perform_causal_analysis(&comprehension.understanding);

        // 7. Perform Pearl counterfactual if applicable
        let pearl_counterfactual = self.perform_pearl_counterfactual(&comprehension.understanding);

        // 8. Build dialogue context
        let context = DialogueContext {
            understanding: comprehension.understanding.clone(),
            metacognition: comprehension.metacognition.clone(),
            memories: comprehension.memory.recalled_memories.clone(),
            counterfactuals: comprehension.counterfactuals.clone(),
            phi: comprehension.consciousness_phi,
            valence: comprehension.understanding.grounded.embodied.valence,
            arousal: comprehension.understanding.grounded.embodied.arousal,
            flow_state: self.flow_state,
        };

        // 9. Generate conscious response
        let response = self.dialogue.generate(&context);
        self.stats.responses_generated += 1;
        self.stats.counterfactuals_explored += comprehension.counterfactuals.len() as u64;

        // 10. Update statistics
        self.stats.memories_stored = self.full_stack.memory_count() as u64;
        self.update_avg_phi(comprehension.consciousness_phi);
        if comprehension.consciousness_phi > self.stats.peak_phi {
            self.stats.peak_phi = comprehension.consciousness_phi;
        }

        // 11. Calculate interaction quality
        let quality_score = self.calculate_quality(&comprehension, &response);

        InteractionResult {
            comprehension,
            response,
            causal_analysis,
            pearl_counterfactual,
            quality_score,
        }
    }

    pub(super) fn update_flow_state(&mut self) {
        if self.phi_history.len() < 2 {
            return;
        }

        // Flow increases with consistent high phi
        let recent_avg: f64 = self.phi_history.iter().rev().take(5).sum::<f64>() / 5.0;
        let variance: f64 = self
            .phi_history
            .iter()
            .rev()
            .take(5)
            .map(|p| (p - recent_avg).powi(2))
            .sum::<f64>()
            / 5.0;

        // High average + low variance = flow state
        self.flow_state = (recent_avg as f32 * (1.0 - variance.sqrt() as f32)).clamp(0.0, 1.0);
    }

    fn perform_causal_analysis(
        &self,
        understanding: &DeepUnderstanding,
    ) -> Option<InterventionResult> {
        let causal = understanding.grounded.causal_structure.as_ref()?;

        // Perform do-intervention: what if cause didn't happen?
        Some(self.causal_model.do_intervention(
            &causal.cause,
            0.0, // Negate the cause
            &causal.effect,
        ))
    }

    fn perform_pearl_counterfactual(
        &self,
        understanding: &DeepUnderstanding,
    ) -> Option<CounterfactualResult> {
        let causal = understanding.grounded.causal_structure.as_ref()?;

        // Build evidence from current state
        let mut evidence = HashMap::new();
        evidence.insert(causal.cause.clone(), 1.0);
        evidence.insert(causal.effect.clone(), 1.0);

        // Counterfactual: what if cause had been different?
        Some(self.causal_model.counterfactual(
            &evidence,
            &causal.cause,
            0.0, // Counterfactual: cause didn't happen
            &causal.effect,
        ))
    }

    pub(super) fn update_avg_phi(&mut self, phi: f64) {
        let n = self.stats.inputs_processed as f64;
        self.stats.avg_phi = (self.stats.avg_phi * (n - 1.0) + phi) / n;
    }

    fn calculate_quality(&self, comp: &ConsciousComprehension, resp: &ConsciousResponse) -> f64 {
        let phi_factor = comp.consciousness_phi;
        let confidence_factor = resp.confidence;
        let coherence_factor = comp.metacognition.coherence;

        (phi_factor * 0.4 + confidence_factor * 0.3 + coherence_factor * 0.3).min(1.0)
    }

    /// Get current flow state
    pub fn flow_state(&self) -> f32 {
        self.flow_state
    }

    /// Get phi trend
    pub fn phi_trend(&self) -> f64 {
        if self.phi_history.len() < 2 {
            return 0.0;
        }
        let recent = self.phi_history.back().copied().unwrap_or(0.0);
        let older = self.phi_history.front().copied().unwrap_or(0.0);
        (recent - older) / self.phi_history.len() as f64
    }

    /// Get statistics
    pub fn stats(&self) -> &ConsciousnessStats {
        &self.stats
    }

    /// Get causal model size
    pub fn causal_model_size(&self) -> (usize, usize) {
        (
            self.causal_model.variable_count(),
            self.causal_model.equation_count(),
        )
    }

    /// Clear history for new conversation
    pub fn clear(&mut self) {
        self.full_stack.clear();
        self.causal_model = StructuralCausalModel::new();
        self.causal_mind = CausalMind::new();
        self.cognitive_core = UnifiedCognitiveCore::new();
        self.phi_history.clear();
        self.flow_state = 0.5;
        self.physics_insights.clear();
    }

    /// Ask a Pearl counterfactual question
    pub fn ask_pearl_counterfactual(
        &self,
        evidence: HashMap<String, f64>,
        intervention_var: &str,
        intervention_val: f64,
        target: &str,
    ) -> CounterfactualResult {
        self.causal_model
            .counterfactual(&evidence, intervention_var, intervention_val, target)
    }

    // =========================================================================
    // CAUSAL MIND QUERIES (HDC-Native Causal Reasoning)
    // =========================================================================

    /// Query CausalMind: Why did X happen? (find causes in HDC space)
    pub fn causal_query_why(&self, concept: &str) -> Vec<CausalExplanation> {
        self.causal_mind.query_why(concept)
    }

    /// Query CausalMind: What if X happens? (predict effects in HDC space)
    pub fn causal_query_what_if(&self, concept: &str) -> Vec<CausalPrediction> {
        self.causal_mind.query_what_if(concept)
    }

    /// Query CausalMind: What if we intervene on X? (do-calculus in HDC)
    pub fn causal_query_intervention(
        &self,
        concept: &str,
        min_strength: f64,
    ) -> Vec<CausalPrediction> {
        self.causal_mind.query_intervention(concept, min_strength)
    }

    /// Get CausalMind's Phi (integrated information from causal structure)
    pub fn causal_mind_phi(&self) -> f64 {
        self.causal_mind.phi()
    }

    // =========================================================================
    // UNIFIED COGNITIVE CORE QUERIES (UCE/UCTS Architecture)
    // =========================================================================

    /// Query UnifiedCognitiveCore: Why did X happen? (causes in unified space)
    pub fn uce_query_why(&self, label: &str) -> Vec<UCEQueryResult> {
        self.cognitive_core.query_why(label)
    }

    /// Query UnifiedCognitiveCore: What does X cause? (effects in unified space)
    pub fn uce_query_effects(&self, label: &str) -> Vec<UCEQueryResult> {
        self.cognitive_core.query_effects(label)
    }

    /// Query UnifiedCognitiveCore: What comes after X? (temporal successors)
    pub fn uce_query_successors(&self, label: &str) -> Vec<UCEQueryResult> {
        self.cognitive_core.query_successors(label)
    }

    /// Find similar concepts in UnifiedCognitiveCore
    pub fn uce_find_similar(&self, label: &str, limit: usize) -> Vec<UCEQueryResult> {
        self.cognitive_core.find_similar(label, limit)
    }

    /// Get UnifiedCognitiveCore's Phi (integrated information from unified elements)
    pub fn uce_phi(&self) -> f64 {
        self.cognitive_core.phi()
    }

    /// Get element count in UnifiedCognitiveCore
    pub fn uce_element_count(&self) -> usize {
        self.cognitive_core.element_count()
    }

    // =========================================================================
    // COMBINED COGNITIVE METRICS
    // =========================================================================

    /// Get integrated Phi from all causal reasoning systems
    ///
    /// Combines:
    /// - Pearl SCM equation count
    /// - CausalMind HDC Phi
    /// - UnifiedCognitiveCore Phi
    pub fn integrated_causal_phi(&self) -> f64 {
        let pearl_contribution = self.causal_model.equation_count() as f64 * 0.1;
        let causal_mind_phi = self.causal_mind.phi();
        let uce_phi = self.cognitive_core.phi();

        // Weighted integration (UCE contributes more as it's the unified representation)
        (pearl_contribution * 0.2 + causal_mind_phi * 0.3 + uce_phi * 0.5).min(1.0)
    }

    /// Get full causal system statistics
    pub fn causal_system_stats(&self) -> CausalSystemStats {
        CausalSystemStats {
            pearl_variables: self.causal_model.variable_count(),
            pearl_equations: self.causal_model.equation_count(),
            causal_mind_concepts: self.causal_mind.concept_count(),
            causal_mind_links: self.causal_mind.link_count(),
            causal_mind_phi: self.causal_mind.phi(),
            uce_elements: self.cognitive_core.element_count(),
            uce_phi: self.cognitive_core.phi(),
            integrated_phi: self.integrated_causal_phi(),
        }
    }

    // =========================================================================
    // PHYSICS & MATH BRIDGE INTEGRATION
    // =========================================================================

    /// Enable the math bridge, initializing the UnifiedMathEngine.
    ///
    /// This creates the engine (which allocates a PrimitiveSystem singleton)
    /// so the being can perform conscious mathematical computation.
    pub fn enable_math_bridge(&mut self) -> &mut Self {
        if self.math_engine.is_none() {
            self.math_engine = Some(UnifiedMathEngine::new());
        }
        self
    }

    /// Perform a conscious mathematical computation.
    ///
    /// Runs the operation through the UnifiedMathEngine, then routes the
    /// resulting BinaryHV encoding through the attention router as a
    /// Semantic modality input. Returns the math result enriched with
    /// consciousness Phi from the being's current state.
    ///
    /// Auto-enables the math bridge if not yet initialized.
    pub fn conscious_compute(
        &mut self,
        op: &str,
        a: &MathValue,
        b: &MathValue,
    ) -> Option<MathInsight> {
        if self.math_engine.is_none() {
            self.enable_math_bridge();
        }
        let engine = self.math_engine.as_ref()?;

        let result = match op {
            "add" | "+" => Some(engine.add(a, b)),
            "subtract" | "-" => Some(engine.subtract(a, b)),
            "multiply" | "*" => Some(engine.multiply(a, b)),
            "divide" | "/" => engine.divide(a, b),
            "sqrt" => Some(engine.sqrt(a)),
            "power" | "^" => Some(engine.power(a, b)),
            _ => None,
        }?;

        // Route through attention as Semantic input
        let input = ModalityInput {
            modality: Modality::Semantic,
            hv: result.encoding,
            salience: result.phi.min(1.0),
            confidence: 0.9,
            timestamp: 0,
            label: Some(format!("{} {} {} = {}", a, op, b, result.value)),
        };
        let current_phi = self.phi_history.back().copied().unwrap_or(0.5);
        let routing = self.attention_router.route(&[input], current_phi);

        // Update phi history with the math operation's Phi
        let consciousness_phi = result.phi * 0.6 + routing.effective_phi * 0.4;
        self.phi_history.push_back(consciousness_phi);
        while self.phi_history.len() > 50 {
            self.phi_history.pop_front();
        }
        self.update_flow_state();

        Some(MathInsight {
            promotions: result.domain_promotions.clone(),
            consciousness_phi,
            result,
        })
    }

    /// Simulate a physical system and process the trajectory through consciousness.
    ///
    /// Creates a PhysicsSimulator, runs the simulation, encodes trajectory
    /// states as BinaryHV, and routes them through the attention router as
    /// Proprioceptive (embodied physics) inputs. Returns a PhysicsInsight
    /// with consciousness metrics.
    pub fn conscious_simulate(
        &mut self,
        simulator: PhysicsSimulator,
        t_end: f64,
        dt: f64,
        sample_count: usize,
    ) -> PhysicsInsight {
        let system_name = simulator.name.clone();

        // 1. Run simulation
        let result = simulator.simulate(t_end, dt);

        // 2. Analyze trajectory
        let analysis = SimulationAnalysis::from_result(&result);
        let is_chaotic = analysis.is_chaotic().unwrap_or(false);

        // 3. Sample and encode as BinaryHV
        let step = (result.states.len() / sample_count.max(1)).max(1);
        let binary_hvs: Vec<BinaryHV> = result
            .states
            .iter()
            .step_by(step)
            .take(sample_count)
            .map(|state| state_to_binary_hv(state))
            .collect();

        // 4. Route through attention as Proprioceptive (embodied physics)
        let inputs: Vec<ModalityInput> = binary_hvs
            .iter()
            .enumerate()
            .map(|(i, hv)| ModalityInput {
                modality: Modality::Proprioceptive,
                hv: *hv,
                salience: 0.7 + 0.3 * (i as f64 / sample_count.max(1) as f64),
                confidence: 0.85,
                timestamp: i as u64,
                label: Some(format!("{system_name}[t={i}]")),
            })
            .collect();

        let current_phi = self.phi_history.back().copied().unwrap_or(0.5);
        let routing = self.attention_router.route(&inputs, current_phi);

        // 5. Update consciousness state
        let phi = routing.effective_phi;
        self.phi_history.push_back(phi);
        while self.phi_history.len() > 50 {
            self.phi_history.pop_front();
        }
        self.update_flow_state();

        let insight = PhysicsInsight {
            system_name,
            phi,
            is_chaotic,
            energy_drift: analysis.energy_drift,
            num_states: binary_hvs.len(),
        };

        self.physics_insights.push(insight.clone());
        insight
    }

    /// Get accumulated physics insights.
    pub fn physics_insights(&self) -> &[PhysicsInsight] {
        &self.physics_insights
    }

    /// Check if the math bridge is enabled.
    pub fn has_math_bridge(&self) -> bool {
        self.math_engine.is_some()
    }
}

/// Statistics for the integrated causal reasoning systems
#[derive(Debug, Clone)]
pub struct CausalSystemStats {
    /// Pearl SCM variable count
    pub pearl_variables: usize,
    /// Pearl SCM equation count
    pub pearl_equations: usize,
    /// CausalMind concept count
    pub causal_mind_concepts: usize,
    /// CausalMind causal link count
    pub causal_mind_links: usize,
    /// CausalMind Phi
    pub causal_mind_phi: f64,
    /// UCE element count
    pub uce_elements: usize,
    /// UCE Phi
    pub uce_phi: f64,
    /// Integrated Phi across all systems
    pub integrated_phi: f64,
}

impl Default for UnifiedConsciousBeing {
    fn default() -> Self {
        Self::new()
    }
}
