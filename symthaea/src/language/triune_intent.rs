// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Triune Intent Integration - A+B+C Consensus Architecture
//!
//! Combines three "brain regions" for hallucination-resistant intent understanding:
//!
//! - **Neocortex (A)**: LLM symbolic parsing → structured intent
//! - **Hippocampus (B)**: HDC+CfC temporal/algebraic intuition
//! - **Homeostat (C)**: Active Inference divergence detection
//!
//! When A and B diverge, Free Energy spikes and the system asks for clarification
//! instead of hallucinating.

use crate::hdc::code_encoder::CodeHDEncoder;
use crate::hdc::code_memory::CodebaseMemory;
use std::collections::HashMap;
use symthaea_core::hdc::ContinuousHV;

// ============================================================================
// Structured Intent (Output of Path A - LLM)
// ============================================================================

/// The structured output from LLM symbolic parsing
#[derive(Debug, Clone)]
pub struct SymbolicIntent {
    /// Primary action(s) with confidence weights
    pub actions: Vec<(IntentAction, f32)>,
    /// What we're operating on
    pub target: IntentTarget,
    /// Language constraint
    pub language: Option<String>,
    /// Pattern hints (e.g., "quicksort", "async", "no unsafe")
    pub patterns: Vec<String>,
    /// Edge cases to handle
    pub edge_cases: Vec<String>,
    /// References to existing code
    pub context_refs: Vec<ContextRef>,
    /// Raw LLM reasoning (for debugging)
    pub reasoning: Option<String>,
}

/// Actions the user might want
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntentAction {
    Create,
    Modify,
    Explain,
    Find,
    Refactor,
    Debug,
    Delete,
    Test,
}

/// What code entity is being targeted
#[derive(Debug, Clone)]
pub struct IntentTarget {
    pub kind: TargetKind,
    pub name: Option<String>,
    pub path: Option<String>,
    pub line_range: Option<(usize, usize)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetKind {
    Function,
    Struct,
    Module,
    File,
    Codebase,
    Unspecified,
}

/// Reference to existing code context
#[derive(Debug, Clone)]
pub struct ContextRef {
    pub description: String,
    pub resolved_path: Option<String>,
    pub resolved_name: Option<String>,
}

// ============================================================================
// Intuitive Intent (Output of Path B - HDC+CfC)
// ============================================================================

/// The intuitive understanding from HDC algebraic reasoning
#[derive(Debug, Clone)]
pub struct IntuitiveIntent {
    /// The intent as a hypervector (composable, algebraically queryable)
    pub intent_hv: ContinuousHV,
    /// Trajectory HV from CfC (encodes conversation history)
    pub trajectory_hv: ContinuousHV,
    /// Algebraic intent: what change is implied? (goal - current)
    pub delta_hv: ContinuousHV,
    /// Similar past intents from memory
    pub similar_intents: Vec<(String, f32)>,
    /// Confidence based on codebase familiarity
    pub grounding_confidence: f32,
}

// ============================================================================
// Divergence Result (Output of Path C - Active Inference)
// ============================================================================

/// The result of comparing symbolic and intuitive intents
#[derive(Debug, Clone)]
pub struct DivergenceAnalysis {
    /// Free Energy / Surprise (0.0 = perfect agreement, 1.0 = total divergence)
    pub free_energy: f32,
    /// Which aspects diverge most
    pub divergence_sources: Vec<DivergenceSource>,
    /// Suggested clarification if energy is high
    pub clarification_question: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DivergenceSource {
    pub aspect: String,
    pub symbolic_value: String,
    pub intuitive_similarity: f32,
    pub explanation: String,
}

// ============================================================================
// Final Intent Result
// ============================================================================

/// The final result of triune intent resolution
#[derive(Debug, Clone)]
pub enum IntentResult {
    /// Consensus reached - safe to proceed
    Actionable {
        intent: SymbolicIntent,
        intuition: IntuitiveIntent,
        confidence: f32,
    },
    /// High divergence - need clarification
    NeedsClarification {
        partial_intent: SymbolicIntent,
        intuition: IntuitiveIntent,
        divergence: DivergenceAnalysis,
    },
    /// Cannot understand at all
    Incomprehensible { raw_input: String, reason: String },
}

// ============================================================================
// The Triune Integrator
// ============================================================================

/// Configuration for the triune system
#[derive(Debug, Clone)]
pub struct TriuneConfig {
    /// Free energy threshold above which we ask for clarification
    pub confusion_threshold: f32,
    /// Minimum confidence to consider intent grounded
    pub grounding_threshold: f32,
    /// HDC dimension
    pub hdc_dim: usize,
    /// CfC hidden dimension
    pub cfc_hidden_dim: usize,
    /// Time constant for CfC decay
    pub cfc_tau: f32,
}

impl Default for TriuneConfig {
    fn default() -> Self {
        Self {
            confusion_threshold: 0.35,
            grounding_threshold: 0.5,
            hdc_dim: 4096,
            cfc_hidden_dim: 256,
            cfc_tau: 1.0,
        }
    }
}

/// The Triune Intent Integrator - combines LLM, HDC+CfC, and Active Inference
pub struct TriuneIntentIntegrator {
    config: TriuneConfig,

    // Path A: Symbolic (LLM)
    llm_parser: LLMIntentParser,

    // Path B: Intuitive (HDC + CfC)
    encoder: CodeHDEncoder,
    cfc_state: CfCTemporalState,

    // Action prototype vectors for encoding symbolic plans
    action_prototypes: HashMap<IntentAction, ContinuousHV>,
    target_prototypes: HashMap<TargetKind, ContinuousHV>,
}

impl TriuneIntentIntegrator {
    /// Create a new integrator with default config
    pub fn new(hdc_dim: usize) -> Self {
        Self::with_config(TriuneConfig {
            hdc_dim,
            ..Default::default()
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: TriuneConfig) -> Self {
        let encoder = CodeHDEncoder::new(config.hdc_dim);
        let cfc_state = CfCTemporalState::new(config.cfc_hidden_dim, config.cfc_tau);

        // Initialize action prototypes
        let action_prototypes = Self::init_action_prototypes(config.hdc_dim);
        let target_prototypes = Self::init_target_prototypes(config.hdc_dim);

        Self {
            config,
            llm_parser: LLMIntentParser::new(),
            encoder,
            cfc_state,
            action_prototypes,
            target_prototypes,
        }
    }

    /// Initialize HDC prototypes for actions
    fn init_action_prototypes(dim: usize) -> HashMap<IntentAction, ContinuousHV> {
        let mut map = HashMap::new();

        // Each action gets a random but fixed prototype
        let actions = [
            IntentAction::Create,
            IntentAction::Modify,
            IntentAction::Explain,
            IntentAction::Find,
            IntentAction::Refactor,
            IntentAction::Debug,
            IntentAction::Delete,
            IntentAction::Test,
        ];

        for (i, action) in actions.iter().enumerate() {
            map.insert(*action, ContinuousHV::random(dim, 0xAC710000 + i as u64));
        }

        map
    }

    /// Initialize HDC prototypes for targets
    fn init_target_prototypes(dim: usize) -> HashMap<TargetKind, ContinuousHV> {
        let mut map = HashMap::new();

        let targets = [
            TargetKind::Function,
            TargetKind::Struct,
            TargetKind::Module,
            TargetKind::File,
            TargetKind::Codebase,
            TargetKind::Unspecified,
        ];

        for (i, target) in targets.iter().enumerate() {
            map.insert(*target, ContinuousHV::random(dim, 0x7A860000 + i as u64));
        }

        map
    }

    // ========================================================================
    // The Main Algorithm
    // ========================================================================

    /// Resolve intent using the triune consensus mechanism
    pub fn resolve_intent(&mut self, input: &str, codebase: &CodebaseMemory) -> IntentResult {
        // ====================================================================
        // PATH A: Symbolic Parsing (LLM - The Neocortex)
        // ====================================================================
        let symbolic_intent = self.llm_parser.parse_to_struct(input);

        // ====================================================================
        // PATH B: Intuitive Encoding (HDC + CfC - The Hippocampus)
        // ====================================================================
        let intuitive_intent = self.compute_intuitive_intent(input, codebase);

        // ====================================================================
        // PATH C: Active Inference Check (The Homeostat)
        // ====================================================================
        let divergence = self.compute_divergence(&symbolic_intent, &intuitive_intent, codebase);

        // ====================================================================
        // Decision: Consensus or Clarification?
        // ====================================================================
        if divergence.free_energy > self.config.confusion_threshold {
            // High surprise! The LLM and HDC disagree.
            IntentResult::NeedsClarification {
                partial_intent: symbolic_intent,
                intuition: intuitive_intent,
                divergence,
            }
        } else {
            // Low energy - consensus reached
            let confidence = 1.0 - divergence.free_energy;
            IntentResult::Actionable {
                intent: symbolic_intent,
                intuition: intuitive_intent,
                confidence,
            }
        }
    }

    // ========================================================================
    // Path B: Intuitive Intent Computation
    // ========================================================================

    fn compute_intuitive_intent(
        &mut self,
        input: &str,
        codebase: &CodebaseMemory,
    ) -> IntuitiveIntent {
        // Encode the input text as HDC
        let input_hv = self.encoder.encode_name(input);

        // Update CfC temporal state (maintains conversation trajectory)
        let trajectory_hv = self.cfc_state.step(&input_hv);

        // Compute algebraic delta: what change does this imply?
        // delta = trajectory ⊖ current_state (unbind operation)
        let current_state_hv = codebase
            .codebase_hv()
            .cloned()
            .unwrap_or_else(|| ContinuousHV::zero(self.config.hdc_dim));
        // unbind = bind with inverse
        let delta_hv = trajectory_hv.bind(&current_state_hv.inverse());

        // Find similar past intents in memory
        let similar_intents = self.find_similar_intents(&input_hv, codebase);

        // Grounding confidence: how well does this match known codebase patterns?
        let grounding_confidence = if let Some(cb_hv) = codebase.codebase_hv() {
            input_hv.similarity(cb_hv).abs().min(1.0)
        } else {
            0.0
        };

        IntuitiveIntent {
            intent_hv: input_hv,
            trajectory_hv,
            delta_hv,
            similar_intents,
            grounding_confidence,
        }
    }

    fn find_similar_intents(
        &self,
        input_hv: &ContinuousHV,
        codebase: &CodebaseMemory,
    ) -> Vec<(String, f32)> {
        // Query codebase memory for similar patterns
        let matches = codebase.query(input_hv, 5);
        matches
            .iter()
            .map(|m| (m.name.clone(), m.similarity))
            .collect()
    }

    // ========================================================================
    // Path C: Divergence Computation (Active Inference)
    // ========================================================================

    fn compute_divergence(
        &self,
        symbolic: &SymbolicIntent,
        intuitive: &IntuitiveIntent,
        _codebase: &CodebaseMemory,
    ) -> DivergenceAnalysis {
        // Encode the symbolic plan as an HDC vector
        let symbolic_hv = self.encode_symbolic_intent(symbolic);

        // Compare with intuitive intent
        let action_similarity = symbolic_hv.similarity(&intuitive.intent_hv);
        let trajectory_similarity = symbolic_hv.similarity(&intuitive.trajectory_hv);

        // Free Energy = 1 - agreement
        // We weight trajectory higher because it encodes conversation history
        let agreement = action_similarity * 0.4 + trajectory_similarity * 0.6;
        let free_energy = (1.0 - agreement).clamp(0.0, 1.0);

        // Identify specific divergence sources
        let mut sources = Vec::new();

        // Check action divergence
        for (action, weight) in &symbolic.actions {
            if let Some(proto) = self.action_prototypes.get(action) {
                let sim = intuitive.intent_hv.similarity(proto);
                if sim < 0.3 && *weight > 0.5 {
                    sources.push(DivergenceSource {
                        aspect: format!("action:{:?}", action),
                        symbolic_value: format!("{:?} (weight: {:.2})", action, weight),
                        intuitive_similarity: sim,
                        explanation: format!(
                            "LLM suggests {:?} but context doesn't support this action",
                            action
                        ),
                    });
                }
            }
        }

        // Check target divergence
        if let Some(proto) = self.target_prototypes.get(&symbolic.target.kind) {
            let sim = intuitive.intent_hv.similarity(proto);
            if sim < 0.2 {
                sources.push(DivergenceSource {
                    aspect: "target".to_string(),
                    symbolic_value: format!("{:?}", symbolic.target.kind),
                    intuitive_similarity: sim,
                    explanation: format!(
                        "LLM targets {:?} but conversation context suggests different focus",
                        symbolic.target.kind
                    ),
                });
            }
        }

        // Generate clarification question if needed
        let clarification_question = if free_energy > self.config.confusion_threshold {
            Some(self.generate_clarification(&sources, symbolic, intuitive))
        } else {
            None
        };

        DivergenceAnalysis {
            free_energy,
            divergence_sources: sources,
            clarification_question,
        }
    }

    /// Encode symbolic intent as HDC for comparison
    fn encode_symbolic_intent(&self, symbolic: &SymbolicIntent) -> ContinuousHV {
        let dim = self.config.hdc_dim;
        let mut components = Vec::new();

        // Encode actions (weighted bundle)
        for (action, weight) in &symbolic.actions {
            if let Some(proto) = self.action_prototypes.get(action) {
                // Scale prototype by weight
                let scaled = proto.scale(*weight);
                components.push(scaled);
            }
        }

        // Encode target
        if let Some(proto) = self.target_prototypes.get(&symbolic.target.kind) {
            components.push(proto.clone());
        }

        // Encode target name if present
        if let Some(name) = &symbolic.target.name {
            components.push(self.encoder.encode_name(name));
        }

        // Encode patterns as additional context
        for pattern in &symbolic.patterns {
            components.push(self.encoder.encode_name(pattern));
        }

        if components.is_empty() {
            ContinuousHV::zero(dim)
        } else {
            ContinuousHV::bundle_owned(&components)
        }
    }

    /// Generate a clarification question when divergence is high
    fn generate_clarification(
        &self,
        sources: &[DivergenceSource],
        symbolic: &SymbolicIntent,
        intuitive: &IntuitiveIntent,
    ) -> String {
        if sources.is_empty() {
            return "I'm not sure I understand. Could you clarify what you'd like me to do?"
                .to_string();
        }

        // Use the most significant divergence source
        let primary = &sources[0];

        if primary.aspect.starts_with("action:") {
            // Action mismatch
            let suggested_action = intuitive
                .similar_intents
                .first()
                .map(|(name, _)| name.as_str())
                .unwrap_or("something else");

            format!(
                "You mentioned {:?}, but based on our conversation, \
                 were you perhaps referring to {}? \
                 Could you clarify?",
                symbolic.actions.first().map(|(a, _)| a),
                suggested_action
            )
        } else if primary.aspect == "target" {
            // Target mismatch
            format!(
                "You're asking about {:?}, but I was tracking a different context. \
                 Are we switching focus, or did you mean something in our current scope?",
                symbolic.target.kind
            )
        } else {
            format!(
                "I detected some ambiguity: {}. Could you help me understand better?",
                primary.explanation
            )
        }
    }

    // ========================================================================
    // State Management
    // ========================================================================

    /// Reset the CfC temporal state (start fresh conversation)
    pub fn reset_conversation(&mut self) {
        self.cfc_state.reset();
    }

    /// Get current trajectory (for debugging/visualization)
    pub fn current_trajectory(&self) -> &ContinuousHV {
        self.cfc_state.trajectory()
    }
}

// ============================================================================
// LLM Intent Parser (Path A)
// ============================================================================

/// Parses natural language to structured intent using LLM
pub struct LLMIntentParser {
    // In production, this would hold an LLMOrgan reference
    // For now, we use heuristic parsing
}

impl LLMIntentParser {
    pub fn new() -> Self {
        Self {}
    }

    /// Parse input to structured intent.
    ///
    /// Currently uses keyword-based heuristics as a placeholder.
    /// In production, this would call the LLMOrgan with a structured prompt
    /// (deferred — requires `reasoning_engine` + LLM integration).
    pub fn parse_to_struct(&self, input: &str) -> SymbolicIntent {
        let input_lower = input.to_lowercase();

        // Detect actions (heuristic - replace with LLM in production)
        let mut actions = Vec::new();

        if input_lower.contains("create")
            || input_lower.contains("write")
            || input_lower.contains("implement")
            || input_lower.contains("add")
        {
            actions.push((IntentAction::Create, 0.8));
        }
        if input_lower.contains("modify")
            || input_lower.contains("change")
            || input_lower.contains("update")
            || input_lower.contains("edit")
        {
            actions.push((IntentAction::Modify, 0.7));
        }
        if input_lower.contains("explain")
            || input_lower.contains("what")
            || input_lower.contains("how")
            || input_lower.contains("why")
        {
            actions.push((IntentAction::Explain, 0.8));
        }
        if input_lower.contains("find")
            || input_lower.contains("search")
            || input_lower.contains("where")
            || input_lower.contains("similar")
        {
            actions.push((IntentAction::Find, 0.7));
        }
        if input_lower.contains("refactor")
            || input_lower.contains("clean")
            || input_lower.contains("reorganize")
        {
            actions.push((IntentAction::Refactor, 0.8));
        }
        if input_lower.contains("debug")
            || input_lower.contains("fix")
            || input_lower.contains("error")
            || input_lower.contains("bug")
        {
            actions.push((IntentAction::Debug, 0.8));
        }
        if input_lower.contains("test") {
            actions.push((IntentAction::Test, 0.7));
        }
        if input_lower.contains("delete") || input_lower.contains("remove") {
            actions.push((IntentAction::Delete, 0.6));
        }

        if actions.is_empty() {
            actions.push((IntentAction::Explain, 0.5)); // Default to explain
        }

        // Detect target kind
        let target_kind = if input_lower.contains("function") || input_lower.contains("fn ") {
            TargetKind::Function
        } else if input_lower.contains("struct") || input_lower.contains("class") {
            TargetKind::Struct
        } else if input_lower.contains("module") || input_lower.contains("mod ") {
            TargetKind::Module
        } else if input_lower.contains("file") {
            TargetKind::File
        } else if input_lower.contains("codebase") || input_lower.contains("project") {
            TargetKind::Codebase
        } else {
            TargetKind::Unspecified
        };

        // Detect language
        let language = if input_lower.contains("rust") {
            Some("rust".to_string())
        } else if input_lower.contains("python") {
            Some("python".to_string())
        } else if input_lower.contains("nix") {
            Some("nix".to_string())
        } else {
            None
        };

        // Extract patterns (simple word extraction - LLM would do better)
        let patterns: Vec<String> = input
            .split_whitespace()
            .filter(|w| w.len() > 3 && !Self::is_common_word(w))
            .map(|s| s.to_lowercase())
            .collect();

        SymbolicIntent {
            actions,
            target: IntentTarget {
                kind: target_kind,
                name: None, // LLM would extract specific names
                path: None,
                line_range: None,
            },
            language,
            patterns,
            edge_cases: Vec::new(),
            context_refs: Vec::new(),
            reasoning: None,
        }
    }

    fn is_common_word(word: &str) -> bool {
        matches!(
            word.to_lowercase().as_str(),
            "the"
                | "and"
                | "for"
                | "that"
                | "with"
                | "this"
                | "from"
                | "have"
                | "will"
                | "would"
                | "could"
                | "should"
                | "please"
                | "want"
                | "need"
                | "like"
                | "make"
                | "using"
                | "into"
        )
    }
}

impl Default for LLMIntentParser {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CfC Temporal State (Path B - Time)
// ============================================================================

/// Maintains temporal state using CfC-inspired dynamics
pub struct CfCTemporalState {
    hidden_dim: usize,
    tau: f32,
    /// Current hidden state
    hidden: Vec<f32>,
    /// Accumulated trajectory as HDC
    trajectory: ContinuousHV,
    /// Step counter
    step_count: usize,
}

impl CfCTemporalState {
    pub fn new(hidden_dim: usize, tau: f32) -> Self {
        Self {
            hidden_dim,
            tau,
            hidden: vec![0.0; hidden_dim],
            trajectory: ContinuousHV::zero(4096), // Will be resized on first step
            step_count: 0,
        }
    }

    /// Process one step of input, update state, return trajectory HV
    pub fn step(&mut self, input_hv: &ContinuousHV) -> ContinuousHV {
        let dim = input_hv.values.len();

        // Resize trajectory if needed
        if self.trajectory.values.len() != dim {
            self.trajectory = ContinuousHV::zero(dim);
        }

        // Simple CfC-inspired update:
        // h(t+1) = (1 - 1/τ) * h(t) + (1/τ) * σ(W * input)
        let decay = 1.0 - 1.0 / self.tau;
        let input_weight = 1.0 / self.tau;

        // Project input to hidden (simple: take first hidden_dim values)
        for i in 0..self.hidden_dim.min(dim) {
            let input_val = input_hv.values[i];
            self.hidden[i] = decay * self.hidden[i] + input_weight * input_val.tanh();
        }

        // Project hidden back to HDC space
        let mut trajectory_values = vec![0.0f32; dim];
        for i in 0..dim {
            let h_idx = i % self.hidden_dim;
            trajectory_values[i] = self.hidden[h_idx];
        }

        // Blend with previous trajectory (accumulate history)
        let blend_factor = 0.7;
        for i in 0..dim {
            self.trajectory.values[i] = blend_factor * self.trajectory.values[i]
                + (1.0 - blend_factor) * trajectory_values[i];
        }

        // Normalize
        let norm: f32 = self
            .trajectory
            .values
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        if norm > 1e-6 {
            for v in &mut self.trajectory.values {
                *v /= norm;
            }
        }

        self.step_count += 1;
        self.trajectory.clone()
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.hidden.fill(0.0);
        self.trajectory = ContinuousHV::zero(self.trajectory.values.len());
        self.step_count = 0;
    }

    /// Get current trajectory
    pub fn trajectory(&self) -> &ContinuousHV {
        &self.trajectory
    }

    /// Get step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_encoder() -> CodeHDEncoder {
        CodeHDEncoder::new(512)
    }

    fn test_memory() -> CodebaseMemory {
        CodebaseMemory::new(test_encoder())
    }

    #[test]
    fn test_triune_creation() {
        let integrator = TriuneIntentIntegrator::new(512);
        assert_eq!(integrator.config.hdc_dim, 512);
        assert!(!integrator.action_prototypes.is_empty());
    }

    #[test]
    fn test_llm_parser_create() {
        let parser = LLMIntentParser::new();
        let intent = parser.parse_to_struct("Create a Rust function that sorts a vector");

        assert!(
            intent
                .actions
                .iter()
                .any(|(a, _)| *a == IntentAction::Create)
        );
        assert_eq!(intent.target.kind, TargetKind::Function);
        assert_eq!(intent.language, Some("rust".to_string()));
    }

    #[test]
    fn test_llm_parser_debug() {
        let parser = LLMIntentParser::new();
        let intent = parser.parse_to_struct("Fix the bug in the login function");

        assert!(
            intent
                .actions
                .iter()
                .any(|(a, _)| *a == IntentAction::Debug)
        );
    }

    #[test]
    fn test_cfc_temporal_accumulation() {
        let mut cfc = CfCTemporalState::new(64, 2.0);
        let input1 = ContinuousHV::random(256, 1);
        let input2 = ContinuousHV::random(256, 2);

        let traj1 = cfc.step(&input1);
        let traj2 = cfc.step(&input2);

        // Trajectory should change
        assert!(traj1.similarity(&traj2) < 0.99);

        // But should retain some history (not completely different)
        assert!(traj1.similarity(&traj2) > 0.1);
    }

    #[test]
    fn test_resolve_intent_basic() {
        let mut integrator = TriuneIntentIntegrator::new(512);
        let memory = test_memory();

        let result = integrator.resolve_intent("Write a function to sort numbers", &memory);

        match result {
            IntentResult::Actionable {
                intent, confidence, ..
            } => {
                assert!(
                    intent
                        .actions
                        .iter()
                        .any(|(a, _)| *a == IntentAction::Create)
                );
                assert!(confidence > 0.0);
            }
            IntentResult::NeedsClarification {
                partial_intent,
                divergence,
                ..
            } => {
                // Also acceptable - just means high uncertainty
                assert!(divergence.free_energy > 0.0);
                assert!(!partial_intent.actions.is_empty());
            }
            IntentResult::Incomprehensible { .. } => {
                panic!("Should understand basic intent");
            }
        }
    }

    #[test]
    fn test_conversation_trajectory() {
        let mut integrator = TriuneIntentIntegrator::new(512);
        let memory = test_memory();

        // First message about sorting
        let _ = integrator.resolve_intent("Let's work on the sorting module", &memory);
        let traj1 = integrator.current_trajectory().clone();

        // Second message about sorting
        let _ = integrator.resolve_intent("Add a quicksort function", &memory);
        let traj2 = integrator.current_trajectory().clone();

        // Trajectory should evolve but maintain coherence
        let similarity = traj1.similarity(&traj2);
        assert!(
            similarity > 0.2,
            "Trajectory should maintain some coherence"
        );
        assert!(similarity < 0.95, "Trajectory should evolve");
    }

    #[test]
    fn test_reset_conversation() {
        let mut integrator = TriuneIntentIntegrator::new(512);
        let memory = test_memory();

        let _ = integrator.resolve_intent("Work on the parser", &memory);
        let traj_before = integrator.current_trajectory().clone();

        integrator.reset_conversation();
        let traj_after = integrator.current_trajectory().clone();

        // After reset, trajectory should be zero/neutral
        let norm: f32 = traj_after.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 0.01, "Trajectory should be reset");
    }

    #[test]
    fn test_symbolic_encoding() {
        let integrator = TriuneIntentIntegrator::new(512);

        let intent1 = SymbolicIntent {
            actions: vec![(IntentAction::Create, 1.0)],
            target: IntentTarget {
                kind: TargetKind::Function,
                name: Some("sort".to_string()),
                path: None,
                line_range: None,
            },
            language: Some("rust".to_string()),
            patterns: vec!["quicksort".to_string()],
            edge_cases: vec![],
            context_refs: vec![],
            reasoning: None,
        };

        let intent2 = SymbolicIntent {
            actions: vec![(IntentAction::Debug, 1.0)],
            target: IntentTarget {
                kind: TargetKind::File,
                name: Some("login".to_string()),
                path: None,
                line_range: None,
            },
            language: None,
            patterns: vec![],
            edge_cases: vec![],
            context_refs: vec![],
            reasoning: None,
        };

        let hv1 = integrator.encode_symbolic_intent(&intent1);
        let hv2 = integrator.encode_symbolic_intent(&intent2);

        // Different intents should have different encodings
        let similarity = hv1.similarity(&hv2);
        assert!(
            similarity < 0.8,
            "Different intents should have different encodings"
        );
    }
}
