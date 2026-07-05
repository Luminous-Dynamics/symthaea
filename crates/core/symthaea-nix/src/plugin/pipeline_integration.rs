// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pipeline Integration — Register NixOS Mind into the Conscious Pipeline
//!
//! Provides the types and traits needed to wire symthaea-nix's active inference
//! engine into the main Symthaea `ConsciousPipeline`. Since symthaea-nix cannot
//! import the main crate, this module defines:
//!
//! - `NixPipelineStage`: stages of NixOS processing for the pipeline
//! - `NixPipelineResult`: domain-specific result that augments `PipelineResult`
//! - `NixPipelineHook`: trait the main crate implements to inject NixOS cognition

use std::path::PathBuf;

use symthaea_core::hdc::ContinuousHV;

use crate::action::executor::{NixOSCommand, SafetyLevel};
use crate::encoding::NixCodebook;
use crate::mind::causal_graph::{NixCausalGraph, NixOSCausalPatterns};
use crate::mind::{ActionCategory, ActionPlan, NixActiveInference};

// =============================================================================
// PIPELINE STAGES
// =============================================================================

/// Stages of NixOS processing within the consciousness pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixPipelineStage {
    /// Observe current system state (sensory input).
    Observe,
    /// Encode observations into HDC space (perception).
    Encode,
    /// Infer user's goal from input + context (cognition).
    InferGoal,
    /// Generate action plan via active inference (decision).
    PlanActions,
    /// Gate action execution based on Φ (consciousness).
    PhiGate,
    /// Execute the selected action (motor output).
    Execute,
    /// Record outcome and learn (consolidation).
    Learn,
}

impl NixPipelineStage {
    /// Human-readable name for display.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Observe => "Observe",
            Self::Encode => "Encode",
            Self::InferGoal => "Infer Goal",
            Self::PlanActions => "Plan Actions",
            Self::PhiGate => "Φ Gate",
            Self::Execute => "Execute",
            Self::Learn => "Learn",
        }
    }

    /// Ordered sequence of all stages.
    pub fn all() -> &'static [NixPipelineStage] {
        &[
            Self::Observe,
            Self::Encode,
            Self::InferGoal,
            Self::PlanActions,
            Self::PhiGate,
            Self::Execute,
            Self::Learn,
        ]
    }
}

// =============================================================================
// PIPELINE RESULT
// =============================================================================

/// NixOS-specific result that augments the main pipeline's `PipelineResult`.
#[derive(Debug, Clone)]
pub struct NixPipelineResult {
    /// The action plan from active inference.
    pub plan: ActionPlan,
    /// The consciousness quadrant at decision time.
    pub quadrant: NixConsciousnessQuadrant,
    /// Current free energy (how far from the goal).
    pub free_energy: f64,
    /// Prediction errors at each hierarchy level.
    pub hierarchy_errors: [f64; 4],
    /// Whether the system was surprised by recent observations.
    pub is_surprised: bool,
    /// Stage at which processing stopped (if early exit).
    pub completed_stage: NixPipelineStage,
    /// The generated NixOS command (if any).
    pub command: Option<NixOSCommand>,
    /// Safety level of the command.
    pub safety_level: SafetyLevel,
    /// Whether Φ gating allowed execution.
    pub phi_allowed: bool,
    /// Working memory items active during this decision.
    pub active_memory: Vec<(String, f64)>,
    /// Knowledge base answer (if a matching article was found).
    pub knowledge_answer: Option<String>,
    /// LLM fallback answer from Ollama (if knowledge base had no match).
    pub ollama_answer: Option<String>,
}

/// Consciousness quadrant for NixOS decision-making.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixConsciousnessQuadrant {
    /// High Φ + High Confidence: deep understanding, execute with trust.
    Confident,
    /// High Φ + Low Confidence: deep but exploring, dry-run first.
    Curious,
    /// Low Φ + High Confidence: pattern-matched routine.
    Habitual,
    /// Low Φ + Low Confidence: genuinely confused, ask for help.
    Confused,
}

impl NixConsciousnessQuadrant {
    /// Determine quadrant from Φ and confidence values.
    pub fn from_metrics(phi: f64, confidence: f64) -> Self {
        match (phi >= 0.5, confidence >= 0.5) {
            (true, true) => Self::Confident,
            (true, false) => Self::Curious,
            (false, true) => Self::Habitual,
            (false, false) => Self::Confused,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Confident => "Confident",
            Self::Curious => "Curious",
            Self::Habitual => "Habitual",
            Self::Confused => "Confused",
        }
    }

    /// Whether this quadrant allows action execution.
    pub fn allows_execution(&self) -> bool {
        matches!(self, Self::Confident | Self::Habitual)
    }

    /// Whether this quadrant suggests asking a clarifying question.
    pub fn suggests_clarification(&self) -> bool {
        matches!(self, Self::Curious | Self::Confused)
    }
}

// =============================================================================
// PIPELINE HOOK TRAIT
// =============================================================================

/// Trait that the main Symthaea crate implements to wire NixOS cognition
/// into the `ConsciousPipeline`.
pub trait NixPipelineHook: Send + Sync {
    /// Called before NixOS processing begins. Returns current system Φ.
    fn pre_process(&self) -> f64;

    /// Run the full NixOS cognition pipeline on user input.
    fn process_nix_input(&mut self, input: &str, phi: f64, confidence: f64) -> NixPipelineResult;

    /// Called after action execution to record the outcome.
    fn post_execute(&mut self, action: &str, success: bool, output: &str);

    /// Provide user feedback for adaptive threshold learning.
    fn feedback(&mut self, was_positive: bool);

    /// Return the causal surprise from the last `post_execute` call.
    ///
    /// 0.0 = outcome was expected by the causal model, 1.0 = completely unexpected.
    fn causal_surprise(&self) -> f32 {
        0.0
    }

    /// Return the number of learned causal edges.
    fn causal_edge_count(&self) -> usize {
        0
    }
}

// =============================================================================
// STANDALONE PIPELINE PROCESSOR
// =============================================================================

/// Standalone NixOS pipeline processor that can be used without the main
/// Symthaea pipeline. Used by the `nix-mind` CLI and TUI.
pub struct NixPipelineProcessor {
    engine: NixActiveInference,
    codebook: NixCodebook,
    phi_threshold: f64,
    skip_observe: bool,
    /// Knowledge base for answering NixOS questions from curated articles.
    knowledge_base: Option<crate::support::knowledge::KnowledgeBase>,
    /// Ollama bridge for LLM fallback when HDC + knowledge base can't answer.
    ollama_bridge: Option<crate::mind::OllamaBridge>,
    /// Causal graph for learning package/service dependency relationships.
    causal_graph: NixCausalGraph,
    /// Counter for learn() calls, used to trigger periodic discover_structure().
    learn_cycle_count: u64,
    /// Number of new edges found in the last discover_structure() call.
    last_discovery_edge_count: usize,
    /// Path for persisting causal graph checkpoints across restarts.
    causal_checkpoint_path: Option<PathBuf>,
}

impl NixPipelineProcessor {
    /// Create a new standalone processor.
    pub fn new() -> Self {
        let mut codebook = NixCodebook::new();
        let knowledge_base = Some(crate::support::knowledge::KnowledgeBase::new(&mut codebook));
        let mut causal_graph = NixCausalGraph::new(42);
        // Bootstrap with known NixOS causal patterns
        for (cause, effect, _) in NixOSCausalPatterns::known_patterns() {
            causal_graph.add_structural_edge(cause, effect, 0.5);
        }
        Self {
            engine: NixActiveInference::new(),
            codebook,
            phi_threshold: 0.3,
            skip_observe: false,
            knowledge_base,
            ollama_bridge: None,
            causal_graph,
            learn_cycle_count: 0,
            last_discovery_edge_count: 0,
            causal_checkpoint_path: None,
        }
    }

    /// Set the minimum Φ for action execution.
    pub fn with_phi_threshold(mut self, threshold: f64) -> Self {
        self.phi_threshold = threshold;
        self
    }

    /// Skip live system observation (useful for testing or offline use).
    pub fn with_skip_observe(mut self, skip: bool) -> Self {
        self.skip_observe = skip;
        self
    }

    /// Enable Ollama LLM fallback for queries the knowledge base can't answer.
    pub fn with_ollama(mut self) -> Self {
        self.ollama_bridge = Some(crate::mind::OllamaBridge::default());
        self
    }

    /// Enable Ollama with custom configuration.
    pub fn with_ollama_config(mut self, config: crate::mind::OllamaBridgeConfig) -> Self {
        self.ollama_bridge = Some(crate::mind::OllamaBridge::new(config));
        self
    }

    /// Disable the knowledge base (for testing).
    pub fn without_knowledge_base(mut self) -> Self {
        self.knowledge_base = None;
        self
    }

    /// Set a checkpoint path for persisting causal graph state.
    ///
    /// If the file already exists, edges and observations are loaded immediately.
    /// After each `discover_structure()` call (every 50 learn cycles), the graph
    /// is auto-saved to this path.
    pub fn set_checkpoint_path(&mut self, path: PathBuf) {
        if path.exists() {
            match self.causal_graph.load(&path) {
                Ok(n) => {
                    tracing::info!(loaded_edges = n, path = %path.display(), "Loaded causal graph checkpoint");
                }
                Err(e) => {
                    tracing::warn!(error = %e, path = %path.display(), "Failed to load causal graph checkpoint");
                }
            }
        }
        self.causal_checkpoint_path = Some(path);
    }

    /// Builder variant of `set_checkpoint_path`.
    pub fn with_checkpoint_path(mut self, path: PathBuf) -> Self {
        self.set_checkpoint_path(path);
        self
    }

    /// How many learn() calls have occurred.
    pub fn learn_cycle_count(&self) -> u64 {
        self.learn_cycle_count
    }

    /// Number of new edges found in the last `discover_structure()` invocation.
    pub fn last_discovery_edge_count(&self) -> usize {
        self.last_discovery_edge_count
    }

    /// Process user input through the full NixOS cognition pipeline.
    #[tracing::instrument(skip(self), fields(phi = %phi, confidence = %confidence))]
    pub fn process(&mut self, input: &str, phi: f64, confidence: f64) -> NixPipelineResult {
        // Stage 1: Observe (skip if configured, e.g. in tests)
        if !self.skip_observe
            && let Ok(snapshot) = crate::observe::SystemObserver::snapshot()
        {
            let mut encoder = crate::encoding::SystemStateEncoder::new(&mut self.codebook);
            let state_hv = encoder.encode_snapshot(&snapshot);
            self.engine.observe_state(state_hv);
        }

        // Stage 2+3: Infer goal + plan actions
        let plan = self.engine.process_input(input);

        // Stage 4: Determine consciousness quadrant
        let quadrant = NixConsciousnessQuadrant::from_metrics(phi, confidence);

        // Get hierarchy state
        let wm = self.engine.world_model();
        let hierarchy = wm.prediction_hierarchy();

        // Determine safety level from best action
        let safety_level = if plan.needs_clarification {
            SafetyLevel::ReadOnly
        } else {
            match plan.actions.first().map(|a| &a.action) {
                Some(ActionCategory::Rebuild) | Some(ActionCategory::Update) => {
                    SafetyLevel::SystemModify
                }
                Some(ActionCategory::Rollback) => SafetyLevel::SystemCritical,
                Some(ActionCategory::GarbageCollect) => SafetyLevel::Destructive,
                Some(ActionCategory::Install) | Some(ActionCategory::Remove) => {
                    SafetyLevel::UserModify
                }
                Some(ActionCategory::Enable) | Some(ActionCategory::Disable) => {
                    SafetyLevel::SystemModify
                }
                Some(ActionCategory::Configure) => SafetyLevel::UserModify,
                _ => SafetyLevel::ReadOnly,
            }
        };

        // Stage 5: Φ gate
        let phi_allowed = phi >= self.phi_threshold && quadrant.allows_execution();

        // Build working memory snapshot
        let active_memory = self
            .engine
            .goal_inference()
            .working_memory()
            .items()
            .iter()
            .map(|item| (item.label.clone(), item.activation))
            .collect();

        // Stage 6: Knowledge base lookup — search curated articles for an answer
        // Encode the query first, then search — avoids simultaneous borrows.
        let knowledge_answer = if plan.needs_clarification || plan.actions.is_empty() {
            let query_hv =
                crate::support::knowledge::KnowledgeBase::encode_text(&mut self.codebook, input);
            self.knowledge_base.as_ref().and_then(|kb| {
                let matches = kb.search_by_hv(&query_hv, 1);
                matches
                    .first()
                    .filter(|m| m.similarity > 0.3)
                    .map(|m| m.article.solution.to_string())
            })
        } else {
            None
        };

        // Stage 7: Ollama LLM fallback — only if knowledge base had no answer
        let ollama_answer = if knowledge_answer.is_none()
            && (plan.needs_clarification || plan.actions.is_empty())
        {
            self.ollama_bridge
                .as_mut()
                .and_then(|bridge| bridge.query_nixos(input, None))
                .map(|resp| resp.text)
        } else {
            None
        };

        NixPipelineResult {
            plan,
            quadrant,
            free_energy: wm.free_energy(),
            hierarchy_errors: hierarchy.errors(),
            is_surprised: hierarchy.is_surprised(),
            completed_stage: if phi_allowed {
                NixPipelineStage::PhiGate
            } else {
                NixPipelineStage::PlanActions
            },
            command: None,
            safety_level,
            phi_allowed,
            active_memory,
            knowledge_answer,
            ollama_answer,
        }
    }

    /// Record outcome and learn from it.
    ///
    /// Updates three learning systems:
    /// 1. Active inference engine — transition model + episodic memory
    /// 2. Causal graph — Hebbian edge updates from observed action→effect pairs
    pub fn learn(&mut self, action: &str, success: bool, state_after: ContinuousHV) {
        let outcome = if success {
            crate::mind::EpisodeOutcome::Success
        } else {
            crate::mind::EpisodeOutcome::Failure("execution failed".into())
        };
        let state_before = self.engine.world_model().system_state().clone();
        let action_cat = ActionCategory::from_command(action);

        // 1. Learn in the active inference engine (transition model + episodic memory)
        self.engine
            .learn_from_outcome(&state_before, action_cat, &state_after, outcome, 0.5);

        // 2. Update the causal graph: the action caused the observed effects.
        //    We model the action command as the cause node, and the success/failure
        //    outcome as observed effects. This accumulates causal knowledge about
        //    which actions lead to which outcomes over many cycles.
        let action_key = format!("action:{}", action);
        let outcome_key = if success {
            format!("outcome:success:{}", action)
        } else {
            format!("outcome:failure:{}", action)
        };
        let predicted_effects = [
            format!("outcome:success:{}", action),
            format!("outcome:failure:{}", action),
        ];
        let predicted_refs: Vec<&str> = predicted_effects.iter().map(|s| s.as_str()).collect();
        let observed_refs: Vec<&str> = vec![outcome_key.as_str()];
        self.causal_graph
            .observe_outcome(&action_key, &observed_refs, &predicted_refs);

        // Also record a numeric observation for the action (1.0 = success, 0.0 = failure)
        // so discover_structure() can find correlations after enough data.
        self.causal_graph
            .observe(&action_key, if success { 1.0 } else { 0.0 });

        // 3. Periodically run causal discovery to find new structural edges.
        //    Every 50 learn cycles, discover_structure() examines accumulated
        //    numeric observations for variable pairs with 20+ data points.
        self.learn_cycle_count += 1;
        if self.learn_cycle_count.is_multiple_of(50) {
            let edges_before = self.causal_graph.edge_count();
            let new_edges = self.causal_graph.discover_structure();
            self.last_discovery_edge_count = new_edges.len();
            let edges_after = self.causal_graph.edge_count();
            tracing::info!(
                cycle = self.learn_cycle_count,
                discovered = new_edges.len(),
                edges_before,
                edges_after,
                net_new = edges_after.saturating_sub(edges_before),
                "Causal structure discovery completed"
            );

            // Auto-save checkpoint after discovery if path is configured
            if let Some(ref path) = self.causal_checkpoint_path {
                if let Err(e) = self.causal_graph.save(path) {
                    tracing::warn!(error = %e, "Failed to save causal graph checkpoint");
                } else {
                    tracing::debug!(path = %path.display(), "Saved causal graph checkpoint");
                }
            }
        }
    }

    /// Access the causal graph (read-only).
    pub fn causal_graph(&self) -> &NixCausalGraph {
        &self.causal_graph
    }

    /// Access the causal graph (mutable).
    pub fn causal_graph_mut(&mut self) -> &mut NixCausalGraph {
        &mut self.causal_graph
    }

    /// Access the underlying engine.
    pub fn engine(&self) -> &NixActiveInference {
        &self.engine
    }

    /// Mutable access to the underlying engine.
    pub fn engine_mut(&mut self) -> &mut NixActiveInference {
        &mut self.engine
    }
}

impl Default for NixPipelineProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_stages() {
        assert_eq!(NixPipelineStage::all().len(), 7);
        assert_eq!(NixPipelineStage::Observe.name(), "Observe");
        assert_eq!(NixPipelineStage::PhiGate.name(), "Φ Gate");
    }

    #[test]
    fn test_consciousness_quadrant() {
        assert_eq!(
            NixConsciousnessQuadrant::from_metrics(0.8, 0.9),
            NixConsciousnessQuadrant::Confident,
        );
        assert_eq!(
            NixConsciousnessQuadrant::from_metrics(0.8, 0.2),
            NixConsciousnessQuadrant::Curious,
        );
        assert_eq!(
            NixConsciousnessQuadrant::from_metrics(0.2, 0.8),
            NixConsciousnessQuadrant::Habitual,
        );
        assert_eq!(
            NixConsciousnessQuadrant::from_metrics(0.2, 0.2),
            NixConsciousnessQuadrant::Confused,
        );
    }

    #[test]
    fn test_quadrant_execution_rules() {
        assert!(NixConsciousnessQuadrant::Confident.allows_execution());
        assert!(NixConsciousnessQuadrant::Habitual.allows_execution());
        assert!(!NixConsciousnessQuadrant::Curious.allows_execution());
        assert!(!NixConsciousnessQuadrant::Confused.allows_execution());
    }

    #[test]
    fn test_quadrant_clarification_rules() {
        assert!(!NixConsciousnessQuadrant::Confident.suggests_clarification());
        assert!(!NixConsciousnessQuadrant::Habitual.suggests_clarification());
        assert!(NixConsciousnessQuadrant::Curious.suggests_clarification());
        assert!(NixConsciousnessQuadrant::Confused.suggests_clarification());
    }

    #[test]
    fn test_standalone_processor() {
        let mut proc = NixPipelineProcessor::new().with_skip_observe(true);
        let result = proc.process("install firefox", 0.7, 0.8);

        assert_eq!(result.quadrant, NixConsciousnessQuadrant::Confident);
        assert!(result.phi_allowed);
        assert!(!result.plan.goal.description.is_empty());
    }

    #[test]
    fn test_low_phi_blocks_execution() {
        let mut proc = NixPipelineProcessor::new()
            .with_phi_threshold(0.5)
            .with_skip_observe(true);
        let result = proc.process("install firefox", 0.2, 0.8);

        assert_eq!(result.quadrant, NixConsciousnessQuadrant::Habitual);
        assert!(!result.phi_allowed);
    }

    #[test]
    fn test_curious_blocks_execution() {
        let mut proc = NixPipelineProcessor::new().with_skip_observe(true);
        let result = proc.process("make it faster", 0.7, 0.3);

        assert_eq!(result.quadrant, NixConsciousnessQuadrant::Curious);
        assert!(!result.phi_allowed);
        assert!(result.quadrant.suggests_clarification());
    }

    #[test]
    fn test_learn_from_outcome() {
        let mut proc = NixPipelineProcessor::new();
        let dim = proc.engine().world_model().system_state().dim();
        let state_after = ContinuousHV::random(dim, 1);
        proc.learn("install firefox", true, state_after);
        assert_eq!(proc.engine().episode_count(), 1);
    }

    #[test]
    fn test_causal_graph_bootstrapped() {
        let proc = NixPipelineProcessor::new();
        // The causal graph should be bootstrapped with known NixOS patterns
        assert!(
            proc.causal_graph().edge_count() > 100,
            "Causal graph should have >100 structural edges from NixOS patterns, got {}",
            proc.causal_graph().edge_count(),
        );
    }

    #[test]
    fn test_learn_updates_causal_graph() {
        let mut proc = NixPipelineProcessor::new();
        let initial_edges = proc.causal_graph().edge_count();

        let dim = proc.engine().world_model().system_state().dim();

        // Learn from multiple outcomes — causal graph should accumulate edges
        for i in 0..5 {
            let state_after = ContinuousHV::random(dim, i + 100);
            proc.learn("install nginx", i % 2 == 0, state_after);
        }

        assert!(
            proc.causal_graph().edge_count() > initial_edges,
            "Causal graph should have new edges after learning (before={}, after={})",
            initial_edges,
            proc.causal_graph().edge_count(),
        );

        // The action→outcome edge should exist
        let conf = proc
            .causal_graph()
            .edge_confidence("action:install nginx", "outcome:success:install nginx");
        assert!(
            conf.is_some(),
            "Should have a causal edge from install action to success outcome"
        );
    }

    #[test]
    fn test_causal_graph_learns_relationships_over_cycles() {
        let mut proc = NixPipelineProcessor::new();
        let dim = proc.engine().world_model().system_state().dim();

        // Simulate 20 cycles of learning
        for i in 0..20 {
            let state_after = ContinuousHV::random(dim, i * 10);
            // Rebuilds mostly succeed
            proc.learn("rebuild switch", true, state_after.clone());
            // GC operations always succeed
            proc.learn("nix-collect-garbage", true, state_after);
        }

        // Verify edges have been strengthened
        let rebuild_conf = proc
            .causal_graph()
            .edge_confidence("action:rebuild switch", "outcome:success:rebuild switch")
            .unwrap_or(0.0);
        assert!(
            rebuild_conf > 0.5,
            "Rebuild→success edge should be strengthened after 20 cycles (got {:.3})",
            rebuild_conf,
        );

        let gc_conf = proc
            .causal_graph()
            .edge_confidence(
                "action:nix-collect-garbage",
                "outcome:success:nix-collect-garbage",
            )
            .unwrap_or(0.0);
        assert!(
            gc_conf > 0.5,
            "GC→success edge should be strengthened after 20 cycles (got {:.3})",
            gc_conf,
        );
    }

    #[test]
    fn test_discover_structure_triggered_at_cadence_50() {
        let mut proc = NixPipelineProcessor::new();
        let dim = proc.engine().world_model().system_state().dim();

        // Feed two correlated actions so discover_structure() has data.
        // Each learn() call observes one numeric value per action key.
        for i in 0..50 {
            let state_after = ContinuousHV::random(dim, i + 200);
            if i % 2 == 0 {
                proc.learn("rebuild switch", true, state_after);
            } else {
                proc.learn("nix-collect-garbage", true, state_after);
            }
        }

        assert_eq!(proc.learn_cycle_count(), 50);
        let discovered = proc.last_discovery_edge_count();
        assert!(
            discovered > 0,
            "discover_structure() should find edges between correlated action keys (got {})",
            discovered,
        );
    }

    #[test]
    fn test_discover_structure_not_triggered_before_50() {
        let mut proc = NixPipelineProcessor::new();
        let dim = proc.engine().world_model().system_state().dim();

        for i in 0..49 {
            let state_after = ContinuousHV::random(dim, i + 300);
            proc.learn("rebuild switch", true, state_after);
        }

        assert_eq!(proc.learn_cycle_count(), 49);
        assert_eq!(proc.last_discovery_edge_count(), 0);
    }

    #[test]
    fn test_causal_graph_save_load_roundtrip() {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("A", "B", 0.9);
        graph.add_structural_edge("B", "C", 0.7);
        graph.observe("A", 1.0);
        graph.observe("A", 2.0);

        let dir = std::env::temp_dir().join("symthaea_nix_test_roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("causal_graph.json");

        graph.save(&path).expect("save should succeed");

        let mut loaded = NixCausalGraph::new(99);
        let n = loaded.load(&path).expect("load should succeed");
        assert_eq!(n, 2, "Should load 2 edges");
        assert_eq!(loaded.edge_count(), 2);

        let ab = loaded.edge_confidence("A", "B");
        assert!(ab.is_some(), "A->B edge should exist after load");
        assert!((ab.unwrap() - 0.9).abs() < 1e-6);

        let bc = loaded.edge_confidence("B", "C");
        assert!(bc.is_some(), "B->C edge should exist after load");
        assert!((bc.unwrap() - 0.7).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_survives_restart() {
        let dir = std::env::temp_dir().join("symthaea_nix_test_checkpoint");
        std::fs::create_dir_all(&dir).unwrap();
        let checkpoint_path = dir.join("causal_checkpoint.json");
        let _ = std::fs::remove_file(&checkpoint_path);

        let edges_after_learning;

        {
            let mut proc =
                NixPipelineProcessor::new().with_checkpoint_path(checkpoint_path.clone());
            let dim = proc.engine().world_model().system_state().dim();

            for i in 0..50 {
                let state_after = ContinuousHV::random(dim, i + 400);
                if i % 2 == 0 {
                    proc.learn("rebuild switch", true, state_after);
                } else {
                    proc.learn("nix-collect-garbage", true, state_after);
                }
            }

            edges_after_learning = proc.causal_graph().edge_count();
            assert!(
                checkpoint_path.exists(),
                "Checkpoint file should exist after 50 learn cycles"
            );
        }

        {
            let proc = NixPipelineProcessor::new().with_checkpoint_path(checkpoint_path.clone());
            let edges_after_reload = proc.causal_graph().edge_count();
            assert!(
                edges_after_reload >= edges_after_learning,
                "Reloaded processor should have >= edges (original={}, reloaded={})",
                edges_after_learning,
                edges_after_reload,
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_e2e_anomaly_process_learn_knowledge_cycle() {
        // End-to-end: detect anomaly → process → learn → verify causal graph + episode
        let mut proc = NixPipelineProcessor::new().with_skip_observe(true);

        // Step 1: Process an anomaly-like input (service failure)
        let result = proc.process("restart nginx after crash", 0.8, 0.7);
        assert!(
            result.phi_allowed,
            "Confident quadrant with high phi should allow execution"
        );
        assert!(!result.plan.goal.description.is_empty());

        // Step 2: Simulate successful outcome and learn
        let dim = proc.engine().world_model().system_state().dim();
        let state_after = ContinuousHV::random(dim, 999);
        let initial_episodes = proc.engine().episode_count();
        let initial_edges = proc.causal_graph().edge_count();

        proc.learn("restart nginx after crash", true, state_after);

        // Step 3: Verify episodic memory grew
        assert_eq!(
            proc.engine().episode_count(),
            initial_episodes + 1,
            "Episodic memory should have one new entry"
        );

        // Step 4: Verify causal graph learned the action→outcome relationship
        assert!(
            proc.causal_graph().edge_count() >= initial_edges,
            "Causal graph should not lose edges after learning"
        );
        let conf = proc.causal_graph().edge_confidence(
            "action:restart nginx after crash",
            "outcome:success:restart nginx after crash",
        );
        assert!(
            conf.is_some(),
            "Should have causal edge from restart action to success outcome"
        );
    }

    #[test]
    fn test_e2e_mixed_outcomes_both_edges_exist() {
        // Multiple learn cycles with mixed success/failure → both edges should exist
        let mut proc = NixPipelineProcessor::new();
        let dim = proc.engine().world_model().system_state().dim();

        // 10 successes + 5 failures for same action
        for i in 0..15 {
            let state_after = ContinuousHV::random(dim, i + 500);
            let success = i < 10;
            proc.learn("upgrade system", success, state_after);
        }

        // Both success and failure edges should exist with non-zero confidence
        let success_conf = proc
            .causal_graph()
            .edge_confidence("action:upgrade system", "outcome:success:upgrade system")
            .unwrap_or(0.0);
        let failure_conf = proc
            .causal_graph()
            .edge_confidence("action:upgrade system", "outcome:failure:upgrade system")
            .unwrap_or(0.0);

        assert!(
            success_conf > 0.0,
            "Success edge should exist after 10 successes (got {:.3})",
            success_conf,
        );
        assert!(
            failure_conf > 0.0,
            "Failure edge should exist after 5 failures (got {:.3})",
            failure_conf,
        );
        assert!(success_conf <= 1.0);
        assert!(failure_conf <= 1.0);
    }
}
