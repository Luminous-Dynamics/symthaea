// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS Mind Integration Bridges
//!
//! Implements the `HippocampusBridge` and `NixPipelineHook` traits defined in
//! `symthaea-nix`, wiring the NixOS conscious mind into the main Symthaea
//! hippocampus and consciousness pipeline.

use std::sync::{Arc, Mutex};
use symthaea_core::hdc::ContinuousHV;

use symthaea_nix::action::{ExecutionResult, NixOSCommand, NixOSExecutor};
use symthaea_nix::mind::causal_graph::NixCausalGraph;
use symthaea_nix::mind::{EpisodeOutcome, HdcWorldModel, SystemEpisode};
use symthaea_nix::plugin::actor_bridge::HippocampusBridge;
use symthaea_nix::plugin::pipeline_integration::{
    NixPipelineHook, NixPipelineProcessor, NixPipelineResult,
};

use crate::brain::actor_model::{ActorRole, ActorSystem};
#[allow(deprecated)]
use crate::memory::hippocampus::{EmotionalValence, HippocampusActor, RecallQuery};

// =============================================================================
// HIPPOCAMPUS BRIDGE
// =============================================================================

/// Bridges NixOS episodic memory into the main Symthaea hippocampus.
#[allow(deprecated)]
pub struct NixHippocampusBridge {
    hippocampus: Arc<Mutex<HippocampusActor>>,
}

#[allow(deprecated)]
impl NixHippocampusBridge {
    /// Create a new bridge wrapping a shared hippocampus.
    pub fn new(hippocampus: Arc<Mutex<HippocampusActor>>) -> Self {
        Self { hippocampus }
    }
}

impl HippocampusBridge for NixHippocampusBridge {
    fn store_episode(&self, episode: &SystemEpisode) -> Result<u64, String> {
        let mut hippo = self.hippocampus.lock().map_err(|e| e.to_string())?;

        let content = format!(
            "NixOS: {} → {}",
            episode.action,
            match &episode.outcome {
                EpisodeOutcome::Success => "success".to_string(),
                EpisodeOutcome::Failure(r) => format!("failure: {}", r),
                EpisodeOutcome::PartialSuccess(r) => format!("partial: {}", r),
                EpisodeOutcome::RolledBack(r) => format!("rollback: {}", r),
            }
        );

        let embedding = episode.state_after.as_slice().to_vec();
        let valence = EmotionalValence::from_f64(episode.emotional_valence);

        // Use block_in_place to allow blocking inside a multi-threaded runtime,
        // or fall back to creating a new runtime if no runtime is active.
        let encode_future = hippo.encode(content, embedding, valence);
        let id = match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(encode_future)),
            Err(_) => {
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| format!("Failed to create runtime: {}", e))?;
                rt.block_on(encode_future)
            }
        }
        .map_err(|e| e.to_string())?;

        Ok(id)
    }

    fn recall_similar(&self, query: &ContinuousHV, top_k: usize) -> Vec<SystemEpisode> {
        let mut hippo = match self.hippocampus.lock() {
            Ok(h) => h,
            Err(_) => return Vec::new(),
        };

        let recall_query = RecallQuery::from_embedding(query.as_slice().to_vec()).with_top_k(top_k);

        let recall_future = hippo.recall(recall_query);
        let result = match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(recall_future)),
            Err(_) => {
                let rt = match tokio::runtime::Runtime::new() {
                    Ok(rt) => rt,
                    Err(_) => return Vec::new(),
                };
                rt.block_on(recall_future)
            }
        };
        let result = match result {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        // Convert MemoryTrace back to SystemEpisode (best-effort reconstruction)
        let dim = query.dim();
        result
            .memories
            .iter()
            .map(|trace| {
                let state_hv = if trace.embedding.len() == dim {
                    ContinuousHV::from_slice(&trace.embedding)
                } else {
                    ContinuousHV::zero(dim)
                };

                SystemEpisode {
                    state_before: ContinuousHV::zero(dim),
                    action: trace.content.clone(),
                    state_after: state_hv,
                    outcome: EpisodeOutcome::Success,
                    phi_at_encoding: trace.phi_at_encoding.unwrap_or(0.0),
                    prediction_error: 0.0,
                    emotional_valence: trace.valence.to_f64(),
                    timestamp: trace.timestamp as i64,
                }
            })
            .collect()
    }

    fn consolidate(&self) -> usize {
        let mut hippo = match self.hippocampus.lock() {
            Ok(h) => h,
            Err(_) => return 0,
        };

        let consolidate_future = hippo.consolidate();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                tokio::task::block_in_place(|| handle.block_on(consolidate_future).unwrap_or(0))
            }
            Err(_) => {
                let rt = match tokio::runtime::Runtime::new() {
                    Ok(rt) => rt,
                    Err(_) => return 0,
                };
                rt.block_on(consolidate_future).unwrap_or(0)
            }
        }
    }
}

// =============================================================================
// PIPELINE HOOK
// =============================================================================

/// Implements `NixPipelineHook` to wire NixOS cognition into the conscious pipeline.
///
/// Receives Φ from the main `ConsciousPipeline` via `process_nix_input()`, ensuring
/// the NixOS active inference engine operates with the real IIT-computed consciousness
/// level rather than an independent estimate.
pub struct NixPipelineHookImpl {
    processor: NixPipelineProcessor,
    bridge: Arc<NixHippocampusBridge>,
    executor: NixOSExecutor,
    /// Φ synced from the main consciousness pipeline (via process_nix_input)
    current_phi: f64,
    positive_count: u32,
    negative_count: u32,
    /// Interaction count since last consolidation
    interactions_since_consolidation: u64,
    /// How often to trigger hippocampal consolidation (0 = disabled)
    consolidation_interval: u64,
    /// Causal graph with Hebbian learning
    causal_graph: NixCausalGraph,
    /// HDC world model for state tracking and drift detection
    world_model: HdcWorldModel,
    /// Last action processed (for Hebbian edge tracking)
    last_action: Option<String>,
    /// Causal surprise from the last `post_execute` call.
    /// 0.0 = outcome matched causal predictions, 1.0 = completely unexpected.
    last_causal_surprise: f32,
}

impl NixPipelineHookImpl {
    /// Create a new pipeline hook.
    pub fn new(processor: NixPipelineProcessor, bridge: Arc<NixHippocampusBridge>) -> Self {
        // Bootstrap causal graph with known NixOS patterns
        let mut causal_graph = NixCausalGraph::new(42);
        for (cause, effect, _) in
            symthaea_nix::mind::causal_graph::NixOSCausalPatterns::known_patterns()
        {
            causal_graph.add_structural_edge(cause, effect, 0.8);
        }

        Self {
            processor,
            bridge,
            executor: NixOSExecutor::new().with_dry_run(true), // Safe default
            current_phi: 0.5,
            positive_count: 0,
            negative_count: 0,
            interactions_since_consolidation: 0,
            consolidation_interval: 25,
            causal_graph,
            world_model: HdcWorldModel::default(),
            last_action: None,
            last_causal_surprise: 0.0,
        }
    }

    /// Enable real execution (not dry-run). Use with caution.
    pub fn with_live_execution(mut self) -> Self {
        self.executor = NixOSExecutor::new();
        self
    }

    /// Execute a NixOS command through the Φ-gated executor.
    ///
    /// Returns the execution result and whether it succeeded.
    pub async fn execute_command(&mut self, command: NixOSCommand) -> (bool, String) {
        let phi = self.current_phi as f32;
        let result = self.executor.execute(command, phi).await;
        match &result {
            ExecutionResult::Success { stdout, .. } => (true, stdout.clone()),
            ExecutionResult::PendingConfirmation { required_phi, .. } => (
                false,
                format!("Needs Φ ≥ {:.2} (current: {:.2})", required_phi, phi),
            ),
            ExecutionResult::RolledBack { error, .. } => (false, format!("Rolled back: {}", error)),
            ExecutionResult::FailedNoRollback { error, .. } => {
                (false, format!("Failed: {}", error))
            }
            ExecutionResult::Blocked { reason, .. } => (false, format!("Blocked: {}", reason)),
        }
    }

    /// Set the consolidation interval.
    pub fn with_consolidation_interval(mut self, interval: u64) -> Self {
        self.consolidation_interval = interval;
        self
    }

    /// Get the number of interactions since last consolidation.
    pub fn interactions_since_consolidation(&self) -> u64 {
        self.interactions_since_consolidation
    }

    /// Access the causal graph (for inspection or persistence).
    pub fn causal_graph(&self) -> &NixCausalGraph {
        &self.causal_graph
    }

    /// Access the HDC world model.
    pub fn world_model(&self) -> &HdcWorldModel {
        &self.world_model
    }

    /// Return the causal surprise from the last `post_execute` call.
    ///
    /// 0.0 = outcome was expected by the causal model, 1.0 = completely unexpected.
    pub fn causal_surprise(&self) -> f32 {
        self.last_causal_surprise
    }
}

impl NixPipelineHook for NixPipelineHookImpl {
    fn pre_process(&self) -> f64 {
        self.current_phi
    }

    fn process_nix_input(&mut self, input: &str, phi: f64, confidence: f64) -> NixPipelineResult {
        // Sync Φ from the main pipeline's IIT computation
        self.current_phi = phi;
        self.processor.process(input, phi, confidence)
    }

    fn post_execute(&mut self, action: &str, success: bool, _output: &str) {
        let dim = self.processor.engine().world_model().system_state().dim();
        let state_after = self.processor.engine().world_model().system_state().clone();
        let free_energy = self.processor.engine().free_energy();

        // Learn from the outcome
        self.processor.learn(action, success, state_after.clone());

        // Update HDC world model with new state observation
        self.world_model.observe(&state_after);

        // Hebbian causal learning: predict side effects, compare with observations
        if let Some(ref last_action) = self.last_action {
            let predictions = self.causal_graph.predict_side_effects(last_action);
            let predicted: Vec<String> = predictions
                .iter()
                .map(|e| e.affected_variable.clone())
                .collect();
            let predicted_refs: Vec<&str> = predicted.iter().map(|s| s.as_str()).collect();

            // Observed effects: the action itself and success/failure
            let mut observed = vec![action];
            if success {
                observed.push("success");
            }

            // Compute causal surprise: how much did predictions diverge from reality?
            if predictions.is_empty() {
                // No predictions — moderately surprising (model has no opinion)
                self.last_causal_surprise = 0.3;
            } else {
                let high_conf_predictions: Vec<&str> = predictions
                    .iter()
                    .filter(|p| p.confidence > 0.5)
                    .map(|p| p.affected_variable.as_str())
                    .collect();

                if high_conf_predictions.is_empty() {
                    // Only low-confidence predictions — low surprise
                    self.last_causal_surprise = 0.1;
                } else {
                    let hits = high_conf_predictions
                        .iter()
                        .filter(|p| observed.contains(p))
                        .count();
                    let miss_ratio = 1.0 - (hits as f32 / high_conf_predictions.len() as f32);
                    // Factor in whether success/failure was expected
                    let outcome_predicted = predicted_refs.contains(&"success") == success;
                    let outcome_surprise = if outcome_predicted { 0.0f32 } else { 0.5 };
                    self.last_causal_surprise =
                        (miss_ratio * 0.5 + outcome_surprise).clamp(0.0, 1.0);
                }
            }

            self.causal_graph
                .observe_outcome(last_action, &observed, &predicted_refs);
        } else {
            // First action ever — no prior causal context, no surprise
            self.last_causal_surprise = 0.0;
        }
        self.last_action = Some(action.to_string());

        // Store as hippocampal episode with real free energy as prediction error
        let episode = SystemEpisode {
            state_before: ContinuousHV::zero(dim),
            action: action.to_string(),
            state_after,
            outcome: if success {
                EpisodeOutcome::Success
            } else {
                EpisodeOutcome::Failure("execution failed".into())
            },
            phi_at_encoding: self.current_phi,
            prediction_error: free_energy,
            emotional_valence: if success { 0.8 } else { -0.8 },
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64,
        };
        let _ = self.bridge.store_episode(&episode);

        // Periodic consolidation
        self.interactions_since_consolidation += 1;
        if self.consolidation_interval > 0
            && self.interactions_since_consolidation >= self.consolidation_interval
        {
            let consolidated = self.bridge.consolidate();
            if consolidated > 0 {
                self.interactions_since_consolidation = 0;
            }
        }
    }

    fn feedback(&mut self, was_positive: bool) {
        if was_positive {
            self.positive_count += 1;
        } else {
            self.negative_count += 1;
        }
        // Adaptive threshold: shift phi based on feedback ratio
        let total = self.positive_count + self.negative_count;
        if total >= 5 {
            let ratio = self.positive_count as f64 / total as f64;
            // If mostly positive, lower threshold (allow more); if mostly negative, raise it
            self.current_phi = 0.3 + (1.0 - ratio) * 0.4;
        }
    }

    fn causal_surprise(&self) -> f32 {
        self.last_causal_surprise
    }

    fn causal_edge_count(&self) -> usize {
        self.causal_graph.edge_count()
    }
}

// =============================================================================
// ACTOR REGISTRATION
// =============================================================================

/// Register NixOS-specific actors in the main Symthaea actor system.
///
/// Spawns 5 actors: observer, inference, planner, executor, memory.
pub fn register_nix_actors(
    actor_system: &mut ActorSystem,
) -> Vec<crate::brain::actor_model::ActorId> {
    let actors = [
        ("nix.observer", ActorRole::Sensor),
        ("nix.inference", ActorRole::Processor),
        ("nix.planner", ActorRole::Decider),
        ("nix.executor", ActorRole::Effector),
        ("nix.memory", ActorRole::Memory),
    ];

    let mut ids = Vec::new();
    for (id, role) in &actors {
        if let Some(actor_id) = actor_system.spawn(*id, role.clone()) {
            ids.push(actor_id);
        }
    }

    // Wire connections: observer → inference → planner → executor → memory
    for i in 0..ids.len().saturating_sub(1) {
        actor_system.connect(&ids[i], &ids[i + 1]);
    }
    // Memory feeds back to inference
    if ids.len() >= 5 {
        actor_system.connect(&ids[4], &ids[1]);
    }

    ids
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::actor_model::ActorSystemConfig;

    #[test]
    fn test_register_nix_actors() {
        let config = ActorSystemConfig::default();
        let mut system = ActorSystem::new(config);
        let ids = register_nix_actors(&mut system);
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn test_hippocampus_bridge_store_recall() {
        let hippo = Arc::new(Mutex::new(HippocampusActor::new(1000)));
        let bridge = NixHippocampusBridge::new(hippo);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dim = 128;
            let episode = SystemEpisode {
                state_before: ContinuousHV::zero(dim),
                action: "install firefox".to_string(),
                state_after: ContinuousHV::random(dim, 42),
                outcome: EpisodeOutcome::Success,
                phi_at_encoding: 0.7,
                prediction_error: 0.1,
                emotional_valence: 0.8,
                timestamp: 1000,
            };

            let id = bridge.store_episode(&episode).unwrap();
            assert!(id > 0);

            let results = bridge.recall_similar(&episode.state_after, 5);
            assert!(!results.is_empty());
        });
    }

    #[test]
    fn test_pipeline_hook_feedback() {
        let hippo = Arc::new(Mutex::new(HippocampusActor::new(1000)));
        let bridge = Arc::new(NixHippocampusBridge::new(hippo));
        let processor = NixPipelineProcessor::new().with_skip_observe(true);
        let mut hook = NixPipelineHookImpl::new(processor, bridge);

        assert!((hook.pre_process() - 0.5).abs() < f64::EPSILON);

        // Provide feedback
        for _ in 0..5 {
            hook.feedback(true);
        }
        // After positive feedback, threshold should be lower
        assert!(hook.current_phi < 0.5);
    }

    #[test]
    fn test_pipeline_hook_process() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let hippo = Arc::new(Mutex::new(HippocampusActor::new(1000)));
            let bridge = Arc::new(NixHippocampusBridge::new(hippo));
            let processor = NixPipelineProcessor::new().with_skip_observe(true);
            let mut hook = NixPipelineHookImpl::new(processor, bridge);

            let result = hook.process_nix_input("install firefox", 0.7, 0.8);
            assert!(!result.plan.goal.description.is_empty());
            assert!(result.phi_allowed);
        });
    }

    #[test]
    fn test_executor_dry_run() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let hippo = Arc::new(Mutex::new(HippocampusActor::new(1000)));
            let bridge = Arc::new(NixHippocampusBridge::new(hippo));
            let processor = NixPipelineProcessor::new().with_skip_observe(true);
            let mut hook = NixPipelineHookImpl::new(processor, bridge);

            // Dry-run search should succeed (ReadOnly safety level, low Φ required)
            let cmd = NixOSCommand::Search {
                query: "vim".to_string(),
                json: false,
            };
            let (success, output) = hook.execute_command(cmd).await;
            assert!(success, "Dry-run search should succeed");
            assert!(
                output.contains("DRY-RUN"),
                "Output should indicate dry run: {}",
                output
            );
        });
    }

    #[test]
    fn test_e2e_process_with_knowledge_base() {
        // Exercise the full cognition chain: input → HDC → active inference → knowledge lookup
        let hippo = Arc::new(Mutex::new(HippocampusActor::new(1000)));
        let bridge = Arc::new(NixHippocampusBridge::new(hippo));
        let processor = NixPipelineProcessor::new().with_skip_observe(true);
        let mut hook = NixPipelineHookImpl::new(processor, bridge);

        // Process a query that the knowledge base should be able to answer
        let result = hook.process_nix_input("nix store gc", 0.7, 0.8);
        assert!(result.phi_allowed);
        assert!(result.quadrant.allows_execution());
        // The plan should have at least one action
        assert!(
            !result.plan.actions.is_empty() || result.plan.needs_clarification,
            "Should produce actions or ask for clarification"
        );
    }

    #[test]
    fn test_e2e_post_execute_causal_learning() {
        // Verify the full causal learning chain: execute → learn → hippocampus
        let hippo = Arc::new(Mutex::new(HippocampusActor::new(1000)));
        let bridge = Arc::new(NixHippocampusBridge::new(hippo));
        let processor = NixPipelineProcessor::new().with_skip_observe(true);
        let mut hook = NixPipelineHookImpl::new(processor, bridge);

        // Process input first to set up state
        let _result = hook.process_nix_input("install firefox", 0.6, 0.7);

        // Record execution outcome — should trigger causal learning + hippocampus store
        hook.post_execute("nix-env -iA firefox", true, "installed firefox-120.0");

        // Causal graph should now have learned from this interaction
        assert!(hook.causal_graph().edge_count() > 0);
        // World model should have been updated
        assert!(hook.world_model().observation_count() > 0);
    }

    #[test]
    fn test_consolidation_triggers() {
        let hippo = Arc::new(Mutex::new(HippocampusActor::new(1000)));
        let bridge = Arc::new(NixHippocampusBridge::new(hippo));
        let processor = NixPipelineProcessor::new().with_skip_observe(true);
        let mut hook = NixPipelineHookImpl::new(processor, bridge).with_consolidation_interval(3);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Simulate 3 post-execute calls to trigger consolidation
            for i in 0..3 {
                hook.post_execute(&format!("action_{}", i), true, "ok");
            }
            // After 3 interactions, consolidation should have been triggered and counter reset
            assert_eq!(hook.interactions_since_consolidation(), 0);
        });
    }

    #[test]
    fn test_causal_surprise_zero_for_first_action() {
        // The very first post_execute has no prior action context, so surprise = 0.0
        let hippo = Arc::new(Mutex::new(HippocampusActor::new(1000)));
        let bridge = Arc::new(NixHippocampusBridge::new(hippo));
        let processor = NixPipelineProcessor::new().with_skip_observe(true);
        let mut hook = NixPipelineHookImpl::new(processor, bridge);

        assert_eq!(
            hook.causal_surprise(),
            0.0,
            "Initial surprise should be 0.0"
        );
        hook.post_execute("nix-env -iA firefox", true, "installed");
        assert_eq!(
            hook.causal_surprise(),
            0.0,
            "First action has no prior causal context, surprise should be 0.0"
        );
    }

    #[test]
    fn test_causal_surprise_nonzero_for_unexpected_outcome() {
        // When the causal model has predictions but the outcome doesn't match,
        // surprise should be > 0.0
        let hippo = Arc::new(Mutex::new(HippocampusActor::new(1000)));
        let bridge = Arc::new(NixHippocampusBridge::new(hippo));
        let processor = NixPipelineProcessor::new().with_skip_observe(true);
        let mut hook = NixPipelineHookImpl::new(processor, bridge);

        // Add a strong structural edge so the causal model has expectations
        hook.causal_graph
            .add_structural_edge("install_firefox", "success", 0.9);

        // First action sets up last_action
        hook.post_execute("install_firefox", true, "installed");
        // Second action: the causal model for "install_firefox" predicted "success"
        // but now we fail an unrelated action — surprise should be nonzero
        hook.post_execute("unrelated_action", false, "failed");
        assert!(
            hook.causal_surprise() > 0.0,
            "Surprise should be > 0.0 when outcome diverges from causal predictions, got {}",
            hook.causal_surprise()
        );
    }

    #[test]
    fn test_causal_edge_count_accessor() {
        let hippo = Arc::new(Mutex::new(HippocampusActor::new(1000)));
        let bridge = Arc::new(NixHippocampusBridge::new(hippo));
        let processor = NixPipelineProcessor::new().with_skip_observe(true);
        let hook = NixPipelineHookImpl::new(processor, bridge);

        // Should have edges from the bootstrapped known NixOS patterns
        assert!(
            hook.causal_edge_count() > 0,
            "Should have bootstrapped causal edges"
        );
        // Verify the trait accessor matches the direct graph accessor
        assert_eq!(
            NixPipelineHook::causal_edge_count(&hook),
            hook.causal_graph().edge_count()
        );
    }
}
