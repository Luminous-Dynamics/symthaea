//! NixOS Mind Integration Bridges
//!
//! Implements the `HippocampusBridge` and `NixPipelineHook` traits defined in
//! `symthaea-nix`, wiring the NixOS conscious mind into the main Symthaea
//! hippocampus and consciousness pipeline.

use std::sync::{Arc, Mutex};
use symthaea_core::hdc::ContinuousHV;

use symthaea_nix::action::{NixOSCommand, NixOSExecutor, ExecutionResult};
use symthaea_nix::mind::{SystemEpisode, EpisodeOutcome, HdcWorldModel};
use symthaea_nix::mind::causal_graph::NixCausalGraph;
use symthaea_nix::plugin::actor_bridge::HippocampusBridge;
use symthaea_nix::plugin::pipeline_integration::{
    NixPipelineHook, NixPipelineResult, NixPipelineProcessor,
};

use crate::memory::hippocampus::{
    HippocampusActor, EmotionalValence, RecallQuery,
};
use crate::brain::actor_model::{ActorSystem, ActorRole};

// =============================================================================
// HIPPOCAMPUS BRIDGE
// =============================================================================

/// Bridges NixOS episodic memory into the main Symthaea hippocampus.
pub struct NixHippocampusBridge {
    hippocampus: Arc<Mutex<HippocampusActor>>,
}

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
            Ok(handle) => tokio::task::block_in_place(|| {
                handle.block_on(encode_future)
            }),
            Err(_) => {
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| format!("Failed to create runtime: {}", e))?;
                rt.block_on(encode_future)
            }
        }.map_err(|e| e.to_string())?;

        Ok(id)
    }

    fn recall_similar(&self, query: &ContinuousHV, top_k: usize) -> Vec<SystemEpisode> {
        let mut hippo = match self.hippocampus.lock() {
            Ok(h) => h,
            Err(_) => return Vec::new(),
        };

        let recall_query = RecallQuery::from_embedding(query.as_slice().to_vec())
            .with_top_k(top_k);

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
        result.memories.iter().map(|trace| {
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
        }).collect()
    }

    fn consolidate(&self) -> usize {
        let mut hippo = match self.hippocampus.lock() {
            Ok(h) => h,
            Err(_) => return 0,
        };

        let consolidate_future = hippo.consolidate();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| {
                handle.block_on(consolidate_future).unwrap_or(0)
            }),
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
}

impl NixPipelineHookImpl {
    /// Create a new pipeline hook.
    pub fn new(
        processor: NixPipelineProcessor,
        bridge: Arc<NixHippocampusBridge>,
    ) -> Self {
        // Bootstrap causal graph with known NixOS patterns
        let mut causal_graph = NixCausalGraph::new(42);
        for (cause, effect, _) in symthaea_nix::mind::causal_graph::NixOSCausalPatterns::known_patterns() {
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
            ExecutionResult::PendingConfirmation { required_phi, .. } => {
                (false, format!("Needs Φ ≥ {:.2} (current: {:.2})", required_phi, phi))
            }
            ExecutionResult::RolledBack { error, .. } => {
                (false, format!("Rolled back: {}", error))
            }
            ExecutionResult::FailedNoRollback { error, .. } => {
                (false, format!("Failed: {}", error))
            }
            ExecutionResult::Blocked { reason, .. } => {
                (false, format!("Blocked: {}", reason))
            }
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
}

impl NixPipelineHook for NixPipelineHookImpl {
    fn pre_process(&self) -> f64 {
        self.current_phi
    }

    fn process_nix_input(
        &mut self,
        input: &str,
        phi: f64,
        confidence: f64,
    ) -> NixPipelineResult {
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
            let predicted: Vec<String> = self.causal_graph
                .predict_side_effects(last_action)
                .iter()
                .map(|e| e.affected_variable.clone())
                .collect();
            let predicted_refs: Vec<&str> = predicted.iter().map(|s| s.as_str()).collect();

            // Observed effects: the action itself and success/failure
            let mut observed = vec![action];
            if success {
                observed.push("success");
            }
            self.causal_graph.observe_outcome(last_action, &observed, &predicted_refs);
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
}

// =============================================================================
// ACTOR REGISTRATION
// =============================================================================

/// Register NixOS-specific actors in the main Symthaea actor system.
///
/// Spawns 5 actors: observer, inference, planner, executor, memory.
pub fn register_nix_actors(actor_system: &mut ActorSystem) -> Vec<crate::brain::actor_model::ActorId> {
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
            let cmd = NixOSCommand::Search { query: "vim".to_string(), json: false };
            let (success, output) = hook.execute_command(cmd).await;
            assert!(success, "Dry-run search should succeed");
            assert!(output.contains("DRY-RUN"), "Output should indicate dry run: {}", output);
        });
    }

    #[test]
    fn test_end_to_end_pipeline_with_nix_hook() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            use crate::integration::conscious_pipeline::{ConsciousPipeline, PipelineConfig};

            // 1. Create the NixOS hook with hippocampus bridge
            let hippo = Arc::new(Mutex::new(HippocampusActor::new(1000)));
            let bridge = Arc::new(NixHippocampusBridge::new(hippo.clone()));
            let processor = NixPipelineProcessor::new().with_skip_observe(true);
            let hook = NixPipelineHookImpl::new(processor, bridge);

            // 2. Create the ConsciousPipeline with the hook attached
            let config = PipelineConfig::default();
            let mut pipeline = ConsciousPipeline::new(config).await.unwrap()
                .with_nix_hook(hook);

            // 3. Process an input through the full pipeline
            let result = pipeline.process("search for htop").await.unwrap();

            // 4. Verify the result has NixOS enrichment
            assert!(result.nix_enrichment.is_some(), "NixOS enrichment should be present");
            let enrichment = result.nix_enrichment.as_ref().unwrap();
            assert!(!enrichment.plan.goal.description.is_empty());

            // 5. Verify consciousness state
            assert!(result.phi > 0.0, "Φ should be positive");

            // 6. Process a second input to verify state accumulates
            let result2 = pipeline.process("install firefox").await.unwrap();
            assert!(result2.nix_enrichment.is_some());

            // 7. Verify consolidation counter increments
            assert_eq!(pipeline.consolidation_counter(), 2);
        });
    }

    #[test]
    fn test_consolidation_triggers() {
        let hippo = Arc::new(Mutex::new(HippocampusActor::new(1000)));
        let bridge = Arc::new(NixHippocampusBridge::new(hippo));
        let processor = NixPipelineProcessor::new().with_skip_observe(true);
        let mut hook = NixPipelineHookImpl::new(processor, bridge)
            .with_consolidation_interval(3);

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
}
