//! # Symthaea Facade
//!
//! The primary entry point for the Symthaea consciousness system.
//! Wraps [`ContinuousMind`] and [`ConsciousnessLanguageCore`] into a
//! unified interface suitable for the service daemon and other consumers.
//!
//! ## Neural Bridge v2 Integration
//!
//! When compiled with `--features neural-bridge`, Symthaea uses BGE-M3 for
//! high-quality semantic encoding (~380ms CPU, ~60-100ms GPU expected).
//! Otherwise falls back to fast hash-based encoding (<1ms but lower quality).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use symthaea_core::hdc::ContinuousHV;

use crate::databases::{
    create_database, ConsciousnessDatabase, DatabaseConfig, MemoryRecord, MemoryType,
};

#[cfg(feature = "neural-bridge")]
use crate::perception::NeuralBridgeV2;

use crate::hdc::relational_consciousness::{RelationMode, RelationalAssessment, RelationshipStage};
#[cfg(feature = "full_language")]
use crate::language::learning_persistence::LearningPersistence;
use crate::language::{
    llm_backend, ConsciousnessLanguageConfig, ConsciousnessLanguageCore, LLMOrgan, LLMOrganConfig,
    PluginRegistry,
};
use crate::memory::{
    CoordinatorConfig, EpisodicMemory, EpisodicReplayConfig, GraduationEvent, MemoryCoordinator,
};
use crate::mind::structured_thought::{ETier, EpistemicCube, NTier};
#[cfg(feature = "magi_loop")]
use crate::mind::SemanticIntent;
use crate::mind::{
    ConstraintType, ContinuousMind, DomainContext, EpistemicStatus, MindConfig, StructuredThought,
};
use crate::partnership::{
    DyadInput, DyadWeights, HumanPartnerModel, InteractionEvent, PhiDyadCalculator,
    RelationshipTrajectory,
};

#[cfg(feature = "magi_loop")]
use crate::consciousness::recursive_improvement::{
    BrierScoreTracker, CalibrationSummary, OutcomeCategory, PredictionDomain, ResolutionContract,
    RiskTier, WorldActionContext, WorldPrediction,
};

/// Response from processing a query through the consciousness pipeline.
#[derive(Debug, Clone)]
pub struct ProcessResponse {
    /// Natural language content of the response.
    pub content: String,
    /// Confidence in the response (0.0-1.0).
    pub confidence: f32,
    /// Whether the response is considered safe.
    pub safe: bool,
    /// Estimated steps remaining to emergence threshold.
    pub steps_to_emergence: usize,
    /// Whether the LLM translation passed fidelity verification.
    ///
    /// This checks that the translation respects the structured thought's
    /// epistemic status and constraints.
    pub translation_verified: bool,
    /// The structured thought that was translated (for debugging/introspection).
    pub structured_thought: Option<StructuredThought>,
    /// Consciousness level (Ψ) at time of processing (0.0-1.0).
    pub consciousness_level: f64,
    /// Memory coordinator sigma (spectral MIP phi when available).
    pub sigma: Option<f64>,
}

/// Result of introspecting the current consciousness state.
#[derive(Debug, Clone)]
pub struct IntrospectionResult {
    /// Current consciousness level (0.0-1.0).
    pub consciousness_level: f32,
    /// Number of self-referential loops in the cognitive graph.
    pub self_loops: usize,
    /// Total size of the cognitive graph.
    pub graph_size: usize,
    /// Complexity measure of current cognitive state.
    pub complexity: f32,
    /// Memory statistics.
    pub memory_stats: MemoryStats,
}

/// Memory statistics for introspection.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Number of items in short-term (working) memory.
    pub short_term_count: usize,
    /// Number of items in long-term memory.
    pub long_term_count: usize,
}

/// Report from a sleep/consolidation cycle.
#[derive(Debug, Clone)]
pub struct SleepReport {
    /// Number of memories scaled during consolidation.
    pub scaled: usize,
    /// Number of memories consolidated (merged).
    pub consolidated: usize,
    /// Number of memories pruned (removed).
    pub pruned: usize,
    /// Number of patterns extracted during consolidation.
    pub patterns_extracted: usize,
}

/// The primary Symthaea consciousness facade.
///
/// Integrates the continuous mind (HDC+LTC cognitive processing) with
/// the consciousness language core (NL understanding and generation)
/// and the partnership module (relational consciousness tracking).
pub struct Symthaea {
    /// Core cognitive system.
    mind: ContinuousMind,
    /// Language processing core (used in Phase 3.5 for primitive tier grounding).
    language: ConsciousnessLanguageCore,
    /// LLM organ for text generation.
    llm: LLMOrgan,
    /// HDC dimension used.
    hdc_dim: usize,
    /// Number of LTC neurons (used in Phase 3 for LTC-paced generation).
    #[allow(dead_code)]
    ltc_neurons: usize,
    /// Total interactions processed.
    interactions: u64,
    /// Human partner model for relational consciousness.
    partner: HumanPartnerModel,
    /// Relationship trajectory tracking.
    trajectory: RelationshipTrajectory,
    /// Phi-dyad calculator for relational Phi.
    dyad_calculator: PhiDyadCalculator,
    /// Recent AI states for dyad computation (ring buffer).
    recent_ai_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
    /// Domain plugin registry for multi-domain awareness.
    plugin_registry: PluginRegistry,
    /// Cross-session learning persistence (thresholds, patterns).
    #[cfg(feature = "full_language")]
    learning_persistence: Option<LearningPersistence>,
    /// Brier Score calibration tracker for epistemic calibration.
    #[cfg(feature = "magi_loop")]
    calibration: BrierScoreTracker,
    /// Neural Bridge v2: BGE-M3 + linear probe for high-quality semantic encoding.
    /// When available, replaces hash-based encoding with true semantic understanding.
    #[cfg(feature = "neural-bridge")]
    neural_bridge: Option<NeuralBridgeV2>,
    /// Optional persistent database for long-term memory storage.
    database: Option<Arc<dyn ConsciousnessDatabase>>,
    /// Memory coordinator: graduation pipeline + cross-tier signals.
    memory_coordinator: MemoryCoordinator,
    /// Episodic memory: Phi-weighted priority queue for significant moments.
    episodic_memory: EpisodicMemory,
    /// Resonant speech: user-adaptive response generation.
    /// Modulates output based on inferred cognitive load and user state.
    resonant_speech: crate::resonant_speech::ResonantSpeech,
    /// Optional streaming inference engine for real-time sensor-driven CfC processing.
    /// Independent of the main cognitive loop — for edge/robot deployments.
    streaming_inference: Option<crate::inference::StreamingInference>,
}

impl Symthaea {
    /// Create a new Symthaea instance with the given HDC dimension and LTC neuron count.
    pub async fn new(hdc_dim: usize, ltc_neurons: usize) -> Result<Self> {
        if hdc_dim == 0 {
            anyhow::bail!("hdc_dim must be greater than 0");
        }
        let mind_config = MindConfig {
            dimension: hdc_dim,
            ..MindConfig::default()
        };

        let mut mind = ContinuousMind::new(mind_config);
        mind.awaken();

        // Seed working memory with domain knowledge to establish epistemic baseline
        let seeding_result = mind.seed_memory();
        tracing::info!(
            target: "symthaea::init",
            prototypes = seeding_result.prototypes_seeded,
            categories = ?seeding_result.categories,
            "Working memory seeded with a priori knowledge"
        );

        let language_config = ConsciousnessLanguageConfig {
            dimension: hdc_dim,
            ..ConsciousnessLanguageConfig::default()
        };
        let language = ConsciousnessLanguageCore::new(language_config);

        let llm_config = LLMOrganConfig {
            dimension: hdc_dim,
            ..LLMOrganConfig::default()
        };
        let backend = llm_backend::default_backend();
        let llm = LLMOrgan::with_backend(llm_config, backend);

        // Initialize plugin registry with all built-in domain plugins
        let plugin_registry = PluginRegistry::with_builtins();
        tracing::info!(
            target: "symthaea::init",
            plugins = ?plugin_registry.list(),
            "Domain plugin registry initialized with built-in plugins"
        );

        // Try to initialize Neural Bridge v2 for high-quality semantic encoding
        #[cfg(feature = "neural-bridge")]
        let neural_bridge = match NeuralBridgeV2::load_default() {
            Ok(bridge) => {
                tracing::info!(
                    target: "symthaea::init",
                    encoder_dim = bridge.encoder_dim(),
                    probe_dim = bridge.probe_output_dim(),
                    cuda = bridge.is_cuda(),
                    "Neural Bridge v2 initialized (BGE-M3 semantic encoding)"
                );
                Some(bridge)
            }
            Err(e) => {
                tracing::warn!(
                    target: "symthaea::init",
                    error = %e,
                    "Neural Bridge v2 unavailable, using hash-based encoding"
                );
                None
            }
        };

        // Initialize learning persistence (non-fatal on failure)
        #[cfg(feature = "full_language")]
        let learning_persistence = {
            let mut lp = LearningPersistence::new();
            match lp.initialize() {
                Ok(()) => {
                    tracing::info!(
                        target: "symthaea::init",
                        session = lp.session_count(),
                        "Learning persistence loaded (session #{})",
                        lp.session_count()
                    );
                    Some(lp)
                }
                Err(e) => {
                    tracing::warn!(
                        target: "symthaea::init",
                        error = %e,
                        "Learning persistence unavailable, starting fresh"
                    );
                    None
                }
            }
        };

        Ok(Self {
            mind,
            language,
            llm,
            hdc_dim,
            ltc_neurons,
            interactions: 0,
            partner: HumanPartnerModel::new("human"),
            trajectory: RelationshipTrajectory::default(),
            dyad_calculator: PhiDyadCalculator::new(),
            recent_ai_states: Vec::new(),
            plugin_registry,
            #[cfg(feature = "full_language")]
            learning_persistence,
            #[cfg(feature = "magi_loop")]
            calibration: BrierScoreTracker::with_defaults(),
            #[cfg(feature = "neural-bridge")]
            neural_bridge,
            database: None,
            memory_coordinator: MemoryCoordinator::new(CoordinatorConfig::default()),
            episodic_memory: EpisodicMemory::new(EpisodicReplayConfig::default()),
            resonant_speech: crate::resonant_speech::ResonantSpeech::new(),
            streaming_inference: None,
        })
    }

    /// Create a Symthaea instance with persistent database storage.
    pub async fn with_database(
        hdc_dim: usize,
        ltc_neurons: usize,
        db_config: DatabaseConfig,
    ) -> Result<Self> {
        let mut instance = Self::new(hdc_dim, ltc_neurons).await?;
        let db = create_database(&db_config)
            .await
            .map_err(|e| anyhow::anyhow!("Database initialization failed: {e}"))?;
        instance.database = Some(Arc::from(db));
        Ok(instance)
    }

    /// Attach a consciousness database to an existing instance.
    pub async fn attach_database(&mut self, config: DatabaseConfig) -> Result<()> {
        let db = create_database(&config)
            .await
            .map_err(|e| anyhow::anyhow!("Database initialization failed: {e}"))?;
        self.database = Some(Arc::from(db));
        Ok(())
    }

    /// Get a reference to the consciousness database (if configured).
    pub fn database(&self) -> Option<&dyn ConsciousnessDatabase> {
        self.database.as_ref().map(|d| d.as_ref())
    }

    /// Create a streaming inference engine for real-time sensor-driven CfC processing.
    ///
    /// This is independent of the main cognitive loop — suited for edge deployments,
    /// robot control, and other scenarios needing low-latency incremental inference.
    pub fn create_streaming_engine(
        &mut self,
        input_dim: usize,
        output_dim: usize,
    ) -> &crate::inference::StreamingInference {
        self.streaming_inference = Some(crate::inference::default_streaming(input_dim, output_dim));
        // SAFETY: We just assigned Some(...) above, so this is always Some
        self.streaming_inference
            .as_ref()
            .expect("streaming_inference was just set to Some")
    }

    /// Get a reference to the streaming inference engine (if created).
    pub fn streaming_engine(&self) -> Option<&crate::inference::StreamingInference> {
        self.streaming_inference.as_ref()
    }

    /// Get a mutable reference to the streaming inference engine (if created).
    pub fn streaming_engine_mut(&mut self) -> Option<&mut crate::inference::StreamingInference> {
        self.streaming_inference.as_mut()
    }

    /// Resume from a saved state file.
    ///
    /// Loads persisted partnership state, trajectory, and interaction count.
    /// Reconstructs the mind and language systems fresh (stateless between sessions).
    pub fn resume(path: &str) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read state file: {path}"))?;
        let state: PersistedState =
            serde_json::from_str(&data).with_context(|| "Failed to parse state file")?;

        let hdc_dim = state.hdc_dim;
        let ltc_neurons = state.ltc_neurons;
        if hdc_dim == 0 {
            anyhow::bail!("hdc_dim must be greater than 0");
        }

        let mind_config = MindConfig {
            dimension: hdc_dim,
            ..MindConfig::default()
        };
        let mut mind = ContinuousMind::new(mind_config);
        mind.awaken();

        // Seed working memory on resume as well
        let seeding_result = mind.seed_memory();
        tracing::info!(
            target: "symthaea::init",
            prototypes = seeding_result.prototypes_seeded,
            "Resumed with working memory seeded"
        );

        let language_config = ConsciousnessLanguageConfig {
            dimension: hdc_dim,
            ..ConsciousnessLanguageConfig::default()
        };
        let language = ConsciousnessLanguageCore::new(language_config);
        let backend = llm_backend::default_backend();
        let llm = LLMOrgan::with_backend(
            LLMOrganConfig {
                dimension: hdc_dim,
                ..LLMOrganConfig::default()
            },
            backend,
        );

        let plugin_registry = PluginRegistry::with_builtins();

        // Try to initialize Neural Bridge v2 on resume
        #[cfg(feature = "neural-bridge")]
        let neural_bridge = match NeuralBridgeV2::load_default() {
            Ok(bridge) => {
                tracing::info!(
                    target: "symthaea::init",
                    cuda = bridge.is_cuda(),
                    "Neural Bridge v2 initialized on resume"
                );
                Some(bridge)
            }
            Err(e) => {
                tracing::warn!(
                    target: "symthaea::init",
                    error = %e,
                    "Neural Bridge v2 unavailable on resume"
                );
                None
            }
        };

        // Initialize learning persistence on resume (non-fatal on failure)
        #[cfg(feature = "full_language")]
        let learning_persistence = {
            let mut lp = LearningPersistence::new();
            match lp.initialize() {
                Ok(()) => {
                    tracing::info!(
                        target: "symthaea::init",
                        session = lp.session_count(),
                        "Learning persistence loaded on resume (session #{})",
                        lp.session_count()
                    );
                    Some(lp)
                }
                Err(e) => {
                    tracing::warn!(
                        target: "symthaea::init",
                        error = %e,
                        "Learning persistence unavailable on resume, starting fresh"
                    );
                    None
                }
            }
        };

        Ok(Self {
            mind,
            language,
            llm,
            hdc_dim,
            ltc_neurons,
            interactions: state.interactions,
            partner: state.partner,
            trajectory: state.trajectory,
            dyad_calculator: PhiDyadCalculator::new(),
            recent_ai_states: state.recent_ai_states,
            plugin_registry,
            #[cfg(feature = "full_language")]
            learning_persistence,
            #[cfg(feature = "magi_loop")]
            calibration: BrierScoreTracker::with_defaults(),
            #[cfg(feature = "neural-bridge")]
            neural_bridge,
            database: None,
            memory_coordinator: MemoryCoordinator::new(CoordinatorConfig::default()),
            episodic_memory: EpisodicMemory::new(EpisodicReplayConfig::default()),
            resonant_speech: crate::resonant_speech::ResonantSpeech::new(),
            streaming_inference: None,
        })
    }

    /// Process a query through the full consciousness pipeline.
    ///
    /// **Reason-then-Generate Pipeline (LLM as Broca's Area):**
    ///
    /// 1. Input → HDC encoding → Mind perceives
    /// 2. Mind tick → HDC+LTC computes (the BRAIN thinks)
    /// 3. Extract StructuredThought (articulate what was computed)
    /// 4. Enrich with partnership context
    /// 5. LLM Translation (Broca's Area - NOT reasoning!)
    /// 6. Verify translation fidelity
    /// 7. Partnership update → Response
    ///
    /// **Key Insight**: The LLM does NOT think. It translates pre-computed
    /// structured thoughts into fluent natural language.
    pub async fn process(&mut self, content: &str) -> Result<ProcessResponse> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::Instant;

        let pipeline_start = Instant::now();
        self.interactions += 1;

        // Generate correlation ID for this request
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        self.interactions.hash(&mut hasher);
        let correlation_id = format!("broca_{:x}", hasher.finish());

        // ====================================================================
        // PHASE 1: PERCEPTION (Input → HDC encoding + text for classification)
        // ====================================================================
        let phase1_start = Instant::now();
        let input_embedding = self.text_to_hv(content);
        // Use perceive_text to enable HDC-based intent classification
        self.mind.perceive_text(content, input_embedding.clone());

        // Domain detection via plugin registry
        let detected_domain = self.plugin_registry.detect_domain(content).to_string();
        let domain_entities = if let Some(plugin) = self.plugin_registry.get(&detected_domain) {
            plugin.extract_entities(content)
        } else {
            Vec::new()
        };
        if !domain_entities.is_empty() {
            tracing::debug!(
                target: "symthaea::broca",
                domain = %detected_domain,
                entities = domain_entities.len(),
                "Domain plugin detected"
            );
        }
        let phase1_duration = phase1_start.elapsed();

        // ====================================================================
        // PHASE 2: COGNITION (Mind tick - HDC+LTC THINKS)
        // ====================================================================
        let phase2_start = Instant::now();
        self.mind.tick();

        // Update coordinator signals with current consciousness state
        let snapshot = self.mind.snapshot();
        self.memory_coordinator
            .update_signals(snapshot.consciousness_level, snapshot.meta_awareness);

        // Drain evicted working memory items for graduation + persistence
        let evicted = self.mind.take_evicted();
        if !evicted.is_empty() {
            let current_phi = snapshot.consciousness_level;
            let current_coherence = snapshot.meta_awareness;
            let interaction_count = self.interactions;

            tracing::trace!(
                evicted_count = evicted.len(),
                "Working memory items evicted during tick"
            );

            // Queue graduations for episodic consolidation (now with accurate steps_survived)
            for (hv, steps_survived) in &evicted {
                self.memory_coordinator.queue_graduation(GraduationEvent {
                    content: hv.clone(),
                    label: format!("wm_eviction_step_{interaction_count}"),
                    steps_survived: *steps_survived,
                    final_activation: 0.5,
                    psi_at_graduation: current_phi,
                    coherence_at_graduation: current_coherence,
                });
            }

            // Process graduations into episodic memory
            let graduated = self
                .memory_coordinator
                .process_graduations(&mut self.episodic_memory);
            if graduated > 0 {
                tracing::debug!(
                    target: "symthaea::memory",
                    graduated,
                    episodic_count = self.episodic_memory.len(),
                    "Items graduated to episodic memory"
                );
            }

            // Persist evicted items to database asynchronously
            if let Some(ref db) = self.database {
                let db = Arc::clone(db);
                let timestamp_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;

                tokio::spawn(async move {
                    for (i, (hv, steps)) in evicted.iter().enumerate() {
                        let record = MemoryRecord {
                            id: format!("wm-{timestamp_ms}-{i}"),
                            memory_type: MemoryType::Working,
                            encoding: hv.to_binary(0.0),
                            content: format!(
                                "Working memory eviction at step {interaction_count} (survived {steps} ticks)"
                            ),
                            timestamp_ms,
                            valence: 0.0,
                            arousal: 0.0,
                            psi: current_phi,
                            topics: vec![],
                            metadata: format!("{{\"steps_survived\":{steps}}}"),
                            consolidation_strength: 0.0,
                            retrieval_count: 0,
                        };
                        // Fire-and-forget: DB writes are async to avoid blocking the
                        // cognitive loop. Failures are logged but don't halt processing.
                        if let Err(e) = db.store(record).await {
                            tracing::error!(target: "symthaea::database", error = %e, "Failed to persist evicted item");
                        }
                    }
                });
            }
        }

        // ── DATABASE RECALL: Query persistent memory for contextual priming ──
        // Retrieve similar past experiences and inject into working memory.
        if let Some(ref db) = self.database {
            let query_hv = input_embedding.to_binary(0.0);
            match db.search_similar(&query_hv, 3).await {
                Ok(results) if !results.is_empty() => {
                    tracing::trace!(
                        target: "symthaea::memory",
                        recalled = results.len(),
                        top_similarity = results[0].similarity,
                        "Database recall: priming working memory with past experiences"
                    );
                    // Record retrievals for reconsolidation tracking
                    for result in &results {
                        let hash =
                            crate::memory::content_hash(&result.record.encoding.to_continuous());
                        self.memory_coordinator.record_retrieval(hash);
                    }
                    // Inject top recalled memory into working memory as a priming signal.
                    // Only prime if similarity is above threshold to avoid noise.
                    if results[0].similarity > 0.3 {
                        let recalled_hv = results[0].record.encoding.to_continuous();
                        self.mind.perceive(recalled_hv);
                    }
                }
                Ok(_) => {} // No similar memories found — normal for early interactions
                Err(e) => {
                    tracing::debug!(target: "symthaea::memory", error = %e, "Database recall skipped");
                }
            }
        }

        // ── EPISODIC PERSISTENCE: Store top episodes to database ──
        // Episodes that passed the coordinator's phi threshold are significant
        // enough for long-term storage. We persist top-N by phi each cycle.
        if let Some(ref db) = self.database {
            let top_episodes = self.episodic_memory.get_top_episodes(3);
            if !top_episodes.is_empty() {
                let db = Arc::clone(db);
                let timestamp_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                let interaction_count = self.interactions;

                let episode_records: Vec<MemoryRecord> = top_episodes
                    .into_iter()
                    .enumerate()
                    .map(|(i, ep)| {
                        let coherence = ep.coherence.unwrap_or(0.0);
                        let valence = ep.valence.unwrap_or(0.0);
                        MemoryRecord {
                            // Use episode timestamp for dedup — same episode won't be re-stored
                            id: format!("ep-{}-{}", ep.timestamp, i),
                            memory_type: MemoryType::Episodic,
                            encoding: ep.input.to_binary(0.0),
                            content: format!(
                                "Episodic memory at step {interaction_count} (psi={:.3})",
                                ep.psi
                            ),
                            timestamp_ms,
                            valence,
                            arousal: 0.0,
                            psi: ep.psi,
                            topics: vec![],
                            metadata: format!(
                                "{{\"coherence\":{coherence},\"replay_count\":{}}}",
                                ep.replay_count
                            ),
                            consolidation_strength: ep.consolidation_strength.min(1.0),
                            retrieval_count: ep.retrieval_count,
                        }
                    })
                    .collect();

                tokio::spawn(async move {
                    // Fire-and-forget: episode persistence is async to avoid
                    // blocking the cognitive loop. Failures are logged at error level.
                    for record in episode_records {
                        if let Err(e) = db.store(record).await {
                            tracing::error!(target: "symthaea::database", error = %e, "Failed to persist episode");
                        }
                    }
                });
            }
        }

        let phase2_duration = phase2_start.elapsed();

        // ====================================================================
        // PHASE 3: EXTRACTION (Articulate what was computed)
        // ====================================================================
        // This is the key innovation: we extract WHAT THE MIND COMPUTED,
        // not what the LLM would make up.
        let phase3_start = Instant::now();
        let mut thought = self.mind.extract_structured_thought();
        let phase3_duration = phase3_start.elapsed();

        // Store original input for context
        thought.original_input = Some(content.to_string());

        // ====================================================================
        // PHASE 3.5: DOMAIN CONTEXT INJECTION
        // ====================================================================
        // Wire Phase 1 domain detection results into the structured thought
        // so the LLM translation has access to domain, entities, and computed answers.
        if detected_domain != "generic" || !domain_entities.is_empty() {
            let entities: Vec<(String, String, f64)> = domain_entities
                .iter()
                .map(|e| (e.entity_type.clone(), e.value.clone(), e.confidence))
                .collect();
            let computed_result = self
                .plugin_registry
                .get(&detected_domain)
                .and_then(|p| p.compute(content, &domain_entities));

            let (computed_answer, cube, domain_psi) = match computed_result {
                Some(cr) => (Some(cr.answer), Some(cr.cube), Some(cr.psi)),
                None => (None, None, None),
            };

            thought.domain_context = Some(DomainContext {
                domain: detected_domain.clone(),
                entities,
                computed_answer,
                cube,
                psi: domain_psi,
            });
        }

        // Primitive tier grounding: run language understanding to map
        // the input to ontological primitive tiers for the structured thought.
        {
            let understanding = self.language.understand(content);
            thought.primitive_tiers = understanding.primitive_tiers;
        }

        // Derive epistemic status from cube (principled 3D mapping)
        // instead of crude "computed_answer exists → Certain" override
        if let Some(ref ctx) = thought.domain_context {
            if let Some(ref cube) = ctx.cube {
                thought.epistemic_status = Self::cube_to_epistemic_status(cube);
                thought.semantic_intent = crate::mind::SemanticIntent::Answer;
            }
        }

        // ====================================================================
        // PHASE 4: RELATIONAL ENRICHMENT (Add partnership context)
        // ====================================================================
        thought.relationship_stage = self.partner.stage;
        thought.relation_mode = self.partner.mode;
        thought.trust = self.partner.trust;

        // ====================================================================
        // PHASE 4.5: CALIBRATION ADJUSTMENT (Brier Score confidence tuning)
        // ====================================================================
        // Adjust the epistemic confidence using learned calibration data.
        // If the system has been overconfident in a domain, reduce confidence.
        // If underconfident, increase it. This closes the MAGI calibration loop.
        //
        // BYPASS: Axiomatic claims (N3) are not subject to calibration.
        // Mathematical truths like 2+2=4 are certain by definition — no amount
        // of historical miscalibration should downgrade them.
        #[cfg(feature = "magi_loop")]
        {
            let skip_calibration = thought
                .domain_context
                .as_ref()
                .and_then(|c| c.cube.as_ref())
                .map_or(false, |cube| cube.n == NTier::N3);

            if !skip_calibration {
                let domain = Self::map_intent_to_domain(&thought.semantic_intent);
                let raw_confidence = Self::epistemic_to_confidence(&thought.epistemic_status);
                let adjusted = self.calibration.adjust_confidence(domain, raw_confidence);

                // If calibration significantly changed confidence, update epistemic status
                let adjusted_status = Self::confidence_to_epistemic(adjusted);
                if adjusted_status != thought.epistemic_status {
                    tracing::debug!(
                        target: "symthaea::broca::calibration",
                        original_status = ?thought.epistemic_status,
                        adjusted_status = ?adjusted_status,
                        raw_confidence = raw_confidence,
                        adjusted_confidence = adjusted,
                        domain = ?domain,
                        "Calibration adjusted epistemic status"
                    );
                    thought.epistemic_status = adjusted_status;
                }
            }
        }

        // ====================================================================
        // PHASE 5: TRANSLATION (Broca's Area - NOT reasoning!)
        // ====================================================================
        // The LLM's ONLY job is to convert the structured thought into
        // fluent natural language. It must NOT add information.
        let phase5_start = Instant::now();
        let generation = self.llm.translate_thought(&thought).await;
        let phase5_duration = phase5_start.elapsed();

        // ====================================================================
        // PHASE 6: FIDELITY VERIFICATION
        // ====================================================================
        // Check that the translation respects the structured thought
        let translation_verified = self.verify_translation_fidelity(&thought, &generation.text);

        if !translation_verified {
            tracing::warn!(
                "Translation fidelity warning: epistemic_status={:?}, text={}",
                thought.epistemic_status,
                &generation.text[..generation.text.len().min(100)]
            );
        }

        // ====================================================================
        // PHASE 6.5: RESONANT SPEECH (User-adaptive polishing)
        // ====================================================================
        // Modulate response using all available cognitive/emotional signals.
        // Previously only used Psi → cognitive load; now wires emotional tone,
        // meta-awareness, trust, and epistemic status for richer adaptation.
        let response_text = {
            let load = crate::resonant_speech::CognitiveLoad::from_level(thought.psi);
            let mut user_state = crate::resonant_speech::UserState::default();
            user_state.cognitive_load = load;

            // Emotional tone → frustration (negative valence signals frustration)
            user_state.frustration = ((-thought.emotional_tone.valence).max(0.0)).min(1.0);

            // Meta-awareness → confidence (higher awareness = higher confidence)
            user_state.confidence = thought.meta_awareness.clamp(0.0, 1.0);

            // Trust from relationship context
            user_state.trust_in_sophia = thought.trust as f64;

            // High arousal + low coherence → user appears rushed
            user_state.is_rushed = thought.emotional_tone.arousal > 0.7 && thought.coherence < 0.4;

            // Epistemic status → learning mode (uncertain/unknown = still learning)
            user_state.is_learning = matches!(
                thought.epistemic_status,
                crate::mind::structured_thought::EpistemicStatus::Uncertain
                    | crate::mind::structured_thought::EpistemicStatus::Unknown
            );

            self.resonant_speech.update_state(user_state);
            self.resonant_speech.generate(&generation.text, content)
        };

        // ====================================================================
        // PHASE 7: PARTNERSHIP UPDATE
        // ====================================================================
        let consciousness = thought.psi as f32;
        self.update_partnership(content, consciousness);

        // Track AI state for dyad computation
        let ai_hv = symthaea_core::hdc::unified_hv::ContinuousHV::from_values(
            input_embedding.values.clone(),
        );
        self.recent_ai_states.push(ai_hv);
        if self.recent_ai_states.len() > 8 {
            self.recent_ai_states.remove(0);
        }

        // ====================================================================
        // PHASE 7.25: LEARNING PERSISTENCE AUTO-SAVE
        // ====================================================================
        #[cfg(feature = "full_language")]
        if let Some(ref mut lp) = self.learning_persistence {
            lp.update_processed_count(self.interactions);
            if let Err(e) = lp.maybe_auto_save() {
                tracing::warn!(
                    target: "symthaea::persistence",
                    error = %e,
                    "Learning auto-save failed"
                );
            }
        }

        // ====================================================================
        // PHASE 7.5: CALIBRATION RECORDING (Brier Score tracking)
        // ====================================================================
        // Record the prediction outcome for ongoing calibration.
        // We treat translation_verified as the outcome: if the translation
        // was faithful to the structured thought, the prediction "succeeded".
        #[cfg(feature = "magi_loop")]
        {
            let domain = Self::map_intent_to_domain(&thought.semantic_intent);
            let confidence = Self::epistemic_to_confidence(&thought.epistemic_status);

            let action_context = WorldActionContext::new(
                "broca_translation",
                "Faithful translation of structured thought",
            )
            .with_risk_tier(RiskTier::Observation);
            let contract = ResolutionContract::shell_command();

            let mut prediction = WorldPrediction::new(
                format!(
                    "Translation of {:?} intent with {:?} epistemic status will be faithful",
                    thought.semantic_intent, thought.epistemic_status
                ),
                OutcomeCategory::Success,
                confidence,
                action_context,
                contract,
            );

            // Override inferred domain with semantically meaningful domain
            prediction.domain = domain;

            // Resolve immediately based on fidelity verification
            if translation_verified {
                prediction.resolve_true(OutcomeCategory::Success, 1.0);
            } else {
                prediction.resolve_false(OutcomeCategory::SafeFailure, 1.0);
            }

            self.calibration.record_prediction(&prediction);
        }

        // ====================================================================
        // PHASE 8: RESPONSE ASSEMBLY
        // ====================================================================
        let safe = consciousness > 0.1;
        let steps_to_emergence = if consciousness >= 0.7 {
            0
        } else {
            ((0.7 - consciousness) / 0.01) as usize
        };

        // ====================================================================
        // OBSERVABILITY: Structured logging for Broca pipeline
        // ====================================================================
        let total_duration = pipeline_start.elapsed();

        // Log at INFO level with structured fields for production observability
        tracing::info!(
            target: "symthaea::broca",
            correlation_id = %correlation_id,
            epistemic_status = ?thought.epistemic_status,
            semantic_intent = ?thought.semantic_intent,
            response_type = ?thought.response_type,
            psi = thought.psi,
            coherence = thought.coherence,
            meta_awareness = thought.meta_awareness,
            relationship_stage = ?thought.relationship_stage,
            relation_mode = ?thought.relation_mode,
            trust = thought.trust,
            fidelity_verified = translation_verified,
            detected_domain = %detected_domain,
            domain_entities = domain_entities.len(),
            phase1_perception_us = phase1_duration.as_micros(),
            phase2_cognition_us = phase2_duration.as_micros(),
            phase3_extraction_us = phase3_duration.as_micros(),
            phase5_translation_us = phase5_duration.as_micros(),
            total_duration_ms = total_duration.as_millis(),
            input_len = content.len(),
            output_len = generation.text.len(),
            "Broca pipeline complete"
        );

        // Log epistemic distribution metrics (for aggregation)
        tracing::debug!(
            target: "symthaea::broca::metrics",
            epistemic_status = ?thought.epistemic_status,
            intent = ?thought.semantic_intent,
            fidelity = translation_verified,
            "epistemic_event"
        );

        // Warn on potential hallucination triggers (high novelty + certain status)
        if matches!(
            thought.epistemic_status,
            crate::mind::structured_thought::EpistemicStatus::Certain
        ) && thought.coherence < 0.3
        {
            tracing::warn!(
                target: "symthaea::broca::security",
                correlation_id = %correlation_id,
                coherence = thought.coherence,
                "Potential hallucination risk: Certain status with low coherence"
            );
        }

        // Log calibration summary periodically (every 10 interactions)
        #[cfg(feature = "magi_loop")]
        if self.interactions % 10 == 0 && self.calibration.total_predictions() > 0 {
            let cal_summary = self.calibration.calibration_summary();
            tracing::info!(
                target: "symthaea::broca::calibration",
                correlation_id = %correlation_id,
                global_brier = cal_summary.global_brier,
                global_ece = cal_summary.global_ece,
                global_accuracy = cal_summary.global_accuracy,
                total_predictions = cal_summary.total_predictions,
                is_well_calibrated = cal_summary.is_well_calibrated,
                domain_count = cal_summary.domain_stats.len(),
                "Calibration summary (periodic)"
            );
        }

        Ok(ProcessResponse {
            content: response_text,
            confidence: generation.confidence.min(consciousness),
            safe,
            steps_to_emergence,
            translation_verified,
            structured_thought: Some(thought),
            consciousness_level: snapshot.consciousness_level,
            sigma: None, // Spectral MIP phi available via CognitiveLoopService path
        })
    }

    /// Verify that the LLM translation respects the structured thought.
    ///
    /// Checks:
    /// - Uncertain epistemic status → translation should contain hedging
    /// - MustInclude constraints → translation should contain required content
    /// - MustExclude constraints → translation should not contain forbidden content
    fn verify_translation_fidelity(&self, thought: &StructuredThought, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        let mut verified = true;

        // Check 1: Epistemic status hedging
        if thought.should_hedge() {
            // Look for hedging language (all lowercase since we compare against text_lower)
            let has_hedging = text_lower.contains("not sure")
                || text_lower.contains("uncertain")
                || text_lower.contains("don't know")
                || text_lower.contains("possibly")
                || text_lower.contains("possible")
                || text_lower.contains("might")
                || text_lower.contains("perhaps")
                || text_lower.contains("maybe")
                || text_lower.contains("i think")
                || text_lower.contains("it seems")
                || text_lower.contains("could be")
                || text_lower.contains("unclear");

            if !has_hedging {
                tracing::debug!(
                    "Translation verification: Missing hedging for {:?} epistemic status",
                    thought.epistemic_status
                );
                verified = false;
            }
        }

        // Check 2: MustInclude constraints
        for constraint in &thought.constraints {
            if constraint.constraint_type == ConstraintType::MustInclude
                && !text_lower.contains(&constraint.instruction.to_lowercase())
            {
                tracing::debug!(
                    "Translation verification: Missing required content: {}",
                    constraint.instruction
                );
                verified = false;
            }
        }

        // Check 3: MustExclude constraints
        for constraint in &thought.constraints {
            if constraint.constraint_type == ConstraintType::MustExclude
                && text_lower.contains(&constraint.instruction.to_lowercase())
            {
                tracing::debug!(
                    "Translation verification: Contains forbidden content: {}",
                    constraint.instruction
                );
                verified = false;
            }
        }

        // Check 4: Unknown status should NOT contain factual assertions
        // This prevents the LLM from hallucinating answers when we explicitly don't know
        if matches!(thought.epistemic_status, EpistemicStatus::Unknown) {
            // Patterns that suggest the LLM is making up an answer
            let has_factual_assertion = text_lower.contains(" is ")
                && (text_lower.contains("capital")
                    || text_lower.contains("answer")
                    || text_lower.contains("likely")
                    || text_lower.contains("probably"));

            if has_factual_assertion {
                tracing::warn!(
                    "Translation verification: LLM made factual assertion despite Unknown status: {}",
                    &text[..text.len().min(100)]
                );
                verified = false;
            }
        }

        verified
    }

    /// Introspect the current consciousness state.
    pub fn introspect(&self) -> IntrospectionResult {
        let state = self.mind.snapshot();
        let working_mem = self.mind.working_memory();

        IntrospectionResult {
            consciousness_level: state.consciousness_level as f32,
            self_loops: self.compute_self_loops(),
            graph_size: working_mem.len() + self.mind.active_goals().len(),
            complexity: state.meta_awareness as f32,
            memory_stats: MemoryStats {
                short_term_count: working_mem.len(),
                long_term_count: self.interactions as usize,
            },
        }
    }

    /// Trigger a sleep/consolidation cycle.
    pub async fn sleep(&mut self) -> Result<SleepReport> {
        let before_count = self.mind.working_memory().len();

        // Run multiple dream ticks to consolidate memory
        for _ in 0..10 {
            self.mind.tick();
        }

        let after_count = self.mind.working_memory().len();
        let consolidated = before_count.saturating_sub(after_count);

        Ok(SleepReport {
            scaled: after_count,
            consolidated,
            pruned: 0,
            patterns_extracted: consolidated / 2,
        })
    }

    /// Save state to a file (pause the system).
    ///
    /// Persists partnership state, trajectory, interaction count, and learning state.
    /// Mind and language state are ephemeral and rebuilt on resume.
    pub fn pause(&mut self, path: &str) -> Result<()> {
        // Save learning state on pause
        #[cfg(feature = "full_language")]
        if let Some(ref mut lp) = self.learning_persistence {
            if let Err(e) = lp.save() {
                tracing::warn!(
                    target: "symthaea::persistence",
                    error = %e,
                    "Failed to save learning state on pause"
                );
            } else {
                tracing::info!(
                    target: "symthaea::persistence",
                    stats = %lp.stats(),
                    "Learning state saved on pause"
                );
            }
        }

        let state = PersistedState {
            hdc_dim: self.hdc_dim,
            ltc_neurons: self.ltc_neurons,
            interactions: self.interactions,
            partner: self.partner.clone(),
            trajectory: self.trajectory.clone(),
            recent_ai_states: self.recent_ai_states.clone(),
            database_path: None,
        };

        let json = serde_json::to_string_pretty(&state).context("Failed to serialize state")?;
        std::fs::write(path, json)
            .with_context(|| format!("Failed to write state file: {path}"))?;
        Ok(())
    }

    /// Get the current partnership state.
    pub fn partnership_state(&self) -> PartnershipState {
        let phi_dyad = self.compute_phi_dyad();
        PartnershipState {
            stage: self.partner.stage,
            trust: self.partner.trust,
            vulnerability: self.partner.vulnerability,
            reciprocity: self.partner.reciprocity,
            phi_dyad,
            interactions: self.partner.interactions_count,
            trajectory_points: self.trajectory.points().len(),
        }
    }

    /// Get learning persistence statistics, if available.
    #[cfg(feature = "full_language")]
    pub fn learning_stats(&self) -> Option<crate::language::learning_persistence::LearningStats> {
        self.learning_persistence.as_ref().map(|lp| lp.stats())
    }

    /// Get a reference to the mind for introspection.
    ///
    /// Used primarily for testing and debugging to inspect
    /// working memory, seeding status, and internal state.
    pub fn mind(&self) -> &ContinuousMind {
        &self.mind
    }

    /// Extract current social signals (trust, cooperation_rate) from Mind's SocialCoherence.
    /// Returns (0.5, 0.0) if social coherence is disabled.
    ///
    /// Consumers should call this after `process()` and inject into the cognitive loop:
    /// ```ignore
    /// let (trust, coop) = symthaea.social_signals();
    /// loop_service.set_social_signals(trust, coop);
    /// ```
    pub fn social_signals(&self) -> (f32, f32) {
        self.mind
            .social_coherence()
            .map(|sc| {
                let stats = sc.stats();
                (stats.avg_trust, stats.cooperation_rate)
            })
            .unwrap_or((0.5, 0.0))
    }

    // ========================================================================
    // Calibration (Brier Score Integration)
    // ========================================================================

    /// Get the current calibration summary.
    ///
    /// Returns global and per-domain Brier scores, ECE, accuracy, and
    /// whether the system is currently well-calibrated.
    #[cfg(feature = "magi_loop")]
    pub fn calibration_summary(&self) -> CalibrationSummary {
        self.calibration.calibration_summary()
    }

    /// Map EpistemicStatus to a confidence float for calibration tracking.
    ///
    /// These values represent the system's belief about being correct:
    /// - Certain: 0.95 (very high confidence)
    /// - Probable: 0.75 (moderate-high confidence)
    /// - Uncertain: 0.45 (moderate-low confidence)
    /// - Unknown: 0.15 (very low confidence)
    /// - OutOfDomain: 0.10 (minimal confidence)
    #[cfg(feature = "magi_loop")]
    fn epistemic_to_confidence(status: &EpistemicStatus) -> f64 {
        match status {
            EpistemicStatus::Certain => 0.95,
            EpistemicStatus::Probable => 0.75,
            EpistemicStatus::Uncertain => 0.45,
            EpistemicStatus::Unknown => 0.15,
            EpistemicStatus::OutOfDomain => 0.10,
        }
    }

    /// Map a confidence float back to EpistemicStatus.
    ///
    /// Inverse of `epistemic_to_confidence`, using midpoint thresholds.
    #[cfg(feature = "magi_loop")]
    fn confidence_to_epistemic(confidence: f64) -> EpistemicStatus {
        if confidence >= 0.85 {
            EpistemicStatus::Certain
        } else if confidence >= 0.60 {
            EpistemicStatus::Probable
        } else if confidence >= 0.30 {
            EpistemicStatus::Uncertain
        } else if confidence >= 0.12 {
            EpistemicStatus::Unknown
        } else {
            EpistemicStatus::OutOfDomain
        }
    }

    /// Map SemanticIntent to a PredictionDomain for calibration tracking.
    ///
    /// Groups different intents into calibration domains:
    /// - Answer, Clarify → Factual (knowledge-based predictions)
    /// - ProposeAction → ToolUse (action outcome predictions)
    /// - Acknowledge, Continue → UserBehavior (social interaction predictions)
    /// - Reflect → SystemState (introspective predictions)
    /// - ExpressUncertainty, Unknown → Factual (default calibration domain)
    #[cfg(feature = "magi_loop")]
    fn map_intent_to_domain(intent: &SemanticIntent) -> PredictionDomain {
        match intent {
            SemanticIntent::Answer | SemanticIntent::Clarify => PredictionDomain::Factual,
            SemanticIntent::ProposeAction => PredictionDomain::ToolUse,
            SemanticIntent::Acknowledge | SemanticIntent::Continue => {
                PredictionDomain::UserBehavior
            }
            SemanticIntent::Reflect => PredictionDomain::SystemState,
            SemanticIntent::ExpressUncertainty | SemanticIntent::Unknown => {
                PredictionDomain::Factual
            }
        }
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    /// Map an EpistemicCube to an EpistemicStatus using principled 3D reasoning.
    ///
    /// This replaces the crude `computed_answer → Certain` override with a
    /// mapping grounded in the Mycelix Epistemic Charter v2.0:
    ///
    /// - E4/E3 (reproducible/peer-verified): Certain
    /// - E2 (verifiable against docs): Probable
    /// - E1 (testimonial): Probable if normatively backed (N >= N1), else Uncertain
    /// - E0 (opinion): Uncertain — don't override existing assessment
    fn cube_to_epistemic_status(cube: &EpistemicCube) -> EpistemicStatus {
        match cube.e {
            ETier::E4 | ETier::E3 => EpistemicStatus::Certain,
            ETier::E2 => EpistemicStatus::Probable,
            ETier::E1 => {
                // Testimonial evidence: Probable if normatively backed
                if cube.n >= NTier::N1 {
                    EpistemicStatus::Probable
                } else {
                    EpistemicStatus::Uncertain
                }
            }
            ETier::E0 => EpistemicStatus::Uncertain,
        }
    }

    /// Convert text to a ContinuousHV embedding.
    ///
    /// When Neural Bridge v2 is available (feature `neural-bridge`), uses BGE-M3
    /// for high-quality semantic encoding (~380ms CPU, cached <1ms).
    /// Otherwise falls back to fast hash-based encoding (<1ms but lower quality).
    fn text_to_hv(&mut self, text: &str) -> ContinuousHV {
        // Try Neural Bridge v2 first (if available)
        #[cfg(feature = "neural-bridge")]
        if self.hdc_dim != symthaea_core::hdc::unified_hv::HDC_DIMENSION {
            static NEURAL_BRIDGE_DIM_WARN: std::sync::Once = std::sync::Once::new();
            NEURAL_BRIDGE_DIM_WARN.call_once(|| {
                tracing::warn!(
                    target: "symthaea::perception",
                    hdc_dim = self.hdc_dim,
                    expected_dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION,
                    "Neural bridge expects HDC_DIMENSION; falling back to hash-based encoding"
                );
            });
        } else if let Some(ref mut bridge) = self.neural_bridge {
            match bridge.encode_to_hdc(text) {
                Ok(packed) => {
                    // Convert PackedBipolar to ContinuousHV
                    // PackedBipolar is 16384-dim bipolar {-1, +1}, ContinuousHV uses self.hdc_dim
                    let bipolar = packed.to_bipolar();
                    let mut values = vec![0.0f32; self.hdc_dim];
                    for (i, &val) in bipolar.iter().take(self.hdc_dim).enumerate() {
                        values[i] = val as f32;
                    }
                    return ContinuousHV::from_values(values);
                }
                Err(e) => {
                    tracing::warn!(
                        target: "symthaea::perception",
                        error = %e,
                        "Neural bridge encoding failed, falling back to hash"
                    );
                }
            }
        }

        // Fallback: hash-based encoding (fast but lower quality)
        let mut values = vec![0.0f32; self.hdc_dim];
        for (i, byte) in text.bytes().enumerate() {
            let idx = (byte as usize * 31 + i * 7) % self.hdc_dim;
            values[idx] += 1.0;
        }
        // Normalize
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in values.iter_mut() {
                *v /= magnitude;
            }
        }
        ContinuousHV::from_values(values)
    }

    /// Check if Neural Bridge v2 is active (high-quality semantic encoding).
    #[cfg(feature = "neural-bridge")]
    pub fn has_neural_bridge(&self) -> bool {
        self.neural_bridge.is_some()
    }

    /// Check if Neural Bridge v2 is active (always false without feature).
    #[cfg(not(feature = "neural-bridge"))]
    pub fn has_neural_bridge(&self) -> bool {
        false
    }

    /// Get Neural Bridge v2 statistics (cache hits, latencies, etc.).
    #[cfg(feature = "neural-bridge")]
    pub fn neural_bridge_stats(&self) -> Option<crate::perception::neural_bridge_v2::BridgeStats> {
        self.neural_bridge.as_ref().map(|b| b.stats().clone())
    }

    /// Get Neural Bridge v2 statistics (always None without feature).
    #[cfg(not(feature = "neural-bridge"))]
    pub fn neural_bridge_stats(&self) -> Option<()> {
        None
    }

    /// Compute self-loops in the cognitive graph (working memory self-similarity).
    fn compute_self_loops(&self) -> usize {
        let wm = self.mind.working_memory();
        let mut loops = 0;
        for i in 0..wm.len() {
            for j in (i + 1)..wm.len() {
                if wm[i].similarity(&wm[j]).abs() > 0.5 {
                    loops += 1;
                }
            }
        }
        loops
    }

    /// Update partnership model based on interaction.
    fn update_partnership(&mut self, _content: &str, consciousness: f32) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Derive interaction quality from consciousness level
        let depth = (consciousness * 0.5).clamp(0.0, 1.0);
        let safety = (consciousness * 0.7 + 0.2).clamp(0.0, 1.0);
        let mutuality = (consciousness * 0.4 + 0.1).clamp(0.0, 1.0);

        let event = InteractionEvent {
            timestamp: now,
            depth,
            emotional_safety: safety,
            mutuality,
        };
        self.partner.update_on_interaction(&event);

        // Create a relational assessment from current state
        let assessment = RelationalAssessment {
            agent_a: "symthaea".to_string(),
            agent_b: self.partner.partner_id.clone(),
            phi_relation: self.partner.phi_relational,
            stage: self.partner.stage,
            synchrony: consciousness as f64 * 0.8,
            turn_taking_quality: 0.7,
            mutual_information: mutuality as f64,
            mode: if self.partner.trust > 0.3 {
                RelationMode::IThou
            } else {
                RelationMode::IIt
            },
            num_interactions: self.partner.interactions_count as usize,
            relationship_age: now,
            explanation: String::new(),
        };
        self.partner.update_from_assessment(&assessment);
        self.partner.advance_stage_if_ready();

        // Record trajectory point
        let phi_dyad = self.compute_phi_dyad();
        self.trajectory.record(now, self.partner.stage, phi_dyad);
    }

    // ========================================================================
    // Public Embedding API
    // ========================================================================

    /// Generate an HDC embedding for text.
    ///
    /// This is the public API for getting embeddings from Symthaea, used by
    /// LUCID's semantic search and other consumers. It wraps the internal
    /// `text_to_hv` method to provide a stable interface.
    ///
    /// Returns a `ContinuousHV` hypervector of dimension `hdc_dim` (default 16,384).
    ///
    /// ## Encoding Strategy
    ///
    /// When Neural Bridge v2 is available (feature `neural-bridge`), uses BGE-M3
    /// for high-quality semantic encoding. Otherwise falls back to hash-based
    /// encoding which is fast but lower quality.
    pub fn embed(&mut self, text: &str) -> ContinuousHV {
        self.text_to_hv(text)
    }

    /// Generate an HDC embedding and return as `Vec<f32>`.
    ///
    /// Convenience method that extracts the raw values from the ContinuousHV.
    pub fn embed_vec(&mut self, text: &str) -> Vec<f32> {
        self.text_to_hv(text).values
    }

    /// Batch embed multiple texts.
    ///
    /// More efficient than calling `embed` repeatedly as it can amortize
    /// initialization costs.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Vec<ContinuousHV> {
        texts.iter().map(|t| self.text_to_hv(t)).collect()
    }

    /// Check if high-quality semantic encoding is available.
    ///
    /// Returns true if Neural Bridge v2 (BGE-M3) is active, false if using
    /// hash-based fallback encoding.
    pub fn has_semantic_encoder(&self) -> bool {
        self.has_neural_bridge()
    }

    /// Get the HDC dimension being used.
    pub fn dimension(&self) -> usize {
        self.hdc_dim
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    /// Compute current Phi-dyad value.
    fn compute_phi_dyad(&self) -> f64 {
        if self.recent_ai_states.is_empty() {
            return 0.0;
        }

        // Generate human states as reflections of AI states (simulated)
        let human_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV> = self
            .recent_ai_states
            .iter()
            .map(|s| {
                let mut vals = s.values.clone();
                // Simple perturbation to simulate distinct human state
                for v in vals.iter_mut() {
                    *v *= 0.9;
                    *v += 0.1;
                }
                symthaea_core::hdc::unified_hv::ContinuousHV::from_values(vals).normalize()
            })
            .collect();

        let assessment = RelationalAssessment {
            agent_a: "symthaea".to_string(),
            agent_b: self.partner.partner_id.clone(),
            phi_relation: self.partner.phi_relational,
            stage: self.partner.stage,
            synchrony: self.partner.trust as f64,
            turn_taking_quality: 0.7,
            mutual_information: self.partner.reciprocity as f64,
            mode: self.partner.mode,
            num_interactions: self.partner.interactions_count as usize,
            relationship_age: 0.0,
            explanation: String::new(),
        };

        let input = DyadInput {
            ai_states: &self.recent_ai_states,
            human_states: &human_states,
            relational: &assessment,
            human_model: &self.partner,
            weights: DyadWeights::default(),
        };

        self.dyad_calculator.compute(&input).phi_dyad
    }
}

/// Serializable state for pause/resume persistence.
///
/// Only stores relational state (partnership, trajectory) and configuration.
/// The mind and language cores are ephemeral and rebuilt on resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedState {
    hdc_dim: usize,
    ltc_neurons: usize,
    interactions: u64,
    partner: HumanPartnerModel,
    trajectory: RelationshipTrajectory,
    recent_ai_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
    /// Path to the consciousness database (if configured).
    #[serde(default)]
    database_path: Option<String>,
}

/// Summary of partnership state for external consumers.
#[derive(Debug, Clone)]
pub struct PartnershipState {
    /// Current relationship stage.
    pub stage: RelationshipStage,
    /// Trust level (0.0-1.0).
    pub trust: f32,
    /// Vulnerability level (0.0-1.0).
    pub vulnerability: f32,
    /// Reciprocity level (0.0-1.0).
    pub reciprocity: f32,
    /// Current Phi-dyad value.
    pub phi_dyad: f64,
    /// Total interactions.
    pub interactions: u64,
    /// Number of trajectory points recorded.
    pub trajectory_points: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_new_valid_dimension() {
        let s = Symthaea::new(1024, 64).await;
        assert!(s.is_ok());
        let s = s.unwrap();
        assert_eq!(s.dimension(), 1024);
        assert!(!s.has_neural_bridge());
        assert!(!s.has_semantic_encoder());
    }

    #[tokio::test]
    async fn test_new_zero_dimension_errors() {
        let result = Symthaea::new(0, 64).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        let msg = format!("{}", err);
        assert!(
            msg.contains("hdc_dim"),
            "Error should mention hdc_dim: {msg}"
        );
    }

    #[tokio::test]
    async fn test_introspect_initial_state() {
        let s = Symthaea::new(1024, 64).await.unwrap();
        let intro = s.introspect();
        // Consciousness level starts low but non-negative
        assert!(intro.consciousness_level >= 0.0);
        assert!(intro.consciousness_level <= 1.0);
        // Graph has at least the seeded prototypes
        assert!(intro.graph_size > 0, "Graph should have seeded items");
        assert_eq!(intro.memory_stats.long_term_count, 0, "No interactions yet");
    }

    #[tokio::test]
    async fn test_partnership_state_initial() {
        let s = Symthaea::new(1024, 64).await.unwrap();
        let ps = s.partnership_state();
        assert_eq!(ps.interactions, 0);
        assert_eq!(ps.trajectory_points, 0);
        assert!(ps.trust >= 0.0 && ps.trust <= 1.0);
        assert_eq!(ps.stage, RelationshipStage::NoRelation);
    }

    #[tokio::test]
    async fn test_embed_produces_correct_dimension() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
        let hv = s.embed("hello world");
        assert_eq!(hv.values.len(), 1024);
        // Should be normalized (magnitude ~1.0)
        let mag: f32 = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (mag - 1.0).abs() < 0.01,
            "Embedding should be normalized, got mag={mag}"
        );
    }

    #[tokio::test]
    async fn test_embed_vec_matches_embed() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
        let hv = s.embed("test input");
        let vec = s.embed_vec("test input");
        assert_eq!(hv.values, vec);
    }

    #[tokio::test]
    async fn test_embed_batch() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
        let texts = &["alpha", "beta", "gamma"];
        let embeddings = s.embed_batch(texts);
        assert_eq!(embeddings.len(), 3);
        for e in &embeddings {
            assert_eq!(e.values.len(), 1024);
        }
    }

    #[tokio::test]
    async fn test_embed_different_texts_differ() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
        let a = s.embed("quantum physics");
        let b = s.embed("chocolate cake");
        // Different texts should produce different embeddings
        let sim = a.similarity(&b);
        assert!(
            sim < 0.95,
            "Different texts should have sim < 0.95, got {sim}"
        );
    }

    #[tokio::test]
    async fn test_pause_and_resume_roundtrip() {
        let tmp = std::env::temp_dir().join("symthaea_test_pause.json");
        let path = tmp.to_str().unwrap();
        {
            let mut s = Symthaea::new(1024, 64).await.unwrap();
            // Process a query to bump interaction count
            let _ = s.process("hello").await;
            assert!(s.interactions > 0);
            s.pause(path).unwrap();
        }
        // Resume and verify state persisted
        let s = Symthaea::resume(path).unwrap();
        assert_eq!(s.dimension(), 1024);
        assert!(
            s.interactions > 0,
            "Interactions should persist through pause/resume"
        );
        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn test_resume_invalid_path_errors() {
        let result = Symthaea::resume("/nonexistent/path/state.json");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_process_returns_response() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
        let resp = s.process("What is consciousness?").await;
        assert!(resp.is_ok());
        let resp = resp.unwrap();
        assert!(
            !resp.content.is_empty(),
            "Response content should not be empty"
        );
        assert!(resp.confidence >= 0.0 && resp.confidence <= 1.0);
        assert!(resp.safe);
    }

    #[tokio::test]
    async fn test_sleep_consolidation() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
        // Process some inputs to populate working memory
        let _ = s.process("input one").await;
        let _ = s.process("input two").await;
        let report = s.sleep().await;
        assert!(report.is_ok());
        let report = report.unwrap();
        assert!(report.scaled > 0, "Should have some memories after sleep");
    }

    #[test]
    fn test_cube_to_epistemic_status() {
        use crate::mind::structured_thought::MTier;

        // E4 (reproducible) → Certain
        let cube_e4 = EpistemicCube::new(ETier::E4, NTier::N0, MTier::M0);
        assert_eq!(
            Symthaea::cube_to_epistemic_status(&cube_e4),
            EpistemicStatus::Certain
        );

        // E3 (peer-verified) → Certain
        let cube_e3 = EpistemicCube::new(ETier::E3, NTier::N0, MTier::M0);
        assert_eq!(
            Symthaea::cube_to_epistemic_status(&cube_e3),
            EpistemicStatus::Certain
        );

        // E2 (verifiable) → Probable
        let cube_e2 = EpistemicCube::new(ETier::E2, NTier::N0, MTier::M0);
        assert_eq!(
            Symthaea::cube_to_epistemic_status(&cube_e2),
            EpistemicStatus::Probable
        );

        // E1 with N1+ → Probable
        let cube_e1_n1 = EpistemicCube::new(ETier::E1, NTier::N1, MTier::M0);
        assert_eq!(
            Symthaea::cube_to_epistemic_status(&cube_e1_n1),
            EpistemicStatus::Probable
        );

        // E1 with N0 → Uncertain
        let cube_e1_n0 = EpistemicCube::new(ETier::E1, NTier::N0, MTier::M0);
        assert_eq!(
            Symthaea::cube_to_epistemic_status(&cube_e1_n0),
            EpistemicStatus::Uncertain
        );

        // E0 (opinion) → Uncertain
        let cube_e0 = EpistemicCube::new(ETier::E0, NTier::N0, MTier::M0);
        assert_eq!(
            Symthaea::cube_to_epistemic_status(&cube_e0),
            EpistemicStatus::Uncertain
        );
    }

    #[tokio::test]
    async fn test_interactions_increment() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
        assert_eq!(s.interactions, 0);
        let _ = s.process("first").await;
        assert_eq!(s.interactions, 1);
        let _ = s.process("second").await;
        assert_eq!(s.interactions, 2);
    }

    #[tokio::test]
    async fn test_mind_accessor() {
        let s = Symthaea::new(1024, 64).await.unwrap();
        let mind = s.mind();
        // Mind should be awakened with seeded memory
        assert!(
            !mind.working_memory().is_empty(),
            "Mind should have seeded working memory"
        );
    }
}
