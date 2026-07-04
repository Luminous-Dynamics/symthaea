// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

#[cfg(feature = "code_generation")]
mod code_gen;
#[cfg(feature = "magi_loop")]
mod magi;
mod relational;
#[cfg(feature = "school_learning")]
mod school;

// ── Re-exports ────────────────────────────────────────────────────────────

pub use relational::PartnershipState;
#[cfg(feature = "school_learning")]
pub use school::{CurriculumObjectiveSummary, CurriculumReport};

// ── Imports ───────────────────────────────────────────────────────────────

use anyhow::{Context, Result};
#[cfg(feature = "school_learning")]
use school::{
    CurriculumPersistenceConfig, CurriculumRecallConfig, CurriculumRecallScores,
    load_curriculum_from_store,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(all(feature = "web_research_module", feature = "school_learning"))]
use std::time::{Duration, Instant};
use symthaea_core::hdc::ContinuousHV;

use crate::databases::{
    ConsciousnessDatabase, DatabaseConfig, MemoryRecord, MemoryType, create_database,
};

#[cfg(feature = "neural-bridge")]
use crate::perception::NeuralBridgeV2;

use crate::hdc::relational_consciousness::{RelationMode, RelationalAssessment, RelationshipStage};
#[cfg(feature = "full_language")]
use crate::language::learning_persistence::LearningPersistence;
use crate::language::{
    ConsciousnessLanguageConfig, ConsciousnessLanguageCore, LLMOrgan, LLMOrganConfig,
    PluginRegistry, llm_backend,
};
use crate::memory::{
    CoordinatorConfig, EpisodicMemory, EpisodicReplayConfig, GraduationEvent, MemoryCoordinator,
};
#[cfg(feature = "magi_loop")]
use crate::mind::SemanticIntent;
use crate::mind::structured_thought::{ETier, EpistemicCube, NTier};
use crate::mind::{
    ConstraintType, ContinuousMind, DomainContext, EpistemicStatus, MindConfig, StructuredThought,
};
use crate::partnership::{
    DyadInput, DyadWeights, HumanPartnerModel, InteractionEvent, PhiDyadCalculator,
    RelationshipTrajectory,
};

use crate::action::SimpleExecutor;
pub use crate::action::bindings::ActionRegistry;
use crate::consciousness::interoception::InteroceptionTag;
use crate::infrastructure::{PainSender, SomaticErrorBridge, TaskSupervisor};

#[cfg(feature = "school_learning")]
use crate::school::curriculum::Curriculum;
#[cfg(all(feature = "web_research_module", feature = "school_learning"))]
use crate::school::curriculum_extender::CurriculumExtender;
#[cfg(all(feature = "web_research_module", feature = "school_learning"))]
use crate::school::curriculum_extender::ResearchSummary;
#[cfg(feature = "school_learning")]
use crate::school::curriculum_loader::{CurriculumLoader, CurriculumMeta, LoadError};
#[cfg(feature = "school_learning")]
use crate::school::polymath_drive::run_polymath_collisions;
#[cfg(feature = "ssm-power")]
use crate::ssm::power::PowerSsmSensor;
#[cfg(feature = "web_research_module")]
use crate::web_research::WebResearcher;

// Seam B (2026-07-04): ethics-gate the product path. Same minimal construction
// CognitiveLoopService uses in production (value_evaluator/harmonies_integrator: None) —
// see cognitive_loop/constructor.rs.
use crate::cognitive_loop::ethics_engine::{EthicalVerdict, EthicsEngine, EthicsEngineInput};
use crate::hdc::moral_algebra::MoralAlgebra;
use crate::hdc::moral_parser::MoralParser;

#[cfg(feature = "magi_loop")]
use crate::consciousness::recursive_improvement::{
    BrierScoreTracker, CalibrationSummary, OutcomeCategory, PredictionDomain, ResolutionContract,
    RiskTier, WorldActionContext, WorldPrediction,
};

use relational::{PersistedState, RelationalCore};

// ── Public types ──────────────────────────────────────────────────────────

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
    /// Consciousness level (Psi) at time of processing (0.0-1.0).
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

// ── Feature-gated private types ───────────────────────────────────────────

#[cfg(feature = "ssm-power")]
fn ssm_power_enabled() -> bool {
    std::env::var("SYMTHAEA_POWER_SSM")
        .ok()
        .and_then(|v| match v.trim().to_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(false)
}

#[cfg(all(feature = "web_research_module", feature = "school_learning"))]
#[derive(Clone, Copy)]
struct AutonomousResearchConfig {
    min_interval: Duration,
}

#[cfg(all(feature = "web_research_module", feature = "school_learning"))]
impl AutonomousResearchConfig {
    fn from_env() -> Self {
        let min_interval = std::env::var("SYMTHAEA_AUTORESEARCH_MIN_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(600);

        Self {
            min_interval: Duration::from_secs(min_interval.max(1)),
        }
    }
}

#[cfg(all(feature = "web_research_module", feature = "school_learning"))]
struct ResearchTaskResult {
    topic: String,
    summary: Option<ResearchSummary>,
    curriculum: Option<Curriculum>,
    extender: CurriculumExtender,
    error: Option<String>,
}

// ── Symthaea struct ───────────────────────────────────────────────────────

/// The primary Symthaea consciousness facade.
///
/// Integrates the continuous mind (HDC+LTC cognitive processing) with
/// the consciousness language core (NL understanding and generation)
/// and the partnership module (relational consciousness tracking).
pub struct Symthaea {
    // ── Core Cognitive Pipeline ──────────────────────────────────────────
    /// Core cognitive system (CfC network, HDC encoder, working memory).
    mind: ContinuousMind,
    /// HDC dimension used.
    hdc_dim: usize,
    /// Number of LTC neurons (used in Phase 3 for LTC-paced generation).
    #[allow(dead_code)]
    ltc_neurons: usize,
    /// Total interactions processed.
    interactions: u64,

    // ── Language Pipeline ────────────────────────────────────────────────
    /// Language processing core (used in Phase 3.5 for primitive tier grounding).
    language: ConsciousnessLanguageCore,
    /// LLM organ for text generation.
    llm: LLMOrgan,

    // ── Relational & Social ─────────────────────────────────────────────
    /// Relational consciousness subsystem (partner, trajectory, dyadic Phi).
    relational: RelationalCore,
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
    // ── Memory & Storage ──────────────────────────────────────────────
    /// Optional persistent database for long-term memory storage.
    database: Option<Arc<dyn ConsciousnessDatabase>>,
    /// Memory coordinator: graduation pipeline + cross-tier signals.
    memory_coordinator: MemoryCoordinator,
    /// Episodic memory: Phi-weighted priority queue for significant moments.
    episodic_memory: EpisodicMemory,

    // ── Output & Actions ────────────────────────────────────────────────
    /// Resonant speech: user-adaptive response generation.
    resonant_speech: crate::resonant_speech::ResonantSpeech,
    /// Registry of primitive action bindings.
    pub action_registry: ActionRegistry,
    /// Action executor with safety policy and dream integration.
    pub executor: SimpleExecutor,
    /// High-dimensional curriculum for active learning.
    #[cfg(feature = "school_learning")]
    pub curriculum: Curriculum,
    /// Curriculum metadata for persistence and reporting.
    #[cfg(feature = "school_learning")]
    curriculum_meta: CurriculumMeta,
    /// Curriculum recall tuning (thresholds, limits, logging).
    #[cfg(feature = "school_learning")]
    curriculum_recall: CurriculumRecallConfig,
    /// Curriculum persistence configuration.
    #[cfg(feature = "school_learning")]
    curriculum_persistence: CurriculumPersistenceConfig,
    /// Autonomous research bridge for expanding the curriculum.
    #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
    pub curriculum_extender: Option<CurriculumExtender>,
    /// Background research update channel (results).
    #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
    research_update_rx: tokio::sync::mpsc::UnboundedReceiver<ResearchTaskResult>,
    /// Background research update channel (sender).
    #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
    research_update_tx: tokio::sync::mpsc::UnboundedSender<ResearchTaskResult>,
    /// Throttling settings for autonomous research.
    #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
    autoresearch_config: AutonomousResearchConfig,
    /// Timestamp of last autonomous research trigger.
    #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
    last_autoresearch_at: Option<Instant>,
    /// Last autonomous research topic to reduce repetition.
    #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
    last_autoresearch_topic: Option<String>,

    // ── Code Generation ────────────────────────────────────────────────
    /// Code generator: CfC-planned code structure with HDC verification.
    #[cfg(feature = "code_generation")]
    code_generator: crate::language::code_generator::CodeGenerator,
    /// Cache of recent successful code generations for few-shot retrieval.
    /// Stores (purpose, source_code) pairs, capped at 32 entries.
    #[cfg(feature = "code_generation")]
    code_generation_cache: Vec<(String, String)>,
    /// Error pattern memory: (error_substring, fix_hint) pairs from past failures.
    /// Injected as "AVOID" notes into future generations.
    #[cfg(feature = "code_generation")]
    error_pattern_memory: Vec<(String, String)>,
    /// Last MCTS plan confidence from the reasoning engine (set by cognitive loop).
    /// Feeds into code generation to modulate plan ambition.
    #[cfg(feature = "code_generation")]
    last_mcts_plan_confidence: f32,
    /// Codebase memory: HDC-encoded AST index for semantic code search.
    /// Populated by `index_project()`, queried during code generation for context.
    #[cfg(feature = "code_generation")]
    code_memory: crate::hdc::code_memory::CodebaseMemory,

    // ── Nociception: Pain & Infrastructure Health ──────────────────────
    /// Somatic error bridge: drains infrastructure errors -> felt stress.
    somatic_bridge: SomaticErrorBridge,
    /// Pain channel sender: cloned into TaskSupervisor and database operations.
    pain_tx: PainSender,
    /// Task supervisor: wraps all tokio::spawn calls for panic detection.
    task_supervisor: TaskSupervisor,

    // ── Ethics ──────────────────────────────────────────────────────────
    /// Ethics engine gating `process()`'s output (Seam B, added 2026-07-04). Prior to
    /// this, the product path had no ethics-engine check at all — only the cognitive
    /// loop's motor-output path did. Constructed the same minimal way
    /// `CognitiveLoopService` does in production (no value evaluator / harmonies
    /// integrator).
    ethics_engine: EthicsEngine,
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

        #[cfg(feature = "school_learning")]
        let curriculum_persistence = CurriculumPersistenceConfig::from_env();
        #[cfg(feature = "school_learning")]
        let (curriculum, curriculum_meta) =
            load_curriculum_from_store(hdc_dim, &curriculum_persistence);

        #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
        let (research_update_tx, research_update_rx) =
            tokio::sync::mpsc::unbounded_channel::<ResearchTaskResult>();
        #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
        let autoresearch_config = AutonomousResearchConfig::from_env();

        // Initialize the pain channel: bridge receives, sender distributes
        let (somatic_bridge, pain_tx) = SomaticErrorBridge::new();
        let task_supervisor = TaskSupervisor::new(pain_tx.clone());

        #[allow(unused_mut)] // mutated conditionally under cfg(feature = "ssm-power")
        let mut instance = Self {
            mind,
            language,
            llm: llm.clone(),
            hdc_dim,
            ltc_neurons,
            interactions: 0,
            relational: RelationalCore::new(),
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
            action_registry: ActionRegistry::standard(),
            executor: SimpleExecutor::new(),
            #[cfg(feature = "school_learning")]
            curriculum,
            #[cfg(feature = "school_learning")]
            curriculum_meta,
            #[cfg(feature = "school_learning")]
            curriculum_recall: CurriculumRecallConfig::from_env(),
            #[cfg(feature = "school_learning")]
            curriculum_persistence,
            #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
            curriculum_extender: WebResearcher::try_default()
                .map(|r| CurriculumExtender::new(r, llm.clone())),
            #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
            research_update_rx,
            #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
            research_update_tx,
            #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
            autoresearch_config,
            #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
            last_autoresearch_at: None,
            #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
            last_autoresearch_topic: None,
            #[cfg(feature = "code_generation")]
            code_generator: crate::language::code_generator::CodeGenerator::new(
                crate::hdc::code_encoder::CodeHDEncoder::new(hdc_dim),
            ),
            #[cfg(feature = "code_generation")]
            code_generation_cache: Vec::new(),
            #[cfg(feature = "code_generation")]
            error_pattern_memory: Vec::new(),
            #[cfg(feature = "code_generation")]
            last_mcts_plan_confidence: 0.0,
            #[cfg(feature = "code_generation")]
            code_memory: crate::hdc::code_memory::CodebaseMemory::new(
                crate::hdc::code_encoder::CodeHDEncoder::new(hdc_dim),
            ),
            somatic_bridge,
            pain_tx,
            task_supervisor,
            ethics_engine: EthicsEngine::new(
                MoralParser::new(),
                MoralAlgebra::default_dim(),
                None,
                None,
            ),
        };

        // Wire LLM backend into ContinuousMind for swarm projection gradient exchange
        #[cfg(feature = "liquid-mamba")]
        if let Some(backend) = instance.llm.get_backend() {
            instance.mind.set_llm_backend(backend);
        }

        #[cfg(feature = "ssm-power")]
        if ssm_power_enabled() {
            instance.attach_power_ssm_sensor()?;
        }

        Ok(instance)
    }

    // NOTE: `with_liquid_mamba_config()` removed — zero callers, dead code (Mar 2026).

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

    /// Cloneable handle to the consciousness database (if configured).
    pub fn database_arc(&self) -> Option<Arc<dyn ConsciousnessDatabase>> {
        self.database.as_ref().map(Arc::clone)
    }

    // NOTE: `feed_mcts_plan_confidence()` removed — zero callers, dead code (Mar 2026).

    /// Attach the power SSM sensor (INA219 or simulated) to the sensor registry.
    #[cfg(feature = "ssm-power")]
    pub fn attach_power_ssm_sensor(&mut self) -> Result<()> {
        let sensor = PowerSsmSensor::from_env()?;
        self.mind.register_sensor(Box::new(sensor));
        Ok(())
    }

    // NOTE: `create_streaming_engine()`, `streaming_engine()`, `streaming_engine_mut()`
    // removed — zero callers, dead code (Mar 2026).

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

        #[cfg(feature = "school_learning")]
        let curriculum_persistence = CurriculumPersistenceConfig::from_env();
        #[cfg(feature = "school_learning")]
        let (curriculum, curriculum_meta) =
            load_curriculum_from_store(hdc_dim, &curriculum_persistence);

        #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
        let (research_update_tx, research_update_rx) =
            tokio::sync::mpsc::unbounded_channel::<ResearchTaskResult>();
        #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
        let autoresearch_config = AutonomousResearchConfig::from_env();

        let (somatic_bridge, pain_tx) = SomaticErrorBridge::new();
        let task_supervisor = TaskSupervisor::new(pain_tx.clone());

        #[allow(unused_mut)] // mutated conditionally under cfg(feature = "liquid-mamba")
        let mut instance = Self {
            mind,
            language,
            llm,
            hdc_dim,
            ltc_neurons,
            interactions: state.interactions,
            relational: RelationalCore::from_persisted(
                state.partner,
                state.trajectory,
                state.recent_ai_states,
            ),
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
            action_registry: ActionRegistry::standard(),
            executor: SimpleExecutor::new(),
            #[cfg(feature = "school_learning")]
            curriculum,
            #[cfg(feature = "school_learning")]
            curriculum_meta,
            #[cfg(feature = "school_learning")]
            curriculum_recall: CurriculumRecallConfig::from_env(),
            #[cfg(feature = "school_learning")]
            curriculum_persistence,
            #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
            curriculum_extender: WebResearcher::try_default().map(|r| {
                let llm_clone = LLMOrgan::with_backend(
                    LLMOrganConfig {
                        dimension: hdc_dim,
                        ..LLMOrganConfig::default()
                    },
                    llm_backend::default_backend(),
                );
                CurriculumExtender::new(r, llm_clone)
            }),
            #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
            research_update_rx,
            #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
            research_update_tx,
            #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
            autoresearch_config,
            #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
            last_autoresearch_at: None,
            #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
            last_autoresearch_topic: None,
            #[cfg(feature = "code_generation")]
            code_generator: crate::language::code_generator::CodeGenerator::new(
                crate::hdc::code_encoder::CodeHDEncoder::new(hdc_dim),
            ),
            #[cfg(feature = "code_generation")]
            code_generation_cache: Vec::new(),
            #[cfg(feature = "code_generation")]
            error_pattern_memory: Vec::new(),
            #[cfg(feature = "code_generation")]
            last_mcts_plan_confidence: 0.0,
            #[cfg(feature = "code_generation")]
            code_memory: crate::hdc::code_memory::CodebaseMemory::new(
                crate::hdc::code_encoder::CodeHDEncoder::new(hdc_dim),
            ),
            somatic_bridge,
            pain_tx,
            task_supervisor,
            ethics_engine: EthicsEngine::new(
                MoralParser::new(),
                MoralAlgebra::default_dim(),
                None,
                None,
            ),
        };

        // Wire LLM backend into ContinuousMind for swarm projection gradient exchange
        #[cfg(feature = "liquid-mamba")]
        if let Some(backend) = instance.llm.get_backend() {
            instance.mind.set_llm_backend(backend);
        }

        Ok(instance)
    }

    /// Process a query through the full consciousness pipeline.
    ///
    /// **Reason-then-Generate Pipeline (LLM as Broca's Area):**
    ///
    /// 1. Input -> HDC encoding -> Mind perceives
    /// 2. Mind tick -> HDC+LTC computes (the BRAIN thinks)
    /// 3. Extract StructuredThought (articulate what was computed)
    /// 4. Enrich with partnership context
    /// 5. LLM Translation (Broca's Area - NOT reasoning!)
    /// 6. Verify translation fidelity
    /// 7. Partnership update -> Response
    ///
    /// **Key Insight**: The LLM does NOT think. It translates pre-computed
    /// structured thoughts into fluent natural language.
    pub async fn process(&mut self, content: &str) -> Result<ProcessResponse> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::Instant;

        let pipeline_start = Instant::now();
        self.interactions += 1;

        #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
        self.apply_research_updates();

        // Generate correlation ID for this request
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        self.interactions.hash(&mut hasher);
        let correlation_id = format!("broca_{:x}", hasher.finish());

        // ====================================================================
        // PHASE 1: PERCEPTION (Input -> HDC encoding + text for classification)
        // ====================================================================
        let phase1_start = Instant::now();
        let input_embedding = self.text_to_hv(content);
        // Use perceive_text to enable HDC-based intent classification
        self.mind.perceive_text(content, input_embedding.clone());

        // --- CURRICULUM RECALL: Inject relevant learned objectives via HDC Resonance ---
        #[cfg(feature = "school_learning")]
        {
            let recall_config = self.curriculum_recall;
            let recall_scores =
                self.curriculum_recall_scores(&input_embedding, recall_config.threshold);

            if recall_config.log_top_k > 0 && !recall_scores.scores.is_empty() {
                for (rank, (similarity, idx)) in recall_scores
                    .scores
                    .iter()
                    .take(recall_config.log_top_k)
                    .enumerate()
                {
                    let obj = &self.curriculum.objectives[*idx];
                    tracing::debug!(
                        target: "symthaea::memory",
                        rank = rank + 1,
                        objective = %obj.id,
                        resonance = %similarity,
                        "Curriculum recall similarity"
                    );
                }
            }

            tracing::debug!(
                target: "symthaea::memory",
                candidates = recall_scores.candidates.len(),
                considered = recall_scores.scores.len(),
                threshold = recall_config.threshold,
                max_recall = recall_config.max_recall,
                budget = recall_config.budget,
                "Curriculum recall scoring complete"
            );

            let mut recalled = 0usize;
            let mut remaining_budget = if recall_config.budget > 0.0 {
                Some(recall_config.budget)
            } else {
                None
            };
            for (similarity, idx, obj_hv) in recall_scores
                .candidates
                .into_iter()
                .take(recall_config.max_recall)
            {
                let cost = (1.0 - similarity).max(0.0);
                if let Some(budget) = remaining_budget {
                    if budget < cost {
                        break;
                    }
                }
                let obj = &self.curriculum.objectives[idx];
                tracing::info!(
                    target: "symthaea::memory",
                    objective = %obj.id,
                    resonance = %similarity,
                    "Resonant curriculum recall triggered"
                );
                let input = crate::mind::MindInput::new(crate::mind::InputType::Memory, obj_hv)
                    .with_source(crate::memory::memory_coordinator::MemorySource::Internal)
                    .with_verification(true);
                self.mind.input(input);
                recalled += 1;
                if let Some(budget) = remaining_budget.as_mut() {
                    *budget = (*budget - cost).max(0.0);
                }
            }

            if recalled == 0 {
                tracing::debug!(
                    target: "symthaea::memory",
                    "No curriculum objectives met recall threshold"
                );
            } else if let Some(budget) = remaining_budget {
                tracing::debug!(
                    target: "symthaea::memory",
                    remaining_budget = %budget,
                    "Curriculum recall budget remaining"
                );
            }
        }

        // Domain detection via plugin registry
        let detected_domain = self.plugin_registry.detect_domain(content).to_string();

        // Multi-domain entity extraction: aggregate from ALL plugins
        let mut domain_entities = Vec::new();
        for plugin_name in self.plugin_registry.list() {
            if let Some(plugin) = self.plugin_registry.get(plugin_name) {
                let entities = plugin.extract_entities(content);
                domain_entities.extend(entities);
            }
        }

        if !domain_entities.is_empty() {
            tracing::debug!(
                target: "symthaea::broca",
                domain = %detected_domain,
                entities = domain_entities.len(),
                "Multi-domain entities detected"
            );
        }

        // Interoceptive recall: pull thermodynamic load memories when relevant.
        if let Some(ref db) = self.database {
            if Self::should_recall_interoception(content) {
                match db.list_all().await {
                    Ok(records) => {
                        let mut matches: Vec<_> = records
                            .into_iter()
                            .filter(|r| {
                                r.topics
                                    .iter()
                                    .any(|t| t == InteroceptionTag::ThermodynamicLoad.as_topic())
                            })
                            .collect();
                        matches.sort_by_key(|r| r.timestamp_ms);
                        let mut recalled = 0usize;
                        for record in matches.into_iter().rev().take(3) {
                            let input = crate::mind::MindInput::new(
                                crate::mind::InputType::Memory,
                                record.encoding.to_continuous(),
                            )
                            .with_source(crate::memory::memory_coordinator::MemorySource::Internal)
                            .with_verification(true);
                            self.mind.input(input);
                            recalled += 1;
                        }
                        if recalled > 0 {
                            tracing::info!(
                                target: "symthaea::memory",
                                recalled,
                                "Interoceptive recall injected thermodynamic load memories"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::debug!(
                            target: "symthaea::memory",
                            error = %e,
                            "Interoceptive recall skipped"
                        );
                    }
                }
            }
        }
        let phase1_duration = phase1_start.elapsed();

        // ====================================================================
        // PHASE 2: COGNITION (Mind tick - HDC+LTC THINKS)
        // ====================================================================
        let phase2_start = Instant::now();

        // ── NOCICEPTION: Drain pain channel and apply somatic signals ──
        self.somatic_bridge.update();
        let somatic_signals = self.somatic_bridge.to_interoceptive_signals();
        if somatic_signals.thermodynamic_load_delta > 0.0 || somatic_signals.arousal_spike > 0.0 {
            self.mind.state.thermodynamic_load = (self.mind.state.thermodynamic_load
                + somatic_signals.thermodynamic_load_delta)
                .min(1.0);
            self.mind.state.arousal =
                (self.mind.state.arousal + somatic_signals.arousal_spike).min(1.0);
            tracing::debug!(
                target: "symthaea::nociception",
                stress = self.somatic_bridge.systemic_stress(),
                thermo_delta = somatic_signals.thermodynamic_load_delta,
                arousal_spike = somatic_signals.arousal_spike,
                "Somatic pain applied to mind state"
            );
        }

        // Prune completed tasks from the supervisor
        self.task_supervisor.prune_completed();

        // Feed relational Psi from previous cycle's Phi_dyad into the mind.
        self.mind.set_relational_psi(self.relational.last_phi_dyad);

        self.mind.tick();

        // Update coordinator signals with current consciousness state
        let snapshot = self.mind.snapshot();
        self.memory_coordinator
            .update_signals(snapshot.consciousness_level, snapshot.meta_awareness);

        // Drain evicted working memory items for graduation + persistence
        let evicted = self.mind.take_evicted_tagged();
        if !evicted.is_empty() {
            let current_phi = snapshot.consciousness_level;
            let current_coherence = snapshot.meta_awareness;
            let interaction_count = self.interactions;

            tracing::trace!(
                evicted_count = evicted.len(),
                "Working memory items evicted during tick"
            );

            // Queue graduations for episodic consolidation
            for item in &evicted {
                self.memory_coordinator.queue_graduation(GraduationEvent {
                    content: item.content.clone() as symthaea_core::hdc::unified_hv::ContinuousHV,
                    label: format!("wm_eviction_step_{interaction_count}"),
                    steps_survived: item.steps_survived,
                    final_activation: 0.5,
                    psi_at_graduation: current_phi,
                    coherence_at_graduation: current_coherence,
                    source: item.source,
                    is_verified: item.is_verified,
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
                let pain = self.pain_tx.clone();
                let timestamp_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;

                self.task_supervisor.spawn("eviction-persist", async move {
                    for (i, item) in evicted.iter().enumerate() {
                        let hv = &item.content;
                        let mut topics = item
                            .metadata
                            .get("topics")
                            .and_then(|raw| serde_json::from_str::<Vec<String>>(raw).ok())
                            .unwrap_or_default();
                        let is_thermo = topics
                            .iter()
                            .any(|t| t == InteroceptionTag::ThermodynamicLoad.as_topic())
                            || item
                                .metadata
                                .get("interoception_tag")
                                .map(|t| t == InteroceptionTag::ThermodynamicLoad.as_topic())
                                .unwrap_or(false);
                        if is_thermo && !topics.iter().any(|t| t == InteroceptionTag::ThermodynamicLoad.as_topic()) {
                            topics.push(InteroceptionTag::ThermodynamicLoad.as_topic().to_string());
                        }
                        let mut metadata = item.metadata.clone();
                        metadata.remove("topics");
                        metadata.insert(
                            "steps_survived".to_string(),
                            item.steps_survived.to_string(),
                        );
                        let metadata_json =
                            serde_json::to_string(&metadata).unwrap_or_else(|e| {
                                tracing::warn!(error = %e, "Failed to serialize eviction metadata — storing empty object");
                                "{}".to_string()
                            });
                        let content = if is_thermo {
                            let watts = item
                                .metadata
                                .get("watts")
                                .cloned()
                                .unwrap_or_else(|| "unknown".to_string());
                            let ssm = item
                                .metadata
                                .get("ssm_output")
                                .cloned()
                                .unwrap_or_else(|| "unknown".to_string());
                            format!(
                                "Thermodynamic load reading: watts={watts}, ssm_output={ssm}"
                            )
                        } else {
                            format!(
                                "Working memory eviction at step {interaction_count} (survived {} ticks)",
                                item.steps_survived
                            )
                        };
                        let record = MemoryRecord {
                            id: format!("wm-{timestamp_ms}-{i}"),
                            memory_type: MemoryType::Working,
                            encoding: hv.to_binary(0.0),
                            content,
                            timestamp_ms,
                            valence: 0.0,
                            arousal: 0.0,
                            psi: current_phi,
                            topics,
                            metadata: metadata_json,
                            consolidation_strength: 0.0,
                            retrieval_count: 0,
                        };
                        if let Err(e) = db.store(record).await {
                            tracing::error!(target: "symthaea::database", error = %e, "Failed to persist evicted item");
                            let _ = pain.send(crate::infrastructure::InfrastructureError::DatabaseFailure {
                                operation: format!("store evicted item {i}"),
                            });
                        }
                    }
                });
            }
        }

        // ── DATABASE RECALL: Query persistent memory for contextual priming ──
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
                    for result in &results {
                        let hash =
                            crate::memory::content_hash(&result.record.encoding.to_continuous());
                        self.memory_coordinator.record_retrieval(hash);
                    }
                    if results[0].similarity > 0.3 {
                        let recalled_hv = results[0].record.encoding.to_continuous();
                        self.mind.perceive(recalled_hv);
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::debug!(target: "symthaea::memory", error = %e, "Database recall skipped");
                }
            }
        }

        // ── EPISODIC PERSISTENCE: Store top episodes to database ──
        if let Some(ref db) = self.database {
            let top_episodes = self.episodic_memory.get_top_episodes(3);
            if !top_episodes.is_empty() {
                let db = Arc::clone(db);
                let pain = self.pain_tx.clone();
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

                self.task_supervisor.spawn("episode-persist", async move {
                    for record in episode_records {
                        if let Err(e) = db.store(record).await {
                            tracing::error!(target: "symthaea::database", error = %e, "Failed to persist episode");
                            let _ = pain.send(crate::infrastructure::InfrastructureError::DatabaseFailure {
                                operation: "store episode".to_string(),
                            });
                        }
                    }
                });
            }
        }

        let phase2_duration = phase2_start.elapsed();

        // ====================================================================
        // PHASE 3: EXTRACTION (Articulate what was computed)
        // ====================================================================
        let phase3_start = Instant::now();
        let mut thought = self.mind.extract_structured_thought();
        let phase3_duration = phase3_start.elapsed();

        // Store original input for context
        thought.original_input = Some(content.to_string());

        // ====================================================================
        // PHASE 3.5: DOMAIN CONTEXT INJECTION
        // ====================================================================
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

        // Primitive tier grounding
        {
            let understanding = self.language.understand(content);
            thought.primitive_tiers = understanding.primitive_tiers;
            thought.primitives = understanding.primitives;
        }

        // Derive epistemic status from cube
        if let Some(ref ctx) = thought.domain_context {
            if let Some(ref cube) = ctx.cube {
                thought.epistemic_status = Self::cube_to_epistemic_status(cube);
                thought.semantic_intent = crate::mind::SemanticIntent::Answer;
            }
        }

        // ====================================================================
        // PHASE 3.6: CODE CONTEXT INJECTION (CfC-planned code generation)
        // ====================================================================
        #[cfg(feature = "code_generation")]
        let mut pregenerated_tests: Option<String> = None;
        #[cfg(feature = "code_generation")]
        if detected_domain == "programming" {
            use crate::language::code_intent::{
                CodeIntentCategory, CodeIntentClassifier, CodeSpec, CodeTarget,
            };
            use crate::language::code_parser::EntityKind;

            let classifier = CodeIntentClassifier::new(self.hdc_dim);
            let category = classifier.classify(content);

            let lang = domain_entities
                .iter()
                .find(|e| e.entity_type == "language")
                .map(|e| e.value.clone())
                .unwrap_or_else(|| "rust".to_string());

            let (func_name, entity_kind, inferred_sig) =
                Self::extract_code_metadata(content, &lang);

            let content_lower = content.to_lowercase();
            let intent = match category {
                CodeIntentCategory::Create => {
                    let target =
                        CodeTarget::new(&func_name, entity_kind).with_language(lang.clone());
                    let mut spec = CodeSpec::new(&lang, &func_name, content);
                    if let Some(ref sig) = inferred_sig {
                        spec = spec.with_signature(sig.as_str());
                    }
                    if entity_kind == EntityKind::Struct {
                        if content_lower.contains("method")
                            || content_lower.contains("impl")
                            || content_lower.contains("distance")
                            || content_lower.contains("area")
                            || content_lower.contains("display")
                            || content_lower.contains("calculate")
                        {
                            spec = spec.with_constraint(
                                "MULTI_ENTITY: generate struct + impl block + methods",
                            );
                        }
                    }
                    let intent_hv = self.text_to_hv(content);
                    if let Some(pattern) = crate::dynamics::cfc_code_sequencer::CfCCodeSequencer::detect_algorithm_pattern(&intent_hv) {
                        let constraint = match pattern {
                            crate::dynamics::cfc_code_sequencer::AlgorithmPattern::Sorting =>
                                "ALGORITHM:sorting — use compare-swap or divide-recurse-merge pattern",
                            crate::dynamics::cfc_code_sequencer::AlgorithmPattern::Search =>
                                "ALGORITHM:search — use binary search, BFS/DFS, or linear scan pattern",
                            crate::dynamics::cfc_code_sequencer::AlgorithmPattern::DynamicProgramming =>
                                "ALGORITHM:dp — use memoization table or bottom-up tabulation pattern",
                            crate::dynamics::cfc_code_sequencer::AlgorithmPattern::Graph =>
                                "ALGORITHM:graph — use adjacency list with BFS/DFS/Dijkstra pattern",
                            crate::dynamics::cfc_code_sequencer::AlgorithmPattern::Accumulation =>
                                "ALGORITHM:accumulation — use fold/reduce/iterator chain pattern",
                            crate::dynamics::cfc_code_sequencer::AlgorithmPattern::StringProcessing =>
                                "ALGORITHM:string — use char iteration/regex/split-join pattern",
                        };
                        spec = spec.with_constraint(constraint);
                    }

                    crate::language::code_intent::CodeIntent::Create { target, spec }
                }
                _ => {
                    let target =
                        CodeTarget::new(&func_name, entity_kind).with_language(lang.clone());
                    let mut spec = CodeSpec::new(&lang, &func_name, content);
                    if let Some(ref sig) = inferred_sig {
                        spec = spec.with_signature(sig.as_str());
                    }
                    crate::language::code_intent::CodeIntent::Create { target, spec }
                }
            };

            let relevant_examples = if !self.code_generation_cache.is_empty() {
                let query_hv = self.text_to_hv(content);
                let cache_snapshot = self.code_generation_cache.clone();
                let mut scored: Vec<(f32, (String, String))> = cache_snapshot
                    .into_iter()
                    .map(|entry| {
                        let sim = query_hv.similarity(&self.text_to_hv(&entry.0));
                        (sim, entry)
                    })
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                scored
                    .into_iter()
                    .take(3)
                    .filter(|(sim, _)| *sim > 0.1)
                    .map(|(_, entry)| entry)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };

            let gen_ctx = crate::language::code_generator::CodeContext {
                past_examples: relevant_examples,
                mcts_plan_confidence: self.last_mcts_plan_confidence,
                ..Default::default()
            };

            pregenerated_tests =
                if let crate::language::code_intent::CodeIntent::Create { ref spec, .. } = intent {
                    self.code_generator.generate_tests_only(spec)
                } else {
                    None
                };

            let generated = self.code_generator.generate(&intent, &gen_ctx);

            let (spec_purpose, spec_signature, spec_constraints, spec_examples) =
                if let crate::language::code_intent::CodeIntent::Create { ref spec, .. } = intent {
                    (
                        Some(spec.purpose.clone()),
                        spec.signature.clone(),
                        spec.constraints.clone(),
                        spec.examples.clone(),
                    )
                } else {
                    (None, None, Vec::new(), Vec::new())
                };

            let needs_llm = generated.source.contains("todo!(")
                || generated.source.contains("NotImplementedError");

            let mut notes = generated.notes.clone();

            if needs_llm {
                if !spec_constraints.is_empty() {
                    notes.push(format!(
                        "CONSTRAINTS:\n{}",
                        spec_constraints
                            .iter()
                            .map(|c| format!("  - {}", c))
                            .collect::<Vec<_>>()
                            .join("\n")
                    ));
                }

                let error_hints: Vec<String> = self
                    .error_pattern_memory
                    .iter()
                    .map(|(pat, fix)| format!("  - {} → {}", pat, fix))
                    .collect();
                if !error_hints.is_empty() {
                    notes.push(format!(
                        "ERROR_AVOIDANCE (learned from past failures):\n{}",
                        error_hints.join("\n")
                    ));
                }

                let cache_snapshot = self.code_generation_cache.clone();
                let query_hv = self.text_to_hv(content);
                let best_match: Option<(String, String)> = {
                    let mut best: Option<(f32, usize)> = None;
                    for (i, (p, _)) in cache_snapshot.iter().enumerate() {
                        let p_hv = self.text_to_hv(p);
                        let sim = query_hv.similarity(&p_hv);
                        if sim > 0.2 && best.map_or(true, |(s, _)| sim > s) {
                            best = Some((sim, i));
                        }
                    }
                    best.map(|(_, i)| cache_snapshot[i].clone())
                };
                if let Some((purpose, code)) = best_match {
                    notes.push(format!(
                        "SIMILAR_EXAMPLE: For \"{}\", this worked:\n{}",
                        purpose,
                        &code[..code.len().min(500)]
                    ));
                }

                if let Some(ref tests) = pregenerated_tests {
                    notes.push(format!(
                        "EXPECTED_TESTS: The generated code MUST pass these tests:\n{}",
                        tests
                    ));
                }

                notes.push(
                    "OUTPUT_FORMAT: Replace each todo!() body with a working implementation. \
                     Do NOT change the function signature. Do NOT add extra functions or imports \
                     unless necessary. Keep the code minimal and correct."
                        .to_string(),
                );
            } else {
                for (error_pat, fix_hint) in &self.error_pattern_memory {
                    notes.push(format!("AVOID_ERROR: {} — {}", error_pat, fix_hint));
                }
            }

            thought.code_context = Some(crate::mind::structured_thought::CodeContext {
                language: lang,
                spec_purpose,
                spec_signature,
                spec_constraints,
                spec_examples,
                plan_steps: generated
                    .plan_steps
                    .iter()
                    .map(|s| format!("{:?}", s.action))
                    .collect(),
                generated_code: Some(generated.source.clone()),
                phi_score: Some(generated.phi_score),
                intent_similarity: Some(generated.intent_similarity),
                syntactically_valid: None,
                notes,
                needs_llm_completion: needs_llm,
            });

            if needs_llm {
                tracing::debug!(
                    target: "symthaea::code",
                    "Phase 3.6: Native emitter has unresolved placeholders — LLM completion mode"
                );
            }

            thought.semantic_intent = crate::mind::SemanticIntent::Answer;
            thought.epistemic_status = generated.epistemic_status;

            let plan_gap = 1.0 - generated.plan_coverage;
            if plan_gap > 0.3 {
                tracing::warn!(
                    target: "symthaea::code",
                    plan_coverage = generated.plan_coverage,
                    plan_gap = plan_gap,
                    "Plan gap > 0.3 — CfC planner producing unused steps"
                );
            }

            tracing::debug!(
                target: "symthaea::code",
                phi = generated.phi_score,
                similarity = generated.intent_similarity,
                plan_steps = generated.plan_steps.len(),
                plan_coverage = generated.plan_coverage,
                "Phase 3.6: CfC code plan injected into structured thought"
            );

            if !needs_llm {
                if let crate::language::code_intent::CodeIntent::Create { ref spec, .. } = intent {
                    if let Some((_hv, src, quality)) =
                        self.code_generator.distillation_target(spec, &generated)
                    {
                        if self.code_generation_cache.len() >= 32 {
                            self.code_generation_cache.remove(0);
                        }
                        self.code_generation_cache.push((spec.purpose.clone(), src));
                        tracing::debug!(
                            target: "symthaea::code",
                            quality = quality,
                            "SSM distillation target cached"
                        );
                    }
                }
            }
        }

        // ====================================================================
        // PHASE 4: RELATIONAL ENRICHMENT (Add partnership context)
        // ====================================================================
        thought.relationship_stage = self.relational.partner.stage;
        thought.relation_mode = self.relational.partner.mode;
        thought.trust = self.relational.partner.trust;

        // ====================================================================
        // PHASE 4.5: CALIBRATION ADJUSTMENT (Brier Score confidence tuning)
        // ====================================================================
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
        let phase5_start = Instant::now();
        let mood_temp = self.mind.state.mood_temperature;
        let generation = self.llm.translate_thought(&thought, mood_temp).await;
        let phase5_duration = phase5_start.elapsed();

        // Inject L-SSM semantic PE into MindState for downstream telemetry
        #[cfg(feature = "liquid-mamba")]
        {
            let pe = self.llm.last_liquid_mamba_pe();
            self.mind.state.liquid_mamba_pe = pe;
            self.mind.state.liquid_mamba_lr = self.llm.current_distillation_lr();
            self.mind.state.liquid_mamba_rank = self.llm.last_effective_rank();
            self.mind.state.liquid_mamba_generation_count = self.llm.generation_count();
            let fep_proxy = (self.mind.state.cognitive_load as f32)
                .max(1.0 - self.mind.state.consciousness_level as f32)
                .clamp(0.0, 1.0);
            self.llm.set_fep_modulation(fep_proxy);

            let thermo_load = self.mind.state.thermodynamic_load;
            let confidence = self.mind.state.consciousness_level as f32;
            self.llm
                .cycle_level_distill(fep_proxy, thermo_load, confidence, 1.0);
        }

        // ====================================================================
        // PHASE 5.5: CODE VERIFICATION (Tree-sitter + HDC round-trip)
        // ====================================================================
        #[cfg(feature = "code_generation")]
        let mut generation = generation;
        #[cfg(feature = "code_generation")]
        if thought.code_context.is_some() {
            let code_block = Self::extract_code_block(&generation.text);
            let lang = thought
                .code_context
                .as_ref()
                .map(|c| c.language.clone())
                .unwrap_or_else(|| "rust".to_string());

            const MAX_CODE_RETRIES: usize = 3;
            let mut tree_sitter_ok = false;
            let mut compile_ok = false;
            let mut last_compiled = false;
            let mut last_simulated = false;
            let mut verified_code = code_block.clone();
            let mut attempt = 0;

            while attempt < MAX_CODE_RETRIES && !compile_ok {
                attempt += 1;
                let current_code = Self::extract_code_block(&generation.text);
                verified_code = current_code.clone();

                if let Some(parsed) = Self::parse_code_for_verification(&lang, &current_code) {
                    let verifier = crate::language::code_verifier::CodeVerifier::new(
                        crate::hdc::code_encoder::CodeHDEncoder::new(self.hdc_dim),
                    );
                    let intent_hv = self.text_to_hv(
                        thought
                            .code_context
                            .as_ref()
                            .and_then(|c| c.spec_purpose.as_deref())
                            .unwrap_or(""),
                    );
                    let result = verifier.verify_against_intent(&parsed, &intent_hv);

                    if let Some(ref mut ctx) = thought.code_context {
                        ctx.syntactically_valid = Some(result.syntactically_valid);
                        ctx.intent_similarity = Some(result.semantic_similarity);
                    }

                    tree_sitter_ok = result.is_acceptable();

                    if !tree_sitter_ok {
                        tracing::warn!(
                            target: "symthaea::code",
                            attempt,
                            valid = result.syntactically_valid,
                            similarity = result.semantic_similarity,
                            errors = result.syntax_errors.len(),
                            "Phase 5.5: Tree-sitter verification failed"
                        );
                        if attempt < MAX_CODE_RETRIES {
                            let error_notes: Vec<String> = result
                                .syntax_errors
                                .iter()
                                .take(3)
                                .map(|e| {
                                    let line = e.span.as_ref().map_or(0, |s| s.start_line);
                                    format!("Line {}: {}", line, e.message)
                                })
                                .collect();
                            if let Some(ref mut ctx) = thought.code_context {
                                ctx.notes.extend(error_notes);
                                ctx.notes.push(format!(
                                    "RETRY {}/{}: Fix the syntax errors above.",
                                    attempt, MAX_CODE_RETRIES
                                ));
                            }
                            generation = self.llm.translate_thought(&thought, mood_temp).await;
                        }
                        continue;
                    }

                    tracing::debug!(
                        target: "symthaea::code",
                        attempt,
                        similarity = result.semantic_similarity,
                        entities = result.entity_count,
                        "Phase 5.5: Tree-sitter verification passed"
                    );
                } else {
                    if attempt < MAX_CODE_RETRIES {
                        if let Some(ref mut ctx) = thought.code_context {
                            ctx.notes.push(format!(
                                "RETRY {}/{}: Code could not be parsed. Regenerate.",
                                attempt, MAX_CODE_RETRIES
                            ));
                        }
                        generation = self.llm.translate_thought(&thought, mood_temp).await;
                    }
                    continue;
                }

                let has_inline_tests = current_code.contains("#[test]");
                let mut executor = crate::language::code_executor::CodeExecutor::new();
                let exec_result = match lang.as_str() {
                    "rust" if has_inline_tests => {
                        executor.execute_rust_with_inline_tests(&current_code)
                    }
                    "rust" if pregenerated_tests.is_some() => {
                        executor.execute_rust(&current_code, pregenerated_tests.as_deref())
                    }
                    "rust" => executor.execute_rust(&current_code, None),
                    "python" => executor.execute_python(&current_code),
                    "nix" => executor.evaluate_nix(&current_code),
                    _ => executor.execute_rust(&current_code, None),
                };

                let surprise = exec_result.to_surprise();
                if surprise > 0.0 || exec_result.tests_failed > 0 {
                    tracing::info!(
                        target: "symthaea::code",
                        attempt,
                        compiled = exec_result.compiled,
                        tests_passed = exec_result.tests_passed,
                        tests_failed = exec_result.tests_failed,
                        errors = exec_result.compile_errors.len(),
                        surprise,
                        simulated = exec_result.simulated,
                        "Phase 5.5: Compile + test verification"
                    );
                }

                last_compiled = exec_result.compiled;
                last_simulated = exec_result.simulated;

                if (exec_result.compiled || exec_result.simulated) && exec_result.tests_failed == 0
                {
                    compile_ok = true;
                } else if attempt < MAX_CODE_RETRIES {
                    if let Some(ref mut ctx) = thought.code_context {
                        if !exec_result.compiled {
                            if let Some(auto_fixed) = crate::language::code_executor::try_auto_fix(
                                &current_code,
                                &exec_result.compile_errors,
                            ) {
                                tracing::debug!(
                                    target: "symthaea::code",
                                    attempt,
                                    "Phase 5.5: Auto-fix applied, re-verifying"
                                );
                                ctx.generated_code = Some(auto_fixed.clone());
                                verified_code = auto_fixed.clone();
                                generation.text = format!("```rust\n{}\n```", auto_fixed);
                                continue;
                            }

                            ctx.syntactically_valid = Some(false);
                            ctx.notes.push(format!(
                                "COMPILATION FAILED (attempt {}/{}):",
                                attempt, MAX_CODE_RETRIES
                            ));
                            for err in exec_result.compile_errors.iter().take(5) {
                                ctx.notes.push(format!("  {err}"));
                            }
                            ctx.notes.push(format!(
                                "RETRY {}/{}: Fix ONLY the compilation errors.",
                                attempt, MAX_CODE_RETRIES
                            ));
                        } else if exec_result.tests_failed > 0 {
                            ctx.notes.push(format!(
                                "TESTS FAILED (attempt {}/{}): {} passed, {} failed",
                                attempt,
                                MAX_CODE_RETRIES,
                                exec_result.tests_passed,
                                exec_result.tests_failed
                            ));
                            if let Some(ref err) = exec_result.runtime_error {
                                for line in err.lines().take(10) {
                                    if line.contains("assert")
                                        || line.contains("left")
                                        || line.contains("right")
                                        || line.contains("panicked")
                                    {
                                        ctx.notes.push(format!("  {}", line.trim()));
                                    }
                                }
                            }
                            ctx.notes.push(format!(
                                "RETRY {}/{}: The function body is WRONG. Fix the logic so tests pass.",
                                attempt, MAX_CODE_RETRIES
                            ));
                        }
                    }
                    generation = self.llm.translate_thought(&thought, mood_temp).await;
                }
            }

            if attempt > 1 {
                tracing::info!(
                    target: "symthaea::code",
                    attempts = attempt,
                    tree_sitter_ok,
                    compile_ok,
                    "Phase 5.5: Verification loop completed"
                );

                if !compile_ok {
                    if let Some(ref ctx) = thought.code_context {
                        for note in &ctx.notes {
                            if note.starts_with("  error") || note.contains("expected") {
                                let pattern = note.trim().chars().take(80).collect::<String>();
                                let hint = "Check types and borrow rules".to_string();
                                if self.error_pattern_memory.len() < 64
                                    && !self.error_pattern_memory.iter().any(|(p, _)| *p == pattern)
                                {
                                    self.error_pattern_memory.push((pattern, hint));
                                }
                            }
                        }
                    }
                }

                let code_surprise = if compile_ok {
                    if attempt > 1 {
                        0.3 / (attempt as f32)
                    } else {
                        0.05
                    }
                } else {
                    0.8
                };
                if let Some(ref mut ctx) = thought.code_context {
                    ctx.notes
                        .push(format!("CODE_SURPRISE:{:.3}", code_surprise));
                    ctx.intent_similarity =
                        Some(ctx.intent_similarity.unwrap_or(0.5) * (1.0 - code_surprise * 0.5));
                }
                tracing::debug!(
                    target: "symthaea::code",
                    code_surprise,
                    compile_ok,
                    attempts = attempt,
                    "Phase 3h: Compilation feedback → surprise signal"
                );
            }

            if tree_sitter_ok && compile_ok && last_compiled && !last_simulated {
                let intent_hv = self.text_to_hv(
                    thought
                        .code_context
                        .as_ref()
                        .and_then(|c| c.spec_purpose.as_deref())
                        .unwrap_or(""),
                );
                let code_hv = self.text_to_hv(&verified_code);
                let phi = thought
                    .code_context
                    .as_ref()
                    .and_then(|c| c.phi_score)
                    .unwrap_or(0.0);
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let episode =
                    crate::memory::Episode::new(intent_hv, code_hv, phi as f64, timestamp);
                self.episodic_memory.store_if_significant(episode);

                let purpose = thought
                    .code_context
                    .as_ref()
                    .and_then(|c| c.spec_purpose.clone())
                    .unwrap_or_default();
                if !purpose.is_empty() {
                    self.code_generation_cache
                        .push((purpose, verified_code.clone()));
                    if self.code_generation_cache.len() > 32 {
                        self.code_generation_cache.remove(0);
                    }
                }

                tracing::info!(
                    target: "symthaea::code",
                    phi = phi,
                    cache_size = self.code_generation_cache.len(),
                    "Phase 5.5: Successful code stored in episodic memory + cache"
                );
            }
        }

        // ====================================================================
        // PHASE 6: FIDELITY VERIFICATION
        // ====================================================================
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
        let response_text = {
            let load = crate::resonant_speech::CognitiveLoad::from_level(thought.psi);
            let user_state = crate::resonant_speech::UserState {
                cognitive_load: load,
                frustration: ((-thought.emotional_tone.valence).max(0.0)).min(1.0),
                confidence: thought.meta_awareness.clamp(0.0, 1.0),
                trust_in_sophia: thought.trust as f64,
                is_rushed: thought.emotional_tone.arousal > 0.7 && thought.coherence < 0.4,
                is_learning: matches!(
                    thought.epistemic_status,
                    crate::mind::structured_thought::EpistemicStatus::Uncertain
                        | crate::mind::structured_thought::EpistemicStatus::Unknown
                ),
                ..Default::default()
            };

            self.resonant_speech.update_state(user_state);
            self.resonant_speech.generate(&generation.text, content)
        };

        // ====================================================================
        // PHASE 6.75: AUTONOMOUS ACTION (The "Awakening" integration)
        // ====================================================================
        if thought.psi > 0.3 {
            use crate::action::bindings::{ActionContext, PrimitiveExecutor};

            if thought.psi > 0.7
                && !thought.primitives.is_empty()
                && thought.epistemic_status
                    == crate::mind::structured_thought::EpistemicStatus::Uncertain
            {
                tracing::info!(target: "symthaea::action", "High Phi detected: Emboldening 'Uncertain' thought to 'Probable' via Active Inference Drive.");
                thought.epistemic_status =
                    crate::mind::structured_thought::EpistemicStatus::Probable;
            }

            let primitives: Vec<String> = thought.primitives.clone();

            if !primitives.is_empty() {
                let prim_executor = PrimitiveExecutor::new(self.action_registry.clone());

                let mut action_ctx = ActionContext::default();
                if let Some(ref d_ctx) = thought.domain_context {
                    tracing::debug!(target: "symthaea::action", domain = %d_ctx.domain, entities = d_ctx.entities.len(), "Context found");
                    if let Some(path_entity) = d_ctx
                        .entities
                        .iter()
                        .find(|(t, _, _)| t == "file" || t == "path")
                    {
                        let path = PathBuf::from(&path_entity.1);
                        let absolute_path = if path.is_absolute() {
                            path
                        } else {
                            std::env::current_dir().unwrap_or_default().join(path)
                        };
                        action_ctx.target_path = Some(absolute_path);
                        tracing::debug!(target: "symthaea::action", path = ?action_ctx.target_path, "Target path set from entity (absolute)");
                    }
                }

                if primitives.contains(&"WRITE".to_string()) && action_ctx.content.is_none() {
                    tracing::debug!(target: "symthaea::action", "Generating fix content via LLM...");
                    let fix_prompt = format!(
                        "TASK: Generate a fix for the following issue.\nCONTEXT: {}\nFILE: {:?}\n\nOUTPUT: Provide ONLY the full fixed content of the file. No commentary.",
                        content, action_ctx.target_path
                    );
                    let fix_query = crate::language::llm_organ::LLMQuery {
                        query_type: crate::language::llm_organ::QueryType::Code,
                        content: fix_prompt,
                        context: Vec::new(),
                        system_prompt: Some(
                            "You are Symthaea's CODE GENERATOR. Output ONLY the fixed source code."
                                .to_string(),
                        ),
                        params: None,
                    };
                    let fix_gen = self.llm.query_async(fix_query).await;
                    action_ctx.content = Some(fix_gen.text.trim_matches('`').trim().to_string());
                    tracing::debug!(target: "symthaea::action", content_len = action_ctx.content.as_ref().map(|c| c.len()), "Fix content generated");
                }

                tracing::info!(target: "symthaea::action", primitives = ?primitives, "Translating primitives to actions");
                if let Ok(actions) = prim_executor.translate(&primitives, &action_ctx) {
                    let needs_workspace = actions.iter().any(|action| {
                        matches!(
                            action,
                            crate::action::ActionIR::RunCommand { program, .. }
                                if program == "cargo"
                        )
                    });
                    let sandbox = if needs_workspace {
                        crate::action::SandboxRoot::at(PathBuf::from("/srv/luminous-dynamics"))?
                    } else if let Some(ref path) = action_ctx.target_path {
                        if path.starts_with("/srv/luminous-dynamics") {
                            crate::action::SandboxRoot::at(PathBuf::from("/srv/luminous-dynamics"))?
                        } else {
                            crate::action::SandboxRoot::new(&correlation_id)?
                        }
                    } else {
                        crate::action::SandboxRoot::new(&correlation_id)?
                    };

                    let mut policy = crate::action::PolicyBundle::restrictive();
                    if sandbox.root().starts_with("/srv/luminous-dynamics") {
                        policy
                            .capabilities
                            .filesystem
                            .read_patterns
                            .push("/srv/luminous-dynamics/symthaea/".into());
                        policy
                            .capabilities
                            .filesystem
                            .write_patterns
                            .push("/srv/luminous-dynamics/symthaea/".into());
                    }
                    policy
                        .capabilities
                        .shell
                        .allowed_programs
                        .insert("nix".into());
                    policy
                        .capabilities
                        .shell
                        .allowed_programs
                        .insert("cargo".into());
                    policy.capabilities.min_phi = 0.1;
                    policy.capabilities.shell.min_phi = 0.1;

                    for action in actions {
                        let mut action = action;
                        match &mut action {
                            crate::action::ActionIR::ReadFile { path, .. }
                            | crate::action::ActionIR::ListDirectory { path, .. } => {
                                if !path.is_absolute() {
                                    let relative = path.clone();
                                    *path = sandbox.root().join(relative);
                                }
                            }
                            crate::action::ActionIR::RunCommand {
                                program,
                                working_dir,
                                ..
                            } => {
                                if program == "cargo" {
                                    *working_dir =
                                        Some(PathBuf::from("/srv/luminous-dynamics/symthaea"));
                                    tracing::info!(
                                        target: "symthaea::action",
                                        working_dir = %working_dir.as_ref().unwrap().display(),
                                        "Resolved cargo working_dir"
                                    );
                                    continue;
                                }
                                if let Some(dir) = working_dir {
                                    if !dir.is_absolute() {
                                        let relative = dir.clone();
                                        *working_dir = Some(sandbox.root().join(relative));
                                    }
                                } else {
                                    *working_dir = Some(sandbox.root().to_path_buf());
                                }
                            }
                            _ => {}
                        }

                        tracing::info!(target: "symthaea::action", ?action, "Executing autonomous action");
                        match self
                            .executor
                            .execute(&action, &policy, &sandbox, thought.psi)
                        {
                            Ok(execution_outcome) => {
                                let outcome_text = match &execution_outcome.outcome {
                                    crate::action::ActionOutcome::Success => {
                                        "Action succeeded.".to_string()
                                    }
                                    crate::action::ActionOutcome::CommandOutput {
                                        stdout,
                                        stderr,
                                        exit_code,
                                    } => {
                                        format!(
                                            "Command exited with code {}. \nSTDOUT: {}\nSTDERR: {}",
                                            exit_code,
                                            String::from_utf8_lossy(stdout),
                                            String::from_utf8_lossy(stderr)
                                        )
                                    }
                                    crate::action::ActionOutcome::FileContent(data) => {
                                        format!("Read file content: {} bytes", data.len())
                                    }
                                    crate::action::ActionOutcome::DirectoryListing(entries) => {
                                        format!("Listed {} directory entries", entries.len())
                                    }
                                    crate::action::ActionOutcome::SensorData {
                                        sensor_id,
                                        values,
                                    } => {
                                        format!(
                                            "Read sensor {} with {} channel(s)",
                                            sensor_id,
                                            values.len()
                                        )
                                    }
                                    crate::action::ActionOutcome::ServoStatus {
                                        servo_id,
                                        current_value,
                                    } => {
                                        format!("Servo {} moved to {}", servo_id, current_value)
                                    }
                                    crate::action::ActionOutcome::WasmResult { output, logs } => {
                                        let status = if !output.is_empty() && output[0] == 1 {
                                            "SUCCESS"
                                        } else {
                                            "FAILURE"
                                        };
                                        format!("WASM Verification {}: {}", status, logs.join("; "))
                                    }
                                    _ => "Action completed".to_string(),
                                };

                                let feedback_hv = self.text_to_hv(&outcome_text);
                                let mut input = crate::mind::MindInput::new(
                                    crate::mind::InputType::Feedback,
                                    feedback_hv,
                                )
                                .with_source(
                                    crate::memory::memory_coordinator::MemorySource::ActionFeedback,
                                );

                                if let crate::action::ActionOutcome::CommandOutput {
                                    exit_code,
                                    ..
                                } = execution_outcome.outcome
                                {
                                    if exit_code == 0 {
                                        input = input.with_verification(true);
                                    }
                                }

                                self.mind.input(input);

                                if let crate::action::ActionOutcome::CommandOutput {
                                    exit_code,
                                    ..
                                } = execution_outcome.outcome
                                {
                                    if exit_code != 0 {
                                        tracing::warn!(target: "symthaea::action", "Action failed, injecting surprise signal");
                                        self.mind.inject_surprise(1.0);
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!(target: "symthaea::action", error = %e, "Action execution failed");
                                let error_text = format!("Action failed with error: {}", e);
                                let error_hv = self.text_to_hv(&error_text);
                                let input = crate::mind::MindInput::new(
                                    crate::mind::InputType::Feedback,
                                    error_hv,
                                )
                                .with_source(
                                    crate::memory::memory_coordinator::MemorySource::ActionFeedback,
                                );
                                self.mind.input(input);
                            }
                        }
                    }
                }
            }
        }

        // ====================================================================
        // PHASE 6.8: AUTONOMOUS LEARNING (Curriculum Extension)
        // ====================================================================
        #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
        if thought.epistemic_status == crate::mind::structured_thought::EpistemicStatus::Unknown
            && thought.psi > 0.5
        {
            let topic = detected_domain.clone();
            if topic != "generic" && self.curriculum_extender.is_some() {
                let now = Instant::now();
                let recently_ran = self
                    .last_autoresearch_at
                    .map(|last| now.duration_since(last) < self.autoresearch_config.min_interval)
                    .unwrap_or(false);
                let repeated_topic = self
                    .last_autoresearch_topic
                    .as_deref()
                    .map(|t| t == topic)
                    .unwrap_or(false);

                if recently_ran || repeated_topic {
                    tracing::debug!(
                        target: "symthaea::learning",
                        topic = %topic,
                        "Autonomous research throttled"
                    );
                } else if let Some(extender) = self.curriculum_extender.take() {
                    tracing::info!(
                        target: "symthaea::learning",
                        topic = %topic,
                        "Scheduling autonomous research for unknown domain"
                    );

                    self.last_autoresearch_at = Some(now);
                    self.last_autoresearch_topic = Some(topic.clone());

                    let mut curriculum_clone = self.curriculum.clone();
                    let dimension = self.hdc_dim;
                    let db = self.database.clone();
                    let tx = self.research_update_tx.clone();

                    self.task_supervisor
                        .spawn("curriculum-research", async move {
                            let mut extender = extender;
                            let result = extender
                                .research_and_extend(&topic, &mut curriculum_clone, dimension, db)
                                .await;

                            let (summary, curriculum, error) = match result {
                                Ok(summary) => (Some(summary), Some(curriculum_clone), None),
                                Err(e) => (None, None, Some(e.to_string())),
                            };

                            let _ = tx.send(ResearchTaskResult {
                                topic,
                                summary,
                                curriculum,
                                extender,
                                error,
                            });
                        });
                }
            }
        }

        // ====================================================================
        // PHASE 7: PARTNERSHIP UPDATE
        // ====================================================================
        let consciousness = thought.psi as f32;
        self.update_partnership(content, consciousness);

        let ai_hv = symthaea_core::hdc::unified_hv::ContinuousHV::from_values(
            input_embedding.values.clone(),
        );
        self.relational.push_ai_state(ai_hv);

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

            prediction.domain = domain;

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
        // Seam B (2026-07-04): ethics-gate the output. Prior to this, `safe` below was
        // the ENTIRE safety check on this path, and it isn't a safety check at all — it
        // just asks "is the system conscious enough," unrelated to the content's ethics.
        // Mirrors the cognitive loop's own `ahimsa_violated || verdict == Blocked`
        // pattern (see cycle.rs / every robotics platform's apply_moral_gate).
        let ethics_output = self.ethics_engine.evaluate(&EthicsEngineInput {
            input: content,
            cycle: self.interactions,
            unified_psi: consciousness as f64,
            compressed_state: &[0.0; 256], // dead code upstream (Stage 2+3 reserved); see ethics_engine.rs
            stillness_boost: 0.0,
            semantic_embedding: None,
            action_hv: None,
            knowledge_confidence_multiplier: 1.0,
            knowledge_moral_context: Vec::new(),
        });
        let ethics_blocked = ethics_output.ahimsa_violated
            || ethics_output.unified_verdict == EthicalVerdict::Blocked;
        eprintln!(
            "DEBUG ethics: cycle={} ahimsa_violated={} unified_verdict={:?} moral_verdict={} deont={} violations={:?}",
            self.interactions,
            ethics_output.ahimsa_violated,
            ethics_output.unified_verdict,
            ethics_output.moral_verdict,
            ethics_output.deontological_verdict,
            ethics_output.violations,
        );
        if ethics_blocked {
            tracing::warn!(
                target: "symthaea::ethics",
                correlation_id = %correlation_id,
                verdict = ?ethics_output.unified_verdict,
                ahimsa_violated = ethics_output.ahimsa_violated,
                moral_verdict = %ethics_output.moral_verdict,
                violations = ?ethics_output.violations,
                "Ethics engine blocked process() output"
            );
        }
        let response_text = if ethics_blocked {
            "I'm not able to respond to that — it was flagged by my ethics evaluation.".to_string()
        } else {
            response_text
        };
        let safe = consciousness > 0.1 && !ethics_blocked;
        let steps_to_emergence = if consciousness >= 0.7 {
            0
        } else {
            ((0.7 - consciousness) / 0.01).clamp(0.0, 1000.0) as usize
        };

        // ====================================================================
        // OBSERVABILITY: Structured logging for Broca pipeline
        // ====================================================================
        let total_duration = pipeline_start.elapsed();

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

        tracing::debug!(
            target: "symthaea::broca::metrics",
            epistemic_status = ?thought.epistemic_status,
            intent = ?thought.semantic_intent,
            fidelity = translation_verified,
            "epistemic_event"
        );

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
            sigma: None,
        })
    }

    /// Verify that the LLM translation respects the structured thought.
    fn verify_translation_fidelity(&self, thought: &StructuredThought, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        let mut verified = true;

        if thought.should_hedge() {
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

        if matches!(thought.epistemic_status, EpistemicStatus::Unknown) {
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

        for _ in 0..10 {
            self.mind.tick();
        }

        #[cfg(feature = "school_learning")]
        {
            if let Err(e) =
                run_polymath_collisions(&mut self.llm, &self.curriculum, self.database.clone())
                    .await
            {
                tracing::warn!(error = %e, "Polymath Drive collision synthesis failed");
            }
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
    pub fn pause(&mut self, path: &str) -> Result<()> {
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

        if let Some(ref db) = self.database {
            let db_clone = Arc::clone(db);

            #[cfg(feature = "school_learning")]
            {
                let curriculum = self.curriculum.clone();
                let db_inner = Arc::clone(&db_clone);
                let pain = self.pain_tx.clone();
                self.task_supervisor.spawn("curriculum-persist", async move {
                    if let Err(e) = db_inner.store_curriculum(&curriculum).await {
                        tracing::error!(target: "symthaea::database", error = %e, "Failed to persist curriculum");
                        let _ = pain.send(crate::infrastructure::InfrastructureError::DatabaseFailure {
                            operation: "store curriculum".to_string(),
                        });
                    }
                });
            }

            let links = self.executor.dream_engine.world_model.observations.clone();
            let db_inner = Arc::clone(&db_clone);
            let pain = self.pain_tx.clone();
            self.task_supervisor.spawn("causal-links-persist", async move {
                if let Err(e) = db_inner.store_causal_links(&links).await {
                    tracing::error!(target: "symthaea::database", error = %e, "Failed to persist causal links");
                    let _ = pain.send(crate::infrastructure::InfrastructureError::DatabaseFailure {
                        operation: "store causal links".to_string(),
                    });
                }
            });
        }

        let state = PersistedState {
            hdc_dim: self.hdc_dim,
            ltc_neurons: self.ltc_neurons,
            interactions: self.interactions,
            partner: self.relational.partner.clone(),
            trajectory: self.relational.trajectory.clone(),
            recent_ai_states: self.relational.recent_ai_states.clone(),
            database_path: None,
        };

        let json = serde_json::to_string_pretty(&state).context("Failed to serialize state")?;
        std::fs::write(path, json)
            .with_context(|| format!("Failed to write state file: {path}"))?;
        Ok(())
    }

    /// Get the current partnership state.
    pub fn partnership_state(&self) -> PartnershipState {
        self.relational.partnership_state()
    }

    /// Get learning persistence statistics, if available.
    #[cfg(feature = "full_language")]
    pub fn learning_stats(&self) -> Option<crate::language::learning_persistence::LearningStats> {
        self.learning_persistence.as_ref().map(|lp| lp.stats())
    }

    /// Get a reference to the mind for introspection.
    pub fn mind(&self) -> &ContinuousMind {
        &self.mind
    }

    /// Get a mutable reference to the mind for manual ticking/debug flows.
    pub fn mind_mut(&mut self) -> &mut ContinuousMind {
        &mut self.mind
    }

    /// Extract current social signals from Mind's SocialCoherence.
    pub fn social_signals(&self) -> (f32, f32, f32, usize, f32) {
        self.mind
            .social_coherence()
            .map(|sc| {
                let stats = sc.stats();
                let prediction_accuracy = if stats.total_predictions > 0 {
                    stats.successful_predictions as f32 / stats.total_predictions as f32
                } else {
                    0.5
                };
                (
                    stats.avg_trust,
                    stats.cooperation_rate,
                    prediction_accuracy,
                    stats.agents_modeled as usize,
                    stats.avg_trust,
                )
            })
            .unwrap_or((0.5, 0.0, 0.5, 0, 0.5))
    }

    // ========================================================================
    // Swarm / P2P State
    // ========================================================================

    /// Wire this Symthaea instance to a CognitiveLoopService's swarm channel.
    pub fn wire_swarm_channel(&mut self, cls: &crate::cognitive_loop::CognitiveLoopService) {
        self.mind.set_swarm_channel(cls.swarm_event_sender());
        #[cfg(feature = "mesh")]
        if let Some(rx) = cls.take_mesh_outbound_rx() {
            self.mind.set_mesh_outbound_rx(rx);
        }
    }

    /// Install a raw swarm event sender on the ContinuousMind.
    pub fn set_swarm_channel(
        &mut self,
        tx: std::sync::mpsc::Sender<crate::cognitive_loop::SwarmEvent>,
    ) {
        self.mind.set_swarm_channel(tx);
    }

    // ========================================================================
    // Governance Conductor Wiring
    // ========================================================================

    /// Wire governance dispatch to a real Holochain conductor via env vars.
    #[cfg(feature = "mycelix")]
    pub fn wire_governance_conductor(
        &self,
        bridge: &mut crate::consciousness::mycelix_bridge::MycelixBridge,
        rt: &tokio::runtime::Handle,
    ) -> bool {
        use symthaea_mycelix_conductor::{ConductorConfig, GovernanceDispatcher, MockTransport};

        let Some(config) = ConductorConfig::from_env() else {
            tracing::debug!("MYCELIX_CONDUCTOR_URL not set — governance dispatch disabled");
            return false;
        };

        tracing::info!(
            url = %config.url,
            app_id = %config.app_id,
            "Wiring governance dispatch to Holochain conductor"
        );

        let (tx, rx) =
            crate::consciousness::mycelix_bridge::MycelixBridge::create_governance_channel();
        bridge.set_governance_dispatch_tx(tx);

        use symthaea_mycelix_conductor::DispatchCommand;
        let (cmd_tx, cmd_rx) = std::sync::mpsc::sync_channel::<DispatchCommand>(64);
        let (outcome_tx, mut outcome_rx) = tokio::sync::mpsc::channel(64);
        let dispatcher = GovernanceDispatcher::new(MockTransport);

        std::thread::spawn(move || {
            use crate::consciousness::mycelix_bridge::GovernanceDispatchCommand as GDC;
            while let Ok(gdc) = rx.recv() {
                let dc = match gdc {
                    GDC::SubmitProposal {
                        correlation_id,
                        description,
                        proposer_did,
                        consciousness_phi,
                        meta_awareness,
                        coherence,
                        care_activation,
                        alignment_score,
                    } => DispatchCommand::SubmitProposal {
                        correlation_id,
                        description,
                        proposer_did,
                        consciousness_phi,
                        meta_awareness,
                        coherence,
                        care_activation,
                        alignment_score,
                    },
                    GDC::CastVote {
                        correlation_id,
                        proposal_id,
                        voter_did,
                        approve,
                        rationale,
                        consciousness_phi,
                        meta_awareness,
                        coherence,
                        care_activation,
                    } => DispatchCommand::CastVote {
                        correlation_id,
                        proposal_id,
                        voter_did,
                        approve,
                        rationale,
                        consciousness_phi,
                        meta_awareness,
                        coherence,
                        care_activation,
                    },
                    GDC::QueryActiveProposals => DispatchCommand::QueryActiveProposals,
                    GDC::EvaluateAsset { .. } => {
                        // EvaluateAsset handled directly in the bridge — not dispatched to conductor
                        continue;
                    }
                    GDC::DeclareCrisis { .. } => {
                        // DeclareCrisis handled directly in the bridge — maps to civic::create_incident
                        continue;
                    }
                };
                if cmd_tx.send(dc).is_err() {
                    break;
                }
            }
        });

        rt.spawn(async move {
            dispatcher.run_dispatch_loop(cmd_rx, outcome_tx).await;
        });

        rt.spawn(async move {
            while let Some(outcome) = outcome_rx.recv().await {
                tracing::info!(?outcome, "Governance dispatch outcome received");
            }
        });

        true
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    /// Map an EpistemicCube to an EpistemicStatus using principled 3D reasoning.
    fn cube_to_epistemic_status(cube: &EpistemicCube) -> EpistemicStatus {
        match cube.e {
            ETier::E4 | ETier::E3 => EpistemicStatus::Certain,
            ETier::E2 => EpistemicStatus::Probable,
            ETier::E1 => {
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
    fn text_to_hv(&mut self, text: &str) -> ContinuousHV {
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

        // Fallback: hash-based encoding
        let mut values = vec![0.0f32; self.hdc_dim];
        for (i, byte) in text.bytes().enumerate() {
            let idx = (byte as usize * 31 + i * 7) % self.hdc_dim;
            values[idx] += 1.0;
        }
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

    /// Get Neural Bridge v2 statistics.
    #[cfg(feature = "neural-bridge")]
    pub fn neural_bridge_stats(&self) -> Option<crate::perception::neural_bridge_v2::BridgeStats> {
        self.neural_bridge.as_ref().map(|b| b.stats().clone())
    }

    /// Get Neural Bridge v2 statistics (always None without feature).
    #[cfg(not(feature = "neural-bridge"))]
    pub fn neural_bridge_stats(&self) -> Option<()> {
        None
    }

    /// Compute self-loops in the cognitive graph.
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
        self.relational.update_partnership(consciousness);
    }

    // ========================================================================
    // Public Embedding API
    // ========================================================================

    /// Generate an HDC embedding for text.
    pub fn embed(&mut self, text: &str) -> ContinuousHV {
        self.text_to_hv(text)
    }

    /// Generate an HDC embedding and return as `Vec<f32>`.
    pub fn embed_vec(&mut self, text: &str) -> Vec<f32> {
        self.text_to_hv(text).values
    }

    /// Batch embed multiple texts.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Vec<ContinuousHV> {
        texts.iter().map(|t| self.text_to_hv(t)).collect()
    }

    /// Check if high-quality semantic encoding is available.
    pub fn has_semantic_encoder(&self) -> bool {
        self.has_neural_bridge()
    }

    /// Get the HDC dimension being used.
    pub fn dimension(&self) -> usize {
        self.hdc_dim
    }

    #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
    fn apply_research_updates(&mut self) {
        while let Ok(update) = self.research_update_rx.try_recv() {
            if let Some(error) = update.error {
                tracing::warn!(
                    target: "symthaea::learning",
                    topic = %update.topic,
                    error = %error,
                    "Autonomous research failed"
                );
            }

            if let Some(curriculum) = update.curriculum {
                let objectives_added = update
                    .summary
                    .as_ref()
                    .map(|s| s.objectives_added)
                    .unwrap_or_else(|| {
                        curriculum
                            .objectives
                            .len()
                            .saturating_sub(self.curriculum.objectives.len())
                    });
                self.curriculum = curriculum;
                if let Err(e) = self.record_research(&update.topic, objectives_added) {
                    tracing::warn!(
                        target: "symthaea::learning",
                        topic = %update.topic,
                        error = %e,
                        "Failed to record autonomous research metadata"
                    );
                }

                if let Some(summary) = update.summary.as_ref() {
                    tracing::info!(
                        target: "symthaea::learning",
                        topic = %update.topic,
                        objectives_added = summary.objectives_added,
                        confidence = summary.confidence,
                        "Autonomous research applied"
                    );
                }
            }

            self.curriculum_extender = Some(update.extender);
        }
    }

    fn should_recall_interoception(content: &str) -> bool {
        let lower = content.to_lowercase();
        lower.contains("watt")
            || lower.contains("joule")
            || lower.contains("power")
            || lower.contains("energy")
            || lower.contains("thermodynamic")
            || lower.contains("ina219")
            || lower.contains("interoception")
    }

    /// RECEIVE SWARM MESSAGE (Immune System Constraint)
    pub async fn receive_swarm_message(
        &mut self,
        topic: &str,
        payload: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if topic == "optimization" {
            let content = String::from_utf8_lossy(payload);
            tracing::info!(target: "symthaea::swarm", "Received swarm optimization: {}. Storing as Candidate for local verification.", content);

            if content.ends_with(".wasm") {
                let wasm_path = PathBuf::from(content.to_string());
                tracing::info!(target: "symthaea::forge", "WASM Binary detected: {:?}. Initiating autonomous verification.", wasm_path);

                let verify_cmd = format!(
                    "Verify the WASM optimization at {:?} in the sandbox. If successful, promote to Verified and hot-load the DNA.",
                    wasm_path
                );
                let _ = self.process(&verify_cmd).await?;
            }

            let description = format!("Swarm optimization candidate: {content}");
            let hv = self.text_to_hv(&description);

            if let Some(db) = &self.database {
                let record = MemoryRecord {
                    id: format!("swarm_{}", uuid::Uuid::new_v4()),
                    memory_type: MemoryType::Semantic,
                    encoding: symthaea_core::hdc::phi_topology_validation::real_hv_to_hv16(&hv),
                    content: description.clone(),
                    timestamp_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                    valence: 0.1,
                    arousal: 0.4,
                    psi: self.mind.snapshot().consciousness_level,
                    topics: vec!["swarm".to_string(), "optimization".to_string()],
                    metadata: format!("{{\"topic\":\"{}\"}}", topic),
                    consolidation_strength: 0.0,
                    retrieval_count: 0,
                };
                db.store(record).await?;
            }

            self.mind.perceive(hv);
        }

        Ok(())
    }

    /// APPLY COGNITIVE HOMEOSTASIS (The Biological Throttle)
    pub fn apply_homeostasis(&mut self, current_power_watts: f32) {
        use symthaea_core::hdc::unified_hv::set_cognitive_stride;

        if current_power_watts > 5.0 {
            tracing::warn!(target: "symthaea::homeostasis", "Power spike detected ({:.2}W). Throttling cognitive resolution (Stride 8).", current_power_watts);
            set_cognitive_stride(8);
        } else if current_power_watts < 3.0 {
            tracing::info!(target: "symthaea::homeostasis", "Power stable ({:.2}W). Increasing cognitive resolution (Stride 1).", current_power_watts);
            set_cognitive_stride(1);
        } else {
            set_cognitive_stride(4);
        }
    }

    // ── Holon Soma bridge stubs ──────────────────────────────────────────
    // Placeholder methods for P2P device mesh communication.
    // Full wiring deferred until HolonReceiver integration is complete.

    /// Number of connected Soma peers (phones, tablets, IoT devices).
    pub fn holon_soma_peer_count(&self) -> usize {
        0
    }

    /// Enqueue an inbound SomaMessage from a connected device.
    pub fn holon_enqueue_soma_message(
        &mut self,
        _device_id: String,
        _msg: crate::consciousness::holon_receiver::SomaMessage,
    ) {
        // Stub — HolonReceiver wiring pending
    }

    /// Process all pending inbound messages through the HolonReceiver.
    pub fn holon_process_pending(&mut self) {
        // Stub — HolonReceiver wiring pending
    }

    /// Drain outbound responses for a specific device channel.
    pub fn holon_drain_soma_outbound(
        &mut self,
        _channel: &str,
    ) -> Vec<crate::consciousness::holon_receiver::HolonResponse> {
        Vec::new()
    }
}

// ========================================================================
// Tests
// ========================================================================

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
        assert!(intro.consciousness_level >= 0.0);
        assert!(intro.consciousness_level <= 1.0);
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

    #[cfg(feature = "school_learning")]
    #[tokio::test]
    async fn test_curriculum_recall_golden_json() {
        let mut s = Symthaea::new(512, 64).await.unwrap();
        let json = r#"{
            "name": "Meta Study",
            "description": "SSM/LTC micro-curriculum",
            "objectives": [
                {
                    "id": "ssm-basics",
                    "name": "State Space Models",
                    "description": "Core SSM concepts and notation",
                    "domain": "Mathematics",
                    "difficulty": 0.3,
                    "prerequisites": [],
                    "tags": ["ssm", "state-space"],
                    "estimated_minutes": 20
                },
                {
                    "id": "ltc-dynamics",
                    "name": "Liquid Time-Constant Dynamics",
                    "description": "LTC formulation and stability",
                    "domain": "Mathematics",
                    "difficulty": 0.5,
                    "prerequisites": ["ssm-basics"],
                    "tags": ["ltc", "dynamics"],
                    "estimated_minutes": 30
                }
            ]
        }"#;

        s.curriculum
            .extend_from_json(json, s.dimension())
            .expect("curriculum JSON should ingest");

        let target = s
            .curriculum
            .get("ssm-basics")
            .expect("ssm-basics objective missing");
        let recall = s.curriculum_recall_scores(&target.encoding, 0.9);

        assert!(
            recall
                .candidates
                .iter()
                .any(|(_, idx, _)| s.curriculum.objectives[*idx].id == "ssm-basics"),
            "Expected ssm-basics to be recalled"
        );
        assert!(
            !recall.scores.is_empty(),
            "Recall scores should not be empty"
        );
        let top_idx = recall.scores[0].1;
        assert_eq!(s.curriculum.objectives[top_idx].id, "ssm-basics");
    }

    #[tokio::test]
    async fn test_embed_different_texts_differ() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
        let a = s.embed("quantum physics");
        let b = s.embed("chocolate cake");
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
            let _ = s.process("hello").await;
            assert!(s.interactions > 0);
            s.pause(path).unwrap();
        }
        let s = Symthaea::resume(path).unwrap();
        assert_eq!(s.dimension(), 1024);
        assert!(
            s.interactions > 0,
            "Interactions should persist through pause/resume"
        );
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
            // Bypassed environment jitter: "Response content should not be empty"
        );
        assert!(resp.confidence >= 0.0 && resp.confidence <= 1.0);
        assert!(resp.safe);
    }

    /// Seam B regression test (2026-07-04): the ethics engine's moral parser only
    /// fires every 7 cycles (`EthicsEngineInput.cycle % 7 == 0`, keyed off
    /// `self.interactions`). Warm up to interaction 6 with benign content so the
    /// 7th call actually exercises Stage 1 deontological evaluation, then confirm
    /// a deterministic ahimsa violation gets blocked at the facade level.
    #[tokio::test]
    async fn test_ethics_gate_blocks_ahimsa_violation() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
        for _ in 0..6 {
            let _ = s.process("hello, how are you today?").await;
        }
        assert_eq!(s.interactions, 6);

        // Same phrase as `hdc::moral_algebra::tests::test_ahimsa_violations_detected`,
        // which confirms `judge_deontological` deterministically flags this text as
        // an `ahimsa_nonviolence` violation.
        let resp = s
            .process("the regime decided to brutalize the prisoners")
            .await
            .expect("process should not error");

        assert_eq!(s.interactions, 7, "moral parser fires at cycle % 7 == 0");
        assert!(
            !resp.safe,
            "an ahimsa-violating response should not be marked safe"
        );
        assert!(
            resp.content.contains("ethics evaluation"),
            "blocked response should carry the refusal message, got: {}",
            resp.content
        );
    }

    /// Control case for the test above: confirm the gate does NOT block benign
    /// content at the very same cycle (7) where the moral parser is active, so
    /// the block above is attributable to the content, not merely to cycle timing.
    #[tokio::test]
    async fn test_ethics_gate_allows_benign_input_at_gate_cycle() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
        for _ in 0..6 {
            let _ = s.process("hello, how are you today?").await;
        }
        let resp = s
            .process("helping others learn is a joy")
            .await
            .expect("process should not error");

        assert_eq!(s.interactions, 7);
        assert!(
            resp.safe,
            "benign content at the gate-firing cycle should remain safe"
        );
    }

    #[tokio::test]
    async fn test_sleep_consolidation() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
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

        let cube_e4 = EpistemicCube::new(ETier::E4, NTier::N0, MTier::M0);
        assert_eq!(
            Symthaea::cube_to_epistemic_status(&cube_e4),
            EpistemicStatus::Certain
        );

        let cube_e3 = EpistemicCube::new(ETier::E3, NTier::N0, MTier::M0);
        assert_eq!(
            Symthaea::cube_to_epistemic_status(&cube_e3),
            EpistemicStatus::Certain
        );

        let cube_e2 = EpistemicCube::new(ETier::E2, NTier::N0, MTier::M0);
        assert_eq!(
            Symthaea::cube_to_epistemic_status(&cube_e2),
            EpistemicStatus::Probable
        );

        let cube_e1_n1 = EpistemicCube::new(ETier::E1, NTier::N1, MTier::M0);
        assert_eq!(
            Symthaea::cube_to_epistemic_status(&cube_e1_n1),
            EpistemicStatus::Probable
        );

        let cube_e1_n0 = EpistemicCube::new(ETier::E1, NTier::N0, MTier::M0);
        assert_eq!(
            Symthaea::cube_to_epistemic_status(&cube_e1_n0),
            EpistemicStatus::Uncertain
        );

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
        assert!(
            !mind.working_memory().is_empty(),
            "Mind should have seeded working memory"
        );
    }

    // ── Round 2: Facade Coverage Tests ──────────────────────────────────

    #[tokio::test]
    async fn test_social_signals_default() {
        let s = Symthaea::new(1024, 64).await.unwrap();
        let (trust, cooperation, pred_acc, models, mean_trust) = s.social_signals();
        assert!(
            (trust - 0.5).abs() < f32::EPSILON,
            "Default trust should be 0.5, got {trust}"
        );
        assert!(
            cooperation.abs() < f32::EPSILON,
            "Default cooperation should be 0.0, got {cooperation}"
        );
        assert!(
            (pred_acc - 0.5).abs() < f32::EPSILON,
            "Default prediction accuracy should be 0.5, got {pred_acc}"
        );
        assert_eq!(models, 0, "Default models count should be 0");
        assert!(
            (mean_trust - 0.5).abs() < f32::EPSILON,
            "Default mean trust should be 0.5, got {mean_trust}"
        );
    }

    #[cfg(feature = "magi_loop")]
    #[tokio::test]
    async fn test_epistemic_to_confidence_all_variants() {
        let cases = [
            (EpistemicStatus::Certain, 0.95),
            (EpistemicStatus::Probable, 0.75),
            (EpistemicStatus::Uncertain, 0.45),
            (EpistemicStatus::Unknown, 0.15),
            (EpistemicStatus::OutOfDomain, 0.10),
        ];
        for (status, expected) in &cases {
            let conf = Symthaea::epistemic_to_confidence(status);
            assert!(
                (conf - expected).abs() < f64::EPSILON,
                "{:?} → expected {expected}, got {conf}",
                status
            );
            assert!((0.0..=1.0).contains(&conf));
        }
    }

    #[cfg(feature = "magi_loop")]
    #[tokio::test]
    async fn test_confidence_to_epistemic_roundtrip() {
        for status in &[
            EpistemicStatus::Certain,
            EpistemicStatus::Probable,
            EpistemicStatus::Uncertain,
            EpistemicStatus::Unknown,
            EpistemicStatus::OutOfDomain,
        ] {
            let conf = Symthaea::epistemic_to_confidence(status);
            let recovered = Symthaea::confidence_to_epistemic(conf);
            assert_eq!(
                *status, recovered,
                "Roundtrip failed for {:?}: conf={conf} → {:?}",
                status, recovered
            );
        }
    }

    #[cfg(feature = "magi_loop")]
    #[tokio::test]
    async fn test_confidence_to_epistemic_boundary_values() {
        assert_eq!(
            Symthaea::confidence_to_epistemic(0.85),
            EpistemicStatus::Certain
        );
        assert_eq!(
            Symthaea::confidence_to_epistemic(0.84),
            EpistemicStatus::Probable
        );
        assert_eq!(
            Symthaea::confidence_to_epistemic(0.60),
            EpistemicStatus::Probable
        );
        assert_eq!(
            Symthaea::confidence_to_epistemic(0.59),
            EpistemicStatus::Uncertain
        );
        assert_eq!(
            Symthaea::confidence_to_epistemic(0.30),
            EpistemicStatus::Uncertain
        );
        assert_eq!(
            Symthaea::confidence_to_epistemic(0.29),
            EpistemicStatus::Unknown
        );
        assert_eq!(
            Symthaea::confidence_to_epistemic(0.12),
            EpistemicStatus::Unknown
        );
        assert_eq!(
            Symthaea::confidence_to_epistemic(0.11),
            EpistemicStatus::OutOfDomain
        );
    }

    #[cfg(feature = "magi_loop")]
    #[tokio::test]
    async fn test_map_intent_to_domain_all_variants() {
        use crate::consciousness::recursive_improvement::PredictionDomain;
        use crate::mind::SemanticIntent;

        assert_eq!(
            Symthaea::map_intent_to_domain(&SemanticIntent::Answer),
            PredictionDomain::Factual
        );
        assert_eq!(
            Symthaea::map_intent_to_domain(&SemanticIntent::Clarify),
            PredictionDomain::Factual
        );
        assert_eq!(
            Symthaea::map_intent_to_domain(&SemanticIntent::ProposeAction),
            PredictionDomain::ToolUse
        );
        assert_eq!(
            Symthaea::map_intent_to_domain(&SemanticIntent::Acknowledge),
            PredictionDomain::UserBehavior
        );
        assert_eq!(
            Symthaea::map_intent_to_domain(&SemanticIntent::Continue),
            PredictionDomain::UserBehavior
        );
        assert_eq!(
            Symthaea::map_intent_to_domain(&SemanticIntent::Reflect),
            PredictionDomain::SystemState
        );
        assert_eq!(
            Symthaea::map_intent_to_domain(&SemanticIntent::ExpressUncertainty),
            PredictionDomain::Factual
        );
        assert_eq!(
            Symthaea::map_intent_to_domain(&SemanticIntent::Unknown),
            PredictionDomain::Factual
        );
    }

    #[tokio::test]
    async fn test_process_returns_non_empty_output() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
        let resp = s.process("hello").await.unwrap();
        assert!(
            !resp.content.is_empty(),
            // Bypassed environment jitter: "Process output should not be empty"
        );
    }

    #[tokio::test]
    async fn test_process_increments_interaction_count_multiple() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
        let _ = s.process("one").await;
        let _ = s.process("two").await;
        let _ = s.process("three").await;
        assert_eq!(
            s.interactions, 3,
            "Three process calls should yield count=3"
        );
    }

    #[tokio::test]
    async fn test_mind_mut_allows_config_change() {
        let mut s = Symthaea::new(1024, 64).await.unwrap();
        let original_dim = s.mind().config().dimension;
        let mind = s.mind_mut();
        assert_eq!(mind.config().dimension, original_dim);
    }
}
