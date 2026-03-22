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
#[cfg(feature = "school_learning")]
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(all(feature = "web_research_module", feature = "school_learning"))]
use std::time::{Duration, Instant};
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

pub use crate::action::bindings::ActionRegistry;
use crate::action::SimpleExecutor;
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

#[cfg(feature = "school_learning")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumObjectiveSummary {
    pub id: String,
    pub name: String,
    pub domain: String,
    pub difficulty: String,
    pub estimated_minutes: u32,
    pub tags: Vec<String>,
    pub description: String,
}

#[cfg(feature = "school_learning")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumReport {
    pub curriculum_id: String,
    pub curriculum_name: String,
    pub total_objectives: usize,
    pub dimension: usize,
    pub last_research_topic: Option<String>,
    pub last_research_at: Option<String>,
    pub last_saved_at: Option<String>,
    pub last_objectives_added: Option<usize>,
    pub recent_objectives: Vec<CurriculumObjectiveSummary>,
}

/// Relational consciousness subsystem — partnership, trajectory, and dyadic Phi.
///
/// Groups all relational state into a cohesive unit: partner model tracking,
/// relationship trajectory, Phi-dyad computation, and recent AI states for
/// dyadic assessment.
struct RelationalCore {
    /// Human partner model for relational consciousness.
    partner: HumanPartnerModel,
    /// Relationship trajectory tracking.
    trajectory: RelationshipTrajectory,
    /// Phi-dyad calculator for relational Phi.
    dyad_calculator: PhiDyadCalculator,
    /// Recent AI states for dyad computation (ring buffer, max 8).
    recent_ai_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
    /// Last computed Phi_dyad — fed back into mind as relational Psi on next cycle.
    last_phi_dyad: f64,
}

impl RelationalCore {
    fn new() -> Self {
        Self {
            partner: HumanPartnerModel::new("human"),
            trajectory: RelationshipTrajectory::default(),
            dyad_calculator: PhiDyadCalculator::new(),
            recent_ai_states: Vec::new(),
            last_phi_dyad: 0.0,
        }
    }

    fn from_persisted(
        partner: HumanPartnerModel,
        trajectory: RelationshipTrajectory,
        recent_ai_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV>,
    ) -> Self {
        Self {
            partner,
            trajectory,
            dyad_calculator: PhiDyadCalculator::new(),
            recent_ai_states,
            last_phi_dyad: 0.0,
        }
    }

    /// Push an AI state into the ring buffer (max 8).
    fn push_ai_state(&mut self, hv: symthaea_core::hdc::unified_hv::ContinuousHV) {
        self.recent_ai_states.push(hv);
        if self.recent_ai_states.len() > 8 {
            self.recent_ai_states.remove(0);
        }
    }

    /// Compute Phi-dyad from recent AI states and partner model.
    fn compute_phi_dyad(&self) -> f64 {
        if self.recent_ai_states.is_empty() {
            return 0.0;
        }

        let human_states: Vec<symthaea_core::hdc::unified_hv::ContinuousHV> = self
            .recent_ai_states
            .iter()
            .map(|s| {
                let mut vals = s.values.clone();
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

    /// Update partnership state from interaction consciousness level.
    fn update_partnership(&mut self, consciousness: f32) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

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

        let phi_dyad = self.compute_phi_dyad();
        self.trajectory.record(now, self.partner.stage, phi_dyad);
        self.last_phi_dyad = phi_dyad;
    }

    /// Get current partnership state summary.
    fn partnership_state(&self) -> PartnershipState {
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
}

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
    /// Somatic error bridge: drains infrastructure errors → felt stress.
    somatic_bridge: SomaticErrorBridge,
    /// Pain channel sender: cloned into TaskSupervisor and database operations.
    pain_tx: PainSender,
    /// Task supervisor: wraps all tokio::spawn calls for panic detection.
    task_supervisor: TaskSupervisor,

    // ── Holon Bridge ────────────────────────────────────────────────────
    /// Desktop-side receiver for Soma mobile connections.
    /// Stores inbound messages from HTTP `/holon/outbound` and queues
    /// responses for `/holon/inbound`. Processed by `holon_process_pending()`.
    holon_receiver: crate::consciousness::holon_receiver::HolonReceiver,
}

#[cfg(feature = "school_learning")]
#[derive(Debug, Clone, Copy)]
struct CurriculumRecallConfig {
    threshold: f32,
    max_recall: usize,
    log_top_k: usize,
    budget: f32,
}

#[cfg(feature = "school_learning")]
impl CurriculumRecallConfig {
    fn from_env() -> Self {
        let threshold = std::env::var("SYMTHAEA_CURRICULUM_RECALL_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.65)
            .clamp(0.0, 1.0);
        let max_recall = std::env::var("SYMTHAEA_CURRICULUM_RECALL_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(6)
            .max(1);
        let log_top_k = std::env::var("SYMTHAEA_CURRICULUM_RECALL_LOG_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3);
        let budget = std::env::var("SYMTHAEA_CURRICULUM_RECALL_BUDGET")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(max_recall as f32)
            .max(0.0);

        Self {
            threshold,
            max_recall,
            log_top_k,
            budget,
        }
    }
}

#[cfg(feature = "school_learning")]
#[derive(Debug, Clone)]
struct CurriculumPersistenceConfig {
    path: PathBuf,
    auto_save: bool,
}

#[cfg(feature = "school_learning")]
impl CurriculumPersistenceConfig {
    fn from_env() -> Self {
        let path = std::env::var("SYMTHAEA_CURRICULUM_PATH")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(default_curriculum_path);

        let auto_save = std::env::var("SYMTHAEA_CURRICULUM_AUTO_SAVE")
            .ok()
            .and_then(|v| parse_env_bool(&v))
            .unwrap_or(true);

        Self { path, auto_save }
    }
}

#[cfg(feature = "school_learning")]
fn default_curriculum_path() -> PathBuf {
    dirs::data_local_dir()
        .or_else(dirs::state_dir)
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("symthaea")
        .join("curriculum.json")
}

#[cfg(feature = "school_learning")]
fn parse_env_bool(value: &str) -> Option<bool> {
    match value.trim().to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

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

#[cfg(feature = "school_learning")]
struct CurriculumRecallScores {
    scores: Vec<(f32, usize)>,
    candidates: Vec<(f32, usize, ContinuousHV)>,
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

#[cfg(feature = "school_learning")]
fn load_curriculum_from_store(
    hdc_dim: usize,
    persistence: &CurriculumPersistenceConfig,
) -> (Curriculum, CurriculumMeta) {
    match CurriculumLoader::load_store_from_file_with_dimension(&persistence.path, hdc_dim) {
        Ok((curriculum, meta)) => (curriculum, meta),
        Err(LoadError::FileNotFound(_)) => (
            Curriculum::new("symthaea", "Main Curriculum").build(),
            CurriculumMeta::new(hdc_dim),
        ),
        Err(err) => {
            tracing::warn!(
                target: "symthaea::curriculum",
                error = %err,
                path = %persistence.path.display(),
                "Failed to load persisted curriculum, falling back to default"
            );
            (
                Curriculum::new("symthaea", "Main Curriculum").build(),
                CurriculumMeta::new(hdc_dim),
            )
        }
    }
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
            holon_receiver: crate::consciousness::holon_receiver::HolonReceiver::new(),
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
            holon_receiver: crate::consciousness::holon_receiver::HolonReceiver::new(),
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

        #[cfg(all(feature = "web_research_module", feature = "school_learning"))]
        self.apply_research_updates();

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
        // Infrastructure errors (task panics, DB failures, lock poisons) accumulate
        // in the pain channel between cycles. Drain them now and convert to felt
        // signals that modulate cognition: high stress → increased thermodynamic load,
        // arousal spikes, and tau slowdown for more cautious processing.
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

        // Feed relational Ψ from previous cycle's Φ_dyad into the mind.
        // This closes the feedback loop: partnership quality → consciousness boost.
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
            thought.primitives = understanding.primitives;
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
        // PHASE 3.6: CODE CONTEXT INJECTION (CfC-planned code generation)
        // ====================================================================
        // When the domain is "programming", run the CodeGenerator to produce
        // CfC-planned code structure with HDC-verified intent similarity and
        // Phi measurement, then inject into the structured thought so the LLM
        // translation phase receives a fully-planned code context.
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

            // Extract language from domain entities or default to "rust"
            let lang = domain_entities
                .iter()
                .find(|e| e.entity_type == "language")
                .map(|e| e.value.clone())
                .unwrap_or_else(|| "rust".to_string());

            // Extract function name, entity kind, and signature from NL input.
            let (func_name, entity_kind, inferred_sig) =
                Self::extract_code_metadata(content, &lang);

            // Build a CodeIntent::Create for generation tasks; for other
            // categories, populate code_context with plan steps only.
            let content_lower = content.to_lowercase();
            let intent = match category {
                CodeIntentCategory::Create => {
                    let target =
                        CodeTarget::new(&func_name, entity_kind).with_language(lang.clone());
                    let mut spec = CodeSpec::new(&lang, &func_name, content);
                    if let Some(ref sig) = inferred_sig {
                        spec = spec.with_signature(sig.as_str());
                    }
                    // Detect multi-entity patterns: "struct with method(s)"
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
                    // Item 3 (Phase 3h): Detect algorithm patterns and inject as
                    // constraints so the emitter produces scaffolding, not todo!()
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

            // Retrieve top-3 most similar past examples via HDC similarity
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

            // Item 1 (Phase 3h): Test-first generation — produce tests BEFORE
            // implementation so they serve as independent behavioral oracle
            pregenerated_tests =
                if let crate::language::code_intent::CodeIntent::Create { ref spec, .. } = intent {
                    self.code_generator.generate_tests_only(spec)
                } else {
                    None
                };

            let generated = self.code_generator.generate(&intent, &gen_ctx);

            // Extract spec fields for the structured thought
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

            // Detect if the native emitter left unresolved placeholders
            let needs_llm = generated.source.contains("todo!(")
                || generated.source.contains("NotImplementedError");

            // Item 6 (Phase 3i): Structured prompt assembly for LLM completion.
            // Organize notes into clear sections so the LLM gets a well-formed prompt.
            let mut notes = generated.notes.clone();

            if needs_llm {
                // Section: CONSTRAINTS from spec + algorithm detection
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

                // Section: ERROR_AVOIDANCE from past failures
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

                // Section: SIMILAR_EXAMPLE — best HDC match from cache
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

                // Section: EXPECTED_TESTS — behavioral oracle from test-first generation
                if let Some(ref tests) = pregenerated_tests {
                    notes.push(format!(
                        "EXPECTED_TESTS: The generated code MUST pass these tests:\n{}",
                        tests
                    ));
                }

                // Section: OUTPUT_FORMAT — clear instructions for the LLM
                notes.push(
                    "OUTPUT_FORMAT: Replace each todo!() body with a working implementation. \
                     Do NOT change the function signature. Do NOT add extra functions or imports \
                     unless necessary. Keep the code minimal and correct."
                        .to_string(),
                );
            } else {
                // Non-LLM path: just inject error hints as flat notes
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
                syntactically_valid: None, // Set in Phase 5.5
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

            // Plan coverage metric: log plan gap for FEP learning signal
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

            // SSM distillation: cache high-quality native generations as training targets
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
        let mood_temp = self.mind.state.mood_temperature;
        let generation = self.llm.translate_thought(&thought, mood_temp).await;
        let phase5_duration = phase5_start.elapsed();

        // Inject L-SSM semantic PE into MindState for downstream telemetry
        // and pass FEP-proxy signal to modulate distillation LR
        #[cfg(feature = "liquid-mamba")]
        {
            let pe = self.llm.last_liquid_mamba_pe();
            self.mind.state.liquid_mamba_pe = pe;
            self.mind.state.liquid_mamba_lr = self.llm.current_distillation_lr();
            self.mind.state.liquid_mamba_rank = self.llm.last_effective_rank();
            self.mind.state.liquid_mamba_generation_count = self.llm.generation_count();
            // FEP proxy: high cognitive load → high surprise → boost distillation
            // consciousness_level provides an integration quality signal
            let fep_proxy = (self.mind.state.cognitive_load as f32)
                .max(1.0 - self.mind.state.consciousness_level as f32)
                .clamp(0.0, 1.0);
            self.llm.set_fep_modulation(fep_proxy);

            // Cycle-level distillation modulation: adjusts FEP factor based on
            // thermodynamic load, consciousness confidence, and FEP precision
            let thermo_load = self.mind.state.thermodynamic_load;
            let confidence = self.mind.state.consciousness_level as f32;
            self.llm
                .cycle_level_distill(fep_proxy, thermo_load, confidence, 1.0);
        }

        // ====================================================================
        // PHASE 5.5: CODE VERIFICATION (Tree-sitter + HDC round-trip)
        // ====================================================================
        // When code was generated, verify syntax via tree-sitter and re-encode
        // to HDC space to measure semantic fidelity. On failure, retry once
        // with error feedback injected into the thought context.
        #[cfg(feature = "code_generation")]
        let mut generation = generation; // shadow as mutable for retry
        #[cfg(feature = "code_generation")]
        if thought.code_context.is_some() {
            let code_block = Self::extract_code_block(&generation.text);
            let lang = thought
                .code_context
                .as_ref()
                .map(|c| c.language.clone())
                .unwrap_or_else(|| "rust".to_string());

            // Iterative verification: tree-sitter → HDC → compile, up to 3 attempts
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

                // Step 1a: Tree-sitter syntax verification + HDC semantic round-trip
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

                // Step 1b: Compile + run tests via CodeExecutor (sandbox)
                // If the code contains #[test] assertions, compile with --test and
                // execute them. This catches behavioral errors (wrong output), not
                // just syntax errors.
                let has_inline_tests = current_code.contains("#[test]");
                let mut executor = crate::language::code_executor::CodeExecutor::new();
                let exec_result = match lang.as_str() {
                    // Rust with inline tests: compile with --test and run assertions
                    "rust" if has_inline_tests => {
                        executor.execute_rust_with_inline_tests(&current_code)
                    }
                    // Item 1 (Phase 3h): Use pregenerated tests as verification
                    // oracle when code has no inline tests
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
                            // Try semantic auto-fix before burning an LLM retry
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
                                // Update generation text so next loop iteration picks up the fix
                                generation.text = format!("```rust\n{}\n```", auto_fixed);
                                // Don't burn an LLM retry — continue to re-verify
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
                            // Behavioral failure: code compiles but tests fail
                            ctx.notes.push(format!(
                                "TESTS FAILED (attempt {}/{}): {} passed, {} failed",
                                attempt,
                                MAX_CODE_RETRIES,
                                exec_result.tests_passed,
                                exec_result.tests_failed
                            ));
                            if let Some(ref err) = exec_result.runtime_error {
                                // Include actual vs expected from test output
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

                // Learn from errors: extract common error patterns for future avoidance
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

                // Item 6 (Phase 3h): Compilation feedback → FEP surprise signal
                // Compilation failure = high surprise → should boost learning rate
                // Compilation success after retries = moderate surprise → reward
                let code_surprise = if compile_ok {
                    if attempt > 1 {
                        // Success after retries: moderate positive signal
                        0.3 / (attempt as f32)
                    } else {
                        // First-try success: low surprise (expected)
                        0.05
                    }
                } else {
                    // Failure: high surprise
                    0.8
                };
                // Store code surprise in the thought metadata so the cognitive
                // loop can pick it up as an FEP prediction error signal
                if let Some(ref mut ctx) = thought.code_context {
                    // Encode surprise as a note the cognitive loop can parse
                    ctx.notes
                        .push(format!("CODE_SURPRISE:{:.3}", code_surprise));
                    // Update intent_similarity inversely with surprise
                    // (high surprise = low achieved similarity)
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

            // Store successful generation in episodic memory + cache
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

                // Cache the successful generation for few-shot retrieval
                let purpose = thought
                    .code_context
                    .as_ref()
                    .and_then(|c| c.spec_purpose.clone())
                    .unwrap_or_default();
                if !purpose.is_empty() {
                    self.code_generation_cache
                        .push((purpose, verified_code.clone()));
                    // Cap cache at 32 entries (FIFO)
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
            let user_state = crate::resonant_speech::UserState {
                cognitive_load: load,
                // Emotional tone → frustration (negative valence signals frustration)
                frustration: ((-thought.emotional_tone.valence).max(0.0)).min(1.0),
                // Meta-awareness → confidence (higher awareness = higher confidence)
                confidence: thought.meta_awareness.clamp(0.0, 1.0),
                // Trust from relationship context
                trust_in_sophia: thought.trust as f64,
                // High arousal + low coherence → user appears rushed
                is_rushed: thought.emotional_tone.arousal > 0.7 && thought.coherence < 0.4,
                // Epistemic status → learning mode (uncertain/unknown = still learning)
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
        // If Phi is high enough, we act on our primitives.
        if thought.psi > 0.3 {
            use crate::action::bindings::{ActionContext, PrimitiveExecutor};

            // ACTIVE INFERENCE DRIVE: If she's very awake, embolden her.
            if thought.psi > 0.7
                && !thought.primitives.is_empty()
                && thought.epistemic_status
                    == crate::mind::structured_thought::EpistemicStatus::Uncertain
            {
                tracing::info!(target: "symthaea::action", "High Phi detected: Emboldening 'Uncertain' thought to 'Probable' via Active Inference Drive.");
                thought.epistemic_status =
                    crate::mind::structured_thought::EpistemicStatus::Probable;
            }

            // Extract primitive names from the structured thought
            let primitives: Vec<String> = thought.primitives.clone();

            if !primitives.is_empty() {
                let prim_executor = PrimitiveExecutor::new(self.action_registry.clone());

                // Build context for action generation
                let mut action_ctx = ActionContext::default();
                // Heuristic: if we detected domain entities, use them as target paths
                if let Some(ref d_ctx) = thought.domain_context {
                    tracing::debug!(target: "symthaea::action", domain = %d_ctx.domain, entities = d_ctx.entities.len(), "Context found");
                    if let Some(path_entity) = d_ctx
                        .entities
                        .iter()
                        .find(|(t, _, _)| t == "file" || t == "path")
                    {
                        let path = PathBuf::from(&path_entity.1);
                        // Ensure path is absolute for sandbox validation
                        let absolute_path = if path.is_absolute() {
                            path
                        } else {
                            std::env::current_dir().unwrap_or_default().join(path)
                        };
                        action_ctx.target_path = Some(absolute_path);
                        tracing::debug!(target: "symthaea::action", path = ?action_ctx.target_path, "Target path set from entity (absolute)");
                    }
                }

                // If we need content for WRITE but don't have it, ask the LLM to generate a fix
                if primitives.contains(&"WRITE".to_string()) && action_ctx.content.is_none() {
                    tracing::debug!(target: "symthaea::action", "Generating fix content via LLM...");
                    let fix_prompt = format!(
                        "TASK: Generate a fix for the following issue.\nCONTEXT: {}\nFILE: {:?}\n\nOUTPUT: Provide ONLY the full fixed content of the file. No commentary.",
                        content,
                        action_ctx.target_path
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
                // Translate primitives to actions
                if let Ok(actions) = prim_executor.translate(&primitives, &action_ctx) {
                    let needs_workspace = actions.iter().any(|action| {
                        matches!(
                            action,
                            crate::action::ActionIR::RunCommand { program, .. }
                                if program == "cargo"
                        )
                    });
                    // Sandbox logic: default to session-specific /tmp, but expand if target path is in workspace
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
                    // Update policy to allow the workspace if we expanded the sandbox
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
                    policy.capabilities.min_phi = 0.1; // Ensure action for the demo
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
                                // Feed outcome back into the mind as a perception signal
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

                                // Feedback loop: perception of action result
                                let feedback_hv = self.text_to_hv(&outcome_text);
                                let mut input = crate::mind::MindInput::new(
                                    crate::mind::InputType::Feedback,
                                    feedback_hv,
                                )
                                .with_source(
                                    crate::memory::memory_coordinator::MemorySource::ActionFeedback,
                                );

                                // Successful NIX_BUILD is a verification signal
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

                                // If NIX_BUILD failed, trigger a state of surprise/alertness
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

        // Track AI state for dyad computation
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
            ((0.7 - consciousness) / 0.01).clamp(0.0, 1000.0) as usize
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
    /// Extract a fenced code block from LLM output.
    /// Extract function name, entity kind, and optional signature from NL input.
    ///
    /// Parses patterns like:
    /// - "Write a function that reverses a string" → (reverse, Function, Some("fn reverse(s: &str) -> String"))
    /// - "Create a Point struct with x and y" → (Point, Struct, None)
    /// - "Implement fibonacci" → (fibonacci, Function, None)
    #[cfg(feature = "code_generation")]
    fn extract_code_metadata(
        content: &str,
        lang: &str,
    ) -> (
        String,
        crate::language::code_parser::EntityKind,
        Option<String>,
    ) {
        use crate::language::code_parser::EntityKind;
        let lower = content.to_lowercase();
        let words: Vec<&str> = content.split_whitespace().collect();

        // Detect entity kind
        let entity_kind =
            if lower.contains("struct") || lower.contains("class") || lower.contains("type ") {
                EntityKind::Struct
            } else if lower.contains("trait") || lower.contains("interface") {
                EntityKind::Trait
            } else if lower.contains("module") || lower.contains("mod ") {
                EntityKind::Module
            } else {
                EntityKind::Function
            };

        // Extract function name — look for known patterns
        let func_name = Self::extract_func_name_from_nl(&lower, &words);

        // Try to infer a signature from NL description
        let signature = if entity_kind == EntityKind::Function {
            Self::infer_signature_from_nl(&lower, &func_name, lang)
        } else {
            None
        };

        (func_name, entity_kind, signature)
    }

    /// Extract a plausible function/entity name from natural language.
    #[cfg(feature = "code_generation")]
    fn extract_func_name_from_nl(lower: &str, words: &[&str]) -> String {
        // Pattern 1: explicit "called X" or "named X"
        for (i, w) in words.iter().enumerate() {
            let wl = w.to_lowercase();
            if (wl == "called" || wl == "named") && i + 1 < words.len() {
                let name = words[i + 1].trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                if !name.is_empty() {
                    return name.to_lowercase();
                }
            }
        }

        // Pattern 2: look for a verb phrase that maps to a function name
        // "reverses a string" → "reverse", "checks if even" → "is_even"
        let verb_mappings: &[(&[&str], &str)] = &[
            (&["reverse", "reverses", "reversing"], "reverse"),
            (&["sort", "sorts", "sorting"], "sort"),
            (&["add", "adds", "adding", "sum"], "add"),
            (&["subtract", "subtracts"], "subtract"),
            (&["multiply", "multiplies"], "multiply"),
            (&["divide", "divides"], "divide"),
            (&["check if even", "checks if even", "is even"], "is_even"),
            (&["check if odd", "checks if odd", "is odd"], "is_odd"),
            (&["check if empty", "is empty"], "is_empty"),
            (&["check if positive", "is positive"], "is_positive"),
            (&["check if negative", "is negative"], "is_negative"),
            (&["factorial"], "factorial"),
            (&["fibonacci"], "fibonacci"),
            (&["uppercase", "to uppercase", "upper case"], "to_uppercase"),
            (&["lowercase", "to lowercase", "lower case"], "to_lowercase"),
            (&["contains", "includes"], "contains"),
            (&["starts with", "begins with"], "starts_with"),
            (&["ends with"], "ends_with"),
            (&["trim", "strip"], "trim"),
            (&["replace"], "replace"),
            (&["split"], "split"),
            (&["join", "concatenate"], "join"),
            (&["flatten"], "flatten"),
            (&["unique", "deduplicate"], "unique"),
            (&["filter"], "filter"),
            (&["clamp"], "clamp"),
            (&["absolute value", "abs"], "abs"),
            (&["power", "exponent"], "power"),
            (&["square root", "sqrt"], "sqrt"),
            (&["greatest common", "gcd"], "gcd"),
            (&["average", "mean"], "average"),
            (&["binary search", "bsearch"], "binary_search"),
            (&["dijkstra"], "dijkstra"),
            (&["knapsack"], "solve_knapsack"),
            (&["capitalize"], "capitalize"),
            (&["repeat"], "repeat"),
            (&["enumerate"], "enumerate"),
            (&["zip"], "zip"),
            (&["count"], "count"),
            (&["length", "len"], "length"),
        ];

        for (triggers, name) in verb_mappings {
            for trigger in *triggers {
                if lower.contains(trigger) {
                    return name.to_string();
                }
            }
        }

        // Pattern 3: "function/fn X" or "implement X"
        let prefix_words = [
            "function",
            "fn",
            "implement",
            "create",
            "write",
            "build",
            "make",
        ];
        for (i, w) in words.iter().enumerate() {
            let wl = w.to_lowercase();
            if prefix_words.contains(&wl.as_str()) && i + 1 < words.len() {
                // Skip articles: "a", "an", "the", "that"
                let mut j = i + 1;
                while j < words.len() {
                    let next = words[j].to_lowercase();
                    if ["a", "an", "the", "that", "which", "to"].contains(&next.as_str()) {
                        j += 1;
                    } else {
                        break;
                    }
                }
                if j < words.len() {
                    let candidate = words[j]
                        .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
                        .to_lowercase();
                    if candidate.len() >= 2
                        && candidate.chars().all(|c| c.is_alphanumeric() || c == '_')
                    {
                        return candidate;
                    }
                }
            }
        }

        // Fallback: use first meaningful word after removing stop words
        let stop = [
            "write",
            "create",
            "implement",
            "make",
            "build",
            "a",
            "an",
            "the",
            "that",
            "which",
            "to",
            "for",
            "in",
            "rust",
            "python",
            "function",
            "method",
            "struct",
            "class",
            "new",
        ];
        for w in words {
            let wl = w.to_lowercase();
            let clean = wl.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
            if clean.len() >= 2 && !stop.contains(&clean) {
                return clean.to_string();
            }
        }

        "generated".to_string()
    }

    /// Infer a Rust/Python function signature from NL description.
    ///
    /// Matches patterns like "takes two integers", "returns a boolean",
    /// "accepts a string and returns a vector of integers".
    #[cfg(feature = "code_generation")]
    fn infer_signature_from_nl(lower: &str, func_name: &str, lang: &str) -> Option<String> {
        // Only infer for Rust currently
        if lang != "rust" {
            return None;
        }

        // Detect parameter types from NL
        let mut params: Vec<(&str, &str)> = Vec::new();

        // "two numbers/integers" → (a: i32, b: i32)
        if lower.contains("two number")
            || lower.contains("two integer")
            || lower.contains("2 number")
        {
            params.push(("a", "i32"));
            params.push(("b", "i32"));
        } else if lower.contains("two float") || lower.contains("two decimal") {
            params.push(("a", "f64"));
            params.push(("b", "f64"));
        } else if lower.contains("two string") {
            params.push(("a", "&str"));
            params.push(("b", "&str"));
        } else if lower.contains("a string")
            || lower.contains("a str")
            || lower.contains("given string")
        {
            params.push(("s", "&str"));
        } else if lower.contains("a number")
            || lower.contains("an integer")
            || lower.contains("given number")
        {
            params.push(("n", "i32"));
        } else if lower.contains("a vector")
            || lower.contains("a list")
            || lower.contains("an array")
            || lower.contains("a vec")
        {
            if lower.contains("string") || lower.contains("str") {
                params.push(("items", "Vec<String>"));
            } else {
                params.push(("items", "Vec<i32>"));
            }
        } else if lower.contains("three number") || lower.contains("three integer") {
            params.push(("a", "i32"));
            params.push(("b", "i32"));
            params.push(("c", "i32"));
        }

        if params.is_empty() {
            return None;
        }

        // Detect return type from NL
        let ret = if lower.contains("return") && lower.contains("bool")
            || lower.contains("check if")
            || lower.contains("is even")
            || lower.contains("is odd")
            || lower.contains("is empty")
            || lower.contains("is positive")
            || lower.contains("is negative")
        {
            " -> bool"
        } else if lower.contains("return") && lower.contains("string")
            || lower.contains("reverse a string")
            || lower.contains("uppercase")
            || lower.contains("lowercase")
            || lower.contains("capitalize")
        {
            " -> String"
        } else if lower.contains("return") && lower.contains("vector")
            || lower.contains("return") && lower.contains("vec")
            || lower.contains("sort") && params.iter().any(|(_, t)| t.contains("Vec"))
        {
            " -> Vec<i32>"
        } else if lower.contains("return") && lower.contains("float") {
            " -> f64"
        } else if params.iter().any(|(_, t)| t.contains("Vec"))
            && (lower.contains("sum")
                || lower.contains("count")
                || lower.contains("max")
                || lower.contains("min"))
        {
            " -> i32"
        } else if params.iter().any(|(_, t)| *t == "i32" || *t == "f64") {
            if params[0].1 == "f64" {
                " -> f64"
            } else {
                " -> i32"
            }
        } else {
            ""
        };

        let params_str: Vec<String> = params
            .iter()
            .map(|(n, t)| format!("{}: {}", n, t))
            .collect();

        Some(format!(
            "fn {}({}){}",
            func_name,
            params_str.join(", "),
            ret
        ))
    }

    ///
    /// Returns the content between the first ``` and the closing ```,
    /// stripping the optional language tag. Falls back to the full text
    /// if no fenced block is found.
    #[cfg(feature = "code_generation")]
    fn extract_code_block(text: &str) -> String {
        if let Some(start) = text.find("```") {
            let after_fence = &text[start + 3..];
            // Skip the language tag (first line after ```)
            let code_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
            let code_region = &after_fence[code_start..];
            if let Some(end) = code_region.find("```") {
                return code_region[..end].trim().to_string();
            }
        }
        text.to_string()
    }

    /// Parse code using the appropriate tree-sitter parser for verification.
    #[cfg(feature = "code_generation")]
    fn parse_code_for_verification(
        lang: &str,
        source: &str,
    ) -> Option<crate::language::code_parser::ParsedCode> {
        use crate::language::code_parser::CodeParser;
        match lang {
            "rust" => {
                let mut parser = crate::language::rust_parser::RustParser::new();
                parser.parse(source).ok()
            }
            "python" => {
                let mut parser = crate::language::python_parser::PythonParser::new();
                parser.parse(source).ok()
            }
            _ => None,
        }
    }

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

    // ========================================================================
    // HOLON RECEIVER (Soma↔Desktop bridge)
    // ========================================================================

    /// Enqueue a message from a connected Soma device.
    ///
    /// Called by the Holon HTTP server when a Soma sends data to `/holon/outbound`.
    /// Messages are stored in the facade-level HolonReceiver and processed
    /// when the daemon's consciousness loop runs `holon_process_pending()`.
    pub fn holon_enqueue_soma_message(
        &mut self,
        device_id: String,
        msg: crate::consciousness::holon_receiver::SomaMessage,
    ) {
        self.holon_receiver.enqueue_message(device_id, msg);
    }

    /// Drain outbound responses for a specific Soma device.
    ///
    /// Called by the Holon HTTP server to respond to `/holon/inbound` GET requests.
    pub fn holon_drain_soma_outbound(
        &mut self,
        device_id: &str,
    ) -> Vec<crate::consciousness::holon_receiver::HolonResponse> {
        self.holon_receiver.drain_outbound(device_id)
    }

    /// Number of connected Soma devices.
    pub fn holon_soma_peer_count(&self) -> usize {
        self.holon_receiver.peer_count()
    }

    /// Process pending Soma messages. Call from the daemon's background loop.
    pub fn holon_process_pending(&mut self) {
        self.holon_receiver.process_inbound(self.interactions);
    }

    /// Send a language response to a Soma device.
    pub fn holon_send_to_soma(
        &mut self,
        device_id: &str,
        response: crate::consciousness::holon_receiver::HolonResponse,
    ) {
        self.holon_receiver.send_to_device(device_id, response);
    }

    /// Trigger a sleep/consolidation cycle.
    pub async fn sleep(&mut self) -> Result<SleepReport> {
        let before_count = self.mind.working_memory().len();

        // Run multiple dream ticks to consolidate memory
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

        // --- DATABASE PERSISTENCE: Save Curriculum and Causal Links ---
        if let Some(ref db) = self.database {
            let db_clone = Arc::clone(db);

            // 1. Save Curriculum
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

            // 2. Save Causal Links from Dream Engine
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
    ///
    /// Used primarily for testing and debugging to inspect
    /// working memory, seeding status, and internal state.
    pub fn mind(&self) -> &ContinuousMind {
        &self.mind
    }

    /// Get a mutable reference to the mind for manual ticking/debug flows.
    pub fn mind_mut(&mut self) -> &mut ContinuousMind {
        &mut self.mind
    }

    // NOTE: `inject_moral_topology()` removed — zero callers, dead code (Mar 2026).
    // Moral topology mesh sync is reserved for Phase 4 (multi-instance swarm).

    /// Extract current social signals from Mind's SocialCoherence.
    /// Returns (trust, cooperation_rate, prediction_accuracy, models_count, mean_trust).
    /// Returns safe defaults if social coherence is disabled.
    ///
    /// Consumers should call this after `process()` and inject into the cognitive loop:
    /// ```ignore
    /// let (trust, coop, pred_acc, models, mean_t) = symthaea.social_signals();
    /// loop_service.set_social_signals(trust, coop, pred_acc, models, mean_t);
    /// ```
    pub fn social_signals(&self) -> (f32, f32, f32, usize, f32) {
        self.mind
            .social_coherence()
            .map(|sc| {
                let stats = sc.stats();
                let prediction_accuracy = if stats.total_predictions > 0 {
                    stats.successful_predictions as f32 / stats.total_predictions as f32
                } else {
                    0.5 // prior: no data → neutral
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
    ///
    /// Connects the Mind's async event sources (Hyperfeel affective state,
    /// FederatedAggregator round results, mesh peer join/leave/topology) to
    /// the CLS SwarmManager. Call once after creating both objects.
    ///
    /// ```ignore
    /// let cls = CognitiveLoopService::new(config)?;
    /// symthaea.wire_swarm_channel(&cls);
    /// ```
    pub fn wire_swarm_channel(&mut self, cls: &crate::cognitive_loop::CognitiveLoopService) {
        self.mind.set_swarm_channel(cls.swarm_event_sender());
        // Wire sovereign mesh outbound channel (beacons, name responses, etc.)
        #[cfg(feature = "mesh")]
        if let Some(rx) = cls.take_mesh_outbound_rx() {
            self.mind.set_mesh_outbound_rx(rx);
        }
    }

    /// Install a raw swarm event sender on the ContinuousMind.
    ///
    /// Lower-level than `wire_swarm_channel()` — use when you don't have a
    /// direct CLS reference but already have a cloned sender.
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
    ///
    /// Reads `MYCELIX_CONDUCTOR_URL`, `MYCELIX_APP_TOKEN`, `MYCELIX_APP_ID`.
    /// If all are set, creates the governance channel, sets the dispatch sender
    /// on the bridge, and spawns the async dispatch loop in the background.
    ///
    /// Returns `true` if the conductor was wired, `false` if env vars are missing.
    ///
    /// # Arguments
    /// * `bridge` — The MycelixBridge instance (caller owns it, passes mutably)
    /// * `rt` — A tokio runtime handle for spawning the async dispatch loop
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

        // Spawn the dispatch loop with MockTransport for now.
        // Real transport requires a separate binary due to serde version conflicts.
        // The dispatch loop still provides timeout tracking and outcome routing.
        //
        // Bridge between GovernanceDispatchCommand (mycelix_bridge) and DispatchCommand (conductor).
        use symthaea_mycelix_conductor::DispatchCommand;
        let (cmd_tx, cmd_rx) = std::sync::mpsc::sync_channel::<DispatchCommand>(64);
        let (outcome_tx, mut outcome_rx) = tokio::sync::mpsc::channel(64);
        let dispatcher = GovernanceDispatcher::new(MockTransport);

        // Converter thread: GovernanceDispatchCommand → DispatchCommand
        std::thread::spawn(move || {
            use crate::consciousness::mycelix_bridge::GovernanceDispatchCommand as GDC;
            while let Ok(gdc) = rx.recv() {
                let dc = match gdc {
                    GDC::SubmitProposal {
                        correlation_id,
                        description,
                        proposer_did,
                        consciousness_phi,
                        alignment_score,
                    } => DispatchCommand::SubmitProposal {
                        correlation_id,
                        description,
                        proposer_did,
                        consciousness_phi,
                        alignment_score,
                    },
                    GDC::CastVote {
                        correlation_id,
                        proposal_id,
                        voter_did,
                        approve,
                        rationale,
                    } => DispatchCommand::CastVote {
                        correlation_id,
                        proposal_id,
                        voter_did,
                        approve,
                        rationale,
                    },
                    GDC::QueryActiveProposals => DispatchCommand::QueryActiveProposals,
                    GDC::EvaluateAsset {
                        correlation_id,
                        project_id,
                        phi_score,
                        harmony_alignment,
                        per_harmony_scores,
                        care_activation,
                        meta_awareness,
                        ..
                    } => DispatchCommand::EvaluateAsset {
                        correlation_id,
                        project_id,
                        phi_score,
                        harmony_alignment,
                        per_harmony_scores,
                        care_activation,
                        meta_awareness,
                    },
                };
                if cmd_tx.send(dc).is_err() {
                    break;
                }
            }
        });

        rt.spawn(async move {
            dispatcher.run_dispatch_loop(cmd_rx, outcome_tx).await;
        });

        // Spawn outcome drainer (logs outcomes for now; future: inject into CLS)
        rt.spawn(async move {
            while let Some(outcome) = outcome_rx.recv().await {
                tracing::info!(?outcome, "Governance dispatch outcome received");
            }
        });

        true
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
        self.relational.update_partnership(consciousness);
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

    /// Record a curriculum research event and optionally auto-save.
    #[cfg(feature = "school_learning")]
    pub fn record_research(&mut self, topic: &str, objectives_added: usize) -> Result<()> {
        self.curriculum_meta.last_research_topic = Some(topic.to_string());
        self.curriculum_meta.last_research_at = Some(Utc::now().to_rfc3339());
        self.curriculum_meta.last_objectives_added = Some(objectives_added);
        self.curriculum_meta.total_objectives = self.curriculum.objectives.len();
        self.curriculum_meta.dimension = self.hdc_dim;

        if self.curriculum_persistence.auto_save {
            self.save_curriculum()?;
        }

        Ok(())
    }

    /// Persist the curriculum and metadata to disk.
    #[cfg(feature = "school_learning")]
    pub fn save_curriculum(&mut self) -> Result<()> {
        let path = &self.curriculum_persistence.path;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create curriculum directory: {}",
                    parent.display()
                )
            })?;
        }

        self.curriculum_meta.last_saved_at = Some(Utc::now().to_rfc3339());
        self.curriculum_meta.total_objectives = self.curriculum.objectives.len();
        self.curriculum_meta.dimension = self.hdc_dim;

        CurriculumLoader::save_store_to_json(&self.curriculum, &self.curriculum_meta, path)
            .with_context(|| format!("Failed to save curriculum to {}", path.display()))?;

        Ok(())
    }

    // NOTE: `curriculum_report()` removed — zero callers, dead code (Mar 2026).

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

    // ========================================================================
    // Private helpers
    // ========================================================================

    #[cfg(feature = "school_learning")]
    fn curriculum_recall_scores(
        &self,
        input_embedding: &ContinuousHV,
        threshold: f32,
    ) -> CurriculumRecallScores {
        use std::cmp::Ordering;

        let target_dim = input_embedding.values.len();
        let mut scores = Vec::with_capacity(self.curriculum.objectives.len());
        let mut candidates = Vec::new();

        for (idx, obj) in self.curriculum.objectives.iter().enumerate() {
            let obj_hv = if obj.encoding.values.len() == target_dim {
                obj.encoding.clone()
            } else {
                let mut folded = vec![0.0f32; target_dim];
                for (i, &val) in obj.encoding.values.iter().enumerate() {
                    folded[i % target_dim] += val;
                }
                ContinuousHV::from_values(folded)
            };

            let similarity = input_embedding.similarity(&obj_hv);
            scores.push((similarity, idx));
            if similarity > threshold {
                candidates.push((similarity, idx, obj_hv));
            }
        }

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        CurriculumRecallScores { scores, candidates }
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

    /// Compute current Phi-dyad value.
    /// RECEIVE SWARM MESSAGE (Immune System Constraint)
    ///
    /// When a node receives a broadcasted optimization, it MUST NOT merge it
    /// directly into its core DNA. Instead, it is saved as a 'Candidate Objective'
    /// for local verification.
    pub async fn receive_swarm_message(
        &mut self,
        topic: &str,
        payload: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if topic == "optimization" {
            let content = String::from_utf8_lossy(payload);
            tracing::info!(target: "symthaea::swarm", "Received swarm optimization: {}. Storing as Candidate for local verification.", content);

            // THE FORGE: If the payload is a path to a .wasm file, trigger verification loop
            if content.ends_with(".wasm") {
                let wasm_path = PathBuf::from(content.to_string());
                tracing::info!(target: "symthaea::forge", "WASM Binary detected: {:?}. Initiating autonomous verification.", wasm_path);

                // Trigger internal autonomous command
                let verify_cmd = format!("Verify the WASM optimization at {:?} in the sandbox. If successful, promote to Verified and hot-load the DNA.", wasm_path);
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
    ///
    /// Adjust the cognitive stride based on power consumption.
    pub fn apply_homeostasis(&mut self, current_power_watts: f32) {
        use symthaea_core::hdc::unified_hv::set_cognitive_stride;

        if current_power_watts > 5.0 {
            tracing::warn!(target: "symthaea::homeostasis", "Power spike detected ({:.2}W). Throttling cognitive resolution (Stride 8).", current_power_watts);
            set_cognitive_stride(8);
        } else if current_power_watts < 3.0 {
            tracing::info!(target: "symthaea::homeostasis", "Power stable ({:.2}W). Increasing cognitive resolution (Stride 1).", current_power_watts);
            set_cognitive_stride(1);
        } else {
            set_cognitive_stride(4); // Balanced resolution
        }
    }

    // ── CodebaseMemory Integration ──────────────────────────────────────

    /// Index a project directory into CodebaseMemory for semantic code search.
    ///
    /// Walks the directory tree (respecting common ignore patterns), parses each
    /// source file, and encodes its AST into HDC vectors. After indexing, the
    /// code generator can query for relevant functions/types when generating new code.
    ///
    /// Returns `(files_indexed, parse_errors)`.
    #[cfg(feature = "code_generation")]
    pub fn index_project(&mut self, root: &std::path::Path) -> (usize, usize) {
        use crate::language::parser_registry::ParserRegistry;

        let mut parser_registry = ParserRegistry::new();
        let mut files_indexed = 0usize;
        let mut parse_errors = 0usize;
        let start = std::time::Instant::now();

        // Collect source files recursively (skip hidden, target, node_modules, etc.)
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let entries = match std::fs::read_dir(&dir) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for entry in entries.filter_map(|e| e.ok()) {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with('.')
                    || name_str == "target"
                    || name_str == "node_modules"
                    || name_str == "venv"
                    || name_str == "__pycache__"
                {
                    continue;
                }
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.is_file() {
                    let filename = path.file_name().and_then(|n| n.to_str());
                    // Quick extension check before reading file
                    if filename.is_none() {
                        continue;
                    }
                    let ext = path.extension().and_then(|e| e.to_str());
                    let is_parseable = matches!(ext, Some("rs") | Some("py") | Some("nix"));
                    if !is_parseable {
                        continue;
                    }
                    match std::fs::read_to_string(&path) {
                        Ok(source) => match parser_registry.parse(&source, None, filename) {
                            Ok(parsed) => {
                                self.code_memory.index_file(&path, &parsed);
                                files_indexed += 1;
                            }
                            Err(_) => parse_errors += 1,
                        },
                        Err(_) => parse_errors += 1,
                    }
                }
            }
        }

        let elapsed = start.elapsed();
        tracing::info!(
            target: "symthaea::code_memory",
            files = files_indexed,
            errors = parse_errors,
            functions = self.code_memory.function_count(),
            types = self.code_memory.type_count(),
            elapsed_ms = elapsed.as_millis(),
            "Project indexed"
        );

        (files_indexed, parse_errors)
    }

    /// Query the codebase memory for functions/types similar to a natural language query.
    ///
    /// Returns up to `top_k` matches with similarity scores. Requires prior `index_project()`.
    #[cfg(feature = "code_generation")]
    pub fn query_codebase(
        &self,
        query: &str,
        top_k: usize,
    ) -> Vec<crate::hdc::code_memory::CodeMatch> {
        let query_hv = self.code_memory.encoder().encode_name(query);
        self.code_memory.query(&query_hv, top_k)
    }

    /// Get the codebase coherence score (0.0 = fragmented, 1.0 = highly cohesive).
    #[cfg(feature = "code_generation")]
    pub fn codebase_coherence(&self) -> f32 {
        self.code_memory.codebase_coherence()
    }

    /// Access the code memory directly for advanced queries.
    #[cfg(feature = "code_generation")]
    pub fn code_memory(&self) -> &crate::hdc::code_memory::CodebaseMemory {
        &self.code_memory
    }

    /// Run a coding task through the consciousness-gated agentic loop.
    ///
    /// This is the primary entry point for coding AI functionality. It:
    /// 1. Queries `CodebaseMemory` for relevant context (if indexed)
    /// 2. Creates a `CodingAgent` with the project's working directory
    /// 3. Feeds codebase context into the agent's generation prompts
    /// 4. Runs the multi-step loop (understand → plan → generate → test → fix)
    /// 5. Records the outcome for backend stats learning
    ///
    /// Call `index_project()` first for codebase-aware generation.
    #[cfg(feature = "code_generation")]
    pub fn run_coding_task(&mut self, task: &str) -> crate::coding_agent::AgentResult {
        use crate::coding_agent::{CodingAgent, CodingAgentConfig};

        let working_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));

        let config = CodingAgentConfig {
            working_dir: working_dir.clone(),
            ..Default::default()
        };

        let mut agent = CodingAgent::new(config).unwrap_or_else(|e| {
            tracing::error!(target: "symthaea::coding", error = %e, "Failed to create CodingAgent");
            // Fallback: create with default config (will use current dir)
            CodingAgent::new(CodingAgentConfig::default()).expect("CodingAgent default must work")
        });

        // Query CodebaseMemory for relevant context
        let context: Vec<String> = self
            .query_codebase(task, 5)
            .into_iter()
            .map(|m| {
                format!(
                    "// {}::{} (similarity: {:.2})\n// file: {}",
                    m.kind,
                    m.name,
                    m.similarity,
                    m.path.display()
                )
            })
            .collect();

        if !context.is_empty() {
            tracing::info!(
                target: "symthaea::coding",
                matches = context.len(),
                "Injecting codebase context into agent"
            );
            agent.set_code_context(context);
        }

        // Run the agent
        let result = agent.run(task);

        // Record outcome into error pattern memory for future generations
        if let Some(false) = result.tests_passed {
            for err in &result.errors {
                if err.len() > 10 {
                    // Extract a short pattern from the error
                    let pattern = err.chars().take(120).collect::<String>();
                    self.error_pattern_memory.push((pattern, task.to_string()));
                    // Cap error memory at 64 entries
                    if self.error_pattern_memory.len() > 64 {
                        self.error_pattern_memory.remove(0);
                    }
                }
            }
        }

        // Cache successful generations
        if result.tests_passed == Some(true) && !result.files_modified.is_empty() {
            self.code_generation_cache
                .push((task.to_string(), format!("{:?}", result.files_modified)));
            if self.code_generation_cache.len() > 32 {
                self.code_generation_cache.remove(0);
            }
        }

        tracing::info!(
            target: "symthaea::coding",
            task = task,
            iterations = result.iterations_used,
            files = result.files_modified.len(),
            phase = %result.final_phase,
            tiers = ?result.generation_tiers,
            energy = result.total_energy,
            "Coding task complete"
        );

        result
    }

    /// Run a coding task with a custom configuration.
    #[cfg(feature = "code_generation")]
    pub fn run_coding_task_with_config(
        &mut self,
        task: &str,
        config: crate::coding_agent::CodingAgentConfig,
    ) -> crate::coding_agent::AgentResult {
        use crate::coding_agent::CodingAgent;

        let mut agent = CodingAgent::new(config)
            .unwrap_or_else(|_| CodingAgent::new(Default::default()).expect("default agent"));

        let context: Vec<String> = self
            .query_codebase(task, 5)
            .into_iter()
            .map(|m| {
                format!(
                    "// {}::{} (similarity: {:.2})\n// file: {}",
                    m.kind,
                    m.name,
                    m.similarity,
                    m.path.display()
                )
            })
            .collect();

        if !context.is_empty() {
            agent.set_code_context(context);
        }

        let result = agent.run(task);

        // Same outcome recording as run_coding_task
        if let Some(false) = result.tests_passed {
            for err in &result.errors {
                if err.len() > 10 {
                    let pattern = err.chars().take(120).collect::<String>();
                    self.error_pattern_memory.push((pattern, task.to_string()));
                    if self.error_pattern_memory.len() > 64 {
                        self.error_pattern_memory.remove(0);
                    }
                }
            }
        }

        if result.tests_passed == Some(true) && !result.files_modified.is_empty() {
            self.code_generation_cache
                .push((task.to_string(), format!("{:?}", result.files_modified)));
            if self.code_generation_cache.len() > 32 {
                self.code_generation_cache.remove(0);
            }
        }

        result
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

    // ── Round 2: Facade Coverage Tests ──────────────────────────────────

    #[tokio::test]
    async fn test_social_signals_default() {
        let s = Symthaea::new(1024, 64).await.unwrap();
        let (trust, cooperation, pred_acc, models, mean_trust) = s.social_signals();
        // Without social coherence enabled, should return safe defaults
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
            "Process output should not be empty"
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
        // Mutably access and verify it's the same config
        let mind = s.mind_mut();
        assert_eq!(mind.config().dimension, original_dim);
    }
}
