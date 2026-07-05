// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Coding Agent: multi-step consciousness-gated coding loop.
//!
//! Drives Symthaea through read -> reason -> write -> test -> fix cycles,
//! using FEP active inference for tool selection and consciousness gating
//! for epistemic honesty.
//!
//! ## Architecture
//!
//! The `CodingAgent` wraps a `CognitiveLoopService` (not the full `Symthaea` facade)
//! and directly drives the cognitive cycle, interpreting motor commands as coding actions.
//! This keeps the agent lightweight while retaining consciousness gating, moral algebra,
//! and all cognitive subsystems.
//!
//! ## Generation Pipeline
//!
//! During the Generating/Fixing phases, the agent:
//! 1. Builds a generation prompt from task + observations + errors + code context
//! 2. Calls `IntelligentDispatcher::generate()` to select the optimal backend
//!    (Native/LocalLLM/CloudLLM) based on consciousness state
//! 3. Writes the generated code to disk
//! 4. Requests `cargo check` via the motor bridge to validate
//!
//! ## State Machine
//!
//! ```text
//! Understanding -> Planning -> Generating -> Testing -> Fixing --> Done
//!      ^              ^           |            |         |
//!      +--            +---        +            +         +
//! ```

mod accessors;
#[cfg(feature = "reasoning_engine")]
mod causal_model;
mod code_utils;
#[cfg(feature = "reasoning_engine")]
pub mod consciousness_bridge;
pub mod error_knowledge;
mod experience;
mod generation;
mod geodesic_gate;
/// MAGI Loop bridge for self-improving code generation predictions.
pub mod magi_code_bridge;
mod planning;
mod prompts;
/// Code self-modification — generates new auto-fix rules from error patterns.
/// Closes the recursive self-improvement loop: observe failures → hypothesize
/// fixes → validate → integrate → permanently improve.
pub mod self_modification;
/// Self-test generation — agent writes its own #[test] modules.
pub mod test_generation;

use crate::action::primitives::{
    Atom, DispatchTier, Molecule, MoleculeExecutor, PlanProfile, PrimitiveValue,
};
use crate::action::{ActionOutcome, PolicyBundle, SandboxRoot};
use crate::coding_experience::{CodingExperience, CodingExperienceStore};
use crate::cognitive_loop::motor_output_bridge::{
    ActionType, MotorActionRequest, MotorOutputBridge, MotorOutputResult,
};
use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, CycleResult};
use crate::consciousness::fep_active_inference::MotorCommandType;
#[cfg(feature = "code_generation")]
use crate::language::code_orchestrator::CodeOrchestrator;
use crate::language::intelligent_dispatcher::{BackendTier, DispatchResult, IntelligentDispatcher};
use crate::language::llm_backend::GenerationParams;
use crate::mind::structured_thought::EpistemicStatus;
use std::path::PathBuf;

/// Current phase of the coding task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskPhase {
    /// Reading files and building context about the codebase.
    Understanding,
    /// Reasoning about the approach — choosing what to change.
    Planning,
    /// Writing code (generating or modifying files).
    Generating,
    /// Running tests or cargo check to validate changes.
    Testing,
    /// Addressing test failures or compiler errors.
    Fixing,
    /// Task complete (or abandoned with partial result).
    Done,
}

impl std::fmt::Display for TaskPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Understanding => write!(f, "Understanding"),
            Self::Planning => write!(f, "Planning"),
            Self::Generating => write!(f, "Generating"),
            Self::Testing => write!(f, "Testing"),
            Self::Fixing => write!(f, "Fixing"),
            Self::Done => write!(f, "Done"),
        }
    }
}

/// Result of a completed coding agent run.
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// Files that were modified during the run.
    pub files_modified: Vec<PathBuf>,
    /// Whether all tests passed (if tests were run).
    pub tests_passed: Option<bool>,
    /// Number of iterations used.
    pub iterations_used: usize,
    /// Phi (consciousness level) trace across iterations.
    pub phi_trace: Vec<f32>,
    /// Final epistemic status (how confident the agent is in its work).
    pub epistemic_status: EpistemicStatus,
    /// Final phase when the agent stopped.
    pub final_phase: TaskPhase,
    /// Accumulated context/observations from the run.
    pub observations: Vec<String>,
    /// Error messages encountered during the run.
    pub errors: Vec<String>,
    /// Backend tiers used for generation (one per generation attempt).
    pub generation_tiers: Vec<BackendTier>,
    /// Total energy consumed across all generations.
    pub total_energy: f64,
    /// Remaining energy budget after execution.
    pub remaining_energy: f32,
    /// Number of unique failure patterns encountered during the run.
    pub failure_pattern_count: usize,
    /// Number of fix attempts that were skipped by deduplication.
    pub dedup_skips: usize,
    /// Number of quality gate rejections.
    pub quality_rejections: usize,
    /// Number of times consciousness gate deferred generation.
    pub consciousness_deferrals: usize,
    /// Whether stuck detection triggered during the run.
    pub stuck_detected: bool,
    /// Auto-generated curriculum lessons from failure patterns encountered.
    #[cfg(feature = "school_learning")]
    pub generated_lessons: Vec<crate::school::code_learning::CodeLesson>,
}

impl AgentResult {
    /// Produce a JSON telemetry summary for dashboard consumption.
    ///
    /// Returns a `serde_json::Value` with all key metrics in a flat structure
    /// suitable for WebSocket streaming or REST API responses.
    pub fn to_telemetry_json(&self) -> serde_json::Value {
        let avg_phi = if self.phi_trace.is_empty() {
            0.0
        } else {
            self.phi_trace.iter().sum::<f32>() / self.phi_trace.len() as f32
        };

        serde_json::json!({
            "phase": format!("{}", self.final_phase),
            "iterations_used": self.iterations_used,
            "consciousness": {
                "avg_phi": avg_phi,
                "min_phi": self.phi_trace.iter().cloned().fold(f32::MAX, f32::min),
                "max_phi": self.phi_trace.iter().cloned().fold(f32::MIN, f32::max),
                "phi_trace": self.phi_trace,
                "samples": self.phi_trace.len(),
            },
            "epistemic_status": format!("{:?}", self.epistemic_status),
            "generation": {
                "tiers": self.generation_tiers.iter().map(|t| t.to_string()).collect::<Vec<_>>(),
                "total_energy": self.total_energy,
                "remaining_energy": self.remaining_energy,
            },
            "files_modified": self.files_modified.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            "tests_passed": self.tests_passed,
            "observations_count": self.observations.len(),
            "errors_count": self.errors.len(),
            "errors_preview": self.errors.iter().take(3).map(|e| {
                let s: String = e.chars().take(100).collect();
                s
            }).collect::<Vec<_>>(),
            "behavioral": {
                "failure_patterns": self.failure_pattern_count,
                "dedup_skips": self.dedup_skips,
                "quality_rejections": self.quality_rejections,
                "consciousness_deferrals": self.consciousness_deferrals,
                "stuck_detected": self.stuck_detected,
            },
        })
    }
}

// ===============================================================================
// Task E types: structured test failures, events, retry strategies, consciousness
// ===============================================================================

/// Parsed test failure with structured fields for targeted fixing.
#[derive(Debug, Clone)]
struct StructuredTestFailure {
    test_name: String,
    failure_kind: TestFailureKind,
    expected: Option<String>,
    actual: Option<String>,
    message: Option<String>,
    file: Option<String>,
    line: Option<usize>,
}

/// Classification of test failure types.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TestFailureKind {
    AssertEq,
    Assert,
    Panic,
    Other,
}

/// Streaming events emitted during agent execution.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    PhaseTransition {
        from: TaskPhase,
        to: TaskPhase,
        iteration: usize,
    },
    Observation(String),
    CodeGenerated {
        tier: BackendTier,
        bytes: usize,
        file: PathBuf,
    },
    TestResult {
        passed: bool,
        error_count: usize,
    },
    ConsciousnessSnapshot {
        phi: f32,
        prediction_error: f32,
        confidence_velocity: f32,
    },
    RetryStrategyChanged(RetryStrategy),
    RequestClarification(String),
    Done(AgentResult),
}

/// Differentiated retry strategies for fixing failures.
#[derive(Debug, Clone, PartialEq)]
pub enum RetryStrategy {
    Default,
    DifferentTemplate,
    DifferentBackend(BackendTier),
    SimplifyScope,
    RequestClarification(String),
}

/// Tracks which retry strategies have been attempted.
#[derive(Debug, Clone)]
struct RetryState {
    strategies_tried: Vec<RetryStrategy>,
    current_strategy: RetryStrategy,
}

impl Default for RetryState {
    fn default() -> Self {
        Self {
            strategies_tried: Vec::new(),
            current_strategy: RetryStrategy::Default,
        }
    }
}

/// Extracted consciousness signals for agent decision-making.
#[derive(Debug, Clone, Default)]
struct ConsciousnessSignals {
    prediction_error: f32,
    confidence_velocity: f32,
    phi: f32,
    phi_slope: f32,
    fep_surprise: f64,
}

/// Configuration for the coding agent.
#[derive(Debug, Clone)]
pub struct CodingAgentConfig {
    /// Maximum iterations before the agent gives up.
    pub max_iterations: usize,
    /// Maximum consecutive failures in a phase before escalating.
    pub max_phase_failures: usize,
    /// Minimum Phi required to proceed with code generation.
    pub min_generation_phi: f64,
    /// Working directory for the project.
    pub working_dir: PathBuf,
    /// Sandbox root for file operations.
    pub sandbox_session: String,
    /// Target file to write generated code into (relative to working_dir).
    /// If None, inferred from task description or defaults to `src/lib.rs`.
    pub target_file: Option<PathBuf>,
    /// Enable real execution (file I/O, cargo check/test) instead of simulated mode.
    /// When true, the motor bridge uses `SimpleExecutor::with_real_commands()` and
    /// the sandbox is rooted at `working_dir` (not `/tmp/symthaea/`).
    pub enable_real_exec: bool,
    /// Use Ollama (qwen2.5-coder:7b) for code generation instead of simulated backend.
    /// Requires Ollama running at localhost:11434.
    pub use_local_llm: bool,
    /// Use Anthropic Claude API as the cloud LLM tier for complex tasks.
    /// Reads `ANTHROPIC_API_KEY` from environment. Falls back gracefully if unavailable.
    pub use_cloud_llm: bool,
    /// Route generation through the unified `CodeOrchestrator` (native + analogy +
    /// LLM, each compiler-verified before acceptance) and record MAGI world-prediction
    /// calibration around every attempt, instead of only the raw `IntelligentDispatcher`
    /// path. Requires the `code_generation` feature. Defaults to `false` — this closes
    /// a real wiring gap (the orchestrator previously had zero call sites in the live
    /// agent loop) without changing default behavior.
    pub use_orchestrator: bool,
    /// Allow `FixRuleGenerator::try_generate_rules()` to hypothesize new self-modification
    /// fix rules from observed error clusters. Defaults to `false`. Observation
    /// (`FixRuleGenerator::observe_error()`) always runs regardless of this flag — only
    /// rule *generation* is gated. Rule *application*/*promotion* (`try_apply_rule`,
    /// `record_rule_outcome`) is never wired into the live agent loop at all, by design:
    /// this is a self-modification pipeline and must never silently mutate the agent's
    /// own fix repertoire without an explicit, separate promotion step.
    pub enable_self_modification: bool,
}

impl Default for CodingAgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            max_phase_failures: 3,
            min_generation_phi: 0.3,
            working_dir: PathBuf::from("."),
            sandbox_session: "coding_agent".to_string(),
            target_file: None,
            enable_real_exec: false,
            use_local_llm: false,
            use_cloud_llm: false,
            use_orchestrator: false,
            enable_self_modification: false,
        }
    }
}

/// A consciousness-gated multi-step coding agent.
///
/// Wraps a `CognitiveLoopService` and drives it through iterative coding cycles.
/// Each iteration runs a cognitive cycle, interprets the FEP motor command,
/// and dispatches the appropriate action (read file, write code, run tests).
pub struct CodingAgent {
    /// The cognitive loop driving consciousness and FEP.
    cognitive_loop: CognitiveLoopService,
    /// Agent configuration.
    config: CodingAgentConfig,
    /// Current phase of the task.
    phase: TaskPhase,
    /// Current iteration count.
    iteration: usize,
    /// Consecutive failures in current phase.
    phase_failures: usize,
    /// Files modified during this run.
    files_modified: Vec<PathBuf>,
    /// Phi trace across iterations.
    phi_trace: Vec<f32>,
    /// Observations accumulated during the run.
    observations: Vec<String>,
    /// Errors accumulated during the run.
    errors: Vec<String>,
    /// Last test/check output for error context.
    last_test_output: Option<String>,
    /// The original task description.
    task: String,
    /// Intelligent dispatcher for consciousness-routed code generation.
    dispatcher: Option<IntelligentDispatcher>,
    /// Last dispatch result for telemetry.
    last_dispatch: Option<DispatchResult>,
    /// Backend tiers used across generations.
    generation_tiers: Vec<BackendTier>,
    /// Last generated code (for writing to disk and feeding into Testing).
    generated_code: Option<String>,
    /// Codebase context: pre-queried relevant code snippets from CodebaseMemory.
    /// Set externally via `set_code_context()` — avoids coupling to the
    /// `code_generation` feature gate.
    code_context: Vec<String>,
    /// Persistent experience store for error patterns and successful generations.
    /// Auto-initialized with in-memory SQLite in constructor.
    experience_store: Option<CodingExperienceStore>,
    /// Accumulated failure patterns during this run: (error_text, count).
    failure_patterns: Vec<(String, usize)>,
    /// Coding attempt history for causal model construction (Phase 6).
    #[cfg(feature = "reasoning_engine")]
    coding_attempts: Vec<causal_model::CodingAttempt>,
    /// Set when native_code_template() returns None — forces LLM tier on next generation.
    /// Cleared after a successful LLM generation so native can be tried again on new tasks.
    native_exhausted: bool,
    /// Prediction error history for trend detection.
    prediction_error_history: Vec<f32>,
    /// Confidence velocity history for trend detection.
    confidence_velocity_history: Vec<f32>,
    /// Optional event channel for streaming agent progress.
    event_sink: Option<std::sync::mpsc::Sender<AgentEvent>>,
    /// Current retry state for differentiated fixing strategies.
    retry_state: RetryState,
    /// Indexed codebase memory for semantic code search (populated by `index_project()`).
    #[cfg(feature = "code_generation")]
    code_memory: Option<crate::hdc::code_memory::CodebaseMemory>,
    /// Cached file sources for source-level context extraction (path -> source text).
    #[cfg(feature = "code_generation")]
    source_cache: std::collections::HashMap<PathBuf, String>,
    /// Current execution plan profile (computed during Planning phase).
    /// Used for energy budgeting and Phi gating before committing to actions.
    current_plan: Option<PlanProfile>,
    /// Remaining energy budget for this run.
    energy_budget: f32,
    /// Tracks attempted fixes as `"{error_sig}:{fix_type}"` keys to skip re-applying same fix.
    attempted_fixes: std::collections::HashSet<String>,
    /// Direct test result flag, set by Testing phase (not inferred from observations).
    tests_passed: Option<bool>,
    /// Counter: fix attempts skipped by deduplication.
    dedup_skips: usize,
    /// Counter: quality gate rejections.
    quality_rejections: usize,
    /// Counter: consciousness gate deferrals.
    consciousness_deferrals: usize,
    /// Whether stuck detection triggered during this run.
    stuck_detected: bool,
    /// Semantic knowledge graph of error patterns → fix strategies.
    /// Accumulates across the run, complementing the flat experience store
    /// with structured error→fix mappings and Bayesian success rates.
    error_knowledge: error_knowledge::CodeErrorKnowledge,
    /// Geodesic Code Synthesis verifier for topological/oracle post-generation checks.
    /// When enabled, generated code is verified for structural invariants (Betti numbers,
    /// convergence prediction) and violations are fed back as hard constraints on the next prompt.
    /// Geodesic Code Synthesis verifier for topological/oracle post-generation checks.
    /// When enabled, generated code is verified for structural invariants (Betti numbers,
    /// convergence prediction) and violations are fed back as hard constraints on the next prompt.
    #[cfg(feature = "geodesic_synthesis")]
    geodesic_verifier: Option<symthaea_geodesic::GeodesicSynthesizer>,
    /// Cached GCS violations from the last verification pass, injected into the next prompt.
    gcs_violations: Vec<String>,
    /// Unified code orchestrator (CodeGenerator + analogy + LLM, all compiler-verified
    /// before acceptance). Populated only when `config.use_orchestrator` is true — see
    /// `try_orchestrator_generation()` in `generation.rs` for the live call site.
    #[cfg(feature = "code_generation")]
    orchestrator: Option<CodeOrchestrator>,
    /// MAGI world-prediction bridge: predicts compile/test success before each
    /// orchestrator attempt and resolves against the actual outcome afterward,
    /// calibrating confidence over time via Brier scoring. Populated alongside
    /// `orchestrator` (both gated on `config.use_orchestrator`).
    #[cfg(feature = "code_generation")]
    magi_bridge: Option<magi_code_bridge::MagiCodeBridge>,
    /// Self-modification engine: observes real compiler-error clusters from the
    /// Fixing phase and (only when `config.enable_self_modification` is set)
    /// hypothesizes new auto-fix rules. Rules are never applied or promoted from
    /// this path — see `generation.rs::observe_errors_for_self_mod()`.
    fix_rule_generator: self_modification::FixRuleGenerator,
}

impl CodingAgent {
    /// Create a new coding agent with default cognitive loop config.
    pub fn new(config: CodingAgentConfig) -> anyhow::Result<Self> {
        let cognitive_loop = CognitiveLoopService::new(CognitiveLoopConfig::default())?;
        Ok(Self::with_cognitive_loop(cognitive_loop, config))
    }

    /// Create a coding agent wrapping an existing cognitive loop.
    pub fn with_cognitive_loop(
        mut cognitive_loop: CognitiveLoopService,
        config: CodingAgentConfig,
    ) -> Self {
        // Install motor output bridge if not already present
        if !cognitive_loop.has_motor_bridge() {
            let bridge = if config.enable_real_exec {
                // Real execution: sandbox rooted at working_dir, real commands enabled
                let sandbox = SandboxRoot::at(config.working_dir.clone()).unwrap_or_else(|_| {
                    SandboxRoot::new(&config.sandbox_session).expect("sandbox")
                });
                let policy = PolicyBundle::restrictive();
                let mut bridge = MotorOutputBridge::new(policy, sandbox);
                bridge.enable_real_execution();
                // Lower thresholds for coding tasks — the CfC cold-starts at
                // ~0.06 Phi and FEP motor confidence starts at ~0.1. Without this,
                // motor output is blocked for the first many cycles.
                // Phi 0.05: allows ReadOnly immediately, Reversible at 0.15, Destructive at 0.35
                // Confidence 0.05: FEP hasn't learned coding-specific confidence yet
                Ok(bridge.with_min_phi(0.05).with_min_confidence(0.05))
            } else {
                MotorOutputBridge::with_defaults()
                    .map(|b| b.with_min_phi(0.05).with_min_confidence(0.05))
            };
            if let Ok(bridge) = bridge {
                cognitive_loop.set_motor_output_bridge(bridge);
            }
        }

        // Select dispatcher: real Ollama or simulated, optionally with cloud LLM
        let cloud_backend: Option<std::sync::Arc<dyn crate::language::llm_backend::LLMBackend>> =
            if config.use_cloud_llm {
                crate::language::anthropic_backend::AnthropicBackend::from_env().map(|b| {
                    std::sync::Arc::new(b)
                        as std::sync::Arc<dyn crate::language::llm_backend::LLMBackend>
                })
            } else {
                None
            };
        let dispatcher = if config.use_local_llm {
            use crate::language::llm_backend::OllamaBackend;
            IntelligentDispatcher::new(std::sync::Arc::new(OllamaBackend::new()), cloud_backend)
                .with_energy_budget(100.0)
        } else {
            IntelligentDispatcher::new(
                std::sync::Arc::new(crate::language::llm_backend::SimulatedBackend),
                cloud_backend,
            )
            .with_energy_budget(100.0)
        };

        let experience_store = Self::try_init_experience_store(&config.working_dir);
        #[cfg(feature = "code_generation")]
        let use_orchestrator = config.use_orchestrator;

        Self {
            cognitive_loop,
            config,
            phase: TaskPhase::Understanding,
            iteration: 0,
            phase_failures: 0,
            files_modified: Vec::new(),
            phi_trace: Vec::new(),
            observations: Vec::new(),
            errors: Vec::new(),
            last_test_output: None,
            task: String::new(),
            dispatcher: Some(dispatcher),
            last_dispatch: None,
            generation_tiers: Vec::new(),
            generated_code: None,
            code_context: Vec::new(),
            experience_store,
            failure_patterns: Vec::new(),
            #[cfg(feature = "reasoning_engine")]
            coding_attempts: Vec::new(),
            native_exhausted: false,
            prediction_error_history: Vec::new(),
            confidence_velocity_history: Vec::new(),
            event_sink: None,
            retry_state: RetryState::default(),
            #[cfg(feature = "code_generation")]
            code_memory: None,
            #[cfg(feature = "code_generation")]
            source_cache: std::collections::HashMap::new(),
            current_plan: None,
            energy_budget: 100.0, // default energy budget per run
            attempted_fixes: std::collections::HashSet::new(),
            tests_passed: None,
            dedup_skips: 0,
            quality_rejections: 0,
            consciousness_deferrals: 0,
            stuck_detected: false,
            error_knowledge: error_knowledge::CodeErrorKnowledge::new(),
            #[cfg(feature = "geodesic_synthesis")]
            geodesic_verifier: None,
            gcs_violations: Vec::new(),
            #[cfg(feature = "code_generation")]
            orchestrator: if use_orchestrator {
                Some(CodeOrchestrator::new())
            } else {
                None
            },
            #[cfg(feature = "code_generation")]
            magi_bridge: if use_orchestrator {
                Some(magi_code_bridge::MagiCodeBridge::new())
            } else {
                None
            },
            fix_rule_generator: self_modification::FixRuleGenerator::new(),
        }
    }

    /// Attempt to create an experience store. Tries persistent (disk-backed) first,
    /// falling back to in-memory if that fails. Returns None on total failure.
    fn try_init_experience_store(working_dir: &std::path::Path) -> Option<CodingExperienceStore> {
        let rt = tokio::runtime::Runtime::new().ok()?;

        // Try persistent store at {working_dir}/.symthaea/experience.db
        let db_dir = working_dir.join(".symthaea");
        if std::fs::create_dir_all(&db_dir).is_ok() {
            let db_path = db_dir.join("experience.db");
            if let Ok(store) = rt.block_on(CodingExperienceStore::persistent(
                &db_path.to_string_lossy(),
            )) {
                tracing::info!(
                    target: "symthaea::coding_agent",
                    path = %db_path.display(),
                    "Persistent experience store initialized"
                );
                return Some(store);
            }
        }

        // Fallback to in-memory
        tracing::debug!(
            target: "symthaea::coding_agent",
            "Falling back to in-memory experience store"
        );
        rt.block_on(async { CodingExperienceStore::new().await.ok() })
    }

    /// Run the agent on a coding task. Returns the result when done or max iterations reached.
    pub fn run(&mut self, task: &str) -> AgentResult {
        self.task = task.to_string();
        self.phase = TaskPhase::Understanding;
        self.iteration = 0;
        self.phase_failures = 0;
        self.files_modified.clear();
        self.phi_trace.clear();
        self.observations.clear();
        self.errors.clear();
        self.last_test_output = None;
        self.generated_code = None;
        self.generation_tiers.clear();
        self.failure_patterns.clear();
        self.native_exhausted = false;
        self.prediction_error_history.clear();
        self.confidence_velocity_history.clear();
        self.retry_state = RetryState::default();
        self.current_plan = None;
        self.energy_budget = 100.0;
        self.attempted_fixes.clear();
        self.tests_passed = None;
        self.dedup_skips = 0;
        self.quality_rejections = 0;
        self.consciousness_deferrals = 0;
        self.stuck_detected = false;
        self.gcs_violations.clear();

        // Auto-index the project if CodebaseMemory hasn't been populated yet.
        // This gives the agent codebase-aware context on every run without
        // requiring the caller to explicitly call index_project().
        #[cfg(feature = "code_generation")]
        if self.code_memory.is_none() && self.config.working_dir.exists() {
            match self.index_project(&self.config.working_dir.clone()) {
                Ok((files, funcs, types)) => {
                    if files > 0 {
                        tracing::info!(
                            target: "symthaea::coding_agent",
                            files, funcs, types,
                            "Auto-indexed project for codebase awareness"
                        );
                    }
                }
                Err(e) => {
                    tracing::debug!(
                        target: "symthaea::coding_agent",
                        error = %e,
                        "Auto-index skipped (non-fatal)"
                    );
                }
            }
        }

        // If we have indexed codebase memory, query for source-level context
        #[cfg(feature = "code_generation")]
        if let Some(ref memory) = self.code_memory {
            let context = Self::build_source_context(memory, &self.source_cache, task);
            if !context.is_empty() {
                self.code_context = context;
            }
        }

        tracing::info!(
            target: "symthaea::coding_agent",
            task = task,
            max_iterations = self.config.max_iterations,
            context_entries = self.code_context.len(),
            "Starting coding agent"
        );

        // Warm up the CfC: run a few idle cycles to let Phi rise from cold-start (~0.0)
        // to a usable level. Without this, the first 3-5 real iterations are wasted
        // because motor output is blocked by low Phi.
        self.warm_up_phi(3);

        while self.phase != TaskPhase::Done && self.iteration < self.config.max_iterations {
            // Energy exhaustion guard
            if self.energy_budget <= 0.0 {
                tracing::warn!(
                    target: "symthaea::coding_agent",
                    iteration = self.iteration,
                    phase = %self.phase,
                    "Energy budget exhausted — terminating early"
                );
                self.observations
                    .push("Energy budget exhausted — terminating".into());
                self.phase = TaskPhase::Done;
                break;
            }

            self.step();
            self.iteration += 1;
        }

        // If we ran out of iterations, mark as done
        if self.phase != TaskPhase::Done {
            tracing::warn!(
                target: "symthaea::coding_agent",
                iterations = self.iteration,
                phase = %self.phase,
                "Max iterations reached"
            );
        }

        // Flush any pending experience writes (fix strategies, templates) to disk
        self.flush_experience_store();

        let result = self.build_result();
        self.emit_event(AgentEvent::Done(result.clone()));
        result
    }

    /// Execute one step of the agent loop.
    fn step(&mut self) {
        // 0. Fast-fail: if native is exhausted and we've burned enough iterations
        // without producing any code, stop. This catches "no LLM available" scenarios.
        // Do NOT fast-fail if a real LLM is configured — let the escalation path work.
        if self.native_exhausted
            && self.generated_code.is_none()
            && self.iteration >= 5
            && !self.config.use_local_llm
        {
            tracing::info!(
                target: "symthaea::coding_agent",
                task = %self.task,
                iteration = self.iteration,
                "Fast-fail: native exhausted, no code produced after {} iterations",
                self.iteration
            );
            self.observations
                .push("Fast-fail: task beyond native capability, no usable code generated".into());
            self.phase = TaskPhase::Done;
            return;
        }

        // 1. FEP plan selection: generate candidate plans, select best via
        //    free-energy minimization, then evaluate safety/budget.
        if let Some((plan, profile)) = self.select_plan_fep() {
            let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);
            let (approved, reason) = self.evaluate_plan(&plan, current_phi);

            if !approved {
                tracing::warn!(
                    target: "symthaea::coding_agent",
                    phase = %self.phase,
                    reason = %reason,
                    "Plan rejected — skipping step"
                );
                self.observations.push(format!("Plan rejected: {reason}"));
            } else {
                tracing::debug!(
                    target: "symthaea::coding_agent",
                    phase = %self.phase,
                    steps = profile.step_count,
                    energy = profile.total_energy,
                    "FEP plan approved"
                );
                self.current_plan = Some(profile.clone());
                self.deduct_energy(&profile);
            }
        }

        // 2. Phase-specific pre-cycle action (code generation, etc.)
        self.pre_cycle_action();

        // 3. Build observation with updated context
        let observation = self.build_observation();

        // 4. Set up the motor request based on current phase
        let motor_request = self.build_motor_request();
        self.cognitive_loop.set_motor_request(motor_request);

        // 4.5. Inject code-specific signals into the cognitive loop.
        #[cfg(feature = "reasoning_engine")]
        self.inject_code_signals();

        // 5. Run one cognitive cycle
        let cycle_result = self.cognitive_loop.cycle(&observation);

        // 5.5. Extract reasoning feedback — defer or diagnose if needed.
        let reasoning = consciousness_bridge::ReasoningFeedback::from_cycle_result(&cycle_result);
        if reasoning.should_defer() && self.phase == TaskPhase::Generating {
            self.consciousness_deferrals += 1;
            self.observations.push(format!(
                "Reasoning deferral: confidence={:.2}",
                reasoning.reasoning_confidence
            ));
            return;
        }
        if reasoning.should_diagnose() && self.phase == TaskPhase::Generating {
            self.observations.push(format!(
                "Reasoning diagnosis: confidence={:.2}",
                reasoning.reasoning_confidence
            ));
            self.phase = TaskPhase::Understanding;
            return;
        }

        // Extract consciousness signals for decision-making
        let signals = self.extract_consciousness_signals(&cycle_result);
        let phi = signals.phi;
        self.phi_trace.push(phi);
        self.prediction_error_history.push(signals.prediction_error);
        self.confidence_velocity_history
            .push(signals.confidence_velocity);
        // Keep histories bounded
        if self.prediction_error_history.len() > 10 {
            self.prediction_error_history.remove(0);
        }
        if self.confidence_velocity_history.len() > 10 {
            self.confidence_velocity_history.remove(0);
        }
        self.emit_event(AgentEvent::ConsciousnessSnapshot {
            phi,
            prediction_error: signals.prediction_error,
            confidence_velocity: signals.confidence_velocity,
        });

        // 6. Check for motor output result
        let motor_result = self.cognitive_loop.take_motor_result();

        // 7. Process the cycle result and motor output
        let phase_before = self.phase.clone();
        self.process_step_result(&cycle_result, motor_result, phi);
        if self.phase != phase_before {
            self.emit_phase_transition(&phase_before, &self.phase.clone());
        }
    }

    // -- Pre-Cycle Actions --

    /// Phase-specific actions performed before the cognitive cycle.
    ///
    /// Understanding and Testing now use molecule-driven execution — all I/O
    /// flows through MoleculeExecutor with energy tracking, phi gating, and
    /// trace recording.
    ///
    /// - Understanding: gathers context via molecule (ReadFile, ListDir, experience hints)
    /// - Generating/Fixing: calls IntelligentDispatcher to generate code
    // Can't collapse the Fixing arm's inner `if` into a match guard: guards only
    // get an immutable borrow of the scrutinee, but try_structured_auto_fix()
    // needs &mut self.
    #[allow(clippy::collapsible_if, clippy::collapsible_match)]
    fn pre_cycle_action(&mut self) {
        match self.phase {
            TaskPhase::Understanding => {
                self.do_understanding_molecule();
            }
            TaskPhase::Fixing => {
                // In Fixing phase, try structured auto-fix BEFORE calling the LLM.
                // If the fix succeeds, skip LLM entirely (saves energy).
                if !self.try_structured_auto_fix() {
                    self.do_generation();
                }
            }
            TaskPhase::Generating => {
                self.do_generation();
            }
            _ => {}
        }
    }

    /// Inject code-specific signals into the cognitive loop before a cycle.
    #[cfg(feature = "reasoning_engine")]
    fn inject_code_signals(&mut self) {
        let signals = consciousness_bridge::CodeSignals::from_agent_state(
            &self.failure_patterns,
            self.iteration,
            self.phase_failures,
            self.generated_code.as_deref(),
            self.energy_budget,
            100.0,
            self.native_exhausted,
        );

        #[cfg(feature = "reasoning_engine")]
        {
            self.cognitive_loop
                .inject_code_context(signals.to_reasoning_context());
        }

        self.cognitive_loop.set_broca_code_channels(
            signals.syntax_complexity,
            signals.type_confidence as f32,
            signals.algorithm_pattern,
            signals.error_likelihood,
        );
    }

    /// Warm up the CfC by running idle cognitive cycles.
    ///
    /// The CfC cold-starts with Phi ~0.0. Running a few cycles with the task
    /// description as input lets the hidden state accumulate temporal context,
    /// raising Phi to a usable level before real work begins. These warm-up
    /// cycles don't consume agent iterations.
    fn warm_up_phi(&mut self, cycles: usize) {
        let observation = format!("WARM_UP: preparing to work on: {}", self.task);
        for i in 0..cycles {
            let result = self.cognitive_loop.cycle(&observation);
            let phi = result.metadata.consciousness.consciousness_level;
            tracing::debug!(
                target: "symthaea::coding_agent",
                cycle = i,
                phi = phi,
                "Warm-up cycle"
            );
        }
    }

    /// Flush pending writes in the experience store to the database.
    ///
    /// Fix strategies and learned templates are queued during the run via
    /// `queue_persist()`. This ensures they reach disk before the agent exits.
    fn flush_experience_store(&mut self) {
        if let Some(ref mut store) = self.experience_store {
            match tokio::runtime::Handle::try_current() {
                Ok(handle) => {
                    tokio::task::block_in_place(|| handle.block_on(store.flush()));
                }
                Err(_) => {
                    if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        rt.block_on(store.flush());
                    }
                }
            }
        }
    }
}

// NOTE: tests.rs has pre-existing broken imports (384 compile errors as of
// 2026-07-05 — E0422/E0425/E0433, stale references to renamed/removed APIs
// accumulated while this file was disconnected from compilation; a real fix
// is a substantial standalone effort, not a quick unblock). Tests live in
// submodules instead — see docs/CODE_ABILITY_IMPROVEMENT_PLAN.md.
// #[cfg(test)]
// #[path = "tests.rs"]
// mod tests;
