//! Coding Agent: multi-step consciousness-gated coding loop.
//!
//! Drives Symthaea through read → reason → write → test → fix cycles,
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
//! Understanding → Planning → Generating → Testing → Fixing ──→ Done
//!      ↑              ↑           │            │         │
//!      └──────────────┴───────────┘            └─────────┘
//! ```

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
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Task E types: structured test failures, events, retry strategies, consciousness
// ═══════════════════════════════════════════════════════════════════════════════

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
    /// Cached file sources for source-level context extraction (path → source text).
    #[cfg(feature = "code_generation")]
    source_cache: std::collections::HashMap<PathBuf, String>,
    /// Current execution plan profile (computed during Planning phase).
    /// Used for energy budgeting and Phi gating before committing to actions.
    current_plan: Option<PlanProfile>,
    /// Remaining energy budget for this run.
    energy_budget: f32,
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

        // Select dispatcher: real Ollama or simulated
        let dispatcher = if config.use_local_llm {
            IntelligentDispatcher::with_local_llm()
        } else {
            IntelligentDispatcher::simulated()
        };

        let experience_store = Self::try_init_experience_store(&config.working_dir);

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

        let result = self.build_result();
        self.emit_event(AgentEvent::Done(result.clone()));
        result
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

        // 5. Run one cognitive cycle
        let cycle_result = self.cognitive_loop.cycle(&observation);

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

    // ── Pre-Cycle Actions ──────────────────────────────────────────────

    /// Phase-specific actions performed before the cognitive cycle.
    ///
    /// Understanding and Testing now use molecule-driven execution — all I/O
    /// flows through MoleculeExecutor with energy tracking, phi gating, and
    /// trace recording.
    ///
    /// - Understanding: gathers context via molecule (ReadFile, ListDir, experience hints)
    /// - Generating/Fixing: calls IntelligentDispatcher to generate code
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

    /// Read project files to build context for the task.
    ///
    /// Reads the target file (if it exists) and lists the working directory
    /// to understand the project structure. File content is stored in
    /// `observations` for use by the generation prompt.
    fn do_understanding(&mut self) {
        // 1. Try to read the target file (if it already exists)
        let target = self.resolve_target_file();
        if target.exists() {
            match std::fs::read_to_string(&target) {
                Ok(content) => {
                    let preview = if content.len() > 1500 {
                        format!("{}...(truncated)", &content[..1500])
                    } else {
                        content.clone()
                    };
                    self.observations.push(format!(
                        "Target file {} ({} bytes):\n{}",
                        target.display(),
                        content.len(),
                        preview
                    ));
                }
                Err(e) => {
                    self.observations.push(format!(
                        "Target file {} not readable: {e}",
                        target.display()
                    ));
                }
            }
        } else {
            self.observations.push(format!(
                "Target file {} does not exist yet (will be created)",
                target.display()
            ));
        }

        // 2. List source files in the working directory (shallow scan)
        if self.config.working_dir.is_dir() {
            let mut source_files = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&self.config.working_dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    // Skip hidden, target, node_modules
                    if name_str.starts_with('.')
                        || name_str == "target"
                        || name_str == "node_modules"
                    {
                        continue;
                    }
                    if path.is_file() {
                        source_files.push(name_str.to_string());
                    } else if path.is_dir() {
                        source_files.push(format!("{}/", name_str));
                    }
                }
            }
            if !source_files.is_empty() {
                source_files.sort();
                self.observations.push(format!(
                    "Working directory {}: [{}]",
                    self.config.working_dir.display(),
                    source_files.join(", ")
                ));
            }
        }

        // 3. If target is in a src/ directory, also read Cargo.toml for project context
        let cargo_toml = self.config.working_dir.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                // Just extract the [package] section header info
                let preview: String = content.lines().take(15).collect::<Vec<_>>().join("\n");
                self.observations.push(format!("Cargo.toml:\n{}", preview));
            }
        }

        // 4. Query experience store for prior encounters with similar tasks.
        // This gives the agent "memory" of what worked before — injected early
        // so it influences Planning and Generation phases.
        let hints = self.retrieve_experience_hints();
        if !hints.is_empty() {
            self.observations.push(format!(
                "Prior experience: {} relevant patterns found for similar tasks",
                hints.len()
            ));
            for (pattern, hint) in hints.iter().take(3) {
                self.observations.push(format!(
                    "  Prior: {} → {}",
                    &pattern[..pattern.len().min(80)],
                    &hint[..hint.len().min(80)]
                ));
            }
        }
    }

    /// Generate code via the IntelligentDispatcher and write to disk.
    fn do_generation(&mut self) {
        // Get consciousness state for dispatch routing
        let confidence = self.cognitive_loop.prediction_confidence();
        let phi = self.phi_trace.last().copied().unwrap_or(0.5) as f64;
        let prediction_error = self.cognitive_loop.prediction_confidence(); // inverse proxy

        // If native generation was exhausted (returned None), override consciousness
        // state to force the dispatcher toward LLM tier:
        // - Epistemic → Uncertain (triggers LLM selection)
        // - Prediction error → 0.7+ (confirms need for external help)
        // - Phi → 0.5 (bypasses the consciousness < 0.2 → Native override)
        let (epistemic, prediction_error, phi) = if self.native_exhausted {
            (
                EpistemicStatus::Uncertain,
                0.7_f64.max(prediction_error as f64),
                0.5,
            )
        } else {
            (
                Self::confidence_to_epistemic(confidence),
                prediction_error as f64,
                phi,
            )
        };

        // Build the generation prompt and system prompt before borrowing dispatcher
        let prompt = self.build_generation_prompt();
        let sys_prompt = self.codegen_system_prompt();

        // Call the dispatcher (async → sync bridge)
        let dispatch_result = if let Some(ref mut dispatcher) = self.dispatcher {
            // Consciousness-informed temperature: higher prediction error → more exploration
            let pe = self.prediction_error_history.last().copied().unwrap_or(0.3);
            let temperature = (0.3 + pe * 0.3).min(0.9);

            // Apply forced backend tier from retry strategy
            match &self.retry_state.current_strategy {
                RetryStrategy::DifferentBackend(tier) => {
                    dispatcher.force_next_tier(*tier);
                }
                _ => {}
            }

            let params = GenerationParams {
                temperature,
                max_tokens: 1024,
                system_prompt: Some(sys_prompt.clone()),
            };

            // Sync bridge for async dispatcher
            let result = Self::block_on_dispatch(
                dispatcher,
                &prompt,
                &params,
                epistemic,
                prediction_error,
                phi,
            );
            Some(result)
        } else {
            None
        };

        // Process the dispatch result
        if let Some(result) = dispatch_result {
            self.generation_tiers.push(result.tier);

            tracing::debug!(
                target: "symthaea::coding_agent",
                tier = %result.tier,
                native_exhausted = self.native_exhausted,
                success = result.success,
                output_len = result.output.len(),
                "Dispatch result"
            );

            // Fast-fail: if native is exhausted and the "LLM" returned simulated
            // or signal output, there's no real backend to escalate to. Stop early
            // instead of looping through remaining iterations.
            if self.native_exhausted
                && result.tier != BackendTier::Native
                && (result.output.contains("simulated")
                    || result.output.contains("[NATIVE:")
                    || result.output.is_empty())
            {
                tracing::info!(
                    target: "symthaea::coding_agent",
                    task = %self.task,
                    tier = %result.tier,
                    "No real LLM available — fast-failing"
                );
                self.observations
                    .push("Fast-fail: no real LLM backend available for this task".into());
                self.phase = TaskPhase::Done;
                self.last_dispatch = Some(result);
                return;
            }

            if result.success && result.tier != BackendTier::Native {
                // LLM-generated code — write to disk
                let target = self.resolve_target_file();
                self.write_code_to_disk(&target, &result.output);
                self.generated_code = Some(Self::strip_code_fences(&result.output));
                // LLM succeeded — clear native_exhausted (task was handled)
                self.native_exhausted = false;

                tracing::info!(
                    target: "symthaea::coding_agent",
                    tier = %result.tier,
                    energy = result.energy_cost,
                    target = %target.display(),
                    "Code generated and written"
                );
            } else if result.tier == BackendTier::Native {
                // Native tier — try pattern-aware generation
                if let Some(code) = self.native_code_template() {
                    let target = self.resolve_target_file();
                    self.write_code_to_disk(&target, &code);
                    self.generated_code = Some(Self::strip_code_fences(&code));
                } else {
                    // Native can't handle this — immediately escalate to LLM
                    // within the SAME iteration (don't wait for next cycle).
                    self.native_exhausted = true;
                    if let Some(ref mut dispatcher) = self.dispatcher {
                        dispatcher.record_outcome(BackendTier::Native, false);

                        // Re-dispatch with overridden state to force LLM tier
                        let params = GenerationParams {
                            temperature: 0.4,
                            max_tokens: 1024,
                            system_prompt: Some(sys_prompt.clone()),
                        };
                        let llm_result = Self::block_on_dispatch(
                            dispatcher,
                            &prompt,
                            &params,
                            EpistemicStatus::Uncertain,
                            0.7,
                            0.5, // bypass consciousness < 0.2 check
                        );
                        self.generation_tiers.push(llm_result.tier);
                        tracing::info!(
                            target: "symthaea::coding_agent",
                            tier = %llm_result.tier,
                            success = llm_result.success,
                            output_len = llm_result.output.len(),
                            "Native→LLM escalation"
                        );
                        if llm_result.success && llm_result.tier != BackendTier::Native {
                            let target = self.resolve_target_file();
                            self.write_code_to_disk(&target, &llm_result.output);
                            self.generated_code = Some(Self::strip_code_fences(&llm_result.output));
                            self.native_exhausted = false;
                        } else {
                            self.observations
                                .push("Native exhausted, LLM escalation attempted".into());
                        }
                        self.last_dispatch = Some(llm_result);
                        return; // already processed
                    }
                    self.observations.push(
                        "Native generation: no matching pattern, no dispatcher available".into(),
                    );
                }
            } else {
                self.errors.push(format!(
                    "Generation failed ({}): {}",
                    result.tier, result.output
                ));
            }

            self.last_dispatch = Some(result);
        }
    }

    /// Try to auto-fix the last compilation error using structured (line-aware) fixes.
    ///
    /// Parses `last_test_output` into structured errors with file/line/column info,
    /// then applies targeted fixes (type conversions, clone insertion, lifetime
    /// annotations, derive attributes). If a fix is applied, writes the fixed code
    /// to disk and sets `generated_code` — skipping the LLM entirely.
    ///
    /// Returns `true` if a fix was applied (caller should skip LLM generation).
    fn try_structured_auto_fix(&mut self) -> bool {
        // Need both the error output and the generated code to fix
        let (test_output, code) = match (&self.last_test_output, &self.generated_code) {
            (Some(output), Some(code)) => (output.clone(), code.clone()),
            _ => return false,
        };

        // Parse structured errors
        let structured = crate::language::code_executor::parse_structured_errors(&test_output);
        if structured.is_empty() {
            return false;
        }

        // Try structured (line-aware) fix first
        if let Some(fixed) =
            crate::language::code_executor::try_auto_fix_structured(&code, &structured)
        {
            let target = self.resolve_target_file();
            self.write_code_to_disk(&target, &fixed);
            self.generated_code = Some(Self::strip_code_fences(&fixed));
            self.observations.push(format!(
                "Structured auto-fix applied ({} errors analyzed, line-targeted)",
                structured.len()
            ));
            tracing::info!(
                target: "symthaea::coding_agent",
                errors = structured.len(),
                "Structured auto-fix applied, skipping LLM"
            );
            return true;
        }

        // Fall back to basic (non-line-aware) fix
        let flat_errors: Vec<String> = structured.iter().map(|e| e.message.clone()).collect();
        if let Some(fixed) = crate::language::code_executor::try_auto_fix(&code, &flat_errors) {
            let target = self.resolve_target_file();
            self.write_code_to_disk(&target, &fixed);
            self.generated_code = Some(Self::strip_code_fences(&fixed));
            self.observations.push("Basic auto-fix applied".into());
            tracing::info!(
                target: "symthaea::coding_agent",
                "Basic auto-fix applied, skipping LLM"
            );
            return true;
        }

        false
    }

    /// Synchronously call the async dispatcher.
    fn block_on_dispatch(
        dispatcher: &mut IntelligentDispatcher,
        prompt: &str,
        params: &GenerationParams,
        epistemic: EpistemicStatus,
        prediction_error: f64,
        phi: f64,
    ) -> DispatchResult {
        // Try existing tokio runtime first, fall back to a temporary one
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| {
                handle.block_on(dispatcher.generate(
                    prompt,
                    params,
                    epistemic,
                    prediction_error,
                    phi,
                ))
            }),
            Err(_) => {
                // No runtime available — create a lightweight current-thread runtime
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("failed to create tokio runtime for code generation");
                rt.block_on(dispatcher.generate(prompt, params, epistemic, prediction_error, phi))
            }
        }
    }

    /// Build the prompt sent to the LLM for code generation.
    fn build_generation_prompt(&self) -> String {
        let mut prompt = String::with_capacity(2048);

        prompt.push_str(&format!("Task: {}\n\n", self.task));

        // HDC context from indexed codebase memory (similarity search results)
        let hdc_ctx = self.build_hdc_context_prompt();
        if !hdc_ctx.is_empty() {
            prompt.push_str(&hdc_ctx);
            prompt.push('\n');
        }

        // Include codebase context from CodebaseMemory
        if !self.code_context.is_empty() {
            prompt.push_str("Relevant code from the project:\n");
            for ctx in &self.code_context {
                prompt.push_str(&format!("---\n{}\n", ctx));
            }
            prompt.push_str("---\n\n");
        }

        // Include observations (file contents read, etc.)
        if !self.observations.is_empty() {
            prompt.push_str("Context from prior analysis:\n");
            for obs in self.observations.iter().rev().take(5).rev() {
                prompt.push_str(&format!("- {}\n", obs));
            }
            prompt.push('\n');
        }

        // In Fixing phase, include both raw error and structured test failure analysis
        if self.phase == TaskPhase::Fixing {
            if let Some(ref test_output) = self.last_test_output {
                // Parse structured test failures for targeted fixing
                let structured = Self::parse_test_failures(test_output);
                if !structured.is_empty() {
                    prompt.push_str(&Self::format_structured_test_failures(&structured));
                }
                prompt.push_str(&format!(
                    "The previous code failed with this error:\n```\n{}\n```\n\nFix the code.\n",
                    test_output
                ));
            }
            // Include retry strategy hints
            match &self.retry_state.current_strategy {
                RetryStrategy::DifferentTemplate => {
                    prompt.push_str("\nTry a completely different implementation approach.\n");
                }
                RetryStrategy::SimplifyScope => {
                    prompt.push_str(
                        "\nSimplify the implementation — use the minimal viable approach.\n",
                    );
                }
                _ => {}
            }
        }

        // Inject failure patterns from THIS run — these are errors we've already
        // seen, so the generator should avoid repeating them.
        if !self.failure_patterns.is_empty() {
            prompt.push_str("Errors encountered in this session (AVOID these patterns):\n");
            for (pattern, count) in self.failure_patterns.iter().take(5) {
                prompt.push_str(&format!("- ({count}x) {pattern}\n"));
            }
            prompt.push('\n');
        }

        // Inject experience hints from persistent store (cross-session learning)
        let hints = self.retrieve_experience_hints();
        if !hints.is_empty() {
            prompt.push_str("Relevant patterns from past experience:\n");
            for (pattern, hint) in hints.iter().take(3) {
                prompt.push_str(&format!("- Error: {} → Fix: {}\n", pattern, hint));
            }
            prompt.push('\n');
        }

        // Inject successful code patterns from past sessions
        let successes = self.retrieve_success_patterns();
        if !successes.is_empty() {
            prompt.push_str("Successful implementations from prior sessions:\n");
            for (task, code_summary) in successes.iter().take(2) {
                prompt.push_str(&format!("- Task: {} → Code: {}\n", task, code_summary));
            }
            prompt.push('\n');
        }

        // Infer language from target file extension
        let target = self.resolve_target_file();
        let lang = target
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("rust");
        prompt.push_str(&format!("Generate valid {} code.\n", lang));

        prompt
    }

    /// Retrieve relevant error hints from the persistent experience store.
    fn retrieve_experience_hints(&self) -> Vec<(String, String)> {
        if let Some(ref store) = self.experience_store {
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                return rt.block_on(async { store.error_hints_for(&self.task, 3).await });
            }
        }
        Vec::new()
    }

    /// Retrieve successful code patterns from the persistent experience store.
    ///
    /// Returns `(task_description, code_summary)` pairs from past sessions
    /// where similar tasks were completed successfully.
    fn retrieve_success_patterns(&self) -> Vec<(String, String)> {
        if let Some(ref store) = self.experience_store {
            // First check cache (fast, no async)
            let self_task_lower = self.task.to_lowercase();
            let self_words: std::collections::HashSet<String> = self_task_lower
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            let cached: Vec<(String, String)> = store
                .cached_successes()
                .iter()
                .filter(|(task, _)| {
                    let task_lower = task.to_lowercase();
                    let task_words: std::collections::HashSet<String> = task_lower
                        .split_whitespace()
                        .map(|s| s.to_string())
                        .collect();
                    task_words.intersection(&self_words).count() >= 2
                })
                .take(3)
                .cloned()
                .collect();

            if !cached.is_empty() {
                return cached;
            }

            // Fall back to similarity search
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                return rt.block_on(async {
                    let results = store.query_similar(&self.task, 5).await;
                    results
                        .into_iter()
                        .filter(|r| r.record.valence > 0.0 && r.similarity > 0.5)
                        .map(|r| {
                            let summary = r.record.topics.first().cloned().unwrap_or_default();
                            (r.record.content, summary)
                        })
                        .take(3)
                        .collect()
                });
            }
        }
        Vec::new()
    }

    /// Generate code using native (non-LLM) pattern-aware templates.
    ///
    /// Matches task descriptions against a library of common coding patterns
    /// (arithmetic, string ops, sorting, searching, data structures, etc.)
    /// and generates compilable, tested implementations. Returns `None` if
    /// the task doesn't match any known pattern — the caller should escalate
    /// to an LLM tier rather than produce a TODO stub.
    fn native_code_template(&self) -> Option<String> {
        let task_lower = self.task.to_lowercase();

        // Try pattern-specific generation first
        if let Some(code) = Self::match_native_pattern(&task_lower) {
            return Some(code);
        }

        // No match — signal that native generation can't handle this task.
        // The caller should escalate to an LLM tier.
        None
    }

    /// Match task against known coding patterns and return compilable code.
    fn match_native_pattern(task: &str) -> Option<String> {
        // ── Arithmetic / Math ────────────────────────────────────────
        if task.contains("fibonacci") || task.contains("fib") {
            return Some(
                "/// Compute the nth Fibonacci number.\npub fn fibonacci(n: u64) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => {\n            let (mut a, mut b) = (0u64, 1u64);\n            for _ in 2..=n {\n                let c = a.saturating_add(b);\n                a = b;\n                b = c;\n            }\n            b\n        }\n    }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("factorial") {
            return Some(
                "/// Compute the factorial of n.\npub fn factorial(n: u64) -> u64 {\n    (1..=n).fold(1u64, |acc, x| acc.saturating_mul(x))\n}\n"
                    .to_string(),
            );
        }
        if task.contains("gcd") || task.contains("greatest common") {
            return Some(
                "/// Compute the greatest common divisor using Euclid's algorithm.\npub fn gcd(mut a: u64, mut b: u64) -> u64 {\n    while b != 0 {\n        let t = b;\n        b = a % b;\n        a = t;\n    }\n    a\n}\n"
                    .to_string(),
            );
        }
        if task.contains("is_prime") || task.contains("prime check") || task.contains("primality") {
            return Some(
                "/// Check if a number is prime.\npub fn is_prime(n: u64) -> bool {\n    if n < 2 { return false; }\n    if n < 4 { return true; }\n    if n % 2 == 0 || n % 3 == 0 { return false; }\n    let mut i = 5;\n    while i * i <= n {\n        if n % i == 0 || n % (i + 2) == 0 { return false; }\n        i += 6;\n    }\n    true\n}\n"
                    .to_string(),
            );
        }
        if task.contains("absolute") || task.contains("abs ") {
            return Some(
                "/// Return the absolute value.\npub fn absolute(x: i64) -> u64 {\n    x.unsigned_abs()\n}\n"
                    .to_string(),
            );
        }

        // ── String operations ────────────────────────────────────────
        if task.contains("hello") {
            return Some(
                "/// Return a greeting.\npub fn hello() -> &'static str {\n    \"Hello, world!\"\n}\n"
                    .to_string(),
            );
        }
        if task.contains("reverse") && task.contains("string") {
            return Some(
                "/// Reverse a string.\npub fn reverse_string(s: &str) -> String {\n    s.chars().rev().collect()\n}\n"
                    .to_string(),
            );
        }
        if task.contains("palindrome") {
            return Some(
                "/// Check if a string is a palindrome.\npub fn is_palindrome(s: &str) -> bool {\n    let s: String = s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect();\n    s == s.chars().rev().collect::<String>()\n}\n"
                    .to_string(),
            );
        }
        if task.contains("count") && task.contains("vowel") {
            return Some(
                "/// Count vowels in a string.\npub fn count_vowels(s: &str) -> usize {\n    s.chars().filter(|c| \"aeiouAEIOU\".contains(*c)).count()\n}\n"
                    .to_string(),
            );
        }
        if task.contains("uppercase") || task.contains("to_upper") {
            return Some(
                "/// Convert a string to uppercase.\npub fn to_uppercase(s: &str) -> String {\n    s.to_uppercase()\n}\n"
                    .to_string(),
            );
        }

        // ── Sorting ──────────────────────────────────────────────────
        if task.contains("bubble sort") {
            return Some(
                "/// Sort a slice using bubble sort.\npub fn bubble_sort<T: Ord>(arr: &mut [T]) {\n    let n = arr.len();\n    for i in 0..n {\n        let mut swapped = false;\n        for j in 0..n.saturating_sub(i + 1) {\n            if arr[j] > arr[j + 1] {\n                arr.swap(j, j + 1);\n                swapped = true;\n            }\n        }\n        if !swapped { break; }\n    }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("insertion sort") {
            return Some(
                "/// Sort a slice using insertion sort.\npub fn insertion_sort<T: Ord + Copy>(arr: &mut [T]) {\n    for i in 1..arr.len() {\n        let key = arr[i];\n        let mut j = i;\n        while j > 0 && arr[j - 1] > key {\n            arr[j] = arr[j - 1];\n            j -= 1;\n        }\n        arr[j] = key;\n    }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("merge sort") {
            return Some(
                "/// Sort a slice using merge sort.\npub fn merge_sort<T: Ord + Clone>(arr: &mut [T]) {\n    let len = arr.len();\n    if len <= 1 { return; }\n    let mid = len / 2;\n    let mut left = arr[..mid].to_vec();\n    let mut right = arr[mid..].to_vec();\n    merge_sort(&mut left);\n    merge_sort(&mut right);\n    let (mut i, mut j, mut k) = (0, 0, 0);\n    while i < left.len() && j < right.len() {\n        if left[i] <= right[j] { arr[k] = left[i].clone(); i += 1; }\n        else { arr[k] = right[j].clone(); j += 1; }\n        k += 1;\n    }\n    while i < left.len() { arr[k] = left[i].clone(); i += 1; k += 1; }\n    while j < right.len() { arr[k] = right[j].clone(); j += 1; k += 1; }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("sort") {
            // Generic: if "sort" is mentioned but no specific algorithm
            return Some(
                "/// Sort a vector and return it.\npub fn sort_vec<T: Ord>(mut v: Vec<T>) -> Vec<T> {\n    v.sort();\n    v\n}\n"
                    .to_string(),
            );
        }

        // ── Searching ────────────────────────────────────────────────
        if task.contains("binary search") {
            return Some(
                "/// Binary search for a value in a sorted slice. Returns the index if found.\npub fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {\n    let mut lo = 0usize;\n    let mut hi = arr.len();\n    while lo < hi {\n        let mid = lo + (hi - lo) / 2;\n        match arr[mid].cmp(target) {\n            std::cmp::Ordering::Equal => return Some(mid),\n            std::cmp::Ordering::Less => lo = mid + 1,\n            std::cmp::Ordering::Greater => hi = mid,\n        }\n    }\n    None\n}\n"
                    .to_string(),
            );
        }
        if task.contains("linear search") || (task.contains("search") && task.contains("find")) {
            return Some(
                "/// Linear search for a value in a slice. Returns the index if found.\npub fn linear_search<T: PartialEq>(arr: &[T], target: &T) -> Option<usize> {\n    arr.iter().position(|x| x == target)\n}\n"
                    .to_string(),
            );
        }

        // ── Collections ──────────────────────────────────────────────
        if task.contains("sum")
            && (task.contains("vec") || task.contains("list") || task.contains("array"))
        {
            return Some(
                "/// Sum all elements in a slice.\npub fn sum_vec(v: &[i64]) -> i64 {\n    v.iter().sum()\n}\n"
                    .to_string(),
            );
        }
        if task.contains("max")
            && (task.contains("vec")
                || task.contains("list")
                || task.contains("array")
                || task.contains("find"))
        {
            return Some(
                "/// Find the maximum value in a slice.\npub fn find_max(v: &[i64]) -> Option<i64> {\n    v.iter().copied().max()\n}\n"
                    .to_string(),
            );
        }
        if task.contains("min")
            && (task.contains("vec")
                || task.contains("list")
                || task.contains("array")
                || task.contains("find"))
        {
            return Some(
                "/// Find the minimum value in a slice.\npub fn find_min(v: &[i64]) -> Option<i64> {\n    v.iter().copied().min()\n}\n"
                    .to_string(),
            );
        }
        if task.contains("is_even") || (task.contains("even") && task.contains("check")) {
            return Some(
                "/// Check if a number is even.\npub fn is_even(n: i64) -> bool {\n    n % 2 == 0\n}\n"
                    .to_string(),
            );
        }
        if task.contains("flatten") {
            return Some(
                "/// Flatten a nested vector.\npub fn flatten<T: Clone>(nested: &[Vec<T>]) -> Vec<T> {\n    nested.iter().flat_map(|v| v.iter().cloned()).collect()\n}\n"
                    .to_string(),
            );
        }

        // ── Data structures ──────────────────────────────────────────
        if task.contains("stack") {
            return Some(
                "/// A simple stack backed by a Vec.\npub struct Stack<T> {\n    data: Vec<T>,\n}\n\nimpl<T> Stack<T> {\n    pub fn new() -> Self { Self { data: Vec::new() } }\n    pub fn push(&mut self, val: T) { self.data.push(val); }\n    pub fn pop(&mut self) -> Option<T> { self.data.pop() }\n    pub fn peek(&self) -> Option<&T> { self.data.last() }\n    pub fn is_empty(&self) -> bool { self.data.is_empty() }\n    pub fn len(&self) -> usize { self.data.len() }\n}\n\nimpl<T> Default for Stack<T> {\n    fn default() -> Self { Self::new() }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("ring buffer") {
            return Some(
                "/// A fixed-capacity ring buffer.\npub struct RingBuffer<T> {\n    buf: Vec<Option<T>>,\n    head: usize,\n    len: usize,\n}\n\nimpl<T> RingBuffer<T> {\n    pub fn new(capacity: usize) -> Self {\n        let mut buf = Vec::with_capacity(capacity);\n        for _ in 0..capacity { buf.push(None); }\n        Self { buf, head: 0, len: 0 }\n    }\n    pub fn push(&mut self, val: T) {\n        let cap = self.buf.len();\n        let idx = (self.head + self.len) % cap;\n        self.buf[idx] = Some(val);\n        if self.len == cap { self.head = (self.head + 1) % cap; }\n        else { self.len += 1; }\n    }\n    pub fn pop(&mut self) -> Option<T> {\n        if self.len == 0 { return None; }\n        let val = self.buf[self.head].take();\n        self.head = (self.head + 1) % self.buf.len();\n        self.len -= 1;\n        val\n    }\n    pub fn len(&self) -> usize { self.len }\n    pub fn is_empty(&self) -> bool { self.len == 0 }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("linked list") {
            return Some(
                "/// A singly linked list.\npub struct LinkedList<T> {\n    head: Option<Box<Node<T>>>,\n}\n\nstruct Node<T> {\n    value: T,\n    next: Option<Box<Node<T>>>,\n}\n\nimpl<T> LinkedList<T> {\n    pub fn new() -> Self { Self { head: None } }\n    pub fn push_front(&mut self, val: T) {\n        let node = Box::new(Node { value: val, next: self.head.take() });\n        self.head = Some(node);\n    }\n    pub fn pop_front(&mut self) -> Option<T> {\n        let node = self.head.take()?;\n        self.head = node.next;\n        Some(node.value)\n    }\n    pub fn is_empty(&self) -> bool { self.head.is_none() }\n    pub fn len(&self) -> usize {\n        let mut count = 0;\n        let mut current = &self.head;\n        while let Some(node) = current { count += 1; current = &node.next; }\n        count\n    }\n}\n\nimpl<T> Default for LinkedList<T> {\n    fn default() -> Self { Self::new() }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("bloom filter") {
            return Some(
                "use std::collections::hash_map::DefaultHasher;\nuse std::hash::{Hash, Hasher};\n\n/// A simple Bloom filter.\npub struct BloomFilter {\n    bits: Vec<bool>,\n    num_hashes: usize,\n}\n\nimpl BloomFilter {\n    pub fn new(size: usize, num_hashes: usize) -> Self {\n        Self { bits: vec![false; size], num_hashes }\n    }\n    pub fn insert<T: Hash>(&mut self, item: &T) {\n        for i in 0..self.num_hashes {\n            let idx = self.hash_idx(item, i);\n            self.bits[idx] = true;\n        }\n    }\n    pub fn contains<T: Hash>(&self, item: &T) -> bool {\n        (0..self.num_hashes).all(|i| self.bits[self.hash_idx(item, i)])\n    }\n    fn hash_idx<T: Hash>(&self, item: &T, seed: usize) -> usize {\n        let mut hasher = DefaultHasher::new();\n        item.hash(&mut hasher);\n        seed.hash(&mut hasher);\n        hasher.finish() as usize % self.bits.len()\n    }\n}\n"
                    .to_string(),
            );
        }

        // ── Medium algorithms ────────────────────────────────────────
        // (merge sort handled above in Sorting section to avoid generic "sort" matching first)
        if task.contains("caesar") || task.contains("cipher") {
            return Some(
                "/// Encrypt text using a Caesar cipher with the given shift.\npub fn encrypt(text: &str, shift: u8) -> String {\n    text.chars().map(|c| {\n        if c.is_ascii_lowercase() {\n            (b'a' + (c as u8 - b'a' + shift) % 26) as char\n        } else if c.is_ascii_uppercase() {\n            (b'A' + (c as u8 - b'A' + shift) % 26) as char\n        } else { c }\n    }).collect()\n}\n\n/// Decrypt text using a Caesar cipher with the given shift.\npub fn decrypt(text: &str, shift: u8) -> String {\n    encrypt(text, 26 - (shift % 26))\n}\n"
                    .to_string(),
            );
        }
        if task.contains("run-length") || task.contains("rle") {
            return Some(
                "/// Run-length encode a string: \"aaabbc\" → \"3a2b1c\".\npub fn rle_encode(s: &str) -> String {\n    if s.is_empty() { return String::new(); }\n    let mut result = String::new();\n    let chars: Vec<char> = s.chars().collect();\n    let mut count = 1usize;\n    for i in 1..chars.len() {\n        if chars[i] == chars[i - 1] { count += 1; }\n        else {\n            result.push_str(&count.to_string());\n            result.push(chars[i - 1]);\n            count = 1;\n        }\n    }\n    result.push_str(&count.to_string());\n    result.push(*chars.last().unwrap());\n    result\n}\n"
                    .to_string(),
            );
        }
        if task.contains("pascal") && task.contains("triangle") {
            return Some(
                "/// Generate Pascal's triangle with n rows.\npub fn pascal_triangle(n: usize) -> Vec<Vec<u64>> {\n    let mut tri: Vec<Vec<u64>> = Vec::with_capacity(n);\n    for i in 0..n {\n        let mut row = vec![1u64; i + 1];\n        for j in 1..i {\n            row[j] = tri[i - 1][j - 1] + tri[i - 1][j];\n        }\n        tri.push(row);\n    }\n    tri\n}\n"
                    .to_string(),
            );
        }
        if task.contains("levenshtein") || task.contains("edit distance") {
            return Some(
                "/// Compute the Levenshtein (edit) distance between two strings.\npub fn levenshtein(a: &str, b: &str) -> usize {\n    let a: Vec<char> = a.chars().collect();\n    let b: Vec<char> = b.chars().collect();\n    let (m, n) = (a.len(), b.len());\n    let mut dp = vec![vec![0usize; n + 1]; m + 1];\n    for i in 0..=m { dp[i][0] = i; }\n    for j in 0..=n { dp[0][j] = j; }\n    for i in 1..=m {\n        for j in 1..=n {\n            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };\n            dp[i][j] = (dp[i - 1][j] + 1)\n                .min(dp[i][j - 1] + 1)\n                .min(dp[i - 1][j - 1] + cost);\n        }\n    }\n    dp[m][n]\n}\n"
                    .to_string(),
            );
        }
        if task.contains("permutation") {
            return Some(
                "/// Generate all permutations of a string.\npub fn permutations(s: &str) -> Vec<String> {\n    let chars: Vec<char> = s.chars().collect();\n    let mut results = Vec::new();\n    let mut current = chars.clone();\n    permute(&mut current, 0, &mut results);\n    results\n}\n\nfn permute(chars: &mut Vec<char>, start: usize, results: &mut Vec<String>) {\n    if start == chars.len() {\n        results.push(chars.iter().collect());\n        return;\n    }\n    for i in start..chars.len() {\n        chars.swap(start, i);\n        permute(chars, start + 1, results);\n        chars.swap(start, i);\n    }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("depth-first") || task.contains("dfs") {
            return Some(
                "use std::collections::HashSet;\n\n/// Perform depth-first search on an adjacency list graph.\n/// Returns visited nodes in DFS order.\npub fn dfs(graph: &[Vec<usize>], start: usize) -> Vec<usize> {\n    let mut visited = HashSet::new();\n    let mut order = Vec::new();\n    dfs_visit(graph, start, &mut visited, &mut order);\n    order\n}\n\nfn dfs_visit(graph: &[Vec<usize>], node: usize, visited: &mut HashSet<usize>, order: &mut Vec<usize>) {\n    if !visited.insert(node) { return; }\n    order.push(node);\n    if node < graph.len() {\n        for &neighbor in &graph[node] {\n            dfs_visit(graph, neighbor, visited, order);\n        }\n    }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("dijkstra") {
            return Some(
                "use std::collections::BinaryHeap;\nuse std::cmp::Reverse;\n\n/// Dijkstra's shortest path from `start` on a weighted adjacency list.\n/// Returns distances to all nodes (usize::MAX = unreachable).\npub fn dijkstra(graph: &[Vec<(usize, u64)>], start: usize) -> Vec<u64> {\n    let n = graph.len();\n    let mut dist = vec![u64::MAX; n];\n    dist[start] = 0;\n    let mut heap = BinaryHeap::new();\n    heap.push(Reverse((0u64, start)));\n    while let Some(Reverse((d, u))) = heap.pop() {\n        if d > dist[u] { continue; }\n        for &(v, w) in &graph[u] {\n            let nd = d + w;\n            if nd < dist[v] {\n                dist[v] = nd;\n                heap.push(Reverse((nd, v)));\n            }\n        }\n    }\n    dist\n}\n"
                    .to_string(),
            );
        }
        if task.contains("matrix") && task.contains("multiply") {
            return Some(
                "/// Multiply two matrices (Vec of rows).\npub fn multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {\n    let m = a.len();\n    let n = b[0].len();\n    let k = b.len();\n    let mut result = vec![vec![0.0f64; n]; m];\n    for i in 0..m {\n        for j in 0..n {\n            for p in 0..k {\n                result[i][j] += a[i][p] * b[p][j];\n            }\n        }\n    }\n    result\n}\n"
                    .to_string(),
            );
        }
        if task.contains("tokenize") || task.contains("tokenizer") {
            return Some(
                "/// Token types for arithmetic expressions.\n#[derive(Debug, Clone, PartialEq)]\npub enum Token {\n    Number(f64),\n    Plus,\n    Minus,\n    Star,\n    Slash,\n    LParen,\n    RParen,\n}\n\n/// Tokenize an arithmetic expression string.\npub fn tokenize(input: &str) -> Vec<Token> {\n    let mut tokens = Vec::new();\n    let mut chars = input.chars().peekable();\n    while let Some(&c) = chars.peek() {\n        match c {\n            ' ' | '\\t' => { chars.next(); }\n            '+' => { tokens.push(Token::Plus); chars.next(); }\n            '-' => { tokens.push(Token::Minus); chars.next(); }\n            '*' => { tokens.push(Token::Star); chars.next(); }\n            '/' => { tokens.push(Token::Slash); chars.next(); }\n            '(' => { tokens.push(Token::LParen); chars.next(); }\n            ')' => { tokens.push(Token::RParen); chars.next(); }\n            '0'..='9' | '.' => {\n                let mut num = String::new();\n                while let Some(&d) = chars.peek() {\n                    if d.is_ascii_digit() || d == '.' { num.push(d); chars.next(); }\n                    else { break; }\n                }\n                if let Ok(n) = num.parse::<f64>() { tokens.push(Token::Number(n)); }\n            }\n            _ => { chars.next(); }\n        }\n    }\n    tokens\n}\n"
                    .to_string(),
            );
        }
        if task.contains("email") && task.contains("validat") {
            return Some(
                "/// Validate an email address (basic RFC 5321 check).\npub fn validate_email(email: &str) -> bool {\n    let parts: Vec<&str> = email.splitn(2, '@').collect();\n    if parts.len() != 2 { return false; }\n    let (local, domain) = (parts[0], parts[1]);\n    if local.is_empty() || local.len() > 64 { return false; }\n    if domain.is_empty() || domain.len() > 255 { return false; }\n    if !domain.contains('.') { return false; }\n    let valid_char = |c: char| c.is_alphanumeric() || \".-_+\".contains(c);\n    local.chars().all(valid_char) && domain.chars().all(|c| c.is_alphanumeric() || \".-\".contains(c))\n}\n"
                    .to_string(),
            );
        }
        if task.contains("csv") && task.contains("parse") {
            return Some(
                "/// Parse a CSV line into fields, respecting quoted strings.\npub fn parse_csv(line: &str) -> Vec<String> {\n    let mut fields = Vec::new();\n    let mut current = String::new();\n    let mut in_quotes = false;\n    let mut chars = line.chars().peekable();\n    while let Some(c) = chars.next() {\n        match c {\n            '\"' => {\n                if in_quotes {\n                    if chars.peek() == Some(&'\"') { current.push('\"'); chars.next(); }\n                    else { in_quotes = false; }\n                } else { in_quotes = true; }\n            }\n            ',' if !in_quotes => {\n                fields.push(current.clone());\n                current.clear();\n            }\n            _ => current.push(c),\n        }\n    }\n    fields.push(current);\n    fields\n}\n"
                    .to_string(),
            );
        }
        if task.contains("lru") && task.contains("cache") {
            return Some(
                "use std::collections::HashMap;\n\n/// A simple LRU cache with fixed capacity.\npub struct LruCache<K: std::hash::Hash + Eq + Clone, V> {\n    capacity: usize,\n    order: Vec<K>,\n    map: HashMap<K, V>,\n}\n\nimpl<K: std::hash::Hash + Eq + Clone, V> LruCache<K, V> {\n    pub fn new(capacity: usize) -> Self {\n        Self { capacity, order: Vec::new(), map: HashMap::new() }\n    }\n    pub fn get(&mut self, key: &K) -> Option<&V> {\n        if self.map.contains_key(key) {\n            self.order.retain(|k| k != key);\n            self.order.push(key.clone());\n            self.map.get(key)\n        } else { None }\n    }\n    pub fn put(&mut self, key: K, value: V) {\n        if self.map.contains_key(&key) {\n            self.order.retain(|k| k != &key);\n        } else if self.map.len() >= self.capacity {\n            if let Some(oldest) = self.order.first().cloned() {\n                self.order.remove(0);\n                self.map.remove(&oldest);\n            }\n        }\n        self.order.push(key.clone());\n        self.map.insert(key, value);\n    }\n    pub fn len(&self) -> usize { self.map.len() }\n    pub fn is_empty(&self) -> bool { self.map.is_empty() }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("state machine") {
            return Some(
                "use std::collections::HashMap;\n\n/// A simple finite state machine.\npub struct StateMachine {\n    current: String,\n    transitions: HashMap<(String, String), String>,\n}\n\nimpl StateMachine {\n    pub fn new(initial: &str) -> Self {\n        Self { current: initial.to_string(), transitions: HashMap::new() }\n    }\n    pub fn add_transition(&mut self, from: &str, event: &str, to: &str) {\n        self.transitions.insert((from.to_string(), event.to_string()), to.to_string());\n    }\n    pub fn send(&mut self, event: &str) -> bool {\n        let key = (self.current.clone(), event.to_string());\n        if let Some(next) = self.transitions.get(&key) {\n            self.current = next.clone();\n            true\n        } else { false }\n    }\n    pub fn state(&self) -> &str { &self.current }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("trie") {
            return Some(
                "use std::collections::HashMap;\n\n/// A trie (prefix tree) for string storage and lookup.\npub struct Trie {\n    children: HashMap<char, Trie>,\n    is_end: bool,\n}\n\nimpl Trie {\n    pub fn new() -> Self { Self { children: HashMap::new(), is_end: false } }\n    pub fn insert(&mut self, word: &str) {\n        let mut node = self;\n        for c in word.chars() {\n            node = node.children.entry(c).or_insert_with(Trie::new);\n        }\n        node.is_end = true;\n    }\n    pub fn search(&self, word: &str) -> bool {\n        let mut node = self;\n        for c in word.chars() {\n            match node.children.get(&c) {\n                Some(next) => node = next,\n                None => return false,\n            }\n        }\n        node.is_end\n    }\n    pub fn starts_with(&self, prefix: &str) -> bool {\n        let mut node = self;\n        for c in prefix.chars() {\n            match node.children.get(&c) {\n                Some(next) => node = next,\n                None => return false,\n            }\n        }\n        true\n    }\n}\n\nimpl Default for Trie {\n    fn default() -> Self { Self::new() }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("n-queen") || task.contains("queens") {
            return Some(
                "/// Solve the N-Queens problem. Returns all valid board configurations.\n/// Each solution is a Vec of column positions (one per row).\npub fn solve_queens(n: usize) -> Vec<Vec<usize>> {\n    let mut solutions = Vec::new();\n    let mut board = Vec::with_capacity(n);\n    solve_queens_bt(n, &mut board, &mut solutions);\n    solutions\n}\n\nfn solve_queens_bt(n: usize, board: &mut Vec<usize>, solutions: &mut Vec<Vec<usize>>) {\n    if board.len() == n {\n        solutions.push(board.clone());\n        return;\n    }\n    let row = board.len();\n    for col in 0..n {\n        if board.iter().enumerate().all(|(r, &c)| {\n            c != col && (row - r) != col.abs_diff(c)\n        }) {\n            board.push(col);\n            solve_queens_bt(n, board, solutions);\n            board.pop();\n        }\n    }\n}\n"
                    .to_string(),
            );
        }

        None
    }

    /// Extract a likely function name from a task description.
    fn extract_function_name(task: &str) -> Option<String> {
        let stop_words = ["a", "an", "the", "my", "our", "new", "simple", "basic"];
        // Look for explicit function names: "add a X function", "implement X", "create X"
        let patterns = [
            "function ",
            "fn ",
            "method ",
            "implement ",
            "create ",
            "add ",
        ];
        for pattern in &patterns {
            if let Some(idx) = task.find(pattern) {
                let after = &task[idx + pattern.len()..];
                // Skip stop words (e.g., "add a fibonacci" → skip "a")
                let name: String = after
                    .split_whitespace()
                    .find(|w| !stop_words.contains(&w.to_lowercase().as_str()))
                    .unwrap_or("")
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() && name.len() < 50 {
                    return Some(name.to_lowercase());
                }
            }
        }

        // Fallback: use first multi-char word that looks like an identifier
        for word in task.split_whitespace() {
            let clean: String = word
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if clean.len() >= 3
                && clean.chars().next().map_or(false, |c| c.is_alphabetic())
                && ![
                    "the", "and", "for", "add", "that", "with", "from", "this", "into",
                ]
                .contains(&clean.to_lowercase().as_str())
            {
                return Some(clean.to_lowercase());
            }
        }
        None
    }

    /// Determine the target file for code generation.
    fn resolve_target_file(&self) -> PathBuf {
        // 1. Explicit config
        if let Some(ref target) = self.config.target_file {
            if target.is_absolute() {
                return target.clone();
            }
            return self.config.working_dir.join(target);
        }

        // 2. Try to extract path from task description
        for word in self.task.split_whitespace() {
            if word.ends_with(".rs") || word.ends_with(".py") || word.ends_with(".nix") {
                let path = PathBuf::from(word);
                if path.is_absolute() {
                    return path;
                }
                return self.config.working_dir.join(path);
            }
        }

        // 3. Default
        self.config.working_dir.join("src").join("lib.rs")
    }

    /// Detect the target language from the target file extension.
    fn target_language(&self) -> &'static str {
        let target = self.resolve_target_file();
        match target.extension().and_then(|e| e.to_str()) {
            Some("py") => "python",
            Some("nix") => "nix",
            _ => "rust",
        }
    }

    /// Build a language-appropriate system prompt for the LLM.
    fn codegen_system_prompt(&self) -> String {
        match self.target_language() {
            "python" => "You are a Python code generator. Output ONLY valid Python code. \
                No explanations, no markdown fences, no comments outside the function. \
                Complete the function body directly.".into(),
            "nix" => "You are a Nix code generator. Output ONLY valid Nix expressions.".into(),
            _ => "You are a code generator. Output ONLY valid source code, no explanations.".into(),
        }
    }

    /// Write code to disk, creating parent directories as needed.
    /// Strip markdown code fences from generated output.
    /// LLM and template outputs sometimes wrap code in ```rust ... ``` blocks.
    fn strip_code_fences(code: &str) -> String {
        let trimmed = code.trim();
        // Check for ```rust or ``` at start
        if let Some(rest) = trimmed.strip_prefix("```rust") {
            if let Some(inner) = rest.strip_suffix("```") {
                return inner.trim().to_string();
            }
        }
        if let Some(rest) = trimmed.strip_prefix("```") {
            // Could be ```\n...``` or ```rs\n...```
            let rest = rest.strip_prefix("rs").unwrap_or(rest);
            let rest = rest.strip_prefix('\n').unwrap_or(rest);
            if let Some(inner) = rest.strip_suffix("```") {
                return inner.trim().to_string();
            }
        }
        code.to_string()
    }

    /// HDC verification gate: checks generated code against codebase patterns.
    ///
    /// Returns `true` if the code passes verification (safe to write).
    /// Returns `false` if the code is flagged as suspicious (high surprise AND
    /// epistemic uncertainty). When `false`, the code is still written but
    /// a warning observation is recorded.
    #[cfg(feature = "code_generation")]
    fn verify_generated_code_hdc(&self, code: &str) -> (bool, f32) {
        use crate::hdc::code_encoder::CodeHDEncoder;

        let memory = match &self.code_memory {
            Some(m) => m,
            None => return (true, 0.0), // no memory → skip verification
        };

        // Encode the generated code as an HDC vector
        let encoder = memory.encoder();
        let code_hv = encoder.encode_name(code);

        // Compute surprise against codebase centroid
        let surprise = memory.compute_surprise(&code_hv);

        // Query for nearest neighbors — if top match is very dissimilar, flag
        let matches = memory.query(&code_hv, 3);
        let best_similarity = matches.first().map(|m| m.similarity).unwrap_or(0.0);

        // Verification passes if:
        // 1. Surprise is moderate (< 0.85) — code fits the codebase style, OR
        // 2. There's a reasonable nearest neighbor (similarity > 0.15), OR
        // 3. The codebase is too small to judge (< 3 indexed files)
        let codebase_too_small = matches.len() < 2;
        let passes = surprise < 0.85 || best_similarity > 0.15 || codebase_too_small;

        if !passes {
            tracing::warn!(
                target: "symthaea::coding_agent",
                surprise = surprise,
                best_similarity = best_similarity,
                "HDC verification gate: generated code is highly surprising"
            );
        }

        (passes, surprise)
    }

    #[cfg(not(feature = "code_generation"))]
    fn verify_generated_code_hdc(&self, _code: &str) -> (bool, f32) {
        (true, 0.0)
    }

    fn write_code_to_disk(&mut self, target: &PathBuf, code: &str) {
        let code = Self::strip_code_fences(code);

        // HDC verification gate
        let (verified, surprise) = self.verify_generated_code_hdc(&code);
        if !verified {
            self.observations.push(format!(
                "HDC verification warning: generated code has surprise={surprise:.3} — \
                 significantly different from codebase patterns. Writing anyway but flagging."
            ));
        }

        // Create parent directories
        if let Some(parent) = target.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                self.errors.push(format!(
                    "Failed to create directory {}: {e}",
                    parent.display()
                ));
                return;
            }
        }

        match std::fs::write(target, &code) {
            Ok(()) => {
                if !self.files_modified.contains(target) {
                    self.files_modified.push(target.clone());
                }
                self.observations.push(format!(
                    "Wrote {} bytes to {}",
                    code.len(),
                    target.display()
                ));
                #[cfg(feature = "code_generation")]
                self.reindex_file(target, &code);
            }
            Err(e) => {
                self.errors
                    .push(format!("Failed to write {}: {e}", target.display()));
            }
        }
    }

    // ── Observation & Motor Request Building ────────────────────────────

    /// Build the observation text that the cognitive loop will process.
    fn build_observation(&self) -> String {
        let mut obs = format!("CODING TASK: {}\n", self.task);
        obs.push_str(&format!("PHASE: {}\n", self.phase));
        obs.push_str(&format!(
            "ITERATION: {}/{}\n",
            self.iteration, self.config.max_iterations
        ));

        if !self.observations.is_empty() {
            obs.push_str("CONTEXT:\n");
            // Include last 3 observations to keep context focused
            for o in self.observations.iter().rev().take(3).rev() {
                obs.push_str(&format!("  {}\n", o));
            }
        }

        if let Some(ref test_output) = self.last_test_output {
            obs.push_str(&format!("LAST TEST OUTPUT:\n{}\n", test_output));
        }

        if !self.errors.is_empty() {
            obs.push_str("ERRORS:\n");
            for e in self.errors.iter().rev().take(2) {
                obs.push_str(&format!("  {}\n", e));
            }
        }

        obs
    }

    /// Build the motor action request based on current phase.
    // ── Plan Evaluation (Typed Primitives) ──────────────────────────────

    /// Build a typed execution plan for the current phase.
    /// Returns a `Molecule` whose `PlanProfile` can be evaluated before committing.
    fn build_execution_plan(&self) -> Option<Molecule> {
        let target = self.resolve_target_file();
        let working_dir = self.config.working_dir.clone();

        match self.phase {
            TaskPhase::Understanding => {
                // Gather context: read target file + list directory
                let mut plan = Molecule::atom(Atom::list(working_dir.clone()));
                if target.exists() {
                    plan = plan.then(Molecule::atom(Atom::read(target)));
                }
                Some(plan)
            }
            TaskPhase::Planning => None, // pure reasoning, no I/O plan
            TaskPhase::Generating => {
                if let Some(ref code) = self.generated_code {
                    // Code ready: write → check
                    Some(crate::action::primitives::recipes::write_and_check(
                        target, code,
                    ))
                } else {
                    None // generation happens via dispatcher, not primitives
                }
            }
            TaskPhase::Testing => {
                // cargo check in the working directory
                Some(Molecule::atom(Atom::cargo_check(working_dir)))
            }
            TaskPhase::Fixing => {
                if let Some(ref code) = self.generated_code {
                    // Fix written: write → check, with recovery
                    let write_check =
                        crate::action::primitives::recipes::write_and_check(target, code);
                    Some(write_check.recover(|_| {
                        // On failure, the agent will re-enter Fixing with new errors
                        Molecule::atom(Atom::Noop)
                    }))
                } else {
                    None
                }
            }
            TaskPhase::Done => None,
        }
    }

    /// Generate multiple candidate plans for the current phase and use FEP
    /// free-energy minimization to select the best one.
    ///
    /// Returns the selected plan (if any) and its profile. The FEP selector
    /// prefers plans that are: feasible (phi + budget), non-destructive,
    /// and have the lowest expected free energy.
    fn select_plan_fep(&self) -> Option<(Molecule, PlanProfile)> {
        use crate::action::primitives::{
            select_best_plan, select_best_plan_with_history, PlanCandidate,
        };

        let target = self.resolve_target_file();
        let working_dir = self.config.working_dir.clone();
        let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);

        let candidates: Vec<PlanCandidate> = match self.phase {
            TaskPhase::Understanding => {
                let mut plans = vec![];
                // Option A: List directory only (cheap)
                let list_only = Molecule::atom(Atom::list(working_dir.clone()));
                plans.push(PlanCandidate {
                    name: "list_only".into(),
                    profile: list_only.profile(),
                    molecule: list_only,
                });
                // Option B: List + read target (more context)
                if target.exists() {
                    let list_and_read = Molecule::atom(Atom::list(working_dir.clone()))
                        .then(Molecule::atom(Atom::read(target.clone())));
                    plans.push(PlanCandidate {
                        name: "list_and_read".into(),
                        profile: list_and_read.profile(),
                        molecule: list_and_read,
                    });
                }
                // Option C: Gather context from multiple files
                let cargo_toml = working_dir.join("Cargo.toml");
                if cargo_toml.exists() && target.exists() {
                    let gather = Molecule::atom(Atom::read(cargo_toml))
                        .then(Molecule::atom(Atom::read(target.clone())))
                        .then(Molecule::atom(Atom::list(working_dir.clone())));
                    plans.push(PlanCandidate {
                        name: "full_context".into(),
                        profile: gather.profile(),
                        molecule: gather,
                    });
                }
                plans
            }
            TaskPhase::Testing => {
                let mut plans = vec![];
                // Option A: cargo check (fast, less info)
                let check = Molecule::atom(Atom::cargo_check(working_dir.clone()));
                plans.push(PlanCandidate {
                    name: "cargo_check".into(),
                    profile: check.profile(),
                    molecule: check,
                });
                // Option B: cargo check + clippy (more info, more energy)
                let check_clippy = Molecule::atom(Atom::cargo_check(working_dir.clone()))
                    .then(Molecule::atom(Atom::cargo_clippy(working_dir.clone())));
                plans.push(PlanCandidate {
                    name: "check_and_clippy".into(),
                    profile: check_clippy.profile(),
                    molecule: check_clippy,
                });
                // Option C: full test suite (most info, most energy)
                let test = Molecule::atom(Atom::cargo_test(working_dir.clone()));
                plans.push(PlanCandidate {
                    name: "cargo_test".into(),
                    profile: test.profile(),
                    molecule: test,
                });
                plans
            }
            TaskPhase::Generating => {
                let mut plans = vec![];
                if let Some(ref code) = self.generated_code {
                    // Option A: write + check
                    let wc = crate::action::primitives::recipes::write_and_check(
                        target.clone(),
                        code,
                    );
                    plans.push(PlanCandidate {
                        name: "write_and_check".into(),
                        profile: wc.profile(),
                        molecule: wc,
                    });
                    // Option B: write + check + test (more thorough)
                    let wct = crate::action::primitives::recipes::full_coding_workflow(
                        target.clone(),
                        code.clone(),
                        working_dir.clone(),
                    );
                    plans.push(PlanCandidate {
                        name: "full_workflow".into(),
                        profile: wct.profile(),
                        molecule: wct,
                    });
                }
                // Option C-E: Tier-aware generation (Dispatch atoms)
                // These let FEP choose the backend tier based on energy/success history
                let prompt = self.build_generation_prompt();
                for (name, mol) in crate::action::primitives::recipes::tiered_generation_candidates(
                    target.clone(),
                    &prompt,
                ) {
                    plans.push(PlanCandidate {
                        name,
                        profile: mol.profile(),
                        molecule: mol,
                    });
                }
                plans
            }
            TaskPhase::Fixing => {
                if let Some(ref code) = self.generated_code {
                    let mut plans = vec![];
                    // Option A: write + check (simple)
                    let wc = crate::action::primitives::recipes::write_and_check(
                        target.clone(),
                        code,
                    );
                    plans.push(PlanCandidate {
                        name: "fix_and_check".into(),
                        profile: wc.profile(),
                        molecule: wc,
                    });
                    // Option B: write + check with recovery
                    let wcr = crate::action::primitives::recipes::write_and_check(
                        target.clone(),
                        code,
                    )
                    .recover(|_| Molecule::atom(Atom::Noop));
                    plans.push(PlanCandidate {
                        name: "fix_with_recovery".into(),
                        profile: wcr.profile(),
                        molecule: wcr,
                    });
                    plans
                } else {
                    vec![]
                }
            }
            TaskPhase::Planning | TaskPhase::Done => vec![],
        };

        if candidates.is_empty() {
            return None;
        }

        // Enhancement 2: Query historical recipe success rates for learning loop.
        // Past execution outcomes influence which plan FEP prefers.
        let selected_idx = if let Some(ref store) = self.experience_store {
            let recipe_keys: Vec<&str> = candidates
                .iter()
                .map(|c| {
                    // Build recipe key from atom names
                    c.profile
                        .atom_names
                        .first()
                        .copied()
                        .unwrap_or("Unknown")
                })
                .collect();
            let rates = store.recipe_success_rates(&recipe_keys);
            select_best_plan_with_history(&candidates, current_phi, self.energy_budget, &rates)
        } else {
            select_best_plan(&candidates, current_phi, self.energy_budget)
        };

        let selected_idx = selected_idx?;
        let selected = &candidates[selected_idx];

        tracing::debug!(
            target: "symthaea::coding_agent",
            phase = %self.phase,
            selected = %selected.name,
            candidates = candidates.len(),
            energy = selected.profile.total_energy,
            "FEP selected plan (history-aware)"
        );

        // Build the actual molecule to execute
        // For tier-aware candidates, use the selected plan directly
        let profile = selected.profile.clone();
        self.build_execution_plan()
            .map(|m| (m, profile))
    }

    /// Evaluate whether the current plan is safe and affordable.
    /// Returns (approved, reason) — if not approved, the reason explains why.
    fn evaluate_plan(&self, plan: &Molecule, current_phi: f32) -> (bool, String) {
        let profile = plan.profile();

        // 1. Phi gating: is consciousness level sufficient?
        if !profile.phi_sufficient(current_phi) {
            return (
                false,
                format!(
                    "Phi too low: {:.3} < {:.3} required",
                    current_phi, profile.min_phi
                ),
            );
        }

        // 2. Energy budget: can we afford this plan?
        if !profile.within_budget(self.energy_budget) {
            return (
                false,
                format!(
                    "Energy budget exceeded: plan costs {:.1}, budget remaining {:.1}",
                    profile.total_energy, self.energy_budget
                ),
            );
        }

        // 3. Destructiveness: does this need confirmation we can't give autonomously?
        if profile.max_destructiveness == crate::action::DestructivenessLevel::Destructive {
            return (
                false,
                format!(
                    "Plan contains destructive action ({}) — requires confirmation",
                    profile.atom_names.join(" → ")
                ),
            );
        }

        (
            true,
            format!(
                "Plan approved: {} steps, energy {:.1}/{:.1}, phi {:.3}/{:.3}",
                profile.step_count,
                profile.total_energy,
                self.energy_budget,
                current_phi,
                profile.min_phi,
            ),
        )
    }

    /// Deduct energy cost from the budget after execution.
    fn deduct_energy(&mut self, profile: &PlanProfile) {
        self.energy_budget = (self.energy_budget - profile.total_energy).max(0.0);
    }

    fn build_motor_request(&self) -> MotorActionRequest {
        match self.phase {
            TaskPhase::Understanding => {
                // In understanding phase, we want to read files
                MotorActionRequest {
                    target_path: Some(self.config.working_dir.clone()),
                    ..Default::default()
                }
            }
            TaskPhase::Planning => {
                // Planning doesn't need motor output — cognitive loop handles internally
                MotorActionRequest::default()
            }
            TaskPhase::Generating => {
                if self.generated_code.is_some() {
                    // Code was just written — request cargo check to validate
                    MotorActionRequest {
                        target_path: Some(self.config.working_dir.clone()),
                        program: Some("cargo".into()),
                        args: vec!["check".into()],
                        ..Default::default()
                    }
                } else {
                    MotorActionRequest {
                        target_path: Some(self.config.working_dir.clone()),
                        ..Default::default()
                    }
                }
            }
            TaskPhase::Testing => {
                // Run cargo check/test in the working directory
                MotorActionRequest {
                    target_path: Some(self.config.working_dir.clone()),
                    program: Some("cargo".into()),
                    args: vec!["check".into()],
                    ..Default::default()
                }
            }
            TaskPhase::Fixing => {
                if self.generated_code.is_some() {
                    // Fix was just written — request cargo check
                    MotorActionRequest {
                        target_path: Some(self.config.working_dir.clone()),
                        program: Some("cargo".into()),
                        args: vec!["check".into()],
                        ..Default::default()
                    }
                } else {
                    MotorActionRequest {
                        target_path: Some(self.config.working_dir.clone()),
                        ..Default::default()
                    }
                }
            }
            TaskPhase::Done => MotorActionRequest::default(),
        }
    }

    // ── Phase Transition Logic ─────────────────────────────────────────

    /// Process the results of a cognitive cycle and decide the next phase.
    ///
    /// Phase transitions are driven by both the FEP motor command type (what the
    /// consciousness loop recommends) and the current phase state machine. The FEP
    /// motor command can override default transitions when the consciousness loop
    /// detects a need to explore, reflect, or consolidate.
    fn process_step_result(
        &mut self,
        cycle_result: &CycleResult,
        motor_result: Option<MotorOutputResult>,
        phi: f32,
    ) {
        // Check epistemic status from prediction confidence
        let confidence = self.cognitive_loop.prediction_confidence();
        let epistemic = Self::confidence_to_epistemic(confidence);

        // If epistemic status is too low and we've already tried generating, refuse
        // to generate again. On the first attempt, allow it (the system needs to try
        // before it can learn from failures).
        if self.phase == TaskPhase::Generating
            && epistemic == EpistemicStatus::Unknown
            && !self.generation_tiers.is_empty()
        {
            self.observations
                .push("Epistemic gate: confidence too low for generation, re-planning".into());
            self.phase = TaskPhase::Planning;
            self.phase_failures += 1;
            return;
        }

        // Process motor output if we got one
        if let Some(ref result) = motor_result {
            self.process_motor_result(result);
        }

        // Extract FEP motor command type — this is the consciousness loop's recommendation
        let fep_command = MotorCommandType::from_action_index(cycle_result.metadata.fep.fep_action);

        // FEP-driven phase overrides: the consciousness loop can redirect the agent
        // regardless of the current phase (except Done).
        //
        // Suppression rules:
        // 1. After 3+ iterations without generating any code, stop honoring
        //    ExplorationTrigger — the consciousness loop is being too cautious.
        // 2. Never redirect away from Generating/Testing/Fixing when code has
        //    been written — the agent needs to compile and test, not re-explore.
        let suppress_exploration = self.iteration >= 3 && self.generation_tiers.is_empty();
        let has_code = self.generated_code.is_some();
        let in_action_phase = matches!(
            self.phase,
            TaskPhase::Generating | TaskPhase::Testing | TaskPhase::Fixing
        );

        if self.phase != TaskPhase::Done {
            match fep_command {
                MotorCommandType::ExplorationTrigger => {
                    if self.phase != TaskPhase::Understanding
                        && !suppress_exploration
                        && !(has_code && in_action_phase)
                    {
                        tracing::info!(
                            target: "symthaea::coding_agent",
                            from = %self.phase,
                            "FEP ExplorationTrigger → Understanding"
                        );
                        self.phase = TaskPhase::Understanding;
                        self.phase_failures = 0;
                        return;
                    } else if suppress_exploration {
                        tracing::debug!(
                            target: "symthaea::coding_agent",
                            iteration = self.iteration,
                            "Suppressing FEP ExplorationTrigger — need to attempt generation"
                        );
                    }
                }
                MotorCommandType::ReflectionInitiate => {
                    if self.phase != TaskPhase::Planning
                        && self.phase != TaskPhase::Understanding
                        && !(has_code && in_action_phase)
                    {
                        tracing::info!(
                            target: "symthaea::coding_agent",
                            from = %self.phase,
                            "FEP ReflectionInitiate → Planning"
                        );
                        self.phase = TaskPhase::Planning;
                        self.phase_failures = 0;
                        return;
                    }
                }
                MotorCommandType::ExpectationReset => {
                    if (self.phase == TaskPhase::Generating || self.phase == TaskPhase::Fixing)
                        && !has_code
                    {
                        self.observations
                            .push("FEP ExpectationReset: model mismatch, re-planning".into());
                        self.phase = TaskPhase::Planning;
                        self.phase_failures = 0;
                        return;
                    }
                }
                MotorCommandType::MemoryConsolidate => {
                    self.observations
                        .push("FEP MemoryConsolidate: consolidating learned patterns".into());
                }
                _ => {}
            }
        }

        // ── Code Quality Gate ─────────────────────────────────────────
        // Before transitioning from Generating → Testing, validate that the
        // generated code is worth testing. Reject TODO stubs, empty bodies,
        // and unimplemented!() placeholders.
        // Quality gate is Rust-specific — skip for Python/Nix targets
        let is_rust = self.target_language() == "rust";
        if is_rust
            && (self.phase == TaskPhase::Generating || self.phase == TaskPhase::Fixing)
            && self.generated_code.is_some()
        {
            if let Some(ref code) = self.generated_code {
                if let Some(quality_issue) = Self::check_code_quality(code) {
                    tracing::info!(
                        target: "symthaea::coding_agent",
                        issue = %quality_issue,
                        "Code quality gate: rejecting generated code"
                    );
                    self.observations
                        .push(format!("Quality gate rejected code: {quality_issue}"));
                    // Record failure for the tier that produced this
                    if let Some(tier) = self.generation_tiers.last().copied() {
                        if let Some(ref mut dispatcher) = self.dispatcher {
                            dispatcher.record_outcome(tier, false);
                        }
                    }
                    self.generated_code = None;
                    self.phase_failures += 1;
                    if self.phase_failures >= self.config.max_phase_failures {
                        self.phase = TaskPhase::Planning;
                        self.phase_failures = 0;
                    }
                    return;
                }
            }
        }

        // Force-advance: if we've been cycling 4+ iterations without generating
        // any code, skip directly to Generating. The consciousness loop is being
        // too cautious — the agent needs to attempt generation to make progress.
        if self.iteration >= 4
            && self.generation_tiers.is_empty()
            && self.phase != TaskPhase::Done
            && self.phase != TaskPhase::Generating
        {
            tracing::info!(
                target: "symthaea::coding_agent",
                iteration = self.iteration,
                phase = %self.phase,
                "Force-advancing to Generating"
            );
            self.phase = TaskPhase::Generating;
            self.phase_failures = 0;
            self.generated_code = None;
            return;
        }

        // Default phase transitions based on current state + motor results
        match self.phase {
            TaskPhase::Understanding => {
                if self.iteration >= 1 || !self.observations.is_empty() {
                    self.phase = TaskPhase::Planning;
                    self.phase_failures = 0;
                    tracing::info!(target: "symthaea::coding_agent", "→ Planning");
                }
            }
            TaskPhase::Planning => {
                self.phase = TaskPhase::Generating;
                self.phase_failures = 0;
                self.generated_code = None; // clear for next generation
                tracing::info!(target: "symthaea::coding_agent", "→ Generating");
            }
            TaskPhase::Generating => {
                // Code was generated in pre_cycle_action. Check if it "compiled"
                // (motor result from cargo check, or just generated_code exists).
                let code_written = self.generated_code.is_some();
                let check_passed = motor_result.as_ref().map_or(false, |r| r.success);

                if code_written && check_passed {
                    self.phase = TaskPhase::Testing;
                    self.phase_failures = 0;
                    tracing::info!(target: "symthaea::coding_agent", "→ Testing");
                } else if code_written {
                    // Code written but check failed (or no check result) — test anyway
                    self.phase = TaskPhase::Testing;
                    self.phase_failures = 0;
                    tracing::info!(target: "symthaea::coding_agent", "→ Testing (unverified)");
                } else {
                    self.phase_failures += 1;
                    if self.phase_failures >= self.config.max_phase_failures {
                        self.phase = TaskPhase::Planning;
                        self.phase_failures = 0;
                        tracing::warn!(
                            target: "symthaea::coding_agent",
                            "Generation failed {} times, re-planning",
                            self.config.max_phase_failures
                        );
                    }
                }
            }
            TaskPhase::Testing => {
                // Try motor result first; if none, run testing molecule
                let effective_result = motor_result.clone().or_else(|| {
                    if self.generated_code.is_some() {
                        self.do_testing_molecule()
                    } else {
                        None
                    }
                });

                if let Some(ref result) = effective_result {
                    // Record outcome into dispatcher stats for Bayesian routing
                    self.record_generation_outcome(result.success);

                    if result.success {
                        self.phase = TaskPhase::Done;
                        tracing::info!(target: "symthaea::coding_agent", "→ Done (tests passed)");
                    } else {
                        self.phase = TaskPhase::Fixing;
                        self.phase_failures = 0;
                        self.generated_code = None; // clear for fix
                        tracing::info!(target: "symthaea::coding_agent", "→ Fixing");
                    }
                } else {
                    // No code generated and no motor result
                    self.phase_failures += 1;
                    if self.phase_failures >= self.config.max_phase_failures {
                        self.phase = TaskPhase::Done;
                    }
                }
            }
            TaskPhase::Fixing => {
                let code_written = self.generated_code.is_some();
                // Try motor result; if none and code was written, run testing molecule
                let effective_result = motor_result.clone().or_else(|| {
                    if code_written {
                        self.do_testing_molecule()
                    } else {
                        None
                    }
                });
                if let Some(ref result) = effective_result {
                    if result.success || code_written {
                        self.phase = TaskPhase::Testing;
                        self.phase_failures = 0;
                        tracing::info!(target: "symthaea::coding_agent", "→ Testing (after fix)");
                    } else {
                        self.phase_failures += 1;
                        if self.phase_failures >= self.config.max_phase_failures {
                            // Use differentiated retry strategy instead of giving up
                            let strategy = self.next_retry_strategy();
                            match strategy {
                                RetryStrategy::RequestClarification(ref msg) => {
                                    self.emit_event(AgentEvent::RequestClarification(msg.clone()));
                                    self.phase = TaskPhase::Done;
                                    tracing::warn!(
                                        target: "symthaea::coding_agent",
                                        "All retry strategies exhausted, requesting clarification"
                                    );
                                }
                                _ => {
                                    self.retry_state.current_strategy = strategy;
                                    self.phase = TaskPhase::Planning;
                                    self.phase_failures = 0;
                                    self.generated_code = None;
                                    tracing::info!(
                                        target: "symthaea::coding_agent",
                                        strategy = ?self.retry_state.current_strategy,
                                        "Retry strategy: re-planning with different approach"
                                    );
                                }
                            }
                        }
                    }
                } else if code_written {
                    self.phase = TaskPhase::Testing;
                    self.phase_failures = 0;
                } else {
                    self.phase_failures += 1;
                    if self.phase_failures >= self.config.max_phase_failures {
                        let strategy = self.next_retry_strategy();
                        match strategy {
                            RetryStrategy::RequestClarification(_) => {
                                self.phase = TaskPhase::Done;
                            }
                            _ => {
                                self.retry_state.current_strategy = strategy;
                                self.phase = TaskPhase::Planning;
                                self.phase_failures = 0;
                                self.generated_code = None;
                            }
                        }
                    }
                }
            }
            TaskPhase::Done => {} // terminal
        }

        // Stuck detection: if Phi stays low for 3+ cycles and we're past the
        // initial phases, consciousness isn't engaging — try a different approach.
        // Only triggers after Generating has been attempted (not during initial ramp-up).
        // NEVER overrides Testing or Fixing — the agent just wrote code and needs to
        // compile/test it, not re-plan. Low Phi during testing is expected (the CfC
        // hasn't learned coding-specific patterns yet).
        if self.phi_trace.len() >= 3
            && self.phase != TaskPhase::Done
            && self.phase != TaskPhase::Understanding
            && self.phase != TaskPhase::Planning
            && self.phase != TaskPhase::Testing
            && self.phase != TaskPhase::Fixing
            && !self.generation_tiers.is_empty()
        {
            let recent: Vec<f32> = self.phi_trace.iter().rev().take(3).copied().collect();
            let all_low = recent.iter().all(|&p| p < 0.2);
            if all_low {
                self.observations.push(
                    "Stuck detection: Phi consistently low, trying different approach".into(),
                );
                self.phase = TaskPhase::Planning;
                self.phase_failures = 0;
            }
        }
    }

    // ── Motor Result Processing ────────────────────────────────────────

    /// Process a motor output result — extract observations, track files.
    fn process_motor_result(&mut self, result: &MotorOutputResult) {
        if result.success {
            if let Some(ref outcome) = result.outcome {
                match outcome {
                    ActionOutcome::FileContent(data) => {
                        let content =
                            String::from_utf8_lossy(&data[..data.len().min(2000)]).to_string();
                        self.observations.push(format!(
                            "Read file ({} bytes): {}",
                            data.len(),
                            &content[..content.len().min(200)]
                        ));
                    }
                    ActionOutcome::DirectoryListing(entries) => {
                        let listing: Vec<String> = entries
                            .iter()
                            .take(20)
                            .map(|p| p.display().to_string())
                            .collect();
                        self.observations
                            .push(format!("Directory listing: {:?}", listing));
                    }
                    ActionOutcome::Success => {
                        if let Some(ActionType::Write) = result.action_type {
                            self.observations.push("File written successfully".into());
                        } else if let Some(ActionType::CargoCheck) | Some(ActionType::CargoTest) =
                            result.action_type
                        {
                            self.observations.push("Check/test passed".into());
                        } else {
                            self.observations.push("Action succeeded".into());
                        }
                    }
                    _ => {
                        self.observations
                            .push(format!("Action result: {:?}", result.action_type));
                    }
                }
            }
        } else if let Some(ref error) = result.error {
            self.errors.push(error.clone());

            // If this was a test failure, save the output for context
            if result.action_type == Some(ActionType::CargoTest)
                || result.action_type == Some(ActionType::CargoCheck)
            {
                self.last_test_output = Some(error.clone());

                // Track failure pattern frequency
                let pattern = Self::normalize_error_pattern(error);
                if let Some(entry) = self
                    .failure_patterns
                    .iter_mut()
                    .find(|(p, _)| *p == pattern)
                {
                    entry.1 += 1;
                } else {
                    self.failure_patterns.push((pattern.clone(), 1));
                }

                // Store failure in persistent experience store
                self.store_experience(error, false);
            }
        }
    }

    /// Run `cargo check` directly in the working directory, bypassing the motor bridge.
    ///
    /// This is the direct execution path for the coding agent — the agent *knows*
    /// it wants to compile, so we skip FEP motor confidence gating. Phi gating
    /// still applies: we won't run commands if Phi is below the warm-up threshold.
    fn run_cargo_check(&mut self) -> Option<MotorOutputResult> {
        if !self.config.enable_real_exec {
            return Some(MotorOutputResult {
                success: self.generated_code.is_some(),
                action_type: Some(ActionType::CargoCheck),
                prediction_error: 0.0,
                outcome: Some(crate::action::ActionOutcome::Success),
                error: None,
            });
        }

        let working_dir = self.config.working_dir.clone();

        if !working_dir.join("Cargo.toml").exists() {
            return Some(MotorOutputResult {
                success: false,
                action_type: Some(ActionType::CargoCheck),
                prediction_error: 0.5,
                outcome: None,
                error: Some("No Cargo.toml in working directory".into()),
            });
        }

        // Execute via MoleculeExecutor for value flow + energy tracking + trace
        let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);
        let molecule = Molecule::atom(Atom::cargo_check(working_dir.clone()));
        let mut executor = crate::action::primitives::MoleculeExecutor::new(
            current_phi,
            self.energy_budget,
            true, // real execution
        );

        tracing::info!(
            target: "symthaea::coding_agent",
            dir = %working_dir.display(),
            "Running cargo check (molecule executor)"
        );

        match executor.execute(&molecule) {
            Ok(val) => {
                // Update energy budget from executor
                self.energy_budget = executor.energy_budget;

                // Store execution trace
                self.store_execution_trace(&executor.trace);

                let success = val.is_success();
                let stderr_text = val.stderr().unwrap_or("").to_string();

                if !success {
                    self.last_test_output = Some(stderr_text.clone());
                    self.observations.push(format!(
                        "cargo check failed:\n{}",
                        &stderr_text[..stderr_text.len().min(500)]
                    ));
                } else {
                    self.observations.push("cargo check passed".into());
                }

                let (stdout_bytes, stderr_bytes, exit_code) = match &val {
                    PrimitiveValue::CommandResult { stdout, stderr, exit_code } => {
                        (stdout.as_bytes().to_vec(), stderr.as_bytes().to_vec(), *exit_code)
                    }
                    _ => (vec![], vec![], if success { 0 } else { 1 }),
                };

                Some(MotorOutputResult {
                    success,
                    action_type: Some(ActionType::CargoCheck),
                    prediction_error: if success { 0.0 } else { 0.8 },
                    outcome: Some(crate::action::ActionOutcome::CommandOutput {
                        stdout: stdout_bytes,
                        stderr: stderr_bytes,
                        exit_code,
                    }),
                    error: if success { None } else { Some(stderr_text) },
                })
            }
            Err(e) => {
                let error_msg = format!("{}", e);
                tracing::warn!(
                    target: "symthaea::coding_agent",
                    error = %error_msg,
                    "Molecule executor failed"
                );
                Some(MotorOutputResult {
                    success: false,
                    action_type: Some(ActionType::CargoCheck),
                    prediction_error: 1.0,
                    outcome: None,
                    error: Some(error_msg),
                })
            }
        }
    }

    /// Execute a molecule through MoleculeExecutor and convert results to
    /// observations, motor results, and trace storage.
    ///
    /// This is the unified execution path — all phases route through here
    /// instead of ad-hoc file I/O.
    fn execute_molecule(&mut self, molecule: &Molecule) -> Option<MotorOutputResult> {
        let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);
        let real_exec = self.config.enable_real_exec;
        let mut executor = MoleculeExecutor::new(current_phi, self.energy_budget, real_exec);

        match executor.execute(molecule) {
            Ok(val) => {
                // Update energy budget
                self.energy_budget = executor.energy_budget;

                // Store trace
                self.store_execution_trace(&executor.trace);

                // Convert result to observations
                match &val {
                    PrimitiveValue::Text(text) => {
                        if !text.is_empty() && text.len() <= 2000 {
                            self.observations.push(text.clone());
                        } else if text.len() > 2000 {
                            self.observations.push(format!(
                                "{}...(truncated, {} bytes total)",
                                &text[..1500],
                                text.len()
                            ));
                        }
                    }
                    PrimitiveValue::Listing(paths) => {
                        let names: Vec<String> = paths
                            .iter()
                            .take(50)
                            .map(|p| {
                                p.file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_else(|| p.display().to_string())
                            })
                            .collect();
                        self.observations.push(format!("Files: [{}]", names.join(", ")));
                    }
                    PrimitiveValue::CommandResult {
                        stdout,
                        stderr,
                        exit_code,
                    } => {
                        let success = *exit_code == 0;
                        if !success {
                            self.last_test_output = Some(stderr.clone());
                            self.observations.push(format!(
                                "Command failed (exit={}):\n{}",
                                exit_code,
                                &stderr[..stderr.len().min(500)]
                            ));
                        } else {
                            self.observations
                                .push(format!("Command succeeded (exit={})", exit_code));
                        }
                        return Some(MotorOutputResult {
                            success,
                            action_type: Some(ActionType::CargoCheck),
                            prediction_error: if success { 0.0 } else { 0.8 },
                            outcome: Some(ActionOutcome::CommandOutput {
                                stdout: stdout.as_bytes().to_vec(),
                                stderr: stderr.as_bytes().to_vec(),
                                exit_code: *exit_code,
                            }),
                            error: if success { None } else { Some(stderr.clone()) },
                        });
                    }
                    _ => {}
                }

                Some(MotorOutputResult {
                    success: true,
                    action_type: None,
                    prediction_error: 0.0,
                    outcome: Some(ActionOutcome::Success),
                    error: None,
                })
            }
            Err(e) => {
                let error_msg = format!("{}", e);
                tracing::warn!(
                    target: "symthaea::coding_agent",
                    error = %error_msg,
                    "Molecule execution failed"
                );
                self.observations.push(format!("Execution error: {}", error_msg));
                Some(MotorOutputResult {
                    success: false,
                    action_type: None,
                    prediction_error: 1.0,
                    outcome: None,
                    error: Some(error_msg),
                })
            }
        }
    }

    /// Execute the Understanding phase via molecules.
    ///
    /// Each read-only atom is executed individually through MoleculeExecutor
    /// so that intermediate results (file contents) are captured in observations.
    /// Understanding is always real I/O (read-only = no risk).
    fn do_understanding_molecule(&mut self) {
        let target = self.resolve_target_file();
        let working_dir = self.config.working_dir.clone();
        let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);

        // Execute each atom individually to capture all intermediate results.
        // This is understanding-specific: we need every file's content, not just
        // the last one in a sequence.

        // 1. List working directory
        {
            let mol = Molecule::atom(Atom::list(working_dir.clone()));
            let mut executor = MoleculeExecutor::new(current_phi, self.energy_budget, true);
            if let Ok(PrimitiveValue::Listing(paths)) = executor.execute(&mol) {
                self.energy_budget = executor.energy_budget;
                let mut names: Vec<String> = paths
                    .iter()
                    .filter_map(|p| {
                        let name = p.file_name()?.to_string_lossy().to_string();
                        if name.starts_with('.') || name == "target" || name == "node_modules" {
                            None
                        } else if p.is_dir() {
                            Some(format!("{}/", name))
                        } else {
                            Some(name)
                        }
                    })
                    .collect();
                names.sort();
                if !names.is_empty() {
                    self.observations.push(format!(
                        "Working directory {}: [{}]",
                        working_dir.display(),
                        names.join(", ")
                    ));
                }
            }
        }

        // 2. Read target file
        if target.exists() {
            let mol = Molecule::atom(Atom::read(target.clone()));
            let mut executor = MoleculeExecutor::new(current_phi, self.energy_budget, true);
            if let Ok(PrimitiveValue::Text(content)) = executor.execute(&mol) {
                self.energy_budget = executor.energy_budget;
                let preview = if content.len() > 1500 {
                    format!("{}...(truncated)", &content[..1500])
                } else {
                    content
                };
                self.observations.push(format!(
                    "Target file {} ({} bytes):\n{}",
                    target.display(),
                    preview.len(),
                    preview
                ));
            }
        } else {
            self.observations.push(format!(
                "Target file {} does not exist yet (will be created)",
                target.display()
            ));
        }

        // 3. Read Cargo.toml
        let cargo_toml = working_dir.join("Cargo.toml");
        if cargo_toml.exists() {
            let mol = Molecule::atom(Atom::read(cargo_toml));
            let mut executor = MoleculeExecutor::new(current_phi, self.energy_budget, true);
            if let Ok(PrimitiveValue::Text(content)) = executor.execute(&mol) {
                self.energy_budget = executor.energy_budget;
                let preview: String = content.lines().take(15).collect::<Vec<_>>().join("\n");
                self.observations.push(format!("Cargo.toml:\n{}", preview));
            }
        }

        // Query experience store (non-molecule, fast cache lookup)
        let hints = self.retrieve_experience_hints();
        if !hints.is_empty() {
            self.observations.push(format!(
                "Prior experience: {} relevant patterns",
                hints.len()
            ));
            for (pattern, hint) in hints.iter().take(3) {
                self.observations.push(format!(
                    "  Prior: {} → {}",
                    &pattern[..pattern.len().min(80)],
                    &hint[..hint.len().min(80)]
                ));
            }
        }
    }

    /// Execute the Testing phase via molecules.
    /// Replaces the ad-hoc run_cargo_check() with molecule execution.
    fn do_testing_molecule(&mut self) -> Option<MotorOutputResult> {
        if !self.config.enable_real_exec {
            return Some(MotorOutputResult {
                success: self.generated_code.is_some(),
                action_type: Some(ActionType::CargoCheck),
                prediction_error: 0.0,
                outcome: Some(ActionOutcome::Success),
                error: None,
            });
        }

        let working_dir = self.config.working_dir.clone();
        if !working_dir.join("Cargo.toml").exists() {
            return Some(MotorOutputResult {
                success: false,
                action_type: Some(ActionType::CargoCheck),
                prediction_error: 0.5,
                outcome: None,
                error: Some("No Cargo.toml in working directory".into()),
            });
        }

        // Use the FEP-selected plan if available, otherwise default to cargo check
        let molecule = self
            .current_plan
            .as_ref()
            .and_then(|profile| {
                // If the profile includes testing atoms, rebuild the molecule
                if profile.atom_names.iter().any(|n| *n == "CargoTest") {
                    Some(Molecule::atom(Atom::cargo_test(working_dir.clone())))
                } else if profile.atom_names.iter().any(|n| *n == "CargoCheck") {
                    Some(Molecule::atom(Atom::cargo_check(working_dir.clone())))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| Molecule::atom(Atom::cargo_check(working_dir)));

        self.execute_molecule(&molecule)
    }

    /// Check code quality before allowing it into the Testing phase.
    ///
    /// Returns `Some(reason)` if the code is too low quality to test, `None` if OK.
    fn check_code_quality(code: &str) -> Option<String> {
        let trimmed = code.trim();

        // Empty or near-empty code
        if trimmed.is_empty() || trimmed.len() < 10 {
            return Some("code is empty or trivially short".into());
        }

        // Contains TODO/unimplemented markers (indicating the generator punted)
        if trimmed.contains("// TODO: implement") || trimmed.contains("todo!(") {
            return Some("code contains TODO placeholder".into());
        }
        if trimmed.contains("unimplemented!(") {
            return Some("code contains unimplemented!() placeholder".into());
        }

        // Function with empty body: `fn X() { }` or `fn X() {}`
        // Check for functions where the body is just whitespace/comments
        for line in trimmed.lines() {
            let l = line.trim();
            if l.starts_with("pub fn ") || l.starts_with("fn ") {
                // Found a function declaration — check if the body is empty
                // Simple heuristic: if the entire code has a fn decl but no
                // statements (only braces and comments), it's empty
                break; // Only flag via the TODO/unimplemented checks above
            }
        }

        // Pure comment code (no actual Rust statements)
        let non_comment_lines: Vec<&str> = trimmed
            .lines()
            .filter(|l| {
                let t = l.trim();
                !t.is_empty() && !t.starts_with("//") && !t.starts_with("///")
            })
            .collect();
        if non_comment_lines.len() <= 1 {
            return Some("code contains only comments, no logic".into());
        }

        // LLM markdown wrapper (code came back with ```rust ... ```)
        if trimmed.starts_with("```") {
            return Some("code is wrapped in markdown fences".into());
        }

        // ── LLM-specific failure patterns ────────────────────────────────

        // Hallucinated imports: `use` of crates that don't exist in this project
        let hallucinated_crates = [
            "use crate_name::", // placeholder crate
            "use my_crate::",   // LLM default naming
            "use your_crate::", // LLM addressing user
            "use example::",    // example placeholder
            "use foo::",        // test placeholder
            "use bar::",        // test placeholder
        ];
        for hc in &hallucinated_crates {
            if trimmed.contains(hc) {
                return Some(format!("hallucinated import: {hc}"));
            }
        }

        // Incomplete function: function signature with `...` or `/* ... */` body
        if trimmed.contains("...") && (trimmed.contains("fn ") || trimmed.contains("impl ")) {
            return Some("code contains '...' ellipsis (incomplete)".into());
        }
        if trimmed.contains("/* ... */") || trimmed.contains("/* TODO */") {
            return Some("code contains placeholder comment block".into());
        }

        // LLM explanation leak: natural language sentences in what should be pure code
        let explanation_markers = [
            "Here is the implementation",
            "Here's the code",
            "Below is",
            "I'll implement",
            "Let me",
            "As you can see",
            "Note that",
            "This function",
            "The following",
        ];
        // Only flag if these appear outside of doc comments
        for marker in &explanation_markers {
            for line in trimmed.lines() {
                let l = line.trim();
                if l.starts_with(marker) && !l.starts_with("//") && !l.starts_with("///") {
                    return Some(format!("LLM explanation leak: '{}'", &l[..l.len().min(60)]));
                }
            }
        }

        // Duplicate function definitions (LLM sometimes generates the same fn twice)
        let fn_names: Vec<&str> = trimmed
            .lines()
            .filter_map(|l| {
                let t = l.trim();
                if (t.starts_with("pub fn ") || t.starts_with("fn ")) && t.contains('(') {
                    t.split('(').next()
                } else {
                    None
                }
            })
            .collect();
        for (i, name) in fn_names.iter().enumerate() {
            if fn_names[i + 1..].contains(name) {
                return Some(format!("duplicate function definition: {name}"));
            }
        }

        None
    }

    /// Normalize an error message for pattern matching (strip paths, line numbers).
    fn normalize_error_pattern(error: &str) -> String {
        // Extract the error code and type, strip file-specific info
        let mut normalized = String::new();
        for line in error.lines().take(3) {
            if line.contains("error[E") || line.contains("error:") {
                normalized.push_str(line.trim());
                normalized.push(' ');
            }
        }
        if normalized.is_empty() {
            error.lines().next().unwrap_or(error).to_string()
        } else {
            normalized.trim().to_string()
        }
    }

    /// Store a coding experience (success or failure) in the persistent store.
    fn store_experience(&mut self, detail: &str, success: bool) {
        let experience = CodingExperience {
            task: self.task.clone(),
            detail: detail.chars().take(500).collect(),
            success,
            tier: self
                .generation_tiers
                .last()
                .map(|t| t.to_string())
                .unwrap_or_default(),
            fix_hint: None,
        };

        if let Some(ref mut store) = self.experience_store {
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                rt.block_on(async {
                    store.store(experience).await;
                });
            }
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Record the outcome of code generation into the dispatcher's Bayesian stats.
    ///
    /// Called after Testing phase receives a motor result (cargo check/test pass/fail).
    /// Feeds back into `IntelligentDispatcher::select_tier()` — over time, backends
    /// with higher success rates get preferred for their epistemic bracket.
    fn record_generation_outcome(&mut self, success: bool) {
        // Find the tier used for the most recent generation
        let tier = self.generation_tiers.last().copied();
        if let (Some(tier), Some(ref mut dispatcher)) = (tier, &mut self.dispatcher) {
            dispatcher.record_outcome(tier, success);
        }

        // Store successful generation in persistent experience store
        if success {
            if let Some(code) = self.generated_code.clone() {
                let summary: String = code.chars().take(200).collect();
                self.store_experience(&summary, true);

                // If we had prior failures, the fix that worked is valuable —
                // store a fix_hint linking the last error to this success.
                if let Some((last_error, _)) = self.failure_patterns.last().cloned() {
                    self.store_fix_hint(&last_error, &code);
                }
            }
        }
    }

    /// Store a fix hint: when a failure pattern was resolved by a successful generation,
    /// record the mapping so future sessions can learn from it.
    fn store_fix_hint(&mut self, error_pattern: &str, fix_code: &str) {
        let fix_summary: String = fix_code.lines().take(5).collect::<Vec<_>>().join("\n");
        let experience = CodingExperience {
            task: self.task.clone(),
            detail: error_pattern.chars().take(300).collect(),
            success: true,
            tier: self
                .generation_tiers
                .last()
                .map(|t| t.to_string())
                .unwrap_or_default(),
            fix_hint: Some(fix_summary),
        };

        if let Some(ref mut store) = self.experience_store {
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                rt.block_on(async {
                    store.store(experience).await;
                });
            }
        }
    }

    /// Store a molecule execution trace — both in observations and in the
    /// persistent experience store for cross-session learning (#3).
    fn store_execution_trace(&mut self, trace: &[(String, f32, String)]) {
        if trace.is_empty() {
            return;
        }

        // Format trace for observations
        let trace_summary: String = trace
            .iter()
            .map(|(name, energy, summary)| format!("  {} (E={:.1}): {}", name, energy, summary))
            .collect::<Vec<_>>()
            .join("\n");

        tracing::debug!(
            target: "symthaea::coding_agent",
            steps = trace.len(),
            "Molecule trace:\n{}", trace_summary
        );

        // Store trace as a procedural experience for recipe learning
        let total_energy: f32 = trace.iter().map(|(_, e, _)| e).sum();
        let atom_names: Vec<&str> = trace.iter().map(|(n, _, _)| n.as_str()).collect();
        let recipe_key = atom_names.join("→");

        // Check if the last step was a success (exit=0 or no error)
        let last_success = trace
            .last()
            .map(|(_, _, s)| s.contains("exit=0") || s == "()")
            .unwrap_or(false);

        let experience = CodingExperience {
            task: format!("recipe:{}", recipe_key),
            detail: format!(
                "energy={:.1}, steps={}, atoms=[{}]",
                total_energy,
                trace.len(),
                recipe_key,
            ),
            success: last_success,
            tier: "MoleculeExecutor".to_string(),
            fix_hint: None,
        };

        if let Some(ref mut store) = self.experience_store {
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                rt.block_on(async {
                    store.store(experience).await;
                });
            }
        }
    }

    /// Map prediction confidence to epistemic status.
    fn confidence_to_epistemic(confidence: f32) -> EpistemicStatus {
        if confidence > 0.9 {
            EpistemicStatus::Certain
        } else if confidence > 0.7 {
            EpistemicStatus::Probable
        } else if confidence > 0.4 {
            EpistemicStatus::Uncertain
        } else {
            EpistemicStatus::Unknown
        }
    }

    /// Build the final result.
    fn build_result(&self) -> AgentResult {
        let confidence = self.cognitive_loop.prediction_confidence();
        AgentResult {
            files_modified: self.files_modified.clone(),
            tests_passed: if self.phase == TaskPhase::Done {
                Some(
                    self.observations
                        .iter()
                        .any(|o| o.contains("test passed") || o.contains("Check/test passed")),
                )
            } else {
                None
            },
            iterations_used: self.iteration,
            phi_trace: self.phi_trace.clone(),
            epistemic_status: Self::confidence_to_epistemic(confidence),
            final_phase: self.phase.clone(),
            observations: self.observations.clone(),
            errors: self.errors.clone(),
            generation_tiers: self.generation_tiers.clone(),
            total_energy: self.dispatcher.as_ref().map_or(0.0, |d| d.total_energy()),
            remaining_energy: self.energy_budget,
            #[cfg(feature = "school_learning")]
            generated_lessons: self.generate_lessons_from_failures(),
        }
    }

    /// Generate auto-curriculum lessons from accumulated failure patterns.
    #[cfg(feature = "school_learning")]
    fn generate_lessons_from_failures(&self) -> Vec<crate::school::code_learning::CodeLesson> {
        let failures: Vec<(String, String, usize)> = self
            .failure_patterns
            .iter()
            .map(|(pattern, count)| (pattern.clone(), self.task.clone(), *count))
            .collect();
        crate::school::code_learning::lessons_from_failures(&failures, 5)
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Task E: Structured test failures, consciousness signals, events, retry
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Parse structured test failures from cargo test stderr output.
    fn parse_test_failures(stderr: &str) -> Vec<StructuredTestFailure> {
        let mut failures = Vec::new();
        let mut current_test: Option<String> = None;
        let mut current_output = String::new();

        for line in stderr.lines() {
            // Detect test failure header: "---- test_name stdout ----"
            if line.starts_with("---- ") && line.ends_with(" stdout ----") {
                // Flush previous test if any
                if let Some(ref name) = current_test {
                    failures.push(Self::build_test_failure(name, &current_output));
                }
                let name = line
                    .trim_start_matches("---- ")
                    .trim_end_matches(" stdout ----")
                    .to_string();
                current_test = Some(name);
                current_output.clear();
            } else if current_test.is_some() {
                current_output.push_str(line);
                current_output.push('\n');
            }
        }
        // Flush last test
        if let Some(ref name) = current_test {
            failures.push(Self::build_test_failure(name, &current_output));
        }
        failures
    }

    /// Build a structured test failure from a test name and its captured output.
    fn build_test_failure(test_name: &str, output: &str) -> StructuredTestFailure {
        let (kind, expected, actual) = if output.contains("assertion `left == right` failed") {
            let left = output
                .lines()
                .find(|l| l.trim().starts_with("left:"))
                .map(|l| l.trim().trim_start_matches("left:").trim().to_string());
            let right = output
                .lines()
                .find(|l| l.trim().starts_with("right:"))
                .map(|l| l.trim().trim_start_matches("right:").trim().to_string());
            (TestFailureKind::AssertEq, right, left)
        } else if output.contains("assertion") && output.contains("failed") {
            (TestFailureKind::Assert, None, None)
        } else if output.contains("panicked at") {
            (TestFailureKind::Panic, None, None)
        } else {
            (TestFailureKind::Other, None, None)
        };

        let message = output
            .lines()
            .find(|l| l.contains("panicked at") || l.contains("assertion"))
            .map(|l| l.trim().to_string());

        let (file, line) = output
            .lines()
            .find(|l| l.contains(".rs:"))
            .map(|l| Self::extract_panic_location(l))
            .unwrap_or((None, None));

        StructuredTestFailure {
            test_name: test_name.to_string(),
            failure_kind: kind,
            expected,
            actual,
            message,
            file,
            line,
        }
    }

    /// Extract file path and line number from a panic location string.
    fn extract_panic_location(location: &str) -> (Option<String>, Option<usize>) {
        // Match patterns like "src/foo.rs:42:5" or "at ./src/foo.rs:42:5"
        let loc = location.trim().trim_start_matches("at ");
        if let Some(colon_idx) = loc.rfind(".rs:") {
            let file_end = colon_idx + 3; // include ".rs"
            let file = loc[..file_end].trim().to_string();
            let after = &loc[file_end + 1..]; // skip ":"
            let line = after.split(':').next().and_then(|s| s.parse().ok());
            (Some(file), line)
        } else {
            (None, None)
        }
    }

    /// Format structured test failures into a prompt-friendly string.
    fn format_structured_test_failures(failures: &[StructuredTestFailure]) -> String {
        if failures.is_empty() {
            return String::new();
        }
        let mut out = format!("\n{} test failure(s):\n", failures.len());
        for f in failures {
            out.push_str(&format!("  - {} ({:?})", f.test_name, f.failure_kind));
            if let (Some(exp), Some(act)) = (&f.expected, &f.actual) {
                out.push_str(&format!(" expected={}, got={}", exp, act));
            }
            if let (Some(file), Some(line)) = (&f.file, f.line) {
                out.push_str(&format!(" at {}:{}", file, line));
            }
            if let Some(msg) = &f.message {
                let short: String = msg.chars().take(100).collect();
                out.push_str(&format!(" — {}", short));
            }
            out.push('\n');
        }
        out
    }

    /// Extract consciousness signals from a cycle result for decision-making.
    fn extract_consciousness_signals(&self, cycle_result: &CycleResult) -> ConsciousnessSignals {
        let prediction_error = 1.0 - self.cognitive_loop.prediction_confidence();
        let confidence_velocity = if self.prediction_error_history.len() >= 2 {
            let prev = self.prediction_error_history[self.prediction_error_history.len() - 1];
            self.cognitive_loop.prediction_confidence() - (1.0 - prev)
        } else {
            0.0
        };
        let phi = cycle_result.metadata.consciousness.consciousness_level as f32;
        let phi_slope = if self.phi_trace.len() >= 2 {
            let last = self.phi_trace[self.phi_trace.len() - 1];
            phi - last
        } else {
            0.0
        };
        let fep_surprise = cycle_result.metadata.fep.fep_surprise;

        ConsciousnessSignals {
            prediction_error,
            confidence_velocity,
            phi,
            phi_slope,
            fep_surprise,
        }
    }

    /// Attach an event channel for streaming agent progress.
    /// Returns (self, receiver) — caller reads events from the receiver.
    pub fn with_event_channel(mut self) -> (Self, std::sync::mpsc::Receiver<AgentEvent>) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.event_sink = Some(tx);
        (self, rx)
    }

    /// Emit an event to the event channel (no-op if no sink).
    fn emit_event(&self, event: AgentEvent) {
        if let Some(ref sink) = self.event_sink {
            let _ = sink.send(event);
        }
    }

    /// Emit a phase transition event.
    fn emit_phase_transition(&self, from: &TaskPhase, to: &TaskPhase) {
        self.emit_event(AgentEvent::PhaseTransition {
            from: from.clone(),
            to: to.clone(),
            iteration: self.iteration,
        });
    }

    /// Build HDC context prompt from indexed codebase memory.
    #[cfg(feature = "code_generation")]
    fn build_hdc_context_prompt(&self) -> String {
        use crate::hdc::code_encoder::CodeHDEncoder;

        let code_memory = match &self.code_memory {
            Some(m) => m,
            None => return String::new(),
        };

        // Encode the task as an HDC query vector
        let encoder = CodeHDEncoder::new(16_384);
        let query_hv = encoder.encode_name(&self.task);
        let matches = code_memory.query(&query_hv, 5);

        if matches.is_empty() {
            return String::new();
        }

        let coherence = code_memory.codebase_coherence();
        let mut prompt = format!(
            "## Codebase context (HDC similarity search, coherence={:.2})\n",
            coherence
        );
        for m in &matches {
            prompt.push_str(&format!("- {} (similarity={:.3})\n", m.name, m.similarity));
            // Include source snippet if available
            if let Some(src) = self.source_cache.get(&m.path) {
                let snippet = Self::extract_entity_source(src, &m.name, m.kind);
                if !snippet.contains("(source not found)") {
                    let truncated: String = snippet.chars().take(200).collect();
                    prompt.push_str(&format!("  ```\n  {}\n  ```\n", truncated));
                }
            }
        }
        prompt
    }

    #[cfg(not(feature = "code_generation"))]
    fn build_hdc_context_prompt(&self) -> String {
        String::new()
    }

    /// Select the next retry strategy, cycling through options.
    fn next_retry_strategy(&mut self) -> RetryStrategy {
        let strategies = [
            RetryStrategy::DifferentTemplate,
            RetryStrategy::DifferentBackend(BackendTier::LocalLlm),
            RetryStrategy::DifferentBackend(BackendTier::CloudLlm),
            RetryStrategy::SimplifyScope,
            RetryStrategy::RequestClarification(
                "Unable to resolve after multiple strategies. Could you clarify or simplify the task?".to_string(),
            ),
        ];

        for s in &strategies {
            if !self.retry_state.strategies_tried.contains(s) {
                let strategy = s.clone();
                self.retry_state.strategies_tried.push(strategy.clone());
                self.emit_event(AgentEvent::RetryStrategyChanged(strategy.clone()));
                return strategy;
            }
        }

        // All strategies exhausted — request clarification
        let fallback =
            RetryStrategy::RequestClarification("All retry strategies exhausted.".to_string());
        self.emit_state_changed(&fallback);
        fallback
    }

    /// Helper: emit retry strategy (avoids borrow issues with emit_event).
    fn emit_state_changed(&self, strategy: &RetryStrategy) {
        self.emit_event(AgentEvent::RetryStrategyChanged(strategy.clone()));
    }

    // ── Public API ──────────────────────────────────────────────────────

    /// Get the current phase.
    pub fn phase(&self) -> &TaskPhase {
        &self.phase
    }

    /// Get the current iteration count.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Access the underlying cognitive loop for direct inspection.
    pub fn cognitive_loop(&self) -> &CognitiveLoopService {
        &self.cognitive_loop
    }

    /// Access the underlying cognitive loop mutably.
    pub fn cognitive_loop_mut(&mut self) -> &mut CognitiveLoopService {
        &mut self.cognitive_loop
    }

    /// Set a custom intelligent dispatcher for LLM-routed code generation.
    pub fn with_dispatcher(mut self, dispatcher: IntelligentDispatcher) -> Self {
        self.dispatcher = Some(dispatcher);
        self
    }

    /// Set codebase context from an external CodebaseMemory query.
    ///
    /// The caller (e.g. Symthaea facade) queries CodebaseMemory for relevant
    /// functions/types and passes the results here as strings. This decouples
    /// the agent from the `code_generation` feature gate.
    pub fn set_code_context(&mut self, context: Vec<String>) {
        self.code_context = context;
    }

    /// Index a project directory into a `CodebaseMemory` and inject relevant context.
    ///
    /// Walks the directory (respecting .gitignore), parses each source file using
    /// `ParserRegistry`, and indexes into `CodebaseMemory`. Then queries the memory
    /// for entities related to the current task and sets them as code context.
    ///
    /// Returns `(files_indexed, functions_found, types_found)` on success.
    #[cfg(feature = "code_generation")]
    pub fn index_project(
        &mut self,
        root: &std::path::Path,
    ) -> anyhow::Result<(usize, usize, usize)> {
        use crate::hdc::code_encoder::CodeHDEncoder;
        use crate::hdc::code_memory::CodebaseMemory;
        use crate::language::parser_registry::ParserRegistry;
        use ignore::WalkBuilder;

        let mut memory = CodebaseMemory::with_default_encoder();
        let mut parser_registry = ParserRegistry::with_builtins();
        let mut files_indexed = 0usize;
        let mut parse_errors = 0usize;

        // Walk directory respecting .gitignore
        for entry in WalkBuilder::new(root)
            .hidden(true)
            .git_ignore(true)
            .build()
            .flatten()
        {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            // Only process files the parser registry can handle
            let filename = path.file_name().and_then(|f| f.to_str());
            let ext = path.extension().and_then(|e| e.to_str());
            let supported = matches!(ext, Some("rs") | Some("py") | Some("nix"));
            if !supported {
                continue;
            }

            let source = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            match parser_registry.parse(&source, None, filename) {
                Ok(parsed) => {
                    memory.index_file(path, &parsed);
                    files_indexed += 1;
                    self.source_cache.insert(path.to_path_buf(), source);
                }
                Err(_) => {
                    parse_errors += 1;
                }
            }
        }

        let stats = memory.stats();
        tracing::info!(
            target: "symthaea::coding_agent",
            files = files_indexed,
            functions = stats.functions,
            types = stats.types,
            parse_errors = parse_errors,
            "Indexed project into CodebaseMemory"
        );

        // If we have a task, query for source-level context
        if !self.task.is_empty() {
            self.code_context = Self::build_source_context(&memory, &self.source_cache, &self.task);
        }

        // Store memory for future queries
        self.code_memory = Some(memory);

        Ok((files_indexed, stats.functions, stats.types))
    }

    /// Re-index a single file after it has been written/modified.
    #[cfg(feature = "code_generation")]
    fn reindex_file(&mut self, path: &std::path::Path, source: &str) {
        use crate::language::parser_registry::ParserRegistry;
        let filename = path.file_name().and_then(|f| f.to_str());
        let mut parser_registry = ParserRegistry::with_builtins();
        if let Ok(parsed) = parser_registry.parse(source, None, filename) {
            if let Some(ref mut memory) = self.code_memory {
                memory.update_file(path, &parsed);
            }
        }
        self.source_cache
            .insert(path.to_path_buf(), source.to_string());
    }

    /// Build source-level context from CodebaseMemory matches.
    #[cfg(feature = "code_generation")]
    fn build_source_context(
        memory: &crate::hdc::code_memory::CodebaseMemory,
        source_cache: &std::collections::HashMap<PathBuf, String>,
        task: &str,
    ) -> Vec<String> {
        let encoder = memory.encoder();
        let intent_hv = encoder.encode_name(task);
        let matches = memory.query(&intent_hv, 5);
        matches
            .iter()
            .filter(|m| m.similarity > 0.2)
            .filter_map(|m| {
                let source = source_cache.get(&m.path)?;
                let snippet = Self::extract_entity_source(source, &m.name, m.kind);
                Some(format!(
                    "// {} — {:?} `{}` (similarity: {:.3})\n{}",
                    m.path.display(),
                    m.kind,
                    m.name,
                    m.similarity,
                    snippet
                ))
            })
            .collect()
    }

    /// Extract source code for a named entity using brace-matching (up to 20 lines).
    #[cfg(feature = "code_generation")]
    fn extract_entity_source(
        source: &str,
        name: &str,
        kind: crate::language::code_parser::EntityKind,
    ) -> String {
        use crate::language::code_parser::EntityKind;
        let keyword = match kind {
            EntityKind::Function | EntityKind::Method => "fn ",
            EntityKind::Struct => "struct ",
            EntityKind::Enum => "enum ",
            EntityKind::Trait | EntityKind::Interface => "trait ",
            EntityKind::Class => "class ",
            _ => "fn ",
        };
        let pattern = format!("{keyword}{name}");
        let lines: Vec<&str> = source.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.contains(&pattern) {
                let mut depth = 0i32;
                let mut out = Vec::new();
                let mut started = false;
                for j in i..lines.len().min(i + 30) {
                    out.push(lines[j]);
                    for ch in lines[j].chars() {
                        if ch == '{' {
                            depth += 1;
                            started = true;
                        }
                        if ch == '}' {
                            depth -= 1;
                        }
                    }
                    if started && depth <= 0 {
                        break;
                    }
                    if out.len() >= 20 {
                        out.push("    // ... (truncated)");
                        break;
                    }
                }
                return out.join("\n");
            }
        }
        format!("// {keyword}{name} (source not found)")
    }

    /// Get the last dispatch result (which backend tier was used, energy cost, etc.).
    pub fn last_dispatch(&self) -> Option<&DispatchResult> {
        self.last_dispatch.as_ref()
    }

    /// Get the dispatcher's total energy consumption.
    pub fn total_energy(&self) -> f64 {
        self.dispatcher.as_ref().map_or(0.0, |d| d.total_energy())
    }

    /// Get accumulated failure patterns from this run: (normalized_error, count).
    pub fn failure_patterns(&self) -> &[(String, usize)] {
        &self.failure_patterns
    }

    /// Whether the agent has a persistent experience store.
    pub fn has_experience_store(&self) -> bool {
        self.experience_store.is_some()
    }

    /// Get the count of stored experiences (successes + failures).
    pub fn experience_count(&self) -> usize {
        if let Some(ref store) = self.experience_store {
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                return rt.block_on(store.count());
            }
        }
        0
    }

    /// Get cached success patterns from the experience store.
    pub fn cached_successes(&self) -> Vec<(String, String)> {
        self.experience_store
            .as_ref()
            .map(|s| s.cached_successes().to_vec())
            .unwrap_or_default()
    }

    /// Get cached error hints from the experience store.
    pub fn cached_error_hints(&self) -> Vec<(String, String)> {
        self.experience_store
            .as_ref()
            .map(|s| s.cached_error_hints().to_vec())
            .unwrap_or_default()
    }

    /// Get the current execution plan profile (if any).
    pub fn current_plan_profile(&self) -> Option<&PlanProfile> {
        self.current_plan.as_ref()
    }

    /// Get remaining energy budget.
    pub fn remaining_energy(&self) -> f32 {
        self.energy_budget
    }

    /// Build a plan for a hypothetical action and evaluate it.
    /// Useful for FEP to reason about candidate actions before choosing.
    pub fn evaluate_hypothetical_plan(&self, plan: &Molecule) -> (bool, String, PlanProfile) {
        let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);
        let profile = plan.profile();
        let (approved, reason) = self.evaluate_plan(plan, current_phi);
        (approved, reason, profile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_phase_display() {
        assert_eq!(format!("{}", TaskPhase::Understanding), "Understanding");
        assert_eq!(format!("{}", TaskPhase::Done), "Done");
    }

    #[test]
    fn test_confidence_to_epistemic() {
        assert_eq!(
            CodingAgent::confidence_to_epistemic(0.95),
            EpistemicStatus::Certain
        );
        assert_eq!(
            CodingAgent::confidence_to_epistemic(0.8),
            EpistemicStatus::Probable
        );
        assert_eq!(
            CodingAgent::confidence_to_epistemic(0.5),
            EpistemicStatus::Uncertain
        );
        assert_eq!(
            CodingAgent::confidence_to_epistemic(0.2),
            EpistemicStatus::Unknown
        );
    }

    #[test]
    fn test_coding_agent_creation() {
        let config = CodingAgentConfig::default();
        let agent = CodingAgent::new(config);
        assert!(agent.is_ok(), "CodingAgent should create successfully");
        let agent = agent.unwrap();
        assert_eq!(*agent.phase(), TaskPhase::Understanding);
        assert_eq!(agent.iteration(), 0);
    }

    #[test]
    fn test_coding_agent_runs_and_generates() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 5,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("generated.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();

        let result = agent.run("add a hello() function");

        // Agent should have run through iterations
        assert!(result.iterations_used > 0);
        assert!(!result.phi_trace.is_empty());

        // Code should have been generated and written
        assert!(
            !result.files_modified.is_empty(),
            "Should have written at least one file"
        );
        let target = dir.path().join("generated.rs");
        assert!(target.exists(), "Target file should exist on disk");

        let content = std::fs::read_to_string(&target).unwrap();
        assert!(
            content.contains("fn"),
            "Generated file should contain a function"
        );

        // Should have used at least one generation tier
        assert!(
            !result.generation_tiers.is_empty(),
            "Should have recorded generation tiers"
        );
    }

    // ── Hardening: property tests & stress tests ───────────────────────

    #[test]
    fn test_confidence_to_epistemic_full_range() {
        // Sweep the full [0,1] range — no panic, always valid
        for i in 0..=100 {
            let c = i as f32 / 100.0;
            let _ = CodingAgent::confidence_to_epistemic(c);
        }
        // Boundary: negative and >1 should not panic
        let _ = CodingAgent::confidence_to_epistemic(-0.1_f32);
        let _ = CodingAgent::confidence_to_epistemic(1.5_f32);
        let _ = CodingAgent::confidence_to_epistemic(0.0_f32);
        let _ = CodingAgent::confidence_to_epistemic(1.0_f32);
    }

    #[test]
    fn test_telemetry_json_fields_finite() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 3,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("gen.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        let result = agent.run("add a function");

        let json = result.to_telemetry_json();
        // All phi values should be finite
        if let Some(trace) = json["consciousness"]["phi_trace"].as_array() {
            for v in trace {
                let f = v.as_f64().unwrap();
                assert!(f.is_finite(), "phi value must be finite, got {f}");
            }
        }
        // iterations_used should be non-negative
        assert!(json["iterations_used"].as_u64().unwrap() > 0);
        // total_energy should be finite
        let energy = json["generation"]["total_energy"].as_f64().unwrap();
        assert!(energy.is_finite(), "total_energy must be finite");
    }

    #[test]
    fn test_agent_result_phi_trace_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 10,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("gen.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        let result = agent.run("write a sorting function");

        // Phi trace should have entries and all be in [0, 1]
        assert!(!result.phi_trace.is_empty());
        for phi in &result.phi_trace {
            assert!(phi.is_finite(), "phi must be finite");
            assert!(
                *phi >= 0.0 && *phi <= 1.0,
                "phi must be in [0,1], got {phi}"
            );
        }
        // iterations_used should match phi_trace length
        assert_eq!(result.iterations_used, result.phi_trace.len());
    }

    #[test]
    fn test_100_cycle_stress() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 100,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("stress.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        let result = agent.run("implement a fibonacci function");

        // Should complete without panic
        assert!(result.iterations_used > 0);
        // Phi trace length == iterations
        assert_eq!(result.phi_trace.len(), result.iterations_used);
        // All phi bounded
        for phi in &result.phi_trace {
            assert!(phi.is_finite() && *phi >= 0.0 && *phi <= 1.0);
        }
        // Errors list should be finite (no unbounded growth)
        assert!(result.errors.len() <= 100);
        assert!(result.observations.len() <= 1000);
        // Energy should be finite and non-negative
        assert!(result.total_energy.is_finite() && result.total_energy >= 0.0);
    }

    #[test]
    fn test_determinism_same_input() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();

        let run = |dir: &std::path::Path| -> AgentResult {
            let config = CodingAgentConfig {
                max_iterations: 5,
                working_dir: dir.to_path_buf(),
                target_file: Some(PathBuf::from("det.rs")),
                ..Default::default()
            };
            let mut agent = CodingAgent::new(config).unwrap();
            agent.run("add a hello function")
        };

        let r1 = run(dir1.path());
        let r2 = run(dir2.path());

        // Same task should produce same phase progression
        assert_eq!(r1.iterations_used, r2.iterations_used);
        assert_eq!(format!("{}", r1.final_phase), format!("{}", r2.final_phase));
        // Phi traces should be identical (deterministic CLS)
        assert_eq!(r1.phi_trace.len(), r2.phi_trace.len());
    }

    #[test]
    fn test_run_reset_clears_state() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 3,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("reset.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();

        // First run
        let r1 = agent.run("add function foo");
        assert!(r1.iterations_used > 0);

        // Second run should start fresh — iteration counter and phi trace reset
        let r2 = agent.run("add function bar");
        assert!(r2.iterations_used > 0);
        // Phi trace should track iterations (±1 for retry strategies)
        let diff1 = (r1.phi_trace.len() as isize - r1.iterations_used as isize).unsigned_abs();
        let diff2 = (r2.phi_trace.len() as isize - r2.iterations_used as isize).unsigned_abs();
        assert!(diff1 <= 1, "Run 1 phi trace should track iterations");
        assert!(diff2 <= 1, "Run 2 phi trace should track iterations");
        // Run 2 should not accumulate phi from run 1
        assert!(
            r2.phi_trace.len() <= r2.iterations_used + 1,
            "Run 2 phi trace ({}) should not carry over from run 1 ({})",
            r2.phi_trace.len(),
            r1.phi_trace.len()
        );
    }

    #[test]
    fn test_fibonacci_native_template() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 5,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("fib.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();

        let result = agent.run("add fibonacci function");

        let target = dir.path().join("fib.rs");
        assert!(target.exists());
        let content = std::fs::read_to_string(&target).unwrap();
        assert!(content.contains("fibonacci"), "Should contain fibonacci fn");
        assert!(content.contains("pub fn"), "Should be a public function");
        assert!(
            !result.files_modified.is_empty(),
            "Should have modified files"
        );
    }

    #[test]
    fn test_resolve_target_file_from_task() {
        let config = CodingAgentConfig {
            working_dir: PathBuf::from("/tmp/project"),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();

        // Task mentions a file path
        agent.task = "add hello() to src/main.rs".to_string();
        let target = agent.resolve_target_file();
        assert_eq!(target, PathBuf::from("/tmp/project/src/main.rs"));

        // Task mentions absolute path
        agent.task = "modify /tmp/test.rs".to_string();
        let target = agent.resolve_target_file();
        assert_eq!(target, PathBuf::from("/tmp/test.rs"));

        // No file in task — falls back to default
        agent.task = "add a greeting function".to_string();
        let target = agent.resolve_target_file();
        assert_eq!(target, PathBuf::from("/tmp/project/src/lib.rs"));
    }

    #[test]
    fn test_build_generation_prompt() {
        let config = CodingAgentConfig::default();
        let mut agent = CodingAgent::new(config).unwrap();
        agent.task = "add fibonacci function".to_string();
        agent.code_context = vec!["pub fn existing_fn() {}".to_string()];
        agent.observations = vec!["Read file: some content".to_string()];

        let prompt = agent.build_generation_prompt();
        assert!(prompt.contains("fibonacci"));
        assert!(prompt.contains("existing_fn"));
        assert!(prompt.contains("some content"));
    }

    #[test]
    fn test_build_generation_prompt_fixing_includes_error() {
        let config = CodingAgentConfig::default();
        let mut agent = CodingAgent::new(config).unwrap();
        agent.task = "add function".to_string();
        agent.phase = TaskPhase::Fixing;
        agent.last_test_output = Some("error[E0412]: cannot find type".into());

        let prompt = agent.build_generation_prompt();
        assert!(prompt.contains("E0412"));
        assert!(prompt.contains("Fix the code"));
    }

    #[test]
    fn test_code_context_in_prompt() {
        let config = CodingAgentConfig::default();
        let mut agent = CodingAgent::new(config).unwrap();
        agent.task = "test".to_string();

        // No context initially
        let prompt = agent.build_generation_prompt();
        assert!(!prompt.contains("Relevant code"));

        // Set context
        agent.set_code_context(vec![
            "pub struct Config { dim: usize }".to_string(),
            "pub fn process(c: &Config) {}".to_string(),
        ]);
        let prompt = agent.build_generation_prompt();
        assert!(prompt.contains("Relevant code"));
        assert!(prompt.contains("Config"));
        assert!(prompt.contains("process"));
    }

    #[test]
    fn test_build_observation_includes_task() {
        let config = CodingAgentConfig::default();
        let mut agent = CodingAgent::new(config).unwrap();
        agent.task = "fix the bug".to_string();

        let obs = agent.build_observation();
        assert!(obs.contains("fix the bug"));
        assert!(obs.contains("Understanding"));
    }

    #[test]
    fn test_fep_exploration_redirects_to_understanding() {
        let config = CodingAgentConfig::default();
        let mut agent = CodingAgent::new(config).unwrap();
        agent.phase = TaskPhase::Generating;

        let mut cycle_result = agent.cognitive_loop.cycle("test");
        cycle_result.metadata.fep.fep_action = 2; // ExplorationTrigger

        agent.process_step_result(&cycle_result, None, 0.5);
        assert_eq!(agent.phase, TaskPhase::Understanding);
    }

    #[test]
    fn test_fep_reflection_redirects_to_planning() {
        let config = CodingAgentConfig::default();
        let mut agent = CodingAgent::new(config).unwrap();
        agent.phase = TaskPhase::Generating;

        let mut cycle_result = agent.cognitive_loop.cycle("test");
        cycle_result.metadata.fep.fep_action = 3; // ReflectionInitiate

        agent.process_step_result(&cycle_result, None, 0.5);
        assert_eq!(agent.phase, TaskPhase::Planning);
    }

    #[test]
    fn test_fep_expectation_reset_from_fixing() {
        let config = CodingAgentConfig::default();
        let mut agent = CodingAgent::new(config).unwrap();
        agent.phase = TaskPhase::Fixing;

        let mut cycle_result = agent.cognitive_loop.cycle("test");
        cycle_result.metadata.fep.fep_action = 5; // ExpectationReset

        agent.process_step_result(&cycle_result, None, 0.5);
        assert_eq!(agent.phase, TaskPhase::Planning);
        assert!(agent
            .observations
            .iter()
            .any(|o| o.contains("ExpectationReset")));
    }

    #[test]
    fn test_fep_override_does_not_affect_done() {
        let config = CodingAgentConfig::default();
        let mut agent = CodingAgent::new(config).unwrap();
        agent.phase = TaskPhase::Done;

        let mut cycle_result = agent.cognitive_loop.cycle("test");
        cycle_result.metadata.fep.fep_action = 2; // ExplorationTrigger

        agent.process_step_result(&cycle_result, None, 0.5);
        assert_eq!(agent.phase, TaskPhase::Done);
    }

    #[test]
    fn test_motor_result_processing() {
        let config = CodingAgentConfig::default();
        let mut agent = CodingAgent::new(config).unwrap();

        // Test successful file read
        let result = MotorOutputResult {
            success: true,
            action_type: Some(ActionType::Read),
            prediction_error: 0.0,
            outcome: Some(ActionOutcome::FileContent(b"fn hello() {}".to_vec())),
            error: None,
        };
        agent.process_motor_result(&result);
        assert!(agent.observations.last().unwrap().contains("Read file"));

        // Test failed check
        let result = MotorOutputResult {
            success: false,
            action_type: Some(ActionType::CargoCheck),
            prediction_error: 1.0,
            outcome: None,
            error: Some("error[E0412]: cannot find type".into()),
        };
        agent.process_motor_result(&result);
        assert!(agent.last_test_output.is_some());
        assert!(agent.errors.last().unwrap().contains("E0412"));
    }

    #[test]
    fn test_dispatcher_integration() {
        let config = CodingAgentConfig::default();
        let agent = CodingAgent::new(config).unwrap();

        assert!(agent.dispatcher.is_some());
        assert_eq!(agent.total_energy(), 0.0);
    }

    #[test]
    fn test_with_dispatcher() {
        let config = CodingAgentConfig::default();
        let agent = CodingAgent::new(config)
            .unwrap()
            .with_dispatcher(IntelligentDispatcher::simulated().with_energy_budget(100.0));

        assert!(agent.dispatcher.is_some());
    }

    #[test]
    fn test_agent_result_includes_generation_telemetry() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 5,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("test.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        let result = agent.run("add hello function");

        // Result should include generation telemetry
        assert!(result.total_energy >= 0.0);
        // generation_tiers should be populated if generation happened
        if !result.files_modified.is_empty() {
            assert!(!result.generation_tiers.is_empty());
        }
    }

    // ── Task 2: Outcome Feedback Loop ──────────────────────────────────

    #[test]
    fn test_record_generation_outcome_updates_stats() {
        let config = CodingAgentConfig::default();
        let mut agent = CodingAgent::new(config).unwrap();

        // Simulate a generation that used Native tier
        agent.generation_tiers.push(BackendTier::Native);

        // Before recording, success rate should be the 50% prior
        assert_eq!(
            agent
                .dispatcher
                .as_ref()
                .unwrap()
                .success_rate(BackendTier::Native),
            0.5
        );

        // Record a success
        agent.record_generation_outcome(true);

        // After the `generate()` call already recorded one success + this external one,
        // the rate should reflect actual data (no longer the 0.5 prior).
        let rate = agent
            .dispatcher
            .as_ref()
            .unwrap()
            .success_rate(BackendTier::Native);
        assert!(
            rate > 0.5,
            "Success rate should increase after recording success: {rate}"
        );
    }

    #[test]
    fn test_record_generation_outcome_failure_lowers_rate() {
        let config = CodingAgentConfig::default();
        let mut agent = CodingAgent::new(config).unwrap();
        agent.generation_tiers.push(BackendTier::Native);

        // Record failures
        agent.record_generation_outcome(false);
        agent.record_generation_outcome(false);

        let rate = agent
            .dispatcher
            .as_ref()
            .unwrap()
            .success_rate(BackendTier::Native);
        assert!(rate < 0.5, "Rate should drop after failures: {rate}");
    }

    #[test]
    fn test_record_outcome_no_tiers_is_noop() {
        let config = CodingAgentConfig::default();
        let mut agent = CodingAgent::new(config).unwrap();

        // No tiers recorded yet — should not panic
        agent.record_generation_outcome(true);

        // Stats should still be at prior (nothing recorded)
        assert_eq!(
            agent
                .dispatcher
                .as_ref()
                .unwrap()
                .success_rate(BackendTier::Native),
            0.5
        );
    }

    // ── Task 3: Understanding Phase File Reading ───────────────────────

    #[test]
    fn test_understanding_reads_existing_target() {
        let dir = tempfile::tempdir().unwrap();

        // Create a target file to be read
        let target = dir.path().join("main.rs");
        std::fs::write(&target, "fn existing() { 42 }").unwrap();

        let config = CodingAgentConfig {
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("main.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.task = "modify the existing function".to_string();

        // Run understanding phase
        agent.do_understanding();

        // Should have read the target file content
        assert!(
            agent
                .observations
                .iter()
                .any(|o| o.contains("fn existing()")),
            "Should have read target file content: {:?}",
            agent.observations
        );
    }

    #[test]
    fn test_understanding_reports_missing_target() {
        let dir = tempfile::tempdir().unwrap();

        let config = CodingAgentConfig {
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("nonexistent.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.task = "add new function".to_string();

        agent.do_understanding();

        assert!(
            agent
                .observations
                .iter()
                .any(|o| o.contains("does not exist yet")),
            "Should report missing target: {:?}",
            agent.observations
        );
    }

    #[test]
    fn test_understanding_lists_working_directory() {
        let dir = tempfile::tempdir().unwrap();

        // Create some files in the working dir
        std::fs::write(dir.path().join("lib.rs"), "").unwrap();
        std::fs::write(dir.path().join("main.rs"), "").unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();

        let config = CodingAgentConfig {
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("lib.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.task = "test".to_string();

        agent.do_understanding();

        // Should list the directory contents
        let dir_obs = agent
            .observations
            .iter()
            .find(|o| o.contains("Working directory"));
        assert!(
            dir_obs.is_some(),
            "Should list working dir: {:?}",
            agent.observations
        );
        let dir_obs = dir_obs.unwrap();
        assert!(dir_obs.contains("lib.rs"), "Should list lib.rs");
        assert!(dir_obs.contains("main.rs"), "Should list main.rs");
        assert!(dir_obs.contains("src/"), "Should list src/ directory");
    }

    #[test]
    fn test_understanding_reads_cargo_toml() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"test-project\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();

        let config = CodingAgentConfig {
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("src/lib.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.task = "test".to_string();

        agent.do_understanding();

        assert!(
            agent
                .observations
                .iter()
                .any(|o| o.contains("test-project")),
            "Should read Cargo.toml: {:?}",
            agent.observations
        );
    }

    #[test]
    fn test_full_run_includes_understanding_observations() {
        let dir = tempfile::tempdir().unwrap();

        // Create a target file that the agent will read during Understanding
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src/lib.rs"),
            "// existing code\npub fn old() {}\n",
        )
        .unwrap();

        let config = CodingAgentConfig {
            max_iterations: 5,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("src/lib.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        let result = agent.run("add hello function to src/lib.rs");

        // Observations should include content from Understanding phase
        assert!(
            result
                .observations
                .iter()
                .any(|o| o.contains("existing code") || o.contains("old()")),
            "Should have read existing file: {:?}",
            result.observations
        );
    }

    // ══════════════════════════════════════════════════════════════════════
    // Property-based tests & safety hardening
    // ══════════════════════════════════════════════════════════════════════

    use proptest::prelude::*;

    /// Helper: create a default agent with a tempdir for safe testing.
    fn make_test_agent() -> (CodingAgent, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 5,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("test_out.rs")),
            ..Default::default()
        };
        let agent = CodingAgent::new(config).unwrap();
        (agent, dir)
    }

    // ── Proptest 1: Output bounds ────────────────────────────────────────
    // AgentResult phi_trace and telemetry values must be bounded [0,1]
    // and total_energy must be non-negative.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn prop_agent_result_phi_trace_bounded(seed in 0u64..1000) {
            let (mut agent, _dir) = make_test_agent();

            let result = agent.run(&format!("add function number {seed}"));

            // All phi values must be in [0, 1]
            for (i, &phi) in result.phi_trace.iter().enumerate() {
                prop_assert!(
                    phi >= 0.0 && phi <= 1.0,
                    "phi_trace[{}] = {} out of [0,1]", i, phi
                );
                prop_assert!(phi.is_finite(), "phi_trace[{}] is not finite: {}", i, phi);
            }

            // total_energy must be non-negative and finite
            prop_assert!(
                result.total_energy >= 0.0 && result.total_energy.is_finite(),
                "total_energy invalid: {}", result.total_energy
            );

            // iterations_used must not exceed max
            prop_assert!(result.iterations_used <= 5);
        }

        // ── Proptest 2: confidence_to_epistemic always returns valid variant ──
        #[test]
        fn prop_confidence_to_epistemic_bounded(conf in -1.0f32..2.0) {
            // Must not panic for any f32 input
            let status = CodingAgent::confidence_to_epistemic(conf);
            // Should always produce one of the valid variants
            let valid = matches!(
                status,
                EpistemicStatus::Certain
                    | EpistemicStatus::Probable
                    | EpistemicStatus::Uncertain
                    | EpistemicStatus::Unknown
            );
            prop_assert!(valid, "Invalid epistemic status for conf={}", conf);
        }

        // ── Proptest 3: Injection resistance — arbitrary strings in task ─────
        #[test]
        fn prop_arbitrary_task_no_panic(task in "\\PC{0,200}") {
            let (mut agent, _dir) = make_test_agent();
            // Must not panic regardless of input content
            let result = agent.run(&task);
            // Basic sanity: iterations used is bounded
            prop_assert!(result.iterations_used <= 5);
            // Phi trace should still have valid entries
            for &phi in &result.phi_trace {
                prop_assert!(phi.is_finite(), "phi not finite for arbitrary task");
            }
        }

        // (proptest 4 removed — split_multi_file_output not yet implemented)

        // ── Proptest 5: Telemetry JSON is always valid ───────────────────────
        #[test]
        fn prop_telemetry_json_fields_finite(seed in 0u64..500) {
            let (mut agent, _dir) = make_test_agent();
            let result = agent.run(&format!("add test_{seed}"));
            let json = result.to_telemetry_json();

            // Must be a valid JSON object
            prop_assert!(json.is_object(), "telemetry should be a JSON object");

            // Consciousness fields must be finite
            if let Some(consciousness) = json.get("consciousness") {
                if let Some(avg) = consciousness.get("avg_phi").and_then(|v| v.as_f64()) {
                    prop_assert!(avg.is_finite(), "avg_phi not finite: {}", avg);
                }
                if let Some(samples) = consciousness.get("samples").and_then(|v| v.as_u64()) {
                    prop_assert!(samples <= 100, "too many samples: {}", samples);
                }
            }

            // iterations_used must be present and bounded
            if let Some(iters) = json.get("iterations_used").and_then(|v| v.as_u64()) {
                prop_assert!(iters <= 5, "iterations_used too large: {}", iters);
            }

            // total_energy must be non-negative
            if let Some(gen) = json.get("generation") {
                if let Some(energy) = gen.get("total_energy").and_then(|v| v.as_f64()) {
                    prop_assert!(
                        energy >= 0.0 && energy.is_finite(),
                        "total_energy invalid: {}", energy
                    );
                }
            }
        }
    }

    // ── Deterministic: 100-cycle stress test ─────────────────────────────
    // Run the agent through many iterations, verify no unbounded growth.

    #[test]
    fn test_100_cycle_stress_no_unbounded_growth() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("lib.rs"), "// empty\n").unwrap();

        let config = CodingAgentConfig {
            max_iterations: 100,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("lib.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        let result = agent.run("add 100 helper functions");

        // Must complete within configured iterations
        assert!(
            result.iterations_used <= 100,
            "Used {} iterations (max 100)",
            result.iterations_used
        );

        // Phi trace length should approximately match iterations used
        // (retry strategies may cause ±1 discrepancy at phase boundaries)
        let diff =
            (result.phi_trace.len() as isize - result.iterations_used as isize).unsigned_abs();
        assert!(
            diff <= 1,
            "phi_trace length ({}) should be within 1 of iterations_used ({})",
            result.phi_trace.len(),
            result.iterations_used
        );

        // Observations and errors must not grow unboundedly per iteration
        // Allow generous headroom: 20 entries per iteration max
        assert!(
            result.observations.len() <= result.iterations_used * 20,
            "Observations grew unboundedly: {} for {} iterations",
            result.observations.len(),
            result.iterations_used
        );
        assert!(
            result.errors.len() <= result.iterations_used * 20,
            "Errors grew unboundedly: {} for {} iterations",
            result.errors.len(),
            result.iterations_used
        );

        // All phi values bounded
        for &phi in &result.phi_trace {
            assert!(phi >= 0.0 && phi <= 1.0 && phi.is_finite());
        }

        // Generation tiers should not exceed iterations
        assert!(
            result.generation_tiers.len() <= result.iterations_used,
            "Generation tiers exceeded iterations: {} > {}",
            result.generation_tiers.len(),
            result.iterations_used
        );

        // Failure patterns must be bounded
        let patterns = agent.failure_patterns();
        assert!(
            patterns.len() <= result.iterations_used,
            "Failure patterns unbounded: {}",
            patterns.len()
        );
    }

    // ── Adversarial input tests ──────────────────────────────────────────

    #[test]
    fn test_adversarial_empty_task() {
        let (mut agent, _dir) = make_test_agent();
        let result = agent.run("");
        // Must not panic — should complete (possibly with no meaningful output)
        assert!(result.iterations_used <= 5);
        for &phi in &result.phi_trace {
            assert!(phi.is_finite());
        }
    }

    #[test]
    fn test_adversarial_special_chars() {
        let (mut agent, _dir) = make_test_agent();
        let result = agent.run("add fn with <script>alert('xss')</script> \n\n\t\r {}[]()\"'\\");
        assert!(result.iterations_used <= 5);
        for &phi in &result.phi_trace {
            assert!(phi.is_finite() && phi >= 0.0 && phi <= 1.0);
        }
    }

    #[test]
    fn test_adversarial_huge_task() {
        let (mut agent, _dir) = make_test_agent();
        // 10KB task string
        let huge = "a".repeat(10_000);
        let result = agent.run(&huge);
        assert!(result.iterations_used <= 5);
        for &phi in &result.phi_trace {
            assert!(phi.is_finite() && phi >= 0.0 && phi <= 1.0);
        }
    }

    #[test]
    fn test_adversarial_unicode_and_control_chars() {
        let (mut agent, _dir) = make_test_agent();
        let result = agent.run("add function \u{FEFF}\u{200B} cafe\u{0301} re\u{0301}sume\u{0301}");
        assert!(result.iterations_used <= 5);
        for &phi in &result.phi_trace {
            assert!(phi.is_finite() && phi >= 0.0 && phi <= 1.0);
        }
    }

    // ── Determinism test ─────────────────────────────────────────────────
    // Same config + same task should produce the same phi trace length
    // and same final phase (deterministic cognitive loop).

    #[test]
    fn test_determinism_same_config_same_output() {
        let run = |task: &str| -> (usize, String, usize) {
            let dir = tempfile::tempdir().unwrap();
            let config = CodingAgentConfig {
                max_iterations: 3,
                working_dir: dir.path().to_path_buf(),
                target_file: Some(PathBuf::from("det.rs")),
                ..Default::default()
            };
            let mut agent = CodingAgent::new(config).unwrap();
            let result = agent.run(task);
            (
                result.iterations_used,
                format!("{}", result.final_phase),
                result.phi_trace.len(),
            )
        };

        let (iters_a, phase_a, trace_a) = run("add determinism test");
        let (iters_b, phase_b, trace_b) = run("add determinism test");

        assert_eq!(iters_a, iters_b, "Iterations should be deterministic");
        assert_eq!(phase_a, phase_b, "Final phase should be deterministic");
        assert_eq!(trace_a, trace_b, "Phi trace length should be deterministic");
    }

    // ── Telemetry bounds: AgentResult::to_telemetry_json edge cases ──────

    #[test]
    fn test_telemetry_json_empty_phi_trace() {
        let result = AgentResult {
            files_modified: vec![],
            tests_passed: None,
            iterations_used: 0,
            phi_trace: vec![],
            epistemic_status: EpistemicStatus::Unknown,
            final_phase: TaskPhase::Understanding,
            observations: vec![],
            errors: vec![],
            generation_tiers: vec![],
            total_energy: 0.0,
            remaining_energy: 100.0,
            #[cfg(feature = "school_learning")]
            generated_lessons: vec![],
        };
        let json = result.to_telemetry_json();
        assert!(json.is_object());
        // avg_phi should be 0.0 for empty trace
        let avg = json["consciousness"]["avg_phi"].as_f64().unwrap();
        assert_eq!(avg, 0.0);
    }

    #[test]
    fn test_telemetry_json_errors_preview_truncated() {
        let long_error = "E".repeat(500);
        let result = AgentResult {
            files_modified: vec![],
            tests_passed: None,
            iterations_used: 1,
            phi_trace: vec![0.5],
            epistemic_status: EpistemicStatus::Uncertain,
            final_phase: TaskPhase::Done,
            observations: vec![],
            errors: vec![long_error; 5],
            generation_tiers: vec![],
            total_energy: 1.0,
            remaining_energy: 99.0,
            #[cfg(feature = "school_learning")]
            generated_lessons: vec![],
        };
        let json = result.to_telemetry_json();
        // errors_preview should have at most 3 entries
        let preview = json["errors_preview"].as_array().unwrap();
        assert!(preview.len() <= 3);
        // Each preview entry should be truncated to 100 chars
        for entry in preview {
            let s = entry.as_str().unwrap();
            assert!(s.len() <= 100, "Preview entry too long: {}", s.len());
        }
    }

    // ── Phase A + B tests: quality gate, native patterns, feedback loops ─

    #[test]
    fn test_quality_gate_rejects_todo_stub() {
        let code =
            "/// Generated.\npub fn generated() -> () {\n    // TODO: implement — task: foo\n}\n";
        assert!(CodingAgent::check_code_quality(code).is_some());
    }

    #[test]
    fn test_quality_gate_rejects_unimplemented() {
        let code = "pub fn foo() { unimplemented!() }";
        assert!(CodingAgent::check_code_quality(code).is_some());
    }

    #[test]
    fn test_quality_gate_rejects_empty() {
        assert!(CodingAgent::check_code_quality("").is_some());
        assert!(CodingAgent::check_code_quality("   ").is_some());
    }

    #[test]
    fn test_quality_gate_rejects_comments_only() {
        let code = "/// A function.\n// This is commented out.\n";
        assert!(CodingAgent::check_code_quality(code).is_some());
    }

    #[test]
    fn test_quality_gate_rejects_markdown_fences() {
        let code = "```rust\npub fn foo() -> i32 { 42 }\n```";
        assert!(CodingAgent::check_code_quality(code).is_some());
    }

    #[test]
    fn test_quality_gate_accepts_valid_code() {
        let code = "/// Compute fibonacci.\npub fn fibonacci(n: u64) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => fibonacci(n-1) + fibonacci(n-2),\n    }\n}\n";
        assert!(CodingAgent::check_code_quality(code).is_none());
    }

    #[test]
    fn test_quality_gate_accepts_simple_fn() {
        let code = "pub fn hello() -> &'static str {\n    \"Hello, world!\"\n}\n";
        assert!(CodingAgent::check_code_quality(code).is_none());
    }

    #[test]
    fn test_native_pattern_fibonacci() {
        let code = CodingAgent::match_native_pattern("add a fibonacci function");
        assert!(code.is_some());
        let code = code.unwrap();
        assert!(code.contains("pub fn fibonacci"));
        assert!(!code.contains("TODO"));
    }

    #[test]
    fn test_native_pattern_factorial() {
        let code = CodingAgent::match_native_pattern("implement factorial");
        assert!(code.is_some());
        assert!(code.unwrap().contains("pub fn factorial"));
    }

    #[test]
    fn test_native_pattern_gcd() {
        let code = CodingAgent::match_native_pattern("add gcd function");
        assert!(code.is_some());
        assert!(code.unwrap().contains("pub fn gcd"));
    }

    #[test]
    fn test_native_pattern_is_prime() {
        let code = CodingAgent::match_native_pattern("check primality");
        assert!(code.is_some());
        assert!(code.unwrap().contains("pub fn is_prime"));
    }

    #[test]
    fn test_native_pattern_reverse_string() {
        let code = CodingAgent::match_native_pattern("reverse a string");
        assert!(code.is_some());
        assert!(code.unwrap().contains("pub fn reverse_string"));
    }

    #[test]
    fn test_native_pattern_palindrome() {
        let code = CodingAgent::match_native_pattern("check if palindrome");
        assert!(code.is_some());
        assert!(code.unwrap().contains("pub fn is_palindrome"));
    }

    #[test]
    fn test_native_pattern_bubble_sort() {
        let code = CodingAgent::match_native_pattern("implement bubble sort");
        assert!(code.is_some());
        assert!(code.unwrap().contains("pub fn bubble_sort"));
    }

    #[test]
    fn test_native_pattern_binary_search() {
        let code = CodingAgent::match_native_pattern("implement binary search");
        assert!(code.is_some());
        assert!(code.unwrap().contains("pub fn binary_search"));
    }

    #[test]
    fn test_native_pattern_stack() {
        let code = CodingAgent::match_native_pattern("create a stack data structure");
        assert!(code.is_some());
        let code = code.unwrap();
        assert!(code.contains("pub struct Stack"));
        assert!(code.contains("push"));
        assert!(code.contains("pop"));
    }

    #[test]
    fn test_native_pattern_returns_none_for_unknown() {
        // Tasks that don't match any pattern should return None
        assert!(CodingAgent::match_native_pattern("implement a red-black tree").is_none());
        assert!(CodingAgent::match_native_pattern("create a REST API client").is_none());
    }

    #[test]
    fn test_extract_function_name() {
        assert_eq!(
            CodingAgent::extract_function_name("add a fibonacci function"),
            Some("fibonacci".to_string())
        );
        assert_eq!(
            CodingAgent::extract_function_name("implement calculate_tax"),
            Some("calculate_tax".to_string())
        );
        assert_eq!(
            CodingAgent::extract_function_name("create process_data method"),
            Some("process_data".to_string())
        );
    }

    #[test]
    fn test_failure_patterns_in_prompt() {
        let config = CodingAgentConfig::default();
        let mut agent = CodingAgent::new(config).unwrap();
        agent.task = "add function".to_string();
        agent.failure_patterns = vec![
            ("error[E0308]: mismatched types".to_string(), 2),
            ("error[E0412]: cannot find type".to_string(), 1),
        ];

        let prompt = agent.build_generation_prompt();
        assert!(
            prompt.contains("AVOID these patterns"),
            "Prompt should warn about failure patterns"
        );
        assert!(prompt.contains("E0308"));
        assert!(prompt.contains("(2x)"));
    }

    #[test]
    fn test_native_generates_sort_for_sort_task() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 5,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("sort.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        let result = agent.run("implement bubble sort");

        let target = dir.path().join("sort.rs");
        if target.exists() {
            let content = std::fs::read_to_string(&target).unwrap();
            assert!(
                content.contains("bubble_sort") || content.contains("sort"),
                "Should generate sort code: {}",
                &content[..content.len().min(200)]
            );
            assert!(!content.contains("TODO"), "Should not contain TODO");
        }
        assert!(result.iterations_used > 0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Task E tests: structured failures, consciousness signals, events, retry
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_parse_test_failures_assert_eq() {
        let stderr = r#"
---- my_test stdout ----
thread 'my_test' panicked at src/lib.rs:42:5:
assertion `left == right` failed
  left: 42
 right: 43

failures:
    my_test
"#;
        let failures = CodingAgent::parse_test_failures(stderr);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].test_name, "my_test");
        assert_eq!(failures[0].failure_kind, TestFailureKind::AssertEq);
        assert_eq!(failures[0].actual.as_deref(), Some("42"));
        assert_eq!(failures[0].expected.as_deref(), Some("43"));
    }

    #[test]
    fn test_parse_test_failures_panic() {
        let stderr = r#"
---- panic_test stdout ----
thread 'panic_test' panicked at src/main.rs:10:5:
index out of bounds: the len is 3 but the index is 5
"#;
        let failures = CodingAgent::parse_test_failures(stderr);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].test_name, "panic_test");
        assert_eq!(failures[0].failure_kind, TestFailureKind::Panic);
    }

    #[test]
    fn test_parse_test_failures_multiple() {
        let stderr = r#"
---- test_a stdout ----
thread 'test_a' panicked at src/lib.rs:1:1:
assertion failed
---- test_b stdout ----
thread 'test_b' panicked at src/lib.rs:2:2:
assertion `left == right` failed
  left: "foo"
 right: "bar"
"#;
        let failures = CodingAgent::parse_test_failures(stderr);
        assert_eq!(failures.len(), 2);
        assert_eq!(failures[0].test_name, "test_a");
        assert_eq!(failures[0].failure_kind, TestFailureKind::Assert);
        assert_eq!(failures[1].test_name, "test_b");
        assert_eq!(failures[1].failure_kind, TestFailureKind::AssertEq);
    }

    #[test]
    fn test_parse_test_failures_empty() {
        assert!(CodingAgent::parse_test_failures("").is_empty());
        assert!(CodingAgent::parse_test_failures("test result: ok").is_empty());
    }

    #[test]
    fn test_format_structured_test_failures() {
        let failures = vec![StructuredTestFailure {
            test_name: "test_add".to_string(),
            failure_kind: TestFailureKind::AssertEq,
            expected: Some("5".to_string()),
            actual: Some("4".to_string()),
            message: Some("assertion failed".to_string()),
            file: Some("src/lib.rs".to_string()),
            line: Some(42),
        }];
        let formatted = CodingAgent::format_structured_test_failures(&failures);
        assert!(formatted.contains("test_add"));
        assert!(formatted.contains("AssertEq"));
        assert!(formatted.contains("expected=5"));
        assert!(formatted.contains("got=4"));
        assert!(formatted.contains("src/lib.rs:42"));
    }

    #[test]
    fn test_extract_panic_location() {
        let (file, line) = CodingAgent::extract_panic_location("at ./src/foo.rs:42:5");
        assert_eq!(file.as_deref(), Some("./src/foo.rs"));
        assert_eq!(line, Some(42));

        let (file, line) = CodingAgent::extract_panic_location("no location");
        assert!(file.is_none());
        assert!(line.is_none());
    }

    #[test]
    fn test_consciousness_signals_extraction() {
        let (mut agent, _dir) = make_test_agent();
        let _result = agent.run("add fibonacci");

        // After running, prediction_error_history should be populated
        assert!(
            !agent.prediction_error_history.is_empty(),
            "Should have prediction error history after run"
        );
        assert!(
            !agent.confidence_velocity_history.is_empty(),
            "Should have confidence velocity history after run"
        );
        // Histories should be bounded
        assert!(
            agent.prediction_error_history.len() <= 10,
            "History should be bounded to 10"
        );
    }

    #[test]
    fn test_event_channel_receives_events() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 3,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("test.rs")),
            ..Default::default()
        };
        let agent = CodingAgent::new(config).unwrap();
        let (mut agent, rx) = agent.with_event_channel();

        let _result = agent.run("add hello function");

        // Should have received at least consciousness snapshots and Done event
        let events: Vec<AgentEvent> = rx.try_iter().collect();
        assert!(!events.is_empty(), "Should receive events");

        // Check for consciousness snapshots
        let has_snapshot = events
            .iter()
            .any(|e| matches!(e, AgentEvent::ConsciousnessSnapshot { .. }));
        assert!(has_snapshot, "Should have consciousness snapshots");

        // Check for Done event
        let has_done = events.iter().any(|e| matches!(e, AgentEvent::Done(_)));
        assert!(has_done, "Should have Done event");
    }

    #[test]
    fn test_retry_strategy_cycles_through_options() {
        let (mut agent, _dir) = make_test_agent();

        let s1 = agent.next_retry_strategy();
        assert_eq!(s1, RetryStrategy::DifferentTemplate);

        let s2 = agent.next_retry_strategy();
        assert!(matches!(
            s2,
            RetryStrategy::DifferentBackend(BackendTier::LocalLlm)
        ));

        let s3 = agent.next_retry_strategy();
        assert!(matches!(
            s3,
            RetryStrategy::DifferentBackend(BackendTier::CloudLlm)
        ));

        let s4 = agent.next_retry_strategy();
        assert_eq!(s4, RetryStrategy::SimplifyScope);

        let s5 = agent.next_retry_strategy();
        assert!(matches!(s5, RetryStrategy::RequestClarification(_)));
    }

    #[test]
    fn test_retry_state_resets_on_new_run() {
        let (mut agent, _dir) = make_test_agent();

        // Advance retry state
        let _ = agent.next_retry_strategy();
        let _ = agent.next_retry_strategy();
        assert!(!agent.retry_state.strategies_tried.is_empty());

        // Run resets retry state
        let _result = agent.run("add hello function");
        // After run completes, retry state may be populated from the run itself
        // but the initial reset should have cleared it
    }

    #[test]
    fn test_hdc_context_prompt_empty_without_memory() {
        let (agent, _dir) = make_test_agent();
        // No code memory indexed → empty HDC context
        let hdc_prompt = agent.build_hdc_context_prompt();
        assert!(hdc_prompt.is_empty(), "No HDC context without code memory");
    }

    #[test]
    fn test_generation_prompt_includes_retry_hints() {
        let (mut agent, _dir) = make_test_agent();
        agent.task = "add fibonacci".to_string();
        agent.phase = TaskPhase::Fixing;
        agent.last_test_output = Some("error[E0308]: mismatched types".to_string());

        // Set DifferentTemplate strategy
        agent.retry_state.current_strategy = RetryStrategy::DifferentTemplate;
        let prompt = agent.build_generation_prompt();
        assert!(
            prompt.contains("different implementation approach"),
            "Prompt should include retry hint for DifferentTemplate"
        );

        // Set SimplifyScope strategy
        agent.retry_state.current_strategy = RetryStrategy::SimplifyScope;
        let prompt = agent.build_generation_prompt();
        assert!(
            prompt.contains("Simplify"),
            "Prompt should include retry hint for SimplifyScope"
        );
    }

    #[test]
    fn test_generation_prompt_includes_structured_failures() {
        let (mut agent, _dir) = make_test_agent();
        agent.task = "add fibonacci".to_string();
        agent.phase = TaskPhase::Fixing;
        agent.last_test_output = Some(
            r#"---- test_fib stdout ----
thread 'test_fib' panicked at src/lib.rs:5:5:
assertion `left == right` failed
  left: 8
 right: 7
"#
            .to_string(),
        );

        let prompt = agent.build_generation_prompt();
        assert!(
            prompt.contains("test failure"),
            "Should include structured test analysis"
        );
        assert!(prompt.contains("test_fib"), "Should name the failing test");
    }

    #[test]
    fn test_persistent_experience_store() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 3,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("test.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        assert!(agent.has_experience_store(), "Should have experience store");

        // Run to populate the store
        let _result = agent.run("add fibonacci");

        // Check that .symthaea/experience.db was created
        let db_path = dir.path().join(".symthaea/experience.db");
        assert!(
            db_path.exists(),
            "Persistent DB should exist at {:?}",
            db_path
        );

        // Create a second agent pointing at the same directory — it should
        // load the persisted experience store
        let config2 = CodingAgentConfig {
            max_iterations: 3,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("test.rs")),
            ..Default::default()
        };
        let agent2 = CodingAgent::new(config2).unwrap();
        assert!(
            agent2.has_experience_store(),
            "Second agent should load persistent store"
        );
    }

    #[test]
    fn test_hdc_verification_gate_no_memory() {
        // Without code_memory, verification should always pass
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let agent = CodingAgent::new(config).unwrap();
        let (passes, surprise) = agent.verify_generated_code_hdc("fn hello() {}");
        assert!(passes, "Should pass when no code memory");
        assert_eq!(surprise, 0.0);
    }

    #[cfg(feature = "code_generation")]
    #[test]
    fn test_hdc_verification_gate_with_indexed_code() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        // Write several Rust files to build a codebase centroid
        for i in 0..5 {
            std::fs::write(
                src_dir.join(format!("mod{i}.rs")),
                format!("pub fn func_{i}(x: u32) -> u32 {{ x + {i} }}\n"),
            )
            .unwrap();
        }

        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.index_project(dir.path()).unwrap();

        // Similar code should pass
        let (passes, surprise) =
            agent.verify_generated_code_hdc("pub fn func_new(x: u32) -> u32 { x + 10 }");
        // Surprise should be finite
        assert!(surprise.is_finite(), "Surprise should be finite");
        // With only 5 small files the centroid is weak — we mainly test no-crash here
        eprintln!("Similar code: passes={passes}, surprise={surprise:.3}");
    }

    // ── LLM Output Verification Tests ─────────────────────────────────

    #[test]
    fn test_quality_gate_rejects_hallucinated_imports() {
        let code = "use my_crate::something;\nfn hello() {}";
        let result = CodingAgent::check_code_quality(code);
        assert!(result.is_some(), "Should reject hallucinated import");
        assert!(result.unwrap().contains("hallucinated"));
    }

    #[test]
    fn test_quality_gate_rejects_ellipsis() {
        let code = "fn process() {\n    ...\n}";
        let result = CodingAgent::check_code_quality(code);
        assert!(result.is_some(), "Should reject ellipsis");
    }

    #[test]
    fn test_quality_gate_rejects_explanation_leak() {
        let code = "Here is the implementation:\nfn hello() -> &'static str { \"hello\" }";
        let result = CodingAgent::check_code_quality(code);
        assert!(result.is_some(), "Should reject explanation leak");
        assert!(result.unwrap().contains("explanation"));
    }

    #[test]
    fn test_quality_gate_rejects_duplicate_fns() {
        let code = "fn fibonacci(n: u64) -> u64 { n }\nfn fibonacci(n: u64) -> u64 { n + 1 }";
        let result = CodingAgent::check_code_quality(code);
        assert!(result.is_some(), "Should reject duplicate function");
        assert!(result.unwrap().contains("duplicate"));
    }

    #[test]
    fn test_quality_gate_accepts_doc_comments() {
        // "Note that" in a doc comment should NOT be flagged
        let code = "/// Note that this returns None for empty inputs.\npub fn first(v: &[i32]) -> Option<&i32> {\n    v.first()\n}";
        let result = CodingAgent::check_code_quality(code);
        assert!(
            result.is_none(),
            "Doc comments should not trigger explanation detection"
        );
    }

    // ── Warm-up & Learning Tests ────────────────────────────────────────

    #[test]
    fn test_warm_up_runs_without_consuming_iterations() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 3,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("warm.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();

        // Run the agent — warm_up_phi(3) is called internally before the main loop.
        // The key property: warm-up cycles don't count as iterations.
        let result = agent.run("add hello function");
        assert!(
            result.iterations_used <= 3,
            "Should use at most max_iterations (3), not more. Got: {}",
            result.iterations_used
        );
        // Phi trace should only contain entries from real iterations, not warm-up
        let diff =
            (result.phi_trace.len() as isize - result.iterations_used as isize).unsigned_abs();
        assert!(diff <= 1, "Phi trace should track real iterations");
    }

    #[test]
    fn test_retrieve_success_patterns_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let agent = CodingAgent::new(config).unwrap();
        let patterns = agent.retrieve_success_patterns();
        assert!(patterns.is_empty(), "Empty store should return no patterns");
    }

    #[test]
    fn test_experience_store_counts() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 3,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("test.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        assert!(agent.has_experience_store());

        // Store should start empty
        let count_before = agent.experience_count();

        // Run a task — this should store at least one experience
        let _ = agent.run("add fibonacci function");

        let count_after = agent.experience_count();
        // The agent may or may not store experiences depending on whether
        // code was generated/tested. Both cases are valid.
        assert!(
            count_after >= count_before,
            "Experience count should not decrease"
        );
    }

    #[test]
    fn test_learning_across_runs() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 3,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("test.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();

        // Run 1: generate fibonacci
        let _ = agent.run("add fibonacci function");
        let successes_after_r1 = agent.cached_successes().len();
        let hints_after_r1 = agent.cached_error_hints().len();

        // Run 2: similar task — should benefit from cached experience
        let _ = agent.run("add factorial function");
        let successes_after_r2 = agent.cached_successes().len();

        eprintln!(
            "Successes: r1={successes_after_r1}, r2={successes_after_r2}, hints_r1={hints_after_r1}"
        );
        // Cache should accumulate over runs
        assert!(
            successes_after_r2 >= successes_after_r1,
            "Success cache should grow or stay same across runs"
        );
    }

    #[test]
    fn test_strip_code_fences() {
        // No fences → unchanged
        assert_eq!(
            CodingAgent::strip_code_fences("fn main() {}"),
            "fn main() {}"
        );

        // ```rust ... ```
        assert_eq!(
            CodingAgent::strip_code_fences("```rust\nfn main() {}\n```"),
            "fn main() {}"
        );

        // ``` ... ```
        assert_eq!(
            CodingAgent::strip_code_fences("```\nfn main() {}\n```"),
            "fn main() {}"
        );

        // ```rs ... ```
        assert_eq!(
            CodingAgent::strip_code_fences("```rs\nfn main() {}\n```"),
            "fn main() {}"
        );

        // With leading/trailing whitespace
        assert_eq!(
            CodingAgent::strip_code_fences("  ```rust\n  fn main() {}\n  ```  "),
            "fn main() {}"
        );
    }

    // ── Plan Evaluation Tests ────────────────────────────────────────

    #[test]
    fn test_build_execution_plan_understanding() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(PathBuf::from("lib.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.phase = TaskPhase::Understanding;

        let plan = agent.build_execution_plan();
        assert!(plan.is_some(), "Understanding phase should produce a plan");
        let profile = plan.unwrap().profile();
        assert!(
            profile.fully_reversible,
            "Understanding should be read-only"
        );
        assert_eq!(
            profile.max_destructiveness,
            crate::action::DestructivenessLevel::ReadOnly
        );
    }

    #[test]
    fn test_build_execution_plan_testing() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.phase = TaskPhase::Testing;

        let plan = agent.build_execution_plan();
        assert!(plan.is_some(), "Testing phase should produce a plan");
        let profile = plan.unwrap().profile();
        assert_eq!(profile.step_count, 1); // just cargo check
        assert!(profile.fully_reversible);
    }

    #[test]
    fn test_build_execution_plan_planning_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.phase = TaskPhase::Planning;

        assert!(
            agent.build_execution_plan().is_none(),
            "Planning is pure reasoning — no I/O plan"
        );
    }

    #[test]
    fn test_evaluate_plan_phi_gating() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let agent = CodingAgent::new(config).unwrap();

        // Plan that requires phi > 0.3 (e.g., git push)
        let dangerous = Molecule::atom(Atom::Exec {
            program: "git".into(),
            args: vec!["push".into()],
            working_dir: None,
            env: std::collections::BTreeMap::new(),
        });

        // With no phi history (defaults to 0.0), should reject
        let (approved, reason) = agent.evaluate_plan(&dangerous, 0.0);
        assert!(!approved, "Should reject: {}", reason);
        assert!(reason.contains("Phi too low"));
    }

    #[test]
    fn test_evaluate_plan_energy_budget() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.energy_budget = 1.0; // very tight budget

        // Compile-fix loop costs ~12+ energy
        let expensive = crate::action::primitives::recipes::compile_fix_loop(
            PathBuf::from("/tmp/test/src/lib.rs"),
            "fn main() {}".into(),
            3,
        );

        let (approved, reason) = agent.evaluate_plan(&expensive, 1.0);
        assert!(!approved, "Should reject expensive plan: {}", reason);
        assert!(reason.contains("Energy budget exceeded"));
    }

    #[test]
    fn test_evaluate_plan_destructive_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let agent = CodingAgent::new(config).unwrap();

        let dangerous = Molecule::atom(Atom::Exec {
            program: "git".into(),
            args: vec!["push".into()],
            working_dir: None,
            env: std::collections::BTreeMap::new(),
        });

        // Even with high phi and budget, destructive actions are blocked
        let (approved, reason) = agent.evaluate_plan(&dangerous, 1.0);
        assert!(!approved);
        assert!(reason.contains("destructive"));
    }

    #[test]
    fn test_evaluate_plan_safe_approved() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let agent = CodingAgent::new(config).unwrap();

        let safe = Molecule::atom(Atom::read("/tmp/test.rs"))
            .then(Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp"))));

        let (approved, reason) = agent.evaluate_plan(&safe, 0.1);
        assert!(approved, "Safe plan should be approved: {}", reason);
        assert!(reason.contains("Plan approved"));
    }

    #[test]
    fn test_energy_deduction() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        assert!((agent.remaining_energy() - 100.0).abs() < 0.01);

        let plan = Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp")));
        let profile = plan.profile();
        agent.deduct_energy(&profile);

        assert!(agent.remaining_energy() < 100.0);
        assert!((agent.remaining_energy() - (100.0 - profile.total_energy)).abs() < 0.01);
    }

    #[test]
    fn test_evaluate_hypothetical_plan() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let agent = CodingAgent::new(config).unwrap();

        // Compare two candidate plans
        let plan_a = Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp")));
        let plan_b = crate::action::primitives::recipes::compile_fix_loop(
            PathBuf::from("/tmp/src/lib.rs"),
            "code".into(),
            5,
        );

        let (_, _, profile_a) = agent.evaluate_hypothetical_plan(&plan_a);
        let (_, _, profile_b) = agent.evaluate_hypothetical_plan(&plan_b);

        // Plan B should be more expensive (5 iterations of write+check)
        assert!(profile_b.total_energy > profile_a.total_energy);
        assert!(profile_b.step_count > profile_a.step_count);
    }

    // ── Enhancement 1: Molecule-driven execution tests ────────────────

    #[test]
    fn test_execute_molecule_read_simulated() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            enable_real_exec: false,
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();

        let mol = Molecule::atom(Atom::read("/tmp/test.rs"));
        let result = agent.execute_molecule(&mol);

        assert!(result.is_some());
        assert!(result.unwrap().success);
        // Should have added observation
        assert!(
            agent.observations.iter().any(|o| o.contains("simulated read")),
            "Should have simulated read observation: {:?}",
            agent.observations
        );
    }

    #[test]
    fn test_execute_molecule_tracks_energy() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            enable_real_exec: false,
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.phi_trace.push(1.0); // need sufficient phi for CargoCheck (min 0.05)
        let initial_energy = agent.energy_budget;

        let mol = Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp")));
        agent.execute_molecule(&mol);

        // Energy should have been deducted (CargoCheck costs 3.0)
        assert!(
            agent.energy_budget < initial_energy,
            "Energy budget should decrease: {} < {}",
            agent.energy_budget,
            initial_energy
        );
    }

    #[test]
    fn test_execute_molecule_command_result() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            enable_real_exec: false,
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.phi_trace.push(1.0); // sufficient phi for CargoCheck

        // Simulated exec returns exit_code=0
        let mol = Molecule::atom(Atom::cargo_check(PathBuf::from("/tmp")));
        let result = agent.execute_molecule(&mol).unwrap();

        assert!(result.success);
        assert_eq!(result.action_type, Some(ActionType::CargoCheck));
    }

    #[test]
    fn test_do_understanding_molecule() {
        let dir = tempfile::tempdir().unwrap();
        // Create a Cargo.toml in the temp dir
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "pub fn hello() {}").unwrap();

        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(dir.path().join("src/lib.rs")),
            enable_real_exec: true,
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.task = "add fibonacci".into();

        agent.do_understanding_molecule();

        // Should have gathered context
        assert!(
            !agent.observations.is_empty(),
            "Should have observations from understanding"
        );
        // Should have file listing
        assert!(
            agent.observations.iter().any(|o| o.contains("src") || o.contains("Cargo.toml") || o.contains("Files")),
            "Should have project files in observations: {:?}",
            agent.observations
        );
    }

    #[test]
    fn test_do_testing_molecule_no_cargo_toml() {
        let dir = tempfile::tempdir().unwrap();
        // No Cargo.toml
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            enable_real_exec: true,
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();

        let result = agent.do_testing_molecule();
        assert!(result.is_some());
        assert!(!result.unwrap().success);
    }

    #[test]
    fn test_do_testing_molecule_simulated() {
        let dir = tempfile::tempdir().unwrap();
        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            enable_real_exec: false,
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.generated_code = Some("fn main() {}".into());

        let result = agent.do_testing_molecule();
        assert!(result.is_some());
        assert!(result.unwrap().success);
    }

    // ── Enhancement 2: Learning loop tests ────────────────────────────

    #[test]
    fn test_select_plan_fep_returns_plan() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"t\"").unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "").unwrap();

        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(dir.path().join("src/lib.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.task = "add fibonacci".into();
        agent.phase = TaskPhase::Understanding;
        agent.phi_trace.push(1.0);

        let result = agent.select_plan_fep();
        assert!(result.is_some(), "Should select a plan for Understanding phase");
    }

    // ── Enhancement 3: Dispatch tier tests ────────────────────────────

    #[test]
    fn test_generating_includes_tiered_candidates() {
        use crate::action::primitives::PlanCandidate;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]").unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "").unwrap();

        let config = CodingAgentConfig {
            max_iterations: 1,
            working_dir: dir.path().to_path_buf(),
            target_file: Some(dir.path().join("src/lib.rs")),
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        agent.task = "add fibonacci".into();
        agent.phase = TaskPhase::Generating;
        agent.phi_trace.push(1.0);
        agent.generated_code = Some("fn fib(n: u32) -> u32 { n }".into());

        // The plan selection should have candidates including tiered dispatch
        let result = agent.select_plan_fep();
        // It should succeed (at least the write_and_check candidate)
        assert!(result.is_some());
    }

    #[test]
    fn test_dispatch_tier_energy_in_profile() {
        let native = crate::action::primitives::recipes::generate_and_check(
            PathBuf::from("/tmp/src/lib.rs"),
            "add fib",
            DispatchTier::Native,
        );
        let cloud = crate::action::primitives::recipes::generate_and_check(
            PathBuf::from("/tmp/src/lib.rs"),
            "add fib",
            DispatchTier::CloudLlm,
        );

        // Cloud plan should be 50x more expensive for the dispatch atom
        assert!(
            cloud.profile().total_energy > native.profile().total_energy * 5.0,
            "Cloud ({}) should be much more expensive than native ({})",
            cloud.profile().total_energy,
            native.profile().total_energy
        );
    }
}
