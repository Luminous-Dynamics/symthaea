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
    /// Indexed codebase memory for semantic code search (populated by `index_project()`).
    #[cfg(feature = "code_generation")]
    code_memory: Option<crate::hdc::code_memory::CodebaseMemory>,
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
                Ok(bridge)
            } else {
                MotorOutputBridge::with_defaults()
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
            experience_store: Self::try_init_experience_store(),
            failure_patterns: Vec::new(),
            #[cfg(feature = "code_generation")]
            code_memory: None,
        }
    }

    /// Attempt to create an in-memory experience store. Returns None on failure
    /// (non-blocking — agent works fine without it).
    fn try_init_experience_store() -> Option<CodingExperienceStore> {
        // Use tokio if available, otherwise skip
        let rt = tokio::runtime::Runtime::new().ok()?;
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

        // If we have indexed codebase memory, query for relevant context
        #[cfg(feature = "code_generation")]
        if let Some(ref memory) = self.code_memory {
            let encoder = memory.encoder();
            let intent_hv = encoder.encode_name(task);
            let matches = memory.query(&intent_hv, 5);
            let context: Vec<String> = matches
                .iter()
                .filter(|m| m.similarity > 0.2)
                .map(|m| {
                    format!(
                        "{:?} `{}` in {} (sim: {:.3})",
                        m.kind,
                        m.name,
                        m.path.display(),
                        m.similarity
                    )
                })
                .collect();
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

        self.build_result()
    }

    /// Execute one step of the agent loop.
    fn step(&mut self) {
        // 1. Phase-specific pre-cycle action (code generation, etc.)
        self.pre_cycle_action();

        // 2. Build observation with updated context
        let observation = self.build_observation();

        // 3. Set up the motor request based on current phase
        let motor_request = self.build_motor_request();
        self.cognitive_loop.set_motor_request(motor_request);

        // 4. Run one cognitive cycle
        let cycle_result = self.cognitive_loop.cycle(&observation);

        // Record Phi from metadata
        let phi = cycle_result.metadata.consciousness_level as f32;
        self.phi_trace.push(phi);

        // 5. Check for motor output result
        let motor_result = self.cognitive_loop.take_motor_result();

        // 6. Process the cycle result and motor output
        self.process_step_result(&cycle_result, motor_result, phi);
    }

    // ── Pre-Cycle Actions ──────────────────────────────────────────────

    /// Phase-specific actions performed before the cognitive cycle.
    ///
    /// - Understanding: reads the target file and nearby source files
    /// - Generating/Fixing: calls IntelligentDispatcher to generate code
    fn pre_cycle_action(&mut self) {
        match self.phase {
            TaskPhase::Understanding => {
                self.do_understanding();
            }
            TaskPhase::Generating | TaskPhase::Fixing => {
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
    }

    /// Generate code via the IntelligentDispatcher and write to disk.
    fn do_generation(&mut self) {
        // Get consciousness state for dispatch routing
        let confidence = self.cognitive_loop.prediction_confidence();
        let epistemic = Self::confidence_to_epistemic(confidence);
        let phi = self.phi_trace.last().copied().unwrap_or(0.5) as f64;
        let prediction_error = self.cognitive_loop.prediction_confidence(); // inverse proxy

        // Build the generation prompt
        let prompt = self.build_generation_prompt();

        // Call the dispatcher (async → sync bridge)
        let dispatch_result = if let Some(ref mut dispatcher) = self.dispatcher {
            let params = GenerationParams {
                temperature: 0.3, // low temp for code generation
                max_tokens: 1024,
                system_prompt: Some(
                    "You are a code generator. Output ONLY valid source code, no explanations."
                        .into(),
                ),
            };

            // Sync bridge for async dispatcher
            let result = Self::block_on_dispatch(
                dispatcher,
                &prompt,
                &params,
                epistemic,
                prediction_error as f64,
                phi,
            );
            Some(result)
        } else {
            None
        };

        // Process the dispatch result
        if let Some(result) = dispatch_result {
            self.generation_tiers.push(result.tier);

            if result.success && result.tier != BackendTier::Native {
                // LLM-generated code — write to disk
                let target = self.resolve_target_file();
                self.write_code_to_disk(&target, &result.output);
                self.generated_code = Some(result.output.clone());

                tracing::info!(
                    target: "symthaea::coding_agent",
                    tier = %result.tier,
                    energy = result.energy_cost,
                    target = %target.display(),
                    "Code generated and written"
                );
            } else if result.tier == BackendTier::Native {
                // Native tier — generate a placeholder that the HDC+CfC pipeline
                // would produce. For now, use a structural template.
                let code = self.native_code_template();
                let target = self.resolve_target_file();
                self.write_code_to_disk(&target, &code);
                self.generated_code = Some(code);
            } else {
                self.errors.push(format!(
                    "Generation failed ({}): {}",
                    result.tier, result.output
                ));
            }

            self.last_dispatch = Some(result);
        }
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

        // In Fixing phase, include the error to fix
        if self.phase == TaskPhase::Fixing {
            if let Some(ref test_output) = self.last_test_output {
                prompt.push_str(&format!(
                    "The previous code failed with this error:\n```\n{}\n```\n\nFix the code.\n",
                    test_output
                ));
            }
        }

        // Inject experience hints from persistent store
        let hints = self.retrieve_experience_hints();
        if !hints.is_empty() {
            prompt.push_str("Relevant patterns from past experience:\n");
            for (pattern, hint) in hints.iter().take(3) {
                prompt.push_str(&format!("- Error: {} → Fix: {}\n", pattern, hint));
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
                return rt.block_on(async {
                    store.error_hints_for(&self.task, 3).await
                });
            }
        }
        Vec::new()
    }

    /// Generate a structural template for native (non-LLM) code generation.
    fn native_code_template(&self) -> String {
        // Extract function/type names from the task
        let task_lower = self.task.to_lowercase();

        // Detect common patterns and generate appropriate templates
        if task_lower.contains("fibonacci") || task_lower.contains("fib") {
            "/// Compute the nth Fibonacci number.\npub fn fibonacci(n: u64) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => {\n            let (mut a, mut b) = (0u64, 1u64);\n            for _ in 2..=n {\n                let c = a.saturating_add(b);\n                a = b;\n                b = c;\n            }\n            b\n        }\n    }\n}\n".to_string()
        } else if task_lower.contains("hello") {
            "/// Return a greeting.\npub fn hello() -> &'static str {\n    \"Hello, world!\"\n}\n"
                .to_string()
        } else {
            // Generic function stub
            format!(
                "/// Generated by Symthaea coding agent.\npub fn generated() -> () {{\n    // TODO: implement — task: {}\n}}\n",
                self.task
            )
        }
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

    /// Write code to disk, creating parent directories as needed.
    fn write_code_to_disk(&mut self, target: &PathBuf, code: &str) {
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

        match std::fs::write(target, code) {
            Ok(()) => {
                if !self.files_modified.contains(target) {
                    self.files_modified.push(target.clone());
                }
                self.observations.push(format!(
                    "Wrote {} bytes to {}",
                    code.len(),
                    target.display()
                ));
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
        if self.phase != TaskPhase::Done {
            match fep_command {
                MotorCommandType::ExplorationTrigger => {
                    if self.phase != TaskPhase::Understanding {
                        tracing::info!(
                            target: "symthaea::coding_agent",
                            from = %self.phase,
                            "FEP ExplorationTrigger → Understanding"
                        );
                        self.phase = TaskPhase::Understanding;
                        self.phase_failures = 0;
                        return;
                    }
                }
                MotorCommandType::ReflectionInitiate => {
                    if self.phase != TaskPhase::Planning && self.phase != TaskPhase::Understanding {
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
                    if self.phase == TaskPhase::Generating || self.phase == TaskPhase::Fixing {
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
                if let Some(ref result) = motor_result {
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
                    // No motor result during testing — if we have generated code, consider done
                    if self.generated_code.is_some() {
                        self.phase = TaskPhase::Done;
                        tracing::info!(target: "symthaea::coding_agent", "→ Done (code written, no test runner)");
                    } else {
                        self.phase_failures += 1;
                        if self.phase_failures >= self.config.max_phase_failures {
                            self.phase = TaskPhase::Done;
                        }
                    }
                }
            }
            TaskPhase::Fixing => {
                let code_written = self.generated_code.is_some();
                if let Some(ref result) = motor_result {
                    if result.success || code_written {
                        self.phase = TaskPhase::Testing;
                        self.phase_failures = 0;
                        tracing::info!(target: "symthaea::coding_agent", "→ Testing (after fix)");
                    } else {
                        self.phase_failures += 1;
                        if self.phase_failures >= self.config.max_phase_failures {
                            self.phase = TaskPhase::Done;
                            tracing::warn!(
                                target: "symthaea::coding_agent",
                                "Fix failed {} times, giving up",
                                self.config.max_phase_failures
                            );
                        }
                    }
                } else if code_written {
                    self.phase = TaskPhase::Testing;
                    self.phase_failures = 0;
                } else {
                    self.phase_failures += 1;
                    if self.phase_failures >= self.config.max_phase_failures {
                        self.phase = TaskPhase::Done;
                    }
                }
            }
            TaskPhase::Done => {} // terminal
        }

        // Stuck detection: if Phi stays low for 3+ cycles and we're past the
        // initial phases, consciousness isn't engaging — try a different approach.
        // Only triggers after Generating has been attempted (not during initial ramp-up).
        if self.phi_trace.len() >= 3
            && self.phase != TaskPhase::Done
            && self.phase != TaskPhase::Understanding
            && self.phase != TaskPhase::Planning
            && !self.generation_tiers.is_empty()
        // only after at least one generation
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
                if let Some(entry) = self.failure_patterns.iter_mut().find(|(p, _)| *p == pattern) {
                    entry.1 += 1;
                } else {
                    self.failure_patterns.push((pattern.clone(), 1));
                }

                // Store failure in persistent experience store
                self.store_experience(error, false);
            }
        }
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
            tier: self.generation_tiers.last().map(|t| t.to_string()).unwrap_or_default(),
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
            if let Some(ref code) = self.generated_code {
                let summary: String = code.chars().take(200).collect();
                self.store_experience(&summary, true);
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

        // If we have a task, query for relevant context and inject it
        if !self.task.is_empty() {
            let encoder = memory.encoder();
            let intent_hv = encoder.encode_name(&self.task);
            let matches = memory.query(&intent_hv, 5);
            let context: Vec<String> = matches
                .iter()
                .filter(|m| m.similarity > 0.2)
                .map(|m| {
                    format!(
                        "{:?} `{}` in {} (sim: {:.3})",
                        m.kind, m.name, m.path.display(), m.similarity
                    )
                })
                .collect();
            if !context.is_empty() {
                self.code_context = context;
            }
        }

        // Store memory for future queries
        self.code_memory = Some(memory);

        Ok((files_indexed, stats.functions, stats.types))
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
}
