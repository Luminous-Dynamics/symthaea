use crate::language::intelligent_dispatcher::BackendTier;
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

/// Parsed test failure with structured fields for targeted fixing.
#[derive(Debug, Clone)]
pub(crate) struct StructuredTestFailure {
    pub(crate) test_name: String,
    pub(crate) failure_kind: TestFailureKind,
    pub(crate) expected: Option<String>,
    pub(crate) actual: Option<String>,
    pub(crate) message: Option<String>,
    pub(crate) file: Option<String>,
    pub(crate) line: Option<usize>,
}

/// Classification of test failure types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TestFailureKind {
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
pub(crate) struct RetryState {
    pub(crate) strategies_tried: Vec<RetryStrategy>,
    pub(crate) current_strategy: RetryStrategy,
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
pub(crate) struct ConsciousnessSignals {
    pub(crate) prediction_error: f32,
    pub(crate) confidence_velocity: f32,
    pub(crate) phi: f32,
    pub(crate) phi_slope: f32,
    pub(crate) fep_surprise: f64,
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
        }
    }
}
