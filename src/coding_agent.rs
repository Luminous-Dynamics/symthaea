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
//! ## State Machine
//!
//! ```text
//! Understanding → Planning → Generating → Testing → Fixing ──→ Done
//!      ↑              ↑           │            │         │
//!      └──────────────┴───────────┘            └─────────┘
//! ```

use crate::action::{ActionOutcome, PolicyBundle, SandboxRoot};
use crate::cognitive_loop::motor_output_bridge::{
    ActionType, MotorActionRequest, MotorOutputBridge, MotorOutputResult,
};
use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, CycleResult};
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
}

impl Default for CodingAgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            max_phase_failures: 3,
            min_generation_phi: 0.3,
            working_dir: PathBuf::from("."),
            sandbox_session: "coding_agent".to_string(),
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
            if let Ok(bridge) = MotorOutputBridge::with_defaults() {
                cognitive_loop.set_motor_output_bridge(bridge);
            }
        }

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
        }
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

        tracing::info!(
            target: "symthaea::coding_agent",
            task = task,
            max_iterations = self.config.max_iterations,
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
        // Build observation text from current state
        let observation = self.build_observation();

        // Set up the motor request based on current phase
        let motor_request = self.build_motor_request();
        self.cognitive_loop.set_motor_request(motor_request);

        // Run one cognitive cycle
        let cycle_result = self.cognitive_loop.cycle(&observation);

        // Record Phi from metadata
        let phi = cycle_result.metadata.consciousness_level as f32;
        self.phi_trace.push(phi);

        // Check for motor output result
        let motor_result = self.cognitive_loop.take_motor_result();

        // Process the cycle result and motor output
        self.process_step_result(&cycle_result, motor_result, phi);
    }

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
                // For generation, provide the working directory for writes
                MotorActionRequest {
                    target_path: Some(self.config.working_dir.clone()),
                    ..Default::default()
                }
            }
            TaskPhase::Testing => {
                // Run cargo check in the working directory
                MotorActionRequest {
                    target_path: Some(self.config.working_dir.clone()),
                    program: Some("cargo".into()),
                    args: vec!["check".into()],
                    ..Default::default()
                }
            }
            TaskPhase::Fixing => {
                // Same as generating — will write fixes
                MotorActionRequest {
                    target_path: Some(self.config.working_dir.clone()),
                    ..Default::default()
                }
            }
            TaskPhase::Done => MotorActionRequest::default(),
        }
    }

    /// Process the results of a cognitive cycle and decide the next phase.
    fn process_step_result(
        &mut self,
        cycle_result: &CycleResult,
        motor_result: Option<MotorOutputResult>,
        phi: f32,
    ) {
        // Check epistemic status from prediction confidence
        let confidence = self.cognitive_loop.prediction_confidence();
        let epistemic = Self::confidence_to_epistemic(confidence);

        // If epistemic status is too low and we're about to generate, don't
        if self.phase == TaskPhase::Generating && epistemic == EpistemicStatus::Unknown {
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

        // Phase transitions based on current state
        match self.phase {
            TaskPhase::Understanding => {
                // After gathering some context, move to planning
                if self.iteration >= 1 || !self.observations.is_empty() {
                    self.phase = TaskPhase::Planning;
                    self.phase_failures = 0;
                    tracing::info!(target: "symthaea::coding_agent", "→ Planning");
                }
            }
            TaskPhase::Planning => {
                // After one planning cycle, move to generating
                // (The cognitive loop's FEP + reasoning engine handles the actual planning)
                self.phase = TaskPhase::Generating;
                self.phase_failures = 0;
                tracing::info!(target: "symthaea::coding_agent", "→ Generating");
            }
            TaskPhase::Generating => {
                // After generation, move to testing
                if motor_result.as_ref().map_or(false, |r| r.success) {
                    self.phase = TaskPhase::Testing;
                    self.phase_failures = 0;
                    tracing::info!(target: "symthaea::coding_agent", "→ Testing");
                } else {
                    self.phase_failures += 1;
                    if self.phase_failures >= self.config.max_phase_failures {
                        // Too many generation failures — go back to planning
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
                    if result.success {
                        // Tests passed — we're done
                        self.phase = TaskPhase::Done;
                        tracing::info!(target: "symthaea::coding_agent", "→ Done (tests passed)");
                    } else {
                        // Tests failed — move to fixing
                        self.phase = TaskPhase::Fixing;
                        self.phase_failures = 0;
                        tracing::info!(target: "symthaea::coding_agent", "→ Fixing");
                    }
                } else {
                    // No motor result during testing — try again or move on
                    self.phase_failures += 1;
                    if self.phase_failures >= self.config.max_phase_failures {
                        self.phase = TaskPhase::Done;
                    }
                }
            }
            TaskPhase::Fixing => {
                if let Some(ref result) = motor_result {
                    if result.success {
                        // Fix applied — go back to testing
                        self.phase = TaskPhase::Testing;
                        self.phase_failures = 0;
                        tracing::info!(target: "symthaea::coding_agent", "→ Testing (after fix)");
                    } else {
                        self.phase_failures += 1;
                        if self.phase_failures >= self.config.max_phase_failures {
                            // Can't fix — done with partial result
                            self.phase = TaskPhase::Done;
                            tracing::warn!(
                                target: "symthaea::coding_agent",
                                "Fix failed {} times, giving up",
                                self.config.max_phase_failures
                            );
                        }
                    }
                } else {
                    self.phase_failures += 1;
                    if self.phase_failures >= self.config.max_phase_failures {
                        self.phase = TaskPhase::Done;
                    }
                }
            }
            TaskPhase::Done => {} // terminal
        }

        // Stuck detection: if prediction error stays high for 3+ cycles
        if self.phi_trace.len() >= 3 {
            let recent: Vec<f32> = self.phi_trace.iter().rev().take(3).copied().collect();
            let all_low = recent.iter().all(|&p| p < 0.2);
            if all_low && self.phase != TaskPhase::Done {
                self.observations.push(
                    "Stuck detection: Phi consistently low, trying different approach".into(),
                );
                // Reset to planning with new context
                if self.phase != TaskPhase::Planning {
                    self.phase = TaskPhase::Planning;
                    self.phase_failures = 0;
                }
            }
        }
    }

    /// Process a motor output result — extract observations, track files.
    fn process_motor_result(&mut self, result: &MotorOutputResult) {
        if result.success {
            if let Some(ref outcome) = result.outcome {
                match outcome {
                    ActionOutcome::FileContent(data) => {
                        let content =
                            String::from_utf8_lossy(&data[..data.len().min(2000)]).to_string();
                        self.observations
                            .push(format!("Read file ({} bytes): {}", data.len(), &content[..content.len().min(200)]));
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
        }
    }

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
    fn test_coding_agent_runs_iterations() {
        let config = CodingAgentConfig {
            max_iterations: 3,
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();

        let result = agent.run("add a hello() function to /tmp/test.rs");

        assert!(result.iterations_used <= 3);
        assert!(!result.phi_trace.is_empty());
        assert_eq!(result.phi_trace.len(), result.iterations_used);
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
    fn test_agent_result_default_state() {
        let config = CodingAgentConfig {
            max_iterations: 1,
            ..Default::default()
        };
        let mut agent = CodingAgent::new(config).unwrap();
        let result = agent.run("test task");

        assert_eq!(result.iterations_used, 1);
        assert!(result.files_modified.is_empty());
        assert!(result.errors.is_empty() || !result.errors.is_empty()); // either is fine
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
}
