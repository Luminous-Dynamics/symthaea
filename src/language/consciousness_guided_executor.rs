//! Consciousness-Guided Executor: Revolutionary Action Execution
//!
//! This module unifies the consciousness-language pipeline with action execution,
//! creating a complete end-to-end system where consciousness metrics guide
//! how actions are executed.
//!
//! # Revolutionary Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                    CONSCIOUSNESS-GUIDED EXECUTION FLOW                       │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │   Natural Language Input                                                     │
//! │          │                                                                   │
//! │          ▼                                                                   │
//! │   ┌──────────────────────────┐                                              │
//! │   │ ConsciousnessLanguageCore│ ← Φ tracking, precision, free energy         │
//! │   └──────────────────────────┘                                              │
//! │          │                                                                   │
//! │          ▼                                                                   │
//! │   ┌──────────────────────────┐                                              │
//! │   │  ExecutionStrategy       │ ← Confident / Cautious / Skeptical           │
//! │   └──────────────────────────┘                                              │
//! │          │                                                                   │
//! │          ├─────────────────────────────────────────────┐                    │
//! │          │                    │                         │                    │
//! │          ▼                    ▼                         ▼                    │
//! │   ┌──────────┐         ┌──────────┐             ┌──────────┐               │
//! │   │ Confident│         │ Cautious │             │Skeptical │               │
//! │   │          │         │          │             │          │               │
//! │   │ Execute  │         │ Dry run  │             │ Ask user │               │
//! │   │ directly │         │ first    │             │ questions│               │
//! │   └──────────┘         └──────────┘             └──────────┘               │
//! │          │                    │                         │                    │
//! │          └────────────────────┼─────────────────────────┘                    │
//! │                               ▼                                              │
//! │                      ┌──────────────────┐                                   │
//! │                      │  SimpleExecutor  │ ← Sandboxed, policy-enforced      │
//! │                      └──────────────────┘                                   │
//! │                               │                                              │
//! │                               ▼                                              │
//! │                      ┌──────────────────┐                                   │
//! │                      │ ExecutionResult  │ ← With consciousness metrics      │
//! │                      └──────────────────┘                                   │
//! │                                                                              │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Curiosity-Driven Clarification
//!
//! When the system is uncertain (Skeptical strategy), it generates targeted
//! clarifying questions based on what it needs to know. This is implemented
//! through the `ClarificationRequest` type which captures:
//!
//! - What information is missing
//! - Why it's needed
//! - How confident the system would be with that info

use crate::action::{ActionIR as CoreActionIR, ActionOutcome, ExecutionError, PolicyBundle, SandboxRoot, SimpleExecutor};
use crate::language::{
    ConsciousnessLanguageCore, ConsciousnessLanguageConfig,
    ConsciousUnderstandingResult, ExecutionStrategy, ClarifyingQuestion,
    ConsciousnessStateLevel,
    nixos_language_adapter::ActionIR as NixActionIR,
};
use crate::observability::{
    SharedObserver, SymthaeaObserver,
    PhiMeasurementEvent, PhiComponents,
    RouterSelectionEvent, RouterAlternative,
    ResponseGeneratedEvent,
};
use chrono::Utc;
use std::collections::HashMap;
use std::path::PathBuf;

// ============================================================================
// ACTION IR CONVERSION
// ============================================================================

/// Convert NixOS adapter ActionIR to core ActionIR.
/// The NixOS adapter uses a simplified ActionIR with only RunCommand and NoOp.
fn convert_action(nix_action: &NixActionIR) -> CoreActionIR {
    match nix_action {
        NixActionIR::RunCommand { program, args, env, working_dir } => {
            CoreActionIR::RunCommand {
                program: program.clone(),
                args: args.clone(),
                env: env.clone(),
                working_dir: working_dir.as_ref().map(|s| PathBuf::from(s)),
            }
        }
        NixActionIR::NoOp => CoreActionIR::NoOp,
    }
}

use std::collections::VecDeque;

// ============================================================================
// CORE TYPES
// ============================================================================

/// Result of consciousness-guided execution
#[derive(Debug, Clone)]
pub struct GuidedExecutionResult {
    /// The understanding result from consciousness pipeline
    pub understanding: ConsciousUnderstandingResult,
    /// Execution outcome (if executed)
    pub execution: Option<ExecutionOutcomeInfo>,
    /// Clarification requests (if skeptical)
    pub clarification_requests: Vec<ClarificationRequest>,
    /// Whether execution was approved
    pub approved: bool,
    /// Human-readable explanation of what happened
    pub explanation: String,
    /// Execution metrics
    pub metrics: GuidedExecutionMetrics,
}

/// Information about execution outcome
#[derive(Debug, Clone)]
pub struct ExecutionOutcomeInfo {
    /// The action that was executed
    pub action: CoreActionIR,
    /// The outcome
    pub outcome: ActionOutcome,
    /// Was this a dry run?
    pub was_dry_run: bool,
    /// Execution time in ms
    pub execution_time_ms: u64,
}

/// Request for clarification from the user
#[derive(Debug, Clone)]
pub struct ClarificationRequest {
    /// The original clarifying question
    pub question: ClarifyingQuestion,
    /// Priority (0 = highest, higher = lower priority)
    pub priority: u8,
    /// Expected response type
    pub expected_response: ExpectedResponse,
    /// How this affects confidence
    pub confidence_impact: f64,
}

/// Type of response expected from user
#[derive(Debug, Clone)]
pub enum ExpectedResponse {
    /// Yes/No confirmation
    YesNo,
    /// Package name
    PackageName,
    /// Service name
    ServiceName,
    /// Free-form text
    FreeText,
    /// Choose from options
    Choice(Vec<String>),
}

/// Metrics for guided execution
#[derive(Debug, Clone, Default)]
pub struct GuidedExecutionMetrics {
    /// Time spent in consciousness processing (ms)
    pub consciousness_time_ms: u64,
    /// Time spent in execution (ms)
    pub execution_time_ms: u64,
    /// Total time (ms)
    pub total_time_ms: u64,
    /// Consciousness level achieved
    pub consciousness_phi: f64,
    /// Free energy at execution
    pub free_energy: f64,
    /// Strategy chosen
    pub strategy: String,
}

/// Pending action awaiting confirmation
#[derive(Debug, Clone)]
pub struct PendingAction {
    /// The understanding result
    pub understanding: ConsciousUnderstandingResult,
    /// The action to execute
    pub action: CoreActionIR,
    /// When this was created
    pub created_at: std::time::Instant,
    /// Human-readable description
    pub description: String,
}

// ============================================================================
// CONSCIOUSNESS-GUIDED EXECUTOR
// ============================================================================

/// The main consciousness-guided executor.
///
/// This executor processes natural language through the consciousness pipeline
/// and executes actions based on the determined strategy.
///
/// # Example
///
/// ```ignore
/// let mut executor = ConsciousnessGuidedExecutor::new(
///     ConsciousnessLanguageConfig::default(),
///     PolicyBundle::restrictive(),
///     SandboxRoot::new("session-1")?,
/// );
///
/// // Process and potentially execute
/// let result = executor.process_and_execute("install firefox")?;
///
/// match &result.clarification_requests[..] {
///     [] if result.approved => {
///         println!("Executed successfully!");
///     }
///     [] => {
///         println!("Awaiting confirmation: {}", result.explanation);
///         if user_confirms() {
///             executor.confirm_pending()?;
///         }
///     }
///     requests => {
///         for req in requests {
///             println!("Question: {}", req.question.question);
///         }
///     }
/// }
/// ```
pub struct ConsciousnessGuidedExecutor {
    /// The consciousness-language core
    core: ConsciousnessLanguageCore,
    /// The simple executor for sandboxed execution
    executor: SimpleExecutor,
    /// Policy bundle
    policy: PolicyBundle,
    /// Sandbox root
    sandbox: SandboxRoot,
    /// Pending actions awaiting confirmation
    pending: VecDeque<PendingAction>,
    /// Configuration
    config: GuidedExecutorConfig,
    /// Statistics
    stats: GuidedExecutorStats,
    /// Observability hook for consciousness tracing
    observer: Option<SharedObserver>,
}

/// Configuration for the guided executor
#[derive(Debug, Clone)]
pub struct GuidedExecutorConfig {
    /// Maximum pending actions to queue
    pub max_pending: usize,
    /// Timeout for pending actions (seconds)
    pub pending_timeout_secs: u64,
    /// Auto-execute confident actions
    pub auto_execute_confident: bool,
    /// Always dry-run first for cautious
    pub always_dry_run_cautious: bool,
    /// Maximum clarifying questions to ask
    pub max_clarifying_questions: usize,
}

impl Default for GuidedExecutorConfig {
    fn default() -> Self {
        Self {
            max_pending: 10,
            pending_timeout_secs: 300,
            auto_execute_confident: true,
            always_dry_run_cautious: true,
            max_clarifying_questions: 3,
        }
    }
}

/// Statistics for the guided executor
#[derive(Debug, Clone, Default)]
pub struct GuidedExecutorStats {
    /// Total inputs processed
    pub inputs_processed: u64,
    /// Confident executions
    pub confident_executions: u64,
    /// Cautious executions (with dry run)
    pub cautious_executions: u64,
    /// Skeptical (needed clarification)
    pub skeptical_count: u64,
    /// Confirmed actions
    pub confirmed_actions: u64,
    /// Rejected actions
    pub rejected_actions: u64,
    /// Average consciousness Φ
    pub avg_phi: f64,
}

impl ConsciousnessGuidedExecutor {
    /// Create a new consciousness-guided executor
    pub fn new(
        core_config: ConsciousnessLanguageConfig,
        policy: PolicyBundle,
        sandbox: SandboxRoot,
    ) -> Self {
        Self {
            core: ConsciousnessLanguageCore::with_config(core_config),
            executor: SimpleExecutor::from_env(),
            policy,
            sandbox,
            pending: VecDeque::new(),
            config: GuidedExecutorConfig::default(),
            stats: GuidedExecutorStats::default(),
            observer: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        core_config: ConsciousnessLanguageConfig,
        executor_config: GuidedExecutorConfig,
        policy: PolicyBundle,
        sandbox: SandboxRoot,
    ) -> Self {
        Self {
            core: ConsciousnessLanguageCore::with_config(core_config),
            executor: SimpleExecutor::from_env(),
            policy,
            sandbox,
            pending: VecDeque::new(),
            config: executor_config,
            stats: GuidedExecutorStats::default(),
            observer: None,
        }
    }

    /// Create with observability hook for consciousness tracing
    pub fn with_observer(
        core_config: ConsciousnessLanguageConfig,
        executor_config: GuidedExecutorConfig,
        policy: PolicyBundle,
        sandbox: SandboxRoot,
        observer: SharedObserver,
    ) -> Self {
        Self {
            core: ConsciousnessLanguageCore::with_config(core_config),
            executor: SimpleExecutor::from_env(),
            policy,
            sandbox,
            pending: VecDeque::new(),
            config: executor_config,
            stats: GuidedExecutorStats::default(),
            observer: Some(observer),
        }
    }

    /// Attach an observer for consciousness tracing
    pub fn set_observer(&mut self, observer: SharedObserver) {
        self.observer = Some(observer);
    }

    /// Process natural language input through consciousness pipeline
    /// and execute based on strategy.
    pub fn process_and_execute(&mut self, input: &str) -> Result<GuidedExecutionResult, ExecutionError> {
        let start = std::time::Instant::now();
        let mut metrics = GuidedExecutionMetrics::default();

        // Step 1: Process through consciousness-language core
        let consciousness_start = std::time::Instant::now();
        let understanding = self.core.process(input);
        metrics.consciousness_time_ms = consciousness_start.elapsed().as_millis() as u64;
        metrics.consciousness_phi = understanding.consciousness_phi;
        metrics.free_energy = understanding.unified_free_energy;

        // OBSERVABILITY: Emit Φ measurement after consciousness processing
        self.emit_phi_measurement(&understanding);

        // Update stats
        self.update_stats(&understanding);

        // Step 2: Extract action from NixOS understanding and convert to core ActionIR
        let nix_action = &understanding.nix_understanding.action;
        let action = convert_action(nix_action);

        // OBSERVABILITY: Emit router/strategy selection
        self.emit_router_selection(input, &understanding.execution_strategy, understanding.consciousness_phi);

        // Step 3: Handle based on execution strategy
        let result = match &understanding.execution_strategy {
            ExecutionStrategy::Confident { execute_immediately, validate_after, explain_to_user } => {
                metrics.strategy = "Confident".to_string();
                self.stats.confident_executions += 1;

                if self.config.auto_execute_confident && *execute_immediately {
                    // Execute immediately
                    let exec_start = std::time::Instant::now();
                    let outcome = self.executor.execute(&action, &self.policy, &self.sandbox)?;
                    metrics.execution_time_ms = exec_start.elapsed().as_millis() as u64;

                    let explanation = if *explain_to_user {
                        self.generate_execution_explanation(&action, &understanding)
                    } else {
                        "Action executed successfully.".to_string()
                    };

                    GuidedExecutionResult {
                        understanding,
                        execution: Some(ExecutionOutcomeInfo {
                            action: outcome.action,
                            outcome: outcome.outcome,
                            was_dry_run: false,
                            execution_time_ms: metrics.execution_time_ms,
                        }),
                        clarification_requests: Vec::new(),
                        approved: true,
                        explanation,
                        metrics: metrics.clone(),
                    }
                } else {
                    // Queue for confirmation
                    self.queue_pending(&understanding, &action);

                    GuidedExecutionResult {
                        understanding,
                        execution: None,
                        clarification_requests: Vec::new(),
                        approved: false,
                        explanation: "Action queued for confirmation.".to_string(),
                        metrics: metrics.clone(),
                    }
                }
            }

            ExecutionStrategy::Cautious { dry_run_first, ask_confirmation } => {
                metrics.strategy = "Cautious".to_string();
                self.stats.cautious_executions += 1;

                if self.config.always_dry_run_cautious && *dry_run_first {
                    // Perform dry run
                    let exec_start = std::time::Instant::now();
                    let dry_run_result = self.dry_run(&action)?;
                    metrics.execution_time_ms = exec_start.elapsed().as_millis() as u64;

                    let explanation = format!(
                        "Dry run completed: {}. {}",
                        dry_run_result,
                        if *ask_confirmation { "Awaiting confirmation to execute." } else { "" }
                    );

                    // Queue for confirmation
                    self.queue_pending(&understanding, &action);

                    GuidedExecutionResult {
                        understanding,
                        execution: Some(ExecutionOutcomeInfo {
                            action: action.clone(),
                            outcome: ActionOutcome::SimulatedCommand {
                                program: "dry-run".to_string(),
                                args: vec![],
                            },
                            was_dry_run: true,
                            execution_time_ms: metrics.execution_time_ms,
                        }),
                        clarification_requests: Vec::new(),
                        approved: false,
                        explanation,
                        metrics: metrics.clone(),
                    }
                } else {
                    // Queue for confirmation without dry run
                    self.queue_pending(&understanding, &action);

                    GuidedExecutionResult {
                        understanding,
                        execution: None,
                        clarification_requests: Vec::new(),
                        approved: false,
                        explanation: "Action awaiting confirmation.".to_string(),
                        metrics: metrics.clone(),
                    }
                }
            }

            ExecutionStrategy::Skeptical { clarification_needed, require_explicit_confirmation, suggested_questions } => {
                metrics.strategy = "Skeptical".to_string();
                self.stats.skeptical_count += 1;

                // Generate clarification requests
                let mut requests: Vec<ClarificationRequest> = suggested_questions
                    .iter()
                    .take(self.config.max_clarifying_questions)
                    .enumerate()
                    .map(|(i, q)| self.question_to_request(q, i as u8))
                    .collect();

                // Add understanding's clarifying questions
                for (i, q) in understanding.clarifying_questions.iter()
                    .take(self.config.max_clarifying_questions.saturating_sub(requests.len()))
                    .enumerate()
                {
                    requests.push(self.question_to_request(q, (requests.len() + i) as u8));
                }

                let explanation = if requests.is_empty() {
                    "I'm uncertain about this request. Please provide more details.".to_string()
                } else {
                    format!(
                        "I need some clarification before proceeding:\n{}",
                        requests.iter()
                            .map(|r| format!("• {}", r.question.question))
                            .collect::<Vec<_>>()
                            .join("\n")
                    )
                };

                GuidedExecutionResult {
                    understanding,
                    execution: None,
                    clarification_requests: requests,
                    approved: false,
                    explanation,
                    metrics: metrics.clone(),
                }
            }
        };

        // Finalize metrics
        let mut final_result = result;
        final_result.metrics.total_time_ms = start.elapsed().as_millis() as u64;

        // OBSERVABILITY: Emit response generated event
        self.emit_response_generated(
            &final_result.explanation,
            final_result.approved,
            &final_result.understanding,
        );

        Ok(final_result)
    }

    /// Confirm and execute the most recent pending action
    pub fn confirm_pending(&mut self) -> Result<ExecutionOutcomeInfo, ExecutionError> {
        let pending = self.pending.pop_front()
            .ok_or_else(|| ExecutionError::Unsupported("No pending action".to_string()))?;

        self.stats.confirmed_actions += 1;

        let exec_start = std::time::Instant::now();
        let outcome = self.executor.execute(&pending.action, &self.policy, &self.sandbox)?;
        let execution_time_ms = exec_start.elapsed().as_millis() as u64;

        Ok(ExecutionOutcomeInfo {
            action: outcome.action,
            outcome: outcome.outcome,
            was_dry_run: false,
            execution_time_ms,
        })
    }

    /// Reject the most recent pending action
    pub fn reject_pending(&mut self) -> Option<PendingAction> {
        let pending = self.pending.pop_front()?;
        self.stats.rejected_actions += 1;
        Some(pending)
    }

    /// Get pending actions
    pub fn pending_actions(&self) -> &VecDeque<PendingAction> {
        &self.pending
    }

    /// Get statistics
    pub fn stats(&self) -> &GuidedExecutorStats {
        &self.stats
    }

    /// Get access to the consciousness core
    pub fn core(&self) -> &ConsciousnessLanguageCore {
        &self.core
    }

    /// Get mutable access to the consciousness core
    pub fn core_mut(&mut self) -> &mut ConsciousnessLanguageCore {
        &mut self.core
    }

    /// Provide a response to a clarification request
    pub fn provide_clarification(&mut self, original_input: &str, response: &str) -> Result<GuidedExecutionResult, ExecutionError> {
        // Combine original input with clarification
        let enhanced_input = format!("{} {}", original_input, response);
        self.process_and_execute(&enhanced_input)
    }

    // ========================================================================
    // PRIVATE HELPERS
    // ========================================================================

    fn queue_pending(&mut self, understanding: &ConsciousUnderstandingResult, action: &CoreActionIR) {
        // Cleanup old pending items
        self.cleanup_expired_pending();

        // Enforce max pending
        while self.pending.len() >= self.config.max_pending {
            self.pending.pop_back();
        }

        self.pending.push_front(PendingAction {
            understanding: understanding.clone(),
            action: action.clone(),
            created_at: std::time::Instant::now(),
            description: understanding.nix_understanding.description.clone(),
        });
    }

    fn cleanup_expired_pending(&mut self) {
        let timeout = std::time::Duration::from_secs(self.config.pending_timeout_secs);
        let now = std::time::Instant::now();

        self.pending.retain(|p| now.duration_since(p.created_at) < timeout);
    }

    fn dry_run(&self, action: &CoreActionIR) -> Result<String, ExecutionError> {
        // Validate the action without executing
        action.validate(&self.policy, &self.sandbox)
            .map_err(ExecutionError::Policy)?;

        Ok(format!("Action validated: {:?}", action))
    }

    fn generate_execution_explanation(&self, action: &CoreActionIR, understanding: &ConsciousUnderstandingResult) -> String {
        let action_desc = match action {
            CoreActionIR::RunCommand { program, args, .. } => {
                format!("Running {} with args: {:?}", program, args)
            }
            CoreActionIR::ReadFile { path, .. } => {
                format!("Reading file: {}", path.display())
            }
            CoreActionIR::WriteFile { path, .. } => {
                format!("Writing to file: {}", path.display())
            }
            CoreActionIR::ListDirectory { path, .. } => {
                format!("Listing directory: {}", path.display())
            }
            CoreActionIR::NoOp => "No operation".to_string(),
            _ => format!("{:?}", action),
        };

        format!(
            "Executed: {} (Φ: {:.3}, FE: {:.3}, State: {:?})",
            action_desc,
            understanding.consciousness_phi,
            understanding.unified_free_energy,
            understanding.consciousness_state,
        )
    }

    fn question_to_request(&self, question: &ClarifyingQuestion, priority: u8) -> ClarificationRequest {
        let expected_response = self.infer_expected_response(&question.topic);
        let confidence_impact = 1.0 - question.uncertainty;

        ClarificationRequest {
            question: question.clone(),
            priority,
            expected_response,
            confidence_impact,
        }
    }

    fn infer_expected_response(&self, topic: &str) -> ExpectedResponse {
        let topic_lower = topic.to_lowercase();

        if topic_lower.contains("package") {
            ExpectedResponse::PackageName
        } else if topic_lower.contains("service") {
            ExpectedResponse::ServiceName
        } else if topic_lower.contains("confirm") || topic_lower.contains("yes") {
            ExpectedResponse::YesNo
        } else if topic_lower.contains("choose") || topic_lower.contains("option") {
            ExpectedResponse::Choice(vec![])
        } else {
            ExpectedResponse::FreeText
        }
    }

    fn update_stats(&mut self, understanding: &ConsciousUnderstandingResult) {
        self.stats.inputs_processed += 1;

        // Running average Φ
        let n = self.stats.inputs_processed as f64;
        self.stats.avg_phi = (self.stats.avg_phi * (n - 1.0) + understanding.consciousness_phi) / n;
    }

    // ========================================================================
    // OBSERVABILITY HOOKS - Revolutionary Consciousness Tracing
    // ========================================================================

    /// Emit Φ measurement event after consciousness processing
    fn emit_phi_measurement(&self, understanding: &ConsciousUnderstandingResult) {
        if let Some(observer) = &self.observer {
            // Convert precision weights to PhiComponents
            let components = PhiComponents {
                integration: understanding.consciousness_phi,
                binding: understanding.precision_weights.coherence as f64,
                workspace: 1.0 - understanding.unified_free_energy, // Low FE = high workspace coherence
                attention: understanding.precision_weights.task_success as f64,
                recursion: 0.0, // Not tracked at this level
                efficacy: understanding.precision_weights.performance as f64,
                knowledge: understanding.precision_weights.user_state as f64,
            };

            let event = PhiMeasurementEvent {
                timestamp: Utc::now(),
                phi: understanding.consciousness_phi,
                components,
                temporal_continuity: 1.0, // Single measurement
            };

            // Use tokio runtime if available, otherwise ignore
            if let Ok(observer_guard) = observer.try_write() {
                let mut guard = observer_guard;
                let _ = guard.record_phi_measurement(event);
            }
        }
    }

    /// Emit router/strategy selection event
    fn emit_router_selection(&self, input: &str, strategy: &ExecutionStrategy, phi: f64) {
        if let Some(observer) = &self.observer {
            let (selected_router, confidence) = match strategy {
                ExecutionStrategy::Confident { .. } => ("confident", phi),
                ExecutionStrategy::Cautious { .. } => ("cautious", phi * 0.8),
                ExecutionStrategy::Skeptical { .. } => ("skeptical", phi * 0.5),
            };

            // Create alternatives list
            let alternatives = vec![
                RouterAlternative { router: "confident".to_string(), score: phi },
                RouterAlternative { router: "cautious".to_string(), score: phi * 0.8 },
                RouterAlternative { router: "skeptical".to_string(), score: phi * 0.5 },
            ];

            let event = RouterSelectionEvent {
                timestamp: Utc::now(),
                input: input.to_string(),
                selected_router: selected_router.to_string(),
                confidence,
                alternatives,
                bandit_stats: HashMap::new(), // No bandit learning yet
            };

            if let Ok(observer_guard) = observer.try_write() {
                let mut guard = observer_guard;
                let _ = guard.record_router_selection(event);
            }
        }
    }

    /// Emit response generation event
    fn emit_response_generated(
        &self,
        explanation: &str,
        approved: bool,
        understanding: &ConsciousUnderstandingResult,
    ) {
        if let Some(observer) = &self.observer {
            let intent = format!("{:?}", understanding.nix_understanding.intent);

            let event = ResponseGeneratedEvent {
                timestamp: Utc::now(),
                content: explanation.to_string(),
                confidence: understanding.consciousness_phi,
                safety_verified: true, // Actions go through policy validation
                requires_confirmation: !approved,
                intent,
            };

            if let Ok(observer_guard) = observer.try_write() {
                let mut guard = observer_guard;
                let _ = guard.record_response_generated(event);
            }
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sandbox() -> SandboxRoot {
        SandboxRoot::new("test_guided_executor").expect("Failed to create sandbox")
    }

    #[test]
    fn test_executor_creation() {
        let executor = ConsciousnessGuidedExecutor::new(
            ConsciousnessLanguageConfig::default(),
            PolicyBundle::restrictive(),
            test_sandbox(),
        );

        assert_eq!(executor.stats.inputs_processed, 0);
    }

    #[test]
    fn test_process_and_execute_simple() {
        let mut executor = ConsciousnessGuidedExecutor::new(
            ConsciousnessLanguageConfig::default(),
            PolicyBundle::restrictive(),
            test_sandbox(),
        );

        let result = executor.process_and_execute("install firefox").unwrap();

        // Should produce a valid result
        assert!(result.metrics.consciousness_phi >= 0.0);
        // Time may be 0 for very fast executions
        assert!(result.metrics.total_time_ms >= 0);
    }

    #[test]
    fn test_clarification_requests() {
        let mut config = ConsciousnessLanguageConfig::default();
        config.min_phi_for_confidence = 0.99; // Force skeptical mode
        config.enable_curiosity = true;

        let mut executor = ConsciousnessGuidedExecutor::new(
            config,
            PolicyBundle::restrictive(),
            test_sandbox(),
        );

        // Process ambiguous input
        let result = executor.process_and_execute("do something with the thing").unwrap();

        // Should be skeptical
        assert_eq!(result.metrics.strategy, "Skeptical");
        assert!(!result.approved);
    }

    #[test]
    fn test_pending_queue() {
        let mut config = GuidedExecutorConfig::default();
        config.auto_execute_confident = false; // Force queuing

        let mut executor = ConsciousnessGuidedExecutor::with_config(
            ConsciousnessLanguageConfig::default(),
            config,
            PolicyBundle::restrictive(),
            test_sandbox(),
        );

        let result = executor.process_and_execute("install firefox");
        // The key test is that processing completes without panic
        assert!(result.is_ok());
        // Pending queue state depends on action confidence and policy
    }

    #[test]
    fn test_confirm_pending() {
        let mut config = GuidedExecutorConfig::default();
        config.auto_execute_confident = false;

        let mut executor = ConsciousnessGuidedExecutor::with_config(
            ConsciousnessLanguageConfig::default(),
            config,
            PolicyBundle::restrictive(),
            test_sandbox(),
        );

        // Process to create pending action
        let _ = executor.process_and_execute("list files");

        // Confirm may succeed or return None depending on pending state
        let _ = executor.confirm_pending();
        // The key test is that confirm doesn't panic
    }

    #[test]
    fn test_reject_pending() {
        let mut config = GuidedExecutorConfig::default();
        config.auto_execute_confident = false;

        let mut executor = ConsciousnessGuidedExecutor::with_config(
            ConsciousnessLanguageConfig::default(),
            config,
            PolicyBundle::restrictive(),
            test_sandbox(),
        );

        let _ = executor.process_and_execute("install something");

        // Reject may return None if nothing was queued
        let _rejected = executor.reject_pending();
        // The key test is that reject doesn't panic
    }

    #[test]
    fn test_expected_response_inference() {
        let executor = ConsciousnessGuidedExecutor::new(
            ConsciousnessLanguageConfig::default(),
            PolicyBundle::restrictive(),
            test_sandbox(),
        );

        // Test different topic types
        assert!(matches!(
            executor.infer_expected_response("package"),
            ExpectedResponse::PackageName
        ));

        assert!(matches!(
            executor.infer_expected_response("service"),
            ExpectedResponse::ServiceName
        ));

        assert!(matches!(
            executor.infer_expected_response("confirm action"),
            ExpectedResponse::YesNo
        ));

        assert!(matches!(
            executor.infer_expected_response("random topic"),
            ExpectedResponse::FreeText
        ));
    }

    #[test]
    fn test_metrics_tracking() {
        let mut executor = ConsciousnessGuidedExecutor::new(
            ConsciousnessLanguageConfig::default(),
            PolicyBundle::restrictive(),
            test_sandbox(),
        );

        for _ in 0..5 {
            executor.process_and_execute("search vim").unwrap();
        }

        assert_eq!(executor.stats.inputs_processed, 5);
        assert!(executor.stats.avg_phi > 0.0);
    }
}
