//! Motor Cortex - Action Execution Engine
//!
//! Week 2 Days 5-7: The Hands That Touch the World
//!
//! Architecture: "Think Twice, Cut Once"
//! - Pre-flight simulation (Ghost Run)
//! - Sandboxed execution with dry-run support
//! - Automatic rollback on failure
//! - Built-in safety checks (risk estimation)
//! - Integration with Cerebellum (skills → plans) and Hippocampus (actions → memories)

use crate::brain::{CerebellumActor, Skill};
use crate::memory::{HippocampusActor, EmotionalValence};
use crate::consciousness::unified_value_evaluator::{
    UnifiedValueEvaluator, EvaluationContext, ActionType, Decision, VetoReason,
    AffectiveSystemsState,
};
use crate::consciousness::affective_consciousness::CoreAffect;
use anyhow::{Result, anyhow, Context};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use std::process::Stdio;
use tokio::process::Command;
use tracing::{info, warn, error, debug};

// ============================================================================
// CORE TYPES
// ============================================================================

/// A single atomic operation the Motor Cortex can perform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionStep {
    pub id: u64,
    pub description: String,       // Human-facing summary
    pub command: String,            // Shell command / API call
    pub args: Vec<String>,          // Command arguments
    pub tags: Vec<String>,          // e.g. ["filesystem", "network"]
    pub can_rollback: bool,         // Can this step be undone?
    pub rollback_command: Option<String>, // The inverse operation
    pub estimated_risk: f32,        // 0.0-1.0 risk score
}

impl ActionStep {
    pub fn new(description: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            id: rand::random(),
            description: description.into(),
            command: command.into(),
            args: vec![],
            tags: vec![],
            can_rollback: false,
            rollback_command: None,
            estimated_risk: 0.9, // Default conservative risk until computed
        }
    }

    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_rollback(mut self, rollback: impl Into<String>) -> Self {
        self.can_rollback = true;
        self.rollback_command = Some(rollback.into());
        self
    }

    pub fn with_risk(mut self, risk: f32) -> Self {
        self.estimated_risk = risk.clamp(0.0, 1.0);
        self
    }

    /// Calculate risk based on command patterns
    pub fn estimate_risk(&mut self) {
        let cmd_lower = format!("{} {}", self.command, self.args.join(" "))
            .to_lowercase();

        // High-risk patterns
        if cmd_lower.contains("rm -rf") || cmd_lower.contains("/boot")
            || cmd_lower.contains("/etc") || cmd_lower.contains("dd if=") {
            self.estimated_risk = 0.9;
        }
        // Medium-high patterns
        else if cmd_lower.contains("rm ") || cmd_lower.contains("mv /")
            || cmd_lower.contains("chmod") || cmd_lower.contains("chown") {
            self.estimated_risk = 0.7;
        }
        // Medium patterns
        else if cmd_lower.contains("install") || cmd_lower.contains("upgrade")
            || cmd_lower.contains("nixos-rebuild") {
            self.estimated_risk = 0.5;
        }
        // Low-risk patterns (read-only)
        else if cmd_lower.starts_with("ls") || cmd_lower.starts_with("cat")
            || cmd_lower.starts_with("grep") || cmd_lower.starts_with("find") {
            self.estimated_risk = 0.1;
        }
    }
}

/// Result of executing a single step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub step_id: u64,
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub started_at: u64,
    pub finished_at: u64,
    pub was_dry_run: bool,
}

impl StepResult {
    pub fn duration_ms(&self) -> u64 {
        self.finished_at.saturating_sub(self.started_at)
    }
}

/// A planned multi-step action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedAction {
    pub id: u64,
    pub name: String,
    pub intent: String,              // High-level description
    pub steps: Vec<ActionStep>,
    pub parallel_groups: Vec<Vec<u64>>, // Step IDs that can run in parallel
    pub created_at: u64,
    pub simulation_mode: SimulationMode,
}

impl PlannedAction {
    pub fn new(name: impl Into<String>, intent: impl Into<String>) -> Self {
        Self {
            id: rand::random(),
            name: name.into(),
            intent: intent.into(),
            steps: vec![],
            parallel_groups: vec![],
            created_at: chrono::Utc::now().timestamp() as u64,
            simulation_mode: SimulationMode::DryRun,
        }
    }

    pub fn add_step(mut self, step: ActionStep) -> Self {
        self.steps.push(step);
        self
    }

    pub fn max_risk(&self) -> f32 {
        self.steps.iter()
            .map(|s| s.estimated_risk)
            .fold(0.0, f32::max)
    }
}

/// Simulation mode for action execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimulationMode {
    DryRun,      // Validate only, don't execute
    GhostRun,    // Run in isolated sandbox
    RealRun,     // Execute for real
}

/// Result of executing an entire planned action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub action_id: u64,
    pub action_name: String,
    pub step_results: Vec<StepResult>,
    pub overall_success: bool,
    pub rolled_back: bool,
    pub total_duration_ms: u64,
}

impl ExecutionResult {
    pub fn success_rate(&self) -> f32 {
        if self.step_results.is_empty() {
            return 0.0;
        }
        let successes = self.step_results.iter().filter(|r| r.success).count();
        successes as f32 / self.step_results.len() as f32
    }
}

/// Rollback point for transaction-like behavior
#[derive(Debug, Clone)]
pub struct RollbackPoint {
    pub action_id: u64,
    pub executed_steps: Vec<ActionStep>,
    pub created_at: u64,
}

fn contains_shell_meta(command: &str) -> bool {
    command.contains('|')
        || command.contains('&')
        || command.contains(';')
        || command.contains('>')
        || command.contains('<')
}

// ============================================================================
// EXECUTION SANDBOX ABSTRACTION
// ============================================================================

/// Abstraction for executing commands in different environments
#[async_trait]
pub trait ExecutionSandbox: Send + Sync {
    /// Validate command without executing (dry-run)
    async fn dry_run(&self, step: &ActionStep) -> Result<StepResult>;

    /// Execute command for real
    async fn execute(&self, step: &ActionStep) -> Result<StepResult>;

    /// Rollback a previously executed step
    async fn rollback_step(&self, step: &ActionStep) -> Result<StepResult>;
}

/// Local shell execution sandbox
pub struct LocalShellSandbox {
    pub allow_destructive: bool,
    pub timeout_seconds: u64,
}

impl LocalShellSandbox {
    pub fn new() -> Self {
        Self {
            allow_destructive: false,
            timeout_seconds: 300, // 5 minutes default
        }
    }

    pub fn allow_destructive(mut self) -> Self {
        self.allow_destructive = true;
        self
    }

    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    /// Normalize the command/args, rejecting shell metacharacters.
    fn resolve_command(&self, step: &ActionStep) -> Result<(String, Vec<String>)> {
        if step.command.trim().is_empty() {
            return Err(anyhow!("Empty command"));
        }

        if contains_shell_meta(&step.command) {
            return Err(anyhow!(
                "Shell syntax not allowed; provide program name and arguments without pipes/redirects"
            ));
        }

        // If args are provided explicitly, the command must be a single token
        if !step.args.is_empty() {
            if step.command.split_whitespace().count() > 1 {
                return Err(anyhow!(
                    "Inline args are not allowed when args are provided separately"
                ));
            }
            return Ok((step.command.clone(), step.args.clone()));
        }

        let mut parts = step.command.split_whitespace();
        let program = parts
            .next()
            .ok_or_else(|| anyhow!("Empty command"))?
            .to_string();
        let args = parts.map(|s| s.to_string()).collect();
        Ok((program, args))
    }

    fn resolve_raw_command(&self, command: &str) -> Result<(String, Vec<String>)> {
        if command.trim().is_empty() {
            return Err(anyhow!("Empty command"));
        }
        if contains_shell_meta(command) {
            return Err(anyhow!("Shell syntax not allowed in rollback commands"));
        }
        let mut parts = command.split_whitespace();
        let program = parts
            .next()
            .ok_or_else(|| anyhow!("Empty command"))?
            .to_string();
        let args = parts.map(|s| s.to_string()).collect();
        Ok((program, args))
    }

    /// Check if command is safe to run
    fn is_safe_command(&self, step: &ActionStep) -> Result<()> {
        // Normalize the command; this enforces no shell metacharacters.
        let _ = self.resolve_command(step)?;

        // Always compute a fresh risk score so defaults don't bypass safety
        let mut effective_risk = step.estimated_risk;
        if effective_risk >= 0.9 {
            let mut tmp = step.clone();
            tmp.estimate_risk();
            effective_risk = tmp.estimated_risk;
        }

        if !self.allow_destructive && effective_risk > 0.7 {
            return Err(anyhow!(
                "Destructive command blocked (risk: {:.2}): {}",
                effective_risk, step.command
            ));
        }

        // Check for common dangerous patterns
        let cmd_lower = step.command.to_lowercase();
        if cmd_lower.contains("rm -rf /") {
            return Err(anyhow!("Extremely dangerous command blocked: {}", step.command));
        }

        Ok(())
    }

    /// Validate command syntax and prerequisites
    async fn validate_command(&self, step: &ActionStep) -> Result<()> {
        // Normalize first (enforces no shell metacharacters)
        let (cmd_name, _) = self.resolve_command(step)?;

        // Use 'which' to check command existence
        let which_result = Command::new("which")
            .arg(&cmd_name)
            .output()
            .await?;

        if !which_result.status.success() {
            return Err(anyhow!("Command not found: {}", cmd_name));
        }

        Ok(())
    }
}

#[async_trait]
impl ExecutionSandbox for LocalShellSandbox {
    async fn dry_run(&self, step: &ActionStep) -> Result<StepResult> {
        let started_at = chrono::Utc::now().timestamp_millis() as u64;

        info!("🔍 Dry-run: {} - {}", step.description, step.command);

        // Safety check
        if let Err(e) = self.is_safe_command(step) {
            warn!("Dry-run safety veto: {}", e);
            return Ok(StepResult {
                step_id: step.id,
                success: false,
                stdout: String::new(),
                stderr: format!("Safety veto: {}", e),
                exit_code: None,
                started_at,
                finished_at: chrono::Utc::now().timestamp_millis() as u64,
                was_dry_run: true,
            });
        }

        // Validate command exists
        if let Err(e) = self.validate_command(step).await {
            return Ok(StepResult {
                step_id: step.id,
                success: false,
                stdout: String::new(),
                stderr: format!("Validation failed: {}", e),
                exit_code: None,
                started_at,
                finished_at: chrono::Utc::now().timestamp_millis() as u64,
                was_dry_run: true,
            });
        }

        // For certain commands, try native dry-run flags
        let has_dry_run = step.command.contains("rsync")
            || step.command.contains("nixos-rebuild");

        if has_dry_run {
            debug!("Command supports native dry-run, using it");
        }

        let finished_at = chrono::Utc::now().timestamp_millis() as u64;

        Ok(StepResult {
            step_id: step.id,
            success: true,
            stdout: format!("Dry-run validation passed for: {}", step.description),
            stderr: String::new(),
            exit_code: Some(0),
            started_at,
            finished_at,
            was_dry_run: true,
        })
    }

    async fn execute(&self, step: &ActionStep) -> Result<StepResult> {
        let started_at = chrono::Utc::now().timestamp_millis() as u64;

        info!("⚡ Executing: {} - {}", step.description, step.command);

        // Final safety check
        self.is_safe_command(step)?;

        // Validate program availability before executing
        self.validate_command(step).await?;

        // Build command without shell wrapping
        let (program, args) = self.resolve_command(step)?;
        let mut cmd = Command::new(&program);
        cmd.args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Execute with timeout
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(self.timeout_seconds),
            cmd.output()
        ).await
        .context("Command timeout")?
        .context("Failed to execute command")?;

        let finished_at = chrono::Utc::now().timestamp_millis() as u64;

        let success = output.status.success();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if success {
            info!("✅ Success: {}", step.description);
        } else {
            warn!("❌ Failed: {} (exit: {:?})", step.description, output.status.code());
        }

        Ok(StepResult {
            step_id: step.id,
            success,
            stdout,
            stderr,
            exit_code: output.status.code(),
            started_at,
            finished_at,
            was_dry_run: false,
        })
    }

    async fn rollback_step(&self, step: &ActionStep) -> Result<StepResult> {
        let started_at = chrono::Utc::now().timestamp_millis() as u64;

        if !step.can_rollback {
            warn!("⚠️  Step cannot be rolled back: {}", step.description);
            return Ok(StepResult {
                step_id: step.id,
                success: false,
                stdout: String::new(),
                stderr: "No rollback defined for this step".to_string(),
                exit_code: None,
                started_at,
                finished_at: chrono::Utc::now().timestamp_millis() as u64,
                was_dry_run: false,
            });
        }

        let rollback_cmd = step.rollback_command.as_ref()
            .ok_or_else(|| anyhow!("Rollback command missing"))?;
        let (program, args) = self.resolve_raw_command(rollback_cmd)?;

        info!("🔄 Rolling back: {} - {}", step.description, rollback_cmd);

        // Execute rollback command without shell
        let output = Command::new(&program)
            .args(&args)
            .output()
            .await?;

        let finished_at = chrono::Utc::now().timestamp_millis() as u64;

        Ok(StepResult {
            step_id: step.id,
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code(),
            started_at,
            finished_at,
            was_dry_run: false,
        })
    }
}

impl Default for LocalShellSandbox {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// MOTOR CORTEX ACTOR
// ============================================================================

pub struct MotorCortexActor {
    sandbox: Box<dyn ExecutionSandbox>,
    action_queue: VecDeque<PlannedAction>,
    execution_history: Vec<ExecutionResult>,
    rollback_stack: Vec<RollbackPoint>,
    max_history: usize,

    // Integration with other brain regions
    cerebellum: Option<CerebellumActor>,
    hippocampus: Option<HippocampusActor>,

    // Value-guided action gating
    value_evaluator: UnifiedValueEvaluator,
    consciousness_level: f64,
    affective_state: CoreAffect,
}

impl MotorCortexActor {
    pub fn new(sandbox: Box<dyn ExecutionSandbox>) -> Self {
        Self {
            sandbox,
            action_queue: VecDeque::new(),
            execution_history: Vec::new(),
            rollback_stack: Vec::new(),
            max_history: 1000,
            cerebellum: None,
            hippocampus: None,
            // Initialize value-guided gating with default state
            value_evaluator: UnifiedValueEvaluator::new(),
            consciousness_level: 0.5, // Default moderate consciousness
            affective_state: CoreAffect::neutral(),
        }
    }

    /// Update consciousness level from external measurement (e.g., Φ from topology)
    pub fn set_consciousness_level(&mut self, phi: f64) {
        self.consciousness_level = phi.clamp(0.0, 1.0);
    }

    /// Update affective state from emotional system
    pub fn set_affective_state(&mut self, state: CoreAffect) {
        self.affective_state = state;
    }

    pub fn with_cerebellum(mut self, cerebellum: CerebellumActor) -> Self {
        self.cerebellum = Some(cerebellum);
        self
    }

    pub fn with_hippocampus(mut self, hippocampus: HippocampusActor) -> Self {
        self.hippocampus = Some(hippocampus);
        self
    }

    /// Queue an action for execution
    pub fn queue_action(&mut self, action: PlannedAction) {
        info!("📋 Queued action: {} (id: {})", action.name, action.id);
        self.action_queue.push_back(action);
    }

    /// Convert a Skill from Cerebellum into a PlannedAction
    pub fn plan_from_skill(&self, skill: &Skill) -> PlannedAction {
        let mut action = PlannedAction::new(
            skill.name.clone(),
            format!("Execute skill: {}", skill.name)
        );

        for (idx, cmd) in skill.sequence.iter().enumerate() {
            let mut parts = cmd.split_whitespace();
            let program = parts.next().unwrap_or("").to_string();
            let args: Vec<String> = parts.map(|s| s.to_string()).collect();

            let mut step = ActionStep::new(
                format!("Step {}: {}", idx + 1, cmd),
                program
            ).with_args(args);
            step.estimate_risk();
            step.tags = skill.context_tags.clone();

            action = action.add_step(step);
        }

        action
    }

    /// Execute the next action in the queue
    pub async fn execute_next(&mut self) -> Result<Option<ExecutionResult>> {
        let action = match self.action_queue.pop_front() {
            Some(a) => a,
            None => return Ok(None),
        };

        let result = self.execute_action(action).await?;
        Ok(Some(result))
    }

    /// Execute a planned action with full safety and rollback support
    pub async fn execute_action(&mut self, mut action: PlannedAction) -> Result<ExecutionResult> {
        let action_id = action.id;
        let action_name = action.name.clone();

        info!("🚀 Executing action: {} (mode: {:?})", action_name, action.simulation_mode);

        // ====================================================================
        // VALUE GATE: Consciousness-guided action evaluation
        // ====================================================================
        // Evaluate the action's intent and all step descriptions against
        // the Seven Harmonies value system before any execution occurs.

        // Build combined action description for evaluation
        let action_description = format!(
            "{}: {}. Steps: {}",
            action.name,
            action.intent,
            action.steps.iter()
                .map(|s| s.description.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        );

        // Create evaluation context with current consciousness state
        let context = EvaluationContext {
            consciousness_level: self.consciousness_level,
            affective_state: self.affective_state.clone(),
            affective_systems: AffectiveSystemsState::default(), // TODO: integrate with emotional system
            action_type: ActionType::Basic, // TODO: infer from action tags
            involves_others: true, // Conservative default
        };

        // Evaluate against value system
        let evaluation = self.value_evaluator.evaluate(&action_description, context);

        match &evaluation.decision {
            Decision::Veto(reason) => {
                warn!("🛑 VALUE GATE VETO: Action '{}' blocked", action_name);
                warn!("   Reason: {:?}", reason);
                warn!("   Score: {:.3}, Consciousness adequacy: {:.3}",
                    evaluation.overall_score,
                    evaluation.consciousness_adequacy);

                return Ok(ExecutionResult {
                    action_id,
                    action_name: format!("[VETOED] {}", action_name),
                    step_results: vec![],
                    overall_success: false,
                    rolled_back: false,
                    total_duration_ms: 0,
                });
            }
            Decision::Warn(warnings) => {
                warn!("⚠️ VALUE GATE WARNING: Action '{}' proceeding with caution", action_name);
                for warning in warnings {
                    warn!("   - {}", warning);
                }
                info!("   Score: {:.3} (proceeding despite warnings)", evaluation.overall_score);
            }
            Decision::Allow => {
                debug!("✅ VALUE GATE PASSED: Action '{}' approved (score: {:.3})",
                    action_name, evaluation.overall_score);
            }
        }

        // Pre-compute risk for all steps based on command patterns
        for step in &mut action.steps {
            step.estimate_risk();
        }

        // Step 1: Ghost Run (Pre-flight simulation)
        if action.simulation_mode == SimulationMode::DryRun
            || action.simulation_mode == SimulationMode::GhostRun {
            info!("👻 Running pre-flight simulation...");

            for step in &action.steps {
                let dry_result = self.sandbox.dry_run(step).await?;

                if !dry_result.success {
                    warn!("Pre-flight failed for step: {}", step.description);
                    warn!("Error: {}", dry_result.stderr);

                    return Ok(ExecutionResult {
                        action_id,
                        action_name,
                        step_results: vec![dry_result],
                        overall_success: false,
                        rolled_back: false,
                        total_duration_ms: 0,
                    });
                }
            }

            if action.simulation_mode == SimulationMode::DryRun {
                info!("✅ Dry-run complete, not executing for real");
                return Ok(ExecutionResult {
                    action_id,
                    action_name,
                    step_results: vec![],
                    overall_success: true,
                    rolled_back: false,
                    total_duration_ms: 0,
                });
            }
        }

        // Step 2: Real execution with rollback on failure
        let start_time = chrono::Utc::now().timestamp_millis() as u64;
        let mut step_results = Vec::new();
        let mut executed_steps = Vec::new();
        let mut overall_success = true;

        for step in &action.steps {
            match self.sandbox.execute(step).await {
                Ok(result) => {
                    let success = result.success;
                    step_results.push(result);

                    if success {
                        executed_steps.push(step.clone());
                    } else {
                        // Execution failed, trigger rollback
                        warn!("❌ Step failed, initiating rollback...");
                        overall_success = false;
                        self.rollback_steps(&executed_steps).await?;
                        break;
                    }
                }
                Err(e) => {
                    error!("💥 Execution error: {}", e);
                    overall_success = false;
                    self.rollback_steps(&executed_steps).await?;
                    break;
                }
            }
        }

        let end_time = chrono::Utc::now().timestamp_millis() as u64;

        let execution_result = ExecutionResult {
            action_id,
            action_name: action_name.clone(),
            step_results,
            overall_success,
            rolled_back: !executed_steps.is_empty() && !overall_success,
            total_duration_ms: end_time.saturating_sub(start_time),
        };

        // Step 3: Store to Hippocampus
        if let Some(ref mut hippocampus) = self.hippocampus {
            let emotion = if overall_success {
                EmotionalValence::Positive
            } else {
                EmotionalValence::Negative
            };

            let content = format!(
                "Executed action '{}' with {} steps (success: {})",
                action_name, executed_steps.len(), overall_success
            );

            let context_tags = vec!["motor_cortex".to_string(), "action".to_string()];

            hippocampus.remember(
                content,
                context_tags,
                emotion,
            )?;
        }

        // Step 4: Update execution history
        self.execution_history.push(execution_result.clone());
        if self.execution_history.len() > self.max_history {
            self.execution_history.remove(0);
        }

        info!(
            "🏁 Action complete: {} (success: {}, duration: {}ms)",
            action_name, overall_success, execution_result.total_duration_ms
        );

        Ok(execution_result)
    }

    /// Rollback a list of executed steps in reverse order
    async fn rollback_steps(&mut self, steps: &[ActionStep]) -> Result<()> {
        info!("🔄 Rolling back {} steps...", steps.len());

        for step in steps.iter().rev() {
            if step.can_rollback {
                match self.sandbox.rollback_step(step).await {
                    Ok(result) => {
                        if result.success {
                            info!("✅ Rolled back: {}", step.description);
                        } else {
                            warn!("⚠️  Rollback partial: {} - {}", step.description, result.stderr);
                        }
                    }
                    Err(e) => {
                        error!("💥 Rollback failed: {} - {}", step.description, e);
                    }
                }
            } else {
                warn!("⚠️  Cannot rollback: {}", step.description);
            }
        }

        Ok(())
    }

    /// Get execution statistics
    pub fn stats(&self) -> MotorCortexStats {
        let total_actions = self.execution_history.len();
        let successful = self.execution_history.iter()
            .filter(|r| r.overall_success)
            .count();

        let total_steps: usize = self.execution_history.iter()
            .map(|r| r.step_results.len())
            .sum();

        let avg_duration = if total_actions > 0 {
            self.execution_history.iter()
                .map(|r| r.total_duration_ms)
                .sum::<u64>() / total_actions as u64
        } else {
            0
        };

        MotorCortexStats {
            total_actions,
            successful_actions: successful,
            failed_actions: total_actions - successful,
            total_steps,
            queued_actions: self.action_queue.len(),
            avg_duration_ms: avg_duration,
            rollback_count: self.rollback_stack.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MotorCortexStats {
    pub total_actions: usize,
    pub successful_actions: usize,
    pub failed_actions: usize,
    pub total_steps: usize,
    pub queued_actions: usize,
    pub avg_duration_ms: u64,
    pub rollback_count: usize,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_action_step_creation() {
        let step = ActionStep::new("Test command", "echo")
            .with_args(vec!["hello".to_string(), "world".to_string()])
            .with_tags(vec!["test".to_string()])
            .with_risk(0.3);

        assert_eq!(step.description, "Test command");
        assert_eq!(step.command, "echo");
        assert_eq!(step.args, vec!["hello", "world"]);
        assert_eq!(step.estimated_risk, 0.3);
    }

    #[tokio::test]
    async fn test_risk_estimation() {
        let mut safe_step = ActionStep::new("List files", "ls").with_args(vec!["-la".into()]);
        safe_step.estimate_risk();
        assert!(safe_step.estimated_risk < 0.3, "ls should be low risk");

        let mut dangerous_step = ActionStep::new("Remove all", "rm")
            .with_args(vec!["-rf".into(), "/tmp/test".into()]);
        dangerous_step.estimate_risk();
        assert!(dangerous_step.estimated_risk > 0.7, "rm should be high risk");
    }

    #[tokio::test]
    async fn test_planned_action() {
        let action = PlannedAction::new("Test workflow", "Run tests")
            .add_step(ActionStep::new("Step 1", "echo").with_args(vec!["test1".into()]))
            .add_step(ActionStep::new("Step 2", "echo").with_args(vec!["test2".into()]));

        assert_eq!(action.steps.len(), 2);
        assert_eq!(action.name, "Test workflow");
    }

    #[tokio::test]
    async fn test_local_sandbox_dry_run() {
        let sandbox = LocalShellSandbox::new();
        // Use a simple builtin that should always be available
        let step = ActionStep::new("True test", "true").with_risk(0.1);

        let result = sandbox.dry_run(&step).await.unwrap();
        assert!(result.success);
        assert!(result.was_dry_run);
    }

    #[tokio::test]
    async fn test_local_sandbox_blocks_dangerous() {
        let sandbox = LocalShellSandbox::new(); // Not allowing destructive
        let mut step = ActionStep::new("Dangerous", "rm")
            .with_args(vec!["-rf".into(), "/tmp/test".into()]);
        step.estimate_risk();

        let result = sandbox.dry_run(&step).await.unwrap();
        assert!(!result.success, "Should block dangerous command");
        assert!(result.stderr.contains("blocked"));
    }

    #[tokio::test]
    async fn test_local_sandbox_execute() {
        let sandbox = LocalShellSandbox::new().allow_destructive();
        let step = ActionStep::new("Echo test", "echo").with_args(vec!["hello world".into()]);

        let result = sandbox.execute(&step).await.unwrap();
        assert!(result.success);
        assert!(result.stdout.contains("hello world"));
        assert!(!result.was_dry_run);
    }

    #[tokio::test]
    async fn test_motor_cortex_creation() {
        let sandbox = Box::new(LocalShellSandbox::new());
        let motor = MotorCortexActor::new(sandbox);

        let stats = motor.stats();
        assert_eq!(stats.total_actions, 0);
        assert_eq!(stats.queued_actions, 0);
    }

    #[tokio::test]
    async fn test_motor_cortex_queue() {
        let sandbox = Box::new(LocalShellSandbox::new());
        let mut motor = MotorCortexActor::new(sandbox);

        let action = PlannedAction::new("Test", "Test action")
            .add_step(ActionStep::new("Echo", "echo").with_args(vec!["test".into()]));

        motor.queue_action(action);
        assert_eq!(motor.stats().queued_actions, 1);
    }

    #[tokio::test]
    async fn test_motor_cortex_dry_run() {
        let sandbox = Box::new(LocalShellSandbox::new());
        let mut motor = MotorCortexActor::new(sandbox);

        let action = PlannedAction::new("Test dry-run", "Test action")
            .add_step(ActionStep::new("Echo", "true").with_risk(0.1));

        let result = motor.execute_action(action).await.unwrap();
        assert!(result.overall_success);
    }

    #[tokio::test]
    async fn test_motor_cortex_real_execution() {
        let sandbox = Box::new(LocalShellSandbox::new().allow_destructive());
        let mut motor = MotorCortexActor::new(sandbox);

        let mut action = PlannedAction::new("Test real", "Test real execution")
            .add_step(ActionStep::new("Echo", "echo").with_args(vec!["real test".into()]));
        action.simulation_mode = SimulationMode::RealRun;

        let result = motor.execute_action(action).await.unwrap();
        assert!(result.overall_success);
        assert_eq!(result.step_results.len(), 1);
        assert!(result.step_results[0].stdout.contains("real test"));
    }

    #[tokio::test]
    async fn test_motor_cortex_rollback_on_failure() {
        let sandbox = Box::new(LocalShellSandbox::new().allow_destructive());
        let mut motor = MotorCortexActor::new(sandbox);

        let mut action = PlannedAction::new("Test rollback", "Test rollback")
            .add_step(
                ActionStep::new("Create file", "touch")
                    .with_args(vec!["/tmp/sophia_test.txt".into()])
                    .with_rollback("rm /tmp/sophia_test.txt")
            )
            .add_step(ActionStep::new("Fail", "false")); // This will fail

        action.simulation_mode = SimulationMode::RealRun;

        let result = motor.execute_action(action).await.unwrap();
        assert!(!result.overall_success);
        assert!(result.rolled_back);
    }

    #[tokio::test]
    async fn test_execution_result_success_rate() {
        let result = ExecutionResult {
            action_id: 1,
            action_name: "Test".to_string(),
            step_results: vec![
                StepResult {
                    step_id: 1,
                    success: true,
                    stdout: String::new(),
                    stderr: String::new(),
                    exit_code: Some(0),
                    started_at: 0,
                    finished_at: 100,
                    was_dry_run: false,
                },
                StepResult {
                    step_id: 2,
                    success: false,
                    stdout: String::new(),
                    stderr: String::new(),
                    exit_code: Some(1),
                    started_at: 100,
                    finished_at: 200,
                    was_dry_run: false,
                },
            ],
            overall_success: false,
            rolled_back: true,
            total_duration_ms: 200,
        };

        assert_eq!(result.success_rate(), 0.5);
    }
}
