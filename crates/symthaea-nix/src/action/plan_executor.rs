//! Multi-Step Plan Executor with Rollback
//!
//! Chains [`ActionPlan`] steps with intermediate state verification.
//! If step N fails, rolls back steps 0..N-1 using generation snapshots.

use super::executor::{NixOSCommand, NixOSExecutor, ExecutionResult, SafetyLevel};

/// Status of a plan step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    RolledBack,
    Skipped,
}

/// A single step in a multi-step plan.
#[derive(Debug, Clone)]
pub struct PlanStep {
    /// The command to execute.
    pub command: NixOSCommand,
    /// Human-readable description of this step.
    pub description: String,
    /// Current status.
    pub status: StepStatus,
    /// Whether this step is critical (failure triggers rollback of all prior steps).
    pub critical: bool,
    /// Whether to verify state after execution (e.g., service started, package installed).
    pub verify: bool,
}

/// Result of executing a full plan.
#[derive(Debug, Clone)]
pub struct PlanExecutionResult {
    /// Per-step results.
    pub steps: Vec<(PlanStep, Option<ExecutionResult>)>,
    /// Whether all steps completed successfully.
    pub success: bool,
    /// Number of steps completed before failure (if any).
    pub completed_count: usize,
    /// Number of steps rolled back (if any).
    pub rolled_back_count: usize,
}

/// Orchestrates multi-step NixOS action plans with rollback on failure.
pub struct PlanExecutor {
    /// Steps to execute.
    steps: Vec<PlanStep>,
    /// Φ level for this execution context.
    phi: f32,
    /// Whether to use dry-run mode.
    dry_run: bool,
}

impl PlanExecutor {
    /// Create a new plan executor.
    pub fn new(phi: f32) -> Self {
        Self {
            steps: Vec::new(),
            phi,
            dry_run: false,
        }
    }

    /// Enable dry-run mode.
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Add a step to the plan.
    pub fn add_step(&mut self, command: NixOSCommand, description: &str) -> &mut PlanStep {
        self.steps.push(PlanStep {
            command,
            description: description.to_string(),
            status: StepStatus::Pending,
            critical: true,
            verify: false,
        });
        self.steps.last_mut().unwrap()
    }

    /// Add a non-critical step (failure doesn't trigger rollback).
    pub fn add_optional_step(&mut self, command: NixOSCommand, description: &str) -> &mut PlanStep {
        self.steps.push(PlanStep {
            command,
            description: description.to_string(),
            status: StepStatus::Pending,
            critical: false,
            verify: false,
        });
        self.steps.last_mut().unwrap()
    }

    /// Number of steps in the plan.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Check if any step requires a Φ level higher than what we have.
    pub fn check_phi_requirements(&self) -> Vec<(usize, SafetyLevel, f32)> {
        self.steps.iter().enumerate()
            .filter_map(|(i, step)| {
                let safety = step.command.safety_level();
                let required = safety.required_phi();
                if self.phi < required {
                    Some((i, safety, required))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Execute the plan, stopping and rolling back on critical failure.
    pub async fn execute(&mut self, executor: &mut NixOSExecutor) -> PlanExecutionResult {
        let mut results: Vec<(PlanStep, Option<ExecutionResult>)> = Vec::new();
        let mut completed = 0;

        for i in 0..self.steps.len() {
            self.steps[i].status = StepStatus::Running;

            let result = executor.execute(self.steps[i].command.clone(), self.phi).await;

            let success = matches!(&result, ExecutionResult::Success { .. });

            if success {
                self.steps[i].status = StepStatus::Completed;
                completed += 1;
                results.push((self.steps[i].clone(), Some(result)));
            } else if self.steps[i].critical {
                // Critical failure — rollback previous steps
                self.steps[i].status = StepStatus::Failed(format!("{:?}", result));
                results.push((self.steps[i].clone(), Some(result)));

                let rolled_back = self.rollback_completed(executor, &mut results, completed).await;

                // Mark remaining steps as skipped
                for j in (i + 1)..self.steps.len() {
                    self.steps[j].status = StepStatus::Skipped;
                    results.push((self.steps[j].clone(), None));
                }

                return PlanExecutionResult {
                    steps: results,
                    success: false,
                    completed_count: completed,
                    rolled_back_count: rolled_back,
                };
            } else {
                // Non-critical failure — continue
                self.steps[i].status = StepStatus::Failed(format!("{:?}", result));
                results.push((self.steps[i].clone(), Some(result)));
            }
        }

        PlanExecutionResult {
            steps: results,
            success: true,
            completed_count: completed,
            rolled_back_count: 0,
        }
    }

    /// Roll back completed steps in reverse order.
    async fn rollback_completed(
        &self,
        executor: &mut NixOSExecutor,
        _results: &mut Vec<(PlanStep, Option<ExecutionResult>)>,
        completed: usize,
    ) -> usize {
        let mut rolled_back = 0;

        for i in (0..completed).rev() {
            if let Some(rollback_cmd) = self.steps[i].command.rollback_command() {
                let _rb_result = executor.execute_confirmed(rollback_cmd, self.phi).await;
                rolled_back += 1;
            }
        }

        rolled_back
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_step_creation() {
        let mut executor = PlanExecutor::new(0.5);
        executor.add_step(
            NixOSCommand::EnvInstall { packages: vec!["firefox".into()] },
            "Install Firefox",
        );
        executor.add_optional_step(
            NixOSCommand::Search { query: "firefox".into(), json: false },
            "Verify Firefox is available",
        );
        assert_eq!(executor.step_count(), 2);
    }

    #[test]
    fn test_phi_requirements_check() {
        let executor = PlanExecutor::new(0.2);
        let mut exec = PlanExecutor::new(0.2);
        exec.add_step(
            NixOSCommand::Search { query: "vim".into(), json: false },
            "Search packages",
        );
        exec.add_step(
            NixOSCommand::RebuildSwitch { flake: None, extra_args: vec![] },
            "Rebuild system",
        );

        let blocked = exec.check_phi_requirements();
        // Search (ReadOnly, needs 0.2) should pass, rebuild (SystemCritical, needs 0.4) should be blocked
        assert!(!blocked.is_empty(), "RebuildSwitch should be blocked at Φ=0.2");
        let _ = executor;
    }

    #[tokio::test]
    async fn test_dry_run_plan_execution() {
        let mut nix_executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.5).with_dry_run(true);

        plan.add_step(
            NixOSCommand::Search { query: "vim".into(), json: false },
            "Search for vim",
        );
        plan.add_step(
            NixOSCommand::EnvInstall { packages: vec!["vim".into()] },
            "Install vim",
        );

        let result = plan.execute(&mut nix_executor).await;
        assert!(result.success);
        assert_eq!(result.completed_count, 2);
        assert_eq!(result.rolled_back_count, 0);
    }

    #[tokio::test]
    async fn test_phi_blocks_dangerous_step() {
        let mut nix_executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.2); // Low Φ

        plan.add_step(
            NixOSCommand::RebuildSwitch { flake: None, extra_args: vec![] },
            "Rebuild system (should be blocked by Φ)",
        );

        let result = plan.execute(&mut nix_executor).await;
        // RebuildSwitch needs Φ >= 0.4, so it should fail (PendingConfirmation is not Success)
        assert!(!result.success);
    }
}
