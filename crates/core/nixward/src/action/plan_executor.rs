// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Step Plan Executor with Rollback
//!
//! Chains `ActionPlan` steps with intermediate state verification.
//! If step N fails, rolls back steps 0..N-1 using generation snapshots.

use super::executor::{ExecutionResult, NixOSCommand, NixOSExecutor, SafetyLevel};

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
    /// Rollback failures: (step_index, error_description).
    pub rollback_failures: Vec<(usize, String)>,
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
        self.steps.last_mut().expect("just pushed")
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
        self.steps.last_mut().expect("just pushed")
    }

    /// Number of steps in the plan.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Check if any step requires a Φ level higher than what we have.
    pub fn check_phi_requirements(&self) -> Vec<(usize, SafetyLevel, f32)> {
        self.steps
            .iter()
            .enumerate()
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

            let result = executor
                .execute(self.steps[i].command.clone(), self.phi)
                .await;

            let success = matches!(&result, ExecutionResult::Success { .. });

            if success {
                self.steps[i].status = StepStatus::Completed;
                completed += 1;
                results.push((self.steps[i].clone(), Some(result)));
            } else if self.steps[i].critical {
                // Critical failure — rollback previous steps
                self.steps[i].status = StepStatus::Failed(format!("{result:?}"));
                results.push((self.steps[i].clone(), Some(result)));

                let (rolled_back, rollback_failures) = self
                    .rollback_completed(executor, &mut results, completed)
                    .await;

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
                    rollback_failures,
                };
            } else {
                // Non-critical failure — continue
                self.steps[i].status = StepStatus::Failed(format!("{result:?}"));
                results.push((self.steps[i].clone(), Some(result)));
            }
        }

        PlanExecutionResult {
            steps: results,
            success: true,
            completed_count: completed,
            rolled_back_count: 0,
            rollback_failures: Vec::new(),
        }
    }

    /// Roll back completed steps in reverse order.
    ///
    /// Returns `(rolled_back_count, rollback_failures)`.
    async fn rollback_completed(
        &self,
        executor: &mut NixOSExecutor,
        _results: &mut Vec<(PlanStep, Option<ExecutionResult>)>,
        completed: usize,
    ) -> (usize, Vec<(usize, String)>) {
        let mut rolled_back = 0;
        let mut failures = Vec::new();

        for i in (0..completed).rev() {
            if let Some(rollback_cmd) = self.steps[i].command.rollback_command() {
                let rb_result = executor.execute_confirmed(rollback_cmd, self.phi).await;
                match &rb_result {
                    ExecutionResult::Success { .. } => {
                        rolled_back += 1;
                    }
                    ExecutionResult::FailedNoRollback { error, .. } => {
                        rolled_back += 1; // attempted
                        failures.push((i, format!("rollback failed: {error}")));
                    }
                    other => {
                        rolled_back += 1;
                        failures.push((i, format!("rollback unexpected result: {other:?}")));
                    }
                }
            }
        }

        (rolled_back, failures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::executor::ChannelOperation;

    #[test]
    fn test_plan_step_creation() {
        let mut executor = PlanExecutor::new(0.5);
        executor.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["firefox".into()],
            },
            "Install Firefox",
        );
        executor.add_optional_step(
            NixOSCommand::Search {
                query: "firefox".into(),
                json: false,
            },
            "Verify Firefox is available",
        );
        assert_eq!(executor.step_count(), 2);
    }

    #[test]
    fn test_phi_requirements_check() {
        let executor = PlanExecutor::new(0.2);
        let mut exec = PlanExecutor::new(0.2);
        exec.add_step(
            NixOSCommand::Search {
                query: "vim".into(),
                json: false,
            },
            "Search packages",
        );
        exec.add_step(
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Rebuild system",
        );

        let blocked = exec.check_phi_requirements();
        // Search (ReadOnly, needs 0.2) should pass, rebuild (SystemCritical, needs 0.4) should be blocked
        assert!(
            !blocked.is_empty(),
            "RebuildSwitch should be blocked at Φ=0.2"
        );
        let _ = executor;
    }

    #[tokio::test]
    async fn test_dry_run_plan_execution() {
        let mut nix_executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.5).with_dry_run(true);

        plan.add_step(
            NixOSCommand::Search {
                query: "vim".into(),
                json: false,
            },
            "Search for vim",
        );
        plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["vim".into()],
            },
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
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Rebuild system (should be blocked by Φ)",
        );

        let result = plan.execute(&mut nix_executor).await;
        // RebuildSwitch needs Φ >= 0.4, so it should fail (PendingConfirmation is not Success)
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_all_steps_succeed_in_order() {
        let mut nix_executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.5);

        plan.add_step(
            NixOSCommand::Search {
                query: "vim".into(),
                json: false,
            },
            "Search for vim",
        );
        plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["vim".into()],
            },
            "Install vim",
        );
        plan.add_step(
            NixOSCommand::Search {
                query: "git".into(),
                json: false,
            },
            "Search for git",
        );

        assert_eq!(plan.step_count(), 3);

        let result = plan.execute(&mut nix_executor).await;
        assert!(result.success);
        assert_eq!(result.completed_count, 3);
        assert_eq!(result.rolled_back_count, 0);
        assert_eq!(result.steps.len(), 3);

        // Verify all steps have results
        for (step, exec_result) in &result.steps {
            assert_eq!(step.status, StepStatus::Completed);
            assert!(exec_result.is_some());
        }
    }

    #[tokio::test]
    async fn test_critical_failure_triggers_rollback() {
        let mut nix_executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.35);

        // Step 0: install (UserModify, needs 0.3) — should succeed
        plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["vim".into()],
            },
            "Install vim",
        );
        // Step 1: rebuild (SystemCritical, needs 0.4) — should fail (phi=0.35 < 0.4)
        plan.add_step(
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Rebuild system",
        );
        // Step 2: should be skipped
        plan.add_step(
            NixOSCommand::Search {
                query: "htop".into(),
                json: false,
            },
            "Search htop",
        );

        let result = plan.execute(&mut nix_executor).await;
        assert!(!result.success);
        assert_eq!(result.completed_count, 1, "Only step 0 should complete");
        assert_eq!(
            result.rolled_back_count, 1,
            "Step 0 (EnvInstall) should be rolled back"
        );
        assert_eq!(result.steps.len(), 3);

        // Step 0: completed (then rolled back)
        assert!(result.steps[0].1.is_some());
        // Step 1: failed (PendingConfirmation)
        assert!(matches!(result.steps[1].0.status, StepStatus::Failed(_)));
        // Step 2: skipped
        assert_eq!(result.steps[2].0.status, StepStatus::Skipped);
        assert!(result.steps[2].1.is_none());
    }

    #[tokio::test]
    async fn test_noncritical_failure_continues() {
        let mut nix_executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.35);

        // Step 0: search (ReadOnly, needs 0.2) — succeeds
        plan.add_step(
            NixOSCommand::Search {
                query: "vim".into(),
                json: false,
            },
            "Search vim",
        );
        // Step 1: rebuild (SystemCritical, needs 0.4) — fails, but NON-CRITICAL
        plan.add_optional_step(
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Optional rebuild",
        );
        // Step 2: search again — should still execute
        plan.add_step(
            NixOSCommand::Search {
                query: "git".into(),
                json: false,
            },
            "Search git",
        );

        let result = plan.execute(&mut nix_executor).await;
        assert!(
            result.success,
            "Plan should succeed despite non-critical failure"
        );
        assert_eq!(result.completed_count, 2, "Steps 0 and 2 succeed");
        assert_eq!(result.rolled_back_count, 0, "No rollback for non-critical");
        assert_eq!(result.steps.len(), 3);

        // Step 1 failed but non-critical
        assert!(matches!(result.steps[1].0.status, StepStatus::Failed(_)));
        assert!(!result.steps[1].0.critical);
    }

    #[tokio::test]
    async fn test_empty_plan_succeeds() {
        let mut nix_executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.5);

        assert_eq!(plan.step_count(), 0);
        let result = plan.execute(&mut nix_executor).await;
        assert!(result.success);
        assert_eq!(result.completed_count, 0);
        assert_eq!(result.rolled_back_count, 0);
        assert!(result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_first_step_critical_failure_no_rollback_needed() {
        let mut nix_executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.1); // Very low Φ

        // Step 0: rebuild (SystemCritical, needs 0.4) — fails immediately
        plan.add_step(
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Rebuild",
        );
        // Step 1: should be skipped
        plan.add_step(
            NixOSCommand::Search {
                query: "vim".into(),
                json: false,
            },
            "Search",
        );

        let result = plan.execute(&mut nix_executor).await;
        assert!(!result.success);
        assert_eq!(
            result.completed_count, 0,
            "Nothing completed before failure"
        );
        assert_eq!(result.rolled_back_count, 0, "Nothing to roll back");
        assert_eq!(result.steps[1].0.status, StepStatus::Skipped);
    }

    #[test]
    fn test_phi_requirements_detailed() {
        let mut plan = PlanExecutor::new(0.35);

        plan.add_step(
            NixOSCommand::Search {
                query: "vim".into(),
                json: false,
            },
            "Search (ReadOnly, 0.2)",
        );
        plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["vim".into()],
            },
            "Install (UserModify, 0.3)",
        );
        plan.add_step(
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Rebuild (SystemCritical, 0.4)",
        );
        plan.add_step(
            NixOSCommand::CollectGarbage {
                older_than_days: None,
                delete_all: true,
            },
            "GC (Destructive, 0.6)",
        );

        let blocked = plan.check_phi_requirements();
        // At phi=0.35: Search (0.2) ok, Install (0.3) ok, Rebuild (0.4) blocked, GC (0.6) blocked
        assert_eq!(blocked.len(), 2, "Rebuild and GC should be blocked");
        assert_eq!(blocked[0].0, 2, "Step 2 (Rebuild) should be first blocked");
        assert_eq!(blocked[0].1, SafetyLevel::SystemCritical);
        assert_eq!(blocked[1].0, 3, "Step 3 (GC) should be second blocked");
        assert_eq!(blocked[1].1, SafetyLevel::Destructive);
    }

    #[test]
    fn test_step_status_enum_variants() {
        assert_eq!(StepStatus::Pending, StepStatus::Pending);
        assert_eq!(StepStatus::Running, StepStatus::Running);
        assert_eq!(StepStatus::Completed, StepStatus::Completed);
        assert_eq!(StepStatus::RolledBack, StepStatus::RolledBack);
        assert_eq!(StepStatus::Skipped, StepStatus::Skipped);
        assert_ne!(StepStatus::Pending, StepStatus::Completed);
        assert_eq!(
            StepStatus::Failed("reason".into()),
            StepStatus::Failed("reason".into())
        );
        assert_ne!(
            StepStatus::Failed("a".into()),
            StepStatus::Failed("b".into())
        );
    }

    #[test]
    fn test_step_critical_vs_optional() {
        let mut plan = PlanExecutor::new(0.5);

        let step = plan.add_step(
            NixOSCommand::Search {
                query: "vim".into(),
                json: false,
            },
            "Critical step",
        );
        assert!(step.critical, "add_step should create critical steps");
        assert!(!step.verify, "Default verify should be false");

        let opt = plan.add_optional_step(
            NixOSCommand::Search {
                query: "git".into(),
                json: false,
            },
            "Optional step",
        );
        assert!(
            !opt.critical,
            "add_optional_step should create non-critical steps"
        );
    }

    #[test]
    fn test_rollback_command_types() {
        // Rebuild variants all produce nixos-rebuild --rollback
        let switch = NixOSCommand::RebuildSwitch {
            flake: None,
            extra_args: vec![],
        };
        let rb = switch.rollback_command().unwrap();
        let (cmd, args) = rb.to_command();
        assert_eq!(cmd, "nixos-rebuild");
        assert!(args.contains(&"--rollback".to_string()));

        // EnvInstall/Remove produce nix-env --rollback
        let install = NixOSCommand::EnvInstall {
            packages: vec!["vim".into()],
        };
        let rb = install.rollback_command().unwrap();
        let (cmd, args) = rb.to_command();
        assert_eq!(cmd, "nix-env");
        assert!(args.contains(&"--rollback".to_string()));

        // HomeManager uses generation-based rollback
        let hm = NixOSCommand::HomeManagerSwitch { flake: None };
        let rb = hm.rollback_command().unwrap();
        let (cmd, _args) = rb.to_command();
        assert_eq!(cmd, "sh");

        // Commands without rollback
        assert!(
            NixOSCommand::Search {
                query: "x".into(),
                json: false
            }
            .rollback_command()
            .is_none()
        );
        assert!(
            NixOSCommand::CollectGarbage {
                older_than_days: None,
                delete_all: false
            }
            .rollback_command()
            .is_none()
        );
        assert!(
            NixOSCommand::Channel {
                operation: ChannelOperation::List
            }
            .rollback_command()
            .is_none()
        );
    }

    #[tokio::test]
    async fn test_multi_step_rollback_reverse_order() {
        let mut nix_executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.35);

        // 3 steps succeed, then step 3 fails critically
        plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["a".into()],
            },
            "Install a",
        );
        plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["b".into()],
            },
            "Install b",
        );
        plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["c".into()],
            },
            "Install c",
        );
        // This needs 0.4, phi=0.35 → fail
        plan.add_step(
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Rebuild (will fail)",
        );

        let result = plan.execute(&mut nix_executor).await;
        assert!(!result.success);
        assert_eq!(result.completed_count, 3);
        // All 3 EnvInstall commands have rollback → 3 rolled back
        assert_eq!(result.rolled_back_count, 3);
    }

    #[tokio::test]
    async fn test_rollback_failures_tracked() {
        let mut nix_executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.35);

        // Step 0: install (succeeds)
        plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["vim".into()],
            },
            "Install vim",
        );
        // Step 1: rebuild (fails due to phi)
        plan.add_step(
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Rebuild",
        );

        let result = plan.execute(&mut nix_executor).await;
        assert!(!result.success);
        // In dry-run mode, rollback "succeeds" — so rollback_failures should be empty
        assert!(
            result.rollback_failures.is_empty(),
            "Dry-run rollbacks should not produce failures"
        );
    }

    #[tokio::test]
    async fn test_success_path_no_rollback_failures() {
        let mut nix_executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.5);
        plan.add_step(
            NixOSCommand::Search {
                query: "vim".into(),
                json: false,
            },
            "Search",
        );
        let result = plan.execute(&mut nix_executor).await;
        assert!(result.success);
        assert!(result.rollback_failures.is_empty());
    }

    #[tokio::test]
    async fn test_rollback_skips_unrollable_steps() {
        let mut nix_executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.35);

        // Search has no rollback command
        plan.add_step(
            NixOSCommand::Search {
                query: "vim".into(),
                json: false,
            },
            "Search (no rollback)",
        );
        // Install has rollback
        plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["vim".into()],
            },
            "Install (has rollback)",
        );
        // Rebuild fails (needs 0.4)
        plan.add_step(
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Rebuild (will fail)",
        );

        let result = plan.execute(&mut nix_executor).await;
        assert!(!result.success);
        assert_eq!(result.completed_count, 2);
        // Only EnvInstall has rollback, Search doesn't → 1 rolled back
        assert_eq!(result.rolled_back_count, 1);
    }
}
