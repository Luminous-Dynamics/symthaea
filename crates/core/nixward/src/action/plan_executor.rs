// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Step Plan Executor with Rollback
//!
//! Chains NixOS action steps with explicit read-only verification and rollback.
//! Plan-level dry-run is authoritative: when enabled, the supplied executor is
//! never used for command dispatch. Rollback respects the normal Phi gate by
//! default and only bypasses it when the caller explicitly declares that the
//! compensation path was pre-authorized upstream.

use super::executor::{ExecutionResult, NixOSCommand, NixOSExecutor, SafetyLevel};

/// Status of a plan step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepStatus {
    Pending,
    Running,
    Verifying,
    Completed,
    Failed(String),
    RolledBack,
    Skipped,
}

/// How rollback commands are authorized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RollbackAuthorization {
    /// Rollback passes through the same Phi gate as every other command.
    #[default]
    RespectPhi,
    /// Rollback was separately authorized as compensation for this exact plan.
    /// This maps to `NixOSExecutor::execute_confirmed` and must not be selected
    /// merely because rollback is desirable.
    PreauthorizedCompensation,
}

/// Error constructing an unsafe plan step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanBuildError {
    /// Verification commands must be read-only so checking a postcondition
    /// cannot itself smuggle a state mutation into the plan.
    VerificationMustBeReadOnly { safety_level: SafetyLevel },
}

impl std::fmt::Display for PlanBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VerificationMustBeReadOnly { safety_level } => write!(
                f,
                "verification command must be read-only, got {safety_level:?}"
            ),
        }
    }
}

impl std::error::Error for PlanBuildError {}

/// A single step in a multi-step plan.
#[derive(Debug, Clone)]
pub struct PlanStep {
    /// The command to execute.
    pub command: NixOSCommand,
    /// Human-readable description of this step.
    pub description: String,
    /// Current status.
    pub status: StepStatus,
    /// Whether failure triggers rollback of applied prior steps.
    pub critical: bool,
    /// Optional explicit read-only postcondition command.
    pub verification: Option<NixOSCommand>,
}

impl PlanStep {
    /// Attach an explicit read-only postcondition check.
    pub fn verify_with(
        &mut self,
        verification: NixOSCommand,
    ) -> Result<&mut Self, PlanBuildError> {
        let safety_level = verification.safety_level();
        if safety_level != SafetyLevel::ReadOnly {
            return Err(PlanBuildError::VerificationMustBeReadOnly { safety_level });
        }
        self.verification = Some(verification);
        Ok(self)
    }
}

/// Result of executing a full plan.
#[derive(Debug, Clone)]
pub struct PlanExecutionResult {
    /// Per-step primary execution results.
    pub steps: Vec<(PlanStep, Option<ExecutionResult>)>,
    /// Verification command results keyed by step index.
    pub verification_results: Vec<(usize, ExecutionResult)>,
    /// Whether the plan met its success policy.
    ///
    /// As before, non-critical step failures do not make the whole plan fail.
    pub success: bool,
    /// Number of steps whose primary execution and required verification completed.
    pub completed_count: usize,
    /// Number of rollback commands attempted.
    pub rolled_back_count: usize,
    /// Rollback failures: `(step_index, error_description)`.
    pub rollback_failures: Vec<(usize, String)>,
}

/// Orchestrates multi-step NixOS action plans with rollback on failure.
pub struct PlanExecutor {
    steps: Vec<PlanStep>,
    phi: f32,
    dry_run: bool,
    rollback_authorization: RollbackAuthorization,
}

impl PlanExecutor {
    /// Create a new plan executor.
    pub fn new(phi: f32) -> Self {
        Self {
            steps: Vec::new(),
            phi,
            dry_run: false,
            rollback_authorization: RollbackAuthorization::RespectPhi,
        }
    }

    /// Enable authoritative plan-level dry-run mode.
    ///
    /// When enabled, `execute()` dispatches through a fresh dry-run executor,
    /// so a live executor supplied by the caller cannot accidentally perform
    /// mutations. Dry-run execution is intentionally not written into the
    /// caller's live execution history.
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Declare that rollback commands for this exact plan have already been
    /// authorized upstream as compensation.
    pub fn with_preauthorized_compensation(mut self) -> Self {
        self.rollback_authorization = RollbackAuthorization::PreauthorizedCompensation;
        self
    }

    /// Current rollback authorization mode.
    pub fn rollback_authorization(&self) -> RollbackAuthorization {
        self.rollback_authorization
    }

    /// Add a critical step to the plan.
    pub fn add_step(&mut self, command: NixOSCommand, description: &str) -> &mut PlanStep {
        self.steps.push(PlanStep {
            command,
            description: description.to_string(),
            status: StepStatus::Pending,
            critical: true,
            verification: None,
        });
        self.steps.last_mut().expect("just pushed")
    }

    /// Add a non-critical step; failure does not trigger rollback of prior steps.
    pub fn add_optional_step(&mut self, command: NixOSCommand, description: &str) -> &mut PlanStep {
        self.steps.push(PlanStep {
            command,
            description: description.to_string(),
            status: StepStatus::Pending,
            critical: false,
            verification: None,
        });
        self.steps.last_mut().expect("just pushed")
    }

    /// Number of steps in the plan.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Check primary and verification commands that require a higher Phi level.
    pub fn check_phi_requirements(&self) -> Vec<(usize, SafetyLevel, f32)> {
        let mut blocked = Vec::new();
        for (index, step) in self.steps.iter().enumerate() {
            for command in std::iter::once(&step.command).chain(step.verification.iter()) {
                let safety = command.safety_level();
                let required = safety.required_phi();
                if self.phi < required {
                    blocked.push((index, safety, required));
                }
            }
        }
        blocked
    }

    /// Execute the plan, stopping and compensating applied steps on critical failure.
    pub async fn execute(&mut self, executor: &mut NixOSExecutor) -> PlanExecutionResult {
        if self.dry_run {
            let mut dry_executor = NixOSExecutor::new().with_dry_run(true);
            self.execute_with(&mut dry_executor).await
        } else {
            self.execute_with(executor).await
        }
    }

    async fn execute_with(&mut self, executor: &mut NixOSExecutor) -> PlanExecutionResult {
        let mut results: Vec<(PlanStep, Option<ExecutionResult>)> = Vec::new();
        let mut verification_results = Vec::new();
        let mut completed = 0;

        for index in 0..self.steps.len() {
            self.steps[index].status = StepStatus::Running;
            let primary = executor
                .execute(self.steps[index].command.clone(), self.phi)
                .await;
            let primary_succeeded = matches!(&primary, ExecutionResult::Success { .. });

            let mut failure = if primary_succeeded {
                None
            } else {
                Some(format!("primary execution failed: {primary:?}"))
            };

            if primary_succeeded
                && let Some(verification) = self.steps[index].verification.clone()
            {
                self.steps[index].status = StepStatus::Verifying;
                let verification_result = executor.execute(verification, self.phi).await;
                let verification_succeeded =
                    matches!(&verification_result, ExecutionResult::Success { .. });
                if !verification_succeeded {
                    failure = Some(format!(
                        "postcondition verification failed: {verification_result:?}"
                    ));
                }
                verification_results.push((index, verification_result));
            }

            if let Some(error) = failure {
                self.steps[index].status = StepStatus::Failed(error);
                results.push((self.steps[index].clone(), Some(primary)));

                if self.steps[index].critical {
                    let (rolled_back_count, rollback_failures) =
                        self.rollback_applied(executor, &mut results).await;
                    for remaining in (index + 1)..self.steps.len() {
                        self.steps[remaining].status = StepStatus::Skipped;
                        results.push((self.steps[remaining].clone(), None));
                    }
                    return PlanExecutionResult {
                        steps: results,
                        verification_results,
                        success: false,
                        completed_count: completed,
                        rolled_back_count,
                        rollback_failures,
                    };
                }

                continue;
            }

            self.steps[index].status = StepStatus::Completed;
            completed += 1;
            results.push((self.steps[index].clone(), Some(primary)));
        }

        PlanExecutionResult {
            steps: results,
            verification_results,
            success: true,
            completed_count: completed,
            rolled_back_count: 0,
            rollback_failures: Vec::new(),
        }
    }

    /// Roll back every processed step whose primary execution actually
    /// succeeded, in reverse execution order. This deliberately keys off the
    /// recorded effect rather than `StepStatus`: a step whose mutation applied
    /// but whose postcondition later failed still requires compensation.
    async fn rollback_applied(
        &mut self,
        executor: &mut NixOSExecutor,
        results: &mut [(PlanStep, Option<ExecutionResult>)],
    ) -> (usize, Vec<(usize, String)>) {
        let mut rolled_back = 0;
        let mut failures = Vec::new();

        for index in (0..results.len()).rev() {
            let primary_applied = results[index]
                .1
                .as_ref()
                .is_some_and(|result| matches!(result, ExecutionResult::Success { .. }));
            if !primary_applied || self.steps[index].status == StepStatus::RolledBack {
                continue;
            }
            let Some(rollback_command) = self.steps[index].command.rollback_command() else {
                continue;
            };

            let rollback_result = match self.rollback_authorization {
                RollbackAuthorization::RespectPhi => {
                    executor.execute(rollback_command, self.phi).await
                }
                RollbackAuthorization::PreauthorizedCompensation => {
                    executor.execute_confirmed(rollback_command, self.phi).await
                }
            };
            rolled_back += 1;

            match &rollback_result {
                ExecutionResult::Success { .. } => {
                    self.steps[index].status = StepStatus::RolledBack;
                    results[index].0.status = StepStatus::RolledBack;
                }
                ExecutionResult::FailedNoRollback { error, .. } => {
                    failures.push((index, format!("rollback failed: {error}")));
                }
                other => {
                    failures.push((index, format!("rollback not admitted or failed: {other:?}")));
                }
            }
        }

        (rolled_back, failures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search(query: &str) -> NixOSCommand {
        NixOSCommand::Search {
            query: query.into(),
            json: false,
        }
    }

    #[test]
    fn step_creation_defaults_to_no_verification() {
        let mut plan = PlanExecutor::new(0.5);
        let step = plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["firefox".into()],
            },
            "Install Firefox",
        );
        assert!(step.critical);
        assert!(step.verification.is_none());
    }

    #[test]
    fn verification_must_be_read_only() {
        let mut plan = PlanExecutor::new(0.5);
        let step = plan.add_step(search("firefox"), "Search");
        let error = step
            .verify_with(NixOSCommand::EnvInstall {
                packages: vec!["firefox".into()],
            })
            .unwrap_err();
        assert!(matches!(
            error,
            PlanBuildError::VerificationMustBeReadOnly {
                safety_level: SafetyLevel::UserModify
            }
        ));
    }

    #[test]
    fn read_only_verification_is_accepted() {
        let mut plan = PlanExecutor::new(0.5);
        let step = plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["firefox".into()],
            },
            "Install Firefox",
        );
        assert!(step.verify_with(search("firefox")).is_ok());
    }

    #[test]
    fn rollback_bypass_requires_explicit_opt_in() {
        assert_eq!(
            PlanExecutor::new(0.5).rollback_authorization(),
            RollbackAuthorization::RespectPhi
        );
        assert_eq!(
            PlanExecutor::new(0.5)
                .with_preauthorized_compensation()
                .rollback_authorization(),
            RollbackAuthorization::PreauthorizedCompensation
        );
    }

    #[test]
    fn phi_requirements_include_verification() {
        let mut plan = PlanExecutor::new(0.1);
        let step = plan.add_step(search("vim"), "Search packages");
        step.verify_with(search("git")).unwrap();
        let blocked = plan.check_phi_requirements();
        assert_eq!(blocked.len(), 2);
        assert!(blocked
            .iter()
            .all(|(_, safety, _)| *safety == SafetyLevel::ReadOnly));
    }

    #[tokio::test]
    async fn plan_level_dry_run_is_authoritative() {
        let mut live_executor = NixOSExecutor::new();
        let mut plan = PlanExecutor::new(0.5).with_dry_run(true);
        plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["nixward-dry-run-sentinel-do-not-install".into()],
            },
            "Dry-run sentinel",
        );

        let result = plan.execute(&mut live_executor).await;
        assert!(result.success);
        match result.steps[0].1.as_ref().unwrap() {
            ExecutionResult::Success { stdout, .. } => assert!(stdout.contains("[DRY-RUN]")),
            other => panic!("expected dry-run success, got {other:?}"),
        }
        assert!(live_executor.history().is_empty());
    }

    #[tokio::test]
    async fn explicit_verification_runs_after_primary_success() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.5);
        let step = plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["vim".into()],
            },
            "Install vim",
        );
        step.verify_with(search("vim")).unwrap();

        let result = plan.execute(&mut executor).await;
        assert!(result.success);
        assert_eq!(result.completed_count, 1);
        assert_eq!(result.verification_results.len(), 1);
        assert!(matches!(
            &result.verification_results[0].1,
            ExecutionResult::Success { .. }
        ));
    }

    #[tokio::test]
    async fn low_phi_blocks_dangerous_step() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.2);
        plan.add_step(
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Rebuild system",
        );
        let result = plan.execute(&mut executor).await;
        assert!(!result.success);
        assert_eq!(result.completed_count, 0);
    }

    #[tokio::test]
    async fn critical_failure_rolls_back_completed_steps() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.35);
        plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["vim".into()],
            },
            "Install vim",
        );
        plan.add_step(
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Rebuild system",
        );
        plan.add_step(search("htop"), "Search htop");

        let result = plan.execute(&mut executor).await;
        assert!(!result.success);
        assert_eq!(result.completed_count, 1);
        assert_eq!(result.rolled_back_count, 1);
        assert!(result.rollback_failures.is_empty());
        assert_eq!(result.steps[0].0.status, StepStatus::RolledBack);
        assert!(matches!(result.steps[1].0.status, StepStatus::Failed(_)));
        assert_eq!(result.steps[2].0.status, StepStatus::Skipped);
    }

    #[tokio::test]
    async fn rollback_tracks_applied_steps_not_completed_count_indices() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.35);
        plan.add_optional_step(
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Optional blocked rebuild",
        );
        plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["vim".into()],
            },
            "Install vim",
        );
        plan.add_step(
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Critical blocked rebuild",
        );

        let result = plan.execute(&mut executor).await;
        assert!(!result.success);
        assert_eq!(result.completed_count, 1);
        assert_eq!(result.rolled_back_count, 1);
        assert!(result.rollback_failures.is_empty());
        assert!(matches!(result.steps[0].0.status, StepStatus::Failed(_)));
        assert_eq!(result.steps[1].0.status, StepStatus::RolledBack);
        assert!(matches!(result.steps[2].0.status, StepStatus::Failed(_)));
    }

    #[tokio::test]
    async fn rollback_includes_applied_step_even_when_verification_failed() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.5);
        plan.add_step(
            NixOSCommand::EnvInstall {
                packages: vec!["vim".into()],
            },
            "Install vim",
        );
        plan.steps[0].status = StepStatus::Failed("verification failed".into());
        let primary = ExecutionResult::Success {
            stdout: "applied".into(),
            stderr: String::new(),
            execution_time_ms: 0,
        };
        let mut results = vec![(plan.steps[0].clone(), Some(primary))];

        let (rolled_back, failures) = plan.rollback_applied(&mut executor, &mut results).await;
        assert_eq!(rolled_back, 1);
        assert!(failures.is_empty());
        assert_eq!(results[0].0.status, StepStatus::RolledBack);
    }

    #[tokio::test]
    async fn noncritical_failure_continues() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.35);
        plan.add_step(search("vim"), "Search vim");
        plan.add_optional_step(
            NixOSCommand::RebuildSwitch {
                flake: None,
                extra_args: vec![],
            },
            "Optional rebuild",
        );
        plan.add_step(search("git"), "Search git");

        let result = plan.execute(&mut executor).await;
        assert!(result.success);
        assert_eq!(result.completed_count, 2);
        assert_eq!(result.rolled_back_count, 0);
        assert!(matches!(result.steps[1].0.status, StepStatus::Failed(_)));
    }

    #[tokio::test]
    async fn empty_plan_succeeds() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        let mut plan = PlanExecutor::new(0.5);
        let result = plan.execute(&mut executor).await;
        assert!(result.success);
        assert_eq!(result.completed_count, 0);
        assert!(result.steps.is_empty());
        assert!(result.verification_results.is_empty());
    }
}
