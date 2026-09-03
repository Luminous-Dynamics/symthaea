// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exact-plan authorization for multi-step NixOS actions.
//!
//! A plan is authorized as a whole, including step order, human-readable
//! descriptions, criticality, verification intent, and rollback commands. If
//! any of those fields change after approval, the plan digest changes and the
//! executor refuses to run it.

use super::context_executor::ContextualExecutor;
use super::execution_context::{AuthorityContext, AuthoritySource, ExecutionContext};
use super::executor::{ExecutionResult, NixOSCommand, SafetyLevel};
use super::plan_executor::{PlanStep, StepStatus};
use serde_json::json;

/// Result of an exact-plan execution.
#[derive(Debug, Clone)]
pub struct ContextPlanExecutionResult {
    pub steps: Vec<(PlanStep, Option<ExecutionResult>)>,
    pub success: bool,
    pub completed_count: usize,
    pub rolled_back_count: usize,
    pub rollback_failures: Vec<(usize, String)>,
    /// Digest of the plan that was actually preflighted.
    pub plan_digest: String,
}

/// A plan whose authority must be bound to its exact canonical digest.
pub struct ContextPlanExecutor {
    steps: Vec<PlanStep>,
}

impl Default for ContextPlanExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextPlanExecutor {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

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

    pub fn add_optional_step(
        &mut self,
        command: NixOSCommand,
        description: &str,
    ) -> &mut PlanStep {
        self.steps.push(PlanStep {
            command,
            description: description.to_string(),
            status: StepStatus::Pending,
            critical: false,
            verify: false,
        });
        self.steps.last_mut().expect("just pushed")
    }

    pub fn steps(&self) -> &[PlanStep] {
        &self.steps
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Canonical digest of all behavior and operator-visible plan text.
    pub fn plan_digest(&self) -> Result<String, String> {
        let canonical: Vec<_> = self
            .steps
            .iter()
            .map(|step| {
                json!({
                    "command": &step.command,
                    "description": &step.description,
                    "critical": step.critical,
                    "verify": step.verify,
                    "rollback": step.command.rollback_command(),
                })
            })
            .collect();
        let encoded = serde_json::to_vec(&canonical)
            .map_err(|error| format!("serialize plan for authority digest: {error}"))?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"nixward-action-plan-authority-v1\0");
        hasher.update(&encoded);
        Ok(hasher.finalize().to_hex().to_string())
    }

    pub fn max_safety(&self) -> SafetyLevel {
        self.steps
            .iter()
            .map(|step| step.command.safety_level())
            .max_by_key(|level| safety_rank(*level))
            .unwrap_or(SafetyLevel::ReadOnly)
    }

    fn modifying(&self) -> bool {
        self.max_safety() != SafetyLevel::ReadOnly
    }

    /// Preflight the exact plan against its authority context.
    pub fn preflight(&self, context: &ExecutionContext) -> Result<String, String> {
        context.validate()?;

        // Existing `PlanStep::verify` has no verifier implementation. The new
        // authority path refuses to imply a guarantee it cannot yet provide.
        if self.steps.iter().any(|step| step.verify) {
            return Err(
                "plan requests post-step verification, but no verifier is bound to this executor"
                    .into(),
            );
        }

        let digest = self.plan_digest()?;
        let max_safety = self.max_safety();
        if !context.allows(max_safety) {
            return Err(format!(
                "authority {:?} with ceiling {:?} does not permit plan maximum {:?}",
                context.authority.source(),
                context.authority.safety_ceiling(),
                max_safety
            ));
        }

        if self.modifying() {
            match context.authority.source() {
                AuthoritySource::UpstreamHumanGate | AuthoritySource::PolicyDecision => {}
                other => {
                    return Err(format!(
                        "modifying plans require exact-plan-bound upstream or policy authority, got {other:?}"
                    ));
                }
            }
            let Some(expected) = context.authority.action_digest() else {
                return Err("modifying plan authority is not bound to a plan digest".into());
            };
            if expected != digest {
                return Err("plan changed after authority was granted".into());
            }
        }

        Ok(digest)
    }

    /// Execute a preflighted exact plan. Each command is rebound to an exact
    /// per-step digest before entering `ContextualExecutor`; the approved plan
    /// therefore cannot be used as a floating capability for another command.
    pub async fn execute(
        &mut self,
        executor: &mut ContextualExecutor,
        context: &ExecutionContext,
    ) -> ContextPlanExecutionResult {
        let plan_digest = match self.preflight(context) {
            Ok(digest) => digest,
            Err(error) => {
                return ContextPlanExecutionResult {
                    steps: Vec::new(),
                    success: false,
                    completed_count: 0,
                    rolled_back_count: 0,
                    rollback_failures: vec![(usize::MAX, format!("preflight failed: {error}"))],
                    plan_digest: self.plan_digest().unwrap_or_else(|_| "invalid".into()),
                };
            }
        };

        let mut results = Vec::new();
        let mut completed = 0usize;

        for index in 0..self.steps.len() {
            self.steps[index].status = StepStatus::Running;
            let command = self.steps[index].command.clone();
            let step_context = match self.bound_step_context(&command, context) {
                Ok(context) => context,
                Err(error) => {
                    self.steps[index].status = StepStatus::Failed(error.clone());
                    results.push((
                        self.steps[index].clone(),
                        Some(ExecutionResult::Blocked {
                            reason: error,
                            safety_level: command.safety_level(),
                        }),
                    ));
                    self.mark_remaining_skipped(index + 1, &mut results);
                    return ContextPlanExecutionResult {
                        steps: results,
                        success: false,
                        completed_count: completed,
                        rolled_back_count: 0,
                        rollback_failures: Vec::new(),
                        plan_digest,
                    };
                }
            };

            let result = executor.execute(command, &step_context).await;
            let success = matches!(&result, ExecutionResult::Success { .. });

            if success {
                self.steps[index].status = StepStatus::Completed;
                completed += 1;
                results.push((self.steps[index].clone(), Some(result)));
                continue;
            }

            self.steps[index].status = StepStatus::Failed(format!("{result:?}"));
            results.push((self.steps[index].clone(), Some(result)));

            if self.steps[index].critical {
                let (rolled_back, rollback_failures) = self
                    .rollback_completed(executor, context, completed)
                    .await;
                self.mark_remaining_skipped(index + 1, &mut results);
                return ContextPlanExecutionResult {
                    steps: results,
                    success: false,
                    completed_count: completed,
                    rolled_back_count: rolled_back,
                    rollback_failures,
                    plan_digest,
                };
            }
        }

        ContextPlanExecutionResult {
            steps: results,
            success: true,
            completed_count: completed,
            rolled_back_count: 0,
            rollback_failures: Vec::new(),
            plan_digest,
        }
    }

    async fn rollback_completed(
        &self,
        executor: &mut ContextualExecutor,
        plan_context: &ExecutionContext,
        completed: usize,
    ) -> (usize, Vec<(usize, String)>) {
        let mut rolled_back = 0usize;
        let mut failures = Vec::new();

        for index in (0..completed).rev() {
            let Some(rollback) = self.steps[index].command.rollback_command() else {
                continue;
            };
            rolled_back += 1;
            let rollback_context = match self.bound_step_context(&rollback, plan_context) {
                Ok(context) => context,
                Err(error) => {
                    failures.push((index, format!("rollback authority binding failed: {error}")));
                    continue;
                }
            };
            let result = executor.execute(rollback, &rollback_context).await;
            if !matches!(result, ExecutionResult::Success { .. }) {
                failures.push((index, format!("rollback failed: {result:?}")));
            }
        }

        (rolled_back, failures)
    }

    fn bound_step_context(
        &self,
        command: &NixOSCommand,
        plan_context: &ExecutionContext,
    ) -> Result<ExecutionContext, String> {
        let digest = ContextualExecutor::command_digest(command)?;
        let ceiling = plan_context.authority.safety_ceiling();
        let authority = match plan_context.authority.source() {
            AuthoritySource::None => AuthorityContext::observe_only(),
            AuthoritySource::DirectOperatorRequest => AuthorityContext::direct_operator_request(),
            AuthoritySource::ExplicitOperatorConfirmation => {
                // This source cannot presently carry an exact binding. It is
                // intentionally unsuitable for modifying multi-step plans.
                AuthorityContext::explicit_operator_confirmation(ceiling)
            }
            AuthoritySource::UpstreamHumanGate => {
                AuthorityContext::upstream_human_gate(ceiling, Some(digest))
            }
            AuthoritySource::PolicyDecision => {
                AuthorityContext::policy_decision(ceiling, Some(digest))
            }
        };
        Ok(ExecutionContext::new(
            authority,
            plan_context.cognition.clone(),
        ))
    }

    fn mark_remaining_skipped(
        &mut self,
        start: usize,
        results: &mut Vec<(PlanStep, Option<ExecutionResult>)>,
    ) {
        for index in start..self.steps.len() {
            self.steps[index].status = StepStatus::Skipped;
            results.push((self.steps[index].clone(), None));
        }
    }
}

fn safety_rank(level: SafetyLevel) -> u8 {
    match level {
        SafetyLevel::ReadOnly => 0,
        SafetyLevel::UserModify => 1,
        SafetyLevel::SystemModify => 2,
        SafetyLevel::SystemCritical => 3,
        SafetyLevel::Destructive => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{CognitiveContext, PhiMeasurement};

    fn rebuild() -> NixOSCommand {
        NixOSCommand::RebuildSwitch {
            flake: None,
            extra_args: vec![],
        }
    }

    fn bound_human_context(plan: &ContextPlanExecutor) -> ExecutionContext {
        ExecutionContext::new(
            AuthorityContext::upstream_human_gate(
                plan.max_safety(),
                Some(plan.plan_digest().unwrap()),
            ),
            CognitiveContext::default(),
        )
    }

    #[tokio::test]
    async fn exact_bound_plan_executes_without_phi() {
        let mut plan = ContextPlanExecutor::new();
        plan.add_step(rebuild(), "Apply reviewed system generation");
        let context = bound_human_context(&plan);
        let mut executor = ContextualExecutor::new().with_dry_run(true);
        let result = plan.execute(&mut executor, &context).await;
        assert!(result.success);
        assert_eq!(result.completed_count, 1);
        assert_eq!(executor.history()[0].measured_phi, None);
    }

    #[tokio::test]
    async fn inserted_step_after_approval_invalidates_plan() {
        let mut plan = ContextPlanExecutor::new();
        plan.add_step(rebuild(), "Apply reviewed system generation");
        let context = bound_human_context(&plan);

        // Escalate after approval. Even though the original context had enough
        // ceiling for the first step, its plan digest no longer matches.
        plan.add_step(
            NixOSCommand::CollectGarbage {
                older_than_days: None,
                delete_all: true,
            },
            "Unexpected destructive cleanup",
        );

        let mut executor = ContextualExecutor::new().with_dry_run(true);
        let result = plan.execute(&mut executor, &context).await;
        assert!(!result.success);
        assert_eq!(result.completed_count, 0);
        assert!(executor.history().is_empty());
    }

    #[test]
    fn changing_operator_visible_description_changes_digest() {
        let mut a = ContextPlanExecutor::new();
        a.add_step(rebuild(), "Safe reviewed rebuild");
        let mut b = ContextPlanExecutor::new();
        b.add_step(rebuild(), "Do something else");
        assert_ne!(a.plan_digest().unwrap(), b.plan_digest().unwrap());
    }

    #[tokio::test]
    async fn high_phi_without_authority_cannot_run_modifying_plan() {
        let mut plan = ContextPlanExecutor::new();
        plan.add_step(rebuild(), "Rebuild");
        let context = ExecutionContext::new(
            AuthorityContext::observe_only(),
            CognitiveContext {
                phi: PhiMeasurement::measured(1.0, "test").unwrap(),
                confidence: Some(1.0),
                free_energy: Some(0.0),
                prediction_error: Some(0.0),
                causal_support: Some(1.0),
            },
        );
        let mut executor = ContextualExecutor::new().with_dry_run(true);
        let result = plan.execute(&mut executor, &context).await;
        assert!(!result.success);
        assert!(executor.history().is_empty());
    }

    #[tokio::test]
    async fn verification_flag_fails_closed_until_verifier_exists() {
        let mut plan = ContextPlanExecutor::new();
        plan.add_step(rebuild(), "Rebuild").verify = true;
        let context = bound_human_context(&plan);
        let mut executor = ContextualExecutor::new().with_dry_run(true);
        let result = plan.execute(&mut executor, &context).await;
        assert!(!result.success);
        assert!(executor.history().is_empty());
    }

    #[tokio::test]
    async fn floating_policy_authority_cannot_authorize_plan() {
        let mut plan = ContextPlanExecutor::new();
        plan.add_step(rebuild(), "Rebuild");
        let context = ExecutionContext::new(
            AuthorityContext::policy_decision(SafetyLevel::SystemCritical, None),
            CognitiveContext::default(),
        );
        let mut executor = ContextualExecutor::new().with_dry_run(true);
        let result = plan.execute(&mut executor, &context).await;
        assert!(!result.success);
        assert!(executor.history().is_empty());
    }
}
