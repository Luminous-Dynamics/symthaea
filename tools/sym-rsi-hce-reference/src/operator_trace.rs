//! Bounded interactive research-operator trace contract.
//!
//! This validates already-recorded operator actions. It does not execute tools,
//! authorize permissions, authenticate artifacts, or establish that the tool
//! outputs are scientifically correct.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperatorMode {
    FixedOneShot,
    InteractiveReact,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperatorActionKind {
    Inspect,
    Hypothesize,
    Execute,
    Observe,
    Revise,
    Submit,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperatorActionDisposition {
    Succeeded,
    Failed,
    Denied,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorTrace {
    pub trace_id: String,
    pub candidate_id: String,
    pub mode: OperatorMode,
    pub max_steps: u32,
    pub max_tool_calls: u32,
    pub wall_clock_budget_ms: u64,
    pub max_resource_cost_units: f64,
    pub allowed_tools: Vec<ToolPermissionSpec>,
    pub actions: Vec<OperatorAction>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolPermissionSpec {
    pub tool_id: String,
    pub permission_id: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorAction {
    pub step: u32,
    pub kind: OperatorActionKind,
    pub start_ms: u64,
    pub end_ms: u64,
    pub disposition: OperatorActionDisposition,
    pub tool_call: bool,
    pub tool_id: Option<String>,
    pub permission_id: Option<String>,
    pub input_sha256: Option<String>,
    pub output_sha256: Option<String>,
    pub resource_cost_units: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct OperatorTraceReport {
    pub trace_id: String,
    pub candidate_id: String,
    pub mode: OperatorMode,
    pub action_count: usize,
    pub tool_call_count: u32,
    pub inspect_count: u32,
    pub hypothesis_count: u32,
    pub execution_count: u32,
    pub successful_executions: u32,
    pub failed_or_denied_executions: u32,
    pub observation_count: u32,
    pub revision_count: u32,
    pub recovery_count: u32,
    pub submitted: bool,
    pub first_action_start_ms: u64,
    pub final_action_end_ms: u64,
    pub total_elapsed_ms: u64,
    pub total_resource_cost_units: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperatorTraceError {
    EmptyIdentifier(&'static str),
    InvalidStepBudget,
    InvalidToolBudget,
    InvalidWallClockBudget,
    InvalidResourceBudget,
    EmptyAllowedTools,
    DuplicateAllowedTool(String),
    EmptyActions,
    TooManySteps { maximum: u32, actual: usize },
    StepSequenceMismatch { expected: u32, actual: u32 },
    InvalidActionInterval(u32),
    NonMonotonicActionTime(u32),
    ActionExceedsWallClockBudget(u32),
    InvalidResourceCost(u32),
    ResourceBudgetExceeded,
    ToolBudgetExceeded,
    ToolCallPermissionMissing(u32),
    ToolNotAllowed { step: u32, tool_id: String },
    ToolPermissionMismatch(u32),
    ToolPermissionUnexpected(u32),
    InvalidActionDigest { step: u32, field: &'static str },
    ToolNotAllowedForAction(u32),
    ToolRequiredForAction(u32),
    HypothesisRequired(u32),
    ObservationRequired(u32),
    RevisionRequired(u32),
    RevisionWithoutFailure(u32),
    SubmitBeforeSuccessfulExecution(u32),
    InvalidSubmit(u32),
    SubmitNotFinal,
    MissingRequiredAction(&'static str),
    TooManyExecutions,
}

impl std::fmt::Display for OperatorTraceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for OperatorTraceError {}

impl OperatorTrace {
    pub fn parse(contents: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(contents)
    }

    pub fn validate(&self) -> Result<OperatorTraceReport, OperatorTraceError> {
        if !non_empty(&self.trace_id) {
            return Err(OperatorTraceError::EmptyIdentifier("trace_id"));
        }
        if !non_empty(&self.candidate_id) {
            return Err(OperatorTraceError::EmptyIdentifier("candidate_id"));
        }
        if self.max_steps == 0 {
            return Err(OperatorTraceError::InvalidStepBudget);
        }
        if self.max_tool_calls == 0 {
            return Err(OperatorTraceError::InvalidToolBudget);
        }
        if self.wall_clock_budget_ms == 0 {
            return Err(OperatorTraceError::InvalidWallClockBudget);
        }
        if !self.max_resource_cost_units.is_finite() || self.max_resource_cost_units < 0.0 {
            return Err(OperatorTraceError::InvalidResourceBudget);
        }
        if self.allowed_tools.is_empty() {
            return Err(OperatorTraceError::EmptyAllowedTools);
        }

        let mut allowed = std::collections::BTreeMap::new();
        let mut permissions = BTreeSet::new();
        for tool in &self.allowed_tools {
            if !non_empty(&tool.tool_id) || !non_empty(&tool.permission_id) {
                return Err(OperatorTraceError::EmptyIdentifier("tool_permission"));
            }
            if allowed.insert(tool.tool_id.as_str(), tool.permission_id.as_str()).is_some() {
                return Err(OperatorTraceError::DuplicateAllowedTool(tool.tool_id.clone()));
            }
            if !permissions.insert(tool.permission_id.as_str()) {
                return Err(OperatorTraceError::DuplicateAllowedTool(tool.permission_id.clone()));
            }
        }
        if self.actions.is_empty() {
            return Err(OperatorTraceError::EmptyActions);
        }
        if self.actions.len() > self.max_steps as usize {
            return Err(OperatorTraceError::TooManySteps {
                maximum: self.max_steps,
                actual: self.actions.len(),
            });
        }

        let mut tool_calls = 0u32;
        let mut inspect_count = 0u32;
        let mut hypothesis_count = 0u32;
        let mut execution_count = 0u32;
        let mut successful_executions = 0u32;
        let mut failed_executions = 0u32;
        let mut observation_count = 0u32;
        let mut revision_count = 0u32;
        let mut recovery_count = 0u32;
        let mut hypothesized = false;
        let mut awaiting_observation: Option<OperatorActionDisposition> = None;
        let mut needs_revision = false;
        let mut submitted = false;
        let mut total_cost = 0.0f64;
        let mut previous_end = None;
        let first_start = self.actions[0].start_ms;
        let final_end = self.actions.last().expect("non-empty actions").end_ms;

        for (index, action) in self.actions.iter().enumerate() {
            let expected_step = index as u32 + 1;
            if action.step != expected_step {
                return Err(OperatorTraceError::StepSequenceMismatch {
                    expected: expected_step,
                    actual: action.step,
                });
            }
            if action.start_ms >= action.end_ms {
                return Err(OperatorTraceError::InvalidActionInterval(action.step));
            }
            if previous_end.is_some_and(|end| action.start_ms < end) {
                return Err(OperatorTraceError::NonMonotonicActionTime(action.step));
            }
            if action.end_ms > self.wall_clock_budget_ms {
                return Err(OperatorTraceError::ActionExceedsWallClockBudget(action.step));
            }
            previous_end = Some(action.end_ms);

            if !action.resource_cost_units.is_finite() || action.resource_cost_units < 0.0 {
                return Err(OperatorTraceError::InvalidResourceCost(action.step));
            }
            total_cost += action.resource_cost_units;
            if !total_cost.is_finite() || total_cost > self.max_resource_cost_units {
                return Err(OperatorTraceError::ResourceBudgetExceeded);
            }

            if submitted {
                return Err(OperatorTraceError::SubmitNotFinal);
            }
            if awaiting_observation.is_some() && action.kind != OperatorActionKind::Observe {
                return Err(OperatorTraceError::ObservationRequired(action.step));
            }
            if needs_revision && action.kind != OperatorActionKind::Revise {
                return Err(OperatorTraceError::RevisionRequired(action.step));
            }

            if action.tool_call {
                tool_calls = tool_calls.saturating_add(1);
                if tool_calls > self.max_tool_calls {
                    return Err(OperatorTraceError::ToolBudgetExceeded);
                }
                let tool_id = action
                    .tool_id
                    .as_deref()
                    .filter(|value| non_empty(value))
                    .ok_or(OperatorTraceError::ToolCallPermissionMissing(action.step))?;
                let permission_id = action
                    .permission_id
                    .as_deref()
                    .filter(|value| non_empty(value))
                    .ok_or(OperatorTraceError::ToolCallPermissionMissing(action.step))?;
                let expected_permission = allowed.get(tool_id).ok_or_else(|| {
                    OperatorTraceError::ToolNotAllowed {
                        step: action.step,
                        tool_id: tool_id.to_owned(),
                    }
                })?;
                if permission_id != *expected_permission {
                    return Err(OperatorTraceError::ToolPermissionMismatch(action.step));
                }
                if action.input_sha256.as_deref().is_none_or(|digest| !valid_digest(digest)) {
                    return Err(OperatorTraceError::InvalidActionDigest {
                        step: action.step,
                        field: "input_sha256",
                    });
                }
                if action.disposition == OperatorActionDisposition::Succeeded
                    && action.output_sha256.as_deref().is_none_or(|digest| !valid_digest(digest))
                {
                    return Err(OperatorTraceError::InvalidActionDigest {
                        step: action.step,
                        field: "output_sha256",
                    });
                }
            } else if action.tool_id.is_some() || action.permission_id.is_some() {
                return Err(OperatorTraceError::ToolPermissionUnexpected(action.step));
            }

            match action.kind {
                OperatorActionKind::Inspect => {
                    inspect_count += 1;
                    if action.disposition == OperatorActionDisposition::Succeeded
                        && action.output_sha256.as_deref().is_none_or(|digest| !valid_digest(digest))
                    {
                        return Err(OperatorTraceError::InvalidActionDigest {
                            step: action.step,
                            field: "output_sha256",
                        });
                    }
                }
                OperatorActionKind::Hypothesize => {
                    if action.tool_call {
                        return Err(OperatorTraceError::ToolNotAllowedForAction(action.step));
                    }
                    if action.disposition != OperatorActionDisposition::Succeeded {
                        return Err(OperatorTraceError::HypothesisRequired(action.step));
                    }
                    if action.output_sha256.as_deref().is_none_or(|digest| !valid_digest(digest)) {
                        return Err(OperatorTraceError::InvalidActionDigest {
                            step: action.step,
                            field: "output_sha256",
                        });
                    }
                    hypothesized = true;
                    hypothesis_count += 1;
                }
                OperatorActionKind::Execute => {
                    if !action.tool_call {
                        return Err(OperatorTraceError::ToolRequiredForAction(action.step));
                    }
                    if !hypothesized {
                        return Err(OperatorTraceError::HypothesisRequired(action.step));
                    }
                    execution_count += 1;
                    if execution_count > self.max_tool_calls {
                        return Err(OperatorTraceError::ToolBudgetExceeded);
                    }
                    awaiting_observation = Some(action.disposition);
                }
                OperatorActionKind::Observe => {
                    if action.tool_call {
                        return Err(OperatorTraceError::ToolNotAllowedForAction(action.step));
                    }
                    let execution_result = awaiting_observation
                        .take()
                        .ok_or(OperatorTraceError::ObservationRequired(action.step))?;
                    if action.disposition != OperatorActionDisposition::Succeeded
                        || action.output_sha256.as_deref().is_none_or(|digest| !valid_digest(digest))
                    {
                        return Err(OperatorTraceError::InvalidActionDigest {
                            step: action.step,
                            field: "output_sha256",
                        });
                    }
                    observation_count += 1;
                    match execution_result {
                        OperatorActionDisposition::Succeeded => {
                            successful_executions += 1;
                            needs_revision = false;
                        }
                        OperatorActionDisposition::Failed | OperatorActionDisposition::Denied => {
                            failed_executions += 1;
                            needs_revision = true;
                        }
                    }
                }
                OperatorActionKind::Revise => {
                    if action.tool_call {
                        return Err(OperatorTraceError::ToolNotAllowedForAction(action.step));
                    }
                    if !needs_revision {
                        return Err(OperatorTraceError::RevisionWithoutFailure(action.step));
                    }
                    if action.disposition != OperatorActionDisposition::Succeeded
                        || action.output_sha256.as_deref().is_none_or(|digest| !valid_digest(digest))
                    {
                        return Err(OperatorTraceError::InvalidActionDigest {
                            step: action.step,
                            field: "output_sha256",
                        });
                    }
                    needs_revision = false;
                    hypothesized = true;
                    revision_count += 1;
                    recovery_count += 1;
                }
                OperatorActionKind::Submit => {
                    if action.tool_call {
                        return Err(OperatorTraceError::ToolNotAllowedForAction(action.step));
                    }
                    if needs_revision || awaiting_observation.is_some() || successful_executions == 0 {
                        return Err(OperatorTraceError::SubmitBeforeSuccessfulExecution(action.step));
                    }
                    if action.disposition != OperatorActionDisposition::Succeeded
                        || action.input_sha256.as_deref().is_none_or(|digest| !valid_digest(digest))
                        || action.output_sha256.as_deref().is_none_or(|digest| !valid_digest(digest))
                    {
                        return Err(OperatorTraceError::InvalidSubmit(action.step));
                    }
                    submitted = true;
                }
            }
        }

        if awaiting_observation.is_some() {
            return Err(OperatorTraceError::ObservationRequired(
                self.actions.len() as u32 + 1,
            ));
        }
        if needs_revision {
            return Err(OperatorTraceError::RevisionRequired(
                self.actions.len() as u32 + 1,
            ));
        }
        if !submitted {
            return Err(OperatorTraceError::MissingRequiredAction("submit"));
        }
        if hypothesis_count == 0 {
            return Err(OperatorTraceError::MissingRequiredAction("hypothesize"));
        }
        if inspect_count == 0 {
            return Err(OperatorTraceError::MissingRequiredAction("inspect"));
        }
        if execution_count == 0 || observation_count == 0 {
            return Err(OperatorTraceError::MissingRequiredAction("execute_and_observe"));
        }
        if self.mode == OperatorMode::InteractiveReact && failed_executions > 0 && recovery_count == 0 {
            return Err(OperatorTraceError::RevisionRequired(self.actions.len() as u32));
        }
        if self.mode == OperatorMode::FixedOneShot && (execution_count > 1 || revision_count > 0) {
            return Err(OperatorTraceError::TooManyExecutions);
        }

        Ok(OperatorTraceReport {
            trace_id: self.trace_id.clone(),
            candidate_id: self.candidate_id.clone(),
            mode: self.mode,
            action_count: self.actions.len(),
            tool_call_count: tool_calls,
            inspect_count,
            hypothesis_count,
            execution_count,
            successful_executions,
            failed_or_denied_executions: failed_executions,
            observation_count,
            revision_count,
            recovery_count,
            submitted,
            first_action_start_ms: first_start,
            final_action_end_ms: final_end,
            total_elapsed_ms: final_end.saturating_sub(first_start),
            total_resource_cost_units: total_cost,
        })
    }
}

pub fn parse_and_validate_operator_trace(
    contents: &str,
) -> Result<OperatorTraceReport, Box<dyn std::error::Error>> {
    let trace = OperatorTrace::parse(contents)?;
    Ok(trace.validate()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = include_str!("../fixtures/operator-trace-v0.json");

    fn valid_trace() -> OperatorTrace {
        OperatorTrace::parse(FIXTURE).expect("operator trace fixture parses")
    }

    #[test]
    fn accepts_interactive_trace_with_failure_observation_revision_and_recovery() {
        let report = valid_trace().validate().expect("interactive trace is valid");
        assert_eq!(report.action_count, 8);
        assert_eq!(report.tool_call_count, 3);
        assert_eq!(report.execution_count, 2);
        assert_eq!(report.successful_executions, 1);
        assert_eq!(report.failed_or_denied_executions, 1);
        assert_eq!(report.observation_count, 2);
        assert_eq!(report.revision_count, 1);
        assert_eq!(report.recovery_count, 1);
        assert!(report.submitted);
        assert!((report.total_resource_cost_units - 3.4).abs() < 1e-12);
    }

    #[test]
    fn rejects_unallowlisted_tool() {
        let mut trace = valid_trace();
        trace.actions[0].tool_id = Some("network-fetch".to_owned());
        assert!(matches!(
            trace.validate(),
            Err(OperatorTraceError::ToolNotAllowed { step: 1, .. })
        ));
    }

    #[test]
    fn rejects_tool_call_without_permission() {
        let mut trace = valid_trace();
        trace.actions[0].permission_id = None;
        assert_eq!(
            trace.validate().expect_err("permission required"),
            OperatorTraceError::ToolCallPermissionMissing(1)
        );
    }

    #[test]
    fn rejects_tool_call_with_wrong_permission_for_allowlisted_tool() {
        let mut trace = valid_trace();
        trace.actions[0].permission_id = Some("perm-exec-sandbox".to_owned());
        assert_eq!(
            trace.validate().expect_err("tool must use its bound permission"),
            OperatorTraceError::ToolPermissionMismatch(1)
        );
    }

    #[test]
    fn rejects_non_contiguous_step_ids() {
        let mut trace = valid_trace();
        trace.actions[1].step = 4;
        assert!(matches!(
            trace.validate(),
            Err(OperatorTraceError::StepSequenceMismatch { expected: 2, actual: 4 })
        ));
    }

    #[test]
    fn rejects_failed_execution_without_observe_and_revise() {
        let mut trace = valid_trace();
        trace.actions.remove(4);
        assert_eq!(
            trace.validate().expect_err("failed execution needs recovery"),
            OperatorTraceError::RevisionRequired(5)
        );
    }

    #[test]
    fn rejects_execution_before_hypothesis() {
        let mut trace = valid_trace();
        trace.actions.remove(1);
        assert_eq!(
            trace.validate().expect_err("hypothesis must precede execute"),
            OperatorTraceError::HypothesisRequired(2)
        );
    }

    #[test]
    fn rejects_tool_budget_overflow() {
        let mut trace = valid_trace();
        trace.max_tool_calls = 2;
        assert_eq!(
            trace.validate().expect_err("three calls exceed limit"),
            OperatorTraceError::ToolBudgetExceeded
        );
    }

    #[test]
    fn rejects_action_past_wall_clock_budget() {
        let mut trace = valid_trace();
        trace.actions[7].end_ms = trace.wall_clock_budget_ms + 1;
        assert_eq!(
            trace.validate().expect_err("action exceeds wall clock"),
            OperatorTraceError::ActionExceedsWallClockBudget(8)
        );
    }

    #[test]
    fn rejects_resource_cost_over_budget() {
        let mut trace = valid_trace();
        trace.max_resource_cost_units = 3.0;
        assert_eq!(
            trace.validate().expect_err("total cost exceeds budget"),
            OperatorTraceError::ResourceBudgetExceeded
        );
    }

    #[test]
    fn rejects_submission_before_observed_success() {
        let mut trace = valid_trace();
        trace.actions[6].kind = OperatorActionKind::Submit;
        assert_eq!(
            trace.validate().expect_err("submit cannot replace observation"),
            OperatorTraceError::SubmitBeforeSuccessfulExecution(7)
        );
    }

    #[test]
    fn rejects_digest_on_tool_output_after_success_if_missing() {
        let mut trace = valid_trace();
        trace.actions[0].output_sha256 = None;
        assert_eq!(
            trace.validate().expect_err("successful tool action needs output digest"),
            OperatorTraceError::InvalidActionDigest { step: 1, field: "output_sha256" }
        );
    }
}
