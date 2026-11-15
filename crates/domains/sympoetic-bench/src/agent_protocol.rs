//! Rust-side contract validation for the Symthaea agent decision bridge.
//!
//! The benchmark harness retains authority over tool execution. This module
//! validates proposed actions and enforces the observation/reconciliation
//! state machine; it does not execute tools or inspect benchmark environments.

use std::collections::HashSet;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const PROTOCOL_VERSION: &str = "symthaea-agent-bridge/0.1";
const ALLOWED_ACTIONS: &[&str] = &["tool_call", "ask", "defer", "final_answer"];
const OBSERVATION_STATUSES: &[&str] = &["success", "failure", "permission_denied", "unknown"];
const RECONCILED_STATUSES: &[&str] = &["success", "failure", "not_executed"];
const OBSERVERS: &[&str] = &["tool", "environment", "evaluator"];

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AgentTurnRequest {
    pub protocol_version: String,
    pub kind: String,
    pub turn_id: String,
    pub run_id: String,
    pub principal_id: String,
    pub task: AgentTask,
    pub constraints: Vec<String>,
    pub available_tools: Vec<AvailableTool>,
    pub history: Vec<HistoryEvent>,
    pub authority: Authority,
    pub budget: DecisionBudget,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AgentTask {
    pub task_id: String,
    pub environment_id: String,
    pub role: String,
    pub goal: String,
    pub brief: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qcat: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AvailableTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryEvent {
    pub event_id: String,
    pub actor: String,
    pub kind: String,
    pub content: String,
    pub trust_class: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Authority {
    pub role_id: String,
    pub allowed_actions: Vec<String>,
    pub policy_digest: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionBudget {
    pub tool_calls_remaining: u64,
    pub max_output_tokens: u64,
    pub deadline_ms_remaining: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AgentDecision {
    pub protocol_version: String,
    pub kind: String,
    pub run_id: String,
    pub turn_id: String,
    pub decision_id: String,
    pub action: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_refs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why_required: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked_on: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolObservation {
    pub protocol_version: String,
    pub kind: String,
    pub run_id: String,
    pub turn_id: String,
    pub call_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_result: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precondition_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub postcondition_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_side_effect: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reversible: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconciled_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observer: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingCall {
    call_id: String,
    tool_name: String,
}

#[derive(Debug, Default)]
pub struct AgentProtocolGuard {
    pending_call: Option<PendingCall>,
    unresolved_unknown: Option<PendingCall>,
    seen_decision_ids: HashSet<String>,
    seen_call_ids: HashSet<String>,
}

impl AgentTurnRequest {
    pub fn validate(&self) -> Result<()> {
        if self.protocol_version != PROTOCOL_VERSION || self.kind != "agent_turn" {
            bail!("unsupported agent-turn protocol version or kind");
        }
        for (label, value) in [
            ("turn_id", self.turn_id.as_str()),
            ("run_id", self.run_id.as_str()),
            ("principal_id", self.principal_id.as_str()),
            ("task_id", self.task.task_id.as_str()),
            ("environment_id", self.task.environment_id.as_str()),
            ("role", self.task.role.as_str()),
            ("goal", self.task.goal.as_str()),
            ("brief", self.task.brief.as_str()),
            ("role_id", self.authority.role_id.as_str()),
        ] {
            require_nonempty(label, value)?;
        }
        require_sha256("authority.policy_digest", &self.authority.policy_digest)?;
        if self.authority.allowed_actions.is_empty() {
            bail!("authority.allowed_actions must not be empty");
        }
        let mut allowed_action_set = HashSet::new();
        for action in &self.authority.allowed_actions {
            require_member("authority action", action, ALLOWED_ACTIONS)?;
            if !allowed_action_set.insert(action.as_str()) {
                bail!("duplicate authority action {action:?}");
            }
        }
        if self.budget.max_output_tokens == 0 || self.budget.deadline_ms_remaining == 0 {
            bail!("decision budget is exhausted");
        }
        let mut tool_names = HashSet::new();
        for tool in &self.available_tools {
            require_nonempty("tool name", &tool.name)?;
            if tool.description.trim().is_empty() {
                bail!("tool {:?} has an empty description", tool.name);
            }
            if !tool.input_schema.is_object() {
                bail!("tool {:?} input_schema must be a JSON object", tool.name);
            }
            if !tool_names.insert(tool.name.as_str()) {
                bail!("duplicate available tool {:?}", tool.name);
            }
        }
        if self.constraints.iter().any(|constraint| constraint.trim().is_empty()) {
            bail!("constraints must not contain empty entries");
        }
        let mut event_ids = HashSet::new();
        for event in &self.history {
            require_nonempty("history event_id", &event.event_id)?;
            require_member(
                "history actor",
                &event.actor,
                &["human", "agent", "tool", "environment", "system"],
            )?;
            require_nonempty("history kind", &event.kind)?;
            require_member(
                "history trust_class",
                &event.trust_class,
                &[
                    "trusted_instruction",
                    "task_data",
                    "tool_observation",
                    "agent_output",
                    "evaluator_annotation",
                ],
            )?;
            if !event_ids.insert(event.event_id.as_str()) {
                bail!("duplicate history event_id {:?}", event.event_id);
            }
        }
        Ok(())
    }
}

impl AgentProtocolGuard {
    pub fn has_unresolved_execution(&self) -> bool {
        self.unresolved_unknown.is_some()
    }

    pub fn validate_decision(
        &mut self,
        request: &AgentTurnRequest,
        decision: &AgentDecision,
    ) -> Result<()> {
        request.validate()?;
        if self.pending_call.is_some() {
            bail!("cannot accept a new decision before the previous tool call is observed");
        }
        if self.unresolved_unknown.is_some() {
            bail!("cannot accept a new decision while an execution outcome is unresolved");
        }
        if decision.protocol_version != PROTOCOL_VERSION || decision.kind != "agent_decision" {
            bail!("unsupported agent-decision protocol version or kind");
        }
        if decision.run_id != request.run_id || decision.turn_id != request.turn_id {
            bail!("decision run_id or turn_id does not match the active request");
        }
        require_nonempty("decision_id", &decision.decision_id)?;
        if self.seen_decision_ids.contains(&decision.decision_id) {
            bail!("duplicate decision_id {:?}", decision.decision_id);
        }
        require_member("decision action", &decision.action, ALLOWED_ACTIONS)?;
        if !request.authority.allowed_actions.contains(&decision.action) {
            bail!("decision action {:?} is outside the granted authority", decision.action);
        }

        match decision.action.as_str() {
            "tool_call" => {
                if request.budget.tool_calls_remaining == 0 {
                    bail!("tool-call budget is exhausted");
                }
                let call_id = decision.call_id.as_deref().context("tool_call requires call_id")?;
                let tool_name = decision
                    .tool_name
                    .as_deref()
                    .context("tool_call requires tool_name")?;
                let arguments = decision.arguments.as_ref().context("tool_call requires arguments")?;
                let idempotency_key = decision
                    .idempotency_key
                    .as_deref()
                    .context("tool_call requires idempotency_key")?;
                let purpose = decision.purpose.as_deref().context("tool_call requires purpose")?;
                require_nonempty("call_id", call_id)?;
                require_nonempty("tool_name", tool_name)?;
                require_nonempty("idempotency_key", idempotency_key)?;
                require_nonempty("purpose", purpose)?;
                if purpose.chars().count() > 500 {
                    bail!("tool-call purpose exceeds 500 characters");
                }
                if !request.available_tools.iter().any(|tool| tool.name == tool_name) {
                    bail!("tool {:?} is not in the role-filtered available tool list", tool_name);
                }
                if self.seen_call_ids.contains(call_id) {
                    bail!("duplicate call_id {:?}", call_id);
                }
                if decision.question.is_some()
                    || decision.why_required.is_some()
                    || decision.reason.is_some()
                    || decision.content.is_some()
                    || !decision.blocked_on.is_empty()
                    || !decision.limitations.is_empty()
                {
                    bail!("tool_call contains fields belonging to another decision type");
                }
                let _ = arguments;
                self.seen_call_ids.insert(call_id.to_owned());
                self.pending_call = Some(PendingCall {
                    call_id: call_id.to_owned(),
                    tool_name: tool_name.to_owned(),
                });
            }
            "ask" => {
                let question = decision.question.as_deref().context("ask requires question")?;
                let why = decision
                    .why_required
                    .as_deref()
                    .context("ask requires why_required")?;
                require_nonempty("question", question)?;
                require_nonempty("why_required", why)?;
                if why.chars().count() > 500 {
                    bail!("why_required exceeds 500 characters");
                }
                ensure_no_tool_fields(decision)?;
                if decision.reason.is_some()
                    || decision.content.is_some()
                    || !decision.evidence_refs.is_empty()
                    || !decision.blocked_on.is_empty()
                    || !decision.limitations.is_empty()
                {
                    bail!("ask contains fields belonging to another decision type");
                }
            }
            "defer" => {
                let reason = decision.reason.as_deref().context("defer requires reason")?;
                require_nonempty("defer reason", reason)?;
                if decision.blocked_on.is_empty()
                    || decision.blocked_on.iter().any(|item| item.trim().is_empty())
                {
                    bail!("defer requires at least one non-empty blocker");
                }
                ensure_no_tool_fields(decision)?;
                if decision.question.is_some()
                    || decision.why_required.is_some()
                    || decision.content.is_some()
                    || !decision.evidence_refs.is_empty()
                    || !decision.limitations.is_empty()
                {
                    bail!("defer contains fields belonging to another decision type");
                }
            }
            "final_answer" => {
                let content = decision.content.as_deref().context("final_answer requires content")?;
                require_nonempty("final_answer content", content)?;
                ensure_no_tool_fields(decision)?;
                if decision.question.is_some()
                    || decision.why_required.is_some()
                    || decision.reason.is_some()
                    || !decision.blocked_on.is_empty()
                {
                    bail!("final_answer contains fields belonging to another decision type");
                }
            }
            _ => unreachable!("action was checked against ALLOWED_ACTIONS"),
        }
        self.seen_decision_ids.insert(decision.decision_id.clone());
        Ok(())
    }

    pub fn apply_observation(
        &mut self,
        request: &AgentTurnRequest,
        observation: &ToolObservation,
    ) -> Result<()> {
        request.validate()?;
        if observation.protocol_version != PROTOCOL_VERSION {
            bail!("unsupported tool-observation protocol version");
        }
        if observation.run_id != request.run_id || observation.turn_id != request.turn_id {
            bail!("observation run_id or turn_id does not match the active request");
        }
        match observation.kind.as_str() {
            "tool_observation" => {
                if self.unresolved_unknown.is_some() {
                    bail!("only execution_reconciled may follow an unresolved unknown outcome");
                }
                if observation.reconciled_status.is_some()
                    || observation.state_digest.is_some()
                    || observation.observer.is_some()
                {
                    bail!("tool_observation contains reconciliation-only fields");
                }
                let pending = self
                    .pending_call
                    .as_ref()
                    .context("tool observation has no pending tool call")?;
                if observation.call_id != pending.call_id {
                    bail!("tool observation call_id does not match the pending call");
                }
                if observation.tool_name.as_deref() != Some(pending.tool_name.as_str()) {
                    bail!("tool observation tool_name does not match the pending call");
                }
                let status = observation.status.as_deref().context("tool observation requires status")?;
                require_member("tool observation status", status, OBSERVATION_STATUSES)?;
                let digest = observation
                    .observation_digest
                    .as_deref()
                    .context("tool observation requires observation_digest")?;
                require_sha256("observation_digest", digest)?;
                if let Some(precondition) = observation.precondition_digest.as_deref() {
                    require_sha256("precondition_digest", precondition)?;
                }
                if let Some(postcondition) = observation.postcondition_digest.as_deref() {
                    require_sha256("postcondition_digest", postcondition)?;
                }
                let external_side_effect = observation.external_side_effect.unwrap_or(false);
                if external_side_effect && status != "unknown" && observation.postcondition_digest.is_none() {
                    bail!("observed external side effect requires postcondition_digest");
                }
                if status == "unknown" && observation.postcondition_digest.is_some() {
                    bail!("unknown outcome must not claim a postcondition digest");
                }
                if observation.external_side_effect.is_none() || observation.reversible.is_none() {
                    bail!("tool_observation requires external_side_effect and reversible flags");
                }
                let pending = self.pending_call.take().expect("checked above");
                if status == "unknown" {
                    self.unresolved_unknown = Some(pending);
                }
            }
            "execution_reconciled" => {
                let pending = self
                    .unresolved_unknown
                    .as_ref()
                    .context("execution_reconciled has no unresolved unknown call")?;
                if observation.call_id != pending.call_id {
                    bail!("reconciliation call_id does not match the unresolved call");
                }
                if observation.tool_name.is_some()
                    || observation.status.is_some()
                    || observation.content.is_some()
                    || observation.structured_result.is_some()
                    || observation.error.is_some()
                    || observation.observation_digest.is_some()
                    || observation.precondition_digest.is_some()
                    || observation.postcondition_digest.is_some()
                    || observation.external_side_effect.is_some()
                    || observation.reversible.is_some()
                {
                    bail!("execution_reconciled contains tool-observation-only fields");
                }
                let status = observation
                    .reconciled_status
                    .as_deref()
                    .context("execution_reconciled requires reconciled_status")?;
                require_member("reconciled_status", status, RECONCILED_STATUSES)?;
                let digest = observation
                    .state_digest
                    .as_deref()
                    .context("execution_reconciled requires state_digest")?;
                require_sha256("state_digest", digest)?;
                let observer = observation
                    .observer
                    .as_deref()
                    .context("execution_reconciled requires observer")?;
                require_member("observer", observer, OBSERVERS)?;
                self.unresolved_unknown = None;
            }
            other => bail!("unsupported observation kind {other:?}"),
        }
        Ok(())
    }
}

fn ensure_no_tool_fields(decision: &AgentDecision) -> Result<()> {
    if decision.call_id.is_some()
        || decision.tool_name.is_some()
        || decision.arguments.is_some()
        || decision.idempotency_key.is_some()
        || decision.purpose.is_some()
    {
        bail!("non-tool decision contains tool-call fields");
    }
    Ok(())
}

fn require_nonempty(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{label} must not be empty");
    }
    Ok(())
}

fn require_member(label: &str, value: &str, allowed: &[&str]) -> Result<()> {
    if !allowed.contains(&value) {
        bail!("{label} has unsupported value {value:?}");
    }
    Ok(())
}

fn require_sha256(label: &str, digest: &str) -> Result<()> {
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} must be a 64-character SHA-256 digest");
    }
    Ok(())
}
