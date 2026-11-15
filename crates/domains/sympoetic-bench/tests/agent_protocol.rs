use serde_json::{Map, json};
use sympoetic_bench::agent_protocol::{
    AgentDecision, AgentProtocolGuard, AgentTask, AgentTurnRequest, Authority,
    AvailableTool, DecisionBudget, HistoryEvent, ToolObservation,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn request() -> AgentTurnRequest {
    AgentTurnRequest {
        protocol_version: "symthaea-agent-bridge/0.1".to_owned(),
        kind: "agent_turn".to_owned(),
        turn_id: "turn-001".to_owned(),
        run_id: "run-001".to_owned(),
        principal_id: "agent-principal".to_owned(),
        task: AgentTask {
            task_id: "task-001".to_owned(),
            environment_id: "env-001".to_owned(),
            role: "operator".to_owned(),
            goal: "Inspect the service state".to_owned(),
            brief: "Inspect the named service and report its state without making changes.".to_owned(),
            domain: Some("hpc_operations".to_owned()),
            qcat: Some("read_only".to_owned()),
        },
        constraints: vec!["Do not mutate the environment".to_owned()],
        available_tools: vec![AvailableTool {
            name: "read_service_status".to_owned(),
            description: "Read status for one service".to_owned(),
            input_schema: json!({
                "type":"object",
                "properties":{"service":{"type":"string"}},
                "required":["service"],
                "additionalProperties":false
            }),
        }],
        history: vec![HistoryEvent {
            event_id: "history-001".to_owned(),
            actor: "human".to_owned(),
            kind: "user_intent".to_owned(),
            content: "Please inspect sshd".to_owned(),
            trust_class: "trusted_instruction".to_owned(),
            evidence_refs: vec![],
        }],
        authority: Authority {
            role_id: "read-only-operator".to_owned(),
            allowed_actions: vec![
                "tool_call".to_owned(),
                "ask".to_owned(),
                "defer".to_owned(),
                "final_answer".to_owned(),
            ],
            policy_digest: digest('a'),
        },
        budget: DecisionBudget {
            tool_calls_remaining: 2,
            max_output_tokens: 1024,
            deadline_ms_remaining: 30_000,
        },
    }
}

fn tool_decision(call_id: &str, tool_name: &str) -> AgentDecision {
    let mut arguments = Map::new();
    arguments.insert("service".to_owned(), json!("sshd"));
    AgentDecision {
        protocol_version: "symthaea-agent-bridge/0.1".to_owned(),
        kind: "agent_decision".to_owned(),
        run_id: "run-001".to_owned(),
        turn_id: "turn-001".to_owned(),
        decision_id: format!("decision-{call_id}"),
        action: "tool_call".to_owned(),
        call_id: Some(call_id.to_owned()),
        tool_name: Some(tool_name.to_owned()),
        arguments: Some(arguments),
        idempotency_key: Some(format!("idempotency-{call_id}")),
        purpose: Some("Read the service's current state".to_owned()),
        evidence_refs: vec![],
        question: None,
        why_required: None,
        reason: None,
        blocked_on: vec![],
        content: None,
        limitations: vec![],
    }
}

fn final_decision(id: &str) -> AgentDecision {
    AgentDecision {
        protocol_version: "symthaea-agent-bridge/0.1".to_owned(),
        kind: "agent_decision".to_owned(),
        run_id: "run-001".to_owned(),
        turn_id: "turn-001".to_owned(),
        decision_id: id.to_owned(),
        action: "final_answer".to_owned(),
        call_id: None,
        tool_name: None,
        arguments: None,
        idempotency_key: None,
        purpose: None,
        evidence_refs: vec!["observation:service-status".to_owned()],
        question: None,
        why_required: None,
        reason: None,
        blocked_on: vec![],
        content: Some("sshd is active".to_owned()),
        limitations: vec!["Status was observed at one instant".to_owned()],
    }
}

fn tool_observation(call_id: &str, status: &str, side_effect: bool) -> ToolObservation {
    ToolObservation {
        protocol_version: "symthaea-agent-bridge/0.1".to_owned(),
        kind: "tool_observation".to_owned(),
        run_id: "run-001".to_owned(),
        turn_id: "turn-001".to_owned(),
        call_id: call_id.to_owned(),
        tool_name: Some("read_service_status".to_owned()),
        status: Some(status.to_owned()),
        content: Some("active".to_owned()),
        structured_result: Some(Map::new()),
        error: None,
        observation_digest: Some(digest('b')),
        precondition_digest: Some(digest('c')),
        postcondition_digest: if side_effect { Some(digest('d')) } else { None },
        external_side_effect: Some(side_effect),
        reversible: Some(true),
        reconciled_status: None,
        state_digest: None,
        observer: None,
    }
}

fn reconcile(call_id: &str) -> ToolObservation {
    ToolObservation {
        protocol_version: "symthaea-agent-bridge/0.1".to_owned(),
        kind: "execution_reconciled".to_owned(),
        run_id: "run-001".to_owned(),
        turn_id: "turn-001".to_owned(),
        call_id: call_id.to_owned(),
        tool_name: None,
        status: None,
        content: None,
        structured_result: None,
        error: None,
        observation_digest: None,
        precondition_digest: None,
        postcondition_digest: None,
        external_side_effect: None,
        reversible: None,
        reconciled_status: Some("not_executed".to_owned()),
        state_digest: Some(digest('e')),
        observer: Some("environment".to_owned()),
    }
}

#[test]
fn serialized_ask_decision_omits_fields_not_allowed_by_its_schema_variant() {
    let decision = AgentDecision {
        protocol_version: "symthaea-agent-bridge/0.1".to_owned(),
        kind: "agent_decision".to_owned(),
        run_id: "run-001".to_owned(),
        turn_id: "turn-001".to_owned(),
        decision_id: "decision-ask".to_owned(),
        action: "ask".to_owned(),
        call_id: None,
        tool_name: None,
        arguments: None,
        idempotency_key: None,
        purpose: None,
        evidence_refs: Vec::new(),
        question: Some("Which maintenance window is approved?".to_owned()),
        why_required: Some("The service restart requires an approved window.".to_owned()),
        reason: None,
        blocked_on: Vec::new(),
        content: None,
        limitations: Vec::new(),
    };
    let value = serde_json::to_value(&decision).unwrap();
    let object = value.as_object().unwrap();
    assert_eq!(object.get("action").and_then(|v| v.as_str()), Some("ask"));
    assert!(object.get("evidence_refs").is_none());
    assert!(object.get("call_id").is_none());
    assert!(object.get("blocked_on").is_none());
    assert!(object.get("content").is_none());
}

#[test]
fn serialized_reconciliation_observation_omits_tool_only_fields() {
    let observation = reconcile("call-serde");
    let value = serde_json::to_value(&observation).unwrap();
    let object = value.as_object().unwrap();
    assert_eq!(
        object.get("kind").and_then(|v| v.as_str()),
        Some("execution_reconciled")
    );
    assert!(object.get("run_id").is_some());
    assert!(object.get("state_digest").is_some());
    assert!(object.get("tool_name").is_none());
    assert!(object.get("status").is_none());
    assert!(object.get("observation_digest").is_none());
}

#[test]
fn run_id_mismatch_is_rejected_even_when_turn_id_matches() {
    let req = request();
    let mut guard = AgentProtocolGuard::default();
    let mut decision = final_decision("decision-wrong-run");
    decision.run_id = "another-run".to_owned();
    assert!(guard.validate_decision(&req, &decision).is_err());
}

#[test]
fn valid_request_and_tool_decision_are_accepted() {
    let req = request();
    req.validate().unwrap();
    let mut guard = AgentProtocolGuard::default();
    guard.validate_decision(&req, &tool_decision("call-1", "read_service_status")).unwrap();
}

#[test]
fn role_filtered_tool_catalogue_is_authoritative() {
    let req = request();
    let mut guard = AgentProtocolGuard::default();
    assert!(guard.validate_decision(&req, &tool_decision("call-2", "delete_cluster")).is_err());
}

#[test]
fn action_outside_granted_authority_is_rejected() {
    let mut req = request();
    req.authority.allowed_actions = vec!["ask".to_owned()];
    let mut guard = AgentProtocolGuard::default();
    assert!(guard.validate_decision(&req, &tool_decision("call-3", "read_service_status")).is_err());
}

#[test]
fn tool_call_cannot_be_followed_by_another_decision_before_observation() {
    let req = request();
    let mut guard = AgentProtocolGuard::default();
    guard.validate_decision(&req, &tool_decision("call-4", "read_service_status")).unwrap();
    assert!(guard.validate_decision(&req, &final_decision("decision-final")).is_err());
    guard.apply_observation(&req, &tool_observation("call-4", "success", false)).unwrap();
    guard.validate_decision(&req, &final_decision("decision-final")).unwrap();
}

#[test]
fn unknown_outcome_blocks_new_decisions_until_independent_reconciliation() {
    let req = request();
    let mut guard = AgentProtocolGuard::default();
    guard.validate_decision(&req, &tool_decision("call-5", "read_service_status")).unwrap();

    let mut unknown = tool_observation("call-5", "unknown", true);
    unknown.postcondition_digest = None;
    guard.apply_observation(&req, &unknown).unwrap();
    assert!(guard.has_unresolved_execution());
    assert!(guard.validate_decision(&req, &final_decision("decision-wait")).is_err());
    assert!(guard.apply_observation(&req, &tool_observation("call-5", "success", false)).is_err());

    guard.apply_observation(&req, &reconcile("call-5")).unwrap();
    assert!(!guard.has_unresolved_execution());
    guard.validate_decision(&req, &final_decision("decision-after-reconcile")).unwrap();
}

#[test]
fn reconciliation_must_match_the_unknown_call() {
    let req = request();
    let mut guard = AgentProtocolGuard::default();
    guard.validate_decision(&req, &tool_decision("call-6", "read_service_status")).unwrap();
    guard.apply_observation(&req, &tool_observation("call-6", "unknown", false)).unwrap();
    assert!(guard.apply_observation(&req, &reconcile("other-call")).is_err());
    assert!(guard.has_unresolved_execution());
}

#[test]
fn successful_external_side_effect_requires_a_postcondition_digest() {
    let req = request();
    let mut guard = AgentProtocolGuard::default();
    guard.validate_decision(&req, &tool_decision("call-7", "read_service_status")).unwrap();
    let mut observation = tool_observation("call-7", "success", true);
    observation.postcondition_digest = None;
    assert!(guard.apply_observation(&req, &observation).is_err());
}

#[test]
fn decision_run_and_turn_ids_must_match() {
    let req = request();
    let mut guard = AgentProtocolGuard::default();
    let mut decision = final_decision("decision-wrong-turn");
    decision.turn_id = "another-turn".to_owned();
    assert!(guard.validate_decision(&req, &decision).is_err());
}

#[test]
fn duplicate_decision_ids_are_rejected_after_a_valid_decision() {
    let req = request();
    let mut guard = AgentProtocolGuard::default();
    let decision = final_decision("decision-duplicate");
    guard.validate_decision(&req, &decision).unwrap();
    assert!(guard.validate_decision(&req, &decision).is_err());
}

#[test]
fn invalid_decision_does_not_consume_its_decision_id_or_call_id() {
    let req = request();
    let mut guard = AgentProtocolGuard::default();
    let mut decision = tool_decision("call-retry", "not-allowed");
    decision.decision_id = "decision-retry".to_owned();
    assert!(guard.validate_decision(&req, &decision).is_err());

    decision.tool_name = Some("read_service_status".to_owned());
    guard.validate_decision(&req, &decision).unwrap();
}
