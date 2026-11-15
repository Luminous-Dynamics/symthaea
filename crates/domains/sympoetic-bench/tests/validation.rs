use sympoetic_bench::{
    Scorecard, Trace,
    metrics::{collaboration_gap, handoff_recovery_rate, partner_order_sensitivity, synergy_delta},
    validate::{validate_corpus, validate_scorecard, validate_trace},
};
use sympoetic_bench::model::{
    ExecutionInfo, HardGates, ProvenanceInfo, Scores, SystemInfo,
    TraceEvent, ToolExecution,
};

const SEED_CORPUS: &str =
    include_str!("../../../../docs/benchmarks/sympoetic/SEED_CORPUS.v0.1.jsonl");

fn digest(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn base_event(id: &str, sequence: u64, event_type: &str) -> TraceEvent {
    TraceEvent {
        event_id: id.to_owned(),
        sequence,
        elapsed_ms: sequence * 10,
        actor: "agent".to_owned(),
        event_type: event_type.to_owned(),
        parent_event_ids: Vec::new(),
        payload_digest: digest('a'),
        payload_storage: "metadata_only".to_owned(),
        consent_scope: "session_only".to_owned(),
        artifact_digest_before: None,
        artifact_digest_after: None,
        external_side_effect: false,
        reversible: true,
        confidence: Some(1.0),
        tool_execution: None,
        attribution: Vec::new(),
        annotation: None,
        reconciled_status: None,
        reconciliation_state_digest: None,
    }
}

fn base_scorecard() -> Scorecard {
    Scorecard {
        schema_version: "sympoetic-scorecard/0.1".to_owned(),
        benchmark_version: "0.1".to_owned(),
        run_id: "test-run-001".to_owned(),
        scenario_id: "SP-CRE-001".to_owned(),
        split: "development".to_owned(),
        condition: "reciprocal_co_creation".to_owned(),
        system: SystemInfo {
            system_name: "fixture".to_owned(),
            model_snapshot: "fixture-model@sha256:abc".to_owned(),
            adapter_revision: "git:fixture".to_owned(),
            prompt_digest: None,
            memory_policy: "none".to_owned(),
            tools_enabled: Vec::new(),
        },
        execution: ExecutionInfo {
            seed: 42,
            status: "complete".to_owned(),
            latency_seconds: 1.0,
            human_time_minutes: 10.0,
            interaction_turns: 8,
            input_tokens: Some(100),
            output_tokens: Some(120),
            cost_usd: Some(0.01),
        },
        provenance: ProvenanceInfo {
            scenario_sha256: digest('b'),
            trace_sha256: digest('c'),
            artifact_sha256: digest('d'),
            scorer_revision: "git:scorer-fixture".to_owned(),
            environment_digest: None,
            consented_trace_uri: None,
        },
        scores: Scores {
            task_quality: Some(0.8),
            constraint_satisfaction: Some(1.0),
            participant_agency_mean: Some(6.0),
            agency_delta_sd_vs_reference: Some(-0.05),
            common_ground_accuracy: Some(0.95),
            repair_success_rate: Some(0.9),
            attribution_precision: Some(0.9),
            attribution_recall: Some(0.9),
            attribution_f1: Some(0.9),
            quality_matched_diversity_ratio: Some(0.95),
            expert_rater_agreement: Some(0.85),
            synergy_delta: Some(0.1),
            collaboration_gap: Some(-0.1),
            partner_order_sensitivity: Some(0.02),
            handoff_recovery_rate: Some(0.9),
        },
        hard_gates: HardGates {
            pass: true,
            authority_violation_count: 0,
            consent_violation_count: 0,
            fabricated_attribution_count: 0,
            hidden_side_effect_count: 0,
            unresolved_unknown_outcome_count: 0,
            reconciliation_bypass_count: 0,
            failure_event_ids: Vec::new(),
        },
        notes: None,
    }
}

#[test]
fn committed_seed_corpus_validates() {
    let (scenarios, receipt) = validate_corpus(SEED_CORPUS.as_bytes()).unwrap();
    assert_eq!(scenarios.len(), 13);
    assert_eq!(receipt.scenario_count, 13);
    assert_eq!(receipt.status, "pass");
    assert_eq!(receipt.scenario_ids[0], "SP-CRE-001");
}

#[test]
fn duplicate_scenario_ids_are_rejected() {
    let first = SEED_CORPUS.lines().next().unwrap();
    let duplicate = format!("{first}\n{first}\n");
    assert!(validate_corpus(duplicate.as_bytes()).is_err());
}

#[test]
fn valid_trace_with_causal_parent_is_accepted() {
    let mut first = base_event("intent", 0, "user_intent");
    first.actor = "human".to_owned();
    let mut second = base_event("proposal", 1, "assistant_proposal");
    second.parent_event_ids.push("intent".to_owned());
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "trace-valid".to_owned(),
        scenario_id: "SP-CRE-001".to_owned(),
        trace_digest: None,
        events: vec![first, second],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    let checked = validate_trace(&trace, &bytes).unwrap();
    assert_eq!(checked.event_count, 2);
    assert_eq!(checked.unresolved_unknown_outcome_count, 0);
}

#[test]
fn forward_or_self_parent_is_rejected() {
    let mut event = base_event("event-1", 0, "assistant_proposal");
    event.parent_event_ids.push("event-1".to_owned());
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "trace-self-parent".to_owned(),
        scenario_id: "SP-CRE-001".to_owned(),
        trace_digest: None,
        events: vec![event],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_err());
}

#[test]
fn unknown_tool_outcome_blocks_next_tool_invocation_until_reconciled() {
    let mut unknown = base_event("tool-result-unknown", 0, "tool_result");
    unknown.actor = "tool".to_owned();
    unknown.tool_execution = Some(ToolExecution {
        tool_name: "fixture_mutation".to_owned(),
        status: "unknown".to_owned(),
        idempotency_key: Some("idem-1".to_owned()),
        precondition_digest: Some(digest('e')),
        postcondition_digest: None,
    });

    let mut invocation = base_event("second-mutation", 1, "tool_invocation");
    invocation.parent_event_ids.push("tool-result-unknown".to_owned());

    let blocked = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "trace-unknown-blocked".to_owned(),
        scenario_id: "SP-ENG-001".to_owned(),
        trace_digest: None,
        events: vec![unknown.clone(), invocation.clone()],
    };
    let bytes = serde_json::to_vec(&blocked).unwrap();
    assert!(validate_trace(&blocked, &bytes).is_err());

    let mut reconciled = base_event("reconciled", 1, "execution_reconciled");
    reconciled.actor = "environment".to_owned();
    reconciled.parent_event_ids.push("tool-result-unknown".to_owned());
    reconciled.reconciled_status = Some("not_executed".to_owned());
    reconciled.reconciliation_state_digest = Some(digest('f'));
    invocation.sequence = 2;
    invocation.elapsed_ms = 20;
    invocation.parent_event_ids.push("reconciled".to_owned());

    let valid = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "trace-reconciled".to_owned(),
        scenario_id: "SP-ENG-001".to_owned(),
        trace_digest: None,
        events: vec![unknown, reconciled, invocation],
    };
    let bytes = serde_json::to_vec(&valid).unwrap();
    let checked = validate_trace(&valid, &bytes).unwrap();
    assert_eq!(checked.unresolved_unknown_outcome_count, 0);
    assert_eq!(checked.reconciliation_count, 1);
}

#[test]
fn consented_content_cannot_be_stored_after_revocation() {
    let mut event = base_event("stored-content", 0, "assistant_proposal");
    event.payload_storage = "encrypted_consented_reference".to_owned();
    event.consent_scope = "revoked".to_owned();
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "trace-revoked-content".to_owned(),
        scenario_id: "SP-CNS-001".to_owned(),
        trace_digest: None,
        events: vec![event],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_err());
}

#[test]
fn only_human_participant_can_change_consent() {
    let event = base_event("forged-consent", 0, "permission_granted");
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "trace-forged-consent".to_owned(),
        scenario_id: "SP-CNS-001".to_owned(),
        trace_digest: None,
        events: vec![event],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_err());
}

#[test]
fn external_side_effect_requires_observed_postcondition() {
    let mut event = base_event("mutating-tool-result", 0, "tool_result");
    event.actor = "tool".to_owned();
    event.external_side_effect = true;
    event.tool_execution = Some(ToolExecution {
        tool_name: "fixture_mutation".to_owned(),
        status: "success".to_owned(),
        idempotency_key: Some("idem-2".to_owned()),
        precondition_digest: Some(digest('a')),
        postcondition_digest: None,
    });
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "trace-missing-postcondition".to_owned(),
        scenario_id: "SP-ENG-001".to_owned(),
        trace_digest: None,
        events: vec![event],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_err());
}

#[test]
fn scorecard_cannot_average_away_a_hard_gate_failure() {
    let mut scorecard = base_scorecard();
    scorecard.hard_gates.consent_violation_count = 1;
    scorecard.hard_gates.pass = true;
    scorecard.hard_gates.failure_event_ids.push("event-consent-1".to_owned());
    assert!(validate_scorecard(&scorecard).is_err());

    scorecard.hard_gates.pass = false;
    assert!(validate_scorecard(&scorecard).is_ok());
}

#[test]
fn attribution_f1_must_match_precision_and_recall() {
    let mut scorecard = base_scorecard();
    scorecard.scores.attribution_precision = Some(0.8);
    scorecard.scores.attribution_recall = Some(0.8);
    scorecard.scores.attribution_f1 = Some(0.5);
    assert!(validate_scorecard(&scorecard).is_err());
}

#[test]
fn synergy_and_collaboration_gap_are_sign_consistent() {
    let synergy = synergy_delta(0.7, 0.8, 0.6).unwrap();
    let gap = collaboration_gap(0.8, 0.6).unwrap();
    assert!((synergy + gap).abs() < 1e-12);
    assert!((partner_order_sensitivity(0.7, 0.6).unwrap() - 0.1).abs() < 1e-12);
    assert_eq!(handoff_recovery_rate(9, 10).unwrap(), Some(0.9));
}

#[test]
fn clean_scorecard_validates() {
    validate_scorecard(&base_scorecard()).unwrap();
}
