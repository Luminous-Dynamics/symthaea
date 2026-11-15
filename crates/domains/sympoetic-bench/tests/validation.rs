use sympoetic_bench::{
    Scorecard, Trace,
    metrics::{
        collaboration_gap, handoff_recovery_rate, partner_order_sensitivity, synergy_delta,
        summarize_interventions, InterventionRecord,
    },
    validate::{validate_corpus, validate_scorecard, validate_trace},
};
use sympoetic_bench::model::{
    ExecutionInfo, HardGates, MetricEvidence, ProvenanceInfo, Scores, SystemInfo,
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
        intervention_action: None,
    }
}

fn fixture_metric_evidence(metric_name: &str, value: f64) -> MetricEvidence {
    let scorer_kind = match metric_name {
        "participant_agency_mean"
        | "agency_delta_sd_vs_reference" => "participant_self_report",
        "task_quality"
        | "quality_matched_diversity_ratio"
        | "expert_rater_agreement"
        | "intervention_appropriateness"
        | "participant_capability_gain_delta"
        | "delayed_retention_gain_delta"
        | "transfer_gain_delta" => "blinded_human",
        "human_planning_rate"
        | "human_first_attempt_rate"
        | "human_monitoring_rate"
        | "human_reflection_rate"
        | "ai_core_step_supply_rate"
        | "ai_core_step_substitution_rate"
        | "human_verification_rate" => "independent_observer",
        _ => "deterministic",
    };
    let sample_unit = if metric_name.starts_with("participant_")
        || metric_name.starts_with("human_")
    {
        "participant"
    } else if metric_name.starts_with("agent_") {
        "claim"
    } else if metric_name.contains("intervention") || metric_name.contains("regret") {
        "decision"
    } else if metric_name.contains("handoff") || metric_name.contains("repair") {
        "event"
    } else if metric_name.contains("synergy") || metric_name.contains("collaboration") {
        "task"
    } else if metric_name == "expert_rater_agreement" {
        "rater"
    } else {
        "run"
    };
    MetricEvidence {
        metric_name: metric_name.to_owned(),
        value,
        scorer_kind: scorer_kind.to_owned(),
        sample_unit: sample_unit.to_owned(),
        sample_size: 10,
        measurement_method: "fixture-method-v1".to_owned(),
        scorer_revision: "git:scorer-fixture".to_owned(),
        evidence_refs: vec![format!("fixture:evidence:{metric_name}")],
        numerator: None,
        denominator: None,
        ci_lower: None,
        ci_upper: None,
    }
}

fn sync_metric_evidence(scorecard: &mut Scorecard, metric_name: &str, value: f64) {
    let receipt = scorecard
        .metric_evidence
        .iter_mut()
        .find(|receipt| receipt.metric_name == metric_name)
        .unwrap();
    receipt.value = value;
    receipt.numerator = None;
    receipt.denominator = None;
    receipt.ci_lower = None;
    receipt.ci_upper = None;
}

fn base_scorecard() -> Scorecard {
    let scores = Scores {
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
        intervention_appropriateness: Some(0.9),
        unnecessary_intervention_rate: Some(0.05),
        missed_critical_intervention_rate: Some(0.02),
        intervention_regret: Some(0.1),
        participant_capability_gain_delta: Some(0.2),
        delayed_retention_gain_delta: Some(0.15),
        transfer_gain_delta: Some(0.1),
        agent_confidence_brier_score: Some(0.08),
        agent_expected_calibration_error: Some(0.06),
        agent_metacognitive_sensitivity_auc: Some(0.91),
        agent_selective_accuracy_at_80pct_coverage: Some(0.95),
        participant_confidence_brier_score: Some(0.11),
        participant_expected_calibration_error: Some(0.08),
        participant_metacognitive_sensitivity_auc: Some(0.82),
        participant_selective_accuracy_at_80pct_coverage: Some(0.88),
        human_planning_rate: Some(0.8),
        human_first_attempt_rate: Some(0.75),
        human_monitoring_rate: Some(0.7),
        human_reflection_rate: Some(0.65),
        ai_core_step_supply_rate: Some(0.2),
        ai_core_step_substitution_rate: Some(0.1),
        human_verification_rate: Some(0.9),
    };
    let metric_evidence = scores
        .entries()
        .into_iter()
        .filter_map(|(metric_name, value)| {
            value.map(|value| fixture_metric_evidence(metric_name, value))
        })
        .collect();

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
        scores,
        metric_evidence,
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
fn intervention_decisions_are_explicit_and_traceable() {
    let mut decision = base_event("decision-ask", 0, "intervention_decision");
    decision.actor = "agent".to_owned();
    decision.intervention_action = Some("ask".to_owned());
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "trace-intervention".to_owned(),
        scenario_id: "SP-INT-001".to_owned(),
        trace_digest: None,
        events: vec![decision],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_ok());
}

#[test]
fn intervention_decision_requires_a_valid_action_and_agent_actor() {
    let missing_action = base_event("decision-missing", 0, "intervention_decision");
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "trace-intervention-missing".to_owned(),
        scenario_id: "SP-INT-001".to_owned(),
        trace_digest: None,
        events: vec![missing_action],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_err());

    let mut invalid_action = base_event("decision-invalid", 0, "intervention_decision");
    invalid_action.intervention_action = Some("always_talk".to_owned());
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "trace-intervention-invalid".to_owned(),
        scenario_id: "SP-INT-001".to_owned(),
        trace_digest: None,
        events: vec![invalid_action],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_err());

    let mut wrong_actor = base_event("decision-wrong-actor", 0, "intervention_decision");
    wrong_actor.intervention_action = Some("step_back".to_owned());
    wrong_actor.actor = "human".to_owned();
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "trace-intervention-wrong-actor".to_owned(),
        scenario_id: "SP-INT-001".to_owned(),
        trace_digest: None,
        events: vec![wrong_actor],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_err());
}

#[test]
fn intervention_adjudications_produce_burden_and_missed_warning_rates() {
    let records = [
        InterventionRecord {
            appropriate: true,
            unnecessary: false,
            critical_opportunity: true,
            critical_missed: false,
            counterfactual_regret: Some(0.1),
        },
        InterventionRecord {
            appropriate: false,
            unnecessary: true,
            critical_opportunity: false,
            critical_missed: false,
            counterfactual_regret: Some(0.3),
        },
        InterventionRecord {
            appropriate: false,
            unnecessary: false,
            critical_opportunity: true,
            critical_missed: true,
            counterfactual_regret: None,
        },
    ];
    let summary = summarize_interventions(&records).unwrap();
    assert_eq!(summary.decision_count, 3);
    assert_eq!(summary.critical_opportunity_count, 2);
    assert!((summary.unnecessary_intervention_rate.unwrap() - 1.0 / 3.0).abs() < 1e-12);
    assert!((summary.missed_critical_intervention_rate.unwrap() - 0.5).abs() < 1e-12);
}

#[test]
fn committed_seed_corpus_validates() {
    let (scenarios, receipt) = validate_corpus(SEED_CORPUS.as_bytes()).unwrap();
    assert_eq!(scenarios.len(), 17);
    assert_eq!(receipt.scenario_count, 17);
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
fn unknown_execution_blocks_non_tool_external_side_effects_too() {
    let mut unknown = base_event("unknown-tool-result", 0, "tool_result");
    unknown.actor = "tool".to_owned();
    unknown.external_side_effect = true;
    unknown.tool_execution = Some(ToolExecution {
        tool_name: "external_mutation".to_owned(),
        status: "unknown".to_owned(),
        idempotency_key: Some("idempotency-unknown".to_owned()),
        precondition_digest: Some(digest('e')),
        postcondition_digest: None,
    });
    let mut artifact_change = base_event("external-artifact-write", 1, "artifact_revised");
    artifact_change.external_side_effect = true;
    artifact_change.parent_event_ids.push("unknown-tool-result".to_owned());

    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "trace-unknown-side-effect".to_owned(),
        scenario_id: "SP-ENG-001".to_owned(),
        trace_digest: None,
        events: vec![unknown, artifact_change],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_err());
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
    sync_metric_evidence(&mut scorecard, "attribution_precision", 0.8);
    sync_metric_evidence(&mut scorecard, "attribution_recall", 0.8);
    sync_metric_evidence(&mut scorecard, "attribution_f1", 0.5);
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
fn scorecard_accepts_negative_capability_gain_deltas_and_rejects_out_of_range_values() {
    let mut scorecard = base_scorecard();
    scorecard.scores.participant_capability_gain_delta = Some(-0.2);
    scorecard.scores.delayed_retention_gain_delta = Some(-0.1);
    scorecard.scores.transfer_gain_delta = Some(0.05);
    sync_metric_evidence(&mut scorecard, "participant_capability_gain_delta", -0.2);
    sync_metric_evidence(&mut scorecard, "delayed_retention_gain_delta", -0.1);
    sync_metric_evidence(&mut scorecard, "transfer_gain_delta", 0.05);
    validate_scorecard(&scorecard).unwrap();

    scorecard.scores.participant_capability_gain_delta = Some(2.1);
    sync_metric_evidence(&mut scorecard, "participant_capability_gain_delta", 2.1);
    assert!(validate_scorecard(&scorecard).is_err());
}

#[test]
fn scorecard_validates_confidence_and_cognitive_process_measures() {
    let mut scorecard = base_scorecard();
    scorecard.scores.agent_metacognitive_sensitivity_auc = Some(0.94);
    scorecard.scores.participant_confidence_brier_score = Some(0.12);
    scorecard.scores.ai_core_step_substitution_rate = Some(0.18);
    sync_metric_evidence(&mut scorecard, "agent_metacognitive_sensitivity_auc", 0.94);
    sync_metric_evidence(&mut scorecard, "participant_confidence_brier_score", 0.12);
    sync_metric_evidence(&mut scorecard, "ai_core_step_substitution_rate", 0.18);
    validate_scorecard(&scorecard).unwrap();

    scorecard.scores.agent_expected_calibration_error = Some(1.1);
    sync_metric_evidence(&mut scorecard, "agent_expected_calibration_error", 1.1);
    assert!(validate_scorecard(&scorecard).is_err());
}

#[test]
fn confidence_assessment_requires_pre_feedback_confidence_and_valid_actor() {
    let mut confidence = base_event("confidence", 0, "confidence_assessment");
    confidence.actor = "human".to_owned();
    confidence.confidence = Some(0.7);
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "confidence-valid".to_owned(),
        scenario_id: "SP-EPI-001".to_owned(),
        trace_digest: None,
        events: vec![confidence.clone()],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_ok());

    confidence.confidence = None;
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "confidence-missing".to_owned(),
        scenario_id: "SP-EPI-001".to_owned(),
        trace_digest: None,
        events: vec![confidence],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_err());
}

#[test]
fn epistemic_source_and_revision_events_require_provenance_links() {
    let source = base_event("source", 0, "assistant_proposal");
    let mut attribution = base_event("attribution", 1, "source_attribution");
    attribution.attribution.push(sympoetic_bench::model::Attribution {
        contributor: "external_source".to_owned(),
        contribution_type: "fact".to_owned(),
        artifact_ref: "source:maintenance-log-1".to_owned(),
        attribution_confidence: 1.0,
        permission_ref: None,
    });
    let mut revision = base_event("revision", 2, "belief_revision");
    revision.parent_event_ids.push("source".to_owned());
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "epistemic-linked".to_owned(),
        scenario_id: "SP-EPI-001".to_owned(),
        trace_digest: None,
        events: vec![source.clone(), attribution, revision],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_ok());

    let mut missing_link = base_event("revision-unlinked", 1, "belief_revision");
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "epistemic-unlinked".to_owned(),
        scenario_id: "SP-EPI-001".to_owned(),
        trace_digest: None,
        events: vec![source, missing_link.clone()],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_err());

    missing_link.event_type = "source_attribution".to_owned();
    missing_link.parent_event_ids.clear();
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "epistemic-no-attribution".to_owned(),
        scenario_id: "SP-EPI-001".to_owned(),
        trace_digest: None,
        events: vec![missing_link],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_err());
}

#[test]
fn contradiction_resolution_requires_two_prior_claims() {
    let first = base_event("claim-a", 0, "assistant_proposal");
    let second = base_event("claim-b", 1, "assistant_challenge");
    let mut resolution = base_event("resolution", 2, "contradiction_resolution");
    resolution.actor = "agent".to_owned();
    resolution.parent_event_ids = vec!["claim-a".to_owned(), "claim-b".to_owned()];
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "contradiction-two-claims".to_owned(),
        scenario_id: "SP-EPI-001".to_owned(),
        trace_digest: None,
        events: vec![first.clone(), second.clone(), resolution],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_ok());

    let mut invalid = base_event("resolution-one-parent", 1, "contradiction_resolution");
    invalid.parent_event_ids.push("claim-a".to_owned());
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "contradiction-one-claim".to_owned(),
        scenario_id: "SP-EPI-001".to_owned(),
        trace_digest: None,
        events: vec![first, invalid],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_err());
}

#[test]
fn cognitive_process_trace_events_enforce_human_and_agent_actors() {
    let mut planning = base_event("planning", 0, "human_planning");
    planning.actor = "human".to_owned();
    let mut ai_step = base_event("ai-step", 1, "ai_core_step_supplied");
    ai_step.actor = "agent".to_owned();
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "cognitive-actors-valid".to_owned(),
        scenario_id: "SP-EPI-001".to_owned(),
        trace_digest: None,
        events: vec![planning.clone(), ai_step],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_ok());

    planning.actor = "agent".to_owned();
    let trace = Trace {
        schema_version: "sympoetic-trace/0.1".to_owned(),
        run_id: "cognitive-actors-invalid".to_owned(),
        scenario_id: "SP-EPI-001".to_owned(),
        trace_digest: None,
        events: vec![planning],
    };
    let bytes = serde_json::to_vec(&trace).unwrap();
    assert!(validate_trace(&trace, &bytes).is_err());
}

#[test]
fn scorecard_requires_one_evidence_receipt_for_every_reported_metric() {
    let mut scorecard = base_scorecard();
    scorecard.metric_evidence.pop();
    assert!(validate_scorecard(&scorecard).is_err());

    let mut scorecard = base_scorecard();
    let duplicate_receipt = scorecard.metric_evidence[0].clone();
    scorecard.metric_evidence.push(duplicate_receipt);
    assert!(validate_scorecard(&scorecard).is_err());
}

#[test]
fn scorecard_metric_receipt_value_must_match_the_reported_score() {
    let mut scorecard = base_scorecard();
    scorecard.metric_evidence[0].value += 0.1;
    assert!(validate_scorecard(&scorecard).is_err());
}

#[test]
fn scorecard_ratio_receipt_must_match_its_numerator_and_denominator() {
    let mut scorecard = base_scorecard();
    scorecard.scores.constraint_satisfaction = Some(0.8);
    {
        let receipt = scorecard
            .metric_evidence
            .iter_mut()
            .find(|receipt| receipt.metric_name == "constraint_satisfaction")
            .unwrap();
        receipt.value = 0.8;
        receipt.numerator = Some(8.0);
        receipt.denominator = Some(10.0);
    }
    validate_scorecard(&scorecard).unwrap();

    {
        let receipt = scorecard
            .metric_evidence
            .iter_mut()
            .find(|receipt| receipt.metric_name == "constraint_satisfaction")
            .unwrap();
        receipt.denominator = Some(9.0);
    }
    assert!(validate_scorecard(&scorecard).is_err());
}

#[test]
fn scorecard_receipts_require_sources_and_complete_uncertainty_intervals() {
    let mut scorecard = base_scorecard();
    scorecard.metric_evidence[0].evidence_refs.clear();
    assert!(validate_scorecard(&scorecard).is_err());

    let mut scorecard = base_scorecard();
    scorecard.metric_evidence[0].ci_lower = Some(0.9);
    assert!(validate_scorecard(&scorecard).is_err());

    let mut scorecard = base_scorecard();
    scorecard.metric_evidence[0].ci_lower = Some(0.7);
    scorecard.metric_evidence[0].ci_upper = Some(0.9);
    validate_scorecard(&scorecard).unwrap();
}

#[test]
fn scorecard_rejects_unrecognized_and_auxiliary_only_scorer_kinds() {
    let mut scorecard = base_scorecard();
    scorecard.metric_evidence[0].scorer_kind = "model_auxiliary".to_owned();
    assert!(validate_scorecard(&scorecard).is_err());

    scorecard.metric_evidence[0].scorer_kind = "unknown".to_owned();
    assert!(validate_scorecard(&scorecard).is_err());
}

#[test]
fn canonical_score_table_and_metric_evidence_receipts_have_one_to_one_coverage() {
    let scorecard = base_scorecard();
    let entries = scorecard.scores.entries();
    let populated_count = entries
        .iter()
        .filter(|(_, value)| value.is_some())
        .count();
    assert_eq!(entries.len(), 37);
    assert_eq!(populated_count, scorecard.metric_evidence.len());

    for (index, (name, _)) in entries.iter().enumerate() {
        assert!(
            entries[index + 1..]
                .iter()
                .all(|(other, _)| other != name),
            "duplicate canonical metric name: {name}"
        );
    }
    validate_scorecard(&scorecard).unwrap();
}

#[test]
fn clean_scorecard_validates() {
    validate_scorecard(&base_scorecard()).unwrap();
}
