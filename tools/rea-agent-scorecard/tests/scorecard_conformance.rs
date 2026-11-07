#[path = "../../../src/rea_compat.rs"]
mod rea_compat;

use rea_agent_scorecard::{
    AgentSubmission, OccurrenceExpectation, OracleManifest, OracleScope, OracleTask, PairBinding,
    PairKind, PairRole, SCORECARD_SCHEMA, SUBMISSION_SCHEMA, Scorecard, evaluate_scorecard,
};
use rea_compat::claims::{ClaimPredicate, ClaimRequest, ClaimScope, evaluate_claim};
use serde_json::{Value, json};

const FIXTURE: &[u8] = include_bytes!(
    "../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
);

#[derive(Clone)]
struct Case {
    task: OracleTask,
    submission: AgentSubmission,
}

#[allow(clippy::too_many_arguments)]
fn make_case(
    task_id: &str,
    subject_sha256: &str,
    normalized_result: Value,
    answer: Option<Value>,
    claim_subject_override: Option<&str>,
    claim_operation_override: Option<&str>,
    claim_asserted_override: Option<Value>,
    force_reported_supported: bool,
    include_replay: bool,
    abstained: bool,
    occurrence: Option<OccurrenceExpectation>,
    pair: Option<PairBinding>,
) -> Case {
    let admitted = rea_compat::verify_evidence_json(FIXTURE).unwrap();
    let mut evidence_value = admitted.value;
    evidence_value["subject"]["digest"]["sha256"] = json!(subject_sha256);
    evidence_value["normalized_result"] = normalized_result.clone();
    evidence_value["provider"]["id"] = json!(format!("scorecard-provider-{task_id}"));
    evidence_value["provider"]["name"] = json!(format!("Scorecard Provider {task_id}"));
    evidence_value["analysis_profile"]["provider"] = evidence_value["provider"].clone();
    evidence_value["analysis_profile"]["digest"] =
        json!(rea_compat::analysis_profile_digest(&evidence_value["analysis_profile"]).unwrap());
    evidence_value["evidence_id"] =
        json!(rea_compat::compute_evidence_id(&evidence_value).unwrap());
    let evidence_bytes = serde_json::to_vec(&evidence_value).unwrap();
    let evidence = rea_compat::verify_evidence_json(&evidence_bytes).unwrap();

    let operation = evidence.value["operation"].as_str().unwrap().to_owned();
    let source_predicate_type = evidence.value["predicate_type"].as_str().unwrap().to_owned();
    let environment = evidence.value.get("environment").unwrap_or(&Value::Null);
    let platform = environment
        .get("platform")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let environment_id = environment
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned);

    let expected_scope = OracleScope {
        subject_sha256: subject_sha256.to_owned(),
        operation: operation.clone(),
        platform: platform.clone(),
        platform_version: None,
        environment_id: environment_id.clone(),
    };
    let expected_answer = normalized_result.clone();
    let task = OracleTask {
        task_id: task_id.to_owned(),
        artifact_sha256: subject_sha256.to_owned(),
        expected_scope,
        expected_answer: expected_answer.clone(),
        fault_family: "synthetic-control".to_owned(),
        pair,
        occurrence,
    };

    let claim = ClaimRequest {
        schema: rea_compat::CLAIM_SCHEMA.to_owned(),
        claim_id: format!("claim-{task_id}"),
        predicate: ClaimPredicate::ProviderReportedObservation,
        scope: ClaimScope {
            subject_sha256: claim_subject_override
                .unwrap_or(subject_sha256)
                .to_owned(),
            operation: claim_operation_override.unwrap_or(&operation).to_owned(),
            platform,
            platform_version: None,
            environment_id,
        },
        asserted_value: claim_asserted_override.unwrap_or(expected_answer.clone()),
        source_predicate_type: Some(source_predicate_type),
        evidence_ids: vec![evidence.evidence_id.clone()],
        assumptions: Vec::new(),
        falsification_condition: "A same-scope verified observation differs".to_owned(),
    };

    let (claim_json, result_json, evidence_json) = if include_replay {
        let mut result =
            serde_json::to_value(evaluate_claim(&claim, std::slice::from_ref(&evidence)).unwrap())
                .unwrap();
        if force_reported_supported {
            result["disposition"] = json!("supported_within_scope");
        }
        (
            Some(serde_json::to_string(&claim).unwrap()),
            Some(serde_json::to_string(&result).unwrap()),
            vec![String::from_utf8(evidence_bytes).unwrap()],
        )
    } else {
        (None, None, Vec::new())
    };

    let submission = AgentSubmission {
        task_id: task_id.to_owned(),
        abstained,
        answer: if abstained {
            Value::Null
        } else {
            answer.unwrap_or(expected_answer)
        },
        claim_json,
        result_json,
        evidence_json,
    };
    Case { task, submission }
}

fn manifest(cases: &[Case]) -> (Vec<u8>, Vec<u8>) {
    let oracle = OracleManifest {
        schema: "luminous.rea.agent-oracle.v0".to_owned(),
        suite_id: "scorecard-conformance-v0".to_owned(),
        tasks: cases.iter().map(|case| case.task.clone()).collect(),
    };
    let submission = rea_agent_scorecard::SubmissionManifest {
        schema: SUBMISSION_SCHEMA.to_owned(),
        suite_id: oracle.suite_id.clone(),
        submissions: cases
            .iter()
            .map(|case| case.submission.clone())
            .collect(),
    };
    (
        serde_json::to_vec(&oracle).unwrap(),
        serde_json::to_vec(&submission).unwrap(),
    )
}

fn score(cases: &[Case]) -> Scorecard {
    let (oracle, submission) = manifest(cases);
    evaluate_scorecard(&oracle, &submission).unwrap()
}

#[test]
fn raw_correct_but_ungrounded_answers_count_as_parametric_leakage() {
    let evidence = rea_compat::verify_evidence_json(FIXTURE).unwrap();
    let expected = evidence.value["normalized_result"].clone();
    let grounded = make_case(
        "grounded",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        expected.clone(),
        None,
        None,
        None,
        None,
        false,
        true,
        false,
        None,
        None,
    );
    let ungrounded = make_case(
        "ungrounded",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        expected,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
    );
    let result = score(&[grounded, ungrounded]);

    assert_eq!(result.schema, SCORECARD_SCHEMA);
    assert_eq!(result.overall.counts.eligible, 2);
    assert_eq!(result.overall.counts.raw_correct, 2);
    assert_eq!(result.overall.counts.evidence_grounded_correct, 1);
    assert_eq!(result.overall.counts.correct_but_ungrounded, 1);
    assert_eq!(result.overall.parametric_leakage_rate.numerator, 1);
    assert_eq!(result.overall.parametric_leakage_rate.denominator, 2);
    assert_eq!(result.overall.parametric_leakage_rate.rate, Some(0.5));
}

#[test]
fn correct_answer_with_agent_claimed_support_but_wrong_value_is_false_supported() {
    let evidence = rea_compat::verify_evidence_json(FIXTURE).unwrap();
    let expected = evidence.value["normalized_result"].clone();
    let wrong = json!({"status":"wrong"});
    let case = make_case(
        "wrong-answer-supported",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        expected,
        Some(wrong),
        None,
        None,
        None,
        false,
        true,
        false,
        None,
        None,
    );
    let result = score(&[case]);

    assert_eq!(result.overall.counts.claimed_supported, 1);
    assert_eq!(result.overall.counts.raw_correct, 0);
    assert_eq!(result.overall.counts.false_supported, 1);
    assert_eq!(result.overall.false_supported_rate.rate, Some(1.0));
    assert_eq!(result.task_results[0].replay_status, "match");
}

#[test]
fn wrong_scope_and_operation_cannot_earn_grounded_credit_even_with_positive_claim_text() {
    let evidence = rea_compat::verify_evidence_json(FIXTURE).unwrap();
    let expected = evidence.value["normalized_result"].clone();
    let wrong_subject = make_case(
        "wrong-subject",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        expected.clone(),
        None,
        Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        None,
        None,
        true,
        true,
        false,
        None,
        None,
    );
    let wrong_operation = make_case(
        "wrong-operation",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        expected,
        None,
        None,
        Some("verify_signature"),
        None,
        true,
        true,
        false,
        None,
        None,
    );
    let result = score(&[wrong_subject, wrong_operation]);

    assert_eq!(result.overall.counts.evidence_grounded_correct, 0);
    assert_eq!(result.overall.counts.false_supported, 2);
    assert!(result.task_results.iter().all(|item| !item.claim_scope_exact));
    assert!(result.task_results.iter().all(|item| item.replay_status == "mismatch"));
}

#[test]
fn duplicate_filename_occurrence_identity_is_part_of_grounded_answer() {
    let expected = json!({
        "path":"classes.dex",
        "occurrence_index":2,
        "raw_name_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "size":42
    });
    let occurrence = OccurrenceExpectation {
        index_json_pointer: "/occurrence_index".to_owned(),
        occurrence_index: 2,
        raw_name_sha256_json_pointer: Some("/raw_name_sha256".to_owned()),
        raw_name_sha256: Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned()),
    };
    let grounded = make_case(
        "duplicate-occurrence-2",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        expected.clone(),
        None,
        None,
        None,
        None,
        false,
        true,
        false,
        Some(occurrence.clone()),
        None,
    );
    let wrong_occurrence = json!({
        "path":"classes.dex",
        "occurrence_index":1,
        "raw_name_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "size":42
    });
    let forged = make_case(
        "duplicate-occurrence-1-forged",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        expected,
        None,
        None,
        None,
        Some(wrong_occurrence),
        true,
        true,
        false,
        Some(occurrence),
        None,
    );
    let result = score(&[grounded, forged]);

    assert_eq!(result.overall.counts.evidence_grounded_correct, 1);
    assert_eq!(result.overall.counts.false_supported, 1);
    assert!(result.task_results[0].occurrence_exact);
    assert!(!result.task_results[1].occurrence_exact);
}

#[test]
fn relevant_mutation_requires_answer_change_and_irrelevant_mutation_requires_invariance() {
    let baseline_value = json!({"status":"complete","occurrence_index":0});
    let relevant_value = json!({"status":"partial","occurrence_index":0});
    let relevant_baseline = make_case(
        "relevant-base",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        baseline_value.clone(),
        None,
        None,
        None,
        None,
        false,
        true,
        false,
        None,
        Some(PairBinding {
            pair_id:"relevant-pair".to_owned(),
            kind:PairKind::RelevantMutation,
            role:PairRole::Baseline,
        }),
    );
    let relevant_mutated = make_case(
        "relevant-mutated",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        relevant_value,
        None,
        None,
        None,
        None,
        false,
        true,
        false,
        None,
        Some(PairBinding {
            pair_id:"relevant-pair".to_owned(),
            kind:PairKind::RelevantMutation,
            role:PairRole::Mutated,
        }),
    );
    let irrelevant_baseline = make_case(
        "irrelevant-base",
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        baseline_value.clone(),
        None,
        None,
        None,
        None,
        false,
        true,
        false,
        None,
        Some(PairBinding {
            pair_id:"irrelevant-pair".to_owned(),
            kind:PairKind::IrrelevantMutation,
            role:PairRole::Baseline,
        }),
    );
    let irrelevant_mutated = make_case(
        "irrelevant-mutated",
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        baseline_value,
        None,
        None,
        None,
        None,
        false,
        true,
        false,
        None,
        Some(PairBinding {
            pair_id:"irrelevant-pair".to_owned(),
            kind:PairKind::IrrelevantMutation,
            role:PairRole::Mutated,
        }),
    );
    let result = score(&[
        relevant_baseline,
        relevant_mutated,
        irrelevant_baseline,
        irrelevant_mutated,
    ]);

    assert_eq!(result.pairs.relevant_mutation_pairs, 1);
    assert_eq!(result.pairs.relevant_mutation_sensitive, 1);
    assert_eq!(result.pairs.relevant_mutation_both_grounded, 1);
    assert_eq!(result.pairs.irrelevant_mutation_pairs, 1);
    assert_eq!(result.pairs.irrelevant_mutation_invariant, 1);
    assert_eq!(result.pairs.irrelevant_mutation_both_grounded, 1);
}

#[test]
fn abstentions_missing_submissions_and_zero_denominators_remain_explicit() {
    let expected = rea_compat::verify_evidence_json(FIXTURE)
        .unwrap()
        .value["normalized_result"]
        .clone();
    let abstained = make_case(
        "abstained",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        expected.clone(),
        None,
        None,
        None,
        None,
        false,
        false,
        true,
        None,
        None,
    );
    let (oracle, submission) = manifest(&[abstained]);
    let mut parsed: Value = serde_json::from_slice(&submission).unwrap();
    parsed["submissions"] = json!([]);
    let scorecard = evaluate_scorecard(&oracle, &serde_json::to_vec(&parsed).unwrap()).unwrap();

    assert_eq!(scorecard.overall.counts.eligible, 1);
    assert_eq!(scorecard.overall.counts.submitted, 0);
    assert_eq!(scorecard.overall.counts.raw_correct, 0);
    assert_eq!(scorecard.overall.counts.abstained, 0);
    assert_eq!(scorecard.overall.parametric_leakage_rate.rate, None);
    assert_eq!(scorecard.overall.answer_coverage.rate, Some(0.0));
}

#[test]
fn duplicate_object_keys_are_rejected_in_oracle_and_submission() {
    let oracle = br#"{
        "schema":"luminous.rea.agent-oracle.v0",
        "suite_id":"s",
        "suite_id":"forged",
        "tasks":[]
    }"#;
    let submission = br#"{
        "schema":"luminous.rea.agent-submission.v0",
        "suite_id":"s",
        "submissions":[],
        "submissions":[]
    }"#;
    let error = evaluate_scorecard(oracle, b"{}").unwrap_err();
    assert_eq!(error.kind, rea_agent_scorecard::ScorecardErrorKind::InvalidJson);
    let error = evaluate_scorecard(
        br#"{"schema":"luminous.rea.agent-oracle.v0","suite_id":"s","tasks":[]}"#,
        submission,
    )
    .unwrap_err();
    assert_eq!(error.kind, rea_agent_scorecard::ScorecardErrorKind::InvalidJson);
}

#[test]
fn unknown_task_ids_and_duplicate_submissions_are_rejected() {
    let expected = rea_compat::verify_evidence_json(FIXTURE)
        .unwrap()
        .value["normalized_result"]
        .clone();
    let case = make_case(
        "known",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        expected,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
    );
    let (oracle, submission) = manifest(&[case]);
    let mut parsed: Value = serde_json::from_slice(&submission).unwrap();
    parsed["submissions"][0]["task_id"] = json!("unknown");
    let error = evaluate_scorecard(&oracle, &serde_json::to_vec(&parsed).unwrap()).unwrap_err();
    assert_eq!(error.kind, rea_agent_scorecard::ScorecardErrorKind::InvalidSubmission);

    let mut parsed: Value = serde_json::from_slice(&submission).unwrap();
    parsed["submissions"].as_array_mut().unwrap().push(parsed["submissions"][0].clone());
    let error = evaluate_scorecard(&oracle, &serde_json::to_vec(&parsed).unwrap()).unwrap_err();
    assert_eq!(error.kind, rea_agent_scorecard::ScorecardErrorKind::InvalidSubmission);
}

#[test]
fn fault_family_denominators_include_missing_submission_tasks() {
    let expected = rea_compat::verify_evidence_json(FIXTURE)
        .unwrap()
        .value["normalized_result"]
        .clone();
    let mut first = make_case(
        "family-a-1",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        expected.clone(),
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
    );
    first.task.fault_family = "format-a".to_owned();
    let mut second = make_case(
        "family-a-2",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        expected,
        None,
        None,
        None,
        None,
        false,
        false,
        false,
        None,
        None,
    );
    second.task.fault_family = "format-a".to_owned();
    let (oracle, submission) = manifest(&[first, second]);
    let mut parsed: Value = serde_json::from_slice(&submission).unwrap();
    parsed["submissions"].as_array_mut().unwrap().pop();
    let result = evaluate_scorecard(&oracle, &serde_json::to_vec(&parsed).unwrap()).unwrap();
    assert_eq!(result.by_fault_family["format-a"].counts.eligible, 2);
    assert_eq!(result.by_fault_family["format-a"].counts.submitted, 1);
}
