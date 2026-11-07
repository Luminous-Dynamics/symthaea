#[path = "../../../src/rea_compat.rs"]
mod rea_compat;

use rea_agent_scorecard::{
    OracleCorpus, ReplayInput, Submission, SubmissionSet, SUBMISSIONS_SCHEMA,
    evaluate_scorecard, parse_oracle_json,
};
use rea_compat::{
    ClaimPredicate, ClaimRequest, ClaimScope, analysis_profile_digest, compute_evidence_id,
    evaluate_claim, verify_evidence_json,
};
use serde_json::json;

fn oracle_bytes() -> &'static [u8] {
    include_bytes!("fixtures/rea-agent-oracle-seed-v0.json")
}

fn replay_input(
    subject_sha256: &str,
    observed_value: Value,
    asserted_value: Value,
    corrupt_result_digest: bool,
) -> ReplayInput {
    let original = verify_evidence_json(include_bytes!(
        "../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
    ))
    .unwrap();

    let mut evidence_value = original.value;
    evidence_value["subject"]["digest"]["sha256"] = json!(subject_sha256);
    evidence_value["normalized_result"] = observed_value;
    evidence_value["analysis_profile"]["digest"] =
        json!(analysis_profile_digest(&evidence_value["analysis_profile"]).unwrap());
    evidence_value["evidence_id"] = json!(compute_evidence_id(&evidence_value).unwrap());

    let evidence_json = serde_json::to_vec(&evidence_value).unwrap();
    let evidence = verify_evidence_json(&evidence_json).unwrap();
    let claim = ClaimRequest {
        schema: rea_compat::CLAIM_SCHEMA.to_owned(),
        claim_id: format!("scorecard-{}", &subject_sha256[..8]),
        predicate: ClaimPredicate::ProviderReportedObservation,
        scope: ClaimScope {
            subject_sha256: subject_sha256.to_owned(),
            operation: "inspect_artifact".to_owned(),
            platform: None,
            platform_version: None,
            environment_id: None,
        },
        asserted_value,
        source_predicate_type: Some("rea.analysis".to_owned()),
        evidence_ids: vec![evidence.evidence_id.clone()],
        assumptions: Vec::new(),
        falsification_condition: "A same-scope verified Evidence record has a different normalized_result".to_owned(),
    };
    let evaluation = evaluate_claim(&claim, &[evidence]).unwrap();
    let mut result_value = serde_json::to_value(&evaluation).unwrap();
    if corrupt_result_digest {
        result_value["derivation_sha256"] = json!("e".repeat(64));
    }

    ReplayInput {
        claim_json: serde_json::to_string(&claim).unwrap(),
        result_json: serde_json::to_string(&result_value).unwrap(),
        evidence_json: vec![String::from_utf8(evidence_json).unwrap()],
    }
}

fn submission(
    task_id: &str,
    answer: Option<Value>,
    occurrence_id: Option<&str>,
    replay: Option<ReplayInput>,
) -> Submission {
    Submission {
        task_id: task_id.to_owned(),
        answer,
        observed_occurrence_id: occurrence_id.map(str::to_owned),
        replay,
    }
}

fn submission_set() -> SubmissionSet {
    SubmissionSet {
        schema: SUBMISSIONS_SCHEMA.to_owned(),
        run_id: "synthetic-run-001".to_owned(),
        agent_id: "test-agent".to_owned(),
        submissions: vec![
            submission(
                "positive-grounded",
                Some(json!({"occurrence_id":"classes.dex#0","status":"complete"})),
                Some("classes.dex#0"),
                Some(replay_input(
                    &"a".repeat(64),
                    json!({"occurrence_id":"classes.dex#0","status":"complete"}),
                    json!({"occurrence_id":"classes.dex#0","status":"complete"}),
                    false,
                )),
            ),
            submission(
                "correct-but-ungrounded",
                Some(json!({"status":"remembered-answer"})),
                None,
                None,
            ),
            submission(
                "false-supported",
                Some(json!({"status":"unsafe"})),
                None,
                Some(replay_input(
                    &"c".repeat(64),
                    json!({"status":"unsafe"}),
                    json!({"status":"unsafe"}),
                    false,
                )),
            ),
            submission(
                "wrong-subject-binding",
                Some(json!({"status":"complete"})),
                None,
                Some(replay_input(
                    &"9".repeat(64),
                    json!({"status":"complete"}),
                    json!({"status":"complete"}),
                    false,
                )),
            ),
            submission(
                "relevant-base",
                Some(json!({"version":1})),
                None,
                Some(replay_input(
                    &"e".repeat(64),
                    json!({"version":1}),
                    json!({"version":1}),
                    false,
                )),
            ),
            submission(
                "relevant-variant",
                Some(json!({"version":1})),
                None,
                Some(replay_input(
                    &"f".repeat(64),
                    json!({"version":2}),
                    json!({"version":1}),
                    false,
                )),
            ),
            submission(
                "irrelevant-base",
                Some(json!({"version":3})),
                None,
                Some(replay_input(
                    &"1".repeat(64),
                    json!({"version":3}),
                    json!({"version":3}),
                    false,
                )),
            ),
            submission(
                "irrelevant-variant",
                Some(json!({"version":3})),
                None,
                Some(replay_input(
                    &"2".repeat(64),
                    json!({"version":3}),
                    json!({"version":3}),
                    false,
                )),
            ),
            submission(
                "replay-digest-mismatch",
                Some(json!({"status":"complete"})),
                None,
                Some(replay_input(
                    &"3".repeat(64),
                    json!({"status":"complete"}),
                    json!({"status":"complete"}),
                    true,
                )),
            ),
            submission(
                "wrong-duplicate-occurrence",
                Some(json!({"occurrence_id":"duplicate-name#0","status":"complete"})),
                Some("duplicate-name#0"),
                Some(replay_input(
                    &"4".repeat(64),
                    json!({"occurrence_id":"duplicate-name#0","status":"complete"}),
                    json!({"occurrence_id":"duplicate-name#0","status":"complete"}),
                    false,
                )),
            ),
            submission("unsupported-control", None, None, None),
        ],
    }
}

#[test]
fn hidden_oracle_scorecard_distinguishes_raw_truth_from_replayed_grounding() {
    let oracle: OracleCorpus = parse_oracle_json(oracle_bytes()).unwrap();
    let submissions = submission_set();
    let scorecard = evaluate_scorecard(&oracle, &submissions).unwrap();
    let metrics = scorecard.metrics;

    assert_eq!(metrics.tasks, 11);
    assert_eq!(metrics.answered, 10);
    assert_eq!(metrics.raw_correct, 7);
    assert_eq!(metrics.grounded_correct, 4);
    assert_eq!(metrics.correct_but_ungrounded, 3);
    assert_eq!(metrics.parametric_leakage_rate, Some(3.0 / 7.0));
    assert_eq!(metrics.positive_claims, 7);
    assert_eq!(metrics.false_supported_claims, 3);
    assert_eq!(metrics.false_supported_rate, Some(3.0 / 7.0));
    assert_eq!(metrics.replay_attempts, 9);
    assert_eq!(metrics.replay_matches, 8);
    assert_eq!(metrics.replay_match_rate, Some(8.0 / 9.0));
    assert_eq!(metrics.invalid_replays, 0);
    assert_eq!(metrics.unsupported_controls, 1);
    assert_eq!(metrics.correct_unsupported_abstentions, 1);
    assert_eq!(metrics.correct_unsupported_abstention_rate, Some(1.0));
    assert_eq!(metrics.relevant_mutation_pairs, 1);
    assert_eq!(metrics.relevant_mutation_sensitive_pairs, 0);
    assert_eq!(metrics.irrelevant_mutation_pairs, 1);
    assert_eq!(metrics.irrelevant_mutation_invariant_pairs, 1);
    assert_eq!(metrics.irrelevant_mutation_rebound_pairs, 1);
    assert_eq!(metrics.raw_accuracy_full_suite, Some(7.0 / 11.0));
    assert_eq!(metrics.raw_accuracy_when_answered, Some(0.7));
    assert_eq!(metrics.answer_coverage, Some(10.0 / 11.0));
    assert_eq!(metrics.grounded_accuracy_full_suite, Some(4.0 / 11.0));

    let wrong_subject = scorecard
        .tasks
        .iter()
        .find(|task| task.task_id == "wrong-subject-binding")
        .unwrap();
    assert!(wrong_subject.raw_correct);
    assert!(!wrong_subject.grounded_correct);
    assert!(wrong_subject.false_supported);

    let duplicate = scorecard
        .tasks
        .iter()
        .find(|task| task.task_id == "wrong-duplicate-occurrence")
        .unwrap();
    assert!(!duplicate.raw_correct);
    assert!(duplicate.positive_claim);
    assert!(duplicate.false_supported);

    let mismatch = scorecard
        .tasks
        .iter()
        .find(|task| task.task_id == "replay-digest-mismatch")
        .unwrap();
    assert!(mismatch.raw_correct);
    assert_eq!(
        mismatch.replay_state,
        rea_agent_scorecard::ReplayState::Mismatch
    );
    assert!(!mismatch.grounded_correct);

    let missing = scorecard
        .tasks
        .iter()
        .find(|task| task.task_id == "unsupported-control")
        .unwrap();
    assert!(missing.correct_unsupported_abstention);
}

#[test]
fn scorecard_input_cannot_smuggle_a_submission_for_an_unknown_task() {
    let oracle: OracleCorpus = parse_oracle_json(oracle_bytes()).unwrap();
    let mut submissions = submission_set();
    submissions.submissions.push(submission(
        "not-in-hidden-oracle",
        Some(json!({"answer":"extra"})),
        None,
        None,
    ));
    let error = evaluate_scorecard(&oracle, &submissions).unwrap_err();
    assert_eq!(
        error.kind,
        rea_agent_scorecard::ScorecardErrorKind::UnknownTask
    );
}
