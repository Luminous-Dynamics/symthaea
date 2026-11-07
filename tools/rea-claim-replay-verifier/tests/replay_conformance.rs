#[path = "../../../src/rea_compat.rs"]
mod rea_compat;
#[path = "../src/lib.rs"]
mod replay_verifier;

use rea_compat::claims::{ClaimPredicate, ClaimRequest, ClaimScope, evaluate_claim};
use replay_verifier::{
    Disposition, MAX_EVIDENCE_RECORDS, ReplayErrorKind, ReplayOutcome, replay_claim_evaluation,
};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn frozen_oracle_corpus_has_independent_expected_labels() {
    let corpus: Value =
        serde_json::from_str(include_str!("fixtures/rea-claim-replay-v0-oracle.json")).unwrap();
    assert_eq!(
        corpus["schema"],
        json!("luminous.rea.claim-replay-oracle.v0")
    );
    assert_eq!(corpus["status"], json!("frozen_public_seed"));
    assert_eq!(
        corpus["provenance"]["authored_independently_of_production_evaluator"],
        json!(true)
    );
    assert_eq!(
        corpus["cases"].as_array().unwrap().len(),
        23,
        "the frozen public seed has exactly 23 independently labeled cases"
    );
}

fn fixture_bytes() -> &'static [u8] {
    include_bytes!(
        "../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
    )
}

fn oracle_label(case_id: &str, field: &str) -> String {
    let corpus: Value =
        serde_json::from_str(include_str!("fixtures/rea-claim-replay-v0-oracle.json"))
            .expect("frozen independent replay oracle parses");
    corpus["cases"]
        .as_array()
        .expect("oracle cases are an array")
        .iter()
        .find(|case| case["id"].as_str() == Some(case_id))
        .unwrap_or_else(|| panic!("oracle case '{case_id}' is missing"))[field]
        .as_str()
        .unwrap_or_else(|| panic!("oracle case '{case_id}' has no {field}"))
        .to_owned()
}

fn oracle_disposition(case_id: &str) -> Disposition {
    match oracle_label(case_id, "expected_disposition").as_str() {
        "supported_within_scope" => Disposition::SupportedWithinScope,
        "refuted_within_scope" => Disposition::RefutedWithinScope,
        "underdetermined" => Disposition::Underdetermined,
        "conflicted" => Disposition::Conflicted,
        "inapplicable" => Disposition::Inapplicable,
        "unverified" => Disposition::Unverified,
        other => panic!("unknown frozen expected disposition '{other}'"),
    }
}

fn oracle_replay_outcome(case_id: &str) -> ReplayOutcome {
    match oracle_label(case_id, "expected_replay_outcome").as_str() {
        "replay_match" => ReplayOutcome::ReplayMatch,
        "replay_mismatch" => ReplayOutcome::ReplayMismatch,
        other => panic!("case '{case_id}' has no replay outcome: '{other}'"),
    }
}

fn json_with_reversed_object_keys(value: &Value) -> String {
    match value {
        Value::Object(object) => {
            let members = object
                .iter()
                .rev()
                .map(|(key, value)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).unwrap(),
                        json_with_reversed_object_keys(value)
                    )
                })
                .collect::<Vec<_>>();
            format!("{{{}}}", members.join(","))
        }
        Value::Array(values) => {
            let elements = values
                .iter()
                .map(json_with_reversed_object_keys)
                .collect::<Vec<_>>();
            format!("[{}]", elements.join(","))
        }
        _ => serde_json::to_string(value).unwrap(),
    }
}

fn provider_claim(evidence_id: String) -> ClaimRequest {
    ClaimRequest {
        schema: rea_compat::CLAIM_SCHEMA.to_owned(),
        claim_id: "replay-fixture-0001".to_owned(),
        predicate: ClaimPredicate::ProviderReportedObservation,
        scope: ClaimScope {
            subject_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                .to_owned(),
            operation: "inspect_artifact".to_owned(),
            platform: None,
            platform_version: None,
            environment_id: None,
        },
        asserted_value: json!({
            "entries": [{"path": "classes.dex", "size": 42}],
            "status": "complete"
        }),
        source_predicate_type: Some("rea.analysis".to_owned()),
        evidence_ids: vec![evidence_id],
        assumptions: Vec::new(),
        falsification_condition:
            "A same-scope verified Evidence record has a different normalized_result".to_owned(),
    }
}

fn produce(claim: &ClaimRequest, evidence: &[rea_compat::VerifiedEvidence]) -> (Vec<u8>, Vec<u8>) {
    let claim_bytes = serde_json::to_vec(claim).unwrap();
    let result = evaluate_claim(claim, evidence).unwrap();
    let result_bytes = serde_json::to_vec(&result).unwrap();
    (claim_bytes, result_bytes)
}

#[test]
fn independent_replay_matches_supported_provider_observation() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let replay = replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &result_bytes).unwrap();
    assert_eq!(
        replay.outcome,
        oracle_replay_outcome("basic-provider-report-support")
    );
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("basic-provider-report-support")
    );
    assert_eq!(replay.expected_disposition, replay.reported_disposition);
    assert_eq!(
        replay.recomputed_derivation_sha256,
        replay.reported_derivation_sha256
    );
}

#[test]
fn independent_replay_uses_jcs_equivalence_for_json_numbers() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut claim = provider_claim(evidence.evidence_id.clone());
    claim.asserted_value["entries"][0]["size"] = json!(42.0);
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let replay = replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("jcs-number-42-equals-42-point-0")
    );
}

#[test]
fn independent_replay_preserves_conflicts_instead_of_voting() {
    let first = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut second_value = first.value.clone();
    second_value["provider"]["id"] = json!("second-provider");
    second_value["provider"]["name"] = json!("Second Provider");
    second_value["analysis_profile"]["provider"] = second_value["provider"].clone();
    second_value["analysis_profile"]["digest"] =
        json!(rea_compat::analysis_profile_digest(&second_value["analysis_profile"]).unwrap());
    second_value["normalized_result"]["status"] = json!("partial");
    second_value["evidence_id"] = json!(rea_compat::compute_evidence_id(&second_value).unwrap());

    let second_bytes = serde_json::to_vec(&second_value).unwrap();
    let second = rea_compat::verify_evidence_json(&second_bytes).unwrap();
    let claim = provider_claim(first.evidence_id.clone());
    let mut claim = claim;
    claim.evidence_ids.push(second.evidence_id.clone());

    let (claim_bytes, result_bytes) = produce(&claim, &[first, second]);
    let replay = replay_claim_evaluation(
        &claim_bytes,
        &[fixture_bytes(), &second_bytes],
        &result_bytes,
    )
    .unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("same-scope-disagreement")
    );
}

#[test]
fn independent_replay_keeps_missing_premises_underdetermined() {
    let missing_id = format!("ev_{}", "b".repeat(64));
    let claim = provider_claim(missing_id);
    let (claim_bytes, result_bytes) = produce(&claim, &[]);

    let replay = replay_claim_evaluation(&claim_bytes, &[], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("referenced-evidence-missing")
    );
    assert_eq!(replay.missing_evidence_count, 1);
}

#[test]
fn independent_replay_reproduces_unverified_predicate_ceiling() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut claim = provider_claim(evidence.evidence_id.clone());
    claim.predicate = ClaimPredicate::ProducerAuthenticity;
    claim.source_predicate_type = None;
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let replay = replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("producer-authenticity-rule-unimplemented")
    );
}

#[test]
fn forged_result_disposition_does_not_pass_on_a_self_digest_alone() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let mut forged: Value = serde_json::from_slice(&result_bytes).unwrap();
    forged["disposition"] = json!("refuted_within_scope");

    // The derivation digest is unkeyed. Recompute it over the forged body so
    // replay must reject the semantic lie, not merely a stale self-hash.
    let mut forged_body = forged.clone();
    forged_body
        .as_object_mut()
        .unwrap()
        .remove("derivation_sha256");
    let forged_canonical = rea_compat::canonical_jcs_bytes(&forged_body).unwrap();
    forged["derivation_sha256"] = json!(rea_compat::sha256_hex(&forged_canonical));

    let forged_bytes = serde_json::to_vec(&forged).unwrap();
    let replay = replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &forged_bytes).unwrap();
    assert_eq!(
        replay.outcome,
        oracle_replay_outcome("forged-disposition-with-valid-shaped-digest")
    );
    assert_ne!(replay.expected_disposition, replay.reported_disposition);
}

#[test]
fn changed_request_cannot_reuse_an_old_result() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (_claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let mut changed = claim;
    changed.falsification_condition = "A different falsification condition".to_owned();
    let changed_bytes = serde_json::to_vec(&changed).unwrap();
    let replay =
        replay_claim_evaluation(&changed_bytes, &[fixture_bytes()], &result_bytes).unwrap();
    assert_eq!(
        replay.outcome,
        oracle_replay_outcome("changed-falsification-condition-old-result")
    );
    assert_ne!(
        replay.reported_derivation_sha256,
        replay.recomputed_derivation_sha256
    );
}

#[test]
fn duplicate_json_keys_are_rejected_recursively_in_claims_and_results() {
    let raw_claim = br#"{
        "schema":"luminous.rea.claim.v0",
        "claim_id":"c1",
        "predicate":"provider_reported_observation",
        "scope":{"subject_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","operation":"inspect_artifact"},
        "asserted_value":{"status":"complete","status":"partial"},
        "source_predicate_type":"rea.analysis",
        "evidence_ids":[],
        "assumptions":[],
        "falsification_condition":"a different result"
    }"#;
    let error = replay_claim_evaluation(raw_claim, &[], b"{}").unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::InvalidJson);

    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);
    let mut result_text = String::from_utf8(result_bytes).unwrap();
    result_text = result_text.replacen(
        "\"claim_id\":\"replay-fixture-0001\"",
        "\"claim_id\":\"replay-fixture-0001\",\"claim_id\":\"forged\"",
        1,
    );
    let error = replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], result_text.as_bytes())
        .unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::InvalidJson);
}

#[test]
fn omitted_nullable_result_field_is_not_silently_defaulted() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let mut result: Value = serde_json::from_slice(&result_bytes).unwrap();
    result["evidence_assessments"][0]
        .as_object_mut()
        .unwrap()
        .remove("provider_version");
    let changed = serde_json::to_vec(&result).unwrap();
    let error = replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &changed).unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::InvalidJson);
}

#[test]
fn unreferenced_evidence_is_not_silently_added_to_the_replay_set() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, std::slice::from_ref(&evidence));

    let mut unrelated = evidence.value.clone();
    unrelated["provider"]["id"] = json!("unreferenced-provider");
    unrelated["provider"]["name"] = json!("Unreferenced Provider");
    unrelated["analysis_profile"]["provider"] = unrelated["provider"].clone();
    unrelated["analysis_profile"]["digest"] =
        json!(rea_compat::analysis_profile_digest(&unrelated["analysis_profile"]).unwrap());
    unrelated["evidence_id"] = json!(rea_compat::compute_evidence_id(&unrelated).unwrap());
    let unrelated_bytes = serde_json::to_vec(&unrelated).unwrap();

    let error = replay_claim_evaluation(
        &claim_bytes,
        &[fixture_bytes(), &unrelated_bytes],
        &result_bytes,
    )
    .unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::UnexpectedEvidence);
}

#[test]
fn independent_expected_labels_cover_wrong_scope_and_missing_platform_version() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();

    let mut wrong_subject = provider_claim(evidence.evidence_id.clone());
    wrong_subject.scope.subject_sha256 =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned();
    let (claim_bytes, result_bytes) = produce(&wrong_subject, std::slice::from_ref(&evidence));
    let replay = replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("wrong-subject-digest")
    );

    let mut wrong_operation = provider_claim(evidence.evidence_id.clone());
    wrong_operation.scope.operation = "verify_signature".to_owned();
    let (claim_bytes, result_bytes) = produce(&wrong_operation, std::slice::from_ref(&evidence));
    let replay = replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("wrong-operation")
    );

    let mut unrepresented_platform_version = provider_claim(evidence.evidence_id.clone());
    unrepresented_platform_version.scope.platform_version = Some("Android 17 / API 37".to_owned());
    let (claim_bytes, result_bytes) = produce(
        &unrepresented_platform_version,
        std::slice::from_ref(&evidence),
    );
    let replay = replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("platform-version-not-represented")
    );
}

#[test]
fn independent_expected_label_refutes_a_same_scope_numeric_near_miss() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut claim = provider_claim(evidence.evidence_id.clone());
    claim.asserted_value["entries"][0]["size"] = json!(42.0001);
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let replay = replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("numeric-near-miss-42-point-0001")
    );
}

#[test]
fn recorded_assumptions_cannot_make_a_claim_supported() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut claim = provider_claim(evidence.evidence_id.clone());
    claim
        .assumptions
        .push("the scanner database is complete".to_owned());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let replay = replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("unverified-assumption-present")
    );
}

#[test]
fn result_schema_and_rule_version_mutations_are_rejected_before_replay() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let mut unknown_field: Value = serde_json::from_slice(&result_bytes).unwrap();
    unknown_field["unreviewed_authority"] = json!("trusted");
    let unknown_bytes = serde_json::to_vec(&unknown_field).unwrap();
    let error =
        replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &unknown_bytes).unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::InvalidResult);

    let mut wrong_rule: Value = serde_json::from_slice(&result_bytes).unwrap();
    wrong_rule["evaluator_version"] = json!("rea-claim-evaluator/v99");
    let wrong_rule_bytes = serde_json::to_vec(&wrong_rule).unwrap();
    let error =
        replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &wrong_rule_bytes).unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::InvalidResult);
}

#[test]
fn duplicate_evidence_input_is_rejected_even_when_both_records_are_valid() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let error = replay_claim_evaluation(
        &claim_bytes,
        &[fixture_bytes(), fixture_bytes()],
        &result_bytes,
    )
    .unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::DuplicateEvidence);
}

#[test]
fn replay_result_comparison_uses_jcs_not_json_number_token_spelling() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let mut result: Value = serde_json::from_slice(&result_bytes).unwrap();
    // The claim and derivation use 42; this result re-encoding uses 42.0.
    // RFC 8785 maps both through ECMAScript binary64 serialization, so this
    // representation-only change must not break deterministic replay.
    result["asserted_value"]["entries"][0]["size"] = json!(42.0);
    let equivalent_bytes = serde_json::to_vec(&result).unwrap();

    let replay =
        replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &equivalent_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("basic-provider-report-support")
    );
}

#[test]
fn malformed_result_evidence_references_are_rejected_as_invalid_results() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);
    let original: Value = serde_json::from_slice(&result_bytes).unwrap();

    let mut malformed_requested = original.clone();
    malformed_requested["requested_evidence_ids"][0] = json!("ev_not-a-sha256");
    assert_eq!(
        oracle_label(
            "malformed-requested-result-evidence-id",
            "expected_replay_outcome"
        ),
        "invalid_input"
    );
    let error = replay_claim_evaluation(
        &claim_bytes,
        &[fixture_bytes()],
        &serde_json::to_vec(&malformed_requested).unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::InvalidResult);

    let mut unrequested_assessment = original.clone();
    unrequested_assessment["evidence_assessments"][0]["evidence_id"] =
        json!(format!("ev_{}", "c".repeat(64)));
    assert_eq!(
        oracle_label(
            "unrequested-result-assessment-id",
            "expected_replay_outcome"
        ),
        "invalid_input"
    );
    let error = replay_claim_evaluation(
        &claim_bytes,
        &[fixture_bytes()],
        &serde_json::to_vec(&unrequested_assessment).unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::InvalidResult);

    let mut duplicate_assessment = original.clone();
    let repeated_assessment = duplicate_assessment["evidence_assessments"][0].clone();
    duplicate_assessment["evidence_assessments"]
        .as_array_mut()
        .unwrap()
        .push(repeated_assessment);
    assert_eq!(
        oracle_label("duplicate-result-assessment-id", "expected_replay_outcome"),
        "invalid_input"
    );
    let error = replay_claim_evaluation(
        &claim_bytes,
        &[fixture_bytes()],
        &serde_json::to_vec(&duplicate_assessment).unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::InvalidResult);

    let mut missing_unrequested = original.clone();
    missing_unrequested["missing_evidence_ids"] = json!([format!("ev_{}", "d".repeat(64))]);
    assert_eq!(
        oracle_label(
            "unrequested-missing-result-evidence-id",
            "expected_replay_outcome"
        ),
        "invalid_input"
    );
    let error = replay_claim_evaluation(
        &claim_bytes,
        &[fixture_bytes()],
        &serde_json::to_vec(&missing_unrequested).unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::InvalidResult);

    let mut assessed_and_missing = original;
    assessed_and_missing["missing_evidence_ids"] =
        json!([rea_compat::verify_evidence_json(fixture_bytes())
            .unwrap()
            .evidence_id]);
    assert_eq!(
        oracle_label(
            "evidence-id-marked-assessed-and-missing",
            "expected_replay_outcome"
        ),
        "invalid_input"
    );
    let error = replay_claim_evaluation(
        &claim_bytes,
        &[fixture_bytes()],
        &serde_json::to_vec(&assessed_and_missing).unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::InvalidResult);
}

fn temporary_cli_dir() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock follows Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("rea-claim-replay-{}-{nonce}", std::process::id()))
}

#[test]
fn bounded_cli_emits_machine_readable_replay_match() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let directory = temporary_cli_dir();
    std::fs::create_dir_all(&directory).unwrap();
    let claim_path = directory.join("claim.json");
    let result_path = directory.join("result.json");
    let evidence_path = directory.join("evidence.json");
    std::fs::write(&claim_path, claim_bytes).unwrap();
    std::fs::write(&result_path, result_bytes).unwrap();
    std::fs::write(&evidence_path, fixture_bytes()).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_rea-claim-replay"))
        .arg(&claim_path)
        .arg(&result_path)
        .arg(&evidence_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(receipt["outcome"], json!("replay_match"));
    assert_eq!(
        receipt["expected_disposition"],
        json!("supported_within_scope")
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn bounded_cli_returns_distinct_nonzero_exit_for_semantic_mismatch() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let mut forged: Value = serde_json::from_slice(&result_bytes).unwrap();
    forged["disposition"] = json!("refuted_within_scope");
    let forged_bytes = serde_json::to_vec(&forged).unwrap();

    let directory = temporary_cli_dir();
    std::fs::create_dir_all(&directory).unwrap();
    let claim_path = directory.join("claim.json");
    let result_path = directory.join("result.json");
    let evidence_path = directory.join("evidence.json");
    std::fs::write(&claim_path, claim_bytes).unwrap();
    std::fs::write(&result_path, forged_bytes).unwrap();
    std::fs::write(&evidence_path, fixture_bytes()).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_rea-claim-replay"))
        .arg(&claim_path)
        .arg(&result_path)
        .arg(&evidence_path)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let receipt: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(receipt["outcome"], json!("replay_mismatch"));
    std::fs::remove_dir_all(directory).unwrap();
}
#[test]
fn replay_preserves_inferred_confidence_and_analyst_authority_without_promotion() {
    let original = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut value = original.value.clone();
    value["confidence"] = json!("inferred");
    value["authority"] = json!("analyst-inference");
    value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());
    let evidence_bytes = serde_json::to_vec(&value).unwrap();
    let evidence = rea_compat::verify_evidence_json(&evidence_bytes).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let replay = replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        Disposition::SupportedWithinScope
    );

    let assessment =
        &serde_json::from_slice::<Value>(&result_bytes).unwrap()["evidence_assessments"][0];
    assert_eq!(assessment["confidence"], json!("inferred"));
    assert_eq!(assessment["authority"], json!("analyst-inference"));
}

#[test]
fn oversized_request_evidence_reference_count_is_a_resource_limit() {
    let mut claim = provider_claim(format!("ev_{}", "a".repeat(64)));
    claim.evidence_ids = (0..=MAX_EVIDENCE_RECORDS)
        .map(|index| format!("ev_{index:064x}"))
        .collect();
    let claim_bytes = serde_json::to_vec(&claim).unwrap();

    let error = replay_claim_evaluation(&claim_bytes, &[], b"{}").unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::ResourceLimit);
}

#[test]
fn oversized_replay_evidence_input_count_is_rejected_before_parsing() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id);
    let claim_bytes = serde_json::to_vec(&claim).unwrap();
    let too_many: Vec<&[u8]> = vec![b"not-json"; MAX_EVIDENCE_RECORDS + 1];

    let error = replay_claim_evaluation(&claim_bytes, &too_many, b"{}").unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::ResourceLimit);
}

#[test]
fn oversized_result_evidence_reference_count_is_a_resource_limit() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);
    let mut result: Value = serde_json::from_slice(&result_bytes).unwrap();
    result["requested_evidence_ids"] = json!(
        (0..=MAX_EVIDENCE_RECORDS)
            .map(|index| format!("ev_{index:064x}"))
            .collect::<Vec<_>>()
    );

    let oversized_result = serde_json::to_vec(&result).unwrap();
    let error =
        replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &oversized_result).unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::ResourceLimit);
}

#[test]
fn cli_rejects_too_many_evidence_paths_before_opening_files() {
    let mut command = Command::new(env!("CARGO_BIN_EXE_rea-claim-replay"));
    command.arg("missing-claim.json").arg("missing-result.json");
    command.args(std::iter::repeat_n(
        "missing-evidence.json",
        MAX_EVIDENCE_RECORDS + 1,
    ));
    let output = command.output().unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Evidence paths"));
}

#[test]
fn replay_projects_only_claim_authorized_fields_after_full_evidence_verification() {
    let original = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut value = original.value;
    value["raw_result"] = json!({
        "opaque_payload": "x".repeat(1024 * 1024),
        "unrelated_annotations": [{"note": "not a claim premise"}]
    });
    value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());
    let evidence_bytes = serde_json::to_vec(&value).unwrap();
    let evidence = rea_compat::verify_evidence_json(&evidence_bytes).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let replay = replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("basic-provider-report-support")
    );
}
#[test]
fn request_and_derivation_digest_mutations_fail_replay_even_when_disposition_is_unchanged() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let original: Value = serde_json::from_slice(&result_bytes).unwrap();
    assert_eq!(original["disposition"], json!("supported_within_scope"));

    let mut wrong_request_digest = original.clone();
    wrong_request_digest["claim_request_sha256"] = json!(format!("{}{}", "b".repeat(63), "c"));
    let error = replay_claim_evaluation(
        &claim_bytes,
        &[fixture_bytes()],
        &serde_json::to_vec(&wrong_request_digest).unwrap(),
    )
    .unwrap();
    assert_eq!(error.outcome, ReplayOutcome::ReplayMismatch);
    assert_eq!(
        error.expected_disposition,
        Disposition::SupportedWithinScope
    );
    assert_eq!(
        error.reported_disposition,
        Disposition::SupportedWithinScope
    );

    let mut wrong_derivation_digest = original;
    wrong_derivation_digest["derivation_sha256"] = json!(format!("{}{}", "d".repeat(63), "e"));
    let error = replay_claim_evaluation(
        &claim_bytes,
        &[fixture_bytes()],
        &serde_json::to_vec(&wrong_derivation_digest).unwrap(),
    )
    .unwrap();
    assert_eq!(error.outcome, ReplayOutcome::ReplayMismatch);
    assert_eq!(
        error.expected_disposition,
        Disposition::SupportedWithinScope
    );
    assert_eq!(
        error.reported_disposition,
        Disposition::SupportedWithinScope
    );
}

#[test]
fn ordered_evidence_references_are_bound_to_the_exact_claim_request() {
    let first = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut second_value = first.value.clone();
    second_value["provider"]["id"] = json!("second-ordering-provider");
    second_value["provider"]["name"] = json!("Second Ordering Provider");
    second_value["analysis_profile"]["provider"] = second_value["provider"].clone();
    second_value["analysis_profile"]["digest"] =
        json!(rea_compat::analysis_profile_digest(&second_value["analysis_profile"]).unwrap());
    second_value["evidence_id"] = json!(rea_compat::compute_evidence_id(&second_value).unwrap());
    let second_bytes = serde_json::to_vec(&second_value).unwrap();
    let second = rea_compat::verify_evidence_json(&second_bytes).unwrap();

    let mut claim = provider_claim(first.evidence_id.clone());
    claim.evidence_ids.push(second.evidence_id.clone());
    let (_claim_bytes, result_bytes) = produce(&claim, &[first.clone(), second.clone()]);

    let mut reordered = claim;
    reordered.evidence_ids.reverse();
    let reordered_bytes = serde_json::to_vec(&reordered).unwrap();
    let error = replay_claim_evaluation(
        &reordered_bytes,
        &[fixture_bytes(), &second_bytes],
        &result_bytes,
    )
    .unwrap();
    assert_eq!(error.outcome, ReplayOutcome::ReplayMismatch);
    assert_ne!(error.claim_request_sha256, error.reported_derivation_sha256);
}

#[test]
fn recursive_json_object_key_permutations_preserve_semantic_replay() {
    let original_evidence: Value = serde_json::from_slice(fixture_bytes()).unwrap();
    let reordered_evidence = json_with_reversed_object_keys(&original_evidence);
    let verified = rea_compat::verify_evidence_json(reordered_evidence.as_bytes()).unwrap();
    assert_eq!(
        verified.evidence_id,
        rea_compat::verify_evidence_json(fixture_bytes())
            .unwrap()
            .evidence_id
    );

    let claim = provider_claim(verified.evidence_id.clone());
    let claim_value = serde_json::to_value(&claim).unwrap();
    let claim_bytes = json_with_reversed_object_keys(&claim_value).into_bytes();

    let result = evaluate_claim(&claim, std::slice::from_ref(&verified)).unwrap();
    let result_value = serde_json::to_value(&result).unwrap();
    let result_bytes = json_with_reversed_object_keys(&result_value).into_bytes();

    let replay = replay_claim_evaluation(
        &claim_bytes,
        &[reordered_evidence.as_bytes()],
        &result_bytes,
    )
    .unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("basic-provider-report-support")
    );
    assert_eq!(
        replay.recomputed_derivation_sha256,
        replay.reported_derivation_sha256
    );
}

#[test]
fn evidence_transport_order_does_not_change_bound_claim_replay() {
    let first = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut second_value = first.value.clone();
    second_value["provider"]["id"] = json!("second-transport-order-provider");
    second_value["provider"]["name"] = json!("Second Transport Order Provider");
    second_value["analysis_profile"]["provider"] = second_value["provider"].clone();
    second_value["analysis_profile"]["digest"] =
        json!(rea_compat::analysis_profile_digest(&second_value["analysis_profile"]).unwrap());
    second_value["evidence_id"] = json!(rea_compat::compute_evidence_id(&second_value).unwrap());
    let second_bytes = serde_json::to_vec(&second_value).unwrap();
    let second = rea_compat::verify_evidence_json(&second_bytes).unwrap();

    let mut claim = provider_claim(first.evidence_id.clone());
    claim.evidence_ids.push(second.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[first, second]);

    // Transport order is not semantic order: evidence is keyed by its bound ID.
    let replay = replay_claim_evaluation(
        &claim_bytes,
        &[&second_bytes, fixture_bytes()],
        &result_bytes,
    )
    .unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("basic-provider-report-support")
    );
    assert_eq!(
        replay.recomputed_derivation_sha256,
        replay.reported_derivation_sha256
    );
}

#[test]
fn evidence_transport_order_is_commutative_for_a_fixed_claim_reference_order() {
    let original = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut evidence_values = Vec::new();
    let mut evidence_bytes = Vec::new();
    for index in 0..3 {
        let mut value = original.value.clone();
        let provider_id = format!("permutation-provider-{index}");
        value["provider"]["id"] = json!(provider_id);
        value["provider"]["name"] = json!(format!("Permutation Provider {index}"));
        value["analysis_profile"]["provider"] = value["provider"].clone();
        value["analysis_profile"]["digest"] =
            json!(rea_compat::analysis_profile_digest(&value["analysis_profile"]).unwrap());
        value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());
        let bytes = serde_json::to_vec(&value).unwrap();
        let verified = rea_compat::verify_evidence_json(&bytes).unwrap();
        evidence_values.push(verified);
        evidence_bytes.push(bytes);
    }

    let mut claim = provider_claim(evidence_values[0].evidence_id.clone());
    claim.evidence_ids = evidence_values
        .iter()
        .map(|evidence| evidence.evidence_id.clone())
        .collect();
    let (claim_bytes, result_bytes) = produce(&claim, &evidence_values);

    // Six transport permutations, while claim.evidence_ids remain in the same
    // order. Replay must be set-based at the transport boundary and
    // deterministic at the derivation boundary.
    let permutations = [
        [0usize, 1usize, 2usize],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for permutation in permutations {
        let supplied: Vec<&[u8]> = permutation
            .iter()
            .map(|index| evidence_bytes[*index].as_slice())
            .collect();
        let replay = replay_claim_evaluation(&claim_bytes, &supplied, &result_bytes).unwrap();
        assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
        assert_eq!(
            replay.expected_disposition,
            Disposition::SupportedWithinScope
        );
        assert_eq!(
            replay.recomputed_derivation_sha256,
            replay.reported_derivation_sha256
        );
    }
}

#[test]
fn raw_evidence_whitespace_is_nonsemantic_but_tampering_remains_detectable() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, std::slice::from_ref(&evidence));

    let pretty = serde_json::to_vec_pretty(&evidence.value).unwrap();
    let pretty_verified = rea_compat::verify_evidence_json(&pretty).unwrap();
    assert_eq!(pretty_verified.evidence_id, evidence.evidence_id);
    assert_ne!(pretty_verified.raw_file_sha256, evidence.raw_file_sha256);

    let replay = replay_claim_evaluation(&claim_bytes, &[&pretty], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);

    let mut tampered: Value = serde_json::from_slice(fixture_bytes()).unwrap();
    tampered["normalized_result"]["entries"][0]["size"] = json!(4200);
    let tampered_bytes = serde_json::to_vec(&tampered).unwrap();
    let error =
        replay_claim_evaluation(&claim_bytes, &[&tampered_bytes], &result_bytes).unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::InvalidEvidence);
}

#[test]
fn more_provider_observations_do_not_upgrade_an_unimplemented_predicate() {
    let original = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut evidence_values = Vec::new();
    let mut evidence_bytes = Vec::new();

    for index in 0..4 {
        let mut value = original.value.clone();
        value["provider"]["id"] = json!(format!("same-lineage-provider-{index}"));
        value["provider"]["name"] = json!(format!("Same Lineage Provider {index}"));
        value["analysis_profile"]["provider"] = value["provider"].clone();
        value["analysis_profile"]["digest"] =
            json!(rea_compat::analysis_profile_digest(&value["analysis_profile"]).unwrap());
        value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());

        let bytes = serde_json::to_vec(&value).unwrap();
        evidence_values.push(rea_compat::verify_evidence_json(&bytes).unwrap());
        evidence_bytes.push(bytes);
    }

    let mut claim = provider_claim(evidence_values[0].evidence_id.clone());
    claim.predicate = ClaimPredicate::ProducerAuthenticity;
    claim.source_predicate_type = None;
    claim.evidence_ids = evidence_values
        .iter()
        .map(|evidence| evidence.evidence_id.clone())
        .collect();

    let (claim_bytes, result_bytes) = produce(&claim, &evidence_values);
    let supplied: Vec<&[u8]> = evidence_bytes.iter().map(Vec::as_slice).collect();
    let replay = replay_claim_evaluation(&claim_bytes, &supplied, &result_bytes).unwrap();

    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(replay.expected_disposition, Disposition::Unverified);
    assert_eq!(replay.reported_disposition, Disposition::Unverified);
    assert_eq!(replay.evidence_count, 4);
    assert_eq!(replay.rule_id, "no-registered-predicate-rule/v0");
}

#[test]
fn deleting_the_only_supporting_premise_cannot_preserve_a_supported_result() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    // The old result says supported, but the replay input omits its only
    // referenced premise. The checker must recompute underdetermined.
    let replay = replay_claim_evaluation(&claim_bytes, &[], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMismatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("referenced-evidence-missing")
    );
    assert_eq!(
        replay.reported_disposition,
        Disposition::SupportedWithinScope
    );
}

#[test]
fn three_same_scope_contradictions_do_not_outvote_one_supporting_observation() {
    let supporting = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut records = vec![supporting.clone()];
    let mut bytes = Vec::new();

    for index in 0..3 {
        let mut value = supporting.value.clone();
        value["provider"]["id"] = json!(format!("contradictory-provider-{index}"));
        value["provider"]["name"] = json!(format!("Contradictory Provider {index}"));
        value["analysis_profile"]["provider"] = value["provider"].clone();
        value["analysis_profile"]["digest"] =
            json!(rea_compat::analysis_profile_digest(&value["analysis_profile"]).unwrap());
        value["normalized_result"]["status"] = json!("partial");
        value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());

        let raw = serde_json::to_vec(&value).unwrap();
        records.push(rea_compat::verify_evidence_json(&raw).unwrap());
        bytes.push(raw);
    }

    let mut claim = provider_claim(supporting.evidence_id.clone());
    claim.evidence_ids.extend(
        records
            .iter()
            .skip(1)
            .map(|record| record.evidence_id.clone()),
    );
    let (claim_bytes, result_bytes) = produce(&claim, &records);

    let mut evidence_inputs = vec![fixture_bytes()];
    evidence_inputs.extend(bytes.iter().map(Vec::as_slice));
    let replay = replay_claim_evaluation(&claim_bytes, &evidence_inputs, &result_bytes).unwrap();

    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("same-scope-disagreement")
    );
    assert_eq!(replay.reported_disposition, Disposition::Conflicted);
}

#[test]
fn inferred_analyst_derived_evidence_does_not_promote_beyond_provider_observation() {
    let original = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut value = original.value.clone();
    value["confidence"] = json!("inferred");
    value["authority"] = json!("analyst-inference");
    value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());

    let evidence_bytes = serde_json::to_vec(&value).unwrap();
    let evidence = rea_compat::verify_evidence_json(&evidence_bytes).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

    let result: Value = serde_json::from_slice(&result_bytes).unwrap();
    assert_eq!(result["disposition"], json!("supported_within_scope"));
    assert_eq!(
        result["evidence_assessments"][0]["confidence"],
        json!("inferred")
    );
    assert_eq!(
        result["evidence_assessments"][0]["authority"],
        json!("analyst-inference")
    );
    assert!(
        result["claim_ceiling"]
            .as_array()
            .unwrap()
            .iter()
            .any(|line| line
                .as_str()
                .unwrap()
                .contains("does not prove the underlying real-world proposition"))
    );

    let replay = replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        oracle_disposition("basic-provider-report-support")
    );
}

#[test]
fn mismatched_source_predicate_type_is_inapplicable_not_silent_promotion() {
    let original = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut value = original.value.clone();
    value["predicate_type"] = json!("rea.other-analysis");
    value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());
    let evidence_bytes = serde_json::to_vec(&value).unwrap();
    let evidence = rea_compat::verify_evidence_json(&evidence_bytes).unwrap();

    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);
    let replay = replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &result_bytes).unwrap();

    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(replay.expected_disposition, Disposition::Inapplicable);
    assert_eq!(replay.reported_disposition, Disposition::Inapplicable);
}

#[test]
fn deterministic_claim_mutation_matrix_never_replays_the_old_result() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, std::slice::from_ref(&evidence));

    let mut mutations = Vec::<(&str, ClaimRequest)>::new();

    let mut changed = claim.clone();
    changed.claim_id.push_str("-changed");
    mutations.push(("claim_id", changed));

    let mut changed = claim.clone();
    changed.predicate = ClaimPredicate::ProducerAuthenticity;
    changed.source_predicate_type = None;
    mutations.push(("predicate", changed));

    let mut changed = claim.clone();
    changed.scope.subject_sha256 =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned();
    mutations.push(("scope.subject_sha256", changed));

    let mut changed = claim.clone();
    changed.scope.operation = "verify_signature".to_owned();
    mutations.push(("scope.operation", changed));

    let mut changed = claim.clone();
    changed.scope.platform = Some("Android".to_owned());
    mutations.push(("scope.platform", changed));

    let mut changed = claim.clone();
    changed.scope.platform_version = Some("Android 17 / API 37".to_owned());
    mutations.push(("scope.platform_version", changed));

    let mut changed = claim.clone();
    changed.scope.environment_id = Some("device-42".to_owned());
    mutations.push(("scope.environment_id", changed));

    let mut changed = claim.clone();
    changed.asserted_value["status"] = json!("partial");
    mutations.push(("asserted_value", changed));

    let mut changed = claim.clone();
    changed.source_predicate_type = Some("rea.other-analysis".to_owned());
    mutations.push(("source_predicate_type", changed));

    let mut changed = claim.clone();
    changed
        .assumptions
        .push("the provider database is complete".to_owned());
    mutations.push(("assumptions", changed));

    let mut changed = claim;
    changed.falsification_condition = "a different falsifier".to_owned();
    mutations.push(("falsification_condition", changed));

    for (field, mutated) in mutations {
        let mutated_bytes = serde_json::to_vec(&mutated).unwrap();
        let replay = replay_claim_evaluation(&mutated_bytes, &[fixture_bytes()], &result_bytes);
        match replay {
            Ok(receipt) => assert_ne!(
                receipt.outcome,
                ReplayOutcome::ReplayMatch,
                "mutation of {field} unexpectedly replayed"
            ),
            Err(error) => assert_ne!(
                error.kind,
                ReplayErrorKind::Canonicalization,
                "mutation of {field} failed for an unrelated canonicalization error"
            ),
        }
    }

    // Keep the original request bytes in use: the matrix mutates each field
    // independently, never the expected baseline.
    assert!(!claim_bytes.is_empty());
}

#[test]
fn contradiction_cardinality_sweep_never_turns_conflict_into_majority_support() {
    let original = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();

    for contradiction_count in [1usize, 2, 3, 7, 15] {
        let mut evidence_values = vec![original.clone()];
        let mut evidence_bytes = Vec::new();

        for index in 0..contradiction_count {
            let mut value = original.value.clone();
            value["provider"]["id"] = json!(format!(
                "cardinality-contradiction-{contradiction_count}-{index}"
            ));
            value["provider"]["name"] = json!(format!(
                "Cardinality Contradiction {contradiction_count} {index}"
            ));
            value["analysis_profile"]["provider"] = value["provider"].clone();
            value["analysis_profile"]["digest"] =
                json!(rea_compat::analysis_profile_digest(&value["analysis_profile"]).unwrap());
            value["normalized_result"]["status"] = json!("partial");
            value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());

            let bytes = serde_json::to_vec(&value).unwrap();
            evidence_values.push(rea_compat::verify_evidence_json(&bytes).unwrap());
            evidence_bytes.push(bytes);
        }

        let mut claim = provider_claim(original.evidence_id.clone());
        claim.evidence_ids = evidence_values
            .iter()
            .map(|record| record.evidence_id.clone())
            .collect();

        let (claim_bytes, result_bytes) = produce(&claim, &evidence_values);
        let mut supplied = vec![fixture_bytes()];
        supplied.extend(evidence_bytes.iter().map(Vec::as_slice));
        let replay = replay_claim_evaluation(&claim_bytes, &supplied, &result_bytes).unwrap();

        assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
        assert_eq!(
            replay.expected_disposition,
            oracle_disposition("same-scope-disagreement")
        );
        assert_eq!(replay.expected_disposition, Disposition::Conflicted);
        assert_eq!(replay.evidence_count, contradiction_count + 1);
    }
}

#[test]
fn unsupported_predicate_closure_sweep_cannot_be_bought_with_more_observations() {
    let original = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let predicates = [
        ClaimPredicate::MemberContentIntegrity,
        ClaimPredicate::ArchiveContentIdentity,
        ClaimPredicate::WholeArchiveCompleteness,
        ClaimPredicate::ProducerAuthenticity,
        ClaimPredicate::SignatureCryptographicValidity,
        ClaimPredicate::SignerIdentity,
        ClaimPredicate::SignerTrust,
        ClaimPredicate::PackageAuthorization,
        ClaimPredicate::PlatformAcceptance,
        ClaimPredicate::ProviderCorroboration,
        ClaimPredicate::CausalFinding,
        ClaimPredicate::SecurityControlPass,
    ];

    for predicate in predicates {
        for evidence_count in [0usize, 1, 2, 4, 8, 16, 32] {
            let mut evidence_values = Vec::new();
            let mut evidence_bytes = Vec::new();

            for index in 0..evidence_count {
                let mut value = original.value.clone();
                value["provider"]["id"] = json!(format!(
                    "unsupported-{predicate:?}-{evidence_count}-{index}"
                ));
                value["provider"]["name"] = json!(format!(
                    "Unsupported {predicate:?} {evidence_count} {index}"
                ));
                value["analysis_profile"]["provider"] = value["provider"].clone();
                value["analysis_profile"]["digest"] =
                    json!(rea_compat::analysis_profile_digest(&value["analysis_profile"]).unwrap());
                value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());

                let bytes = serde_json::to_vec(&value).unwrap();
                evidence_values.push(rea_compat::verify_evidence_json(&bytes).unwrap());
                evidence_bytes.push(bytes);
            }

            let placeholder = evidence_values
                .first()
                .map(|record| record.evidence_id.clone())
                .unwrap_or_else(|| format!("ev_{}", "e".repeat(64)));
            let mut claim = provider_claim(placeholder);
            claim.predicate = predicate;
            claim.source_predicate_type = None;
            claim.evidence_ids = evidence_values
                .iter()
                .map(|record| record.evidence_id.clone())
                .collect();

            let (claim_bytes, result_bytes) = produce(&claim, &evidence_values);
            let supplied: Vec<&[u8]> = evidence_bytes.iter().map(Vec::as_slice).collect();
            let replay = replay_claim_evaluation(&claim_bytes, &supplied, &result_bytes).unwrap();

            assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
            assert_eq!(
                replay.expected_disposition,
                oracle_disposition("producer-authenticity-rule-unimplemented")
            );
            assert_eq!(replay.expected_disposition, Disposition::Unverified);
            assert_eq!(replay.evidence_count, evidence_count);
        }
    }
}

#[test]
fn bounded_cli_rejects_oversized_sparse_files_before_reading_payloads() {
    let directory = temporary_cli_dir();
    std::fs::create_dir_all(&directory).unwrap();

    let small_claim = directory.join("small-claim.json");
    let small_result = directory.join("small-result.json");
    std::fs::write(&small_claim, b"{}").unwrap();
    std::fs::write(&small_result, b"{}").unwrap();

    let oversized_claim = directory.join("oversized-claim.json");
    std::fs::File::create(&oversized_claim)
        .unwrap()
        .set_len(1024 * 1024 + 1)
        .unwrap();

    let oversized_result = directory.join("oversized-result.json");
    std::fs::File::create(&oversized_result)
        .unwrap()
        .set_len(4 * 1024 * 1024 + 1)
        .unwrap();

    let oversized_evidence = directory.join("oversized-evidence.json");
    std::fs::File::create(&oversized_evidence)
        .unwrap()
        .set_len(symthaea_rea_compat::MAX_EVIDENCE_JSON_BYTES as u64 + 1)
        .unwrap();

    let cases = [
        (
            "claim",
            vec![oversized_claim.as_os_str(), small_result.as_os_str()],
        ),
        (
            "result",
            vec![small_claim.as_os_str(), oversized_result.as_os_str()],
        ),
        (
            "Evidence",
            vec![
                small_claim.as_os_str(),
                small_result.as_os_str(),
                oversized_evidence.as_os_str(),
            ],
        ),
    ];

    for (label, arguments) in cases {
        let output = Command::new(env!("CARGO_BIN_EXE_rea-claim-replay"))
            .args(arguments)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2), "{label} input");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("exceeds"),
            "{label} input should fail at byte-limit preflight: {stderr}"
        );
    }

    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn self_reported_population_completeness_cannot_upgrade_whole_archive_claims() {
    let original = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let reported_completeness = [
        json!({
            "kind": "declared_query",
            "query": "all statements for issuer during interval T",
            "snapshot_id": "snapshot-01",
            "page_count": 4,
            "terminal_cursor_seen": true,
            "all_pages_seen": true,
            "observed_count": 400,
            "reported_total_count": 400,
            "hidden_filter": null
        }),
        json!({
            "kind": "issuer_participation",
            "issuer_claims_all_issued_statements_submitted": true,
            "independent_issuer_history_audit": null,
            "withheld_statement_check": null
        }),
        json!({
            "kind": "transparency_view",
            "all_registered_entries_enumerated": true,
            "checkpoint_verified": true,
            "issuer_submission_completeness_established": false
        }),
    ];

    let mut evidence_values = Vec::new();
    let mut evidence_bytes = Vec::new();
    for (index, report) in reported_completeness.into_iter().enumerate() {
        let mut value = original.value.clone();
        value["provider"]["id"] = json!(format!("population-provider-{index}"));
        value["provider"]["name"] = json!(format!("Population Provider {index}"));
        value["analysis_profile"]["provider"] = value["provider"].clone();
        value["analysis_profile"]["digest"] =
            json!(rea_compat::analysis_profile_digest(&value["analysis_profile"]).unwrap());
        value["confidence"] = json!("observed");
        value["authority"] = json!("external-service");
        value["normalized_result"]["population_completeness"] = report;
        value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());

        let bytes = serde_json::to_vec(&value).unwrap();
        evidence_values.push(rea_compat::verify_evidence_json(&bytes).unwrap());
        evidence_bytes.push(bytes);
    }

    // The narrow provider-report meta-claim may be supported: it says only
    // that the provider reported these exact normalized pagination fields.
    // This does not establish completeness of issuer history or the population.
    let mut report_claim = provider_claim(evidence_values[0].evidence_id.clone());
    report_claim.asserted_value = evidence_values[0].value["normalized_result"].clone();
    let (report_claim_bytes, report_result_bytes) =
        produce(&report_claim, std::slice::from_ref(&evidence_values[0]));
    let report_replay = replay_claim_evaluation(
        &report_claim_bytes,
        &[evidence_bytes[0].as_slice()],
        &report_result_bytes,
    )
    .unwrap();
    assert_eq!(report_replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        report_replay.expected_disposition,
        Disposition::SupportedWithinScope
    );
    let report_result: Value = serde_json::from_slice(&report_result_bytes).unwrap();
    assert!(
        report_result["claim_ceiling"]
            .as_array()
            .unwrap()
            .iter()
            .any(|line| line
                .as_str()
                .unwrap()
                .contains("does not prove the underlying real-world proposition"))
    );

    let mut claim = provider_claim(evidence_values[0].evidence_id.clone());
    claim.predicate = ClaimPredicate::WholeArchiveCompleteness;
    claim.source_predicate_type = None;
    claim.asserted_value = json!({
        "whole_archive_complete": true,
        "issuer_history_complete": true
    });
    claim.evidence_ids = evidence_values
        .iter()
        .map(|evidence| evidence.evidence_id.clone())
        .collect();

    let (claim_bytes, result_bytes) = produce(&claim, &evidence_values);
    let supplied: Vec<&[u8]> = evidence_bytes.iter().map(Vec::as_slice).collect();
    let replay = replay_claim_evaluation(&claim_bytes, &supplied, &result_bytes).unwrap();

    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(replay.expected_disposition, Disposition::Unverified);
    assert_eq!(replay.reported_disposition, Disposition::Unverified);
    assert_eq!(replay.evidence_count, 3);
    assert_eq!(replay.rule_id, "no-registered-predicate-rule/v0");

    let result: Value = serde_json::from_slice(&result_bytes).unwrap();
    assert!(
        result["evidence_assessments"]
            .as_array()
            .unwrap()
            .iter()
            .all(|assessment| assessment["kind"] == json!("not_evaluated"))
    );
    assert!(
        result["claim_ceiling"]
            .as_array()
            .unwrap()
            .iter()
            .any(|line| line
                .as_str()
                .unwrap()
                .contains("does not prove the underlying real-world proposition"))
    );
}

#[test]
fn independent_replay_reproduces_precision_warnings_and_binds_them_to_derivation() {
    let original = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let mut value = original.value.clone();
    value["normalized_result"]["large_count"] = json!(9_007_199_254_740_993_u64);
    value["parameters"]["large_offset"] = json!(18_446_744_073_709_551_615_u64);
    value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());

    let evidence_bytes = serde_json::to_vec(&value).unwrap();
    let evidence = rea_compat::verify_evidence_json(&evidence_bytes).unwrap();
    let mut claim = provider_claim(evidence.evidence_id.clone());
    claim.asserted_value = evidence.value["normalized_result"].clone();
    let (claim_bytes, result_bytes) = produce(&claim, std::slice::from_ref(&evidence));

    let replay = replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        Disposition::SupportedWithinScope
    );

    let mut result: Value = serde_json::from_slice(&result_bytes).unwrap();
    let warnings = result["numeric_precision_warnings"]
        .as_array()
        .expect("numeric precision warnings are a required result field");
    assert_eq!(warnings.len(), 3);
    assert_eq!(replay.numeric_precision_warnings.len(), warnings.len());
    assert_eq!(
        replay.numeric_precision_warnings[0].integer_value,
        "9007199254740993"
    );
    assert_eq!(warnings[0]["source"], json!("asserted_value"));
    assert_eq!(warnings[0]["integer_value"], json!("9007199254740993"));
    assert_eq!(warnings[1]["source"], json!("evidence_record"));
    assert_eq!(
        warnings[1]["json_pointer"],
        json!("/normalized_result/large_count")
    );
    assert_eq!(warnings[2]["integer_value"], json!("18446744073709551615"));

    // The warning array is required, not a defaulted or UI-only annotation.
    let mut missing_warnings = result.clone();
    missing_warnings
        .as_object_mut()
        .unwrap()
        .remove("numeric_precision_warnings");
    let missing_bytes = serde_json::to_vec(&missing_warnings).unwrap();
    let error =
        replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &missing_bytes).unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::InvalidJson);

    // A precision warning is part of the complete derivation. Changing it
    // must fail independent replay.
    result["numeric_precision_warnings"][0]["integer_value"] = json!("9007199254740992");
    let changed = serde_json::to_vec(&result).unwrap();
    let replay = replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &changed).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMismatch);
}

#[test]
fn removing_environment_scope_dimensions_cannot_upgrade_a_specific_observation() {
    let mut evidence_value = rea_compat::verify_evidence_json(fixture_bytes())
        .unwrap()
        .value;
    evidence_value["environment"] = json!({
        "id": "device-42",
        "platform": "Android",
        "architecture": "aarch64",
        "isolation": "process"
    });
    evidence_value["evidence_id"] =
        json!(rea_compat::compute_evidence_id(&evidence_value).unwrap());
    let evidence_bytes = serde_json::to_vec(&evidence_value).unwrap();
    let evidence = rea_compat::verify_evidence_json(&evidence_bytes).unwrap();

    let mut exact_scope = provider_claim(evidence.evidence_id.clone());
    exact_scope.scope.platform = Some("Android".to_owned());
    exact_scope.scope.environment_id = Some("device-42".to_owned());
    let (exact_claim_bytes, exact_result_bytes) =
        produce(&exact_scope, std::slice::from_ref(&evidence));
    let exact =
        replay_claim_evaluation(&exact_claim_bytes, &[&evidence_bytes], &exact_result_bytes)
            .unwrap();
    assert_eq!(exact.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        exact.expected_disposition,
        Disposition::SupportedWithinScope
    );

    for (label, platform, environment_id) in [
        ("platform", None, Some("device-42".to_owned())),
        ("environment_id", Some("Android".to_owned()), None),
        ("both", None, None),
    ] {
        let mut broadened = exact_scope.clone();
        broadened.scope.platform = platform;
        broadened.scope.environment_id = environment_id;
        let (claim_bytes, result_bytes) = produce(&broadened, std::slice::from_ref(&evidence));
        let replay =
            replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &result_bytes).unwrap();

        assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch, "{label}");
        assert_eq!(
            replay.expected_disposition,
            Disposition::Underdetermined,
            "removing {label} scope unexpectedly retained positive support"
        );

        let result: Value = serde_json::from_slice(&result_bytes).unwrap();
        assert!(
            result["evidence_assessments"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["kind"] == json!("scope_unverifiable")),
            "removing {label} scope did not preserve the missing-scope reason"
        );

        let stale_positive =
            replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &exact_result_bytes).unwrap();
        assert_eq!(
            stale_positive.outcome,
            ReplayOutcome::ReplayMismatch,
            "old positive result replayed after {label} scope was removed"
        );
        assert_eq!(
            stale_positive.expected_disposition,
            Disposition::Underdetermined
        );
    }
}

#[test]
fn confidence_and_authority_labels_cannot_promote_precision_warned_unimplemented_predicates() {
    let original = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let confidence_levels = ["observed", "derived", "inferred"];
    let authority_classes = [
        "shipped-artifact",
        "controlled-replay",
        "historical-reference",
        "external-service",
        "analyst-inference",
    ];
    let unsupported_predicates = [
        ClaimPredicate::MemberContentIntegrity,
        ClaimPredicate::ArchiveContentIdentity,
        ClaimPredicate::WholeArchiveCompleteness,
        ClaimPredicate::ProducerAuthenticity,
        ClaimPredicate::SignatureCryptographicValidity,
        ClaimPredicate::SignerIdentity,
        ClaimPredicate::SignerTrust,
        ClaimPredicate::PackageAuthorization,
        ClaimPredicate::PlatformAcceptance,
        ClaimPredicate::ProviderCorroboration,
        ClaimPredicate::CausalFinding,
        ClaimPredicate::SecurityControlPass,
    ];

    for predicate in unsupported_predicates {
        for confidence in confidence_levels {
            for authority in authority_classes {
                let mut value = original.value.clone();
                value["confidence"] = json!(confidence);
                value["authority"] = json!(authority);
                value["normalized_result"]["large_count"] = json!(9_007_199_254_740_993_u64);
                value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());
                let evidence_bytes = serde_json::to_vec(&value).unwrap();
                let evidence = rea_compat::verify_evidence_json(&evidence_bytes).unwrap();

                let mut claim = provider_claim(evidence.evidence_id.clone());
                claim.predicate = predicate;
                claim.source_predicate_type = None;
                let (claim_bytes, result_bytes) = produce(&claim, std::slice::from_ref(&evidence));
                let replay =
                    replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &result_bytes)
                        .unwrap();

                assert_eq!(
                    replay.outcome,
                    ReplayOutcome::ReplayMatch,
                    "{predicate:?}/{confidence}/{authority}"
                );
                assert_eq!(
                    replay.expected_disposition,
                    Disposition::Unverified,
                    "{predicate:?}/{confidence}/{authority} changed claim authority"
                );
                let result: Value = serde_json::from_slice(&result_bytes).unwrap();
                assert_eq!(
                    result["evidence_assessments"][0]["kind"],
                    json!("not_evaluated")
                );
                assert_eq!(result["rule_id"], json!("no-registered-predicate-rule/v0"));
                assert!(
                    result["numeric_precision_warnings"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|warning| warning["json_pointer"]
                            == json!("/normalized_result/large_count")),
                    "precision diagnostics were not retained for {predicate:?}/{confidence}/{authority}"
                );
            }
        }
    }
}

#[test]
fn unknown_temporal_scope_fields_are_rejected_instead_of_ignored() {
    let evidence = rea_compat::verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let (claim_bytes, result_bytes) = produce(&claim, std::slice::from_ref(&evidence));

    let mut claim_value: Value = serde_json::from_slice(&claim_bytes).unwrap();
    claim_value["scope"]["as_of"] = json!("2026-11-07T08:00:00Z");
    claim_value["scope"]["freshness_policy"] = json!("max-age-24h");
    let changed_claim = serde_json::to_vec(&claim_value).unwrap();

    let error =
        replay_claim_evaluation(&changed_claim, &[fixture_bytes()], &result_bytes).unwrap_err();
    assert_eq!(error.kind, ReplayErrorKind::InvalidClaim);
}

#[test]
fn issuer_and_transparency_self_completeness_reports_do_not_prove_population_completeness() {
    let mut value = rea_compat::verify_evidence_json(fixture_bytes())
        .unwrap()
        .value;
    value["normalized_result"] = json!({
        "registered_statements": [
            {"statement_id": "stmt-001", "subject": "artifact-A"}
        ],
        "registered_count": 1,
        "all_registered_statements_retrieved": true,
        "terminal_pagination_cursor_seen": true,
        "issuer_signed_all_issued_declaration": true,
        "issuer_issued_statement_population_independently_audited": false,
        "excluded_statement_population": "unknown"
    });
    value["raw_result"] = json!({
        "query": "all statements for artifact-A",
        "filters": [],
        "pagination": {"terminal": true},
        "source_note": "issuer claims all statements were submitted"
    });
    value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());
    let evidence_bytes = serde_json::to_vec(&value).unwrap();
    let evidence = rea_compat::verify_evidence_json(&evidence_bytes).unwrap();

    let mut report_claim = provider_claim(evidence.evidence_id.clone());
    report_claim.asserted_value = evidence.value["normalized_result"].clone();
    let (report_claim_bytes, report_result_bytes) =
        produce(&report_claim, std::slice::from_ref(&evidence));
    let report_replay = replay_claim_evaluation(
        &report_claim_bytes,
        &[&evidence_bytes],
        &report_result_bytes,
    )
    .unwrap();
    assert_eq!(report_replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        report_replay.expected_disposition,
        Disposition::SupportedWithinScope
    );
    let report_result: Value = serde_json::from_slice(&report_result_bytes).unwrap();
    assert!(
        report_result["claim_ceiling"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item
                .as_str()
                .unwrap()
                .contains("does not prove the underlying real-world proposition"))
    );

    let mut completeness_claim = report_claim;
    completeness_claim.predicate = ClaimPredicate::WholeArchiveCompleteness;
    completeness_claim.source_predicate_type = None;
    let (claim_bytes, result_bytes) = produce(&completeness_claim, std::slice::from_ref(&evidence));
    let replay = replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &result_bytes).unwrap();

    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(replay.expected_disposition, Disposition::Unverified);
    let result: Value = serde_json::from_slice(&result_bytes).unwrap();
    assert_eq!(result["rule_id"], json!("no-registered-predicate-rule/v0"));
    assert_eq!(
        result["evidence_assessments"][0]["kind"],
        json!("not_evaluated")
    );
    assert!(
        result["claim_ceiling"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item
                .as_str()
                .unwrap()
                .contains("No predicate-specific verifier"))
    );
}

#[test]
fn terminal_pagination_and_self_reported_all_pages_do_not_prove_population_completeness() {
    let mut evidence_value = rea_compat::verify_evidence_json(fixture_bytes())
        .unwrap()
        .value;
    evidence_value["normalized_result"] = json!({
        "query": {
            "query_id": "snapshot-query-17",
            "filters": [],
            "source_snapshot_id": null,
            "declared_total_pages": 3
        },
        "pages": [
            {"page": 1, "cursor_in": null, "cursor_out": "cursor-1", "items": ["a", "b"]},
            {"page": 3, "cursor_in": "cursor-2", "cursor_out": "terminal", "items": ["e"]}
        ],
        "observed_page_numbers": [1, 3],
        "terminal_cursor_seen": true,
        "all_pages_seen": true,
        "missing_page_numbers": [2],
        "issuer_participation_independently_audited": false
    });
    evidence_value["evidence_id"] =
        json!(rea_compat::compute_evidence_id(&evidence_value).unwrap());
    let evidence_bytes = serde_json::to_vec(&evidence_value).unwrap();
    let evidence = rea_compat::verify_evidence_json(&evidence_bytes).unwrap();

    let mut observation_claim = provider_claim(evidence.evidence_id.clone());
    observation_claim.asserted_value = evidence.value["normalized_result"].clone();
    let (observation_claim_bytes, observation_result_bytes) =
        produce(&observation_claim, std::slice::from_ref(&evidence));
    let observation_replay = replay_claim_evaluation(
        &observation_claim_bytes,
        &[&evidence_bytes],
        &observation_result_bytes,
    )
    .unwrap();
    assert_eq!(observation_replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        observation_replay.expected_disposition,
        Disposition::SupportedWithinScope
    );

    let mut completeness_claim = observation_claim;
    completeness_claim.predicate = ClaimPredicate::WholeArchiveCompleteness;
    completeness_claim.source_predicate_type = None;
    let (claim_bytes, result_bytes) = produce(&completeness_claim, std::slice::from_ref(&evidence));
    let replay = replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &result_bytes).unwrap();

    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(replay.expected_disposition, Disposition::Unverified);
    let result: Value = serde_json::from_slice(&result_bytes).unwrap();
    assert_eq!(result["rule_id"], json!("no-registered-predicate-rule/v0"));
    assert_eq!(
        result["evidence_assessments"][0]["kind"],
        json!("not_evaluated")
    );
    assert!(
        result["claim_ceiling"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item
                .as_str()
                .unwrap()
                .contains("No predicate-specific verifier"))
    );
}

#[test]
fn incomplete_downstream_processing_cannot_promote_whole_population_completeness() {
    let mut value = rea_compat::verify_evidence_json(fixture_bytes())
        .unwrap()
        .value;
    value["normalized_result"] = json!({
        "declared_query": {
            "query_id": "snapshot-query-23",
            "source_snapshot_id": "snapshot-23",
            "expected_items": ["item-a", "item-b", "item-c"]
        },
        "enumeration": {
            "enumerated_item_ids": ["item-a", "item-b", "item-c"],
            "terminal_cursor_seen": true
        },
        "acquisition": {
            "acquired_item_ids": ["item-a", "item-b", "item-c"],
            "acquisition_failures": []
        },
        "processing": {
            "processed_item_ids": ["item-a", "item-c"],
            "failed_item_ids": ["item-b"],
            "failure_reasons": [{"item_id": "item-b", "reason": "unsupported_format"}],
            "provider_claims_all_items_processed": true
        }
    });
    value["evidence_id"] = json!(rea_compat::compute_evidence_id(&value).unwrap());
    let evidence_bytes = serde_json::to_vec(&value).unwrap();
    let evidence = rea_compat::verify_evidence_json(&evidence_bytes).unwrap();

    let mut observation_claim = provider_claim(evidence.evidence_id.clone());
    observation_claim.asserted_value = evidence.value["normalized_result"].clone();
    let (observation_claim_bytes, observation_result_bytes) =
        produce(&observation_claim, std::slice::from_ref(&evidence));
    let observation_replay = replay_claim_evaluation(
        &observation_claim_bytes,
        &[&evidence_bytes],
        &observation_result_bytes,
    )
    .unwrap();
    assert_eq!(observation_replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        observation_replay.expected_disposition,
        Disposition::SupportedWithinScope
    );

    let mut completeness_claim = observation_claim;
    completeness_claim.predicate = ClaimPredicate::WholeArchiveCompleteness;
    completeness_claim.source_predicate_type = None;
    let (claim_bytes, result_bytes) = produce(&completeness_claim, std::slice::from_ref(&evidence));
    let replay = replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &result_bytes).unwrap();

    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(replay.expected_disposition, Disposition::Unverified);
    let result: Value = serde_json::from_slice(&result_bytes).unwrap();
    assert_eq!(result["rule_id"], json!("no-registered-predicate-rule/v0"));
    assert_eq!(
        result["evidence_assessments"][0]["kind"],
        json!("not_evaluated")
    );
}

#[test]
fn unsafe_integer_precision_warnings_survive_recursive_key_permutations() {
    let mut evidence_value = rea_compat::verify_evidence_json(fixture_bytes())
        .unwrap()
        .value;
    evidence_value["normalized_result"]["large_count"] = json!(9_007_199_254_740_993_u64);
    evidence_value["normalized_result"]["asserted_counter"] = json!(9_007_199_254_740_994_u64);
    evidence_value["parameters"]["large_offset"] = json!(18_446_744_073_709_551_615_u64);
    evidence_value["evidence_id"] =
        json!(rea_compat::compute_evidence_id(&evidence_value).unwrap());

    let evidence_bytes = serde_json::to_vec(&evidence_value).unwrap();
    let evidence = rea_compat::verify_evidence_json(&evidence_bytes).unwrap();
    let mut claim = provider_claim(evidence.evidence_id.clone());
    claim.asserted_value = evidence.value["normalized_result"].clone();
    let (claim_bytes, result_bytes) = produce(&claim, std::slice::from_ref(&evidence));

    let original_replay =
        replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &result_bytes).unwrap();
    assert_eq!(original_replay.outcome, ReplayOutcome::ReplayMatch);

    let reversed_claim =
        json_with_reversed_object_keys(&serde_json::from_slice::<Value>(&claim_bytes).unwrap());
    let reversed_evidence = json_with_reversed_object_keys(&evidence_value);
    let reversed_result =
        json_with_reversed_object_keys(&serde_json::from_slice::<Value>(&result_bytes).unwrap());

    let replay = replay_claim_evaluation(
        reversed_claim.as_bytes(),
        &[reversed_evidence.as_bytes()],
        reversed_result.as_bytes(),
    )
    .unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(
        replay.expected_disposition,
        Disposition::SupportedWithinScope
    );
    assert_eq!(
        replay.numeric_precision_warnings,
        original_replay.numeric_precision_warnings
    );
    assert_eq!(
        replay.recomputed_derivation_sha256,
        original_replay.recomputed_derivation_sha256
    );
}
