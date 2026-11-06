#[path = "../../../src/rea_compat.rs"]
mod rea_compat;
#[path = "../src/lib.rs"]
mod replay_verifier;

use rea_compat::claims::{
    ClaimDisposition, ClaimPredicate, ClaimRequest, ClaimScope, evaluate_claim,
};
use replay_verifier::{ReplayErrorKind, ReplayOutcome, replay_claim_evaluation};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn frozen_oracle_corpus_has_independent_expected_labels() {
    let corpus: Value = serde_json::from_str(include_str!(
        "fixtures/rea-claim-replay-v0-oracle.json"
    ))
    .unwrap();
    assert_eq!(corpus["schema"], json!("luminous.rea.claim-replay-oracle.v0"));
    assert_eq!(corpus["status"], json!("frozen_public_seed"));
    assert_eq!(
        corpus["provenance"]["authored_independently_of_production_evaluator"],
        json!(true)
    );
    assert!(
        corpus["cases"].as_array().unwrap().len() >= 18,
        "the frozen finite oracle must retain all expected cases"
    );
}

fn fixture_bytes() -> &'static [u8] {
    include_bytes!(
        "../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
    )
}


fn oracle_label(case_id: &str, field: &str) -> String {
    let corpus: Value = serde_json::from_str(include_str!(
        "fixtures/rea-claim-replay-v0-oracle.json"
    ))
    .expect("frozen independent replay oracle parses");
    corpus["cases"]
        .as_array()
        .expect("oracle cases are an array")
        .iter()
        .find(|case| case["id"] == case_id)
        .unwrap_or_else(|| panic!("oracle case '{case_id}' is missing"))[field]
        .as_str()
        .unwrap_or_else(|| panic!("oracle case '{case_id}' has no {field}"))
        .to_owned()
}

fn oracle_disposition(case_id: &str) -> ClaimDisposition {
    match oracle_label(case_id, "expected_disposition").as_str() {
        "supported_within_scope" => ClaimDisposition::SupportedWithinScope,
        "refuted_within_scope" => ClaimDisposition::RefutedWithinScope,
        "underdetermined" => ClaimDisposition::Underdetermined,
        "conflicted" => ClaimDisposition::Conflicted,
        "inapplicable" => ClaimDisposition::Inapplicable,
        "unverified" => ClaimDisposition::Unverified,
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
    assert_eq!(replay.expected_disposition, oracle_disposition("same-scope-disagreement"));
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
    assert_eq!(replay.expected_disposition, oracle_disposition("producer-authenticity-rule-unimplemented"));
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
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence]);

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
    let (claim_bytes, result_bytes) = produce(&claim, &[evidence.clone()]);

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
    assert_eq!(replay.expected_disposition, oracle_disposition("wrong-subject-digest"));

    let mut wrong_operation = provider_claim(evidence.evidence_id.clone());
    wrong_operation.scope.operation = "verify_signature".to_owned();
    let (claim_bytes, result_bytes) = produce(&wrong_operation, std::slice::from_ref(&evidence));
    let replay = replay_claim_evaluation(&claim_bytes, &[fixture_bytes()], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(replay.expected_disposition, oracle_disposition("wrong-operation"));

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
    assert_eq!(replay.expected_disposition, oracle_disposition("unverified-assumption-present"));
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

    let replay =
        replay_claim_evaluation(&claim_bytes, &[&evidence_bytes], &result_bytes).unwrap();
    assert_eq!(replay.outcome, ReplayOutcome::ReplayMatch);
    assert_eq!(replay.expected_disposition, ClaimDisposition::SupportedWithinScope);

    let assessment = &serde_json::from_slice::<Value>(&result_bytes).unwrap()
        ["evidence_assessments"][0];
    assert_eq!(assessment["confidence"], json!("inferred"));
    assert_eq!(assessment["authority"], json!("analyst-inference"));
}

