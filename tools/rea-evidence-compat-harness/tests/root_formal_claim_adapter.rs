#[path = "../../../src/rea_compat.rs"]
mod rea_compat;

use rea_compat::formal::{
    FORMAL_PROVIDER_OBSERVATION_SCHEMA, FormalMappingErrorKind, FormalSemanticPolarity,
    export_formal_provider_observation, parse_formal_provider_observation_json,
    validate_formal_provider_observation,
};
use rea_compat::{
    ClaimDisposition, ClaimPredicate, ClaimRequest, ClaimScope, analysis_profile_digest,
    compute_evidence_id, evaluate_claim, verify_evidence_json,
};
use serde_json::{Value, json};

fn fixture_bytes() -> &'static [u8] {
    include_bytes!(
        "../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
    )
}

fn provider_claim(evidence_id: String) -> ClaimRequest {
    ClaimRequest {
        schema: rea_compat::CLAIM_SCHEMA.to_owned(),
        claim_id: "formal-export-fixture-0001".to_owned(),
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

fn second_report(first: &rea_compat::VerifiedEvidence) -> (Vec<u8>, rea_compat::VerifiedEvidence) {
    let mut value = first.value.clone();
    value["provider"]["id"] = json!("independent-report-2");
    value["provider"]["name"] = json!("Independent Report 2");
    value["analysis_profile"]["provider"] = value["provider"].clone();
    value["analysis_profile"]["digest"] =
        json!(analysis_profile_digest(&value["analysis_profile"]).unwrap());
    value["normalized_result"]["status"] = json!("partial");
    value["evidence_id"] = json!(compute_evidence_id(&value).unwrap());
    let bytes = serde_json::to_vec(&value).unwrap();
    let verified = verify_evidence_json(&bytes).unwrap();
    (bytes, verified)
}

#[test]
fn supported_provider_observation_exports_only_formal_meta_claim() {
    let evidence = verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let result = evaluate_claim(&claim, std::slice::from_ref(&evidence)).unwrap();

    let formal = export_formal_provider_observation(&result).unwrap();
    assert_eq!(formal.schema, FORMAL_PROVIDER_OBSERVATION_SCHEMA);
    assert_eq!(
        formal.semantic_polarity,
        FormalSemanticPolarity::PositiveSupport
    );
    assert_eq!(
        formal.rea_disposition,
        ClaimDisposition::SupportedWithinScope
    );
    assert_eq!(
        formal.binding.claim_request_sha256,
        result.claim_request_sha256
    );
    assert_eq!(formal.binding.derivation_sha256, result.derivation_sha256);
    assert_eq!(formal.binding.requested_evidence_ids, claim.evidence_ids);
    assert_eq!(formal.binding.evaluator_rule_generation, 0);
    assert_eq!(
        formal.binding.predicate,
        ClaimPredicate::ProviderReportedObservation
    );
    validate_formal_provider_observation(&formal).unwrap();
}

#[test]
fn refuted_provider_observation_exports_meta_claim_counterevidence_not_artifact_false() {
    let evidence = verify_evidence_json(fixture_bytes()).unwrap();
    let mut claim = provider_claim(evidence.evidence_id.clone());
    claim.asserted_value["status"] = json!("partial");
    let result = evaluate_claim(&claim, &[evidence]).unwrap();
    assert_eq!(result.disposition, ClaimDisposition::RefutedWithinScope);

    let formal = export_formal_provider_observation(&result).unwrap();
    assert_eq!(
        formal.semantic_polarity,
        FormalSemanticPolarity::SemanticCounterevidence
    );
    assert_eq!(formal.rea_disposition, ClaimDisposition::RefutedWithinScope);
    assert!(
        formal
            .claim_ceiling
            .iter()
            .any(|item| item.contains("meta-claim"))
    );
}

#[test]
fn contradictory_provider_observations_export_no_semantic_conclusion() {
    let first = verify_evidence_json(fixture_bytes()).unwrap();
    let (second_bytes, second) = second_report(&first);
    let mut claim = provider_claim(first.evidence_id.clone());
    claim.evidence_ids.push(second.evidence_id.clone());
    let result = evaluate_claim(&claim, &[first, second]).unwrap();
    assert_eq!(result.disposition, ClaimDisposition::Conflicted);

    let formal = export_formal_provider_observation(&result).unwrap();
    assert_eq!(
        formal.semantic_polarity,
        FormalSemanticPolarity::NoSemanticConclusion
    );
    assert_eq!(formal.rea_evaluation.evidence_assessments.len(), 2);
    assert!(
        formal
            .rea_evaluation
            .evidence_assessments
            .iter()
            .any(|item| { item.kind == rea_compat::EvidenceAssessmentKind::Supports })
    );
    assert!(
        formal
            .rea_evaluation
            .evidence_assessments
            .iter()
            .any(|item| { item.kind == rea_compat::EvidenceAssessmentKind::Contradicts })
    );
    assert!(!second_bytes.is_empty());
}

#[test]
fn missing_premise_exports_underdetermined_without_positive_or_negative_polarity() {
    let missing = format!("ev_{}", "b".repeat(64));
    let claim = provider_claim(missing);
    let result = evaluate_claim(&claim, &[]).unwrap();
    assert_eq!(result.disposition, ClaimDisposition::Underdetermined);

    let formal = export_formal_provider_observation(&result).unwrap();
    assert_eq!(
        formal.semantic_polarity,
        FormalSemanticPolarity::NoSemanticConclusion
    );
    assert_eq!(
        formal.rea_evaluation.missing_evidence_ids,
        claim.evidence_ids
    );
}

#[test]
fn unimplemented_predicate_cannot_enter_provider_observation_formal_class() {
    let evidence = verify_evidence_json(fixture_bytes()).unwrap();
    let mut claim = provider_claim(evidence.evidence_id.clone());
    claim.predicate = ClaimPredicate::ProducerAuthenticity;
    claim.source_predicate_type = None;
    let result = evaluate_claim(&claim, &[evidence]).unwrap();
    assert_eq!(result.disposition, ClaimDisposition::Unverified);

    let error = export_formal_provider_observation(&result).unwrap_err();
    assert_eq!(error.kind, FormalMappingErrorKind::UnsupportedPredicate);
}

#[test]
fn altered_request_digest_or_scope_cannot_be_exported_as_same_formal_binding() {
    let evidence = verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let result = evaluate_claim(&claim, &[evidence]).unwrap();

    let mut altered_digest = result.clone();
    altered_digest.claim_request_sha256 = "b".repeat(64);
    let error = export_formal_provider_observation(&altered_digest).unwrap_err();
    assert_eq!(error.kind, FormalMappingErrorKind::InvalidDigest);

    let mut altered_scope = result;
    altered_scope.scope.platform = Some("android".to_owned());
    let error = export_formal_provider_observation(&altered_scope).unwrap_err();
    assert_eq!(error.kind, FormalMappingErrorKind::InvalidDigest);
}

#[test]
fn reference_partition_rejects_duplicate_assessment_and_unrequested_ids() {
    let evidence = verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let result = evaluate_claim(&claim, &[evidence]).unwrap();
    let mut altered = result;
    altered
        .evidence_assessments
        .push(altered.evidence_assessments[0].clone());

    let error = export_formal_provider_observation(&altered).unwrap_err();
    assert_eq!(error.kind, FormalMappingErrorKind::InvalidReferences);
}

#[test]
fn formal_envelope_serialization_round_trips_and_rejects_unknown_fields() {
    let evidence = verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let result = evaluate_claim(&claim, &[evidence]).unwrap();
    let formal = export_formal_provider_observation(&result).unwrap();

    let bytes = serde_json::to_vec(&formal).unwrap();
    let parsed = parse_formal_provider_observation_json(&bytes).unwrap();
    assert_eq!(parsed, formal);

    let mut value: Value = serde_json::from_slice(&bytes).unwrap();
    value["unreviewed_authority"] = json!("trusted");
    let altered = serde_json::to_vec(&value).unwrap();
    let error = parse_formal_provider_observation_json(&altered).unwrap_err();
    assert_eq!(error.kind, FormalMappingErrorKind::InvalidJson);

    let text = String::from_utf8(bytes).unwrap();
    let duplicate = text.replacen(
        "\"schema\":\"luminous.rea.formal-provider-observation.v1\"",
        "\"schema\":\"luminous.rea.formal-provider-observation.v1\",\"schema\":\"forged\"",
        1,
    );
    let error = parse_formal_provider_observation_json(duplicate.as_bytes()).unwrap_err();
    assert_eq!(error.kind, FormalMappingErrorKind::InvalidJson);
}

#[test]
fn formal_envelope_byte_limit_is_enforced() {
    let oversized = vec![b' '; rea_compat::formal::MAX_FORMAL_PROVIDER_OBSERVATION_JSON_BYTES + 1];
    let error = parse_formal_provider_observation_json(&oversized).unwrap_err();
    assert_eq!(error.kind, FormalMappingErrorKind::InputTooLarge);
}

#[test]
fn assumptions_and_inapplicable_scope_never_export_positive_or_negative_polarity() {
    let evidence = verify_evidence_json(fixture_bytes()).unwrap();

    let mut assumed = provider_claim(evidence.evidence_id.clone());
    assumed
        .assumptions
        .push("the scanner database is complete".to_owned());
    let result = evaluate_claim(&assumed, std::slice::from_ref(&evidence)).unwrap();
    assert_eq!(result.disposition, ClaimDisposition::Unverified);
    let formal = export_formal_provider_observation(&result).unwrap();
    assert_eq!(
        formal.semantic_polarity,
        FormalSemanticPolarity::NoSemanticConclusion
    );

    let mut wrong_subject = provider_claim(evidence.evidence_id.clone());
    wrong_subject.scope.subject_sha256 =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned();
    let result = evaluate_claim(&wrong_subject, &[evidence]).unwrap();
    assert_eq!(result.disposition, ClaimDisposition::Inapplicable);
    let formal = export_formal_provider_observation(&result).unwrap();
    assert_eq!(
        formal.semantic_polarity,
        FormalSemanticPolarity::NoSemanticConclusion
    );
}

#[test]
fn formal_rule_generation_is_a_versioned_wire_value_and_schema_contract() {
    let evidence = verify_evidence_json(fixture_bytes()).unwrap();
    let claim = provider_claim(evidence.evidence_id.clone());
    let result = evaluate_claim(&claim, &[evidence]).unwrap();
    let mut formal = export_formal_provider_observation(&result).unwrap();
    assert_eq!(formal.binding.evaluator_rule_generation, 0);

    formal.binding.evaluator_rule_generation = 1;
    let error = validate_formal_provider_observation(&formal).unwrap_err();
    assert_eq!(error.kind, FormalMappingErrorKind::InvalidEvaluation);

    let schema: Value = serde_json::from_str(include_str!(
        "../../../docs/formal/rea-formal-provider-observation-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(
        schema["properties"]["binding"]["properties"]["evaluator_rule_generation"]["const"],
        json!(0)
    );
}
