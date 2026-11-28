//! Semantic guards for the NixWard synthetic security-advisory corpus.
//!
//! These tests validate fixture invariants only. They do not authenticate live
//! host evidence, query advisories, or qualify a production security advisor.

use std::collections::HashSet;

use serde_json::Value;

const SCHEMA_JSON: &str = include_str!(
    "../../../../research/nixward-capability-advisory-v0.1/schemas/security-advisory-corpus-v0.1.schema.json"
);
const CORPUS_JSON: &str = include_str!(
    "../../../../research/nixward-capability-advisory-v0.1/scenarios/security-advisory-corpus-v0.1.json"
);

fn evidence_has(case: &Value, kind: &str) -> bool {
    case["evidence"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|item| item["kind"].as_str() == Some(kind))
}

fn has_fresh_exact_evidence(case: &Value, kind: &str) -> bool {
    case["evidence"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|item| {
            item["kind"].as_str() == Some(kind)
                && item["subject_binding"].as_str() == Some("exact")
                && item["freshness"].as_str() == Some("fresh")
                && matches!(
                    item["integrity"].as_str(),
                    Some("authenticated" | "verified_local")
                )
        })
}

fn expected<'a>(case: &'a Value, field: &str) -> &'a str {
    case["expected"][field]
        .as_str()
        .unwrap_or_else(|| panic!("{} is missing expected.{field}", case["case_id"]))
}

#[test]
fn schema_and_corpus_are_parseable_and_version_bound() {
    let schema: Value = serde_json::from_str(SCHEMA_JSON).expect("schema must be valid JSON");
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("corpus must be valid JSON");

    assert_eq!(
        schema["$schema"].as_str(),
        Some("https://json-schema.org/draft/2020-12/schema")
    );
    assert_eq!(
        schema["$id"].as_str(),
        Some("urn:luminous-dynamics:nixward:security-advisory-corpus:0.1")
    );
    assert_eq!(corpus["schema_version"].as_str(), Some("0.1"));
    assert_eq!(
        corpus["profile_id"].as_str(),
        Some("NIXWARD-CAPABILITY-ADVISORY-V0.1")
    );
    assert_eq!(corpus["synthetic_only"].as_bool(), Some(true));
    assert!(
        corpus["cases"].as_array().is_some_and(|cases| cases.len() >= 12),
        "the frozen corpus must retain at least twelve cases"
    );
}

#[test]
fn case_and_evidence_identities_are_unique_and_complete() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("valid corpus JSON");
    let cases = corpus["cases"].as_array().expect("cases array");
    let mut case_ids = HashSet::new();

    for case in cases {
        let case_id = case["case_id"].as_str().expect("case_id");
        assert!(case_ids.insert(case_id), "duplicate case identity: {case_id}");
        assert!(
            case["cluster_id"].as_str().is_some_and(|value| !value.is_empty()),
            "{case_id}: independent cluster identity is required"
        );
        assert!(
            case["subject"]["host_id"]
                .as_str()
                .is_some_and(|value| !value.is_empty()),
            "{case_id}: exact host identity is required"
        );
        assert!(
            case["subject"]["nixpkgs_revision"]
                .as_str()
                .is_some_and(|value| !value.is_empty()),
            "{case_id}: target nixpkgs revision is required"
        );

        let evidence = case["evidence"].as_array().expect("evidence array");
        let mut evidence_ids = HashSet::new();
        for item in evidence {
            let evidence_id = item["evidence_id"].as_str().expect("evidence_id");
            assert!(
                evidence_ids.insert(evidence_id),
                "{case_id}: duplicate evidence identity {evidence_id}"
            );
            assert!(item["source_revision"].as_str().is_some());
            assert!(item["observed_at"].as_str().is_some());
            assert!(item["subject_binding"].as_str().is_some());
            assert!(item["freshness"].as_str().is_some());
        }
    }
}

#[test]
fn affected_and_not_affected_claims_require_exact_package_evidence() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("valid corpus JSON");
    let cases = corpus["cases"].as_array().expect("cases array");

    for case in cases {
        let applicability = expected(case, "applicability");
        if applicability == "affected" || applicability == "not_affected" {
            assert!(
                case["subject"]["package"].is_object(),
                "{}: definitive package applicability requires a package derivation subject",
                case["case_id"]
            );
            assert!(
                has_fresh_exact_evidence(case, "pinned_flake_lock"),
                "{}: definitive package applicability requires exact pinned source evidence",
                case["case_id"]
            );
            assert!(
                has_fresh_exact_evidence(case, "nix_store_derivation"),
                "{}: definitive package applicability requires exact derivation evidence",
                case["case_id"]
            );
            assert_eq!(
                case["advisory"]["status"].as_str(),
                Some("current"),
                "{}: definitive applicability requires a current advisory record",
                case["case_id"]
            );
        }
    }
}

#[test]
fn confirmed_exposure_requires_live_listener_and_external_reachability() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("valid corpus JSON");
    let cases = corpus["cases"].as_array().expect("cases array");

    for case in cases {
        if expected(case, "exposure") == "confirmed" {
            assert!(
                has_fresh_exact_evidence(case, "runtime_listener"),
                "{}: confirmed exposure requires a fresh exact listener observation",
                case["case_id"]
            );
            assert!(
                has_fresh_exact_evidence(case, "external_reachability_probe"),
                "{}: confirmed exposure requires a fresh exact external probe",
                case["case_id"]
            );
        }
    }
}

#[test]
fn stale_or_mismatched_evidence_cannot_support_a_definitive_safe_claim() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("valid corpus JSON");
    let cases = corpus["cases"].as_array().expect("cases array");

    for case in cases {
        let has_unusable_evidence = case["evidence"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|item| {
                matches!(
                    item["freshness"].as_str(),
                    Some("stale" | "unknown")
                ) || matches!(
                    item["subject_binding"].as_str(),
                    Some("mismatched" | "unbound")
                ) || matches!(item["integrity"].as_str(), Some("unsigned" | "unknown"))
            });

        if has_unusable_evidence {
            assert!(
                expected(case, "applicability") != "not_affected",
                "{}: stale or mismatched evidence cannot prove not_affected",
                case["case_id"]
            );
            assert!(
                expected(case, "exposure") != "not_confirmed",
                "{}: stale or mismatched evidence cannot prove not_confirmed",
                case["case_id"]
            );
        }
    }
}

#[test]
fn negative_reachability_claim_is_scoped_to_observed_path() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("valid corpus JSON");
    let cases = corpus["cases"].as_array().expect("cases array");

    let case = cases
        .iter()
        .find(|case| case["case_id"].as_str() == Some("NWA-016"))
        .expect("NWA-016 negative reachability control");

    assert_eq!(expected(case, "exposure"), "not_confirmed");
    assert!(evidence_has(case, "effective_firewall"));
    assert!(evidence_has(case, "external_reachability_probe"));
    assert!(
        case["expected"]["must_not_claim"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|claim| claim.as_str() == Some("not_reachable_from_every_network")),
        "the negative probe must not be generalized to every network path"
    );
}

#[test]
fn intentional_public_service_is_not_automatically_classified_as_a_defect() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("valid corpus JSON");
    let cases = corpus["cases"].as_array().expect("cases array");

    let case = cases
        .iter()
        .find(|case| case["case_id"].as_str() == Some("NWA-014"))
        .expect("NWA-014 intentional public service control");

    assert_eq!(expected(case, "exposure"), "confirmed");
    assert_eq!(expected(case, "disposition"), "inform_only");
    assert!(evidence_has(case, "user_intent"));
    assert!(evidence_has(case, "external_reachability_probe"));
}
