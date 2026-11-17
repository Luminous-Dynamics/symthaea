//! Structural validation for the biologically inspired manipulator HLB fixture corpus.
//!
//! These tests validate the machine-readable contract scaffold only. They do
//! not execute robot dynamics, local reflexes, tactile sensing, HLB inference,
//! actuator admission, or physical safety behavior.

use std::collections::BTreeSet;

use serde_json::Value;

const FIXTURE_JSON: &str =
    include_str!("fixtures/hlb_bio_contract_fixtures_v0_1.json");

fn fixture_set() -> Value {
    serde_json::from_str(FIXTURE_JSON).expect("M1 fixture corpus must be valid JSON")
}

#[test]
fn fixture_corpus_has_stable_schema_and_unique_cases() {
    let root = fixture_set();

    assert_eq!(
        root["schema_id"].as_str(),
        Some("org.luminousdynamics.symthaea.manip_hlb_bio_fixture_set.v0_1")
    );
    assert_eq!(root["version"].as_str(), Some("0.1.0"));
    assert_eq!(root["status"].as_str(), Some("contract-scaffold-structural-test-source-added-not-run"));

    let cases = root["cases"]
        .as_array()
        .expect("M1 fixture corpus must contain a cases array");
    assert_eq!(cases.len(), 12, "unexpected M1 fixture count");

    let mut ids = BTreeSet::new();
    for case in cases {
        let id = case["id"]
            .as_str()
            .expect("every M1 fixture must have a string id");
        assert!(ids.insert(id), "duplicate M1 fixture id: {id}");
        assert!(
            case["purpose"].as_str().is_some_and(|purpose| !purpose.is_empty()),
            "fixture {id} must declare its purpose"
        );
        assert!(
            case["inputs"].is_object(),
            "fixture {id} must declare an input object"
        );
        assert!(
            case["expected"].is_object(),
            "fixture {id} must declare an expected-result object"
        );
        assert!(
            case["invariants"].as_array().is_some_and(|items| !items.is_empty()),
            "fixture {id} must declare at least one invariant"
        );
    }
}

#[test]
fn fixture_contract_keeps_epistemic_and_authority_domains_separate() {
    let root = fixture_set();
    let semantics = &root["semantics"];

    for key in [
        "plant_clock_authoritative",
        "missing_is_not_zero",
        "requested_is_not_applied",
        "prediction_is_not_measurement",
        "hdc_similarity_is_not_probability",
        "cognitive_readiness_is_not_plant_safety",
        "local_fallback_is_independent_of_hlb",
        "unverified_memory_is_not_executable_procedure",
    ] {
        assert_eq!(
            semantics[key].as_bool(),
            Some(true),
            "required semantic invariant must remain true: {key}"
        );
    }
    assert_eq!(semantics["proposal_default"].as_str(), Some("ABSTAIN"));
    assert_eq!(root["disposition_vocabulary"][0].as_str(), Some("ACCEPT_FOR_EVALUATION"));
    assert!(root["disposition_vocabulary"]
        .as_array()
        .is_some_and(|items| items.iter().any(|item| item.as_str() == Some("VETO_PHYSICAL"))));
}

#[test]
fn critical_negative_controls_remain_explicit() {
    let root = fixture_set();
    let cases = root["cases"].as_array().expect("cases array");
    let by_id: std::collections::BTreeMap<_, _> = cases
        .iter()
        .map(|case| (case["id"].as_str().expect("fixture id"), case))
        .collect();

    let missing = by_id["M1-001-missing-is-not-zero"];
    assert_eq!(missing["expected"]["region_b"].as_str(), Some("UNAVAILABLE"));
    assert_eq!(missing["expected"]["aggregate_contact_claim"].as_str(), Some("ABSTAIN"));

    let stale = by_id["M1-002-stale-contact-observation"];
    assert_eq!(stale["expected"]["disposition"].as_str(), Some("REJECT_STALE"));

    let wrong_frame = by_id["M1-003-frame-realization-mismatch"];
    assert_eq!(wrong_frame["expected"]["residual"].as_str(), Some("NOT_COMPUTABLE"));

    let wrong_memory = by_id["M1-005-wrong-configuration-memory"];
    assert_eq!(wrong_memory["expected"]["memory_use"].as_str(), Some("DIAGNOSTIC_ONLY"));
    assert_eq!(wrong_memory["expected"]["disposition"].as_str(), Some("REJECT_IDENTITY"));

    let late_hlb = by_id["M1-008-hlb-timeout"];
    assert_eq!(late_hlb["expected"]["proposal"].as_str(), Some("REJECTED_EXPIRED"));
    assert_eq!(late_hlb["expected"]["disposition"].as_str(), Some("FALLBACK_LOCAL"));

    let physical_veto = by_id["M1-009-independent-physical-veto"];
    assert_eq!(physical_veto["expected"]["disposition"].as_str(), Some("VETO_PHYSICAL"));

    let time_separation = by_id["M1-011-cognition-cadence-does-not-advance-plant"];
    assert_eq!(
        time_separation["expected"]["plant_elapsed_ns_each_run"].as_u64(),
        Some(2_000_000)
    );
}
