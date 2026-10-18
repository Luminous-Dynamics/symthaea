//! Structural and hand-computable contract checks for the v1 decision-quality corpus.
//!
//! These tests qualify the fixture itself. They do not run a decision policy,
//! prove VOI/calibration, or establish real-world culinary outcomes.

use serde_json::Value;
use std::collections::HashSet;

const CORPUS: &str =
    include_str!("../fixtures/culinary-decision-quality-v1/corpus.json");

fn corpus() -> Value {
    serde_json::from_str(CORPUS).expect("decision-quality corpus must be valid JSON")
}

fn case_by_id<'a>(document: &'a Value, id: &str) -> &'a Value {
    document["cases"]
        .as_array()
        .expect("corpus cases must be an array")
        .iter()
        .find(|case| case["id"].as_str() == Some(id))
        .unwrap_or_else(|| panic!("missing required corpus case {id}"))
}

fn candidate_by_id<'a>(case: &'a Value, id: &str) -> &'a Value {
    case["policy_input"]["candidates"]
        .as_array()
        .expect("policy-visible candidates must be an array")
        .iter()
        .find(|candidate| candidate["id"].as_str() == Some(id))
        .unwrap_or_else(|| panic!("missing candidate {id}"))
}

fn number(value: &Value, field: &str) -> i64 {
    value[field]
        .as_i64()
        .unwrap_or_else(|| panic!("expected integer field {field}"))
}

#[test]
fn corpus_is_synthetic_versioned_and_has_unique_case_ids() {
    let document = corpus();
    assert_eq!(document["corpus_id"], "culinary-decision-quality-voi-v1");
    assert_eq!(document["corpus_version"], "1.0.0");
    assert_eq!(document["evidence_class"], "synthetic_contract_fixture");
    assert_eq!(document["not_real_world_data"], true);

    let cases = document["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 12);

    let ids: Vec<&str> = cases
        .iter()
        .map(|case| case["id"].as_str().expect("case ID must be a string"))
        .collect();
    let unique: HashSet<&str> = ids.iter().copied().collect();
    assert_eq!(unique.len(), ids.len(), "corpus case IDs must be unique");

    for case in cases {
        assert!(case["policy_input"].is_object());
        assert!(case["oracle_only_expected"].is_object());
        assert!(!case["policy_input"].get("oracle_only_expected").is_some());
        assert!(matches!(
            case["oracle_only_expected"]["disposition"].as_str(),
            Some("recommend")
                | Some("preserve_choice_set")
                | Some("request_measurement")
                | Some("human_review")
                | Some("no_feasible_candidate")
        ));
    }
}

#[test]
fn stale_price_that_can_change_the_decision_requests_a_measurement() {
    let document = corpus();
    let case = case_by_id(&document, "DQ-001-stale-price-can-flip-choice");
    let gamma = candidate_by_id(case, "basket_gamma");

    assert_eq!(gamma["cost"]["freshness"], "stale");
    assert_eq!(gamma["cost"]["status"], "observed");
    assert_eq!(
        case["oracle_only_expected"]["disposition"],
        "request_measurement"
    );
    assert_eq!(
        case["oracle_only_expected"]["measurement_id"],
        "remeasure_gamma_item_prices"
    );
    assert!(case["oracle_only_expected"]["must_not"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value == "treat_stale_74_as_current_price"));
}

#[test]
fn unknown_allergen_gate_requires_review_not_score_compensation() {
    let document = corpus();
    let case = case_by_id(&document, "DQ-002-unknown-allergen-gate");
    for candidate in case["policy_input"]["candidates"].as_array().unwrap() {
        assert_eq!(candidate["hard_gates"]["allergen"], "unknown");
    }
    assert_eq!(
        case["oracle_only_expected"]["disposition"],
        "human_review"
    );
    assert_eq!(
        case["oracle_only_expected"]["selected_candidate_ids"].as_array().unwrap().len(),
        0
    );
}

#[test]
fn a_stably_dominating_candidate_does_not_trigger_an_unnecessary_query() {
    let document = corpus();
    let case = case_by_id(&document, "DQ-004-stable-dominance-do-not-overquery");
    let alpha = candidate_by_id(case, "basket_alpha");
    let beta = candidate_by_id(case, "basket_beta");
    let alpha_scenarios = alpha["scenario_metrics"].as_object().unwrap();
    let beta_scenarios = beta["scenario_metrics"].as_object().unwrap();

    assert_eq!(alpha_scenarios.len(), beta_scenarios.len());
    for (scenario_id, alpha_metrics) in alpha_scenarios {
        let beta_metrics = &beta_scenarios[scenario_id];
        assert!(number(alpha_metrics, "cost") <= number(beta_metrics, "cost"));
        assert!(
            number(alpha_metrics, "prep_minutes") <= number(beta_metrics, "prep_minutes")
        );
        assert!(
            number(alpha_metrics, "cost") < number(beta_metrics, "cost")
                || number(alpha_metrics, "prep_minutes") < number(beta_metrics, "prep_minutes")
        );
    }

    assert_eq!(
        case["oracle_only_expected"]["disposition"],
        "recommend"
    );
    assert_eq!(
        case["oracle_only_expected"]["selected_candidate_ids"][0],
        "basket_alpha"
    );
    assert!(case["oracle_only_expected"]["measurement_id"].is_null());
}

#[test]
fn worst_group_affordability_constraint_beats_the_aggregate_average() {
    let document = corpus();
    let case = case_by_id(&document, "DQ-006-worst-group-floor-not-average");
    let alpha = candidate_by_id(case, "portfolio_alpha");
    let beta = candidate_by_id(case, "portfolio_beta");

    assert!(
        number(&alpha["group_costs_zar"], "group_b")
            > number(&alpha["maximum_per_group_zar"], "group_b")
    );
    for (group, value) in beta["group_costs_zar"].as_object().unwrap() {
        assert!(
            value.as_i64().unwrap()
                <= beta["maximum_per_group_zar"][group].as_i64().unwrap(),
            "selected portfolio must satisfy the declared floor for {group}"
        );
    }
    assert_eq!(
        case["oracle_only_expected"]["selected_candidate_ids"][0],
        "portfolio_beta"
    );
}

#[test]
fn permutation_case_encodes_a_real_tradeoff_and_preserves_the_choice_set() {
    let document = corpus();
    let case = case_by_id(&document, "DQ-012-input-permutation-invariance");
    let alpha = candidate_by_id(case, "basket_alpha");
    let beta = candidate_by_id(case, "basket_beta");

    for scenario_id in ["baseline", "shock"] {
        let alpha_metrics = &alpha["scenario_metrics"][scenario_id];
        let beta_metrics = &beta["scenario_metrics"][scenario_id];
        assert!(
            number(alpha_metrics, "cost") > number(beta_metrics, "cost"),
            "beta is cheaper in scenario {scenario_id}"
        );
        assert!(
            number(alpha_metrics, "prep_minutes") < number(beta_metrics, "prep_minutes"),
            "alpha is faster in scenario {scenario_id}"
        );
    }
    assert_eq!(
        case["oracle_only_expected"]["disposition"],
        "preserve_choice_set"
    );
    assert_eq!(
        case["oracle_only_expected"]["canonical_output_equality"],
        true
    );
}

#[test]
fn simulation_and_future_known_at_records_cannot_become_current_observation() {
    let document = corpus();

    let simulation = case_by_id(&document, "DQ-007-simulated-stock-is-not-observed-stock");
    assert_eq!(
        simulation["policy_input"]["candidates"][0]["ingredient_availability"]["tomato"]["status"],
        "simulated"
    );
    assert!(simulation["oracle_only_expected"]["must_not"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value == "label_simulated_stock_as_observed"));

    let temporal = case_by_id(&document, "DQ-009-known-at-cutoff-prevents-time-travel");
    assert_eq!(
        temporal["oracle_only_expected"]["ineligible_source_refs"][0],
        "synthetic:received-later"
    );
    assert_eq!(
        temporal["oracle_only_expected"]["eligible_source_refs"][0],
        "synthetic:known-then"
    );
}
