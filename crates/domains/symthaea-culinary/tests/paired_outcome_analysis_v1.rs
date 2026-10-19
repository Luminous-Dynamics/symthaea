// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use symthaea_culinary::paired_outcome_analysis::analyze_paired_outcomes_v1;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../fixtures/culinary-experiment-v1/synthetic-paired-cooking-study.json"
    ))
    .unwrap()
}

// Test-only transformation exercises the "declared qualified observation" path.
// Values and source locators remain synthetic; the validator intentionally does
// not authenticate sources, and these records must never be cited as experiment results.
fn qualified_observation_contract_fixture() -> Value {
    let mut study = fixture();
    for block in study["blocks"].as_array_mut().unwrap() {
        for run in block["runs"].as_array_mut().unwrap() {
            let run_id = run["run_id"].as_str().unwrap().to_owned();
            let known_at = run["started_at"].as_str().unwrap().to_owned();
            for outcome in run["outcomes"].as_array_mut().unwrap() {
                let outcome_id = outcome["outcome_id"].as_str().unwrap().to_owned();
                outcome["status"] = Value::String("observed".into());
                outcome["evidence"]["source_ref"] =
                    Value::String(format!("urn:test-only:source:{run_id}:{outcome_id}"));
                outcome["evidence"]["source_version"] = Value::String("test-only".into());
                outcome["evidence"]["known_at"] = Value::String(known_at.clone());
                outcome["evidence"]["evidence_status"] = Value::String("qualified".into());
            }
        }
    }
    study
}

fn summary<'a>(
    analysis: &'a symthaea_culinary::paired_outcome_analysis::PairedOutcomeAnalysisV1,
    outcome_id: &str,
) -> &'a symthaea_culinary::paired_outcome_analysis::PairedOutcomeSummaryV1 {
    analysis.summaries.iter().find(|item| item.outcome_id == outcome_id).unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "actual={actual:.15e} expected={expected:.15e}"
    );
}

#[test]
fn synthetic_or_estimated_fixture_cannot_be_analyzed_as_empirical() {
    let error = analyze_paired_outcomes_v1(&fixture()).unwrap_err();
    assert!(error.iter().any(|message| message.contains("estimated")));
    assert!(error.iter().any(|message| message.contains("qualified")));
}

#[test]
fn every_outcome_is_reported_separately_with_paired_differences() {
    let result = analyze_paired_outcomes_v1(&qualified_observation_contract_fixture()).unwrap();
    assert_eq!(result.block_count, 2);
    assert_eq!(result.summaries.len(), 3);
    assert_eq!(result.primary_outcome_id, "prepared_yield_g");
    assert!(result.assumptions.iter().any(|text| text.contains("no p-values")));
    assert!(result.assumptions.iter().any(|text| text.contains("not sampling intervals")));
}

#[test]
fn mean_pairwise_contrast_is_variant_minus_control() {
    let result = analyze_paired_outcomes_v1(&qualified_observation_contract_fixture()).unwrap();
    close(summary(&result, "prepared_yield_g").mean_paired_difference, 6.0);
    close(summary(&result, "discarded_material_g").mean_paired_difference, -3.0);
    close(summary(&result, "energy_input_wh").mean_paired_difference, 4.5);
}

#[test]
fn measurement_bounds_propagate_without_becoming_confidence_intervals() {
    let result = analyze_paired_outcomes_v1(&qualified_observation_contract_fixture()).unwrap();
    let yield_summary = summary(&result, "prepared_yield_g");
    close(yield_summary.mean_measurement_bound_lower, 0.0);
    close(yield_summary.mean_measurement_bound_upper, 12.0);
    let waste_summary = summary(&result, "discarded_material_g");
    close(waste_summary.mean_measurement_bound_lower, -5.0);
    close(waste_summary.mean_measurement_bound_upper, -1.0);
    let energy_summary = summary(&result, "energy_input_wh");
    close(energy_summary.mean_measurement_bound_lower, 0.5);
    close(energy_summary.mean_measurement_bound_upper, 8.5);
}

#[test]
fn outcome_directions_remain_separate_and_no_global_winner_is_emitted() {
    let result = analyze_paired_outcomes_v1(&qualified_observation_contract_fixture()).unwrap();
    let yield_summary = summary(&result, "prepared_yield_g");
    assert_eq!(yield_summary.variant_directional_wins, Some(2));
    assert_eq!(yield_summary.control_directional_wins, Some(0));
    let waste_summary = summary(&result, "discarded_material_g");
    assert_eq!(waste_summary.variant_directional_wins, Some(2));
    let energy_summary = summary(&result, "energy_input_wh");
    assert_eq!(energy_summary.variant_directional_wins, Some(0));
    assert_eq!(energy_summary.control_directional_wins, Some(2));
}

#[test]
fn reversing_block_run_and_outcome_array_order_preserves_report() {
    let original = qualified_observation_contract_fixture();
    let expected = analyze_paired_outcomes_v1(&original).unwrap();
    let mut reversed = original;
    reversed["blocks"].as_array_mut().unwrap().reverse();
    for block in reversed["blocks"].as_array_mut().unwrap() {
        block["runs"].as_array_mut().unwrap().reverse();
        for run in block["runs"].as_array_mut().unwrap() {
            run["outcomes"].as_array_mut().unwrap().reverse();
        }
    }
    let actual = analyze_paired_outcomes_v1(&reversed).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn evidence_spoof_from_synthetic_source_is_rejected() {
    let mut study = qualified_observation_contract_fixture();
    study["blocks"][0]["runs"][0]["outcomes"][0]["evidence"]["source_ref"] =
        Value::String("urn:synthetic:promoted-source".into());
    let error = analyze_paired_outcomes_v1(&study).unwrap_err();
    assert!(error.iter().any(|message| message.contains("cannot mark a synthetic source qualified")));
}

#[test]
fn frozen_analysis_plan_is_mandatory() {
    let mut study = qualified_observation_contract_fixture();
    study["analysis_plan"]["missing_data_policy"] = Value::String("impute-zero".into());
    assert!(analyze_paired_outcomes_v1(&study).is_err());
}

#[test]
fn analysis_plan_cannot_select_an_undeclared_primary_outcome() {
    let mut study = qualified_observation_contract_fixture();
    study["analysis_plan"]["primary_outcome_id"] = Value::String("posthoc-outcome".into());
    let error = analyze_paired_outcomes_v1(&study).unwrap_err();
    assert!(error.iter().any(|message| message.contains("primary_outcome_id")));
}

#[test]
fn target_outcomes_use_distance_to_target_for_directional_tallies() {
    let mut study = qualified_observation_contract_fixture();
    study["outcome_definitions"][0]["direction"] = Value::String("target".into());
    study["outcome_definitions"][0]["target_value"] = serde_json::json!(248.0);
    let result = analyze_paired_outcomes_v1(&study).unwrap();
    let yield_summary = summary(&result, "prepared_yield_g");
    assert_eq!(yield_summary.variant_directional_wins, Some(0));
    assert_eq!(yield_summary.control_directional_wins, Some(2));
}

#[test]
fn observe_only_outcome_does_not_emit_directional_wins() {
    let mut study = qualified_observation_contract_fixture();
    study["outcome_definitions"][1]["direction"] = Value::String("observe".into());
    study["outcome_definitions"][1]["target_value"] = Value::Null;
    let result = analyze_paired_outcomes_v1(&study).unwrap();
    let observed = summary(&result, "discarded_material_g");
    assert_eq!(observed.variant_directional_wins, None);
    assert_eq!(observed.control_directional_wins, None);
    assert_eq!(observed.directional_ties, None);
}

#[test]
fn duplicate_outcome_values_are_not_silently_dropped() {
    let mut study = qualified_observation_contract_fixture();
    let duplicate = study["blocks"][0]["runs"][0]["outcomes"][0].clone();
    study["blocks"][0]["runs"][0]["outcomes"].as_array_mut().unwrap().push(duplicate);
    assert!(analyze_paired_outcomes_v1(&study).is_err());
}

#[test]
fn non_finite_aggregate_difference_fails_closed() {
    let mut study = qualified_observation_contract_fixture();
    for block in study["blocks"].as_array_mut().unwrap() {
        for run in block["runs"].as_array_mut().unwrap() {
            for outcome in run["outcomes"].as_array_mut().unwrap() {
                if outcome["outcome_id"] == "prepared_yield_g" {
                    let value = if run["condition_id"] == "baseline" { 0.0 } else { 1e308 };
                    outcome["value"] = serde_json::json!(value);
                    outcome["lower"] = serde_json::json!(value);
                    outcome["upper"] = serde_json::json!(value);
                }
            }
        }
    }
    assert!(analyze_paired_outcomes_v1(&study).is_err());
}
