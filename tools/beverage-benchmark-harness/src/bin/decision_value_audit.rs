// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Audits a predeclared matched process-intervention comparison.
//!
//! This does not prove randomization, source authenticity, independence, or
//! causality. It makes the comparison boundary, quality gates, effect metric,
//! and uncertainty calculation explicit before a plant claim is made.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const SCHEMA_VERSION: &str = "beverage-decision-value-v1";
const BOOTSTRAP_REPLICATES: usize = 5_000;
const CLAIM_CEILING: &str = "This audit checks declared matched comparisons, resource intensities per 1,000 litres of saleable quality-conforming output, non-compensatory quality/safety gates, a predeclared primary target, and a deterministic 95% percentile bootstrap interval over declared independent units. It does not authenticate source measurements, prove independent units, validate randomization or matching, establish causality, establish statistical power, qualify a food-safety method, or prove a production saving.";

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum Design {
    BlockRandomized,
    MatchedControl,
    UncontrolledPrePost,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum EvidenceClass {
    SyntheticContractFixture,
    MeasuredProspective,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum ResourceMetric {
    Water,
    Energy,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionManifest {
    schema_version: String,
    evidence_class: EvidenceClass,
    design: Design,
    dataset_id: String,
    dataset_sha256: String,
    primary_resource_metric: ResourceMetric,
    required_primary_reduction_pct: f64,
    maximum_secondary_resource_increase_pct: f64,
    minimum_independent_units: usize,
    comparisons: Vec<MatchedComparison>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MatchedComparison {
    comparison_id: String,
    independent_unit_id: String,
    product_stratum: String,
    baseline_period_id: String,
    intervention_period_id: String,
    baseline_meter_boundary_ref: String,
    intervention_meter_boundary_ref: String,
    baseline_functional_unit: String,
    intervention_functional_unit: String,
    baseline_product_mix_ref: String,
    intervention_product_mix_ref: String,
    baseline_output_basis: String,
    intervention_output_basis: String,
    baseline_saleable_output_l: f64,
    intervention_saleable_output_l: f64,
    baseline_water_l: f64,
    intervention_water_l: f64,
    baseline_energy_kwh: f64,
    intervention_energy_kwh: f64,
    baseline_quality_gate_passed: bool,
    intervention_quality_gate_passed: bool,
    baseline_safety_gate_passed: bool,
    intervention_safety_gate_passed: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct EffectSummary {
    baseline_mean_per_1000_l_saleable: f64,
    intervention_mean_per_1000_l_saleable: f64,
    mean_reduction_pct: f64,
    percentile_bootstrap_95pct_low: f64,
    percentile_bootstrap_95pct_high: f64,
    bootstrap_replicates: usize,
}

#[derive(Debug, Serialize)]
struct DecisionReceipt {
    receipt_schema: &'static str,
    status: &'static str,
    manifest_sha256: String,
    evidence_class: Option<EvidenceClass>,
    design: Option<Design>,
    dataset_id: Option<String>,
    dataset_sha256_format_valid: bool,
    primary_resource_metric: Option<ResourceMetric>,
    independent_unit_count: usize,
    comparison_count: usize,
    design_gate_passed: bool,
    unique_comparison_ids: bool,
    unique_independent_units: bool,
    meter_boundaries_match: bool,
    functional_units_match: bool,
    product_mix_matches: bool,
    saleable_output_basis_passed: bool,
    positive_finite_measurements: bool,
    quality_gates_passed: bool,
    safety_gates_passed: bool,
    minimum_independent_units_met: bool,
    primary_effect: Option<EffectSummary>,
    secondary_effect: Option<EffectSummary>,
    primary_target_gate_passed: bool,
    secondary_resource_guardrail_passed: bool,
    hard_gates_passed: bool,
    validation_errors: Vec<String>,
    claim_ceiling: &'static str,
}

#[derive(Debug, Default)]
struct Checks {
    comparison_ids: BTreeSet<String>,
    independent_units: BTreeSet<String>,
    product_strata: BTreeSet<String>,
    cohort_meter_boundaries: BTreeSet<String>,
    cohort_functional_units: BTreeSet<String>,
    cohort_product_mixes: BTreeSet<String>,
    meter_boundaries_match: bool,
    functional_units_match: bool,
    product_mix_matches: bool,
    saleable_output_basis_passed: bool,
    positive_finite_measurements: bool,
    quality_gates_passed: bool,
    safety_gates_passed: bool,
}

fn hex_sha256(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}")
            .expect("writing to a String cannot fail");
    }
    output
}

fn digest_is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn clean(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

fn parse_failure_receipt(bytes: &[u8], error: String) -> DecisionReceipt {
    DecisionReceipt {
        receipt_schema: "beverage-decision-value-receipt-v1",
        status: "FAIL",
        manifest_sha256: hex_sha256(bytes),
        evidence_class: None,
        design: None,
        dataset_id: None,
        dataset_sha256_format_valid: false,
        primary_resource_metric: None,
        independent_unit_count: 0,
        comparison_count: 0,
        design_gate_passed: false,
        unique_comparison_ids: false,
        unique_independent_units: false,
        meter_boundaries_match: false,
        functional_units_match: false,
        product_mix_matches: false,
        saleable_output_basis_passed: false,
        positive_finite_measurements: false,
        quality_gates_passed: false,
        safety_gates_passed: false,
        minimum_independent_units_met: false,
        primary_effect: None,
        secondary_effect: None,
        primary_target_gate_passed: false,
        secondary_resource_guardrail_passed: false,
        hard_gates_passed: false,
        validation_errors: vec![format!("manifest_parse_error:{error}")],
        claim_ceiling: CLAIM_CEILING,
    }
}

fn paired_reduction_pct(baseline: f64, intervention: f64) -> f64 {
    100.0 * (baseline - intervention) / baseline
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    let index = ((sorted_values.len() - 1) as f64 * p).round() as usize;
    sorted_values[index]
}

fn bootstrap_mean_interval(values: &[f64], seed: u64) -> (f64, f64) {
    let mut state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
    let mut samples = Vec::with_capacity(BOOTSTRAP_REPLICATES);

    // A tiny deterministic xorshift64* generator keeps receipts reproducible
    // without fetching a random source or adding a runtime dependency.
    let next_index = |state: &mut u64| -> usize {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        let value = (*state).wrapping_mul(0x2545f4914f6cdd1d);
        (value as usize) % values.len()
    };

    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut total = 0.0;
        for _ in 0..values.len() {
            total += values[next_index(&mut state)];
        }
        samples.push(total / values.len() as f64);
    }

    samples.sort_by(f64::total_cmp);
    (percentile(&samples, 0.025), percentile(&samples, 0.975))
}

fn effect_summary(
    baseline: &[f64],
    intervention: &[f64],
    manifest_digest: &str,
    salt: u64,
) -> EffectSummary {
    let reductions = baseline
        .iter()
        .zip(intervention)
        .map(|(before, after)| paired_reduction_pct(*before, *after))
        .collect::<Vec<_>>();
    let mut seed_bytes = [0_u8; 8];
    let raw_digest = Sha256::digest(format!("{manifest_digest}:{salt}").as_bytes());
    seed_bytes.copy_from_slice(&raw_digest[..8]);
    let (low, high) = bootstrap_mean_interval(&reductions, u64::from_le_bytes(seed_bytes));

    EffectSummary {
        baseline_mean_per_1000_l_saleable: mean(baseline),
        intervention_mean_per_1000_l_saleable: mean(intervention),
        mean_reduction_pct: mean(&reductions),
        percentile_bootstrap_95pct_low: low,
        percentile_bootstrap_95pct_high: high,
        bootstrap_replicates: BOOTSTRAP_REPLICATES,
    }
}

fn audit_manifest_bytes(bytes: &[u8]) -> DecisionReceipt {
    let manifest: DecisionManifest = match serde_json::from_slice(bytes) {
        Ok(manifest) => manifest,
        Err(error) => return parse_failure_receipt(bytes, error.to_string()),
    };

    let digest = hex_sha256(bytes);
    let mut validation_errors = Vec::new();

    if manifest.schema_version != SCHEMA_VERSION {
        validation_errors.push("schema_version_mismatch".to_owned());
    }
    if !clean(&manifest.dataset_id) {
        validation_errors.push("invalid_dataset_id".to_owned());
    }
    let dataset_sha256_format_valid = digest_is_lower_hex_sha256(&manifest.dataset_sha256);
    if !dataset_sha256_format_valid {
        validation_errors.push("dataset_sha256_must_be_64_lowercase_hex_characters".to_owned());
    }
    if !manifest.required_primary_reduction_pct.is_finite()
        || manifest.required_primary_reduction_pct <= 0.0
        || manifest.required_primary_reduction_pct >= 100.0
    {
        validation_errors.push("required_primary_reduction_pct_must_be_between_0_and_100".to_owned());
    }
    if !manifest.maximum_secondary_resource_increase_pct.is_finite()
        || manifest.maximum_secondary_resource_increase_pct < 0.0
        || manifest.maximum_secondary_resource_increase_pct >= 100.0
    {
        validation_errors.push("secondary_resource_guardrail_must_be_between_0_inclusive_and_100_exclusive".to_owned());
    }
    if manifest.minimum_independent_units < 2 {
        validation_errors.push("minimum_independent_units_must_be_at_least_2".to_owned());
    }
    let design_gate_passed = manifest.design != Design::UncontrolledPrePost;
    if !design_gate_passed {
        validation_errors.push("uncontrolled_pre_post_design_cannot_pass_decision_value_gate".to_owned());
    }

    let mut checks = Checks {
        meter_boundaries_match: true,
        functional_units_match: true,
        product_mix_matches: true,
        saleable_output_basis_passed: true,
        positive_finite_measurements: true,
        quality_gates_passed: true,
        safety_gates_passed: true,
        ..Checks::default()
    };

    let mut water_baseline = Vec::new();
    let mut water_intervention = Vec::new();
    let mut energy_baseline = Vec::new();
    let mut energy_intervention = Vec::new();

    for comparison in &manifest.comparisons {
        if !clean(&comparison.comparison_id) || !checks.comparison_ids.insert(comparison.comparison_id.clone()) {
            validation_errors.push(format!("empty_or_duplicate_comparison_id:{}", comparison.comparison_id));
        }
        if !clean(&comparison.independent_unit_id) || !checks.independent_units.insert(comparison.independent_unit_id.clone()) {
            validation_errors.push(format!("empty_or_duplicate_independent_unit_id:{}", comparison.independent_unit_id));
        }

        if !clean(&comparison.product_stratum)
            || !clean(&comparison.baseline_period_id)
            || !clean(&comparison.intervention_period_id)
            || comparison.baseline_period_id == comparison.intervention_period_id
        {
            validation_errors.push(format!("invalid_stratum_or_period_pair:{}", comparison.comparison_id));
        }

        checks.meter_boundaries_match &= clean(&comparison.baseline_meter_boundary_ref)
            && comparison.baseline_meter_boundary_ref == comparison.intervention_meter_boundary_ref;
        checks.functional_units_match &= clean(&comparison.baseline_functional_unit)
            && comparison.baseline_functional_unit == comparison.intervention_functional_unit;
        checks.product_mix_matches &= clean(&comparison.baseline_product_mix_ref)
            && comparison.baseline_product_mix_ref == comparison.intervention_product_mix_ref;
        if clean(&comparison.product_stratum) {
            checks.product_strata.insert(comparison.product_stratum.clone());
        }
        if clean(&comparison.baseline_meter_boundary_ref) {
            checks
                .cohort_meter_boundaries
                .insert(comparison.baseline_meter_boundary_ref.clone());
        }
        if clean(&comparison.baseline_functional_unit) {
            checks
                .cohort_functional_units
                .insert(comparison.baseline_functional_unit.clone());
        }
        if clean(&comparison.baseline_product_mix_ref) {
            checks
                .cohort_product_mixes
                .insert(comparison.baseline_product_mix_ref.clone());
        }
        checks.saleable_output_basis_passed &=
            comparison.baseline_output_basis == "SALEABLE_QUALITY_CONFORMING_OUTPUT"
                && comparison.intervention_output_basis == "SALEABLE_QUALITY_CONFORMING_OUTPUT";

        let measurements = [
            comparison.baseline_saleable_output_l,
            comparison.intervention_saleable_output_l,
            comparison.baseline_water_l,
            comparison.intervention_water_l,
            comparison.baseline_energy_kwh,
            comparison.intervention_energy_kwh,
        ];
        checks.positive_finite_measurements &= measurements
            .iter()
            .all(|value| value.is_finite() && *value > 0.0);
        checks.quality_gates_passed &= comparison.baseline_quality_gate_passed
            && comparison.intervention_quality_gate_passed;
        checks.safety_gates_passed &= comparison.baseline_safety_gate_passed
            && comparison.intervention_safety_gate_passed;

        if measurements.iter().any(|value| !value.is_finite() || *value <= 0.0) {
            continue;
        }

        let baseline_volume_k_l = comparison.baseline_saleable_output_l / 1000.0;
        let intervention_volume_k_l = comparison.intervention_saleable_output_l / 1000.0;
        water_baseline.push(comparison.baseline_water_l / baseline_volume_k_l);
        water_intervention.push(comparison.intervention_water_l / intervention_volume_k_l);
        energy_baseline.push(comparison.baseline_energy_kwh / baseline_volume_k_l);
        energy_intervention.push(comparison.intervention_energy_kwh / intervention_volume_k_l);
    }

    checks.meter_boundaries_match &= checks.cohort_meter_boundaries.len() == 1;
    checks.functional_units_match &= checks.cohort_functional_units.len() == 1;
    checks.product_mix_matches &= checks.cohort_product_mixes.len() == 1;
    let single_product_stratum = checks.product_strata.len() == 1;

    if !checks.meter_boundaries_match {
        validation_errors.push("meter_boundary_refs_must_match_within_pairs_and_across_cohort".to_owned());
    }
    if !checks.functional_units_match {
        validation_errors.push("functional_units_must_match_within_pairs_and_across_cohort".to_owned());
    }
    if !checks.product_mix_matches {
        validation_errors.push("product_mix_refs_must_match_within_pairs_and_across_cohort".to_owned());
    }
    if !single_product_stratum {
        validation_errors.push("product_strata_must_not_be_pooled_in_v1".to_owned());
    }
    if !checks.saleable_output_basis_passed {
        validation_errors.push("both_periods_must_use_saleable_quality_conforming_output".to_owned());
    }
    if !checks.positive_finite_measurements {
        validation_errors.push("all_resource_and_saleable_output_values_must_be_finite_and_positive".to_owned());
    }
    if !checks.quality_gates_passed {
        validation_errors.push("quality_gate_failure_is_non_compensatory".to_owned());
    }
    if !checks.safety_gates_passed {
        validation_errors.push("safety_gate_failure_is_non_compensatory".to_owned());
    }

    let normalized_intensities_are_finite = water_baseline
        .iter()
        .chain(&water_intervention)
        .chain(&energy_baseline)
        .chain(&energy_intervention)
        .all(|value| value.is_finite() && *value > 0.0);
    checks.positive_finite_measurements &= normalized_intensities_are_finite;
    if !normalized_intensities_are_finite {
        validation_errors.push("normalized_resource_intensities_must_be_finite_and_positive".to_owned());
    }

    let independent_unit_count = checks.independent_units.len();
    let minimum_independent_units_met =
        independent_unit_count >= manifest.minimum_independent_units
            && manifest.minimum_independent_units >= 2;
    if !minimum_independent_units_met {
        validation_errors.push("minimum_independent_unit_count_not_met".to_owned());
    }
    let unique_comparison_ids = checks.comparison_ids.len() == manifest.comparisons.len()
        && manifest.comparisons.iter().all(|row| clean(&row.comparison_id));
    let unique_independent_units = checks.independent_units.len() == manifest.comparisons.len()
        && manifest.comparisons.iter().all(|row| clean(&row.independent_unit_id));
    if manifest.comparisons.len() < manifest.minimum_independent_units {
        validation_errors.push("comparison_count_below_preregistered_minimum".to_owned());
    }
    if water_baseline.len() != manifest.comparisons.len()
        || energy_baseline.len() != manifest.comparisons.len()
    {
        validation_errors.push("invalid_measurement_rows_prevent_complete_effect_estimate".to_owned());
    }

    let effect_input_valid = validation_errors.is_empty()
        && unique_comparison_ids
        && unique_independent_units
        && checks.meter_boundaries_match
        && checks.functional_units_match
        && checks.product_mix_matches
        && single_product_stratum
        && checks.saleable_output_basis_passed
        && checks.positive_finite_measurements
        && checks.quality_gates_passed
        && checks.safety_gates_passed
        && minimum_independent_units_met
        && design_gate_passed;

    let (water_effect, energy_effect) = if effect_input_valid {
        (
            Some(effect_summary(&water_baseline, &water_intervention, &digest, 1)),
            Some(effect_summary(&energy_baseline, &energy_intervention, &digest, 2)),
        )
    } else {
        (None, None)
    };

    let (primary_effect, secondary_effect) = match manifest.primary_resource_metric {
        ResourceMetric::Water => (water_effect, energy_effect),
        ResourceMetric::Energy => (energy_effect, water_effect),
    };
    let primary_target_gate_passed = primary_effect
        .map(|effect| effect.percentile_bootstrap_95pct_low >= manifest.required_primary_reduction_pct)
        .unwrap_or(false);
    if !primary_target_gate_passed {
        validation_errors.push("primary_resource_reduction_lower_95pct_bound_does_not_meet_preregistered_target".to_owned());
    }
    let secondary_resource_guardrail_passed = secondary_effect
        .map(|effect| effect.percentile_bootstrap_95pct_low >= -manifest.maximum_secondary_resource_increase_pct)
        .unwrap_or(false);
    if !secondary_resource_guardrail_passed {
        validation_errors.push("secondary_resource_increase_exceeds_preregistered_guardrail_or_interval_not_estimable".to_owned());
    }

    let hard_gates_passed = design_gate_passed
        && unique_comparison_ids
        && unique_independent_units
        && checks.meter_boundaries_match
        && checks.functional_units_match
        && checks.product_mix_matches
        && single_product_stratum
        && checks.saleable_output_basis_passed
        && checks.positive_finite_measurements
        && checks.quality_gates_passed
        && checks.safety_gates_passed
        && minimum_independent_units_met;

    let passed = effect_input_valid && primary_target_gate_passed && secondary_resource_guardrail_passed;
    let status = if !passed {
        "FAIL"
    } else {
        match manifest.evidence_class {
            EvidenceClass::SyntheticContractFixture => "SYNTHETIC_CONTRACT_PASS_NOT_EMPIRICAL",
            EvidenceClass::MeasuredProspective => "DECISION_GATE_PASS_PENDING_INDEPENDENT_REVIEW",
        }
    };

    DecisionReceipt {
        receipt_schema: "beverage-decision-value-receipt-v1",
        status,
        manifest_sha256: digest,
        evidence_class: Some(manifest.evidence_class),
        design: Some(manifest.design),
        dataset_id: Some(manifest.dataset_id),
        dataset_sha256_format_valid,
        primary_resource_metric: Some(manifest.primary_resource_metric),
        independent_unit_count,
        comparison_count: manifest.comparisons.len(),
        design_gate_passed,
        unique_comparison_ids,
        unique_independent_units,
        meter_boundaries_match: checks.meter_boundaries_match,
        functional_units_match: checks.functional_units_match,
        product_mix_matches: checks.product_mix_matches,
        saleable_output_basis_passed: checks.saleable_output_basis_passed,
        positive_finite_measurements: checks.positive_finite_measurements,
        quality_gates_passed: checks.quality_gates_passed,
        safety_gates_passed: checks.safety_gates_passed,
        minimum_independent_units_met,
        primary_effect,
        secondary_effect,
        primary_target_gate_passed,
        secondary_resource_guardrail_passed,
        hard_gates_passed,
        validation_errors,
        claim_ceiling: CLAIM_CEILING,
    }
}

fn parse_args() -> Result<(PathBuf, PathBuf), String> {
    let mut input = None;
    let mut output = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing_value_for_{flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--out" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown_argument_{flag}")),
        }
    }

    Ok((
        input.ok_or_else(|| "missing_--input".to_owned())?,
        output.ok_or_else(|| "missing_--out".to_owned())?,
    ))
}

fn run() -> Result<ExitCode, String> {
    let (input_path, output_path) = parse_args()?;
    let input_bytes = fs::read(&input_path)
        .map_err(|error| format!("manifest_read_failed:{error}"))?;
    let receipt = audit_manifest_bytes(&input_bytes);
    let passed = receipt.status != "FAIL";
    let serialized = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("receipt_serialization_failed:{error}"))?;
    fs::write(&output_path, serialized)
        .map_err(|error| format!("receipt_write_failed:{error}"))?;
    if passed {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::FAILURE)
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("beverage-decision-value-audit:{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &[u8] =
        include_bytes!("../../../../docs/research/beverage-decision-value-synthetic-v1.json");

    fn fixture_value() -> Value {
        serde_json::from_slice(FIXTURE)
            .expect("checked-in fixture must be valid JSON")
    }

    fn audit_value(value: Value) -> DecisionReceipt {
        let bytes = serde_json::to_vec(&value)
            .expect("test fixture serialization must succeed");
        audit_manifest_bytes(&bytes)
    }

    #[test]
    fn synthetic_paired_fixture_passes_without_claiming_empirical_success() {
        let receipt = audit_manifest_bytes(FIXTURE);
        assert_eq!(receipt.status, "SYNTHETIC_CONTRACT_PASS_NOT_EMPIRICAL", "{:?}", receipt.validation_errors);
        assert!(receipt.hard_gates_passed);
        assert!(receipt.primary_target_gate_passed);
        assert!(receipt.secondary_resource_guardrail_passed);
        assert!(receipt.primary_effect.unwrap().percentile_bootstrap_95pct_low >= 10.0);
    }

    #[test]
    fn uncontrolled_pre_post_cannot_pass() {
        let mut value = fixture_value();
        value["design"] = "UNCONTROLLED_PRE_POST".into();
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.design_gate_passed);
    }

    #[test]
    fn mismatched_meter_boundary_fails_closed() {
        let mut value = fixture_value();
        value["comparisons"][0]["intervention_meter_boundary_ref"] = "packaging_excluded".into();
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.meter_boundaries_match);
    }

    #[test]
    fn mismatched_product_mix_fails_closed() {
        let mut value = fixture_value();
        value["comparisons"][1]["intervention_product_mix_ref"] = "different_mix".into();
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.product_mix_matches);
    }

    #[test]
    fn total_output_cannot_replace_saleable_quality_conforming_output() {
        let mut value = fixture_value();
        value["comparisons"][0]["intervention_output_basis"] = "TOTAL_OUTPUT".into();
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.saleable_output_basis_passed);
    }

    #[test]
    fn quality_gate_failure_is_not_offset_by_savings() {
        let mut value = fixture_value();
        value["comparisons"][0]["intervention_quality_gate_passed"] = false.into();
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.quality_gates_passed);
    }

    #[test]
    fn duplicate_independent_units_are_not_extra_evidence() {
        let mut value = fixture_value();
        let first = value["comparisons"][0]["independent_unit_id"].clone();
        value["comparisons"][7]["independent_unit_id"] = first;
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.unique_independent_units);
    }

    #[test]
    fn pooling_different_product_strata_fails_closed() {
        let mut value = fixture_value();
        value["comparisons"][2]["product_stratum"] = "beer_stout".into();
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(receipt
            .validation_errors
            .iter()
            .any(|error| error == "product_strata_must_not_be_pooled_in_v1"));
    }

    #[test]
    fn secondary_resource_guardrail_is_non_compensatory() {
        let mut value = fixture_value();
        for row in value["comparisons"].as_array_mut().unwrap() {
            let baseline = row["baseline_energy_kwh"].as_f64().unwrap();
            row["intervention_energy_kwh"] = (baseline * 1.05).into();
        }
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.secondary_resource_guardrail_passed);
    }

    #[test]
    fn unsupported_dataset_digest_fails_closed() {
        let mut value = fixture_value();
        value["dataset_sha256"] = "z".repeat(64).into();
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.dataset_sha256_format_valid);
    }

    #[test]
    fn insufficient_declared_independent_units_fails() {
        let mut value = fixture_value();
        value["minimum_independent_units"] = 10.into();
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.minimum_independent_units_met);
    }

    #[test]
    fn malformed_json_emits_fail_receipt() {
        let receipt = audit_manifest_bytes(b"{");
        assert_eq!(receipt.status, "FAIL");
        assert!(receipt.validation_errors[0].starts_with("manifest_parse_error:"));
    }
}
