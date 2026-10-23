//! Evidence-bounded static energy-ledger verifier for two-wheeler powertrain studies.
//!
//! A pass checks the declared synthetic accounting rules only. It does not
//! establish measured energy efficiency, net energy gain, vehicle performance,
//! physical qualification, or regulatory approval.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const ALLOWED_DECISIONS: &[&str] = &[
    "ACCEPT_LEDGER_ONLY",
    "BLOCK",
    "INCOMPLETE_LEDGER",
    "INDETERMINATE",
    "REJECT_CLAIM",
];

#[derive(Debug, Deserialize)]
struct Manifest {
    manifest_id: String,
    schema_version: String,
    status: String,
    evidence_class: String,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    scenario_id: String,
    class: String,
    title: String,
    facts: Value,
    expected_decision: String,
    expected_reasons: Vec<String>,
    must_not_claim: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CaseResult {
    scenario_id: String,
    class: String,
    title: String,
    expected_decision: String,
    computed_decision: String,
    decision_matches: bool,
    computed_reasons: Vec<String>,
    expected_reasons: Vec<String>,
    computed_input_energy_j: Option<f64>,
    computed_output_energy_j: Option<f64>,
    must_not_claim: Vec<String>,
}

#[derive(Debug, Serialize)]
struct VerificationReport {
    report_version: String,
    manifest_id: String,
    manifest_sha256: String,
    evidence_class: String,
    case_count: usize,
    matched_cases: usize,
    mismatched_cases: usize,
    fixture_conformance_pass: bool,
    claim_ceiling: String,
    cases: Vec<CaseResult>,
}

pub fn run(input: &Path, output: Option<&Path>) -> Result<()> {
    let raw = fs::read(input)
        .with_context(|| format!("reading powertrain ledger manifest {}", input.display()))?;
    let manifest_sha256 = format!("{:x}", Sha256::digest(&raw));
    let manifest: Manifest = serde_json::from_slice(&raw)
        .with_context(|| format!("parsing powertrain ledger manifest {}", input.display()))?;
    validate_manifest(&manifest)?;

    let mut results = Vec::with_capacity(manifest.cases.len());
    for case in &manifest.cases {
        let (computed_decision, computed_reasons) = evaluate_case(&case.facts);
        let (input_total, output_total) = energy_totals(&case.facts);
        results.push(CaseResult {
            scenario_id: case.scenario_id.clone(),
            class: case.class.clone(),
            title: case.title.clone(),
            expected_decision: case.expected_decision.clone(),
            decision_matches: computed_decision == case.expected_decision,
            computed_decision,
            computed_reasons,
            expected_reasons: case.expected_reasons.clone(),
            computed_input_energy_j: input_total,
            computed_output_energy_j: output_total,
            must_not_claim: case.must_not_claim.clone(),
        });
    }

    let matched_cases = results.iter().filter(|case| case.decision_matches).count();
    let mismatched_cases = results.len() - matched_cases;
    let report = VerificationReport {
        report_version: "0.1.0".to_string(),
        manifest_id: manifest.manifest_id,
        manifest_sha256,
        evidence_class: manifest.evidence_class,
        case_count: results.len(),
        matched_cases,
        mismatched_cases,
        fixture_conformance_pass: mismatched_cases == 0,
        claim_ceiling: "Synthetic energy-accounting contract only. No physical efficiency, net-energy, propulsion, safety, or regulatory claim.".to_string(),
        cases: results,
    };
    let encoded = serde_json::to_string_pretty(&report)?;

    if let Some(output) = output {
        if let Some(parent) = output.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("creating report directory {}", parent.display()))?;
            }
        }
        fs::write(output, format!("{encoded}\n"))
            .with_context(|| format!("writing powertrain ledger report {}", output.display()))?;
    }
    println!("{encoded}");

    if mismatched_cases != 0 {
        bail!(
            "powertrain energy-ledger fixture mismatch: {mismatched_cases} of {} decisions differ",
            report.case_count
        );
    }
    Ok(())
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    if manifest.schema_version != "0.1.0" {
        bail!("unsupported manifest schema version: {}", manifest.schema_version);
    }
    if manifest.status != "frozen_expected_cases_not_yet_executed" {
        bail!("unexpected manifest status: {}", manifest.status);
    }
    if manifest.evidence_class != "Synthetic" {
        bail!("powertrain ledger fixture must remain Synthetic evidence");
    }
    if manifest.cases.is_empty() {
        bail!("manifest has no cases");
    }

    let mut ids = BTreeSet::new();
    let mut positive_controls = 0usize;
    let mut negative_controls = 0usize;
    for case in &manifest.cases {
        if case.scenario_id.trim().is_empty() || !ids.insert(case.scenario_id.as_str()) {
            bail!("empty or duplicate scenario ID: {}", case.scenario_id);
        }
        match case.class.as_str() {
            "positive_control" => positive_controls += 1,
            "negative_control" => negative_controls += 1,
            other => bail!("unknown case class {other} in {}", case.scenario_id),
        }
        if !ALLOWED_DECISIONS.contains(&case.expected_decision.as_str()) {
            bail!(
                "unknown expected decision {:?} in {}",
                case.expected_decision,
                case.scenario_id
            );
        }
        if case.expected_reasons.is_empty() || case.must_not_claim.is_empty() {
            bail!("case {} must record rationale and claim ceilings", case.scenario_id);
        }
        if !case.facts.is_object() {
            bail!("case {} facts must be a JSON object", case.scenario_id);
        }
        validate_streams(&case.facts, &case.scenario_id)?;
    }
    if positive_controls == 0 || negative_controls == 0 {
        bail!("manifest needs at least one positive and one negative control");
    }
    Ok(())
}

fn validate_streams(facts: &Value, scenario_id: &str) -> Result<()> {
    for key in ["input_streams", "output_streams"] {
        let Some(streams) = facts.get(key) else {
            continue;
        };
        let Some(streams) = streams.as_array() else {
            bail!("{scenario_id}: {key} must be an array");
        };
        let mut stream_ids = BTreeSet::new();
        for (index, stream) in streams.iter().enumerate() {
            let energy = stream.get("energy_j").and_then(Value::as_f64);
            let uncertainty = stream.get("uncertainty_j").and_then(Value::as_f64);
            if energy.is_none_or(|value| !value.is_finite() || value < 0.0) {
                bail!("{scenario_id}: {key}[{index}] has invalid energy_j");
            }
            if uncertainty.is_none_or(|value| !value.is_finite() || value < 0.0) {
                bail!("{scenario_id}: {key}[{index}] has invalid uncertainty_j");
            }
            let Some(stream_id) = stream.get("id").and_then(Value::as_str) else {
                bail!("{scenario_id}: {key}[{index}] is missing a stream ID");
            };
            if stream_id.is_empty() || !stream_ids.insert(stream_id) {
                bail!("{scenario_id}: {key}[{index}] has an empty or duplicate stream ID");
            }
        }
    }
    Ok(())
}

fn string<'a>(facts: &'a Value, key: &str) -> Option<&'a str> {
    facts.get(key).and_then(Value::as_str)
}

fn boolean(facts: &Value, key: &str) -> Option<bool> {
    facts.get(key).and_then(Value::as_bool)
}

fn energy_totals(facts: &Value) -> (Option<f64>, Option<f64>) {
    fn total(facts: &Value, key: &str) -> Option<f64> {
        let streams = facts.get(key)?.as_array()?;
        let mut sum = 0.0;
        for stream in streams {
            let value = stream.get("energy_j")?.as_f64()?;
            sum += value;
        }
        Some(sum)
    }
    (total(facts, "input_streams"), total(facts, "output_streams"))
}

/// Returns a conservative lower bound over the summed output streams and an
/// upper bound over summed input streams. Summed uncertainties are deliberately
/// conservative; no statistical independence is assumed.
fn balance_bounds(facts: &Value) -> Option<(f64, f64)> {
    fn sum_bounds(facts: &Value, key: &str, lower: bool) -> Option<f64> {
        let streams = facts.get(key)?.as_array()?;
        let mut total = 0.0;
        for stream in streams {
            let energy = stream.get("energy_j")?.as_f64()?;
            let uncertainty = stream.get("uncertainty_j")?.as_f64()?;
            total += if lower {
                (energy - uncertainty).max(0.0)
            } else {
                energy + uncertainty
            };
        }
        Some(total)
    }
    Some((
        sum_bounds(facts, "output_streams", true)?,
        sum_bounds(facts, "input_streams", false)?,
    ))
}

fn evaluate_case(facts: &Value) -> (String, Vec<String>) {
    let claim = string(facts, "claim_kind").unwrap_or("");

    // These claim-specific blocks take precedence over generic missing-field
    // findings so that an obviously invalid conclusion is never softened into
    // a mere completeness warning.
    if claim == "facility_net_electric"
        && (string(facts, "output_basis") == Some("neutron_count_only")
            || boolean(facts, "independent_calorimetry") != Some(true))
    {
        return (
            "REJECT_CLAIM".into(),
            vec!["neutron detection is not a calibrated useful-energy measurement".into()],
        );
    }
    if claim == "motorcycle_wheel_power"
        && (string(facts, "output_basis") == Some("specific_impulse_only")
            || boolean(facts, "independent_thrust_test") != Some(true))
    {
        return (
            "REJECT_CLAIM".into(),
            vec!["specific impulse alone does not establish road-relevant thrust-to-power or wheel force".into()],
        );
    }
    if boolean(facts, "boundary_declared") != Some(true) {
        return (
            "INDETERMINATE".into(),
            vec!["system boundary is not established".into()],
        );
    }
    if boolean(facts, "comparison_boundaries_match") == Some(false) {
        return (
            "BLOCK".into(),
            vec!["candidate and baseline use different energy boundaries".into()],
        );
    }
    if claim == "well_to_wheel_efficiency"
        && boolean(facts, "upstream_stages_included") == Some(false)
    {
        return (
            "BLOCK".into(),
            vec!["upstream energy supply stages are omitted from a well-to-wheel claim".into()],
        );
    }
    if claim == "plasma_assisted_net_gain"
        && (boolean(facts, "all_auxiliary_inputs_included") != Some(true)
            || boolean(facts, "auxiliary_inputs_complete") != Some(true))
    {
        return (
            "BLOCK".into(),
            vec!["plasma subsystem parasitic energy is not fully accounted for".into()],
        );
    }
    if boolean(facts, "unknown_input_imputed_as_zero") == Some(true)
        || boolean(facts, "input_inventory_complete") != Some(true)
    {
        return (
            "INCOMPLETE_LEDGER".into(),
            vec!["unknown or incomplete input energy cannot be silently treated as zero".into()],
        );
    }
    if boolean(facts, "auxiliary_inputs_complete") != Some(true) {
        return (
            "BLOCK".into(),
            vec!["required auxiliary energy inputs are incomplete".into()],
        );
    }
    if claim == "powertrain_comparison"
        && boolean(facts, "comparison_boundaries_match") != Some(true)
    {
        return (
            "INDETERMINATE".into(),
            vec!["matching comparison-boundary evidence is not established".into()],
        );
    }
    if string(facts, "boundary_id").is_none_or(str::is_empty)
        || string(facts, "boundary_definition").is_none_or(str::is_empty)
    {
        return (
            "INDETERMINATE".into(),
            vec!["boundary identity and definition are missing".into()],
        );
    }
    if boolean(facts, "energy_metered_at_boundary") != Some(true) {
        return (
            "INDETERMINATE".into(),
            vec!["energy measurement at the declared boundary is not established".into()],
        );
    }
    if boolean(facts, "complete_energy_balance") == Some(true) {
        if facts.get("input_streams").and_then(Value::as_array).is_none_or(Vec::is_empty)
            || facts.get("output_streams").and_then(Value::as_array).is_none_or(Vec::is_empty)
        {
            return (
                "INCOMPLETE_LEDGER".into(),
                vec!["a complete balance requires non-empty input and output stream inventories".into()],
            );
        }
        let Some((output_lower, input_upper)) = balance_bounds(facts) else {
            return (
                "INCOMPLETE_LEDGER".into(),
                vec!["complete energy balance is asserted without input/output stream measurements".into()],
            );
        };
        if output_lower > input_upper {
            return (
                "BLOCK".into(),
                vec!["output lower bound exceeds input upper bound for a declared complete balance".into()],
            );
        }
        return (
            "ACCEPT_LEDGER_ONLY".into(),
            vec!["synthetic balance closes within conservative declared uncertainty bounds".into()],
        );
    }
    if claim == "well_to_wheel_efficiency"
        && boolean(facts, "upstream_stages_included") == Some(true)
    {
        return (
            "ACCEPT_LEDGER_ONLY".into(),
            vec!["synthetic ledger includes the declared well-to-wheel boundary".into()],
        );
    }
    (
        "INDETERMINATE".into(),
        vec!["available facts do not establish a complete declared energy-accounting contract".into()],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../docs/research/data/two_wheeler_powertrain_energy_ledger_v0_1.json");

    #[test]
    fn frozen_energy_ledger_cases_match() {
        let manifest: Manifest = serde_json::from_str(FIXTURE).expect("fixture parses");
        validate_manifest(&manifest).expect("manifest is structurally valid");
        for case in &manifest.cases {
            let (computed, reasons) = evaluate_case(&case.facts);
            assert_eq!(
                computed, case.expected_decision,
                "{}: {}",
                case.scenario_id, case.title
            );
            assert!(!reasons.is_empty(), "{} has no rationale", case.scenario_id);
        }
    }

    #[test]
    fn energy_stream_totals_are_recomputed() {
        let facts = serde_json::json!({
            "input_streams":[{"id":"source","energy_j":100.0,"uncertainty_j":1.0}],
            "output_streams":[{"id":"work","energy_j":40.0,"uncertainty_j":2.0}]
        });
        let (input, output) = energy_totals(&facts);
        assert_eq!(input, Some(100.0));
        assert_eq!(output, Some(40.0));
    }

    #[test]
    fn unknown_inputs_are_not_zero() {
        let facts = serde_json::json!({
            "boundary_declared":true,
            "input_inventory_complete":false,
            "unknown_input_imputed_as_zero":true,
            "auxiliary_inputs_complete":true,
            "energy_metered_at_boundary":true,
            "claim_kind":"well_to_wheel_efficiency"
        });
        let (decision, _) = evaluate_case(&facts);
        assert_eq!(decision, "INCOMPLETE_LEDGER");
    }

    #[test]
    fn differing_comparison_boundaries_are_blocked() {
        let facts = serde_json::json!({
            "boundary_declared":true,
            "comparison_boundaries_match":false,
            "input_inventory_complete":true,
            "auxiliary_inputs_complete":true,
            "energy_metered_at_boundary":true,
            "claim_kind":"powertrain_comparison"
        });
        let (decision, _) = evaluate_case(&facts);
        assert_eq!(decision, "BLOCK");
    }

    #[test]
    fn neutron_counts_cannot_establish_net_electric_output() {
        let facts = serde_json::json!({
            "boundary_declared":true,
            "input_inventory_complete":false,
            "auxiliary_inputs_complete":false,
            "energy_metered_at_boundary":false,
            "claim_kind":"facility_net_electric",
            "output_basis":"neutron_count_only",
            "independent_calorimetry":false
        });
        let (decision, _) = evaluate_case(&facts);
        assert_eq!(decision, "REJECT_CLAIM");
    }

    #[test]
    fn specific_impulse_does_not_prove_motorcycle_wheel_power() {
        let facts = serde_json::json!({
            "boundary_declared":true,
            "input_inventory_complete":true,
            "auxiliary_inputs_complete":true,
            "energy_metered_at_boundary":true,
            "claim_kind":"motorcycle_wheel_power",
            "output_basis":"specific_impulse_only",
            "independent_thrust_test":false
        });
        let (decision, _) = evaluate_case(&facts);
        assert_eq!(decision, "REJECT_CLAIM");
    }

    #[test]
    fn complete_balance_must_close_with_uncertainty() {
        let facts = serde_json::json!({
            "boundary_declared":true,
            "input_inventory_complete":true,
            "auxiliary_inputs_complete":true,
            "energy_metered_at_boundary":true,
            "complete_energy_balance":true,
            "input_streams":[{"id":"input","energy_j":100.0,"uncertainty_j":1.0}],
            "output_streams":[{"id":"output","energy_j":150.0,"uncertainty_j":1.0}],
            "claim_kind":"conversion_efficiency"
        });
        let (decision, _) = evaluate_case(&facts);
        assert_eq!(decision, "BLOCK");
    }

    #[test]
    fn plasma_ignition_parasitics_are_non_optional() {
        let facts = serde_json::json!({
            "boundary_declared":true,
            "input_inventory_complete":true,
            "auxiliary_inputs_complete":false,
            "all_auxiliary_inputs_included":false,
            "energy_metered_at_boundary":true,
            "claim_kind":"plasma_assisted_net_gain"
        });
        let (decision, _) = evaluate_case(&facts);
        assert_eq!(decision, "BLOCK");
    }

    #[test]
    fn missing_boundary_definition_cannot_be_passed() {
        let facts = serde_json::json!({
            "boundary_declared":true,
            "input_inventory_complete":true,
            "auxiliary_inputs_complete":true,
            "energy_metered_at_boundary":true,
            "complete_energy_balance":false,
            "claim_kind":"conversion_efficiency"
        });
        let (decision, _) = evaluate_case(&facts);
        assert_eq!(decision, "INDETERMINATE");
    }

    #[test]
    fn empty_stream_inventory_cannot_support_complete_balance() {
        let facts = serde_json::json!({
            "boundary_declared":true,
            "boundary_id":"SYNTHETIC-BOUNDARY",
            "boundary_definition":"Synthetic, declared boundary",
            "input_inventory_complete":true,
            "auxiliary_inputs_complete":true,
            "energy_metered_at_boundary":true,
            "complete_energy_balance":true,
            "input_streams":[],
            "output_streams":[],
            "claim_kind":"conversion_efficiency"
        });
        let (decision, _) = evaluate_case(&facts);
        assert_eq!(decision, "INCOMPLETE_LEDGER");
    }
}
