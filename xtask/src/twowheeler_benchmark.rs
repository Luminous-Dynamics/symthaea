//! Evidence-bounded verifier for the synthetic two-wheeler A0 fixture.
//!
//! This evaluates declared decision rules; it is not a bicycle physics solver,
//! standards-compliance engine, physical qualification tool, or factory optimizer.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const ALLOWED_DECISIONS: &[&str] = &[
    "ACCEPT_SYNTHETIC_CANDIDATE_ONLY",
    "BLOCK",
    "INDETERMINATE",
    "INCOMPLETE_ECONOMICS",
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

/// Read the frozen fixture, recompute each expected decision from facts, and
/// optionally persist a deterministic JSON report. A successful exit means
/// only that the fixture's expected decisions agree with these software rules.
pub fn run(input: &Path, output: Option<&Path>) -> Result<()> {
    let raw = fs::read(input)
        .with_context(|| format!("reading benchmark manifest {}", input.display()))?;
    let manifest_sha256 = format!("{:x}", Sha256::digest(&raw));
    let manifest: Manifest = serde_json::from_slice(&raw)
        .with_context(|| format!("parsing benchmark manifest {}", input.display()))?;
    validate_manifest(&manifest)?;

    let mut cases = Vec::with_capacity(manifest.cases.len());
    for case in &manifest.cases {
        let (computed_decision, computed_reasons) =
            evaluate_case(&case.facts, &manifest.evidence_class);
        let decision_matches = computed_decision == case.expected_decision;
        cases.push(CaseResult {
            scenario_id: case.scenario_id.clone(),
            class: case.class.clone(),
            title: case.title.clone(),
            expected_decision: case.expected_decision.clone(),
            computed_decision,
            decision_matches,
            computed_reasons,
            expected_reasons: case.expected_reasons.clone(),
            must_not_claim: case.must_not_claim.clone(),
        });
    }

    let matched_cases = cases.iter().filter(|case| case.decision_matches).count();
    let mismatched_cases = cases.len() - matched_cases;
    let report = VerificationReport {
        report_version: "0.1.0".to_string(),
        manifest_id: manifest.manifest_id,
        manifest_sha256,
        evidence_class: manifest.evidence_class,
        case_count: cases.len(),
        matched_cases,
        mismatched_cases,
        fixture_conformance_pass: mismatched_cases == 0,
        claim_ceiling: "Synthetic decision-contract conformance only; no physical, safety, economic, roadworthiness, or regulatory claim.".to_string(),
        cases,
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
            .with_context(|| format!("writing verification report {}", output.display()))?;
    }
    println!("{encoded}");

    if mismatched_cases > 0 {
        bail!(
            "two-wheeler fixture decision mismatch: {mismatched_cases} of {} cases differ",
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
        bail!("A0 contract fixture must remain Synthetic evidence");
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
        if case.expected_reasons.is_empty() {
            bail!("case {} has no expected rationale", case.scenario_id);
        }
        if case.must_not_claim.is_empty() {
            bail!("case {} has no claim ceiling", case.scenario_id);
        }
        if !case.facts.is_object() {
            bail!("case {} facts must be a JSON object", case.scenario_id);
        }
    }
    if positive_controls == 0 || negative_controls == 0 {
        bail!("manifest needs at least one positive and one negative control");
    }
    Ok(())
}

fn string<'a>(facts: &'a Value, key: &str) -> Option<&'a str> {
    facts.get(key).and_then(Value::as_str)
}

fn boolean(facts: &Value, key: &str) -> Option<bool> {
    facts.get(key).and_then(Value::as_bool)
}

fn number(facts: &Value, key: &str) -> Option<f64> {
    facts.get(key).and_then(Value::as_f64)
}

fn interval(facts: &Value, key: &str) -> Option<(f64, f64)> {
    let values = facts.get(key)?.as_array()?;
    if values.len() != 2 {
        return None;
    }
    Some((values[0].as_f64()?, values[1].as_f64()?))
}

fn evaluate_case(facts: &Value, evidence_class: &str) -> (String, Vec<String>) {
    let claim = string(facts, "requested_claim").unwrap_or("");

    if string(facts, "hard_safety_gate") == Some("failed") {
        return (
            "BLOCK".to_string(),
            vec!["hard safety constraints cannot be compensated by an aggregate score".into()],
        );
    }

    if boolean(facts, "required_input_fields_present") == Some(false) {
        return (
            "BLOCK".to_string(),
            vec!["required input fields are missing".into()],
        );
    }
    if boolean(facts, "units_consistent") == Some(false) {
        return (
            "BLOCK".to_string(),
            vec!["declared physical units are inconsistent".into()],
        );
    }

    for (left, right, reason) in [
        (
            "configuration_id",
            "evidence_configuration_id",
            "configuration identity differs from the evidence subject",
        ),
        (
            "battery_configuration_id",
            "electrical_evidence_configuration_id",
            "battery/electrical evidence is bound to another configuration",
        ),
    ] {
        if let (Some(a), Some(b)) = (string(facts, left), string(facts, right)) {
            if a != b {
                return ("BLOCK".to_string(), vec![reason.to_string()]);
            }
        }
    }
    if boolean(facts, "evidence_configuration_matches") == Some(false) {
        return (
            "BLOCK".to_string(),
            vec!["evidence configuration does not match candidate configuration".into()],
        );
    }

    if let Some(required) = facts.get("required_evidence").and_then(Value::as_array) {
        let present: BTreeSet<&str> = facts
            .get("present_evidence")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let missing: Vec<&str> = required
            .iter()
            .filter_map(Value::as_str)
            .filter(|item| !present.contains(item))
            .collect();
        if !missing.is_empty() {
            return (
                "BLOCK".to_string(),
                vec![format!("required evidence is missing: {}", missing.join(", "))],
            );
        }
    }

    if let (Some(vehicle_profile), Some(evidence_profile)) = (
        string(facts, "vehicle_profile"),
        string(facts, "evidence_profile"),
    ) {
        if vehicle_profile != evidence_profile
            && string(facts, "applicability_review") != Some("complete")
        {
            return (
                "BLOCK".to_string(),
                vec!["vehicle-class applicability has not been established".into()],
            );
        }
    }

    if let (Some(requirement), Some(observed)) = (
        string(facts, "requirement_quantity"),
        string(facts, "observed_quantity"),
    ) {
        if requirement != observed {
            return (
                "BLOCK".to_string(),
                vec!["requirement and observation describe different physical quantities".into()],
            );
        }
    }
    if let (Some(required_unit), Some(observed_unit)) = (
        string(facts, "requirement_unit"),
        string(facts, "observed_unit"),
    ) {
        if required_unit != observed_unit {
            // A boolean/string flag is not a validated conversion. Until a
            // typed conversion model and its provenance are bound into the
            // fixture contract, mismatched units must be rejected.
            return (
                "BLOCK".to_string(),
                vec!["unit mismatch has no validated typed conversion evidence".into()],
            );
        }
    }

    if let (Some(model_scope), Some(requested_environment)) = (
        string(facts, "model_scope"),
        string(facts, "requested_environment"),
    ) {
        if model_scope != requested_environment
            && string(facts, "out_of_domain_policy") == Some("unspecified")
        {
            return (
                "INDETERMINATE".to_string(),
                vec!["requested environment exceeds declared model scope".into()],
            );
        }
    }

    if claim == "braking_qualification"
        && boolean(facts, "required_test_matrix_complete") == Some(false)
    {
        return (
            "REJECT_CLAIM".to_string(),
            vec!["one observation does not satisfy the declared braking test matrix".into()],
        );
    }

    if facts.get("price").is_some_and(Value::is_null)
        && number(facts, "imputed_value") == Some(0.0)
    {
        return (
            "INCOMPLETE_ECONOMICS".to_string(),
            vec!["unknown cost input was silently imputed as zero".into()],
        );
    }

    if let (Some(a), Some(b), Some(capacity)) = (
        interval(facts, "operation_a_interval"),
        interval(facts, "operation_b_interval"),
        facts.get("resource_capacity").and_then(Value::as_u64),
    ) {
        let overlaps = a.0 < b.1 && b.0 < a.1;
        if overlaps && capacity <= 1 {
            return (
                "BLOCK".to_string(),
                vec!["exclusive resource has overlapping operation assignments".into()],
            );
        }
    }

    if boolean(facts, "inspection_required") == Some(true)
        && boolean(facts, "inspection_operation_present") == Some(false)
    {
        return (
            "BLOCK".to_string(),
            vec!["mandatory inspection operation is missing".into()],
        );
    }

    for key in ["candidate_cost_includes_scrap", "candidate_cost_includes_rework"] {
        if boolean(facts, key) == Some(false) {
            return (
                "INCOMPLETE_ECONOMICS".to_string(),
                vec!["candidate cost omits scrap or rework".into()],
            );
        }
    }
    for key in ["candidate_cost_includes_scrap", "candidate_cost_includes_rework"] {
        if boolean(facts, key).is_some()
            && boolean(facts, key) != boolean(
                facts,
                key.replace("candidate_", "baseline_").as_str(),
            )
        {
            return (
                "INCOMPLETE_ECONOMICS".to_string(),
                vec!["candidate and baseline cost boundaries differ".into()],
            );
        }
    }

    if string(facts, "repair_task_status") == Some("performed")
        && boolean(facts, "required_post_repair_verification") == Some(true)
        && boolean(facts, "post_repair_evidence_present") == Some(false)
    {
        return (
            "BLOCK".to_string(),
            vec!["repair has no required post-repair verification evidence".into()],
        );
    }

    if string(facts, "source_evidence_class") == Some("Synthetic")
        && string(facts, "requested_output_evidence_class") == Some("RegulatoryReceipt")
        && boolean(facts, "external_authority_receipt_present") == Some(false)
    {
        return (
            "REJECT_CLAIM".to_string(),
            vec!["synthetic evidence cannot create an external regulatory receipt".into()],
        );
    }

    if boolean(facts, "used_for_model_fitting") == Some(true)
        && boolean(facts, "reported_as_held_out") == Some(true)
    {
        return (
            "REJECT_CLAIM".to_string(),
            vec!["model-fitting data cannot also establish independent holdout validation".into()],
        );
    }

    if number(facts, "prototype_count") == Some(1.0) && claim == "fleet_reliability" {
        return (
            "REJECT_CLAIM".to_string(),
            vec!["one prototype does not establish fleet reliability".into()],
        );
    }

    if claim == "candidate_screening_only"
        && evidence_class == "Synthetic"
        && boolean(facts, "required_input_fields_present") == Some(true)
        && boolean(facts, "units_consistent") == Some(true)
        && boolean(facts, "evidence_configuration_matches") == Some(true)
    {
        return (
            "ACCEPT_SYNTHETIC_CANDIDATE_ONLY".to_string(),
            vec!["internally consistent synthetic case; no physical claim admitted".into()],
        );
    }

    (
        "INDETERMINATE".to_string(),
        vec!["no complete decision rule establishes the requested claim".into()],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../docs/research/data/two_wheeler_a0_benchmark_v0_1.json");

    #[test]
    fn frozen_cases_match_independently_computed_decisions() {
        let manifest: Manifest = serde_json::from_str(FIXTURE).expect("fixture parses");
        validate_manifest(&manifest).expect("manifest contract is valid");
        for case in &manifest.cases {
            let (computed, reasons) = evaluate_case(&case.facts, &manifest.evidence_class);
            assert_eq!(
                computed, case.expected_decision,
                "{}: {}",
                case.scenario_id, case.title
            );
            assert!(
                !reasons.is_empty(),
                "{} produced no explanation",
                case.scenario_id
            );
        }
    }

    #[test]
    fn fixture_identities_are_unique_and_controls_are_present() {
        let manifest: Manifest = serde_json::from_str(FIXTURE).expect("fixture parses");
        let mut ids = BTreeSet::new();
        assert!(manifest.cases.iter().all(|case| ids.insert(&case.scenario_id)));
        assert!(manifest.cases.iter().any(|case| case.class == "positive_control"));
        assert!(manifest.cases.iter().any(|case| case.class == "negative_control"));
    }

    #[test]
    fn missing_price_is_not_zero_cost() {
        let facts = serde_json::json!({
            "cost_item": "critical_bearing_supply",
            "price": null,
            "missingness_reason": "no_verified_quote",
            "imputed_value": 0,
            "requested_claim": "complete_lifecycle_cost_comparison"
        });
        let (decision, _) = evaluate_case(&facts, "Synthetic");
        assert_eq!(decision, "INCOMPLETE_ECONOMICS");
    }

    #[test]
    fn failed_safety_gate_cannot_be_compensated_by_score() {
        let facts = serde_json::json!({
            "aggregate_score": 0.97,
            "hard_safety_gate": "failed",
            "requested_claim": "overall_design_pass"
        });
        let (decision, _) = evaluate_case(&facts, "Synthetic");
        assert_eq!(decision, "BLOCK");
    }

    #[test]
    fn incompatible_physical_quantities_are_blocked() {
        let facts = serde_json::json!({
            "requirement_quantity": "braking_force",
            "requirement_unit": "N",
            "observed_quantity": "mass",
            "observed_unit": "kg",
            "explicit_conversion_model": true,
            "requested_claim": "requirement_satisfied"
        });
        let (decision, _) = evaluate_case(&facts, "Synthetic");
        assert_eq!(decision, "BLOCK");
    }

    #[test]
    fn out_of_domain_model_use_is_indeterminate() {
        let facts = serde_json::json!({
            "model_scope": "urban_duty_cycle_only",
            "requested_environment": "rough_terrain_high_payload",
            "out_of_domain_policy": "unspecified",
            "requested_claim": "robust_design_pass"
        });
        let (decision, _) = evaluate_case(&facts, "Synthetic");
        assert_eq!(decision, "INDETERMINATE");
    }

    #[test]
    fn overlapping_exclusive_factory_resource_is_blocked() {
        let facts = serde_json::json!({
            "resource_id": "WELD_FIXTURE_SYN_1",
            "operation_a_interval": [0, 10],
            "operation_b_interval": [5, 12],
            "resource_capacity": 1,
            "requested_claim": "valid_schedule"
        });
        let (decision, _) = evaluate_case(&facts, "Synthetic");
        assert_eq!(decision, "BLOCK");
    }
}
