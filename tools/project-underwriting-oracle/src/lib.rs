//! Deterministic semantic gates for the public synthetic project-underwriting corpus.
//!
//! This evaluator qualifies input readiness and risk-calculation readiness only.
//! It does not issue investment recommendations or establish real-project bankability.

use std::collections::HashSet;

use serde_json::{Value, json};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disposition {
    pub case_id: String,
    pub input_disposition: String,
    pub risk_disposition: String,
    pub decision_authority: String,
    pub blockers: Vec<String>,
    pub warnings: Vec<String>,
}

impl Disposition {
    pub fn to_value(&self) -> Value {
        json!({
            "case_id": self.case_id,
            "input_disposition": self.input_disposition,
            "risk_disposition": self.risk_disposition,
            "decision_authority": self.decision_authority,
            "blockers": self.blockers,
            "warnings": self.warnings,
        })
    }
}

fn string<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn array<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn timestamp_le(left: &str, right: &str) -> bool {
    // The v1 corpus canonicalizes timestamps to UTC RFC 3339 with a trailing Z.
    // Lexical ordering is therefore chronological ordering for this frozen profile.
    left.ends_with('Z') && right.ends_with('Z') && left <= right
}

fn covers_decision_time(evidence: &Value, decision_time: &str) -> bool {
    let valid_time = match evidence.get("valid_time") {
        Some(value) => value,
        None => return false,
    };
    match (
        string(valid_time, "from"),
        string(valid_time, "to"),
    ) {
        (Some(from), Some(to)) => {
            timestamp_le(from, decision_time) && timestamp_le(decision_time, to)
        }
        _ => false,
    }
}

fn find_evidence<'a>(case: &'a Value, evidence_id: &str) -> Option<&'a Value> {
    array(case, "evidence")
        .iter()
        .find(|evidence| string(evidence, "evidence_id") == Some(evidence_id))
}

fn evidence_usable(
    evidence: &Value,
    claim_id: &str,
    decision_time: &str,
    permitted_classes: &[&str],
    expected_subject: &str,
    expected_boundary: &str,
    expected_issuer_role: &str,
    require_valid_at_decision: bool,
) -> bool {
    let evidence_class = string(evidence, "evidence_class").unwrap_or_default();
    if !permitted_classes.contains(&evidence_class) {
        return false;
    }
    if string(evidence, "claim_id") != Some(claim_id)
        || string(evidence, "record_state") != Some("current")
        || string(evidence, "subject_id") != Some(expected_subject)
        || string(evidence, "boundary_id") != Some(expected_boundary)
        || string(evidence, "issuer_role") != Some(expected_issuer_role)
    {
        return false;
    }
    if string(evidence, "issuer_id").is_none_or(str::is_empty)
        || string(evidence, "boundary_id").is_none_or(str::is_empty)
    {
        return false;
    }
    let known_at = match string(evidence, "known_at") {
        Some(value) => value,
        None => return false,
    };
    if !timestamp_le(known_at, decision_time) {
        return false;
    }
    if evidence_class == "issuer_signed"
        && string(evidence, "authenticity_status") != Some("verified")
    {
        return false;
    }
    !require_valid_at_decision || covers_decision_time(evidence, decision_time)
}

fn claim_has_evidence(
    case: &Value,
    claim_id: &str,
    decision_time: &str,
    permitted_classes: &[&str],
    expected_subject: &str,
    expected_issuer_role: &str,
    require_valid_at_decision: bool,
) -> bool {
    let claim = array(case, "claims")
        .iter()
        .find(|claim| string(claim, "claim_id") == Some(claim_id));
    let Some(claim) = claim else {
        return false;
    };
    if string(claim, "assessment") != Some("supported") {
        return false;
    }
    array(claim, "evidence_refs").iter().any(|reference| {
        let Some(evidence_id) = reference.as_str() else {
            return false;
        };
        find_evidence(case, evidence_id).is_some_and(|evidence| {
            evidence_usable(
                evidence,
                claim_id,
                decision_time,
                permitted_classes,
                expected_subject,
                expected_boundary,
                expected_issuer_role,
                require_valid_at_decision,
            )
        })
    })
}

fn has_material_source_conflict(case: &Value, decision_time: &str) -> bool {
    let critical_claims: Vec<&str> = array(case, "claims")
        .iter()
        .filter(|claim| {
            matches!(
                string(claim, "materiality"),
                Some("critical" | "high")
            )
        })
        .filter_map(|claim| string(claim, "claim_id"))
        .collect();

    array(case, "evidence").iter().any(|evidence| {
        let known_at = string(evidence, "known_at").unwrap_or_default();
        string(evidence, "evidence_class") == Some("contradictory_document")
            && string(evidence, "record_state") == Some("current")
            && critical_claims.contains(&string(evidence, "claim_id").unwrap_or_default())
            && timestamp_le(known_at, decision_time)
    })
}

fn evidence_references_resolve(case: &Value) -> bool {
    let evidence = array(case, "evidence");
    let evidence_ids: HashSet<&str> = evidence
        .iter()
        .filter_map(|item| string(item, "evidence_id"))
        .collect();
    if evidence_ids.len() != evidence.len() {
        return false;
    }

    let claim_ids: HashSet<&str> = array(case, "claims")
        .iter()
        .filter_map(|item| string(item, "claim_id"))
        .collect();
    if array(case, "claims").len() != claim_ids.len()
        || evidence.iter().any(|item| {
            !claim_ids.contains(string(item, "claim_id").unwrap_or_default())
        })
    {
        return false;
    }

    let mut references = Vec::new();
    for claim in array(case, "claims") {
        extend_references(&mut references, claim, "evidence_refs");
    }
    if let Some(grid) = case.get("grid_connection") {
        extend_references(&mut references, grid, "evidence_refs");
    }
    if let Some(model) = case.get("physical_model") {
        extend_references(&mut references, model, "calibration_evidence_refs");
        for measurement in array(model, "measurements") {
            if let Some(reference) = string(measurement, "evidence_ref") {
                references.push(reference);
            }
        }
    }
    if let Some(finance) = case.get("financial_case") {
        extend_references(&mut references, finance, "fx_evidence_refs");
        for flow in array(finance, "cash_flows") {
            extend_references(&mut references, flow, "evidence_refs");
        }
    }
    for scenario in array(case, "scenarios") {
        extend_references(&mut references, scenario, "evidence_refs");
    }
    references.into_iter().all(|reference| evidence_ids.contains(reference))
}

fn extend_references<'a>(target: &mut Vec<&'a str>, value: &'a Value, key: &str) {
    target.extend(array(value, key).iter().filter_map(Value::as_str));
}

fn model_domain_qualified(case: &Value, decision_time: &str) -> bool {
    let project = match case.get("project") {
        Some(value) => value,
        None => return false,
    };
    let model = match case.get("physical_model") {
        Some(value) => value,
        None => return false,
    };
    let scope = match model.get("qualification_scope") {
        Some(value) => value,
        None => return false,
    };
    let capacity = string(project, "capacity_mw")
        .and_then(|value| value.parse::<f64>().ok());
    let maximum = string(scope, "max_capacity_mw")
        .and_then(|value| value.parse::<f64>().ok());
    let (Some(capacity), Some(maximum)) = (capacity, maximum) else {
        return false;
    };
    let effective_from = string(scope, "effective_from").unwrap_or_default();
    let effective_until = string(scope, "effective_until").unwrap_or_default();

    string(model, "qualification") == Some("qualified")
        && capacity.is_finite()
        && maximum.is_finite()
        && capacity > 0.0
        && capacity <= maximum
        && timestamp_le(effective_from, decision_time)
        && timestamp_le(decision_time, effective_until)
}

fn model_has_independent_calibration(case: &Value, decision_time: &str) -> bool {
    let model = match case.get("physical_model") {
        Some(value) => value,
        None => return false,
    };
    let site_id = case
        .get("site")
        .and_then(|site| string(site, "site_id"))
        .unwrap_or_default();

    array(model, "calibration_evidence_refs").iter().any(|reference| {
        let Some(evidence_id) = reference.as_str() else {
            return false;
        };
        find_evidence(case, evidence_id).is_some_and(|evidence| {
            evidence_usable(
                evidence,
                "yield_baseline",
                decision_time,
                &["independently_reconciled_measurement"],
                site_id,
                "plant-metering-boundary-v1",
                "independent_verifier",
                false,
            )
        })
    })
}

fn valid_cash_flow_schedule(case: &Value) -> bool {
    let finance = match case.get("financial_case") {
        Some(value) => value,
        None => return false,
    };
    let currency = string(finance, "cash_flow_currency").unwrap_or_default();
    if currency.is_empty() || string(finance, "day_count") != Some("actual_365_fixed") {
        return false;
    }

    let flows = array(finance, "cash_flows");
    if flows.is_empty() {
        return false;
    }
    let mut previous_date: Option<&str> = None;
    for flow in flows {
        let Some(date) = string(flow, "date") else {
            return false;
        };
        if !date.ends_with('Z') || previous_date.is_some_and(|previous| previous > date) {
            return false;
        }
        if string(flow, "currency") != Some(currency) {
            return false;
        }
        let amount = string(flow, "amount_decimal")
            .and_then(|value| value.parse::<f64>().ok());
        if amount.is_none_or(|value| !value.is_finite()) {
            return false;
        }
        if array(flow, "evidence_refs").is_empty() {
            return false;
        }
        previous_date = Some(date);
    }
    true
}

fn has_fx_evidence(case: &Value, decision_time: &str) -> bool {
    let finance = match case.get("financial_case") {
        Some(value) => value,
        None => return false,
    };
    if string(finance, "fx_status") != Some("evidenced")
        || string(finance, "fx_rate")
            .and_then(|value| value.parse::<f64>().ok())
            .is_none_or(|rate| !rate.is_finite() || rate <= 0.0)
    {
        return false;
    }
    let fx_refs = array(finance, "fx_evidence_refs");
    fx_refs.iter().any(|reference| {
        let Some(evidence_id) = reference.as_str() else {
            return false;
        };
        find_evidence(case, evidence_id).is_some_and(|evidence| {
            evidence_usable(
                evidence,
                "fx_basis",
                decision_time,
                &["issuer_signed"],
                case.get("site")
                    .and_then(|site| string(site, "site_id"))
                    .unwrap_or_default(),
                "fx-observation-boundary-v1",
                "central_bank",
                true,
            )
        })
    })
}

pub fn evaluate_case(case: &Value) -> Disposition {
    let case_id = string(case, "case_id").unwrap_or("missing-case-id").to_owned();
    let decision_time = string(case, "decision_time").unwrap_or_default();
    let site_id = case
        .get("site")
        .and_then(|site| string(site, "site_id"))
        .unwrap_or_default();
    let mut blockers = Vec::new();
    let mut warnings = Vec::new();

    if has_material_source_conflict(case, decision_time) {
        blockers.push("MATERIAL_SOURCE_CONFLICT".to_owned());
        warnings.push(
            "Conflicting current evidence remains visible; do not majority-vote source count."
                .to_owned(),
        );
    }
    if !evidence_references_resolve(case) {
        blockers.push("EVIDENCE_REFERENCE_INTEGRITY_FAILURE".to_owned());
    }

    let grid = case.get("grid_connection");
    let grid_subject = grid
        .and_then(|value| string(value, "point_of_connection_id"))
        .unwrap_or_default();
    let grid_status = grid
        .and_then(|value| string(value, "status"))
        .unwrap_or_default();
    let requested = grid
        .and_then(|value| string(value, "requested_capacity_mw"))
        .and_then(|value| value.parse::<f64>().ok());
    let allocated = grid
        .and_then(|value| string(value, "allocated_capacity_mw"))
        .and_then(|value| value.parse::<f64>().ok());
    let grid_boundary = format!("grid-poc-{grid_subject}");
    let grid_evidence = claim_has_evidence(
        case,
        "grid_right",
        decision_time,
        &["issuer_signed"],
        grid_subject,
        &grid_boundary,
        "system_operator",
        true,
    );
    if !matches!(grid_status, "connection_agreement_signed" | "energized")
        || !grid_evidence
        || requested.zip(allocated).is_none_or(|(request, allocation)| {
            !request.is_finite()
                || !allocation.is_finite()
                || request <= 0.0
                || allocation < request
        })
    {
        blockers.push("GRID_EVIDENCE_BELOW_REQUIRED_CLASS".to_owned());
    }

    if !model_domain_qualified(case, decision_time)
        || !claim_has_evidence(
            case,
            "physical_performance",
            decision_time,
            &["validated_model_output"],
            site_id,
            "plant-ac-boundary-v1",
            "model_registry",
            false,
        )
        || !model_has_independent_calibration(case, decision_time)
    {
        blockers.push("PHYSICAL_MODEL_OUTSIDE_QUALIFIED_DOMAIN".to_owned());
    }

    if !claim_has_evidence(
        case,
        "yield_baseline",
        decision_time,
        &["independently_reconciled_measurement"],
        site_id,
        "plant-metering-boundary-v1",
        "independent_verifier",
        false,
    ) {
        blockers.push("YIELD_BASELINE_NOT_RECONCILED".to_owned());
    }

    if !claim_has_evidence(
        case,
        "revenue_tariff",
        decision_time,
        &["issuer_signed"],
        site_id,
        "project-offtake-boundary-v1",
        "offtaker",
        true,
    ) {
        blockers.push("NO_DECISION_TIME_VALID_TARIFF_EVIDENCE".to_owned());
        let future_known_tariff = array(case, "evidence").iter().any(|evidence| {
            string(evidence, "claim_id") == Some("revenue_tariff")
                && string(evidence, "known_at")
                    .is_some_and(|known_at| known_at.ends_with('Z') && decision_time < known_at)
        });
        if future_known_tariff {
            warnings.push(
                "Evidence with known_at after decision_time must be excluded even when valid_time overlaps."
                    .to_owned(),
            );
        }
    }

    if !claim_has_evidence(
        case,
        "capex_basis",
        decision_time,
        &["issuer_signed"],
        site_id,
        "project-epc-cost-boundary-v1",
        "epc_contractor",
        true,
    ) {
        blockers.push("CAPEX_BASIS_NOT_QUALIFIED".to_owned());
    }

    if !valid_cash_flow_schedule(case) {
        blockers.push("CASH_FLOW_SCHEDULE_INVALID".to_owned());
    }

    let finance = case.get("financial_case");
    let reporting_currency = finance
        .and_then(|value| string(value, "reporting_currency"))
        .unwrap_or_default();
    let cash_flow_currency = finance
        .and_then(|value| string(value, "cash_flow_currency"))
        .unwrap_or_default();
    if reporting_currency != cash_flow_currency && !has_fx_evidence(case, decision_time) {
        blockers.push("REPORTING_CURRENCY_CONVERSION_MISSING".to_owned());
    }

    let dependence = array(case, "scenarios")
        .first()
        .and_then(|scenario| string(scenario, "dependence"))
        .unwrap_or("unknown");
    let risk_disposition = match dependence {
        "joint_modelled" | "common_factor_modelled" => {
            "scenario_model_qualified_for_synthetic_use"
        }
        "independence_assumed" => {
            warnings.push(
                "Independence is an explicit assumption; joint tail metrics are not robust to dependence misspecification."
                    .to_owned(),
            );
            "dependence_assumption_only"
        }
        _ => {
            warnings.push(
                "Do not report joint tail-risk probabilities as calibrated when dependence is unknown."
                    .to_owned(),
            );
            "indeterminate"
        }
    };

    blockers.sort();
    blockers.dedup();
    warnings.sort();
    warnings.dedup();

    Disposition {
        case_id,
        input_disposition: if blockers.is_empty() {
            "eligible_for_synthetic_valuation".to_owned()
        } else {
            "blocked".to_owned()
        },
        risk_disposition: if blockers.is_empty() {
            risk_disposition.to_owned()
        } else {
            "not_evaluated".to_owned()
        },
        decision_authority: "none".to_owned(),
        blockers,
        warnings,
    }
}

pub fn evaluate_corpus(corpus: &Value) -> Vec<Disposition> {
    array(corpus, "cases").iter().map(evaluate_case).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn public_corpus() -> Value {
        serde_json::from_str(include_str!(
            "../../../docs/research/project-underwriting/v1/public-cases.json"
        ))
        .expect("public corpus must be valid JSON")
    }

    fn reference() -> Value {
        serde_json::from_str(include_str!(
            "../../../docs/research/project-underwriting/v1/reference-expected.json"
        ))
        .expect("reference outcomes must be valid JSON")
    }

    #[test]
    fn public_cases_match_transparent_reference_dispositions() {
        let results = evaluate_corpus(&public_corpus());
        let expected = reference()["expected"].as_array().unwrap();
        assert_eq!(results.len(), expected.len());
        for result in results {
            let expected_case = expected
                .iter()
                .find(|item| item["case_id"] == result.case_id)
                .expect("every case must have one expected disposition");
            assert_eq!(result.to_value()["input_disposition"], expected_case["input_disposition"]);
            assert_eq!(result.to_value()["risk_disposition"], expected_case["risk_disposition"]);
            assert_eq!(result.to_value()["decision_authority"], expected_case["decision_authority"]);
            assert_eq!(result.to_value()["blockers"], expected_case["blockers"]);
            assert_eq!(result.to_value()["warnings"], expected_case["warnings"]);
        }
    }

    #[test]
    fn future_known_tariff_is_excluded_from_decision_frontier() {
        let mut corpus = public_corpus();
        let case = corpus["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|item| item["case_id"] == "solar-storage-base-001")
            .unwrap();
        let tariff = case["evidence"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|item| item["evidence_id"] == "ev_tariff")
            .unwrap();
        tariff["known_at"] = json!("2026-08-02T00:00:00Z");
        let result = evaluate_case(case);
        assert!(result.blockers.iter().any(|value| value == "NO_DECISION_TIME_VALID_TARIFF_EVIDENCE"));
    }

    #[test]
    fn site_or_point_of_connection_substitution_is_rejected() {
        let mut corpus = public_corpus();
        let case = corpus["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|item| item["case_id"] == "solar-storage-base-001")
            .unwrap();
        let grid = case["evidence"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|item| item["evidence_id"] == "ev_grid_agreement")
            .unwrap();
        grid["subject_id"] = json!("different-point-of-connection");
        let result = evaluate_case(case);
        assert!(result.blockers.iter().any(|value| value == "GRID_EVIDENCE_BELOW_REQUIRED_CLASS"));
    }

    #[test]
    fn unknown_dependence_does_not_become_a_joint_tail_probability() {
        let corpus = public_corpus();
        let case = corpus["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["case_id"] == "joint-dependence-unknown-006")
            .unwrap();
        let result = evaluate_case(case);
        assert_eq!(result.input_disposition, "eligible_for_synthetic_valuation");
        assert_eq!(result.risk_disposition, "indeterminate");
    }

    #[test]
    fn an_out_of_domain_model_is_not_qualified_by_its_label() {
        let corpus = public_corpus();
        let case = corpus["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["case_id"] == "model-domain-exceeded-003")
            .unwrap();
        let result = evaluate_case(case);
        assert!(result.blockers.iter().any(|value| value == "PHYSICAL_MODEL_OUTSIDE_QUALIFIED_DOMAIN"));
    }

    #[test]
    fn dangling_evidence_reference_fails_closed() {
        let mut corpus = public_corpus();
        let case = corpus["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|item| item["case_id"] == "solar-storage-base-001")
            .unwrap();
        case["evidence"]
            .as_array_mut()
            .unwrap()
            .retain(|item| item["evidence_id"] != "ev_grid_agreement");
        let result = evaluate_case(case);
        assert!(result.blockers.iter().any(|value| value == "EVIDENCE_REFERENCE_INTEGRITY_FAILURE"));
    }

    #[test]
    fn evidence_with_a_disputed_record_state_cannot_satisfy_a_gate() {
        let mut corpus = public_corpus();
        let case = corpus["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|item| item["case_id"] == "solar-storage-base-001")
            .unwrap();
        let grid = case["evidence"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|item| item["evidence_id"] == "ev_grid_agreement")
            .unwrap();
        grid["record_state"] = json!("disputed");
        let result = evaluate_case(case);
        assert!(result.blockers.iter().any(|value| value == "GRID_EVIDENCE_BELOW_REQUIRED_CLASS"));
    }
}
