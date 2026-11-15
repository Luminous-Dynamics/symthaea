use chrono::{DateTime, Utc};
use jsonschema::draft202012;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;

const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateBundle {
    schema_version: String,
    corpus_id: String,
    candidate_id: String,
    candidate_version: String,
    predictions: Vec<CandidatePrediction>,
    limitations: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidatePrediction {
    case_id: String,
    decision_time_utc: String,
    disposition: String,
    cause: String,
    supporting_evidence_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CaseResult {
    case_id: String,
    scenario_class: String,
    expected_disposition: String,
    predicted_disposition: String,
    disposition_correct: bool,
    expected_primary_cause: String,
    predicted_cause: String,
    attribution_made: bool,
    attribution_supported: bool,
    cause_output_correct: bool,
    evidence_references_valid: bool,
    evidence_kinds: Vec<String>,
    unsupported_attribution: bool,
    false_biofouling_attribution: bool,
}

#[derive(Debug, Serialize)]
struct EvaluationReport {
    evaluator: &'static str,
    evaluator_version: &'static str,
    evaluation_contract: &'static str,
    claim_stage: &'static str,
    corpus_id: String,
    candidate_id: String,
    candidate_version: String,
    candidate_outcome: &'static str,
    case_count: usize,
    matched_prediction_count: usize,
    correct_dispositions: usize,
    disposition_accuracy: Option<f64>,
    abstained_count: usize,
    non_abstained_count: usize,
    coverage: Option<f64>,
    selective_accuracy: Option<f64>,
    predicted_attributions: usize,
    supported_correct_attributions: usize,
    attribution_precision: Option<f64>,
    unsupported_attributions: usize,
    false_biofouling_attributions: usize,
    confusion_matrix: BTreeMap<String, BTreeMap<String, usize>>,
    case_results: Vec<CaseResult>,
    diagnostics: Vec<String>,
    claim_ceiling: &'static str,
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid JSON in {path}: {error}"))
}

fn parse_time(value: Option<&str>) -> Option<DateTime<Utc>> {
    value
        .and_then(|text| DateTime::parse_from_rfc3339(text).ok())
        .map(|date| date.with_timezone(&Utc))
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn is_abstention(disposition: &str) -> bool {
    matches!(disposition, "INDETERMINATE" | "OUT_OF_DOMAIN")
}

fn required_evidence_kinds(cause: &str) -> &'static [&'static str] {
    match cause {
        "biofouling" => &["hull_inspection"],
        "weather_or_environment" => &["weather_observation"],
        "propulsor_damage_or_blockage" => &["propulsor_inspection"],
        "biofouling_cleaned" => &["cleaning_record", "hull_inspection"],
        _ => &[],
    }
}

fn evidence_kinds_satisfy(cause: &str, kinds: &BTreeSet<String>) -> bool {
    let required = required_evidence_kinds(cause);
    !required.is_empty() && required.iter().all(|kind| kinds.contains(*kind))
}

fn attribution_is_supported(
    predicted_cause: &str,
    truth_cause: &str,
    truth_supports_attribution: bool,
    evidence_sufficient: bool,
) -> bool {
    predicted_cause != "none"
        && predicted_cause == truth_cause
        && truth_supports_attribution
        && evidence_sufficient
}

fn is_false_biofouling_attribution(
    predicted_cause: &str,
    attribution_supported: bool,
) -> bool {
    predicted_cause == "biofouling" && !attribution_supported
}

fn validate_candidate_schema(schema: &Value, candidate: &Value) -> Result<(), String> {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        return Err("candidate schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if !draft202012::meta::is_valid(schema) {
        return Err("candidate schema does not validate against the Draft 2020-12 meta-schema".to_owned());
    }
    let validator = draft202012::options()
        .should_validate_formats(true)
        .build(schema)
        .map_err(|error| format!("cannot compile candidate schema: {error}"))?;
    let errors = validator
        .iter_errors(candidate)
        .map(|error| format!("{}: {error}", error.instance_path()))
        .collect::<Vec<_>>();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("candidate output violates its schema: {}", errors.join("; ")))
    }
}

fn evaluate_case(
    case: &Value,
    prediction: &CandidatePrediction,
    diagnostics: &mut Vec<String>,
) -> Result<CaseResult, String> {
    let case_id = case
        .get("case_id")
        .and_then(Value::as_str)
        .ok_or_else(|| "corpus case has no case_id".to_owned())?;
    let scenario_class = case
        .get("scenario_class")
        .and_then(Value::as_str)
        .unwrap_or("unspecified")
        .to_owned();
    let points = case
        .get("decision_points")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{case_id}: decision_points must be an array"))?;
    let final_point = points
        .last()
        .ok_or_else(|| format!("{case_id}: at least one decision point is required"))?;
    let expected_time = final_point
        .get("decision_time_utc")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{case_id}: terminal decision time is missing"))?;
    let expected_time = parse_time(Some(expected_time))
        .ok_or_else(|| format!("{case_id}: terminal decision time is invalid"))?;
    let predicted_time = parse_time(Some(&prediction.decision_time_utc))
        .ok_or_else(|| format!("{case_id}: candidate decision time is invalid"))?;
    if expected_time != predicted_time {
        return Err(format!(
            "{case_id}: candidate decision time does not match the terminal benchmark decision"
        ));
    }

    let truth = case
        .get("challenge_truth")
        .ok_or_else(|| format!("{case_id}: evaluator-only challenge truth is missing"))?;
    let expected_disposition = truth
        .get("expected_disposition")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{case_id}: expected disposition is missing"))?
        .to_owned();
    let truth_cause = truth
        .get("primary_cause")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{case_id}: evaluator-only primary cause is missing"))?
        .to_owned();
    let truth_supports = truth
        .get("attribution_supported")
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{case_id}: attribution_supported must be boolean"))?;

    let visible_evidence = final_point
        .pointer("/feature_snapshot/evidence")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut evidence_index: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for item in &visible_evidence {
        if let Some(id) = item.get("evidence_id").and_then(Value::as_str) {
            evidence_index.entry(id.to_owned()).or_default().push(item);
        }
    }

    let mut evidence_refs_valid = true;
    let mut evidence_kinds = BTreeSet::new();
    let decision_time = parse_time(Some(&prediction.decision_time_utc))
        .ok_or_else(|| format!("{case_id}: candidate decision time is invalid"))?;
    for evidence_id in &prediction.supporting_evidence_ids {
        let Some(matches) = evidence_index.get(evidence_id) else {
            evidence_refs_valid = false;
            diagnostics.push(format!("{case_id}: evidence reference {evidence_id} is not visible at the decision point"));
            continue;
        };
        if matches.len() != 1 {
            evidence_refs_valid = false;
            diagnostics.push(format!("{case_id}: evidence reference {evidence_id} is ambiguous"));
            continue;
        }
        let item = matches[0];
        let status = item.get("status").and_then(Value::as_str).unwrap_or("unknown");
        let available = parse_time(item.get("available_at_utc").and_then(Value::as_str));
        if status != "confirmed" || !matches!(available.as_ref(), Some(time) if time <= &decision_time) {
            evidence_refs_valid = false;
            diagnostics.push(format!("{case_id}: evidence reference {evidence_id} is unconfirmed, invalidly timed, or unavailable at decision time"));
            continue;
        }
        if let Some(kind) = item.get("kind").and_then(Value::as_str) {
            evidence_kinds.insert(kind.to_owned());
        }
    }

    if prediction.cause == "none" && !prediction.supporting_evidence_ids.is_empty() {
        return Err(format!("{case_id}: cause=none must not carry causal evidence references"));
    }

    let evidence_sufficient =
        evidence_refs_valid && evidence_kinds_satisfy(&prediction.cause, &evidence_kinds);
    let attribution_made = prediction.cause != "none";
    let attribution_supported = attribution_is_supported(
        &prediction.cause,
        &truth_cause,
        truth_supports,
        evidence_sufficient,
    );
    let unsupported_attribution = attribution_made && !attribution_supported;
    let false_biofouling_attribution =
        is_false_biofouling_attribution(&prediction.cause, attribution_supported);
    let cause_output_correct = if attribution_made {
        attribution_supported
    } else {
        !truth_supports
    };
    let disposition_correct = prediction.disposition == expected_disposition;

    Ok(CaseResult {
        case_id: case_id.to_owned(),
        scenario_class,
        expected_disposition,
        predicted_disposition: prediction.disposition.clone(),
        disposition_correct,
        expected_primary_cause: truth_cause,
        predicted_cause: prediction.cause.clone(),
        attribution_made,
        attribution_supported,
        cause_output_correct,
        evidence_references_valid: evidence_refs_valid,
        evidence_kinds: evidence_kinds.into_iter().collect(),
        unsupported_attribution,
        false_biofouling_attribution,
    })
}

fn evaluate(corpus: &Value, bundle: &CandidateBundle) -> Result<EvaluationReport, String> {
    if bundle.schema_version != "0.1.0" {
        return Err("unsupported candidate output schema_version".to_owned());
    }
    if bundle.candidate_id.trim().is_empty() || bundle.candidate_version.trim().is_empty() {
        return Err("candidate_id and candidate_version must be non-empty".to_owned());
    }
    if bundle.limitations.is_empty() {
        return Err("candidate limitations must be explicitly declared".to_owned());
    }
    let corpus_id = corpus
        .get("corpus_id")
        .and_then(Value::as_str)
        .ok_or_else(|| "corpus_id is missing from the benchmark corpus".to_owned())?;
    if bundle.corpus_id != corpus_id {
        return Err("candidate corpus_id does not match the benchmark corpus".to_owned());
    }
    if corpus.get("schema_version").and_then(Value::as_str) != Some("0.1.0") {
        return Err("unsupported benchmark corpus version".to_owned());
    }
    let cases = corpus
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| "benchmark cases must be an array".to_owned())?;

    let mut predictions: BTreeMap<String, &CandidatePrediction> = BTreeMap::new();
    for prediction in &bundle.predictions {
        if predictions
            .insert(prediction.case_id.clone(), prediction)
            .is_some()
        {
            return Err(format!("duplicate candidate prediction for {}", prediction.case_id));
        }
    }

    let corpus_ids = cases
        .iter()
        .filter_map(|case| case.get("case_id").and_then(Value::as_str).map(str::to_owned))
        .collect::<BTreeSet<_>>();
    if corpus_ids.len() != cases.len() {
        return Err("benchmark corpus contains duplicate or missing case identities".to_owned());
    }
    let candidate_ids = predictions.keys().cloned().collect::<BTreeSet<_>>();
    if corpus_ids != candidate_ids {
        return Err("candidate predictions must cover every corpus case exactly once and contain no extra cases".to_owned());
    }

    let mut diagnostics = Vec::new();
    let mut case_results = Vec::with_capacity(cases.len());
    for case in cases {
        let case_id = case
            .get("case_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "benchmark case_id is missing".to_owned())?;
        let prediction = predictions
            .get(case_id)
            .ok_or_else(|| format!("candidate prediction missing for {case_id}"))?;
        case_results.push(evaluate_case(case, prediction, &mut diagnostics)?);
    }

    let case_count = case_results.len();
    let correct_dispositions = case_results
        .iter()
        .filter(|result| result.disposition_correct)
        .count();
    let abstained_count = case_results
        .iter()
        .filter(|result| is_abstention(&result.predicted_disposition))
        .count();
    let non_abstained_count = case_count.saturating_sub(abstained_count);
    let correct_non_abstained = case_results
        .iter()
        .filter(|result| {
            !is_abstention(&result.predicted_disposition) && result.disposition_correct
        })
        .count();
    let predicted_attributions = case_results
        .iter()
        .filter(|result| result.attribution_made)
        .count();
    let supported_correct_attributions = case_results
        .iter()
        .filter(|result| result.attribution_supported)
        .count();
    let unsupported_attributions = case_results
        .iter()
        .filter(|result| result.unsupported_attribution)
        .count();
    let false_biofouling_attributions = case_results
        .iter()
        .filter(|result| result.false_biofouling_attribution)
        .count();

    let mut confusion_matrix: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for result in &case_results {
        *confusion_matrix
            .entry(result.expected_disposition.clone())
            .or_default()
            .entry(result.predicted_disposition.clone())
            .or_default() += 1;
    }

    let all_dispositions_correct = correct_dispositions == case_count;
    let all_cause_outputs_correct = case_results.iter().all(|result| result.cause_output_correct);
    let candidate_outcome = if unsupported_attributions > 0 {
        "REJECTED"
    } else if all_dispositions_correct && all_cause_outputs_correct {
        "ACCEPTED_SYNTHETIC_SMOKE"
    } else {
        "NOT_QUALIFIED"
    };

    Ok(EvaluationReport {
        evaluator: "marine-performance-drift-evaluator",
        evaluator_version: "0.1.0",
        evaluation_contract: "VALID",
        claim_stage: "synthetic_contract_smoke_only",
        corpus_id: corpus_id.to_owned(),
        candidate_id: bundle.candidate_id.clone(),
        candidate_version: bundle.candidate_version.clone(),
        candidate_outcome,
        case_count,
        matched_prediction_count: predictions.len(),
        correct_dispositions,
        disposition_accuracy: ratio(correct_dispositions, case_count),
        abstained_count,
        non_abstained_count,
        coverage: ratio(non_abstained_count, case_count),
        selective_accuracy: ratio(correct_non_abstained, non_abstained_count),
        predicted_attributions,
        supported_correct_attributions,
        attribution_precision: ratio(supported_correct_attributions, predicted_attributions),
        unsupported_attributions,
        false_biofouling_attributions,
        confusion_matrix,
        case_results,
        diagnostics,
        claim_ceiling: "This report tests candidate-output correspondence, evidence-reference semantics, selective disposition scoring and cause-attribution safeguards against a synthetic challenge corpus only. It does not establish real-vessel accuracy, hydrodynamic validity, fuel savings, cleaning efficacy, safety, class approval or operational permission.",
    })
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!(
            "usage: marine-performance-drift-evaluator <corpus.json> <candidate-schema.json> <candidate-run.json>"
        );
        std::process::exit(64);
    }

    let corpus = match read_json(&args[0]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(64);
        }
    };
    let schema = match read_json(&args[1]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(64);
        }
    };
    let candidate_value = match read_json(&args[2]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(64);
        }
    };

    if let Err(error) = validate_candidate_schema(&schema, &candidate_value) {
        eprintln!("{error}");
        std::process::exit(65);
    }
    let bundle: CandidateBundle = match serde_json::from_value(candidate_value) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid candidate run: {error}");
            std::process::exit(65);
        }
    };

    match evaluate(&corpus, &bundle) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).unwrap_or_else(|error| {
                    format!("{{\"evaluation_contract\":\"INVALID\",\"error\":\"{error}\"}}")
                })
            );
        }
        Err(error) => {
            eprintln!("candidate evaluation rejected: {error}");
            std::process::exit(65);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biofouling_requires_hull_inspection_not_power_residual() {
        let calibration_only = BTreeSet::from(["calibration_record".to_owned()]);
        let weather_only = BTreeSet::from(["weather_observation".to_owned()]);
        let hull_inspection = BTreeSet::from(["hull_inspection".to_owned()]);
        assert!(!evidence_kinds_satisfy("biofouling", &calibration_only));
        assert!(!evidence_kinds_satisfy("biofouling", &weather_only));
        assert!(evidence_kinds_satisfy("biofouling", &hull_inspection));
    }

    #[test]
    fn cleaned_hull_requires_both_cleaning_and_inspection_evidence() {
        let cleaning_only = BTreeSet::from(["cleaning_record".to_owned()]);
        let both = BTreeSet::from([
            "cleaning_record".to_owned(),
            "hull_inspection".to_owned(),
        ]);
        assert!(!evidence_kinds_satisfy("biofouling_cleaned", &cleaning_only));
        assert!(evidence_kinds_satisfy("biofouling_cleaned", &both));
    }

    #[test]
    fn environmental_evidence_cannot_support_biofouling() {
        let kinds = BTreeSet::from(["weather_observation".to_owned()]);
        assert!(!evidence_kinds_satisfy("biofouling", &kinds));
    }

    #[test]
    fn false_biofouling_attribution_is_a_hard_negative() {
        assert!(is_false_biofouling_attribution("biofouling", false));
        assert!(!is_false_biofouling_attribution("weather_or_environment", false));
        assert!(!is_false_biofouling_attribution("biofouling", true));
    }

    #[test]
    fn attribution_requires_truth_support_and_cause_specific_evidence() {
        assert!(!attribution_is_supported(
            "biofouling",
            "biofouling",
            false,
            true
        ));
        assert!(!attribution_is_supported(
            "biofouling",
            "weather_or_environment",
            true,
            true
        ));
        assert!(!attribution_is_supported(
            "biofouling",
            "biofouling",
            true,
            false
        ));
        assert!(attribution_is_supported(
            "biofouling",
            "biofouling",
            true,
            true
        ));
    }

    #[test]
    fn indeterminate_and_out_of_domain_are_abstentions() {
        assert!(is_abstention("INDETERMINATE"));
        assert!(is_abstention("OUT_OF_DOMAIN"));
        assert!(!is_abstention("DRIFT_DETECTED"));
        assert!(!is_abstention("NO_DRIFT_DETECTED"));
    }

    #[test]
    fn ratios_are_undefined_when_the_denominator_is_zero() {
        assert_eq!(ratio(0, 0), None);
        assert_eq!(ratio(3, 4), Some(0.75));
    }
}
