use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const CORPUS_VERSION: &str = "ev-early-window-outcomes/0.1";
const EPS: f64 = 1e-9;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Corpus {
    pub corpus_version: String,
    pub cases: Vec<Case>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub case_id: String,
    pub expected_disposition: String,
    pub asset_ref: String,
    pub configuration_sha256: String,
    pub prediction_event_ref: String,
    pub modality: String,
    pub prediction_cutoff_seconds: f64,
    pub features: Vec<FeatureReceipt>,
    pub targets: Vec<TargetReceipt>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureReceipt {
    pub feature_id: String,
    pub feature_sha256: String,
    pub source_class: String,
    pub source_window_start_seconds: f64,
    pub source_window_end_seconds: f64,
    pub available_at_seconds: f64,
    pub uses_target_or_future_data: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetReceipt {
    pub target_id: String,
    pub asset_ref: String,
    pub configuration_sha256: String,
    pub kind: String,
    pub unit: String,
    pub target_value: Option<f64>,
    pub label_source: String,
    pub evidence_ref: String,
    pub evidence_sha256: String,
    pub reference_protocol_ref: String,
    pub reference_protocol_sha256: String,
    pub reference_condition_ref: String,
    pub reference_condition_sha256: String,
    pub reference_time_seconds: f64,
    pub target_window_start_seconds: f64,
    pub target_window_end_seconds: f64,
    pub horizon_seconds: f64,
    pub uncertainty: Option<f64>,
    pub coverage_fraction: f64,
    pub target_resolution: String,
    pub censoring_state: String,
    pub observed_duration_seconds: Option<f64>,
    pub competing_event_code: Option<String>,
    pub load_profile_ref: Option<String>,
    pub load_profile_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TargetAudit {
    pub target_id: String,
    pub kind: String,
    pub disposition: String,
    pub evidence_grade: String,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CaseAudit {
    pub case_id: String,
    pub accepted: bool,
    pub errors: Vec<String>,
    pub targets: Vec<TargetAudit>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AuditReport {
    pub corpus_version: String,
    pub input_sha256: String,
    pub canonical_corpus_sha256: String,
    pub report_sha256: String,
    pub case_count: usize,
    pub accepted_count: usize,
    pub rejected_count: usize,
    pub right_censored_rdt_count: usize,
    pub competing_event_rdt_count: usize,
    pub not_computable_target_count: usize,
    pub cases: Vec<CaseAudit>,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn schema_validator(bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("schema JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("schema compilation: {error}"))
}

fn validate_schema(validator: &Validator, value: &Value) -> Result<(), String> {
    let errors: Vec<String> = validator
        .iter_errors(value)
        .map(|error| error.to_string())
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("schema errors: {}", errors.join("; ")))
    }
}

fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn validate_feature(case: &Case, feature: &FeatureReceipt) -> Result<(), String> {
    if !nonempty(&feature.feature_id) || !is_sha256(&feature.feature_sha256) {
        return Err(format!(
            "FEATURE_ID_OR_DIGEST_INVALID:{}",
            feature.feature_id
        ));
    }
    if !finite_nonnegative(feature.source_window_start_seconds)
        || !finite_nonnegative(feature.source_window_end_seconds)
        || !finite_nonnegative(feature.available_at_seconds)
        || feature.source_window_start_seconds > feature.source_window_end_seconds
    {
        return Err(format!("FEATURE_TIME_INVALID:{}", feature.feature_id));
    }
    if feature.uses_target_or_future_data {
        return Err(format!(
            "FEATURE_TARGET_OR_FUTURE_LEAKAGE:{}",
            feature.feature_id
        ));
    }
    if !matches!(
        feature.source_class.as_str(),
        "OBSERVED_PREFIX" | "DERIVED_PREFIX"
    ) {
        return Err(format!("FEATURE_SOURCE_NOT_CAUSAL:{}", feature.feature_id));
    }
    if feature.source_window_end_seconds > case.prediction_cutoff_seconds + EPS
        || feature.available_at_seconds > case.prediction_cutoff_seconds + EPS
    {
        return Err(format!(
            "FEATURE_UNAVAILABLE_AT_CUTOFF:{}",
            feature.feature_id
        ));
    }
    Ok(())
}

fn target_audit(target: &TargetReceipt, case: &Case) -> Result<TargetAudit, String> {
    let cutoff = case.prediction_cutoff_seconds;
    if !nonempty(&target.target_id)
        || target.asset_ref != case.asset_ref
        || target.configuration_sha256 != case.configuration_sha256
        || !is_sha256(&target.configuration_sha256)
        || !nonempty(&target.evidence_ref)
        || !is_sha256(&target.evidence_sha256)
        || !nonempty(&target.reference_protocol_ref)
        || !is_sha256(&target.reference_protocol_sha256)
        || !nonempty(&target.reference_condition_ref)
        || !is_sha256(&target.reference_condition_sha256)
        || !nonempty(&target.unit)
    {
        return Err(format!(
            "TARGET_EVIDENCE_BINDING_INVALID:{}",
            target.target_id
        ));
    }
    for value in [
        target.reference_time_seconds,
        target.target_window_start_seconds,
        target.target_window_end_seconds,
        target.horizon_seconds,
        target.coverage_fraction,
    ] {
        if !value.is_finite() {
            return Err(format!(
                "TARGET_TIME_OR_COVERAGE_NONFINITE:{}",
                target.target_id
            ));
        }
    }
    if target.reference_time_seconds <= cutoff + EPS
        || target.target_window_start_seconds < cutoff - EPS
        || target.target_window_end_seconds <= target.target_window_start_seconds
        || target.horizon_seconds <= 0.0
        || target.coverage_fraction < 0.0
        || target.coverage_fraction > 1.0
    {
        return Err(format!(
            "TARGET_WINDOW_OR_REFERENCE_INVALID:{}",
            target.target_id
        ));
    }
    if target
        .uncertainty
        .is_some_and(|value| !finite_nonnegative(value))
    {
        return Err(format!("TARGET_UNCERTAINTY_INVALID:{}", target.target_id));
    }
    if target.target_value.is_some_and(|value| !value.is_finite()) {
        return Err(format!("TARGET_VALUE_NONFINITE:{}", target.target_id));
    }

    match target.kind.as_str() {
        "SOH" => {
            if !matches!(target.unit.as_str(), "fraction" | "percent")
                || target.censoring_state != "NOT_APPLICABLE"
                || target.target_value.is_none()
                || target.uncertainty.is_none()
                || !matches!(
                    target.label_source.as_str(),
                    "INDEPENDENT_REFERENCE_TEST" | "OPERATIONAL_DERIVED"
                )
            {
                return Err(format!("SOH_TARGET_CONTRACT_INVALID:{}", target.target_id));
            }
            let value = target.target_value.expect("checked");
            let upper = if target.unit == "fraction" {
                1.0
            } else {
                100.0
            };
            if !(0.0..=upper).contains(&value) {
                return Err(format!("SOH_TARGET_OUT_OF_RANGE:{}", target.target_id));
            }
            let grade = if target.label_source == "INDEPENDENT_REFERENCE_TEST" {
                "INDEPENDENT_PHYSICAL_REFERENCE"
            } else {
                "OPERATIONAL_DERIVED_REFERENCE_NOT_INDEPENDENT"
            };
            Ok(TargetAudit {
                target_id: target.target_id.clone(),
                kind: target.kind.clone(),
                disposition: "ADMISSIBLE_EXACT".to_owned(),
                evidence_grade: grade.to_owned(),
                reason: None,
            })
        }
        "REMAINING_DISCHARGE_TIME" => {
            if target.unit != "seconds"
                || target.uncertainty.is_none()
                || !nonempty(target.load_profile_ref.as_deref().unwrap_or(""))
                || !is_sha256(target.load_profile_sha256.as_deref().unwrap_or(""))
                || !finite_nonnegative(target.observed_duration_seconds.unwrap_or(f64::NAN))
            {
                return Err(format!(
                    "RDT_PROTOCOL_OR_DURATION_INVALID:{}",
                    target.target_id
                ));
            }
            let observed = target.observed_duration_seconds.expect("checked");
            if observed <= 0.0 || observed > target.horizon_seconds + EPS {
                return Err(format!(
                    "RDT_OBSERVED_DURATION_OUT_OF_RANGE:{}",
                    target.target_id
                ));
            }
            match target.censoring_state.as_str() {
                "EXACT_ENDPOINT_EVENT" => {
                    if target.label_source != "INDEPENDENT_REFERENCE_TEST"
                        || target.target_value.is_none()
                        || (target.target_value.expect("checked") - observed).abs() > EPS
                        || target.competing_event_code.is_some()
                    {
                        return Err(format!("RDT_EXACT_EVENT_INVALID:{}", target.target_id));
                    }
                    Ok(TargetAudit {
                        target_id: target.target_id.clone(),
                        kind: target.kind.clone(),
                        disposition: "ADMISSIBLE_EXACT_EVENT".to_owned(),
                        evidence_grade: "OBSERVED_ENDPOINT_UNDER_DECLARED_LOAD".to_owned(),
                        reason: None,
                    })
                }
                "RIGHT_CENSORED" => {
                    if target.target_value.is_some() || target.competing_event_code.is_some() {
                        return Err(format!(
                            "RDT_CENSORING_CONVERTED_TO_POINT_LABEL:{}",
                            target.target_id
                        ));
                    }
                    Ok(TargetAudit {
                        target_id: target.target_id.clone(),
                        kind: target.kind.clone(),
                        disposition: "ADMISSIBLE_RIGHT_CENSORED".to_owned(),
                        evidence_grade: "CENSORING_PRESERVED_FOR_SURVIVAL_METRICS".to_owned(),
                        reason: Some("Do not score as an exact remaining-time target".to_owned()),
                    })
                }
                "COMPETING_EVENT" => {
                    if target.target_value.is_some()
                        || !nonempty(target.competing_event_code.as_deref().unwrap_or(""))
                    {
                        return Err(format!("RDT_COMPETING_EVENT_INVALID:{}", target.target_id));
                    }
                    Ok(TargetAudit {
                        target_id: target.target_id.clone(),
                        kind: target.kind.clone(),
                        disposition: "ADMISSIBLE_COMPETING_EVENT".to_owned(),
                        evidence_grade: "COMPETING_RISK_PRESERVED".to_owned(),
                        reason: Some(
                            "Preserve event type; do not relabel as endpoint time".to_owned(),
                        ),
                    })
                }
                _ => Err(format!(
                    "RDT_CENSORING_STATE_UNSUPPORTED:{}",
                    target.target_id
                )),
            }
        }
        "PEAK_TEMPERATURE" => {
            if target.unit != "degC"
                || !matches!(
                    target.target_resolution.as_str(),
                    "CELL" | "MODULE" | "PACK"
                )
                || !matches!(
                    target.label_source.as_str(),
                    "INDEPENDENT_THERMAL_MEASUREMENT" | "OPERATIONAL_DERIVED"
                )
                || target.censoring_state != "NOT_APPLICABLE"
                || target.target_window_end_seconds > cutoff + target.horizon_seconds + EPS
            {
                return Err(format!(
                    "PEAK_TEMPERATURE_TARGET_CONTRACT_INVALID:{}",
                    target.target_id
                ));
            }
            if target.coverage_fraction < 1.0 - EPS {
                if target.target_value.is_some() {
                    return Err(format!(
                        "INCOMPLETE_THERMAL_COVERAGE_CLAIMED_AS_PEAK:{}",
                        target.target_id
                    ));
                }
                return Ok(TargetAudit {
                    target_id: target.target_id.clone(),
                    kind: target.kind.clone(),
                    disposition: "NOT_COMPUTABLE".to_owned(),
                    evidence_grade: "INCOMPLETE_THERMAL_OBSERVABILITY".to_owned(),
                    reason: Some(
                        "Observed population does not establish the declared maximum".to_owned(),
                    ),
                });
            }
            if target.uncertainty.is_none() {
                return Err(format!(
                    "PEAK_TEMPERATURE_UNCERTAINTY_MISSING:{}",
                    target.target_id
                ));
            }
            let value = target
                .target_value
                .ok_or_else(|| format!("PEAK_TEMPERATURE_VALUE_MISSING:{}", target.target_id))?;
            Ok(TargetAudit {
                target_id: target.target_id.clone(),
                kind: target.kind.clone(),
                disposition: "ADMISSIBLE_EXACT".to_owned(),
                evidence_grade: if target.label_source == "INDEPENDENT_THERMAL_MEASUREMENT" {
                    "INDEPENDENT_THERMAL_REFERENCE"
                } else {
                    "OPERATIONAL_THERMAL_REFERENCE_NOT_INDEPENDENT"
                }
                .to_owned(),
                reason: Some(format!(
                    "Maximum is qualified only at {} resolution",
                    target.target_resolution
                )),
            })
        }
        "POWER_ENVELOPE" => {
            if target.unit != "W"
                || target.uncertainty.is_none()
                || target.target_value.is_none()
                || target.target_value.is_some_and(|value| value < 0.0)
                || target.label_source != "QUALIFIED_PHYSICAL_MODEL"
                || target.target_resolution != "PACK"
                || target.coverage_fraction < 1.0 - EPS
                || target.censoring_state != "NOT_APPLICABLE"
                || target.target_window_end_seconds > cutoff + target.horizon_seconds + EPS
            {
                return Err(format!(
                    "POWER_ENVELOPE_TARGET_CONTRACT_INVALID:{}",
                    target.target_id
                ));
            }
            Ok(TargetAudit {
                target_id: target.target_id.clone(),
                kind: target.kind.clone(),
                disposition: "ADMISSIBLE_MODEL_OUTPUT".to_owned(),
                evidence_grade: "MODEL_DERIVED_NOT_PHYSICAL_MEASUREMENT".to_owned(),
                reason: Some(
                    "Must not be presented as an independently measured power label".to_owned(),
                ),
            })
        }
        _ => Err(format!("TARGET_KIND_UNSUPPORTED:{}", target.kind)),
    }
}

fn validate_case(case: &Case) -> Result<Vec<TargetAudit>, Vec<String>> {
    let mut errors = Vec::new();
    if !nonempty(&case.case_id)
        || !nonempty(&case.asset_ref)
        || !nonempty(&case.prediction_event_ref)
        || !is_sha256(&case.configuration_sha256)
        || !matches!(
            case.modality.as_str(),
            "PARTIAL_DISCHARGE" | "PARTIAL_CHARGE"
        )
        || !finite_nonnegative(case.prediction_cutoff_seconds)
        || case.prediction_cutoff_seconds <= 0.0
    {
        errors.push("PREDICTION_IDENTITY_OR_CUTOFF_INVALID".to_owned());
    }
    if case.features.is_empty() {
        errors.push("NO_CAUSAL_FEATURE_RECEIPTS".to_owned());
    }
    let mut feature_ids = BTreeSet::new();
    for feature in &case.features {
        if !feature_ids.insert(feature.feature_id.clone()) {
            errors.push(format!("DUPLICATE_FEATURE_ID:{}", feature.feature_id));
        }
        if let Err(error) = validate_feature(case, feature) {
            errors.push(error);
        }
    }
    if case.targets.is_empty() {
        errors.push("NO_TARGET_RECEIPTS".to_owned());
    }
    let mut target_ids = BTreeSet::new();
    let mut audits = Vec::new();
    for target in &case.targets {
        if !target_ids.insert(target.target_id.clone()) {
            errors.push(format!("DUPLICATE_TARGET_ID:{}", target.target_id));
        }
        match target_audit(target, case) {
            Ok(audit) => audits.push(audit),
            Err(error) => errors.push(error),
        }
    }
    if errors.is_empty() {
        Ok(audits)
    } else {
        Err(errors)
    }
}

fn canonicalize(corpus: &mut Corpus) {
    corpus
        .cases
        .sort_by(|left, right| left.case_id.cmp(&right.case_id));
    for case in &mut corpus.cases {
        case.features
            .sort_by(|left, right| left.feature_id.cmp(&right.feature_id));
        case.targets
            .sort_by(|left, right| left.target_id.cmp(&right.target_id));
    }
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<AuditReport, String> {
    let validator = schema_validator(schema_bytes)?;
    let value: Value =
        serde_json::from_slice(corpus_bytes).map_err(|error| format!("corpus JSON: {error}"))?;
    validate_schema(&validator, &value)?;
    let mut corpus: Corpus =
        serde_json::from_value(value).map_err(|error| format!("corpus decode: {error}"))?;
    if corpus.corpus_version != CORPUS_VERSION || corpus.cases.is_empty() {
        return Err("unsupported or empty early-window outcome corpus".to_owned());
    }
    canonicalize(&mut corpus);
    let canonical = serde_json::to_vec(&corpus)
        .map_err(|error| format!("canonical corpus serialization: {error}"))?;
    let mut seen = BTreeSet::new();
    let mut reports = Vec::new();
    let mut accepted_count = 0usize;
    let mut rejected_count = 0usize;
    let mut right_censored_rdt_count = 0usize;
    let mut competing_event_rdt_count = 0usize;
    let mut not_computable_target_count = 0usize;

    for case in &corpus.cases {
        if !seen.insert(case.case_id.clone()) {
            return Err(format!("duplicate case identity: {}", case.case_id));
        }
        if !matches!(case.expected_disposition.as_str(), "ACCEPT" | "REJECT") {
            return Err(format!("unknown expected disposition: {}", case.case_id));
        }
        let outcome = validate_case(case);
        let accepted = outcome.is_ok();
        if accepted != (case.expected_disposition == "ACCEPT") {
            let detail = outcome
                .as_ref()
                .err()
                .map(|errors| errors.join("; "))
                .unwrap_or_else(|| "unexpected acceptance".to_owned());
            return Err(format!(
                "fixture expectation mismatch for {}: accepted={}, expected={}: {}",
                case.case_id, accepted, case.expected_disposition, detail
            ));
        }
        let (targets, errors) = match outcome {
            Ok(targets) => {
                accepted_count += 1;
                (targets, Vec::new())
            }
            Err(errors) => {
                rejected_count += 1;
                (Vec::new(), errors)
            }
        };
        for target in &targets {
            if target.disposition == "ADMISSIBLE_RIGHT_CENSORED" {
                right_censored_rdt_count += 1;
            } else if target.disposition == "ADMISSIBLE_COMPETING_EVENT" {
                competing_event_rdt_count += 1;
            } else if target.disposition == "NOT_COMPUTABLE" {
                not_computable_target_count += 1;
            }
        }
        reports.push(CaseAudit {
            case_id: case.case_id.clone(),
            accepted,
            errors,
            targets,
        });
    }

    let mut report = AuditReport {
        corpus_version: corpus.corpus_version,
        input_sha256: sha256(corpus_bytes),
        canonical_corpus_sha256: sha256(&canonical),
        report_sha256: String::new(),
        case_count: reports.len(),
        accepted_count,
        rejected_count,
        right_censored_rdt_count,
        competing_event_rdt_count,
        not_computable_target_count,
        cases: reports,
    };
    let bytes =
        serde_json::to_vec(&report).map_err(|error| format!("report serialization: {error}"))?;
    report.report_sha256 = sha256(&bytes);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_early_window_outcomes_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/early_window_outcomes_v0_1.json");

    fn report() -> AuditReport {
        audit_corpus(SCHEMA, CORPUS).expect("frozen early-window outcome corpus")
    }

    #[test]
    fn frozen_corpus_preserves_exact_censored_and_uncomputable_target_states() {
        let report = report();
        assert_eq!(report.case_count, 14);
        assert_eq!(report.accepted_count, 5);
        assert_eq!(report.rejected_count, 9);
        assert_eq!(report.right_censored_rdt_count, 1);
        assert_eq!(report.competing_event_rdt_count, 1);
        assert_eq!(report.not_computable_target_count, 1);
    }

    #[test]
    fn right_censored_rdt_is_not_scored_as_an_exact_remaining_time() {
        let report = report();
        let case = report
            .cases
            .iter()
            .find(|case| case.case_id == "rdt-right-censored-preserved")
            .expect("censored case");
        assert_eq!(case.targets[0].disposition, "ADMISSIBLE_RIGHT_CENSORED");
    }

    #[test]
    fn competing_event_is_preserved_as_a_distinct_event_type() {
        let report = report();
        let case = report
            .cases
            .iter()
            .find(|case| case.case_id == "rdt-competing-event-preserved")
            .expect("competing event case");
        assert_eq!(case.targets[0].disposition, "ADMISSIBLE_COMPETING_EVENT");
    }

    #[test]
    fn incomplete_thermal_coverage_is_not_promoted_to_a_pack_peak() {
        let report = report();
        let case = report
            .cases
            .iter()
            .find(|case| case.case_id == "thermal-coverage-not-computable")
            .expect("thermal coverage case");
        assert_eq!(case.targets[0].disposition, "NOT_COMPUTABLE");
    }

    #[test]
    fn operational_soh_label_is_not_misrepresented_as_independent_ground_truth() {
        let report = report();
        let case = report
            .cases
            .iter()
            .find(|case| case.case_id == "operational-soh-reference-labelled")
            .expect("operational label case");
        assert_eq!(
            case.targets[0].evidence_grade,
            "OPERATIONAL_DERIVED_REFERENCE_NOT_INDEPENDENT"
        );
    }

    #[test]
    fn model_derived_power_envelope_is_not_labelled_as_physical_measurement() {
        let report = report();
        let case = report
            .cases
            .iter()
            .find(|case| case.case_id == "multioutput-30s-accepted")
            .expect("multioutput case");
        assert_eq!(
            case.targets[0].evidence_grade,
            "MODEL_DERIVED_NOT_PHYSICAL_MEASUREMENT"
        );
    }

    #[test]
    fn feature_and_target_array_order_does_not_change_canonical_report() {
        let original = report();
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).expect("corpus");
        corpus.cases.reverse();
        for case in &mut corpus.cases {
            case.features.reverse();
            case.targets.reverse();
        }
        let bytes = serde_json::to_vec(&corpus).expect("permuted corpus");
        let replay = audit_corpus(SCHEMA, &bytes).expect("permuted audit");
        assert_eq!(
            original.canonical_corpus_sha256,
            replay.canonical_corpus_sha256
        );
        assert_eq!(
            serde_json::to_value(&original.cases).unwrap(),
            serde_json::to_value(&replay.cases).unwrap()
        );
        assert_eq!(original.report_sha256, replay.report_sha256);
    }

    #[test]
    fn feature_after_cutoff_is_rejected() {
        let mut corpus: Value = serde_json::from_slice(CORPUS).expect("corpus");
        let case = corpus["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|case| case["case_id"] == "multioutput-30s-accepted")
            .expect("multioutput case");
        case["features"][0]["available_at_seconds"] = Value::from(31.0);
        case["expected_disposition"] = Value::String("REJECT".into());
        let bytes = serde_json::to_vec(&corpus).expect("mutated corpus");
        let report = audit_corpus(SCHEMA, &bytes).expect("expected reject is recorded");
        assert_eq!(report.rejected_count, 10);
    }

    #[test]
    fn censored_rdt_cannot_be_relabelled_with_a_point_estimate() {
        let mut corpus: Value = serde_json::from_slice(CORPUS).expect("corpus");
        let case = corpus["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|case| case["case_id"] == "rdt-right-censored-preserved")
            .expect("censored case");
        case["targets"][0]["target_value"] = Value::from(900.0);
        case["expected_disposition"] = Value::String("REJECT".into());
        let bytes = serde_json::to_vec(&corpus).expect("mutated corpus");
        let report = audit_corpus(SCHEMA, &bytes).expect("expected reject is recorded");
        assert_eq!(report.rejected_count, 10);
    }

    #[test]
    fn partial_thermal_coverage_cannot_claim_a_numeric_global_maximum() {
        let mut corpus: Value = serde_json::from_slice(CORPUS).expect("corpus");
        let case = corpus["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|case| case["case_id"] == "thermal-coverage-not-computable")
            .expect("thermal case");
        case["targets"][0]["target_value"] = Value::from(61.0);
        case["expected_disposition"] = Value::String("REJECT".into());
        let bytes = serde_json::to_vec(&corpus).expect("mutated corpus");
        assert!(audit_corpus(SCHEMA, &bytes).is_ok());
    }

    #[test]
    fn unknown_target_fields_are_rejected() {
        let mut corpus: Value = serde_json::from_slice(CORPUS).expect("corpus");
        corpus["cases"][0]["targets"][0]["future_soh_hint"] = Value::Bool(true);
        assert!(serde_json::from_value::<Corpus>(corpus).is_err());
    }

    #[test]
    fn multioutput_case_contains_independent_evidence_grades_per_target() {
        let report = report();
        let case = report
            .cases
            .iter()
            .find(|case| case.case_id == "multioutput-30s-accepted")
            .expect("multioutput case");
        assert_eq!(case.targets.len(), 4);
        assert!(case.targets.iter().any(|target| target.kind == "SOH"
            && target.evidence_grade == "INDEPENDENT_PHYSICAL_REFERENCE"));
        assert!(
            case.targets
                .iter()
                .any(|target| target.kind == "POWER_ENVELOPE"
                    && target.evidence_grade == "MODEL_DERIVED_NOT_PHYSICAL_MEASUREMENT")
        );
    }
    #[test]
    fn every_output_label_must_bind_to_the_prediction_asset_and_configuration() {
        let report = report();
        let case = report
            .cases
            .iter()
            .find(|case| case.case_id == "target-asset-configuration-mismatch-rejected")
            .expect("target identity negative control");
        assert!(!case.accepted);
        assert!(
            case.errors
                .iter()
                .any(|error| error.contains("TARGET_EVIDENCE_BINDING_INVALID"))
        );
    }

    #[test]
    fn exact_rdt_requires_measurement_uncertainty() {
        let mut corpus: Value = serde_json::from_slice(CORPUS).expect("corpus");
        let case = corpus["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|case| case["case_id"] == "multioutput-30s-accepted")
            .expect("multioutput case");
        let rdt = case["targets"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|target| target["kind"] == "REMAINING_DISCHARGE_TIME")
            .expect("RDT target");
        rdt["uncertainty"] = Value::Null;
        case["expected_disposition"] = Value::String("REJECT".into());
        let bytes = serde_json::to_vec(&corpus).expect("mutated corpus");
        let report = audit_corpus(SCHEMA, &bytes).expect("expected reject is audited");
        assert_eq!(report.rejected_count, 10);
    }
}
