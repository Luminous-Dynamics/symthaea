use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CORPUS_VERSION: &str = "ev-mission-outcome-audit/0.1";
const CONFIDENCE_95: f64 = 0.95;
const EPS: f64 = 1e-9;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Corpus {
    pub corpus_version: String,
    pub cases: Vec<ValidationCase>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AssetIdentity {
    pub vehicle_ref: String,
    pub pack_ref: String,
    pub configuration_sha256: String,
    pub topology_sha256: String,
}

impl PartialEq for AssetIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.vehicle_ref == other.vehicle_ref
            && self.pack_ref == other.pack_ref
            && self.configuration_sha256 == other.configuration_sha256
            && self.topology_sha256 == other.topology_sha256
    }
}

impl Eq for AssetIdentity {}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredMeasurement {
    pub constraint: String,
    pub scope_ref: String,
    pub unit: String,
    pub limit_value: f64,
    pub relation: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedMeasurement {
    pub constraint: String,
    pub scope_ref: String,
    pub unit: String,
    pub measured_value: f64,
    pub sensor_ref: String,
    pub calibration_ref: String,
    pub calibration_sha256: String,
    pub measured_at_ms: u64,
    pub measurement_ref: String,
    pub measurement_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub asset: AssetIdentity,
    pub mission_profile_ref: String,
    pub mission_profile_sha256: String,
    pub mission_report_ref: String,
    pub mission_report_sha256: String,
    pub evaluator_version: String,
    pub model_artifact_ref: String,
    pub model_artifact_sha256: String,
    pub split_manifest_ref: String,
    pub split_manifest_sha256: String,
    pub evaluation_partition: String,
    pub split_frozen_at_ms: u64,
    pub prediction_frozen_at_ms: u64,
    pub disposition: String,
    pub required_measurements: Vec<RequiredMeasurement>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalReference {
    pub asset: AssetIdentity,
    pub mission_profile_ref: String,
    pub mission_profile_sha256: String,
    pub evidence_class: String,
    pub protocol_ref: String,
    pub protocol_sha256: String,
    pub evidence_ref: String,
    pub evidence_sha256: String,
    pub outcome: String,
    pub outcome_reason: Option<String>,
    pub observed_at_ms: u64,
    pub measurements: Vec<ObservedMeasurement>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationCase {
    pub case_id: String,
    pub prediction: Prediction,
    pub reference: PhysicalReference,
}

#[derive(Clone, Debug, Serialize)]
pub struct CaseAudit {
    pub case_id: String,
    pub pack_ref: String,
    pub evidence_class: String,
    pub predicted_disposition: String,
    pub reference_outcome: String,
    pub classification: String,
    pub scoring_eligible: bool,
    pub measurement_count: usize,
    pub covered_measurement_count: usize,
    pub violated_constraints: Vec<String>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AuditReport {
    pub corpus_version: String,
    pub canonical_corpus_sha256: String,
    pub report_sha256: String,
    pub case_count: usize,
    pub true_feasible_count: usize,
    pub false_feasible_count: usize,
    pub true_infeasible_count: usize,
    pub false_infeasible_count: usize,
    pub abstained_count: usize,
    pub not_scorable_count: usize,
    pub conflicting_reference_count: usize,
    pub rejected_reference_count: usize,
    pub synthetic_scorable_count: usize,
    pub physical_reference_case_count: usize,
    pub physical_scorable_count: usize,
    pub physical_predicted_feasible_count: usize,
    pub physical_false_feasible_count: usize,
    pub physical_predicted_infeasible_count: usize,
    pub physical_false_infeasible_count: usize,
    pub physical_false_feasible_upper_95_one_sided: Option<f64>,
    pub physical_false_infeasible_upper_95_one_sided: Option<f64>,
    pub source_grade: String,
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

fn valid_relation(constraint: &str, relation: &str) -> bool {
    match constraint {
        "DISCHARGE_POWER"
        | "CHARGE_ACCEPTANCE"
        | "ENERGY_RESERVE"
        | "SOC_MINIMUM"
        | "VOLTAGE_MINIMUM" => relation == "MINIMUM_REQUIRED",
        "SOC_MAXIMUM" | "VOLTAGE_MAXIMUM" | "TEMPERATURE_MAXIMUM" => {
            relation == "MAXIMUM_ALLOWED"
        }
        _ => false,
    }
}

fn compile_schema(bytes: &[u8]) -> Result<Validator, String> {
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
        Err(format!(
            "mission outcome schema errors: {}",
            errors.join("; ")
        ))
    }
}

fn identity_is_valid(asset: &AssetIdentity) -> bool {
    nonempty(&asset.vehicle_ref)
        && nonempty(&asset.pack_ref)
        && is_sha256(&asset.configuration_sha256)
        && is_sha256(&asset.topology_sha256)
}

fn measurement_key(constraint: &str, scope_ref: &str, unit: &str) -> String {
    format!("{constraint}\u{1f}{scope_ref}\u{1f}{unit}")
}

fn canonicalize(corpus: &mut Corpus) {
    corpus.cases.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    for case in &mut corpus.cases {
        case.prediction.required_measurements.sort_by(|left, right| {
            (&left.constraint, &left.scope_ref, &left.unit)
                .cmp(&(&right.constraint, &right.scope_ref, &right.unit))
        });
        case.reference.measurements.sort_by(|left, right| {
            (&left.constraint, &left.scope_ref, &left.unit)
                .cmp(&(&right.constraint, &right.scope_ref, &right.unit))
        });
    }
}

fn validate_case_structure(case: &ValidationCase) -> Vec<String> {
    let mut errors = Vec::new();
    let prediction = &case.prediction;
    let reference = &case.reference;
    if !identity_is_valid(&prediction.asset)
        || !nonempty(&prediction.mission_profile_ref)
        || !is_sha256(&prediction.mission_profile_sha256)
        || !nonempty(&prediction.mission_report_ref)
        || !is_sha256(&prediction.mission_report_sha256)
        || !nonempty(&prediction.evaluator_version)
        || !nonempty(&prediction.model_artifact_ref)
        || !is_sha256(&prediction.model_artifact_sha256)
        || !nonempty(&prediction.split_manifest_ref)
        || !is_sha256(&prediction.split_manifest_sha256)
        || prediction.evaluation_partition != "TEST_BLIND"
        || prediction.split_frozen_at_ms > prediction.prediction_frozen_at_ms
        || prediction.prediction_frozen_at_ms >= reference.observed_at_ms
    {
        errors.push("PREDICTION_IDENTITY_SPLIT_OR_FREEZE_INVALID".to_owned());
    }
    if ![
        "FEASIBLE_WITHIN_DECLARED_ENVELOPE",
        "INFEASIBLE",
        "NOT_COMPUTABLE",
        "UNSUPPORTED_DOMAIN",
        "CONFLICTING_EVIDENCE",
        "REJECTED_INPUT",
    ]
    .contains(&prediction.disposition.as_str())
    {
        errors.push("PREDICTION_DISPOSITION_UNKNOWN".to_owned());
    }
    if prediction.required_measurements.is_empty() {
        errors.push("PREDICTION_HAS_NO_REQUIRED_MEASUREMENTS".to_owned());
    }
    let mut required_keys = BTreeSet::new();
    for required in &prediction.required_measurements {
        if !nonempty(&required.scope_ref)
            || !nonempty(&required.unit)
            || !required.limit_value.is_finite()
            || !valid_relation(&required.constraint, &required.relation)
            || !required_keys.insert(measurement_key(
                &required.constraint,
                &required.scope_ref,
                &required.unit,
            ))
        {
            errors.push("PREDICTION_REQUIRED_MEASUREMENT_INVALID_OR_DUPLICATED".to_owned());
            break;
        }
    }
    if !identity_is_valid(&reference.asset)
        || !nonempty(&reference.mission_profile_ref)
        || !is_sha256(&reference.mission_profile_sha256)
        || !nonempty(&reference.protocol_ref)
        || !is_sha256(&reference.protocol_sha256)
        || !nonempty(&reference.evidence_ref)
        || !is_sha256(&reference.evidence_sha256)
    {
        errors.push("REFERENCE_IDENTITY_OR_DIGEST_INVALID".to_owned());
    }
    if ![
        "SYNTHETIC_KNOWN_TRUTH",
        "INDEPENDENT_PHYSICAL_REFERENCE",
    ]
    .contains(&reference.evidence_class.as_str())
    {
        errors.push("REFERENCE_EVIDENCE_CLASS_UNKNOWN".to_owned());
    }
    if ![
        "MISSION_COMPLETED_WITHIN_LIMITS",
        "BATTERY_CONSTRAINT_EXCEEDED",
        "NON_BATTERY_ABORT",
        "INCONCLUSIVE",
    ]
    .contains(&reference.outcome.as_str())
    {
        errors.push("REFERENCE_OUTCOME_UNKNOWN".to_owned());
    }
    if matches!(
        reference.outcome.as_str(),
        "BATTERY_CONSTRAINT_EXCEEDED" | "NON_BATTERY_ABORT" | "INCONCLUSIVE"
    ) && reference
        .outcome_reason
        .as_deref()
        .is_none_or(|reason| reason.trim().is_empty())
    {
        errors.push("REFERENCE_OUTCOME_REASON_REQUIRED".to_owned());
    }
    for observed in &reference.measurements {
        if !nonempty(&observed.constraint)
            || !nonempty(&observed.scope_ref)
            || !nonempty(&observed.unit)
            || !observed.measured_value.is_finite()
            || !nonempty(&observed.sensor_ref)
            || !nonempty(&observed.calibration_ref)
            || !is_sha256(&observed.calibration_sha256)
            || observed.measured_at_ms <= prediction.prediction_frozen_at_ms
            || observed.measured_at_ms > reference.observed_at_ms
            || !nonempty(&observed.measurement_ref)
            || !is_sha256(&observed.measurement_sha256)
        {
            errors.push("REFERENCE_MEASUREMENT_INVALID".to_owned());
            break;
        }
    }
    errors
}

fn classify_reference(case: &ValidationCase) -> (String, Vec<String>, usize, Option<String>) {
    let prediction = &case.prediction;
    let reference = &case.reference;
    if prediction.asset != reference.asset
        || prediction.mission_profile_ref != reference.mission_profile_ref
        || prediction.mission_profile_sha256 != reference.mission_profile_sha256
    {
        return (
            "CONFLICTING_REFERENCE".to_owned(),
            Vec::new(),
            0,
            Some("independent reference does not bind to the prediction asset/configuration/profile".to_owned()),
        );
    }
    if ![
        "SYNTHETIC_KNOWN_TRUTH",
        "INDEPENDENT_PHYSICAL_REFERENCE",
    ]
    .contains(&reference.evidence_class.as_str())
    {
        return (
            "REJECTED_REFERENCE".to_owned(),
            Vec::new(),
            0,
            Some("reference evidence class is not admitted".to_owned()),
        );
    }

    let mut required = BTreeMap::new();
    for measurement in &prediction.required_measurements {
        let key = measurement_key(
            &measurement.constraint,
            &measurement.scope_ref,
            &measurement.unit,
        );
        if required.insert(key, measurement).is_some() {
            return (
                "REJECTED_REFERENCE".to_owned(),
                Vec::new(),
                0,
                Some("prediction has duplicate required measurement bindings".to_owned()),
            );
        }
    }
    let mut observed = BTreeMap::new();
    for measurement in &reference.measurements {
        let key = measurement_key(
            &measurement.constraint,
            &measurement.scope_ref,
            &measurement.unit,
        );
        if !required.contains_key(&key) {
            return (
                "CONFLICTING_REFERENCE".to_owned(),
                Vec::new(),
                observed.len(),
                Some("reference includes an observation outside the frozen mission measurement contract".to_owned()),
            );
        }
        if observed.insert(key, measurement).is_some() {
            return (
                "CONFLICTING_REFERENCE".to_owned(),
                Vec::new(),
                observed.len(),
                Some("reference duplicates a required measurement binding".to_owned()),
            );
        }
    }
    let covered = observed.len();
    if covered != required.len() {
        return (
            "NOT_SCORABLE".to_owned(),
            Vec::new(),
            covered,
            Some("independent reference does not cover every required mission constraint/scope".to_owned()),
        );
    }

    let mut violations = Vec::new();
    for (key, requirement) in &required {
        let measurement = observed.get(key).expect("coverage checked");
        let within = match requirement.relation.as_str() {
            "MINIMUM_REQUIRED" => measurement.measured_value + EPS >= requirement.limit_value,
            "MAXIMUM_ALLOWED" => measurement.measured_value <= requirement.limit_value + EPS,
            _ => false,
        };
        if !within {
            violations.push(requirement.constraint.clone());
        }
    }
    violations.sort();
    violations.dedup();

    match reference.outcome.as_str() {
        "MISSION_COMPLETED_WITHIN_LIMITS" if violations.is_empty() => {
            ("SCORABLE_FEASIBLE".to_owned(), violations, covered, None)
        }
        "BATTERY_CONSTRAINT_EXCEEDED" if !violations.is_empty() => {
            ("SCORABLE_INFEASIBLE".to_owned(), violations, covered, None)
        }
        "NON_BATTERY_ABORT" if violations.is_empty() => (
            "NOT_SCORABLE".to_owned(),
            violations,
            covered,
            Some("mission aborted for a non-battery reason; battery feasibility is unobserved".to_owned()),
        ),
        "INCONCLUSIVE" => (
            "NOT_SCORABLE".to_owned(),
            violations,
            covered,
            reference.outcome_reason.clone(),
        ),
        "MISSION_COMPLETED_WITHIN_LIMITS" | "BATTERY_CONSTRAINT_EXCEEDED"
        | "NON_BATTERY_ABORT" => (
            "CONFLICTING_REFERENCE".to_owned(),
            violations,
            covered,
            Some("declared physical outcome contradicts measured constraint observations".to_owned()),
        ),
        _ => (
            "NOT_SCORABLE".to_owned(),
            violations,
            covered,
            Some("physical outcome is not scorable".to_owned()),
        ),
    }
}

fn audit_case(case: &ValidationCase) -> CaseAudit {
    let errors = validate_case_structure(case);
    if !errors.is_empty() {
        return CaseAudit {
            case_id: case.case_id.clone(),
            pack_ref: case.prediction.asset.pack_ref.clone(),
            evidence_class: case.reference.evidence_class.clone(),
            predicted_disposition: case.prediction.disposition.clone(),
            reference_outcome: case.reference.outcome.clone(),
            classification: "REJECTED_REFERENCE".to_owned(),
            scoring_eligible: false,
            measurement_count: case.reference.measurements.len(),
            covered_measurement_count: 0,
            violated_constraints: Vec::new(),
            reason: Some(errors.join("; ")),
        };
    }

    let (reference_state, violations, covered, reason) = classify_reference(case);
    let classification = match reference_state.as_str() {
        "SCORABLE_FEASIBLE" => match case.prediction.disposition.as_str() {
            "FEASIBLE_WITHIN_DECLARED_ENVELOPE" => "TRUE_FEASIBLE",
            "INFEASIBLE" => "FALSE_INFEASIBLE",
            _ => "ABSTAINED",
        },
        "SCORABLE_INFEASIBLE" => match case.prediction.disposition.as_str() {
            "FEASIBLE_WITHIN_DECLARED_ENVELOPE" => "FALSE_FEASIBLE",
            "INFEASIBLE" => "TRUE_INFEASIBLE",
            _ => "ABSTAINED",
        },
        "NOT_SCORABLE" => {
            if case.prediction.disposition == "FEASIBLE_WITHIN_DECLARED_ENVELOPE"
                || case.prediction.disposition == "INFEASIBLE"
            {
                "NOT_SCORABLE"
            } else {
                "ABSTAINED"
            }
        }
        "CONFLICTING_REFERENCE" => "CONFLICTING_REFERENCE",
        _ => "REJECTED_REFERENCE",
    }
    .to_owned();

    let scoring_eligible = matches!(
        classification.as_str(),
        "TRUE_FEASIBLE"
            | "FALSE_FEASIBLE"
            | "TRUE_INFEASIBLE"
            | "FALSE_INFEASIBLE"
    );
    CaseAudit {
        case_id: case.case_id.clone(),
        pack_ref: case.prediction.asset.pack_ref.clone(),
        evidence_class: case.reference.evidence_class.clone(),
        predicted_disposition: case.prediction.disposition.clone(),
        reference_outcome: case.reference.outcome.clone(),
        classification,
        scoring_eligible,
        measurement_count: case.reference.measurements.len(),
        covered_measurement_count: covered,
        violated_constraints: violations,
        reason,
    }
}

fn log_choose(n: usize, k: usize) -> f64 {
    let k = k.min(n.saturating_sub(k));
    let mut result = 0.0;
    for i in 1..=k {
        result += ((n + 1 - i) as f64).ln() - (i as f64).ln();
    }
    result
}

fn binomial_cdf_at_most(k: usize, n: usize, p: f64) -> f64 {
    if k >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }

    let log_probability_at_k = log_choose(n, k)
        + (k as f64) * p.ln()
        + ((n - k) as f64) * (-p).ln_1p();
    let mut probability = log_probability_at_k.exp();
    let mut cumulative = probability;
    for i in (1..=k).rev() {
        probability *= (i as f64 / (n - i + 1) as f64) * ((1.0 - p) / p);
        cumulative += probability;
    }
    cumulative.clamp(0.0, 1.0)
}

/// Exact one-sided Clopper-Pearson upper confidence bound for a binomial rate.
/// The bound solves P[X <= observed_errors | n, p_upper] = 1 - confidence.
pub fn clopper_pearson_upper(
    observed_errors: usize,
    trials: usize,
    confidence: f64,
) -> Result<f64, String> {
    if trials == 0 {
        return Err("binomial confidence bound requires at least one trial".to_owned());
    }
    if observed_errors > trials {
        return Err("observed errors cannot exceed the trial count".to_owned());
    }
    if !(0.0 < confidence && confidence < 1.0) {
        return Err("confidence must lie strictly between zero and one".to_owned());
    }
    if observed_errors == trials {
        return Ok(1.0);
    }
    let alpha = 1.0 - confidence;
    let mut lower = observed_errors as f64 / trials as f64;
    let mut upper = 1.0;
    for _ in 0..100 {
        let midpoint = (lower + upper) / 2.0;
        let cdf = binomial_cdf_at_most(observed_errors, trials, midpoint);
        if cdf > alpha {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    Ok(upper)
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<AuditReport, String> {
    let validator = compile_schema(schema_bytes)?;
    let value: Value =
        serde_json::from_slice(corpus_bytes).map_err(|error| format!("corpus JSON: {error}"))?;
    validate_schema(&validator, &value)?;
    let mut corpus: Corpus =
        serde_json::from_value(value).map_err(|error| format!("corpus decode: {error}"))?;
    if corpus.corpus_version != CORPUS_VERSION || corpus.cases.is_empty() {
        return Err("unsupported or empty mission outcome audit corpus".to_owned());
    }
    canonicalize(&mut corpus);

    let canonical = serde_json::to_vec(&corpus)
        .map_err(|error| format!("canonical corpus serialization: {error}"))?;
    let mut case_ids = BTreeSet::new();
    let mut pack_ids = BTreeSet::new();
    for case in &corpus.cases {
        if !case_ids.insert(case.case_id.clone()) {
            return Err(format!("duplicate case identity: {}", case.case_id));
        }
        if !pack_ids.insert(case.prediction.asset.pack_ref.clone()) {
            return Err(format!(
                "duplicate pack identity would violate independent pack-mission sampling: {}",
                case.prediction.asset.pack_ref
            ));
        }
    }

    let cases: Vec<CaseAudit> = corpus.cases.iter().map(audit_case).collect();
    let count = |classification: &str| {
        cases
            .iter()
            .filter(|row| row.classification == classification)
            .count()
    };
    let synthetic = |classification: &str| {
        cases
            .iter()
            .filter(|row| {
                row.evidence_class == "SYNTHETIC_KNOWN_TRUTH"
                    && row.classification == classification
            })
            .count()
    };
    let physical = |classification: &str| {
        cases
            .iter()
            .filter(|row| {
                row.evidence_class == "INDEPENDENT_PHYSICAL_REFERENCE"
                    && row.classification == classification
            })
            .count()
    };
    let physical_predicted_feasible =
        physical("TRUE_FEASIBLE") + physical("FALSE_FEASIBLE");
    let physical_predicted_infeasible =
        physical("TRUE_INFEASIBLE") + physical("FALSE_INFEASIBLE");
    let physical_scorable = physical_predicted_feasible + physical_predicted_infeasible;
    let physical_false_feasible = physical("FALSE_FEASIBLE");
    let physical_false_infeasible = physical("FALSE_INFEASIBLE");
    let physical_reference_case_count = cases
        .iter()
        .filter(|row| row.evidence_class == "INDEPENDENT_PHYSICAL_REFERENCE")
        .count();
    let has_synthetic_references = cases
        .iter()
        .any(|row| row.evidence_class == "SYNTHETIC_KNOWN_TRUTH");

    let source_grade = match (physical_reference_case_count > 0, has_synthetic_references) {
        (false, true) => "SYNTHETIC_ONLY",
        (true, true) => "MIXED_SYNTHETIC_AND_PHYSICAL_REFERENCE_INPUTS",
        (true, false) => "PHYSICAL_REFERENCE_INPUTS_PRESENT_NOT_A_QUALIFICATION",
        (false, false) => "NO_REFERENCE_CLASSIFIED",
    }
    .to_owned();

    let mut report = AuditReport {
        corpus_version: corpus.corpus_version,
        canonical_corpus_sha256: sha256(&canonical),
        report_sha256: String::new(),
        case_count: cases.len(),
        true_feasible_count: count("TRUE_FEASIBLE"),
        false_feasible_count: count("FALSE_FEASIBLE"),
        true_infeasible_count: count("TRUE_INFEASIBLE"),
        false_infeasible_count: count("FALSE_INFEASIBLE"),
        abstained_count: count("ABSTAINED"),
        not_scorable_count: count("NOT_SCORABLE"),
        conflicting_reference_count: count("CONFLICTING_REFERENCE"),
        rejected_reference_count: count("REJECTED_REFERENCE"),
        synthetic_scorable_count: synthetic("TRUE_FEASIBLE")
            + synthetic("FALSE_FEASIBLE")
            + synthetic("TRUE_INFEASIBLE")
            + synthetic("FALSE_INFEASIBLE"),
        physical_reference_case_count,
        physical_scorable_count: physical_scorable,
        physical_predicted_feasible_count: physical_predicted_feasible,
        physical_false_feasible_count: physical_false_feasible,
        physical_predicted_infeasible_count: physical_predicted_infeasible,
        physical_false_infeasible_count: physical_false_infeasible,
        physical_false_feasible_upper_95_one_sided: if physical_predicted_feasible > 0 {
            Some(clopper_pearson_upper(
                physical_false_feasible,
                physical_predicted_feasible,
                CONFIDENCE_95,
            )?)
        } else {
            None
        },
        physical_false_infeasible_upper_95_one_sided: if physical_predicted_infeasible > 0 {
            Some(clopper_pearson_upper(
                physical_false_infeasible,
                physical_predicted_infeasible,
                0.95,
            )?)
        } else {
            None
        },
        source_grade,
        cases,
    };
    let report_bytes = serde_json::to_vec(&report)
        .map_err(|error| format!("report serialization: {error}"))?;
    report.report_sha256 = sha256(&report_bytes);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_mission_outcome_audit_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/mission_outcome_audit_v0_1.json");

    fn report() -> AuditReport {
        audit_corpus(SCHEMA, CORPUS).expect("frozen mission outcome corpus")
    }

    fn row<'a>(report: &'a AuditReport, id: &str) -> &'a CaseAudit {
        report
            .cases
            .iter()
            .find(|case| case.case_id == id)
            .expect("case audit")
    }

    #[test]
    fn frozen_corpus_separates_false_feasible_false_infeasible_and_abstention() {
        let report = report();
        assert_eq!(report.case_count, 11);
        assert_eq!(report.true_feasible_count, 1);
        assert_eq!(report.false_feasible_count, 2);
        assert_eq!(report.true_infeasible_count, 1);
        assert_eq!(report.false_infeasible_count, 1);
        assert_eq!(report.abstained_count, 1);
        assert_eq!(report.not_scorable_count, 3);
        assert_eq!(report.conflicting_reference_count, 2);
        assert_eq!(report.rejected_reference_count, 0);
        assert_eq!(report.synthetic_scorable_count, 5);
        assert_eq!(report.physical_scorable_count, 0);
        assert_eq!(report.source_grade, "SYNTHETIC_ONLY");
        assert_eq!(report.physical_false_feasible_upper_95_one_sided, None);
    }

    #[test]
    fn thermal_and_reserve_misses_are_both_false_feasible() {
        let report = report();
        assert_eq!(row(&report, "false_feasible_thermal").classification, "FALSE_FEASIBLE");
        assert_eq!(row(&report, "false_feasible_reserve").classification, "FALSE_FEASIBLE");
    }

    #[test]
    fn a_correctly_rejected_physical_mission_is_true_infeasible() {
        let report = report();
        let row = row(&report, "infeasible_correct");
        assert_eq!(row.classification, "TRUE_INFEASIBLE");
        assert_eq!(row.violated_constraints, vec!["DISCHARGE_POWER".to_owned()]);
    }

    #[test]
    fn an_overly_conservative_prediction_is_counted_as_false_infeasible() {
        let report = report();
        assert_eq!(row(&report, "false_infeasible").classification, "FALSE_INFEASIBLE");
    }

    #[test]
    fn an_unsupported_prediction_is_an_abstention_not_a_correct_rejection() {
        let report = report();
        assert_eq!(row(&report, "abstained_out_of_domain").classification, "ABSTAINED");
    }

    #[test]
    fn non_battery_aborts_and_inconclusive_runs_are_not_scorable() {
        let report = report();
        assert_eq!(row(&report, "non_battery_abort").classification, "NOT_SCORABLE");
        assert_eq!(row(&report, "inconclusive_reference").classification, "NOT_SCORABLE");
    }

    #[test]
    fn reference_identity_mismatch_is_not_used_as_ground_truth() {
        let report = report();
        assert_eq!(
            row(&report, "reference_asset_mismatch").classification,
            "CONFLICTING_REFERENCE"
        );
    }

    #[test]
    fn reference_outcome_must_agree_with_numeric_measurements() {
        let report = report();
        let row = row(&report, "completed_but_measurement_failed");
        assert_eq!(row.classification, "CONFLICTING_REFERENCE");
        assert_eq!(row.violated_constraints, vec!["TEMPERATURE_MAXIMUM".to_owned()]);
    }

    #[test]
    fn incomplete_physical_measurement_coverage_is_not_scorable() {
        let report = report();
        assert_eq!(
            row(&report, "incomplete_reference_measurements").classification,
            "NOT_SCORABLE"
        );
        assert_eq!(
            row(&report, "incomplete_reference_measurements").covered_measurement_count,
            4
        );
    }

    #[test]
    fn zero_false_feasible_cases_bound_risk_at_5_percent_after_59_independent_packs() {
        let upper = clopper_pearson_upper(0, 59, 0.95).unwrap();
        assert!(upper < 0.05);
        assert!(upper > 0.049);
    }

    #[test]
    fn zero_false_feasible_cases_bound_risk_below_1_percent_after_299_packs() {
        let upper = clopper_pearson_upper(0, 299, 0.95).unwrap();
        assert!(upper < 0.01);
    }

    #[test]
    fn nonzero_false_feasible_count_uses_exact_one_sided_bound() {
        let zero = clopper_pearson_upper(0, 59, 0.95).unwrap();
        let one = clopper_pearson_upper(1, 59, 0.95).unwrap();
        assert!(one > zero);
        assert!(one < 0.1);
    }

    #[test]
    fn duplicate_pack_identity_is_rejected_as_non_independent_sampling() {
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).expect("corpus");
        corpus.cases[1].prediction.asset.pack_ref =
            corpus.cases[0].prediction.asset.pack_ref.clone();
        corpus.cases[1].reference.asset.pack_ref =
            corpus.cases[0].reference.asset.pack_ref.clone();
        let bytes = serde_json::to_vec(&corpus).expect("duplicate-pack corpus");
        let error = audit_corpus(SCHEMA, &bytes).expect_err("duplicate pack must be rejected");
        assert!(error.contains("duplicate pack identity"));
    }

    #[test]
    fn non_temporal_case_and_measurement_order_is_canonical() {
        let original = report();
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).expect("corpus");
        corpus.cases.reverse();
        for case in &mut corpus.cases {
            case.prediction.required_measurements.reverse();
            case.reference.measurements.reverse();
        }
        let bytes = serde_json::to_vec(&corpus).expect("permuted corpus");
        let replay = audit_corpus(SCHEMA, &bytes).expect("permuted audit");
        assert_eq!(original.canonical_corpus_sha256, replay.canonical_corpus_sha256);
        assert_eq!(original.report_sha256, replay.report_sha256);
        assert_eq!(
            serde_json::to_value(&original.cases).unwrap(),
            serde_json::to_value(&replay.cases).unwrap()
        );
    }
    #[test]
    fn physical_false_feasible_bound_uses_only_independent_scorable_packs() {
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).expect("corpus");
        let template = corpus.cases[0].clone();
        corpus.cases.clear();
        for index in 0..59 {
            let mut case = template.clone();
            let suffix = format!("{index:02x}");
            let digest = suffix.repeat(32);
            case.case_id = format!("physical-holdout-{index:03}");
            case.prediction.asset.vehicle_ref = format!("vehicle-holdout-{index:03}");
            case.prediction.asset.pack_ref = format!("pack-holdout-{index:03}");
            case.prediction.asset.configuration_sha256 = digest.clone();
            case.prediction.asset.topology_sha256 = digest.clone();
            case.prediction.mission_profile_ref = format!("mission-holdout-{index:03}");
            case.prediction.mission_profile_sha256 = digest.clone();
            case.prediction.mission_report_ref = format!("mission-report-holdout-{index:03}");
            case.prediction.mission_report_sha256 = digest.clone();
            case.reference.asset = case.prediction.asset.clone();
            case.reference.mission_profile_ref = case.prediction.mission_profile_ref.clone();
            case.reference.mission_profile_sha256 = case.prediction.mission_profile_sha256.clone();
            case.reference.evidence_class = "INDEPENDENT_PHYSICAL_REFERENCE".to_owned();
            case.reference.evidence_ref = format!("physical-reference-{index:03}");
            case.reference.evidence_sha256 = digest.clone();
            case.reference.protocol_sha256 = digest.clone();
            for (measurement_index, measurement) in case.reference.measurements.iter_mut().enumerate() {
                measurement.measurement_ref =
                    format!("physical-measurement-{index:03}-{measurement_index:02}");
                measurement.measurement_sha256 = digest.clone();
            }
            corpus.cases.push(case);
        }
        let bytes = serde_json::to_vec(&corpus).expect("physical holdout corpus");
        let report = audit_corpus(SCHEMA, &bytes).expect("physical holdout audit");
        assert_eq!(report.physical_reference_case_count, 59);
        assert_eq!(report.physical_predicted_feasible_count, 59);
        assert_eq!(report.physical_false_feasible_count, 0);
        let upper = report.physical_false_feasible_upper_95_one_sided.unwrap();
        assert!(upper < 0.05 && upper > 0.049);
    }

    #[test]
    fn only_test_blind_predictions_can_enter_the_outcome_scoreboard() {
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).expect("corpus");
        corpus.cases[0].prediction.evaluation_partition = "TRAIN".to_owned();
        let bytes = serde_json::to_vec(&corpus).expect("training partition corpus");
        let report = audit_corpus(SCHEMA, &bytes).expect("invalid prediction is recorded");
        assert_eq!(row(&report, "feasible_correct").classification, "REJECTED_REFERENCE");
    }

    #[test]
    fn prediction_and_split_must_be_frozen_before_physical_observation() {
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).expect("corpus");
        let item = corpus.cases.iter_mut()
            .find(|item| item.case_id == "feasible_correct")
            .unwrap();
        item.prediction.prediction_frozen_at_ms = item.reference.observed_at_ms;
        let bytes = serde_json::to_vec(&corpus).expect("late prediction corpus");
        let report = audit_corpus(SCHEMA, &bytes).expect("late prediction is rejected");
        assert_eq!(row(&report, "feasible_correct").classification, "REJECTED_REFERENCE");
    }

    #[test]
    fn physical_measurements_must_be_after_prediction_freeze_and_calibration_bound() {
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).expect("corpus");
        let item = corpus.cases.iter_mut()
            .find(|item| item.case_id == "feasible_correct")
            .unwrap();
        item.reference.measurements[0].measured_at_ms =
            item.prediction.prediction_frozen_at_ms;
        let bytes = serde_json::to_vec(&corpus).expect("pre-freeze measurement corpus");
        let report = audit_corpus(SCHEMA, &bytes).expect("pre-freeze observation rejected");
        assert_eq!(row(&report, "feasible_correct").classification, "REJECTED_REFERENCE");
        assert!(row(&report, "feasible_correct").reason.as_deref().unwrap().contains("REFERENCE_MEASUREMENT_INVALID"));
    }

    #[test]
    fn battery_constraint_failure_requires_an_explicit_reason() {
        let mut corpus: Corpus = serde_json::from_slice(CORPUS).expect("corpus");
        let item = corpus.cases.iter_mut()
            .find(|item| item.case_id == "false_feasible_thermal")
            .unwrap();
        item.reference.outcome_reason = None;
        let bytes = serde_json::to_vec(&corpus).expect("missing failure reason corpus");
        let report = audit_corpus(SCHEMA, &bytes).expect("missing failure reason rejected");
        assert_eq!(row(&report, "false_feasible_thermal").classification, "REJECTED_REFERENCE");
    }

}
