use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;

const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";

#[derive(Debug, Serialize)]
struct Diagnostic {
    path: String,
    code: String,
    message: String,
}

#[derive(Default)]
struct Findings {
    diagnostics: Vec<Diagnostic>,
    invalid: bool,
    blocked: bool,
    incomparable: bool,
    incomplete: bool,
}

impl Findings {
    fn push(
        &mut self,
        path: impl Into<String>,
        code: &'static str,
        message: impl Into<String>,
    ) {
        match code {
            "INVALID" => self.invalid = true,
            "BLOCKED" => self.blocked = true,
            "INCOMPARABLE" => self.incomparable = true,
            "INCOMPLETE" => self.incomplete = true,
            _ => {}
        }
        self.diagnostics.push(Diagnostic {
            path: path.into(),
            code: code.to_owned(),
            message: message.into(),
        });
    }

    fn disposition(&self) -> &'static str {
        if self.invalid {
            "INVALID"
        } else if self.incomplete {
            "INCOMPLETE"
        } else if self.blocked {
            "BLOCKED"
        } else if self.incomparable {
            "INCOMPARABLE"
        } else {
            "PERFORMANCE_DRIFT_CORPUS_VALID"
        }
    }
}

#[derive(Debug, Serialize)]
struct AuditReport {
    auditor: &'static str,
    auditor_version: &'static str,
    corpus_id: Option<String>,
    disposition: &'static str,
    claim_ceiling: &'static str,
    case_count: usize,
    decision_point_count: usize,
    scenario_class_count: usize,
    challenge_label_count: usize,
    diagnostics: Vec<Diagnostic>,
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

fn time_is_not_after(
    candidate: Option<&DateTime<Utc>>,
    boundary: Option<&DateTime<Utc>>,
) -> bool {
    matches!((candidate, boundary), (Some(candidate), Some(boundary)) if candidate <= boundary)
}

fn training_cutoff_precedes_observation(
    cutoff: Option<&DateTime<Utc>>,
    observed: Option<&DateTime<Utc>>,
) -> bool {
    matches!((cutoff, observed), (Some(cutoff), Some(observed)) if cutoff < observed)
}

fn interval_is_ordered(prediction: &Value) -> bool {
    let mean = prediction
        .get("predicted_shaft_power_kw")
        .and_then(Value::as_f64);
    let lower = prediction.get("lower_bound_kw").and_then(Value::as_f64);
    let upper = prediction.get("upper_bound_kw").and_then(Value::as_f64);
    matches!((lower, mean, upper), (Some(lower), Some(mean), Some(upper)) if lower <= mean && mean <= upper)
}

fn has_hidden_label(feature_snapshot: &Value) -> bool {
    feature_snapshot.get("challenge_truth").is_some()
        || feature_snapshot.get("expected_disposition").is_some()
        || feature_snapshot.get("primary_cause").is_some()
        || feature_snapshot.get("attribution_supported").is_some()
        || feature_snapshot.get("label_available_at_utc").is_some()
}

fn validate_schema(schema: &Value, corpus: &Value, findings: &mut Findings) {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        findings.push(
            "/$schema",
            "INVALID",
            "only JSON Schema Draft 2020-12 is supported",
        );
        return;
    }

    if !jsonschema::draft202012::meta::is_valid(schema) {
        findings.push(
            "/",
            "INVALID",
            "schema does not validate against the Draft 2020-12 meta-schema",
        );
        return;
    }

    let validator = match jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(schema)
    {
        Ok(validator) => validator,
        Err(error) => {
            findings.push(
                "/",
                "INVALID",
                format!("cannot compile performance-drift schema: {error}"),
            );
            return;
        }
    };

    for error in validator.iter_errors(corpus) {
        findings.push(
            error.instance_path().to_string(),
            "INVALID",
            format!("performance-drift corpus schema violation: {error}"),
        );
    }
}

fn check_case(case: &Value, index: usize, findings: &mut Findings) -> usize {
    let case_path = format!("/cases/{index}");
    let Some(decision_points) = case.get("decision_points").and_then(Value::as_array) else {
        findings.push(
            format!("{case_path}/decision_points"),
            "INVALID",
            "decision_points must be an array",
        );
        return 0;
    };

    let model_cutoff = parse_time(
        case.pointer("/reference_model/trained_through_utc")
            .and_then(Value::as_str),
    );
    if model_cutoff.is_none() {
        findings.push(
            format!("{case_path}/reference_model/trained_through_utc"),
            "INVALID",
            "reference model training cutoff must be a valid RFC3339 timestamp",
        );
    }

    let mut prior_decision_nanos: Option<i64> = None;

    for (point_index, point) in decision_points.iter().enumerate() {
        let point_path = format!("{case_path}/decision_points/{point_index}");
        let decision = parse_time(point.get("decision_time_utc").and_then(Value::as_str));
        let observed = parse_time(
            point.pointer("/feature_snapshot/observed_at_utc")
                .and_then(Value::as_str),
        );
        let available = parse_time(
            point.pointer("/feature_snapshot/available_at_utc")
                .and_then(Value::as_str),
        );

        if decision.is_none() {
            findings.push(
                format!("{point_path}/decision_time_utc"),
                "INVALID",
                "decision time must be a valid RFC3339 timestamp",
            );
        }
        if observed.is_none() {
            findings.push(
                format!("{point_path}/feature_snapshot/observed_at_utc"),
                "INVALID",
                "observation time must be a valid RFC3339 timestamp",
            );
        }
        if available.is_none() {
            findings.push(
                format!("{point_path}/feature_snapshot/available_at_utc"),
                "INVALID",
                "feature availability time must be a valid RFC3339 timestamp",
            );
        }

        let decision_nanos = decision.as_ref().and_then(|value| value.timestamp_nanos_opt());
        if let (Some(current), Some(previous)) = (decision_nanos, prior_decision_nanos) {
            if current <= previous {
                findings.push(
                    format!("{point_path}/decision_time_utc"),
                    "INVALID",
                    "decision points must be strictly increasing in time",
                );
            }
        }
        prior_decision_nanos = decision_nanos;

        if !time_is_not_after(observed.as_ref(), decision.as_ref()) {
            findings.push(
                format!("{point_path}/feature_snapshot/observed_at_utc"),
                "INVALID",
                "an observation later than its decision time is future-data leakage",
            );
        }
        if !time_is_not_after(available.as_ref(), decision.as_ref()) {
            findings.push(
                format!("{point_path}/feature_snapshot/available_at_utc"),
                "INVALID",
                "a feature unavailable at decision time cannot be used by the online policy",
            );
        }

        if let (Some(observed), Some(available)) = (observed.as_ref(), available.as_ref()) {
            if available < observed {
                findings.push(
                    format!("{point_path}/feature_snapshot/available_at_utc"),
                    "INVALID",
                    "feature availability cannot precede the source observation time",
                );
            }
        }

        if let (Some(cutoff), Some(observed)) = (model_cutoff.as_ref(), observed.as_ref()) {
            if !training_cutoff_precedes_observation(Some(cutoff), Some(observed)) {
                findings.push(
                    format!("{case_path}/reference_model/trained_through_utc"),
                    "INVALID",
                    "the reference model must be trained strictly before the scored observation; otherwise the scored observation may leak into its own prediction",
                );
            }
        }
        if !time_is_not_after(model_cutoff.as_ref(), decision.as_ref()) {
            findings.push(
                format!("{case_path}/reference_model/trained_through_utc"),
                "INVALID",
                "reference model training cutoff is later than the decision time",
            );
        }

        let feature_snapshot = point.get("feature_snapshot").unwrap_or(&Value::Null);
        if has_hidden_label(feature_snapshot) {
            findings.push(
                format!("{point_path}/feature_snapshot"),
                "INVALID",
                "evaluator-only challenge truth or label fields leaked into candidate-visible features",
            );
        }

        let calibration = feature_snapshot
            .get("calibration_status")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        match calibration {
            "valid" => {}
            "stale" | "invalid" => findings.push(
                format!("{point_path}/feature_snapshot/calibration_status"),
                "BLOCKED",
                format!("calibration status is {calibration}; measurement-based drift attribution is blocked"),
            ),
            _ => findings.push(
                format!("{point_path}/feature_snapshot/calibration_status"),
                "INCOMPLETE",
                "calibration state is unknown; do not treat measurements as calibrated",
            ),
        }

        if let Some(evidence) = feature_snapshot.get("evidence").and_then(Value::as_array) {
            for (evidence_index, item) in evidence.iter().enumerate() {
                let evidence_path =
                    format!("{point_path}/feature_snapshot/evidence/{evidence_index}");
                let evidence_time =
                    parse_time(item.get("available_at_utc").and_then(Value::as_str));
                if evidence_time.is_none() {
                    findings.push(
                        format!("{evidence_path}/available_at_utc"),
                        "INVALID",
                        "evidence availability time must be a valid RFC3339 timestamp",
                    );
                } else if !time_is_not_after(evidence_time.as_ref(), decision.as_ref()) {
                    findings.push(
                        format!("{evidence_path}/available_at_utc"),
                        "INVALID",
                        "evidence not available by decision time cannot support that decision",
                    );
                }

                if item.get("kind").and_then(Value::as_str) == Some("calibration_record") {
                    match item.get("status").and_then(Value::as_str).unwrap_or("unknown") {
                        "confirmed" => {}
                        "negative" | "revoked" => findings.push(
                            format!("{evidence_path}/status"),
                            "BLOCKED",
                            "calibration evidence is negative or revoked",
                        ),
                        _ => findings.push(
                            format!("{evidence_path}/status"),
                            "INCOMPLETE",
                            "calibration evidence is incomplete or unknown",
                        ),
                    }
                }
            }
        }

        let prediction = point
            .get("reference_prediction")
            .unwrap_or(&Value::Null);
        if !interval_is_ordered(prediction) {
            findings.push(
                format!("{point_path}/reference_prediction"),
                "INVALID",
                "prediction interval must satisfy lower_bound <= predicted_value <= upper_bound",
            );
        }
    }

    let label_time = parse_time(
        case.pointer("/challenge_truth/label_available_at_utc")
            .and_then(Value::as_str),
    );
    if label_time.is_none() {
        findings.push(
            format!("{case_path}/challenge_truth/label_available_at_utc"),
            "INVALID",
            "evaluator label availability must be a valid RFC3339 timestamp",
        );
    }

    decision_points.len()
}

fn check_corpus(corpus: &Value, findings: &mut Findings) -> (usize, usize, usize) {
    if corpus.get("schema_version").and_then(Value::as_str) != Some("0.1.0") {
        findings.push(
            "/schema_version",
            "INVALID",
            "unsupported performance-drift corpus version",
        );
    }

    if corpus.get("corpus_status").and_then(Value::as_str) != Some("synthetic") {
        findings.push(
            "/corpus_status",
            "INVALID",
            "v0.1.0 challenge corpus must explicitly declare synthetic status",
        );
    } else {
        findings.push(
            "/corpus_status",
            "INCOMPLETE",
            "synthetic challenge fixtures can test audit semantics only and can never qualify physical performance or cause attribution",
        );
    }

    let generated = parse_time(corpus.get("generated_at_utc").and_then(Value::as_str));
    if generated.is_none() {
        findings.push(
            "/generated_at_utc",
            "INVALID",
            "corpus generation time must be a valid RFC3339 timestamp",
        );
    }

    let Some(cases) = corpus.get("cases").and_then(Value::as_array) else {
        findings.push("/cases", "INVALID", "cases must be an array");
        return (0, 0, 0);
    };

    let mut ids = BTreeSet::new();
    let mut decision_count = 0;
    let mut scenario_classes = BTreeSet::new();

    for (index, case) in cases.iter().enumerate() {
        let id = case.get("case_id").and_then(Value::as_str).unwrap_or("");
        if id.is_empty() {
            findings.push(
                format!("/cases/{index}/case_id"),
                "INVALID",
                "case_id must be non-empty",
            );
        } else if !ids.insert(id.to_owned()) {
            findings.push(
                format!("/cases/{index}/case_id"),
                "INVALID",
                format!("duplicate case_id '{id}'"),
            );
        }

        if let Some(class) = case.get("scenario_class").and_then(Value::as_str) {
            scenario_classes.insert(class.to_owned());
        }

        decision_count += check_case(case, index, findings);
    }

    (cases.len(), decision_count, scenario_classes.len())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!(
            "usage: marine-performance-drift-audit <schema.json> <synthetic-corpus.json>"
        );
        std::process::exit(64);
    }

    let schema = match read_json(&args[0]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(64);
        }
    };
    let corpus = match read_json(&args[1]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(64);
        }
    };

    let mut findings = Findings::default();
    validate_schema(&schema, &corpus, &mut findings);
    let (case_count, decision_point_count, scenario_class_count) =
        check_corpus(&corpus, &mut findings);

    let challenge_label_count = corpus
        .get("cases")
        .and_then(Value::as_array)
        .map(|cases| {
            cases
                .iter()
                .filter(|case| case.get("challenge_truth").is_some())
                .count()
        })
        .unwrap_or(0);

    let report = AuditReport {
        auditor: "marine-performance-drift-audit",
        auditor_version: "0.1.0",
        corpus_id: corpus
            .get("corpus_id")
            .and_then(Value::as_str)
            .map(str::to_owned),
        disposition: findings.disposition(),
        claim_ceiling: "A valid report establishes only schema and time-causality contract checks for the declared corpus; synthetic inputs never establish vessel performance, fouling cause, cleaning efficacy, or operational permission.",
        case_count,
        decision_point_count,
        scenario_class_count,
        challenge_label_count,
        diagnostics: findings.diagnostics,
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report).unwrap_or_else(|error| {
            format!(
                "{{\"disposition\":\"INVALID\",\"diagnostic\":\"cannot serialize report: {error}\"}}"
            )
        })
    );

    if matches!(
        report.disposition,
        "INVALID" | "BLOCKED" | "INCOMPARABLE" | "INCOMPLETE"
    ) {
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn feature_availability_must_not_follow_decision_time() {
        let decision = parse_time(Some("2026-01-01T08:00:00Z")).unwrap();
        let available = parse_time(Some("2026-01-01T08:00:01Z")).unwrap();
        assert!(!time_is_not_after(Some(&available), Some(&decision)));
    }

    #[test]
    fn model_cutoff_must_precede_scored_observation() {
        let earlier = parse_time(Some("2025-12-31T23:59:59Z")).unwrap();
        let equal = parse_time(Some("2026-01-01T08:00:00Z")).unwrap();
        let later = parse_time(Some("2026-01-01T08:00:01Z")).unwrap();
        assert!(training_cutoff_precedes_observation(Some(&earlier), Some(&equal)));
        assert!(!training_cutoff_precedes_observation(Some(&equal), Some(&equal)));
        assert!(!training_cutoff_precedes_observation(Some(&later), Some(&equal)));
    }

    #[test]
    fn evaluator_truth_is_not_a_candidate_feature() {
        let feature = json!({
            "observed_at_utc": "2026-01-01T08:00:00Z",
            "challenge_truth": {"primary_cause": "biofouling"}
        });
        assert!(has_hidden_label(&feature));
    }

    #[test]
    fn prediction_interval_requires_ordered_bounds() {
        let good = json!({
            "lower_bound_kw": 90.0,
            "predicted_shaft_power_kw": 100.0,
            "upper_bound_kw": 110.0
        });
        let bad = json!({
            "lower_bound_kw": 105.0,
            "predicted_shaft_power_kw": 100.0,
            "upper_bound_kw": 110.0
        });
        assert!(interval_is_ordered(&good));
        assert!(!interval_is_ordered(&bad));
    }

    #[test]
    fn calibration_state_blocks_or_incompletes_claims() {
        let mut findings = Findings::default();
        findings.push("/calibration_status", "BLOCKED", "invalid calibration");
        assert_eq!(findings.disposition(), "BLOCKED");

        let mut synthetic_with_blocked_case = Findings::default();
        synthetic_with_blocked_case.push("/calibration_status", "BLOCKED", "invalid calibration");
        synthetic_with_blocked_case.push("/corpus_status", "INCOMPLETE", "synthetic corpus");
        assert_eq!(synthetic_with_blocked_case.disposition(), "INCOMPLETE");

        let mut unknown = Findings::default();
        unknown.push("/calibration_status", "INCOMPLETE", "unknown calibration");
        assert_eq!(unknown.disposition(), "INCOMPLETE");
    }

    #[test]
    fn synthetic_corpus_never_qualifies_as_physical_evidence() {
        let corpus = json!({
            "schema_version": "0.1.0",
            "corpus_status": "synthetic",
            "generated_at_utc": "2026-11-14T00:00:00Z",
            "cases": []
        });
        let mut findings = Findings::default();
        let (case_count, decision_count, scenario_count) = check_corpus(&corpus, &mut findings);
        assert_eq!((case_count, decision_count, scenario_count), (0, 0, 0));
        assert_eq!(findings.disposition(), "INCOMPLETE");
    }
}
