use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
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
    incomparable: bool,
    incomplete: bool,
}

impl Findings {
    fn push(&mut self, path: impl Into<String>, code: &'static str, message: impl Into<String>) {
        match code {
            "INVALID" => self.invalid = true,
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
        } else if self.incomparable {
            "INCOMPARABLE"
        } else if self.incomplete {
            "INCOMPLETE"
        } else {
            "CONTEXT_LINKS_RESOLVED"
        }
    }
}

#[derive(Debug, Serialize)]
struct JoinReport {
    auditor: &'static str,
    auditor_version: &'static str,
    join_id: Option<String>,
    disposition: &'static str,
    context_links_valid: bool,
    timebase_binding_valid: bool,
    source_record_statuses: Vec<String>,
    claim_ceiling: &'static str,
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

fn validate_schema(schema: &Value, record: &Value, label: &str, findings: &mut Findings) {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        findings.push(format!("/{label}/$schema"), "INVALID", "only Draft 2020-12 JSON Schema is supported");
        return;
    }
    if !jsonschema::draft202012::meta::is_valid(schema) {
        findings.push(format!("/{label}"), "INVALID", "schema does not validate against the Draft 2020-12 meta-schema");
        return;
    }
    let validator = match jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(schema)
    {
        Ok(validator) => validator,
        Err(error) => {
            findings.push(format!("/{label}"), "INVALID", format!("cannot compile schema: {error}"));
            return;
        }
    };
    for error in validator.iter_errors(record) {
        findings.push(
            format!("/{label}{}", error.instance_path()),
            "INVALID",
            format!("{label} record schema violation: {error}"),
        );
    }
}

fn compare_value(
    join: &Value,
    source: &Value,
    join_pointer: &str,
    source_pointer: &str,
    label: &str,
    findings: &mut Findings,
) {
    if join.pointer(join_pointer) != source.pointer(source_pointer) {
        findings.push(
            join_pointer,
            "INVALID",
            format!("{label} in join manifest does not equal the linked source record"),
        );
    }
}

fn check_join(join: &Value, acoustic: &Value, bio: &Value, findings: &mut Findings) -> Vec<String> {
    let mut statuses = Vec::new();
    for (label, record) in [("acoustic", acoustic), ("biofouling", bio)] {
        let status = record.get("record_status").and_then(Value::as_str).unwrap_or("unknown");
        statuses.push(format!("{label}:{status}"));
        if status != "measured" {
            findings.push(
                format!("/{label}/record_status"),
                "INCOMPLETE",
                format!("{label} record is {status}; contextual linkage cannot upgrade synthetic/modelled data to measured evidence"),
            );
        }
    }
    if join.get("record_status").and_then(Value::as_str) != Some("measured") {
        findings.push(
            "/record_status",
            "INCOMPLETE",
            "join manifest is not a measured source record; it is a linkage receipt only",
        );
    }
    if join.get("claim_mode").and_then(Value::as_str) != Some("contextual_linkage_only") {
        findings.push(
            "/claim_mode",
            "INVALID",
            "cross-record join must be limited to contextual_linkage_only",
        );
    }

    for source_name in ["acoustic", "biofouling"] {
        let identity = join.get("vessel_identity");
        let linked = match source_name {
            "acoustic" => acoustic.pointer("/vessel/identity"),
            "biofouling" => bio.pointer("/vessel/identity"),
            _ => None,
        };
        if identity != linked {
            findings.push(
                "/vessel_identity",
                "INCOMPARABLE",
                format!("{source_name} vessel identity/revision differs from join identity"),
            );
        }
    }

    let join_service = join.get("service_fingerprint");
    if acoustic.get("service_fingerprint") != join_service {
        findings.push(
            "/service_fingerprint",
            "INCOMPARABLE",
            "acoustic service fingerprint differs from the join manifest",
        );
    }
    if bio.get("service_fingerprint") != join_service {
        findings.push(
            "/service_fingerprint",
            "INCOMPARABLE",
            "biofouling service fingerprint differs from the join manifest",
        );
    }

    compare_value(
        join,
        acoustic,
        "/acoustic_link/record_id",
        "/record_id",
        "acoustic record_id",
        findings,
    );
    compare_value(
        join,
        acoustic,
        "/acoustic_link/raw_trace_sha256",
        "/raw_trace_sha256",
        "acoustic raw trace digest",
        findings,
    );
    compare_value(
        join,
        acoustic,
        "/acoustic_link/measurement_start",
        "/measurement_window/start",
        "acoustic measurement start",
        findings,
    );
    compare_value(
        join,
        acoustic,
        "/acoustic_link/measurement_end",
        "/measurement_window/end",
        "acoustic measurement end",
        findings,
    );
    compare_value(
        join,
        bio,
        "/hull_inspection_link/record_id",
        "/record_id",
        "biofouling record_id",
        findings,
    );
    compare_value(
        join,
        bio,
        "/hull_inspection_link/inspection_id",
        "/inspection/inspection_id",
        "hull inspection identity",
        findings,
    );
    compare_value(
        join,
        bio,
        "/hull_inspection_link/raw_data_sha256",
        "/inspection/raw_data_sha256",
        "hull inspection digest",
        findings,
    );
    compare_value(
        join,
        bio,
        "/cleaning_link/cleaning_id",
        "/cleaning_event/cleaning_id",
        "cleaning event identity",
        findings,
    );
    compare_value(
        join,
        bio,
        "/cleaning_link/start",
        "/cleaning_event/start",
        "cleaning event start",
        findings,
    );
    compare_value(
        join,
        bio,
        "/cleaning_link/end",
        "/cleaning_event/end",
        "cleaning event end",
        findings,
    );
    compare_value(
        join,
        bio,
        "/performance_links/before_observation_id",
        "/performance_comparison/before/observation_id",
        "pre-intervention observation identity",
        findings,
    );
    compare_value(
        join,
        bio,
        "/performance_links/before_trace_sha256",
        "/performance_comparison/before/source_trace_sha256",
        "pre-intervention trace digest",
        findings,
    );
    compare_value(
        join,
        bio,
        "/performance_links/after_observation_id",
        "/performance_comparison/after/observation_id",
        "post-intervention observation identity",
        findings,
    );
    compare_value(
        join,
        bio,
        "/performance_links/after_trace_sha256",
        "/performance_comparison/after/source_trace_sha256",
        "post-intervention trace digest",
        findings,
    );

    let acoustic_start = parse_time(join.pointer("/acoustic_link/measurement_start").and_then(Value::as_str));
    let acoustic_end = parse_time(join.pointer("/acoustic_link/measurement_end").and_then(Value::as_str));
    let inspection_time = parse_time(join.pointer("/hull_inspection_link/inspection_date").and_then(Value::as_str));
    let cleaning_start = parse_time(join.pointer("/cleaning_link/start").and_then(Value::as_str));
    let cleaning_end = parse_time(join.pointer("/cleaning_link/end").and_then(Value::as_str));
    let maximum_separation = join
        .pointer("/time_alignment_policy/max_acoustic_inspection_separation_seconds")
        .and_then(Value::as_i64)
        .unwrap_or(-1);

    match (acoustic_start.as_ref(), acoustic_end.as_ref()) {
        (Some(start), Some(end)) if start < end => {}
        (Some(_), Some(_)) => findings.push(
            "/acoustic_link/measurement_end",
            "INVALID",
            "acoustic measurement window end must follow its start",
        ),
        _ => findings.push(
            "/acoustic_link",
            "INVALID",
            "acoustic measurement start/end must be valid RFC3339 timestamps",
        ),
    }

    match (
        acoustic_start.as_ref(),
        acoustic_end.as_ref(),
        inspection_time.as_ref(),
    ) {
        (Some(start), Some(end), Some(inspection)) => {
            let separation = if inspection < start {
                start.signed_duration_since(inspection.clone()).num_seconds()
            } else if inspection > end {
                inspection.signed_duration_since(end.clone()).num_seconds()
            } else {
                0
            };
            if separation < 0 || separation > maximum_separation {
                findings.push(
                    "/hull_inspection_link/inspection_date",
                    "INCOMPARABLE",
                    format!(
                        "hull inspection is {separation} seconds from acoustic window; declared maximum is {maximum_separation}"
                    ),
                );
            }
        }
        _ => findings.push(
            "/time_alignment_policy",
            "INVALID",
            "acoustic window and hull inspection time must be valid and resolvable",
        ),
    }

    match (acoustic_end.as_ref(), cleaning_start.as_ref(), cleaning_end.as_ref()) {
        (Some(acoustic_end), Some(cleaning_start), Some(cleaning_end))
            if acoustic_end <= cleaning_start && cleaning_start < cleaning_end => {}
        (Some(_), Some(_), Some(_)) => findings.push(
            "/cleaning_link",
            "INCOMPARABLE",
            "the baseline acoustic measurement must finish before cleaning begins, and cleaning start must precede cleaning end",
        ),
        _ => findings.push(
            "/cleaning_link",
            "INVALID",
            "cleaning timestamps must be valid RFC3339 values",
        ),
    }

    let before = bio.pointer("/performance_comparison/before");
    let after = bio.pointer("/performance_comparison/after");
    if let (Some(acoustic_start), Some(before)) = (
        acoustic_start.as_ref(),
        before,
    ) {
        if parse_time(before.get("observed_at").and_then(Value::as_str)).as_ref() != Some(acoustic_start) {
            findings.push(
                "/performance_links/before_observation_id",
                "INCOMPARABLE",
                "pre-intervention propulsion observation is not aligned to the acoustic baseline start",
            );
        }
    }
    if let (Some(cleaning_end), Some(after)) = (
        cleaning_end.as_ref(),
        after,
    ) {
        match parse_time(after.get("observed_at").and_then(Value::as_str)) {
            Some(after_time) if after_time > cleaning_end.clone() => {}
            Some(_) => findings.push(
                "/performance_links/after_observation_id",
                "INCOMPARABLE",
                "post-intervention propulsion observation must occur after cleaning completion",
            ),
            None => findings.push(
                "/performance_links/after_observation_id",
                "INVALID",
                "post-intervention observation time is invalid",
            ),
        }
    }

    let acoustic_condition = acoustic.get("operating_condition").unwrap_or(&Value::Null);
    let before_condition = before.unwrap_or(&Value::Null);
    for (acoustic_field, before_field, label) in [
        ("speed_through_water_kn", "speed_through_water_kn", "speed through water"),
        ("shaft_rpm", "shaft_rpm", "shaft RPM"),
        ("draft_m", "draft_m", "draft"),
        ("trim_m", "trim_m", "trim"),
        ("shaft_power_kw", "shaft_power_kw", "baseline shaft power"),
    ] {
        if acoustic_condition.get(acoustic_field) != before_condition.get(before_field) {
            findings.push(
                format!("/performance_links/before_observation_id/{label}"),
                "INCOMPARABLE",
                format!("acoustic baseline and hull performance baseline differ in {label}"),
            );
        }
    }

    let context_links_valid =
        !findings.invalid && !findings.incomparable;
    statuses
}


fn check_timebase_binding(
    binding: Option<&Value>,
    acoustic: &Value,
    join: &Value,
    findings: &mut Findings,
) -> bool {
    let Some(binding) = binding else {
        findings.push(
            "/timebase_binding",
            "INCOMPLETE",
            "acoustic-to-trial timebase binding receipt was not supplied",
        );
        return false;
    };

    let mut matched = true;
    for (report_pointer, expected, label) in [
        (
            "/acoustic_record_id",
            acoustic.get("record_id"),
            "acoustic record identity",
        ),
        (
            "/service_fingerprint",
            acoustic.get("service_fingerprint"),
            "service fingerprint",
        ),
        (
            "/acoustic_raw_trace_sha256",
            acoustic.get("raw_trace_sha256"),
            "acoustic raw-trace digest",
        ),
        (
            "/service_fingerprint",
            join.get("service_fingerprint"),
            "join service fingerprint",
        ),
    ] {
        if binding.pointer(report_pointer) != expected {
            findings.push(
                "/timebase_binding",
                "INCOMPARABLE",
                format!("timebase binding receipt does not match the linked {label}"),
            );
            matched = false;
        }
    }

    if binding.get("binding_valid").and_then(Value::as_bool) != Some(true) {
        let disposition = binding.get("disposition").and_then(Value::as_str).unwrap_or("unknown");
        let code = match disposition {
            "INVALID" => "INVALID",
            "INCOMPARABLE" => "INCOMPARABLE",
            _ => "INCOMPLETE",
        };
        findings.push(
            "/timebase_binding/binding_valid",
            code,
            "shared-clock binding is absent, unresolved or not structurally valid",
        );
        matched = false;
    }
    if binding.get("common_reference_clock_id").and_then(Value::as_str).is_none_or(str::is_empty)
        || binding.get("acoustic_timebase_id").and_then(Value::as_str).is_none_or(str::is_empty)
    {
        findings.push(
            "/timebase_binding",
            "INCOMPLETE",
            "binding receipt does not identify the common reference clock and acoustic timebase",
        );
        matched = false;
    }
    matched
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 && args.len() != 8 {
        return Err("usage: marine-hull-state-evidence-join-audit <join.schema.json> <join.json> <acoustic.schema.json> <acoustic.json> <bio.schema.json> <bio.json> [timebase-binding-report.json]".to_owned());
    }
    let join_schema = read_json(&args[1])?;
    let join = read_json(&args[2])?;
    let acoustic_schema = read_json(&args[3])?;
    let acoustic = read_json(&args[4])?;
    let bio_schema = read_json(&args[5])?;
    let bio = read_json(&args[6])?;
    let binding_report = if args.len() == 8 {
        Some(read_json(&args[7])?)
    } else {
        None
    };

    let mut findings = Findings::default();
    validate_schema(&join_schema, &join, "join", &mut findings);
    validate_schema(&acoustic_schema, &acoustic, "acoustic", &mut findings);
    validate_schema(&bio_schema, &bio, "biofouling", &mut findings);
    let statuses = check_join(&join, &acoustic, &bio, &mut findings);
    let context_links_valid = !findings.invalid && !findings.incomparable;
    let timebase_binding_valid =
        check_timebase_binding(binding_report.as_ref(), &acoustic, &join, &mut findings);

    let report = JoinReport {
        auditor: "marine-hull-state-evidence-join-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        join_id: join.get("join_id").and_then(Value::as_str).map(str::to_owned),
        disposition: findings.disposition(),
        context_links_valid,
        timebase_binding_valid,
        source_record_statuses: statuses,
        claim_ceiling: "Identity, service, trace-reference, shared-clock/timebase and event-time join checks only. A context join is not source authenticity, calibration qualification, causal inference, measured fuel reduction, acoustic reduction, organism identification or ecological impact evidence.",
        diagnostics: findings.diagnostics,
    };

    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(if report.disposition == "CONTEXT_LINKS_RESOLVED" { 0 } else { 2 })
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const JOIN_SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-hull-state-evidence-join-v0.1.0.schema.json");
    const JOIN_FIXTURE: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-marine-hull-state-evidence-join-v0.1.0.json");
    const ACOUSTIC_SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-acoustic-evidence-v0.1.0.schema.json");
    const ACOUSTIC_FIXTURE: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-marine-acoustic-evidence-v0.1.0.json");
    const BIO_SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-biofouling-evidence-v0.1.0.schema.json");
    const BIO_FIXTURE: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-marine-biofouling-evidence-v0.1.0.json");

    fn records() -> (Value, Value, Value) {
        (
            serde_json::from_str(JOIN_FIXTURE).expect("join fixture JSON"),
            serde_json::from_str(ACOUSTIC_FIXTURE).expect("acoustic fixture JSON"),
            serde_json::from_str(BIO_FIXTURE).expect("biofouling fixture JSON"),
        )
    }

    #[test]
    fn committed_join_and_source_fixtures_match_their_schemas() {
        let (join, acoustic, bio) = records();
        for (schema_text, record) in [
            (JOIN_SCHEMA, &join),
            (ACOUSTIC_SCHEMA, &acoustic),
            (BIO_SCHEMA, &bio),
        ] {
            let schema: Value = serde_json::from_str(schema_text).expect("schema JSON");
            assert!(jsonschema::draft202012::meta::is_valid(&schema));
            let validator = jsonschema::draft202012::options()
                .should_validate_formats(true)
                .build(&schema)
                .expect("schema compiles");
            assert!(validator.is_valid(record));
        }
    }

    #[test]
    fn synthetic_context_join_resolves_but_remains_incomplete() {
        let (join, acoustic, bio) = records();
        let mut findings = Findings::default();
        let statuses = check_join(&join, &acoustic, &bio, &mut findings);
        assert_eq!(
            statuses,
            vec![
                "acoustic:synthetic".to_owned(),
                "biofouling:synthetic".to_owned()
            ]
        );
        assert!(!findings.invalid);
        assert!(!findings.incomparable);
        assert!(findings.incomplete);
    }

    #[test]
    fn measured_source_records_can_resolve_context_without_claiming_causality() {
        let (mut join, mut acoustic, mut bio) = records();
        join["record_status"] = serde_json::json!("measured");
        acoustic["record_status"] = serde_json::json!("measured");
        bio["record_status"] = serde_json::json!("measured");
        let mut findings = Findings::default();
        let statuses = check_join(&join, &acoustic, &bio, &mut findings);
        assert_eq!(findings.disposition(), "CONTEXT_LINKS_RESOLVED");
        assert_eq!(statuses, vec!["acoustic:measured", "biofouling:measured"]);
        assert!(!findings.invalid);
        assert!(!findings.incomparable);
        assert!(!findings.incomplete);
    }

    #[test]
    fn cross_vessel_revision_is_incomparable() {
        let (join, mut acoustic, bio) = records();
        acoustic["vessel"]["identity"]["revision"] = serde_json::json!("B");
        let mut findings = Findings::default();
        check_join(&join, &acoustic, &bio, &mut findings);
        assert!(findings.incomparable);
    }

    #[test]
    fn service_fingerprint_mismatch_is_incomparable() {
        let (join, acoustic, mut bio) = records();
        bio["service_fingerprint"] =
            serde_json::json!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        let mut findings = Findings::default();
        check_join(&join, &acoustic, &bio, &mut findings);
        assert!(findings.incomparable);
    }

    #[test]
    fn raw_trace_digest_mismatch_is_invalid() {
        let (join, mut acoustic, bio) = records();
        acoustic["raw_trace_sha256"] =
            serde_json::json!("9999999999999999999999999999999999999999999999999999999999999999");
        let mut findings = Findings::default();
        check_join(&join, &acoustic, &bio, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn inspection_outside_declared_time_alignment_is_incomparable() {
        let (join, mut acoustic, bio) = records();
        acoustic["measurement_window"]["start"] =
            serde_json::json!("2026-10-14T10:00:00Z");
        acoustic["measurement_window"]["end"] =
            serde_json::json!("2026-10-14T10:01:00Z");
        join["acoustic_link"]["measurement_start"] =
            serde_json::json!("2026-10-14T10:00:00Z");
        join["acoustic_link"]["measurement_end"] =
            serde_json::json!("2026-10-14T10:01:00Z");
        // The join now agrees with the acoustic trace, but the hull inspection falls outside the declared time tolerance.
        let mut findings = Findings::default();
        check_join(&join, &acoustic, &bio, &mut findings);
        assert!(findings.incomparable);
    }

    #[test]
    fn acoustic_baseline_must_precede_cleaning() {
        let (mut join, acoustic, bio) = records();
        join["cleaning_link"]["start"] =
            serde_json::json!("2026-10-15T10:00:30Z");
        join["cleaning_link"]["end"] =
            serde_json::json!("2026-10-15T11:00:00Z");
        let mut findings = Findings::default();
        check_join(&join, &acoustic, &bio, &mut findings);
        assert!(findings.incomparable);
    }

    #[test]
    fn acoustic_speed_mismatch_is_incomparable() {
        let (join, mut acoustic, bio) = records();
        acoustic["operating_condition"]["speed_through_water_kn"] = serde_json::json!(13);
        let mut findings = Findings::default();
        check_join(&join, &acoustic, &bio, &mut findings);
        assert!(findings.incomparable);
    }

    fn synthetic_binding_receipt(join: &Value, acoustic: &Value) -> Value {
        serde_json::json!({
            "acoustic_record_id": acoustic["record_id"],
            "service_fingerprint": acoustic["service_fingerprint"],
            "acoustic_raw_trace_sha256": acoustic["raw_trace_sha256"],
            "common_reference_clock_id": "SYNTHETIC-UTC-REFERENCE-001",
            "acoustic_timebase_id": "SYNTHETIC-ACOUSTIC-TIMEBASE-001",
            "binding_valid": true,
            "disposition": "INCOMPLETE",
            "claim_ceiling": "Synthetic linkage fixture only."
        })
    }

    #[test]
    fn absent_shared_clock_receipt_is_incomplete() {
        let (join, acoustic, _) = records();
        let mut findings = Findings::default();
        let valid = check_timebase_binding(None, &acoustic, &join, &mut findings);
        assert!(!valid);
        assert!(findings.incomplete);
    }

    #[test]
    fn shared_clock_receipt_must_match_exact_acoustic_trace() {
        let (join, acoustic, _) = records();
        let mut receipt = synthetic_binding_receipt(&join, &acoustic);
        receipt["acoustic_raw_trace_sha256"] = serde_json::json!("f".repeat(64));
        let mut findings = Findings::default();
        let valid = check_timebase_binding(Some(&receipt), &acoustic, &join, &mut findings);
        assert!(!valid);
        assert!(findings.incomparable);
    }

    #[test]
    fn synthetic_shared_clock_receipt_resolves_context_without_qualifying_sources() {
        let (join, acoustic, bio) = records();
        let receipt = synthetic_binding_receipt(&join, &acoustic);
        let mut findings = Findings::default();
        let statuses = check_join(&join, &acoustic, &bio, &mut findings);
        let binding_valid =
            check_timebase_binding(Some(&receipt), &acoustic, &join, &mut findings);
        assert!(binding_valid);
        assert_eq!(
            statuses,
            vec!["acoustic:synthetic".to_owned(), "biofouling:synthetic".to_owned()]
        );
        assert!(!findings.invalid);
        assert!(!findings.incomparable);
        assert!(findings.incomplete);
        assert_eq!(findings.disposition(), "INCOMPLETE");
    }

    #[test]
    fn rejected_binding_receipt_cannot_resolve_timebase_join() {
        let (join, acoustic, _) = records();
        let mut receipt = synthetic_binding_receipt(&join, &acoustic);
        receipt["binding_valid"] = serde_json::json!(false);
        receipt["disposition"] = serde_json::json!("INCOMPARABLE");
        let mut findings = Findings::default();
        let valid = check_timebase_binding(Some(&receipt), &acoustic, &join, &mut findings);
        assert!(!valid);
        assert!(findings.incomparable);
    }

}
