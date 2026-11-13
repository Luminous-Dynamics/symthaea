use chrono::DateTime;
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

#[derive(Debug, Serialize)]
struct AuditReport {
    auditor: &'static str,
    auditor_version: &'static str,
    record_id: Option<String>,
    disposition: &'static str,
    claim_ceiling: &'static str,
    expected_gases: usize,
    observed_gases: usize,
    explicitly_unmeasured_gases: usize,
    diagnostics: Vec<Diagnostic>,
}

#[derive(Default)]
struct Findings {
    diagnostics: Vec<Diagnostic>,
    invalid: bool,
    incomplete: bool,
}

impl Findings {
    fn push(&mut self, path: impl Into<String>, code: &'static str, message: impl Into<String>) {
        match code {
            "INVALID" => self.invalid = true,
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
        } else {
            "EMISSIONS_EVIDENCE_CONTRACT_VALID"
        }
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid JSON in {path}: {error}"))
}

fn validate_schema(schema: &Value, instance: &Value, findings: &mut Findings) {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        findings.push("/$schema", "INVALID", "only JSON Schema Draft 2020-12 is supported");
        return;
    }
    if !jsonschema::draft202012::meta::is_valid(schema) {
        findings.push("/", "INVALID", "schema does not validate against the Draft 2020-12 meta-schema");
        return;
    }
    let validator = match jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(schema)
    {
        Ok(validator) => validator,
        Err(error) => {
            findings.push("/", "INVALID", format!("cannot compile schema: {error}"));
            return;
        }
    };
    for error in validator.iter_errors(instance) {
        findings.push(
            error.instance_path().to_string(),
            "INVALID",
            format!("emissions record schema violation: {error}"),
        );
    }
}

fn check_operating_context(record: &Value, findings: &mut Findings) {
    let context = record.get("measurement_context").unwrap_or(&Value::Null);
    let fingerprint = context.get("service_fingerprint").and_then(Value::as_str);
    if fingerprint.is_none_or(|value|
        value.len() != 64 || !value.chars().all(|character| character.is_ascii_hexdigit())
    ) {
        findings.push(
            "/measurement_context/service_fingerprint",
            "INVALID",
            "v0.2 measurement context requires a 64-character hexadecimal service fingerprint",
        );
    }

    let installation = context.get("wind_assist_installation").and_then(Value::as_str);
    let states: &[Value] = context
        .get("wind_device_states")
        .and_then(Value::as_array)
        .map(|values| values.as_slice())
        .unwrap_or(&[]);
    match installation {
        Some("installed") if states.is_empty() => findings.push(
            "/measurement_context/wind_device_states",
            "INCOMPLETE",
            "wind assistance is declared installed but no device operating states are recorded",
        ),
        Some("not_installed") if !states.is_empty() => findings.push(
            "/measurement_context/wind_device_states",
            "INVALID",
            "wind assistance is declared not installed but device states are present",
        ),
        Some("unknown") | None => findings.push(
            "/measurement_context/wind_assist_installation",
            "INCOMPLETE",
            "wind-assist installation status is unknown",
        ),
        _ => {}
    }

    let mut device_ids = BTreeSet::new();
    for (index, device) in states.iter().enumerate() {
        let path = format!("/measurement_context/wind_device_states/{index}");
        if let Some(id) = device.get("device_id").and_then(Value::as_str) {
            if !device_ids.insert(id.to_owned()) {
                findings.push(
                    format!("{path}/device_id"),
                    "INVALID",
                    format!("duplicate wind-device identity '{id}'"),
                );
            }
        }
        let commanded = device.get("commanded_state").and_then(Value::as_str);
        let achieved = device.get("achieved_state").and_then(Value::as_str);
        if achieved == Some("unknown") || achieved.is_none() {
            findings.push(
                format!("{path}/achieved_state"),
                "INCOMPLETE",
                "achieved wind-device state is unknown; commanded state is not a substitute",
            );
        } else if commanded.is_some() && commanded != achieved {
            findings.push(
                &path,
                "INCOMPLETE",
                "commanded and achieved WAPS states differ; preserve this as non-adherence and distinguish intention-to-treat from treatment-on-treated analysis",
            );
        }
    }

    let loading = context.get("loading_condition").and_then(Value::as_str);
    if loading == Some("unknown") || loading.is_none() {
        findings.push(
            "/measurement_context/loading_condition",
            "INCOMPLETE",
            "loading condition is unknown",
        );
    }

    let has_active_device = states.iter().any(|device| {
        matches!(
            device.get("achieved_state").and_then(Value::as_str),
            Some("active") | Some("feathered") | Some("stowed") | Some("retracted") | Some("faulted")
        )
    });
    if installation == Some("installed") && has_active_device {
        for field in [
            "true_wind_speed_kn",
            "true_wind_from_bow_deg",
            "apparent_wind_speed_kn",
            "apparent_wind_from_bow_deg",
        ] {
            if context.get(field).and_then(Value::as_f64).is_none() {
                findings.push(
                    format!("/measurement_context/{field}"),
                    "INCOMPLETE",
                    "wind-device exposure is recorded without the corresponding wind condition",
                );
            }
        }
    }
}

fn check_record(record: &Value, findings: &mut Findings) {
    let schema_version = record.get("schema_version").and_then(Value::as_str).unwrap_or("");
    if schema_version != "0.1.0" && schema_version != "0.2.0" {
        findings.push("/schema_version", "INVALID", "unsupported gas emissions evidence schema version");
    }
    if schema_version == "0.2.0" {
        check_operating_context(record, findings);
    }

    if record.get("record_status").and_then(Value::as_str) == Some("synthetic") {
        findings.push(
            "/record_status",
            "INCOMPLETE",
            "synthetic software fixture cannot establish physical emissions performance",
        );
    }

    let method = record.get("measurement_method").and_then(Value::as_str).unwrap_or("");
    let required_resolution = match method {
        "test_bed_measurement" | "onboard_measurement" => Some("MEPC.414(84)"),
        "engine_load_monitoring" => Some("MEPC.415(84)"),
        "cems" => Some("MEPC.416(84)"),
        _ => None,
    };
    let declared_resolutions: BTreeSet<String> = record
        .get("source_resolution_ids")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    if let Some(required) = required_resolution {
        if !declared_resolutions.contains(required) {
            findings.push(
                "/source_resolution_ids",
                "INVALID",
                format!("measurement method '{method}' must cite {required}"),
            );
        }
    } else {
        findings.push("/measurement_method", "INVALID", "unsupported measurement method");
    }

    let window = record.get("measurement_window").unwrap_or(&Value::Null);
    let start = window.get("start").and_then(Value::as_str);
    let end = window.get("end").and_then(Value::as_str);
    let parsed_window = match (start, end) {
        (Some(start), Some(end)) => match (
            DateTime::parse_from_rfc3339(start),
            DateTime::parse_from_rfc3339(end),
        ) {
            (Ok(start), Ok(end)) if start < end => Some((start, end)),
            (Ok(_), Ok(_)) => {
                findings.push(
                    "/measurement_window/end",
                    "INVALID",
                    "measurement window end must be strictly later than start",
                );
                None
            }
            _ => {
                findings.push(
                    "/measurement_window",
                    "INVALID",
                    "measurement window timestamps must be valid RFC3339 date-times",
                );
                None
            }
        },
        _ => None,
    };

    let expected_count = window.get("expected_sample_count").and_then(Value::as_u64);
    let observed_count = window.get("observed_sample_count").and_then(Value::as_u64);
    let missing_count = window.get("missing_sample_count").and_then(Value::as_u64);
    if let (Some(expected), Some(observed), Some(missing)) =
        (expected_count, observed_count, missing_count)
    {
        if expected == 0 || observed.saturating_add(missing) != expected {
            findings.push(
                "/measurement_window",
                "INVALID",
                "sample counts must satisfy expected > 0 and observed + missing = expected",
            );
        } else {
            let minimum = window.get("minimum_observed_fraction").and_then(Value::as_f64);
            if let Some(minimum) = minimum {
                let coverage = observed as f64 / expected as f64;
                if coverage < minimum {
                    findings.push(
                        "/measurement_window/observed_sample_count",
                        "INCOMPLETE",
                        format!("sample coverage {coverage:.6} is below the predeclared minimum {minimum:.6}"),
                    );
                }
            }
        }
    }

    if window.get("window_semantics").and_then(Value::as_str)
        != Some("start_inclusive_end_exclusive")
    {
        findings.push(
            "/measurement_window/window_semantics",
            "INVALID",
            "v0.1 windows must be start-inclusive/end-exclusive",
        );
    }

    if let (Some((start, end)), Some(expected), Some(rate), Some(tolerance)) = (
        parsed_window.as_ref(),
        expected_count,
        window.get("sample_rate_hz").and_then(Value::as_f64),
        window.get("rate_count_tolerance_samples").and_then(Value::as_f64),
    ) {
        let duration_ns = end.clone().signed_duration_since(start.clone()).num_nanoseconds();
        match duration_ns {
            Some(duration_ns) if duration_ns > 0 => {
                let implied = duration_ns as f64 / 1_000_000_000.0 * rate;
                let difference = (expected as f64 - implied).abs();
                if difference > tolerance {
                    findings.push(
                        "/measurement_window/expected_sample_count",
                        "INVALID",
                        format!("expected sample count {expected} differs from rate/window estimate {implied:.6} by more than tolerance {tolerance:.6}"),
                    );
                }
            }
            _ => findings.push(
                "/measurement_window",
                "INVALID",
                "measurement duration cannot be represented as a positive nanosecond interval",
            ),
        }
    }

    let expected_gases: BTreeSet<String> = record
        .get("expected_gases")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let observations = record.get("gas_observations").and_then(Value::as_array);
    let mut observed_gases = BTreeSet::new();
    let mut observation_ids = BTreeSet::new();

    if let Some(observations) = observations {
        for (index, observation) in observations.iter().enumerate() {
            let path = format!("/gas_observations/{index}");
            if let Some(id) = observation.get("observation_id").and_then(Value::as_str) {
                if !observation_ids.insert(id.to_owned()) {
                    findings.push(format!("{path}/observation_id"), "INVALID", format!("duplicate observation id '{id}'"));
                }
            }
            let gas = observation.get("gas").and_then(Value::as_str);
            if let Some(gas) = gas {
                if !observed_gases.insert(gas.to_owned()) {
                    findings.push(format!("{path}/gas"), "INVALID", format!("duplicate gas observation '{gas}' in one record"));
                }
                if !expected_gases.contains(gas) {
                    findings.push(format!("{path}/gas"), "INVALID", format!("gas '{gas}' is not in expected_gases"));
                }
            }

            if observation.get("mass_rate_kg_per_hour").and_then(Value::as_f64).is_none_or(|rate| rate < 0.0) {
                findings.push(
                    format!("{path}/mass_rate_kg_per_hour"),
                    "INVALID",
                    "gas mass rate must be numeric and non-negative; unknown values must be represented as unmeasured, not zero",
                );
            }

            match observation.pointer("/uncertainty/status").and_then(Value::as_str) {
                Some("quantified") => {
                    let standard = observation.pointer("/uncertainty/standard_uncertainty_kg_per_hour").and_then(Value::as_f64);
                    let coverage = observation.pointer("/uncertainty/coverage_factor").and_then(Value::as_f64);
                    if standard.is_none_or(|value| value < 0.0) || coverage.is_none_or(|value| value <= 0.0) {
                        findings.push(format!("{path}/uncertainty"), "INVALID", "quantified uncertainty requires non-negative standard uncertainty and positive coverage factor");
                    }
                }
                Some("unknown") => findings.push(
                    format!("{path}/uncertainty"),
                    "INCOMPLETE",
                    "gas measurement uncertainty is explicitly unknown; no default is inferred",
                ),
                _ => findings.push(format!("{path}/uncertainty"), "INVALID", "uncertainty state must be quantified or unknown"),
            }

            let evidence = observation.get("evidence").unwrap_or(&Value::Null);
            let mode = evidence.get("mode").and_then(Value::as_str).unwrap_or("");
            match (method, mode) {
                ("test_bed_measurement" | "onboard_measurement" | "cems", "measured") => {
                    match evidence.get("calibration_status").and_then(Value::as_str) {
                        Some("in_tolerance") => {}
                        Some("out_of_tolerance") => findings.push(
                            format!("{path}/evidence/calibration_status"),
                            "INVALID",
                            "out-of-tolerance calibration invalidates the gas observation",
                        ),
                        Some("stale") | Some("unknown") | None => findings.push(
                            format!("{path}/evidence/calibration_status"),
                            "INCOMPLETE",
                            "calibration status is not evidenced as in tolerance",
                        ),
                        _ => findings.push(format!("{path}/evidence/calibration_status"), "INVALID", "unsupported calibration status"),
                    }
                    if evidence.get("rights_status").and_then(Value::as_str) == Some("unknown") {
                        findings.push(format!("{path}/evidence/rights_status"), "INCOMPLETE", "measurement trace rights/access status is unknown");
                    }
                    if evidence.get("timebase_id") != window.get("timebase_id") {
                        findings.push(format!("{path}/evidence/timebase_id"), "INVALID", "gas observation timebase does not match the measurement window timebase");
                    }
                }
                ("engine_load_monitoring", "engine_load_calculated") => {
                    if evidence.get("rights_status").and_then(Value::as_str) == Some("unknown") {
                        findings.push(format!("{path}/evidence/rights_status"), "INCOMPLETE", "engine-load input trace rights/access status is unknown");
                    }
                }
                _ => findings.push(
                    format!("{path}/evidence/mode"),
                    "INVALID",
                    format!("measurement method '{method}' is incompatible with evidence mode '{mode}'"),
                ),
            }
        }
    }

    let unmeasured = record.get("unmeasured_gases").and_then(Value::as_array);
    let mut explicitly_unmeasured = BTreeSet::new();
    if let Some(unmeasured) = unmeasured {
        for (index, item) in unmeasured.iter().enumerate() {
            let path = format!("/unmeasured_gases/{index}");
            if let Some(gas) = item.get("gas").and_then(Value::as_str) {
                if !explicitly_unmeasured.insert(gas.to_owned()) {
                    findings.push(format!("{path}/gas"), "INVALID", format!("duplicate unmeasured gas '{gas}'"));
                }
                if !expected_gases.contains(gas) {
                    findings.push(format!("{path}/gas"), "INVALID", format!("unmeasured gas '{gas}' is not in expected_gases"));
                }
                if observed_gases.contains(gas) {
                    findings.push(format!("{path}/gas"), "INVALID", format!("gas '{gas}' is both observed and explicitly unmeasured"));
                }
            }
        }
    }

    for gas in expected_gases.iter().filter(|gas|
        !observed_gases.contains(*gas) && !explicitly_unmeasured.contains(*gas)
    ) {
        findings.push(
            "/unmeasured_gases",
            "INCOMPLETE",
            format!("expected gas '{gas}' has no observation and no explicit unmeasured reason; it must not be assumed to be zero"),
        );
    }

    let rated = record.pointer("/engine/rated_power_kw").and_then(Value::as_f64);
    let load = record.pointer("/engine/average_load_percent").and_then(Value::as_f64);
    if rated.is_none_or(|value| value <= 0.0) || load.is_none_or(|value| !(0.0..=100.0).contains(&value)) {
        findings.push("/engine", "INVALID", "engine rated power and average load must be within their declared physical domains");
    }
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: marine-emissions-evidence-audit <schema.json> <record.json>".to_owned());
    }
    let schema = read_json(&args[1])?;
    let record = read_json(&args[2])?;
    let mut findings = Findings::default();
    validate_schema(&schema, &record, &mut findings);
    check_record(&record, &mut findings);

    let report = AuditReport {
        auditor: "marine-emissions-evidence-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        record_id: record.get("record_id").and_then(Value::as_str).map(str::to_owned),
        disposition: findings.disposition(),
        claim_ceiling: "Schema and emissions-evidence consistency only. This does not validate analyzer truth, calibrator traceability, statutory compliance, emissions superiority or sensor authenticity.",
        expected_gases: record.get("expected_gases").and_then(Value::as_array).map_or(0, Vec::len),
        observed_gases: record.get("gas_observations").and_then(Value::as_array).map_or(0, Vec::len),
        explicitly_unmeasured_gases: record.get("unmeasured_gases").and_then(Value::as_array).map_or(0, Vec::len),
        diagnostics: findings.diagnostics,
    };
    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(if report.disposition == "EMISSIONS_EVIDENCE_CONTRACT_VALID" { 0 } else { 2 })
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

    const EMISSIONS_SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-emissions-evidence-v0.1.0.schema.json");
    const SYNTHETIC_EMISSIONS: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-gas-emissions-v0.1.0.json");
    const EMISSIONS_SCHEMA_V2: &str =
        include_str!("../../../../docs/research/marine/marine-emissions-evidence-v0.2.0.schema.json");
    const SYNTHETIC_EMISSIONS_V2: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-gas-emissions-v0.2.0.json");

    #[test]
    fn committed_emissions_schema_and_fixture_validate() {
        let schema: Value = serde_json::from_str(EMISSIONS_SCHEMA).expect("emissions schema JSON");
        let record: Value = serde_json::from_str(SYNTHETIC_EMISSIONS).expect("emissions fixture JSON");
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("emissions schema compiles");
        assert!(validator.is_valid(&record));
    }

    #[test]
    fn synthetic_emissions_fixture_is_never_a_physical_result() {
        let record: Value = serde_json::from_str(SYNTHETIC_EMISSIONS).expect("emissions fixture JSON");
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.incomplete);
        assert!(!findings.invalid, "{:?}", findings.diagnostics);
    }

    #[test]
    fn v0_2_context_schema_and_fixture_validate() {
        let schema: Value = serde_json::from_str(EMISSIONS_SCHEMA_V2).expect("emissions v0.2 schema JSON");
        let record: Value = serde_json::from_str(SYNTHETIC_EMISSIONS_V2).expect("emissions v0.2 fixture JSON");
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("emissions v0.2 schema compiles");
        assert!(validator.is_valid(&record));
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.incomplete); // Synthetic, rights unknown and uncertainty unknown.
        assert!(!findings.invalid, "{:?}", findings.diagnostics);
    }

    #[test]
    fn v0_2_duplicate_wind_device_identity_is_invalid() {
        let mut record: Value = serde_json::from_str(SYNTHETIC_EMISSIONS_V2).expect("v0.2 fixture JSON");
        let mut second = record["measurement_context"]["wind_device_states"][0].clone();
        second["configuration_revision"] = serde_json::json!("rev-b");
        record["measurement_context"]["wind_device_states"].as_array_mut().unwrap().push(second);
        let mut findings = Findings::default();
        check_operating_context(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn v0_2_commanded_and_achieved_waps_states_are_not_conflated() {
        let mut record: Value = serde_json::from_str(SYNTHETIC_EMISSIONS_V2).expect("v0.2 fixture JSON");
        record["measurement_context"]["wind_device_states"][0]["achieved_state"] =
            serde_json::json!("unavailable");
        let mut findings = Findings::default();
        check_operating_context(&record, &mut findings);
        assert!(findings.incomplete);
        assert!(!findings.invalid);
    }

    #[test]
    fn v0_2_unknown_achieved_state_is_incomplete_even_when_commanded_active() {
        let mut record: Value = serde_json::from_str(SYNTHETIC_EMISSIONS_V2).expect("v0.2 fixture JSON");
        record["measurement_context"]["wind_device_states"][0]["achieved_state"] =
            serde_json::json!("unknown");
        let mut findings = Findings::default();
        check_operating_context(&record, &mut findings);
        assert!(findings.incomplete);
    }

    #[test]
    fn v0_2_installed_waps_requires_wind_observations() {
        let mut record: Value = serde_json::from_str(SYNTHETIC_EMISSIONS_V2).expect("v0.2 fixture JSON");
        record["measurement_context"].as_object_mut().unwrap().remove("apparent_wind_speed_kn");
        let mut findings = Findings::default();
        check_operating_context(&record, &mut findings);
        assert!(findings.incomplete);
    }

    #[test]
    fn expected_gas_without_observation_or_reason_is_incomplete_not_zero() {
        let mut record: Value = serde_json::from_str(SYNTHETIC_EMISSIONS).expect("emissions fixture JSON");
        record["gas_observations"].as_array_mut().unwrap().pop();
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.incomplete);
    }

    #[test]
    fn out_of_tolerance_calibration_invalidates_measured_gas() {
        let mut record: Value = serde_json::from_str(SYNTHETIC_EMISSIONS).expect("emissions fixture JSON");
        record["gas_observations"][0]["evidence"]["calibration_status"] = serde_json::json!("out_of_tolerance");
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn engine_load_monitoring_requires_engine_load_calculated_evidence() {
        let mut record: Value = serde_json::from_str(SYNTHETIC_EMISSIONS).expect("emissions fixture JSON");
        record["measurement_method"] = serde_json::json!("engine_load_monitoring");
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn sample_counts_must_reconcile() {
        let mut record: Value = serde_json::from_str(SYNTHETIC_EMISSIONS).expect("emissions fixture JSON");
        record["measurement_window"]["missing_sample_count"] = serde_json::json!(1);
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn gas_observation_must_share_window_timebase() {
        let mut record: Value = serde_json::from_str(SYNTHETIC_EMISSIONS).expect("emissions fixture JSON");
        record["gas_observations"][0]["evidence"]["timebase_id"] = serde_json::json!("OTHER-CLOCK");
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }
}
