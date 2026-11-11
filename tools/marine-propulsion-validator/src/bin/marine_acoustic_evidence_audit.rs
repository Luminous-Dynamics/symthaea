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

#[derive(Debug, Serialize)]
struct AuditReport {
    auditor: &'static str,
    auditor_version: &'static str,
    record_id: Option<String>,
    standard_id: Option<String>,
    disposition: &'static str,
    claim_ceiling: &'static str,
    spectral_band_count: usize,
    raw_trace_present: bool,
    calibration_id: Option<String>,
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
            "ACOUSTIC_EVIDENCE_CONTRACT_VALID"
        }
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid JSON in {path}: {error}"))
}

fn parse_time(value: Option<&str>) -> Option<DateTime<Utc>> {
    value.and_then(|text| DateTime::parse_from_rfc3339(text).ok()).map(|v| v.with_timezone(&Utc))
}

fn check_window(window: &Value, path: &str, findings: &mut Findings) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
    let start = parse_time(window.get("start").and_then(Value::as_str));
    let end = parse_time(window.get("end").and_then(Value::as_str));
    match (start, end) {
        (Some(start), Some(end)) if start < end => Some((start, end)),
        (Some(_), Some(_)) => {
            findings.push(format!("{path}/end"), "INVALID", "window end must be strictly later than start");
            None
        }
        _ => {
            findings.push(path, "INVALID", "window timestamps must be valid RFC3339 date-times");
            None
        }
    }
}

fn validate_schema(schema: &Value, record: &Value, findings: &mut Findings) {
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
            findings.push("/", "INVALID", format!("cannot compile acoustic evidence schema: {error}"));
            return;
        }
    };
    for error in validator.iter_errors(record) {
        findings.push(
            error.instance_path().to_string(),
            "INVALID",
            format!("acoustic evidence schema violation: {error}"),
        );
    }
}

fn check_record(record: &Value, findings: &mut Findings) {
    if record.get("schema_version").and_then(Value::as_str) != Some("0.1.0") {
        findings.push("/schema_version", "INVALID", "unsupported acoustic evidence schema version");
    }
    if record.get("record_status").and_then(Value::as_str) == Some("synthetic") {
        findings.push(
            "/record_status",
            "INCOMPLETE",
            "synthetic fixture is software-test data and cannot establish ship acoustic performance",
        );
    }
    if record.get("record_status").and_then(Value::as_str) == Some("modelled") {
        findings.push(
            "/record_status",
            "INCOMPLETE",
            "modelled acoustic output is not a measured ISO ship-noise trial",
        );
    }

    let standard = record.get("standard_id").and_then(Value::as_str).unwrap_or("");
    let metric = record.get("metric_kind").and_then(Value::as_str).unwrap_or("");
    let expected_band = match standard {
        "ISO 17208-1:2016" | "ISO 17208-2:2019" => Some("one_third_octave"),
        "ISO 17208-3:2025" => Some("decidecade"),
        _ => {
            findings.push("/standard_id", "INVALID", "unsupported ISO 17208 standard part");
            None
        }
    };

    match standard {
        "ISO 17208-1:2016" if metric != "radiated_noise_level" => findings.push(
            "/metric_kind",
            "INVALID",
            "ISO 17208-1 radiated-noise measurements must be declared as radiated_noise_level",
        ),
        "ISO 17208-2:2019" if metric != "equivalent_monopole_source_level" => findings.push(
            "/metric_kind",
            "INVALID",
            "ISO 17208-2 output must be declared as equivalent_monopole_source_level",
        ),
        "ISO 17208-3:2025"
            if !matches!(metric, "radiated_noise_level" | "equivalent_monopole_source_level") =>
        {
            findings.push(
                "/metric_kind",
                "INVALID",
                "ISO 17208-3 metric must be radiated_noise_level or equivalent_monopole_source_level",
            )
        }
        _ => {}
    }

    if let Some(amendments) = record.get("standard_amendments").and_then(Value::as_array) {
        for (index, amendment) in amendments.iter().enumerate() {
            if amendment.as_str().is_none_or(|value| value.trim().is_empty()) {
                findings.push(
                    format!("/standard_amendments/{index}"),
                    "INVALID",
                    "standard amendment entries must be non-empty strings",
                );
            }
        }
    }

    let loa = record.pointer("/vessel/length_overall_m").and_then(Value::as_f64);
    let water_depth = record.pointer("/test_site/water_depth_m").and_then(Value::as_f64);
    if let (Some(loa), Some(depth)) = (loa, water_depth) {
        let deep_water_threshold = 150.0_f64.max(1.5 * loa);
        match standard {
            "ISO 17208-1:2016" | "ISO 17208-2:2019" if depth <= deep_water_threshold => {
                findings.push(
                    "/test_site/water_depth_m",
                    "INVALID",
                    format!(
                        "deep-water method requires water depth greater than max(150 m, 1.5 × LOA = {deep_water_threshold:.3} m)"
                    ),
                );
            }
            "ISO 17208-3:2025" if depth > deep_water_threshold => {
                findings.push(
                    "/test_site/water_depth_m",
                    "INCOMPLETE",
                    "declared shallow-water method is being used at a depth above the ISO 17208 deep-water boundary; review method selection",
                );
            }
            _ => {}
        }
    }

    if record
        .pointer("/test_site/water_depth_method")
        .and_then(Value::as_str)
        == Some("unknown")
    {
        findings.push(
            "/test_site/water_depth_method",
            "INCOMPLETE",
            "test-site depth is not surveyed or charted",
        );
    }

    let service_fingerprint = record.get("service_fingerprint").and_then(Value::as_str);
    if service_fingerprint.is_none_or(|value| {
        value.len() != 64 || !value.chars().all(|character| character.is_ascii_hexdigit())
    }) {
        findings.push(
            "/service_fingerprint",
            "INVALID",
            "service fingerprint must be a 64-character hexadecimal digest",
        );
    }

    let measurement_window = check_window(
        record.get("measurement_window").unwrap_or(&Value::Null),
        "/measurement_window",
        findings,
    );

    let calibration_start = parse_time(
        record.pointer("/hydrophone/calibration_valid_from").and_then(Value::as_str),
    );
    let calibration_end = parse_time(
        record.pointer("/hydrophone/calibration_valid_until").and_then(Value::as_str),
    );
    match (measurement_window.as_ref(), calibration_start, calibration_end) {
        (Some((start, end)), Some(valid_from), Some(valid_until)) => {
            if valid_from >= valid_until || valid_from > *start || valid_until < *end {
                findings.push(
                    "/hydrophone",
                    "INVALID",
                    "hydrophone calibration validity must cover the complete acoustic measurement window",
                );
            }
        }
        _ => findings.push(
            "/hydrophone",
            "INVALID",
            "hydrophone calibration validity dates must be valid and resolvable",
        ),
    }

    match record.pointer("/hydrophone/calibration_rights_status").and_then(Value::as_str) {
        Some("public") | Some("permissioned") | Some("restricted") => {}
        Some("unknown") | None => findings.push(
            "/hydrophone/calibration_rights_status",
            "INCOMPLETE",
            "calibration certificate rights/access status is unknown",
        ),
        _ => findings.push(
            "/hydrophone/calibration_rights_status",
            "INVALID",
            "unsupported calibration rights/access status",
        ),
    }

    let cal_min = record.pointer("/hydrophone/calibration_frequency_min_hz").and_then(Value::as_f64);
    let cal_max = record.pointer("/hydrophone/calibration_frequency_max_hz").and_then(Value::as_f64);
    let cal_uncertainty = record.pointer("/hydrophone/calibration_uncertainty_db").and_then(Value::as_f64);
    if matches!((cal_min, cal_max), (Some(min), Some(max)) if min >= max) {
        findings.push(
            "/hydrophone/calibration_frequency_max_hz",
            "INVALID",
            "hydrophone calibration maximum frequency must exceed its minimum",
        );
    }
    if cal_uncertainty.is_some_and(|value| value < 0.0) {
        findings.push(
            "/hydrophone/calibration_uncertainty_db",
            "INVALID",
            "calibration uncertainty cannot be negative",
        );
    }

    let sample = record.get("sample_window").unwrap_or(&Value::Null);
    let sample_rate = sample.get("sample_rate_hz").and_then(Value::as_f64);
    let expected = sample.get("expected_sample_count").and_then(Value::as_u64);
    let observed = sample.get("observed_sample_count").and_then(Value::as_u64);
    let missing = sample.get("missing_sample_count").and_then(Value::as_u64);
    let minimum_coverage = sample.get("minimum_observed_fraction").and_then(Value::as_f64);

    match (expected, observed, missing) {
        (Some(expected), Some(observed), Some(missing)) => {
            if expected == 0 || observed.saturating_add(missing) != expected {
                findings.push(
                    "/sample_window",
                    "INVALID",
                    "sample counts must satisfy expected > 0 and observed + missing = expected",
                );
            } else if let Some(minimum) = minimum_coverage {
                let coverage = observed as f64 / expected as f64;
                if coverage < minimum {
                    findings.push(
                        "/sample_window/observed_sample_count",
                        "INCOMPLETE",
                        format!(
                            "observed sample coverage {coverage:.6} is below the predeclared threshold {minimum:.6}"
                        ),
                    );
                }
            }
        }
        _ => findings.push("/sample_window", "INVALID", "expected, observed and missing sample counts must all be present"),
    }

    if sample.get("window_semantics").and_then(Value::as_str) != Some("start_inclusive_end_exclusive") {
        findings.push(
            "/sample_window/window_semantics",
            "INVALID",
            "sample windows must be start-inclusive/end-exclusive",
        );
    }

    if let (Some((start, end)), Some(rate), Some(expected)) =
        (measurement_window.as_ref(), sample_rate, expected)
    {
        let duration_ns = end.clone().signed_duration_since(start.clone()).num_nanoseconds();
        match duration_ns {
            Some(duration_ns) if duration_ns > 0 && rate.is_finite() && rate > 0.0 => {
                let expected_from_rate = duration_ns as f64 / 1_000_000_000.0 * rate;
                let difference = (expected as f64 - expected_from_rate).abs();
                if difference > 1.0 {
                    findings.push(
                        "/sample_window/expected_sample_count",
                        "INVALID",
                        format!(
                            "expected sample count {expected} differs from sample-rate/window estimate {expected_from_rate:.3} by more than one sample"
                        ),
                    );
                }
            }
            _ => findings.push(
                "/sample_window",
                "INVALID",
                "sample rate and acquisition window must define a finite positive expected sample count",
            ),
        }
    }

    if matches!(metric, "equivalent_monopole_source_level")
        && record.pointer("/measurement_geometry/aspect").and_then(Value::as_str)
            != Some("broadside")
    {
        findings.push(
            "/measurement_geometry/aspect",
            "INVALID",
            "equivalent monopole source-level comparison requires broadside aspect under the declared ISO 17208-2 procedure",
        );
    }
    if metric == "equivalent_monopole_source_level"
        && record.pointer("/measurement_geometry/source_depth_m").and_then(Value::as_f64).is_none()
    {
        findings.push(
            "/measurement_geometry/source_depth_m",
            "INVALID",
            "equivalent monopole source level requires a declared nominal source depth",
        );
    }

    let bands = record.get("spectral_bands").and_then(Value::as_array);
    let Some(bands) = bands else {
        findings.push("/spectral_bands", "INVALID", "spectral bands are missing");
        return;
    };
    if bands.is_empty() {
        findings.push("/spectral_bands", "INVALID", "at least one spectral band is required");
    }

    let mut centres = BTreeSet::new();
    let mut previous_upper = None;
    for (index, band) in bands.iter().enumerate() {
        let path = format!("/spectral_bands/{index}");
        let band_type = band.get("band_type").and_then(Value::as_str);
        if Some(band_type.unwrap_or("")) != expected_band {
            findings.push(
                format!("{path}/band_type"),
                "INVALID",
                format!(
                    "band type must match standard profile {}",
                    expected_band.unwrap_or("<unsupported>")
                ),
            );
        }

        let lower = band.get("lower_frequency_hz").and_then(Value::as_f64);
        let centre = band.get("centre_frequency_hz").and_then(Value::as_f64);
        let upper = band.get("upper_frequency_hz").and_then(Value::as_f64);
        match (lower, centre, upper) {
            (Some(lower), Some(centre), Some(upper))
                if lower > 0.0 && lower < centre && centre < upper =>
            {
                if !centres.insert(centre.to_bits()) {
                    findings.push(
                        format!("{path}/centre_frequency_hz"),
                        "INVALID",
                        "duplicate spectral-band centre frequency",
                    );
                }
                if previous_upper.is_some_and(|previous| lower < previous) {
                    findings.push(
                        format!("{path}/lower_frequency_hz"),
                        "INVALID",
                        "spectral bands must be sorted and non-overlapping",
                    );
                }
                previous_upper = Some(upper);

                if let (Some(min), Some(max)) = (cal_min, cal_max) {
                    if lower < min || upper > max {
                        findings.push(
                            &path,
                            "INVALID",
                            "hydrophone calibration frequency range does not cover this complete spectral band",
                        );
                    }
                } else {
                    findings.push(
                        "/hydrophone",
                        "INVALID",
                        "calibration frequency range is unavailable",
                    );
                }
            }
            _ => findings.push(
                &path,
                "INVALID",
                "spectral band must satisfy 0 < lower < centre < upper frequency",
            ),
        }

        if band.get("expanded_uncertainty_db")
            .and_then(Value::as_f64)
            .is_none_or(|value| value < 0.0)
        {
            findings.push(
                format!("{path}/expanded_uncertainty_db"),
                "INVALID",
                "expanded acoustic uncertainty must be known and non-negative",
            );
        }
        if band.get("coverage_factor")
            .and_then(Value::as_f64)
            .is_none_or(|value| value <= 0.0)
        {
            findings.push(
                format!("{path}/coverage_factor"),
                "INVALID",
                "coverage factor must be known and strictly positive",
            );
        }
    }

    if let Some(ambient) = record.get("ambient_reference") {
        check_window(
            ambient.get("measurement_window").unwrap_or(&Value::Null),
            "/ambient_reference/measurement_window",
            findings,
        );
        if ambient.get("raw_trace_sha256").and_then(Value::as_str).is_none() {
            findings.push(
                "/ambient_reference/raw_trace_sha256",
                "INVALID",
                "ambient reference requires its own raw-trace digest",
            );
        }
    } else {
        findings.push(
            "/ambient_reference",
            "INCOMPLETE",
            "ambient/background sound reference is absent; do not infer background sound is zero",
        );
    }

    if record.pointer("/operating_condition/wind_assist_installation").and_then(Value::as_str)
        == Some("installed")
    {
        let states = record.pointer("/operating_condition/wind_device_states").and_then(Value::as_array);
        if states.is_none_or(|values| values.is_empty()) {
            findings.push(
                "/operating_condition/wind_device_states",
                "INCOMPLETE",
                "wind assistance is installed but no achieved device state is recorded",
            );
        }
        if record.pointer("/operating_condition/wind_speed_kn").and_then(Value::as_f64).is_none()
            || record.pointer("/operating_condition/wind_from_bow_deg").and_then(Value::as_f64).is_none()
        {
            findings.push(
                "/operating_condition",
                "INCOMPLETE",
                "installed WAPS acoustic trial requires measured wind speed and direction",
            );
        }
    }
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: marine-acoustic-evidence-audit <schema.json> <record.json>".to_owned());
    }
    let schema = read_json(&args[1])?;
    let record = read_json(&args[2])?;
    let mut findings = Findings::default();

    validate_schema(&schema, &record, &mut findings);
    check_record(&record, &mut findings);

    let report = AuditReport {
        auditor: "marine-acoustic-evidence-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        record_id: record.get("record_id").and_then(Value::as_str).map(str::to_owned),
        standard_id: record.get("standard_id").and_then(Value::as_str).map(str::to_owned),
        disposition: findings.disposition(),
        claim_ceiling: "Schema and acoustic measurement-evidence contract checks only. This report does not certify ISO measurement conformity, hydrophone calibration, source-level validity, ecological impact, quiet-ship status, seaworthiness or regulatory compliance.",
        spectral_band_count: record.get("spectral_bands").and_then(Value::as_array).map_or(0, Vec::len),
        raw_trace_present: record.get("raw_trace_sha256").and_then(Value::as_str).is_some(),
        calibration_id: record.pointer("/hydrophone/calibration_record_id").and_then(Value::as_str).map(str::to_owned),
        diagnostics: findings.diagnostics,
    };

    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(if report.disposition == "ACOUSTIC_EVIDENCE_CONTRACT_VALID" { 0 } else { 2 })
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

    const ACOUSTIC_SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-acoustic-evidence-v0.1.0.schema.json");
    const ACOUSTIC_FIXTURE: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-marine-acoustic-evidence-v0.1.0.json");

    fn fixture() -> Value {
        serde_json::from_str(ACOUSTIC_FIXTURE).expect("acoustic fixture JSON")
    }

    #[test]
    fn committed_acoustic_schema_and_fixture_are_structurally_valid() {
        let schema: Value = serde_json::from_str(ACOUSTIC_SCHEMA).expect("acoustic schema JSON");
        let record = fixture();
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("acoustic schema compiles");
        assert!(validator.is_valid(&record));
    }

    #[test]
    fn synthetic_acoustic_fixture_is_never_physical_qualification() {
        let mut findings = Findings::default();
        check_record(&fixture(), &mut findings);
        assert!(findings.incomplete);
        assert!(!findings.invalid);
    }

    #[test]
    fn deep_water_method_rejects_insufficient_test_depth() {
        let mut record = fixture();
        record["test_site"]["water_depth_m"] = serde_json::json!(240.0);
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn shallow_water_method_requires_decidecade_bands() {
        let mut record = fixture();
        record["standard_id"] = serde_json::json!("ISO 17208-3:2025");
        record["test_site"]["water_depth_m"] = serde_json::json!(100.0);
        record["spectral_bands"][0]["band_type"] = serde_json::json!("one_third_octave");
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn calibration_must_cover_complete_measurement_window() {
        let mut record = fixture();
        record["hydrophone"]["calibration_valid_until"] =
            serde_json::json!("2026-10-15T10:00:30Z");
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn calibration_must_cover_each_reported_frequency_band() {
        let mut record = fixture();
        record["hydrophone"]["calibration_frequency_max_hz"] = serde_json::json!(400.0);
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn invalid_sample_count_arithmetic_is_rejected() {
        let mut record = fixture();
        record["sample_window"]["observed_sample_count"] = serde_json::json!(2_879_999);
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn source_level_requires_broadside_and_nominal_source_depth() {
        let mut record = fixture();
        record["standard_id"] = serde_json::json!("ISO 17208-2:2019");
        record["metric_kind"] = serde_json::json!("equivalent_monopole_source_level");
        record["measurement_geometry"]["aspect"] = serde_json::json!("other");
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn duplicate_or_overlapping_bands_are_rejected() {
        let mut record = fixture();
        let first = record["spectral_bands"][0].clone();
        record["spectral_bands"][1] = first;
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn unknown_calibration_rights_and_missing_ambient_reference_remain_incomplete() {
        let mut record = fixture();
        record["hydrophone"]["calibration_rights_status"] = serde_json::json!("unknown");
        record.as_object_mut().expect("record object").remove("ambient_reference");
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.incomplete);
        assert!(!findings.invalid);
    }
}
