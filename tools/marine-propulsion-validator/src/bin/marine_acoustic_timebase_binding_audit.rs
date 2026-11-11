use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;

const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";
const REQUIRED_PROPULSION_CHANNELS: [(&str, &str); 5] = [
    ("shaft_torque", "Nm"),
    ("shaft_rotational_speed", "rpm"),
    ("speed_through_water", "kn"),
    ("vessel_position", "m"),
    ("shaft_angle", "deg"),
];

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
            "ACOUSTIC_TRIAL_TIMEBASE_BINDING_VALID"
        }
    }
}

#[derive(Debug, Serialize)]
struct AuditReport {
    auditor: &'static str,
    auditor_version: &'static str,
    acoustic_record_id: Option<String>,
    trial_case_id: Option<String>,
    service_fingerprint: Option<String>,
    acoustic_raw_trace_sha256: Option<String>,
    acoustic_timebase_id: Option<String>,
    common_reference_clock_id: Option<String>,
    disposition: &'static str,
    binding_valid: bool,
    synchronized_channel_count: usize,
    required_propulsion_channels: Vec<String>,
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

fn time_to_seconds(unit: &str) -> Option<f64> {
    match unit {
        "s" | "sec" => Some(1.0),
        "ms" => Some(1.0e-3),
        "us" => Some(1.0e-6),
        "min" => Some(60.0),
        "h" => Some(3600.0),
        "day" => Some(86400.0),
        _ => None,
    }
}

fn frequency_to_hz(unit: &str) -> Option<f64> {
    match unit {
        "Hz" | "1/s" => Some(1.0),
        _ => None,
    }
}

fn validate_schema(schema: &Value, instance: &Value, label: &str, findings: &mut Findings) {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        findings.push(format!("/{label}/$schema"), "INVALID", "only JSON Schema Draft 2020-12 is supported");
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
    for error in validator.iter_errors(instance) {
        findings.push(
            format!("/{label}{}", error.instance_path()),
            "INVALID",
            format!("{label} record schema violation: {error}"),
        );
    }
}

fn valid_sha256(value: Option<&str>) -> bool {
    value.is_some_and(|digest| {
        digest.len() == 64 && digest.chars().all(|character| character.is_ascii_hexdigit())
    })
}

fn quantity_seconds(quantity: &Value) -> Option<f64> {
    let value = quantity.get("value").and_then(Value::as_f64)?;
    let unit = quantity.get("unit").and_then(Value::as_str)?;
    let converted = value * time_to_seconds(unit)?;
    converted.is_finite().then_some(converted)
}

fn uncertainty_is_known(quantity: &Value) -> bool {
    matches!(
        quantity.pointer("/uncertainty/kind").and_then(Value::as_str),
        Some("standard" | "interval")
    )
}

fn check_quantity_uncertainty(quantity: &Value, path: &str, findings: &mut Findings) {
    let numeric_value = quantity.get("value").and_then(Value::as_f64);
    if numeric_value.is_none() || quantity.get("unit").and_then(Value::as_str).is_none() {
        findings.push(path, "INVALID", "quantity must carry a numeric value and explicit unit");
        return;
    }
    if numeric_value.is_some_and(|value| !value.is_finite() || value < 0.0) {
        findings.push(path, "INVALID", "uncertainty bounds and magnitudes must be finite and non-negative");
    }
    if !uncertainty_is_known(quantity) {
        findings.push(
            format!("{path}/uncertainty"),
            "INCOMPLETE",
            "measurement uncertainty is unknown; a numeric placeholder cannot qualify synchronization",
        );
    }
}

fn window_covers(channel: &Value, start: &DateTime<Utc>, end: &DateTime<Utc>) -> bool {
    match (
        parse_time(channel.get("sample_window_start").and_then(Value::as_str)),
        parse_time(channel.get("sample_window_end").and_then(Value::as_str)),
    ) {
        (Some(channel_start), Some(channel_end)) => channel_start <= start.clone() && channel_end >= end.clone(),
        _ => false,
    }
}

fn check_channel_window_and_samples(
    channel: &Value,
    start: &DateTime<Utc>,
    end: &DateTime<Utc>,
    policy: &Value,
    path: &str,
    findings: &mut Findings,
) {
    if !window_covers(channel, start, end) {
        findings.push(
            format!("{path}/sample_window"),
            "INCOMPARABLE",
            "channel sample window does not cover the complete acoustic measurement window",
        );
    }

    let expected = channel.get("expected_sample_count").and_then(Value::as_u64);
    let observed = channel.get("observed_sample_count").and_then(Value::as_u64);
    let missing = channel.get("missing_sample_count").and_then(Value::as_u64);
    match (expected, observed, missing) {
        (Some(expected), Some(observed), Some(missing)) if expected > 0 => {
            if observed.saturating_add(missing) != expected {
                findings.push(
                    format!("{path}/sample_counts"),
                    "INVALID",
                    "observed plus missing samples must equal expected sample count",
                );
            }
            let minimum = policy
                .get("minimum_observed_fraction")
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            if observed as f64 / expected as f64 < minimum {
                findings.push(
                    format!("{path}/observed_sample_count"),
                    "INCOMPLETE",
                    "channel sample coverage is below the predeclared trial policy",
                );
            }
        }
        _ => findings.push(
            format!("{path}/sample_counts"),
            "INVALID",
            "expected, observed and missing sample counts must be present with expected > 0",
        ),
    }

    let sample_rate = channel.pointer("/sample_rate_hz/value").and_then(Value::as_f64);
    let sample_rate_unit = channel.pointer("/sample_rate_hz/unit").and_then(Value::as_str);
    match (sample_rate, sample_rate_unit.and_then(frequency_to_hz)) {
        (Some(rate), Some(scale)) if rate.is_finite() && rate > 0.0 => {
            let expected_count = (end.signed_duration_since(start.clone()).num_nanoseconds().unwrap_or(0) as f64)
                / 1_000_000_000.0
                * rate
                * scale;
            let tolerance = policy
                .get("rate_count_tolerance_samples")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            if let Some(declared) = expected {
                if (declared as f64 - expected_count).abs() > tolerance {
                    findings.push(
                        format!("{path}/expected_sample_count"),
                        "INVALID",
                        format!(
                            "expected count {declared} disagrees with sample rate/window estimate {expected_count:.3} beyond tolerance {tolerance}"
                        ),
                    );
                }
            }
        }
        _ => findings.push(
            format!("{path}/sample_rate_hz"),
            "INVALID",
            "sample rate must be finite, positive and expressed in Hz or 1/s",
        ),
    }
}

fn check_channel(
    channel: &Value,
    trial: &Value,
    start: &DateTime<Utc>,
    end: &DateTime<Utc>,
    expected_certificate: Option<&str>,
    path: &str,
    findings: &mut Findings,
) -> Option<String> {
    let calibrations = trial.get("calibration_records").and_then(Value::as_array);
    let calibration_id = channel.get("calibration_record_id").and_then(Value::as_str);
    let calibration_matches: Vec<&Value> = calibrations
        .into_iter()
        .flatten()
        .filter(|calibration| calibration.get("calibration_record_id").and_then(Value::as_str) == calibration_id)
        .collect();
    match channel.pointer("/provenance/class").and_then(Value::as_str) {
        Some("measured") => {}
        Some("synthetic" | "assumed" | "vendor_nominal" | "published" | "derived") | None => findings.push(
            format!("{path}/provenance/class"),
            "INCOMPLETE",
            "propulsion/acoustic channel is not a directly measured trial channel",
        ),
        _ => findings.push(
            format!("{path}/provenance/class"),
            "INVALID",
            "unsupported channel provenance class",
        ),
    }
    if calibration_matches.len() != 1 {
        findings.push(
            format!("{path}/calibration_record_id"),
            "INVALID",
            "channel calibration reference must resolve to exactly one registry record",
        );
    } else {
        let calibration = calibration_matches[0];
        if calibration.get("instrument_identity") != channel.get("instrument_identity")
            || calibration.get("physical_quantity") != channel.get("physical_quantity")
            || calibration.get("unit") != channel.get("unit")
        {
            findings.push(
                format!("{path}/calibration_record_id"),
                "INVALID",
                "calibration instrument identity, physical quantity and unit must exactly match the channel",
            );
        }
        if let Some(expected_certificate) = expected_certificate {
            if calibration.get("certificate_sha256").and_then(Value::as_str) != Some(expected_certificate) {
                findings.push(
                    format!("{path}/calibration_record_id"),
                    "INVALID",
                    "hydrophone calibration certificate digest differs from the acoustic evidence record",
                );
            }
        }
        let valid_from = parse_time(calibration.get("valid_from").and_then(Value::as_str));
        let valid_until = parse_time(calibration.get("valid_until").and_then(Value::as_str));
        match (valid_from, valid_until) {
            (Some(from), Some(until)) if from <= start.clone() && until >= end.clone() && from < until => {}
            _ => findings.push(
                format!("{path}/calibration_record_id"),
                "INVALID",
                "calibration validity must cover the complete acoustic measurement window",
            ),
        }
        match calibration.get("disposition").and_then(Value::as_str) {
            Some("in_tolerance") => {}
            Some("out_of_tolerance") => findings.push(
                format!("{path}/calibration_record_id"),
                "INVALID",
                "referenced calibration is out of tolerance",
            ),
            Some("stale" | "unknown") | None => findings.push(
                format!("{path}/calibration_record_id"),
                "INCOMPLETE",
                "referenced calibration is stale or unknown",
            ),
            _ => findings.push(
                format!("{path}/calibration_record_id"),
                "INVALID",
                "unsupported calibration disposition",
            ),
        }
        match calibration.get("rights_status").and_then(Value::as_str) {
            Some("public" | "permissioned") => {}
            Some("restricted" | "unknown" | "not_applicable") | None => findings.push(
                format!("{path}/calibration_record_id/rights_status"),
                "INCOMPLETE",
                "calibration evidence rights/access do not permit independent review",
            ),
            _ => findings.push(
                format!("{path}/calibration_record_id/rights_status"),
                "INVALID",
                "unsupported calibration rights status",
            ),
        }
        if !uncertainty_is_known(calibration.get("uncertainty").unwrap_or(&Value::Null)) {
            findings.push(
                format!("{path}/calibration_record_id/uncertainty"),
                "INCOMPLETE",
                "calibration uncertainty is unknown",
            );
        }
    }

    let timebases = trial.get("timebases").and_then(Value::as_array);
    let timebase_id = channel.get("timebase_id").and_then(Value::as_str);
    let timebase_matches: Vec<&Value> = timebases
        .into_iter()
        .flatten()
        .filter(|timebase| timebase.get("timebase_id").and_then(Value::as_str) == timebase_id)
        .collect();
    if timebase_matches.len() != 1 {
        findings.push(
            format!("{path}/timebase_id"),
            "INVALID",
            "channel timebase reference must resolve to exactly one registry record",
        );
        return None;
    }

    let timebase = timebase_matches[0];
    let reference_clock = timebase.get("reference_clock_id").and_then(Value::as_str);
    if reference_clock.is_none_or(str::is_empty) {
        findings.push(
            format!("{path}/timebase_id/reference_clock_id"),
            "INVALID",
            "timebase must declare a non-empty reference clock identity",
        );
    }
    let valid_from = parse_time(timebase.get("valid_from").and_then(Value::as_str));
    let valid_until = parse_time(timebase.get("valid_until").and_then(Value::as_str));
    match (valid_from, valid_until) {
        (Some(from), Some(until)) if from <= *start && until >= *end && from < until => {}
        _ => findings.push(
            format!("{path}/timebase_id"),
            "INVALID",
            "timebase validity must cover the complete acoustic measurement window",
        ),
    }

    if channel.get("sample_rate_basis").and_then(Value::as_str) != Some("measured") {
        findings.push(
            format!("{path}/sample_rate_basis"),
            "INCOMPLETE",
            "a nominal sampling-rate declaration is not independent evidence of the acquisition clock rate",
        );
    }
    let sync = timebase.get("synchronization_uncertainty").unwrap_or(&Value::Null);
    let drift = timebase.get("drift_uncertainty_ppm").unwrap_or(&Value::Null);
    let alignment = channel.get("time_alignment_uncertainty").unwrap_or(&Value::Null);
    check_quantity_uncertainty(sync, &format!("{path}/timebase/synchronization_uncertainty"), findings);
    check_quantity_uncertainty(drift, &format!("{path}/timebase/drift_uncertainty_ppm"), findings);
    check_quantity_uncertainty(alignment, &format!("{path}/time_alignment_uncertainty"), findings);
    if drift.get("unit").and_then(Value::as_str) != Some("ppm") {
        findings.push(
            format!("{path}/timebase/drift_uncertainty_ppm/unit"),
            "INVALID",
            "residual timebase drift uncertainty must use ppm",
        );
    }
    if let (Some(sync_seconds), Some(alignment_seconds), Some(drift_ppm), Some(epoch)) = (
        quantity_seconds(sync),
        quantity_seconds(alignment),
        drift.get("value").and_then(Value::as_f64),
        parse_time(timebase.get("reference_epoch").and_then(Value::as_str)),
    ) {
        if drift_ppm < 0.0 {
            findings.push(
                format!("{path}/timebase/drift_uncertainty_ppm/value"),
                "INVALID",
                "residual drift uncertainty cannot be negative",
            );
        }
        let elapsed_start = start
            .signed_duration_since(epoch.clone())
            .num_nanoseconds()
            .map(|value| (value as f64).abs() / 1.0e9);
        let elapsed_end = end
            .signed_duration_since(epoch)
            .num_nanoseconds()
            .map(|value| (value as f64).abs() / 1.0e9);
        let elapsed = match (elapsed_start, elapsed_end) {
            (Some(a), Some(b)) => a.max(b),
            _ => {
                findings.push(
                    format!("{path}/timebase/reference_epoch"),
                    "INCOMPLETE",
                    "reference epoch is too distant for supported nanosecond duration arithmetic",
                );
                0.0
            }
        };
        let drift_seconds = elapsed * drift_ppm.abs() * 1.0e-6;
        let required_bound = sync_seconds + drift_seconds;
        if alignment_seconds + 1.0e-12 < required_bound {
            findings.push(
                format!("{path}/time_alignment_uncertainty"),
                "INVALID",
                format!(
                    "declared channel alignment bound {alignment_seconds:.9}s is below clock plus accumulated drift bound {required_bound:.9}s"
                ),
            );
        }
    } else {
        findings.push(
            format!("{path}/time_alignment_uncertainty"),
            "INCOMPLETE",
            "clock uncertainty, drift, reference epoch or unit conversion is unresolved",
        );
    }

    match timebase.get("rights_status").and_then(Value::as_str) {
        Some("public" | "permissioned") => {}
        Some("restricted" | "unknown" | "not_applicable") | None => findings.push(
            format!("{path}/timebase/rights_status"),
            "INCOMPLETE",
            "timebase evidence rights/access do not permit independent review",
        ),
        _ => findings.push(
            format!("{path}/timebase/rights_status"),
            "INVALID",
            "unsupported timebase rights status",
        ),
    }

    check_channel_window_and_samples(channel, start, end, &trial["sample_completeness_policy"], path, findings);
    reference_clock.map(str::to_owned)
}

fn check_binding(acoustic: &Value, case: &Value, findings: &mut Findings) -> (bool, usize, Option<String>, Option<String>) {
    if acoustic.get("record_status").and_then(Value::as_str) != Some("measured") {
        findings.push(
            "/acoustic/record_status",
            "INCOMPLETE",
            "acoustic record is not measured; synthetic/modelled data cannot qualify a trial binding",
        );
    }
    if case.get("record_status").and_then(Value::as_str) != Some("measured") {
        findings.push(
            "/trial_case/record_status",
            "INCOMPLETE",
            "trial case is not measured; synthetic/modelled data cannot qualify a trial binding",
        );
    }

    let trial = case.get("trial_evidence").unwrap_or(&Value::Null);
    if trial.is_null() {
        findings.push("/trial_case/trial_evidence", "INCOMPLETE", "trial evidence is absent");
        return (false, 0, None, None);
    }
    if !matches!(
        trial.get("validation_stage").and_then(Value::as_str),
        Some("sea_trial" | "in_service")
    ) {
        findings.push(
            "/trial_case/trial_evidence/validation_stage",
            "INCOMPLETE",
            "only an instrumented sea-trial or in-service record can support a full-scale acoustic/propulsion binding",
        );
    }

    let acoustic_service = acoustic.get("service_fingerprint").and_then(Value::as_str);
    let trial_service = trial.get("service_fingerprint").and_then(Value::as_str);
    let case_service = case.pointer("/comparison/service_fingerprint").and_then(Value::as_str);
    if acoustic_service.is_none() || acoustic_service != trial_service || trial_service != case_service {
        findings.push(
            "/trial_case/trial_evidence/service_fingerprint",
            "INCOMPARABLE",
            "acoustic record, trial evidence and case comparison must share the exact service fingerprint",
        );
    }

    let start = parse_time(acoustic.pointer("/measurement_window/start").and_then(Value::as_str));
    let end = parse_time(acoustic.pointer("/measurement_window/end").and_then(Value::as_str));
    let (Some(start), Some(end)) = (start, end) else {
        findings.push("/acoustic/measurement_window", "INVALID", "acoustic measurement window must contain valid RFC3339 timestamps");
        return (false, 0, None, None);
    };
    if start >= end {
        findings.push("/acoustic/measurement_window", "INVALID", "acoustic measurement window end must follow start");
        return (false, 0, None, None);
    }

    let channels = trial.get("channels").and_then(Value::as_array);
    let Some(channels) = channels else {
        findings.push("/trial_case/trial_evidence/channels", "INVALID", "trial channels are missing");
        return (false, 0, None, None);
    };
    let hydrophone = acoustic.get("hydrophone").unwrap_or(&Value::Null);
    let hydrophone_identity = hydrophone.get("instrument_identity");
    let hydrophone_calibration_id = hydrophone.get("calibration_record_id").and_then(Value::as_str);
    let acoustic_trace = acoustic.get("raw_trace_sha256").and_then(Value::as_str);
    let acoustic_sample = acoustic.get("sample_window").unwrap_or(&Value::Null);
    let acoustic_rate = acoustic_sample.get("sample_rate_hz").and_then(Value::as_f64);
    let acoustic_expected = acoustic_sample.get("expected_sample_count").and_then(Value::as_u64);
    let acoustic_observed = acoustic_sample.get("observed_sample_count").and_then(Value::as_u64);
    let acoustic_missing = acoustic_sample.get("missing_sample_count").and_then(Value::as_u64);

    let hydrophone_matches: Vec<&Value> = channels
        .iter()
        .filter(|channel| {
            channel.get("physical_quantity").and_then(Value::as_str) == Some("underwater_sound_pressure")
                && channel.get("instrument_identity") == hydrophone_identity
                && channel.get("calibration_record_id").and_then(Value::as_str) == hydrophone_calibration_id
                && channel.get("raw_trace_sha256").and_then(Value::as_str) == acoustic_trace
        })
        .collect();
    if hydrophone_matches.len() != 1 {
        findings.push(
            "/trial_case/trial_evidence/channels",
            "INVALID",
            "hydrophone must bind to exactly one trial channel by instrument identity, calibration ID and raw-trace digest",
        );
        return (false, 0, None, None);
    }
    let hydrophone_channel = hydrophone_matches[0];
    let hydrophone_channel_id = hydrophone_channel.get("channel_id").and_then(Value::as_str).unwrap_or("<unknown>");
    let hydrophone_sample_rate = hydrophone_channel.pointer("/sample_rate_hz/value").and_then(Value::as_f64);
    let hydrophone_sample_rate_unit = hydrophone_channel.pointer("/sample_rate_hz/unit").and_then(Value::as_str);
    if !matches!((acoustic_rate, hydrophone_sample_rate, hydrophone_sample_rate_unit.and_then(frequency_to_hz)),
        (Some(acoustic_rate), Some(channel_rate), Some(scale)) if (acoustic_rate - channel_rate * scale).abs() <= 1.0e-9)
    {
        findings.push(
            "/trial_case/trial_evidence/channels/sample_rate_hz",
            "INVALID",
            "hydrophone channel sample rate must equal the acoustic record sample rate after reviewed unit conversion",
        );
    }
    if hydrophone_channel.get("sample_window_start").and_then(Value::as_str)
        != acoustic.pointer("/measurement_window/start").and_then(Value::as_str)
        || hydrophone_channel.get("sample_window_end").and_then(Value::as_str)
            != acoustic.pointer("/measurement_window/end").and_then(Value::as_str)
    {
        findings.push(
            "/trial_case/trial_evidence/channels/sample_window",
            "INCOMPARABLE",
            "hydrophone trial channel and acoustic record must have identical half-open measurement windows",
        );
    }
    if hydrophone_channel.get("expected_sample_count").and_then(Value::as_u64) != acoustic_expected
        || hydrophone_channel.get("observed_sample_count").and_then(Value::as_u64) != acoustic_observed
        || hydrophone_channel.get("missing_sample_count").and_then(Value::as_u64) != acoustic_missing
    {
        findings.push(
            "/trial_case/trial_evidence/channels/sample_counts",
            "INVALID",
            "hydrophone channel sample counts must exactly match the acoustic record",
        );
    }

    let calibration_certificate = hydrophone.get("calibration_certificate_sha256").and_then(Value::as_str);
    let hydrophone_clock = check_channel(
        hydrophone_channel,
        trial,
        &start,
        &end,
        calibration_certificate,
        &format!("/channels/{hydrophone_channel_id}"),
        findings,
    );

    let mut matched_quantities = BTreeSet::new();
    let mut reference_clock = hydrophone_clock.clone();
    let hydrophone_timebase_id = hydrophone_channel.get("timebase_id").and_then(Value::as_str);
    let mut synchronized_count = 1usize;

    for (physical_quantity, expected_unit) in REQUIRED_PROPULSION_CHANNELS {
        let matches: Vec<&Value> = channels
            .iter()
            .filter(|channel| channel.get("physical_quantity").and_then(Value::as_str) == Some(physical_quantity))
            .collect();
        if matches.len() != 1 {
            findings.push(
                format!("/trial_case/trial_evidence/channels/{physical_quantity}"),
                "INCOMPLETE",
                format!("exactly one {physical_quantity} channel is required; found {}", matches.len()),
            );
            continue;
        }
        let channel = matches[0];
        let channel_id = channel.get("channel_id").and_then(Value::as_str).unwrap_or("<unknown>");
        if channel.get("timebase_id").and_then(Value::as_str) != hydrophone_timebase_id {
            findings.push(
                format!("/channels/{channel_id}/timebase_id"),
                "INCOMPARABLE",
                "initial implementation requires the hydrophone and propulsion channels to resolve through the same versioned timebase record",
            );
        }
        if channel.get("unit").and_then(Value::as_str) != Some(expected_unit) {
            findings.push(
                format!("/channels/{channel_id}/unit"),
                "INVALID",
                format!("{physical_quantity} channel must use reviewed unit {expected_unit}"),
            );
        }
        if !matched_quantities.insert(physical_quantity) {
            findings.push(
                format!("/channels/{channel_id}/physical_quantity"),
                "INVALID",
                "duplicate required propulsion channel",
            );
        }
        let channel_clock = check_channel(
            channel,
            trial,
            &start,
            &end,
            None,
            &format!("/channels/{channel_id}"),
            findings,
        );
        match (reference_clock.as_deref(), channel_clock.as_deref()) {
            (Some(expected), Some(actual)) if expected != actual => findings.push(
                format!("/channels/{channel_id}/timebase_id"),
                "INCOMPARABLE",
                "propulsion channel does not share the hydrophone's reference clock identity",
            ),
            (None, _) | (_, None) => findings.push(
                format!("/channels/{channel_id}/timebase_id"),
                "INCOMPLETE",
                "channel reference clock could not be resolved",
            ),
            _ => {}
        }
        if channel_clock.is_some() {
            synchronized_count += 1;
        }
    }

    let binding_valid = !findings.invalid
        && !findings.incomparable
        && matched_quantities.len() == REQUIRED_PROPULSION_CHANNELS.len()
        && hydrophone_clock.is_some()
        && reference_clock.is_some();

    let common_clock = reference_clock.take();
    let timebase_id = hydrophone_channel.get("timebase_id").and_then(Value::as_str).map(str::to_owned);
    (binding_valid, synchronized_count, common_clock, timebase_id)
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        return Err("usage: marine-acoustic-timebase-binding-audit <acoustic.schema.json> <acoustic.json> <propulsion-case.schema.json> <propulsion-case.json>".to_owned());
    }
    let acoustic_schema = read_json(&args[1])?;
    let acoustic = read_json(&args[2])?;
    let case_schema = read_json(&args[3])?;
    let case = read_json(&args[4])?;
    let mut findings = Findings::default();
    validate_schema(&acoustic_schema, &acoustic, "acoustic", &mut findings);
    validate_schema(&case_schema, &case, "trial_case", &mut findings);
    let (binding_valid, synchronized_count, reference_clock, timebase_id) =
        check_binding(&acoustic, &case, &mut findings);

    let report = AuditReport {
        auditor: "marine-acoustic-timebase-binding-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        acoustic_record_id: acoustic.get("record_id").and_then(Value::as_str).map(str::to_owned),
        trial_case_id: case.get("case_id").and_then(Value::as_str).map(str::to_owned),
        service_fingerprint: acoustic.get("service_fingerprint").and_then(Value::as_str).map(str::to_owned),
        acoustic_raw_trace_sha256: acoustic.get("raw_trace_sha256").and_then(Value::as_str).map(str::to_owned),
        acoustic_timebase_id: timebase_id,
        common_reference_clock_id: reference_clock,
        disposition: findings.disposition(),
        binding_valid,
        synchronized_channel_count: synchronized_count,
        required_propulsion_channels: REQUIRED_PROPULSION_CHANNELS
            .iter()
            .map(|(quantity, _)| (*quantity).to_owned())
            .collect(),
        claim_ceiling: "Exact hydrophone/trial trace, calibration, timebase and synchronized propulsion-channel linkage only. This report does not authenticate sensors or certificates, prove clock integrity, qualify ISO conformity, reproduce cavitation physics, estimate causal effects, establish acoustic reduction or prove energy savings.",
        diagnostics: findings.diagnostics,
    };
    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(if report.binding_valid && report.disposition == "ACOUSTIC_TRIAL_TIMEBASE_BINDING_VALID" { 0 } else { 2 })
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

    const ACOUSTIC_SCHEMA: &str = include_str!("../../../../docs/research/marine/marine-acoustic-evidence-v0.1.0.schema.json");
    const ACOUSTIC_FIXTURE: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-marine-acoustic-timebase-v0.1.0.json");
    const CASE_SCHEMA: &str = include_str!("../../../../docs/research/marine/marine-propulsion-case-v0.4.0.schema.json");
    const CASE_FIXTURE: &str = include_str!("../../../../docs/research/marine/fixtures/synthetic-marine-environmental-trial-case-v0.4.0.json");

    fn records() -> (Value, Value) {
        (
            serde_json::from_str(ACOUSTIC_FIXTURE).expect("acoustic fixture JSON"),
            serde_json::from_str(CASE_FIXTURE).expect("trial case fixture JSON"),
        )
    }

    #[test]
    fn committed_acoustic_and_environmental_case_fixtures_match_their_schemas() {
        let (acoustic, case) = records();
        for (schema_text, record) in [(ACOUSTIC_SCHEMA, &acoustic), (CASE_SCHEMA, &case)] {
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
    fn synthetic_binding_resolves_structure_but_never_qualifies() {
        let (acoustic, case) = records();
        let mut findings = Findings::default();
        let (binding_valid, count, clock, _) = check_binding(&acoustic, &case, &mut findings);
        assert!(binding_valid);
        assert_eq!(count, 6);
        assert_eq!(clock.as_deref(), Some("SYNTHETIC-UTC-REFERENCE-001"));
        assert!(findings.incomplete);
        assert_eq!(findings.disposition(), "INCOMPLETE");
    }

    #[test]
    fn wrong_hydrophone_trace_digest_is_invalid() {
        let (acoustic, mut case) = records();
        let channel = case["trial_evidence"]["channels"].as_array_mut().unwrap()
            .iter_mut().find(|item| item["physical_quantity"] == "underwater_sound_pressure").unwrap();
        channel["raw_trace_sha256"] = serde_json::json!("f".repeat(64));
        let mut findings = Findings::default();
        let (valid, _, _, _) = check_binding(&acoustic, &case, &mut findings);
        assert!(!valid);
        assert!(findings.invalid);
    }

    #[test]
    fn missing_shaft_torque_channel_is_incomplete_and_not_binding_valid() {
        let (acoustic, mut case) = records();
        case["trial_evidence"]["channels"].as_array_mut().unwrap()
            .retain(|item| item["physical_quantity"] != "shaft_torque");
        let mut findings = Findings::default();
        let (valid, _, _, _) = check_binding(&acoustic, &case, &mut findings);
        assert!(!valid);
        assert!(findings.incomplete);
    }

    #[test]
    fn different_reference_clock_is_incomparable() {
        let (acoustic, mut case) = records();
        let mut alternate = case["trial_evidence"]["timebases"][1].clone();
        alternate["timebase_id"] = serde_json::json!("SYNTHETIC-OTHER-CLOCK");
        alternate["reference_clock_id"] = serde_json::json!("OTHER-REFERENCE-CLOCK");
        case["trial_evidence"]["timebases"].as_array_mut().unwrap().push(alternate);
        let torque = case["trial_evidence"]["channels"].as_array_mut().unwrap()
            .iter_mut().find(|item| item["physical_quantity"] == "shaft_torque").unwrap();
        torque["timebase_id"] = serde_json::json!("SYNTHETIC-OTHER-CLOCK");
        let mut findings = Findings::default();
        let (valid, _, _, _) = check_binding(&acoustic, &case, &mut findings);
        assert!(!valid);
        assert!(findings.incomparable);
    }

    #[test]
    fn hydrophone_calibration_certificate_digest_must_match() {
        let (acoustic, mut case) = records();
        let cal = case["trial_evidence"]["calibration_records"].as_array_mut().unwrap()
            .iter_mut().find(|item| item["calibration_record_id"] == "SYNTHETIC-CAL-001").unwrap();
        cal["certificate_sha256"] = serde_json::json!("e".repeat(64));
        let mut findings = Findings::default();
        let (valid, _, _, _) = check_binding(&acoustic, &case, &mut findings);
        assert!(!valid);
        assert!(findings.invalid);
    }

    #[test]
    fn expired_timebase_cannot_support_the_acoustic_window() {
        let (acoustic, mut case) = records();
        case["trial_evidence"]["timebases"].as_array_mut().unwrap()
            .iter_mut().find(|item| item["timebase_id"] == "SYNTHETIC-ACOUSTIC-TIMEBASE-001").unwrap()
            ["valid_until"] = serde_json::json!("2026-10-15T09:59:59Z");
        let mut findings = Findings::default();
        let (valid, _, _, _) = check_binding(&acoustic, &case, &mut findings);
        assert!(!valid);
        assert!(findings.invalid);
    }

    #[test]
    fn unresolved_clock_uncertainty_remains_incomplete_not_zero() {
        let (acoustic, case) = records();
        let mut findings = Findings::default();
        let (valid, _, _, _) = check_binding(&acoustic, &case, &mut findings);
        assert!(valid);
        assert!(findings.incomplete);
    }

    #[test]
    fn acoustic_window_must_exactly_match_hydrophone_channel_window() {
        let (acoustic, mut case) = records();
        let channel = case["trial_evidence"]["channels"].as_array_mut().unwrap()
            .iter_mut().find(|item| item["physical_quantity"] == "underwater_sound_pressure").unwrap();
        channel["sample_window_end"] = serde_json::json!("2026-10-15T10:00:59Z");
        let mut findings = Findings::default();
        let (valid, _, _, _) = check_binding(&acoustic, &case, &mut findings);
        assert!(!valid);
        assert!(findings.incomparable);
    }
}
