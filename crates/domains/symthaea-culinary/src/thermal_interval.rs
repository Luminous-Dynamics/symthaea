// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Interval-preserving thermal analysis over evidence-bearing observations.
//!
//! This evaluator does not flatten bounded observations into a point-only
//! ThermalTrajectory. It propagates lower, nominal and upper temperature tracks
//! through the same caller-supplied piecewise-linear D/z model and returns three
//! conditional numerical outputs. Those bounds are not statistical confidence
//! intervals unless the input evidence explicitly gives them that meaning.
//!
//! This is not a food-safety, pasteurization, doneness, or regulatory verdict.
//! D/z applicability to a particular organism and food matrix requires separate
//! evidence and validation. Source labels are declarations, not authentication.

use serde_json::Value;

use crate::thermal_observation::validate_thermal_observation_log_v1;

const MEASUREMENT_KINDS: &[&str] = &[
    "product_core_temperature",
    "product_surface_temperature",
    "process_medium_temperature",
    "ambient_temperature",
    "equipment_temperature",
];
const EVIDENCE_STATUSES: &[&str] = &["synthetic", "unreviewed", "reviewed", "qualified"];

/// D/z parameters and their declared scope/provenance.
///
/// The parameter values are supplied by the caller. This type does not select,
/// validate scientifically, or recommend a microorganism/food-specific model.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalKineticsModelV1 {
    pub model_id: String,
    pub target_organism: String,
    pub food_matrix: String,
    pub d_ref_minutes: f64,
    pub t_ref_c: f64,
    pub z_c: f64,
    pub evidence: Value,
}

/// A mathematical envelope conditional on the selected observations and D/z model.
///
/// The type deliberately contains no safe/unsafe or pass/fail field. The returned
/// log10-reduction values are model outputs, not a certification of food safety.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalIntervalEnvelopeV1 {
    pub log_id: String,
    pub trace_id: String,
    pub process_id: String,
    pub sensor_id: String,
    pub measurement_kind: String,
    /// Process steps in the order they appear in the selected time series.
    pub process_step_ids: Vec<String>,
    pub sample_count: usize,
    pub observed_count: usize,
    pub estimated_count: usize,
    pub start_utc: String,
    pub end_utc: String,
    pub elapsed_minutes: f64,
    pub kinetics_model: ThermalKineticsModelV1,
    pub lower_peak_temperature_c: f64,
    pub nominal_peak_temperature_c: f64,
    pub upper_peak_temperature_c: f64,
    pub lower_log10_reduction: f64,
    pub nominal_log10_reduction: f64,
    pub upper_log10_reduction: f64,
    /// Reasons that the log/model should not be treated as empirically qualified.
    pub qualification_blockers: Vec<String>,
    pub assumptions: Vec<String>,
}

#[derive(Debug, Clone)]
struct Sample {
    timestamp_seconds: f64,
    measured_at: String,
    step_id: String,
    location_description: String,
    lower_c: f64,
    value_c: f64,
    upper_c: f64,
    status: String,
}

#[derive(Debug, Clone, Copy)]
struct CurvePoint {
    time_minutes: f64,
    temperature_c: f64,
}

fn non_empty<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty())
}

fn finite_number(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64).filter(|number| number.is_finite())
}

fn parse_component(value: &str) -> Option<u32> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}

fn leap_year(year: u32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: u32, month: u32) -> Option<u32> {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 => Some(if leap_year(year) { 29 } else { 28 }),
        _ => None,
    }
}

fn days_from_civil(year: u32, month: u32, day: u32) -> i64 {
    // Gregorian civil date -> days from an arbitrary epoch. Only differences
    // are used, so the chosen epoch is immaterial.
    let mut y = i64::from(year);
    if month <= 2 {
        y -= 1;
    }
    let era = y.div_euclid(400);
    let year_of_era = y - era * 400;
    let adjusted_month = i64::from(month) + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * adjusted_month + 2) / 5 + i64::from(day) - 1;
    let day_of_era =
        year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era
}

/// Parse the UTC RFC3339 subset used by thermal observations and return seconds.
///
/// Leap seconds are rejected rather than silently normalized. Date, clock and
/// fractional-second fields are checked; a timestamp that merely looks shaped
/// correctly is not enough to calculate elapsed time.
fn parse_utc_seconds(value: &str) -> Option<f64> {
    if !value.is_ascii() || !value.ends_with('Z') {
        return None;
    }
    let without_z = value.strip_suffix('Z')?;
    let (clock, fractional) = match without_z.split_once('.') {
        Some((clock, fraction)) => {
            if fraction.is_empty() || !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            (clock, Some(fraction))
        }
        None => (without_z, None),
    };
    let bytes = clock.as_bytes();
    if bytes.len() != 19
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return None;
    }
    let year = parse_component(&clock[0..4])?;
    let month = parse_component(&clock[5..7])?;
    let day = parse_component(&clock[8..10])?;
    let hour = parse_component(&clock[11..13])?;
    let minute = parse_component(&clock[14..16])?;
    let second = parse_component(&clock[17..19])?;

    if year == 0
        || month == 0
        || day == 0
        || day > days_in_month(year, month)?
        || hour > 23
        || minute > 59
        || second > 59
    {
        return None;
    }
    let fractional_seconds = match fractional {
        Some(digits) => format!("0.{digits}").parse::<f64>().ok()?,
        None => 0.0,
    };
    let whole_seconds = days_from_civil(year, month, day) as f64 * 86_400.0
        + f64::from(hour) * 3_600.0
        + f64::from(minute) * 60.0
        + f64::from(second);
    let result = whole_seconds + fractional_seconds;
    result.is_finite().then_some(result)
}

fn validate_kinetics_model(
    model: &ThermalKineticsModelV1,
    errors: &mut Vec<String>,
    blockers: &mut Vec<String>,
) {
    if model.model_id.trim().is_empty() {
        errors.push("kinetics model_id is required".to_owned());
    }
    if model.target_organism.trim().is_empty() {
        errors.push("kinetics target_organism is required".to_owned());
    }
    if model.food_matrix.trim().is_empty() {
        errors.push("kinetics food_matrix is required".to_owned());
    }
    if !model.d_ref_minutes.is_finite() || model.d_ref_minutes <= 0.0 {
        errors.push("kinetics d_ref_minutes must be finite and positive".to_owned());
    }
    if !model.t_ref_c.is_finite() {
        errors.push("kinetics t_ref_c must be finite".to_owned());
    }
    if !model.z_c.is_finite() || model.z_c <= 0.0 {
        errors.push("kinetics z_c must be finite and positive".to_owned());
    }

    let source = non_empty(&model.evidence, "source_ref");
    let version = non_empty(&model.evidence, "source_version");
    let known_at = non_empty(&model.evidence, "known_at");
    let status = non_empty(&model.evidence, "evidence_status");
    if source.is_none() {
        errors.push("kinetics evidence.source_ref is required".to_owned());
    }
    if version.is_none() {
        errors.push("kinetics evidence.source_version is required".to_owned());
    }
    if !known_at.is_some_and(|timestamp| parse_utc_seconds(timestamp).is_some()) {
        errors.push("kinetics evidence.known_at must be a valid UTC timestamp ending in Z".to_owned());
    }
    if !status.is_some_and(|value| EVIDENCE_STATUSES.contains(&value)) {
        errors.push("kinetics evidence.evidence_status is invalid".to_owned());
    }
    if status == Some("qualified") && source.is_some_and(|value| value.starts_with("urn:synthetic:")) {
        errors.push("synthetic kinetics evidence cannot be marked qualified".to_owned());
    }
    if status != Some("qualified") {
        blockers.push(format!(
            "kinetics model evidence status is {:?}, not qualified",
            status.unwrap_or("missing")
        ));
    }
    blockers.push(
        "D/z applicability to this organism and food matrix is not independently validated here"
            .to_owned(),
    );
}

fn peak_temperature(points: &[CurvePoint]) -> f64 {
    points
        .iter()
        .map(|point| point.temperature_c)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Exact integral for an exponential lethality rate along a piecewise-linear
/// temperature segment. The exp_m1 form preserves precision for small ramps.
fn log10_reduction(
    points: &[CurvePoint],
    d_ref_minutes: f64,
    t_ref_c: f64,
    z_c: f64,
) -> Option<f64> {
    let mut total = 0.0;
    for segment in points.windows(2) {
        let left = segment[0];
        let right = segment[1];
        let dt = right.time_minutes - left.time_minutes;
        if !dt.is_finite() || dt <= 0.0 {
            return None;
        }
        let rate_at_left = 10.0_f64.powf((left.temperature_c - t_ref_c) / z_c)
            / d_ref_minutes;
        let exponent =
            std::f64::consts::LN_10 * (right.temperature_c - left.temperature_c) / z_c;
        let ramp_factor = if exponent == 0.0 {
            1.0
        } else {
            exponent.exp_m1() / exponent
        };
        let contribution = dt * rate_at_left * ramp_factor;
        if !rate_at_left.is_finite()
            || !ramp_factor.is_finite()
            || !contribution.is_finite()
            || contribution < 0.0
        {
            return None;
        }
        total += contribution;
        if !total.is_finite() {
            return None;
        }
    }
    Some(total)
}

/// Evaluate lower/nominal/upper thermal tracks without discarding input intervals.
///
/// The selected sequence must use a single sensor, measurement kind and
/// unchanged placement description; it needs at least two strictly ordered,
/// known bounded readings. The calculation assumes piecewise-linear
/// interpolation between time points and constant caller-supplied D/z
/// parameters. Unknown samples or invalid elapsed time fail closed.
pub fn evaluate_thermal_interval_envelope_v1(
    trace: &Value,
    log: &Value,
    sensor_id: &str,
    measurement_kind: &str,
    model: &ThermalKineticsModelV1,
) -> Result<ThermalIntervalEnvelopeV1, Vec<String>> {
    let mut errors = Vec::new();
    let mut blockers = Vec::new();
    validate_kinetics_model(model, &mut errors, &mut blockers);

    let log_report = match validate_thermal_observation_log_v1(trace, log) {
        Ok(report) => Some(report),
        Err(nested) => {
            errors.extend(nested.into_iter().map(|error| format!("observation log: {error}")));
            None
        }
    };
    if !MEASUREMENT_KINDS.contains(&measurement_kind) {
        errors.push(format!("unsupported measurement_kind {measurement_kind:?}"));
    }
    if sensor_id.trim().is_empty() {
        errors.push("sensor_id selector is required".to_owned());
    }
    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }

    let report = log_report.expect("errors were returned above when log validation failed");
    blockers.extend(report.blockers);

    let log_id = non_empty(log, "log_id").unwrap_or_default().to_owned();
    let trace_id = non_empty(trace, "trace_id").unwrap_or_default().to_owned();
    let process_id = non_empty(trace, "process_id").unwrap_or_default().to_owned();

    let observations = log
        .get("observations")
        .and_then(Value::as_array)
        .expect("validated observation log has an observations array");
    let mut samples = Vec::new();
    for observation in observations.iter().filter(|observation| {
        non_empty(observation, "sensor_id") == Some(sensor_id)
            && non_empty(observation, "measurement_kind") == Some(measurement_kind)
    }) {
        let observation_id = non_empty(observation, "observation_id").unwrap_or("<missing>");
        let status = observation
            .get("temperature_c")
            .and_then(|temperature| non_empty(temperature, "status"))
            .unwrap_or("missing");
        if status == "unknown" {
            errors.push(format!(
                "selected observation {observation_id:?} has an unknown temperature and cannot form a numeric envelope"
            ));
            continue;
        }
        if !matches!(status, "observed" | "estimated") {
            errors.push(format!(
                "selected observation {observation_id:?} has invalid temperature status"
            ));
            continue;
        }
        let Some(measured_at) = non_empty(observation, "measured_at") else {
            errors.push(format!("selected observation {observation_id:?} is missing measured_at"));
            continue;
        };
        let Some(timestamp_seconds) = parse_utc_seconds(measured_at) else {
            errors.push(format!(
                "selected observation {observation_id:?} has an invalid UTC timestamp"
            ));
            continue;
        };
        let temperature = observation.get("temperature_c").unwrap_or(&Value::Null);
        let parsed = (
            finite_number(temperature, "lower_c"),
            finite_number(temperature, "value_c"),
            finite_number(temperature, "upper_c"),
        );
        let (Some(lower_c), Some(value_c), Some(upper_c)) = parsed else {
            errors.push(format!(
                "selected observation {observation_id:?} lacks finite lower/value/upper temperatures"
            ));
            continue;
        };
        if lower_c > value_c || value_c > upper_c {
            errors.push(format!(
                "selected observation {observation_id:?} violates lower_c <= value_c <= upper_c"
            ));
            continue;
        }
        samples.push(Sample {
            timestamp_seconds,
            measured_at: measured_at.to_owned(),
            step_id: non_empty(observation, "step_id").unwrap_or_default().to_owned(),
            location_description: non_empty(observation, "location_description")
                .unwrap_or_default()
                .to_owned(),
            lower_c,
            value_c,
            upper_c,
            status: status.to_owned(),
        });
    }

    if samples.len() < 2 {
        errors.push("at least two selected known bounded observations are required".to_owned());
    }
    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }

    samples.sort_by(|left, right| left.timestamp_seconds.total_cmp(&right.timestamp_seconds));
    for pair in samples.windows(2) {
        if pair[0].timestamp_seconds >= pair[1].timestamp_seconds {
            errors.push("selected observations must have unique, strictly increasing timestamps".to_owned());
        }
    }
    let first_location = samples[0].location_description.as_str();
    if samples
        .iter()
        .any(|sample| sample.location_description != first_location)
    {
        errors.push(
            "selected series changes location_description; split it into separate trajectories"
                .to_owned(),
        );
    }
    let start_seconds = samples[0].timestamp_seconds;
    let elapsed_minutes = (samples[samples.len() - 1].timestamp_seconds - start_seconds) / 60.0;
    if !elapsed_minutes.is_finite() || elapsed_minutes <= 0.0 {
        errors.push("selected observations do not span a positive finite duration".to_owned());
    }
    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }

    let start_utc = samples[0].measured_at.clone();
    let end_utc = samples[samples.len() - 1].measured_at.clone();
    let mut process_step_ids = Vec::new();
    let mut observed_count = 0;
    let mut estimated_count = 0;
    let mut lower_curve = Vec::with_capacity(samples.len());
    let mut nominal_curve = Vec::with_capacity(samples.len());
    let mut upper_curve = Vec::with_capacity(samples.len());
    for sample in &samples {
        if process_step_ids.last() != Some(&sample.step_id) {
            process_step_ids.push(sample.step_id.clone());
        }
        match sample.status.as_str() {
            "observed" => observed_count += 1,
            "estimated" => estimated_count += 1,
            _ => {}
        }
        let time_minutes = (sample.timestamp_seconds - start_seconds) / 60.0;
        lower_curve.push(CurvePoint { time_minutes, temperature_c: sample.lower_c });
        nominal_curve.push(CurvePoint { time_minutes, temperature_c: sample.value_c });
        upper_curve.push(CurvePoint { time_minutes, temperature_c: sample.upper_c });
    }

    let lower_log10_reduction = log10_reduction(
        &lower_curve,
        model.d_ref_minutes,
        model.t_ref_c,
        model.z_c,
    );
    let nominal_log10_reduction = log10_reduction(
        &nominal_curve,
        model.d_ref_minutes,
        model.t_ref_c,
        model.z_c,
    );
    let upper_log10_reduction = log10_reduction(
        &upper_curve,
        model.d_ref_minutes,
        model.t_ref_c,
        model.z_c,
    );
    let (Some(lower_log10_reduction), Some(nominal_log10_reduction), Some(upper_log10_reduction)) =
        (lower_log10_reduction, nominal_log10_reduction, upper_log10_reduction)
    else {
        return Err(vec![
            "thermal envelope produced a non-finite or invalid D/z integral; no result emitted"
                .to_owned(),
        ]);
    };

    blockers.sort();
    blockers.dedup();
    Ok(ThermalIntervalEnvelopeV1 {
        log_id,
        trace_id,
        process_id,
        sensor_id: sensor_id.to_owned(),
        measurement_kind: measurement_kind.to_owned(),
        process_step_ids,
        sample_count: samples.len(),
        observed_count,
        estimated_count,
        start_utc,
        end_utc,
        elapsed_minutes,
        kinetics_model: model.clone(),
        lower_peak_temperature_c: peak_temperature(&lower_curve),
        nominal_peak_temperature_c: peak_temperature(&nominal_curve),
        upper_peak_temperature_c: peak_temperature(&upper_curve),
        lower_log10_reduction,
        nominal_log10_reduction,
        upper_log10_reduction,
        qualification_blockers: blockers,
        assumptions: vec![
            "Temperature lower/value/upper tracks are interpolated piecewise-linearly between samples; behavior between observations is not directly measured.".to_owned(),
            "D/z outputs are conditional on the exact caller-supplied constant-parameter model and are not a food-safety, pasteurization, doneness, or regulatory assessment.".to_owned(),
            "Temperature bounds are not statistical confidence intervals unless source evidence explicitly defines their coverage semantics.".to_owned(),
            "Evidence-status labels are declarations; this function does not authenticate sources or independently calibrate sensors.".to_owned(),
            "Sensor placement, food geometry, spatial cold spots, model validity, and regulatory applicability are not established by this calculation.".to_owned(),
        ],
    })
}
