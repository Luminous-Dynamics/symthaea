// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Source-bearing thermal observations bound to a Culinary process trace.
//!
//! This module preserves observation times, sensor identity, placement,
//! calibration status, uncertainty and provenance. It does not determine food
//! safety, doneness, pasteurization, microbial reduction, or sensory quality.
//! A temperature log is only as representative as its sensor placement and
//! evidence; numeric readings alone are not a safety verdict.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const SENSOR_KINDS: &[&str] = &[
    "thermocouple", "thermistor", "infrared", "oven_controller", "other",
];
const MEASUREMENT_KINDS: &[&str] = &[
    "product_core_temperature",
    "product_surface_temperature",
    "process_medium_temperature",
    "ambient_temperature",
    "equipment_temperature",
];
const CALIBRATION_STATUSES: &[&str] = &["calibrated", "unverified", "expired", "unknown"];
const VALUE_STATUSES: &[&str] = &["observed", "estimated", "unknown"];

/// Summary of thermal observation log validation, never a food-safety verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThermalObservationLogResultV1 {
    pub observation_count: usize,
    pub sensor_count: usize,
    pub observed_count: usize,
    pub estimated_count: usize,
    pub unknown_count: usize,
    /// True only when every source/sensor/observation record is marked qualified.
    /// This reports declared labels, not source authenticity or sensor calibration.
    pub all_evidence_records_marked_qualified: bool,
    pub blockers: Vec<String>,
}

fn non_empty<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty())
}

fn number(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
}

fn timestamp_shape(value: &str) -> bool {
    let b = value.as_bytes();
    if b.len() < 20
        || b.get(4) != Some(&b'-')
        || b.get(7) != Some(&b'-')
        || b.get(10) != Some(&b'T')
        || b.get(13) != Some(&b':')
        || b.get(16) != Some(&b':')
        || !b[..4].iter().all(u8::is_ascii_digit)
        || !b[5..7].iter().all(u8::is_ascii_digit)
        || !b[8..10].iter().all(u8::is_ascii_digit)
        || !b[11..13].iter().all(u8::is_ascii_digit)
        || !b[14..16].iter().all(u8::is_ascii_digit)
        || !b[17..19].iter().all(u8::is_ascii_digit)
    {
        return false;
    }
    let tz_start = if b.last() == Some(&b'Z') {
        b.len() - 1
    } else if b.len() >= 25 {
        let start = b.len() - 6;
        if !matches!(b[start], b'+' | b'-')
            || b[start + 3] != b':'
            || !b[start + 1..start + 3].iter().all(u8::is_ascii_digit)
            || !b[start + 4..start + 6].iter().all(u8::is_ascii_digit)
        {
            return false;
        }
        start
    } else {
        return false;
    };
    if tz_start == 19 {
        return true;
    }
    b.get(19) == Some(&b'.')
        && tz_start > 20
        && b[20..tz_start].iter().all(u8::is_ascii_digit)
}

fn utc_timestamp_shape(value: &str) -> bool {
    timestamp_shape(value) && value.ends_with('Z')
}

fn validate_evidence(
    value: &Value,
    label: &str,
    errors: &mut Vec<String>,
) -> bool {
    let source = non_empty(value, "source_ref");
    let version = non_empty(value, "source_version");
    let known_at = non_empty(value, "known_at");
    let status = non_empty(value, "evidence_status");
    if source.is_none() {
        errors.push(format!("{label}.source_ref is required"));
    }
    if version.is_none() {
        errors.push(format!("{label}.source_version is required"));
    }
    if !known_at.is_some_and(timestamp_shape) {
        errors.push(format!("{label}.known_at must have an RFC3339 timestamp shape"));
    }
    if !matches!(status, Some("synthetic" | "unreviewed" | "reviewed" | "qualified")) {
        errors.push(format!("{label}.evidence_status is invalid"));
    }
    if status == Some("qualified")
        && source.is_some_and(|s| s.starts_with("urn:synthetic:"))
    {
        errors.push(format!("{label} cannot mark a synthetic source qualified"));
        return false;
    }
    status == Some("qualified")
        && source.is_some()
        && version.is_some()
        && known_at.is_some_and(timestamp_shape)
}

fn finite_nonnegative(value: &Value, key: &str) -> Option<f64> {
    number(value, key).filter(|x| *x >= 0.0)
}

fn validate_measurement(
    value: &Value,
    label: &str,
    errors: &mut Vec<String>,
    blockers: &mut Vec<String>,
) -> Option<&'static str> {
    if non_empty(value, "unit") != Some("degC") {
        errors.push(format!("{label}.unit must be degC"));
    }
    match non_empty(value, "status") {
        Some("observed" | "estimated") => {
            let (reading, lower, upper) = (
                number(value, "value_c"),
                number(value, "lower_c"),
                number(value, "upper_c"),
            );
            match (reading, lower, upper) {
                (Some(reading), Some(lower), Some(upper))
                    if lower <= reading && reading <= upper =>
                {
                    if value.get("unknown_reason").is_some() {
                        errors.push(format!("{label} must not include unknown_reason for a bounded reading"));
                    }
                }
                _ => errors.push(format!(
                    "{label} must satisfy finite lower_c <= value_c <= upper_c"
                )),
            }
            non_empty(value, "status")
        }
        Some("unknown") => {
            let all_null = ["value_c", "lower_c", "upper_c"]
                .iter()
                .all(|key| value.get(key).is_some_and(Value::is_null));
            if !all_null {
                errors.push(format!("{label} with unknown status must set all temperature values to null"));
            }
            if non_empty(value, "unknown_reason").is_none() {
                errors.push(format!("{label} with unknown status requires unknown_reason"));
            }
            blockers.push(format!("{label} is explicitly unknown"));
            Some("unknown")
        }
        _ => {
            errors.push(format!("{label}.status must be observed, estimated, or unknown"));
            None
        }
    }
}

/// Validate a thermal observation log against the existing process trace.
///
/// Requires exact trace/process IDs, observation-to-step and observation-to-sensor
/// references, unique observation IDs, unique per-sensor/per-step sequence
/// indices, and monotonic timestamps along each declared sequence. It reports
/// evidence labels but cannot authenticate the evidence, validate a calibration,
/// or infer safety/doneness. All temperatures are explicit Celsius intervals.
pub fn validate_thermal_observation_log_v1(
    trace: &Value,
    log: &Value,
) -> Result<ThermalObservationLogResultV1, Vec<String>> {
    let mut errors = Vec::new();
    let mut blockers = Vec::new();
    let mut all_qualified = true;

    if non_empty(log, "schema_version") != Some("1.0.0") {
        errors.push("schema_version must be 1.0.0".to_owned());
    }
    if non_empty(log, "log_id").is_none() {
        errors.push("log_id is required".to_owned());
    }
    if non_empty(log, "trace_id") != non_empty(trace, "trace_id")
        || non_empty(log, "trace_id").is_none()
    {
        errors.push("log.trace_id must match the referenced process trace".to_owned());
    }
    if non_empty(log, "process_id") != non_empty(trace, "process_id")
        || non_empty(log, "process_id").is_none()
    {
        errors.push("log.process_id must match the referenced process trace".to_owned());
    }

    let trace_steps: BTreeSet<String> = trace
        .get("steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|step| non_empty(step, "step_id").map(str::to_owned))
        .collect();

    let sensors: &[Value] = match log.get("sensors").and_then(Value::as_array) {
        Some(items) if !items.is_empty() => items.as_slice(),
        _ => {
            errors.push("sensors must be a non-empty array".to_owned());
            &[]
        }
    };
    let observations: &[Value] = match log.get("observations").and_then(Value::as_array) {
        Some(items) if !items.is_empty() => items.as_slice(),
        _ => {
            errors.push("observations must be a non-empty array".to_owned());
            &[]
        }
    };

    let mut sensor_ids = BTreeSet::new();
    let mut sensor_index = BTreeMap::<String, usize>::new();
    for (index, sensor) in sensors.iter().enumerate() {
        let label = format!("sensors[{index}]");
        let Some(id) = non_empty(sensor, "sensor_id") else {
            errors.push(format!("{label}.sensor_id is required"));
            continue;
        };
        if !sensor_ids.insert(id.to_owned()) {
            errors.push(format!("duplicate sensor_id {id:?}"));
            continue;
        }
        sensor_index.insert(id.to_owned(), index);

        if !non_empty(sensor, "sensor_kind").is_some_and(|kind| SENSOR_KINDS.contains(&kind)) {
            errors.push(format!("{label}.sensor_kind is invalid"));
        }
        if !non_empty(sensor, "calibration_status")
            .is_some_and(|status| CALIBRATION_STATUSES.contains(&status))
        {
            errors.push(format!("{label}.calibration_status is invalid"));
        }
        match finite_nonnegative(sensor, "stated_accuracy_c") {
            Some(_) => {}
            None => errors.push(format!("{label}.stated_accuracy_c must be finite and non-negative")),
        }
        let calibration_status = non_empty(sensor, "calibration_status");
        let calibrated_at = non_empty(sensor, "last_calibrated_at");
        let calibration_evidence = sensor.get("calibration_evidence").unwrap_or(&Value::Null);
        let calibration_evidence_ok = validate_evidence(
            calibration_evidence,
            &format!("{label}.calibration_evidence"),
            &mut errors,
        );
        all_qualified &= calibration_evidence_ok;

        if calibration_status == Some("calibrated") {
            if !calibrated_at.is_some_and(utc_timestamp_shape) {
                errors.push(format!("{label} marked calibrated needs a valid UTC last_calibrated_at ending in Z"));
            }
            if non_empty(calibration_evidence, "evidence_status") == Some("synthetic")
                || non_empty(calibration_evidence, "source_ref")
                    .is_some_and(|s| s.starts_with("urn:synthetic:"))
            {
                errors.push(format!("{label} cannot claim calibration from synthetic evidence"));
            }
        } else if calibrated_at.is_some_and(|s| !utc_timestamp_shape(s)) {
            errors.push(format!("{label}.last_calibrated_at must be a valid UTC timestamp ending in Z or null"));
        }
        if calibration_status != Some("calibrated") {
            blockers.push(format!(
                "sensor {id:?} calibration status is {:?}",
                calibration_status.unwrap_or("missing")
            ));
        }
    }

    let mut observation_ids = BTreeSet::new();
    let mut sequence = BTreeMap::<(String, String, u64), String>::new();
    let mut timeline = BTreeMap::<(String, String), Vec<(u64, String)>>::new();
    let mut observed_count = 0;
    let mut estimated_count = 0;
    let mut unknown_count = 0;

    for (index, observation) in observations.iter().enumerate() {
        let label = format!("observations[{index}]");
        let Some(id) = non_empty(observation, "observation_id") else {
            errors.push(format!("{label}.observation_id is required"));
            continue;
        };
        if !observation_ids.insert(id.to_owned()) {
            errors.push(format!("duplicate observation_id {id:?}"));
        }
        let Some(step_id) = non_empty(observation, "step_id") else {
            errors.push(format!("{label}.step_id is required"));
            continue;
        };
        if !trace_steps.contains(step_id) {
            errors.push(format!("{label} references step {step_id:?} absent from process trace"));
        }
        let Some(sensor_id) = non_empty(observation, "sensor_id") else {
            errors.push(format!("{label}.sensor_id is required"));
            continue;
        };
        if !sensor_index.contains_key(sensor_id) {
            errors.push(format!("{label} references unknown sensor {sensor_id:?}"));
        }
        if !non_empty(observation, "measurement_kind")
            .is_some_and(|kind| MEASUREMENT_KINDS.contains(&kind))
        {
            errors.push(format!("{label}.measurement_kind is invalid"));
        }
        if non_empty(observation, "location_description").is_none() {
            errors.push(format!("{label}.location_description is required"));
        }
        let Some(timestamp) = non_empty(observation, "measured_at") else {
            errors.push(format!("{label}.measured_at is required"));
            continue;
        };
        if !utc_timestamp_shape(timestamp) {
            errors.push(format!("{label}.measured_at must be an RFC3339 UTC timestamp ending in Z"));
        }
        let Some(seq) = observation.get("sequence_index").and_then(Value::as_u64) else {
            errors.push(format!("{label}.sequence_index must be an unsigned integer"));
            continue;
        };
        let timeline_key = (sensor_id.to_owned(), step_id.to_owned());
        if sequence
            .insert((sensor_id.to_owned(), step_id.to_owned(), seq), id.to_owned())
            .is_some()
        {
            errors.push(format!(
                "{label} duplicates sequence_index {seq} for sensor {sensor_id:?} in step {step_id:?}"
            ));
        }
        timeline.entry(timeline_key).or_default().push((seq, timestamp.to_owned()));

        if !non_empty(observation, "measurement_kind")
            .is_some_and(|kind| MEASUREMENT_KINDS.contains(&kind))
        {
            // Invalid values were reported above; do not mask them here.
        }
        match validate_measurement(
            observation.get("temperature_c").unwrap_or(&Value::Null),
            &format!("{label}.temperature_c"),
            &mut errors,
            &mut blockers,
        ) {
            Some("observed") => observed_count += 1,
            Some("estimated") => estimated_count += 1,
            Some("unknown") => unknown_count += 1,
            _ => {}
        }
        all_qualified &= validate_evidence(
            observation.get("evidence").unwrap_or(&Value::Null),
            &format!("{label}.evidence"),
            &mut errors,
        );
    }

    for ((sensor_id, step_id), points) in &mut timeline {
        points.sort_by_key(|(sequence, _)| *sequence);
        for pair in points.windows(2) {
            if pair[0].1 >= pair[1].1 {
                errors.push(format!(
                    "timestamps must strictly increase for sensor {sensor_id:?} in step {step_id:?} by sequence_index"
                ));
            }
        }
    }

    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }
    blockers.sort();
    blockers.dedup();
    Ok(ThermalObservationLogResultV1 {
        observation_count: observations.len(),
        sensor_count: sensors.len(),
        observed_count,
        estimated_count,
        unknown_count,
        all_evidence_records_marked_qualified: all_qualified,
        blockers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace() -> Value {
        serde_json::from_str(include_str!(
            "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-trace.json"
        ))
        .unwrap()
    }

    fn log() -> Value {
        serde_json::from_str(include_str!(
            "../fixtures/thermal-observation-v1/synthetic-boil-temperature-log.json"
        ))
        .unwrap()
    }

    #[test]
    fn synthetic_log_binds_to_trace_without_issuing_safety_verdict() {
        let result = validate_thermal_observation_log_v1(&trace(), &log()).unwrap();
        assert_eq!(result.observation_count, 3);
        assert_eq!(result.sensor_count, 1);
        assert_eq!(result.estimated_count, 3);
        assert_eq!(result.unknown_count, 0);
        assert!(!result.all_evidence_records_marked_qualified);
    }

    #[test]
    fn mismatched_trace_id_fails_closed() {
        let mut value = log();
        value["trace_id"] = Value::String("other-trace".into());
        assert!(validate_thermal_observation_log_v1(&trace(), &value).is_err());
    }

    #[test]
    fn unknown_sensor_fails_closed() {
        let mut value = log();
        value["observations"][0]["sensor_id"] = Value::String("missing-probe".into());
        assert!(validate_thermal_observation_log_v1(&trace(), &value).is_err());
    }

    #[test]
    fn missing_step_fails_closed() {
        let mut value = log();
        value["observations"][0]["step_id"] = Value::String("missing-step".into());
        assert!(validate_thermal_observation_log_v1(&trace(), &value).is_err());
    }

    #[test]
    fn duplicate_observation_ids_fail_closed() {
        let mut value = log();
        value["observations"][1]["observation_id"] = value["observations"][0]["observation_id"].clone();
        assert!(validate_thermal_observation_log_v1(&trace(), &value).is_err());
    }

    #[test]
    fn duplicate_sequence_indices_fail_closed() {
        let mut value = log();
        value["observations"][1]["sequence_index"] = serde_json::json!(0);
        assert!(validate_thermal_observation_log_v1(&trace(), &value).is_err());
    }

    #[test]
    fn timestamp_must_increase_in_sequence_order() {
        let mut value = log();
        value["observations"][2]["measured_at"] = value["observations"][0]["measured_at"].clone();
        assert!(validate_thermal_observation_log_v1(&trace(), &value).is_err());
    }

    #[test]
    fn unknown_temperature_remains_unknown_and_adds_blocker() {
        let mut value = log();
        value["observations"][2]["temperature_c"] = serde_json::json!({
            "status":"unknown", "value_c":null, "lower_c":null, "upper_c":null,
            "unit":"degC", "unknown_reason":"probe disconnected during final interval"
        });
        let result = validate_thermal_observation_log_v1(&trace(), &value).unwrap();
        assert_eq!(result.unknown_count, 1);
        assert!(result.blockers.iter().any(|b| b.contains("explicitly unknown")));
    }

    #[test]
    fn invalid_interval_fails_closed() {
        let mut value = log();
        value["observations"][0]["temperature_c"]["lower_c"] = serde_json::json!(30.0);
        assert!(validate_thermal_observation_log_v1(&trace(), &value).is_err());
    }

    #[test]
    fn sensor_marked_calibrated_with_synthetic_calibration_fails_closed() {
        let mut value = log();
        value["sensors"][0]["calibration_status"] = Value::String("calibrated".into());
        value["sensors"][0]["last_calibrated_at"] =
            Value::String("2026-10-18T00:00:00Z".into());
        assert!(validate_thermal_observation_log_v1(&trace(), &value).is_err());
    }

    #[test]
    fn observation_array_order_does_not_change_result() {
        let t = trace();
        let value = log();
        let expected = validate_thermal_observation_log_v1(&t, &value).unwrap();
        let mut reversed = value;
        reversed["observations"].as_array_mut().unwrap().reverse();
        let actual = validate_thermal_observation_log_v1(&t, &reversed).unwrap();
        assert_eq!(actual, expected);
    }


    #[test]
    fn invalid_referenced_trace_fails_closed() {
        let mut value = trace();
        value["steps"][0]["step_id"] = Value::String("changed-step".into());
        assert!(validate_thermal_observation_log_v1(&value, &log()).is_err());
    }

    #[test]
    fn synthetic_source_cannot_be_qualified() {
        let mut value = log();
        value["observations"][0]["evidence"]["evidence_status"] =
            Value::String("qualified".into());
        assert!(validate_thermal_observation_log_v1(&trace(), &value).is_err());
    }
}
