// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed integration of a raw power trace over an explicitly bounded task window.
//!
//! This arithmetic core does not validate a real wattmeter, task definition, power
//! boundary, calibration, IEC procedure, or physical display. The caller must bind
//! its result to those identities in the measurement-run record.

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerSample {
    /// Monotonic time in seconds on the declared acquisition clock.
    pub time_seconds: f64,
    /// Power in watts measured at the declared accounting boundary.
    pub power_watts: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnergyIntegrationConfig {
    pub window_start_seconds: f64,
    pub window_end_seconds: f64,
    /// Largest raw-sample interval allowed to intersect the integration window.
    pub max_sample_gap_seconds: f64,
    /// Explicitly names the meter/power accounting boundary, not just the DUT.
    pub accounting_boundary_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnergyIntegrationReport {
    pub energy_joules: f64,
    pub window_start_seconds: f64,
    pub window_end_seconds: f64,
    pub window_duration_seconds: f64,
    pub accounting_boundary_id: String,
    pub raw_sample_count: usize,
    /// Number of points after clipping/interpolating the window boundaries.
    pub integration_point_count: usize,
    pub max_intersecting_sample_gap_seconds: f64,
    pub integration_method: &'static str,
    pub claim_ceiling: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnergyTraceValidationReport {
    pub trace_id: String,
    pub trace_sha256: String,
    pub schema_sha256: String,
    pub integration: EnergyIntegrationReport,
    pub attempted_tasks: u64,
    pub successful_tasks: u64,
    pub energy_per_successful_task_joules: Option<f64>,
}

impl EnergyTraceValidationReport {
    /// Stable report that preserves the synthetic-only claim boundary.
    #[must_use]
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "verifier": "open-display-platform-verify",
            "verifier_version": env!("CARGO_PKG_VERSION"),
            "trace_id": self.trace_id,
            "trace_sha256": self.trace_sha256,
            "schema_sha256": self.schema_sha256,
            "record_state": "SYNTHETIC_TEST_ONLY",
            "energy_integration": {
                "energy_joules": self.integration.energy_joules,
                "window_start_seconds": self.integration.window_start_seconds,
                "window_end_seconds": self.integration.window_end_seconds,
                "window_duration_seconds": self.integration.window_duration_seconds,
                "accounting_boundary_id": self.integration.accounting_boundary_id,
                "raw_sample_count": self.integration.raw_sample_count,
                "integration_point_count": self.integration.integration_point_count,
                "max_intersecting_sample_gap_seconds": self.integration.max_intersecting_sample_gap_seconds,
                "integration_method": self.integration.integration_method,
                "claim_ceiling": self.integration.claim_ceiling
            },
            "task_outcomes": {
                "attempted_tasks": self.attempted_tasks,
                "successful_tasks": self.successful_tasks,
                "energy_per_successful_task_joules": self.energy_per_successful_task_joules
            },
            "status": "SYNTHETIC_ARITHMETIC_VALID_UNQUALIFIED",
            "claim_ceiling": "Synthetic power-trace arithmetic only. No wattmeter, calibration, power boundary suitability, IEC compliance, task utility, or physical display result is qualified."
        })
    }
}

/// Integrate a power trace over a bounded task window.
///
/// The trace must cover both window boundaries. If a boundary falls between raw
/// samples, power is linearly interpolated only within the bracketing sample pair.
/// Any raw-sample gap intersecting the window that exceeds the preregistered cap
/// rejects the metric; gaps are never silently bridged beyond that cap.
///
/// # Errors
///
/// Rejects empty/duplicate identities, invalid bounds, invalid or negative power,
/// non-monotonic timestamps, insufficient window coverage, excessive sample gaps,
/// and non-finite integrated energy.
#[must_use]
pub fn integrate_power_trace(
    samples: &[PowerSample],
    config: &EnergyIntegrationConfig,
) -> Result<EnergyIntegrationReport, Vec<String>> {
    let mut errors = Vec::new();

    if config.accounting_boundary_id.trim().is_empty() {
        errors.push("accounting boundary identity must be nonempty".to_owned());
    }
    if !config.window_start_seconds.is_finite()
        || !config.window_end_seconds.is_finite()
        || config.window_end_seconds <= config.window_start_seconds
    {
        errors.push("energy window bounds must be finite and end after start".to_owned());
    }
    if !config.max_sample_gap_seconds.is_finite() || config.max_sample_gap_seconds <= 0.0 {
        errors.push("maximum sample gap must be finite and positive".to_owned());
    }
    if samples.len() < 2 {
        errors.push("at least two power samples are required".to_owned());
    }

    let mut previous_time: Option<f64> = None;
    for (index, sample) in samples.iter().enumerate() {
        if !sample.time_seconds.is_finite() {
            errors.push(format!("power sample {index} timestamp must be finite"));
        }
        if !sample.power_watts.is_finite() || sample.power_watts < 0.0 {
            errors.push(format!(
                "power sample {index} must have finite nonnegative watts"
            ));
        }
        if previous_time.is_some_and(|previous| sample.time_seconds <= previous) {
            errors.push(format!(
                "power sample timestamps must be strictly increasing at index {index}"
            ));
        }
        previous_time = Some(sample.time_seconds);
    }

    if let (Some(first), Some(last)) = (samples.first(), samples.last()) {
        if config.window_start_seconds < first.time_seconds
            || config.window_end_seconds > last.time_seconds
        {
            errors.push("power trace does not cover both energy window boundaries".to_owned());
        }
    }

    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }

    let mut max_intersecting_sample_gap_seconds = 0.0_f64;
    for pair in samples.windows(2) {
        if pair[1].time_seconds > config.window_start_seconds
            && pair[0].time_seconds < config.window_end_seconds
        {
            let gap = pair[1].time_seconds - pair[0].time_seconds;
            max_intersecting_sample_gap_seconds = max_intersecting_sample_gap_seconds.max(gap);
        }
    }
    if max_intersecting_sample_gap_seconds > config.max_sample_gap_seconds {
        return Err(vec![format!(
            "power sample gap exceeds predeclared maximum: {} seconds, allowed {}",
            max_intersecting_sample_gap_seconds, config.max_sample_gap_seconds
        )]);
    }

    let start_power = interpolate_power(samples, config.window_start_seconds);
    let end_power = interpolate_power(samples, config.window_end_seconds);
    let mut integration_points = Vec::with_capacity(samples.len() + 2);
    integration_points.push(PowerSample {
        time_seconds: config.window_start_seconds,
        power_watts: start_power,
    });
    integration_points.extend(
        samples
            .iter()
            .copied()
            .filter(|sample| {
                sample.time_seconds > config.window_start_seconds
                    && sample.time_seconds < config.window_end_seconds
            }),
    );
    integration_points.push(PowerSample {
        time_seconds: config.window_end_seconds,
        power_watts: end_power,
    });

    let mut energy_joules = 0.0_f64;
    for pair in integration_points.windows(2) {
        let delta_seconds = pair[1].time_seconds - pair[0].time_seconds;
        if !delta_seconds.is_finite() || delta_seconds <= 0.0 {
            return Err(vec![
                "clipped integration points are not strictly time-ordered".to_owned(),
            ]);
        }
        energy_joules +=
            (pair[0].power_watts + pair[1].power_watts) * 0.5 * delta_seconds;
        if !energy_joules.is_finite() {
            return Err(vec!["integrated energy is not finite".to_owned()]);
        }
    }

    Ok(EnergyIntegrationReport {
        energy_joules,
        window_start_seconds: config.window_start_seconds,
        window_end_seconds: config.window_end_seconds,
        window_duration_seconds: config.window_end_seconds - config.window_start_seconds,
        accounting_boundary_id: config.accounting_boundary_id.clone(),
        raw_sample_count: samples.len(),
        integration_point_count: integration_points.len(),
        max_intersecting_sample_gap_seconds,
        integration_method: "TRAPEZOIDAL_WITH_LINEAR_BOUNDARY_INTERPOLATION",
        claim_ceiling: "ARITHMETIC_ONLY_NOT_PHYSICAL_POWER_QUALIFICATION",
    })
}

fn interpolate_power(samples: &[PowerSample], time_seconds: f64) -> f64 {
    if let Some(sample) = samples
        .iter()
        .find(|sample| sample.time_seconds == time_seconds)
    {
        return sample.power_watts;
    }

    for pair in samples.windows(2) {
        if pair[0].time_seconds < time_seconds && time_seconds < pair[1].time_seconds {
            let ratio = (time_seconds - pair[0].time_seconds)
                / (pair[1].time_seconds - pair[0].time_seconds);
            return pair[0].power_watts
                + ratio * (pair[1].power_watts - pair[0].power_watts);
        }
    }

    // Coverage and ordering checks make this branch unreachable for a valid trace.
    f64::NAN
}

/// Validate and integrate a synthetic-only power-trace record against Draft 2020-12.
///
/// The input must explicitly remain SYNTHETIC_TEST_ONLY; this version intentionally
/// does not promote user-supplied physical samples into validated measurements.
///
/// # Errors
///
/// Returns schema, semantic, trace-integrity, sampling-gap or task-count errors.
pub fn validate_energy_trace_str(
    trace_text: &str,
    schema_text: &str,
) -> Result<EnergyTraceValidationReport, Vec<String>> {
    let schema: Value = match serde_json::from_str(schema_text) {
        Ok(value) => value,
        Err(error) => return Err(vec![format!("energy schema JSON parse error: {error}")]),
    };
    let trace: Value = match serde_json::from_str(trace_text) {
        Ok(value) => value,
        Err(error) => return Err(vec![format!("energy trace JSON parse error: {error}")]),
    };

    let mut errors = Vec::new();
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        errors.push("energy schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    match jsonschema::draft202012::new(&schema) {
        Ok(validator) => {
            for error in validator.iter_errors(&trace) {
                errors.push(format!("energy trace JSON Schema violation: {error}"));
            }
        }
        Err(error) => errors.push(format!("energy schema compilation failed: {error}")),
    }

    if trace.get("record_state").and_then(Value::as_str) != Some("SYNTHETIC_TEST_ONLY") {
        errors.push("energy trace v0.1 only accepts SYNTHETIC_TEST_ONLY records".to_owned());
    }
    if trace.get("claim_ceiling").and_then(Value::as_str)
        != Some("SYNTHETIC_POWER_TRACE_ONLY_NOT_PHYSICAL_QUALIFICATION")
    {
        errors.push("energy trace claim ceiling is invalid".to_owned());
    }

    let Some(samples_value) = trace.get("samples").and_then(Value::as_array) else {
        errors.push("energy trace samples must be an array".to_owned());
        return Err(errors);
    };
    let samples: Vec<PowerSample> = samples_value
        .iter()
        .map(|sample| PowerSample {
            time_seconds: sample
                .get("time_seconds")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN),
            power_watts: sample
                .get("power_watts")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NAN),
        })
        .collect();

    let config = EnergyIntegrationConfig {
        window_start_seconds: trace
            .get("window_start_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN),
        window_end_seconds: trace
            .get("window_end_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN),
        max_sample_gap_seconds: trace
            .get("max_sample_gap_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN),
        accounting_boundary_id: trace
            .get("accounting_boundary_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
    };

    let integration = match integrate_power_trace(&samples, &config) {
        Ok(report) => Some(report),
        Err(trace_errors) => {
            errors.extend(trace_errors);
            None
        }
    };
    let attempted_tasks = trace
        .get("attempted_tasks")
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX);
    let successful_tasks = trace
        .get("successful_tasks")
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX);
    let energy_per_successful_task_joules = if let Some(report) = integration.as_ref() {
        match energy_per_successful_task(
            report.energy_joules,
            attempted_tasks,
            successful_tasks,
        ) {
            Ok(value) => value,
            Err(task_errors) => {
                errors.extend(task_errors);
                None
            }
        }
    } else {
        None
    };

    errors.sort();
    errors.dedup();
    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(EnergyTraceValidationReport {
        trace_id: trace
            .get("trace_id")
            .and_then(Value::as_str)
            .expect("schema validates trace_id")
            .to_owned(),
        trace_sha256: super::sha256_hex(trace_text.as_bytes()),
        schema_sha256: super::sha256_hex(schema_text.as_bytes()),
        integration: integration.expect("valid energy trace must produce integration"),
        attempted_tasks,
        successful_tasks,
        energy_per_successful_task_joules,
    })
}

/// Calculate energy per successful task without inventing a value when none succeed.
///
/// # Errors
///
/// Rejects non-finite/negative energy and success counts larger than attempted
/// tasks. Returns None when there were no successful tasks.
#[must_use]
pub fn energy_per_successful_task(
    energy_joules: f64,
    attempted_tasks: u64,
    successful_tasks: u64,
) -> Result<Option<f64>, Vec<String>> {
    if !energy_joules.is_finite() || energy_joules < 0.0 {
        return Err(vec!["total energy must be finite and nonnegative".to_owned()]);
    }
    if successful_tasks > attempted_tasks {
        return Err(vec![
            "successful task count cannot exceed attempted task count".to_owned(),
        ]);
    }
    if successful_tasks == 0 {
        return Ok(None);
    }
    let result = energy_joules / successful_tasks as f64;
    if !result.is_finite() {
        return Err(vec!["energy per successful task is not finite".to_owned()]);
    }
    Ok(Some(result))
}

#[cfg(test)]
mod tests {
    use super::{
        EnergyIntegrationConfig, PowerSample, energy_per_successful_task, integrate_power_trace,
    };
    use serde_json::Value;

    const ENERGY_TRACE_TEXT: &str =
        include_str!("../../../docs/engineering/display-energy-trace-synthetic-v0.1.json");
    const ENERGY_SCHEMA_TEXT: &str =
        include_str!("../../../docs/engineering/display-energy-trace.schema.json");

    fn sample(time_seconds: f64, power_watts: f64) -> PowerSample {
        PowerSample {
            time_seconds,
            power_watts,
        }
    }

    fn config(start: f64, end: f64, max_gap: f64) -> EnergyIntegrationConfig {
        EnergyIntegrationConfig {
            window_start_seconds: start,
            window_end_seconds: end,
            max_sample_gap_seconds: max_gap,
            accounting_boundary_id: "MONITOR-AC-WALL-001".to_owned(),
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual} expected {expected} within {tolerance}"
        );
    }

    #[test]
    fn checked_in_synthetic_energy_trace_reports_energy_and_success_normalization() {
        let report = super::validate_energy_trace_str(ENERGY_TRACE_TEXT, ENERGY_SCHEMA_TEXT)
            .expect("checked-in synthetic trace should validate");
        assert_eq!(report.trace_id, "ODRH-ENERGY-TRACE-SYNTHETIC-001");
        close(report.integration.energy_joules, 35.0, 1e-12);
        assert_eq!(report.energy_per_successful_task_joules, Some(17.5));
        assert_eq!(
            report.to_json()["status"].as_str(),
            Some("SYNTHETIC_ARITHMETIC_VALID_UNQUALIFIED")
        );
        assert_eq!(
            report.to_json()["claim_ceiling"].as_str(),
            Some("Synthetic power-trace arithmetic only. No wattmeter, calibration, power boundary suitability, IEC compliance, task utility, or physical display result is qualified.")
        );
    }

    #[test]
    fn synthetic_trace_rejects_a_schedule_gap_above_its_declared_cap() {
        let mut trace: Value =
            serde_json::from_str(ENERGY_TRACE_TEXT).expect("energy trace fixture must parse");
        trace["max_sample_gap_seconds"] = serde_json::json!(0.5);
        let mutated = serde_json::to_string(&trace).expect("mutated trace must serialize");
        let errors = super::validate_energy_trace_str(&mutated, ENERGY_SCHEMA_TEXT)
            .expect_err("one-second samples must fail a half-second cap");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("sample gap exceeds"))
        );
    }

    #[test]
    fn synthetic_trace_refuses_physical_record_promotion() {
        let mut trace: Value =
            serde_json::from_str(ENERGY_TRACE_TEXT).expect("energy trace fixture must parse");
        trace["record_state"] = Value::String("ACQUIRED_UNQUALIFIED".to_owned());
        let mutated = serde_json::to_string(&trace).expect("mutated trace must serialize");
        assert!(
            super::validate_energy_trace_str(&mutated, ENERGY_SCHEMA_TEXT)
                .expect_err("synthetic validator cannot qualify physical data")
                .iter()
                .any(|error| error.contains("SYNTHETIC_TEST_ONLY"))
        );
    }

    #[test]
    fn known_answer_trace_integrates_trapezoidally_in_joules() {
        let trace = [sample(0.0, 10.0), sample(1.0, 20.0), sample(2.0, 20.0)];
        let report = integrate_power_trace(&trace, &config(0.0, 2.0, 1.0))
            .expect("known-answer trace should satisfy its schedule");
        close(report.energy_joules, 35.0, 1e-12);
        assert_eq!(report.raw_sample_count, 3);
        assert_eq!(report.integration_point_count, 3);
        assert_eq!(report.accounting_boundary_id, "MONITOR-AC-WALL-001");
        assert_eq!(
            report.integration_method,
            "TRAPEZOIDAL_WITH_LINEAR_BOUNDARY_INTERPOLATION"
        );
        assert_eq!(
            report.claim_ceiling,
            "ARITHMETIC_ONLY_NOT_PHYSICAL_POWER_QUALIFICATION"
        );
    }

    #[test]
    fn task_window_boundaries_are_interpolated_only_inside_covered_trace() {
        let trace = [sample(0.0, 10.0), sample(1.0, 20.0), sample(2.0, 20.0)];
        let report = integrate_power_trace(&trace, &config(0.5, 1.5, 1.0))
            .expect("window boundaries are covered by adjacent samples");
        close(report.energy_joules, 18.75, 1e-12);
        close(report.window_duration_seconds, 1.0, 1e-12);
        assert_eq!(report.integration_point_count, 3);
    }

    #[test]
    fn sample_gap_over_cap_is_rejected_not_bridged() {
        let trace = [sample(0.0, 10.0), sample(2.0, 10.0)];
        let error = integrate_power_trace(&trace, &config(0.0, 2.0, 1.0))
            .expect_err("two-second gap must fail a one-second limit");
        assert!(error.iter().any(|item| item.contains("sample gap exceeds")));
    }

    #[test]
    fn trace_must_cover_both_window_boundaries() {
        let trace = [sample(1.0, 10.0), sample(2.0, 10.0)];
        let error = integrate_power_trace(&trace, &config(0.0, 2.0, 1.0))
            .expect_err("cannot extrapolate power before first sample");
        assert!(error.iter().any(|item| item.contains("does not cover")));
    }

    #[test]
    fn invalid_power_and_nonmonotonic_time_are_rejected() {
        let negative = [sample(0.0, 10.0), sample(1.0, -0.1)];
        assert!(integrate_power_trace(&negative, &config(0.0, 1.0, 1.0)).is_err());

        let non_finite = [sample(0.0, 10.0), sample(1.0, f64::NAN)];
        assert!(integrate_power_trace(&non_finite, &config(0.0, 1.0, 1.0)).is_err());

        let unordered = [sample(0.0, 10.0), sample(0.0, 20.0)];
        assert!(integrate_power_trace(&unordered, &config(0.0, 1.0, 1.0)).is_err());
    }

    #[test]
    fn zero_successes_produce_no_efficiency_number() {
        assert_eq!(energy_per_successful_task(35.0, 3, 0), Ok(None));
        assert_eq!(energy_per_successful_task(35.0, 3, 1), Ok(Some(35.0)));
        assert_eq!(energy_per_successful_task(35.0, 3, 2), Ok(Some(17.5)));
    }

    #[test]
    fn impossible_task_counts_and_energy_values_are_rejected() {
        assert!(energy_per_successful_task(1.0, 1, 2).is_err());
        assert!(energy_per_successful_task(-1.0, 1, 1).is_err());
        assert!(energy_per_successful_task(f64::INFINITY, 1, 1).is_err());
    }
}
