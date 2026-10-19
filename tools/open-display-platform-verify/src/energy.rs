// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed integration of a raw power trace over an explicitly bounded task window.
//!
//! This arithmetic core does not validate a real wattmeter, task definition, power
//! boundary, calibration, IEC procedure, or physical display. The caller must bind
//! its result to those identities in the measurement-run record.

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
    if !config.max_sample_gap_seconds.is_finite()
        || config.max_sample_gap_seconds <= 0.0
    {
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
