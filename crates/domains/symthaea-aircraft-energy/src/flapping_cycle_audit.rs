// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evidence-preserving energy audit for a measured flapping-wing cycle.
//!
//! This is a telemetry accounting contract, not a wing-aerodynamics or
//! ornithopter-performance model. It integrates explicitly sampled mechanical
//! joint power (torque × angular velocity) and independently sampled electrical
//! bus power. It never equates negative mechanical work with energy recovered
//! electrically: negative joint power can be dissipated, stored elastically,
//! or returned through a regenerative drive, and these cases require different
//! measurements.
//!
//! Between adjacent samples, each *power channel* is assumed piecewise-linear;
//! trapezoidal integration exactly integrates that declared representation.
//! The code does not infer what happened between samples, correct sensor lag,
//! compensate drivetrain loss, or estimate measurement uncertainty.
//!
//! Sign conventions identified by trace.sign_convention_id:
//! - joint torque × joint angular velocity > 0: mechanical power delivered
//!   by the actuator into the wing mechanism;
//! - joint torque × joint angular velocity < 0: mechanical power flowing back
//!   from the wing mechanism toward the joint actuator;
//! - electrical bus power > 0: power drawn from the measured bus;
//! - electrical bus power < 0: power returned to the measured bus.
//!
//! A caller must assert that samples form a complete, phase-consistent cycle.
//! The function can check that the declaration is present, but cannot prove
//! phase closure or independently authenticate the source artifact.

/// One synchronized sample from a flapping mechanism.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlappingCycleSample {
    /// Sample time in seconds on one monotonic time base.
    pub time_s: f64,
    /// Signed drive/joint torque, N·m, under the declared sign convention.
    pub joint_torque_nm: f64,
    /// Signed joint angular velocity, rad/s, under the declared sign convention.
    pub joint_angular_velocity_rad_s: f64,
    /// Signed electrical power at a named bus measurement boundary, W.
    /// Positive means drawn from the bus; negative means returned to the bus.
    pub electrical_bus_power_w: f64,
}

/// Trace and provenance for one complete wingbeat or declared multi-wingbeat cycle.
#[derive(Debug, Clone, PartialEq)]
pub struct FlappingCycleTrace {
    pub cycle_id: String,
    /// Immutable source revision or acquisition-run identifier.
    pub source_revision: String,
    /// SHA-256 of the source artifact as declared by the caller. This module
    /// checks syntax only; it does not resolve or authenticate the source bytes.
    pub source_sha256: String,
    /// Names the measured torque, velocity and electrical-power sign conventions
    /// and the electrical bus boundary. It must be specific and non-empty.
    pub sign_convention_id: String,
    /// Caller attests that the window starts/ends at phase-consistent cycle
    /// boundaries. False is rejected; the declaration is not independently proved.
    pub cycle_complete: bool,
    pub samples: Vec<FlappingCycleSample>,
}

/// Diagnostics from one complete trace. All energy quantities are joules.
#[derive(Debug, Clone, PartialEq)]
pub struct FlappingCycleEnergyAudit {
    pub cycle_id: String,
    pub source_revision: String,
    pub source_sha256: String,
    pub sign_convention_id: String,
    pub sample_count: usize,
    pub duration_s: f64,
    /// Integral of positive joint power; mechanical work delivered toward the wing.
    pub positive_mechanical_work_j: f64,
    /// Magnitude of the integral of negative joint power; not necessarily recovered.
    pub negative_mechanical_work_magnitude_j: f64,
    /// Positive work minus negative-work magnitude under the piecewise-linear
    /// power interpolation; not electrical energy consumption.
    pub net_mechanical_work_j: f64,
    /// Integral of positive electrical bus power.
    pub electrical_energy_drawn_from_bus_j: f64,
    /// Magnitude of the integral of negative electrical bus power.
    pub electrical_energy_returned_to_bus_j: f64,
    /// Drawn bus energy minus energy returned to the same declared bus.
    pub net_electrical_energy_from_bus_j: f64,
}

/// Fail-closed errors for incomplete or unrepresentable telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlappingCycleAuditError {
    EmptyField { field: &'static str },
    InvalidSha256,
    IncompleteCycle,
    TooFewSamples { actual: usize },
    NonFiniteSample { index: usize, field: &'static str },
    NonIncreasingTime { index: usize },
    DerivedValueOutOfRange { field: &'static str },
}

/// Audit one declared complete cycle. This preserves signed mechanical/electrical
/// energy separately; it does not calculate aerodynamic efficiency, battery
/// efficiency, elastic storage efficiency, actuator efficiency, or flight endurance.
pub fn audit_flapping_cycle(
    trace: FlappingCycleTrace,
) -> Result<FlappingCycleEnergyAudit, FlappingCycleAuditError> {
    for (field, value) in [
        ("cycle_id", trace.cycle_id.as_str()),
        ("source_revision", trace.source_revision.as_str()),
        ("sign_convention_id", trace.sign_convention_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(FlappingCycleAuditError::EmptyField { field });
        }
    }

    if trace.source_sha256.len() != 64
        || !trace.source_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(FlappingCycleAuditError::InvalidSha256);
    }
    if !trace.cycle_complete {
        return Err(FlappingCycleAuditError::IncompleteCycle);
    }
    if trace.samples.len() < 2 {
        return Err(FlappingCycleAuditError::TooFewSamples {
            actual: trace.samples.len(),
        });
    }

    for (index, sample) in trace.samples.iter().enumerate() {
        for (field, value) in [
            ("time_s", sample.time_s),
            ("joint_torque_nm", sample.joint_torque_nm),
            ("joint_angular_velocity_rad_s", sample.joint_angular_velocity_rad_s),
            ("electrical_bus_power_w", sample.electrical_bus_power_w),
        ] {
            if !value.is_finite() {
                return Err(FlappingCycleAuditError::NonFiniteSample { index, field });
            }
        }
        if index > 0 && sample.time_s <= trace.samples[index - 1].time_s {
            return Err(FlappingCycleAuditError::NonIncreasingTime { index });
        }
        let mechanical_power = sample.joint_torque_nm * sample.joint_angular_velocity_rad_s;
        if !mechanical_power.is_finite() {
            return Err(FlappingCycleAuditError::DerivedValueOutOfRange {
                field: "mechanical_power_w",
            });
        }
    }

    let first_time = trace.samples[0].time_s;
    let last_time = trace.samples[trace.samples.len() - 1].time_s;
    let duration_s = last_time - first_time;
    if !duration_s.is_finite() || duration_s <= 0.0 {
        return Err(FlappingCycleAuditError::DerivedValueOutOfRange {
            field: "duration_s",
        });
    }

    let mut positive_mechanical_work_j = 0.0;
    let mut negative_mechanical_work_magnitude_j = 0.0;
    let mut electrical_energy_drawn_from_bus_j = 0.0;
    let mut electrical_energy_returned_to_bus_j = 0.0;

    for index in 0..trace.samples.len() - 1 {
        let left = trace.samples[index];
        let right = trace.samples[index + 1];
        let dt = right.time_s - left.time_s;
        let left_mechanical_power = left.joint_torque_nm * left.joint_angular_velocity_rad_s;
        let right_mechanical_power = right.joint_torque_nm * right.joint_angular_velocity_rad_s;

        let (mechanical_positive, mechanical_negative) =
            integrate_signed_linear(left_mechanical_power, right_mechanical_power, dt)?;
        let (electrical_positive, electrical_negative) =
            integrate_signed_linear(left.electrical_bus_power_w, right.electrical_bus_power_w, dt)?;

        positive_mechanical_work_j += mechanical_positive;
        negative_mechanical_work_magnitude_j += mechanical_negative;
        electrical_energy_drawn_from_bus_j += electrical_positive;
        electrical_energy_returned_to_bus_j += electrical_negative;

        for (field, value) in [
            ("positive_mechanical_work_j", positive_mechanical_work_j),
            ("negative_mechanical_work_magnitude_j", negative_mechanical_work_magnitude_j),
            ("electrical_energy_drawn_from_bus_j", electrical_energy_drawn_from_bus_j),
            ("electrical_energy_returned_to_bus_j", electrical_energy_returned_to_bus_j),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(FlappingCycleAuditError::DerivedValueOutOfRange { field });
            }
        }
    }

    let net_mechanical_work_j =
        positive_mechanical_work_j - negative_mechanical_work_magnitude_j;
    let net_electrical_energy_from_bus_j =
        electrical_energy_drawn_from_bus_j - electrical_energy_returned_to_bus_j;

    for (field, value) in [
        ("net_mechanical_work_j", net_mechanical_work_j),
        ("net_electrical_energy_from_bus_j", net_electrical_energy_from_bus_j),
    ] {
        if !value.is_finite() {
            return Err(FlappingCycleAuditError::DerivedValueOutOfRange { field });
        }
    }

    Ok(FlappingCycleEnergyAudit {
        cycle_id: trace.cycle_id,
        source_revision: trace.source_revision,
        source_sha256: trace.source_sha256,
        sign_convention_id: trace.sign_convention_id,
        sample_count: trace.samples.len(),
        duration_s,
        positive_mechanical_work_j,
        negative_mechanical_work_magnitude_j,
        net_mechanical_work_j,
        electrical_energy_drawn_from_bus_j,
        electrical_energy_returned_to_bus_j,
        net_electrical_energy_from_bus_j,
    })
}

/// Integrate a piecewise-linear power channel over one positive-duration interval.
/// Returns (positive energy, magnitude of negative energy), both in joules.
/// A sign-changing interval is split at its linear zero crossing, avoiding the
/// error of assigning its whole trapezoid to one sign.
fn integrate_signed_linear(
    p0: f64,
    p1: f64,
    dt: f64,
) -> Result<(f64, f64), FlappingCycleAuditError> {
    if !p0.is_finite() || !p1.is_finite() || !dt.is_finite() || dt <= 0.0 {
        return Err(FlappingCycleAuditError::DerivedValueOutOfRange {
            field: "integration_interval",
        });
    }

    let result = if p0 >= 0.0 && p1 >= 0.0 {
        (0.5 * (p0 + p1) * dt, 0.0)
    } else if p0 <= 0.0 && p1 <= 0.0 {
        (0.0, 0.5 * (-p0 - p1) * dt)
    } else if p0 > 0.0 {
        let positive_duration = dt * (p0 / (p0 - p1));
        (
            0.5 * p0 * positive_duration,
            0.5 * (-p1) * (dt - positive_duration),
        )
    } else {
        let negative_duration = dt * ((-p0) / (p1 - p0));
        (
            0.5 * p1 * (dt - negative_duration),
            0.5 * (-p0) * negative_duration,
        )
    };

    if !result.0.is_finite()
        || !result.1.is_finite()
        || result.0 < 0.0
        || result.1 < 0.0
    {
        return Err(FlappingCycleAuditError::DerivedValueOutOfRange {
            field: "integrated_energy_j",
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(time_s: f64, torque_nm: f64, omega_rad_s: f64, electrical_w: f64) -> FlappingCycleSample {
        FlappingCycleSample {
            time_s,
            joint_torque_nm: torque_nm,
            joint_angular_velocity_rad_s: omega_rad_s,
            electrical_bus_power_w: electrical_w,
        }
    }

    fn trace(samples: Vec<FlappingCycleSample>) -> FlappingCycleTrace {
        FlappingCycleTrace {
            cycle_id: "wingbeat-0001".into(),
            source_revision: "bench-run-r1".into(),
            source_sha256: "a".repeat(64),
            sign_convention_id: "joint-power-into-wing_bus-power-drawn-positive_v1".into(),
            cycle_complete: true,
            samples,
        }
    }

    #[test]
    fn integrates_constant_positive_mechanical_and_electrical_power() {
        let result = audit_flapping_cycle(trace(vec![
            sample(0.0, 2.0, 10.0, 25.0),
            sample(1.0, 2.0, 10.0, 25.0),
        ])).unwrap();

        assert!((result.positive_mechanical_work_j - 20.0).abs() < 1e-12);
        assert_eq!(result.negative_mechanical_work_magnitude_j, 0.0);
        assert!((result.net_mechanical_work_j - 20.0).abs() < 1e-12);
        assert!((result.electrical_energy_drawn_from_bus_j - 25.0).abs() < 1e-12);
        assert_eq!(result.electrical_energy_returned_to_bus_j, 0.0);
        assert!((result.net_electrical_energy_from_bus_j - 25.0).abs() < 1e-12);
    }

    #[test]
    fn splits_sign_changing_mechanical_power_at_its_zero_crossing() {
        let values = trace(vec![
            sample(0.0, 1.0, 10.0, 0.0),
            sample(1.0, 1.0, -10.0, 0.0),
        ]);
        // The joint power samples are +10 W and -10 W.
        let result = audit_flapping_cycle(values).unwrap();
        assert!((result.positive_mechanical_work_j - 2.5).abs() < 1e-12);
        assert!((result.negative_mechanical_work_magnitude_j - 2.5).abs() < 1e-12);
        assert!(result.net_mechanical_work_j.abs() < 1e-12);
    }

    #[test]
    fn electrical_regeneration_is_not_conflated_with_mechanical_return() {
        let result = audit_flapping_cycle(trace(vec![
            sample(0.0, 1.0, 10.0, 10.0),
            sample(1.0, 1.0, -10.0, -10.0),
        ])).unwrap();

        assert!(result.negative_mechanical_work_magnitude_j > 0.0);
        assert!(result.electrical_energy_returned_to_bus_j > 0.0);
        assert!((result.positive_mechanical_work_j - 2.5).abs() < 1e-12);
        assert!((result.electrical_energy_drawn_from_bus_j - 5.0).abs() < 1e-12);
        assert!((result.electrical_energy_returned_to_bus_j - 2.5).abs() < 1e-12);
    }

    #[test]
    fn all_negative_power_reports_returned_magnitudes_and_negative_net() {
        let result = audit_flapping_cycle(trace(vec![
            sample(0.0, 1.0, -10.0, -5.0),
            sample(2.0, 1.0, -10.0, -5.0),
        ])).unwrap();

        assert_eq!(result.positive_mechanical_work_j, 0.0);
        assert!((result.negative_mechanical_work_magnitude_j - 20.0).abs() < 1e-12);
        assert!((result.net_mechanical_work_j + 20.0).abs() < 1e-12);
        assert!((result.electrical_energy_returned_to_bus_j - 10.0).abs() < 1e-12);
        assert!((result.net_electrical_energy_from_bus_j + 10.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_incomplete_cycles_and_too_few_samples() {
        let mut incomplete = trace(vec![sample(0.0, 1.0, 1.0, 1.0), sample(1.0, 1.0, 1.0, 1.0)]);
        incomplete.cycle_complete = false;
        assert_eq!(audit_flapping_cycle(incomplete), Err(FlappingCycleAuditError::IncompleteCycle));
        assert_eq!(
            audit_flapping_cycle(trace(vec![sample(0.0, 1.0, 1.0, 1.0)])),
            Err(FlappingCycleAuditError::TooFewSamples { actual: 1 })
        );
    }

    #[test]
    fn rejects_bad_provenance_and_empty_identity_fields() {
        let mut bad_hash = trace(vec![sample(0.0, 1.0, 1.0, 1.0), sample(1.0, 1.0, 1.0, 1.0)]);
        bad_hash.source_sha256 = "abc".into();
        assert_eq!(audit_flapping_cycle(bad_hash), Err(FlappingCycleAuditError::InvalidSha256));

        let mut empty = trace(vec![sample(0.0, 1.0, 1.0, 1.0), sample(1.0, 1.0, 1.0, 1.0)]);
        empty.sign_convention_id = "  ".into();
        assert_eq!(
            audit_flapping_cycle(empty),
            Err(FlappingCycleAuditError::EmptyField { field: "sign_convention_id" })
        );
    }

    #[test]
    fn rejects_non_finite_channels_and_non_increasing_time() {
        assert_eq!(
            audit_flapping_cycle(trace(vec![
                sample(0.0, 1.0, 1.0, 1.0),
                sample(f64::NAN, 1.0, 1.0, 1.0),
            ])),
            Err(FlappingCycleAuditError::NonFiniteSample { index: 1, field: "time_s" })
        );
        assert_eq!(
            audit_flapping_cycle(trace(vec![
                sample(0.0, 1.0, 1.0, 1.0),
                sample(0.0, 1.0, 1.0, 1.0),
            ])),
            Err(FlappingCycleAuditError::NonIncreasingTime { index: 1 })
        );
    }

    #[test]
    fn rejects_overflowed_mechanical_power() {
        assert_eq!(
            audit_flapping_cycle(trace(vec![
                sample(0.0, f64::MAX, 2.0, 0.0),
                sample(1.0, f64::MAX, 2.0, 0.0),
            ])),
            Err(FlappingCycleAuditError::DerivedValueOutOfRange { field: "mechanical_power_w" })
        );
    }

    #[test]
    fn audit_preserves_source_identity_and_sample_count() {
        let result = audit_flapping_cycle(trace(vec![
            sample(2.0, 1.0, 1.0, 1.0),
            sample(4.0, 1.0, 1.0, 1.0),
            sample(5.0, 1.0, 1.0, 1.0),
        ])).unwrap();
        assert_eq!(result.sample_count, 3);
        assert_eq!(result.duration_s, 3.0);
        assert_eq!(result.cycle_id, "wingbeat-0001");
        assert_eq!(result.source_revision, "bench-run-r1");
        assert_eq!(result.sign_convention_id, "joint-power-into-wing_bus-power-drawn-positive_v1");
    }

    #[test]
    fn deterministic_output_for_identical_pinned_trace() {
        let input = trace(vec![
            sample(0.0, 1.0, 2.0, 3.0),
            sample(0.2, 1.5, 3.0, 4.0),
            sample(0.6, -1.0, 4.0, -2.0),
        ]);
        assert_eq!(audit_flapping_cycle(input.clone()), audit_flapping_cycle(input));
    }
}
