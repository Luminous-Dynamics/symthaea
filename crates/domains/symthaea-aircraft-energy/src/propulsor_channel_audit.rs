// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Cross-channel consistency diagnostics for propulsor power data.
//!
//! This module checks whether reported shaft power agrees with torque × angular
//! speed and, only when explicitly declared comparable, whether electrical
//! input power falls below reported shaft output beyond a declared tolerance.
//! It never edits source values or upgrades data to physically validated status.
//!
//! Tolerances are caller-declared audit thresholds. They are not measurement
//! uncertainty, confidence intervals, calibration evidence, or proof of accuracy.
//! The caller must define them before inspecting the result and must preserve the
//! source channels, acquisition boundaries, and their uncertainty separately.

/// Whether electrical-input and shaft-output measurements share a comparable
/// motoring-system boundary for this operating point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElectricalShaftBoundary {
    /// Both values describe the same powered motor/propulsor chain and instant,
    /// with input measured upstream and shaft output downstream.
    SameMotoringChain,
    /// The measurements refer to different or incompatible boundaries.
    DifferentOrNonComparable,
    /// The source does not establish whether the boundaries are comparable.
    Unknown,
}

/// Outcome of the torque/RPM-to-shaft-power consistency check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaftPowerConsistency {
    WithinDeclaredTolerance,
    ExceedsDeclaredTolerance,
}

/// Outcome of the electrical-input versus shaft-output check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElectricalPowerCheck {
    /// Input power is not below shaft output by more than the declared tolerance.
    /// This is only a one-sided consistency check, not proof that the system is
    /// efficient, calibrated, or physically valid.
    InputNotBelowShaftBeyondTolerance,
    /// Comparable input power is below reported shaft output beyond tolerance.
    InputBelowShaftBeyondTolerance,
    /// No comparison was made because the boundaries are not comparable.
    NotEvaluatedDifferentBoundary,
    /// No comparison was made because the boundary relationship is unknown.
    NotEvaluatedUnknownBoundary,
}

/// Inputs from one identified propulsor operating point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropulsorPowerAuditInput {
    /// Measured shaft rotational speed, revolutions per minute.
    pub rpm: f64,
    /// Reported propulsor shaft torque, N·m.
    pub torque_nm: f64,
    /// Reported shaft power, W. It may be measured or derived; preserve that
    /// channel-origin distinction in the caller's provenance record.
    pub reported_shaft_power_w: f64,
    /// Reported electrical input power, W.
    pub electrical_input_power_w: f64,
    /// Preregistered absolute tolerance for power-channel comparisons, W.
    pub absolute_tolerance_w: f64,
    /// Preregistered relative tolerance fraction (e.g. 0.02 for 2%).
    pub relative_tolerance_fraction: f64,
    /// Explicit relationship between electrical and shaft measurement boundaries.
    pub electrical_shaft_boundary: ElectricalShaftBoundary,
}

/// A diagnostic result. Findings indicate inconsistencies; they do not repair
/// source data or make it eligible for an aircraft-operational decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropulsorPowerAudit {
    /// Shaft power calculated independently from torque and rotational speed.
    pub shaft_power_from_torque_rpm_w: f64,
    /// Signed residual: reported shaft power minus torque × angular speed, W.
    pub shaft_power_residual_w: f64,
    /// Tolerance used for the shaft-power residual, W.
    pub shaft_power_tolerance_w: f64,
    pub shaft_power_consistency: ShaftPowerConsistency,
    /// Signed margin: electrical input minus reported shaft output, W.
    pub electrical_minus_shaft_power_w: f64,
    /// One-sided cross-boundary diagnostic, where applicable.
    pub electrical_power_check: ElectricalPowerCheck,
    /// Reported shaft/electrical power ratio, only supplied when the input and
    /// output boundaries were explicitly declared comparable and input > 0.
    /// A ratio > 1 is retained as a diagnostic rather than silently clamped.
    pub implied_shaft_to_electrical_power_ratio: Option<f64>,
}

/// Invalid audit inputs or arithmetic that cannot be represented safely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropulsorPowerAuditError {
    NonFiniteInput { field: &'static str },
    NonPositiveInput { field: &'static str },
    NegativeInput { field: &'static str },
    DerivedValueOutOfRange,
}

/// Check cross-channel power arithmetic without treating the check as physical
/// validation. Source measurements and tolerances remain caller-owned evidence.
///
/// The shaft-power oracle uses:
///
///     omega_rad_s = rpm × 2π / 60
///     P_shaft_W = torque_Nm × omega_rad_s
///
/// Electrical input is compared with shaft output only when the caller explicitly
/// asserts a comparable motoring-chain boundary. This function does not infer
/// missing channels, apply a motor-efficiency default, estimate uncertainty, or
/// accept/reject a complete performance map by itself.
pub fn audit_propulsor_power_channels(
    input: PropulsorPowerAuditInput,
) -> Result<PropulsorPowerAudit, PropulsorPowerAuditError> {
    for (field, value) in [
        ("rpm", input.rpm),
        ("torque_nm", input.torque_nm),
        ("reported_shaft_power_w", input.reported_shaft_power_w),
        ("electrical_input_power_w", input.electrical_input_power_w),
        ("absolute_tolerance_w", input.absolute_tolerance_w),
        ("relative_tolerance_fraction", input.relative_tolerance_fraction),
    ] {
        if !value.is_finite() {
            return Err(PropulsorPowerAuditError::NonFiniteInput { field });
        }
    }

    if input.rpm <= 0.0 {
        return Err(PropulsorPowerAuditError::NonPositiveInput { field: "rpm" });
    }

    for (field, value) in [
        ("torque_nm", input.torque_nm),
        ("reported_shaft_power_w", input.reported_shaft_power_w),
        ("electrical_input_power_w", input.electrical_input_power_w),
        ("absolute_tolerance_w", input.absolute_tolerance_w),
        ("relative_tolerance_fraction", input.relative_tolerance_fraction),
    ] {
        if value < 0.0 {
            return Err(PropulsorPowerAuditError::NegativeInput { field });
        }
    }

    let angular_speed_rad_s = input.rpm * std::f64::consts::TAU / 60.0;
    let shaft_power_from_torque_rpm_w = input.torque_nm * angular_speed_rad_s;
    if !angular_speed_rad_s.is_finite() || !shaft_power_from_torque_rpm_w.is_finite() {
        return Err(PropulsorPowerAuditError::DerivedValueOutOfRange);
    }

    let shaft_power_tolerance_w = tolerance(
        input.absolute_tolerance_w,
        input.relative_tolerance_fraction,
        shaft_power_from_torque_rpm_w.max(input.reported_shaft_power_w),
    )?;
    let shaft_power_residual_w =
        input.reported_shaft_power_w - shaft_power_from_torque_rpm_w;
    if !shaft_power_residual_w.is_finite() {
        return Err(PropulsorPowerAuditError::DerivedValueOutOfRange);
    }
    let shaft_power_consistency = if shaft_power_residual_w.abs() <= shaft_power_tolerance_w {
        ShaftPowerConsistency::WithinDeclaredTolerance
    } else {
        ShaftPowerConsistency::ExceedsDeclaredTolerance
    };

    let electrical_minus_shaft_power_w =
        input.electrical_input_power_w - input.reported_shaft_power_w;
    if !electrical_minus_shaft_power_w.is_finite() {
        return Err(PropulsorPowerAuditError::DerivedValueOutOfRange);
    }

    let (electrical_power_check, implied_shaft_to_electrical_power_ratio) =
        match input.electrical_shaft_boundary {
            ElectricalShaftBoundary::SameMotoringChain => {
                let chain_tolerance_w = tolerance(
                    input.absolute_tolerance_w,
                    input.relative_tolerance_fraction,
                    input
                        .electrical_input_power_w
                        .max(input.reported_shaft_power_w),
                )?;
                let check = if input.reported_shaft_power_w
                    - input.electrical_input_power_w
                    > chain_tolerance_w
                {
                    ElectricalPowerCheck::InputBelowShaftBeyondTolerance
                } else {
                    ElectricalPowerCheck::InputNotBelowShaftBeyondTolerance
                };
                let ratio = if input.electrical_input_power_w > 0.0 {
                    let value =
                        input.reported_shaft_power_w / input.electrical_input_power_w;
                    if !value.is_finite() {
                        return Err(PropulsorPowerAuditError::DerivedValueOutOfRange);
                    }
                    Some(value)
                } else {
                    None
                };
                (check, ratio)
            }
            ElectricalShaftBoundary::DifferentOrNonComparable => (
                ElectricalPowerCheck::NotEvaluatedDifferentBoundary,
                None,
            ),
            ElectricalShaftBoundary::Unknown => (
                ElectricalPowerCheck::NotEvaluatedUnknownBoundary,
                None,
            ),
        };

    Ok(PropulsorPowerAudit {
        shaft_power_from_torque_rpm_w,
        shaft_power_residual_w,
        shaft_power_tolerance_w,
        shaft_power_consistency,
        electrical_minus_shaft_power_w,
        electrical_power_check,
        implied_shaft_to_electrical_power_ratio,
    })
}

fn tolerance(
    absolute_tolerance_w: f64,
    relative_tolerance_fraction: f64,
    magnitude_w: f64,
) -> Result<f64, PropulsorPowerAuditError> {
    let value = absolute_tolerance_w + relative_tolerance_fraction * magnitude_w;
    if !value.is_finite() || value < 0.0 {
        return Err(PropulsorPowerAuditError::DerivedValueOutOfRange);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> PropulsorPowerAuditInput {
        PropulsorPowerAuditInput {
            rpm: 3_000.0,
            torque_nm: 2.0,
            reported_shaft_power_w: 200.0 * std::f64::consts::PI,
            electrical_input_power_w: 700.0,
            absolute_tolerance_w: 0.01,
            relative_tolerance_fraction: 1.0e-6,
            electrical_shaft_boundary: ElectricalShaftBoundary::SameMotoringChain,
        }
    }

    #[test]
    fn torque_rpm_oracle_matches_known_answer() {
        let result = audit_propulsor_power_channels(input()).unwrap();
        // At 3,000 rpm, omega = 100π rad/s; at 2 N·m, P = 200π W.
        assert!((result.shaft_power_from_torque_rpm_w - 200.0 * std::f64::consts::PI).abs() < 1e-10);
        assert_eq!(result.shaft_power_consistency, ShaftPowerConsistency::WithinDeclaredTolerance);
    }

    #[test]
    fn material_shaft_power_disagreement_is_reported_not_rewritten() {
        let mut values = input();
        values.reported_shaft_power_w = 500.0;
        let result = audit_propulsor_power_channels(values).unwrap();
        assert_eq!(result.shaft_power_consistency, ShaftPowerConsistency::ExceedsDeclaredTolerance);
        assert!(result.shaft_power_residual_w < 0.0);
        assert_eq!(values.reported_shaft_power_w, 500.0);
    }

    #[test]
    fn input_power_below_shaft_output_is_flagged_for_comparable_motor_chain() {
        let mut values = input();
        values.electrical_input_power_w = 600.0;
        let result = audit_propulsor_power_channels(values).unwrap();
        assert_eq!(result.electrical_power_check, ElectricalPowerCheck::InputBelowShaftBeyondTolerance);
        assert!(result.implied_shaft_to_electrical_power_ratio.unwrap() > 1.0);
    }

    #[test]
    fn incomparable_boundaries_are_not_judged_by_motor_chain_rule() {
        let mut values = input();
        values.electrical_input_power_w = 1.0;
        values.electrical_shaft_boundary = ElectricalShaftBoundary::DifferentOrNonComparable;
        let result = audit_propulsor_power_channels(values).unwrap();
        assert_eq!(result.electrical_power_check, ElectricalPowerCheck::NotEvaluatedDifferentBoundary);
        assert_eq!(result.implied_shaft_to_electrical_power_ratio, None);
    }

    #[test]
    fn unknown_boundaries_are_not_silently_assumed_comparable() {
        let mut values = input();
        values.electrical_shaft_boundary = ElectricalShaftBoundary::Unknown;
        let result = audit_propulsor_power_channels(values).unwrap();
        assert_eq!(result.electrical_power_check, ElectricalPowerCheck::NotEvaluatedUnknownBoundary);
        assert_eq!(result.implied_shaft_to_electrical_power_ratio, None);
    }

    #[test]
    fn small_shaft_residual_can_pass_only_the_declared_tolerance() {
        let mut values = input();
        values.reported_shaft_power_w += 0.005;
        let result = audit_propulsor_power_channels(values).unwrap();
        assert_eq!(result.shaft_power_consistency, ShaftPowerConsistency::WithinDeclaredTolerance);
    }

    #[test]
    fn non_finite_and_negative_channels_fail_closed() {
        let mut values = input();
        values.rpm = f64::NAN;
        assert_eq!(
            audit_propulsor_power_channels(values),
            Err(PropulsorPowerAuditError::NonFiniteInput { field: "rpm" })
        );

        let mut values = input();
        values.torque_nm = -1.0;
        assert_eq!(
            audit_propulsor_power_channels(values),
            Err(PropulsorPowerAuditError::NegativeInput { field: "torque_nm" })
        );
    }

    #[test]
    fn non_positive_rpm_and_invalid_tolerances_fail_closed() {
        let mut values = input();
        values.rpm = 0.0;
        assert_eq!(
            audit_propulsor_power_channels(values),
            Err(PropulsorPowerAuditError::NonPositiveInput { field: "rpm" })
        );

        let mut values = input();
        values.absolute_tolerance_w = -0.1;
        assert_eq!(
            audit_propulsor_power_channels(values),
            Err(PropulsorPowerAuditError::NegativeInput { field: "absolute_tolerance_w" })
        );
    }

    #[test]
    fn zero_power_channels_are_supported_when_the_operating_point_is_valid() {
        let mut values = input();
        values.torque_nm = 0.0;
        values.reported_shaft_power_w = 0.0;
        values.electrical_input_power_w = 0.0;
        let result = audit_propulsor_power_channels(values).unwrap();
        assert_eq!(result.shaft_power_consistency, ShaftPowerConsistency::WithinDeclaredTolerance);
        assert_eq!(result.implied_shaft_to_electrical_power_ratio, None);
    }

    #[test]
    fn an_overflowed_tolerance_is_not_returned_as_a_valid_diagnostic() {
        let mut values = input();
        values.relative_tolerance_fraction = f64::MAX;
        assert_eq!(
            audit_propulsor_power_channels(values),
            Err(PropulsorPowerAuditError::DerivedValueOutOfRange)
        );
    }
}
