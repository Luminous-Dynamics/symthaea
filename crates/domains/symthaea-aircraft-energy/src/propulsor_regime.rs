// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evidence-aware operating-regime labels for signed propulsor measurements.
//!
//! Conventional propeller maps often concentrate on positive RPM, positive
//! shaft input power, and positive propulsive thrust. That is useful for normal
//! motoring operation but can hide other behaviors: reverse flow, reverse
//! rotation, negative thrust, or power extracted from a windmilling propeller.
//!
//! This module does not claim a new physical law. It defines an auditable
//! *regime vocabulary* so future data adapters and benchmarks can preserve
//! operating modes instead of dropping them into one positive-thrust map.
//! Every classification uses caller-supplied, preregistered deadbands; these
//! deadbands classify near-zero states and are not measurement uncertainties.
//!
//! Sign conventions (must match the source manifest):
//! - axial flow > 0: conventional forward/advance flow through the propulsor;
//! - RPM > 0: source-defined conventional rotation direction;
//! - thrust > 0: force in the declared vehicle-forward thrust direction;
//! - shaft power > 0: power supplied into the propulsor shaft (motoring);
//! - shaft power < 0: power extracted from the propulsor shaft (turbining).
//!
//! A reverse-flow or reverse-rotation label is descriptive, not a claim that
//! the point is physically safe, stable, efficient, or suitable for a mission.
//! The module does not calculate performance coefficients or approve hardware.

/// Signed operating point with explicit SI units and documented sign meaning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedPropulsorOperatingPoint {
    /// Axial flow speed, m/s; sign convention described at module level.
    pub axial_flow_m_s: f64,
    /// Signed revolutions per minute.
    pub rpm: f64,
    /// Signed thrust in newtons, positive in the declared vehicle-forward direction.
    pub thrust_n: f64,
    /// Signed shaft power in watts; positive means power flows into the propulsor.
    pub shaft_power_into_propulsor_w: f64,
}

/// Caller-declared deadbands used to distinguish near-zero measurements.
/// These are classification thresholds, not uncertainty estimates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropulsorRegimeDeadbands {
    pub axial_flow_m_s: f64,
    pub rpm: f64,
    pub thrust_n: f64,
    pub shaft_power_w: f64,
}

/// Direction/state of axial flow relative to the declared convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxialFlowState {
    NearStatic,
    ForwardAxialFlow,
    ReverseAxialFlow,
}

/// Direction/state of shaft rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationState {
    NearStopped,
    PositiveRotation,
    ReverseRotation,
}

/// Signed thrust state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrustState {
    NearZero,
    PositiveThrust,
    NegativeThrust,
}

/// Direction of shaft power flow under the declared sign convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaftPowerState {
    NearZero,
    PowerIntoShaft,
    PowerExtractedFromShaft,
}

/// A named operating archetype, not a physical qualification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropulsorOperatingRegime {
    /// RPM and shaft power are both within their explicit near-zero deadbands.
    StoppedRotor,
    /// Rotor is within the RPM deadband but reported shaft power is not.
    /// Investigate synchronization, units, sign conventions, or channel origins.
    InconsistentPowerWithStoppedRotor,
    /// Near-static axial flow, positive rotation, power into shaft, positive thrust.
    StaticPoweredThrust,
    /// Near-static axial flow with negative thrust while power is supplied.
    StaticPoweredNegativeThrust,
    /// Near-static flow, power supplied, and thrust within deadband.
    StaticPoweredLowThrust,
    /// Near-static flow and net shaft power extraction.
    StaticPowerExtraction,
    /// Near-static flow and nonzero rotation but shaft power near zero.
    StaticRotatingNearZeroPower,
    /// Forward axial flow, positive rotation, shaft power supplied, positive thrust.
    ForwardPoweredPropulsion,
    /// Forward axial flow with power supplied but negative thrust.
    ForwardPoweredNegativeThrust,
    /// Forward axial flow with power supplied but thrust within deadband.
    ForwardPoweredLowThrust,
    /// Forward axial flow and shaft power extracted from the rotating propulsor.
    ForwardFlowPowerExtraction,
    /// Forward axial flow, rotating propulsor, but shaft power within deadband.
    ForwardFlowUnpoweredRotation,
    /// Negative axial flow; the other signed states remain available in the report.
    ReverseFlowOperation,
    /// Reverse rotation without a higher-priority stopped or reverse-flow condition.
    ReverseRotationOperation,
    /// A valid combination that does not fit a named archetype above.
    CombinedOrUnclassified,
}

/// All classified axes are retained even when the operating point has an unusual
/// combination. Use these fields for filters/plots rather than relying on the
/// single archetype label alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropulsorRegimeClassification {
    pub flow: AxialFlowState,
    pub rotation: RotationState,
    pub thrust: ThrustState,
    pub shaft_power: ShaftPowerState,
    pub regime: PropulsorOperatingRegime,
}

/// Input validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropulsorRegimeError {
    NonFiniteInput { field: &'static str },
    NegativeDeadband { field: &'static str },
    NonFiniteDeadband { field: &'static str },
}

/// Classify one signed point using preregistered deadbands.
///
/// The decision order is intentionally documented: stopped rotor consistency
/// first, reverse flow second, reverse rotation third, then static and forward
/// motoring/power-extraction archetypes. All four signed axes are always
/// returned, so a priority label cannot erase a mixed-mode signature.
pub fn classify_propulsor_regime(
    point: SignedPropulsorOperatingPoint,
    deadbands: PropulsorRegimeDeadbands,
) -> Result<PropulsorRegimeClassification, PropulsorRegimeError> {
    for (field, value) in [
        ("axial_flow_m_s", point.axial_flow_m_s),
        ("rpm", point.rpm),
        ("thrust_n", point.thrust_n),
        ("shaft_power_into_propulsor_w", point.shaft_power_into_propulsor_w),
    ] {
        if !value.is_finite() {
            return Err(PropulsorRegimeError::NonFiniteInput { field });
        }
    }

    for (field, value) in [
        ("axial_flow_m_s", deadbands.axial_flow_m_s),
        ("rpm", deadbands.rpm),
        ("thrust_n", deadbands.thrust_n),
        ("shaft_power_w", deadbands.shaft_power_w),
    ] {
        if !value.is_finite() {
            return Err(PropulsorRegimeError::NonFiniteDeadband { field });
        }
        if value < 0.0 {
            return Err(PropulsorRegimeError::NegativeDeadband { field });
        }
    }

    let flow = if point.axial_flow_m_s.abs() <= deadbands.axial_flow_m_s {
        AxialFlowState::NearStatic
    } else if point.axial_flow_m_s > 0.0 {
        AxialFlowState::ForwardAxialFlow
    } else {
        AxialFlowState::ReverseAxialFlow
    };

    let rotation = if point.rpm.abs() <= deadbands.rpm {
        RotationState::NearStopped
    } else if point.rpm > 0.0 {
        RotationState::PositiveRotation
    } else {
        RotationState::ReverseRotation
    };

    let thrust = if point.thrust_n.abs() <= deadbands.thrust_n {
        ThrustState::NearZero
    } else if point.thrust_n > 0.0 {
        ThrustState::PositiveThrust
    } else {
        ThrustState::NegativeThrust
    };

    let shaft_power =
        if point.shaft_power_into_propulsor_w.abs() <= deadbands.shaft_power_w {
            ShaftPowerState::NearZero
        } else if point.shaft_power_into_propulsor_w > 0.0 {
            ShaftPowerState::PowerIntoShaft
        } else {
            ShaftPowerState::PowerExtractedFromShaft
        };

    let regime = if rotation == RotationState::NearStopped {
        if shaft_power == ShaftPowerState::NearZero {
            PropulsorOperatingRegime::StoppedRotor
        } else {
            PropulsorOperatingRegime::InconsistentPowerWithStoppedRotor
        }
    } else if flow == AxialFlowState::ReverseAxialFlow {
        PropulsorOperatingRegime::ReverseFlowOperation
    } else if rotation == RotationState::ReverseRotation {
        PropulsorOperatingRegime::ReverseRotationOperation
    } else {
        match (flow, shaft_power, thrust) {
            (
                AxialFlowState::NearStatic,
                ShaftPowerState::PowerIntoShaft,
                ThrustState::PositiveThrust,
            ) => PropulsorOperatingRegime::StaticPoweredThrust,
            (
                AxialFlowState::NearStatic,
                ShaftPowerState::PowerIntoShaft,
                ThrustState::NegativeThrust,
            ) => PropulsorOperatingRegime::StaticPoweredNegativeThrust,
            (AxialFlowState::NearStatic, ShaftPowerState::PowerIntoShaft, ThrustState::NearZero) => {
                PropulsorOperatingRegime::StaticPoweredLowThrust
            }
            (
                AxialFlowState::NearStatic,
                ShaftPowerState::PowerExtractedFromShaft,
                _,
            ) => PropulsorOperatingRegime::StaticPowerExtraction,
            (AxialFlowState::NearStatic, ShaftPowerState::NearZero, _) => {
                PropulsorOperatingRegime::StaticRotatingNearZeroPower
            }
            (
                AxialFlowState::ForwardAxialFlow,
                ShaftPowerState::PowerIntoShaft,
                ThrustState::PositiveThrust,
            ) => PropulsorOperatingRegime::ForwardPoweredPropulsion,
            (
                AxialFlowState::ForwardAxialFlow,
                ShaftPowerState::PowerIntoShaft,
                ThrustState::NegativeThrust,
            ) => PropulsorOperatingRegime::ForwardPoweredNegativeThrust,
            (
                AxialFlowState::ForwardAxialFlow,
                ShaftPowerState::PowerIntoShaft,
                ThrustState::NearZero,
            ) => PropulsorOperatingRegime::ForwardPoweredLowThrust,
            (
                AxialFlowState::ForwardAxialFlow,
                ShaftPowerState::PowerExtractedFromShaft,
                _,
            ) => PropulsorOperatingRegime::ForwardFlowPowerExtraction,
            (AxialFlowState::ForwardAxialFlow, ShaftPowerState::NearZero, _) => {
                PropulsorOperatingRegime::ForwardFlowUnpoweredRotation
            }
            _ => PropulsorOperatingRegime::CombinedOrUnclassified,
        }
    };

    Ok(PropulsorRegimeClassification {
        flow,
        rotation,
        thrust,
        shaft_power,
        regime,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deadbands() -> PropulsorRegimeDeadbands {
        PropulsorRegimeDeadbands {
            axial_flow_m_s: 0.05,
            rpm: 5.0,
            thrust_n: 0.1,
            shaft_power_w: 1.0,
        }
    }

    fn point(flow: f64, rpm: f64, thrust: f64, power: f64) -> SignedPropulsorOperatingPoint {
        SignedPropulsorOperatingPoint {
            axial_flow_m_s: flow,
            rpm,
            thrust_n: thrust,
            shaft_power_into_propulsor_w: power,
        }
    }

    #[test]
    fn classifies_forward_powered_propulsion() {
        let result = classify_propulsor_regime(point(15.0, 4_000.0, 120.0, 2_500.0), deadbands()).unwrap();
        assert_eq!(result.regime, PropulsorOperatingRegime::ForwardPoweredPropulsion);
        assert_eq!(result.flow, AxialFlowState::ForwardAxialFlow);
        assert_eq!(result.shaft_power, ShaftPowerState::PowerIntoShaft);
    }

    #[test]
    fn classifies_static_powered_thrust_without_static_efficiency_claim() {
        let result = classify_propulsor_regime(point(0.0, 4_000.0, 120.0, 2_500.0), deadbands()).unwrap();
        assert_eq!(result.regime, PropulsorOperatingRegime::StaticPoweredThrust);
    }

    #[test]
    fn static_powered_low_thrust_has_its_own_regime() {
        let result = classify_propulsor_regime(point(0.0, 2_000.0, 0.05, 100.0), deadbands()).unwrap();
        assert_eq!(result.regime, PropulsorOperatingRegime::StaticPoweredLowThrust);
    }

    #[test]
    fn identifies_forward_flow_with_power_extraction_as_windmilling_regime() {
        let result = classify_propulsor_regime(point(20.0, 3_000.0, 25.0, -400.0), deadbands()).unwrap();
        assert_eq!(result.regime, PropulsorOperatingRegime::ForwardFlowPowerExtraction);
        assert_eq!(result.shaft_power, ShaftPowerState::PowerExtractedFromShaft);
    }

    #[test]
    fn identifies_reverse_flow_without_erasing_other_axes() {
        let result = classify_propulsor_regime(point(-8.0, -800.0, -4.0, 25.0), deadbands()).unwrap();
        assert_eq!(result.regime, PropulsorOperatingRegime::ReverseFlowOperation);
        assert_eq!(result.rotation, RotationState::ReverseRotation);
        assert_eq!(result.thrust, ThrustState::NegativeThrust);
        assert_eq!(result.shaft_power, ShaftPowerState::PowerIntoShaft);
    }

    #[test]
    fn identifies_reverse_rotation_when_flow_is_not_reversed() {
        let result = classify_propulsor_regime(point(5.0, -1_200.0, 10.0, 80.0), deadbands()).unwrap();
        assert_eq!(result.regime, PropulsorOperatingRegime::ReverseRotationOperation);
    }

    #[test]
    fn identifies_powered_negative_thrust_as_distinct_from_normal_propulsion() {
        let result = classify_propulsor_regime(point(10.0, 3_000.0, -15.0, 600.0), deadbands()).unwrap();
        assert_eq!(result.regime, PropulsorOperatingRegime::ForwardPoweredNegativeThrust);
    }

    #[test]
    fn stopped_rotor_with_near_zero_power_is_not_mislabelled_as_propulsive() {
        let result = classify_propulsor_regime(point(10.0, 2.0, 0.0, 0.4), deadbands()).unwrap();
        assert_eq!(result.regime, PropulsorOperatingRegime::StoppedRotor);
    }

    #[test]
    fn stopped_rotor_with_material_power_is_flagged_as_inconsistent() {
        let result = classify_propulsor_regime(point(0.0, 0.0, 0.0, 100.0), deadbands()).unwrap();
        assert_eq!(result.regime, PropulsorOperatingRegime::InconsistentPowerWithStoppedRotor);
    }

    #[test]
    fn explicit_deadbands_control_near_zero_classification() {
        let result = classify_propulsor_regime(point(0.04, 3.0, 0.08, 0.5), deadbands()).unwrap();
        assert_eq!(result.flow, AxialFlowState::NearStatic);
        assert_eq!(result.rotation, RotationState::NearStopped);
        assert_eq!(result.thrust, ThrustState::NearZero);
        assert_eq!(result.shaft_power, ShaftPowerState::NearZero);
    }

    #[test]
    fn invalid_and_non_finite_channels_fail_closed() {
        assert_eq!(
            classify_propulsor_regime(point(f64::NAN, 100.0, 1.0, 1.0), deadbands()),
            Err(PropulsorRegimeError::NonFiniteInput { field: "axial_flow_m_s" })
        );

        let mut invalid = deadbands();
        invalid.rpm = -1.0;
        assert_eq!(
            classify_propulsor_regime(point(1.0, 100.0, 1.0, 1.0), invalid),
            Err(PropulsorRegimeError::NegativeDeadband { field: "rpm" })
        );
    }

    #[test]
    fn all_near_zero_point_has_explicit_stopped_label() {
        let result = classify_propulsor_regime(point(0.0, 0.0, 0.0, 0.0), deadbands()).unwrap();
        assert_eq!(result.regime, PropulsorOperatingRegime::StoppedRotor);
    }

    #[test]
    fn identical_point_and_deadbands_are_deterministic() {
        let sample = point(12.0, 3_000.0, 20.0, 750.0);
        assert_eq!(
            classify_propulsor_regime(sample, deadbands()).unwrap(),
            classify_propulsor_regime(sample, deadbands()).unwrap()
        );
    }
}
