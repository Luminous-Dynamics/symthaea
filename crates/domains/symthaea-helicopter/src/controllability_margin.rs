// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-derived controllability margins.
//!
//! Fault labels alone do not establish whether a vehicle can still satisfy the
//! demanded motion. This module converts actuator effectiveness and rotor speed
//! into conservative per-axis acceleration capacity, compares that capacity to
//! the active virtual-control demand, and exposes the remaining margin.

use serde::{Deserialize, Serialize};

use crate::control_allocation::{ActuatorHealth, VirtualControlDemand};
use crate::types::HelicopterState;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ControllabilityMarginConfig {
    pub nominal_main_rotor_rpm: f64,
    pub nominal_tail_rotor_rpm: f64,
    pub maximum_vertical_accel_mps2: f64,
    pub maximum_roll_accel_rad_s2: f64,
    pub maximum_pitch_accel_rad_s2: f64,
    pub maximum_yaw_accel_rad_s2: f64,
    pub minimum_retained_authority: f64,
    pub degraded_margin_fraction: f64,
}

impl Default for ControllabilityMarginConfig {
    fn default() -> Self {
        Self {
            nominal_main_rotor_rpm: 3_300.0,
            nominal_tail_rotor_rpm: 2_000.0,
            maximum_vertical_accel_mps2: 4.0,
            maximum_roll_accel_rad_s2: 1.8,
            maximum_pitch_accel_rad_s2: 1.8,
            maximum_yaw_accel_rad_s2: 1.2,
            minimum_retained_authority: 0.12,
            degraded_margin_fraction: 0.25,
        }
    }
}

impl ControllabilityMarginConfig {
    pub fn validate(&self) -> Result<(), ControllabilityMarginError> {
        let positive = [
            self.nominal_main_rotor_rpm,
            self.nominal_tail_rotor_rpm,
            self.maximum_vertical_accel_mps2,
            self.maximum_roll_accel_rad_s2,
            self.maximum_pitch_accel_rad_s2,
            self.maximum_yaw_accel_rad_s2,
        ];
        if positive
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
            || !self.minimum_retained_authority.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_retained_authority)
            || !self.degraded_margin_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.degraded_margin_fraction)
        {
            return Err(ControllabilityMarginError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlAxis {
    Vertical,
    Roll,
    Pitch,
    Yaw,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisControllabilityMargin {
    pub axis: ControlAxis,
    pub retained_authority: f64,
    pub available_accel: f64,
    pub demanded_accel: f64,
    pub absolute_margin: f64,
    pub margin_fraction: f64,
    pub demand_satisfied: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControllabilityState {
    Nominal,
    Degraded,
    Uncontrollable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControllabilityAssessment {
    pub state: ControllabilityState,
    pub axes: [AxisControllabilityMargin; 4],
    pub limiting_axis: ControlAxis,
    pub minimum_margin_fraction: f64,
}

impl ControllabilityAssessment {
    pub fn axis(&self, axis: ControlAxis) -> &AxisControllabilityMargin {
        self.axes
            .iter()
            .find(|margin| margin.axis == axis)
            .expect("all four control axes are always present")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControllabilityMarginError {
    InvalidConfig,
    NonFiniteState,
    NonFiniteDemand,
    InvalidHealth,
}

#[derive(Debug, Clone)]
pub struct ControllabilityMarginEvaluator {
    config: ControllabilityMarginConfig,
}

impl Default for ControllabilityMarginEvaluator {
    fn default() -> Self {
        Self {
            config: ControllabilityMarginConfig::default(),
        }
    }
}

impl ControllabilityMarginEvaluator {
    pub fn new(config: ControllabilityMarginConfig) -> Result<Self, ControllabilityMarginError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn assess(
        &self,
        state: &HelicopterState,
        health: ActuatorHealth,
        demand: VirtualControlDemand,
    ) -> Result<ControllabilityAssessment, ControllabilityMarginError> {
        self.config.validate()?;
        if !state.is_finite() {
            return Err(ControllabilityMarginError::NonFiniteState);
        }
        if !demand.is_finite() {
            return Err(ControllabilityMarginError::NonFiniteDemand);
        }
        health
            .validate()
            .map_err(|_| ControllabilityMarginError::InvalidHealth)?;

        let main_rpm_fraction =
            (state.main_rotor_rpm / self.config.nominal_main_rotor_rpm).clamp(0.0, 1.0);
        let tail_rpm_fraction =
            (state.tail_rotor_rpm / self.config.nominal_tail_rotor_rpm).clamp(0.0, 1.0);

        // Lift scales approximately with rotor-speed squared in the reduced-order
        // model, while cyclic moment authority is treated as first order in RPM.
        let vertical_authority =
            health.collective.min(health.main_rotor) * main_rpm_fraction.powi(2);
        let roll_authority = health.cyclic_lat * health.main_rotor * main_rpm_fraction;
        let pitch_authority = health.cyclic_lon * health.main_rotor * main_rpm_fraction;
        let yaw_authority =
            health.pedal * health.tail_rotor * tail_rpm_fraction * main_rpm_fraction;

        let axes = [
            axis_margin(
                ControlAxis::Vertical,
                vertical_authority,
                self.config.maximum_vertical_accel_mps2,
                demand.vertical_accel_mps2,
            ),
            axis_margin(
                ControlAxis::Roll,
                roll_authority,
                self.config.maximum_roll_accel_rad_s2,
                demand.roll_accel_rad_s2,
            ),
            axis_margin(
                ControlAxis::Pitch,
                pitch_authority,
                self.config.maximum_pitch_accel_rad_s2,
                demand.pitch_accel_rad_s2,
            ),
            axis_margin(
                ControlAxis::Yaw,
                yaw_authority,
                self.config.maximum_yaw_accel_rad_s2,
                demand.yaw_accel_rad_s2,
            ),
        ];

        let limiting = axes
            .iter()
            .min_by(|left, right| left.margin_fraction.total_cmp(&right.margin_fraction))
            .expect("fixed-size axis set is non-empty");
        let minimum_margin_fraction = limiting.margin_fraction;
        let insufficient_authority = axes.iter().any(|axis| {
            axis.retained_authority < self.config.minimum_retained_authority
                || !axis.demand_satisfied
        });
        let state = if insufficient_authority {
            ControllabilityState::Uncontrollable
        } else if minimum_margin_fraction < self.config.degraded_margin_fraction {
            ControllabilityState::Degraded
        } else {
            ControllabilityState::Nominal
        };

        Ok(ControllabilityAssessment {
            state,
            axes,
            limiting_axis: limiting.axis,
            minimum_margin_fraction,
        })
    }
}

fn axis_margin(
    axis: ControlAxis,
    retained_authority: f64,
    nominal_capacity: f64,
    demand: f64,
) -> AxisControllabilityMargin {
    let retained_authority = retained_authority.clamp(0.0, 1.0);
    let available_accel = nominal_capacity * retained_authority;
    let demanded_accel = demand.abs();
    let absolute_margin = available_accel - demanded_accel;
    let margin_fraction = if available_accel > 1.0e-9 {
        absolute_margin / available_accel
    } else if demanded_accel <= 1.0e-9 {
        0.0
    } else {
        -1.0
    };
    AxisControllabilityMargin {
        axis,
        retained_authority,
        available_accel,
        demanded_accel,
        absolute_margin,
        margin_fraction,
        demand_satisfied: absolute_margin >= -1.0e-9,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_hover_has_positive_margins() {
        let assessment = ControllabilityMarginEvaluator::default()
            .assess(
                &HelicopterState::hover(20.0),
                ActuatorHealth::default(),
                VirtualControlDemand::default(),
            )
            .unwrap();
        assert_eq!(assessment.state, ControllabilityState::Nominal);
        assert!(assessment.minimum_margin_fraction > 0.9);
    }

    #[test]
    fn lost_tail_authority_is_uncontrollable() {
        let mut health = ActuatorHealth::default();
        health.tail_rotor = 0.0;
        health.pedal = 0.0;
        let assessment = ControllabilityMarginEvaluator::default()
            .assess(
                &HelicopterState::hover(20.0),
                health,
                VirtualControlDemand {
                    yaw_accel_rad_s2: 0.2,
                    ..VirtualControlDemand::default()
                },
            )
            .unwrap();
        assert_eq!(assessment.state, ControllabilityState::Uncontrollable);
        assert_eq!(assessment.limiting_axis, ControlAxis::Yaw);
    }

    #[test]
    fn low_rotor_speed_reduces_vertical_capacity_quadratically() {
        let evaluator = ControllabilityMarginEvaluator::default();
        let nominal = evaluator
            .assess(
                &HelicopterState::hover(20.0),
                ActuatorHealth::default(),
                VirtualControlDemand::default(),
            )
            .unwrap();
        let mut slow = HelicopterState::hover(20.0);
        slow.main_rotor_rpm *= 0.5;
        let degraded = evaluator
            .assess(
                &slow,
                ActuatorHealth::default(),
                VirtualControlDemand::default(),
            )
            .unwrap();
        let ratio = degraded.axis(ControlAxis::Vertical).available_accel
            / nominal.axis(ControlAxis::Vertical).available_accel;
        assert!((ratio - 0.25).abs() < 1.0e-9);
    }
}
