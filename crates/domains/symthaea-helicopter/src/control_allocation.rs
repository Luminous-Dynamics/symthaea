// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fault-aware allocation from virtual flight demands to helicopter actuators.
//!
//! Guidance and learned residual controllers should express desired vertical,
//! roll, pitch, and yaw response. This allocator converts those demands into
//! bounded collective/cyclic/pedal/governor commands while reporting which axes
//! could not be realized under current actuator health.

use serde::{Deserialize, Serialize};

use crate::perturbations::PerturbationEffects;
use crate::types::{HelicopterCommand, HelicopterState};

/// Normalized actuator effectiveness. One is nominal; zero is unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActuatorHealth {
    pub collective: f64,
    pub cyclic_lon: f64,
    pub cyclic_lat: f64,
    pub pedal: f64,
    pub main_rotor: f64,
    pub tail_rotor: f64,
}

impl Default for ActuatorHealth {
    fn default() -> Self {
        Self {
            collective: 1.0,
            cyclic_lon: 1.0,
            cyclic_lat: 1.0,
            pedal: 1.0,
            main_rotor: 1.0,
            tail_rotor: 1.0,
        }
    }
}

impl ActuatorHealth {
    pub fn validate(&self) -> Result<(), ControlAllocationError> {
        for value in [
            self.collective,
            self.cyclic_lon,
            self.cyclic_lat,
            self.pedal,
            self.main_rotor,
            self.tail_rotor,
        ] {
            if !value.is_finite() {
                return Err(ControlAllocationError::NonFiniteHealth);
            }
            if !(0.0..=1.0).contains(&value) {
                return Err(ControlAllocationError::HealthOutOfRange);
            }
        }
        Ok(())
    }

    /// Conservative intersection of independently estimated actuator health.
    pub fn min_with(self, other: Self) -> Self {
        Self {
            collective: self.collective.min(other.collective),
            cyclic_lon: self.cyclic_lon.min(other.cyclic_lon),
            cyclic_lat: self.cyclic_lat.min(other.cyclic_lat),
            pedal: self.pedal.min(other.pedal),
            main_rotor: self.main_rotor.min(other.main_rotor),
            tail_rotor: self.tail_rotor.min(other.tail_rotor),
        }
    }

    /// Compile simulator perturbation state into the allocator's authority map.
    pub fn from_perturbation_effects(effects: PerturbationEffects) -> Self {
        Self {
            main_rotor: if effects.engine_available {
                effects.main_rotor_efficiency
            } else {
                0.0
            },
            tail_rotor: effects.tail_rotor_efficiency,
            pedal: effects.tail_rotor_efficiency,
            ..Self::default()
        }
    }
}

/// Virtual control request in SI-like units used by the reduced-order stack.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct VirtualControlDemand {
    /// Desired vertical acceleration relative to trim, m/s².
    pub vertical_accel_mps2: f64,
    /// Desired roll angular acceleration, rad/s².
    pub roll_accel_rad_s2: f64,
    /// Desired pitch angular acceleration, rad/s².
    pub pitch_accel_rad_s2: f64,
    /// Desired yaw angular acceleration, rad/s².
    pub yaw_accel_rad_s2: f64,
}

impl VirtualControlDemand {
    pub fn is_finite(&self) -> bool {
        [
            self.vertical_accel_mps2,
            self.roll_accel_rad_s2,
            self.pitch_accel_rad_s2,
            self.yaw_accel_rad_s2,
        ]
        .iter()
        .all(|value| value.is_finite())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlAllocationError {
    NonFiniteDemand,
    NonFiniteHealth,
    HealthOutOfRange,
    InvalidConfig,
}

/// Mapping and degradation policy for the reduced-order actuator model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ControlAllocationConfig {
    pub collective_per_vertical_accel: f64,
    pub governor_per_vertical_accel: f64,
    pub cyclic_per_angular_accel: f64,
    pub pedal_per_yaw_accel: f64,
    /// Tail effectiveness below this value declares yaw unavailable.
    pub minimum_tail_authority: f64,
    /// Main-rotor command cap used to reduce reaction torque after tail loss.
    pub tail_failure_thrust_cap: f64,
    /// Main rotor RPM below which maneuver authority is progressively reduced.
    pub minimum_maneuver_rpm: f64,
}

impl Default for ControlAllocationConfig {
    fn default() -> Self {
        Self {
            collective_per_vertical_accel: 0.06,
            governor_per_vertical_accel: 0.025,
            cyclic_per_angular_accel: 0.20,
            pedal_per_yaw_accel: 0.18,
            minimum_tail_authority: 0.15,
            tail_failure_thrust_cap: 0.48,
            minimum_maneuver_rpm: 2_200.0,
        }
    }
}

impl ControlAllocationConfig {
    pub fn validate(&self) -> Result<(), ControlAllocationError> {
        let positive = [
            self.collective_per_vertical_accel,
            self.governor_per_vertical_accel,
            self.cyclic_per_angular_accel,
            self.pedal_per_yaw_accel,
            self.minimum_maneuver_rpm,
        ];
        if positive.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || !self.minimum_tail_authority.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_tail_authority)
            || !self.tail_failure_thrust_cap.is_finite()
            || !(0.0..=1.0).contains(&self.tail_failure_thrust_cap)
        {
            return Err(ControlAllocationError::InvalidConfig);
        }
        Ok(())
    }
}

/// Allocation evidence retained for safety and benchmark analysis.
#[derive(Debug, Clone, Copy)]
pub struct ControlAllocationResult {
    pub command: HelicopterCommand,
    /// collective, cyclic_lon, cyclic_lat, pedal, thrust, tail_rotor
    pub saturated: [bool; 6],
    /// vertical, roll, pitch, yaw
    pub degraded_axes: [bool; 4],
    /// Requested minus approximately realized virtual demand.
    pub residual: [f64; 4],
}

#[derive(Debug, Clone)]
pub struct FaultAwareControlAllocator {
    config: ControlAllocationConfig,
}

impl Default for FaultAwareControlAllocator {
    fn default() -> Self {
        Self {
            config: ControlAllocationConfig::default(),
        }
    }
}

impl FaultAwareControlAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: ControlAllocationConfig) -> Result<Self, ControlAllocationError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn allocate(
        &self,
        state: &HelicopterState,
        demand: VirtualControlDemand,
        health: ActuatorHealth,
    ) -> Result<ControlAllocationResult, ControlAllocationError> {
        self.config.validate()?;
        health.validate()?;
        if !demand.is_finite() {
            return Err(ControlAllocationError::NonFiniteDemand);
        }

        let rpm_authority = (state.main_rotor_rpm / self.config.minimum_maneuver_rpm)
            .clamp(0.0, 1.0)
            * health.main_rotor;
        let roll_authority = health.cyclic_lat * rpm_authority;
        let pitch_authority = health.cyclic_lon * rpm_authority;
        let yaw_authority = health.pedal * health.tail_rotor;

        let requested = HelicopterCommand {
            collective: (0.30
                + self.config.collective_per_vertical_accel * demand.vertical_accel_mps2)
                as f32,
            cyclic_lon: (self.config.cyclic_per_angular_accel
                * demand.pitch_accel_rad_s2
                * pitch_authority) as f32,
            cyclic_lat: (-self.config.cyclic_per_angular_accel
                * demand.roll_accel_rad_s2
                * roll_authority) as f32,
            pedal: (self.config.pedal_per_yaw_accel * demand.yaw_accel_rad_s2 * yaw_authority)
                as f32,
            thrust: (0.60 + self.config.governor_per_vertical_accel * demand.vertical_accel_mps2)
                as f32,
            tail_rotor: (0.50 * health.tail_rotor) as f32,
        };

        let mut command = requested.clamped();
        command.collective = (command.collective as f64 * health.collective) as f32;
        command.thrust = (command.thrust as f64 * health.main_rotor) as f32;
        command.cyclic_lon = (command.cyclic_lon as f64 * health.cyclic_lon) as f32;
        command.cyclic_lat = (command.cyclic_lat as f64 * health.cyclic_lat) as f32;

        let tail_failed = yaw_authority < self.config.minimum_tail_authority;
        if tail_failed {
            command.pedal = 0.0;
            command.tail_rotor = 0.0;
            command.thrust = command
                .thrust
                .min(self.config.tail_failure_thrust_cap as f32);
            // Avoid demanding a climb that increases main-rotor reaction torque
            // while anti-torque authority is unavailable.
            command.collective = command.collective.min(0.30);
        }
        command = command.clamped();

        let saturated = [
            (command.collective - requested.collective).abs() > 1.0e-6,
            (command.cyclic_lon - requested.cyclic_lon).abs() > 1.0e-6,
            (command.cyclic_lat - requested.cyclic_lat).abs() > 1.0e-6,
            (command.pedal - requested.pedal).abs() > 1.0e-6,
            (command.thrust - requested.thrust).abs() > 1.0e-6,
            (command.tail_rotor - requested.tail_rotor).abs() > 1.0e-6,
        ];

        let realized_vertical = ((command.collective as f64 - 0.30)
            / self.config.collective_per_vertical_accel)
            .min((command.thrust as f64 - 0.60) / self.config.governor_per_vertical_accel);
        let realized_roll =
            -command.cyclic_lat as f64 / self.config.cyclic_per_angular_accel.max(1.0e-9);
        let realized_pitch =
            command.cyclic_lon as f64 / self.config.cyclic_per_angular_accel.max(1.0e-9);
        let realized_yaw = command.pedal as f64 / self.config.pedal_per_yaw_accel.max(1.0e-9);

        Ok(ControlAllocationResult {
            command,
            saturated,
            degraded_axes: [
                health.main_rotor < 0.999 || health.collective < 0.999,
                roll_authority < 0.999,
                pitch_authority < 0.999,
                tail_failed,
            ],
            residual: [
                demand.vertical_accel_mps2 - realized_vertical,
                demand.roll_accel_rad_s2 - realized_roll,
                demand.pitch_accel_rad_s2 - realized_pitch,
                demand.yaw_accel_rad_s2 - realized_yaw,
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_allocation_maps_all_axes() {
        let allocator = FaultAwareControlAllocator::new();
        let state = HelicopterState::hover(20.0);
        let result = allocator
            .allocate(
                &state,
                VirtualControlDemand {
                    vertical_accel_mps2: 1.0,
                    roll_accel_rad_s2: 0.5,
                    pitch_accel_rad_s2: -0.5,
                    yaw_accel_rad_s2: 0.25,
                },
                ActuatorHealth::default(),
            )
            .unwrap();
        assert!(result.command.collective > 0.30);
        assert!(result.command.cyclic_lat < 0.0);
        assert!(result.command.cyclic_lon < 0.0);
        assert!(result.command.pedal > 0.0);
        assert!(!result.degraded_axes[3]);
    }

    #[test]
    fn tail_failure_reports_yaw_residual_and_reduces_torque_demand() {
        let allocator = FaultAwareControlAllocator::new();
        let state = HelicopterState::hover(20.0);
        let result = allocator
            .allocate(
                &state,
                VirtualControlDemand {
                    vertical_accel_mps2: 2.0,
                    yaw_accel_rad_s2: 1.0,
                    ..VirtualControlDemand::default()
                },
                ActuatorHealth {
                    pedal: 0.0,
                    tail_rotor: 0.0,
                    ..ActuatorHealth::default()
                },
            )
            .unwrap();
        assert_eq!(result.command.pedal, 0.0);
        assert_eq!(result.command.tail_rotor, 0.0);
        assert!(result.command.thrust <= 0.48);
        assert!(result.degraded_axes[3]);
        assert!(result.residual[3] > 0.9);
    }

    #[test]
    fn low_rotor_rpm_degrades_cyclic_axes() {
        let allocator = FaultAwareControlAllocator::new();
        let mut state = HelicopterState::hover(20.0);
        state.main_rotor_rpm = 1_100.0;
        let result = allocator
            .allocate(
                &state,
                VirtualControlDemand {
                    roll_accel_rad_s2: 1.0,
                    pitch_accel_rad_s2: 1.0,
                    ..VirtualControlDemand::default()
                },
                ActuatorHealth::default(),
            )
            .unwrap();
        assert!(result.degraded_axes[1]);
        assert!(result.degraded_axes[2]);
        assert!(result.command.cyclic_lon.abs() < 0.20);
        assert!(result.command.cyclic_lat.abs() < 0.20);
    }

    #[test]
    fn perturbation_effects_compile_to_health() {
        let health = ActuatorHealth::from_perturbation_effects(PerturbationEffects {
            engine_available: false,
            tail_rotor_efficiency: 0.0,
            ..PerturbationEffects::default()
        });
        assert_eq!(health.main_rotor, 0.0);
        assert_eq!(health.tail_rotor, 0.0);
    }
}
