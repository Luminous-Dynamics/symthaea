// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rotor-regime-aware command protection.
//!
//! Authority tiers bound generic maneuvering, but they do not by themselves
//! protect rotor energy or escape hazardous aerodynamic regimes. This module is
//! a deterministic final command guard that emits every intervention as
//! evidence. It is a reduced-order research policy, not a certified envelope.

use serde::{Deserialize, Serialize};

use crate::rotor_dynamics::RotorFlightRegime;
use crate::types::HelicopterCommand;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RotorEnvelopeProtectionConfig {
    pub maximum_main_rpm: f64,
    pub minimum_control_rpm: f64,
    pub minimum_flare_energy_margin_j: f64,
    pub flare_release_altitude_m: f64,
    pub low_rpm_collective_cap: f32,
    pub energy_preservation_collective_cap: f32,
    pub overspeed_thrust_cap: f32,
    pub overspeed_collective_cap: f32,
    pub vortex_collective_cap: f32,
    pub vortex_escape_cyclic_lon: f32,
    pub minimum_tail_authority: f64,
    pub tail_degraded_collective_cap: f32,
    pub tail_degraded_pedal_cap: f32,
}

impl Default for RotorEnvelopeProtectionConfig {
    fn default() -> Self {
        Self {
            maximum_main_rpm: 5_300.0,
            minimum_control_rpm: 1_800.0,
            minimum_flare_energy_margin_j: 0.0,
            flare_release_altitude_m: 6.0,
            low_rpm_collective_cap: 0.18,
            energy_preservation_collective_cap: 0.12,
            overspeed_thrust_cap: 0.35,
            overspeed_collective_cap: 0.20,
            vortex_collective_cap: 0.15,
            vortex_escape_cyclic_lon: 0.25,
            minimum_tail_authority: 0.35,
            tail_degraded_collective_cap: 0.30,
            tail_degraded_pedal_cap: 0.20,
        }
    }
}

impl RotorEnvelopeProtectionConfig {
    pub fn validate(&self) -> Result<(), RotorEnvelopeProtectionError> {
        if !self.maximum_main_rpm.is_finite()
            || !self.minimum_control_rpm.is_finite()
            || self.maximum_main_rpm <= self.minimum_control_rpm
            || self.minimum_control_rpm <= 0.0
            || !self.minimum_flare_energy_margin_j.is_finite()
            || !self.flare_release_altitude_m.is_finite()
            || self.flare_release_altitude_m < 0.0
            || !self.minimum_tail_authority.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_tail_authority)
        {
            return Err(RotorEnvelopeProtectionError::InvalidConfiguration);
        }
        for value in [
            self.low_rpm_collective_cap,
            self.energy_preservation_collective_cap,
            self.overspeed_thrust_cap,
            self.overspeed_collective_cap,
            self.vortex_collective_cap,
            self.tail_degraded_collective_cap,
            self.tail_degraded_pedal_cap,
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(RotorEnvelopeProtectionError::InvalidConfiguration);
            }
        }
        if !self.vortex_escape_cyclic_lon.is_finite()
            || !(-1.0..=1.0).contains(&self.vortex_escape_cyclic_lon)
        {
            return Err(RotorEnvelopeProtectionError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotorEnvelopeProtectionError {
    InvalidConfiguration,
    NonFiniteObservation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotorEnvelopeObservation {
    pub flight_regime: RotorFlightRegime,
    pub main_rotor_rpm: f64,
    pub flare_energy_margin_j: f64,
    pub altitude_agl_m: f64,
    pub tail_control_authority: f64,
}

impl RotorEnvelopeObservation {
    fn validate(&self) -> Result<(), RotorEnvelopeProtectionError> {
        if !self.main_rotor_rpm.is_finite()
            || !self.flare_energy_margin_j.is_finite()
            || !self.altitude_agl_m.is_finite()
            || !self.tail_control_authority.is_finite()
        {
            return Err(RotorEnvelopeProtectionError::NonFiniteObservation);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotorEnvelopeIntervention {
    MainRotorOverspeed,
    LowRotorRpm,
    PreserveAutorotationEnergy,
    VortexRingEscape,
    TailAuthorityDegraded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotorEnvelopeProtectionResult {
    pub requested: HelicopterCommand,
    pub protected: HelicopterCommand,
    pub interventions: Vec<RotorEnvelopeIntervention>,
}

#[derive(Debug, Clone)]
pub struct RotorEnvelopeProtector {
    config: RotorEnvelopeProtectionConfig,
}

impl RotorEnvelopeProtector {
    pub fn new(
        config: RotorEnvelopeProtectionConfig,
    ) -> Result<Self, RotorEnvelopeProtectionError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn protect(
        &self,
        requested: HelicopterCommand,
        observation: RotorEnvelopeObservation,
    ) -> Result<RotorEnvelopeProtectionResult, RotorEnvelopeProtectionError> {
        self.config.validate()?;
        observation.validate()?;
        let mut protected = requested.clamped();
        let mut interventions = Vec::new();

        if observation.main_rotor_rpm > self.config.maximum_main_rpm {
            protected.thrust = protected.thrust.min(self.config.overspeed_thrust_cap);
            protected.collective = protected
                .collective
                .min(self.config.overspeed_collective_cap);
            interventions.push(RotorEnvelopeIntervention::MainRotorOverspeed);
        }

        if observation.main_rotor_rpm < self.config.minimum_control_rpm {
            protected.collective = protected.collective.min(self.config.low_rpm_collective_cap);
            // Cyclic cannot be assumed effective at low rotor speed.
            protected.cyclic_lon = protected.cyclic_lon.clamp(-0.20, 0.20);
            protected.cyclic_lat = protected.cyclic_lat.clamp(-0.20, 0.20);
            interventions.push(RotorEnvelopeIntervention::LowRotorRpm);
        }

        if observation.flight_regime == RotorFlightRegime::Autorotation
            && observation.flare_energy_margin_j < self.config.minimum_flare_energy_margin_j
            && observation.altitude_agl_m > self.config.flare_release_altitude_m
        {
            protected.collective = protected
                .collective
                .min(self.config.energy_preservation_collective_cap);
            interventions.push(RotorEnvelopeIntervention::PreserveAutorotationEnergy);
        }

        if observation.flight_regime == RotorFlightRegime::VortexRingExposure {
            protected.collective = protected.collective.min(self.config.vortex_collective_cap);
            if self.config.vortex_escape_cyclic_lon >= 0.0 {
                protected.cyclic_lon = protected
                    .cyclic_lon
                    .max(self.config.vortex_escape_cyclic_lon);
            } else {
                protected.cyclic_lon = protected
                    .cyclic_lon
                    .min(self.config.vortex_escape_cyclic_lon);
            }
            interventions.push(RotorEnvelopeIntervention::VortexRingEscape);
        }

        if observation.tail_control_authority < self.config.minimum_tail_authority {
            protected.collective = protected
                .collective
                .min(self.config.tail_degraded_collective_cap);
            protected.pedal = protected.pedal.clamp(
                -self.config.tail_degraded_pedal_cap,
                self.config.tail_degraded_pedal_cap,
            );
            interventions.push(RotorEnvelopeIntervention::TailAuthorityDegraded);
        }

        Ok(RotorEnvelopeProtectionResult {
            requested,
            protected: protected.clamped(),
            interventions,
        })
    }
}

impl Default for RotorEnvelopeProtector {
    fn default() -> Self {
        Self::new(RotorEnvelopeProtectionConfig::default())
            .expect("default rotor envelope protection config must remain valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(regime: RotorFlightRegime) -> RotorEnvelopeObservation {
        RotorEnvelopeObservation {
            flight_regime: regime,
            main_rotor_rpm: 3_300.0,
            flare_energy_margin_j: 10_000.0,
            altitude_agl_m: 20.0,
            tail_control_authority: 1.0,
        }
    }

    #[test]
    fn normal_regime_preserves_valid_command() {
        let command = HelicopterCommand::hover();
        let result = RotorEnvelopeProtector::default()
            .protect(command, observation(RotorFlightRegime::Normal))
            .unwrap();
        assert_eq!(result.protected.to_ctrl(), command.to_ctrl());
        assert!(result.interventions.is_empty());
    }

    #[test]
    fn vortex_ring_reduces_collective_and_commands_escape() {
        let command = HelicopterCommand {
            collective: 0.8,
            cyclic_lon: -0.5,
            ..HelicopterCommand::hover()
        };
        let result = RotorEnvelopeProtector::default()
            .protect(command, observation(RotorFlightRegime::VortexRingExposure))
            .unwrap();
        assert!(result.protected.collective <= 0.15);
        assert!(result.protected.cyclic_lon >= 0.25);
        assert!(
            result
                .interventions
                .contains(&RotorEnvelopeIntervention::VortexRingEscape)
        );
    }

    #[test]
    fn autorotation_preserves_energy_until_flare_height() {
        let mut obs = observation(RotorFlightRegime::Autorotation);
        obs.flare_energy_margin_j = -1.0;
        let result = RotorEnvelopeProtector::default()
            .protect(
                HelicopterCommand {
                    collective: 0.7,
                    thrust: 0.0,
                    ..HelicopterCommand::hover()
                },
                obs,
            )
            .unwrap();
        assert!(result.protected.collective <= 0.12);
        assert!(
            result
                .interventions
                .contains(&RotorEnvelopeIntervention::PreserveAutorotationEnergy)
        );
    }

    #[test]
    fn tail_failure_caps_collective_and_pedal() {
        let mut obs = observation(RotorFlightRegime::Normal);
        obs.tail_control_authority = 0.0;
        let result = RotorEnvelopeProtector::default()
            .protect(
                HelicopterCommand {
                    collective: 0.8,
                    pedal: 1.0,
                    ..HelicopterCommand::hover()
                },
                obs,
            )
            .unwrap();
        assert!(result.protected.collective <= 0.30);
        assert!(result.protected.pedal <= 0.20);
    }
}
