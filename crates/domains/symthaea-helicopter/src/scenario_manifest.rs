// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned, deterministic flight-scenario manifests.
//!
//! A qualification claim must identify the initial condition, environment,
//! fault timing, expected terminal class, and exact physics cadence. This
//! module validates that declaration and compiles time-based perturbations into
//! the simulator's step-based schedule without ad hoc rounding at call sites.

use serde::{Deserialize, Serialize};

use crate::perturbations::{HelicopterPerturbation, PerturbationSchedule};
use crate::terrain_safety::AxisAlignedGeofence;
use crate::wind_model::WindConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScenarioExpectedOutcome {
    RemainAirborne,
    SafeTouchdown,
    ControlledEmergencyLanding,
    DemonstrateFailure,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedPerturbation {
    pub perturbation: HelicopterPerturbation,
    pub start_time_s: f64,
    pub end_time_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightScenarioManifest {
    pub schema_version: String,
    pub scenario_id: String,
    pub seed: u64,
    pub physics_hz: f64,
    pub duration_s: f64,
    pub initial_altitude_m: f64,
    pub wind: WindConfig,
    pub geofence: AxisAlignedGeofence,
    pub perturbations: Vec<TimedPerturbation>,
    pub expected_outcome: ScenarioExpectedOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioManifestError {
    InvalidIdentity,
    NonFiniteValue,
    InvalidCadence,
    InvalidDuration,
    InvalidInitialCondition,
    InvalidWind,
    InvalidGeofence,
    InvalidPerturbationWindow,
    PerturbationOutsideScenario,
    StepCountOverflow,
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct CompiledFlightScenario {
    pub manifest: FlightScenarioManifest,
    pub total_steps: usize,
    pub perturbation_schedule: PerturbationSchedule,
    pub canonical_manifest_json: Vec<u8>,
    pub manifest_digest_fnv1a64: String,
}

impl FlightScenarioManifest {
    pub fn validate(&self) -> Result<(), ScenarioManifestError> {
        if self.schema_version.trim().is_empty() || self.scenario_id.trim().is_empty() {
            return Err(ScenarioManifestError::InvalidIdentity);
        }
        if !self.physics_hz.is_finite()
            || !self.duration_s.is_finite()
            || !self.initial_altitude_m.is_finite()
        {
            return Err(ScenarioManifestError::NonFiniteValue);
        }
        if self.physics_hz <= 0.0 || self.physics_hz > 10_000.0 {
            return Err(ScenarioManifestError::InvalidCadence);
        }
        if self.duration_s <= 0.0 {
            return Err(ScenarioManifestError::InvalidDuration);
        }
        if self.initial_altitude_m < 0.0 {
            return Err(ScenarioManifestError::InvalidInitialCondition);
        }
        if !valid_wind(&self.wind) {
            return Err(ScenarioManifestError::InvalidWind);
        }
        if !self.geofence.validate() {
            return Err(ScenarioManifestError::InvalidGeofence);
        }
        for entry in &self.perturbations {
            if !entry.start_time_s.is_finite()
                || entry.end_time_s.is_some_and(|end| !end.is_finite())
            {
                return Err(ScenarioManifestError::NonFiniteValue);
            }
            if entry.start_time_s < 0.0
                || entry
                    .end_time_s
                    .is_some_and(|end| end <= entry.start_time_s)
            {
                return Err(ScenarioManifestError::InvalidPerturbationWindow);
            }
            if entry.start_time_s >= self.duration_s
                || entry.end_time_s.is_some_and(|end| end > self.duration_s)
            {
                return Err(ScenarioManifestError::PerturbationOutsideScenario);
            }
        }
        Ok(())
    }

    pub fn compile(&self) -> Result<CompiledFlightScenario, ScenarioManifestError> {
        self.validate()?;
        let exact_steps = self.duration_s * self.physics_hz;
        if !exact_steps.is_finite() || exact_steps > usize::MAX as f64 {
            return Err(ScenarioManifestError::StepCountOverflow);
        }
        let total_steps = exact_steps.ceil() as usize;
        let mut entries = self.perturbations.clone();
        entries.sort_by(|left, right| {
            left.start_time_s
                .total_cmp(&right.start_time_s)
                .then_with(|| {
                    perturbation_rank(&left.perturbation)
                        .cmp(&perturbation_rank(&right.perturbation))
                })
        });
        let mut schedule = PerturbationSchedule::new();
        for entry in &entries {
            let start_step = seconds_to_step(entry.start_time_s, self.physics_hz, total_steps);
            let clear_step = entry
                .end_time_s
                .map(|time_s| seconds_to_step(time_s, self.physics_hz, total_steps));
            schedule = schedule.add(entry.perturbation.clone(), start_step, clear_step);
        }

        let mut canonical = self.clone();
        canonical.perturbations = entries;
        let canonical_manifest_json = serde_json::to_vec(&canonical)
            .map_err(|_| ScenarioManifestError::SerializationFailed)?;
        let digest = fnv1a64(&canonical_manifest_json);
        Ok(CompiledFlightScenario {
            manifest: canonical,
            total_steps,
            perturbation_schedule: schedule,
            canonical_manifest_json,
            manifest_digest_fnv1a64: format!("fnv1a64:{digest:016x}"),
        })
    }
}

fn seconds_to_step(time_s: f64, physics_hz: f64, total_steps: usize) -> usize {
    (time_s * physics_hz).round().clamp(0.0, total_steps as f64) as usize
}

fn valid_wind(config: &WindConfig) -> bool {
    config.steady_wind.iter().all(|value| value.is_finite())
        && config.gust_intensity.is_finite()
        && config.gust_intensity >= 0.0
        && config.gust_bandwidth.is_finite()
        && config.gust_bandwidth > 0.0
        && config.rotor_radius.is_finite()
        && config.rotor_radius > 0.0
}

fn perturbation_rank(perturbation: &HelicopterPerturbation) -> u8 {
    match perturbation {
        HelicopterPerturbation::EngineFlameout => 0,
        HelicopterPerturbation::TailRotorFailure => 1,
        HelicopterPerturbation::RotorDegradation { .. } => 2,
        HelicopterPerturbation::PayloadDrop { .. } => 3,
        HelicopterPerturbation::Crosswind { .. } => 4,
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> FlightScenarioManifest {
        FlightScenarioManifest {
            schema_version: "symthaea-helicopter-scenario-v1".to_string(),
            scenario_id: "engine-out-autorotation".to_string(),
            seed: 42,
            physics_hz: 300.0,
            duration_s: 10.0,
            initial_altitude_m: 100.0,
            wind: WindConfig::light_wind(),
            geofence: AxisAlignedGeofence {
                min_east_m: -1000.0,
                max_east_m: 1000.0,
                min_north_m: -1000.0,
                max_north_m: 1000.0,
                min_altitude_m: 0.0,
                max_altitude_m: 500.0,
            },
            perturbations: vec![TimedPerturbation {
                perturbation: HelicopterPerturbation::EngineFlameout,
                start_time_s: 2.0,
                end_time_s: None,
            }],
            expected_outcome: ScenarioExpectedOutcome::ControlledEmergencyLanding,
        }
    }

    #[test]
    fn time_manifest_compiles_to_exact_step_schedule() {
        let compiled = manifest().compile().unwrap();
        assert_eq!(compiled.total_steps, 3000);
        assert!(compiled.perturbation_schedule.active_at(599).is_empty());
        assert_eq!(compiled.perturbation_schedule.active_at(600).len(), 1);
    }

    #[test]
    fn canonical_digest_is_order_independent_for_perturbation_declaration() {
        let mut first = manifest();
        first.perturbations.push(TimedPerturbation {
            perturbation: HelicopterPerturbation::Crosswind { force_n: 1000.0 },
            start_time_s: 1.0,
            end_time_s: Some(3.0),
        });
        let mut second = first.clone();
        second.perturbations.reverse();
        assert_eq!(
            first.compile().unwrap().manifest_digest_fnv1a64,
            second.compile().unwrap().manifest_digest_fnv1a64
        );
    }

    #[test]
    fn perturbation_outside_duration_is_rejected() {
        let mut invalid = manifest();
        invalid.perturbations[0].start_time_s = invalid.duration_s;
        assert_eq!(
            invalid.compile().err(),
            Some(ScenarioManifestError::PerturbationOutsideScenario)
        );
    }
}
