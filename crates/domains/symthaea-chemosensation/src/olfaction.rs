// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hardware-independent metal-oxide (MOX) olfaction simulator.
//!
//! This is a research fixture for cognition/calibration tests, not a vendor
//! model. It captures the properties the perception stack must handle:
//! cross-sensitive channels, concentration dependence, humidity confounding,
//! and asymmetric rise/recovery dynamics.

use crate::{
    CalibrationState, ChemicalChannel, ChemicalModality, ChemicalObservation,
    EnvironmentReading, MeasurementUnit, SensorHealth,
};

#[derive(Debug, Clone, PartialEq)]
pub enum OlfactorySimulationError {
    InvalidStepDuration(f32),
    InvalidConcentration(f32),
    InvalidTemperature(f32),
    InvalidHumidity(f32),
    AffinityCountMismatch { expected: usize, actual: usize },
    InvalidAffinity { index: usize, value: f32 },
    InvalidChannelModel(String),
}

#[derive(Debug, Clone)]
pub struct MoxChannelModel {
    pub name: String,
    pub baseline_ohms: f32,
    pub sensitivity: f32,
    pub humidity_coefficient: f32,
    pub rise_tau_s: f32,
    pub recovery_tau_s: f32,
}

impl MoxChannelModel {
    pub fn new(name: impl Into<String>, baseline_ohms: f32, sensitivity: f32) -> Self {
        assert!(
            baseline_ohms.is_finite() && baseline_ohms > 0.0,
            "MOX baseline resistance must be finite and positive"
        );
        assert!(
            sensitivity.is_finite() && sensitivity >= 0.0,
            "MOX sensitivity must be finite and non-negative"
        );
        Self {
            name: name.into(),
            baseline_ohms,
            sensitivity,
            humidity_coefficient: 0.1,
            rise_tau_s: 1.0,
            recovery_tau_s: 3.0,
        }
    }

    fn is_valid(&self) -> bool {
        self.baseline_ohms.is_finite()
            && self.baseline_ohms > 0.0
            && self.sensitivity.is_finite()
            && self.sensitivity >= 0.0
            && self.humidity_coefficient.is_finite()
            && self.rise_tau_s.is_finite()
            && self.rise_tau_s > 0.0
            && self.recovery_tau_s.is_finite()
            && self.recovery_tau_s > 0.0
    }
}

#[derive(Debug, Clone)]
pub struct OlfactoryStimulus {
    /// Total volatile concentration represented by this fixture.
    pub concentration_ppm: f32,
    /// Per-channel affinity/cross-sensitivity coefficients.
    pub affinities: Vec<f32>,
    pub temperature_c: f32,
    pub humidity_rh: f32,
}

impl OlfactoryStimulus {
    pub fn clean_air(channel_count: usize, temperature_c: f32, humidity_rh: f32) -> Self {
        Self {
            concentration_ppm: 0.0,
            affinities: vec![0.0; channel_count],
            temperature_c,
            humidity_rh,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MoxArraySimulator {
    channels: Vec<MoxChannelModel>,
    response_state: Vec<f32>,
    calibration_id: String,
}

impl MoxArraySimulator {
    pub fn new(channels: Vec<MoxChannelModel>) -> Self {
        let response_state = vec![0.0; channels.len()];
        Self {
            channels,
            response_state,
            calibration_id: "mox-sim-v1".into(),
        }
    }

    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    pub fn set_calibration_id(&mut self, id: impl Into<String>) {
        self.calibration_id = id.into();
    }

    fn validate_step(
        &self,
        stimulus: &OlfactoryStimulus,
        dt_s: f32,
    ) -> Result<(), OlfactorySimulationError> {
        if !dt_s.is_finite() || dt_s < 0.0 {
            return Err(OlfactorySimulationError::InvalidStepDuration(dt_s));
        }
        if !stimulus.concentration_ppm.is_finite() || stimulus.concentration_ppm < 0.0 {
            return Err(OlfactorySimulationError::InvalidConcentration(
                stimulus.concentration_ppm,
            ));
        }
        if !stimulus.temperature_c.is_finite() {
            return Err(OlfactorySimulationError::InvalidTemperature(
                stimulus.temperature_c,
            ));
        }
        if !stimulus.humidity_rh.is_finite()
            || !(0.0..=100.0).contains(&stimulus.humidity_rh)
        {
            return Err(OlfactorySimulationError::InvalidHumidity(
                stimulus.humidity_rh,
            ));
        }
        if stimulus.affinities.len() != self.channels.len() {
            return Err(OlfactorySimulationError::AffinityCountMismatch {
                expected: self.channels.len(),
                actual: stimulus.affinities.len(),
            });
        }
        for (index, &affinity) in stimulus.affinities.iter().enumerate() {
            if !affinity.is_finite() || affinity < 0.0 {
                return Err(OlfactorySimulationError::InvalidAffinity {
                    index,
                    value: affinity,
                });
            }
        }
        if let Some(channel) = self.channels.iter().find(|channel| !channel.is_valid()) {
            return Err(OlfactorySimulationError::InvalidChannelModel(
                channel.name.clone(),
            ));
        }
        Ok(())
    }

    /// Advance the simulated sensor array and emit one raw observation.
    ///
    /// Validation completes before any temporal state is mutated, so malformed
    /// stimuli cannot partially advance the simulated nose.
    pub fn step(
        &mut self,
        stimulus: &OlfactoryStimulus,
        dt_s: f32,
        timestamp_us: u64,
    ) -> Result<ChemicalObservation, OlfactorySimulationError> {
        self.validate_step(stimulus, dt_s)?;

        let concentration = stimulus.concentration_ppm;
        let humidity_delta = (stimulus.humidity_rh - 50.0) / 50.0;
        let mut observed = Vec::with_capacity(self.channels.len());

        for (index, channel) in self.channels.iter().enumerate() {
            let affinity = stimulus.affinities[index];

            // Log concentration captures broad MOX dynamic range without
            // pretending to be a vendor-specific transfer function.
            let chemical_drive = channel.sensitivity * affinity * concentration.ln_1p();
            let humidity_factor = (1.0 + channel.humidity_coefficient * humidity_delta).max(0.1);
            let target = chemical_drive * humidity_factor;

            let current = self.response_state[index];
            let tau = if target >= current {
                channel.rise_tau_s
            } else {
                channel.recovery_tau_s
            };
            let alpha = 1.0 - (-dt_s / tau).exp();
            let next = current + alpha * (target - current);
            self.response_state[index] = next.max(0.0);

            // Reducing resistance with increasing response is sufficient for
            // testing temporal fingerprints; real MOX polarity depends on
            // material and gas chemistry.
            let resistance = channel.baseline_ohms / (1.0 + self.response_state[index]);
            observed.push(ChemicalChannel {
                name: channel.name.clone(),
                raw_value: resistance,
                unit: MeasurementUnit::Ohms,
                calibration: CalibrationState::identity(self.calibration_id.clone()),
                health: SensorHealth::default(),
            });
        }

        Ok(ChemicalObservation::new(
            timestamp_us,
            ChemicalModality::Olfactory,
            "mox-array-simulator",
            observed,
        )
        .with_environment(EnvironmentReading {
            temperature_c: Some(stimulus.temperature_c),
            humidity_rh: Some(stimulus.humidity_rh),
            pressure_pa: None,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simulator() -> MoxArraySimulator {
        MoxArraySimulator::new(vec![
            MoxChannelModel::new("mox-a", 100_000.0, 1.0),
            MoxChannelModel::new("mox-b", 120_000.0, 0.5),
        ])
    }

    #[test]
    fn odor_changes_cross_sensitive_array() {
        let mut sim = simulator();
        let clean = sim
            .step(&OlfactoryStimulus::clean_air(2, 25.0, 50.0), 1.0, 0)
            .unwrap();
        let odor = sim
            .step(
                &OlfactoryStimulus {
                    concentration_ppm: 10.0,
                    affinities: vec![1.0, 0.25],
                    temperature_c: 25.0,
                    humidity_rh: 50.0,
                },
                1.0,
                1_000_000,
            )
            .unwrap();

        assert!(odor.channels[0].raw_value < clean.channels[0].raw_value);
        assert!(odor.channels[1].raw_value < clean.channels[1].raw_value);
        assert_eq!(odor.modality, ChemicalModality::Olfactory);
    }

    #[test]
    fn one_time_constant_matches_analytical_fixture_value() {
        let mut sim = MoxArraySimulator::new(vec![MoxChannelModel::new(
            "mox-reference",
            100_000.0,
            1.0,
        )]);
        let observation = sim
            .step(
                &OlfactoryStimulus {
                    // ln(1 + concentration) = 1, so target response = 1 at RH 50%.
                    concentration_ppm: std::f32::consts::E - 1.0,
                    affinities: vec![1.0],
                    temperature_c: 25.0,
                    humidity_rh: 50.0,
                },
                1.0,
                0,
            )
            .unwrap();

        // Starting from zero with tau=1 s and dt=1 s gives response
        // 1 - exp(-1), hence R = 100000 / (2 - exp(-1)) = 61269.98 ohm.
        assert!((observation.channels[0].raw_value - 61_269.98).abs() < 1.0);
    }

    #[test]
    fn response_has_memory_and_recovers_over_time() {
        let mut sim = simulator();
        let odor = OlfactoryStimulus {
            concentration_ppm: 20.0,
            affinities: vec![1.0, 1.0],
            temperature_c: 25.0,
            humidity_rh: 50.0,
        };
        let exposed = sim.step(&odor, 2.0, 0).unwrap();
        let purge = OlfactoryStimulus::clean_air(2, 25.0, 50.0);
        let early = sim.step(&purge, 0.1, 100_000).unwrap();
        let later = sim.step(&purge, 10.0, 10_100_000).unwrap();

        assert!(early.channels[0].raw_value < later.channels[0].raw_value);
        assert!(exposed.channels[0].raw_value < later.channels[0].raw_value);
    }

    #[test]
    fn humidity_changes_response_without_changing_odor_identity_fixture() {
        let odor = |humidity_rh| OlfactoryStimulus {
            concentration_ppm: 5.0,
            affinities: vec![1.0, 0.5],
            temperature_c: 25.0,
            humidity_rh,
        };
        let mut dry_sim = simulator();
        let mut humid_sim = simulator();
        let dry = dry_sim.step(&odor(20.0), 2.0, 0).unwrap();
        let humid = humid_sim.step(&odor(80.0), 2.0, 0).unwrap();
        assert_ne!(dry.channels[0].raw_value, humid.channels[0].raw_value);
    }

    #[test]
    fn affinity_count_mismatch_is_rejected() {
        let mut sim = simulator();
        let stimulus = OlfactoryStimulus {
            concentration_ppm: 5.0,
            affinities: vec![1.0],
            temperature_c: 25.0,
            humidity_rh: 50.0,
        };
        assert!(matches!(
            sim.step(&stimulus, 1.0, 0),
            Err(OlfactorySimulationError::AffinityCountMismatch { .. })
        ));
    }

    #[test]
    fn invalid_stimulus_does_not_mutate_temporal_state() {
        let mut tested = simulator();
        let mut control = simulator();
        let bad = OlfactoryStimulus {
            concentration_ppm: f32::NAN,
            affinities: vec![1.0, 1.0],
            temperature_c: 25.0,
            humidity_rh: 50.0,
        };
        assert!(tested.step(&bad, 1.0, 0).is_err());

        let good = OlfactoryStimulus {
            concentration_ppm: 10.0,
            affinities: vec![1.0, 0.5],
            temperature_c: 25.0,
            humidity_rh: 50.0,
        };
        let after_error = tested.step(&good, 1.0, 1).unwrap();
        let untouched = control.step(&good, 1.0, 1).unwrap();
        assert_eq!(after_error, untouched);
    }

    #[test]
    fn invalid_step_duration_is_rejected() {
        let mut sim = simulator();
        let stimulus = OlfactoryStimulus::clean_air(2, 25.0, 50.0);
        assert!(matches!(
            sim.step(&stimulus, f32::NAN, 0),
            Err(OlfactorySimulationError::InvalidStepDuration(_))
        ));
    }
}
