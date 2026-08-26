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
        Self {
            name: name.into(),
            baseline_ohms: baseline_ohms.max(1.0),
            sensitivity: sensitivity.max(0.0),
            humidity_coefficient: 0.1,
            rise_tau_s: 1.0,
            recovery_tau_s: 3.0,
        }
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

    /// Advance the simulated sensor array and emit one raw observation.
    pub fn step(
        &mut self,
        stimulus: &OlfactoryStimulus,
        dt_s: f32,
        timestamp_us: u64,
    ) -> ChemicalObservation {
        let concentration = stimulus.concentration_ppm.max(0.0);
        let humidity_delta = (stimulus.humidity_rh.clamp(0.0, 100.0) - 50.0) / 50.0;
        let dt_s = dt_s.max(0.0);

        let mut observed = Vec::with_capacity(self.channels.len());

        for (index, channel) in self.channels.iter().enumerate() {
            let affinity = stimulus
                .affinities
                .get(index)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);

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
            }
            .max(1e-3);
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

        ChemicalObservation::new(
            timestamp_us,
            ChemicalModality::Olfactory,
            "mox-array-simulator",
            observed,
        )
        .with_environment(EnvironmentReading {
            temperature_c: Some(stimulus.temperature_c),
            humidity_rh: Some(stimulus.humidity_rh),
            pressure_pa: None,
        })
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
        let clean = sim.step(&OlfactoryStimulus::clean_air(2, 25.0, 50.0), 1.0, 0);
        let odor = sim.step(
            &OlfactoryStimulus {
                concentration_ppm: 10.0,
                affinities: vec![1.0, 0.25],
                temperature_c: 25.0,
                humidity_rh: 50.0,
            },
            1.0,
            1_000_000,
        );

        assert!(odor.channels[0].raw_value < clean.channels[0].raw_value);
        assert!(odor.channels[1].raw_value < clean.channels[1].raw_value);
        assert_eq!(odor.modality, ChemicalModality::Olfactory);
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
        let exposed = sim.step(&odor, 2.0, 0);
        let purge = OlfactoryStimulus::clean_air(2, 25.0, 50.0);
        let early = sim.step(&purge, 0.1, 100_000);
        let later = sim.step(&purge, 10.0, 10_100_000);

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
        let dry = dry_sim.step(&odor(20.0), 2.0, 0);
        let humid = humid_sim.step(&odor(80.0), 2.0, 0);
        assert_ne!(dry.channels[0].raw_value, humid.channels[0].raw_value);
    }
}
