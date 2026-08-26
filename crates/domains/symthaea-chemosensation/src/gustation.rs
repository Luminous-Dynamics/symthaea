// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hardware-independent electronic-tongue simulator.
//!
//! The simulator combines direct pH/conductivity channels with cross-sensitive
//! potentiometric electrodes. Electrode voltage uses Symthaea's existing
//! biophysics Nernst implementation rather than duplicating electrochemical
//! physics. It is a test fixture, not a calibrated food-analysis instrument.

use crate::{
    CalibrationState, ChemicalChannel, ChemicalModality, ChemicalObservation,
    EnvironmentReading, MeasurementUnit, SensorHealth,
};
use symthaea_core::physics::BiophysicsEncoder;

#[derive(Debug, Clone)]
pub struct PotentiometricChannelModel {
    pub name: String,
    /// Ionic charge used in the Nernst term. Must be non-zero.
    pub valence: i32,
    /// Cross-sensitivity weights over the stimulus ion-activity vector.
    pub selectivity: Vec<f32>,
    pub reference_mv: f32,
}

impl PotentiometricChannelModel {
    pub fn new(name: impl Into<String>, valence: i32, selectivity: Vec<f32>) -> Self {
        assert_ne!(valence, 0, "potentiometric channel valence must be non-zero");
        Self {
            name: name.into(),
            valence,
            selectivity,
            reference_mv: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GustatoryStimulus {
    pub ph: f32,
    pub conductivity_s_m: f32,
    /// Normalized effective ion activities for the simulator's latent species.
    pub ion_activities: Vec<f32>,
    pub temperature_c: f32,
}

#[derive(Debug, Clone)]
pub struct ElectronicTongueSimulator {
    electrodes: Vec<PotentiometricChannelModel>,
    calibration_id: String,
}

impl ElectronicTongueSimulator {
    pub fn new(electrodes: Vec<PotentiometricChannelModel>) -> Self {
        Self {
            electrodes,
            calibration_id: "tongue-sim-v1".into(),
        }
    }

    pub fn electrode_count(&self) -> usize {
        self.electrodes.len()
    }

    pub fn set_calibration_id(&mut self, id: impl Into<String>) {
        self.calibration_id = id.into();
    }

    pub fn sample(&self, stimulus: &GustatoryStimulus, timestamp_us: u64) -> ChemicalObservation {
        let calibration = || CalibrationState::identity(self.calibration_id.clone());
        let mut channels = Vec::with_capacity(self.electrodes.len() + 2);

        channels.push(ChemicalChannel {
            name: "ph".into(),
            raw_value: stimulus.ph,
            unit: MeasurementUnit::Ph,
            calibration: calibration(),
            health: SensorHealth::default(),
        });
        channels.push(ChemicalChannel {
            name: "conductivity".into(),
            raw_value: stimulus.conductivity_s_m.max(0.0),
            unit: MeasurementUnit::SiemensPerMeter,
            calibration: calibration(),
            health: SensorHealth::default(),
        });

        let temperature_k = stimulus.temperature_c + 273.15;
        for electrode in &self.electrodes {
            let activity = electrode
                .selectivity
                .iter()
                .enumerate()
                .map(|(i, weight)| {
                    weight.max(0.0)
                        * stimulus
                            .ion_activities
                            .get(i)
                            .copied()
                            .unwrap_or(0.0)
                            .max(0.0)
                })
                .sum::<f32>()
                .max(1e-9);

            // Treat the reference side as unit activity. This yields a
            // deterministic cross-sensitive electrode response suitable for
            // testing concentration, temperature, and mixture effects.
            let nernst_mv = BiophysicsEncoder::nernst_potential_mv(
                electrode.valence,
                activity as f64,
                1.0,
                temperature_k as f64,
            ) as f32;

            channels.push(ChemicalChannel {
                name: electrode.name.clone(),
                raw_value: electrode.reference_mv + nernst_mv,
                unit: MeasurementUnit::Millivolts,
                calibration: calibration(),
                health: SensorHealth::default(),
            });
        }

        ChemicalObservation::new(
            timestamp_us,
            ChemicalModality::Gustatory,
            "electronic-tongue-simulator",
            channels,
        )
        .with_environment(EnvironmentReading {
            temperature_c: Some(stimulus.temperature_c),
            humidity_rh: None,
            pressure_pa: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simulator() -> ElectronicTongueSimulator {
        ElectronicTongueSimulator::new(vec![
            PotentiometricChannelModel::new("ion-a", 1, vec![1.0, 0.1]),
            PotentiometricChannelModel::new("ion-b", -1, vec![0.1, 1.0]),
        ])
    }

    #[test]
    fn sample_preserves_direct_ph_and_conductivity_channels() {
        let observation = simulator().sample(
            &GustatoryStimulus {
                ph: 4.2,
                conductivity_s_m: 1.7,
                ion_activities: vec![0.1, 0.2],
                temperature_c: 25.0,
            },
            7,
        );

        assert_eq!(observation.modality, ChemicalModality::Gustatory);
        assert!((observation.channels[0].raw_value - 4.2).abs() < 1e-6);
        assert!((observation.channels[1].raw_value - 1.7).abs() < 1e-6);
    }

    #[test]
    fn changing_ion_activity_changes_cross_sensitive_voltage() {
        let sim = simulator();
        let low = sim.sample(
            &GustatoryStimulus {
                ph: 7.0,
                conductivity_s_m: 1.0,
                ion_activities: vec![0.01, 0.01],
                temperature_c: 25.0,
            },
            0,
        );
        let high = sim.sample(
            &GustatoryStimulus {
                ph: 7.0,
                conductivity_s_m: 1.0,
                ion_activities: vec![0.5, 0.01],
                temperature_c: 25.0,
            },
            1,
        );

        assert_ne!(low.channels[2].raw_value, high.channels[2].raw_value);
    }

    #[test]
    fn temperature_changes_nernst_response() {
        let sim = simulator();
        let stimulus = |temperature_c| GustatoryStimulus {
            ph: 7.0,
            conductivity_s_m: 1.0,
            ion_activities: vec![0.1, 0.1],
            temperature_c,
        };
        let cold = sim.sample(&stimulus(5.0), 0);
        let warm = sim.sample(&stimulus(40.0), 1);
        assert_ne!(cold.channels[2].raw_value, warm.channels[2].raw_value);
    }
}
