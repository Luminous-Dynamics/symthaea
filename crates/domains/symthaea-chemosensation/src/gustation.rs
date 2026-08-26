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

#[derive(Debug, Clone, PartialEq)]
pub enum GustatorySimulationError {
    InvalidPh(f32),
    InvalidConductivity(f32),
    InvalidTemperature(f32),
    InvalidIonActivity { index: usize, value: f32 },
    SpeciesCountMismatch {
        electrode: String,
        expected: usize,
        actual: usize,
    },
    InvalidElectrodeModel(String),
}

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
        assert!(
            !selectivity.is_empty()
                && selectivity
                    .iter()
                    .all(|weight| weight.is_finite() && *weight >= 0.0),
            "potentiometric selectivity must be non-empty, finite, and non-negative"
        );
        Self {
            name: name.into(),
            valence,
            selectivity,
            reference_mv: 0.0,
        }
    }

    fn is_valid(&self) -> bool {
        self.valence != 0
            && self.reference_mv.is_finite()
            && !self.selectivity.is_empty()
            && self
                .selectivity
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0)
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

    fn validate_stimulus(
        &self,
        stimulus: &GustatoryStimulus,
    ) -> Result<(), GustatorySimulationError> {
        if !stimulus.ph.is_finite() {
            return Err(GustatorySimulationError::InvalidPh(stimulus.ph));
        }
        if !stimulus.conductivity_s_m.is_finite() || stimulus.conductivity_s_m < 0.0 {
            return Err(GustatorySimulationError::InvalidConductivity(
                stimulus.conductivity_s_m,
            ));
        }
        if !stimulus.temperature_c.is_finite() || stimulus.temperature_c + 273.15 <= 0.0 {
            return Err(GustatorySimulationError::InvalidTemperature(
                stimulus.temperature_c,
            ));
        }
        for (index, &activity) in stimulus.ion_activities.iter().enumerate() {
            if !activity.is_finite() || activity < 0.0 {
                return Err(GustatorySimulationError::InvalidIonActivity {
                    index,
                    value: activity,
                });
            }
        }
        for electrode in &self.electrodes {
            if !electrode.is_valid() {
                return Err(GustatorySimulationError::InvalidElectrodeModel(
                    electrode.name.clone(),
                ));
            }
            if electrode.selectivity.len() != stimulus.ion_activities.len() {
                return Err(GustatorySimulationError::SpeciesCountMismatch {
                    electrode: electrode.name.clone(),
                    expected: electrode.selectivity.len(),
                    actual: stimulus.ion_activities.len(),
                });
            }
        }
        Ok(())
    }

    pub fn sample(
        &self,
        stimulus: &GustatoryStimulus,
        timestamp_us: u64,
    ) -> Result<ChemicalObservation, GustatorySimulationError> {
        self.validate_stimulus(stimulus)?;

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
            raw_value: stimulus.conductivity_s_m,
            unit: MeasurementUnit::SiemensPerMeter,
            calibration: calibration(),
            health: SensorHealth::default(),
        });

        let temperature_k = stimulus.temperature_c + 273.15;
        for electrode in &self.electrodes {
            let activity = electrode
                .selectivity
                .iter()
                .zip(&stimulus.ion_activities)
                .map(|(weight, ion_activity)| weight * ion_activity)
                .sum::<f32>()
                .max(1e-9);

            // Treat the reference side as unit activity. The 1e-9 floor is a
            // simulator detection floor that prevents log(0), not an inferred
            // chemical identity.
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

        Ok(ChemicalObservation::new(
            timestamp_us,
            ChemicalModality::Gustatory,
            "electronic-tongue-simulator",
            channels,
        )
        .with_environment(EnvironmentReading {
            temperature_c: Some(stimulus.temperature_c),
            humidity_rh: None,
            pressure_pa: None,
        }))
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
        let observation = simulator()
            .sample(
                &GustatoryStimulus {
                    ph: 4.2,
                    conductivity_s_m: 1.7,
                    ion_activities: vec![0.1, 0.2],
                    temperature_c: 25.0,
                },
                7,
            )
            .unwrap();

        assert_eq!(observation.modality, ChemicalModality::Gustatory);
        assert!((observation.channels[0].raw_value - 4.2).abs() < 1e-6);
        assert!((observation.channels[1].raw_value - 1.7).abs() < 1e-6);
    }

    #[test]
    fn changing_ion_activity_changes_cross_sensitive_voltage() {
        let sim = simulator();
        let low = sim
            .sample(
                &GustatoryStimulus {
                    ph: 7.0,
                    conductivity_s_m: 1.0,
                    ion_activities: vec![0.01, 0.01],
                    temperature_c: 25.0,
                },
                0,
            )
            .unwrap();
        let high = sim
            .sample(
                &GustatoryStimulus {
                    ph: 7.0,
                    conductivity_s_m: 1.0,
                    ion_activities: vec![0.5, 0.01],
                    temperature_c: 25.0,
                },
                1,
            )
            .unwrap();

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
        let cold = sim.sample(&stimulus(5.0), 0).unwrap();
        let warm = sim.sample(&stimulus(40.0), 1).unwrap();
        assert_ne!(cold.channels[2].raw_value, warm.channels[2].raw_value);
    }

    #[test]
    fn invalid_temperature_is_rejected() {
        let sim = simulator();
        let stimulus = GustatoryStimulus {
            ph: 7.0,
            conductivity_s_m: 1.0,
            ion_activities: vec![0.1, 0.1],
            temperature_c: -273.15,
        };
        assert!(matches!(
            sim.sample(&stimulus, 0),
            Err(GustatorySimulationError::InvalidTemperature(_))
        ));
    }

    #[test]
    fn species_count_mismatch_is_rejected() {
        let sim = simulator();
        let stimulus = GustatoryStimulus {
            ph: 7.0,
            conductivity_s_m: 1.0,
            ion_activities: vec![0.1],
            temperature_c: 25.0,
        };
        assert!(matches!(
            sim.sample(&stimulus, 0),
            Err(GustatorySimulationError::SpeciesCountMismatch { .. })
        ));
    }

    #[test]
    fn non_finite_ion_activity_is_rejected() {
        let sim = simulator();
        let stimulus = GustatoryStimulus {
            ph: 7.0,
            conductivity_s_m: 1.0,
            ion_activities: vec![0.1, f32::NAN],
            temperature_c: 25.0,
        };
        assert!(matches!(
            sim.sample(&stimulus, 0),
            Err(GustatorySimulationError::InvalidIonActivity { index: 1, .. })
        ));
    }
}
