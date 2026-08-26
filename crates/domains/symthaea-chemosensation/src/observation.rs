// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed raw observations shared by olfaction and gustation.

use crate::{CalibrationState, ChemicalClockDomainId, SensorHealth};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChemicalModality {
    Olfactory,
    Gustatory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeasurementUnit {
    /// Raw sensor-native value when no physical calibration is available yet.
    Arbitrary,
    PartsPerMillion,
    PartsPerBillion,
    Ohms,
    SiemensPerMeter,
    Millivolts,
    Ph,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChemicalChannel {
    pub name: String,
    /// Raw, uncorrected measurement preserved for evidence/provenance.
    pub raw_value: f32,
    pub unit: MeasurementUnit,
    pub calibration: CalibrationState,
    pub health: SensorHealth,
}

impl ChemicalChannel {
    /// Calibrated value when both measurement and calibration are valid.
    pub fn calibrated_value(&self) -> Option<f32> {
        self.calibration.apply(self.raw_value)
    }

    /// Confidence contribution in [0, 1]. A channel without a valid calibrated
    /// measurement contributes no confidence, regardless of health metadata.
    pub fn effective_confidence(&self) -> f32 {
        if self.calibrated_value().is_none() {
            return 0.0;
        }
        self.health.confidence_factor() * (1.0 - self.calibration.normalized_drift())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct EnvironmentReading {
    pub temperature_c: Option<f32>,
    pub humidity_rh: Option<f32>,
    pub pressure_pa: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChemicalObservation {
    /// Timestamp expressed in microseconds in `clock_domain`, when one is known.
    /// Without a clock domain this value is still preserved evidence, but it must
    /// not be compared across independent sources as though synchronization were
    /// established.
    pub timestamp_us: u64,
    /// Identity of the timebase against which `timestamp_us` is expressed.
    ///
    /// `None` means unspecified, not Unix time. The ID names a comparison domain
    /// only; it does not prove synchronization quality or authenticity.
    ///
    /// The serde default intentionally maps legacy wire observations that predate
    /// this field to `None`; it never upgrades them into an assumed epoch.
    #[serde(default)]
    pub clock_domain: Option<ChemicalClockDomainId>,
    pub modality: ChemicalModality,
    pub source: String,
    pub channels: Vec<ChemicalChannel>,
    pub environment: EnvironmentReading,
}

impl ChemicalObservation {
    pub fn new(
        timestamp_us: u64,
        modality: ChemicalModality,
        source: impl Into<String>,
        channels: Vec<ChemicalChannel>,
    ) -> Self {
        Self {
            timestamp_us,
            clock_domain: None,
            modality,
            source: source.into(),
            channels,
            environment: EnvironmentReading::default(),
        }
    }

    pub fn with_clock_domain(mut self, clock_domain: ChemicalClockDomainId) -> Self {
        self.clock_domain = Some(clock_domain);
        self
    }

    pub fn with_environment(mut self, environment: EnvironmentReading) -> Self {
        self.environment = environment;
        self
    }

    /// Mean confidence across available channels. Empty observations are not
    /// treated as confident evidence.
    pub fn mean_confidence(&self) -> f32 {
        if self.channels.is_empty() {
            return 0.0;
        }
        self.channels
            .iter()
            .map(ChemicalChannel::effective_confidence)
            .sum::<f32>()
            / self.channels.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CalibrationId, CalibrationState, SensorHealth};

    fn channel(raw: f32) -> ChemicalChannel {
        ChemicalChannel {
            name: "voc".into(),
            raw_value: raw,
            unit: MeasurementUnit::Arbitrary,
            calibration: CalibrationState {
                id: CalibrationId::new("test"),
                baseline: 1.0,
                gain: 2.0,
                drift: 0.1,
            },
            health: SensorHealth::default(),
        }
    }

    #[test]
    fn raw_measurement_is_preserved_while_calibrated_view_is_derived() {
        let c = channel(3.0);
        assert!((c.raw_value - 3.0).abs() < 1e-6);
        assert!((c.calibrated_value().unwrap() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn corrupt_raw_measurement_does_not_gain_a_calibrated_value() {
        assert!(channel(f32::NAN).calibrated_value().is_none());
    }

    #[test]
    fn corrupt_raw_measurement_has_zero_confidence() {
        assert_eq!(channel(f32::NAN).effective_confidence(), 0.0);
    }

    #[test]
    fn observation_confidence_accounts_for_drift() {
        let observation = ChemicalObservation::new(
            42,
            ChemicalModality::Olfactory,
            "simulated-nose",
            vec![channel(3.0)],
        );
        assert!((observation.mean_confidence() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn new_observations_do_not_imply_a_clock_domain() {
        let observation = ChemicalObservation::new(
            42,
            ChemicalModality::Olfactory,
            "simulated-nose",
            vec![channel(3.0)],
        );
        assert!(observation.clock_domain.is_none());

        let clock = ChemicalClockDomainId::new("fixture/monotonic").unwrap();
        let clocked = observation.with_clock_domain(clock.clone());
        assert_eq!(clocked.clock_domain.as_ref(), Some(&clock));
    }

    #[test]
    fn legacy_wire_observation_defaults_clock_to_unspecified() {
        let legacy = serde_json::json!({
            "timestamp_us": 42,
            "modality": "Olfactory",
            "source": "legacy-nose",
            "channels": [],
            "environment": {
                "temperature_c": null,
                "humidity_rh": null,
                "pressure_pa": null
            }
        });
        let decoded: ChemicalObservation = serde_json::from_value(legacy).unwrap();
        assert_eq!(decoded.timestamp_us, 42);
        assert_eq!(decoded.source, "legacy-nose");
        assert!(decoded.clock_domain.is_none());
    }

    #[test]
    fn empty_observation_has_zero_confidence() {
        let observation = ChemicalObservation::new(
            42,
            ChemicalModality::Gustatory,
            "simulated-tongue",
            vec![],
        );
        assert_eq!(observation.mean_confidence(), 0.0);
    }
}
