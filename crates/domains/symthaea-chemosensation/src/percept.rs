// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-preserving chemical percepts.
//!
//! A [`ChemicalPercept`] is a derived cognitive representation paired with the
//! exact [`ChemicalObservation`] that produced it. Learned labels and later
//! semantic hypotheses must not replace this evidence object.

use crate::{
    ChemicalFingerprint, ChemicalFingerprintEncoder, ChemicalObservation, FingerprintError,
};

/// A cognitive-ready chemical representation with its source evidence intact.
#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalPercept {
    /// Exact raw observation used to derive the fingerprint.
    pub evidence: ChemicalObservation,
    /// Derived, calibrated HDC representation plus quality metadata.
    pub fingerprint: ChemicalFingerprint,
}

impl ChemicalPercept {
    /// Evidence timestamp, preserved from the transducer observation.
    pub fn timestamp_us(&self) -> u64 {
        self.evidence.timestamp_us
    }

    /// Effective percept confidence inherited from the usable sensor channels.
    pub fn confidence(&self) -> f32 {
        self.fingerprint.confidence
    }
}

/// Thin boundary from physical evidence to a cognitive-ready percept.
#[derive(Debug, Clone)]
pub struct ChemicalPerceptEncoder {
    fingerprint_encoder: ChemicalFingerprintEncoder,
}

impl ChemicalPerceptEncoder {
    pub fn new(fingerprint_encoder: ChemicalFingerprintEncoder) -> Self {
        Self {
            fingerprint_encoder,
        }
    }

    pub fn fingerprint_encoder(&self) -> &ChemicalFingerprintEncoder {
        &self.fingerprint_encoder
    }

    /// Derive a percept without mutating or replacing the source observation.
    ///
    /// `Ok(None)` means no configured, trustworthy channel was available. It is
    /// absence of usable evidence, not a zero-valued chemical percept.
    pub fn encode(
        &self,
        observation: &ChemicalObservation,
    ) -> Result<Option<ChemicalPercept>, FingerprintError> {
        let Some(fingerprint) = self.fingerprint_encoder.encode(observation)? else {
            return Ok(None);
        };

        Ok(Some(ChemicalPercept {
            evidence: observation.clone(),
            fingerprint,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CalibrationState, ChannelEncodingSpec, ChemicalChannel, ChemicalModality,
        MeasurementUnit, SensorHealth,
    };

    fn encoder() -> ChemicalPerceptEncoder {
        let fingerprint_encoder = ChemicalFingerprintEncoder::new(vec![ChannelEncodingSpec::new(
            "voc",
            MeasurementUnit::PartsPerMillion,
            0.0,
            100.0,
            16,
            11,
            101,
        )])
        .unwrap();
        ChemicalPerceptEncoder::new(fingerprint_encoder)
    }

    fn observation(raw_value: f32) -> ChemicalObservation {
        ChemicalObservation::new(
            123,
            ChemicalModality::Olfactory,
            "nose-a",
            vec![ChemicalChannel {
                name: "voc".into(),
                raw_value,
                unit: MeasurementUnit::PartsPerMillion,
                calibration: CalibrationState::identity("cal-a"),
                health: SensorHealth::default(),
            }],
        )
    }

    #[test]
    fn percept_preserves_exact_source_evidence() {
        let observation = observation(12.5);
        let percept = encoder().encode(&observation).unwrap().unwrap();

        assert_eq!(percept.evidence, observation);
        assert_eq!(percept.timestamp_us(), 123);
        assert_eq!(percept.evidence.source, "nose-a");
        assert_eq!(percept.fingerprint.used_channels, 1);
    }

    #[test]
    fn no_usable_channel_is_absence_not_zero_percept() {
        let mut observation = observation(12.5);
        observation.channels[0].health.score = 0.0;
        assert!(encoder().encode(&observation).unwrap().is_none());
    }

    #[test]
    fn corrupt_configured_measurement_remains_an_integrity_error() {
        let observation = observation(f32::NAN);
        assert!(matches!(
            encoder().encode(&observation),
            Err(FingerprintError::InvalidMeasurement(name)) if name == "voc"
        ));
    }
}
