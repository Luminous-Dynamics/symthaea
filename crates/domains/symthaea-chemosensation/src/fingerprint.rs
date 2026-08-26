// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Role-bound HDC fingerprints for calibrated chemical observations.

use std::collections::HashMap;

use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

use crate::{ChemicalModality, ChemicalObservation, ScalarHdcEncoder};

#[derive(Debug, Clone)]
pub struct ChannelEncodingSpec {
    pub name: String,
    pub scalar: ScalarHdcEncoder,
    role: ContinuousHV,
}

impl ChannelEncodingSpec {
    pub fn new(
        name: impl Into<String>,
        min: f32,
        max: f32,
        anchor_count: usize,
        scalar_seed: u64,
        role_seed: u64,
    ) -> Self {
        Self {
            name: name.into(),
            scalar: ScalarHdcEncoder::new(min, max, anchor_count, scalar_seed),
            role: ContinuousHV::random(HDC_DIMENSION, role_seed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChemicalFingerprintEncoder {
    specs: HashMap<String, ChannelEncodingSpec>,
    olfactory_role: ContinuousHV,
    gustatory_role: ContinuousHV,
}

impl ChemicalFingerprintEncoder {
    pub fn new(specs: Vec<ChannelEncodingSpec>) -> Self {
        Self {
            specs: specs
                .into_iter()
                .map(|spec| (spec.name.clone(), spec))
                .collect(),
            olfactory_role: ContinuousHV::random(HDC_DIMENSION, 0x0F1A_C700_0000_0001),
            gustatory_role: ContinuousHV::random(HDC_DIMENSION, 0x0F1A_C700_0000_0002),
        }
    }

    pub fn configured_channels(&self) -> usize {
        self.specs.len()
    }

    /// Encode calibrated channel values into a modality-bound chemical
    /// fingerprint. Raw observations remain unchanged and should be retained as
    /// evidence/provenance alongside this derived representation.
    ///
    /// Channels without an encoding spec are ignored. Returns `None` if no
    /// configured channel is present in the observation.
    pub fn encode(&self, observation: &ChemicalObservation) -> Option<ContinuousHV> {
        let mut channels: Vec<_> = observation
            .channels
            .iter()
            .filter_map(|channel| self.specs.get(&channel.name).map(|spec| (channel, spec)))
            .collect();

        // Make the fingerprint independent of hardware/driver channel ordering.
        channels.sort_by(|(a, _), (b, _)| a.name.cmp(&b.name));

        let bound: Vec<ContinuousHV> = channels
            .into_iter()
            .map(|(channel, spec)| {
                let value_hv = spec.scalar.encode(channel.calibrated_value());
                spec.role.bind(&value_hv)
            })
            .collect();

        if bound.is_empty() {
            return None;
        }

        let refs: Vec<&ContinuousHV> = bound.iter().collect();
        let bundled = ContinuousHV::bundle(&refs);
        let modality_role = match observation.modality {
            ChemicalModality::Olfactory => &self.olfactory_role,
            ChemicalModality::Gustatory => &self.gustatory_role,
        };

        let mut fingerprint = modality_role.bind(&bundled);
        fingerprint.l2_normalize();
        Some(fingerprint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CalibrationState, ChemicalChannel, ChemicalObservation, MeasurementUnit, SensorHealth,
    };

    fn channel(name: &str, raw: f32) -> ChemicalChannel {
        ChemicalChannel {
            name: name.into(),
            raw_value: raw,
            unit: MeasurementUnit::Arbitrary,
            calibration: CalibrationState::identity("test"),
            health: SensorHealth::default(),
        }
    }

    fn encoder() -> ChemicalFingerprintEncoder {
        ChemicalFingerprintEncoder::new(vec![
            ChannelEncodingSpec::new("a", 0.0, 100.0, 16, 11, 101),
            ChannelEncodingSpec::new("b", 0.0, 100.0, 16, 12, 102),
        ])
    }

    #[test]
    fn encoding_is_deterministic_and_order_invariant() {
        let encoder = encoder();
        let a = ChemicalObservation::new(
            0,
            ChemicalModality::Olfactory,
            "sensor",
            vec![channel("a", 20.0), channel("b", 40.0)],
        );
        let b = ChemicalObservation::new(
            1,
            ChemicalModality::Olfactory,
            "sensor",
            vec![channel("b", 40.0), channel("a", 20.0)],
        );

        assert_eq!(encoder.encode(&a), encoder.encode(&b));
    }

    #[test]
    fn nearby_chemistry_is_more_similar_than_distant_chemistry() {
        let encoder = ChemicalFingerprintEncoder::new(vec![ChannelEncodingSpec::new(
            "a", 0.0, 100.0, 16, 11, 101,
        )]);
        let observation = |value| {
            ChemicalObservation::new(
                0,
                ChemicalModality::Olfactory,
                "sensor",
                vec![channel("a", value)],
            )
        };

        let center = encoder.encode(&observation(50.0)).unwrap();
        let near = encoder.encode(&observation(51.0)).unwrap();
        let far = encoder.encode(&observation(90.0)).unwrap();
        assert!(center.similarity(&near) > center.similarity(&far));
    }

    #[test]
    fn modalities_remain_distinguishable_for_same_measurements() {
        let encoder = encoder();
        let odor = ChemicalObservation::new(
            0,
            ChemicalModality::Olfactory,
            "sensor",
            vec![channel("a", 20.0)],
        );
        let taste = ChemicalObservation::new(
            0,
            ChemicalModality::Gustatory,
            "sensor",
            vec![channel("a", 20.0)],
        );

        let odor_hv = encoder.encode(&odor).unwrap();
        let taste_hv = encoder.encode(&taste).unwrap();
        assert!(odor_hv.similarity(&taste_hv) < 0.5);
    }

    #[test]
    fn unknown_channels_do_not_create_fake_percepts() {
        let encoder = encoder();
        let observation = ChemicalObservation::new(
            0,
            ChemicalModality::Olfactory,
            "sensor",
            vec![channel("unknown", 20.0)],
        );
        assert!(encoder.encode(&observation).is_none());
    }
}
