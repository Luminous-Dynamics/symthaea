// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Role-bound HDC fingerprints for calibrated chemical observations.

use std::collections::{HashMap, HashSet};
use std::fmt;

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::{HDC_DIMENSION, unified_hv::ContinuousHV};

use crate::{ChemicalModality, ChemicalObservation, MeasurementUnit, ScalarHdcEncoder};

/// Content identity of the complete chemical HDC coordinate system.
///
/// The digest covers channel names/units/ranges, every scalar anchor vector,
/// every channel-role vector, and both modality-role vectors. Two fingerprints
/// may therefore be compared geometrically only when their space IDs match.
///
/// This is representation identity, not an authenticity signature. It tells
/// downstream code whether two vectors were produced in the same coordinate
/// system; provenance/authenticity belongs to the observation/evidence layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChemicalEncodingSpaceId(pub [u8; 32]);

impl ChemicalEncodingSpaceId {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ChemicalEncodingSpaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ChannelEncodingSpec {
    pub name: String,
    pub unit: MeasurementUnit,
    pub scalar: ScalarHdcEncoder,
    role: ContinuousHV,
}

impl ChannelEncodingSpec {
    pub fn new(
        name: impl Into<String>,
        unit: MeasurementUnit,
        min: f32,
        max: f32,
        anchor_count: usize,
        scalar_seed: u64,
        role_seed: u64,
    ) -> Self {
        Self {
            name: name.into(),
            unit,
            scalar: ScalarHdcEncoder::new(min, max, anchor_count, scalar_seed),
            role: ContinuousHV::random(HDC_DIMENSION, role_seed),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FingerprintConfigError {
    DuplicateChannelSpec(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FingerprintError {
    DuplicateChannel(String),
    UnitMismatch {
        channel: String,
        expected: MeasurementUnit,
        actual: MeasurementUnit,
    },
    InvalidMeasurement(String),
}

/// Derived chemical representation plus evidence-quality metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalFingerprint {
    pub vector: ContinuousHV,
    pub confidence: f32,
    pub used_channels: usize,
    pub ignored_channels: usize,
    /// Content-addressed identity of the coordinate system that produced
    /// `vector`. Downstream similarity/fusion must require matching IDs.
    pub encoding_space_id: ChemicalEncodingSpaceId,
}

#[derive(Debug, Clone)]
pub struct ChemicalFingerprintEncoder {
    specs: HashMap<String, ChannelEncodingSpec>,
    olfactory_role: ContinuousHV,
    gustatory_role: ContinuousHV,
    encoding_space_id: ChemicalEncodingSpaceId,
}

impl ChemicalFingerprintEncoder {
    pub fn new(specs: Vec<ChannelEncodingSpec>) -> Result<Self, FingerprintConfigError> {
        let mut by_name = HashMap::with_capacity(specs.len());
        for spec in specs {
            let name = spec.name.clone();
            if by_name.insert(name.clone(), spec).is_some() {
                return Err(FingerprintConfigError::DuplicateChannelSpec(name));
            }
        }

        let olfactory_role = ContinuousHV::random(HDC_DIMENSION, 0x0F1A_C700_0000_0001);
        let gustatory_role = ContinuousHV::random(HDC_DIMENSION, 0x0F1A_C700_0000_0002);
        let encoding_space_id =
            compute_encoding_space_id(&by_name, &olfactory_role, &gustatory_role);

        Ok(Self {
            specs: by_name,
            olfactory_role,
            gustatory_role,
            encoding_space_id,
        })
    }

    pub fn configured_channels(&self) -> usize {
        self.specs.len()
    }

    pub fn encoding_space_id(&self) -> ChemicalEncodingSpaceId {
        self.encoding_space_id
    }

    /// Encode calibrated channel values into a modality-bound chemical
    /// fingerprint. Raw observations remain unchanged and should be retained as
    /// evidence/provenance alongside this derived representation.
    ///
    /// Unknown channels are ignored for forward-compatible sensor arrays. A
    /// configured channel with a wrong unit, duplicate name, or invalid numeric
    /// measurement is an integrity error and is never silently coerced.
    pub fn encode(
        &self,
        observation: &ChemicalObservation,
    ) -> Result<Option<ChemicalFingerprint>, FingerprintError> {
        let mut channels: Vec<_> = observation.channels.iter().collect();
        channels.sort_by(|a, b| a.name.cmp(&b.name));

        let mut seen = HashSet::with_capacity(channels.len());
        let mut bound = Vec::new();
        let mut confidences = Vec::new();
        let mut ignored_channels = 0usize;

        for channel in channels {
            if !seen.insert(channel.name.as_str()) {
                return Err(FingerprintError::DuplicateChannel(channel.name.clone()));
            }

            let Some(spec) = self.specs.get(&channel.name) else {
                ignored_channels += 1;
                continue;
            };

            if channel.unit != spec.unit {
                return Err(FingerprintError::UnitMismatch {
                    channel: channel.name.clone(),
                    expected: spec.unit,
                    actual: channel.unit,
                });
            }

            let value = channel
                .calibrated_value()
                .ok_or_else(|| FingerprintError::InvalidMeasurement(channel.name.clone()))?;
            let value_hv = spec
                .scalar
                .encode(value)
                .ok_or_else(|| FingerprintError::InvalidMeasurement(channel.name.clone()))?;
            let confidence = channel.effective_confidence();

            if confidence <= 0.0 {
                ignored_channels += 1;
                continue;
            }

            bound.push(spec.role.bind(&value_hv).scale(confidence));
            confidences.push(confidence);
        }

        if bound.is_empty() {
            return Ok(None);
        }

        let refs: Vec<&ContinuousHV> = bound.iter().collect();
        let bundled = ContinuousHV::bundle(&refs);
        let modality_role = match observation.modality {
            ChemicalModality::Olfactory => &self.olfactory_role,
            ChemicalModality::Gustatory => &self.gustatory_role,
        };

        let mut vector = modality_role.bind(&bundled);
        vector.l2_normalize();
        let confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;

        Ok(Some(ChemicalFingerprint {
            vector,
            confidence,
            used_channels: confidences.len(),
            ignored_channels,
            encoding_space_id: self.encoding_space_id,
        }))
    }
}

fn compute_encoding_space_id(
    specs: &HashMap<String, ChannelEncodingSpec>,
    olfactory_role: &ContinuousHV,
    gustatory_role: &ContinuousHV,
) -> ChemicalEncodingSpaceId {
    let mut hasher = Hasher::new();
    put_bytes(&mut hasher, b"symthaea-chemosensation-encoding-space-v1");

    let mut ordered: Vec<&ChannelEncodingSpec> = specs.values().collect();
    ordered.sort_by(|left, right| left.name.cmp(&right.name));
    put_u64(&mut hasher, ordered.len() as u64);

    for spec in ordered {
        put_bytes(&mut hasher, spec.name.as_bytes());
        put_u8(&mut hasher, unit_tag(spec.unit));
        put_u32(&mut hasher, spec.scalar.min().to_bits());
        put_u32(&mut hasher, spec.scalar.max().to_bits());
        put_u64(&mut hasher, spec.scalar.anchor_count() as u64);
        for anchor in spec.scalar.anchors() {
            hash_hv(&mut hasher, anchor);
        }
        hash_hv(&mut hasher, &spec.role);
    }

    hash_hv(&mut hasher, olfactory_role);
    hash_hv(&mut hasher, gustatory_role);
    ChemicalEncodingSpaceId(*hasher.finalize().as_bytes())
}

fn hash_hv(hasher: &mut Hasher, hv: &ContinuousHV) {
    put_u64(hasher, hv.values.len() as u64);
    for value in &hv.values {
        put_u32(hasher, value.to_bits());
    }
}

fn unit_tag(unit: MeasurementUnit) -> u8 {
    match unit {
        MeasurementUnit::Arbitrary => 1,
        MeasurementUnit::PartsPerMillion => 2,
        MeasurementUnit::PartsPerBillion => 3,
        MeasurementUnit::Ohms => 4,
        MeasurementUnit::SiemensPerMeter => 5,
        MeasurementUnit::Millivolts => 6,
        MeasurementUnit::Ph => 7,
    }
}

fn put_bytes(hasher: &mut Hasher, bytes: &[u8]) {
    put_u64(hasher, bytes.len() as u64);
    hasher.update(bytes);
}

fn put_u8(hasher: &mut Hasher, value: u8) {
    hasher.update(&[value]);
}

fn put_u32(hasher: &mut Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}

fn put_u64(hasher: &mut Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CalibrationState, ChemicalChannel, ChemicalObservation, SensorHealth};

    fn channel(name: &str, raw: f32) -> ChemicalChannel {
        ChemicalChannel {
            name: name.into(),
            raw_value: raw,
            unit: MeasurementUnit::Arbitrary,
            calibration: CalibrationState::identity("test"),
            health: SensorHealth::default(),
        }
    }

    fn spec(name: &str, scalar_seed: u64, role_seed: u64) -> ChannelEncodingSpec {
        ChannelEncodingSpec::new(
            name,
            MeasurementUnit::Arbitrary,
            0.0,
            100.0,
            16,
            scalar_seed,
            role_seed,
        )
    }

    fn encoder() -> ChemicalFingerprintEncoder {
        ChemicalFingerprintEncoder::new(vec![spec("a", 11, 101), spec("b", 12, 102)])
            .unwrap()
    }

    #[test]
    fn encoding_space_identity_is_deterministic_and_spec_order_invariant() {
        let a = ChemicalFingerprintEncoder::new(vec![spec("a", 11, 101), spec("b", 12, 102)])
            .unwrap();
        let b = ChemicalFingerprintEncoder::new(vec![spec("b", 12, 102), spec("a", 11, 101)])
            .unwrap();
        assert_eq!(a.encoding_space_id(), b.encoding_space_id());
    }

    #[test]
    fn changing_actual_coordinate_system_changes_space_identity() {
        let a = ChemicalFingerprintEncoder::new(vec![spec("a", 11, 101)]).unwrap();
        let scalar_changed = ChemicalFingerprintEncoder::new(vec![spec("a", 12, 101)]).unwrap();
        let role_changed = ChemicalFingerprintEncoder::new(vec![spec("a", 11, 102)]).unwrap();
        assert_ne!(a.encoding_space_id(), scalar_changed.encoding_space_id());
        assert_ne!(a.encoding_space_id(), role_changed.encoding_space_id());
    }

    #[test]
    fn fingerprint_carries_the_encoder_space_identity() {
        let encoder = encoder();
        let observation = ChemicalObservation::new(
            0,
            ChemicalModality::Olfactory,
            "sensor",
            vec![channel("a", 20.0)],
        );
        let fingerprint = encoder.encode(&observation).unwrap().unwrap();
        assert_eq!(fingerprint.encoding_space_id, encoder.encoding_space_id());
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
        let encoder = ChemicalFingerprintEncoder::new(vec![spec("a", 11, 101)]).unwrap();
        let observation = |value| {
            ChemicalObservation::new(
                0,
                ChemicalModality::Olfactory,
                "sensor",
                vec![channel("a", value)],
            )
        };

        let center = encoder.encode(&observation(50.0)).unwrap().unwrap();
        let near = encoder.encode(&observation(51.0)).unwrap().unwrap();
        let far = encoder.encode(&observation(90.0)).unwrap().unwrap();
        assert!(center.vector.similarity(&near.vector) > center.vector.similarity(&far.vector));
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

        let odor_hv = encoder.encode(&odor).unwrap().unwrap();
        let taste_hv = encoder.encode(&taste).unwrap().unwrap();
        assert!(odor_hv.vector.similarity(&taste_hv.vector) < 0.5);
        assert_eq!(odor_hv.encoding_space_id, taste_hv.encoding_space_id);
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
        assert!(encoder.encode(&observation).unwrap().is_none());
    }

    #[test]
    fn unit_mismatch_is_an_integrity_error() {
        let encoder = encoder();
        let mut wrong = channel("a", 20.0);
        wrong.unit = MeasurementUnit::Ohms;
        let observation = ChemicalObservation::new(
            0,
            ChemicalModality::Olfactory,
            "sensor",
            vec![wrong],
        );
        assert!(matches!(
            encoder.encode(&observation),
            Err(FingerprintError::UnitMismatch { .. })
        ));
    }

    #[test]
    fn duplicate_channel_specs_are_rejected() {
        assert!(matches!(
            ChemicalFingerprintEncoder::new(vec![spec("a", 11, 101), spec("a", 12, 102)]),
            Err(FingerprintConfigError::DuplicateChannelSpec(name)) if name == "a"
        ));
    }

    #[test]
    fn duplicate_observation_channels_are_rejected() {
        let encoder = encoder();
        let observation = ChemicalObservation::new(
            0,
            ChemicalModality::Olfactory,
            "sensor",
            vec![channel("a", 20.0), channel("a", 21.0)],
        );
        assert!(matches!(
            encoder.encode(&observation),
            Err(FingerprintError::DuplicateChannel(name)) if name == "a"
        ));
    }

    #[test]
    fn invalid_measurement_is_not_collapsed_to_range_endpoint() {
        let encoder = encoder();
        let observation = ChemicalObservation::new(
            0,
            ChemicalModality::Olfactory,
            "sensor",
            vec![channel("a", f32::NAN)],
        );
        assert!(matches!(
            encoder.encode(&observation),
            Err(FingerprintError::InvalidMeasurement(name)) if name == "a"
        ));
    }

    #[test]
    fn zero_confidence_channels_do_not_influence_fingerprint() {
        let encoder = encoder();
        let only_a = ChemicalObservation::new(
            0,
            ChemicalModality::Olfactory,
            "sensor",
            vec![channel("a", 20.0)],
        );
        let mut dead_b = channel("b", 90.0);
        dead_b.health.score = 0.0;
        let with_dead_b = ChemicalObservation::new(
            0,
            ChemicalModality::Olfactory,
            "sensor",
            vec![channel("a", 20.0), dead_b],
        );

        let baseline = encoder.encode(&only_a).unwrap().unwrap();
        let hardened = encoder.encode(&with_dead_b).unwrap().unwrap();
        assert_eq!(baseline.vector, hardened.vector);
        assert_eq!(baseline.encoding_space_id, hardened.encoding_space_id);
        assert_eq!(hardened.used_channels, 1);
        assert_eq!(hardened.ignored_channels, 1);
    }
}
