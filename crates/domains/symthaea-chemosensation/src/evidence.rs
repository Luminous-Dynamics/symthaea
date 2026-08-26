// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Content-addressed receipts for raw chemical evidence.
//!
//! These IDs identify observations, not derived interpretations. The raw-value,
//! calibration, health, environment, source, modality, and timestamp content is
//! covered while channel ordering is canonicalized. HDC encoding-space identity
//! is intentionally excluded: the same physical evidence re-encoded under a new
//! representation should keep the same evidence identity.

use std::fmt;

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use crate::{
    ChemicalChannel, ChemicalModality, ChemicalObservation, ChemicalPercept, MeasurementUnit,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChemicalObservationId([u8; 32]);

impl ChemicalObservationId {
    pub fn from_observation(observation: &ChemicalObservation) -> Self {
        Self(hash_observation(observation))
    }

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ChemicalObservationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_hex(f, &self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChemicalEvidenceBundleId([u8; 32]);

impl ChemicalEvidenceBundleId {
    /// Build an order-invariant receipt for a set of raw observations.
    /// Duplicate observations remain represented because duplicate IDs are not
    /// deduplicated before hashing.
    pub fn from_observations(observations: &[&ChemicalObservation]) -> Self {
        let mut ids: Vec<ChemicalObservationId> = observations
            .iter()
            .map(|observation| ChemicalObservationId::from_observation(observation))
            .collect();
        ids.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));

        let mut hasher = Hasher::new();
        put_bytes(&mut hasher, b"symthaea-chemosensation-evidence-bundle-v1");
        put_u64(&mut hasher, ids.len() as u64);
        for id in ids {
            hasher.update(id.as_bytes());
        }
        Self(*hasher.finalize().as_bytes())
    }

    pub fn from_percepts(percepts: &[ChemicalPercept]) -> Self {
        let observations: Vec<&ChemicalObservation> =
            percepts.iter().map(|percept| &percept.evidence).collect();
        Self::from_observations(&observations)
    }

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ChemicalEvidenceBundleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_hex(f, &self.0)
    }
}

fn hash_observation(observation: &ChemicalObservation) -> [u8; 32] {
    let mut hasher = Hasher::new();
    put_bytes(&mut hasher, b"symthaea-chemosensation-observation-v1");
    put_u8(&mut hasher, modality_tag(observation.modality));
    put_u64(&mut hasher, observation.timestamp_us);
    put_bytes(&mut hasher, observation.source.as_bytes());
    put_optional_f32(&mut hasher, observation.environment.temperature_c);
    put_optional_f32(&mut hasher, observation.environment.humidity_rh);
    put_optional_f32(&mut hasher, observation.environment.pressure_pa);

    let mut channels: Vec<&ChemicalChannel> = observation.channels.iter().collect();
    channels.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| unit_tag(left.unit).cmp(&unit_tag(right.unit)))
            .then_with(|| left.raw_value.to_bits().cmp(&right.raw_value.to_bits()))
            .then_with(|| left.calibration.id.0.cmp(&right.calibration.id.0))
            .then_with(|| {
                left.calibration
                    .baseline
                    .to_bits()
                    .cmp(&right.calibration.baseline.to_bits())
            })
            .then_with(|| {
                left.calibration
                    .gain
                    .to_bits()
                    .cmp(&right.calibration.gain.to_bits())
            })
            .then_with(|| {
                left.calibration
                    .drift
                    .to_bits()
                    .cmp(&right.calibration.drift.to_bits())
            })
            .then_with(|| left.health.score.to_bits().cmp(&right.health.score.to_bits()))
            .then_with(|| left.health.saturated.cmp(&right.health.saturated))
            .then_with(|| left.health.contaminated.cmp(&right.health.contaminated))
    });

    put_u64(&mut hasher, channels.len() as u64);
    for channel in channels {
        put_bytes(&mut hasher, channel.name.as_bytes());
        put_u8(&mut hasher, unit_tag(channel.unit));
        put_u32(&mut hasher, channel.raw_value.to_bits());
        put_bytes(&mut hasher, channel.calibration.id.0.as_bytes());
        put_u32(&mut hasher, channel.calibration.baseline.to_bits());
        put_u32(&mut hasher, channel.calibration.gain.to_bits());
        put_u32(&mut hasher, channel.calibration.drift.to_bits());
        put_u32(&mut hasher, channel.health.score.to_bits());
        put_bool(&mut hasher, channel.health.saturated);
        put_bool(&mut hasher, channel.health.contaminated);
    }

    *hasher.finalize().as_bytes()
}

fn modality_tag(modality: ChemicalModality) -> u8 {
    match modality {
        ChemicalModality::Olfactory => 1,
        ChemicalModality::Gustatory => 2,
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

fn put_optional_f32(hasher: &mut Hasher, value: Option<f32>) {
    match value {
        Some(value) => {
            put_u8(hasher, 1);
            put_u32(hasher, value.to_bits());
        }
        None => put_u8(hasher, 0),
    }
}

fn put_bytes(hasher: &mut Hasher, bytes: &[u8]) {
    put_u64(hasher, bytes.len() as u64);
    hasher.update(bytes);
}

fn put_bool(hasher: &mut Hasher, value: bool) {
    put_u8(hasher, u8::from(value));
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

fn write_hex(f: &mut fmt::Formatter<'_>, bytes: &[u8; 32]) -> fmt::Result {
    for byte in bytes {
        write!(f, "{byte:02x}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CalibrationState, ChemicalFingerprint, EnvironmentReading, SensorHealth,
    };
    use symthaea_core::hdc::{HDC_DIMENSION, unified_hv::ContinuousHV};

    fn channel(name: &str, raw_value: f32) -> ChemicalChannel {
        ChemicalChannel {
            name: name.into(),
            raw_value,
            unit: MeasurementUnit::PartsPerMillion,
            calibration: CalibrationState::identity("cal-v1"),
            health: SensorHealth::default(),
        }
    }

    fn observation(channels: Vec<ChemicalChannel>) -> ChemicalObservation {
        ChemicalObservation::new(42, ChemicalModality::Olfactory, "nose-a", channels)
            .with_environment(EnvironmentReading {
                temperature_c: Some(22.5),
                humidity_rh: Some(0.45),
                pressure_pa: Some(101_325.0),
            })
    }

    fn percept(observation: ChemicalObservation, space_byte: u8, seed: u64) -> ChemicalPercept {
        ChemicalPercept {
            evidence: observation,
            fingerprint: ChemicalFingerprint {
                vector: ContinuousHV::random(HDC_DIMENSION, seed),
                confidence: 0.9,
                used_channels: 1,
                ignored_channels: 0,
                encoding_space_id: crate::ChemicalEncodingSpaceId::from_bytes([space_byte; 32]),
            },
        }
    }

    #[test]
    fn observation_id_is_channel_order_invariant() {
        let a = observation(vec![channel("voc", 1.0), channel("no2", 2.0)]);
        let b = observation(vec![channel("no2", 2.0), channel("voc", 1.0)]);
        assert_eq!(
            ChemicalObservationId::from_observation(&a),
            ChemicalObservationId::from_observation(&b)
        );
    }

    #[test]
    fn changing_raw_or_calibration_content_changes_observation_id() {
        let base = observation(vec![channel("voc", 1.0)]);
        let mut raw_changed = base.clone();
        raw_changed.channels[0].raw_value = 1.1;
        let mut calibration_changed = base.clone();
        calibration_changed.channels[0].calibration.drift = 0.1;

        let base_id = ChemicalObservationId::from_observation(&base);
        assert_ne!(base_id, ChemicalObservationId::from_observation(&raw_changed));
        assert_ne!(
            base_id,
            ChemicalObservationId::from_observation(&calibration_changed)
        );
    }

    #[test]
    fn bundle_id_is_component_order_invariant() {
        let a = percept(observation(vec![channel("voc", 1.0)]), 7, 1);
        let mut second_observation = observation(vec![channel("voc", 2.0)]);
        second_observation.source = "nose-b".into();
        let b = percept(second_observation, 7, 2);

        assert_eq!(
            ChemicalEvidenceBundleId::from_percepts(&[a.clone(), b.clone()]),
            ChemicalEvidenceBundleId::from_percepts(&[b, a])
        );
    }

    #[test]
    fn evidence_identity_survives_representation_migration() {
        let raw = observation(vec![channel("voc", 1.0)]);
        let old = percept(raw.clone(), 7, 1);
        let new = percept(raw, 8, 2);

        assert_ne!(old.fingerprint.encoding_space_id, new.fingerprint.encoding_space_id);
        assert_ne!(old.fingerprint.vector, new.fingerprint.vector);
        assert_eq!(
            ChemicalEvidenceBundleId::from_percepts(&[old]),
            ChemicalEvidenceBundleId::from_percepts(&[new])
        );
    }

    #[test]
    fn duplicate_observations_are_not_silently_deduplicated() {
        let raw = observation(vec![channel("voc", 1.0)]);
        let one = ChemicalEvidenceBundleId::from_observations(&[&raw]);
        let two = ChemicalEvidenceBundleId::from_observations(&[&raw, &raw]);
        assert_ne!(one, two);
    }
}
