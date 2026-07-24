// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic random-stream registry and coupling audit.
//!
//! A single reused PRNG stream can accidentally correlate wind, sensor noise,
//! actuator faults, and benchmark sampling. This registry derives independent
//! stable seeds from one campaign seed and a declared stream identity, while
//! rejecting duplicate identities and derived-seed collisions.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RandomStreamPurpose {
    Wind,
    AtmosphericTurbulence,
    SensorNoise,
    SensorFaults,
    ActuatorFaults,
    NavigationDropout,
    ScenarioSampling,
    BootstrapResampling,
    LearnedControllerExploration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RandomStreamSpec {
    pub stream_id: String,
    pub component_id: String,
    pub purpose: RandomStreamPurpose,
    pub replica: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivedRandomStream {
    pub stream_id: String,
    pub component_id: String,
    pub purpose: RandomStreamPurpose,
    pub replica: u32,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RandomStreamManifest {
    pub schema_version: String,
    pub campaign_seed: u64,
    pub streams: Vec<DerivedRandomStream>,
}

impl RandomStreamManifest {
    pub fn canonical_json(&self) -> Result<Vec<u8>, RandomStreamError> {
        let mut canonical = self.clone();
        canonical.streams.sort_by(|left, right| {
            left.stream_id
                .cmp(&right.stream_id)
                .then_with(|| left.component_id.cmp(&right.component_id))
                .then_with(|| left.purpose.cmp(&right.purpose))
                .then_with(|| left.replica.cmp(&right.replica))
        });
        serde_json::to_vec(&canonical).map_err(|_| RandomStreamError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, RandomStreamError> {
        Ok(format!("fnv1a64:{:016x}", fnv1a64(&self.canonical_json()?)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomStreamError {
    EmptyRegistry,
    EmptyIdentity,
    DuplicateStreamId,
    DuplicateSemanticStream,
    DerivedSeedCollision,
    UnknownStream,
    SerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomStreamRegistry {
    manifest: RandomStreamManifest,
    seeds_by_id: BTreeMap<String, u64>,
}

impl RandomStreamRegistry {
    pub fn new(
        schema_version: impl Into<String>,
        campaign_seed: u64,
        specs: Vec<RandomStreamSpec>,
    ) -> Result<Self, RandomStreamError> {
        let schema_version = schema_version.into();
        if schema_version.trim().is_empty() || specs.is_empty() {
            return Err(RandomStreamError::EmptyRegistry);
        }

        let mut stream_ids = BTreeSet::new();
        let mut semantic_keys = BTreeSet::new();
        let mut seeds = BTreeSet::new();
        let mut streams = Vec::with_capacity(specs.len());
        for spec in specs {
            if spec.stream_id.trim().is_empty() || spec.component_id.trim().is_empty() {
                return Err(RandomStreamError::EmptyIdentity);
            }
            if !stream_ids.insert(spec.stream_id.clone()) {
                return Err(RandomStreamError::DuplicateStreamId);
            }
            if !semantic_keys.insert((spec.component_id.clone(), spec.purpose, spec.replica)) {
                return Err(RandomStreamError::DuplicateSemanticStream);
            }
            let seed = derive_seed(campaign_seed, &spec);
            if !seeds.insert(seed) {
                return Err(RandomStreamError::DerivedSeedCollision);
            }
            streams.push(DerivedRandomStream {
                stream_id: spec.stream_id,
                component_id: spec.component_id,
                purpose: spec.purpose,
                replica: spec.replica,
                seed,
            });
        }
        streams.sort_by(|left, right| left.stream_id.cmp(&right.stream_id));
        let seeds_by_id = streams
            .iter()
            .map(|stream| (stream.stream_id.clone(), stream.seed))
            .collect();
        Ok(Self {
            manifest: RandomStreamManifest {
                schema_version,
                campaign_seed,
                streams,
            },
            seeds_by_id,
        })
    }

    pub fn seed(&self, stream_id: &str) -> Result<u64, RandomStreamError> {
        self.seeds_by_id
            .get(stream_id)
            .copied()
            .ok_or(RandomStreamError::UnknownStream)
    }

    pub fn manifest(&self) -> &RandomStreamManifest {
        &self.manifest
    }
}

fn derive_seed(campaign_seed: u64, spec: &RandomStreamSpec) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"symthaea-helicopter-rng-v1\0");
    bytes.extend_from_slice(&campaign_seed.to_le_bytes());
    bytes.extend_from_slice(spec.stream_id.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(spec.component_id.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&(spec.purpose as u32).to_le_bytes());
    bytes.extend_from_slice(&spec.replica.to_le_bytes());
    let seed = fnv1a64(&bytes);
    // Xorshift-style generators can lock at zero. Preserve determinism while
    // ensuring the derived seed is always usable by such generators.
    if seed == 0 { 0x9e3779b97f4a7c15 } else { seed }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specs() -> Vec<RandomStreamSpec> {
        vec![
            RandomStreamSpec {
                stream_id: "wind-primary".into(),
                component_id: "wind-model".into(),
                purpose: RandomStreamPurpose::Wind,
                replica: 0,
            },
            RandomStreamSpec {
                stream_id: "imu-a-noise".into(),
                component_id: "imu-a".into(),
                purpose: RandomStreamPurpose::SensorNoise,
                replica: 0,
            },
        ]
    }

    #[test]
    fn derivation_is_stable_and_streams_are_distinct() {
        let first = RandomStreamRegistry::new("1", 42, specs()).unwrap();
        let second = RandomStreamRegistry::new("1", 42, specs()).unwrap();
        assert_eq!(first.seed("wind-primary"), second.seed("wind-primary"));
        assert_ne!(
            first.seed("wind-primary").unwrap(),
            first.seed("imu-a-noise").unwrap()
        );
        assert_eq!(
            first.manifest().digest_fnv1a64().unwrap(),
            second.manifest().digest_fnv1a64().unwrap()
        );
    }

    #[test]
    fn campaign_seed_changes_all_derived_streams() {
        let first = RandomStreamRegistry::new("1", 42, specs()).unwrap();
        let second = RandomStreamRegistry::new("1", 43, specs()).unwrap();
        assert_ne!(
            first.seed("wind-primary").unwrap(),
            second.seed("wind-primary").unwrap()
        );
    }

    #[test]
    fn duplicate_semantic_stream_is_rejected_even_with_new_label() {
        let mut duplicate = specs();
        duplicate.push(RandomStreamSpec {
            stream_id: "wind-alias".into(),
            component_id: "wind-model".into(),
            purpose: RandomStreamPurpose::Wind,
            replica: 0,
        });
        assert_eq!(
            RandomStreamRegistry::new("1", 42, duplicate),
            Err(RandomStreamError::DuplicateSemanticStream)
        );
    }
}
