// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned, independently generated rigid-body dynamics oracle datasets.
//!
//! Datasets record the exact generalized state and an oracle snapshot emitted
//! by a separately identified build or engine. Admission refuses self-oracles,
//! duplicate states, unstable coordinate ordering, and malformed model hashes.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::dynamics_oracle::{
    DynamicsOracleCase, DynamicsOracleTolerances, compare_floating_base_dynamics,
};
use crate::floating_base::FloatingBaseDynamicsSnapshot;
use crate::morphology::HumanoidMorphology;

pub const DYNAMICS_ORACLE_DATASET_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsOracleDatasetManifest {
    pub schema_version: u32,
    pub dataset_id: String,
    pub generator_id: String,
    pub generator_build_id: String,
    pub engine_id: String,
    pub model_artifact_sha256: String,
    pub morphology: HumanoidMorphology,
    pub generalized_coordinate_order: Vec<String>,
    pub contact_site_order: Vec<String>,
    pub generated_unix_millis: u64,
}

impl DynamicsOracleDatasetManifest {
    fn validate_shape(&self) -> bool {
        self.schema_version == DYNAMICS_ORACLE_DATASET_SCHEMA_VERSION
            && !self.dataset_id.trim().is_empty()
            && !self.generator_id.trim().is_empty()
            && !self.generator_build_id.trim().is_empty()
            && !self.engine_id.trim().is_empty()
            && is_lower_hex_sha256(&self.model_artifact_sha256)
            && self.generalized_coordinate_order.len() == 6 + self.morphology.num_actuators()
            && unique_nonempty(&self.generalized_coordinate_order)
            && !self.contact_site_order.is_empty()
            && unique_nonempty(&self.contact_site_order)
            && self.generated_unix_millis > 0
    }

    pub fn validate_for_candidate(&self, candidate_build_id: &str) -> bool {
        self.validate_shape()
            && !candidate_build_id.trim().is_empty()
            && self.generator_build_id != candidate_build_id
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsOracleDatasetCase {
    pub case_id: String,
    pub state_fingerprint: u64,
    pub generalized_position: Vec<f64>,
    pub generalized_velocity: Vec<f64>,
    pub actuator_command: Vec<f64>,
    pub oracle: FloatingBaseDynamicsSnapshot,
}

impl DynamicsOracleDatasetCase {
    pub fn validate(&self, manifest: &DynamicsOracleDatasetManifest) -> bool {
        let nv = 6 + manifest.morphology.num_actuators();
        !self.case_id.trim().is_empty()
            && self.state_fingerprint != 0
            && self.generalized_velocity.len() == nv
            && self.generalized_position.len() == nv + 1
            && self.actuator_command.len() == manifest.morphology.num_actuators()
            && self
                .generalized_position
                .iter()
                .chain(self.generalized_velocity.iter())
                .chain(self.actuator_command.iter())
                .all(|value| value.is_finite())
            && fingerprint_state(
                &self.generalized_position,
                &self.generalized_velocity,
                &self.actuator_command,
            ) == self.state_fingerprint
            && self.oracle.validate()
            && self.oracle.morphology == manifest.morphology
            && self
                .oracle
                .contacts
                .iter()
                .map(|contact| contact.site_id.as_str())
                .eq(manifest.contact_site_order.iter().map(String::as_str))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsOracleDataset {
    pub manifest: DynamicsOracleDatasetManifest,
    pub cases: Vec<DynamicsOracleDatasetCase>,
}

impl DynamicsOracleDataset {
    pub fn validate_for_candidate(&self, candidate_build_id: &str) -> bool {
        self.manifest.validate_for_candidate(candidate_build_id)
            && !self.cases.is_empty()
            && self.cases.iter().all(|case| case.validate(&self.manifest))
            && self
                .cases
                .iter()
                .map(|case| case.case_id.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                == self.cases.len()
            && self
                .cases
                .iter()
                .map(|case| case.state_fingerprint)
                .collect::<BTreeSet<_>>()
                .len()
                == self.cases.len()
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        fs::write(path, bytes)
    }

    pub fn load_json(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        serde_json::from_slice(&bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }

    pub fn compare_candidates(
        &self,
        candidate_build_id: &str,
        candidates: &BTreeMap<u64, FloatingBaseDynamicsSnapshot>,
        tolerances: DynamicsOracleTolerances,
    ) -> Option<Vec<DynamicsOracleCase>> {
        self.validate_for_candidate(candidate_build_id)
            .then_some(())?;
        let mut reports = Vec::with_capacity(self.cases.len());
        for case in &self.cases {
            let candidate = candidates.get(&case.state_fingerprint)?;
            reports.push(DynamicsOracleCase {
                case_id: case.case_id.clone(),
                sampled_state_fingerprint: case.state_fingerprint,
                report: compare_floating_base_dynamics(candidate, &case.oracle, tolerances),
            });
        }
        Some(reports)
    }
}

/// Stable non-cryptographic fingerprint for duplicate-state rejection.
pub fn fingerprint_state(position: &[f64], velocity: &[f64], command: &[f64]) -> u64 {
    let mut state = 0xcbf29ce484222325u64;
    for tag in [0x51u8, 0x56, 0x43] {
        state ^= tag as u64;
        state = state.wrapping_mul(0x100000001b3);
        let values = match tag {
            0x51 => position,
            0x56 => velocity,
            _ => command,
        };
        state ^= values.len() as u64;
        state = state.wrapping_mul(0x100000001b3);
        for value in values {
            for byte in value.to_bits().to_le_bytes() {
                state ^= byte as u64;
                state = state.wrapping_mul(0x100000001b3);
            }
        }
    }
    state.max(1)
}

fn unique_nonempty(values: &[String]) -> bool {
    values.iter().all(|value| !value.trim().is_empty())
        && values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_order_and_bit_sensitive() {
        let one = fingerprint_state(&[0.0, 1.0], &[2.0], &[3.0]);
        let two = fingerprint_state(&[1.0, 0.0], &[2.0], &[3.0]);
        let three = fingerprint_state(&[0.0, 1.0], &[2.0], &[3.0000001]);
        assert_ne!(one, two);
        assert_ne!(one, three);
    }

    #[test]
    fn sha256_identity_requires_canonical_lower_hex() {
        assert!(is_lower_hex_sha256(&"a".repeat(64)));
        assert!(!is_lower_hex_sha256(&"A".repeat(64)));
        assert!(!is_lower_hex_sha256("abc"));
    }
}

/// Incremental builder intended for a separately compiled oracle generator.
/// The caller supplies the independently produced solver snapshot and exact
/// generalized state; duplicate states and contact-order drift are rejected.
pub struct DynamicsOracleDatasetBuilder {
    manifest: DynamicsOracleDatasetManifest,
    cases: Vec<DynamicsOracleDatasetCase>,
    case_ids: BTreeSet<String>,
    state_fingerprints: BTreeSet<u64>,
}

impl DynamicsOracleDatasetBuilder {
    pub fn new(manifest: DynamicsOracleDatasetManifest) -> Option<Self> {
        manifest.validate_shape().then_some(Self {
            manifest,
            cases: Vec::new(),
            case_ids: BTreeSet::new(),
            state_fingerprints: BTreeSet::new(),
        })
    }

    pub fn push_case(
        &mut self,
        case_id: impl Into<String>,
        generalized_position: Vec<f64>,
        generalized_velocity: Vec<f64>,
        actuator_command: Vec<f64>,
        oracle: FloatingBaseDynamicsSnapshot,
    ) -> bool {
        let case_id = case_id.into();
        let state_fingerprint = fingerprint_state(
            &generalized_position,
            &generalized_velocity,
            &actuator_command,
        );
        let case = DynamicsOracleDatasetCase {
            case_id: case_id.clone(),
            state_fingerprint,
            generalized_position,
            generalized_velocity,
            actuator_command,
            oracle,
        };
        if self.case_ids.contains(&case_id)
            || self.state_fingerprints.contains(&state_fingerprint)
            || !case.validate(&self.manifest)
        {
            return false;
        }
        self.case_ids.insert(case_id);
        self.state_fingerprints.insert(state_fingerprint);
        self.cases.push(case);
        true
    }

    pub fn finish(self, candidate_build_id: &str) -> Option<DynamicsOracleDataset> {
        let dataset = DynamicsOracleDataset {
            manifest: self.manifest,
            cases: self.cases,
        };
        dataset
            .validate_for_candidate(candidate_build_id)
            .then_some(dataset)
    }
}
