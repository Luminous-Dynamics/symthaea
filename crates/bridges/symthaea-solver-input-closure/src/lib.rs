// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical transitive input closure for external engineering solvers.
//!
//! A top-level input digest is not a complete reproducibility claim when a
//! solver may discover transitively referenced files, generated meshes/cases,
//! plugins, configuration or environment-dependent inputs. This crate provides
//! one fail-closed closure contract for all such adapters.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

const CLOSURE_VERSION: &str = "solver-input-closure-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SolverInputRole {
    Primary,
    Referenced,
    GeneratedIntermediate,
    RuntimePlugin,
    Configuration,
}

impl SolverInputRole {
    fn tag(self) -> u8 {
        match self {
            Self::Primary => 0,
            Self::Referenced => 1,
            Self::GeneratedIntermediate => 2,
            Self::RuntimePlugin => 3,
            Self::Configuration => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverInputArtifact {
    pub logical_id: String,
    pub role: SolverInputRole,
    pub digest_blake3: String,
    pub byte_len: u64,
    /// Exact logical parents from which this artifact was derived/referenced.
    pub parent_ids: Vec<String>,
}

impl SolverInputArtifact {
    pub fn from_bytes(
        logical_id: impl Into<String>,
        role: SolverInputRole,
        bytes: &[u8],
    ) -> Self {
        Self {
            logical_id: logical_id.into(),
            role,
            digest_blake3: blake3::hash(bytes).to_hex().to_string(),
            byte_len: bytes.len() as u64,
            parent_ids: Vec::new(),
        }
    }

    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_ids.push(parent_id.into());
        self
    }
}

/// Environment values are stored only by digest so qualification identity can
/// bind them without requiring secrets or machine-local values to be serialized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentBinding {
    pub key: String,
    pub value_digest_blake3: String,
}

impl EnvironmentBinding {
    pub fn from_value(key: impl Into<String>, value: impl AsRef<[u8]>) -> Self {
        Self {
            key: key.into(),
            value_digest_blake3: blake3::hash(value.as_ref()).to_hex().to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverInputClosure {
    pub closure_version: String,
    pub solver_id: String,
    pub solver_version: String,
    pub solver_executable_digest_blake3: String,
    pub artifacts: Vec<SolverInputArtifact>,
    pub environment: Vec<EnvironmentBinding>,
    /// Ambient discovery mechanisms explicitly disabled by the adapter, e.g.
    /// user startup files or uncontrolled model search paths.
    pub prohibited_ambient: Vec<String>,
    pub closure_digest_blake3: String,
}

impl SolverInputClosure {
    pub fn validate_identity(&self) -> Result<(), ClosureError> {
        validate_digest(
            "closure_digest_blake3",
            &self.closure_digest_blake3,
        )?;
        let recomputed = compute_closure_digest(
            &self.solver_id,
            &self.solver_version,
            &self.solver_executable_digest_blake3,
            &self.artifacts,
            &self.environment,
            &self.prohibited_ambient,
        );
        if recomputed != self.closure_digest_blake3 {
            return Err(ClosureError::ClosureDigestMismatch {
                expected: self.closure_digest_blake3.clone(),
                actual: recomputed,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct SolverInputClosureBuilder {
    solver_id: String,
    solver_version: String,
    solver_executable_digest_blake3: String,
    artifacts: Vec<SolverInputArtifact>,
    environment: Vec<EnvironmentBinding>,
    prohibited_ambient: Vec<String>,
    unresolved_inputs: Vec<String>,
}

impl SolverInputClosureBuilder {
    pub fn new(
        solver_id: impl Into<String>,
        solver_version: impl Into<String>,
        solver_executable_bytes: &[u8],
    ) -> Self {
        Self {
            solver_id: solver_id.into(),
            solver_version: solver_version.into(),
            solver_executable_digest_blake3: blake3::hash(solver_executable_bytes)
                .to_hex()
                .to_string(),
            ..Self::default()
        }
    }

    /// Use when the executable bytes were hashed by a trusted outer layer.
    pub fn with_solver_executable_digest(mut self, digest: impl Into<String>) -> Self {
        self.solver_executable_digest_blake3 = digest.into();
        self
    }

    pub fn add_artifact(mut self, artifact: SolverInputArtifact) -> Self {
        self.artifacts.push(artifact);
        self
    }

    pub fn bind_environment(mut self, binding: EnvironmentBinding) -> Self {
        self.environment.push(binding);
        self
    }

    pub fn prohibit_ambient(mut self, mechanism: impl Into<String>) -> Self {
        self.prohibited_ambient.push(mechanism.into());
        self
    }

    pub fn unresolved(mut self, logical_reference: impl Into<String>) -> Self {
        self.unresolved_inputs.push(logical_reference.into());
        self
    }

    pub fn finalize(mut self) -> Result<SolverInputClosure, ClosureError> {
        if self.solver_id.trim().is_empty() || self.solver_version.trim().is_empty() {
            return Err(ClosureError::EmptySolverIdentity);
        }
        validate_digest(
            "solver_executable_digest_blake3",
            &self.solver_executable_digest_blake3,
        )?;

        self.unresolved_inputs.sort();
        self.unresolved_inputs.dedup();
        if !self.unresolved_inputs.is_empty() {
            return Err(ClosureError::UnresolvedInputs(self.unresolved_inputs));
        }

        let mut ids = BTreeSet::new();
        let mut primary_count = 0usize;
        for artifact in &mut self.artifacts {
            if artifact.logical_id.trim().is_empty() {
                return Err(ClosureError::EmptyLogicalId);
            }
            if !ids.insert(artifact.logical_id.clone()) {
                return Err(ClosureError::DuplicateArtifactId(
                    artifact.logical_id.clone(),
                ));
            }
            validate_digest("artifact.digest_blake3", &artifact.digest_blake3)?;
            if artifact.role == SolverInputRole::Primary {
                primary_count += 1;
            }
            artifact.parent_ids.sort();
            artifact.parent_ids.dedup();
            if artifact
                .parent_ids
                .iter()
                .any(|parent| parent == &artifact.logical_id)
            {
                return Err(ClosureError::SelfParent(artifact.logical_id.clone()));
            }
            if artifact.role == SolverInputRole::GeneratedIntermediate
                && artifact.parent_ids.is_empty()
            {
                return Err(ClosureError::GeneratedIntermediateMissingParent(
                    artifact.logical_id.clone(),
                ));
            }
        }
        if primary_count != 1 {
            return Err(ClosureError::PrimaryArtifactCount(primary_count));
        }
        for artifact in &self.artifacts {
            for parent in &artifact.parent_ids {
                if !ids.contains(parent) {
                    return Err(ClosureError::MissingParent {
                        artifact: artifact.logical_id.clone(),
                        parent: parent.clone(),
                    });
                }
            }
        }
        self.artifacts.sort_by(|a, b| {
            a.role
                .tag()
                .cmp(&b.role.tag())
                .then_with(|| a.logical_id.cmp(&b.logical_id))
        });

        let mut environment_keys = BTreeSet::new();
        for binding in &self.environment {
            if binding.key.trim().is_empty() {
                return Err(ClosureError::EmptyEnvironmentKey);
            }
            if !environment_keys.insert(binding.key.clone()) {
                return Err(ClosureError::DuplicateEnvironmentKey(binding.key.clone()));
            }
            validate_digest("environment.value_digest_blake3", &binding.value_digest_blake3)?;
        }
        self.environment.sort_by(|a, b| a.key.cmp(&b.key));

        self.prohibited_ambient.retain(|item| !item.trim().is_empty());
        self.prohibited_ambient.sort();
        self.prohibited_ambient.dedup();

        let closure_digest_blake3 = compute_closure_digest(
            &self.solver_id,
            &self.solver_version,
            &self.solver_executable_digest_blake3,
            &self.artifacts,
            &self.environment,
            &self.prohibited_ambient,
        );

        Ok(SolverInputClosure {
            closure_version: CLOSURE_VERSION.into(),
            solver_id: self.solver_id,
            solver_version: self.solver_version,
            solver_executable_digest_blake3: self.solver_executable_digest_blake3,
            artifacts: self.artifacts,
            environment: self.environment,
            prohibited_ambient: self.prohibited_ambient,
            closure_digest_blake3,
        })
    }
}

fn validate_digest(field: &'static str, digest: &str) -> Result<(), ClosureError> {
    let valid = digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit());
    if valid {
        Ok(())
    } else {
        Err(ClosureError::InvalidDigest {
            field,
            digest: digest.to_string(),
        })
    }
}

fn compute_closure_digest(
    solver_id: &str,
    solver_version: &str,
    solver_executable_digest: &str,
    artifacts: &[SolverInputArtifact],
    environment: &[EnvironmentBinding],
    prohibited_ambient: &[String],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_str(&mut hasher, CLOSURE_VERSION);
    hash_str(&mut hasher, solver_id);
    hash_str(&mut hasher, solver_version);
    hash_str(&mut hasher, solver_executable_digest);

    hash_u64(&mut hasher, artifacts.len() as u64);
    for artifact in artifacts {
        hasher.update(&[artifact.role.tag()]);
        hash_str(&mut hasher, &artifact.logical_id);
        hash_str(&mut hasher, &artifact.digest_blake3);
        hash_u64(&mut hasher, artifact.byte_len);
        hash_u64(&mut hasher, artifact.parent_ids.len() as u64);
        for parent in &artifact.parent_ids {
            hash_str(&mut hasher, parent);
        }
    }

    hash_u64(&mut hasher, environment.len() as u64);
    for binding in environment {
        hash_str(&mut hasher, &binding.key);
        hash_str(&mut hasher, &binding.value_digest_blake3);
    }

    hash_u64(&mut hasher, prohibited_ambient.len() as u64);
    for mechanism in prohibited_ambient {
        hash_str(&mut hasher, mechanism);
    }

    hasher.finalize().to_hex().to_string()
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ClosureError {
    #[error("solver id/version cannot be empty")]
    EmptySolverIdentity,
    #[error("artifact logical id cannot be empty")]
    EmptyLogicalId,
    #[error("environment key cannot be empty")]
    EmptyEnvironmentKey,
    #[error("invalid BLAKE3 digest for {field}: {digest:?}")]
    InvalidDigest { field: &'static str, digest: String },
    #[error("duplicate artifact logical id {0:?}")]
    DuplicateArtifactId(String),
    #[error("expected exactly one primary artifact, found {0}")]
    PrimaryArtifactCount(usize),
    #[error("artifact {0:?} cannot list itself as a parent")]
    SelfParent(String),
    #[error("generated intermediate {0:?} requires at least one parent")]
    GeneratedIntermediateMissingParent(String),
    #[error("artifact {artifact:?} references missing parent {parent:?}")]
    MissingParent { artifact: String, parent: String },
    #[error("duplicate environment key {0:?}")]
    DuplicateEnvironmentKey(String),
    #[error("solver input closure is incomplete; unresolved inputs: {0:?}")]
    UnresolvedInputs(Vec<String>),
    #[error("closure digest mismatch: expected {expected}, recomputed {actual}")]
    ClosureDigestMismatch { expected: String, actual: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_builder() -> SolverInputClosureBuilder {
        SolverInputClosureBuilder::new("solver", "1.0", b"solver-executable")
            .add_artifact(SolverInputArtifact::from_bytes(
                "primary",
                SolverInputRole::Primary,
                b"primary-input",
            ))
            .prohibit_ambient("user-startup-file")
    }

    #[test]
    fn transitive_artifact_changes_closure_identity() {
        let a = base_builder()
            .add_artifact(SolverInputArtifact::from_bytes(
                "model-lib",
                SolverInputRole::Referenced,
                b"version-a",
            ))
            .finalize()
            .unwrap();
        let b = base_builder()
            .add_artifact(SolverInputArtifact::from_bytes(
                "model-lib",
                SolverInputRole::Referenced,
                b"version-b",
            ))
            .finalize()
            .unwrap();
        assert_ne!(a.closure_digest_blake3, b.closure_digest_blake3);
    }

    #[test]
    fn ordering_does_not_change_closure_identity() {
        let x = SolverInputArtifact::from_bytes("x", SolverInputRole::Referenced, b"x");
        let y = SolverInputArtifact::from_bytes("y", SolverInputRole::Referenced, b"y");
        let a = base_builder()
            .add_artifact(x.clone())
            .add_artifact(y.clone())
            .finalize()
            .unwrap();
        let b = base_builder()
            .add_artifact(y)
            .add_artifact(x)
            .finalize()
            .unwrap();
        assert_eq!(a.closure_digest_blake3, b.closure_digest_blake3);
    }

    #[test]
    fn unresolved_reference_fails_closed() {
        assert!(matches!(
            base_builder().unresolved("vendor-model.lib").finalize(),
            Err(ClosureError::UnresolvedInputs(items)) if items == vec!["vendor-model.lib"]
        ));
    }

    #[test]
    fn generated_intermediate_requires_bound_parent() {
        let generated = SolverInputArtifact::from_bytes(
            "mesh",
            SolverInputRole::GeneratedIntermediate,
            b"mesh-bytes",
        );
        assert!(matches!(
            base_builder().add_artifact(generated).finalize(),
            Err(ClosureError::GeneratedIntermediateMissingParent(id)) if id == "mesh"
        ));
    }

    #[test]
    fn generated_parent_must_exist() {
        let generated = SolverInputArtifact::from_bytes(
            "mesh",
            SolverInputRole::GeneratedIntermediate,
            b"mesh-bytes",
        )
        .with_parent("missing-cad");
        assert!(matches!(
            base_builder().add_artifact(generated).finalize(),
            Err(ClosureError::MissingParent { artifact, parent })
                if artifact == "mesh" && parent == "missing-cad"
        ));
    }

    #[test]
    fn environment_binding_changes_identity_without_storing_value() {
        let a = base_builder()
            .bind_environment(EnvironmentBinding::from_value("OMP_NUM_THREADS", "1"))
            .finalize()
            .unwrap();
        let b = base_builder()
            .bind_environment(EnvironmentBinding::from_value("OMP_NUM_THREADS", "2"))
            .finalize()
            .unwrap();
        assert_ne!(a.closure_digest_blake3, b.closure_digest_blake3);
        assert_eq!(a.environment[0].key, "OMP_NUM_THREADS");
        assert_ne!(a.environment[0].value_digest_blake3, "1");
    }

    #[test]
    fn exactly_one_primary_is_required() {
        let none = SolverInputClosureBuilder::new("solver", "1", b"exe")
            .finalize()
            .unwrap_err();
        assert_eq!(none, ClosureError::PrimaryArtifactCount(0));

        let two = base_builder()
            .add_artifact(SolverInputArtifact::from_bytes(
                "other-primary",
                SolverInputRole::Primary,
                b"other",
            ))
            .finalize()
            .unwrap_err();
        assert_eq!(two, ClosureError::PrimaryArtifactCount(2));
    }

    #[test]
    fn tampering_is_detected_by_identity_validation() {
        let mut closure = base_builder().finalize().unwrap();
        closure.artifacts[0].byte_len += 1;
        assert!(matches!(
            closure.validate_identity(),
            Err(ClosureError::ClosureDigestMismatch { .. })
        ));
    }
}
