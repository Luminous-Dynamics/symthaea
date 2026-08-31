use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{AllostaticForecastSnapshot, INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION};

pub const EVIDENCE_CAPSULE_SCHEMA_VERSION: u16 = 1;

/// Machine-readable identifier for the prospective mechanism used by a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastBasisId {
    Kinematic,
    DynamicsAwareConstantDrive,
}

impl From<&AllostaticForecastSnapshot> for ForecastBasisId {
    fn from(value: &AllostaticForecastSnapshot) -> Self {
        match value {
            AllostaticForecastSnapshot::Kinematic { .. } => Self::Kinematic,
            AllostaticForecastSnapshot::DynamicsAwareConstantDrive { .. } => {
                Self::DynamicsAwareConstantDrive
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactDigest {
    pub name: String,
    pub sha256: String,
}

impl ArtifactDigest {
    pub fn new(name: impl Into<String>, sha256: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sha256: sha256.into(),
        }
    }
}

/// Caller-populated provenance for one evidence-bearing interoception run.
///
/// The library deliberately does not discover Git state, toolchain identity, or
/// file hashes itself. Those identities must be captured by the qualification
/// harness and supplied explicitly so a runtime cannot silently invent provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCapsuleManifest {
    pub schema_version: u16,
    pub source_commit: String,
    pub cargo_lock_sha256: String,
    pub flake_lock_sha256: Option<String>,
    pub rust_toolchain_sha256: Option<String>,
    pub rustc_vv: String,
    pub cargo_vv: String,
    pub target_triple: String,
    pub architecture: String,
    pub experiment_id: String,
    pub forecast_basis: ForecastBasisId,
    pub experiment_config_sha256: String,
    pub input_sequence_sha256: String,
    pub snapshot_schema_version: u16,
    pub evidence_plane_sha256: String,
    pub artifacts: Vec<ArtifactDigest>,
}

impl EvidenceCapsuleManifest {
    /// Return deterministic validation errors without mutating the manifest.
    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.schema_version != EVIDENCE_CAPSULE_SCHEMA_VERSION {
            errors.push(format!(
                "unsupported evidence capsule schema version: {}",
                self.schema_version
            ));
        }
        if self.snapshot_schema_version != INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION {
            errors.push(format!(
                "snapshot schema version mismatch: {}",
                self.snapshot_schema_version
            ));
        }
        if !is_lower_hex(&self.source_commit, 40) {
            errors.push("source_commit must be a 40-character lowercase Git SHA-1".into());
        }
        validate_sha256("cargo_lock_sha256", &self.cargo_lock_sha256, &mut errors);
        if let Some(value) = &self.flake_lock_sha256 {
            validate_sha256("flake_lock_sha256", value, &mut errors);
        }
        if let Some(value) = &self.rust_toolchain_sha256 {
            validate_sha256("rust_toolchain_sha256", value, &mut errors);
        }
        validate_sha256(
            "experiment_config_sha256",
            &self.experiment_config_sha256,
            &mut errors,
        );
        validate_sha256(
            "input_sequence_sha256",
            &self.input_sequence_sha256,
            &mut errors,
        );
        validate_sha256(
            "evidence_plane_sha256",
            &self.evidence_plane_sha256,
            &mut errors,
        );

        for (name, value) in [
            ("rustc_vv", self.rustc_vv.as_str()),
            ("cargo_vv", self.cargo_vv.as_str()),
            ("target_triple", self.target_triple.as_str()),
            ("architecture", self.architecture.as_str()),
            ("experiment_id", self.experiment_id.as_str()),
        ] {
            if value.trim().is_empty() {
                errors.push(format!("{name} must not be empty"));
            }
        }

        if self.artifacts.is_empty() {
            errors.push("at least one raw result artifact digest is required".into());
        }

        let mut artifact_names = BTreeSet::new();
        for artifact in &self.artifacts {
            if artifact.name.trim().is_empty() {
                errors.push("artifact names must not be empty".into());
            } else if !artifact_names.insert(artifact.name.as_str()) {
                errors.push(format!("duplicate artifact name: {}", artifact.name));
            }
            validate_sha256(
                &format!("artifact[{}].sha256", artifact.name),
                &artifact.sha256,
                &mut errors,
            );
        }

        errors
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let errors = self.validation_errors();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

fn validate_sha256(name: &str, value: &str, errors: &mut Vec<String>) {
    if !is_lower_hex(value, 64) {
        errors.push(format!(
            "{name} must be a 64-character lowercase SHA-256 digest"
        ));
    }
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
