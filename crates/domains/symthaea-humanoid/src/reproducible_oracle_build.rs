// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible-build evidence for independently generated MuJoCo oracles.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const ORACLE_BUILD_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OracleBuildManifest {
    pub schema_version: u32,
    pub build_id: String,
    pub source_repository: String,
    pub source_revision: String,
    pub source_tree_sha256: String,
    pub cargo_lock_sha256: String,
    pub toolchain_id: String,
    pub target_triple: String,
    pub nix_derivation: Option<String>,
    pub features: Vec<String>,
    pub build_command: Vec<String>,
    pub binary_sha256: String,
    pub model_sha256: String,
    pub generated_unix_millis: u64,
}

impl OracleBuildManifest {
    pub fn validate(&self) -> bool {
        self.schema_version == ORACLE_BUILD_MANIFEST_SCHEMA_VERSION
            && !self.build_id.trim().is_empty()
            && !self.source_repository.trim().is_empty()
            && !self.source_revision.trim().is_empty()
            && is_sha256(&self.source_tree_sha256)
            && is_sha256(&self.cargo_lock_sha256)
            && !self.toolchain_id.trim().is_empty()
            && !self.target_triple.trim().is_empty()
            && self
                .nix_derivation
                .as_ref()
                .map(|v| !v.trim().is_empty())
                .unwrap_or(true)
            && self.features.iter().all(|v| !v.trim().is_empty())
            && self.features.iter().collect::<BTreeSet<_>>().len() == self.features.len()
            && !self.build_command.is_empty()
            && self.build_command.iter().all(|v| !v.trim().is_empty())
            && is_sha256(&self.binary_sha256)
            && is_sha256(&self.model_sha256)
            && self.generated_unix_millis > 0
    }

    /// Identity of immutable build inputs. Environment identity is deliberately
    /// excluded so independently provisioned builders can still prove that they
    /// consumed the same source, lockfile, feature set, model, and command.
    pub fn reproducibility_input_identity(&self) -> Option<String> {
        self.validate().then(|| {
            let mut features = self.features.clone();
            features.sort();
            format!(
                "{}:{}:{}:{}:{}:{}",
                self.source_revision,
                self.source_tree_sha256,
                self.cargo_lock_sha256,
                features.join(","),
                self.build_command.join("\u{1f}"),
                self.model_sha256,
            )
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleReproducibilityCertificate {
    pub schema_version: u32,
    pub build_ids: Vec<String>,
    pub distinct_build_environments: usize,
    pub identical_binary_hash: Option<String>,
    pub identical_model_hash: Option<String>,
    pub passed: bool,
    pub failures: Vec<String>,
}

pub fn certify_reproducible_oracle_builds(
    manifests: &[OracleBuildManifest],
    minimum_independent_builds: usize,
) -> OracleReproducibilityCertificate {
    let mut failures = Vec::new();
    let valid = manifests
        .iter()
        .filter(|m| m.validate())
        .collect::<Vec<_>>();
    if minimum_independent_builds < 2 {
        failures.push("at least two independent builds must be required".to_string());
    }
    if valid.len() < minimum_independent_builds {
        failures.push(format!("only {} valid builds supplied", valid.len()));
    }
    let build_ids = valid.iter().map(|m| m.build_id.clone()).collect::<Vec<_>>();
    if build_ids.iter().collect::<BTreeSet<_>>().len() != build_ids.len() {
        failures.push("build identities are not unique".to_string());
    }
    let environments = valid
        .iter()
        .map(|m| {
            format!(
                "{}:{}:{}",
                m.toolchain_id,
                m.target_triple,
                m.nix_derivation.as_deref().unwrap_or("none")
            )
        })
        .collect::<BTreeSet<_>>();
    if environments.len() < minimum_independent_builds {
        failures.push("builds do not come from enough independent environments".to_string());
    }
    let binary_hashes = valid
        .iter()
        .map(|m| m.binary_sha256.as_str())
        .collect::<BTreeSet<_>>();
    let model_hashes = valid
        .iter()
        .map(|m| m.model_sha256.as_str())
        .collect::<BTreeSet<_>>();
    if binary_hashes.len() != 1 {
        failures.push("oracle binaries are not reproducible".to_string());
    }
    if model_hashes.len() != 1 {
        failures.push("oracle model hashes differ".to_string());
    }
    let source_ids = valid
        .iter()
        .filter_map(|m| m.reproducibility_input_identity())
        .collect::<BTreeSet<_>>();
    if source_ids.len() != 1 {
        failures.push("build inputs are not identical".to_string());
    }
    OracleReproducibilityCertificate {
        schema_version: 1,
        build_ids,
        distinct_build_environments: environments.len(),
        identical_binary_hash: (binary_hashes.len() == 1)
            .then(|| (*binary_hashes.first().unwrap()).to_string()),
        identical_model_hash: (model_hashes.len() == 1)
            .then(|| (*model_hashes.first().unwrap()).to_string()),
        passed: failures.is_empty(),
        failures,
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn manifest(id: &str, env: &str) -> OracleBuildManifest {
        OracleBuildManifest {
            schema_version: 1,
            build_id: id.into(),
            source_repository: "repo".into(),
            source_revision: "rev".into(),
            source_tree_sha256: "a".repeat(64),
            cargo_lock_sha256: "b".repeat(64),
            toolchain_id: env.into(),
            target_triple: "x86_64-linux".into(),
            nix_derivation: Some(format!("drv-{env}")),
            features: vec!["mujoco".into()],
            build_command: vec!["cargo".into(), "build".into()],
            binary_sha256: "c".repeat(64),
            model_sha256: "d".repeat(64),
            generated_unix_millis: 1,
        }
    }
    #[test]
    fn independent_identical_builds_pass() {
        let cert = certify_reproducible_oracle_builds(
            &[manifest("a", "rust-a"), manifest("b", "rust-b")],
            2,
        );
        assert!(cert.passed, "{:?}", cert.failures);
    }
}
