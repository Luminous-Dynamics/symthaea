// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build provenance and reproducibility evidence.
//!
//! A release artifact is bound to source, lockfile, compiler, target, feature,
//! dependency graph, SBOM, build environment, and output digests. Reproducible
//! builds require two independently declared builds to agree on identity and
//! outputs; deterministic FNV report digests are not authenticity signatures.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildOutputArtifact {
    pub artifact_path: String,
    pub digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildProvenanceManifest {
    pub schema_version: String,
    pub build_id: String,
    pub source_tree_digest: String,
    pub cargo_lock_digest: String,
    pub dependency_graph_digest: String,
    pub sbom_digest: Option<String>,
    pub rustc_version: String,
    pub host_triple: String,
    pub target_triple: String,
    pub profile: String,
    pub enabled_features: Vec<String>,
    pub rustflags: Vec<String>,
    pub build_environment_digest: String,
    pub source_date_epoch: Option<u64>,
    pub hermetic: bool,
    pub network_disabled: bool,
    pub output_artifacts: Vec<BuildOutputArtifact>,
    pub authenticity_reference: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildProvenancePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub required_target_triple: String,
    pub required_profile: String,
    pub required_features: Vec<String>,
    pub forbidden_rustflags: Vec<String>,
    pub require_hermetic: bool,
    pub require_network_disabled: bool,
    pub require_sbom: bool,
    pub require_source_date_epoch: bool,
    pub require_authenticity: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildProvenanceStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildProvenanceIssue {
    MissingDigest(String),
    InvalidDigest(String),
    TargetMismatch { expected: String, observed: String },
    ProfileMismatch { expected: String, observed: String },
    MissingFeature(String),
    DuplicateFeature(String),
    ForbiddenRustflag(String),
    NonHermeticBuild,
    NetworkAccessEnabled,
    MissingSbom,
    MissingSourceDateEpoch,
    MissingAuthenticity,
    DuplicateOutputPath(String),
    NoOutputArtifacts,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildProvenanceReport {
    pub schema_version: String,
    pub policy_id: String,
    pub build_id: String,
    pub status: BuildProvenanceStatus,
    pub issues: Vec<BuildProvenanceIssue>,
    pub normalized_features: Vec<String>,
    pub output_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildReproducibilityIssue {
    SourceTreeMismatch,
    LockfileMismatch,
    DependencyGraphMismatch,
    CompilerMismatch,
    TargetMismatch,
    ProfileMismatch,
    FeatureMismatch,
    RustflagMismatch,
    BuildEnvironmentMismatch,
    SourceDateEpochMismatch,
    OutputSetMismatch,
    OutputDigestMismatch(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildReproducibilityReport {
    pub schema_version: String,
    pub first_build_id: String,
    pub second_build_id: String,
    pub reproducible: bool,
    pub issues: Vec<BuildReproducibilityIssue>,
}

impl BuildProvenanceReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, BuildProvenanceError> {
        let mut canonical = self.clone();
        canonical.normalized_features.sort();
        canonical.issues.sort_by_key(provenance_issue_sort_key);
        serde_json::to_vec(&canonical).map_err(|_| BuildProvenanceError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, BuildProvenanceError> {
        fnv1a64(&self.canonical_json()?)
    }
}

impl BuildReproducibilityReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, BuildProvenanceError> {
        let mut canonical = self.clone();
        canonical.issues.sort_by_key(repro_issue_sort_key);
        serde_json::to_vec(&canonical).map_err(|_| BuildProvenanceError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, BuildProvenanceError> {
        fnv1a64(&self.canonical_json()?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildProvenanceError {
    InvalidPolicy,
    InvalidManifest,
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct BuildProvenanceVerifier {
    policy: BuildProvenancePolicy,
}

impl BuildProvenanceVerifier {
    pub fn new(policy: BuildProvenancePolicy) -> Result<Self, BuildProvenanceError> {
        let features: BTreeSet<_> = policy.required_features.iter().collect();
        let flags: BTreeSet<_> = policy.forbidden_rustflags.iter().collect();
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.required_target_triple.trim().is_empty()
            || policy.required_profile.trim().is_empty()
            || features.len() != policy.required_features.len()
            || flags.len() != policy.forbidden_rustflags.len()
        {
            return Err(BuildProvenanceError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn evaluate(
        &self,
        manifest: &BuildProvenanceManifest,
    ) -> Result<BuildProvenanceReport, BuildProvenanceError> {
        validate_manifest_identity(manifest)?;
        let mut issues = Vec::new();
        for (name, digest) in [
            ("source_tree_digest", manifest.source_tree_digest.as_str()),
            ("cargo_lock_digest", manifest.cargo_lock_digest.as_str()),
            (
                "dependency_graph_digest",
                manifest.dependency_graph_digest.as_str(),
            ),
            (
                "build_environment_digest",
                manifest.build_environment_digest.as_str(),
            ),
        ] {
            check_digest(name, digest, &mut issues);
        }
        if manifest.target_triple != self.policy.required_target_triple {
            issues.push(BuildProvenanceIssue::TargetMismatch {
                expected: self.policy.required_target_triple.clone(),
                observed: manifest.target_triple.clone(),
            });
        }
        if manifest.profile != self.policy.required_profile {
            issues.push(BuildProvenanceIssue::ProfileMismatch {
                expected: self.policy.required_profile.clone(),
                observed: manifest.profile.clone(),
            });
        }
        let mut normalized_features = manifest.enabled_features.clone();
        normalized_features.sort();
        for pair in normalized_features.windows(2) {
            if pair[0] == pair[1] {
                issues.push(BuildProvenanceIssue::DuplicateFeature(pair[0].clone()));
            }
        }
        for required in &self.policy.required_features {
            if !normalized_features.contains(required) {
                issues.push(BuildProvenanceIssue::MissingFeature(required.clone()));
            }
        }
        for forbidden in &self.policy.forbidden_rustflags {
            if manifest
                .rustflags
                .iter()
                .any(|flag| flag.contains(forbidden))
            {
                issues.push(BuildProvenanceIssue::ForbiddenRustflag(forbidden.clone()));
            }
        }
        if self.policy.require_hermetic && !manifest.hermetic {
            issues.push(BuildProvenanceIssue::NonHermeticBuild);
        }
        if self.policy.require_network_disabled && !manifest.network_disabled {
            issues.push(BuildProvenanceIssue::NetworkAccessEnabled);
        }
        if self.policy.require_sbom {
            match manifest.sbom_digest.as_deref() {
                Some(digest) => check_digest("sbom_digest", digest, &mut issues),
                None => issues.push(BuildProvenanceIssue::MissingSbom),
            }
        }
        if self.policy.require_source_date_epoch && manifest.source_date_epoch.is_none() {
            issues.push(BuildProvenanceIssue::MissingSourceDateEpoch);
        }
        if self.policy.require_authenticity
            && manifest
                .authenticity_reference
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
        {
            issues.push(BuildProvenanceIssue::MissingAuthenticity);
        }
        if manifest.output_artifacts.is_empty() {
            issues.push(BuildProvenanceIssue::NoOutputArtifacts);
        }
        let mut paths = BTreeSet::new();
        for artifact in &manifest.output_artifacts {
            if artifact.artifact_path.trim().is_empty() {
                return Err(BuildProvenanceError::InvalidManifest);
            }
            if !paths.insert(artifact.artifact_path.clone()) {
                issues.push(BuildProvenanceIssue::DuplicateOutputPath(
                    artifact.artifact_path.clone(),
                ));
            }
            check_digest(
                &format!("output:{}", artifact.artifact_path),
                &artifact.digest,
                &mut issues,
            );
        }

        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                BuildProvenanceIssue::MissingDigest(_)
                    | BuildProvenanceIssue::MissingSbom
                    | BuildProvenanceIssue::MissingSourceDateEpoch
                    | BuildProvenanceIssue::MissingAuthenticity
                    | BuildProvenanceIssue::NoOutputArtifacts
            )
        });
        let failed = issues.iter().any(|issue| {
            !matches!(
                issue,
                BuildProvenanceIssue::MissingDigest(_)
                    | BuildProvenanceIssue::MissingSbom
                    | BuildProvenanceIssue::MissingSourceDateEpoch
                    | BuildProvenanceIssue::MissingAuthenticity
                    | BuildProvenanceIssue::NoOutputArtifacts
            )
        });
        let status = if failed {
            BuildProvenanceStatus::Fail
        } else if incomplete {
            BuildProvenanceStatus::Incomplete
        } else {
            BuildProvenanceStatus::Pass
        };
        Ok(BuildProvenanceReport {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            build_id: manifest.build_id.clone(),
            status,
            issues,
            normalized_features,
            output_count: manifest.output_artifacts.len(),
        })
    }

    pub fn compare_reproducibility(
        &self,
        first: &BuildProvenanceManifest,
        second: &BuildProvenanceManifest,
    ) -> Result<BuildReproducibilityReport, BuildProvenanceError> {
        validate_manifest_identity(first)?;
        validate_manifest_identity(second)?;
        let mut issues = Vec::new();
        if first.source_tree_digest != second.source_tree_digest {
            issues.push(BuildReproducibilityIssue::SourceTreeMismatch);
        }
        if first.cargo_lock_digest != second.cargo_lock_digest {
            issues.push(BuildReproducibilityIssue::LockfileMismatch);
        }
        if first.dependency_graph_digest != second.dependency_graph_digest {
            issues.push(BuildReproducibilityIssue::DependencyGraphMismatch);
        }
        if first.rustc_version != second.rustc_version || first.host_triple != second.host_triple {
            issues.push(BuildReproducibilityIssue::CompilerMismatch);
        }
        if first.target_triple != second.target_triple {
            issues.push(BuildReproducibilityIssue::TargetMismatch);
        }
        if first.profile != second.profile {
            issues.push(BuildReproducibilityIssue::ProfileMismatch);
        }
        if normalized(&first.enabled_features) != normalized(&second.enabled_features) {
            issues.push(BuildReproducibilityIssue::FeatureMismatch);
        }
        if normalized(&first.rustflags) != normalized(&second.rustflags) {
            issues.push(BuildReproducibilityIssue::RustflagMismatch);
        }
        if first.build_environment_digest != second.build_environment_digest {
            issues.push(BuildReproducibilityIssue::BuildEnvironmentMismatch);
        }
        if first.source_date_epoch != second.source_date_epoch {
            issues.push(BuildReproducibilityIssue::SourceDateEpochMismatch);
        }

        let first_outputs = output_map(first)?;
        let second_outputs = output_map(second)?;
        if first_outputs.keys().collect::<Vec<_>>() != second_outputs.keys().collect::<Vec<_>>() {
            issues.push(BuildReproducibilityIssue::OutputSetMismatch);
        }
        for (path, first_digest) in &first_outputs {
            if let Some(second_digest) = second_outputs.get(path)
                && first_digest != second_digest
            {
                issues.push(BuildReproducibilityIssue::OutputDigestMismatch(
                    path.clone(),
                ));
            }
        }
        Ok(BuildReproducibilityReport {
            schema_version: self.policy.schema_version.clone(),
            first_build_id: first.build_id.clone(),
            second_build_id: second.build_id.clone(),
            reproducible: issues.is_empty(),
            issues,
        })
    }
}

fn validate_manifest_identity(
    manifest: &BuildProvenanceManifest,
) -> Result<(), BuildProvenanceError> {
    if manifest.schema_version.trim().is_empty()
        || manifest.build_id.trim().is_empty()
        || manifest.rustc_version.trim().is_empty()
        || manifest.host_triple.trim().is_empty()
        || manifest.target_triple.trim().is_empty()
        || manifest.profile.trim().is_empty()
    {
        return Err(BuildProvenanceError::InvalidManifest);
    }
    Ok(())
}

fn check_digest(name: &str, digest: &str, issues: &mut Vec<BuildProvenanceIssue>) {
    if digest.trim().is_empty() {
        issues.push(BuildProvenanceIssue::MissingDigest(name.into()));
    } else if !digest.contains(':') || digest.ends_with(':') {
        issues.push(BuildProvenanceIssue::InvalidDigest(name.into()));
    }
}

fn output_map(
    manifest: &BuildProvenanceManifest,
) -> Result<BTreeMap<String, String>, BuildProvenanceError> {
    let mut outputs = BTreeMap::new();
    for artifact in &manifest.output_artifacts {
        if artifact.artifact_path.trim().is_empty()
            || artifact.digest.trim().is_empty()
            || outputs
                .insert(artifact.artifact_path.clone(), artifact.digest.clone())
                .is_some()
        {
            return Err(BuildProvenanceError::InvalidManifest);
        }
    }
    Ok(outputs)
}

fn normalized(values: &[String]) -> Vec<String> {
    let mut values = values.to_vec();
    values.sort();
    values
}

fn fnv1a64(bytes: &[u8]) -> Result<String, BuildProvenanceError> {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(format!("fnv1a64:{hash:016x}"))
}

fn provenance_issue_sort_key(issue: &BuildProvenanceIssue) -> String {
    format!("{issue:?}")
}

fn repro_issue_sort_key(issue: &BuildReproducibilityIssue) -> String {
    format!("{issue:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> BuildProvenancePolicy {
        BuildProvenancePolicy {
            schema_version: "symthaea.helicopter.build-provenance.v1".into(),
            policy_id: "release-build-policy".into(),
            required_target_triple: "x86_64-unknown-linux-gnu".into(),
            required_profile: "release".into(),
            required_features: vec!["hardware".into()],
            forbidden_rustflags: vec!["target-cpu=native".into()],
            require_hermetic: true,
            require_network_disabled: true,
            require_sbom: true,
            require_source_date_epoch: true,
            require_authenticity: true,
        }
    }

    fn manifest(id: &str) -> BuildProvenanceManifest {
        BuildProvenanceManifest {
            schema_version: "symthaea.helicopter.build-provenance.v1".into(),
            build_id: id.into(),
            source_tree_digest: "sha256:source".into(),
            cargo_lock_digest: "sha256:lock".into(),
            dependency_graph_digest: "sha256:deps".into(),
            sbom_digest: Some("sha256:sbom".into()),
            rustc_version: "rustc 1.94.0".into(),
            host_triple: "x86_64-unknown-linux-gnu".into(),
            target_triple: "x86_64-unknown-linux-gnu".into(),
            profile: "release".into(),
            enabled_features: vec!["hardware".into()],
            rustflags: vec!["-Cdebuginfo=0".into()],
            build_environment_digest: "sha256:env".into(),
            source_date_epoch: Some(1_784_652_800),
            hermetic: true,
            network_disabled: true,
            output_artifacts: vec![BuildOutputArtifact {
                artifact_path: "bin/helicopter".into(),
                digest: "sha256:output".into(),
            }],
            authenticity_reference: Some("signature:builder".into()),
        }
    }

    #[test]
    fn complete_manifest_passes() {
        let verifier = BuildProvenanceVerifier::new(policy()).unwrap();
        let report = verifier.evaluate(&manifest("build-a")).unwrap();
        assert_eq!(report.status, BuildProvenanceStatus::Pass);
    }

    #[test]
    fn native_cpu_flag_is_rejected() {
        let verifier = BuildProvenanceVerifier::new(policy()).unwrap();
        let mut build = manifest("build-a");
        build.rustflags.push("-Ctarget-cpu=native".into());
        let report = verifier.evaluate(&build).unwrap();
        assert_eq!(report.status, BuildProvenanceStatus::Fail);
    }

    #[test]
    fn missing_sbom_is_incomplete() {
        let verifier = BuildProvenanceVerifier::new(policy()).unwrap();
        let mut build = manifest("build-a");
        build.sbom_digest = None;
        let report = verifier.evaluate(&build).unwrap();
        assert_eq!(report.status, BuildProvenanceStatus::Incomplete);
    }

    #[test]
    fn identical_outputs_are_reproducible() {
        let verifier = BuildProvenanceVerifier::new(policy()).unwrap();
        let report = verifier
            .compare_reproducibility(&manifest("build-a"), &manifest("build-b"))
            .unwrap();
        assert!(report.reproducible);
    }

    #[test]
    fn output_mismatch_is_not_reproducible() {
        let verifier = BuildProvenanceVerifier::new(policy()).unwrap();
        let first = manifest("build-a");
        let mut second = manifest("build-b");
        second.output_artifacts[0].digest = "sha256:different".into();
        let report = verifier.compare_reproducibility(&first, &second).unwrap();
        assert!(!report.reproducible);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            BuildReproducibilityIssue::OutputDigestMismatch(path) if path == "bin/helicopter"
        )));
    }
}
