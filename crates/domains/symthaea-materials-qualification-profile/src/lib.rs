// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preregistered execution profiles for sealed-target materials qualification.
//!
//! `symthaea-materials-qualification` records what a Phase-A runner observed.
//! This crate adds the stronger, prior-commitment theorem required by
//! MAG-QUAL-002: the generator, execution environment, workspace, process
//! environment, benchmark-data mount set, network/cache policy, and orchestrator
//! must be committed before prediction generation and must match the later
//! generation receipt exactly.
//!
//! A matching profile does not prove that a hostile host contained no transformed
//! target information. It proves that the observed run agrees with a frozen
//! execution contract. Strong blindness still depends on a real isolated runner
//! and evaluator-only target escrow.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use symthaea_materials_benchmarks::{BenchmarkArtifactRef, BlindBenchmarkManifest};
use symthaea_materials_qualification::{
    BlindGenerationReceipt, GenerationNetworkPolicy, QualificationError,
};
use thiserror::Error;

/// Immutable Phase-A execution profile committed before prediction generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationExecutionProfile {
    /// Profile schema version.
    pub schema_version: u32,
    /// Stable profile lineage identifier.
    pub profile_id: String,
    /// Benchmark identity this profile qualifies.
    pub benchmark_id: String,
    /// Exact public benchmark manifest digest.
    pub manifest_sha256: String,
    /// Exact generator/model/workflow artifact expected in the submission.
    pub generator_artifact_sha256: String,
    /// Exact orchestrator implementation responsible for enforcing isolation.
    pub orchestrator_artifact_sha256: String,
    /// Stable runner/container identity expected during Phase A.
    pub runner_identity: String,
    /// Exact Nix closure/container/environment manifest expected during Phase A.
    pub execution_environment: BenchmarkArtifactRef,
    /// Exact preregistered workspace inventory expected during Phase A.
    pub workspace_inventory: BenchmarkArtifactRef,
    /// Exact preregistered sanitized process-environment inventory.
    pub process_environment_inventory: BenchmarkArtifactRef,
    /// Exact public generator-view digest derived from the benchmark manifest.
    pub search_phase_view_sha256: String,
    /// Exact benchmark-data artifacts allowed to be mounted into Phase A.
    pub mounted_benchmark_artifacts: Vec<BenchmarkArtifactRef>,
    /// Required network policy.
    pub network_policy: GenerationNetworkPolicy,
    /// Whether a reusable cache is permitted. Strong qualification requires false.
    pub cache_enabled: bool,
    /// Whether evaluator credentials may be visible. Strong qualification requires false.
    pub evaluator_credentials_present: bool,
    /// Whether a sealed target path may be visible. Strong qualification requires false.
    pub sealed_target_path_present: bool,
}

impl GenerationExecutionProfile {
    /// Validate this preregistered profile against the public benchmark manifest.
    pub fn validate_for(
        &self,
        manifest: &BlindBenchmarkManifest,
    ) -> Result<(), ProfileError> {
        if self.schema_version != 1 {
            return Err(ProfileError::UnsupportedProfileSchema(self.schema_version));
        }
        nonempty("profile_id", &self.profile_id)?;
        nonempty("runner_identity", &self.runner_identity)?;
        if self.benchmark_id != manifest.benchmark_id {
            return Err(ProfileError::BenchmarkIdentityMismatch);
        }
        manifest.validate()?;
        let manifest_sha = manifest.manifest_sha256()?;
        if self.manifest_sha256 != manifest_sha {
            return Err(ProfileError::ManifestDigestMismatch);
        }
        validate_sha256(&self.generator_artifact_sha256)?;
        validate_sha256(&self.orchestrator_artifact_sha256)?;
        validate_artifact(&self.execution_environment)?;
        validate_artifact(&self.workspace_inventory)?;
        validate_artifact(&self.process_environment_inventory)?;

        let expected_search_view = sha256_hex(&serde_json::to_vec(&manifest.search_phase_view()?)?);
        if self.search_phase_view_sha256 != expected_search_view {
            return Err(ProfileError::SearchPhaseViewDigestMismatch);
        }

        if self.network_policy != GenerationNetworkPolicy::Offline {
            return Err(ProfileError::NetworkMustBeOffline);
        }
        if self.cache_enabled {
            return Err(ProfileError::CacheMustBeDisabled);
        }
        if self.evaluator_credentials_present {
            return Err(ProfileError::EvaluatorCredentialsForbidden);
        }
        if self.sealed_target_path_present {
            return Err(ProfileError::SealedTargetPathForbidden);
        }

        for artifact in [
            &self.execution_environment,
            &self.workspace_inventory,
            &self.process_environment_inventory,
        ] {
            if artifact
                .sha256
                .eq_ignore_ascii_case(&manifest.sealed_targets_sha256)
            {
                return Err(ProfileError::SealedTargetIdentityInProfile);
            }
        }

        let expected = expected_generation_artifacts(manifest);
        if artifact_keys(&expected)? != artifact_keys(&self.mounted_benchmark_artifacts)? {
            return Err(ProfileError::MountedBenchmarkSetMismatch);
        }
        for artifact in &self.mounted_benchmark_artifacts {
            if artifact
                .sha256
                .eq_ignore_ascii_case(&manifest.sealed_targets_sha256)
            {
                return Err(ProfileError::SealedTargetIdentityInProfile);
            }
        }
        Ok(())
    }

    /// Deterministic digest of this complete preregistered execution profile.
    pub fn profile_sha256(&self) -> Result<String, ProfileError> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Evidence that one frozen MAG-QUAL-001 generation receipt matched a profile
/// committed before prediction generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationProfileBinding {
    /// Binding schema version.
    pub schema_version: u32,
    /// Exact preregistered profile digest.
    pub profile_sha256: String,
    /// Exact frozen generation-receipt digest.
    pub generation_receipt_sha256: String,
    /// Exact orchestrator implementation observed for the run.
    pub observed_orchestrator_artifact_sha256: String,
}

impl GenerationProfileBinding {
    /// Deterministic digest of the profile-to-receipt binding.
    pub fn binding_sha256(&self) -> Result<String, ProfileError> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Bind a validated MAG-QUAL-001 generation receipt to a preregistered profile.
///
/// `submission_bytes` are required so the underlying generation receipt is
/// revalidated against the exact frozen submission before the stronger profile
/// theorem is granted.
pub fn bind_generation_receipt(
    profile: &GenerationExecutionProfile,
    manifest: &BlindBenchmarkManifest,
    receipt: &BlindGenerationReceipt,
    submission_bytes: &[u8],
    observed_orchestrator_artifact_sha256: &str,
) -> Result<GenerationProfileBinding, ProfileError> {
    profile.validate_for(manifest)?;
    receipt.validate_against(manifest, submission_bytes)?;
    validate_sha256(observed_orchestrator_artifact_sha256)?;

    if profile.generator_artifact_sha256 != receipt.generator_artifact_sha256 {
        return Err(ProfileError::GeneratorArtifactMismatch);
    }
    if profile.search_phase_view_sha256 != receipt.search_phase_view_sha256 {
        return Err(ProfileError::SearchPhaseViewDigestMismatch);
    }
    if profile.runner_identity != receipt.isolation.runner_identity {
        return Err(ProfileError::RunnerIdentityMismatch);
    }
    if profile.execution_environment != receipt.isolation.execution_environment {
        return Err(ProfileError::ExecutionEnvironmentMismatch);
    }
    if profile.workspace_inventory != receipt.isolation.workspace_inventory {
        return Err(ProfileError::WorkspaceInventoryMismatch);
    }
    if profile.process_environment_inventory != receipt.isolation.process_environment_inventory {
        return Err(ProfileError::ProcessEnvironmentMismatch);
    }
    if profile.network_policy != receipt.isolation.network_policy {
        return Err(ProfileError::NetworkPolicyMismatch);
    }
    if profile.cache_enabled != receipt.isolation.cache_enabled {
        return Err(ProfileError::CachePolicyMismatch);
    }
    if profile.evaluator_credentials_present != receipt.isolation.evaluator_credentials_present {
        return Err(ProfileError::EvaluatorCredentialPolicyMismatch);
    }
    if profile.sealed_target_path_present != receipt.isolation.sealed_target_path_present {
        return Err(ProfileError::SealedTargetPathPolicyMismatch);
    }
    if artifact_keys(&profile.mounted_benchmark_artifacts)?
        != artifact_keys(&receipt.isolation.mounted_benchmark_artifacts)?
    {
        return Err(ProfileError::MountedBenchmarkSetMismatch);
    }
    if !profile
        .orchestrator_artifact_sha256
        .eq_ignore_ascii_case(observed_orchestrator_artifact_sha256)
    {
        return Err(ProfileError::OrchestratorArtifactMismatch);
    }

    Ok(GenerationProfileBinding {
        schema_version: 1,
        profile_sha256: profile.profile_sha256()?,
        generation_receipt_sha256: receipt.receipt_sha256()?,
        observed_orchestrator_artifact_sha256: observed_orchestrator_artifact_sha256
            .to_ascii_lowercase(),
    })
}

fn expected_generation_artifacts(manifest: &BlindBenchmarkManifest) -> Vec<BenchmarkArtifactRef> {
    let mut artifacts = Vec::with_capacity(manifest.allowed_training_artifacts.len() + 2);
    artifacts.push(manifest.search_space_artifact.clone());
    artifacts.extend(manifest.allowed_training_artifacts.iter().cloned());
    let split_artifact = match &manifest.split_policy {
        symthaea_materials_benchmarks::BenchmarkSplitPolicy::CompositionFamily {
            split_artifact,
        }
        | symthaea_materials_benchmarks::BenchmarkSplitPolicy::StructureCluster {
            split_artifact,
            ..
        }
        | symthaea_materials_benchmarks::BenchmarkSplitPolicy::FamilyAndStructure {
            split_artifact,
            ..
        } => split_artifact.clone(),
    };
    artifacts.push(split_artifact);
    artifacts
}

fn artifact_keys(artifacts: &[BenchmarkArtifactRef]) -> Result<HashSet<String>, ProfileError> {
    let mut keys = HashSet::new();
    let mut ids = HashSet::new();
    for artifact in artifacts {
        validate_artifact(artifact)?;
        if !ids.insert(artifact.artifact_id.clone()) {
            return Err(ProfileError::DuplicateArtifactId(artifact.artifact_id.clone()));
        }
        keys.insert(format!(
            "{}:{}",
            artifact.artifact_id,
            artifact.sha256.to_ascii_lowercase()
        ));
    }
    Ok(keys)
}

fn validate_artifact(artifact: &BenchmarkArtifactRef) -> Result<(), ProfileError> {
    nonempty("artifact_id", &artifact.artifact_id)?;
    validate_sha256(&artifact.sha256)
}

fn validate_sha256(value: &str) -> Result<(), ProfileError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(ProfileError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn nonempty(field: &'static str, value: &str) -> Result<(), ProfileError> {
    if value.trim().is_empty() {
        Err(ProfileError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// MAG-QUAL-002 preregistration/profile validation failure.
#[derive(Debug, Error)]
pub enum ProfileError {
    /// Underlying benchmark contract failed validation.
    #[error(transparent)]
    Benchmark(#[from] symthaea_materials_benchmarks::BenchmarkError),
    /// Underlying MAG-QUAL-001 receipt failed validation.
    #[error(transparent)]
    Qualification(#[from] QualificationError),
    /// JSON serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Unsupported profile schema.
    #[error("unsupported generation execution profile schema: {0}")]
    UnsupportedProfileSchema(u32),
    /// Required field empty.
    #[error("required profile field is empty: {0}")]
    EmptyField(&'static str),
    /// SHA-256 malformed.
    #[error("invalid profile SHA-256")]
    InvalidSha256,
    /// Profile benchmark ID did not match the public manifest.
    #[error("profile benchmark identity mismatch")]
    BenchmarkIdentityMismatch,
    /// Profile manifest digest did not match the public manifest.
    #[error("profile manifest digest mismatch")]
    ManifestDigestMismatch,
    /// Search-phase public view changed between preregistration and generation.
    #[error("profile search-phase view digest mismatch")]
    SearchPhaseViewDigestMismatch,
    /// Strong profile requires offline generation.
    #[error("profile network policy must be offline")]
    NetworkMustBeOffline,
    /// Strong profile forbids reusable generation caches.
    #[error("profile cache must be disabled")]
    CacheMustBeDisabled,
    /// Strong profile forbids evaluator credentials in Phase A.
    #[error("profile forbids evaluator credentials during generation")]
    EvaluatorCredentialsForbidden,
    /// Strong profile forbids a sealed-target path in Phase A.
    #[error("profile forbids sealed target path during generation")]
    SealedTargetPathForbidden,
    /// Sealed-target digest appeared where only generation artifacts belong.
    #[error("sealed target identity appears in preregistered generation profile")]
    SealedTargetIdentityInProfile,
    /// Expected/mounted benchmark-data set mismatch.
    #[error("preregistered benchmark-data mount set mismatch")]
    MountedBenchmarkSetMismatch,
    /// Duplicate artifact identifier.
    #[error("duplicate profile artifact ID: {0}")]
    DuplicateArtifactId(String),
    /// Generator implementation changed after preregistration.
    #[error("observed generator artifact does not match preregistered profile")]
    GeneratorArtifactMismatch,
    /// Runner identity changed after preregistration.
    #[error("observed runner identity does not match preregistered profile")]
    RunnerIdentityMismatch,
    /// Execution environment changed after preregistration.
    #[error("observed execution environment does not match preregistered profile")]
    ExecutionEnvironmentMismatch,
    /// Workspace inventory changed after preregistration.
    #[error("observed workspace inventory does not match preregistered profile")]
    WorkspaceInventoryMismatch,
    /// Sanitized process environment changed after preregistration.
    #[error("observed process environment does not match preregistered profile")]
    ProcessEnvironmentMismatch,
    /// Network policy changed after preregistration.
    #[error("observed network policy does not match preregistered profile")]
    NetworkPolicyMismatch,
    /// Cache policy changed after preregistration.
    #[error("observed cache policy does not match preregistered profile")]
    CachePolicyMismatch,
    /// Evaluator-secret visibility changed after preregistration.
    #[error("observed evaluator credential state does not match preregistered profile")]
    EvaluatorCredentialPolicyMismatch,
    /// Target-path visibility changed after preregistration.
    #[error("observed target-path state does not match preregistered profile")]
    SealedTargetPathPolicyMismatch,
    /// Orchestrator implementation changed after preregistration.
    #[error("observed orchestrator artifact does not match preregistered profile")]
    OrchestratorArtifactMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_materials_benchmarks::{
        BenchmarkMetricKind, BenchmarkMetricSpec, BenchmarkPrediction,
        BenchmarkPublicationSource, BenchmarkSplitPolicy, BenchmarkSubmission, PredictionValue,
    };
    use symthaea_materials_qualification::{
        GenerationIsolationAttestation, freeze_generation_receipt,
    };

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C64: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D64: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E64: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const F64: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn artifact(id: &str, sha: &str) -> BenchmarkArtifactRef {
        BenchmarkArtifactRef {
            artifact_id: id.to_string(),
            sha256: sha.to_string(),
        }
    }

    fn sealed_bytes() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "benchmark_id": "MAG-QUAL-002-fixture",
            "targets": [{
                "target_id": "heldout-1",
                "metric_id": "formation_energy",
                "expected": {"Scalar": {"value": -0.2, "unit": "eV/atom"}}
            }]
        }))
        .unwrap()
    }

    fn manifest() -> BlindBenchmarkManifest {
        BlindBenchmarkManifest {
            schema_version: 1,
            benchmark_id: "MAG-QUAL-002-fixture".to_string(),
            search_space_artifact: artifact("search-space", A64),
            allowed_training_artifacts: vec![artifact("training", B64)],
            split_policy: BenchmarkSplitPolicy::FamilyAndStructure {
                method_id: "family+structure-v1".to_string(),
                split_artifact: artifact("split", C64),
            },
            sealed_targets_sha256: sha256_hex(&sealed_bytes()),
            sources: vec![BenchmarkPublicationSource {
                source_id: "fixture-source".to_string(),
                title: "fixture".to_string(),
                doi: "10.0000/fixture".to_string(),
                date: "2026-01-01".to_string(),
            }],
            metrics: vec![BenchmarkMetricSpec {
                metric_id: "formation_energy".to_string(),
                property_id: "formation_energy".to_string(),
                kind: BenchmarkMetricKind::ScalarRegression,
                unit: "eV/atom".to_string(),
                condition_signature: "ground-state".to_string(),
            }],
        }
    }

    fn submission_bytes(manifest: &BlindBenchmarkManifest) -> Vec<u8> {
        serde_json::to_vec(&BenchmarkSubmission {
            benchmark_id: manifest.benchmark_id.clone(),
            manifest_sha256: manifest.manifest_sha256().unwrap(),
            generator_artifact_sha256: D64.to_string(),
            search_trace_sha256: E64.to_string(),
            predictions: vec![BenchmarkPrediction {
                target_id: "heldout-1".to_string(),
                metric_id: "formation_energy".to_string(),
                prediction: PredictionValue::Scalar {
                    value: -0.18,
                    unit: "eV/atom".to_string(),
                },
            }],
        })
        .unwrap()
    }

    fn isolation(manifest: &BlindBenchmarkManifest) -> GenerationIsolationAttestation {
        GenerationIsolationAttestation {
            schema_version: 1,
            runner_identity: "runner-image-v1".to_string(),
            execution_environment: artifact("execution-env", F64),
            workspace_inventory: artifact("workspace", E64),
            process_environment_inventory: artifact("process-env", D64),
            mounted_benchmark_artifacts: expected_generation_artifacts(manifest),
            network_policy: GenerationNetworkPolicy::Offline,
            cache_enabled: false,
            evaluator_credentials_present: false,
            sealed_target_path_present: false,
        }
    }

    fn profile(manifest: &BlindBenchmarkManifest) -> GenerationExecutionProfile {
        GenerationExecutionProfile {
            schema_version: 1,
            profile_id: "mag-qual-002-fixture-profile".to_string(),
            benchmark_id: manifest.benchmark_id.clone(),
            manifest_sha256: manifest.manifest_sha256().unwrap(),
            generator_artifact_sha256: D64.to_string(),
            orchestrator_artifact_sha256: C64.to_string(),
            runner_identity: "runner-image-v1".to_string(),
            execution_environment: artifact("execution-env", F64),
            workspace_inventory: artifact("workspace", E64),
            process_environment_inventory: artifact("process-env", D64),
            search_phase_view_sha256: sha256_hex(
                &serde_json::to_vec(&manifest.search_phase_view().unwrap()).unwrap(),
            ),
            mounted_benchmark_artifacts: expected_generation_artifacts(manifest),
            network_policy: GenerationNetworkPolicy::Offline,
            cache_enabled: false,
            evaluator_credentials_present: false,
            sealed_target_path_present: false,
        }
    }

    #[test]
    fn exact_preregistered_profile_binds_generation_receipt() {
        let manifest = manifest();
        let bytes = submission_bytes(&manifest);
        let receipt = freeze_generation_receipt(&manifest, &bytes, isolation(&manifest)).unwrap();
        let profile = profile(&manifest);
        let binding = bind_generation_receipt(&profile, &manifest, &receipt, &bytes, C64).unwrap();
        assert_eq!(binding.profile_sha256, profile.profile_sha256().unwrap());
        assert_eq!(binding.generation_receipt_sha256, receipt.receipt_sha256().unwrap());
        assert_eq!(binding.binding_sha256().unwrap().len(), 64);
    }

    #[test]
    fn observed_workspace_drift_fails_closed() {
        let manifest = manifest();
        let bytes = submission_bytes(&manifest);
        let mut observed = isolation(&manifest);
        observed.workspace_inventory = artifact("workspace", A64);
        let receipt = freeze_generation_receipt(&manifest, &bytes, observed).unwrap();
        assert!(matches!(
            bind_generation_receipt(&profile(&manifest), &manifest, &receipt, &bytes, C64),
            Err(ProfileError::WorkspaceInventoryMismatch)
        ));
    }

    #[test]
    fn changed_orchestrator_fails_even_when_run_attestation_matches() {
        let manifest = manifest();
        let bytes = submission_bytes(&manifest);
        let receipt = freeze_generation_receipt(&manifest, &bytes, isolation(&manifest)).unwrap();
        assert!(matches!(
            bind_generation_receipt(&profile(&manifest), &manifest, &receipt, &bytes, A64),
            Err(ProfileError::OrchestratorArtifactMismatch)
        ));
    }

    #[test]
    fn preregistered_profile_rejects_extra_benchmark_data() {
        let manifest = manifest();
        let mut profile = profile(&manifest);
        profile
            .mounted_benchmark_artifacts
            .push(artifact("extra-dataset", F64));
        assert!(matches!(
            profile.validate_for(&manifest),
            Err(ProfileError::MountedBenchmarkSetMismatch)
        ));
    }

    #[test]
    fn profile_identity_changes_when_generator_or_environment_changes() {
        let manifest = manifest();
        let a = profile(&manifest);
        let mut b = a.clone();
        b.generator_artifact_sha256 = A64.to_string();
        assert_ne!(a.profile_sha256().unwrap(), b.profile_sha256().unwrap());

        let mut c = a.clone();
        c.execution_environment = artifact("execution-env", A64);
        assert_ne!(a.profile_sha256().unwrap(), c.profile_sha256().unwrap());
    }
}
