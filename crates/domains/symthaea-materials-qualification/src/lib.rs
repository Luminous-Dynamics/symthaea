// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Two-phase sealed-target qualification for materials-discovery benchmarks.
//!
//! Benchmark definitions and qualification authority are intentionally separate.
//! `symthaea-materials-benchmarks` defines what gets measured; this crate decides
//! whether a particular run was frozen under a strong blindness contract before
//! evaluator-only target bytes were disclosed.
//!
//! The strong MAG-QUAL-001 profile requires Phase A to run offline, without
//! reusable caches or evaluator credentials, with an exact manifest-derived data
//! mount. Phase B then scores only the byte-identical submission frozen by Phase A.
//! A benchmark PASS remains reproduction evidence and does not advance MAT-001
//! material-discovery authority by itself.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use symthaea_materials_benchmarks::{
    BenchmarkArtifactRef, BenchmarkError, BenchmarkScorecard, BenchmarkSplitPolicy,
    BenchmarkSubmission, BlindBenchmarkManifest, PredictionValue, evaluate_blind_submission,
};
use thiserror::Error;

/// Network policy visible to the Phase-A generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationNetworkPolicy {
    /// Network namespace disabled after exact inputs are mounted.
    Offline,
    /// Development-only mode; never accepted by the strong qualification profile.
    Unrestricted,
}

/// Exact execution-state attestation for target-blind prediction generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationIsolationAttestation {
    /// Attestation schema version.
    pub schema_version: u32,
    /// Stable runner/container identity.
    pub runner_identity: String,
    /// Exact Nix closure/container/environment manifest.
    pub execution_environment: BenchmarkArtifactRef,
    /// Exact inventory of filesystem content visible to the generator.
    pub workspace_inventory: BenchmarkArtifactRef,
    /// Exact sanitized process-environment inventory.
    pub process_environment_inventory: BenchmarkArtifactRef,
    /// Benchmark data artifacts mounted into Phase A.
    ///
    /// Under the strong profile this must match the public manifest exactly:
    /// search space + allowed training artifacts + split artifact, with no extras.
    pub mounted_benchmark_artifacts: Vec<BenchmarkArtifactRef>,
    /// Network state during actual prediction generation.
    pub network_policy: GenerationNetworkPolicy,
    /// Whether a reusable cache was mounted during prediction generation.
    pub cache_enabled: bool,
    /// Whether evaluator-only credentials/secrets were present in Phase A.
    pub evaluator_credentials_present: bool,
    /// Whether an evaluator target path/file was present in the Phase-A workspace.
    pub sealed_target_path_present: bool,
}

impl GenerationIsolationAttestation {
    /// Validate the strong MAG-QUAL-001 generation profile against a benchmark manifest.
    pub fn validate_for(
        &self,
        manifest: &BlindBenchmarkManifest,
    ) -> Result<(), QualificationError> {
        if self.schema_version != 1 {
            return Err(QualificationError::UnsupportedIsolationSchema(
                self.schema_version,
            ));
        }
        nonempty("runner_identity", &self.runner_identity)?;
        validate_artifact(&self.execution_environment)?;
        validate_artifact(&self.workspace_inventory)?;
        validate_artifact(&self.process_environment_inventory)?;

        if self.network_policy != GenerationNetworkPolicy::Offline {
            return Err(QualificationError::GenerationNetworkNotOffline);
        }
        if self.cache_enabled {
            return Err(QualificationError::GenerationCacheEnabled);
        }
        if self.evaluator_credentials_present {
            return Err(QualificationError::EvaluatorCredentialsVisibleDuringGeneration);
        }
        if self.sealed_target_path_present {
            return Err(QualificationError::SealedTargetPathVisibleDuringGeneration);
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
                return Err(QualificationError::SealedTargetVisibleDuringGeneration);
            }
        }

        let expected = expected_generation_artifacts(manifest)?;
        let expected_keys = artifact_keys(&expected)?;
        let mounted_keys = artifact_keys(&self.mounted_benchmark_artifacts)?;

        for artifact in &self.mounted_benchmark_artifacts {
            if artifact
                .sha256
                .eq_ignore_ascii_case(&manifest.sealed_targets_sha256)
            {
                return Err(QualificationError::SealedTargetVisibleDuringGeneration);
            }
        }

        if expected_keys != mounted_keys {
            let missing = expected_keys
                .difference(&mounted_keys)
                .cloned()
                .collect::<Vec<_>>();
            let unexpected = mounted_keys
                .difference(&expected_keys)
                .cloned()
                .collect::<Vec<_>>();
            return Err(QualificationError::MountedBenchmarkSetMismatch {
                missing,
                unexpected,
            });
        }
        Ok(())
    }

    /// Deterministic digest of the complete Phase-A isolation attestation.
    pub fn attestation_sha256(&self) -> Result<String, QualificationError> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Frozen prediction-generation receipt created before target disclosure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindGenerationReceipt {
    /// Receipt schema version.
    pub schema_version: u32,
    /// Benchmark identity.
    pub benchmark_id: String,
    /// Exact public benchmark-manifest digest.
    pub manifest_sha256: String,
    /// Public SHA commitment to evaluator-only target bytes.
    pub sealed_targets_sha256: String,
    /// Exact serialized submission digest.
    pub submission_sha256: String,
    /// Generator/model/workflow artifact bound by the submission.
    pub generator_artifact_sha256: String,
    /// Search trace / SearchMemory artifact bound by the submission.
    pub search_trace_sha256: String,
    /// Exact public generator view derived from the benchmark manifest.
    pub search_phase_view_sha256: String,
    /// Complete Phase-A isolation attestation.
    pub isolation: GenerationIsolationAttestation,
}

impl BlindGenerationReceipt {
    /// Revalidate this receipt against the public manifest and exact submission bytes.
    pub fn validate_against(
        &self,
        manifest: &BlindBenchmarkManifest,
        submission_bytes: &[u8],
    ) -> Result<(), QualificationError> {
        if self.schema_version != 1 {
            return Err(QualificationError::UnsupportedReceiptSchema(
                self.schema_version,
            ));
        }
        manifest.validate()?;
        let submission = parse_submission(submission_bytes)?;
        validate_submission(manifest, &submission)?;

        let manifest_sha = manifest.manifest_sha256()?;
        if self.benchmark_id != manifest.benchmark_id {
            return Err(QualificationError::BenchmarkIdentityMismatch);
        }
        if self.manifest_sha256 != manifest_sha {
            return Err(QualificationError::ManifestDigestMismatch);
        }
        if self.sealed_targets_sha256 != manifest.sealed_targets_sha256 {
            return Err(QualificationError::SealedTargetCommitmentMismatch);
        }
        if self.submission_sha256 != sha256_hex(submission_bytes) {
            return Err(QualificationError::FrozenSubmissionDigestMismatch);
        }
        if self.generator_artifact_sha256 != submission.generator_artifact_sha256
            || self.search_trace_sha256 != submission.search_trace_sha256
        {
            return Err(QualificationError::SubmissionLineageMismatch);
        }
        if self.search_phase_view_sha256 != search_phase_view_sha256(manifest)? {
            return Err(QualificationError::SearchPhaseViewMismatch);
        }
        self.isolation.validate_for(manifest)
    }

    /// Deterministic SHA-256 of the complete generation receipt.
    pub fn receipt_sha256(&self) -> Result<String, QualificationError> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Freeze exact prediction bytes and the Phase-A environment before target disclosure.
pub fn freeze_generation_receipt(
    manifest: &BlindBenchmarkManifest,
    submission_bytes: &[u8],
    isolation: GenerationIsolationAttestation,
) -> Result<BlindGenerationReceipt, QualificationError> {
    manifest.validate()?;
    isolation.validate_for(manifest)?;
    let submission = parse_submission(submission_bytes)?;
    validate_submission(manifest, &submission)?;

    let receipt = BlindGenerationReceipt {
        schema_version: 1,
        benchmark_id: manifest.benchmark_id.clone(),
        manifest_sha256: manifest.manifest_sha256()?,
        sealed_targets_sha256: manifest.sealed_targets_sha256.clone(),
        submission_sha256: sha256_hex(submission_bytes),
        generator_artifact_sha256: submission.generator_artifact_sha256.clone(),
        search_trace_sha256: submission.search_trace_sha256.clone(),
        search_phase_view_sha256: search_phase_view_sha256(manifest)?,
        isolation,
    };
    receipt.validate_against(manifest, submission_bytes)?;
    Ok(receipt)
}

/// Evaluator-side environment bindings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationEnvironment {
    /// Exact evaluator executable/workflow artifact SHA-256.
    pub evaluator_artifact_sha256: String,
    /// Exact Nix closure/container/evaluator-environment SHA-256.
    pub execution_environment_sha256: String,
}

impl EvaluationEnvironment {
    fn validate(&self) -> Result<(), QualificationError> {
        validate_sha256(&self.evaluator_artifact_sha256)?;
        validate_sha256(&self.execution_environment_sha256)
    }
}

/// Phase-B receipt binding target disclosure and scoring to frozen Phase-A bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlindEvaluationReceipt {
    /// Receipt schema version.
    pub schema_version: u32,
    /// SHA-256 of the complete frozen Phase-A receipt.
    pub generation_receipt_sha256: String,
    /// Exact Phase-A submission digest.
    pub submission_sha256: String,
    /// Exact evaluator/workflow artifact.
    pub evaluator_artifact_sha256: String,
    /// Exact evaluator execution environment.
    pub evaluation_environment_sha256: String,
    /// Exact target bytes used for scoring.
    pub sealed_targets_sha256: String,
    /// Exact serialized scorecard digest.
    pub scorecard_sha256: String,
    /// Benchmark scorecard. This remains benchmark evidence, not MAT-001 authority.
    pub scorecard: BenchmarkScorecard,
}

impl BlindEvaluationReceipt {
    /// Deterministic SHA-256 of the complete evaluation receipt.
    pub fn receipt_sha256(&self) -> Result<String, QualificationError> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Evaluate only the byte-identical submission frozen by Phase A.
pub fn evaluate_frozen_generation(
    manifest: &BlindBenchmarkManifest,
    submission_bytes: &[u8],
    generation_receipt: &BlindGenerationReceipt,
    sealed_target_bytes: &[u8],
    environment: &EvaluationEnvironment,
) -> Result<BlindEvaluationReceipt, QualificationError> {
    generation_receipt.validate_against(manifest, submission_bytes)?;
    environment.validate()?;

    let submission = parse_submission(submission_bytes)?;
    let scorecard = evaluate_blind_submission(manifest, &submission, sealed_target_bytes)?;
    let scorecard_sha256 = sha256_hex(&serde_json::to_vec(&scorecard)?);

    Ok(BlindEvaluationReceipt {
        schema_version: 1,
        generation_receipt_sha256: generation_receipt.receipt_sha256()?,
        submission_sha256: generation_receipt.submission_sha256.clone(),
        evaluator_artifact_sha256: environment
            .evaluator_artifact_sha256
            .to_ascii_lowercase(),
        evaluation_environment_sha256: environment
            .execution_environment_sha256
            .to_ascii_lowercase(),
        sealed_targets_sha256: sha256_hex(sealed_target_bytes),
        scorecard_sha256,
        scorecard,
    })
}

/// Deterministic digest of the exact public view allowed into Phase A.
pub fn search_phase_view_sha256(
    manifest: &BlindBenchmarkManifest,
) -> Result<String, QualificationError> {
    let view = manifest.search_phase_view()?;
    Ok(sha256_hex(&serde_json::to_vec(&view)?))
}

fn parse_submission(bytes: &[u8]) -> Result<BenchmarkSubmission, QualificationError> {
    serde_json::from_slice(bytes)
        .map_err(|error| QualificationError::SubmissionSchema(error.to_string()))
}

fn validate_submission(
    manifest: &BlindBenchmarkManifest,
    submission: &BenchmarkSubmission,
) -> Result<(), QualificationError> {
    nonempty("submission benchmark_id", &submission.benchmark_id)?;
    validate_sha256(&submission.manifest_sha256)?;
    validate_sha256(&submission.generator_artifact_sha256)?;
    validate_sha256(&submission.search_trace_sha256)?;
    if submission.benchmark_id != manifest.benchmark_id {
        return Err(QualificationError::BenchmarkIdentityMismatch);
    }
    if submission.manifest_sha256 != manifest.manifest_sha256()? {
        return Err(QualificationError::ManifestDigestMismatch);
    }

    let metric_ids = manifest
        .metrics
        .iter()
        .map(|metric| metric.metric_id.as_str())
        .collect::<HashSet<_>>();
    let mut target_ids = HashSet::new();
    for prediction in &submission.predictions {
        nonempty("target_id", &prediction.target_id)?;
        nonempty("prediction metric_id", &prediction.metric_id)?;
        if !metric_ids.contains(prediction.metric_id.as_str()) {
            return Err(QualificationError::UnknownPredictionMetric(
                prediction.metric_id.clone(),
            ));
        }
        if !target_ids.insert(prediction.target_id.as_str()) {
            return Err(QualificationError::DuplicatePredictionTarget(
                prediction.target_id.clone(),
            ));
        }
        match &prediction.prediction {
            PredictionValue::Scalar { value, unit } => {
                if !value.is_finite() {
                    return Err(QualificationError::NonFinitePrediction {
                        target_id: prediction.target_id.clone(),
                    });
                }
                nonempty("prediction unit", unit)?;
            }
            PredictionValue::Binary { .. } => {}
        }
    }
    Ok(())
}

fn expected_generation_artifacts(
    manifest: &BlindBenchmarkManifest,
) -> Result<Vec<BenchmarkArtifactRef>, QualificationError> {
    manifest.validate()?;
    let mut artifacts = Vec::with_capacity(manifest.allowed_training_artifacts.len() + 2);
    artifacts.push(manifest.search_space_artifact.clone());
    artifacts.extend(manifest.allowed_training_artifacts.iter().cloned());
    let split = match &manifest.split_policy {
        BenchmarkSplitPolicy::CompositionFamily { split_artifact }
        | BenchmarkSplitPolicy::StructureCluster { split_artifact, .. }
        | BenchmarkSplitPolicy::FamilyAndStructure { split_artifact, .. } => split_artifact,
    };
    artifacts.push(split.clone());
    Ok(artifacts)
}

fn artifact_keys(
    artifacts: &[BenchmarkArtifactRef],
) -> Result<HashSet<String>, QualificationError> {
    let mut ids = HashSet::new();
    let mut keys = HashSet::new();
    for artifact in artifacts {
        validate_artifact(artifact)?;
        if !ids.insert(artifact.artifact_id.as_str()) {
            return Err(QualificationError::DuplicateArtifactId(
                artifact.artifact_id.clone(),
            ));
        }
        keys.insert(format!(
            "{}:{}",
            artifact.artifact_id,
            artifact.sha256.to_ascii_lowercase()
        ));
    }
    Ok(keys)
}

fn validate_artifact(artifact: &BenchmarkArtifactRef) -> Result<(), QualificationError> {
    nonempty("artifact_id", &artifact.artifact_id)?;
    validate_sha256(&artifact.sha256)
}

fn validate_sha256(value: &str) -> Result<(), QualificationError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(QualificationError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn nonempty(field: &'static str, value: &str) -> Result<(), QualificationError> {
    if value.trim().is_empty() {
        Err(QualificationError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// MAG-QUAL-001 qualification failure.
#[derive(Debug, Error)]
pub enum QualificationError {
    /// Underlying benchmark contract failed validation or sealed evaluation.
    #[error(transparent)]
    Benchmark(#[from] BenchmarkError),
    /// JSON serialization failed while binding a receipt.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Submission bytes were not valid public submission JSON.
    #[error("submission schema error: {0}")]
    SubmissionSchema(String),
    /// Receipt schema unsupported.
    #[error("unsupported generation receipt schema: {0}")]
    UnsupportedReceiptSchema(u32),
    /// Isolation schema unsupported.
    #[error("unsupported generation isolation schema: {0}")]
    UnsupportedIsolationSchema(u32),
    /// Required text empty.
    #[error("required qualification field is empty: {0}")]
    EmptyField(&'static str),
    /// SHA-256 malformed.
    #[error("invalid qualification SHA-256")]
    InvalidSha256,
    /// Benchmark identity differed across phases.
    #[error("qualification benchmark identity mismatch")]
    BenchmarkIdentityMismatch,
    /// Public manifest digest differed across phases.
    #[error("qualification manifest digest mismatch")]
    ManifestDigestMismatch,
    /// Receipt did not preserve the public target commitment.
    #[error("sealed-target commitment mismatch")]
    SealedTargetCommitmentMismatch,
    /// Exact serialized submission changed after Phase A freeze.
    #[error("frozen submission bytes changed after generation")]
    FrozenSubmissionDigestMismatch,
    /// Generator/search-trace lineage changed relative to frozen submission.
    #[error("frozen submission lineage mismatch")]
    SubmissionLineageMismatch,
    /// Public search-phase view changed relative to the frozen receipt.
    #[error("public search-phase view changed after generation")]
    SearchPhaseViewMismatch,
    /// Strong qualification requires no network in prediction generation.
    #[error("generation network must be offline for strong qualification")]
    GenerationNetworkNotOffline,
    /// Strong qualification forbids reusable caches in Phase A.
    #[error("generation cache must be disabled for strong qualification")]
    GenerationCacheEnabled,
    /// Evaluator-only credentials leaked into Phase A.
    #[error("evaluator credentials were visible during generation")]
    EvaluatorCredentialsVisibleDuringGeneration,
    /// Evaluator target file/path existed in Phase A.
    #[error("sealed target path was visible during generation")]
    SealedTargetPathVisibleDuringGeneration,
    /// Exact sealed target bytes appeared among a Phase-A artifact binding.
    #[error("sealed target artifact was visible during generation")]
    SealedTargetVisibleDuringGeneration,
    /// Mounted benchmark data did not equal the public manifest-derived input set.
    #[error("mounted benchmark artifact set mismatch; missing={missing:?}, unexpected={unexpected:?}")]
    MountedBenchmarkSetMismatch {
        /// Required artifact keys absent from Phase A.
        missing: Vec<String>,
        /// Extra benchmark-data artifact keys visible in Phase A.
        unexpected: Vec<String>,
    },
    /// Artifact identifiers repeated within an inventory.
    #[error("duplicate artifact ID: {0}")]
    DuplicateArtifactId(String),
    /// Submission referenced a metric not declared by the public manifest.
    #[error("unknown prediction metric: {0}")]
    UnknownPredictionMetric(String),
    /// Submission repeated a target ID.
    #[error("duplicate prediction target: {0}")]
    DuplicatePredictionTarget(String),
    /// Submitted scalar was NaN or infinity.
    #[error("non-finite prediction for target: {target_id}")]
    NonFinitePrediction {
        /// Target whose prediction was non-finite.
        target_id: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_materials_benchmarks::{
        BenchmarkMetricKind, BenchmarkMetricSpec, BenchmarkPrediction, BenchmarkPublicationSource,
    };

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C64: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D64: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E64: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const F64: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const ZERO64: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    const ONE64: &str = "1111111111111111111111111111111111111111111111111111111111111111";

    fn artifact(id: &str, sha: &str) -> BenchmarkArtifactRef {
        BenchmarkArtifactRef {
            artifact_id: id.to_string(),
            sha256: sha.to_string(),
        }
    }

    fn sealed_targets() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "benchmark_id": "MAG-QUAL-fixture",
            "targets": [{
                "target_id": "Fe5Co18Zr6-k1",
                "metric_id": "k1",
                "expected": { "Scalar": { "value": 1.1, "unit": "MJ/m3" } }
            }]
        }))
        .unwrap()
    }

    fn manifest() -> BlindBenchmarkManifest {
        BlindBenchmarkManifest {
            schema_version: 1,
            benchmark_id: "MAG-QUAL-fixture".to_string(),
            search_space_artifact: artifact("search-space", A64),
            allowed_training_artifacts: vec![artifact("training", B64)],
            split_policy: BenchmarkSplitPolicy::FamilyAndStructure {
                method_id: "family+structure-v1".to_string(),
                split_artifact: artifact("split", C64),
            },
            sealed_targets_sha256: sha256_hex(&sealed_targets()),
            sources: vec![BenchmarkPublicationSource {
                source_id: "fixture-source".to_string(),
                title: "Fixture source".to_string(),
                doi: "10.0000/fixture".to_string(),
                date: "2026-01-01".to_string(),
            }],
            metrics: vec![BenchmarkMetricSpec {
                metric_id: "k1".to_string(),
                property_id: "magnetocrystalline_anisotropy_k1".to_string(),
                kind: BenchmarkMetricKind::ScalarRegression,
                unit: "MJ/m3".to_string(),
                condition_signature: "T=0K".to_string(),
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
                target_id: "Fe5Co18Zr6-k1".to_string(),
                metric_id: "k1".to_string(),
                prediction: PredictionValue::Scalar {
                    value: 1.0,
                    unit: "MJ/m3".to_string(),
                },
            }],
        })
        .unwrap()
    }

    fn isolation() -> GenerationIsolationAttestation {
        GenerationIsolationAttestation {
            schema_version: 1,
            runner_identity: "offline-container-fixture".to_string(),
            execution_environment: artifact("generation-environment", F64),
            workspace_inventory: artifact("workspace-inventory", ZERO64),
            process_environment_inventory: artifact("process-environment", ONE64),
            mounted_benchmark_artifacts: vec![
                artifact("search-space", A64),
                artifact("training", B64),
                artifact("split", C64),
            ],
            network_policy: GenerationNetworkPolicy::Offline,
            cache_enabled: false,
            evaluator_credentials_present: false,
            sealed_target_path_present: false,
        }
    }

    #[test]
    fn freezes_byte_exact_submission_before_target_disclosure() {
        let manifest = manifest();
        let bytes = submission_bytes(&manifest);
        let receipt = freeze_generation_receipt(&manifest, &bytes, isolation()).unwrap();
        assert_eq!(receipt.submission_sha256, sha256_hex(&bytes));
        receipt.validate_against(&manifest, &bytes).unwrap();
    }

    #[test]
    fn semantically_equivalent_but_byte_changed_submission_is_rejected() {
        let manifest = manifest();
        let bytes = submission_bytes(&manifest);
        let receipt = freeze_generation_receipt(&manifest, &bytes, isolation()).unwrap();
        let mut changed = bytes.clone();
        changed.push(b'\n');
        assert!(matches!(
            receipt.validate_against(&manifest, &changed),
            Err(QualificationError::FrozenSubmissionDigestMismatch)
        ));
    }

    #[test]
    fn extra_benchmark_data_mount_is_rejected() {
        let manifest = manifest();
        let bytes = submission_bytes(&manifest);
        let mut isolation = isolation();
        isolation
            .mounted_benchmark_artifacts
            .push(artifact("extra", F64));
        assert!(matches!(
            freeze_generation_receipt(&manifest, &bytes, isolation),
            Err(QualificationError::MountedBenchmarkSetMismatch { .. })
        ));
    }

    #[test]
    fn sealed_target_cannot_be_mounted_in_generation() {
        let manifest = manifest();
        let bytes = submission_bytes(&manifest);
        let mut isolation = isolation();
        isolation.mounted_benchmark_artifacts[0] = artifact(
            "search-space",
            &manifest.sealed_targets_sha256,
        );
        assert!(matches!(
            freeze_generation_receipt(&manifest, &bytes, isolation),
            Err(QualificationError::SealedTargetVisibleDuringGeneration)
        ));
    }

    #[test]
    fn unrestricted_network_is_not_strong_blind_qualification() {
        let manifest = manifest();
        let bytes = submission_bytes(&manifest);
        let mut isolation = isolation();
        isolation.network_policy = GenerationNetworkPolicy::Unrestricted;
        assert!(matches!(
            freeze_generation_receipt(&manifest, &bytes, isolation),
            Err(QualificationError::GenerationNetworkNotOffline)
        ));
    }

    #[test]
    fn evaluator_scores_only_the_frozen_submission() {
        let manifest = manifest();
        let bytes = submission_bytes(&manifest);
        let generation = freeze_generation_receipt(&manifest, &bytes, isolation()).unwrap();
        let environment = EvaluationEnvironment {
            evaluator_artifact_sha256: A64.to_string(),
            execution_environment_sha256: B64.to_string(),
        };
        let evaluated = evaluate_frozen_generation(
            &manifest,
            &bytes,
            &generation,
            &sealed_targets(),
            &environment,
        )
        .unwrap();
        assert_eq!(evaluated.submission_sha256, sha256_hex(&bytes));
        assert_eq!(evaluated.sealed_targets_sha256, manifest.sealed_targets_sha256);
        assert_eq!(evaluated.scorecard.scalar_count, 1);
        assert!((evaluated.scorecard.scalar_mae.unwrap() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn wrong_target_bytes_remain_a_hard_failure() {
        let manifest = manifest();
        let bytes = submission_bytes(&manifest);
        let generation = freeze_generation_receipt(&manifest, &bytes, isolation()).unwrap();
        let environment = EvaluationEnvironment {
            evaluator_artifact_sha256: A64.to_string(),
            execution_environment_sha256: B64.to_string(),
        };
        assert!(matches!(
            evaluate_frozen_generation(
                &manifest,
                &bytes,
                &generation,
                br#"{\"wrong\":\"targets\"}"#,
                &environment,
            ),
            Err(QualificationError::Benchmark(
                BenchmarkError::SealedTargetDigestMismatch
            ))
        ));
    }
}
