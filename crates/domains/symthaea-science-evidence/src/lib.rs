// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-domain scientific experiment manifests and verification receipts.
//!
//! The contract is designed for computational physics, quantum simulation,
//! causal-set experiments, GR solvers, sensor-analysis pipelines and other
//! scientific crates. It binds:
//! - exact source commit and working-tree cleanliness;
//! - toolchain and platform labels;
//! - unit / coordinate conventions and equations / assumptions;
//! - experiment configuration, seeds, baseline and uncertainty method;
//! - content digests for inputs and output artifacts;
//! - claims, non-claims and falsification conditions;
//! - check status and exact-head evidence.
//!
//! Crucial rule: queued, running, skipped, failed and passed are different
//! states. A passed check must be attached to the exact source commit and to a
//! non-empty log artifact digest. A run cannot be marked passed if any listed
//! required check is queued, skipped, running, failed or missing.
//!
//! This crate creates local reproducibility records. It does not authenticate
//! the caller, sign receipts, establish physical truth or replace Mycelix
//! attestation. BLAKE3 digests bind bytes; they do not prove that the bytes
//! are scientifically correct.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

pub const SCIENCE_EVIDENCE_SCHEMA: &str = "symthaea-science-evidence-v2";

/// Theory prediction completeness and cross-model comparability contracts.
pub mod model_discrimination;
pub mod model_recovery;
pub mod theory_prediction;

pub use model_discrimination::{
    CandidateExperiment, DetectionAwareCandidateExperiment, DetectionAwareExperimentScore,
    DetectionAwareModelDiscriminationReport, DetectionAwarePriorScenarioEIG,
    DiscriminationHypothesis, ExperimentBlocker, ExperimentDiscriminationScore,
    ExperimentDiscriminationStatus, GaussianModelPrediction, ModelDetectionProbability,
    MAX_DISCRIMINATION_EXPERIMENTS, MAX_DISCRIMINATION_HYPOTHESES, MAX_EIG_QUADRATURE_INTERVALS,
    MAX_MODEL_PRIOR_SCENARIOS, ModelDiscriminationConfig, ModelDiscriminationError,
    ModelDiscriminationReport, ModelPriorScenario, ModelUniverseCoverage, PriorScenarioEIG,
    gaussian_predictive_kl_nats, rank_detection_aware_model_discriminating_experiments,
    rank_model_discriminating_experiments,
};
pub use model_recovery::{
    MAX_MODEL_RECOVERY_GENERATORS, MAX_MODEL_RECOVERY_TRIALS, ModelRecoveryError,
    ModelRecoveryReport, RecoveryGenerator, evaluate_model_recovery,
};
pub use theory_prediction::{
    ComparisonBlocker, MeasurementNoiseModel, ModelPredictionComparison, NoiseCalibrationStatus,
    OTHER_UNRESOLVED_MODEL_ID, ObservableContract, PredictionCoverage, PredictionDisposition,
    PredictiveUncertaintyStatus, THEORY_PREDICTION_SCHEMA, TheoryModel, TheoryModelRole,
    TheoryPredictionError, TheoryPredictionLedger, TheoryPredictionRecord,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ClaimBoundary {
    /// Code-path or data-contract implementation check only.
    ImplementationCheck,
    /// Reproduces an algebraic identity or a mathematically derived reference.
    AnalyticReference,
    /// Local numerical simulation without external physical measurement.
    NumericalSimulation,
    /// Instrumental observation with an explicit calibration chain.
    CalibratedObservation,
    /// Execution on named external hardware or a physical experimental setup.
    HardwareOrLaboratoryDemonstration,
    /// Independent implementation or laboratory reproduces the result.
    IndependentReplication,
    /// Interpretation that links a model to a physical theory.
    PhysicalInterpretation,
    /// Readiness claim for a deployed or production system.
    ProductionCapability,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum VerificationStatus {
    NotRun,
    Queued,
    Running,
    Passed,
    Failed,
    Skipped,
}

impl VerificationStatus {
    pub fn is_passed(self) -> bool {
        self == Self::Passed
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactDigest {
    /// Repository-relative or caller-declared stable artifact path.
    pub path: String,
    /// Lowercase 64-character BLAKE3 hexadecimal digest of the raw bytes.
    pub blake3_hex: String,
    pub byte_length: u64,
}

impl ArtifactDigest {
    pub fn from_bytes(path: impl Into<String>, bytes: &[u8]) -> Self {
        Self {
            path: path.into(),
            blake3_hex: blake3::hash(bytes).to_hex().to_string(),
            byte_length: bytes.len() as u64,
        }
    }

    /// Recompute both the byte length and BLAKE3 digest against actual bytes.
    pub fn verify_bytes(&self, bytes: &[u8]) -> Result<(), ManifestError> {
        self.validate()?;
        let actual_digest = blake3::hash(bytes).to_hex().to_string();
        if self.byte_length != bytes.len() as u64 || self.blake3_hex != actual_digest {
            return Err(ManifestError::ArtifactBytesMismatch(self.path.clone()));
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require_nonempty("artifact path", &self.path)?;
        if !is_lower_hex_digest(&self.blake3_hex) {
            return Err(ManifestError::InvalidDigest {
                field: format!("artifact {}", self.path),
            });
        }
        if self.byte_length == 0 {
            return Err(ManifestError::EmptyArtifact(self.path.clone()));
        }
        Ok(())
    }
}

/// Immutable experiment specification. Configuration is represented as
/// JSON data (not Debug text) and serialized in the struct's declared field
/// order. The digest is deterministic for this schema/version; consumers
/// should preserve the schema identifier when archiving the manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperimentManifest {
    pub schema: String,
    pub study_id: String,
    pub protocol_id: String,
    /// Full 40-character Git commit SHA, not a branch name or abbreviated SHA.
    pub source_commit_sha: String,
    pub source_tree_clean: bool,
    pub toolchain: String,
    pub platform: String,
    pub coordinate_and_unit_conventions: Vec<String>,
    pub equations_and_assumptions: Vec<String>,
    pub configuration: serde_json::Value,
    pub input_artifacts: Vec<ArtifactDigest>,
    /// Required when there are no external input files, e.g. a closed-form proof.
    pub input_scope_note: String,
    pub seed_policy: String,
    pub seeds: Vec<u64>,
    /// State the baseline explicitly; "none: analytic identity" is valid.
    pub baseline_description: String,
    pub primary_metrics: Vec<String>,
    /// Complete, frozen set of checks required for a run to qualify as passed.
    pub required_checks: Vec<String>,
    /// State the uncertainty method, including explicitly "not applicable".
    pub uncertainty_method: String,
    pub claim_boundary: ClaimBoundary,
    pub claims: Vec<String>,
    pub non_claims: Vec<String>,
    pub falsification_conditions: Vec<String>,
}

impl ExperimentManifest {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.schema != SCIENCE_EVIDENCE_SCHEMA {
            return Err(ManifestError::UnsupportedSchema(self.schema.clone()));
        }
        require_nonempty("study_id", &self.study_id)?;
        require_nonempty("protocol_id", &self.protocol_id)?;
        if !is_git_sha(&self.source_commit_sha) {
            return Err(ManifestError::InvalidCommitSha);
        }
        require_nonempty("toolchain", &self.toolchain)?;
        require_nonempty("platform", &self.platform)?;
        require_nonempty("input_scope_note", &self.input_scope_note)?;
        require_nonempty("seed_policy", &self.seed_policy)?;
        require_nonempty("baseline_description", &self.baseline_description)?;
        require_nonempty("uncertainty_method", &self.uncertainty_method)?;
        require_nonempty_list(
            "coordinate_and_unit_conventions",
            &self.coordinate_and_unit_conventions,
        )?;
        require_nonempty_list("equations_and_assumptions", &self.equations_and_assumptions)?;
        require_nonempty_list("primary_metrics", &self.primary_metrics)?;
        require_nonempty_list("required_checks", &self.required_checks)?;
        ensure_unique_strings(&self.required_checks)?;
        require_nonempty_list("claims", &self.claims)?;
        require_nonempty_list("non_claims", &self.non_claims)?;
        require_nonempty_list("falsification_conditions", &self.falsification_conditions)?;

        ensure_unique_artifact_paths(&self.input_artifacts)?;
        for artifact in &self.input_artifacts {
            artifact.validate()?;
        }
        Ok(())
    }

    /// BLAKE3 digest of the serialized, validated manifest.
    pub fn digest(&self) -> Result<String, ManifestError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| ManifestError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VerificationCheck {
    pub name: String,
    pub status: VerificationStatus,
    /// Full source commit checked by the runner.
    pub source_commit_sha: String,
    /// Passed checks must bind a non-empty stdout/log artifact.
    pub log_artifact: Option<ArtifactDigest>,
    pub notes: String,
}

impl VerificationCheck {
    fn validate_against(&self, manifest: &ExperimentManifest) -> Result<(), ManifestError> {
        require_nonempty("verification check name", &self.name)?;
        require_nonempty("verification check notes", &self.notes)?;
        if !is_git_sha(&self.source_commit_sha)
            || self.source_commit_sha != manifest.source_commit_sha
        {
            return Err(ManifestError::StaleHead {
                check: self.name.clone(),
                expected: manifest.source_commit_sha.clone(),
                actual: self.source_commit_sha.clone(),
            });
        }
        if self.status == VerificationStatus::Passed {
            let log = self
                .log_artifact
                .as_ref()
                .ok_or_else(|| ManifestError::PassedCheckWithoutLog(self.name.clone()))?;
            log.validate()?;
        } else if let Some(log) = &self.log_artifact {
            log.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RunEnvironment {
    pub toolchain: String,
    pub platform: String,
    pub runner_label: String,
}

impl RunEnvironment {
    fn validate(&self) -> Result<(), ManifestError> {
        require_nonempty("run environment toolchain", &self.toolchain)?;
        require_nonempty("run environment platform", &self.platform)?;
        require_nonempty("runner_label", &self.runner_label)
    }
}

/// A run receipt is a factual record, not an aspirational plan.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RunReceipt {
    pub manifest_blake3_hex: String,
    pub source_commit_sha: String,
    pub status: VerificationStatus,
    /// The checks listed here are the complete required-check set for this receipt.
    pub checks: Vec<VerificationCheck>,
    pub output_artifacts: Vec<ArtifactDigest>,
    pub environment: RunEnvironment,
}

impl RunReceipt {
    pub fn validate_against(&self, manifest: &ExperimentManifest) -> Result<(), ManifestError> {
        manifest.validate()?;
        self.environment.validate()?;
        if !is_lower_hex_digest(&self.manifest_blake3_hex) {
            return Err(ManifestError::InvalidDigest {
                field: "run receipt manifest".to_string(),
            });
        }
        let expected_digest = manifest.digest()?;
        if self.manifest_blake3_hex != expected_digest {
            return Err(ManifestError::ManifestDigestMismatch);
        }
        if self.source_commit_sha != manifest.source_commit_sha {
            return Err(ManifestError::StaleRunHead {
                expected: manifest.source_commit_sha.clone(),
                actual: self.source_commit_sha.clone(),
            });
        }
        if self.environment.toolchain != manifest.toolchain
            || self.environment.platform != manifest.platform
        {
            return Err(ManifestError::RunEnvironmentMismatch);
        }
        if self.status == VerificationStatus::Passed && !manifest.source_tree_clean {
            return Err(ManifestError::DirtySourceTreeCannotPass);
        }
        let mut observed_checks = BTreeSet::new();
        for check in &self.checks {
            if !observed_checks.insert(check.name.as_str()) {
                return Err(ManifestError::DuplicateCheck(check.name.clone()));
            }
            check.validate_against(manifest)?;
        }
        let required_checks: BTreeSet<&str> = manifest
            .required_checks
            .iter()
            .map(String::as_str)
            .collect();
        if observed_checks != required_checks {
            return Err(ManifestError::RequiredCheckSetMismatch);
        }
        ensure_unique_artifact_paths(&self.output_artifacts)?;
        for artifact in &self.output_artifacts {
            artifact.validate()?;
        }

        if self.status == VerificationStatus::Passed {
            if self.output_artifacts.is_empty() {
                return Err(ManifestError::PassedRunWithoutOutputs);
            }
            if self
                .checks
                .iter()
                .any(|check| check.status != VerificationStatus::Passed)
            {
                return Err(ManifestError::PassedRunWithNonPassingCheck);
            }
        }
        Ok(())
    }

    /// Whether a receipt is eligible to support a passing CI / verification claim.
    /// This returns false for queued, skipped, running, failed or unverified states.
    pub fn qualifies_as_pass(&self, manifest: &ExperimentManifest) -> bool {
        self.status == VerificationStatus::Passed && self.validate_against(manifest).is_ok()
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ManifestError {
    #[error("unsupported manifest schema: {0}")]
    UnsupportedSchema(String),
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("{0} must contain at least one non-empty value")]
    EmptyList(&'static str),
    #[error("source commit must be a full 40-character hexadecimal SHA")]
    InvalidCommitSha,
    #[error("invalid BLAKE3 digest for {field}")]
    InvalidDigest { field: String },
    #[error("artifact {0} is empty")]
    EmptyArtifact(String),
    #[error("artifact bytes do not match the recorded length/digest: {0}")]
    ArtifactBytesMismatch(String),
    #[error("artifact path is duplicated: {0}")]
    DuplicateArtifactPath(String),
    #[error("required check name is duplicated: {0}")]
    DuplicateRequiredCheck(String),
    #[error("verification check name is duplicated: {0}")]
    DuplicateCheck(String),
    #[error("run receipt check set does not exactly match the manifest's required checks")]
    RequiredCheckSetMismatch,
    #[error("manifest serialization failed: {0}")]
    Serialization(String),
    #[error("run receipt manifest digest does not match its manifest")]
    ManifestDigestMismatch,
    #[error("run head {actual} does not match manifest head {expected}")]
    StaleRunHead { expected: String, actual: String },
    #[error("passed check {check} is for head {actual}, expected {expected}")]
    StaleHead {
        check: String,
        expected: String,
        actual: String,
    },
    #[error("passed check {0} has no log artifact digest")]
    PassedCheckWithoutLog(String),
    #[error("a passed run has no output artifacts")]
    PassedRunWithoutOutputs,
    #[error("a run marked passed contains a queued, running, skipped, not-run or failed check")]
    PassedRunWithNonPassingCheck,
    #[error("run environment toolchain or platform does not match the manifest")]
    RunEnvironmentMismatch,
    #[error("a run from a dirty source tree cannot qualify as passed")]
    DirtySourceTreeCannotPass,
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), ManifestError> {
    if value.trim().is_empty() {
        Err(ManifestError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn require_nonempty_list(field: &'static str, values: &[String]) -> Result<(), ManifestError> {
    if values.is_empty() || values.iter().any(|value| value.trim().is_empty()) {
        Err(ManifestError::EmptyList(field))
    } else {
        Ok(())
    }
}

fn ensure_unique_strings(values: &[String]) -> Result<(), ManifestError> {
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value.as_str()) {
            return Err(ManifestError::DuplicateRequiredCheck(value.clone()));
        }
    }
    Ok(())
}

fn ensure_unique_artifact_paths(artifacts: &[ArtifactDigest]) -> Result<(), ManifestError> {
    let mut seen = BTreeSet::new();
    for artifact in artifacts {
        if !seen.insert(artifact.path.as_str()) {
            return Err(ManifestError::DuplicateArtifactPath(artifact.path.clone()));
        }
    }
    Ok(())
}

fn is_git_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn manifest() -> ExperimentManifest {
        ExperimentManifest {
            schema: SCIENCE_EVIDENCE_SCHEMA.to_string(),
            study_id: "ctgp-causal-set-reference".to_string(),
            protocol_id: "minkowski-1p1-fixed-n-v1".to_string(),
            source_commit_sha: "0123456789abcdef0123456789abcdef01234567".to_string(),
            source_tree_clean: true,
            toolchain: "rustc 1.96.0".to_string(),
            platform: "x86_64-unknown-linux-gnu".to_string(),
            coordinate_and_unit_conventions: vec![
                "u=t+x, v=t-x".to_string(),
                "c=1; proper_time_T=1".to_string(),
            ],
            equations_and_assumptions: vec![
                "causal order: u_a<=u_b && v_a<=v_b with one strict".to_string(),
                "fixed-cardinality Poisson sprinkling in flat 1+1D Minkowski".to_string(),
            ],
            configuration: json!({"seed": 42, "events": 1500}),
            input_artifacts: vec![ArtifactDigest::from_bytes("fixture.json", b"{\"seed\":42}")],
            input_scope_note: "one frozen JSON fixture".to_string(),
            seed_policy: "one deterministic SplitMix64 seed".to_string(),
            seeds: vec![42],
            baseline_description: "analytic Minkowski ordering-fraction r=0.5".to_string(),
            primary_metrics: vec!["ordering_fraction".to_string(), "MM_dimension".to_string()],
            required_checks: vec![
                "unit-tests".to_string(),
                "analytic-oracle".to_string(),
                "format-check".to_string(),
            ],
            uncertainty_method: "multi-seed sample SD and t interval".to_string(),
            claim_boundary: ClaimBoundary::NumericalSimulation,
            claims: vec![
                "the finite-sample estimator approaches the flat 1+1D reference".to_string(),
            ],
            non_claims: vec!["does not establish causal-set quantum gravity".to_string()],
            falsification_conditions: vec![
                "ordering fraction fails to approach 0.5 under increasing N".to_string(),
            ],
        }
    }

    fn passing_receipt(manifest: &ExperimentManifest) -> RunReceipt {
        RunReceipt {
            manifest_blake3_hex: manifest.digest().unwrap(),
            source_commit_sha: manifest.source_commit_sha.clone(),
            status: VerificationStatus::Passed,
            checks: vec![
                VerificationCheck {
                    name: "unit-tests".to_string(),
                    status: VerificationStatus::Passed,
                    source_commit_sha: manifest.source_commit_sha.clone(),
                    log_artifact: Some(ArtifactDigest::from_bytes(
                        "unit-tests.log",
                        b"test result: ok",
                    )),
                    notes: "exact-head Rust test run".to_string(),
                },
                VerificationCheck {
                    name: "analytic-oracle".to_string(),
                    status: VerificationStatus::Passed,
                    source_commit_sha: manifest.source_commit_sha.clone(),
                    log_artifact: Some(ArtifactDigest::from_bytes(
                        "oracle.log",
                        b"oracle result: match",
                    )),
                    notes: "independent analytic reference matched".to_string(),
                },
                VerificationCheck {
                    name: "format-check".to_string(),
                    status: VerificationStatus::Passed,
                    source_commit_sha: manifest.source_commit_sha.clone(),
                    log_artifact: Some(ArtifactDigest::from_bytes(
                        "format.log",
                        b"format result: clean",
                    )),
                    notes: "format check completed on the exact source head".to_string(),
                },
            ],
            output_artifacts: vec![ArtifactDigest::from_bytes("report.json", b"{}")],
            environment: RunEnvironment {
                toolchain: "rustc 1.96.0".to_string(),
                platform: "x86_64-unknown-linux-gnu".to_string(),
                runner_label: "local-reference".to_string(),
            },
        }
    }

    #[test]
    fn manifest_digest_binds_configuration_and_input_bytes() {
        let first = manifest();
        let first_digest = first.digest().unwrap();
        let mut changed_config = first.clone();
        changed_config.configuration = json!({"seed": 42, "events": 1501});
        assert_ne!(first_digest, changed_config.digest().unwrap());

        let mut changed_input = first.clone();
        changed_input.input_artifacts[0] =
            ArtifactDigest::from_bytes("fixture.json", b"{\"seed\":43}");
        assert_ne!(first_digest, changed_input.digest().unwrap());
    }

    #[test]
    fn artifact_digest_recomputes_against_raw_bytes() {
        let digest = ArtifactDigest::from_bytes("input.bin", b"frozen-input");
        assert!(digest.verify_bytes(b"frozen-input").is_ok());
        assert_eq!(
            digest.verify_bytes(b"tampered-input"),
            Err(ManifestError::ArtifactBytesMismatch(
                "input.bin".to_string()
            ))
        );
    }

    #[test]
    fn duplicate_artifact_paths_and_required_check_names_are_rejected() {
        let mut duplicate_path_manifest = manifest();
        let input = duplicate_path_manifest.input_artifacts[0].clone();
        duplicate_path_manifest.input_artifacts.push(input);
        assert_eq!(
            duplicate_path_manifest.validate(),
            Err(ManifestError::DuplicateArtifactPath(
                "fixture.json".to_string()
            ))
        );

        let mut manifest = manifest();
        manifest.required_checks.push("unit-tests".to_string());
        assert_eq!(
            manifest.validate(),
            Err(ManifestError::DuplicateRequiredCheck(
                "unit-tests".to_string()
            ))
        );
    }

    #[test]
    fn receipt_must_contain_exactly_the_frozen_required_check_set() {
        let manifest = manifest();
        let mut missing = passing_receipt(&manifest);
        missing.checks.pop();
        assert_eq!(
            missing.validate_against(&manifest),
            Err(ManifestError::RequiredCheckSetMismatch)
        );

        let mut duplicate = passing_receipt(&manifest);
        duplicate.checks[1].name = duplicate.checks[0].name.clone();
        assert_eq!(
            duplicate.validate_against(&manifest),
            Err(ManifestError::DuplicateCheck("unit-tests".to_string()))
        );

        let mut unexpected = passing_receipt(&manifest);
        unexpected.checks[2].name = "unregistered-extra-check".to_string();
        assert_eq!(
            unexpected.validate_against(&manifest),
            Err(ManifestError::RequiredCheckSetMismatch)
        );
    }

    #[test]
    fn exact_head_receipt_with_logs_and_outputs_qualifies() {
        let manifest = manifest();
        let receipt = passing_receipt(&manifest);
        assert!(receipt.qualifies_as_pass(&manifest));
    }

    #[test]
    fn queued_or_skipped_checks_never_qualify_as_pass() {
        let manifest = manifest();
        let mut receipt = passing_receipt(&manifest);
        receipt.checks[0].status = VerificationStatus::Queued;
        assert_eq!(
            receipt.validate_against(&manifest),
            Err(ManifestError::PassedRunWithNonPassingCheck)
        );
        assert!(!receipt.qualifies_as_pass(&manifest));

        receipt.status = VerificationStatus::Queued;
        assert!(!receipt.qualifies_as_pass(&manifest));
    }

    #[test]
    fn stale_head_and_missing_pass_log_fail_closed() {
        let manifest = manifest();
        let mut stale = passing_receipt(&manifest);
        stale.checks[0].source_commit_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string();
        assert!(matches!(
            stale.validate_against(&manifest),
            Err(ManifestError::StaleHead { .. })
        ));

        let mut no_log = passing_receipt(&manifest);
        no_log.checks[0].log_artifact = None;
        assert_eq!(
            no_log.validate_against(&manifest),
            Err(ManifestError::PassedCheckWithoutLog(
                "unit-tests".to_string()
            ))
        );
    }

    #[test]
    fn failed_or_skipped_states_are_not_conflated_with_pass() {
        for status in [
            VerificationStatus::NotRun,
            VerificationStatus::Queued,
            VerificationStatus::Running,
            VerificationStatus::Failed,
            VerificationStatus::Skipped,
        ] {
            assert!(!status.is_passed());
        }
        assert!(VerificationStatus::Passed.is_passed());
    }

    #[test]
    fn dirty_tree_and_environment_mismatch_cannot_qualify_as_pass() {
        let mut dirty_manifest = manifest();
        dirty_manifest.source_tree_clean = false;
        let receipt = passing_receipt(&dirty_manifest);
        assert_eq!(
            receipt.validate_against(&dirty_manifest),
            Err(ManifestError::DirtySourceTreeCannotPass)
        );

        let manifest = manifest();
        let mut receipt = passing_receipt(&manifest);
        receipt.environment.toolchain = "rustc 1.97.0".to_string();
        assert_eq!(
            receipt.validate_against(&manifest),
            Err(ManifestError::RunEnvironmentMismatch)
        );
    }

    #[test]
    fn a_passed_run_requires_every_check_and_an_output_artifact() {
        let manifest = manifest();
        let mut no_checks = passing_receipt(&manifest);
        no_checks.checks.clear();
        assert_eq!(
            no_checks.validate_against(&manifest),
            Err(ManifestError::RequiredCheckSetMismatch)
        );

        let mut no_outputs = passing_receipt(&manifest);
        no_outputs.output_artifacts.clear();
        assert_eq!(
            no_outputs.validate_against(&manifest),
            Err(ManifestError::PassedRunWithoutOutputs)
        );
    }
}
