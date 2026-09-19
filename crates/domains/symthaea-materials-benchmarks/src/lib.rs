// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Leakage-resistant materials-discovery reproduction benchmarks.
//!
//! The search/generation side receives an exact search-space artifact, allowed
//! training artifacts, and a split definition. Ground-truth target values are kept
//! in a separately SHA-bound sealed evaluator artifact and are supplied only to the
//! evaluation function. This creates an enforceable data-flow boundary rather than
//! relying on a promise that the generator will not inspect answers.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

const SHA256_HEX_LEN: usize = 64;

/// Immutable artifact identity used by a benchmark.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkArtifactRef {
    /// Stable human/machine identifier.
    pub artifact_id: String,
    /// SHA-256 of exact artifact bytes.
    pub sha256: String,
}

impl BenchmarkArtifactRef {
    fn validate(&self) -> Result<(), BenchmarkError> {
        nonempty("artifact_id", &self.artifact_id)?;
        sha256(&self.sha256)
    }
}

/// Published source used to define or audit a reproduction benchmark.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkPublicationSource {
    /// Stable source identifier.
    pub source_id: String,
    /// Publication title.
    pub title: String,
    /// DOI when available.
    pub doi: String,
    /// Publication/acceptance date in ISO form.
    pub date: String,
}

/// Holdout policy used to test generalization rather than row-level interpolation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkSplitPolicy {
    /// Hold out entire composition families.
    CompositionFamily {
        /// Split-definition artifact.
        split_artifact: BenchmarkArtifactRef,
    },
    /// Hold out structural clusters/neighborhoods.
    StructureCluster {
        /// Structural similarity/clustering method.
        method_id: String,
        /// Split-definition artifact.
        split_artifact: BenchmarkArtifactRef,
    },
    /// Require both family and structural holdout constraints.
    FamilyAndStructure {
        /// Structural similarity/clustering method.
        method_id: String,
        /// Split-definition artifact.
        split_artifact: BenchmarkArtifactRef,
    },
}

impl BenchmarkSplitPolicy {
    fn validate(&self) -> Result<(), BenchmarkError> {
        match self {
            Self::CompositionFamily { split_artifact } => split_artifact.validate(),
            Self::StructureCluster {
                method_id,
                split_artifact,
            }
            | Self::FamilyAndStructure {
                method_id,
                split_artifact,
            } => {
                nonempty("split method_id", method_id)?;
                split_artifact.validate()
            }
        }
    }

    fn artifact(&self) -> &BenchmarkArtifactRef {
        match self {
            Self::CompositionFamily { split_artifact }
            | Self::StructureCluster { split_artifact, .. }
            | Self::FamilyAndStructure { split_artifact, .. } => split_artifact,
        }
    }
}

/// Kind of target scored by a blind benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkMetricKind {
    /// Numeric property regression.
    ScalarRegression,
    /// Binary candidate/hit recovery.
    BinaryRecovery,
}

/// Public metric definition; values remain sealed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkMetricSpec {
    /// Stable metric identifier.
    pub metric_id: String,
    /// Scientific property/decision identifier.
    pub property_id: String,
    /// Target kind.
    pub kind: BenchmarkMetricKind,
    /// Exact unit; binary targets use `bool`.
    pub unit: String,
    /// Exact MAT-008-style condition signature or benchmark condition signature.
    pub condition_signature: String,
}

impl BenchmarkMetricSpec {
    fn validate(&self) -> Result<(), BenchmarkError> {
        nonempty("metric_id", &self.metric_id)?;
        nonempty("property_id", &self.property_id)?;
        nonempty("metric unit", &self.unit)?;
        nonempty("condition_signature", &self.condition_signature)?;
        if self.kind == BenchmarkMetricKind::BinaryRecovery && self.unit != "bool" {
            return Err(BenchmarkError::BinaryMetricMustUseBoolUnit(
                self.metric_id.clone(),
            ));
        }
        Ok(())
    }
}

/// Public benchmark contract. The target artifact digest is public; target bytes are not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindBenchmarkManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Stable benchmark identifier.
    pub benchmark_id: String,
    /// Exact search-space/candidate artifact available to the generator.
    pub search_space_artifact: BenchmarkArtifactRef,
    /// Exact training/reference artifacts the generator is allowed to consume.
    pub allowed_training_artifacts: Vec<BenchmarkArtifactRef>,
    /// Leakage-resistant holdout definition.
    pub split_policy: BenchmarkSplitPolicy,
    /// SHA-256 of the target artifact, whose bytes are evaluator-only.
    pub sealed_targets_sha256: String,
    /// Public source publications.
    pub sources: Vec<BenchmarkPublicationSource>,
    /// Public target/metric definitions without target values.
    pub metrics: Vec<BenchmarkMetricSpec>,
}

impl BlindBenchmarkManifest {
    /// Validate the manifest and leakage boundary.
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.schema_version != 1 {
            return Err(BenchmarkError::UnsupportedSchemaVersion(self.schema_version));
        }
        nonempty("benchmark_id", &self.benchmark_id)?;
        self.search_space_artifact.validate()?;
        self.split_policy.validate()?;
        sha256(&self.sealed_targets_sha256)?;
        if self.sources.is_empty() {
            return Err(BenchmarkError::NoSources);
        }
        if self.metrics.is_empty() {
            return Err(BenchmarkError::NoMetrics);
        }

        let mut artifact_ids = HashSet::new();
        let mut training_hashes = HashSet::new();
        for artifact in &self.allowed_training_artifacts {
            artifact.validate()?;
            if !artifact_ids.insert(artifact.artifact_id.as_str()) {
                return Err(BenchmarkError::DuplicateArtifactId(
                    artifact.artifact_id.clone(),
                ));
            }
            training_hashes.insert(artifact.sha256.to_ascii_lowercase());
        }
        for forbidden in [
            &self.search_space_artifact.sha256,
            &self.split_policy.artifact().sha256,
        ] {
            if forbidden.eq_ignore_ascii_case(&self.sealed_targets_sha256) {
                return Err(BenchmarkError::TargetLeakageBoundaryViolation);
            }
        }
        if training_hashes.contains(&self.sealed_targets_sha256.to_ascii_lowercase()) {
            return Err(BenchmarkError::TargetLeakageBoundaryViolation);
        }

        let mut source_ids = HashSet::new();
        for source in &self.sources {
            nonempty("source_id", &source.source_id)?;
            nonempty("source title", &source.title)?;
            nonempty("source doi", &source.doi)?;
            nonempty("source date", &source.date)?;
            if !source_ids.insert(source.source_id.as_str()) {
                return Err(BenchmarkError::DuplicateSourceId(source.source_id.clone()));
            }
        }

        let mut metric_ids = HashSet::new();
        for metric in &self.metrics {
            metric.validate()?;
            if !metric_ids.insert(metric.metric_id.as_str()) {
                return Err(BenchmarkError::DuplicateMetricId(metric.metric_id.clone()));
            }
        }
        Ok(())
    }

    /// Deterministic SHA-256 of the validated manifest serialization.
    pub fn manifest_sha256(&self) -> Result<String, BenchmarkError> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }

    /// Produce the generator-visible view. Target bytes are structurally absent.
    pub fn search_phase_view(&self) -> Result<SearchPhaseView, BenchmarkError> {
        Ok(SearchPhaseView {
            benchmark_id: self.benchmark_id.clone(),
            manifest_sha256: self.manifest_sha256()?,
            search_space_artifact: self.search_space_artifact.clone(),
            allowed_training_artifacts: self.allowed_training_artifacts.clone(),
            split_policy: self.split_policy.clone(),
            sealed_targets_sha256: self.sealed_targets_sha256.clone(),
            metrics: self.metrics.clone(),
        })
    }
}

/// Data the generator/search phase may receive. It deliberately contains no target values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchPhaseView {
    /// Benchmark identity.
    pub benchmark_id: String,
    /// Exact public manifest digest.
    pub manifest_sha256: String,
    /// Candidate/search-space artifact.
    pub search_space_artifact: BenchmarkArtifactRef,
    /// Allowed training artifacts.
    pub allowed_training_artifacts: Vec<BenchmarkArtifactRef>,
    /// Holdout definition.
    pub split_policy: BenchmarkSplitPolicy,
    /// Digest of evaluator-only targets, for later binding.
    pub sealed_targets_sha256: String,
    /// Public metric definitions.
    pub metrics: Vec<BenchmarkMetricSpec>,
}

/// One prediction submitted for blind evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkPrediction {
    /// Evaluator target identifier; value is hidden until evaluation.
    pub target_id: String,
    /// Public metric identifier.
    pub metric_id: String,
    /// Predicted value/class.
    pub prediction: PredictionValue,
}

/// Submitted prediction value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PredictionValue {
    /// Numeric prediction with explicit unit.
    Scalar {
        /// Predicted value.
        value: f64,
        /// Unit.
        unit: String,
    },
    /// Binary candidate/hit prediction.
    Binary {
        /// Predicted class.
        positive: bool,
    },
}

/// Generator output presented to the sealed evaluator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkSubmission {
    /// Benchmark identity.
    pub benchmark_id: String,
    /// Exact public manifest digest used during generation.
    pub manifest_sha256: String,
    /// Exact generator/model/workflow artifact.
    pub generator_artifact_sha256: String,
    /// Exact search trace / candidate ledger artifact.
    pub search_trace_sha256: String,
    /// Predictions made before target disclosure.
    pub predictions: Vec<BenchmarkPrediction>,
}

impl BenchmarkSubmission {
    fn validate(&self) -> Result<(), BenchmarkError> {
        nonempty("submission benchmark_id", &self.benchmark_id)?;
        sha256(&self.manifest_sha256)?;
        sha256(&self.generator_artifact_sha256)?;
        sha256(&self.search_trace_sha256)?;
        let mut targets = HashSet::new();
        for prediction in &self.predictions {
            nonempty("target_id", &prediction.target_id)?;
            nonempty("prediction metric_id", &prediction.metric_id)?;
            if !targets.insert(prediction.target_id.as_str()) {
                return Err(BenchmarkError::DuplicatePredictionTarget(
                    prediction.target_id.clone(),
                ));
            }
            match &prediction.prediction {
                PredictionValue::Scalar { value, unit } => {
                    finite("prediction scalar", *value)?;
                    nonempty("prediction unit", unit)?;
                }
                PredictionValue::Binary { .. } => {}
            }
        }
        Ok(())
    }
}

/// Blind evaluation result. This is benchmark evidence, not MAT-001 materials authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkScorecard {
    /// Benchmark identity.
    pub benchmark_id: String,
    /// Bound public manifest digest.
    pub manifest_sha256: String,
    /// Bound sealed target artifact digest.
    pub sealed_targets_sha256: String,
    /// Fraction of sealed targets for which a prediction was submitted.
    pub coverage_fraction: f64,
    /// Scalar target count actually scored.
    pub scalar_count: u32,
    /// Scalar mean absolute error.
    pub scalar_mae: Option<f64>,
    /// Scalar root mean squared error.
    pub scalar_rmse: Option<f64>,
    /// Binary true positives.
    pub true_positive: u32,
    /// Binary false positives.
    pub false_positive: u32,
    /// Binary false negatives.
    pub false_negative: u32,
    /// Binary true negatives.
    pub true_negative: u32,
    /// Binary precision when defined.
    pub precision: Option<f64>,
    /// Binary recall when defined.
    pub recall: Option<f64>,
    /// F1 when precision and recall are defined and nonzero in sum.
    pub f1: Option<f64>,
    /// Target IDs with no submitted prediction.
    pub missing_target_ids: Vec<String>,
}

/// Evaluate a submission using evaluator-only sealed target bytes.
pub fn evaluate_blind_submission(
    manifest: &BlindBenchmarkManifest,
    submission: &BenchmarkSubmission,
    sealed_target_bytes: &[u8],
) -> Result<BenchmarkScorecard, BenchmarkError> {
    manifest.validate()?;
    submission.validate()?;
    let manifest_sha = manifest.manifest_sha256()?;
    if submission.benchmark_id != manifest.benchmark_id {
        return Err(BenchmarkError::BenchmarkIdentityMismatch);
    }
    if submission.manifest_sha256 != manifest_sha {
        return Err(BenchmarkError::ManifestDigestMismatch);
    }
    if sha256_hex(sealed_target_bytes) != manifest.sealed_targets_sha256 {
        return Err(BenchmarkError::SealedTargetDigestMismatch);
    }

    let target_file: SealedTargetFile = serde_json::from_slice(sealed_target_bytes)
        .map_err(|error| BenchmarkError::SealedTargetSchema(error.to_string()))?;
    if target_file.schema_version != 1 || target_file.benchmark_id != manifest.benchmark_id {
        return Err(BenchmarkError::SealedTargetSchema(
            "target schema version or benchmark ID mismatch".to_string(),
        ));
    }

    let metric_map: HashMap<&str, &BenchmarkMetricSpec> =
        manifest.metrics.iter().map(|m| (m.metric_id.as_str(), m)).collect();
    let predictions: HashMap<&str, &BenchmarkPrediction> = submission
        .predictions
        .iter()
        .map(|p| (p.target_id.as_str(), p))
        .collect();

    let mut seen_targets = HashSet::new();
    let mut missing = Vec::new();
    let mut scalar_count = 0_u32;
    let mut abs_error_sum = 0.0;
    let mut squared_error_sum = 0.0;
    let mut tp = 0_u32;
    let mut fp = 0_u32;
    let mut fn_ = 0_u32;
    let mut tn = 0_u32;

    for target in &target_file.targets {
        nonempty("sealed target_id", &target.target_id)?;
        nonempty("sealed metric_id", &target.metric_id)?;
        if !seen_targets.insert(target.target_id.as_str()) {
            return Err(BenchmarkError::DuplicateSealedTarget(target.target_id.clone()));
        }
        let metric = metric_map
            .get(target.metric_id.as_str())
            .ok_or_else(|| BenchmarkError::UnknownMetric(target.metric_id.clone()))?;
        validate_expected_against_metric(&target.expected, metric)?;

        let Some(prediction) = predictions.get(target.target_id.as_str()) else {
            missing.push(target.target_id.clone());
            continue;
        };
        if prediction.metric_id != target.metric_id {
            return Err(BenchmarkError::PredictionMetricMismatch(
                target.target_id.clone(),
            ));
        }
        match (&target.expected, &prediction.prediction) {
            (
                SealedExpected::Scalar { value, unit },
                PredictionValue::Scalar {
                    value: predicted,
                    unit: predicted_unit,
                },
            ) => {
                if unit != predicted_unit {
                    return Err(BenchmarkError::PredictionUnitMismatch {
                        target_id: target.target_id.clone(),
                        expected: unit.clone(),
                        actual: predicted_unit.clone(),
                    });
                }
                finite("sealed scalar target", *value)?;
                let error = predicted - value;
                scalar_count += 1;
                abs_error_sum += error.abs();
                squared_error_sum += error * error;
            }
            (
                SealedExpected::Binary { positive: expected },
                PredictionValue::Binary { positive: predicted },
            ) => match (*expected, *predicted) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_ += 1,
                (false, false) => tn += 1,
            },
            _ => {
                return Err(BenchmarkError::PredictionTypeMismatch(
                    target.target_id.clone(),
                ));
            }
        }
    }

    for prediction in &submission.predictions {
        if !seen_targets.contains(prediction.target_id.as_str()) {
            return Err(BenchmarkError::UnknownPredictionTarget(
                prediction.target_id.clone(),
            ));
        }
    }

    let total_targets = target_file.targets.len();
    if total_targets == 0 {
        return Err(BenchmarkError::NoSealedTargets);
    }
    let scored = total_targets - missing.len();
    let coverage_fraction = scored as f64 / total_targets as f64;
    let scalar_mae = (scalar_count > 0).then(|| abs_error_sum / scalar_count as f64);
    let scalar_rmse =
        (scalar_count > 0).then(|| (squared_error_sum / scalar_count as f64).sqrt());
    let precision = ratio(tp, tp + fp);
    let recall = ratio(tp, tp + fn_);
    let f1 = match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        _ => None,
    };

    Ok(BenchmarkScorecard {
        benchmark_id: manifest.benchmark_id.clone(),
        manifest_sha256: manifest_sha,
        sealed_targets_sha256: manifest.sealed_targets_sha256.clone(),
        coverage_fraction,
        scalar_count,
        scalar_mae,
        scalar_rmse,
        true_positive: tp,
        false_positive: fp,
        false_negative: fn_,
        true_negative: tn,
        precision,
        recall,
        f1,
        missing_target_ids: missing,
    })
}

fn validate_expected_against_metric(
    expected: &SealedExpected,
    metric: &BenchmarkMetricSpec,
) -> Result<(), BenchmarkError> {
    match (expected, metric.kind) {
        (SealedExpected::Scalar { value, unit }, BenchmarkMetricKind::ScalarRegression) => {
            finite("sealed scalar", *value)?;
            if unit != &metric.unit {
                return Err(BenchmarkError::SealedTargetUnitMismatch(
                    metric.metric_id.clone(),
                ));
            }
            Ok(())
        }
        (SealedExpected::Binary { .. }, BenchmarkMetricKind::BinaryRecovery) => Ok(()),
        _ => Err(BenchmarkError::SealedTargetTypeMismatch(
            metric.metric_id.clone(),
        )),
    }
}

fn ratio(numerator: u32, denominator: u32) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SealedTargetFile {
    schema_version: u32,
    benchmark_id: String,
    targets: Vec<SealedTarget>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SealedTarget {
    target_id: String,
    metric_id: String,
    expected: SealedExpected,
}

#[derive(Debug, Deserialize)]
enum SealedExpected {
    Scalar { value: f64, unit: String },
    Binary { positive: bool },
}

/// Publicly reported claim value used to audit source consistency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PublicClaimValue {
    /// Exact numeric value.
    Exact { value: f64, unit: String },
    /// Strict upper-bound claim.
    LessThan { value: f64, unit: String },
    /// Integer count.
    Count { value: u64 },
}

/// One public literature anchor. These anchors are not the sealed benchmark targets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublicBenchmarkClaim {
    /// Stable claim identifier.
    pub claim_id: String,
    /// Source publication identifier.
    pub source_id: String,
    /// Material/search subject the claim applies to.
    pub subject_id: String,
    /// Property/quantity identifier.
    pub metric_id: String,
    /// Claimed value/relation.
    pub value: PublicClaimValue,
}

/// Detected inconsistency between public claims.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicClaimConflict {
    /// First claim ID.
    pub left_claim_id: String,
    /// Second claim ID.
    pub right_claim_id: String,
    /// Human-readable reason.
    pub reason: String,
}

/// Detect direct internal conflicts without silently choosing a preferred statement.
pub fn detect_public_claim_conflicts(
    claims: &[PublicBenchmarkClaim],
) -> Result<Vec<PublicClaimConflict>, BenchmarkError> {
    for claim in claims {
        nonempty("claim_id", &claim.claim_id)?;
        nonempty("claim source_id", &claim.source_id)?;
        nonempty("claim subject_id", &claim.subject_id)?;
        nonempty("claim metric_id", &claim.metric_id)?;
        validate_public_claim_value(&claim.value)?;
    }
    let mut conflicts = Vec::new();
    for i in 0..claims.len() {
        for j in (i + 1)..claims.len() {
            let left = &claims[i];
            let right = &claims[j];
            if left.source_id != right.source_id
                || left.subject_id != right.subject_id
                || left.metric_id != right.metric_id
            {
                continue;
            }
            if public_values_conflict(&left.value, &right.value)? {
                conflicts.push(PublicClaimConflict {
                    left_claim_id: left.claim_id.clone(),
                    right_claim_id: right.claim_id.clone(),
                    reason: "public claims from the same source cannot both hold exactly as encoded"
                        .to_string(),
                });
            }
        }
    }
    Ok(conflicts)
}

fn validate_public_claim_value(value: &PublicClaimValue) -> Result<(), BenchmarkError> {
    match value {
        PublicClaimValue::Exact { value, unit }
        | PublicClaimValue::LessThan { value, unit } => {
            finite("public claim value", *value)?;
            nonempty("public claim unit", unit)
        }
        PublicClaimValue::Count { .. } => Ok(()),
    }
}

fn public_values_conflict(
    left: &PublicClaimValue,
    right: &PublicClaimValue,
) -> Result<bool, BenchmarkError> {
    match (left, right) {
        (
            PublicClaimValue::Exact { value, unit },
            PublicClaimValue::LessThan {
                value: threshold,
                unit: threshold_unit,
            },
        )
        | (
            PublicClaimValue::LessThan {
                value: threshold,
                unit: threshold_unit,
            },
            PublicClaimValue::Exact { value, unit },
        ) => {
            if unit != threshold_unit {
                return Err(BenchmarkError::PublicClaimUnitMismatch);
            }
            Ok(*value >= *threshold)
        }
        (
            PublicClaimValue::Exact { value: a, unit: ua },
            PublicClaimValue::Exact { value: b, unit: ub },
        ) => {
            if ua != ub {
                return Err(BenchmarkError::PublicClaimUnitMismatch);
            }
            Ok(a.to_bits() != b.to_bits())
        }
        (PublicClaimValue::Count { value: a }, PublicClaimValue::Count { value: b }) => Ok(a != b),
        _ => Ok(false),
    }
}

/// Public source metadata for MAG-001 reproduction.
pub fn mag001_sources() -> Vec<BenchmarkPublicationSource> {
    vec![
        BenchmarkPublicationSource {
            source_id: "prm-2026-fe-co-zr".to_string(),
            title: "Accelerated discovery and design of Fe-Co-Zr magnets with tunable magnetic anisotropy through machine learning and parallel computing".to_string(),
            doi: "10.1103/jx5f-kzl8".to_string(),
            date: "2026-02-09".to_string(),
        },
        BenchmarkPublicationSource {
            source_id: "prm-2026-fe-co-s".to_string(),
            title: "Machine-learning-guided high-throughput discovery of rare earth-free Fe-Co-S magnets".to_string(),
            doi: "10.1103/dr17-jbzc".to_string(),
            date: "2026-08-27".to_string(),
        },
    ]
}

/// Public anchor claims used to verify source ingestion and preserve reported contradictions.
pub fn mag001_public_anchor_claims() -> Vec<PublicBenchmarkClaim> {
    vec![
        PublicBenchmarkClaim {
            claim_id: "fecs-highlighted-hull-threshold".to_string(),
            source_id: "prm-2026-fe-co-s".to_string(),
            subject_id: "Fe17Co4S4".to_string(),
            metric_id: "energy_above_hull".to_string(),
            value: PublicClaimValue::LessThan {
                value: 0.1,
                unit: "eV/atom".to_string(),
            },
        },
        PublicBenchmarkClaim {
            claim_id: "fecs-fe17co4s4-exact-hull".to_string(),
            source_id: "prm-2026-fe-co-s".to_string(),
            subject_id: "Fe17Co4S4".to_string(),
            metric_id: "energy_above_hull".to_string(),
            value: PublicClaimValue::Exact {
                value: 0.107,
                unit: "eV/atom".to_string(),
            },
        },
        PublicBenchmarkClaim {
            claim_id: "fecs-fe17co4s4-js".to_string(),
            source_id: "prm-2026-fe-co-s".to_string(),
            subject_id: "Fe17Co4S4".to_string(),
            metric_id: "saturation_polarization".to_string(),
            value: PublicClaimValue::Exact {
                value: 1.88,
                unit: "T".to_string(),
            },
        },
        PublicBenchmarkClaim {
            claim_id: "fecs-fe17co4s4-k1".to_string(),
            source_id: "prm-2026-fe-co-s".to_string(),
            subject_id: "Fe17Co4S4".to_string(),
            metric_id: "magnetocrystalline_anisotropy_k1".to_string(),
            value: PublicClaimValue::Exact {
                value: 0.71,
                unit: "MJ/m3".to_string(),
            },
        },
        PublicBenchmarkClaim {
            claim_id: "feczr-stable-count".to_string(),
            source_id: "prm-2026-fe-co-zr".to_string(),
            subject_id: "Fe-Co-Zr-search".to_string(),
            metric_id: "reported_stable_compound_count".to_string(),
            value: PublicClaimValue::Count { value: 9 },
        },
        PublicBenchmarkClaim {
            claim_id: "feczr-metastable-count".to_string(),
            source_id: "prm-2026-fe-co-zr".to_string(),
            subject_id: "Fe-Co-Zr-search".to_string(),
            metric_id: "reported_low_energy_metastable_count".to_string(),
            value: PublicClaimValue::Count { value: 81 },
        },
        PublicBenchmarkClaim {
            claim_id: "feczr-fe5co18zr6-k1".to_string(),
            source_id: "prm-2026-fe-co-zr".to_string(),
            subject_id: "Fe5Co18Zr6".to_string(),
            metric_id: "magnetocrystalline_anisotropy_k1".to_string(),
            value: PublicClaimValue::Exact {
                value: 1.1,
                unit: "MJ/m3".to_string(),
            },
        },
    ]
}

fn nonempty(field: &'static str, value: &str) -> Result<(), BenchmarkError> {
    if value.trim().is_empty() {
        Err(BenchmarkError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn sha256(value: &str) -> Result<(), BenchmarkError> {
    if value.len() != SHA256_HEX_LEN || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        Err(BenchmarkError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn finite(field: &'static str, value: f64) -> Result<(), BenchmarkError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(BenchmarkError::NonFiniteValue { field, value })
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Benchmark validation/evaluation failure.
#[derive(Debug, Error)]
pub enum BenchmarkError {
    /// Unsupported manifest schema.
    #[error("unsupported benchmark schema version: {0}")]
    UnsupportedSchemaVersion(u32),
    /// Required text field empty.
    #[error("required benchmark field is empty: {0}")]
    EmptyField(&'static str),
    /// Malformed SHA-256.
    #[error("invalid SHA-256")]
    InvalidSha256,
    /// No publication sources supplied.
    #[error("benchmark has no sources")]
    NoSources,
    /// No metric definitions supplied.
    #[error("benchmark has no metrics")]
    NoMetrics,
    /// Duplicate artifact ID.
    #[error("duplicate artifact ID: {0}")]
    DuplicateArtifactId(String),
    /// Duplicate source ID.
    #[error("duplicate source ID: {0}")]
    DuplicateSourceId(String),
    /// Duplicate metric ID.
    #[error("duplicate metric ID: {0}")]
    DuplicateMetricId(String),
    /// Binary metric unit was not bool.
    #[error("binary metric must use bool unit: {0}")]
    BinaryMetricMustUseBoolUnit(String),
    /// Sealed targets appeared in the generator-visible artifact set.
    #[error("sealed target artifact violates the search/evaluation leakage boundary")]
    TargetLeakageBoundaryViolation,
    /// Submission benchmark identity mismatch.
    #[error("submission benchmark identity mismatch")]
    BenchmarkIdentityMismatch,
    /// Submission was built against another manifest.
    #[error("submission manifest digest mismatch")]
    ManifestDigestMismatch,
    /// Sealed bytes did not match public target digest.
    #[error("sealed target digest mismatch")]
    SealedTargetDigestMismatch,
    /// Sealed target schema invalid.
    #[error("sealed target schema error: {0}")]
    SealedTargetSchema(String),
    /// No sealed targets were supplied.
    #[error("sealed target artifact contains no targets")]
    NoSealedTargets,
    /// Duplicate target in sealed file.
    #[error("duplicate sealed target: {0}")]
    DuplicateSealedTarget(String),
    /// Submission repeated target ID.
    #[error("duplicate prediction target: {0}")]
    DuplicatePredictionTarget(String),
    /// Metric referenced by sealed target does not exist.
    #[error("unknown metric: {0}")]
    UnknownMetric(String),
    /// Prediction metric disagreed with target metric.
    #[error("prediction metric mismatch for target: {0}")]
    PredictionMetricMismatch(String),
    /// Prediction type disagreed with target kind.
    #[error("prediction type mismatch for target: {0}")]
    PredictionTypeMismatch(String),
    /// Prediction referred to a non-existent sealed target.
    #[error("prediction target not present in sealed artifact: {0}")]
    UnknownPredictionTarget(String),
    /// Prediction unit mismatch.
    #[error("prediction unit mismatch for {target_id}: expected {expected}, got {actual}")]
    PredictionUnitMismatch {
        /// Target ID.
        target_id: String,
        /// Expected unit.
        expected: String,
        /// Actual unit.
        actual: String,
    },
    /// Sealed target type disagreed with public metric kind.
    #[error("sealed target type mismatch for metric: {0}")]
    SealedTargetTypeMismatch(String),
    /// Sealed target unit disagreed with public metric unit.
    #[error("sealed target unit mismatch for metric: {0}")]
    SealedTargetUnitMismatch(String),
    /// Public claim units incompatible.
    #[error("public claim unit mismatch")]
    PublicClaimUnitMismatch,
    /// Non-finite value.
    #[error("non-finite benchmark value in {field}: {value}")]
    NonFiniteValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// JSON serialization/deserialization error.
    #[error("benchmark JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C64: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D64: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn sealed_fixture() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "benchmark_id": "MAG-001-fixture",
            "targets": [
                {
                    "target_id": "heldout-001-formation",
                    "metric_id": "formation_energy",
                    "expected": { "Scalar": { "value": -0.25, "unit": "eV/atom" } }
                },
                {
                    "target_id": "heldout-001-recovery",
                    "metric_id": "promising_candidate",
                    "expected": { "Binary": { "positive": true } }
                },
                {
                    "target_id": "heldout-002-recovery",
                    "metric_id": "promising_candidate",
                    "expected": { "Binary": { "positive": false } }
                }
            ]
        }))
        .unwrap()
    }

    fn manifest() -> BlindBenchmarkManifest {
        let sealed = sealed_fixture();
        BlindBenchmarkManifest {
            schema_version: 1,
            benchmark_id: "MAG-001-fixture".to_string(),
            search_space_artifact: BenchmarkArtifactRef {
                artifact_id: "mag-search-space".to_string(),
                sha256: A64.to_string(),
            },
            allowed_training_artifacts: vec![BenchmarkArtifactRef {
                artifact_id: "training-corpus".to_string(),
                sha256: B64.to_string(),
            }],
            split_policy: BenchmarkSplitPolicy::FamilyAndStructure {
                method_id: "composition-family+soap-loco-v1".to_string(),
                split_artifact: BenchmarkArtifactRef {
                    artifact_id: "holdout-split".to_string(),
                    sha256: C64.to_string(),
                },
            },
            sealed_targets_sha256: sha256_hex(&sealed),
            sources: mag001_sources(),
            metrics: vec![
                BenchmarkMetricSpec {
                    metric_id: "formation_energy".to_string(),
                    property_id: "formation_energy".to_string(),
                    kind: BenchmarkMetricKind::ScalarRegression,
                    unit: "eV/atom".to_string(),
                    condition_signature: "electronic-ground-state|bound-method".to_string(),
                },
                BenchmarkMetricSpec {
                    metric_id: "promising_candidate".to_string(),
                    property_id: "benchmark_candidate_recovery".to_string(),
                    kind: BenchmarkMetricKind::BinaryRecovery,
                    unit: "bool".to_string(),
                    condition_signature: "MAG-001-screening-rule-v1".to_string(),
                },
            ],
        }
    }

    fn submission(manifest: &BlindBenchmarkManifest) -> BenchmarkSubmission {
        BenchmarkSubmission {
            benchmark_id: manifest.benchmark_id.clone(),
            manifest_sha256: manifest.manifest_sha256().unwrap(),
            generator_artifact_sha256: D64.to_string(),
            search_trace_sha256: C64.to_string(),
            predictions: vec![
                BenchmarkPrediction {
                    target_id: "heldout-001-formation".to_string(),
                    metric_id: "formation_energy".to_string(),
                    prediction: PredictionValue::Scalar {
                        value: -0.20,
                        unit: "eV/atom".to_string(),
                    },
                },
                BenchmarkPrediction {
                    target_id: "heldout-001-recovery".to_string(),
                    metric_id: "promising_candidate".to_string(),
                    prediction: PredictionValue::Binary { positive: true },
                },
                BenchmarkPrediction {
                    target_id: "heldout-002-recovery".to_string(),
                    metric_id: "promising_candidate".to_string(),
                    prediction: PredictionValue::Binary { positive: false },
                },
            ],
        }
    }

    #[test]
    fn search_phase_view_contains_digest_but_no_target_values() {
        let view = manifest().search_phase_view().unwrap();
        let serialized = serde_json::to_string(&view).unwrap();
        assert!(serialized.contains("sealed_targets_sha256"));
        assert!(!serialized.contains("-0.25"));
    }

    #[test]
    fn target_artifact_cannot_be_allowed_training_input() {
        let mut manifest = manifest();
        manifest.allowed_training_artifacts.push(BenchmarkArtifactRef {
            artifact_id: "leak".to_string(),
            sha256: manifest.sealed_targets_sha256.clone(),
        });
        assert!(matches!(
            manifest.validate(),
            Err(BenchmarkError::TargetLeakageBoundaryViolation)
        ));
    }

    #[test]
    fn sealed_evaluation_binds_manifest_and_scores_predictions() {
        let manifest = manifest();
        let score = evaluate_blind_submission(&manifest, &submission(&manifest), &sealed_fixture())
            .unwrap();
        assert_eq!(score.coverage_fraction, 1.0);
        assert_eq!(score.scalar_count, 1);
        assert!((score.scalar_mae.unwrap() - 0.05).abs() < 1e-12);
        assert!((score.scalar_rmse.unwrap() - 0.05).abs() < 1e-12);
        assert_eq!(score.true_positive, 1);
        assert_eq!(score.true_negative, 1);
        assert_eq!(score.false_positive, 0);
        assert_eq!(score.false_negative, 0);
        assert_eq!(score.precision, Some(1.0));
        assert_eq!(score.recall, Some(1.0));
        assert_eq!(score.f1, Some(1.0));
    }

    #[test]
    fn wrong_sealed_bytes_are_rejected_before_scoring() {
        let manifest = manifest();
        let result = evaluate_blind_submission(&manifest, &submission(&manifest), b"{} wrong");
        assert!(matches!(result, Err(BenchmarkError::SealedTargetDigestMismatch)));
    }

    #[test]
    fn fe_co_s_public_abstract_contradiction_is_preserved() {
        let conflicts = detect_public_claim_conflicts(&mag001_public_anchor_claims()).unwrap();
        assert_eq!(conflicts.len(), 1);
        let conflict = &conflicts[0];
        assert!(conflict.left_claim_id.contains("hull") || conflict.right_claim_id.contains("hull"));
    }
}
