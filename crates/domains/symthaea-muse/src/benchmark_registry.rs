// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproducible external music-benchmark profiles and evaluation manifests.
//!
//! This module is the music-specific registry/adapter contract. It binds the
//! native benchmark protocol, source/split/scorer identity, model/API version,
//! prompt and preprocessing, inference budget, failure accounting and
//! evaluator receipt reference. It does not implement candidate/oracle
//! isolation, signing or replay protection; those authorities belong to the
//! shared EVAL-ORACLE programme (#9437 / #9621).
//!
//! A leaderboard score without a complete manifest and evaluator receipt is
//! provisional. Scores from different benchmark protocols are never silently
//! pooled or represented as direct comparisons.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MUSIC_BENCHMARK_PROFILE_SCHEMA_V1: &str = "mel-music-benchmark-profile-v1";
pub const MUSIC_BENCHMARK_RUN_SCHEMA_V1: &str = "mel-music-benchmark-run-v1";

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| error.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicBenchmarkFamilyV1 {
    TemporalGrounding,
    TemporalReasoning,
    SymbolicAudioAlignment,
    GlobalMusicUnderstanding,
    MusicGenerationPreference,
    SelectiveParticipation,
    CrossCulturalEvaluation,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicBenchmarkUseV1 {
    EvaluationOnly,
    Training,
    Redistribution,
    PublicRelease,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicBenchmarkRightsStatusV1 {
    Cleared,
    Restricted,
    Unresolved,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicBenchmarkRightsV1 {
    pub status: MusicBenchmarkRightsStatusV1,
    /// Human-readable license, permission, or access statement reference.
    pub rights_reference: String,
    /// Digest of a rights/permission record where one exists.
    pub rights_evidence_sha256: Option<String>,
    /// Uses expressly permitted by the source owner/license.
    pub permitted_uses: Vec<MusicBenchmarkUseV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicBenchmarkProfileV1 {
    pub schema_version: String,
    pub benchmark_id: String,
    pub benchmark_title: String,
    pub benchmark_family: MusicBenchmarkFamilyV1,
    /// Exact public release, commit, or captured snapshot identifier.
    pub benchmark_version: String,
    pub protocol_reference: String,
    /// Native task/track identifier; one profile represents one native track.
    pub native_track_id: String,
    pub source_manifest_sha256: String,
    pub split_manifest_sha256: String,
    pub task_protocol_sha256: String,
    pub scorer_id: String,
    pub scorer_version: String,
    pub scorer_sha256: String,
    /// Stable comparison scope. Do not compare across scopes without a new protocol.
    pub comparison_scope_id: String,
    pub intended_use: MusicBenchmarkUseV1,
    pub rights: MusicBenchmarkRightsV1,
}

#[derive(Serialize)]
struct MusicBenchmarkProfileCommitmentV1<'a> {
    schema_version: &'a str,
    benchmark_id: &'a str,
    benchmark_title: &'a str,
    benchmark_family: MusicBenchmarkFamilyV1,
    benchmark_version: &'a str,
    protocol_reference: &'a str,
    native_track_id: &'a str,
    source_manifest_sha256: &'a str,
    split_manifest_sha256: &'a str,
    task_protocol_sha256: &'a str,
    scorer_id: &'a str,
    scorer_version: &'a str,
    scorer_sha256: &'a str,
    comparison_scope_id: &'a str,
    intended_use: MusicBenchmarkUseV1,
    rights: &'a MusicBenchmarkRightsV1,
}

impl MusicBenchmarkProfileV1 {
    pub fn expected_profile_sha256(&self) -> Result<String, String> {
        sha256_json(&MusicBenchmarkProfileCommitmentV1 {
            schema_version: &self.schema_version,
            benchmark_id: &self.benchmark_id,
            benchmark_title: &self.benchmark_title,
            benchmark_family: self.benchmark_family,
            benchmark_version: &self.benchmark_version,
            protocol_reference: &self.protocol_reference,
            native_track_id: &self.native_track_id,
            source_manifest_sha256: &self.source_manifest_sha256,
            split_manifest_sha256: &self.split_manifest_sha256,
            task_protocol_sha256: &self.task_protocol_sha256,
            scorer_id: &self.scorer_id,
            scorer_version: &self.scorer_version,
            scorer_sha256: &self.scorer_sha256,
            comparison_scope_id: &self.comparison_scope_id,
            intended_use: self.intended_use,
            rights: &self.rights,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicBenchmarkRunDispositionV1 {
    OfficiallyReported,
    LocallyReproduced,
    Exploratory,
    Provisional,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicBenchmarkModelIdentityV1 {
    pub provider: String,
    pub model_id: String,
    /// Immutable model/API release identifier, not a mutable alias such as "latest".
    pub immutable_version: String,
    /// UTC snapshot date/time of the endpoint or model release.
    pub observed_at_utc: String,
    pub endpoint_id: String,
    /// Required for locally available weights; optional for hosted APIs.
    pub model_artifact_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicBenchmarkInferenceBudgetV1 {
    pub maximum_requests: u64,
    pub maximum_audio_seconds: f64,
    pub maximum_input_tokens: u64,
    pub maximum_output_tokens: u64,
    pub maximum_retries_per_request: u32,
    pub maximum_concurrency: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicBenchmarkExecutionConfigV1 {
    pub prompt_template_sha256: String,
    pub system_prompt_sha256: Option<String>,
    pub adapter_sha256: String,
    pub input_preprocessing_sha256: String,
    pub decoding_config_sha256: String,
    pub runtime_environment_sha256: String,
    pub runner_commit_sha: String,
    pub precision: String,
    pub seed: Option<u64>,
    pub budget: MusicBenchmarkInferenceBudgetV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicBenchmarkFailurePolicyV1 {
    EveryAttemptRetainedInDenominator,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicBenchmarkMetricV1 {
    /// Must match the native benchmark scorer's metric identifier.
    pub metric_id: String,
    pub scope_id: String,
    pub direction: MusicBenchmarkMetricDirectionV1,
    pub value: f64,
    pub unit: String,
    pub denominator: u64,
    pub numerator: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicBenchmarkMetricDirectionV1 {
    HigherIsBetter,
    LowerIsBetter,
    DescriptiveOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicBenchmarkResourceUsageV1 {
    pub requests: u64,
    pub audio_seconds: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub retries: u64,
    pub maximum_observed_concurrency: u32,
    pub wall_time_ms: u64,
    /// Energy is absent unless independently metered.
    pub measured_energy_millijoules: Option<u64>,
    pub energy_measurement_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicBenchmarkRunManifestV1 {
    pub schema_version: String,
    pub run_id: String,
    pub disposition: MusicBenchmarkRunDispositionV1,
    pub profile_sha256: String,
    pub model: MusicBenchmarkModelIdentityV1,
    pub execution: MusicBenchmarkExecutionConfigV1,
    pub failure_policy: MusicBenchmarkFailurePolicyV1,
    /// Each attempted evaluation item has exactly one terminal disposition.
    pub attempted_items: u64,
    pub succeeded_items: u64,
    pub failed_items: u64,
    pub timed_out_items: u64,
    pub unparseable_items: u64,
    pub abstained_items: u64,
    pub resource_usage: MusicBenchmarkResourceUsageV1,
    pub metrics: Vec<MusicBenchmarkMetricV1>,
    /// Candidate/raw outputs are content-addressed; bytes live in the evaluator artifact store.
    pub candidate_output_sha256: String,
    pub raw_output_sha256: String,
    /// Reference to the independent EVAL-ORACLE receipt; no signature is implemented here.
    pub evaluator_receipt_sha256: String,
    pub run_manifest_sha256: String,
}

#[derive(Serialize)]
struct MusicBenchmarkRunCommitmentV1<'a> {
    schema_version: &'a str,
    run_id: &'a str,
    disposition: MusicBenchmarkRunDispositionV1,
    profile_sha256: &'a str,
    model: &'a MusicBenchmarkModelIdentityV1,
    execution: &'a MusicBenchmarkExecutionConfigV1,
    failure_policy: MusicBenchmarkFailurePolicyV1,
    attempted_items: u64,
    succeeded_items: u64,
    failed_items: u64,
    timed_out_items: u64,
    unparseable_items: u64,
    abstained_items: u64,
    resource_usage: &'a MusicBenchmarkResourceUsageV1,
    metrics: &'a [MusicBenchmarkMetricV1],
    candidate_output_sha256: &'a str,
    raw_output_sha256: &'a str,
    evaluator_receipt_sha256: &'a str,
}

impl MusicBenchmarkRunManifestV1 {
    pub fn expected_run_manifest_sha256(&self) -> Result<String, String> {
        sha256_json(&MusicBenchmarkRunCommitmentV1 {
            schema_version: &self.schema_version,
            run_id: &self.run_id,
            disposition: self.disposition,
            profile_sha256: &self.profile_sha256,
            model: &self.model,
            execution: &self.execution,
            failure_policy: self.failure_policy,
            attempted_items: self.attempted_items,
            succeeded_items: self.succeeded_items,
            failed_items: self.failed_items,
            timed_out_items: self.timed_out_items,
            unparseable_items: self.unparseable_items,
            abstained_items: self.abstained_items,
            resource_usage: &self.resource_usage,
            metrics: &self.metrics,
            candidate_output_sha256: &self.candidate_output_sha256,
            raw_output_sha256: &self.raw_output_sha256,
            evaluator_receipt_sha256: &self.evaluator_receipt_sha256,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicBenchmarkRegistryIssueV1 {
    WrongProfileSchema,
    WrongRunSchema,
    EmptyField { field: String },
    InvalidDigest { field: String },
    MissingRightsPermissionEvidence,
    IntendedUseNotPermitted,
    UnresolvedRightsForIntendedUse,
    DuplicatePermittedUse,
    InvalidBudget { field: String },
    InvalidResourceUsage { field: String },
    ResourceBudgetExceeded { field: String },
    FailureItemsNotInDenominator,
    EmptyMetricSet,
    InvalidMetric { metric_index: usize, field: String },
    MetricDenominatorExceedsAttempts { metric_index: usize },
    InvalidMetricValue { metric_index: usize },
    RunCommitmentMismatch,
    ProfileCommitmentMismatch,
}

impl std::fmt::Display for MusicBenchmarkRegistryIssueV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MusicBenchmarkRegistryIssueV1 {}

fn push_required(issues: &mut Vec<MusicBenchmarkRegistryIssueV1>, field: &str, value: &str) {
    if value.trim().is_empty() {
        issues.push(MusicBenchmarkRegistryIssueV1::EmptyField {
            field: field.to_owned(),
        });
    }
}

fn push_digest(issues: &mut Vec<MusicBenchmarkRegistryIssueV1>, field: &str, value: &str) {
    if !is_lower_hex_digest(value) {
        issues.push(MusicBenchmarkRegistryIssueV1::InvalidDigest {
            field: field.to_owned(),
        });
    }
}

pub fn validate_music_benchmark_profile_v1(
    profile: &MusicBenchmarkProfileV1,
) -> Vec<MusicBenchmarkRegistryIssueV1> {
    let mut issues = Vec::new();
    if profile.schema_version != MUSIC_BENCHMARK_PROFILE_SCHEMA_V1 {
        issues.push(MusicBenchmarkRegistryIssueV1::WrongProfileSchema);
    }
    for (field, value) in [
        ("benchmark_id", profile.benchmark_id.as_str()),
        ("benchmark_title", profile.benchmark_title.as_str()),
        ("benchmark_version", profile.benchmark_version.as_str()),
        ("protocol_reference", profile.protocol_reference.as_str()),
        ("native_track_id", profile.native_track_id.as_str()),
        ("scorer_id", profile.scorer_id.as_str()),
        ("scorer_version", profile.scorer_version.as_str()),
        ("comparison_scope_id", profile.comparison_scope_id.as_str()),
        ("rights_reference", profile.rights.rights_reference.as_str()),
    ] {
        push_required(&mut issues, field, value);
    }
    for (field, digest) in [
        ("source_manifest_sha256", profile.source_manifest_sha256.as_str()),
        ("split_manifest_sha256", profile.split_manifest_sha256.as_str()),
        ("task_protocol_sha256", profile.task_protocol_sha256.as_str()),
        ("scorer_sha256", profile.scorer_sha256.as_str()),
    ] {
        push_digest(&mut issues, field, digest);
    }
    if let Some(digest) = &profile.rights.rights_evidence_sha256 {
        push_digest(&mut issues, "rights_evidence_sha256", digest);
    }
    for (index, usage) in profile.rights.permitted_uses.iter().enumerate() {
        if profile.rights.permitted_uses[..index].contains(usage) {
            issues.push(MusicBenchmarkRegistryIssueV1::DuplicatePermittedUse);
            break;
        }
    }
    if profile.rights.status == MusicBenchmarkRightsStatusV1::Cleared
        && profile.rights.rights_evidence_sha256.is_none()
    {
        issues.push(MusicBenchmarkRegistryIssueV1::MissingRightsPermissionEvidence);
    }
    if profile.rights.status == MusicBenchmarkRightsStatusV1::Unresolved
        && matches!(
            profile.intended_use,
            MusicBenchmarkUseV1::Training
                | MusicBenchmarkUseV1::Redistribution
                | MusicBenchmarkUseV1::PublicRelease
        )
    {
        issues.push(MusicBenchmarkRegistryIssueV1::UnresolvedRightsForIntendedUse);
    }
    if !profile.rights.permitted_uses.contains(&profile.intended_use) {
        issues.push(MusicBenchmarkRegistryIssueV1::IntendedUseNotPermitted);
    }
    if profile.expected_profile_sha256().is_err() {
        issues.push(MusicBenchmarkRegistryIssueV1::ProfileCommitmentMismatch);
    }
    issues
}

pub fn validate_music_benchmark_run_v1(
    profile: &MusicBenchmarkProfileV1,
    run: &MusicBenchmarkRunManifestV1,
) -> Vec<MusicBenchmarkRegistryIssueV1> {
    let mut issues = validate_music_benchmark_profile_v1(profile);
    if run.schema_version != MUSIC_BENCHMARK_RUN_SCHEMA_V1 {
        issues.push(MusicBenchmarkRegistryIssueV1::WrongRunSchema);
    }
    for (field, value) in [
        ("run_id", run.run_id.as_str()),
        ("model.provider", run.model.provider.as_str()),
        ("model.model_id", run.model.model_id.as_str()),
        ("model.immutable_version", run.model.immutable_version.as_str()),
        ("model.observed_at_utc", run.model.observed_at_utc.as_str()),
        ("model.endpoint_id", run.model.endpoint_id.as_str()),
        ("execution.precision", run.execution.precision.as_str()),
        ("execution.runner_commit_sha", run.execution.runner_commit_sha.as_str()),
    ] {
        push_required(&mut issues, field, value);
    }
    for (field, digest) in [
        ("profile_sha256", run.profile_sha256.as_str()),
        ("prompt_template_sha256", run.execution.prompt_template_sha256.as_str()),
        ("adapter_sha256", run.execution.adapter_sha256.as_str()),
        ("input_preprocessing_sha256", run.execution.input_preprocessing_sha256.as_str()),
        ("decoding_config_sha256", run.execution.decoding_config_sha256.as_str()),
        ("runtime_environment_sha256", run.execution.runtime_environment_sha256.as_str()),
        ("candidate_output_sha256", run.candidate_output_sha256.as_str()),
        ("raw_output_sha256", run.raw_output_sha256.as_str()),
        ("evaluator_receipt_sha256", run.evaluator_receipt_sha256.as_str()),
    ] {
        push_digest(&mut issues, field, digest);
    }
    if let Some(digest) = &run.execution.system_prompt_sha256 {
        push_digest(&mut issues, "system_prompt_sha256", digest);
    }
    if let Some(digest) = &run.model.model_artifact_sha256 {
        push_digest(&mut issues, "model_artifact_sha256", digest);
    }
    if run.expected_run_manifest_sha256().is_err() {
        issues.push(MusicBenchmarkRegistryIssueV1::RunCommitmentMismatch);
    }
    let expected_profile = profile.expected_profile_sha256().ok();
    if expected_profile.as_deref() != Some(run.profile_sha256.as_str()) {
        issues.push(MusicBenchmarkRegistryIssueV1::ProfileCommitmentMismatch);
    }

    let counts_sum = run.succeeded_items
        .saturating_add(run.failed_items)
        .saturating_add(run.timed_out_items)
        .saturating_add(run.unparseable_items)
        .saturating_add(run.abstained_items);
    if counts_sum != run.attempted_items
        || run.failure_policy != MusicBenchmarkFailurePolicyV1::EveryAttemptRetainedInDenominator
    {
        issues.push(MusicBenchmarkRegistryIssueV1::FailureItemsNotInDenominator);
    }

    let budget = run.execution.budget;
    if budget.maximum_requests == 0
        || budget.maximum_audio_seconds < 0.0
        || !budget.maximum_audio_seconds.is_finite()
        || budget.maximum_input_tokens == 0
        || budget.maximum_output_tokens == 0
        || budget.maximum_concurrency == 0
    {
        issues.push(MusicBenchmarkRegistryIssueV1::InvalidBudget {
            field: "execution.budget".to_owned(),
        });
    }
    let usage = &run.resource_usage;
    if !usage.audio_seconds.is_finite() || usage.audio_seconds < 0.0 {
        issues.push(MusicBenchmarkRegistryIssueV1::InvalidResourceUsage {
            field: "audio_seconds".to_owned(),
        });
    }
    if usage.requests > budget.maximum_requests
        || usage.audio_seconds > budget.maximum_audio_seconds
        || usage.input_tokens > budget.maximum_input_tokens
        || usage.output_tokens > budget.maximum_output_tokens
        || usage.retries > budget.maximum_requests
        || usage.maximum_observed_concurrency > budget.maximum_concurrency
    {
        issues.push(MusicBenchmarkRegistryIssueV1::ResourceBudgetExceeded {
            field: "resource_usage".to_owned(),
        });
    }
    if usage.measured_energy_millijoules.is_some()
        != usage.energy_measurement_sha256.is_some()
    {
        issues.push(MusicBenchmarkRegistryIssueV1::InvalidResourceUsage {
            field: "energy_measurement_sha256".to_owned(),
        });
    }
    if let Some(digest) = &usage.energy_measurement_sha256 {
        push_digest(&mut issues, "energy_measurement_sha256", digest);
    }
    if run.metrics.is_empty() {
        issues.push(MusicBenchmarkRegistryIssueV1::EmptyMetricSet);
    }
    for (index, metric) in run.metrics.iter().enumerate() {
        push_required(&mut issues, "metric_id", &metric.metric_id);
        push_required(&mut issues, "metric_scope_id", &metric.scope_id);
        push_required(&mut issues, "metric_unit", &metric.unit);
        if !metric.value.is_finite() {
            issues.push(MusicBenchmarkRegistryIssueV1::InvalidMetricValue {
                metric_index: index,
            });
        }
        if metric.denominator == 0 || metric.denominator > run.attempted_items {
            issues.push(MusicBenchmarkRegistryIssueV1::MetricDenominatorExceedsAttempts {
                metric_index: index,
            });
        }
        if let Some(numerator) = metric.numerator {
            if !numerator.is_finite() || numerator < 0.0 || numerator > metric.denominator as f64 {
                issues.push(MusicBenchmarkRegistryIssueV1::InvalidMetric {
                    metric_index: index,
                    field: "numerator".to_owned(),
                });
            }
        }
    }
    issues
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicBenchmarkComparabilityV1 {
    DirectlyComparable,
    DescriptiveOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicBenchmarkComparisonV1 {
    pub comparability: MusicBenchmarkComparabilityV1,
    pub reasons: Vec<String>,
}

pub fn compare_music_benchmark_runs_v1(
    left_profile: &MusicBenchmarkProfileV1,
    left_run: &MusicBenchmarkRunManifestV1,
    right_profile: &MusicBenchmarkProfileV1,
    right_run: &MusicBenchmarkRunManifestV1,
) -> MusicBenchmarkComparisonV1 {
    let mut reasons = Vec::new();
    if left_profile.benchmark_id != right_profile.benchmark_id {
        reasons.push("benchmark_id differs".to_owned());
    }
    if left_profile.benchmark_version != right_profile.benchmark_version {
        reasons.push("benchmark_version differs".to_owned());
    }
    if left_profile.native_track_id != right_profile.native_track_id {
        reasons.push("native_track_id differs".to_owned());
    }
    if left_profile.source_manifest_sha256 != right_profile.source_manifest_sha256 {
        reasons.push("source_manifest_sha256 differs".to_owned());
    }
    if left_profile.split_manifest_sha256 != right_profile.split_manifest_sha256 {
        reasons.push("split_manifest_sha256 differs".to_owned());
    }
    if left_profile.task_protocol_sha256 != right_profile.task_protocol_sha256 {
        reasons.push("task_protocol_sha256 differs".to_owned());
    }
    if left_profile.scorer_sha256 != right_profile.scorer_sha256 {
        reasons.push("scorer_sha256 differs".to_owned());
    }
    if left_profile.comparison_scope_id != right_profile.comparison_scope_id {
        reasons.push("comparison_scope_id differs".to_owned());
    }
    if left_run.execution.budget != right_run.execution.budget {
        reasons.push("inference budgets differ".to_owned());
    }
    if left_run.execution.prompt_template_sha256 != right_run.execution.prompt_template_sha256 {
        reasons.push("prompt templates differ".to_owned());
    }
    if left_run.execution.system_prompt_sha256 != right_run.execution.system_prompt_sha256 {
        reasons.push("system prompts differ".to_owned());
    }
    if left_run.execution.adapter_sha256 != right_run.execution.adapter_sha256 {
        reasons.push("adapters differ".to_owned());
    }
    if left_run.execution.input_preprocessing_sha256
        != right_run.execution.input_preprocessing_sha256
    {
        reasons.push("input preprocessing differs".to_owned());
    }
    if left_run.execution.decoding_config_sha256 != right_run.execution.decoding_config_sha256 {
        reasons.push("decoding configurations differ".to_owned());
    }
    if reasons.is_empty() {
        MusicBenchmarkComparisonV1 {
            comparability: MusicBenchmarkComparabilityV1::DirectlyComparable,
            reasons,
        }
    } else {
        MusicBenchmarkComparisonV1 {
            comparability: MusicBenchmarkComparabilityV1::DescriptiveOnly,
            reasons,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> MusicBenchmarkProfileV1 {
        let mut value = MusicBenchmarkProfileV1 {
            schema_version: MUSIC_BENCHMARK_PROFILE_SCHEMA_V1.to_owned(),
            benchmark_id: "mustbench".to_owned(),
            benchmark_title: "MusTBench".to_owned(),
            benchmark_family: MusicBenchmarkFamilyV1::TemporalReasoning,
            benchmark_version: "2026-release-1".to_owned(),
            protocol_reference: "https://github.com/sony/MusTBench".to_owned(),
            native_track_id: "TSG".to_owned(),
            source_manifest_sha256: "a".repeat(64),
            split_manifest_sha256: "b".repeat(64),
            task_protocol_sha256: "c".repeat(64),
            scorer_id: "mustbench-native-scorer".to_owned(),
            scorer_version: "1.0.0".to_owned(),
            scorer_sha256: "d".repeat(64),
            comparison_scope_id: "mustbench-tsg-official-test-v1".to_owned(),
            intended_use: MusicBenchmarkUseV1::EvaluationOnly,
            rights: MusicBenchmarkRightsV1 {
                status: MusicBenchmarkRightsStatusV1::Cleared,
                rights_reference: "official benchmark terms".to_owned(),
                rights_evidence_sha256: Some("e".repeat(64)),
                permitted_uses: vec![MusicBenchmarkUseV1::EvaluationOnly],
            },
        };
        let _ = value.expected_profile_sha256().unwrap();
        value
    }

    fn run(profile: &MusicBenchmarkProfileV1) -> MusicBenchmarkRunManifestV1 {
        let budget = MusicBenchmarkInferenceBudgetV1 {
            maximum_requests: 100,
            maximum_audio_seconds: 1_000.0,
            maximum_input_tokens: 50_000,
            maximum_output_tokens: 10_000,
            maximum_retries_per_request: 1,
            maximum_concurrency: 1,
        };
        let model = MusicBenchmarkModelIdentityV1 {
            provider: "local".to_owned(),
            model_id: "melothaea-baseline".to_owned(),
            immutable_version: "git:0123456789abcdef".to_owned(),
            observed_at_utc: "2026-11-04T00:00:00Z".to_owned(),
            endpoint_id: "local-cli-v1".to_owned(),
            model_artifact_sha256: Some("f".repeat(64)),
        };
        let execution = MusicBenchmarkExecutionConfigV1 {
            prompt_template_sha256: "1".repeat(64),
            system_prompt_sha256: Some("2".repeat(64)),
            adapter_sha256: "3".repeat(64),
            input_preprocessing_sha256: "4".repeat(64),
            decoding_config_sha256: "5".repeat(64),
            runtime_environment_sha256: "6".repeat(64),
            runner_commit_sha: "abcdef0123456789".to_owned(),
            precision: "f32".to_owned(),
            seed: Some(7),
            budget,
        };
        let usage = MusicBenchmarkResourceUsageV1 {
            requests: 80,
            audio_seconds: 800.0,
            input_tokens: 40_000,
            output_tokens: 8_000,
            retries: 3,
            maximum_observed_concurrency: 1,
            wall_time_ms: 10_000,
            measured_energy_millijoules: None,
            energy_measurement_sha256: None,
        };
        let mut manifest = MusicBenchmarkRunManifestV1 {
            schema_version: MUSIC_BENCHMARK_RUN_SCHEMA_V1.to_owned(),
            run_id: "run-001".to_owned(),
            disposition: MusicBenchmarkRunDispositionV1::LocallyReproduced,
            profile_sha256: profile.expected_profile_sha256().unwrap(),
            model,
            execution,
            failure_policy: MusicBenchmarkFailurePolicyV1::EveryAttemptRetainedInDenominator,
            attempted_items: 80,
            succeeded_items: 72,
            failed_items: 2,
            timed_out_items: 2,
            unparseable_items: 1,
            abstained_items: 3,
            resource_usage: usage,
            metrics: vec![
                MusicBenchmarkMetricV1 {
                    metric_id: "hit_at_3s".to_owned(),
                    scope_id: "TSG".to_owned(),
                    direction: MusicBenchmarkMetricDirectionV1::HigherIsBetter,
                    value: 0.72,
                    unit: "fraction".to_owned(),
                    denominator: 80,
                    numerator: Some(58.0),
                },
            ],
            candidate_output_sha256: "7".repeat(64),
            raw_output_sha256: "8".repeat(64),
            evaluator_receipt_sha256: "9".repeat(64),
            run_manifest_sha256: String::new(),
        };
        manifest.run_manifest_sha256 = manifest.expected_run_manifest_sha256().unwrap();
        manifest
    }

    #[test]
    fn valid_benchmark_profile_and_run_are_accepted() {
        let profile = profile();
        let run = run(&profile);
        assert!(validate_music_benchmark_profile_v1(&profile).is_empty());
        assert!(validate_music_benchmark_run_v1(&profile, &run).is_empty());
    }

    #[test]
    fn unresolved_rights_cannot_authorize_training() {
        let mut profile = profile();
        profile.intended_use = MusicBenchmarkUseV1::Training;
        profile.rights.status = MusicBenchmarkRightsStatusV1::Unresolved;
        profile.rights.permitted_uses = vec![MusicBenchmarkUseV1::Training];
        assert!(validate_music_benchmark_profile_v1(&profile)
            .contains(&MusicBenchmarkRegistryIssueV1::UnresolvedRightsForIntendedUse));
    }

    #[test]
    fn restricted_rights_must_explicitly_permit_the_declared_use() {
        let mut profile = profile();
        profile.rights.status = MusicBenchmarkRightsStatusV1::Restricted;
        profile.rights.permitted_uses.clear();
        assert!(validate_music_benchmark_profile_v1(&profile)
            .contains(&MusicBenchmarkRegistryIssueV1::IntendedUseNotPermitted));
    }

    #[test]
    fn all_failures_and_abstentions_remain_in_the_attempt_denominator() {
        let profile = profile();
        let mut run = run(&profile);
        run.failed_items = 0;
        run.run_manifest_sha256 = run.expected_run_manifest_sha256().unwrap();
        assert!(validate_music_benchmark_run_v1(&profile, &run)
            .contains(&MusicBenchmarkRegistryIssueV1::FailureItemsNotInDenominator));
    }

    #[test]
    fn resource_budget_overrun_is_rejected() {
        let profile = profile();
        let mut run = run(&profile);
        run.resource_usage.output_tokens = 10_001;
        run.run_manifest_sha256 = run.expected_run_manifest_sha256().unwrap();
        assert!(validate_music_benchmark_run_v1(&profile, &run)
            .contains(&MusicBenchmarkRegistryIssueV1::ResourceBudgetExceeded {
                field: "resource_usage".to_owned()
            }));
    }

    #[test]
    fn scorer_mismatch_makes_runs_descriptive_not_directly_comparable() {
        let left_profile = profile();
        let left_run = run(&left_profile);
        let mut right_profile = left_profile.clone();
        right_profile.scorer_sha256 = "e".repeat(64);
        let right_run = run(&right_profile);
        let comparison = compare_music_benchmark_runs_v1(
            &left_profile,
            &left_run,
            &right_profile,
            &right_run,
        );
        assert_eq!(
            comparison.comparability,
            MusicBenchmarkComparabilityV1::DescriptiveOnly
        );
        assert!(comparison.reasons.iter().any(|reason| reason.contains("scorer_sha256")));
    }

    #[test]
    fn different_budgets_are_not_directly_comparable() {
        let profile = profile();
        let left = run(&profile);
        let mut right = left.clone();
        right.execution.budget.maximum_requests += 1;
        let comparison = compare_music_benchmark_runs_v1(&profile, &left, &profile, &right);
        assert_eq!(
            comparison.comparability,
            MusicBenchmarkComparabilityV1::DescriptiveOnly
        );
        assert!(comparison.reasons.iter().any(|reason| reason.contains("budgets")));
    }

    #[test]
    fn run_digest_commits_raw_outputs_and_evaluator_receipt() {
        let profile = profile();
        let mut run = run(&profile);
        let original = run.run_manifest_sha256.clone();
        run.raw_output_sha256 = "a".repeat(64);
        assert_ne!(run.expected_run_manifest_sha256().unwrap(), original);
    }
}
