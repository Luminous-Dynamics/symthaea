//! Task-specific, provenance-bound MLPerf Tiny-aligned static comparison contracts.
//!
//! This module does not implement MLPerf LoadGen, EnergyRunner, the official reference models,
//! or an official submission. It prevents local research comparisons from silently changing
//! the published v1.4 workload, quality metric, data splits, processing contract, or hardware
//! tier. A formal MLPerf claim additionally requires the exact current rules/run pack and
//! submission review from MLCommons.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CONTRACT_SCHEMA: &str = "symthaea.mlperf-tiny-static-contract";
const RUN_SCHEMA: &str = "symthaea.mlperf-tiny-static-run";
const VERSION: &str = "0.1.0";
const SUITE_VERSION: &str = "1.4";
const CONTRACT_DOMAIN: &[u8] = b"symthaea.mlperf-tiny-static-contract.v0.1";
const RUN_DOMAIN: &[u8] = b"symthaea.mlperf-tiny-static-run.v0.1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlperfTinyContractError {
    pub field: String,
    pub message: String,
}

impl MlperfTinyContractError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for MlperfTinyContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for MlperfTinyContractError {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlperfTinyTask {
    KeywordSpotting,
    VisualWakeWords,
    ImageClassification,
    AnomalyDetection,
    StreamingWakeWord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlperfTinyDivision {
    Closed,
    Open,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlperfTinyHardwareTier {
    CortexMScalarCpu,
    CortexMDspCpu,
    CortexMMveCpu,
    IntegratedSignalProcessor,
    NpuAccelerated,
    HeterogeneousCpuNpu,
    RiscVScalarCpu,
    RiscVVectorCpu,
    OtherCpuOnly,
    OtherAccelerated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlperfTinyMeasurementScope {
    SingleInference,
    ContinuousStream,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlperfTinyQualityMetric {
    Top1Accuracy,
    RocAuc,
    StreamingFalsePositivesAndFalseNegatives,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlperfTinyQualityPolicy {
    EnforcePublishedTarget,
    ReportMeasuredQualityOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MlperfTinyQualityTarget {
    pub metric: MlperfTinyQualityMetric,
    /// Parts per million. Used for accuracy/AUC only.
    pub minimum_score_ppm: Option<u32>,
    /// Used for the continuous streaming wake-word workload only.
    pub maximum_false_positives: Option<u32>,
    pub maximum_false_negatives: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlperfTinyTaskDefinition {
    pub task: MlperfTinyTask,
    pub dataset_name: &'static str,
    pub reference_model: &'static str,
    pub measurement_scope: MlperfTinyMeasurementScope,
    pub quality_target: MlperfTinyQualityTarget,
}

/// Definitions reflected on the public MLPerf Tiny v1.4 results page.
/// The exact official rules/run-pack artifact must still be pinned per experiment.
pub fn mlperf_tiny_v14_task_definition(task: MlperfTinyTask) -> MlperfTinyTaskDefinition {
    match task {
        MlperfTinyTask::KeywordSpotting => MlperfTinyTaskDefinition {
            task,
            dataset_name: "Speech Commands",
            reference_model: "DS-CNN",
            measurement_scope: MlperfTinyMeasurementScope::SingleInference,
            quality_target: MlperfTinyQualityTarget {
                metric: MlperfTinyQualityMetric::Top1Accuracy,
                minimum_score_ppm: Some(900_000),
                maximum_false_positives: None,
                maximum_false_negatives: None,
            },
        },
        MlperfTinyTask::VisualWakeWords => MlperfTinyTaskDefinition {
            task,
            dataset_name: "COCO Visual Wake Words",
            reference_model: "MobileNet visual-wake-word 96x96",
            measurement_scope: MlperfTinyMeasurementScope::SingleInference,
            quality_target: MlperfTinyQualityTarget {
                metric: MlperfTinyQualityMetric::Top1Accuracy,
                minimum_score_ppm: Some(800_000),
                maximum_false_positives: None,
                maximum_false_negatives: None,
            },
        },
        MlperfTinyTask::ImageClassification => MlperfTinyTaskDefinition {
            task,
            dataset_name: "CIFAR-10",
            reference_model: "ResNet",
            measurement_scope: MlperfTinyMeasurementScope::SingleInference,
            quality_target: MlperfTinyQualityTarget {
                metric: MlperfTinyQualityMetric::Top1Accuracy,
                minimum_score_ppm: Some(850_000),
                maximum_false_positives: None,
                maximum_false_negatives: None,
            },
        },
        MlperfTinyTask::AnomalyDetection => MlperfTinyTaskDefinition {
            task,
            dataset_name: "ToyADMOS / Toy Car",
            reference_model: "Deep autoencoder",
            measurement_scope: MlperfTinyMeasurementScope::SingleInference,
            quality_target: MlperfTinyQualityTarget {
                metric: MlperfTinyQualityMetric::RocAuc,
                minimum_score_ppm: Some(850_000),
                maximum_false_positives: None,
                maximum_false_negatives: None,
            },
        },
        MlperfTinyTask::StreamingWakeWord => MlperfTinyTaskDefinition {
            task,
            dataset_name: "Speech Commands + MUSAN streaming workload",
            reference_model: "1D DS-CNN",
            measurement_scope: MlperfTinyMeasurementScope::ContinuousStream,
            quality_target: MlperfTinyQualityTarget {
                metric: MlperfTinyQualityMetric::StreamingFalsePositivesAndFalseNegatives,
                minimum_score_ppm: None,
                maximum_false_positives: Some(8),
                maximum_false_negatives: Some(8),
            },
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MlperfTinyStaticWorkloadContract {
    pub schema: String,
    pub version: String,
    pub suite_version: String,
    /// Digest of the exact v1.4 results/task release artifact used to define this research run.
    pub suite_release_sha256: String,
    /// Exact rule pack used. Do not assume the accessible v1.2 rules file is identical to v1.4.
    pub rules_version: String,
    pub rules_document_sha256: String,
    pub task: MlperfTinyTask,
    pub division: MlperfTinyDivision,
    pub quality_policy: MlperfTinyQualityPolicy,
    pub dataset_name: String,
    pub dataset_sha256: String,
    pub accuracy_split_sha256: String,
    pub performance_split_sha256: String,
    pub calibration_split_sha256: Option<String>,
    pub reference_model_sha256: Option<String>,
    pub reference_weights_sha256: Option<String>,
    pub reference_preprocessing_sha256: Option<String>,
    pub reference_postprocessing_sha256: Option<String>,
    pub candidate_model_sha256: String,
    pub candidate_weights_sha256: String,
    pub candidate_training_recipe_sha256: Option<String>,
    pub candidate_quantization_recipe_sha256: Option<String>,
    pub preprocessing_sha256: String,
    pub postprocessing_sha256: String,
    pub candidate_artifact_sha256: String,
    pub candidate_configuration_sha256: String,
    pub compiler_sha256: String,
    pub firmware_sha256: String,
    pub runtime: String,
    pub target_id: String,
    pub board_revision: String,
    pub cpu_core: String,
    pub accelerator_profile: String,
    pub hardware_tier: MlperfTinyHardwareTier,
    pub sram_budget_bytes: u64,
    pub protected_sram_reserve_bytes: u64,
    pub flash_budget_bytes: u64,
    pub p95_deadline_us: u64,
    pub measurement_scope: MlperfTinyMeasurementScope,
    pub contract_sha256: String,
}

pub fn compute_mlperf_tiny_contract_sha256(
    contract: &MlperfTinyStaticWorkloadContract,
) -> Result<String, MlperfTinyContractError> {
    let mut canonical = contract.clone();
    canonical.contract_sha256.clear();
    let bytes = serde_json::to_vec(&canonical).map_err(|error| {
        MlperfTinyContractError::new("contract", format!("cannot serialize: {error}"))
    })?;
    Ok(digest(CONTRACT_DOMAIN, &bytes))
}

fn require_digest(field: &str, value: &str) -> Result<(), MlperfTinyContractError> {
    if !is_sha256(value) {
        return Err(MlperfTinyContractError::new(field, "must be lowercase SHA-256 hex"));
    }
    Ok(())
}

fn require_non_empty(field: &str, value: &str) -> Result<(), MlperfTinyContractError> {
    if value.trim().is_empty() {
        return Err(MlperfTinyContractError::new(field, "must be non-empty"));
    }
    Ok(())
}

pub fn validate_mlperf_tiny_static_contract(
    contract: &MlperfTinyStaticWorkloadContract,
) -> Result<(), MlperfTinyContractError> {
    if contract.schema != CONTRACT_SCHEMA
        || contract.version != VERSION
        || contract.suite_version != SUITE_VERSION
    {
        return Err(MlperfTinyContractError::new(
            "schema_version",
            format!("expected schema {CONTRACT_SCHEMA:?}, contract {VERSION:?}, suite {SUITE_VERSION:?}"),
        ));
    }

    for (field, value) in [
        ("rules_version", contract.rules_version.as_str()),
        ("dataset_name", contract.dataset_name.as_str()),
        ("runtime", contract.runtime.as_str()),
        ("target_id", contract.target_id.as_str()),
        ("board_revision", contract.board_revision.as_str()),
        ("cpu_core", contract.cpu_core.as_str()),
        ("accelerator_profile", contract.accelerator_profile.as_str()),
    ] {
        require_non_empty(field, value)?;
    }
    for (field, value) in [
        ("suite_release_sha256", contract.suite_release_sha256.as_str()),
        ("rules_document_sha256", contract.rules_document_sha256.as_str()),
        ("dataset_sha256", contract.dataset_sha256.as_str()),
        ("accuracy_split_sha256", contract.accuracy_split_sha256.as_str()),
        ("performance_split_sha256", contract.performance_split_sha256.as_str()),
        ("candidate_model_sha256", contract.candidate_model_sha256.as_str()),
        ("candidate_weights_sha256", contract.candidate_weights_sha256.as_str()),
        ("preprocessing_sha256", contract.preprocessing_sha256.as_str()),
        ("postprocessing_sha256", contract.postprocessing_sha256.as_str()),
        ("candidate_artifact_sha256", contract.candidate_artifact_sha256.as_str()),
        ("candidate_configuration_sha256", contract.candidate_configuration_sha256.as_str()),
        ("compiler_sha256", contract.compiler_sha256.as_str()),
        ("firmware_sha256", contract.firmware_sha256.as_str()),
        ("contract_sha256", contract.contract_sha256.as_str()),
    ] {
        require_digest(field, value)?;
    }
    if contract.accuracy_split_sha256 == contract.performance_split_sha256 {
        return Err(MlperfTinyContractError::new(
            "data_splits",
            "accuracy and performance splits must be distinct",
        ));
    }
    if contract.sram_budget_bytes == 0
        || contract.protected_sram_reserve_bytes >= contract.sram_budget_bytes
        || contract.flash_budget_bytes == 0
        || contract.p95_deadline_us == 0
    {
        return Err(MlperfTinyContractError::new(
            "target_budgets",
            "SRAM/flash budgets and p95 deadline must be positive; protected reserve must be less than SRAM",
        ));
    }

    if let Some(calibration) = &contract.calibration_split_sha256 {
        require_digest("calibration_split_sha256", calibration)?;
        if calibration == &contract.accuracy_split_sha256
            || calibration == &contract.performance_split_sha256
        {
            return Err(MlperfTinyContractError::new(
                "calibration_split_sha256",
                "calibration split must be distinct from accuracy/performance splits",
            ));
        }
    }

    let task_definition = mlperf_tiny_v14_task_definition(contract.task);
    if contract.dataset_name != task_definition.dataset_name
        || contract.measurement_scope != task_definition.measurement_scope
    {
        return Err(MlperfTinyContractError::new(
            "task_definition",
            "dataset name or measurement scope differs from the public v1.4 task definition",
        ));
    }

    match contract.division {
        MlperfTinyDivision::Closed => {
            if contract.quality_policy != MlperfTinyQualityPolicy::EnforcePublishedTarget {
                return Err(MlperfTinyContractError::new(
                    "quality_policy",
                    "Closed division must enforce the task's published quality target",
                ));
            }
            for (field, value) in [
                ("reference_model_sha256", contract.reference_model_sha256.as_deref()),
                ("reference_weights_sha256", contract.reference_weights_sha256.as_deref()),
                ("reference_preprocessing_sha256", contract.reference_preprocessing_sha256.as_deref()),
                ("reference_postprocessing_sha256", contract.reference_postprocessing_sha256.as_deref()),
            ] {
                let value = value.ok_or_else(|| {
                    MlperfTinyContractError::new(field, "required for Closed division")
                })?;
                require_digest(field, value)?;
            }
            if contract.candidate_model_sha256 != contract.reference_model_sha256.as_deref().unwrap_or("")
                || contract.candidate_weights_sha256 != contract.reference_weights_sha256.as_deref().unwrap_or("")
            {
                return Err(MlperfTinyContractError::new(
                    "reference_model",
                    "Closed division must preserve the exact reference model and weights",
                ));
            }
            if contract.preprocessing_sha256 != contract.reference_preprocessing_sha256.as_deref().unwrap_or("")
                || contract.postprocessing_sha256 != contract.reference_postprocessing_sha256.as_deref().unwrap_or("")
            {
                return Err(MlperfTinyContractError::new(
                    "processing_contract",
                    "Closed division must preserve reference preprocessing and postprocessing",
                ));
            }
            if contract.candidate_training_recipe_sha256.is_some() {
                return Err(MlperfTinyContractError::new(
                    "candidate_training_recipe_sha256",
                    "Closed division does not allow retraining",
                ));
            }
        }
        MlperfTinyDivision::Open => {
            if contract.reference_model_sha256.is_some() || contract.reference_weights_sha256.is_some() {
                return Err(MlperfTinyContractError::new(
                    "reference_model",
                    "Open division must describe its own candidate model/weights rather than pinning Closed reference weights",
                ));
            }
            if let Some(recipe) = &contract.candidate_training_recipe_sha256 {
                require_digest("candidate_training_recipe_sha256", recipe)?;
            } else {
                return Err(MlperfTinyContractError::new(
                    "candidate_training_recipe_sha256",
                    "Open division must bind the model training/initialization recipe, including an explicit no-training recipe",
                ));
            }
            if contract.reference_preprocessing_sha256.is_some()
                || contract.reference_postprocessing_sha256.is_some()
            {
                return Err(MlperfTinyContractError::new(
                    "reference_processing",
                    "Open division must declare candidate processing directly; reference processing fields are reserved for Closed division",
                ));
            }
        }
    }

    if contract.candidate_quantization_recipe_sha256.is_some() {
        require_digest(
            "candidate_quantization_recipe_sha256",
            contract.candidate_quantization_recipe_sha256.as_deref().unwrap(),
        )?;
        if contract.calibration_split_sha256.is_none() {
            return Err(MlperfTinyContractError::new(
                "calibration_split_sha256",
                "quantized candidate must bind the calibration split",
            ));
        }
    }

    if compute_mlperf_tiny_contract_sha256(contract)? != contract.contract_sha256 {
        return Err(MlperfTinyContractError::new(
            "contract_sha256",
            "does not match canonical contract",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum MlperfTinyQualityResult {
    Top1Accuracy {
        score_ppm: u32,
        sample_count: usize,
    },
    RocAuc {
        score_ppm: u32,
        positive_count: usize,
        negative_count: usize,
    },
    StreamingWakeWord {
        false_positives: u32,
        false_negatives: u32,
        listening_duration_seconds: u64,
        stream_artifact_sha256: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MlperfTinyEnergyObservation {
    pub scope: MlperfTinyMeasurementScope,
    /// Mean energy for the declared measurement scope.
    pub energy_uj: f64,
    /// Expanded uncertainty for the same measurand and scope.
    pub expanded_uncertainty_uj: f64,
    pub replicate_count: usize,
    pub uncertainty_receipt_sha256: String,
    pub measurement_receipt_sha256: String,
    pub raw_power_trace_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MlperfTinyStaticRunReceipt {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub contract_sha256: String,
    pub candidate_artifact_sha256: String,
    pub candidate_configuration_sha256: String,
    pub target_id: String,
    pub board_revision: String,
    pub cpu_core: String,
    pub accelerator_profile: String,
    pub hardware_tier: MlperfTinyHardwareTier,
    pub device_session_sha256: String,
    pub measurement_protocol_sha256: String,
    pub quality_result: MlperfTinyQualityResult,
    pub latency_samples_us: Vec<u64>,
    pub p50_latency_us: u64,
    pub p95_latency_us: u64,
    pub p99_latency_us: u64,
    pub peak_sram_bytes: u64,
    pub flash_bytes: u64,
    pub energy: Option<MlperfTinyEnergyObservation>,
    pub evaluator_output_sha256: String,
    pub runner_receipt_sha256: String,
    pub run_receipt_sha256: String,
}

pub fn nearest_rank_percentile_u64(
    samples: &[u64],
    numerator: usize,
    denominator: usize,
) -> Result<u64, MlperfTinyContractError> {
    if samples.is_empty() || denominator == 0 || numerator == 0 || numerator > denominator {
        return Err(MlperfTinyContractError::new("percentile", "invalid samples or percentile fraction"));
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let scaled = numerator.checked_mul(sorted.len()).ok_or_else(|| {
        MlperfTinyContractError::new("percentile", "rank multiplication overflow")
    })?;
    let rounded = scaled.checked_add(denominator - 1).ok_or_else(|| {
        MlperfTinyContractError::new("percentile", "rank rounding overflow")
    })?;
    let rank = rounded / denominator;
    Ok(sorted[rank.saturating_sub(1).min(sorted.len() - 1)])
}

pub fn compute_mlperf_tiny_static_run_receipt_sha256(
    receipt: &MlperfTinyStaticRunReceipt,
) -> Result<String, MlperfTinyContractError> {
    let mut canonical = receipt.clone();
    canonical.run_receipt_sha256.clear();
    let bytes = serde_json::to_vec(&canonical).map_err(|error| {
        MlperfTinyContractError::new("run_receipt", format!("cannot serialize: {error}"))
    })?;
    Ok(digest(RUN_DOMAIN, &bytes))
}

fn validate_quality_result(
    contract: &MlperfTinyStaticWorkloadContract,
    result: &MlperfTinyQualityResult,
) -> Result<(), MlperfTinyContractError> {
    let target = mlperf_tiny_v14_task_definition(contract.task).quality_target;
    let score_ok = |score: u32, sample_count: usize| {
        score <= 1_000_000 && sample_count > 0
            && (contract.quality_policy == MlperfTinyQualityPolicy::ReportMeasuredQualityOnly
                || Some(score) >= target.minimum_score_ppm)
    };
    match (target.metric, result) {
        (
            MlperfTinyQualityMetric::Top1Accuracy,
            MlperfTinyQualityResult::Top1Accuracy { score_ppm, sample_count },
        ) => {
            if !score_ok(*score_ppm, *sample_count) {
                return Err(MlperfTinyContractError::new("quality_result", "top-1 score is invalid or below the enforced public target"));
            }
        }
        (
            MlperfTinyQualityMetric::RocAuc,
            MlperfTinyQualityResult::RocAuc { score_ppm, positive_count, negative_count },
        ) => {
            if !score_ok(*score_ppm, positive_count.saturating_add(*negative_count))
                || *positive_count == 0 || *negative_count == 0
            {
                return Err(MlperfTinyContractError::new("quality_result", "AUC result is invalid or below the enforced public target"));
            }
        }
        (
            MlperfTinyQualityMetric::StreamingFalsePositivesAndFalseNegatives,
            MlperfTinyQualityResult::StreamingWakeWord {
                false_positives,
                false_negatives,
                listening_duration_seconds,
                stream_artifact_sha256,
            },
        ) => {
            if *listening_duration_seconds == 0 || !is_sha256(stream_artifact_sha256) {
                return Err(MlperfTinyContractError::new("quality_result.streaming", "duration or stream artifact digest is invalid"));
            }
            if contract.quality_policy == MlperfTinyQualityPolicy::EnforcePublishedTarget
                && (*false_positives > target.maximum_false_positives.unwrap_or(u32::MAX)
                    || *false_negatives > target.maximum_false_negatives.unwrap_or(u32::MAX))
            {
                return Err(MlperfTinyContractError::new("quality_result.streaming", "false-positive or false-negative count exceeds public target"));
            }
        }
        _ => {
            return Err(MlperfTinyContractError::new(
                "quality_result.metric",
                "result metric does not match the frozen workload task",
            ));
        }
    }
    Ok(())
}

pub fn validate_mlperf_tiny_static_run(
    contract: &MlperfTinyStaticWorkloadContract,
    receipt: &MlperfTinyStaticRunReceipt,
) -> Result<(), MlperfTinyContractError> {
    validate_mlperf_tiny_static_contract(contract)?;
    if receipt.schema != RUN_SCHEMA || receipt.version != VERSION {
        return Err(MlperfTinyContractError::new("run.schema_version", "unsupported run receipt schema/version"));
    }
    require_non_empty("run_id", &receipt.run_id)?;
    for (field, value) in [
        ("contract_sha256", receipt.contract_sha256.as_str()),
        ("candidate_artifact_sha256", receipt.candidate_artifact_sha256.as_str()),
        ("candidate_configuration_sha256", receipt.candidate_configuration_sha256.as_str()),
        ("evaluator_output_sha256", receipt.evaluator_output_sha256.as_str()),
        ("runner_receipt_sha256", receipt.runner_receipt_sha256.as_str()),
        ("run_receipt_sha256", receipt.run_receipt_sha256.as_str()),
    ] {
        require_digest(field, value)?;
    }
    for (field, value) in [
        ("device_session_sha256", receipt.device_session_sha256.as_str()),
        ("measurement_protocol_sha256", receipt.measurement_protocol_sha256.as_str()),
    ] {
        require_digest(field, value)?;
    }
    if receipt.contract_sha256 != contract.contract_sha256
        || receipt.candidate_artifact_sha256 != contract.candidate_artifact_sha256
        || receipt.candidate_configuration_sha256 != contract.candidate_configuration_sha256
        || receipt.target_id != contract.target_id
        || receipt.board_revision != contract.board_revision
        || receipt.cpu_core != contract.cpu_core
        || receipt.accelerator_profile != contract.accelerator_profile
        || receipt.hardware_tier != contract.hardware_tier
    {
        return Err(MlperfTinyContractError::new("run.identity", "run receipt is not bound to the exact contract, artifact, configuration or target"));
    }
    if receipt.latency_samples_us.is_empty()
        || receipt.latency_samples_us.iter().any(|value| *value == 0)
        || receipt.p50_latency_us != nearest_rank_percentile_u64(&receipt.latency_samples_us, 50, 100)?
        || receipt.p95_latency_us != nearest_rank_percentile_u64(&receipt.latency_samples_us, 95, 100)?
        || receipt.p99_latency_us != nearest_rank_percentile_u64(&receipt.latency_samples_us, 99, 100)?
    {
        return Err(MlperfTinyContractError::new("run.latency", "raw latency samples are missing/invalid or percentile summaries do not reproduce by nearest rank"));
    }
    let available_sram = contract.sram_budget_bytes
        .checked_sub(contract.protected_sram_reserve_bytes)
        .ok_or_else(|| MlperfTinyContractError::new("target_budgets", "protected SRAM reserve exceeds target budget"))?;
    if receipt.peak_sram_bytes == 0
        || receipt.flash_bytes == 0
        || receipt.peak_sram_bytes > available_sram
        || receipt.flash_bytes > contract.flash_budget_bytes
    {
        return Err(MlperfTinyContractError::new(
            "run.memory",
            "measured peak SRAM/flash is invalid or exceeds the target budget after protected reserve",
        ));
    }
    if receipt.p95_latency_us > contract.p95_deadline_us {
        return Err(MlperfTinyContractError::new(
            "run.p95_latency_us",
            "p95 latency exceeds the frozen target deadline",
        ));
    }
    validate_quality_result(contract, &receipt.quality_result)?;

    if let Some(energy) = &receipt.energy {
        if energy.scope != contract.measurement_scope
            || !energy.energy_uj.is_finite()
            || energy.energy_uj <= 0.0
            || !energy.expanded_uncertainty_uj.is_finite()
            || energy.expanded_uncertainty_uj < 0.0
            || energy.replicate_count < 2
        {
            return Err(MlperfTinyContractError::new(
                "run.energy",
                "energy scope/mean/uncertainty is invalid or fewer than two energy replicates were supplied",
            ));
        }
        require_digest("run.energy.uncertainty_receipt_sha256", &energy.uncertainty_receipt_sha256)?;
        require_digest("run.energy.measurement_receipt_sha256", &energy.measurement_receipt_sha256)?;
        require_digest("run.energy.raw_power_trace_sha256", &energy.raw_power_trace_sha256)?;
    }
    if compute_mlperf_tiny_static_run_receipt_sha256(receipt)? != receipt.run_receipt_sha256 {
        return Err(MlperfTinyContractError::new("run_receipt_sha256", "does not match canonical run receipt"));
    }
    Ok(())
}

pub fn validate_mlperf_tiny_pairwise_comparability(
    left: &MlperfTinyStaticWorkloadContract,
    right: &MlperfTinyStaticWorkloadContract,
) -> Result<(), MlperfTinyContractError> {
    validate_mlperf_tiny_static_contract(left)?;
    validate_mlperf_tiny_static_contract(right)?;
    if left.contract_sha256 == right.contract_sha256 {
        return Err(MlperfTinyContractError::new("comparison", "candidate contracts must be distinct"));
    }
    if left.suite_version != right.suite_version
        || left.suite_release_sha256 != right.suite_release_sha256
        || left.rules_version != right.rules_version
        || left.rules_document_sha256 != right.rules_document_sha256
        || left.task != right.task
        || left.division != right.division
        || left.quality_policy != right.quality_policy
        || left.dataset_sha256 != right.dataset_sha256
        || left.accuracy_split_sha256 != right.accuracy_split_sha256
        || left.performance_split_sha256 != right.performance_split_sha256
        || left.calibration_split_sha256 != right.calibration_split_sha256
        || left.measurement_scope != right.measurement_scope
        || left.target_id != right.target_id
        || left.board_revision != right.board_revision
        || left.cpu_core != right.cpu_core
        || left.accelerator_profile != right.accelerator_profile
        || left.hardware_tier != right.hardware_tier
    {
        return Err(MlperfTinyContractError::new(
            "comparison",
            "task, rules, division, dataset/splits, measurement scope and target hardware tier must match",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn closed_contract(task: MlperfTinyTask) -> MlperfTinyStaticWorkloadContract {
        let definition = mlperf_tiny_v14_task_definition(task);
        let mut contract = MlperfTinyStaticWorkloadContract {
            schema: CONTRACT_SCHEMA.into(),
            version: VERSION.into(),
            suite_version: SUITE_VERSION.into(),
            suite_release_sha256: sha('0'),
            rules_version: "pinned-rules-fixture-v1".into(),
            rules_document_sha256: sha('a'),
            task,
            division: MlperfTinyDivision::Closed,
            quality_policy: MlperfTinyQualityPolicy::EnforcePublishedTarget,
            dataset_name: definition.dataset_name.into(),
            dataset_sha256: sha('b'),
            accuracy_split_sha256: sha('c'),
            performance_split_sha256: sha('d'),
            calibration_split_sha256: Some(sha('e')),
            reference_model_sha256: Some(sha('f')),
            reference_weights_sha256: Some(sha('1')),
            reference_preprocessing_sha256: Some(sha('2')),
            reference_postprocessing_sha256: Some(sha('3')),
            candidate_model_sha256: sha('f'),
            candidate_weights_sha256: sha('1'),
            candidate_training_recipe_sha256: None,
            candidate_quantization_recipe_sha256: Some(sha('5')),
            preprocessing_sha256: sha('2'),
            postprocessing_sha256: sha('3'),
            candidate_artifact_sha256: sha('6'),
            candidate_configuration_sha256: sha('7'),
            compiler_sha256: sha('8'),
            firmware_sha256: sha('9'),
            runtime: "tflm-cmsis-nn-fixture".into(),
            target_id: "cortex-m55-fixture".into(),
            board_revision: "board-rev-a".into(),
            cpu_core: "cortex-m55".into(),
            accelerator_profile: "mve-cpu-only-npu-disabled".into(),
            hardware_tier: MlperfTinyHardwareTier::CortexMMveCpu,
            sram_budget_bytes: 512 * 1024,
            protected_sram_reserve_bytes: 64 * 1024,
            flash_budget_bytes: 2 * 1024 * 1024,
            p95_deadline_us: 100_000,
            measurement_scope: definition.measurement_scope,
            contract_sha256: String::new(),
        };
        contract.contract_sha256 = compute_mlperf_tiny_contract_sha256(&contract).unwrap();
        contract
    }

    fn open_contract() -> MlperfTinyStaticWorkloadContract {
        let mut contract = closed_contract(MlperfTinyTask::KeywordSpotting);
        contract.division = MlperfTinyDivision::Open;
        contract.quality_policy = MlperfTinyQualityPolicy::ReportMeasuredQualityOnly;
        contract.reference_model_sha256 = None;
        contract.reference_weights_sha256 = None;
        contract.reference_preprocessing_sha256 = None;
        contract.reference_postprocessing_sha256 = None;
        contract.candidate_training_recipe_sha256 = Some(sha('0'));
        contract.candidate_quantization_recipe_sha256 = None;
        contract.contract_sha256 = compute_mlperf_tiny_contract_sha256(&contract).unwrap();
        contract
    }

    #[test]
    fn v14_task_definitions_match_published_quality_targets() {
        let kws = mlperf_tiny_v14_task_definition(MlperfTinyTask::KeywordSpotting);
        assert_eq!(kws.quality_target.minimum_score_ppm, Some(900_000));
        let vww = mlperf_tiny_v14_task_definition(MlperfTinyTask::VisualWakeWords);
        assert_eq!(vww.quality_target.minimum_score_ppm, Some(800_000));
        let image = mlperf_tiny_v14_task_definition(MlperfTinyTask::ImageClassification);
        assert_eq!(image.quality_target.minimum_score_ppm, Some(850_000));
        let anomaly = mlperf_tiny_v14_task_definition(MlperfTinyTask::AnomalyDetection);
        assert_eq!(anomaly.quality_target.minimum_score_ppm, Some(850_000));
        let streaming = mlperf_tiny_v14_task_definition(MlperfTinyTask::StreamingWakeWord);
        assert_eq!(streaming.quality_target.maximum_false_positives, Some(8));
        assert_eq!(streaming.quality_target.maximum_false_negatives, Some(8));
        assert_eq!(streaming.measurement_scope, MlperfTinyMeasurementScope::ContinuousStream);
    }

    #[test]
    fn closed_division_preserves_reference_processing_and_target_quality() {
        for task in [
            MlperfTinyTask::KeywordSpotting,
            MlperfTinyTask::VisualWakeWords,
            MlperfTinyTask::ImageClassification,
            MlperfTinyTask::AnomalyDetection,
            MlperfTinyTask::StreamingWakeWord,
        ] {
            let contract = closed_contract(task);
            validate_mlperf_tiny_static_contract(&contract).unwrap();
        }
        let mut invalid = closed_contract(MlperfTinyTask::KeywordSpotting);
        invalid.preprocessing_sha256 = sha('a');
        invalid.contract_sha256 = compute_mlperf_tiny_contract_sha256(&invalid).unwrap();
        assert!(validate_mlperf_tiny_static_contract(&invalid).is_err());
    }

    #[test]
    fn quantized_closed_contract_requires_separate_calibration_split() {
        let mut invalid = closed_contract(MlperfTinyTask::KeywordSpotting);
        invalid.calibration_split_sha256 = None;
        invalid.contract_sha256 = compute_mlperf_tiny_contract_sha256(&invalid).unwrap();
        assert!(validate_mlperf_tiny_static_contract(&invalid).is_err());
    }

    #[test]
    fn open_division_allows_new_model_but_still_binds_dataset_and_training_recipe() {
        let contract = open_contract();
        validate_mlperf_tiny_static_contract(&contract).unwrap();
        let mut missing_recipe = contract.clone();
        missing_recipe.candidate_training_recipe_sha256 = None;
        missing_recipe.contract_sha256 = compute_mlperf_tiny_contract_sha256(&missing_recipe).unwrap();
        assert!(validate_mlperf_tiny_static_contract(&missing_recipe).is_err());
    }

    #[test]
    fn static_task_metric_cannot_be_substituted() {
        let mut contract = closed_contract(MlperfTinyTask::AnomalyDetection);
        contract.measurement_scope = MlperfTinyMeasurementScope::ContinuousStream;
        contract.contract_sha256 = compute_mlperf_tiny_contract_sha256(&contract).unwrap();
        assert!(validate_mlperf_tiny_static_contract(&contract).is_err());
    }

    #[test]
    fn accelerator_tiers_do_not_share_a_pairwise_leaderboard() {
        let left = closed_contract(MlperfTinyTask::KeywordSpotting);
        let mut right = left.clone();
        right.hardware_tier = MlperfTinyHardwareTier::NpuAccelerated;
        right.contract_sha256 = compute_mlperf_tiny_contract_sha256(&right).unwrap();
        assert!(validate_mlperf_tiny_pairwise_comparability(&left, &right).is_err());
    }

    #[test]
    fn run_receipt_recomputes_raw_latency_percentiles() {
        let contract = closed_contract(MlperfTinyTask::KeywordSpotting);
        let mut run = MlperfTinyStaticRunReceipt {
            schema: RUN_SCHEMA.into(),
            version: VERSION.into(),
            run_id: "run-01".into(),
            contract_sha256: contract.contract_sha256.clone(),
            candidate_artifact_sha256: contract.candidate_artifact_sha256.clone(),
            candidate_configuration_sha256: contract.candidate_configuration_sha256.clone(),
            target_id: contract.target_id.clone(),
            board_revision: contract.board_revision.clone(),
            cpu_core: contract.cpu_core.clone(),
            accelerator_profile: contract.accelerator_profile.clone(),
            hardware_tier: contract.hardware_tier,
            device_session_sha256: sha('e'),
            measurement_protocol_sha256: sha('f'),
            quality_result: MlperfTinyQualityResult::Top1Accuracy {
                score_ppm: 920_000,
                sample_count: 1_000,
            },
            latency_samples_us: vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1_000],
            p50_latency_us: 500,
            p95_latency_us: 1_000,
            p99_latency_us: 1_000,
            peak_sram_bytes: 64 * 1024,
            flash_bytes: 128 * 1024,
            energy: None,
            evaluator_output_sha256: sha('a'),
            runner_receipt_sha256: sha('b'),
            run_receipt_sha256: String::new(),
        };
        run.run_receipt_sha256 = compute_mlperf_tiny_static_run_receipt_sha256(&run).unwrap();
        validate_mlperf_tiny_static_run(&contract, &run).unwrap();
        run.p95_latency_us = 900;
        run.run_receipt_sha256 = compute_mlperf_tiny_static_run_receipt_sha256(&run).unwrap();
        assert!(validate_mlperf_tiny_static_run(&contract, &run).is_err());
    }

    #[test]
    fn quality_targets_are_enforced_only_when_the_pinned_policy_says_so() {
        let mut contract = open_contract();
        let mut run = MlperfTinyStaticRunReceipt {
            schema: RUN_SCHEMA.into(),
            version: VERSION.into(),
            run_id: "open-run".into(),
            contract_sha256: contract.contract_sha256.clone(),
            candidate_artifact_sha256: contract.candidate_artifact_sha256.clone(),
            candidate_configuration_sha256: contract.candidate_configuration_sha256.clone(),
            target_id: contract.target_id.clone(),
            board_revision: contract.board_revision.clone(),
            cpu_core: contract.cpu_core.clone(),
            accelerator_profile: contract.accelerator_profile.clone(),
            hardware_tier: contract.hardware_tier,
            device_session_sha256: sha('e'),
            measurement_protocol_sha256: sha('f'),
            quality_result: MlperfTinyQualityResult::Top1Accuracy {
                score_ppm: 800_000,
                sample_count: 1_000,
            },
            latency_samples_us: vec![100, 200],
            p50_latency_us: 100,
            p95_latency_us: 200,
            p99_latency_us: 200,
            peak_sram_bytes: 8_192,
            flash_bytes: 16_384,
            energy: None,
            evaluator_output_sha256: sha('c'),
            runner_receipt_sha256: sha('d'),
            run_receipt_sha256: String::new(),
        };
        contract.contract_sha256 = compute_mlperf_tiny_contract_sha256(&contract).unwrap();
        run.contract_sha256 = contract.contract_sha256.clone();
        assert!(validate_mlperf_tiny_static_contract(&contract).is_ok());
        run.run_receipt_sha256 = compute_mlperf_tiny_static_run_receipt_sha256(&run).unwrap();
        validate_mlperf_tiny_static_run(&contract, &run).unwrap();
    }

    #[test]
    fn pairwise_comparability_requires_same_workload_data_and_device_tier() {
        let left = closed_contract(MlperfTinyTask::KeywordSpotting);
        let mut right = left.clone();
        right.candidate_artifact_sha256 = sha('a');
        right.candidate_model_sha256 = sha('b');
        right.contract_sha256 = compute_mlperf_tiny_contract_sha256(&right).unwrap();
        validate_mlperf_tiny_pairwise_comparability(&left, &right).unwrap();
        right.dataset_sha256 = sha('e');
        right.contract_sha256 = compute_mlperf_tiny_contract_sha256(&right).unwrap();
        assert!(validate_mlperf_tiny_pairwise_comparability(&left, &right).is_err());
    }
}
