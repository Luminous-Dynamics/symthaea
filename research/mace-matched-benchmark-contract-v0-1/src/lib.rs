// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reproducible benchmark run identity and matched-comparison validation.
//! Every manifest represents exactly one benchmark track and protocol.
//! This crate deliberately refuses cross-track score aggregation.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION_V1: &str = "mace_benchmark_run_manifest_v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKindV1 { ExternalReported, LocalReproduction, LocalExperiment }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirectionV1 { HigherIsBetter, LowerIsBetter, DiagnosticOnly }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricObservationV1 {
    pub metric_id: String,
    pub value: f64,
    pub unit: String,
    pub direction: MetricDirectionV1,
    pub definition_sha256: String,
    pub denominator: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceObservationV1 {
    pub p50_latency_ms: Option<f64>,
    pub p95_latency_ms: Option<f64>,
    pub p99_latency_ms: Option<f64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub context_tokens_peak: Option<u64>,
    pub persistent_memory_bytes: Option<u64>,
    pub index_bytes: Option<u64>,
    pub provenance_sidecar_bytes: Option<u64>,
    pub tombstone_journal_bytes: Option<u64>,
    pub cpu_microseconds: Option<u64>,
    pub gpu_microseconds: Option<u64>,
    pub energy_microwatt_hours: Option<u64>,
    pub monetary_micro_usd: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRunManifestV1 {
    pub schema_version: String,
    pub record_type: String,
    pub run_id: String,
    pub evidence_kind: EvidenceKindV1,
    pub benchmark_id: String,
    pub track_id: String,
    pub protocol_version: String,
    pub protocol_sha256: String,
    pub evaluator_sha256: String,
    pub dataset_id: String,
    pub dataset_revision_sha256: String,
    pub split_id: String,
    pub split_sha256: String,
    pub seed_set_sha256: String,
    pub scored_task_count: u64,
    pub independent_cluster_count: u64,
    pub dependence_unit: String,
    pub system_name: String,
    pub system_revision_sha256: Option<String>,
    pub adapter_sha256: Option<String>,
    pub memory_method_id: String,
    pub memory_method_family: String,
    pub reader_model_id: String,
    pub reader_model_revision_sha256: String,
    pub reader_prompt_sha256: String,
    pub judge_model_id: String,
    pub judge_model_revision_sha256: String,
    pub tool_contract_sha256: String,
    pub resource_budget_sha256: String,
    pub context_token_budget: u64,
    pub max_model_calls: u64,
    pub wall_clock_budget_ms: u64,
    pub cpu_budget_microseconds: Option<u64>,
    pub gpu_budget_microseconds: Option<u64>,
    pub energy_budget_microwatt_hours: Option<u64>,
    pub comparison_group_id: Option<String>,
    pub arm_id: Option<String>,
    pub matched_controls_sha256: Option<String>,
    pub candidate_saw_evaluator_labels: bool,
    pub evaluator_isolated: bool,
    pub evidence_uri: Option<String>,
    pub result_artifact_sha256: Option<String>,
    pub resource_receipt_sha256: Option<String>,
    pub metrics: Vec<MetricObservationV1>,
    pub resources: ResourceObservationV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchmarkContractRejectV1 {
    InvalidIdentity,
    InvalidDigest,
    InvalidCounts,
    InvalidBudget,
    InvalidMetric,
    DuplicateMetric,
    InvalidResourceObservation,
    ExternalEvidenceMissing,
    LocalEvidenceMissing,
    InvalidComparisonMembership,
    TooFewComparisonArms,
    DuplicateArm,
    MixedComparisonGroup,
    MixedEvidenceKind,
    MixedBenchmark,
    MixedTrack,
    MixedProtocol,
    MixedDatasetOrSplit,
    MixedReader,
    MixedPrompt,
    MixedJudge,
    MixedToolContract,
    MixedEvaluator,
    MixedSeedSet,
    MixedResourceBudget,
    MixedEvaluationIsolation,
    MixedMetricDefinitions,
    MixedCounts,
    MixedDependenceUnit,
    MatchedControlsDigestMismatch,
    SerializationFailure,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchedComparisonReceiptV1 {
    pub schema_version: String,
    pub record_type: String,
    pub comparison_group_id: String,
    pub benchmark_id: String,
    pub track_id: String,
    pub protocol_sha256: String,
    pub dataset_revision_sha256: String,
    pub split_sha256: String,
    pub reader_model_revision_sha256: String,
    pub reader_prompt_sha256: String,
    pub judge_model_revision_sha256: String,
    pub tool_contract_sha256: String,
    pub evaluator_sha256: String,
    pub seed_set_sha256: String,
    pub resource_budget_sha256: String,
    pub matched_controls_sha256: String,
    pub arm_ids: Vec<String>,
    pub run_ids: Vec<String>,
    pub metric_definition_set_sha256: String,
    pub receipt_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}
fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
fn nonempty(value: &str, max_len: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max_len
}
fn digest_serialized<T: Serialize>(value: &T) -> Result<String, BenchmarkContractRejectV1> {
    serde_json::to_vec(value).map(|bytes| sha256_hex(&bytes)).map_err(|_| BenchmarkContractRejectV1::SerializationFailure)
}

fn metric_definition_set_sha256(metrics: &[MetricObservationV1]) -> Result<String, BenchmarkContractRejectV1> {
    let mut definitions: Vec<(&str, &str, MetricDirectionV1, &str)> = metrics.iter()
        .map(|metric| (metric.metric_id.as_str(), metric.unit.as_str(), metric.direction, metric.definition_sha256.as_str(), metric.denominator))
        .collect();
    definitions.sort_unstable();
    digest_serialized(&definitions)
}

#[derive(Serialize)]
struct MatchedControlsProjectionV1<'a> {
    benchmark_id: &'a str,
    track_id: &'a str,
    protocol_version: &'a str,
    protocol_sha256: &'a str,
    evaluator_sha256: &'a str,
    dataset_id: &'a str,
    dataset_revision_sha256: &'a str,
    split_id: &'a str,
    split_sha256: &'a str,
    seed_set_sha256: &'a str,
    scored_task_count: u64,
    independent_cluster_count: u64,
    dependence_unit: &'a str,
    reader_model_id: &'a str,
    reader_model_revision_sha256: &'a str,
    reader_prompt_sha256: &'a str,
    judge_model_id: &'a str,
    judge_model_revision_sha256: &'a str,
    tool_contract_sha256: &'a str,
    evaluator_isolated: bool,
    candidate_saw_evaluator_labels: bool,
    resource_budget_sha256: &'a str,
    context_token_budget: u64,
    max_model_calls: u64,
    wall_clock_budget_ms: u64,
    cpu_budget_microseconds: Option<u64>,
    gpu_budget_microseconds: Option<u64>,
    energy_budget_microwatt_hours: Option<u64>,
    metric_definition_set_sha256: String,
}

/// Compute the digest of every fixed control in a memory-method comparison.
pub fn compute_matched_controls_sha256_v1(
    manifest: &BenchmarkRunManifestV1,
) -> Result<String, BenchmarkContractRejectV1> {
    let projection = MatchedControlsProjectionV1 {
        benchmark_id: &manifest.benchmark_id,
        track_id: &manifest.track_id,
        protocol_version: &manifest.protocol_version,
        protocol_sha256: &manifest.protocol_sha256,
        evaluator_sha256: &manifest.evaluator_sha256,
        dataset_id: &manifest.dataset_id,
        dataset_revision_sha256: &manifest.dataset_revision_sha256,
        split_id: &manifest.split_id,
        split_sha256: &manifest.split_sha256,
        seed_set_sha256: &manifest.seed_set_sha256,
        scored_task_count: manifest.scored_task_count,
        independent_cluster_count: manifest.independent_cluster_count,
        dependence_unit: &manifest.dependence_unit,
        reader_model_id: &manifest.reader_model_id,
        reader_model_revision_sha256: &manifest.reader_model_revision_sha256,
        reader_prompt_sha256: &manifest.reader_prompt_sha256,
        judge_model_id: &manifest.judge_model_id,
        judge_model_revision_sha256: &manifest.judge_model_revision_sha256,
        tool_contract_sha256: &manifest.tool_contract_sha256,
        evaluator_isolated: manifest.evaluator_isolated,
        candidate_saw_evaluator_labels: manifest.candidate_saw_evaluator_labels,
        resource_budget_sha256: &manifest.resource_budget_sha256,
        context_token_budget: manifest.context_token_budget,
        max_model_calls: manifest.max_model_calls,
        wall_clock_budget_ms: manifest.wall_clock_budget_ms,
        cpu_budget_microseconds: manifest.cpu_budget_microseconds,
        gpu_budget_microseconds: manifest.gpu_budget_microseconds,
        energy_budget_microwatt_hours: manifest.energy_budget_microwatt_hours,
        metric_definition_set_sha256: metric_definition_set_sha256(&manifest.metrics)?,
    };
    digest_serialized(&projection)
}

fn validate_resources(resources: &ResourceObservationV1) -> Result<(), BenchmarkContractRejectV1> {
    for value in [resources.p50_latency_ms, resources.p95_latency_ms, resources.p99_latency_ms].into_iter().flatten() {
        if !value.is_finite() || value < 0.0 { return Err(BenchmarkContractRejectV1::InvalidResourceObservation); }
    }
    if resources.p50_latency_ms.zip(resources.p95_latency_ms).is_some_and(|(a,b)| a>b)
        || resources.p95_latency_ms.zip(resources.p99_latency_ms).is_some_and(|(a,b)| a>b) {
        return Err(BenchmarkContractRejectV1::InvalidResourceObservation);
    }
    Ok(())
}

/// Validate a single external reference or locally executed benchmark run.
pub fn validate_run_manifest_v1(manifest: &BenchmarkRunManifestV1) -> Result<(), BenchmarkContractRejectV1> {
    if manifest.schema_version != SCHEMA_VERSION_V1 || manifest.record_type != "benchmark_run_manifest"
        || !nonempty(&manifest.run_id,128) || !nonempty(&manifest.benchmark_id,128)
        || !nonempty(&manifest.track_id,128) || !nonempty(&manifest.protocol_version,128)
        || !nonempty(&manifest.dataset_id,256) || !nonempty(&manifest.split_id,128)
        || !nonempty(&manifest.dependence_unit,128) || !nonempty(&manifest.system_name,256)
        || !nonempty(&manifest.memory_method_id,128) || !nonempty(&manifest.memory_method_family,128)
        || !nonempty(&manifest.reader_model_id,256) || !nonempty(&manifest.judge_model_id,256) {
        return Err(BenchmarkContractRejectV1::InvalidIdentity);
    }
    for value in [
        manifest.protocol_sha256.as_str(), manifest.evaluator_sha256.as_str(),
        manifest.dataset_revision_sha256.as_str(), manifest.split_sha256.as_str(),
        manifest.seed_set_sha256.as_str(), manifest.reader_model_revision_sha256.as_str(),
        manifest.reader_prompt_sha256.as_str(), manifest.judge_model_revision_sha256.as_str(),
        manifest.tool_contract_sha256.as_str(), manifest.resource_budget_sha256.as_str(),
    ] {
        if !valid_digest(value) { return Err(BenchmarkContractRejectV1::InvalidDigest); }
    }
    for value in [
        manifest.system_revision_sha256.as_deref(), manifest.adapter_sha256.as_deref(),
        manifest.matched_controls_sha256.as_deref(), manifest.result_artifact_sha256.as_deref(),
        manifest.resource_receipt_sha256.as_deref(),
    ].into_iter().flatten() {
        if !valid_digest(value) { return Err(BenchmarkContractRejectV1::InvalidDigest); }
    }
    if manifest.scored_task_count == 0 || manifest.independent_cluster_count == 0
        || manifest.independent_cluster_count > manifest.scored_task_count {
        return Err(BenchmarkContractRejectV1::InvalidCounts);
    }
    if manifest.context_token_budget == 0 || manifest.max_model_calls == 0 || manifest.wall_clock_budget_ms == 0 {
        return Err(BenchmarkContractRejectV1::InvalidBudget);
    }
    if manifest.candidate_saw_evaluator_labels || !manifest.evaluator_isolated {
        return Err(BenchmarkContractRejectV1::MixedEvaluationIsolation);
    }
    if manifest.metrics.is_empty() { return Err(BenchmarkContractRejectV1::InvalidMetric); }
    let mut metric_ids = BTreeSet::new();
    for metric in &manifest.metrics {
        if !nonempty(&metric.metric_id,128) || !nonempty(&metric.unit,64) || !metric.value.is_finite()
            || metric.denominator == Some(0) || !valid_digest(&metric.definition_sha256) {
            return Err(BenchmarkContractRejectV1::InvalidMetric);
        }
        if !metric_ids.insert(metric.metric_id.as_str()) { return Err(BenchmarkContractRejectV1::DuplicateMetric); }
    }
    validate_resources(&manifest.resources)?;

    match manifest.evidence_kind {
        EvidenceKindV1::ExternalReported => {
            if !manifest.evidence_uri.as_deref().is_some_and(|uri| uri.starts_with("https://") && uri.len() <= 2048) {
                return Err(BenchmarkContractRejectV1::ExternalEvidenceMissing);
            }
        }
        EvidenceKindV1::LocalReproduction | EvidenceKindV1::LocalExperiment => {
            if !manifest.system_revision_sha256.as_deref().is_some_and(valid_digest)
                || !manifest.adapter_sha256.as_deref().is_some_and(valid_digest)
                || !manifest.result_artifact_sha256.as_deref().is_some_and(valid_digest)
                || !manifest.resource_receipt_sha256.as_deref().is_some_and(valid_digest) {
                return Err(BenchmarkContractRejectV1::LocalEvidenceMissing);
            }
        }
    }
    match (manifest.comparison_group_id.as_deref(),manifest.arm_id.as_deref(),manifest.matched_controls_sha256.as_deref()) {
        (None,None,None) => {}
        (Some(group),Some(arm),Some(controls)) if nonempty(group,128)&&nonempty(arm,128)&&valid_digest(controls) => {}
        _ => return Err(BenchmarkContractRejectV1::InvalidComparisonMembership),
    }
    Ok(())
}

/// Compare memory methods only when all task, reader, prompt, judge, tool,
/// evaluator, seed and budget controls match exactly. It rejects cross-track
/// pooling and does not compute a score or statistical significance.
pub fn validate_matched_comparison_v1(manifests: &[BenchmarkRunManifestV1]) -> Result<MatchedComparisonReceiptV1,BenchmarkContractRejectV1> {
    if manifests.len()<2 { return Err(BenchmarkContractRejectV1::TooFewComparisonArms); }
    for manifest in manifests {
        validate_run_manifest_v1(manifest)?;
        if !matches!(manifest.evidence_kind,EvidenceKindV1::LocalReproduction|EvidenceKindV1::LocalExperiment) {
            return Err(BenchmarkContractRejectV1::MixedEvidenceKind);
        }
    }
    let first=&manifests[0];
    let group=first.comparison_group_id.as_deref().ok_or(BenchmarkContractRejectV1::InvalidComparisonMembership)?;
    let controls=first.matched_controls_sha256.as_deref().ok_or(BenchmarkContractRejectV1::InvalidComparisonMembership)?;
    let mut arms=BTreeSet::new();
    let mut runs=BTreeSet::new();
    let first_metric_defs=metric_definition_set_sha256(&first.metrics)?;
    for manifest in manifests {
        if manifest.comparison_group_id.as_deref()!=Some(group)||manifest.matched_controls_sha256.as_deref()!=Some(controls) {
            return Err(BenchmarkContractRejectV1::MixedComparisonGroup);
        }
        if !arms.insert(manifest.arm_id.as_deref().unwrap_or_default())||!runs.insert(manifest.run_id.as_str()) {
            return Err(BenchmarkContractRejectV1::DuplicateArm);
        }
        if manifest.benchmark_id!=first.benchmark_id { return Err(BenchmarkContractRejectV1::MixedBenchmark); }
        if manifest.track_id!=first.track_id { return Err(BenchmarkContractRejectV1::MixedTrack); }
        if manifest.protocol_sha256!=first.protocol_sha256||manifest.protocol_version!=first.protocol_version {
            return Err(BenchmarkContractRejectV1::MixedProtocol);
        }
        if manifest.dataset_id!=first.dataset_id||manifest.dataset_revision_sha256!=first.dataset_revision_sha256
            ||manifest.split_id!=first.split_id||manifest.split_sha256!=first.split_sha256 {
            return Err(BenchmarkContractRejectV1::MixedDatasetOrSplit);
        }
        if manifest.reader_model_id!=first.reader_model_id||manifest.reader_model_revision_sha256!=first.reader_model_revision_sha256 {
            return Err(BenchmarkContractRejectV1::MixedReader);
        }
        if manifest.reader_prompt_sha256!=first.reader_prompt_sha256 { return Err(BenchmarkContractRejectV1::MixedPrompt); }
        if manifest.judge_model_id!=first.judge_model_id||manifest.judge_model_revision_sha256!=first.judge_model_revision_sha256 {
            return Err(BenchmarkContractRejectV1::MixedJudge);
        }
        if manifest.tool_contract_sha256!=first.tool_contract_sha256 { return Err(BenchmarkContractRejectV1::MixedToolContract); }
        if manifest.evaluator_sha256!=first.evaluator_sha256 { return Err(BenchmarkContractRejectV1::MixedEvaluator); }
        if manifest.seed_set_sha256!=first.seed_set_sha256 { return Err(BenchmarkContractRejectV1::MixedSeedSet); }
        if manifest.resource_budget_sha256!=first.resource_budget_sha256
            ||manifest.context_token_budget!=first.context_token_budget
            ||manifest.max_model_calls!=first.max_model_calls
            ||manifest.wall_clock_budget_ms!=first.wall_clock_budget_ms
            ||manifest.cpu_budget_microseconds!=first.cpu_budget_microseconds
            ||manifest.gpu_budget_microseconds!=first.gpu_budget_microseconds
            ||manifest.energy_budget_microwatt_hours!=first.energy_budget_microwatt_hours {
            return Err(BenchmarkContractRejectV1::MixedResourceBudget);
        }
        if manifest.candidate_saw_evaluator_labels||!manifest.evaluator_isolated {
            return Err(BenchmarkContractRejectV1::MixedEvaluationIsolation);
        }
        if manifest.scored_task_count != first.scored_task_count
            || manifest.independent_cluster_count != first.independent_cluster_count {
            return Err(BenchmarkContractRejectV1::MixedCounts);
        }
        if manifest.dependence_unit != first.dependence_unit {
            return Err(BenchmarkContractRejectV1::MixedDependenceUnit);
        }
        if metric_definition_set_sha256(&manifest.metrics)?!=first_metric_defs {
            return Err(BenchmarkContractRejectV1::MixedMetricDefinitions);
        }
        let expected_controls = compute_matched_controls_sha256_v1(manifest)?;
        if manifest.matched_controls_sha256.as_deref() != Some(expected_controls.as_str()) {
            return Err(BenchmarkContractRejectV1::MatchedControlsDigestMismatch);
        }
    }
    let mut receipt=MatchedComparisonReceiptV1 {
        schema_version:SCHEMA_VERSION_V1.into(),record_type:"matched_comparison_receipt".into(),
        comparison_group_id:group.into(),benchmark_id:first.benchmark_id.clone(),track_id:first.track_id.clone(),
        protocol_sha256:first.protocol_sha256.clone(),dataset_revision_sha256:first.dataset_revision_sha256.clone(),
        split_sha256:first.split_sha256.clone(),reader_model_revision_sha256:first.reader_model_revision_sha256.clone(),
        reader_prompt_sha256:first.reader_prompt_sha256.clone(),judge_model_revision_sha256:first.judge_model_revision_sha256.clone(),
        tool_contract_sha256:first.tool_contract_sha256.clone(),evaluator_sha256:first.evaluator_sha256.clone(),
        seed_set_sha256:first.seed_set_sha256.clone(),resource_budget_sha256:first.resource_budget_sha256.clone(),
        matched_controls_sha256:controls.into(),
        arm_ids:arms.into_iter().map(str::to_owned).collect(),run_ids:runs.into_iter().map(str::to_owned).collect(),
        metric_definition_set_sha256:first_metric_defs,receipt_sha256:String::new(),
    };
    let mut unsigned=receipt.clone();unsigned.receipt_sha256.clear();
    receipt.receipt_sha256=digest_serialized(&unsigned)?;
    Ok(receipt)
}

impl MatchedComparisonReceiptV1 {
    /// Recompute the commitment; it is not an external signature.
    pub fn verify_integrity(&self)->bool {
        let mut unsigned=self.clone();unsigned.receipt_sha256.clear();
        digest_serialized(&unsigned).map(|value|value==self.receipt_sha256).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn digest(value:&str)->String { sha256_hex(value.as_bytes()) }

    fn metrics()->Vec<MetricObservationV1> {
        vec![
            MetricObservationV1 { metric_id:"task_accuracy".into(),value:0.7,unit:"fraction".into(),
                direction:MetricDirectionV1::HigherIsBetter,definition_sha256:digest("accuracy-definition"),denominator:Some(100) },
            MetricObservationV1 { metric_id:"p95_latency".into(),value:120.0,unit:"ms".into(),
                direction:MetricDirectionV1::LowerIsBetter,definition_sha256:digest("latency-definition"),denominator:Some(100) }
        ]
    }

    fn manifest(arm:&str,method:&str)->BenchmarkRunManifestV1 {
        BenchmarkRunManifestV1 {
            schema_version:SCHEMA_VERSION_V1.into(),record_type:"benchmark_run_manifest".into(),
            run_id:format!("run-{arm}"),evidence_kind:EvidenceKindV1::LocalExperiment,
            benchmark_id:"longmemeval_v2".into(),track_id:"small".into(),
            protocol_version:"official-insert-query-reader-v2".into(),protocol_sha256:digest("protocol"),
            evaluator_sha256:digest("evaluator"),dataset_id:"longmemeval-v2".into(),
            dataset_revision_sha256:digest("dataset"),split_id:"official-small-test".into(),
            split_sha256:digest("split"),seed_set_sha256:digest("seed-set"),scored_task_count:100,
            independent_cluster_count:50,dependence_unit:"independent-trajectory-cluster".into(),
            system_name:format!("system-{method}"),system_revision_sha256:Some(digest(&format!("system-{method}"))),
            adapter_sha256:Some(digest(&format!("adapter-{method}"))),memory_method_id:method.into(),
            memory_method_family:method.into(),reader_model_id:"fixed-reader".into(),
            reader_model_revision_sha256:digest("reader"),reader_prompt_sha256:digest("prompt"),
            judge_model_id:"official-judge".into(),judge_model_revision_sha256:digest("judge"),
            tool_contract_sha256:digest("tools"),resource_budget_sha256:digest("budget"),
            context_token_budget:16000,max_model_calls:3,wall_clock_budget_ms:200000,
            cpu_budget_microseconds:Some(10000000),gpu_budget_microseconds:Some(20000000),
            energy_budget_microwatt_hours:None,comparison_group_id:Some("paired-lmev2-small".into()),
            arm_id:Some(arm.into()),matched_controls_sha256:None,
            candidate_saw_evaluator_labels:false,evaluator_isolated:true,evidence_uri:None,
            result_artifact_sha256:Some(digest(&format!("result-{arm}"))),
            resource_receipt_sha256:Some(digest(&format!("resources-{arm}"))),metrics:metrics(),
            resources:ResourceObservationV1 {
                p50_latency_ms:Some(80.0),p95_latency_ms:Some(120.0),p99_latency_ms:Some(200.0),
                input_tokens:Some(100000),output_tokens:Some(10000),context_tokens_peak:Some(8000),
                persistent_memory_bytes:Some(1000000),index_bytes:Some(250000),
                provenance_sidecar_bytes:Some(40000),tombstone_journal_bytes:Some(12000),
                cpu_microseconds:Some(1000000),gpu_microseconds:Some(2000000),
                energy_microwatt_hours:None,monetary_micro_usd:Some(150000)
            }
        };
        value.matched_controls_sha256 = Some(compute_matched_controls_sha256_v1(&value).unwrap());
        value
    }

    #[test]
    fn matched_memory_methods_emit_a_verifiable_receipt() {
        let receipt=validate_matched_comparison_v1(&[manifest("bm25","bm25"),manifest("mace","mace-evidence-gated")]).unwrap();
        assert_eq!(receipt.arm_ids.len(),2);
        assert_eq!(receipt.benchmark_id,"longmemeval_v2");
        assert_eq!(receipt.track_id,"small");
        assert!(receipt.verify_integrity());
    }

    #[test]
    fn cross_track_and_cross_benchmark_comparisons_are_rejected() {
        let first=manifest("textual","bm25");
        let mut second=manifest("multimodal","mace");second.track_id="multimodal".into();
        assert_eq!(validate_matched_comparison_v1(&[first.clone(),second]),Err(BenchmarkContractRejectV1::MixedTrack));
        let mut third=manifest("other","mace");third.benchmark_id="ama_bench".into();
        assert_eq!(validate_matched_comparison_v1(&[first,third]),Err(BenchmarkContractRejectV1::MixedBenchmark));
    }

    #[test]
    fn reader_prompt_judge_and_tool_contract_are_frozen() {
        let first=manifest("bm25","bm25");
        let mut changed=manifest("mace","mace");changed.reader_model_revision_sha256=digest("other-reader");
        assert_eq!(validate_matched_comparison_v1(&[first.clone(),changed]),Err(BenchmarkContractRejectV1::MixedReader));
        let mut changed=manifest("mace","mace");changed.reader_prompt_sha256=digest("other-prompt");
        assert_eq!(validate_matched_comparison_v1(&[first.clone(),changed]),Err(BenchmarkContractRejectV1::MixedPrompt));
        let mut changed=manifest("mace","mace");changed.judge_model_revision_sha256=digest("other-judge");
        assert_eq!(validate_matched_comparison_v1(&[first.clone(),changed]),Err(BenchmarkContractRejectV1::MixedJudge));
        let mut changed=manifest("mace","mace");changed.tool_contract_sha256=digest("other-tools");
        assert_eq!(validate_matched_comparison_v1(&[first,changed]),Err(BenchmarkContractRejectV1::MixedToolContract));
    }

    #[test]
    fn data_split_protocol_seed_and_evaluator_are_frozen() {
        let first=manifest("bm25","bm25");
        let mut changed=manifest("mace","mace");changed.split_sha256=digest("other-split");
        assert_eq!(validate_matched_comparison_v1(&[first.clone(),changed]),Err(BenchmarkContractRejectV1::MixedDatasetOrSplit));
        let mut changed=manifest("mace","mace");changed.protocol_sha256=digest("other-protocol");
        assert_eq!(validate_matched_comparison_v1(&[first.clone(),changed]),Err(BenchmarkContractRejectV1::MixedProtocol));
        let mut changed=manifest("mace","mace");changed.seed_set_sha256=digest("other-seeds");
        assert_eq!(validate_matched_comparison_v1(&[first.clone(),changed]),Err(BenchmarkContractRejectV1::MixedSeedSet));
        let mut changed=manifest("mace","mace");changed.evaluator_sha256=digest("other-evaluator");
        assert_eq!(validate_matched_comparison_v1(&[first,changed]),Err(BenchmarkContractRejectV1::MixedEvaluator));
    }

    #[test]
    fn all_resource_budgets_must_match() {
        let first=manifest("bm25","bm25");
        let mut changed=manifest("mace","mace");changed.context_token_budget+=1;
        assert_eq!(validate_matched_comparison_v1(&[first.clone(),changed]),Err(BenchmarkContractRejectV1::MixedResourceBudget));
        let mut changed=manifest("mace","mace");changed.energy_budget_microwatt_hours=Some(50);
        assert_eq!(validate_matched_comparison_v1(&[first,changed]),Err(BenchmarkContractRejectV1::MixedResourceBudget));
    }

    #[test]
    fn evaluator_truth_leakage_or_shared_process_is_rejected() {
        let mut leaked=manifest("leak","leak");leaked.candidate_saw_evaluator_labels=true;
        assert_eq!(validate_run_manifest_v1(&leaked),Err(BenchmarkContractRejectV1::MixedEvaluationIsolation));
        let mut shared=manifest("shared","shared");shared.evaluator_isolated=false;
        assert_eq!(validate_run_manifest_v1(&shared),Err(BenchmarkContractRejectV1::MixedEvaluationIsolation));
    }

    #[test]
    fn metric_definitions_match_while_observed_values_may_differ() {
        let first=manifest("bm25","bm25");
        let mut candidate=manifest("mace","mace");candidate.metrics[0].value=0.9;candidate.metrics[1].value=250.0;
        assert!(validate_matched_comparison_v1(&[first.clone(),candidate.clone()]).is_ok());
        candidate.metrics[0].definition_sha256=digest("other-definition");
        assert_eq!(validate_matched_comparison_v1(&[first,candidate]),Err(BenchmarkContractRejectV1::MixedMetricDefinitions));
    }

    #[test]
    fn external_reported_results_are_not_local_reproduction_arms() {
        let mut external=manifest("external","vendor");external.evidence_kind=EvidenceKindV1::ExternalReported;
        external.evidence_uri=Some("https://example.org/results".into());
        external.system_revision_sha256=None;external.adapter_sha256=None;
        external.result_artifact_sha256=None;external.resource_receipt_sha256=None;
        assert!(validate_run_manifest_v1(&external).is_ok());
        assert_eq!(validate_matched_comparison_v1(&[external,manifest("local","mace")]),Err(BenchmarkContractRejectV1::MixedEvidenceKind));
    }

    #[test]
    fn duplicate_arms_and_single_arm_sets_fail_closed() {
        assert_eq!(validate_matched_comparison_v1(&[manifest("one","bm25")]),Err(BenchmarkContractRejectV1::TooFewComparisonArms));
        let first=manifest("bm25","bm25");let mut second=manifest("copy","mace");second.arm_id=Some("bm25".into());
        assert_eq!(validate_matched_comparison_v1(&[first,second]),Err(BenchmarkContractRejectV1::DuplicateArm));
    }

    #[test]
    fn invalid_latency_quantiles_empty_or_duplicate_metrics_fail() {
        let mut invalid=manifest("bad","bad");invalid.resources.p50_latency_ms=Some(200.0);invalid.resources.p95_latency_ms=Some(100.0);
        assert_eq!(validate_run_manifest_v1(&invalid),Err(BenchmarkContractRejectV1::InvalidResourceObservation));
        let mut empty=manifest("empty","empty");empty.metrics.clear();
        assert_eq!(validate_run_manifest_v1(&empty),Err(BenchmarkContractRejectV1::InvalidMetric));
        let mut duplicate=manifest("duplicate","duplicate");duplicate.metrics.push(duplicate.metrics[0].clone());
        assert_eq!(validate_run_manifest_v1(&duplicate),Err(BenchmarkContractRejectV1::DuplicateMetric));
    }

    #[test]
    fn nonfinite_metric_values_and_invalid_denominators_fail() {
        let mut invalid=manifest("nan","nan");invalid.metrics[0].value=f64::NAN;
        assert_eq!(validate_run_manifest_v1(&invalid),Err(BenchmarkContractRejectV1::InvalidMetric));
        let mut invalid=manifest("zero-denom","zero-denom");invalid.metrics[0].denominator=Some(0);
        assert_eq!(validate_run_manifest_v1(&invalid),Err(BenchmarkContractRejectV1::InvalidMetric));
    }

    #[test]
    fn matched_controls_digest_is_recomputed_not_trusted() {
        let mut first = manifest("bm25", "bm25");
        let mut candidate = manifest("mace", "mace");
        first.reader_prompt_sha256 = digest("changed-on-both-arms");
        candidate.reader_prompt_sha256 = digest("changed-on-both-arms");
        assert_eq!(
            validate_matched_comparison_v1(&[first, candidate]),
            Err(BenchmarkContractRejectV1::MatchedControlsDigestMismatch)
        );
    }

    #[test]
    fn scored_task_counts_and_dependence_units_must_match() {
        let first = manifest("bm25", "bm25");
        let mut candidate = manifest("mace", "mace");
        candidate.scored_task_count += 1;
        assert_eq!(
            validate_matched_comparison_v1(&[first.clone(), candidate]),
            Err(BenchmarkContractRejectV1::MixedCounts)
        );
        let mut candidate = manifest("mace", "mace");
        candidate.dependence_unit = "message_row".into();
        assert_eq!(
            validate_matched_comparison_v1(&[first, candidate]),
            Err(BenchmarkContractRejectV1::MixedDependenceUnit)
        );
    }


    #[test]
    fn metric_denominators_are_part_of_the_matched_definition() {
        let first = manifest("bm25", "bm25");
        let mut candidate = manifest("mace", "mace");
        candidate.metrics[0].denominator = Some(99);
        candidate.matched_controls_sha256 = first.matched_controls_sha256.clone();
        assert_eq!(
            validate_matched_comparison_v1(&[first, candidate]),
            Err(BenchmarkContractRejectV1::MixedMetricDefinitions)
        );
    }

}
