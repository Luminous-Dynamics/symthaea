//! Evidence-bound candidate scoring and Pareto evaluation for temporal/active-sensing TinyML tracks.
//!
//! A candidate is not ranked from a scalar summary alone. This module replays the causal
//! prequential trace, recomputes matched application-event uncertainty from the underlying
//! phase-energy receipt, validates target hardware measurements, checks hard constraints,
//! and returns a non-dominated frontier across quality, reliability, latency, energy and memory.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::application_event_uncertainty::{
    evaluate_application_event_energy_uncertainty,
    validate_application_event_energy_uncertainty_receipt,
    ApplicationEventEnergyUncertaintyReceipt, ApplicationEventRepeatabilityProtocol,
};
use crate::edge_intelligence_benchmark::{
    audit_prequential_execution, validate_edge_benchmark_manifest,
    validate_temporal_stream_dataset, EdgeBenchmarkManifest, EdgeBenchmarkTrack,
    EdgeCandidateDefinition, PrequentialAuditReceipt, PrequentialExecutionTrace,
    TemporalStreamDataset,
};
use crate::metrology_uncertainty::MetrologyUncertaintyManifest;
use crate::phase_energy_audit::{
    validate_phase_energy_audit_receipt, ApplicationPhaseProtocolManifest,
    PhaseEnergyAuditReceipt,
};

const HARDWARE_SCHEMA: &str = "symthaea.edge-hardware-measurement";
const EVIDENCE_SCHEMA: &str = "symthaea.edge-candidate-evidence-run";
const PARETO_SCHEMA: &str = "symthaea.edge-candidate-pareto-evaluation";
const VERSION: &str = "0.1.0";
const HARDWARE_DOMAIN: &[u8] = b"symthaea.edge-hardware-measurement.v0.1";
const EVIDENCE_DOMAIN: &[u8] = b"symthaea.edge-candidate-evidence-run.v0.1";
const PARETO_DOMAIN: &[u8] = b"symthaea.edge-candidate-pareto-evaluation.v0.1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeEvidenceError {
    pub field: String,
    pub message: String,
}
impl EdgeEvidenceError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}
impl fmt::Display for EdgeEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for EdgeEvidenceError {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
fn digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(bytes);
    format!("{:x}", h.finalize())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeHardwareMeasurementReceipt {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub candidate_id: String,
    pub artifact_sha256: String,
    pub target_id: String,
    pub board_revision: String,
    pub cpu_core: String,
    pub accelerator_profile: String,
    pub compiler_sha256: String,
    pub firmware_sha256: String,
    pub linker_map_sha256: String,
    pub measurement_protocol_sha256: String,
    pub device_session_sha256: String,
    /// One latency observation per declared application event or inference invocation.
    pub latency_samples_us: Vec<u64>,
    pub peak_sram_bytes: u64,
    pub flash_bytes: u64,
    pub sensor_samples_per_hour: Option<f64>,
    pub online_update_energy_uj: Option<f64>,
    pub adaptation_bytes_written: Option<u64>,
    pub hardware_receipt_sha256: String,
}

pub fn compute_edge_hardware_measurement_sha256(
    receipt: &EdgeHardwareMeasurementReceipt,
) -> Result<String, EdgeEvidenceError> {
    let mut canonical = receipt.clone();
    canonical.hardware_receipt_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| EdgeEvidenceError::new("hardware_receipt", error.to_string()))?;
    Ok(digest(HARDWARE_DOMAIN, &bytes))
}

pub fn validate_edge_hardware_measurement_receipt(
    receipt: &EdgeHardwareMeasurementReceipt,
) -> Result<(), EdgeEvidenceError> {
    if receipt.schema != HARDWARE_SCHEMA || receipt.version != VERSION {
        return Err(EdgeEvidenceError::new("hardware.schema_version", "unsupported hardware receipt schema/version"));
    }
    for (field, value) in [
        ("artifact_sha256", receipt.artifact_sha256.as_str()),
        ("compiler_sha256", receipt.compiler_sha256.as_str()),
        ("firmware_sha256", receipt.firmware_sha256.as_str()),
        ("linker_map_sha256", receipt.linker_map_sha256.as_str()),
        ("measurement_protocol_sha256", receipt.measurement_protocol_sha256.as_str()),
        ("device_session_sha256", receipt.device_session_sha256.as_str()),
        ("hardware_receipt_sha256", receipt.hardware_receipt_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            return Err(EdgeEvidenceError::new(format!("hardware.{field}"), "must be lowercase SHA-256 hex"));
        }
    }
    for (field, value) in [
        ("run_id", receipt.run_id.as_str()),
        ("candidate_id", receipt.candidate_id.as_str()),
        ("target_id", receipt.target_id.as_str()),
        ("board_revision", receipt.board_revision.as_str()),
        ("cpu_core", receipt.cpu_core.as_str()),
        ("accelerator_profile", receipt.accelerator_profile.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(EdgeEvidenceError::new(format!("hardware.{field}"), "must be non-empty"));
        }
    }
    if receipt.latency_samples_us.is_empty()
        || receipt.latency_samples_us.iter().any(|latency| *latency == 0)
        || receipt.peak_sram_bytes == 0
        || receipt.flash_bytes == 0
    {
        return Err(EdgeEvidenceError::new("hardware.measurements", "latency samples, peak SRAM and flash must be positive"));
    }
    if receipt.sensor_samples_per_hour.is_some_and(|value| !value.is_finite() || value <= 0.0)
        || receipt.online_update_energy_uj.is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(EdgeEvidenceError::new("hardware.optional_metrics", "optional sensor/update metrics are outside valid bounds"));
    }
    if compute_edge_hardware_measurement_sha256(receipt)? != receipt.hardware_receipt_sha256 {
        return Err(EdgeEvidenceError::new("hardware_receipt_sha256", "does not match canonical hardware receipt"));
    }
    Ok(())
}

/// Nearest-rank p95 latency, returned in milliseconds.
pub fn p95_latency_ms(samples_us: &[u64]) -> Result<f64, EdgeEvidenceError> {
    if samples_us.is_empty() || samples_us.iter().any(|value| *value == 0) {
        return Err(EdgeEvidenceError::new("latency_samples_us", "must contain positive observations"));
    }
    let mut sorted = samples_us.to_vec();
    sorted.sort_unstable();
    let rank = (95 * sorted.len() + 99) / 100;
    let index = rank.saturating_sub(1).min(sorted.len() - 1);
    Ok(sorted[index] as f64 / 1_000.0)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeCandidateEvidenceBundle {
    pub run_id: String,
    pub candidate_id: String,
    pub hardware_measurement: EdgeHardwareMeasurementReceipt,
    pub prequential_trace: PrequentialExecutionTrace,
    pub prequential_audit_receipt: PrequentialAuditReceipt,
    pub phase_energy_receipt: PhaseEnergyAuditReceipt,
    pub phase_protocol: ApplicationPhaseProtocolManifest,
    pub repeatability_protocol: ApplicationEventRepeatabilityProtocol,
    pub metrology_manifest: MetrologyUncertaintyManifest,
    pub event_energy_uncertainty_receipt: ApplicationEventEnergyUncertaintyReceipt,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeCandidateEvidenceRun {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub benchmark_manifest_sha256: String,
    pub dataset_sha256: String,
    /// Exactly one evidence bundle per candidate in the frozen manifest, sorted by candidate ID.
    pub candidates: Vec<EdgeCandidateEvidenceBundle>,
    pub run_sha256: String,
}

pub fn compute_edge_candidate_evidence_run_sha256(
    run: &EdgeCandidateEvidenceRun,
) -> Result<String, EdgeEvidenceError> {
    let mut canonical = run.clone();
    canonical.run_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| EdgeEvidenceError::new("edge_evidence_run", error.to_string()))?;
    Ok(digest(EVIDENCE_DOMAIN, &bytes))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeCandidateParetoMetrics {
    pub candidate_id: String,
    pub candidate_family: String,
    pub hard_feasible: bool,
    pub rejection_reasons: Vec<String>,
    pub quality_score: f64,
    pub false_alarms_per_hour: f64,
    pub missed_events_per_hour: f64,
    pub coverage: f64,
    pub brier_score: f64,
    pub p95_latency_ms: f64,
    pub mean_energy_per_successful_event_uj: f64,
    pub expanded_energy_uncertainty_uj: f64,
    pub energy_upper_bound_uj: f64,
    pub peak_sram_bytes: u64,
    pub flash_bytes: u64,
    pub sensor_samples_per_hour: Option<f64>,
    pub online_update_energy_uj: Option<f64>,
    pub adaptation_bytes_written: Option<u64>,
    pub prequential_audit_sha256: String,
    pub phase_energy_receipt_sha256: String,
    pub event_energy_uncertainty_receipt_sha256: String,
    pub hardware_receipt_sha256: String,
    pub pareto_front_member: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeCandidateParetoEvaluation {
    pub schema: String,
    pub version: String,
    pub run_sha256: String,
    pub benchmark_manifest_sha256: String,
    pub dataset_sha256: String,
    pub candidates: Vec<EdgeCandidateParetoMetrics>,
    pub pareto_candidate_ids: Vec<String>,
    pub evaluation_sha256: String,
}

fn quality_score(
    quality_metric_id: &str,
    prequential: &PrequentialAuditReceipt,
) -> Result<f64, EdgeEvidenceError> {
    let score = match quality_metric_id {
        "accuracy" | "top1_accuracy" => Some(prequential.metrics.accuracy),
        "balanced_accuracy" => prequential.metrics.balanced_accuracy,
        "selective_accuracy" => prequential.metrics.selective_accuracy,
        "coverage" => Some(prequential.metrics.coverage),
        other => return Err(EdgeEvidenceError::new(
            "quality_metric_id",
            format!("unsupported prequential metric {other:?}; use accuracy, balanced_accuracy, selective_accuracy or coverage"),
        )),
    };
    score.ok_or_else(|| EdgeEvidenceError::new("quality_metric_id", "requested metric is undefined for this stream"))
}

fn validate_candidate_bundle(
    manifest: &EdgeBenchmarkManifest,
    dataset: &TemporalStreamDataset,
    run_id: &str,
    definition: &EdgeCandidateDefinition,
    bundle: &EdgeCandidateEvidenceBundle,
) -> Result<EdgeCandidateParetoMetrics, EdgeEvidenceError> {
    if bundle.run_id != run_id || bundle.candidate_id != definition.candidate_id {
        return Err(EdgeEvidenceError::new("candidate.identity", "evidence bundle run/candidate IDs do not match the frozen manifest"));
    }
    let hardware = &bundle.hardware_measurement;
    validate_edge_hardware_measurement_receipt(hardware)?;
    if hardware.run_id != run_id
        || hardware.candidate_id != definition.candidate_id
        || hardware.artifact_sha256 != definition.artifact_sha256
        || hardware.latency_samples_us.len() != dataset.observations.len()
        || hardware.target_id != manifest.target.target_id
        || hardware.board_revision != manifest.target.board_revision
        || hardware.cpu_core != manifest.target.cpu_core
        || hardware.accelerator_profile != manifest.target.accelerator_profile
    {
        return Err(EdgeEvidenceError::new("hardware.identity", "hardware measurement does not match run, artifact or frozen target"));
    }

    let trace = &bundle.prequential_trace;
    if trace.run_id != run_id
        || trace.candidate_id != definition.candidate_id
        || trace.candidate_artifact_sha256 != definition.artifact_sha256
        || trace.dataset_sha256 != dataset.dataset_sha256
        || trace.energy_receipt_sha256 != bundle.phase_energy_receipt.receipt_sha256
        || trace.metrology_receipt_sha256 != bundle.event_energy_uncertainty_receipt.receipt_sha256
    {
        return Err(EdgeEvidenceError::new("prequential_trace.binding", "prequential trace does not bind exact run, candidate, dataset, phase energy and event uncertainty receipts"));
    }
    let prequential = audit_prequential_execution(dataset, trace)
        .map_err(|error| EdgeEvidenceError::new("prequential_audit", error.to_string()))?;
    if prequential != bundle.prequential_audit_receipt {
        return Err(EdgeEvidenceError::new("prequential_audit_receipt", "supplied prequential audit receipt does not match independent replay"));
    }

    validate_phase_energy_audit_receipt(&bundle.phase_energy_receipt)
        .map_err(|error| EdgeEvidenceError::new("phase_energy_receipt", error.to_string()))?;
    if bundle.phase_energy_receipt.phase_protocol_sha256 != bundle.phase_protocol.protocol_sha256 {
        return Err(EdgeEvidenceError::new("phase_protocol.binding", "phase energy receipt uses a different phase protocol"));
    }
    let rebuilt_event_energy = evaluate_application_event_energy_uncertainty(
        &bundle.phase_energy_receipt,
        &bundle.phase_protocol,
        &bundle.repeatability_protocol,
        &bundle.metrology_manifest,
    ).map_err(|error| EdgeEvidenceError::new("event_energy_uncertainty", error.to_string()))?;
    validate_application_event_energy_uncertainty_receipt(&bundle.event_energy_uncertainty_receipt)
        .map_err(|error| EdgeEvidenceError::new("event_energy_uncertainty_receipt", error.to_string()))?;
    if rebuilt_event_energy != bundle.event_energy_uncertainty_receipt {
        return Err(EdgeEvidenceError::new("event_energy_uncertainty_receipt", "supplied uncertainty receipt differs from independent recomputation"));
    }
    if bundle.event_energy_uncertainty_receipt.candidate_artifact_sha256 != definition.artifact_sha256
        || bundle.event_energy_uncertainty_receipt.device_session_sha256 != hardware.device_session_sha256
    {
        return Err(EdgeEvidenceError::new("event_energy_uncertainty.identity", "event energy evidence is for a different artifact or physical device session"));
    }

    let score = quality_score(&manifest.quality_metric_id, &prequential)?;
    let latency = p95_latency_ms(&hardware.latency_samples_us)?;
    let mut rejection_reasons = Vec::new();
    let available_sram = manifest.target.sram_budget_bytes
        .checked_sub(manifest.target.protected_sram_reserve_bytes)
        .ok_or_else(|| EdgeEvidenceError::new("target.sram_budget", "protected reserve exceeds total SRAM"))?;
    if score < manifest.minimum_quality_score {
        rejection_reasons.push("quality_below_minimum".to_owned());
    }
    if prequential.metrics.false_alarms_per_hour > manifest.maximum_false_alarms_per_hour {
        rejection_reasons.push("false_alarm_limit_exceeded".to_owned());
    }
    if prequential.metrics.missed_events_per_hour > manifest.maximum_missed_events_per_hour {
        rejection_reasons.push("missed_event_limit_exceeded".to_owned());
    }
    if prequential.metrics.coverage < manifest.minimum_coverage {
        rejection_reasons.push("coverage_below_minimum".to_owned());
    }
    if latency > manifest.target.p95_deadline_ms as f64 {
        rejection_reasons.push("p95_deadline_missed".to_owned());
    }
    if hardware.peak_sram_bytes > available_sram {
        rejection_reasons.push("sram_budget_exceeded".to_owned());
    }
    if hardware.flash_bytes > manifest.target.flash_budget_bytes {
        rejection_reasons.push("flash_budget_exceeded".to_owned());
    }

    Ok(EdgeCandidateParetoMetrics {
        candidate_id: definition.candidate_id.clone(),
        candidate_family: format!("{:?}", definition.family),
        hard_feasible: rejection_reasons.is_empty(),
        rejection_reasons,
        quality_score: score,
        false_alarms_per_hour: prequential.metrics.false_alarms_per_hour,
        missed_events_per_hour: prequential.metrics.missed_events_per_hour,
        coverage: prequential.metrics.coverage,
        brier_score: prequential.metrics.brier_score,
        p95_latency_ms: latency,
        mean_energy_per_successful_event_uj: rebuilt_event_energy.mean_energy_uj,
        expanded_energy_uncertainty_uj: rebuilt_event_energy.expanded_total_uncertainty_uj,
        energy_upper_bound_uj: rebuilt_event_energy.upper_bound_uj,
        peak_sram_bytes: hardware.peak_sram_bytes,
        flash_bytes: hardware.flash_bytes,
        sensor_samples_per_hour: hardware.sensor_samples_per_hour,
        online_update_energy_uj: hardware.online_update_energy_uj,
        adaptation_bytes_written: hardware.adaptation_bytes_written,
        prequential_audit_sha256: prequential.receipt_sha256,
        phase_energy_receipt_sha256: bundle.phase_energy_receipt.receipt_sha256.clone(),
        event_energy_uncertainty_receipt_sha256: rebuilt_event_energy.receipt_sha256,
        hardware_receipt_sha256: hardware.hardware_receipt_sha256.clone(),
        pareto_front_member: false,
    })
}

fn objectives(metrics: &EdgeCandidateParetoMetrics) -> [f64; 9] {
    [
        metrics.quality_score,
        -metrics.false_alarms_per_hour,
        -metrics.missed_events_per_hour,
        -metrics.p95_latency_ms,
        -metrics.energy_upper_bound_uj,
        -(metrics.peak_sram_bytes as f64),
        -(metrics.flash_bytes as f64),
        metrics.coverage,
        -metrics.brier_score,
    ]
}
fn dominates(left: &[f64; 9], right: &[f64; 9]) -> bool {
    let no_worse = left.iter().zip(right).all(|(l, r)| l >= r);
    let strictly_better = left.iter().zip(right).any(|(l, r)| l > r);
    no_worse && strictly_better
}

/// Replay all candidate evidence and produce a feasibility-filtered Pareto frontier.
///
/// This v0.1 path covers temporal-adaptation and active-sensing application events. Static
/// inference remains a separate MLPerf Tiny-compatible track and should not be mixed with
/// event-energy semantics or accelerator tiers.
pub fn evaluate_edge_candidate_evidence_run(
    manifest: &EdgeBenchmarkManifest,
    dataset: &TemporalStreamDataset,
    run: &EdgeCandidateEvidenceRun,
) -> Result<EdgeCandidateParetoEvaluation, EdgeEvidenceError> {
    validate_edge_benchmark_manifest(manifest)
        .map_err(|error| EdgeEvidenceError::new("benchmark_manifest", error.to_string()))?;
    validate_temporal_stream_dataset(dataset)
        .map_err(|error| EdgeEvidenceError::new("temporal_dataset", error.to_string()))?;
    if !matches!(manifest.track, EdgeBenchmarkTrack::TemporalAdaptation | EdgeBenchmarkTrack::ActiveSensing) {
        return Err(EdgeEvidenceError::new("benchmark_track", "evidence evaluator only supports temporal and active-sensing tracks"));
    }
    if run.schema != EVIDENCE_SCHEMA || run.version != VERSION || run.run_id.trim().is_empty() {
        return Err(EdgeEvidenceError::new("run.schema_version", "unsupported evidence run schema/version"));
    }
    if run.benchmark_manifest_sha256 != manifest.manifest_sha256
        || run.dataset_sha256 != dataset.dataset_sha256
    {
        return Err(EdgeEvidenceError::new("run.binding", "run is bound to a different frozen manifest or dataset"));
    }
    if !is_sha256(&run.run_sha256) || compute_edge_candidate_evidence_run_sha256(run)? != run.run_sha256 {
        return Err(EdgeEvidenceError::new("run.run_sha256", "does not match canonical evidence run"));
    }
    if run.candidates.len() != manifest.candidate_definitions.len()
        || run.candidates.windows(2).any(|pair| pair[0].candidate_id >= pair[1].candidate_id)
    {
        return Err(EdgeEvidenceError::new("run.candidates", "run must contain exactly one sorted evidence bundle per frozen candidate"));
    }

    let mut metrics = Vec::with_capacity(run.candidates.len());
    for (definition, bundle) in manifest.candidate_definitions.iter().zip(&run.candidates) {
        metrics.push(validate_candidate_bundle(manifest, dataset, &run.run_id, definition, bundle)?);
    }

    let mut frontier_ids = Vec::new();
    for index in 0..metrics.len() {
        if !metrics[index].hard_feasible {
            continue;
        }
        let current = objectives(&metrics[index]);
        let is_dominated = metrics.iter().enumerate().any(|(other_index, other)| {
            other_index != index
                && other.hard_feasible
                && dominates(&objectives(other), &current)
        });
        if !is_dominated {
            frontier_ids.push(metrics[index].candidate_id.clone());
        }
    }
    frontier_ids.sort();
    for item in &mut metrics {
        item.pareto_front_member = frontier_ids.contains(&item.candidate_id);
    }

    let mut evaluation = EdgeCandidateParetoEvaluation {
        schema: PARETO_SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        run_sha256: run.run_sha256.clone(),
        benchmark_manifest_sha256: manifest.manifest_sha256.clone(),
        dataset_sha256: dataset.dataset_sha256.clone(),
        candidates: metrics,
        pareto_candidate_ids: frontier_ids,
        evaluation_sha256: String::new(),
    };
    let mut canonical = evaluation.clone();
    canonical.evaluation_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| EdgeEvidenceError::new("pareto_evaluation", error.to_string()))?;
    evaluation.evaluation_sha256 = digest(PARETO_DOMAIN, &bytes);
    Ok(evaluation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p95_latency_uses_nearest_rank_in_milliseconds() {
        assert_eq!(p95_latency_ms(&[1_000, 2_000, 3_000, 4_000]).unwrap(), 4.0);
        assert_eq!(p95_latency_ms(&[1_000, 2_000, 3_000, 4_000, 5_000]).unwrap(), 5.0);
    }

    #[test]
    fn invalid_latency_samples_are_rejected() {
        assert!(p95_latency_ms(&[]).is_err());
        assert!(p95_latency_ms(&[1, 0, 2]).is_err());
    }

    #[test]
    fn hardware_receipt_binds_measurements_and_artifact() {
        let mut receipt = EdgeHardwareMeasurementReceipt {
            schema: HARDWARE_SCHEMA.into(),
            version: VERSION.into(),
            run_id: "run-1".into(),
            candidate_id: "candidate-1".into(),
            artifact_sha256: std::iter::repeat_n('a', 64).collect(),
            target_id: "target-1".into(),
            board_revision: "rev-a".into(),
            cpu_core: "cortex-m55".into(),
            accelerator_profile: "cpu-only".into(),
            compiler_sha256: std::iter::repeat_n('b', 64).collect(),
            firmware_sha256: std::iter::repeat_n('c', 64).collect(),
            linker_map_sha256: std::iter::repeat_n('d', 64).collect(),
            measurement_protocol_sha256: std::iter::repeat_n('e', 64).collect(),
            device_session_sha256: std::iter::repeat_n('f', 64).collect(),
            latency_samples_us: vec![1_000, 2_000, 3_000],
            peak_sram_bytes: 128 * 1024,
            flash_bytes: 512 * 1024,
            sensor_samples_per_hour: Some(360.0),
            online_update_energy_uj: Some(1.25),
            adaptation_bytes_written: Some(128),
            hardware_receipt_sha256: String::new(),
        };
        receipt.hardware_receipt_sha256 = compute_edge_hardware_measurement_sha256(&receipt).unwrap();
        validate_edge_hardware_measurement_receipt(&receipt).unwrap();

        receipt.flash_bytes += 1;
        assert!(validate_edge_hardware_measurement_receipt(&receipt).is_err());
    }
}
