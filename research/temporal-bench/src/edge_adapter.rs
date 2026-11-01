//! Multi-timescale temporal competence adapter for edge-intelligence evaluation.
//!
//! This module separates three artifacts:
//! 1. a frozen task/split/candidate manifest;
//! 2. candidate-produced output receipts that contain no oracle or label fields;
//! 3. evaluator-owned metric receipts that bind the candidate output to an oracle commitment.
//!
//! The types and digests prevent accidental oracle inclusion in candidate receipts. They are
//! NOT a process sandbox. Confirmatory runs must execute the candidate separately from the
//! evaluator with OS-level file/IPC permissions and independently replayable receipts.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt;

use super::digest::CanonicalHasher;

const MANIFEST_SCHEMA: &str = "symthaea.temporal.edge-adapter-manifest";
const CANDIDATE_SCHEMA: &str = "symthaea.temporal.edge-candidate-output";
const EVALUATOR_SCHEMA: &str = "symthaea.temporal.edge-evaluator-receipt";
const SYSTEM_SCHEMA: &str = "symthaea.temporal.edge-system-receipt";
const VERSION: &str = "0.1.0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgeAdapterError {
    pub field: String,
    pub message: String,
}

impl EdgeAdapterError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for EdgeAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for EdgeAdapterError {}

fn digest_is_nonzero(value: &[u8; 32]) -> bool {
    value.iter().any(|byte| *byte != 0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum TemporalTaskId {
    T1SparseEvent = 1,
    T2DelayedCueRecall = 2,
    T3MultiTimescale = 3,
    T4OnlineDrift = 4,
    T5IrregularMissing = 5,
    T6RetentionInvalidation = 6,
    T7PhasePrediction = 7,
    T8DelayedMatchSample = 8,
    T9EvidenceAccumulation = 9,
    T10BitemporalKnowledge = 10,
}

impl TemporalTaskId {
    pub const ALL: [Self; 10] = [
        Self::T1SparseEvent,
        Self::T2DelayedCueRecall,
        Self::T3MultiTimescale,
        Self::T4OnlineDrift,
        Self::T5IrregularMissing,
        Self::T6RetentionInvalidation,
        Self::T7PhasePrediction,
        Self::T8DelayedMatchSample,
        Self::T9EvidenceAccumulation,
        Self::T10BitemporalKnowledge,
    ];
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum SplitRole {
    Training = 0,
    Development = 1,
    Calibration = 2,
    Confirmatory = 3,
}

impl SplitRole {
    const ALL: [Self; 4] = [
        Self::Training,
        Self::Development,
        Self::Calibration,
        Self::Confirmatory,
    ];
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum BenchmarkTrack {
    Algorithm = 0,
    System = 1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum BaselineFamily {
    ThresholdEma = 0,
    StaticTinyMl = 1,
    AdaptiveTinyMl = 2,
    Gru = 3,
    Lstm = 4,
    GruD = 5,
    Tcn = 6,
    OdeRnn = 7,
    LatentOde = 8,
    AdaptiveLif = 9,
    BrainTraceDRtrl = 10,
    BrainTracePpProp = 11,
    EProp = 12,
    Stdl = 13,
    Bptt = 14,
    HdcOnly = 15,
    CfcOnly = 16,
    HdcLtc = 17,
    HdcLtcConfidenceGated = 18,
    AssociativeMemory = 19,
    KeyValueOverwrite = 20,
    RecencyDecay = 21,
    BitemporalLedger = 22,
    Rag = 23,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetricDirection {
    Maximize,
    Minimize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TaskMetricDefinition {
    pub metric_id: &'static str,
    pub unit: &'static str,
    pub direction: MetricDirection,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalTaskDefinition {
    pub task_id: TemporalTaskId,
    pub short_name: &'static str,
    pub capability: &'static str,
    pub metrics: &'static [TaskMetricDefinition],
    pub required_baselines: &'static [BaselineFamily],
}

const T1_METRICS: &[TaskMetricDefinition] = &[
    TaskMetricDefinition { metric_id: "target_event_precision_ppm", unit: "ppm", direction: MetricDirection::Maximize },
    TaskMetricDefinition { metric_id: "target_event_recall_ppm", unit: "ppm", direction: MetricDirection::Maximize },
    TaskMetricDefinition { metric_id: "idle_false_alarms_per_hour_milli", unit: "milli_false_alarms_per_hour", direction: MetricDirection::Minimize },
    TaskMetricDefinition { metric_id: "decision_latency_p95_us", unit: "microseconds", direction: MetricDirection::Minimize },
    TaskMetricDefinition { metric_id: "duplicate_activation_count", unit: "events", direction: MetricDirection::Minimize },
];
const T2_METRICS: &[TaskMetricDefinition] = &[
    TaskMetricDefinition { metric_id: "recall_accuracy_ppm", unit: "ppm", direction: MetricDirection::Maximize },
    TaskMetricDefinition { metric_id: "worst_delay_accuracy_ppm", unit: "ppm", direction: MetricDirection::Maximize },
    TaskMetricDefinition { metric_id: "mean_recall_latency_us", unit: "microseconds", direction: MetricDirection::Minimize },
];
const T3_METRICS: &[TaskMetricDefinition] = &[
    TaskMetricDefinition { metric_id: "state_accuracy_ppm", unit: "ppm", direction: MetricDirection::Maximize },
    TaskMetricDefinition { metric_id: "forecast_mae_milliunits", unit: "milliunits", direction: MetricDirection::Minimize },
    TaskMetricDefinition { metric_id: "worst_timescale_accuracy_ppm", unit: "ppm", direction: MetricDirection::Maximize },
];
const T4_METRICS: &[TaskMetricDefinition] = &[
    TaskMetricDefinition { metric_id: "change_detection_delay_steps", unit: "steps", direction: MetricDirection::Minimize },
    TaskMetricDefinition { metric_id: "drift_recovery_delay_steps", unit: "steps", direction: MetricDirection::Minimize },
    TaskMetricDefinition { metric_id: "false_change_count", unit: "events", direction: MetricDirection::Minimize },
    TaskMetricDefinition { metric_id: "prior_regime_retention_ppm", unit: "ppm", direction: MetricDirection::Maximize },
    TaskMetricDefinition { metric_id: "cumulative_regret_milli", unit: "milli_loss", direction: MetricDirection::Minimize },
];
const T5_METRICS: &[TaskMetricDefinition] = &[
    TaskMetricDefinition { metric_id: "accuracy_by_gap_ppm", unit: "ppm", direction: MetricDirection::Maximize },
    TaskMetricDefinition { metric_id: "accuracy_by_missingness_ppm", unit: "ppm", direction: MetricDirection::Maximize },
    TaskMetricDefinition { metric_id: "invalid_input_acceptance_count", unit: "events", direction: MetricDirection::Minimize },
];
const T6_METRICS: &[TaskMetricDefinition] = &[
    TaskMetricDefinition { metric_id: "valid_recall_before_invalidation_ppm", unit: "ppm", direction: MetricDirection::Maximize },
    TaskMetricDefinition { metric_id: "stale_recall_after_invalidation_ppm", unit: "ppm", direction: MetricDirection::Minimize },
    TaskMetricDefinition { metric_id: "invalidation_response_latency_steps", unit: "steps", direction: MetricDirection::Minimize },
];
const T7_METRICS: &[TaskMetricDefinition] = &[
    TaskMetricDefinition { metric_id: "phase_error_millidegrees", unit: "millidegrees", direction: MetricDirection::Minimize },
    TaskMetricDefinition { metric_id: "next_crossing_error_us", unit: "microseconds", direction: MetricDirection::Minimize },
    TaskMetricDefinition { metric_id: "prediction_coverage_ppm", unit: "ppm", direction: MetricDirection::Maximize },
];
const T8_METRICS: &[TaskMetricDefinition] = &[
    TaskMetricDefinition { metric_id: "dms_accuracy_ppm", unit: "ppm", direction: MetricDirection::Maximize },
    TaskMetricDefinition { metric_id: "worst_delay_accuracy_ppm", unit: "ppm", direction: MetricDirection::Maximize },
    TaskMetricDefinition { metric_id: "invalid_prediction_count", unit: "events", direction: MetricDirection::Minimize },
];
const T9_METRICS: &[TaskMetricDefinition] = &[
    TaskMetricDefinition { metric_id: "evidence_accuracy_ppm", unit: "ppm", direction: MetricDirection::Maximize },
    TaskMetricDefinition { metric_id: "evidence_false_alarm_count", unit: "events", direction: MetricDirection::Minimize },
    TaskMetricDefinition { metric_id: "evidence_decision_latency_us", unit: "microseconds", direction: MetricDirection::Minimize },
];
const T10_METRICS: &[TaskMetricDefinition] = &[
    TaskMetricDefinition { metric_id: "bitemporal_exact_accuracy_ppm", unit: "ppm", direction: MetricDirection::Maximize },
    TaskMetricDefinition { metric_id: "future_knowledge_leak_count", unit: "events", direction: MetricDirection::Minimize },
    TaskMetricDefinition { metric_id: "stale_fact_use_count", unit: "events", direction: MetricDirection::Minimize },
    TaskMetricDefinition { metric_id: "historical_as_known_accuracy_ppm", unit: "ppm", direction: MetricDirection::Maximize },
];

const T1_BASELINES: &[BaselineFamily] = &[
    BaselineFamily::ThresholdEma, BaselineFamily::StaticTinyMl, BaselineFamily::Tcn,
    BaselineFamily::Gru, BaselineFamily::AdaptiveLif, BaselineFamily::HdcOnly,
    BaselineFamily::CfcOnly, BaselineFamily::HdcLtc,
];
const T2_BASELINES: &[BaselineFamily] = &[
    BaselineFamily::Gru, BaselineFamily::Lstm, BaselineFamily::AssociativeMemory,
    BaselineFamily::HdcOnly, BaselineFamily::HdcLtc,
];
const T3_BASELINES: &[BaselineFamily] = &[
    BaselineFamily::Gru, BaselineFamily::Lstm, BaselineFamily::Tcn,
    BaselineFamily::OdeRnn, BaselineFamily::LatentOde, BaselineFamily::CfcOnly,
    BaselineFamily::HdcLtc,
];
const T4_BASELINES: &[BaselineFamily] = &[
    BaselineFamily::ThresholdEma, BaselineFamily::GruD, BaselineFamily::BrainTraceDRtrl,
    BaselineFamily::BrainTracePpProp, BaselineFamily::EProp,
    BaselineFamily::HdcLtcConfidenceGated,
];
const T5_BASELINES: &[BaselineFamily] = &[
    BaselineFamily::GruD, BaselineFamily::OdeRnn, BaselineFamily::LatentOde,
    BaselineFamily::CfcOnly, BaselineFamily::HdcLtc,
];
const T6_BASELINES: &[BaselineFamily] = &[
    BaselineFamily::KeyValueOverwrite, BaselineFamily::RecencyDecay,
    BaselineFamily::AssociativeMemory, BaselineFamily::HdcOnly,
    BaselineFamily::BitemporalLedger, BaselineFamily::HdcLtc,
];
const T7_BASELINES: &[BaselineFamily] = &[
    BaselineFamily::Gru, BaselineFamily::Lstm, BaselineFamily::Tcn,
    BaselineFamily::OdeRnn, BaselineFamily::CfcOnly, BaselineFamily::HdcLtc,
];
const T8_BASELINES: &[BaselineFamily] = &[
    BaselineFamily::Gru, BaselineFamily::Lstm, BaselineFamily::AdaptiveLif,
    BaselineFamily::BrainTraceDRtrl, BaselineFamily::BrainTracePpProp,
    BaselineFamily::EProp, BaselineFamily::HdcLtc,
];
const T9_BASELINES: &[BaselineFamily] = &[
    BaselineFamily::Bptt, BaselineFamily::AdaptiveLif, BaselineFamily::BrainTracePpProp,
    BaselineFamily::EProp, BaselineFamily::HdcLtc,
];
const T10_BASELINES: &[BaselineFamily] = &[
    BaselineFamily::KeyValueOverwrite, BaselineFamily::RecencyDecay,
    BaselineFamily::BitemporalLedger, BaselineFamily::Rag, BaselineFamily::HdcLtc,
];

pub fn temporal_task_definition(task_id: TemporalTaskId) -> TemporalTaskDefinition {
    match task_id {
        TemporalTaskId::T1SparseEvent => TemporalTaskDefinition {
            task_id, short_name: "T1-sparse-event", capability: "sparse event detection, timing and idle false alarms",
            metrics: T1_METRICS, required_baselines: T1_BASELINES,
        },
        TemporalTaskId::T2DelayedCueRecall => TemporalTaskDefinition {
            task_id, short_name: "T2-delayed-cue", capability: "associative recall by delay and distractor similarity",
            metrics: T2_METRICS, required_baselines: T2_BASELINES,
        },
        TemporalTaskId::T3MultiTimescale => TemporalTaskDefinition {
            task_id, short_name: "T3-multiscale", capability: "joint fast, medium and slow state estimation/forecasting",
            metrics: T3_METRICS, required_baselines: T3_BASELINES,
        },
        TemporalTaskId::T4OnlineDrift => TemporalTaskDefinition {
            task_id, short_name: "T4-online-drift", capability: "causal change detection, adaptation and prior-regime retention",
            metrics: T4_METRICS, required_baselines: T4_BASELINES,
        },
        TemporalTaskId::T5IrregularMissing => TemporalTaskDefinition {
            task_id, short_name: "T5-irregular-missing", capability: "irregular-time updates and missing-observation robustness",
            metrics: T5_METRICS, required_baselines: T5_BASELINES,
        },
        TemporalTaskId::T6RetentionInvalidation => TemporalTaskDefinition {
            task_id, short_name: "T6-retention-invalidation", capability: "selective retention, interference resistance and invalidation",
            metrics: T6_METRICS, required_baselines: T6_BASELINES,
        },
        TemporalTaskId::T7PhasePrediction => TemporalTaskDefinition {
            task_id, short_name: "T7-phase-prediction", capability: "phase/frequency tracking and future crossing prediction",
            metrics: T7_METRICS, required_baselines: T7_BASELINES,
        },
        TemporalTaskId::T8DelayedMatchSample => TemporalTaskDefinition {
            task_id, short_name: "T8-delayed-match", capability: "delayed match-to-sample with temporal credit assignment",
            metrics: T8_METRICS, required_baselines: T8_BASELINES,
        },
        TemporalTaskId::T9EvidenceAccumulation => TemporalTaskDefinition {
            task_id, short_name: "T9-evidence-accumulation", capability: "sequential evidence accumulation and delayed decision",
            metrics: T9_METRICS, required_baselines: T9_BASELINES,
        },
        TemporalTaskId::T10BitemporalKnowledge => TemporalTaskDefinition {
            task_id, short_name: "T10-bitemporal-knowledge", capability: "valid-time/known-time correctness, correction and retraction",
            metrics: T10_METRICS, required_baselines: T10_BASELINES,
        },
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SeedAssignment {
    pub task_id: TemporalTaskId,
    /// Stable source-family identifier. All seeds from one source family must stay in one split.
    pub source_family_id: u64,
    pub seed: u64,
    pub split: SplitRole,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateArtifact {
    pub candidate_id: String,
    pub family: BaselineFamily,
    pub artifact_sha256: [u8; 32],
    pub configuration_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HardwareTier {
    McuScalar,
    McuDsp,
    McuVector,
    IntegratedSignalProcessor,
    Npu,
    HeterogeneousCpuNpu,
    Neuromorphic,
    Other,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SystemTarget {
    pub target_id: String,
    pub board_revision: String,
    pub cpu_core: String,
    pub accelerator_profile: String,
    pub hardware_tier: HardwareTier,
    pub sram_budget_bytes: u64,
    pub protected_sram_reserve_bytes: u64,
    pub flash_budget_bytes: u64,
    pub p95_deadline_us: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalBenchmarkManifest {
    pub schema: String,
    pub version: String,
    /// The temporal-bench source commit from which task generators were built.
    pub temporal_suite_commit: String,
    pub track: BenchmarkTrack,
    /// All ten preregistered tasks are explicit and sorted by T1..T10.
    pub tasks: Vec<TemporalTaskId>,
    /// One pinned implementation per candidate ID; IDs are lexically sorted.
    pub candidates: Vec<CandidateArtifact>,
    pub seed_assignments: Vec<SeedAssignment>,
    /// Present for System track and absent for Algorithm track.
    pub system_target: Option<SystemTarget>,
    pub manifest_sha256: [u8; 32],
}

fn hash_text(hasher: &mut CanonicalHasher, value: &str) {
    hasher.bytes(value.as_bytes());
}

fn hash_task(hasher: &mut CanonicalHasher, task: TemporalTaskId) {
    hasher.u8(task as u8);
}

fn hash_split(hasher: &mut CanonicalHasher, split: SplitRole) {
    hasher.u8(split as u8);
}

fn hash_baseline(hasher: &mut CanonicalHasher, baseline: BaselineFamily) {
    hasher.u8(baseline as u8);
}

fn hash_target(hasher: &mut CanonicalHasher, target: &SystemTarget) {
    hash_text(hasher, &target.target_id);
    hash_text(hasher, &target.board_revision);
    hash_text(hasher, &target.cpu_core);
    hash_text(hasher, &target.accelerator_profile);
    hasher.u8(target.hardware_tier as u8);
    hasher.u64(target.sram_budget_bytes);
    hasher.u64(target.protected_sram_reserve_bytes);
    hasher.u64(target.flash_budget_bytes);
    hasher.u64(target.p95_deadline_us);
}

pub fn compute_temporal_benchmark_manifest_sha256(
    manifest: &TemporalBenchmarkManifest,
) -> [u8; 32] {
    let mut hasher = CanonicalHasher::new(b"symthaea.temporal.edge-adapter-manifest.v0.1");
    hash_text(&mut hasher, &manifest.schema);
    hash_text(&mut hasher, &manifest.version);
    hash_text(&mut hasher, &manifest.temporal_suite_commit);
    hasher.u8(manifest.track as u8);
    hasher.usize(manifest.tasks.len());
    for task in &manifest.tasks {
        hash_task(&mut hasher, *task);
    }
    hasher.usize(manifest.candidates.len());
    for candidate in &manifest.candidates {
        hash_text(&mut hasher, &candidate.candidate_id);
        hash_baseline(&mut hasher, candidate.family);
        hasher.bytes(&candidate.artifact_sha256);
        hasher.bytes(&candidate.configuration_sha256);
    }
    hasher.usize(manifest.seed_assignments.len());
    for assignment in &manifest.seed_assignments {
        hash_task(&mut hasher, assignment.task_id);
        hasher.u64(assignment.source_family_id);
        hasher.u64(assignment.seed);
        hash_split(&mut hasher, assignment.split);
    }
    match &manifest.system_target {
        Some(target) => {
            hasher.u8(1);
            hash_target(&mut hasher, target);
        }
        None => hasher.u8(0),
    }
    hasher.finish()
}

pub fn validate_temporal_benchmark_manifest(
    manifest: &TemporalBenchmarkManifest,
) -> Result<(), EdgeAdapterError> {
    if manifest.schema != MANIFEST_SCHEMA || manifest.version != VERSION {
        return Err(EdgeAdapterError::new("schema_version", "unsupported manifest schema/version"));
    }
    if manifest.temporal_suite_commit.len() != 40
        || !manifest.temporal_suite_commit.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(EdgeAdapterError::new("temporal_suite_commit", "must be a lowercase 40-character Git commit SHA"));
    }
    if manifest.tasks != TemporalTaskId::ALL {
        return Err(EdgeAdapterError::new("tasks", "must contain the complete T1..T10 suite in canonical order"));
    }
    if manifest.candidates.is_empty()
        || manifest.candidates.windows(2).any(|pair| pair[0].candidate_id >= pair[1].candidate_id)
    {
        return Err(EdgeAdapterError::new("candidates", "must be non-empty and sorted by unique candidate_id"));
    }
    let mut candidate_ids = HashSet::with_capacity(manifest.candidates.len());
    let mut candidate_families = BTreeSet::new();
    for (index, candidate) in manifest.candidates.iter().enumerate() {
        if candidate.candidate_id.trim().is_empty()
            || !candidate_ids.insert(candidate.candidate_id.as_str())
            || !digest_is_nonzero(&candidate.artifact_sha256)
            || !digest_is_nonzero(&candidate.configuration_sha256)
        {
            return Err(EdgeAdapterError::new(format!("candidates[{index}]"), "candidate identity or artifact/configuration digest is invalid"));
        }
        candidate_families.insert(candidate.family);
    }
    for task in &manifest.tasks {
        for required in temporal_task_definition(*task).required_baselines {
            if !candidate_families.contains(required) {
                return Err(EdgeAdapterError::new(
                    format!("candidates.required_baseline.{task:?}"),
                    format!("required baseline family {required:?} is missing"),
                ));
            }
        }
    }

    if manifest.seed_assignments.is_empty() {
        return Err(EdgeAdapterError::new("seed_assignments", "must not be empty"));
    }
    let mut assignment_keys = HashSet::with_capacity(manifest.seed_assignments.len());
    let mut source_family_splits = HashMap::<(TemporalTaskId, u64), SplitRole>::new();
    let mut splits_by_task = HashMap::<TemporalTaskId, BTreeSet<SplitRole>>::new();
    let mut previous = None::<(TemporalTaskId, u64, u64, SplitRole)>;
    for (index, assignment) in manifest.seed_assignments.iter().enumerate() {
        if !manifest.tasks.contains(&assignment.task_id) {
            return Err(EdgeAdapterError::new(format!("seed_assignments[{index}].task_id"), "task is not in the frozen suite"));
        }
        let key = (assignment.task_id, assignment.source_family_id, assignment.seed, assignment.split);
        if !assignment_keys.insert(key) {
            return Err(EdgeAdapterError::new(format!("seed_assignments[{index}]"), "duplicate seed assignment"));
        }
        if let Some(split) = source_family_splits.get(&(assignment.task_id, assignment.source_family_id)) {
            if *split != assignment.split {
                return Err(EdgeAdapterError::new(
                    format!("seed_assignments[{index}].source_family_id"),
                    "one source family cannot be reused across train/development/calibration/confirmatory splits",
                ));
            }
        } else {
            source_family_splits.insert((assignment.task_id, assignment.source_family_id), assignment.split);
        }
        splits_by_task.entry(assignment.task_id).or_default().insert(assignment.split);
        if let Some(previous_key) = previous {
            if key < previous_key {
                return Err(EdgeAdapterError::new(
                    format!("seed_assignments[{index}]"),
                    "assignments must be sorted by task, source family, seed and split",
                ));
            }
        }
        previous = Some(key);
    }
    for task in &manifest.tasks {
        let assigned = splits_by_task.get(task).ok_or_else(|| {
            EdgeAdapterError::new(format!("seed_assignments.{task:?}"), "task has no seed assignments")
        })?;
        for split in SplitRole::ALL {
            if !assigned.contains(&split) {
                return Err(EdgeAdapterError::new(
                    format!("seed_assignments.{task:?}.{split:?}"),
                    "each task requires training, development, calibration and confirmatory source families",
                ));
            }
        }
    }

    match (&manifest.track, &manifest.system_target) {
        (BenchmarkTrack::Algorithm, None) => {}
        (BenchmarkTrack::Algorithm, Some(_)) => {
            return Err(EdgeAdapterError::new("system_target", "Algorithm track must not claim a hardware target"));
        }
        (BenchmarkTrack::System, Some(target)) => {
            if target.target_id.trim().is_empty()
                || target.board_revision.trim().is_empty()
                || target.cpu_core.trim().is_empty()
                || target.accelerator_profile.trim().is_empty()
                || target.sram_budget_bytes == 0
                || target.protected_sram_reserve_bytes >= target.sram_budget_bytes
                || target.flash_budget_bytes == 0
                || target.p95_deadline_us == 0
            {
                return Err(EdgeAdapterError::new("system_target", "invalid system identity, memory budget, reserve or p95 deadline"));
            }
        }
        (BenchmarkTrack::System, None) => {
            return Err(EdgeAdapterError::new("system_target", "System track requires an explicit target contract"));
        }
    }

    if compute_temporal_benchmark_manifest_sha256(manifest) != manifest.manifest_sha256 {
        return Err(EdgeAdapterError::new("manifest_sha256", "does not match canonical manifest"));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateOutputReceipt {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub candidate_id: String,
    pub task_id: TemporalTaskId,
    pub split: SplitRole,
    pub source_family_id: u64,
    pub seed: u64,
    /// Digest of the public input only. The candidate receipt intentionally contains no oracle digest or labels.
    pub public_task_sha256: [u8; 32],
    pub candidate_artifact_sha256: [u8; 32],
    pub candidate_configuration_sha256: [u8; 32],
    pub prediction_output_sha256: [u8; 32],
    pub action_trace_sha256: [u8; 32],
    pub model_state_trace_sha256: [u8; 32],
    pub persistent_state_bytes: u64,
    pub peak_working_set_bytes: u64,
    pub memory_read_bytes: u64,
    pub memory_write_bytes: u64,
    pub candidate_receipt_sha256: [u8; 32],
}

pub fn compute_candidate_output_receipt_sha256(receipt: &CandidateOutputReceipt) -> [u8; 32] {
    let mut hasher = CanonicalHasher::new(b"symthaea.temporal.edge-candidate-output.v0.1");
    hash_text(&mut hasher, &receipt.schema);
    hash_text(&mut hasher, &receipt.version);
    hash_text(&mut hasher, &receipt.run_id);
    hash_text(&mut hasher, &receipt.candidate_id);
    hash_task(&mut hasher, receipt.task_id);
    hash_split(&mut hasher, receipt.split);
    hasher.u64(receipt.source_family_id);
    hasher.u64(receipt.seed);
    hasher.bytes(&receipt.public_task_sha256);
    hasher.bytes(&receipt.candidate_artifact_sha256);
    hasher.bytes(&receipt.candidate_configuration_sha256);
    hasher.bytes(&receipt.prediction_output_sha256);
    hasher.bytes(&receipt.action_trace_sha256);
    hasher.bytes(&receipt.model_state_trace_sha256);
    hasher.u64(receipt.persistent_state_bytes);
    hasher.u64(receipt.peak_working_set_bytes);
    hasher.u64(receipt.memory_read_bytes);
    hasher.u64(receipt.memory_write_bytes);
    hasher.finish()
}

pub fn validate_candidate_output_receipt(
    manifest: &TemporalBenchmarkManifest,
    receipt: &CandidateOutputReceipt,
) -> Result<(), EdgeAdapterError> {
    validate_temporal_benchmark_manifest(manifest)?;
    if receipt.schema != CANDIDATE_SCHEMA || receipt.version != VERSION || receipt.run_id.trim().is_empty() {
        return Err(EdgeAdapterError::new("candidate_receipt.schema_version", "unsupported schema/version or empty run ID"));
    }
    let candidate = manifest.candidates.iter().find(|candidate| candidate.candidate_id == receipt.candidate_id)
        .ok_or_else(|| EdgeAdapterError::new("candidate_receipt.candidate_id", "candidate is not in the frozen manifest"))?;
    if candidate.artifact_sha256 != receipt.candidate_artifact_sha256
        || candidate.configuration_sha256 != receipt.candidate_configuration_sha256
    {
        return Err(EdgeAdapterError::new("candidate_receipt.artifact", "artifact/configuration differs from the frozen candidate manifest"));
    }
    if !manifest.seed_assignments.iter().any(|assignment| {
        assignment.task_id == receipt.task_id
            && assignment.source_family_id == receipt.source_family_id
            && assignment.seed == receipt.seed
            && assignment.split == receipt.split
    }) {
        return Err(EdgeAdapterError::new("candidate_receipt.split", "task/source-family/seed/split tuple is not preregistered"));
    }
    if !digest_is_nonzero(&receipt.public_task_sha256)
        || !digest_is_nonzero(&receipt.prediction_output_sha256)
        || !digest_is_nonzero(&receipt.action_trace_sha256)
        || !digest_is_nonzero(&receipt.model_state_trace_sha256)
        || receipt.persistent_state_bytes > receipt.peak_working_set_bytes
    {
        return Err(EdgeAdapterError::new("candidate_receipt.evidence", "invalid input/output/trace digest or state/working-set byte accounting"));
    }
    if compute_candidate_output_receipt_sha256(receipt) != receipt.candidate_receipt_sha256 {
        return Err(EdgeAdapterError::new("candidate_receipt_sha256", "does not match canonical candidate receipt"));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetricObservation {
    pub metric_id: String,
    /// Examples: "all", "delay_64_high_similarity", "regime_3", "valid_at_10_known_at_20".
    pub stratum_id: String,
    /// Integer value in the exact unit/scale defined by the task metric contract.
    pub value_scaled: i64,
    pub sample_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluatorResultReceipt {
    pub schema: String,
    pub version: String,
    pub candidate_receipt_sha256: [u8; 32],
    /// Private evaluator commitment. This digest never appears in CandidateOutputReceipt.
    pub oracle_sha256: [u8; 32],
    pub evaluator_sha256: [u8; 32],
    pub metrics: Vec<MetricObservation>,
    pub evaluator_receipt_sha256: [u8; 32],
}

pub fn compute_evaluator_result_receipt_sha256(receipt: &EvaluatorResultReceipt) -> [u8; 32] {
    let mut hasher = CanonicalHasher::new(b"symthaea.temporal.edge-evaluator-receipt.v0.1");
    hash_text(&mut hasher, &receipt.schema);
    hash_text(&mut hasher, &receipt.version);
    hasher.bytes(&receipt.candidate_receipt_sha256);
    hasher.bytes(&receipt.oracle_sha256);
    hasher.bytes(&receipt.evaluator_sha256);
    hasher.usize(receipt.metrics.len());
    for metric in &receipt.metrics {
        hash_text(&mut hasher, &metric.metric_id);
        hash_text(&mut hasher, &metric.stratum_id);
        hasher.u64(metric.value_scaled as u64);
        hasher.u64(metric.sample_count);
    }
    hasher.finish()
}

pub fn validate_evaluator_result_receipt(
    candidate: &CandidateOutputReceipt,
    receipt: &EvaluatorResultReceipt,
) -> Result<(), EdgeAdapterError> {
    if receipt.schema != EVALUATOR_SCHEMA || receipt.version != VERSION {
        return Err(EdgeAdapterError::new("evaluator_receipt.schema_version", "unsupported schema/version"));
    }
    if receipt.candidate_receipt_sha256 != candidate.candidate_receipt_sha256
        || !digest_is_nonzero(&receipt.oracle_sha256)
        || !digest_is_nonzero(&receipt.evaluator_sha256)
    {
        return Err(EdgeAdapterError::new("evaluator_receipt.binding", "candidate, oracle or evaluator digest is invalid"));
    }
    let definition = temporal_task_definition(candidate.task_id);
    if receipt.metrics.is_empty()
        || receipt.metrics.windows(2).any(|pair| {
            (&pair[0].metric_id, &pair[0].stratum_id) >= (&pair[1].metric_id, &pair[1].stratum_id)
        })
    {
        return Err(EdgeAdapterError::new("evaluator_receipt.metrics", "metrics must be non-empty, unique and sorted by metric ID/stratum"));
    }
    let allowed: HashSet<&str> = definition.metrics.iter().map(|metric| metric.metric_id).collect();
    let mut present = HashSet::<&str>::new();
    for (index, metric) in receipt.metrics.iter().enumerate() {
        if metric.metric_id.trim().is_empty()
            || metric.stratum_id.trim().is_empty()
            || metric.sample_count == 0
            || metric.value_scaled < 0
            || !allowed.contains(metric.metric_id.as_str())
        {
            return Err(EdgeAdapterError::new(format!("evaluator_receipt.metrics[{index}]"), "metric is unknown, empty, negative or has no sample denominator"));
        }
        present.insert(metric.metric_id.as_str());
    }
    for definition in definition.metrics {
        if !present.contains(definition.metric_id) {
            return Err(EdgeAdapterError::new(
                format!("evaluator_receipt.metrics.{}", definition.metric_id),
                "required task metric is missing",
            ));
        }
    }
    if compute_evaluator_result_receipt_sha256(receipt) != receipt.evaluator_receipt_sha256 {
        return Err(EdgeAdapterError::new("evaluator_receipt_sha256", "does not match canonical evaluator receipt"));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SystemMeasurementReceipt {
    pub schema: String,
    pub version: String,
    pub candidate_receipt_sha256: [u8; 32],
    pub evaluator_receipt_sha256: [u8; 32],
    pub target_id: String,
    pub board_revision: String,
    pub cpu_core: String,
    pub accelerator_profile: String,
    pub hardware_tier: HardwareTier,
    pub compiler_sha256: [u8; 32],
    pub firmware_sha256: [u8; 32],
    pub linker_map_sha256: [u8; 32],
    pub measurement_protocol_sha256: [u8; 32],
    pub device_session_sha256: [u8; 32],
    pub latency_samples_us: Vec<u64>,
    pub p50_latency_us: u64,
    pub p95_latency_us: u64,
    pub p99_latency_us: u64,
    pub peak_sram_bytes: u64,
    pub flash_bytes: u64,
    /// Complete application-event measurand, not model-only inference energy.
    pub mean_energy_per_successful_event_uj: f64,
    pub expanded_energy_uncertainty_uj: f64,
    pub energy_uncertainty_receipt_sha256: [u8; 32],
    pub raw_power_trace_sha256: [u8; 32],
    pub system_receipt_sha256: [u8; 32],
}

pub fn nearest_rank_percentile_u64(
    samples: &[u64],
    numerator: usize,
    denominator: usize,
) -> Result<u64, EdgeAdapterError> {
    if samples.is_empty() || denominator == 0 || numerator == 0 || numerator > denominator {
        return Err(EdgeAdapterError::new("latency.percentile", "invalid samples or percentile"));
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let scaled = numerator.checked_mul(sorted.len()).ok_or_else(|| {
        EdgeAdapterError::new("latency.percentile", "rank multiplication overflow")
    })?;
    let rounded = scaled.checked_add(denominator - 1).ok_or_else(|| {
        EdgeAdapterError::new("latency.percentile", "rank rounding overflow")
    })?;
    let rank = rounded / denominator;
    Ok(sorted[rank.saturating_sub(1).min(sorted.len() - 1)])
}

pub fn compute_system_measurement_receipt_sha256(receipt: &SystemMeasurementReceipt) -> [u8; 32] {
    let mut hasher = CanonicalHasher::new(b"symthaea.temporal.edge-system-receipt.v0.1");
    hash_text(&mut hasher, &receipt.schema);
    hash_text(&mut hasher, &receipt.version);
    hasher.bytes(&receipt.candidate_receipt_sha256);
    hasher.bytes(&receipt.evaluator_receipt_sha256);
    hash_text(&mut hasher, &receipt.target_id);
    hash_text(&mut hasher, &receipt.board_revision);
    hash_text(&mut hasher, &receipt.cpu_core);
    hash_text(&mut hasher, &receipt.accelerator_profile);
    hasher.u8(receipt.hardware_tier as u8);
    hasher.bytes(&receipt.compiler_sha256);
    hasher.bytes(&receipt.firmware_sha256);
    hasher.bytes(&receipt.linker_map_sha256);
    hasher.bytes(&receipt.measurement_protocol_sha256);
    hasher.bytes(&receipt.device_session_sha256);
    hasher.usize(receipt.latency_samples_us.len());
    for sample in &receipt.latency_samples_us {
        hasher.u64(*sample);
    }
    hasher.u64(receipt.p50_latency_us);
    hasher.u64(receipt.p95_latency_us);
    hasher.u64(receipt.p99_latency_us);
    hasher.u64(receipt.peak_sram_bytes);
    hasher.u64(receipt.flash_bytes);
    hasher.u64(receipt.mean_energy_per_successful_event_uj.to_bits());
    hasher.u64(receipt.expanded_energy_uncertainty_uj.to_bits());
    hasher.bytes(&receipt.energy_uncertainty_receipt_sha256);
    hasher.bytes(&receipt.raw_power_trace_sha256);
    hasher.finish()
}

pub fn validate_system_measurement_receipt(
    manifest: &TemporalBenchmarkManifest,
    candidate: &CandidateOutputReceipt,
    evaluator: &EvaluatorResultReceipt,
    receipt: &SystemMeasurementReceipt,
) -> Result<(), EdgeAdapterError> {
    validate_temporal_benchmark_manifest(manifest)?;
    validate_candidate_output_receipt(manifest, candidate)?;
    validate_evaluator_result_receipt(candidate, evaluator)?;
    if manifest.track != BenchmarkTrack::System {
        return Err(EdgeAdapterError::new("system_track", "system receipt is not accepted on the Algorithm track"));
    }
    let target = manifest.system_target.as_ref().ok_or_else(|| {
        EdgeAdapterError::new("system_target", "missing target contract")
    })?;
    if receipt.schema != SYSTEM_SCHEMA || receipt.version != VERSION {
        return Err(EdgeAdapterError::new("system_receipt.schema_version", "unsupported schema/version"));
    }
    if receipt.candidate_receipt_sha256 != candidate.candidate_receipt_sha256
        || receipt.evaluator_receipt_sha256 != evaluator.evaluator_receipt_sha256
    {
        return Err(EdgeAdapterError::new("system_receipt.binding", "candidate or evaluator receipt digest mismatch"));
    }
    if receipt.target_id != target.target_id
        || receipt.board_revision != target.board_revision
        || receipt.cpu_core != target.cpu_core
        || receipt.accelerator_profile != target.accelerator_profile
        || receipt.hardware_tier != target.hardware_tier
    {
        return Err(EdgeAdapterError::new("system_receipt.target", "hardware identity/tier differs from the frozen target"));
    }
    for (field, value) in [
        ("compiler_sha256", &receipt.compiler_sha256),
        ("firmware_sha256", &receipt.firmware_sha256),
        ("linker_map_sha256", &receipt.linker_map_sha256),
        ("measurement_protocol_sha256", &receipt.measurement_protocol_sha256),
        ("device_session_sha256", &receipt.device_session_sha256),
        ("energy_uncertainty_receipt_sha256", &receipt.energy_uncertainty_receipt_sha256),
        ("raw_power_trace_sha256", &receipt.raw_power_trace_sha256),
        ("system_receipt_sha256", &receipt.system_receipt_sha256),
    ] {
        if !digest_is_nonzero(value) {
            return Err(EdgeAdapterError::new(format!("system_receipt.{field}"), "digest must be non-zero"));
        }
    }
    if receipt.latency_samples_us.is_empty()
        || receipt.latency_samples_us.iter().any(|sample| *sample == 0)
        || receipt.p50_latency_us != nearest_rank_percentile_u64(&receipt.latency_samples_us, 50, 100)?
        || receipt.p95_latency_us != nearest_rank_percentile_u64(&receipt.latency_samples_us, 95, 100)?
        || receipt.p99_latency_us != nearest_rank_percentile_u64(&receipt.latency_samples_us, 99, 100)?
    {
        return Err(EdgeAdapterError::new("system_receipt.latency", "raw latency sample/percentile mismatch"));
    }
    let available_sram = target.sram_budget_bytes.checked_sub(target.protected_sram_reserve_bytes)
        .ok_or_else(|| EdgeAdapterError::new("system_target.sram_budget", "protected reserve exceeds total SRAM"))?;
    if receipt.peak_sram_bytes == 0
        || receipt.flash_bytes == 0
        || receipt.peak_sram_bytes > available_sram
        || receipt.flash_bytes > target.flash_budget_bytes
        || receipt.p95_latency_us > target.p95_deadline_us
    {
        return Err(EdgeAdapterError::new("system_receipt.resource_budget", "SRAM, flash or p95 latency exceeds the frozen system target"));
    }
    if !receipt.mean_energy_per_successful_event_uj.is_finite()
        || receipt.mean_energy_per_successful_event_uj <= 0.0
        || !receipt.expanded_energy_uncertainty_uj.is_finite()
        || receipt.expanded_energy_uncertainty_uj < 0.0
    {
        return Err(EdgeAdapterError::new("system_receipt.energy", "complete-event energy or expanded uncertainty is invalid"));
    }
    if compute_system_measurement_receipt_sha256(receipt) != receipt.system_receipt_sha256 {
        return Err(EdgeAdapterError::new("system_receipt_sha256", "does not match canonical system receipt"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn candidate_list() -> Vec<CandidateArtifact> {
        let families = [
            BaselineFamily::ThresholdEma, BaselineFamily::StaticTinyMl, BaselineFamily::AdaptiveTinyMl,
            BaselineFamily::Gru, BaselineFamily::Lstm, BaselineFamily::GruD, BaselineFamily::Tcn,
            BaselineFamily::OdeRnn, BaselineFamily::LatentOde, BaselineFamily::AdaptiveLif,
            BaselineFamily::BrainTraceDRtrl, BaselineFamily::BrainTracePpProp, BaselineFamily::EProp,
            BaselineFamily::Stdl, BaselineFamily::Bptt, BaselineFamily::HdcOnly, BaselineFamily::CfcOnly,
            BaselineFamily::HdcLtc, BaselineFamily::HdcLtcConfidenceGated, BaselineFamily::AssociativeMemory,
            BaselineFamily::KeyValueOverwrite, BaselineFamily::RecencyDecay, BaselineFamily::BitemporalLedger,
            BaselineFamily::Rag,
        ];
        let mut candidates: Vec<_> = families.iter().enumerate().map(|(index, family)| {
            CandidateArtifact {
                candidate_id: format!("baseline-{index:02}"),
                family: *family,
                artifact_sha256: hash(index as u8 + 1),
                configuration_sha256: hash(index as u8 + 40),
            }
        }).collect();
        candidates.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
        candidates
    }

    fn assignments() -> Vec<SeedAssignment> {
        let mut values = Vec::new();
        for task in TemporalTaskId::ALL {
            for split in SplitRole::ALL {
                let split_id = split as u64;
                let family_id = (task as u64) * 100 + split_id + 1;
                values.push(SeedAssignment {
                    task_id: task,
                    source_family_id: family_id,
                    seed: 10_000 + task as u64 * 10 + split_id,
                    split,
                });
            }
        }
        values.sort_by_key(|item| (item.task_id, item.source_family_id, item.seed, item.split));
        values
    }

    fn target() -> SystemTarget {
        SystemTarget {
            target_id: "fixture-mcu".into(),
            board_revision: "rev-a".into(),
            cpu_core: "cortex-m55".into(),
            accelerator_profile: "cpu-only-npu-disabled".into(),
            hardware_tier: HardwareTier::McuVector,
            sram_budget_bytes: 512 * 1024,
            protected_sram_reserve_bytes: 64 * 1024,
            flash_budget_bytes: 2 * 1024 * 1024,
            p95_deadline_us: 100_000,
        }
    }

    fn manifest(track: BenchmarkTrack) -> TemporalBenchmarkManifest {
        let mut value = TemporalBenchmarkManifest {
            schema: MANIFEST_SCHEMA.into(),
            version: VERSION.into(),
            temporal_suite_commit: "158e642d747300d0945c114729579fa6e0391cda".into(),
            track,
            tasks: TemporalTaskId::ALL.to_vec(),
            candidates: candidate_list(),
            seed_assignments: assignments(),
            system_target: if track == BenchmarkTrack::System { Some(target()) } else { None },
            manifest_sha256: [0; 32],
        };
        value.manifest_sha256 = compute_temporal_benchmark_manifest_sha256(&value);
        value
    }

    fn candidate_receipt(manifest: &TemporalBenchmarkManifest) -> CandidateOutputReceipt {
        let artifact = manifest.candidates.iter().find(|candidate| candidate.family == BaselineFamily::HdcLtc).unwrap();
        let assignment = manifest.seed_assignments.iter().find(|assignment| {
            assignment.task_id == TemporalTaskId::T2DelayedCueRecall && assignment.split == SplitRole::Confirmatory
        }).unwrap();
        let mut receipt = CandidateOutputReceipt {
            schema: CANDIDATE_SCHEMA.into(),
            version: VERSION.into(),
            run_id: "edge-temporal-run-1".into(),
            candidate_id: artifact.candidate_id.clone(),
            task_id: assignment.task_id,
            split: assignment.split,
            source_family_id: assignment.source_family_id,
            seed: assignment.seed,
            public_task_sha256: hash(11),
            candidate_artifact_sha256: artifact.artifact_sha256,
            candidate_configuration_sha256: artifact.configuration_sha256,
            prediction_output_sha256: hash(12),
            action_trace_sha256: hash(13),
            model_state_trace_sha256: hash(14),
            persistent_state_bytes: 4_096,
            peak_working_set_bytes: 8_192,
            memory_read_bytes: 50_000,
            memory_write_bytes: 2_000,
            candidate_receipt_sha256: [0; 32],
        };
        receipt.candidate_receipt_sha256 = compute_candidate_output_receipt_sha256(&receipt);
        receipt
    }

    fn evaluator_receipt(candidate: &CandidateOutputReceipt) -> EvaluatorResultReceipt {
        let mut receipt = EvaluatorResultReceipt {
            schema: EVALUATOR_SCHEMA.into(),
            version: VERSION.into(),
            candidate_receipt_sha256: candidate.candidate_receipt_sha256,
            oracle_sha256: hash(15),
            evaluator_sha256: hash(16),
            metrics: vec![
                MetricObservation { metric_id: "mean_recall_latency_us".into(), stratum_id: "all".into(), value_scaled: 10_000, sample_count: 100 },
                MetricObservation { metric_id: "recall_accuracy_ppm".into(), stratum_id: "all".into(), value_scaled: 950_000, sample_count: 100 },
                MetricObservation { metric_id: "worst_delay_accuracy_ppm".into(), stratum_id: "all".into(), value_scaled: 900_000, sample_count: 20 },
            ],
            evaluator_receipt_sha256: [0; 32],
        };
        receipt.metrics.sort_by(|left, right| {
            (&left.metric_id, &left.stratum_id).cmp(&(&right.metric_id, &right.stratum_id))
        });
        receipt.evaluator_receipt_sha256 = compute_evaluator_result_receipt_sha256(&receipt);
        receipt
    }

    #[test]
    fn manifest_requires_all_task_baselines_and_source_family_disjoint_splits() {
        let manifest = manifest(BenchmarkTrack::Algorithm);
        validate_temporal_benchmark_manifest(&manifest).unwrap();

        let mut leaked = manifest.clone();
        let confirmatory = leaked.seed_assignments.iter_mut().find(|assignment| {
            assignment.task_id == TemporalTaskId::T4OnlineDrift && assignment.split == SplitRole::Confirmatory
        }).unwrap();
        confirmatory.source_family_id = leaked.seed_assignments.iter().find(|assignment| {
            assignment.task_id == TemporalTaskId::T4OnlineDrift && assignment.split == SplitRole::Training
        }).unwrap().source_family_id;
        leaked.manifest_sha256 = compute_temporal_benchmark_manifest_sha256(&leaked);
        assert!(validate_temporal_benchmark_manifest(&leaked).is_err());
    }

    #[test]
    fn candidate_receipt_has_no_oracle_field_and_binds_public_input_only() {
        let manifest = manifest(BenchmarkTrack::Algorithm);
        let receipt = candidate_receipt(&manifest);
        validate_candidate_output_receipt(&manifest, &receipt).unwrap();

        let mut tampered = receipt.clone();
        tampered.public_task_sha256 = hash(99);
        assert!(validate_candidate_output_receipt(&manifest, &tampered).is_err());
    }

    #[test]
    fn evaluator_receipt_is_separate_and_requires_all_task_metrics() {
        let manifest = manifest(BenchmarkTrack::Algorithm);
        let candidate = candidate_receipt(&manifest);
        let evaluator = evaluator_receipt(&candidate);
        validate_evaluator_result_receipt(&candidate, &evaluator).unwrap();

        let mut missing_metric = evaluator.clone();
        missing_metric.metrics.pop();
        missing_metric.evaluator_receipt_sha256 = compute_evaluator_result_receipt_sha256(&missing_metric);
        assert!(validate_evaluator_result_receipt(&candidate, &missing_metric).is_err());
    }

    #[test]
    fn system_receipt_recomputes_latency_and_enforces_hardware_budgets() {
        let manifest = manifest(BenchmarkTrack::System);
        let candidate = candidate_receipt(&manifest);
        let evaluator = evaluator_receipt(&candidate);
        let target = manifest.system_target.as_ref().unwrap();
        let mut receipt = SystemMeasurementReceipt {
            schema: SYSTEM_SCHEMA.into(),
            version: VERSION.into(),
            candidate_receipt_sha256: candidate.candidate_receipt_sha256,
            evaluator_receipt_sha256: evaluator.evaluator_receipt_sha256,
            target_id: target.target_id.clone(),
            board_revision: target.board_revision.clone(),
            cpu_core: target.cpu_core.clone(),
            accelerator_profile: target.accelerator_profile.clone(),
            hardware_tier: target.hardware_tier,
            compiler_sha256: hash(21),
            firmware_sha256: hash(22),
            linker_map_sha256: hash(23),
            measurement_protocol_sha256: hash(24),
            device_session_sha256: hash(25),
            latency_samples_us: vec![1_000, 2_000, 3_000, 4_000, 5_000],
            p50_latency_us: 3_000,
            p95_latency_us: 5_000,
            p99_latency_us: 5_000,
            peak_sram_bytes: 128 * 1024,
            flash_bytes: 512 * 1024,
            mean_energy_per_successful_event_uj: 25.0,
            expanded_energy_uncertainty_uj: 1.0,
            energy_uncertainty_receipt_sha256: hash(26),
            raw_power_trace_sha256: hash(27),
            system_receipt_sha256: [0; 32],
        };
        receipt.system_receipt_sha256 = compute_system_measurement_receipt_sha256(&receipt);
        validate_system_measurement_receipt(&manifest, &candidate, &evaluator, &receipt).unwrap();

        receipt.peak_sram_bytes = target.sram_budget_bytes;
        receipt.system_receipt_sha256 = compute_system_measurement_receipt_sha256(&receipt);
        assert!(validate_system_measurement_receipt(&manifest, &candidate, &evaluator, &receipt).is_err());
    }
}
