//! Workload-regime descriptors and family-held-out evaluation utilities.
//!
//! A regime profile describes the workload, not the architecture. It preserves
//! operating conditions such as temporal horizon, event density, working set,
//! memory movement, update cadence and precision. Optional measurements remain
//! absent when unknown; missing values are never interpreted as zero.
use std::collections::{HashMap, HashSet};

use crate::architecture_evidence::{
    ArchitectureEvidenceError, ArchitectureEvidenceRecord,
    validate_architecture_evidence_record,
    validate_architecture_evidence_record_for_comparison,
};
use crate::architecture_portfolio::{MemoryInterval, MetricInterval};

/// How the workload is presented to the compute system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadExecutionMode {
    /// A finite batch is submitted and completed as one job.
    Batch,
    /// Inputs arrive continuously or incrementally and are processed online.
    Streaming,
    /// Workload combines bounded batches with a continuous or incremental path.
    Hybrid,
}

/// Dominant temporal structure of the workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadTemporalPattern {
    /// No meaningful temporal dependency; samples can be treated independently.
    Static,
    /// Repeating or phase-locked temporal structure.
    Periodic,
    /// Persistent input or state evolution over a declared time horizon.
    Continuous,
    /// Irregular activity with concentrated bursts.
    Bursty,
    /// Asynchronous events with variable inter-arrival times.
    EventDriven,
    /// Non-periodic temporal structure with irregular timing.
    Irregular,
}

/// Numeric semantics that the candidate implementation must preserve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadPrecisionProfile {
    /// Exact integer, bitwise or other explicitly exact arithmetic.
    Exact,
    /// Fixed-point arithmetic with details in precision_contract_id.
    FixedPoint,
    /// IEEE binary16 or compatible half-precision format.
    Float16,
    /// bfloat16 or compatible brain floating-point format.
    BFloat16,
    /// IEEE binary32 or compatible single-precision format.
    Float32,
    /// IEEE binary64 or compatible double-precision format.
    Float64,
    /// Multiple numeric formats with a separately declared mixed-precision contract.
    Mixed,
}

/// Origin of the workload-regime characterization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegimeCharacterizationOrigin {
    /// Declared by the workload specification or benchmark contract.
    DeclaredWorkload,
    /// Calculated from a preserved raw input/event trace.
    MeasuredInputTrace,
    /// Calculated from a named frozen dataset or corpus.
    DatasetDerived,
    /// Transcribed from an external paper or published benchmark description.
    PublishedDescription,
}

/// Communication fabric used at a named source/destination boundary.
///
/// This is independent of the compute architecture family. Optical
/// chip-to-chip communication is not optical matrix computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationFabricFamily {
    /// On-chip network-on-chip or local digital interconnect.
    OnChipNetwork,
    /// DDR/LPDDR memory channel.
    DdrMemoryChannel,
    /// High-bandwidth memory channel or stack interface.
    HbmMemoryChannel,
    /// Peripheral Component Interconnect Express.
    Pcie,
    /// Compute Express Link.
    Cxl,
    /// Ethernet-family networking.
    Ethernet,
    /// InfiniBand-family networking.
    InfiniBand,
    /// Optical or co-packaged chip-to-chip interconnect.
    OpticalChipToChip,
    /// Event-address or asynchronous neuromorphic routing network.
    EventAddressNetwork,
    /// A separately identified communication fabric.
    Other,
}

/// Inclusive interval for a count such as batch size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountInterval {
    /// Lower count, inclusive.
    pub lower: u64,
    /// Upper count, inclusive.
    pub upper: u64,
}

impl CountInterval {
    /// Construct a nonzero, ordered count interval.
    pub fn new(lower: u64, upper: u64) -> Result<Self, WorkloadRegimeError> {
        if lower == 0 || lower > upper {
            return Err(WorkloadRegimeError::InvalidCountInterval);
        }
        Ok(Self { lower, upper })
    }

    fn is_valid(self) -> bool {
        self.lower > 0 && self.lower <= self.upper
    }
}

/// Measured or bounded traffic at one named memory or communication boundary.
///
/// Read and write byte counts are per the enclosing profile's unit_of_work.
/// Do not add two rows that describe the same bytes crossing the same boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct CommunicationBoundaryProfile {
    /// Stable boundary identifier, e.g. host-to-device PCIe or HBM-to-SRAM.
    pub boundary_id: String,
    /// Communication fabric used for this boundary, if known.
    pub fabric: Option<CommunicationFabricFamily>,
    /// Bytes read across this boundary per unit of work, if measured.
    pub read_bytes_per_unit: Option<MemoryInterval>,
    /// Bytes written across this boundary per unit of work, if measured.
    pub written_bytes_per_unit: Option<MemoryInterval>,
    /// Sustained boundary bandwidth in bytes per second, if measured.
    pub bandwidth_bytes_per_second: Option<MetricInterval>,
    /// One-way or round-trip latency in seconds, defined by boundary_id.
    pub latency_seconds: Option<MetricInterval>,
    /// Link energy in joules per bit, only when directly measured.
    pub energy_joules_per_bit: Option<MetricInterval>,
}

/// Source-bound description of one workload's operating regime.
///
/// It must bind to a concrete observation, exact workload, workload family,
/// holdout group and comparison contract. Regime metadata is not architecture
/// performance and must not be used as a substitute for measured outcomes.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureWorkloadRegime {
    /// Stable identifier for this regime profile.
    pub regime_profile_id: String,
    /// Observation this profile describes.
    pub observation_record_id: String,
    /// Exact workload fingerprint from the bound observation.
    pub workload_fingerprint: String,
    /// Stable workload family identity used for grouped holdout splits.
    pub workload_family_id: String,
    /// Source dataset/task group that must not cross evaluation partitions.
    pub holdout_group_id: String,
    /// Exact comparison-contract fingerprint from the bound observation.
    pub comparison_contract_fingerprint: String,
    /// Where the regime characterization came from.
    pub origin: RegimeCharacterizationOrigin,
    /// DOI, URL, repository path or immutable source locator.
    pub source_reference: String,
    /// SHA-256 of the source trace, dataset, task definition or publication.
    pub source_artifact_sha256: String,
    /// SHA-256 of the extraction procedure for trace- or dataset-derived fields.
    pub extraction_procedure_sha256: Option<String>,
    /// Batch, streaming or hybrid input mode.
    pub execution_mode: WorkloadExecutionMode,
    /// Dominant temporal structure.
    pub temporal_pattern: WorkloadTemporalPattern,
    /// Declared unit for every per-unit traffic and batch measurement.
    pub unit_of_work: String,
    /// Mean event or sample arrival rate in events/s, if applicable.
    pub event_rate_hz: Option<MetricInterval>,
    /// Peak sliding-window rate divided by full-trace mean event rate.
    pub burst_peak_to_mean_ratio: Option<MetricInterval>,
    /// Window duration used to compute burst_peak_to_mean_ratio, in seconds.
    pub burst_window_seconds: Option<MetricInterval>,
    /// Fraction of units/coordinates active during the declared observation window.
    pub active_fraction: Option<MetricInterval>,
    /// Longest task-relevant temporal dependency horizon, in seconds.
    pub temporal_horizon_seconds: Option<MetricInterval>,
    /// Required resident working set, in bytes.
    pub working_set_bytes: Option<MemoryInterval>,
    /// State update or adaptation frequency, in updates per second.
    pub state_update_rate_hz: Option<MetricInterval>,
    /// Maximum allowed end-to-end response deadline, in seconds.
    pub deadline_seconds: Option<MetricInterval>,
    /// Batch cardinality distribution/range. Required for Batch mode.
    pub batch_size: Option<CountInterval>,
    /// Numeric semantics of the task/accepted answer.
    pub precision_profile: WorkloadPrecisionProfile,
    /// Versioned numeric tolerance, quantization or exactness contract identity.
    pub precision_contract_id: String,
    /// Bytes, latency, bandwidth and optional energy at each distinct link/tier.
    pub communication_boundaries: Vec<CommunicationBoundaryProfile>,
}

/// Evidence observation paired with its workload regime descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct RegimeBoundEvidenceRecord {
    /// Provenance-validated architecture measurement.
    pub evidence: ArchitectureEvidenceRecord,
    /// Workload regime described by the evidence record.
    pub regime: ArchitectureWorkloadRegime,
}

/// Error raised by malformed or inconsistently bound workload-regime profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadRegimeError {
    /// An identity, source locator, unit or precision contract is empty.
    EmptyField,
    /// Profile refers to a different observation.
    RecordIdMismatch,
    /// Profile refers to a different workload fingerprint.
    WorkloadMismatch,
    /// Profile refers to a different comparison contract.
    ContractMismatch,
    /// Workload family appears in more than one evaluation partition.
    WorkloadFamilyLeakage,
    /// Holdout dataset/source group appears in more than one partition.
    HoldoutGroupLeakage,
    /// A regime query contains an invalid interval or domain value.
    InvalidQuery,
    /// A required digest is not exactly 64 hexadecimal characters.
    InvalidDigest,
    /// Trace/dataset origin has no extraction procedure digest.
    MissingExtractionProcedure,
    /// A numeric interval is malformed or has an invalid domain.
    InvalidMetricInterval,
    /// A byte interval is inverted.
    InvalidMemoryInterval,
    /// A count interval is zero or inverted.
    InvalidCountInterval,
    /// Burst ratio and its measurement window must be present together.
    IncompleteBurstinessDefinition,
    /// Burst peak-to-mean ratio must be at least one.
    InvalidBurstinessRatio,
    /// Batch execution requires an explicit nonzero batch-size interval.
    MissingBatchSize,
    /// A communication boundary has no traffic, latency, bandwidth or energy field.
    EmptyCommunicationBoundary,
    /// Communication boundary identifiers must be unique within a profile.
    DuplicateCommunicationBoundary,
    /// A measured bandwidth or latency must be strictly positive.
    InvalidCommunicationMetric,
    /// Provenance validation failed for the underlying architecture record.
    Evidence(ArchitectureEvidenceError),
}

/// Return true when a source digest has exactly 64 hexadecimal characters.
fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Check an optional non-negative interval.
fn valid_non_negative(interval: Option<MetricInterval>) -> bool {
    interval.is_none_or(|value| {
        value.lower.is_finite() && value.upper.is_finite()
            && value.lower >= 0.0 && value.lower <= value.upper
    })
}

/// Check an optional strictly-positive interval.
fn valid_positive(interval: Option<MetricInterval>) -> bool {
    interval.is_none_or(|value| {
        value.lower.is_finite() && value.upper.is_finite()
            && value.lower > 0.0 && value.lower <= value.upper
    })
}

/// Validate one regime profile independently of its bound evidence record.
pub fn validate_workload_regime_profile(
    profile: &ArchitectureWorkloadRegime,
) -> Result<(), WorkloadRegimeError> {
    for value in [
        profile.regime_profile_id.as_str(),
        profile.observation_record_id.as_str(),
        profile.workload_fingerprint.as_str(),
        profile.workload_family_id.as_str(),
        profile.holdout_group_id.as_str(),
        profile.comparison_contract_fingerprint.as_str(),
        profile.source_reference.as_str(),
        profile.unit_of_work.as_str(),
        profile.precision_contract_id.as_str(),
    ] {
        if value.trim().is_empty() {
            return Err(WorkloadRegimeError::EmptyField);
        }
    }

    if !is_sha256(&profile.source_artifact_sha256) {
        return Err(WorkloadRegimeError::InvalidDigest);
    }
    match profile.origin {
        RegimeCharacterizationOrigin::MeasuredInputTrace
        | RegimeCharacterizationOrigin::DatasetDerived => {
            let Some(digest) = profile.extraction_procedure_sha256.as_deref() else {
                return Err(WorkloadRegimeError::MissingExtractionProcedure);
            };
            if !is_sha256(digest) {
                return Err(WorkloadRegimeError::InvalidDigest);
            }
        }
        RegimeCharacterizationOrigin::DeclaredWorkload
        | RegimeCharacterizationOrigin::PublishedDescription => {
            if profile.extraction_procedure_sha256
                .as_deref()
                .is_some_and(|digest| !is_sha256(digest))
            {
                return Err(WorkloadRegimeError::InvalidDigest);
            }
        }
    }

    if !valid_non_negative(profile.event_rate_hz)
        || !valid_non_negative(profile.state_update_rate_hz)
        || !valid_non_negative(profile.active_fraction)
    {
        return Err(WorkloadRegimeError::InvalidMetricInterval);
    }
    if profile.active_fraction.is_some_and(|value| value.upper > 1.0) {
        return Err(WorkloadRegimeError::InvalidMetricInterval);
    }
    if !valid_positive(profile.temporal_horizon_seconds)
        || !valid_positive(profile.deadline_seconds)
    {
        return Err(WorkloadRegimeError::InvalidMetricInterval);
    }

    match (profile.burst_peak_to_mean_ratio, profile.burst_window_seconds) {
        (Some(ratio), Some(window)) => {
            if !valid_non_negative(Some(ratio)) || ratio.lower < 1.0 {
                return Err(WorkloadRegimeError::InvalidBurstinessRatio);
            }
            if !valid_positive(Some(window)) {
                return Err(WorkloadRegimeError::InvalidMetricInterval);
            }
            if profile.event_rate_hz.is_none_or(|rate| rate.upper <= 0.0) {
                return Err(WorkloadRegimeError::InvalidBurstinessRatio);
            }
        }
        (None, None) => {}
        _ => return Err(WorkloadRegimeError::IncompleteBurstinessDefinition),
    }

    if profile
        .working_set_bytes
        .is_some_and(|value| value.lower_bytes > value.upper_bytes)
    {
        return Err(WorkloadRegimeError::InvalidMemoryInterval);
    }
    if profile.execution_mode == WorkloadExecutionMode::Batch
        && profile.batch_size.is_none()
    {
        return Err(WorkloadRegimeError::MissingBatchSize);
    }
    if profile.batch_size.is_some_and(|value| !value.is_valid()) {
        return Err(WorkloadRegimeError::InvalidCountInterval);
    }

    let mut boundary_ids = HashSet::new();
    for boundary in &profile.communication_boundaries {
        if boundary.boundary_id.trim().is_empty() {
            return Err(WorkloadRegimeError::EmptyField);
        }
        if !boundary_ids.insert(boundary.boundary_id.as_str()) {
            return Err(WorkloadRegimeError::DuplicateCommunicationBoundary);
        }
        if boundary.read_bytes_per_unit.is_none()
            && boundary.written_bytes_per_unit.is_none()
            && boundary.bandwidth_bytes_per_second.is_none()
            && boundary.latency_seconds.is_none()
            && boundary.energy_joules_per_bit.is_none()
        {
            return Err(WorkloadRegimeError::EmptyCommunicationBoundary);
        }
        for value in [
            boundary.read_bytes_per_unit,
            boundary.written_bytes_per_unit,
        ]
        .into_iter()
        .flatten()
        {
            if value.lower_bytes > value.upper_bytes {
                return Err(WorkloadRegimeError::InvalidMemoryInterval);
            }
        }
        if !valid_positive(boundary.bandwidth_bytes_per_second)
            || !valid_positive(boundary.latency_seconds)
            || !valid_non_negative(boundary.energy_joules_per_bit)
        {
            return Err(WorkloadRegimeError::InvalidCommunicationMetric);
        }
    }

    Ok(())
}

/// Validate that a workload regime is bound to its evidence observation exactly.
pub fn validate_regime_bound_evidence_record(
    record: &RegimeBoundEvidenceRecord,
) -> Result<(), WorkloadRegimeError> {
    validate_architecture_evidence_record(&record.evidence)
        .map_err(WorkloadRegimeError::Evidence)?;
    validate_workload_regime_profile(&record.regime)?;
    let observation = &record.evidence.observation;
    if record.regime.observation_record_id != observation.record_id {
        return Err(WorkloadRegimeError::RecordIdMismatch);
    }
    if record.regime.workload_fingerprint != observation.workload_fingerprint {
        return Err(WorkloadRegimeError::WorkloadMismatch);
    }
    if record.regime.comparison_contract_fingerprint
        != observation.comparison_contract_fingerprint
    {
        return Err(WorkloadRegimeError::ContractMismatch);
    }
    Ok(())
}

/// Validate a regime-bound record for system-level comparison or routing.
pub fn validate_regime_bound_evidence_record_for_comparison(
    record: &RegimeBoundEvidenceRecord,
) -> Result<(), WorkloadRegimeError> {
    validate_regime_bound_evidence_record(record)?;
    validate_architecture_evidence_record_for_comparison(&record.evidence)
        .map_err(WorkloadRegimeError::Evidence)
}

/// A query over a workload operating region.
///
/// Numeric constraints match only when the stored interval overlaps the query
/// interval. If a requested field is unknown in the profile, the match fails;
/// unknown is never interpreted as zero or as an unconstrained pass.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkloadRegimeQuery {
    /// Required execution mode, when constrained.
    pub execution_mode: Option<WorkloadExecutionMode>,
    /// Required temporal pattern, when constrained.
    pub temporal_pattern: Option<WorkloadTemporalPattern>,
    /// Event-rate operating region in events/s.
    pub event_rate_hz: Option<MetricInterval>,
    /// Peak-to-mean event-rate operating region.
    pub burst_peak_to_mean_ratio: Option<MetricInterval>,
    /// Active fraction operating region in [0, 1].
    pub active_fraction: Option<MetricInterval>,
    /// Temporal horizon operating region in seconds.
    pub temporal_horizon_seconds: Option<MetricInterval>,
    /// Working-set operating region in bytes.
    pub working_set_bytes: Option<MemoryInterval>,
    /// State-update operating region in updates/s.
    pub state_update_rate_hz: Option<MetricInterval>,
    /// Deadline operating region in seconds.
    pub deadline_seconds: Option<MetricInterval>,
    /// Batch cardinality operating region.
    pub batch_size: Option<CountInterval>,
}

/// Return whether two valid metric intervals overlap inclusively.
fn intervals_overlap(left: MetricInterval, right: MetricInterval) -> bool {
    left.lower <= right.upper && right.lower <= left.upper
}

/// Return whether two valid memory intervals overlap inclusively.
fn memory_intervals_overlap(left: MemoryInterval, right: MemoryInterval) -> bool {
    left.lower_bytes <= right.upper_bytes && right.lower_bytes <= left.upper_bytes
}

/// Match one profile against a workload-regime slice query.
pub fn workload_regime_matches_query(
    profile: &ArchitectureWorkloadRegime,
    query: &WorkloadRegimeQuery,
) -> bool {
    if validate_workload_regime_query(query).is_err() {
        return false;
    }
    if query.execution_mode.is_some_and(|value| value != profile.execution_mode)
        || query.temporal_pattern.is_some_and(|value| value != profile.temporal_pattern)
    {
        return false;
    }
    macro_rules! metric_match {
        ($field:ident) => {
            if let Some(query_interval) = query.$field {
                if profile
                    .$field
                    .is_none_or(|profile_interval| {
                        !intervals_overlap(profile_interval, query_interval)
                    })
                {
                    return false;
                }
            }
        };
    }
    metric_match!(event_rate_hz);
    metric_match!(burst_peak_to_mean_ratio);
    metric_match!(active_fraction);
    metric_match!(temporal_horizon_seconds);
    metric_match!(state_update_rate_hz);
    metric_match!(deadline_seconds);

    if let Some(query_interval) = query.working_set_bytes {
        if profile
            .working_set_bytes
            .is_none_or(|profile_interval| {
                !memory_intervals_overlap(profile_interval, query_interval)
            })
        {
            return false;
        }
    }
    if let Some(query_interval) = query.batch_size {
        if profile
            .batch_size
            .is_none_or(|profile_interval| {
                profile_interval.lower > query_interval.upper
                    || query_interval.lower > profile_interval.upper
            })
        {
            return false;
        }
    }
    true
}

/// Evaluation partition for workload-family holdout discipline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureEvaluationPartition {
    /// Model development and training data.
    Development,
    /// Tuning/selection data, not used for final claims.
    Validation,
    /// One-shot sealed confirmatory data.
    Confirmatory,
    /// External final test data.
    Test,
}

/// One workload family and data group assigned to an evaluation partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkloadHoldoutAssignment {
    /// Stable workload family; must not cross partitions.
    pub workload_family_id: String,
    /// Stable dataset/fixture/source group; must not cross partitions.
    pub holdout_group_id: String,
    /// Evaluation partition.
    pub partition: ArchitectureEvaluationPartition,
}

/// Leakage or malformed-assignment error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadHoldoutError {
    /// A workload family appears in more than one partition.
    WorkloadFamilyLeakage,
    /// A holdout data group appears in more than one partition.
    HoldoutGroupLeakage,
    /// A family or group identifier is blank.
    EmptyIdentifier,
}

/// Validate every supplied filter bound before querying the atlas.
pub fn validate_workload_regime_query(
    query: &WorkloadRegimeQuery,
) -> Result<(), WorkloadRegimeError> {
    if !valid_non_negative(query.event_rate_hz)
        || !valid_non_negative(query.burst_peak_to_mean_ratio)
        || query
            .burst_peak_to_mean_ratio
            .is_some_and(|value| value.lower < 1.0)
        || !valid_non_negative(query.state_update_rate_hz)
        || !valid_non_negative(query.active_fraction)
    {
        return Err(WorkloadRegimeError::InvalidQuery);
    }
    if query.active_fraction.is_some_and(|value| value.upper > 1.0)
        || !valid_positive(query.temporal_horizon_seconds)
        || !valid_positive(query.deadline_seconds)
    {
        return Err(WorkloadRegimeError::InvalidQuery);
    }
    if query
        .working_set_bytes
        .is_some_and(|value| value.lower_bytes > value.upper_bytes)
        || query.batch_size.is_some_and(|value| !value.is_valid())
    {
        return Err(WorkloadRegimeError::InvalidQuery);
    }
    Ok(())
}

/// Validate that workload families and data groups are disjoint across partitions.
///
/// Multiple records from the same family/group may appear within one partition;
/// no family or group may appear in two partitions, including confirmatory/test.
pub fn validate_workload_holdout_assignments(
    assignments: &[WorkloadHoldoutAssignment],
) -> Result<(), WorkloadHoldoutError> {
    let mut family_partitions = HashMap::new();
    let mut group_partitions = HashMap::new();
    for assignment in assignments {
        if assignment.workload_family_id.trim().is_empty()
            || assignment.holdout_group_id.trim().is_empty()
        {
            return Err(WorkloadHoldoutError::EmptyIdentifier);
        }
        if family_partitions
            .insert(
                assignment.workload_family_id.as_str(),
                assignment.partition,
            )
            .is_some_and(|previous| previous != assignment.partition)
        {
            return Err(WorkloadHoldoutError::WorkloadFamilyLeakage);
        }
        if group_partitions
            .insert(
                assignment.holdout_group_id.as_str(),
                assignment.partition,
            )
            .is_some_and(|previous| previous != assignment.partition)
        {
            return Err(WorkloadHoldoutError::HoldoutGroupLeakage);
        }
    }
    Ok(())
}

/// Validate partition separation for an existing collection of regime-bound records.
pub fn validate_regime_record_holdout_separation(
    records: &[(&RegimeBoundEvidenceRecord, ArchitectureEvaluationPartition)],
) -> Result<(), WorkloadRegimeError> {
    for (record, _) in records {
        validate_regime_bound_evidence_record(record)?;
    }
    let assignments: Vec<WorkloadHoldoutAssignment> = records
        .iter()
        .map(|(record, partition)| WorkloadHoldoutAssignment {
            workload_family_id: record.regime.workload_family_id.clone(),
            holdout_group_id: record.regime.holdout_group_id.clone(),
            partition: *partition,
        })
        .collect();
    validate_workload_holdout_assignments(&assignments).map_err(|error| match error {
        WorkloadHoldoutError::EmptyIdentifier => WorkloadRegimeError::EmptyField,
        WorkloadHoldoutError::WorkloadFamilyLeakage => {
            WorkloadRegimeError::WorkloadFamilyLeakage
        }
        WorkloadHoldoutError::HoldoutGroupLeakage => {
            WorkloadRegimeError::HoldoutGroupLeakage
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture_evidence::{
        ArchitectureBenchmarkSuite, ArchitectureEvidenceOrigin, ArchitectureEvidenceReceipt,
    };
    use crate::architecture_portfolio::{
        ArchitectureEvidenceClass, ArchitectureMeasurementBoundary,
        ArchitectureResourceEnvelope, ArchitectureWorkloadKind, ComputeArchitectureFamily,
    };

    fn interval(lower: f64, upper: f64) -> MetricInterval {
        MetricInterval { lower, upper }
    }

    fn regime() -> ArchitectureWorkloadRegime {
        ArchitectureWorkloadRegime {
            regime_profile_id: "event-stream-regime-a".to_owned(),
            observation_record_id: "event-cpu-record".to_owned(),
            workload_fingerprint: "event-workload-sha256".to_owned(),
            workload_family_id: "event-perception-family".to_owned(),
            holdout_group_id: "event-dataset-session-a".to_owned(),
            comparison_contract_fingerprint: "event-contract-sha256".to_owned(),
            origin: RegimeCharacterizationOrigin::MeasuredInputTrace,
            source_reference: "https://example.invalid/synthetic-event-trace".to_owned(),
            source_artifact_sha256: "a".repeat(64),
            extraction_procedure_sha256: Some("b".repeat(64)),
            execution_mode: WorkloadExecutionMode::Streaming,
            temporal_pattern: WorkloadTemporalPattern::Bursty,
            unit_of_work: "one accepted event window".to_owned(),
            event_rate_hz: Some(interval(100.0, 1_000.0)),
            burst_peak_to_mean_ratio: Some(interval(2.0, 5.0)),
            burst_window_seconds: Some(interval(0.01, 0.02)),
            active_fraction: Some(interval(0.05, 0.4)),
            temporal_horizon_seconds: Some(interval(1.0, 4.0)),
            working_set_bytes: Some(MemoryInterval {
                lower_bytes: 4096,
                upper_bytes: 8192,
            }),
            state_update_rate_hz: Some(interval(50.0, 100.0)),
            deadline_seconds: Some(interval(0.001, 0.005)),
            batch_size: None,
            precision_profile: WorkloadPrecisionProfile::Exact,
            precision_contract_id: "exact-event-count-contract-v1".to_owned(),
            communication_boundaries: vec![
                CommunicationBoundaryProfile {
                    boundary_id: "sensor-to-host".to_owned(),
                    fabric: Some(CommunicationFabricFamily::Pcie),
                    read_bytes_per_unit: Some(MemoryInterval {
                        lower_bytes: 128,
                        upper_bytes: 256,
                    }),
                    written_bytes_per_unit: None,
                    bandwidth_bytes_per_second: Some(interval(1e8, 2e8)),
                    latency_seconds: Some(interval(1e-6, 2e-6)),
                    energy_joules_per_bit: None,
                },
                CommunicationBoundaryProfile {
                    boundary_id: "host-to-accelerator".to_owned(),
                    fabric: Some(CommunicationFabricFamily::OpticalChipToChip),
                    read_bytes_per_unit: Some(MemoryInterval {
                        lower_bytes: 64,
                        upper_bytes: 128,
                    }),
                    written_bytes_per_unit: None,
                    bandwidth_bytes_per_second: Some(interval(1e9, 2e9)),
                    latency_seconds: None,
                    energy_joules_per_bit: Some(interval(1e-13, 2e-13)),
                },
            ],
        }
    }

    fn evidence_record() -> ArchitectureEvidenceRecord {
        let observation = ArchitectureBenchmarkObservation {
            record_id: "event-cpu-record".to_owned(),
            architecture_id: "cpu-reference".to_owned(),
            architecture_family: ComputeArchitectureFamily::ClassicalCpu,
            workload_kind: ArchitectureWorkloadKind::SparseEventPerception,
            workload_fingerprint: "event-workload-sha256".to_owned(),
            comparison_contract_fingerprint: "event-contract-sha256".to_owned(),
            evidence_class: ArchitectureEvidenceClass::PhysicalPrototype,
            measurement_boundary: ArchitectureMeasurementBoundary::SensorToDecision,
            quality: interval(0.95, 0.97),
            resources: Some(ArchitectureResourceEnvelope {
                wall_time_seconds: interval(0.001, 0.002),
                energy_joules: interval(0.1, 0.2),
                monetary_cost_usd: interval(0.001, 0.002),
                peak_memory: MemoryInterval {
                    lower_bytes: 4096,
                    upper_bytes: 8192,
                },
            }),
        };
        let receipt = ArchitectureEvidenceReceipt {
            observation_record_id: observation.record_id.clone(),
            workload_fingerprint: observation.workload_fingerprint.clone(),
            comparison_contract_fingerprint: observation
                .comparison_contract_fingerprint
                .clone(),
            benchmark_suite: ArchitectureBenchmarkSuite::SymthaeaCanonical,
            source_reference: "https://example.invalid/synthetic-benchmark".to_owned(),
            benchmark_version: "v1.synthetic".to_owned(),
            workload_case_id: "event-task-synthetic".to_owned(),
            origin: ArchitectureEvidenceOrigin::DirectMeasurement,
            source_revision: Some("c".repeat(40)),
            source_artifact_sha256: "d".repeat(64),
            runner_artifact_sha256: Some("e".repeat(64)),
            fixture_sha256: "f".repeat(64),
            run_manifest_sha256: "1".repeat(64),
            environment_sha256: "2".repeat(64),
            raw_output_sha256: "3".repeat(64),
            interval_analysis_sha256: "4".repeat(64),
            summary_sha256: "5".repeat(64),
            executor_identity: "synthetic-device-lab".to_owned(),
            device_identity: Some("synthetic-accelerator".to_owned()),
            toolchain_identity: Some("synthetic-toolchain-1.0".to_owned()),
            repeat_count: Some(30),
            failed_run_count: Some(0),
            interval_method_id: Some("synthetic-bootstrap-v1".to_owned()),
            aggregation_method_id: Some("synthetic-median-v1".to_owned()),
        };
        ArchitectureEvidenceRecord {
            observation,
            receipt,
        }
    }

    fn bound_record() -> RegimeBoundEvidenceRecord {
        RegimeBoundEvidenceRecord {
            evidence: evidence_record(),
            regime: regime(),
        }
    }

    #[test]
    fn valid_measured_trace_profile_is_accepted() {
        assert_eq!(
            validate_workload_regime_profile(&regime()),
            Ok(())
        );
    }

    #[test]
    fn regime_must_bind_to_exact_observation_identity() {
        let mut record = bound_record();
        record.regime.observation_record_id = "other-record".to_owned();
        assert_eq!(
            validate_regime_bound_evidence_record(&record),
            Err(WorkloadRegimeError::RecordIdMismatch)
        );
    }

    #[test]
    fn regime_must_bind_to_exact_workload_and_contract() {
        let mut record = bound_record();
        record.regime.workload_fingerprint = "other-workload".to_owned();
        assert_eq!(
            validate_regime_bound_evidence_record(&record),
            Err(WorkloadRegimeError::WorkloadMismatch)
        );
        let mut record = bound_record();
        record.regime.comparison_contract_fingerprint = "other-contract".to_owned();
        assert_eq!(
            validate_regime_bound_evidence_record(&record),
            Err(WorkloadRegimeError::ContractMismatch)
        );
    }

    #[test]
    fn trace_derived_profile_requires_extraction_procedure_digest() {
        let mut profile = regime();
        profile.extraction_procedure_sha256 = None;
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::MissingExtractionProcedure)
        );
    }

    #[test]
    fn malformed_source_and_extraction_digests_are_rejected() {
        let mut profile = regime();
        profile.source_artifact_sha256 = "not-a-digest".to_owned();
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::InvalidDigest)
        );
        let mut profile = regime();
        profile.extraction_procedure_sha256 = Some("z".repeat(64));
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::InvalidDigest)
        );
    }

    #[test]
    fn unknown_metrics_remain_absent_and_are_not_zero() {
        let mut profile = regime();
        profile.event_rate_hz = None;
        profile.burst_peak_to_mean_ratio = None;
        profile.burst_window_seconds = None;
        assert_eq!(validate_workload_regime_profile(&profile), Ok(()));
        let query = WorkloadRegimeQuery {
            event_rate_hz: Some(interval(0.0, 100.0)),
            ..empty_query()
        };
        assert!(!workload_regime_matches_query(&profile, &query));
    }

    #[test]
    fn query_can_slice_event_burstiness() {
        let profile = regime();
        let query = WorkloadRegimeQuery {
            burst_peak_to_mean_ratio: Some(interval(4.0, 6.0)),
            ..empty_query()
        };
        assert!(workload_regime_matches_query(&profile, &query));
    }

    #[test]
    fn active_fraction_must_remain_within_zero_and_one() {
        let mut profile = regime();
        profile.active_fraction = Some(interval(0.0, 1.1));
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::InvalidMetricInterval)
        );
    }

    #[test]
    fn negative_event_rates_are_rejected() {
        let mut profile = regime();
        profile.event_rate_hz = Some(interval(-1.0, 2.0));
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::InvalidMetricInterval)
        );
    }

    #[test]
    fn burstiness_requires_ratio_and_window_together() {
        let mut profile = regime();
        profile.burst_window_seconds = None;
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::IncompleteBurstinessDefinition)
        );
        let mut profile = regime();
        profile.burst_peak_to_mean_ratio = None;
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::IncompleteBurstinessDefinition)
        );
    }

    #[test]
    fn burst_peak_to_mean_ratio_must_be_at_least_one() {
        let mut profile = regime();
        profile.burst_peak_to_mean_ratio = Some(interval(0.5, 2.0));
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::InvalidBurstinessRatio)
        );
    }

    #[test]
    fn burstiness_requires_a_nonzero_mean_event_rate() {
        let mut profile = regime();
        profile.event_rate_hz = Some(interval(0.0, 0.0));
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::InvalidBurstinessRatio)
        );
    }

    #[test]
    fn batch_mode_requires_explicit_nonzero_batch_size() {
        let mut profile = regime();
        profile.execution_mode = WorkloadExecutionMode::Batch;
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::MissingBatchSize)
        );
        let mut profile = regime();
        profile.execution_mode = WorkloadExecutionMode::Batch;
        profile.batch_size = Some(CountInterval { lower: 1, upper: 8 });
        assert_eq!(validate_workload_regime_profile(&profile), Ok(()));
    }

    #[test]
    fn invalid_memory_and_batch_intervals_are_rejected() {
        let mut profile = regime();
        profile.working_set_bytes = Some(MemoryInterval {
            lower_bytes: 10,
            upper_bytes: 9,
        });
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::InvalidMemoryInterval)
        );
        let mut profile = regime();
        profile.batch_size = Some(CountInterval { lower: 0, upper: 2 });
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::InvalidCountInterval)
        );
    }

    #[test]
    fn communication_boundaries_require_nonempty_unique_ids_and_metrics() {
        let mut profile = regime();
        profile.communication_boundaries[0].boundary_id = " ".to_owned();
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::EmptyField)
        );
        let mut profile = regime();
        profile.communication_boundaries[1].boundary_id =
            profile.communication_boundaries[0].boundary_id.clone();
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::DuplicateCommunicationBoundary)
        );
        let mut profile = regime();
        profile.communication_boundaries[0].read_bytes_per_unit = None;
        profile.communication_boundaries[0].bandwidth_bytes_per_second = None;
        profile.communication_boundaries[0].latency_seconds = None;
        profile.communication_boundaries[0].energy_joules_per_bit = None;
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::EmptyCommunicationBoundary)
        );
    }

    #[test]
    fn communication_bandwidth_and_latency_must_be_positive() {
        let mut profile = regime();
        profile.communication_boundaries[0].bandwidth_bytes_per_second =
            Some(interval(0.0, 1.0));
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::InvalidCommunicationMetric)
        );
        let mut profile = regime();
        profile.communication_boundaries[0].latency_seconds =
            Some(interval(-1.0, 1.0));
        assert_eq!(
            validate_workload_regime_profile(&profile),
            Err(WorkloadRegimeError::InvalidCommunicationMetric)
        );
    }

    #[test]
    fn query_uses_interval_overlap_and_exact_categorical_regime() {
        let profile = regime();
        let query = WorkloadRegimeQuery {
            execution_mode: Some(WorkloadExecutionMode::Streaming),
            temporal_pattern: Some(WorkloadTemporalPattern::Bursty),
            event_rate_hz: Some(interval(500.0, 1500.0)),
            ..empty_query()
        };
        assert!(workload_regime_matches_query(&profile, &query));
        let query = WorkloadRegimeQuery {
            temporal_pattern: Some(WorkloadTemporalPattern::Static),
            ..empty_query()
        };
        assert!(!workload_regime_matches_query(&profile, &query));
    }

    #[test]
    fn query_fails_closed_when_requested_attribute_is_unknown() {
        let mut profile = regime();
        profile.working_set_bytes = None;
        let query = WorkloadRegimeQuery {
            working_set_bytes: Some(MemoryInterval {
                lower_bytes: 1,
                upper_bytes: 100_000,
            }),
            ..empty_query()
        };
        assert!(!workload_regime_matches_query(&profile, &query));
    }

    #[test]
    fn query_matches_working_set_and_batch_interval_overlap() {
        let mut profile = regime();
        profile.execution_mode = WorkloadExecutionMode::Batch;
        profile.batch_size = Some(CountInterval { lower: 4, upper: 16 });
        let query = WorkloadRegimeQuery {
            batch_size: Some(CountInterval { lower: 8, upper: 32 }),
            working_set_bytes: Some(MemoryInterval {
                lower_bytes: 8000,
                upper_bytes: 9000,
            }),
            ..empty_query()
        };
        assert!(workload_regime_matches_query(&profile, &query));
    }

    #[test]
    fn invalid_regime_query_fails_closed() {
        let profile = regime();
        let query = WorkloadRegimeQuery {
            event_rate_hz: Some(interval(10.0, 1.0)),
            ..empty_query()
        };
        assert_eq!(
            validate_workload_regime_query(&query),
            Err(WorkloadRegimeError::InvalidQuery)
        );
        assert!(!workload_regime_matches_query(&profile, &query));
    }

    #[test]
    fn holdout_validator_rejects_workload_family_leakage() {
        let assignments = vec![
            assignment("hdc-family-a", "dataset-a", ArchitectureEvaluationPartition::Development),
            assignment("hdc-family-a", "dataset-b", ArchitectureEvaluationPartition::Confirmatory),
        ];
        assert_eq!(
            validate_workload_holdout_assignments(&assignments),
            Err(WorkloadHoldoutError::WorkloadFamilyLeakage)
        );
    }

    #[test]
    fn holdout_validator_rejects_holdout_group_leakage() {
        let assignments = vec![
            assignment(
                "hdc-family-a",
                "same-source-dataset",
                ArchitectureEvaluationPartition::Development,
            ),
            assignment(
                "hdc-family-b",
                "same-source-dataset",
                ArchitectureEvaluationPartition::Test,
            ),
        ];
        assert_eq!(
            validate_workload_holdout_assignments(&assignments),
            Err(WorkloadHoldoutError::HoldoutGroupLeakage)
        );
    }

    #[test]
    fn same_partition_repeats_are_allowed_but_identifiers_must_be_present() {
        let assignments = vec![
            assignment("hdc-family-a", "dataset-a", ArchitectureEvaluationPartition::Development),
            assignment("hdc-family-a", "dataset-a", ArchitectureEvaluationPartition::Development),
            assignment("hdc-family-b", "dataset-b", ArchitectureEvaluationPartition::Confirmatory),
        ];
        assert_eq!(
            validate_workload_holdout_assignments(&assignments),
            Ok(())
        );
        let invalid = vec![assignment("", "dataset", ArchitectureEvaluationPartition::Development)];
        assert_eq!(
            validate_workload_holdout_assignments(&invalid),
            Err(WorkloadHoldoutError::EmptyIdentifier)
        );
    }

    #[test]
    fn regime_comparison_requires_evidence_uncertainty_provenance() {
        let mut record = bound_record();
        record.evidence.receipt.interval_method_id = None;
        assert_eq!(
            validate_regime_bound_evidence_record(&record),
            Ok(())
        );
        assert_eq!(
            validate_regime_bound_evidence_record_for_comparison(&record),
            Err(WorkloadRegimeError::Evidence(
                ArchitectureEvidenceError::UnspecifiedIntervalMethod
            ))
        );
    }

    fn empty_query() -> WorkloadRegimeQuery {
        WorkloadRegimeQuery {
            execution_mode: None,
            temporal_pattern: None,
            event_rate_hz: None,
            burst_peak_to_mean_ratio: None,
            active_fraction: None,
            temporal_horizon_seconds: None,
            working_set_bytes: None,
            state_update_rate_hz: None,
            deadline_seconds: None,
            batch_size: None,
        }
    }

    fn assignment(
        family: &str,
        group: &str,
        partition: ArchitectureEvaluationPartition,
    ) -> WorkloadHoldoutAssignment {
        WorkloadHoldoutAssignment {
            workload_family_id: family.to_owned(),
            holdout_group_id: group.to_owned(),
            partition,
        }
    }
}
