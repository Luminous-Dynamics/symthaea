//! Source-bound provenance receipts for cross-paradigm architecture observations.
//!
//! Receipts bind an observation to benchmark identity, source artifacts, runner,
//! fixture, environment, raw output and interval analysis. A digest establishes
//! content identity only; it does not authenticate an actor, prove chain of
//! custody, or certify that a physical instrument measured the claimed quantity.
use crate::architecture_portfolio::{
    compare_architecture_pareto, validate_architecture_observation,
    ArchitectureBenchmarkObservation, ArchitectureComparisonError,
    ArchitectureEvidenceClass, ArchitectureParetoRelation,
};
use crate::architecture_routing::{
    select_architecture_candidate, ArchitectureRoutingError, ArchitectureRoutingReport,
    ArchitectureRoutingRequest,
};

/// External or canonical benchmark suite whose semantics the adapter preserves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureBenchmarkSuite {
    /// Frozen Symthaea-owned canonical task and fixture suite.
    SymthaeaCanonical,
    /// NeuroBench algorithm or deployed-system track.
    NeuroBench,
    /// MLPerf Tiny.
    MlPerfTiny,
    /// MLPerf Edge.
    MlPerfEdge,
    /// MLPerf Power.
    MlPerfPower,
    /// QOBLIB optimization benchmark.
    Qoblib,
    /// QUOPS quantum capability benchmark.
    Quops,
    /// QED-C application benchmark suite.
    QedC,
    /// IonQ application-level benchmark framework.
    IonqApplications,
    /// A separately identified benchmark suite.
    Other,
}

/// Where the reported observation originates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureEvidenceOrigin {
    /// A run executed by the submitting team on the identified target.
    DirectMeasurement,
    /// A result reported by a publication or third-party benchmark record.
    PublishedResult,
    /// An executed software, mathematical or hardware-behavior simulation.
    SimulationExecution,
    /// An analytical or algorithmic reference without an executed system.
    AnalyticalReference,
}

/// Source-bound receipt for one architecture portfolio observation.
///
/// All required digests are SHA-256 identities of the exact corresponding
/// artifacts. Optional identities remain unknown when the source does not
/// disclose them; an adapter must not invent toolchains or runner code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureEvidenceReceipt {
    /// Observation record this receipt qualifies.
    pub observation_record_id: String,
    /// Exact workload fingerprint from the observation.
    pub workload_fingerprint: String,
    /// Exact comparison-contract fingerprint from the observation.
    pub comparison_contract_fingerprint: String,
    /// Benchmark suite that owns the workload semantics.
    pub benchmark_suite: ArchitectureBenchmarkSuite,
    /// Public DOI, URL, repository path or immutable source-registry reference.
    pub source_reference: String,
    /// Exact suite/version or declared upstream result release.
    pub benchmark_version: String,
    /// Stable upstream workload/case identifier.
    pub workload_case_id: String,
    /// Evidence origin; distinct from the architecture evidence maturity.
    pub origin: ArchitectureEvidenceOrigin,
    /// Optional Git commit SHA for version-controlled source.
    pub source_revision: Option<String>,
    /// SHA-256 of the canonical source, paper, specification or reference artifact.
    pub source_artifact_sha256: String,
    /// SHA-256 of the runner or executable artifact, when available.
    pub runner_artifact_sha256: Option<String>,
    /// SHA-256 of the exact input fixture or dataset manifest.
    pub fixture_sha256: String,
    /// SHA-256 of the run manifest and all declared configuration parameters.
    pub run_manifest_sha256: String,
    /// SHA-256 of the environment/device/toolchain declaration.
    pub environment_sha256: String,
    /// SHA-256 of preserved raw output or its immutable bundle.
    pub raw_output_sha256: String,
    /// SHA-256 of the analysis artifact that produced the supplied intervals.
    pub interval_analysis_sha256: String,
    /// SHA-256 of the normalized observation/summary artifact.
    pub summary_sha256: String,
    /// Identity of the executing lab, organization, system or publishing source.
    pub executor_identity: String,
    /// Concrete physical target; required for prototype/production evidence.
    pub device_identity: Option<String>,
    /// Exact compiler, runtime, simulator or analysis toolchain when disclosed.
    pub toolchain_identity: Option<String>,
    /// Total attempts, if reported by the source or captured by the runner.
    pub repeat_count: Option<u64>,
    /// Failed attempts, if reported/captured; paired with repeat_count.
    pub failed_run_count: Option<u64>,
    /// Declared uncertainty/interval procedure; absent when the source does not report it.
    pub interval_method_id: Option<String>,
    /// Declared replicate aggregation procedure; absent when not reported.
    pub aggregation_method_id: Option<String>,
}

/// Complete evidence record admitted to the capability atlas.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureEvidenceRecord {
    /// Normalized architecture observation.
    pub observation: ArchitectureBenchmarkObservation,
    /// Provenance receipt bound to this exact observation.
    pub receipt: ArchitectureEvidenceReceipt,
}

/// Rejection reason for malformed or mismatched evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureEvidenceError {
    /// The observation failed its portfolio data contract.
    InvalidObservation(ArchitectureComparisonError),
    /// A required identifier or method field is blank.
    EmptyField,
    /// Receipt refers to another observation record.
    RecordIdMismatch,
    /// Receipt refers to another workload.
    WorkloadMismatch,
    /// Receipt refers to another comparison contract.
    ContractMismatch,
    /// A required SHA-256 digest is not exactly 64 hexadecimal characters.
    InvalidDigest,
    /// Optional Git source revision is not a 40- or 64-character hexadecimal SHA.
    InvalidSourceRevision,
    /// Repeat/failed counts are incomplete, zero, or internally inconsistent.
    InvalidRepeatCounts,
    /// A system-level comparison has no declared repeat-count provenance.
    UnspecifiedRunCounts,
    /// A system-level comparison has no declared interval-construction method.
    UnspecifiedIntervalMethod,
    /// A system-level comparison has no declared replicate aggregation method.
    UnspecifiedAggregationMethod,
    /// Evidence origin cannot support the observation's declared evidence class.
    IncompatibleEvidenceOrigin,
    /// Physical prototype/production evidence has no concrete device identity.
    MissingDeviceIdentity,
    /// Direct/simulated execution has no identified toolchain.
    MissingToolchainIdentity,
    /// Direct/simulated execution has no runner artifact digest.
    MissingRunnerArtifact,
    /// An evidence-bound selection failed its routing request contract.
    Routing(ArchitectureRoutingError),
}

/// Check whether a required string contains a non-whitespace value.
fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

/// SHA-256 digest syntax check. This checks shape, not cryptographic correctness.
fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Git SHA-1 or SHA-256 syntax check. This checks shape, not repository membership.
fn is_source_revision(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Validate an observation and its provenance receipt as a single atlas record.
pub fn validate_architecture_evidence_record(
    record: &ArchitectureEvidenceRecord,
) -> Result<(), ArchitectureEvidenceError> {
    validate_architecture_evidence_receipt(&record.observation, &record.receipt)
}

/// Validate exact record/workload/contract binding and evidence-origin compatibility.
pub fn validate_architecture_evidence_receipt(
    observation: &ArchitectureBenchmarkObservation,
    receipt: &ArchitectureEvidenceReceipt,
) -> Result<(), ArchitectureEvidenceError> {
    validate_architecture_observation(observation)
        .map_err(ArchitectureEvidenceError::InvalidObservation)?;
    if receipt.observation_record_id != observation.record_id {
        return Err(ArchitectureEvidenceError::RecordIdMismatch);
    }
    if receipt.workload_fingerprint != observation.workload_fingerprint {
        return Err(ArchitectureEvidenceError::WorkloadMismatch);
    }
    if receipt.comparison_contract_fingerprint
        != observation.comparison_contract_fingerprint
    {
        return Err(ArchitectureEvidenceError::ContractMismatch);
    }

    for value in [
        receipt.source_reference.as_str(),
        receipt.benchmark_version.as_str(),
        receipt.workload_case_id.as_str(),
        receipt.executor_identity.as_str(),
    ] {
        if !non_empty(value) {
            return Err(ArchitectureEvidenceError::EmptyField);
        }
    }

    let required_digests = [
        receipt.source_artifact_sha256.as_str(),
        receipt.fixture_sha256.as_str(),
        receipt.run_manifest_sha256.as_str(),
        receipt.environment_sha256.as_str(),
        receipt.raw_output_sha256.as_str(),
        receipt.interval_analysis_sha256.as_str(),
        receipt.summary_sha256.as_str(),
    ];
    if !required_digests.into_iter().all(is_sha256) {
        return Err(ArchitectureEvidenceError::InvalidDigest);
    }
    if let Some(runner_digest) = receipt.runner_artifact_sha256.as_deref() {
        if !is_sha256(runner_digest) {
            return Err(ArchitectureEvidenceError::InvalidDigest);
        }
    }
    if let Some(revision) = receipt.source_revision.as_deref() {
        if !is_source_revision(revision) {
            return Err(ArchitectureEvidenceError::InvalidSourceRevision);
        }
    }
    match (receipt.repeat_count, receipt.failed_run_count) {
        (Some(total), Some(failed)) if total > 0 && failed <= total => {}
        (None, None)
            if matches!(
                receipt.origin,
                ArchitectureEvidenceOrigin::PublishedResult
                    | ArchitectureEvidenceOrigin::AnalyticalReference
            ) => {}
        _ => return Err(ArchitectureEvidenceError::InvalidRepeatCounts),
    }
    if receipt.interval_method_id.as_deref().is_some_and(|value| !non_empty(value))
        || receipt.aggregation_method_id.as_deref().is_some_and(|value| !non_empty(value))
    {
        return Err(ArchitectureEvidenceError::EmptyField);
    }
    if receipt.device_identity.as_deref().is_some_and(|value| !non_empty(value))
        || receipt
            .toolchain_identity
            .as_deref()
            .is_some_and(|value| !non_empty(value))
    {
        return Err(ArchitectureEvidenceError::EmptyField);
    }

    match receipt.origin {
        ArchitectureEvidenceOrigin::DirectMeasurement => {
            if !matches!(
                observation.evidence_class,
                ArchitectureEvidenceClass::PhysicalPrototype
                    | ArchitectureEvidenceClass::ProductionSystem
            ) {
                return Err(ArchitectureEvidenceError::IncompatibleEvidenceOrigin);
            }
            if receipt.device_identity.is_none() {
                return Err(ArchitectureEvidenceError::MissingDeviceIdentity);
            }
            if receipt.toolchain_identity.is_none() {
                return Err(ArchitectureEvidenceError::MissingToolchainIdentity);
            }
            if receipt.runner_artifact_sha256.is_none() {
                return Err(ArchitectureEvidenceError::MissingRunnerArtifact);
            }
            if receipt.repeat_count.is_none() {
                return Err(ArchitectureEvidenceError::InvalidRepeatCounts);
            }
            if receipt.interval_method_id.is_none() {
                return Err(ArchitectureEvidenceError::UnspecifiedIntervalMethod);
            }
            if receipt.aggregation_method_id.is_none() {
                return Err(ArchitectureEvidenceError::UnspecifiedAggregationMethod);
            }
        }
        ArchitectureEvidenceOrigin::PublishedResult => {
            if matches!(
                observation.evidence_class,
                ArchitectureEvidenceClass::PhysicalPrototype
                    | ArchitectureEvidenceClass::ProductionSystem
            ) && receipt.device_identity.is_none()
            {
                return Err(ArchitectureEvidenceError::MissingDeviceIdentity);
            }
        }
        ArchitectureEvidenceOrigin::SimulationExecution => {
            if !matches!(
                observation.evidence_class,
                ArchitectureEvidenceClass::Simulation
                    | ArchitectureEvidenceClass::HardwareEmulation
            ) {
                return Err(ArchitectureEvidenceError::IncompatibleEvidenceOrigin);
            }
            if receipt.toolchain_identity.is_none() {
                return Err(ArchitectureEvidenceError::MissingToolchainIdentity);
            }
            if receipt.runner_artifact_sha256.is_none() {
                return Err(ArchitectureEvidenceError::MissingRunnerArtifact);
            }
            if receipt.repeat_count.is_none() {
                return Err(ArchitectureEvidenceError::InvalidRepeatCounts);
            }
            if receipt.interval_method_id.is_none() {
                return Err(ArchitectureEvidenceError::UnspecifiedIntervalMethod);
            }
            if receipt.aggregation_method_id.is_none() {
                return Err(ArchitectureEvidenceError::UnspecifiedAggregationMethod);
            }
        }
        ArchitectureEvidenceOrigin::AnalyticalReference => {
            if observation.evidence_class != ArchitectureEvidenceClass::AlgorithmicReference {
                return Err(ArchitectureEvidenceError::IncompatibleEvidenceOrigin);
            }
        }
    }

    Ok(())
}

/// Validate an evidence record for system-level comparison or routing.
///
/// A structurally valid published record may omit repeat counts or interval
/// methods when its source does not report them. Such a record is retained as
/// provenance, but cannot support system-level ranking until those inputs exist.
pub fn validate_architecture_evidence_record_for_comparison(
    record: &ArchitectureEvidenceRecord,
) -> Result<(), ArchitectureEvidenceError> {
    validate_architecture_evidence_record(record)?;
    if record.observation.measurement_boundary
        != crate::architecture_portfolio::ArchitectureMeasurementBoundary::AlgorithmOnly
    {
        if record.receipt.repeat_count.is_none() {
            return Err(ArchitectureEvidenceError::UnspecifiedRunCounts);
        }
        if record.receipt.interval_method_id.is_none() {
            return Err(ArchitectureEvidenceError::UnspecifiedIntervalMethod);
        }
        if record.receipt.aggregation_method_id.is_none() {
            return Err(ArchitectureEvidenceError::UnspecifiedAggregationMethod);
        }
    }
    Ok(())
}

/// Compare two architecture observations only after validating both source receipts.
pub fn compare_architecture_evidence_records(
    left: &ArchitectureEvidenceRecord,
    right: &ArchitectureEvidenceRecord,
) -> Result<ArchitectureParetoRelation, ArchitectureEvidenceError> {
    validate_architecture_evidence_record_for_comparison(left)?;
    validate_architecture_evidence_record_for_comparison(right)?;
    compare_architecture_pareto(&left.observation, &right.observation)
        .map_err(ArchitectureEvidenceError::InvalidObservation)
}

/// Select from a capability-atlas slice only after every receipt has been validated.
///
/// A malformed or unproven record rejects the complete slice rather than being
/// silently skipped. Candidate-level feasibility exclusions remain in the
/// returned routing report after the provenance layer succeeds.
pub fn select_architecture_candidate_from_evidence(
    request: &ArchitectureRoutingRequest,
    records: &[ArchitectureEvidenceRecord],
) -> Result<ArchitectureRoutingReport, ArchitectureEvidenceError> {
    for record in records {
        validate_architecture_evidence_record_for_comparison(record)?;
    }
    let observations: Vec<ArchitectureBenchmarkObservation> = records
        .iter()
        .map(|record| record.observation.clone())
        .collect();
    select_architecture_candidate(request, &observations)
        .map_err(ArchitectureEvidenceError::Routing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture_portfolio::{
        ArchitectureMeasurementBoundary, ArchitectureResourceEnvelope,
        ArchitectureWorkloadKind, ComputeArchitectureFamily, MemoryInterval, MetricInterval,
    };
    use crate::architecture_routing::{
        ArchitectureResourceLimits, ArchitectureRoutingPriority, ArchitectureRoutingRequest,
    };

    fn interval(lower: f64, upper: f64) -> MetricInterval {
        MetricInterval { lower, upper }
    }

    fn physical_observation(
        id: &str,
        family: ComputeArchitectureFamily,
        time: (f64, f64),
    ) -> ArchitectureBenchmarkObservation {
        ArchitectureBenchmarkObservation {
            record_id: id.to_owned(),
            architecture_id: format!("device-{id}"),
            architecture_family: family,
            workload_kind: ArchitectureWorkloadKind::HdcAlgebra,
            workload_fingerprint: "hdc-query-fixture-sha256".to_owned(),
            comparison_contract_fingerprint: "hdc-system-contract-sha256".to_owned(),
            evidence_class: ArchitectureEvidenceClass::PhysicalPrototype,
            measurement_boundary: ArchitectureMeasurementBoundary::ApplicationEndToEnd,
            quality: interval(0.95, 0.96),
            resources: Some(ArchitectureResourceEnvelope {
                wall_time_seconds: interval(time.0, time.1),
                energy_joules: interval(2.0, 3.0),
                monetary_cost_usd: interval(0.01, 0.02),
                peak_memory: MemoryInterval {
                    lower_bytes: 1024,
                    upper_bytes: 2048,
                },
            }),
        }
    }

    fn algorithmic_observation() -> ArchitectureBenchmarkObservation {
        ArchitectureBenchmarkObservation {
            record_id: "algorithm-reference".to_owned(),
            architecture_id: "reference-algorithm".to_owned(),
            architecture_family: ComputeArchitectureFamily::ClassicalCpu,
            workload_kind: ArchitectureWorkloadKind::HdcAlgebra,
            workload_fingerprint: "hdc-query-fixture-sha256".to_owned(),
            comparison_contract_fingerprint: "hdc-algorithm-contract-sha256".to_owned(),
            evidence_class: ArchitectureEvidenceClass::AlgorithmicReference,
            measurement_boundary: ArchitectureMeasurementBoundary::AlgorithmOnly,
            quality: interval(0.95, 0.96),
            resources: None,
        }
    }

    fn receipt(
        observation: &ArchitectureBenchmarkObservation,
        origin: ArchitectureEvidenceOrigin,
    ) -> ArchitectureEvidenceReceipt {
        ArchitectureEvidenceReceipt {
            observation_record_id: observation.record_id.clone(),
            workload_fingerprint: observation.workload_fingerprint.clone(),
            comparison_contract_fingerprint: observation.comparison_contract_fingerprint.clone(),
            benchmark_suite: ArchitectureBenchmarkSuite::SymthaeaCanonical,
            source_reference: "https://example.invalid/synthetic-source".to_owned(),
            benchmark_version: "v1.synthetic".to_owned(),
            workload_case_id: "hdc-retrieval-synthetic-01".to_owned(),
            origin,
            source_revision: Some("c".repeat(40)),
            source_artifact_sha256: "a".repeat(64),
            runner_artifact_sha256: Some("b".repeat(64)),
            fixture_sha256: "d".repeat(64),
            run_manifest_sha256: "e".repeat(64),
            environment_sha256: "f".repeat(64),
            raw_output_sha256: "1".repeat(64),
            interval_analysis_sha256: "2".repeat(64),
            summary_sha256: "3".repeat(64),
            executor_identity: "synthetic-test-executor".to_owned(),
            device_identity: Some("synthetic-test-device".to_owned()),
            toolchain_identity: Some("synthetic-toolchain-1.0".to_owned()),
            repeat_count: Some(10),
            failed_run_count: Some(0),
            interval_method_id: Some("synthetic-interval-method".to_owned()),
            aggregation_method_id: Some("synthetic-aggregation-method".to_owned()),
        }
    }

    fn record(
        observation: ArchitectureBenchmarkObservation,
        origin: ArchitectureEvidenceOrigin,
    ) -> ArchitectureEvidenceRecord {
        let receipt = receipt(&observation, origin);
        ArchitectureEvidenceRecord {
            observation,
            receipt,
        }
    }

    fn physical_receipt(
        observation: &ArchitectureBenchmarkObservation,
    ) -> ArchitectureEvidenceReceipt {
        receipt(observation, ArchitectureEvidenceOrigin::DirectMeasurement)
    }

    #[test]
    fn direct_physical_observation_requires_complete_source_receipt() {
        let observation = physical_observation(
            "cpu-baseline",
            ComputeArchitectureFamily::ClassicalCpu,
            (0.9, 1.0),
        );
        let evidence = record(
            observation,
            ArchitectureEvidenceOrigin::DirectMeasurement,
        );
        assert_eq!(validate_architecture_evidence_record(&evidence), Ok(()));
    }

    #[test]
    fn published_physical_result_is_separately_identified() {
        let observation = physical_observation(
            "published-neuromorphic",
            ComputeArchitectureFamily::NeuromorphicDigital,
            (0.4, 0.5),
        );
        let mut published = receipt(
            &observation,
            ArchitectureEvidenceOrigin::PublishedResult,
        );
        published.runner_artifact_sha256 = None;
        published.toolchain_identity = None;
        published.repeat_count = None;
        published.failed_run_count = None;
        published.interval_method_id = None;
        published.aggregation_method_id = None;
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &published),
            Ok(())
        );
    }

    #[test]
    fn published_result_with_unknown_run_statistics_is_retained_but_not_ranked() {
        let observation = physical_observation(
            "published-neuromorphic",
            ComputeArchitectureFamily::NeuromorphicDigital,
            (0.4, 0.5),
        );
        let mut published = receipt(
            &observation,
            ArchitectureEvidenceOrigin::PublishedResult,
        );
        published.runner_artifact_sha256 = None;
        published.toolchain_identity = None;
        published.repeat_count = None;
        published.failed_run_count = None;
        published.interval_method_id = None;
        published.aggregation_method_id = None;
        let record = ArchitectureEvidenceRecord {
            observation,
            receipt: published,
        };
        assert_eq!(validate_architecture_evidence_record(&record), Ok(()));
        assert_eq!(
            validate_architecture_evidence_record_for_comparison(&record),
            Err(ArchitectureEvidenceError::UnspecifiedRunCounts)
        );
    }

    #[test]
    fn run_count_fields_must_be_reported_as_a_pair() {
        let observation = physical_observation(
            "published-neuromorphic",
            ComputeArchitectureFamily::NeuromorphicDigital,
            (0.4, 0.5),
        );
        let mut published = receipt(
            &observation,
            ArchitectureEvidenceOrigin::PublishedResult,
        );
        published.repeat_count = Some(10);
        published.failed_run_count = None;
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &published),
            Err(ArchitectureEvidenceError::InvalidRepeatCounts)
        );
    }

    #[test]
    fn observation_record_identity_must_match_receipt() {
        let observation = physical_observation(
            "cpu-baseline",
            ComputeArchitectureFamily::ClassicalCpu,
            (0.9, 1.0),
        );
        let mut evidence = physical_receipt(&observation);
        evidence.observation_record_id = "another-record".to_owned();
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Err(ArchitectureEvidenceError::RecordIdMismatch)
        );
    }

    #[test]
    fn workload_and_contract_fingerprints_must_match() {
        let observation = physical_observation(
            "cpu-baseline",
            ComputeArchitectureFamily::ClassicalCpu,
            (0.9, 1.0),
        );
        let mut evidence = physical_receipt(&observation);
        evidence.workload_fingerprint = "different-workload".to_owned();
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Err(ArchitectureEvidenceError::WorkloadMismatch)
        );
        let mut evidence = physical_receipt(&observation);
        evidence.comparison_contract_fingerprint = "different-contract".to_owned();
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Err(ArchitectureEvidenceError::ContractMismatch)
        );
    }

    #[test]
    fn malformed_digest_is_rejected() {
        let observation = physical_observation(
            "cpu-baseline",
            ComputeArchitectureFamily::ClassicalCpu,
            (0.9, 1.0),
        );
        let mut evidence = physical_receipt(&observation);
        evidence.raw_output_sha256 = "not-a-sha256".to_owned();
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Err(ArchitectureEvidenceError::InvalidDigest)
        );
    }

    #[test]
    fn non_hex_digest_is_rejected() {
        let observation = physical_observation(
            "cpu-baseline",
            ComputeArchitectureFamily::ClassicalCpu,
            (0.9, 1.0),
        );
        let mut evidence = physical_receipt(&observation);
        evidence.fixture_sha256 = "z".repeat(64);
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Err(ArchitectureEvidenceError::InvalidDigest)
        );
    }

    #[test]
    fn malformed_optional_git_revision_is_rejected() {
        let observation = physical_observation(
            "cpu-baseline",
            ComputeArchitectureFamily::ClassicalCpu,
            (0.9, 1.0),
        );
        let mut evidence = physical_receipt(&observation);
        evidence.source_revision = Some("abc123".to_owned());
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Err(ArchitectureEvidenceError::InvalidSourceRevision)
        );
    }

    #[test]
    fn repeat_counts_must_be_nonzero_and_failed_runs_cannot_exceed_total() {
        let observation = physical_observation(
            "cpu-baseline",
            ComputeArchitectureFamily::ClassicalCpu,
            (0.9, 1.0),
        );
        let mut evidence = physical_receipt(&observation);
        evidence.repeat_count = 0;
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Err(ArchitectureEvidenceError::InvalidRepeatCounts)
        );
        evidence.repeat_count = 3;
        evidence.failed_run_count = 4;
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Err(ArchitectureEvidenceError::InvalidRepeatCounts)
        );
    }

    #[test]
    fn direct_measurement_requires_device_toolchain_and_runner_identity() {
        let observation = physical_observation(
            "cpu-baseline",
            ComputeArchitectureFamily::ClassicalCpu,
            (0.9, 1.0),
        );
        let mut evidence = physical_receipt(&observation);
        evidence.device_identity = None;
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Err(ArchitectureEvidenceError::MissingDeviceIdentity)
        );
        let mut evidence = physical_receipt(&observation);
        evidence.toolchain_identity = None;
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Err(ArchitectureEvidenceError::MissingToolchainIdentity)
        );
        let mut evidence = physical_receipt(&observation);
        evidence.runner_artifact_sha256 = None;
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Err(ArchitectureEvidenceError::MissingRunnerArtifact)
        );
    }

    #[test]
    fn simulation_origin_cannot_be_promoted_to_physical_evidence() {
        let mut observation = physical_observation(
            "simulated-device",
            ComputeArchitectureFamily::NeuromorphicDigital,
            (0.4, 0.5),
        );
        observation.evidence_class = ArchitectureEvidenceClass::Simulation;
        let evidence = receipt(
            &observation,
            ArchitectureEvidenceOrigin::SimulationExecution,
        );
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Ok(())
        );
        let mut mismatched = evidence;
        mismatched.origin = ArchitectureEvidenceOrigin::DirectMeasurement;
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &mismatched),
            Err(ArchitectureEvidenceError::IncompatibleEvidenceOrigin)
        );
    }

    #[test]
    fn analytical_reference_is_only_valid_for_algorithmic_evidence() {
        let observation = algorithmic_observation();
        let evidence = receipt(
            &observation,
            ArchitectureEvidenceOrigin::AnalyticalReference,
        );
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Ok(())
        );
    }

    #[test]
    fn algorithmic_observation_cannot_claim_direct_physical_measurement() {
        let observation = algorithmic_observation();
        let evidence = receipt(
            &observation,
            ArchitectureEvidenceOrigin::DirectMeasurement,
        );
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Err(ArchitectureEvidenceError::IncompatibleEvidenceOrigin)
        );
    }

    #[test]
    fn missing_physical_device_identity_in_published_result_is_rejected() {
        let observation = physical_observation(
            "published-result",
            ComputeArchitectureFamily::NeuromorphicDigital,
            (0.4, 0.5),
        );
        let mut evidence = receipt(
            &observation,
            ArchitectureEvidenceOrigin::PublishedResult,
        );
        evidence.device_identity = None;
        assert_eq!(
            validate_architecture_evidence_receipt(&observation, &evidence),
            Err(ArchitectureEvidenceError::MissingDeviceIdentity)
        );
    }

    #[test]
    fn evidence_bound_comparison_validates_both_receipts() {
        let left = record(
            physical_observation(
                "cpu-baseline",
                ComputeArchitectureFamily::ClassicalCpu,
                (0.9, 1.0),
            ),
            ArchitectureEvidenceOrigin::DirectMeasurement,
        );
        let mut right_observation = physical_observation(
            "neuromorphic-candidate",
            ComputeArchitectureFamily::NeuromorphicDigital,
            (0.4, 0.5),
        );
        right_observation.quality = interval(0.97, 0.98);
        let right_resources = right_observation.resources.as_mut().unwrap();
        right_resources.energy_joules = interval(1.0, 1.5);
        right_resources.monetary_cost_usd = interval(0.005, 0.009);
        right_resources.peak_memory = MemoryInterval {
            lower_bytes: 512,
            upper_bytes: 768,
        };
        let right = record(
            right_observation,
            ArchitectureEvidenceOrigin::DirectMeasurement,
        );
        assert_eq!(
            compare_architecture_evidence_records(&left, &right),
            Ok(ArchitectureParetoRelation::RightDominates)
        );
    }

    #[test]
    fn evidence_bound_comparison_rejects_malformed_receipt() {
        let left = record(
            physical_observation(
                "cpu-baseline",
                ComputeArchitectureFamily::ClassicalCpu,
                (0.9, 1.0),
            ),
            ArchitectureEvidenceOrigin::DirectMeasurement,
        );
        let mut right = record(
            physical_observation(
                "neuromorphic-candidate",
                ComputeArchitectureFamily::NeuromorphicDigital,
                (0.4, 0.5),
            ),
            ArchitectureEvidenceOrigin::DirectMeasurement,
        );
        right.receipt.summary_sha256 = "missing-output-digest".to_owned();
        assert_eq!(
            compare_architecture_evidence_records(&left, &right),
            Err(ArchitectureEvidenceError::InvalidDigest)
        );
    }

    #[test]
    fn evidence_bound_selector_requires_every_receipt() {
        let request = ArchitectureRoutingRequest {
            workload_kind: ArchitectureWorkloadKind::HdcAlgebra,
            workload_fingerprint: "hdc-query-fixture-sha256".to_owned(),
            comparison_contract_fingerprint: "hdc-system-contract-sha256".to_owned(),
            evidence_class: ArchitectureEvidenceClass::PhysicalPrototype,
            measurement_boundary: ArchitectureMeasurementBoundary::ApplicationEndToEnd,
            minimum_quality: 0.90,
            resource_limits: ArchitectureResourceLimits {
                max_wall_time_seconds: 2.0,
                max_energy_joules: 5.0,
                max_monetary_cost_usd: 0.05,
                max_peak_memory_bytes: 4096,
            },
            priority: ArchitectureRoutingPriority::LowestWallTime,
        };
        let records = vec![
            record(
                physical_observation(
                    "cpu-baseline",
                    ComputeArchitectureFamily::ClassicalCpu,
                    (0.9, 1.0),
                ),
                ArchitectureEvidenceOrigin::DirectMeasurement,
            ),
            record(
                physical_observation(
                    "neuromorphic-candidate",
                    ComputeArchitectureFamily::NeuromorphicDigital,
                    (0.4, 0.5),
                ),
                ArchitectureEvidenceOrigin::DirectMeasurement,
            ),
        ];
        let report = select_architecture_candidate_from_evidence(&request, &records).unwrap();
        assert_eq!(
            report.selected_record_id.as_deref(),
            Some("neuromorphic-candidate")
        );
    }

    #[test]
    fn evidence_bound_selector_rejects_one_bad_receipt_instead_of_skipping_it() {
        let request = ArchitectureRoutingRequest {
            workload_kind: ArchitectureWorkloadKind::HdcAlgebra,
            workload_fingerprint: "hdc-query-fixture-sha256".to_owned(),
            comparison_contract_fingerprint: "hdc-system-contract-sha256".to_owned(),
            evidence_class: ArchitectureEvidenceClass::PhysicalPrototype,
            measurement_boundary: ArchitectureMeasurementBoundary::ApplicationEndToEnd,
            minimum_quality: 0.90,
            resource_limits: ArchitectureResourceLimits {
                max_wall_time_seconds: 2.0,
                max_energy_joules: 5.0,
                max_monetary_cost_usd: 0.05,
                max_peak_memory_bytes: 4096,
            },
            priority: ArchitectureRoutingPriority::LowestWallTime,
        };
        let mut invalid = record(
            physical_observation(
                "cpu-baseline",
                ComputeArchitectureFamily::ClassicalCpu,
                (0.9, 1.0),
            ),
            ArchitectureEvidenceOrigin::DirectMeasurement,
        );
        invalid.receipt.fixture_sha256 = "unverified".to_owned();
        assert_eq!(
            select_architecture_candidate_from_evidence(&request, &[invalid]),
            Err(ArchitectureEvidenceError::InvalidDigest)
        );
    }
}
