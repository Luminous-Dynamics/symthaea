//! Cross-paradigm architecture evidence and conservative Pareto comparability.
//!
//! This module provides a common vocabulary for recording architecture families
//! and comparing quality/resource envelopes. It does not execute workloads,
//! estimate missing measurements, or supply statistical inference. Input
//! intervals must come from a separately qualified, workload-matched analysis.
//! Comparisons fail closed when workload, contract, evidence class, or
//! measurement boundary differs.

/// Broad compute architecture family. These are distinct families, not a ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeArchitectureFamily {
    /// Conventional scalar/vector and multicore CPU systems.
    ClassicalCpu,
    /// Conventional SIMD/SIMT GPU systems.
    ClassicalGpu,
    /// Fixed-function neural or systolic processing units.
    Npu,
    /// Field-programmable gate arrays and reconfigurable dataflow fabrics.
    Fpga,
    /// Digital asynchronous, event-driven, or spiking neuromorphic systems.
    NeuromorphicDigital,
    /// Mixed-signal or accelerated analog neuromorphic systems.
    NeuromorphicMixedSignal,
    /// Digital processing integrated within DRAM/SRAM banks or a memory-module logic layer.
    DigitalProcessingInMemory,
    /// Digital compute adjacent to memory arrays, but outside the arrays and memory module.
    NearMemoryCompute,
    /// Analog compute-in-memory, including resistive and flash arrays.
    AnalogComputeInMemory,
    /// Classical optical/photonic linear or matrix computation.
    PhotonicLinear,
    /// Photonic physical-reservoir computation.
    PhotonicReservoir,
    /// Gate-model superconducting quantum processors.
    QuantumGateSuperconducting,
    /// Gate-model trapped-ion quantum processors.
    QuantumGateTrappedIon,
    /// Gate-model neutral-atom quantum processors.
    QuantumGateNeutralAtom,
    /// Gate-model photonic quantum processors.
    QuantumGatePhotonic,
    /// Quantum annealing and quantum Ising processors.
    QuantumAnnealing,
    /// Non-photonic physical reservoir computing substrates.
    PhysicalReservoir,
    /// Spintronic compute substrates.
    Spintronic,
    /// Probabilistic or p-bit compute substrates.
    Probabilistic,
    /// Reversible or adiabatic computing substrates.
    Reversible,
    /// Explicit heterogeneous combinations of two or more compute families.
    HybridHeterogeneous,
}

/// Strength of the evidence behind an architecture result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureEvidenceClass {
    /// Algorithmic reference or complexity estimate without a hardware model.
    AlgorithmicReference,
    /// Software or mathematical simulation.
    Simulation,
    /// Hardware emulation, FPGA emulation, or cycle-accurate model.
    HardwareEmulation,
    /// Measurement on an identified physical prototype.
    PhysicalPrototype,
    /// Measurement on an identified deployed/production system.
    ProductionSystem,
}

/// Resource measurement boundary. Different boundaries are never compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureMeasurementBoundary {
    /// Algorithm/model footprint or operation complexity only.
    AlgorithmOnly,
    /// A kernel, device, chip, or accelerator measured in isolation.
    KernelOrDevice,
    /// Runtime/system boundary including host/device execution overhead.
    RuntimeEndToEnd,
    /// Complete application execution through independently checked output.
    ApplicationEndToEnd,
    /// Physical sensor/input through processing to the task-facing decision.
    SensorToDecision,
}

/// Workload class used to identify the scope of a comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureWorkloadKind {
    /// HDC binding, bundling, permutation, or associative retrieval.
    HdcAlgebra,
    /// Sparse event-stream perception or temporal event classification.
    SparseEventPerception,
    /// Streaming sensor processing and multi-sensor fusion.
    StreamingSensorFusion,
    /// Sparse recurrent or continuous-time state evolution.
    ContinuousTimeDynamics,
    /// Few-shot, continual, or online learning and adaptation.
    ContinualLearning,
    /// Bounded edge robotics or control workload.
    EdgeControl,
    /// Combinatorial or constrained optimization.
    CombinatorialOptimization,
    /// Dense tensor, matrix, or conventional inference workload.
    DenseTensorInference,
    /// A separately specified workload not covered by the named classes.
    Other,
}

/// Closed interval for a quality or continuous resource measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricInterval {
    /// Lower bound, inclusive.
    pub lower: f64,
    /// Upper bound, inclusive.
    pub upper: f64,
}

impl MetricInterval {
    /// Construct a finite ordered interval.
    pub fn new(lower: f64, upper: f64) -> Result<Self, ArchitectureComparisonError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(ArchitectureComparisonError::InvalidInterval);
        }
        Ok(Self { lower, upper })
    }

    fn is_valid(self) -> bool {
        self.lower.is_finite() && self.upper.is_finite() && self.lower <= self.upper
    }
}

/// Integer interval for peak working-set memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryInterval {
    /// Lower bound in bytes, inclusive.
    pub lower_bytes: u64,
    /// Upper bound in bytes, inclusive.
    pub upper_bytes: u64,
}

impl MemoryInterval {
    /// Construct a finite ordered memory interval.
    pub fn new(
        lower_bytes: u64,
        upper_bytes: u64,
    ) -> Result<Self, ArchitectureComparisonError> {
        if lower_bytes > upper_bytes {
            return Err(ArchitectureComparisonError::InvalidInterval);
        }
        Ok(Self {
            lower_bytes,
            upper_bytes,
        })
    }
}

/// Complete resource envelope required by this first Pareto comparator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArchitectureResourceEnvelope {
    /// Wall-clock time for the declared measurement boundary, in seconds.
    pub wall_time_seconds: MetricInterval,
    /// Energy consumed within the declared boundary, in joules.
    pub energy_joules: MetricInterval,
    /// Monetary execution cost within the declared boundary, in USD.
    pub monetary_cost_usd: MetricInterval,
    /// Peak working-set memory within the declared boundary, in bytes.
    pub peak_memory: MemoryInterval,
}

impl ArchitectureResourceEnvelope {
    fn is_valid(self) -> bool {
        self.wall_time_seconds.is_valid()
            && self.energy_joules.is_valid()
            && self.monetary_cost_usd.is_valid()
            && self.wall_time_seconds.lower >= 0.0
            && self.energy_joules.lower >= 0.0
            && self.monetary_cost_usd.lower >= 0.0
            && self.peak_memory.lower_bytes <= self.peak_memory.upper_bytes
    }
}

/// One summary observation for one architecture and one frozen workload contract.
///
/// The quality interval must describe the exact metric in the comparison
/// contract. The resource intervals must all use the same system boundary.
/// Independent block/family inference and multiplicity correction happen
/// upstream; this record is not a substitute for QARCH-015/016/017.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureBenchmarkObservation {
    /// Unique observation/report identifier.
    pub record_id: String,
    /// Concrete system or candidate identity, not just a family label.
    pub architecture_id: String,
    /// Broad architecture family.
    pub architecture_family: ComputeArchitectureFamily,
    /// Workload category.
    pub workload_kind: ArchitectureWorkloadKind,
    /// Fingerprint of the exact input workload or benchmark instance.
    pub workload_fingerprint: String,
    /// Digest binding task, objective, quality contract, verifier and resource boundary.
    pub comparison_contract_fingerprint: String,
    /// Evidence maturity of this observation.
    pub evidence_class: ArchitectureEvidenceClass,
    /// Actual measurement boundary used.
    pub measurement_boundary: ArchitectureMeasurementBoundary,
    /// Quality interval; larger is better under the shared contract.
    pub quality: MetricInterval,
    /// Complete resource vector for system-level observations; lower is better.
    /// Algorithm-only references have no physical resource envelope and cannot
    /// enter the system-level Pareto comparator.
    pub resources: Option<ArchitectureResourceEnvelope>,
}

/// Conservative relation between two architecture observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureParetoRelation {
    /// Left is no worse on quality and every resource, and strictly better on at least one.
    LeftDominates,
    /// Right is no worse on quality and every resource, and strictly better on at least one.
    RightDominates,
    /// Quality and/or resource dimensions robustly favor different observations.
    Tradeoff,
    /// The supplied intervals overlap or do not establish a robust ordering.
    Unresolved,
    /// All compared quality and resource intervals are identical.
    Equivalent,
}

/// Rejected comparison or malformed observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureComparisonError {
    /// A required record, architecture, workload, or contract identifier is blank.
    EmptyIdentifier,
    /// A continuous interval is non-finite or has its lower bound above its upper bound.
    InvalidInterval,
    /// A resource interval includes a negative value.
    NegativeResource,
    /// A system-level comparison lacks one or both complete resource envelopes.
    MissingResourceEnvelope,
    /// Algorithm-only complexity references cannot enter the resource Pareto comparator.
    AlgorithmOnlyNotComparable,
    /// Algorithm-only references must not carry a physical resource envelope.
    UnexpectedResourceEnvelope,
    /// Workload fingerprints differ.
    WorkloadMismatch,
    /// Comparison-contract digests differ.
    ContractMismatch,
    /// Evidence classes differ (for example, simulation versus physical prototype).
    EvidenceClassMismatch,
    /// Measurement boundaries differ (for example, kernel-only versus end-to-end).
    MeasurementBoundaryMismatch,
}

/// Validate the identity, intervals and non-negative resource contract.
pub fn validate_architecture_observation(
    observation: &ArchitectureBenchmarkObservation,
) -> Result<(), ArchitectureComparisonError> {
    if observation.record_id.trim().is_empty()
        || observation.architecture_id.trim().is_empty()
        || observation.workload_fingerprint.trim().is_empty()
        || observation.comparison_contract_fingerprint.trim().is_empty()
    {
        return Err(ArchitectureComparisonError::EmptyIdentifier);
    }
    if !observation.quality.is_valid() {
        return Err(ArchitectureComparisonError::InvalidInterval);
    }
    match (observation.measurement_boundary, observation.resources) {
        (ArchitectureMeasurementBoundary::AlgorithmOnly, Some(_)) => {
            return Err(ArchitectureComparisonError::UnexpectedResourceEnvelope);
        }
        (ArchitectureMeasurementBoundary::AlgorithmOnly, None) => {}
        (_, None) => return Err(ArchitectureComparisonError::MissingResourceEnvelope),
        (_, Some(resources)) => {
            if resources.wall_time_seconds.lower < 0.0
                || resources.energy_joules.lower < 0.0
                || resources.monetary_cost_usd.lower < 0.0
            {
                return Err(ArchitectureComparisonError::NegativeResource);
            }
            if !resources.is_valid() {
                return Err(ArchitectureComparisonError::InvalidInterval);
            }
        }
    }
    Ok(())
}

/// Compare two observations only when their workload, claim contract, evidence
/// class, and measurement boundary are identical.
///
/// Dominance is conservative: the better observation's entire quality interval
/// must be no worse than the other's, and every resource upper bound must be no
/// greater than the other's lower bound. Overlapping uncertainty is not treated
/// as a win. This operation does not calculate confidence intervals or p-values;
/// those must be produced by the appropriate independent statistical procedure.
pub fn compare_architecture_pareto(
    left: &ArchitectureBenchmarkObservation,
    right: &ArchitectureBenchmarkObservation,
) -> Result<ArchitectureParetoRelation, ArchitectureComparisonError> {
    validate_architecture_observation(left)?;
    validate_architecture_observation(right)?;

    if left.workload_fingerprint != right.workload_fingerprint
        || left.workload_kind != right.workload_kind
    {
        return Err(ArchitectureComparisonError::WorkloadMismatch);
    }
    if left.comparison_contract_fingerprint != right.comparison_contract_fingerprint {
        return Err(ArchitectureComparisonError::ContractMismatch);
    }
    if left.evidence_class != right.evidence_class {
        return Err(ArchitectureComparisonError::EvidenceClassMismatch);
    }
    if left.measurement_boundary != right.measurement_boundary {
        return Err(ArchitectureComparisonError::MeasurementBoundaryMismatch);
    }
    if left.measurement_boundary == ArchitectureMeasurementBoundary::AlgorithmOnly {
        return Err(ArchitectureComparisonError::AlgorithmOnlyNotComparable);
    }
    let left_resources = left.resources.ok_or(ArchitectureComparisonError::MissingResourceEnvelope)?;
    let right_resources = right.resources.ok_or(ArchitectureComparisonError::MissingResourceEnvelope)?;

    let same = left.quality == right.quality
        && left_resources.wall_time_seconds == right_resources.wall_time_seconds
        && left_resources.energy_joules == right_resources.energy_joules
        && left_resources.monetary_cost_usd == right_resources.monetary_cost_usd
        && left_resources.peak_memory == right_resources.peak_memory;
    if same {
        return Ok(ArchitectureParetoRelation::Equivalent);
    }

    let left_quality_no_worse = left.quality.lower >= right.quality.upper;
    let right_quality_no_worse = right.quality.lower >= left.quality.upper;

    let left_resources_no_worse = resources_no_worse(left_resources, right_resources);
    let right_resources_no_worse = resources_no_worse(right_resources, left_resources);

    let left_strict = left.quality.lower > right.quality.upper
        || resources_strictly_better(left_resources, right_resources);
    let right_strict = right.quality.lower > left.quality.upper
        || resources_strictly_better(right_resources, left_resources);

    if left_quality_no_worse && left_resources_no_worse && left_strict {
        return Ok(ArchitectureParetoRelation::LeftDominates);
    }
    if right_quality_no_worse && right_resources_no_worse && right_strict {
        return Ok(ArchitectureParetoRelation::RightDominates);
    }
    if left_strict && right_strict {
        return Ok(ArchitectureParetoRelation::Tradeoff);
    }
    Ok(ArchitectureParetoRelation::Unresolved)
}

fn resources_no_worse(
    candidate: ArchitectureResourceEnvelope,
    comparison: ArchitectureResourceEnvelope,
) -> bool {
    candidate.wall_time_seconds.upper <= comparison.wall_time_seconds.lower
        && candidate.energy_joules.upper <= comparison.energy_joules.lower
        && candidate.monetary_cost_usd.upper <= comparison.monetary_cost_usd.lower
        && candidate.peak_memory.upper_bytes <= comparison.peak_memory.lower_bytes
}

fn resources_strictly_better(
    candidate: ArchitectureResourceEnvelope,
    comparison: ArchitectureResourceEnvelope,
) -> bool {
    candidate.wall_time_seconds.upper < comparison.wall_time_seconds.lower
        || candidate.energy_joules.upper < comparison.energy_joules.lower
        || candidate.monetary_cost_usd.upper < comparison.monetary_cost_usd.lower
        || candidate.peak_memory.upper_bytes < comparison.peak_memory.lower_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(lower: f64, upper: f64) -> MetricInterval {
        MetricInterval { lower, upper }
    }

    fn observation(
        id: &str,
        family: ComputeArchitectureFamily,
        quality: MetricInterval,
        resources: ArchitectureResourceEnvelope,
    ) -> ArchitectureBenchmarkObservation {
        ArchitectureBenchmarkObservation {
            record_id: format!("record-{id}"),
            architecture_id: id.to_owned(),
            architecture_family: family,
            workload_kind: ArchitectureWorkloadKind::HdcAlgebra,
            workload_fingerprint: "workload-sha256".to_owned(),
            comparison_contract_fingerprint: "contract-sha256".to_owned(),
            evidence_class: ArchitectureEvidenceClass::PhysicalPrototype,
            measurement_boundary: ArchitectureMeasurementBoundary::ApplicationEndToEnd,
            quality,
            resources: Some(resources),
        }
    }

    fn resource_envelope(
        time: (f64, f64),
        energy: (f64, f64),
        cost: (f64, f64),
        memory: (u64, u64),
    ) -> ArchitectureResourceEnvelope {
        ArchitectureResourceEnvelope {
            wall_time_seconds: interval(time.0, time.1),
            energy_joules: interval(energy.0, energy.1),
            monetary_cost_usd: interval(cost.0, cost.1),
            peak_memory: MemoryInterval {
                lower_bytes: memory.0,
                upper_bytes: memory.1,
            },
        }
    }

    fn algorithm_reference() -> ArchitectureBenchmarkObservation {
        ArchitectureBenchmarkObservation {
            record_id: "algorithm-reference".to_owned(),
            architecture_id: "algorithm-only-reference".to_owned(),
            architecture_family: ComputeArchitectureFamily::ClassicalCpu,
            workload_kind: ArchitectureWorkloadKind::HdcAlgebra,
            workload_fingerprint: "workload-sha256".to_owned(),
            comparison_contract_fingerprint: "contract-sha256".to_owned(),
            evidence_class: ArchitectureEvidenceClass::AlgorithmicReference,
            measurement_boundary: ArchitectureMeasurementBoundary::AlgorithmOnly,
            quality: interval(0.94, 0.95),
            resources: None,
        }
    }

    fn cpu() -> ArchitectureBenchmarkObservation {
        observation(
            "cpu",
            ComputeArchitectureFamily::ClassicalCpu,
            interval(0.94, 0.95),
            resource_envelope((1.8, 2.0), (9.0, 10.0), (0.08, 0.09), (8_000, 9_000)),
        )
    }

    fn neuromorphic() -> ArchitectureBenchmarkObservation {
        observation(
            "loihi-prototype",
            ComputeArchitectureFamily::NeuromorphicDigital,
            interval(0.96, 0.97),
            resource_envelope((0.8, 0.9), (3.0, 4.0), (0.03, 0.04), (4_000, 5_000)),
        )
    }

    #[test]
    fn memory_centric_compute_families_remain_distinct() {
        assert_ne!(
            ComputeArchitectureFamily::DigitalProcessingInMemory,
            ComputeArchitectureFamily::NearMemoryCompute
        );
        assert_ne!(
            ComputeArchitectureFamily::DigitalProcessingInMemory,
            ComputeArchitectureFamily::AnalogComputeInMemory
        );
        assert_ne!(
            ComputeArchitectureFamily::NearMemoryCompute,
            ComputeArchitectureFamily::AnalogComputeInMemory
        );
    }

    #[test]
    fn algorithm_only_reference_is_valid_but_not_a_system_pareto_point() {
        let reference = algorithm_reference();
        assert_eq!(validate_architecture_observation(&reference), Ok(()));
        assert_eq!(
            compare_architecture_pareto(&reference, &reference),
            Err(ArchitectureComparisonError::AlgorithmOnlyNotComparable)
        );
    }

    #[test]
    fn system_level_observation_requires_complete_resource_envelope() {
        let mut invalid = cpu();
        invalid.resources = None;
        assert_eq!(
            validate_architecture_observation(&invalid),
            Err(ArchitectureComparisonError::MissingResourceEnvelope)
        );
    }

    #[test]
    fn algorithm_only_reference_cannot_claim_physical_resources() {
        let mut invalid = algorithm_reference();
        invalid.resources = Some(resource_envelope(
            (1.0, 1.0),
            (1.0, 1.0),
            (0.1, 0.1),
            (1024, 1024),
        ));
        assert_eq!(
            validate_architecture_observation(&invalid),
            Err(ArchitectureComparisonError::UnexpectedResourceEnvelope)
        );
    }

    #[test]
    fn architecture_families_remain_distinct() {
        assert_ne!(
            ComputeArchitectureFamily::PhotonicLinear,
            ComputeArchitectureFamily::QuantumGatePhotonic
        );
        assert_ne!(
            ComputeArchitectureFamily::NeuromorphicDigital,
            ComputeArchitectureFamily::AnalogComputeInMemory
        );
        assert_ne!(
            ComputeArchitectureFamily::QuantumAnnealing,
            ComputeArchitectureFamily::QuantumGateNeutralAtom
        );
    }

    #[test]
    fn robustly_better_architecture_dominates() {
        assert_eq!(
            compare_architecture_pareto(&neuromorphic(), &cpu()).unwrap(),
            ArchitectureParetoRelation::LeftDominates
        );
    }

    #[test]
    fn right_side_can_dominate() {
        assert_eq!(
            compare_architecture_pareto(&cpu(), &neuromorphic()).unwrap(),
            ArchitectureParetoRelation::RightDominates
        );
    }

    #[test]
    fn quality_resource_conflict_is_a_tradeoff() {
        let quality_focused = observation(
            "quality-system",
            ComputeArchitectureFamily::PhotonicLinear,
            interval(0.99, 1.0),
            resource_envelope((2.0, 2.2), (8.0, 9.0), (0.10, 0.11), (16_000, 17_000)),
        );
        let resource_focused = observation(
            "resource-system",
            ComputeArchitectureFamily::NeuromorphicMixedSignal,
            interval(0.90, 0.92),
            resource_envelope((0.3, 0.4), (1.0, 1.5), (0.01, 0.02), (1_000, 2_000)),
        );
        assert_eq!(
            compare_architecture_pareto(&quality_focused, &resource_focused).unwrap(),
            ArchitectureParetoRelation::Tradeoff
        );
    }

    #[test]
    fn overlapping_intervals_do_not_create_a_win() {
        let left = observation(
            "left",
            ComputeArchitectureFamily::ClassicalGpu,
            interval(0.94, 0.97),
            resource_envelope((0.8, 1.2), (4.0, 6.0), (0.04, 0.06), (4_000, 6_000)),
        );
        let right = observation(
            "right",
            ComputeArchitectureFamily::Npu,
            interval(0.95, 0.98),
            resource_envelope((0.9, 1.1), (4.5, 5.5), (0.045, 0.055), (4_500, 5_500)),
        );
        assert_eq!(
            compare_architecture_pareto(&left, &right).unwrap(),
            ArchitectureParetoRelation::Unresolved
        );
    }

    #[test]
    fn identical_metric_envelopes_are_equivalent() {
        let left = cpu();
        let mut right = cpu();
        right.architecture_id = "cpu-revision-b".to_owned();
        right.architecture_family = ComputeArchitectureFamily::ClassicalGpu;
        right.record_id = "record-cpu-revision-b".to_owned();
        assert_eq!(
            compare_architecture_pareto(&left, &right).unwrap(),
            ArchitectureParetoRelation::Equivalent
        );
    }

    #[test]
    fn different_workloads_are_not_comparable() {
        let left = cpu();
        let mut right = neuromorphic();
        right.workload_fingerprint = "different-workload".to_owned();
        assert_eq!(
            compare_architecture_pareto(&left, &right),
            Err(ArchitectureComparisonError::WorkloadMismatch)
        );
    }

    #[test]
    fn different_contracts_are_not_comparable() {
        let left = cpu();
        let mut right = neuromorphic();
        right.comparison_contract_fingerprint = "different-contract".to_owned();
        assert_eq!(
            compare_architecture_pareto(&left, &right),
            Err(ArchitectureComparisonError::ContractMismatch)
        );
    }

    #[test]
    fn simulation_is_not_compared_as_physical_prototype_evidence() {
        let left = cpu();
        let mut right = neuromorphic();
        right.evidence_class = ArchitectureEvidenceClass::Simulation;
        assert_eq!(
            compare_architecture_pareto(&left, &right),
            Err(ArchitectureComparisonError::EvidenceClassMismatch)
        );
    }

    #[test]
    fn device_only_is_not_compared_as_application_end_to_end() {
        let left = cpu();
        let mut right = neuromorphic();
        right.measurement_boundary = ArchitectureMeasurementBoundary::KernelOrDevice;
        assert_eq!(
            compare_architecture_pareto(&left, &right),
            Err(ArchitectureComparisonError::MeasurementBoundaryMismatch)
        );
    }

    #[test]
    fn blank_record_identity_is_rejected() {
        let mut invalid = cpu();
        invalid.architecture_id = "  ".to_owned();
        assert_eq!(
            validate_architecture_observation(&invalid),
            Err(ArchitectureComparisonError::EmptyIdentifier)
        );
    }

    #[test]
    fn non_finite_quality_is_rejected() {
        let mut invalid = cpu();
        invalid.quality.upper = f64::INFINITY;
        assert_eq!(
            validate_architecture_observation(&invalid),
            Err(ArchitectureComparisonError::InvalidInterval)
        );
    }

    #[test]
    fn negative_resource_is_rejected() {
        let mut invalid = cpu();
        invalid.resources.as_mut().unwrap().energy_joules.lower = -0.1;
        assert_eq!(
            validate_architecture_observation(&invalid),
            Err(ArchitectureComparisonError::NegativeResource)
        );
    }

    #[test]
    fn inverted_interval_is_rejected() {
        let invalid = MetricInterval {
            lower: 2.0,
            upper: 1.0,
        };
        let mut record = cpu();
        record.resources.as_mut().unwrap().wall_time_seconds = invalid;
        assert_eq!(
            validate_architecture_observation(&record),
            Err(ArchitectureComparisonError::InvalidInterval)
        );
    }

    #[test]
    fn constructors_reject_invalid_intervals() {
        assert_eq!(
            MetricInterval::new(f64::NAN, 1.0),
            Err(ArchitectureComparisonError::InvalidInterval)
        );
        assert_eq!(
            MetricInterval::new(2.0, 1.0),
            Err(ArchitectureComparisonError::InvalidInterval)
        );
        assert_eq!(
            MemoryInterval::new(9, 8),
            Err(ArchitectureComparisonError::InvalidInterval)
        );
    }
}
