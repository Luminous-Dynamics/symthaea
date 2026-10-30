//! Constraint-aware, evidence-bounded selection over measured architecture candidates.
//!
//! This is a deterministic qualification policy, not a learned runtime router.
//! It selects only among observations already measured under the exact requested
//! workload, contract, evidence class, and system boundary. It never estimates
//! missing measurements or treats an uncertain interval as feasible.
use std::cmp::Ordering;
use std::collections::HashSet;

use crate::architecture_portfolio::{
    validate_architecture_observation, ArchitectureBenchmarkObservation,
    ArchitectureComparisonError, ArchitectureEvidenceClass, ArchitectureMeasurementBoundary,
    ArchitectureResourceEnvelope, ArchitectureWorkloadKind,
};

/// User-declared first resource objective for deterministic lexicographic selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureRoutingPriority {
    /// Minimize the conservative upper bound on wall-clock time first.
    LowestWallTime,
    /// Minimize the conservative upper bound on energy first.
    LowestEnergy,
    /// Minimize the conservative upper bound on monetary cost first.
    LowestMonetaryCost,
    /// Minimize the conservative upper bound on peak memory first.
    LowestPeakMemory,
}

/// Hard resource ceilings. A candidate is feasible only when every interval's
/// upper bound is at or below its corresponding ceiling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArchitectureResourceLimits {
    /// Maximum wall-clock time in seconds.
    pub max_wall_time_seconds: f64,
    /// Maximum energy in joules.
    pub max_energy_joules: f64,
    /// Maximum monetary execution cost in USD.
    pub max_monetary_cost_usd: f64,
    /// Maximum peak working-set size in bytes.
    pub max_peak_memory_bytes: u64,
}

/// Frozen query defining the exact scope in which routing is allowed.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureRoutingRequest {
    /// Workload category.
    pub workload_kind: ArchitectureWorkloadKind,
    /// Fingerprint of the exact workload/instance.
    pub workload_fingerprint: String,
    /// Digest of task, objective, quality metric, verifier and boundary contract.
    pub comparison_contract_fingerprint: String,
    /// Required evidence maturity. Evidence classes are never silently mixed.
    pub evidence_class: ArchitectureEvidenceClass,
    /// Required system measurement boundary. Algorithm-only routing is forbidden.
    pub measurement_boundary: ArchitectureMeasurementBoundary,
    /// Minimum acceptable quality; the candidate's entire interval must clear it.
    pub minimum_quality: f64,
    /// Hard resource ceilings.
    pub resource_limits: ArchitectureResourceLimits,
    /// Explicit lexicographic objective among candidates satisfying every constraint.
    pub priority: ArchitectureRoutingPriority,
}

/// Why a measured candidate was not eligible for selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureRoutingExclusion {
    /// The observation itself failed the portfolio data contract.
    InvalidObservation(ArchitectureComparisonError),
    /// Workload category or exact workload fingerprint differs.
    WorkloadMismatch,
    /// Comparison contract digest differs.
    ContractMismatch,
    /// Evidence maturity differs from the request.
    EvidenceClassMismatch,
    /// Measurement boundary differs from the request.
    MeasurementBoundaryMismatch,
    /// Algorithm-only evidence cannot be used for system-level selection.
    AlgorithmOnlyEvidence,
    /// No complete physical/system resource envelope is available.
    MissingResourceEnvelope,
    /// Even the upper end of the quality interval is below the minimum.
    QualityBelowThreshold,
    /// The quality interval overlaps the threshold and therefore does not prove it.
    QualityNotProven,
    /// Even the lower end of the resource interval exceeds the hard ceiling.
    WallTimeLimitExceeded,
    /// The wall-time interval overlaps the ceiling and does not prove feasibility.
    WallTimeNotProven,
    /// Even the lower end of the resource interval exceeds the hard ceiling.
    EnergyLimitExceeded,
    /// The energy interval overlaps the ceiling and does not prove feasibility.
    EnergyNotProven,
    /// Even the lower end of the resource interval exceeds the hard ceiling.
    MonetaryCostLimitExceeded,
    /// The cost interval overlaps the ceiling and does not prove feasibility.
    MonetaryCostNotProven,
    /// Even the lower end of the memory interval exceeds the hard ceiling.
    PeakMemoryLimitExceeded,
    /// The memory interval overlaps the ceiling and does not prove feasibility.
    PeakMemoryNotProven,
}

/// Candidate-level result. An empty exclusion list means the candidate is eligible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureRoutingCandidate {
    /// Unique observation/report identifier.
    pub record_id: String,
    /// Concrete hardware/runtime candidate identifier.
    pub architecture_id: String,
    /// All applicable reasons for exclusion; empty means eligible.
    pub exclusions: Vec<ArchitectureRoutingExclusion>,
}

impl ArchitectureRoutingCandidate {
    /// Whether all hard constraints and evidence-scope checks were satisfied.
    pub fn is_eligible(&self) -> bool {
        self.exclusions.is_empty()
    }
}

/// Why the selector did not return a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureRoutingAbstention {
    /// No supplied observation proved every requested condition.
    NoEligibleCandidate,
}

/// Auditable selection outcome. The eligible IDs are ordered by the declared policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureRoutingReport {
    /// Candidate checks, preserved in input order with explicit rejection reasons.
    pub candidates: Vec<ArchitectureRoutingCandidate>,
    /// Eligible observation IDs in deterministic selection order.
    pub eligible_record_ids: Vec<String>,
    /// Selected observation ID, or None when the selector abstains.
    pub selected_record_id: Option<String>,
    /// Selected concrete architecture ID, or None when the selector abstains.
    pub selected_architecture_id: Option<String>,
    /// Present exactly when no candidate was eligible.
    pub abstention: Option<ArchitectureRoutingAbstention>,
}

/// Invalid request or ambiguous candidate-set condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureRoutingError {
    /// Workload and comparison-contract fingerprints must be non-empty.
    EmptyIdentifier,
    /// The minimum-quality threshold must be finite.
    InvalidQualityThreshold,
    /// Resource ceilings must be finite and non-negative.
    InvalidResourceLimit,
    /// An algorithm-only request cannot select a physical/system implementation.
    AlgorithmOnlyRequest,
    /// Candidate observation IDs must be unique within a selection call.
    DuplicateRecordId,
}

fn validate_request(request: &ArchitectureRoutingRequest) -> Result<(), ArchitectureRoutingError> {
    if request.workload_fingerprint.trim().is_empty()
        || request.comparison_contract_fingerprint.trim().is_empty()
    {
        return Err(ArchitectureRoutingError::EmptyIdentifier);
    }
    if !request.minimum_quality.is_finite() {
        return Err(ArchitectureRoutingError::InvalidQualityThreshold);
    }
    let limits = request.resource_limits;
    if !limits.max_wall_time_seconds.is_finite()
        || !limits.max_energy_joules.is_finite()
        || !limits.max_monetary_cost_usd.is_finite()
        || limits.max_wall_time_seconds < 0.0
        || limits.max_energy_joules < 0.0
        || limits.max_monetary_cost_usd < 0.0
    {
        return Err(ArchitectureRoutingError::InvalidResourceLimit);
    }
    if request.measurement_boundary == ArchitectureMeasurementBoundary::AlgorithmOnly {
        return Err(ArchitectureRoutingError::AlgorithmOnlyRequest);
    }
    Ok(())
}

fn append_interval_exclusions(
    candidate: &ArchitectureBenchmarkObservation,
    request: &ArchitectureRoutingRequest,
    resources: ArchitectureResourceEnvelope,
    exclusions: &mut Vec<ArchitectureRoutingExclusion>,
) {
    let quality = candidate.quality;
    if quality.upper < request.minimum_quality {
        exclusions.push(ArchitectureRoutingExclusion::QualityBelowThreshold);
    } else if quality.lower < request.minimum_quality {
        exclusions.push(ArchitectureRoutingExclusion::QualityNotProven);
    }

    let limits = request.resource_limits;
    if resources.wall_time_seconds.lower > limits.max_wall_time_seconds {
        exclusions.push(ArchitectureRoutingExclusion::WallTimeLimitExceeded);
    } else if resources.wall_time_seconds.upper > limits.max_wall_time_seconds {
        exclusions.push(ArchitectureRoutingExclusion::WallTimeNotProven);
    }

    if resources.energy_joules.lower > limits.max_energy_joules {
        exclusions.push(ArchitectureRoutingExclusion::EnergyLimitExceeded);
    } else if resources.energy_joules.upper > limits.max_energy_joules {
        exclusions.push(ArchitectureRoutingExclusion::EnergyNotProven);
    }

    if resources.monetary_cost_usd.lower > limits.max_monetary_cost_usd {
        exclusions.push(ArchitectureRoutingExclusion::MonetaryCostLimitExceeded);
    } else if resources.monetary_cost_usd.upper > limits.max_monetary_cost_usd {
        exclusions.push(ArchitectureRoutingExclusion::MonetaryCostNotProven);
    }

    if resources.peak_memory.lower_bytes > limits.max_peak_memory_bytes {
        exclusions.push(ArchitectureRoutingExclusion::PeakMemoryLimitExceeded);
    } else if resources.peak_memory.upper_bytes > limits.max_peak_memory_bytes {
        exclusions.push(ArchitectureRoutingExclusion::PeakMemoryNotProven);
    }
}

fn compare_candidates(
    left: &ArchitectureRoutingCandidate,
    right: &ArchitectureRoutingCandidate,
    observations: &[ArchitectureBenchmarkObservation],
    priority: ArchitectureRoutingPriority,
) -> Ordering {
    let left_observation = observations
        .iter()
        .find(|observation| observation.record_id == left.record_id)
        .expect("candidate observations are drawn from the supplied slice");
    let right_observation = observations
        .iter()
        .find(|observation| observation.record_id == right.record_id)
        .expect("candidate observations are drawn from the supplied slice");
    let left_resources = left_observation.resources.expect("eligible candidates have resources");
    let right_resources = right_observation.resources.expect("eligible candidates have resources");

    let time = || {
        left_resources.wall_time_seconds.upper.total_cmp(&right_resources.wall_time_seconds.upper)
    };
    let energy = || {
        left_resources.energy_joules.upper.total_cmp(&right_resources.energy_joules.upper)
    };
    let cost = || {
        left_resources.monetary_cost_usd.upper.total_cmp(&right_resources.monetary_cost_usd.upper)
    };
    let memory = || {
        left_resources.peak_memory.upper_bytes.cmp(&right_resources.peak_memory.upper_bytes)
    };

    let resource_order = match priority {
        ArchitectureRoutingPriority::LowestWallTime => time().then_with(energy).then_with(cost).then_with(memory),
        ArchitectureRoutingPriority::LowestEnergy => energy().then_with(time).then_with(cost).then_with(memory),
        ArchitectureRoutingPriority::LowestMonetaryCost => cost().then_with(time).then_with(energy).then_with(memory),
        ArchitectureRoutingPriority::LowestPeakMemory => memory().then_with(time).then_with(energy).then_with(cost),
    };

    resource_order
        .then_with(|| right_observation.quality.lower.total_cmp(&left_observation.quality.lower))
        .then_with(|| left.architecture_id.cmp(&right.architecture_id))
        .then_with(|| left.record_id.cmp(&right.record_id))
}

/// Select from previously measured candidates using conservative hard constraints.
///
/// Selection is deliberately stateless. It does not learn, switch devices at
/// runtime, estimate unmeasured architectures, or account for migration/cold-start
/// overhead. Those costs must be measured and included in the supplied resource
/// envelope before runtime-routing claims are made.
pub fn select_architecture_candidate(
    request: &ArchitectureRoutingRequest,
    observations: &[ArchitectureBenchmarkObservation],
) -> Result<ArchitectureRoutingReport, ArchitectureRoutingError> {
    validate_request(request)?;

    let mut seen_ids = HashSet::new();
    for observation in observations {
        if !observation.record_id.trim().is_empty()
            && !seen_ids.insert(observation.record_id.as_str())
        {
            return Err(ArchitectureRoutingError::DuplicateRecordId);
        }
    }

    let mut candidates = Vec::with_capacity(observations.len());
    for observation in observations {
        let mut exclusions = Vec::new();
        if let Err(error) = validate_architecture_observation(observation) {
            exclusions.push(ArchitectureRoutingExclusion::InvalidObservation(error));
            candidates.push(ArchitectureRoutingCandidate {
                record_id: observation.record_id.clone(),
                architecture_id: observation.architecture_id.clone(),
                exclusions,
            });
            continue;
        }

        if observation.workload_kind != request.workload_kind
            || observation.workload_fingerprint != request.workload_fingerprint
        {
            exclusions.push(ArchitectureRoutingExclusion::WorkloadMismatch);
        }
        if observation.comparison_contract_fingerprint != request.comparison_contract_fingerprint {
            exclusions.push(ArchitectureRoutingExclusion::ContractMismatch);
        }
        if observation.evidence_class != request.evidence_class {
            exclusions.push(ArchitectureRoutingExclusion::EvidenceClassMismatch);
        }
        if observation.measurement_boundary != request.measurement_boundary {
            exclusions.push(ArchitectureRoutingExclusion::MeasurementBoundaryMismatch);
        }
        if observation.measurement_boundary == ArchitectureMeasurementBoundary::AlgorithmOnly {
            exclusions.push(ArchitectureRoutingExclusion::AlgorithmOnlyEvidence);
        }

        match observation.resources {
            Some(resources) => append_interval_exclusions(observation, request, resources, &mut exclusions),
            None => exclusions.push(ArchitectureRoutingExclusion::MissingResourceEnvelope),
        }

        candidates.push(ArchitectureRoutingCandidate {
            record_id: observation.record_id.clone(),
            architecture_id: observation.architecture_id.clone(),
            exclusions,
        });
    }

    let mut eligible: Vec<&ArchitectureRoutingCandidate> =
        candidates.iter().filter(|candidate| candidate.is_eligible()).collect();
    eligible.sort_by(|left, right| compare_candidates(left, right, observations, request.priority));

    let eligible_record_ids: Vec<String> =
        eligible.iter().map(|candidate| candidate.record_id.clone()).collect();
    let selected = eligible.first();
    let selected_record_id = selected.map(|candidate| candidate.record_id.clone());
    let selected_architecture_id = selected.map(|candidate| candidate.architecture_id.clone());
    let abstention = if selected.is_none() {
        Some(ArchitectureRoutingAbstention::NoEligibleCandidate)
    } else {
        None
    };

    Ok(ArchitectureRoutingReport {
        candidates,
        eligible_record_ids,
        selected_record_id,
        selected_architecture_id,
        abstention,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture_portfolio::{
        ArchitectureResourceEnvelope, ComputeArchitectureFamily, MemoryInterval, MetricInterval,
    };

    fn interval(lower: f64, upper: f64) -> MetricInterval {
        MetricInterval { lower, upper }
    }

    fn observation(
        record_id: &str,
        architecture_id: &str,
        family: ComputeArchitectureFamily,
        time: (f64, f64),
        energy: (f64, f64),
        quality: (f64, f64),
    ) -> ArchitectureBenchmarkObservation {
        ArchitectureBenchmarkObservation {
            record_id: record_id.to_owned(),
            architecture_id: architecture_id.to_owned(),
            architecture_family: family,
            workload_kind: ArchitectureWorkloadKind::SparseEventPerception,
            workload_fingerprint: "event-stream-sha256".to_owned(),
            comparison_contract_fingerprint: "sensor-to-decision-contract".to_owned(),
            evidence_class: ArchitectureEvidenceClass::PhysicalPrototype,
            measurement_boundary: ArchitectureMeasurementBoundary::SensorToDecision,
            quality: interval(quality.0, quality.1),
            resources: Some(ArchitectureResourceEnvelope {
                wall_time_seconds: interval(time.0, time.1),
                energy_joules: interval(energy.0, energy.1),
                monetary_cost_usd: interval(0.01, 0.02),
                peak_memory: MemoryInterval { lower_bytes: 1024, upper_bytes: 2048 },
            }),
        }
    }

    fn request(priority: ArchitectureRoutingPriority) -> ArchitectureRoutingRequest {
        ArchitectureRoutingRequest {
            workload_kind: ArchitectureWorkloadKind::SparseEventPerception,
            workload_fingerprint: "event-stream-sha256".to_owned(),
            comparison_contract_fingerprint: "sensor-to-decision-contract".to_owned(),
            evidence_class: ArchitectureEvidenceClass::PhysicalPrototype,
            measurement_boundary: ArchitectureMeasurementBoundary::SensorToDecision,
            minimum_quality: 0.90,
            resource_limits: ArchitectureResourceLimits {
                max_wall_time_seconds: 1.0,
                max_energy_joules: 10.0,
                max_monetary_cost_usd: 0.10,
                max_peak_memory_bytes: 4096,
            },
            priority,
        }
    }

    fn two_candidates() -> Vec<ArchitectureBenchmarkObservation> {
        vec![
            observation("slow-low-energy", "fpga-a", ComputeArchitectureFamily::Fpga,
                (0.70, 0.80), (2.0, 3.0), (0.94, 0.96)),
            observation("fast-high-energy", "neuromorphic-b", ComputeArchitectureFamily::NeuromorphicDigital,
                (0.30, 0.40), (5.0, 6.0), (0.93, 0.95)),
        ]
    }

    #[test]
    fn selects_lowest_wall_time_when_all_constraints_are_proven() {
        let report = select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestWallTime), &two_candidates()).unwrap();
        assert_eq!(report.selected_record_id.as_deref(), Some("fast-high-energy"));
        assert_eq!(report.eligible_record_ids, vec!["fast-high-energy", "slow-low-energy"]);
        assert_eq!(report.abstention, None);
    }

    #[test]
    fn explicit_energy_priority_can_choose_a_different_architecture() {
        let report = select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestEnergy), &two_candidates()).unwrap();
        assert_eq!(report.selected_record_id.as_deref(), Some("slow-low-energy"));
    }

    #[test]
    fn quality_interval_must_prove_the_minimum() {
        let mut candidate = two_candidates().remove(0);
        candidate.quality = interval(0.89, 0.92);
        let report = select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestWallTime), &[candidate]).unwrap();
        assert_eq!(report.selected_record_id, None);
        assert_eq!(report.candidates[0].exclusions, vec![ArchitectureRoutingExclusion::QualityNotProven]);
    }

    #[test]
    fn quality_entirely_below_minimum_is_a_definite_exclusion() {
        let mut candidate = two_candidates().remove(0);
        candidate.quality = interval(0.80, 0.89);
        let report = select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestWallTime), &[candidate]).unwrap();
        assert_eq!(report.candidates[0].exclusions, vec![ArchitectureRoutingExclusion::QualityBelowThreshold]);
    }

    #[test]
    fn resource_interval_must_fit_entirely_below_ceiling() {
        let mut candidate = two_candidates().remove(0);
        candidate.resources.as_mut().unwrap().wall_time_seconds = interval(0.80, 1.10);
        let report = select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestWallTime), &[candidate]).unwrap();
        assert_eq!(report.candidates[0].exclusions, vec![ArchitectureRoutingExclusion::WallTimeNotProven]);
    }

    #[test]
    fn definite_resource_violation_is_distinguished_from_uncertainty() {
        let mut candidate = two_candidates().remove(0);
        candidate.resources.as_mut().unwrap().wall_time_seconds = interval(1.10, 1.20);
        let report = select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestWallTime), &[candidate]).unwrap();
        assert_eq!(report.candidates[0].exclusions, vec![ArchitectureRoutingExclusion::WallTimeLimitExceeded]);
    }

    #[test]
    fn workload_mismatch_is_never_routed() {
        let mut candidate = two_candidates().remove(0);
        candidate.workload_fingerprint = "different-event-stream".to_owned();
        let report = select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestWallTime), &[candidate]).unwrap();
        assert!(report.candidates[0].exclusions.contains(&ArchitectureRoutingExclusion::WorkloadMismatch));
        assert_eq!(report.selected_record_id, None);
    }

    #[test]
    fn contract_and_evidence_mismatches_are_excluded() {
        let mut candidate = two_candidates().remove(0);
        candidate.comparison_contract_fingerprint = "different-contract".to_owned();
        candidate.evidence_class = ArchitectureEvidenceClass::Simulation;
        let report = select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestWallTime), &[candidate]).unwrap();
        assert!(report.candidates[0].exclusions.contains(&ArchitectureRoutingExclusion::ContractMismatch));
        assert!(report.candidates[0].exclusions.contains(&ArchitectureRoutingExclusion::EvidenceClassMismatch));
    }

    #[test]
    fn measurement_boundary_mismatch_is_excluded() {
        let mut candidate = two_candidates().remove(0);
        candidate.measurement_boundary = ArchitectureMeasurementBoundary::RuntimeEndToEnd;
        let report = select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestWallTime), &[candidate]).unwrap();
        assert!(report.candidates[0].exclusions.contains(&ArchitectureRoutingExclusion::MeasurementBoundaryMismatch));
    }

    #[test]
    fn algorithm_only_request_is_rejected() {
        let mut invalid = request(ArchitectureRoutingPriority::LowestWallTime);
        invalid.measurement_boundary = ArchitectureMeasurementBoundary::AlgorithmOnly;
        assert_eq!(
            select_architecture_candidate(&invalid, &[]),
            Err(ArchitectureRoutingError::AlgorithmOnlyRequest)
        );
    }

    #[test]
    fn malformed_candidate_is_reported_not_selected() {
        let mut candidate = two_candidates().remove(0);
        candidate.quality = interval(f64::NAN, 0.95);
        let report = select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestWallTime), &[candidate]).unwrap();
        assert!(matches!(report.candidates[0].exclusions.as_slice(),
            [ArchitectureRoutingExclusion::InvalidObservation(ArchitectureComparisonError::InvalidInterval)]));
        assert_eq!(report.selected_record_id, None);
    }

    #[test]
    fn duplicate_record_ids_are_rejected() {
        let mut candidates = two_candidates();
        candidates[1].record_id = candidates[0].record_id.clone();
        assert_eq!(
            select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestWallTime), &candidates),
            Err(ArchitectureRoutingError::DuplicateRecordId)
        );
    }

    #[test]
    fn empty_or_ineligible_portfolio_abstains() {
        let report = select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestWallTime), &[]).unwrap();
        assert_eq!(report.selected_record_id, None);
        assert_eq!(report.abstention, Some(ArchitectureRoutingAbstention::NoEligibleCandidate));
    }

    #[test]
    fn ties_are_deterministic_by_architecture_then_record_identity() {
        let mut candidates = two_candidates();
        candidates[1].resources = candidates[0].resources;
        candidates[0].architecture_id = "same-family-a".to_owned();
        candidates[1].architecture_id = "same-family-b".to_owned();
        let report = select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestWallTime), &candidates).unwrap();
        assert_eq!(report.selected_record_id.as_deref(), Some("slow-low-energy"));
    }

    #[test]
    fn missing_system_resources_exclude_candidate() {
        let mut candidate = two_candidates().remove(0);
        candidate.resources = None;
        let report = select_architecture_candidate(&request(ArchitectureRoutingPriority::LowestWallTime), &[candidate]).unwrap();
        assert_eq!(report.candidates[0].exclusions, vec![
            ArchitectureRoutingExclusion::InvalidObservation(ArchitectureComparisonError::MissingResourceEnvelope)
        ]);
    }

    #[test]
    fn invalid_limits_and_non_finite_quality_threshold_are_rejected() {
        let mut invalid = request(ArchitectureRoutingPriority::LowestWallTime);
        invalid.resource_limits.max_energy_joules = f64::INFINITY;
        assert_eq!(
            select_architecture_candidate(&invalid, &[]),
            Err(ArchitectureRoutingError::InvalidResourceLimit)
        );
        let mut invalid = request(ArchitectureRoutingPriority::LowestWallTime);
        invalid.minimum_quality = f64::NAN;
        assert_eq!(
            select_architecture_candidate(&invalid, &[]),
            Err(ArchitectureRoutingError::InvalidQualityThreshold)
        );
    }
}
