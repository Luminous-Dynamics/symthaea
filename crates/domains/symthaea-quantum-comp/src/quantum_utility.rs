//! End-to-end quantum utility crossover gate.
//!
//! The model compares the resource required to obtain a verified, acceptable
//! answer against a matched classical baseline. It is deliberately backend-
//! agnostic: photonic and stateful-fusion estimates can feed the attempt-cycle
//! envelope, while measured or modeled physics enters through explicit bounds.
//!
//! This is not a quantum advantage detector. It only compares declared bounds.
//! In particular, model-only dominance is a hypothesis, not empirical evidence.

use std::error::Error;
use std::fmt;

/// A finite, ordered interval for a non-negative resource quantity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonNegativeInterval {
    lower: f64,
    upper: f64,
}

impl NonNegativeInterval {
    /// Create a finite interval with non-negative endpoints.
    pub fn new(lower: f64, upper: f64) -> Result<Self, QuantumUtilityError> {
        if !lower.is_finite()
            || !upper.is_finite()
            || lower < 0.0
            || upper < lower
        {
            return Err(QuantumUtilityError::InvalidInterval);
        }
        Ok(Self { lower, upper })
    }

    /// Exact interval for a finite non-negative value.
    pub fn point(value: f64) -> Result<Self, QuantumUtilityError> {
        Self::new(value, value)
    }

    /// Conservative lower endpoint.
    pub fn lower(self) -> f64 {
        self.lower
    }

    /// Conservative upper endpoint.
    pub fn upper(self) -> f64 {
        self.upper
    }
}

/// Interval for a probability in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbabilityInterval {
    lower: f64,
    upper: f64,
}

impl ProbabilityInterval {
    /// Create a valid probability interval.
    pub fn new(lower: f64, upper: f64) -> Result<Self, QuantumUtilityError> {
        if !lower.is_finite()
            || !upper.is_finite()
            || lower < 0.0
            || upper > 1.0
            || upper < lower
        {
            return Err(QuantumUtilityError::InvalidProbability);
        }
        Ok(Self { lower, upper })
    }

    /// Exact interval for a probability.
    pub fn point(value: f64) -> Result<Self, QuantumUtilityError> {
        Self::new(value, value)
    }

    /// Conservative lower endpoint.
    pub fn lower(self) -> f64 {
        self.lower
    }

    /// Conservative upper endpoint.
    pub fn upper(self) -> f64 {
        self.upper
    }
}

/// Resource objective being compared. Run one comparison per objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceMetric {
    /// End-to-end wall-clock seconds per verified acceptable answer.
    WallClockSeconds,
    /// End-to-end joules per verified acceptable answer.
    EnergyJoules,
    /// End-to-end cost in US dollars per verified acceptable answer.
    CostUsd,
}

/// Provenance class for the inputs used to construct a candidate envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceClass {
    /// Measurements from a named backend and declared run configuration.
    MeasuredBackend,
    /// Simulator calibrated against documented physical measurements.
    CalibratedSimulation,
    /// Analytic estimate without a calibrated end-to-end simulator.
    AnalyticModel,
    /// Synthetic fixture used to test logic, not to characterize a device.
    SyntheticFixture,
}

/// Whether a candidate's quality interval clears a predeclared threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityGate {
    /// The entire quality interval reaches or exceeds the threshold.
    Meets,
    /// Even the upper quality bound is below the threshold.
    Fails,
    /// The interval straddles the threshold.
    Uncertain,
}

/// Expected resource interval per accepted solution under independent retries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExpectedResource {
    /// Lower bound is finite; upper is absent when success may be arbitrarily rare.
    Reachable {
        lower: f64,
        upper: Option<f64>,
    },
    /// The declared upper bound on accepted-answer probability is zero.
    Unreachable,
}

/// Expected resource and quality gate for one candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateEstimate {
    /// Expected resource per verified acceptable answer.
    pub expected_resource: ExpectedResource,
    /// Result of the shared quality threshold.
    pub quality_gate: QualityGate,
    /// Evidence provenance carried from the input manifest.
    pub evidence: EvidenceClass,
}

/// A candidate pipeline's declared end-to-end resource bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateProfile {
    workload_fingerprint: String,
    metric: ResourceMetric,
    setup_cost: NonNegativeInterval,
    attempt_cycle_cost: NonNegativeInterval,
    accepted_answer_probability: ProbabilityInterval,
    output_quality: ProbabilityInterval,
    evidence: EvidenceClass,
}

impl CandidateProfile {
    /// Create a candidate profile.
    ///
    /// setup_cost is the one-time cost for one requested problem instance.
    /// attempt_cycle_cost must contain all recurring end-to-end work for one
    /// statistically independent attempt, including verification. The
    /// accepted-answer probability must include the declared acceptance and
    /// verifier criteria. These inputs are conservative envelopes, not
    /// confidence intervals unless the manifest says how they were derived.
    pub fn new(
        workload_fingerprint: impl Into<String>,
        metric: ResourceMetric,
        setup_cost: NonNegativeInterval,
        attempt_cycle_cost: NonNegativeInterval,
        accepted_answer_probability: ProbabilityInterval,
        output_quality: ProbabilityInterval,
        evidence: EvidenceClass,
    ) -> Result<Self, QuantumUtilityError> {
        let workload_fingerprint = workload_fingerprint.into();
        if workload_fingerprint.trim().is_empty() {
            return Err(QuantumUtilityError::EmptyWorkloadFingerprint);
        }
        Ok(Self {
            workload_fingerprint,
            metric,
            setup_cost,
            attempt_cycle_cost,
            accepted_answer_probability,
            output_quality,
            evidence,
        })
    }

    /// Workload identity that must match the comparison partner.
    pub fn workload_fingerprint(&self) -> &str {
        &self.workload_fingerprint
    }

    /// Resource objective used by this profile.
    pub fn metric(&self) -> ResourceMetric {
        self.metric
    }

    /// Compute conservative expected cost per accepted answer.
    ///
    /// The calculation assumes independent attempts with stationary
    /// acceptance probability. Adaptive retry processes should first be
    /// summarized by the bounded fusion-process model; correlated retries
    /// require a different model and must not be forced into this formula.
    pub fn estimate(&self, quality_threshold: f64) -> Result<CandidateEstimate, QuantumUtilityError> {
        validate_quality_threshold(quality_threshold)?;
        let quality_gate = quality_gate(self.output_quality, quality_threshold);
        let p = self.accepted_answer_probability;

        let expected_resource = if p.upper == 0.0 {
            ExpectedResource::Unreachable
        } else {
            let lower = self.setup_cost.lower + self.attempt_cycle_cost.lower / p.upper;
            if !lower.is_finite() {
                return Err(QuantumUtilityError::DerivedValueOverflow);
            }
            let upper = if p.lower == 0.0 {
                None
            } else {
                let value =
                    self.setup_cost.upper + self.attempt_cycle_cost.upper / p.lower;
                if !value.is_finite() {
                    return Err(QuantumUtilityError::DerivedValueOverflow);
                }
                Some(value)
            };
            ExpectedResource::Reachable { lower, upper }
        };

        Ok(CandidateEstimate {
            expected_resource,
            quality_gate,
            evidence: self.evidence,
        })
    }
}

/// Outcome based only on declared bounds and the common quality threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtilityDecision {
    /// Quantum's resource upper bound is below the classical lower bound.
    QuantumLowerCostWithinDeclaredBounds,
    /// Classical's resource upper bound is below the quantum lower bound.
    ClassicalLowerCostWithinDeclaredBounds,
    /// Both candidates declare zero probability of an accepted answer.
    NeitherCandidateCanReturnAnAcceptedAnswer,
    /// The quantum candidate fails the common quality threshold.
    QuantumFailsQualityGate,
    /// The classical candidate fails the common quality threshold.
    ClassicalFailsQualityGate,
    /// Neither candidate passes the quality gate.
    NeitherCandidatePassesQualityGate,
    /// Bounds overlap, are open-ended, or otherwise cannot establish dominance.
    Inconclusive,
}

/// Evidence summary for the pair; distinct from the numeric decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceAssessment {
    /// Both sides are measured on named backends under the declared protocol.
    BothMeasured,
    /// At least one side is a synthetic fixture.
    ContainsSyntheticFixture,
    /// Neither side is measured on a backend.
    ModelOnly,
    /// One side is measured and the other is not.
    MixedEvidence,
}

/// Reproducible comparison result for a single workload and objective.
#[derive(Debug, Clone, PartialEq)]
pub struct UtilityComparisonReport {
    /// Fingerprint shared by both candidate profiles.
    pub workload_fingerprint: String,
    /// Resource objective that was compared.
    pub metric: ResourceMetric,
    /// Common minimum output quality.
    pub quality_threshold: f64,
    /// Quantum candidate estimate.
    pub quantum: CandidateEstimate,
    /// Classical candidate estimate.
    pub classical: CandidateEstimate,
    /// Decision from conservative declared bounds.
    pub decision: UtilityDecision,
    /// Evidence class of the pair.
    pub evidence_assessment: EvidenceAssessment,
}

/// Errors from invalid utility manifests or non-finite calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumUtilityError {
    /// Bounds were non-finite, negative, or unordered.
    InvalidInterval,
    /// Probability bounds did not lie within [0, 1].
    InvalidProbability,
    /// Quality threshold did not lie within [0, 1].
    InvalidQualityThreshold,
    /// Workload fingerprint was empty.
    EmptyWorkloadFingerprint,
    /// Candidate workload fingerprints differed.
    WorkloadMismatch,
    /// Candidate resource metrics differed.
    MetricMismatch,
    /// A derived expected-resource endpoint overflowed.
    DerivedValueOverflow,
}

impl fmt::Display for QuantumUtilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidInterval => "invalid non-negative interval",
            Self::InvalidProbability => "invalid probability interval",
            Self::InvalidQualityThreshold => "invalid quality threshold",
            Self::EmptyWorkloadFingerprint => "empty workload fingerprint",
            Self::WorkloadMismatch => "candidate workloads do not match",
            Self::MetricMismatch => "candidate resource metrics do not match",
            Self::DerivedValueOverflow => "derived expected resource overflowed",
        };
        f.write_str(message)
    }
}

impl Error for QuantumUtilityError {}

/// Compare a quantum pipeline with its strongest credible classical baseline.
pub fn compare_utility(
    quantum: &CandidateProfile,
    classical: &CandidateProfile,
    quality_threshold: f64,
) -> Result<UtilityComparisonReport, QuantumUtilityError> {
    validate_quality_threshold(quality_threshold)?;
    if quantum.workload_fingerprint != classical.workload_fingerprint {
        return Err(QuantumUtilityError::WorkloadMismatch);
    }
    if quantum.metric != classical.metric {
        return Err(QuantumUtilityError::MetricMismatch);
    }

    let quantum_estimate = quantum.estimate(quality_threshold)?;
    let classical_estimate = classical.estimate(quality_threshold)?;
    let decision = decide(
        quantum_estimate,
        classical_estimate,
    );

    Ok(UtilityComparisonReport {
        workload_fingerprint: quantum.workload_fingerprint.clone(),
        metric: quantum.metric,
        quality_threshold,
        quantum: quantum_estimate,
        classical: classical_estimate,
        decision,
        evidence_assessment: assess_evidence(quantum.evidence, classical.evidence),
    })
}

fn validate_quality_threshold(value: f64) -> Result<(), QuantumUtilityError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(QuantumUtilityError::InvalidQualityThreshold);
    }
    Ok(())
}

fn quality_gate(quality: ProbabilityInterval, threshold: f64) -> QualityGate {
    if quality.lower >= threshold {
        QualityGate::Meets
    } else if quality.upper < threshold {
        QualityGate::Fails
    } else {
        QualityGate::Uncertain
    }
}

fn decide(quantum: CandidateEstimate, classical: CandidateEstimate) -> UtilityDecision {
    use ExpectedResource::{Reachable, Unreachable};
    use QualityGate::{Fails, Meets, Uncertain};
    use UtilityDecision::*;

    match (quantum.quality_gate, classical.quality_gate) {
        (Fails, Fails) => return NeitherCandidatePassesQualityGate,
        (Fails, _) => return QuantumFailsQualityGate,
        (_, Fails) => return ClassicalFailsQualityGate,
        (Uncertain, _) | (_, Uncertain) => return Inconclusive,
        (Meets, Meets) => {}
    }

    match (quantum.expected_resource, classical.expected_resource) {
        (Unreachable, Unreachable) => NeitherCandidateCanReturnAnAcceptedAnswer,
        (Reachable { .. }, Unreachable) => QuantumLowerCostWithinDeclaredBounds,
        (Unreachable, Reachable { .. }) => ClassicalLowerCostWithinDeclaredBounds,
        (
            Reachable { upper: Some(q_upper), .. },
            Reachable { lower: c_lower, .. },
        ) if q_upper < c_lower => QuantumLowerCostWithinDeclaredBounds,
        (
            Reachable { lower: q_lower, .. },
            Reachable { upper: Some(c_upper), .. },
        ) if c_upper < q_lower => ClassicalLowerCostWithinDeclaredBounds,
        _ => Inconclusive,
    }
}

fn assess_evidence(quantum: EvidenceClass, classical: EvidenceClass) -> EvidenceAssessment {
    use EvidenceAssessment::*;
    use EvidenceClass::*;
    if quantum == MeasuredBackend && classical == MeasuredBackend {
        BothMeasured
    } else if quantum == SyntheticFixture || classical == SyntheticFixture {
        ContainsSyntheticFixture
    } else if quantum != MeasuredBackend && classical != MeasuredBackend {
        ModelOnly
    } else {
        MixedEvidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(lower: f64, upper: f64) -> NonNegativeInterval {
        NonNegativeInterval::new(lower, upper).unwrap()
    }

    fn probability(lower: f64, upper: f64) -> ProbabilityInterval {
        ProbabilityInterval::new(lower, upper).unwrap()
    }

    fn candidate(
        workload: &str,
        metric: ResourceMetric,
        setup: (f64, f64),
        cycle: (f64, f64),
        success: (f64, f64),
        quality: (f64, f64),
        evidence: EvidenceClass,
    ) -> CandidateProfile {
        CandidateProfile::new(
            workload,
            metric,
            interval(setup.0, setup.1),
            interval(cycle.0, cycle.1),
            probability(success.0, success.1),
            probability(quality.0, quality.1),
            evidence,
        )
        .unwrap()
    }

    #[test]
    fn expected_cost_includes_setup_and_retry_factor() {
        let candidate = candidate(
            "chemistry-small-v1",
            ResourceMetric::WallClockSeconds,
            (2.0, 2.0),
            (10.0, 10.0),
            (0.5, 0.5),
            (0.99, 0.99),
            EvidenceClass::SyntheticFixture,
        );
        assert_eq!(
            candidate.estimate(0.95).unwrap().expected_resource,
            ExpectedResource::Reachable {
                lower: 22.0,
                upper: Some(22.0),
            }
        );
    }

    #[test]
    fn separated_bounds_can_show_quantum_cost_dominance() {
        let quantum = candidate(
            "material-observable-v1",
            ResourceMetric::CostUsd,
            (1.0, 2.0),
            (1.0, 2.0),
            (0.9, 1.0),
            (0.99, 1.0),
            EvidenceClass::AnalyticModel,
        );
        let classical = candidate(
            "material-observable-v1",
            ResourceMetric::CostUsd,
            (8.0, 9.0),
            (5.0, 6.0),
            (0.5, 0.6),
            (0.98, 1.0),
            EvidenceClass::AnalyticModel,
        );
        let report = compare_utility(&quantum, &classical, 0.95).unwrap();
        assert_eq!(
            report.decision,
            UtilityDecision::QuantumLowerCostWithinDeclaredBounds
        );
        assert_eq!(report.evidence_assessment, EvidenceAssessment::ModelOnly);
    }

    #[test]
    fn overlapping_bounds_are_inconclusive() {
        let quantum = candidate(
            "same-task",
            ResourceMetric::EnergyJoules,
            (1.0, 4.0),
            (4.0, 8.0),
            (0.5, 0.9),
            (0.98, 1.0),
            EvidenceClass::MeasuredBackend,
        );
        let classical = candidate(
            "same-task",
            ResourceMetric::EnergyJoules,
            (2.0, 3.0),
            (5.0, 7.0),
            (0.7, 0.9),
            (0.98, 1.0),
            EvidenceClass::MeasuredBackend,
        );
        let report = compare_utility(&quantum, &classical, 0.95).unwrap();
        assert_eq!(report.decision, UtilityDecision::Inconclusive);
        assert_eq!(report.evidence_assessment, EvidenceAssessment::BothMeasured);
    }

    #[test]
    fn quality_failure_blocks_resource_dominance() {
        let quantum = candidate(
            "task",
            ResourceMetric::WallClockSeconds,
            (0.1, 0.1),
            (0.1, 0.1),
            (1.0, 1.0),
            (0.6, 0.8),
            EvidenceClass::MeasuredBackend,
        );
        let classical = candidate(
            "task",
            ResourceMetric::WallClockSeconds,
            (10.0, 10.0),
            (10.0, 10.0),
            (1.0, 1.0),
            (0.99, 1.0),
            EvidenceClass::MeasuredBackend,
        );
        let report = compare_utility(&quantum, &classical, 0.95).unwrap();
        assert_eq!(report.decision, UtilityDecision::QuantumFailsQualityGate);
    }

    #[test]
    fn zero_lower_probability_keeps_upper_cost_open() {
        let quantum = candidate(
            "task",
            ResourceMetric::CostUsd,
            (0.0, 0.0),
            (1.0, 2.0),
            (0.0, 0.9),
            (0.99, 1.0),
            EvidenceClass::AnalyticModel,
        );
        let estimate = quantum.estimate(0.95).unwrap();
        assert_eq!(
            estimate.expected_resource,
            ExpectedResource::Reachable {
                lower: 1.0 / 0.9,
                upper: None,
            }
        );
        let classical = candidate(
            "task",
            ResourceMetric::CostUsd,
            (0.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.5),
            (0.99, 1.0),
            EvidenceClass::AnalyticModel,
        );
        assert_eq!(
            compare_utility(&quantum, &classical, 0.95)
                .unwrap()
                .decision,
            UtilityDecision::Inconclusive
        );
    }

    #[test]
    fn zero_success_probability_cannot_win_unfairly() {
        let quantum = candidate(
            "task",
            ResourceMetric::CostUsd,
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.99, 1.0),
            EvidenceClass::SyntheticFixture,
        );
        let classical = candidate(
            "task",
            ResourceMetric::CostUsd,
            (0.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.5),
            (0.99, 1.0),
            EvidenceClass::MeasuredBackend,
        );
        assert_eq!(
            compare_utility(&quantum, &classical, 0.95)
                .unwrap()
                .decision,
            UtilityDecision::ClassicalLowerCostWithinDeclaredBounds
        );
    }

    #[test]
    fn both_unreachable_candidates_do_not_produce_dominance() {
        let quantum = candidate(
            "task",
            ResourceMetric::CostUsd,
            (1.0, 1.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (0.99, 1.0),
            EvidenceClass::AnalyticModel,
        );
        let classical = quantum.clone();
        assert_eq!(
            compare_utility(&quantum, &classical, 0.95)
                .unwrap()
                .decision,
            UtilityDecision::NeitherCandidateCanReturnAnAcceptedAnswer
        );
    }

    #[test]
    fn workload_and_metric_mismatches_fail_closed() {
        let quantum = candidate(
            "task-a",
            ResourceMetric::CostUsd,
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            EvidenceClass::MeasuredBackend,
        );
        let wrong_workload = candidate(
            "task-b",
            ResourceMetric::CostUsd,
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            EvidenceClass::MeasuredBackend,
        );
        assert_eq!(
            compare_utility(&quantum, &wrong_workload, 0.95),
            Err(QuantumUtilityError::WorkloadMismatch)
        );

        let wrong_metric = candidate(
            "task-a",
            ResourceMetric::EnergyJoules,
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            EvidenceClass::MeasuredBackend,
        );
        assert_eq!(
            compare_utility(&quantum, &wrong_metric, 0.95),
            Err(QuantumUtilityError::MetricMismatch)
        );
    }

    #[test]
    fn invalid_bounds_and_thresholds_are_rejected() {
        assert_eq!(
            NonNegativeInterval::new(2.0, 1.0),
            Err(QuantumUtilityError::InvalidInterval)
        );
        assert_eq!(
            ProbabilityInterval::new(0.0, 1.1),
            Err(QuantumUtilityError::InvalidProbability)
        );
        let profile = candidate(
            "task",
            ResourceMetric::CostUsd,
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            EvidenceClass::MeasuredBackend,
        );
        assert_eq!(
            profile.estimate(f64::NAN),
            Err(QuantumUtilityError::InvalidQualityThreshold)
        );
        assert_eq!(
            CandidateProfile::new(
                " ",
                ResourceMetric::CostUsd,
                interval(0.0, 0.0),
                interval(0.0, 0.0),
                probability(1.0, 1.0),
                probability(1.0, 1.0),
                EvidenceClass::MeasuredBackend,
            ),
            Err(QuantumUtilityError::EmptyWorkloadFingerprint)
        );
    }

    #[test]
    fn derived_overflow_is_an_error_not_a_false_result() {
        let profile = candidate(
            "task",
            ResourceMetric::CostUsd,
            (0.0, 0.0),
            (f64::MAX, f64::MAX),
            (f64::MIN_POSITIVE, f64::MIN_POSITIVE),
            (1.0, 1.0),
            EvidenceClass::AnalyticModel,
        );
        assert_eq!(
            profile.estimate(0.95),
            Err(QuantumUtilityError::DerivedValueOverflow)
        );
    }

    #[test]
    fn zero_quality_probability_is_valid_but_does_not_meet_positive_threshold() {
        let quality = probability(0.0, 0.0);
        assert_eq!(quality_gate(quality, 0.01), QualityGate::Fails);
    }
}
