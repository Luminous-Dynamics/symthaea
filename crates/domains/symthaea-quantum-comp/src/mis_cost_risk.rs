//! Cost-sensitive paired evaluation for quantum-informed optimization experiments.
//!
//! Quality, feasibility, target attainment and resource usage stay separate until a
//! preregistered risk configuration combines them. This module is not a QPU adapter and
//! cannot attest that caller-supplied costs were measured honestly.
//!
//! Each active resource dimension is divided by a declared reference amount before it
//! is weighted. Do not tune scales or weights after seeing confirmatory results. Wall
//! time cannot be weighted together with its constituent stage timings: that double
//! counts elapsed time.

use core::fmt;
use std::collections::BTreeMap;

use crate::significance::PairedDifferenceSummary;

/// Operational-cost dimension recorded by a benchmark run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MisCostMetric {
    /// Candidate-level observations, not necessarily physical device executions.
    CandidateObservations,
    /// Unique physical device shots.
    PhysicalShots,
    /// Unique submitted/executed circuit batches.
    CircuitBatches,
    /// Circuit compile/transpile time in milliseconds.
    CompilationMillis,
    /// Calibration overhead in milliseconds.
    CalibrationMillis,
    /// Classical preprocessing time in milliseconds.
    PreprocessingMillis,
    /// Classical solver/inference CPU/GPU execution time in milliseconds.
    ClassicalComputeMillis,
    /// Queue/wait time in milliseconds, reported independently of active execution.
    QueueMillis,
    /// End-to-end elapsed wall-clock time in milliseconds.
    WallClockMillis,
    /// Measured or explicitly estimated energy in millijoules.
    EnergyMillijoules,
    /// Direct monetary cost in millionths of a named three-letter currency.
    MoneyMicrounits,
}

impl MisCostMetric {
    /// Stable machine-readable label used in reports and manifests.
    pub const fn label(self) -> &'static str {
        match self {
            Self::CandidateObservations => "candidate_observations",
            Self::PhysicalShots => "physical_shots",
            Self::CircuitBatches => "circuit_batches",
            Self::CompilationMillis => "compilation_ms",
            Self::CalibrationMillis => "calibration_ms",
            Self::PreprocessingMillis => "preprocessing_ms",
            Self::ClassicalComputeMillis => "classical_compute_ms",
            Self::QueueMillis => "queue_ms",
            Self::WallClockMillis => "wall_clock_ms",
            Self::EnergyMillijoules => "energy_mj",
            Self::MoneyMicrounits => "money_microunits",
        }
    }
}

/// Explicit evidence class for a run; a policy comparison may compare classes, but may not
/// silently aggregate mixed classes under one policy identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MisEvidenceClass {
    /// Seeded independent-arm policy simulation or synthetic fixture.
    SyntheticArms,
    /// Replayed, preserved outputs from a named existing artifact/dataset.
    ClassicalReplay,
    /// Executed classical algorithm (CPU or GPU).
    ClassicalSolver,
    /// Ideal quantum-circuit simulator without hardware noise.
    IdealQuantumSimulator,
    /// Tensor-network or other approximate quantum-state simulation.
    TensorNetworkSimulator,
    /// Circuit was exported but not executed by a backend.
    CircuitExportOnly,
    /// A physical quantum processing unit actually executed the circuit.
    PhysicalQpu,
}

impl MisEvidenceClass {
    /// Stable machine-readable label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::SyntheticArms => "synthetic_arms",
            Self::ClassicalReplay => "classical_replay",
            Self::ClassicalSolver => "classical_solver",
            Self::IdealQuantumSimulator => "ideal_quantum_simulator",
            Self::TensorNetworkSimulator => "tensor_network_simulator",
            Self::CircuitExportOnly => "circuit_export_only",
            Self::PhysicalQpu => "physical_qpu",
        }
    }
}

/// Distinct cost dimensions for one run.
///
/// Timings may overlap and are not automatically additive. Use end-to-end wall time
/// or selected stage timings in a scalar risk, not both. None means an optional energy
/// or monetary metric was not recorded; it does not mean zero.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MisOperationalCost {
    /// Candidate observations consumed; None means not recorded.
    pub candidate_observations: Option<u64>,
    /// Unique physical device shots consumed; None means not recorded.
    pub physical_shots: Option<u64>,
    /// Unique circuit batches/jobs; None means not recorded.
    pub circuit_batches: Option<u64>,
    /// Circuit compilation/transpilation time in milliseconds; None means not recorded.
    pub compilation_millis: Option<u64>,
    /// Calibration overhead in milliseconds; None means not recorded.
    pub calibration_millis: Option<u64>,
    /// Classical preprocessing time in milliseconds; None means not recorded.
    pub preprocessing_millis: Option<u64>,
    /// Classical solver/inference execution time in milliseconds; None means not recorded.
    pub classical_compute_millis: Option<u64>,
    /// Queue/wait time in milliseconds; None means not recorded.
    pub queue_millis: Option<u64>,
    /// End-to-end elapsed wall-clock time in milliseconds; None means not recorded.
    pub wall_clock_millis: Option<u64>,
    /// Energy in millijoules, if measured or explicitly estimated.
    pub energy_millijoules: Option<u64>,
    /// Direct cost in millionths of currency_code, if recorded.
    pub money_microunits: Option<u64>,
    /// Three-letter uppercase currency code for money_microunits.
    pub currency_code: Option<String>,
}

impl MisOperationalCost {
    /// Returns an amount, or None if that optional dimension was not recorded.
    pub fn amount(&self, metric: MisCostMetric) -> Option<u64> {
        match metric {
            MisCostMetric::CandidateObservations => self.candidate_observations,
            MisCostMetric::PhysicalShots => self.physical_shots,
            MisCostMetric::CircuitBatches => self.circuit_batches,
            MisCostMetric::CompilationMillis => self.compilation_millis,
            MisCostMetric::CalibrationMillis => self.calibration_millis,
            MisCostMetric::PreprocessingMillis => self.preprocessing_millis,
            MisCostMetric::ClassicalComputeMillis => self.classical_compute_millis,
            MisCostMetric::QueueMillis => self.queue_millis,
            MisCostMetric::WallClockMillis => self.wall_clock_millis,
            MisCostMetric::EnergyMillijoules => self.energy_millijoules,
            MisCostMetric::MoneyMicrounits => self.money_microunits,
        }
    }

    fn validate(&self) -> Result<(), MisCostRiskError> {
        match (self.money_microunits, self.currency_code.as_deref()) {
            (None, None) => Ok(()),
            (Some(_), Some(code))
                if code.len() == 3 && code.bytes().all(|byte| byte.is_ascii_uppercase()) => Ok(()),
            _ => Err(MisCostRiskError::InvalidCurrencyPair),
        }
    }
}

/// One normalized resource-cost term in a preregistered risk function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MisCostWeight {
    /// Cost dimension.
    pub metric: MisCostMetric,
    /// Observed amount divided by this reference amount before weighting; must be positive.
    pub reference_amount: u64,
    /// Nonnegative dimensionless weight.
    pub weight: f64,
}

/// Preregistered conversion from heterogeneous outcomes/costs to scalar decision risk.
///
/// Normalized quality loss supplied in each record must be in [0, 1] and defined before
/// confirmatory evaluation. A fixed penalty applies to infeasibility and target failure.
/// Each cost component contributes weight times observed amount divided by reference.
#[derive(Debug, Clone, PartialEq)]
pub struct MisCostRiskConfig {
    /// Weight on normalized solution-quality loss.
    pub quality_loss_weight: f64,
    /// Penalty for output failing independent feasibility verification.
    pub infeasible_penalty: f64,
    /// Penalty for failing the preregistered solution-quality target.
    pub target_miss_penalty: f64,
    /// Weighted, scale-normalized operational resource dimensions.
    pub costs: Vec<MisCostWeight>,
}

impl MisCostRiskConfig {
    /// Creates and validates a risk configuration.
    pub fn new(
        quality_loss_weight: f64,
        infeasible_penalty: f64,
        target_miss_penalty: f64,
        costs: Vec<MisCostWeight>,
    ) -> Result<Self, MisCostRiskError> {
        let config = Self { quality_loss_weight, infeasible_penalty, target_miss_penalty, costs };
        config.validate()?;
        Ok(config)
    }

    /// Validates weights, scales, duplicate dimensions and time-accounting boundaries.
    pub fn validate(&self) -> Result<(), MisCostRiskError> {
        for (name, value) in [
            ("quality_loss_weight", self.quality_loss_weight),
            ("infeasible_penalty", self.infeasible_penalty),
            ("target_miss_penalty", self.target_miss_penalty),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(MisCostRiskError::InvalidWeight { name });
            }
        }

        let mut seen = Vec::with_capacity(self.costs.len());
        let mut any_positive = self.quality_loss_weight > 0.0
            || self.infeasible_penalty > 0.0 || self.target_miss_penalty > 0.0;
        for component in &self.costs {
            if component.reference_amount == 0 {
                return Err(MisCostRiskError::ZeroReferenceAmount { metric: component.metric });
            }
            if !component.weight.is_finite() || component.weight < 0.0 {
                return Err(MisCostRiskError::InvalidCostComponentWeight { metric: component.metric });
            }
            if seen.contains(&component.metric) {
                return Err(MisCostRiskError::DuplicateCostMetric { metric: component.metric });
            }
            seen.push(component.metric);
            any_positive |= component.weight > 0.0;
        }

        let wall = seen.contains(&MisCostMetric::WallClockMillis);
        let stage_time = [
            MisCostMetric::CompilationMillis, MisCostMetric::CalibrationMillis,
            MisCostMetric::PreprocessingMillis, MisCostMetric::ClassicalComputeMillis,
            MisCostMetric::QueueMillis,
        ].iter().any(|metric| seen.contains(metric));
        if wall && stage_time {
            return Err(MisCostRiskError::OverlappingTimeMetrics);
        }
        if !any_positive {
            return Err(MisCostRiskError::ZeroRiskFunction);
        }
        Ok(())
    }

    fn uses_money(&self) -> bool {
        self.costs.iter().any(|c| c.metric == MisCostMetric::MoneyMicrounits && c.weight > 0.0)
    }
}

/// Caller-supplied identity of the exact workload and decision context.
///
/// All fields are opaque identifiers: this type binds comparisons but does not attest
/// graph provenance. The caller should use canonical graph/configuration digests,
/// a fixed cohort ID, a unique repetition seed, and a verifier version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisWorkloadIdentity {
    /// Frozen cohort/dataset version shared by the entire comparison.
    pub cohort_id: String,
    /// Canonical digest of the original graph instance.
    pub graph_digest: String,
    /// Repetition seed shared by all paired policies.
    pub repetition_seed: u64,
    /// Digest of reducer implementation/configuration.
    pub reducer_config_digest: String,
    /// Digest of candidate-generation/pool policy and ordering.
    pub candidate_pool_digest: String,
    /// Identifier/version of the independent solution verifier.
    pub verifier_version: String,
}

impl MisWorkloadIdentity {
    fn validate(&self) -> Result<(), MisCostRiskError> {
        for (field, value) in [
            ("cohort_id", self.cohort_id.as_str()),
            ("graph_digest", self.graph_digest.as_str()),
            ("reducer_config_digest", self.reducer_config_digest.as_str()),
            ("candidate_pool_digest", self.candidate_pool_digest.as_str()),
            ("verifier_version", self.verifier_version.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(MisCostRiskError::EmptyWorkloadIdentityField { field });
            }
        }
        Ok(())
    }
}

/// One method outcome for a matched workload/repetition.
#[derive(Debug, Clone, PartialEq)]
pub struct MisCostRiskObservation {
    /// Stable key shared by methods on the same instance/repetition.
    pub comparison_key: String,
    /// Full graph/cohort/seed/reducer/candidate-pool/verifier identity for this outcome.
    pub workload: MisWorkloadIdentity,
    /// How this result was produced; never inferred from policy name.
    pub evidence_class: MisEvidenceClass,
    /// Stable reference to raw output, replay artifact, simulator run, or hardware receipt.
    pub evidence_reference: String,
    /// Stable identifier for the evaluated method/policy.
    pub policy_id: String,
    /// Whether the independent feasibility verifier accepted the solution.
    pub feasible: bool,
    /// Whether the preregistered quality target was reached.
    pub target_reached: bool,
    /// Caller-defined normalized quality loss in [0, 1], where lower is better.
    pub normalized_quality_loss: f64,
    /// Full operational cost vector; absent optional values remain absent.
    pub cost: MisOperationalCost,
}

impl MisCostRiskObservation {
    /// Validates identity, normalized loss and optional cost metadata.
    pub fn validate(&self) -> Result<(), MisCostRiskError> {
        if self.comparison_key.trim().is_empty() {
            return Err(MisCostRiskError::EmptyComparisonKey);
        }
        if self.policy_id.trim().is_empty() {
            return Err(MisCostRiskError::EmptyPolicyId);
        }
        self.workload.validate()?;
        if self.evidence_reference.trim().is_empty() {
            return Err(MisCostRiskError::EmptyEvidenceReference);
        }
        if !self.normalized_quality_loss.is_finite()
            || !(0.0..=1.0).contains(&self.normalized_quality_loss) {
            return Err(MisCostRiskError::InvalidNormalizedQualityLoss);
        }
        if self.target_reached && !self.feasible {
            return Err(MisCostRiskError::InfeasibleTargetClaim);
        }
        self.cost.validate()
    }
}

/// Interpretable breakdown of one run's scalar risk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MisCostRiskBreakdown {
    /// quality_loss_weight times normalized_quality_loss.
    pub quality_term: f64,
    /// Applied penalty for infeasible output, zero when feasible.
    pub infeasible_term: f64,
    /// Applied penalty for missing the target, zero when attained.
    pub target_miss_term: f64,
    /// Sum of normalized weighted cost dimensions.
    pub normalized_cost_term: f64,
    /// Sum of all terms; lower is better under this configuration.
    pub total_risk: f64,
}

/// Paired risk summary on identical workload keys.
///
/// Positive baseline-minus-candidate risk means the candidate has lower risk. The
/// inherited confidence interval is a normal approximation; for publication use stronger
/// paired inference and account for multiple comparisons.
#[derive(Debug, Clone, PartialEq)]
pub struct MisPairedCostRiskComparison {
    /// Policy identifier for the baseline.
    pub baseline_policy_id: String,
    /// Policy identifier for the candidate.
    pub candidate_policy_id: String,
    /// Number of matched workload/repetition pairs.
    pub paired_count: usize,
    /// Summary of baseline risk minus candidate risk on matched pairs.
    pub baseline_minus_candidate: PairedDifferenceSummary,
}

/// Cost-risk configuration, validation or paired comparison error.
#[derive(Debug, Clone, PartialEq)]
pub enum MisCostRiskError {
    /// Named scalar weight is negative or non-finite.
    InvalidWeight { /// Configuration field name.
        name: &'static str,
    },
    /// Per-dimension weight is negative or non-finite.
    InvalidCostComponentWeight { /// Cost dimension.
        metric: MisCostMetric,
    },
    /// Cost normalization reference must be positive.
    ZeroReferenceAmount { /// Cost dimension.
        metric: MisCostMetric,
    },
    /// One metric is weighted more than once.
    DuplicateCostMetric { /// Cost dimension.
        metric: MisCostMetric,
    },
    /// Wall time and component stage times were both weighted.
    OverlappingTimeMetrics,
    /// At least one scalar risk term must have a positive coefficient.
    ZeroRiskFunction,
    /// Comparison key is missing.
    EmptyComparisonKey,
    /// A workload identity component is missing.
    EmptyWorkloadIdentityField {
        /// Identity field that is missing.
        field: &'static str,
    },
    /// Workload identity does not match for a paired comparison key.
    MismatchedWorkloadIdentity {
        /// Matched key with different workload context.
        key: String,
    },
    /// A comparison combined more than one benchmark cohort.
    MismatchedCohort {
        /// Record whose cohort differs from the initial cohort.
        key: String,
    },
    /// Policy identifier is missing.
    EmptyPolicyId,
    /// Evidence reference is missing.
    EmptyEvidenceReference,
    /// One policy identifier mixes evidence classes.
    MixedEvidenceClasses {
        /// Policy side that mixes evidence classes.
        side: &'static str,
    },
    /// Normalized quality loss is not finite or outside [0, 1].
    InvalidNormalizedQualityLoss,
    /// An infeasible solution was marked as having reached the solution-quality target.
    InfeasibleTargetClaim,
    /// Monetary amount and currency must be both present or both absent.
    InvalidCurrencyPair,
    /// A comparison side repeats a comparison key.
    DuplicateComparisonKey { /// Side containing the duplicate.
        side: &'static str,
        /// Repeated comparison key.
        key: String,
    },
    /// Baseline and candidate do not contain identical comparison keys.
    MismatchedComparisonKeys,
    /// A comparison side mixes policy identifiers.
    MixedPolicyIds { /// Side containing mixed identifiers.
        side: &'static str,
    },
    /// Paired records use different currencies.
    CurrencyMismatch { /// Matched comparison key.
        key: String,
    },
    /// A required configured cost dimension was not recorded.
    MissingCostMetric { /// Matched comparison key.
        key: String,
        /// Side that is missing the dimension.
        side: &'static str,
        /// Missing dimension.
        metric: MisCostMetric,
    },
    /// Computed risk is non-finite.
    NonFiniteRisk,
    /// Risk exceeds the lightweight existing f32 summary helper's range.
    RiskOutsideSummaryRange,
    /// Paired comparison tolerance is negative or non-finite.
    InvalidTieTolerance,
    /// Paired summary could not be computed.
    EmptyPairedSummary,
}

impl fmt::Display for MisCostRiskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWeight { name } => write!(f, "{name} must be finite and nonnegative"),
            Self::InvalidCostComponentWeight { metric } =>
                write!(f, "weight for {} must be finite and nonnegative", metric.label()),
            Self::ZeroReferenceAmount { metric } =>
                write!(f, "reference amount for {} must be positive", metric.label()),
            Self::DuplicateCostMetric { metric } =>
                write!(f, "metric {} is weighted more than once", metric.label()),
            Self::OverlappingTimeMetrics =>
                write!(f, "wall-clock time cannot be weighted with component-stage timings"),
            Self::ZeroRiskFunction => write!(f, "risk configuration must contain a positive term"),
            Self::EmptyComparisonKey => write!(f, "comparison key must not be empty"),
            Self::EmptyWorkloadIdentityField { field } =>
                write!(f, "workload identity field {field} must not be empty"),
            Self::MismatchedWorkloadIdentity { key } =>
                write!(f, "paired records have different workload identity for {key}"),
            Self::MismatchedCohort { key } =>
                write!(f, "comparison record {key} belongs to a different cohort"),
            Self::EmptyPolicyId => write!(f, "policy identifier must not be empty"),
            Self::EmptyEvidenceReference => write!(f, "evidence reference must not be empty"),
            Self::MixedEvidenceClasses { side } =>
                write!(f, "{side} policy mixes evidence classes"),
            Self::InvalidNormalizedQualityLoss =>
                write!(f, "normalized quality loss must be finite and in [0, 1]"),
            Self::InfeasibleTargetClaim =>
                write!(f, "an infeasible solution cannot count as reaching the quality target"),
            Self::InvalidCurrencyPair =>
                write!(f, "money amount and uppercase three-letter currency code must appear together"),
            Self::DuplicateComparisonKey { side, key } =>
                write!(f, "{side} has duplicate comparison key {key}"),
            Self::MismatchedComparisonKeys =>
                write!(f, "baseline and candidate comparison keys do not match"),
            Self::MixedPolicyIds { side } => write!(f, "{side} mixes policy identifiers"),
            Self::CurrencyMismatch { key } =>
                write!(f, "baseline and candidate currencies differ for paired key {key}"),
            Self::MissingCostMetric { key, side, metric } =>
                write!(f, "{side} record {key} lacks configured metric {}", metric.label()),
            Self::NonFiniteRisk => write!(f, "cost-risk evaluation produced a non-finite value"),
            Self::RiskOutsideSummaryRange =>
                write!(f, "paired risk does not fit the f32 summary helper's range"),
            Self::InvalidTieTolerance => write!(f, "paired-comparison tie tolerance must be finite and nonnegative"),
            Self::EmptyPairedSummary => write!(f, "paired cost-risk summary could not be computed"),
        }
    }
}

impl std::error::Error for MisCostRiskError {}

/// Computes a risk breakdown for one observation under a frozen configuration.
pub fn evaluate_mis_cost_risk(
    observation: &MisCostRiskObservation,
    config: &MisCostRiskConfig,
) -> Result<MisCostRiskBreakdown, MisCostRiskError> {
    config.validate()?;
    observation.validate()?;

    let quality_term = config.quality_loss_weight * observation.normalized_quality_loss;
    let infeasible_term = if observation.feasible { 0.0 } else { config.infeasible_penalty };
    let target_miss_term = if observation.target_reached { 0.0 } else { config.target_miss_penalty };
    let mut normalized_cost_term = 0.0;
    for component in &config.costs {
        if component.weight == 0.0 { continue; }
        let amount = observation.cost.amount(component.metric).ok_or_else(|| {
            MisCostRiskError::MissingCostMetric {
                key: observation.comparison_key.clone(),
                side: "observation",
                metric: component.metric,
            }
        })?;
        normalized_cost_term += component.weight
            * (amount as f64 / component.reference_amount as f64);
    }
    let total_risk = quality_term + infeasible_term + target_miss_term + normalized_cost_term;
    if !quality_term.is_finite() || !infeasible_term.is_finite()
        || !target_miss_term.is_finite() || !normalized_cost_term.is_finite()
        || !total_risk.is_finite() {
        return Err(MisCostRiskError::NonFiniteRisk);
    }
    Ok(MisCostRiskBreakdown { quality_term, infeasible_term, target_miss_term, normalized_cost_term, total_risk })
}

/// Compares methods on exact matched keys using one frozen risk configuration.
///
/// Positive delta means candidate risk is lower. Duplicate/missing keys, mixed policy
/// IDs, missing configured measurements, currency mismatches, and incompatible units fail
/// closed. Monetary comparisons require the same currency for each matched workload.
pub fn compare_paired_mis_cost_risk(
    baseline: &[MisCostRiskObservation],
    candidate: &[MisCostRiskObservation],
    config: &MisCostRiskConfig,
    tie_tolerance: f32,
) -> Result<MisPairedCostRiskComparison, MisCostRiskError> {
    config.validate()?;
    if baseline.is_empty() || candidate.is_empty() {
        return Err(MisCostRiskError::EmptyPairedSummary);
    }
    if !tie_tolerance.is_finite() || tie_tolerance < 0.0 {
        return Err(MisCostRiskError::InvalidTieTolerance);
    }
    let baseline_map = index_records(baseline, "baseline")?;
    let candidate_map = index_records(candidate, "candidate")?;
    if baseline_map.keys().ne(candidate_map.keys()) {
        return Err(MisCostRiskError::MismatchedComparisonKeys);
    }
    let baseline_policy_id = policy_id(baseline, "baseline")?;
    let candidate_policy_id = policy_id(candidate, "candidate")?;
    let baseline_evidence_class = policy_evidence_class(baseline, "baseline")?;
    let candidate_evidence_class = policy_evidence_class(candidate, "candidate")?;

    // A comparison must represent one frozen cohort and exact matched workload contexts.
    let expected_cohort = baseline.first()
        .ok_or(MisCostRiskError::EmptyPairedSummary)?
        .workload.cohort_id.as_str();
    for row in baseline.iter().chain(candidate.iter()) {
        if row.workload.cohort_id != expected_cohort {
            return Err(MisCostRiskError::MismatchedCohort {
                key: row.comparison_key.clone(),
            });
        }
    }
    for (key, base) in &baseline_map {
        let cand = candidate_map.get(key).ok_or(MisCostRiskError::MismatchedComparisonKeys)?;
        if base.workload != cand.workload {
            return Err(MisCostRiskError::MismatchedWorkloadIdentity {
                key: (*key).to_owned(),
            });
        }
    }

    // Ordered maps pair runs deterministically regardless of input slice order.
    let mut baseline_risk = Vec::with_capacity(baseline.len());
    let mut candidate_risk = Vec::with_capacity(candidate.len());
    for (key, base) in &baseline_map {
        let cand = candidate_map.get(key).ok_or(MisCostRiskError::MismatchedComparisonKeys)?;
        let a = evaluate_mis_cost_risk(base, config)?.total_risk;
        let b = evaluate_mis_cost_risk(cand, config)?.total_risk;
        if a > f32::MAX as f64 || b > f32::MAX as f64 {
            return Err(MisCostRiskError::RiskOutsideSummaryRange);
        }
        baseline_risk.push(a as f32);
        candidate_risk.push(b as f32);
    }

    // A monetary scalar risk is only meaningful when every run uses one currency.
    // Matching USD against USD within each pair is insufficient if a later pair uses EUR.
    if config.uses_money() {
        let mut cohort_currency: Option<String> = None;
        for row in baseline.iter().chain(candidate.iter()) {
            let Some(code) = row.cost.currency_code.as_deref() else {
                return Err(MisCostRiskError::MissingCostMetric {
                    key: row.comparison_key.clone(),
                    side: "paired-cohort",
                    metric: MisCostMetric::MoneyMicrounits,
                });
            };
            match cohort_currency.as_deref() {
                Some(expected) if expected != code => {
                    return Err(MisCostRiskError::CurrencyMismatch {
                        key: row.comparison_key.clone(),
                    });
                }
                None => cohort_currency = Some(code.to_owned()),
                _ => {}
            }
        }
    }

    let summary = PairedDifferenceSummary::from_pairs(&baseline_risk, &candidate_risk, tie_tolerance)
        .ok_or(MisCostRiskError::EmptyPairedSummary)?;
    Ok(MisPairedCostRiskComparison {
        baseline_policy_id,
        candidate_policy_id,
        baseline_evidence_class,
        candidate_evidence_class,
        paired_count: summary.count,
        baseline_minus_candidate: summary,
    })
}

fn index_records<'a>(
    records: &'a [MisCostRiskObservation],
    side: &'static str,
) -> Result<BTreeMap<&'a str, &'a MisCostRiskObservation>, MisCostRiskError> {
    let mut indexed = BTreeMap::new();
    for record in records {
        record.validate()?;
        if indexed.insert(record.comparison_key.as_str(), record).is_some() {
            return Err(MisCostRiskError::DuplicateComparisonKey {
                side,
                key: record.comparison_key.clone(),
            });
        }
    }
    Ok(indexed)
}

fn policy_id(records: &[MisCostRiskObservation], side: &'static str) -> Result<String, MisCostRiskError> {
    let first = records.first().ok_or(MisCostRiskError::EmptyPairedSummary)?;
    if records.iter().any(|record| record.policy_id != first.policy_id) {
        return Err(MisCostRiskError::MixedPolicyIds { side });
    }
    Ok(first.policy_id.clone())
}

fn policy_evidence_class(
    records: &[MisCostRiskObservation],
    side: &'static str,
) -> Result<MisEvidenceClass, MisCostRiskError> {
    let first = records.first().ok_or(MisCostRiskError::EmptyPairedSummary)?;
    if records.iter().any(|record| record.evidence_class != first.evidence_class) {
        return Err(MisCostRiskError::MixedEvidenceClasses { side });
    }
    Ok(first.evidence_class)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MisCostRiskConfig {
        MisCostRiskConfig::new(2.0, 10.0, 4.0, vec![
            MisCostWeight { metric: MisCostMetric::PhysicalShots, reference_amount: 100, weight: 0.5 },
            MisCostWeight { metric: MisCostMetric::WallClockMillis, reference_amount: 1_000, weight: 0.5 },
        ]).expect("valid risk config")
    }

    fn record(key: &str, policy: &str, quality_loss: f64, shots: u64, wall_ms: u64) -> MisCostRiskObservation {
        let (graph_digest, seed) = key.split_once(':').expect("fixture comparison key");
        MisCostRiskObservation {
            comparison_key: key.to_owned(),
            workload: MisWorkloadIdentity {
                cohort_id: "fixture-cohort-v1".to_owned(),
                graph_digest: graph_digest.to_owned(),
                repetition_seed: seed.parse().expect("fixture seed"),
                reducer_config_digest: "fixture-reducer-v1".to_owned(),
                candidate_pool_digest: "fixture-candidates-v1".to_owned(),
                verifier_version: "fixture-verifier-v1".to_owned(),
            },
            policy_id: policy.to_owned(),
            evidence_class: MisEvidenceClass::SyntheticArms,
            evidence_reference: format!("fixture:{policy}:{key}"),
            feasible: true, target_reached: true, normalized_quality_loss: quality_loss,
            cost: MisOperationalCost { physical_shots: Some(shots), wall_clock_millis: Some(wall_ms), ..MisOperationalCost::default() },
        }
    }

    #[test]
    fn risk_breakdown_applies_quality_and_cost_scales() {
        let score = evaluate_mis_cost_risk(&record("g1:0", "classical", 0.25, 50, 500), &config()).unwrap();
        assert!((score.quality_term - 0.5).abs() < 1e-12);
        assert!((score.normalized_cost_term - 0.5).abs() < 1e-12);
        assert!((score.total_risk - 1.0).abs() < 1e-12);
    }

    #[test]
    fn infeasibility_and_target_miss_penalties_are_explicit() {
        let mut row = record("g1:0", "candidate", 0.0, 10, 10);
        row.feasible = false;
        row.target_reached = false;
        let score = evaluate_mis_cost_risk(&row, &config()).unwrap();
        assert_eq!(score.infeasible_term, 10.0);
        assert_eq!(score.target_miss_term, 4.0);
    }

    #[test]
    fn matched_records_pair_by_key_not_input_order() {
        let baseline = vec![record("g2:7", "baseline", 0.5, 200, 1_500), record("g1:9", "baseline", 0.5, 200, 1_500)];
        let candidate = vec![record("g1:9", "candidate", 0.1, 100, 500), record("g2:7", "candidate", 0.1, 100, 500)];
        let result = compare_paired_mis_cost_risk(&baseline, &candidate, &config(), 1e-6).unwrap();
        assert_eq!(result.paired_count, 2);
        assert_eq!(result.baseline_minus_candidate.a_wins, 2);
        assert!(result.baseline_minus_candidate.delta.mean > 0.0);
    }

    #[test]
    fn mismatched_or_duplicate_keys_fail_closed() {
        let baseline = vec![record("g1:0", "baseline", 0.2, 20, 200)];
        let candidate = vec![record("g2:0", "candidate", 0.2, 20, 200)];
        assert_eq!(
            compare_paired_mis_cost_risk(&baseline, &candidate, &config(), 1e-6).unwrap_err(),
            MisCostRiskError::MismatchedComparisonKeys
        );
        let duplicates = vec![record("g1:0", "baseline", 0.2, 20, 200), record("g1:0", "baseline", 0.2, 20, 200)];
        assert!(matches!(
            compare_paired_mis_cost_risk(&duplicates, &candidate, &config(), 1e-6),
            Err(MisCostRiskError::DuplicateComparisonKey { .. })
        ));
        let matched = vec![record("g1:0", "baseline", 0.2, 20, 200)];
        let matched_candidate = vec![record("g1:0", "candidate", 0.1, 10, 100)];
        assert_eq!(
            compare_paired_mis_cost_risk(&matched, &matched_candidate, &config(), f32::NAN).unwrap_err(),
            MisCostRiskError::InvalidTieTolerance
        );
    }

    #[test]
    fn pairing_rejects_same_key_with_different_graph_or_reducer_context() {
        let baseline = vec![record("g1:0", "baseline", 0.2, 20, 200)];
        let mut candidate = record("g1:0", "candidate", 0.1, 10, 100);
        candidate.workload.graph_digest = "different-graph".to_owned();
        assert_eq!(
            compare_paired_mis_cost_risk(&baseline, &[candidate.clone()], &config(), 1e-6).unwrap_err(),
            MisCostRiskError::MismatchedWorkloadIdentity { key: "g1:0".to_owned() }
        );

        candidate.workload.graph_digest = "g1".to_owned();
        candidate.workload.reducer_config_digest = "different-reducer".to_owned();
        assert_eq!(
            compare_paired_mis_cost_risk(&baseline, &[candidate], &config(), 1e-6).unwrap_err(),
            MisCostRiskError::MismatchedWorkloadIdentity { key: "g1:0".to_owned() }
        );
    }

    #[test]
    fn one_policy_cannot_mix_synthetic_and_physical_evidence() {
        let baseline = vec![
            record("g1:0", "baseline", 0.2, 20, 200),
            record("g2:1", "baseline", 0.2, 20, 200),
        ];
        let candidate = vec![
            record("g1:0", "candidate", 0.1, 10, 100),
            record("g2:1", "candidate", 0.1, 10, 100),
        ];
        let mut mixed = baseline.clone();
        mixed[1].evidence_class = MisEvidenceClass::PhysicalQpu;
        assert_eq!(
            compare_paired_mis_cost_risk(&mixed, &candidate, &config(), 1e-6).unwrap_err(),
            MisCostRiskError::MixedEvidenceClasses { side: "baseline" }
        );
        let result = compare_paired_mis_cost_risk(&baseline, &candidate, &config(), 1e-6).unwrap();
        assert_eq!(result.baseline_evidence_class, MisEvidenceClass::SyntheticArms);
        assert_eq!(result.candidate_evidence_class, MisEvidenceClass::SyntheticArms);
    }

    #[test]
    fn evidence_reference_is_required() {
        let mut row = record("g1:0", "candidate", 0.1, 10, 100);
        row.evidence_reference.clear();
        assert_eq!(row.validate(), Err(MisCostRiskError::EmptyEvidenceReference));
    }

    #[test]
    fn comparison_rejects_multiple_cohorts_even_when_pair_keys_match() {
        let baseline = vec![
            record("g1:0", "baseline", 0.2, 20, 200),
            record("g2:1", "baseline", 0.2, 20, 200),
        ];
        let mut candidate = vec![
            record("g1:0", "candidate", 0.1, 10, 100),
            record("g2:1", "candidate", 0.1, 10, 100),
        ];
        candidate[1].workload.cohort_id = "another-cohort".to_owned();
        assert_eq!(
            compare_paired_mis_cost_risk(&baseline, &candidate, &config(), 1e-6).unwrap_err(),
            MisCostRiskError::MismatchedCohort { key: "g2:1".to_owned() }
        );
    }

    #[test]
    fn overlapping_time_dimensions_are_rejected() {
        let error = MisCostRiskConfig::new(1.0, 1.0, 1.0, vec![
            MisCostWeight { metric: MisCostMetric::WallClockMillis, reference_amount: 1_000, weight: 1.0 },
            MisCostWeight { metric: MisCostMetric::QueueMillis, reference_amount: 1_000, weight: 1.0 },
        ]).unwrap_err();
        assert_eq!(error, MisCostRiskError::OverlappingTimeMetrics);
    }

    #[test]
    fn money_requires_currency_and_pairs_must_match_it() {
        let mut invalid = record("g1:0", "baseline", 0.2, 20, 200);
        invalid.cost.money_microunits = Some(500);
        assert_eq!(invalid.validate(), Err(MisCostRiskError::InvalidCurrencyPair));

        let cfg = MisCostRiskConfig::new(1.0, 1.0, 1.0, vec![
            MisCostWeight { metric: MisCostMetric::MoneyMicrounits, reference_amount: 1_000, weight: 1.0 },
        ]).unwrap();
        let mut baseline = record("g1:0", "baseline", 0.2, 20, 200);
        baseline.cost.money_microunits = Some(500);
        baseline.cost.currency_code = Some("USD".to_owned());
        let mut candidate = record("g1:0", "candidate", 0.2, 20, 200);
        candidate.cost.money_microunits = Some(500);
        candidate.cost.currency_code = Some("EUR".to_owned());
        assert_eq!(
            compare_paired_mis_cost_risk(&[baseline], &[candidate], &cfg, 1e-6).unwrap_err(),
            MisCostRiskError::CurrencyMismatch { key: "g1:0".to_owned() }
        );

        let mut baseline_usd = record("g1:0", "baseline", 0.2, 20, 200);
        baseline_usd.cost.money_microunits = Some(100);
        baseline_usd.cost.currency_code = Some("USD".to_owned());
        let mut candidate_usd = record("g1:0", "candidate", 0.2, 20, 200);
        candidate_usd.cost.money_microunits = Some(100);
        candidate_usd.cost.currency_code = Some("USD".to_owned());

        let mut baseline_eur = record("g2:0", "baseline", 0.2, 20, 200);
        baseline_eur.cost.money_microunits = Some(100);
        baseline_eur.cost.currency_code = Some("EUR".to_owned());
        let mut candidate_eur = record("g2:0", "candidate", 0.2, 20, 200);
        candidate_eur.cost.money_microunits = Some(100);
        candidate_eur.cost.currency_code = Some("EUR".to_owned());
        assert_eq!(
            compare_paired_mis_cost_risk(
                &[baseline_usd, baseline_eur],
                &[candidate_usd, candidate_eur],
                &cfg,
                1e-6,
            ).unwrap_err(),
            MisCostRiskError::CurrencyMismatch { key: "g2:0".to_owned() }
        );
    }

    #[test]
    fn every_unrecorded_cost_dimension_is_missing_not_zero() {
        let cfg = MisCostRiskConfig::new(1.0, 1.0, 1.0, vec![
            MisCostWeight { metric: MisCostMetric::PhysicalShots, reference_amount: 100, weight: 1.0 },
        ]).unwrap();
        assert_eq!(
            evaluate_mis_cost_risk(&record("g1:0", "candidate", 0.0, 0, 0), &cfg).unwrap_err(),
            MisCostRiskError::MissingCostMetric {
                key: "g1:0".to_owned(), side: "observation", metric: MisCostMetric::PhysicalShots,
            }
        );

        let energy_cfg = MisCostRiskConfig::new(1.0, 1.0, 1.0, vec![
            MisCostWeight { metric: MisCostMetric::EnergyMillijoules, reference_amount: 1_000, weight: 1.0 },
        ]).unwrap();
        assert_eq!(
            evaluate_mis_cost_risk(&record("g1:0", "candidate", 0.0, 0, 0), &energy_cfg).unwrap_err(),
            MisCostRiskError::MissingCostMetric {
                key: "g1:0".to_owned(), side: "observation", metric: MisCostMetric::EnergyMillijoules,
            }
        );
    }

    #[test]
    fn infeasible_result_cannot_claim_target_attainment() {
        let mut row = record("g1:0", "candidate", 0.0, 10, 10);
        row.feasible = false;
        row.target_reached = true;
        assert_eq!(row.validate(), Err(MisCostRiskError::InfeasibleTargetClaim));
    }
}
