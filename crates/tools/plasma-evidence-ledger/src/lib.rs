//! Fail-closed evidence and error-budget primitives for kinetic-plasma trajectories.
//!
//! This crate qualifies recorded evidence against caller-supplied,
//! equation-cohort-specific policies. It does not create physical error bounds
//! or certify a solver. Missing, stale, non-finite, empirical-only, or
//! under-specified evidence must never be upgraded to a stronger claim.
//!
//! Error components may be summed only when they refer to the same metric,
//! cohort, norm/units, provenance and trajectory segment. Probabilistic
//! components use a union bound; independence is never assumed. Empirical
//! bounds are deliberately excluded from qualified aggregation.

#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

/// Equation and discretization cohort. Policies are versioned per cohort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EquationCohort {
    VlasovPoissonPic,
    VlasovMaxwellPic,
    CollisionalVlasovMaxwellLandau,
    ReducedQuasineutralVlasovAmpere,
}

impl EquationCohort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VlasovPoissonPic => "vlasov_poisson_pic",
            Self::VlasovMaxwellPic => "vlasov_maxwell_pic",
            Self::CollisionalVlasovMaxwellLandau => "collisional_vlasov_maxwell_landau",
            Self::ReducedQuasineutralVlasovAmpere => "reduced_quasineutral_vlasov_ampere",
        }
    }
}

/// Stable metric identifiers shared across benchmark profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricId {
    ChargeOrMass,
    Momentum,
    KineticEnergy,
    FieldEnergy,
    TotalEnergy,
    GaussPoissonResidual,
    ChargeContinuityResidual,
    MagneticDivergenceResidual,
    EntropyProduction,
    PhaseSpaceMmd,
    PhaseSpaceW1,
    FieldL2Error,
    ParticleForceL2Error,
}

impl MetricId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChargeOrMass => "charge_or_mass",
            Self::Momentum => "momentum",
            Self::KineticEnergy => "kinetic_energy",
            Self::FieldEnergy => "field_energy",
            Self::TotalEnergy => "total_energy",
            Self::GaussPoissonResidual => "gauss_poisson_residual",
            Self::ChargeContinuityResidual => "charge_continuity_residual",
            Self::MagneticDivergenceResidual => "magnetic_divergence_residual",
            Self::EntropyProduction => "entropy_production",
            Self::PhaseSpaceMmd => "phase_space_mmd",
            Self::PhaseSpaceW1 => "phase_space_w1",
            Self::FieldL2Error => "field_l2_error",
            Self::ParticleForceL2Error => "particle_force_l2_error",
        }
    }
}

/// Physical applicability of a metric to the selected equation cohort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applicability {
    Applicable,
    NotApplicable,
    Unavailable,
}

/// Evidence strength. Empirical intervals are never promoted to proof.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundKind {
    Deterministic,
    Probabilistic { confidence_level: f64 },
    Empirical,
    Unavailable,
}

/// Qualification state for one metric or one trajectory segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationStatus {
    Pass,
    Fail,
    Unresolved,
    NotApplicable,
}

/// Whether a cohort policy permits an explicit not-applicable metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplicabilityRule {
    MustBeApplicable,
    NotApplicableAllowed,
}

/// Identity of the source, input, configuration and operator that produced evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// Stable identity shared by all evidence generated from one trajectory run.
    pub run_id: String,
    /// Revision of the code that produced this particular evidence record.
    pub source_revision: String,
    /// Digest of the trajectory input shared by every record in the run.
    pub input_digest: String,
    /// Digest of the frozen run configuration shared by every record.
    pub config_digest: String,
    /// Solver or diagnostic operator identity for this particular record.
    pub operator_id: String,
}

impl Provenance {
    pub fn validate(&self) -> Result<(), LedgerError> {
        for (field, value) in [
            ("run_id", self.run_id.as_str()),
            ("source_revision", self.source_revision.as_str()),
            ("input_digest", self.input_digest.as_str()),
            ("config_digest", self.config_digest.as_str()),
            ("operator_id", self.operator_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(LedgerError::MissingProvenance { field });
            }
        }
        Ok(())
    }

    /// Whether two evidence receipts refer to the same frozen trajectory run.
    ///
    /// Source revisions and operator IDs are intentionally excluded: separate
    /// independently versioned diagnostics may evaluate the same run.
    pub fn same_run(&self, other: &Self) -> bool {
        self.run_id == other.run_id
            && self.input_digest == other.input_digest
            && self.config_digest == other.config_digest
    }
}

/// One metric observation over a bounded trajectory segment.
///
/// The raw observable may be signed. Error values, interval endpoints and
/// tolerances are nonnegative magnitudes in the declared units. An estimate
/// without a qualified interval remains unresolved.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricObservation {
    pub cohort: EquationCohort,
    pub metric: MetricId,
    pub units: String,
    pub time_start: f64,
    pub time_end: f64,
    pub applicability: Applicability,
    pub applicability_reason: Option<String>,
    pub raw_value: Option<f64>,
    pub error_estimate: Option<f64>,
    pub lower_error_bound: Option<f64>,
    pub upper_error_bound: Option<f64>,
    pub acceptance_tolerance: Option<f64>,
    pub bound_kind: BoundKind,
    pub provenance: Provenance,
}

impl MetricObservation {
    /// Validate supplied values and their metadata.
    ///
    /// Missing fields are permitted in the record so they can be reported as
    /// unresolved; malformed supplied fields are rejected.
    pub fn validate(&self) -> Result<(), LedgerError> {
        if self.units.trim().is_empty() {
            return Err(LedgerError::MissingUnits);
        }
        validate_time_interval(self.time_start, self.time_end)?;
        self.provenance.validate()?;

        if let Some(reason) = &self.applicability_reason {
            if reason.trim().is_empty() {
                return Err(LedgerError::EmptyReason);
            }
        }

        match self.applicability {
            Applicability::NotApplicable | Applicability::Unavailable => {
                if self.applicability_reason.is_none() {
                    return Err(LedgerError::MissingReason);
                }
                if self.bound_kind != BoundKind::Unavailable
                    || self.raw_value.is_some()
                    || self.error_estimate.is_some()
                    || self.lower_error_bound.is_some()
                    || self.upper_error_bound.is_some()
                    || self.acceptance_tolerance.is_some()
                {
                    return Err(LedgerError::UnexpectedUnavailablePayload);
                }
            }
            Applicability::Applicable => {
                if self.bound_kind == BoundKind::Unavailable {
                    return Err(LedgerError::UnavailableBoundForApplicableMetric);
                }
                if let BoundKind::Probabilistic { confidence_level } = self.bound_kind {
                    validate_confidence(confidence_level)?;
                }
                validate_optional_finite(self.raw_value, false)?;
                validate_optional_finite(self.error_estimate, true)?;
                validate_optional_finite(self.lower_error_bound, true)?;
                validate_optional_finite(self.upper_error_bound, true)?;
                validate_optional_finite(self.acceptance_tolerance, true)?;

                if let (Some(lower), Some(upper)) = (self.lower_error_bound, self.upper_error_bound)
                {
                    if lower > upper {
                        return Err(LedgerError::InvalidErrorInterval);
                    }
                }
                if let (Some(estimate), Some(lower), Some(upper)) = (
                    self.error_estimate,
                    self.lower_error_bound,
                    self.upper_error_bound,
                ) {
                    if estimate < lower || estimate > upper {
                        return Err(LedgerError::EstimateOutsideErrorInterval);
                    }
                }
            }
        }
        Ok(())
    }

    /// Evaluate using only the declared evidence strength and interval.
    pub fn status(&self) -> Result<QualificationStatus, LedgerError> {
        self.validate()?;
        match self.applicability {
            Applicability::NotApplicable => return Ok(QualificationStatus::NotApplicable),
            Applicability::Unavailable => return Ok(QualificationStatus::Unresolved),
            Applicability::Applicable => {}
        }

        let (Some(_raw), Some(_estimate), Some(lower), Some(upper), Some(tolerance)) = (
            self.raw_value,
            self.error_estimate,
            self.lower_error_bound,
            self.upper_error_bound,
            self.acceptance_tolerance,
        ) else {
            return Ok(QualificationStatus::Unresolved);
        };

        // An empirical interval is useful descriptive evidence, but does not
        // establish a deterministic or probabilistic qualification bound.
        if self.bound_kind == BoundKind::Empirical {
            return Ok(QualificationStatus::Unresolved);
        }

        if upper <= tolerance {
            Ok(QualificationStatus::Pass)
        } else if lower > tolerance {
            Ok(QualificationStatus::Fail)
        } else {
            Ok(QualificationStatus::Unresolved)
        }
    }
}

/// One required metric in an equation-cohort-specific policy.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricRequirement {
    pub metric: MetricId,
    pub units: String,
    pub acceptance_tolerance: f64,
    pub applicability_rule: ApplicabilityRule,
}

/// Caller-supplied policy. No universal required-metric set is assumed.
#[derive(Debug, Clone, PartialEq)]
pub struct CohortPolicy {
    pub version: String,
    pub cohort: EquationCohort,
    pub required_metrics: Vec<MetricRequirement>,
}

impl CohortPolicy {
    pub fn validate(&self) -> Result<(), LedgerError> {
        if self.version.trim().is_empty() {
            return Err(LedgerError::MissingPolicyVersion);
        }
        if self.required_metrics.is_empty() {
            return Err(LedgerError::EmptyPolicy);
        }
        for requirement in &self.required_metrics {
            if requirement.units.trim().is_empty() {
                return Err(LedgerError::MissingUnits);
            }
            if !requirement.acceptance_tolerance.is_finite()
                || requirement.acceptance_tolerance < 0.0
            {
                return Err(LedgerError::InvalidTolerance);
            }
        }
        for index in 0..self.required_metrics.len() {
            if self.required_metrics[index + 1..]
                .iter()
                .any(|other| other.metric == self.required_metrics[index].metric)
            {
                return Err(LedgerError::DuplicateMetricRequirement {
                    metric: self.required_metrics[index].metric,
                });
            }
        }
        Ok(())
    }
}

/// All observations for one bounded trajectory segment.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryLedger {
    pub cohort: EquationCohort,
    pub time_start: f64,
    pub time_end: f64,
    pub provenance: Provenance,
    pub observations: Vec<MetricObservation>,
}

impl TrajectoryLedger {
    /// Evaluate all requirements in the supplied cohort policy.
    ///
    /// Missing metrics, stale provenance, and units/tolerance mismatches remain
    /// unresolved. A malformed record, cohort mismatch or segment mismatch is
    /// an error and cannot produce a pass.
    pub fn qualify(&self, policy: &CohortPolicy) -> Result<TrajectoryQualification, LedgerError> {
        policy.validate()?;
        validate_time_interval(self.time_start, self.time_end)?;
        self.provenance.validate()?;
        if self.cohort != policy.cohort {
            return Err(LedgerError::CohortMismatch);
        }

        for observation in &self.observations {
            observation.validate()?;
            if observation.cohort != self.cohort {
                return Err(LedgerError::CohortMismatch);
            }
            if observation.time_start != self.time_start || observation.time_end != self.time_end {
                return Err(LedgerError::SegmentMismatch);
            }
        }
        for index in 0..self.observations.len() {
            if self.observations[index + 1..]
                .iter()
                .any(|other| other.metric == self.observations[index].metric)
            {
                return Err(LedgerError::DuplicateMetricObservation {
                    metric: self.observations[index].metric,
                });
            }
        }

        let mut decisions = Vec::with_capacity(policy.required_metrics.len());
        for requirement in &policy.required_metrics {
            let Some(observation) = self
                .observations
                .iter()
                .find(|item| item.metric == requirement.metric)
            else {
                decisions.push(MetricDecision::unresolved(
                    requirement.metric,
                    None,
                    "required metric is missing",
                ));
                continue;
            };

            if !observation.provenance.same_run(&self.provenance) {
                decisions.push(MetricDecision::unresolved(
                    requirement.metric,
                    Some(observation.bound_kind),
                    "observation provenance does not match trajectory",
                ));
                continue;
            }
            if observation.units != requirement.units {
                decisions.push(MetricDecision::unresolved(
                    requirement.metric,
                    Some(observation.bound_kind),
                    "metric units do not match cohort policy",
                ));
                continue;
            }
            if observation.applicability == Applicability::Applicable
                && observation.acceptance_tolerance != Some(requirement.acceptance_tolerance)
            {
                decisions.push(MetricDecision::unresolved(
                    requirement.metric,
                    Some(observation.bound_kind),
                    "recorded tolerance does not match frozen policy",
                ));
                continue;
            }

            let status = match observation.applicability {
                Applicability::NotApplicable => {
                    if requirement.applicability_rule == ApplicabilityRule::NotApplicableAllowed {
                        QualificationStatus::NotApplicable
                    } else {
                        QualificationStatus::Unresolved
                    }
                }
                Applicability::Unavailable => QualificationStatus::Unresolved,
                Applicability::Applicable => observation.status()?,
            };
            let reason = match status {
                QualificationStatus::Pass => None,
                QualificationStatus::Fail => Some("lower error bound exceeds tolerance".to_owned()),
                QualificationStatus::Unresolved => observation
                    .applicability_reason
                    .clone()
                    .or_else(|| Some("evidence is insufficient to qualify the metric".to_owned())),
                QualificationStatus::NotApplicable => observation.applicability_reason.clone(),
            };
            decisions.push(MetricDecision {
                metric: requirement.metric,
                status,
                bound_kind: Some(observation.bound_kind),
                reason,
            });
        }

        let status = summarize_decisions(&decisions);
        Ok(TrajectoryQualification {
            policy_version: policy.version.clone(),
            cohort: self.cohort,
            status,
            decisions,
        })
    }
}

/// Per-metric decision returned from cohort-policy evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricDecision {
    pub metric: MetricId,
    pub status: QualificationStatus,
    pub bound_kind: Option<BoundKind>,
    pub reason: Option<String>,
}

impl MetricDecision {
    fn unresolved(metric: MetricId, bound_kind: Option<BoundKind>, reason: &str) -> Self {
        Self {
            metric,
            status: QualificationStatus::Unresolved,
            bound_kind,
            reason: Some(reason.to_owned()),
        }
    }
}

/// Overall result for one trajectory segment.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryQualification {
    pub policy_version: String,
    pub cohort: EquationCohort,
    pub status: QualificationStatus,
    pub decisions: Vec<MetricDecision>,
}

fn summarize_decisions(decisions: &[MetricDecision]) -> QualificationStatus {
    if decisions.is_empty() {
        return QualificationStatus::Unresolved;
    }
    if decisions
        .iter()
        .any(|decision| decision.status == QualificationStatus::Fail)
    {
        return QualificationStatus::Fail;
    }
    if decisions
        .iter()
        .any(|decision| decision.status == QualificationStatus::Unresolved)
    {
        return QualificationStatus::Unresolved;
    }
    if decisions
        .iter()
        .all(|decision| decision.status == QualificationStatus::NotApplicable)
    {
        return QualificationStatus::NotApplicable;
    }
    QualificationStatus::Pass
}

/// One component of an error budget for a single target quantity and segment.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorComponentBound {
    pub cohort: EquationCohort,
    pub target_metric: MetricId,
    pub units: String,
    pub time_start: f64,
    pub time_end: f64,
    pub provenance: Provenance,
    pub source: String,
    pub upper_bound: f64,
    pub bound_kind: BoundKind,
}

/// Bound strength after conservative same-quantity aggregation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregatedBoundKind {
    Deterministic,
    Probabilistic { confidence_lower_bound: f64 },
}

/// Conservative aggregate. Root-sum-square aggregation is intentionally absent.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregatedErrorBound {
    pub cohort: EquationCohort,
    pub target_metric: MetricId,
    pub units: String,
    pub time_start: f64,
    pub time_end: f64,
    pub component_provenance: Vec<Provenance>,
    pub upper_bound: f64,
    pub bound_kind: AggregatedBoundKind,
    pub component_count: usize,
}

/// Sum only components that bound the same metric, norm, units, cohort and segment.
///
/// Deterministic bounds sum directly. Probabilistic bounds use a union bound:
/// the joint confidence lower bound is 1 - sum(1 - confidence_i). Independence
/// is not assumed. Empirical/unavailable components cannot enter a qualified sum.
pub fn aggregate_error_components(
    components: &[ErrorComponentBound],
) -> Result<AggregatedErrorBound, LedgerError> {
    let first = components.first().ok_or(LedgerError::EmptyErrorBudget)?;
    if first.units.trim().is_empty() {
        return Err(LedgerError::MissingUnits);
    }
    if first.source.trim().is_empty() {
        return Err(LedgerError::MissingErrorSource);
    }
    validate_time_interval(first.time_start, first.time_end)?;
    first.provenance.validate()?;

    let cohort = first.cohort;
    let target_metric = first.target_metric;
    let units = first.units.clone();
    let time_start = first.time_start;
    let time_end = first.time_end;
    let provenance = first.provenance.clone();

    let mut upper_sum = CompensatedSum::default();
    let mut alpha_sum = CompensatedSum::default();

    for component in components {
        if component.cohort != cohort {
            return Err(LedgerError::ErrorBudgetCohortMismatch);
        }
        if component.target_metric != target_metric {
            return Err(LedgerError::ErrorBudgetTargetMismatch);
        }
        if component.units != units {
            return Err(LedgerError::ErrorBudgetUnitMismatch);
        }
        if component.time_start != time_start || component.time_end != time_end {
            return Err(LedgerError::ErrorBudgetSegmentMismatch);
        }
        if !component.provenance.same_run(&provenance) {
            return Err(LedgerError::ErrorBudgetProvenanceMismatch);
        }
        if component.source.trim().is_empty() {
            return Err(LedgerError::MissingErrorSource);
        }
        if !component.upper_bound.is_finite() || component.upper_bound < 0.0 {
            return Err(LedgerError::InvalidErrorBound);
        }

        match component.bound_kind {
            BoundKind::Deterministic => {}
            BoundKind::Probabilistic { confidence_level } => {
                validate_confidence(confidence_level)?;
                alpha_sum.add(1.0 - confidence_level);
            }
            BoundKind::Empirical | BoundKind::Unavailable => {
                return Err(LedgerError::UnqualifiedErrorComponent);
            }
        }
        upper_sum.add(component.upper_bound);
    }

    let upper_bound = upper_sum.total();
    let alpha = alpha_sum.total();
    if !upper_bound.is_finite() || !alpha.is_finite() {
        return Err(LedgerError::NonFiniteAggregate);
    }
    if alpha > 1.0 {
        return Err(LedgerError::InvalidConfidenceBudget);
    }

    let bound_kind = if alpha == 0.0 {
        AggregatedBoundKind::Deterministic
    } else {
        AggregatedBoundKind::Probabilistic {
            confidence_lower_bound: 1.0 - alpha,
        }
    };

    Ok(AggregatedErrorBound {
        cohort,
        target_metric,
        units,
        time_start,
        time_end,
        component_provenance: components
            .iter()
            .map(|component| component.provenance.clone())
            .collect(),
        upper_bound,
        bound_kind,
        component_count: components.len(),
    })
}

#[derive(Debug, Default, Clone, Copy)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum + self.correction
    }
}

fn validate_time_interval(start: f64, end: f64) -> Result<(), LedgerError> {
    if !start.is_finite() || !end.is_finite() || end < start {
        return Err(LedgerError::InvalidTimeInterval);
    }
    Ok(())
}

fn validate_optional_finite(value: Option<f64>, nonnegative: bool) -> Result<(), LedgerError> {
    if let Some(value) = value {
        if !value.is_finite() || (nonnegative && value < 0.0) {
            return Err(LedgerError::InvalidNumericValue);
        }
    }
    Ok(())
}

fn validate_confidence(confidence: f64) -> Result<(), LedgerError> {
    if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0 {
        return Err(LedgerError::InvalidConfidence);
    }
    Ok(())
}

/// Malformed evidence or an unjustified aggregation request.
#[derive(Debug, Clone, PartialEq)]
pub enum LedgerError {
    MissingProvenance { field: &'static str },
    MissingUnits,
    InvalidTimeInterval,
    EmptyReason,
    MissingReason,
    UnexpectedUnavailablePayload,
    UnavailableBoundForApplicableMetric,
    InvalidNumericValue,
    InvalidErrorInterval,
    EstimateOutsideErrorInterval,
    InvalidTolerance,
    InvalidConfidence,
    MissingPolicyVersion,
    EmptyPolicy,
    DuplicateMetricRequirement { metric: MetricId },
    CohortMismatch,
    SegmentMismatch,
    DuplicateMetricObservation { metric: MetricId },
    EmptyErrorBudget,
    MissingErrorSource,
    InvalidErrorBound,
    UnqualifiedErrorComponent,
    ErrorBudgetCohortMismatch,
    ErrorBudgetTargetMismatch,
    ErrorBudgetUnitMismatch,
    ErrorBudgetSegmentMismatch,
    ErrorBudgetProvenanceMismatch,
    NonFiniteAggregate,
    InvalidConfidenceBudget,
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingProvenance { field } => write!(f, "missing provenance field {field}"),
            Self::MissingUnits => write!(f, "metric units must be non-empty"),
            Self::InvalidTimeInterval => write!(f, "time interval must be finite and ordered"),
            Self::EmptyReason => write!(f, "applicability reason cannot be empty"),
            Self::MissingReason => write!(f, "not-applicable or unavailable metrics need a reason"),
            Self::UnexpectedUnavailablePayload => {
                write!(f, "unavailable metrics cannot carry numeric evidence")
            }
            Self::UnavailableBoundForApplicableMetric => {
                write!(f, "applicable metrics cannot use an unavailable bound")
            }
            Self::InvalidNumericValue => write!(f, "numeric evidence is invalid or non-finite"),
            Self::InvalidErrorInterval => {
                write!(f, "error interval lower bound exceeds upper bound")
            }
            Self::EstimateOutsideErrorInterval => {
                write!(f, "error estimate lies outside its declared interval")
            }
            Self::InvalidTolerance => {
                write!(f, "acceptance tolerance must be finite and nonnegative")
            }
            Self::InvalidConfidence => write!(f, "confidence level must be finite and in (0,1)"),
            Self::MissingPolicyVersion => write!(f, "cohort policy version must be non-empty"),
            Self::EmptyPolicy => write!(f, "cohort policy must define required metrics"),
            Self::DuplicateMetricRequirement { metric } => {
                write!(f, "metric {} is required more than once", metric.as_str())
            }
            Self::CohortMismatch => write!(f, "cohort mismatch"),
            Self::SegmentMismatch => write!(
                f,
                "observation time interval differs from trajectory segment"
            ),
            Self::DuplicateMetricObservation { metric } => {
                write!(
                    f,
                    "metric {} is observed more than once in one segment",
                    metric.as_str()
                )
            }
            Self::EmptyErrorBudget => write!(f, "error budget must contain at least one component"),
            Self::MissingErrorSource => write!(f, "error component source must be non-empty"),
            Self::InvalidErrorBound => write!(f, "error component upper bound is invalid"),
            Self::UnqualifiedErrorComponent => {
                write!(
                    f,
                    "empirical/unavailable components cannot enter a qualified error sum"
                )
            }
            Self::ErrorBudgetCohortMismatch => write!(f, "error-budget cohorts differ"),
            Self::ErrorBudgetTargetMismatch => write!(f, "error-budget target metrics differ"),
            Self::ErrorBudgetUnitMismatch => write!(f, "error-budget units differ"),
            Self::ErrorBudgetSegmentMismatch => write!(f, "error-budget time segments differ"),
            Self::ErrorBudgetProvenanceMismatch => write!(f, "error-budget provenance differs"),
            Self::NonFiniteAggregate => write!(f, "aggregated error bound is non-finite"),
            Self::InvalidConfidenceBudget => {
                write!(f, "union-bound failure probabilities sum to more than one")
            }
        }
    }
}

impl Error for LedgerError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> Provenance {
        Provenance {
            run_id: "run-0001".to_owned(),
            source_revision: "0123456789abcdef".to_owned(),
            input_digest: "input-sha256".to_owned(),
            config_digest: "config-sha256".to_owned(),
            operator_id: "periodic-vp-fd-v1".to_owned(),
        }
    }

    fn observation(
        metric: MetricId,
        estimate: f64,
        lower: f64,
        upper: f64,
        tolerance: f64,
        bound_kind: BoundKind,
    ) -> MetricObservation {
        MetricObservation {
            cohort: EquationCohort::VlasovPoissonPic,
            metric,
            units: "dimensionless".to_owned(),
            time_start: 0.0,
            time_end: 1.0,
            applicability: Applicability::Applicable,
            applicability_reason: None,
            raw_value: Some(0.0),
            error_estimate: Some(estimate),
            lower_error_bound: Some(lower),
            upper_error_bound: Some(upper),
            acceptance_tolerance: Some(tolerance),
            bound_kind,
            provenance: provenance(),
        }
    }

    fn requirement(metric: MetricId, tolerance: f64) -> MetricRequirement {
        MetricRequirement {
            metric,
            units: "dimensionless".to_owned(),
            acceptance_tolerance: tolerance,
            applicability_rule: ApplicabilityRule::MustBeApplicable,
        }
    }

    fn policy(required_metrics: Vec<MetricRequirement>) -> CohortPolicy {
        CohortPolicy {
            version: "vp-pic-test-v1".to_owned(),
            cohort: EquationCohort::VlasovPoissonPic,
            required_metrics,
        }
    }

    fn ledger(observations: Vec<MetricObservation>) -> TrajectoryLedger {
        TrajectoryLedger {
            cohort: EquationCohort::VlasovPoissonPic,
            time_start: 0.0,
            time_end: 1.0,
            provenance: provenance(),
            observations,
        }
    }

    #[test]
    fn lower_upper_bound_decisions_are_pass_fail_or_unresolved() {
        let pass = observation(
            MetricId::FieldL2Error,
            0.04,
            0.02,
            0.08,
            0.1,
            BoundKind::Deterministic,
        );
        assert_eq!(pass.status().unwrap(), QualificationStatus::Pass);

        let fail = observation(
            MetricId::FieldL2Error,
            0.15,
            0.12,
            0.2,
            0.1,
            BoundKind::Deterministic,
        );
        assert_eq!(fail.status().unwrap(), QualificationStatus::Fail);

        let overlaps = observation(
            MetricId::FieldL2Error,
            0.1,
            0.08,
            0.12,
            0.1,
            BoundKind::Deterministic,
        );
        assert_eq!(overlaps.status().unwrap(), QualificationStatus::Unresolved);
    }

    #[test]
    fn empirical_only_observations_are_never_promoted_to_pass() {
        let empirical = observation(
            MetricId::FieldL2Error,
            0.02,
            0.01,
            0.03,
            0.1,
            BoundKind::Empirical,
        );
        assert_eq!(empirical.status().unwrap(), QualificationStatus::Unresolved);
    }

    #[test]
    fn missing_values_tolerance_and_unavailable_measurements_are_unresolved() {
        let mut missing = observation(
            MetricId::FieldL2Error,
            0.02,
            0.01,
            0.03,
            0.1,
            BoundKind::Deterministic,
        );
        missing.upper_error_bound = None;
        assert_eq!(missing.status().unwrap(), QualificationStatus::Unresolved);

        missing.upper_error_bound = Some(0.03);
        missing.acceptance_tolerance = None;
        assert_eq!(missing.status().unwrap(), QualificationStatus::Unresolved);

        let unavailable = MetricObservation {
            cohort: EquationCohort::VlasovPoissonPic,
            metric: MetricId::EntropyProduction,
            units: "dimensionless".to_owned(),
            time_start: 0.0,
            time_end: 1.0,
            applicability: Applicability::Unavailable,
            applicability_reason: Some("collision operator absent".to_owned()),
            raw_value: None,
            error_estimate: None,
            lower_error_bound: None,
            upper_error_bound: None,
            acceptance_tolerance: None,
            bound_kind: BoundKind::Unavailable,
            provenance: provenance(),
        };
        assert_eq!(
            unavailable.status().unwrap(),
            QualificationStatus::Unresolved
        );
    }

    #[test]
    fn explicit_not_applicable_metric_is_retained_with_reason() {
        let item = MetricObservation {
            cohort: EquationCohort::VlasovPoissonPic,
            metric: MetricId::MagneticDivergenceResidual,
            units: "1/m".to_owned(),
            time_start: 0.0,
            time_end: 1.0,
            applicability: Applicability::NotApplicable,
            applicability_reason: Some(
                "electrostatic cohort has no evolved magnetic field".to_owned(),
            ),
            raw_value: None,
            error_estimate: None,
            lower_error_bound: None,
            upper_error_bound: None,
            acceptance_tolerance: None,
            bound_kind: BoundKind::Unavailable,
            provenance: provenance(),
        };
        assert_eq!(item.status().unwrap(), QualificationStatus::NotApplicable);
    }

    #[test]
    fn trajectory_requires_all_metrics_and_preserves_bound_strength() {
        let field = observation(
            MetricId::FieldL2Error,
            0.04,
            0.02,
            0.08,
            0.1,
            BoundKind::Probabilistic {
                confidence_level: 0.95,
            },
        );
        let energy = observation(
            MetricId::TotalEnergy,
            0.01,
            0.005,
            0.02,
            0.03,
            BoundKind::Deterministic,
        );
        let mut energy = energy;
        energy.provenance.source_revision = "energy-diagnostic-build".to_owned();
        energy.provenance.operator_id = "energy-ledger-v2".to_owned();
        let result = ledger(vec![field, energy])
            .qualify(&policy(vec![
                requirement(MetricId::FieldL2Error, 0.1),
                requirement(MetricId::TotalEnergy, 0.03),
            ]))
            .unwrap();
        assert_eq!(result.status, QualificationStatus::Pass);
        assert_eq!(result.decisions.len(), 2);
        assert_eq!(
            result.decisions[0].bound_kind,
            Some(BoundKind::Probabilistic {
                confidence_level: 0.95
            })
        );
    }

    #[test]
    fn missing_required_metric_keeps_trajectory_unresolved() {
        let field = observation(
            MetricId::FieldL2Error,
            0.02,
            0.01,
            0.03,
            0.1,
            BoundKind::Deterministic,
        );
        let result = ledger(vec![field])
            .qualify(&policy(vec![
                requirement(MetricId::FieldL2Error, 0.1),
                requirement(MetricId::TotalEnergy, 0.1),
            ]))
            .unwrap();
        assert_eq!(result.status, QualificationStatus::Unresolved);
        assert_eq!(
            result.decisions[1].reason.as_deref(),
            Some("required metric is missing")
        );
    }

    #[test]
    fn not_applicable_is_allowed_only_when_policy_says_so() {
        let not_applicable = MetricObservation {
            cohort: EquationCohort::VlasovPoissonPic,
            metric: MetricId::MagneticDivergenceResidual,
            units: "1/m".to_owned(),
            time_start: 0.0,
            time_end: 1.0,
            applicability: Applicability::NotApplicable,
            applicability_reason: Some("no magnetic field in this equation cohort".to_owned()),
            raw_value: None,
            error_estimate: None,
            lower_error_bound: None,
            upper_error_bound: None,
            acceptance_tolerance: None,
            bound_kind: BoundKind::Unavailable,
            provenance: provenance(),
        };
        let mut req = requirement(MetricId::MagneticDivergenceResidual, 0.0);
        req.units = "1/m".to_owned();
        req.applicability_rule = ApplicabilityRule::NotApplicableAllowed;
        let result = ledger(vec![not_applicable])
            .qualify(&policy(vec![req]))
            .unwrap();
        assert_eq!(result.status, QualificationStatus::NotApplicable);
    }

    #[test]
    fn stale_provenance_is_unresolved_not_accepted() {
        let mut item = observation(
            MetricId::FieldL2Error,
            0.02,
            0.01,
            0.03,
            0.1,
            BoundKind::Deterministic,
        );
        item.provenance.run_id = "different-run".to_owned();
        let result = ledger(vec![item])
            .qualify(&policy(vec![requirement(MetricId::FieldL2Error, 0.1)]))
            .unwrap();
        assert_eq!(result.status, QualificationStatus::Unresolved);
        assert!(
            result.decisions[0]
                .reason
                .as_deref()
                .unwrap()
                .contains("provenance")
        );
    }

    #[test]
    fn cohort_segment_and_duplicate_mismatches_fail_closed() {
        let mut wrong_cohort = observation(
            MetricId::FieldL2Error,
            0.02,
            0.01,
            0.03,
            0.1,
            BoundKind::Deterministic,
        );
        wrong_cohort.cohort = EquationCohort::VlasovMaxwellPic;
        assert_eq!(
            ledger(vec![wrong_cohort])
                .qualify(&policy(vec![requirement(MetricId::FieldL2Error, 0.1)])),
            Err(LedgerError::CohortMismatch)
        );

        let mut wrong_segment = observation(
            MetricId::FieldL2Error,
            0.02,
            0.01,
            0.03,
            0.1,
            BoundKind::Deterministic,
        );
        wrong_segment.time_end = 0.5;
        assert_eq!(
            ledger(vec![wrong_segment])
                .qualify(&policy(vec![requirement(MetricId::FieldL2Error, 0.1)])),
            Err(LedgerError::SegmentMismatch)
        );

        let duplicate_policy = policy(vec![
            requirement(MetricId::FieldL2Error, 0.1),
            requirement(MetricId::FieldL2Error, 0.2),
        ]);
        assert!(matches!(
            duplicate_policy.validate(),
            Err(LedgerError::DuplicateMetricRequirement { .. })
        ));
    }

    fn component(source: &str, upper_bound: f64, bound_kind: BoundKind) -> ErrorComponentBound {
        ErrorComponentBound {
            cohort: EquationCohort::VlasovPoissonPic,
            target_metric: MetricId::FieldL2Error,
            units: "V/m".to_owned(),
            time_start: 0.0,
            time_end: 1.0,
            provenance: provenance(),
            source: source.to_owned(),
            upper_bound,
            bound_kind,
        }
    }

    #[test]
    fn deterministic_error_components_use_conservative_sum() {
        let result = aggregate_error_components(&[
            component("particle_sampling", 0.2, BoundKind::Deterministic),
            component("field_operator", 0.3, BoundKind::Deterministic),
        ])
        .unwrap();
        assert!((result.upper_bound - 0.5).abs() < 1.0e-14);
        assert_eq!(result.bound_kind, AggregatedBoundKind::Deterministic);
        assert_eq!(result.component_count, 2);
    }

    #[test]
    fn probabilistic_components_use_union_bound_without_independence() {
        let result = aggregate_error_components(&[
            component(
                "particle_sampling",
                0.2,
                BoundKind::Probabilistic {
                    confidence_level: 0.95,
                },
            ),
            component(
                "field_operator",
                0.1,
                BoundKind::Probabilistic {
                    confidence_level: 0.99,
                },
            ),
            component("time_integration", 0.05, BoundKind::Deterministic),
        ])
        .unwrap();
        assert!((result.upper_bound - 0.35).abs() < 1.0e-14);
        assert_eq!(
            result.bound_kind,
            AggregatedBoundKind::Probabilistic {
                confidence_lower_bound: 0.94
            }
        );
    }

    #[test]
    fn distinct_diagnostic_revisions_can_share_one_run_identity() {
        let mut second = component("independent_mmd_oracle", 0.2, BoundKind::Deterministic);
        second.provenance.source_revision = "diagnostic-revision-b".to_owned();
        second.provenance.operator_id = "independent-mmd-v2".to_owned();
        let result = aggregate_error_components(&[
            component("solver-reference", 0.1, BoundKind::Deterministic),
            second,
        ])
        .unwrap();
        assert!((result.upper_bound - 0.3).abs() < 1.0e-14);
        assert_eq!(result.component_provenance.len(), 2);
        assert_ne!(
            result.component_provenance[0].source_revision,
            result.component_provenance[1].source_revision
        );
        assert_ne!(
            result.component_provenance[0].operator_id,
            result.component_provenance[1].operator_id
        );
    }

    #[test]
    fn empirical_components_cannot_enter_qualified_error_budget() {
        assert_eq!(
            aggregate_error_components(&[component(
                "measured_residual",
                0.1,
                BoundKind::Empirical
            )]),
            Err(LedgerError::UnqualifiedErrorComponent)
        );
    }

    #[test]
    fn different_metric_units_cohort_segment_or_provenance_reject_aggregation() {
        let first = component("sampling", 0.1, BoundKind::Deterministic);

        let mut changed = component("field", 0.2, BoundKind::Deterministic);
        changed.target_metric = MetricId::ParticleForceL2Error;
        assert_eq!(
            aggregate_error_components(&[first.clone(), changed]),
            Err(LedgerError::ErrorBudgetTargetMismatch)
        );

        let mut changed = component("field", 0.2, BoundKind::Deterministic);
        changed.units = "V".to_owned();
        assert_eq!(
            aggregate_error_components(&[first.clone(), changed]),
            Err(LedgerError::ErrorBudgetUnitMismatch)
        );

        let mut changed = component("field", 0.2, BoundKind::Deterministic);
        changed.cohort = EquationCohort::VlasovMaxwellPic;
        assert_eq!(
            aggregate_error_components(&[first.clone(), changed]),
            Err(LedgerError::ErrorBudgetCohortMismatch)
        );

        let mut changed = component("field", 0.2, BoundKind::Deterministic);
        changed.time_end = 2.0;
        assert_eq!(
            aggregate_error_components(&[first.clone(), changed]),
            Err(LedgerError::ErrorBudgetSegmentMismatch)
        );

        let mut changed = component("field", 0.2, BoundKind::Deterministic);
        changed.provenance.run_id = "other-run".to_owned();
        assert_eq!(
            aggregate_error_components(&[first, changed]),
            Err(LedgerError::ErrorBudgetProvenanceMismatch)
        );
    }

    #[test]
    fn invalid_numbers_confidence_and_provenance_fail_closed() {
        assert!(validate_confidence(1.0).is_err());
        assert_eq!(
            aggregate_error_components(&[component(
                "sampling",
                f64::NAN,
                BoundKind::Deterministic
            )]),
            Err(LedgerError::InvalidErrorBound)
        );

        let mut invalid = observation(
            MetricId::FieldL2Error,
            0.02,
            0.03,
            0.01,
            0.1,
            BoundKind::Deterministic,
        );
        assert_eq!(invalid.validate(), Err(LedgerError::InvalidErrorInterval));

        invalid = observation(
            MetricId::FieldL2Error,
            0.02,
            0.01,
            0.03,
            0.1,
            BoundKind::Deterministic,
        );
        invalid.raw_value = Some(f64::INFINITY);
        assert_eq!(invalid.validate(), Err(LedgerError::InvalidNumericValue));

        let mut missing = provenance();
        missing.input_digest.clear();
        assert_eq!(
            missing.validate(),
            Err(LedgerError::MissingProvenance {
                field: "input_digest"
            })
        );
    }
}
