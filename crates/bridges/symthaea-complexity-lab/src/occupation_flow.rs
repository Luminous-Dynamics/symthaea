//! Exact finite-horizon occupation-flow LPs for stagewise-free interval Markov models.
//!
//! This module encodes an explicitly row-separable, stagewise-independent
//! interval transition model as a rational LP. It does not support shared
//! parameters, cross-row coupling, cross-time coupling, or history-dependent
//! ambiguity. The source-to-LP construction is exposed for independent
//! primal/dual verification through `lp_certificate`.

use crate::lp_certificate::{
    verify_primal_dual_gap, verify_primal_dual_optimality, LpCertificateError,
    PrimalDualCertificate, RationalLp, VerifiedLpGap, VerifiedLpOptimum,
};
use crate::semiring::CheckedRational;
use std::error::Error;
use std::fmt;
use std::cmp::Ordering;
use sha2::{Digest, Sha256};

/// Closed rational interval for one transition probability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransitionInterval {
    pub lower: CheckedRational,
    pub upper: CheckedRational,
}

/// A finite-horizon Markov model with stagewise-independent, row-separable
/// interval transition uncertainty.
///
/// `transitions[t][i][j]` bounds P(X_(t+1)=j | X_t=i). Each row's feasible
/// transition distributions are the intersection of these intervals with the
/// probability simplex. Rows and stages are independently selectable; this is
/// the exact scope condition required by the occupation-flow reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagewiseIntervalMarkovModel {
    initial: Vec<CheckedRational>,
    transitions: Vec<Vec<Vec<TransitionInterval>>>,
}

impl StagewiseIntervalMarkovModel {
    pub fn new(
        initial: Vec<CheckedRational>,
        transitions: Vec<Vec<Vec<TransitionInterval>>>,
    ) -> Result<Self, OccupationFlowError> {
        if initial.is_empty() {
            return Err(OccupationFlowError::EmptyStateSpace);
        }
        let states = initial.len();
        for (state, probability) in initial.iter().enumerate() {
            if probability.is_negative() || compare_nonnegative(*probability, CheckedRational::one()) == Ordering::Greater {
                return Err(OccupationFlowError::InvalidInitialProbability { state });
            }
        }
        let initial_mass = checked_sum(&initial)?;
        if initial_mass != CheckedRational::one() {
            return Err(OccupationFlowError::InitialDistributionDoesNotSumToOne);
        }

        for (stage, matrix) in transitions.iter().enumerate() {
            if matrix.len() != states {
                return Err(OccupationFlowError::TransitionRowCountMismatch {
                    stage,
                    expected: states,
                    actual: matrix.len(),
                });
            }
            for (row, intervals) in matrix.iter().enumerate() {
                if intervals.len() != states {
                    return Err(OccupationFlowError::TransitionColumnCountMismatch {
                        stage,
                        row,
                        expected: states,
                        actual: intervals.len(),
                    });
                }
                let mut lower_sum = CheckedRational::zero();
                let mut upper_sum = CheckedRational::zero();
                for (column, interval) in intervals.iter().enumerate() {
                    if interval.lower.is_negative()
                        || interval.upper.is_negative()
                        || compare_nonnegative(interval.upper, CheckedRational::one()) == Ordering::Greater
                        || compare_nonnegative(interval.lower, interval.upper) == Ordering::Greater
                    {
                        return Err(OccupationFlowError::InvalidTransitionInterval {
                            stage,
                            row,
                            column,
                        });
                    }
                    lower_sum = lower_sum.checked_add(interval.lower)
                        .map_err(|_| OccupationFlowError::ArithmeticOverflow)?;
                    upper_sum = upper_sum.checked_add(interval.upper)
                        .map_err(|_| OccupationFlowError::ArithmeticOverflow)?;
                }
                if compare_nonnegative(lower_sum, CheckedRational::one()) == Ordering::Greater
                    || compare_nonnegative(upper_sum, CheckedRational::one()) == Ordering::Less
                {
                    return Err(OccupationFlowError::InfeasibleTransitionRow { stage, row });
                }
            }
        }

        Ok(Self { initial, transitions })
    }

    pub fn state_count(&self) -> usize {
        self.initial.len()
    }

    /// Number of occupancy layers, including the initial distribution.
    pub fn period_count(&self) -> usize {
        self.transitions.len() + 1
    }

    pub fn initial(&self) -> &[CheckedRational] {
        &self.initial
    }

    pub fn transitions(&self) -> &[Vec<Vec<TransitionInterval>>] {
        &self.transitions
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UtilityDirection {
    Maximize,
    Minimize,
}

/// Hard pre-allocation limits for dense occupation-flow LP construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OccupationFlowBudget {
    pub max_variables: usize,
    pub max_constraints: usize,
    pub max_matrix_entries: usize,
    /// Maximum bit width of any rational numerator or denominator.
    pub max_rational_bit_width: usize,
}

/// Canonical variable layout for the occupation-flow LP.
///
/// First are all occupancy variables in (time, state) order, followed by all
/// transition-flow variables in (stage, source-state, destination-state) order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OccupationFlowLayout {
    periods: usize,
    states: usize,
    flow_start: usize,
    variable_count: usize,
}

impl OccupationFlowLayout {
    pub fn period_count(&self) -> usize { self.periods }
    pub fn state_count(&self) -> usize { self.states }
    pub fn variable_count(&self) -> usize { self.variable_count }

    pub fn occupancy_variable(&self, time: usize, state: usize) -> Option<usize> {
        if time >= self.periods || state >= self.states {
            return None;
        }
        time.checked_mul(self.states)?.checked_add(state)
    }

    pub fn flow_variable(&self, stage: usize, source: usize, destination: usize) -> Option<usize> {
        if stage >= self.periods.saturating_sub(1) || source >= self.states || destination >= self.states {
            return None;
        }
        let offset = stage.checked_mul(self.states.checked_mul(self.states)?)?
            .checked_add(source.checked_mul(self.states)?)?
            .checked_add(destination)?;
        self.flow_start.checked_add(offset)
    }
}

/// Constructed rational LP plus the canonical source-variable layout.
///
/// For `Minimize`, the LP maximizes the negated utility. Convert the verified
/// LP objective back to the original utility using `utility_from_lp_objective`.
/// Verified utility optimum bound to both source and compiled LP identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedOccupationFlowOptimum {
    pub source_model_sha256: [u8; 32],
    pub compiled_lp_sha256: [u8; 32],
    pub lp_objective: CheckedRational,
    pub utility: CheckedRational,
}

/// Proof material bound to one source model and one compiled LP.
/// Mutating either digest invalidates replay against the corresponding LP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OccupationFlowCertificate {
    pub source_model_sha256: [u8; 32],
    pub compiled_lp_sha256: [u8; 32],
    pub primal_dual: PrimalDualCertificate,
}

/// Verified original-utility bracket for maximize or minimize queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedOccupationFlowGap {
    pub source_model_sha256: [u8; 32],
    pub compiled_lp_sha256: [u8; 32],
    pub lower_utility: CheckedRational,
    pub upper_utility: CheckedRational,
    pub absolute_gap: CheckedRational,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OccupationFlowLp {
    lp: RationalLp,
    layout: OccupationFlowLayout,
    direction: UtilityDirection,
    source_model_sha256: [u8; 32],
    compiled_lp_sha256: [u8; 32],
}

impl OccupationFlowLp {
    pub fn problem(&self) -> &RationalLp { &self.lp }
    pub fn layout(&self) -> OccupationFlowLayout { self.layout }
    pub fn direction(&self) -> UtilityDirection { self.direction }
    /// SHA-256 of the versioned canonical source interval-model transcript.
    pub fn source_model_sha256(&self) -> [u8; 32] { self.source_model_sha256 }
    /// SHA-256 of the versioned canonical compiled LP transcript, including
    /// the source model digest, direction, matrix, RHS and objective.
    pub fn compiled_lp_sha256(&self) -> [u8; 32] { self.compiled_lp_sha256 }

    /// Attach the source and compiled-LP identities to candidate proof
    /// material. This does not validate the candidate; call a verify method.
    pub fn bind_certificate(
        &self,
        primal_dual: PrimalDualCertificate,
    ) -> OccupationFlowCertificate {
        OccupationFlowCertificate {
            source_model_sha256: self.source_model_sha256,
            compiled_lp_sha256: self.compiled_lp_sha256,
            primal_dual,
        }
    }

    /// Verify a digest-bound exact primal/dual certificate.
    pub fn verify_bound_optimality(
        &self,
        certificate: &OccupationFlowCertificate,
    ) -> Result<VerifiedOccupationFlowOptimum, OccupationFlowError> {
        self.verify_certificate_identity(certificate)?;
        self.verify_primal_dual_optimality(&certificate.primal_dual)
    }

    /// Verify a digest-bound primal/dual interval certificate.
    pub fn verify_bound_gap(
        &self,
        certificate: &OccupationFlowCertificate,
        requested_gap: CheckedRational,
    ) -> Result<VerifiedOccupationFlowGap, OccupationFlowError> {
        self.verify_certificate_identity(certificate)?;
        self.verify_primal_dual_gap(&certificate.primal_dual, requested_gap)
    }

    fn verify_certificate_identity(
        &self,
        certificate: &OccupationFlowCertificate,
    ) -> Result<(), OccupationFlowError> {
        if certificate.source_model_sha256 != self.source_model_sha256 {
            return Err(OccupationFlowError::CertificateSourceDigestMismatch);
        }
        if certificate.compiled_lp_sha256 != self.compiled_lp_sha256 {
            return Err(OccupationFlowError::CertificateCompiledLpDigestMismatch);
        }
        Ok(())
    }

    /// Verify exact primal/dual optimality and return a result bound to the
    /// source-model and compiled-LP SHA-256 identities.
    pub fn verify_primal_dual_optimality(
        &self,
        certificate: &PrimalDualCertificate,
    ) -> Result<VerifiedOccupationFlowOptimum, OccupationFlowError> {
        let VerifiedLpOptimum { objective } =
            verify_primal_dual_optimality(&self.lp, certificate)?;
        Ok(VerifiedOccupationFlowOptimum {
            source_model_sha256: self.source_model_sha256,
            compiled_lp_sha256: self.compiled_lp_sha256,
            lp_objective: objective,
            utility: self.utility_from_lp_objective(objective)?,
        })
    }

    /// Verify a primal/dual delta bracket and transform it back to the
    /// original utility orientation. For minimization, the LP maximizes -U,
    /// so the utility interval reverses and changes sign.
    pub fn verify_primal_dual_gap(
        &self,
        certificate: &PrimalDualCertificate,
        requested_gap: CheckedRational,
    ) -> Result<VerifiedOccupationFlowGap, OccupationFlowError> {
        let VerifiedLpGap {
            lower_bound,
            upper_bound,
            absolute_gap,
        } = verify_primal_dual_gap(&self.lp, certificate, requested_gap)?;
        let (lower_utility, upper_utility) = match self.direction {
            UtilityDirection::Maximize => (lower_bound, upper_bound),
            UtilityDirection::Minimize => (
                negate_rational(upper_bound)?,
                negate_rational(lower_bound)?,
            ),
        };
        Ok(VerifiedOccupationFlowGap {
            source_model_sha256: self.source_model_sha256,
            compiled_lp_sha256: self.compiled_lp_sha256,
            lower_utility,
            upper_utility,
            absolute_gap,
        })
    }

    pub fn utility_from_lp_objective(
        &self,
        objective: CheckedRational,
    ) -> Result<CheckedRational, OccupationFlowError> {
        match self.direction {
            UtilityDirection::Maximize => Ok(objective),
            UtilityDirection::Minimize => {
                let numerator = objective.numerator().checked_neg()
                    .ok_or(OccupationFlowError::ArithmeticOverflow)?;
                CheckedRational::new(numerator, objective.denominator())
                    .map_err(|_| OccupationFlowError::ArithmeticOverflow)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OccupationFlowError {
    EmptyStateSpace,
    InvalidInitialProbability { state: usize },
    InitialDistributionDoesNotSumToOne,
    TransitionRowCountMismatch { stage: usize, expected: usize, actual: usize },
    TransitionColumnCountMismatch { stage: usize, row: usize, expected: usize, actual: usize },
    InvalidTransitionInterval { stage: usize, row: usize, column: usize },
    InfeasibleTransitionRow { stage: usize, row: usize },
    ScorePeriodCountMismatch { expected: usize, actual: usize },
    ScoreStateCountMismatch { time: usize, expected: usize, actual: usize },
    ArithmeticOverflow,
    DimensionOverflow,
    BudgetExceeded {
        resource: &'static str,
        required: usize,
        limit: usize,
    },
    AllocationFailed {
        resource: &'static str,
        requested: usize,
    },
    CertificateSourceDigestMismatch,
    CertificateCompiledLpDigestMismatch,
    Lp(LpCertificateError),
}

impl fmt::Display for OccupationFlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyStateSpace => write!(f, "occupation-flow model must have at least one state"),
            Self::InvalidInitialProbability { state } => write!(f, "invalid initial probability at state {state}"),
            Self::InitialDistributionDoesNotSumToOne => write!(f, "initial distribution does not sum exactly to one"),
            Self::TransitionRowCountMismatch { stage, expected, actual } => write!(f, "transition stage {stage} has {actual} rows; expected {expected}"),
            Self::TransitionColumnCountMismatch { stage, row, expected, actual } => write!(f, "transition stage {stage}, row {row} has {actual} columns; expected {expected}"),
            Self::InvalidTransitionInterval { stage, row, column } => write!(f, "invalid transition interval at stage {stage}, row {row}, column {column}"),
            Self::InfeasibleTransitionRow { stage, row } => write!(f, "transition interval row is disjoint from the probability simplex at stage {stage}, row {row}"),
            Self::ScorePeriodCountMismatch { expected, actual } => write!(f, "score has {actual} periods; expected {expected}"),
            Self::ScoreStateCountMismatch { time, expected, actual } => write!(f, "score at time {time} has {actual} states; expected {expected}"),
            Self::ArithmeticOverflow => write!(f, "checked rational overflow in occupation-flow construction"),
            Self::DimensionOverflow => write!(f, "occupation-flow LP dimensions overflow usize"),
            Self::BudgetExceeded { resource, required, limit } => write!(
                f,
                "occupation-flow {resource} requires {required}; configured limit is {limit}"
            ),
            Self::AllocationFailed { resource, requested } => write!(
                f,
                "unable to reserve occupation-flow {resource} capacity {requested}"
            ),
            Self::CertificateSourceDigestMismatch => {
                write!(f, "occupation-flow source-model digest mismatch")
            }
            Self::CertificateCompiledLpDigestMismatch => {
                write!(f, "occupation-flow compiled-LP digest mismatch")
            }
            Self::Lp(error) => write!(f, "LP construction failed: {error}"),
        }
    }
}

impl Error for OccupationFlowError {}

impl From<LpCertificateError> for OccupationFlowError {
    fn from(value: LpCertificateError) -> Self {
        Self::Lp(value)
    }
}

fn rational_bit_width(value: CheckedRational) -> usize {
    let numerator = value.numerator().unsigned_abs();
    let denominator = value.denominator() as u128;
    let numerator_bits = if numerator == 0 {
        0
    } else {
        (u128::BITS - numerator.leading_zeros()) as usize
    };
    let denominator_bits = (u128::BITS - denominator.leading_zeros()) as usize;
    numerator_bits.max(denominator_bits)
}

fn check_rational_bit_width(
    value: CheckedRational,
    budget: OccupationFlowBudget,
) -> Result<(), OccupationFlowError> {
    let required = rational_bit_width(value);
    if required > budget.max_rational_bit_width {
        return Err(OccupationFlowError::BudgetExceeded {
            resource: "rational numerator/denominator bit width",
            required,
            limit: budget.max_rational_bit_width,
        });
    }
    Ok(())
}

fn validate_score_matrix(
    model: &StagewiseIntervalMarkovModel,
    scores: &[Vec<CheckedRational>],
    budget: OccupationFlowBudget,
) -> Result<(), OccupationFlowError> {
    if scores.len() != model.period_count() {
        return Err(OccupationFlowError::ScorePeriodCountMismatch {
            expected: model.period_count(),
            actual: scores.len(),
        });
    }
    for (time, row) in scores.iter().enumerate() {
        if row.len() != model.state_count() {
            return Err(OccupationFlowError::ScoreStateCountMismatch {
                time,
                expected: model.state_count(),
                actual: row.len(),
            });
        }
        for value in row {
            check_rational_bit_width(*value, budget)?;
        }
    }
    Ok(())
}

fn preflight_dimensions(
    model: &StagewiseIntervalMarkovModel,
    budget: OccupationFlowBudget,
) -> Result<(OccupationFlowLayout, usize), OccupationFlowError> {
    let periods = model.period_count();
    let states = model.state_count();

    for probability in &model.initial {
        check_rational_bit_width(*probability, budget)?;
    }
    for stage in &model.transitions {
        for row in stage {
            for interval in row {
                check_rational_bit_width(interval.lower, budget)?;
                check_rational_bit_width(interval.upper, budget)?;
            }
        }
    }

    let occupancy_count = periods.checked_mul(states)
        .ok_or(OccupationFlowError::DimensionOverflow)?;
    let flow_count = model.transitions.len()
        .checked_mul(states)
        .and_then(|v| v.checked_mul(states))
        .ok_or(OccupationFlowError::DimensionOverflow)?;
    let variable_count = occupancy_count.checked_add(flow_count)
        .ok_or(OccupationFlowError::DimensionOverflow)?;
    let constraint_count = states.checked_mul(2)
        .and_then(|initial_rows| {
            let per_stage = states.checked_mul(4)?
                .checked_add(states.checked_mul(states)?.checked_mul(2)?)?;
            model.transitions.len().checked_mul(per_stage)?
                .checked_add(initial_rows)
        })
        .ok_or(OccupationFlowError::DimensionOverflow)?;
    let matrix_entries = constraint_count.checked_mul(variable_count)
        .ok_or(OccupationFlowError::DimensionOverflow)?;

    for (resource, required, limit) in [
        ("variables", variable_count, budget.max_variables),
        ("constraints", constraint_count, budget.max_constraints),
        ("dense matrix entries", matrix_entries, budget.max_matrix_entries),
    ] {
        if required > limit {
            return Err(OccupationFlowError::BudgetExceeded { resource, required, limit });
        }
    }

    Ok((
        OccupationFlowLayout {
            periods,
            states,
            flow_start: occupancy_count,
            variable_count,
        },
        constraint_count,
    ))
}

/// Build an LP for the worst-case regret of one action against a comparator.
///
/// The objective is U_comparator(omega) - U_selected(omega), optimized over
/// one shared occupation-flow trajectory. This preserves dependence between
/// the two score functions; subtracting independently optimized utility
/// intervals would generally be invalid.
///
/// Scope boundary: both alternatives are evaluated under the same uncertain
/// transition dynamics and differ only through their state-score functions.
/// This is not a general MDP regret compiler for actions that induce different
/// transition kernels or policy-dependent occupancy processes. Such models
/// require an explicit joint coupling of the action-conditional kernels.
pub fn build_pairwise_worst_case_regret_lp(
    model: &StagewiseIntervalMarkovModel,
    selected_scores: &[Vec<CheckedRational>],
    comparator_scores: &[Vec<CheckedRational>],
    budget: OccupationFlowBudget,
) -> Result<OccupationFlowLp, OccupationFlowError> {
    let periods = model.period_count();
    let states = model.state_count();
    for scores in [selected_scores, comparator_scores] {
        if scores.len() != periods {
            return Err(OccupationFlowError::ScorePeriodCountMismatch {
                expected: periods,
                actual: scores.len(),
            });
        }
        for (time, row) in scores.iter().enumerate() {
            if row.len() != states {
                return Err(OccupationFlowError::ScoreStateCountMismatch {
                    time,
                    expected: states,
                    actual: row.len(),
                });
            }
        }
    }

    // Reject oversized inputs and an over-budget LP before constructing the
    // temporary score-difference grid.
    validate_score_matrix(model, selected_scores, budget)?;
    validate_score_matrix(model, comparator_scores, budget)?;
    let _ = preflight_dimensions(model, budget)?;

    let mut regret_scores = Vec::new();
    regret_scores.try_reserve_exact(periods).map_err(|_| {
        OccupationFlowError::AllocationFailed {
            resource: "regret score rows",
            requested: periods,
        }
    })?;
    for time in 0..periods {
        let mut row = Vec::new();
        row.try_reserve_exact(states).map_err(|_| {
            OccupationFlowError::AllocationFailed {
                resource: "regret score entries",
                requested: states,
            }
        })?;
        for state in 0..states {
            let comparator = comparator_scores[time][state];
            let selected = selected_scores[time][state];
            let negative_selected = CheckedRational::new(
                selected.numerator().checked_neg()
                    .ok_or(OccupationFlowError::ArithmeticOverflow)?,
                selected.denominator(),
            ).map_err(|_| OccupationFlowError::ArithmeticOverflow)?;
            row.push(comparator.checked_add(negative_selected)
                .map_err(|_| OccupationFlowError::ArithmeticOverflow)?);
        }
        regret_scores.push(row);
    }

    build_occupation_flow_lp(model, &regret_scores, UtilityDirection::Maximize, budget)
}

/// Build the exact occupation-flow LP for cumulative finite-horizon utility.
///
/// Variables:
/// - omega[t,i] >= 0: state occupancy at each time;
/// - f[t,i,j] >= 0: probability mass flowing from i to j at stage t.
///
/// Constraints encode the initial distribution, row-flow conservation,
/// next-state occupancy conservation and interval transition bounds.
/// Equalities are represented by paired <= rows so the resulting LP matches
/// the verifier's canonical standard form.
pub fn build_occupation_flow_lp(
    model: &StagewiseIntervalMarkovModel,
    state_scores: &[Vec<CheckedRational>],
    direction: UtilityDirection,
    budget: OccupationFlowBudget,
) -> Result<OccupationFlowLp, OccupationFlowError> {
    validate_score_matrix(model, state_scores, budget)?;
    let (layout, constraint_count) = preflight_dimensions(model, budget)?;
    let periods = model.period_count();
    let states = model.state_count();

    let mut matrix: Vec<Vec<CheckedRational>> = Vec::new();
    matrix.try_reserve_exact(constraint_count).map_err(|_| {
        OccupationFlowError::AllocationFailed {
            resource: "constraint rows",
            requested: constraint_count,
        }
    })?;
    let mut rhs: Vec<CheckedRational> = Vec::new();
    rhs.try_reserve_exact(constraint_count).map_err(|_| {
        OccupationFlowError::AllocationFailed {
            resource: "right-hand-side entries",
            requested: constraint_count,
        }
    })?;

    // Initial occupancy: omega[0,i] = initial[i].
    for state in 0..states {
        let mut row = zero_row(layout.variable_count())?;
        row[layout.occupancy_variable(0, state).unwrap()] = CheckedRational::one();
        add_equality(&mut matrix, &mut rhs, row, model.initial[state])?;
    }

    for stage in 0..model.transitions.len() {
        // Outflow from each source state equals its current occupancy.
        for source in 0..states {
            let mut row = zero_row(layout.variable_count())?;
            row[layout.occupancy_variable(stage, source).unwrap()] =
                CheckedRational::from_integer(-1);
            for destination in 0..states {
                row[layout.flow_variable(stage, source, destination).unwrap()] =
                    CheckedRational::one();
            }
            add_equality(&mut matrix, &mut rhs, row, CheckedRational::zero())?;
        }

        // Inflow to each destination state equals next-period occupancy.
        for destination in 0..states {
            let mut row = zero_row(layout.variable_count())?;
            row[layout.occupancy_variable(stage + 1, destination).unwrap()] =
                CheckedRational::one();
            for source in 0..states {
                row[layout.flow_variable(stage, source, destination).unwrap()] =
                    CheckedRational::from_integer(-1);
            }
            add_equality(&mut matrix, &mut rhs, row, CheckedRational::zero())?;
        }

        // Each row's local transition bounds: lower*omega <= flow <= upper*omega.
        for source in 0..states {
            for destination in 0..states {
                let interval = model.transitions[stage][source][destination];
                let occupancy = layout.occupancy_variable(stage, source).unwrap();
                let flow = layout.flow_variable(stage, source, destination).unwrap();

                let mut lower_row = zero_row(layout.variable_count())?;
                lower_row[occupancy] = interval.lower;
                lower_row[flow] = CheckedRational::from_integer(-1);
                add_inequality(&mut matrix, &mut rhs, lower_row, CheckedRational::zero());

                let mut upper_row = zero_row(layout.variable_count())?;
                upper_row[flow] = CheckedRational::one();
                upper_row[occupancy] = CheckedRational::new(
                    interval.upper.numerator().checked_neg()
                        .ok_or(OccupationFlowError::ArithmeticOverflow)?,
                    interval.upper.denominator(),
                ).map_err(|_| OccupationFlowError::ArithmeticOverflow)?;
                add_inequality(&mut matrix, &mut rhs, upper_row, CheckedRational::zero());
            }
        }
    }

    let mut objective = zero_row(layout.variable_count())?;
    for time in 0..periods {
        for state in 0..states {
            let score = state_scores[time][state];
            objective[layout.occupancy_variable(time, state).unwrap()] = match direction {
                UtilityDirection::Maximize => score,
                UtilityDirection::Minimize => CheckedRational::new(
                    score.numerator().checked_neg()
                        .ok_or(OccupationFlowError::ArithmeticOverflow)?,
                    score.denominator(),
                ).map_err(|_| OccupationFlowError::ArithmeticOverflow)?,
            };
        }
    }

    let lp = RationalLp::new(matrix, rhs, objective)?;
    let source_model_sha256 = canonical_interval_model_sha256(model);
    let compiled_lp_sha256 = canonical_occupation_lp_sha256(
        &lp,
        layout,
        direction,
        source_model_sha256,
    );
    Ok(OccupationFlowLp {
        lp,
        layout,
        direction,
        source_model_sha256,
        compiled_lp_sha256,
    })
}

/// Versioned canonical SHA-256 identity of a stagewise interval model.
///
/// Encoding contract (all integers little-endian):
/// magic bytes, state count (u64), period count (u64), initial probabilities
/// in state order (numerator i128, denominator i128), then every transition
/// interval in stage/source/destination order (lower rational, upper rational).
/// This is streamed into SHA-256; no full transcript buffer is allocated.
pub fn canonical_interval_model_sha256(
    model: &StagewiseIntervalMarkovModel,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"SYMPGM-OCCUPATION-MODEL\0\x01");
    hasher.update((model.state_count() as u64).to_le_bytes());
    hasher.update((model.period_count() as u64).to_le_bytes());
    for probability in &model.initial {
        hash_rational(&mut hasher, *probability);
    }
    for stage in &model.transitions {
        for row in stage {
            for interval in row {
                hash_rational(&mut hasher, interval.lower);
                hash_rational(&mut hasher, interval.upper);
            }
        }
    }
    hasher.finalize().into()
}

/// Versioned canonical identity of the compiled standard-form LP.
/// Source digest and direction are included to bind the semantics, not just
/// the dimensions and resulting coefficient matrix.
fn canonical_occupation_lp_sha256(
    lp: &RationalLp,
    layout: OccupationFlowLayout,
    direction: UtilityDirection,
    source_model_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"SYMPGM-OCCUPATION-LP\0\x01");
    hasher.update(source_model_sha256);
    hasher.update((layout.period_count() as u64).to_le_bytes());
    hasher.update((layout.state_count() as u64).to_le_bytes());
    hasher.update((lp.row_count() as u64).to_le_bytes());
    hasher.update((lp.column_count() as u64).to_le_bytes());
    hasher.update([match direction {
        UtilityDirection::Maximize => 0,
        UtilityDirection::Minimize => 1,
    }]);
    for row in lp.matrix() {
        for coefficient in row {
            hash_rational(&mut hasher, *coefficient);
        }
    }
    for rhs in lp.rhs() {
        hash_rational(&mut hasher, *rhs);
    }
    for coefficient in lp.objective() {
        hash_rational(&mut hasher, *coefficient);
    }
    hasher.finalize().into()
}

fn hash_rational(hasher: &mut Sha256, value: CheckedRational) {
    hasher.update(value.numerator().to_le_bytes());
    hasher.update(value.denominator().to_le_bytes());
}

pub fn sha256_hex(digest: [u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn zero_row(length: usize) -> Result<Vec<CheckedRational>, OccupationFlowError> {
    let mut row = Vec::new();
    row.try_reserve_exact(length).map_err(|_| OccupationFlowError::AllocationFailed {
        resource: "dense constraint row",
        requested: length,
    })?;
    row.resize(length, CheckedRational::zero());
    Ok(row)
}

fn add_inequality(
    matrix: &mut Vec<Vec<CheckedRational>>,
    rhs: &mut Vec<CheckedRational>,
    row: Vec<CheckedRational>,
    bound: CheckedRational,
) {
    matrix.push(row);
    rhs.push(bound);
}

fn add_equality(
    matrix: &mut Vec<Vec<CheckedRational>>,
    rhs: &mut Vec<CheckedRational>,
    row: Vec<CheckedRational>,
    value: CheckedRational,
) -> Result<(), OccupationFlowError> {
    let mut negated = Vec::new();
    negated.try_reserve_exact(row.len()).map_err(|_| {
        OccupationFlowError::AllocationFailed {
            resource: "negated equality row",
            requested: row.len(),
        }
    })?;
    for coefficient in &row {
        let numerator = coefficient.numerator().checked_neg()
            .ok_or(OccupationFlowError::ArithmeticOverflow)?;
        negated.push(CheckedRational::new(numerator, coefficient.denominator())
            .map_err(|_| OccupationFlowError::ArithmeticOverflow)?);
    }
    matrix.push(row);
    rhs.push(value);
    let negative_value = CheckedRational::new(
        value.numerator().checked_neg().ok_or(OccupationFlowError::ArithmeticOverflow)?,
        value.denominator(),
    ).map_err(|_| OccupationFlowError::ArithmeticOverflow)?;
    matrix.push(negated);
    rhs.push(negative_value);
    Ok(())
}

fn negate_rational(value: CheckedRational) -> Result<CheckedRational, OccupationFlowError> {
    let numerator = value.numerator().checked_neg()
        .ok_or(OccupationFlowError::ArithmeticOverflow)?;
    CheckedRational::new(numerator, value.denominator())
        .map_err(|_| OccupationFlowError::ArithmeticOverflow)
}

fn checked_sum(values: &[CheckedRational]) -> Result<CheckedRational, OccupationFlowError> {
    let mut sum = CheckedRational::zero();
    for value in values {
        sum = sum.checked_add(*value)
            .map_err(|_| OccupationFlowError::ArithmeticOverflow)?;
    }
    Ok(sum)
}

fn compare_nonnegative(left: CheckedRational, right: CheckedRational) -> Ordering {
    let (mut ln, mut ld) = (left.numerator() as u128, left.denominator() as u128);
    let (mut rn, mut rd) = (right.numerator() as u128, right.denominator() as u128);
    let mut reversed = false;
    loop {
        let li = ln / ld;
        let ri = rn / rd;
        if li != ri {
            let order = li.cmp(&ri);
            return if reversed { order.reverse() } else { order };
        }
        let lr = ln % ld;
        let rr = rn % rd;
        match (lr == 0, rr == 0) {
            (true, true) => return Ordering::Equal,
            (true, false) => return if reversed { Ordering::Greater } else { Ordering::Less },
            (false, true) => return if reversed { Ordering::Less } else { Ordering::Greater },
            (false, false) => {
                ln = ld;
                ld = lr;
                rn = rd;
                rd = rr;
                reversed = !reversed;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp_certificate::{
        verify_primal_dual_optimality, PrimalDualCertificate,
    };

    fn q(n: i128, d: i128) -> CheckedRational {
        CheckedRational::new(n, d).unwrap()
    }

    fn interval(lo_n: i128, lo_d: i128, hi_n: i128, hi_d: i128) -> TransitionInterval {
        TransitionInterval { lower: q(lo_n, lo_d), upper: q(hi_n, hi_d) }
    }

    fn fixture() -> StagewiseIntervalMarkovModel {
        StagewiseIntervalMarkovModel::new(
            vec![q(1, 1), q(0, 1)],
            vec![
                vec![
                    vec![interval(2, 5, 4, 5), interval(1, 5, 3, 5)],
                    vec![interval(3, 10, 7, 10), interval(3, 10, 7, 10)],
                ],
                vec![
                    vec![interval(1, 2, 9, 10), interval(1, 10, 1, 2)],
                    vec![interval(1, 5, 3, 5), interval(2, 5, 4, 5)],
                ],
            ],
        ).unwrap()
    }

    fn budget(max_entries: usize) -> OccupationFlowBudget {
        OccupationFlowBudget {
            max_variables: 14,
            max_constraints: 36,
            max_matrix_entries: max_entries,
            max_rational_bit_width: 64,
        }
    }

    fn sparse_vector(
        length: usize,
        entries: &[(usize, i128, i128)],
    ) -> Vec<CheckedRational> {
        let mut values = vec![q(0, 1); length];
        for &(index, numerator, denominator) in entries {
            values[index] = q(numerator, denominator);
        }
        values
    }

    #[test]
    fn constructs_exact_stagewise_interval_occupation_lp() {
        let model = fixture();
        let scores = vec![
            vec![q(0, 1), q(0, 1)],
            vec![q(0, 1), q(1, 1)],
            vec![q(0, 1), q(1, 1)],
        ];
        let built = build_occupation_flow_lp(&model, &scores, UtilityDirection::Maximize, budget(504)).unwrap();
        assert_eq!(built.layout().period_count(), 3);
        assert_eq!(built.layout().state_count(), 2);
        assert_eq!(built.layout().variable_count(), 14);
        assert_eq!(built.problem().row_count(), 36);
        assert_eq!(built.problem().column_count(), 14);
    }

    #[test]
    fn exact_lp_certificates_prove_the_four_utility_endpoints() {
        let model = fixture();
        let scores_a = vec![
            vec![q(0, 1), q(0, 1)],
            vec![q(0, 1), q(1, 1)],
            vec![q(0, 1), q(1, 1)],
        ];
        let scores_b = vec![
            vec![q(0, 1), q(0, 1)],
            vec![q(1, 1), q(0, 1)],
            vec![q(0, 1), q(0, 1)],
        ];

        let certificates = [
            (
                &scores_a,
                UtilityDirection::Maximize,
                q(32, 25),
                vec![(0,1,1),(2,2,5),(3,3,5),(4,8,25),(5,17,25),(6,2,5),(7,3,5),(10,1,5),(11,1,5),(12,3,25),(13,12,25)],
                vec![(0,32,25),(2,9,5),(4,1,2),(6,9,5),(8,1,2),(10,9,5),(15,13,10),(26,1,1),(31,1,1),(35,1,1)],
            ),
            (
                &scores_a,
                UtilityDirection::Minimize,
                q(9, 25),
                vec![(0,1,1),(2,4,5),(3,1,5),(4,21,25),(5,4,25),(6,4,5),(7,1,5),(10,18,25),(11,2,25),(12,3,25),(13,2,25)],
                vec![(1,9,25),(5,1,10),(9,1,10),(11,7,5),(14,13,10),(27,1,1),(30,1,1),(34,1,1)],
            ),
            (
                &scores_b,
                UtilityDirection::Maximize,
                q(4, 5),
                vec![(0,1,1),(2,4,5),(3,1,5),(4,19,25),(5,6,25),(6,4,5),(7,1,5),(10,18,25),(11,2,25),(12,1,25),(13,4,25)],
                vec![(0,4,5),(2,1,1),(4,1,1),(6,1,1),(8,1,1),(14,1,1)],
            ),
            (
                &scores_b,
                UtilityDirection::Minimize,
                q(2, 5),
                vec![(0,1,1),(2,2,5),(3,3,5),(4,12,25),(5,13,25),(6,2,5),(7,3,5),(10,9,25),(11,1,25),(12,3,25),(13,12,25)],
                vec![(1,2,5),(5,1,1),(9,1,1),(15,1,1)],
            ),
        ];

        for (scores, direction, expected_utility, primal_entries, dual_entries) in certificates {
            let built = build_occupation_flow_lp(&model, scores, direction, budget(504)).unwrap();
            let certificate = PrimalDualCertificate {
                primal: sparse_vector(14, &primal_entries),
                dual: sparse_vector(36, &dual_entries),
            };
            let bound_certificate = built.bind_certificate(certificate);
            let verified = built.verify_bound_optimality(&bound_certificate).unwrap();
            assert_eq!(verified.utility, expected_utility);
            assert_eq!(verified.source_model_sha256, built.source_model_sha256());
            assert_eq!(verified.compiled_lp_sha256, built.compiled_lp_sha256());
        }
    }

    #[test]
    fn endpoint_enumeration_confirms_utility_intervals_and_minimax_regret() {
        let lower = [
            [q(1, 5), q(3, 10)],
            [q(1, 10), q(2, 5)],
        ];
        let upper = [
            [q(3, 5), q(7, 10)],
            [q(1, 2), q(4, 5)],
        ];

        let one_minus = |value: CheckedRational| {
            let negative = CheckedRational::new(
                value.numerator().checked_neg().unwrap(),
                value.denominator(),
            ).unwrap();
            CheckedRational::one().checked_add(negative).unwrap()
        };

        let mut min_a: Option<CheckedRational> = None;
        let mut max_a: Option<CheckedRational> = None;
        let mut min_b: Option<CheckedRational> = None;
        let mut max_b: Option<CheckedRational> = None;
        let mut max_regret_a: Option<CheckedRational> = None;
        let mut max_regret_b: Option<CheckedRational> = None;

        // Four independently selectable local rows, each at one of its two
        // endpoints: 2^4 = 16 complete stagewise transition selections.
        for mask in 0usize..16 {
            let selected = |stage: usize, row: usize, bit: usize| {
                if mask & (1usize << bit) == 0 {
                    lower[stage][row]
                } else {
                    upper[stage][row]
                }
            };

            let p01_stage0 = selected(0, 0, 0);
            let _p11_stage0 = selected(0, 1, 1);
            let p01_stage1 = selected(1, 0, 2);
            let p11_stage1 = selected(1, 1, 3);

            let occupancy_t2_state1 = p01_stage0;
            let occupancy_t2_state0 = one_minus(occupancy_t2_state1);
            let occupancy_t3_state1 = occupancy_t2_state0
                .checked_mul(p01_stage1).unwrap()
                .checked_add(occupancy_t2_state1.checked_mul(p11_stage1).unwrap())
                .unwrap();

            let utility_a = occupancy_t2_state1
                .checked_add(occupancy_t3_state1).unwrap();
            let utility_b = occupancy_t2_state0;

            min_a = Some(min_a.map_or(utility_a, |v| if compare_nonnegative(utility_a, v) == Ordering::Less { utility_a } else { v }));
            max_a = Some(max_a.map_or(utility_a, |v| if compare_nonnegative(utility_a, v) == Ordering::Greater { utility_a } else { v }));
            min_b = Some(min_b.map_or(utility_b, |v| if compare_nonnegative(utility_b, v) == Ordering::Less { utility_b } else { v }));
            max_b = Some(max_b.map_or(utility_b, |v| if compare_nonnegative(utility_b, v) == Ordering::Greater { utility_b } else { v }));

            let (regret_a, regret_b) = match compare_nonnegative(utility_a, utility_b) {
                Ordering::Less => (
                    CheckedRational::new(
                        utility_b.numerator().checked_mul(utility_a.denominator())
                            .and_then(|v| v.checked_sub(utility_a.numerator().checked_mul(utility_b.denominator())?))
                            .unwrap(),
                        utility_b.denominator().checked_mul(utility_a.denominator()).unwrap(),
                    ).unwrap(),
                    CheckedRational::zero(),
                ),
                Ordering::Greater => (
                    CheckedRational::zero(),
                    CheckedRational::new(
                        utility_a.numerator().checked_mul(utility_b.denominator())
                            .and_then(|v| v.checked_sub(utility_b.numerator().checked_mul(utility_a.denominator())?))
                            .unwrap(),
                        utility_a.denominator().checked_mul(utility_b.denominator()).unwrap(),
                    ).unwrap(),
                ),
                Ordering::Equal => (CheckedRational::zero(), CheckedRational::zero()),
            };
            max_regret_a = Some(max_regret_a.map_or(regret_a, |v| if compare_nonnegative(regret_a, v) == Ordering::Greater { regret_a } else { v }));
            max_regret_b = Some(max_regret_b.map_or(regret_b, |v| if compare_nonnegative(regret_b, v) == Ordering::Greater { regret_b } else { v }));
        }

        assert_eq!(min_a.unwrap(), q(9, 25));
        assert_eq!(max_a.unwrap(), q(32, 25));
        assert_eq!(min_b.unwrap(), q(2, 5));
        assert_eq!(max_b.unwrap(), q(4, 5));
        assert_eq!(max_regret_a.unwrap(), q(11, 25));
        assert_eq!(max_regret_b.unwrap(), q(22, 25));
    }

    #[test]
    fn pairwise_regret_lp_uses_one_shared_transition_trajectory() {
        let model = fixture();
        let scores_a = vec![
            vec![q(0, 1), q(0, 1)],
            vec![q(0, 1), q(1, 1)],
            vec![q(0, 1), q(1, 1)],
        ];
        let scores_b = vec![
            vec![q(0, 1), q(0, 1)],
            vec![q(1, 1), q(0, 1)],
            vec![q(0, 1), q(0, 1)],
        ];

        let regret_a = build_pairwise_worst_case_regret_lp(
            &model,
            &scores_a,
            &scores_b,
            budget(504),
        ).unwrap();
        let certificate_a = PrimalDualCertificate {
            primal: sparse_vector(
                14,
                &[(0,1,1),(2,4,5),(3,1,5),(4,21,25),(5,4,25),(6,4,5),(7,1,5),(10,18,25),(11,2,25),(12,3,25),(13,2,25)],
            ),
            dual: sparse_vector(
                36,
                &[(0,11,25),(2,9,10),(4,9,10),(6,9,10),(8,9,10),(11,7,5),(14,23,10),(27,1,1),(30,1,1),(34,1,1)],
            ),
        };
        let verified_a = verify_primal_dual_optimality(regret_a.problem(), &certificate_a).unwrap();
        assert_eq!(verified_a.objective, q(11, 25));

        let regret_b = build_pairwise_worst_case_regret_lp(
            &model,
            &scores_b,
            &scores_a,
            budget(504),
        ).unwrap();
        let certificate_b = PrimalDualCertificate {
            primal: sparse_vector(
                14,
                &[(0,1,1),(2,2,5),(3,3,5),(4,8,25),(5,17,25),(6,2,5),(7,3,5),(10,1,5),(11,1,5),(12,3,25),(13,12,25)],
            ),
            dual: sparse_vector(
                36,
                &[(0,22,25),(2,9,5),(5,1,2),(6,9,5),(9,1,2),(10,9,5),(15,23,10),(26,1,1),(31,1,1),(35,1,1)],
            ),
        };
        let verified_b = verify_primal_dual_optimality(regret_b.problem(), &certificate_b).unwrap();
        assert_eq!(verified_b.objective, q(22, 25));
        assert!(verified_a.objective < verified_b.objective);
    }

    #[test]
    fn canonical_model_and_compiled_lp_digests_bind_inputs_and_objective() {
        let model = fixture();
        let model_digest = canonical_interval_model_sha256(&model);
        assert_eq!(model_digest, canonical_interval_model_sha256(&model));
        assert_eq!(
            sha256_hex(model_digest),
            "b1fb89c3775613d7b9aeb20857e6eb1a15b4133419e6fa3a307ddec6c906c466"
        );

        let scores_a = vec![
            vec![q(0, 1), q(0, 1)],
            vec![q(0, 1), q(1, 1)],
            vec![q(0, 1), q(1, 1)],
        ];
        let scores_b = vec![
            vec![q(0, 1), q(0, 1)],
            vec![q(1, 1), q(0, 1)],
            vec![q(0, 1), q(0, 1)],
        ];
        let a = build_occupation_flow_lp(
            &model,
            &scores_a,
            UtilityDirection::Maximize,
            budget(504),
        ).unwrap();
        let a_replay = build_occupation_flow_lp(
            &model,
            &scores_a,
            UtilityDirection::Maximize,
            budget(504),
        ).unwrap();
        let b = build_occupation_flow_lp(
            &model,
            &scores_b,
            UtilityDirection::Maximize,
            budget(504),
        ).unwrap();
        let a_min = build_occupation_flow_lp(
            &model,
            &scores_a,
            UtilityDirection::Minimize,
            budget(504),
        ).unwrap();

        assert_eq!(a.source_model_sha256(), model_digest);
        assert_eq!(a.source_model_sha256(), a_replay.source_model_sha256());
        assert_eq!(a.compiled_lp_sha256(), a_replay.compiled_lp_sha256());
        assert_eq!(
            sha256_hex(a.compiled_lp_sha256()),
            "882d77b1c59cede87ee5f04f69bf8345d85d26f4ad6443d2840cfbd839186ce4"
        );
        assert_ne!(a.compiled_lp_sha256(), b.compiled_lp_sha256());
        assert_ne!(a.compiled_lp_sha256(), a_min.compiled_lp_sha256());

        let altered_model = StagewiseIntervalMarkovModel::new(
            model.initial.clone(),
            {
                let mut transitions = model.transitions.clone();
                transitions[0][0][0].upper = q(3, 4);
                transitions
            },
        ).unwrap();
        assert_ne!(
            model_digest,
            canonical_interval_model_sha256(&altered_model)
        );
    }

    #[test]
    fn bound_certificates_reject_source_and_compiled_lp_digest_mutations() {
        let model = fixture();
        let scores = vec![
            vec![q(0, 1), q(0, 1)],
            vec![q(0, 1), q(1, 1)],
            vec![q(0, 1), q(1, 1)],
        ];
        let built = build_occupation_flow_lp(
            &model,
            &scores,
            UtilityDirection::Maximize,
            budget(504),
        ).unwrap();
        let certificate = built.bind_certificate(PrimalDualCertificate {
            primal: sparse_vector(
                14,
                &[(0,1,1),(2,2,5),(3,3,5),(4,8,25),(5,17,25),(6,2,5),(7,3,5),(10,1,5),(11,1,5),(12,3,25),(13,12,25)],
            ),
            dual: sparse_vector(
                36,
                &[(0,32,25),(2,9,5),(4,1,2),(6,9,5),(8,1,2),(10,9,5),(15,13,10),(26,1,1),(31,1,1),(35,1,1)],
            ),
        });
        let verified = built.verify_bound_optimality(&certificate).unwrap();
        assert_eq!(verified.utility, q(32, 25));
        assert_eq!(verified.source_model_sha256, built.source_model_sha256());
        assert_eq!(verified.compiled_lp_sha256, built.compiled_lp_sha256());

        let mut altered_source = certificate.clone();
        altered_source.source_model_sha256[0] ^= 1;
        assert_eq!(
            built.verify_bound_optimality(&altered_source),
            Err(OccupationFlowError::CertificateSourceDigestMismatch)
        );

        let mut altered_lp = certificate;
        altered_lp.compiled_lp_sha256[31] ^= 1;
        assert_eq!(
            built.verify_bound_optimality(&altered_lp),
            Err(OccupationFlowError::CertificateCompiledLpDigestMismatch)
        );
    }

    #[test]
    fn verifies_digest_bound_minimization_gap_in_original_utility_orientation() {
        let model = fixture();
        let scores = vec![
            vec![q(0, 1), q(0, 1)],
            vec![q(0, 1), q(1, 1)],
            vec![q(0, 1), q(1, 1)],
        ];
        let built = build_occupation_flow_lp(
            &model,
            &scores,
            UtilityDirection::Minimize,
            budget(504),
        ).unwrap();

        // The primal point is the max-utility endpoint (U=32/25), hence
        // -U=-32/25 is a valid lower bound for the maximization of -U.
        // The dual witness proves the maximum of -U is -9/25.
        let certificate = built.bind_certificate(PrimalDualCertificate {
            primal: sparse_vector(
                14,
                &[(0,1,1),(2,2,5),(3,3,5),(4,8,25),(5,17,25),(6,2,5),(7,3,5),(10,1,5),(11,1,5),(12,3,25),(13,12,25)],
            ),
            dual: sparse_vector(
                36,
                &[(1,9,25),(5,1,10),(9,1,10),(11,7,5),(14,13,10),(27,1,1),(30,1,1),(34,1,1)],
            ),
        });
        let verified = built.verify_bound_gap(&certificate, q(23, 25)).unwrap();
        assert_eq!(verified.lower_utility, q(9, 25));
        assert_eq!(verified.upper_utility, q(32, 25));
        assert_eq!(verified.absolute_gap, q(23, 25));
        assert_eq!(verified.source_model_sha256, built.source_model_sha256());
        assert_eq!(verified.compiled_lp_sha256, built.compiled_lp_sha256());

        assert!(matches!(
            built.verify_bound_gap(&certificate, q(22, 25)),
            Err(OccupationFlowError::Lp(LpCertificateError::GapExceedsRequested { .. }))
        ));
    }

    #[test]
    fn rejects_infeasible_interval_rows_and_invalid_dimensions() {
        let infeasible = StagewiseIntervalMarkovModel::new(
            vec![q(1, 1), q(0, 1)],
            vec![vec![
                vec![interval(3, 4, 9, 10), interval(3, 4, 9, 10)],
                vec![interval(0, 1, 1, 1), interval(0, 1, 1, 1)],
            ]],
        );
        assert!(matches!(infeasible, Err(OccupationFlowError::InfeasibleTransitionRow { stage: 0, row: 0 })));

        let model = fixture();
        let bad_scores = vec![vec![q(0, 1), q(0, 1)]];
        assert!(matches!(
            build_occupation_flow_lp(&model, &bad_scores, UtilityDirection::Maximize, budget(504)),
            Err(OccupationFlowError::ScorePeriodCountMismatch { .. })
        ));
    }

    #[test]
    fn rejects_occupation_flow_budget_before_dense_matrix_allocation() {
        let model = fixture();
        let scores = vec![
            vec![q(0, 1), q(0, 1)],
            vec![q(0, 1), q(1, 1)],
            vec![q(0, 1), q(1, 1)],
        ];
        assert_eq!(
            build_occupation_flow_lp(
                &model,
                &scores,
                UtilityDirection::Maximize,
                budget(503),
            ),
            Err(OccupationFlowError::BudgetExceeded {
                resource: "dense matrix entries",
                required: 504,
                limit: 503,
            })
        );
    }

    #[test]
    fn rejects_rational_bit_width_over_budget_before_matrix_allocation() {
        let model = fixture();
        let scores = vec![
            vec![q(0, 1), q(0, 1)],
            vec![q(0, 1), q(1, 1)],
            vec![q(0, 1), q(1, 1)],
        ];
        let tight = OccupationFlowBudget {
            max_variables: 14,
            max_constraints: 36,
            max_matrix_entries: 504,
            max_rational_bit_width: 4,
        };
        assert_eq!(
            build_occupation_flow_lp(&model, &scores, UtilityDirection::Maximize, tight),
            Err(OccupationFlowError::BudgetExceeded {
                resource: "rational numerator/denominator bit width",
                required: 5,
                limit: 4,
            })
        );
    }

    #[test]
    fn verifies_interval_row_with_narrow_and_exact_zero_bounds() {
        let valid = StagewiseIntervalMarkovModel::new(
            vec![q(1, 1), q(0, 1)],
            vec![vec![
                vec![interval(1, 1, 1, 1), interval(0, 1, 0, 1)],
                vec![interval(0, 1, 0, 1), interval(1, 1, 1, 1)],
            ]],
        );
        assert!(valid.is_ok());
        let built = build_occupation_flow_lp(
            valid.as_ref().unwrap(),
            &vec![vec![q(1, 1), q(0, 1)], vec![q(0, 1), q(1, 1)]],
            UtilityDirection::Maximize,
            OccupationFlowBudget {
                max_variables: 6,
                max_constraints: 20,
                max_matrix_entries: 120,
                max_rational_bit_width: 64,
            },
        ).unwrap();
        assert_eq!(built.problem().row_count(), 20);
        assert_eq!(built.problem().column_count(), 6);
    }
}
