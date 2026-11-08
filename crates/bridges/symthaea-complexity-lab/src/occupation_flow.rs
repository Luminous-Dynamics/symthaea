//! Exact finite-horizon occupation-flow LPs for stagewise-free interval Markov models.
//!
//! This module encodes an explicitly row-separable, stagewise-independent
//! interval transition model as a rational LP. It does not support shared
//! parameters, cross-row coupling, cross-time coupling, or history-dependent
//! ambiguity. The source-to-LP construction is exposed for independent
//! primal/dual verification through `lp_certificate`.

use crate::lp_certificate::{LpCertificateError, RationalLp};
use crate::semiring::CheckedRational;
use std::error::Error;
use std::fmt;
use std::cmp::Ordering;

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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OccupationFlowLp {
    lp: RationalLp,
    layout: OccupationFlowLayout,
    direction: UtilityDirection,
}

impl OccupationFlowLp {
    pub fn problem(&self) -> &RationalLp { &self.lp }
    pub fn layout(&self) -> OccupationFlowLayout { self.layout }
    pub fn direction(&self) -> UtilityDirection { self.direction }

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

fn preflight_dimensions(
    model: &StagewiseIntervalMarkovModel,
    budget: OccupationFlowBudget,
) -> Result<(OccupationFlowLayout, usize), OccupationFlowError> {
    let periods = model.period_count();
    let states = model.state_count();
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
/// the two action utilities; subtracting independently optimized utility
/// intervals would generally be invalid.
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

    // Reject an over-budget LP before constructing the temporary score-difference grid.
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
    let periods = model.period_count();
    let states = model.state_count();
    if state_scores.len() != periods {
        return Err(OccupationFlowError::ScorePeriodCountMismatch {
            expected: periods,
            actual: state_scores.len(),
        });
    }
    for (time, scores) in state_scores.iter().enumerate() {
        if scores.len() != states {
            return Err(OccupationFlowError::ScoreStateCountMismatch {
                time,
                expected: states,
                actual: scores.len(),
            });
        }
    }

    let (layout, constraint_count) = preflight_dimensions(model, budget)?;

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
        let mut row = zero_row(variable_count)?;
        row[layout.occupancy_variable(0, state).unwrap()] = CheckedRational::one();
        add_equality(&mut matrix, &mut rhs, row, model.initial[state])?;
    }

    for stage in 0..model.transitions.len() {
        // Outflow from each source state equals its current occupancy.
        for source in 0..states {
            let mut row = zero_row(variable_count)?;
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
            let mut row = zero_row(variable_count)?;
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

                let mut lower_row = zero_row(variable_count)?;
                lower_row[occupancy] = interval.lower;
                lower_row[flow] = CheckedRational::from_integer(-1);
                add_inequality(&mut matrix, &mut rhs, lower_row, CheckedRational::zero());

                let mut upper_row = zero_row(variable_count)?;
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

    let mut objective = zero_row(variable_count)?;
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
    Ok(OccupationFlowLp { lp, layout, direction })
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
            let verified = verify_primal_dual_optimality(built.problem(), &certificate).unwrap();
            assert_eq!(
                built.utility_from_lp_objective(verified.objective).unwrap(),
                expected_utility
            );
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
            },
        ).unwrap();
        assert_eq!(built.problem().row_count(), 20);
        assert_eq!(built.problem().column_count(), 6);
    }
}
