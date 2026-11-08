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
        if stage + 1 >= self.periods || source >= self.states || destination >= self.states {
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

    let occupancy_count = periods.checked_mul(states)
        .ok_or(OccupationFlowError::DimensionOverflow)?;
    let flow_count = model.transitions.len()
        .checked_mul(states)
        .and_then(|v| v.checked_mul(states))
        .ok_or(OccupationFlowError::DimensionOverflow)?;
    let variable_count = occupancy_count.checked_add(flow_count)
        .ok_or(OccupationFlowError::DimensionOverflow)?;
    let layout = OccupationFlowLayout {
        periods,
        states,
        flow_start: occupancy_count,
        variable_count,
    };

    let mut matrix: Vec<Vec<CheckedRational>> = Vec::new();
    let mut rhs: Vec<CheckedRational> = Vec::new();

    // Initial occupancy: omega[0,i] = initial[i].
    for state in 0..states {
        let mut row = zero_row(variable_count);
        row[layout.occupancy_variable(0, state).unwrap()] = CheckedRational::one();
        add_equality(&mut matrix, &mut rhs, row, model.initial[state])?;
    }

    for stage in 0..model.transitions.len() {
        // Outflow from each source state equals its current occupancy.
        for source in 0..states {
            let mut row = zero_row(variable_count);
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
            let mut row = zero_row(variable_count);
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

                let mut lower_row = zero_row(variable_count);
                lower_row[occupancy] = interval.lower;
                lower_row[flow] = CheckedRational::from_integer(-1);
                add_inequality(&mut matrix, &mut rhs, lower_row, CheckedRational::zero());

                let mut upper_row = zero_row(variable_count);
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

    let mut objective = zero_row(variable_count);
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

fn zero_row(length: usize) -> Vec<CheckedRational> {
    vec![CheckedRational::zero(); length]
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
    matrix.push(row.clone());
    rhs.push(value);
    let mut negated = Vec::with_capacity(row.len());
    for coefficient in row {
        let numerator = coefficient.numerator().checked_neg()
            .ok_or(OccupationFlowError::ArithmeticOverflow)?;
        negated.push(CheckedRational::new(numerator, coefficient.denominator())
            .map_err(|_| OccupationFlowError::ArithmeticOverflow)?);
    }
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
        verify_primal_dual_optimality, FarkasCertificate, PrimalDualCertificate,
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
        let built = build_occupation_flow_lp(&model, &scores, UtilityDirection::Maximize).unwrap();
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
                vec![(0,32,25),(2,9,5),(4,1,2),(6,9,5),(12,1,2),(14,9,5),(18,1,1),(23,13,10),(31,1,1),(35,1,1)],
            ),
            (
                &scores_a,
                UtilityDirection::Minimize,
                q(9, 25),
                vec![(0,1,1),(2,4,5),(3,1,5),(4,21,25),(5,4,25),(6,4,5),(7,1,5),(10,18,25),(11,2,25),(12,3,25),(13,2,25)],
                vec![(1,9,25),(5,1,10),(13,1,10),(15,7,5),(19,1,1),(22,13,10),(30,1,1),(34,1,1)],
            ),
            (
                &scores_b,
                UtilityDirection::Maximize,
                q(4, 5),
                vec![(0,1,1),(2,4,5),(3,1,5),(4,19,25),(5,6,25),(6,4,5),(7,1,5),(10,18,25),(11,2,25),(12,1,25),(13,4,25)],
                vec![(0,4,5),(2,1,1),(4,1,1),(6,1,1),(12,1,1),(22,1,1)],
            ),
            (
                &scores_b,
                UtilityDirection::Minimize,
                q(2, 5),
                vec![(0,1,1),(2,2,5),(3,3,5),(4,12,25),(5,13,25),(6,2,5),(7,3,5),(10,9,25),(11,1,25),(12,3,25),(13,12,25)],
                vec![(1,2,5),(5,1,1),(13,1,1),(23,1,1)],
            ),
        ];

        for (scores, direction, expected_utility, primal_entries, dual_entries) in certificates {
            let built = build_occupation_flow_lp(&model, scores, direction).unwrap();
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
            build_occupation_flow_lp(&model, &bad_scores, UtilityDirection::Maximize),
            Err(OccupationFlowError::ScorePeriodCountMismatch { .. })
        ));
    }

    #[test]
    fn verifies_interval_row_with_narrow_and_exact_zero_bounds() {
        let valid = StagewiseIntervalMarkovModel::new(
            vec![q(1, 1), q(0, 1)],
            vec![vec![
                vec![interval(1, 1), interval(0, 1)],
                vec![interval(0, 1), interval(1, 1)],
            ]],
        );
        assert!(valid.is_ok());
        let built = build_occupation_flow_lp(
            valid.as_ref().unwrap(),
            &vec![vec![q(1, 1), q(0, 1)], vec![q(0, 1), q(1, 1)]],
            UtilityDirection::Maximize,
        ).unwrap();
        assert_eq!(built.problem().row_count(), 18);
        assert_eq!(built.problem().column_count(), 6);
    }
}
