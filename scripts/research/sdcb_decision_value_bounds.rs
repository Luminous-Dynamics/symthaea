//! Conservative expected-utility bounds for finite outcome likelihoods.
//!
//! Given per-cell probability enclosures and the exact normalization constraint
//! sum(p_i) = 1, this module computes lower/upper bounds for linear expected
//! utility. It uses the LP dual with candidate multipliers taken from the
//! finite utility values, and outward-rounded f64 arithmetic for each bound.
//!
//! Scope: interval boxes plus a single normalization equality. Additional
//! marginal/cross-moment constraints require a richer LP and are unsupported.
//! This is an evaluator, not a planner, evidence ledger, or action authorizer.

use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionBoundError {
    EmptyOutcomeSet,
    LengthMismatch,
    InvalidProbabilityInterval,
    InvalidUtility,
    InfeasibleProbabilityIntervals,
    ArithmeticFailure,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbabilityInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpectedUtilityBounds {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionSelection {
    pub uniquely_best_action: Option<usize>,
}

/// Exact nonnegative binary64 sum, in units of 2^-1074.
///
/// 18 limbs cover the 1075 bits needed for values <= 1 plus the carry
/// required when summing up to 64 outcome cells.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ExactProbabilitySum {
    limbs: [u64; 18],
}

impl ExactProbabilitySum {
    fn zero() -> Self {
        Self { limbs: [0; 18] }
    }

    fn add_word(&mut self, mut index: usize, mut value: u64) -> Result<(), DecisionBoundError> {
        while value != 0 {
            if index >= self.limbs.len() {
                return Err(DecisionBoundError::ArithmeticFailure);
            }
            let (sum, carry) = self.limbs[index].overflowing_add(value);
            self.limbs[index] = sum;
            value = u64::from(carry);
            index += 1;
        }
        Ok(())
    }

    fn add_probability(&mut self, value: f64) -> Result<(), DecisionBoundError> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(DecisionBoundError::InvalidProbabilityInterval);
        }
        if value == 0.0 {
            return Ok(());
        }

        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as usize;
        let fraction = bits & ((1_u64 << 52) - 1);
        let (mantissa, shift) = if exponent == 0 {
            (fraction, 0_usize)
        } else {
            ((1_u64 << 52) | fraction, exponent - 1)
        };

        let word = shift / 64;
        let offset = shift % 64;
        self.add_word(word, mantissa << offset)?;
        if offset != 0 {
            self.add_word(word + 1, mantissa >> (64 - offset))?;
        }
        Ok(())
    }

    fn cmp_exact(&self, other: &Self) -> Ordering {
        for index in (0..self.limbs.len()).rev() {
            match self.limbs[index].cmp(&other.limbs[index]) {
                Ordering::Equal => {}
                different => return different,
            }
        }
        Ordering::Equal
    }

    fn one() -> Self {
        let mut one = Self::zero();
        // The exact binary64 representation of 1.0 is represented by the
        // same conversion used for ordinary probability endpoints.
        one.add_probability(1.0).expect("one is a valid probability");
        one
    }
}

fn add_down(a: f64, b: f64) -> f64 {
    if a == 0.0 { return b; }
    if b == 0.0 { return a; }
    (a + b).next_down()
}

fn add_up(a: f64, b: f64) -> f64 {
    if a == 0.0 { return b; }
    if b == 0.0 { return a; }
    (a + b).next_up()
}

fn mul_down(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 { return 0.0; }
    (a * b).next_down()
}

fn mul_up(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 { return 0.0; }
    (a * b).next_up()
}

fn validate_probability_box(
    probabilities: &[ProbabilityInterval],
) -> Result<(), DecisionBoundError> {
    if probabilities.is_empty() {
        return Err(DecisionBoundError::EmptyOutcomeSet);
    }

    let mut lower_sum = ExactProbabilitySum::zero();
    let mut upper_sum = ExactProbabilitySum::zero();
    for interval in probabilities {
        if !interval.lower.is_finite()
            || !interval.upper.is_finite()
            || interval.lower < 0.0
            || interval.upper > 1.0
            || interval.lower > interval.upper
        {
            return Err(DecisionBoundError::InvalidProbabilityInterval);
        }
        lower_sum.add_probability(interval.lower)?;
        upper_sum.add_probability(interval.upper)?;
    }

    let one = ExactProbabilitySum::one();
    if lower_sum.cmp_exact(&one) == Ordering::Greater
        || upper_sum.cmp_exact(&one) == Ordering::Less
    {
        return Err(DecisionBoundError::InfeasibleProbabilityIntervals);
    }
    Ok(())
}

/// Compute conservative bounds on E[u] = sum_i p_i * u_i.
///
/// Each p_i lies in its declared interval and the probability vector is
/// constrained to sum exactly to one. The dual identity
///
///   E[u] = lambda + sum_i (u_i - lambda) * p_i
///
/// gives a valid lower/upper bound for every finite lambda. Evaluating all
/// unique utility values includes an optimal breakpoint for this box-plus-sum
/// linear programme. Arithmetic is expanded outward by one representable
/// binary64 value after each nontrivial operation.
///
/// Input endpoints and utilities are interpreted as exact binary64 values.
/// No additional marginal or dependence constraints are inferred.
pub fn expected_utility_bounds(
    probabilities: &[ProbabilityInterval],
    utilities: &[f64],
) -> Result<ExpectedUtilityBounds, DecisionBoundError> {
    if probabilities.len() != utilities.len() {
        return Err(DecisionBoundError::LengthMismatch);
    }
    validate_probability_box(probabilities)?;
    if utilities.iter().any(|value| !value.is_finite()) {
        return Err(DecisionBoundError::InvalidUtility);
    }

    let mut best_lower = f64::NEG_INFINITY;
    let mut best_upper = f64::INFINITY;

    // A dual optimum exists at one of the finite utility breakpoints.
    for &lambda in utilities {
        let mut candidate_lower = lambda;
        let mut candidate_upper = lambda;

        for (probability, &utility) in probabilities.iter().zip(utilities) {
            if utility == lambda {
                continue;
            }
            let difference = utility - lambda;
            if !difference.is_finite() {
                return Err(DecisionBoundError::ArithmeticFailure);
            }
            let difference_lower = difference.next_down();
            let difference_upper = difference.next_up();

            let (term_lower, term_upper) = if utility > lambda {
                (
                    mul_down(difference_lower, probability.lower),
                    mul_up(difference_upper, probability.upper),
                )
            } else {
                (
                    mul_down(difference_lower, probability.upper),
                    mul_up(difference_upper, probability.lower),
                )
            };

            candidate_lower = add_down(candidate_lower, term_lower);
            candidate_upper = add_up(candidate_upper, term_upper);
        }

        if !candidate_lower.is_finite() || !candidate_upper.is_finite() {
            return Err(DecisionBoundError::ArithmeticFailure);
        }
        best_lower = best_lower.max(candidate_lower);
        best_upper = best_upper.min(candidate_upper);
    }

    if !best_lower.is_finite() || !best_upper.is_finite() || best_lower > best_upper {
        return Err(DecisionBoundError::ArithmeticFailure);
    }

    Ok(ExpectedUtilityBounds {
        lower: best_lower,
        upper: best_upper,
    })
}

/// Return an action index only if its lower value bound strictly exceeds every
/// competing action's upper bound. Otherwise preserve an unresolved result.
pub fn select_uniquely_best_action(
    action_bounds: &[ExpectedUtilityBounds],
) -> ActionSelection {
    for (candidate_index, candidate) in action_bounds.iter().enumerate() {
        let dominates_all = action_bounds.iter().enumerate().all(|(other_index, other)| {
            candidate_index == other_index || candidate.lower > other.upper
        });
        if dominates_all {
            return ActionSelection {
                uniquely_best_action: Some(candidate_index),
            };
        }
    }
    ActionSelection {
        uniquely_best_action: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn exact_unit_sum_accumulator_accepts_binary_fraction_partition() {
        let mut sum = ExactProbabilitySum::zero();
        sum.add_probability(0.25).unwrap();
        sum.add_probability(0.5).unwrap();
        sum.add_probability(0.125).unwrap();
        sum.add_probability(0.125).unwrap();
        assert_eq!(sum.cmp_exact(&ExactProbabilitySum::one()), Ordering::Equal);
    }

    #[test]
    fn exact_unit_sum_accumulator_detects_decimal_float_overage() {
        let mut sum = ExactProbabilitySum::zero();
        sum.add_probability(0.2).unwrap();
        sum.add_probability(0.8).unwrap();
        assert_eq!(sum.cmp_exact(&ExactProbabilitySum::one()), Ordering::Greater);
    }

    #[test]
    fn utility_bounds_match_box_and_normalization_linear_program() {
        let probabilities = [
            ProbabilityInterval { lower: 0.2, upper: 0.4 },
            ProbabilityInterval { lower: 0.3, upper: 0.5 },
            ProbabilityInterval { lower: 0.1, upper: 0.5 },
        ];
        let utilities = [0.0, 10.0, 4.0];
        let result = expected_utility_bounds(&probabilities, &utilities).unwrap();
        assert!(result.lower <= 4.2);
        assert!(result.upper >= 6.2);
        close(result.lower, 4.2, 2.0e-14);
        close(result.upper, 6.2, 2.0e-14);
    }

    #[test]
    fn exact_probabilities_produce_a_tight_utility_enclosure() {
        let probabilities = [
            ProbabilityInterval { lower: 0.875, upper: 0.875 },
            ProbabilityInterval { lower: 0.125, upper: 0.125 },
        ];
        let result = expected_utility_bounds(&probabilities, [0.0, 10.0].as_slice()).unwrap();
        assert!(result.lower <= 1.25);
        assert!(result.upper >= 1.25);
        close(result.lower, 1.25, 2.0e-14);
        close(result.upper, 1.25, 2.0e-14);
    }

    #[test]
    fn negative_utilities_are_bounded_correctly() {
        let probabilities = [
            ProbabilityInterval { lower: 0.5, upper: 0.5 },
            ProbabilityInterval { lower: 0.5, upper: 0.5 },
        ];
        let result = expected_utility_bounds(&probabilities, &[-8.0, -2.0]).unwrap();
        assert!(result.lower <= -5.0);
        assert!(result.upper >= -5.0);
        close(result.lower, -5.0, 2.0e-14);
        close(result.upper, -5.0, 2.0e-14);
    }

    #[test]
    fn normalization_infeasibility_rejects() {
        let probabilities = [
            ProbabilityInterval { lower: 0.75, upper: 0.8 },
            ProbabilityInterval { lower: 0.75, upper: 0.9 },
        ];
        assert_eq!(
            expected_utility_bounds(&probabilities, &[1.0, 2.0]).unwrap_err(),
            DecisionBoundError::InfeasibleProbabilityIntervals
        );
    }

    #[test]
    fn action_is_selected_only_when_value_intervals_separate() {
        let probabilities = [
            ProbabilityInterval { lower: 0.875, upper: 0.875 },
            ProbabilityInterval { lower: 0.125, upper: 0.125 },
        ];
        let action_a = expected_utility_bounds(&probabilities, &[0.0, 10.0]).unwrap();
        let action_b = expected_utility_bounds(&probabilities, &[4.0, 4.0]).unwrap();
        let result = select_uniquely_best_action(&[action_a, action_b]);
        assert_eq!(result.uniquely_best_action, Some(1));
    }

    #[test]
    fn overlapping_action_values_remain_unresolved() {
        let result = select_uniquely_best_action(&[
            ExpectedUtilityBounds { lower: 1.0, upper: 4.0 },
            ExpectedUtilityBounds { lower: 3.0, upper: 5.0 },
        ]);
        assert_eq!(result.uniquely_best_action, None);
    }

    #[test]
    fn malformed_inputs_reject() {
        assert_eq!(
            expected_utility_bounds(&[], &[]).unwrap_err(),
            DecisionBoundError::EmptyOutcomeSet
        );
        assert_eq!(
            expected_utility_bounds(
                &[ProbabilityInterval { lower: -0.1, upper: 0.4 }],
                &[1.0],
            ).unwrap_err(),
            DecisionBoundError::InvalidProbabilityInterval
        );
        assert_eq!(
            expected_utility_bounds(
                &[ProbabilityInterval { lower: 0.0, upper: 1.0 }],
                &[f64::NAN],
            ).unwrap_err(),
            DecisionBoundError::InvalidUtility
        );
    }
}
