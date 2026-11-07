//! Exact rational probability semantics over finite factor graphs.
//!
//! This module uses the generic exact bucket-elimination engine, but adds a
//! probability-specific contract: local potentials must be nonnegative,
//! exact zeros represent impossible assignments, and conditioning with zero
//! partition mass is an error rather than a uniform fallback.
//!
//! All coefficients are normalized rationals backed by checked i128
//! arithmetic. This is an exact bounded reference, not arbitrary precision
//! and not a scalable replacement for approximate inference.

use crate::semiring::{
    CheckedRational, EliminationError, EliminationStats, ExactFactorGraph, RationalError,
};
use std::error::Error;
use std::fmt;

/// Failure while interpreting exact semiring weights as probability potentials.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RationalInferenceError {
    Elimination(EliminationError),
    Arithmetic(RationalError),
    NegativePotential {
        factor_index: usize,
        numerator: i128,
        denominator: i128,
    },
    ZeroPartitionFunction,
    MarginalNormalizationFailure {
        variable_index: usize,
    },
}
impl fmt::Display for RationalInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Elimination(error) => write!(f, "exact elimination failed: {error}"),
            Self::Arithmetic(error) => write!(f, "rational arithmetic failed: {error}"),
            Self::NegativePotential { factor_index, numerator, denominator } => write!(
                f,
                "factor {factor_index} contains negative probability potential {numerator}/{denominator}"
            ),
            Self::ZeroPartitionFunction => write!(
                f,
                "evidence has zero joint mass; posterior is undefined"
            ),
            Self::MarginalNormalizationFailure { variable_index } => write!(
                f,
                "exact posterior marginal for variable {variable_index} does not sum to one"
            ),
        }
    }
}
impl Error for RationalInferenceError {}
impl From<EliminationError> for RationalInferenceError {
    fn from(value: EliminationError) -> Self { Self::Elimination(value) }
}
impl From<RationalError> for RationalInferenceError {
    fn from(value: RationalError) -> Self { Self::Arithmetic(value) }
}

/// Exact rational posterior over a finite factor graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RationalPosterior {
    /// Unnormalized joint mass after applying all model and evidence factors.
    pub partition_function: CheckedRational,
    /// P(X_i = state | all factors/evidence) in variable-index order.
    pub marginals: Vec<Vec<CheckedRational>>,
    /// Work statistics for the partition-function elimination.
    pub partition_stats: EliminationStats,
    /// Number of additional conditioned eliminations used for marginal queries.
    pub conditioned_eliminations: usize,
}

/// Compute exact posterior marginals by repeated conditioned variable elimination.
///
/// For every variable i and state s, add an indicator factor
/// I[X_i=s], compute Z_i,s, then return Z_i,s / Z. This deliberately favors
/// a transparent and independently checkable reference over performance.
///
/// The caller must supply an elimination order containing every variable
/// exactly once. All existing factors and added evidence factors must have
/// nonnegative rational potentials. The evidence is represented by ordinary
/// indicator factors; for example, evidence B=1 on a binary variable is
/// represented by a unary factor [0,1].
///
/// The operation count is (1 + sum_i cardinality_i) complete VE executions,
/// so this is a small-instance oracle. Exact arithmetic is bounded by i128;
/// overflow is reported, never wrapped or approximated.
pub fn posterior_marginals(
    graph: &ExactFactorGraph<CheckedRational>,
    elimination_order: &[usize],
) -> Result<RationalPosterior, RationalInferenceError> {
    for (factor_index, factor) in graph.factors().iter().enumerate() {
        for value in &factor.values {
            if value.is_negative() {
                return Err(RationalInferenceError::NegativePotential {
                    factor_index,
                    numerator: value.numerator(),
                    denominator: value.denominator(),
                });
            }
        }
    }

    let (partition_function, partition_stats) = graph.evaluate(elimination_order)?;
    if partition_function.is_zero() {
        return Err(RationalInferenceError::ZeroPartitionFunction);
    }

    let mut marginals = Vec::with_capacity(graph.variable_count());
    let mut conditioned_eliminations = 0usize;
    for (variable_index, &cardinality) in graph.cardinalities().iter().enumerate() {
        let mut variable_marginal = Vec::with_capacity(cardinality);
        for state in 0..cardinality {
            let mut indicator = vec![CheckedRational::zero(); cardinality];
            indicator[state] = CheckedRational::one();

            let mut conditioned = graph.clone();
            conditioned.add_factor(vec![variable_index], indicator)?;
            let (state_mass, _) = conditioned.evaluate(elimination_order)?;
            let posterior_probability = state_mass.checked_div(partition_function)?;
            variable_marginal.push(posterior_probability);
            conditioned_eliminations = conditioned_eliminations
                .checked_add(1)
                .ok_or(RationalInferenceError::Arithmetic(RationalError::Overflow))?;
        }

        let marginal_sum = variable_marginal.iter().try_fold(
            CheckedRational::zero(),
            |sum, probability| sum.checked_add(*probability),
        )?;
        if marginal_sum != CheckedRational::one() {
            return Err(RationalInferenceError::MarginalNormalizationFailure {
                variable_index,
            });
        }
        marginals.push(variable_marginal);
    }

    Ok(RationalPosterior {
        partition_function,
        marginals,
        partition_stats,
        conditioned_eliminations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::{ExactFactorGraph, SemiringError};

    fn q(numerator: i128, denominator: i128) -> CheckedRational {
        CheckedRational::new(numerator, denominator).unwrap()
    }

    fn small_bn(reverse_factors: bool) -> ExactFactorGraph<CheckedRational> {
        let mut graph = ExactFactorGraph::new(
            vec![2, 2],
            CheckedRational::zero(),
            CheckedRational::one(),
        )
        .unwrap();

        let prior_a = vec![q(1, 3), q(2, 3)];
        // Scope [A,B], B is the fastest-changing table coordinate:
        // P(B|A=0)=[3/4,1/4], P(B|A=1)=[1/5,4/5].
        let likelihood_b_given_a = vec![q(3, 4), q(1, 4), q(1, 5), q(4, 5)];
        if reverse_factors {
            graph.add_factor(vec![0, 1], likelihood_b_given_a).unwrap();
            graph.add_factor(vec![0], prior_a).unwrap();
        } else {
            graph.add_factor(vec![0], prior_a).unwrap();
            graph.add_factor(vec![0, 1], likelihood_b_given_a).unwrap();
        }
        graph
    }

    #[test]
    fn exact_bn_marginals_match_analytic_rational_solution() {
        let graph = small_bn(false);
        let posterior = posterior_marginals(&graph, &[0, 1]).unwrap();

        assert_eq!(posterior.partition_function, CheckedRational::one());
        assert_eq!(posterior.marginals[0], vec![q(1, 3), q(2, 3)]);
        assert_eq!(posterior.marginals[1], vec![q(23, 60), q(37, 60)]);
        assert_eq!(posterior.conditioned_eliminations, 4);
    }

    #[test]
    fn evidence_conditioning_returns_exact_posterior_and_evidence_probability() {
        let mut graph = small_bn(false);
        // Observe B=1. Its likelihood is an indicator factor.
        graph.add_factor(vec![1], vec![CheckedRational::zero(), CheckedRational::one()]).unwrap();

        let posterior = posterior_marginals(&graph, &[1, 0]).unwrap();
        assert_eq!(posterior.partition_function, q(37, 60));
        assert_eq!(posterior.marginals[0], vec![q(5, 37), q(32, 37)]);
        assert_eq!(posterior.marginals[1], vec![q(0, 1), q(1, 1)]);
    }

    #[test]
    fn factor_insertion_and_elimination_order_do_not_change_posterior() {
        let left = posterior_marginals(&small_bn(false), &[0, 1]).unwrap();
        let right = posterior_marginals(&small_bn(true), &[1, 0]).unwrap();
        assert_eq!(left.partition_function, right.partition_function);
        assert_eq!(left.marginals, right.marginals);
    }

    #[test]
    fn impossible_evidence_fails_closed_instead_of_fabricating_posterior() {
        let mut graph = ExactFactorGraph::new(
            vec![2],
            CheckedRational::zero(),
            CheckedRational::one(),
        )
        .unwrap();
        graph.add_factor(vec![0], vec![CheckedRational::one(), CheckedRational::zero()]).unwrap();
        graph.add_factor(vec![0], vec![CheckedRational::zero(), CheckedRational::one()]).unwrap();

        assert_eq!(
            posterior_marginals(&graph, &[0]),
            Err(RationalInferenceError::ZeroPartitionFunction)
        );
    }

    #[test]
    fn negative_probability_potentials_are_rejected() {
        let mut graph = ExactFactorGraph::new(
            vec![2],
            CheckedRational::zero(),
            CheckedRational::one(),
        )
        .unwrap();
        graph.add_factor(vec![0], vec![q(1, 2), q(-1, 2)]).unwrap();

        assert!(matches!(
            posterior_marginals(&graph, &[0]),
            Err(RationalInferenceError::NegativePotential { factor_index: 0, .. })
        ));
    }

    #[test]
    fn checked_rational_is_canonical_and_cross_cancels() {
        assert_eq!(q(2, 4), q(1, 2));
        assert_eq!(q(-2, -4), q(1, 2));
        assert_eq!(q(0, -7), CheckedRational::zero());
        assert_eq!(q(i128::MIN, -2), q(1i128 << 126, 1));
        assert_eq!(q(6, 35).checked_mul(q(14, 9)).unwrap(), q(4, 15));
        assert_eq!(q(2, 3).checked_div(q(4, 9)).unwrap(), q(3, 2));
        assert_eq!(q(1, 6).checked_add(q(1, 3)).unwrap(), q(1, 2));
    }

    #[test]
    fn rational_overflow_and_zero_divisor_are_explicit_errors() {
        assert_eq!(q(i128::MAX, 1).checked_add(CheckedRational::one()), Err(RationalError::Overflow));
        assert_eq!(q(1, 2).checked_div(CheckedRational::zero()), Err(RationalError::ZeroDenominator));

        // Generic VE surfaces coefficient overflow as a domain error.
        let lhs = CheckedRational::from_integer(i128::MAX);
        let rhs = CheckedRational::one();
        let error = ExactSemiring::checked_add(&lhs, &rhs).unwrap_err();
        assert!(matches!(error, SemiringError::Domain(_)));
    }
}
