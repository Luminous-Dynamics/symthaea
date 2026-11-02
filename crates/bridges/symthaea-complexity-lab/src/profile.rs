//! Exact multivariate weight profiles over bounded truncated polynomial rings.
//!
//! A profile is a sparse element of
//! Z[W_0, ..., W_(s-1)] / (W_0^N0, ..., W_(s-1)^N(s-1)).
//! Coefficients use checked i128 arithmetic. This is exact within that
//! bounded representation; it is not an arbitrary-precision production
//! backend and it does not implement the subexponential counting algorithm.
//!
//! The Boolean factor evaluator is an intentionally exponential, 24-variable
//! oracle for checking small instances against independent implementations.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

pub const MAX_REFERENCE_VARIABLES: usize = 24;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProfileError {
    ZeroTruncationBound { axis: usize },
    ExponentDimensionMismatch { expected: usize, actual: usize },
    IncompatibleTruncation { left: Vec<u32>, right: Vec<u32> },
    CoefficientOverflow { operation: &'static str },
    InvalidFactorScope { factor: usize, reason: &'static str },
    FactorTableLengthOverflow { factor: usize },
    FactorTableLengthMismatch { factor: usize, expected: usize, actual: usize },
    VariableOutOfRange { factor: usize, variable: usize, variable_count: usize },
    EnumerationLimitExceeded { variables: usize, maximum: usize },
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTruncationBound { axis } => write!(f, "truncation bound on axis {axis} must be positive"),
            Self::ExponentDimensionMismatch { expected, actual } => write!(f, "exponent has {actual} coordinates; expected {expected}"),
            Self::IncompatibleTruncation { left, right } => write!(f, "incompatible profile rings: {left:?} versus {right:?}"),
            Self::CoefficientOverflow { operation } => write!(f, "coefficient overflow during {operation}"),
            Self::InvalidFactorScope { factor, reason } => write!(f, "factor {factor} has invalid scope: {reason}"),
            Self::FactorTableLengthOverflow { factor } => write!(f, "factor {factor} truth-table length overflows usize"),
            Self::FactorTableLengthMismatch { factor, expected, actual } => write!(f, "factor {factor} has {actual} table entries; expected {expected}"),
            Self::VariableOutOfRange { factor, variable, variable_count } => write!(f, "factor {factor} references variable {variable}, but only {variable_count} variables exist"),
            Self::EnumerationLimitExceeded { variables, maximum } => write!(f, "brute-force oracle received {variables} variables; its limit is {maximum}"),
        }
    }
}
impl Error for ProfileError {}

/// Sparse exact polynomial in a multivariate truncated integer ring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightProfile {
    bounds: Vec<u32>,
    terms: BTreeMap<Vec<u32>, i128>,
}

impl WeightProfile {
    pub fn zero(bounds: Vec<u32>) -> Result<Self, ProfileError> {
        for (axis, bound) in bounds.iter().enumerate() {
            if *bound == 0 {
                return Err(ProfileError::ZeroTruncationBound { axis });
            }
        }
        Ok(Self { bounds, terms: BTreeMap::new() })
    }

    pub fn one(bounds: Vec<u32>) -> Result<Self, ProfileError> {
        let mut result = Self::zero(bounds)?;
        result.terms.insert(vec![0; result.bounds.len()], 1);
        Ok(result)
    }

    /// Create coefficient * monomial. Out-of-range monomials are zero.
    pub fn monomial(bounds: Vec<u32>, exponents: Vec<u32>, coefficient: i128) -> Result<Self, ProfileError> {
        let mut result = Self::zero(bounds)?;
        result.check_exponents(&exponents)?;
        if coefficient != 0 && exponents.iter().zip(&result.bounds).all(|(e, b)| e < b) {
            result.terms.insert(exponents, coefficient);
        }
        Ok(result)
    }

    pub fn bounds(&self) -> &[u32] { &self.bounds }
    pub fn support_size(&self) -> usize { self.terms.len() }
    pub fn is_zero(&self) -> bool { self.terms.is_empty() }

    pub fn coefficient(&self, exponents: &[u32]) -> Result<i128, ProfileError> {
        self.check_exponents(exponents)?;
        if exponents.iter().zip(&self.bounds).any(|(e, b)| e >= b) {
            return Ok(0);
        }
        Ok(*self.terms.get(exponents).unwrap_or(&0))
    }

    pub fn terms(&self) -> impl Iterator<Item = (&[u32], i128)> {
        self.terms.iter().map(|(e, c)| (e.as_slice(), *c))
    }

    pub fn checked_add(&self, other: &Self) -> Result<Self, ProfileError> {
        self.check_same_ring(other)?;
        let mut result = self.clone();
        for (e, c) in &other.terms {
            insert_coefficient(&mut result.terms, e.clone(), *c, "profile addition")?;
        }
        Ok(result)
    }

    pub fn checked_mul(&self, other: &Self) -> Result<Self, ProfileError> {
        self.check_same_ring(other)?;
        let mut terms = BTreeMap::new();
        for (left_e, left_c) in &self.terms {
            for (right_e, right_c) in &other.terms {
                let mut exponent = Vec::with_capacity(self.bounds.len());
                let mut truncated = false;
                for axis in 0..self.bounds.len() {
                    // Both stored exponents are below the bound. Compare
                    // against the remaining room before adding, so an
                    // exponent sum beyond u32::MAX is simply truncated (it
                    // cannot be represented in this quotient ring).
                    let remaining = self.bounds[axis] - left_e[axis];
                    if right_e[axis] >= remaining {
                        truncated = true;
                        break;
                    }
                    exponent.push(left_e[axis] + right_e[axis]);
                }
                if truncated { continue; }
                let coefficient = left_c.checked_mul(*right_c)
                    .ok_or(ProfileError::CoefficientOverflow { operation: "profile multiplication" })?;
                insert_coefficient(&mut terms, exponent, coefficient, "profile multiplication accumulation")?;
            }
        }
        Ok(Self { bounds: self.bounds.clone(), terms })
    }

    pub fn checked_scale(&self, scalar: i128) -> Result<Self, ProfileError> {
        let mut terms = BTreeMap::new();
        for (e, c) in &self.terms {
            let value = c.checked_mul(scalar)
                .ok_or(ProfileError::CoefficientOverflow { operation: "profile scalar multiplication" })?;
            insert_coefficient(&mut terms, e.clone(), value, "profile scalar accumulation")?;
        }
        Ok(Self { bounds: self.bounds.clone(), terms })
    }

    pub fn checked_shift(&self, shift: &[u32]) -> Result<Self, ProfileError> {
        self.check_exponents(shift)?;
        let monomial = Self::monomial(self.bounds.clone(), shift.to_vec(), 1)?;
        self.checked_mul(&monomial)
    }

    pub fn checked_total(&self) -> Result<i128, ProfileError> {
        let mut total = 0i128;
        for value in self.terms.values() {
            total = total.checked_add(*value)
                .ok_or(ProfileError::CoefficientOverflow { operation: "profile total" })?;
        }
        Ok(total)
    }

    pub fn maximum_exponent(&self, axis: usize) -> Option<u32> {
        if axis >= self.bounds.len() { return None; }
        self.terms.keys().map(|e| e[axis]).max()
    }

    pub fn checked_count_at_maximum(&self, axis: usize) -> Result<Option<i128>, ProfileError> {
        if axis >= self.bounds.len() {
            return Err(ProfileError::ExponentDimensionMismatch {
                expected: self.bounds.len(),
                actual: axis.saturating_add(1),
            });
        }
        let Some(maximum) = self.maximum_exponent(axis) else { return Ok(None); };
        let mut total = 0i128;
        for (e, c) in &self.terms {
            if e[axis] == maximum {
                total = total.checked_add(*c)
                    .ok_or(ProfileError::CoefficientOverflow { operation: "count at profile maximum" })?;
            }
        }
        Ok(Some(total))
    }

    /// Pareto-maximal exponent vectors. Intended for sparse, modest-size profiles.
    pub fn pareto_maxima(&self) -> Vec<Vec<u32>> {
        let points: Vec<Vec<u32>> = self.terms.keys().cloned().collect();
        points.iter().filter(|candidate| {
            !points.iter().any(|other| {
                other != *candidate
                    && other.iter().zip(candidate.iter()).all(|(a, b)| a >= b)
                    && other.iter().zip(candidate.iter()).any(|(a, b)| a > b)
            })
        }).cloned().collect()
    }

    fn check_exponents(&self, exponents: &[u32]) -> Result<(), ProfileError> {
        if exponents.len() != self.bounds.len() {
            return Err(ProfileError::ExponentDimensionMismatch { expected: self.bounds.len(), actual: exponents.len() });
        }
        Ok(())
    }

    fn check_same_ring(&self, other: &Self) -> Result<(), ProfileError> {
        if self.bounds != other.bounds {
            return Err(ProfileError::IncompatibleTruncation { left: self.bounds.clone(), right: other.bounds.clone() });
        }
        Ok(())
    }
}

fn insert_coefficient(
    terms: &mut BTreeMap<Vec<u32>, i128>,
    exponent: Vec<u32>,
    coefficient: i128,
    operation: &'static str,
) -> Result<(), ProfileError> {
    if coefficient == 0 { return Ok(()); }
    let updated = terms.get(&exponent).copied().unwrap_or(0)
        .checked_add(coefficient)
        .ok_or(ProfileError::CoefficientOverflow { operation })?;
    if updated == 0 { terms.remove(&exponent); } else { terms.insert(exponent, updated); }
    Ok(())
}

/// One local truth-table value: an integer coefficient times a weight monomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalMonomial {
    pub exponents: Vec<u32>,
    pub coefficient: i128,
}
impl LocalMonomial {
    pub fn new(exponents: Vec<u32>, coefficient: i128) -> Self {
        Self { exponents, coefficient }
    }
}

/// A local factor on a strictly increasing scope of Boolean variable indices.
/// Truth-table bit i corresponds to scope[i].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanMonomialFactor {
    pub scope: Vec<usize>,
    pub values: Vec<LocalMonomial>,
}
impl BooleanMonomialFactor {
    pub fn new(scope: Vec<usize>, values: Vec<LocalMonomial>) -> Self {
        Self { scope, values }
    }
}

/// Exhaustive reference evaluator. Its exponential cost and variable cap are
/// explicit: it is an oracle, not the subexponential profile-counting method.
pub fn enumerate_boolean_weight_profile(
    variable_count: usize,
    bounds: Vec<u32>,
    factors: &[BooleanMonomialFactor],
) -> Result<WeightProfile, ProfileError> {
    if variable_count > MAX_REFERENCE_VARIABLES {
        return Err(ProfileError::EnumerationLimitExceeded { variables: variable_count, maximum: MAX_REFERENCE_VARIABLES });
    }

    let mut profile = WeightProfile::zero(bounds)?;
    for (factor_index, factor) in factors.iter().enumerate() {
        if factor.scope.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ProfileError::InvalidFactorScope { factor: factor_index, reason: "scope must be strictly increasing" });
        }
        for &variable in &factor.scope {
            if variable >= variable_count {
                return Err(ProfileError::VariableOutOfRange { factor: factor_index, variable, variable_count });
            }
        }
        let expected = 1usize.checked_shl(factor.scope.len() as u32)
            .ok_or(ProfileError::FactorTableLengthOverflow { factor: factor_index })?;
        if factor.values.len() != expected {
            return Err(ProfileError::FactorTableLengthMismatch { factor: factor_index, expected, actual: factor.values.len() });
        }
        for value in &factor.values {
            if value.exponents.len() != profile.bounds.len() {
                return Err(ProfileError::ExponentDimensionMismatch { expected: profile.bounds.len(), actual: value.exponents.len() });
            }
        }
    }

    let assignment_count = 1usize.checked_shl(variable_count as u32)
        .ok_or(ProfileError::EnumerationLimitExceeded { variables: variable_count, maximum: MAX_REFERENCE_VARIABLES })?;

    for assignment in 0..assignment_count {
        let mut exponent = vec![0u32; profile.bounds.len()];
        let mut coefficient = 1i128;
        let mut vanishes = false;

        for factor in factors {
            let mut local_index = 0usize;
            for (local_axis, variable) in factor.scope.iter().enumerate() {
                if (assignment >> variable) & 1 == 1 {
                    local_index |= 1usize << local_axis;
                }
            }
            let value = &factor.values[local_index];

            if value.coefficient == 0 {
                vanishes = true;
                break;
            }
            for axis in 0..profile.bounds.len() {
                let remaining = profile.bounds[axis] - exponent[axis];
                if value.exponents[axis] >= remaining {
                    vanishes = true;
                    break;
                }
                exponent[axis] += value.exponents[axis];
            }
            if vanishes { break; }

            coefficient = coefficient.checked_mul(value.coefficient)
                .ok_or(ProfileError::CoefficientOverflow { operation: "factor-product evaluation" })?;
            if coefficient == 0 { vanishes = true; break; }
        }

        if !vanishes {
            insert_coefficient(&mut profile.terms, exponent, coefficient, "assignment-profile accumulation")?;
        }
    }

    Ok(profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn local(exponents: &[u32], coefficient: i128) -> LocalMonomial {
        LocalMonomial::new(exponents.to_vec(), coefficient)
    }

    #[test]
    fn triangle_max_cut_profile_is_exact() {
        let edge = |u: usize, v: usize| BooleanMonomialFactor::new(
            vec![u, v],
            vec![local(&[0], 1), local(&[1], 1), local(&[1], 1), local(&[0], 1)],
        );
        let profile = enumerate_boolean_weight_profile(3, vec![4], &[edge(0,1), edge(1,2), edge(0,2)]).unwrap();
        assert_eq!(profile.coefficient(&[0]).unwrap(), 2);
        assert_eq!(profile.coefficient(&[1]).unwrap(), 0);
        assert_eq!(profile.coefficient(&[2]).unwrap(), 6);
        assert_eq!(profile.coefficient(&[3]).unwrap(), 0);
        assert_eq!(profile.checked_total().unwrap(), 8);
        assert_eq!(profile.maximum_exponent(0), Some(2));
        assert_eq!(profile.checked_count_at_maximum(0).unwrap(), Some(6));
    }

    #[test]
    fn multiobjective_profile_preserves_pareto_tradeoffs() {
        let at_most_one = BooleanMonomialFactor::new(
            vec![0, 1],
            vec![local(&[0,0], 1), local(&[0,0], 1), local(&[0,0], 1), local(&[0,0], 0)],
        );
        let choice_0 = BooleanMonomialFactor::new(vec![0], vec![local(&[0,0], 1), local(&[1,0], 1)]);
        let choice_1 = BooleanMonomialFactor::new(vec![1], vec![local(&[0,0], 1), local(&[0,1], 1)]);
        let profile = enumerate_boolean_weight_profile(2, vec![3,3], &[at_most_one, choice_0, choice_1]).unwrap();
        assert_eq!(profile.coefficient(&[0,0]).unwrap(), 1);
        assert_eq!(profile.coefficient(&[1,0]).unwrap(), 1);
        assert_eq!(profile.coefficient(&[0,1]).unwrap(), 1);
        assert_eq!(profile.coefficient(&[1,1]).unwrap(), 0);
        assert_eq!(profile.pareto_maxima(), vec![vec![0,1], vec![1,0]]);
    }

    #[test]
    fn truncation_discards_out_of_range_monomials() {
        let a = WeightProfile::monomial(vec![3], vec![2], 4).unwrap();
        let b = WeightProfile::monomial(vec![3], vec![1], 5).unwrap();
        assert!(a.checked_mul(&b).unwrap().is_zero());
    }

    #[test]
    fn signed_coefficients_cancel_exactly() {
        let a = WeightProfile::monomial(vec![4], vec![2], 9).unwrap();
        let b = WeightProfile::monomial(vec![4], vec![2], -9).unwrap();
        assert!(a.checked_add(&b).unwrap().is_zero());
    }

    #[test]
    fn coefficient_overflow_is_rejected() {
        let a = WeightProfile::monomial(vec![1], vec![0], i128::MAX).unwrap();
        let b = WeightProfile::monomial(vec![1], vec![0], 2).unwrap();
        assert!(matches!(a.checked_add(&b), Err(ProfileError::CoefficientOverflow { .. })));
    }

    #[test]
    fn incompatible_truncation_rings_are_rejected() {
        let a = WeightProfile::one(vec![2]).unwrap();
        let b = WeightProfile::one(vec![3]).unwrap();
        assert!(matches!(a.checked_add(&b), Err(ProfileError::IncompatibleTruncation { .. })));
    }

    #[test]
    fn empty_scope_factor_is_a_global_scalar() {
        let factor = BooleanMonomialFactor::new(vec![], vec![local(&[0], 3)]);
        let profile = enumerate_boolean_weight_profile(2, vec![1], &[factor]).unwrap();
        assert_eq!(profile.coefficient(&[0]).unwrap(), 12);
    }

    #[test]
    fn malformed_scopes_fail_closed() {
        let duplicate = BooleanMonomialFactor::new(vec![0,0], vec![local(&[0], 1); 4]);
        assert!(matches!(enumerate_boolean_weight_profile(1, vec![2], &[duplicate]), Err(ProfileError::InvalidFactorScope { .. })));
        let out_of_range = BooleanMonomialFactor::new(vec![2], vec![local(&[0], 1); 2]);
        assert!(matches!(enumerate_boolean_weight_profile(2, vec![2], &[out_of_range]), Err(ProfileError::VariableOutOfRange { .. })));
    }

    #[test]
    fn empty_factor_system_counts_all_assignments() {
        let profile = enumerate_boolean_weight_profile(5, vec![2], &[]).unwrap();
        assert_eq!(profile.coefficient(&[0]).unwrap(), 32);
        assert_eq!(profile.checked_total().unwrap(), 32);
    }
}
