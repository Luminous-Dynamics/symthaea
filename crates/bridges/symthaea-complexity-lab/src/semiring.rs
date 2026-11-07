//! Exact semiring algebras and variable elimination for finite factor systems.
//!
//! Separate from the f64 belief-propagation crate. Exact variable elimination
//! is generic over a semiring; its cost is exponential in induced width.

use crate::profile::{ProfileError, WeightProfile};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemiringError {
    CountOverflow { operation: &'static str },
    ScoreOverflow,
    ObjectiveVectorOverflow { axis: usize },
    DimensionMismatch { expected: usize, actual: usize },
    Domain(String),
}
impl fmt::Display for SemiringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CountOverflow { operation } => write!(f, "count overflow during {operation}"),
            Self::ScoreOverflow => write!(f, "objective score overflow"),
            Self::ObjectiveVectorOverflow { axis } => write!(f, "objective vector overflow on axis {axis}"),
            Self::DimensionMismatch { expected, actual } => write!(f, "dimension {actual}; expected {expected}"),
            Self::Domain(reason) => write!(f, "coefficient-domain error: {reason}"),
        }
    }
}
impl Error for SemiringError {}

/// Values implement a commutative semiring: associative/commutative addition,
/// associative/commutative multiplication, distributivity, and identities.
/// The evaluator receives explicit domain-specific zero and one values. The
/// current factor engine does not support noncommutative coefficient products.
pub trait ExactSemiring: Clone {
    fn checked_add(&self, other: &Self) -> Result<Self, SemiringError>;
    fn checked_mul(&self, other: &Self) -> Result<Self, SemiringError>;
}


/// Failure in bounded exact rational arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RationalError {
    ZeroDenominator,
    Overflow,
}
impl fmt::Display for RationalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDenominator => write!(f, "rational denominator is zero"),
            Self::Overflow => write!(f, "checked rational arithmetic overflow"),
        }
    }
}
impl Error for RationalError {}

/// Canonical rational value backed by checked i128 numerator/denominator.
///
/// The denominator is always positive and the pair is reduced by gcd.
/// This is exact only while every intermediate fits in i128; operations
/// return an error rather than silently wrapping or approximating.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CheckedRational {
    numerator: i128,
    denominator: i128,
}
impl CheckedRational {
    pub fn new(numerator: i128, denominator: i128) -> Result<Self, RationalError> {
        if denominator == 0 {
            return Err(RationalError::ZeroDenominator);
        }
        let divisor = gcd_u128(numerator.unsigned_abs(), denominator.unsigned_abs());
        let mut numerator = divide_signed_by_unsigned(numerator, divisor)?;
        let mut denominator = divide_signed_by_unsigned(denominator, divisor)?;
        if denominator < 0 {
            numerator = numerator.checked_neg().ok_or(RationalError::Overflow)?;
            denominator = denominator.checked_neg().ok_or(RationalError::Overflow)?;
        }
        if denominator <= 0 {
            return Err(RationalError::Overflow);
        }
        Ok(Self { numerator, denominator })
    }

    pub fn from_integer(value: i128) -> Self {
        Self { numerator: value, denominator: 1 }
    }

    pub const fn zero() -> Self {
        Self { numerator: 0, denominator: 1 }
    }

    pub const fn one() -> Self {
        Self { numerator: 1, denominator: 1 }
    }

    pub const fn numerator(self) -> i128 { self.numerator }
    pub const fn denominator(self) -> i128 { self.denominator }
    pub const fn is_zero(self) -> bool { self.numerator == 0 }
    pub const fn is_negative(self) -> bool { self.numerator < 0 }

    pub fn checked_add(self, other: Self) -> Result<Self, RationalError> {
        let common = gcd_u128(self.denominator as u128, other.denominator as u128);
        let self_scale = divide_signed_by_unsigned(
            other.denominator,
            common,
        )?;
        let other_scale = divide_signed_by_unsigned(
            self.denominator,
            common,
        )?;
        let left = self.numerator.checked_mul(self_scale).ok_or(RationalError::Overflow)?;
        let right = other.numerator.checked_mul(other_scale).ok_or(RationalError::Overflow)?;
        let numerator = left.checked_add(right).ok_or(RationalError::Overflow)?;
        let denominator = self.denominator.checked_mul(self_scale).ok_or(RationalError::Overflow)?;
        Self::new(numerator, denominator)
    }

    pub fn checked_mul(self, other: Self) -> Result<Self, RationalError> {
        // Cross-cancel before multiplication to preserve representable results.
        let cancel_left = gcd_u128(self.numerator.unsigned_abs(), other.denominator as u128);
        let cancel_right = gcd_u128(other.numerator.unsigned_abs(), self.denominator as u128);
        let left_num = divide_signed_by_unsigned(self.numerator, cancel_left)?;
        let right_den = divide_signed_by_unsigned(other.denominator, cancel_left)?;
        let right_num = divide_signed_by_unsigned(other.numerator, cancel_right)?;
        let left_den = divide_signed_by_unsigned(self.denominator, cancel_right)?;
        let numerator = left_num.checked_mul(right_num).ok_or(RationalError::Overflow)?;
        let denominator = left_den.checked_mul(right_den).ok_or(RationalError::Overflow)?;
        Self::new(numerator, denominator)
    }

    pub fn checked_div(self, other: Self) -> Result<Self, RationalError> {
        if other.numerator == 0 {
            return Err(RationalError::ZeroDenominator);
        }
        // (a/b)/(c/d) = (a*d)/(b*c), cross-cancel first.
        let cancel_left = gcd_u128(self.numerator.unsigned_abs(), other.numerator.unsigned_abs());
        let cancel_right = gcd_u128(other.denominator as u128, self.denominator as u128);
        let left_num = divide_signed_by_unsigned(self.numerator, cancel_left)?;
        let other_num = divide_signed_by_unsigned(other.numerator, cancel_left)?;
        let other_den = divide_signed_by_unsigned(other.denominator, cancel_right)?;
        let left_den = divide_signed_by_unsigned(self.denominator, cancel_right)?;
        let numerator = left_num.checked_mul(other_den).ok_or(RationalError::Overflow)?;
        let denominator = left_den.checked_mul(other_num).ok_or(RationalError::Overflow)?;
        Self::new(numerator, denominator)
    }
}
impl fmt::Display for CheckedRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}
impl ExactSemiring for CheckedRational {
    fn checked_add(&self, other: &Self) -> Result<Self, SemiringError> {
        (*self).checked_add(*other).map_err(|error| SemiringError::Domain(error.to_string()))
    }
    fn checked_mul(&self, other: &Self) -> Result<Self, SemiringError> {
        (*self).checked_mul(*other).map_err(|error| SemiringError::Domain(error.to_string()))
    }
}

fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

fn divide_signed_by_unsigned(value: i128, divisor: u128) -> Result<i128, RationalError> {
    if divisor == 0 {
        return Err(RationalError::ZeroDenominator);
    }
    if divisor == (1u128 << 127) {
        return if value == i128::MIN { Ok(-1) } else { Err(RationalError::Overflow) };
    }
    let divisor = i128::try_from(divisor).map_err(|_| RationalError::Overflow)?;
    value.checked_div(divisor).ok_or(RationalError::Overflow)
}

impl ExactSemiring for bool {
    fn checked_add(&self, other: &Self) -> Result<Self, SemiringError> { Ok(*self || *other) }
    fn checked_mul(&self, other: &Self) -> Result<Self, SemiringError> { Ok(*self && *other) }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct CheckedCount(pub u128);
impl ExactSemiring for CheckedCount {
    fn checked_add(&self, other: &Self) -> Result<Self, SemiringError> {
        self.0.checked_add(other.0).map(Self).ok_or(SemiringError::CountOverflow { operation: "addition" })
    }
    fn checked_mul(&self, other: &Self) -> Result<Self, SemiringError> {
        self.0.checked_mul(other.0).map(Self).ok_or(SemiringError::CountOverflow { operation: "multiplication" })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxCountTerm {
    pub score: i128,
    pub count: u128,
}

/// MaxCount algebra: max score, summing counts on ties; multiplication adds
/// independent scores and multiplies their multiplicities.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaxCountValue(pub Option<MaxCountTerm>);
impl MaxCountValue {
    pub fn zero() -> Self { Self(None) }
    pub fn one() -> Self { Self(Some(MaxCountTerm { score: 0, count: 1 })) }
    pub fn candidate(score: i128, count: u128) -> Self {
        if count == 0 { Self::zero() } else { Self(Some(MaxCountTerm { score, count })) }
    }
    pub fn optimum(self) -> Option<MaxCountTerm> { self.0 }
}
impl ExactSemiring for MaxCountValue {
    fn checked_add(&self, other: &Self) -> Result<Self, SemiringError> {
        match (self.0, other.0) {
            (None, x) | (x, None) => Ok(Self(x)),
            (Some(a), Some(b)) if a.score > b.score => Ok(Self(Some(a))),
            (Some(a), Some(b)) if b.score > a.score => Ok(Self(Some(b))),
            (Some(a), Some(b)) => {
                let count = a.count.checked_add(b.count).ok_or(SemiringError::CountOverflow { operation: "MaxCount tie aggregation" })?;
                Ok(Self(Some(MaxCountTerm { score: a.score, count })))
            }
        }
    }
    fn checked_mul(&self, other: &Self) -> Result<Self, SemiringError> {
        let (Some(a), Some(b)) = (self.0, other.0) else { return Ok(Self::zero()); };
        let score = a.score.checked_add(b.score).ok_or(SemiringError::ScoreOverflow)?;
        let count = a.count.checked_mul(b.count).ok_or(SemiringError::CountOverflow { operation: "MaxCount multiplicity multiplication" })?;
        Ok(Self::candidate(score, count))
    }
}

/// Exact Pareto antichain with multiplicity at each nondominated vector.
/// Multiplication is Minkowski addition; addition merges equal vectors and
/// removes dominated vectors. Frontier size can be exponential.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParetoCountValue {
    dimensions: usize,
    points: BTreeMap<Vec<i128>, u128>,
}
impl ParetoCountValue {
    pub fn zero(dimensions: usize) -> Self { Self { dimensions, points: BTreeMap::new() } }
    pub fn one(dimensions: usize) -> Self { Self::point(vec![0; dimensions], 1) }
    pub fn point(vector: Vec<i128>, count: u128) -> Self {
        let dimensions = vector.len();
        let mut points = BTreeMap::new();
        if count > 0 { points.insert(vector, count); }
        Self { dimensions, points }
    }
    pub fn dimensions(&self) -> usize { self.dimensions }
    pub fn points(&self) -> impl Iterator<Item = (&[i128], u128)> {
        self.points.iter().map(|(v, c)| (v.as_slice(), *c))
    }
    pub fn is_zero(&self) -> bool { self.points.is_empty() }
    fn check_dimensions(&self, other: &Self) -> Result<(), SemiringError> {
        if self.dimensions != other.dimensions {
            return Err(SemiringError::DimensionMismatch { expected: self.dimensions, actual: other.dimensions });
        }
        Ok(())
    }
    fn prune(points: &mut BTreeMap<Vec<i128>, u128>) {
        let candidates: Vec<Vec<i128>> = points.keys().cloned().collect();
        let dominated: Vec<Vec<i128>> = candidates.iter().filter(|candidate| {
            candidates.iter().any(|other| other != *candidate
                && other.iter().zip(candidate.iter()).all(|(a,b)| a >= b)
                && other.iter().zip(candidate.iter()).any(|(a,b)| a > b))
        }).cloned().collect();
        for vector in dominated { points.remove(&vector); }
    }
}
impl ExactSemiring for ParetoCountValue {
    fn checked_add(&self, other: &Self) -> Result<Self, SemiringError> {
        self.check_dimensions(other)?;
        let mut points = self.points.clone();
        for (vector, count) in &other.points {
            let value = points.get(vector).copied().unwrap_or(0)
                .checked_add(*count)
                .ok_or(SemiringError::CountOverflow { operation: "Pareto tie aggregation" })?;
            if value == 0 { points.remove(vector); } else { points.insert(vector.clone(), value); }
        }
        Self::prune(&mut points);
        Ok(Self { dimensions: self.dimensions, points })
    }
    fn checked_mul(&self, other: &Self) -> Result<Self, SemiringError> {
        self.check_dimensions(other)?;
        let mut points = BTreeMap::new();
        for (left, lc) in &self.points {
            for (right, rc) in &other.points {
                let mut vector = Vec::with_capacity(self.dimensions);
                for axis in 0..self.dimensions {
                    vector.push(left[axis].checked_add(right[axis]).ok_or(SemiringError::ObjectiveVectorOverflow { axis })?);
                }
                let count = lc.checked_mul(*rc).ok_or(SemiringError::CountOverflow { operation: "Pareto product" })?;
                let value = points.get(&vector).copied().unwrap_or(0u128)
                    .checked_add(count)
                    .ok_or(SemiringError::CountOverflow { operation: "Pareto collision aggregation" })?;
                points.insert(vector, value);
            }
        }
        Self::prune(&mut points);
        Ok(Self { dimensions: self.dimensions, points })
    }
}

impl ExactSemiring for WeightProfile {
    fn checked_add(&self, other: &Self) -> Result<Self, SemiringError> {
        WeightProfile::checked_add(self, other).map_err(|e: ProfileError| SemiringError::Domain(e.to_string()))
    }
    fn checked_mul(&self, other: &Self) -> Result<Self, SemiringError> {
        WeightProfile::checked_mul(self, other).map_err(|e: ProfileError| SemiringError::Domain(e.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EliminationError {
    ZeroCardinality { variable: usize },
    InvalidScope { factor: usize, reason: &'static str },
    VariableOutOfRange { factor: usize, variable: usize },
    TableSizeOverflow { factor: Option<usize> },
    TableLengthMismatch { factor: usize, expected: usize, actual: usize },
    InvalidEliminationOrder { reason: &'static str },
    Semiring(SemiringError),
}
impl fmt::Display for EliminationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCardinality { variable } => write!(f, "variable {variable} has zero cardinality"),
            Self::InvalidScope { factor, reason } => write!(f, "factor {factor} invalid scope: {reason}"),
            Self::VariableOutOfRange { factor, variable } => write!(f, "factor {factor} variable {variable} out of range"),
            Self::TableSizeOverflow { factor } => write!(f, "factor table size overflow: {factor:?}"),
            Self::TableLengthMismatch { factor, expected, actual } => write!(f, "factor {factor} has {actual} entries; expected {expected}"),
            Self::InvalidEliminationOrder { reason } => write!(f, "invalid elimination order: {reason}"),
            Self::Semiring(error) => write!(f, "semiring evaluation error: {error}"),
        }
    }
}
impl Error for EliminationError {}
impl From<SemiringError> for EliminationError { fn from(value: SemiringError) -> Self { Self::Semiring(value) } }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFactor<S> {
    pub scope: Vec<usize>,
    pub values: Vec<S>,
}
impl<S: Clone> ExactFactor<S> {
    pub fn new(index: usize, scope: Vec<usize>, values: Vec<S>, cardinalities: &[usize]) -> Result<Self, EliminationError> {
        if scope.windows(2).any(|p| p[0] >= p[1]) {
            return Err(EliminationError::InvalidScope { factor: index, reason: "scope must be strictly increasing" });
        }
        let mut expected = 1usize;
        for &variable in &scope {
            if variable >= cardinalities.len() {
                return Err(EliminationError::VariableOutOfRange { factor: index, variable });
            }
            expected = expected.checked_mul(cardinalities[variable])
                .ok_or(EliminationError::TableSizeOverflow { factor: Some(index) })?;
        }
        if values.len() != expected {
            return Err(EliminationError::TableLengthMismatch { factor: index, expected, actual: values.len() });
        }
        Ok(Self { scope, values })
    }
    fn table_index(&self, assignment: &[usize], cardinalities: &[usize]) -> usize {
        self.scope.iter().fold(0usize, |index, &v| index * cardinalities[v] + assignment[v])
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EliminationStats {
    pub eliminated_variables: usize,
    pub maximum_bucket_variables: usize,
    pub maximum_bucket_assignments: usize,
    pub maximum_output_factor_entries: usize,
    pub semiring_additions: u128,
    pub semiring_multiplications: u128,
    pub generated_factors: usize,
}

/// Exact finite-domain factor graph evaluated by bucket variable elimination.
#[derive(Clone, Debug)]
pub struct ExactFactorGraph<S> {
    cardinalities: Vec<usize>,
    factors: Vec<ExactFactor<S>>,
    zero: S,
    one: S,
}
impl<S: ExactSemiring> ExactFactorGraph<S> {
    pub fn new(cardinalities: Vec<usize>, zero: S, one: S) -> Result<Self, EliminationError> {
        for (variable, cardinality) in cardinalities.iter().enumerate() {
            if *cardinality == 0 { return Err(EliminationError::ZeroCardinality { variable }); }
        }
        Ok(Self { cardinalities, factors: Vec::new(), zero, one })
    }
    pub fn variable_count(&self) -> usize { self.cardinalities.len() }
    pub fn cardinalities(&self) -> &[usize] { &self.cardinalities }
    pub fn add_factor(&mut self, scope: Vec<usize>, values: Vec<S>) -> Result<usize, EliminationError> {
        let index = self.factors.len();
        self.factors.push(ExactFactor::new(index, scope, values, &self.cardinalities)?);
        Ok(index)
    }
    pub fn factors(&self) -> &[ExactFactor<S>] { &self.factors }

    /// Must list every variable once. Complexity is exponential in the
    /// maximum induced bucket size.
    pub fn evaluate(&self, order: &[usize]) -> Result<(S, EliminationStats), EliminationError> {
        if order.len() != self.cardinalities.len() {
            return Err(EliminationError::InvalidEliminationOrder { reason: "must list every variable exactly once" });
        }
        let mut seen = vec![false; self.cardinalities.len()];
        for &v in order {
            if v >= seen.len() || seen[v] {
                return Err(EliminationError::InvalidEliminationOrder { reason: "out-of-range or duplicate variable" });
            }
            seen[v] = true;
        }

        let mut factors = self.factors.clone();
        let mut stats = EliminationStats::default();

        for &eliminated in order {
            let bucket_indices: Vec<usize> = factors.iter().enumerate()
                .filter_map(|(i, f)| f.scope.contains(&eliminated).then_some(i)).collect();
            let mut joined = BTreeSet::new();
            for &i in &bucket_indices { joined.extend(factors[i].scope.iter().copied()); }
            joined.remove(&eliminated);
            let output_scope: Vec<usize> = joined.into_iter().collect();

            let mut output_entries = 1usize;
            for &v in &output_scope {
                output_entries = output_entries.checked_mul(self.cardinalities[v])
                    .ok_or(EliminationError::TableSizeOverflow { factor: None })?;
            }
            let bucket_variables = output_scope.len() + 1;
            let bucket_assignments = output_entries.checked_mul(self.cardinalities[eliminated])
                .ok_or(EliminationError::TableSizeOverflow { factor: None })?;
            stats.maximum_bucket_variables = stats.maximum_bucket_variables.max(bucket_variables);
            stats.maximum_bucket_assignments = stats.maximum_bucket_assignments.max(bucket_assignments);
            stats.maximum_output_factor_entries = stats.maximum_output_factor_entries.max(output_entries);

            let mut output = Vec::with_capacity(output_entries);
            for output_index in 0..output_entries {
                let mut assignment = vec![0usize; self.cardinalities.len()];
                decode_assignment(output_index, &output_scope, &self.cardinalities, &mut assignment);
                let mut sum = self.zero.clone();
                for state in 0..self.cardinalities[eliminated] {
                    assignment[eliminated] = state;
                    let mut product = self.one.clone();
                    for &factor_index in &bucket_indices {
                        let factor = &factors[factor_index];
                        let value = &factor.values[factor.table_index(&assignment, &self.cardinalities)];
                        product = product.checked_mul(value)?;
                        stats.semiring_multiplications += 1;
                    }
                    sum = sum.checked_add(&product)?;
                    stats.semiring_additions += 1;
                }
                output.push(sum);
            }

            let mut retained = Vec::with_capacity(factors.len() + 1);
            for (i, factor) in factors.into_iter().enumerate() {
                if !bucket_indices.contains(&i) { retained.push(factor); }
            }
            retained.push(ExactFactor { scope: output_scope, values: output });
            factors = retained;
            stats.eliminated_variables += 1;
            stats.generated_factors += 1;
        }

        let mut result = self.one.clone();
        for factor in factors {
            if !factor.scope.is_empty() {
                return Err(EliminationError::InvalidEliminationOrder { reason: "factor retained an uneliminated variable" });
            }
            result = result.checked_mul(&factor.values[0])?;
            stats.semiring_multiplications += 1;
        }
        Ok((result, stats))
    }
}

fn decode_assignment(mut index: usize, scope: &[usize], cards: &[usize], assignment: &mut [usize]) {
    for &v in scope.iter().rev() {
        assignment[v] = index % cards[v];
        index /= cards[v];
    }
}


/// Declared algebraic capabilities of an exact coefficient domain.
///
/// This is a planner contract, not a proof that the laws hold. Each concrete
/// domain must separately supply tests/formal obligations for its claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlgebraCapabilities {
    pub semiring_operations: bool,
    pub commutative_multiplication: bool,
    pub additive_inverses: bool,
    pub rational_scalars: bool,
    pub exact_scalar_embedding: bool,
    pub final_coefficients_nonnegative: bool,
    pub coefficient_bound_proven: bool,
    /// The local objective factors have an exact additive score embedding.
    pub local_scalarization_verified: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactAlgorithm {
    BucketVariableElimination,
    SharedMeasurementInterpolation,
    SignedNetworkSum,
    CarryFreeScalarization,
}

/// The semantic answer requested by a caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputContract {
    Feasibility,
    ExactCount,
    WeightedPartitionFunction,
    OptimumWithMultiplicity,
    FixedLexicographicOptimumWithMultiplicity,
    FullObjectiveProfile,
    ParetoFrontierWithMultiplicity,
}

/// Default coefficient algebra selected for an output contract. The final
/// algorithm choice is still constrained by the algebra compatibility gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuggestedAlgebra {
    Boolean,
    NaturalCount,
    CommutativeQAlgebra,
    MaxCount,
    WeightProfile,
    ParetoCount,
}

pub fn suggested_algebra(contract: OutputContract) -> SuggestedAlgebra {
    match contract {
        OutputContract::Feasibility => SuggestedAlgebra::Boolean,
        OutputContract::ExactCount => SuggestedAlgebra::NaturalCount,
        OutputContract::WeightedPartitionFunction => SuggestedAlgebra::CommutativeQAlgebra,
        OutputContract::OptimumWithMultiplicity => SuggestedAlgebra::MaxCount,
        OutputContract::FixedLexicographicOptimumWithMultiplicity => SuggestedAlgebra::MaxCount,
        OutputContract::FullObjectiveProfile => SuggestedAlgebra::WeightProfile,
        OutputContract::ParetoFrontierWithMultiplicity => SuggestedAlgebra::ParetoCount,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompatibilityError {
    SemiringRequired,
    CommutativeMultiplicationRequired,
    AdditiveInversesRequired,
    RationalScalarsRequired,
    ExactScalarEmbeddingRequired,
    NonnegativeFinalCoefficientsRequired,
    ProvenCoefficientBoundRequired,
    LocalScalarizationNotVerified,
}

impl fmt::Display for CompatibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SemiringRequired => write!(f, "algorithm requires semiring operations"),
            Self::CommutativeMultiplicationRequired => write!(f, "algorithm requires commutative multiplication"),
            Self::AdditiveInversesRequired => write!(f, "algorithm requires additive inverses"),
            Self::RationalScalarsRequired => write!(f, "algorithm requires rational scalar multiplication"),
            Self::ExactScalarEmbeddingRequired => write!(f, "algorithm requires an exact ordered scalar embedding"),
            Self::NonnegativeFinalCoefficientsRequired => write!(f, "algorithm requires nonnegative final profile coefficients"),
            Self::ProvenCoefficientBoundRequired => write!(f, "algorithm requires a proved coefficient upper bound"),
            Self::LocalScalarizationNotVerified => write!(f, "algorithm requires a verified factor-to-scalar homomorphism"),
        }
    }
}

/// Fail-closed algorithm/domain compatibility check.
///
/// The caller must still prove or qualify the declared algebra laws; this
/// function only prevents obvious category errors such as routing MaxCount
/// through rational Hermite interpolation.
pub fn check_algorithm_compatibility(
    algorithm: ExactAlgorithm,
    algebra: AlgebraCapabilities,
) -> Result<(), CompatibilityError> {
    match algorithm {
        ExactAlgorithm::BucketVariableElimination => {
            if !algebra.semiring_operations {
                return Err(CompatibilityError::SemiringRequired);
            }
            if !algebra.commutative_multiplication {
                return Err(CompatibilityError::CommutativeMultiplicationRequired);
            }
            Ok(())
        }
        ExactAlgorithm::SharedMeasurementInterpolation => {
            if !algebra.semiring_operations {
                return Err(CompatibilityError::SemiringRequired);
            }
            if !algebra.commutative_multiplication {
                return Err(CompatibilityError::CommutativeMultiplicationRequired);
            }
            if !algebra.additive_inverses {
                return Err(CompatibilityError::AdditiveInversesRequired);
            }
            if !algebra.rational_scalars {
                return Err(CompatibilityError::RationalScalarsRequired);
            }
            Ok(())
        }
        ExactAlgorithm::SignedNetworkSum => {
            if !algebra.semiring_operations {
                return Err(CompatibilityError::SemiringRequired);
            }
            if !algebra.commutative_multiplication {
                return Err(CompatibilityError::CommutativeMultiplicationRequired);
            }
            if !algebra.additive_inverses {
                return Err(CompatibilityError::AdditiveInversesRequired);
            }
            Ok(())
        }
        ExactAlgorithm::CarryFreeScalarization => {
            if !algebra.semiring_operations {
                return Err(CompatibilityError::SemiringRequired);
            }
            if !algebra.exact_scalar_embedding {
                return Err(CompatibilityError::ExactScalarEmbeddingRequired);
            }
            if !algebra.final_coefficients_nonnegative {
                return Err(CompatibilityError::NonnegativeFinalCoefficientsRequired);
            }
            if !algebra.coefficient_bound_proven {
                return Err(CompatibilityError::ProvenCoefficientBoundRequired);
            }
            if !algebra.local_scalarization_verified {
                return Err(CompatibilityError::LocalScalarizationNotVerified);
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_semiring_counts_triangle_independent_sets() {
        let mut g = ExactFactorGraph::new(vec![2,2,2], CheckedCount(0), CheckedCount(1)).unwrap();
        for (u,v) in [(0,1),(1,2),(0,2)] {
            g.add_factor(vec![u,v], vec![CheckedCount(1),CheckedCount(1),CheckedCount(1),CheckedCount(0)]).unwrap();
        }
        let (value, stats) = g.evaluate(&[0,1,2]).unwrap();
        assert_eq!(value, CheckedCount(4));
        assert_eq!(stats.eliminated_variables, 3);
        assert!(stats.maximum_bucket_variables >= 2);
    }

    #[test]
    fn boolean_semiring_answers_triangle_independent_set_feasibility() {
        let mut g = ExactFactorGraph::new(vec![2,2,2], false, true).unwrap();
        for (u,v) in [(0,1),(1,2),(0,2)] {
            g.add_factor(vec![u,v], vec![true,true,true,false]).unwrap();
        }
        let (value, _) = g.evaluate(&[2,1,0]).unwrap();
        assert!(value);
    }

    #[test]
    fn max_count_returns_maxcut_optimum_and_multiplicity() {
        let mut g = ExactFactorGraph::new(vec![2,2,2], MaxCountValue::zero(), MaxCountValue::one()).unwrap();
        for (u,v) in [(0,1),(1,2),(0,2)] {
            g.add_factor(vec![u,v], vec![0,1,1,0].into_iter().map(|s| MaxCountValue::candidate(s,1)).collect()).unwrap();
        }
        let (value, _) = g.evaluate(&[0,1,2]).unwrap();
        assert_eq!(value.optimum(), Some(MaxCountTerm { score: 2, count: 6 }));
    }

    #[test]
    fn pareto_semiring_keeps_nondominated_vectors_and_counts() {
        let mut g = ExactFactorGraph::new(vec![2,2], ParetoCountValue::zero(2), ParetoCountValue::one(2)).unwrap();
        g.add_factor(vec![0,1], vec![
            ParetoCountValue::point(vec![0,0],1),
            ParetoCountValue::point(vec![0,1],1),
            ParetoCountValue::point(vec![1,0],1),
            ParetoCountValue::zero(2),
        ]).unwrap();
        let (value, _) = g.evaluate(&[0,1]).unwrap();
        let points: Vec<(Vec<i128>,u128)> = value.points().map(|(v,c)|(v.to_vec(),c)).collect();
        assert_eq!(points, vec![(vec![0,1],1),(vec![1,0],1)]);
    }

    #[test]
    fn profile_ring_semiring_matches_triangle_maxcut_profile() {
        let bounds = vec![4];
        let zero = WeightProfile::zero(bounds.clone()).unwrap();
        let one = WeightProfile::one(bounds.clone()).unwrap();
        let w = WeightProfile::monomial(bounds.clone(), vec![1], 1).unwrap();
        let mut g = ExactFactorGraph::new(vec![2,2,2], zero, one).unwrap();
        for (u,v) in [(0,1),(1,2),(0,2)] {
            let equal = WeightProfile::one(bounds.clone()).unwrap();
            g.add_factor(vec![u,v], vec![equal.clone(),w.clone(),w.clone(),equal]).unwrap();
        }
        let (p, _) = g.evaluate(&[0,1,2]).unwrap();
        assert_eq!(p.coefficient(&[0]).unwrap(), 2);
        assert_eq!(p.coefficient(&[2]).unwrap(), 6);
        assert_eq!(p.checked_total().unwrap(), 8);
    }

    #[test]
    fn malformed_factors_and_elimination_orders_fail_closed() {
        let mut g = ExactFactorGraph::new(vec![2,2], CheckedCount(0), CheckedCount(1)).unwrap();
        assert!(matches!(g.add_factor(vec![1,0], vec![CheckedCount(1);4]), Err(EliminationError::InvalidScope{..})));
        assert!(matches!(g.add_factor(vec![0,2], vec![CheckedCount(1);4]), Err(EliminationError::VariableOutOfRange{..})));
        assert!(matches!(g.add_factor(vec![0,1], vec![CheckedCount(1);3]), Err(EliminationError::TableLengthMismatch{..})));
        assert!(matches!(g.evaluate(&[0]), Err(EliminationError::InvalidEliminationOrder{..})));
        assert!(matches!(g.evaluate(&[0,0]), Err(EliminationError::InvalidEliminationOrder{..})));
    }

    #[test]
    fn count_overflow_fails_closed() {
        let mut g = ExactFactorGraph::new(vec![1], CheckedCount(0), CheckedCount(1)).unwrap();
        g.add_factor(vec![0], vec![CheckedCount(u128::MAX)]).unwrap();
        // A zero-dimensional product does not overflow; multiplying max by one stays max.
        assert_eq!(g.evaluate(&[0]).unwrap().0, CheckedCount(u128::MAX));
        assert!(CheckedCount(u128::MAX).checked_add(&CheckedCount(1)).is_err());
    }


    #[test]
    fn max_count_ties_add_multiplicities_and_products_add_scores() {
        let a = MaxCountValue::candidate(7, 2);
        let b = MaxCountValue::candidate(7, 5);
        let c = MaxCountValue::candidate(4, 100);
        assert_eq!(a.checked_add(&b).unwrap().optimum(), Some(MaxCountTerm { score: 7, count: 7 }));
        assert_eq!(a.checked_add(&c).unwrap().optimum(), Some(MaxCountTerm { score: 7, count: 2 }));
        assert_eq!(
            a.checked_mul(&b).unwrap().optimum(),
            Some(MaxCountTerm { score: 14, count: 10 })
        );
    }

    #[test]
    fn pareto_pruning_is_preserved_by_minkowski_addition() {
        let dominated = ParetoCountValue::point(vec![1, 1], 3);
        let dominator = ParetoCountValue::point(vec![2, 1], 1);
        let context = ParetoCountValue::point(vec![4, -2], 1);
        let pruned = dominated.checked_add(&dominator).unwrap();
        let left = pruned.checked_mul(&context).unwrap();
        let right = dominated.checked_mul(&context).unwrap()
            .checked_add(&dominator.checked_mul(&context).unwrap()).unwrap();
        assert_eq!(left, right);
        assert_eq!(
            left.points().map(|(v,c)|(v.to_vec(),c)).collect::<Vec<_>>(),
            vec![(vec![6,-1],1)]
        );
    }

    #[test]
    fn elimination_order_changes_width_not_the_exact_answer() {
        let mut g = ExactFactorGraph::new(vec![2,2,2,2], CheckedCount(0), CheckedCount(1)).unwrap();
        for (u,v) in [(0,1),(1,2),(2,3)] {
            g.add_factor(vec![u,v], vec![CheckedCount(1);4]).unwrap();
        }
        let (left, left_stats) = g.evaluate(&[0,1,2,3]).unwrap();
        let (right, right_stats) = g.evaluate(&[1,2,0,3]).unwrap();
        assert_eq!(left, right);
        assert_eq!(left, CheckedCount(16));
        assert!(left_stats.maximum_bucket_variables <= right_stats.maximum_bucket_variables);
    }


    #[test]
    fn output_contract_selects_an_algebra_with_the_same_semantics() {
        assert_eq!(suggested_algebra(OutputContract::Feasibility), SuggestedAlgebra::Boolean);
        assert_eq!(suggested_algebra(OutputContract::ExactCount), SuggestedAlgebra::NaturalCount);
        assert_eq!(suggested_algebra(OutputContract::WeightedPartitionFunction), SuggestedAlgebra::CommutativeQAlgebra);
        assert_eq!(suggested_algebra(OutputContract::OptimumWithMultiplicity), SuggestedAlgebra::MaxCount);
        assert_eq!(suggested_algebra(OutputContract::FixedLexicographicOptimumWithMultiplicity), SuggestedAlgebra::MaxCount);
        assert_eq!(suggested_algebra(OutputContract::FullObjectiveProfile), SuggestedAlgebra::WeightProfile);
        assert_eq!(suggested_algebra(OutputContract::ParetoFrontierWithMultiplicity), SuggestedAlgebra::ParetoCount);
    }

    #[test]
    fn planner_rejects_interpolation_over_max_count_but_allows_variable_elimination() {
        let max_count = AlgebraCapabilities {
            semiring_operations: true,
            commutative_multiplication: true,
            additive_inverses: false,
            rational_scalars: false,
            exact_scalar_embedding: false,
            final_coefficients_nonnegative: false,
            coefficient_bound_proven: false,
            local_scalarization_verified: false,
        };
        assert_eq!(
            check_algorithm_compatibility(ExactAlgorithm::BucketVariableElimination, max_count),
            Ok(())
        );
        assert_eq!(
            check_algorithm_compatibility(ExactAlgorithm::SharedMeasurementInterpolation, max_count),
            Err(CompatibilityError::AdditiveInversesRequired)
        );
        assert_eq!(
            check_algorithm_compatibility(ExactAlgorithm::SignedNetworkSum, max_count),
            Err(CompatibilityError::AdditiveInversesRequired)
        );
    }

    #[test]
    fn planner_distinguishes_integer_ring_from_q_algebra() {
        let integer_profile_ring = AlgebraCapabilities {
            semiring_operations: true,
            commutative_multiplication: true,
            additive_inverses: true,
            rational_scalars: false,
            exact_scalar_embedding: true,
            final_coefficients_nonnegative: true,
            coefficient_bound_proven: true,
            local_scalarization_verified: true,
        };
        assert_eq!(
            check_algorithm_compatibility(ExactAlgorithm::SignedNetworkSum, integer_profile_ring),
            Ok(())
        );
        assert_eq!(
            check_algorithm_compatibility(ExactAlgorithm::SharedMeasurementInterpolation, integer_profile_ring),
            Err(CompatibilityError::RationalScalarsRequired)
        );
        assert_eq!(
            check_algorithm_compatibility(ExactAlgorithm::CarryFreeScalarization, integer_profile_ring),
            Ok(())
        );
    }

    #[test]
    fn planner_rejects_carry_free_without_positive_profile_and_bound() {
        let signed_profile = AlgebraCapabilities {
            semiring_operations: true,
            commutative_multiplication: true,
            additive_inverses: true,
            rational_scalars: false,
            exact_scalar_embedding: true,
            final_coefficients_nonnegative: false,
            coefficient_bound_proven: true,
            local_scalarization_verified: true,
        };
        assert_eq!(
            check_algorithm_compatibility(ExactAlgorithm::CarryFreeScalarization, signed_profile),
            Err(CompatibilityError::NonnegativeFinalCoefficientsRequired)
        );
        let unbounded = AlgebraCapabilities {
            final_coefficients_nonnegative: true,
            coefficient_bound_proven: false,
            ..signed_profile
        };
        assert_eq!(
            check_algorithm_compatibility(ExactAlgorithm::CarryFreeScalarization, unbounded),
            Err(CompatibilityError::ProvenCoefficientBoundRequired)
        );
    }

    #[test]
    fn unconstrained_variables_count_all_states() {
        let g = ExactFactorGraph::new(vec![2,3,4], CheckedCount(0), CheckedCount(1)).unwrap();
        let (value, _) = g.evaluate(&[0,1,2]).unwrap();
        assert_eq!(value, CheckedCount(24));
    }
}
