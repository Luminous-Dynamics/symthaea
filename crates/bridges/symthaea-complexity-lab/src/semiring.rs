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
    StatisticsOverflow { metric: &'static str },
    InvalidRandomDraw { upper_bound: u128, draw: u128 },
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
            Self::StatisticsOverflow { metric } => write!(f, "elimination statistics overflow: {metric}"),
            Self::InvalidRandomDraw { upper_bound, draw } => write!(f, "uniform sampler returned {draw}, outside [0, {upper_bound})"),
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
    /// Extra exact support-count arithmetic performed by witness-aware MaxCount.
    pub support_count_additions: u128,
    pub support_count_multiplications: u128,
    pub generated_factors: usize,
}

/// One separator-assignment row in an exact MaxCount backpointer table.
/// States and their branch multiplicities are parallel arrays.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxCountBackpointerRow {
    pub separator_assignment: Vec<usize>,
    pub maximizing_states: Vec<usize>,
    pub branch_multiplicities: Vec<u128>,
    /// Number of distinct partial assignments represented by each branch.
    /// This is not the weighted semiring multiplicity above.
    pub branch_assignment_counts: Vec<u128>,
}

/// All locally maximizing states for an eliminated variable, indexed by the
/// assignment of variables that remain in its separator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxCountBackpointer {
    pub variable: usize,
    pub separator: Vec<usize>,
    pub rows: Vec<MaxCountBackpointerRow>,
}

/// Measure used when sampling among optimal assignments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaxCountSamplingMeasure {
    /// Every distinct variable assignment has equal probability.
    DistinctAssignments,
    /// Assignment probability is proportional to its MaxCount multiplicity.
    WeightedMultiplicity,
}

/// Exact optimum, semiring multiplicity, one canonical witness, and the
/// complete local argmax tables needed to represent the optimal-assignment DAG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxCountWitnessResult {
    pub value: MaxCountValue,
    /// Number of distinct full variable assignments attaining the optimum.
    /// Unlike MaxCountTerm.count, local factor multiplicity weights do not
    /// inflate this quantity.
    pub distinct_optimal_assignments: u128,
    pub witness: Option<Vec<usize>>,
    pub backpointers: Vec<MaxCountBackpointer>,
    pub stats: EliminationStats,
}


/// Checked counters are part of the execution receipt: overflow must not
/// silently wrap and make the recorded work understated.
fn increment_u128(counter: &mut u128, metric: &'static str) -> Result<(), EliminationError> {
    *counter = counter.checked_add(1).ok_or(EliminationError::StatisticsOverflow { metric })?;
    Ok(())
}

fn increment_usize(counter: &mut usize, metric: &'static str) -> Result<(), EliminationError> {
    *counter = counter.checked_add(1).ok_or(EliminationError::StatisticsOverflow { metric })?;
    Ok(())
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
                        increment_u128(&mut stats.semiring_multiplications, "semiring multiplications")?;
                    }
                    sum = sum.checked_add(&product)?;
                    increment_u128(&mut stats.semiring_additions, "semiring additions")?;
                }
                output.push(sum);
            }

            let mut retained = Vec::with_capacity(factors.len() + 1);
            for (i, factor) in factors.into_iter().enumerate() {
                if !bucket_indices.contains(&i) { retained.push(factor); }
            }
            retained.push(ExactFactor { scope: output_scope, values: output });
            factors = retained;
            increment_usize(&mut stats.eliminated_variables, "eliminated variables")?;
            increment_usize(&mut stats.generated_factors, "generated factors")?;
        }

        let mut result = self.one.clone();
        for factor in factors {
            if !factor.scope.is_empty() {
                return Err(EliminationError::InvalidEliminationOrder { reason: "factor retained an uneliminated variable" });
            }
            result = result.checked_mul(&factor.values[0])?;
            increment_u128(&mut stats.semiring_multiplications, "semiring multiplications")?;
        }
        Ok((result, stats))
    }
}



/// Fail closed if the weighted MaxCount tables and parallel distinct-support
/// tables cease to describe the same factor scopes/cells.
fn validate_support_factor_alignment(
    factors: &[ExactFactor<MaxCountValue>],
    support_factors: &[ExactFactor<u128>],
) -> Result<(), EliminationError> {
    if factors.len() != support_factors.len() {
        return Err(EliminationError::InvalidEliminationOrder {
            reason: "MaxCount and support factor counts diverged",
        });
    }
    for (factor, support) in factors.iter().zip(support_factors.iter()) {
        if factor.scope != support.scope || factor.values.len() != support.values.len() {
            return Err(EliminationError::InvalidEliminationOrder {
                reason: "MaxCount and support factor tables are misaligned",
            });
        }
        for (weighted, distinct) in factor.values.iter().zip(support.values.iter()) {
            if weighted.optimum().is_none() != (*distinct == 0) {
                return Err(EliminationError::InvalidEliminationOrder {
                    reason: "MaxCount feasibility and distinct support disagree",
                });
            }
        }
    }
    Ok(())
}


/// Validate one row and return its total distinct assignment support.
fn maxcount_backpointer_row_support(
    row: &MaxCountBackpointerRow,
    variable_cardinality: usize,
) -> Result<u128, EliminationError> {
    if row.maximizing_states.len() != row.branch_multiplicities.len()
        || row.maximizing_states.len() != row.branch_assignment_counts.len()
    {
        return Err(EliminationError::InvalidEliminationOrder {
            reason: "backpointer row parallel branch arrays are invalid",
        });
    }
    let mut total_support = 0u128;
    let mut previous_state = None;
    for i in 0..row.maximizing_states.len() {
        let state = row.maximizing_states[i];
        let multiplicity = row.branch_multiplicities[i];
        let support = row.branch_assignment_counts[i];
        if state >= variable_cardinality
            || multiplicity == 0
            || support == 0
            || previous_state.is_some_and(|previous| state <= previous)
        {
            return Err(EliminationError::InvalidEliminationOrder {
                reason: "backpointer branch state/count is invalid",
            });
        }
        total_support = total_support.checked_add(support)
            .ok_or(SemiringError::CountOverflow {
                operation: "sampler row support aggregation",
            })?;
        previous_state = Some(state);
    }
    Ok(total_support)
}

/// Return a row's sampling mass under the requested measure after validating
/// its aligned state, multiplicity, and support arrays.
fn maxcount_backpointer_row_weight(
    row: &MaxCountBackpointerRow,
    variable_cardinality: usize,
    measure: MaxCountSamplingMeasure,
) -> Result<u128, EliminationError> {
    let support_total = maxcount_backpointer_row_support(row, variable_cardinality)?;
    match measure {
        MaxCountSamplingMeasure::DistinctAssignments => Ok(support_total),
        MaxCountSamplingMeasure::WeightedMultiplicity => {
            row.branch_multiplicities.iter().try_fold(0u128, |total, &count| {
                total.checked_add(count).ok_or(SemiringError::CountOverflow {
                    operation: "sampler row multiplicity aggregation",
                }.into())
            })
        }
    }
}

/// MaxCount-specialized elimination retaining every local maximizer.
/// The reverse pass returns a canonical optimum witness without conditional
/// re-solves. Backpointer multiplicities are distinct from assignment count:
/// weighted factor multiplicity may exceed the number of variable assignments.
impl ExactFactorGraph<MaxCountValue> {
    /// Exact MaxCount elimination with one witness and all local argmax rows.
    ///
    /// Every original factor-table cell represents exactly one local
    /// assignment, irrespective of its MaxCount multiplicity weight. During
    /// elimination, support counts multiply only across disjoint sets of
    /// previously eliminated variables and add across tied eliminated states.
    /// This computes the number of distinct global optimal assignments in the
    /// same pass as the weighted MaxCount multiplicity.
    pub fn evaluate_with_witness(&self, order: &[usize]) -> Result<MaxCountWitnessResult, EliminationError> {
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
        // Each feasible original table cell corresponds to one local
        // assignment, even when its weighted multiplicity is greater than 1.
        let mut support_factors: Vec<ExactFactor<u128>> = self.factors.iter().map(|factor| {
            ExactFactor {
                scope: factor.scope.clone(),
                values: factor.values.iter()
                    .map(|value| if value.optimum().is_some() { 1u128 } else { 0u128 })
                    .collect(),
            }
        }).collect();
        let mut stats = EliminationStats::default();
        let mut backpointers = Vec::with_capacity(order.len());

        for &eliminated in order {
            validate_support_factor_alignment(&factors, &support_factors)?;
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
            let mut output_support = Vec::with_capacity(output_entries);
            let mut rows = Vec::with_capacity(output_entries);
            for output_index in 0..output_entries {
                let mut assignment = vec![0usize; self.cardinalities.len()];
                decode_assignment(output_index, &output_scope, &self.cardinalities, &mut assignment);
                let mut best_score: Option<i128> = None;
                let mut best_count = 0u128;
                let mut best_support = 0u128;
                let mut maximizing_states = Vec::new();
                let mut branch_multiplicities = Vec::new();
                let mut branch_assignment_counts = Vec::new();

                for state in 0..self.cardinalities[eliminated] {
                    assignment[eliminated] = state;
                    let mut product = self.one.clone();
                    let mut support_product = 1u128;
                    for &factor_index in &bucket_indices {
                        let factor = &factors[factor_index];
                        let cell = factor.table_index(&assignment, &self.cardinalities);
                        product = product.checked_mul(&factor.values[cell])?;
                        increment_u128(&mut stats.semiring_multiplications, "semiring multiplications")?;
                        if product.optimum().is_none() {
                            support_product = 0;
                            break;
                        }
                        support_product = support_product.checked_mul(support_factors[factor_index].values[cell])
                            .ok_or(SemiringError::CountOverflow { operation: "distinct assignment support multiplication" })?;
                        increment_u128(&mut stats.support_count_multiplications, "support count multiplications")?;
                    }

                    if let Some(term) = product.optimum() {
                        match best_score {
                            None => {
                                best_score = Some(term.score);
                                best_count = term.count;
                                best_support = support_product;
                                maximizing_states.clear();
                                branch_multiplicities.clear();
                                branch_assignment_counts.clear();
                                maximizing_states.push(state);
                                branch_multiplicities.push(term.count);
                                branch_assignment_counts.push(support_product);
                            }
                            Some(score) if term.score > score => {
                                best_score = Some(term.score);
                                best_count = term.count;
                                best_support = support_product;
                                maximizing_states.clear();
                                branch_multiplicities.clear();
                                branch_assignment_counts.clear();
                                maximizing_states.push(state);
                                branch_multiplicities.push(term.count);
                                branch_assignment_counts.push(support_product);
                            }
                            Some(score) if term.score == score => {
                                best_count = best_count.checked_add(term.count)
                                    .ok_or(SemiringError::CountOverflow { operation: "MaxCount witness tie aggregation" })?;
                                best_support = best_support.checked_add(support_product)
                                    .ok_or(SemiringError::CountOverflow { operation: "distinct assignment support tie aggregation" })?;
                                increment_u128(&mut stats.support_count_additions, "support count additions")?;
                                maximizing_states.push(state);
                                branch_multiplicities.push(term.count);
                                branch_assignment_counts.push(support_product);
                            }
                            Some(_) => {}
                        }
                    }
                    increment_u128(&mut stats.semiring_additions, "semiring additions")?;
                }
                let value = match best_score {
                    Some(score) => MaxCountValue::candidate(score, best_count),
                    None => MaxCountValue::zero(),
                };
                output.push(value);
                output_support.push(best_support);
                rows.push(MaxCountBackpointerRow {
                    separator_assignment: output_scope.iter().map(|&v| assignment[v]).collect(),
                    maximizing_states,
                    branch_multiplicities,
                    branch_assignment_counts,
                });
            }

            let mut retained = Vec::with_capacity(factors.len() + 1);
            let mut retained_support = Vec::with_capacity(support_factors.len() + 1);
            for (i, factor) in factors.into_iter().enumerate() {
                if !bucket_indices.contains(&i) {
                    retained.push(factor);
                    retained_support.push(support_factors[i].clone());
                }
            }
            retained.push(ExactFactor { scope: output_scope.clone(), values: output });
            retained_support.push(ExactFactor { scope: output_scope.clone(), values: output_support });
            factors = retained;
            support_factors = retained_support;
            backpointers.push(MaxCountBackpointer {
                variable: eliminated,
                separator: output_scope,
                rows,
            });
            increment_usize(&mut stats.eliminated_variables, "eliminated variables")?;
            increment_usize(&mut stats.generated_factors, "generated factors")?;
        }

        validate_support_factor_alignment(&factors, &support_factors)?;
        let mut result = self.one.clone();
        let mut distinct_optimal_assignments = 1u128;
        for (factor, support_factor) in factors.iter().zip(support_factors.iter()) {
            if !factor.scope.is_empty() || !support_factor.scope.is_empty() {
                return Err(EliminationError::InvalidEliminationOrder { reason: "factor retained an uneliminated variable" });
            }
            result = result.checked_mul(&factor.values[0])?;
            increment_u128(&mut stats.semiring_multiplications, "semiring multiplications")?;
            distinct_optimal_assignments = distinct_optimal_assignments
                .checked_mul(support_factor.values[0])
                .ok_or(SemiringError::CountOverflow { operation: "distinct assignment support scalar multiplication" })?;
            increment_u128(&mut stats.support_count_multiplications, "support count multiplications")?;
        }
        if result.optimum().is_none() {
            distinct_optimal_assignments = 0;
        }

        let witness = if result.optimum().is_none() {
            None
        } else {
            let mut chosen = vec![None; self.cardinalities.len()];
            for backpointer in backpointers.iter().rev() {
                let key: Vec<usize> = backpointer.separator.iter()
                    .map(|&v| chosen[v].ok_or(EliminationError::InvalidEliminationOrder {
                        reason: "backpointer separator was not assigned during reverse traversal"
                    }))
                    .collect::<Result<_, _>>()?;
                let row_index = backpointer.separator.iter().zip(key.iter())
                    .fold(0usize, |index, (&v, &state)| index * self.cardinalities[v] + state);
                let row = backpointer.rows.get(row_index)
                    .ok_or(EliminationError::InvalidEliminationOrder { reason: "missing backpointer row" })?;
                if row.separator_assignment != key {
                    return Err(EliminationError::InvalidEliminationOrder { reason: "backpointer row encoding mismatch" });
                }
                let state = *row.maximizing_states.first()
                    .ok_or(EliminationError::InvalidEliminationOrder { reason: "empty backpointer for feasible optimum" })?;
                chosen[backpointer.variable] = Some(state);
            }
            let assignment: Vec<usize> = chosen.into_iter()
                .map(|state| state.ok_or(EliminationError::InvalidEliminationOrder { reason: "witness omitted a variable" }))
                .collect::<Result<_, _>>()?;

            let mut witness_value = self.one.clone();
            for factor in &self.factors {
                let value = &factor.values[factor.table_index(&assignment, &self.cardinalities)];
                witness_value = witness_value.checked_mul(value)?;
            }
            match (witness_value.optimum(), result.optimum()) {
                (Some(witness_term), Some(optimum)) if witness_term.score == optimum.score => Some(assignment),
                _ => return Err(EliminationError::InvalidEliminationOrder { reason: "recovered witness does not attain the optimum" }),
            }
        };

        Ok(MaxCountWitnessResult { value: result, distinct_optimal_assignments, witness, backpointers, stats })
    }

    /// Draw an exactly uniform distinct optimal assignment from the
    /// backpointer DAG without enumerating all optimal assignments.
    ///
    /// The callback must return an unbiased integer in [0, upper_bound).
    /// It is called once per backpointer row visited in reverse elimination
    /// order. Branch probability is proportional to the number of distinct
    /// partial assignments represented by that branch.
    pub fn sample_optimal_assignment<F>(
        &self,
        result: &MaxCountWitnessResult,
        uniform_below: F,
    ) -> Result<Option<Vec<usize>>, EliminationError>
    where
        F: FnMut(u128) -> u128,
    {
        self.sample_optimal_assignment_by_measure(
            result,
            MaxCountSamplingMeasure::DistinctAssignments,
            uniform_below,
        )
    }

    /// Draw an optimum assignment with probability proportional to its
    /// MaxCount multiplicity. This is intentionally different from uniform
    /// sampling over distinct assignments.
    pub fn sample_optimal_assignment_by_multiplicity<F>(
        &self,
        result: &MaxCountWitnessResult,
        uniform_below: F,
    ) -> Result<Option<Vec<usize>>, EliminationError>
    where
        F: FnMut(u128) -> u128,
    {
        self.sample_optimal_assignment_by_measure(
            result,
            MaxCountSamplingMeasure::WeightedMultiplicity,
            uniform_below,
        )
    }

    /// Shared exact backpointer sampler for both measures.
    pub fn sample_optimal_assignment_by_measure<F>(
        &self,
        result: &MaxCountWitnessResult,
        measure: MaxCountSamplingMeasure,
        mut uniform_below: F,
    ) -> Result<Option<Vec<usize>>, EliminationError>
    where
        F: FnMut(u128) -> u128,
    {
        let optimum = match result.value.optimum() {
            None => {
                if result.distinct_optimal_assignments != 0 || result.witness.is_some() {
                    return Err(EliminationError::InvalidEliminationOrder {
                        reason: "infeasible MaxCount result has inconsistent support or witness",
                    });
                }
                return Ok(None);
            }
            Some(term) => term,
        };
        let expected_total = match measure {
            MaxCountSamplingMeasure::DistinctAssignments => result.distinct_optimal_assignments,
            MaxCountSamplingMeasure::WeightedMultiplicity => optimum.count,
        };
        if expected_total == 0 {
            return Err(EliminationError::InvalidEliminationOrder {
                reason: "feasible optimum has zero sampling mass",
            });
        }
        let n = self.cardinalities.len();
        if result.backpointers.len() != n {
            return Err(EliminationError::InvalidEliminationOrder {
                reason: "backpointer count does not match variable count",
            });
        }

        let mut position = vec![usize::MAX; n];
        for (i, backpointer) in result.backpointers.iter().enumerate() {
            let variable = backpointer.variable;
            if variable >= n || position[variable] != usize::MAX {
                return Err(EliminationError::InvalidEliminationOrder {
                    reason: "backpointer variables are not a permutation",
                });
            }
            position[variable] = i;
        }

        let mut root_mass = 1u128;
        for (step, backpointer) in result.backpointers.iter().enumerate() {
            if backpointer.separator.windows(2).any(|pair| pair[0] >= pair[1])
                || backpointer.separator.iter().any(|&v| {
                    v >= n || v == backpointer.variable || position[v] <= step
                })
            {
                return Err(EliminationError::InvalidEliminationOrder {
                    reason: "backpointer separator is not a later-variable scope",
                });
            }
            let mut expected_rows = 1usize;
            for &v in &backpointer.separator {
                expected_rows = expected_rows.checked_mul(self.cardinalities[v])
                    .ok_or(EliminationError::TableSizeOverflow { factor: None })?;
            }
            if backpointer.rows.len() != expected_rows {
                return Err(EliminationError::InvalidEliminationOrder {
                    reason: "backpointer row count does not match separator cardinality",
                });
            }
            if backpointer.separator.is_empty() {
                let root_row = backpointer.rows.get(0)
                    .ok_or(EliminationError::InvalidEliminationOrder {
                        reason: "root backpointer is missing its empty-separator row",
                    })?;
                if !root_row.separator_assignment.is_empty() {
                    return Err(EliminationError::InvalidEliminationOrder {
                        reason: "root backpointer row has a nonempty separator key",
                    });
                }
                let mass = maxcount_backpointer_row_weight(
                    root_row,
                    self.cardinalities[backpointer.variable],
                    measure,
                )?;
                root_mass = root_mass.checked_mul(mass)
                    .ok_or(SemiringError::CountOverflow {
                        operation: "sampler root mass product",
                    })?;
            }
        }
        if root_mass != expected_total {
            return Err(EliminationError::InvalidEliminationOrder {
                reason: "backpointer root mass disagrees with requested sampling total",
            });
        }

        let mut chosen = vec![None; n];
        for backpointer in result.backpointers.iter().rev() {
            let key: Vec<usize> = backpointer.separator.iter()
                .map(|&v| chosen[v].ok_or(EliminationError::InvalidEliminationOrder {
                    reason: "sampler reached an unassigned separator variable",
                }))
                .collect::<Result<_, _>>()?;
            let row_index = backpointer.separator.iter().zip(key.iter())
                .fold(0usize, |index, (&v, &state)| index * self.cardinalities[v] + state);
            let row = backpointer.rows.get(row_index)
                .ok_or(EliminationError::InvalidEliminationOrder {
                    reason: "sampler backpointer row index out of range",
                })?;
            if row.separator_assignment != key {
                return Err(EliminationError::InvalidEliminationOrder {
                    reason: "sampler backpointer row key mismatch",
                });
            }
            let total_mass = maxcount_backpointer_row_weight(
                row,
                self.cardinalities[backpointer.variable],
                measure,
            )?;
            if total_mass == 0 {
                return Err(EliminationError::InvalidEliminationOrder {
                    reason: "reachable backpointer row has zero sampling mass",
                });
            }
            let draw = uniform_below(total_mass);
            if draw >= total_mass {
                return Err(EliminationError::InvalidRandomDraw {
                    upper_bound: total_mass,
                    draw,
                });
            }
            let weights = match measure {
                MaxCountSamplingMeasure::DistinctAssignments => &row.branch_assignment_counts,
                MaxCountSamplingMeasure::WeightedMultiplicity => &row.branch_multiplicities,
            };
            let mut offset = draw;
            let mut selected_state = None;
            for (state, &weight) in row.maximizing_states.iter().zip(weights.iter()) {
                if offset < weight {
                    selected_state = Some(*state);
                    break;
                }
                offset -= weight;
            }
            chosen[backpointer.variable] = Some(selected_state.ok_or(
                EliminationError::InvalidEliminationOrder {
                    reason: "sampler failed to select a weighted branch interval",
                }
            )?);
        }

        let assignment: Vec<usize> = chosen.into_iter()
            .map(|state| state.ok_or(EliminationError::InvalidEliminationOrder {
                reason: "sampler did not assign every variable",
            }))
            .collect::<Result<_, _>>()?;
        let mut evaluated = self.one.clone();
        for factor in &self.factors {
            let value = &factor.values[factor.table_index(&assignment, &self.cardinalities)];
            evaluated = evaluated.checked_mul(value)?;
        }
        match evaluated.optimum() {
            Some(term) if term.score == optimum.score => Ok(Some(assignment)),
            _ => Err(EliminationError::InvalidEliminationOrder {
                reason: "sampled assignment does not attain the reported optimum",
            }),
        }
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
    OptimumWithWitnessAndMultiplicity,
    OptimumWithWitnessMultiplicityAndDistinctAssignmentCount,
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
    MaxCountWithDistinctSupport,
    WeightProfile,
    ParetoCount,
}

pub fn suggested_algebra(contract: OutputContract) -> SuggestedAlgebra {
    match contract {
        OutputContract::Feasibility => SuggestedAlgebra::Boolean,
        OutputContract::ExactCount => SuggestedAlgebra::NaturalCount,
        OutputContract::WeightedPartitionFunction => SuggestedAlgebra::CommutativeQAlgebra,
        OutputContract::OptimumWithMultiplicity => SuggestedAlgebra::MaxCount,
        OutputContract::OptimumWithWitnessAndMultiplicity => SuggestedAlgebra::MaxCount,
        OutputContract::OptimumWithWitnessMultiplicityAndDistinctAssignmentCount => SuggestedAlgebra::MaxCountWithDistinctSupport,
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
    fn max_count_witness_backpointers_recover_triangle_optimum_and_all_ties() {
        let mut g = ExactFactorGraph::new(vec![2,2,2], MaxCountValue::zero(), MaxCountValue::one()).unwrap();
        for (u,v) in [(0,1),(1,2),(0,2)] {
            g.add_factor(vec![u,v], vec![0,1,1,0].into_iter().map(|s| MaxCountValue::candidate(s,1)).collect()).unwrap();
        }
        let result = g.evaluate_with_witness(&[0,1,2]).unwrap();
        assert_eq!(result.value.optimum(), Some(MaxCountTerm { score: 2, count: 6 }));
        assert_eq!(result.distinct_optimal_assignments, 6);
        let witness = result.witness.unwrap();
        let score: i128 = [(0,1),(1,2),(0,2)].iter()
            .map(|&(u,v)| if witness[u] != witness[v] { 1i128 } else { 0i128 }).sum();
        assert_eq!(score, 2);
        assert_eq!(result.backpointers.len(), 3);
        assert!(result.backpointers.iter().any(|bp| bp.rows.iter().any(|row| row.maximizing_states.len() > 1)));
    }

    #[test]
    fn max_count_backpointer_dag_enumerates_every_triangle_maxcut_assignment() {
        fn visit(
            position: isize,
            backpointers: &[MaxCountBackpointer],
            assignment: &mut [Option<usize>],
            output: &mut Vec<Vec<usize>>,
        ) {
            if position < 0 {
                output.push(assignment.iter().map(|v| v.unwrap()).collect());
                return;
            }
            let backpointer = &backpointers[position as usize];
            let key: Vec<usize> = backpointer.separator.iter()
                .map(|&v| assignment[v].unwrap())
                .collect();
            let row_index = backpointer.separator.iter().zip(key.iter())
                .fold(0usize, |index, (&v, &state)| index * [2usize, 2, 2][v] + state);
            let row = &backpointer.rows[row_index];
            assert_eq!(row.separator_assignment, key);
            for &state in &row.maximizing_states {
                assignment[backpointer.variable] = Some(state);
                visit(position - 1, backpointers, assignment, output);
            }
            assignment[backpointer.variable] = None;
        }

        let mut graph = ExactFactorGraph::new(vec![2,2,2], MaxCountValue::zero(), MaxCountValue::one()).unwrap();
        for (u, v) in [(0,1), (1,2), (0,2)] {
            graph.add_factor(vec![u,v], vec![0,1,1,0]
                .into_iter().map(|score| MaxCountValue::candidate(score, 1)).collect()).unwrap();
        }
        let result = graph.evaluate_with_witness(&[0,1,2]).unwrap();
        let mut actual = Vec::new();
        visit(
            result.backpointers.len() as isize - 1,
            &result.backpointers,
            &mut vec![None; 3],
            &mut actual,
        );
        actual.sort();
        let mut expected = Vec::new();
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    let cut = (if a != b { 1 } else { 0 })
                        + (if b != c { 1 } else { 0 })
                        + (if a != c { 1 } else { 0 });
                    if cut == 2 { expected.push(vec![a,b,c]); }
                }
            }
        }
        expected.sort();
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 6);
    }

    #[test]
    fn elimination_statistics_counters_fail_closed_on_overflow() {
        let mut count = u128::MAX;
        assert!(matches!(
            increment_u128(&mut count, "test counter"),
            Err(EliminationError::StatisticsOverflow { metric: "test counter" })
        ));
        let mut steps = usize::MAX;
        assert!(matches!(
            increment_usize(&mut steps, "test steps"),
            Err(EliminationError::StatisticsOverflow { metric: "test steps" })
        ));
    }

    #[test]
    fn max_count_witness_keeps_semiring_multiplicity_distinct_from_assignment_count() {
        let mut g = ExactFactorGraph::new(vec![2], MaxCountValue::zero(), MaxCountValue::one()).unwrap();
        g.add_factor(vec![0], vec![
            MaxCountValue::candidate(5,4),
            MaxCountValue::candidate(4,1),
        ]).unwrap();
        let result = g.evaluate_with_witness(&[0]).unwrap();
        assert_eq!(result.value.optimum(), Some(MaxCountTerm { score: 5, count: 4 }));
        assert_eq!(result.distinct_optimal_assignments, 1);
        assert_eq!(result.witness, Some(vec![0]));
        assert_eq!(result.backpointers[0].rows[0].maximizing_states, vec![0]);
        assert_eq!(result.backpointers[0].rows[0].branch_multiplicities, vec![4]);
        assert_eq!(result.backpointers[0].rows[0].branch_assignment_counts, vec![1]);
    }


    #[test]
    fn max_count_witness_returns_distinct_support_for_unconstrained_variables() {
        let graph = ExactFactorGraph::new(vec![2,3,4], MaxCountValue::zero(), MaxCountValue::one()).unwrap();
        let result = graph.evaluate_with_witness(&[0,1,2]).unwrap();
        assert_eq!(result.value.optimum(), Some(MaxCountTerm { score: 0, count: 24 }));
        assert_eq!(result.distinct_optimal_assignments, 24);
        assert_eq!(result.witness.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn max_count_weighted_ties_separate_multiplicity_from_distinct_support() {
        let mut graph = ExactFactorGraph::new(vec![2], MaxCountValue::zero(), MaxCountValue::one()).unwrap();
        graph.add_factor(vec![0], vec![
            MaxCountValue::candidate(5, 4),
            MaxCountValue::candidate(5, 2),
        ]).unwrap();
        let result = graph.evaluate_with_witness(&[0]).unwrap();
        assert_eq!(result.value.optimum(), Some(MaxCountTerm { score: 5, count: 6 }));
        assert_eq!(result.distinct_optimal_assignments, 2);
        assert_eq!(result.witness, Some(vec![0]));
        assert_eq!(result.backpointers[0].rows[0].maximizing_states, vec![0, 1]);
        assert_eq!(result.backpointers[0].rows[0].branch_multiplicities, vec![4, 2]);
        assert_eq!(result.backpointers[0].rows[0].branch_assignment_counts, vec![1, 1]);
    }

    #[test]
    fn exact_uniform_optimum_sampler_returns_only_maxcut_optima() {
        let mut graph = ExactFactorGraph::new(vec![2,2,2], MaxCountValue::zero(), MaxCountValue::one()).unwrap();
        for (u,v) in [(0,1),(1,2),(0,2)] {
            graph.add_factor(vec![u,v], vec![0,1,1,0]
                .into_iter().map(|score| MaxCountValue::candidate(score, 1)).collect()).unwrap();
        }
        let result = graph.evaluate_with_witness(&[0,1,2]).unwrap();
        let first = graph.sample_optimal_assignment(&result, |_| 0).unwrap().unwrap();
        let last = graph.sample_optimal_assignment(&result, |bound| bound - 1).unwrap().unwrap();
        for assignment in [first, last] {
            let score: i128 = [(0,1),(1,2),(0,2)].iter()
                .map(|&(u,v)| if assignment[u] != assignment[v] { 1i128 } else { 0i128 }).sum();
            assert_eq!(score, 2);
        }
        assert!(matches!(
            graph.sample_optimal_assignment(&result, |_| u128::MAX),
            Err(EliminationError::InvalidRandomDraw { .. })
        ));
        let mut inconsistent = result.clone();
        inconsistent.distinct_optimal_assignments += 1;
        assert!(graph.sample_optimal_assignment(&inconsistent, |_| 0).is_err());
    }

    #[test]
    fn multiplicity_weighted_sampler_uses_a_different_distribution_than_uniform_support() {
        let mut graph = ExactFactorGraph::new(vec![2], MaxCountValue::zero(), MaxCountValue::one()).unwrap();
        graph.add_factor(vec![0], vec![
            MaxCountValue::candidate(5, 4),
            MaxCountValue::candidate(5, 2),
        ]).unwrap();
        let result = graph.evaluate_with_witness(&[0]).unwrap();
        assert_eq!(result.distinct_optimal_assignments, 2);
        assert_eq!(result.value.optimum().unwrap().count, 6);

        // Distinct-uniform sampling assigns one half of the integer interval
        // to each state, while multiplicity-weighted sampling uses 4:2.
        assert_eq!(
            graph.sample_optimal_assignment(&result, |_| 0).unwrap(),
            Some(vec![0])
        );
        assert_eq!(
            graph.sample_optimal_assignment(&result, |_| 1).unwrap(),
            Some(vec![1])
        );
        assert_eq!(
            graph.sample_optimal_assignment_by_multiplicity(&result, |_| 3).unwrap(),
            Some(vec![0])
        );
        assert_eq!(
            graph.sample_optimal_assignment_by_multiplicity(&result, |_| 4).unwrap(),
            Some(vec![1])
        );
    }

    #[test]
    fn exact_uniform_optimum_sampler_handles_infeasible_and_empty_assignment() {
        let mut infeasible = ExactFactorGraph::new(vec![2], MaxCountValue::zero(), MaxCountValue::one()).unwrap();
        infeasible.add_factor(vec![0], vec![MaxCountValue::zero(), MaxCountValue::zero()]).unwrap();
        let result = infeasible.evaluate_with_witness(&[0]).unwrap();
        assert_eq!(infeasible.sample_optimal_assignment(&result, |_| 0).unwrap(), None);

        let empty = ExactFactorGraph::new(vec![], MaxCountValue::zero(), MaxCountValue::one()).unwrap();
        let empty_result = empty.evaluate_with_witness(&[]).unwrap();
        assert_eq!(empty_result.distinct_optimal_assignments, 1);
        assert_eq!(empty.sample_optimal_assignment(&empty_result, |_| 0).unwrap(), Some(vec![]));
    }

    #[test]
    fn exact_uniform_optimum_sampler_multiplies_disconnected_component_supports() {
        let mut graph = ExactFactorGraph::new(vec![2,3], MaxCountValue::zero(), MaxCountValue::one()).unwrap();
        graph.add_factor(vec![0], vec![
            MaxCountValue::candidate(0, 2),
            MaxCountValue::candidate(0, 3),
        ]).unwrap();
        graph.add_factor(vec![1], vec![
            MaxCountValue::candidate(0, 1),
            MaxCountValue::candidate(0, 4),
            MaxCountValue::candidate(0, 5),
        ]).unwrap();
        let result = graph.evaluate_with_witness(&[0,1]).unwrap();
        assert_eq!(result.value.optimum(), Some(MaxCountTerm { score: 0, count: 50 }));
        assert_eq!(result.distinct_optimal_assignments, 6);
        let first = graph.sample_optimal_assignment(&result, |_| 0).unwrap().unwrap();
        let last = graph.sample_optimal_assignment(&result, |bound| bound - 1).unwrap().unwrap();
        assert_ne!(first, last);
        for assignment in [first, last] {
            assert!(assignment[0] < 2 && assignment[1] < 3);
        }
    }

    #[test]
    fn max_count_dual_factor_tables_reject_scope_and_feasibility_misalignment() {
        let weighted = vec![ExactFactor {
            scope: vec![0],
            values: vec![MaxCountValue::candidate(3, 1), MaxCountValue::zero()],
        }];
        let wrong_scope = vec![ExactFactor {
            scope: vec![1],
            values: vec![1u128, 0u128],
        }];
        assert!(validate_support_factor_alignment(&weighted, &wrong_scope).is_err());

        let wrong_feasibility = vec![ExactFactor {
            scope: vec![0],
            values: vec![0u128, 0u128],
        }];
        assert!(validate_support_factor_alignment(&weighted, &wrong_feasibility).is_err());

        let correct = vec![ExactFactor {
            scope: vec![0],
            values: vec![1u128, 0u128],
        }];
        assert!(validate_support_factor_alignment(&weighted, &correct).is_ok());
    }

    #[test]
    fn max_count_support_count_overflow_fails_closed() {
        let mut graph = ExactFactorGraph::new(vec![2], MaxCountValue::zero(), MaxCountValue::one()).unwrap();
        graph.add_factor(vec![0], vec![
            MaxCountValue::candidate(5, 1),
            MaxCountValue::candidate(5, 1),
        ]).unwrap();
        let result = graph.evaluate_with_witness(&[0]).unwrap();
        assert_eq!(result.distinct_optimal_assignments, 2);
        let mut counter = u128::MAX;
        assert!(matches!(
            increment_u128(&mut counter, "support overflow test"),
            Err(EliminationError::StatisticsOverflow { metric: "support overflow test" })
        ));
    }

    #[test]
    fn max_count_witness_rejects_missing_duplicate_orders_and_handles_infeasible_instance() {
        let mut g = ExactFactorGraph::new(vec![2], MaxCountValue::zero(), MaxCountValue::one()).unwrap();
        assert!(g.evaluate_with_witness(&[]).is_err());
        assert!(g.evaluate_with_witness(&[0,0]).is_err());
        g.add_factor(vec![0], vec![MaxCountValue::zero(), MaxCountValue::zero()]).unwrap();
        let result = g.evaluate_with_witness(&[0]).unwrap();
        assert_eq!(result.value.optimum(), None);
        assert_eq!(result.witness, None);
    }

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
        assert_eq!(suggested_algebra(OutputContract::OptimumWithWitnessAndMultiplicity), SuggestedAlgebra::MaxCount);
        assert_eq!(
            suggested_algebra(OutputContract::OptimumWithWitnessMultiplicityAndDistinctAssignmentCount),
            SuggestedAlgebra::MaxCountWithDistinctSupport
        );
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