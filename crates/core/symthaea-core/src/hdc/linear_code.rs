// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Research-only GF(2) kernel for the linear-code HDC recovery comparator.
//!
//! This module implements the algebraic representation and recovery path only.
//! It is not integrated into the production HDC defaults and does not claim
//! production-grade decoding or noise correction.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryCodeword {
    dimension: usize,
    words: Vec<u64>,
}

impl BinaryCodeword {
    pub fn zero(dimension: usize) -> Self {
        Self {
            dimension,
            words: vec![0; words_for(dimension)],
        }
    }

    pub fn from_words(dimension: usize, words: Vec<u64>) -> Self {
        assert_eq!(
            words.len(),
            words_for(dimension),
            "packed word count must exactly match dimension"
        );
        let padding_mask = !last_word_mask(dimension);
        if let Some(&last) = words.last() {
            assert_eq!(
                last & padding_mask,
                0,
                "packed representation contains set padding bits"
            );
        }
        Self { dimension, words }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn bit(&self, index: usize) -> bool {
        assert!(index < self.dimension);
        (self.words[index / 64] >> (index % 64)) & 1 == 1
    }

    pub fn set_bit(&mut self, index: usize, value: bool) {
        assert!(index < self.dimension);
        let mask = 1u64 << (index % 64);
        if value {
            self.words[index / 64] |= mask;
        } else {
            self.words[index / 64] &= !mask;
        }
    }

    pub fn xor_assign(&mut self, other: &Self) {
        assert_eq!(self.dimension, other.dimension);
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a ^= *b;
        }
    }

    pub fn weight(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Convert the Boolean codeword to Raviv's bipolar HDC convention.
    /// GF(2) zero maps to +1 and one maps to -1, so XOR corresponds exactly
    /// to bipolar Hadamard binding.
    pub fn to_bipolar(&self) -> Vec<i8> {
        (0..self.dimension)
            .map(|index| if self.bit(index) { -1 } else { 1 })
            .collect()
    }

    /// Construct a Boolean codeword from a bipolar observation.
    /// Returns None when an observation contains a value other than -1 or +1.
    pub fn from_bipolar(values: &[i8]) -> Option<Self> {
        let mut codeword = Self::zero(values.len());
        for (index, &value) in values.iter().enumerate() {
            match value {
                1 => {}
                -1 => codeword.set_bit(index, true),
                _ => return None,
            }
        }
        Some(codeword)
    }

    /// Boolean-field binding: XOR of the packed codewords.
    pub fn bound(&self, other: &Self) -> Self {
        let mut bound = self.clone();
        bound.xor_assign(other);
        bound
    }
}

#[derive(Debug, Clone)]
pub struct RandomLinearCode {
    dimension: usize,
    rank: usize,
    basis: Vec<BinaryCodeword>,
}

impl RandomLinearCode {
    pub fn generate(dimension: usize, rank: usize, seed: u64) -> Self {
        assert!(dimension > 0, "dimension must be positive");
        assert!(
            rank > 0 && rank <= dimension,
            "rank must be in 1..=dimension"
        );
        // ChaCha12Rng is a named, portable generator; unlike StdRng, its
        // algorithm is fixed, making seed-defined fixtures reproducible across
        // supported platforms and rand releases that preserve this API.
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let mut basis = Vec::with_capacity(rank);

        while basis.len() < rank {
            let mut candidate = BinaryCodeword::zero(dimension);
            for word in &mut candidate.words {
                *word = rng.r#gen();
            }
            if !dimension.is_multiple_of(64)
                && let Some(last) = candidate.words.last_mut()
            {
                *last &= last_word_mask(dimension);
            }
            if extends_span(&basis, &candidate) {
                basis.push(candidate);
            }
        }

        Self {
            dimension,
            rank,
            basis,
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn basis(&self) -> &[BinaryCodeword] {
        &self.basis
    }

    /// Compute a canonical fingerprint of the generated codebook.
    ///
    /// The fingerprint commits to the representation version, dimension, rank,
    /// basis ordering, and packed basis words. It is an evidence identifier,
    /// not a security credential or a substitute for the recorded source and
    /// dependency provenance.
    pub fn fingerprint(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-hdc-linear-code-v1\\0");
        hasher.update(&(self.dimension as u64).to_le_bytes());
        hasher.update(&(self.rank as u64).to_le_bytes());
        hasher.update(&(self.basis.len() as u64).to_le_bytes());
        for vector in &self.basis {
            hasher.update(&(vector.dimension() as u64).to_le_bytes());
            hasher.update(&(vector.words().len() as u64).to_le_bytes());
            for word in vector.words() {
                hasher.update(&word.to_le_bytes());
            }
        }
        *hasher.finalize().as_bytes()
    }

    /// Construct a code from an explicitly supplied independent basis.
    ///
    /// This is useful for reproducing the paper's subcode construction:
    /// choose one parent-code basis and partition it into factor subcode
    /// bases. The constructor rejects malformed or dependent bases so the
    /// resulting object remains a genuine [n,k]_2 code.
    pub fn from_basis(basis: Vec<BinaryCodeword>) -> Option<Self> {
        let dimension = basis.first()?.dimension();
        if basis.iter().any(|vector| vector.dimension() != dimension) {
            return None;
        }
        if basis
            .iter()
            .any(|vector| vector.words.iter().all(|word| *word == 0))
        {
            return None;
        }
        let rank = basis_rank(&basis, dimension);
        (rank == basis.len()).then_some(Self {
            dimension,
            rank,
            basis,
        })
    }

    /// Generate a parent linear code and two subcodes whose bases partition
    /// the parent's basis. This realizes the direct-sum construction
    /// C = K × V used by the research comparator.
    pub fn generate_direct_sum(
        dimension: usize,
        left_rank: usize,
        right_rank: usize,
        seed: u64,
    ) -> Option<(Self, Self, Self)> {
        assert!(dimension > 0, "dimension must be positive");
        assert!(left_rank > 0, "left rank must be positive");
        assert!(right_rank > 0, "right rank must be positive");
        let total_rank = left_rank.checked_add(right_rank)?;
        if total_rank > dimension {
            return None;
        }

        let parent = Self::generate(dimension, total_rank, seed);
        let left = Self::from_basis(parent.basis[..left_rank].to_vec())?;
        let right = Self::from_basis(parent.basis[left_rank..].to_vec())?;
        Some((parent, left, right))
    }

    /// Test whether a word belongs to this code's Boolean subspace.
    pub fn contains(&self, word: &BinaryCodeword) -> bool {
        if word.dimension() != self.dimension {
            return false;
        }
        let mut extended = self.basis.clone();
        extended.push(word.clone());
        basis_rank(&extended, self.dimension) == self.rank
    }

    pub fn encode(&self, message: &[bool]) -> BinaryCodeword {
        assert_eq!(message.len(), self.rank);
        let mut codeword = BinaryCodeword::zero(self.dimension);
        for (bit, generator) in message.iter().zip(&self.basis) {
            if *bit {
                codeword.xor_assign(generator);
            }
        }
        codeword
    }

    pub fn enumerate(&self) -> Vec<BinaryCodeword> {
        assert!(self.rank < usize::BITS as usize);
        let count = 1usize << self.rank;
        (0..count)
            .map(|mask| {
                let message: Vec<bool> = (0..self.rank).map(|bit| (mask >> bit) & 1 == 1).collect();
                self.encode(&message)
            })
            .collect()
    }
}

fn words_for(dimension: usize) -> usize {
    dimension.div_ceil(64)
}

fn last_word_mask(dimension: usize) -> u64 {
    let remainder = dimension % 64;
    if remainder == 0 {
        u64::MAX
    } else {
        (1u64 << remainder) - 1
    }
}

fn extends_span(basis: &[BinaryCodeword], candidate: &BinaryCodeword) -> bool {
    if candidate.words.iter().all(|word| *word == 0) {
        return false;
    }
    let before = basis_rank(basis, candidate.dimension);
    let mut extended = basis.to_vec();
    extended.push(candidate.clone());
    basis_rank(&extended, candidate.dimension) > before
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinearCodeWork {
    pub span_membership_checks: usize,
    pub basis_rank_pivots: usize,
    pub basis_rank_row_xor_words: usize,
    pub basis_rank_input_word_copies: usize,
    pub solve_basis_bit_probes: usize,
    pub solve_matrix_word_cells: usize,
    pub solve_pivots: usize,
    pub solve_row_xor_words: usize,
    pub retained_generators: usize,
    pub projection_word_xor_ops: usize,
}
/// Exact symbolic cardinality of the form 2^e.
///
/// The exponent representation is exact even when the expanded cardinality
/// does not fit in a machine integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExactPowerOfTwo {
    exponent: usize,
}

impl ExactPowerOfTwo {
    pub const fn new(exponent: usize) -> Self {
        Self { exponent }
    }

    pub const fn exponent(self) -> usize {
        self.exponent
    }

    pub const fn is_one(self) -> bool {
        self.exponent == 0
    }
}

/// Exact algebraic geometry of the factor-to-bound map over GF(2).
///
/// Let Delta be the sum of factor dimensions and r be the rank of the
/// concatenated factor generators. The kernel dimension is Delta-r, and
/// every representable target therefore has exactly 2^(Delta-r) factorizations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearCodeAlgebra {
    pub factor_dimension_sum: usize,
    pub union_generator_rank: usize,
    pub kernel_dimension: usize,
    pub raw_factor_tuple_count: ExactPowerOfTwo,
    pub reachable_target_count: ExactPowerOfTwo,
    pub factorization_count_per_target: ExactPowerOfTwo,
    pub unique_factorization: bool,
    /// Minimum number of factor groups participating in a non-zero dependency.
    /// None means the factor spaces are jointly independent.
    pub dependency_order: Option<usize>,
}

impl LinearCodeAlgebra {
    /// Check the exact cardinality conservation law
    ///
    ///     Delta = rank(union) + kernel_dimension
    ///
    /// and its exponent form
    ///
    ///     raw_tuple = reachable_targets * fiber_size.
    ///
    /// Both identities are exact because the factor-to-bound map is linear over GF(2).
    pub const fn satisfies_conservation_law(self) -> bool {
        let Some(rank_sum) = self.union_generator_rank.checked_add(self.kernel_dimension) else {
            return false;
        };
        let Some(exponent_sum) = self
            .reachable_target_count
            .exponent()
            .checked_add(self.factorization_count_per_target.exponent())
        else {
            return false;
        };
        self.factor_dimension_sum == rank_sum
            && self.raw_factor_tuple_count.exponent() == self.factor_dimension_sum
            && self.reachable_target_count.exponent() == self.union_generator_rank
            && self.factorization_count_per_target.exponent() == self.kernel_dimension
            && self.raw_factor_tuple_count.exponent() == exponent_sum
            && self.unique_factorization == (self.kernel_dimension == 0)
            && self.factorization_count_per_target.is_one() == self.unique_factorization
    }

    pub const fn raw_factor_tuple_exponent(self) -> usize {
        self.raw_factor_tuple_count.exponent()
    }

    pub const fn reachable_target_exponent(self) -> usize {
        self.reachable_target_count.exponent()
    }

    pub const fn factorization_exponent(self) -> usize {
        self.factorization_count_per_target.exponent()
    }
}

fn extends_span_with_work(
    basis: &[BinaryCodeword],
    candidate: &BinaryCodeword,
    work: &mut LinearCodeWork,
) -> bool {
    if candidate.words.iter().all(|word| *word == 0) {
        return false;
    }
    work.span_membership_checks += 1;
    let before = basis_rank_counted(basis, candidate.dimension, work);
    let mut extended = basis.to_vec();
    extended.push(candidate.clone());
    basis_rank_counted(&extended, candidate.dimension, work) > before
}
pub fn solve_linear_combination(
    target: &BinaryCodeword,
    basis: &[BinaryCodeword],
) -> Option<Vec<bool>> {
    solve_linear_combination_with_work(target, basis).0
}

/// Solve a GF(2) linear combination and return deterministic packed-work
/// counters alongside the result. These counters are algorithmic work units,
/// not wall-clock measurements.
pub fn solve_linear_combination_with_work(
    target: &BinaryCodeword,
    basis: &[BinaryCodeword],
) -> (Option<Vec<bool>>, LinearCodeWork) {
    let mut work = LinearCodeWork::default();
    let result = solve_linear_combination_counted(target, basis, &mut work);
    (result, work)
}

fn solve_linear_combination_counted(
    target: &BinaryCodeword,
    basis: &[BinaryCodeword],
    work: &mut LinearCodeWork,
) -> Option<Vec<bool>> {
    let dimension = target.dimension();
    if basis.iter().any(|vector| vector.dimension() != dimension) {
        return None;
    }

    let coefficient_words = basis.len().div_ceil(64);
    let augmented_word = coefficient_words;
    work.solve_matrix_word_cells += dimension * (coefficient_words + 1);
    let augmented_mask = 1u64;
    let mut rows: Vec<Vec<u64>> = (0..dimension)
        .map(|row| {
            let mut equation = vec![0u64; coefficient_words + 1];
            for (index, vector) in basis.iter().enumerate() {
                work.solve_basis_bit_probes += 1;
                if vector.bit(row) {
                    equation[index / 64] |= 1u64 << (index % 64);
                }
            }
            if target.bit(row) {
                equation[augmented_word] |= augmented_mask;
            }
            equation
        })
        .collect();

    let mut pivot_row = 0usize;
    let mut pivot_columns = Vec::with_capacity(basis.len());

    for column in 0..basis.len() {
        let word = column / 64;
        let mask = 1u64 << (column % 64);
        let Some(found) = (pivot_row..rows.len()).find(|&row| rows[row][word] & mask != 0) else {
            continue;
        };
        rows.swap(pivot_row, found);

        for row in 0..rows.len() {
            if row != pivot_row && rows[row][word] & mask != 0 {
                work.solve_row_xor_words += rows[row].len();
                for cell in 0..rows[row].len() {
                    rows[row][cell] ^= rows[pivot_row][cell];
                }
            }
        }

        work.solve_pivots += 1;
        pivot_columns.push((pivot_row, column));
        pivot_row += 1;
        if pivot_row == rows.len() {
            break;
        }
    }

    if rows.iter().any(|row| {
        row[..coefficient_words].iter().all(|word| *word == 0)
            && row[augmented_word] & augmented_mask != 0
    }) {
        return None;
    }

    let mut coefficients = vec![false; basis.len()];
    for &(row, column) in &pivot_columns {
        let word = column / 64;
        let mask = 1u64 << (column % 64);
        coefficients[column] = rows[row][augmented_word] & augmented_mask != 0;
        debug_assert!(rows[row][word] & mask != 0);
    }
    Some(coefficients)
}

/// Recover two factors from a clean bound when their linear-code subspaces
/// form a direct sum. This is the two-factor specialization of Raviv's
/// generator-basis binding-recovery construction.
pub fn recover_direct_sum_bound(
    target: &BinaryCodeword,
    left: &RandomLinearCode,
    right: &RandomLinearCode,
) -> Option<(BinaryCodeword, BinaryCodeword)> {
    let factors = recover_independent_bound(target, &[left, right])?;
    let mut factors = factors.into_iter();
    Some((factors.next()?, factors.next()?))
}

/// Recover a deterministic representative factorization for a clean XOR bound.
///
/// The maximal-independent-subset construction and GF(2) solve follow Raviv's
/// Theorem 2. For product/direct-sum factors, the retained generators are
/// already partitioned by factor and this specializes to the paper's exact
/// recovery setting. For overlapping factors, the paper explicitly allows
/// non-unique factorizations; this implementation adds a deterministic
/// owner-based representative rule by assigning each retained generator to
/// the first factor whose supplied basis contains it. That owner rule is a
/// Symthaea research extension, not a claim that the paper specifies a unique
/// projection for overlapping bases. Callers needing uniqueness guarantees
/// should use recover_independent_bound instead.
pub fn recover_linear_bound(
    target: &BinaryCodeword,
    factors: &[&RandomLinearCode],
) -> Option<Vec<BinaryCodeword>> {
    recover_linear_bound_with_work(target, factors).0
}

/// Recover a clean bound and return deterministic work counters for the
/// maximal-independent-subset construction and GF(2) solve.
pub fn recover_linear_bound_with_work(
    target: &BinaryCodeword,
    factors: &[&RandomLinearCode],
) -> (Option<Vec<BinaryCodeword>>, LinearCodeWork) {
    let mut work = LinearCodeWork::default();

    if factors.is_empty() {
        return (None, work);
    }

    let dimension = target.dimension();
    if factors.iter().any(|factor| factor.dimension() != dimension) {
        return (None, work);
    }

    let mut independent_basis = Vec::new();
    let mut owners = Vec::new();

    for (factor_index, factor) in factors.iter().enumerate() {
        for generator in factor.basis() {
            if extends_span_with_work(&independent_basis, generator, &mut work) {
                independent_basis.push(generator.clone());
                owners.push(factor_index);
                work.retained_generators += 1;
            }
        }
    }

    let coefficients = match solve_linear_combination_counted(target, &independent_basis, &mut work)
    {
        Some(coefficients) => coefficients,
        None => return (None, work),
    };

    let mut recovered = factors
        .iter()
        .map(|_| BinaryCodeword::zero(dimension))
        .collect::<Vec<_>>();

    for ((coefficient, generator), &owner) in
        coefficients.iter().zip(&independent_basis).zip(&owners)
    {
        if *coefficient {
            work.projection_word_xor_ops += generator.words.len();
            recovered[owner].xor_assign(generator);
        }
    }

    (Some(recovered), work)
}

/// Recover factors from a clean XOR bound when the participating linear-code
/// generator bases are jointly independent.
///
/// This is the F-factor specialization of Raviv's generator-basis
/// bound-recovery construction in which the maximal independent subset of the
/// union is the union itself. The caller receives one codeword per factor,
/// preserving factor order. Overlapping/dependent factor bases are rejected
/// rather than silently turning a non-unique recovery problem into a unique
/// result.
pub fn recover_independent_bound(
    target: &BinaryCodeword,
    factors: &[&RandomLinearCode],
) -> Option<Vec<BinaryCodeword>> {
    if factors.is_empty() {
        return None;
    }

    let dimension = target.dimension();
    if factors.iter().any(|factor| factor.dimension() != dimension) {
        return None;
    }

    let total_rank = checked_factor_dimension_sum(factors)?;
    let mut basis = Vec::with_capacity(total_rank);
    for factor in factors {
        basis.extend(factor.basis().iter().cloned());
    }

    if basis.len() != total_rank || basis_rank(&basis, dimension) != total_rank {
        return None;
    }

    let coefficients = solve_linear_combination(target, &basis)?;
    let mut offset = 0usize;
    let mut recovered = Vec::with_capacity(factors.len());

    for factor in factors {
        let end = offset.checked_add(factor.rank())?;
        recovered.push(factor.encode(&coefficients[offset..end]));
        offset = end;
    }

    Some(recovered)
}
pub fn basis_rank(vectors: &[BinaryCodeword], dimension: usize) -> usize {
    let mut work = LinearCodeWork::default();
    basis_rank_counted(vectors, dimension, &mut work)
}

fn basis_rank_counted(
    vectors: &[BinaryCodeword],
    dimension: usize,
    work: &mut LinearCodeWork,
) -> usize {
    let mut rows = Vec::with_capacity(vectors.len());
    for vector in vectors {
        if vector.dimension == dimension {
            work.basis_rank_input_word_copies += vector.words.len();
            rows.push(vector.clone());
        }
    }

    let mut rank = 0usize;
    for column in (0..dimension).rev() {
        let pivot = rows[rank..]
            .iter()
            .position(|row| row.bit(column))
            .map(|offset| rank + offset);
        let Some(pivot) = pivot else {
            continue;
        };

        rows.swap(rank, pivot);
        work.basis_rank_pivots += 1;
        let pivot_row = rows[rank].clone();
        for (row, current) in rows.iter_mut().enumerate() {
            if row != rank && current.bit(column) {
                work.basis_rank_row_xor_words += current.words.len();
                current.xor_assign(&pivot_row);
            }
        }
        rank += 1;
        if rank == rows.len() {
            break;
        }
    }
    rank
}

/// Deterministic non-zero GF(2) dependency witness for an ordered factor basis presentation.
///
/// The coefficients are aligned with the concatenation of factor generators in factor order.
/// `factor_support` records exactly which factor groups participate. The witness is canonical
/// relative to that ordered presentation: the first generator that fails maximal-independent
/// extension is selected, then solved against the preceding independent generators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearCodeDependencyWitness {
    pub generator_coefficients: Vec<bool>,
    pub factor_support: Vec<usize>,
    /// The first generator index whose addition created this kernel witness.
    pub dependent_generator_index: usize,
}

impl LinearCodeDependencyWitness {
    pub fn generator_support_size(&self) -> usize {
        self.generator_coefficients
            .iter()
            .filter(|bit| **bit)
            .count()
    }

    pub fn factor_support_size(&self) -> usize {
        self.factor_support.len()
    }

    /// Verify the witness against the ordered factor generator presentation.
    ///
    /// The coefficient vector must be non-zero, have exactly one slot per
    /// supplied generator, identify the declared factor support, and XOR to
    /// the zero word.
    pub fn verifies_against(&self, factors: &[&RandomLinearCode]) -> bool {
        if factors.is_empty() {
            return false;
        }
        let dimension = factors[0].dimension();
        if factors.iter().any(|factor| factor.dimension() != dimension) {
            return false;
        }
        let Some(total_rank) = checked_factor_dimension_sum(factors) else {
            return false;
        };
        if self.generator_coefficients.len() != total_rank
            || self.generator_coefficients.iter().all(|bit| !*bit)
            || self.dependent_generator_index >= total_rank
            || !self.generator_coefficients[self.dependent_generator_index]
        {
            return false;
        }

        let offsets = factor_offsets(factors);
        let expected_support = factors
            .iter()
            .enumerate()
            .filter_map(|(index, _)| {
                let start = offsets[index];
                let end = offsets[index + 1];
                self.generator_coefficients[start..end]
                    .iter()
                    .any(|bit| *bit)
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        if expected_support != self.factor_support || expected_support.len() < 2 {
            return false;
        }

        let mut sum = BinaryCodeword::zero(dimension);
        for (&coefficient, generator) in self
            .generator_coefficients
            .iter()
            .zip(factors.iter().flat_map(|factor| factor.basis().iter()))
        {
            if coefficient {
                sum.xor_assign(generator);
            }
        }
        sum == BinaryCodeword::zero(dimension)
    }
}

/// Compute exact rank/nullity and multiplicity structure for a factor tuple.
///
/// The factor coefficient spaces form a linear domain of dimension Delta = sum(k_i).
/// Concatenating their generator bases defines the map into the ambient Boolean space.
/// Its image has dimension r, so every non-empty fiber has cardinality 2^(Delta-r).
pub fn factorization_algebra(factors: &[&RandomLinearCode]) -> Option<LinearCodeAlgebra> {
    if factors.is_empty() {
        return None;
    }

    let dimension = factors[0].dimension();
    if factors.iter().any(|factor| factor.dimension() != dimension) {
        return None;
    }

    let factor_dimension_sum = factors
        .iter()
        .try_fold(0usize, |sum, factor| sum.checked_add(factor.rank()))?;

    let combined_basis = concatenate_factor_bases(factors, factor_dimension_sum);
    let union_generator_rank = basis_rank(&combined_basis, dimension);
    let kernel_dimension = factor_dimension_sum - union_generator_rank;

    let dependency_order = if kernel_dimension == 0 {
        None
    } else {
        minimum_dependent_factor_order(factors)
    };

    Some(LinearCodeAlgebra {
        factor_dimension_sum,
        union_generator_rank,
        kernel_dimension,
        raw_factor_tuple_count: ExactPowerOfTwo::new(factor_dimension_sum),
        reachable_target_count: ExactPowerOfTwo::new(union_generator_rank),
        factorization_count_per_target: ExactPowerOfTwo::new(kernel_dimension),
        unique_factorization: kernel_dimension == 0,
        dependency_order,
    })
}

/// Explicit affine fiber of the factor-to-bound map for one representable target.
///
/// The representative is one coefficient tuple in the ordered concatenated generator basis.
/// Every other factorization is obtained by XORing that representative with a GF(2) combination
/// of the supplied kernel-basis witnesses. The exact fiber cardinality is therefore carried
/// alongside its constructive ambiguity directions.
///
/// The certificate also carries private provenance and integrity fingerprints
/// bound to the target, ordered factor presentation, representative, kernel
/// witnesses, factor supports, and exact cardinality. Public-field mutation
/// therefore makes bounded operations fail closed rather than silently using
/// stale certificate geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearCodeFactorizationFiber {
    pub representative_coefficients: Vec<bool>,
    pub kernel_basis: Vec<LinearCodeDependencyWitness>,
    pub cardinality: ExactPowerOfTwo,
    /// Immutable binding to the exact target and ordered factor presentation used
    /// to construct this certificate. Public-field mutation cannot refresh it.
    source_fingerprint: [u8; 32],
    /// Immutable integrity tag over the certificate fields and source binding.
    integrity_fingerprint: [u8; 32],
}

fn factorization_source_fingerprint(
    target: &BinaryCodeword,
    factors: &[&RandomLinearCode],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-hdc-linear-code-affine-source-v1\0");
    hasher.update(&(target.dimension() as u64).to_le_bytes());
    hasher.update(&(target.words().len() as u64).to_le_bytes());
    for word in target.words() {
        hasher.update(&word.to_le_bytes());
    }
    hasher.update(&(factors.len() as u64).to_le_bytes());
    for factor in factors {
        hasher.update(&(factor.dimension() as u64).to_le_bytes());
        hasher.update(&(factor.rank() as u64).to_le_bytes());
        hasher.update(&(factor.basis().len() as u64).to_le_bytes());
        for generator in factor.basis() {
            hasher.update(&(generator.dimension() as u64).to_le_bytes());
            hasher.update(&(generator.words().len() as u64).to_le_bytes());
            for word in generator.words() {
                hasher.update(&word.to_le_bytes());
            }
        }
    }
    *hasher.finalize().as_bytes()
}

impl LinearCodeFactorizationFiber {
    fn computed_integrity_fingerprint(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-hdc-linear-code-affine-integrity-v1\0");
        hasher.update(&self.source_fingerprint);
        hasher.update(&(self.representative_coefficients.len() as u64).to_le_bytes());
        for coefficient in &self.representative_coefficients {
            hasher.update(&[*coefficient as u8]);
        }
        hasher.update(&(self.kernel_basis.len() as u64).to_le_bytes());
        for witness in &self.kernel_basis {
            hasher.update(&(witness.generator_coefficients.len() as u64).to_le_bytes());
            for coefficient in &witness.generator_coefficients {
                hasher.update(&[*coefficient as u8]);
            }
            hasher.update(&(witness.factor_support.len() as u64).to_le_bytes());
            for factor_index in &witness.factor_support {
                hasher.update(&(*factor_index as u64).to_le_bytes());
            }
            hasher.update(&(witness.dependent_generator_index as u64).to_le_bytes());
        }
        hasher.update(&(self.cardinality.exponent() as u64).to_le_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Recompute the certificate fingerprint from its current public fields.
    ///
    /// For an unmodified certificate this equals the immutable integrity tag.
    /// Public-field mutation changes the computed fingerprint and is rejected by
    /// bounded operations because the stored tag cannot be refreshed externally.
    pub fn fingerprint(&self) -> [u8; 32] {
        self.computed_integrity_fingerprint()
    }

    fn integrity_is_valid(&self) -> bool {
        self.integrity_fingerprint == self.computed_integrity_fingerprint()
    }
}

/// Iterator over a bounded affine factorization fiber.
#[derive(Debug)]
pub struct LinearCodeFactorizationFiberIter<'a> {
    fiber: &'a LinearCodeFactorizationFiber,
    next_mask: usize,
    total: usize,
}

impl Iterator for LinearCodeFactorizationFiberIter<'_> {
    type Item = Vec<bool>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_mask >= self.total {
            return None;
        }

        let mask = self.next_mask;
        self.next_mask += 1;
        let mut coefficients = self.fiber.representative_coefficients.clone();
        for (kernel_index, witness) in self.fiber.kernel_basis.iter().enumerate() {
            if (mask >> kernel_index) & 1 == 1 {
                for (index, coefficient) in witness.generator_coefficients.iter().enumerate() {
                    coefficients[index] ^= *coefficient;
                }
            }
        }
        Some(coefficients)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total - self.next_mask;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for LinearCodeFactorizationFiberIter<'_> {}

impl LinearCodeFactorizationFiber {
    fn basis_is_well_formed(&self) -> bool {
        if !self.integrity_is_valid() {
            return false;
        }
        if self.cardinality.exponent() != self.kernel_basis.len()
            || self.kernel_basis.iter().any(|witness| {
                witness.generator_coefficients.len() != self.representative_coefficients.len()
            })
        {
            return false;
        }

        let mut dependent_indices = self
            .kernel_basis
            .iter()
            .map(|witness| witness.dependent_generator_index)
            .collect::<Vec<_>>();
        dependent_indices.sort_unstable();
        dependent_indices.dedup();
        if dependent_indices.len() != self.kernel_basis.len()
            || dependent_indices
                .iter()
                .any(|&index| index >= self.representative_coefficients.len())
            || self
                .kernel_basis
                .iter()
                .any(|witness| !witness.generator_coefficients[witness.dependent_generator_index])
        {
            return false;
        }

        let coefficient_basis = self
            .kernel_basis
            .iter()
            .map(|witness| {
                let mut vector = BinaryCodeword::zero(self.representative_coefficients.len());
                for (index, coefficient) in witness.generator_coefficients.iter().enumerate() {
                    vector.set_bit(index, *coefficient);
                }
                vector
            })
            .collect::<Vec<_>>();
        basis_rank(&coefficient_basis, self.representative_coefficients.len())
            == self.kernel_basis.len()
    }

    /// Return one coefficient tuple selected by a GF(2) mask over the kernel basis.
    ///
    /// The returned tuple is guaranteed to remain in the same affine fiber when the
    /// certificate is valid. A mask length mismatch is rejected.
    pub fn coefficients_for_mask(&self, mask: &[bool]) -> Option<Vec<bool>> {
        if mask.len() != self.kernel_basis.len() || !self.basis_is_well_formed() {
            return None;
        }

        let mut coefficients = self.representative_coefficients.clone();
        for (enabled, witness) in mask.iter().zip(&self.kernel_basis) {
            if *enabled {
                for (index, coefficient) in witness.generator_coefficients.iter().enumerate() {
                    coefficients[index] ^= *coefficient;
                }
            }
        }
        Some(coefficients)
    }

    /// Create a bounded iterator over every coefficient tuple in this affine fiber.
    ///
    /// Enumeration is deliberately opt-in and fail-closed: the symbolic cardinality must fit
    /// in the host `usize` and be no larger than `max_fibers`. Paper-scale qualification should
    /// remain symbolic rather than expanding the fiber.
    pub fn iter_bounded(&self, max_fibers: usize) -> Option<LinearCodeFactorizationFiberIter<'_>> {
        if !self.basis_is_well_formed() {
            return None;
        }

        let shift = u32::try_from(self.cardinality.exponent()).ok()?;
        let total = 1usize.checked_shl(shift)?;
        if total > max_fibers {
            return None;
        }

        Some(LinearCodeFactorizationFiberIter {
            fiber: self,
            next_mask: 0,
            total,
        })
    }

    /// Verify the certificate against the ordered factor presentation and target.
    ///
    /// This checks the representative, the kernel witness count, every witness itself,
    /// and the exact symbolic cardinality 2^d.
    pub fn verifies_against(&self, target: &BinaryCodeword, factors: &[&RandomLinearCode]) -> bool {
        let Some(algebra) = factorization_algebra(factors) else {
            return false;
        };
        if target.dimension() != factors[0].dimension()
            || self.representative_coefficients.len() != algebra.factor_dimension_sum
            || self.kernel_basis.len() != algebra.kernel_dimension
            || self.cardinality.exponent() != algebra.kernel_dimension
        {
            return false;
        }

        let expected_source_fingerprint = factorization_source_fingerprint(target, factors);
        if self.source_fingerprint != expected_source_fingerprint || !self.basis_is_well_formed() {
            return false;
        }

        let mut dependent_indices = self
            .kernel_basis
            .iter()
            .map(|witness| witness.dependent_generator_index)
            .collect::<Vec<_>>();
        dependent_indices.sort_unstable();
        dependent_indices.dedup();
        if dependent_indices.len() != self.kernel_basis.len() {
            return false;
        }

        let combined_basis = concatenate_factor_bases(factors, algebra.factor_dimension_sum);
        let mut reconstructed = BinaryCodeword::zero(target.dimension());
        for (&coefficient, generator) in
            self.representative_coefficients.iter().zip(&combined_basis)
        {
            if coefficient {
                reconstructed.xor_assign(generator);
            }
        }
        reconstructed == *target
            && self
                .kernel_basis
                .iter()
                .all(|witness| witness.verifies_against(factors))
    }
}
/// Return a deterministic basis of the kernel of the factor-to-bound map.
///
/// One witness is emitted for every generator that fails the maximal-independent-prefix
/// construction. Each witness therefore has a distinct dependent-generator coordinate, so the
/// witnesses are linearly independent and span the complete kernel. The basis is canonical
/// relative to the ordered factor-generator presentation. It is not a minimum-weight dependency
/// basis.
pub fn factorization_kernel_basis(
    factors: &[&RandomLinearCode],
) -> Option<Vec<LinearCodeDependencyWitness>> {
    if factors.is_empty() {
        return None;
    }

    let dimension = factors[0].dimension();
    if factors.iter().any(|factor| factor.dimension() != dimension) {
        return None;
    }

    let total_rank = factors
        .iter()
        .try_fold(0usize, |sum, factor| sum.checked_add(factor.rank()))?;
    let factor_offsets = factor_offsets(factors);
    let mut independent_basis = Vec::with_capacity(total_rank);
    let mut independent_indices = Vec::with_capacity(total_rank);
    let mut kernel_basis = Vec::new();
    let mut current_index = 0usize;

    for factor in factors {
        for generator in factor.basis() {
            if extends_span(&independent_basis, generator) {
                independent_indices.push(current_index);
                independent_basis.push(generator.clone());
            } else {
                let combination = solve_linear_combination(generator, &independent_basis)?;
                let mut coefficients = vec![false; total_rank];
                for (&coefficient, &original_index) in
                    combination.iter().zip(independent_indices.iter())
                {
                    coefficients[original_index] = coefficient;
                }
                coefficients[current_index] = true;

                let factor_support = factors
                    .iter()
                    .enumerate()
                    .filter_map(|(index, _)| {
                        let start = factor_offsets[index];
                        let end = factor_offsets[index + 1];
                        coefficients[start..end]
                            .iter()
                            .any(|bit| *bit)
                            .then_some(index)
                    })
                    .collect::<Vec<_>>();

                let witness = LinearCodeDependencyWitness {
                    generator_coefficients: coefficients,
                    factor_support,
                    dependent_generator_index: current_index,
                };
                if !witness.verifies_against(factors) {
                    return None;
                }
                kernel_basis.push(witness);
            }
            current_index += 1;
        }
    }

    let expected_kernel_dimension = total_rank - independent_basis.len();
    if kernel_basis.len() != expected_kernel_dimension {
        return None;
    }

    let coefficient_basis = kernel_basis
        .iter()
        .map(|witness| {
            let mut vector = BinaryCodeword::zero(total_rank);
            for (index, coefficient) in witness.generator_coefficients.iter().enumerate() {
                vector.set_bit(index, *coefficient);
            }
            vector
        })
        .collect::<Vec<_>>();
    let mut dependent_indices = kernel_basis
        .iter()
        .map(|witness| witness.dependent_generator_index)
        .collect::<Vec<_>>();
    dependent_indices.sort_unstable();
    dependent_indices.dedup();
    if dependent_indices.len() != kernel_basis.len()
        || kernel_basis.iter().any(|witness| {
            witness.dependent_generator_index >= total_rank
                || !witness.generator_coefficients[witness.dependent_generator_index]
        })
        || basis_rank(&coefficient_basis, total_rank) != expected_kernel_dimension
        || kernel_basis
            .iter()
            .any(|witness| !witness.verifies_against(factors))
    {
        return None;
    }

    Some(kernel_basis)
}

/// Return the first canonical non-zero kernel witness, when the factor spaces are dependent.
pub fn factorization_dependency_witness(
    factors: &[&RandomLinearCode],
) -> Option<LinearCodeDependencyWitness> {
    factorization_kernel_basis(factors)?.into_iter().next()
}

fn checked_factor_dimension_sum(factors: &[&RandomLinearCode]) -> Option<usize> {
    factors
        .iter()
        .try_fold(0usize, |sum, factor| sum.checked_add(factor.rank()))
}

fn factor_offsets(factors: &[&RandomLinearCode]) -> Vec<usize> {
    debug_assert!(checked_factor_dimension_sum(factors).is_some());
    let mut offsets = Vec::with_capacity(factors.len() + 1);
    offsets.push(0);
    for factor in factors {
        offsets.push(offsets.last().copied().unwrap_or(0) + factor.rank());
    }
    offsets
}
/// Construct the full affine-fiber certificate for one representable target.
///
/// The certificate contains one deterministic representative coefficient tuple, the complete
/// kernel basis, and the exact cardinality 2^d. No exhaustive enumeration of the fiber is needed.
pub fn factorization_affine_fiber(
    target: &BinaryCodeword,
    factors: &[&RandomLinearCode],
) -> Option<LinearCodeFactorizationFiber> {
    let algebra = factorization_algebra(factors)?;
    if target.dimension() != factors[0].dimension() {
        return None;
    }

    let combined_basis = concatenate_factor_bases(factors, algebra.factor_dimension_sum);
    let representative_coefficients = solve_linear_combination(target, &combined_basis)?;
    let kernel_basis = factorization_kernel_basis(factors)?;
    if kernel_basis.len() != algebra.kernel_dimension {
        return None;
    }
    let source_fingerprint = factorization_source_fingerprint(target, factors);
    let mut fiber = LinearCodeFactorizationFiber {
        representative_coefficients,
        kernel_basis,
        cardinality: algebra.factorization_count_per_target,
        source_fingerprint,
        integrity_fingerprint: [0; 32],
    };
    fiber.integrity_fingerprint = fiber.computed_integrity_fingerprint();

    if fiber.representative_coefficients.len() != algebra.factor_dimension_sum
        || !fiber.basis_is_well_formed()
    {
        return None;
    }
    let mut reconstructed = BinaryCodeword::zero(target.dimension());
    for (&coefficient, generator) in fiber
        .representative_coefficients
        .iter()
        .zip(&combined_basis)
    {
        if coefficient {
            reconstructed.xor_assign(generator);
        }
    }
    if reconstructed != *target {
        return None;
    }

    let zero_mask = vec![false; fiber.kernel_basis.len()];
    if fiber.coefficients_for_mask(&zero_mask)? != fiber.representative_coefficients {
        return None;
    }
    Some(fiber)
}
/// Return the exact fiber cardinality for a target in the factor-span.
/// None means that the target is not representable by the supplied factors.
pub fn factorization_count_for_target(
    target: &BinaryCodeword,
    factors: &[&RandomLinearCode],
) -> Option<ExactPowerOfTwo> {
    let algebra = factorization_algebra(factors)?;
    if target.dimension() != factors[0].dimension() {
        return None;
    }

    let combined_basis = concatenate_factor_bases(factors, algebra.factor_dimension_sum);
    solve_linear_combination(target, &combined_basis)
        .map(|_| algebra.factorization_count_per_target)
}

fn concatenate_factor_bases(factors: &[&RandomLinearCode], capacity: usize) -> Vec<BinaryCodeword> {
    let mut basis = Vec::with_capacity(capacity);
    for factor in factors {
        basis.extend(factor.basis().iter().cloned());
    }
    basis
}

fn minimum_dependent_factor_order(factors: &[&RandomLinearCode]) -> Option<usize> {
    if factors.len() < 2 {
        return None;
    }

    for subset_size in 2..=factors.len() {
        let mut chosen = Vec::with_capacity(subset_size);
        if has_dependent_factor_subset(factors, subset_size, 0, &mut chosen) {
            return Some(subset_size);
        }
    }

    None
}

fn has_dependent_factor_subset(
    factors: &[&RandomLinearCode],
    subset_size: usize,
    start: usize,
    chosen: &mut Vec<usize>,
) -> bool {
    if chosen.len() == subset_size {
        let Some(rank_sum) = chosen
            .iter()
            .try_fold(0usize, |sum, &index| sum.checked_add(factors[index].rank()))
        else {
            return false;
        };
        let mut basis = Vec::new();
        for &index in chosen.iter() {
            basis.extend(factors[index].basis().iter().cloned());
        }
        return basis_rank(&basis, factors[0].dimension()) < rank_sum;
    }

    let remaining = factors.len() - start;
    if remaining < subset_size - chosen.len() {
        return false;
    }

    for index in start..factors.len() {
        chosen.push(index);
        if has_dependent_factor_subset(factors, subset_size, index + 1, chosen) {
            return true;
        }
        chosen.pop();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conservation_law_rejects_overflowed_public_values() {
        let rank_overflow = LinearCodeAlgebra {
            factor_dimension_sum: 0,
            union_generator_rank: usize::MAX,
            kernel_dimension: 1,
            raw_factor_tuple_count: ExactPowerOfTwo::new(0),
            reachable_target_count: ExactPowerOfTwo::new(0),
            factorization_count_per_target: ExactPowerOfTwo::new(0),
            unique_factorization: false,
            dependency_order: None,
        };
        assert!(!rank_overflow.satisfies_conservation_law());

        let exponent_overflow = LinearCodeAlgebra {
            factor_dimension_sum: 0,
            union_generator_rank: 0,
            kernel_dimension: 0,
            raw_factor_tuple_count: ExactPowerOfTwo::new(0),
            reachable_target_count: ExactPowerOfTwo::new(usize::MAX),
            factorization_count_per_target: ExactPowerOfTwo::new(1),
            unique_factorization: true,
            dependency_order: None,
        };
        assert!(!exponent_overflow.satisfies_conservation_law());
    }

    #[test]
    fn conservation_law_rejects_inconsistent_public_exponents() {
        let inconsistent = LinearCodeAlgebra {
            factor_dimension_sum: 3,
            union_generator_rank: 2,
            kernel_dimension: 1,
            raw_factor_tuple_count: ExactPowerOfTwo::new(3),
            reachable_target_count: ExactPowerOfTwo::new(5),
            factorization_count_per_target: ExactPowerOfTwo::new(1),
            unique_factorization: false,
            dependency_order: Some(3),
        };
        assert!(!inconsistent.satisfies_conservation_law());
    }

    #[test]
    fn affine_fiber_certificate_tracks_representative_kernel_and_cardinality() {
        let code = RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(3, vec![0b001])])
            .expect("code");
        let factors = [&code, &code];
        let target = code.encode(&[true]);
        let fiber = factorization_affine_fiber(&target, &factors).expect("affine fiber");

        assert_eq!(fiber.representative_coefficients.len(), 2);
        assert_eq!(fiber.kernel_basis.len(), 1);
        assert_eq!(fiber.cardinality.exponent(), 1);
        assert!(fiber.verifies_against(&target, &factors));
        assert_eq!(
            fiber.coefficients_for_mask(&[false]).expect("zero mask"),
            fiber.representative_coefficients,
        );
        let alternate = fiber.coefficients_for_mask(&[true]).expect("kernel mask");
        assert_ne!(alternate, fiber.representative_coefficients);

        let mut zero = BinaryCodeword::zero(3);
        for (coefficient, generator) in alternate
            .iter()
            .zip(factors.iter().flat_map(|factor| factor.basis().iter()))
        {
            if *coefficient {
                zero.xor_assign(generator);
            }
        }
        assert_eq!(zero, target);
        assert!(fiber.coefficients_for_mask(&[]).is_none());
    }
    #[test]
    fn dependency_kernel_basis_is_deterministic_and_complete() {
        let c1 = RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(2, vec![0b01])])
            .expect("c1");
        let c2 = RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(2, vec![0b10])])
            .expect("c2");
        let c3 = RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(2, vec![0b11])])
            .expect("c3");
        let factors = [&c1, &c2, &c3];

        let algebra = factorization_algebra(&factors).expect("algebra");
        let first = factorization_kernel_basis(&factors).expect("kernel basis");
        let second = factorization_kernel_basis(&factors).expect("kernel basis");
        assert_eq!(first, second);
        assert_eq!(first.len(), algebra.kernel_dimension);
        assert_eq!(first.len(), 1);
        let dependent_indices = first
            .iter()
            .map(|witness| witness.dependent_generator_index)
            .collect::<Vec<_>>();
        for witness in &first {
            for index in &dependent_indices {
                assert_eq!(
                    witness.generator_coefficients[*index],
                    *index == witness.dependent_generator_index
                );
            }
        }

        let witness = &first[0];
        assert_eq!(witness.generator_coefficients, vec![true, true, true]);
        assert_eq!(witness.factor_support, vec![0, 1, 2]);
        assert_eq!(witness.dependent_generator_index, 2);
        assert_eq!(witness.generator_support_size(), 3);
        assert_eq!(witness.factor_support_size(), 3);
        assert!(witness.verifies_against(&factors));

        let mut coefficient_basis = Vec::new();
        for witness in &first {
            let mut vector = BinaryCodeword::zero(witness.generator_coefficients.len());
            for (index, coefficient) in witness.generator_coefficients.iter().enumerate() {
                vector.set_bit(index, *coefficient);
            }
            coefficient_basis.push(vector);
        }
        assert_eq!(
            basis_rank(&coefficient_basis, witness.generator_coefficients.len(),),
            algebra.kernel_dimension,
        );

        println!(
            "DEPENDENCY_KERNEL_BASIS=fixture=three-way;kernel_dimension={};witness_count={};witness_generator_support_size={};witness_factor_support_size={};verified=true",
            algebra.kernel_dimension,
            first.len(),
            witness.generator_support_size(),
            witness.factor_support_size(),
        );
    }
    #[test]
    fn algebra_is_factor_order_invariant_but_kernel_certificate_is_presentation_bound() {
        let c1 = RandomLinearCode::from_basis(vec![
            BinaryCodeword::from_words(3, vec![0b001]),
            BinaryCodeword::from_words(3, vec![0b010]),
        ])
        .expect("c1");
        let c2 = RandomLinearCode::from_basis(vec![
            BinaryCodeword::from_words(3, vec![0b010]),
            BinaryCodeword::from_words(3, vec![0b100]),
        ])
        .expect("c2");
        let c3 = RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(3, vec![0b100])])
            .expect("c3");

        let forward = [&c1, &c2, &c3];
        let reverse = [&c3, &c2, &c1];
        let a = factorization_algebra(&forward).expect("forward algebra");
        let b = factorization_algebra(&reverse).expect("reverse algebra");

        assert_eq!(a.factor_dimension_sum, b.factor_dimension_sum);
        assert_eq!(a.union_generator_rank, b.union_generator_rank);
        assert_eq!(a.kernel_dimension, b.kernel_dimension);
        assert_eq!(a.raw_factor_tuple_count, b.raw_factor_tuple_count);
        assert_eq!(a.reachable_target_count, b.reachable_target_count);
        assert_eq!(
            a.factorization_count_per_target,
            b.factorization_count_per_target
        );
        assert_eq!(a.unique_factorization, b.unique_factorization);
        assert_eq!(a.dependency_order, b.dependency_order);

        let forward_kernel = factorization_kernel_basis(&forward).expect("forward kernel");
        let reverse_kernel = factorization_kernel_basis(&reverse).expect("reverse kernel");
        assert_eq!(forward_kernel.len(), reverse_kernel.len());
        assert!(
            forward_kernel
                .iter()
                .all(|witness| witness.verifies_against(&forward))
        );
        assert!(
            reverse_kernel
                .iter()
                .all(|witness| witness.verifies_against(&reverse))
        );
        assert_ne!(forward_kernel, reverse_kernel);
    }
    #[test]
    fn independent_factors_have_no_dependency_witness() {
        let (_, left, right) =
            RandomLinearCode::generate_direct_sum(32, 3, 4, 0x51A7).expect("valid direct sum");
        assert!(factorization_dependency_witness(&[&left, &right]).is_none());
    }
    #[test]
    fn factorization_algebra_reports_unique_direct_sum() {
        let (parent, left, right) =
            RandomLinearCode::generate_direct_sum(64, 3, 4, 0xA11CE).expect("valid direct sum");
        let factors: Vec<&RandomLinearCode> = vec![&left, &right];
        let algebra = factorization_algebra(&factors).expect("valid factor algebra");

        assert_eq!(algebra.factor_dimension_sum, parent.rank());
        assert_eq!(algebra.union_generator_rank, parent.rank());
        assert_eq!(algebra.kernel_dimension, 0);
        assert!(algebra.unique_factorization);
        assert_eq!(algebra.raw_factor_tuple_count.exponent(), parent.rank());
        assert_eq!(algebra.reachable_target_count.exponent(), parent.rank());
        assert_eq!(algebra.factorization_count_per_target.exponent(), 0);
        assert!(algebra.satisfies_conservation_law());
        assert_eq!(algebra.dependency_order, None);
    }

    #[test]
    fn factorization_algebra_reports_overlap_and_higher_order_dependency() {
        let c1 = RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(2, vec![0b01])])
            .expect("c1");
        let c2 = RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(2, vec![0b10])])
            .expect("c2");
        let c3 = RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(2, vec![0b11])])
            .expect("c3");

        for pair in [[&c1, &c2], [&c1, &c3], [&c2, &c3]] {
            let algebra = factorization_algebra(&pair).expect("pair algebra");
            assert_eq!(algebra.kernel_dimension, 0);
            assert!(algebra.unique_factorization);
            assert_eq!(algebra.dependency_order, None);
        }

        let algebra = factorization_algebra(&[&c1, &c2, &c3]).expect("triple algebra");
        assert_eq!(algebra.factor_dimension_sum, 3);
        assert_eq!(algebra.union_generator_rank, 2);
        assert_eq!(algebra.kernel_dimension, 1);
        assert!(!algebra.unique_factorization);
        assert_eq!(algebra.dependency_order, Some(3));
        assert_eq!(algebra.factorization_count_per_target.exponent(), 1);
        assert!(algebra.satisfies_conservation_law());
    }
    #[test]
    fn direct_sum_subcodes_partition_a_parent_basis() {
        let (parent, left, right) =
            RandomLinearCode::generate_direct_sum(96, 6, 6, 0x5150).expect("valid direct sum");
        assert_eq!(parent.rank(), 12);
        assert_eq!(left.rank(), 6);
        assert_eq!(right.rank(), 6);

        let mut combined = left.basis().to_vec();
        combined.extend(right.basis().iter().cloned());
        assert_eq!(combined, parent.basis());
        assert_eq!(basis_rank(&combined, 96), parent.rank());
        for word in left.enumerate() {
            assert!(parent.contains(&word));
        }
        for word in right.enumerate() {
            assert!(parent.contains(&word));
        }
    }

    #[test]
    fn from_basis_rejects_dependent_basis() {
        let code = RandomLinearCode::generate(64, 5, 0xA11CE);
        let mut dependent = code.basis().to_vec();
        dependent.push(code.basis()[0].clone());
        assert!(RandomLinearCode::from_basis(dependent).is_none());
    }

    #[test]
    fn codebook_fingerprint_is_reproducible_and_seed_sensitive() {
        let a = RandomLinearCode::generate(96, 8, 0xC0DE);
        let b = RandomLinearCode::generate(96, 8, 0xC0DE);
        let c = RandomLinearCode::generate(96, 8, 0xC0DF);
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_ne!(a.fingerprint(), c.fingerprint());
    }

    #[test]
    fn deterministic_generation_is_reproducible() {
        let a = RandomLinearCode::generate(130, 11, 0xC0DE);
        let b = RandomLinearCode::generate(130, 11, 0xC0DE);
        assert_eq!(a.basis(), b.basis());
        assert_eq!(a.rank(), 11);
    }

    #[test]
    fn generated_basis_has_declared_rank() {
        let code = RandomLinearCode::generate(257, 17, 0xBEEF);
        assert_eq!(basis_rank(code.basis(), code.dimension()), code.rank());
    }

    #[test]
    fn code_is_closed_under_xor() {
        let code = RandomLinearCode::generate(64, 6, 0x1234);
        let words = code.enumerate();
        for a in &words {
            for b in &words {
                let mut sum = a.clone();
                sum.xor_assign(b);
                assert!(words.iter().any(|candidate| candidate == &sum));
            }
        }
    }

    #[test]
    fn enumeration_has_exact_subspace_cardinality() {
        let code = RandomLinearCode::generate(80, 7, 0xCAFE);
        assert_eq!(code.enumerate().len(), 1 << 7);
    }

    #[test]
    #[should_panic(expected = "packed representation contains set padding bits")]
    fn set_padding_bits_are_rejected() {
        let _ = BinaryCodeword::from_words(65, vec![u64::MAX, u64::MAX]);
    }

    #[test]
    #[should_panic(expected = "packed word count must exactly match dimension")]
    fn packed_word_count_mismatch_is_rejected() {
        let _ = BinaryCodeword::from_words(65, vec![0; 1]);
    }

    #[test]
    #[should_panic(expected = "rank must be in 1..=dimension")]
    fn invalid_rank_is_rejected() {
        let _ = RandomLinearCode::generate(8, 9, 1);
    }

    #[test]
    fn zero_is_in_every_generated_code() {
        let code = RandomLinearCode::generate(97, 9, 0xFACE);
        assert!(code.contains(&BinaryCodeword::zero(97)));
    }

    #[test]
    fn membership_rejects_same_dimension_non_members() {
        let code = RandomLinearCode::generate(64, 5, 0x123456);
        let outsider = BinaryCodeword::from_words(64, vec![u64::MAX]);
        assert!(!code.contains(&outsider));
    }

    #[test]
    fn bipolar_round_trip_is_exact() {
        let code = RandomLinearCode::generate(73, 8, 0xABCD);
        for word in code.enumerate() {
            let bipolar = word.to_bipolar();
            assert_eq!(BinaryCodeword::from_bipolar(&bipolar), Some(word));
        }
    }

    #[test]
    fn bipolar_mapping_matches_raviv_boolean_convention() {
        let mut word = BinaryCodeword::zero(3);
        assert_eq!(word.to_bipolar(), vec![1, 1, 1]);

        word.set_bit(0, true);
        assert_eq!(word.to_bipolar(), vec![-1, 1, 1]);
        assert_eq!(BinaryCodeword::from_bipolar(&[-1, 1, 1]), Some(word));
    }

    #[test]
    fn bipolar_binding_matches_boolean_xor() {
        let code = RandomLinearCode::generate(73, 8, 0xABCD);
        let left = code.encode(&[true, false, true, false, true, false, false, true]);
        let right = code.encode(&[false, true, true, false, false, true, false, false]);
        let boolean_bound = left.bound(&right);

        let left_bipolar = left.to_bipolar();
        let right_bipolar = right.to_bipolar();
        let hadamard: Vec<i8> = left_bipolar
            .iter()
            .zip(&right_bipolar)
            .map(|(left, right)| left * right)
            .collect();

        assert_eq!(BinaryCodeword::from_bipolar(&hadamard), Some(boolean_bound));
    }

    #[test]
    fn exhaustive_small_space_hadamard_xor_equivalence() {
        let dimension = 6;
        let words: Vec<_> = (0..(1usize << dimension))
            .map(|mask| BinaryCodeword::from_words(dimension, vec![mask as u64]))
            .collect();

        for left in &words {
            for right in &words {
                let boolean_bound = left.bound(right);
                let hadamard: Vec<i8> = left
                    .to_bipolar()
                    .iter()
                    .zip(right.to_bipolar())
                    .map(|(left, right)| left * right)
                    .collect();
                assert_eq!(BinaryCodeword::from_bipolar(&hadamard), Some(boolean_bound));
            }
        }
    }

    #[test]
    fn malformed_bipolar_observation_is_rejected() {
        assert_eq!(BinaryCodeword::from_bipolar(&[-1, 0, 1]), None);
    }

    #[test]
    fn linear_combination_solver_recovers_unique_coefficients() {
        let left = RandomLinearCode::generate(96, 6, 0x1111);
        let right = RandomLinearCode::generate(96, 6, 0x2222);
        let mut basis = left.basis().to_vec();
        basis.extend(right.basis().iter().cloned());

        assert_eq!(basis_rank(&basis, 96), basis.len());

        let left_message = [true, false, true, false, true, false];
        let right_message = [false, true, true, false, false, true];
        let left_word = left.encode(&left_message);
        let right_word = right.encode(&right_message);
        let composite = left_word.bound(&right_word);

        let coefficients =
            solve_linear_combination(&composite, &basis).expect("composite must be in span");
        let expected: Vec<bool> = left_message.into_iter().chain(right_message).collect();
        assert_eq!(coefficients, expected);

        let mut reconstructed = BinaryCodeword::zero(96);
        for (coefficient, generator) in coefficients.iter().zip(&basis) {
            if *coefficient {
                reconstructed.xor_assign(generator);
            }
        }
        assert_eq!(reconstructed, composite);
    }

    #[test]
    fn exhaustive_solver_matches_membership_on_small_space() {
        let dimension = 8;
        let code = RandomLinearCode::generate(dimension, 4, 0x5A5A);

        for mask in 0..(1usize << dimension) {
            let target = BinaryCodeword::from_words(dimension, vec![mask as u64]);
            let solved = solve_linear_combination(&target, code.basis());

            if code.contains(&target) {
                let coefficients = solved.expect("every codeword has a basis representation");
                let mut reconstructed = BinaryCodeword::zero(dimension);
                for (coefficient, generator) in coefficients.iter().zip(code.basis()) {
                    if *coefficient {
                        reconstructed.xor_assign(generator);
                    }
                }
                assert_eq!(reconstructed, target);
            } else {
                assert!(
                    solved.is_none(),
                    "solver must reject every target outside the code span: mask={mask:#x}"
                );
            }
        }
    }

    #[test]
    fn linear_combination_solver_rejects_outside_span() {
        let code = RandomLinearCode::generate(64, 5, 0x5151);
        let outsider = BinaryCodeword::from_words(64, vec![u64::MAX]);
        assert!(!code.contains(&outsider));
        assert!(solve_linear_combination(&outsider, code.basis()).is_none());
    }

    #[test]
    fn boolean_binding_preserves_code_membership() {
        let code = RandomLinearCode::generate(80, 7, 0xBADA55);
        let a = code.encode(&[true, false, true, false, false, true, false]);
        let b = code.encode(&[false, true, true, false, true, false, false]);
        let bound = a.bound(&b);
        assert!(code.contains(&bound));
        assert_eq!(bound.dimension(), 80);
    }
}
