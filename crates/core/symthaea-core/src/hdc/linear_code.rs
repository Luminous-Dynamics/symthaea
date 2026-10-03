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

    pub fn from_words(dimension: usize, mut words: Vec<u64>) -> Self {
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

/// Recover a representative factorization for a clean XOR bound using the
/// maximal-independent-subset construction from Raviv's Theorem 2.
///
/// The returned factorization is valid when one exists, but it is deliberately
/// not labeled unique: overlapping factor subcodes can admit multiple valid
/// decompositions. Each retained generator is assigned deterministically to the
/// first factor whose supplied basis contains it. This is a faithful
/// representation-level implementation of the paper's constructive recovery
/// path; callers needing a uniqueness guarantee should use
/// recover_independent_bound instead.
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

    let coefficients =
        match solve_linear_combination_counted(target, &independent_basis, &mut work) {
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

    let total_rank = factors.iter().map(|factor| factor.rank()).sum::<usize>();
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
        let end = offset + factor.rank();
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

#[cfg(test)]
mod tests {
    use super::*;

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
