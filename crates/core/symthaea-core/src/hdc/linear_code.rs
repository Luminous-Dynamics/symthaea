// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Research-only GF(2) kernel for the linear-code HDC recovery comparator.
//!
//! This module intentionally stops at the algebraic substrate. It does not
//! claim to implement Raviv's recovery algorithm yet.

use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryCodeword {
    dimension: usize,
    words: Vec<u64>,
}

impl BinaryCodeword {
    pub fn zero(dimension: usize) -> Self {
        Self { dimension, words: vec![0; words_for(dimension)] }
    }

    pub fn from_words(dimension: usize, mut words: Vec<u64>) -> Self {
        words.resize(words_for(dimension), 0);
        if let Some(last) = words.last_mut() {
            *last &= last_word_mask(dimension);
        }
        Self { dimension, words }
    }

    pub fn dimension(&self) -> usize { self.dimension }

    pub fn bit(&self, index: usize) -> bool {
        assert!(index < self.dimension);
        (self.words[index / 64] >> (index % 64)) & 1 == 1
    }

    pub fn set_bit(&mut self, index: usize, value: bool) {
        assert!(index < self.dimension);
        let mask = 1u64 << (index % 64);
        if value { self.words[index / 64] |= mask; }
        else { self.words[index / 64] &= !mask; }
    }

    pub fn xor_assign(&mut self, other: &Self) {
        assert_eq!(self.dimension, other.dimension);
        for (a, b) in self.words.iter_mut().zip(&other.words) { *a ^= *b; }
    }

    pub fn weight(&self) -> usize {
        self.words.iter().map(|word| word.count_ones() as usize).sum()
    }

    pub fn words(&self) -> &[u64] { &self.words }

    /// Convert the Boolean codeword to the bipolar HDC observation convention.
    /// GF(2) zero maps to -1 and one maps to +1.
    pub fn to_bipolar(&self) -> Vec<i8> {
        (0..self.dimension)
            .map(|index| if self.bit(index) { 1 } else { -1 })
            .collect()
    }

    /// Construct a Boolean codeword from a bipolar observation.
    /// Returns None when an observation contains a value other than -1 or +1.
    pub fn from_bipolar(values: &[i8]) -> Option<Self> {
        let mut codeword = Self::zero(values.len());
        for (index, &value) in values.iter().enumerate() {
            match value {
                -1 => {}
                1 => codeword.set_bit(index, true),
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
        assert!(rank > 0 && rank <= dimension, "rank must be in 1..=dimension");
        let mut rng = StdRng::seed_from_u64(seed);
        let mut basis = Vec::with_capacity(rank);

        while basis.len() < rank {
            let mut candidate = BinaryCodeword::zero(dimension);
            for word in &mut candidate.words { *word = rng.gen(); }
            if dimension % 64 != 0 {
                if let Some(last) = candidate.words.last_mut() {
                    *last &= last_word_mask(dimension);
                }
            }
            if extends_span(&basis, &candidate) { basis.push(candidate); }
        }

        Self { dimension, rank, basis }
    }

    pub fn dimension(&self) -> usize { self.dimension }
    pub fn rank(&self) -> usize { self.rank }
    pub fn basis(&self) -> &[BinaryCodeword] { &self.basis }

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
            if *bit { codeword.xor_assign(generator); }
        }
        codeword
    }

    pub fn enumerate(&self) -> Vec<BinaryCodeword> {
        assert!(self.rank < usize::BITS as usize);
        let count = 1usize << self.rank;
        (0..count).map(|mask| {
            let message: Vec<bool> = (0..self.rank)
                .map(|bit| (mask >> bit) & 1 == 1)
                .collect();
            self.encode(&message)
        }).collect()
    }
}

fn words_for(dimension: usize) -> usize { dimension.div_ceil(64) }

fn last_word_mask(dimension: usize) -> u64 {
    let remainder = dimension % 64;
    if remainder == 0 { u64::MAX } else { (1u64 << remainder) - 1 }
}

fn extends_span(basis: &[BinaryCodeword], candidate: &BinaryCodeword) -> bool {
    if candidate.words.iter().all(|word| *word == 0) { return false; }
    let before = basis_rank(basis, candidate.dimension);
    let mut extended = basis.to_vec();
    extended.push(candidate.clone());
    basis_rank(&extended, candidate.dimension) > before
}

/// Solve target = XOR_i(coefficients[i] * basis[i]) over GF(2).
///
/// This is the algebraic core of the research bound-recovery comparator.
/// It returns one coefficient vector when a solution exists. Callers that
/// require unique factorization must independently verify that the supplied
/// basis is linearly independent (for example, by checking
/// basis_rank(basis, dimension) == basis.len()).
pub fn solve_linear_combination(
    target: &BinaryCodeword,
    basis: &[BinaryCodeword],
) -> Option<Vec<bool>> {
    let dimension = target.dimension();
    if basis.iter().any(|vector| vector.dimension() != dimension) {
        return None;
    }

    // Augmented rows: [basis coefficients | target bit], represented as a
    // packed vector so Gaussian elimination remains entirely in GF(2).
    let width = basis.len() + 1;
    let mut rows: Vec<Vec<bool>> = (0..dimension)
        .map(|row| {
            let mut equation = Vec::with_capacity(width);
            for vector in basis {
                equation.push(vector.bit(row));
            }
            equation.push(target.bit(row));
            equation
        })
        .collect();

    let mut pivot_row = 0usize;
    let mut pivot_columns = Vec::with_capacity(basis.len());

    for column in 0..basis.len() {
        let Some(found) = (pivot_row..rows.len()).find(|&row| rows[row][column]) else {
            continue;
        };
        rows.swap(pivot_row, found);

        for row in 0..rows.len() {
            if row != pivot_row && rows[row][column] {
                for bit in column..width {
                    rows[row][bit] ^= rows[pivot_row][bit];
                }
            }
        }

        pivot_columns.push((pivot_row, column));
        pivot_row += 1;
        if pivot_row == rows.len() {
            break;
        }
    }

    // 0 = 1 means the target is outside the span.
    if rows.iter().any(|row| row[..basis.len()].iter().all(|bit| !*bit) && row[basis.len()]) {
        return None;
    }

    // With a full-rank basis this is the unique coefficient vector. If the
    // basis is dependent, leave free variables at zero and return one valid
    // solution; callers needing uniqueness must check rank separately.
    let mut coefficients = vec![false; basis.len()];
    for &(row, column) in &pivot_columns {
        coefficients[column] = rows[row][basis.len()];
    }
    Some(coefficients)
}

pub fn basis_rank(vectors: &[BinaryCodeword], dimension: usize) -> usize {
    let mut rows: Vec<BinaryCodeword> = vectors.iter()
        .filter(|vector| vector.dimension == dimension)
        .cloned()
        .collect();

    let mut rank = 0usize;
    for column in (0..dimension).rev() {
        let pivot = rows[rank..].iter()
            .position(|row| row.bit(column))
            .map(|offset| rank + offset);
        let Some(pivot) = pivot else { continue; };

        rows.swap(rank, pivot);
        let pivot_row = rows[rank].clone();
        for row in 0..rows.len() {
            if row != rank && rows[row].bit(column) {
                rows[row].xor_assign(&pivot_row);
            }
        }
        rank += 1;
        if rank == rows.len() { break; }
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn unused_high_bits_are_masked() {
        let vector = BinaryCodeword::from_words(65, vec![u64::MAX, u64::MAX]);
        assert_eq!(vector.weight(), 65);
        assert!(vector.bit(64));
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

        let coefficients = solve_linear_combination(&composite, &basis).expect("composite must be in span");
        let expected: Vec<bool> = left_message
            .into_iter()
            .chain(right_message)
            .collect();
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
