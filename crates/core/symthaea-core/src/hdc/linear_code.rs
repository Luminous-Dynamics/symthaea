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
}
