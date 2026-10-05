// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Research-only syndrome decoder for the linear-code HDC comparator.
//!
//! This module deliberately implements a different algorithmic path from the
//! exhaustive codeword oracle and from the existing GF(2) coefficient solver:
//! it derives a parity-check matrix, computes a received-word syndrome, then
//! searches error patterns in increasing Hamming weight. The search bound is
//! explicit, so successful correction is only claimed when a minimum-weight
//! syndrome match is found within that bound.
//!
//! The implementation is intentionally bounded rather than asymptotically
//! efficient. Its purpose is to establish an independent noisy-decoder
//! semantics on small deterministic fixtures before any stronger decoder
//! family is considered.

use super::linear_code::{BinaryCodeword, RandomLinearCode};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SyndromeDecoderWork {
    /// Number of complete error patterns examined, including weight zero.
    pub error_patterns_examined: usize,
    /// Number of selected error positions XORed into the running syndrome.
    pub syndrome_column_xors: usize,
    /// Number of Hamming weights searched, including weight zero.
    pub weights_examined: usize,
    /// Number of minimum-weight error patterns with the received syndrome.
    pub matching_error_patterns: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundedDistanceDecode {
    /// Exactly one minimum-weight error pattern was found within the bound.
    Unique {
        codeword: BinaryCodeword,
        error: BinaryCodeword,
        distance: usize,
    },
    /// More than one minimum-weight error pattern was found at the same
    /// distance, so unique bounded-distance decoding is not available.
    Ambiguous {
        distance: usize,
        matching_error_patterns: usize,
    },
    /// No error pattern at or below the requested bound has the received
    /// syndrome, so the observation is not decoded by this bounded search.
    NoMatchWithinBound { max_error_weight: usize },
    /// The requested bound exceeds the ambient block length.
    InvalidBound {
        max_error_weight: usize,
        dimension: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParityCheckMatrix {
    /// Rows of H, each of length n.
    rows: Vec<BinaryCodeword>,
    /// Columns of H, each of length n-k. These are the syndromes of
    /// single-coordinate error patterns.
    columns: Vec<BinaryCodeword>,
    dimension: usize,
}

impl ParityCheckMatrix {
    /// Derive a full-rank parity-check matrix from an independent generator
    /// basis using a separately implemented GF(2) row-reduction path.
    ///
    /// The generator rows are reduced to RREF. For every non-pivot column f,
    /// a parity-check row is constructed with h_f = 1 and h_p equal to the
    /// RREF coefficient in the pivot row whose pivot is p. This guarantees
    /// G H^T = 0 while retaining n-k independent check rows.
    pub fn from_code(code: &RandomLinearCode) -> Option<Self> {
        let dimension = code.dimension();
        let mut reduced = code.basis().to_vec();
        let mut pivot_columns = Vec::with_capacity(code.rank());
        let mut pivot_row = 0usize;

        for column in 0..dimension {
            let Some(found) = (pivot_row..reduced.len()).find(|&row| reduced[row].bit(column))
            else {
                continue;
            };

            reduced.swap(pivot_row, found);

            for row in 0..reduced.len() {
                if row != pivot_row && reduced[row].bit(column) {
                    let pivot = reduced[pivot_row].clone();
                    reduced[row].xor_assign(&pivot);
                }
            }

            pivot_columns.push(column);
            pivot_row += 1;
            if pivot_row == reduced.len() {
                break;
            }
        }

        if pivot_columns.len() != code.rank() {
            return None;
        }

        let mut is_pivot = vec![false; dimension];
        for &column in &pivot_columns {
            is_pivot[column] = true;
        }

        let mut rows = Vec::with_capacity(dimension - code.rank());
        for free_column in 0..dimension {
            if is_pivot[free_column] {
                continue;
            }

            let mut check = BinaryCodeword::zero(dimension);
            check.set_bit(free_column, true);

            for (row, &pivot_column) in pivot_columns.iter().enumerate() {
                if reduced[row].bit(free_column) {
                    check.set_bit(pivot_column, true);
                }
            }
            rows.push(check);
        }

        let syndrome_dimension = rows.len();
        let mut columns = (0..dimension)
            .map(|_| BinaryCodeword::zero(syndrome_dimension))
            .collect::<Vec<_>>();

        for (row_index, row) in rows.iter().enumerate() {
            for column in 0..dimension {
                if row.bit(column) {
                    columns[column].set_bit(row_index, true);
                }
            }
        }

        Some(Self {
            rows,
            columns,
            dimension,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn syndrome_dimension(&self) -> usize {
        self.rows.len()
    }

    pub fn rows(&self) -> &[BinaryCodeword] {
        &self.rows
    }

    pub fn columns(&self) -> &[BinaryCodeword] {
        &self.columns
    }

    /// Compute H e^T for a binary word e.
    pub fn syndrome(&self, word: &BinaryCodeword) -> Option<BinaryCodeword> {
        if word.dimension() != self.dimension {
            return None;
        }

        let mut syndrome = BinaryCodeword::zero(self.syndrome_dimension());
        for column in 0..self.dimension {
            if word.bit(column) {
                syndrome.xor_assign(&self.columns[column]);
            }
        }
        Some(syndrome)
    }

    /// Canonical evidence fingerprint of H.
    pub fn fingerprint(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-hdc-linear-code-parity-check-v1\0");
        hasher.update(&(self.dimension as u64).to_le_bytes());
        hasher.update(&(self.rows.len() as u64).to_le_bytes());
        for row in &self.rows {
            hasher.update(&(row.dimension() as u64).to_le_bytes());
            hasher.update(&(row.words().len() as u64).to_le_bytes());
            for word in row.words() {
                hasher.update(&word.to_le_bytes());
            }
        }
        *hasher.finalize().as_bytes()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedDistanceSyndromeDecoder {
    parity_check: ParityCheckMatrix,
}

impl BoundedDistanceSyndromeDecoder {
    pub fn from_code(code: &RandomLinearCode) -> Option<Self> {
        Some(Self {
            parity_check: ParityCheckMatrix::from_code(code)?,
        })
    }

    pub fn parity_check(&self) -> &ParityCheckMatrix {
        &self.parity_check
    }

    pub fn decode(
        &self,
        observation: &BinaryCodeword,
        max_error_weight: usize,
    ) -> BoundedDistanceDecode {
        self.decode_with_work(observation, max_error_weight).0
    }

    /// Decode by minimum-weight syndrome search, returning exact deterministic
    /// work counters alongside the outcome.
    ///
    /// The decoder does not enumerate codewords. It enumerates error patterns
    /// by weight and compares their syndromes with the received syndrome.
    pub fn decode_with_work(
        &self,
        observation: &BinaryCodeword,
        max_error_weight: usize,
    ) -> (BoundedDistanceDecode, SyndromeDecoderWork) {
        let mut work = SyndromeDecoderWork::default();

        if observation.dimension() != self.parity_check.dimension() {
            return (
                BoundedDistanceDecode::InvalidObservationDimension {
                    observation_dimension: observation.dimension(),
                    dimension: self.parity_check.dimension(),
                },
                work,
            );
        }

        if max_error_weight > self.parity_check.dimension() {
            return (
                BoundedDistanceDecode::InvalidBound {
                    max_error_weight,
                    dimension: self.parity_check.dimension(),
                },
                work,
            );
        }

        let Some(observed_syndrome) = self.parity_check.syndrome(observation) else {
            unreachable!("dimension checked above");
        };

        for weight in 0..=max_error_weight {
            work.weights_examined += 1;
            let mut running_syndrome = BinaryCodeword::zero(self.parity_check.syndrome_dimension());
            let mut selected = Vec::with_capacity(weight);
            let mut first_error = None;
            let mut matches = 0usize;

            search_exact_weight(
                &self.parity_check.columns,
                &observed_syndrome,
                weight,
                0,
                &mut selected,
                &mut running_syndrome,
                &mut work,
                &mut matches,
                &mut first_error,
            );

            if matches == 0 {
                continue;
            }

            work.matching_error_patterns = matches;
            let error = first_error.expect("match count implies a stored representative");

            if matches == 1 {
                let mut codeword = observation.clone();
                codeword.xor_assign(&error);
                return (
                    BoundedDistanceDecode::Unique {
                        codeword,
                        error,
                        distance: weight,
                    },
                    work,
                );
            }

            return (
                BoundedDistanceDecode::Ambiguous {
                    distance: weight,
                    matching_error_patterns: matches,
                },
                work,
            );
        }

        (
            BoundedDistanceDecode::NoMatchWithinBound { max_error_weight },
            work,
        )
    }
}

fn search_exact_weight(
    columns: &[BinaryCodeword],
    observed_syndrome: &BinaryCodeword,
    target_weight: usize,
    start: usize,
    selected: &mut Vec<usize>,
    running_syndrome: &mut BinaryCodeword,
    work: &mut SyndromeDecoderWork,
    matches: &mut usize,
    first_error: &mut Option<BinaryCodeword>,
) {
    if selected.len() == target_weight {
        work.error_patterns_examined += 1;
        if *running_syndrome == *observed_syndrome {
            *matches += 1;
            if first_error.is_none() {
                let mut error = BinaryCodeword::zero(columns.len());
                for &index in selected.iter() {
                    error.set_bit(index, true);
                }
                *first_error = Some(error);
            }
        }
        return;
    }

    let remaining = target_weight - selected.len();
    if remaining == 0 || start >= columns.len() {
        return;
    }

    let last_start = columns.len() - remaining;
    for index in start..=last_start {
        running_syndrome.xor_assign(&columns[index]);
        work.syndrome_column_xors += 1;
        selected.push(index);

        search_exact_weight(
            columns,
            observed_syndrome,
            target_weight,
            index + 1,
            selected,
            running_syndrome,
            work,
            matches,
            first_error,
        );

        selected.pop();
        running_syndrome.xor_assign(&columns[index]);
    }
}
