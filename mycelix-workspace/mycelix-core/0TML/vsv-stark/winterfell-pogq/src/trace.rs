// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Execution Trace Builder for PoGQ AIR
//!
//! Constructs the trace matrix from PoGQ inputs and witness data.

use math::fields::f128::BaseElement;
use math::{FieldElement, StarkField}; // Import for ZERO constant and as_int()
use winterfell::matrix::ColMatrix;

use crate::air::{PublicInputs, TRACE_WIDTH};

/// Decompose a u64 value into n bits (LSB first)
///
/// Returns a vector of n bits where bit[i] = (value >> i) & 1
/// Used for range-check constraints via bit decomposition.
///
/// # Example
/// ```text
/// let bits = decompose_bits(5, 8);  // 5 = 0b00000101
/// assert_eq!(bits, vec![1, 0, 1, 0, 0, 0, 0, 0]);  // LSB first
/// ```
fn decompose_bits(value: u64, n_bits: usize) -> Vec<u64> {
    (0..n_bits).map(|i| (value >> i) & 1).collect()
}

/// Trace builder for PoGQ execution
pub struct TraceBuilder {
    trace_length: usize,
    public: PublicInputs,
    witness_scores: Vec<u64>, // x_t values (Q16.16, fits in u64)
}

impl TraceBuilder {
    /// Create new trace builder
    ///
    /// # Arguments
    /// * `public` - Public inputs (parameters + initial state + expected output)
    /// * `witness_scores` - Private witness (hybrid scores for each round)
    ///
    /// # Panics
    /// Panics if trace_length is not a power of 2.
    pub fn new(public: PublicInputs, witness_scores: Vec<u64>) -> Self {
        assert!(
            public.trace_length.is_power_of_two(),
            "Trace length must be power of 2"
        );
        assert_eq!(
            witness_scores.len(),
            public.trace_length,
            "Witness length must match trace length"
        );

        Self {
            trace_length: public.trace_length,
            public,
            witness_scores,
        }
    }

    /// Build execution trace by simulating PoGQ state machine
    pub fn build(self) -> ColMatrix<BaseElement> {
        let mut trace = Vec::with_capacity(TRACE_WIDTH);
        for _ in 0..TRACE_WIDTH {
            trace.push(vec![BaseElement::ZERO; self.trace_length]);
        }

        const SCALE: u64 = 65536; // Q16.16 scale factor

        // Initialize state at row 0
        trace[0][0] = BaseElement::from(self.public.ema_init);
        trace[1][0] = BaseElement::from(self.public.viol_init);
        trace[2][0] = BaseElement::from(self.public.clear_init);
        trace[3][0] = BaseElement::from(self.public.quar_init);
        trace[4][0] = BaseElement::from(self.witness_scores[0]);
        trace[5][0] = BaseElement::from(self.public.threshold);
        trace[6][0] = BaseElement::from(self.public.round_init);
        trace[11][0] = BaseElement::ZERO;  // Remainder unused for first update

        // LEAN Range Checks: Bit decomposition for row 0
        // Columns 12-27: rem_t bits (16 bits, all 0 for first row)
        let rem_0_bits = decompose_bits(0, 16);
        for (bit_idx, &bit_val) in rem_0_bits.iter().enumerate() {
            trace[12 + bit_idx][0] = BaseElement::from(bit_val);
        }

        // Columns 28-43: x_t bits (16 bits from witness_scores[0])
        let x_0_bits = decompose_bits(self.witness_scores[0], 16);
        for (bit_idx, &bit_val) in x_0_bits.iter().enumerate() {
            trace[28 + bit_idx][0] = BaseElement::from(bit_val);
        }

        // Execute PoGQ state transitions using projected counters
        // Key fix: Selectors at row i use PROJECTED next counters based on row i,
        // not actual next counters from row i+1. This aligns with AIR constraints.
        for i in 0..self.trace_length {
            let ema_t = trace[0][i].as_int() as u64;
            let viol_t = trace[1][i].as_int() as u64;
            let clear_t = trace[2][i].as_int() as u64;
            let quar_t = trace[3][i].as_int() as u64;
            let round_t = trace[6][i].as_int() as u64;
            let x_t = self.witness_scores[i];

            // Compute selectors based on current row
            let is_violation = if x_t < self.public.threshold { 1u64 } else { 0u64 };
            let is_warmup = if round_t <= self.public.w { 1u64 } else { 0u64 };

            // Project what next counters WILL BE (without actually computing them yet)
            // This matches AIR constraint expectations at row i
            let projected_viol = is_violation * (viol_t + 1);
            let projected_clear = (1 - is_violation) * (clear_t + 1);

            // Compute transition selectors using projected counters
            // is_entering: check quar_t==0 to prevent redundant entering
            let is_entering = if projected_viol >= self.public.k && quar_t == 0 { 1u64 } else { 0u64 };
            let is_release = if quar_t == 1 && projected_clear >= self.public.m { 1u64 } else { 0u64 };

            // Set selectors at current row i
            trace[7][i] = BaseElement::from(is_violation);
            trace[8][i] = BaseElement::from(is_warmup);
            trace[9][i] = BaseElement::from(is_release);
            trace[10][i] = BaseElement::from(is_entering);

            // Compute EMA update for next row
            let sum = (self.public.beta as u128) * (ema_t as u128)
                    + ((SCALE as u128 - self.public.beta as u128) * (x_t as u128));
            let ema_next = (sum / (SCALE as u128)) as u64;
            let remainder = (sum - (ema_next as u128) * (SCALE as u128)) as u64;

            // If not last row, update next row i+1
            if i < self.trace_length - 1 {
                // Compute next quarantine state using selectors from current row
                let quar_next = if is_warmup == 1 {
                    0  // Warm-up override
                } else if is_release == 1 {
                    0  // Release from quarantine
                } else if is_entering == 1 {
                    1  // Enter quarantine
                } else {
                    quar_t  // Maintain state
                };

                trace[0][i + 1] = BaseElement::from(ema_next);
                trace[1][i + 1] = BaseElement::from(projected_viol);
                trace[2][i + 1] = BaseElement::from(projected_clear);
                trace[3][i + 1] = BaseElement::from(quar_next);
                trace[4][i + 1] = BaseElement::from(self.witness_scores[i + 1]);
                trace[5][i + 1] = BaseElement::from(self.public.threshold);
                trace[6][i + 1] = BaseElement::from(round_t + 1);
                trace[11][i + 1] = BaseElement::from(remainder);

                // LEAN Range Checks: Bit decomposition for row i+1
                // Columns 12-27: rem_t bits (16 bits from remainder)
                let rem_bits = decompose_bits(remainder, 16);
                for (bit_idx, &bit_val) in rem_bits.iter().enumerate() {
                    trace[12 + bit_idx][i + 1] = BaseElement::from(bit_val);
                }

                // Columns 28-43: x_t bits (16 bits from witness_scores[i+1])
                let x_bits = decompose_bits(self.witness_scores[i + 1], 16);
                for (bit_idx, &bit_val) in x_bits.iter().enumerate() {
                    trace[28 + bit_idx][i + 1] = BaseElement::from(bit_val);
                }
            }
        }

        // Verify final output matches expected
        let final_quar = trace[3][self.trace_length - 1].as_int() as u64;
        assert_eq!(
            final_quar, self.public.quar_out,
            "Trace final state doesn't match expected output"
        );

        let matrix = ColMatrix::new(trace);
        eprintln!("DEBUG TraceBuilder: matrix.num_cols()={}, num_rows()={}",
                  matrix.num_cols(), matrix.num_rows());
        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::{SCALE, AIR_SCHEMA_REV};

    fn q(val: f32) -> u64 {
        (val * SCALE as f32) as u64
    }

    #[test]
    fn test_trace_builder_simple() {
        let public = PublicInputs {
            beta: q(0.85),
            w: 3,
            k: 2,
            m: 3,
            threshold: q(0.90),
            ema_init: q(0.85),
            viol_init: 0,
            clear_init: 0,
            quar_init: 0,
            round_init: 0,
            quar_out: 0, // Expected: no quarantine
            trace_length: 4, // Power of 2
            prov_hash: [0, 0, 0, 0],
            profile_id: 128,
            air_rev: AIR_SCHEMA_REV,
        };

        // Witness: all scores above threshold (no violations)
        let witness_scores = vec![q(0.92), q(0.91), q(0.93), q(0.94)];

        let builder = TraceBuilder::new(public, witness_scores);
        let trace = builder.build();

        // Verify dimensions
        assert_eq!(trace.num_cols(), TRACE_WIDTH);
        assert_eq!(trace.num_rows(), 4);

        // Verify final state
        let final_quar = trace.get(3, 3).as_int();
        assert_eq!(final_quar, 0); // No quarantine
    }
}
