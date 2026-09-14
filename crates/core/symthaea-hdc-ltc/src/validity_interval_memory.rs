// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Checkpoint-validity associative memory over the temporal phasor algebra.
//!
//! Physical time and causal/version time are deliberately separated here.
//! `TemporalAxis` remains a generic continuous group action. This memory assigns
//! each fully observed world state a checkpoint coordinate `n + 1/2` and encodes
//! a value valid for checkpoints `[start, end)` as the finite phasor series
//!
//! `S[start,end) = sum_{n=start}^{end-1} T(n + 1/2)`.
//!
//! The series is evaluated analytically per Fourier channel using the Dirichlet
//! kernel, so write cost is `O(D)` regardless of interval length. For frequencies
//! sampled uniformly from `[-pi, pi)`, distinct integer checkpoint offsets are
//! orthogonal in expectation. Finite-dimensional cleanup error is therefore a
//! measurable crosstalk question rather than an implicit timestamp heuristic.
//!
//! ## Finite-precision causal-time contract
//!
//! Discrete checkpoint `n` is mapped to the binary64 coordinate `n + 1/2`.
//! That half-integer is represented exactly only while `n < 2^52`, because the
//! odd numerator `2n + 1` must fit the 53-bit binary64 significand. Public
//! checkpoint queries therefore reject `n >= 2^52`; closed spans may end at
//! `2^52` but may not contain a checkpoint at or beyond that boundary.
//!
//! This is a coordinate-representation guarantee only. It does not claim exact
//! trigonometric argument reduction or unlimited-duration historical memory.

use crate::continuous_hv::UnitaryRole;
use crate::temporal_phasor::{TemporalAlgebraError, TemporalAxis, TemporalPhasor};
use std::fmt;

/// Exclusive upper bound for a discrete checkpoint whose center `n + 1/2` is
/// exactly representable as a binary64 value.
pub const EXACT_CAUSAL_CHECKPOINT_LIMIT: u64 = 1_u64 << 52;

const SINC_TAYLOR_THRESHOLD: f64 = 1.0e-4;

#[derive(Debug, Clone, PartialEq)]
pub enum ValidityMemoryError {
    ZeroDimension,
    DimensionMismatch { expected: usize, actual: usize },
    InvalidCheckpointInterval { start: u64, end_exclusive: u64 },
    CheckpointOutOfExactRange { checkpoint: u64, limit_exclusive: u64 },
    CheckpointIntervalOutOfExactRange {
        start: u64,
        end_exclusive: u64,
        limit_exclusive: u64,
    },
    TooFewCandidates { count: usize },
    Temporal(TemporalAlgebraError),
    NonFiniteScore,
}

impl fmt::Display for ValidityMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "validity memory dimension must be non-zero"),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "validity memory dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidCheckpointInterval { start, end_exclusive } => write!(
                f,
                "validity interval must satisfy start < end_exclusive, got [{start}, {end_exclusive})"
            ),
            Self::CheckpointOutOfExactRange {
                checkpoint,
                limit_exclusive,
            } => write!(
                f,
                "checkpoint {checkpoint} is outside the exact binary64 causal-coordinate domain 0..{limit_exclusive}"
            ),
            Self::CheckpointIntervalOutOfExactRange {
                start,
                end_exclusive,
                limit_exclusive,
            } => write!(
                f,
                "validity interval [{start}, {end_exclusive}) exceeds the exact binary64 causal-coordinate domain 0..={limit_exclusive} for interval boundaries"
            ),
            Self::TooFewCandidates { count } => write!(
                f,
                "validity cleanup requires at least two candidates, got {count}"
            ),
            Self::Temporal(error) => write!(f, "validity memory temporal error: {error}"),
            Self::NonFiniteScore => write!(f, "validity memory produced a non-finite score"),
        }
    }
}

impl std::error::Error for ValidityMemoryError {}

impl From<TemporalAlgebraError> for ValidityMemoryError {
    fn from(value: TemporalAlgebraError) -> Self {
        Self::Temporal(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidityCleanupResult {
    pub best_index: usize,
    pub best_score: f64,
    pub second_score: f64,
    pub margin: f64,
}

/// Superposed Fourier-domain archive of closed key/value validity spans.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidityIntervalMemory {
    real: Vec<f64>,
    imag: Vec<f64>,
    spans_written: usize,
}

impl ValidityIntervalMemory {
    pub fn new(dim: usize) -> Result<Self, ValidityMemoryError> {
        if dim == 0 {
            return Err(ValidityMemoryError::ZeroDimension);
        }
        Ok(Self {
            real: vec![0.0; dim],
            imag: vec![0.0; dim],
            spans_written: 0,
        })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.real.len()
    }

    #[inline]
    pub fn spans_written(&self) -> usize {
        self.spans_written
    }

    pub fn clear(&mut self) {
        self.real.fill(0.0);
        self.imag.fill(0.0);
        self.spans_written = 0;
    }

    /// Add one closed validity span for `key -> value`.
    ///
    /// The association role is `key * value`. Because both operands are bipolar
    /// unitary roles, cleanup with the same key and candidate value is an exact
    /// sign unbinding before temporal correlation.
    ///
    /// `start` must be a valid checkpoint and `end_exclusive` may be at most
    /// [`EXACT_CAUSAL_CHECKPOINT_LIMIT`], so every represented checkpoint center
    /// remains an exactly representable half-integer before phase evaluation.
    pub fn write_span(
        &mut self,
        axis: &TemporalAxis,
        key: &UnitaryRole,
        value: &UnitaryRole,
        start: u64,
        end_exclusive: u64,
    ) -> Result<(), ValidityMemoryError> {
        self.check_axis_and_role(axis, key)?;
        self.check_role(value)?;
        validate_checkpoint_interval(start, end_exclusive)?;

        let association = key.compose(value);
        let count = end_exclusive - start;
        // Both boundaries are exact integers in the declared domain; their sum
        // is at most 2^53 and therefore still exactly representable in binary64.
        let midpoint = 0.5 * (start as f64 + end_exclusive as f64);

        for (i, &omega) in axis.frequencies().iter().enumerate() {
            let amplitude = dirichlet_amplitude(count, omega);
            let phase = omega * midpoint;
            let (sin, cos) = phase.sin_cos();
            let sign = association.as_slice()[i] as f64;
            self.real[i] += sign * amplitude * cos;
            self.imag[i] += sign * amplitude * sin;
        }

        self.spans_written += 1;
        Ok(())
    }

    /// Score one candidate value at a discrete checkpoint.
    ///
    /// Checkpoint `n` is represented at `n + 1/2`, safely inside its half-open
    /// validity cell. The score is the mean real correlation after sign-unbinding
    /// the key/value association. `n` must satisfy `n < 2^52`.
    pub fn score_candidate(
        &self,
        axis: &TemporalAxis,
        key: &UnitaryRole,
        value: &UnitaryRole,
        checkpoint: u64,
    ) -> Result<f64, ValidityMemoryError> {
        self.check_axis_and_role(axis, key)?;
        self.check_role(value)?;
        validate_checkpoint(checkpoint)?;
        let point = axis.at(checkpoint as f64 + 0.5)?;
        self.score_candidate_at_point(key, value, &point)
    }

    /// Cleanup against a fixed candidate codebook and report the winning margin.
    ///
    /// The temporal query phasor is constructed exactly once per cleanup. Candidate
    /// competition then reuses that point, avoiding repeated trigonometric work and
    /// making query cost one `O(D)` temporal-role construction plus `O(CD)` cleanup.
    pub fn cleanup(
        &self,
        axis: &TemporalAxis,
        key: &UnitaryRole,
        candidates: &[UnitaryRole],
        checkpoint: u64,
    ) -> Result<ValidityCleanupResult, ValidityMemoryError> {
        if candidates.len() < 2 {
            return Err(ValidityMemoryError::TooFewCandidates {
                count: candidates.len(),
            });
        }
        self.check_axis_and_role(axis, key)?;
        for candidate in candidates {
            self.check_role(candidate)?;
        }
        validate_checkpoint(checkpoint)?;
        let point = axis.at(checkpoint as f64 + 0.5)?;

        let mut best_index = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        let mut second_score = f64::NEG_INFINITY;
        for (index, candidate) in candidates.iter().enumerate() {
            let score = self.score_candidate_at_point(key, candidate, &point)?;
            if score > best_score {
                second_score = best_score;
                best_score = score;
                best_index = index;
            } else if score > second_score {
                second_score = score;
            }
        }

        Ok(ValidityCleanupResult {
            best_index,
            best_score,
            second_score,
            margin: best_score - second_score,
        })
    }

    fn score_candidate_at_point(
        &self,
        key: &UnitaryRole,
        value: &UnitaryRole,
        point: &TemporalPhasor,
    ) -> Result<f64, ValidityMemoryError> {
        self.check_role(key)?;
        self.check_role(value)?;
        if point.dim() != self.dim() {
            return Err(ValidityMemoryError::DimensionMismatch {
                expected: self.dim(),
                actual: point.dim(),
            });
        }

        let mut sum = 0.0_f64;
        for i in 0..self.dim() {
            let sign = (key.as_slice()[i] * value.as_slice()[i]) as f64;
            sum += sign * (point.real()[i] * self.real[i] + point.imag()[i] * self.imag[i]);
        }
        let score = sum / self.dim() as f64;
        if score.is_finite() {
            Ok(score)
        } else {
            Err(ValidityMemoryError::NonFiniteScore)
        }
    }

    fn check_axis_and_role(
        &self,
        axis: &TemporalAxis,
        role: &UnitaryRole,
    ) -> Result<(), ValidityMemoryError> {
        if axis.dim() != self.dim() {
            return Err(ValidityMemoryError::DimensionMismatch {
                expected: self.dim(),
                actual: axis.dim(),
            });
        }
        self.check_role(role)
    }

    fn check_role(&self, role: &UnitaryRole) -> Result<(), ValidityMemoryError> {
        if role.dim() == self.dim() {
            Ok(())
        } else {
            Err(ValidityMemoryError::DimensionMismatch {
                expected: self.dim(),
                actual: role.dim(),
            })
        }
    }
}

#[inline]
fn validate_checkpoint(checkpoint: u64) -> Result<(), ValidityMemoryError> {
    if checkpoint < EXACT_CAUSAL_CHECKPOINT_LIMIT {
        Ok(())
    } else {
        Err(ValidityMemoryError::CheckpointOutOfExactRange {
            checkpoint,
            limit_exclusive: EXACT_CAUSAL_CHECKPOINT_LIMIT,
        })
    }
}

fn validate_checkpoint_interval(start: u64, end_exclusive: u64) -> Result<(), ValidityMemoryError> {
    if start >= end_exclusive {
        return Err(ValidityMemoryError::InvalidCheckpointInterval {
            start,
            end_exclusive,
        });
    }
    if start >= EXACT_CAUSAL_CHECKPOINT_LIMIT || end_exclusive > EXACT_CAUSAL_CHECKPOINT_LIMIT {
        return Err(ValidityMemoryError::CheckpointIntervalOutOfExactRange {
            start,
            end_exclusive,
            limit_exclusive: EXACT_CAUSAL_CHECKPOINT_LIMIT,
        });
    }
    Ok(())
}

/// Stable `sin(x) / x` with an even Taylor series around zero.
///
/// The x^6 truncation is far below binary64 rounding error at the selected
/// threshold. Outside the threshold the direct quotient is well conditioned.
#[inline]
fn stable_sinc(value: f64) -> f64 {
    if value.abs() <= SINC_TAYLOR_THRESHOLD {
        let x2 = value * value;
        1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 - x2 / 5040.0))
    } else {
        value.sin() / value
    }
}

/// Dirichlet amplitude for `count` consecutive checkpoint centers.
///
/// Algebraically this is `sin(count*x) / sin(x)` for `x = omega / 2`, but the
/// sinc ratio avoids the invalid denominator-only near-zero shortcut: both the
/// numerator and denominator receive their correct small-angle treatment.
#[inline]
fn dirichlet_amplitude(count: u64, omega: f64) -> f64 {
    let count_f64 = count as f64;
    let half_omega = 0.5 * omega;
    count_f64 * stable_sinc(count_f64 * half_omega) / stable_sinc(half_omega)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPAN_SUM_TOL: f64 = 1e-6;

    fn roles(count: usize, dim: usize, seed: u64) -> Vec<UnitaryRole> {
        (0..count)
            .map(|index| UnitaryRole::new(dim, seed + index as u64))
            .collect()
    }

    fn explicit_span_components(
        axis: &TemporalAxis,
        key: &UnitaryRole,
        value: &UnitaryRole,
        start: u64,
        end_exclusive: u64,
    ) -> (Vec<f64>, Vec<f64>) {
        let association = key.compose(value);
        let mut real = vec![0.0; axis.dim()];
        let mut imag = vec![0.0; axis.dim()];
        for checkpoint in start..end_exclusive {
            let time = checkpoint as f64 + 0.5;
            for (index, &omega) in axis.frequencies().iter().enumerate() {
                let phase = omega * time;
                let (sin, cos) = phase.sin_cos();
                let sign = association.as_slice()[index] as f64;
                real[index] += sign * cos;
                imag[index] += sign * sin;
            }
        }
        (real, imag)
    }

    fn max_component_error(
        memory: &ValidityIntervalMemory,
        expected_real: &[f64],
        expected_imag: &[f64],
    ) -> f64 {
        memory
            .real
            .iter()
            .zip(&memory.imag)
            .zip(expected_real.iter().zip(expected_imag))
            .flat_map(|((&actual_real, &actual_imag), (&expected_real, &expected_imag))| {
                [
                    (actual_real - expected_real).abs(),
                    (actual_imag - expected_imag).abs(),
                ]
            })
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn single_key_recovers_predecessor_value_at_every_checkpoint() {
        let dim = 4096;
        let axis = TemporalAxis::new(dim, 1).unwrap();
        let key = UnitaryRole::new(dim, 2);
        let values = roles(3, dim, 10);
        let mut memory = ValidityIntervalMemory::new(dim).unwrap();
        memory.write_span(&axis, &key, &values[0], 0, 5).unwrap();
        memory.write_span(&axis, &key, &values[1], 5, 11).unwrap();
        memory.write_span(&axis, &key, &values[2], 11, 20).unwrap();

        for checkpoint in 0..20 {
            let expected = if checkpoint < 5 {
                0
            } else if checkpoint < 11 {
                1
            } else {
                2
            };
            let result = memory.cleanup(&axis, &key, &values, checkpoint).unwrap();
            assert_eq!(result.best_index, expected, "checkpoint={checkpoint}");
            assert!(result.margin > 0.0, "checkpoint={checkpoint}, result={result:?}");
        }
    }

    #[test]
    fn cleanup_and_direct_candidate_scores_are_identical() {
        let dim = 1024;
        let axis = TemporalAxis::new(dim, 11).unwrap();
        let key = UnitaryRole::new(dim, 12);
        let values = roles(4, dim, 20);
        let mut memory = ValidityIntervalMemory::new(dim).unwrap();
        memory.write_span(&axis, &key, &values[0], 0, 7).unwrap();
        memory.write_span(&axis, &key, &values[1], 7, 15).unwrap();

        let checkpoint = 9;
        let direct = values
            .iter()
            .map(|value| memory.score_candidate(&axis, &key, value, checkpoint).unwrap())
            .collect::<Vec<_>>();
        let cleanup = memory.cleanup(&axis, &key, &values, checkpoint).unwrap();
        let mut ranked = direct.clone();
        ranked.sort_by(|a, b| b.total_cmp(a));
        let expected_best = direct
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap();
        assert_eq!(cleanup.best_index, expected_best);
        assert_eq!(cleanup.best_score.to_bits(), ranked[0].to_bits());
        assert_eq!(cleanup.second_score.to_bits(), ranked[1].to_bits());
    }

    #[test]
    fn sinc_taylor_matches_direct_formula_across_threshold() {
        for multiplier in [-1.001_f64, -1.0, -0.999, 0.0, 0.999, 1.0, 1.001] {
            let value = SINC_TAYLOR_THRESHOLD * multiplier;
            let expected = if value == 0.0 {
                1.0
            } else {
                value.sin() / value
            };
            let error = (stable_sinc(value) - expected).abs();
            assert!(error <= 2.0e-15, "value={value:e}, error={error:e}");
        }
    }

    #[test]
    fn tiny_frequency_large_span_does_not_collapse_to_count() {
        let count = 1_000_000_000_000_u64;
        let omega = 1.0e-13_f64;
        let half_omega = 0.5 * omega;
        let expected = (count as f64 * half_omega).sin() / half_omega.sin();
        let actual = dirichlet_amplitude(count, omega);
        let relative_error = ((actual - expected) / expected).abs();
        assert!(relative_error < 2.0e-15, "relative_error={relative_error:e}");
        assert!(
            (actual - count as f64).abs() > 1.0e8,
            "large-span tiny-frequency amplitude was incorrectly collapsed to count"
        );
    }

    #[test]
    fn analytic_spans_match_independent_direct_sums_across_log_offsets() {
        let dim = 512;
        let axis = TemporalAxis::new(dim, 15).unwrap();
        let key = UnitaryRole::new(dim, 16);
        let value = UnitaryRole::new(dim, 17);
        let offsets = [0_u64, 10, 1_000, 100_000, 1_000_000];
        let lengths = [1_u64, 7, 31, 257];
        let mut global_max_error = 0.0_f64;

        for start in offsets {
            for length in lengths {
                let end = start + length;
                let mut analytic = ValidityIntervalMemory::new(dim).unwrap();
                analytic.write_span(&axis, &key, &value, start, end).unwrap();
                let (expected_real, expected_imag) =
                    explicit_span_components(&axis, &key, &value, start, end);
                let error = max_component_error(&analytic, &expected_real, &expected_imag);
                global_max_error = global_max_error.max(error);
                assert!(
                    error < SPAN_SUM_TOL,
                    "analytic/direct-sum divergence at start={start}, length={length}: {error:e}"
                );
                assert_eq!(analytic.spans_written(), 1);
            }
        }

        eprintln!("max analytic/direct-sum component error={global_max_error:.17e}");
    }

    #[test]
    fn exact_causal_coordinate_boundary_is_fail_closed() {
        let dim = 64;
        let axis = TemporalAxis::new(dim, 18).unwrap();
        let key = UnitaryRole::new(dim, 19);
        let values = roles(2, dim, 20);
        let mut memory = ValidityIntervalMemory::new(dim).unwrap();

        memory
            .write_span(
                &axis,
                &key,
                &values[0],
                EXACT_CAUSAL_CHECKPOINT_LIMIT - 2,
                EXACT_CAUSAL_CHECKPOINT_LIMIT,
            )
            .unwrap();
        assert!(
            memory
                .score_candidate(
                    &axis,
                    &key,
                    &values[0],
                    EXACT_CAUSAL_CHECKPOINT_LIMIT - 1,
                )
                .unwrap()
                .is_finite()
        );
        assert!(matches!(
            memory.score_candidate(
                &axis,
                &key,
                &values[0],
                EXACT_CAUSAL_CHECKPOINT_LIMIT,
            ),
            Err(ValidityMemoryError::CheckpointOutOfExactRange { .. })
        ));
        assert!(matches!(
            memory.cleanup(
                &axis,
                &key,
                &values,
                EXACT_CAUSAL_CHECKPOINT_LIMIT,
            ),
            Err(ValidityMemoryError::CheckpointOutOfExactRange { .. })
        ));
        assert!(matches!(
            memory.write_span(
                &axis,
                &key,
                &values[0],
                EXACT_CAUSAL_CHECKPOINT_LIMIT - 1,
                EXACT_CAUSAL_CHECKPOINT_LIMIT + 1,
            ),
            Err(ValidityMemoryError::CheckpointIntervalOutOfExactRange { .. })
        ));
        assert!(matches!(
            memory.write_span(
                &axis,
                &key,
                &values[0],
                EXACT_CAUSAL_CHECKPOINT_LIMIT,
                EXACT_CAUSAL_CHECKPOINT_LIMIT + 1,
            ),
            Err(ValidityMemoryError::CheckpointIntervalOutOfExactRange { .. })
        ));
    }

    #[test]
    fn independent_keys_do_not_share_history() {
        let dim = 4096;
        let axis = TemporalAxis::new(dim, 20).unwrap();
        let keys = roles(2, dim, 30);
        let values = roles(3, dim, 40);
        let mut memory = ValidityIntervalMemory::new(dim).unwrap();

        memory.write_span(&axis, &keys[0], &values[0], 0, 8).unwrap();
        memory.write_span(&axis, &keys[0], &values[1], 8, 16).unwrap();
        memory.write_span(&axis, &keys[1], &values[2], 0, 6).unwrap();
        memory.write_span(&axis, &keys[1], &values[0], 6, 16).unwrap();

        for checkpoint in 0..16 {
            let first_expected = if checkpoint < 8 { 0 } else { 1 };
            let second_expected = if checkpoint < 6 { 2 } else { 0 };
            assert_eq!(
                memory.cleanup(&axis, &keys[0], &values, checkpoint).unwrap().best_index,
                first_expected
            );
            assert_eq!(
                memory.cleanup(&axis, &keys[1], &values, checkpoint).unwrap().best_index,
                second_expected
            );
        }
    }

    #[test]
    fn moderate_superposition_retains_positive_cleanup_margin() {
        let dim = 8192;
        let axis = TemporalAxis::new(dim, 100).unwrap();
        let keys = roles(8, dim, 200);
        let values = roles(4, dim, 400);
        let mut memory = ValidityIntervalMemory::new(dim).unwrap();

        // Eight independent subjects, four successive values each, four
        // checkpoints per validity span: 32 closed spans / 128 checkpoint facts.
        for (key_index, key) in keys.iter().enumerate() {
            for span in 0..4u64 {
                let value_index = (key_index + span as usize) % values.len();
                memory
                    .write_span(&axis, key, &values[value_index], span * 4, (span + 1) * 4)
                    .unwrap();
            }
        }

        let mut smallest_margin = f64::INFINITY;
        for (key_index, key) in keys.iter().enumerate() {
            for checkpoint in 0..16u64 {
                let expected = (key_index + (checkpoint / 4) as usize) % values.len();
                let result = memory.cleanup(&axis, key, &values, checkpoint).unwrap();
                assert_eq!(result.best_index, expected, "key={key_index}, checkpoint={checkpoint}");
                smallest_margin = smallest_margin.min(result.margin);
            }
        }
        assert!(smallest_margin > 0.0, "smallest cleanup margin={smallest_margin}");
        assert_eq!(memory.spans_written(), 32);
    }

    #[test]
    fn invalid_spans_and_candidate_sets_fail_closed() {
        let dim = 64;
        let axis = TemporalAxis::new(dim, 500).unwrap();
        let key = UnitaryRole::new(dim, 501);
        let value = UnitaryRole::new(dim, 502);
        let mut memory = ValidityIntervalMemory::new(dim).unwrap();

        assert!(matches!(
            memory.write_span(&axis, &key, &value, 4, 4),
            Err(ValidityMemoryError::InvalidCheckpointInterval { .. })
        ));
        assert!(matches!(
            memory.cleanup(&axis, &key, std::slice::from_ref(&value), 0),
            Err(ValidityMemoryError::TooFewCandidates { count: 1 })
        ));
    }
}
