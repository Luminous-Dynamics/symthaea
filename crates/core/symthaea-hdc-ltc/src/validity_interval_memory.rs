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

use crate::continuous_hv::UnitaryRole;
use crate::temporal_phasor::{TemporalAlgebraError, TemporalAxis, TemporalPhasor};
use std::fmt;

const SERIES_EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub enum ValidityMemoryError {
    ZeroDimension,
    DimensionMismatch { expected: usize, actual: usize },
    InvalidCheckpointInterval { start: u64, end_exclusive: u64 },
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
        if start >= end_exclusive {
            return Err(ValidityMemoryError::InvalidCheckpointInterval {
                start,
                end_exclusive,
            });
        }

        let association = key.compose(value);
        let count = end_exclusive - start;
        let midpoint = start as f64 + count as f64 / 2.0;

        for (i, &omega) in axis.frequencies().iter().enumerate() {
            let half_omega = 0.5 * omega;
            let denominator = half_omega.sin();
            let amplitude = if denominator.abs() <= SERIES_EPSILON {
                count as f64
            } else {
                (0.5 * count as f64 * omega).sin() / denominator
            };
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
    /// the key/value association.
    pub fn score_candidate(
        &self,
        axis: &TemporalAxis,
        key: &UnitaryRole,
        value: &UnitaryRole,
        checkpoint: u64,
    ) -> Result<f64, ValidityMemoryError> {
        self.check_axis_and_role(axis, key)?;
        self.check_role(value)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn roles(count: usize, dim: usize, seed: u64) -> Vec<UnitaryRole> {
        (0..count)
            .map(|index| UnitaryRole::new(dim, seed + index as u64))
            .collect()
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
    fn analytic_span_matches_explicit_checkpoint_sum_at_large_offset() {
        let dim = 1024;
        let axis = TemporalAxis::new(dim, 15).unwrap();
        let key = UnitaryRole::new(dim, 16);
        let value = UnitaryRole::new(dim, 17);
        let start = 1_000_000u64;
        let end = start + 257;

        let mut analytic = ValidityIntervalMemory::new(dim).unwrap();
        analytic.write_span(&axis, &key, &value, start, end).unwrap();

        let mut explicit = ValidityIntervalMemory::new(dim).unwrap();
        for checkpoint in start..end {
            explicit
                .write_span(&axis, &key, &value, checkpoint, checkpoint + 1)
                .unwrap();
        }

        let max_abs_error = analytic
            .real
            .iter()
            .zip(&analytic.imag)
            .zip(explicit.real.iter().zip(&explicit.imag))
            .flat_map(|((&analytic_real, &analytic_imag), (&explicit_real, &explicit_imag))| {
                [
                    (analytic_real - explicit_real).abs(),
                    (analytic_imag - explicit_imag).abs(),
                ]
            })
            .fold(0.0_f64, f64::max);

        assert!(
            max_abs_error < 1e-6,
            "analytic Dirichlet span diverged from explicit checkpoint sum: {max_abs_error}"
        );
        assert_eq!(analytic.spans_written(), 1);
        assert_eq!(explicit.spans_written(), 257);
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
