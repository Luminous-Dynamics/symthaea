use std::fmt;
use symthaea_hdc_ltc::{TemporalAxis, UnitaryRole};

pub const ORACLE_EXACT_CAUSAL_CHECKPOINT_LIMIT: u64 = 1_u64 << 52;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OracleSpan {
    pub key_index: usize,
    pub value_index: usize,
    pub start: u64,
    pub end_exclusive: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OracleError {
    EmptyKeys,
    EmptyValues,
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    KeyIndexOutOfRange {
        index: usize,
        count: usize,
    },
    ValueIndexOutOfRange {
        index: usize,
        count: usize,
    },
    InvalidSpan {
        start: u64,
        end_exclusive: u64,
    },
    CheckpointOutOfRange {
        checkpoint: u64,
    },
    SpanOutOfRange {
        start: u64,
        end_exclusive: u64,
    },
}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyKeys => write!(f, "direct-sum oracle requires at least one key"),
            Self::EmptyValues => write!(f, "direct-sum oracle requires at least one candidate value"),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "direct-sum oracle dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::KeyIndexOutOfRange { index, count } => {
                write!(f, "direct-sum oracle key index {index} is outside 0..{count}")
            }
            Self::ValueIndexOutOfRange { index, count } => write!(
                f,
                "direct-sum oracle value index {index} is outside 0..{count}"
            ),
            Self::InvalidSpan {
                start,
                end_exclusive,
            } => write!(
                f,
                "direct-sum oracle span must satisfy start < end_exclusive, got [{start}, {end_exclusive})"
            ),
            Self::CheckpointOutOfRange { checkpoint } => write!(
                f,
                "direct-sum oracle checkpoint {checkpoint} is outside 0..{ORACLE_EXACT_CAUSAL_CHECKPOINT_LIMIT}"
            ),
            Self::SpanOutOfRange {
                start,
                end_exclusive,
            } => write!(
                f,
                "direct-sum oracle span [{start}, {end_exclusive}) exceeds exact causal coordinates"
            ),
        }
    }
}

impl std::error::Error for OracleError {}

/// Independent explicit checkpoint-sum oracle for the validity-memory score.
///
/// This support module deliberately does not instantiate the production validity
/// archive and does not call its write, cleanup, or candidate-score methods.
/// It also deliberately avoids production's large-argument `sin_cos(omega * t)`
/// phase path. For checkpoint `n`, it constructs
///
/// `T(n + 1/2) = exp(i*omega/2) * exp(i*omega)^n`
///
/// using small-angle trigonometric primitives plus integer complex
/// exponentiation by squaring. Large-offset comparison therefore exercises a
/// different phase-evaluation path as well as a different archive construction.
///
/// The returned vector contains one score for every supplied candidate value.
pub fn explicit_candidate_scores(
    axis: &TemporalAxis,
    keys: &[UnitaryRole],
    values: &[UnitaryRole],
    spans: &[OracleSpan],
    query_key_index: usize,
    checkpoint: u64,
) -> Result<Vec<f64>, OracleError> {
    validate_inputs(axis, keys, values, spans, query_key_index, checkpoint)?;

    let mut archive_real = vec![0.0_f64; axis.dim()];
    let mut archive_imag = vec![0.0_f64; axis.dim()];

    for span in spans {
        let key = &keys[span.key_index];
        let value = &values[span.value_index];
        for represented_checkpoint in span.start..span.end_exclusive {
            for (channel, &omega) in axis.frequencies().iter().enumerate() {
                let (point_real, point_imag) = checkpoint_phasor(omega, represented_checkpoint);
                let sign = (key.as_slice()[channel] * value.as_slice()[channel]) as f64;
                archive_real[channel] += sign * point_real;
                archive_imag[channel] += sign * point_imag;
            }
        }
    }

    let query_key = &keys[query_key_index];
    let mut scores = vec![0.0_f64; values.len()];

    for (candidate_index, candidate) in values.iter().enumerate() {
        let mut sum = 0.0_f64;
        for (channel, &omega) in axis.frequencies().iter().enumerate() {
            let (query_real, query_imag) = checkpoint_phasor(omega, checkpoint);
            let sign =
                (query_key.as_slice()[channel] * candidate.as_slice()[channel]) as f64;
            sum += sign
                * (query_real * archive_real[channel] + query_imag * archive_imag[channel]);
        }
        scores[candidate_index] = sum / axis.dim() as f64;
    }

    Ok(scores)
}

/// Construct `exp(i * omega * (checkpoint + 1/2))` without evaluating a
/// trigonometric function at a large absolute phase.
fn checkpoint_phasor(omega: f64, checkpoint: u64) -> (f64, f64) {
    let (step_imag, step_real) = omega.sin_cos();
    let (half_imag, half_real) = (0.5 * omega).sin_cos();
    let integer = unit_complex_pow((step_real, step_imag), checkpoint);
    normalize_unit(complex_mul((half_real, half_imag), integer))
}

/// Integer power of a unit complex number using exponentiation by squaring.
///
/// Renormalization keeps rounding drift from growing with the bit length of a
/// large checkpoint while preserving the intended unit-circle group element.
fn unit_complex_pow(mut base: (f64, f64), mut exponent: u64) -> (f64, f64) {
    let mut result = (1.0_f64, 0.0_f64);
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = normalize_unit(complex_mul(result, base));
        }
        exponent >>= 1;
        if exponent != 0 {
            base = normalize_unit(complex_mul(base, base));
        }
    }
    result
}

#[inline]
fn complex_mul(left: (f64, f64), right: (f64, f64)) -> (f64, f64) {
    (
        left.0 * right.0 - left.1 * right.1,
        left.0 * right.1 + left.1 * right.0,
    )
}

#[inline]
fn normalize_unit(value: (f64, f64)) -> (f64, f64) {
    let norm = value.0.hypot(value.1);
    (value.0 / norm, value.1 / norm)
}

fn validate_inputs(
    axis: &TemporalAxis,
    keys: &[UnitaryRole],
    values: &[UnitaryRole],
    spans: &[OracleSpan],
    query_key_index: usize,
    checkpoint: u64,
) -> Result<(), OracleError> {
    if keys.is_empty() {
        return Err(OracleError::EmptyKeys);
    }
    if values.is_empty() {
        return Err(OracleError::EmptyValues);
    }
    if query_key_index >= keys.len() {
        return Err(OracleError::KeyIndexOutOfRange {
            index: query_key_index,
            count: keys.len(),
        });
    }
    if checkpoint >= ORACLE_EXACT_CAUSAL_CHECKPOINT_LIMIT {
        return Err(OracleError::CheckpointOutOfRange { checkpoint });
    }

    for role in keys.iter().chain(values) {
        if role.dim() != axis.dim() {
            return Err(OracleError::DimensionMismatch {
                expected: axis.dim(),
                actual: role.dim(),
            });
        }
    }

    for span in spans {
        if span.start >= span.end_exclusive {
            return Err(OracleError::InvalidSpan {
                start: span.start,
                end_exclusive: span.end_exclusive,
            });
        }
        if span.start >= ORACLE_EXACT_CAUSAL_CHECKPOINT_LIMIT
            || span.end_exclusive > ORACLE_EXACT_CAUSAL_CHECKPOINT_LIMIT
        {
            return Err(OracleError::SpanOutOfRange {
                start: span.start,
                end_exclusive: span.end_exclusive,
            });
        }
        if span.key_index >= keys.len() {
            return Err(OracleError::KeyIndexOutOfRange {
                index: span.key_index,
                count: keys.len(),
            });
        }
        if span.value_index >= values.len() {
            return Err(OracleError::ValueIndexOutOfRange {
                index: span.value_index,
                count: values.len(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_phase_path_matches_direct_formula_at_small_offsets() {
        for omega in [0.0, 1.0e-13, 0.125, -0.5, 1.0, -2.75] {
            for checkpoint in [0_u64, 1, 2, 7, 31, 127] {
                let (actual_real, actual_imag) = checkpoint_phasor(omega, checkpoint);
                let phase = omega * (checkpoint as f64 + 0.5);
                let (expected_imag, expected_real) = phase.sin_cos();
                assert!((actual_real - expected_real).abs() < 1.0e-13);
                assert!((actual_imag - expected_imag).abs() < 1.0e-13);
            }
        }
    }

    #[test]
    fn integer_phase_path_preserves_unit_modulus_at_large_offsets() {
        for omega in [1.0e-13, 0.125, -0.5, 1.0, -2.75] {
            for checkpoint in [1_000_000_u64, 1_000_000_000, 1_000_000_000_000] {
                let (real, imag) = checkpoint_phasor(omega, checkpoint);
                assert!((real.hypot(imag) - 1.0).abs() < 1.0e-14);
            }
        }
    }
}
