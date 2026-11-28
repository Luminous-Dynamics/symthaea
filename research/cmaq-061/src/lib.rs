#![forbid(unsafe_code)]

//! Bounded, previsible unary encoders for sequential conditional paired tests.
//!
//! A pair is always transformed by one immutable encoder version. Any update
//! derived from that pair is staged as the *next* version and cannot be used
//! until the evidence transaction for the current pair has committed.

use sha2::{Digest, Sha256};
use std::fmt;

const DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-061/PREVISIBLE-ENCODER/V1\0";

#[derive(Clone, Debug, PartialEq)]
pub enum EncoderError {
    InvalidDimensions,
    NonFiniteParameter,
    NonFiniteInput,
    InvalidNormBound,
    InvalidLearningRate,
    StaleTransaction,
    AlreadyCommitted,
    EvidenceCommitRejected,
}
impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for EncoderError {}

#[derive(Clone, Debug, PartialEq)]
pub struct EncodedPair {
    pub version: u64,
    pub encoder_digest: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub difference: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct PreparedEncoderUpdate {
    base_version: u64,
    base_digest: String,
    next_weights: Vec<Vec<f64>>,
    next_digest: String,
    pair_sequence: u64,
}

#[derive(Clone, Debug)]
pub struct PredictableEncoder {
    weights: Vec<Vec<f64>>,
    norm_bound: f64,
    version: u64,
    last_sequence: Option<u64>,
    digest: String,
}

impl PredictableEncoder {
    /// Construct a frozen unary map. Each row is one output coordinate.
    /// The output is deterministically clipped to the closed L2 ball.
    pub fn new(weights: Vec<Vec<f64>>, norm_bound: f64) -> Result<Self, EncoderError> {
        if weights.is_empty() || weights[0].is_empty()
            || weights.iter().any(|row| row.len() != weights[0].len())
        {
            return Err(EncoderError::InvalidDimensions);
        }
        if !norm_bound.is_finite() || norm_bound <= 0.0 {
            return Err(EncoderError::InvalidNormBound);
        }
        if weights.iter().flatten().any(|v| !v.is_finite()) {
            return Err(EncoderError::NonFiniteParameter);
        }
        let digest = digest_weights(&weights, norm_bound);
        Ok(Self { weights, norm_bound, version: 0, last_sequence: None, digest })
    }

    pub fn version(&self) -> u64 { self.version }
    pub fn digest(&self) -> &str { &self.digest }
    pub fn input_dimension(&self) -> usize { self.weights[0].len() }
    pub fn output_dimension(&self) -> usize { self.weights.len() }

    /// Apply the same frozen unary map to both members of the current pair.
    /// No state is mutated by this operation.
    pub fn encode_pair(
        &self,
        sequence: u64,
        x: &[f64],
        y: &[f64],
    ) -> Result<EncodedPair, EncoderError> {
        if x.len() != self.input_dimension() || y.len() != self.input_dimension() {
            return Err(EncoderError::InvalidDimensions);
        }
        if x.iter().chain(y).any(|v| !v.is_finite()) {
            return Err(EncoderError::NonFiniteInput);
        }
        if self.last_sequence.is_some_and(|last| sequence <= last) {
            return Err(EncoderError::StaleTransaction);
        }
        let encoded_x = self.encode_one(x);
        let encoded_y = self.encode_one(y);
        let difference = encoded_y.iter().zip(&encoded_x).map(|(a, b)| a - b).collect();
        Ok(EncodedPair {
            version: self.version,
            encoder_digest: self.digest.clone(),
            x: encoded_x,
            y: encoded_y,
            difference,
        })
    }

    /// Stage a symmetric, bounded update for the *next* encoder version.
    /// The update uses only the unordered pair mean (x+y)/2. It cannot alter
    /// the already-returned EncodedPair or the current evidence factor.
    pub fn prepare_next_version(
        &self,
        sequence: u64,
        x: &[f64],
        y: &[f64],
        learning_rate: f64,
        max_parameter_step: f64,
    ) -> Result<PreparedEncoderUpdate, EncoderError> {
        if x.len() != self.input_dimension() || y.len() != self.input_dimension() {
            return Err(EncoderError::InvalidDimensions);
        }
        if x.iter().chain(y).any(|v| !v.is_finite()) {
            return Err(EncoderError::NonFiniteInput);
        }
        if !learning_rate.is_finite() || !(0.0..=1.0).contains(&learning_rate) {
            return Err(EncoderError::InvalidLearningRate);
        }
        if !max_parameter_step.is_finite() || max_parameter_step <= 0.0 {
            return Err(EncoderError::InvalidNormBound);
        }
        if self.last_sequence.is_some_and(|last| sequence <= last) {
            return Err(EncoderError::StaleTransaction);
        }

        let mut next = self.weights.clone();
        let mean = x.iter().zip(y).map(|(a, b)| (a + b) * 0.5).collect::<Vec<_>>();
        for (row_index, row) in next.iter_mut().enumerate() {
            let activation = dot(&self.weights[row_index], &mean).tanh();
            for (weight, input) in row.iter_mut().zip(&mean) {
                let raw_step = learning_rate * activation * input;
                let step = raw_step.clamp(-max_parameter_step, max_parameter_step);
                *weight += step;
            }
        }
        let next_digest = digest_weights(&next, self.norm_bound);
        Ok(PreparedEncoderUpdate {
            base_version: self.version,
            base_digest: self.digest.clone(),
            next_weights: next,
            next_digest,
            pair_sequence: sequence,
        })
    }

    /// Commit the parent evidence transaction first, then advance the encoder.
    /// The callback is invoked only after the prepared update is validated; a
    /// rejected evidence commit leaves the encoder version unchanged.
    pub fn commit_after_evidence<F>(
        &mut self,
        prepared: PreparedEncoderUpdate,
        commit_evidence: F,
    ) -> Result<(), EncoderError>
    where
        F: FnOnce(u64, &str) -> bool,
    {
        if prepared.base_version != self.version || prepared.base_digest != self.digest
            || self.last_sequence.is_some_and(|last| prepared.pair_sequence <= last)
        {
            return Err(EncoderError::StaleTransaction);
        }
        let next_version = self.version.checked_add(1).ok_or(EncoderError::StaleTransaction)?;
        if !commit_evidence(self.version, &self.digest) {
            return Err(EncoderError::EvidenceCommitRejected);
        }
        self.weights = prepared.next_weights;
        self.digest = prepared.next_digest;
        self.version = next_version;
        self.last_sequence = Some(prepared.pair_sequence);
        Ok(())
    }

    fn encode_one(&self, input: &[f64]) -> Vec<f64> {
        let mut output = self.weights.iter()
            .map(|row| dot(row, input).tanh())
            .collect::<Vec<_>>();
        let norm = output.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > self.norm_bound {
            let scale = self.norm_bound / norm;
            for value in &mut output { *value *= scale; }
        }
        output
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn digest_weights(weights: &[Vec<f64>], norm_bound: f64) -> String {
    let mut h = Sha256::new();
    h.update(DOMAIN);
    h.update((weights.len() as u64).to_be_bytes());
    h.update((weights[0].len() as u64).to_be_bytes());
    h.update(canonical(norm_bound).to_be_bytes());
    for row in weights {
        for value in row {
            h.update(canonical(*value).to_be_bytes());
        }
    }
    format!("{:x}", h.finalize())
}

fn canonical(value: f64) -> u64 {
    if value == 0.0 { 0.0f64.to_bits() } else { value.to_bits() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder() -> PredictableEncoder {
        PredictableEncoder::new(vec![vec![0.5, -0.25], vec![0.1, 0.8]], 1.0).unwrap()
    }

    #[test]
    fn same_unary_map_is_applied_to_both_pair_members() {
        let encoder = encoder();
        let pair = encoder.encode_pair(1, &[0.2, 0.4], &[0.7, -0.1]).unwrap();
        let x_again = encoder.encode_pair(1, &[0.2, 0.4], &[0.2, 0.4]).unwrap();
        assert_eq!(pair.x, x_again.x);
        assert_eq!(pair.version, 0);
        assert_eq!(pair.encoder_digest, encoder.digest());
    }

    #[test]
    fn output_is_inside_configured_ball() {
        let encoder = encoder();
        let pair = encoder.encode_pair(1, &[1e6, -1e6], &[-1e6, 1e6]).unwrap();
        for output in [&pair.x, &pair.y] {
            let norm = output.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(norm <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn staging_does_not_change_current_version_or_digest() {
        let encoder = encoder();
        let version = encoder.version();
        let digest = encoder.digest().to_string();
        let staged = encoder.prepare_next_version(1, &[0.2, 0.4], &[0.7, -0.1], 0.01, 0.001).unwrap();
        assert_eq!(encoder.version(), version);
        assert_eq!(encoder.digest(), digest);
        assert_ne!(staged.next_digest, digest);
    }

    #[test]
    fn next_version_update_is_pair_order_invariant() {
        let encoder = encoder();
        let forward = encoder.prepare_next_version(
            1, &[0.2, 0.4], &[0.7, -0.1], 0.01, 0.001
        ).unwrap();
        let reverse = encoder.prepare_next_version(
            1, &[0.7, -0.1], &[0.2, 0.4], 0.01, 0.001
        ).unwrap();
        assert_eq!(forward.next_digest, reverse.next_digest);
    }

    #[test]
    fn pair_difference_changes_sign_when_pair_order_is_swapped() {
        let encoder = encoder();
        let forward = encoder.encode_pair(1, &[0.2, 0.4], &[0.7, -0.1]).unwrap();
        let reverse = encoder.encode_pair(1, &[0.7, -0.1], &[0.2, 0.4]).unwrap();
        for (a, b) in forward.difference.iter().zip(&reverse.difference) {
            assert!((a + b).abs() < 1e-12);
        }
    }

    #[test]
    fn only_commit_advances_version_and_rejects_replay() {
        let mut encoder = encoder();
        let staged = encoder.prepare_next_version(1, &[0.2, 0.4], &[0.7, -0.1], 0.01, 0.001).unwrap();
        encoder.commit_after_evidence(staged.clone(), |_, _| true).unwrap();
        assert_eq!(encoder.version(), 1);
        assert!(matches!(encoder.commit_after_evidence(staged, |_, _| true), Err(EncoderError::StaleTransaction)));
    }

    #[test]
    fn nonfinite_and_invalid_inputs_fail_closed() {
        let encoder = encoder();
        assert!(matches!(encoder.encode_pair(1, &[f64::NAN, 0.0], &[0.0, 0.0]), Err(EncoderError::NonFiniteInput)));
        assert!(matches!(encoder.prepare_next_version(1, &[0.0, 0.0], &[0.0, 0.0], 1.1, 0.01), Err(EncoderError::InvalidLearningRate)));
        assert!(PredictableEncoder::new(vec![vec![1.0], vec![1.0, 2.0]], 1.0).is_err());
    }

    #[test]
    fn rejected_evidence_commit_does_not_advance_encoder() {
        let mut encoder = encoder();
        let digest = encoder.digest().to_string();
        let staged = encoder.prepare_next_version(1, &[0.2, 0.4], &[0.7, -0.1], 0.01, 0.001).unwrap();
        assert!(matches!(
            encoder.commit_after_evidence(staged, |_, _| false),
            Err(EncoderError::EvidenceCommitRejected)
        ));
        assert_eq!(encoder.version(), 0);
        assert_eq!(encoder.digest(), digest);
    }

    #[test]
    fn evidence_callback_observes_old_encoder_version() {
        let mut encoder = encoder();
        let staged = encoder.prepare_next_version(1, &[0.2, 0.4], &[0.7, -0.1], 0.01, 0.001).unwrap();
        let encoder_digest = encoder.digest().to_string();
        encoder.commit_after_evidence(staged, |version, digest| {
            assert_eq!(version, 0);
            assert_eq!(digest, encoder_digest.as_str());
            true
        }).unwrap();
        assert_eq!(encoder.version(), 1);
    }

    #[test]
    fn sequence_must_increase() {
        let mut encoder = encoder();
        let staged = encoder.prepare_next_version(4, &[0.0, 0.0], &[1.0, 1.0], 0.1, 0.01).unwrap();
        encoder.commit_after_evidence(staged, || true).unwrap();
        assert!(matches!(encoder.encode_pair(4, &[0.0, 0.0], &[1.0, 1.0]), Err(EncoderError::StaleTransaction)));
    }
}
