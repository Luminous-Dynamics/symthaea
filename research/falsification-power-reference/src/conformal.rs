//! Finite-reference conformal e-process for simulator-defined nulls.
//!
//! This is a small SLICE-style reference, not a reproduction of the full
//! SLICE method. The caller supplies a positive score fixed predictably
//! before the current observation and fresh reference draws from the
//! candidate null model. Under conditional exchangeability, the normalized
//! observed score has conditional expectation 1/(K+1).

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

const MAX_REFERENCE_DRAWS: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConformalError {
    EmptyIdentifier,
    InvalidAlpha,
    InvalidReferenceCount,
    TooManyReferenceDraws,
    InvalidObservedScore,
    InvalidReferenceScore { index: usize },
    ReferenceCountMismatch { expected: usize, actual: usize },
    PredictionAlreadyPending,
    NoPredictionPending,
    IdentityMismatch,
    DuplicateObservationId,
    DuplicateReferenceBatchId,
    NumericalOverflow,
    ReceiptCandidateMismatch,
    ReceiptReplayMismatch,
}

impl fmt::Display for ConformalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Error for ConformalError {}

#[derive(Debug, Clone, PartialEq)]
struct PendingScoreCommitment {
    observation_id: String,
    decision_digest: String,
    reference_batch_id: String,
    score_id: String,
    reference_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalUpdate {
    pub observations: u64,
    pub log_e_increment: f64,
    pub log_e_value: f64,
    pub running_max_log_e_value: f64,
    pub anytime_p_value: f64,
    pub crossed_alpha: bool,
}

/// Full, versioned record for one simulator-backed null observation.
/// Raw positive scores are retained so an independent replay can reproduce
/// the exact finite-reference update.
#[derive(Debug, Clone, PartialEq)]
pub struct ConformalObservationReceipt {
    pub schema_version: u32,
    pub candidate_id: String,
    pub observation_id: String,
    pub decision_digest: String,
    pub score_id: String,
    pub reference_batch_id: String,
    pub observed_score: f64,
    pub reference_scores: Vec<f64>,
    pub update: ConformalUpdate,
}

impl ConformalObservationReceipt {
    /// Canonical versioned bytes for deterministic replay.
    /// This encoding is not a cryptographic digest or signature.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.falsification.conformal_receipt.v1\0".to_vec();
        bytes.extend_from_slice(&self.schema_version.to_le_bytes());
        append_string(&mut bytes, &self.candidate_id);
        append_string(&mut bytes, &self.observation_id);
        append_string(&mut bytes, &self.decision_digest);
        append_string(&mut bytes, &self.score_id);
        append_string(&mut bytes, &self.reference_batch_id);
        bytes.extend_from_slice(&self.observed_score.to_bits().to_le_bytes());
        bytes.extend_from_slice(&(self.reference_scores.len() as u64).to_le_bytes());
        for score in &self.reference_scores {
            bytes.extend_from_slice(&score.to_bits().to_le_bytes());
        }
        bytes.extend_from_slice(&self.update.observations.to_le_bytes());
        bytes.extend_from_slice(&self.update.log_e_increment.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.update.log_e_value.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.update.running_max_log_e_value.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.update.anytime_p_value.to_bits().to_le_bytes());
        bytes.push(u8::from(self.update.crossed_alpha));
        bytes
    }
}

fn append_string(target: &mut Vec<u8>, value: &str) {
    target.extend_from_slice(&(value.len() as u64).to_le_bytes());
    target.extend_from_slice(value.as_bytes());
}

/// Predictable finite-reference conformal e-process for one candidate null.
///
/// At each look, commit the observation identity, candidate-specific decision
/// digest, score implementation identity and fresh reference-batch identity.
/// The caller then submits the positive score for the real observation and
/// the same score applied to K fresh draws from the candidate null.
///
/// The process enforces local chronology and unique IDs. It cannot prove that
/// the caller used the same score function on all K+1 values, that the draws
/// are fresh/independent, or that no outcome leaked into the score choice.
#[derive(Debug, Clone, PartialEq)]
pub struct FiniteReferenceConformalEProcess {
    candidate_id: String,
    alpha: f64,
    pending: Option<PendingScoreCommitment>,
    consumed_observation_ids: BTreeSet<String>,
    consumed_reference_batch_ids: BTreeSet<String>,
    observations: u64,
    log_e_value: f64,
    running_max_log_e_value: f64,
}

impl FiniteReferenceConformalEProcess {
    pub fn new(
        candidate_id: impl Into<String>,
        alpha: f64,
    ) -> Result<Self, ConformalError> {
        let candidate_id = candidate_id.into();
        validate_identifier(&candidate_id)?;
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(ConformalError::InvalidAlpha);
        }
        Ok(Self {
            candidate_id,
            alpha,
            pending: None,
            consumed_observation_ids: BTreeSet::new(),
            consumed_reference_batch_ids: BTreeSet::new(),
            observations: 0,
            log_e_value: 0.0,
            running_max_log_e_value: 0.0,
        })
    }

    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn observations(&self) -> u64 {
        self.observations
    }

    pub fn log_e_value(&self) -> f64 {
        self.log_e_value
    }

    pub fn running_max_log_e_value(&self) -> f64 {
        self.running_max_log_e_value
    }

    pub fn commit_score(
        &mut self,
        observation_id: impl Into<String>,
        decision_digest: impl Into<String>,
        score_id: impl Into<String>,
        reference_batch_id: impl Into<String>,
        reference_count: usize,
    ) -> Result<(), ConformalError> {
        if self.pending.is_some() {
            return Err(ConformalError::PredictionAlreadyPending);
        }
        let observation_id = observation_id.into();
        let decision_digest = decision_digest.into();
        let score_id = score_id.into();
        let reference_batch_id = reference_batch_id.into();
        validate_identifier(&observation_id)?;
        validate_identifier(&decision_digest)?;
        validate_identifier(&score_id)?;
        validate_identifier(&reference_batch_id)?;
        if reference_count == 0 {
            return Err(ConformalError::InvalidReferenceCount);
        }
        if reference_count > MAX_REFERENCE_DRAWS {
            return Err(ConformalError::TooManyReferenceDraws);
        }
        if self.consumed_observation_ids.contains(&observation_id) {
            return Err(ConformalError::DuplicateObservationId);
        }
        if self
            .consumed_reference_batch_ids
            .contains(&reference_batch_id)
        {
            return Err(ConformalError::DuplicateReferenceBatchId);
        }

        self.pending = Some(PendingScoreCommitment {
            observation_id,
            decision_digest,
            reference_batch_id,
            score_id,
            reference_count,
        });
        Ok(())
    }

    pub fn observe(
        &mut self,
        observation_id: &str,
        decision_digest: &str,
        score_id: &str,
        reference_batch_id: &str,
        observed_score: f64,
        reference_scores: &[f64],
    ) -> Result<ConformalObservationReceipt, ConformalError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or(ConformalError::NoPredictionPending)?;
        if pending.observation_id != observation_id
            || pending.decision_digest != decision_digest
            || pending.score_id != score_id
            || pending.reference_batch_id != reference_batch_id
        {
            return Err(ConformalError::IdentityMismatch);
        }
        if self.consumed_observation_ids.contains(observation_id) {
            return Err(ConformalError::DuplicateObservationId);
        }
        if self
            .consumed_reference_batch_ids
            .contains(reference_batch_id)
        {
            return Err(ConformalError::DuplicateReferenceBatchId);
        }
        if !observed_score.is_finite() || observed_score <= 0.0 {
            return Err(ConformalError::InvalidObservedScore);
        }
        if reference_scores.len() != pending.reference_count {
            return Err(ConformalError::ReferenceCountMismatch {
                expected: pending.reference_count,
                actual: reference_scores.len(),
            });
        }
        for (index, score) in reference_scores.iter().enumerate() {
            if !score.is_finite() || *score <= 0.0 {
                return Err(ConformalError::InvalidReferenceScore { index });
            }
        }

        // log[(K+1) * s_obs / sum_{j=0}^K s_j], evaluated stably.
        let observed_log_score = observed_score.ln();
        let reference_log_scores = reference_scores
            .iter()
            .map(|score| score.ln())
            .collect::<Vec<_>>();
        let maximum_log_score = reference_log_scores
            .iter()
            .copied()
            .fold(observed_log_score, f64::max);
        let scaled_sum = (observed_log_score - maximum_log_score).exp()
            + reference_log_scores
                .iter()
                .map(|score| (*score - maximum_log_score).exp())
                .sum::<f64>();
        if !scaled_sum.is_finite() || scaled_sum <= 0.0 {
            return Err(ConformalError::NumericalOverflow);
        }
        let log_denominator = maximum_log_score + scaled_sum.ln();
        let log_factor = ((pending.reference_count + 1) as f64).ln()
            + observed_log_score
            - log_denominator;
        if !log_factor.is_finite() {
            return Err(ConformalError::NumericalOverflow);
        }
        let next_log_e_value = self.log_e_value + log_factor;
        if !next_log_e_value.is_finite() {
            return Err(ConformalError::NumericalOverflow);
        }
        let next_running_max = self.running_max_log_e_value.max(next_log_e_value);
        let next_observations = self
            .observations
            .checked_add(1)
            .ok_or(ConformalError::NumericalOverflow)?;
        let anytime_p_value = if next_running_max <= 0.0 {
            1.0
        } else {
            (-next_running_max)
                .exp()
                .clamp(f64::MIN_POSITIVE, 1.0)
        };
        let crossed_alpha = next_running_max >= -self.alpha.ln();

        let receipt = ConformalObservationReceipt {
            schema_version: 1,
            candidate_id: self.candidate_id.clone(),
            observation_id: observation_id.to_owned(),
            decision_digest: decision_digest.to_owned(),
            score_id: score_id.to_owned(),
            reference_batch_id: reference_batch_id.to_owned(),
            observed_score,
            reference_scores: reference_scores.to_vec(),
            update: ConformalUpdate {
                observations: next_observations,
                log_e_increment: log_factor,
                log_e_value: next_log_e_value,
                running_max_log_e_value: next_running_max,
                anytime_p_value,
                crossed_alpha,
            },
        };

        self.log_e_value = next_log_e_value;
        self.running_max_log_e_value = next_running_max;
        self.observations = next_observations;
        self.consumed_observation_ids.insert(observation_id.to_owned());
        self.consumed_reference_batch_ids
            .insert(reference_batch_id.to_owned());
        self.pending = None;

        Ok(receipt)
    }
}

fn validate_identifier(value: &str) -> Result<(), ConformalError> {
    if value.trim().is_empty() {
        Err(ConformalError::EmptyIdentifier)
    } else {
        Ok(())
    }
}

/// Recompute a complete transcript from the exact score and reference inputs.
/// This checks deterministic arithmetic and record consistency only; the
/// caller remains responsible for proving score predictability and fresh
/// conditional reference simulation.
pub fn verify_conformal_transcript(
    candidate_id: &str,
    alpha: f64,
    receipts: &[ConformalObservationReceipt],
) -> Result<(), ConformalError> {
    validate_identifier(candidate_id)?;
    let mut process = FiniteReferenceConformalEProcess::new(candidate_id, alpha)?;
    for receipt in receipts {
        if receipt.schema_version != 1 || receipt.candidate_id != candidate_id {
            return Err(ConformalError::ReceiptCandidateMismatch);
        }
        process.commit_score(
            receipt.observation_id.clone(),
            receipt.decision_digest.clone(),
            receipt.score_id.clone(),
            receipt.reference_batch_id.clone(),
            receipt.reference_scores.len(),
        )?;
        let replayed = process.observe(
            &receipt.observation_id,
            &receipt.decision_digest,
            &receipt.score_id,
            &receipt.reference_batch_id,
            receipt.observed_score,
            &receipt.reference_scores,
        )?;
        if replayed != *receipt {
            return Err(ConformalError::ReceiptReplayMismatch);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformal_factor_matches_the_finite_reference_formula() {
        let mut process = FiniteReferenceConformalEProcess::new("candidate-rotation", 0.05)
            .unwrap();
        process
            .commit_score("obs-1", "decision-1", "score-v1", "refs-1", 2)
            .unwrap();
        let receipt = process
            .observe("obs-1", "decision-1", "score-v1", "refs-1", 3.0, &[1.0, 2.0])
            .unwrap();
        assert!((receipt.update.log_e_increment - 1.5_f64.ln()).abs() < 1.0e-12);
        assert_eq!(receipt.update.observations, 1);
    }

    #[test]
    fn conformal_factor_is_invariant_to_common_positive_score_scaling() {
        let mut first =
            FiniteReferenceConformalEProcess::new("candidate-reflection", 0.05).unwrap();
        first
            .commit_score("obs-1", "decision-1", "score-v1", "refs-1", 2)
            .unwrap();
        let first_receipt = first
            .observe("obs-1", "decision-1", "score-v1", "refs-1", 2.0, &[1.0, 3.0])
            .unwrap();

        let mut scaled =
            FiniteReferenceConformalEProcess::new("candidate-reflection", 0.05).unwrap();
        scaled
            .commit_score("obs-1", "decision-1", "score-v1", "refs-1", 2)
            .unwrap();
        let scaled_receipt = scaled
            .observe(
                "obs-1",
                "decision-1",
                "score-v1",
                "refs-1",
                2.0e200,
                &[1.0e200, 3.0e200],
            )
            .unwrap();

        assert!(
            (first_receipt.update.log_e_increment - scaled_receipt.update.log_e_increment).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn invalid_scores_and_count_mismatches_do_not_consume_pending_state() {
        let mut process =
            FiniteReferenceConformalEProcess::new("candidate-translation", 0.05).unwrap();
        process
            .commit_score("obs-1", "decision-1", "score-v1", "refs-1", 2)
            .unwrap();
        assert_eq!(
            process.observe("obs-1", "decision-1", "score-v1", "refs-1", 1.0, &[1.0]),
            Err(ConformalError::ReferenceCountMismatch {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            process.observe(
                "obs-1",
                "decision-1",
                "score-v1",
                "refs-1",
                0.0,
                &[1.0, 2.0]
            ),
            Err(ConformalError::InvalidObservedScore)
        );
        assert_eq!(
            process.observe(
                "obs-1",
                "decision-1",
                "score-v1",
                "refs-1",
                1.0,
                &[1.0, f64::NAN]
            ),
            Err(ConformalError::InvalidReferenceScore { index: 1 })
        );
        assert_eq!(process.observations(), 0);
        assert!(process.pending.is_some());
    }

    #[test]
    fn score_commitment_and_reference_batches_cannot_be_reused() {
        let mut process =
            FiniteReferenceConformalEProcess::new("candidate-glide", 0.05).unwrap();
        process
            .commit_score("obs-1", "decision-1", "score-v1", "refs-1", 1)
            .unwrap();
        process
            .observe("obs-1", "decision-1", "score-v1", "refs-1", 1.0, &[1.0])
            .unwrap();
        assert_eq!(
            process.commit_score("obs-2", "decision-2", "score-v1", "refs-1", 1),
            Err(ConformalError::DuplicateReferenceBatchId)
        );
        assert_eq!(
            process.commit_score("obs-1", "decision-3", "score-v1", "refs-2", 1),
            Err(ConformalError::DuplicateObservationId)
        );
    }

    #[test]
    fn mismatched_score_or_decision_identity_is_rejected_without_mutation() {
        let mut process =
            FiniteReferenceConformalEProcess::new("candidate-mirror", 0.05).unwrap();
        process
            .commit_score("obs-1", "decision-1", "score-v1", "refs-1", 1)
            .unwrap();
        assert_eq!(
            process.observe("obs-1", "decision-1", "score-v2", "refs-1", 1.0, &[1.0]),
            Err(ConformalError::IdentityMismatch)
        );
        assert_eq!(
            process.observe("obs-1", "wrong-decision", "score-v1", "refs-1", 1.0, &[1.0]),
            Err(ConformalError::IdentityMismatch)
        );
        assert_eq!(process.observations(), 0);
        assert!(process.pending.is_some());
    }

    #[test]
    fn running_maximum_is_not_erased_by_later_low_conformal_factors() {
        let mut process =
            FiniteReferenceConformalEProcess::new("candidate-symmetry", 0.05).unwrap();
        process
            .commit_score("obs-1", "decision-1", "score-v1", "refs-1", 2)
            .unwrap();
        let first = process
            .observe("obs-1", "decision-1", "score-v1", "refs-1", 100.0, &[1.0, 1.0])
            .unwrap();
        process
            .commit_score("obs-2", "decision-2", "score-v1", "refs-2", 2)
            .unwrap();
        let second = process
            .observe("obs-2", "decision-2", "score-v1", "refs-2", 0.01, &[10.0, 10.0])
            .unwrap();
        assert!(second.update.log_e_value < first.update.log_e_value);
        assert_eq!(
            second.update.running_max_log_e_value,
            first.update.running_max_log_e_value
        );
        assert_eq!(
            second.update.anytime_p_value,
            first.update.anytime_p_value
        );
        assert!(second.update.crossed_alpha);
    }

    #[test]
    fn transcript_receipts_replay_exactly_and_detect_forgery() {
        let mut process =
            FiniteReferenceConformalEProcess::new("candidate-hamiltonian", 0.05).unwrap();
        let mut receipts = Vec::new();
        for step in 0..3 {
            let observation_id = format!("obs-{step}");
            let decision_digest = format!("decision-{step}");
            let reference_batch_id = format!("batch-{step}");
            process
                .commit_score(
                    observation_id.clone(),
                    decision_digest.clone(),
                    "score-profile-v1",
                    reference_batch_id.clone(),
                    2,
                )
                .unwrap();
            receipts.push(
                process
                    .observe(
                        &observation_id,
                        &decision_digest,
                        "score-profile-v1",
                        &reference_batch_id,
                        1.5 + step as f64,
                        &[1.0, 2.0],
                    )
                    .unwrap(),
            );
        }
        verify_conformal_transcript("candidate-hamiltonian", 0.05, &receipts).unwrap();

        let mut forged = receipts.clone();
        forged[1].observed_score = 99.0;
        assert_eq!(
            verify_conformal_transcript("candidate-hamiltonian", 0.05, &forged),
            Err(ConformalError::ReceiptReplayMismatch)
        );
    }

    #[test]
    fn complete_receipt_serialization_binds_all_scores_and_update_fields() {
        let mut process =
            FiniteReferenceConformalEProcess::new("candidate-coupled-chain", 0.05).unwrap();
        process
            .commit_score("obs-1", "decision-1", "score-v1", "refs-1", 2)
            .unwrap();
        let receipt = process
            .observe("obs-1", "decision-1", "score-v1", "refs-1", 2.0, &[1.0, 3.0])
            .unwrap();
        let first = receipt.canonical_bytes();
        let mut changed = receipt.clone();
        changed.reference_scores[0] = 2.0;
        assert_ne!(first, changed.canonical_bytes());
    }
}
