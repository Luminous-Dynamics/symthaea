// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Anytime-valid paired evaluation for sequential WeightProgram candidates.
//!
//! Every candidate is frozen before its confirmatory evaluation stream begins.
//! Each observation compares baseline and candidate losses on the same episode,
//! with both losses normalized to [0, 1]. Improvement is baseline loss minus
//! candidate loss, so positive values favor the candidate.
//!
//! A union-bound Hoeffding confidence sequence spends one campaign-wide alpha
//! budget across candidate indices and every possible interim look. This
//! protects against optional stopping and repeated candidate evaluation, so
//! long as each candidate's confirmatory episodes are fresh, its model/protocol
//! digests remain fixed, and a candidate is not edited mid-stream. New edits
//! require a new candidate index and a fresh confirmatory stream.
//!
//! This is a statistical qualification primitive, not a verifier of task
//! correctness, safety or model identity. A signed independent verifier must
//! still bind its report digest to the WeightProgram acceptance attestation.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const DIGEST_HEX_LEN: usize = 64;

/// Immutable identities bound to one candidate's confirmatory evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightVerificationBinding {
    /// One-based candidate ordinal in a preregistered repair campaign.
    pub candidate_index: u64,
    /// Digest of the baseline model evaluated on each paired episode.
    pub baseline_model_digest: String,
    /// Digest of the frozen candidate model evaluated on each paired episode.
    pub candidate_model_digest: String,
    /// Digest of the frozen task, split, loss, and evaluation protocol.
    pub evaluation_protocol_digest: String,
}

/// Alpha-spending and practical-effect requirements for a repair candidate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SequentialWeightVerificationConfig {
    /// Total family-wise error budget across all candidate indices and looks.
    pub campaign_alpha: f64,
    /// Minimum practically meaningful average loss reduction in [0, 1].
    pub minimum_improvement: f64,
    /// Minimum paired episodes before a candidate can be qualified.
    pub minimum_observations: u64,
    /// Maximum paired episodes allowed for this candidate.
    pub maximum_observations: u64,
}

/// One paired observation from a frozen confirmatory episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedLossObservation {
    /// Stable episode identifier; duplicates are rejected.
    pub episode_id: String,
    /// Baseline loss, normalized to [0, 1].
    pub baseline_loss: f64,
    /// Candidate loss on the same episode, normalized to [0, 1].
    pub candidate_loss: f64,
}

/// State of a candidate's anytime-valid comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequentialWeightVerificationStatus {
    /// More paired confirmatory episodes may be collected.
    Collecting,
    /// The lower confidence bound exceeds the required improvement.
    BenefitEstablished,
    /// The upper confidence bound is below the required improvement.
    BenefitNotEstablished,
    /// The observation budget ended before either conclusion was supported.
    InconclusiveAtBudget,
}

/// Replayable confidence-sequence report for one frozen candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialWeightVerificationReport {
    /// Model and protocol identities.
    pub binding: WeightVerificationBinding,
    /// Frozen sequential testing policy.
    pub config: SequentialWeightVerificationConfig,
    /// Number of unique paired episodes processed.
    pub observation_count: u64,
    /// Mean paired improvement; positive favors the candidate.
    pub mean_improvement: f64,
    /// Time-uniform lower confidence bound.
    pub lower_confidence_bound: f64,
    /// Time-uniform upper confidence bound.
    pub upper_confidence_bound: f64,
    /// Confidence radius used at the current look.
    pub confidence_radius: f64,
    /// Local alpha allocation for this candidate at this observation count.
    pub alpha_at_look: f64,
    /// Digest of ordered episode IDs and paired losses.
    pub observation_digest: String,
    /// Digest of the complete report claims.
    pub report_digest: String,
    /// Current qualification status.
    pub status: SequentialWeightVerificationStatus,
}

/// Failure from sequential repair verification.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SequentialWeightVerificationError {
    #[error("candidate index must be one-based and non-zero")]
    InvalidCandidateIndex,
    #[error("model or protocol digest is not a 64-character lowercase hexadecimal digest: {0}")]
    InvalidDigest(&'static str),
    #[error("campaign alpha must be in (0, 1) and minimum improvement in [0, 1]")]
    InvalidConfig,
    #[error("minimum observations must be non-zero and no greater than maximum observations")]
    InvalidObservationBudget,
    #[error("episode identifier must be non-empty")]
    EmptyEpisodeId,
    #[error("episode identifier has already been evaluated: {0}")]
    DuplicateEpisodeId(String),
    #[error("baseline or candidate loss is non-finite or outside [0, 1]")]
    InvalidLoss,
    #[error("candidate observation budget is exhausted")]
    ObservationBudgetExhausted,
    #[error("candidate evaluation is already terminal")]
    AlreadyTerminal,
    #[error("failed to serialize sequential verification report")]
    ReportSerialization,
}

/// Incremental verifier for one frozen baseline/candidate/protocol triple.
#[derive(Debug, Clone)]
pub struct SequentialWeightVerification {
    binding: WeightVerificationBinding,
    config: SequentialWeightVerificationConfig,
    seen_episode_ids: BTreeSet<String>,
    observation_count: u64,
    mean_improvement: f64,
    observation_hasher: blake3::Hasher,
    terminal: bool,
}

impl SequentialWeightVerification {
    /// Start a new confirmatory stream for a frozen candidate.
    pub fn new(
        binding: WeightVerificationBinding,
        config: SequentialWeightVerificationConfig,
    ) -> Result<Self, SequentialWeightVerificationError> {
        validate_binding(&binding)?;
        validate_config(config)?;

        let mut observation_hasher = blake3::Hasher::new();
        observation_hasher.update(b"symthaea-weight-verification-observations-v1\0");

        Ok(Self {
            binding,
            config,
            seen_episode_ids: BTreeSet::new(),
            observation_count: 0,
            mean_improvement: 0.0,
            observation_hasher,
            terminal: false,
        })
    }

    /// Add one paired episode and return the updated confidence-sequence report.
    ///
    /// A candidate cannot be modified after the first observation because its
    /// model digest is fixed in the verifier binding. Each episode identifier
    /// may appear only once.
    pub fn observe(
        &mut self,
        observation: PairedLossObservation,
    ) -> Result<SequentialWeightVerificationReport, SequentialWeightVerificationError> {
        if self.terminal {
            return Err(SequentialWeightVerificationError::AlreadyTerminal);
        }
        if self.observation_count >= self.config.maximum_observations {
            return Err(SequentialWeightVerificationError::ObservationBudgetExhausted);
        }
        if observation.episode_id.trim().is_empty() {
            return Err(SequentialWeightVerificationError::EmptyEpisodeId);
        }
        if !observation.baseline_loss.is_finite()
            || !(0.0..=1.0).contains(&observation.baseline_loss)
            || !observation.candidate_loss.is_finite()
            || !(0.0..=1.0).contains(&observation.candidate_loss)
        {
            return Err(SequentialWeightVerificationError::InvalidLoss);
        }
        if self.seen_episode_ids.contains(&observation.episode_id) {
            return Err(SequentialWeightVerificationError::DuplicateEpisodeId(
                observation.episode_id,
            ));
        }

        let improvement = observation.baseline_loss - observation.candidate_loss;
        self.seen_episode_ids.insert(observation.episode_id.clone());
        self.observation_count += 1;
        self.mean_improvement +=
            (improvement - self.mean_improvement) / self.observation_count as f64;

        let episode_id_bytes = observation.episode_id.as_bytes();
        self.observation_hasher
            .update(&(episode_id_bytes.len() as u64).to_le_bytes());
        self.observation_hasher.update(episode_id_bytes);
        self.observation_hasher
            .update(&observation.baseline_loss.to_bits().to_le_bytes());
        self.observation_hasher
            .update(&observation.candidate_loss.to_bits().to_le_bytes());

        let report = self.current_report()?;
        self.terminal = matches!(
            report.status,
            SequentialWeightVerificationStatus::BenefitEstablished
                | SequentialWeightVerificationStatus::BenefitNotEstablished
                | SequentialWeightVerificationStatus::InconclusiveAtBudget
        );
        Ok(report)
    }

    /// Return the current report without consuming another episode.
    pub fn report(
        &self,
    ) -> Result<SequentialWeightVerificationReport, SequentialWeightVerificationError> {
        self.current_report()
    }

    fn current_report(
        &self,
    ) -> Result<SequentialWeightVerificationReport, SequentialWeightVerificationError> {
        let n = self.observation_count;
        let n_float = n as f64;
        let candidate = self.binding.candidate_index as f64;
        let alpha_denominator = candidate * (candidate + 1.0) * n_float * (n_float + 1.0);
        let alpha_at_look = self.config.campaign_alpha / alpha_denominator;
        let log_tail_ratio = std::f64::consts::LN_2
            - self.config.campaign_alpha.ln()
            + candidate.ln()
            + (candidate + 1.0).ln()
            + n_float.ln()
            + (n_float + 1.0).ln();
        let confidence_radius = if n == 0 {
            f64::INFINITY
        } else {
            (2.0 * log_tail_ratio / n_float).sqrt()
        };
        let lower_confidence_bound = if n == 0 {
            -1.0
        } else {
            (self.mean_improvement - confidence_radius).max(-1.0)
        };
        let upper_confidence_bound = if n == 0 {
            1.0
        } else {
            (self.mean_improvement + confidence_radius).min(1.0)
        };

        let status = if n < self.config.minimum_observations {
            SequentialWeightVerificationStatus::Collecting
        } else if lower_confidence_bound > self.config.minimum_improvement {
            SequentialWeightVerificationStatus::BenefitEstablished
        } else if upper_confidence_bound < self.config.minimum_improvement {
            SequentialWeightVerificationStatus::BenefitNotEstablished
        } else if n >= self.config.maximum_observations {
            SequentialWeightVerificationStatus::InconclusiveAtBudget
        } else {
            SequentialWeightVerificationStatus::Collecting
        };

        let observation_digest = self
            .observation_hasher
            .clone()
            .finalize()
            .to_hex()
            .to_string();
        let claims = ReportDigestClaims {
            binding: &self.binding,
            config: self.config,
            observation_count: n,
            mean_improvement: self.mean_improvement,
            lower_confidence_bound,
            upper_confidence_bound,
            confidence_radius,
            alpha_at_look,
            observation_digest: &observation_digest,
            status,
        };
        let encoded =
            serde_json::to_vec(&claims).map_err(|_| SequentialWeightVerificationError::ReportSerialization)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-weight-verification-report-v1\0");
        hasher.update(&(encoded.len() as u64).to_le_bytes());
        hasher.update(&encoded);
        let report_digest = hasher.finalize().to_hex().to_string();

        Ok(SequentialWeightVerificationReport {
            binding: self.binding.clone(),
            config: self.config,
            observation_count: n,
            mean_improvement: self.mean_improvement,
            lower_confidence_bound,
            upper_confidence_bound,
            confidence_radius,
            alpha_at_look,
            observation_digest,
            report_digest,
            status,
        })
    }
}

#[derive(Serialize)]
struct ReportDigestClaims<'a> {
    binding: &'a WeightVerificationBinding,
    config: SequentialWeightVerificationConfig,
    observation_count: u64,
    mean_improvement: f64,
    lower_confidence_bound: f64,
    upper_confidence_bound: f64,
    confidence_radius: f64,
    alpha_at_look: f64,
    observation_digest: &'a str,
    status: SequentialWeightVerificationStatus,
}

fn validate_binding(
    binding: &WeightVerificationBinding,
) -> Result<(), SequentialWeightVerificationError> {
    if binding.candidate_index == 0 {
        return Err(SequentialWeightVerificationError::InvalidCandidateIndex);
    }
    for (name, value) in [
        ("baseline_model_digest", binding.baseline_model_digest.as_str()),
        ("candidate_model_digest", binding.candidate_model_digest.as_str()),
        (
            "evaluation_protocol_digest",
            binding.evaluation_protocol_digest.as_str(),
        ),
    ] {
        if value.len() != DIGEST_HEX_LEN
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(SequentialWeightVerificationError::InvalidDigest(name));
        }
    }
    if binding.baseline_model_digest == binding.candidate_model_digest {
        return Err(SequentialWeightVerificationError::InvalidDigest(
            "candidate must differ from baseline",
        ));
    }
    Ok(())
}

fn validate_config(
    config: SequentialWeightVerificationConfig,
) -> Result<(), SequentialWeightVerificationError> {
    if !config.campaign_alpha.is_finite()
        || !(0.0..1.0).contains(&config.campaign_alpha)
        || !config.minimum_improvement.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_improvement)
    {
        return Err(SequentialWeightVerificationError::InvalidConfig);
    }
    if config.minimum_observations == 0
        || config.minimum_observations > config.maximum_observations
    {
        return Err(SequentialWeightVerificationError::InvalidObservationBudget);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(candidate_index: u64) -> WeightVerificationBinding {
        WeightVerificationBinding {
            candidate_index,
            baseline_model_digest: "a".repeat(64),
            candidate_model_digest: "b".repeat(64),
            evaluation_protocol_digest: "c".repeat(64),
        }
    }

    fn config(minimum_improvement: f64, maximum_observations: u64) -> SequentialWeightVerificationConfig {
        SequentialWeightVerificationConfig {
            campaign_alpha: 0.05,
            minimum_improvement,
            minimum_observations: 2,
            maximum_observations,
        }
    }

    fn observation(index: u64, baseline_loss: f64, candidate_loss: f64) -> PairedLossObservation {
        PairedLossObservation {
            episode_id: format!("episode-{index}"),
            baseline_loss,
            candidate_loss,
        }
    }

    #[test]
    fn strong_paired_improvement_is_established_with_anytime_valid_bounds() {
        let mut verifier = SequentialWeightVerification::new(binding(1), config(0.05, 2_000))
            .expect("valid verifier");
        let mut report = verifier.report().expect("empty report");
        for index in 0..600 {
            report = verifier
                .observe(observation(index, 0.9, 0.5))
                .expect("paired episode");
            if report.status == SequentialWeightVerificationStatus::BenefitEstablished {
                break;
            }
        }
        assert_eq!(
            report.status,
            SequentialWeightVerificationStatus::BenefitEstablished
        );
        assert!(report.lower_confidence_bound > report.config.minimum_improvement);
        assert_eq!(report.observation_count, 600);
    }

    #[test]
    fn no_effect_is_not_misreported_as_superiority() {
        let mut verifier = SequentialWeightVerification::new(binding(1), config(0.1, 6_000))
            .expect("valid verifier");
        let mut report = verifier.report().expect("empty report");
        for index in 0..6_000 {
            report = verifier
                .observe(observation(index, 0.5, 0.5))
                .expect("paired episode");
            if report.status != SequentialWeightVerificationStatus::Collecting {
                break;
            }
        }
        assert_eq!(
            report.status,
            SequentialWeightVerificationStatus::BenefitNotEstablished
        );
        assert!(report.upper_confidence_bound < report.config.minimum_improvement);
    }

    #[test]
    fn later_candidates_receive_smaller_alpha_allocation() {
        let mut first = SequentialWeightVerification::new(binding(1), config(0.2, 100))
            .expect("first candidate");
        let mut later = SequentialWeightVerification::new(binding(7), config(0.2, 100))
            .expect("later candidate");
        let first_report = first.observe(observation(0, 0.8, 0.5)).expect("first look");
        let later_report = later.observe(observation(0, 0.8, 0.5)).expect("later look");
        assert!(later_report.alpha_at_look < first_report.alpha_at_look);
        assert!(later_report.confidence_radius > first_report.confidence_radius);
    }

    #[test]
    fn duplicate_episode_ids_are_rejected_without_consuming_budget() {
        let mut verifier = SequentialWeightVerification::new(binding(1), config(0.1, 10))
            .expect("valid verifier");
        verifier
            .observe(observation(1, 0.8, 0.4))
            .expect("first episode");
        assert!(matches!(
            verifier.observe(observation(1, 0.2, 0.1)),
            Err(SequentialWeightVerificationError::DuplicateEpisodeId(_))
        ));
        assert_eq!(verifier.report().expect("report").observation_count, 1);
    }

    #[test]
    fn malformed_digests_losses_and_budgets_fail_closed() {
        let mut invalid_binding = binding(1);
        invalid_binding.candidate_model_digest = "not-a-digest".into();
        assert!(matches!(
            SequentialWeightVerification::new(invalid_binding, config(0.1, 10)),
            Err(SequentialWeightVerificationError::InvalidDigest(_))
        ));

        let mut verifier = SequentialWeightVerification::new(binding(1), config(0.1, 10))
            .expect("valid verifier");
        assert!(matches!(
            verifier.observe(observation(1, 1.1, 0.1)),
            Err(SequentialWeightVerificationError::InvalidLoss)
        ));
        assert_eq!(verifier.report().expect("report").observation_count, 0);
    }

    #[test]
    fn report_digest_is_replay_deterministic_and_binds_the_evidence_stream() {
        let mut left = SequentialWeightVerification::new(binding(2), config(0.05, 20))
            .expect("left verifier");
        let mut right = SequentialWeightVerification::new(binding(2), config(0.05, 20))
            .expect("right verifier");
        let episodes = [
            observation(1, 0.8, 0.4),
            observation(2, 0.7, 0.3),
            observation(3, 0.9, 0.5),
        ];
        let mut left_report = left.report().expect("initial");
        let mut right_report = right.report().expect("initial");
        for episode in episodes {
            left_report = left.observe(episode.clone()).expect("left");
            right_report = right.observe(episode).expect("right");
        }
        assert_eq!(left_report.observation_digest, right_report.observation_digest);
        assert_eq!(left_report.report_digest, right_report.report_digest);
        assert_ne!(
            left_report.report_digest,
            left.observe(observation(4, 0.1, 0.1))
                .expect_err("new episode after terminal state")
                .to_string()
        );
    }

    #[test]
    fn inconclusive_budget_is_terminal_and_cannot_be_extended() {
        let mut verifier = SequentialWeightVerification::new(binding(1), config(0.5, 2))
            .expect("valid verifier");
        verifier
            .observe(observation(1, 0.5, 0.5))
            .expect("first episode");
        let report = verifier
            .observe(observation(2, 0.5, 0.5))
            .expect("budget-ending episode");
        assert_eq!(
            report.status,
            SequentialWeightVerificationStatus::InconclusiveAtBudget
        );
        assert!(matches!(
            verifier.observe(observation(3, 0.9, 0.1)),
            Err(SequentialWeightVerificationError::AlreadyTerminal)
        ));
    }
}
