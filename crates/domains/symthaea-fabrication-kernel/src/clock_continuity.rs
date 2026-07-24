// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Continuity evidence between quorum-derived clock epochs.

use crate::attestation::SignatureAlgorithm;
use crate::clock::VerifiedClockWindow;
use crate::crypto_digest::{Sha256, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CLOCK_CONTINUITY_SCHEMA: &str = "symthaea.fabrication.clock-continuity.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClockContinuityPolicy {
    pub maximum_epoch_step: u64,
    pub maximum_forward_gap_ms: u64,
    pub maximum_consensus_jump_ms: u64,
    pub minimum_shared_sources: usize,
    pub require_shared_algorithm: bool,
}

impl Default for ClockContinuityPolicy {
    fn default() -> Self {
        Self {
            maximum_epoch_step: 1,
            maximum_forward_gap_ms: 10_000,
            maximum_consensus_jump_ms: 60_000,
            minimum_shared_sources: 1,
            require_shared_algorithm: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedClockContinuity {
    pub schema_version: String,
    pub previous_evidence_digest: Sha256Digest,
    pub successor_evidence_digest: Sha256Digest,
    pub previous_epoch: u64,
    pub successor_epoch: u64,
    pub forward_gap_ms: u64,
    pub consensus_jump_ms: u64,
    pub shared_sources: Vec<String>,
    pub shared_algorithms: Vec<SignatureAlgorithm>,
    pub continuity_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockContinuityError {
    InvalidPolicy,
    InvalidWindow,
    EpochRollback,
    EpochStepTooLarge,
    TimeRegression,
    ForwardGapTooLarge,
    ConsensusJumpTooLarge,
    InsufficientSharedSources,
    MissingSharedAlgorithm,
    DigestMismatch,
    Encoding(String),
}

pub fn verify_clock_continuity(
    previous: &VerifiedClockWindow,
    successor: &VerifiedClockWindow,
    policy: &ClockContinuityPolicy,
) -> Result<VerifiedClockContinuity, ClockContinuityError> {
    validate_policy(policy)?;
    previous
        .validate()
        .map_err(|_| ClockContinuityError::InvalidWindow)?;
    successor
        .validate()
        .map_err(|_| ClockContinuityError::InvalidWindow)?;
    if successor.epoch <= previous.epoch {
        return Err(ClockContinuityError::EpochRollback);
    }
    if successor.epoch - previous.epoch > policy.maximum_epoch_step {
        return Err(ClockContinuityError::EpochStepTooLarge);
    }
    if successor.consensus_unix_ms < previous.consensus_unix_ms {
        return Err(ClockContinuityError::TimeRegression);
    }
    let forward_gap_ms = successor
        .lower_unix_ms
        .saturating_sub(previous.upper_unix_ms);
    if forward_gap_ms > policy.maximum_forward_gap_ms {
        return Err(ClockContinuityError::ForwardGapTooLarge);
    }
    let consensus_jump_ms = successor.consensus_unix_ms - previous.consensus_unix_ms;
    if consensus_jump_ms > policy.maximum_consensus_jump_ms {
        return Err(ClockContinuityError::ConsensusJumpTooLarge);
    }
    let previous_sources = previous.source_ids.iter().cloned().collect::<BTreeSet<_>>();
    let successor_sources = successor
        .source_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let shared_sources = previous_sources
        .intersection(&successor_sources)
        .cloned()
        .collect::<Vec<_>>();
    if shared_sources.len() < policy.minimum_shared_sources {
        return Err(ClockContinuityError::InsufficientSharedSources);
    }
    let previous_algorithms = previous.algorithms.iter().cloned().collect::<BTreeSet<_>>();
    let successor_algorithms = successor
        .algorithms
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let shared_algorithms = previous_algorithms
        .intersection(&successor_algorithms)
        .cloned()
        .collect::<Vec<_>>();
    if policy.require_shared_algorithm && shared_algorithms.is_empty() {
        return Err(ClockContinuityError::MissingSharedAlgorithm);
    }
    let mut evidence = VerifiedClockContinuity {
        schema_version: CLOCK_CONTINUITY_SCHEMA.into(),
        previous_evidence_digest: previous.evidence_digest,
        successor_evidence_digest: successor.evidence_digest,
        previous_epoch: previous.epoch,
        successor_epoch: successor.epoch,
        forward_gap_ms,
        consensus_jump_ms,
        shared_sources,
        shared_algorithms,
        continuity_digest: Sha256Digest([0; 32]),
    };
    validate_clock_continuity_shape(&evidence)?;
    evidence.continuity_digest = digest_clock_continuity_fields(&evidence)?;
    Ok(evidence)
}

pub fn digest_clock_continuity(
    evidence: &VerifiedClockContinuity,
) -> Result<Sha256Digest, ClockContinuityError> {
    validate_clock_continuity_shape(evidence)?;
    let expected = digest_clock_continuity_fields(evidence)?;
    if expected != evidence.continuity_digest {
        return Err(ClockContinuityError::DigestMismatch);
    }
    Ok(expected)
}

fn validate_clock_continuity_shape(
    evidence: &VerifiedClockContinuity,
) -> Result<(), ClockContinuityError> {
    if evidence.schema_version != CLOCK_CONTINUITY_SCHEMA
        || evidence.previous_evidence_digest.0 == [0; 32]
        || evidence.successor_evidence_digest.0 == [0; 32]
        || evidence.previous_epoch == 0
        || evidence.successor_epoch <= evidence.previous_epoch
        || evidence.shared_sources.is_empty()
        || evidence
            .shared_sources
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || evidence
            .shared_algorithms
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || evidence.shared_sources.iter().any(|source| {
            source.trim().is_empty()
                || source != source.trim()
                || source.len() > 128
                || source.chars().any(char::is_control)
        })
        || evidence
            .shared_algorithms
            .iter()
            .any(|algorithm| !algorithm.is_canonical())
    {
        return Err(ClockContinuityError::InvalidWindow);
    }
    Ok(())
}

fn digest_clock_continuity_fields(
    evidence: &VerifiedClockContinuity,
) -> Result<Sha256Digest, ClockContinuityError> {
    let mut canonical = evidence.clone();
    canonical.continuity_digest = Sha256Digest([0; 32]);
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| ClockContinuityError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.clock-continuity-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn validate_policy(policy: &ClockContinuityPolicy) -> Result<(), ClockContinuityError> {
    if policy.maximum_epoch_step == 0
        || policy.maximum_consensus_jump_ms == 0
        || policy.minimum_shared_sources == 0
    {
        return Err(ClockContinuityError::InvalidPolicy);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_epoch_step_is_rejected() {
        let policy = ClockContinuityPolicy {
            maximum_epoch_step: 0,
            ..ClockContinuityPolicy::default()
        };
        assert_eq!(
            validate_policy(&policy),
            Err(ClockContinuityError::InvalidPolicy)
        );
    }
}
