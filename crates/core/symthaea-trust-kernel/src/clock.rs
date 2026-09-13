// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quorum-derived wall-clock evidence and monotonic epoch tracking.
//!
//! Protocol identifiers intentionally preserve fabrication V1 compatibility.

use crate::digest::{Sha256Digest, domain_hash};
use crate::signature::{DetachedSignature, SignatureAlgorithm};
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

pub const CLOCK_OBSERVATION_SCHEMA: &str = "symthaea.fabrication.clock-observation.v1";
const CLOCK_OBSERVATION_DIGEST_DOMAIN: &[u8] = b"symthaea.fabrication.clock-observation-digest.v1\0";
const VERIFIED_CLOCK_WINDOW_DOMAIN: &[u8] = b"symthaea.fabrication.verified-clock-window.v1\0";
const CLOCK_EPOCH_TRACKER_DOMAIN: &[u8] = b"symthaea.fabrication.clock-epoch-tracker.v1\0";
pub const MAX_CLOCK_OBSERVATIONS: usize = 32;
pub const MAX_CLOCK_SOURCE_ID_BYTES: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockObservation {
    pub schema_version: String,
    pub source_id: String,
    pub observed_unix_ms: u64,
    pub uncertainty_ms: u64,
    pub epoch: u64,
    pub signature: DetachedSignature,
}

pub trait ClockObservationVerifier {
    fn verify_clock_observation(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClockQuorumPolicy {
    pub minimum_distinct_sources: usize,
    pub maximum_observations: usize,
    pub maximum_uncertainty_ms: u64,
    pub maximum_consensus_width_ms: u64,
    pub require_algorithm_diversity: bool,
}

impl Default for ClockQuorumPolicy {
    fn default() -> Self {
        Self {
            minimum_distinct_sources: 2,
            maximum_observations: 8,
            maximum_uncertainty_ms: 5_000,
            maximum_consensus_width_ms: 10_000,
            require_algorithm_diversity: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedClockWindow {
    pub lower_unix_ms: u64,
    pub upper_unix_ms: u64,
    pub consensus_unix_ms: u64,
    pub epoch: u64,
    pub source_ids: Vec<String>,
    pub algorithms: Vec<SignatureAlgorithm>,
    pub trust_snapshot_digest: Sha256Digest,
    pub evidence_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockViolation {
    InvalidPolicy,
    TooManyObservations { actual: usize, maximum: usize },
    UnsupportedSchema(String),
    InvalidSourceId(String),
    DuplicateSource(String),
    DuplicateSigner(String),
    ZeroEpoch(String),
    UncertaintyTooLarge(String),
    IntervalOverflow(String),
    SnapshotStale,
    SignerIneligible(String),
    SignatureInvalid(String),
    VerificationFailed(String),
    EpochMismatch { expected: u64, actual: u64 },
    InsufficientSources { actual: usize, required: usize },
    AlgorithmDiversityMissing,
    NoCommonInterval,
    ConsensusTooWide { actual_ms: u64, maximum_ms: u64 },
    Encoding(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockEpochTracker {
    latest_epoch: Option<u64>,
    latest_consensus_unix_ms: Option<u64>,
    latest_evidence_digest: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockTrackingError {
    InvalidTrackerState,
    Encoding(String),
    InvalidWindow,
    EpochRollback { latest: u64, proposed: u64 },
    EpochCollision { epoch: u64 },
    TimeRegression { latest: u64, proposed: u64 },
}

impl VerifiedClockWindow {
    pub fn validate(&self) -> Result<(), ClockTrackingError> {
        if self.epoch == 0
            || self.lower_unix_ms > self.consensus_unix_ms
            || self.consensus_unix_ms > self.upper_unix_ms
            || self.source_ids.is_empty()
            || self.algorithms.is_empty()
            || self.trust_snapshot_digest.0 == [0; 32]
            || self.evidence_digest.0 == [0; 32]
        {
            return Err(ClockTrackingError::InvalidWindow);
        }
        Ok(())
    }
}

impl ClockEpochTracker {
    pub fn validate(&self) -> Result<(), ClockTrackingError> {
        match (
            self.latest_epoch,
            self.latest_consensus_unix_ms,
            self.latest_evidence_digest,
        ) {
            (None, None, None) => Ok(()),
            (Some(epoch), Some(_), Some(digest)) if epoch > 0 && digest.0 != [0; 32] => Ok(()),
            _ => Err(ClockTrackingError::InvalidTrackerState),
        }
    }

    pub fn accept(&mut self, window: &VerifiedClockWindow) -> Result<(), ClockTrackingError> {
        self.validate()?;
        window.validate()?;
        if let Some(latest) = self.latest_epoch {
            if window.epoch < latest {
                return Err(ClockTrackingError::EpochRollback {
                    latest,
                    proposed: window.epoch,
                });
            }
            if window.epoch == latest {
                if self.latest_evidence_digest == Some(window.evidence_digest) {
                    return Ok(());
                }
                return Err(ClockTrackingError::EpochCollision { epoch: window.epoch });
            }
        }
        if let Some(latest) = self.latest_consensus_unix_ms {
            if window.consensus_unix_ms < latest {
                return Err(ClockTrackingError::TimeRegression {
                    latest,
                    proposed: window.consensus_unix_ms,
                });
            }
        }
        self.latest_epoch = Some(window.epoch);
        self.latest_consensus_unix_ms = Some(window.consensus_unix_ms);
        self.latest_evidence_digest = Some(window.evidence_digest);
        Ok(())
    }

    pub fn latest_epoch(&self) -> Option<u64> {
        self.latest_epoch
    }

    pub fn latest_consensus_unix_ms(&self) -> Option<u64> {
        self.latest_consensus_unix_ms
    }

    pub fn latest_evidence_digest(&self) -> Option<Sha256Digest> {
        self.latest_evidence_digest
    }
}

pub fn canonical_clock_observation_bytes(
    observation: &ClockObservation,
) -> Result<Vec<u8>, ClockViolation> {
    validate_observation_shape(observation, u64::MAX)?;
    serde_json::to_vec(&(
        &observation.schema_version,
        &observation.source_id,
        observation.observed_unix_ms,
        observation.uncertainty_ms,
        observation.epoch,
        &observation.signature.algorithm,
        &observation.signature.key_id,
    ))
    .map_err(|error| ClockViolation::Encoding(error.to_string()))
}

pub fn digest_clock_observation(
    observation: &ClockObservation,
) -> Result<Sha256Digest, ClockViolation> {
    let bytes = canonical_clock_observation_bytes(observation)?;
    Ok(domain_hash(CLOCK_OBSERVATION_DIGEST_DOMAIN, &bytes))
}

pub fn verify_clock_quorum(
    observations: &[ClockObservation],
    policy: &ClockQuorumPolicy,
    trust_snapshot: &TrustSnapshot,
    evaluation_time_unix_s: u64,
    verifier: &dyn ClockObservationVerifier,
) -> Result<VerifiedClockWindow, Vec<ClockViolation>> {
    let mut violations = Vec::new();
    if policy.minimum_distinct_sources == 0
        || policy.maximum_observations == 0
        || policy.minimum_distinct_sources > policy.maximum_observations
        || policy.maximum_observations > MAX_CLOCK_OBSERVATIONS
        || policy.maximum_consensus_width_ms == 0
    {
        violations.push(ClockViolation::InvalidPolicy);
    }
    if observations.len() > policy.maximum_observations {
        violations.push(ClockViolation::TooManyObservations {
            actual: observations.len(),
            maximum: policy.maximum_observations,
        });
    }
    if !trust_snapshot.is_fresh_at(evaluation_time_unix_s) {
        violations.push(ClockViolation::SnapshotStale);
    }

    let mut expected_epoch = None;
    let mut seen_sources = BTreeSet::new();
    let mut accepted_sources = BTreeSet::new();
    let mut seen_signers = BTreeSet::new();
    let mut algorithms = BTreeSet::new();
    let mut lower = 0u64;
    let mut upper = u64::MAX;
    let mut accepted_digests = Vec::new();

    for observation in observations {
        if let Err(violation) =
            validate_observation_shape(observation, policy.maximum_uncertainty_ms)
        {
            violations.push(violation);
            continue;
        }
        if let Some(expected) = expected_epoch {
            if observation.epoch != expected {
                violations.push(ClockViolation::EpochMismatch {
                    expected,
                    actual: observation.epoch,
                });
                continue;
            }
        } else {
            expected_epoch = Some(observation.epoch);
        }
        if !seen_sources.insert(observation.source_id.clone()) {
            violations.push(ClockViolation::DuplicateSource(observation.source_id.clone()));
            continue;
        }
        let signer = (
            observation.signature.algorithm.clone(),
            observation.signature.key_id.clone(),
        );
        if !seen_signers.insert(signer) {
            violations.push(ClockViolation::DuplicateSigner(
                observation.signature.key_id.clone(),
            ));
            continue;
        }
        match trust_snapshot.key_eligibility(
            &observation.signature.algorithm,
            &observation.signature.key_id,
            KeyUsage::ClockAuthority,
            evaluation_time_unix_s,
        ) {
            KeyEligibility::Eligible => {}
            _ => {
                violations.push(ClockViolation::SignerIneligible(
                    observation.signature.key_id.clone(),
                ));
                continue;
            }
        }
        let message = match canonical_clock_observation_bytes(observation) {
            Ok(message) => message,
            Err(violation) => {
                violations.push(violation);
                continue;
            }
        };
        match verifier.verify_clock_observation(
            &observation.signature.algorithm,
            &observation.signature.key_id,
            &message,
            &observation.signature.signature,
        ) {
            Ok(true) => {}
            Ok(false) => {
                violations.push(ClockViolation::SignatureInvalid(
                    observation.signature.key_id.clone(),
                ));
                continue;
            }
            Err(error) => {
                violations.push(ClockViolation::VerificationFailed(error));
                continue;
            }
        }
        let observation_lower = observation
            .observed_unix_ms
            .saturating_sub(observation.uncertainty_ms);
        let Some(observation_upper) = observation
            .observed_unix_ms
            .checked_add(observation.uncertainty_ms)
        else {
            violations.push(ClockViolation::IntervalOverflow(observation.source_id.clone()));
            continue;
        };
        lower = lower.max(observation_lower);
        upper = upper.min(observation_upper);
        accepted_sources.insert(observation.source_id.clone());
        algorithms.insert(observation.signature.algorithm.clone());
        match digest_clock_observation(observation) {
            Ok(digest) => accepted_digests.push(digest),
            Err(violation) => violations.push(violation),
        }
    }

    if accepted_sources.len() < policy.minimum_distinct_sources {
        violations.push(ClockViolation::InsufficientSources {
            actual: accepted_sources.len(),
            required: policy.minimum_distinct_sources,
        });
    }
    if policy.require_algorithm_diversity && algorithms.len() < 2 {
        violations.push(ClockViolation::AlgorithmDiversityMissing);
    }
    if lower > upper {
        violations.push(ClockViolation::NoCommonInterval);
    } else if upper - lower > policy.maximum_consensus_width_ms {
        violations.push(ClockViolation::ConsensusTooWide {
            actual_ms: upper - lower,
            maximum_ms: policy.maximum_consensus_width_ms,
        });
    }
    if !violations.is_empty() {
        return Err(violations);
    }

    accepted_digests.sort();
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(|error| vec![ClockViolation::Encoding(format!("{error:?}"))])?;
    let consensus_unix_ms = lower + (upper - lower) / 2;
    let source_ids = accepted_sources.into_iter().collect::<Vec<_>>();
    let algorithms = algorithms.into_iter().collect::<Vec<_>>();
    let expected_epoch = expected_epoch.unwrap_or(0);

    let mut hasher = Sha256::new();
    hasher.update(VERIFIED_CLOCK_WINDOW_DOMAIN);
    hasher.update(expected_epoch.to_le_bytes());
    hasher.update(lower.to_le_bytes());
    hasher.update(upper.to_le_bytes());
    hasher.update(trust_snapshot_digest.0);
    for digest in &accepted_digests {
        hasher.update(digest.0);
    }
    let evidence_digest = Sha256Digest(hasher.finalize().into());

    Ok(VerifiedClockWindow {
        lower_unix_ms: lower,
        upper_unix_ms: upper,
        consensus_unix_ms,
        epoch: expected_epoch,
        source_ids,
        algorithms,
        trust_snapshot_digest,
        evidence_digest,
    })
}

pub fn digest_clock_epoch_tracker(
    tracker: &ClockEpochTracker,
) -> Result<Sha256Digest, ClockTrackingError> {
    tracker.validate()?;
    let bytes = serde_json::to_vec(tracker)
        .map_err(|error| ClockTrackingError::Encoding(error.to_string()))?;
    Ok(domain_hash(CLOCK_EPOCH_TRACKER_DOMAIN, &bytes))
}

fn validate_observation_shape(
    observation: &ClockObservation,
    maximum_uncertainty_ms: u64,
) -> Result<(), ClockViolation> {
    if observation.schema_version != CLOCK_OBSERVATION_SCHEMA {
        return Err(ClockViolation::UnsupportedSchema(observation.source_id.clone()));
    }
    if observation.source_id.trim().is_empty()
        || observation.source_id != observation.source_id.trim()
        || observation.source_id.len() > MAX_CLOCK_SOURCE_ID_BYTES
        || observation.source_id.chars().any(char::is_control)
    {
        return Err(ClockViolation::InvalidSourceId(observation.source_id.clone()));
    }
    if observation.epoch == 0 {
        return Err(ClockViolation::ZeroEpoch(observation.source_id.clone()));
    }
    if observation.uncertainty_ms > maximum_uncertainty_ms {
        return Err(ClockViolation::UncertaintyTooLarge(observation.source_id.clone()));
    }
    if !observation.signature.algorithm.is_canonical()
        || observation.signature.key_id.trim().is_empty()
        || observation.signature.key_id != observation.signature.key_id.trim()
        || observation.signature.signature.is_empty()
    {
        return Err(ClockViolation::SignatureInvalid(observation.source_id.clone()));
    }
    Ok(())
}
