// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quorum-derived wall-clock evidence and monotonic epoch tracking.
//!
//! Fabrication authority must not rely on one unauthenticated local clock. This
//! module verifies independent signed observations, derives a bounded consensus
//! interval, and remembers the latest accepted epoch so persisted state cannot
//! silently move time backwards.

use crate::attestation::{DetachedSignature, SignatureAlgorithm};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CLOCK_OBSERVATION_SCHEMA: &str = "symthaea.fabrication.clock-observation.v1";
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
                return Err(ClockTrackingError::EpochCollision {
                    epoch: window.epoch,
                });
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
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.clock-observation-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
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
            violations.push(ClockViolation::DuplicateSource(
                observation.source_id.clone(),
            ));
            continue;
        }
        let signer = (
            observation.signature.algorithm.clone(),
            observation.signature.key_id.clone(),
        );
        if !seen_signers.insert(signer.clone()) {
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
            violations.push(ClockViolation::IntervalOverflow(
                observation.source_id.clone(),
            ));
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
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.verified-clock-window.v1\0");
    let expected_epoch = expected_epoch.unwrap_or(0);
    hasher.update(&expected_epoch.to_le_bytes());
    hasher.update(&lower.to_le_bytes());
    hasher.update(&upper.to_le_bytes());
    hasher.update(&trust_snapshot_digest.0);
    for digest in &accepted_digests {
        hasher.update(&digest.0);
    }
    let evidence_digest = hasher.finalize();

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
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.clock-epoch-tracker.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn validate_observation_shape(
    observation: &ClockObservation,
    maximum_uncertainty_ms: u64,
) -> Result<(), ClockViolation> {
    if observation.schema_version != CLOCK_OBSERVATION_SCHEMA {
        return Err(ClockViolation::UnsupportedSchema(
            observation.source_id.clone(),
        ));
    }
    if observation.source_id.trim().is_empty()
        || observation.source_id != observation.source_id.trim()
        || observation.source_id.len() > MAX_CLOCK_SOURCE_ID_BYTES
        || observation.source_id.chars().any(char::is_control)
    {
        return Err(ClockViolation::InvalidSourceId(
            observation.source_id.clone(),
        ));
    }
    if observation.epoch == 0 {
        return Err(ClockViolation::ZeroEpoch(observation.source_id.clone()));
    }
    if observation.uncertainty_ms > maximum_uncertainty_ms {
        return Err(ClockViolation::UncertaintyTooLarge(
            observation.source_id.clone(),
        ));
    }
    if !observation.signature.algorithm.is_canonical()
        || observation.signature.key_id.trim().is_empty()
        || observation.signature.key_id != observation.signature.key_id.trim()
        || observation.signature.signature.is_empty()
    {
        return Err(ClockViolation::SignatureInvalid(
            observation.source_id.clone(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord};

    struct Accept;
    impl ClockObservationVerifier for Accept {
        fn verify_clock_observation(
            &self,
            _algorithm: &SignatureAlgorithm,
            _key_id: &str,
            _message: &[u8],
            _signature: &[u8],
        ) -> Result<bool, String> {
            Ok(true)
        }
    }

    fn snapshot() -> TrustSnapshot {
        let usages = BTreeSet::from([KeyUsage::ClockAuthority]);
        TrustSnapshot::new(
            1,
            1,
            10_000,
            vec![
                KeyTrustRecord {
                    algorithm: SignatureAlgorithm::Ed25519,
                    key_id: "clock-a".into(),
                    not_before_unix_s: 1,
                    not_after_unix_s: None,
                    status: KeyLifecycleStatus::Active,
                    usages: usages.clone(),
                },
                KeyTrustRecord {
                    algorithm: SignatureAlgorithm::MlDsa65,
                    key_id: "clock-b".into(),
                    not_before_unix_s: 1,
                    not_after_unix_s: None,
                    status: KeyLifecycleStatus::Active,
                    usages,
                },
            ],
        )
        .unwrap()
    }

    fn observation(
        source: &str,
        algorithm: SignatureAlgorithm,
        key: &str,
        time: u64,
    ) -> ClockObservation {
        ClockObservation {
            schema_version: CLOCK_OBSERVATION_SCHEMA.into(),
            source_id: source.into(),
            observed_unix_ms: time,
            uncertainty_ms: 100,
            epoch: 7,
            signature: DetachedSignature {
                algorithm,
                key_id: key.into(),
                signature: vec![1],
            },
        }
    }

    #[test]
    fn quorum_derives_intersection_and_tracker_rejects_rollback() {
        let verified = verify_clock_quorum(
            &[
                observation("a", SignatureAlgorithm::Ed25519, "clock-a", 1_000_000),
                observation("b", SignatureAlgorithm::MlDsa65, "clock-b", 1_000_050),
            ],
            &ClockQuorumPolicy::default(),
            &snapshot(),
            100,
            &Accept,
        )
        .unwrap();
        assert_eq!(
            (verified.lower_unix_ms, verified.upper_unix_ms),
            (999_950, 1_000_100)
        );
        let mut tracker = ClockEpochTracker::default();
        tracker.accept(&verified).unwrap();
        let mut older = verified.clone();
        older.epoch = 6;
        assert!(matches!(
            tracker.accept(&older),
            Err(ClockTrackingError::EpochRollback { .. })
        ));
    }

    #[test]
    fn disjoint_time_intervals_fail_closed() {
        let result = verify_clock_quorum(
            &[
                observation("a", SignatureAlgorithm::Ed25519, "clock-a", 1_000),
                observation("b", SignatureAlgorithm::MlDsa65, "clock-b", 2_000),
            ],
            &ClockQuorumPolicy::default(),
            &snapshot(),
            100,
            &Accept,
        );
        assert!(
            result
                .unwrap_err()
                .contains(&ClockViolation::NoCommonInterval)
        );
    }
}
