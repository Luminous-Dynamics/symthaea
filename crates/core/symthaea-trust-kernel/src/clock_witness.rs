// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verifier-emitted companion evidence for a frozen V1 clock window.
//!
//! `VerifiedClockWindow` intentionally remains byte/protocol compatible with
//! the fabrication V1 surface. This module records the exact accepted
//! observation/signer witness needed by later non-circular clock-evaluation
//! policy without mutating that legacy protocol.

use crate::clock::{
    CLOCK_OBSERVATION_SCHEMA, ClockObservation, ClockObservationVerifier, ClockQuorumPolicy,
    ClockViolation, MAX_CLOCK_OBSERVATIONS, VerifiedClockWindow, canonical_clock_observation_bytes,
    digest_clock_observation,
};
use crate::digest::Sha256Digest;
use crate::signature::SignatureAlgorithm;
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

pub const CLOCK_WINDOW_EVALUATION_WITNESS_SCHEMA: &str =
    "symthaea.trust.clock-window-evaluation-witness.v1";
const CLOCK_WINDOW_EVALUATION_WITNESS_DOMAIN: &[u8] =
    b"symthaea.trust.clock-window-evaluation-witness-digest.v1\0";
const LEGACY_VERIFIED_CLOCK_WINDOW_DOMAIN: &[u8] =
    b"symthaea.fabrication.verified-clock-window.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ClockSignerV1 {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptedClockObservationV1 {
    pub source_id: String,
    pub observed_unix_ms: u64,
    pub uncertainty_ms: u64,
    pub epoch: u64,
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub observation_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockWindowEvaluationWitnessV1 {
    pub schema_version: String,
    pub window_evidence_digest: Sha256Digest,
    pub trust_snapshot_digest: Sha256Digest,
    pub epoch: u64,
    pub accepted_observation_digests: Vec<Sha256Digest>,
    pub signers: Vec<ClockSignerV1>,
    pub observations: Vec<AcceptedClockObservationV1>,
    pub witness_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockWindowWitnessError {
    UnsupportedSchema,
    InvalidShape,
    NonCanonicalObservationOrder,
    NonCanonicalObservationDigestOrder,
    NonCanonicalSignerOrder,
    ObservationDigestMismatch(String),
    ObservationSetMismatch,
    SignerSetMismatch,
    WindowBindingMismatch,
    WitnessDigestMismatch,
}

/// Single-pass clock verification that emits both the frozen V1 window and the
/// exact companion witness. The signature verifier is called exactly once for
/// each candidate observation considered by this invocation.
pub fn verify_clock_quorum_with_witness(
    observations: &[ClockObservation],
    policy: &ClockQuorumPolicy,
    trust_snapshot: &TrustSnapshot,
    evaluation_time_unix_s: u64,
    verifier: &dyn ClockObservationVerifier,
) -> Result<(VerifiedClockWindow, ClockWindowEvaluationWitnessV1), Vec<ClockViolation>> {
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
    let mut accepted_observations = Vec::new();

    for observation in observations {
        if let Err(violation) = validate_observation_shape_for_witness(
            observation,
            policy.maximum_uncertainty_ms,
        ) {
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
            Ok(digest) => {
                accepted_digests.push(digest);
                accepted_observations.push(AcceptedClockObservationV1 {
                    source_id: observation.source_id.clone(),
                    observed_unix_ms: observation.observed_unix_ms,
                    uncertainty_ms: observation.uncertainty_ms,
                    epoch: observation.epoch,
                    algorithm: observation.signature.algorithm.clone(),
                    key_id: observation.signature.key_id.clone(),
                    observation_digest: digest,
                });
            }
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
    accepted_observations.sort_by_key(|observation| observation.observation_digest);
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(|error| vec![ClockViolation::Encoding(format!("{error:?}"))])?;
    let consensus_unix_ms = lower + (upper - lower) / 2;
    let source_ids = accepted_sources.into_iter().collect::<Vec<_>>();
    let algorithms = algorithms.into_iter().collect::<Vec<_>>();
    let expected_epoch = expected_epoch.unwrap_or(0);
    let evidence_digest = legacy_window_digest(
        expected_epoch,
        lower,
        upper,
        trust_snapshot_digest,
        &accepted_digests,
    );

    let window = VerifiedClockWindow {
        lower_unix_ms: lower,
        upper_unix_ms: upper,
        consensus_unix_ms,
        epoch: expected_epoch,
        source_ids,
        algorithms,
        trust_snapshot_digest,
        evidence_digest,
    };
    let witness = build_witness(&window, accepted_observations, accepted_digests);
    Ok((window, witness))
}

/// Verify that the companion witness reconstructs the exact frozen V1 window
/// and that its own content-addressed identity is intact.
pub fn verify_clock_window_evaluation_witness(
    window: &VerifiedClockWindow,
    witness: &ClockWindowEvaluationWitnessV1,
) -> Result<Sha256Digest, ClockWindowWitnessError> {
    validate_witness_shape(witness)?;

    let mut recomputed_digests = Vec::with_capacity(witness.observations.len());
    let mut recomputed_signers = BTreeSet::new();
    let mut sources = BTreeSet::new();
    let mut algorithms = BTreeSet::new();
    let mut lower = 0u64;
    let mut upper = u64::MAX;

    for observation in &witness.observations {
        if observation.epoch != witness.epoch {
            return Err(ClockWindowWitnessError::InvalidShape);
        }
        let synthetic = ClockObservation {
            schema_version: CLOCK_OBSERVATION_SCHEMA.to_string(),
            source_id: observation.source_id.clone(),
            observed_unix_ms: observation.observed_unix_ms,
            uncertainty_ms: observation.uncertainty_ms,
            epoch: observation.epoch,
            signature: crate::signature::DetachedSignature {
                algorithm: observation.algorithm.clone(),
                key_id: observation.key_id.clone(),
                signature: vec![1],
            },
        };
        let digest = digest_clock_observation(&synthetic)
            .map_err(|_| ClockWindowWitnessError::ObservationDigestMismatch(
                observation.source_id.clone(),
            ))?;
        if digest != observation.observation_digest {
            return Err(ClockWindowWitnessError::ObservationDigestMismatch(
                observation.source_id.clone(),
            ));
        }
        let observation_upper = observation
            .observed_unix_ms
            .checked_add(observation.uncertainty_ms)
            .ok_or(ClockWindowWitnessError::InvalidShape)?;
        lower = lower.max(
            observation
                .observed_unix_ms
                .saturating_sub(observation.uncertainty_ms),
        );
        upper = upper.min(observation_upper);
        if !sources.insert(observation.source_id.clone()) {
            return Err(ClockWindowWitnessError::InvalidShape);
        }
        let signer = ClockSignerV1 {
            algorithm: observation.algorithm.clone(),
            key_id: observation.key_id.clone(),
        };
        if !recomputed_signers.insert(signer) {
            return Err(ClockWindowWitnessError::InvalidShape);
        }
        algorithms.insert(observation.algorithm.clone());
        recomputed_digests.push(digest);
    }

    recomputed_digests.sort();
    if recomputed_digests != witness.accepted_observation_digests {
        return Err(ClockWindowWitnessError::ObservationSetMismatch);
    }
    if recomputed_signers.into_iter().collect::<Vec<_>>() != witness.signers {
        return Err(ClockWindowWitnessError::SignerSetMismatch);
    }
    if lower > upper {
        return Err(ClockWindowWitnessError::InvalidShape);
    }

    let source_ids = sources.into_iter().collect::<Vec<_>>();
    let algorithms = algorithms.into_iter().collect::<Vec<_>>();
    let expected_window_digest = legacy_window_digest(
        witness.epoch,
        lower,
        upper,
        witness.trust_snapshot_digest,
        &witness.accepted_observation_digests,
    );
    let consensus = lower + (upper - lower) / 2;
    if witness.window_evidence_digest != expected_window_digest
        || window.evidence_digest != expected_window_digest
        || window.trust_snapshot_digest != witness.trust_snapshot_digest
        || window.epoch != witness.epoch
        || window.lower_unix_ms != lower
        || window.upper_unix_ms != upper
        || window.consensus_unix_ms != consensus
        || window.source_ids != source_ids
        || window.algorithms != algorithms
    {
        return Err(ClockWindowWitnessError::WindowBindingMismatch);
    }

    let expected_witness_digest = compute_witness_digest(witness);
    if witness.witness_digest != expected_witness_digest {
        return Err(ClockWindowWitnessError::WitnessDigestMismatch);
    }
    Ok(expected_witness_digest)
}

pub fn digest_clock_window_evaluation_witness(
    witness: &ClockWindowEvaluationWitnessV1,
) -> Result<Sha256Digest, ClockWindowWitnessError> {
    validate_witness_shape(witness)?;
    let expected = compute_witness_digest(witness);
    if witness.witness_digest != expected {
        return Err(ClockWindowWitnessError::WitnessDigestMismatch);
    }
    Ok(expected)
}

fn build_witness(
    window: &VerifiedClockWindow,
    observations: Vec<AcceptedClockObservationV1>,
    accepted_observation_digests: Vec<Sha256Digest>,
) -> ClockWindowEvaluationWitnessV1 {
    let signers = observations
        .iter()
        .map(|observation| ClockSignerV1 {
            algorithm: observation.algorithm.clone(),
            key_id: observation.key_id.clone(),
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut witness = ClockWindowEvaluationWitnessV1 {
        schema_version: CLOCK_WINDOW_EVALUATION_WITNESS_SCHEMA.to_string(),
        window_evidence_digest: window.evidence_digest,
        trust_snapshot_digest: window.trust_snapshot_digest,
        epoch: window.epoch,
        accepted_observation_digests,
        signers,
        observations,
        witness_digest: Sha256Digest([0; 32]),
    };
    witness.witness_digest = compute_witness_digest(&witness);
    witness
}

fn validate_witness_shape(
    witness: &ClockWindowEvaluationWitnessV1,
) -> Result<(), ClockWindowWitnessError> {
    if witness.schema_version != CLOCK_WINDOW_EVALUATION_WITNESS_SCHEMA {
        return Err(ClockWindowWitnessError::UnsupportedSchema);
    }
    if witness.epoch == 0
        || witness.window_evidence_digest.0 == [0; 32]
        || witness.trust_snapshot_digest.0 == [0; 32]
        || witness.witness_digest.0 == [0; 32]
        || witness.observations.is_empty()
        || witness.observations.len() > MAX_CLOCK_OBSERVATIONS
        || witness.accepted_observation_digests.len() != witness.observations.len()
        || witness.signers.len() != witness.observations.len()
    {
        return Err(ClockWindowWitnessError::InvalidShape);
    }
    if witness
        .accepted_observation_digests
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(ClockWindowWitnessError::NonCanonicalObservationDigestOrder);
    }
    if witness.signers.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(ClockWindowWitnessError::NonCanonicalSignerOrder);
    }
    if witness
        .observations
        .windows(2)
        .any(|pair| pair[0].observation_digest >= pair[1].observation_digest)
    {
        return Err(ClockWindowWitnessError::NonCanonicalObservationOrder);
    }
    Ok(())
}

fn legacy_window_digest(
    epoch: u64,
    lower: u64,
    upper: u64,
    trust_snapshot_digest: Sha256Digest,
    accepted_digests: &[Sha256Digest],
) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(LEGACY_VERIFIED_CLOCK_WINDOW_DOMAIN);
    hasher.update(epoch.to_le_bytes());
    hasher.update(lower.to_le_bytes());
    hasher.update(upper.to_le_bytes());
    hasher.update(trust_snapshot_digest.0);
    for digest in accepted_digests {
        hasher.update(digest.0);
    }
    Sha256Digest(hasher.finalize().into())
}

fn compute_witness_digest(witness: &ClockWindowEvaluationWitnessV1) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(CLOCK_WINDOW_EVALUATION_WITNESS_DOMAIN);
    update_string(&mut hasher, &witness.schema_version);
    hasher.update(witness.window_evidence_digest.0);
    hasher.update(witness.trust_snapshot_digest.0);
    hasher.update(witness.epoch.to_le_bytes());
    update_count(&mut hasher, witness.accepted_observation_digests.len());
    for digest in &witness.accepted_observation_digests {
        hasher.update(digest.0);
    }
    update_count(&mut hasher, witness.signers.len());
    for signer in &witness.signers {
        update_algorithm(&mut hasher, &signer.algorithm);
        update_string(&mut hasher, &signer.key_id);
    }
    update_count(&mut hasher, witness.observations.len());
    for observation in &witness.observations {
        update_string(&mut hasher, &observation.source_id);
        hasher.update(observation.observed_unix_ms.to_le_bytes());
        hasher.update(observation.uncertainty_ms.to_le_bytes());
        hasher.update(observation.epoch.to_le_bytes());
        update_algorithm(&mut hasher, &observation.algorithm);
        update_string(&mut hasher, &observation.key_id);
        hasher.update(observation.observation_digest.0);
    }
    Sha256Digest(hasher.finalize().into())
}

fn update_count(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn update_string(hasher: &mut Sha256, value: &str) {
    update_count(hasher, value.len());
    hasher.update(value.as_bytes());
}

fn update_algorithm(hasher: &mut Sha256, algorithm: &SignatureAlgorithm) {
    match algorithm {
        SignatureAlgorithm::Ed25519 => hasher.update([0]),
        SignatureAlgorithm::MlDsa65 => hasher.update([1]),
        SignatureAlgorithm::MlDsa87 => hasher.update([2]),
        SignatureAlgorithm::Other(name) => {
            hasher.update([3]);
            update_string(hasher, name);
        }
    }
}

fn validate_observation_shape_for_witness(
    observation: &ClockObservation,
    maximum_uncertainty_ms: u64,
) -> Result<(), ClockViolation> {
    if observation.schema_version != CLOCK_OBSERVATION_SCHEMA {
        return Err(ClockViolation::UnsupportedSchema(observation.source_id.clone()));
    }
    if observation.source_id.trim().is_empty()
        || observation.source_id != observation.source_id.trim()
        || observation.source_id.len() > crate::clock::MAX_CLOCK_SOURCE_ID_BYTES
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
