// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Permit-aware admission of the first accepted clock basis.
//!
//! Callers provide original signed observations, not a precomputed window or
//! witness. The existing single-pass verifier emits both; this module then
//! verifies exact witness reconstruction and compatibility with the private
//! pre-candidate evaluation permit.

use crate::clock::{ClockObservation, ClockObservationVerifier, ClockQuorumPolicy, ClockViolation};
use crate::clock_evaluation_permit::{ClockEvaluationPermitIdV2, ClockEvaluationPermitV2};
use crate::clock_witness::{
    ClockWindowWitnessError, verify_clock_quorum_with_witness,
    verify_clock_window_evaluation_witness,
};
use crate::digest::{Sha256Digest, domain_hash};
use crate::trust::{TrustSnapshot, TrustSnapshotError, digest_trust_snapshot};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const ACCEPTED_CLOCK_BASIS_SCHEMA: &str = "symthaea.trust.accepted-clock-basis.v3";
const ACCEPTED_CLOCK_BASIS_DOMAIN: &[u8] = b"symthaea.trust.accepted-clock-basis.v3\0";
const BOOTSTRAP_ACCEPTANCE_KIND: &str = "Bootstrap";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AcceptedClockBasisIdV3(Sha256Digest);

#[derive(Debug, Clone)]
#[must_use]
pub struct AcceptedClockBasisV3 {
    id: AcceptedClockBasisIdV3,
    permit_id: ClockEvaluationPermitIdV2,
    trust_snapshot_digest: Sha256Digest,
    clock_window_evidence_digest: Sha256Digest,
    clock_window_witness_digest: Sha256Digest,
    epoch: u64,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    consensus_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcceptedClockBasisError {
    TrustSnapshotInvalid(TrustSnapshotError),
    PermitSnapshotMismatch,
    ClockVerification(Vec<ClockViolation>),
    WitnessVerification(ClockWindowWitnessError),
    WindowSnapshotMismatch,
    WindowOutsidePermit,
    UnpermittedSigner(String),
    InsufficientSigners { actual: usize, required: usize },
    AlgorithmDiversityMissing,
    ObservationIntervalOverflow(String),
    ObservationIntervalOutsidePermit(String),
    Encoding(String),
}

impl AcceptedClockBasisIdV3 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl AcceptedClockBasisV3 {
    pub fn id(&self) -> AcceptedClockBasisIdV3 {
        self.id
    }

    pub fn permit_id(&self) -> ClockEvaluationPermitIdV2 {
        self.permit_id
    }

    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }

    pub fn clock_window_evidence_digest(&self) -> Sha256Digest {
        self.clock_window_evidence_digest
    }

    pub fn clock_window_witness_digest(&self) -> Sha256Digest {
        self.clock_window_witness_digest
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn lower_unix_ms(&self) -> u64 {
        self.lower_unix_ms
    }

    pub fn upper_unix_ms(&self) -> u64 {
        self.upper_unix_ms
    }

    pub fn consensus_unix_ms(&self) -> u64 {
        self.consensus_unix_ms
    }
}

pub fn accept_bootstrap_clock_basis_v3(
    permit: &ClockEvaluationPermitV2,
    observations: &[ClockObservation],
    quorum_policy: &ClockQuorumPolicy,
    trust_snapshot: &TrustSnapshot,
    verifier: &dyn ClockObservationVerifier,
) -> Result<AcceptedClockBasisV3, AcceptedClockBasisError> {
    let snapshot_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(AcceptedClockBasisError::TrustSnapshotInvalid)?;
    if snapshot_digest != permit.trust_snapshot_digest() {
        return Err(AcceptedClockBasisError::PermitSnapshotMismatch);
    }

    // This lifecycle instant is derived entirely from pre-candidate permit
    // authority. Candidate observations cannot select it.
    let verifier_evaluation_time_unix_s = permit.evaluation_upper_unix_ms() / 1_000;
    let (window, witness) = verify_clock_quorum_with_witness(
        observations,
        quorum_policy,
        trust_snapshot,
        verifier_evaluation_time_unix_s,
        verifier,
    )
    .map_err(AcceptedClockBasisError::ClockVerification)?;

    let witness_digest = verify_clock_window_evaluation_witness(&window, &witness)
        .map_err(AcceptedClockBasisError::WitnessVerification)?;

    if window.trust_snapshot_digest != permit.trust_snapshot_digest() {
        return Err(AcceptedClockBasisError::WindowSnapshotMismatch);
    }
    if window.lower_unix_ms < permit.evaluation_lower_unix_ms()
        || window.upper_unix_ms > permit.evaluation_upper_unix_ms()
    {
        return Err(AcceptedClockBasisError::WindowOutsidePermit);
    }

    let eligible = permit
        .eligible_clock_keys()
        .iter()
        .map(|key| (key.algorithm().clone(), key.key_id().to_string()))
        .collect::<BTreeSet<_>>();
    let signers = witness
        .signers
        .iter()
        .map(|signer| (signer.algorithm.clone(), signer.key_id.clone()))
        .collect::<BTreeSet<_>>();

    for (_, key_id) in signers.difference(&eligible) {
        return Err(AcceptedClockBasisError::UnpermittedSigner(key_id.clone()));
    }
    if signers.len() < permit.minimum_clock_authority_keys() {
        return Err(AcceptedClockBasisError::InsufficientSigners {
            actual: signers.len(),
            required: permit.minimum_clock_authority_keys(),
        });
    }
    if permit.require_algorithm_diversity() {
        let algorithms = signers
            .iter()
            .map(|(algorithm, _)| algorithm.clone())
            .collect::<BTreeSet<_>>();
        if algorithms.len() < 2 {
            return Err(AcceptedClockBasisError::AlgorithmDiversityMissing);
        }
    }

    for observation in &witness.observations {
        let lower = observation
            .observed_unix_ms
            .saturating_sub(observation.uncertainty_ms);
        let upper = observation
            .observed_unix_ms
            .checked_add(observation.uncertainty_ms)
            .ok_or_else(|| {
                AcceptedClockBasisError::ObservationIntervalOverflow(observation.source_id.clone())
            })?;
        if lower < permit.evaluation_lower_unix_ms()
            || upper > permit.evaluation_upper_unix_ms()
        {
            return Err(AcceptedClockBasisError::ObservationIntervalOutsidePermit(
                observation.source_id.clone(),
            ));
        }
    }

    let id = AcceptedClockBasisIdV3(compute_basis_id(
        permit.id(),
        snapshot_digest,
        window.evidence_digest,
        witness_digest,
        window.epoch,
        window.lower_unix_ms,
        window.upper_unix_ms,
        window.consensus_unix_ms,
    )?);

    Ok(AcceptedClockBasisV3 {
        id,
        permit_id: permit.id(),
        trust_snapshot_digest: snapshot_digest,
        clock_window_evidence_digest: window.evidence_digest,
        clock_window_witness_digest: witness_digest,
        epoch: window.epoch,
        lower_unix_ms: window.lower_unix_ms,
        upper_unix_ms: window.upper_unix_ms,
        consensus_unix_ms: window.consensus_unix_ms,
    })
}

#[allow(clippy::too_many_arguments)]
fn compute_basis_id(
    permit_id: ClockEvaluationPermitIdV2,
    trust_snapshot_digest: Sha256Digest,
    clock_window_evidence_digest: Sha256Digest,
    clock_window_witness_digest: Sha256Digest,
    epoch: u64,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    consensus_unix_ms: u64,
) -> Result<Sha256Digest, AcceptedClockBasisError> {
    let bytes = canonical_json_bytes([
        (
            "acceptance_kind",
            Value::String(BOOTSTRAP_ACCEPTANCE_KIND.to_string()),
        ),
        (
            "clock_window_evidence_digest",
            Value::String(clock_window_evidence_digest.to_hex()),
        ),
        (
            "clock_window_witness_digest",
            Value::String(clock_window_witness_digest.to_hex()),
        ),
        ("consensus_unix_ms", Value::from(consensus_unix_ms)),
        ("epoch", Value::from(epoch)),
        ("lower_unix_ms", Value::from(lower_unix_ms)),
        ("permit_id", Value::String(permit_id.to_hex())),
        (
            "schema",
            Value::String(ACCEPTED_CLOCK_BASIS_SCHEMA.to_string()),
        ),
        (
            "trust_snapshot_digest",
            Value::String(trust_snapshot_digest.to_hex()),
        ),
        ("upper_unix_ms", Value::from(upper_unix_ms)),
    ])?;
    Ok(domain_hash(ACCEPTED_CLOCK_BASIS_DOMAIN, &bytes))
}

fn canonical_json_bytes<const N: usize>(
    fields: [(&str, Value); N],
) -> Result<Vec<u8>, AcceptedClockBasisError> {
    let map = fields
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect::<BTreeMap<_, _>>();
    serde_json::to_vec(&map).map_err(|error| AcceptedClockBasisError::Encoding(error.to_string()))
}
