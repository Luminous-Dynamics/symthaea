// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Permit-aware bootstrap clock acceptance.
//!
//! This is the first layer where original signed clock observations can become
//! an opaque accepted-time capability. The runtime quorum policy and lifecycle
//! evaluation instant are derived from the private permit, never supplied by
//! the caller or candidate clock.

use crate::clock::{ClockObservation, ClockObservationVerifier, ClockViolation, VerifiedClockWindow};
use crate::clock_evaluation_permit::{
    ClockEvaluationPermitError, ClockEvaluationPermitIdV3, ClockEvaluationPermitV3,
    ClockQuorumPolicyRevisionIdV1,
};
use crate::clock_witness::{
    ClockWindowEvaluationWitnessV1, ClockWindowWitnessError,
    verify_clock_quorum_with_witness, verify_clock_window_evaluation_witness,
};
use crate::digest::{Sha256Digest, domain_hash};
use crate::trust::{TrustSnapshot, TrustSnapshotError, digest_trust_snapshot};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const ACCEPTED_CLOCK_BASIS_SCHEMA: &str = "symthaea.trust.accepted-clock-basis.v4";
const ACCEPTED_CLOCK_BASIS_DOMAIN: &[u8] = b"symthaea.trust.accepted-clock-basis.v4\0";
const ACCEPTANCE_KIND_BOOTSTRAP: &str = "Bootstrap";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AcceptedClockBasisIdV4(Sha256Digest);

#[derive(Debug, Clone)]
#[must_use]
pub struct AcceptedClockBasisV4 {
    id: AcceptedClockBasisIdV4,
    permit_id: ClockEvaluationPermitIdV3,
    clock_quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    trust_snapshot_digest: Sha256Digest,
    clock_window_evidence_digest: Sha256Digest,
    clock_window_witness_digest: Sha256Digest,
    epoch: u64,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    consensus_unix_ms: u64,
    window: VerifiedClockWindow,
    witness: ClockWindowEvaluationWitnessV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcceptedClockBasisError {
    PermitInvalid(ClockEvaluationPermitError),
    TrustSnapshotInvalid(TrustSnapshotError),
    TrustSnapshotMismatch,
    QuorumVerificationFailed(Vec<ClockViolation>),
    WitnessInvalid(ClockWindowWitnessError),
    WindowTrustSnapshotMismatch,
    WindowOutsidePermit,
    UnpermittedSigner(String),
    ObservationIntervalOverflow(String),
    ObservationIntervalOutsidePermit(String),
    Encoding(String),
}

impl AcceptedClockBasisIdV4 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl AcceptedClockBasisV4 {
    pub fn id(&self) -> AcceptedClockBasisIdV4 {
        self.id
    }

    pub fn permit_id(&self) -> ClockEvaluationPermitIdV3 {
        self.permit_id
    }

    pub fn clock_quorum_policy_id(&self) -> ClockQuorumPolicyRevisionIdV1 {
        self.clock_quorum_policy_id
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

    pub fn verified_window(&self) -> &VerifiedClockWindow {
        &self.window
    }

    pub fn evaluation_witness(&self) -> &ClockWindowEvaluationWitnessV1 {
        &self.witness
    }
}

pub fn accept_bootstrap_clock_basis_v4(
    permit: &ClockEvaluationPermitV3,
    observations: &[ClockObservation],
    trust_snapshot: &TrustSnapshot,
    verifier: &dyn ClockObservationVerifier,
) -> Result<AcceptedClockBasisV4, AcceptedClockBasisError> {
    let snapshot_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(AcceptedClockBasisError::TrustSnapshotInvalid)?;
    if snapshot_digest != permit.trust_snapshot_digest() {
        return Err(AcceptedClockBasisError::TrustSnapshotMismatch);
    }

    let quorum_policy = permit
        .runtime_quorum_policy()
        .map_err(AcceptedClockBasisError::PermitInvalid)?;

    // This instant is derived solely from pre-candidate permit authority.
    // Whole-envelope permit construction has already proved the snapshot and
    // eligible key pool remain valid through this point.
    let evaluation_time_unix_s = permit.evaluation_upper_unix_ms() / 1_000;

    let (window, witness) = verify_clock_quorum_with_witness(
        observations,
        &quorum_policy,
        trust_snapshot,
        evaluation_time_unix_s,
        verifier,
    )
    .map_err(AcceptedClockBasisError::QuorumVerificationFailed)?;

    let witness_digest = verify_clock_window_evaluation_witness(&window, &witness)
        .map_err(AcceptedClockBasisError::WitnessInvalid)?;

    if window.trust_snapshot_digest != permit.trust_snapshot_digest() {
        return Err(AcceptedClockBasisError::WindowTrustSnapshotMismatch);
    }
    if window.lower_unix_ms < permit.evaluation_lower_unix_ms()
        || window.upper_unix_ms > permit.evaluation_upper_unix_ms()
    {
        return Err(AcceptedClockBasisError::WindowOutsidePermit);
    }

    let eligible_signers = permit
        .eligible_clock_keys()
        .iter()
        .map(|key| (key.algorithm().clone(), key.key_id().to_string()))
        .collect::<BTreeSet<_>>();

    for signer in &witness.signers {
        if !eligible_signers.contains(&(signer.algorithm.clone(), signer.key_id.clone())) {
            return Err(AcceptedClockBasisError::UnpermittedSigner(
                signer.key_id.clone(),
            ));
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
                AcceptedClockBasisError::ObservationIntervalOverflow(
                    observation.source_id.clone(),
                )
            })?;
        if lower < permit.evaluation_lower_unix_ms()
            || upper > permit.evaluation_upper_unix_ms()
        {
            return Err(AcceptedClockBasisError::ObservationIntervalOutsidePermit(
                observation.source_id.clone(),
            ));
        }
    }

    let id = AcceptedClockBasisIdV4(compute_basis_id(
        permit.id(),
        permit.clock_quorum_policy_id(),
        window.trust_snapshot_digest,
        window.evidence_digest,
        witness_digest,
        window.epoch,
        window.lower_unix_ms,
        window.upper_unix_ms,
        window.consensus_unix_ms,
    )?);

    Ok(AcceptedClockBasisV4 {
        id,
        permit_id: permit.id(),
        clock_quorum_policy_id: permit.clock_quorum_policy_id(),
        trust_snapshot_digest: window.trust_snapshot_digest,
        clock_window_evidence_digest: window.evidence_digest,
        clock_window_witness_digest: witness_digest,
        epoch: window.epoch,
        lower_unix_ms: window.lower_unix_ms,
        upper_unix_ms: window.upper_unix_ms,
        consensus_unix_ms: window.consensus_unix_ms,
        window,
        witness,
    })
}

#[allow(clippy::too_many_arguments)]
fn compute_basis_id(
    permit_id: ClockEvaluationPermitIdV3,
    clock_quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
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
            Value::String(ACCEPTANCE_KIND_BOOTSTRAP.to_string()),
        ),
        (
            "clock_quorum_policy_id",
            Value::String(clock_quorum_policy_id.to_hex()),
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
        ("schema", Value::String(ACCEPTED_CLOCK_BASIS_SCHEMA.to_string())),
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
