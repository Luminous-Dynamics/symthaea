// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unified-policy bootstrap clock acceptance.
//!
//! Original signed observations can become an opaque accepted-time capability
//! only through the quorum policy retained by a private V4 evaluation permit.
//! The resulting V5 basis retains that originating permit so successor authority
//! inherits the exact evaluation, quorum, continuity, and trust-snapshot lineage.

use crate::clock::{ClockObservation, ClockObservationVerifier, ClockViolation, VerifiedClockWindow};
use crate::clock_evaluation_permit::{
    ClockContinuityPolicyRevisionIdV1, ClockEvaluationPermitError, ClockEvaluationPermitIdV4,
    ClockEvaluationPermitV4, ClockEvaluationPolicyIdV4, ClockQuorumPolicyRevisionIdV1,
};
use crate::clock_witness::{
    ClockWindowEvaluationWitnessV1, ClockWindowWitnessError,
    verify_clock_quorum_with_witness, verify_clock_window_evaluation_witness,
};
use crate::digest::{domain_hash, Sha256Digest};
use crate::trust::{digest_trust_snapshot, TrustSnapshot, TrustSnapshotError};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const ACCEPTED_CLOCK_BASIS_SCHEMA: &str = "symthaea.trust.accepted-clock-basis.v5";
const ACCEPTED_CLOCK_BASIS_DOMAIN: &[u8] = b"symthaea.trust.accepted-clock-basis.v5\0";
const ACCEPTANCE_KIND_BOOTSTRAP: &str = "Bootstrap";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AcceptedClockBasisIdV5(Sha256Digest);

#[derive(Debug, Clone)]
#[must_use]
pub struct AcceptedClockBasisV5 {
    id: AcceptedClockBasisIdV5,
    clock_evaluation_policy_id: ClockEvaluationPolicyIdV4,
    clock_quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    clock_continuity_policy_id: ClockContinuityPolicyRevisionIdV1,
    trust_snapshot_digest: Sha256Digest,
    clock_window_evidence_digest: Sha256Digest,
    clock_window_witness_digest: Sha256Digest,
    epoch: u64,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    consensus_unix_ms: u64,
    originating_permit: ClockEvaluationPermitV4,
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

impl AcceptedClockBasisIdV5 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl AcceptedClockBasisV5 {
    pub fn id(&self) -> AcceptedClockBasisIdV5 {
        self.id
    }

    pub fn permit_id(&self) -> ClockEvaluationPermitIdV4 {
        self.originating_permit.id()
    }

    pub fn clock_evaluation_policy_id(&self) -> ClockEvaluationPolicyIdV4 {
        self.clock_evaluation_policy_id
    }

    pub fn clock_quorum_policy_id(&self) -> ClockQuorumPolicyRevisionIdV1 {
        self.clock_quorum_policy_id
    }

    pub fn clock_continuity_policy_id(&self) -> ClockContinuityPolicyRevisionIdV1 {
        self.clock_continuity_policy_id
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

    pub fn originating_permit(&self) -> &ClockEvaluationPermitV4 {
        &self.originating_permit
    }

    pub fn verified_window(&self) -> &VerifiedClockWindow {
        &self.window
    }

    pub fn evaluation_witness(&self) -> &ClockWindowEvaluationWitnessV1 {
        &self.witness
    }
}

pub fn accept_bootstrap_clock_basis_v5(
    permit: &ClockEvaluationPermitV4,
    observations: &[ClockObservation],
    trust_snapshot: &TrustSnapshot,
    verifier: &dyn ClockObservationVerifier,
) -> Result<AcceptedClockBasisV5, AcceptedClockBasisError> {
    let snapshot_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(AcceptedClockBasisError::TrustSnapshotInvalid)?;
    if snapshot_digest != permit.trust_snapshot_digest() {
        return Err(AcceptedClockBasisError::TrustSnapshotMismatch);
    }

    let quorum_policy = permit
        .runtime_quorum_policy()
        .map_err(AcceptedClockBasisError::PermitInvalid)?;

    // Candidate observations cannot choose the lifecycle instant used to verify
    // their own signing keys. It is derived from pre-candidate permit authority.
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

    let id = AcceptedClockBasisIdV5(compute_basis_id(
        permit.id(),
        permit.evaluation_policy_id(),
        permit.clock_quorum_policy_id(),
        permit.clock_continuity_policy_id(),
        window.trust_snapshot_digest,
        window.evidence_digest,
        witness_digest,
        window.epoch,
        window.lower_unix_ms,
        window.upper_unix_ms,
        window.consensus_unix_ms,
    )?);

    Ok(AcceptedClockBasisV5 {
        id,
        clock_evaluation_policy_id: permit.evaluation_policy_id(),
        clock_quorum_policy_id: permit.clock_quorum_policy_id(),
        clock_continuity_policy_id: permit.clock_continuity_policy_id(),
        trust_snapshot_digest: window.trust_snapshot_digest,
        clock_window_evidence_digest: window.evidence_digest,
        clock_window_witness_digest: witness_digest,
        epoch: window.epoch,
        lower_unix_ms: window.lower_unix_ms,
        upper_unix_ms: window.upper_unix_ms,
        consensus_unix_ms: window.consensus_unix_ms,
        originating_permit: permit.clone(),
        window,
        witness,
    })
}

#[allow(clippy::too_many_arguments)]
fn compute_basis_id(
    permit_id: ClockEvaluationPermitIdV4,
    clock_evaluation_policy_id: ClockEvaluationPolicyIdV4,
    clock_quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    clock_continuity_policy_id: ClockContinuityPolicyRevisionIdV1,
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
            "clock_continuity_policy_id",
            Value::String(clock_continuity_policy_id.to_hex()),
        ),
        (
            "clock_evaluation_policy_id",
            Value::String(clock_evaluation_policy_id.to_hex()),
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
