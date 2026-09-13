// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Normal-operation successor clock authority derived from a prior accepted basis.
//!
//! Caller-supplied policy/snapshot records are witness material only at the
//! binding boundary. They must reproduce identities already committed by the
//! prior opaque `AcceptedClockBasisV5` before an opaque successor context can
//! exist. Successor permit derivation and successor acceptance then take no
//! caller policy, snapshot, lifecycle time, precomputed window, or witness.

use crate::accepted_clock_basis::{AcceptedClockBasisIdV5, AcceptedClockBasisV5};
use crate::clock::{ClockObservation, ClockObservationVerifier, ClockViolation, VerifiedClockWindow};
use crate::clock_evaluation_permit::{
    ClockContinuityPolicyRevisionIdV1, ClockContinuityPolicyRevisionV1,
    ClockEvaluationPermitError, ClockEvaluationPolicyIdV4, ClockEvaluationPolicyV4,
    ClockQuorumPolicyRevisionIdV1, ClockQuorumPolicyRevisionV1,
};
use crate::clock_witness::{
    ClockWindowEvaluationWitnessV1, ClockWindowWitnessError,
    verify_clock_quorum_with_witness, verify_clock_window_evaluation_witness,
};
use crate::continuity::{
    ClockContinuityError, VerifiedClockContinuity, verify_clock_continuity,
};
use crate::digest::{Sha256Digest, domain_hash};
use crate::signature::SignatureAlgorithm;
use crate::trust::{
    KeyLifecycleStatus, KeyUsage, TrustSnapshot, TrustSnapshotError, digest_trust_snapshot,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const CLOCK_SUCCESSOR_EVALUATION_PERMIT_SCHEMA: &str =
    "symthaea.trust.clock-successor-evaluation-permit.v1";
pub const ACCEPTED_CLOCK_SUCCESSOR_BASIS_SCHEMA: &str =
    "symthaea.trust.accepted-clock-basis.v6";

const CLOCK_SUCCESSOR_EVALUATION_PERMIT_DOMAIN: &[u8] =
    b"symthaea.trust.clock-successor-evaluation-permit.v1\0";
const ACCEPTED_CLOCK_SUCCESSOR_BASIS_DOMAIN: &[u8] =
    b"symthaea.trust.accepted-clock-basis.v6\0";
const ACCEPTANCE_KIND_CONTINUOUS: &str = "Continuous";

#[derive(Debug, Clone)]
#[must_use]
pub struct ClockSuccessorAuthorityContextV1 {
    prior_basis_id: AcceptedClockBasisIdV5,
    prior_window: VerifiedClockWindow,
    evaluation_policy: ClockEvaluationPolicyV4,
    quorum_policy: ClockQuorumPolicyRevisionV1,
    continuity_policy: ClockContinuityPolicyRevisionV1,
    trust_snapshot: TrustSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuccessorClockAuthorityKeyV1 {
    algorithm: SignatureAlgorithm,
    key_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockSuccessorEvaluationPermitIdV1(Sha256Digest);

#[derive(Debug, Clone)]
#[must_use]
pub struct ClockSuccessorEvaluationPermitV1 {
    id: ClockSuccessorEvaluationPermitIdV1,
    prior_basis_id: AcceptedClockBasisIdV5,
    prior_clock_window_evidence_digest: Sha256Digest,
    evaluation_policy_id: ClockEvaluationPolicyIdV4,
    clock_quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    clock_continuity_policy_id: ClockContinuityPolicyRevisionIdV1,
    trust_snapshot_digest: Sha256Digest,
    evaluation_lower_unix_ms: u64,
    evaluation_upper_unix_ms: u64,
    minimum_eligible_clock_authority_keys: usize,
    require_eligible_algorithm_diversity: bool,
    eligible_clock_keys: Vec<SuccessorClockAuthorityKeyV1>,
    context: ClockSuccessorAuthorityContextV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AcceptedClockBasisIdV6(Sha256Digest);

#[derive(Debug, Clone)]
#[must_use]
pub struct AcceptedClockBasisV6 {
    id: AcceptedClockBasisIdV6,
    successor_permit: ClockSuccessorEvaluationPermitV1,
    prior_basis_id: AcceptedClockBasisIdV5,
    clock_evaluation_policy_id: ClockEvaluationPolicyIdV4,
    clock_quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    clock_continuity_policy_id: ClockContinuityPolicyRevisionIdV1,
    trust_snapshot_digest: Sha256Digest,
    prior_clock_window_evidence_digest: Sha256Digest,
    clock_window_evidence_digest: Sha256Digest,
    clock_window_witness_digest: Sha256Digest,
    clock_continuity_digest: Sha256Digest,
    epoch: u64,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    consensus_unix_ms: u64,
    window: VerifiedClockWindow,
    witness: ClockWindowEvaluationWitnessV1,
    continuity: VerifiedClockContinuity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockSuccessorError {
    EvaluationPolicyInvalid(ClockEvaluationPermitError),
    EvaluationPolicyMismatch,
    QuorumPolicyMismatch,
    ContinuityPolicyMismatch,
    TrustSnapshotInvalid(TrustSnapshotError),
    TrustSnapshotMismatch,
    PriorWindowInvalid,
    PriorWindowMismatch,
    PriorWitnessInvalid(ClockWindowWitnessError),
    PriorWitnessMismatch,
    TransitionEnvelopeOverflow,
    TimeScaleOverflow,
    SnapshotNotValidForEnvelope,
    InsufficientClockAuthorityKeys { actual: usize, required: usize },
    EligibleAlgorithmDiversityMissing,
    PermitIdentityMismatch,
    QuorumVerificationFailed(Vec<ClockViolation>),
    SuccessorWitnessInvalid(ClockWindowWitnessError),
    WindowTrustSnapshotMismatch,
    WindowOutsidePermit,
    UnpermittedSigner(String),
    ObservationIntervalOverflow(String),
    ObservationIntervalOutsidePermit(String),
    ContinuityInvalid(ClockContinuityError),
    Encoding(String),
}

impl SuccessorClockAuthorityKeyV1 {
    pub fn algorithm(&self) -> &SignatureAlgorithm {
        &self.algorithm
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }
}

impl ClockSuccessorEvaluationPermitIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl AcceptedClockBasisIdV6 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl ClockSuccessorAuthorityContextV1 {
    pub fn prior_basis_id(&self) -> AcceptedClockBasisIdV5 {
        self.prior_basis_id
    }

    pub fn clock_evaluation_policy_id(&self) -> ClockEvaluationPolicyIdV4 {
        self.evaluation_policy.id()
    }

    pub fn clock_quorum_policy_id(&self) -> ClockQuorumPolicyRevisionIdV1 {
        self.quorum_policy.id()
    }

    pub fn clock_continuity_policy_id(&self) -> ClockContinuityPolicyRevisionIdV1 {
        self.continuity_policy.id()
    }

    pub fn trust_snapshot_digest(&self) -> Result<Sha256Digest, ClockSuccessorError> {
        digest_trust_snapshot(&self.trust_snapshot).map_err(ClockSuccessorError::TrustSnapshotInvalid)
    }
}

impl ClockSuccessorEvaluationPermitV1 {
    pub fn id(&self) -> ClockSuccessorEvaluationPermitIdV1 {
        self.id
    }

    pub fn prior_basis_id(&self) -> AcceptedClockBasisIdV5 {
        self.prior_basis_id
    }

    pub fn prior_clock_window_evidence_digest(&self) -> Sha256Digest {
        self.prior_clock_window_evidence_digest
    }

    pub fn clock_evaluation_policy_id(&self) -> ClockEvaluationPolicyIdV4 {
        self.evaluation_policy_id
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

    pub fn evaluation_lower_unix_ms(&self) -> u64 {
        self.evaluation_lower_unix_ms
    }

    pub fn evaluation_upper_unix_ms(&self) -> u64 {
        self.evaluation_upper_unix_ms
    }

    pub fn eligible_clock_keys(&self) -> &[SuccessorClockAuthorityKeyV1] {
        &self.eligible_clock_keys
    }
}

impl AcceptedClockBasisV6 {
    pub fn id(&self) -> AcceptedClockBasisIdV6 {
        self.id
    }

    pub fn successor_permit_id(&self) -> ClockSuccessorEvaluationPermitIdV1 {
        self.successor_permit.id()
    }

    pub fn prior_basis_id(&self) -> AcceptedClockBasisIdV5 {
        self.prior_basis_id
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

    pub fn prior_clock_window_evidence_digest(&self) -> Sha256Digest {
        self.prior_clock_window_evidence_digest
    }

    pub fn clock_window_evidence_digest(&self) -> Sha256Digest {
        self.clock_window_evidence_digest
    }

    pub fn clock_window_witness_digest(&self) -> Sha256Digest {
        self.clock_window_witness_digest
    }

    pub fn clock_continuity_digest(&self) -> Sha256Digest {
        self.clock_continuity_digest
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

    pub fn verified_continuity(&self) -> &VerifiedClockContinuity {
        &self.continuity
    }
}

/// Bind caller-provided witness records to authority already committed by the
/// prior V5 capability. These records do not grant authority by themselves.
pub fn bind_clock_successor_authority_context_v1(
    prior_basis: &AcceptedClockBasisV5,
    evaluation_policy: &ClockEvaluationPolicyV4,
    trust_snapshot: &TrustSnapshot,
) -> Result<ClockSuccessorAuthorityContextV1, ClockSuccessorError> {
    evaluation_policy
        .validate()
        .map_err(ClockSuccessorError::EvaluationPolicyInvalid)?;
    if evaluation_policy.id() != prior_basis.clock_evaluation_policy_id() {
        return Err(ClockSuccessorError::EvaluationPolicyMismatch);
    }
    if evaluation_policy.clock_quorum_policy_id() != prior_basis.clock_quorum_policy_id() {
        return Err(ClockSuccessorError::QuorumPolicyMismatch);
    }
    if evaluation_policy.clock_continuity_policy_id() != prior_basis.clock_continuity_policy_id() {
        return Err(ClockSuccessorError::ContinuityPolicyMismatch);
    }

    let originating_permit = prior_basis.originating_permit();
    if originating_permit.evaluation_policy_id() != prior_basis.clock_evaluation_policy_id() {
        return Err(ClockSuccessorError::EvaluationPolicyMismatch);
    }
    if originating_permit.clock_quorum_policy_id() != prior_basis.clock_quorum_policy_id() {
        return Err(ClockSuccessorError::QuorumPolicyMismatch);
    }
    if originating_permit.clock_continuity_policy_id() != prior_basis.clock_continuity_policy_id() {
        return Err(ClockSuccessorError::ContinuityPolicyMismatch);
    }
    originating_permit
        .clock_quorum_policy()
        .validate()
        .map_err(ClockSuccessorError::EvaluationPolicyInvalid)?;
    originating_permit
        .clock_continuity_policy()
        .validate()
        .map_err(ClockSuccessorError::EvaluationPolicyInvalid)?;
    if originating_permit.clock_quorum_policy().id() != prior_basis.clock_quorum_policy_id() {
        return Err(ClockSuccessorError::QuorumPolicyMismatch);
    }
    if originating_permit.clock_continuity_policy().id() != prior_basis.clock_continuity_policy_id() {
        return Err(ClockSuccessorError::ContinuityPolicyMismatch);
    }

    trust_snapshot
        .validate()
        .map_err(ClockSuccessorError::TrustSnapshotInvalid)?;
    let snapshot_digest =
        digest_trust_snapshot(trust_snapshot).map_err(ClockSuccessorError::TrustSnapshotInvalid)?;
    if snapshot_digest != prior_basis.trust_snapshot_digest()
        || snapshot_digest != originating_permit.trust_snapshot_digest()
    {
        return Err(ClockSuccessorError::TrustSnapshotMismatch);
    }

    let prior_window = prior_basis.verified_window();
    prior_window
        .validate()
        .map_err(|_| ClockSuccessorError::PriorWindowInvalid)?;
    if prior_window.evidence_digest != prior_basis.clock_window_evidence_digest()
        || prior_window.trust_snapshot_digest != snapshot_digest
        || prior_window.epoch != prior_basis.epoch()
        || prior_window.lower_unix_ms != prior_basis.lower_unix_ms()
        || prior_window.upper_unix_ms != prior_basis.upper_unix_ms()
        || prior_window.consensus_unix_ms != prior_basis.consensus_unix_ms()
    {
        return Err(ClockSuccessorError::PriorWindowMismatch);
    }
    let prior_witness_digest = verify_clock_window_evaluation_witness(
        prior_window,
        prior_basis.evaluation_witness(),
    )
    .map_err(ClockSuccessorError::PriorWitnessInvalid)?;
    if prior_witness_digest != prior_basis.clock_window_witness_digest() {
        return Err(ClockSuccessorError::PriorWitnessMismatch);
    }

    Ok(ClockSuccessorAuthorityContextV1 {
        prior_basis_id: prior_basis.id(),
        prior_window: prior_window.clone(),
        evaluation_policy: evaluation_policy.clone(),
        quorum_policy: originating_permit.clock_quorum_policy().clone(),
        continuity_policy: originating_permit.clock_continuity_policy().clone(),
        trust_snapshot: trust_snapshot.clone(),
    })
}

/// Derive a pre-candidate successor permit solely from an already-bound prior
/// authority context. No candidate clock evidence participates in this step.
pub fn derive_clock_successor_evaluation_permit_v1(
    context: &ClockSuccessorAuthorityContextV1,
) -> Result<ClockSuccessorEvaluationPermitV1, ClockSuccessorError> {
    validate_context(context)?;
    let lower_unix_ms = context.prior_window.lower_unix_ms;
    let upper_unix_ms = context
        .prior_window
        .upper_unix_ms
        .checked_add(context.evaluation_policy.max_transition_ms())
        .ok_or(ClockSuccessorError::TransitionEnvelopeOverflow)?;
    let snapshot_digest =
        digest_trust_snapshot(&context.trust_snapshot).map_err(ClockSuccessorError::TrustSnapshotInvalid)?;
    let eligible_clock_keys = eligible_clock_authority_keys_for_envelope(
        &context.trust_snapshot,
        lower_unix_ms,
        upper_unix_ms,
        &context.evaluation_policy,
    )?;
    let id = ClockSuccessorEvaluationPermitIdV1(compute_successor_permit_id(
        context.prior_basis_id,
        context.prior_window.evidence_digest,
        context.evaluation_policy.id(),
        context.quorum_policy.id(),
        context.continuity_policy.id(),
        snapshot_digest,
        lower_unix_ms,
        upper_unix_ms,
        context.evaluation_policy.minimum_eligible_clock_authority_keys(),
        context.evaluation_policy.require_eligible_algorithm_diversity(),
        &eligible_clock_keys,
    )?);

    Ok(ClockSuccessorEvaluationPermitV1 {
        id,
        prior_basis_id: context.prior_basis_id,
        prior_clock_window_evidence_digest: context.prior_window.evidence_digest,
        evaluation_policy_id: context.evaluation_policy.id(),
        clock_quorum_policy_id: context.quorum_policy.id(),
        clock_continuity_policy_id: context.continuity_policy.id(),
        trust_snapshot_digest: snapshot_digest,
        evaluation_lower_unix_ms: lower_unix_ms,
        evaluation_upper_unix_ms: upper_unix_ms,
        minimum_eligible_clock_authority_keys: context
            .evaluation_policy
            .minimum_eligible_clock_authority_keys(),
        require_eligible_algorithm_diversity: context
            .evaluation_policy
            .require_eligible_algorithm_diversity(),
        eligible_clock_keys,
        context: context.clone(),
    })
}

/// Verify original signed successor observations under the pre-candidate permit,
/// then prove exact continuity to the prior accepted window and mint V6.
pub fn accept_successor_clock_basis_v6(
    permit: &ClockSuccessorEvaluationPermitV1,
    observations: &[ClockObservation],
    verifier: &dyn ClockObservationVerifier,
) -> Result<AcceptedClockBasisV6, ClockSuccessorError> {
    validate_successor_permit(permit)?;
    let quorum_policy = permit
        .context
        .quorum_policy
        .to_runtime_policy()
        .map_err(ClockSuccessorError::EvaluationPolicyInvalid)?;
    let evaluation_time_unix_s = permit.evaluation_upper_unix_ms / 1_000;
    let (window, witness) = verify_clock_quorum_with_witness(
        observations,
        &quorum_policy,
        &permit.context.trust_snapshot,
        evaluation_time_unix_s,
        verifier,
    )
    .map_err(ClockSuccessorError::QuorumVerificationFailed)?;
    let witness_digest = verify_clock_window_evaluation_witness(&window, &witness)
        .map_err(ClockSuccessorError::SuccessorWitnessInvalid)?;

    if window.trust_snapshot_digest != permit.trust_snapshot_digest {
        return Err(ClockSuccessorError::WindowTrustSnapshotMismatch);
    }
    if window.lower_unix_ms < permit.evaluation_lower_unix_ms
        || window.upper_unix_ms > permit.evaluation_upper_unix_ms
    {
        return Err(ClockSuccessorError::WindowOutsidePermit);
    }

    let eligible_signers = permit
        .eligible_clock_keys
        .iter()
        .map(|key| (key.algorithm.clone(), key.key_id.clone()))
        .collect::<BTreeSet<_>>();
    for signer in &witness.signers {
        if !eligible_signers.contains(&(signer.algorithm.clone(), signer.key_id.clone())) {
            return Err(ClockSuccessorError::UnpermittedSigner(signer.key_id.clone()));
        }
    }
    for observation in &witness.observations {
        let lower = observation
            .observed_unix_ms
            .saturating_sub(observation.uncertainty_ms);
        let upper = observation
            .observed_unix_ms
            .checked_add(observation.uncertainty_ms)
            .ok_or_else(|| ClockSuccessorError::ObservationIntervalOverflow(
                observation.source_id.clone(),
            ))?;
        if lower < permit.evaluation_lower_unix_ms || upper > permit.evaluation_upper_unix_ms {
            return Err(ClockSuccessorError::ObservationIntervalOutsidePermit(
                observation.source_id.clone(),
            ));
        }
    }

    let continuity_policy = permit
        .context
        .continuity_policy
        .to_runtime_policy()
        .map_err(ClockSuccessorError::EvaluationPolicyInvalid)?;
    let continuity = verify_clock_continuity(
        &permit.context.prior_window,
        &window,
        &continuity_policy,
    )
    .map_err(ClockSuccessorError::ContinuityInvalid)?;

    let id = AcceptedClockBasisIdV6(compute_successor_basis_id(
        permit.id,
        permit.prior_basis_id,
        permit.evaluation_policy_id,
        permit.clock_quorum_policy_id,
        permit.clock_continuity_policy_id,
        permit.trust_snapshot_digest,
        permit.prior_clock_window_evidence_digest,
        window.evidence_digest,
        witness_digest,
        continuity.continuity_digest,
        window.epoch,
        window.lower_unix_ms,
        window.upper_unix_ms,
        window.consensus_unix_ms,
    )?);

    Ok(AcceptedClockBasisV6 {
        id,
        successor_permit: permit.clone(),
        prior_basis_id: permit.prior_basis_id,
        clock_evaluation_policy_id: permit.evaluation_policy_id,
        clock_quorum_policy_id: permit.clock_quorum_policy_id,
        clock_continuity_policy_id: permit.clock_continuity_policy_id,
        trust_snapshot_digest: permit.trust_snapshot_digest,
        prior_clock_window_evidence_digest: permit.prior_clock_window_evidence_digest,
        clock_window_evidence_digest: window.evidence_digest,
        clock_window_witness_digest: witness_digest,
        clock_continuity_digest: continuity.continuity_digest,
        epoch: window.epoch,
        lower_unix_ms: window.lower_unix_ms,
        upper_unix_ms: window.upper_unix_ms,
        consensus_unix_ms: window.consensus_unix_ms,
        window,
        witness,
        continuity,
    })
}

fn validate_context(context: &ClockSuccessorAuthorityContextV1) -> Result<(), ClockSuccessorError> {
    context
        .evaluation_policy
        .validate()
        .map_err(ClockSuccessorError::EvaluationPolicyInvalid)?;
    context
        .quorum_policy
        .validate()
        .map_err(ClockSuccessorError::EvaluationPolicyInvalid)?;
    context
        .continuity_policy
        .validate()
        .map_err(ClockSuccessorError::EvaluationPolicyInvalid)?;
    if context.evaluation_policy.clock_quorum_policy_id() != context.quorum_policy.id() {
        return Err(ClockSuccessorError::QuorumPolicyMismatch);
    }
    if context.evaluation_policy.clock_continuity_policy_id() != context.continuity_policy.id() {
        return Err(ClockSuccessorError::ContinuityPolicyMismatch);
    }
    context
        .trust_snapshot
        .validate()
        .map_err(ClockSuccessorError::TrustSnapshotInvalid)?;
    let digest = digest_trust_snapshot(&context.trust_snapshot)
        .map_err(ClockSuccessorError::TrustSnapshotInvalid)?;
    if digest != context.prior_window.trust_snapshot_digest {
        return Err(ClockSuccessorError::TrustSnapshotMismatch);
    }
    context
        .prior_window
        .validate()
        .map_err(|_| ClockSuccessorError::PriorWindowInvalid)?;
    Ok(())
}

fn validate_successor_permit(
    permit: &ClockSuccessorEvaluationPermitV1,
) -> Result<(), ClockSuccessorError> {
    validate_context(&permit.context)?;
    if permit.prior_basis_id != permit.context.prior_basis_id
        || permit.prior_clock_window_evidence_digest != permit.context.prior_window.evidence_digest
        || permit.evaluation_policy_id != permit.context.evaluation_policy.id()
        || permit.clock_quorum_policy_id != permit.context.quorum_policy.id()
        || permit.clock_continuity_policy_id != permit.context.continuity_policy.id()
    {
        return Err(ClockSuccessorError::PermitIdentityMismatch);
    }
    let snapshot_digest = digest_trust_snapshot(&permit.context.trust_snapshot)
        .map_err(ClockSuccessorError::TrustSnapshotInvalid)?;
    if snapshot_digest != permit.trust_snapshot_digest {
        return Err(ClockSuccessorError::TrustSnapshotMismatch);
    }
    let expected = compute_successor_permit_id(
        permit.prior_basis_id,
        permit.prior_clock_window_evidence_digest,
        permit.evaluation_policy_id,
        permit.clock_quorum_policy_id,
        permit.clock_continuity_policy_id,
        permit.trust_snapshot_digest,
        permit.evaluation_lower_unix_ms,
        permit.evaluation_upper_unix_ms,
        permit.minimum_eligible_clock_authority_keys,
        permit.require_eligible_algorithm_diversity,
        &permit.eligible_clock_keys,
    )?;
    if expected != permit.id.as_digest() {
        return Err(ClockSuccessorError::PermitIdentityMismatch);
    }
    Ok(())
}

fn eligible_clock_authority_keys_for_envelope(
    trust_snapshot: &TrustSnapshot,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    policy: &ClockEvaluationPolicyV4,
) -> Result<Vec<SuccessorClockAuthorityKeyV1>, ClockSuccessorError> {
    let snapshot_lower_ms = seconds_to_millis(trust_snapshot.issued_at_unix_s)?;
    let snapshot_upper_ms = seconds_to_millis(trust_snapshot.expires_at_unix_s)?;
    if lower_unix_ms > upper_unix_ms
        || lower_unix_ms < snapshot_lower_ms
        || upper_unix_ms >= snapshot_upper_ms
    {
        return Err(ClockSuccessorError::SnapshotNotValidForEnvelope);
    }
    let mut eligible = Vec::new();
    for key in &trust_snapshot.keys {
        if key.status != KeyLifecycleStatus::Active || !key.usages.contains(&KeyUsage::ClockAuthority) {
            continue;
        }
        let key_lower_ms = seconds_to_millis(key.not_before_unix_s)?;
        if lower_unix_ms < key_lower_ms {
            continue;
        }
        if let Some(not_after_unix_s) = key.not_after_unix_s {
            let key_upper_ms = seconds_to_millis(not_after_unix_s)?;
            if upper_unix_ms >= key_upper_ms {
                continue;
            }
        }
        eligible.push(SuccessorClockAuthorityKeyV1 {
            algorithm: key.algorithm.clone(),
            key_id: key.key_id.clone(),
        });
    }
    eligible.sort_by(|left, right| {
        (&left.algorithm, left.key_id.as_str()).cmp(&(&right.algorithm, right.key_id.as_str()))
    });
    if eligible.len() < policy.minimum_eligible_clock_authority_keys() {
        return Err(ClockSuccessorError::InsufficientClockAuthorityKeys {
            actual: eligible.len(),
            required: policy.minimum_eligible_clock_authority_keys(),
        });
    }
    if policy.require_eligible_algorithm_diversity() {
        let algorithms = eligible
            .iter()
            .map(|key| key.algorithm.clone())
            .collect::<BTreeSet<_>>();
        if algorithms.len() < 2 {
            return Err(ClockSuccessorError::EligibleAlgorithmDiversityMissing);
        }
    }
    Ok(eligible)
}

#[allow(clippy::too_many_arguments)]
fn compute_successor_permit_id(
    prior_basis_id: AcceptedClockBasisIdV5,
    prior_window_digest: Sha256Digest,
    evaluation_policy_id: ClockEvaluationPolicyIdV4,
    quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    continuity_policy_id: ClockContinuityPolicyRevisionIdV1,
    trust_snapshot_digest: Sha256Digest,
    evaluation_lower_unix_ms: u64,
    evaluation_upper_unix_ms: u64,
    minimum_eligible_clock_authority_keys: usize,
    require_eligible_algorithm_diversity: bool,
    eligible_clock_keys: &[SuccessorClockAuthorityKeyV1],
) -> Result<Sha256Digest, ClockSuccessorError> {
    let eligible_keys_json = eligible_clock_keys
        .iter()
        .map(|key| {
            Value::Array(vec![
                Value::String(algorithm_identity(&key.algorithm)),
                Value::String(key.key_id.clone()),
            ])
        })
        .collect::<Vec<_>>();
    let bytes = canonical_json_bytes([
        ("clock_continuity_policy_id", Value::String(continuity_policy_id.to_hex())),
        ("clock_quorum_policy_id", Value::String(quorum_policy_id.to_hex())),
        ("eligible_clock_keys", Value::Array(eligible_keys_json)),
        ("evaluation_lower_unix_ms", Value::from(evaluation_lower_unix_ms)),
        ("evaluation_policy_id", Value::String(evaluation_policy_id.to_hex())),
        ("evaluation_upper_unix_ms", Value::from(evaluation_upper_unix_ms)),
        ("minimum_eligible_clock_authority_keys", Value::from(minimum_eligible_clock_authority_keys as u64)),
        ("prior_basis_id", Value::String(prior_basis_id.to_hex())),
        ("prior_clock_window_evidence_digest", Value::String(prior_window_digest.to_hex())),
        ("require_eligible_algorithm_diversity", Value::Bool(require_eligible_algorithm_diversity)),
        ("schema", Value::String(CLOCK_SUCCESSOR_EVALUATION_PERMIT_SCHEMA.to_string())),
        ("trust_snapshot_digest", Value::String(trust_snapshot_digest.to_hex())),
    ])?;
    Ok(domain_hash(CLOCK_SUCCESSOR_EVALUATION_PERMIT_DOMAIN, &bytes))
}

#[allow(clippy::too_many_arguments)]
fn compute_successor_basis_id(
    successor_permit_id: ClockSuccessorEvaluationPermitIdV1,
    prior_basis_id: AcceptedClockBasisIdV5,
    evaluation_policy_id: ClockEvaluationPolicyIdV4,
    quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    continuity_policy_id: ClockContinuityPolicyRevisionIdV1,
    trust_snapshot_digest: Sha256Digest,
    prior_window_digest: Sha256Digest,
    successor_window_digest: Sha256Digest,
    successor_witness_digest: Sha256Digest,
    continuity_digest: Sha256Digest,
    epoch: u64,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    consensus_unix_ms: u64,
) -> Result<Sha256Digest, ClockSuccessorError> {
    let bytes = canonical_json_bytes([
        ("acceptance_kind", Value::String(ACCEPTANCE_KIND_CONTINUOUS.to_string())),
        ("clock_continuity_digest", Value::String(continuity_digest.to_hex())),
        ("clock_continuity_policy_id", Value::String(continuity_policy_id.to_hex())),
        ("clock_evaluation_policy_id", Value::String(evaluation_policy_id.to_hex())),
        ("clock_quorum_policy_id", Value::String(quorum_policy_id.to_hex())),
        ("clock_window_evidence_digest", Value::String(successor_window_digest.to_hex())),
        ("clock_window_witness_digest", Value::String(successor_witness_digest.to_hex())),
        ("consensus_unix_ms", Value::from(consensus_unix_ms)),
        ("epoch", Value::from(epoch)),
        ("lower_unix_ms", Value::from(lower_unix_ms)),
        ("prior_basis_id", Value::String(prior_basis_id.to_hex())),
        ("prior_clock_window_evidence_digest", Value::String(prior_window_digest.to_hex())),
        ("schema", Value::String(ACCEPTED_CLOCK_SUCCESSOR_BASIS_SCHEMA.to_string())),
        ("successor_permit_id", Value::String(successor_permit_id.to_hex())),
        ("trust_snapshot_digest", Value::String(trust_snapshot_digest.to_hex())),
        ("upper_unix_ms", Value::from(upper_unix_ms)),
    ])?;
    Ok(domain_hash(ACCEPTED_CLOCK_SUCCESSOR_BASIS_DOMAIN, &bytes))
}

fn algorithm_identity(algorithm: &SignatureAlgorithm) -> String {
    match algorithm {
        SignatureAlgorithm::Ed25519 => "Ed25519".to_string(),
        SignatureAlgorithm::MlDsa65 => "MlDsa65".to_string(),
        SignatureAlgorithm::MlDsa87 => "MlDsa87".to_string(),
        SignatureAlgorithm::Other(name) => format!("Other:{name}"),
    }
}

fn seconds_to_millis(value: u64) -> Result<u64, ClockSuccessorError> {
    value.checked_mul(1_000).ok_or(ClockSuccessorError::TimeScaleOverflow)
}

fn canonical_json_bytes<const N: usize>(
    entries: [(&str, Value); N],
) -> Result<Vec<u8>, ClockSuccessorError> {
    let map = entries
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect::<BTreeMap<_, _>>();
    serde_json::to_vec(&map).map_err(|error| ClockSuccessorError::Encoding(error.to_string()))
}
