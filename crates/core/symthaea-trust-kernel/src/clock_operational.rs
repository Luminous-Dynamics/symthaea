// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Recursive normal-operation clock authority.
//!
//! A single opaque `OperationalClockBasisV1` type represents both the
//! bootstrap-admitted clock state and every later continuity-admitted state.
//! The same type is therefore the output of one transition and the authority
//! input to the next transition.

use crate::accepted_clock_basis::{AcceptedClockBasisIdV5, AcceptedClockBasisV5};
use crate::clock::{
    ClockObservation, ClockObservationVerifier, ClockViolation, VerifiedClockWindow,
};
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

pub const OPERATIONAL_CLOCK_BASIS_SCHEMA: &str =
    "symthaea.trust.clock-operational-basis.v1";
pub const OPERATIONAL_CLOCK_SUCCESSOR_PERMIT_SCHEMA: &str =
    "symthaea.trust.clock-successor-evaluation-permit.v2";

const OPERATIONAL_CLOCK_BASIS_DOMAIN: &[u8] =
    b"symthaea.trust.clock-operational-basis.v1\0";
const OPERATIONAL_CLOCK_SUCCESSOR_PERMIT_DOMAIN: &[u8] =
    b"symthaea.trust.clock-successor-evaluation-permit.v2\0";
const BASIS_KIND_BOOTSTRAP: &str = "Bootstrap";
const BASIS_KIND_CONTINUOUS: &str = "Continuous";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationalClockBasisKindV1 {
    Bootstrap,
    Continuous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationalClockBasisIdV1(Sha256Digest);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationalClockAuthorityKeyV1 {
    algorithm: SignatureAlgorithm,
    key_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockSuccessorEvaluationPermitIdV2(Sha256Digest);

#[derive(Debug, Clone)]
enum OperationalClockOriginV1 {
    Bootstrap {
        source_basis_id: AcceptedClockBasisIdV5,
    },
    Continuous {
        prior_operational_basis_id: OperationalClockBasisIdV1,
        successor_permit_id: ClockSuccessorEvaluationPermitIdV2,
        prior_window: VerifiedClockWindow,
        continuity: VerifiedClockContinuity,
    },
}

#[derive(Debug, Clone)]
#[must_use]
pub struct OperationalClockBasisV1 {
    id: OperationalClockBasisIdV1,
    origin: OperationalClockOriginV1,
    evaluation_policy: ClockEvaluationPolicyV4,
    quorum_policy: ClockQuorumPolicyRevisionV1,
    continuity_policy: ClockContinuityPolicyRevisionV1,
    trust_snapshot: TrustSnapshot,
    trust_snapshot_digest: Sha256Digest,
    window: VerifiedClockWindow,
    witness: ClockWindowEvaluationWitnessV1,
    witness_digest: Sha256Digest,
}

#[derive(Debug, Clone)]
#[must_use]
pub struct ClockSuccessorEvaluationPermitV2 {
    id: ClockSuccessorEvaluationPermitIdV2,
    prior_basis: OperationalClockBasisV1,
    evaluation_lower_unix_ms: u64,
    evaluation_upper_unix_ms: u64,
    eligible_clock_keys: Vec<OperationalClockAuthorityKeyV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationalClockError {
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
    BasisIdentityMismatch,
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

impl OperationalClockBasisIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl ClockSuccessorEvaluationPermitIdV2 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl OperationalClockAuthorityKeyV1 {
    pub fn algorithm(&self) -> &SignatureAlgorithm {
        &self.algorithm
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }
}

impl OperationalClockBasisV1 {
    pub fn id(&self) -> OperationalClockBasisIdV1 {
        self.id
    }

    pub fn kind(&self) -> OperationalClockBasisKindV1 {
        match &self.origin {
            OperationalClockOriginV1::Bootstrap { .. } => OperationalClockBasisKindV1::Bootstrap,
            OperationalClockOriginV1::Continuous { .. } => OperationalClockBasisKindV1::Continuous,
        }
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

    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }

    pub fn clock_window_evidence_digest(&self) -> Sha256Digest {
        self.window.evidence_digest
    }

    pub fn clock_window_witness_digest(&self) -> Sha256Digest {
        self.witness_digest
    }

    pub fn epoch(&self) -> u64 {
        self.window.epoch
    }

    pub fn lower_unix_ms(&self) -> u64 {
        self.window.lower_unix_ms
    }

    pub fn upper_unix_ms(&self) -> u64 {
        self.window.upper_unix_ms
    }

    pub fn consensus_unix_ms(&self) -> u64 {
        self.window.consensus_unix_ms
    }

    pub fn verified_window(&self) -> &VerifiedClockWindow {
        &self.window
    }

    pub fn evaluation_witness(&self) -> &ClockWindowEvaluationWitnessV1 {
        &self.witness
    }

    pub fn verified_continuity(&self) -> Option<&VerifiedClockContinuity> {
        match &self.origin {
            OperationalClockOriginV1::Continuous { continuity, .. } => Some(continuity),
            OperationalClockOriginV1::Bootstrap { .. } => None,
        }
    }

    pub fn predecessor_operational_basis_id(&self) -> Option<OperationalClockBasisIdV1> {
        match &self.origin {
            OperationalClockOriginV1::Continuous {
                prior_operational_basis_id,
                ..
            } => Some(*prior_operational_basis_id),
            OperationalClockOriginV1::Bootstrap { .. } => None,
        }
    }
}

impl ClockSuccessorEvaluationPermitV2 {
    pub fn id(&self) -> ClockSuccessorEvaluationPermitIdV2 {
        self.id
    }

    pub fn prior_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.prior_basis.id()
    }

    pub fn prior_clock_window_evidence_digest(&self) -> Sha256Digest {
        self.prior_basis.clock_window_evidence_digest()
    }

    pub fn clock_evaluation_policy_id(&self) -> ClockEvaluationPolicyIdV4 {
        self.prior_basis.clock_evaluation_policy_id()
    }

    pub fn clock_quorum_policy_id(&self) -> ClockQuorumPolicyRevisionIdV1 {
        self.prior_basis.clock_quorum_policy_id()
    }

    pub fn clock_continuity_policy_id(&self) -> ClockContinuityPolicyRevisionIdV1 {
        self.prior_basis.clock_continuity_policy_id()
    }

    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.prior_basis.trust_snapshot_digest()
    }

    pub fn evaluation_lower_unix_ms(&self) -> u64 {
        self.evaluation_lower_unix_ms
    }

    pub fn evaluation_upper_unix_ms(&self) -> u64 {
        self.evaluation_upper_unix_ms
    }

    pub fn eligible_clock_keys(&self) -> &[OperationalClockAuthorityKeyV1] {
        &self.eligible_clock_keys
    }
}

/// Convert an already-accepted V5 bootstrap clock basis into the recursive
/// operational authority type. Caller-provided policy/snapshot values are
/// witness records only; they must reproduce identities already committed by
/// the opaque V5 basis.
pub fn bind_bootstrap_operational_clock_basis_v1(
    prior_basis: &AcceptedClockBasisV5,
    evaluation_policy: &ClockEvaluationPolicyV4,
    trust_snapshot: &TrustSnapshot,
) -> Result<OperationalClockBasisV1, OperationalClockError> {
    evaluation_policy
        .validate()
        .map_err(OperationalClockError::EvaluationPolicyInvalid)?;
    if evaluation_policy.id() != prior_basis.clock_evaluation_policy_id() {
        return Err(OperationalClockError::EvaluationPolicyMismatch);
    }
    if evaluation_policy.clock_quorum_policy_id() != prior_basis.clock_quorum_policy_id() {
        return Err(OperationalClockError::QuorumPolicyMismatch);
    }
    if evaluation_policy.clock_continuity_policy_id() != prior_basis.clock_continuity_policy_id() {
        return Err(OperationalClockError::ContinuityPolicyMismatch);
    }

    let originating_permit = prior_basis.originating_permit();
    originating_permit
        .clock_quorum_policy()
        .validate()
        .map_err(OperationalClockError::EvaluationPolicyInvalid)?;
    originating_permit
        .clock_continuity_policy()
        .validate()
        .map_err(OperationalClockError::EvaluationPolicyInvalid)?;
    if originating_permit.evaluation_policy_id() != evaluation_policy.id() {
        return Err(OperationalClockError::EvaluationPolicyMismatch);
    }
    if originating_permit.clock_quorum_policy_id() != prior_basis.clock_quorum_policy_id()
        || originating_permit.clock_quorum_policy().id() != prior_basis.clock_quorum_policy_id()
    {
        return Err(OperationalClockError::QuorumPolicyMismatch);
    }
    if originating_permit.clock_continuity_policy_id()
        != prior_basis.clock_continuity_policy_id()
        || originating_permit.clock_continuity_policy().id()
            != prior_basis.clock_continuity_policy_id()
    {
        return Err(OperationalClockError::ContinuityPolicyMismatch);
    }

    trust_snapshot
        .validate()
        .map_err(OperationalClockError::TrustSnapshotInvalid)?;
    let snapshot_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(OperationalClockError::TrustSnapshotInvalid)?;
    if snapshot_digest != prior_basis.trust_snapshot_digest()
        || snapshot_digest != originating_permit.trust_snapshot_digest()
    {
        return Err(OperationalClockError::TrustSnapshotMismatch);
    }

    let window = prior_basis.verified_window();
    window
        .validate()
        .map_err(|_| OperationalClockError::PriorWindowInvalid)?;
    if window.evidence_digest != prior_basis.clock_window_evidence_digest()
        || window.trust_snapshot_digest != snapshot_digest
        || window.epoch != prior_basis.epoch()
        || window.lower_unix_ms != prior_basis.lower_unix_ms()
        || window.upper_unix_ms != prior_basis.upper_unix_ms()
        || window.consensus_unix_ms != prior_basis.consensus_unix_ms()
    {
        return Err(OperationalClockError::PriorWindowMismatch);
    }
    let witness_digest = verify_clock_window_evaluation_witness(
        window,
        prior_basis.evaluation_witness(),
    )
    .map_err(OperationalClockError::PriorWitnessInvalid)?;
    if witness_digest != prior_basis.clock_window_witness_digest() {
        return Err(OperationalClockError::PriorWitnessMismatch);
    }

    let id = OperationalClockBasisIdV1(compute_bootstrap_operational_basis_id(
        prior_basis.id(),
        evaluation_policy.id(),
        originating_permit.clock_quorum_policy().id(),
        originating_permit.clock_continuity_policy().id(),
        snapshot_digest,
        window.evidence_digest,
        witness_digest,
        window.epoch,
        window.lower_unix_ms,
        window.upper_unix_ms,
        window.consensus_unix_ms,
    )?);

    let basis = OperationalClockBasisV1 {
        id,
        origin: OperationalClockOriginV1::Bootstrap {
            source_basis_id: prior_basis.id(),
        },
        evaluation_policy: evaluation_policy.clone(),
        quorum_policy: originating_permit.clock_quorum_policy().clone(),
        continuity_policy: originating_permit.clock_continuity_policy().clone(),
        trust_snapshot: trust_snapshot.clone(),
        trust_snapshot_digest: snapshot_digest,
        window: window.clone(),
        witness: prior_basis.evaluation_witness().clone(),
        witness_digest,
    };
    validate_operational_basis(&basis)?;
    Ok(basis)
}

/// Derive the next evaluation permit before any candidate successor clock
/// observation is supplied.
pub fn derive_operational_clock_successor_permit_v2(
    prior_basis: &OperationalClockBasisV1,
) -> Result<ClockSuccessorEvaluationPermitV2, OperationalClockError> {
    validate_operational_basis(prior_basis)?;
    let evaluation_lower_unix_ms = prior_basis.window.lower_unix_ms;
    let evaluation_upper_unix_ms = prior_basis
        .window
        .upper_unix_ms
        .checked_add(prior_basis.evaluation_policy.max_transition_ms())
        .ok_or(OperationalClockError::TransitionEnvelopeOverflow)?;
    let eligible_clock_keys = eligible_clock_authority_keys_for_envelope(
        &prior_basis.trust_snapshot,
        evaluation_lower_unix_ms,
        evaluation_upper_unix_ms,
        &prior_basis.evaluation_policy,
    )?;
    let id = ClockSuccessorEvaluationPermitIdV2(compute_successor_permit_id(
        prior_basis.id,
        prior_basis.window.evidence_digest,
        prior_basis.evaluation_policy.id(),
        prior_basis.quorum_policy.id(),
        prior_basis.continuity_policy.id(),
        prior_basis.trust_snapshot_digest,
        evaluation_lower_unix_ms,
        evaluation_upper_unix_ms,
        prior_basis
            .evaluation_policy
            .minimum_eligible_clock_authority_keys(),
        prior_basis
            .evaluation_policy
            .require_eligible_algorithm_diversity(),
        &eligible_clock_keys,
    )?);

    Ok(ClockSuccessorEvaluationPermitV2 {
        id,
        prior_basis: prior_basis.clone(),
        evaluation_lower_unix_ms,
        evaluation_upper_unix_ms,
        eligible_clock_keys,
    })
}

/// Verify original signed observations under a pre-candidate permit, prove
/// continuity from the immediately prior operational window, and return the
/// same opaque operational-basis type for the next transition.
pub fn accept_operational_clock_successor_v1(
    permit: &ClockSuccessorEvaluationPermitV2,
    observations: &[ClockObservation],
    verifier: &dyn ClockObservationVerifier,
) -> Result<OperationalClockBasisV1, OperationalClockError> {
    validate_successor_permit(permit)?;
    let prior = &permit.prior_basis;
    let quorum_policy = prior
        .quorum_policy
        .to_runtime_policy()
        .map_err(OperationalClockError::EvaluationPolicyInvalid)?;
    let evaluation_time_unix_s = permit.evaluation_upper_unix_ms / 1_000;
    let (window, witness) = verify_clock_quorum_with_witness(
        observations,
        &quorum_policy,
        &prior.trust_snapshot,
        evaluation_time_unix_s,
        verifier,
    )
    .map_err(OperationalClockError::QuorumVerificationFailed)?;
    let witness_digest = verify_clock_window_evaluation_witness(&window, &witness)
        .map_err(OperationalClockError::SuccessorWitnessInvalid)?;

    if window.trust_snapshot_digest != prior.trust_snapshot_digest {
        return Err(OperationalClockError::WindowTrustSnapshotMismatch);
    }
    if window.lower_unix_ms < permit.evaluation_lower_unix_ms
        || window.upper_unix_ms > permit.evaluation_upper_unix_ms
    {
        return Err(OperationalClockError::WindowOutsidePermit);
    }

    let eligible_signers = permit
        .eligible_clock_keys
        .iter()
        .map(|key| (key.algorithm.clone(), key.key_id.clone()))
        .collect::<BTreeSet<_>>();
    for signer in &witness.signers {
        if !eligible_signers.contains(&(signer.algorithm.clone(), signer.key_id.clone())) {
            return Err(OperationalClockError::UnpermittedSigner(
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
                OperationalClockError::ObservationIntervalOverflow(
                    observation.source_id.clone(),
                )
            })?;
        if lower < permit.evaluation_lower_unix_ms
            || upper > permit.evaluation_upper_unix_ms
        {
            return Err(OperationalClockError::ObservationIntervalOutsidePermit(
                observation.source_id.clone(),
            ));
        }
    }

    let continuity_policy = prior
        .continuity_policy
        .to_runtime_policy()
        .map_err(OperationalClockError::EvaluationPolicyInvalid)?;
    let continuity = verify_clock_continuity(
        &prior.window,
        &window,
        &continuity_policy,
    )
    .map_err(OperationalClockError::ContinuityInvalid)?;

    let id = OperationalClockBasisIdV1(compute_continuous_operational_basis_id(
        prior.id,
        permit.id,
        prior.evaluation_policy.id(),
        prior.quorum_policy.id(),
        prior.continuity_policy.id(),
        prior.trust_snapshot_digest,
        prior.window.evidence_digest,
        window.evidence_digest,
        witness_digest,
        continuity.continuity_digest,
        window.epoch,
        window.lower_unix_ms,
        window.upper_unix_ms,
        window.consensus_unix_ms,
    )?);

    let basis = OperationalClockBasisV1 {
        id,
        origin: OperationalClockOriginV1::Continuous {
            prior_operational_basis_id: prior.id,
            successor_permit_id: permit.id,
            prior_window: prior.window.clone(),
            continuity,
        },
        evaluation_policy: prior.evaluation_policy.clone(),
        quorum_policy: prior.quorum_policy.clone(),
        continuity_policy: prior.continuity_policy.clone(),
        trust_snapshot: prior.trust_snapshot.clone(),
        trust_snapshot_digest: prior.trust_snapshot_digest,
        window,
        witness,
        witness_digest,
    };
    validate_operational_basis(&basis)?;
    Ok(basis)
}

fn validate_operational_basis(
    basis: &OperationalClockBasisV1,
) -> Result<(), OperationalClockError> {
    basis
        .evaluation_policy
        .validate()
        .map_err(OperationalClockError::EvaluationPolicyInvalid)?;
    basis
        .quorum_policy
        .validate()
        .map_err(OperationalClockError::EvaluationPolicyInvalid)?;
    basis
        .continuity_policy
        .validate()
        .map_err(OperationalClockError::EvaluationPolicyInvalid)?;
    if basis.evaluation_policy.clock_quorum_policy_id() != basis.quorum_policy.id() {
        return Err(OperationalClockError::QuorumPolicyMismatch);
    }
    if basis.evaluation_policy.clock_continuity_policy_id() != basis.continuity_policy.id() {
        return Err(OperationalClockError::ContinuityPolicyMismatch);
    }
    basis
        .trust_snapshot
        .validate()
        .map_err(OperationalClockError::TrustSnapshotInvalid)?;
    let snapshot_digest = digest_trust_snapshot(&basis.trust_snapshot)
        .map_err(OperationalClockError::TrustSnapshotInvalid)?;
    if snapshot_digest != basis.trust_snapshot_digest
        || basis.window.trust_snapshot_digest != snapshot_digest
    {
        return Err(OperationalClockError::TrustSnapshotMismatch);
    }
    basis
        .window
        .validate()
        .map_err(|_| OperationalClockError::PriorWindowInvalid)?;
    let witness_digest = verify_clock_window_evaluation_witness(
        &basis.window,
        &basis.witness,
    )
    .map_err(OperationalClockError::PriorWitnessInvalid)?;
    if witness_digest != basis.witness_digest {
        return Err(OperationalClockError::PriorWitnessMismatch);
    }

    let expected = match &basis.origin {
        OperationalClockOriginV1::Bootstrap { source_basis_id } => {
            compute_bootstrap_operational_basis_id(
                *source_basis_id,
                basis.evaluation_policy.id(),
                basis.quorum_policy.id(),
                basis.continuity_policy.id(),
                snapshot_digest,
                basis.window.evidence_digest,
                witness_digest,
                basis.window.epoch,
                basis.window.lower_unix_ms,
                basis.window.upper_unix_ms,
                basis.window.consensus_unix_ms,
            )?
        }
        OperationalClockOriginV1::Continuous {
            prior_operational_basis_id,
            successor_permit_id,
            prior_window,
            continuity,
        } => {
            prior_window
                .validate()
                .map_err(|_| OperationalClockError::PriorWindowInvalid)?;
            if prior_window.trust_snapshot_digest != snapshot_digest {
                return Err(OperationalClockError::TrustSnapshotMismatch);
            }
            let runtime_continuity = basis
                .continuity_policy
                .to_runtime_policy()
                .map_err(OperationalClockError::EvaluationPolicyInvalid)?;
            let recomputed = verify_clock_continuity(
                prior_window,
                &basis.window,
                &runtime_continuity,
            )
            .map_err(OperationalClockError::ContinuityInvalid)?;
            if recomputed.continuity_digest != continuity.continuity_digest {
                return Err(OperationalClockError::ContinuityInvalid(
                    ClockContinuityError::DigestMismatch,
                ));
            }
            compute_continuous_operational_basis_id(
                *prior_operational_basis_id,
                *successor_permit_id,
                basis.evaluation_policy.id(),
                basis.quorum_policy.id(),
                basis.continuity_policy.id(),
                snapshot_digest,
                prior_window.evidence_digest,
                basis.window.evidence_digest,
                witness_digest,
                continuity.continuity_digest,
                basis.window.epoch,
                basis.window.lower_unix_ms,
                basis.window.upper_unix_ms,
                basis.window.consensus_unix_ms,
            )?
        }
    };
    if expected != basis.id.as_digest() {
        return Err(OperationalClockError::BasisIdentityMismatch);
    }
    Ok(())
}

fn validate_successor_permit(
    permit: &ClockSuccessorEvaluationPermitV2,
) -> Result<(), OperationalClockError> {
    validate_operational_basis(&permit.prior_basis)?;
    let expected_lower = permit.prior_basis.window.lower_unix_ms;
    let expected_upper = permit
        .prior_basis
        .window
        .upper_unix_ms
        .checked_add(permit.prior_basis.evaluation_policy.max_transition_ms())
        .ok_or(OperationalClockError::TransitionEnvelopeOverflow)?;
    if permit.evaluation_lower_unix_ms != expected_lower
        || permit.evaluation_upper_unix_ms != expected_upper
    {
        return Err(OperationalClockError::PermitIdentityMismatch);
    }
    let eligible = eligible_clock_authority_keys_for_envelope(
        &permit.prior_basis.trust_snapshot,
        expected_lower,
        expected_upper,
        &permit.prior_basis.evaluation_policy,
    )?;
    if eligible != permit.eligible_clock_keys {
        return Err(OperationalClockError::PermitIdentityMismatch);
    }
    let expected = compute_successor_permit_id(
        permit.prior_basis.id,
        permit.prior_basis.window.evidence_digest,
        permit.prior_basis.evaluation_policy.id(),
        permit.prior_basis.quorum_policy.id(),
        permit.prior_basis.continuity_policy.id(),
        permit.prior_basis.trust_snapshot_digest,
        expected_lower,
        expected_upper,
        permit
            .prior_basis
            .evaluation_policy
            .minimum_eligible_clock_authority_keys(),
        permit
            .prior_basis
            .evaluation_policy
            .require_eligible_algorithm_diversity(),
        &eligible,
    )?;
    if expected != permit.id.as_digest() {
        return Err(OperationalClockError::PermitIdentityMismatch);
    }
    Ok(())
}

fn eligible_clock_authority_keys_for_envelope(
    trust_snapshot: &TrustSnapshot,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    policy: &ClockEvaluationPolicyV4,
) -> Result<Vec<OperationalClockAuthorityKeyV1>, OperationalClockError> {
    let snapshot_lower_ms = seconds_to_millis(trust_snapshot.issued_at_unix_s)?;
    let snapshot_upper_ms = seconds_to_millis(trust_snapshot.expires_at_unix_s)?;
    if lower_unix_ms > upper_unix_ms
        || lower_unix_ms < snapshot_lower_ms
        || upper_unix_ms >= snapshot_upper_ms
    {
        return Err(OperationalClockError::SnapshotNotValidForEnvelope);
    }
    let mut eligible = Vec::new();
    for key in &trust_snapshot.keys {
        if key.status != KeyLifecycleStatus::Active
            || !key.usages.contains(&KeyUsage::ClockAuthority)
        {
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
        eligible.push(OperationalClockAuthorityKeyV1 {
            algorithm: key.algorithm.clone(),
            key_id: key.key_id.clone(),
        });
    }
    eligible.sort_by(|left, right| {
        (&left.algorithm, left.key_id.as_str())
            .cmp(&(&right.algorithm, right.key_id.as_str()))
    });
    if eligible.len() < policy.minimum_eligible_clock_authority_keys() {
        return Err(OperationalClockError::InsufficientClockAuthorityKeys {
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
            return Err(OperationalClockError::EligibleAlgorithmDiversityMissing);
        }
    }
    Ok(eligible)
}

#[allow(clippy::too_many_arguments)]
fn compute_bootstrap_operational_basis_id(
    source_basis_id: AcceptedClockBasisIdV5,
    evaluation_policy_id: ClockEvaluationPolicyIdV4,
    quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    continuity_policy_id: ClockContinuityPolicyRevisionIdV1,
    trust_snapshot_digest: Sha256Digest,
    window_digest: Sha256Digest,
    witness_digest: Sha256Digest,
    epoch: u64,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    consensus_unix_ms: u64,
) -> Result<Sha256Digest, OperationalClockError> {
    let bytes = canonical_json_bytes([
        ("basis_kind", Value::String(BASIS_KIND_BOOTSTRAP.to_string())),
        (
            "clock_continuity_policy_id",
            Value::String(continuity_policy_id.to_hex()),
        ),
        (
            "clock_evaluation_policy_id",
            Value::String(evaluation_policy_id.to_hex()),
        ),
        (
            "clock_quorum_policy_id",
            Value::String(quorum_policy_id.to_hex()),
        ),
        (
            "clock_window_evidence_digest",
            Value::String(window_digest.to_hex()),
        ),
        (
            "clock_window_witness_digest",
            Value::String(witness_digest.to_hex()),
        ),
        ("consensus_unix_ms", Value::from(consensus_unix_ms)),
        ("epoch", Value::from(epoch)),
        ("lower_unix_ms", Value::from(lower_unix_ms)),
        (
            "schema",
            Value::String(OPERATIONAL_CLOCK_BASIS_SCHEMA.to_string()),
        ),
        ("source_basis_id", Value::String(source_basis_id.to_hex())),
        (
            "trust_snapshot_digest",
            Value::String(trust_snapshot_digest.to_hex()),
        ),
        ("upper_unix_ms", Value::from(upper_unix_ms)),
    ])?;
    Ok(domain_hash(OPERATIONAL_CLOCK_BASIS_DOMAIN, &bytes))
}

#[allow(clippy::too_many_arguments)]
fn compute_continuous_operational_basis_id(
    prior_operational_basis_id: OperationalClockBasisIdV1,
    successor_permit_id: ClockSuccessorEvaluationPermitIdV2,
    evaluation_policy_id: ClockEvaluationPolicyIdV4,
    quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    continuity_policy_id: ClockContinuityPolicyRevisionIdV1,
    trust_snapshot_digest: Sha256Digest,
    prior_window_digest: Sha256Digest,
    window_digest: Sha256Digest,
    witness_digest: Sha256Digest,
    continuity_digest: Sha256Digest,
    epoch: u64,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    consensus_unix_ms: u64,
) -> Result<Sha256Digest, OperationalClockError> {
    let bytes = canonical_json_bytes([
        ("basis_kind", Value::String(BASIS_KIND_CONTINUOUS.to_string())),
        (
            "clock_continuity_digest",
            Value::String(continuity_digest.to_hex()),
        ),
        (
            "clock_continuity_policy_id",
            Value::String(continuity_policy_id.to_hex()),
        ),
        (
            "clock_evaluation_policy_id",
            Value::String(evaluation_policy_id.to_hex()),
        ),
        (
            "clock_quorum_policy_id",
            Value::String(quorum_policy_id.to_hex()),
        ),
        (
            "clock_window_evidence_digest",
            Value::String(window_digest.to_hex()),
        ),
        (
            "clock_window_witness_digest",
            Value::String(witness_digest.to_hex()),
        ),
        ("consensus_unix_ms", Value::from(consensus_unix_ms)),
        ("epoch", Value::from(epoch)),
        ("lower_unix_ms", Value::from(lower_unix_ms)),
        (
            "prior_clock_window_evidence_digest",
            Value::String(prior_window_digest.to_hex()),
        ),
        (
            "prior_operational_basis_id",
            Value::String(prior_operational_basis_id.to_hex()),
        ),
        (
            "schema",
            Value::String(OPERATIONAL_CLOCK_BASIS_SCHEMA.to_string()),
        ),
        (
            "successor_permit_id",
            Value::String(successor_permit_id.to_hex()),
        ),
        (
            "trust_snapshot_digest",
            Value::String(trust_snapshot_digest.to_hex()),
        ),
        ("upper_unix_ms", Value::from(upper_unix_ms)),
    ])?;
    Ok(domain_hash(OPERATIONAL_CLOCK_BASIS_DOMAIN, &bytes))
}

#[allow(clippy::too_many_arguments)]
fn compute_successor_permit_id(
    prior_operational_basis_id: OperationalClockBasisIdV1,
    prior_window_digest: Sha256Digest,
    evaluation_policy_id: ClockEvaluationPolicyIdV4,
    quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    continuity_policy_id: ClockContinuityPolicyRevisionIdV1,
    trust_snapshot_digest: Sha256Digest,
    evaluation_lower_unix_ms: u64,
    evaluation_upper_unix_ms: u64,
    minimum_eligible_clock_authority_keys: usize,
    require_eligible_algorithm_diversity: bool,
    eligible_clock_keys: &[OperationalClockAuthorityKeyV1],
) -> Result<Sha256Digest, OperationalClockError> {
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
        (
            "clock_continuity_policy_id",
            Value::String(continuity_policy_id.to_hex()),
        ),
        (
            "clock_quorum_policy_id",
            Value::String(quorum_policy_id.to_hex()),
        ),
        ("eligible_clock_keys", Value::Array(eligible_keys_json)),
        (
            "evaluation_lower_unix_ms",
            Value::from(evaluation_lower_unix_ms),
        ),
        (
            "evaluation_policy_id",
            Value::String(evaluation_policy_id.to_hex()),
        ),
        (
            "evaluation_upper_unix_ms",
            Value::from(evaluation_upper_unix_ms),
        ),
        (
            "minimum_eligible_clock_authority_keys",
            Value::from(minimum_eligible_clock_authority_keys as u64),
        ),
        (
            "prior_operational_basis_id",
            Value::String(prior_operational_basis_id.to_hex()),
        ),
        (
            "prior_clock_window_evidence_digest",
            Value::String(prior_window_digest.to_hex()),
        ),
        (
            "require_eligible_algorithm_diversity",
            Value::Bool(require_eligible_algorithm_diversity),
        ),
        (
            "schema",
            Value::String(OPERATIONAL_CLOCK_SUCCESSOR_PERMIT_SCHEMA.to_string()),
        ),
        (
            "trust_snapshot_digest",
            Value::String(trust_snapshot_digest.to_hex()),
        ),
    ])?;
    Ok(domain_hash(OPERATIONAL_CLOCK_SUCCESSOR_PERMIT_DOMAIN, &bytes))
}

fn algorithm_identity(algorithm: &SignatureAlgorithm) -> String {
    match algorithm {
        SignatureAlgorithm::Ed25519 => "Ed25519".to_string(),
        SignatureAlgorithm::MlDsa65 => "MlDsa65".to_string(),
        SignatureAlgorithm::MlDsa87 => "MlDsa87".to_string(),
        SignatureAlgorithm::Other(name) => format!("Other:{name}"),
    }
}

fn seconds_to_millis(value: u64) -> Result<u64, OperationalClockError> {
    value
        .checked_mul(1_000)
        .ok_or(OperationalClockError::TimeScaleOverflow)
}

fn canonical_json_bytes<const N: usize>(
    entries: [(&str, Value); N],
) -> Result<Vec<u8>, OperationalClockError> {
    let map = entries
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect::<BTreeMap<_, _>>();
    serde_json::to_vec(&map).map_err(|error| OperationalClockError::Encoding(error.to_string()))
}
