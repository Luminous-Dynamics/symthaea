// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stable normal-operation clock authority over accepted-clock-basis V6 wire semantics.
//!
//! A versioned evidence/capability record should not force a new Rust protocol type
//! for every clock epoch. This module binds once to an already accepted V6 basis and
//! exact matching policy/snapshot witness records, then carries that authority
//! privately across repeated successor transitions. Every newly accepted basis is
//! still hashed under `symthaea.trust.accepted-clock-basis.v6`; there is no V7 wire
//! protocol here.

use crate::clock::{ClockObservation, ClockObservationVerifier, ClockViolation, VerifiedClockWindow};
use crate::clock_evaluation_permit::{
    ClockContinuityPolicyRevisionIdV1, ClockContinuityPolicyRevisionV1,
    ClockEvaluationPermitError, ClockEvaluationPolicyIdV4, ClockEvaluationPolicyV4,
    ClockQuorumPolicyRevisionIdV1, ClockQuorumPolicyRevisionV1,
};
use crate::clock_successor::{
    AcceptedClockBasisV6, ClockSuccessorError, CLOCK_SUCCESSOR_EVALUATION_PERMIT_SCHEMA,
};
use crate::clock_witness::{
    ClockWindowEvaluationWitnessV1, ClockWindowWitnessError,
    verify_clock_quorum_with_witness, verify_clock_window_evaluation_witness,
};
use crate::continuity::{
    ClockContinuityError, VerifiedClockContinuity, digest_clock_continuity,
    verify_clock_continuity,
};
use crate::digest::{Sha256Digest, domain_hash};
use crate::signature::SignatureAlgorithm;
use crate::trust::{
    KeyLifecycleStatus, KeyUsage, TrustSnapshot, TrustSnapshotError, digest_trust_snapshot,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const CONTINUOUS_CLOCK_BASIS_RUNTIME_SCHEMA: &str =
    "symthaea.trust.continuous-clock-basis-runtime.v1";
pub const ACCEPTED_CLOCK_BASIS_V6_WIRE_SCHEMA: &str = "symthaea.trust.accepted-clock-basis.v6";

const CLOCK_SUCCESSOR_EVALUATION_PERMIT_DOMAIN: &[u8] =
    b"symthaea.trust.clock-successor-evaluation-permit.v1\0";
const ACCEPTED_CLOCK_BASIS_V6_DOMAIN: &[u8] = b"symthaea.trust.accepted-clock-basis.v6\0";
const ACCEPTANCE_KIND_CONTINUOUS: &str = "Continuous";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContinuousClockBasisIdV1(Sha256Digest);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContinuousClockSuccessorPermitIdV1(Sha256Digest);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuousClockAuthorityKeyV1 {
    algorithm: SignatureAlgorithm,
    key_id: String,
}

/// Stable runtime carrier for authority already established by a V6 accepted basis.
///
/// The policy/snapshot records retained here are witness material whose exact IDs
/// were checked against the originating opaque V6 capability. They cannot be
/// selected again during normal successor progression.
#[derive(Debug, Clone)]
#[must_use]
pub struct ContinuousClockBasisV1 {
    wire_basis_id: ContinuousClockBasisIdV1,
    evaluation_policy: ClockEvaluationPolicyV4,
    quorum_policy: ClockQuorumPolicyRevisionV1,
    continuity_policy: ClockContinuityPolicyRevisionV1,
    trust_snapshot: TrustSnapshot,
    window: VerifiedClockWindow,
    witness: ClockWindowEvaluationWitnessV1,
    continuity: VerifiedClockContinuity,
}

#[derive(Debug, Clone)]
#[must_use]
pub struct ContinuousClockSuccessorPermitV1 {
    id: ContinuousClockSuccessorPermitIdV1,
    prior_basis_id: ContinuousClockBasisIdV1,
    prior_clock_window_evidence_digest: Sha256Digest,
    evaluation_policy_id: ClockEvaluationPolicyIdV4,
    clock_quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    clock_continuity_policy_id: ClockContinuityPolicyRevisionIdV1,
    trust_snapshot_digest: Sha256Digest,
    evaluation_lower_unix_ms: u64,
    evaluation_upper_unix_ms: u64,
    minimum_eligible_clock_authority_keys: usize,
    require_eligible_algorithm_diversity: bool,
    eligible_clock_keys: Vec<ContinuousClockAuthorityKeyV1>,
    basis: ContinuousClockBasisV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContinuousClockError {
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
    PriorContinuityInvalid(ClockContinuityError),
    PriorContinuityMismatch,
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

impl ContinuousClockBasisIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

impl ContinuousClockSuccessorPermitIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

impl ContinuousClockAuthorityKeyV1 {
    pub fn algorithm(&self) -> &SignatureAlgorithm { &self.algorithm }
    pub fn key_id(&self) -> &str { &self.key_id }
}

impl ContinuousClockBasisV1 {
    pub fn id(&self) -> ContinuousClockBasisIdV1 { self.wire_basis_id }
    pub fn wire_v6_digest(&self) -> Sha256Digest { self.wire_basis_id.as_digest() }
    pub fn clock_evaluation_policy_id(&self) -> ClockEvaluationPolicyIdV4 { self.evaluation_policy.id() }
    pub fn clock_quorum_policy_id(&self) -> ClockQuorumPolicyRevisionIdV1 { self.quorum_policy.id() }
    pub fn clock_continuity_policy_id(&self) -> ClockContinuityPolicyRevisionIdV1 { self.continuity_policy.id() }
    pub fn trust_snapshot_digest(&self) -> Result<Sha256Digest, ContinuousClockError> {
        digest_trust_snapshot(&self.trust_snapshot).map_err(ContinuousClockError::TrustSnapshotInvalid)
    }
    pub fn epoch(&self) -> u64 { self.window.epoch }
    pub fn lower_unix_ms(&self) -> u64 { self.window.lower_unix_ms }
    pub fn upper_unix_ms(&self) -> u64 { self.window.upper_unix_ms }
    pub fn consensus_unix_ms(&self) -> u64 { self.window.consensus_unix_ms }
    pub fn verified_window(&self) -> &VerifiedClockWindow { &self.window }
    pub fn evaluation_witness(&self) -> &ClockWindowEvaluationWitnessV1 { &self.witness }
    pub fn verified_continuity(&self) -> &VerifiedClockContinuity { &self.continuity }
}

impl ContinuousClockSuccessorPermitV1 {
    pub fn id(&self) -> ContinuousClockSuccessorPermitIdV1 { self.id }
    pub fn prior_basis_id(&self) -> ContinuousClockBasisIdV1 { self.prior_basis_id }
    pub fn prior_clock_window_evidence_digest(&self) -> Sha256Digest { self.prior_clock_window_evidence_digest }
    pub fn clock_evaluation_policy_id(&self) -> ClockEvaluationPolicyIdV4 { self.evaluation_policy_id }
    pub fn clock_quorum_policy_id(&self) -> ClockQuorumPolicyRevisionIdV1 { self.clock_quorum_policy_id }
    pub fn clock_continuity_policy_id(&self) -> ClockContinuityPolicyRevisionIdV1 { self.clock_continuity_policy_id }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest { self.trust_snapshot_digest }
    pub fn evaluation_lower_unix_ms(&self) -> u64 { self.evaluation_lower_unix_ms }
    pub fn evaluation_upper_unix_ms(&self) -> u64 { self.evaluation_upper_unix_ms }
    pub fn eligible_clock_keys(&self) -> &[ContinuousClockAuthorityKeyV1] { &self.eligible_clock_keys }
}

/// Upgrade an opaque one-step V6 basis into the stable runtime carrier.
///
/// The supplied records are non-authoritative witnesses. Every record is
/// content-validated and must match an ID/digest already committed by `prior`.
pub fn bind_continuous_clock_basis_v1(
    prior: &AcceptedClockBasisV6,
    evaluation_policy: &ClockEvaluationPolicyV4,
    quorum_policy: &ClockQuorumPolicyRevisionV1,
    continuity_policy: &ClockContinuityPolicyRevisionV1,
    trust_snapshot: &TrustSnapshot,
) -> Result<ContinuousClockBasisV1, ContinuousClockError> {
    evaluation_policy.validate().map_err(ContinuousClockError::EvaluationPolicyInvalid)?;
    quorum_policy.validate().map_err(ContinuousClockError::EvaluationPolicyInvalid)?;
    continuity_policy.validate().map_err(ContinuousClockError::EvaluationPolicyInvalid)?;

    if evaluation_policy.id() != prior.clock_evaluation_policy_id() {
        return Err(ContinuousClockError::EvaluationPolicyMismatch);
    }
    if quorum_policy.id() != prior.clock_quorum_policy_id()
        || evaluation_policy.clock_quorum_policy_id() != prior.clock_quorum_policy_id()
    {
        return Err(ContinuousClockError::QuorumPolicyMismatch);
    }
    if continuity_policy.id() != prior.clock_continuity_policy_id()
        || evaluation_policy.clock_continuity_policy_id() != prior.clock_continuity_policy_id()
    {
        return Err(ContinuousClockError::ContinuityPolicyMismatch);
    }

    trust_snapshot.validate().map_err(ContinuousClockError::TrustSnapshotInvalid)?;
    let snapshot_digest =
        digest_trust_snapshot(trust_snapshot).map_err(ContinuousClockError::TrustSnapshotInvalid)?;
    if snapshot_digest != prior.trust_snapshot_digest() {
        return Err(ContinuousClockError::TrustSnapshotMismatch);
    }

    let window = prior.verified_window();
    window.validate().map_err(|_| ContinuousClockError::PriorWindowInvalid)?;
    if window.evidence_digest != prior.clock_window_evidence_digest()
        || window.trust_snapshot_digest != snapshot_digest
        || window.epoch != prior.epoch()
        || window.lower_unix_ms != prior.lower_unix_ms()
        || window.upper_unix_ms != prior.upper_unix_ms()
        || window.consensus_unix_ms != prior.consensus_unix_ms()
    {
        return Err(ContinuousClockError::PriorWindowMismatch);
    }

    let witness_digest = verify_clock_window_evaluation_witness(window, prior.evaluation_witness())
        .map_err(ContinuousClockError::PriorWitnessInvalid)?;
    if witness_digest != prior.clock_window_witness_digest() {
        return Err(ContinuousClockError::PriorWitnessMismatch);
    }

    let continuity_digest = digest_clock_continuity(prior.verified_continuity())
        .map_err(ContinuousClockError::PriorContinuityInvalid)?;
    if continuity_digest != prior.clock_continuity_digest()
        || prior.verified_continuity().successor_evidence_digest != window.evidence_digest
        || prior.verified_continuity().successor_epoch != window.epoch
        || prior.verified_continuity().previous_evidence_digest
            != prior.prior_clock_window_evidence_digest()
    {
        return Err(ContinuousClockError::PriorContinuityMismatch);
    }

    Ok(ContinuousClockBasisV1 {
        wire_basis_id: ContinuousClockBasisIdV1(prior.id().as_digest()),
        evaluation_policy: evaluation_policy.clone(),
        quorum_policy: quorum_policy.clone(),
        continuity_policy: continuity_policy.clone(),
        trust_snapshot: trust_snapshot.clone(),
        window: window.clone(),
        witness: prior.evaluation_witness().clone(),
        continuity: prior.verified_continuity().clone(),
    })
}

/// Derive the next permit from the stable basis only. No policy/snapshot/time or
/// candidate clock input is accepted on the normal path.
pub fn derive_continuous_clock_successor_permit_v1(
    basis: &ContinuousClockBasisV1,
) -> Result<ContinuousClockSuccessorPermitV1, ContinuousClockError> {
    validate_continuous_basis(basis)?;
    let lower_unix_ms = basis.window.lower_unix_ms;
    let upper_unix_ms = basis
        .window
        .upper_unix_ms
        .checked_add(basis.evaluation_policy.max_transition_ms())
        .ok_or(ContinuousClockError::TransitionEnvelopeOverflow)?;
    let snapshot_digest =
        digest_trust_snapshot(&basis.trust_snapshot).map_err(ContinuousClockError::TrustSnapshotInvalid)?;
    let eligible_clock_keys = eligible_keys_for_envelope(
        &basis.trust_snapshot,
        lower_unix_ms,
        upper_unix_ms,
        &basis.evaluation_policy,
    )?;
    let id = ContinuousClockSuccessorPermitIdV1(compute_successor_permit_id(
        basis.wire_basis_id,
        basis.window.evidence_digest,
        basis.evaluation_policy.id(),
        basis.quorum_policy.id(),
        basis.continuity_policy.id(),
        snapshot_digest,
        lower_unix_ms,
        upper_unix_ms,
        basis.evaluation_policy.minimum_eligible_clock_authority_keys(),
        basis.evaluation_policy.require_eligible_algorithm_diversity(),
        &eligible_clock_keys,
    )?);

    Ok(ContinuousClockSuccessorPermitV1 {
        id,
        prior_basis_id: basis.wire_basis_id,
        prior_clock_window_evidence_digest: basis.window.evidence_digest,
        evaluation_policy_id: basis.evaluation_policy.id(),
        clock_quorum_policy_id: basis.quorum_policy.id(),
        clock_continuity_policy_id: basis.continuity_policy.id(),
        trust_snapshot_digest: snapshot_digest,
        evaluation_lower_unix_ms: lower_unix_ms,
        evaluation_upper_unix_ms: upper_unix_ms,
        minimum_eligible_clock_authority_keys: basis.evaluation_policy.minimum_eligible_clock_authority_keys(),
        require_eligible_algorithm_diversity: basis.evaluation_policy.require_eligible_algorithm_diversity(),
        eligible_clock_keys,
        basis: basis.clone(),
    })
}

/// Advance the stable V6 wire authority by one epoch. The returned value is the
/// same runtime type and the same accepted-clock-basis.v6 wire schema.
pub fn advance_continuous_clock_basis_v1(
    permit: &ContinuousClockSuccessorPermitV1,
    observations: &[ClockObservation],
    verifier: &dyn ClockObservationVerifier,
) -> Result<ContinuousClockBasisV1, ContinuousClockError> {
    validate_continuous_permit(permit)?;
    let quorum = permit
        .basis
        .quorum_policy
        .to_runtime_policy()
        .map_err(ContinuousClockError::EvaluationPolicyInvalid)?;
    let evaluation_time_unix_s = permit.evaluation_upper_unix_ms / 1_000;
    let (window, witness) = verify_clock_quorum_with_witness(
        observations,
        &quorum,
        &permit.basis.trust_snapshot,
        evaluation_time_unix_s,
        verifier,
    )
    .map_err(ContinuousClockError::QuorumVerificationFailed)?;
    let witness_digest = verify_clock_window_evaluation_witness(&window, &witness)
        .map_err(ContinuousClockError::SuccessorWitnessInvalid)?;

    if window.trust_snapshot_digest != permit.trust_snapshot_digest {
        return Err(ContinuousClockError::WindowTrustSnapshotMismatch);
    }
    if window.lower_unix_ms < permit.evaluation_lower_unix_ms
        || window.upper_unix_ms > permit.evaluation_upper_unix_ms
    {
        return Err(ContinuousClockError::WindowOutsidePermit);
    }

    let eligible_signers = permit
        .eligible_clock_keys
        .iter()
        .map(|key| (key.algorithm.clone(), key.key_id.clone()))
        .collect::<BTreeSet<_>>();
    for signer in &witness.signers {
        if !eligible_signers.contains(&(signer.algorithm.clone(), signer.key_id.clone())) {
            return Err(ContinuousClockError::UnpermittedSigner(signer.key_id.clone()));
        }
    }
    for observation in &witness.observations {
        let lower = observation.observed_unix_ms.saturating_sub(observation.uncertainty_ms);
        let upper = observation
            .observed_unix_ms
            .checked_add(observation.uncertainty_ms)
            .ok_or_else(|| ContinuousClockError::ObservationIntervalOverflow(
                observation.source_id.clone(),
            ))?;
        if lower < permit.evaluation_lower_unix_ms || upper > permit.evaluation_upper_unix_ms {
            return Err(ContinuousClockError::ObservationIntervalOutsidePermit(
                observation.source_id.clone(),
            ));
        }
    }

    let continuity_runtime = permit
        .basis
        .continuity_policy
        .to_runtime_policy()
        .map_err(ContinuousClockError::EvaluationPolicyInvalid)?;
    let continuity = verify_clock_continuity(
        &permit.basis.window,
        &window,
        &continuity_runtime,
    )
    .map_err(ContinuousClockError::ContinuityInvalid)?;

    let wire_basis_id = ContinuousClockBasisIdV1(compute_v6_basis_id(
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

    Ok(ContinuousClockBasisV1 {
        wire_basis_id,
        evaluation_policy: permit.basis.evaluation_policy.clone(),
        quorum_policy: permit.basis.quorum_policy.clone(),
        continuity_policy: permit.basis.continuity_policy.clone(),
        trust_snapshot: permit.basis.trust_snapshot.clone(),
        window,
        witness,
        continuity,
    })
}

fn validate_continuous_basis(basis: &ContinuousClockBasisV1) -> Result<(), ContinuousClockError> {
    basis.evaluation_policy.validate().map_err(ContinuousClockError::EvaluationPolicyInvalid)?;
    basis.quorum_policy.validate().map_err(ContinuousClockError::EvaluationPolicyInvalid)?;
    basis.continuity_policy.validate().map_err(ContinuousClockError::EvaluationPolicyInvalid)?;
    if basis.evaluation_policy.clock_quorum_policy_id() != basis.quorum_policy.id() {
        return Err(ContinuousClockError::QuorumPolicyMismatch);
    }
    if basis.evaluation_policy.clock_continuity_policy_id() != basis.continuity_policy.id() {
        return Err(ContinuousClockError::ContinuityPolicyMismatch);
    }
    basis.trust_snapshot.validate().map_err(ContinuousClockError::TrustSnapshotInvalid)?;
    let snapshot_digest =
        digest_trust_snapshot(&basis.trust_snapshot).map_err(ContinuousClockError::TrustSnapshotInvalid)?;
    if snapshot_digest != basis.window.trust_snapshot_digest {
        return Err(ContinuousClockError::TrustSnapshotMismatch);
    }
    basis.window.validate().map_err(|_| ContinuousClockError::PriorWindowInvalid)?;
    let witness_digest = verify_clock_window_evaluation_witness(&basis.window, &basis.witness)
        .map_err(ContinuousClockError::PriorWitnessInvalid)?;
    if witness_digest != basis.witness.witness_digest {
        return Err(ContinuousClockError::PriorWitnessMismatch);
    }
    let continuity_digest = digest_clock_continuity(&basis.continuity)
        .map_err(ContinuousClockError::PriorContinuityInvalid)?;
    if continuity_digest != basis.continuity.continuity_digest
        || basis.continuity.successor_evidence_digest != basis.window.evidence_digest
        || basis.continuity.successor_epoch != basis.window.epoch
    {
        return Err(ContinuousClockError::PriorContinuityMismatch);
    }
    Ok(())
}

fn validate_continuous_permit(
    permit: &ContinuousClockSuccessorPermitV1,
) -> Result<(), ContinuousClockError> {
    validate_continuous_basis(&permit.basis)?;
    if permit.prior_basis_id != permit.basis.wire_basis_id
        || permit.prior_clock_window_evidence_digest != permit.basis.window.evidence_digest
        || permit.evaluation_policy_id != permit.basis.evaluation_policy.id()
        || permit.clock_quorum_policy_id != permit.basis.quorum_policy.id()
        || permit.clock_continuity_policy_id != permit.basis.continuity_policy.id()
    {
        return Err(ContinuousClockError::PermitIdentityMismatch);
    }
    let snapshot_digest =
        digest_trust_snapshot(&permit.basis.trust_snapshot).map_err(ContinuousClockError::TrustSnapshotInvalid)?;
    if snapshot_digest != permit.trust_snapshot_digest {
        return Err(ContinuousClockError::TrustSnapshotMismatch);
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
        return Err(ContinuousClockError::PermitIdentityMismatch);
    }
    Ok(())
}

fn eligible_keys_for_envelope(
    snapshot: &TrustSnapshot,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    policy: &ClockEvaluationPolicyV4,
) -> Result<Vec<ContinuousClockAuthorityKeyV1>, ContinuousClockError> {
    let snapshot_lower_ms = seconds_to_millis(snapshot.issued_at_unix_s)?;
    let snapshot_upper_ms = seconds_to_millis(snapshot.expires_at_unix_s)?;
    if lower_unix_ms > upper_unix_ms
        || lower_unix_ms < snapshot_lower_ms
        || upper_unix_ms >= snapshot_upper_ms
    {
        return Err(ContinuousClockError::SnapshotNotValidForEnvelope);
    }
    let mut eligible = Vec::new();
    for key in &snapshot.keys {
        if key.status != KeyLifecycleStatus::Active || !key.usages.contains(&KeyUsage::ClockAuthority) {
            continue;
        }
        let key_lower_ms = seconds_to_millis(key.not_before_unix_s)?;
        if lower_unix_ms < key_lower_ms {
            continue;
        }
        if let Some(not_after_unix_s) = key.not_after_unix_s {
            let key_upper_ms = seconds_to_millis(not_after_unix_s)?;
            if upper_unix_ms >= key_upper_ms { continue; }
        }
        eligible.push(ContinuousClockAuthorityKeyV1 {
            algorithm: key.algorithm.clone(),
            key_id: key.key_id.clone(),
        });
    }
    eligible.sort_by(|left, right| {
        (&left.algorithm, left.key_id.as_str()).cmp(&(&right.algorithm, right.key_id.as_str()))
    });
    if eligible.len() < policy.minimum_eligible_clock_authority_keys() {
        return Err(ContinuousClockError::InsufficientClockAuthorityKeys {
            actual: eligible.len(),
            required: policy.minimum_eligible_clock_authority_keys(),
        });
    }
    if policy.require_eligible_algorithm_diversity() {
        let algorithms = eligible.iter().map(|key| key.algorithm.clone()).collect::<BTreeSet<_>>();
        if algorithms.len() < 2 {
            return Err(ContinuousClockError::EligibleAlgorithmDiversityMissing);
        }
    }
    Ok(eligible)
}

#[allow(clippy::too_many_arguments)]
fn compute_successor_permit_id(
    prior_basis_id: ContinuousClockBasisIdV1,
    prior_window_digest: Sha256Digest,
    evaluation_policy_id: ClockEvaluationPolicyIdV4,
    quorum_policy_id: ClockQuorumPolicyRevisionIdV1,
    continuity_policy_id: ClockContinuityPolicyRevisionIdV1,
    trust_snapshot_digest: Sha256Digest,
    evaluation_lower_unix_ms: u64,
    evaluation_upper_unix_ms: u64,
    minimum_eligible_clock_authority_keys: usize,
    require_eligible_algorithm_diversity: bool,
    eligible_clock_keys: &[ContinuousClockAuthorityKeyV1],
) -> Result<Sha256Digest, ContinuousClockError> {
    let eligible_keys_json = eligible_clock_keys.iter().map(|key| {
        Value::Array(vec![
            Value::String(algorithm_identity(&key.algorithm)),
            Value::String(key.key_id.clone()),
        ])
    }).collect::<Vec<_>>();
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
fn compute_v6_basis_id(
    successor_permit_id: ContinuousClockSuccessorPermitIdV1,
    prior_basis_id: ContinuousClockBasisIdV1,
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
) -> Result<Sha256Digest, ContinuousClockError> {
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
        ("schema", Value::String(ACCEPTED_CLOCK_BASIS_V6_WIRE_SCHEMA.to_string())),
        ("successor_permit_id", Value::String(successor_permit_id.to_hex())),
        ("trust_snapshot_digest", Value::String(trust_snapshot_digest.to_hex())),
        ("upper_unix_ms", Value::from(upper_unix_ms)),
    ])?;
    Ok(domain_hash(ACCEPTED_CLOCK_BASIS_V6_DOMAIN, &bytes))
}

fn algorithm_identity(algorithm: &SignatureAlgorithm) -> String {
    match algorithm {
        SignatureAlgorithm::Ed25519 => "Ed25519".to_string(),
        SignatureAlgorithm::MlDsa65 => "MlDsa65".to_string(),
        SignatureAlgorithm::MlDsa87 => "MlDsa87".to_string(),
        SignatureAlgorithm::Other(name) => format!("Other:{name}"),
    }
}

fn seconds_to_millis(value: u64) -> Result<u64, ContinuousClockError> {
    value.checked_mul(1_000).ok_or(ContinuousClockError::TimeScaleOverflow)
}

fn canonical_json_bytes<const N: usize>(
    entries: [(&str, Value); N],
) -> Result<Vec<u8>, ContinuousClockError> {
    let map = entries.into_iter().map(|(key, value)| (key.to_string(), value)).collect::<BTreeMap<_, _>>();
    serde_json::to_vec(&map).map_err(|error| ContinuousClockError::Encoding(error.to_string()))
}
