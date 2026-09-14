// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interval-safe, exact-evidence trust-snapshot rotation authority.
//!
//! The kernel's portable signed rotation proposal remains the wire/evidence format. This bridge
//! replaces caller-selected scalar evaluation time with an operational-clock interval, requalifies
//! every rotation signer against real lifecycle and compromise state, uses checked transition
//! arithmetic, and independently verifies every exact raw rotation signature before minting live
//! authority.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::rotation::{
    KeyRotationPolicy, MAX_ROTATION_KEY_ID_BYTES, MAX_ROTATION_SIGNATURE_BYTES,
    MAX_ROTATION_SIGNATURES, SignedTrustRotationProposal, TrustRotationProposal,
    canonical_rotation_proposal_bytes, digest_rotation_policy, digest_rotation_proposal,
};
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceEvaluationEnvelopeV1,
    ClockGovernanceTimeError, OperationalClockBasisIdV1, OperationalClockBasisV1,
    derive_clock_governance_evaluation_envelope_v1,
};

pub const CLOCK_GOVERNED_TRUST_ROTATION_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-trust-rotation.v1";
pub const MAX_TRUST_ROTATION_EXACT_VERIFIERS: usize = 16;

const SIGNED_ROTATION_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-trust-rotation-signed-evidence.v1\0";
const SIGNER_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-trust-rotation-signer-set.v1\0";
const VERIFIER_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-trust-rotation-verifier-set.v1\0";
const AUTHORITY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-trust-rotation.v1\0";

pub trait ExactTrustRotationVerifierV1 {
    fn provider_id(&self) -> &str;
    fn verification_policy_digest(&self) -> Sha256Digest;
    fn verify_rotation_signature(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactTrustRotationVerificationPolicyV1 {
    pub minimum_distinct_providers: usize,
    pub maximum_providers: usize,
}

impl Default for ExactTrustRotationVerificationPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_distinct_providers: 2,
            maximum_providers: 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedTrustRotationIdV1(Sha256Digest);

impl ClockGovernedTrustRotationIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedTrustRotationV1 {
    id: ClockGovernedTrustRotationIdV1,
    proposal: TrustRotationProposal,
    current_snapshot_digest: Sha256Digest,
    proposed_snapshot_digest: Sha256Digest,
    proposal_digest: Sha256Digest,
    signed_evidence_digest: Sha256Digest,
    policy_digest: Sha256Digest,
    signer_set_digest: Sha256Digest,
    signer_count: usize,
    containment_state_digest: Sha256Digest,
    containment_generation: u64,
    compromise_tracker_digest: Sha256Digest,
    verifier_set_digest: Sha256Digest,
    verifier_count: usize,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
    activates_at_unix_ms: u64,
    overlap_required_until_unix_ms: u64,
}

impl ClockGovernedTrustRotationV1 {
    pub fn id(&self) -> ClockGovernedTrustRotationIdV1 {
        self.id
    }
    pub fn proposal(&self) -> &TrustRotationProposal {
        &self.proposal
    }
    pub fn proposed_snapshot(&self) -> &TrustSnapshot {
        &self.proposal.proposed_snapshot
    }
    pub fn current_snapshot_digest(&self) -> Sha256Digest {
        self.current_snapshot_digest
    }
    pub fn proposed_snapshot_digest(&self) -> Sha256Digest {
        self.proposed_snapshot_digest
    }
    pub fn proposal_digest(&self) -> Sha256Digest {
        self.proposal_digest
    }
    pub fn signed_evidence_digest(&self) -> Sha256Digest {
        self.signed_evidence_digest
    }
    pub fn policy_digest(&self) -> Sha256Digest {
        self.policy_digest
    }
    pub fn signer_set_digest(&self) -> Sha256Digest {
        self.signer_set_digest
    }
    pub fn signer_count(&self) -> usize {
        self.signer_count
    }
    pub fn containment_state_digest(&self) -> Sha256Digest {
        self.containment_state_digest
    }
    pub fn containment_generation(&self) -> u64 {
        self.containment_generation
    }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }
    pub fn verifier_set_digest(&self) -> Sha256Digest {
        self.verifier_set_digest
    }
    pub fn verifier_count(&self) -> usize {
        self.verifier_count
    }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
    pub fn activates_at_unix_ms(&self) -> u64 {
        self.activates_at_unix_ms
    }
    pub fn overlap_required_until_unix_ms(&self) -> u64 {
        self.overlap_required_until_unix_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockGovernedTrustRotationError {
    InvalidPolicy(String),
    CurrentSnapshotInvalid(String),
    ProposedSnapshotInvalid(String),
    CurrentSnapshotNotValidAcrossEnvelope(ClockGovernanceTimeError),
    CurrentSnapshotDigestMismatch,
    SequenceOverflow,
    SequenceNotAdjacent { current: u64, proposed: u64 },
    ActivationIssueMismatch,
    ActivationMayBePast,
    ActivationMayBeTooLate,
    EmergencyRotationNotAllowed,
    TimeScaleOverflow,
    OverlapOverflow,
    KeyRemovedWithoutLifecycleRecord(String),
    UsageCoverageMissing { usage: KeyUsage, actual: usize, required: usize },
    OverlapCoverageMissing { usage: KeyUsage, actual: usize, required: usize },
    TooManySignatures { actual: usize, maximum: usize },
    InvalidSigner(String),
    DuplicateSigner(String),
    SignerUnknown(String),
    SignerNotActive(String),
    SignerUsageNotAllowed(String),
    SignerNotValidAcrossEnvelope {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    SignerCompromisedAcrossEnvelope(String),
    CompromiseTimeInvalid { key_id: String, reason: ClockGovernanceTimeError },
    InsufficientSignatures { actual: usize, required: usize },
    MissingAlgorithmDiversity,
    ContainmentStateInvalid(String),
    InvalidVerificationPolicy,
    InsufficientVerificationProviders { actual: usize, required: usize },
    TooManyVerificationProviders { actual: usize, maximum: usize },
    InvalidVerificationProvider(String),
    DuplicateVerificationProvider(String),
    SignatureRejected { provider: String, key_id: String },
    VerificationProviderError { provider: String, reason: String },
    ProposalEncoding(String),
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct VerifierCommitment {
    provider_id: String,
    verification_policy_digest: String,
}

#[derive(Debug, Clone, Serialize)]
struct TrustRotationAuthorityCommitment {
    schema: &'static str,
    current_snapshot_digest: String,
    proposed_snapshot_digest: String,
    current_snapshot_sequence: u64,
    proposed_snapshot_sequence: u64,
    proposal_digest: String,
    signed_evidence_digest: String,
    policy_digest: String,
    signer_set_digest: String,
    signer_count: usize,
    containment_state_digest: String,
    containment_generation: u64,
    compromise_tracker_digest: String,
    verifier_set_digest: String,
    verifier_count: usize,
    clock_envelope_id: String,
    operational_basis_id: String,
    activates_at_unix_ms: u64,
    overlap_required_until_unix_ms: u64,
    emergency: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_clock_governed_trust_rotation_v1(
    signed: &SignedTrustRotationProposal,
    current_snapshot: &TrustSnapshot,
    policy: &KeyRotationPolicy,
    containment_state: &FabricationContainmentState,
    operational_basis: &OperationalClockBasisV1,
    verification_policy: &ExactTrustRotationVerificationPolicyV1,
    verification_providers: &[&dyn ExactTrustRotationVerifierV1],
) -> Result<ClockGovernedTrustRotationV1, Vec<ClockGovernedTrustRotationError>> {
    let mut violations = Vec::new();

    if let Err(errors) = policy.validate() {
        violations.push(ClockGovernedTrustRotationError::InvalidPolicy(format!(
            "{errors:?}"
        )));
    }
    if let Err(error) = current_snapshot.validate() {
        violations.push(ClockGovernedTrustRotationError::CurrentSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if let Err(error) = signed.proposal.proposed_snapshot.validate() {
        violations.push(ClockGovernedTrustRotationError::ProposedSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if let Err(error) = containment_state.validate() {
        violations.push(ClockGovernedTrustRotationError::ContainmentStateInvalid(format!(
            "{error:?}"
        )));
    }

    let clock = match derive_clock_governance_evaluation_envelope_v1(operational_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(ClockGovernedTrustRotationError::CurrentSnapshotNotValidAcrossEnvelope(
                error,
            ));
            return Err(violations);
        }
    };
    if let Err(reason) = clock.require_valid_across_seconds_window(
        current_snapshot.issued_at_unix_s,
        current_snapshot.expires_at_unix_s,
    ) {
        violations.push(
            ClockGovernedTrustRotationError::CurrentSnapshotNotValidAcrossEnvelope(reason),
        );
    }

    let current_snapshot_digest = match digest_trust_snapshot(current_snapshot) {
        Ok(value) => value,
        Err(error) => {
            violations.push(ClockGovernedTrustRotationError::CurrentSnapshotInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if signed.proposal.current_snapshot_digest != current_snapshot_digest {
        violations.push(ClockGovernedTrustRotationError::CurrentSnapshotDigestMismatch);
    }
    let expected_sequence = match current_snapshot.sequence.checked_add(1) {
        Some(value) => value,
        None => {
            violations.push(ClockGovernedTrustRotationError::SequenceOverflow);
            0
        }
    };
    if signed.proposal.proposed_snapshot.sequence != expected_sequence {
        violations.push(ClockGovernedTrustRotationError::SequenceNotAdjacent {
            current: current_snapshot.sequence,
            proposed: signed.proposal.proposed_snapshot.sequence,
        });
    }
    if signed.proposal.activates_at_unix_s
        != signed.proposal.proposed_snapshot.issued_at_unix_s
    {
        violations.push(ClockGovernedTrustRotationError::ActivationIssueMismatch);
    }
    if signed.proposal.emergency && !policy.allow_emergency_revocation {
        violations.push(ClockGovernedTrustRotationError::EmergencyRotationNotAllowed);
    }

    let activates_at_unix_ms = match seconds_to_millis(signed.proposal.activates_at_unix_s) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            0
        }
    };
    if activates_at_unix_ms < clock.upper_unix_ms() {
        violations.push(ClockGovernedTrustRotationError::ActivationMayBePast);
    }
    let maximum_delay_ms = match seconds_to_millis(policy.maximum_activation_delay_s) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            0
        }
    };
    let latest_activation = match clock.lower_unix_ms().checked_add(maximum_delay_ms) {
        Some(value) => value,
        None => {
            violations.push(ClockGovernedTrustRotationError::TimeScaleOverflow);
            0
        }
    };
    if activates_at_unix_ms > latest_activation {
        violations.push(ClockGovernedTrustRotationError::ActivationMayBeTooLate);
    }
    let overlap_ms = match seconds_to_millis(policy.minimum_overlap_s) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            0
        }
    };
    let overlap_required_until_unix_ms = match activates_at_unix_ms.checked_add(overlap_ms) {
        Some(value) => value,
        None => {
            violations.push(ClockGovernedTrustRotationError::OverlapOverflow);
            activates_at_unix_ms
        }
    };

    validate_snapshot_transition(
        current_snapshot,
        &signed.proposal,
        policy,
        containment_state,
        overlap_required_until_unix_ms,
        &mut violations,
    );

    if signed.signatures.len() > policy.maximum_signatures
        || signed.signatures.len() > MAX_ROTATION_SIGNATURES
    {
        violations.push(ClockGovernedTrustRotationError::TooManySignatures {
            actual: signed.signatures.len(),
            maximum: policy.maximum_signatures.min(MAX_ROTATION_SIGNATURES),
        });
    }

    let proposal_bytes = match canonical_rotation_proposal_bytes(&signed.proposal) {
        Ok(value) => value,
        Err(error) => {
            violations.push(ClockGovernedTrustRotationError::ProposalEncoding(format!(
                "{error:?}"
            )));
            Vec::new()
        }
    };
    let mut signer_identities = BTreeSet::new();
    for signature in signed.signatures.iter().take(MAX_ROTATION_SIGNATURES) {
        if !signature.algorithm.is_canonical()
            || invalid_identifier(&signature.key_id)
            || signature.signature.is_empty()
            || signature.signature.len() > MAX_ROTATION_SIGNATURE_BYTES
        {
            violations.push(ClockGovernedTrustRotationError::InvalidSigner(
                signature.key_id.clone(),
            ));
            continue;
        }
        if !signer_identities.insert((signature.algorithm.clone(), signature.key_id.clone())) {
            violations.push(ClockGovernedTrustRotationError::DuplicateSigner(
                signature.key_id.clone(),
            ));
            continue;
        }
        requalify_rotation_signer(
            &signature.algorithm,
            &signature.key_id,
            current_snapshot,
            containment_state,
            &clock,
            &mut violations,
        );
    }
    if signer_identities.len() < policy.minimum_valid_signatures {
        violations.push(ClockGovernedTrustRotationError::InsufficientSignatures {
            actual: signer_identities.len(),
            required: policy.minimum_valid_signatures,
        });
    }
    if policy.require_algorithm_diversity
        && signer_identities
            .iter()
            .map(|(algorithm, _)| algorithm)
            .collect::<BTreeSet<_>>()
            .len()
            < 2
    {
        violations.push(ClockGovernedTrustRotationError::MissingAlgorithmDiversity);
    }

    if !valid_verification_policy(verification_policy) {
        violations.push(ClockGovernedTrustRotationError::InvalidVerificationPolicy);
    }
    if verification_providers.len() < verification_policy.minimum_distinct_providers {
        violations.push(
            ClockGovernedTrustRotationError::InsufficientVerificationProviders {
                actual: verification_providers.len(),
                required: verification_policy.minimum_distinct_providers,
            },
        );
    }
    if verification_providers.len() > verification_policy.maximum_providers
        || verification_providers.len() > MAX_TRUST_ROTATION_EXACT_VERIFIERS
    {
        violations.push(ClockGovernedTrustRotationError::TooManyVerificationProviders {
            actual: verification_providers.len(),
            maximum: verification_policy
                .maximum_providers
                .min(MAX_TRUST_ROTATION_EXACT_VERIFIERS),
        });
        return Err(violations);
    }
    let verifier_commitments = match validate_verification_providers(verification_providers) {
        Ok(value) => value,
        Err(mut errors) => {
            violations.append(&mut errors);
            Vec::new()
        }
    };
    for provider in verification_providers {
        for signature in &signed.signatures {
            match provider.verify_rotation_signature(
                &signature.algorithm,
                &signature.key_id,
                &proposal_bytes,
                &signature.signature,
            ) {
                Ok(true) => {}
                Ok(false) => violations.push(ClockGovernedTrustRotationError::SignatureRejected {
                    provider: provider.provider_id().to_string(),
                    key_id: signature.key_id.clone(),
                }),
                Err(reason) => violations.push(
                    ClockGovernedTrustRotationError::VerificationProviderError {
                        provider: provider.provider_id().to_string(),
                        reason: format!("{}: {reason}", signature.key_id),
                    },
                ),
            }
        }
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let proposed_snapshot_digest = digest_trust_snapshot(&signed.proposal.proposed_snapshot)
        .map_err(|error| {
            vec![ClockGovernedTrustRotationError::ProposedSnapshotInvalid(format!(
                "{error:?}"
            ))]
        })?;
    let proposal_digest = digest_rotation_proposal(&signed.proposal)
        .map_err(|error| vec![ClockGovernedTrustRotationError::ProposalEncoding(format!("{error:?}"))])?;
    let policy_digest = digest_rotation_policy(policy)
        .map_err(|errors| vec![ClockGovernedTrustRotationError::InvalidPolicy(format!("{errors:?}"))])?;
    let signed_evidence_digest = hash_serializable(SIGNED_ROTATION_EVIDENCE_DOMAIN, signed)
        .map_err(|error| vec![error])?;
    let signer_set = signer_identities.into_iter().collect::<Vec<_>>();
    let signer_set_digest = hash_serializable(SIGNER_SET_DOMAIN, &signer_set)
        .map_err(|error| vec![error])?;
    let containment_state_digest = digest_containment_state(containment_state).map_err(|error| {
        vec![ClockGovernedTrustRotationError::ContainmentStateInvalid(format!(
            "{error:?}"
        ))]
    })?;
    let compromise_tracker_digest =
        digest_signer_compromise_tracker(&containment_state.signer_compromise_tracker).map_err(
            |error| {
                vec![ClockGovernedTrustRotationError::ContainmentStateInvalid(format!(
                    "{error:?}"
                ))]
            },
        )?;
    let verifier_set_digest = hash_serializable(VERIFIER_SET_DOMAIN, &verifier_commitments)
        .map_err(|error| vec![error])?;

    let commitment = TrustRotationAuthorityCommitment {
        schema: CLOCK_GOVERNED_TRUST_ROTATION_SCHEMA,
        current_snapshot_digest: current_snapshot_digest.to_hex(),
        proposed_snapshot_digest: proposed_snapshot_digest.to_hex(),
        current_snapshot_sequence: current_snapshot.sequence,
        proposed_snapshot_sequence: signed.proposal.proposed_snapshot.sequence,
        proposal_digest: proposal_digest.to_hex(),
        signed_evidence_digest: signed_evidence_digest.to_hex(),
        policy_digest: policy_digest.to_hex(),
        signer_set_digest: signer_set_digest.to_hex(),
        signer_count: signer_set.len(),
        containment_state_digest: containment_state_digest.to_hex(),
        containment_generation: containment_state.generation,
        compromise_tracker_digest: compromise_tracker_digest.to_hex(),
        verifier_set_digest: verifier_set_digest.to_hex(),
        verifier_count: verifier_commitments.len(),
        clock_envelope_id: clock.id().to_hex(),
        operational_basis_id: operational_basis.id().to_hex(),
        activates_at_unix_ms,
        overlap_required_until_unix_ms,
        emergency: signed.proposal.emergency,
    };
    let id = ClockGovernedTrustRotationIdV1(
        hash_serializable(AUTHORITY_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(ClockGovernedTrustRotationV1 {
        id,
        proposal: signed.proposal.clone(),
        current_snapshot_digest,
        proposed_snapshot_digest,
        proposal_digest,
        signed_evidence_digest,
        policy_digest,
        signer_set_digest,
        signer_count: signer_set.len(),
        containment_state_digest,
        containment_generation: containment_state.generation,
        compromise_tracker_digest,
        verifier_set_digest,
        verifier_count: verifier_commitments.len(),
        clock_envelope_id: clock.id(),
        operational_basis_id: operational_basis.id(),
        activates_at_unix_ms,
        overlap_required_until_unix_ms,
    })
}

fn validate_snapshot_transition(
    current: &TrustSnapshot,
    proposal: &TrustRotationProposal,
    policy: &KeyRotationPolicy,
    containment_state: &FabricationContainmentState,
    overlap_required_until_unix_ms: u64,
    violations: &mut Vec<ClockGovernedTrustRotationError>,
) {
    let proposed = &proposal.proposed_snapshot;
    let proposed_identities = proposed
        .keys
        .iter()
        .map(|key| (key.algorithm.clone(), key.key_id.clone()))
        .collect::<BTreeSet<_>>();
    for key in &current.keys {
        if !proposed_identities.contains(&(key.algorithm.clone(), key.key_id.clone())) {
            violations.push(
                ClockGovernedTrustRotationError::KeyRemovedWithoutLifecycleRecord(
                    key.key_id.clone(),
                ),
            );
        }
    }

    for (usage, required) in &policy.minimum_active_keys_per_usage {
        let required = usize::from(*required);
        let active_at_activation = proposed
            .keys
            .iter()
            .filter(|key| {
                eligible_record_at(key, *usage, proposal.activates_at_unix_s)
                    && !compromised_at_or_before(
                        key,
                        *usage,
                        proposal.activates_at_unix_s,
                        containment_state,
                    )
            })
            .count();
        if active_at_activation < required {
            violations.push(ClockGovernedTrustRotationError::UsageCoverageMissing {
                usage: *usage,
                actual: active_at_activation,
                required,
            });
        }

        if !proposal.emergency && policy.minimum_overlap_s > 0 {
            let current_identities = current
                .keys
                .iter()
                .filter(|key| {
                    eligible_record_at(key, *usage, proposal.activates_at_unix_s)
                        && !compromised_before_ms(
                            key,
                            *usage,
                            overlap_required_until_unix_ms,
                            containment_state,
                        )
                })
                .map(|key| (key.algorithm.clone(), key.key_id.clone()))
                .collect::<BTreeSet<_>>();
            let overlap = proposed
                .keys
                .iter()
                .filter(|key| {
                    current_identities.contains(&(key.algorithm.clone(), key.key_id.clone()))
                        && eligible_record_through_ms(
                            key,
                            *usage,
                            proposal.activates_at_unix_s,
                            overlap_required_until_unix_ms,
                        )
                        && !compromised_before_ms(
                            key,
                            *usage,
                            overlap_required_until_unix_ms,
                            containment_state,
                        )
                })
                .count();
            if overlap < required {
                violations.push(ClockGovernedTrustRotationError::OverlapCoverageMissing {
                    usage: *usage,
                    actual: overlap,
                    required,
                });
            }
        }
    }
}

fn eligible_record_at(key: &KeyTrustRecord, usage: KeyUsage, unix_s: u64) -> bool {
    key.status == KeyLifecycleStatus::Active
        && unix_s >= key.not_before_unix_s
        && key
            .not_after_unix_s
            .is_none_or(|not_after| unix_s < not_after)
        && key.usages.contains(&usage)
}

fn eligible_record_through_ms(
    key: &KeyTrustRecord,
    usage: KeyUsage,
    start_unix_s: u64,
    end_unix_ms: u64,
) -> bool {
    if !eligible_record_at(key, usage, start_unix_s) {
        return false;
    }
    match key.not_after_unix_s {
        Some(not_after) => not_after
            .checked_mul(1_000)
            .is_some_and(|not_after_ms| not_after_ms > end_unix_ms),
        None => true,
    }
}

fn compromised_at_or_before(
    key: &KeyTrustRecord,
    usage: KeyUsage,
    unix_s: u64,
    containment_state: &FabricationContainmentState,
) -> bool {
    containment_state.signer_compromise_tracker.is_compromised_at(
        &key.algorithm,
        &key.key_id,
        usage,
        unix_s,
    )
}

fn compromised_before_ms(
    key: &KeyTrustRecord,
    usage: KeyUsage,
    unix_ms: u64,
    containment_state: &FabricationContainmentState,
) -> bool {
    containment_state
        .signer_compromise_tracker
        .records()
        .iter()
        .filter(|record| {
            record.signer.algorithm == key.algorithm
                && record.signer.key_id == key.key_id
                && record.affected_usages.contains(&usage)
        })
        .any(|record| {
            record
                .effective_at_unix_s
                .checked_mul(1_000)
                .is_none_or(|effective_ms| effective_ms < unix_ms)
        })
}

fn requalify_rotation_signer(
    algorithm: &SignatureAlgorithm,
    key_id: &str,
    current_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
    clock: &ClockGovernanceEvaluationEnvelopeV1,
    violations: &mut Vec<ClockGovernedTrustRotationError>,
) {
    let Some(record) = current_snapshot
        .keys
        .iter()
        .find(|record| &record.algorithm == algorithm && record.key_id == key_id)
    else {
        violations.push(ClockGovernedTrustRotationError::SignerUnknown(
            key_id.to_string(),
        ));
        return;
    };
    if record.status != KeyLifecycleStatus::Active {
        violations.push(ClockGovernedTrustRotationError::SignerNotActive(
            key_id.to_string(),
        ));
    }
    if !record.usages.contains(&KeyUsage::TrustRotation) {
        violations.push(ClockGovernedTrustRotationError::SignerUsageNotAllowed(
            key_id.to_string(),
        ));
    }
    if let Err(reason) = clock.require_valid_across_optional_seconds_window(
        record.not_before_unix_s,
        record.not_after_unix_s,
    ) {
        violations.push(ClockGovernedTrustRotationError::SignerNotValidAcrossEnvelope {
            key_id: key_id.to_string(),
            reason,
        });
    }
    for compromise in containment_state
        .signer_compromise_tracker
        .records()
        .iter()
        .filter(|record| {
            &record.signer.algorithm == algorithm
                && record.signer.key_id == key_id
                && record.affected_usages.contains(&KeyUsage::TrustRotation)
        })
    {
        match clock.require_effective_time_after_envelope_seconds(compromise.effective_at_unix_s) {
            Ok(()) => {}
            Err(ClockGovernanceTimeError::EventMayAlreadyBeEffective) => violations.push(
                ClockGovernedTrustRotationError::SignerCompromisedAcrossEnvelope(
                    key_id.to_string(),
                ),
            ),
            Err(reason) => violations.push(
                ClockGovernedTrustRotationError::CompromiseTimeInvalid {
                    key_id: key_id.to_string(),
                    reason,
                },
            ),
        }
    }
}

fn valid_verification_policy(policy: &ExactTrustRotationVerificationPolicyV1) -> bool {
    policy.minimum_distinct_providers > 0
        && policy.maximum_providers > 0
        && policy.minimum_distinct_providers <= policy.maximum_providers
        && policy.maximum_providers <= MAX_TRUST_ROTATION_EXACT_VERIFIERS
}

fn validate_verification_providers(
    providers: &[&dyn ExactTrustRotationVerifierV1],
) -> Result<Vec<VerifierCommitment>, Vec<ClockGovernedTrustRotationError>> {
    let mut violations = Vec::new();
    let mut seen = BTreeSet::new();
    let mut commitments = Vec::with_capacity(providers.len());
    for provider in providers {
        let provider_id = provider.provider_id().to_string();
        let policy_digest = provider.verification_policy_digest();
        if invalid_identifier(&provider_id) || policy_digest == Sha256Digest([0; 32]) {
            violations.push(ClockGovernedTrustRotationError::InvalidVerificationProvider(
                provider_id,
            ));
            continue;
        }
        if !seen.insert(provider_id.clone()) {
            violations.push(ClockGovernedTrustRotationError::DuplicateVerificationProvider(
                provider_id,
            ));
            continue;
        }
        commitments.push(VerifierCommitment {
            provider_id,
            verification_policy_digest: policy_digest.to_hex(),
        });
    }
    commitments.sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
    if violations.is_empty() {
        Ok(commitments)
    } else {
        Err(violations)
    }
}

fn invalid_identifier(value: &str) -> bool {
    value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_ROTATION_KEY_ID_BYTES
        || value.chars().any(char::is_control)
}

fn seconds_to_millis(value: u64) -> Result<u64, ClockGovernedTrustRotationError> {
    value
        .checked_mul(1_000)
        .ok_or(ClockGovernedTrustRotationError::TimeScaleOverflow)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, ClockGovernedTrustRotationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ClockGovernedTrustRotationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_verifier_policy_defaults_to_two_providers() {
        let policy = ExactTrustRotationVerificationPolicyV1::default();
        assert_eq!(policy.minimum_distinct_providers, 2);
        assert!(valid_verification_policy(&policy));
    }

    #[test]
    fn authority_identity_binds_emergency_semantics() {
        let first = TrustRotationAuthorityCommitment {
            schema: CLOCK_GOVERNED_TRUST_ROTATION_SCHEMA,
            current_snapshot_digest: "11".repeat(32),
            proposed_snapshot_digest: "22".repeat(32),
            current_snapshot_sequence: 1,
            proposed_snapshot_sequence: 2,
            proposal_digest: "33".repeat(32),
            signed_evidence_digest: "44".repeat(32),
            policy_digest: "55".repeat(32),
            signer_set_digest: "66".repeat(32),
            signer_count: 2,
            containment_state_digest: "77".repeat(32),
            containment_generation: 1,
            compromise_tracker_digest: "88".repeat(32),
            verifier_set_digest: "99".repeat(32),
            verifier_count: 2,
            clock_envelope_id: "aa".repeat(32),
            operational_basis_id: "bb".repeat(32),
            activates_at_unix_ms: 1_000,
            overlap_required_until_unix_ms: 2_000,
            emergency: false,
        };
        let mut second = first.clone();
        second.emergency = true;
        assert_ne!(
            hash_serializable(AUTHORITY_DOMAIN, &first).unwrap(),
            hash_serializable(AUTHORITY_DOMAIN, &second).unwrap()
        );
    }
}
