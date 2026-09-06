// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signature, VerifyingKey};
use symthaea_authority::Digest32;
use symthaea_iot_actuation_effect_reconciliation_challenge::EffectReconciliationChallengeV1;

use crate::{
    EffectOutcomeClaimV1, EffectOutcomeError, EffectOutcomePolicyV1, EffectOutcomeTrustHead,
    EffectOutcomeTrustRegistry, PhysicalEffectOutcomeEvidenceV1,
};

/// Guard-owned fixed verifier for one exact device-class physical-effect outcome profile.
#[derive(Debug)]
pub struct GuardPhysicalEffectOutcomeState {
    policy: EffectOutcomePolicyV1,
    anchored_policy_digest: Digest32,
    trust_registry: EffectOutcomeTrustRegistry,
    anchored_trust_head: EffectOutcomeTrustHead,
}

impl GuardPhysicalEffectOutcomeState {
    pub fn new(
        policy: EffectOutcomePolicyV1,
        anchored_policy_digest: Digest32,
        trust_registry: EffectOutcomeTrustRegistry,
        anchored_trust_head: EffectOutcomeTrustHead,
    ) -> Result<Self, EffectOutcomeError> {
        if policy.digest()? != anchored_policy_digest {
            return Err(EffectOutcomeError::AnchoredPolicyDigestMismatch);
        }
        if trust_registry.head() != anchored_trust_head {
            return Err(EffectOutcomeError::AnchoredTrustHeadMismatch);
        }
        Ok(Self {
            policy,
            anchored_policy_digest,
            trust_registry,
            anchored_trust_head,
        })
    }

    pub const fn anchored_policy_digest(&self) -> Digest32 {
        self.anchored_policy_digest
    }

    pub const fn anchored_trust_head(&self) -> EffectOutcomeTrustHead {
        self.anchored_trust_head
    }

    pub fn verify_evidence(
        &self,
        evidence: PhysicalEffectOutcomeEvidenceV1,
        challenge: &EffectReconciliationChallengeV1,
    ) -> Result<VerifiedPhysicalEffectOutcomeEvidence, EffectOutcomeError> {
        self.verify_evidence_at(evidence, challenge, system_unix_ms()?)
    }

    pub(crate) fn verify_evidence_at(
        &self,
        evidence: PhysicalEffectOutcomeEvidenceV1,
        challenge: &EffectReconciliationChallengeV1,
        now_unix_ms: u64,
    ) -> Result<VerifiedPhysicalEffectOutcomeEvidence, EffectOutcomeError> {
        self.policy.validate()?;
        if self.policy.digest()? != self.anchored_policy_digest {
            return Err(EffectOutcomeError::AnchoredPolicyDigestMismatch);
        }
        if self.trust_registry.head() != self.anchored_trust_head {
            return Err(EffectOutcomeError::AnchoredTrustHeadMismatch);
        }

        challenge
            .validate()
            .map_err(|_| EffectOutcomeError::InvalidReconciliationChallenge)?;
        if !challenge.is_fresh_at(now_unix_ms) {
            return Err(EffectOutcomeError::ReconciliationChallengeNotFresh);
        }
        evidence.validate_structure()?;
        let body = &evidence.body;

        if body.device != *challenge.device() || body.device != self.policy.device {
            return Err(EffectOutcomeError::EvidenceDeviceMismatch);
        }
        if body.operation != *challenge.operation() || body.operation != self.policy.operation {
            return Err(EffectOutcomeError::EvidenceOperationMismatch);
        }
        if body.executor != *challenge.executor() {
            return Err(EffectOutcomeError::EvidenceExecutorMismatch);
        }
        let challenge_digest = challenge
            .digest()
            .map_err(|_| EffectOutcomeError::InvalidReconciliationChallenge)?;
        if body.challenge_digest != challenge_digest {
            return Err(EffectOutcomeError::EvidenceChallengeMismatch);
        }
        if body.command_digest != challenge.command_digest() {
            return Err(EffectOutcomeError::EvidenceCommandMismatch);
        }
        if body.sequence != challenge.sequence() {
            return Err(EffectOutcomeError::EvidenceSequenceMismatch);
        }
        if body.outcome_profile_digest != self.policy.exact_outcome_profile_digest {
            return Err(EffectOutcomeError::EvidenceOutcomeProfileMismatch);
        }
        if !self
            .policy
            .accepted_reference_values
            .contains(&body.reference_values_digest)
        {
            return Err(EffectOutcomeError::EvidenceReferenceValuesDenied);
        }
        if body.appraisal_policy_digest != self.policy.exact_appraisal_policy_digest {
            return Err(EffectOutcomeError::EvidenceAppraisalPolicyMismatch);
        }
        if !self.policy.allowed_verifier_ids.contains(&body.verifier_id) {
            return Err(EffectOutcomeError::EvidenceVerifierDenied);
        }
        if !self.policy.allowed_claim_kinds.contains(&body.claim.kind()) {
            return Err(EffectOutcomeError::EvidenceClaimKindDenied);
        }

        if body.evidence_issued_at_unix_ms < challenge.issued_at_unix_ms() {
            return Err(EffectOutcomeError::EvidencePredatesChallenge);
        }
        if body.evidence_issued_at_unix_ms < self.trust_registry.snapshot().issued_at_unix_ms {
            return Err(EffectOutcomeError::EvidencePredatesCurrentTrustGeneration);
        }
        if body.evidence_expires_at_unix_ms > challenge.expires_at_unix_ms() {
            return Err(EffectOutcomeError::EvidenceOutlivesChallenge);
        }
        if now_unix_ms < body.evidence_issued_at_unix_ms
            || now_unix_ms >= body.evidence_expires_at_unix_ms
        {
            return Err(EffectOutcomeError::EvidenceNotFresh);
        }
        let lifetime_ms = body
            .evidence_expires_at_unix_ms
            .checked_sub(body.evidence_issued_at_unix_ms)
            .ok_or(EffectOutcomeError::InvalidEvidenceWindow)?;
        if lifetime_ms == 0 || lifetime_ms > self.policy.max_evidence_lifetime_ms {
            return Err(EffectOutcomeError::EvidenceLifetimeExceedsPolicy);
        }

        validate_claim_causality(body.claim, body.evidence_issued_at_unix_ms, challenge)?;

        let current_key = self.trust_registry.exact_active_key(body, now_unix_ms)?;
        if lifetime_ms > current_key.max_evidence_lifetime_ms {
            return Err(EffectOutcomeError::EvidenceLifetimeExceedsPolicy);
        }
        let message = body.signature_message()?;
        let signature = Signature::try_from(evidence.signature.as_ref())
            .map_err(|_| EffectOutcomeError::InvalidEvidenceSignature)?;
        let verifying_key = VerifyingKey::from_bytes(&current_key.public_key)
            .map_err(|_| EffectOutcomeError::InvalidVerifierPublicKey)?;
        verifying_key
            .verify_strict(&message, &signature)
            .map_err(|_| EffectOutcomeError::InvalidEvidenceSignature)?;

        let evidence_digest = evidence.digest()?;
        let body_digest = body.digest()?;
        let verifier_id = current_key.verifier_id.clone();
        let key_id = current_key.key_id.clone();
        let key_digest = current_key.digest()?;
        let verifier_key_not_after_unix_ms = current_key.not_after_unix_ms;
        let trust_snapshot_expires_at_unix_ms = self.trust_registry.snapshot().expires_at_unix_ms;
        let challenge_expires_at_unix_ms = challenge.expires_at_unix_ms();
        let valid_until_unix_ms = body
            .evidence_expires_at_unix_ms
            .min(verifier_key_not_after_unix_ms)
            .min(trust_snapshot_expires_at_unix_ms)
            .min(challenge_expires_at_unix_ms);
        if now_unix_ms >= valid_until_unix_ms {
            return Err(EffectOutcomeError::EvidenceNotFresh);
        }

        Ok(VerifiedPhysicalEffectOutcomeEvidence {
            evidence,
            evidence_digest,
            body_digest,
            challenge_digest,
            challenge_journal_generation: challenge.journal_generation(),
            challenge_journal_digest: challenge.journal_digest(),
            challenge_expires_at_unix_ms,
            verifier_id,
            key_id,
            key_digest,
            trust_head: self.anchored_trust_head,
            policy_digest: self.anchored_policy_digest,
            verified_at_unix_ms: now_unix_ms,
            verifier_key_not_after_unix_ms,
            trust_snapshot_expires_at_unix_ms,
            valid_until_unix_ms,
        })
    }
}

fn validate_claim_causality(
    claim: EffectOutcomeClaimV1,
    evidence_issued_at_unix_ms: u64,
    challenge: &EffectReconciliationChallengeV1,
) -> Result<(), EffectOutcomeError> {
    match claim {
        EffectOutcomeClaimV1::ExecutionAndPostcondition {
            effect_recorded_at_unix_ms,
            postcondition_observed_at_unix_ms,
            ..
        } => {
            if effect_recorded_at_unix_ms < challenge.attempt_common_fenced_at_unix_ms()
                || effect_recorded_at_unix_ms >= challenge.attempt_wall_valid_until_unix_ms()
            {
                return Err(EffectOutcomeError::ExecutionRecordOutsideActuationWindow);
            }
            if postcondition_observed_at_unix_ms < challenge.issued_at_unix_ms() {
                return Err(EffectOutcomeError::PostconditionObservationNotFresh);
            }
            if postcondition_observed_at_unix_ms < effect_recorded_at_unix_ms
                || postcondition_observed_at_unix_ms > evidence_issued_at_unix_ms
            {
                return Err(EffectOutcomeError::PostconditionObservationTimeInvalid);
            }
        }
        EffectOutcomeClaimV1::NonExecution {
            coverage_from_unix_ms,
            coverage_through_unix_ms,
            ..
        } => {
            if coverage_from_unix_ms > challenge.attempt_common_fenced_at_unix_ms()
                || coverage_through_unix_ms < challenge.attempt_wall_valid_until_unix_ms()
            {
                return Err(EffectOutcomeError::NonExecutionCoverageIncomplete);
            }
            if coverage_through_unix_ms > evidence_issued_at_unix_ms {
                return Err(EffectOutcomeError::NonExecutionCoverageAfterEvidence);
            }
        }
    }
    Ok(())
}

/// Opaque proof that one exact class-specific outcome claim passed fixed cryptography and the
/// guard-owned policy/trust boundary at `verified_at_unix_ms`.
///
/// This object does not close the effect-attempt journal and is not itself a terminal reconciliation
/// decision. A later current fence and journal-head equality check are still required.
#[derive(Debug)]
pub struct VerifiedPhysicalEffectOutcomeEvidence {
    evidence: PhysicalEffectOutcomeEvidenceV1,
    evidence_digest: Digest32,
    body_digest: Digest32,
    challenge_digest: Digest32,
    challenge_journal_generation: u64,
    challenge_journal_digest: Digest32,
    challenge_expires_at_unix_ms: u64,
    verifier_id: String,
    key_id: String,
    key_digest: Digest32,
    trust_head: EffectOutcomeTrustHead,
    policy_digest: Digest32,
    verified_at_unix_ms: u64,
    verifier_key_not_after_unix_ms: u64,
    trust_snapshot_expires_at_unix_ms: u64,
    valid_until_unix_ms: u64,
}

impl VerifiedPhysicalEffectOutcomeEvidence {
    pub fn evidence(&self) -> &PhysicalEffectOutcomeEvidenceV1 {
        &self.evidence
    }

    pub const fn evidence_digest(&self) -> Digest32 {
        self.evidence_digest
    }

    pub const fn body_digest(&self) -> Digest32 {
        self.body_digest
    }

    pub const fn challenge_digest(&self) -> Digest32 {
        self.challenge_digest
    }

    pub const fn challenge_journal_generation(&self) -> u64 {
        self.challenge_journal_generation
    }

    pub const fn challenge_journal_digest(&self) -> Digest32 {
        self.challenge_journal_digest
    }

    pub const fn challenge_expires_at_unix_ms(&self) -> u64 {
        self.challenge_expires_at_unix_ms
    }

    pub fn verifier_id(&self) -> &str {
        &self.verifier_id
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub const fn key_digest(&self) -> Digest32 {
        self.key_digest
    }

    pub const fn trust_head(&self) -> EffectOutcomeTrustHead {
        self.trust_head
    }

    pub const fn policy_digest(&self) -> Digest32 {
        self.policy_digest
    }

    pub const fn verified_at_unix_ms(&self) -> u64 {
        self.verified_at_unix_ms
    }

    pub const fn verifier_key_not_after_unix_ms(&self) -> u64 {
        self.verifier_key_not_after_unix_ms
    }

    pub const fn trust_snapshot_expires_at_unix_ms(&self) -> u64 {
        self.trust_snapshot_expires_at_unix_ms
    }

    pub const fn valid_until_unix_ms(&self) -> u64 {
        self.valid_until_unix_ms
    }
}

fn system_unix_ms() -> Result<u64, EffectOutcomeError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| EffectOutcomeError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| EffectOutcomeError::SystemClockOverflow)
}
