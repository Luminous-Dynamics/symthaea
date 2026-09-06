// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Current fencing for one already verified physical-effect outcome proof.
//!
//! Historical verification establishes that one exact signed claim was valid at one instant. A
//! later journal-closing boundary must not assume it remains current after verifier revocation,
//! policy replacement, key expiry, trust-snapshot expiry or challenge expiry.

use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signature, VerifyingKey};
use symthaea_authority::Digest32;

use crate::{
    EffectOutcomeError, EffectOutcomePolicyV1, EffectOutcomeTrustHead, EffectOutcomeTrustRegistry,
    VerifiedPhysicalEffectOutcomeEvidence,
};

#[derive(Debug)]
pub struct CurrentPhysicalEffectOutcomeGuard {
    policy: EffectOutcomePolicyV1,
    anchored_policy_digest: Digest32,
    trust_registry: EffectOutcomeTrustRegistry,
    anchored_trust_head: EffectOutcomeTrustHead,
}

impl CurrentPhysicalEffectOutcomeGuard {
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

    pub fn fence_current<'a>(
        &'a self,
        proof: &'a VerifiedPhysicalEffectOutcomeEvidence,
    ) -> Result<CurrentPhysicalEffectOutcomeFence<'a>, EffectOutcomeError> {
        self.fence_current_at(proof, system_unix_ms()?)
    }

    fn fence_current_at<'a>(
        &'a self,
        proof: &'a VerifiedPhysicalEffectOutcomeEvidence,
        now_unix_ms: u64,
    ) -> Result<CurrentPhysicalEffectOutcomeFence<'a>, EffectOutcomeError> {
        self.policy.validate()?;
        if self.policy.digest()? != self.anchored_policy_digest
            || proof.policy_digest() != self.anchored_policy_digest
        {
            return Err(EffectOutcomeError::CurrentProofPolicyMismatch);
        }
        if self.trust_registry.head() != self.anchored_trust_head
            || proof.trust_head() != self.anchored_trust_head
        {
            return Err(EffectOutcomeError::CurrentProofTrustHeadMismatch);
        }
        if now_unix_ms < proof.verified_at_unix_ms() {
            return Err(EffectOutcomeError::CurrentProofClockRegressed);
        }

        let evidence = proof.evidence();
        evidence.validate_structure()?;
        let body = &evidence.body;
        if body.device != self.policy.device
            || body.operation != self.policy.operation
            || body.outcome_profile_digest != self.policy.exact_outcome_profile_digest
            || body.appraisal_policy_digest != self.policy.exact_appraisal_policy_digest
            || !self
                .policy
                .accepted_reference_values
                .contains(&body.reference_values_digest)
            || !self.policy.allowed_verifier_ids.contains(&body.verifier_id)
            || !self.policy.allowed_claim_kinds.contains(&body.claim.kind())
        {
            return Err(EffectOutcomeError::CurrentProofPolicyMismatch);
        }
        if body.digest()? != proof.body_digest() || evidence.digest()? != proof.evidence_digest() {
            return Err(EffectOutcomeError::CurrentProofCommitmentMismatch);
        }

        let current_key = self.trust_registry.exact_active_key(body, now_unix_ms)?;
        if current_key.verifier_id != proof.verifier_id()
            || current_key.key_id != proof.key_id()
            || current_key.digest()? != proof.key_digest()
        {
            return Err(EffectOutcomeError::CurrentProofVerifierKeyMismatch);
        }

        let message = body.signature_message()?;
        let signature = Signature::try_from(evidence.signature.as_ref())
            .map_err(|_| EffectOutcomeError::InvalidEvidenceSignature)?;
        let verifying_key = VerifyingKey::from_bytes(&current_key.public_key)
            .map_err(|_| EffectOutcomeError::InvalidVerifierPublicKey)?;
        verifying_key
            .verify_strict(&message, &signature)
            .map_err(|_| EffectOutcomeError::InvalidEvidenceSignature)?;

        let verifier_key_not_after_unix_ms = current_key.not_after_unix_ms;
        let trust_snapshot_expires_at_unix_ms = self.trust_registry.snapshot().expires_at_unix_ms;
        let valid_until_unix_ms = body
            .evidence_expires_at_unix_ms
            .min(verifier_key_not_after_unix_ms)
            .min(trust_snapshot_expires_at_unix_ms)
            .min(proof.challenge_expires_at_unix_ms());
        if now_unix_ms >= valid_until_unix_ms {
            return Err(EffectOutcomeError::CurrentProofWindowElapsed);
        }

        Ok(CurrentPhysicalEffectOutcomeFence {
            _guard: self,
            proof,
            fenced_at_unix_ms: now_unix_ms,
            evidence_expires_at_unix_ms: body.evidence_expires_at_unix_ms,
            verifier_key_not_after_unix_ms,
            trust_snapshot_expires_at_unix_ms,
            challenge_expires_at_unix_ms: proof.challenge_expires_at_unix_ms(),
            valid_until_unix_ms,
        })
    }
}

/// Borrowed evidence that the exact signed outcome proof remains current under the exact guard
/// policy/trust/key and every natural expiry boundary.
///
/// This is still not a journal-closing capability. A later writer must additionally compare the
/// current rollback-protected effect-attempt head to the challenge head retained by `proof()`.
#[derive(Debug)]
pub struct CurrentPhysicalEffectOutcomeFence<'a> {
    _guard: &'a CurrentPhysicalEffectOutcomeGuard,
    proof: &'a VerifiedPhysicalEffectOutcomeEvidence,
    fenced_at_unix_ms: u64,
    evidence_expires_at_unix_ms: u64,
    verifier_key_not_after_unix_ms: u64,
    trust_snapshot_expires_at_unix_ms: u64,
    challenge_expires_at_unix_ms: u64,
    valid_until_unix_ms: u64,
}

impl<'a> CurrentPhysicalEffectOutcomeFence<'a> {
    pub const fn proof(&self) -> &'a VerifiedPhysicalEffectOutcomeEvidence {
        self.proof
    }

    pub const fn fenced_at_unix_ms(&self) -> u64 {
        self.fenced_at_unix_ms
    }

    pub const fn evidence_expires_at_unix_ms(&self) -> u64 {
        self.evidence_expires_at_unix_ms
    }

    pub const fn verifier_key_not_after_unix_ms(&self) -> u64 {
        self.verifier_key_not_after_unix_ms
    }

    pub const fn trust_snapshot_expires_at_unix_ms(&self) -> u64 {
        self.trust_snapshot_expires_at_unix_ms
    }

    pub const fn challenge_expires_at_unix_ms(&self) -> u64 {
        self.challenge_expires_at_unix_ms
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
