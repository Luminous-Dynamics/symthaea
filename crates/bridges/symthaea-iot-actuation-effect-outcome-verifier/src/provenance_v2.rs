// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Policy-bound V2 physical-effect outcome evidence.
//!
//! V1 remains byte- and domain-stable. V2 reuses the exact V1 semantic body as a nested canonical
//! payload, then adds the authoritative outcome-policy generation/digest under distinct V2 policy,
//! body, signature and object domains. The single fixed Ed25519 signature covers the complete V2
//! body, including policy identity.

use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signature, VerifyingKey};
use symthaea_authority::Digest32;
use symthaea_iot_effect_outcome_policy_bound_protocol::{
    ExpectedOutcomePolicyIdentityV2, PolicyBoundEffectReconciliationChallengeV2,
};
use thiserror::Error;

use crate::{
    EFFECT_OUTCOME_ED25519_ALGORITHM, EffectOutcomeClaimV1, EffectOutcomeError,
    EffectOutcomePolicyV1, EffectOutcomeTrustHead, EffectOutcomeTrustRegistry,
    PhysicalEffectOutcomeEvidenceBodyV1,
};

pub const EFFECT_OUTCOME_POLICY_V2_SCHEMA_VERSION: u16 = 2;
pub const EFFECT_OUTCOME_EVIDENCE_V2_SCHEMA_VERSION: u16 = 2;
pub const EFFECT_OUTCOME_EVIDENCE_V2_WIRE_MAGIC: &[u8] =
    b"SYMTHAEA-IOT-EFFECT-OUTCOME-EVIDENCE-V2-POLICY-BOUND\0";

const EFFECT_OUTCOME_POLICY_V2_DOMAIN: &[u8] =
    b"symthaea-iot-effect-outcome-policy-v2-policy-bound\0";
const EVIDENCE_BODY_V2_DOMAIN: &[u8] =
    b"symthaea-iot-effect-outcome-evidence-body-v2-policy-bound\0";
const EVIDENCE_SIGNATURE_V2_DOMAIN: &[u8] =
    b"symthaea-iot-effect-outcome-evidence-signature-v2-policy-bound\0";
const EVIDENCE_OBJECT_V2_DOMAIN: &[u8] =
    b"symthaea-iot-effect-outcome-evidence-object-v2-policy-bound\0";

/// Policy-controlled provenance requirement.
///
/// The terminal reconciliation layer must never invent a downgrade switch. A V1 historical proof
/// is acceptable only when the held authoritative V2 policy is explicitly compatibility-enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectOutcomeEvidenceProvenanceModeV2 {
    FreshGuardReauthorizationAllowed,
    SignedPolicyIdentityRequired,
}

impl EffectOutcomeEvidenceProvenanceModeV2 {
    pub const fn tag(self) -> u8 {
        match self {
            Self::FreshGuardReauthorizationAllowed => 0,
            Self::SignedPolicyIdentityRequired => 1,
        }
    }
}

/// V2 policy identity wrapping the exact V1 semantic policy.
///
/// This preserves V1 policy bytes/digest while making the provenance requirement itself part of
/// the authoritative policy commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectOutcomePolicyV2 {
    pub schema_version: u16,
    pub base: EffectOutcomePolicyV1,
    pub provenance_mode: EffectOutcomeEvidenceProvenanceModeV2,
}

impl EffectOutcomePolicyV2 {
    pub fn strict(base: EffectOutcomePolicyV1) -> Result<Self, PolicyBoundOutcomeError> {
        let policy = Self {
            schema_version: EFFECT_OUTCOME_POLICY_V2_SCHEMA_VERSION,
            base,
            provenance_mode: EffectOutcomeEvidenceProvenanceModeV2::SignedPolicyIdentityRequired,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn compatibility(base: EffectOutcomePolicyV1) -> Result<Self, PolicyBoundOutcomeError> {
        let policy = Self {
            schema_version: EFFECT_OUTCOME_POLICY_V2_SCHEMA_VERSION,
            base,
            provenance_mode:
                EffectOutcomeEvidenceProvenanceModeV2::FreshGuardReauthorizationAllowed,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn validate(&self) -> Result<(), PolicyBoundOutcomeError> {
        if self.schema_version != EFFECT_OUTCOME_POLICY_V2_SCHEMA_VERSION {
            return Err(PolicyBoundOutcomeError::UnsupportedPolicySchema);
        }
        self.base.validate()?;
        Ok(())
    }

    pub const fn generation(&self) -> u64 {
        self.base.generation
    }

    pub fn base_digest(&self) -> Result<Digest32, PolicyBoundOutcomeError> {
        Ok(self.base.digest()?)
    }

    pub const fn allows_v1_fresh_reauthorization(&self) -> bool {
        matches!(
            self.provenance_mode,
            EffectOutcomeEvidenceProvenanceModeV2::FreshGuardReauthorizationAllowed
        )
    }

    pub fn digest(&self) -> Result<Digest32, PolicyBoundOutcomeError> {
        self.validate()?;
        let base_digest = self.base.digest()?;
        let mut h = blake3::Hasher::new();
        h.update(EFFECT_OUTCOME_POLICY_V2_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.base.generation.to_be_bytes());
        h.update(&base_digest.0);
        h.update(&[self.provenance_mode.tag()]);
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    pub fn expected_identity(&self) -> Result<ExpectedOutcomePolicyIdentityV2, PolicyBoundOutcomeError> {
        Ok(ExpectedOutcomePolicyIdentityV2::new(
            self.generation(),
            self.digest()?,
        )?)
    }
}

/// V2 body signed by the trusted device-class outcome verifier.
///
/// `semantic` is the exact canonical V1 semantic body, but its `challenge_digest` must name the V2
/// policy-bound challenge digest. The V2 signature therefore cannot be replayed as a V1 evidence
/// object and policy identity is inside the signed representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalEffectOutcomeEvidenceBodyV2 {
    pub schema_version: u16,
    pub semantic: PhysicalEffectOutcomeEvidenceBodyV1,
    pub outcome_policy_generation: u64,
    pub outcome_policy_digest: Digest32,
}

impl PhysicalEffectOutcomeEvidenceBodyV2 {
    pub fn validate_structure(&self) -> Result<(), PolicyBoundOutcomeError> {
        if self.schema_version != EFFECT_OUTCOME_EVIDENCE_V2_SCHEMA_VERSION {
            return Err(PolicyBoundOutcomeError::UnsupportedEvidenceSchema);
        }
        self.semantic.validate_structure()?;
        if self.outcome_policy_generation == 0 {
            return Err(PolicyBoundOutcomeError::SignedPolicyGenerationZero);
        }
        if self.outcome_policy_digest == Digest32([0; 32]) {
            return Err(PolicyBoundOutcomeError::SignedPolicyDigestZero);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, PolicyBoundOutcomeError> {
        self.validate_structure()?;
        let semantic = self.semantic.canonical_bytes()?;
        let mut out = Vec::with_capacity(
            EFFECT_OUTCOME_EVIDENCE_V2_WIRE_MAGIC.len() + 2 + 8 + semantic.len() + 8 + 32,
        );
        out.extend_from_slice(EFFECT_OUTCOME_EVIDENCE_V2_WIRE_MAGIC);
        out.extend_from_slice(&self.schema_version.to_be_bytes());
        out.extend_from_slice(&(semantic.len() as u64).to_be_bytes());
        out.extend_from_slice(&semantic);
        out.extend_from_slice(&self.outcome_policy_generation.to_be_bytes());
        out.extend_from_slice(&self.outcome_policy_digest.0);
        Ok(out)
    }

    pub fn digest(&self) -> Result<Digest32, PolicyBoundOutcomeError> {
        Ok(domain_hash(EVIDENCE_BODY_V2_DOMAIN, &self.canonical_bytes()?))
    }

    pub fn signature_message(&self) -> Result<Vec<u8>, PolicyBoundOutcomeError> {
        let bytes = self.canonical_bytes()?;
        let mut message = Vec::with_capacity(EVIDENCE_SIGNATURE_V2_DOMAIN.len() + 8 + bytes.len());
        message.extend_from_slice(EVIDENCE_SIGNATURE_V2_DOMAIN);
        message.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
        message.extend_from_slice(&bytes);
        Ok(message)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalEffectOutcomeEvidenceV2 {
    pub body: PhysicalEffectOutcomeEvidenceBodyV2,
    pub signature: [u8; 64],
}

impl PhysicalEffectOutcomeEvidenceV2 {
    pub fn validate_structure(&self) -> Result<(), PolicyBoundOutcomeError> {
        self.body.validate_structure()?;
        if self.signature == [0; 64] {
            return Err(PolicyBoundOutcomeError::InvalidEvidenceSignature);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, PolicyBoundOutcomeError> {
        self.validate_structure()?;
        let body_digest = self.body.digest()?;
        let mut bytes = Vec::with_capacity(96);
        bytes.extend_from_slice(&body_digest.0);
        bytes.extend_from_slice(&self.signature);
        Ok(domain_hash(EVIDENCE_OBJECT_V2_DOMAIN, &bytes))
    }
}

/// Fixed owner-local V2 verifier. No verifier, key, algorithm, policy, trust head or clock is
/// caller-selectable.
#[derive(Debug)]
pub struct GuardPhysicalEffectOutcomeStateV2 {
    policy: EffectOutcomePolicyV2,
    anchored_policy_digest: Digest32,
    trust_registry: EffectOutcomeTrustRegistry,
    anchored_trust_head: EffectOutcomeTrustHead,
}

impl GuardPhysicalEffectOutcomeStateV2 {
    pub fn new(
        policy: EffectOutcomePolicyV2,
        anchored_policy_digest: Digest32,
        trust_registry: EffectOutcomeTrustRegistry,
        anchored_trust_head: EffectOutcomeTrustHead,
    ) -> Result<Self, PolicyBoundOutcomeError> {
        if policy.digest()? != anchored_policy_digest {
            return Err(PolicyBoundOutcomeError::AnchoredPolicyDigestMismatch);
        }
        if trust_registry.head() != anchored_trust_head {
            return Err(PolicyBoundOutcomeError::AnchoredTrustHeadMismatch);
        }
        Ok(Self {
            policy,
            anchored_policy_digest,
            trust_registry,
            anchored_trust_head,
        })
    }

    pub const fn policy(&self) -> &EffectOutcomePolicyV2 {
        &self.policy
    }

    pub const fn anchored_policy_generation(&self) -> u64 {
        self.policy.base.generation
    }

    pub const fn anchored_policy_digest(&self) -> Digest32 {
        self.anchored_policy_digest
    }

    pub const fn anchored_trust_head(&self) -> EffectOutcomeTrustHead {
        self.anchored_trust_head
    }

    pub fn verify_policy_bound_evidence(
        &self,
        evidence: PhysicalEffectOutcomeEvidenceV2,
        challenge: &PolicyBoundEffectReconciliationChallengeV2,
    ) -> Result<VerifiedPhysicalEffectOutcomeEvidenceV2, PolicyBoundOutcomeError> {
        self.verify_policy_bound_evidence_at(evidence, challenge, system_unix_ms()?)
    }

    pub(crate) fn verify_policy_bound_evidence_at(
        &self,
        evidence: PhysicalEffectOutcomeEvidenceV2,
        challenge: &PolicyBoundEffectReconciliationChallengeV2,
        now_unix_ms: u64,
    ) -> Result<VerifiedPhysicalEffectOutcomeEvidenceV2, PolicyBoundOutcomeError> {
        self.policy.validate()?;
        if self.policy.digest()? != self.anchored_policy_digest {
            return Err(PolicyBoundOutcomeError::AnchoredPolicyDigestMismatch);
        }
        if self.trust_registry.head() != self.anchored_trust_head {
            return Err(PolicyBoundOutcomeError::AnchoredTrustHeadMismatch);
        }
        challenge.validate()?;
        if !challenge.is_fresh_at(now_unix_ms) {
            return Err(PolicyBoundOutcomeError::ReconciliationChallengeNotFresh);
        }

        let expected = challenge.expected_policy();
        if expected.generation() != self.policy.generation()
            || expected.digest() != self.anchored_policy_digest
        {
            return Err(PolicyBoundOutcomeError::ChallengePolicyIdentityMismatch);
        }

        evidence.validate_structure()?;
        let body = &evidence.body;
        if body.outcome_policy_generation != self.policy.generation()
            || body.outcome_policy_digest != self.anchored_policy_digest
            || body.outcome_policy_generation != expected.generation()
            || body.outcome_policy_digest != expected.digest()
        {
            return Err(PolicyBoundOutcomeError::SignedPolicyIdentityMismatch);
        }

        let semantic = &body.semantic;
        let base_policy = &self.policy.base;
        if semantic.device != *challenge.device() || semantic.device != base_policy.device {
            return Err(PolicyBoundOutcomeError::EvidenceDeviceMismatch);
        }
        if semantic.operation != *challenge.operation() || semantic.operation != base_policy.operation {
            return Err(PolicyBoundOutcomeError::EvidenceOperationMismatch);
        }
        if semantic.executor != *challenge.executor() {
            return Err(PolicyBoundOutcomeError::EvidenceExecutorMismatch);
        }
        let challenge_digest = challenge.digest()?;
        if semantic.challenge_digest != challenge_digest {
            return Err(PolicyBoundOutcomeError::EvidenceChallengeMismatch);
        }
        if semantic.command_digest != challenge.command_digest() {
            return Err(PolicyBoundOutcomeError::EvidenceCommandMismatch);
        }
        if semantic.sequence != challenge.sequence() {
            return Err(PolicyBoundOutcomeError::EvidenceSequenceMismatch);
        }
        if semantic.outcome_profile_digest != base_policy.exact_outcome_profile_digest {
            return Err(PolicyBoundOutcomeError::EvidenceOutcomeProfileMismatch);
        }
        if !base_policy
            .accepted_reference_values
            .contains(&semantic.reference_values_digest)
        {
            return Err(PolicyBoundOutcomeError::EvidenceReferenceValuesDenied);
        }
        if semantic.appraisal_policy_digest != base_policy.exact_appraisal_policy_digest {
            return Err(PolicyBoundOutcomeError::EvidenceAppraisalPolicyMismatch);
        }
        if !base_policy.allowed_verifier_ids.contains(&semantic.verifier_id) {
            return Err(PolicyBoundOutcomeError::EvidenceVerifierDenied);
        }
        if !base_policy.allowed_claim_kinds.contains(&semantic.claim.kind()) {
            return Err(PolicyBoundOutcomeError::EvidenceClaimKindDenied);
        }
        if semantic.algorithm != EFFECT_OUTCOME_ED25519_ALGORITHM {
            return Err(PolicyBoundOutcomeError::UnsupportedEvidenceAlgorithm);
        }

        if semantic.evidence_issued_at_unix_ms < challenge.issued_at_unix_ms() {
            return Err(PolicyBoundOutcomeError::EvidencePredatesChallenge);
        }
        if semantic.evidence_issued_at_unix_ms < self.trust_registry.snapshot().issued_at_unix_ms {
            return Err(PolicyBoundOutcomeError::EvidencePredatesCurrentTrustGeneration);
        }
        if semantic.evidence_expires_at_unix_ms > challenge.expires_at_unix_ms() {
            return Err(PolicyBoundOutcomeError::EvidenceOutlivesChallenge);
        }
        if now_unix_ms < semantic.evidence_issued_at_unix_ms
            || now_unix_ms >= semantic.evidence_expires_at_unix_ms
        {
            return Err(PolicyBoundOutcomeError::EvidenceNotFresh);
        }
        let lifetime_ms = semantic
            .evidence_expires_at_unix_ms
            .checked_sub(semantic.evidence_issued_at_unix_ms)
            .ok_or(PolicyBoundOutcomeError::InvalidEvidenceWindow)?;
        if lifetime_ms == 0 || lifetime_ms > base_policy.max_evidence_lifetime_ms {
            return Err(PolicyBoundOutcomeError::EvidenceLifetimeExceedsPolicy);
        }

        validate_claim_causality_v2(
            semantic.claim,
            semantic.evidence_issued_at_unix_ms,
            challenge,
        )?;

        let current_key = self.trust_registry.exact_active_key(semantic, now_unix_ms)?;
        if lifetime_ms > current_key.max_evidence_lifetime_ms {
            return Err(PolicyBoundOutcomeError::EvidenceLifetimeExceedsPolicy);
        }
        let message = body.signature_message()?;
        let signature = Signature::try_from(evidence.signature.as_ref())
            .map_err(|_| PolicyBoundOutcomeError::InvalidEvidenceSignature)?;
        let verifying_key = VerifyingKey::from_bytes(&current_key.public_key)
            .map_err(|_| PolicyBoundOutcomeError::InvalidVerifierPublicKey)?;
        verifying_key
            .verify_strict(&message, &signature)
            .map_err(|_| PolicyBoundOutcomeError::InvalidEvidenceSignature)?;

        let evidence_digest = evidence.digest()?;
        let body_digest = body.digest()?;
        let verifier_id = current_key.verifier_id.clone();
        let key_id = current_key.key_id.clone();
        let key_digest = current_key.digest()?;
        let verifier_key_not_after_unix_ms = current_key.not_after_unix_ms;
        let trust_snapshot_expires_at_unix_ms = self.trust_registry.snapshot().expires_at_unix_ms;
        let challenge_expires_at_unix_ms = challenge.expires_at_unix_ms();
        let valid_until_unix_ms = semantic
            .evidence_expires_at_unix_ms
            .min(verifier_key_not_after_unix_ms)
            .min(trust_snapshot_expires_at_unix_ms)
            .min(challenge_expires_at_unix_ms);
        if now_unix_ms >= valid_until_unix_ms {
            return Err(PolicyBoundOutcomeError::EvidenceNotFresh);
        }

        Ok(VerifiedPhysicalEffectOutcomeEvidenceV2 {
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
            policy_generation: self.policy.generation(),
            policy_digest: self.anchored_policy_digest,
            base_policy_digest: self.policy.base_digest()?,
            provenance_mode: self.policy.provenance_mode,
            verified_at_unix_ms: now_unix_ms,
            verifier_key_not_after_unix_ms,
            trust_snapshot_expires_at_unix_ms,
            valid_until_unix_ms,
        })
    }
}

#[derive(Debug)]
pub struct VerifiedPhysicalEffectOutcomeEvidenceV2 {
    evidence: PhysicalEffectOutcomeEvidenceV2,
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
    policy_generation: u64,
    policy_digest: Digest32,
    base_policy_digest: Digest32,
    provenance_mode: EffectOutcomeEvidenceProvenanceModeV2,
    verified_at_unix_ms: u64,
    verifier_key_not_after_unix_ms: u64,
    trust_snapshot_expires_at_unix_ms: u64,
    valid_until_unix_ms: u64,
}

impl VerifiedPhysicalEffectOutcomeEvidenceV2 {
    pub const fn evidence(&self) -> &PhysicalEffectOutcomeEvidenceV2 {
        &self.evidence
    }
    pub const fn evidence_digest(&self) -> Digest32 { self.evidence_digest }
    pub const fn body_digest(&self) -> Digest32 { self.body_digest }
    pub const fn challenge_digest(&self) -> Digest32 { self.challenge_digest }
    pub const fn challenge_journal_generation(&self) -> u64 { self.challenge_journal_generation }
    pub const fn challenge_journal_digest(&self) -> Digest32 { self.challenge_journal_digest }
    pub const fn challenge_expires_at_unix_ms(&self) -> u64 { self.challenge_expires_at_unix_ms }
    pub fn verifier_id(&self) -> &str { &self.verifier_id }
    pub fn key_id(&self) -> &str { &self.key_id }
    pub const fn key_digest(&self) -> Digest32 { self.key_digest }
    pub const fn trust_head(&self) -> EffectOutcomeTrustHead { self.trust_head }
    pub const fn policy_generation(&self) -> u64 { self.policy_generation }
    pub const fn policy_digest(&self) -> Digest32 { self.policy_digest }
    pub const fn base_policy_digest(&self) -> Digest32 { self.base_policy_digest }
    pub const fn provenance_mode(&self) -> EffectOutcomeEvidenceProvenanceModeV2 { self.provenance_mode }
    pub const fn verified_at_unix_ms(&self) -> u64 { self.verified_at_unix_ms }
    pub const fn verifier_key_not_after_unix_ms(&self) -> u64 { self.verifier_key_not_after_unix_ms }
    pub const fn trust_snapshot_expires_at_unix_ms(&self) -> u64 { self.trust_snapshot_expires_at_unix_ms }
    pub const fn valid_until_unix_ms(&self) -> u64 { self.valid_until_unix_ms }
}

/// Current V2 guard used immediately before terminal reconciliation. This remains non-authorizing.
#[derive(Debug)]
pub struct CurrentPhysicalEffectOutcomeGuardV2 {
    policy: EffectOutcomePolicyV2,
    anchored_policy_digest: Digest32,
    trust_registry: EffectOutcomeTrustRegistry,
    anchored_trust_head: EffectOutcomeTrustHead,
}

impl CurrentPhysicalEffectOutcomeGuardV2 {
    pub fn new(
        policy: EffectOutcomePolicyV2,
        anchored_policy_digest: Digest32,
        trust_registry: EffectOutcomeTrustRegistry,
        anchored_trust_head: EffectOutcomeTrustHead,
    ) -> Result<Self, PolicyBoundOutcomeError> {
        if policy.digest()? != anchored_policy_digest {
            return Err(PolicyBoundOutcomeError::AnchoredPolicyDigestMismatch);
        }
        if trust_registry.head() != anchored_trust_head {
            return Err(PolicyBoundOutcomeError::AnchoredTrustHeadMismatch);
        }
        Ok(Self { policy, anchored_policy_digest, trust_registry, anchored_trust_head })
    }

    pub const fn policy(&self) -> &EffectOutcomePolicyV2 { &self.policy }
    pub const fn anchored_policy_generation(&self) -> u64 { self.policy.base.generation }
    pub const fn anchored_policy_digest(&self) -> Digest32 { self.anchored_policy_digest }
    pub const fn anchored_trust_head(&self) -> EffectOutcomeTrustHead { self.anchored_trust_head }

    pub fn fence_current<'a>(
        &'a self,
        proof: &'a VerifiedPhysicalEffectOutcomeEvidenceV2,
    ) -> Result<CurrentPhysicalEffectOutcomeFenceV2<'a>, PolicyBoundOutcomeError> {
        self.fence_current_at(proof, system_unix_ms()?)
    }

    pub(crate) fn fence_current_at<'a>(
        &'a self,
        proof: &'a VerifiedPhysicalEffectOutcomeEvidenceV2,
        now_unix_ms: u64,
    ) -> Result<CurrentPhysicalEffectOutcomeFenceV2<'a>, PolicyBoundOutcomeError> {
        self.policy.validate()?;
        if proof.policy_generation() != self.policy.generation() {
            return Err(PolicyBoundOutcomeError::CurrentProofPolicyGenerationMismatch);
        }
        if self.policy.digest()? != self.anchored_policy_digest
            || proof.policy_digest() != self.anchored_policy_digest
        {
            return Err(PolicyBoundOutcomeError::CurrentProofPolicyMismatch);
        }
        if self.trust_registry.head() != self.anchored_trust_head
            || proof.trust_head() != self.anchored_trust_head
        {
            return Err(PolicyBoundOutcomeError::CurrentProofTrustHeadMismatch);
        }
        if now_unix_ms < proof.verified_at_unix_ms() {
            return Err(PolicyBoundOutcomeError::CurrentProofClockRegressed);
        }

        let evidence = proof.evidence();
        evidence.validate_structure()?;
        let body = &evidence.body;
        let semantic = &body.semantic;
        if body.outcome_policy_generation != self.policy.generation()
            || body.outcome_policy_digest != self.anchored_policy_digest
            || semantic.device != self.policy.base.device
            || semantic.operation != self.policy.base.operation
            || semantic.outcome_profile_digest != self.policy.base.exact_outcome_profile_digest
            || semantic.appraisal_policy_digest != self.policy.base.exact_appraisal_policy_digest
            || !self.policy.base.accepted_reference_values.contains(&semantic.reference_values_digest)
            || !self.policy.base.allowed_verifier_ids.contains(&semantic.verifier_id)
            || !self.policy.base.allowed_claim_kinds.contains(&semantic.claim.kind())
        {
            return Err(PolicyBoundOutcomeError::CurrentProofPolicyMismatch);
        }
        if semantic.challenge_digest != proof.challenge_digest()
            || body.digest()? != proof.body_digest()
            || evidence.digest()? != proof.evidence_digest()
        {
            return Err(PolicyBoundOutcomeError::CurrentProofCommitmentMismatch);
        }

        let current_key = self.trust_registry.exact_active_key(semantic, now_unix_ms)?;
        if current_key.verifier_id != proof.verifier_id()
            || current_key.key_id != proof.key_id()
            || current_key.digest()? != proof.key_digest()
        {
            return Err(PolicyBoundOutcomeError::CurrentProofVerifierKeyMismatch);
        }
        let message = body.signature_message()?;
        let signature = Signature::try_from(evidence.signature.as_ref())
            .map_err(|_| PolicyBoundOutcomeError::InvalidEvidenceSignature)?;
        let verifying_key = VerifyingKey::from_bytes(&current_key.public_key)
            .map_err(|_| PolicyBoundOutcomeError::InvalidVerifierPublicKey)?;
        verifying_key
            .verify_strict(&message, &signature)
            .map_err(|_| PolicyBoundOutcomeError::InvalidEvidenceSignature)?;

        let verifier_key_not_after_unix_ms = current_key.not_after_unix_ms;
        let trust_snapshot_expires_at_unix_ms = self.trust_registry.snapshot().expires_at_unix_ms;
        let valid_until_unix_ms = semantic
            .evidence_expires_at_unix_ms
            .min(verifier_key_not_after_unix_ms)
            .min(trust_snapshot_expires_at_unix_ms)
            .min(proof.challenge_expires_at_unix_ms());
        if now_unix_ms >= valid_until_unix_ms {
            return Err(PolicyBoundOutcomeError::CurrentProofWindowElapsed);
        }

        Ok(CurrentPhysicalEffectOutcomeFenceV2 {
            _guard: self,
            proof,
            fenced_at_unix_ms: now_unix_ms,
            evidence_expires_at_unix_ms: semantic.evidence_expires_at_unix_ms,
            verifier_key_not_after_unix_ms,
            trust_snapshot_expires_at_unix_ms,
            challenge_expires_at_unix_ms: proof.challenge_expires_at_unix_ms(),
            valid_until_unix_ms,
        })
    }
}

#[derive(Debug)]
pub struct CurrentPhysicalEffectOutcomeFenceV2<'a> {
    _guard: &'a CurrentPhysicalEffectOutcomeGuardV2,
    proof: &'a VerifiedPhysicalEffectOutcomeEvidenceV2,
    fenced_at_unix_ms: u64,
    evidence_expires_at_unix_ms: u64,
    verifier_key_not_after_unix_ms: u64,
    trust_snapshot_expires_at_unix_ms: u64,
    challenge_expires_at_unix_ms: u64,
    valid_until_unix_ms: u64,
}

impl<'a> CurrentPhysicalEffectOutcomeFenceV2<'a> {
    pub const fn proof(&self) -> &'a VerifiedPhysicalEffectOutcomeEvidenceV2 { self.proof }
    pub const fn fenced_at_unix_ms(&self) -> u64 { self.fenced_at_unix_ms }
    pub const fn evidence_expires_at_unix_ms(&self) -> u64 { self.evidence_expires_at_unix_ms }
    pub const fn verifier_key_not_after_unix_ms(&self) -> u64 { self.verifier_key_not_after_unix_ms }
    pub const fn trust_snapshot_expires_at_unix_ms(&self) -> u64 { self.trust_snapshot_expires_at_unix_ms }
    pub const fn challenge_expires_at_unix_ms(&self) -> u64 { self.challenge_expires_at_unix_ms }
    pub const fn valid_until_unix_ms(&self) -> u64 { self.valid_until_unix_ms }
}

fn validate_claim_causality_v2(
    claim: EffectOutcomeClaimV1,
    evidence_issued_at_unix_ms: u64,
    challenge: &PolicyBoundEffectReconciliationChallengeV2,
) -> Result<(), PolicyBoundOutcomeError> {
    match claim {
        EffectOutcomeClaimV1::ExecutionAndPostcondition {
            effect_recorded_at_unix_ms,
            postcondition_observed_at_unix_ms,
            ..
        } => {
            if effect_recorded_at_unix_ms < challenge.attempt_common_fenced_at_unix_ms()
                || effect_recorded_at_unix_ms >= challenge.attempt_wall_valid_until_unix_ms()
            {
                return Err(PolicyBoundOutcomeError::ExecutionRecordOutsideActuationWindow);
            }
            if postcondition_observed_at_unix_ms < challenge.issued_at_unix_ms() {
                return Err(PolicyBoundOutcomeError::PostconditionObservationNotFresh);
            }
            if postcondition_observed_at_unix_ms < effect_recorded_at_unix_ms
                || postcondition_observed_at_unix_ms > evidence_issued_at_unix_ms
            {
                return Err(PolicyBoundOutcomeError::PostconditionObservationTimeInvalid);
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
                return Err(PolicyBoundOutcomeError::NonExecutionCoverageIncomplete);
            }
            if coverage_through_unix_ms > evidence_issued_at_unix_ms {
                return Err(PolicyBoundOutcomeError::NonExecutionCoverageAfterEvidence);
            }
        }
    }
    Ok(())
}

fn system_unix_ms() -> Result<u64, PolicyBoundOutcomeError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| PolicyBoundOutcomeError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| PolicyBoundOutcomeError::SystemClockOverflow)
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> Digest32 {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    h.update(&(bytes.len() as u64).to_be_bytes());
    h.update(bytes);
    Digest32(*h.finalize().as_bytes())
}

#[derive(Debug, Error)]
pub enum PolicyBoundOutcomeError {
    #[error("legacy outcome verifier validation failed: {0}")]
    Base(#[from] EffectOutcomeError),
    #[error("policy-bound challenge validation failed: {0}")]
    Challenge(#[from] symthaea_iot_effect_outcome_policy_bound_protocol::PolicyBoundChallengeV2Error),
    #[error("unsupported V2 outcome policy schema")]
    UnsupportedPolicySchema,
    #[error("unsupported V2 outcome evidence schema")]
    UnsupportedEvidenceSchema,
    #[error("signed outcome-policy generation is zero")]
    SignedPolicyGenerationZero,
    #[error("signed outcome-policy digest is zero")]
    SignedPolicyDigestZero,
    #[error("V2 outcome policy does not match its independent anchor")]
    AnchoredPolicyDigestMismatch,
    #[error("V2 outcome verifier trust does not match its independent anchor")]
    AnchoredTrustHeadMismatch,
    #[error("policy-bound challenge advertises another outcome policy identity")]
    ChallengePolicyIdentityMismatch,
    #[error("signed V2 evidence names another outcome policy identity")]
    SignedPolicyIdentityMismatch,
    #[error("policy-bound reconciliation challenge is not fresh")]
    ReconciliationChallengeNotFresh,
    #[error("V2 evidence targets another device")]
    EvidenceDeviceMismatch,
    #[error("V2 evidence targets another operation")]
    EvidenceOperationMismatch,
    #[error("V2 evidence targets another executor")]
    EvidenceExecutorMismatch,
    #[error("V2 evidence is bound to another reconciliation challenge")]
    EvidenceChallengeMismatch,
    #[error("V2 evidence is bound to another command")]
    EvidenceCommandMismatch,
    #[error("V2 evidence sequence differs from the challenge")]
    EvidenceSequenceMismatch,
    #[error("V2 evidence outcome profile differs from policy")]
    EvidenceOutcomeProfileMismatch,
    #[error("V2 evidence reference values are denied")]
    EvidenceReferenceValuesDenied,
    #[error("V2 evidence appraisal policy differs from policy")]
    EvidenceAppraisalPolicyMismatch,
    #[error("V2 evidence verifier is denied")]
    EvidenceVerifierDenied,
    #[error("V2 evidence claim kind is denied")]
    EvidenceClaimKindDenied,
    #[error("V2 evidence uses an unsupported algorithm")]
    UnsupportedEvidenceAlgorithm,
    #[error("V2 evidence predates the challenge")]
    EvidencePredatesChallenge,
    #[error("V2 evidence predates current verifier trust")]
    EvidencePredatesCurrentTrustGeneration,
    #[error("V2 evidence outlives the challenge")]
    EvidenceOutlivesChallenge,
    #[error("V2 evidence is not fresh")]
    EvidenceNotFresh,
    #[error("V2 evidence time window is invalid")]
    InvalidEvidenceWindow,
    #[error("V2 evidence lifetime exceeds policy")]
    EvidenceLifetimeExceedsPolicy,
    #[error("V2 execution record is outside the original actuation window")]
    ExecutionRecordOutsideActuationWindow,
    #[error("V2 postcondition observation is not fresh for the challenge")]
    PostconditionObservationNotFresh,
    #[error("V2 postcondition observation time is causally inconsistent")]
    PostconditionObservationTimeInvalid,
    #[error("V2 non-execution proof does not cover the full actuation window")]
    NonExecutionCoverageIncomplete,
    #[error("V2 non-execution coverage extends beyond evidence issuance")]
    NonExecutionCoverageAfterEvidence,
    #[error("V2 evidence signature is invalid")]
    InvalidEvidenceSignature,
    #[error("V2 verifier public key is invalid")]
    InvalidVerifierPublicKey,
    #[error("V2 current proof policy generation differs from guard policy")]
    CurrentProofPolicyGenerationMismatch,
    #[error("V2 current proof policy differs from guard policy")]
    CurrentProofPolicyMismatch,
    #[error("V2 current proof trust head differs from guard trust")]
    CurrentProofTrustHeadMismatch,
    #[error("V2 current proof clock regressed")]
    CurrentProofClockRegressed,
    #[error("V2 current proof commitments changed")]
    CurrentProofCommitmentMismatch,
    #[error("V2 current proof verifier/key identity changed")]
    CurrentProofVerifierKeyMismatch,
    #[error("V2 current proof validity window elapsed")]
    CurrentProofWindowElapsed,
    #[error("system wall clock is before Unix epoch")]
    SystemClockBeforeUnixEpoch,
    #[error("system wall clock does not fit in u64 milliseconds")]
    SystemClockOverflow,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use symthaea_authority::{Digest32, Operation, ResourceRef};

    use super::*;
    use crate::{
        EFFECT_OUTCOME_POLICY_SCHEMA_VERSION, EffectOutcomeClaimKindV1,
        MAX_EFFECT_OUTCOME_EVIDENCE_LIFETIME_MS,
    };

    fn base_policy(generation: u64) -> EffectOutcomePolicyV1 {
        EffectOutcomePolicyV1 {
            schema_version: EFFECT_OUTCOME_POLICY_SCHEMA_VERSION,
            generation,
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("qualification.effect".into()),
            allowed_verifier_ids: BTreeSet::from(["verifier:outcome-a".into()]),
            allowed_claim_kinds: BTreeSet::from([
                EffectOutcomeClaimKindV1::ExecutionAndPostcondition,
                EffectOutcomeClaimKindV1::NonExecution,
            ]),
            accepted_reference_values: BTreeSet::from([Digest32([0x44; 32])]),
            exact_outcome_profile_digest: Digest32([0x55; 32]),
            exact_appraisal_policy_digest: Digest32([0x66; 32]),
            max_evidence_lifetime_ms: MAX_EFFECT_OUTCOME_EVIDENCE_LIFETIME_MS,
        }
    }

    #[test]
    fn provenance_mode_is_part_of_v2_policy_identity() {
        let strict = EffectOutcomePolicyV2::strict(base_policy(7)).unwrap();
        let compatibility = EffectOutcomePolicyV2::compatibility(base_policy(7)).unwrap();
        assert_ne!(strict.digest().unwrap(), compatibility.digest().unwrap());
        assert!(!strict.allows_v1_fresh_reauthorization());
        assert!(compatibility.allows_v1_fresh_reauthorization());
    }

    #[test]
    fn generation_changes_v2_policy_identity_even_when_semantics_match() {
        let a = EffectOutcomePolicyV2::strict(base_policy(1)).unwrap();
        let b = EffectOutcomePolicyV2::strict(base_policy(3)).unwrap();
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }
}
