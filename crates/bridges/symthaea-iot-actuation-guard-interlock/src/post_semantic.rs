// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Current controller-key trust and guard-owned interlock policy for the corrected
//! post-semantic physical-effect lineage.
//!
//! This module deliberately does not coerce the stronger post-semantic controller
//! statement into the legacy `PhysicalInterlockReportV1` / `VerifiedPhysicalInterlock`
//! path. The controller signature authenticates a different, stronger statement digest
//! that additionally binds the privileged challenge and exact accepted device appraisal.

use std::time::{SystemTime, UNIX_EPOCH};

use symthaea_authority::Digest32;
use symthaea_iot_actuation_guard_post_semantic_controller::{
    DecodedPostSemanticControllerEvidence, PostSemanticControllerChallengeV1,
};
use symthaea_iot_final_gate::FinalActuatorGateError;
use symthaea_iot_interlock_ed25519::Ed25519Rfc8032InterlockVerifier;
use symthaea_iot_interlock_trust::{
    InterlockTrustError, InterlockTrustHead, verify_interlock_key_binding,
};
use thiserror::Error;

use super::GuardInterlockState;

impl GuardInterlockState {
    /// Re-verify one exact post-semantic controller statement under current controller
    /// trust and the guard-owned physical-interlock policy.
    ///
    /// Both the decoded evidence and its exact privileged challenge are consumed by value
    /// so the resulting proof can retain the complete causal lineage for later final/JIT
    /// composition. Relying-party time is read inside the guard.
    pub fn verify_post_semantic_controller(
        &self,
        evidence: DecodedPostSemanticControllerEvidence,
        challenge: PostSemanticControllerChallengeV1,
    ) -> Result<VerifiedPostSemanticPhysicalInterlock, PostSemanticGuardInterlockError> {
        self.verify_post_semantic_controller_at(evidence, challenge, system_unix_ms()?)
    }

    fn verify_post_semantic_controller_at(
        &self,
        evidence: DecodedPostSemanticControllerEvidence,
        challenge: PostSemanticControllerChallengeV1,
        now_unix_ms: u64,
    ) -> Result<VerifiedPostSemanticPhysicalInterlock, PostSemanticGuardInterlockError> {
        self.policy.validate()?;
        let actual_policy_digest = self.policy.digest()?;
        if actual_policy_digest != self.anchored_policy_digest {
            return Err(PostSemanticGuardInterlockError::AnchoredPolicyDigestMismatch);
        }
        if self.trust_registry.head() != self.anchored_trust_head {
            return Err(PostSemanticGuardInterlockError::AnchoredInterlockTrustHeadMismatch);
        }

        challenge
            .validate()
            .map_err(|_| PostSemanticGuardInterlockError::InvalidPostSemanticChallenge)?;
        if now_unix_ms < challenge.issued_at_unix_ms()
            || now_unix_ms >= challenge.expires_at_unix_ms()
        {
            return Err(PostSemanticGuardInterlockError::PostSemanticChallengeNotFresh);
        }

        let report = evidence.report();
        report
            .validate_structure()
            .map_err(|_| PostSemanticGuardInterlockError::InvalidControllerStatement)?;
        let statement = &report.statement;
        let statement_digest = statement
            .digest()
            .map_err(|_| PostSemanticGuardInterlockError::InvalidControllerStatement)?;

        // Re-establish all parser correlations at the trust boundary. The decoded object
        // is opaque, but these checks make the cryptographic verification locally robust
        // against accidental future parser/API weakening.
        let challenge_digest = challenge
            .digest()
            .map_err(|_| PostSemanticGuardInterlockError::InvalidPostSemanticChallenge)?;
        if statement.challenge_digest != challenge_digest {
            return Err(PostSemanticGuardInterlockError::ChallengeBindingMismatch);
        }
        if statement.device_attestation_result_digest
            != challenge.device_attestation_object_digest()
        {
            return Err(PostSemanticGuardInterlockError::DeviceRealityBindingMismatch);
        }
        if statement.device != *challenge.device() {
            return Err(PostSemanticGuardInterlockError::InterlockDeviceMismatch);
        }
        if statement.envelope_digest != challenge.envelope_digest() {
            return Err(PostSemanticGuardInterlockError::EnvelopeCommitmentMismatch);
        }
        if statement.semantic_head != challenge.semantic_head() {
            return Err(PostSemanticGuardInterlockError::SemanticHeadMismatch);
        }
        if statement.transport_trust_head != challenge.transport_trust_head() {
            return Err(PostSemanticGuardInterlockError::TransportTrustHeadMismatch);
        }
        if statement.checked_at_unix_ms < challenge.semantic_persisted_at_unix_ms()
            || statement.checked_at_unix_ms < challenge.issued_at_unix_ms()
        {
            return Err(PostSemanticGuardInterlockError::ReportPredatesPostSemanticChallenge);
        }
        if statement.expires_at_unix_ms > challenge.expires_at_unix_ms() {
            return Err(PostSemanticGuardInterlockError::ReportOutlivesPostSemanticChallenge);
        }
        if now_unix_ms < statement.checked_at_unix_ms
            || now_unix_ms >= statement.expires_at_unix_ms
        {
            return Err(PostSemanticGuardInterlockError::ReportNotFreshAtRelyingParty);
        }

        // Apply the existing guard-owned physical-interlock policy directly to the
        // stronger statement. Do not call legacy `verify_physical_interlock()`: that API
        // would require the raw evidence to authenticate a different legacy digest.
        if statement.device != self.policy.device {
            return Err(PostSemanticGuardInterlockError::InterlockDeviceMismatch);
        }
        if !self.policy.allowed_controllers.contains(&statement.controller_id) {
            return Err(PostSemanticGuardInterlockError::InterlockControllerDenied);
        }
        if statement.asserted_interlocks != self.policy.required_interlocks {
            return Err(PostSemanticGuardInterlockError::InterlockSetMismatch);
        }
        let report_lifetime_ms = statement
            .expires_at_unix_ms
            .checked_sub(statement.checked_at_unix_ms)
            .ok_or(PostSemanticGuardInterlockError::InvalidControllerStatement)?;
        if report_lifetime_ms == 0 || report_lifetime_ms > self.policy.max_report_lifetime_ms {
            return Err(PostSemanticGuardInterlockError::InterlockLifetimeExceedsPolicy);
        }

        // Conservative anti-retroactivity: after a trust generation is issued, evidence
        // observed before that generation cannot become valid merely because a newly
        // introduced key has an older nominal `not_before` timestamp.
        if statement.checked_at_unix_ms < self.trust_registry.snapshot().issued_at_unix_ms {
            return Err(PostSemanticGuardInterlockError::ReportPredatesCurrentTrustGeneration);
        }

        let binding = verify_interlock_key_binding(
            &self.trust_registry,
            &statement.controller_id,
            statement_digest,
            report.evidence_digest,
            evidence.raw_interlock_evidence(),
            statement.checked_at_unix_ms,
            now_unix_ms,
            &Ed25519Rfc8032InterlockVerifier,
        )?;

        if binding.controller_id() != statement.controller_id.as_str()
            || binding.report_digest() != statement_digest
            || binding.evidence_digest() != report.evidence_digest
            || binding.trust_head() != self.anchored_trust_head
        {
            return Err(PostSemanticGuardInterlockError::InternalBindingCompositionMismatch);
        }

        let controller_key_id = binding.key_id().to_owned();
        let controller_key_digest = binding.key_digest();
        let interlock_trust_head = binding.trust_head();
        let response_digest = evidence.response_digest();
        let evidence_digest = report.evidence_digest;

        Ok(VerifiedPostSemanticPhysicalInterlock {
            evidence,
            challenge,
            policy_digest: self.anchored_policy_digest,
            interlock_trust_head,
            controller_key_id,
            controller_key_digest,
            statement_digest,
            evidence_digest,
            response_digest,
            verified_at_unix_ms: now_unix_ms,
        })
    }
}

/// Opaque proof that the exact post-semantic controller statement passed current concrete
/// controller-key trust and the guard-owned physical-interlock policy.
///
/// This type is non-clone and non-serializable. It is still not final actuator authority.
#[derive(Debug)]
pub struct VerifiedPostSemanticPhysicalInterlock {
    evidence: DecodedPostSemanticControllerEvidence,
    challenge: PostSemanticControllerChallengeV1,
    policy_digest: Digest32,
    interlock_trust_head: InterlockTrustHead,
    controller_key_id: String,
    controller_key_digest: Digest32,
    statement_digest: Digest32,
    evidence_digest: Digest32,
    response_digest: Digest32,
    verified_at_unix_ms: u64,
}

impl VerifiedPostSemanticPhysicalInterlock {
    pub fn challenge(&self) -> &PostSemanticControllerChallengeV1 {
        &self.challenge
    }

    pub fn evidence(&self) -> &DecodedPostSemanticControllerEvidence {
        &self.evidence
    }

    pub const fn policy_digest(&self) -> Digest32 {
        self.policy_digest
    }

    pub const fn interlock_trust_head(&self) -> InterlockTrustHead {
        self.interlock_trust_head
    }

    pub fn controller_id(&self) -> &str {
        &self.evidence.report().statement.controller_id
    }

    pub fn controller_key_id(&self) -> &str {
        &self.controller_key_id
    }

    pub const fn controller_key_digest(&self) -> Digest32 {
        self.controller_key_digest
    }

    pub const fn statement_digest(&self) -> Digest32 {
        self.statement_digest
    }

    pub const fn evidence_digest(&self) -> Digest32 {
        self.evidence_digest
    }

    pub const fn response_digest(&self) -> Digest32 {
        self.response_digest
    }

    pub const fn verified_at_unix_ms(&self) -> u64 {
        self.verified_at_unix_ms
    }

    pub fn into_parts(
        self,
    ) -> (
        DecodedPostSemanticControllerEvidence,
        PostSemanticControllerChallengeV1,
    ) {
        (self.evidence, self.challenge)
    }
}

fn system_unix_ms() -> Result<u64, PostSemanticGuardInterlockError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| PostSemanticGuardInterlockError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis())
        .map_err(|_| PostSemanticGuardInterlockError::SystemClockOverflow)
}

#[derive(Debug, Error)]
pub enum PostSemanticGuardInterlockError {
    #[error("guard-owned physical-interlock policy is invalid: {0}")]
    Policy(#[from] FinalActuatorGateError),
    #[error("guard-owned current controller trust or fixed Ed25519 verification failed: {0}")]
    InterlockTrust(#[from] InterlockTrustError),
    #[error("guard interlock policy differs from its independently retained digest")]
    AnchoredPolicyDigestMismatch,
    #[error("guard controller trust differs from its independently retained head")]
    AnchoredInterlockTrustHeadMismatch,
    #[error("post-semantic controller challenge is invalid")]
    InvalidPostSemanticChallenge,
    #[error("post-semantic controller challenge is not fresh at relying-party verification")]
    PostSemanticChallengeNotFresh,
    #[error("post-semantic controller statement/report is invalid")]
    InvalidControllerStatement,
    #[error("controller statement does not bind the exact privileged challenge")]
    ChallengeBindingMismatch,
    #[error("controller statement does not bind the exact authenticated device appraisal")]
    DeviceRealityBindingMismatch,
    #[error("controller statement targets another physical device")]
    InterlockDeviceMismatch,
    #[error("controller statement binds another physical-effect envelope")]
    EnvelopeCommitmentMismatch,
    #[error("controller statement binds another durable semantic head")]
    SemanticHeadMismatch,
    #[error("controller statement binds another Xenia transport-trust generation")]
    TransportTrustHeadMismatch,
    #[error("controller report predates semantic persistence or challenge issuance")]
    ReportPredatesPostSemanticChallenge,
    #[error("controller report outlives the privileged post-semantic challenge")]
    ReportOutlivesPostSemanticChallenge,
    #[error("controller report is stale or future-dated at relying-party verification")]
    ReportNotFreshAtRelyingParty,
    #[error("controller is denied by guard-owned physical-interlock policy")]
    InterlockControllerDenied,
    #[error("controller asserted interlocks differ from the exact guard-owned required set")]
    InterlockSetMismatch,
    #[error("controller report lifetime exceeds the guard-owned policy ceiling")]
    InterlockLifetimeExceedsPolicy,
    #[error("controller report predates the current controller-trust generation")]
    ReportPredatesCurrentTrustGeneration,
    #[error("current controller-key binding disagrees with the exact statement/evidence pair")]
    InternalBindingCompositionMismatch,
    #[error("guard system clock is before Unix epoch")]
    SystemClockBeforeUnixEpoch,
    #[error("guard system wall-clock milliseconds overflow")]
    SystemClockOverflow,
}
