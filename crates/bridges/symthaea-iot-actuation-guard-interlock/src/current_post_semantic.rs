// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Current controller/interlock fencing for one already verified post-semantic proof.
//!
//! Historical post-semantic verification proves that one exact controller statement and raw
//! evidence were trusted at an earlier boundary. This module re-establishes the complete causal
//! and policy binding under independently anchored current controller trust immediately before a
//! later physical attempt. It reuses `verify_interlock_key_binding` for active-key selection and
//! fixed Ed25519 verification rather than duplicating controller-key lifecycle logic.
//!
//! Success remains non-authorizing. The returned fence borrows both the exact current guard and
//! historical proof so a higher-level actuation linearization boundary can keep the checked state
//! pinned for one attempt.

use std::time::{SystemTime, UNIX_EPOCH};

use symthaea_authority::Digest32;
use symthaea_iot_final_gate::{FinalActuatorGateError, PhysicalInterlockPolicyV1};
use symthaea_iot_interlock_ed25519::Ed25519Rfc8032InterlockVerifier;
use symthaea_iot_interlock_trust::{
    InterlockTrustError, InterlockTrustHead, InterlockTrustRegistry, verify_interlock_key_binding,
};
use thiserror::Error;

use crate::VerifiedPostSemanticPhysicalInterlock;

/// Guard-owned current controller trust and physical-interlock policy for one later attempt.
#[derive(Debug)]
pub struct CurrentPostSemanticInterlockGuard {
    policy: PhysicalInterlockPolicyV1,
    anchored_policy_digest: Digest32,
    trust_registry: InterlockTrustRegistry,
    anchored_trust_head: InterlockTrustHead,
}

impl CurrentPostSemanticInterlockGuard {
    /// Bind current policy/trust only when both match independently retained anchors.
    pub fn new(
        policy: PhysicalInterlockPolicyV1,
        anchored_policy_digest: Digest32,
        trust_registry: InterlockTrustRegistry,
        anchored_trust_head: InterlockTrustHead,
    ) -> Result<Self, CurrentPostSemanticInterlockError> {
        policy.validate()?;
        if policy.digest()? != anchored_policy_digest {
            return Err(CurrentPostSemanticInterlockError::AnchoredPolicyDigestMismatch);
        }
        if trust_registry.head() != anchored_trust_head {
            return Err(CurrentPostSemanticInterlockError::AnchoredTrustHeadMismatch);
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

    pub const fn anchored_trust_head(&self) -> InterlockTrustHead {
        self.anchored_trust_head
    }

    /// Re-establish current controller trust using guard-local time and fixed Ed25519.
    ///
    /// No clock, verifier, public key, policy, registry or trust head is caller-selectable.
    pub fn fence_current<'a>(
        &'a self,
        proof: &'a VerifiedPostSemanticPhysicalInterlock,
    ) -> Result<CurrentPostSemanticInterlockFence<'a>, CurrentPostSemanticInterlockError> {
        self.fence_current_at(proof, system_unix_ms()?)
    }

    fn fence_current_at<'a>(
        &'a self,
        proof: &'a VerifiedPostSemanticPhysicalInterlock,
        now_unix_ms: u64,
    ) -> Result<CurrentPostSemanticInterlockFence<'a>, CurrentPostSemanticInterlockError> {
        self.policy.validate()?;
        if self.policy.digest()? != self.anchored_policy_digest {
            return Err(CurrentPostSemanticInterlockError::AnchoredPolicyDigestMismatch);
        }
        if self.trust_registry.head() != self.anchored_trust_head {
            return Err(CurrentPostSemanticInterlockError::AnchoredTrustHeadMismatch);
        }
        if proof.policy_digest() != self.anchored_policy_digest {
            return Err(CurrentPostSemanticInterlockError::ProofPolicyMismatch);
        }
        if proof.interlock_trust_head() != self.anchored_trust_head {
            return Err(CurrentPostSemanticInterlockError::ProofTrustHeadMismatch);
        }
        if now_unix_ms < proof.verified_at_unix_ms() {
            return Err(CurrentPostSemanticInterlockError::ClockRegressed);
        }

        let challenge = proof.challenge();
        challenge
            .validate()
            .map_err(|_| CurrentPostSemanticInterlockError::InvalidChallenge)?;
        if now_unix_ms < challenge.issued_at_unix_ms()
            || now_unix_ms >= challenge.expires_at_unix_ms()
        {
            return Err(CurrentPostSemanticInterlockError::ChallengeNotFresh);
        }

        let evidence = proof.evidence();
        let report = evidence.report();
        report
            .validate_structure()
            .map_err(|_| CurrentPostSemanticInterlockError::InvalidControllerStatement)?;
        let statement = &report.statement;
        let statement_digest = statement
            .digest()
            .map_err(|_| CurrentPostSemanticInterlockError::InvalidControllerStatement)?;
        if statement_digest != proof.statement_digest()
            || report.evidence_digest != proof.evidence_digest()
            || evidence.response_digest() != proof.response_digest()
        {
            return Err(CurrentPostSemanticInterlockError::ProofCommitmentMismatch);
        }

        // Re-establish the full post-semantic causal correlation locally at the current boundary.
        let challenge_digest = challenge
            .digest()
            .map_err(|_| CurrentPostSemanticInterlockError::InvalidChallenge)?;
        if statement.challenge_digest != challenge_digest {
            return Err(CurrentPostSemanticInterlockError::ChallengeBindingMismatch);
        }
        if statement.device_attestation_result_digest
            != challenge.device_attestation_object_digest()
        {
            return Err(CurrentPostSemanticInterlockError::DeviceRealityBindingMismatch);
        }
        if statement.device != *challenge.device() || statement.device != self.policy.device {
            return Err(CurrentPostSemanticInterlockError::DeviceMismatch);
        }
        if statement.envelope_digest != challenge.envelope_digest() {
            return Err(CurrentPostSemanticInterlockError::EnvelopeMismatch);
        }
        if statement.semantic_head != challenge.semantic_head() {
            return Err(CurrentPostSemanticInterlockError::SemanticHeadMismatch);
        }
        if statement.transport_trust_head != challenge.transport_trust_head() {
            return Err(CurrentPostSemanticInterlockError::TransportTrustHeadMismatch);
        }
        if statement.checked_at_unix_ms < challenge.semantic_persisted_at_unix_ms()
            || statement.checked_at_unix_ms < challenge.issued_at_unix_ms()
        {
            return Err(CurrentPostSemanticInterlockError::ReportPredatesChallenge);
        }
        if statement.expires_at_unix_ms > challenge.expires_at_unix_ms() {
            return Err(CurrentPostSemanticInterlockError::ReportOutlivesChallenge);
        }
        if now_unix_ms < statement.checked_at_unix_ms || now_unix_ms >= statement.expires_at_unix_ms
        {
            return Err(CurrentPostSemanticInterlockError::ReportNotFresh);
        }

        if !self
            .policy
            .allowed_controllers
            .contains(&statement.controller_id)
        {
            return Err(CurrentPostSemanticInterlockError::ControllerDenied);
        }
        if statement.asserted_interlocks != self.policy.required_interlocks {
            return Err(CurrentPostSemanticInterlockError::InterlockSetMismatch);
        }
        let report_lifetime_ms = statement
            .expires_at_unix_ms
            .checked_sub(statement.checked_at_unix_ms)
            .ok_or(CurrentPostSemanticInterlockError::InvalidControllerStatement)?;
        if report_lifetime_ms == 0 || report_lifetime_ms > self.policy.max_report_lifetime_ms {
            return Err(CurrentPostSemanticInterlockError::ReportLifetimeExceedsPolicy);
        }
        if statement.checked_at_unix_ms < self.trust_registry.snapshot().issued_at_unix_ms {
            return Err(CurrentPostSemanticInterlockError::ReportPredatesCurrentTrustGeneration);
        }

        // This is the sole active-key selection and current cryptographic verification path.
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
        if binding.controller_id() != proof.controller_id()
            || binding.key_id() != proof.controller_key_id()
            || binding.key_digest() != proof.controller_key_digest()
            || binding.report_digest() != proof.statement_digest()
            || binding.evidence_digest() != proof.evidence_digest()
            || binding.trust_head() != self.anchored_trust_head
        {
            return Err(CurrentPostSemanticInterlockError::CurrentKeyBindingMismatch);
        }

        // Exact lookup only after `verify_interlock_key_binding` selected and verified the current
        // key. This reads its natural expiry; it is not a second active-key selection algorithm.
        let current_key = self
            .trust_registry
            .snapshot()
            .keys
            .iter()
            .find(|key| {
                key.controller_id == binding.controller_id() && key.key_id == binding.key_id()
            })
            .ok_or(CurrentPostSemanticInterlockError::CurrentKeyRecordMissing)?;
        if current_key.digest()? != binding.key_digest() {
            return Err(CurrentPostSemanticInterlockError::CurrentKeyBindingMismatch);
        }

        let controller_report_expires_at_unix_ms = statement.expires_at_unix_ms;
        let controller_key_not_after_unix_ms = current_key.not_after_unix_ms;
        let trust_snapshot_expires_at_unix_ms = self.trust_registry.snapshot().expires_at_unix_ms;
        let valid_until_unix_ms = controller_report_expires_at_unix_ms
            .min(controller_key_not_after_unix_ms)
            .min(trust_snapshot_expires_at_unix_ms);
        if now_unix_ms >= valid_until_unix_ms {
            return Err(CurrentPostSemanticInterlockError::CurrentWindowElapsed);
        }

        Ok(CurrentPostSemanticInterlockFence {
            _guard: self,
            proof,
            fenced_at_unix_ms: now_unix_ms,
            controller_report_expires_at_unix_ms,
            controller_key_not_after_unix_ms,
            trust_snapshot_expires_at_unix_ms,
            valid_until_unix_ms,
        })
    }
}

/// Borrowed proof that one exact post-semantic controller/interlock proof remains current now.
#[derive(Debug)]
pub struct CurrentPostSemanticInterlockFence<'a> {
    _guard: &'a CurrentPostSemanticInterlockGuard,
    proof: &'a VerifiedPostSemanticPhysicalInterlock,
    fenced_at_unix_ms: u64,
    controller_report_expires_at_unix_ms: u64,
    controller_key_not_after_unix_ms: u64,
    trust_snapshot_expires_at_unix_ms: u64,
    valid_until_unix_ms: u64,
}

impl<'a> CurrentPostSemanticInterlockFence<'a> {
    pub const fn proof(&self) -> &'a VerifiedPostSemanticPhysicalInterlock {
        self.proof
    }

    pub const fn fenced_at_unix_ms(&self) -> u64 {
        self.fenced_at_unix_ms
    }

    pub const fn controller_report_expires_at_unix_ms(&self) -> u64 {
        self.controller_report_expires_at_unix_ms
    }

    pub const fn controller_key_not_after_unix_ms(&self) -> u64 {
        self.controller_key_not_after_unix_ms
    }

    pub const fn trust_snapshot_expires_at_unix_ms(&self) -> u64 {
        self.trust_snapshot_expires_at_unix_ms
    }

    /// Earliest exclusive natural expiry across report, exact selected key and trust snapshot.
    pub const fn valid_until_unix_ms(&self) -> u64 {
        self.valid_until_unix_ms
    }
}

fn system_unix_ms() -> Result<u64, CurrentPostSemanticInterlockError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| CurrentPostSemanticInterlockError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| CurrentPostSemanticInterlockError::ClockOverflow)
}

#[derive(Debug, Error)]
pub enum CurrentPostSemanticInterlockError {
    #[error("guard-owned current physical-interlock policy is invalid: {0}")]
    Policy(#[from] FinalActuatorGateError),
    #[error("guard-owned current controller trust or fixed Ed25519 verification failed: {0}")]
    InterlockTrust(#[from] InterlockTrustError),
    #[error("current interlock policy differs from its independently retained digest")]
    AnchoredPolicyDigestMismatch,
    #[error("current controller trust differs from its independently retained head")]
    AnchoredTrustHeadMismatch,
    #[error("historical post-semantic proof was created under another interlock policy")]
    ProofPolicyMismatch,
    #[error("historical post-semantic proof was created under another controller-trust generation")]
    ProofTrustHeadMismatch,
    #[error("system wall clock regressed behind historical interlock verification")]
    ClockRegressed,
    #[error("retained post-semantic challenge is invalid")]
    InvalidChallenge,
    #[error("retained post-semantic challenge is no longer fresh")]
    ChallengeNotFresh,
    #[error("retained post-semantic controller statement/report is invalid")]
    InvalidControllerStatement,
    #[error("retained post-semantic proof commitments no longer reproduce its exact evidence")]
    ProofCommitmentMismatch,
    #[error("controller statement no longer binds the exact privileged challenge")]
    ChallengeBindingMismatch,
    #[error("controller statement no longer binds the exact authenticated device appraisal")]
    DeviceRealityBindingMismatch,
    #[error("controller statement targets another device")]
    DeviceMismatch,
    #[error("controller statement binds another physical-effect envelope")]
    EnvelopeMismatch,
    #[error("controller statement binds another durable semantic head")]
    SemanticHeadMismatch,
    #[error("controller statement binds another Xenia transport-trust generation")]
    TransportTrustHeadMismatch,
    #[error("controller report predates semantic persistence or challenge issuance")]
    ReportPredatesChallenge,
    #[error("controller report outlives the privileged challenge")]
    ReportOutlivesChallenge,
    #[error("controller report is stale or future-dated at current fencing")]
    ReportNotFresh,
    #[error("controller is denied by current guard-owned interlock policy")]
    ControllerDenied,
    #[error("controller asserted interlocks differ from current guard-owned policy")]
    InterlockSetMismatch,
    #[error("controller report lifetime exceeds current guard-owned policy")]
    ReportLifetimeExceedsPolicy,
    #[error("controller report predates the current controller-trust generation")]
    ReportPredatesCurrentTrustGeneration,
    #[error("current controller-key binding differs from the exact historical proof")]
    CurrentKeyBindingMismatch,
    #[error("fixed current verification selected a controller key whose exact record is missing")]
    CurrentKeyRecordMissing,
    #[error("current controller report/key/trust validity window has elapsed")]
    CurrentWindowElapsed,
    #[error("guard system clock is before Unix epoch")]
    SystemClockBeforeUnixEpoch,
    #[error("guard system wall-clock milliseconds overflow")]
    ClockOverflow,
}
