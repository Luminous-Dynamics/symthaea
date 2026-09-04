// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Guard-owned physical-interlock policy and current controller-key verification.
//!
//! This stage consumes [`VerifiedGuardIngress`], which already proves that Xenia
//! authenticated the exact physical-effect bytes and that the portable interlock report
//! names the same envelope/device/transport-trust generation. It adds only guard-local
//! facts that must never come from the IPC caller:
//!
//! - an exact physical-interlock policy bound to an independently retained digest;
//! - a current anti-rollback controller-key registry bound to an independently retained
//!   [`InterlockTrustHead`]; and
//! - the fixed RFC 8032 Ed25519 controller-evidence verifier.
//!
//! Success remains **non-authorizing**. Device semantic acceptance, final permit
//! composition, complete JIT revocation fencing and HAL/device I/O are deliberately
//! outside this crate.

#![deny(unsafe_code)]

use std::time::{SystemTime, UNIX_EPOCH};

use symthaea_authority::Digest32;
use symthaea_iot_actuation_guard::VerifiedGuardIngress;
use symthaea_iot_actuation_guard_protocol::DecodedGuardEvidence;
use symthaea_iot_final_gate::{
    FinalActuatorGateError, HardwareInterlockEvidenceVerifier, PhysicalInterlockPolicyV1,
    VerifiedPhysicalInterlock, verify_physical_interlock,
};
use symthaea_iot_interlock_ed25519::Ed25519Rfc8032InterlockVerifier;
use symthaea_iot_interlock_trust::{
    InterlockTrustError, InterlockTrustHead, InterlockTrustRegistry,
    VerifiedInterlockKeyBinding, verify_interlock_key_binding,
};
use symthaea_iot_transport_receipt::VerifiedTransportEnvelope;
use thiserror::Error;

/// Guard-owned controller/policy state for one fixed deployment boundary.
#[derive(Debug)]
pub struct GuardInterlockState {
    policy: PhysicalInterlockPolicyV1,
    anchored_policy_digest: Digest32,
    trust_registry: InterlockTrustRegistry,
    anchored_trust_head: InterlockTrustHead,
}

impl GuardInterlockState {
    /// Construct only when both locally loaded objects match independently retained
    /// anchors. Neither anchor is accepted over actuation IPC.
    pub fn new(
        policy: PhysicalInterlockPolicyV1,
        anchored_policy_digest: Digest32,
        trust_registry: InterlockTrustRegistry,
        anchored_trust_head: InterlockTrustHead,
    ) -> Result<Self, GuardInterlockError> {
        let actual_policy_digest = policy.digest()?;
        if actual_policy_digest != anchored_policy_digest {
            return Err(GuardInterlockError::AnchoredPolicyDigestMismatch);
        }
        if trust_registry.head() != anchored_trust_head {
            return Err(GuardInterlockError::AnchoredInterlockTrustHeadMismatch);
        }
        Ok(Self {
            policy,
            anchored_policy_digest,
            trust_registry,
            anchored_trust_head,
        })
    }

    /// Exact guard-owned physical-interlock policy commitment.
    pub const fn anchored_policy_digest(&self) -> Digest32 {
        self.anchored_policy_digest
    }

    /// Exact current anti-rollback controller-trust generation.
    pub const fn anchored_trust_head(&self) -> InterlockTrustHead {
        self.anchored_trust_head
    }

    /// Consume one already-verified ingress object and add current controller trust and
    /// exact guard-owned physical-interlock policy. The local wall clock is read inside
    /// the guard; callers cannot supply relying-party time.
    pub fn verify_ingress(
        &self,
        ingress: VerifiedGuardIngress,
    ) -> Result<VerifiedGuardInterlockEvidence, GuardInterlockError> {
        self.verify_ingress_at(ingress, system_unix_ms()?)
    }

    fn verify_ingress_at(
        &self,
        ingress: VerifiedGuardIngress,
        now_unix_ms: u64,
    ) -> Result<VerifiedGuardInterlockEvidence, GuardInterlockError> {
        let actual_policy_digest = self.policy.digest()?;
        if actual_policy_digest != self.anchored_policy_digest {
            return Err(GuardInterlockError::AnchoredPolicyDigestMismatch);
        }
        if self.trust_registry.head() != self.anchored_trust_head {
            return Err(GuardInterlockError::AnchoredInterlockTrustHeadMismatch);
        }
        if now_unix_ms < ingress.verified_at_unix_ms() {
            return Err(GuardInterlockError::ClockRegressedSinceIngress);
        }

        let (decoded, transport, ingress_report_digest) = ingress.into_parts();
        let report = decoded.interlock_report().clone();
        let report_digest = report.digest()?;
        if report_digest != ingress_report_digest {
            return Err(GuardInterlockError::IngressReportDigestMismatch);
        }

        // Conservative generation fencing: after any controller-trust generation is
        // issued, reports observed before that generation are not accepted under it.
        // This prevents a newly introduced key with an old not_before timestamp from
        // retroactively authenticating previously untrusted hardware evidence.
        if report.checked_at_unix_ms < self.trust_registry.snapshot().issued_at_unix_ms {
            return Err(GuardInterlockError::ReportPredatesCurrentTrustGeneration);
        }
        if now_unix_ms < report.checked_at_unix_ms || now_unix_ms >= report.expires_at_unix_ms {
            return Err(GuardInterlockError::ReportNotFreshBeforeControllerVerification);
        }

        let binding = verify_interlock_key_binding(
            &self.trust_registry,
            &report.controller_id,
            report_digest,
            report.evidence_digest,
            decoded.raw_interlock_evidence(),
            report.checked_at_unix_ms,
            now_unix_ms,
            &Ed25519Rfc8032InterlockVerifier,
        )?;

        let already_verified = CurrentBindingEvidence {
            binding: &binding,
            raw_evidence: decoded.raw_interlock_evidence(),
        };
        let physical_interlock = verify_physical_interlock(
            &self.policy,
            report,
            decoded.raw_interlock_evidence(),
            now_unix_ms,
            &already_verified,
        )?;

        if physical_interlock.report_digest() != binding.report_digest()
            || physical_interlock.evidence_digest() != binding.evidence_digest()
            || physical_interlock.controller_id() != binding.controller_id()
        {
            return Err(GuardInterlockError::InternalBindingCompositionMismatch);
        }

        Ok(VerifiedGuardInterlockEvidence {
            decoded,
            transport,
            physical_interlock,
            policy_digest: self.anchored_policy_digest,
            interlock_trust_head: binding.trust_head(),
            controller_key_id: binding.key_id().to_owned(),
            controller_key_digest: binding.key_digest(),
            verified_at_unix_ms: now_unix_ms,
        })
    }
}

/// Local adapter proving that `verify_physical_interlock` is consuming the exact raw
/// evidence/report pair already verified under current controller-key trust.
///
/// It performs no cryptography itself. The fixed Ed25519 verification occurred in
/// `verify_interlock_key_binding` immediately before this adapter is constructed.
struct CurrentBindingEvidence<'a> {
    binding: &'a VerifiedInterlockKeyBinding,
    raw_evidence: &'a [u8],
}

impl HardwareInterlockEvidenceVerifier for CurrentBindingEvidence<'_> {
    fn verify_interlock_evidence(
        &self,
        controller_id: &str,
        report_digest: Digest32,
        raw_evidence: &[u8],
    ) -> bool {
        controller_id == self.binding.controller_id()
            && report_digest == self.binding.report_digest()
            && raw_evidence == self.raw_evidence
    }
}

/// Opaque proof that guard ingress evidence also passed current controller-key trust and
/// the guard-owned physical-interlock policy.
///
/// This type is non-clone and non-serializable and is **not** actuator authority.
#[derive(Debug)]
pub struct VerifiedGuardInterlockEvidence {
    decoded: DecodedGuardEvidence,
    transport: VerifiedTransportEnvelope,
    physical_interlock: VerifiedPhysicalInterlock,
    policy_digest: Digest32,
    interlock_trust_head: InterlockTrustHead,
    controller_key_id: String,
    controller_key_digest: Digest32,
    verified_at_unix_ms: u64,
}

impl VerifiedGuardInterlockEvidence {
    /// Audit-only commitment to the exact portable request accepted by the ingress gate.
    pub const fn request_digest(&self) -> Digest32 {
        self.decoded.request_digest()
    }

    /// Exact guard-owned physical-interlock policy commitment.
    pub const fn policy_digest(&self) -> Digest32 {
        self.policy_digest
    }

    /// Current anti-rollback controller-trust generation used for verification.
    pub const fn interlock_trust_head(&self) -> InterlockTrustHead {
        self.interlock_trust_head
    }

    /// Exact current controller key identity used for the report signature.
    pub fn controller_key_id(&self) -> &str {
        &self.controller_key_id
    }

    /// Commitment to the exact current controller key record.
    pub const fn controller_key_digest(&self) -> Digest32 {
        self.controller_key_digest
    }

    /// Corrected controller-authenticated report-content digest.
    pub const fn report_digest(&self) -> Digest32 {
        self.physical_interlock.report_digest()
    }

    /// Independent commitment to the exact controller evidence/signature bytes.
    pub const fn evidence_digest(&self) -> Digest32 {
        self.physical_interlock.evidence_digest()
    }

    /// Guard-local relying-party time for this current-trust verification.
    pub const fn verified_at_unix_ms(&self) -> u64 {
        self.verified_at_unix_ms
    }

    /// Consume this non-authorizing stage for later device-semantic/final-gate work.
    /// Exact portable receipt/evidence bytes remain available in `DecodedGuardEvidence`
    /// for the final JIT re-verification stage.
    pub fn into_parts(
        self,
    ) -> (
        DecodedGuardEvidence,
        VerifiedTransportEnvelope,
        VerifiedPhysicalInterlock,
    ) {
        (self.decoded, self.transport, self.physical_interlock)
    }
}

fn system_unix_ms() -> Result<u64, GuardInterlockError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| GuardInterlockError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| GuardInterlockError::SystemClockOverflow)
}

/// Fail-closed guard-local controller-trust/policy errors.
#[derive(Debug, Error)]
pub enum GuardInterlockError {
    /// Local interlock policy failed its own structural/digest contract or final policy
    /// verification rejected the report.
    #[error("guard-owned physical interlock policy/report verification failed: {0}")]
    FinalGate(#[from] FinalActuatorGateError),
    /// Current controller-key trust or fixed signature verification failed.
    #[error("guard-owned interlock controller trust verification failed: {0}")]
    InterlockTrust(#[from] InterlockTrustError),
    /// Loaded local policy differs from its independently retained commitment.
    #[error("guard interlock policy does not match independently anchored digest")]
    AnchoredPolicyDigestMismatch,
    /// Loaded controller-trust registry differs from its independently retained head.
    #[error("guard interlock trust registry does not match independently anchored head")]
    AnchoredInterlockTrustHeadMismatch,
    /// Guard-local wall clock moved backwards after ingress verification.
    #[error("guard clock regressed after Xenia ingress verification")]
    ClockRegressedSinceIngress,
    /// Ingress-carried report object does not reproduce the digest previously checked by
    /// the guard ingress stage.
    #[error("guard interlock report digest changed between ingress and controller verification")]
    IngressReportDigestMismatch,
    /// Current controller trust generation was issued after the hardware observation.
    #[error("physical interlock report predates the current controller-trust generation")]
    ReportPredatesCurrentTrustGeneration,
    /// Report is stale/future-dated before expensive controller verification begins.
    #[error("physical interlock report is not fresh before controller verification")]
    ReportNotFreshBeforeControllerVerification,
    /// Two local proof objects unexpectedly disagree after consuming the same exact pair.
    #[error("internal controller-trust and physical-interlock bindings disagree")]
    InternalBindingCompositionMismatch,
    /// System wall clock is before Unix epoch.
    #[error("guard system clock is before Unix epoch")]
    SystemClockBeforeUnixEpoch,
    /// System wall-clock milliseconds do not fit the protocol time domain.
    #[error("guard system clock overflow")]
    SystemClockOverflow,
}
