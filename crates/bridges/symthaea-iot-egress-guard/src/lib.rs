// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verified-posture composition immediately before physical egress.
//!
//! `symthaea-iot-actuation` deliberately keeps policy/authority evaluation separate
//! from device attestation. This crate composes those independent proofs without
//! collapsing them into one ambient "trusted" bit.
//!
//! The safe product path is:
//!
//! ```text
//! ArmedActuationPermit
//!   + original registry-issued ActuationPolicyHandle
//!   + fresh VerifiedDevicePosture
//!   + current verifier-trust registry
//!   + current actuation-policy registry
//!   -> existing exact preflight, using ONLY verified posture runtime state
//!   -> PostureBoundEgressPermit
//!   -> authenticated Xenia/device egress
//! ```
//!
//! Shipped transport adapters should consume [`PostureBoundEgressPermit`], not a
//! raw lower-layer ready permit. The lower crates remain independently useful for
//! deterministic policy/accounting tests and non-product composition.

#![deny(unsafe_code)]

use symthaea_action_runtime::GrantAccount;
use symthaea_authority::{
    AuthorityEpoch, CapabilityGrant, Digest32, NegativeAuthorityFact, ResourceRef,
};
use symthaea_iot_actuation::{
    ActuationError, ArmedActuationPermit, PreflightOutcome, PreflightRejection,
    PreflightSequenceAmbiguity, ReadyActuationPermit,
};
use symthaea_iot_authority::DeviceCommand;
use symthaea_iot_durable_runtime::{
    DurableEffectTransition, DurableIoTHead, DurableUnknownPhysicalEffect,
};
use symthaea_iot_policy::{
    ActuationPolicyHandle, ActuationPolicyHead, ActuationPolicyRegistry,
};
use symthaea_iot_posture::{VerifiedDevicePosture, VerifierTrustHead, VerifierTrustRegistry};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostureGuardBlockReason {
    AdmittedPolicyBindingMismatch,
    PostureDeviceMismatch {
        expected: ResourceRef,
        attested: ResourceRef,
    },
    VerifierTrustGenerationChanged {
        attested: VerifierTrustHead,
        current: VerifierTrustHead,
    },
    PostureNotFresh,
    PosturePredatesPolicySelection {
        appraised_at_unix_s: u64,
        policy_selected_at_unix_s: u64,
    },
}

#[derive(Debug)]
pub struct PostureGuardBlocked {
    pub reason: PostureGuardBlockReason,
    pub armed: ArmedActuationPermit,
}

#[derive(Debug)]
pub enum PostureBindingOutcome {
    Ready(Box<PostureBoundEgressPermit>),
    Rejected(Box<PreflightRejection>),
    SequenceAmbiguous(Box<PreflightSequenceAmbiguity>),
    NotAttempted(Box<PostureGuardBlocked>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PostureBinding {
    device: ResourceRef,
    result_digest: Digest32,
    evidence_digest: Digest32,
    reference_values_digest: Digest32,
    appraisal_policy_digest: Digest32,
    challenge_digest: Digest32,
    verifier_id: String,
    trust_head: VerifierTrustHead,
    appraised_at_unix_s: u64,
    expires_at_unix_s: u64,
}

impl PostureBinding {
    fn from_verified(posture: &VerifiedDevicePosture) -> Self {
        Self {
            device: posture.device().clone(),
            result_digest: posture.result_digest(),
            evidence_digest: posture.evidence_digest(),
            reference_values_digest: posture.reference_values_digest(),
            appraisal_policy_digest: posture.appraisal_policy_digest(),
            challenge_digest: posture.challenge_digest(),
            verifier_id: posture.verifier_id().to_string(),
            trust_head: posture.trust_head(),
            appraised_at_unix_s: posture.appraised_at_unix_s(),
            expires_at_unix_s: posture.expires_at_unix_s(),
        }
    }
}

#[derive(Debug)]
pub struct PostureBoundEgressPermit {
    inner: ReadyActuationPermit,
    posture: PostureBinding,
}

impl PostureBoundEgressPermit {
    pub fn command(&self) -> &DeviceCommand {
        self.inner.command()
    }

    pub fn proposal_digest(&self) -> Digest32 {
        self.inner.proposal_digest()
    }

    pub fn policy_digest(&self) -> Digest32 {
        self.inner.policy_digest()
    }

    pub fn policy_registry_head(&self) -> ActuationPolicyHead {
        self.inner.policy_registry_head()
    }

    pub const fn armed_head(&self) -> DurableIoTHead {
        self.inner.armed_head()
    }

    pub const fn validated_at_unix_s(&self) -> u64 {
        self.inner.validated_at_unix_s()
    }

    pub fn posture_device(&self) -> &ResourceRef {
        &self.posture.device
    }

    pub fn posture_result_digest(&self) -> Digest32 {
        self.posture.result_digest
    }

    pub fn posture_evidence_digest(&self) -> Digest32 {
        self.posture.evidence_digest
    }

    pub fn posture_reference_values_digest(&self) -> Digest32 {
        self.posture.reference_values_digest
    }

    pub fn posture_appraisal_policy_digest(&self) -> Digest32 {
        self.posture.appraisal_policy_digest
    }

    pub fn posture_challenge_digest(&self) -> Digest32 {
        self.posture.challenge_digest
    }

    pub fn posture_verifier_id(&self) -> &str {
        &self.posture.verifier_id
    }

    pub const fn posture_trust_head(&self) -> VerifierTrustHead {
        self.posture.trust_head
    }

    pub const fn posture_appraised_at_unix_s(&self) -> u64 {
        self.posture.appraised_at_unix_s
    }

    pub const fn posture_expires_at_unix_s(&self) -> u64 {
        self.posture.expires_at_unix_s
    }

    pub fn into_unknown(self) -> DurableUnknownPhysicalEffect {
        self.inner.into_unknown()
    }

    pub fn observed_applied(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<DurableEffectTransition, ActuationError> {
        self.inner.observed_applied(account, grant)
    }

    pub fn proven_not_dispatched(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<DurableEffectTransition, ActuationError> {
        self.inner.proven_not_dispatched(account, grant)
    }
}

pub fn revalidate_armed_with_verified_posture(
    armed: ArmedActuationPermit,
    grant: &CapabilityGrant,
    account: &mut GrantAccount,
    now_unix_s: u64,
    current_epoch: AuthorityEpoch,
    negative_facts: &[NegativeAuthorityFact],
    posture: &VerifiedDevicePosture,
    current_verifier_trust: &VerifierTrustRegistry,
    admitted_policy: &ActuationPolicyHandle<'_>,
    current_policy_registry: &ActuationPolicyRegistry,
) -> Result<PostureBindingOutcome, ActuationError> {
    if admitted_policy.registry_head() != armed.policy_registry_head()
        || admitted_policy.policy_digest() != armed.policy_digest()
    {
        return Ok(PostureBindingOutcome::NotAttempted(Box::new(
            PostureGuardBlocked {
                reason: PostureGuardBlockReason::AdmittedPolicyBindingMismatch,
                armed,
            },
        )));
    }

    if posture.device() != &admitted_policy.policy().device {
        return Ok(PostureBindingOutcome::NotAttempted(Box::new(
            PostureGuardBlocked {
                reason: PostureGuardBlockReason::PostureDeviceMismatch {
                    expected: admitted_policy.policy().device.clone(),
                    attested: posture.device().clone(),
                },
                armed,
            },
        )));
    }

    if posture.trust_head() != current_verifier_trust.head() {
        return Ok(PostureBindingOutcome::NotAttempted(Box::new(
            PostureGuardBlocked {
                reason: PostureGuardBlockReason::VerifierTrustGenerationChanged {
                    attested: posture.trust_head(),
                    current: current_verifier_trust.head(),
                },
                armed,
            },
        )));
    }

    if !posture.is_fresh_at(now_unix_s) {
        return Ok(PostureBindingOutcome::NotAttempted(Box::new(
            PostureGuardBlocked {
                reason: PostureGuardBlockReason::PostureNotFresh,
                armed,
            },
        )));
    }

    if posture.appraised_at_unix_s() < admitted_policy.selected_at_unix_s() {
        return Ok(PostureBindingOutcome::NotAttempted(Box::new(
            PostureGuardBlocked {
                reason: PostureGuardBlockReason::PosturePredatesPolicySelection {
                    appraised_at_unix_s: posture.appraised_at_unix_s(),
                    policy_selected_at_unix_s: admitted_policy.selected_at_unix_s(),
                },
                armed,
            },
        )));
    }

    match armed.revalidate_before_send(
        grant,
        account,
        now_unix_s,
        current_epoch,
        negative_facts,
        posture.runtime_state(),
        current_policy_registry,
    )? {
        PreflightOutcome::Ready(ready) => {
            debug_assert_eq!(&ready.command().device, posture.device());
            debug_assert_eq!(
                ready.command().expected_firmware,
                posture.runtime_state().running_firmware
            );
            Ok(PostureBindingOutcome::Ready(Box::new(
                PostureBoundEgressPermit {
                    inner: *ready,
                    posture: PostureBinding::from_verified(posture),
                },
            )))
        }
        PreflightOutcome::Rejected(rejected) => Ok(PostureBindingOutcome::Rejected(rejected)),
        PreflightOutcome::SequenceAmbiguous(ambiguous) => {
            Ok(PostureBindingOutcome::SequenceAmbiguous(ambiguous))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posture_guard_reason_is_stable_data() {
        let reason = PostureGuardBlockReason::PostureNotFresh;
        assert_eq!(reason, PostureGuardBlockReason::PostureNotFresh);
    }
}
