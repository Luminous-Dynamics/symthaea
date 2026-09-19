// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDATTENTIONPOLICY-526: require current policy admission of the exact
//! Android attention requirement.
//!
//! QUAL-ANDROIDATTENTION-524 proves that an observation satisfies a supplied
//! requirement whose policy context/generation/root are current. That alone does
//! not prove that the current policy actually selected that exact requirement.
//! This tranche closes that gap by binding current policy to one exact
//! `AndroidAttentionRequirementId` and recomputing 524 under that admission.

use core::fmt;

use crate::assurance_android_attention::{
    AndroidAttentionCertificateId, AndroidAttentionError, AndroidAttentionObservation,
    AndroidAttentionObservationId, AndroidAttentionPolicyContextId,
    AndroidAttentionPolicyStateRoot, AndroidAttentionRequirement,
    AndroidAttentionRequirementId, CurrentAndroidAttentionPolicy,
};

const ADMISSION_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-attention-policy-admission\0";
const CERTIFICATE_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-attention-policy-certificate\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub [u8; 32]);

        impl $name {
            pub const ZERO: Self = Self([0; 32]);
            pub const fn as_bytes(&self) -> &[u8; 32] { &self.0 }
            pub fn is_zero(&self) -> bool { self.0 == [0; 32] }
        }
    };
}

digest_id!(AndroidAttentionPolicyAdmissionId);
digest_id!(AndroidAttentionPolicyCertificateId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidAttentionPolicyAdmission {
    pub policy_context_id: AndroidAttentionPolicyContextId,
    pub policy_generation: u64,
    pub policy_state_root: AndroidAttentionPolicyStateRoot,
    pub authorized_requirement_id: AndroidAttentionRequirementId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidAttentionPolicyCertificate {
    pub admission_id: AndroidAttentionPolicyAdmissionId,
    pub requirement_id: AndroidAttentionRequirementId,
    pub observation_id: AndroidAttentionObservationId,
    pub attention_certificate_id: AndroidAttentionCertificateId,
    pub policy_context_id: AndroidAttentionPolicyContextId,
    pub policy_generation: u64,
    pub policy_state_root: AndroidAttentionPolicyStateRoot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidAttentionPolicyError {
    ZeroPolicyContext,
    ZeroPolicyGeneration,
    ZeroPolicyStateRoot,
    ZeroAuthorizedRequirement,
    RequirementNotAuthorized,
    Attention(AndroidAttentionError),
}

impl fmt::Display for AndroidAttentionPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AndroidAttentionPolicyError {}

impl From<AndroidAttentionError> for AndroidAttentionPolicyError {
    fn from(value: AndroidAttentionError) -> Self { Self::Attention(value) }
}

impl AndroidAttentionPolicyAdmission {
    pub fn validate(&self) -> Result<(), AndroidAttentionPolicyError> {
        if self.policy_context_id.is_zero() {
            return Err(AndroidAttentionPolicyError::ZeroPolicyContext);
        }
        if self.policy_generation == 0 {
            return Err(AndroidAttentionPolicyError::ZeroPolicyGeneration);
        }
        if self.policy_state_root.is_zero() {
            return Err(AndroidAttentionPolicyError::ZeroPolicyStateRoot);
        }
        if self.authorized_requirement_id.is_zero() {
            return Err(AndroidAttentionPolicyError::ZeroAuthorizedRequirement);
        }
        Ok(())
    }

    pub fn admission_id(
        &self,
    ) -> Result<AndroidAttentionPolicyAdmissionId, AndroidAttentionPolicyError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(ADMISSION_DOMAIN);
        hasher.update(self.policy_context_id.as_bytes());
        hasher.update(&self.policy_generation.to_le_bytes());
        hasher.update(self.policy_state_root.as_bytes());
        hasher.update(self.authorized_requirement_id.as_bytes());
        Ok(AndroidAttentionPolicyAdmissionId(*hasher.finalize().as_bytes()))
    }

    pub fn certify(
        &self,
        requirement: &AndroidAttentionRequirement,
        observation: &AndroidAttentionObservation,
    ) -> Result<AndroidAttentionPolicyCertificate, AndroidAttentionPolicyError> {
        self.validate()?;
        let requirement_id = requirement.requirement_id()?;
        if requirement_id != self.authorized_requirement_id {
            return Err(AndroidAttentionPolicyError::RequirementNotAuthorized);
        }

        let current = CurrentAndroidAttentionPolicy {
            policy_context_id: self.policy_context_id,
            policy_generation: self.policy_generation,
            policy_state_root: self.policy_state_root,
        };
        let attention_certificate = requirement.evaluate(&current, observation)?;

        Ok(AndroidAttentionPolicyCertificate {
            admission_id: self.admission_id()?,
            requirement_id,
            observation_id: observation.observation_id()?,
            attention_certificate_id: attention_certificate.certificate_id()?,
            policy_context_id: self.policy_context_id,
            policy_generation: self.policy_generation,
            policy_state_root: self.policy_state_root,
        })
    }
}

impl AndroidAttentionPolicyCertificate {
    pub fn certificate_id(
        &self,
    ) -> Result<AndroidAttentionPolicyCertificateId, AndroidAttentionPolicyError> {
        if self.admission_id.is_zero() || self.requirement_id.is_zero() {
            return Err(AndroidAttentionPolicyError::ZeroAuthorizedRequirement);
        }
        if self.observation_id.is_zero() || self.attention_certificate_id.is_zero() {
            return Err(AndroidAttentionPolicyError::RequirementNotAuthorized);
        }
        if self.policy_context_id.is_zero() {
            return Err(AndroidAttentionPolicyError::ZeroPolicyContext);
        }
        if self.policy_generation == 0 {
            return Err(AndroidAttentionPolicyError::ZeroPolicyGeneration);
        }
        if self.policy_state_root.is_zero() {
            return Err(AndroidAttentionPolicyError::ZeroPolicyStateRoot);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.admission_id.as_bytes());
        hasher.update(self.requirement_id.as_bytes());
        hasher.update(self.observation_id.as_bytes());
        hasher.update(self.attention_certificate_id.as_bytes());
        hasher.update(self.policy_context_id.as_bytes());
        hasher.update(&self.policy_generation.to_le_bytes());
        hasher.update(self.policy_state_root.as_bytes());
        Ok(AndroidAttentionPolicyCertificateId(*hasher.finalize().as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_android_attention::AndroidAttentionObservation;
    use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

    fn d(v: u8) -> [u8; 32] { [v; 32] }

    fn requirement() -> AndroidAttentionRequirement {
        AndroidAttentionRequirement {
            policy_context_id: AndroidAttentionPolicyContextId(d(1)),
            policy_generation: 4,
            policy_state_root: AndroidAttentionPolicyStateRoot(d(2)),
            trusted_surface_id: TrustedSurfaceId(d(3)),
            minimum_sdk_int: 31,
            require_window_focus: true,
            require_flag_secure: true,
            require_hide_application_overlays: true,
            require_filter_touches_when_obscured: true,
            require_view_attached: true,
            require_view_shown: true,
            require_top_resumed: true,
            forbid_multi_window: true,
        }
    }

    fn observation() -> AndroidAttentionObservation {
        AndroidAttentionObservation {
            sdk_int: 34,
            trusted_surface_id: TrustedSurfaceId(d(3)),
            surface_generation: 5,
            interaction_generation: 6,
            window_has_focus: true,
            flag_secure_set: true,
            hide_application_overlays_requested: true,
            filter_touches_when_obscured_enabled: true,
            view_attached_to_window: true,
            view_shown: true,
            activity_top_resumed: true,
            activity_in_multi_window_mode: false,
        }
    }

    fn admission() -> AndroidAttentionPolicyAdmission {
        AndroidAttentionPolicyAdmission {
            policy_context_id: AndroidAttentionPolicyContextId(d(1)),
            policy_generation: 4,
            policy_state_root: AndroidAttentionPolicyStateRoot(d(2)),
            authorized_requirement_id: requirement().requirement_id().unwrap(),
        }
    }

    #[test]
    fn exact_requirement_is_admitted() {
        let cert = admission().certify(&requirement(), &observation()).unwrap();
        assert!(!cert.certificate_id().unwrap().is_zero());
    }

    #[test]
    fn weaker_caller_selected_requirement_is_rejected() {
        let mut weaker = requirement();
        weaker.require_hide_application_overlays = false;
        assert_eq!(
            admission().certify(&weaker, &observation()),
            Err(AndroidAttentionPolicyError::RequirementNotAuthorized)
        );
    }

    #[test]
    fn stale_policy_generation_does_not_match_requirement() {
        let mut stale = admission();
        stale.policy_generation = 5;
        assert!(matches!(
            stale.certify(&requirement(), &observation()),
            Err(AndroidAttentionPolicyError::Attention(
                AndroidAttentionError::PolicyGenerationMismatch
            ))
        ));
    }
}
