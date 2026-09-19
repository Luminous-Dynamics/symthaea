// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDMOTIONPOLICY-527: require current policy admission of the exact
//! Android MotionEvent requirement.
//!
//! QUAL-ANDROIDMOTION-525 proves that one Android MotionEvent satisfies a supplied
//! motion requirement and is bound to one trusted-attention observation plus one
//! measured touch observation. QUAL-ANDROIDATTENTIONPOLICY-526 separately proves
//! that current policy admitted the exact nested attention requirement. Neither
//! theorem, by itself, prevents a caller from selecting a weaker 525 motion
//! requirement (for example, allowing partial obscuration or multiple pointers).
//!
//! This tranche closes that gap. One exact motion-policy admission names the
//! authorized `AndroidMotionEventRequirementId`, binds it to the exact 526
//! attention-policy admission, and recomputes both 526 and 525 before issuing a
//! certificate.

use core::fmt;

use crate::assurance_android_attention::{
    AndroidAttentionObservation, AndroidAttentionPolicyContextId, AndroidAttentionPolicyStateRoot,
    AndroidAttentionRequirement, AndroidAttentionRequirementId, CurrentAndroidAttentionPolicy,
};
use crate::assurance_android_attention_policy::{
    AndroidAttentionPolicyAdmission, AndroidAttentionPolicyAdmissionId,
    AndroidAttentionPolicyCertificateId, AndroidAttentionPolicyError,
};
use crate::assurance_android_motion_event::{
    certify_android_motion_event, AndroidMotionEventCertificateId, AndroidMotionEventError,
    AndroidMotionEventObservation, AndroidMotionEventObservationId, AndroidMotionEventRequirement,
    AndroidMotionEventRequirementId,
};
use crate::assurance_soma_interaction::{TouchEventObservation, TouchEventObservationId};

const ADMISSION_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-motion-policy-admission\0";
const CERTIFICATE_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-motion-policy-certificate\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub [u8; 32]);

        impl $name {
            pub const ZERO: Self = Self([0; 32]);

            pub const fn as_bytes(&self) -> &[u8; 32] {
                &self.0
            }

            pub fn is_zero(&self) -> bool {
                self.0 == [0; 32]
            }
        }
    };
}

digest_id!(AndroidMotionPolicyAdmissionId);
digest_id!(AndroidMotionPolicyCertificateId);

/// Exact current-policy admission for one Android MotionEvent requirement.
///
/// `attention_policy_admission_id` is deliberately explicit: motion policy cannot
/// silently float to another 526 admission that happens to share a context or
/// generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidMotionPolicyAdmission {
    pub policy_context_id: AndroidAttentionPolicyContextId,
    pub policy_generation: u64,
    pub policy_state_root: AndroidAttentionPolicyStateRoot,
    pub attention_policy_admission_id: AndroidAttentionPolicyAdmissionId,
    pub authorized_motion_requirement_id: AndroidMotionEventRequirementId,
}

/// Proof that one exact MotionEvent observation was accepted under one exact
/// current motion-policy admission, with both 526 and 525 recomputed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidMotionPolicyCertificate {
    pub admission_id: AndroidMotionPolicyAdmissionId,
    pub motion_requirement_id: AndroidMotionEventRequirementId,
    pub motion_observation_id: AndroidMotionEventObservationId,
    pub motion_certificate_id: AndroidMotionEventCertificateId,
    pub attention_policy_admission_id: AndroidAttentionPolicyAdmissionId,
    pub attention_policy_certificate_id: AndroidAttentionPolicyCertificateId,
    pub attention_requirement_id: AndroidAttentionRequirementId,
    pub touch_observation_id: TouchEventObservationId,
    pub policy_context_id: AndroidAttentionPolicyContextId,
    pub policy_generation: u64,
    pub policy_state_root: AndroidAttentionPolicyStateRoot,
    pub interaction_generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidMotionPolicyError {
    ZeroPolicyContext,
    ZeroPolicyGeneration,
    ZeroPolicyStateRoot,
    ZeroAttentionPolicyAdmission,
    ZeroAuthorizedMotionRequirement,
    PolicyContextMismatch,
    PolicyGenerationMismatch,
    PolicyStateRootMismatch,
    AttentionPolicyAdmissionMismatch,
    MotionRequirementNotAuthorized,
    AttentionPolicy(AndroidAttentionPolicyError),
    Motion(AndroidMotionEventError),
}

impl fmt::Display for AndroidMotionPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AndroidMotionPolicyError {}

impl From<AndroidAttentionPolicyError> for AndroidMotionPolicyError {
    fn from(value: AndroidAttentionPolicyError) -> Self {
        Self::AttentionPolicy(value)
    }
}

impl From<AndroidMotionEventError> for AndroidMotionPolicyError {
    fn from(value: AndroidMotionEventError) -> Self {
        Self::Motion(value)
    }
}

impl AndroidMotionPolicyAdmission {
    pub fn validate(&self) -> Result<(), AndroidMotionPolicyError> {
        if self.policy_context_id.is_zero() {
            return Err(AndroidMotionPolicyError::ZeroPolicyContext);
        }
        if self.policy_generation == 0 {
            return Err(AndroidMotionPolicyError::ZeroPolicyGeneration);
        }
        if self.policy_state_root.is_zero() {
            return Err(AndroidMotionPolicyError::ZeroPolicyStateRoot);
        }
        if self.attention_policy_admission_id.is_zero() {
            return Err(AndroidMotionPolicyError::ZeroAttentionPolicyAdmission);
        }
        if self.authorized_motion_requirement_id.is_zero() {
            return Err(AndroidMotionPolicyError::ZeroAuthorizedMotionRequirement);
        }
        Ok(())
    }

    pub fn admission_id(
        &self,
    ) -> Result<AndroidMotionPolicyAdmissionId, AndroidMotionPolicyError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(ADMISSION_DOMAIN);
        hasher.update(self.policy_context_id.as_bytes());
        hasher.update(&self.policy_generation.to_le_bytes());
        hasher.update(self.policy_state_root.as_bytes());
        hasher.update(self.attention_policy_admission_id.as_bytes());
        hasher.update(self.authorized_motion_requirement_id.as_bytes());
        Ok(AndroidMotionPolicyAdmissionId(*hasher.finalize().as_bytes()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn certify(
        &self,
        attention_admission: &AndroidAttentionPolicyAdmission,
        attention_requirement: &AndroidAttentionRequirement,
        attention_observation: &AndroidAttentionObservation,
        motion_requirement: &AndroidMotionEventRequirement,
        motion_observation: &AndroidMotionEventObservation,
        touch_observation: &TouchEventObservation,
    ) -> Result<AndroidMotionPolicyCertificate, AndroidMotionPolicyError> {
        self.validate()?;

        if attention_admission.policy_context_id != self.policy_context_id {
            return Err(AndroidMotionPolicyError::PolicyContextMismatch);
        }
        if attention_admission.policy_generation != self.policy_generation {
            return Err(AndroidMotionPolicyError::PolicyGenerationMismatch);
        }
        if attention_admission.policy_state_root != self.policy_state_root {
            return Err(AndroidMotionPolicyError::PolicyStateRootMismatch);
        }

        let attention_policy_admission_id = attention_admission.admission_id()?;
        if attention_policy_admission_id != self.attention_policy_admission_id {
            return Err(AndroidMotionPolicyError::AttentionPolicyAdmissionMismatch);
        }

        let motion_requirement_id = motion_requirement.requirement_id()?;
        if motion_requirement_id != self.authorized_motion_requirement_id {
            return Err(AndroidMotionPolicyError::MotionRequirementNotAuthorized);
        }

        // Recompute QUAL-ANDROIDATTENTIONPOLICY-526 rather than trusting a supplied
        // certificate or merely comparing a nested requirement ID.
        let attention_policy_certificate =
            attention_admission.certify(attention_requirement, attention_observation)?;
        let attention_policy_certificate_id = attention_policy_certificate.certificate_id()?;

        // Recompute QUAL-ANDROIDMOTION-525 under the exact same current policy
        // context/generation/root that was admitted above.
        let current_attention_policy = CurrentAndroidAttentionPolicy {
            policy_context_id: self.policy_context_id,
            policy_generation: self.policy_generation,
            policy_state_root: self.policy_state_root,
        };
        let motion_certificate = certify_android_motion_event(
            attention_requirement,
            &current_attention_policy,
            attention_observation,
            motion_requirement,
            motion_observation,
            touch_observation,
        )?;

        Ok(AndroidMotionPolicyCertificate {
            admission_id: self.admission_id()?,
            motion_requirement_id,
            motion_observation_id: motion_observation.observation_id()?,
            motion_certificate_id: motion_certificate.certificate_id()?,
            attention_policy_admission_id,
            attention_policy_certificate_id,
            attention_requirement_id: attention_requirement.requirement_id()?,
            touch_observation_id: touch_observation
                .observation_id()
                .map_err(|_| AndroidMotionPolicyError::Motion(
                    AndroidMotionEventError::TouchInteractionGenerationMismatch,
                ))?,
            policy_context_id: self.policy_context_id,
            policy_generation: self.policy_generation,
            policy_state_root: self.policy_state_root,
            interaction_generation: attention_observation.interaction_generation,
        })
    }
}

impl AndroidMotionPolicyCertificate {
    pub fn certificate_id(
        &self,
    ) -> Result<AndroidMotionPolicyCertificateId, AndroidMotionPolicyError> {
        if self.admission_id.is_zero() {
            return Err(AndroidMotionPolicyError::ZeroAuthorizedMotionRequirement);
        }
        if self.motion_requirement_id.is_zero() {
            return Err(AndroidMotionPolicyError::ZeroAuthorizedMotionRequirement);
        }
        if self.motion_observation_id.is_zero() || self.motion_certificate_id.is_zero() {
            return Err(AndroidMotionPolicyError::MotionRequirementNotAuthorized);
        }
        if self.attention_policy_admission_id.is_zero()
            || self.attention_policy_certificate_id.is_zero()
        {
            return Err(AndroidMotionPolicyError::ZeroAttentionPolicyAdmission);
        }
        if self.attention_requirement_id.is_zero() || self.touch_observation_id.is_zero() {
            return Err(AndroidMotionPolicyError::MotionRequirementNotAuthorized);
        }
        if self.policy_context_id.is_zero() {
            return Err(AndroidMotionPolicyError::ZeroPolicyContext);
        }
        if self.policy_generation == 0 {
            return Err(AndroidMotionPolicyError::ZeroPolicyGeneration);
        }
        if self.policy_state_root.is_zero() {
            return Err(AndroidMotionPolicyError::ZeroPolicyStateRoot);
        }
        if self.interaction_generation == 0 {
            return Err(AndroidMotionPolicyError::MotionRequirementNotAuthorized);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.admission_id.as_bytes());
        hasher.update(self.motion_requirement_id.as_bytes());
        hasher.update(self.motion_observation_id.as_bytes());
        hasher.update(self.motion_certificate_id.as_bytes());
        hasher.update(self.attention_policy_admission_id.as_bytes());
        hasher.update(self.attention_policy_certificate_id.as_bytes());
        hasher.update(self.attention_requirement_id.as_bytes());
        hasher.update(self.touch_observation_id.as_bytes());
        hasher.update(self.policy_context_id.as_bytes());
        hasher.update(&self.policy_generation.to_le_bytes());
        hasher.update(self.policy_state_root.as_bytes());
        hasher.update(&self.interaction_generation.to_le_bytes());
        Ok(AndroidMotionPolicyCertificateId(*hasher.finalize().as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_android_motion_event::AndroidMotionEventRequirement;
    use crate::assurance_soma_interaction::{
        CanonicalUnitF32, ObservedTouchAction, TouchInputProfileId,
    };
    use symthaea_core::assurance_interaction_continuity::InputAttesterId;
    use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

    fn d(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn attention_requirement() -> AndroidAttentionRequirement {
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

    fn attention_observation() -> AndroidAttentionObservation {
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

    fn attention_admission() -> AndroidAttentionPolicyAdmission {
        AndroidAttentionPolicyAdmission {
            policy_context_id: AndroidAttentionPolicyContextId(d(1)),
            policy_generation: 4,
            policy_state_root: AndroidAttentionPolicyStateRoot(d(2)),
            authorized_requirement_id: attention_requirement().requirement_id().unwrap(),
        }
    }

    fn motion_requirement() -> AndroidMotionEventRequirement {
        AndroidMotionEventRequirement {
            attention_requirement_id: attention_requirement().requirement_id().unwrap(),
            reject_fully_obscured: true,
            reject_partially_obscured: true,
            require_single_pointer: true,
        }
    }

    fn motion_observation() -> AndroidMotionEventObservation {
        AndroidMotionEventObservation::new(
            34,
            0,
            ObservedTouchAction::Down,
            1,
            0.25,
            0.75,
            0.5,
            1234,
        )
        .unwrap()
    }

    fn touch_observation() -> TouchEventObservation {
        TouchEventObservation {
            input_profile_id: TouchInputProfileId(d(7)),
            input_attester_id: InputAttesterId(d(8)),
            interaction_generation: 6,
            input_sequence: 1,
            x: CanonicalUnitF32::from_value(0.25).unwrap(),
            y: CanonicalUnitF32::from_value(0.75).unwrap(),
            pressure: CanonicalUnitF32::from_value(0.5).unwrap(),
            action: ObservedTouchAction::Down,
            platform_timestamp_ms: 1234,
        }
    }

    fn admission() -> AndroidMotionPolicyAdmission {
        AndroidMotionPolicyAdmission {
            policy_context_id: AndroidAttentionPolicyContextId(d(1)),
            policy_generation: 4,
            policy_state_root: AndroidAttentionPolicyStateRoot(d(2)),
            attention_policy_admission_id: attention_admission().admission_id().unwrap(),
            authorized_motion_requirement_id: motion_requirement().requirement_id().unwrap(),
        }
    }

    #[test]
    fn exact_motion_requirement_is_admitted() {
        let certificate = admission()
            .certify(
                &attention_admission(),
                &attention_requirement(),
                &attention_observation(),
                &motion_requirement(),
                &motion_observation(),
                &touch_observation(),
            )
            .unwrap();
        assert!(!certificate.certificate_id().unwrap().is_zero());
    }

    #[test]
    fn weaker_caller_selected_motion_requirement_is_rejected() {
        let mut weaker = motion_requirement();
        weaker.reject_partially_obscured = false;
        assert_eq!(
            admission().certify(
                &attention_admission(),
                &attention_requirement(),
                &attention_observation(),
                &weaker,
                &motion_observation(),
                &touch_observation(),
            ),
            Err(AndroidMotionPolicyError::MotionRequirementNotAuthorized)
        );
    }

    #[test]
    fn foreign_attention_policy_admission_is_rejected() {
        let mut foreign = attention_admission();
        foreign.authorized_requirement_id = AndroidAttentionRequirementId(d(99));
        assert_eq!(
            admission().certify(
                &foreign,
                &attention_requirement(),
                &attention_observation(),
                &motion_requirement(),
                &motion_observation(),
                &touch_observation(),
            ),
            Err(AndroidMotionPolicyError::AttentionPolicyAdmissionMismatch)
        );
    }

    #[test]
    fn stale_policy_generation_is_rejected_before_motion_evaluation() {
        let mut stale = admission();
        stale.policy_generation = 5;
        assert_eq!(
            stale.certify(
                &attention_admission(),
                &attention_requirement(),
                &attention_observation(),
                &motion_requirement(),
                &motion_observation(),
                &touch_observation(),
            ),
            Err(AndroidMotionPolicyError::PolicyGenerationMismatch)
        );
    }
}
