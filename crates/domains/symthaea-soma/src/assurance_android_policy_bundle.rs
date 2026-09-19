// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDPOLICYBUNDLE-529: canonical closure of Android touch-assurance policy.
//!
//! Event-time JNI must not be allowed to choose the policies under which its own
//! evidence is judged. This module therefore closes over the exact policy objects
//! required by QUAL-INGRESSPOLICY-523, QUAL-ANDROIDATTENTIONPOLICY-526, and
//! QUAL-ANDROIDMOTIONPOLICY-527 before any Android touch is processed.
//!
//! A bundle identity is issued only after every component identity is recomputed
//! and all cross-component relationships are exact. The bundle is policy evidence;
//! this module does not authenticate who installed it or make it current. A later
//! session/authority layer must do that explicitly.

use core::fmt;

use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

use crate::assurance_android_attention::{
    AndroidAttentionError, AndroidAttentionRequirement, AndroidAttentionRequirementId,
};
use crate::assurance_android_attention_policy::{
    AndroidAttentionPolicyAdmission, AndroidAttentionPolicyAdmissionId,
    AndroidAttentionPolicyError,
};
use crate::assurance_android_motion_event::{
    AndroidMotionEventError, AndroidMotionEventRequirement, AndroidMotionEventRequirementId,
};
use crate::assurance_android_motion_policy::{
    AndroidMotionPolicyAdmission, AndroidMotionPolicyAdmissionId, AndroidMotionPolicyError,
};
use crate::assurance_ingress_policy::{
    IngressAdmissionPolicy, IngressAdmissionPolicyId, IngressPolicyError,
};
use crate::assurance_platform_ingress::{
    PlatformIngressError, PlatformIngressKind, PlatformIngressProfile, PlatformIngressProfileId,
};

const BUNDLE_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-touch-policy-bundle\0";

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AndroidTouchPolicyBundleId(pub [u8; 32]);

impl AndroidTouchPolicyBundleId {
    pub const ZERO: Self = Self([0; 32]);

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn is_zero(&self) -> bool {
        self.0 == [0; 32]
    }
}

/// Complete semantic policy closure required for an assurance-bearing Android
/// touch. Installation/currentness authority is deliberately out of scope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyBundle {
    pub ingress_policy: IngressAdmissionPolicy,
    pub ingress_profile: PlatformIngressProfile,
    pub attention_policy_admission: AndroidAttentionPolicyAdmission,
    pub attention_requirement: AndroidAttentionRequirement,
    pub motion_policy_admission: AndroidMotionPolicyAdmission,
    pub motion_requirement: AndroidMotionEventRequirement,
}

/// Recomputed component identities for a validated bundle. These are useful to
/// pin later session installation without reinterpreting caller-provided fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyBundleIdentity {
    pub bundle_id: AndroidTouchPolicyBundleId,
    pub ingress_policy_id: IngressAdmissionPolicyId,
    pub ingress_profile_id: PlatformIngressProfileId,
    pub attention_policy_admission_id: AndroidAttentionPolicyAdmissionId,
    pub attention_requirement_id: AndroidAttentionRequirementId,
    pub motion_policy_admission_id: AndroidMotionPolicyAdmissionId,
    pub motion_requirement_id: AndroidMotionEventRequirementId,
    pub trusted_surface_id: TrustedSurfaceId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidTouchPolicyBundleError {
    NotAndroidIngress,
    TouchIngressNotAllowed,
    IngressProfileMismatch,
    TrustedSurfaceMismatch,
    AttentionPolicyContextMismatch,
    AttentionPolicyGenerationMismatch,
    AttentionPolicyStateRootMismatch,
    AttentionRequirementNotAuthorized,
    MotionPolicyContextMismatch,
    MotionPolicyGenerationMismatch,
    MotionPolicyStateRootMismatch,
    AttentionPolicyAdmissionMismatch,
    MotionRequirementNotAuthorized,
    MotionAttentionRequirementMismatch,
    IngressPolicy(IngressPolicyError),
    IngressProfile(PlatformIngressError),
    Attention(AndroidAttentionError),
    AttentionPolicy(AndroidAttentionPolicyError),
    Motion(AndroidMotionEventError),
    MotionPolicy(AndroidMotionPolicyError),
}

impl fmt::Display for AndroidTouchPolicyBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AndroidTouchPolicyBundleError {}

impl From<IngressPolicyError> for AndroidTouchPolicyBundleError {
    fn from(value: IngressPolicyError) -> Self {
        Self::IngressPolicy(value)
    }
}

impl From<PlatformIngressError> for AndroidTouchPolicyBundleError {
    fn from(value: PlatformIngressError) -> Self {
        Self::IngressProfile(value)
    }
}

impl From<AndroidAttentionError> for AndroidTouchPolicyBundleError {
    fn from(value: AndroidAttentionError) -> Self {
        Self::Attention(value)
    }
}

impl From<AndroidAttentionPolicyError> for AndroidTouchPolicyBundleError {
    fn from(value: AndroidAttentionPolicyError) -> Self {
        Self::AttentionPolicy(value)
    }
}

impl From<AndroidMotionEventError> for AndroidTouchPolicyBundleError {
    fn from(value: AndroidMotionEventError) -> Self {
        Self::Motion(value)
    }
}

impl From<AndroidMotionPolicyError> for AndroidTouchPolicyBundleError {
    fn from(value: AndroidMotionPolicyError) -> Self {
        Self::MotionPolicy(value)
    }
}

impl AndroidTouchPolicyBundle {
    pub fn validate(
        &self,
    ) -> Result<AndroidTouchPolicyBundleIdentity, AndroidTouchPolicyBundleError> {
        self.ingress_policy.validate()?;
        self.ingress_profile.validate()?;
        self.attention_policy_admission.validate()?;
        self.attention_requirement.validate()?;
        self.motion_policy_admission.validate()?;
        self.motion_requirement.validate()?;

        if self.ingress_policy.platform != PlatformIngressKind::AndroidJni
            || self.ingress_profile.platform != PlatformIngressKind::AndroidJni
        {
            return Err(AndroidTouchPolicyBundleError::NotAndroidIngress);
        }
        if !self.ingress_policy.allow_touch {
            return Err(AndroidTouchPolicyBundleError::TouchIngressNotAllowed);
        }

        let ingress_profile_id = self.ingress_profile.profile_id()?;
        if self.ingress_policy.ingress_profile_id != ingress_profile_id {
            return Err(AndroidTouchPolicyBundleError::IngressProfileMismatch);
        }
        if self.ingress_policy.trusted_surface_id != self.attention_requirement.trusted_surface_id {
            return Err(AndroidTouchPolicyBundleError::TrustedSurfaceMismatch);
        }

        if self.attention_policy_admission.policy_context_id
            != self.attention_requirement.policy_context_id
        {
            return Err(AndroidTouchPolicyBundleError::AttentionPolicyContextMismatch);
        }
        if self.attention_policy_admission.policy_generation
            != self.attention_requirement.policy_generation
        {
            return Err(AndroidTouchPolicyBundleError::AttentionPolicyGenerationMismatch);
        }
        if self.attention_policy_admission.policy_state_root
            != self.attention_requirement.policy_state_root
        {
            return Err(AndroidTouchPolicyBundleError::AttentionPolicyStateRootMismatch);
        }

        let attention_requirement_id = self.attention_requirement.requirement_id()?;
        if self.attention_policy_admission.authorized_requirement_id != attention_requirement_id {
            return Err(AndroidTouchPolicyBundleError::AttentionRequirementNotAuthorized);
        }
        let attention_policy_admission_id = self.attention_policy_admission.admission_id()?;

        if self.motion_policy_admission.policy_context_id
            != self.attention_policy_admission.policy_context_id
        {
            return Err(AndroidTouchPolicyBundleError::MotionPolicyContextMismatch);
        }
        if self.motion_policy_admission.policy_generation
            != self.attention_policy_admission.policy_generation
        {
            return Err(AndroidTouchPolicyBundleError::MotionPolicyGenerationMismatch);
        }
        if self.motion_policy_admission.policy_state_root
            != self.attention_policy_admission.policy_state_root
        {
            return Err(AndroidTouchPolicyBundleError::MotionPolicyStateRootMismatch);
        }
        if self.motion_policy_admission.attention_policy_admission_id
            != attention_policy_admission_id
        {
            return Err(AndroidTouchPolicyBundleError::AttentionPolicyAdmissionMismatch);
        }

        let motion_requirement_id = self.motion_requirement.requirement_id()?;
        if self.motion_policy_admission.authorized_motion_requirement_id != motion_requirement_id {
            return Err(AndroidTouchPolicyBundleError::MotionRequirementNotAuthorized);
        }
        if self.motion_requirement.attention_requirement_id != attention_requirement_id {
            return Err(AndroidTouchPolicyBundleError::MotionAttentionRequirementMismatch);
        }

        let ingress_policy_id = self.ingress_policy.policy_id()?;
        let motion_policy_admission_id = self.motion_policy_admission.admission_id()?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(BUNDLE_DOMAIN);
        hasher.update(ingress_policy_id.as_bytes());
        hasher.update(ingress_profile_id.as_bytes());
        hasher.update(attention_policy_admission_id.as_bytes());
        hasher.update(attention_requirement_id.as_bytes());
        hasher.update(motion_policy_admission_id.as_bytes());
        hasher.update(motion_requirement_id.as_bytes());
        let bundle_id = AndroidTouchPolicyBundleId(*hasher.finalize().as_bytes());

        Ok(AndroidTouchPolicyBundleIdentity {
            bundle_id,
            ingress_policy_id,
            ingress_profile_id,
            attention_policy_admission_id,
            attention_requirement_id,
            motion_policy_admission_id,
            motion_requirement_id,
            trusted_surface_id: self.attention_requirement.trusted_surface_id,
        })
    }

    pub fn bundle_id(
        &self,
    ) -> Result<AndroidTouchPolicyBundleId, AndroidTouchPolicyBundleError> {
        Ok(self.validate()?.bundle_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_android_attention::{
        AndroidAttentionPolicyContextId, AndroidAttentionPolicyStateRoot,
    };
    use crate::assurance_ingress_policy::{IngressPolicyContextId, IngressPolicyStateRoot};
    use crate::assurance_soma_interaction::{ScreenCaptureProfileId, TouchInputProfileId};
    use symthaea_core::assurance_interaction_continuity::InputAttesterId;

    fn d(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn ingress_profile(platform: PlatformIngressKind) -> PlatformIngressProfile {
        PlatformIngressProfile {
            platform,
            abi_version: 1,
            capture_profile_id: ScreenCaptureProfileId(d(20)),
            touch_input_profile_id: TouchInputProfileId(d(21)),
            max_frame_bytes: 1024,
        }
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

    fn motion_admission() -> AndroidMotionPolicyAdmission {
        AndroidMotionPolicyAdmission {
            policy_context_id: AndroidAttentionPolicyContextId(d(1)),
            policy_generation: 4,
            policy_state_root: AndroidAttentionPolicyStateRoot(d(2)),
            attention_policy_admission_id: attention_admission().admission_id().unwrap(),
            authorized_motion_requirement_id: motion_requirement().requirement_id().unwrap(),
        }
    }

    fn bundle(platform: PlatformIngressKind) -> AndroidTouchPolicyBundle {
        let profile = ingress_profile(platform);
        let policy = IngressAdmissionPolicy {
            policy_context_id: IngressPolicyContextId(d(22)),
            policy_generation: 7,
            policy_state_root: IngressPolicyStateRoot(d(23)),
            platform,
            ingress_profile_id: profile.profile_id().unwrap(),
            trusted_surface_id: TrustedSurfaceId(d(3)),
            input_attester_id: InputAttesterId(d(8)),
            allow_frame: false,
            allow_touch: true,
        };
        AndroidTouchPolicyBundle {
            ingress_policy: policy,
            ingress_profile: profile,
            attention_policy_admission: attention_admission(),
            attention_requirement: attention_requirement(),
            motion_policy_admission: motion_admission(),
            motion_requirement: motion_requirement(),
        }
    }

    #[test]
    fn exact_policy_closure_has_stable_identity() {
        let bundle = bundle(PlatformIngressKind::AndroidJni);
        let identity = bundle.validate().unwrap();
        assert!(!identity.bundle_id.is_zero());
        assert_eq!(identity.bundle_id, bundle.bundle_id().unwrap());
        assert_eq!(identity.trusted_surface_id, TrustedSurfaceId(d(3)));
    }

    #[test]
    fn non_android_profile_is_rejected() {
        assert_eq!(
            bundle(PlatformIngressKind::IosCAbi).validate(),
            Err(AndroidTouchPolicyBundleError::NotAndroidIngress)
        );
    }

    #[test]
    fn surface_mismatch_between_ingress_and_attention_is_rejected() {
        let mut bundle = bundle(PlatformIngressKind::AndroidJni);
        bundle.ingress_policy.trusted_surface_id = TrustedSurfaceId(d(99));
        assert_eq!(
            bundle.validate(),
            Err(AndroidTouchPolicyBundleError::TrustedSurfaceMismatch)
        );
    }

    #[test]
    fn weaker_attention_requirement_is_not_silently_substituted() {
        let mut bundle = bundle(PlatformIngressKind::AndroidJni);
        bundle.attention_requirement.require_hide_application_overlays = false;
        assert_eq!(
            bundle.validate(),
            Err(AndroidTouchPolicyBundleError::AttentionRequirementNotAuthorized)
        );
    }

    #[test]
    fn weaker_motion_requirement_is_not_silently_substituted() {
        let mut bundle = bundle(PlatformIngressKind::AndroidJni);
        bundle.motion_requirement.reject_partially_obscured = false;
        assert_eq!(
            bundle.validate(),
            Err(AndroidTouchPolicyBundleError::MotionRequirementNotAuthorized)
        );
    }

    #[test]
    fn motion_requirement_cannot_float_to_another_attention_requirement() {
        let mut bundle = bundle(PlatformIngressKind::AndroidJni);
        bundle.motion_requirement.attention_requirement_id = AndroidAttentionRequirementId(d(77));
        assert_eq!(
            bundle.validate(),
            Err(AndroidTouchPolicyBundleError::MotionRequirementNotAuthorized)
        );
    }
}
