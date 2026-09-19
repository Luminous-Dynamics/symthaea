// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDTOUCHJOIN-528: join Android MotionEvent policy evidence to the
//! exact policy-admitted native touch ingress evidence.
//!
//! QUAL-INGRESSPOLICY-523 proves that one exact measured touch observation and
//! ingress receipt satisfy the current strict platform-ingress policy.
//! QUAL-ANDROIDMOTIONPOLICY-527 proves that one exact Android MotionEvent and
//! trusted-attention observation satisfy the current Android motion policy.
//! Those proofs are independent until their shared touch identity and Android
//! context are checked explicitly.
//!
//! This tranche recomputes both sides, requires Android JNI ingress, binds the
//! same `TouchEventObservationId`, requires the ingress policy's trusted surface
//! to equal the trusted-attention surface, and requires the attention snapshot
//! and MotionEvent observation to report the same Android SDK level.

use core::fmt;

use symthaea_core::assurance_interaction_continuity::InputAttesterId;
use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

use crate::assurance_android_attention::{
    AndroidAttentionObservation, AndroidAttentionRequirement,
};
use crate::assurance_android_attention_policy::AndroidAttentionPolicyAdmission;
use crate::assurance_android_motion_event::{
    AndroidMotionEventObservation, AndroidMotionEventRequirement,
};
use crate::assurance_android_motion_policy::{
    AndroidMotionPolicyAdmission, AndroidMotionPolicyAdmissionId,
    AndroidMotionPolicyCertificateId, AndroidMotionPolicyError,
};
use crate::assurance_ingress_policy::{
    IngressAdmissionPolicy, IngressAdmissionPolicyId, IngressPolicyError,
    IngressPolicySnapshot, TouchIngressAdmissionCertificate,
    TouchIngressAdmissionCertificateId,
};
use crate::assurance_platform_ingress::{
    PlatformIngressKind, PlatformIngressProfile, PlatformIngressProfileId,
    TouchIngressReceipt, TouchIngressReceiptId,
};
use crate::assurance_soma_interaction::{TouchEventObservation, TouchEventObservationId};

const CERTIFICATE_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-touch-join-certificate\0";

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AndroidTouchJoinCertificateId(pub [u8; 32]);

impl AndroidTouchJoinCertificateId {
    pub const ZERO: Self = Self([0; 32]);

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn is_zero(&self) -> bool {
        self.0 == [0; 32]
    }
}

/// Proof that exact 523 ingress evidence and exact 527 Android motion evidence
/// describe the same assurance-bearing Android touch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchJoinCertificate {
    pub ingress_admission_certificate_id: TouchIngressAdmissionCertificateId,
    pub motion_policy_certificate_id: AndroidMotionPolicyCertificateId,
    pub ingress_policy_id: IngressAdmissionPolicyId,
    pub motion_policy_admission_id: AndroidMotionPolicyAdmissionId,
    pub ingress_profile_id: PlatformIngressProfileId,
    pub touch_observation_id: TouchEventObservationId,
    pub ingress_receipt_id: TouchIngressReceiptId,
    pub trusted_surface_id: TrustedSurfaceId,
    pub input_attester_id: InputAttesterId,
    pub sdk_int: u32,
    pub interaction_generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidTouchJoinError {
    NotAndroidIngress,
    SdkMismatch,
    TrustedSurfaceMismatch,
    TouchObservationMismatch,
    InteractionGenerationMismatch,
    ZeroCertificateIdentity,
    ZeroSdk,
    Ingress(IngressPolicyError),
    MotionPolicy(AndroidMotionPolicyError),
}

impl fmt::Display for AndroidTouchJoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AndroidTouchJoinError {}

impl From<IngressPolicyError> for AndroidTouchJoinError {
    fn from(value: IngressPolicyError) -> Self {
        Self::Ingress(value)
    }
}

impl From<AndroidMotionPolicyError> for AndroidTouchJoinError {
    fn from(value: AndroidMotionPolicyError) -> Self {
        Self::MotionPolicy(value)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn certify_android_touch_join(
    ingress_policy: &IngressAdmissionPolicy,
    current_ingress_policy: &IngressPolicySnapshot,
    ingress_profile: &PlatformIngressProfile,
    touch_observation: &TouchEventObservation,
    touch_receipt: &TouchIngressReceipt,
    motion_policy_admission: &AndroidMotionPolicyAdmission,
    attention_admission: &AndroidAttentionPolicyAdmission,
    attention_requirement: &AndroidAttentionRequirement,
    attention_observation: &AndroidAttentionObservation,
    motion_requirement: &AndroidMotionEventRequirement,
    motion_observation: &AndroidMotionEventObservation,
) -> Result<AndroidTouchJoinCertificate, AndroidTouchJoinError> {
    if ingress_policy.platform != PlatformIngressKind::AndroidJni
        || ingress_profile.platform != PlatformIngressKind::AndroidJni
    {
        return Err(AndroidTouchJoinError::NotAndroidIngress);
    }

    // AndroidAttentionGuard and AndroidMotionEventGate both source this value
    // from Build.VERSION.SDK_INT. A mismatch is therefore contradictory evidence,
    // not a value to normalize or choose between.
    if attention_observation.sdk_int != motion_observation.sdk_int {
        return Err(AndroidTouchJoinError::SdkMismatch);
    }

    // 523's touch theorem authenticates the input attester/profile but a touch
    // observation itself carries no surface identity. Bind its policy-selected
    // surface to the trusted-attention surface here.
    if ingress_policy.trusted_surface_id != attention_observation.trusted_surface_id {
        return Err(AndroidTouchJoinError::TrustedSurfaceMismatch);
    }

    let ingress_certificate = TouchIngressAdmissionCertificate::admit(
        ingress_policy,
        current_ingress_policy,
        ingress_profile,
        touch_observation,
        touch_receipt,
    )?;
    let ingress_admission_certificate_id = ingress_certificate.certificate_id()?;

    let motion_policy_certificate = motion_policy_admission.certify(
        attention_admission,
        attention_requirement,
        attention_observation,
        motion_requirement,
        motion_observation,
        touch_observation,
    )?;
    let motion_policy_certificate_id = motion_policy_certificate.certificate_id()?;

    if ingress_certificate.touch_observation_id != motion_policy_certificate.touch_observation_id {
        return Err(AndroidTouchJoinError::TouchObservationMismatch);
    }
    if touch_observation.interaction_generation
        != motion_policy_certificate.interaction_generation
    {
        return Err(AndroidTouchJoinError::InteractionGenerationMismatch);
    }

    Ok(AndroidTouchJoinCertificate {
        ingress_admission_certificate_id,
        motion_policy_certificate_id,
        ingress_policy_id: ingress_certificate.policy_id,
        motion_policy_admission_id: motion_policy_certificate.admission_id,
        ingress_profile_id: ingress_certificate.ingress_profile_id,
        touch_observation_id: ingress_certificate.touch_observation_id,
        ingress_receipt_id: ingress_certificate.ingress_receipt_id,
        trusted_surface_id: ingress_policy.trusted_surface_id,
        input_attester_id: ingress_certificate.input_attester_id,
        sdk_int: attention_observation.sdk_int,
        interaction_generation: touch_observation.interaction_generation,
    })
}

impl AndroidTouchJoinCertificate {
    pub fn certificate_id(
        &self,
    ) -> Result<AndroidTouchJoinCertificateId, AndroidTouchJoinError> {
        if self.ingress_admission_certificate_id.is_zero()
            || self.motion_policy_certificate_id.is_zero()
            || self.ingress_policy_id.is_zero()
            || self.motion_policy_admission_id.is_zero()
            || self.ingress_profile_id.is_zero()
            || self.touch_observation_id.is_zero()
            || self.ingress_receipt_id.is_zero()
            || self.trusted_surface_id.is_zero()
            || self.input_attester_id.is_zero()
        {
            return Err(AndroidTouchJoinError::ZeroCertificateIdentity);
        }
        if self.sdk_int == 0 {
            return Err(AndroidTouchJoinError::ZeroSdk);
        }
        if self.interaction_generation == 0 {
            return Err(AndroidTouchJoinError::InteractionGenerationMismatch);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.ingress_admission_certificate_id.as_bytes());
        hasher.update(self.motion_policy_certificate_id.as_bytes());
        hasher.update(self.ingress_policy_id.as_bytes());
        hasher.update(self.motion_policy_admission_id.as_bytes());
        hasher.update(self.ingress_profile_id.as_bytes());
        hasher.update(self.touch_observation_id.as_bytes());
        hasher.update(self.ingress_receipt_id.as_bytes());
        hasher.update(self.trusted_surface_id.as_bytes());
        hasher.update(self.input_attester_id.as_bytes());
        hasher.update(&self.sdk_int.to_le_bytes());
        hasher.update(&self.interaction_generation.to_le_bytes());
        Ok(AndroidTouchJoinCertificateId(*hasher.finalize().as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_android_attention::{
        AndroidAttentionPolicyContextId, AndroidAttentionPolicyStateRoot,
        AndroidAttentionRequirementId,
    };
    use crate::assurance_android_attention_policy::AndroidAttentionPolicyAdmissionId;
    use crate::assurance_ingress_policy::{
        IngressPolicyContextId, IngressPolicyStateRoot,
    };
    use crate::assurance_soma_interaction::{
        ObservedTouchAction, ScreenCaptureProfileId, TouchInputProfileId,
    };

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

    fn ingress_policy(profile: &PlatformIngressProfile) -> IngressAdmissionPolicy {
        IngressAdmissionPolicy {
            policy_context_id: IngressPolicyContextId(d(22)),
            policy_generation: 7,
            policy_state_root: IngressPolicyStateRoot(d(23)),
            platform: profile.platform,
            ingress_profile_id: profile.profile_id().unwrap(),
            trusted_surface_id: TrustedSurfaceId(d(3)),
            input_attester_id: InputAttesterId(d(8)),
            allow_frame: false,
            allow_touch: true,
        }
    }

    fn current_ingress(policy: &IngressAdmissionPolicy) -> IngressPolicySnapshot {
        IngressPolicySnapshot {
            policy_context_id: policy.policy_context_id,
            policy_generation: policy.policy_generation,
            policy_state_root: policy.policy_state_root,
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

    fn motion_policy_admission() -> AndroidMotionPolicyAdmission {
        AndroidMotionPolicyAdmission {
            policy_context_id: AndroidAttentionPolicyContextId(d(1)),
            policy_generation: 4,
            policy_state_root: AndroidAttentionPolicyStateRoot(d(2)),
            attention_policy_admission_id: attention_admission().admission_id().unwrap(),
            authorized_motion_requirement_id: motion_requirement().requirement_id().unwrap(),
        }
    }

    fn motion_observation(sdk_int: u32) -> AndroidMotionEventObservation {
        AndroidMotionEventObservation::new(
            sdk_int,
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

    fn admitted_touch(
        profile: &PlatformIngressProfile,
        policy: &IngressAdmissionPolicy,
    ) -> (TouchEventObservation, TouchIngressReceipt) {
        profile
            .accept_touch(
                policy.input_attester_id,
                6,
                1,
                0.25,
                0.75,
                0,
                0.5,
                1234,
            )
            .unwrap()
    }

    #[test]
    fn exact_android_touch_evidence_is_joined() {
        let profile = ingress_profile(PlatformIngressKind::AndroidJni);
        let policy = ingress_policy(&profile);
        let current = current_ingress(&policy);
        let (touch, receipt) = admitted_touch(&profile, &policy);

        let certificate = certify_android_touch_join(
            &policy,
            &current,
            &profile,
            &touch,
            &receipt,
            &motion_policy_admission(),
            &attention_admission(),
            &attention_requirement(),
            &attention_observation(),
            &motion_requirement(),
            &motion_observation(34),
        )
        .unwrap();

        assert_eq!(certificate.touch_observation_id, touch.observation_id().unwrap());
        assert!(!certificate.certificate_id().unwrap().is_zero());
    }

    #[test]
    fn ios_ingress_cannot_be_laundered_as_android_touch_evidence() {
        let profile = ingress_profile(PlatformIngressKind::IosCAbi);
        let policy = ingress_policy(&profile);
        let current = current_ingress(&policy);
        let (touch, receipt) = admitted_touch(&profile, &policy);

        assert_eq!(
            certify_android_touch_join(
                &policy,
                &current,
                &profile,
                &touch,
                &receipt,
                &motion_policy_admission(),
                &attention_admission(),
                &attention_requirement(),
                &attention_observation(),
                &motion_requirement(),
                &motion_observation(34),
            ),
            Err(AndroidTouchJoinError::NotAndroidIngress)
        );
    }

    #[test]
    fn contradictory_android_sdk_evidence_is_rejected() {
        let profile = ingress_profile(PlatformIngressKind::AndroidJni);
        let policy = ingress_policy(&profile);
        let current = current_ingress(&policy);
        let (touch, receipt) = admitted_touch(&profile, &policy);

        assert_eq!(
            certify_android_touch_join(
                &policy,
                &current,
                &profile,
                &touch,
                &receipt,
                &motion_policy_admission(),
                &attention_admission(),
                &attention_requirement(),
                &attention_observation(),
                &motion_requirement(),
                &motion_observation(33),
            ),
            Err(AndroidTouchJoinError::SdkMismatch)
        );
    }

    #[test]
    fn touch_policy_surface_must_match_attention_surface() {
        let profile = ingress_profile(PlatformIngressKind::AndroidJni);
        let mut policy = ingress_policy(&profile);
        policy.trusted_surface_id = TrustedSurfaceId(d(99));
        let current = current_ingress(&policy);
        let (touch, receipt) = admitted_touch(&profile, &policy);

        assert_eq!(
            certify_android_touch_join(
                &policy,
                &current,
                &profile,
                &touch,
                &receipt,
                &motion_policy_admission(),
                &attention_admission(),
                &attention_requirement(),
                &attention_observation(),
                &motion_requirement(),
                &motion_observation(34),
            ),
            Err(AndroidTouchJoinError::TrustedSurfaceMismatch)
        );
    }

    #[test]
    fn motion_policy_identity_is_not_substitutable() {
        let profile = ingress_profile(PlatformIngressKind::AndroidJni);
        let policy = ingress_policy(&profile);
        let current = current_ingress(&policy);
        let (touch, receipt) = admitted_touch(&profile, &policy);
        let mut foreign = motion_policy_admission();
        foreign.attention_policy_admission_id = AndroidAttentionPolicyAdmissionId(d(88));

        assert!(matches!(
            certify_android_touch_join(
                &policy,
                &current,
                &profile,
                &touch,
                &receipt,
                &foreign,
                &attention_admission(),
                &attention_requirement(),
                &attention_observation(),
                &motion_requirement(),
                &motion_observation(34),
            ),
            Err(AndroidTouchJoinError::MotionPolicy(
                AndroidMotionPolicyError::AttentionPolicyAdmissionMismatch
            ))
        ));
    }

    #[test]
    fn attention_requirement_identity_is_not_implicitly_rewritten() {
        let profile = ingress_profile(PlatformIngressKind::AndroidJni);
        let policy = ingress_policy(&profile);
        let current = current_ingress(&policy);
        let (touch, receipt) = admitted_touch(&profile, &policy);
        let mut motion = motion_requirement();
        motion.attention_requirement_id = AndroidAttentionRequirementId(d(77));

        assert!(matches!(
            certify_android_touch_join(
                &policy,
                &current,
                &profile,
                &touch,
                &receipt,
                &motion_policy_admission(),
                &attention_admission(),
                &attention_requirement(),
                &attention_observation(),
                &motion,
                &motion_observation(34),
            ),
            Err(AndroidTouchJoinError::MotionPolicy(
                AndroidMotionPolicyError::MotionRequirementNotAuthorized
            ))
        ));
    }
}
