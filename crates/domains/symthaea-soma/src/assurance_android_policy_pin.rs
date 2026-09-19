// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDPOLICYPIN-527: pin one exact admitted Android trusted-touch policy
//! inside Rust before untrusted event/JNI data is evaluated.
//!
//! QUAL-ANDROIDATTENTIONPOLICY-526 deliberately treats the admission object as
//! prerequisite policy evidence; it does not authenticate or activate the higher
//! policy plane. This tranche does not change that boundary. Instead it closes a
//! different runtime gap: after trusted native code supplies an admitted policy,
//! event-side code must not be able to substitute a weaker attention requirement,
//! a different motion requirement, or a different policy context/generation/root.
//!
//! The pin owns those immutable values with private fields. Every event is checked
//! by recomputing both 526 and 525 from raw observations. No caller-supplied
//! certificate is accepted as proof.

use core::fmt;

use crate::assurance_android_attention::{
    AndroidAttentionError, AndroidAttentionObservation, AndroidAttentionPolicyContextId,
    AndroidAttentionPolicyStateRoot, AndroidAttentionRequirement,
    AndroidAttentionRequirementId, CurrentAndroidAttentionPolicy,
};
use crate::assurance_android_attention_policy::{
    AndroidAttentionPolicyAdmission, AndroidAttentionPolicyAdmissionId,
    AndroidAttentionPolicyCertificateId, AndroidAttentionPolicyError,
};
use crate::assurance_android_motion_event::{
    certify_android_motion_event, AndroidMotionEventCertificateId, AndroidMotionEventError,
    AndroidMotionEventObservation, AndroidMotionEventRequirement,
    AndroidMotionEventRequirementId,
};
use crate::assurance_soma_interaction::TouchEventObservation;

const PIN_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/android-policy-pin\0";
const CERTIFICATE_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-policy-pin-certificate\0";

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

digest_id!(AndroidTrustedTouchPolicyPinId);
digest_id!(AndroidTrustedTouchCertificateId);

/// Immutable Rust-owned policy selected before untrusted event data is handled.
///
/// Construction proves internal consistency of the supplied prerequisite policy
/// evidence. It does **not** prove that the admission itself was authenticated or
/// activated by the higher policy plane; that remains an explicit upstream
/// prerequisite inherited from 526.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AndroidTrustedTouchPolicyPin {
    admission: AndroidAttentionPolicyAdmission,
    attention_requirement: AndroidAttentionRequirement,
    motion_requirement: AndroidMotionEventRequirement,
    admission_id: AndroidAttentionPolicyAdmissionId,
    attention_requirement_id: AndroidAttentionRequirementId,
    motion_requirement_id: AndroidMotionEventRequirementId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTrustedTouchCertificate {
    pub policy_pin_id: AndroidTrustedTouchPolicyPinId,
    pub admission_id: AndroidAttentionPolicyAdmissionId,
    pub policy_certificate_id: AndroidAttentionPolicyCertificateId,
    pub motion_certificate_id: AndroidMotionEventCertificateId,
    pub interaction_generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidTrustedTouchError {
    Attention(AndroidAttentionError),
    Policy(AndroidAttentionPolicyError),
    Motion(AndroidMotionEventError),
    PolicyContextMismatch,
    PolicyGenerationMismatch,
    PolicyStateRootMismatch,
    RequirementNotAdmitted,
    MotionRequirementMismatch,
    RecomputedAttentionMismatch,
    ZeroCertificateIdentity,
    ZeroInteractionGeneration,
}

impl fmt::Display for AndroidTrustedTouchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AndroidTrustedTouchError {}

impl From<AndroidAttentionError> for AndroidTrustedTouchError {
    fn from(value: AndroidAttentionError) -> Self {
        Self::Attention(value)
    }
}

impl From<AndroidAttentionPolicyError> for AndroidTrustedTouchError {
    fn from(value: AndroidAttentionPolicyError) -> Self {
        Self::Policy(value)
    }
}

impl From<AndroidMotionEventError> for AndroidTrustedTouchError {
    fn from(value: AndroidMotionEventError) -> Self {
        Self::Motion(value)
    }
}

impl AndroidTrustedTouchPolicyPin {
    /// Freeze an exact 526 admission + 524 requirement + 525 motion requirement.
    ///
    /// Policy identity is checked before requirement identity so stale policy
    /// generation/root failures remain distinguishable from weaker-requirement
    /// substitution attempts.
    pub fn new(
        admission: AndroidAttentionPolicyAdmission,
        attention_requirement: AndroidAttentionRequirement,
        motion_requirement: AndroidMotionEventRequirement,
    ) -> Result<Self, AndroidTrustedTouchError> {
        admission.validate()?;
        attention_requirement.validate()?;
        motion_requirement.validate()?;

        if admission.policy_context_id != attention_requirement.policy_context_id {
            return Err(AndroidTrustedTouchError::PolicyContextMismatch);
        }
        if admission.policy_generation != attention_requirement.policy_generation {
            return Err(AndroidTrustedTouchError::PolicyGenerationMismatch);
        }
        if admission.policy_state_root != attention_requirement.policy_state_root {
            return Err(AndroidTrustedTouchError::PolicyStateRootMismatch);
        }

        let admission_id = admission.admission_id()?;
        let attention_requirement_id = attention_requirement.requirement_id()?;
        if admission.authorized_requirement_id != attention_requirement_id {
            return Err(AndroidTrustedTouchError::RequirementNotAdmitted);
        }
        if motion_requirement.attention_requirement_id != attention_requirement_id {
            return Err(AndroidTrustedTouchError::MotionRequirementMismatch);
        }
        let motion_requirement_id = motion_requirement.requirement_id()?;

        Ok(Self {
            admission,
            attention_requirement,
            motion_requirement,
            admission_id,
            attention_requirement_id,
            motion_requirement_id,
        })
    }

    pub const fn admission_id(&self) -> AndroidAttentionPolicyAdmissionId {
        self.admission_id
    }

    pub const fn attention_requirement_id(&self) -> AndroidAttentionRequirementId {
        self.attention_requirement_id
    }

    pub const fn motion_requirement_id(&self) -> AndroidMotionEventRequirementId {
        self.motion_requirement_id
    }

    pub const fn policy_context_id(&self) -> AndroidAttentionPolicyContextId {
        self.admission.policy_context_id
    }

    pub const fn policy_generation(&self) -> u64 {
        self.admission.policy_generation
    }

    pub const fn policy_state_root(&self) -> AndroidAttentionPolicyStateRoot {
        self.admission.policy_state_root
    }

    pub fn pin_id(&self) -> Result<AndroidTrustedTouchPolicyPinId, AndroidTrustedTouchError> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PIN_DOMAIN);
        hasher.update(self.admission_id.as_bytes());
        hasher.update(self.attention_requirement_id.as_bytes());
        hasher.update(self.motion_requirement_id.as_bytes());
        Ok(AndroidTrustedTouchPolicyPinId(*hasher.finalize().as_bytes()))
    }

    /// Recompute 526 and 525 against this immutable pin.
    ///
    /// `attention_observation`, `motion_observation`, and `touch_observation` are
    /// treated as untrusted event evidence. They may be rejected, but they cannot
    /// select or alter the pinned policy.
    pub fn certify_event(
        &self,
        attention_observation: &AndroidAttentionObservation,
        motion_observation: &AndroidMotionEventObservation,
        touch_observation: &TouchEventObservation,
    ) -> Result<AndroidTrustedTouchCertificate, AndroidTrustedTouchError> {
        let policy_certificate = self
            .admission
            .certify(&self.attention_requirement, attention_observation)?;
        let policy_certificate_id = policy_certificate.certificate_id()?;

        let current_policy = CurrentAndroidAttentionPolicy {
            policy_context_id: self.admission.policy_context_id,
            policy_generation: self.admission.policy_generation,
            policy_state_root: self.admission.policy_state_root,
        };
        let motion_certificate = certify_android_motion_event(
            &self.attention_requirement,
            &current_policy,
            attention_observation,
            &self.motion_requirement,
            motion_observation,
            touch_observation,
        )?;
        let motion_certificate_id = motion_certificate.certificate_id()?;

        // Both theorems independently recompute QUAL-ANDROIDATTENTION-524. They
        // must converge on the same exact observation/certificate identities.
        if policy_certificate.attention_certificate_id
            != motion_certificate.attention_certificate_id
            || policy_certificate.observation_id != motion_certificate.attention_observation_id
            || policy_certificate.requirement_id != motion_certificate.attention_requirement_id
        {
            return Err(AndroidTrustedTouchError::RecomputedAttentionMismatch);
        }
        if motion_certificate.interaction_generation == 0 {
            return Err(AndroidTrustedTouchError::ZeroInteractionGeneration);
        }

        Ok(AndroidTrustedTouchCertificate {
            policy_pin_id: self.pin_id()?,
            admission_id: self.admission_id,
            policy_certificate_id,
            motion_certificate_id,
            interaction_generation: motion_certificate.interaction_generation,
        })
    }
}

impl AndroidTrustedTouchCertificate {
    pub fn certificate_id(
        &self,
    ) -> Result<AndroidTrustedTouchCertificateId, AndroidTrustedTouchError> {
        if self.policy_pin_id.is_zero()
            || self.admission_id.is_zero()
            || self.policy_certificate_id.is_zero()
            || self.motion_certificate_id.is_zero()
        {
            return Err(AndroidTrustedTouchError::ZeroCertificateIdentity);
        }
        if self.interaction_generation == 0 {
            return Err(AndroidTrustedTouchError::ZeroInteractionGeneration);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.policy_pin_id.as_bytes());
        hasher.update(self.admission_id.as_bytes());
        hasher.update(self.policy_certificate_id.as_bytes());
        hasher.update(self.motion_certificate_id.as_bytes());
        hasher.update(&self.interaction_generation.to_le_bytes());
        Ok(AndroidTrustedTouchCertificateId(*hasher.finalize().as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_android_attention::{
        AndroidAttentionObservation, AndroidAttentionPolicyContextId,
        AndroidAttentionPolicyStateRoot,
    };
    use crate::assurance_android_motion_event::{
        AndroidMotionEventObservation, MOTION_FLAG_WINDOW_IS_PARTIALLY_OBSCURED,
    };
    use crate::assurance_soma_interaction::{
        CanonicalUnitF32, ObservedTouchAction, TouchEventObservation, TouchInputProfileId,
    };
    use symthaea_core::assurance_interaction_continuity::InputAttesterId;
    use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

    fn d(v: u8) -> [u8; 32] {
        [v; 32]
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

    fn admission() -> AndroidAttentionPolicyAdmission {
        let requirement = attention_requirement();
        AndroidAttentionPolicyAdmission {
            policy_context_id: requirement.policy_context_id,
            policy_generation: requirement.policy_generation,
            policy_state_root: requirement.policy_state_root,
            authorized_requirement_id: requirement.requirement_id().unwrap(),
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

    fn motion(flags: u32) -> AndroidMotionEventObservation {
        AndroidMotionEventObservation::new(
            34,
            flags,
            ObservedTouchAction::Down,
            1,
            0.25,
            0.75,
            0.5,
            1234,
        )
        .unwrap()
    }

    fn touch() -> TouchEventObservation {
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

    fn pin() -> AndroidTrustedTouchPolicyPin {
        AndroidTrustedTouchPolicyPin::new(
            admission(),
            attention_requirement(),
            motion_requirement(),
        )
        .unwrap()
    }

    #[test]
    fn exact_event_recomputes_both_upstream_theorems() {
        let certificate = pin()
            .certify_event(&attention_observation(), &motion(0), &touch())
            .unwrap();
        assert!(!certificate.certificate_id().unwrap().is_zero());
    }

    #[test]
    fn weaker_attention_requirement_cannot_be_pinned() {
        let mut weaker = attention_requirement();
        weaker.require_top_resumed = false;
        assert_eq!(
            AndroidTrustedTouchPolicyPin::new(admission(), weaker, motion_requirement()),
            Err(AndroidTrustedTouchError::RequirementNotAdmitted)
        );
    }

    #[test]
    fn stale_policy_generation_cannot_be_pinned() {
        let mut stale = attention_requirement();
        stale.policy_generation += 1;
        assert_eq!(
            AndroidTrustedTouchPolicyPin::new(admission(), stale, motion_requirement()),
            Err(AndroidTrustedTouchError::PolicyGenerationMismatch)
        );
    }

    #[test]
    fn motion_requirement_cannot_target_another_attention_requirement() {
        let mut other_motion = motion_requirement();
        other_motion.attention_requirement_id = AndroidAttentionRequirementId(d(9));
        assert_eq!(
            AndroidTrustedTouchPolicyPin::new(admission(), attention_requirement(), other_motion),
            Err(AndroidTrustedTouchError::MotionRequirementMismatch)
        );
    }

    #[test]
    fn partially_obscured_event_is_rejected_by_pinned_policy() {
        assert!(matches!(
            pin().certify_event(
                &attention_observation(),
                &motion(MOTION_FLAG_WINDOW_IS_PARTIALLY_OBSCURED),
                &touch(),
            ),
            Err(AndroidTrustedTouchError::Motion(
                AndroidMotionEventError::PartiallyObscured
            ))
        ));
    }

    #[test]
    fn touch_substitution_is_rejected_by_pinned_policy() {
        let mut changed_touch = touch();
        changed_touch.platform_timestamp_ms += 1;
        assert!(matches!(
            pin().certify_event(&attention_observation(), &motion(0), &changed_touch),
            Err(AndroidTrustedTouchError::Motion(
                AndroidMotionEventError::TouchTimestampMismatch
            ))
        ));
    }
}
