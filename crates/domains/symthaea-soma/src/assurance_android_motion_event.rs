// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDMOTION-525: bind one Android MotionEvent to current trusted-attention state.
//!
//! A clean Android attention snapshot cannot prove that a later touch event was
//! unobscured. Android supplies per-event obscuration flags, so this theorem keeps
//! them in the evidence identity and rejects events whose exact flags violate the
//! current motion-event requirement. Full and partial obscuration are separate
//! facts. Partial obscuration is only observable from API 29 onward.
//!
//! This theorem also preserves the single-pointer limitation of the existing
//! measured-touch model. It does not infer human identity from a platform event,
//! and `event_time_ms` is committed as Android uptime-based data rather than used
//! as a causal clock.

use core::fmt;

use crate::assurance_android_attention::{
    AndroidAttentionError, AndroidAttentionObservation, AndroidAttentionRequirement,
    AndroidAttentionRequirementId, CurrentAndroidAttentionPolicy,
};
use crate::assurance_soma_interaction::{
    CanonicalUnitF32, ObservedTouchAction, TouchEventObservation, TouchEventObservationId,
};

const REQUIREMENT_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-motion-requirement\0";
const OBSERVATION_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-motion-observation\0";
const CERTIFICATE_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-motion-certificate\0";

pub const ANDROID_API_PARTIAL_OBSCURATION: u32 = 29;
pub const MOTION_FLAG_WINDOW_IS_OBSCURED: u32 = 0x0000_0001;
pub const MOTION_FLAG_WINDOW_IS_PARTIALLY_OBSCURED: u32 = 0x0000_0002;

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

digest_id!(AndroidMotionEventRequirementId);
digest_id!(AndroidMotionEventObservationId);
digest_id!(AndroidMotionEventCertificateId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidMotionEventRequirement {
    pub attention_requirement_id: AndroidAttentionRequirementId,
    pub reject_fully_obscured: bool,
    pub reject_partially_obscured: bool,
    pub require_single_pointer: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidMotionEventObservation {
    pub sdk_int: u32,
    pub event_flags: u32,
    pub action: ObservedTouchAction,
    pub pointer_count: u32,
    pub x: CanonicalUnitF32,
    pub y: CanonicalUnitF32,
    pub pressure: CanonicalUnitF32,
    /// `MotionEvent.getEventTime()` uses Android's uptimeMillis time base.
    pub event_time_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidMotionEventCertificate {
    pub motion_requirement_id: AndroidMotionEventRequirementId,
    pub motion_observation_id: AndroidMotionEventObservationId,
    pub attention_requirement_id: AndroidAttentionRequirementId,
    pub attention_observation_id: crate::assurance_android_attention::AndroidAttentionObservationId,
    pub attention_certificate_id: crate::assurance_android_attention::AndroidAttentionCertificateId,
    pub touch_observation_id: TouchEventObservationId,
    pub interaction_generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidMotionEventError {
    ZeroAttentionRequirement,
    AttentionRequirementMismatch,
    Attention(AndroidAttentionError),
    ZeroSdk,
    ZeroPointerCount,
    FullyObscured,
    PartialObscurationUnavailable,
    PartiallyObscured,
    MultiplePointers,
    TouchInteractionGenerationMismatch,
    TouchActionMismatch,
    TouchXMismatch,
    TouchYMismatch,
    TouchPressureMismatch,
    TouchTimestampMismatch,
}

impl fmt::Display for AndroidMotionEventError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AndroidMotionEventError {}

impl From<AndroidAttentionError> for AndroidMotionEventError {
    fn from(value: AndroidAttentionError) -> Self {
        Self::Attention(value)
    }
}

fn hash_bool(hasher: &mut blake3::Hasher, value: bool) {
    hasher.update(&[u8::from(value)]);
}

impl AndroidMotionEventRequirement {
    pub fn validate(&self) -> Result<(), AndroidMotionEventError> {
        if self.attention_requirement_id.is_zero() {
            return Err(AndroidMotionEventError::ZeroAttentionRequirement);
        }
        Ok(())
    }

    pub fn requirement_id(
        &self,
    ) -> Result<AndroidMotionEventRequirementId, AndroidMotionEventError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(REQUIREMENT_DOMAIN);
        hasher.update(self.attention_requirement_id.as_bytes());
        hash_bool(&mut hasher, self.reject_fully_obscured);
        hash_bool(&mut hasher, self.reject_partially_obscured);
        hash_bool(&mut hasher, self.require_single_pointer);
        Ok(AndroidMotionEventRequirementId(*hasher.finalize().as_bytes()))
    }
}

impl AndroidMotionEventObservation {
    pub fn new(
        sdk_int: u32,
        event_flags: u32,
        action: ObservedTouchAction,
        pointer_count: u32,
        x: f32,
        y: f32,
        pressure: f32,
        event_time_ms: u64,
    ) -> Result<Self, AndroidMotionEventError> {
        if sdk_int == 0 {
            return Err(AndroidMotionEventError::ZeroSdk);
        }
        if pointer_count == 0 {
            return Err(AndroidMotionEventError::ZeroPointerCount);
        }
        let x = CanonicalUnitF32::from_value(x)
            .map_err(|_| AndroidMotionEventError::TouchXMismatch)?;
        let y = CanonicalUnitF32::from_value(y)
            .map_err(|_| AndroidMotionEventError::TouchYMismatch)?;
        let pressure = CanonicalUnitF32::from_value(pressure)
            .map_err(|_| AndroidMotionEventError::TouchPressureMismatch)?;
        Ok(Self {
            sdk_int,
            event_flags,
            action,
            pointer_count,
            x,
            y,
            pressure,
            event_time_ms,
        })
    }

    pub fn is_fully_obscured(&self) -> bool {
        self.event_flags & MOTION_FLAG_WINDOW_IS_OBSCURED != 0
    }

    pub fn is_partially_obscured(&self) -> bool {
        self.event_flags & MOTION_FLAG_WINDOW_IS_PARTIALLY_OBSCURED != 0
    }

    pub fn observation_id(
        &self,
    ) -> Result<AndroidMotionEventObservationId, AndroidMotionEventError> {
        if self.sdk_int == 0 {
            return Err(AndroidMotionEventError::ZeroSdk);
        }
        if self.pointer_count == 0 {
            return Err(AndroidMotionEventError::ZeroPointerCount);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(OBSERVATION_DOMAIN);
        hasher.update(&self.sdk_int.to_le_bytes());
        hasher.update(&self.event_flags.to_le_bytes());
        hasher.update(&[self.action as u8]);
        hasher.update(&self.pointer_count.to_le_bytes());
        hasher.update(&self.x.bits().to_le_bytes());
        hasher.update(&self.y.bits().to_le_bytes());
        hasher.update(&self.pressure.bits().to_le_bytes());
        hasher.update(&self.event_time_ms.to_le_bytes());
        Ok(AndroidMotionEventObservationId(*hasher.finalize().as_bytes()))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn certify_android_motion_event(
    attention_requirement: &AndroidAttentionRequirement,
    current_attention_policy: &CurrentAndroidAttentionPolicy,
    attention_observation: &AndroidAttentionObservation,
    motion_requirement: &AndroidMotionEventRequirement,
    motion_observation: &AndroidMotionEventObservation,
    touch_observation: &TouchEventObservation,
) -> Result<AndroidMotionEventCertificate, AndroidMotionEventError> {
    let attention_requirement_id = attention_requirement.requirement_id()?;
    if motion_requirement.attention_requirement_id != attention_requirement_id {
        return Err(AndroidMotionEventError::AttentionRequirementMismatch);
    }
    motion_requirement.validate()?;

    // Recompute QUAL-ANDROIDATTENTION-524 rather than trusting a supplied certificate.
    let attention_certificate = attention_requirement.evaluate(
        current_attention_policy,
        attention_observation,
    )?;
    let attention_certificate_id = attention_certificate.certificate_id()?;

    if motion_requirement.reject_fully_obscured && motion_observation.is_fully_obscured() {
        return Err(AndroidMotionEventError::FullyObscured);
    }
    if motion_requirement.reject_partially_obscured {
        if motion_observation.sdk_int < ANDROID_API_PARTIAL_OBSCURATION {
            return Err(AndroidMotionEventError::PartialObscurationUnavailable);
        }
        if motion_observation.is_partially_obscured() {
            return Err(AndroidMotionEventError::PartiallyObscured);
        }
    }
    if motion_requirement.require_single_pointer && motion_observation.pointer_count != 1 {
        return Err(AndroidMotionEventError::MultiplePointers);
    }

    if touch_observation.interaction_generation != attention_observation.interaction_generation {
        return Err(AndroidMotionEventError::TouchInteractionGenerationMismatch);
    }
    if touch_observation.action != motion_observation.action {
        return Err(AndroidMotionEventError::TouchActionMismatch);
    }
    if touch_observation.x != motion_observation.x {
        return Err(AndroidMotionEventError::TouchXMismatch);
    }
    if touch_observation.y != motion_observation.y {
        return Err(AndroidMotionEventError::TouchYMismatch);
    }
    if touch_observation.pressure != motion_observation.pressure {
        return Err(AndroidMotionEventError::TouchPressureMismatch);
    }
    if touch_observation.platform_timestamp_ms != motion_observation.event_time_ms {
        return Err(AndroidMotionEventError::TouchTimestampMismatch);
    }

    Ok(AndroidMotionEventCertificate {
        motion_requirement_id: motion_requirement.requirement_id()?,
        motion_observation_id: motion_observation.observation_id()?,
        attention_requirement_id,
        attention_observation_id: attention_observation.observation_id()?,
        attention_certificate_id,
        touch_observation_id: touch_observation.observation_id().map_err(|_| {
            AndroidMotionEventError::TouchInteractionGenerationMismatch
        })?,
        interaction_generation: attention_observation.interaction_generation,
    })
}

impl AndroidMotionEventCertificate {
    pub fn certificate_id(
        &self,
    ) -> Result<AndroidMotionEventCertificateId, AndroidMotionEventError> {
        if self.motion_requirement_id.is_zero() {
            return Err(AndroidMotionEventError::ZeroAttentionRequirement);
        }
        if self.motion_observation_id.is_zero() {
            return Err(AndroidMotionEventError::ZeroSdk);
        }
        if self.attention_requirement_id.is_zero() {
            return Err(AndroidMotionEventError::ZeroAttentionRequirement);
        }
        if self.attention_observation_id.is_zero() {
            return Err(AndroidMotionEventError::ZeroSdk);
        }
        if self.attention_certificate_id.is_zero() {
            return Err(AndroidMotionEventError::ZeroAttentionRequirement);
        }
        if self.touch_observation_id.is_zero() {
            return Err(AndroidMotionEventError::TouchInteractionGenerationMismatch);
        }
        if self.interaction_generation == 0 {
            return Err(AndroidMotionEventError::TouchInteractionGenerationMismatch);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.motion_requirement_id.as_bytes());
        hasher.update(self.motion_observation_id.as_bytes());
        hasher.update(self.attention_requirement_id.as_bytes());
        hasher.update(self.attention_observation_id.as_bytes());
        hasher.update(self.attention_certificate_id.as_bytes());
        hasher.update(self.touch_observation_id.as_bytes());
        hasher.update(&self.interaction_generation.to_le_bytes());
        Ok(AndroidMotionEventCertificateId(*hasher.finalize().as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_android_attention::{
        AndroidAttentionPolicyContextId, AndroidAttentionPolicyStateRoot,
    };
    use crate::assurance_soma_interaction::TouchInputProfileId;
    use symthaea_core::assurance_interaction_continuity::InputAttesterId;
    use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

    fn digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn attention_requirement() -> AndroidAttentionRequirement {
        AndroidAttentionRequirement {
            policy_context_id: AndroidAttentionPolicyContextId(digest(1)),
            policy_generation: 3,
            policy_state_root: AndroidAttentionPolicyStateRoot(digest(2)),
            trusted_surface_id: TrustedSurfaceId(digest(3)),
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

    fn current_policy() -> CurrentAndroidAttentionPolicy {
        CurrentAndroidAttentionPolicy {
            policy_context_id: AndroidAttentionPolicyContextId(digest(1)),
            policy_generation: 3,
            policy_state_root: AndroidAttentionPolicyStateRoot(digest(2)),
        }
    }

    fn attention_observation() -> AndroidAttentionObservation {
        AndroidAttentionObservation {
            sdk_int: 34,
            trusted_surface_id: TrustedSurfaceId(digest(3)),
            surface_generation: 4,
            interaction_generation: 5,
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

    fn motion_requirement() -> AndroidMotionEventRequirement {
        AndroidMotionEventRequirement {
            attention_requirement_id: attention_requirement().requirement_id().unwrap(),
            reject_fully_obscured: true,
            reject_partially_obscured: true,
            require_single_pointer: true,
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
            input_profile_id: TouchInputProfileId(digest(7)),
            input_attester_id: InputAttesterId(digest(8)),
            interaction_generation: 5,
            input_sequence: 1,
            x: CanonicalUnitF32::from_value(0.25).unwrap(),
            y: CanonicalUnitF32::from_value(0.75).unwrap(),
            pressure: CanonicalUnitF32::from_value(0.5).unwrap(),
            action: ObservedTouchAction::Down,
            platform_timestamp_ms: 1234,
        }
    }

    #[test]
    fn exact_unobscured_single_pointer_event_is_accepted() {
        let cert = certify_android_motion_event(
            &attention_requirement(),
            &current_policy(),
            &attention_observation(),
            &motion_requirement(),
            &motion(0),
            &touch(),
        )
        .unwrap();
        assert!(!cert.certificate_id().unwrap().is_zero());
    }

    #[test]
    fn direct_obscuration_is_rejected() {
        assert_eq!(
            certify_android_motion_event(
                &attention_requirement(),
                &current_policy(),
                &attention_observation(),
                &motion_requirement(),
                &motion(MOTION_FLAG_WINDOW_IS_OBSCURED),
                &touch(),
            ),
            Err(AndroidMotionEventError::FullyObscured)
        );
    }

    #[test]
    fn partial_obscuration_is_rejected() {
        assert_eq!(
            certify_android_motion_event(
                &attention_requirement(),
                &current_policy(),
                &attention_observation(),
                &motion_requirement(),
                &motion(MOTION_FLAG_WINDOW_IS_PARTIALLY_OBSCURED),
                &touch(),
            ),
            Err(AndroidMotionEventError::PartiallyObscured)
        );
    }

    #[test]
    fn partial_obscuration_requirement_fails_closed_before_api_29() {
        let mut older = motion(0);
        older.sdk_int = 28;
        assert_eq!(
            certify_android_motion_event(
                &attention_requirement(),
                &current_policy(),
                &attention_observation(),
                &motion_requirement(),
                &older,
                &touch(),
            ),
            Err(AndroidMotionEventError::PartialObscurationUnavailable)
        );
    }

    #[test]
    fn multi_pointer_event_is_rejected() {
        let mut multi = motion(0);
        multi.pointer_count = 2;
        assert_eq!(
            certify_android_motion_event(
                &attention_requirement(),
                &current_policy(),
                &attention_observation(),
                &motion_requirement(),
                &multi,
                &touch(),
            ),
            Err(AndroidMotionEventError::MultiplePointers)
        );
    }

    #[test]
    fn motion_and_touch_semantics_must_match() {
        let mut changed_touch = touch();
        changed_touch.platform_timestamp_ms = 1235;
        assert_eq!(
            certify_android_motion_event(
                &attention_requirement(),
                &current_policy(),
                &attention_observation(),
                &motion_requirement(),
                &motion(0),
                &changed_touch,
            ),
            Err(AndroidMotionEventError::TouchTimestampMismatch)
        );
    }
}
