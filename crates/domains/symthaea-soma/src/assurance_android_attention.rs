// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDATTENTION-524: policy-bound Android trusted-attention state.
//!
//! Android does not expose one trustworthy "secure attention" boolean. This
//! module therefore keeps independently observable protections independent:
//! window focus, secure-window state, non-system application-overlay hiding,
//! obscured-touch filtering, view attachment/visibility, top-resumed state, and
//! multi-window state.
//!
//! A certificate proves only that one exact observation satisfies one exact,
//! current attention requirement. It does **not** prove that no trusted/system
//! overlay exists, that Android or the kernel is uncompromised, that framebuffer
//! capture equals physical display output, or that a human perceived/comprehended
//! the presentation.

use core::fmt;

use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

const REQUIREMENT_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-attention-requirement\0";
const OBSERVATION_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-attention-observation\0";
const CERTIFICATE_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-attention-certificate\0";

pub const ANDROID_API_TOP_RESUMED: u32 = 29;
pub const ANDROID_API_HIDE_OVERLAY_WINDOWS: u32 = 31;

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

digest_id!(AndroidAttentionPolicyContextId);
digest_id!(AndroidAttentionPolicyStateRoot);
digest_id!(AndroidAttentionRequirementId);
digest_id!(AndroidAttentionObservationId);
digest_id!(AndroidAttentionCertificateId);

/// Current policy requirements for one exact trusted Android surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidAttentionRequirement {
    pub policy_context_id: AndroidAttentionPolicyContextId,
    pub policy_generation: u64,
    pub policy_state_root: AndroidAttentionPolicyStateRoot,
    pub trusted_surface_id: TrustedSurfaceId,
    pub minimum_sdk_int: u32,
    pub require_window_focus: bool,
    pub require_flag_secure: bool,
    pub require_hide_application_overlays: bool,
    pub require_filter_touches_when_obscured: bool,
    pub require_view_attached: bool,
    pub require_view_shown: bool,
    pub require_top_resumed: bool,
    pub forbid_multi_window: bool,
}

/// The exact current policy snapshot supplied by the policy plane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CurrentAndroidAttentionPolicy {
    pub policy_context_id: AndroidAttentionPolicyContextId,
    pub policy_generation: u64,
    pub policy_state_root: AndroidAttentionPolicyStateRoot,
}

/// Platform facts observed for one Android interaction surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidAttentionObservation {
    pub sdk_int: u32,
    pub trusted_surface_id: TrustedSurfaceId,
    pub surface_generation: u64,
    pub interaction_generation: u64,
    pub window_has_focus: bool,
    pub flag_secure_set: bool,
    /// True only when the application successfully invoked Android 12+'s
    /// `Window.setHideOverlayWindows(true)` through the qualified adapter.
    /// This is a request/state-of-adapter fact, not proof that all overlays are absent.
    pub hide_application_overlays_requested: bool,
    pub filter_touches_when_obscured_enabled: bool,
    pub view_attached_to_window: bool,
    pub view_shown: bool,
    /// Meaningful only on API >= 29. On older Android versions this is false and
    /// policies that require it fail with `TopResumedUnavailable`.
    pub activity_top_resumed: bool,
    pub activity_in_multi_window_mode: bool,
}

/// Evidence that the exact observation satisfied the exact current policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidAttentionCertificate {
    pub requirement_id: AndroidAttentionRequirementId,
    pub observation_id: AndroidAttentionObservationId,
    pub policy_context_id: AndroidAttentionPolicyContextId,
    pub policy_generation: u64,
    pub policy_state_root: AndroidAttentionPolicyStateRoot,
    pub trusted_surface_id: TrustedSurfaceId,
    pub surface_generation: u64,
    pub interaction_generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidAttentionError {
    ZeroPolicyContext,
    ZeroPolicyGeneration,
    ZeroPolicyStateRoot,
    ZeroTrustedSurface,
    ZeroMinimumSdk,
    ZeroSurfaceGeneration,
    ZeroInteractionGeneration,
    PolicyContextMismatch,
    PolicyGenerationMismatch,
    PolicyStateRootMismatch,
    TrustedSurfaceMismatch,
    SdkTooOld,
    WindowNotFocused,
    SecureWindowRequired,
    OverlayHidingUnavailable,
    OverlayHidingNotRequested,
    ObscuredTouchFilteringRequired,
    ViewNotAttached,
    ViewNotShown,
    TopResumedUnavailable,
    ActivityNotTopResumed,
    MultiWindowForbidden,
}

impl fmt::Display for AndroidAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AndroidAttentionError {}

fn hash_bool(hasher: &mut blake3::Hasher, value: bool) {
    hasher.update(&[u8::from(value)]);
}

impl AndroidAttentionRequirement {
    pub fn validate(&self) -> Result<(), AndroidAttentionError> {
        if self.policy_context_id.is_zero() {
            return Err(AndroidAttentionError::ZeroPolicyContext);
        }
        if self.policy_generation == 0 {
            return Err(AndroidAttentionError::ZeroPolicyGeneration);
        }
        if self.policy_state_root.is_zero() {
            return Err(AndroidAttentionError::ZeroPolicyStateRoot);
        }
        if self.trusted_surface_id.is_zero() {
            return Err(AndroidAttentionError::ZeroTrustedSurface);
        }
        if self.minimum_sdk_int == 0 {
            return Err(AndroidAttentionError::ZeroMinimumSdk);
        }
        Ok(())
    }

    pub fn requirement_id(&self) -> Result<AndroidAttentionRequirementId, AndroidAttentionError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(REQUIREMENT_DOMAIN);
        hasher.update(self.policy_context_id.as_bytes());
        hasher.update(&self.policy_generation.to_le_bytes());
        hasher.update(self.policy_state_root.as_bytes());
        hasher.update(self.trusted_surface_id.as_bytes());
        hasher.update(&self.minimum_sdk_int.to_le_bytes());
        hash_bool(&mut hasher, self.require_window_focus);
        hash_bool(&mut hasher, self.require_flag_secure);
        hash_bool(&mut hasher, self.require_hide_application_overlays);
        hash_bool(&mut hasher, self.require_filter_touches_when_obscured);
        hash_bool(&mut hasher, self.require_view_attached);
        hash_bool(&mut hasher, self.require_view_shown);
        hash_bool(&mut hasher, self.require_top_resumed);
        hash_bool(&mut hasher, self.forbid_multi_window);
        Ok(AndroidAttentionRequirementId(*hasher.finalize().as_bytes()))
    }

    pub fn evaluate(
        &self,
        current_policy: &CurrentAndroidAttentionPolicy,
        observation: &AndroidAttentionObservation,
    ) -> Result<AndroidAttentionCertificate, AndroidAttentionError> {
        self.validate()?;
        current_policy.validate()?;
        observation.validate()?;

        if current_policy.policy_context_id != self.policy_context_id {
            return Err(AndroidAttentionError::PolicyContextMismatch);
        }
        if current_policy.policy_generation != self.policy_generation {
            return Err(AndroidAttentionError::PolicyGenerationMismatch);
        }
        if current_policy.policy_state_root != self.policy_state_root {
            return Err(AndroidAttentionError::PolicyStateRootMismatch);
        }
        if observation.trusted_surface_id != self.trusted_surface_id {
            return Err(AndroidAttentionError::TrustedSurfaceMismatch);
        }
        if observation.sdk_int < self.minimum_sdk_int {
            return Err(AndroidAttentionError::SdkTooOld);
        }
        if self.require_window_focus && !observation.window_has_focus {
            return Err(AndroidAttentionError::WindowNotFocused);
        }
        if self.require_flag_secure && !observation.flag_secure_set {
            return Err(AndroidAttentionError::SecureWindowRequired);
        }
        if self.require_hide_application_overlays {
            if observation.sdk_int < ANDROID_API_HIDE_OVERLAY_WINDOWS {
                return Err(AndroidAttentionError::OverlayHidingUnavailable);
            }
            if !observation.hide_application_overlays_requested {
                return Err(AndroidAttentionError::OverlayHidingNotRequested);
            }
        }
        if self.require_filter_touches_when_obscured
            && !observation.filter_touches_when_obscured_enabled
        {
            return Err(AndroidAttentionError::ObscuredTouchFilteringRequired);
        }
        if self.require_view_attached && !observation.view_attached_to_window {
            return Err(AndroidAttentionError::ViewNotAttached);
        }
        if self.require_view_shown && !observation.view_shown {
            return Err(AndroidAttentionError::ViewNotShown);
        }
        if self.require_top_resumed {
            if observation.sdk_int < ANDROID_API_TOP_RESUMED {
                return Err(AndroidAttentionError::TopResumedUnavailable);
            }
            if !observation.activity_top_resumed {
                return Err(AndroidAttentionError::ActivityNotTopResumed);
            }
        }
        if self.forbid_multi_window && observation.activity_in_multi_window_mode {
            return Err(AndroidAttentionError::MultiWindowForbidden);
        }

        Ok(AndroidAttentionCertificate {
            requirement_id: self.requirement_id()?,
            observation_id: observation.observation_id()?,
            policy_context_id: self.policy_context_id,
            policy_generation: self.policy_generation,
            policy_state_root: self.policy_state_root,
            trusted_surface_id: observation.trusted_surface_id,
            surface_generation: observation.surface_generation,
            interaction_generation: observation.interaction_generation,
        })
    }
}

impl CurrentAndroidAttentionPolicy {
    pub fn validate(&self) -> Result<(), AndroidAttentionError> {
        if self.policy_context_id.is_zero() {
            return Err(AndroidAttentionError::ZeroPolicyContext);
        }
        if self.policy_generation == 0 {
            return Err(AndroidAttentionError::ZeroPolicyGeneration);
        }
        if self.policy_state_root.is_zero() {
            return Err(AndroidAttentionError::ZeroPolicyStateRoot);
        }
        Ok(())
    }
}

impl AndroidAttentionObservation {
    pub fn validate(&self) -> Result<(), AndroidAttentionError> {
        if self.sdk_int == 0 {
            return Err(AndroidAttentionError::SdkTooOld);
        }
        if self.trusted_surface_id.is_zero() {
            return Err(AndroidAttentionError::ZeroTrustedSurface);
        }
        if self.surface_generation == 0 {
            return Err(AndroidAttentionError::ZeroSurfaceGeneration);
        }
        if self.interaction_generation == 0 {
            return Err(AndroidAttentionError::ZeroInteractionGeneration);
        }
        Ok(())
    }

    pub fn observation_id(&self) -> Result<AndroidAttentionObservationId, AndroidAttentionError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(OBSERVATION_DOMAIN);
        hasher.update(&self.sdk_int.to_le_bytes());
        hasher.update(self.trusted_surface_id.as_bytes());
        hasher.update(&self.surface_generation.to_le_bytes());
        hasher.update(&self.interaction_generation.to_le_bytes());
        hash_bool(&mut hasher, self.window_has_focus);
        hash_bool(&mut hasher, self.flag_secure_set);
        hash_bool(&mut hasher, self.hide_application_overlays_requested);
        hash_bool(&mut hasher, self.filter_touches_when_obscured_enabled);
        hash_bool(&mut hasher, self.view_attached_to_window);
        hash_bool(&mut hasher, self.view_shown);
        hash_bool(&mut hasher, self.activity_top_resumed);
        hash_bool(&mut hasher, self.activity_in_multi_window_mode);
        Ok(AndroidAttentionObservationId(*hasher.finalize().as_bytes()))
    }
}

impl AndroidAttentionCertificate {
    pub fn certificate_id(
        &self,
    ) -> Result<AndroidAttentionCertificateId, AndroidAttentionError> {
        if self.requirement_id.is_zero() {
            return Err(AndroidAttentionError::ZeroPolicyContext);
        }
        if self.observation_id.is_zero() {
            return Err(AndroidAttentionError::ZeroPolicyStateRoot);
        }
        if self.policy_context_id.is_zero() {
            return Err(AndroidAttentionError::ZeroPolicyContext);
        }
        if self.policy_generation == 0 {
            return Err(AndroidAttentionError::ZeroPolicyGeneration);
        }
        if self.policy_state_root.is_zero() {
            return Err(AndroidAttentionError::ZeroPolicyStateRoot);
        }
        if self.trusted_surface_id.is_zero() {
            return Err(AndroidAttentionError::ZeroTrustedSurface);
        }
        if self.surface_generation == 0 {
            return Err(AndroidAttentionError::ZeroSurfaceGeneration);
        }
        if self.interaction_generation == 0 {
            return Err(AndroidAttentionError::ZeroInteractionGeneration);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.requirement_id.as_bytes());
        hasher.update(self.observation_id.as_bytes());
        hasher.update(self.policy_context_id.as_bytes());
        hasher.update(&self.policy_generation.to_le_bytes());
        hasher.update(self.policy_state_root.as_bytes());
        hasher.update(self.trusted_surface_id.as_bytes());
        hasher.update(&self.surface_generation.to_le_bytes());
        hasher.update(&self.interaction_generation.to_le_bytes());
        Ok(AndroidAttentionCertificateId(*hasher.finalize().as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn requirement() -> AndroidAttentionRequirement {
        AndroidAttentionRequirement {
            policy_context_id: AndroidAttentionPolicyContextId(digest(1)),
            policy_generation: 7,
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

    fn current() -> CurrentAndroidAttentionPolicy {
        CurrentAndroidAttentionPolicy {
            policy_context_id: AndroidAttentionPolicyContextId(digest(1)),
            policy_generation: 7,
            policy_state_root: AndroidAttentionPolicyStateRoot(digest(2)),
        }
    }

    fn observation() -> AndroidAttentionObservation {
        AndroidAttentionObservation {
            sdk_int: 34,
            trusted_surface_id: TrustedSurfaceId(digest(3)),
            surface_generation: 9,
            interaction_generation: 10,
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

    #[test]
    fn exact_attention_state_is_accepted() {
        let cert = requirement().evaluate(&current(), &observation()).unwrap();
        assert!(!cert.certificate_id().unwrap().is_zero());
    }

    #[test]
    fn stale_policy_root_is_rejected() {
        let mut stale = current();
        stale.policy_state_root = AndroidAttentionPolicyStateRoot(digest(99));
        assert_eq!(
            requirement().evaluate(&stale, &observation()),
            Err(AndroidAttentionError::PolicyStateRootMismatch)
        );
    }

    #[test]
    fn overlay_hiding_requirement_fails_before_api_31() {
        let mut observed = observation();
        observed.sdk_int = 30;
        assert_eq!(
            requirement().evaluate(&current(), &observed),
            Err(AndroidAttentionError::SdkTooOld)
        );

        let mut req = requirement();
        req.minimum_sdk_int = 24;
        assert_eq!(
            req.evaluate(&current(), &observed),
            Err(AndroidAttentionError::OverlayHidingUnavailable)
        );
    }

    #[test]
    fn losing_focus_invalidates_attention() {
        let mut observed = observation();
        observed.window_has_focus = false;
        assert_eq!(
            requirement().evaluate(&current(), &observed),
            Err(AndroidAttentionError::WindowNotFocused)
        );
    }

    #[test]
    fn partial_platform_protection_does_not_count_as_full_requirement() {
        let mut observed = observation();
        observed.hide_application_overlays_requested = false;
        assert_eq!(
            requirement().evaluate(&current(), &observed),
            Err(AndroidAttentionError::OverlayHidingNotRequested)
        );
    }

    #[test]
    fn multi_window_can_be_forbidden_independently() {
        let mut observed = observation();
        observed.activity_in_multi_window_mode = true;
        assert_eq!(
            requirement().evaluate(&current(), &observed),
            Err(AndroidAttentionError::MultiWindowForbidden)
        );
    }

    #[test]
    fn observation_identity_changes_with_any_material_attention_fact() {
        let base = observation().observation_id().unwrap();
        let mut changed = observation();
        changed.filter_touches_when_obscured_enabled = false;
        assert_ne!(base, changed.observation_id().unwrap());
    }
}
