// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-INGRESSPOLICY-523: current-policy admission for strict platform ingress.
//!
//! QUAL-FFIINPUT-520 establishes that an input satisfies one exact strict ingress
//! profile. QUAL-NATIVEINGRESS-521 and QUAL-ANDROIDINGRESS-522 carry that profile
//! through native/platform boundaries. None of those facts authorize a caller to
//! choose which profile is currently trusted. This module closes that gap.
//!
//! A 523 admission certificate is issued only when the active policy snapshot is
//! exact, the supplied 520 profile recomputes to the profile authorized by that
//! policy, the expected platform/surface/input-attester identities match, and the
//! exact observation/receipt relationship is independently reconstructed.

use core::fmt;

use symthaea_core::assurance_interaction_continuity::InputAttesterId;
use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

use crate::assurance_platform_ingress::{
    FrameIngressReceipt, FrameIngressReceiptId, PlatformIngressError, PlatformIngressKind,
    PlatformIngressProfile, PlatformIngressProfileId, TouchIngressReceipt, TouchIngressReceiptId,
    RGB8_CHANNELS,
};
use crate::assurance_soma_interaction::{
    ObservedTouchAction, ScreenFrameObservation, ScreenFrameObservationId, SomaInteractionError,
    TouchEventObservation, TouchEventObservationId,
};

const POLICY_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/ingress-admission-policy\0";
const FRAME_CERT_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/frame-ingress-admission\0";
const TOUCH_CERT_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/touch-ingress-admission\0";

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

digest_id!(IngressPolicyContextId);
digest_id!(IngressPolicyStateRoot);
digest_id!(IngressAdmissionPolicyId);
digest_id!(FrameIngressAdmissionCertificateId);
digest_id!(TouchIngressAdmissionCertificateId);

/// Current policy snapshot supplied by the policy plane. It is intentionally
/// small: currentness is exact equality, not a timestamp or "latest" heuristic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IngressPolicySnapshot {
    pub policy_context_id: IngressPolicyContextId,
    pub policy_generation: u64,
    pub policy_state_root: IngressPolicyStateRoot,
}

/// Exact policy-authorized ingress relationship for one platform boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IngressAdmissionPolicy {
    pub policy_context_id: IngressPolicyContextId,
    pub policy_generation: u64,
    pub policy_state_root: IngressPolicyStateRoot,
    pub platform: PlatformIngressKind,
    pub ingress_profile_id: PlatformIngressProfileId,
    pub trusted_surface_id: TrustedSurfaceId,
    pub input_attester_id: InputAttesterId,
    pub allow_frame: bool,
    pub allow_touch: bool,
}

/// Proof that one exact frame observation/receipt was admitted by the exact
/// current ingress policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameIngressAdmissionCertificate {
    pub policy_id: IngressAdmissionPolicyId,
    pub policy_state_root: IngressPolicyStateRoot,
    pub policy_generation: u64,
    pub ingress_profile_id: PlatformIngressProfileId,
    pub frame_observation_id: ScreenFrameObservationId,
    pub ingress_receipt_id: FrameIngressReceiptId,
    pub trusted_surface_id: TrustedSurfaceId,
}

/// Proof that one exact touch observation/receipt was admitted by the exact
/// current ingress policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TouchIngressAdmissionCertificate {
    pub policy_id: IngressAdmissionPolicyId,
    pub policy_state_root: IngressPolicyStateRoot,
    pub policy_generation: u64,
    pub ingress_profile_id: PlatformIngressProfileId,
    pub touch_observation_id: TouchEventObservationId,
    pub ingress_receipt_id: TouchIngressReceiptId,
    pub input_attester_id: InputAttesterId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IngressPolicyError {
    ZeroPolicyContext,
    ZeroPolicyGeneration,
    ZeroPolicyStateRoot,
    ZeroIngressProfile,
    ZeroTrustedSurface,
    ZeroInputAttester,
    NoCapabilityEnabled,
    PolicyContextMismatch,
    PolicyGenerationMismatch,
    PolicyStateRootMismatch,
    PlatformMismatch,
    IngressProfileMismatch,
    FrameNotAllowed,
    TouchNotAllowed,
    TrustedSurfaceMismatch,
    InputAttesterMismatch,
    CaptureProfileMismatch,
    TouchProfileMismatch,
    FrameObservationMismatch,
    TouchObservationMismatch,
    FrameDimensionsMismatch,
    TouchSemanticsMismatch,
    PlatformIngress(PlatformIngressError),
    Observation(SomaInteractionError),
}

impl fmt::Display for IngressPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for IngressPolicyError {}

impl From<PlatformIngressError> for IngressPolicyError {
    fn from(value: PlatformIngressError) -> Self {
        Self::PlatformIngress(value)
    }
}

impl From<SomaInteractionError> for IngressPolicyError {
    fn from(value: SomaInteractionError) -> Self {
        Self::Observation(value)
    }
}

impl IngressPolicySnapshot {
    pub fn validate(&self) -> Result<(), IngressPolicyError> {
        if self.policy_context_id.is_zero() {
            return Err(IngressPolicyError::ZeroPolicyContext);
        }
        if self.policy_generation == 0 {
            return Err(IngressPolicyError::ZeroPolicyGeneration);
        }
        if self.policy_state_root.is_zero() {
            return Err(IngressPolicyError::ZeroPolicyStateRoot);
        }
        Ok(())
    }
}

impl IngressAdmissionPolicy {
    pub fn validate(&self) -> Result<(), IngressPolicyError> {
        IngressPolicySnapshot {
            policy_context_id: self.policy_context_id,
            policy_generation: self.policy_generation,
            policy_state_root: self.policy_state_root,
        }
        .validate()?;
        if self.ingress_profile_id.is_zero() {
            return Err(IngressPolicyError::ZeroIngressProfile);
        }
        if self.trusted_surface_id.is_zero() {
            return Err(IngressPolicyError::ZeroTrustedSurface);
        }
        if self.input_attester_id.is_zero() {
            return Err(IngressPolicyError::ZeroInputAttester);
        }
        if !self.allow_frame && !self.allow_touch {
            return Err(IngressPolicyError::NoCapabilityEnabled);
        }
        Ok(())
    }

    pub fn policy_id(&self) -> Result<IngressAdmissionPolicyId, IngressPolicyError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DOMAIN);
        hasher.update(self.policy_context_id.as_bytes());
        hasher.update(&self.policy_generation.to_le_bytes());
        hasher.update(self.policy_state_root.as_bytes());
        hasher.update(&[self.platform as u8]);
        hasher.update(self.ingress_profile_id.as_bytes());
        hasher.update(self.trusted_surface_id.as_bytes());
        hasher.update(self.input_attester_id.as_bytes());
        hasher.update(&[u8::from(self.allow_frame), u8::from(self.allow_touch)]);
        Ok(IngressAdmissionPolicyId(*hasher.finalize().as_bytes()))
    }

    pub fn verify_current(
        &self,
        current: &IngressPolicySnapshot,
    ) -> Result<(), IngressPolicyError> {
        self.validate()?;
        current.validate()?;
        if current.policy_context_id != self.policy_context_id {
            return Err(IngressPolicyError::PolicyContextMismatch);
        }
        if current.policy_generation != self.policy_generation {
            return Err(IngressPolicyError::PolicyGenerationMismatch);
        }
        if current.policy_state_root != self.policy_state_root {
            return Err(IngressPolicyError::PolicyStateRootMismatch);
        }
        Ok(())
    }

    fn verify_profile(
        &self,
        profile: &PlatformIngressProfile,
    ) -> Result<PlatformIngressProfileId, IngressPolicyError> {
        profile.validate()?;
        if profile.platform != self.platform {
            return Err(IngressPolicyError::PlatformMismatch);
        }
        let profile_id = profile.profile_id()?;
        if profile_id != self.ingress_profile_id {
            return Err(IngressPolicyError::IngressProfileMismatch);
        }
        Ok(profile_id)
    }
}

impl FrameIngressAdmissionCertificate {
    pub fn admit(
        policy: &IngressAdmissionPolicy,
        current: &IngressPolicySnapshot,
        profile: &PlatformIngressProfile,
        observation: &ScreenFrameObservation,
        receipt: &FrameIngressReceipt,
    ) -> Result<Self, IngressPolicyError> {
        policy.verify_current(current)?;
        if !policy.allow_frame {
            return Err(IngressPolicyError::FrameNotAllowed);
        }
        let profile_id = policy.verify_profile(profile)?;
        observation.validate()?;
        let observation_id = observation.observation_id()?;
        if observation.surface_id != policy.trusted_surface_id {
            return Err(IngressPolicyError::TrustedSurfaceMismatch);
        }
        if observation.capture_profile_id != profile.capture_profile_id {
            return Err(IngressPolicyError::CaptureProfileMismatch);
        }
        if receipt.ingress_profile_id != profile_id {
            return Err(IngressPolicyError::IngressProfileMismatch);
        }
        if receipt.frame_observation_id != observation_id {
            return Err(IngressPolicyError::FrameObservationMismatch);
        }
        if receipt.width != observation.width
            || receipt.height != observation.height
            || receipt.channels != RGB8_CHANNELS
        {
            return Err(IngressPolicyError::FrameDimensionsMismatch);
        }
        let expected_len = profile.expected_rgb_len(
            observation.width,
            observation.height,
            RGB8_CHANNELS,
        )?;
        if receipt.expected_len != expected_len || receipt.actual_len != expected_len {
            return Err(IngressPolicyError::FrameDimensionsMismatch);
        }
        let receipt_id = receipt.receipt_id()?;
        Ok(Self {
            policy_id: policy.policy_id()?,
            policy_state_root: current.policy_state_root,
            policy_generation: current.policy_generation,
            ingress_profile_id: profile_id,
            frame_observation_id: observation_id,
            ingress_receipt_id: receipt_id,
            trusted_surface_id: observation.surface_id,
        })
    }

    pub fn certificate_id(
        &self,
    ) -> Result<FrameIngressAdmissionCertificateId, IngressPolicyError> {
        validate_common_certificate(
            self.policy_id,
            self.policy_state_root,
            self.policy_generation,
            self.ingress_profile_id,
        )?;
        if self.frame_observation_id.is_zero() {
            return Err(IngressPolicyError::FrameObservationMismatch);
        }
        if self.ingress_receipt_id.is_zero() {
            return Err(IngressPolicyError::FrameObservationMismatch);
        }
        if self.trusted_surface_id.is_zero() {
            return Err(IngressPolicyError::ZeroTrustedSurface);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(FRAME_CERT_DOMAIN);
        hasher.update(self.policy_id.as_bytes());
        hasher.update(self.policy_state_root.as_bytes());
        hasher.update(&self.policy_generation.to_le_bytes());
        hasher.update(self.ingress_profile_id.as_bytes());
        hasher.update(self.frame_observation_id.as_bytes());
        hasher.update(self.ingress_receipt_id.as_bytes());
        hasher.update(self.trusted_surface_id.as_bytes());
        Ok(FrameIngressAdmissionCertificateId(*hasher.finalize().as_bytes()))
    }
}

impl TouchIngressAdmissionCertificate {
    pub fn admit(
        policy: &IngressAdmissionPolicy,
        current: &IngressPolicySnapshot,
        profile: &PlatformIngressProfile,
        observation: &TouchEventObservation,
        receipt: &TouchIngressReceipt,
    ) -> Result<Self, IngressPolicyError> {
        policy.verify_current(current)?;
        if !policy.allow_touch {
            return Err(IngressPolicyError::TouchNotAllowed);
        }
        let profile_id = policy.verify_profile(profile)?;
        observation.validate()?;
        let observation_id = observation.observation_id()?;
        if observation.input_attester_id != policy.input_attester_id {
            return Err(IngressPolicyError::InputAttesterMismatch);
        }
        if observation.input_profile_id != profile.touch_input_profile_id {
            return Err(IngressPolicyError::TouchProfileMismatch);
        }
        if receipt.ingress_profile_id != profile_id {
            return Err(IngressPolicyError::IngressProfileMismatch);
        }
        if receipt.touch_observation_id != observation_id {
            return Err(IngressPolicyError::TouchObservationMismatch);
        }
        if receipt.action != observed_action_code(observation.action)
            || receipt.timestamp_ms != observation.platform_timestamp_ms
        {
            return Err(IngressPolicyError::TouchSemanticsMismatch);
        }
        let receipt_id = receipt.receipt_id()?;
        Ok(Self {
            policy_id: policy.policy_id()?,
            policy_state_root: current.policy_state_root,
            policy_generation: current.policy_generation,
            ingress_profile_id: profile_id,
            touch_observation_id: observation_id,
            ingress_receipt_id: receipt_id,
            input_attester_id: observation.input_attester_id,
        })
    }

    pub fn certificate_id(
        &self,
    ) -> Result<TouchIngressAdmissionCertificateId, IngressPolicyError> {
        validate_common_certificate(
            self.policy_id,
            self.policy_state_root,
            self.policy_generation,
            self.ingress_profile_id,
        )?;
        if self.touch_observation_id.is_zero() {
            return Err(IngressPolicyError::TouchObservationMismatch);
        }
        if self.ingress_receipt_id.is_zero() {
            return Err(IngressPolicyError::TouchObservationMismatch);
        }
        if self.input_attester_id.is_zero() {
            return Err(IngressPolicyError::ZeroInputAttester);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(TOUCH_CERT_DOMAIN);
        hasher.update(self.policy_id.as_bytes());
        hasher.update(self.policy_state_root.as_bytes());
        hasher.update(&self.policy_generation.to_le_bytes());
        hasher.update(self.ingress_profile_id.as_bytes());
        hasher.update(self.touch_observation_id.as_bytes());
        hasher.update(self.ingress_receipt_id.as_bytes());
        hasher.update(self.input_attester_id.as_bytes());
        Ok(TouchIngressAdmissionCertificateId(*hasher.finalize().as_bytes()))
    }
}

fn validate_common_certificate(
    policy_id: IngressAdmissionPolicyId,
    policy_state_root: IngressPolicyStateRoot,
    policy_generation: u64,
    ingress_profile_id: PlatformIngressProfileId,
) -> Result<(), IngressPolicyError> {
    if policy_id.is_zero() {
        return Err(IngressPolicyError::ZeroPolicyContext);
    }
    if policy_state_root.is_zero() {
        return Err(IngressPolicyError::ZeroPolicyStateRoot);
    }
    if policy_generation == 0 {
        return Err(IngressPolicyError::ZeroPolicyGeneration);
    }
    if ingress_profile_id.is_zero() {
        return Err(IngressPolicyError::ZeroIngressProfile);
    }
    Ok(())
}

const fn observed_action_code(action: ObservedTouchAction) -> u8 {
    match action {
        ObservedTouchAction::Down => 0,
        ObservedTouchAction::Move => 1,
        ObservedTouchAction::Up => 2,
        ObservedTouchAction::Cancel => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_platform_ingress::PlatformIngressProfile;
    use crate::assurance_soma_interaction::{ScreenCaptureProfileId, TouchInputProfileId};
    use crate::touch_body::{TouchAction, TouchEvent};

    fn bytes(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    fn profile() -> PlatformIngressProfile {
        PlatformIngressProfile {
            platform: PlatformIngressKind::AndroidJni,
            abi_version: 1,
            capture_profile_id: ScreenCaptureProfileId(bytes(1)),
            touch_input_profile_id: TouchInputProfileId(bytes(2)),
            max_frame_bytes: 1024,
        }
    }

    fn policy(profile: &PlatformIngressProfile) -> IngressAdmissionPolicy {
        IngressAdmissionPolicy {
            policy_context_id: IngressPolicyContextId(bytes(10)),
            policy_generation: 7,
            policy_state_root: IngressPolicyStateRoot(bytes(11)),
            platform: PlatformIngressKind::AndroidJni,
            ingress_profile_id: profile.profile_id().unwrap(),
            trusted_surface_id: TrustedSurfaceId(bytes(12)),
            input_attester_id: InputAttesterId(bytes(13)),
            allow_frame: true,
            allow_touch: true,
        }
    }

    fn current(policy: &IngressAdmissionPolicy) -> IngressPolicySnapshot {
        IngressPolicySnapshot {
            policy_context_id: policy.policy_context_id,
            policy_generation: policy.policy_generation,
            policy_state_root: policy.policy_state_root,
        }
    }

    #[test]
    fn current_policy_admits_exact_frame_receipt() {
        let profile = profile();
        let policy = policy(&profile);
        let frame = vec![9_u8; 12];
        let (observation, receipt) = profile
            .accept_frame(policy.trusted_surface_id, 3, 4, 0, 2, 2, 3, &frame)
            .unwrap();
        let cert = FrameIngressAdmissionCertificate::admit(
            &policy,
            &current(&policy),
            &profile,
            &observation,
            &receipt,
        )
        .unwrap();
        assert!(!cert.certificate_id().unwrap().is_zero());
    }

    #[test]
    fn caller_selected_profile_is_rejected() {
        let profile = profile();
        let policy = policy(&profile);
        let mut caller_profile = profile;
        caller_profile.max_frame_bytes = 2048;
        let frame = vec![9_u8; 12];
        let (observation, receipt) = caller_profile
            .accept_frame(policy.trusted_surface_id, 3, 4, 0, 2, 2, 3, &frame)
            .unwrap();
        assert_eq!(
            FrameIngressAdmissionCertificate::admit(
                &policy,
                &current(&policy),
                &caller_profile,
                &observation,
                &receipt,
            )
            .unwrap_err(),
            IngressPolicyError::IngressProfileMismatch
        );
    }

    #[test]
    fn stale_policy_root_is_rejected() {
        let profile = profile();
        let policy = policy(&profile);
        let frame = vec![9_u8; 12];
        let (observation, receipt) = profile
            .accept_frame(policy.trusted_surface_id, 3, 4, 0, 2, 2, 3, &frame)
            .unwrap();
        let mut stale = current(&policy);
        stale.policy_state_root = IngressPolicyStateRoot(bytes(99));
        assert_eq!(
            FrameIngressAdmissionCertificate::admit(
                &policy,
                &stale,
                &profile,
                &observation,
                &receipt,
            )
            .unwrap_err(),
            IngressPolicyError::PolicyStateRootMismatch
        );
    }

    #[test]
    fn wrong_surface_is_rejected() {
        let profile = profile();
        let policy = policy(&profile);
        let frame = vec![9_u8; 12];
        let (observation, receipt) = profile
            .accept_frame(TrustedSurfaceId(bytes(90)), 3, 4, 0, 2, 2, 3, &frame)
            .unwrap();
        assert_eq!(
            FrameIngressAdmissionCertificate::admit(
                &policy,
                &current(&policy),
                &profile,
                &observation,
                &receipt,
            )
            .unwrap_err(),
            IngressPolicyError::TrustedSurfaceMismatch
        );
    }

    #[test]
    fn current_policy_admits_exact_touch_receipt() {
        let profile = profile();
        let policy = policy(&profile);
        let event = TouchEvent {
            x: 0.25,
            y: 0.75,
            action: TouchAction::Up,
            pressure: 0.5,
            timestamp_ms: 123,
        };
        let observation = TouchEventObservation::from_touch_event(
            profile.touch_input_profile_id,
            policy.input_attester_id,
            4,
            5,
            &event,
        )
        .unwrap();
        let receipt = TouchIngressReceipt {
            ingress_profile_id: profile.profile_id().unwrap(),
            touch_observation_id: observation.observation_id().unwrap(),
            action: 2,
            timestamp_ms: 123,
        };
        let cert = TouchIngressAdmissionCertificate::admit(
            &policy,
            &current(&policy),
            &profile,
            &observation,
            &receipt,
        )
        .unwrap();
        assert!(!cert.certificate_id().unwrap().is_zero());
    }

    #[test]
    fn touch_receipt_semantic_substitution_is_rejected() {
        let profile = profile();
        let policy = policy(&profile);
        let event = TouchEvent {
            x: 0.25,
            y: 0.75,
            action: TouchAction::Up,
            pressure: 0.5,
            timestamp_ms: 123,
        };
        let observation = TouchEventObservation::from_touch_event(
            profile.touch_input_profile_id,
            policy.input_attester_id,
            4,
            5,
            &event,
        )
        .unwrap();
        let receipt = TouchIngressReceipt {
            ingress_profile_id: profile.profile_id().unwrap(),
            touch_observation_id: observation.observation_id().unwrap(),
            action: 0,
            timestamp_ms: 123,
        };
        assert_eq!(
            TouchIngressAdmissionCertificate::admit(
                &policy,
                &current(&policy),
                &profile,
                &observation,
                &receipt,
            )
            .unwrap_err(),
            IngressPolicyError::TouchSemanticsMismatch
        );
    }
}
