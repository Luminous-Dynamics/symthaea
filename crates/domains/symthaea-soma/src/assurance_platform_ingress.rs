// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-FFIINPUT-520: strict platform ingress contract for measured Soma interaction.
//!
//! The legacy native FFI accepts convenience inputs that may be normalized before
//! reaching Soma (for example clamped touch values or unknown touch actions mapped
//! to Cancel). That is useful for application compatibility but unsuitable as an
//! assurance boundary. This module defines the stricter accepted language for
//! assurance-bearing framebuffer/touch ingress.
//!
//! The theorem is intentionally local: an accepted receipt proves that the exact
//! supplied Rust slice / touch values satisfy the qualified ingress profile and
//! produce the exact QUAL-SOMAINTERACTION-517 observation identity. It does not
//! establish JNI/C pointer validity, OS/compositor integrity, physical display
//! fidelity, or human origin of an input event. A native adapter must call this
//! contract (or independently prove equivalent semantics) before claiming 520.

use core::fmt;

use symthaea_core::assurance_interaction_continuity::InputAttesterId;
use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

use crate::assurance_soma_interaction::{
    ScreenCaptureProfileId, ScreenFrameObservation, ScreenFrameObservationId,
    SomaInteractionError, TouchEventObservation, TouchEventObservationId, TouchInputProfileId,
};
use crate::touch_body::{TouchAction, TouchEvent};

const PROFILE_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/platform-ingress-profile\0";
const FRAME_RECEIPT_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/frame-ingress-receipt\0";
const TOUCH_RECEIPT_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/touch-ingress-receipt\0";

pub const RGB8_CHANNELS: u32 = 3;

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

digest_id!(PlatformIngressProfileId);
digest_id!(FrameIngressReceiptId);
digest_id!(TouchIngressReceiptId);

/// Closed platform boundary class. This identifies the adapter semantics, not a
/// claim that the underlying OS is trustworthy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum PlatformIngressKind {
    AndroidJni = 1,
    IosCAbi = 2,
}

/// Qualified strict-ingress profile for one platform adapter lineage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlatformIngressProfile {
    pub platform: PlatformIngressKind,
    pub abi_version: u32,
    pub capture_profile_id: ScreenCaptureProfileId,
    pub touch_input_profile_id: TouchInputProfileId,
    /// Hard upper bound on accepted RGB payload size before any observation is
    /// constructed. This is part of the security semantics, not a tuning hint.
    pub max_frame_bytes: u64,
}

/// Evidence that one exact framebuffer slice satisfied strict ingress and yielded
/// one exact QUAL-SOMAINTERACTION-517 frame observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameIngressReceipt {
    pub ingress_profile_id: PlatformIngressProfileId,
    pub frame_observation_id: ScreenFrameObservationId,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub expected_len: u64,
    pub actual_len: u64,
}

/// Evidence that one exact touch tuple satisfied strict ingress and yielded one
/// exact QUAL-SOMAINTERACTION-517 touch observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TouchIngressReceipt {
    pub ingress_profile_id: PlatformIngressProfileId,
    pub touch_observation_id: TouchEventObservationId,
    pub action: u8,
    pub timestamp_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlatformIngressError {
    ZeroAbiVersion,
    ZeroCaptureProfile,
    ZeroTouchProfile,
    ZeroMaxFrameBytes,
    ZeroWidth,
    ZeroHeight,
    UnsupportedChannels,
    DimensionOverflow,
    FrameTooLarge,
    FrameLengthMismatch,
    LengthNotRepresentable,
    ZeroSurface,
    ZeroSurfaceGeneration,
    ZeroInteractionGeneration,
    ZeroFrameSequence,
    ZeroInputAttester,
    ZeroInputSequence,
    InvalidTouchAction,
    NonFiniteTouchValue,
    TouchValueOutOfRange,
    Observation(SomaInteractionError),
}

impl fmt::Display for PlatformIngressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PlatformIngressError {}

impl From<SomaInteractionError> for PlatformIngressError {
    fn from(value: SomaInteractionError) -> Self {
        Self::Observation(value)
    }
}

impl PlatformIngressProfile {
    pub fn validate(&self) -> Result<(), PlatformIngressError> {
        if self.abi_version == 0 {
            return Err(PlatformIngressError::ZeroAbiVersion);
        }
        if self.capture_profile_id.is_zero() {
            return Err(PlatformIngressError::ZeroCaptureProfile);
        }
        if self.touch_input_profile_id.is_zero() {
            return Err(PlatformIngressError::ZeroTouchProfile);
        }
        if self.max_frame_bytes == 0 {
            return Err(PlatformIngressError::ZeroMaxFrameBytes);
        }
        Ok(())
    }

    pub fn profile_id(&self) -> Result<PlatformIngressProfileId, PlatformIngressError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROFILE_DOMAIN);
        hasher.update(&[self.platform as u8]);
        hasher.update(&self.abi_version.to_le_bytes());
        hasher.update(self.capture_profile_id.as_bytes());
        hasher.update(self.touch_input_profile_id.as_bytes());
        hasher.update(&self.max_frame_bytes.to_le_bytes());
        Ok(PlatformIngressProfileId(*hasher.finalize().as_bytes()))
    }

    /// Compute the exact accepted RGB byte length using checked `u64` arithmetic.
    pub fn expected_rgb_len(
        &self,
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<u64, PlatformIngressError> {
        self.validate()?;
        if width == 0 {
            return Err(PlatformIngressError::ZeroWidth);
        }
        if height == 0 {
            return Err(PlatformIngressError::ZeroHeight);
        }
        if channels != RGB8_CHANNELS {
            return Err(PlatformIngressError::UnsupportedChannels);
        }
        let expected = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(u64::from(channels)))
            .ok_or(PlatformIngressError::DimensionOverflow)?;
        if expected > self.max_frame_bytes {
            return Err(PlatformIngressError::FrameTooLarge);
        }
        Ok(expected)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn accept_frame(
        &self,
        surface_id: TrustedSurfaceId,
        surface_generation: u64,
        interaction_generation: u64,
        frame_sequence: u64,
        width: u32,
        height: u32,
        channels: u32,
        frame_rgb: &[u8],
    ) -> Result<(ScreenFrameObservation, FrameIngressReceipt), PlatformIngressError> {
        if surface_id.is_zero() {
            return Err(PlatformIngressError::ZeroSurface);
        }
        if surface_generation == 0 {
            return Err(PlatformIngressError::ZeroSurfaceGeneration);
        }
        if interaction_generation == 0 {
            return Err(PlatformIngressError::ZeroInteractionGeneration);
        }
        // Soma's ScreenVisionBridge begins at frame sequence 0, so zero is a
        // valid first frame and is intentionally not rejected here.

        let expected_len = self.expected_rgb_len(width, height, channels)?;
        let actual_len = u64::try_from(frame_rgb.len())
            .map_err(|_| PlatformIngressError::LengthNotRepresentable)?;
        if actual_len != expected_len {
            return Err(PlatformIngressError::FrameLengthMismatch);
        }
        // QUAL-SOMAINTERACTION-517 additionally checks width*height*3 using
        // checked `usize` arithmetic before hashing the exact bytes.
        let observation = ScreenFrameObservation::from_rgb(
            self.capture_profile_id,
            surface_id,
            surface_generation,
            interaction_generation,
            frame_sequence,
            width,
            height,
            frame_rgb,
        )?;
        let receipt = FrameIngressReceipt {
            ingress_profile_id: self.profile_id()?,
            frame_observation_id: observation.observation_id()?,
            width,
            height,
            channels,
            expected_len,
            actual_len,
        };
        Ok((observation, receipt))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn accept_touch(
        &self,
        input_attester_id: InputAttesterId,
        interaction_generation: u64,
        input_sequence: u64,
        x: f32,
        y: f32,
        action: u8,
        pressure: f32,
        timestamp_ms: u64,
    ) -> Result<(TouchEventObservation, TouchIngressReceipt), PlatformIngressError> {
        if input_attester_id.is_zero() {
            return Err(PlatformIngressError::ZeroInputAttester);
        }
        if interaction_generation == 0 {
            return Err(PlatformIngressError::ZeroInteractionGeneration);
        }
        if input_sequence == 0 {
            return Err(PlatformIngressError::ZeroInputSequence);
        }
        validate_unit_value(x)?;
        validate_unit_value(y)?;
        validate_unit_value(pressure)?;
        let action = decode_touch_action(action)?;

        // Preserve exact accepted values. No clamping, defaulting, or action
        // substitution occurs in the assurance ingress.
        let event = TouchEvent {
            x,
            y,
            action,
            pressure,
            timestamp_ms,
        };
        let observation = TouchEventObservation::from_touch_event(
            self.touch_input_profile_id,
            input_attester_id,
            interaction_generation,
            input_sequence,
            &event,
        )?;
        let receipt = TouchIngressReceipt {
            ingress_profile_id: self.profile_id()?,
            touch_observation_id: observation.observation_id()?,
            action: action_code(action),
            timestamp_ms,
        };
        Ok((observation, receipt))
    }
}

impl FrameIngressReceipt {
    pub fn receipt_id(&self) -> Result<FrameIngressReceiptId, PlatformIngressError> {
        if self.ingress_profile_id.is_zero() {
            return Err(PlatformIngressError::ZeroAbiVersion);
        }
        if self.frame_observation_id.is_zero() {
            return Err(PlatformIngressError::FrameLengthMismatch);
        }
        if self.expected_len != self.actual_len {
            return Err(PlatformIngressError::FrameLengthMismatch);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(FRAME_RECEIPT_DOMAIN);
        hasher.update(self.ingress_profile_id.as_bytes());
        hasher.update(self.frame_observation_id.as_bytes());
        hasher.update(&self.width.to_le_bytes());
        hasher.update(&self.height.to_le_bytes());
        hasher.update(&self.channels.to_le_bytes());
        hasher.update(&self.expected_len.to_le_bytes());
        hasher.update(&self.actual_len.to_le_bytes());
        Ok(FrameIngressReceiptId(*hasher.finalize().as_bytes()))
    }
}

impl TouchIngressReceipt {
    pub fn receipt_id(&self) -> Result<TouchIngressReceiptId, PlatformIngressError> {
        if self.ingress_profile_id.is_zero() {
            return Err(PlatformIngressError::ZeroAbiVersion);
        }
        if self.touch_observation_id.is_zero() {
            return Err(PlatformIngressError::InvalidTouchAction);
        }
        if self.action > 3 {
            return Err(PlatformIngressError::InvalidTouchAction);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(TOUCH_RECEIPT_DOMAIN);
        hasher.update(self.ingress_profile_id.as_bytes());
        hasher.update(self.touch_observation_id.as_bytes());
        hasher.update(&[self.action]);
        hasher.update(&self.timestamp_ms.to_le_bytes());
        Ok(TouchIngressReceiptId(*hasher.finalize().as_bytes()))
    }
}

fn validate_unit_value(value: f32) -> Result<(), PlatformIngressError> {
    if !value.is_finite() {
        return Err(PlatformIngressError::NonFiniteTouchValue);
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(PlatformIngressError::TouchValueOutOfRange);
    }
    Ok(())
}

fn decode_touch_action(action: u8) -> Result<TouchAction, PlatformIngressError> {
    match action {
        0 => Ok(TouchAction::Down),
        1 => Ok(TouchAction::Move),
        2 => Ok(TouchAction::Up),
        3 => Ok(TouchAction::Cancel),
        _ => Err(PlatformIngressError::InvalidTouchAction),
    }
}

fn action_code(action: TouchAction) -> u8 {
    match action {
        TouchAction::Down => 0,
        TouchAction::Move => 1,
        TouchAction::Up => 2,
        TouchAction::Cancel => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    fn profile(max_frame_bytes: u64) -> PlatformIngressProfile {
        PlatformIngressProfile {
            platform: PlatformIngressKind::AndroidJni,
            abi_version: 1,
            capture_profile_id: ScreenCaptureProfileId(bytes(1)),
            touch_input_profile_id: TouchInputProfileId(bytes(2)),
            max_frame_bytes,
        }
    }

    #[test]
    fn exact_rgb_payload_is_accepted() {
        let profile = profile(1024);
        let frame = vec![7_u8; 2 * 3 * 3];
        let (observation, receipt) = profile
            .accept_frame(
                TrustedSurfaceId(bytes(3)),
                4,
                5,
                0,
                2,
                3,
                3,
                &frame,
            )
            .unwrap();
        assert_eq!(receipt.expected_len, 18);
        assert_eq!(receipt.actual_len, 18);
        assert_eq!(receipt.frame_observation_id, observation.observation_id().unwrap());
        assert!(!receipt.receipt_id().unwrap().is_zero());
    }

    #[test]
    fn extra_or_short_frame_bytes_are_rejected() {
        let profile = profile(1024);
        for len in [17_usize, 19_usize] {
            let frame = vec![0_u8; len];
            assert_eq!(
                profile
                    .accept_frame(
                        TrustedSurfaceId(bytes(3)),
                        4,
                        5,
                        0,
                        2,
                        3,
                        3,
                        &frame,
                    )
                    .unwrap_err(),
                PlatformIngressError::FrameLengthMismatch
            );
        }
    }

    #[test]
    fn frame_size_uses_checked_arithmetic_and_bound() {
        let huge = profile(u64::MAX);
        assert_eq!(
            huge.expected_rgb_len(u32::MAX, u32::MAX, 3).unwrap_err(),
            PlatformIngressError::DimensionOverflow
        );
        let bounded = profile(16);
        assert_eq!(
            bounded.expected_rgb_len(2, 3, 3).unwrap_err(),
            PlatformIngressError::FrameTooLarge
        );
    }

    #[test]
    fn non_rgb_channels_are_rejected() {
        assert_eq!(
            profile(1024).expected_rgb_len(2, 2, 4).unwrap_err(),
            PlatformIngressError::UnsupportedChannels
        );
    }

    #[test]
    fn invalid_touch_action_is_rejected_not_mapped_to_cancel() {
        let err = profile(1024)
            .accept_touch(InputAttesterId(bytes(4)), 5, 6, 0.5, 0.5, 9, 0.5, 10)
            .unwrap_err();
        assert_eq!(err, PlatformIngressError::InvalidTouchAction);
    }

    #[test]
    fn touch_values_are_rejected_not_clamped() {
        let profile = profile(1024);
        assert_eq!(
            profile
                .accept_touch(InputAttesterId(bytes(4)), 5, 6, f32::NAN, 0.5, 0, 0.5, 10)
                .unwrap_err(),
            PlatformIngressError::NonFiniteTouchValue
        );
        assert_eq!(
            profile
                .accept_touch(InputAttesterId(bytes(4)), 5, 6, 1.1, 0.5, 0, 0.5, 10)
                .unwrap_err(),
            PlatformIngressError::TouchValueOutOfRange
        );
    }

    #[test]
    fn exact_touch_semantics_reach_517_observation() {
        let profile = profile(1024);
        let (observation, receipt) = profile
            .accept_touch(
                InputAttesterId(bytes(4)),
                5,
                6,
                -0.0,
                1.0,
                2,
                0.25,
                1234,
            )
            .unwrap();
        assert_eq!(receipt.action, 2);
        assert_eq!(receipt.timestamp_ms, 1234);
        assert_eq!(receipt.touch_observation_id, observation.observation_id().unwrap());
        assert!(!receipt.receipt_id().unwrap().is_zero());
    }

    #[test]
    fn timestamp_changes_touch_identity() {
        let profile = profile(1024);
        let (a, _) = profile
            .accept_touch(InputAttesterId(bytes(4)), 5, 6, 0.5, 0.5, 0, 0.5, 100)
            .unwrap();
        let (b, _) = profile
            .accept_touch(InputAttesterId(bytes(4)), 5, 6, 0.5, 0.5, 0, 0.5, 101)
            .unwrap();
        assert_ne!(a.observation_id().unwrap(), b.observation_id().unwrap());
    }
}
