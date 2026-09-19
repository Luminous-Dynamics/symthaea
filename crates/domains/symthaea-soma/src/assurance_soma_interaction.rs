// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-SOMAINTERACTION-517: measured framebuffer and touch-event evidence.
//!
//! The presentation-assurance chain through QUAL-INTERACTIONCONTINUITY-514
//! establishes structural lineage between one trusted-surface delivery statement
//! and one confirmation-input statement. This module grounds the next layer in
//! Soma's real embodiment seams: raw RGB framebuffer bytes accepted by the screen
//! path and `TouchEvent` values accepted by the touch path.
//!
//! These observations prove only exact adapter inputs. A framebuffer capture is
//! not, by itself, proof of what a human physically perceived, of overlay absence,
//! or of compositor/display-hardware integrity. Likewise a platform touch event is
//! not, by itself, proof of the human who generated it. Those remain separate
//! platform-attestation and trusted-attention propositions.

use core::fmt;

use symthaea_core::assurance_interaction_continuity::{
    ConfirmationInputObservation, ConfirmationInputObservationId, InputAttesterId,
    InteractionContinuityCertificate, InteractionContinuityCertificateId,
    InteractionContinuityError,
};
use symthaea_core::assurance_trusted_surface::{
    SurfaceDeliveryObservation, SurfaceDeliveryObservationId, TrustedSurfaceError,
    TrustedSurfaceId,
};

use crate::touch_body::{TouchAction, TouchEvent};

const FRAME_BYTES_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/framebuffer-bytes\0";
const FRAME_OBSERVATION_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/framebuffer-observation\0";
const TOUCH_OBSERVATION_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/touch-observation\0";
const BUNDLE_DOMAIN: &[u8] = b"symthaea.soma.presentation.v1/measured-interaction-bundle\0";

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

digest_id!(ScreenCaptureProfileId);
digest_id!(TouchInputProfileId);
digest_id!(FramebufferDigest);
digest_id!(ScreenFrameObservationId);
digest_id!(TouchEventObservationId);
digest_id!(MeasuredInteractionBundleId);

/// Pixel format whose exact byte layout is committed by a frame observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ObservedPixelFormat {
    /// Packed row-major RGB, eight bits per channel, exactly three bytes/pixel.
    Rgb8 = 1,
}

/// Canonicalized finite value in the closed unit interval [0, 1].
///
/// `-0.0` is canonicalized to `+0.0`; all other finite in-range values retain
/// their exact IEEE-754 `f32` bit pattern. NaN, infinities, and out-of-range
/// values are rejected rather than normalized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanonicalUnitF32(u32);

impl CanonicalUnitF32 {
    pub fn from_value(value: f32) -> Result<Self, SomaInteractionError> {
        if !value.is_finite() {
            return Err(SomaInteractionError::NonFiniteTouchValue);
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(SomaInteractionError::TouchValueOutOfRange);
        }
        let canonical = if value == 0.0 { 0.0 } else { value };
        Ok(Self(canonical.to_bits()))
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub fn value(self) -> f32 {
        f32::from_bits(self.0)
    }
}

/// Closed touch-action representation committed by measured input evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ObservedTouchAction {
    Down = 1,
    Move = 2,
    Up = 3,
    Cancel = 4,
}

impl From<TouchAction> for ObservedTouchAction {
    fn from(value: TouchAction) -> Self {
        match value {
            TouchAction::Down => Self::Down,
            TouchAction::Move => Self::Move,
            TouchAction::Up => Self::Up,
            TouchAction::Cancel => Self::Cancel,
        }
    }
}

/// Exact observation of RGB bytes supplied to Soma's screen-capture path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScreenFrameObservation {
    pub capture_profile_id: ScreenCaptureProfileId,
    pub surface_id: TrustedSurfaceId,
    pub surface_generation: u64,
    pub interaction_generation: u64,
    pub frame_sequence: u64,
    pub width: u32,
    pub height: u32,
    pub pixel_format: ObservedPixelFormat,
    pub framebuffer_digest: FramebufferDigest,
}

/// Exact canonical observation of one `TouchEvent` supplied to Soma.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TouchEventObservation {
    pub input_profile_id: TouchInputProfileId,
    pub input_attester_id: InputAttesterId,
    pub interaction_generation: u64,
    pub input_sequence: u64,
    pub x: CanonicalUnitF32,
    pub y: CanonicalUnitF32,
    pub pressure: CanonicalUnitF32,
    pub action: ObservedTouchAction,
    /// Platform timestamp is committed as reported, but is not used as a causal
    /// ordering theorem because Soma's source documents permit epoch or boot time.
    pub platform_timestamp_ms: u64,
}

/// Structural linkage between measured Soma observations and the already-defined
/// trusted-delivery / interaction-continuity evidence graph.
///
/// This bundle does not re-prove QUAL-INTERACTIONCONTINUITY-514; its certificate
/// ID is a parent reference whose validity remains a prerequisite theorem.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeasuredInteractionBundle {
    pub frame_observation_id: ScreenFrameObservationId,
    pub touch_observation_id: TouchEventObservationId,
    pub delivery_observation_id: SurfaceDeliveryObservationId,
    pub input_observation_id: ConfirmationInputObservationId,
    pub continuity_certificate_id: InteractionContinuityCertificateId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SomaInteractionError {
    ZeroCaptureProfile,
    ZeroInputProfile,
    ZeroSurface,
    ZeroInputAttester,
    ZeroSurfaceGeneration,
    ZeroInteractionGeneration,
    ZeroInputSequence,
    ZeroWidth,
    ZeroHeight,
    DimensionOverflow,
    FrameLengthMismatch,
    NonFiniteTouchValue,
    TouchValueOutOfRange,
    SurfaceMismatch,
    SurfaceGenerationMismatch,
    InteractionGenerationMismatch,
    InputAttesterMismatch,
    InputSequenceMismatch,
    PredecessorDeliveryMismatch,
    ContinuityInputMismatch,
    ContinuityDeliveryMismatch,
    ContinuityPresentationMismatch,
    ContinuityConfirmationMismatch,
    TrustedSurface(TrustedSurfaceError),
    InteractionContinuity(InteractionContinuityError),
}

impl fmt::Display for SomaInteractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SomaInteractionError {}

impl From<TrustedSurfaceError> for SomaInteractionError {
    fn from(value: TrustedSurfaceError) -> Self {
        Self::TrustedSurface(value)
    }
}

impl From<InteractionContinuityError> for SomaInteractionError {
    fn from(value: InteractionContinuityError) -> Self {
        Self::InteractionContinuity(value)
    }
}

impl ScreenFrameObservation {
    #[allow(clippy::too_many_arguments)]
    pub fn from_rgb(
        capture_profile_id: ScreenCaptureProfileId,
        surface_id: TrustedSurfaceId,
        surface_generation: u64,
        interaction_generation: u64,
        frame_sequence: u64,
        width: u32,
        height: u32,
        frame_rgb: &[u8],
    ) -> Result<Self, SomaInteractionError> {
        if capture_profile_id.is_zero() {
            return Err(SomaInteractionError::ZeroCaptureProfile);
        }
        if surface_id.is_zero() {
            return Err(SomaInteractionError::ZeroSurface);
        }
        if surface_generation == 0 {
            return Err(SomaInteractionError::ZeroSurfaceGeneration);
        }
        if interaction_generation == 0 {
            return Err(SomaInteractionError::ZeroInteractionGeneration);
        }
        if width == 0 {
            return Err(SomaInteractionError::ZeroWidth);
        }
        if height == 0 {
            return Err(SomaInteractionError::ZeroHeight);
        }

        let expected_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or(SomaInteractionError::DimensionOverflow)?;
        if frame_rgb.len() != expected_len {
            return Err(SomaInteractionError::FrameLengthMismatch);
        }

        let mut bytes_hasher = blake3::Hasher::new();
        bytes_hasher.update(FRAME_BYTES_DOMAIN);
        bytes_hasher.update(&(frame_rgb.len() as u64).to_le_bytes());
        bytes_hasher.update(frame_rgb);
        let framebuffer_digest = FramebufferDigest(*bytes_hasher.finalize().as_bytes());

        Ok(Self {
            capture_profile_id,
            surface_id,
            surface_generation,
            interaction_generation,
            frame_sequence,
            width,
            height,
            pixel_format: ObservedPixelFormat::Rgb8,
            framebuffer_digest,
        })
    }

    pub fn validate(&self) -> Result<(), SomaInteractionError> {
        if self.capture_profile_id.is_zero() {
            return Err(SomaInteractionError::ZeroCaptureProfile);
        }
        if self.surface_id.is_zero() {
            return Err(SomaInteractionError::ZeroSurface);
        }
        if self.surface_generation == 0 {
            return Err(SomaInteractionError::ZeroSurfaceGeneration);
        }
        if self.interaction_generation == 0 {
            return Err(SomaInteractionError::ZeroInteractionGeneration);
        }
        if self.width == 0 {
            return Err(SomaInteractionError::ZeroWidth);
        }
        if self.height == 0 {
            return Err(SomaInteractionError::ZeroHeight);
        }
        if self.framebuffer_digest.is_zero() {
            return Err(SomaInteractionError::FrameLengthMismatch);
        }
        Ok(())
    }

    pub fn observation_id(&self) -> Result<ScreenFrameObservationId, SomaInteractionError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(FRAME_OBSERVATION_DOMAIN);
        hasher.update(self.capture_profile_id.as_bytes());
        hasher.update(self.surface_id.as_bytes());
        hasher.update(&self.surface_generation.to_le_bytes());
        hasher.update(&self.interaction_generation.to_le_bytes());
        hasher.update(&self.frame_sequence.to_le_bytes());
        hasher.update(&self.width.to_le_bytes());
        hasher.update(&self.height.to_le_bytes());
        hasher.update(&[self.pixel_format as u8]);
        hasher.update(self.framebuffer_digest.as_bytes());
        Ok(ScreenFrameObservationId(*hasher.finalize().as_bytes()))
    }
}

impl TouchEventObservation {
    pub fn from_touch_event(
        input_profile_id: TouchInputProfileId,
        input_attester_id: InputAttesterId,
        interaction_generation: u64,
        input_sequence: u64,
        event: &TouchEvent,
    ) -> Result<Self, SomaInteractionError> {
        if input_profile_id.is_zero() {
            return Err(SomaInteractionError::ZeroInputProfile);
        }
        if input_attester_id.is_zero() {
            return Err(SomaInteractionError::ZeroInputAttester);
        }
        if interaction_generation == 0 {
            return Err(SomaInteractionError::ZeroInteractionGeneration);
        }
        if input_sequence == 0 {
            return Err(SomaInteractionError::ZeroInputSequence);
        }

        Ok(Self {
            input_profile_id,
            input_attester_id,
            interaction_generation,
            input_sequence,
            x: CanonicalUnitF32::from_value(event.x)?,
            y: CanonicalUnitF32::from_value(event.y)?,
            pressure: CanonicalUnitF32::from_value(event.pressure)?,
            action: event.action.into(),
            platform_timestamp_ms: event.timestamp_ms,
        })
    }

    pub fn validate(&self) -> Result<(), SomaInteractionError> {
        if self.input_profile_id.is_zero() {
            return Err(SomaInteractionError::ZeroInputProfile);
        }
        if self.input_attester_id.is_zero() {
            return Err(SomaInteractionError::ZeroInputAttester);
        }
        if self.interaction_generation == 0 {
            return Err(SomaInteractionError::ZeroInteractionGeneration);
        }
        if self.input_sequence == 0 {
            return Err(SomaInteractionError::ZeroInputSequence);
        }
        for value in [self.x, self.y, self.pressure] {
            CanonicalUnitF32::from_value(value.value())?;
        }
        Ok(())
    }

    pub fn observation_id(&self) -> Result<TouchEventObservationId, SomaInteractionError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(TOUCH_OBSERVATION_DOMAIN);
        hasher.update(self.input_profile_id.as_bytes());
        hasher.update(self.input_attester_id.as_bytes());
        hasher.update(&self.interaction_generation.to_le_bytes());
        hasher.update(&self.input_sequence.to_le_bytes());
        hasher.update(&self.x.bits().to_le_bytes());
        hasher.update(&self.y.bits().to_le_bytes());
        hasher.update(&self.pressure.bits().to_le_bytes());
        hasher.update(&[self.action as u8]);
        hasher.update(&self.platform_timestamp_ms.to_le_bytes());
        Ok(TouchEventObservationId(*hasher.finalize().as_bytes()))
    }
}

impl MeasuredInteractionBundle {
    pub fn link(
        frame: &ScreenFrameObservation,
        touch: &TouchEventObservation,
        delivery_observation: &SurfaceDeliveryObservation,
        input_observation: &ConfirmationInputObservation,
        continuity_certificate: &InteractionContinuityCertificate,
    ) -> Result<Self, SomaInteractionError> {
        frame.validate()?;
        touch.validate()?;
        delivery_observation.validate()?;
        input_observation.validate()?;

        let delivery_observation_id = delivery_observation.observation_id()?;
        let input_observation_id = input_observation.observation_id()?;
        let continuity_certificate_id = continuity_certificate.certificate_id()?;

        if frame.surface_id != delivery_observation.surface_id {
            return Err(SomaInteractionError::SurfaceMismatch);
        }
        if frame.surface_generation != delivery_observation.surface_generation {
            return Err(SomaInteractionError::SurfaceGenerationMismatch);
        }
        if frame.interaction_generation != input_observation.interaction_generation
            || touch.interaction_generation != input_observation.interaction_generation
        {
            return Err(SomaInteractionError::InteractionGenerationMismatch);
        }
        if touch.input_attester_id != input_observation.input_attester_id {
            return Err(SomaInteractionError::InputAttesterMismatch);
        }
        if touch.input_sequence != input_observation.input_sequence {
            return Err(SomaInteractionError::InputSequenceMismatch);
        }
        if input_observation.predecessor_delivery_observation_id != delivery_observation_id {
            return Err(SomaInteractionError::PredecessorDeliveryMismatch);
        }
        if continuity_certificate.input_observation_id != input_observation_id {
            return Err(SomaInteractionError::ContinuityInputMismatch);
        }
        if continuity_certificate.predecessor_delivery_observation_id != delivery_observation_id {
            return Err(SomaInteractionError::ContinuityDeliveryMismatch);
        }
        if continuity_certificate.presentation_id != input_observation.presentation_id {
            return Err(SomaInteractionError::ContinuityPresentationMismatch);
        }
        if continuity_certificate.confirmation_id != input_observation.confirmation_id {
            return Err(SomaInteractionError::ContinuityConfirmationMismatch);
        }

        Ok(Self {
            frame_observation_id: frame.observation_id()?,
            touch_observation_id: touch.observation_id()?,
            delivery_observation_id,
            input_observation_id,
            continuity_certificate_id,
        })
    }

    pub fn bundle_id(&self) -> Result<MeasuredInteractionBundleId, SomaInteractionError> {
        if self.frame_observation_id.is_zero()
            || self.touch_observation_id.is_zero()
            || self.delivery_observation_id.is_zero()
            || self.input_observation_id.is_zero()
            || self.continuity_certificate_id.is_zero()
        {
            return Err(SomaInteractionError::ContinuityInputMismatch);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(BUNDLE_DOMAIN);
        hasher.update(self.frame_observation_id.as_bytes());
        hasher.update(self.touch_observation_id.as_bytes());
        hasher.update(self.delivery_observation_id.as_bytes());
        hasher.update(self.input_observation_id.as_bytes());
        hasher.update(self.continuity_certificate_id.as_bytes());
        Ok(MeasuredInteractionBundleId(*hasher.finalize().as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::assurance::{
        ActionRequestId, AuthenticationContextId, ConfirmationId, InteractionContextId,
        PresentationContextId, PresentationId, PrincipalId,
    };
    use symthaea_core::assurance_interaction_continuity::{
        InputEventNonce, InteractionContinuityProfileId,
    };
    use symthaea_core::assurance_render_artifact::{
        RenderArtifactCertificateId, RenderArtifactSetRoot,
    };
    use symthaea_core::assurance_trusted_surface::{
        InteractionNonce, SurfaceAttesterId, SurfaceDeliveryCertificateId,
        TrustedSurfaceProfileId,
    };

    fn bytes(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    fn frame(surface: TrustedSurfaceId, generation: u64, interaction: u64) -> ScreenFrameObservation {
        ScreenFrameObservation::from_rgb(
            ScreenCaptureProfileId(bytes(1)),
            surface,
            generation,
            interaction,
            0,
            2,
            1,
            &[1, 2, 3, 4, 5, 6],
        )
        .unwrap()
    }

    fn touch(attester: InputAttesterId, interaction: u64, sequence: u64) -> TouchEventObservation {
        TouchEventObservation::from_touch_event(
            TouchInputProfileId(bytes(2)),
            attester,
            interaction,
            sequence,
            &TouchEvent {
                x: 0.25,
                y: 0.75,
                action: TouchAction::Up,
                pressure: 0.5,
                timestamp_ms: 1234,
            },
        )
        .unwrap()
    }

    fn delivery(surface: TrustedSurfaceId, generation: u64) -> SurfaceDeliveryObservation {
        SurfaceDeliveryObservation {
            surface_profile_id: TrustedSurfaceProfileId(bytes(10)),
            surface_id: surface,
            attester_id: SurfaceAttesterId(bytes(11)),
            surface_generation: generation,
            presentation_id: PresentationId(bytes(12)),
            presentation_context_id: PresentationContextId(bytes(13)),
            render_artifact_certificate_id: RenderArtifactCertificateId(bytes(14)),
            artifact_set_root: RenderArtifactSetRoot(bytes(15)),
            freshness_certificate_id: symthaea_core::assurance_presentation_currentness::PresentationFreshnessCertificateId(bytes(16)),
            delivery_sequence: 1,
            interaction_nonce: InteractionNonce(bytes(17)),
        }
    }

    fn input(
        attester: InputAttesterId,
        interaction: u64,
        sequence: u64,
        delivery_id: SurfaceDeliveryObservationId,
    ) -> ConfirmationInputObservation {
        ConfirmationInputObservation {
            continuity_profile_id: InteractionContinuityProfileId(bytes(20)),
            principal_id: PrincipalId(bytes(21)),
            action_request_id: ActionRequestId(bytes(22)),
            presentation_id: PresentationId(bytes(12)),
            confirmation_id: ConfirmationId(bytes(23)),
            presentation_context_id: PresentationContextId(bytes(13)),
            interaction_context_id: InteractionContextId(bytes(24)),
            authentication_context_id: AuthenticationContextId(bytes(25)),
            surface_delivery_certificate_id: SurfaceDeliveryCertificateId(bytes(26)),
            predecessor_delivery_observation_id: delivery_id,
            input_attester_id: attester,
            interaction_generation: interaction,
            input_sequence: sequence,
            input_nonce: InputEventNonce(bytes(27)),
        }
    }

    fn continuity(input: &ConfirmationInputObservation) -> InteractionContinuityCertificate {
        InteractionContinuityCertificate {
            continuity_profile_id: input.continuity_profile_id,
            input_observation_id: input.observation_id().unwrap(),
            principal_id: input.principal_id,
            action_request_id: input.action_request_id,
            presentation_id: input.presentation_id,
            confirmation_id: input.confirmation_id,
            surface_delivery_certificate_id: input.surface_delivery_certificate_id,
            predecessor_delivery_observation_id: input.predecessor_delivery_observation_id,
        }
    }

    #[test]
    fn exact_frame_bytes_define_identity() {
        let surface = TrustedSurfaceId(bytes(30));
        let a = frame(surface, 3, 4);
        let b = ScreenFrameObservation::from_rgb(
            ScreenCaptureProfileId(bytes(1)),
            surface,
            3,
            4,
            0,
            2,
            1,
            &[1, 2, 3, 4, 5, 7],
        )
        .unwrap();
        assert_ne!(a.observation_id().unwrap(), b.observation_id().unwrap());
    }

    #[test]
    fn malformed_frame_length_is_rejected() {
        assert_eq!(
            ScreenFrameObservation::from_rgb(
                ScreenCaptureProfileId(bytes(1)),
                TrustedSurfaceId(bytes(30)),
                3,
                4,
                0,
                2,
                1,
                &[1, 2, 3],
            ),
            Err(SomaInteractionError::FrameLengthMismatch)
        );
    }

    #[test]
    fn negative_zero_touch_is_canonicalized() {
        let value = CanonicalUnitF32::from_value(-0.0).unwrap();
        assert_eq!(value.bits(), 0.0f32.to_bits());
    }

    #[test]
    fn non_finite_touch_is_rejected() {
        assert_eq!(
            CanonicalUnitF32::from_value(f32::NAN),
            Err(SomaInteractionError::NonFiniteTouchValue)
        );
    }

    #[test]
    fn touch_change_changes_identity() {
        let attester = InputAttesterId(bytes(31));
        let a = touch(attester, 4, 5);
        let mut b = a;
        b.action = ObservedTouchAction::Down;
        assert_ne!(a.observation_id().unwrap(), b.observation_id().unwrap());
    }

    #[test]
    fn exact_measured_interaction_links() {
        let surface = TrustedSurfaceId(bytes(30));
        let attester = InputAttesterId(bytes(31));
        let delivery = delivery(surface, 3);
        let input = input(attester, 4, 5, delivery.observation_id().unwrap());
        let bundle = MeasuredInteractionBundle::link(
            &frame(surface, 3, 4),
            &touch(attester, 4, 5),
            &delivery,
            &input,
            &continuity(&input),
        )
        .unwrap();
        assert_ne!(bundle.bundle_id().unwrap(), MeasuredInteractionBundleId::ZERO);
    }

    #[test]
    fn wrong_surface_generation_is_rejected() {
        let surface = TrustedSurfaceId(bytes(30));
        let attester = InputAttesterId(bytes(31));
        let delivery = delivery(surface, 3);
        let input = input(attester, 4, 5, delivery.observation_id().unwrap());
        assert_eq!(
            MeasuredInteractionBundle::link(
                &frame(surface, 2, 4),
                &touch(attester, 4, 5),
                &delivery,
                &input,
                &continuity(&input),
            ),
            Err(SomaInteractionError::SurfaceGenerationMismatch)
        );
    }

    #[test]
    fn wrong_input_sequence_is_rejected() {
        let surface = TrustedSurfaceId(bytes(30));
        let attester = InputAttesterId(bytes(31));
        let delivery = delivery(surface, 3);
        let input = input(attester, 4, 5, delivery.observation_id().unwrap());
        assert_eq!(
            MeasuredInteractionBundle::link(
                &frame(surface, 3, 4),
                &touch(attester, 4, 6),
                &delivery,
                &input,
                &continuity(&input),
            ),
            Err(SomaInteractionError::InputSequenceMismatch)
        );
    }
}
