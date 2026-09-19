// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-SOMAINTERACTIONVERIFY-518: recompute measured Soma interaction evidence
//! from the raw framebuffer and touch-event inputs.
//!
//! QUAL-SOMAINTERACTION-517 defines content-addressed observations and a structural
//! bundle, but serialized observation structs are evidence statements rather than
//! self-authenticating truth. This module closes that local verifier gap: it
//! reconstructs the frame observation from exact RGB bytes, reconstructs the touch
//! observation from the exact `TouchEvent`, then reconstructs the measured bundle
//! and requires exact equality at every boundary.
//!
//! This still does not prove that the platform capture path is complete, that no
//! overlay escaped capture, that the touch event came from the authenticated human,
//! or that framebuffer pixels equal physical display output. Those remain platform
//! attestation / trusted-attention propositions.

use core::fmt;

use symthaea_core::assurance_interaction_continuity::{
    ConfirmationInputObservation, ConfirmationInputObservationId,
    InteractionContinuityCertificate, InteractionContinuityCertificateId,
};
use symthaea_core::assurance_trusted_surface::{
    SurfaceDeliveryObservation, SurfaceDeliveryObservationId,
};

use crate::assurance_soma_interaction::{
    MeasuredInteractionBundle, MeasuredInteractionBundleId, ScreenFrameObservation,
    ScreenFrameObservationId, SomaInteractionError, TouchEventObservation,
    TouchEventObservationId,
};
use crate::touch_body::TouchEvent;

const CERTIFICATE_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/measured-interaction-verification-certificate\0";

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

digest_id!(MeasuredInteractionVerificationCertificateId);

/// Result of recomputing the complete 517 measured-evidence path from raw inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeasuredInteractionVerificationCertificate {
    pub frame_observation_id: ScreenFrameObservationId,
    pub touch_observation_id: TouchEventObservationId,
    pub bundle_id: MeasuredInteractionBundleId,
    pub delivery_observation_id: SurfaceDeliveryObservationId,
    pub input_observation_id: ConfirmationInputObservationId,
    pub continuity_certificate_id: InteractionContinuityCertificateId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SomaInteractionVerificationError {
    FrameObservationMismatch,
    TouchObservationMismatch,
    BundleMismatch,
    ZeroCertificateComponent,
    Measured(SomaInteractionError),
}

impl fmt::Display for SomaInteractionVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SomaInteractionVerificationError {}

impl From<SomaInteractionError> for SomaInteractionVerificationError {
    fn from(value: SomaInteractionError) -> Self {
        Self::Measured(value)
    }
}

impl MeasuredInteractionVerificationCertificate {
    #[allow(clippy::too_many_arguments)]
    pub fn verify_raw(
        expected_frame: &ScreenFrameObservation,
        frame_rgb: &[u8],
        expected_touch: &TouchEventObservation,
        touch_event: &TouchEvent,
        delivery_observation: &SurfaceDeliveryObservation,
        input_observation: &ConfirmationInputObservation,
        continuity_certificate: &InteractionContinuityCertificate,
        expected_bundle: &MeasuredInteractionBundle,
    ) -> Result<Self, SomaInteractionVerificationError> {
        let recomputed_frame = ScreenFrameObservation::from_rgb(
            expected_frame.capture_profile_id,
            expected_frame.surface_id,
            expected_frame.surface_generation,
            expected_frame.interaction_generation,
            expected_frame.frame_sequence,
            expected_frame.width,
            expected_frame.height,
            frame_rgb,
        )?;
        if &recomputed_frame != expected_frame {
            return Err(SomaInteractionVerificationError::FrameObservationMismatch);
        }

        let recomputed_touch = TouchEventObservation::from_touch_event(
            expected_touch.input_profile_id,
            expected_touch.input_attester_id,
            expected_touch.interaction_generation,
            expected_touch.input_sequence,
            touch_event,
        )?;
        if &recomputed_touch != expected_touch {
            return Err(SomaInteractionVerificationError::TouchObservationMismatch);
        }

        let recomputed_bundle = MeasuredInteractionBundle::link(
            &recomputed_frame,
            &recomputed_touch,
            delivery_observation,
            input_observation,
            continuity_certificate,
        )?;
        if &recomputed_bundle != expected_bundle {
            return Err(SomaInteractionVerificationError::BundleMismatch);
        }

        Ok(Self {
            frame_observation_id: recomputed_frame.observation_id()?,
            touch_observation_id: recomputed_touch.observation_id()?,
            bundle_id: recomputed_bundle.bundle_id()?,
            delivery_observation_id: delivery_observation.observation_id().map_err(SomaInteractionError::from)?,
            input_observation_id: input_observation.observation_id().map_err(SomaInteractionError::from)?,
            continuity_certificate_id: continuity_certificate
                .certificate_id()
                .map_err(SomaInteractionError::from)?,
        })
    }

    pub fn certificate_id(
        &self,
    ) -> Result<MeasuredInteractionVerificationCertificateId, SomaInteractionVerificationError> {
        if self.frame_observation_id.is_zero()
            || self.touch_observation_id.is_zero()
            || self.bundle_id.is_zero()
            || self.delivery_observation_id.is_zero()
            || self.input_observation_id.is_zero()
            || self.continuity_certificate_id.is_zero()
        {
            return Err(SomaInteractionVerificationError::ZeroCertificateComponent);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.frame_observation_id.as_bytes());
        hasher.update(self.touch_observation_id.as_bytes());
        hasher.update(self.bundle_id.as_bytes());
        hasher.update(self.delivery_observation_id.as_bytes());
        hasher.update(self.input_observation_id.as_bytes());
        hasher.update(self.continuity_certificate_id.as_bytes());
        Ok(MeasuredInteractionVerificationCertificateId(
            *hasher.finalize().as_bytes(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_soma_interaction::{
        ScreenCaptureProfileId, TouchInputProfileId,
    };
    use crate::touch_body::TouchAction;
    use symthaea_core::assurance::{
        ActionRequestId, AuthenticationContextId, ConfirmationId, InteractionContextId,
        PresentationContextId, PresentationId, PrincipalId,
    };
    use symthaea_core::assurance_interaction_continuity::{
        InputAttesterId, InputEventNonce, InteractionContinuityProfileId,
    };
    use symthaea_core::assurance_render_artifact::{
        RenderArtifactCertificateId, RenderArtifactSetRoot,
    };
    use symthaea_core::assurance_trusted_surface::{
        InteractionNonce, SurfaceAttesterId, SurfaceDeliveryCertificateId,
        TrustedSurfaceId, TrustedSurfaceProfileId,
    };

    fn bytes(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    fn fixture() -> (
        Vec<u8>,
        ScreenFrameObservation,
        TouchEvent,
        TouchEventObservation,
        SurfaceDeliveryObservation,
        ConfirmationInputObservation,
        InteractionContinuityCertificate,
        MeasuredInteractionBundle,
    ) {
        let surface = TrustedSurfaceId(bytes(1));
        let input_attester = InputAttesterId(bytes(2));
        let frame_rgb = vec![1, 2, 3, 4, 5, 6];
        let frame = ScreenFrameObservation::from_rgb(
            ScreenCaptureProfileId(bytes(3)),
            surface,
            7,
            8,
            0,
            2,
            1,
            &frame_rgb,
        )
        .unwrap();
        let touch_event = TouchEvent {
            x: 0.25,
            y: 0.75,
            action: TouchAction::Up,
            pressure: 0.5,
            timestamp_ms: 2222,
        };
        let touch = TouchEventObservation::from_touch_event(
            TouchInputProfileId(bytes(4)),
            input_attester,
            8,
            9,
            &touch_event,
        )
        .unwrap();
        let delivery = SurfaceDeliveryObservation {
            surface_profile_id: TrustedSurfaceProfileId(bytes(5)),
            surface_id: surface,
            attester_id: SurfaceAttesterId(bytes(6)),
            surface_generation: 7,
            presentation_id: PresentationId(bytes(7)),
            presentation_context_id: PresentationContextId(bytes(8)),
            render_artifact_certificate_id: RenderArtifactCertificateId(bytes(9)),
            artifact_set_root: RenderArtifactSetRoot(bytes(10)),
            freshness_certificate_id: symthaea_core::assurance_presentation_currentness::PresentationFreshnessCertificateId(bytes(11)),
            delivery_sequence: 1,
            interaction_nonce: InteractionNonce(bytes(12)),
        };
        let delivery_id = delivery.observation_id().unwrap();
        let input = ConfirmationInputObservation {
            continuity_profile_id: InteractionContinuityProfileId(bytes(13)),
            principal_id: PrincipalId(bytes(14)),
            action_request_id: ActionRequestId(bytes(15)),
            presentation_id: delivery.presentation_id,
            confirmation_id: ConfirmationId(bytes(16)),
            presentation_context_id: delivery.presentation_context_id,
            interaction_context_id: InteractionContextId(bytes(17)),
            authentication_context_id: AuthenticationContextId(bytes(18)),
            surface_delivery_certificate_id: SurfaceDeliveryCertificateId(bytes(19)),
            predecessor_delivery_observation_id: delivery_id,
            input_attester_id: input_attester,
            interaction_generation: 8,
            input_sequence: 9,
            input_nonce: InputEventNonce(bytes(20)),
        };
        let continuity = InteractionContinuityCertificate {
            continuity_profile_id: input.continuity_profile_id,
            input_observation_id: input.observation_id().unwrap(),
            principal_id: input.principal_id,
            action_request_id: input.action_request_id,
            presentation_id: input.presentation_id,
            confirmation_id: input.confirmation_id,
            surface_delivery_certificate_id: input.surface_delivery_certificate_id,
            predecessor_delivery_observation_id: input.predecessor_delivery_observation_id,
        };
        let bundle = MeasuredInteractionBundle::link(
            &frame,
            &touch,
            &delivery,
            &input,
            &continuity,
        )
        .unwrap();
        (
            frame_rgb,
            frame,
            touch_event,
            touch,
            delivery,
            input,
            continuity,
            bundle,
        )
    }

    #[test]
    fn exact_raw_inputs_verify() {
        let (frame_rgb, frame, event, touch, delivery, input, continuity, bundle) = fixture();
        let cert = MeasuredInteractionVerificationCertificate::verify_raw(
            &frame,
            &frame_rgb,
            &touch,
            &event,
            &delivery,
            &input,
            &continuity,
            &bundle,
        )
        .unwrap();
        assert_ne!(
            cert.certificate_id().unwrap(),
            MeasuredInteractionVerificationCertificateId::ZERO
        );
    }

    #[test]
    fn changed_frame_bytes_are_rejected() {
        let (mut frame_rgb, frame, event, touch, delivery, input, continuity, bundle) = fixture();
        frame_rgb[0] ^= 1;
        assert_eq!(
            MeasuredInteractionVerificationCertificate::verify_raw(
                &frame,
                &frame_rgb,
                &touch,
                &event,
                &delivery,
                &input,
                &continuity,
                &bundle,
            ),
            Err(SomaInteractionVerificationError::FrameObservationMismatch)
        );
    }

    #[test]
    fn changed_touch_event_is_rejected() {
        let (frame_rgb, frame, mut event, touch, delivery, input, continuity, bundle) = fixture();
        event.x = 0.5;
        assert_eq!(
            MeasuredInteractionVerificationCertificate::verify_raw(
                &frame,
                &frame_rgb,
                &touch,
                &event,
                &delivery,
                &input,
                &continuity,
                &bundle,
            ),
            Err(SomaInteractionVerificationError::TouchObservationMismatch)
        );
    }

    #[test]
    fn forged_frame_digest_is_rejected_by_recomputation() {
        let (frame_rgb, mut frame, event, touch, delivery, input, continuity, bundle) = fixture();
        frame.framebuffer_digest = crate::assurance_soma_interaction::FramebufferDigest(bytes(99));
        assert_eq!(
            MeasuredInteractionVerificationCertificate::verify_raw(
                &frame,
                &frame_rgb,
                &touch,
                &event,
                &delivery,
                &input,
                &continuity,
                &bundle,
            ),
            Err(SomaInteractionVerificationError::FrameObservationMismatch)
        );
    }

    #[test]
    fn wrong_bundle_is_rejected() {
        let (frame_rgb, frame, event, touch, delivery, input, continuity, mut bundle) = fixture();
        bundle.touch_observation_id = TouchEventObservationId(bytes(98));
        assert_eq!(
            MeasuredInteractionVerificationCertificate::verify_raw(
                &frame,
                &frame_rgb,
                &touch,
                &event,
                &delivery,
                &input,
                &continuity,
                &bundle,
            ),
            Err(SomaInteractionVerificationError::BundleMismatch)
        );
    }
}
