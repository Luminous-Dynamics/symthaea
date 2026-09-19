// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-INTERACTIONCONTINUITY-514: bind confirmation input to the exact
//! interaction lineage in which its presentation was delivered.
//!
//! QUAL-TRUSTEDSURFACE-512 establishes a structural delivery observation for an
//! exact current presentation. QUAL-CONFIRMUSE-513 establishes one-use semantic
//! confirmation authority. This module closes the lineage gap between those two
//! theorems: a confirmation input must name the exact prior surface-delivery
//! observation and must agree on principal, action, presentation, interaction
//! context, authentication context, input attester, and interaction generation.
//!
//! The causal predecessor link is structural and does not depend on wall-clock
//! timestamps. This module does not authenticate the input attester, prove human
//! perception/comprehension, or prove physical input-device integrity. Those are
//! separate propositions.

use core::fmt;

use crate::assurance::{
    ActionRequestId, AuthenticationContextId, ConfirmationBinding, ConfirmationId,
    InteractionContextId, PresentationContextId, PresentationId, PresentationManifest,
    PrincipalId, ValidationError,
};
use crate::assurance_presentation_currentness::{
    PresentationCurrentnessError, PresentationFreshnessCertificate, PresentationSnapshot,
};
use crate::assurance_render_artifact::{
    FieldRenderArtifact, RenderArtifactCertificate, RenderArtifactError, WarningRenderArtifact,
};
use crate::assurance_trusted_surface::{
    SurfaceDeliveryCertificate, SurfaceDeliveryCertificateId, SurfaceDeliveryObservation,
    SurfaceDeliveryObservationId, TrustedSurfaceError, TrustedSurfaceProfile,
    TrustedSurfaceProfileId,
};

const PROFILE_DOMAIN: &[u8] = b"symthaea.presentation.v1/interaction-continuity-profile\0";
const INPUT_DOMAIN: &[u8] = b"symthaea.presentation.v1/confirmation-input-observation\0";
const CERTIFICATE_DOMAIN: &[u8] = b"symthaea.presentation.v1/interaction-continuity-certificate\0";

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

digest_id!(InputAttesterId);
digest_id!(InteractionContinuityProfileId);
digest_id!(InputEventNonce);
digest_id!(ConfirmationInputObservationId);
digest_id!(InteractionContinuityCertificateId);

/// Policy-qualified interaction lineage for one exact principal/action pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InteractionContinuityProfile {
    pub principal_id: PrincipalId,
    pub action_request_id: ActionRequestId,
    pub presentation_context_id: PresentationContextId,
    pub interaction_context_id: InteractionContextId,
    pub authentication_context_id: AuthenticationContextId,
    pub surface_profile_id: TrustedSurfaceProfileId,
    pub input_attester_id: InputAttesterId,
    pub interaction_generation: u64,
}

/// One input-adapter statement that a confirmation input followed one exact
/// surface-delivery observation in the same qualified interaction lineage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfirmationInputObservation {
    pub continuity_profile_id: InteractionContinuityProfileId,
    pub principal_id: PrincipalId,
    pub action_request_id: ActionRequestId,
    pub presentation_id: PresentationId,
    pub confirmation_id: ConfirmationId,
    pub presentation_context_id: PresentationContextId,
    pub interaction_context_id: InteractionContextId,
    pub authentication_context_id: AuthenticationContextId,
    pub surface_delivery_certificate_id: SurfaceDeliveryCertificateId,
    pub predecessor_delivery_observation_id: SurfaceDeliveryObservationId,
    pub input_attester_id: InputAttesterId,
    pub interaction_generation: u64,
    pub input_sequence: u64,
    pub input_nonce: InputEventNonce,
}

/// Derived structural proof that confirmation input and presentation delivery
/// belong to the same exact interaction lineage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InteractionContinuityCertificate {
    pub continuity_profile_id: InteractionContinuityProfileId,
    pub input_observation_id: ConfirmationInputObservationId,
    pub principal_id: PrincipalId,
    pub action_request_id: ActionRequestId,
    pub presentation_id: PresentationId,
    pub confirmation_id: ConfirmationId,
    pub surface_delivery_certificate_id: SurfaceDeliveryCertificateId,
    pub predecessor_delivery_observation_id: SurfaceDeliveryObservationId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InteractionContinuityError {
    ZeroPrincipal,
    ZeroActionRequest,
    ZeroPresentationContext,
    ZeroInteractionContext,
    ZeroAuthenticationContext,
    ZeroSurfaceProfile,
    ZeroInputAttester,
    ZeroInteractionGeneration,
    ZeroPresentation,
    ZeroConfirmation,
    ZeroSurfaceDeliveryCertificate,
    ZeroPredecessorDeliveryObservation,
    ZeroInputSequence,
    ZeroInputNonce,
    ProfileMismatch,
    PrincipalMismatch,
    ActionMismatch,
    PresentationContextMismatch,
    InteractionContextMismatch,
    AuthenticationContextMismatch,
    SurfaceProfileMismatch,
    InputAttesterMismatch,
    InteractionGenerationMismatch,
    PresentationMismatch,
    ConfirmationMismatch,
    SurfaceDeliveryCertificateMismatch,
    PredecessorDeliveryMismatch,
    SurfaceCertificateMismatch,
    Presentation(ValidationError),
    RenderArtifact(RenderArtifactError),
    Currentness(PresentationCurrentnessError),
    TrustedSurface(TrustedSurfaceError),
}

impl fmt::Display for InteractionContinuityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for InteractionContinuityError {}

impl From<ValidationError> for InteractionContinuityError {
    fn from(value: ValidationError) -> Self {
        Self::Presentation(value)
    }
}

impl From<RenderArtifactError> for InteractionContinuityError {
    fn from(value: RenderArtifactError) -> Self {
        Self::RenderArtifact(value)
    }
}

impl From<PresentationCurrentnessError> for InteractionContinuityError {
    fn from(value: PresentationCurrentnessError) -> Self {
        Self::Currentness(value)
    }
}

impl From<TrustedSurfaceError> for InteractionContinuityError {
    fn from(value: TrustedSurfaceError) -> Self {
        Self::TrustedSurface(value)
    }
}

impl InteractionContinuityProfile {
    pub fn validate(&self) -> Result<(), InteractionContinuityError> {
        if self.principal_id.is_zero() {
            return Err(InteractionContinuityError::ZeroPrincipal);
        }
        if self.action_request_id.is_zero() {
            return Err(InteractionContinuityError::ZeroActionRequest);
        }
        if self.presentation_context_id.is_zero() {
            return Err(InteractionContinuityError::ZeroPresentationContext);
        }
        if self.interaction_context_id.is_zero() {
            return Err(InteractionContinuityError::ZeroInteractionContext);
        }
        if self.authentication_context_id.is_zero() {
            return Err(InteractionContinuityError::ZeroAuthenticationContext);
        }
        if self.surface_profile_id.is_zero() {
            return Err(InteractionContinuityError::ZeroSurfaceProfile);
        }
        if self.input_attester_id.is_zero() {
            return Err(InteractionContinuityError::ZeroInputAttester);
        }
        if self.interaction_generation == 0 {
            return Err(InteractionContinuityError::ZeroInteractionGeneration);
        }
        Ok(())
    }

    pub fn profile_id(
        &self,
    ) -> Result<InteractionContinuityProfileId, InteractionContinuityError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROFILE_DOMAIN);
        hasher.update(self.principal_id.as_bytes());
        hasher.update(self.action_request_id.as_bytes());
        hasher.update(self.presentation_context_id.as_bytes());
        hasher.update(self.interaction_context_id.as_bytes());
        hasher.update(self.authentication_context_id.as_bytes());
        hasher.update(self.surface_profile_id.as_bytes());
        hasher.update(self.input_attester_id.as_bytes());
        hasher.update(&self.interaction_generation.to_le_bytes());
        Ok(InteractionContinuityProfileId(*hasher.finalize().as_bytes()))
    }
}

impl ConfirmationInputObservation {
    pub fn validate(&self) -> Result<(), InteractionContinuityError> {
        if self.continuity_profile_id.is_zero() {
            return Err(InteractionContinuityError::ProfileMismatch);
        }
        if self.principal_id.is_zero() {
            return Err(InteractionContinuityError::ZeroPrincipal);
        }
        if self.action_request_id.is_zero() {
            return Err(InteractionContinuityError::ZeroActionRequest);
        }
        if self.presentation_id.is_zero() {
            return Err(InteractionContinuityError::ZeroPresentation);
        }
        if self.confirmation_id.is_zero() {
            return Err(InteractionContinuityError::ZeroConfirmation);
        }
        if self.presentation_context_id.is_zero() {
            return Err(InteractionContinuityError::ZeroPresentationContext);
        }
        if self.interaction_context_id.is_zero() {
            return Err(InteractionContinuityError::ZeroInteractionContext);
        }
        if self.authentication_context_id.is_zero() {
            return Err(InteractionContinuityError::ZeroAuthenticationContext);
        }
        if self.surface_delivery_certificate_id.is_zero() {
            return Err(InteractionContinuityError::ZeroSurfaceDeliveryCertificate);
        }
        if self.predecessor_delivery_observation_id.is_zero() {
            return Err(InteractionContinuityError::ZeroPredecessorDeliveryObservation);
        }
        if self.input_attester_id.is_zero() {
            return Err(InteractionContinuityError::ZeroInputAttester);
        }
        if self.interaction_generation == 0 {
            return Err(InteractionContinuityError::ZeroInteractionGeneration);
        }
        if self.input_sequence == 0 {
            return Err(InteractionContinuityError::ZeroInputSequence);
        }
        if self.input_nonce.is_zero() {
            return Err(InteractionContinuityError::ZeroInputNonce);
        }
        Ok(())
    }

    pub fn observation_id(
        &self,
    ) -> Result<ConfirmationInputObservationId, InteractionContinuityError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(INPUT_DOMAIN);
        hasher.update(self.continuity_profile_id.as_bytes());
        hasher.update(self.principal_id.as_bytes());
        hasher.update(self.action_request_id.as_bytes());
        hasher.update(self.presentation_id.as_bytes());
        hasher.update(self.confirmation_id.as_bytes());
        hasher.update(self.presentation_context_id.as_bytes());
        hasher.update(self.interaction_context_id.as_bytes());
        hasher.update(self.authentication_context_id.as_bytes());
        hasher.update(self.surface_delivery_certificate_id.as_bytes());
        hasher.update(self.predecessor_delivery_observation_id.as_bytes());
        hasher.update(self.input_attester_id.as_bytes());
        hasher.update(&self.interaction_generation.to_le_bytes());
        hasher.update(&self.input_sequence.to_le_bytes());
        hasher.update(self.input_nonce.as_bytes());
        Ok(ConfirmationInputObservationId(*hasher.finalize().as_bytes()))
    }
}

impl InteractionContinuityCertificate {
    #[allow(clippy::too_many_arguments)]
    pub fn from_input(
        profile: &InteractionContinuityProfile,
        confirmation: &ConfirmationBinding,
        manifest: &PresentationManifest,
        surface_profile: &TrustedSurfaceProfile,
        field_artifacts: &[FieldRenderArtifact],
        warning_artifacts: &[WarningRenderArtifact],
        render_certificate: &RenderArtifactCertificate,
        freshness_certificate: &PresentationFreshnessCertificate,
        presented_snapshot: &PresentationSnapshot,
        current_snapshot: &PresentationSnapshot,
        delivery_observation: &SurfaceDeliveryObservation,
        delivery_certificate: &SurfaceDeliveryCertificate,
        input_observation: &ConfirmationInputObservation,
    ) -> Result<Self, InteractionContinuityError> {
        profile.validate()?;
        manifest.validate()?;
        confirmation.verify_manifest(manifest)?;
        input_observation.validate()?;

        let expected_delivery = SurfaceDeliveryCertificate::from_delivery(
            surface_profile,
            manifest,
            field_artifacts,
            warning_artifacts,
            render_certificate,
            freshness_certificate,
            presented_snapshot,
            current_snapshot,
            delivery_observation,
        )?;
        if delivery_certificate != &expected_delivery {
            return Err(InteractionContinuityError::SurfaceCertificateMismatch);
        }

        let profile_id = profile.profile_id()?;
        let surface_profile_id = surface_profile.profile_id()?;
        let presentation_id = manifest.presentation_id()?;
        let confirmation_id = confirmation.confirmation_id()?;
        let delivery_certificate_id = delivery_certificate.certificate_id()?;
        let delivery_observation_id = delivery_observation.observation_id()?;

        if profile.principal_id != confirmation.principal_id
            || profile.principal_id != manifest.principal_id
        {
            return Err(InteractionContinuityError::PrincipalMismatch);
        }
        if profile.action_request_id != confirmation.action_request_id
            || profile.action_request_id != manifest.action_request_id
        {
            return Err(InteractionContinuityError::ActionMismatch);
        }
        if profile.presentation_context_id != manifest.presentation_context_id
            || profile.presentation_context_id != surface_profile.presentation_context_id
        {
            return Err(InteractionContinuityError::PresentationContextMismatch);
        }
        if profile.interaction_context_id != confirmation.interaction_context_id {
            return Err(InteractionContinuityError::InteractionContextMismatch);
        }
        if profile.authentication_context_id != confirmation.authentication_context_id {
            return Err(InteractionContinuityError::AuthenticationContextMismatch);
        }
        if profile.surface_profile_id != surface_profile_id {
            return Err(InteractionContinuityError::SurfaceProfileMismatch);
        }

        if input_observation.continuity_profile_id != profile_id {
            return Err(InteractionContinuityError::ProfileMismatch);
        }
        if input_observation.principal_id != profile.principal_id {
            return Err(InteractionContinuityError::PrincipalMismatch);
        }
        if input_observation.action_request_id != profile.action_request_id {
            return Err(InteractionContinuityError::ActionMismatch);
        }
        if input_observation.presentation_id != presentation_id {
            return Err(InteractionContinuityError::PresentationMismatch);
        }
        if input_observation.confirmation_id != confirmation_id {
            return Err(InteractionContinuityError::ConfirmationMismatch);
        }
        if input_observation.presentation_context_id != profile.presentation_context_id {
            return Err(InteractionContinuityError::PresentationContextMismatch);
        }
        if input_observation.interaction_context_id != profile.interaction_context_id {
            return Err(InteractionContinuityError::InteractionContextMismatch);
        }
        if input_observation.authentication_context_id != profile.authentication_context_id {
            return Err(InteractionContinuityError::AuthenticationContextMismatch);
        }
        if input_observation.surface_delivery_certificate_id != delivery_certificate_id {
            return Err(InteractionContinuityError::SurfaceDeliveryCertificateMismatch);
        }
        if input_observation.predecessor_delivery_observation_id != delivery_observation_id {
            return Err(InteractionContinuityError::PredecessorDeliveryMismatch);
        }
        if input_observation.input_attester_id != profile.input_attester_id {
            return Err(InteractionContinuityError::InputAttesterMismatch);
        }
        if input_observation.interaction_generation != profile.interaction_generation {
            return Err(InteractionContinuityError::InteractionGenerationMismatch);
        }

        Ok(Self {
            continuity_profile_id: profile_id,
            input_observation_id: input_observation.observation_id()?,
            principal_id: profile.principal_id,
            action_request_id: profile.action_request_id,
            presentation_id,
            confirmation_id,
            surface_delivery_certificate_id: delivery_certificate_id,
            predecessor_delivery_observation_id: delivery_observation_id,
        })
    }

    pub fn certificate_id(
        &self,
    ) -> Result<InteractionContinuityCertificateId, InteractionContinuityError> {
        self.validate_nonzero()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.continuity_profile_id.as_bytes());
        hasher.update(self.input_observation_id.as_bytes());
        hasher.update(self.principal_id.as_bytes());
        hasher.update(self.action_request_id.as_bytes());
        hasher.update(self.presentation_id.as_bytes());
        hasher.update(self.confirmation_id.as_bytes());
        hasher.update(self.surface_delivery_certificate_id.as_bytes());
        hasher.update(self.predecessor_delivery_observation_id.as_bytes());
        Ok(InteractionContinuityCertificateId(*hasher.finalize().as_bytes()))
    }

    fn validate_nonzero(&self) -> Result<(), InteractionContinuityError> {
        if self.continuity_profile_id.is_zero() {
            return Err(InteractionContinuityError::ProfileMismatch);
        }
        if self.input_observation_id.is_zero() {
            return Err(InteractionContinuityError::ProfileMismatch);
        }
        if self.principal_id.is_zero() {
            return Err(InteractionContinuityError::ZeroPrincipal);
        }
        if self.action_request_id.is_zero() {
            return Err(InteractionContinuityError::ZeroActionRequest);
        }
        if self.presentation_id.is_zero() {
            return Err(InteractionContinuityError::ZeroPresentation);
        }
        if self.confirmation_id.is_zero() {
            return Err(InteractionContinuityError::ZeroConfirmation);
        }
        if self.surface_delivery_certificate_id.is_zero() {
            return Err(InteractionContinuityError::ZeroSurfaceDeliveryCertificate);
        }
        if self.predecessor_delivery_observation_id.is_zero() {
            return Err(InteractionContinuityError::ZeroPredecessorDeliveryObservation);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance::{
        ConfirmationNonce, IntentId, LocaleProfileId, MaterialEffect, MaterialField,
        MaterialFieldKind, PresentedField, PresentedWarning, RendererProfileId,
        RenderedCommitment, SemanticCommitment, WarningClass,
    };
    use crate::assurance_materiality::{MaterialityProfileId, PolicyContextId};
    use crate::assurance_presentation_currentness::{
        AuthorityStateRoot, InformationStateRoot, IntentStateRoot, TemporalStateRoot,
        TransactionSnapshotRoot,
    };
    use crate::assurance_render_artifact::{RenderFormatId, RenderModality};
    use crate::assurance_trusted_surface::{
        InteractionNonce, SurfaceAttesterId, SurfaceClass, TrustedSurfaceId,
    };

    fn bytes(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    fn field_artifacts() -> Vec<FieldRenderArtifact> {
        vec![FieldRenderArtifact {
            kind: MaterialFieldKind::ActionClass,
            semantic: SemanticCommitment(bytes(10)),
            modality: RenderModality::Visual,
            format_id: RenderFormatId(bytes(50)),
            payload: b"Transfer".to_vec(),
        }]
    }

    fn warning_artifacts() -> Vec<WarningRenderArtifact> {
        vec![WarningRenderArtifact {
            class: WarningClass::FinancialCost,
            modality: RenderModality::Visual,
            format_id: RenderFormatId(bytes(51)),
            payload: b"This moves real funds.".to_vec(),
        }]
    }

    fn manifest(fields: &[FieldRenderArtifact], warnings: &[WarningRenderArtifact]) -> PresentationManifest {
        let renderer = RendererProfileId(bytes(5));
        let locale = LocaleProfileId(bytes(6));
        PresentationManifest {
            action_request_id: ActionRequestId(bytes(1)),
            intent_id: IntentId(bytes(2)),
            principal_id: PrincipalId(bytes(3)),
            presentation_context_id: PresentationContextId(bytes(4)),
            renderer_profile_id: renderer,
            locale_profile_id: locale,
            presentation_generation: 7,
            material_effect: MaterialEffect {
                fields: fields
                    .iter()
                    .map(|field| MaterialField {
                        kind: field.kind,
                        semantic: field.semantic,
                    })
                    .collect(),
            },
            presented_fields: fields
                .iter()
                .map(|field| PresentedField {
                    kind: field.kind,
                    semantic: field.semantic,
                    rendered: field.rendered_commitment(renderer, locale).unwrap(),
                })
                .collect(),
            warnings: warnings
                .iter()
                .map(|warning| PresentedWarning {
                    class: warning.class,
                    rendered: warning.warning_commitment(renderer, locale).unwrap(),
                })
                .collect(),
        }
    }

    fn snapshot() -> PresentationSnapshot {
        PresentationSnapshot {
            policy_context_id: PolicyContextId(bytes(30)),
            policy_generation: 4,
            action_request_id: ActionRequestId(bytes(1)),
            materiality_profile_id: MaterialityProfileId(bytes(31)),
            intent_state_root: IntentStateRoot(bytes(32)),
            authority_state_root: AuthorityStateRoot(bytes(33)),
            information_state_root: InformationStateRoot(bytes(34)),
            temporal_state_root: TemporalStateRoot(bytes(35)),
            transaction_snapshot_root: TransactionSnapshotRoot(bytes(36)),
        }
    }

    fn surface_profile() -> TrustedSurfaceProfile {
        TrustedSurfaceProfile {
            surface_id: TrustedSurfaceId(bytes(40)),
            attester_id: SurfaceAttesterId(bytes(41)),
            presentation_context_id: PresentationContextId(bytes(4)),
            surface_class: SurfaceClass::VisualDisplay,
            surface_generation: 9,
            allowed_modality_mask: 1,
        }
    }

    fn continuity_profile(surface: &TrustedSurfaceProfile) -> InteractionContinuityProfile {
        InteractionContinuityProfile {
            principal_id: PrincipalId(bytes(3)),
            action_request_id: ActionRequestId(bytes(1)),
            presentation_context_id: PresentationContextId(bytes(4)),
            interaction_context_id: InteractionContextId(bytes(60)),
            authentication_context_id: AuthenticationContextId(bytes(61)),
            surface_profile_id: surface.profile_id().unwrap(),
            input_attester_id: InputAttesterId(bytes(62)),
            interaction_generation: 11,
        }
    }

    #[allow(clippy::type_complexity)]
    fn fixture() -> (
        PresentationManifest,
        Vec<FieldRenderArtifact>,
        Vec<WarningRenderArtifact>,
        RenderArtifactCertificate,
        PresentationFreshnessCertificate,
        PresentationSnapshot,
        TrustedSurfaceProfile,
        SurfaceDeliveryObservation,
        SurfaceDeliveryCertificate,
        ConfirmationBinding,
        InteractionContinuityProfile,
    ) {
        let fields = field_artifacts();
        let warnings = warning_artifacts();
        let m = manifest(&fields, &warnings);
        let render = RenderArtifactCertificate::from_manifest(&m, &fields, &warnings).unwrap();
        let snap = snapshot();
        let fresh = PresentationFreshnessCertificate::from_manifest(&m, &snap).unwrap();
        let surface = surface_profile();
        let delivery_observation = SurfaceDeliveryObservation {
            surface_profile_id: surface.profile_id().unwrap(),
            surface_id: surface.surface_id,
            attester_id: surface.attester_id,
            surface_generation: surface.surface_generation,
            presentation_id: m.presentation_id().unwrap(),
            presentation_context_id: surface.presentation_context_id,
            render_artifact_certificate_id: render.certificate_id().unwrap(),
            artifact_set_root: render.artifact_set_root,
            freshness_certificate_id: fresh.certificate_id().unwrap(),
            delivery_sequence: 1,
            interaction_nonce: InteractionNonce(bytes(42)),
        };
        let delivery = SurfaceDeliveryCertificate::from_delivery(
            &surface,
            &m,
            &fields,
            &warnings,
            &render,
            &fresh,
            &snap,
            &snap,
            &delivery_observation,
        )
        .unwrap();
        let confirmation = ConfirmationBinding::from_manifest(
            &m,
            InteractionContextId(bytes(60)),
            AuthenticationContextId(bytes(61)),
            ConfirmationNonce(bytes(63)),
        )
        .unwrap();
        let continuity = continuity_profile(&surface);
        (
            m,
            fields,
            warnings,
            render,
            fresh,
            snap,
            surface,
            delivery_observation,
            delivery,
            confirmation,
            continuity,
        )
    }

    fn input_observation(
        profile: &InteractionContinuityProfile,
        confirmation: &ConfirmationBinding,
        delivery_observation: &SurfaceDeliveryObservation,
        delivery: &SurfaceDeliveryCertificate,
    ) -> ConfirmationInputObservation {
        ConfirmationInputObservation {
            continuity_profile_id: profile.profile_id().unwrap(),
            principal_id: profile.principal_id,
            action_request_id: profile.action_request_id,
            presentation_id: confirmation.presentation_id,
            confirmation_id: confirmation.confirmation_id().unwrap(),
            presentation_context_id: profile.presentation_context_id,
            interaction_context_id: profile.interaction_context_id,
            authentication_context_id: profile.authentication_context_id,
            surface_delivery_certificate_id: delivery.certificate_id().unwrap(),
            predecessor_delivery_observation_id: delivery_observation.observation_id().unwrap(),
            input_attester_id: profile.input_attester_id,
            interaction_generation: profile.interaction_generation,
            input_sequence: 2,
            input_nonce: InputEventNonce(bytes(64)),
        }
    }

    #[test]
    fn exact_input_continuity_certifies() {
        let (m, fields, warnings, render, fresh, snap, surface, delivery_obs, delivery, confirmation, profile) = fixture();
        let input = input_observation(&profile, &confirmation, &delivery_obs, &delivery);
        let cert = InteractionContinuityCertificate::from_input(
            &profile,
            &confirmation,
            &m,
            &surface,
            &fields,
            &warnings,
            &render,
            &fresh,
            &snap,
            &snap,
            &delivery_obs,
            &delivery,
            &input,
        )
        .unwrap();
        assert_ne!(cert.certificate_id().unwrap(), InteractionContinuityCertificateId::ZERO);
    }

    #[test]
    fn wrong_authentication_context_is_rejected() {
        let (m, fields, warnings, render, fresh, snap, surface, delivery_obs, delivery, confirmation, profile) = fixture();
        let mut input = input_observation(&profile, &confirmation, &delivery_obs, &delivery);
        input.authentication_context_id = AuthenticationContextId(bytes(90));
        assert_eq!(
            InteractionContinuityCertificate::from_input(
                &profile, &confirmation, &m, &surface, &fields, &warnings, &render,
                &fresh, &snap, &snap, &delivery_obs, &delivery, &input,
            ),
            Err(InteractionContinuityError::AuthenticationContextMismatch)
        );
    }

    #[test]
    fn wrong_predecessor_delivery_is_rejected() {
        let (m, fields, warnings, render, fresh, snap, surface, delivery_obs, delivery, confirmation, profile) = fixture();
        let mut input = input_observation(&profile, &confirmation, &delivery_obs, &delivery);
        input.predecessor_delivery_observation_id = SurfaceDeliveryObservationId(bytes(91));
        assert_eq!(
            InteractionContinuityCertificate::from_input(
                &profile, &confirmation, &m, &surface, &fields, &warnings, &render,
                &fresh, &snap, &snap, &delivery_obs, &delivery, &input,
            ),
            Err(InteractionContinuityError::PredecessorDeliveryMismatch)
        );
    }

    #[test]
    fn stale_interaction_generation_is_rejected() {
        let (m, fields, warnings, render, fresh, snap, surface, delivery_obs, delivery, confirmation, profile) = fixture();
        let mut input = input_observation(&profile, &confirmation, &delivery_obs, &delivery);
        input.interaction_generation += 1;
        assert_eq!(
            InteractionContinuityCertificate::from_input(
                &profile, &confirmation, &m, &surface, &fields, &warnings, &render,
                &fresh, &snap, &snap, &delivery_obs, &delivery, &input,
            ),
            Err(InteractionContinuityError::InteractionGenerationMismatch)
        );
    }

    #[test]
    fn wrong_input_attester_is_rejected() {
        let (m, fields, warnings, render, fresh, snap, surface, delivery_obs, delivery, confirmation, profile) = fixture();
        let mut input = input_observation(&profile, &confirmation, &delivery_obs, &delivery);
        input.input_attester_id = InputAttesterId(bytes(92));
        assert_eq!(
            InteractionContinuityCertificate::from_input(
                &profile, &confirmation, &m, &surface, &fields, &warnings, &render,
                &fresh, &snap, &snap, &delivery_obs, &delivery, &input,
            ),
            Err(InteractionContinuityError::InputAttesterMismatch)
        );
    }

    #[test]
    fn wrong_principal_is_rejected() {
        let (m, fields, warnings, render, fresh, snap, surface, delivery_obs, delivery, confirmation, profile) = fixture();
        let mut input = input_observation(&profile, &confirmation, &delivery_obs, &delivery);
        input.principal_id = PrincipalId(bytes(93));
        assert_eq!(
            InteractionContinuityCertificate::from_input(
                &profile, &confirmation, &m, &surface, &fields, &warnings, &render,
                &fresh, &snap, &snap, &delivery_obs, &delivery, &input,
            ),
            Err(InteractionContinuityError::PrincipalMismatch)
        );
    }

    #[test]
    fn input_nonce_changes_observation_identity() {
        let (_m, _fields, _warnings, _render, _fresh, _snap, _surface, delivery_obs, delivery, confirmation, profile) = fixture();
        let a = input_observation(&profile, &confirmation, &delivery_obs, &delivery);
        let mut b = a;
        b.input_nonce = InputEventNonce(bytes(94));
        assert_ne!(a.observation_id().unwrap(), b.observation_id().unwrap());
    }

    #[test]
    fn input_sequence_changes_observation_identity() {
        let (_m, _fields, _warnings, _render, _fresh, _snap, _surface, delivery_obs, delivery, confirmation, profile) = fixture();
        let a = input_observation(&profile, &confirmation, &delivery_obs, &delivery);
        let mut b = a;
        b.input_sequence += 1;
        assert_ne!(a.observation_id().unwrap(), b.observation_id().unwrap());
    }
}
