// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-TRUSTEDSURFACE-512: bind exact presentation artifacts to one qualified
//! interaction surface and one attested delivery event.
//!
//! QUAL-RENDERARTIFACT-510 establishes which exact representation bytes were
//! committed by a renderer. QUAL-PRESENTCURRENT-511 establishes that the
//! presentation remains current with respect to the conservatively bound state
//! roots. This module composes those theorems with an action-facing surface
//! profile and an adapter delivery observation.
//!
//! The theorem is deliberately structural. It establishes that a named attester
//! reported delivery of the exact artifact set to the exact qualified surface
//! profile while the presentation was current. It does not establish physical
//! pixels/audio, visibility, absence of occlusion, human perception,
//! comprehension, or authenticity of the attester itself. Attester
//! authentication and physical-delivery assurance remain separate theorems.

use core::fmt;

use crate::assurance::{PresentationContextId, PresentationId, PresentationManifest, ValidationError};
use crate::assurance_presentation_currentness::{
    PresentationCurrentnessError, PresentationFreshnessCertificate,
    PresentationFreshnessCertificateId, PresentationSnapshot,
};
use crate::assurance_render_artifact::{
    FieldRenderArtifact, RenderArtifactCertificate, RenderArtifactCertificateId,
    RenderArtifactError, RenderArtifactSetRoot, RenderModality, WarningRenderArtifact,
};

const PROFILE_DOMAIN: &[u8] = b"symthaea.presentation.v1/trusted-surface-profile\0";
const OBSERVATION_DOMAIN: &[u8] = b"symthaea.presentation.v1/surface-delivery-observation\0";
const CERTIFICATE_DOMAIN: &[u8] = b"symthaea.presentation.v1/surface-delivery-certificate\0";

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

digest_id!(TrustedSurfaceId);
digest_id!(SurfaceAttesterId);
digest_id!(TrustedSurfaceProfileId);
digest_id!(InteractionNonce);
digest_id!(SurfaceDeliveryObservationId);
digest_id!(SurfaceDeliveryCertificateId);

/// Qualified class of interaction endpoint. The class is descriptive; policy
/// still decides which profile is sufficient for a particular action.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum SurfaceClass {
    VisualDisplay = 1,
    Terminal = 2,
    VoiceEndpoint = 3,
    AccessibilityEndpoint = 4,
    MachineEndpoint = 5,
}

/// Exact surface profile in which a presentation may be delivered.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrustedSurfaceProfile {
    pub surface_id: TrustedSurfaceId,
    pub attester_id: SurfaceAttesterId,
    pub presentation_context_id: PresentationContextId,
    pub surface_class: SurfaceClass,
    pub surface_generation: u64,
    /// Bitset over the closed `RenderModality` catalog. Bit zero corresponds to
    /// `Visual`, bit one to `Terminal`, and so on.
    pub allowed_modality_mask: u8,
}

/// Statement supplied by the qualified surface adapter for one delivery event.
/// Authenticating that the statement really came from `attester_id` is outside
/// this tranche and must be established by the deployment's attestation layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceDeliveryObservation {
    pub surface_profile_id: TrustedSurfaceProfileId,
    pub surface_id: TrustedSurfaceId,
    pub attester_id: SurfaceAttesterId,
    pub surface_generation: u64,
    pub presentation_id: PresentationId,
    pub presentation_context_id: PresentationContextId,
    pub render_artifact_certificate_id: RenderArtifactCertificateId,
    pub artifact_set_root: RenderArtifactSetRoot,
    pub freshness_certificate_id: PresentationFreshnessCertificateId,
    pub delivery_sequence: u64,
    pub interaction_nonce: InteractionNonce,
}

/// Derived proof that the exact render/freshness certificates and exact adapter
/// observation all describe the same current presentation and qualified surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceDeliveryCertificate {
    pub observation_id: SurfaceDeliveryObservationId,
    pub surface_profile_id: TrustedSurfaceProfileId,
    pub presentation_id: PresentationId,
    pub render_artifact_certificate_id: RenderArtifactCertificateId,
    pub freshness_certificate_id: PresentationFreshnessCertificateId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrustedSurfaceError {
    ZeroSurfaceId,
    ZeroAttesterId,
    ZeroPresentationContext,
    ZeroSurfaceGeneration,
    EmptyAllowedModalitySet,
    ZeroSurfaceProfile,
    ZeroPresentationId,
    ZeroRenderArtifactCertificate,
    ZeroArtifactSetRoot,
    ZeroFreshnessCertificate,
    ZeroDeliverySequence,
    ZeroInteractionNonce,
    ProfileMismatch,
    SurfaceMismatch,
    AttesterMismatch,
    SurfaceGenerationMismatch,
    PresentationContextMismatch,
    PresentationMismatch,
    RenderArtifactCertificateMismatch,
    ArtifactSetMismatch,
    FreshnessCertificateMismatch,
    UnsupportedModality(RenderModality),
    Presentation(ValidationError),
    RenderArtifact(RenderArtifactError),
    Currentness(PresentationCurrentnessError),
}

impl fmt::Display for TrustedSurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for TrustedSurfaceError {}

impl From<ValidationError> for TrustedSurfaceError {
    fn from(value: ValidationError) -> Self {
        Self::Presentation(value)
    }
}

impl From<RenderArtifactError> for TrustedSurfaceError {
    fn from(value: RenderArtifactError) -> Self {
        Self::RenderArtifact(value)
    }
}

impl From<PresentationCurrentnessError> for TrustedSurfaceError {
    fn from(value: PresentationCurrentnessError) -> Self {
        Self::Currentness(value)
    }
}

impl TrustedSurfaceProfile {
    pub fn validate(&self) -> Result<(), TrustedSurfaceError> {
        if self.surface_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroSurfaceId);
        }
        if self.attester_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroAttesterId);
        }
        if self.presentation_context_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroPresentationContext);
        }
        if self.surface_generation == 0 {
            return Err(TrustedSurfaceError::ZeroSurfaceGeneration);
        }
        if self.allowed_modality_mask == 0 {
            return Err(TrustedSurfaceError::EmptyAllowedModalitySet);
        }
        Ok(())
    }

    pub fn profile_id(&self) -> Result<TrustedSurfaceProfileId, TrustedSurfaceError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROFILE_DOMAIN);
        hasher.update(self.surface_id.as_bytes());
        hasher.update(self.attester_id.as_bytes());
        hasher.update(self.presentation_context_id.as_bytes());
        hasher.update(&[self.surface_class as u8]);
        hasher.update(&self.surface_generation.to_le_bytes());
        hasher.update(&[self.allowed_modality_mask]);
        Ok(TrustedSurfaceProfileId(*hasher.finalize().as_bytes()))
    }

    pub fn allows(&self, modality: RenderModality) -> bool {
        let bit = 1u8 << ((modality as u8) - 1);
        self.allowed_modality_mask & bit != 0
    }

    fn verify_artifact_modalities(
        &self,
        fields: &[FieldRenderArtifact],
        warnings: &[WarningRenderArtifact],
    ) -> Result<(), TrustedSurfaceError> {
        for artifact in fields {
            if !self.allows(artifact.modality) {
                return Err(TrustedSurfaceError::UnsupportedModality(artifact.modality));
            }
        }
        for artifact in warnings {
            if !self.allows(artifact.modality) {
                return Err(TrustedSurfaceError::UnsupportedModality(artifact.modality));
            }
        }
        Ok(())
    }
}

impl SurfaceDeliveryObservation {
    pub fn validate(&self) -> Result<(), TrustedSurfaceError> {
        if self.surface_profile_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroSurfaceProfile);
        }
        if self.surface_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroSurfaceId);
        }
        if self.attester_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroAttesterId);
        }
        if self.surface_generation == 0 {
            return Err(TrustedSurfaceError::ZeroSurfaceGeneration);
        }
        if self.presentation_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroPresentationId);
        }
        if self.presentation_context_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroPresentationContext);
        }
        if self.render_artifact_certificate_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroRenderArtifactCertificate);
        }
        if self.artifact_set_root.is_zero() {
            return Err(TrustedSurfaceError::ZeroArtifactSetRoot);
        }
        if self.freshness_certificate_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroFreshnessCertificate);
        }
        if self.delivery_sequence == 0 {
            return Err(TrustedSurfaceError::ZeroDeliverySequence);
        }
        if self.interaction_nonce.is_zero() {
            return Err(TrustedSurfaceError::ZeroInteractionNonce);
        }
        Ok(())
    }

    pub fn observation_id(&self) -> Result<SurfaceDeliveryObservationId, TrustedSurfaceError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(OBSERVATION_DOMAIN);
        hasher.update(self.surface_profile_id.as_bytes());
        hasher.update(self.surface_id.as_bytes());
        hasher.update(self.attester_id.as_bytes());
        hasher.update(&self.surface_generation.to_le_bytes());
        hasher.update(self.presentation_id.as_bytes());
        hasher.update(self.presentation_context_id.as_bytes());
        hasher.update(self.render_artifact_certificate_id.as_bytes());
        hasher.update(self.artifact_set_root.as_bytes());
        hasher.update(self.freshness_certificate_id.as_bytes());
        hasher.update(&self.delivery_sequence.to_le_bytes());
        hasher.update(self.interaction_nonce.as_bytes());
        Ok(SurfaceDeliveryObservationId(*hasher.finalize().as_bytes()))
    }
}

impl SurfaceDeliveryCertificate {
    #[allow(clippy::too_many_arguments)]
    pub fn from_delivery(
        profile: &TrustedSurfaceProfile,
        manifest: &PresentationManifest,
        field_artifacts: &[FieldRenderArtifact],
        warning_artifacts: &[WarningRenderArtifact],
        render_certificate: &RenderArtifactCertificate,
        freshness_certificate: &PresentationFreshnessCertificate,
        presented_snapshot: &PresentationSnapshot,
        current_snapshot: &PresentationSnapshot,
        observation: &SurfaceDeliveryObservation,
    ) -> Result<Self, TrustedSurfaceError> {
        profile.validate()?;
        manifest.validate()?;
        observation.validate()?;

        if manifest.presentation_context_id != profile.presentation_context_id {
            return Err(TrustedSurfaceError::PresentationContextMismatch);
        }
        profile.verify_artifact_modalities(field_artifacts, warning_artifacts)?;

        render_certificate.verify_manifest(manifest, field_artifacts, warning_artifacts)?;
        freshness_certificate.verify_current(manifest, presented_snapshot, current_snapshot)?;

        let profile_id = profile.profile_id()?;
        let presentation_id = manifest.presentation_id()?;
        let render_certificate_id = render_certificate.certificate_id()?;
        let freshness_certificate_id = freshness_certificate.certificate_id()?;

        if observation.surface_profile_id != profile_id {
            return Err(TrustedSurfaceError::ProfileMismatch);
        }
        if observation.surface_id != profile.surface_id {
            return Err(TrustedSurfaceError::SurfaceMismatch);
        }
        if observation.attester_id != profile.attester_id {
            return Err(TrustedSurfaceError::AttesterMismatch);
        }
        if observation.surface_generation != profile.surface_generation {
            return Err(TrustedSurfaceError::SurfaceGenerationMismatch);
        }
        if observation.presentation_context_id != profile.presentation_context_id {
            return Err(TrustedSurfaceError::PresentationContextMismatch);
        }
        if observation.presentation_id != presentation_id {
            return Err(TrustedSurfaceError::PresentationMismatch);
        }
        if observation.render_artifact_certificate_id != render_certificate_id {
            return Err(TrustedSurfaceError::RenderArtifactCertificateMismatch);
        }
        if observation.artifact_set_root != render_certificate.artifact_set_root {
            return Err(TrustedSurfaceError::ArtifactSetMismatch);
        }
        if observation.freshness_certificate_id != freshness_certificate_id {
            return Err(TrustedSurfaceError::FreshnessCertificateMismatch);
        }

        Ok(Self {
            observation_id: observation.observation_id()?,
            surface_profile_id: profile_id,
            presentation_id,
            render_artifact_certificate_id: render_certificate_id,
            freshness_certificate_id,
        })
    }

    pub fn certificate_id(&self) -> Result<SurfaceDeliveryCertificateId, TrustedSurfaceError> {
        self.validate_nonzero()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.observation_id.as_bytes());
        hasher.update(self.surface_profile_id.as_bytes());
        hasher.update(self.presentation_id.as_bytes());
        hasher.update(self.render_artifact_certificate_id.as_bytes());
        hasher.update(self.freshness_certificate_id.as_bytes());
        Ok(SurfaceDeliveryCertificateId(*hasher.finalize().as_bytes()))
    }

    fn validate_nonzero(&self) -> Result<(), TrustedSurfaceError> {
        if self.observation_id.is_zero() {
            return Err(TrustedSurfaceError::ProfileMismatch);
        }
        if self.surface_profile_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroSurfaceProfile);
        }
        if self.presentation_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroPresentationId);
        }
        if self.render_artifact_certificate_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroRenderArtifactCertificate);
        }
        if self.freshness_certificate_id.is_zero() {
            return Err(TrustedSurfaceError::ZeroFreshnessCertificate);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance::{
        ActionRequestId, IntentId, LocaleProfileId, MaterialEffect, MaterialField,
        MaterialFieldKind, PresentedField, PresentedWarning, PrincipalId, RenderedCommitment,
        RendererProfileId, SemanticCommitment, WarningClass,
    };
    use crate::assurance_materiality::{MaterialityProfileId, PolicyContextId};
    use crate::assurance_presentation_currentness::{
        AuthorityStateRoot, InformationStateRoot, IntentStateRoot, TemporalStateRoot,
        TransactionSnapshotRoot,
    };
    use crate::assurance_render_artifact::{RenderFormatId, WarningRenderArtifact};

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
                    .map(|artifact| MaterialField {
                        kind: artifact.kind,
                        semantic: artifact.semantic,
                    })
                    .collect(),
            },
            presented_fields: fields
                .iter()
                .map(|artifact| PresentedField {
                    kind: artifact.kind,
                    semantic: artifact.semantic,
                    rendered: artifact.rendered_commitment(renderer, locale).unwrap(),
                })
                .collect(),
            warnings: warnings
                .iter()
                .map(|artifact| PresentedWarning {
                    class: artifact.class,
                    rendered: artifact.warning_commitment(renderer, locale).unwrap(),
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

    fn profile() -> TrustedSurfaceProfile {
        TrustedSurfaceProfile {
            surface_id: TrustedSurfaceId(bytes(60)),
            attester_id: SurfaceAttesterId(bytes(61)),
            presentation_context_id: PresentationContextId(bytes(4)),
            surface_class: SurfaceClass::VisualDisplay,
            surface_generation: 9,
            allowed_modality_mask: 1,
        }
    }

    fn fixture() -> (
        PresentationManifest,
        Vec<FieldRenderArtifact>,
        Vec<WarningRenderArtifact>,
        RenderArtifactCertificate,
        PresentationFreshnessCertificate,
        PresentationSnapshot,
    ) {
        let fields = field_artifacts();
        let warnings = warning_artifacts();
        let manifest = manifest(&fields, &warnings);
        let render = RenderArtifactCertificate::from_manifest(&manifest, &fields, &warnings).unwrap();
        let snapshot = snapshot();
        let freshness = PresentationFreshnessCertificate::from_manifest(&manifest, &snapshot).unwrap();
        (manifest, fields, warnings, render, freshness, snapshot)
    }

    fn observation(
        profile: &TrustedSurfaceProfile,
        manifest: &PresentationManifest,
        render: &RenderArtifactCertificate,
        freshness: &PresentationFreshnessCertificate,
    ) -> SurfaceDeliveryObservation {
        SurfaceDeliveryObservation {
            surface_profile_id: profile.profile_id().unwrap(),
            surface_id: profile.surface_id,
            attester_id: profile.attester_id,
            surface_generation: profile.surface_generation,
            presentation_id: manifest.presentation_id().unwrap(),
            presentation_context_id: profile.presentation_context_id,
            render_artifact_certificate_id: render.certificate_id().unwrap(),
            artifact_set_root: render.artifact_set_root,
            freshness_certificate_id: freshness.certificate_id().unwrap(),
            delivery_sequence: 1,
            interaction_nonce: InteractionNonce(bytes(70)),
        }
    }

    #[test]
    fn exact_delivery_binds_exact_surface_and_current_presentation() {
        let (m, fields, warnings, render, freshness, snap) = fixture();
        let profile = profile();
        let obs = observation(&profile, &m, &render, &freshness);
        let cert = SurfaceDeliveryCertificate::from_delivery(
            &profile, &m, &fields, &warnings, &render, &freshness, &snap, &snap, &obs,
        )
        .unwrap();
        assert!(!cert.certificate_id().unwrap().is_zero());
    }

    #[test]
    fn wrong_surface_is_rejected() {
        let (m, fields, warnings, render, freshness, snap) = fixture();
        let profile = profile();
        let mut obs = observation(&profile, &m, &render, &freshness);
        obs.surface_id = TrustedSurfaceId(bytes(90));
        assert_eq!(
            SurfaceDeliveryCertificate::from_delivery(
                &profile, &m, &fields, &warnings, &render, &freshness, &snap, &snap, &obs,
            ),
            Err(TrustedSurfaceError::SurfaceMismatch)
        );
    }

    #[test]
    fn wrong_attester_is_rejected() {
        let (m, fields, warnings, render, freshness, snap) = fixture();
        let profile = profile();
        let mut obs = observation(&profile, &m, &render, &freshness);
        obs.attester_id = SurfaceAttesterId(bytes(91));
        assert_eq!(
            SurfaceDeliveryCertificate::from_delivery(
                &profile, &m, &fields, &warnings, &render, &freshness, &snap, &snap, &obs,
            ),
            Err(TrustedSurfaceError::AttesterMismatch)
        );
    }

    #[test]
    fn stale_surface_generation_is_rejected() {
        let (m, fields, warnings, render, freshness, snap) = fixture();
        let profile = profile();
        let mut obs = observation(&profile, &m, &render, &freshness);
        obs.surface_generation -= 1;
        assert_eq!(
            SurfaceDeliveryCertificate::from_delivery(
                &profile, &m, &fields, &warnings, &render, &freshness, &snap, &snap, &obs,
            ),
            Err(TrustedSurfaceError::SurfaceGenerationMismatch)
        );
    }

    #[test]
    fn wrong_presentation_context_is_rejected() {
        let (mut m, fields, warnings, render, freshness, snap) = fixture();
        m.presentation_context_id = PresentationContextId(bytes(92));
        let profile = profile();
        let obs = observation(&profile, &m, &render, &freshness);
        assert_eq!(
            SurfaceDeliveryCertificate::from_delivery(
                &profile, &m, &fields, &warnings, &render, &freshness, &snap, &snap, &obs,
            ),
            Err(TrustedSurfaceError::PresentationContextMismatch)
        );
    }

    #[test]
    fn unsupported_modality_is_rejected() {
        let (m, mut fields, warnings, _render, freshness, snap) = fixture();
        fields[0].modality = RenderModality::Voice;
        let render = RenderArtifactCertificate::from_manifest(&m, &fields, &warnings);
        assert!(render.is_err());

        let profile = profile();
        assert_eq!(
            profile.verify_artifact_modalities(&fields, &warnings),
            Err(TrustedSurfaceError::UnsupportedModality(RenderModality::Voice))
        );
        let _ = (freshness, snap);
    }

    #[test]
    fn artifact_set_substitution_is_rejected() {
        let (m, fields, warnings, render, freshness, snap) = fixture();
        let profile = profile();
        let mut obs = observation(&profile, &m, &render, &freshness);
        obs.artifact_set_root = RenderArtifactSetRoot(bytes(93));
        assert_eq!(
            SurfaceDeliveryCertificate::from_delivery(
                &profile, &m, &fields, &warnings, &render, &freshness, &snap, &snap, &obs,
            ),
            Err(TrustedSurfaceError::ArtifactSetMismatch)
        );
    }

    #[test]
    fn stale_presentation_state_blocks_delivery_certificate() {
        let (m, fields, warnings, render, freshness, presented) = fixture();
        let mut current = presented;
        current.authority_state_root = AuthorityStateRoot(bytes(94));
        let profile = profile();
        let obs = observation(&profile, &m, &render, &freshness);
        assert!(matches!(
            SurfaceDeliveryCertificate::from_delivery(
                &profile, &m, &fields, &warnings, &render, &freshness,
                &presented, &current, &obs,
            ),
            Err(TrustedSurfaceError::Currentness(
                PresentationCurrentnessError::AuthorityStateChanged
            ))
        ));
    }

    #[test]
    fn nonce_changes_delivery_observation_identity() {
        let (m, _fields, _warnings, render, freshness, _snap) = fixture();
        let profile = profile();
        let a = observation(&profile, &m, &render, &freshness);
        let mut b = a;
        b.interaction_nonce = InteractionNonce(bytes(95));
        assert_ne!(a.observation_id().unwrap(), b.observation_id().unwrap());
    }

    #[test]
    fn delivery_sequence_changes_observation_identity() {
        let (m, _fields, _warnings, render, freshness, _snap) = fixture();
        let profile = profile();
        let a = observation(&profile, &m, &render, &freshness);
        let mut b = a;
        b.delivery_sequence = 2;
        assert_ne!(a.observation_id().unwrap(), b.observation_id().unwrap());
    }
}
