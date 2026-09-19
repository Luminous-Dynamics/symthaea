// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-CONFIRMUSE-513: current-surface confirmation eligibility and linear use.
//!
//! QUAL-PRESENT-508 defines the exact `ConfirmationBinding`. QUAL-TRUSTEDSURFACE-512
//! binds the exact current presentation/artifacts to one qualified delivery
//! surface. This module composes those objects before creating any consumable
//! confirmation authority, then models a one-way `Available -> Consumed`
//! transition for one exact downstream execution intent.
//!
//! The state transition is a semantic linearity theorem, not a distributed
//! storage theorem. Two processes racing on the same stale `Available` snapshot
//! still require the Transaction Kernel's compare-and-swap/fencing semantics to
//! ensure only one transition commits. Authentication of the principal and of
//! the surface attester also remain upstream propositions.

use core::fmt;

use crate::assurance::{
    ActionRequestId, ConfirmationBinding, ConfirmationId, PresentationId,
    PresentationManifest, PrincipalId, ValidationError,
};
use crate::assurance_presentation_currentness::{
    PresentationCurrentnessError, PresentationFreshnessCertificate,
    PresentationSnapshot,
};
use crate::assurance_render_artifact::{
    FieldRenderArtifact, RenderArtifactCertificate, RenderArtifactError,
    WarningRenderArtifact,
};
use crate::assurance_trusted_surface::{
    SurfaceDeliveryCertificate, SurfaceDeliveryCertificateId, SurfaceDeliveryObservation,
    TrustedSurfaceError, TrustedSurfaceProfile,
};

const ELIGIBILITY_DOMAIN: &[u8] = b"symthaea.presentation.v1/confirmation-eligibility\0";
const USE_STATE_DOMAIN: &[u8] = b"symthaea.presentation.v1/confirmation-use-state\0";
const CONSUMPTION_DOMAIN: &[u8] = b"symthaea.presentation.v1/confirmation-consumption\0";

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

digest_id!(ConfirmationEligibilityCertificateId);
digest_id!(ExecutionIntentId);
digest_id!(ConfirmationUseStateId);
digest_id!(ConsumptionNonce);
digest_id!(ConfirmationConsumptionReceiptId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfirmationEligibilityCertificate {
    pub principal_id: PrincipalId,
    pub action_request_id: ActionRequestId,
    pub presentation_id: PresentationId,
    pub confirmation_id: ConfirmationId,
    pub surface_delivery_certificate_id: SurfaceDeliveryCertificateId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfirmationUseStatus {
    Available,
    Consumed {
        execution_intent_id: ExecutionIntentId,
        consumption_receipt_id: ConfirmationConsumptionReceiptId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfirmationUseState {
    pub eligibility_certificate_id: ConfirmationEligibilityCertificateId,
    pub confirmation_id: ConfirmationId,
    pub surface_delivery_certificate_id: SurfaceDeliveryCertificateId,
    pub resource_generation: u64,
    pub status: ConfirmationUseStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfirmationConsumptionReceipt {
    pub eligibility_certificate_id: ConfirmationEligibilityCertificateId,
    pub confirmation_id: ConfirmationId,
    pub surface_delivery_certificate_id: SurfaceDeliveryCertificateId,
    pub execution_intent_id: ExecutionIntentId,
    pub prior_state_id: ConfirmationUseStateId,
    pub prior_generation: u64,
    pub next_generation: u64,
    pub consumption_nonce: ConsumptionNonce,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfirmationUseError {
    ZeroPrincipal,
    ZeroActionRequest,
    ZeroPresentation,
    ZeroConfirmation,
    ZeroSurfaceDeliveryCertificate,
    ZeroEligibilityCertificate,
    ZeroExecutionIntent,
    ZeroConsumptionNonce,
    ZeroGeneration,
    GenerationOverflow,
    ConfirmationPresentationMismatch,
    ConfirmationActionMismatch,
    ConfirmationPrincipalMismatch,
    SurfacePresentationMismatch,
    SurfaceCertificateMismatch,
    EligibilityMismatch,
    AlreadyConsumed,
    Presentation(ValidationError),
    RenderArtifact(RenderArtifactError),
    Currentness(PresentationCurrentnessError),
    TrustedSurface(TrustedSurfaceError),
}

impl fmt::Display for ConfirmationUseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ConfirmationUseError {}

impl From<ValidationError> for ConfirmationUseError {
    fn from(value: ValidationError) -> Self {
        Self::Presentation(value)
    }
}

impl From<RenderArtifactError> for ConfirmationUseError {
    fn from(value: RenderArtifactError) -> Self {
        Self::RenderArtifact(value)
    }
}

impl From<PresentationCurrentnessError> for ConfirmationUseError {
    fn from(value: PresentationCurrentnessError) -> Self {
        Self::Currentness(value)
    }
}

impl From<TrustedSurfaceError> for ConfirmationUseError {
    fn from(value: TrustedSurfaceError) -> Self {
        Self::TrustedSurface(value)
    }
}

impl ConfirmationEligibilityCertificate {
    #[allow(clippy::too_many_arguments)]
    pub fn from_current_delivery(
        confirmation: &ConfirmationBinding,
        manifest: &PresentationManifest,
        surface_profile: &TrustedSurfaceProfile,
        field_artifacts: &[FieldRenderArtifact],
        warning_artifacts: &[WarningRenderArtifact],
        render_certificate: &RenderArtifactCertificate,
        freshness_certificate: &PresentationFreshnessCertificate,
        presented_snapshot: &PresentationSnapshot,
        current_snapshot: &PresentationSnapshot,
        observation: &SurfaceDeliveryObservation,
        delivery_certificate: &SurfaceDeliveryCertificate,
    ) -> Result<Self, ConfirmationUseError> {
        manifest.validate()?;
        confirmation.verify_manifest(manifest)?;

        let expected_delivery = SurfaceDeliveryCertificate::from_delivery(
            surface_profile,
            manifest,
            field_artifacts,
            warning_artifacts,
            render_certificate,
            freshness_certificate,
            presented_snapshot,
            current_snapshot,
            observation,
        )?;
        if delivery_certificate != &expected_delivery {
            return Err(ConfirmationUseError::SurfaceCertificateMismatch);
        }

        let presentation_id = manifest.presentation_id()?;
        if confirmation.presentation_id != presentation_id {
            return Err(ConfirmationUseError::ConfirmationPresentationMismatch);
        }
        if confirmation.action_request_id != manifest.action_request_id {
            return Err(ConfirmationUseError::ConfirmationActionMismatch);
        }
        if confirmation.principal_id != manifest.principal_id {
            return Err(ConfirmationUseError::ConfirmationPrincipalMismatch);
        }
        if delivery_certificate.presentation_id != presentation_id {
            return Err(ConfirmationUseError::SurfacePresentationMismatch);
        }

        Ok(Self {
            principal_id: manifest.principal_id,
            action_request_id: manifest.action_request_id,
            presentation_id,
            confirmation_id: confirmation.confirmation_id()?,
            surface_delivery_certificate_id: delivery_certificate.certificate_id()?,
        })
    }

    pub fn certificate_id(
        &self,
    ) -> Result<ConfirmationEligibilityCertificateId, ConfirmationUseError> {
        self.validate_nonzero()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(ELIGIBILITY_DOMAIN);
        hasher.update(self.principal_id.as_bytes());
        hasher.update(self.action_request_id.as_bytes());
        hasher.update(self.presentation_id.as_bytes());
        hasher.update(self.confirmation_id.as_bytes());
        hasher.update(self.surface_delivery_certificate_id.as_bytes());
        Ok(ConfirmationEligibilityCertificateId(
            *hasher.finalize().as_bytes(),
        ))
    }

    fn validate_nonzero(&self) -> Result<(), ConfirmationUseError> {
        if self.principal_id.is_zero() {
            return Err(ConfirmationUseError::ZeroPrincipal);
        }
        if self.action_request_id.is_zero() {
            return Err(ConfirmationUseError::ZeroActionRequest);
        }
        if self.presentation_id.is_zero() {
            return Err(ConfirmationUseError::ZeroPresentation);
        }
        if self.confirmation_id.is_zero() {
            return Err(ConfirmationUseError::ZeroConfirmation);
        }
        if self.surface_delivery_certificate_id.is_zero() {
            return Err(ConfirmationUseError::ZeroSurfaceDeliveryCertificate);
        }
        Ok(())
    }
}

impl ConfirmationUseState {
    pub fn available(
        eligibility: &ConfirmationEligibilityCertificate,
    ) -> Result<Self, ConfirmationUseError> {
        let eligibility_certificate_id = eligibility.certificate_id()?;
        Ok(Self {
            eligibility_certificate_id,
            confirmation_id: eligibility.confirmation_id,
            surface_delivery_certificate_id: eligibility.surface_delivery_certificate_id,
            resource_generation: 1,
            status: ConfirmationUseStatus::Available,
        })
    }

    pub fn state_id(&self) -> Result<ConfirmationUseStateId, ConfirmationUseError> {
        self.validate_nonzero()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(USE_STATE_DOMAIN);
        hasher.update(self.eligibility_certificate_id.as_bytes());
        hasher.update(self.confirmation_id.as_bytes());
        hasher.update(self.surface_delivery_certificate_id.as_bytes());
        hasher.update(&self.resource_generation.to_le_bytes());
        match self.status {
            ConfirmationUseStatus::Available => hasher.update(&[1]),
            ConfirmationUseStatus::Consumed {
                execution_intent_id,
                consumption_receipt_id,
            } => {
                hasher.update(&[2]);
                hasher.update(execution_intent_id.as_bytes());
                hasher.update(consumption_receipt_id.as_bytes());
            }
        }
        Ok(ConfirmationUseStateId(*hasher.finalize().as_bytes()))
    }

    pub fn consume(
        &self,
        execution_intent_id: ExecutionIntentId,
        consumption_nonce: ConsumptionNonce,
    ) -> Result<(Self, ConfirmationConsumptionReceipt), ConfirmationUseError> {
        self.validate_nonzero()?;
        if execution_intent_id.is_zero() {
            return Err(ConfirmationUseError::ZeroExecutionIntent);
        }
        if consumption_nonce.is_zero() {
            return Err(ConfirmationUseError::ZeroConsumptionNonce);
        }
        if !matches!(self.status, ConfirmationUseStatus::Available) {
            return Err(ConfirmationUseError::AlreadyConsumed);
        }

        let next_generation = self
            .resource_generation
            .checked_add(1)
            .ok_or(ConfirmationUseError::GenerationOverflow)?;
        let prior_state_id = self.state_id()?;

        let provisional_receipt = ConfirmationConsumptionReceipt {
            eligibility_certificate_id: self.eligibility_certificate_id,
            confirmation_id: self.confirmation_id,
            surface_delivery_certificate_id: self.surface_delivery_certificate_id,
            execution_intent_id,
            prior_state_id,
            prior_generation: self.resource_generation,
            next_generation,
            consumption_nonce,
        };
        let receipt_id = provisional_receipt.receipt_id()?;

        let next = Self {
            eligibility_certificate_id: self.eligibility_certificate_id,
            confirmation_id: self.confirmation_id,
            surface_delivery_certificate_id: self.surface_delivery_certificate_id,
            resource_generation: next_generation,
            status: ConfirmationUseStatus::Consumed {
                execution_intent_id,
                consumption_receipt_id: receipt_id,
            },
        };

        Ok((next, provisional_receipt))
    }

    fn validate_nonzero(&self) -> Result<(), ConfirmationUseError> {
        if self.eligibility_certificate_id.is_zero() {
            return Err(ConfirmationUseError::ZeroEligibilityCertificate);
        }
        if self.confirmation_id.is_zero() {
            return Err(ConfirmationUseError::ZeroConfirmation);
        }
        if self.surface_delivery_certificate_id.is_zero() {
            return Err(ConfirmationUseError::ZeroSurfaceDeliveryCertificate);
        }
        if self.resource_generation == 0 {
            return Err(ConfirmationUseError::ZeroGeneration);
        }
        if let ConfirmationUseStatus::Consumed {
            execution_intent_id,
            consumption_receipt_id,
        } = self.status
        {
            if execution_intent_id.is_zero() {
                return Err(ConfirmationUseError::ZeroExecutionIntent);
            }
            if consumption_receipt_id.is_zero() {
                return Err(ConfirmationUseError::EligibilityMismatch);
            }
        }
        Ok(())
    }
}

impl ConfirmationConsumptionReceipt {
    pub fn receipt_id(
        &self,
    ) -> Result<ConfirmationConsumptionReceiptId, ConfirmationUseError> {
        self.validate_nonzero()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CONSUMPTION_DOMAIN);
        hasher.update(self.eligibility_certificate_id.as_bytes());
        hasher.update(self.confirmation_id.as_bytes());
        hasher.update(self.surface_delivery_certificate_id.as_bytes());
        hasher.update(self.execution_intent_id.as_bytes());
        hasher.update(self.prior_state_id.as_bytes());
        hasher.update(&self.prior_generation.to_le_bytes());
        hasher.update(&self.next_generation.to_le_bytes());
        hasher.update(self.consumption_nonce.as_bytes());
        Ok(ConfirmationConsumptionReceiptId(
            *hasher.finalize().as_bytes(),
        ))
    }

    fn validate_nonzero(&self) -> Result<(), ConfirmationUseError> {
        if self.eligibility_certificate_id.is_zero() {
            return Err(ConfirmationUseError::ZeroEligibilityCertificate);
        }
        if self.confirmation_id.is_zero() {
            return Err(ConfirmationUseError::ZeroConfirmation);
        }
        if self.surface_delivery_certificate_id.is_zero() {
            return Err(ConfirmationUseError::ZeroSurfaceDeliveryCertificate);
        }
        if self.execution_intent_id.is_zero() {
            return Err(ConfirmationUseError::ZeroExecutionIntent);
        }
        if self.prior_state_id.is_zero() {
            return Err(ConfirmationUseError::EligibilityMismatch);
        }
        if self.prior_generation == 0 || self.next_generation == 0 {
            return Err(ConfirmationUseError::ZeroGeneration);
        }
        let expected_next = self
            .prior_generation
            .checked_add(1)
            .ok_or(ConfirmationUseError::GenerationOverflow)?;
        if self.next_generation != expected_next {
            return Err(ConfirmationUseError::EligibilityMismatch);
        }
        if self.consumption_nonce.is_zero() {
            return Err(ConfirmationUseError::ZeroConsumptionNonce);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance::{
        AuthenticationContextId, ConfirmationNonce, IntentId, InteractionContextId,
        LocaleProfileId, MaterialEffect, MaterialField, MaterialFieldKind,
        PresentedField, PresentedWarning, PresentationContextId, RendererProfileId,
        SemanticCommitment, WarningClass,
    };
    use crate::assurance_materiality::{MaterialityProfileId, PolicyContextId};
    use crate::assurance_presentation_currentness::{
        AuthorityStateRoot, InformationStateRoot, IntentStateRoot, TemporalStateRoot,
        TransactionSnapshotRoot,
    };
    use crate::assurance_render_artifact::{
        RenderFormatId, RenderModality, WarningRenderArtifact,
    };
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

    fn surface_profile() -> TrustedSurfaceProfile {
        TrustedSurfaceProfile {
            surface_id: TrustedSurfaceId(bytes(60)),
            attester_id: SurfaceAttesterId(bytes(61)),
            presentation_context_id: PresentationContextId(bytes(4)),
            surface_class: SurfaceClass::VisualDisplay,
            surface_generation: 9,
            allowed_modality_mask: 1,
        }
    }

    #[allow(clippy::type_complexity)]
    fn fixture() -> (
        ConfirmationBinding,
        PresentationManifest,
        TrustedSurfaceProfile,
        Vec<FieldRenderArtifact>,
        Vec<WarningRenderArtifact>,
        RenderArtifactCertificate,
        PresentationFreshnessCertificate,
        PresentationSnapshot,
        SurfaceDeliveryObservation,
        SurfaceDeliveryCertificate,
    ) {
        let fields = field_artifacts();
        let warnings = warning_artifacts();
        let manifest = manifest(&fields, &warnings);
        let confirmation = ConfirmationBinding::from_manifest(
            &manifest,
            InteractionContextId(bytes(71)),
            AuthenticationContextId(bytes(72)),
            ConfirmationNonce(bytes(73)),
        )
        .unwrap();
        let render = RenderArtifactCertificate::from_manifest(&manifest, &fields, &warnings).unwrap();
        let snap = snapshot();
        let freshness = PresentationFreshnessCertificate::from_manifest(&manifest, &snap).unwrap();
        let surface = surface_profile();
        let observation = SurfaceDeliveryObservation {
            surface_profile_id: surface.profile_id().unwrap(),
            surface_id: surface.surface_id,
            attester_id: surface.attester_id,
            surface_generation: surface.surface_generation,
            presentation_id: manifest.presentation_id().unwrap(),
            presentation_context_id: surface.presentation_context_id,
            render_artifact_certificate_id: render.certificate_id().unwrap(),
            artifact_set_root: render.artifact_set_root,
            freshness_certificate_id: freshness.certificate_id().unwrap(),
            delivery_sequence: 1,
            interaction_nonce: InteractionNonce(bytes(74)),
        };
        let delivery = SurfaceDeliveryCertificate::from_delivery(
            &surface,
            &manifest,
            &fields,
            &warnings,
            &render,
            &freshness,
            &snap,
            &snap,
            &observation,
        )
        .unwrap();
        (
            confirmation,
            manifest,
            surface,
            fields,
            warnings,
            render,
            freshness,
            snap,
            observation,
            delivery,
        )
    }

    fn eligibility() -> ConfirmationEligibilityCertificate {
        let (
            confirmation,
            manifest,
            surface,
            fields,
            warnings,
            render,
            freshness,
            snap,
            observation,
            delivery,
        ) = fixture();
        ConfirmationEligibilityCertificate::from_current_delivery(
            &confirmation,
            &manifest,
            &surface,
            &fields,
            &warnings,
            &render,
            &freshness,
            &snap,
            &snap,
            &observation,
            &delivery,
        )
        .unwrap()
    }

    #[test]
    fn exact_current_delivery_creates_eligibility() {
        let eligible = eligibility();
        assert!(!eligible.certificate_id().unwrap().is_zero());
    }

    #[test]
    fn hand_modified_delivery_certificate_is_rejected() {
        let (
            confirmation,
            manifest,
            surface,
            fields,
            warnings,
            render,
            freshness,
            snap,
            observation,
            mut delivery,
        ) = fixture();
        delivery.presentation_id = PresentationId(bytes(90));
        assert_eq!(
            ConfirmationEligibilityCertificate::from_current_delivery(
                &confirmation,
                &manifest,
                &surface,
                &fields,
                &warnings,
                &render,
                &freshness,
                &snap,
                &snap,
                &observation,
                &delivery,
            ),
            Err(ConfirmationUseError::SurfaceCertificateMismatch)
        );
    }

    #[test]
    fn currentness_drift_blocks_eligibility() {
        let (
            confirmation,
            manifest,
            surface,
            fields,
            warnings,
            render,
            freshness,
            presented,
            observation,
            delivery,
        ) = fixture();
        let mut current = presented;
        current.authority_state_root = AuthorityStateRoot(bytes(91));
        assert!(matches!(
            ConfirmationEligibilityCertificate::from_current_delivery(
                &confirmation,
                &manifest,
                &surface,
                &fields,
                &warnings,
                &render,
                &freshness,
                &presented,
                &current,
                &observation,
                &delivery,
            ),
            Err(ConfirmationUseError::TrustedSurface(
                TrustedSurfaceError::Currentness(
                    PresentationCurrentnessError::AuthorityStateChanged
                )
            ))
        ));
    }

    #[test]
    fn available_confirmation_consumes_once_in_state_machine() {
        let eligible = eligibility();
        let state = ConfirmationUseState::available(&eligible).unwrap();
        let (consumed, receipt) = state
            .consume(ExecutionIntentId(bytes(80)), ConsumptionNonce(bytes(81)))
            .unwrap();
        assert_eq!(consumed.resource_generation, 2);
        assert!(!receipt.receipt_id().unwrap().is_zero());
        assert!(matches!(consumed.status, ConfirmationUseStatus::Consumed { .. }));
        assert_eq!(
            consumed.consume(ExecutionIntentId(bytes(80)), ConsumptionNonce(bytes(82))),
            Err(ConfirmationUseError::AlreadyConsumed)
        );
    }

    #[test]
    fn different_execution_intent_changes_consumption_receipt() {
        let eligible = eligibility();
        let state = ConfirmationUseState::available(&eligible).unwrap();
        let (_, a) = state
            .consume(ExecutionIntentId(bytes(80)), ConsumptionNonce(bytes(81)))
            .unwrap();
        let (_, b) = state
            .consume(ExecutionIntentId(bytes(82)), ConsumptionNonce(bytes(81)))
            .unwrap();
        assert_ne!(a.receipt_id().unwrap(), b.receipt_id().unwrap());
    }

    #[test]
    fn consumption_nonce_changes_receipt_identity() {
        let eligible = eligibility();
        let state = ConfirmationUseState::available(&eligible).unwrap();
        let (_, a) = state
            .consume(ExecutionIntentId(bytes(80)), ConsumptionNonce(bytes(81)))
            .unwrap();
        let (_, b) = state
            .consume(ExecutionIntentId(bytes(80)), ConsumptionNonce(bytes(82)))
            .unwrap();
        assert_ne!(a.receipt_id().unwrap(), b.receipt_id().unwrap());
    }

    #[test]
    fn zero_execution_intent_is_rejected() {
        let eligible = eligibility();
        let state = ConfirmationUseState::available(&eligible).unwrap();
        assert_eq!(
            state.consume(ExecutionIntentId::ZERO, ConsumptionNonce(bytes(81))),
            Err(ConfirmationUseError::ZeroExecutionIntent)
        );
    }

    #[test]
    fn malformed_receipt_generation_overflow_is_rejected() {
        let receipt = ConfirmationConsumptionReceipt {
            eligibility_certificate_id: ConfirmationEligibilityCertificateId(bytes(1)),
            confirmation_id: ConfirmationId(bytes(2)),
            surface_delivery_certificate_id: SurfaceDeliveryCertificateId(bytes(3)),
            execution_intent_id: ExecutionIntentId(bytes(4)),
            prior_state_id: ConfirmationUseStateId(bytes(5)),
            prior_generation: u64::MAX,
            next_generation: 1,
            consumption_nonce: ConsumptionNonce(bytes(6)),
        };
        assert_eq!(
            receipt.receipt_id(),
            Err(ConfirmationUseError::GenerationOverflow)
        );
    }
}
