// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-RENDERARTIFACT-510: derive presentation commitments from exact render artifacts.
//!
//! QUAL-PRESENT-508 binds `RenderedCommitment` values into a presentation, but a
//! caller could otherwise invent those 32-byte values without demonstrating what
//! concrete representation they commit to. This module makes the next boundary
//! explicit: exact field/warning render payload bytes, their format identity,
//! modality, semantic identity, renderer profile, and locale profile determine
//! the commitments accepted by a `PresentationManifest`.
//!
//! This still does not prove that a display, speaker, terminal, accessibility
//! surface, or other device actually delivered those bytes to a human. That is a
//! later trusted-surface theorem. Here we establish artifact identity only.

use core::fmt;

use crate::assurance::{
    LocaleProfileId, MaterialFieldKind, PresentationId, PresentationManifest, RenderedCommitment,
    RendererProfileId, SemanticCommitment, ValidationError, WarningClass, WarningCommitment,
};

const FIELD_ARTIFACT_DOMAIN: &[u8] = b"symthaea.presentation.v1/field-render-artifact\0";
const WARNING_ARTIFACT_DOMAIN: &[u8] = b"symthaea.presentation.v1/warning-render-artifact\0";
const ARTIFACT_SET_DOMAIN: &[u8] = b"symthaea.presentation.v1/render-artifact-set\0";
const CERTIFICATE_DOMAIN: &[u8] = b"symthaea.presentation.v1/render-artifact-certificate\0";

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

digest_id!(RenderFormatId);
digest_id!(RenderArtifactSetRoot);
digest_id!(RenderArtifactCertificateId);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum RenderModality {
    Visual = 1,
    Terminal = 2,
    Voice = 3,
    Accessibility = 4,
    MachineCanonical = 5,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldRenderArtifact {
    pub kind: MaterialFieldKind,
    pub semantic: SemanticCommitment,
    pub modality: RenderModality,
    pub format_id: RenderFormatId,
    /// Exact renderer output represented in the semantics named by `format_id`.
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WarningRenderArtifact {
    pub class: WarningClass,
    pub modality: RenderModality,
    pub format_id: RenderFormatId,
    /// Exact renderer output represented in the semantics named by `format_id`.
    pub payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderArtifactCertificate {
    pub presentation_id: PresentationId,
    pub renderer_profile_id: RendererProfileId,
    pub locale_profile_id: LocaleProfileId,
    pub artifact_set_root: RenderArtifactSetRoot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderArtifactError {
    ZeroRendererProfile,
    ZeroLocaleProfile,
    ZeroFormatId,
    ZeroSemanticCommitment(MaterialFieldKind),
    EmptyFieldPayload(MaterialFieldKind),
    EmptyWarningPayload(WarningClass),
    FieldArtifactCountMismatch,
    WarningArtifactCountMismatch,
    FieldKindMismatch {
        expected: MaterialFieldKind,
        actual: MaterialFieldKind,
    },
    FieldSemanticMismatch(MaterialFieldKind),
    FieldCommitmentMismatch(MaterialFieldKind),
    WarningClassMismatch {
        expected: WarningClass,
        actual: WarningClass,
    },
    WarningCommitmentMismatch(WarningClass),
    CertificatePresentationMismatch,
    CertificateRendererMismatch,
    CertificateLocaleMismatch,
    CertificateArtifactSetMismatch,
    Presentation(ValidationError),
}

impl fmt::Display for RenderArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RenderArtifactError {}

impl From<ValidationError> for RenderArtifactError {
    fn from(value: ValidationError) -> Self {
        Self::Presentation(value)
    }
}

impl FieldRenderArtifact {
    pub fn rendered_commitment(
        &self,
        renderer_profile_id: RendererProfileId,
        locale_profile_id: LocaleProfileId,
    ) -> Result<RenderedCommitment, RenderArtifactError> {
        require_profiles(renderer_profile_id, locale_profile_id)?;
        if self.semantic.is_zero() {
            return Err(RenderArtifactError::ZeroSemanticCommitment(self.kind));
        }
        if self.format_id.is_zero() {
            return Err(RenderArtifactError::ZeroFormatId);
        }
        if self.payload.is_empty() {
            return Err(RenderArtifactError::EmptyFieldPayload(self.kind));
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(FIELD_ARTIFACT_DOMAIN);
        hasher.update(renderer_profile_id.as_bytes());
        hasher.update(locale_profile_id.as_bytes());
        hasher.update(&[self.modality as u8]);
        hasher.update(&[self.kind as u8]);
        hasher.update(self.semantic.as_bytes());
        hasher.update(self.format_id.as_bytes());
        put_bytes(&mut hasher, &self.payload);
        Ok(RenderedCommitment(*hasher.finalize().as_bytes()))
    }
}

impl WarningRenderArtifact {
    pub fn warning_commitment(
        &self,
        renderer_profile_id: RendererProfileId,
        locale_profile_id: LocaleProfileId,
    ) -> Result<WarningCommitment, RenderArtifactError> {
        require_profiles(renderer_profile_id, locale_profile_id)?;
        if self.format_id.is_zero() {
            return Err(RenderArtifactError::ZeroFormatId);
        }
        if self.payload.is_empty() {
            return Err(RenderArtifactError::EmptyWarningPayload(self.class));
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(WARNING_ARTIFACT_DOMAIN);
        hasher.update(renderer_profile_id.as_bytes());
        hasher.update(locale_profile_id.as_bytes());
        hasher.update(&[self.modality as u8]);
        hasher.update(&[self.class as u8]);
        hasher.update(self.format_id.as_bytes());
        put_bytes(&mut hasher, &self.payload);
        Ok(WarningCommitment(*hasher.finalize().as_bytes()))
    }
}

impl RenderArtifactCertificate {
    /// Establishes that every renderer commitment in the exact presentation was
    /// derived from the supplied exact render artifacts.
    pub fn from_manifest(
        manifest: &PresentationManifest,
        field_artifacts: &[FieldRenderArtifact],
        warning_artifacts: &[WarningRenderArtifact],
    ) -> Result<Self, RenderArtifactError> {
        manifest.validate()?;
        if field_artifacts.len() != manifest.presented_fields.len() {
            return Err(RenderArtifactError::FieldArtifactCountMismatch);
        }
        if warning_artifacts.len() != manifest.warnings.len() {
            return Err(RenderArtifactError::WarningArtifactCountMismatch);
        }

        for (presented, artifact) in manifest.presented_fields.iter().zip(field_artifacts) {
            if presented.kind != artifact.kind {
                return Err(RenderArtifactError::FieldKindMismatch {
                    expected: presented.kind,
                    actual: artifact.kind,
                });
            }
            if presented.semantic != artifact.semantic {
                return Err(RenderArtifactError::FieldSemanticMismatch(presented.kind));
            }
            let actual = artifact.rendered_commitment(
                manifest.renderer_profile_id,
                manifest.locale_profile_id,
            )?;
            if presented.rendered != actual {
                return Err(RenderArtifactError::FieldCommitmentMismatch(
                    presented.kind,
                ));
            }
        }

        for (presented, artifact) in manifest.warnings.iter().zip(warning_artifacts) {
            if presented.class != artifact.class {
                return Err(RenderArtifactError::WarningClassMismatch {
                    expected: presented.class,
                    actual: artifact.class,
                });
            }
            let actual = artifact.warning_commitment(
                manifest.renderer_profile_id,
                manifest.locale_profile_id,
            )?;
            if presented.rendered != actual {
                return Err(RenderArtifactError::WarningCommitmentMismatch(
                    presented.class,
                ));
            }
        }

        let presentation_id = manifest.presentation_id()?;
        let artifact_set_root = compute_artifact_set_root(
            manifest.renderer_profile_id,
            manifest.locale_profile_id,
            presentation_id,
            field_artifacts,
            warning_artifacts,
        )?;

        Ok(Self {
            presentation_id,
            renderer_profile_id: manifest.renderer_profile_id,
            locale_profile_id: manifest.locale_profile_id,
            artifact_set_root,
        })
    }

    pub fn certificate_id(&self) -> Result<RenderArtifactCertificateId, RenderArtifactError> {
        self.validate_nonzero()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.presentation_id.as_bytes());
        hasher.update(self.renderer_profile_id.as_bytes());
        hasher.update(self.locale_profile_id.as_bytes());
        hasher.update(self.artifact_set_root.as_bytes());
        Ok(RenderArtifactCertificateId(*hasher.finalize().as_bytes()))
    }

    /// Recomputes all artifact commitments and requires the exact same certificate.
    pub fn verify_manifest(
        &self,
        manifest: &PresentationManifest,
        field_artifacts: &[FieldRenderArtifact],
        warning_artifacts: &[WarningRenderArtifact],
    ) -> Result<(), RenderArtifactError> {
        self.validate_nonzero()?;
        let expected = Self::from_manifest(manifest, field_artifacts, warning_artifacts)?;
        if self.presentation_id != expected.presentation_id {
            return Err(RenderArtifactError::CertificatePresentationMismatch);
        }
        if self.renderer_profile_id != expected.renderer_profile_id {
            return Err(RenderArtifactError::CertificateRendererMismatch);
        }
        if self.locale_profile_id != expected.locale_profile_id {
            return Err(RenderArtifactError::CertificateLocaleMismatch);
        }
        if self.artifact_set_root != expected.artifact_set_root {
            return Err(RenderArtifactError::CertificateArtifactSetMismatch);
        }
        Ok(())
    }

    fn validate_nonzero(&self) -> Result<(), RenderArtifactError> {
        if self.presentation_id.is_zero() {
            return Err(RenderArtifactError::CertificatePresentationMismatch);
        }
        if self.renderer_profile_id.is_zero() {
            return Err(RenderArtifactError::ZeroRendererProfile);
        }
        if self.locale_profile_id.is_zero() {
            return Err(RenderArtifactError::ZeroLocaleProfile);
        }
        if self.artifact_set_root.is_zero() {
            return Err(RenderArtifactError::CertificateArtifactSetMismatch);
        }
        Ok(())
    }
}

fn compute_artifact_set_root(
    renderer_profile_id: RendererProfileId,
    locale_profile_id: LocaleProfileId,
    presentation_id: PresentationId,
    field_artifacts: &[FieldRenderArtifact],
    warning_artifacts: &[WarningRenderArtifact],
) -> Result<RenderArtifactSetRoot, RenderArtifactError> {
    require_profiles(renderer_profile_id, locale_profile_id)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(ARTIFACT_SET_DOMAIN);
    hasher.update(renderer_profile_id.as_bytes());
    hasher.update(locale_profile_id.as_bytes());
    hasher.update(presentation_id.as_bytes());
    put_len(&mut hasher, field_artifacts.len());
    for artifact in field_artifacts {
        hasher.update(
            artifact
                .rendered_commitment(renderer_profile_id, locale_profile_id)?
                .as_bytes(),
        );
    }
    put_len(&mut hasher, warning_artifacts.len());
    for artifact in warning_artifacts {
        hasher.update(
            artifact
                .warning_commitment(renderer_profile_id, locale_profile_id)?
                .as_bytes(),
        );
    }
    Ok(RenderArtifactSetRoot(*hasher.finalize().as_bytes()))
}

fn require_profiles(
    renderer_profile_id: RendererProfileId,
    locale_profile_id: LocaleProfileId,
) -> Result<(), RenderArtifactError> {
    if renderer_profile_id.is_zero() {
        return Err(RenderArtifactError::ZeroRendererProfile);
    }
    if locale_profile_id.is_zero() {
        return Err(RenderArtifactError::ZeroLocaleProfile);
    }
    Ok(())
}

fn put_len(hasher: &mut blake3::Hasher, len: usize) {
    hasher.update(&(len as u64).to_le_bytes());
}

fn put_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    put_len(hasher, bytes.len());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance::{
        ActionRequestId, IntentId, MaterialEffect, MaterialField, PresentedField,
        PresentedWarning, PresentationContextId, PrincipalId,
    };

    fn bytes(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    fn renderer() -> RendererProfileId {
        RendererProfileId(bytes(5))
    }

    fn locale() -> LocaleProfileId {
        LocaleProfileId(bytes(6))
    }

    fn format(seed: u8) -> RenderFormatId {
        RenderFormatId(bytes(seed))
    }

    fn field_artifacts() -> Vec<FieldRenderArtifact> {
        vec![
            FieldRenderArtifact {
                kind: MaterialFieldKind::ActionClass,
                semantic: SemanticCommitment(bytes(10)),
                modality: RenderModality::Visual,
                format_id: format(50),
                payload: b"Transfer".to_vec(),
            },
            FieldRenderArtifact {
                kind: MaterialFieldKind::Destination,
                semantic: SemanticCommitment(bytes(11)),
                modality: RenderModality::Visual,
                format_id: format(50),
                payload: b"Alice / account 4821".to_vec(),
            },
            FieldRenderArtifact {
                kind: MaterialFieldKind::Amount,
                semantic: SemanticCommitment(bytes(12)),
                modality: RenderModality::Visual,
                format_id: format(50),
                payload: b"$100.00 USD".to_vec(),
            },
        ]
    }

    fn warning_artifacts() -> Vec<WarningRenderArtifact> {
        vec![WarningRenderArtifact {
            class: WarningClass::FinancialCost,
            modality: RenderModality::Visual,
            format_id: format(51),
            payload: b"This transfer moves real funds.".to_vec(),
        }]
    }

    fn manifest(
        fields: &[FieldRenderArtifact],
        warnings: &[WarningRenderArtifact],
    ) -> PresentationManifest {
        PresentationManifest {
            action_request_id: ActionRequestId(bytes(1)),
            intent_id: IntentId(bytes(2)),
            principal_id: PrincipalId(bytes(3)),
            presentation_context_id: PresentationContextId(bytes(4)),
            renderer_profile_id: renderer(),
            locale_profile_id: locale(),
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
                    rendered: field.rendered_commitment(renderer(), locale()).unwrap(),
                })
                .collect(),
            warnings: warnings
                .iter()
                .map(|warning| PresentedWarning {
                    class: warning.class,
                    rendered: warning.warning_commitment(renderer(), locale()).unwrap(),
                })
                .collect(),
        }
    }

    #[test]
    fn exact_artifacts_certify_exact_manifest() {
        let fields = field_artifacts();
        let warnings = warning_artifacts();
        let m = manifest(&fields, &warnings);
        let cert = RenderArtifactCertificate::from_manifest(&m, &fields, &warnings).unwrap();
        assert!(cert.verify_manifest(&m, &fields, &warnings).is_ok());
    }

    #[test]
    fn payload_drift_changes_field_commitment() {
        let mut changed = field_artifacts();
        let original = changed[2]
            .rendered_commitment(renderer(), locale())
            .unwrap();
        changed[2].payload = b"$1,000.00 USD".to_vec();
        assert_ne!(
            original,
            changed[2]
                .rendered_commitment(renderer(), locale())
                .unwrap()
        );
    }

    #[test]
    fn semantic_drift_changes_field_commitment() {
        let mut changed = field_artifacts();
        let original = changed[1]
            .rendered_commitment(renderer(), locale())
            .unwrap();
        changed[1].semantic = SemanticCommitment(bytes(99));
        assert_ne!(
            original,
            changed[1]
                .rendered_commitment(renderer(), locale())
                .unwrap()
        );
    }

    #[test]
    fn format_drift_changes_field_commitment() {
        let mut changed = field_artifacts();
        let original = changed[0]
            .rendered_commitment(renderer(), locale())
            .unwrap();
        changed[0].format_id = format(99);
        assert_ne!(
            original,
            changed[0]
                .rendered_commitment(renderer(), locale())
                .unwrap()
        );
    }

    #[test]
    fn modality_drift_changes_field_commitment() {
        let mut changed = field_artifacts();
        let original = changed[0]
            .rendered_commitment(renderer(), locale())
            .unwrap();
        changed[0].modality = RenderModality::Voice;
        assert_ne!(
            original,
            changed[0]
                .rendered_commitment(renderer(), locale())
                .unwrap()
        );
    }

    #[test]
    fn omitted_artifact_is_rejected() {
        let fields = field_artifacts();
        let warnings = warning_artifacts();
        let m = manifest(&fields, &warnings);
        assert_eq!(
            RenderArtifactCertificate::from_manifest(&m, &fields[..2], &warnings),
            Err(RenderArtifactError::FieldArtifactCountMismatch)
        );
    }

    #[test]
    fn field_payload_drift_is_rejected_against_frozen_manifest() {
        let fields = field_artifacts();
        let warnings = warning_artifacts();
        let m = manifest(&fields, &warnings);
        let mut changed = fields.clone();
        changed[2].payload = b"$1,000.00 USD".to_vec();
        assert_eq!(
            RenderArtifactCertificate::from_manifest(&m, &changed, &warnings),
            Err(RenderArtifactError::FieldCommitmentMismatch(
                MaterialFieldKind::Amount
            ))
        );
    }

    #[test]
    fn warning_payload_drift_is_rejected_against_frozen_manifest() {
        let fields = field_artifacts();
        let warnings = warning_artifacts();
        let m = manifest(&fields, &warnings);
        let mut changed = warnings.clone();
        changed[0].payload = b"No material consequence.".to_vec();
        assert_eq!(
            RenderArtifactCertificate::from_manifest(&m, &fields, &changed),
            Err(RenderArtifactError::WarningCommitmentMismatch(
                WarningClass::FinancialCost
            ))
        );
    }

    #[test]
    fn locale_changes_artifact_commitment() {
        let fields = field_artifacts();
        let original = fields[2]
            .rendered_commitment(renderer(), locale())
            .unwrap();
        assert_ne!(
            original,
            fields[2]
                .rendered_commitment(renderer(), LocaleProfileId(bytes(77)))
                .unwrap()
        );
    }

    #[test]
    fn certificate_binds_exact_presentation() {
        let fields = field_artifacts();
        let warnings = warning_artifacts();
        let m = manifest(&fields, &warnings);
        let cert = RenderArtifactCertificate::from_manifest(&m, &fields, &warnings).unwrap();
        let mut changed = m.clone();
        changed.presentation_generation += 1;
        assert_eq!(
            cert.verify_manifest(&changed, &fields, &warnings),
            Err(RenderArtifactError::CertificatePresentationMismatch)
        );
    }
}
