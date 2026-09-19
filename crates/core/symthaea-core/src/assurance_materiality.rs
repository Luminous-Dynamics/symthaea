// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-MATERIALEFFECT-509: policy-defined material-effect completeness.
//!
//! QUAL-PRESENT-508 proves that declared material semantics are bound faithfully
//! into a presentation. This module closes the immediately preceding omission
//! boundary: an upstream policy profile declares which semantic fields and
//! warnings are required for one exact action, and a certificate is valid only
//! when the exact action effect satisfies that exact policy profile.
//!
//! This module deliberately does not decide which effects are material. That is
//! policy. It also does not prove renderer fidelity, localization equivalence,
//! human comprehension, authentication strength, or single-use confirmation.

use core::fmt;

use crate::assurance::{
    ActionRequestId, MaterialEffect, MaterialEffectRoot, MaterialFieldKind,
    PresentationManifest, SemanticCommitment, ValidationError, WarningClass,
};

const PROFILE_DOMAIN: &[u8] = b"symthaea.presentation.v1/materiality-profile\0";
const CERTIFICATE_DOMAIN: &[u8] = b"symthaea.presentation.v1/material-effect-certificate\0";

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

digest_id!(PolicyContextId);
digest_id!(MaterialityProfileId);
digest_id!(MaterialEffectCertificateId);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialityProfile {
    /// Exact policy/trust-domain context in which the generation is meaningful.
    pub policy_context_id: PolicyContextId,
    /// Active policy generation that gave this profile meaning.
    pub policy_generation: u64,
    /// Exact action for which this derived profile applies.
    pub action_request_id: ActionRequestId,
    /// Canonical semantic identity of the action/effect class this profile covers.
    pub effect_class: SemanticCommitment,
    /// Strictly ascending canonical set of fields that must be present.
    pub required_fields: Vec<MaterialFieldKind>,
    /// Strictly ascending canonical set of warnings that must be presented.
    pub required_warnings: Vec<WarningClass>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialEffectCertificate {
    pub action_request_id: ActionRequestId,
    pub policy_context_id: PolicyContextId,
    pub materiality_profile_id: MaterialityProfileId,
    pub material_effect_root: MaterialEffectRoot,
    pub policy_generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialityError {
    ZeroActionRequest,
    ZeroPolicyContext,
    ZeroEffectClass,
    ZeroPolicyGeneration,
    EmptyRequiredFieldSet,
    ActionClassNotRequired,
    NonCanonicalRequiredFieldOrder,
    DuplicateRequiredField(MaterialFieldKind),
    NonCanonicalRequiredWarningOrder,
    DuplicateRequiredWarning(WarningClass),
    MissingRequiredField(MaterialFieldKind),
    EffectClassMismatch,
    MissingRequiredWarning(WarningClass),
    ProfileMismatch,
    ProfileActionMismatch,
    PolicyContextMismatch,
    CertificateActionMismatch,
    CertificateEffectMismatch,
    CertificatePolicyContextMismatch,
    CertificatePolicyGenerationMismatch,
    StaleMaterialityProfile {
        profile_generation: u64,
        current_generation: u64,
    },
    Presentation(ValidationError),
}

impl fmt::Display for MaterialityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MaterialityError {}

impl From<ValidationError> for MaterialityError {
    fn from(value: ValidationError) -> Self {
        Self::Presentation(value)
    }
}

impl MaterialityProfile {
    pub fn validate(&self) -> Result<(), MaterialityError> {
        if self.policy_context_id.is_zero() {
            return Err(MaterialityError::ZeroPolicyContext);
        }
        if self.policy_generation == 0 {
            return Err(MaterialityError::ZeroPolicyGeneration);
        }
        if self.action_request_id.is_zero() {
            return Err(MaterialityError::ZeroActionRequest);
        }
        if self.effect_class.is_zero() {
            return Err(MaterialityError::ZeroEffectClass);
        }
        if self.required_fields.is_empty() {
            return Err(MaterialityError::EmptyRequiredFieldSet);
        }
        validate_required_field_order(&self.required_fields)?;
        validate_required_warning_order(&self.required_warnings)?;
        if !self
            .required_fields
            .contains(&MaterialFieldKind::ActionClass)
        {
            return Err(MaterialityError::ActionClassNotRequired);
        }
        Ok(())
    }

    pub fn profile_id(&self) -> Result<MaterialityProfileId, MaterialityError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROFILE_DOMAIN);
        hasher.update(self.policy_context_id.as_bytes());
        hasher.update(&self.policy_generation.to_le_bytes());
        hasher.update(self.action_request_id.as_bytes());
        hasher.update(self.effect_class.as_bytes());
        put_len(&mut hasher, self.required_fields.len());
        for field in &self.required_fields {
            hasher.update(&[*field as u8]);
        }
        put_len(&mut hasher, self.required_warnings.len());
        for warning in &self.required_warnings {
            hasher.update(&[*warning as u8]);
        }
        Ok(MaterialityProfileId(*hasher.finalize().as_bytes()))
    }

    /// Establishes that this exact action's declared material effect is complete
    /// with respect to this exact action-specific policy profile.
    pub fn certify(
        &self,
        material_effect: &MaterialEffect,
    ) -> Result<MaterialEffectCertificate, MaterialityError> {
        self.validate()?;
        material_effect.validate()?;

        let action_class = material_effect
            .fields
            .iter()
            .find(|field| field.kind == MaterialFieldKind::ActionClass)
            .ok_or(MaterialityError::MissingRequiredField(
                MaterialFieldKind::ActionClass,
            ))?;
        if action_class.semantic != self.effect_class {
            return Err(MaterialityError::EffectClassMismatch);
        }

        for required in &self.required_fields {
            if !material_effect
                .fields
                .iter()
                .any(|field| field.kind == *required)
            {
                return Err(MaterialityError::MissingRequiredField(*required));
            }
        }

        Ok(MaterialEffectCertificate {
            action_request_id: self.action_request_id,
            policy_context_id: self.policy_context_id,
            materiality_profile_id: self.profile_id()?,
            material_effect_root: material_effect.root()?,
            policy_generation: self.policy_generation,
        })
    }
}

impl MaterialEffectCertificate {
    pub fn certificate_id(&self) -> Result<MaterialEffectCertificateId, MaterialityError> {
        self.validate_nonzero()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CERTIFICATE_DOMAIN);
        hasher.update(self.action_request_id.as_bytes());
        hasher.update(self.policy_context_id.as_bytes());
        hasher.update(self.materiality_profile_id.as_bytes());
        hasher.update(self.material_effect_root.as_bytes());
        hasher.update(&self.policy_generation.to_le_bytes());
        Ok(MaterialEffectCertificateId(*hasher.finalize().as_bytes()))
    }

    /// Revalidates the completeness certificate against the exact presentation
    /// and an externally established active policy context/generation.
    pub fn verify_manifest(
        &self,
        profile: &MaterialityProfile,
        manifest: &PresentationManifest,
        current_policy_context_id: PolicyContextId,
        current_policy_generation: u64,
    ) -> Result<(), MaterialityError> {
        self.validate_nonzero()?;
        profile.validate()?;
        manifest.validate()?;

        if current_policy_context_id.is_zero() {
            return Err(MaterialityError::ZeroPolicyContext);
        }
        if profile.policy_context_id != current_policy_context_id {
            return Err(MaterialityError::PolicyContextMismatch);
        }
        if profile.policy_generation != current_policy_generation {
            return Err(MaterialityError::StaleMaterialityProfile {
                profile_generation: profile.policy_generation,
                current_generation: current_policy_generation,
            });
        }
        if profile.action_request_id != manifest.action_request_id {
            return Err(MaterialityError::ProfileActionMismatch);
        }
        if self.policy_context_id != profile.policy_context_id {
            return Err(MaterialityError::CertificatePolicyContextMismatch);
        }
        if self.policy_generation != profile.policy_generation {
            return Err(MaterialityError::CertificatePolicyGenerationMismatch);
        }
        if self.materiality_profile_id != profile.profile_id()? {
            return Err(MaterialityError::ProfileMismatch);
        }
        if self.action_request_id != manifest.action_request_id {
            return Err(MaterialityError::CertificateActionMismatch);
        }
        if self.material_effect_root != manifest.material_effect.root()? {
            return Err(MaterialityError::CertificateEffectMismatch);
        }

        let expected = profile.certify(&manifest.material_effect)?;
        if self != &expected {
            return Err(MaterialityError::ProfileMismatch);
        }

        for required in &profile.required_warnings {
            if !manifest
                .warnings
                .iter()
                .any(|warning| warning.class == *required)
            {
                return Err(MaterialityError::MissingRequiredWarning(*required));
            }
        }

        Ok(())
    }

    fn validate_nonzero(&self) -> Result<(), MaterialityError> {
        if self.action_request_id.is_zero() {
            return Err(MaterialityError::ZeroActionRequest);
        }
        if self.policy_context_id.is_zero() {
            return Err(MaterialityError::ZeroPolicyContext);
        }
        if self.materiality_profile_id.is_zero() {
            return Err(MaterialityError::ProfileMismatch);
        }
        if self.material_effect_root.is_zero() {
            return Err(MaterialityError::CertificateEffectMismatch);
        }
        if self.policy_generation == 0 {
            return Err(MaterialityError::ZeroPolicyGeneration);
        }
        Ok(())
    }
}

fn validate_required_field_order(fields: &[MaterialFieldKind]) -> Result<(), MaterialityError> {
    let mut previous = None;
    for field in fields {
        let tag = *field as u8;
        if let Some(prev) = previous {
            if tag < prev {
                return Err(MaterialityError::NonCanonicalRequiredFieldOrder);
            }
            if tag == prev {
                return Err(MaterialityError::DuplicateRequiredField(*field));
            }
        }
        previous = Some(tag);
    }
    Ok(())
}

fn validate_required_warning_order(warnings: &[WarningClass]) -> Result<(), MaterialityError> {
    let mut previous = None;
    for warning in warnings {
        let tag = *warning as u8;
        if let Some(prev) = previous {
            if tag < prev {
                return Err(MaterialityError::NonCanonicalRequiredWarningOrder);
            }
            if tag == prev {
                return Err(MaterialityError::DuplicateRequiredWarning(*warning));
            }
        }
        previous = Some(tag);
    }
    Ok(())
}

fn put_len(hasher: &mut blake3::Hasher, len: usize) {
    hasher.update(&(len as u64).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance::{
        AuthenticationContextId, ConfirmationBinding, ConfirmationNonce, InteractionContextId,
        LocaleProfileId, MaterialField, PresentedField, PresentedWarning, PresentationContextId,
        PrincipalId, RenderedCommitment, RendererProfileId, WarningCommitment,
    };

    fn bytes(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    fn policy_context() -> PolicyContextId {
        PolicyContextId(bytes(60))
    }

    fn effect_class() -> SemanticCommitment {
        SemanticCommitment(bytes(10))
    }

    fn action_request() -> ActionRequestId {
        ActionRequestId(bytes(1))
    }

    fn profile() -> MaterialityProfile {
        MaterialityProfile {
            policy_context_id: policy_context(),
            policy_generation: 7,
            action_request_id: action_request(),
            effect_class: effect_class(),
            required_fields: vec![
                MaterialFieldKind::ActionClass,
                MaterialFieldKind::Destination,
                MaterialFieldKind::Amount,
            ],
            required_warnings: vec![WarningClass::FinancialCost],
        }
    }

    fn manifest() -> PresentationManifest {
        PresentationManifest {
            action_request_id: action_request(),
            intent_id: crate::assurance::IntentId(bytes(2)),
            principal_id: PrincipalId(bytes(3)),
            presentation_context_id: PresentationContextId(bytes(4)),
            renderer_profile_id: RendererProfileId(bytes(5)),
            locale_profile_id: LocaleProfileId(bytes(6)),
            presentation_generation: 9,
            material_effect: MaterialEffect {
                fields: vec![
                    MaterialField {
                        kind: MaterialFieldKind::ActionClass,
                        semantic: effect_class(),
                    },
                    MaterialField {
                        kind: MaterialFieldKind::Destination,
                        semantic: SemanticCommitment(bytes(11)),
                    },
                    MaterialField {
                        kind: MaterialFieldKind::Amount,
                        semantic: SemanticCommitment(bytes(12)),
                    },
                ],
            },
            presented_fields: vec![
                PresentedField {
                    kind: MaterialFieldKind::ActionClass,
                    semantic: effect_class(),
                    rendered: RenderedCommitment(bytes(20)),
                },
                PresentedField {
                    kind: MaterialFieldKind::Destination,
                    semantic: SemanticCommitment(bytes(11)),
                    rendered: RenderedCommitment(bytes(21)),
                },
                PresentedField {
                    kind: MaterialFieldKind::Amount,
                    semantic: SemanticCommitment(bytes(12)),
                    rendered: RenderedCommitment(bytes(22)),
                },
            ],
            warnings: vec![PresentedWarning {
                class: WarningClass::FinancialCost,
                rendered: WarningCommitment(bytes(30)),
            }],
        }
    }

    #[test]
    fn stable_profile_has_stable_identity() {
        assert_eq!(profile().profile_id().unwrap(), profile().profile_id().unwrap());
    }

    #[test]
    fn action_class_must_be_required_by_profile() {
        let mut p = profile();
        p.required_fields.remove(0);
        assert_eq!(p.validate(), Err(MaterialityError::ActionClassNotRequired));
    }

    #[test]
    fn noncanonical_required_field_order_is_rejected() {
        let mut p = profile();
        p.required_fields.swap(0, 1);
        assert_eq!(
            p.validate(),
            Err(MaterialityError::NonCanonicalRequiredFieldOrder)
        );
    }

    #[test]
    fn missing_required_field_is_rejected() {
        let p = profile();
        let mut m = manifest();
        m.material_effect.fields.pop();
        assert_eq!(
            p.certify(&m.material_effect),
            Err(MaterialityError::MissingRequiredField(
                MaterialFieldKind::Amount
            ))
        );
    }

    #[test]
    fn wrong_effect_class_is_rejected() {
        let p = profile();
        let mut m = manifest();
        m.material_effect.fields[0].semantic = SemanticCommitment(bytes(99));
        assert_eq!(
            p.certify(&m.material_effect),
            Err(MaterialityError::EffectClassMismatch)
        );
    }

    #[test]
    fn certificate_binds_exact_action_and_effect() {
        let p = profile();
        let m = manifest();
        let cert = p.certify(&m.material_effect).unwrap();
        assert!(
            cert.verify_manifest(&p, &m, policy_context(), 7)
                .is_ok()
        );

        let mut changed = m.clone();
        changed.action_request_id = ActionRequestId(bytes(88));
        assert_eq!(
            cert.verify_manifest(&p, &changed, policy_context(), 7),
            Err(MaterialityError::ProfileActionMismatch)
        );
    }

    #[test]
    fn same_generation_in_wrong_policy_context_is_rejected() {
        let p = profile();
        let m = manifest();
        let cert = p.certify(&m.material_effect).unwrap();
        assert_eq!(
            cert.verify_manifest(&p, &m, PolicyContextId(bytes(61)), 7),
            Err(MaterialityError::PolicyContextMismatch)
        );
    }

    #[test]
    fn stale_materiality_profile_is_rejected() {
        let p = profile();
        let m = manifest();
        let cert = p.certify(&m.material_effect).unwrap();
        assert_eq!(
            cert.verify_manifest(&p, &m, policy_context(), 8),
            Err(MaterialityError::StaleMaterialityProfile {
                profile_generation: 7,
                current_generation: 8,
            })
        );
    }

    #[test]
    fn required_warning_omission_is_rejected() {
        let p = profile();
        let mut m = manifest();
        let cert = p.certify(&m.material_effect).unwrap();
        m.warnings.clear();
        assert_eq!(
            cert.verify_manifest(&p, &m, policy_context(), 7),
            Err(MaterialityError::MissingRequiredWarning(
                WarningClass::FinancialCost
            ))
        );
    }

    #[test]
    fn profile_cannot_be_replayed_onto_another_action() {
        let p = profile();
        let mut m = manifest();
        m.action_request_id = ActionRequestId(bytes(88));
        let cert = p.certify(&m.material_effect).unwrap();
        assert_eq!(
            cert.verify_manifest(&p, &m, policy_context(), 7),
            Err(MaterialityError::ProfileActionMismatch)
        );
    }

    #[test]
    fn profile_substitution_invalidates_certificate() {
        let p = profile();
        let m = manifest();
        let cert = p.certify(&m.material_effect).unwrap();
        let mut substituted = p.clone();
        substituted.required_warnings.push(WarningClass::RightsImpact);
        substituted.required_warnings.sort();
        assert_eq!(
            cert.verify_manifest(&substituted, &m, policy_context(), 7),
            Err(MaterialityError::ProfileMismatch)
        );
    }

    #[test]
    fn materiality_certificate_composes_with_confirmation_binding() {
        let p = profile();
        let m = manifest();
        let cert = p.certify(&m.material_effect).unwrap();
        assert!(
            cert.verify_manifest(&p, &m, policy_context(), 7)
                .is_ok()
        );

        let confirmation = ConfirmationBinding::from_manifest(
            &m,
            InteractionContextId(bytes(40)),
            AuthenticationContextId(bytes(41)),
            ConfirmationNonce(bytes(42)),
        )
        .unwrap();
        assert!(confirmation.verify_manifest(&m).is_ok());
    }
}
