// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-PRESENT-508: canonical presentation binding for authority-bearing confirmations.
//!
//! This module intentionally does not render UI and does not claim renderer fidelity,
//! localization equivalence, authentication strength, human comprehension, modality
//! qualification, or single-use confirmation consumption. It establishes only the
//! smaller structural theorem needed by those later layers:
//!
//! - material effect semantics have an identity before any presentation exists;
//! - canonical material fields cannot be silently reordered or duplicated;
//! - every material field must have exactly one presented semantic match;
//! - rendering changes alter presentation identity without altering effect identity;
//! - action, intent, principal, effect, presentation, or generation drift invalidates
//!   the corresponding confirmation binding.

use core::fmt;

const MATERIAL_EFFECT_DOMAIN: &[u8] = b"symthaea.presentation.v1/material-effect\0";
const PRESENTATION_DOMAIN: &[u8] = b"symthaea.presentation.v1/presentation\0";
const CONFIRMATION_DOMAIN: &[u8] = b"symthaea.presentation.v1/confirmation\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub [u8; 32]);

        impl $name {
            pub const ZERO: Self = Self([0; 32]);

            pub const fn new(bytes: [u8; 32]) -> Self {
                Self(bytes)
            }

            pub const fn as_bytes(&self) -> &[u8; 32] {
                &self.0
            }

            pub fn is_zero(&self) -> bool {
                self.0 == [0; 32]
            }
        }
    };
}

digest_id!(ActionRequestId);
digest_id!(IntentId);
digest_id!(PrincipalId);
digest_id!(PresentationContextId);
digest_id!(RendererProfileId);
digest_id!(LocaleProfileId);
digest_id!(InteractionContextId);
digest_id!(AuthenticationContextId);
digest_id!(MaterialEffectRoot);
digest_id!(PresentationId);
digest_id!(ConfirmationId);
digest_id!(SemanticCommitment);
digest_id!(RenderedCommitment);
digest_id!(WarningCommitment);
digest_id!(ConfirmationNonce);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum MaterialFieldKind {
    ActionClass = 1,
    Resource = 2,
    Destination = 3,
    Amount = 4,
    Recipient = 5,
    InformationDisclosure = 6,
    Irreversibility = 7,
    Budget = 8,
    Jurisdiction = 9,
    ExternalSideEffect = 10,
}

impl MaterialFieldKind {
    const fn tag(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum WarningClass {
    Irreversible = 1,
    PublicDisclosure = 2,
    ExternalCommitment = 3,
    FinancialCost = 4,
    RightsImpact = 5,
    PrivacyImpact = 6,
    DestructiveEffect = 7,
    CrossJurisdiction = 8,
    ReducedAssurance = 9,
}

impl WarningClass {
    const fn tag(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialField {
    pub kind: MaterialFieldKind,
    /// Upstream canonical semantic commitment. A single field may commit to a
    /// canonical collection, for example several recipients or resources.
    pub semantic: SemanticCommitment,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialEffect {
    /// Strictly ascending by `MaterialFieldKind`; alternate orders are rejected.
    pub fields: Vec<MaterialField>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PresentedField {
    pub kind: MaterialFieldKind,
    pub semantic: SemanticCommitment,
    /// Commitment to the exact renderer-specific representation shown.
    pub rendered: RenderedCommitment,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PresentedWarning {
    pub class: WarningClass,
    pub rendered: WarningCommitment,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresentationManifest {
    pub action_request_id: ActionRequestId,
    pub intent_id: IntentId,
    pub principal_id: PrincipalId,
    pub presentation_context_id: PresentationContextId,
    pub renderer_profile_id: RendererProfileId,
    pub locale_profile_id: LocaleProfileId,
    pub presentation_generation: u64,
    pub material_effect: MaterialEffect,
    pub presented_fields: Vec<PresentedField>,
    pub warnings: Vec<PresentedWarning>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfirmationBinding {
    pub principal_id: PrincipalId,
    pub action_request_id: ActionRequestId,
    pub intent_id: IntentId,
    pub presentation_id: PresentationId,
    pub material_effect_root: MaterialEffectRoot,
    pub interaction_context_id: InteractionContextId,
    pub authentication_context_id: AuthenticationContextId,
    pub presentation_generation: u64,
    pub nonce: ConfirmationNonce,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationError {
    ZeroIdentity(&'static str),
    ZeroGeneration,
    EmptyMaterialEffect,
    ZeroSemanticCommitment(MaterialFieldKind),
    NonCanonicalMaterialOrder,
    DuplicateMaterialField(MaterialFieldKind),
    NonCanonicalPresentedOrder,
    DuplicatePresentedField(MaterialFieldKind),
    PresentedFieldCountMismatch,
    PresentedFieldKindMismatch {
        expected: MaterialFieldKind,
        actual: MaterialFieldKind,
    },
    PresentedSemanticMismatch(MaterialFieldKind),
    ZeroRenderedCommitment(MaterialFieldKind),
    NonCanonicalWarningOrder,
    DuplicateWarning(WarningClass),
    ZeroWarningCommitment(WarningClass),
    ActionRequestMismatch,
    IntentMismatch,
    PrincipalMismatch,
    PresentationMismatch,
    MaterialEffectMismatch,
    PresentationGenerationMismatch,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ValidationError {}

impl MaterialEffect {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.fields.is_empty() {
            return Err(ValidationError::EmptyMaterialEffect);
        }
        validate_material_order(&self.fields)?;
        for field in &self.fields {
            if field.semantic.is_zero() {
                return Err(ValidationError::ZeroSemanticCommitment(field.kind));
            }
        }
        Ok(())
    }

    /// Computes effect identity without requiring any renderer or presentation.
    pub fn root(&self) -> Result<MaterialEffectRoot, ValidationError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(MATERIAL_EFFECT_DOMAIN);
        put_len(&mut hasher, self.fields.len());
        for field in &self.fields {
            hasher.update(&[field.kind.tag()]);
            hasher.update(field.semantic.as_bytes());
        }
        Ok(MaterialEffectRoot(*hasher.finalize().as_bytes()))
    }
}

impl PresentationManifest {
    /// Rejects alternate canonical order instead of normalizing it.
    pub fn validate(&self) -> Result<(), ValidationError> {
        require_nonzero(self.action_request_id.0, "action_request_id")?;
        require_nonzero(self.intent_id.0, "intent_id")?;
        require_nonzero(self.principal_id.0, "principal_id")?;
        require_nonzero(self.presentation_context_id.0, "presentation_context_id")?;
        require_nonzero(self.renderer_profile_id.0, "renderer_profile_id")?;
        require_nonzero(self.locale_profile_id.0, "locale_profile_id")?;
        if self.presentation_generation == 0 {
            return Err(ValidationError::ZeroGeneration);
        }

        self.material_effect.validate()?;
        validate_presented_order(&self.presented_fields)?;
        validate_warning_order(&self.warnings)?;

        if self.material_effect.fields.len() != self.presented_fields.len() {
            return Err(ValidationError::PresentedFieldCountMismatch);
        }

        for (material, presented) in self
            .material_effect
            .fields
            .iter()
            .zip(&self.presented_fields)
        {
            if material.kind != presented.kind {
                return Err(ValidationError::PresentedFieldKindMismatch {
                    expected: material.kind,
                    actual: presented.kind,
                });
            }
            if material.semantic != presented.semantic {
                return Err(ValidationError::PresentedSemanticMismatch(material.kind));
            }
            if presented.rendered.is_zero() {
                return Err(ValidationError::ZeroRenderedCommitment(material.kind));
            }
        }

        for warning in &self.warnings {
            if warning.rendered.is_zero() {
                return Err(ValidationError::ZeroWarningCommitment(warning.class));
            }
        }
        Ok(())
    }

    pub fn material_effect_root(&self) -> Result<MaterialEffectRoot, ValidationError> {
        self.material_effect.root()
    }

    pub fn presentation_id(&self) -> Result<PresentationId, ValidationError> {
        self.validate()?;
        let material_root = self.material_effect.root()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(PRESENTATION_DOMAIN);
        hasher.update(self.action_request_id.as_bytes());
        hasher.update(self.intent_id.as_bytes());
        hasher.update(self.principal_id.as_bytes());
        hasher.update(self.presentation_context_id.as_bytes());
        hasher.update(self.renderer_profile_id.as_bytes());
        hasher.update(self.locale_profile_id.as_bytes());
        hasher.update(&self.presentation_generation.to_le_bytes());
        hasher.update(material_root.as_bytes());

        put_len(&mut hasher, self.presented_fields.len());
        for field in &self.presented_fields {
            hasher.update(&[field.kind.tag()]);
            hasher.update(field.semantic.as_bytes());
            hasher.update(field.rendered.as_bytes());
        }
        put_len(&mut hasher, self.warnings.len());
        for warning in &self.warnings {
            hasher.update(&[warning.class.tag()]);
            hasher.update(warning.rendered.as_bytes());
        }
        Ok(PresentationId(*hasher.finalize().as_bytes()))
    }
}

impl ConfirmationBinding {
    pub fn from_manifest(
        manifest: &PresentationManifest,
        interaction_context_id: InteractionContextId,
        authentication_context_id: AuthenticationContextId,
        nonce: ConfirmationNonce,
    ) -> Result<Self, ValidationError> {
        require_nonzero(interaction_context_id.0, "interaction_context_id")?;
        require_nonzero(authentication_context_id.0, "authentication_context_id")?;
        require_nonzero(nonce.0, "nonce")?;
        manifest.validate()?;

        Ok(Self {
            principal_id: manifest.principal_id,
            action_request_id: manifest.action_request_id,
            intent_id: manifest.intent_id,
            presentation_id: manifest.presentation_id()?,
            material_effect_root: manifest.material_effect.root()?,
            interaction_context_id,
            authentication_context_id,
            presentation_generation: manifest.presentation_generation,
            nonce,
        })
    }

    pub fn confirmation_id(&self) -> Result<ConfirmationId, ValidationError> {
        self.validate_nonzero()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CONFIRMATION_DOMAIN);
        hasher.update(self.principal_id.as_bytes());
        hasher.update(self.action_request_id.as_bytes());
        hasher.update(self.intent_id.as_bytes());
        hasher.update(self.presentation_id.as_bytes());
        hasher.update(self.material_effect_root.as_bytes());
        hasher.update(self.interaction_context_id.as_bytes());
        hasher.update(self.authentication_context_id.as_bytes());
        hasher.update(&self.presentation_generation.to_le_bytes());
        hasher.update(self.nonce.as_bytes());
        Ok(ConfirmationId(*hasher.finalize().as_bytes()))
    }

    /// Checks that this binding still refers to exactly the current manifest.
    pub fn verify_manifest(&self, manifest: &PresentationManifest) -> Result<(), ValidationError> {
        self.validate_nonzero()?;
        manifest.validate()?;
        if self.principal_id != manifest.principal_id {
            return Err(ValidationError::PrincipalMismatch);
        }
        if self.action_request_id != manifest.action_request_id {
            return Err(ValidationError::ActionRequestMismatch);
        }
        if self.intent_id != manifest.intent_id {
            return Err(ValidationError::IntentMismatch);
        }
        if self.presentation_generation != manifest.presentation_generation {
            return Err(ValidationError::PresentationGenerationMismatch);
        }
        if self.material_effect_root != manifest.material_effect.root()? {
            return Err(ValidationError::MaterialEffectMismatch);
        }
        if self.presentation_id != manifest.presentation_id()? {
            return Err(ValidationError::PresentationMismatch);
        }
        Ok(())
    }

    fn validate_nonzero(&self) -> Result<(), ValidationError> {
        require_nonzero(self.principal_id.0, "principal_id")?;
        require_nonzero(self.action_request_id.0, "action_request_id")?;
        require_nonzero(self.intent_id.0, "intent_id")?;
        require_nonzero(self.presentation_id.0, "presentation_id")?;
        require_nonzero(self.material_effect_root.0, "material_effect_root")?;
        require_nonzero(self.interaction_context_id.0, "interaction_context_id")?;
        require_nonzero(self.authentication_context_id.0, "authentication_context_id")?;
        require_nonzero(self.nonce.0, "nonce")?;
        if self.presentation_generation == 0 {
            return Err(ValidationError::ZeroGeneration);
        }
        Ok(())
    }
}

fn require_nonzero(value: [u8; 32], field: &'static str) -> Result<(), ValidationError> {
    if value == [0; 32] {
        Err(ValidationError::ZeroIdentity(field))
    } else {
        Ok(())
    }
}

fn put_len(hasher: &mut blake3::Hasher, len: usize) {
    hasher.update(&(len as u64).to_le_bytes());
}

fn validate_material_order(fields: &[MaterialField]) -> Result<(), ValidationError> {
    let mut previous = None;
    for field in fields {
        let tag = field.kind.tag();
        if let Some(prev) = previous {
            if tag < prev {
                return Err(ValidationError::NonCanonicalMaterialOrder);
            }
            if tag == prev {
                return Err(ValidationError::DuplicateMaterialField(field.kind));
            }
        }
        previous = Some(tag);
    }
    Ok(())
}

fn validate_presented_order(fields: &[PresentedField]) -> Result<(), ValidationError> {
    let mut previous = None;
    for field in fields {
        let tag = field.kind.tag();
        if let Some(prev) = previous {
            if tag < prev {
                return Err(ValidationError::NonCanonicalPresentedOrder);
            }
            if tag == prev {
                return Err(ValidationError::DuplicatePresentedField(field.kind));
            }
        }
        previous = Some(tag);
    }
    Ok(())
}

fn validate_warning_order(warnings: &[PresentedWarning]) -> Result<(), ValidationError> {
    let mut previous = None;
    for warning in warnings {
        let tag = warning.class.tag();
        if let Some(prev) = previous {
            if tag < prev {
                return Err(ValidationError::NonCanonicalWarningOrder);
            }
            if tag == prev {
                return Err(ValidationError::DuplicateWarning(warning.class));
            }
        }
        previous = Some(tag);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    fn manifest() -> PresentationManifest {
        PresentationManifest {
            action_request_id: ActionRequestId(bytes(1)),
            intent_id: IntentId(bytes(2)),
            principal_id: PrincipalId(bytes(3)),
            presentation_context_id: PresentationContextId(bytes(4)),
            renderer_profile_id: RendererProfileId(bytes(5)),
            locale_profile_id: LocaleProfileId(bytes(6)),
            presentation_generation: 7,
            material_effect: MaterialEffect {
                fields: vec![
                    MaterialField {
                        kind: MaterialFieldKind::ActionClass,
                        semantic: SemanticCommitment(bytes(10)),
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
                    semantic: SemanticCommitment(bytes(10)),
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
    fn material_effect_identity_precedes_presentation() {
        let effect = MaterialEffect {
            fields: vec![MaterialField {
                kind: MaterialFieldKind::ActionClass,
                semantic: SemanticCommitment(bytes(10)),
            }],
        };
        assert_ne!(effect.root().unwrap(), MaterialEffectRoot::ZERO);
    }

    #[test]
    fn stable_manifest_has_stable_ids() {
        let a = manifest();
        let b = manifest();
        assert_eq!(a.material_effect_root().unwrap(), b.material_effect_root().unwrap());
        assert_eq!(a.presentation_id().unwrap(), b.presentation_id().unwrap());
    }

    #[test]
    fn noncanonical_material_order_is_rejected() {
        let mut m = manifest();
        m.material_effect.fields.swap(0, 1);
        assert_eq!(m.validate(), Err(ValidationError::NonCanonicalMaterialOrder));
    }

    #[test]
    fn omitted_presented_field_is_rejected() {
        let mut m = manifest();
        m.presented_fields.pop();
        assert_eq!(m.validate(), Err(ValidationError::PresentedFieldCountMismatch));
    }

    #[test]
    fn semantic_substitution_is_rejected() {
        let mut m = manifest();
        m.presented_fields[1].semantic = SemanticCommitment(bytes(99));
        assert_eq!(
            m.validate(),
            Err(ValidationError::PresentedSemanticMismatch(
                MaterialFieldKind::Destination
            ))
        );
    }

    #[test]
    fn rendering_drift_changes_presentation_not_effect() {
        let a = manifest();
        let mut b = manifest();
        b.presented_fields[2].rendered = RenderedCommitment(bytes(88));
        assert_eq!(a.material_effect_root().unwrap(), b.material_effect_root().unwrap());
        assert_ne!(a.presentation_id().unwrap(), b.presentation_id().unwrap());
    }

    #[test]
    fn material_effect_drift_invalidates_confirmation() {
        let original = manifest();
        let confirmation = ConfirmationBinding::from_manifest(
            &original,
            InteractionContextId(bytes(40)),
            AuthenticationContextId(bytes(41)),
            ConfirmationNonce(bytes(42)),
        )
        .unwrap();
        let mut changed = manifest();
        changed.material_effect.fields[2].semantic = SemanticCommitment(bytes(77));
        changed.presented_fields[2].semantic = SemanticCommitment(bytes(77));
        assert_eq!(
            confirmation.verify_manifest(&changed),
            Err(ValidationError::MaterialEffectMismatch)
        );
    }

    #[test]
    fn presentation_drift_invalidates_confirmation() {
        let original = manifest();
        let confirmation = ConfirmationBinding::from_manifest(
            &original,
            InteractionContextId(bytes(40)),
            AuthenticationContextId(bytes(41)),
            ConfirmationNonce(bytes(42)),
        )
        .unwrap();
        let mut changed = manifest();
        changed.presented_fields[0].rendered = RenderedCommitment(bytes(87));
        assert_eq!(
            confirmation.verify_manifest(&changed),
            Err(ValidationError::PresentationMismatch)
        );
    }

    #[test]
    fn action_drift_invalidates_confirmation() {
        let original = manifest();
        let confirmation = ConfirmationBinding::from_manifest(
            &original,
            InteractionContextId(bytes(40)),
            AuthenticationContextId(bytes(41)),
            ConfirmationNonce(bytes(42)),
        )
        .unwrap();
        let mut changed = manifest();
        changed.action_request_id = ActionRequestId(bytes(90));
        assert_eq!(
            confirmation.verify_manifest(&changed),
            Err(ValidationError::ActionRequestMismatch)
        );
    }

    #[test]
    fn confirmation_identity_binds_nonce() {
        let m = manifest();
        let a = ConfirmationBinding::from_manifest(
            &m,
            InteractionContextId(bytes(40)),
            AuthenticationContextId(bytes(41)),
            ConfirmationNonce(bytes(42)),
        )
        .unwrap();
        let b = ConfirmationBinding::from_manifest(
            &m,
            InteractionContextId(bytes(40)),
            AuthenticationContextId(bytes(41)),
            ConfirmationNonce(bytes(43)),
        )
        .unwrap();
        assert_ne!(a.confirmation_id().unwrap(), b.confirmation_id().unwrap());
    }
}
