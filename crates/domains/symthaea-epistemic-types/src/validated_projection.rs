// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable structurally validated wrappers for MEL-EPI projections.
//!
//! These wrappers close the temporal-validity gap between mutable/raw DTOs and
//! source-specific admission. They prove only that the contained generic object
//! satisfied its structural/canonical validator at construction time.
//!
//! ```text
//! raw/deserialized DTO
//!     != structurally validated wrapper
//!     != source-native admitted evidence
//! ```
//!
//! No validated wrapper in this module is a credential and none can be built by
//! unconstrained Serde deserialization.

use std::convert::TryFrom;

use crate::claim_scope::{
    ClaimDeclarationStatusV1, ClaimScopeError, ClaimScopeProjectionV1,
};
use crate::namespaced_code::NamespacedCodeV1;
use crate::semantic_evidence::{EvidenceSemanticIdV1, SemanticTranscriptV1};
use crate::source_reference::{
    EvidenceSourceRefV1, SemanticPreservationV1, SourceNativeCommitmentV1,
    SourceReferenceError,
};

/// Immutable, structurally validated MEL-EPI source reference.
///
/// Construction runs the full generic source-reference validator and freezes
/// the canonical semantic payload. The wrapper intentionally does not implement
/// `Deserialize`, does not expose mutable access, and does not imply that the
/// source-native artifact or either commitment was independently recomputed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedEvidenceSourceRefV1 {
    raw: EvidenceSourceRefV1,
    semantic_payload: SemanticTranscriptV1,
}

impl ValidatedEvidenceSourceRefV1 {
    /// Borrow the raw DTO read-only.
    ///
    /// Returning to a mutable raw value requires explicit demotion through
    /// [`Self::into_raw`], after which this wrapper's validation guarantee no
    /// longer applies.
    pub fn as_raw(&self) -> &EvidenceSourceRefV1 {
        &self.raw
    }

    /// Explicitly demote the validated wrapper back to a mutable/raw DTO.
    ///
    /// Any subsequent semantic mutation requires a fresh `TryFrom` validation
    /// before the object may again be treated as structurally validated.
    pub fn into_raw(self) -> EvidenceSourceRefV1 {
        self.raw
    }

    /// Canonical source-reference payload cached at validation time.
    pub fn semantic_payload(&self) -> &SemanticTranscriptV1 {
        &self.semantic_payload
    }

    pub fn semantic_id(&self) -> &EvidenceSemanticIdV1 {
        &self.raw.semantic_id
    }

    pub fn native_commitment(&self) -> &SourceNativeCommitmentV1 {
        &self.raw.native_commitment
    }

    pub fn preservation(&self) -> &SemanticPreservationV1 {
        &self.raw.preservation
    }
}

impl TryFrom<EvidenceSourceRefV1> for ValidatedEvidenceSourceRefV1 {
    type Error = SourceReferenceError;

    fn try_from(raw: EvidenceSourceRefV1) -> Result<Self, Self::Error> {
        raw.validate()?;
        // `semantic_payload` revalidates internally today. That is intentionally
        // paid once at wrapper construction so all later reads are validation-
        // stable and require no mutable access or repeat parse/shape checks.
        let semantic_payload = raw.semantic_payload()?;
        Ok(Self {
            raw,
            semantic_payload,
        })
    }
}

impl TryFrom<&EvidenceSourceRefV1> for ValidatedEvidenceSourceRefV1 {
    type Error = SourceReferenceError;

    fn try_from(raw: &EvidenceSourceRefV1) -> Result<Self, Self::Error> {
        Self::try_from(raw.clone())
    }
}

/// Immutable, structurally validated closed-world claim-scope projection.
///
/// The wrapper means only that the generic projection is canonical and that its
/// cached declaration sets/source reference passed generic validation. It does
/// not mean that a domain/source verifier admitted the declared claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedClaimScopeProjectionV1 {
    raw: ClaimScopeProjectionV1,
    semantic_payload: SemanticTranscriptV1,
}

impl ValidatedClaimScopeProjectionV1 {
    pub fn as_raw(&self) -> &ClaimScopeProjectionV1 {
        &self.raw
    }

    /// Explicitly demote this validated projection back to an ordinary DTO.
    pub fn into_raw(self) -> ClaimScopeProjectionV1 {
        self.raw
    }

    /// Canonical claim-scope payload cached at validation time.
    pub fn semantic_payload(&self) -> &SemanticTranscriptV1 {
        &self.semantic_payload
    }

    /// Source reference that was structurally validated as part of this wrapper.
    ///
    /// The returned value remains the raw DTO type so callers cannot confuse
    /// parent projection validation with source-native admission.
    pub fn source_ref(&self) -> &EvidenceSourceRefV1 {
        &self.raw.source_ref
    }

    pub fn scope_profile_id(&self) -> &NamespacedCodeV1 {
        &self.raw.scope_profile_id
    }

    pub fn establishes(&self) -> &[NamespacedCodeV1] {
        &self.raw.establishes
    }

    pub fn explicit_nonclaims(&self) -> &[NamespacedCodeV1] {
        &self.raw.explicit_nonclaims
    }

    /// Query the frozen declaration sets without revalidating the projection.
    ///
    /// The vocabulary remains deliberately declaration-oriented: even this
    /// structurally validated wrapper is not a source-admitted credential.
    pub fn declaration_status(
        &self,
        claim: &NamespacedCodeV1,
    ) -> ClaimDeclarationStatusV1 {
        if self.raw.establishes.binary_search(claim).is_ok() {
            ClaimDeclarationStatusV1::DeclaredEstablished
        } else if self.raw.explicit_nonclaims.binary_search(claim).is_ok() {
            ClaimDeclarationStatusV1::ExplicitNonclaim
        } else {
            ClaimDeclarationStatusV1::NotDeclared
        }
    }
}

impl TryFrom<ClaimScopeProjectionV1> for ValidatedClaimScopeProjectionV1 {
    type Error = ClaimScopeError;

    fn try_from(raw: ClaimScopeProjectionV1) -> Result<Self, Self::Error> {
        raw.validate()?;
        let semantic_payload = raw.semantic_payload()?;
        Ok(Self {
            raw,
            semantic_payload,
        })
    }
}

impl TryFrom<&ClaimScopeProjectionV1> for ValidatedClaimScopeProjectionV1 {
    type Error = ClaimScopeError;

    fn try_from(raw: &ClaimScopeProjectionV1) -> Result<Self, Self::Error> {
        Self::try_from(raw.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claim_scope::CLAIM_SCOPE_PROJECTION_VERSION_V1;
    use crate::semantic_evidence::EvidenceSemanticIdV1;
    use crate::source_reference::{
        EVIDENCE_SOURCE_REF_VERSION_V1, SemanticPreservationV1,
        SourceNativeCommitmentV1,
    };

    fn code(value: &str) -> NamespacedCodeV1 {
        NamespacedCodeV1::new(value).unwrap()
    }

    fn raw_source_ref() -> EvidenceSourceRefV1 {
        EvidenceSourceRefV1 {
            source_ref_version: EVIDENCE_SOURCE_REF_VERSION_V1.into(),
            semantic_id: EvidenceSemanticIdV1::from_sha256_digest(
                "muse.collection-close",
                "v1",
                "projection-v1",
                "study-01-close",
                [0x11; 32],
            )
            .unwrap(),
            native_commitment: SourceNativeCommitmentV1 {
                scheme_id: code("muse.collection-close.commitment-v1"),
                sha256_hex: "22".repeat(32),
            },
            preservation: SemanticPreservationV1::LosslessUnderProfile,
        }
    }

    fn raw_claim_scope() -> ClaimScopeProjectionV1 {
        ClaimScopeProjectionV1 {
            claim_scope_version: CLAIM_SCOPE_PROJECTION_VERSION_V1.into(),
            source_ref: raw_source_ref(),
            scope_profile_id: code("muse.collection-close.claim-profile-v1"),
            establishes: vec![
                code("muse.collection-close.evidence-counts-reconciled"),
                code("muse.collection-close.protocol-bound"),
            ],
            explicit_nonclaims: vec![
                code("muse.collection-close.signers-authorized"),
                code("muse.collection-close.timestamps-trusted"),
            ],
        }
    }

    #[test]
    fn invalid_raw_source_ref_cannot_enter_validated_state() {
        let mut raw = raw_source_ref();
        raw.source_ref_version = "future-v2".into();
        assert!(ValidatedEvidenceSourceRefV1::try_from(raw).is_err());
    }

    #[test]
    fn valid_source_ref_caches_canonical_payload() {
        let raw = raw_source_ref();
        let expected = raw.semantic_payload().unwrap();
        let validated = ValidatedEvidenceSourceRefV1::try_from(raw).unwrap();
        assert_eq!(validated.semantic_payload(), &expected);
        assert_eq!(
            validated.native_commitment().scheme_id.as_str(),
            "muse.collection-close.commitment-v1"
        );
    }

    #[test]
    fn mutation_requires_explicit_demotion_and_revalidation() {
        let validated = ValidatedEvidenceSourceRefV1::try_from(raw_source_ref()).unwrap();
        let mut raw = validated.into_raw();
        raw.native_commitment.sha256_hex = "A".repeat(64);
        assert!(ValidatedEvidenceSourceRefV1::try_from(raw).is_err());
    }

    #[test]
    fn serde_round_trip_yields_raw_then_requires_validation() {
        let json = serde_json::to_string(&raw_source_ref()).unwrap();
        let raw: EvidenceSourceRefV1 = serde_json::from_str(&json).unwrap();
        let validated = ValidatedEvidenceSourceRefV1::try_from(raw).unwrap();
        assert_eq!(validated.semantic_id().record_id, "study-01-close");
    }

    #[test]
    fn invalid_raw_claim_scope_cannot_enter_validated_state() {
        let mut raw = raw_claim_scope();
        raw.establishes = vec![code("muse.claim.zeta"), code("muse.claim.alpha")];
        assert!(ValidatedClaimScopeProjectionV1::try_from(raw).is_err());
    }

    #[test]
    fn validated_claim_scope_queries_without_credential_semantics() {
        let validated =
            ValidatedClaimScopeProjectionV1::try_from(raw_claim_scope()).unwrap();

        assert_eq!(
            validated.declaration_status(&code("muse.collection-close.protocol-bound")),
            ClaimDeclarationStatusV1::DeclaredEstablished
        );
        assert_eq!(
            validated.declaration_status(&code("muse.collection-close.timestamps-trusted")),
            ClaimDeclarationStatusV1::ExplicitNonclaim
        );
        assert_eq!(
            validated.declaration_status(&code("muse.collection-close.causal-effect")),
            ClaimDeclarationStatusV1::NotDeclared
        );
    }

    #[test]
    fn claim_scope_demotion_invalidates_typestate_after_mutation() {
        let validated =
            ValidatedClaimScopeProjectionV1::try_from(raw_claim_scope()).unwrap();
        let frozen_payload = validated.semantic_payload().clone();
        let mut raw = validated.into_raw();
        raw.explicit_nonclaims
            .push(code("muse.collection-close.signoffs-authenticated"));
        // Appending this code makes the vector non-canonical for this fixture;
        // the raw DTO cannot regain validated status without explicit repair.
        assert!(ValidatedClaimScopeProjectionV1::try_from(raw).is_err());
        assert!(!frozen_payload.as_bytes().is_empty());
    }
}
