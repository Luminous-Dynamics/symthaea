// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Positive closed-world claim-scope projections.
//!
//! This module defines a canonical declaration surface over one exact source
//! reference. It deliberately does not turn generic serialized data into trusted
//! authority. Source-specific verifiers/admission profiles must still establish
//! the source reference and the meaning of every declared positive claim.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::namespaced_code::{NamespacedCodeError, NamespacedCodeV1};
use crate::semantic_evidence::{SemanticTranscriptError, SemanticTranscriptV1};
use crate::source_reference::{EvidenceSourceRefV1, SourceReferenceError};

pub const CLAIM_SCOPE_PROJECTION_VERSION_V1: &str =
    "melothaea-claim-scope-projection-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimDeclarationStatusV1 {
    DeclaredEstablished,
    ExplicitNonclaim,
    NotDeclared,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClaimScopeError {
    WrongVersion { found: String },
    InvalidSourceRef(SourceReferenceError),
    InvalidScopeProfile(NamespacedCodeError),
    EstablishesNotStrictlyIncreasing {
        previous: String,
        next: String,
    },
    ExplicitNonclaimsNotStrictlyIncreasing {
        previous: String,
        next: String,
    },
    ClaimSetOverlap { claim: String },
    InvalidClaimCode(NamespacedCodeError),
    Transcript(SemanticTranscriptError),
}

impl fmt::Display for ClaimScopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongVersion { found } => write!(
                f,
                "unsupported claim-scope projection version: {found}"
            ),
            Self::InvalidSourceRef(error) => write!(f, "invalid source reference: {error}"),
            Self::InvalidScopeProfile(error) => {
                write!(f, "invalid claim-scope profile id: {error}")
            }
            Self::EstablishesNotStrictlyIncreasing { previous, next } => write!(
                f,
                "established claim codes must be strictly increasing and unique: previous={previous}, next={next}"
            ),
            Self::ExplicitNonclaimsNotStrictlyIncreasing { previous, next } => write!(
                f,
                "explicit nonclaim codes must be strictly increasing and unique: previous={previous}, next={next}"
            ),
            Self::ClaimSetOverlap { claim } => write!(
                f,
                "claim cannot be both declared established and an explicit nonclaim: {claim}"
            ),
            Self::InvalidClaimCode(error) => write!(f, "invalid claim code: {error}"),
            Self::Transcript(error) => write!(f, "claim-scope transcript error: {error}"),
        }
    }
}

impl std::error::Error for ClaimScopeError {}

/// Canonical positive closed-world declaration over one exact source reference.
///
/// A valid generic projection is still ordinary data. It records what one exact
/// scope profile *declares* over one exact source ref. A source-specific verifier
/// must re-establish the native source and admission profile before a consumer
/// may treat `establishes` as trustworthy authority.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimScopeProjectionV1 {
    pub claim_scope_version: String,
    pub source_ref: EvidenceSourceRefV1,
    pub scope_profile_id: NamespacedCodeV1,
    pub establishes: Vec<NamespacedCodeV1>,
    pub explicit_nonclaims: Vec<NamespacedCodeV1>,
}

impl ClaimScopeProjectionV1 {
    pub fn validate(&self) -> Result<(), ClaimScopeError> {
        if self.claim_scope_version != CLAIM_SCOPE_PROJECTION_VERSION_V1 {
            return Err(ClaimScopeError::WrongVersion {
                found: self.claim_scope_version.clone(),
            });
        }

        self.source_ref
            .validate()
            .map_err(ClaimScopeError::InvalidSourceRef)?;
        self.scope_profile_id
            .validate()
            .map_err(ClaimScopeError::InvalidScopeProfile)?;

        validate_sorted_claims(
            &self.establishes,
            ClaimSetKind::Establishes,
        )?;
        validate_sorted_claims(
            &self.explicit_nonclaims,
            ClaimSetKind::ExplicitNonclaims,
        )?;

        for claim in &self.establishes {
            if self.explicit_nonclaims.binary_search(claim).is_ok() {
                return Err(ClaimScopeError::ClaimSetOverlap {
                    claim: claim.as_str().to_owned(),
                });
            }
        }

        Ok(())
    }

    /// Return the declaration status for one canonical claim code.
    ///
    /// This method intentionally avoids the name `is_established`: a generic
    /// projection can only declare what its profile says. Trustworthy admission
    /// still belongs to the source-specific verifier.
    pub fn declaration_status(
        &self,
        claim: &NamespacedCodeV1,
    ) -> Result<ClaimDeclarationStatusV1, ClaimScopeError> {
        self.validate()?;
        claim
            .validate()
            .map_err(ClaimScopeError::InvalidClaimCode)?;

        if self.establishes.binary_search(claim).is_ok() {
            Ok(ClaimDeclarationStatusV1::DeclaredEstablished)
        } else if self.explicit_nonclaims.binary_search(claim).is_ok() {
            Ok(ClaimDeclarationStatusV1::ExplicitNonclaim)
        } else {
            Ok(ClaimDeclarationStatusV1::NotDeclared)
        }
    }

    /// Canonical typed payload for this claim-scope projection.
    ///
    /// Deterministic bytes do not confer authority. The source-specific verifier
    /// must independently establish the exact source ref and scope profile.
    pub fn semantic_payload(&self) -> Result<SemanticTranscriptV1, ClaimScopeError> {
        self.validate()?;
        let source_payload = self
            .source_ref
            .semantic_payload()
            .map_err(ClaimScopeError::InvalidSourceRef)?;

        let mut payload = SemanticTranscriptV1::new();
        payload
            .push_utf8(1, &self.claim_scope_version)
            .map_err(ClaimScopeError::Transcript)?;
        payload
            .push_bytes(2, source_payload.as_bytes())
            .map_err(ClaimScopeError::Transcript)?;
        payload
            .push_utf8(3, self.scope_profile_id.as_str())
            .map_err(ClaimScopeError::Transcript)?;
        payload
            .push_sequence(4, &claim_code_bytes(&self.establishes))
            .map_err(ClaimScopeError::Transcript)?;
        payload
            .push_sequence(5, &claim_code_bytes(&self.explicit_nonclaims))
            .map_err(ClaimScopeError::Transcript)?;
        Ok(payload)
    }
}

#[derive(Clone, Copy)]
enum ClaimSetKind {
    Establishes,
    ExplicitNonclaims,
}

fn validate_sorted_claims(
    claims: &[NamespacedCodeV1],
    kind: ClaimSetKind,
) -> Result<(), ClaimScopeError> {
    for claim in claims {
        claim
            .validate()
            .map_err(ClaimScopeError::InvalidClaimCode)?;
    }

    for pair in claims.windows(2) {
        let previous = pair[0].as_str();
        let next = pair[1].as_str();
        if previous >= next {
            return Err(match kind {
                ClaimSetKind::Establishes => {
                    ClaimScopeError::EstablishesNotStrictlyIncreasing {
                        previous: previous.to_owned(),
                        next: next.to_owned(),
                    }
                }
                ClaimSetKind::ExplicitNonclaims => {
                    ClaimScopeError::ExplicitNonclaimsNotStrictlyIncreasing {
                        previous: previous.to_owned(),
                        next: next.to_owned(),
                    }
                }
            });
        }
    }
    Ok(())
}

fn claim_code_bytes(claims: &[NamespacedCodeV1]) -> Vec<Vec<u8>> {
    claims
        .iter()
        .map(|claim| claim.as_str().as_bytes().to_vec())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_evidence::EvidenceSemanticIdV1;
    use crate::source_reference::{
        EVIDENCE_SOURCE_REF_VERSION_V1, SemanticPreservationV1, SourceNativeCommitmentV1,
    };

    fn code(value: &str) -> NamespacedCodeV1 {
        NamespacedCodeV1::new(value).unwrap()
    }

    fn source_ref() -> EvidenceSourceRefV1 {
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

    fn projection() -> ClaimScopeProjectionV1 {
        ClaimScopeProjectionV1 {
            claim_scope_version: CLAIM_SCOPE_PROJECTION_VERSION_V1.into(),
            source_ref: source_ref(),
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
    fn canonical_projection_validates() {
        let value = projection();
        value.validate().unwrap();
        assert!(!value.semantic_payload().unwrap().as_bytes().is_empty());
    }

    #[test]
    fn declaration_status_is_closed_world() {
        let value = projection();
        assert_eq!(
            value
                .declaration_status(&code("muse.collection-close.protocol-bound"))
                .unwrap(),
            ClaimDeclarationStatusV1::DeclaredEstablished
        );
        assert_eq!(
            value
                .declaration_status(&code("muse.collection-close.timestamps-trusted"))
                .unwrap(),
            ClaimDeclarationStatusV1::ExplicitNonclaim
        );
        assert_eq!(
            value
                .declaration_status(&code("muse.collection-close.causal-effect"))
                .unwrap(),
            ClaimDeclarationStatusV1::NotDeclared
        );
    }

    #[test]
    fn establishes_must_be_sorted_and_unique() {
        let mut value = projection();
        value.establishes = vec![
            code("muse.claim.zeta"),
            code("muse.claim.alpha"),
        ];
        assert!(matches!(
            value.validate(),
            Err(ClaimScopeError::EstablishesNotStrictlyIncreasing { .. })
        ));

        value.establishes = vec![code("muse.claim.alpha"), code("muse.claim.alpha")];
        assert!(matches!(
            value.validate(),
            Err(ClaimScopeError::EstablishesNotStrictlyIncreasing { .. })
        ));
    }

    #[test]
    fn explicit_nonclaims_must_be_sorted_and_unique() {
        let mut value = projection();
        value.explicit_nonclaims = vec![
            code("muse.claim.zeta"),
            code("muse.claim.alpha"),
        ];
        assert!(matches!(
            value.validate(),
            Err(ClaimScopeError::ExplicitNonclaimsNotStrictlyIncreasing { .. })
        ));

        value.explicit_nonclaims =
            vec![code("muse.claim.alpha"), code("muse.claim.alpha")];
        assert!(matches!(
            value.validate(),
            Err(ClaimScopeError::ExplicitNonclaimsNotStrictlyIncreasing { .. })
        ));
    }

    #[test]
    fn established_and_nonclaim_sets_cannot_overlap() {
        let mut value = projection();
        value.establishes = vec![code("muse.claim.same")];
        value.explicit_nonclaims = vec![code("muse.claim.same")];
        assert_eq!(
            value.validate(),
            Err(ClaimScopeError::ClaimSetOverlap {
                claim: "muse.claim.same".into()
            })
        );
    }

    #[test]
    fn source_reference_substitution_changes_payload() {
        let left = projection();
        let mut right = left.clone();
        right.source_ref.semantic_id.record_id = "study-02-close".into();
        assert_ne!(
            left.semantic_payload().unwrap().as_bytes(),
            right.semantic_payload().unwrap().as_bytes()
        );
    }

    #[test]
    fn scope_profile_substitution_changes_payload() {
        let left = projection();
        let mut right = left.clone();
        right.scope_profile_id = code("muse.collection-close.claim-profile-v2");
        assert_ne!(
            left.semantic_payload().unwrap().as_bytes(),
            right.semantic_payload().unwrap().as_bytes()
        );
    }

    #[test]
    fn positive_claim_change_changes_payload() {
        let left = projection();
        let mut right = left.clone();
        right.establishes.push(code("muse.collection-close.raw-evidence-rederived"));
        right.establishes.sort();
        right.validate().unwrap();
        assert_ne!(
            left.semantic_payload().unwrap().as_bytes(),
            right.semantic_payload().unwrap().as_bytes()
        );
    }

    #[test]
    fn explicit_nonclaim_change_changes_payload() {
        let left = projection();
        let mut right = left.clone();
        right
            .explicit_nonclaims
            .push(code("muse.collection-close.signoffs-authenticated"));
        right.explicit_nonclaims.sort();
        right.validate().unwrap();
        assert_ne!(
            left.semantic_payload().unwrap().as_bytes(),
            right.semantic_payload().unwrap().as_bytes()
        );
    }

    #[test]
    fn json_round_trip_still_requires_projection_validation() {
        let value = projection();
        let json = serde_json::to_string(&value).unwrap();
        let mut decoded: ClaimScopeProjectionV1 = serde_json::from_str(&json).unwrap();
        decoded.validate().unwrap();

        decoded.claim_scope_version = "future-v2".into();
        assert!(matches!(
            decoded.validate(),
            Err(ClaimScopeError::WrongVersion { .. })
        ));
    }
}
