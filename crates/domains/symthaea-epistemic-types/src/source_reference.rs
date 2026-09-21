// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Non-authoritative cross-domain source references for epistemic artifacts.
//!
//! A valid `EvidenceSourceRefV1` proves only that its generic shape is internally
//! canonical. Source authority remains with the source-native validator and
//! commitment theorem. A source-specific adapter must independently validate the
//! native artifact, recompute its native commitment, construct/recompute the
//! MEL-EPI semantic identity, and only then admit this reference into a stronger
//! claim scope.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::namespaced_code::{NamespacedCodeError, NamespacedCodeV1};
use crate::semantic_evidence::{
    EvidenceSemanticIdV1, SemanticTranscriptError, SemanticTranscriptV1,
};

pub const EVIDENCE_SOURCE_REF_VERSION_V1: &str = "melothaea-evidence-source-ref-v1";
const PRESERVATION_LOSSLESS_V1: &str = "lossless-under-profile";
const PRESERVATION_PROJECTED_V1: &str = "projected-with-loss";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceReferenceError {
    WrongVersion { found: String },
    InvalidSemanticId(SemanticTranscriptError),
    InvalidSchemeId(NamespacedCodeError),
    InvalidNativeSha256Hex,
    EmptyProjectionLosses,
    ProjectionLossesNotStrictlyIncreasing {
        previous: String,
        next: String,
    },
    InvalidLossCode(NamespacedCodeError),
    Transcript(SemanticTranscriptError),
}

impl fmt::Display for SourceReferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongVersion { found } => write!(
                f,
                "unsupported evidence source-reference version: {found}"
            ),
            Self::InvalidSemanticId(error) => {
                write!(f, "invalid MEL-EPI semantic identity: {error}")
            }
            Self::InvalidSchemeId(error) => {
                write!(f, "invalid source-native commitment scheme id: {error}")
            }
            Self::InvalidNativeSha256Hex => write!(
                f,
                "source-native commitment digest must be exactly 64 lower-case hexadecimal characters"
            ),
            Self::EmptyProjectionLosses => write!(
                f,
                "ProjectedWithLoss requires at least one explicit loss code"
            ),
            Self::ProjectionLossesNotStrictlyIncreasing { previous, next } => write!(
                f,
                "projection loss codes must be strictly increasing and unique: previous={previous}, next={next}"
            ),
            Self::InvalidLossCode(error) => write!(f, "invalid projection loss code: {error}"),
            Self::Transcript(error) => write!(f, "source-reference transcript error: {error}"),
        }
    }
}

impl std::error::Error for SourceReferenceError {}

/// One exact source-native commitment theorem and its SHA-256 value.
///
/// `scheme_id` is semantic. It names what was committed/rederived, not merely
/// the hash algorithm. Two profiles that both use SHA-256 but commit different
/// source semantics must use different scheme IDs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceNativeCommitmentV1 {
    pub scheme_id: NamespacedCodeV1,
    pub sha256_hex: String,
}

impl SourceNativeCommitmentV1 {
    pub fn validate(&self) -> Result<(), SourceReferenceError> {
        self.scheme_id
            .validate()
            .map_err(SourceReferenceError::InvalidSchemeId)?;
        decode_lower_hex_32(&self.sha256_hex)
            .ok_or(SourceReferenceError::InvalidNativeSha256Hex)?;
        Ok(())
    }

    pub fn digest_bytes(&self) -> Result<[u8; 32], SourceReferenceError> {
        self.validate()?;
        decode_lower_hex_32(&self.sha256_hex)
            .ok_or(SourceReferenceError::InvalidNativeSha256Hex)
    }
}

/// Semantic preservation ceiling for a generic projection.
///
/// `LosslessUnderProfile` means the frozen source-specific projection profile
/// declares no known loss for the semantics that profile is designed to carry.
/// It does not mean byte-identical serialization or preservation of every
/// conceivable future interpretation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticPreservationV1 {
    LosslessUnderProfile,
    ProjectedWithLoss { loss_codes: Vec<NamespacedCodeV1> },
}

impl SemanticPreservationV1 {
    pub fn validate(&self) -> Result<(), SourceReferenceError> {
        match self {
            Self::LosslessUnderProfile => Ok(()),
            Self::ProjectedWithLoss { loss_codes } => {
                if loss_codes.is_empty() {
                    return Err(SourceReferenceError::EmptyProjectionLosses);
                }
                for code in loss_codes {
                    code.validate()
                        .map_err(SourceReferenceError::InvalidLossCode)?;
                }
                for pair in loss_codes.windows(2) {
                    let previous = pair[0].as_str();
                    let next = pair[1].as_str();
                    if previous >= next {
                        return Err(
                            SourceReferenceError::ProjectionLossesNotStrictlyIncreasing {
                                previous: previous.to_owned(),
                                next: next.to_owned(),
                            },
                        );
                    }
                }
                Ok(())
            }
        }
    }

    fn profile_code(&self) -> &'static str {
        match self {
            Self::LosslessUnderProfile => PRESERVATION_LOSSLESS_V1,
            Self::ProjectedWithLoss { .. } => PRESERVATION_PROJECTED_V1,
        }
    }

    fn loss_code_bytes(&self) -> Vec<Vec<u8>> {
        match self {
            Self::LosslessUnderProfile => Vec::new(),
            Self::ProjectedWithLoss { loss_codes } => loss_codes
                .iter()
                .map(|code| code.as_str().as_bytes().to_vec())
                .collect(),
        }
    }
}

/// Shape-only reference tying one MEL-EPI semantic identity to one source-native
/// commitment and an explicit semantic-preservation ceiling.
///
/// Validation of this struct does **not** recompute either commitment and does
/// not prove that the source-native artifact exists, is valid, or was honestly
/// projected. Source-specific admission must perform those checks independently.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceSourceRefV1 {
    pub source_ref_version: String,
    pub semantic_id: EvidenceSemanticIdV1,
    pub native_commitment: SourceNativeCommitmentV1,
    pub preservation: SemanticPreservationV1,
}

impl EvidenceSourceRefV1 {
    pub fn validate(&self) -> Result<(), SourceReferenceError> {
        if self.source_ref_version != EVIDENCE_SOURCE_REF_VERSION_V1 {
            return Err(SourceReferenceError::WrongVersion {
                found: self.source_ref_version.clone(),
            });
        }
        self.semantic_id
            .validate()
            .map_err(SourceReferenceError::InvalidSemanticId)?;
        self.native_commitment.validate()?;
        self.preservation.validate()?;
        Ok(())
    }

    /// Canonical MEL-EPI payload for this generic source reference.
    ///
    /// This function is deterministic after shape validation, but remains
    /// non-authoritative: it does not re-run the source-native validator or
    /// recompute either digest from source evidence.
    pub fn semantic_payload(&self) -> Result<SemanticTranscriptV1, SourceReferenceError> {
        self.validate()?;

        let mut semantic_id = SemanticTranscriptV1::new();
        semantic_id
            .push_utf8(1, &self.semantic_id.digest_algorithm)
            .map_err(SourceReferenceError::Transcript)?;
        semantic_id
            .push_utf8(2, &self.semantic_id.transcript_version)
            .map_err(SourceReferenceError::Transcript)?;
        semantic_id
            .push_utf8(3, &self.semantic_id.namespace)
            .map_err(SourceReferenceError::Transcript)?;
        semantic_id
            .push_utf8(4, &self.semantic_id.schema_version)
            .map_err(SourceReferenceError::Transcript)?;
        semantic_id
            .push_utf8(5, &self.semantic_id.profile_id)
            .map_err(SourceReferenceError::Transcript)?;
        semantic_id
            .push_utf8(6, &self.semantic_id.record_id)
            .map_err(SourceReferenceError::Transcript)?;
        semantic_id
            .push_sha256_digest_reference(
                7,
                self.semantic_id
                    .digest_bytes()
                    .map_err(SourceReferenceError::InvalidSemanticId)?,
            )
            .map_err(SourceReferenceError::Transcript)?;

        let mut payload = SemanticTranscriptV1::new();
        payload
            .push_utf8(1, &self.source_ref_version)
            .map_err(SourceReferenceError::Transcript)?;
        payload
            .push_bytes(2, semantic_id.as_bytes())
            .map_err(SourceReferenceError::Transcript)?;
        payload
            .push_utf8(3, self.native_commitment.scheme_id.as_str())
            .map_err(SourceReferenceError::Transcript)?;
        payload
            .push_sha256_digest_reference(4, self.native_commitment.digest_bytes()?)
            .map_err(SourceReferenceError::Transcript)?;
        payload
            .push_utf8(5, self.preservation.profile_code())
            .map_err(SourceReferenceError::Transcript)?;
        payload
            .push_sequence(6, &self.preservation.loss_code_bytes())
            .map_err(SourceReferenceError::Transcript)?;
        Ok(payload)
    }
}

fn decode_lower_hex_32(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }

    let bytes = value.as_bytes();
    let mut out = [0u8; 32];
    for (index, slot) in out.iter_mut().enumerate() {
        let high = hex_nibble(bytes[index * 2])?;
        let low = hex_nibble(bytes[index * 2 + 1])?;
        *slot = (high << 4) | low;
    }
    Some(out)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn semantic_id() -> EvidenceSemanticIdV1 {
        EvidenceSemanticIdV1::from_sha256_digest(
            "muse.test",
            "native-v1",
            "projection-v1",
            "record-01",
            [0x22; 32],
        )
        .unwrap()
    }

    fn native_commitment() -> SourceNativeCommitmentV1 {
        SourceNativeCommitmentV1 {
            scheme_id: NamespacedCodeV1::new("muse.test.native-commitment-v1").unwrap(),
            sha256_hex: "33".repeat(32),
        }
    }

    fn lossless_ref() -> EvidenceSourceRefV1 {
        EvidenceSourceRefV1 {
            source_ref_version: EVIDENCE_SOURCE_REF_VERSION_V1.into(),
            semantic_id: semantic_id(),
            native_commitment: native_commitment(),
            preservation: SemanticPreservationV1::LosslessUnderProfile,
        }
    }

    #[test]
    fn lossless_reference_is_shape_valid() {
        let value = lossless_ref();
        value.validate().unwrap();
        assert!(!value.semantic_payload().unwrap().as_bytes().is_empty());
    }

    #[test]
    fn generic_json_round_trip_still_requires_object_validation() {
        let value = lossless_ref();
        let json = serde_json::to_string(&value).unwrap();
        let mut decoded: EvidenceSourceRefV1 = serde_json::from_str(&json).unwrap();
        decoded.validate().unwrap();

        decoded.source_ref_version = "future-v2".into();
        assert!(matches!(
            decoded.validate(),
            Err(SourceReferenceError::WrongVersion { .. })
        ));
    }

    #[test]
    fn native_digest_is_canonical_lower_hex() {
        let mut value = lossless_ref();
        value.native_commitment.sha256_hex = "A".repeat(64);
        assert_eq!(
            value.validate(),
            Err(SourceReferenceError::InvalidNativeSha256Hex)
        );
    }

    #[test]
    fn projected_loss_requires_nonempty_sorted_unique_codes() {
        let mut value = lossless_ref();
        value.preservation = SemanticPreservationV1::ProjectedWithLoss {
            loss_codes: Vec::new(),
        };
        assert_eq!(
            value.validate(),
            Err(SourceReferenceError::EmptyProjectionLosses)
        );

        value.preservation = SemanticPreservationV1::ProjectedWithLoss {
            loss_codes: vec![
                NamespacedCodeV1::new("muse.loss.zeta").unwrap(),
                NamespacedCodeV1::new("muse.loss.alpha").unwrap(),
            ],
        };
        assert!(matches!(
            value.validate(),
            Err(SourceReferenceError::ProjectionLossesNotStrictlyIncreasing { .. })
        ));

        value.preservation = SemanticPreservationV1::ProjectedWithLoss {
            loss_codes: vec![
                NamespacedCodeV1::new("muse.loss.alpha").unwrap(),
                NamespacedCodeV1::new("muse.loss.alpha").unwrap(),
            ],
        };
        assert!(matches!(
            value.validate(),
            Err(SourceReferenceError::ProjectionLossesNotStrictlyIncreasing { .. })
        ));
    }

    #[test]
    fn scheme_substitution_changes_generic_semantic_payload() {
        let left = lossless_ref();
        let mut right = left.clone();
        right.native_commitment.scheme_id =
            NamespacedCodeV1::new("muse.test.other-commitment-v1").unwrap();

        assert_ne!(
            left.semantic_payload().unwrap().as_bytes(),
            right.semantic_payload().unwrap().as_bytes()
        );
    }

    #[test]
    fn native_digest_substitution_changes_generic_semantic_payload() {
        let left = lossless_ref();
        let mut right = left.clone();
        right.native_commitment.sha256_hex = "44".repeat(32);

        assert_ne!(
            left.semantic_payload().unwrap().as_bytes(),
            right.semantic_payload().unwrap().as_bytes()
        );
    }

    #[test]
    fn preservation_ceiling_changes_generic_semantic_payload() {
        let left = lossless_ref();
        let mut right = left.clone();
        right.preservation = SemanticPreservationV1::ProjectedWithLoss {
            loss_codes: vec![
                NamespacedCodeV1::new("muse.loss.display-metadata").unwrap(),
                NamespacedCodeV1::new("muse.loss.operator-notes").unwrap(),
            ],
        };
        right.validate().unwrap();

        assert_ne!(
            left.semantic_payload().unwrap().as_bytes(),
            right.semantic_payload().unwrap().as_bytes()
        );
    }

    #[test]
    fn semantic_identity_substitution_changes_generic_semantic_payload() {
        let left = lossless_ref();
        let mut right = left.clone();
        right.semantic_id.record_id = "record-02".into();

        assert_ne!(
            left.semantic_payload().unwrap().as_bytes(),
            right.semantic_payload().unwrap().as_bytes()
        );
    }
}
