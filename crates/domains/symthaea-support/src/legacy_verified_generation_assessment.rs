// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audit-safe binding from externally verified semantic review into one legacy
//! qualification-generation assessment.
//!
//! The runtime verified wrapper remains non-serializable and non-cloneable. This
//! module persists only commitments sufficient to prove which semantic trust
//! lineage backed an assessment.
//!
//! ```text
//! runtime verified wrapper != persistent evidence artifact
//! persistent binding != ability to recreate trust
//! semantic trust binding != source evidence
//! source evidence != competence
//! ```

use crate::it_qualification::ItQualificationMatrixV1;
use crate::legacy_computing::LegacyComputingPackV1;
use crate::legacy_qualification_profile::LegacyQualificationProfileV1;
use crate::legacy_qualification_profile_v3::{
    legacy_qualification_manifest_commitment_v1, LegacyQualificationGenerationAssessmentV1,
    LegacyQualificationManifestV1,
};
use crate::legacy_qualification_source_ledger_v3::LegacyQualificationSourceLedgerV3;
use crate::legacy_semantic_replacement::LegacySemanticReplacementReviewLedgerV1;
use crate::legacy_semantic_review_verification::{
    assess_legacy_qualification_generation_with_verified_semantic_review_v1,
    legacy_semantic_replacement_review_ledger_commitment_v1,
    LegacySemanticReviewVerificationErrorV1, VerifiedLegacySemanticReplacementReviewsV1,
};
use crate::legacy_source_artifacts::LegacySourceArtifactLedgerV1;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

pub const LEGACY_VERIFIED_GENERATION_ASSESSMENT_SCHEMA_V1: &str =
    "symthaea-it-legacy-verified-generation-assessment-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySemanticReviewQualificationBindingV1 {
    pub semantic_review_ledger_blake3: String,
    pub verifier_profile: String,
    pub semantic_trust_bundle_blake3: String,
    pub verified_at_unix_ms: u64,
    pub predecessor_manifest_blake3: String,
    pub successor_manifest_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyVerifiedQualificationGenerationAssessmentV1 {
    pub schema_version: String,
    pub generation_assessment: LegacyQualificationGenerationAssessmentV1,
    /// None only for canonical generation 1, which has no predecessor replacement.
    pub semantic_review_binding: Option<LegacySemanticReviewQualificationBindingV1>,
    /// BLAKE3 over the exact typed generation assessment + semantic-review binding.
    pub assessment_binding_blake3: String,
}

pub fn assess_legacy_qualification_generation_with_bound_verified_semantic_review_v1(
    pack: &LegacyComputingPackV1,
    profile: &LegacyQualificationProfileV1,
    matrix: &ItQualificationMatrixV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    source_ledger: &LegacyQualificationSourceLedgerV3,
    predecessor_manifest: Option<&LegacyQualificationManifestV1>,
    manifest: &LegacyQualificationManifestV1,
    semantic_reviews: &LegacySemanticReplacementReviewLedgerV1,
    verified_reviews: Option<&VerifiedLegacySemanticReplacementReviewsV1>,
) -> Result<LegacyVerifiedQualificationGenerationAssessmentV1, LegacyVerifiedGenerationAssessmentErrorV1>
{
    let generation_assessment =
        assess_legacy_qualification_generation_with_verified_semantic_review_v1(
            pack,
            profile,
            matrix,
            artifacts,
            source_ledger,
            predecessor_manifest,
            manifest,
            semantic_reviews,
            verified_reviews,
        )?;

    let semantic_review_binding = match manifest.generation {
        1 => None,
        _ => {
            let predecessor = predecessor_manifest.ok_or(
                LegacyVerifiedGenerationAssessmentErrorV1::MissingPredecessorManifest,
            )?;
            let verified = verified_reviews.ok_or(
                LegacyVerifiedGenerationAssessmentErrorV1::MissingVerifiedReviews,
            )?;
            let review_ledger_blake3 =
                legacy_semantic_replacement_review_ledger_commitment_v1(semantic_reviews)?;
            if review_ledger_blake3 != verified.ledger_blake3() {
                return Err(LegacyVerifiedGenerationAssessmentErrorV1::ReviewLedgerMismatch);
            }
            let predecessor_manifest_blake3 =
                legacy_qualification_manifest_commitment_v1(predecessor)
                    .map_err(|err| LegacyVerifiedGenerationAssessmentErrorV1::Manifest(err.to_string()))?;
            let successor_manifest_blake3 = generation_assessment.manifest_blake3.clone();
            Ok::<_, LegacyVerifiedGenerationAssessmentErrorV1>(
                LegacySemanticReviewQualificationBindingV1 {
                    semantic_review_ledger_blake3: review_ledger_blake3,
                    verifier_profile: verified.verifier_profile().to_string(),
                    semantic_trust_bundle_blake3: verified.trust_bundle_blake3().to_string(),
                    verified_at_unix_ms: verified.verified_at_unix_ms(),
                    predecessor_manifest_blake3,
                    successor_manifest_blake3,
                },
            )?
            .into()
        }
    };

    let assessment_binding_blake3 = legacy_verified_generation_assessment_binding_v1(
        &generation_assessment,
        semantic_review_binding.as_ref(),
    )?;

    Ok(LegacyVerifiedQualificationGenerationAssessmentV1 {
        schema_version: LEGACY_VERIFIED_GENERATION_ASSESSMENT_SCHEMA_V1.into(),
        generation_assessment,
        semantic_review_binding,
        assessment_binding_blake3,
    })
}

pub fn legacy_verified_generation_assessment_binding_v1(
    generation_assessment: &LegacyQualificationGenerationAssessmentV1,
    semantic_review_binding: Option<&LegacySemanticReviewQualificationBindingV1>,
) -> Result<String, LegacyVerifiedGenerationAssessmentErrorV1> {
    let assessment_bytes = serde_json::to_vec(generation_assessment)
        .map_err(|err| LegacyVerifiedGenerationAssessmentErrorV1::Serialization(err.to_string()))?;
    let semantic_bytes = serde_json::to_vec(&semantic_review_binding)
        .map_err(|err| LegacyVerifiedGenerationAssessmentErrorV1::Serialization(err.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    frame(
        &mut hasher,
        b"schema",
        LEGACY_VERIFIED_GENERATION_ASSESSMENT_SCHEMA_V1.as_bytes(),
    );
    frame(&mut hasher, b"generation_assessment", &assessment_bytes);
    frame(&mut hasher, b"semantic_review_binding", &semantic_bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn frame(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

#[derive(Debug)]
pub enum LegacyVerifiedGenerationAssessmentErrorV1 {
    Verification(LegacySemanticReviewVerificationErrorV1),
    MissingPredecessorManifest,
    MissingVerifiedReviews,
    ReviewLedgerMismatch,
    Manifest(String),
    Serialization(String),
}

impl fmt::Display for LegacyVerifiedGenerationAssessmentErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verification(err) => {
                write!(f, "verified legacy generation assessment failed: {err}")
            }
            Self::MissingPredecessorManifest => {
                write!(f, "verified legacy generation assessment requires predecessor manifest")
            }
            Self::MissingVerifiedReviews => {
                write!(f, "verified legacy generation assessment requires verified reviews")
            }
            Self::ReviewLedgerMismatch => {
                write!(f, "verified semantic wrapper does not match current review ledger")
            }
            Self::Manifest(err) => write!(f, "manifest commitment failed: {err}"),
            Self::Serialization(err) => {
                write!(f, "verified generation assessment serialization failed: {err}")
            }
        }
    }
}

impl Error for LegacyVerifiedGenerationAssessmentErrorV1 {}

impl From<LegacySemanticReviewVerificationErrorV1>
    for LegacyVerifiedGenerationAssessmentErrorV1
{
    fn from(value: LegacySemanticReviewVerificationErrorV1) -> Self {
        Self::Verification(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legacy_qualification_profile_v3::{
        build_legacy_qualification_successor_manifest_v1,
        initial_legacy_qualification_manifest_v1,
    };
    use crate::legacy_semantic_replacement::{
        legacy_claim_replacement_review_binding_v1, LegacyClaimReplacementReviewReceiptV1,
        LegacySemanticReplacementJudgmentV1, LegacySemanticReplacementReviewLedgerV1,
        LegacySemanticReviewMethodV1,
    };
    use crate::legacy_semantic_review_verification::{
        verify_legacy_semantic_replacement_reviews_v1, LegacySemanticReviewTrustEvidenceV1,
        LegacySemanticReviewVerifierV1,
    };
    use crate::standards_registry::TechnicalClaimIdV1;
    use crate::{
        build_legacy_five_platform_portfolio_v1, exhaustive_legacy_qualification_profile_v1,
    };
    use std::collections::BTreeMap;

    struct AllowVerifier;

    impl LegacySemanticReviewVerifierV1 for AllowVerifier {
        fn verifier_profile(&self) -> &str {
            "bound-assessment-verifier-v1"
        }

        fn verify_claim_receipt(
            &self,
            receipt: &LegacyClaimReplacementReviewReceiptV1,
        ) -> Result<LegacySemanticReviewTrustEvidenceV1, String> {
            Ok(LegacySemanticReviewTrustEvidenceV1 {
                trust_evidence_blake3: blake3::hash(receipt.replacement_binding_blake3.as_bytes())
                    .to_hex()
                    .to_string(),
            })
        }

        fn verify_procedure_receipt(
            &self,
            receipt: &crate::legacy_semantic_replacement::LegacyProcedureReplacementReviewReceiptV1,
        ) -> Result<LegacySemanticReviewTrustEvidenceV1, String> {
            Ok(LegacySemanticReviewTrustEvidenceV1 {
                trust_evidence_blake3: blake3::hash(receipt.replacement_binding_blake3.as_bytes())
                    .to_hex()
                    .to_string(),
            })
        }
    }

    #[test]
    fn generation_two_persists_semantic_trust_lineage_in_assessment() {
        let (mut pack, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let profile = exhaustive_legacy_qualification_profile_v1();
        let predecessor = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        let old = pack.sources.claims().next().unwrap().clone();
        let new_id = TechnicalClaimIdV1(format!("{}:bound", old.id.0));
        let mut new_claim = old.clone();
        new_claim.id = new_id.clone();
        new_claim.statement = format!("{} [bound rebaseline]", old.statement);
        pack.sources.register_claim(new_claim.clone()).unwrap();
        let successor = build_legacy_qualification_successor_manifest_v1(
            &pack,
            &predecessor,
            BTreeMap::from([(old.id.clone(), new_id.clone())]),
            BTreeMap::new(),
        )
        .unwrap();
        let predecessor_digest = legacy_qualification_manifest_commitment_v1(&predecessor).unwrap();
        let successor_digest = legacy_qualification_manifest_commitment_v1(&successor).unwrap();
        let basis = "e".repeat(64);
        let reviewed_at = 1_800_000_000_500;
        let binding = legacy_claim_replacement_review_binding_v1(
            &old,
            &new_claim,
            &predecessor_digest,
            &successor_digest,
            LegacySemanticReplacementJudgmentV1::Equivalent,
            LegacySemanticReviewMethodV1::HumanReview,
            "bound-assessment-human-review-v1",
            &basis,
            reviewed_at,
        )
        .unwrap();
        let mut reviews = LegacySemanticReplacementReviewLedgerV1::new();
        reviews
            .register_claim_receipt(
                &pack,
                &predecessor,
                &successor,
                LegacyClaimReplacementReviewReceiptV1 {
                    predecessor_claim_id: old.id,
                    successor_claim_id: new_id,
                    predecessor_manifest_blake3: predecessor_digest,
                    successor_manifest_blake3: successor_digest,
                    judgment: LegacySemanticReplacementJudgmentV1::Equivalent,
                    review_method: LegacySemanticReviewMethodV1::HumanReview,
                    reviewer_profile: "bound-assessment-human-review-v1".into(),
                    review_basis_blake3: basis,
                    reviewed_at_unix_ms: reviewed_at,
                    replacement_binding_blake3: binding,
                },
            )
            .unwrap();
        let verified = verify_legacy_semantic_replacement_reviews_v1(
            &pack,
            &predecessor,
            &successor,
            &reviews,
            &AllowVerifier,
            1_800_000_000_600,
        )
        .unwrap();
        let result = assess_legacy_qualification_generation_with_bound_verified_semantic_review_v1(
            &pack,
            &profile,
            &matrix,
            &LegacySourceArtifactLedgerV1::new(),
            &LegacyQualificationSourceLedgerV3::new(),
            Some(&predecessor),
            &successor,
            &reviews,
            Some(&verified),
        )
        .unwrap();
        let semantic = result.semantic_review_binding.as_ref().unwrap();
        assert_eq!(semantic.semantic_review_ledger_blake3, verified.ledger_blake3());
        assert_eq!(semantic.semantic_trust_bundle_blake3, verified.trust_bundle_blake3());
        assert_eq!(semantic.verifier_profile, verified.verifier_profile());
        assert_eq!(semantic.successor_manifest_blake3, result.generation_assessment.manifest_blake3);
        assert_eq!(
            result.assessment_binding_blake3,
            legacy_verified_generation_assessment_binding_v1(
                &result.generation_assessment,
                result.semantic_review_binding.as_ref(),
            )
            .unwrap()
        );
    }

    #[test]
    fn generation_one_has_no_semantic_trust_binding() {
        let (pack, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let profile = exhaustive_legacy_qualification_profile_v1();
        let manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        let result = assess_legacy_qualification_generation_with_bound_verified_semantic_review_v1(
            &pack,
            &profile,
            &matrix,
            &LegacySourceArtifactLedgerV1::new(),
            &LegacyQualificationSourceLedgerV3::new(),
            None,
            &manifest,
            &LegacySemanticReplacementReviewLedgerV1::new(),
            None,
        )
        .unwrap();
        assert!(result.semantic_review_binding.is_none());
        assert_eq!(result.generation_assessment.manifest_generation, 1);
    }
}
