// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! External trust boundary for semantic replacement review receipts.
//!
//! Serialized semantic receipts are audit records, not trusted review authority.
//! An injected verifier must validate the external reviewer/profile evidence before
//! a runtime-only verified wrapper can be used by the strict qualification path.
//!
//! ```text
//! serialized semantic receipt != trusted reviewer evidence
//! trusted reviewer evidence != semantic equivalence
//! semantic equivalence != source evidence
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
use crate::legacy_semantic_replacement::{
    assess_legacy_qualification_generation_with_semantic_review_v1,
    require_legacy_semantic_replacement_equivalence_v1,
    LegacyClaimReplacementReviewReceiptV1, LegacyProcedureReplacementReviewReceiptV1,
    LegacySemanticReplacementErrorV1, LegacySemanticReplacementReviewLedgerV1,
};
use crate::legacy_source_artifacts::LegacySourceArtifactLedgerV1;
use crate::standards_registry::TechnicalClaimIdV1;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

pub const LEGACY_SEMANTIC_REVIEW_VERIFICATION_SCHEMA_V1: &str =
    "symthaea-it-legacy-semantic-review-verification-v1";

/// External verifier result for one receipt. The verifier owns the meaning of
/// `trust_evidence_blake3` (signature bundle, operator approval, Xenia receipt,
/// policy-engine attestation, etc.). Support only binds and rechecks it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacySemanticReviewTrustEvidenceV1 {
    pub trust_evidence_blake3: String,
}

pub trait LegacySemanticReviewVerifierV1 {
    fn verifier_profile(&self) -> &str;

    fn verify_claim_receipt(
        &self,
        receipt: &LegacyClaimReplacementReviewReceiptV1,
    ) -> Result<LegacySemanticReviewTrustEvidenceV1, String>;

    fn verify_procedure_receipt(
        &self,
        receipt: &LegacyProcedureReplacementReviewReceiptV1,
    ) -> Result<LegacySemanticReviewTrustEvidenceV1, String>;
}

/// Runtime-only trust wrapper. Intentionally not Clone/Serialize/Deserialize.
/// Private fields prevent deserialization from recreating verified review state.
#[derive(Debug)]
pub struct VerifiedLegacySemanticReplacementReviewsV1 {
    ledger_blake3: String,
    predecessor_manifest_blake3: String,
    successor_manifest_blake3: String,
    verifier_profile: String,
    claim_trust_evidence: BTreeMap<TechnicalClaimIdV1, String>,
    procedure_trust_evidence: BTreeMap<String, String>,
    trust_bundle_blake3: String,
    verified_at_unix_ms: u64,
}

impl VerifiedLegacySemanticReplacementReviewsV1 {
    pub fn ledger_blake3(&self) -> &str {
        &self.ledger_blake3
    }

    pub fn verifier_profile(&self) -> &str {
        &self.verifier_profile
    }

    pub fn trust_bundle_blake3(&self) -> &str {
        &self.trust_bundle_blake3
    }

    pub fn verified_at_unix_ms(&self) -> u64 {
        self.verified_at_unix_ms
    }
}

pub fn legacy_semantic_replacement_review_ledger_commitment_v1(
    reviews: &LegacySemanticReplacementReviewLedgerV1,
) -> Result<String, LegacySemanticReviewVerificationErrorV1> {
    let encoded = serde_json::to_vec(reviews)
        .map_err(|err| LegacySemanticReviewVerificationErrorV1::Serialization(err.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    frame(
        &mut hasher,
        b"schema",
        LEGACY_SEMANTIC_REVIEW_VERIFICATION_SCHEMA_V1.as_bytes(),
    );
    frame(&mut hasher, b"review_ledger", &encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

pub fn verify_legacy_semantic_replacement_reviews_v1<V: LegacySemanticReviewVerifierV1>(
    pack: &LegacyComputingPackV1,
    predecessor: &LegacyQualificationManifestV1,
    successor: &LegacyQualificationManifestV1,
    reviews: &LegacySemanticReplacementReviewLedgerV1,
    verifier: &V,
    verified_at_unix_ms: u64,
) -> Result<VerifiedLegacySemanticReplacementReviewsV1, LegacySemanticReviewVerificationErrorV1> {
    if verified_at_unix_ms == 0 {
        return Err(LegacySemanticReviewVerificationErrorV1::InvalidField(
            "verification timestamp must be non-zero".into(),
        ));
    }
    let verifier_profile = verifier.verifier_profile().trim();
    if verifier_profile.is_empty() {
        return Err(LegacySemanticReviewVerificationErrorV1::InvalidField(
            "verifier profile must be non-empty".into(),
        ));
    }

    // First prove structural completeness + semantic equivalence internally.
    let assessment = require_legacy_semantic_replacement_equivalence_v1(
        pack,
        predecessor,
        successor,
        reviews,
    )?;

    let mut latest_reviewed_at_unix_ms = 0u64;
    let mut claim_trust_evidence = BTreeMap::new();
    for old_id in successor.claim_replacements.keys() {
        let receipt = reviews.claim_receipt(old_id).ok_or_else(|| {
            LegacySemanticReviewVerificationErrorV1::MissingClaimReceipt(old_id.clone())
        })?;
        latest_reviewed_at_unix_ms = latest_reviewed_at_unix_ms.max(receipt.reviewed_at_unix_ms);
        let evidence = verifier
            .verify_claim_receipt(receipt)
            .map_err(|message| LegacySemanticReviewVerificationErrorV1::VerifierRejected {
                subject: old_id.0.clone(),
                message,
            })?;
        require_blake3(&evidence.trust_evidence_blake3, "claim trust evidence digest")?;
        claim_trust_evidence.insert(
            old_id.clone(),
            evidence.trust_evidence_blake3.trim().to_ascii_lowercase(),
        );
    }

    let mut procedure_trust_evidence = BTreeMap::new();
    for old_id in successor.procedure_replacements.keys() {
        let receipt = reviews.procedure_receipt(old_id).ok_or_else(|| {
            LegacySemanticReviewVerificationErrorV1::MissingProcedureReceipt(old_id.clone())
        })?;
        latest_reviewed_at_unix_ms = latest_reviewed_at_unix_ms.max(receipt.reviewed_at_unix_ms);
        let evidence = verifier
            .verify_procedure_receipt(receipt)
            .map_err(|message| LegacySemanticReviewVerificationErrorV1::VerifierRejected {
                subject: old_id.clone(),
                message,
            })?;
        require_blake3(
            &evidence.trust_evidence_blake3,
            "procedure trust evidence digest",
        )?;
        procedure_trust_evidence.insert(
            old_id.clone(),
            evidence.trust_evidence_blake3.trim().to_ascii_lowercase(),
        );
    }

    if verified_at_unix_ms < latest_reviewed_at_unix_ms {
        return Err(
            LegacySemanticReviewVerificationErrorV1::VerificationPredatesSemanticReview {
                latest_reviewed_at_unix_ms,
                verified_at_unix_ms,
            },
        );
    }

    let ledger_blake3 = legacy_semantic_replacement_review_ledger_commitment_v1(reviews)?;
    let predecessor_manifest_blake3 = assessment.predecessor_manifest_blake3;
    let successor_manifest_blake3 = assessment.successor_manifest_blake3;
    let trust_bundle_blake3 = trust_bundle_commitment(
        &ledger_blake3,
        &predecessor_manifest_blake3,
        &successor_manifest_blake3,
        verifier_profile,
        &claim_trust_evidence,
        &procedure_trust_evidence,
        verified_at_unix_ms,
    )?;

    Ok(VerifiedLegacySemanticReplacementReviewsV1 {
        ledger_blake3,
        predecessor_manifest_blake3,
        successor_manifest_blake3,
        verifier_profile: verifier_profile.into(),
        claim_trust_evidence,
        procedure_trust_evidence,
        trust_bundle_blake3,
        verified_at_unix_ms,
    })
}

/// Revalidate the runtime wrapper immediately before qualification. This catches
/// review-ledger replacement, manifest drift, or wrapper/ledger substitution.
pub fn validate_verified_legacy_semantic_replacement_reviews_v1(
    pack: &LegacyComputingPackV1,
    predecessor: &LegacyQualificationManifestV1,
    successor: &LegacyQualificationManifestV1,
    reviews: &LegacySemanticReplacementReviewLedgerV1,
    verified: &VerifiedLegacySemanticReplacementReviewsV1,
) -> Result<(), LegacySemanticReviewVerificationErrorV1> {
    require_legacy_semantic_replacement_equivalence_v1(pack, predecessor, successor, reviews)?;
    let ledger_blake3 = legacy_semantic_replacement_review_ledger_commitment_v1(reviews)?;
    if ledger_blake3 != verified.ledger_blake3 {
        return Err(LegacySemanticReviewVerificationErrorV1::VerifiedLedgerMismatch);
    }
    let predecessor_manifest_blake3 = legacy_qualification_manifest_commitment_v1(predecessor)
        .map_err(LegacySemanticReplacementErrorV1::from)?;
    let successor_manifest_blake3 = legacy_qualification_manifest_commitment_v1(successor)
        .map_err(LegacySemanticReplacementErrorV1::from)?;
    if predecessor_manifest_blake3 != verified.predecessor_manifest_blake3 {
        return Err(
            LegacySemanticReviewVerificationErrorV1::VerifiedPredecessorManifestMismatch,
        );
    }
    if successor_manifest_blake3 != verified.successor_manifest_blake3 {
        return Err(LegacySemanticReviewVerificationErrorV1::VerifiedSuccessorManifestMismatch);
    }
    let expected_bundle = trust_bundle_commitment(
        &verified.ledger_blake3,
        &verified.predecessor_manifest_blake3,
        &verified.successor_manifest_blake3,
        &verified.verifier_profile,
        &verified.claim_trust_evidence,
        &verified.procedure_trust_evidence,
        verified.verified_at_unix_ms,
    )?;
    if expected_bundle != verified.trust_bundle_blake3 {
        return Err(LegacySemanticReviewVerificationErrorV1::TrustBundleMismatch);
    }
    Ok(())
}

/// Strongest V1 rebaseline assessment path. Generation 1 has no semantic
/// replacements and therefore requires `verified_reviews=None`. Later generations
/// require the runtime-only externally verified wrapper.
pub fn assess_legacy_qualification_generation_with_verified_semantic_review_v1(
    pack: &LegacyComputingPackV1,
    profile: &LegacyQualificationProfileV1,
    matrix: &ItQualificationMatrixV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    source_ledger: &LegacyQualificationSourceLedgerV3,
    predecessor_manifest: Option<&LegacyQualificationManifestV1>,
    manifest: &LegacyQualificationManifestV1,
    semantic_reviews: &LegacySemanticReplacementReviewLedgerV1,
    verified_reviews: Option<&VerifiedLegacySemanticReplacementReviewsV1>,
) -> Result<LegacyQualificationGenerationAssessmentV1, LegacySemanticReviewVerificationErrorV1> {
    match manifest.generation {
        1 => {
            if verified_reviews.is_some() {
                return Err(
                    LegacySemanticReviewVerificationErrorV1::UnexpectedInitialVerification,
                );
            }
        }
        _ => {
            let predecessor = predecessor_manifest.ok_or(
                LegacySemanticReviewVerificationErrorV1::MissingPredecessorManifest,
            )?;
            let verified = verified_reviews.ok_or(
                LegacySemanticReviewVerificationErrorV1::MissingVerifiedReviews,
            )?;
            validate_verified_legacy_semantic_replacement_reviews_v1(
                pack,
                predecessor,
                manifest,
                semantic_reviews,
                verified,
            )?;
        }
    }

    assess_legacy_qualification_generation_with_semantic_review_v1(
        pack,
        profile,
        matrix,
        artifacts,
        source_ledger,
        predecessor_manifest,
        manifest,
        semantic_reviews,
    )
    .map_err(LegacySemanticReviewVerificationErrorV1::Semantic)
}

fn trust_bundle_commitment(
    ledger_blake3: &str,
    predecessor_manifest_blake3: &str,
    successor_manifest_blake3: &str,
    verifier_profile: &str,
    claim_trust_evidence: &BTreeMap<TechnicalClaimIdV1, String>,
    procedure_trust_evidence: &BTreeMap<String, String>,
    verified_at_unix_ms: u64,
) -> Result<String, LegacySemanticReviewVerificationErrorV1> {
    require_blake3(ledger_blake3, "review ledger digest")?;
    require_blake3(predecessor_manifest_blake3, "predecessor manifest digest")?;
    require_blake3(successor_manifest_blake3, "successor manifest digest")?;
    if verifier_profile.trim().is_empty() {
        return Err(LegacySemanticReviewVerificationErrorV1::InvalidField(
            "verifier profile must be non-empty".into(),
        ));
    }
    let claim_bytes = serde_json::to_vec(claim_trust_evidence)
        .map_err(|err| LegacySemanticReviewVerificationErrorV1::Serialization(err.to_string()))?;
    let procedure_bytes = serde_json::to_vec(procedure_trust_evidence)
        .map_err(|err| LegacySemanticReviewVerificationErrorV1::Serialization(err.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    frame(
        &mut hasher,
        b"schema",
        LEGACY_SEMANTIC_REVIEW_VERIFICATION_SCHEMA_V1.as_bytes(),
    );
    frame(&mut hasher, b"review_ledger", ledger_blake3.as_bytes());
    frame(
        &mut hasher,
        b"predecessor_manifest",
        predecessor_manifest_blake3.as_bytes(),
    );
    frame(
        &mut hasher,
        b"successor_manifest",
        successor_manifest_blake3.as_bytes(),
    );
    frame(&mut hasher, b"verifier_profile", verifier_profile.as_bytes());
    frame(&mut hasher, b"claim_trust_evidence", &claim_bytes);
    frame(&mut hasher, b"procedure_trust_evidence", &procedure_bytes);
    frame(
        &mut hasher,
        b"verified_at_unix_ms",
        &verified_at_unix_ms.to_le_bytes(),
    );
    Ok(hasher.finalize().to_hex().to_string())
}

fn require_blake3(
    value: &str,
    field: &'static str,
) -> Result<(), LegacySemanticReviewVerificationErrorV1> {
    let value = value.trim();
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(LegacySemanticReviewVerificationErrorV1::InvalidField(
            format!("{field} must be a 64-character hex BLAKE3 digest"),
        ));
    }
    Ok(())
}

fn frame(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

#[derive(Debug)]
pub enum LegacySemanticReviewVerificationErrorV1 {
    Semantic(LegacySemanticReplacementErrorV1),
    InvalidField(String),
    Serialization(String),
    MissingClaimReceipt(TechnicalClaimIdV1),
    MissingProcedureReceipt(String),
    VerifierRejected { subject: String, message: String },
    VerificationPredatesSemanticReview {
        latest_reviewed_at_unix_ms: u64,
        verified_at_unix_ms: u64,
    },
    VerifiedLedgerMismatch,
    VerifiedPredecessorManifestMismatch,
    VerifiedSuccessorManifestMismatch,
    TrustBundleMismatch,
    MissingPredecessorManifest,
    MissingVerifiedReviews,
    UnexpectedInitialVerification,
}

impl fmt::Display for LegacySemanticReviewVerificationErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Semantic(err) => write!(f, "legacy semantic review verification failed: {err}"),
            Self::InvalidField(value) => write!(f, "invalid semantic review verification field: {value}"),
            Self::Serialization(value) => write!(f, "semantic review verification serialization failed: {value}"),
            Self::MissingClaimReceipt(id) => write!(f, "missing semantic claim review receipt {}", id.0),
            Self::MissingProcedureReceipt(id) => write!(f, "missing semantic procedure review receipt {id}"),
            Self::VerifierRejected { subject, message } => write!(f, "external semantic verifier rejected {subject}: {message}"),
            Self::VerificationPredatesSemanticReview { latest_reviewed_at_unix_ms, verified_at_unix_ms } => write!(f, "semantic verification timestamp {verified_at_unix_ms} predates latest review {latest_reviewed_at_unix_ms}"),
            Self::VerifiedLedgerMismatch => write!(f, "verified semantic review wrapper does not match current review ledger"),
            Self::VerifiedPredecessorManifestMismatch => write!(f, "verified semantic review wrapper predecessor manifest changed"),
            Self::VerifiedSuccessorManifestMismatch => write!(f, "verified semantic review wrapper successor manifest changed"),
            Self::TrustBundleMismatch => write!(f, "verified semantic review trust bundle commitment mismatched"),
            Self::MissingPredecessorManifest => write!(f, "verified semantic review requires predecessor manifest"),
            Self::MissingVerifiedReviews => write!(f, "qualification generation is missing externally verified semantic reviews"),
            Self::UnexpectedInitialVerification => write!(f, "generation 1 cannot carry semantic replacement verification"),
        }
    }
}

impl Error for LegacySemanticReviewVerificationErrorV1 {}

impl From<LegacySemanticReplacementErrorV1> for LegacySemanticReviewVerificationErrorV1 {
    fn from(value: LegacySemanticReplacementErrorV1) -> Self {
        Self::Semantic(value)
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
        legacy_claim_replacement_review_binding_v1,
        LegacyClaimReplacementReviewReceiptV1,
        LegacySemanticReplacementJudgmentV1,
        LegacySemanticReviewMethodV1,
    };
    use crate::standards_registry::TechnicalClaimIdV1;
    use crate::build_legacy_five_platform_portfolio_v1;
    use std::collections::BTreeMap;

    struct AllowVerifier;

    impl LegacySemanticReviewVerifierV1 for AllowVerifier {
        fn verifier_profile(&self) -> &str {
            "test-external-semantic-verifier-v1"
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
            receipt: &LegacyProcedureReplacementReviewReceiptV1,
        ) -> Result<LegacySemanticReviewTrustEvidenceV1, String> {
            Ok(LegacySemanticReviewTrustEvidenceV1 {
                trust_evidence_blake3: blake3::hash(receipt.replacement_binding_blake3.as_bytes())
                    .to_hex()
                    .to_string(),
            })
        }
    }

    struct RejectVerifier;

    impl LegacySemanticReviewVerifierV1 for RejectVerifier {
        fn verifier_profile(&self) -> &str {
            "test-reject-semantic-verifier-v1"
        }

        fn verify_claim_receipt(
            &self,
            _receipt: &LegacyClaimReplacementReviewReceiptV1,
        ) -> Result<LegacySemanticReviewTrustEvidenceV1, String> {
            Err("untrusted reviewer evidence".into())
        }

        fn verify_procedure_receipt(
            &self,
            _receipt: &LegacyProcedureReplacementReviewReceiptV1,
        ) -> Result<LegacySemanticReviewTrustEvidenceV1, String> {
            Err("untrusted reviewer evidence".into())
        }
    }

    fn claim_review_fixture() -> (
        LegacyComputingPackV1,
        LegacyQualificationManifestV1,
        LegacyQualificationManifestV1,
        LegacySemanticReplacementReviewLedgerV1,
    ) {
        let (mut pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let predecessor = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        let old = pack.sources.claims().next().unwrap().clone();
        let new_id = TechnicalClaimIdV1(format!("{}:verified-rebaseline", old.id.0));
        let mut new_claim = old.clone();
        new_claim.id = new_id.clone();
        new_claim.statement = format!("{} [verified rebaseline]", old.statement);
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
        let basis = "d".repeat(64);
        let reviewed_at = 1_800_000_000_300;
        let binding = legacy_claim_replacement_review_binding_v1(
            &old,
            &new_claim,
            &predecessor_digest,
            &successor_digest,
            LegacySemanticReplacementJudgmentV1::Equivalent,
            LegacySemanticReviewMethodV1::HumanReview,
            "test-human-reviewer-v1",
            &basis,
            reviewed_at,
        )
        .unwrap();
        let receipt = LegacyClaimReplacementReviewReceiptV1 {
            predecessor_claim_id: old.id,
            successor_claim_id: new_id,
            predecessor_manifest_blake3: predecessor_digest,
            successor_manifest_blake3: successor_digest,
            judgment: LegacySemanticReplacementJudgmentV1::Equivalent,
            review_method: LegacySemanticReviewMethodV1::HumanReview,
            reviewer_profile: "test-human-reviewer-v1".into(),
            review_basis_blake3: basis,
            reviewed_at_unix_ms: reviewed_at,
            replacement_binding_blake3: binding,
        };
        let mut reviews = LegacySemanticReplacementReviewLedgerV1::new();
        reviews
            .register_claim_receipt(&pack, &predecessor, &successor, receipt)
            .unwrap();
        (pack, predecessor, successor, reviews)
    }

    #[test]
    fn external_verifier_creates_runtime_only_bound_wrapper() {
        let (pack, predecessor, successor, reviews) = claim_review_fixture();
        let verified = verify_legacy_semantic_replacement_reviews_v1(
            &pack,
            &predecessor,
            &successor,
            &reviews,
            &AllowVerifier,
            1_800_000_000_400,
        )
        .unwrap();
        assert_eq!(verified.verifier_profile(), "test-external-semantic-verifier-v1");
        assert_eq!(
            verified.ledger_blake3(),
            legacy_semantic_replacement_review_ledger_commitment_v1(&reviews).unwrap()
        );
        validate_verified_legacy_semantic_replacement_reviews_v1(
            &pack,
            &predecessor,
            &successor,
            &reviews,
            &verified,
        )
        .unwrap();
    }

    #[test]
    fn verifier_rejection_fails_closed() {
        let (pack, predecessor, successor, reviews) = claim_review_fixture();
        assert!(matches!(
            verify_legacy_semantic_replacement_reviews_v1(
                &pack,
                &predecessor,
                &successor,
                &reviews,
                &RejectVerifier,
                1_800_000_000_400,
            ),
            Err(LegacySemanticReviewVerificationErrorV1::VerifierRejected { .. })
        ));
    }

    #[test]
    fn verification_cannot_predate_semantic_review() {
        let (pack, predecessor, successor, reviews) = claim_review_fixture();
        assert!(matches!(
            verify_legacy_semantic_replacement_reviews_v1(
                &pack,
                &predecessor,
                &successor,
                &reviews,
                &AllowVerifier,
                1_800_000_000_299,
            ),
            Err(
                LegacySemanticReviewVerificationErrorV1::VerificationPredatesSemanticReview {
                    ..
                }
            )
        ));
    }
}
