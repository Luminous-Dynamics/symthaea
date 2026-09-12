// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic review receipts for generation-bound legacy qualification replacements.
//!
//! Structural manifest compatibility is necessary but not sufficient to carry a
//! qualification role across a rebaseline. A replacement can preserve platform,
//! applicability, category, authority footprint, and source lineage while still
//! changing the meaning of the claim or procedure.
//!
//! This module therefore makes semantic review an explicit, digest-bound step:
//!
//! ```text
//! structural replacement != semantic equivalence
//! semantic review record != qualification admission
//! Equivalent judgment + complete exact receipts -> admissible carry-forward
//! Narrower/Broader/ChangedMeaning -> new qualification role required
//! ```
//!
//! The review receipt contains no authority token and does not implement reviewer
//! identity/signatures. `reviewer_profile` and `review_basis_blake3` are bindings
//! to externally governed review evidence.

use crate::it_qualification::ItQualificationMatrixV1;
use crate::legacy_computing::{LegacyComputingPackV1, LegacyProcedureV1};
use crate::legacy_qualification_profile::LegacyQualificationProfileV1;
use crate::legacy_qualification_profile_v3::{
    assess_legacy_qualification_generation_v1,
    legacy_qualification_manifest_commitment_v1,
    validate_legacy_qualification_manifest_transition_v1,
    LegacyQualificationGenerationAssessmentV1,
    LegacyQualificationManifestErrorV1,
    LegacyQualificationManifestV1,
    LegacyQualificationProfileErrorV3,
};
use crate::legacy_qualification_source_ledger_v3::LegacyQualificationSourceLedgerV3;
use crate::legacy_source_artifacts::LegacySourceArtifactLedgerV1;
use crate::standards_registry::{TechnicalClaimIdV1, TechnicalKnowledgeClaimV1};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_SEMANTIC_REPLACEMENT_LEDGER_SCHEMA_V1: &str =
    "symthaea-it-legacy-semantic-replacement-ledger-v1";

/// Semantic relationship between the predecessor and successor object.
/// Only `Equivalent` may inherit an existing qualification role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegacySemanticReplacementJudgmentV1 {
    Equivalent,
    Narrower,
    Broader,
    ChangedMeaning,
}

/// V1 deliberately requires human semantic review. Deterministic tooling may
/// assist, but cannot be the sole basis for semantic-equivalence admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegacySemanticReviewMethodV1 {
    HumanReview,
    HumanAndDeterministic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyClaimReplacementReviewReceiptV1 {
    pub predecessor_claim_id: TechnicalClaimIdV1,
    pub successor_claim_id: TechnicalClaimIdV1,
    pub predecessor_manifest_blake3: String,
    pub successor_manifest_blake3: String,
    pub judgment: LegacySemanticReplacementJudgmentV1,
    pub review_method: LegacySemanticReviewMethodV1,
    pub reviewer_profile: String,
    /// Digest of external review notes/evidence. The notes themselves need not be
    /// embedded in the repository.
    pub review_basis_blake3: String,
    pub reviewed_at_unix_ms: u64,
    /// BLAKE3 over exact old/new typed content + all receipt metadata above.
    pub replacement_binding_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyProcedureReplacementReviewReceiptV1 {
    pub predecessor_procedure_id: String,
    pub successor_procedure_id: String,
    pub predecessor_manifest_blake3: String,
    pub successor_manifest_blake3: String,
    pub judgment: LegacySemanticReplacementJudgmentV1,
    pub review_method: LegacySemanticReviewMethodV1,
    pub reviewer_profile: String,
    pub review_basis_blake3: String,
    pub reviewed_at_unix_ms: u64,
    pub replacement_binding_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySemanticReplacementReviewLedgerV1 {
    pub schema_version: String,
    claim_receipts: BTreeMap<TechnicalClaimIdV1, LegacyClaimReplacementReviewReceiptV1>,
    procedure_receipts: BTreeMap<String, LegacyProcedureReplacementReviewReceiptV1>,
}

impl Default for LegacySemanticReplacementReviewLedgerV1 {
    fn default() -> Self {
        Self {
            schema_version: LEGACY_SEMANTIC_REPLACEMENT_LEDGER_SCHEMA_V1.into(),
            claim_receipts: BTreeMap::new(),
            procedure_receipts: BTreeMap::new(),
        }
    }
}

impl LegacySemanticReplacementReviewLedgerV1 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn claim_receipt(
        &self,
        predecessor_claim_id: &TechnicalClaimIdV1,
    ) -> Option<&LegacyClaimReplacementReviewReceiptV1> {
        self.claim_receipts.get(predecessor_claim_id)
    }

    pub fn procedure_receipt(
        &self,
        predecessor_procedure_id: &str,
    ) -> Option<&LegacyProcedureReplacementReviewReceiptV1> {
        self.procedure_receipts.get(predecessor_procedure_id)
    }

    pub fn register_claim_receipt(
        &mut self,
        pack: &LegacyComputingPackV1,
        predecessor: &LegacyQualificationManifestV1,
        successor: &LegacyQualificationManifestV1,
        receipt: LegacyClaimReplacementReviewReceiptV1,
    ) -> Result<bool, LegacySemanticReplacementErrorV1> {
        self.validate_schema()?;
        validate_legacy_claim_replacement_review_receipt_v1(
            pack,
            predecessor,
            successor,
            &receipt,
        )?;
        if let Some(existing) = self.claim_receipts.get(&receipt.predecessor_claim_id) {
            if existing == &receipt {
                return Ok(false);
            }
            return Err(LegacySemanticReplacementErrorV1::ClaimReceiptConflict(
                receipt.predecessor_claim_id,
            ));
        }
        self.claim_receipts
            .insert(receipt.predecessor_claim_id.clone(), receipt);
        Ok(true)
    }

    pub fn register_procedure_receipt(
        &mut self,
        pack: &LegacyComputingPackV1,
        predecessor: &LegacyQualificationManifestV1,
        successor: &LegacyQualificationManifestV1,
        receipt: LegacyProcedureReplacementReviewReceiptV1,
    ) -> Result<bool, LegacySemanticReplacementErrorV1> {
        self.validate_schema()?;
        validate_legacy_procedure_replacement_review_receipt_v1(
            pack,
            predecessor,
            successor,
            &receipt,
        )?;
        if let Some(existing) = self
            .procedure_receipts
            .get(&receipt.predecessor_procedure_id)
        {
            if existing == &receipt {
                return Ok(false);
            }
            return Err(
                LegacySemanticReplacementErrorV1::ProcedureReceiptConflict(
                    receipt.predecessor_procedure_id,
                ),
            );
        }
        self.procedure_receipts
            .insert(receipt.predecessor_procedure_id.clone(), receipt);
        Ok(true)
    }

    pub fn validate_complete_equivalence(
        &self,
        pack: &LegacyComputingPackV1,
        predecessor: &LegacyQualificationManifestV1,
        successor: &LegacyQualificationManifestV1,
    ) -> Result<LegacySemanticReplacementAssessmentV1, LegacySemanticReplacementErrorV1> {
        self.validate_schema()?;
        validate_legacy_qualification_manifest_transition_v1(pack, predecessor, successor)?;

        let expected_claims = successor
            .claim_replacements
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let expected_procedures = successor
            .procedure_replacements
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();

        let missing_claim_receipts = expected_claims
            .iter()
            .filter(|id| self.claim_receipt(id).is_none())
            .cloned()
            .collect::<BTreeSet<_>>();
        let missing_procedure_receipts = expected_procedures
            .iter()
            .filter(|id| self.procedure_receipt(id).is_none())
            .cloned()
            .collect::<BTreeSet<_>>();

        let extraneous_claim_receipts = self
            .claim_receipts
            .keys()
            .filter(|id| !expected_claims.contains(*id))
            .cloned()
            .collect::<BTreeSet<_>>();
        let extraneous_procedure_receipts = self
            .procedure_receipts
            .keys()
            .filter(|id| !expected_procedures.contains(*id))
            .cloned()
            .collect::<BTreeSet<_>>();

        let mut non_equivalent_claims = BTreeSet::new();
        let mut non_equivalent_procedures = BTreeSet::new();

        for id in &expected_claims {
            if let Some(receipt) = self.claim_receipt(id) {
                validate_legacy_claim_replacement_review_receipt_v1(
                    pack,
                    predecessor,
                    successor,
                    receipt,
                )?;
                if receipt.judgment != LegacySemanticReplacementJudgmentV1::Equivalent {
                    non_equivalent_claims.insert(id.clone());
                }
            }
        }
        for id in &expected_procedures {
            if let Some(receipt) = self.procedure_receipt(id) {
                validate_legacy_procedure_replacement_review_receipt_v1(
                    pack,
                    predecessor,
                    successor,
                    receipt,
                )?;
                if receipt.judgment != LegacySemanticReplacementJudgmentV1::Equivalent {
                    non_equivalent_procedures.insert(id.clone());
                }
            }
        }

        let complete = missing_claim_receipts.is_empty()
            && missing_procedure_receipts.is_empty()
            && extraneous_claim_receipts.is_empty()
            && extraneous_procedure_receipts.is_empty();
        let semantically_equivalent = complete
            && non_equivalent_claims.is_empty()
            && non_equivalent_procedures.is_empty();

        Ok(LegacySemanticReplacementAssessmentV1 {
            schema_version: LEGACY_SEMANTIC_REPLACEMENT_LEDGER_SCHEMA_V1.into(),
            predecessor_manifest_blake3: legacy_qualification_manifest_commitment_v1(
                predecessor,
            )?,
            successor_manifest_blake3: legacy_qualification_manifest_commitment_v1(successor)?,
            required_claim_receipts: expected_claims.len(),
            required_procedure_receipts: expected_procedures.len(),
            missing_claim_receipts,
            missing_procedure_receipts,
            extraneous_claim_receipts,
            extraneous_procedure_receipts,
            non_equivalent_claims,
            non_equivalent_procedures,
            complete,
            semantically_equivalent,
        })
    }

    fn validate_schema(&self) -> Result<(), LegacySemanticReplacementErrorV1> {
        if self.schema_version != LEGACY_SEMANTIC_REPLACEMENT_LEDGER_SCHEMA_V1 {
            return Err(LegacySemanticReplacementErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySemanticReplacementAssessmentV1 {
    pub schema_version: String,
    pub predecessor_manifest_blake3: String,
    pub successor_manifest_blake3: String,
    pub required_claim_receipts: usize,
    pub required_procedure_receipts: usize,
    pub missing_claim_receipts: BTreeSet<TechnicalClaimIdV1>,
    pub missing_procedure_receipts: BTreeSet<String>,
    pub extraneous_claim_receipts: BTreeSet<TechnicalClaimIdV1>,
    pub extraneous_procedure_receipts: BTreeSet<String>,
    pub non_equivalent_claims: BTreeSet<TechnicalClaimIdV1>,
    pub non_equivalent_procedures: BTreeSet<String>,
    pub complete: bool,
    pub semantically_equivalent: bool,
}

pub fn legacy_claim_replacement_review_binding_v1(
    old: &TechnicalKnowledgeClaimV1,
    new: &TechnicalKnowledgeClaimV1,
    predecessor_manifest_blake3: &str,
    successor_manifest_blake3: &str,
    judgment: LegacySemanticReplacementJudgmentV1,
    review_method: LegacySemanticReviewMethodV1,
    reviewer_profile: &str,
    review_basis_blake3: &str,
    reviewed_at_unix_ms: u64,
) -> Result<String, LegacySemanticReplacementErrorV1> {
    validate_review_metadata(
        predecessor_manifest_blake3,
        successor_manifest_blake3,
        reviewer_profile,
        review_basis_blake3,
        reviewed_at_unix_ms,
    )?;
    semantic_binding(
        b"claim_replacement",
        old,
        new,
        predecessor_manifest_blake3,
        successor_manifest_blake3,
        judgment,
        review_method,
        reviewer_profile,
        review_basis_blake3,
        reviewed_at_unix_ms,
    )
}

pub fn legacy_procedure_replacement_review_binding_v1(
    old: &LegacyProcedureV1,
    new: &LegacyProcedureV1,
    predecessor_manifest_blake3: &str,
    successor_manifest_blake3: &str,
    judgment: LegacySemanticReplacementJudgmentV1,
    review_method: LegacySemanticReviewMethodV1,
    reviewer_profile: &str,
    review_basis_blake3: &str,
    reviewed_at_unix_ms: u64,
) -> Result<String, LegacySemanticReplacementErrorV1> {
    validate_review_metadata(
        predecessor_manifest_blake3,
        successor_manifest_blake3,
        reviewer_profile,
        review_basis_blake3,
        reviewed_at_unix_ms,
    )?;
    semantic_binding(
        b"procedure_replacement",
        old,
        new,
        predecessor_manifest_blake3,
        successor_manifest_blake3,
        judgment,
        review_method,
        reviewer_profile,
        review_basis_blake3,
        reviewed_at_unix_ms,
    )
}

pub fn validate_legacy_claim_replacement_review_receipt_v1(
    pack: &LegacyComputingPackV1,
    predecessor: &LegacyQualificationManifestV1,
    successor: &LegacyQualificationManifestV1,
    receipt: &LegacyClaimReplacementReviewReceiptV1,
) -> Result<(), LegacySemanticReplacementErrorV1> {
    validate_legacy_qualification_manifest_transition_v1(pack, predecessor, successor)?;
    let expected_new = successor
        .claim_replacements
        .get(&receipt.predecessor_claim_id)
        .ok_or_else(|| LegacySemanticReplacementErrorV1::UnexpectedClaimReceipt(
            receipt.predecessor_claim_id.clone(),
        ))?;
    if expected_new != &receipt.successor_claim_id {
        return Err(LegacySemanticReplacementErrorV1::ClaimReplacementMismatch {
            old: receipt.predecessor_claim_id.clone(),
            expected: expected_new.clone(),
            observed: receipt.successor_claim_id.clone(),
        });
    }
    validate_manifest_digests(
        predecessor,
        successor,
        &receipt.predecessor_manifest_blake3,
        &receipt.successor_manifest_blake3,
    )?;
    let old = pack
        .sources
        .claim(&receipt.predecessor_claim_id)
        .ok_or_else(|| LegacySemanticReplacementErrorV1::UnknownClaim(
            receipt.predecessor_claim_id.clone(),
        ))?;
    let new = pack
        .sources
        .claim(&receipt.successor_claim_id)
        .ok_or_else(|| LegacySemanticReplacementErrorV1::UnknownClaim(
            receipt.successor_claim_id.clone(),
        ))?;
    let expected = legacy_claim_replacement_review_binding_v1(
        old,
        new,
        &receipt.predecessor_manifest_blake3,
        &receipt.successor_manifest_blake3,
        receipt.judgment,
        receipt.review_method,
        &receipt.reviewer_profile,
        &receipt.review_basis_blake3,
        receipt.reviewed_at_unix_ms,
    )?;
    if expected != receipt.replacement_binding_blake3.trim().to_ascii_lowercase() {
        return Err(LegacySemanticReplacementErrorV1::BindingMismatch(
            receipt.predecessor_claim_id.0.clone(),
        ));
    }
    Ok(())
}

pub fn validate_legacy_procedure_replacement_review_receipt_v1(
    pack: &LegacyComputingPackV1,
    predecessor: &LegacyQualificationManifestV1,
    successor: &LegacyQualificationManifestV1,
    receipt: &LegacyProcedureReplacementReviewReceiptV1,
) -> Result<(), LegacySemanticReplacementErrorV1> {
    validate_legacy_qualification_manifest_transition_v1(pack, predecessor, successor)?;
    let expected_new = successor
        .procedure_replacements
        .get(&receipt.predecessor_procedure_id)
        .ok_or_else(|| LegacySemanticReplacementErrorV1::UnexpectedProcedureReceipt(
            receipt.predecessor_procedure_id.clone(),
        ))?;
    if expected_new != &receipt.successor_procedure_id {
        return Err(LegacySemanticReplacementErrorV1::ProcedureReplacementMismatch {
            old: receipt.predecessor_procedure_id.clone(),
            expected: expected_new.clone(),
            observed: receipt.successor_procedure_id.clone(),
        });
    }
    validate_manifest_digests(
        predecessor,
        successor,
        &receipt.predecessor_manifest_blake3,
        &receipt.successor_manifest_blake3,
    )?;
    let old = find_procedure(pack, &receipt.predecessor_procedure_id).ok_or_else(|| {
        LegacySemanticReplacementErrorV1::UnknownProcedure(
            receipt.predecessor_procedure_id.clone(),
        )
    })?;
    let new = find_procedure(pack, &receipt.successor_procedure_id).ok_or_else(|| {
        LegacySemanticReplacementErrorV1::UnknownProcedure(
            receipt.successor_procedure_id.clone(),
        )
    })?;
    let expected = legacy_procedure_replacement_review_binding_v1(
        old,
        new,
        &receipt.predecessor_manifest_blake3,
        &receipt.successor_manifest_blake3,
        receipt.judgment,
        receipt.review_method,
        &receipt.reviewer_profile,
        &receipt.review_basis_blake3,
        receipt.reviewed_at_unix_ms,
    )?;
    if expected != receipt.replacement_binding_blake3.trim().to_ascii_lowercase() {
        return Err(LegacySemanticReplacementErrorV1::BindingMismatch(
            receipt.predecessor_procedure_id.clone(),
        ));
    }
    Ok(())
}

/// Strict carry-forward admission. Structural compatibility is checked first,
/// then every replacement must have an exact receipt with an `Equivalent`
/// semantic judgment.
pub fn require_legacy_semantic_replacement_equivalence_v1(
    pack: &LegacyComputingPackV1,
    predecessor: &LegacyQualificationManifestV1,
    successor: &LegacyQualificationManifestV1,
    reviews: &LegacySemanticReplacementReviewLedgerV1,
) -> Result<LegacySemanticReplacementAssessmentV1, LegacySemanticReplacementErrorV1> {
    let assessment = reviews.validate_complete_equivalence(pack, predecessor, successor)?;
    if !assessment.complete {
        return Err(LegacySemanticReplacementErrorV1::IncompleteReview {
            missing_claims: assessment.missing_claim_receipts,
            missing_procedures: assessment.missing_procedure_receipts,
            extraneous_claims: assessment.extraneous_claim_receipts,
            extraneous_procedures: assessment.extraneous_procedure_receipts,
        });
    }
    if !assessment.semantically_equivalent {
        return Err(LegacySemanticReplacementErrorV1::NonEquivalentReplacement {
            claims: assessment.non_equivalent_claims,
            procedures: assessment.non_equivalent_procedures,
        });
    }
    Ok(assessment)
}

/// Preferred generation-assessment path when generation > 1 carries an existing
/// qualification role forward. Generation 1 has no replacements and therefore
/// requires an empty semantic-review ledger.
pub fn assess_legacy_qualification_generation_with_semantic_review_v1(
    pack: &LegacyComputingPackV1,
    profile: &LegacyQualificationProfileV1,
    matrix: &ItQualificationMatrixV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    source_ledger: &LegacyQualificationSourceLedgerV3,
    predecessor_manifest: Option<&LegacyQualificationManifestV1>,
    manifest: &LegacyQualificationManifestV1,
    semantic_reviews: &LegacySemanticReplacementReviewLedgerV1,
) -> Result<LegacyQualificationGenerationAssessmentV1, LegacySemanticReplacementErrorV1> {
    match manifest.generation {
        1 => {
            if !semantic_reviews.claim_receipts.is_empty()
                || !semantic_reviews.procedure_receipts.is_empty()
            {
                return Err(LegacySemanticReplacementErrorV1::UnexpectedInitialReviews);
            }
        }
        _ => {
            let predecessor = predecessor_manifest.ok_or(
                LegacySemanticReplacementErrorV1::MissingPredecessorManifest,
            )?;
            require_legacy_semantic_replacement_equivalence_v1(
                pack,
                predecessor,
                manifest,
                semantic_reviews,
            )?;
        }
    }
    assess_legacy_qualification_generation_v1(
        pack,
        profile,
        matrix,
        artifacts,
        source_ledger,
        predecessor_manifest,
        manifest,
    )
    .map_err(LegacySemanticReplacementErrorV1::Qualification)
}

fn semantic_binding<T: Serialize, U: Serialize>(
    domain: &[u8],
    old: &T,
    new: &U,
    predecessor_manifest_blake3: &str,
    successor_manifest_blake3: &str,
    judgment: LegacySemanticReplacementJudgmentV1,
    review_method: LegacySemanticReviewMethodV1,
    reviewer_profile: &str,
    review_basis_blake3: &str,
    reviewed_at_unix_ms: u64,
) -> Result<String, LegacySemanticReplacementErrorV1> {
    let old_bytes = serde_json::to_vec(old)
        .map_err(|err| LegacySemanticReplacementErrorV1::Serialization(err.to_string()))?;
    let new_bytes = serde_json::to_vec(new)
        .map_err(|err| LegacySemanticReplacementErrorV1::Serialization(err.to_string()))?;
    let judgment_bytes = serde_json::to_vec(&judgment)
        .map_err(|err| LegacySemanticReplacementErrorV1::Serialization(err.to_string()))?;
    let method_bytes = serde_json::to_vec(&review_method)
        .map_err(|err| LegacySemanticReplacementErrorV1::Serialization(err.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    frame(&mut hasher, b"schema", LEGACY_SEMANTIC_REPLACEMENT_LEDGER_SCHEMA_V1.as_bytes());
    frame(&mut hasher, b"domain", domain);
    frame(&mut hasher, b"old", &old_bytes);
    frame(&mut hasher, b"new", &new_bytes);
    frame(&mut hasher, b"predecessor_manifest", predecessor_manifest_blake3.as_bytes());
    frame(&mut hasher, b"successor_manifest", successor_manifest_blake3.as_bytes());
    frame(&mut hasher, b"judgment", &judgment_bytes);
    frame(&mut hasher, b"review_method", &method_bytes);
    frame(&mut hasher, b"reviewer_profile", reviewer_profile.as_bytes());
    frame(&mut hasher, b"review_basis", review_basis_blake3.as_bytes());
    frame(&mut hasher, b"reviewed_at_unix_ms", &reviewed_at_unix_ms.to_le_bytes());
    Ok(hasher.finalize().to_hex().to_string())
}

fn validate_manifest_digests(
    predecessor: &LegacyQualificationManifestV1,
    successor: &LegacyQualificationManifestV1,
    predecessor_digest: &str,
    successor_digest: &str,
) -> Result<(), LegacySemanticReplacementErrorV1> {
    let expected_predecessor = legacy_qualification_manifest_commitment_v1(predecessor)?;
    let expected_successor = legacy_qualification_manifest_commitment_v1(successor)?;
    if predecessor_digest.trim().to_ascii_lowercase() != expected_predecessor {
        return Err(LegacySemanticReplacementErrorV1::PredecessorManifestMismatch);
    }
    if successor_digest.trim().to_ascii_lowercase() != expected_successor {
        return Err(LegacySemanticReplacementErrorV1::SuccessorManifestMismatch);
    }
    Ok(())
}

fn validate_review_metadata(
    predecessor_manifest_blake3: &str,
    successor_manifest_blake3: &str,
    reviewer_profile: &str,
    review_basis_blake3: &str,
    reviewed_at_unix_ms: u64,
) -> Result<(), LegacySemanticReplacementErrorV1> {
    require_blake3(predecessor_manifest_blake3, "predecessor manifest digest")?;
    require_blake3(successor_manifest_blake3, "successor manifest digest")?;
    require_nonempty(reviewer_profile, "reviewer profile")?;
    require_blake3(review_basis_blake3, "review basis digest")?;
    if reviewed_at_unix_ms == 0 {
        return Err(LegacySemanticReplacementErrorV1::InvalidField(
            "semantic review timestamp must be non-zero".into(),
        ));
    }
    Ok(())
}

fn find_procedure<'a>(pack: &'a LegacyComputingPackV1, id: &str) -> Option<&'a LegacyProcedureV1> {
    pack.procedures.iter().find(|procedure| procedure.id == id)
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), LegacySemanticReplacementErrorV1> {
    if value.trim().is_empty() {
        return Err(LegacySemanticReplacementErrorV1::InvalidField(format!(
            "{field} must be non-empty"
        )));
    }
    Ok(())
}

fn require_blake3(value: &str, field: &'static str) -> Result<(), LegacySemanticReplacementErrorV1> {
    let value = value.trim();
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(LegacySemanticReplacementErrorV1::InvalidField(format!(
            "{field} must be a 64-character hex BLAKE3 digest"
        )));
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
pub enum LegacySemanticReplacementErrorV1 {
    Manifest(LegacyQualificationManifestErrorV1),
    Qualification(LegacyQualificationProfileErrorV3),
    UnsupportedSchema(String),
    InvalidField(String),
    Serialization(String),
    UnknownClaim(TechnicalClaimIdV1),
    UnknownProcedure(String),
    UnexpectedClaimReceipt(TechnicalClaimIdV1),
    UnexpectedProcedureReceipt(String),
    ClaimReplacementMismatch {
        old: TechnicalClaimIdV1,
        expected: TechnicalClaimIdV1,
        observed: TechnicalClaimIdV1,
    },
    ProcedureReplacementMismatch {
        old: String,
        expected: String,
        observed: String,
    },
    PredecessorManifestMismatch,
    SuccessorManifestMismatch,
    BindingMismatch(String),
    ClaimReceiptConflict(TechnicalClaimIdV1),
    ProcedureReceiptConflict(String),
    MissingPredecessorManifest,
    UnexpectedInitialReviews,
    IncompleteReview {
        missing_claims: BTreeSet<TechnicalClaimIdV1>,
        missing_procedures: BTreeSet<String>,
        extraneous_claims: BTreeSet<TechnicalClaimIdV1>,
        extraneous_procedures: BTreeSet<String>,
    },
    NonEquivalentReplacement {
        claims: BTreeSet<TechnicalClaimIdV1>,
        procedures: BTreeSet<String>,
    },
}

impl fmt::Display for LegacySemanticReplacementErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(err) => write!(f, "legacy semantic replacement manifest error: {err}"),
            Self::Qualification(err) => write!(f, "legacy semantic replacement qualification error: {err}"),
            Self::UnsupportedSchema(value) => write!(f, "unsupported legacy semantic replacement schema {value}"),
            Self::InvalidField(value) => write!(f, "invalid legacy semantic replacement field: {value}"),
            Self::Serialization(value) => write!(f, "legacy semantic replacement serialization failed: {value}"),
            Self::UnknownClaim(id) => write!(f, "legacy semantic replacement references unknown claim {}", id.0),
            Self::UnknownProcedure(id) => write!(f, "legacy semantic replacement references unknown procedure {id}"),
            Self::UnexpectedClaimReceipt(id) => write!(f, "semantic receipt supplied for non-replaced claim {}", id.0),
            Self::UnexpectedProcedureReceipt(id) => write!(f, "semantic receipt supplied for non-replaced procedure {id}"),
            Self::ClaimReplacementMismatch { old, expected, observed } => write!(f, "claim semantic receipt {} expected successor {}, observed {}", old.0, expected.0, observed.0),
            Self::ProcedureReplacementMismatch { old, expected, observed } => write!(f, "procedure semantic receipt {old} expected successor {expected}, observed {observed}"),
            Self::PredecessorManifestMismatch => write!(f, "semantic receipt predecessor manifest digest mismatched"),
            Self::SuccessorManifestMismatch => write!(f, "semantic receipt successor manifest digest mismatched"),
            Self::BindingMismatch(id) => write!(f, "semantic replacement binding mismatched for {id}"),
            Self::ClaimReceiptConflict(id) => write!(f, "semantic claim receipt for {} was rebound", id.0),
            Self::ProcedureReceiptConflict(id) => write!(f, "semantic procedure receipt for {id} was rebound"),
            Self::MissingPredecessorManifest => write!(f, "semantic replacement review requires predecessor manifest"),
            Self::UnexpectedInitialReviews => write!(f, "generation 1 cannot contain semantic replacement reviews"),
            Self::IncompleteReview { missing_claims, missing_procedures, extraneous_claims, extraneous_procedures } => write!(f, "semantic replacement review is incomplete: {} missing claim, {} missing procedure, {} extraneous claim, {} extraneous procedure receipts", missing_claims.len(), missing_procedures.len(), extraneous_claims.len(), extraneous_procedures.len()),
            Self::NonEquivalentReplacement { claims, procedures } => write!(f, "semantic replacement review found {} non-equivalent claims and {} non-equivalent procedures", claims.len(), procedures.len()),
        }
    }
}

impl Error for LegacySemanticReplacementErrorV1 {}

impl From<LegacyQualificationManifestErrorV1> for LegacySemanticReplacementErrorV1 {
    fn from(value: LegacyQualificationManifestErrorV1) -> Self {
        Self::Manifest(value)
    }
}

impl From<LegacyQualificationProfileErrorV3> for LegacySemanticReplacementErrorV1 {
    fn from(value: LegacyQualificationProfileErrorV3) -> Self {
        Self::Qualification(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legacy_qualification_profile_v3::{
        build_legacy_qualification_successor_manifest_v1,
        initial_legacy_qualification_manifest_v1,
    };
    use crate::standards_registry::TechnicalClaimIdV1;
    use crate::{
        build_legacy_five_platform_portfolio_v1,
        exhaustive_legacy_qualification_profile_v1,
    };

    fn claim_rebaseline(
        judgment: LegacySemanticReplacementJudgmentV1,
    ) -> (
        LegacyComputingPackV1,
        LegacyQualificationManifestV1,
        LegacyQualificationManifestV1,
        LegacyClaimReplacementReviewReceiptV1,
    ) {
        let (mut pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let predecessor = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        let old = pack.sources.claims().next().unwrap().clone();
        let new_id = TechnicalClaimIdV1(format!("{}:rebaseline", old.id.0));
        let mut new_claim = old.clone();
        new_claim.id = new_id.clone();
        new_claim.statement = format!("{} [re-reviewed baseline]", old.statement);
        pack.sources.register_claim(new_claim.clone()).unwrap();
        let successor = build_legacy_qualification_successor_manifest_v1(
            &pack,
            &predecessor,
            BTreeMap::from([(old.id.clone(), new_id.clone())]),
            BTreeMap::new(),
        )
        .unwrap();
        let pred_digest = legacy_qualification_manifest_commitment_v1(&predecessor).unwrap();
        let succ_digest = legacy_qualification_manifest_commitment_v1(&successor).unwrap();
        let basis = "a".repeat(64);
        let reviewed_at = 1_800_000_000_100;
        let binding = legacy_claim_replacement_review_binding_v1(
            &old,
            &new_claim,
            &pred_digest,
            &succ_digest,
            judgment,
            LegacySemanticReviewMethodV1::HumanAndDeterministic,
            "legacy-semantic-review-v1",
            &basis,
            reviewed_at,
        )
        .unwrap();
        let receipt = LegacyClaimReplacementReviewReceiptV1 {
            predecessor_claim_id: old.id,
            successor_claim_id: new_id,
            predecessor_manifest_blake3: pred_digest,
            successor_manifest_blake3: succ_digest,
            judgment,
            review_method: LegacySemanticReviewMethodV1::HumanAndDeterministic,
            reviewer_profile: "legacy-semantic-review-v1".into(),
            review_basis_blake3: basis,
            reviewed_at_unix_ms: reviewed_at,
            replacement_binding_blake3: binding,
        };
        (pack, predecessor, successor, receipt)
    }

    #[test]
    fn equivalent_claim_receipt_allows_semantic_carry_forward() {
        let (pack, predecessor, successor, receipt) =
            claim_rebaseline(LegacySemanticReplacementJudgmentV1::Equivalent);
        let mut reviews = LegacySemanticReplacementReviewLedgerV1::new();
        assert!(reviews
            .register_claim_receipt(&pack, &predecessor, &successor, receipt)
            .unwrap());
        let assessment = require_legacy_semantic_replacement_equivalence_v1(
            &pack,
            &predecessor,
            &successor,
            &reviews,
        )
        .unwrap();
        assert!(assessment.complete);
        assert!(assessment.semantically_equivalent);
    }

    #[test]
    fn changed_meaning_is_recordable_but_cannot_inherit_qualification_role() {
        let (pack, predecessor, successor, receipt) =
            claim_rebaseline(LegacySemanticReplacementJudgmentV1::ChangedMeaning);
        let old_id = receipt.predecessor_claim_id.clone();
        let mut reviews = LegacySemanticReplacementReviewLedgerV1::new();
        reviews
            .register_claim_receipt(&pack, &predecessor, &successor, receipt)
            .unwrap();
        let assessment = reviews
            .validate_complete_equivalence(&pack, &predecessor, &successor)
            .unwrap();
        assert!(assessment.complete);
        assert!(!assessment.semantically_equivalent);
        assert!(assessment.non_equivalent_claims.contains(&old_id));
        assert!(matches!(
            require_legacy_semantic_replacement_equivalence_v1(
                &pack,
                &predecessor,
                &successor,
                &reviews,
            ),
            Err(LegacySemanticReplacementErrorV1::NonEquivalentReplacement { .. })
        ));
    }

    #[test]
    fn missing_receipt_fails_closed() {
        let (pack, predecessor, successor, _receipt) =
            claim_rebaseline(LegacySemanticReplacementJudgmentV1::Equivalent);
        let reviews = LegacySemanticReplacementReviewLedgerV1::new();
        assert!(matches!(
            require_legacy_semantic_replacement_equivalence_v1(
                &pack,
                &predecessor,
                &successor,
                &reviews,
            ),
            Err(LegacySemanticReplacementErrorV1::IncompleteReview { .. })
        ));
    }

    #[test]
    fn strict_generation_assessment_checks_semantics_before_source_readiness() {
        let (pack, predecessor, successor, receipt) =
            claim_rebaseline(LegacySemanticReplacementJudgmentV1::Equivalent);
        let mut reviews = LegacySemanticReplacementReviewLedgerV1::new();
        reviews
            .register_claim_receipt(&pack, &predecessor, &successor, receipt)
            .unwrap();
        let profile = exhaustive_legacy_qualification_profile_v1();
        let (_, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let source_ledger = LegacyQualificationSourceLedgerV3::new();
        let assessment = assess_legacy_qualification_generation_with_semantic_review_v1(
            &pack,
            &profile,
            &matrix,
            &artifacts,
            &source_ledger,
            Some(&predecessor),
            &successor,
            &reviews,
        )
        .unwrap();
        assert_eq!(assessment.manifest_generation, 2);
        assert_eq!(assessment.ready_requirements, 0);
        assert!(!assessment.source_readiness.source_evidence_ready);
    }
}
