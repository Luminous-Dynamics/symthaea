// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Immutable mathematical claims and structural evidence receipts.
//!
//! A claim is sealed before evidence exists. Every receipt binds the exact claim
//! digest and frozen-challenge digest. Receipts record what an evidence producer
//! *declares* happened; sealing proves structural integrity and subject binding,
//! not that an executable verifier actually ran.
//!
//! That distinction is intentional and security-critical. Formal-looking receipt
//! kinds (`LeanKernel`, `LeanComparator`, `IndependentChecker`) remain useful as a
//! common evidence schema, but generic [`EvidenceReceipt`] values do **not** confer
//! mathematical authority. Verifier-specific crates must authenticate execution
//! and wrap/attest these receipts before any authority transition is permitted.
//!
//! Independence metadata is still lineage-aware so structural evidence can be
//! audited for correlated failure domains without confusing that audit with
//! executable provenance.

use crate::spec::{FrozenChallenge, Sha256Digest};
use serde::{Deserialize, Serialize};

pub const CLAIM_VERSION: &str = "symthaea.math-claim.v2";
pub const EVIDENCE_RECEIPT_VERSION: &str = "symthaea.math-evidence-receipt.v2";
const CLAIM_DOMAIN: &str = "symthaea.math-claim-digest.v2";
const RECEIPT_DOMAIN: &str = "symthaea.math-evidence-receipt-digest.v2";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum MathematicalClaimKind {
    Conjecture,
    Lemma,
    Theorem,
    Equivalence,
    Counterexample,
    Bound,
    SpecialCase,
    TechniqueCandidate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimDraft {
    pub schema_version: String,
    pub claim_id: String,
    pub challenge_sha256: Sha256Digest,
    pub statement_sha256: Sha256Digest,
    pub assumptions_sha256: Sha256Digest,
    pub kind: MathematicalClaimKind,
    pub created_by: String,
    /// Commitment to the model/prompt/tool/context boundary that generated the claim.
    pub context_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimIssue {
    WrongSchemaVersion { found: String },
    EmptyField { field: &'static str },
    ChallengeMismatch,
}

impl ClaimDraft {
    pub fn for_challenge(
        challenge: &FrozenChallenge,
        claim_id: impl Into<String>,
        statement_sha256: Sha256Digest,
        assumptions_sha256: Sha256Digest,
        kind: MathematicalClaimKind,
        created_by: impl Into<String>,
        context_sha256: Sha256Digest,
    ) -> Self {
        Self {
            schema_version: CLAIM_VERSION.into(),
            claim_id: claim_id.into(),
            challenge_sha256: challenge.challenge_sha256().clone(),
            statement_sha256,
            assumptions_sha256,
            kind,
            created_by: created_by.into(),
            context_sha256,
        }
    }

    pub fn seal(self, challenge: &FrozenChallenge) -> Result<SealedClaim, Vec<ClaimIssue>> {
        let mut issues = Vec::new();
        if self.schema_version != CLAIM_VERSION {
            issues.push(ClaimIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        for (field, value) in [
            ("claim_id", self.claim_id.as_str()),
            ("created_by", self.created_by.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(ClaimIssue::EmptyField { field });
            }
        }
        if self.challenge_sha256 != *challenge.challenge_sha256() {
            issues.push(ClaimIssue::ChallengeMismatch);
        }
        if !issues.is_empty() {
            return Err(issues);
        }

        let claim_sha256 = compute_claim_digest(&self);
        Ok(SealedClaim {
            draft: self,
            claim_sha256,
        })
    }
}

/// Immutable mathematical claim identity, sealed before evidence is attached.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SealedClaim {
    draft: ClaimDraft,
    claim_sha256: Sha256Digest,
}

impl SealedClaim {
    pub fn claim_id(&self) -> &str {
        &self.draft.claim_id
    }

    pub fn challenge_sha256(&self) -> &Sha256Digest {
        &self.draft.challenge_sha256
    }

    pub fn statement_sha256(&self) -> &Sha256Digest {
        &self.draft.statement_sha256
    }

    pub fn assumptions_sha256(&self) -> &Sha256Digest {
        &self.draft.assumptions_sha256
    }

    pub fn kind(&self) -> MathematicalClaimKind {
        self.draft.kind
    }

    pub fn claim_sha256(&self) -> &Sha256Digest {
        &self.claim_sha256
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EvidenceKind {
    NumericalSample,
    ExhaustiveFinite,
    SymbolicIdentity,
    SmtSample,
    SmtUniversal,
    LeanKernel,
    LeanComparator,
    IndependentChecker,
    Counterexample,
    LiteratureSearch,
    HumanReview,
}

impl EvidenceKind {
    /// Whether this receipt *declares* an executable formal-verifier provenance.
    ///
    /// This is a schema property only. It does not authenticate that the verifier ran.
    pub fn is_formal_verifier(self) -> bool {
        matches!(
            self,
            Self::LeanKernel | Self::LeanComparator | Self::IndependentChecker
        )
    }

    fn tag(self) -> &'static str {
        match self {
            Self::NumericalSample => "numerical-sample",
            Self::ExhaustiveFinite => "exhaustive-finite",
            Self::SymbolicIdentity => "symbolic-identity",
            Self::SmtSample => "smt-sample",
            Self::SmtUniversal => "smt-universal",
            Self::LeanKernel => "lean-kernel",
            Self::LeanComparator => "lean-comparator",
            Self::IndependentChecker => "independent-checker",
            Self::Counterexample => "counterexample",
            Self::LiteratureSearch => "literature-search",
            Self::HumanReview => "human-review",
        }
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EvidenceVerdict {
    Supports,
    Refutes,
    Inconclusive,
    Invalid,
}

impl EvidenceVerdict {
    fn tag(self) -> &'static str {
        match self {
            Self::Supports => "supports",
            Self::Refutes => "refutes",
            Self::Inconclusive => "inconclusive",
            Self::Invalid => "invalid",
        }
    }
}

/// Mutable structural receipt draft.
///
/// All fields are public so evidence producers can populate a common schema. A
/// caller can therefore construct a formal-looking draft; callers must never use
/// successful sealing as proof that Lean/SMT/another checker actually executed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReceiptDraft {
    pub schema_version: String,
    pub receipt_id: String,
    pub claim_sha256: Sha256Digest,
    pub challenge_sha256: Sha256Digest,
    pub kind: EvidenceKind,
    pub verdict: EvidenceVerdict,
    pub artifact_sha256: Sha256Digest,
    pub producer_id: String,
    /// Commitment to a correlated implementation/runtime/organization/model domain.
    pub trust_domain_sha256: Sha256Digest,
    /// Declared executable identity for formal-verifier receipt kinds.
    pub producer_binary_sha256: Option<Sha256Digest>,
    pub environment_sha256: Sha256Digest,
    pub created_at_utc: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceIssue {
    WrongSchemaVersion { found: String },
    EmptyField { field: &'static str },
    ClaimMismatch,
    ChallengeMismatch,
    MissingVerifierBinary { kind: EvidenceKind },
}

impl EvidenceReceiptDraft {
    #[allow(clippy::too_many_arguments)]
    pub fn for_claim(
        claim: &SealedClaim,
        receipt_id: impl Into<String>,
        kind: EvidenceKind,
        verdict: EvidenceVerdict,
        artifact_sha256: Sha256Digest,
        producer_id: impl Into<String>,
        trust_domain_sha256: Sha256Digest,
        producer_binary_sha256: Option<Sha256Digest>,
        environment_sha256: Sha256Digest,
        created_at_utc: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: EVIDENCE_RECEIPT_VERSION.into(),
            receipt_id: receipt_id.into(),
            claim_sha256: claim.claim_sha256().clone(),
            challenge_sha256: claim.challenge_sha256().clone(),
            kind,
            verdict,
            artifact_sha256,
            producer_id: producer_id.into(),
            trust_domain_sha256,
            producer_binary_sha256,
            environment_sha256,
            created_at_utc: created_at_utc.into(),
        }
    }

    /// Seal subject identity and receipt integrity.
    ///
    /// This validates the receipt contract. It deliberately does not authenticate
    /// execution provenance; verifier-specific code must do that separately.
    pub fn seal_for_claim(
        self,
        claim: &SealedClaim,
    ) -> Result<EvidenceReceipt, Vec<EvidenceIssue>> {
        let mut issues = Vec::new();
        if self.schema_version != EVIDENCE_RECEIPT_VERSION {
            issues.push(EvidenceIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        for (field, value) in [
            ("receipt_id", self.receipt_id.as_str()),
            ("producer_id", self.producer_id.as_str()),
            ("created_at_utc", self.created_at_utc.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(EvidenceIssue::EmptyField { field });
            }
        }
        if self.claim_sha256 != *claim.claim_sha256() {
            issues.push(EvidenceIssue::ClaimMismatch);
        }
        if self.challenge_sha256 != *claim.challenge_sha256() {
            issues.push(EvidenceIssue::ChallengeMismatch);
        }
        if self.kind.is_formal_verifier() && self.producer_binary_sha256.is_none() {
            issues.push(EvidenceIssue::MissingVerifierBinary { kind: self.kind });
        }
        if !issues.is_empty() {
            return Err(issues);
        }

        let receipt_sha256 = compute_receipt_digest(&self);
        Ok(EvidenceReceipt {
            draft: self,
            receipt_sha256,
        })
    }
}

/// Immutable structural evidence event bound to one exact sealed claim/challenge.
///
/// Immutability protects the event after sealing; it is not executable attestation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidenceReceipt {
    draft: EvidenceReceiptDraft,
    receipt_sha256: Sha256Digest,
}

impl EvidenceReceipt {
    pub fn receipt_id(&self) -> &str {
        &self.draft.receipt_id
    }

    pub fn receipt_sha256(&self) -> &Sha256Digest {
        &self.receipt_sha256
    }

    pub fn claim_sha256(&self) -> &Sha256Digest {
        &self.draft.claim_sha256
    }

    pub fn challenge_sha256(&self) -> &Sha256Digest {
        &self.draft.challenge_sha256
    }

    pub fn kind(&self) -> EvidenceKind {
        self.draft.kind
    }

    pub fn verdict(&self) -> EvidenceVerdict {
        self.draft.verdict
    }

    pub fn artifact_sha256(&self) -> &Sha256Digest {
        &self.draft.artifact_sha256
    }

    pub fn producer_id(&self) -> &str {
        &self.draft.producer_id
    }

    pub fn trust_domain_sha256(&self) -> &Sha256Digest {
        &self.draft.trust_domain_sha256
    }

    pub fn producer_binary_sha256(&self) -> Option<&Sha256Digest> {
        self.draft.producer_binary_sha256.as_ref()
    }

    pub fn environment_sha256(&self) -> &Sha256Digest {
        &self.draft.environment_sha256
    }

    pub fn created_at_utc(&self) -> &str {
        &self.draft.created_at_utc
    }
}

/// Structural classification of formal-looking receipt declarations.
///
/// This type intentionally does **not** contain the word `Authority`: generic
/// receipts are caller-constructible and cannot authenticate that a verifier ran.
/// Verifier-specific authenticated wrappers must establish real authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeclaredFormalEvidence {
    None,
    KernelReceipt,
    ComparatorReceipt,
    IndependentReceiptPair,
}

impl DeclaredFormalEvidence {
    /// Classify supporting, subject-matched receipt declarations.
    ///
    /// `IndependentReceiptPair` additionally requires different trust-domain and
    /// binary commitments, but those commitments are still declarations here.
    pub fn from_receipts(claim: &SealedClaim, receipts: &[EvidenceReceipt]) -> Self {
        let matching: Vec<_> = receipts
            .iter()
            .filter(|receipt| {
                receipt.claim_sha256() == claim.claim_sha256()
                    && receipt.challenge_sha256() == claim.challenge_sha256()
                    && receipt.verdict() == EvidenceVerdict::Supports
            })
            .collect();

        let has_kernel = matching
            .iter()
            .any(|receipt| receipt.kind() == EvidenceKind::LeanKernel);
        let comparators: Vec<_> = matching
            .iter()
            .copied()
            .filter(|receipt| receipt.kind() == EvidenceKind::LeanComparator)
            .collect();
        let independent: Vec<_> = matching
            .iter()
            .copied()
            .filter(|receipt| receipt.kind() == EvidenceKind::IndependentChecker)
            .collect();

        let independent_pair_exists = comparators.iter().any(|comparator| {
            independent
                .iter()
                .any(|checker| declared_verifier_lineages_differ(comparator, checker))
        });

        if independent_pair_exists {
            Self::IndependentReceiptPair
        } else if !comparators.is_empty() {
            Self::ComparatorReceipt
        } else if has_kernel {
            Self::KernelReceipt
        } else {
            Self::None
        }
    }
}

fn declared_verifier_lineages_differ(left: &EvidenceReceipt, right: &EvidenceReceipt) -> bool {
    if left.trust_domain_sha256() == right.trust_domain_sha256() {
        return false;
    }
    match (
        left.producer_binary_sha256(),
        right.producer_binary_sha256(),
    ) {
        (Some(left_binary), Some(right_binary)) => left_binary != right_binary,
        _ => false,
    }
}

fn compute_claim_digest(claim: &ClaimDraft) -> Sha256Digest {
    let mut bytes = Vec::new();
    put_text(&mut bytes, CLAIM_DOMAIN);
    put_text(&mut bytes, &claim.schema_version);
    put_text(&mut bytes, &claim.claim_id);
    put_text(&mut bytes, claim.challenge_sha256.as_str());
    put_text(&mut bytes, claim.statement_sha256.as_str());
    put_text(&mut bytes, claim.assumptions_sha256.as_str());
    put_text(&mut bytes, claim_kind_tag(claim.kind));
    put_text(&mut bytes, &claim.created_by);
    put_text(&mut bytes, claim.context_sha256.as_str());
    Sha256Digest::of_bytes(&bytes)
}

fn compute_receipt_digest(receipt: &EvidenceReceiptDraft) -> Sha256Digest {
    let mut bytes = Vec::new();
    put_text(&mut bytes, RECEIPT_DOMAIN);
    put_text(&mut bytes, &receipt.schema_version);
    put_text(&mut bytes, &receipt.receipt_id);
    put_text(&mut bytes, receipt.claim_sha256.as_str());
    put_text(&mut bytes, receipt.challenge_sha256.as_str());
    put_text(&mut bytes, receipt.kind.tag());
    put_text(&mut bytes, receipt.verdict.tag());
    put_text(&mut bytes, receipt.artifact_sha256.as_str());
    put_text(&mut bytes, &receipt.producer_id);
    put_text(&mut bytes, receipt.trust_domain_sha256.as_str());
    match &receipt.producer_binary_sha256 {
        Some(digest) => {
            put_text(&mut bytes, "binary-present");
            put_text(&mut bytes, digest.as_str());
        }
        None => put_text(&mut bytes, "binary-absent"),
    }
    put_text(&mut bytes, receipt.environment_sha256.as_str());
    put_text(&mut bytes, &receipt.created_at_utc);
    Sha256Digest::of_bytes(&bytes)
}

fn claim_kind_tag(kind: MathematicalClaimKind) -> &'static str {
    match kind {
        MathematicalClaimKind::Conjecture => "conjecture",
        MathematicalClaimKind::Lemma => "lemma",
        MathematicalClaimKind::Theorem => "theorem",
        MathematicalClaimKind::Equivalence => "equivalence",
        MathematicalClaimKind::Counterexample => "counterexample",
        MathematicalClaimKind::Bound => "bound",
        MathematicalClaimKind::SpecialCase => "special-case",
        MathematicalClaimKind::TechniqueCandidate => "technique-candidate",
    }
}

fn put_text(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_be_bytes());
    output.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{
        FormalizationReview, FormalizationReviewKind, MathematicalSpecification,
        SourceReference, SpecificationStatus,
    };

    fn digest(label: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(label.as_bytes())
    }

    fn frozen_challenge() -> FrozenChallenge {
        let mut spec = MathematicalSpecification::new(
            "CLAIM-TEST",
            SourceReference {
                locator: "https://example.invalid/problem".into(),
                revision: "v1".into(),
                retrieved_at_utc: "2026-09-18T00:00:00Z".into(),
            },
            digest("source"),
            digest("lean"),
            digest("defs"),
            "lean-test",
            "mathlib-test",
        );
        let lean = spec.lean_statement_sha256.clone();
        let defs = spec.definitions_sha256.clone();
        for (reviewer, domain, kind) in [
            (
                "reviewer-a",
                "review-lineage-a",
                FormalizationReviewKind::DefinitionsAudit,
            ),
            (
                "reviewer-a",
                "review-lineage-a",
                FormalizationReviewKind::SemanticReview,
            ),
            (
                "reviewer-b",
                "review-lineage-b",
                FormalizationReviewKind::IndependentFormalization,
            ),
        ] {
            spec.add_review(FormalizationReview {
                reviewer_id: reviewer.into(),
                independence_domain_sha256: digest(domain),
                method_sha256: digest(&format!("method:{reviewer}:{domain}:{kind:?}")),
                kind,
                lean_statement_sha256: lean.clone(),
                definitions_sha256: defs.clone(),
                notes: "bound review".into(),
            });
        }
        spec.advance_to(SpecificationStatus::FrozenChallenge).unwrap();
        spec.freeze().unwrap()
    }

    fn sealed_claim(challenge: &FrozenChallenge) -> SealedClaim {
        ClaimDraft::for_challenge(
            challenge,
            "lemma-1",
            digest("lemma statement"),
            digest("lemma assumptions"),
            MathematicalClaimKind::Lemma,
            "symthaea-test-agent",
            digest("context"),
        )
        .seal(challenge)
        .unwrap()
    }

    fn receipt(
        claim: &SealedClaim,
        id: &str,
        kind: EvidenceKind,
        verdict: EvidenceVerdict,
        trust_domain: &str,
        binary: &str,
    ) -> EvidenceReceipt {
        EvidenceReceiptDraft::for_claim(
            claim,
            id,
            kind,
            verdict,
            digest(id),
            "test-producer",
            digest(trust_domain),
            kind.is_formal_verifier().then(|| digest(binary)),
            digest("environment"),
            "2026-09-18T00:00:00Z",
        )
        .seal_for_claim(claim)
        .unwrap()
    }

    #[test]
    fn sealed_claim_is_bound_to_frozen_challenge() {
        let challenge = frozen_challenge();
        let claim = sealed_claim(&challenge);
        assert_eq!(claim.challenge_sha256(), challenge.challenge_sha256());
        assert_ne!(claim.claim_sha256(), claim.statement_sha256());
    }

    #[test]
    fn evidence_for_another_claim_is_rejected() {
        let challenge = frozen_challenge();
        let claim_a = sealed_claim(&challenge);
        let claim_b = ClaimDraft::for_challenge(
            &challenge,
            "lemma-2",
            digest("other statement"),
            digest("other assumptions"),
            MathematicalClaimKind::Lemma,
            "symthaea-test-agent",
            digest("context"),
        )
        .seal(&challenge)
        .unwrap();
        let draft = EvidenceReceiptDraft::for_claim(
            &claim_a,
            "receipt-a",
            EvidenceKind::LeanKernel,
            EvidenceVerdict::Supports,
            digest("proof"),
            "lean",
            digest("lean-domain"),
            Some(digest("lean-binary")),
            digest("environment"),
            "2026-09-18T00:00:00Z",
        );
        assert!(
            draft
                .seal_for_claim(&claim_b)
                .unwrap_err()
                .contains(&EvidenceIssue::ClaimMismatch)
        );
    }

    #[test]
    fn formal_declared_receipt_requires_binary_identity() {
        let challenge = frozen_challenge();
        let claim = sealed_claim(&challenge);
        let draft = EvidenceReceiptDraft::for_claim(
            &claim,
            "receipt-a",
            EvidenceKind::LeanComparator,
            EvidenceVerdict::Supports,
            digest("proof"),
            "lean-comparator",
            digest("domain"),
            None,
            digest("environment"),
            "2026-09-18T00:00:00Z",
        );
        assert!(
            draft
                .seal_for_claim(&claim)
                .unwrap_err()
                .contains(&EvidenceIssue::MissingVerifierBinary {
                    kind: EvidenceKind::LeanComparator
                })
        );
    }

    #[test]
    fn numerical_evidence_has_no_declared_formal_level() {
        let challenge = frozen_challenge();
        let claim = sealed_claim(&challenge);
        let numerical = receipt(
            &claim,
            "numeric",
            EvidenceKind::NumericalSample,
            EvidenceVerdict::Supports,
            "numerical-domain",
            "unused",
        );
        assert_eq!(
            DeclaredFormalEvidence::from_receipts(&claim, &[numerical]),
            DeclaredFormalEvidence::None
        );
    }

    #[test]
    fn caller_constructed_kernel_receipt_is_only_a_declaration() {
        let challenge = frozen_challenge();
        let claim = sealed_claim(&challenge);
        // No Lean process runs here. This is exactly why the generic classifier
        // must never be named or treated as mathematical authority.
        let forged_but_structurally_valid = receipt(
            &claim,
            "claimed-kernel",
            EvidenceKind::LeanKernel,
            EvidenceVerdict::Supports,
            "claimed-domain",
            "claimed-binary",
        );
        assert_eq!(
            DeclaredFormalEvidence::from_receipts(&claim, &[forged_but_structurally_valid]),
            DeclaredFormalEvidence::KernelReceipt
        );
    }

    #[test]
    fn distinct_declared_lineages_form_an_independent_receipt_pair() {
        let challenge = frozen_challenge();
        let claim = sealed_claim(&challenge);
        let comparator = receipt(
            &claim,
            "comparator",
            EvidenceKind::LeanComparator,
            EvidenceVerdict::Supports,
            "lean-domain",
            "lean-comparator-binary",
        );
        let checker = receipt(
            &claim,
            "checker",
            EvidenceKind::IndependentChecker,
            EvidenceVerdict::Supports,
            "nanoda-domain",
            "nanoda-binary",
        );
        assert_eq!(
            DeclaredFormalEvidence::from_receipts(&claim, &[comparator, checker]),
            DeclaredFormalEvidence::IndependentReceiptPair
        );
    }

    #[test]
    fn same_domain_or_same_binary_does_not_form_independent_pair() {
        let challenge = frozen_challenge();
        let claim = sealed_claim(&challenge);
        let comparator = receipt(
            &claim,
            "comparator",
            EvidenceKind::LeanComparator,
            EvidenceVerdict::Supports,
            "shared-domain",
            "binary-a",
        );
        let same_domain = receipt(
            &claim,
            "checker-a",
            EvidenceKind::IndependentChecker,
            EvidenceVerdict::Supports,
            "shared-domain",
            "binary-b",
        );
        assert_eq!(
            DeclaredFormalEvidence::from_receipts(&claim, &[comparator.clone(), same_domain]),
            DeclaredFormalEvidence::ComparatorReceipt
        );

        let same_binary = receipt(
            &claim,
            "checker-b",
            EvidenceKind::IndependentChecker,
            EvidenceVerdict::Supports,
            "other-domain",
            "binary-a",
        );
        assert_eq!(
            DeclaredFormalEvidence::from_receipts(&claim, &[comparator, same_binary]),
            DeclaredFormalEvidence::ComparatorReceipt
        );
    }

    #[test]
    fn refuting_or_inconclusive_receipts_do_not_raise_declared_level() {
        let challenge = frozen_challenge();
        let claim = sealed_claim(&challenge);
        let refuted = receipt(
            &claim,
            "refuted",
            EvidenceKind::LeanComparator,
            EvidenceVerdict::Refutes,
            "domain-a",
            "binary-a",
        );
        let unknown = receipt(
            &claim,
            "unknown",
            EvidenceKind::IndependentChecker,
            EvidenceVerdict::Inconclusive,
            "domain-b",
            "binary-b",
        );
        assert_eq!(
            DeclaredFormalEvidence::from_receipts(&claim, &[refuted, unknown]),
            DeclaredFormalEvidence::None
        );
    }
}
