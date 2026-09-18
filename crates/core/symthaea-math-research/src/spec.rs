// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bearing mathematical research specification boundary.
//!
//! Two identities are deliberately separated:
//!
//! - `challenge_sha256`: the mathematical subject itself — source revision,
//!   statement, definitions, and formal environment declaration;
//! - `qualification_sha256`: the exact review evidence/lineages that qualified
//!   that subject for proof/disproof work.
//!
//! The same theorem should retain the same subject identity when independently
//! re-reviewed, while authority-bearing downstream layers can still require the
//! exact qualification lineage they consumed.

use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

pub const SPECIFICATION_VERSION: &str = "symthaea.math-specification.v3";
const CHALLENGE_DOMAIN: &str = "symthaea.math-frozen-challenge.v3";
const QUALIFICATION_DOMAIN: &str = "symthaea.math-specification-qualification.v1";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    pub fn parse(value: impl Into<String>) -> Result<Self, DigestError> {
        let value = value.into();
        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(DigestError);
        }
        Ok(Self(value.to_ascii_lowercase()))
    }

    pub fn of_bytes(bytes: &[u8]) -> Self {
        let digest = Sha256::digest(bytes);
        let mut text = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write as _;
            write!(&mut text, "{byte:02x}").expect("writing to String cannot fail");
        }
        Self(text)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(value).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DigestError;

impl fmt::Display for DigestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("expected a 64-character hexadecimal SHA-256 digest")
    }
}

impl std::error::Error for DigestError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceReference {
    pub locator: String,
    pub revision: String,
    /// Audit metadata; excluded from mathematical subject identity but included
    /// in qualification identity.
    pub retrieved_at_utc: String,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum FormalizationReviewKind {
    DefinitionsAudit,
    SemanticReview,
    IndependentFormalization,
    EquivalenceCheck,
}

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct FormalizationReview {
    /// Attribution only; reviewer labels do not prove independence.
    pub reviewer_id: String,
    /// Commitment to correlated model/context/tool/organization lineage.
    pub independence_domain_sha256: Sha256Digest,
    /// Commitment to the concrete review procedure/prompt/method.
    pub method_sha256: Sha256Digest,
    pub kind: FormalizationReviewKind,
    pub lean_statement_sha256: Sha256Digest,
    pub definitions_sha256: Sha256Digest,
    pub notes: String,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum SpecificationStatus {
    Imported,
    SyntaxValidated,
    DefinitionsAudited,
    SemanticallyReviewed,
    IndependentlyReviewed,
    FrozenChallenge,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MathematicalSpecification {
    pub schema_version: String,
    pub specification_id: String,
    pub source: SourceReference,
    pub source_statement_sha256: Sha256Digest,
    pub lean_statement_sha256: Sha256Digest,
    pub definitions_sha256: Sha256Digest,
    pub lean_toolchain: String,
    pub mathlib_revision: String,
    pub reviews: Vec<FormalizationReview>,
    pub status: SpecificationStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecificationIssue {
    WrongSchemaVersion { found: String },
    EmptyField { field: &'static str },
    EmptyReviewIdentity { index: usize },
    EmptyReviewNotes { reviewer_id: String },
    ReviewStatementMismatch { reviewer_id: String },
    ReviewDefinitionsMismatch { reviewer_id: String },
    DuplicateReview {
        reviewer_id: String,
        kind: FormalizationReviewKind,
    },
    MissingDefinitionsAudit,
    MissingSemanticReview,
    MissingIndependentConfirmation,
    IndependentConfirmationNotCrossLineage,
    TooFewIndependentLineages { found: usize, required: usize },
    StatusRegression {
        from: SpecificationStatus,
        to: SpecificationStatus,
    },
    NotFrozenChallenge { found: SpecificationStatus },
}

impl MathematicalSpecification {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        specification_id: impl Into<String>,
        source: SourceReference,
        source_statement_sha256: Sha256Digest,
        lean_statement_sha256: Sha256Digest,
        definitions_sha256: Sha256Digest,
        lean_toolchain: impl Into<String>,
        mathlib_revision: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: SPECIFICATION_VERSION.into(),
            specification_id: specification_id.into(),
            source,
            source_statement_sha256,
            lean_statement_sha256,
            definitions_sha256,
            lean_toolchain: lean_toolchain.into(),
            mathlib_revision: mathlib_revision.into(),
            reviews: Vec::new(),
            status: SpecificationStatus::Imported,
        }
    }

    pub fn add_review(&mut self, review: FormalizationReview) {
        self.reviews.push(review);
    }

    /// Advance specification qualification monotonically. The proposed state is
    /// validated before it becomes durable.
    pub fn advance_to(
        &mut self,
        next: SpecificationStatus,
    ) -> Result<(), Vec<SpecificationIssue>> {
        if next < self.status {
            return Err(vec![SpecificationIssue::StatusRegression {
                from: self.status,
                to: next,
            }]);
        }
        let previous = self.status;
        self.status = next;
        let issues = self.validate();
        if issues.is_empty() {
            Ok(())
        } else {
            self.status = previous;
            Err(issues)
        }
    }

    pub fn validate(&self) -> Vec<SpecificationIssue> {
        let mut issues = Vec::new();
        if self.schema_version != SPECIFICATION_VERSION {
            issues.push(SpecificationIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        for (field, value) in [
            ("specification_id", self.specification_id.as_str()),
            ("source.locator", self.source.locator.as_str()),
            ("source.revision", self.source.revision.as_str()),
            ("source.retrieved_at_utc", self.source.retrieved_at_utc.as_str()),
            ("lean_toolchain", self.lean_toolchain.as_str()),
            ("mathlib_revision", self.mathlib_revision.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(SpecificationIssue::EmptyField { field });
            }
        }

        let mut unique_reviews = BTreeSet::new();
        for (index, review) in self.reviews.iter().enumerate() {
            if review.reviewer_id.trim().is_empty() {
                issues.push(SpecificationIssue::EmptyReviewIdentity { index });
            }
            if review.notes.trim().is_empty() {
                issues.push(SpecificationIssue::EmptyReviewNotes {
                    reviewer_id: review.reviewer_id.clone(),
                });
            }
            if review.lean_statement_sha256 != self.lean_statement_sha256 {
                issues.push(SpecificationIssue::ReviewStatementMismatch {
                    reviewer_id: review.reviewer_id.clone(),
                });
            }
            if review.definitions_sha256 != self.definitions_sha256 {
                issues.push(SpecificationIssue::ReviewDefinitionsMismatch {
                    reviewer_id: review.reviewer_id.clone(),
                });
            }
            if !unique_reviews.insert(review.clone()) {
                issues.push(SpecificationIssue::DuplicateReview {
                    reviewer_id: review.reviewer_id.clone(),
                    kind: review.kind,
                });
            }
        }

        if self.status >= SpecificationStatus::DefinitionsAudited
            && !self
                .reviews
                .iter()
                .any(|review| review.kind == FormalizationReviewKind::DefinitionsAudit)
        {
            issues.push(SpecificationIssue::MissingDefinitionsAudit);
        }
        if self.status >= SpecificationStatus::SemanticallyReviewed
            && !self
                .reviews
                .iter()
                .any(|review| review.kind == FormalizationReviewKind::SemanticReview)
        {
            issues.push(SpecificationIssue::MissingSemanticReview);
        }

        if self.status >= SpecificationStatus::IndependentlyReviewed {
            let semantic_domains: BTreeSet<_> = self
                .reviews
                .iter()
                .filter(|review| review.kind == FormalizationReviewKind::SemanticReview)
                .map(|review| &review.independence_domain_sha256)
                .collect();
            let confirmations: Vec<_> = self
                .reviews
                .iter()
                .filter(|review| {
                    matches!(
                        review.kind,
                        FormalizationReviewKind::IndependentFormalization
                            | FormalizationReviewKind::EquivalenceCheck
                    )
                })
                .collect();
            if confirmations.is_empty() {
                issues.push(SpecificationIssue::MissingIndependentConfirmation);
            }

            let relevant_domains: BTreeSet<_> = self
                .reviews
                .iter()
                .filter(|review| {
                    matches!(
                        review.kind,
                        FormalizationReviewKind::SemanticReview
                            | FormalizationReviewKind::IndependentFormalization
                            | FormalizationReviewKind::EquivalenceCheck
                    )
                })
                .map(|review| &review.independence_domain_sha256)
                .collect();
            if relevant_domains.len() < 2 {
                issues.push(SpecificationIssue::TooFewIndependentLineages {
                    found: relevant_domains.len(),
                    required: 2,
                });
            }

            if !confirmations.is_empty()
                && !confirmations.iter().any(|confirmation| {
                    semantic_domains
                        .iter()
                        .any(|semantic| *semantic != &confirmation.independence_domain_sha256)
                })
            {
                issues.push(SpecificationIssue::IndependentConfirmationNotCrossLineage);
            }
        }
        issues
    }

    /// Freeze both the mathematical subject and the review evidence that
    /// qualified it. Proof outputs are not inputs to either identity.
    pub fn freeze(self) -> Result<FrozenChallenge, Vec<SpecificationIssue>> {
        let issues = self.validate();
        if !issues.is_empty() {
            return Err(issues);
        }
        if self.status != SpecificationStatus::FrozenChallenge {
            return Err(vec![SpecificationIssue::NotFrozenChallenge {
                found: self.status,
            }]);
        }

        let challenge_sha256 = self.compute_challenge_digest();
        let qualification_sha256 = self.compute_qualification_digest(&challenge_sha256);
        Ok(FrozenChallenge {
            specification: self,
            challenge_sha256,
            qualification_sha256,
        })
    }

    fn compute_challenge_digest(&self) -> Sha256Digest {
        let mut bytes = Vec::new();
        put_text(&mut bytes, CHALLENGE_DOMAIN);
        put_text(&mut bytes, &self.specification_id);
        put_text(&mut bytes, &self.source.locator);
        put_text(&mut bytes, &self.source.revision);
        put_text(&mut bytes, self.source_statement_sha256.as_str());
        put_text(&mut bytes, self.lean_statement_sha256.as_str());
        put_text(&mut bytes, self.definitions_sha256.as_str());
        put_text(&mut bytes, &self.lean_toolchain);
        put_text(&mut bytes, &self.mathlib_revision);
        Sha256Digest::of_bytes(&bytes)
    }

    fn compute_qualification_digest(&self, challenge_sha256: &Sha256Digest) -> Sha256Digest {
        let mut bytes = Vec::new();
        put_text(&mut bytes, QUALIFICATION_DOMAIN);
        put_text(&mut bytes, &self.schema_version);
        put_text(&mut bytes, challenge_sha256.as_str());
        put_text(&mut bytes, &self.source.retrieved_at_utc);
        put_text(&mut bytes, specification_status_tag(self.status));

        let mut reviews = self.reviews.clone();
        reviews.sort();
        for review in reviews {
            put_text(&mut bytes, review_kind_tag(review.kind));
            put_text(&mut bytes, &review.reviewer_id);
            put_text(&mut bytes, review.independence_domain_sha256.as_str());
            put_text(&mut bytes, review.method_sha256.as_str());
            put_text(&mut bytes, review.lean_statement_sha256.as_str());
            put_text(&mut bytes, review.definitions_sha256.as_str());
            put_text(&mut bytes, &review.notes);
        }
        Sha256Digest::of_bytes(&bytes)
    }
}

/// Immutable reviewed subject presented to proof/disproof systems.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenChallenge {
    specification: MathematicalSpecification,
    /// Stable identity of the mathematical subject.
    challenge_sha256: Sha256Digest,
    /// Identity of the exact evidence/review lineage that qualified this subject.
    qualification_sha256: Sha256Digest,
}

impl FrozenChallenge {
    pub fn specification(&self) -> &MathematicalSpecification {
        &self.specification
    }
    pub fn challenge_sha256(&self) -> &Sha256Digest {
        &self.challenge_sha256
    }
    pub fn qualification_sha256(&self) -> &Sha256Digest {
        &self.qualification_sha256
    }
}

fn review_kind_tag(kind: FormalizationReviewKind) -> &'static str {
    match kind {
        FormalizationReviewKind::DefinitionsAudit => "definitions-audit",
        FormalizationReviewKind::SemanticReview => "semantic-review",
        FormalizationReviewKind::IndependentFormalization => "independent-formalization",
        FormalizationReviewKind::EquivalenceCheck => "equivalence-check",
    }
}

fn specification_status_tag(status: SpecificationStatus) -> &'static str {
    match status {
        SpecificationStatus::Imported => "imported",
        SpecificationStatus::SyntaxValidated => "syntax-validated",
        SpecificationStatus::DefinitionsAudited => "definitions-audited",
        SpecificationStatus::SemanticallyReviewed => "semantically-reviewed",
        SpecificationStatus::IndependentlyReviewed => "independently-reviewed",
        SpecificationStatus::FrozenChallenge => "frozen-challenge",
    }
}

fn put_text(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_be_bytes());
    output.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(label.as_bytes())
    }

    fn source() -> SourceReference {
        SourceReference {
            locator: "https://example.invalid/problem".into(),
            revision: "problem-v1".into(),
            retrieved_at_utc: "2026-09-18T00:00:00Z".into(),
        }
    }

    fn spec() -> MathematicalSpecification {
        MathematicalSpecification::new(
            "TEST-001",
            source(),
            digest("informal"),
            digest("lean"),
            digest("definitions"),
            "leanprover/lean4:v4.24.0",
            "mathlib-test-revision",
        )
    }

    fn add_review(
        spec: &mut MathematicalSpecification,
        reviewer: &str,
        domain: &str,
        kind: FormalizationReviewKind,
    ) {
        spec.add_review(FormalizationReview {
            reviewer_id: reviewer.into(),
            independence_domain_sha256: digest(domain),
            method_sha256: digest(&format!("method:{reviewer}:{domain}:{kind:?}")),
            kind,
            lean_statement_sha256: spec.lean_statement_sha256.clone(),
            definitions_sha256: spec.definitions_sha256.clone(),
            notes: "reviewed against the pinned mathematical source".into(),
        });
    }

    fn add_full_review_set(spec: &mut MathematicalSpecification) {
        add_review(spec, "reviewer-a", "lineage-a", FormalizationReviewKind::DefinitionsAudit);
        add_review(spec, "reviewer-a", "lineage-a", FormalizationReviewKind::SemanticReview);
        add_review(
            spec,
            "reviewer-b",
            "lineage-b",
            FormalizationReviewKind::IndependentFormalization,
        );
    }

    fn freeze(mut spec: MathematicalSpecification) -> FrozenChallenge {
        spec.advance_to(SpecificationStatus::FrozenChallenge).unwrap();
        spec.freeze().unwrap()
    }

    #[test]
    fn digest_parser_rejects_malformed_or_serde_bypass() {
        assert!(Sha256Digest::parse("abc").is_err());
        assert!(Sha256Digest::parse("z".repeat(64)).is_err());
        assert!(Sha256Digest::parse("AA".repeat(32)).is_ok());
        assert!(serde_json::from_str::<Sha256Digest>("\"abc\"").is_err());
    }

    #[test]
    fn same_lineage_labels_are_not_independent() {
        let mut spec = spec();
        add_review(&mut spec, "a", "shared", FormalizationReviewKind::DefinitionsAudit);
        add_review(&mut spec, "a", "shared", FormalizationReviewKind::SemanticReview);
        add_review(
            &mut spec,
            "b",
            "shared",
            FormalizationReviewKind::IndependentFormalization,
        );
        let issues = spec
            .advance_to(SpecificationStatus::IndependentlyReviewed)
            .unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            SpecificationIssue::TooFewIndependentLineages { found: 1, required: 2 }
        )));
    }

    #[test]
    fn independent_confirmation_must_cross_semantic_lineage() {
        let mut spec = spec();
        add_review(&mut spec, "defs", "lineage-b", FormalizationReviewKind::DefinitionsAudit);
        add_review(&mut spec, "semantic", "lineage-a", FormalizationReviewKind::SemanticReview);
        add_review(
            &mut spec,
            "independent",
            "lineage-a",
            FormalizationReviewKind::IndependentFormalization,
        );
        let issues = spec
            .advance_to(SpecificationStatus::IndependentlyReviewed)
            .unwrap_err();
        assert!(issues.contains(&SpecificationIssue::IndependentConfirmationNotCrossLineage));
    }

    #[test]
    fn duplicate_review_does_not_create_extra_evidence() {
        let mut spec = spec();
        let review = FormalizationReview {
            reviewer_id: "a".into(),
            independence_domain_sha256: digest("lineage-a"),
            method_sha256: digest("method-a"),
            kind: FormalizationReviewKind::DefinitionsAudit,
            lean_statement_sha256: spec.lean_statement_sha256.clone(),
            definitions_sha256: spec.definitions_sha256.clone(),
            notes: "same review".into(),
        };
        spec.add_review(review.clone());
        spec.add_review(review);
        assert!(spec.validate().iter().any(|issue| matches!(
            issue,
            SpecificationIssue::DuplicateReview { .. }
        )));
    }

    #[test]
    fn subject_identity_is_stable_but_qualification_tracks_review_lineage() {
        let mut left = spec();
        add_full_review_set(&mut left);
        let left = freeze(left);

        let mut reordered = spec();
        add_review(
            &mut reordered,
            "reviewer-b",
            "lineage-b",
            FormalizationReviewKind::IndependentFormalization,
        );
        add_review(
            &mut reordered,
            "reviewer-a",
            "lineage-a",
            FormalizationReviewKind::SemanticReview,
        );
        add_review(
            &mut reordered,
            "reviewer-a",
            "lineage-a",
            FormalizationReviewKind::DefinitionsAudit,
        );
        let reordered = freeze(reordered);
        assert_eq!(left.challenge_sha256(), reordered.challenge_sha256());
        assert_eq!(left.qualification_sha256(), reordered.qualification_sha256());

        let mut independently_requalified = spec();
        add_review(
            &mut independently_requalified,
            "reviewer-a",
            "lineage-a",
            FormalizationReviewKind::DefinitionsAudit,
        );
        add_review(
            &mut independently_requalified,
            "reviewer-a",
            "lineage-a",
            FormalizationReviewKind::SemanticReview,
        );
        add_review(
            &mut independently_requalified,
            "reviewer-c",
            "lineage-c",
            FormalizationReviewKind::EquivalenceCheck,
        );
        let independently_requalified = freeze(independently_requalified);
        assert_eq!(left.challenge_sha256(), independently_requalified.challenge_sha256());
        assert_ne!(left.qualification_sha256(), independently_requalified.qualification_sha256());
    }

    #[test]
    fn proof_search_cannot_freeze_unreviewed_subject() {
        let issues = spec().freeze().unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            SpecificationIssue::NotFrozenChallenge { found: SpecificationStatus::Imported }
        )));
    }
}
